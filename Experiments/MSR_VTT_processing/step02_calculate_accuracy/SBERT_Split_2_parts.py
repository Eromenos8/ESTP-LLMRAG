import torch
import pandas as pd
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import os
import numpy as np
from torch.cuda.amp import autocast
import faiss
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

@dataclass
class SimilarityResult:
    """存储相似度计算结果的数据类"""
    video_files: List[str]
    similarity_scores: List[float]
    generated_texts: List[str]
    query_time: float
    match_positions: List[int]

class TextDataset(Dataset):
    """文本数据集类，用于批处理"""
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class SplitSBERTAnalyzer:
    """支持数据集分割的SBERT深度分析器"""
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        """设置日志配置"""
        log_file = f'split_sbert_analysis_{self.timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def split_datasets(self, df: pd.DataFrame, video_text_map: Dict[str, str], test_size: float = 0.5, random_state: int = 42) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame]:
        """
        将数据集分割成两部分，确保视频文本映射和数据集对应
        
        Args:
            df: 原始DataFrame
            video_text_map: 原始视频文本映射
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            (part1_video_text_map, part2_video_text_map, part1_df, part2_df)
        """
        # 获取所有唯一的video_id
        video_ids = df['video_id'].unique()
        
        # 分割video_ids
        video_ids_part1, video_ids_part2 = train_test_split(
            video_ids, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # 分割DataFrame
        part1_df = df[df['video_id'].isin(video_ids_part1)]
        part2_df = df[df['video_id'].isin(video_ids_part2)]
        
        # 分割video_text_map
        part1_video_text_map = {
            k: v for k, v in video_text_map.items() 
            if any(vid in k for vid in video_ids_part1)
        }
        part2_video_text_map = {
            k: v for k, v in video_text_map.items() 
            if any(vid in k for vid in video_ids_part2)
        }
        
        self.logger.info(f"Split dataset - Part 1: {len(part1_df)} queries, {len(part1_video_text_map)} videos")
        self.logger.info(f"Split dataset - Part 2: {len(part2_df)} queries, {len(part2_video_text_map)} videos")
        
        return part1_video_text_map, part2_video_text_map, part1_df, part2_df

    def setup_analysis_dir(self, base_dir: str, part_name: str) -> str:
        """创建分析结果目录"""
        analysis_dir = os.path.join(base_dir, 'split_analysis_results', f'{part_name}_{self.timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        """批量编码文本"""
        dataset = TextDataset(texts)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        embeddings = []

        for batch in tqdm(dataloader, desc=desc):
            with autocast():
                batch_embedding = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embeddings.append(batch_embedding)

        return torch.cat(embeddings, dim=0)

    def find_top_k_similar(self, query: str, index: faiss.IndexFlatIP, 
                          video_text_map: Dict[str, str], video_files: List[str], 
                          video_id: str, k: int = 10) -> SimilarityResult:
        """查找最相似的前k个文本"""
        start_time = time.time()

        with autocast():
            query_embedding = self.model.encode(
                [query],
                convert_to_tensor=True,
                show_progress_bar=False
            )

        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)

        similarities, indices = index.search(query_embedding_np, k)
        
        top_k_videos = [video_files[idx] for idx in indices[0]]
        top_k_scores = similarities[0].tolist()
        top_k_texts = [video_text_map[video] for video in top_k_videos]
        
        match_positions = [i for i, video in enumerate(top_k_videos) if video_id in video]
        
        query_time = time.time() - start_time

        return SimilarityResult(
            video_files=top_k_videos,
            similarity_scores=top_k_scores,
            generated_texts=top_k_texts,
            query_time=query_time,
            match_positions=match_positions
        )

    def calculate_accuracy_at_k(self, results: List[Dict], k: int) -> float:
        """计算前k个结果的准确率"""
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def plot_comparison_accuracy_curve(self, accuracies1: List[float], accuracies2: List[float], 
                                     output_path: str):
        """
        绘制两部分数据的准确率对比曲线
        
        Args:
            accuracies1: 第一部分准确率列表
            accuracies2: 第二部分准确率列表
            output_path: 输出文件路径
        """
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(accuracies1) + 1), accuracies1, 'b-', marker='o', label='Part 1')
        plt.plot(range(1, len(accuracies2) + 1), accuracies2, 'r-', marker='s', label='Part 2')
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: Part 1 vs Part 2')
        plt.grid(True)
        plt.legend()
        
        # 添加数值标签
        for i, (acc1, acc2) in enumerate(zip(accuracies1, accuracies2)):
            plt.annotate(f'{acc1:.4f}', 
                        (i + 1, acc1),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        color='blue')
            plt.annotate(f'{acc2:.4f}', 
                        (i + 1, acc2),
                        textcoords="offset points", 
                        xytext=(0,-20), 
                        ha='center',
                        color='red')
        
        plt.savefig(output_path)
        plt.close()

    def analyze_part(self, data_dir: str, video_text_map: Dict[str, str], 
                    df: pd.DataFrame, part_name: str) -> Tuple[str, List[float]]:
        """
        分析单个部分的数据
        
        Args:
            data_dir: 数据目录路径
            video_text_map: 该部分的视频文本映射
            df: 该部分的DataFrame
            part_name: 部分名称
            
        Returns:
            (分析目录路径, 准确率列表)
        """
        analysis_dir = self.setup_analysis_dir(data_dir, part_name)
        
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())

        # 构建索引
        self.logger.info(f"构建{part_name}文本索引...")
        text_embeddings = self.encode_texts(texts, f"Encoding {part_name} texts")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        # 处理查询
        results = []
        total_queries = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=total_queries, desc=f"Processing {part_name} queries"):
            result = self.find_top_k_similar(
                row['caption'], 
                index, 
                video_text_map, 
                video_files,
                row['video_id']
            )
            
            results.append({
                'video_id': row['video_id'],
                'caption': row['caption'],
                'match_positions': result.match_positions,
                **{f'rank{i+1}_video': result.video_files[i] for i in range(10)},
                **{f'rank{i+1}_score': result.similarity_scores[i] for i in range(10)},
                **{f'rank{i+1}_text': result.generated_texts[i] for i in range(10)},
                'query_time': result.query_time
            })

        # 保存详细结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(analysis_dir, f'{part_name}_detailed_results.csv'), index=False)

        # 计算准确率
        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy

        # 保存准确率统计
        with open(os.path.join(analysis_dir, f'{part_name}_accuracy_stats.json'), 'w') as f:
            json.dump(accuracy_stats, f, indent=2)

        # 生成分析报告
        report = {
            'timestamp': self.timestamp,
            'total_queries': total_queries,
            'accuracy_stats': accuracy_stats,
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': str(self.model),
            'device': str(self.device)
        }

        with open(os.path.join(analysis_dir, f'{part_name}_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        return analysis_dir, accuracies

    def analyze_queries(self, data_dir: str, test_size: float = 0.5):
        """
        分析两部分数据并比较结果
        
        Args:
            data_dir: 数据目录路径
            test_size: 测试集比例
        """
        # 加载数据
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)

        # 分割数据集
        part1_video_text_map, part2_video_text_map, part1_df, part2_df = self.split_datasets(
            df, video_text_map, test_size
        )

        # 分析两部分数据
        analysis_dir1, accuracies1 = self.analyze_part(
            data_dir, part1_video_text_map, part1_df, "part1"
        )
        analysis_dir2, accuracies2 = self.analyze_part(
            data_dir, part2_video_text_map, part2_df, "part2"
        )

        # 绘制对比曲线
        comparison_dir = os.path.join(data_dir, 'split_analysis_results', f'comparison_{self.timestamp}')
        os.makedirs(comparison_dir, exist_ok=True)
        self.plot_comparison_accuracy_curve(
            accuracies1,
            accuracies2,
            os.path.join(comparison_dir, 'accuracy_comparison.png')
        )

        # 保存比较报告
        comparison_report = {
            'timestamp': self.timestamp,
            'part1_stats': {
                'total_queries': len(part1_df),
                'total_videos': len(part1_video_text_map),
                'accuracies': accuracies1
            },
            'part2_stats': {
                'total_queries': len(part2_df),
                'total_videos': len(part2_video_text_map),
                'accuracies': accuracies2
            },
            'accuracy_differences': [a1 - a2 for a1, a2 in zip(accuracies1, accuracies2)]
        }

        # 保存报告到文件
        with open(os.path.join(comparison_dir, 'comparison_report.json'), 'w') as f:
            json.dump(comparison_report, f, indent=2)

        # 打印比较结果
        print("\nComparison Summary:")
        print(f"Part 1: {len(part1_df)} queries, {len(part1_video_text_map)} videos")
        print(f"Part 2: {len(part2_df)} queries, {len(part2_video_text_map)} videos")
        print("\nAccuracy Comparison:")
        for k in range(1, 11):
            print(f"Top-{k}:")
            print(f"  Part 1: {accuracies1[k-1]:.4f}")
            print(f"  Part 2: {accuracies2[k-1]:.4f}")
            print(f"  Difference: {accuracies1[k-1] - accuracies2[k-1]:.4f}")

        self.logger.info(f"Comparison results saved in: {comparison_dir}")
        return comparison_dir

def main():
    """主函数"""
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    analyzer = SplitSBERTAnalyzer(batch_size=128)
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行分割分析
    comparison_dir = analyzer.analyze_queries(data_dir, test_size=0.5)
    print(f"\nAnalysis results saved in: {comparison_dir}")

if __name__ == "__main__":
    main()