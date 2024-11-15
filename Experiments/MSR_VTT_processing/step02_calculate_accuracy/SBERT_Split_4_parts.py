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
    """支持数据集四分割的SBERT深度分析器"""
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

    def split_datasets_four_parts(self, df: pd.DataFrame, video_text_map: Dict[str, str], random_state: int = 42) -> Tuple[List[Dict], List[pd.DataFrame]]:
        """
        将数据集分割成四部分，确保视频文本映射和数据集对应
        
        Args:
            df: 原始DataFrame
            video_text_map: 原始视频文本映射
            random_state: 随机种子
            
        Returns:
            (video_text_maps_list, dfs_list)
        """
        # 获取所有唯一的video_id
        video_ids = df['video_id'].unique()
        
        # 第一次分割成两部分
        video_ids_half1, video_ids_half2 = train_test_split(
            video_ids, 
            test_size=0.5, 
            random_state=random_state
        )
        
        # 进一步分割每一半
        video_ids_part1, video_ids_part2 = train_test_split(
            video_ids_half1,
            test_size=0.5,
            random_state=random_state
        )
        
        video_ids_part3, video_ids_part4 = train_test_split(
            video_ids_half2,
            test_size=0.5,
            random_state=random_state
        )
        
        # 分割DataFrame
        part_dfs = []
        part_video_text_maps = []
        
        for part_ids in [video_ids_part1, video_ids_part2, video_ids_part3, video_ids_part4]:
            # 分割DataFrame
            part_df = df[df['video_id'].isin(part_ids)]
            part_dfs.append(part_df)
            
            # 分割video_text_map
            part_map = {
                k: v for k, v in video_text_map.items() 
                if any(vid in k for vid in part_ids)
            }
            part_video_text_maps.append(part_map)
            
            self.logger.info(f"Split dataset - Part {len(part_dfs)}: {len(part_df)} queries, {len(part_map)} videos")
        
        return part_video_text_maps, part_dfs

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

    def plot_four_part_comparison_curve(self, accuracies_list: List[List[float]], 
                                      output_path: str):
        """
        绘制四部分数据的准确率对比曲线
        
        Args:
            accuracies_list: 包含四个部分准确率列表的列表
            output_path: 输出文件路径
        """
        plt.figure(figsize=(15, 10))
        colors = ['blue', 'red', 'green', 'purple']
        markers = ['o', 's', '^', 'D']
        
        for i, accuracies in enumerate(accuracies_list, 1):
            plt.plot(range(1, len(accuracies) + 1), accuracies, 
                    color=colors[i-1], marker=markers[i-1], 
                    label=f'Part {i}')
            
            # 添加数值标签
            for j, acc in enumerate(accuracies):
                plt.annotate(f'{acc:.4f}', 
                            (j + 1, acc),
                            textcoords="offset points", 
                            xytext=(0, 5 + (i-1)*(-15)), # 错开标签位置
                            ha='center',
                            color=colors[i-1])
        
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: Four Parts')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
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

    def analyze_queries(self, data_dir: str):
        """
        分析四部分数据并比较结果
        
        Args:
            data_dir: 数据目录路径
        """
        # 加载数据
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)

        # 分割数据集为四部分
        part_video_text_maps, part_dfs = self.split_datasets_four_parts(df, video_text_map)

        # 分析四个部分数据
        analysis_results = []
        accuracies_list = []
        
        for i, (part_map, part_df) in enumerate(zip(part_video_text_maps, part_dfs), 1):
            analysis_dir, accuracies = self.analyze_part(
                data_dir, part_map, part_df, f"part{i}"
            )
            analysis_results.append((analysis_dir, accuracies))
            accuracies_list.append(accuracies)

        # 创建比较目录
        comparison_dir = os.path.join(data_dir, 'split_analysis_results', f'comparison_{self.timestamp}')
        os.makedirs(comparison_dir, exist_ok=True)

        # 绘制四部分对比曲线
        self.plot_four_part_comparison_curve(
            accuracies_list,
            os.path.join(comparison_dir, 'accuracy_comparison.png')
        )

        # 计算各部分之间的准确率差异
        accuracy_differences = {}
        for i in range(4):
            for j in range(i + 1, 4):
                key = f'part{i+1}_vs_part{j+1}'
                accuracy_differences[key] = [
                    accuracies_list[i][k] - accuracies_list[j][k] 
                    for k in range(len(accuracies_list[i]))
                ]

        # 保存比较报告
        comparison_report = {
            'timestamp': self.timestamp,
            'parts_stats': [
                {
                    'part_number': i+1,
                    'total_queries': len(part_df),
                    'total_videos': len(part_map),
                    'accuracies': accs
                }
                for i, (part_df, part_map, accs) in enumerate(zip(part_dfs, part_video_text_maps, accuracies_list))
            ],
            'accuracy_differences': accuracy_differences
        }

        with open(os.path.join(comparison_dir, 'comparison_report.json'), 'w') as f:
            json.dump(comparison_report, f, indent=2)

        # 打印比较结果
        print("\nComparison Summary:")
        for i, (part_df, part_map) in enumerate(zip(part_dfs, part_video_text_maps), 1):
            print(f"Part {i}: {len(part_df)} queries, {len(part_map)} videos")

        print("\nAccuracy Comparison:")
        for k in range(1, 11):
            print(f"\nTop-{k}:")
            for i, accuracies in enumerate(accuracies_list, 1):
                print(f"  Part {i}: {accuracies[k-1]:.4f}")
            
            # 打印各部分之间的差异
            print("  Differences:")
            for i in range(4):
                for j in range(i + 1, 4):
                    diff = accuracies_list[i][k-1] - accuracies_list[j][k-1]
                    print(f"    Part {i+1} vs Part {j+1}: {diff:.4f}")

        self.logger.info(f"Comparison results saved in: {comparison_dir}")
        return comparison_dir

def main():
    """主函数"""
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    analyzer = SplitSBERTAnalyzer(batch_size=128)
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行四分割分析
    comparison_dir = analyzer.analyze_queries(data_dir)
    print(f"\nAnalysis results saved in: {comparison_dir}")

if __name__ == "__main__":
    main()