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

@dataclass
class SimilarityResult:
    """
    存储相似度计算结果的数据类
    
    Attributes:
        video_files: 前十个最相似视频的文件名列表
        similarity_scores: 对应的相似度分数列表
        generated_texts: 对应的生成文本列表
        query_time: 查询耗时
        match_positions: 匹配出现的位置列表
    """
    video_files: List[str]
    similarity_scores: List[float]
    generated_texts: List[str]
    query_time: float
    match_positions: List[int]  # 记录匹配出现在第几位

class TextDataset(Dataset):
    """
    文本数据集类，用于批处理
    """
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class SBERTAnalyzer:
    """SBERT深度分析器"""
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 128):
        """
        初始化SBERT分析器
        
        Args:
            model_name: SBERT模型名称
            batch_size: 批处理大小
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        """设置日志配置"""
        log_file = f'sbert_analysis_{self.timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        """
        批量编码文本
        
        Args:
            texts: 待编码的文本列表
            desc: 进度条描述
            
        Returns:
            文本向量张量
        """
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

    def setup_analysis_dir(self, base_dir: str) -> str:
        """
        创建分析结果目录
        
        Args:
            base_dir: 基础目录路径
            
        Returns:
            分析结果目录路径
        """
        analysis_dir = os.path.join(base_dir, 'analysis_results', f'analysis_{self.timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

    def find_top_k_similar(
        self,
        query: str,
        index: faiss.IndexFlatIP,
        video_text_map: Dict[str, str],
        video_files: List[str],
        video_id: str,
        k: int = 10
    ) -> SimilarityResult:
        """
        查找最相似的前k个文本并分析匹配位置
        
        Args:
            query: 查询文本
            index: FAISS索引
            video_text_map: 视频文本映射
            video_files: 视频文件列表
            video_id: 原始视频ID
            k: 返回的结果数量
            
        Returns:
            相似度结果对象
        """
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
        
        # 找出所有匹配位置
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
        """
        计算前k个结果的准确率
        
        Args:
            results: 结果列表
            k: 考虑的结果数量
            
        Returns:
            准确率
        """
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def plot_accuracy_curve(self, accuracies: List[float], output_path: str):
        """
        绘制准确率曲线
        
        Args:
            accuracies: 准确率列表
            output_path: 输出文件路径
        """
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-', marker='o')
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Top-K Results')
        plt.grid(True)
        
        # 添加数值标签
        for i, accuracy in enumerate(accuracies):
            plt.annotate(f'{accuracy:.4f}', 
                        (i + 1, accuracy),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.savefig(output_path)
        plt.close()

    def analyze_queries(self, data_dir: str):
        """
        深入分析查询结果
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            分析结果目录路径
        """
        # 创建分析目录
        analysis_dir = self.setup_analysis_dir(data_dir)
        
        # 加载数据
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)
            
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())

        # 构建索引
        self.logger.info("构建文本索引...")
        text_embeddings = self.encode_texts(texts, "Encoding generated texts")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        # 处理查询
        results = []
        total_queries = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=total_queries, desc="Processing queries"):
            result = self.find_top_k_similar(
                row['caption'], 
                index, 
                video_text_map, 
                video_files,
                row['video_id']
            )
            
            # 记录详细结果
            results.append({
                'video_id': row['video_id'],
                'caption': row['caption'],
                'match_positions': result.match_positions,
                **{f'rank{i+1}_video': result.video_files[i] for i in range(10)},
                **{f'rank{i+1}_score': result.similarity_scores[i] for i in range(10)},
                **{f'rank{i+1}_text': result.generated_texts[i] for i in range(10)},
                'query_time': result.query_time
            })

            # 定期打印进度
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_queries} queries")

        # 保存详细结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(analysis_dir, 'detailed_results.csv'), index=False)

        # 计算不同K值的准确率
        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy
            print(f"Top-{k} Accuracy: {accuracy:.4f}")

        # 保存准确率统计
        with open(os.path.join(analysis_dir, 'accuracy_stats.json'), 'w') as f:
            json.dump(accuracy_stats, f, indent=2)

        # 绘制准确率曲线
        self.plot_accuracy_curve(
            accuracies,
            os.path.join(analysis_dir, 'accuracy_curve.png')
        )

        # 生成分析报告
        report = {
            'timestamp': self.timestamp,
            'total_queries': total_queries,
            'accuracy_stats': accuracy_stats,
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': str(self.model),
            'device': str(self.device)
        }

        # 保存分析报告
        with open(os.path.join(analysis_dir, 'analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Analysis completed. Results saved in {analysis_dir}")
        print(f"\nAccuracy Summary:")
        for k in range(1, 11):
            print(f"Top-{k}: {accuracy_stats[f'top{k}_accuracy']:.4f}")
        
        return analysis_dir

def main():
    """主函数"""
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    analyzer = SBERTAnalyzer(batch_size=128)
    analysis_dir = analyzer.analyze_queries(data_dir)
    print(f"\nAnalysis results saved in: {analysis_dir}")

if __name__ == "__main__":
    main()