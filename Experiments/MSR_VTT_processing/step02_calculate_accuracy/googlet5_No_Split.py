import torch
import pandas as pd
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
    """存储相似度计算结果的数据类"""
    video_files: List[str]
    similarity_scores: List[float]
    generated_texts: List[str]
    query_time: float
    match_positions: List[int]

class TextDataset(Dataset):
    """文本数据集类，用于批处理"""
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 为T5添加前缀
        text = f"encode text: {text}"
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class T5Analyzer:
    """使用T5的深度分析器"""
    def __init__(self, model_name: str = 'google-t5/t5-base', batch_size: int = 16):
        """
        初始化T5分析器
        
        Args:
            model_name: T5模型名称
            batch_size: 批处理大小（T5较大，batch_size要小一些）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 使用国内镜像加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            mirror='https://hf-mirror.com'
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            mirror='https://hf-mirror.com'
        )
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        """设置日志配置"""
        log_file = f't5_analysis_{self.timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_embeddings(self, model_output):
        """
        从T5输出获取嵌入向量
        对所有decoder输出层取平均作为文本的表示
        """
        # 获取最后一层隐藏状态
        last_hidden_state = model_output.last_hidden_state
        
        # 对序列维度取平均，得到文本表示
        embeddings = torch.mean(last_hidden_state, dim=1)
        return embeddings

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        """批量编码文本"""
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        embeddings = []

        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast():
                outputs = self.model.encoder(**batch, return_dict=True)
                batch_embeddings = self.get_embeddings(outputs)
                embeddings.append(batch_embeddings.detach())

        return torch.cat(embeddings, dim=0)

    def find_top_k_similar(
        self,
        query: str,
        index: faiss.IndexFlatIP,
        video_text_map: Dict[str, str],
        video_files: List[str],
        video_id: str,
        k: int = 10
    ) -> SimilarityResult:
        """查找最相似的前k个文本"""
        start_time = time.time()

        # 为查询添加前缀
        query = f"encode text: {query}"
        query_inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with autocast():
            query_outputs = self.model.encoder(**query_inputs, return_dict=True)
            query_embedding = self.get_embeddings(query_outputs).detach()

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

    def setup_analysis_dir(self, base_dir: str) -> str:
        """创建分析结果目录"""
        analysis_dir = os.path.join(base_dir, 'analysis_results', f't5_analysis_{self.timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

    def calculate_accuracy_at_k(self, results: List[Dict], k: int) -> float:
        """计算前k个结果的准确率"""
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def analyze_matching_pairs(self, video_text_map: Dict[str, str], 
                             df: pd.DataFrame) -> Dict:
        """分析匹配对的统计信息"""
        video_ids = df['video_id'].unique()
        matching_stats = {
            'total_videos': len(video_ids),
            'total_texts': len(video_text_map),
            'matching_pairs': {}
        }
        
        for video_id in video_ids:
            matches = [k for k in video_text_map.keys() if video_id in k]
            matching_stats['matching_pairs'][video_id] = len(matches)
        
        matching_stats['avg_matches_per_video'] = np.mean(list(matching_stats['matching_pairs'].values()))
        matching_stats['min_matches'] = min(matching_stats['matching_pairs'].values())
        matching_stats['max_matches'] = max(matching_stats['matching_pairs'].values())
        
        return matching_stats

    def plot_accuracy_curve(self, accuracies: List[float], output_path: str):
        """绘制准确率曲线"""
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-', marker='o')
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('T5 Model Accuracy vs Top-K Results')
        plt.grid(True)
        
        for i, accuracy in enumerate(accuracies):
            plt.annotate(f'{accuracy:.4f}', 
                        (i + 1, accuracy),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_queries(self, data_dir: str):
        """分析查询结果"""
        analysis_dir = self.setup_analysis_dir(data_dir)
        
        # 加载数据
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)

        # 分析匹配对统计
        matching_stats = self.analyze_matching_pairs(video_text_map, df)
            
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())

        # 构建索引
        self.logger.info("构建文本索引...")
        text_embeddings = self.encode_texts(texts, "Encoding texts")
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
            
            results.append({
                'video_id': row['video_id'],
                'caption': row['caption'],
                'match_positions': result.match_positions,
                'num_matches': len([k for k in video_text_map.keys() if row['video_id'] in k]),
                **{f'rank{i+1}_video': result.video_files[i] for i in range(10)},
                **{f'rank{i+1}_score': result.similarity_scores[i] for i in range(10)},
                **{f'rank{i+1}_text': result.generated_texts[i] for i in range(10)},
                'query_time': result.query_time
            })

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_queries} queries")

        # 保存详细结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(analysis_dir, 'detailed_results.csv'), index=False)

        # 计算准确率
        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy

        # 绘制准确率曲线
        self.plot_accuracy_curve(
            accuracies,
            os.path.join(analysis_dir, 'accuracy_curve.png')
        )

        # 生成分析报告
        report = {
            'timestamp': self.timestamp,
            'total_queries': total_queries,
            'matching_stats': matching_stats,
            'accuracy_stats': accuracy_stats,
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': 't5-base',
            'device': str(self.device)
        }

        with open(os.path.join(analysis_dir, 'analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        # 打印结果摘要
        print("\nAnalysis Summary:")
        print(f"Total Queries: {total_queries}")
        print(f"Total Videos: {matching_stats['total_videos']}")
        print(f"Avg matches per video: {matching_stats['avg_matches_per_video']:.2f}")
        print(f"Min/Max matches: {matching_stats['min_matches']}/{matching_stats['max_matches']}")
        
        print("\nAccuracy Results:")
        for k in range(1, 11):
            print(f"Top-{k}: {accuracy_stats[f'top{k}_accuracy']:.4f}")

        self.logger.info(f"Analysis completed. Results saved in {analysis_dir}")
        return analysis_dir

def main():
    """主函数"""
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    analyzer = T5Analyzer(batch_size=16)  # T5模型batch_size设置小一些
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行分析
    analysis_dir = analyzer.analyze_queries(data_dir)
    print(f"\nAnalysis results saved in: {analysis_dir}")

if __name__ == "__main__":
    main()