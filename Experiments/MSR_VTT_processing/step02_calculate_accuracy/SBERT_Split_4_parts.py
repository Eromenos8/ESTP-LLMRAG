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
from sklearn.model_selection import train_test_split

@dataclass
class SimilarityResult:
    video_files: List[str]
    similarity_scores: List[float]
    generated_texts: List[str]
    query_time: float
    match_positions: List[int]

class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class SplitSBERTAnalyzer:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
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
        video_ids = df['video_id'].unique()
        video_ids_half1, video_ids_half2 = train_test_split(video_ids, test_size=0.5, random_state=random_state)
        video_ids_part1, video_ids_part2 = train_test_split(video_ids_half1, test_size=0.5, random_state=random_state)
        video_ids_part3, video_ids_part4 = train_test_split(video_ids_half2, test_size=0.5, random_state=random_state)
        part_dfs = []
        part_video_text_maps = []

        for part_ids in [video_ids_part1, video_ids_part2, video_ids_part3, video_ids_part4]:
            part_df = df[df['video_id'].isin(part_ids)]
            part_dfs.append(part_df)
            part_map = {k: v for k, v in video_text_map.items() if any(vid in k for vid in part_ids)}
            part_video_text_maps.append(part_map)
            self.logger.info(f"Split dataset - Part {len(part_dfs)}: {len(part_df)} queries, {len(part_map)} videos")

        return part_video_text_maps, part_dfs

    def setup_analysis_dir(self, base_dir: str, part_name: str) -> str:
        analysis_dir = os.path.join(base_dir, 'split_analysis_results', f'{part_name}_{self.timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        dataset = TextDataset(texts)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        embeddings = []

        for batch in tqdm(dataloader, desc=desc):
            with autocast():
                batch_embedding = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                embeddings.append(batch_embedding)

        return torch.cat(embeddings, dim=0)

    def find_top_k_similar(self, query: str, index: faiss.IndexFlatIP, video_text_map: Dict[str, str], video_files: List[str], video_id: str, k: int = 10) -> SimilarityResult:
        start_time = time.time()
        with autocast():
            query_embedding = self.model.encode([query], convert_to_tensor=True, show_progress_bar=False)
        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)
        similarities, indices = index.search(query_embedding_np, k)
        top_k_videos = [video_files[idx] for idx in indices[0]]
        top_k_scores = similarities[0].tolist()
        top_k_texts = [video_text_map[video] for video in top_k_videos]
        match_positions = [i for i, video in enumerate(top_k_videos) if video_id in video]
        query_time = time.time() - start_time
        return SimilarityResult(top_k_videos, top_k_scores, top_k_texts, query_time, match_positions)

    def calculate_accuracy_at_k(self, results: List[Dict], k: int) -> float:
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def plot_four_part_comparison_curve(self, accuracies_list: List[List[float]], output_path: str):
        plt.figure(figsize=(15, 10))
        colors = ['blue', 'red', 'green', 'purple']
        markers = ['o', 's', '^', 'D']
        for i, accuracies in enumerate(accuracies_list, 1):
            plt.plot(range(1, len(accuracies) + 1), accuracies, color=colors[i-1], marker=markers[i-1], label=f'Part {i}')
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: Four Parts')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def analyze_part(self, data_dir: str, video_text_map: Dict[str, str], df: pd.DataFrame, part_name: str) -> Tuple[str, List[float]]:
        analysis_dir = self.setup_analysis_dir(data_dir, part_name)
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())
        text_embeddings = self.encode_texts(texts, f"Encoding {part_name} texts")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {part_name} queries"):
            result = self.find_top_k_similar(row['caption'], index, video_text_map, video_files, row['video_id'])
            results.append({
                'video_id': row['video_id'],
                'caption': row['caption'],
                'match_positions': result.match_positions,
                'query_time': result.query_time,
                **{f'rank{i+1}_video': result.video_files[i] for i in range(10)},
                **{f'rank{i+1}_score': result.similarity_scores[i] for i in range(10)},
                **{f'rank{i+1}_text': result.generated_texts[i] for i in range(10)}
            })
        accuracies = [self.calculate_accuracy_at_k(results, k) for k in range(1, 11)]
        return analysis_dir, accuracies

    def analyze_queries(self, data_dir: str):
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)
        part_video_text_maps, part_dfs = self.split_datasets_four_parts(df, video_text_map)
        analysis_results = []
        accuracies_list = []
        for i, (part_map, part_df) in enumerate(zip(part_video_text_maps, part_dfs), 1):
            analysis_dir, accuracies = self.analyze_part(data_dir, part_map, part_df, f"part{i}")
            analysis_results.append((analysis_dir, accuracies))
            accuracies_list.append(accuracies)
        comparison_dir = os.path.join(data_dir, 'split_analysis_results', f'comparison_{self.timestamp}')
        os.makedirs(comparison_dir, exist_ok=True)
        self.plot_four_part_comparison_curve(accuracies_list, os.path.join(comparison_dir, 'accuracy_comparison.png'))
        print("\nComparison Summary:")
        for i, (part_df, part_map) in enumerate(zip(part_dfs, part_video_text_maps), 1):
            print(f"Part {i}: {len(part_df)} queries, {len(part_map)} videos")

def main():
        data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
        analyzer = SplitSBERTAnalyzer(batch_size=128)
        analyzer.analyze_queries(data_dir)

if __name__ == "__main__":
    main()