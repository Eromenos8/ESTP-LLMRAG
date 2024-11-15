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

@dataclass
class SimilarityResult:
    youtube_ids: List[str]
    similarity_scores: List[float]
    descriptions: List[str]
    query_time: float
    match_positions: List[int]

class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class AudioCapsMatcher:
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 batch_size: int = 128,
                 cache_dir: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using cache directory: {cache_dir}")
        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        try:
            print("load model...")
            self.model = SentenceTransformer(model_name)
            print("load success")
        except Exception as e:
            print(f"load fail: {str(e)}")
            sys.exit(1)
                
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'sbert_matching_{self.timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, train_path: str, matched_results_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        train_df = pd.read_csv(train_path)
        matched_results_df = pd.read_csv(matched_results_path)
        
        
        common_ids = set(train_df['youtube_id']).intersection(set(matched_results_df['youtube_id']))
        
        
        print(f"Train.csv total: {len(train_df)}")
        print(f"Matched Results: {len(matched_results_df)}")
        print(f"common youtube_id: {len(common_ids)}")
        print(f"Not match: {len(train_df) - len(common_ids)}")
        
        
        train_filtered = train_df[train_df['youtube_id'].isin(common_ids)]
        matched_filtered = matched_results_df[matched_results_df['youtube_id'].isin(common_ids)]
        
        return train_filtered, matched_filtered

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
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

    def find_top_k_similar(
        self,
        query: str,
        index: faiss.IndexFlatIP,
        descriptions: List[str],
        youtube_ids: List[str],
        original_youtube_id: str,
        k: int = 10
    ) -> SimilarityResult:
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
        
        top_k_ids = [youtube_ids[idx] for idx in indices[0]]
        top_k_scores = similarities[0].tolist()
        top_k_descs = [descriptions[idx] for idx in indices[0]]
        
        
        match_positions = [i for i, youtube_id in enumerate(top_k_ids) 
                         if youtube_id == original_youtube_id]
        
        query_time = time.time() - start_time

        return SimilarityResult(
            youtube_ids=top_k_ids,
            similarity_scores=top_k_scores,
            descriptions=top_k_descs,
            query_time=query_time,
            match_positions=match_positions
        )

    def calculate_accuracy_at_k(self, results: List[Dict], k: int) -> float:
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def plot_accuracy_curve(self, accuracies: List[float], output_path: str):
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-', marker='o')
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Top-K Results')
        plt.grid(True)
        
        for i, accuracy in enumerate(accuracies):
            plt.annotate(f'{accuracy:.4f}', 
                        (i + 1, accuracy),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.savefig(output_path)
        plt.close()

    def process_and_match(self, train_path: str, matched_results_path: str, output_dir: str):
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        train_filtered, matched_filtered = self.prepare_data(train_path, matched_results_path)
        
        
        descriptions = matched_filtered['description'].tolist()
        youtube_ids = matched_filtered['youtube_id'].tolist()

        
        self.logger.info("index...")
        text_embeddings = self.encode_texts(descriptions, "coding description text")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        
        results = []
        
        total_captions = len(train_filtered)
        for idx, row in tqdm(train_filtered.iterrows(), total=total_captions, desc="handle data"):
            result = self.find_top_k_similar(
                row['caption'],
                index,
                descriptions,
                youtube_ids,
                row['youtube_id']
            )
            
            
            results.append({
                'original_youtube_id': row['youtube_id'],
                'caption': row['caption'],
                'match_positions': result.match_positions,
                **{f'rank{j+1}_youtube_id': result.youtube_ids[j] for j in range(10)},
                **{f'rank{j+1}_score': result.similarity_scores[j] for j in range(10)},
                **{f'rank{j+1}_desc': result.descriptions[j] for j in range(10)},
                'query_time': result.query_time
            })

        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)

        
        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy
            self.logger.info(f"Top-{k} Accuracy: {accuracy:.4f}")

        
        with open(os.path.join(output_dir, 'accuracy_stats.json'), 'w') as f:
            json.dump(accuracy_stats, f, indent=2)

        
        self.plot_accuracy_curve(
            accuracies,
            os.path.join(output_dir, 'accuracy_curve.png')
        )

        
        report = {
            'timestamp': self.timestamp,
            'total_captions': total_captions,
            'common_youtube_ids': len(set(youtube_ids)),
            'accuracy_stats': accuracy_stats,
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': str(self.model),
            'device': str(self.device)
        }

        
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"output path {output_dir}")
        print(f"\nsummary:")
        for k in range(1, 11):
            print(f"Top-{k}: {accuracy_stats[f'top{k}_accuracy']:.4f}")

def main():
    
    train_path = "/root/autodl-tmp/AudioCaps/train.csv"
    matched_results_path = "/root/autodl-tmp/VideoLLama2-audio/VideoLLaMA2/audio_descriptions/matched_results_filtered.csv"
    output_dir = "/root/autodl-tmp/AudioCaps/SBERT_matched_result"
    cache_dir = "/root/autodl-tmp/model_cache"
    
    
    os.makedirs(cache_dir, exist_ok=True)
    
    
    matcher = AudioCapsMatcher(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        batch_size=128,
        cache_dir=cache_dir
    )
    matcher.process_and_match(train_path, matched_results_path, output_dir)

if __name__ == "__main__":
    main()