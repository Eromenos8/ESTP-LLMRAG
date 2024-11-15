import torch
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import os
from torch.cuda.amp import autocast
import faiss
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

class ComparisonAnalyzer:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 batch_size: int = 128,
                 cache_dir: str = None,
                 n_splits: int = 8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_splits = n_splits
        print(f"Using device: {self.device}")
        print(f"Data will be split into {n_splits} parts")
        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            
        self.model = SentenceTransformer(model_name).to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def split_data(self, matched_results_path: str, train_filtered_path: str) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        matched_df = pd.read_csv(matched_results_path)
        train_df = pd.read_csv(train_filtered_path)
        
        matched_df = matched_df.sample(frac=1, random_state=42).reset_index(drop=True)
        splits = np.array_split(matched_df, self.n_splits)
        
        paired_splits = []
        for split in splits:
            split_ids = set(split['youtube_id'])
            train_split = train_df[train_df['youtube_id'].isin(split_ids)]
            paired_splits.append((split, train_split))
            
        return paired_splits

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> np.ndarray:
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch = texts[i:i + self.batch_size]
            with autocast():
                batch_embedding = self.model.encode(batch, convert_to_tensor=True)
                embeddings.append(batch_embedding.cpu().numpy())
        return np.vstack(embeddings)

    def calculate_accuracy(self, query_texts: List[str], ref_texts: List[str],
                         query_ids: List[str], ref_ids: List[str]) -> List[float]:
        query_embeddings = self.encode_texts(query_texts, "Encoding queries")
        ref_embeddings = self.encode_texts(ref_texts, "Encoding references")
        
        index = faiss.IndexFlatIP(ref_embeddings.shape[1])
        faiss.normalize_L2(ref_embeddings)
        index.add(ref_embeddings)
        
        faiss.normalize_L2(query_embeddings)
        similarities, indices = index.search(query_embeddings, 10)
        
        accuracies = []
        for k in range(1, 11):
            correct = 0
            for i, query_id in enumerate(query_ids):
                top_k_ids = [ref_ids[idx] for idx in indices[i][:k]]
                if query_id in top_k_ids:
                    correct += 1
            accuracies.append(correct / len(query_ids))
            
        return accuracies

    def plot_comparison(self, all_accuracies: List[List[float]], output_dir: str):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
        
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_splits))
        markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '8', 'o', 's'][:self.n_splits]
        
        for i, accuracies in enumerate(all_accuracies):
            ax1.plot(range(1, 11), accuracies, 
                    label=f'Split {i+1}',
                    color=colors[i],
                    marker=markers[i],
                    linewidth=2,
                    markersize=8)
            
            for k, acc in enumerate(accuracies):
                ax1.annotate(f'{acc:.3f}', 
                           (k+1, acc),
                           textcoords="offset points",
                           xytext=(0,5),
                           ha='center',
                           fontsize=8,
                           color=colors[i])
        
        ax1.set_xlabel('Top-K', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy Comparison Across All Data Splits', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        mean_accuracies = np.mean(all_accuracies, axis=0)
        std_accuracies = np.std(all_accuracies, axis=0)
        
        ax2.plot(range(1, 11), mean_accuracies, 
                label='Mean Accuracy',
                color='blue',
                marker='o',
                linewidth=2,
                markersize=8)
        
        ax2.fill_between(range(1, 11), 
                        mean_accuracies - std_accuracies,
                        mean_accuracies + std_accuracies,
                        alpha=0.2,
                        color='blue',
                        label='±1 STD')
        
        for k, (mean, std) in enumerate(zip(mean_accuracies, std_accuracies)):
            ax2.annotate(f'Mean: {mean:.3f}\nSTD: {std:.3f}', 
                        (k+1, mean),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=10)
        
        ax2.set_xlabel('Top-K', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Average Accuracy Across All Splits', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison_8splits.png'), 
                   dpi=300, 
                   bbox_inches='tight')
        plt.close()

    def run_comparison(self, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)
        
        matched_path = "/root/autodl-tmp/VideoLLama2-audio/VideoLLaMA2/audio_descriptions/matched_results_filtered.csv"
        train_path = "/root/autodl-tmp/AudioCaps/train_filtered.csv"
        
        splits = self.split_data(matched_path, train_path)
        
        all_accuracies = []
        split_stats = []
        
        for i, (matched_split, train_split) in enumerate(splits, 1):
            print(f"\nhandle {i}/{self.n_splits} part")
            print(f"matched_split 大小: {len(matched_split)}")
            print(f"train_split 大小: {len(train_split)}")
            
            accuracies = self.calculate_accuracy(
                train_split['caption'].tolist(),
                matched_split['description'].tolist(),
                train_split['youtube_id'].tolist(),
                matched_split['youtube_id'].tolist()
            )
            
            all_accuracies.append(accuracies)
            
            split_stats.append({
                'split_id': i,
                'matched_size': len(matched_split),
                'train_size': len(train_split),
                'common_ids': len(set(matched_split['youtube_id']).intersection(set(train_split['youtube_id']))),
                'accuracies': {f'top_{k+1}': acc for k, acc in enumerate(accuracies)}
            })
        
        mean_accuracies = np.mean(all_accuracies, axis=0)
        std_accuracies = np.std(all_accuracies, axis=0)
        
        split_stats.append({
            'split_id': 'average',
            'accuracies': {
                f'top_{k+1}': {
                    'mean': mean_accuracies[k],
                    'std': std_accuracies[k]
                } for k in range(10)
            }
        })
        
        with open(os.path.join(output_dir, 'split_statistics_8splits.json'), 'w') as f:
            json.dump(split_stats, f, indent=2)
            
        self.plot_comparison(all_accuracies, output_dir)
        
        print("\nTop-K for each part:")
        for i, accs in enumerate(all_accuracies, 1):
            print(f"\n part {i} :")
            for k, acc in enumerate(accs, 1):
                print(f"Top-{k}: {acc:.4f}")
                
        print("\nmean accuracies:")
        for k, (mean, std) in enumerate(zip(mean_accuracies, std_accuracies), 1):
            print(f"Top-{k}: {mean:.4f} ± {std:.4f}")

def main():
    output_dir = "/root/autodl-tmp/AudioCaps/SBERT_comparison_results"
    cache_dir = "/root/autodl-tmp/model_cache"
    
    analyzer = ComparisonAnalyzer(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        batch_size=128,
        cache_dir=cache_dir,
        n_splits=8
    )
    
    analyzer.run_comparison(output_dir)

if __name__ == "__main__":
    main()