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
import sys

@dataclass
class SimilarityResult:
    youtube_ids: List[str]
    similarity_scores: List[float]
    descriptions: List[str]
    query_time: float
    match_positions: List[int]

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = min(512, max(len(self.tokenizer.encode(text)) for text in texts))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = f"encode text: {text}"  
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class AudioCapsMatcher:
    def __init__(self, 
                 model_name: str = 'google-t5/t5-base',
                 batch_size: int = 16,
                 cache_dir: str = None,
                 num_parts: int = 2): 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_parts = num_parts 
        print(f"Using device: {self.device}")
        print(f"Data will be split into {num_parts} parts")
        
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using cache directory: {cache_dir}")
        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        try:
            print("load model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
        log_file = os.path.join(log_dir, f't5_matching_{self.timestamp}.log')
        
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
        last_hidden_state = model_output.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1)
        return embeddings

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        embeddings = []

        self.model.eval()
        
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda'):
                outputs = self.model.encoder(**batch, return_dict=True)
                batch_embeddings = self.get_embeddings(outputs).detach()
                embeddings.append(batch_embeddings.cpu())
            
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        final_embeddings = torch.cat(embeddings, dim=0)
        return final_embeddings

    def split_datasets(self, train_df: pd.DataFrame, matched_df: pd.DataFrame, 
                      num_parts: int = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if num_parts is None:
            num_parts = self.num_parts

        matched_splits = np.array_split(matched_df.sample(frac=1, random_state=42), num_parts)
        
        paired_splits = []
        for matched_split in matched_splits:
            split_ids = set(matched_split['youtube_id'])
            train_split = train_df[train_df['youtube_id'].isin(split_ids)]
            paired_splits.append((matched_split, train_split))
            
            print(f"Split size - Matched: {len(matched_split)}, Train: {len(train_split)}")
            
        return paired_splits

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

        query = f"encode text: {query}"
        query_inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.amp.autocast('cuda'):
            query_outputs = self.model.encoder(**query_inputs, return_dict=True)
            query_embedding = self.get_embeddings(query_outputs).detach()
            query_embedding = query_embedding.cpu()

        query_embedding_np = query_embedding.numpy()
        faiss.normalize_L2(query_embedding_np)

        similarities, indices = index.search(query_embedding_np, k)
        
        top_k_ids = [youtube_ids[idx] for idx in indices[0]]
        top_k_scores = similarities[0].tolist()
        top_k_descs = [descriptions[idx] for idx in indices[0]]
        match_positions = [i for i, youtube_id in enumerate(top_k_ids) 
                         if original_youtube_id == youtube_id]
        
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

    def plot_accuracy_comparison(self, all_accuracies: List[List[float]], 
                             all_stats: List[Dict], output_dir: str):
        plt.figure(figsize=(20, 12))
        
        n_parts = len(all_accuracies)
        if n_parts <= 4:
            nrows, ncols = 1, 1 
        else:
            nrows, ncols = 2, 1 
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 12*nrows))
        if nrows == 1 and ncols == 1:
            axs = [axs] 
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_parts))
        markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '8'][:n_parts]
        
        parts_per_plot = (n_parts + 1) // 2 if nrows == 2 else n_parts
        
        for plot_idx in range(nrows):
            ax = axs[plot_idx] if nrows == 2 else axs[0]
            start_idx = plot_idx * parts_per_plot
            end_idx = min(start_idx + parts_per_plot, n_parts)
            
            for i in range(start_idx, end_idx):
                accuracies = all_accuracies[i]
                stats = all_stats[i]
                
                ax.plot(range(1, len(accuracies) + 1), accuracies,
                       marker=markers[i], color=colors[i],
                       label=f'Part {i+1} (Size: {stats["total_queries"]})')
                
                for j, acc in enumerate(accuracies):
                    ax.annotate(f'{acc:.3f}',
                              (j + 1, acc),
                              textcoords="offset points",
                              xytext=(0, 10 if i % 2 == 0 else -15),
                              ha='center',
                              color=colors[i],
                              fontsize=8)
            
            mean_accuracies = np.mean(all_accuracies[start_idx:end_idx], axis=0)
            ax.plot(range(1, len(mean_accuracies) + 1), mean_accuracies,
                   marker='*', color='red', linewidth=2,
                   label='Average')
            
            for j, mean_acc in enumerate(mean_accuracies):
                ax.annotate(f'Avg: {mean_acc:.3f}',
                          (j + 1, mean_acc),
                          textcoords="offset points",
                          xytext=(0, 20),
                          ha='center',
                          color='red',
                          fontsize=8)
            
            ax.set_xlabel('Top-K')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'T5 Accuracy Comparison (Parts {start_idx+1}-{end_idx})')
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'accuracy_comparison_{n_parts}parts.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def process_single_part(self, matched_df: pd.DataFrame, train_df: pd.DataFrame,
                          part_id: int, output_dir: str) -> Tuple[List[float], Dict]:
        part_dir = os.path.join(output_dir, f'part{part_id}')
        os.makedirs(part_dir, exist_ok=True)
        
        descriptions = matched_df['description'].tolist()
        youtube_ids = matched_df['youtube_id'].tolist()

        self.logger.info(f"构建Part {part_id}的文本索引...")
        text_embeddings = self.encode_texts(descriptions, f"编码Part {part_id}描述文本")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        results = []
        total_queries = len(train_df)
        
        for idx, row in tqdm(train_df.iterrows(), total=total_queries, 
                           desc=f"处理Part {part_id}数据集"):
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
        results_df.to_csv(os.path.join(part_dir, 'detailed_results.csv'), index=False)

        accuracies = []
        stats = {
            'total_queries': total_queries,
            'total_candidates': len(descriptions),
            'avg_query_time': np.mean([r['query_time'] for r in results])
        }

        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            stats[f'top{k}_accuracy'] = accuracy

        return accuracies, stats

    def process_and_match(self, train_path: str, matched_results_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        train_filtered, matched_filtered = self.prepare_data(train_path, matched_results_path)
        
        splits = self.split_datasets(train_filtered, matched_filtered, self.num_parts)
        
        all_accuracies = []
        all_stats = []
        
        for i, (matched_split, train_split) in enumerate(splits, 1):
            print(f"\nhandle part {i}/{self.num_parts} ")
            accuracies, stats = self.process_single_part(
                matched_split, 
                train_split, 
                i, 
                output_dir
            )
            all_accuracies.append(accuracies)
            all_stats.append(stats)

        self.plot_accuracy_comparison(all_accuracies, all_stats, output_dir)
        
        overall_report = {
            'timestamp': self.timestamp,
            'model_name': 't5-base',
            'device': str(self.device),
            'num_parts': self.num_parts,
            'parts_stats': all_stats,
            'average_accuracies': {
                f'top_{k+1}': float(np.mean([acc[k] for acc in all_accuracies]))
                for k in range(10)
            }
        }

        with open(os.path.join(output_dir, 'overall_report.json'), 'w') as f:
            json.dump(overall_report, f, indent=2)

        print("\nsummary:")
        for part_id, accuracies in enumerate(all_accuracies, 1):
            print(f"\nPart {part_id}:")
            for k, acc in enumerate(accuracies, 1):
                print(f"Top-{k}: {acc:.4f}")
            
        print("\nmean accuracies:")
        mean_accuracies = np.mean(all_accuracies, axis=0)
        std_accuracies = np.std(all_accuracies, axis=0)
        for k, (mean, std) in enumerate(zip(mean_accuracies, std_accuracies), 1):
            print(f"Top-{k}: {mean:.4f} ± {std:.4f}")

def main():

    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '4'
    torch.backends.cudnn.benchmark = True
    
    train_path = "/root/autodl-tmp/AudioCaps/train.csv"
    matched_results_path = "/root/autodl-tmp/VideoLLama2-audio/VideoLLaMA2/audio_descriptions/matched_results_filtered.csv"
    cache_dir = "/root/autodl-tmp/model_cache"
    
    # split in different parts
    for num_parts in [1, 2, 4, 8]:
        print(f"\n测试 {num_parts} 个parts的情况")
        output_dir = f"/root/autodl-tmp/AudioCaps/T5_matched_result_{num_parts}parts"
        
        matcher = AudioCapsMatcher(
            model_name='google-t5/t5-base',
            batch_size=16,
            cache_dir=cache_dir,
            num_parts=num_parts
        )
        matcher.process_and_match(train_path, matched_results_path, output_dir)

if __name__ == "__main__":
    main()

