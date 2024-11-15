import torch
import pandas as pd
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
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
import re

@dataclass
class SimilarityResult:
    video_files: List[str]
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
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class ClothoALBERTMatcher:
    def __init__(self, 
                 model_name: str = 'albert/albert-base-v2',
                 batch_size: int = 32,
                 cache_dir: str = None,
                 n_splits: int = 4):
                 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_splits = n_splits
        print(f"Using device: {self.device}")
        print(f"Data will be split into {n_splits} parts")
        
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using cache directory: {cache_dir}")
        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        try:
            print("Loading model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            sys.exit(1)
                
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'albert_clotho_matching_{self.timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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
                outputs = self.model(**batch, return_dict=True)
                batch_embeddings = self.mean_pooling(outputs, batch['attention_mask']).detach()
                embeddings.append(batch_embeddings.cpu())
            
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        final_embeddings = torch.cat(embeddings, dim=0)
        return final_embeddings

    def prepare_and_split_data(self, clotho_df: pd.DataFrame, results_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        print("\nData preprocessing:")
        print(f"Original Clotho dataset size: {len(clotho_df)}")
        print(f"Original Results dataset size: {len(results_df)}")

        def clean_filename(filename):
            try:
                filename = filename.replace('.wav', '').lower()
                filename = re.sub(r'[^a-z\s]', '', filename)
                return ' '.join(filename.split())
            except:
                return filename

        results_df['original_filename'] = results_df['filename']
        clotho_df['original_filename'] = clotho_df['file_name']
        results_df['clean_filename'] = results_df['filename'].apply(clean_filename)
        clotho_df['clean_filename'] = clotho_df['file_name'].apply(clean_filename)

        print("\nCleaned filename samples:")
        print(pd.DataFrame({
            'Original': clotho_df['file_name'].head(),
            'Cleaned': clotho_df['clean_filename'].head()
        }))

        common_files = set(results_df['clean_filename']) & set(clotho_df['clean_filename'])
        print(f"\nNumber of common files: {len(common_files)}")

        if len(common_files) == 0:
            raise ValueError("No matching files found, please check filename format")

        clotho_filtered = clotho_df[clotho_df['clean_filename'].isin(common_files)].copy()
        results_filtered = results_df[results_df['clean_filename'].isin(common_files)].copy()

        print(f"Filtered Clotho dataset size: {len(clotho_filtered)}")
        print(f"Filtered Results dataset size: {len(results_filtered)}")

        clotho_filtered = clotho_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
        splits = np.array_split(clotho_filtered, self.n_splits)

        for i, split in enumerate(splits, 1):
            split_files = set(split['clean_filename'])
            matching_results = results_filtered[results_filtered['clean_filename'].isin(split_files)]
            print(f"\nSplit {i} statistics:")
            print(f"Clotho samples: {len(split)}")
            print(f"Corresponding Results samples: {len(matching_results)}")
            print(f"Unique files: {len(split_files)}")

        return splits, results_filtered

    def find_top_k_similar(self, 
                          query: str, 
                          index: faiss.IndexFlatIP,
                          descriptions: List[str], 
                          filenames: List[str],
                          original_filename: str, 
                          k: int = 10) -> SimilarityResult:
        start_time = time.time()

        query_inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.amp.autocast('cuda'):
            query_outputs = self.model(**query_inputs, return_dict=True)
            query_embedding = self.mean_pooling(query_outputs, query_inputs['attention_mask']).detach()
            query_embedding = query_embedding.cpu()

        query_embedding_np = query_embedding.numpy()
        faiss.normalize_L2(query_embedding_np)

        similarities, indices = index.search(query_embedding_np, k)
        
        top_k_files = [filenames[idx] for idx in indices[0]]
        top_k_scores = similarities[0].tolist()
        top_k_descs = [descriptions[idx] for idx in indices[0]]
        
        original_filename = re.sub(r'[^a-z\s]', '', original_filename.replace('.wav', '').lower())
        match_positions = [i for i, fname in enumerate(top_k_files) 
                         if re.sub(r'[^a-z\s]', '', fname.replace('.wav', '').lower()) == original_filename]
        
        return SimilarityResult(
            video_files=top_k_files,
            similarity_scores=top_k_scores,
            descriptions=top_k_descs,
            query_time=time.time() - start_time,
            match_positions=match_positions
        )

    def process_split(self, clotho_split: pd.DataFrame, results_df: pd.DataFrame,
                     split_id: int) -> List[float]:
        print(f"\nProcessing part {split_id}/{self.n_splits} (size: {len(clotho_split)})")
        
        split_files = set(clotho_split['clean_filename'])
        matching_results = results_df[results_df['clean_filename'].isin(split_files)]

        descriptions = matching_results['description'].tolist()
        filenames = matching_results['filename'].tolist()

        if not descriptions:
            raise ValueError(f"Split {split_id} has no matching descriptions")

        text_embeddings = self.encode_texts(descriptions, f"Split {split_id}: Encoding descriptions")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        results = []
        for idx, row in tqdm(clotho_split.iterrows(), total=len(clotho_split), 
                           desc=f"Processing split {split_id}"):
            file_name = row['file_name']
            
            for i in range(1, 6):
                caption_key = f'caption_{i}'
                if caption_key in row and isinstance(row[caption_key], str):
                    result = self.find_top_k_similar(
                        row[caption_key],
                        index,
                        descriptions,
                        filenames,
                        file_name
                    )
                    
                    results.append({
                        'split_id': split_id,
                        'original_file': file_name,
                        'caption': row[caption_key],
                        'match_positions': result.match_positions,
                        'top_k_scores': result.similarity_scores[:5]
                    })

        accuracies = []
        for k in range(1, 11):
            accuracy = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
            accuracy = accuracy / len(results) if results else 0
            accuracies.append(accuracy)
            
        return accuracies

    def plot_comparison(self, all_accuracies: List[List[float]], output_dir: str):
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
                ax.plot(range(1, 11), accuracies,
                       marker=markers[i], color=colors[i],
                       label=f'Part {i+1}')
                
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
            ax.set_title(f'ALBERT Accuracy Comparison (Parts {start_idx+1}-{end_idx})')
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'accuracy_comparison_{n_parts}parts.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        return np.mean(all_accuracies, axis=0), np.std(all_accuracies, axis=0)

    def match_captions(self, results_path: str, clotho_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            results_df = pd.read_csv(results_path)
            clotho_df = pd.read_csv(clotho_path)
        except Exception as e:
            self.logger.error(f"Failed to read data files: {str(e)}")
            raise

        clotho_splits, results_filtered = self.prepare_and_split_data(clotho_df, results_df)
        
        all_accuracies = []
        split_stats = []
        
        for i, split in enumerate(clotho_splits, 1):
            try:
                accuracies = self.process_split(split, results_filtered, i)
                all_accuracies.append(accuracies)
                
                split_stats.append({
                    'split_id': i,
                    'split_size': len(split),
                    'accuracies': {f'top_{k+1}': acc for k, acc in enumerate(accuracies)}
                })
            except Exception as e:
                self.logger.error(f"Error processing split {i}: {str(e)}")
                continue

        mean_accuracies, std_accuracies = self.plot_comparison(all_accuracies, output_dir)
        
        overall_report = {
            'timestamp': self.timestamp,
            'model_name': 'albert-base-v2',
            'device': str(self.device),
            'n_splits': self.n_splits,
            'total_samples': sum(len(split) for split in clotho_splits),
            'samples_per_split': [len(split) for split in clotho_splits],
            'mean_accuracies': mean_accuracies.tolist(),
            'std_accuracies': std_accuracies.tolist(),
            'split_stats': split_stats
        }

        with open(os.path.join(output_dir, f'overall_report_{self.n_splits}parts.json'), 'w') as f:
            json.dump(overall_report, f, indent=2)
        
        print("\nTop-K accuracy for each split:")
        for i, accs in enumerate(all_accuracies, 1):
            print(f"\nPart {i}:")
            for k, acc in enumerate(accs, 1):
                print(f"Top-{k}: {acc:.4f}")
                
        print("\nMean accuracy ± standard deviation:")
        for k, (mean, std) in enumerate(zip(mean_accuracies, std_accuracies), 1):
            print(f"Top-{k}: {mean:.4f} ± {std:.4f}")

        self.logger.info(f"Analysis completed. Results saved in {output_dir}")
        return overall_report

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '4'
    torch.backends.cudnn.benchmark = True
    
    results_path = "/root/autodl-tmp/VideoLLama2-audio/VideoLLaMA2/clotho_audio_descriptions/results.csv"
    clotho_path = "/root/autodl-tmp/dataset_clotho/clotho_captions_development.csv"
    cache_dir = "/root/autodl-tmp/model_cache"
    
    os.makedi