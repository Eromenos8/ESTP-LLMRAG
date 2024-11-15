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
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class SplitAudioCaptionMatcher:
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 batch_size: int = 128,
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
        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        try:
            print("Loading the model...")
            self.model = self._load_model_with_retry(model_name)
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
        log_file = os.path.join(log_dir, f'sbert_matching_split_{self.timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def prepare_and_split_data(self, clotho_df: pd.DataFrame, results_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        print("\nData preprocessing:")
        print(f"Original Clotho dataset size: {len(clotho_df)}")
        print(f"Original Results dataset size: {len(results_df)}")

        # Regex for timestamp and prefix removal
        timestamp_pattern = r'_\d+_\d+'
        prefix_pattern = r'^[-_0-9]+'

        # Clean filenames in Clotho dataset
        def clean_clotho_filename(filename):
            try:
                filename = filename.replace('.wav', '')
                filename = filename.replace('&quot;', '')
                filename = filename.replace('&amp;', '')
                filename = filename.replace('_', ' ')
                filename = filename.lower().strip()
                filename = re.sub(r'[^a-z\s]', '', filename)
                filename = ' '.join(filename.split())
                return filename
            except:
                return filename

        # Clean filenames in Results dataset
        def clean_results_filename(filename):
            try:
                filename = filename.replace('.wav', '')
                filename = re.sub(timestamp_pattern, '', filename)
                filename = re.sub(prefix_pattern, '', filename)
                filename = filename.replace('_', ' ')
                filename = filename.lower().strip()
                filename = re.sub(r'[^a-z\s]', '', filename)
                filename = ' '.join(filename.split())
                return filename
            except:
                return filename

        # Apply cleaning functions and add columns for cleaned filenames
        results_df['original_filename'] = results_df['filename']
        clotho_df['original_filename'] = clotho_df['file_name']
        
        results_df['clean_filename'] = results_df['filename'].apply(clean_results_filename)
        clotho_df['clean_filename'] = clotho_df['file_name'].apply(clean_clotho_filename)

        # Print some samples for validation
        print("\nSample cleaned filenames:")
        print("Clotho samples:")
        print(pd.DataFrame({
            'original': clotho_df['file_name'].head(),
            'cleaned': clotho_df['clean_filename'].head()
        }))
        print("\nResults samples:")
        print(pd.DataFrame({
            'original': results_df['filename'].head(),
            'cleaned': results_df['clean_filename'].head()
        }))

        # Find common filenames
        common_files = set(results_df['clean_filename']) & set(clotho_df['clean_filename'])
        print(f"\nNumber of common files: {len(common_files)}")
        if len(common_files) > 0:
            print("\nExamples of matches:")
            sample_matches = list(common_files)[:5]
            for clean_name in sample_matches:
                clotho_original = clotho_df[clotho_df['clean_filename'] == clean_name]['file_name'].iloc[0]
                results_original = results_df[results_df['clean_filename'] == clean_name]['filename'].iloc[0]
                print(f"Clean name: {clean_name}")
                print(f"Clotho: {clotho_original}")
                print(f"Results: {results_original}")
                print()

        if len(common_files) == 0:
            raise ValueError("No matching files found. Check filename formats.")

        # Filter datasets
        clotho_filtered = clotho_df[clotho_df['clean_filename'].isin(common_files)].copy()
        results_filtered = results_df[results_df['clean_filename'].isin(common_files)].copy()

        print(f"Filtered Clotho dataset size: {len(clotho_filtered)}")
        print(f"Filtered Results dataset size: {len(results_filtered)}")

        # Shuffle and split Clotho data
        clotho_filtered = clotho_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
        splits = np.array_split(clotho_filtered, self.n_splits)

        # Validate splits
        for i, split in enumerate(splits, 1):
            split_files = set(split['clean_filename'])
            matching_results = results_filtered[results_filtered['clean_filename'].isin(split_files)]
            print(f"\nSplit {i} stats:")
            print(f"Clotho samples: {len(split)}")
            print(f"Matching Results samples: {len(matching_results)}")
            print(f"Unique files: {len(split_files)}")

        return splits, results_filtered

    def _load_model_with_retry(self, model_name: str, max_retries: int = 3) -> SentenceTransformer:
        for attempt in range(max_retries):
            try:
                return SentenceTransformer(model_name)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Loading failed, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(5)

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = ' '.join(text.split())
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text

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

    def find_top_k_similar(self, query: str, index: faiss.IndexFlatIP,
                          descriptions: List[str], filenames: List[str],
                          original_filename: str, k: int = 10) -> SimilarityResult:
        start_time = time.time()

        query = self.preprocess_text(query)
        with autocast():
            query_embedding = self.model.encode(
                [query],
                convert_to_tensor=True,
                show_progress_bar=False
            )

        query_embedding_np = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding_np)

        similarities, indices = index.search(query_embedding_np, k)
        
        top_k_files = [filenames[idx] for idx in indices[0]]
        top_k_scores = similarities[0].tolist()
        top_k_descs = [descriptions[idx] for idx in indices[0]]
        
        # Remove ".wav" for comparison
        original_filename = re.sub(r'[^\w\s-]', '', original_filename.replace('.wav', '').lower())
        match_positions = [i for i, fname in enumerate(top_k_files) 
                         if original_filename in re.sub(r'[^\w\s-]', '', fname.replace('.wav', '').lower())]
        
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
        
        # Get files in the current split
        split_files = set(clotho_split['clean_filename'])
        matching_results = results_df[results_df['clean_filename'].isin(split_files)]

        # Preprocess descriptions
        descriptions = [self.preprocess_text(desc) for desc in matching_results['description'].tolist()]
        filenames = matching_results['filename'].tolist()

        if not descriptions:
            raise ValueError(f"Split {split_id} has no matching descriptions.")

        # Build the index
        text_embeddings = self.encode_texts(descriptions, f"Split {split_id}: Encoding descriptions")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        # Process queries
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
                        'match_positions': result.match_positions
                    })

        # Calculate accuracies
        accuracies = []
        for k in range(1, 11):
            accuracy = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
            accuracy = accuracy / len(results) if results else 0
            accuracies.append(accuracy)
            
        return accuracies
    
    def plot_comparison(self, all_accuracies: List[List[float]], output_dir: str):
        try:
            # Create figure and subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
            
            # Set colors and markers
            colors = plt.cm.viridis(np.linspace(0, 1, self.n_splits))
            markers = ['o', 's', '^', 'D']
            
            # First subplot: accuracy curves for all splits
            for i, accuracies in enumerate(all_accuracies):
                ax1.plot(range(1, 11), accuracies, 
                        label=f'Split {i+1}',
                        color=colors[i],
                        marker=markers[i % len(markers)],
                        linewidth=2,
                        markersize=8)
                
                # Add value labels
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
            
            # Second subplot: mean and std
            mean_accuracies = np.mean(all_accuracies, axis=0)
            std_accuracies = np.std(all_accuracies, axis=0)
            
            ax2.plot(range(1, 11), mean_accuracies, 
                    label='Mean Accuracy',
                    color='blue',
                    marker='o',
                    linewidth=2,
                    markersize=8)
            
            # Add std shading
            ax2.fill_between(range(1, 11), 
                            mean_accuracies - std_accuracies,
                            mean_accuracies + std_accuracies,
                            alpha=0.2,
                            color='blue',
                            label='±1 STD')
            
            # Add value labels
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
            
            # Adjust layout and save plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), 
                       dpi=300, 
                       bbox_inches='tight')
            plt.close()

            return mean_accuracies, std_accuracies
            
        except Exception as e:
            print(f"Error while plotting: {str(e)}")
            # Return results even if plotting fails
            mean_accuracies = np.mean(all_accuracies, axis=0)
            std_accuracies = np.std(all_accuracies, axis=0)
            return mean_accuracies, std_accuracies

    def match_captions(self, results_path: str, clotho_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        # Load datasets
        try:
            results_df = pd.read_csv(results_path)
            clotho_df = pd.read_csv(clotho_path)
        except Exception as e:
            self.logger.error(f"Failed to read data files: {str(e)}")
            raise

        # Prepare and split data
        clotho_splits, results_filtered = self.prepare_and_split_data(clotho_df, results_df)
        
        # Store all results
        all_accuracies = []
        split_stats = []
        
        # Process each split
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
        
        if not all_accuracies:
            raise ValueError("All splits failed to process.")

        # Calculate and plot overall results
        mean_accuracies, std_accuracies = self.plot_comparison(all_accuracies, output_dir)
        
        # Add mean statistics
        split_stats.append({
            'split_id': 'average',
            'accuracies': {
                f'top_{k+1}': {
                    'mean': float(mean_accuracies[k]),
                    'std': float(std_accuracies[k])
                } for k in range(10)
            }
        })
        
        # Save statistics
        with open(os.path.join(output_dir, 'split_statistics.json'), 'w') as f:
            json.dump(split_stats, f, indent=2)
        
        # Save experiment info
        experiment_info = {
            'timestamp': self.timestamp,
            'model_name': str(self.model),
            'device': str(self.device),
            'batch_size': self.batch_size,
            'n_splits': self.n_splits,
            'total_samples': sum(len(split) for split in clotho_splits),
            'results_per_split': [len(split) for split in clotho_splits],
            'mean_accuracies': mean_accuracies.tolist(),
            'std_accuracies': std_accuracies.tolist()
        }
        
        # Save experiment info
        with open(os.path.join(output_dir, 'experiment_info.json'), 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        # Print results
        print("\nTop-K accuracies for each split:")
        for i, accs in enumerate(all_accuracies, 1):
            print(f"\nSplit {i}:")
            for k, acc in enumerate(accs, 1):
                print(f"Top-{k}: {acc:.4f}")
                
        print("\nMean ± Std:")
        for k, (mean, std) in enumerate(zip(mean_accuracies, std_accuracies), 1):
            print(f"Top-{k}: {mean:.4f} ± {std:.4f}")

        self.logger.info(f"Analysis completed. Results saved in {output_dir}")
        return split_stats

def main():
    # Set paths
    results_path = "/root/autodl-tmp/VideoLLama2-audio/VideoLLaMA2/clotho_audio_descriptions/results.csv"
    clotho_path = "/root/autodl-tmp/dataset_clotho/clotho_captions_development.csv"
    output_dir = "/root/autodl-tmp/dataset_clotho/SBERT_matched_result_splits"
    cache_dir = "/root/autodl-tmp/model_cache"
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Create matcher instance and execute matching
        matcher = SplitAudioCaptionMatcher(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            batch_size=128,
            cache_dir=cache_dir,
            n_splits=2  # Divide into 2 splits
        )
        
        # Execute matching
        matcher.match_captions(results_path, clotho_path, output_dir)
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()