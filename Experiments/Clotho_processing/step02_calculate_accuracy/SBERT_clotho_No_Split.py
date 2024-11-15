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
    """Data class to store similarity calculation results."""
    video_files: List[str]
    similarity_scores: List[float]
    descriptions: List[str]
    query_time: float
    match_positions: List[int]

class TextDataset(Dataset):
    """Dataset class for text batch processing."""
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class AudioCaptionMatcher:
    """Audio caption matcher class."""
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',  # Default English model
                 batch_size: int = 128,
                 cache_dir: str = None):
        """Initialize the matcher."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set cache directory
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using cache directory: {cache_dir}")
        
        # Set model download mirror
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # Try to load the model
        try:
            print("Loading the model...")
            self.model = self._load_model_with_retry(model_name, max_retries=3)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            print("Trying a backup model...")
            try:
                backup_model = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # Backup English model
                self.model = self._load_model_with_retry(backup_model, max_retries=3)
                print("Backup model loaded successfully!")
            except Exception as e:
                print(f"Backup model loading also failed: {str(e)}")
                sys.exit(1)
                
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def preprocess_text(self, text: str) -> str:
        """Preprocess the input text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and escape sequences
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = text.replace('.wav', '')
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra spaces again
        text = ' '.join(text.split())
        
        return text

    def setup_logging(self):
        """Configure logging."""
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

    def _load_model_with_retry(self, model_name: str, max_retries: int = 3) -> SentenceTransformer:
        """Load the model with retry mechanism."""
        last_exception = None
        for attempt in range(max_retries):
            try:
                return SentenceTransformer(model_name)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Loading failed, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(5)
        raise last_exception

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        """Batch encode texts."""
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
        filenames: List[str],
        original_filename: str,
        k: int = 10
    ) -> SimilarityResult:
        """Find top-K similar texts."""
        start_time = time.time()

        # Preprocess query text
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
        
        match_positions = [i for i, fname in enumerate(top_k_files) if original_filename in fname]
        
        query_time = time.time() - start_time

        return SimilarityResult(
            video_files=top_k_files,
            similarity_scores=top_k_scores,
            descriptions=top_k_descs,
            query_time=query_time,
            match_positions=match_positions
        )

    def calculate_accuracy_at_k(self, results: List[Dict], k: int) -> float:
        """Calculate Top-K accuracy."""
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def plot_accuracy_curve(self, accuracies: List[float], output_path: str):
        """Plot accuracy curve."""
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

    def match_captions(self, results_path: str, clotho_path: str, output_dir: str):
        """Process dataset and perform matching."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        try:
            results_df = pd.read_csv(results_path)
            clotho_df = pd.read_csv(clotho_path)
        except Exception as e:
            self.logger.error(f"Failed to read data files: {str(e)}")
            return

        # Preprocess all descriptions
        descriptions = [self.preprocess_text(desc) for desc in results_df['description'].tolist()]
        filenames = results_df['filename'].tolist()

        # Build index
        self.logger.info("Building text index...")
        text_embeddings = self.encode_texts(descriptions, "Encoding audio descriptions")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        # Process each caption
        results = []
        
        total_files = len(clotho_df)
        for idx, row in tqdm(clotho_df.iterrows(), total=total_files, desc="Processing dataset"):
            file_name = row['file_name']
            
            # Process each caption
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
                    
                    # Record results
                    results.append({
                        'original_file': file_name,
                        'caption': row[caption_key],
                        'caption_number': i,
                        'match_positions': result.match_positions,
                        **{f'rank{j+1}_file': result.video_files[j] for j in range(10)},
                        **{f'rank{j+1}_score': result.similarity_scores[j] for j in range(10)},
                        **{f'rank{j+1}_desc': result.descriptions[j] for j in range(10)},
                        'query_time': result.query_time
                    })

            if (idx + 1) % 100 == 0:
                self.logger.info(f"Processed {idx + 1}/{total_files} files")

        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)

        # Calculate accuracy for different K values
        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy
            self.logger.info(f"Top-{k} Accuracy: {accuracy:.4f}")

        # Save accuracy stats
        with open(os.path.join(output_dir, 'accuracy_stats.json'), 'w') as f:
            json.dump(accuracy_stats, f, indent=2)

        # Plot accuracy curve
        self.plot_accuracy_curve(
            accuracies,
            os.path.join(output_dir, 'accuracy_curve.png')
        )

        # Generate analysis report
        report = {
            'timestamp': self.timestamp,
            'total_captions': len(results),
            'total_files': total_files,
            'accuracy_stats': accuracy_stats,
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': str(self.model),
            'device': str(self.device)
        }

        # Save analysis report
        with open(os.path.join(output_dir, 'analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Analysis completed. Results saved in {output_dir}")
        print(f"\nAccuracy Summary:")
        for k in range(1, 11):
            print(f"Top-{k}: {accuracy_stats[f'top{k}_accuracy']:.4f}")

def main():
    """Main function."""
    # Set paths
    results_path = "/root/autodl-tmp/VideoLLama2-audio/VideoLLaMA2/audio_descriptions/results.csv"
    clotho_path = "/root/autodl-tmp/dataset_clotho/clotho_captions_development.csv"
    output_dir = "/root/autodl-tmp/dataset_clotho/SBERT_matched_result"
    cache_dir = "/root/autodl-tmp/model_cache"
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create matcher instance and perform matching
    matcher = AudioCaptionMatcher(
        model_name='sentence-transformers/all-MiniLM-L6-v2',  # Default English model
        batch_size=128,
        cache_dir=cache_dir
    )
    matcher.match_captions(results_path, clotho_path, output_dir)

if __name__ == "__main__":
    main()