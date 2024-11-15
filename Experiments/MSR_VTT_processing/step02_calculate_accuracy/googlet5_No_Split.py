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
    video_files: List[str]
    similarity_scores: List[float]
    generated_texts: List[str]
    query_time: float
    match_positions: List[int]

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Add prefix for T5
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
    def __init__(self, model_name: str = 'google-t5/t5-base', batch_size: int = 16):
        # Initialize T5 analyzer
        # Args:
        #   model_name: T5 model name
        #   batch_size: Batch size (smaller for T5 due to model size)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model using domestic mirror
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
        last_hidden_state = model_output.last_hidden_state
        embeddings = torch.mean(last_hidden_state, dim=1)
        return embeddings

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
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
        # Find top k similar texts
        start_time = time.time()

        # Add prefix to query
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
        analysis_dir = os.path.join(base_dir, 'analysis_results', f't5_analysis_{self.timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

    def calculate_accuracy_at_k(self, results: List[Dict], k: int) -> float:
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def analyze_matching_pairs(self, video_text_map: Dict[str, str], 
                             df: pd.DataFrame) -> Dict:
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
        analysis_dir = self.setup_analysis_dir(data_dir)
        
        # Load data
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)

        # Analyze matching pair statistics
        matching_stats = self.analyze_matching_pairs(video_text_map, df)
            
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())

        # Build index
        self.logger.info("Building text index...")
        text_embeddings = self.encode_texts(texts, "Encoding texts")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        # Process queries
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

        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(analysis_dir, 'detailed_results.csv'), index=False)

        # Calculate accuracy
        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy

        # Plot accuracy curve
        self.plot_accuracy_curve(
            accuracies,
            os.path.join(analysis_dir, 'accuracy_curve.png')
        )

        # Generate analysis report
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

        # Print result summary
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
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    analyzer = T5Analyzer(batch_size=16)  # Use smaller batch size for T5 model
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run analysis
    analysis_dir = analyzer.analyze_queries(data_dir)
    print(f"\nAnalysis results saved in: {analysis_dir}")

if __name__ == "__main__":
    main()