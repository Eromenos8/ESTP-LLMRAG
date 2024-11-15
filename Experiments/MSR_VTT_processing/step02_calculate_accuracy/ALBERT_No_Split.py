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
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

class ALBERTAnalyzer:
    def __init__(self, model_name: str = 'albert/albert-base-v2', batch_size: int = 32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model using domestic mirror
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            mirror='https://hf-mirror.com'
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            mirror='https://hf-mirror.com'
        )
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        # Set up logging configuration
        log_file = f'albert_analysis_{self.timestamp}.log'
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
        # Perform mean pooling on model output
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        # Batch encode texts
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        embeddings = []

        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast():
                outputs = self.model(**batch)
                embeddings.append(self.mean_pooling(outputs, batch['attention_mask']).detach())

        return torch.cat(embeddings, dim=0)

    def setup_analysis_dir(self, base_dir: str) -> str:
        # Create analysis results directory
        analysis_dir = os.path.join(base_dir, 'analysis_results', f'albert_analysis_{self.timestamp}')
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
        # Find top k similar texts
        start_time = time.time()

        # Encode query text
        query_inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with autocast():
            query_outputs = self.model(**query_inputs)
            query_embedding = self.mean_pooling(
                query_outputs,
                query_inputs['attention_mask']
            ).detach()

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

    def calculate_accuracy_at_k(self, results: List[Dict], k: int) -> float:
        # Calculate accuracy for top k results
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def plot_accuracy_curve(self, accuracies: List[float], output_path: str):
        # Plot accuracy curve
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-', marker='o')
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('ALBERT Model Accuracy vs Top-K Results')
        plt.grid(True)
        
        for i, accuracy in enumerate(accuracies):
            plt.annotate(f'{accuracy:.4f}', 
                        (i + 1, accuracy),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.savefig(output_path)
        plt.close()

    def analyze_queries(self, data_dir: str):
        analysis_dir = self.setup_analysis_dir(data_dir)
        
        # Load data
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)
            
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())

        # Build text index
        self.logger.info("Building text index...")
        text_embeddings = self.encode_texts(texts, "Encoding generated texts")
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
                **{f'rank{i+1}_video': result.video_files[i] for i in range(10)},
                **{f'rank{i+1}_score': result.similarity_scores[i] for i in range(10)},
                **{f'rank{i+1}_text': result.generated_texts[i] for i in range(10)},
                'query_time': result.query_time
            })

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_queries} queries")

        # Save results and generate report
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(analysis_dir, 'detailed_results.csv'), index=False)

        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy
            print(f"Top-{k} Accuracy: {accuracy:.4f}")

        with open(os.path.join(analysis_dir, 'accuracy_stats.json'), 'w') as f:
            json.dump(accuracy_stats, f, indent=2)

        self.plot_accuracy_curve(
            accuracies,
            os.path.join(analysis_dir, 'accuracy_curve.png')
        )

        report = {
            'timestamp': self.timestamp,
            'total_queries': total_queries,
            'accuracy_stats': accuracy_stats,
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': 'albert-base-v2',
            'device': str(self.device)
        }

        with open(os.path.join(analysis_dir, 'analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Analysis completed. Results saved in {analysis_dir}")
        print(f"\nAccuracy Summary:")
        for k in range(1, 11):
            print(f"Top-{k}: {accuracy_stats[f'top{k}_accuracy']:.4f}")
        
        return analysis_dir

def main():
    # Main function
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    analyzer = ALBERTAnalyzer(batch_size=32)  # ALBERT model requires smaller batch size
    analysis_dir = analyzer.analyze_queries(data_dir)
    print(f"\nAnalysis results saved in: {analysis_dir}")

if __name__ == "__main__":
    main()