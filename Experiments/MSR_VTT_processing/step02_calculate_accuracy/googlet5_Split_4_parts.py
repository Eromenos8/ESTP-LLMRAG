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
from sklearn.model_selection import train_test_split 

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

class MultiPartT5Analyzer:
    def __init__(self, model_name: str = 'google-t5/t5-base', batch_size: int = 16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, mirror='https://hf-mirror.com')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, mirror='https://hf-mirror.com')
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        log_file = f't5_multipart_analysis_{self.timestamp}.log'
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

    def split_datasets(self, df: pd.DataFrame, video_text_map: Dict[str, str], num_parts: int = 2, random_state: int = 42) -> Tuple[List[Dict], List[pd.DataFrame]]:
        print(f"\nOriginal Data Statistics:")
        print(f"Total captions: {len(df)}")
        print(f"Total unique video_ids: {len(df['video_id'].unique())}")
        print(f"Total video texts: {len(video_text_map)}")
        video_ids = df['video_id'].unique()
        test_size = 1.0 / num_parts
        part_dfs = []
        part_video_text_maps = []
        remaining_video_ids = video_ids

        for i in range(num_parts - 1):
            current_test_size = 1.0 / (num_parts - i)
            part_video_ids, remaining_video_ids = train_test_split(
                remaining_video_ids,
                test_size=(1 - current_test_size),
                random_state=random_state + i
            )
            part_df = df[df['video_id'].isin(part_video_ids)]
            part_video_text_map = {k: v for k, v in video_text_map.items() if any(vid in k for vid in part_video_ids)}
            part_dfs.append(part_df)
            part_video_text_maps.append(part_video_text_map)

        last_part_df = df[df['video_id'].isin(remaining_video_ids)]
        last_part_video_text_map = {k: v for k, v in video_text_map.items() if any(vid in k for vid in remaining_video_ids)}
        part_dfs.append(last_part_df)
        part_video_text_maps.append(last_part_video_text_map)
        return part_video_text_maps, part_dfs

    def find_top_k_similar(
        self,
        query: str,
        index: faiss.IndexFlatIP,
        video_text_map: Dict[str, str],
        video_files: List[str],
        video_id: str,
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
        
        top_k_videos = [video_files[idx] for idx in indices[0]]
        top_k_scores = similarities[0].tolist()
        top_k_texts = [video_text_map[video] for video in top_k_videos]
        video_key = f"video{video_id.replace('video', '')}.mp4"
        match_positions = [i for i, video in enumerate(top_k_videos) if video == video_key]
        query_time = time.time() - start_time

        return SimilarityResult(
            video_files=top_k_videos,
            similarity_scores=top_k_scores,
            generated_texts=top_k_texts,
            query_time=query_time,
            match_positions=match_positions
        )

    def setup_analysis_dir(self, base_dir: str, part_name: str) -> str:
        analysis_dir = os.path.join(base_dir, 'multipart_analysis_results', f'{part_name}_{self.timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

    def calculate_accuracy_at_k(self, results: List[Dict], k: int) -> float:
        correct = sum(1 for r in results if any(pos < k for pos in r['match_positions']))
        return correct / len(results)

    def plot_multipart_comparison(self, all_accuracies: List[List[float]], all_matching_stats: List[Dict], output_path: str):
        plt.figure(figsize=(15, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_accuracies)))
        for i, (accuracies, stats) in enumerate(zip(all_accuracies, all_matching_stats)):
            label = (f'Part {i+1} (Queries: {stats["total_queries"]}, Avg Matches: {stats["avg_matches"]:.1f})')
            plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', color=colors[i], label=label)
            for j, acc in enumerate(accuracies):
                plt.annotate(f'{acc:.4f}', (j + 1, acc), textcoords="offset points", xytext=(0, 10 if i % 2 == 0 else -15), ha='center', color=colors[i], fontsize=8)
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('T5 Model: Accuracy Comparison Across Parts')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_part(self, data_dir: str, video_text_map: Dict[str, str], 
                     df: pd.DataFrame, part_name: str) -> Tuple[str, List[float], Dict]:
        analysis_dir = self.setup_analysis_dir(data_dir, part_name)
        matching_stats = {
            'total_queries': len(df),
            'total_unique_videos': len(df['video_id'].unique()),
            'total_video_texts': len(video_text_map),
            'matches_per_video': {
                vid: len([k for k, v in video_text_map.items() 
                          if f"video{vid.replace('video', '')}.mp4" == k])
                for vid in df['video_id'].unique()
            }
        }
        matching_stats['avg_matches'] = np.mean(list(matching_stats['matches_per_video'].values()))
        matching_stats['min_matches'] = min(matching_stats['matches_per_video'].values())
        matching_stats['max_matches'] = max(matching_stats['matches_per_video'].values())
        
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())
        print(f"\nProcessing {part_name}")
        print(f"Number of texts to encode: {len(texts)}")
        self.logger.info(f"Building {part_name} text index...")
        text_embeddings = self.encode_texts(texts, f"Encoding {part_name} texts")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        results = []
        total_queries = len(df)
        for idx, row in tqdm(df.iterrows(), total=total_queries, desc=f"Processing {part_name} queries"):
            try:
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
                    'num_matches': matching_stats['matches_per_video'][row['video_id']],
                    **{f'rank{i+1}_video': result.video_files[i] for i in range(10)},
                    **{f'rank{i+1}_score': result.similarity_scores[i] for i in range(10)},
                    **{f'rank{i+1}_text': result.generated_texts[i] for i in range(10)},
                    'query_time': result.query_time
                })
                if (idx + 1) % 1000 == 0:
                    print(f"Processed {idx + 1}/{total_queries} queries")
                    print(f"Last match positions: {result.match_positions}")
                    print(f"Query: {row['video_id']} -> {[v for v in result.video_files[:3]]}")
            except Exception as e:
                self.logger.error(f"Error processing query {idx} ({row['video_id']}): {str(e)}")
                continue

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(analysis_dir, f'{part_name}_detailed_results.csv'), index=False)
        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy
            print(f"Top-{k} Accuracy: {accuracy:.4f}")
        
        report = {
            'timestamp': self.timestamp,
            'part_name': part_name,
            'matching_stats': matching_stats,
            'accuracy_stats': accuracy_stats,
            'total_processed_queries': len(results),
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': 't5-base',
            'device': str(self.device)
        }

        with open(os.path.join(analysis_dir, f'{part_name}_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        return analysis_dir, accuracies, matching_stats

    def analyze_queries(self, data_dir: str, num_parts: int = 2):
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)
        part_video_text_maps, part_dfs = self.split_datasets(df, video_text_map, num_parts)
        all_accuracies = []
        all_matching_stats = []
        all_analysis_dirs = []

        for i in range(num_parts):
            print(f"\nAnalyzing Part {i+1}...")
            try:
                analysis_dir, accuracies, matching_stats = self.analyze_part(
                    data_dir, 
                    part_video_text_maps[i], 
                    part_dfs[i], 
                    f"part{i+1}"
                )
                all_accuracies.append(accuracies)
                all_matching_stats.append(matching_stats)
                all_analysis_dirs.append(analysis_dir)
            except Exception as e:
                self.logger.error(f"Error analyzing part {i+1}: {str(e)}")
                continue

        comparison_dir = os.path.join(data_dir, 'T5_4parts_analysis_results', 
                                    f't5_comparison_{num_parts}parts_{self.timestamp}')
        os.makedirs(comparison_dir, exist_ok=True)

        if all_accuracies:
            self.plot_multipart_comparison(
                all_accuracies,
                all_matching_stats,
                os.path.join(comparison_dir, 'accuracy_comparison.png')
            )
            comparison_report = {
                'timestamp': self.timestamp,
                'model': 't5-base',
                'num_parts': num_parts,
                'parts_stats': [
                    {
                        'part_number': i + 1,
                        'total_queries': stats['total_queries'],
                        'total_unique_videos': stats['total_unique_videos'],
                        'total_video_texts': stats['total_video_texts'],
                        'avg_matches': stats['avg_matches'],
                        'min_matches': stats.get('min_matches', 0),
                        'max_matches': stats.get('max_matches', 0),
                        'accuracies': accs
                    }
                    for i, (stats, accs) in enumerate(zip(all_matching_stats, all_accuracies))
                ]
            }

            with open(os.path.join(comparison_dir, 'comparison_report.json'), 'w') as f:
                json.dump(comparison_report, f, indent=2)

            print(f"\nT5 {num_parts}-Part Comparison Summary:")
            for i, stats in enumerate(all_matching_stats):
                print(f"\nPart {i+1}:")
                print(f"  Queries: {stats['total_queries']}")
                print(f"  Unique Videos: {stats['total_unique_videos']}")
                print(f"  Video Texts: {stats['total_video_texts']}")
                print(f"  Avg matches per video: {stats['avg_matches']:.2f}")
                print(f"  Min/Max matches: {stats['min_matches']}/{stats['max_matches']}")
            print("\nAccuracy Comparison:")
            for k in range(1, 11):
                print(f"Top-{k}:")
                for i, accs in enumerate(all_accuracies):
                    print(f"  Part {i+1}: {accs[k-1]:.4f}")

        self.logger.info(f"Comparison results saved in: {comparison_dir}")
        return comparison_dir

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '4'
    torch.backends.cudnn.benchmark = True
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    np.random.seed(42)
    torch.manual_seed(42)
    num_parts = 4
    analyzer = MultiPartT5Analyzer(batch_size=24)
    comparison_dir = analyzer.analyze_queries(data_dir, num_parts=num_parts)
    print(f"\nAnalysis results saved in: {comparison_dir}")

if __name__ == "__main__":
    main()