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

class MultiPartALBERTAnalyzer:
    def __init__(self, model_name: str = 'albert/albert-base-v2', batch_size: int = 32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
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
        log_file = f'albert_multipart_analysis_{self.timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def split_datasets(self, df: pd.DataFrame, video_text_map: Dict[str, str], 
                      num_parts: int = 2, random_state: int = 42) -> Tuple[List[Dict], List[pd.DataFrame]]:

        
        np.random.seed(random_state)
        
        # Create mapping from video_id to all related texts
        video_text_groups = {}
        for video_id in df['video_id'].unique():
            # Get all text pairs containing this video_id
            related_videos = [k for k in video_text_map.keys() if video_id in k]
            video_text_groups[video_id] = related_videos
        
        # Ensure each video_id has at least one matching pair
        valid_video_ids = []
        for video_id, related_videos in video_text_groups.items():
            if len(related_videos) >= 2:  # At least two related texts
                valid_video_ids.append(video_id)
        
        if len(valid_video_ids) < num_parts:
            raise ValueError(f"Only {len(valid_video_ids)} valid video IDs, cannot split into {num_parts} parts")
        
        # Randomly shuffle valid_video_ids
        shuffled_ids = np.random.permutation(valid_video_ids)
        
        # Evenly distribute video_ids to parts
        video_ids_parts = np.array_split(shuffled_ids, num_parts)
        
        # If split parts exceed num_parts (due to indivisibility), merge excess parts into the last part
        if len(video_ids_parts) > num_parts:
            video_ids_parts[num_parts-1] = np.concatenate(video_ids_parts[num_parts-1:])
            video_ids_parts = video_ids_parts[:num_parts]
        
        # Split DataFrame and video_text_map
        part_dfs = []
        part_video_text_maps = []
        
        for part_video_ids in video_ids_parts:
            # Split DataFrame
            part_df = df[df['video_id'].isin(part_video_ids)]
            part_dfs.append(part_df)
            
            # Split video_text_map
            part_video_text_map = {
                k: v for k, v in video_text_map.items() 
                if any(vid in k for vid in part_video_ids)
            }
            part_video_text_maps.append(part_video_text_map)
            
            self.logger.info(f"Split dataset - Part size: {len(part_df)} queries, {len(part_video_text_map)} videos")
        
        return part_video_text_maps, part_dfs

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], desc: str = "Encoding") -> torch.Tensor:
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        embeddings = []

        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast():
                outputs = self.model(**batch)
                embeddings.append(self.mean_pooling(outputs, batch['attention_mask']).detach())

        return torch.cat(embeddings, dim=0)
    
    def setup_analysis_dir(self, base_dir: str, part_name: str) -> str:
        analysis_dir = os.path.join(base_dir, 'multipart_analysis_results', f'{part_name}_{self.timestamp}')
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

    def plot_multipart_accuracy_curve(self, all_accuracies: List[List[float]], 
                                    all_matching_stats: List[Dict],
                                    output_path: str):
        plt.figure(figsize=(15, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_accuracies)))
        
        for i, (accuracies, stats) in enumerate(zip(all_accuracies, all_matching_stats)):
            avg_matches = stats['avg_matches_per_video']
            plt.plot(range(1, len(accuracies) + 1), accuracies, 
                    marker='o', color=colors[i], 
                    label=f'Part {i+1} (avg matches: {avg_matches:.2f})')
            
            for j, acc in enumerate(accuracies):
                plt.annotate(f'{acc:.4f}', 
                            (j + 1, acc),
                            textcoords="offset points", 
                            xytext=(0, 10 if i % 2 == 0 else -15), 
                            ha='center',
                            color=colors[i],
                            fontsize=8)
        
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Comparison Across Parts')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_part(self, data_dir: str, video_text_map: Dict[str, str], 
                    df: pd.DataFrame, part_name: str) -> Tuple[str, List[float], Dict]:
        # Analyze a single part of data
        analysis_dir = self.setup_analysis_dir(data_dir, part_name)
        
        # Analyze matching pair statistics
        matching_stats = self.analyze_matching_pairs(video_text_map, df)
        
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())

        self.logger.info(f"Building {part_name} text index...")
        text_embeddings = self.encode_texts(texts, f"Encoding {part_name} texts")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        results = []
        total_queries = len(df)
        
        for idx, row in tqdm(df.iterrows(), total=total_queries, desc=f"Processing {part_name} queries"):
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

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(analysis_dir, f'{part_name}_detailed_results.csv'), index=False)

        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy

        # Add matching pair statistics to report
        report = {
            'timestamp': self.timestamp,
            'total_queries': total_queries,
            'accuracy_stats': accuracy_stats,
            'matching_stats': matching_stats,
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': 'albert-base-v2',
            'device': str(self.device)
        }

        with open(os.path.join(analysis_dir, f'{part_name}_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        return analysis_dir, accuracies, matching_stats

    def analyze_queries(self, data_dir: str, num_parts: int = 2):
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)

        # Split dataset
        part_video_text_maps, part_dfs = self.split_datasets(df, video_text_map, num_parts)

        # Analyze each part
        all_accuracies = []
        all_matching_stats = []
        all_analysis_dirs = []
        
        for i in range(num_parts):
            analysis_dir, accuracies, matching_stats = self.analyze_part(
                data_dir, 
                part_video_text_maps[i], 
                part_dfs[i], 
                f"part{i+1}"
            )
            all_accuracies.append(accuracies)
            all_matching_stats.append(matching_stats)
            all_analysis_dirs.append(analysis_dir)

        # Create comparison results directory
        comparison_dir = os.path.join(data_dir, 'multipart_analysis_results', 
                                    f'comparison_{num_parts}parts_{self.timestamp}')
        os.makedirs(comparison_dir, exist_ok=True)

        # Plot comparison curves
        self.plot_multipart_accuracy_curve(
            all_accuracies,
            all_matching_stats,
            os.path.join(comparison_dir, 'accuracy_comparison.png')
        )

        # Generate comparison report
        comparison_report = {
            'timestamp': self.timestamp,
            'num_parts': num_parts,
            'parts_stats': [
                {
                    'total_queries': len(part_dfs[i]),
                    'total_videos': len(part_video_text_maps[i]),
                    'matching_stats': all_matching_stats[i],
                    'accuracies': all_accuracies[i]
                }
                for i in range(num_parts)
            ]
        }

        with open(os.path.join(comparison_dir, 'comparison_report.json'), 'w') as f:
            json.dump(comparison_report, f, indent=2)

        # Print comparison results
        print(f"\n{num_parts}-Part Comparison Summary:")
        for i in range(num_parts):
            stats = all_matching_stats[i]
            print(f"\nPart {i+1}:")
            print(f"  Queries: {len(part_dfs[i])}")
            print(f"  Videos: {stats['total_videos']}")
            print(f"  Avg matches per video: {stats['avg_matches_per_video']:.2f}")
            print(f"  Min/Max matches: {stats['min_matches']}/{stats['max_matches']}")
        
        print("\nAccuracy Comparison:")
        for k in range(1, 11):
            print(f"Top-{k}:")
            for i in range(num_parts):
                print(f"  Part {i+1}: {all_accuracies[i][k-1]:.4f}")

        self.logger.info(f"Comparison results saved in: {comparison_dir}")
        return comparison_dir

def main():
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    analyzer = MultiPartALBERTAnalyzer(batch_size=32)
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run analysis
    num_parts = 8  # Can be modified to 2, 4, 8 etc
    comparison_dir = analyzer.analyze_queries(data_dir, num_parts=num_parts)
    print(f"\nAnalysis results saved in: {comparison_dir}")

if __name__ == "__main__":
    main()