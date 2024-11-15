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
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class SplitSBERTAnalyzer:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', batch_size: int = 128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

    def setup_logging(self):
        log_file = f'split_sbert_analysis_{self.timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def split_datasets_eight_parts(self, df: pd.DataFrame, video_text_map: Dict[str, str], random_state: int = 42) -> Tuple[List[Dict], List[pd.DataFrame]]:
        # Get all unique video IDs
        video_ids = df['video_id'].unique()
        
        # First split into two halves
        video_ids_half1, video_ids_half2 = train_test_split(
            video_ids, 
            test_size=0.5, 
            random_state=random_state
        )
        
        # Split each half into quarters
        video_ids_quarter1, video_ids_quarter2 = train_test_split(
            video_ids_half1,
            test_size=0.5,
            random_state=random_state
        )
        
        video_ids_quarter3, video_ids_quarter4 = train_test_split(
            video_ids_half2,
            test_size=0.5,
            random_state=random_state
        )
        
        # Split each quarter into halves
        video_ids_part1, video_ids_part2 = train_test_split(
            video_ids_quarter1,
            test_size=0.5,
            random_state=random_state
        )
        
        video_ids_part3, video_ids_part4 = train_test_split(
            video_ids_quarter2,
            test_size=0.5,
            random_state=random_state
        )
        
        video_ids_part5, video_ids_part6 = train_test_split(
            video_ids_quarter3,
            test_size=0.5,
            random_state=random_state
        )
        
        video_ids_part7, video_ids_part8 = train_test_split(
            video_ids_quarter4,
            test_size=0.5,
            random_state=random_state
        )
        
        # Create dataframes and mappings for each part
        part_dfs = []
        part_video_text_maps = []
        
        for i, part_ids in enumerate([video_ids_part1, video_ids_part2, video_ids_part3, 
                                    video_ids_part4, video_ids_part5, video_ids_part6, 
                                    video_ids_part7, video_ids_part8], 1):
            part_df = df[df['video_id'].isin(part_ids)]
            part_dfs.append(part_df)
            
            part_map = {
                k: v for k, v in video_text_map.items() 
                if any(vid in k for vid in part_ids)
            }
            part_video_text_maps.append(part_map)
            
            self.logger.info(f"Split dataset - Part {i}: {len(part_df)} queries, {len(part_map)} videos")
        
        return part_video_text_maps, part_dfs

    def setup_analysis_dir(self, base_dir: str, part_name: str) -> str:
        analysis_dir = os.path.join(base_dir, 'split_analysis_results', f'{part_name}_{self.timestamp}')
        os.makedirs(analysis_dir, exist_ok=True)
        return analysis_dir

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
                          video_text_map: Dict[str, str], video_files: List[str], 
                          video_id: str, k: int = 10) -> SimilarityResult:
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
    
    def plot_eight_part_comparison_curve(self, accuracies_list: List[List[float]], output_path: str):
        """
        Plot accuracy comparison for eight parts
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p']
        
        # Plot first 4 parts
        for i in range(4):
            ax1.plot(range(1, len(accuracies_list[i]) + 1), accuracies_list[i], 
                    color=colors[i], marker=markers[i], 
                    label=f'Part {i+1}')
            for j, acc in enumerate(accuracies_list[i]):
                ax1.annotate(f'{acc:.4f}', (j + 1, acc), textcoords="offset points", xytext=(0, 5 + i*(-15)), ha='center', color=colors[i])
        
        ax1.set_xlabel('Top-K')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison: Parts 1-4')
        ax1.grid(True)
        ax1.legend()
        
        # Plot last 4 parts
        for i in range(4, 8):
            ax2.plot(range(1, len(accuracies_list[i]) + 1), accuracies_list[i], 
                    color=colors[i], marker=markers[i], 
                    label=f'Part {i+1}')
            for j, acc in enumerate(accuracies_list[i]):
                ax2.annotate(f'{acc:.4f}', (j + 1, acc), textcoords="offset points", xytext=(0, 5 + (i-4)*(-15)), ha='center', color=colors[i])
        
        ax2.set_xlabel('Top-K')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Comparison: Parts 5-8')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Combined plot for all parts
        plt.figure(figsize=(20, 12))
        for i in range(8):
            plt.plot(range(1, len(accuracies_list[i]) + 1), accuracies_list[i], 
                    color=colors[i], marker=markers[i], 
                    label=f'Part {i+1}')
        
        plt.xlabel('Top-K')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: All Parts')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_combined.png'))
        plt.close()

    def analyze_part(self, data_dir: str, video_text_map: Dict[str, str], df: pd.DataFrame, part_name: str) -> Tuple[str, List[float]]:
        analysis_dir = self.setup_analysis_dir(data_dir, part_name)
        
        texts = list(video_text_map.values())
        video_files = list(video_text_map.keys())

        # Build FAISS index
        self.logger.info(f"Building index for {part_name}...")
        text_embeddings = self.encode_texts(texts, f"Encoding {part_name} texts")
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        embeddings_np = text_embeddings.cpu().numpy()
        faiss.normalize_L2(embeddings_np)
        index.add(embeddings_np)
        
        results = []
        total_queries = len(df)
        
        # Process queries
        for idx, row in tqdm(df.iterrows(), total=total_queries, desc=f"Processing {part_name} queries"):
            result = self.find_top_k_similar(row['caption'], index, video_text_map, video_files, row['video_id'])
            results.append({
                'video_id': row['video_id'],
                'caption': row['caption'],
                'match_positions': result.match_positions,
                **{f'rank{i+1}_video': result.video_files[i] for i in range(10)},
                **{f'rank{i+1}_score': result.similarity_scores[i] for i in range(10)},
                **{f'rank{i+1}_text': result.generated_texts[i] for i in range(10)},
                'query_time': result.query_time
            })

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(analysis_dir, f'{part_name}_detailed_results.csv'), index=False)

        # Calculate accuracy
        accuracies = []
        accuracy_stats = {}
        for k in range(1, 11):
            accuracy = self.calculate_accuracy_at_k(results, k)
            accuracies.append(accuracy)
            accuracy_stats[f'top{k}_accuracy'] = accuracy

        with open(os.path.join(analysis_dir, f'{part_name}_accuracy_stats.json'), 'w') as f:
            json.dump(accuracy_stats, f, indent=2)

        report = {
            'timestamp': self.timestamp,
            'total_queries': total_queries,
            'accuracy_stats': accuracy_stats,
            'average_query_time': sum(r['query_time'] for r in results) / len(results),
            'model_name': str(self.model),
            'device': str(self.device)
        }

        with open(os.path.join(analysis_dir, f'{part_name}_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        return analysis_dir, accuracies

    def analyze_queries(self, data_dir: str):
        df = pd.read_csv(os.path.join(data_dir, 'initial_processed_dataset.csv'))
        with open(os.path.join(data_dir, 'parsed_video_text_map.json'), 'r') as f:
            video_text_map = json.load(f)

        # Split dataset into eight parts
        part_video_text_maps, part_dfs = self.split_datasets_eight_parts(df, video_text_map)

        # Analyze each part
        analysis_results = []
        accuracies_list = []
        
        for i, (part_map, part_df) in enumerate(zip(part_video_text_maps, part_dfs), 1):
            analysis_dir, accuracies = self.analyze_part(data_dir, part_map, part_df, f"part{i}")
            analysis_results.append((analysis_dir, accuracies))
            accuracies_list.append(accuracies)

        comparison_dir = os.path.join(data_dir, 'split_analysis_results', f'comparison_{self.timestamp}')
        os.makedirs(comparison_dir, exist_ok=True)

        # Plot comparison
        self.plot_eight_part_comparison_curve(accuracies_list, os.path.join(comparison_dir, 'accuracy_comparison.png'))

        # Calculate accuracy differences
        accuracy_differences = {}
        for i in range(8):
            for j in range(i + 1, 8):
                key = f'part{i+1}_vs_part{j+1}'
                accuracy_differences[key] = [accuracies_list[i][k] - accuracies_list[j][k] for k in range(len(accuracies_list[i]))]

        # Save comparison report
        stats = {
            'mean_accuracies': [np.mean(acc) for acc in accuracies_list],
            'std_accuracies': [np.std(acc) for acc in accuracies_list],
            'max_accuracies': [np.max(acc) for acc in accuracies_list],
            'min_accuracies': [np.min(acc) for acc in accuracies_list]
        }

        comparison_report = {
            'timestamp': self.timestamp,
            'parts_stats': [
                {
                    'part_number': i+1,
                    'total_queries': len(part_df),
                    'total_videos': len(part_map),
                    'accuracies': accs.tolist() if isinstance(accs, np.ndarray) else accs,
                    'mean_accuracy': stats['mean_accuracies'][i],
                    'std_accuracy': stats['std_accuracies'][i],
                    'max_accuracy': stats['max_accuracies'][i],
                    'min_accuracy': stats['min_accuracies'][i]
                }
                for i, (part_df, part_map, accs) in enumerate(zip(part_dfs, part_video_text_maps, accuracies_list))
            ],
            'accuracy_differences': accuracy_differences,
            'overall_stats': {
                'mean_accuracy_across_parts': np.mean(stats['mean_accuracies']),
                'std_accuracy_across_parts': np.std(stats['mean_accuracies']),
                'max_accuracy_diff': max(max(diffs) for diffs in accuracy_differences.values()),
                'min_accuracy_diff': min(min(diffs) for diffs in accuracy_differences.values())
            }
        }

        with open(os.path.join(comparison_dir, 'comparison_report.json'), 'w') as f:
            json.dump(comparison_report, f, indent=2)

        print("\nComparison Summary:")
        for i, (part_df, part_map) in enumerate(zip(part_dfs, part_video_text_maps), 1):
            print(f"Part {i}: {len(part_df)} queries, {len(part_map)} videos")

        print("\nAccuracy Comparison:")
        for k in range(1, 11):
            print(f"\nTop-{k}:")
            for i, accuracies in enumerate(accuracies_list, 1):
                print(f"  Part {i}: {accuracies[k-1]:.4f}")

        print("\nOverall Statistics:")
        print(f"Mean Accuracy Across Parts: {comparison_report['overall_stats']['mean_accuracy_across_parts']:.4f}")
        print(f"Std Accuracy Across Parts: {comparison_report['overall_stats']['std_accuracy_across_parts']:.4f}")
        print(f"Max Accuracy Difference: {comparison_report['overall_stats']['max_accuracy_diff']:.4f}")
        print(f"Min Accuracy Difference: {comparison_report['overall_stats']['min_accuracy_diff']:.4f}")

        self.logger.info(f"Comparison results saved in: {comparison_dir}")
        return comparison_dir

def main():
    data_dir = "/root/autodl-tmp/dataprocessing/prepared_data"
    analyzer = SplitSBERTAnalyzer(batch_size=128)
    
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run analysis
    comparison_dir = analyzer.analyze_queries(data_dir)
    print(f"\nAnalysis results saved in: {comparison_dir}")

if __name__ == "__main__":
    main()
