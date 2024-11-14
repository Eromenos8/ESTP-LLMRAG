import numpy as np
import torch
from numpy import ndarray
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import cv2
import torch
from transformers import CLIPVisionModelWithProjection
from pathlib import Path
import faiss
from faiss import normalize_L2


class VideoRetriever:
    VEC_LENGTH = 512

    def __init__(self, video_folder_path: Path):
        self.video_folder_path = video_folder_path
        self.text_model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        self.tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        self.faiss_index = faiss.IndexFlatIP(VideoRetriever.VEC_LENGTH)
        self.video_paths = list(video_folder_path.glob("*.mp4"))

    def get_text_embedding(self, query_txt: str) -> ndarray:
        inputs = self.tokenizer(text=query_txt, return_tensors="pt")
        outputs = self.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        final_output = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
        final_output = final_output.cpu().detach().numpy()
        final_output = final_output.reshape(1, -1)
        return final_output

    @staticmethod
    def _video2image(video_path, frame_rate=1.0, size=224):
        def preprocess(_size, n_px):
            return Compose([
                Resize(_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(_size),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])(n_px)

        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps < 1:
            images = np.zeros([3, size, size], dtype=np.float32)
            print("ERROR: problem reading video file: ", video_path)
        else:
            total_duration = (frameCount + fps - 1) // fps
            start_sec, end_sec = 0, total_duration
            interval = fps / frame_rate
            frames_idx = np.floor(np.arange(start_sec * fps, end_sec * fps, interval))
            images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)

            for i, idx in enumerate(frames_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_frame = i
                images[i, :, :, :] = preprocess(size, Image.fromarray(frame).convert("RGB"))

            images = images[:last_frame + 1]

        cap.release()
        video_frames = torch.tensor(images)
        return video_frames

    def get_vision_embedding(self, video_path: str) -> ndarray:
        video = VideoRetriever._video2image(video_path)
        model = self.vision_model.eval()
        visual_output = model(video)
        visual_output = visual_output["image_embeds"]
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = torch.mean(visual_output, dim=0)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = visual_output.numpy(force=True)
        visual_output = visual_output.reshape(1, -1)
        return visual_output

    def build_vec_db(self):
        for video_path in self.video_paths:
            video_path = video_path.resolve()
            vision_embedding = self.get_vision_embedding(video_path)
            self.faiss_index.add(vision_embedding)

    def find_match_videos(self, query_txt: str, top_k: int):
        text_embedding = self.get_text_embedding(query_txt)
        distances, indices = self.faiss_index.search(text_embedding, top_k)
        distances = distances[0]
        match_videos = []
        for index in list(indices[0]):
            if index >= 0:
                match_videos.append(self.video_paths[index])
        return match_videos, distances

    def reset(self, video_folder_path: Path):
        self.faiss_index.reset()
        self.video_paths = list(video_folder_path.glob("*.mp4"))


if __name__ == "__main__":
    retriever = VideoRetriever(video_folder_path=Path("../videos/"))
    retriever.build_vec_db()
    lst, sim = retriever.find_match_videos("A man is sitting before a computer", 10)
    print(lst, sim)
