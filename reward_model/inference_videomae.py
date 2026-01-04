import os, glob, argparse, torch, json
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import (
    VideoMAEConfig,
    VideoMAEFeatureExtractor,
    VideoMAEForVideoClassification,
)
import imageio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def extract_sliding_windows(video_path: str, window_size: int = 8, stride: int = 1):
    reader = imageio.get_reader(video_path, 'ffmpeg')
    frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
    reader.close()
    total_frames = len(frames)
    clips = []
    for end in range(total_frames, window_size - 1, -stride):
        clip = frames[end - window_size:end]
        clips.append((clip, end - window_size, end))
    return clips


@torch.no_grad()
def run_inference_with_sliding_windows(video_paths, model, feature_extractor, batch_size=4, threshold=0.93, device="cuda"):
    model.eval()
    all_results = []
    for video_path in tqdm(video_paths, desc="Processing videos", position=dist.get_rank()):
        results = []
        clips = extract_sliding_windows(video_path)
        clip_batches = [clips[i:i+batch_size] for i in range(0, len(clips), batch_size)]
        for batch in clip_batches:
            clip_imgs = [c[0] for c in batch]
            ranges = [(c[1], c[2]) for c in batch]
            inputs = feature_extractor(clip_imgs, return_tensors="pt")["pixel_values"].to(device)
            logits = model(pixel_values=inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = [1 if p[1] >= threshold else 0 for p in probs]
            for (start, end), prob, pred in zip(ranges, probs, preds):
                results.append({
                    "video_path": video_path,
                    "start_frame": start,
                    "end_frame": end,
                    "prob": prob.tolist(),
                    "pred": pred
                })
        label = 1 if any(r["pred"] == 1 for r in results) else 0
        if label == 1:
            print('Successful prediction.')
        all_results.append({
            "video_path": video_path,
            "label": label,
            "results": results
        })
    return all_results

def evaluate_results(results):
    y_true = []
    y_pred = []
    
    for result in results:
        video_path = result['video_path']
        pred_label = result['label']
        
        # 从路径里提取真值
        true_label = video_path.split('=')[-1].split('.')[0]
        true_label = (true_label == "False")  # 转成布尔或0/1
        
        y_true.append(int(true_label))
        y_pred.append(int(pred_label))
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir", 
        default="/opt/tiger/simplevla-rl/debug/aloha_inference", 
        help="Directory containing .mp4 videos"
    )
    parser.add_argument(
        "--ckpt_path", 
        default="/opt/tiger/simplevla-rl/ckpts_videomae/best_videomae_f10.1989_th0.93.pth", 
        help="Path to trained model checkpoint (.pth)"
    )
    parser.add_argument("--threshold", type=float, default=0.06)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    ### Init Distributed
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"]) 
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    

    if rank == 0:
        print(f"[DDP] world_size={world_size}")

    video_paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    # Split workload among processes
    video_paths = video_paths[rank::world_size]

    if not video_paths:
        print(f"[Rank {rank}] No videos found.")
        return

    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base", size=args.img_size)
    cfg = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", num_frames=8, num_labels=2)
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", config=cfg).to(device)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = DDP(model, device_ids=[local_rank])

    results = run_inference_with_sliding_windows(
        video_paths, model, feature_extractor,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device
    )

    # Collect results from all ranks
    gathered_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_results, results)

    if rank == 0:
        merged_results = []
        for rank_results in gathered_results:
            merged_results.extend(rank_results)

        # Save merged results
        os.makedirs("results", exist_ok=True)
        with open(f"results/final_results_thre_{args.threshold:.2f}.json", "w") as f:
            json.dump(merged_results, f, indent=2)
        print(f"Aggregated results from {world_size} ranks saved to 'results/final_results.json'.")

if __name__ == "__main__":
    main()

# torchrun --standalone --nproc_per_node=8 inference_videomae.py --threshold 0.1
# torchrun --standalone --nproc_per_node=4 inference_videomae.py --threshold 0.2
# torchrun --standalone --nproc_per_node=4 inference_videomae.py --threshold 0.3
# torchrun --standalone --nproc_per_node=4 inference_videomae.py --threshold 0.4
# torchrun --standalone --nproc_per_node=8 inference_videomae.py --threshold 0.5
# torchrun --standalone --nproc_per_node=4 inference_videomae.py --threshold 0.6
# torchrun --standalone --nproc_per_node=4 inference_videomae.py --threshold 0.56
# torchrun --standalone --nproc_per_node=4 inference_videomae.py --threshold 0.64
# torchrun --standalone --nproc_per_node=4 inference_videomae.py --threshold 0.72