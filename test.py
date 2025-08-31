# test
import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloader import load_data_paths, MedicalImageDataset3D, Compose, CTNormalize, ToTensor
from model import UNet3D
from metric import calculate_metrics
from utils import ensure_dir

def sliding_window_inference(model, image, patch_size, overlap, device, num_classes):
    model.eval()
    B, C, D, H, W = image.shape
    pd, ph, pw = patch_size
    
    stride_d = int(pd * (1 - overlap)) if overlap < 1 and pd < D else pd
    stride_h = int(ph * (1 - overlap)) if overlap < 1 and ph < H else ph
    stride_w = int(pw * (1 - overlap)) if overlap < 1 and pw < W else pw
    
    output_logits = torch.zeros((B, num_classes, D, H, W), device=device)
    count_map = torch.zeros((B, 1, D, H, W), device=device)

    for d in range(0, D - pd + 1, stride_d):
        for h in range(0, H - ph + 1, stride_h):
            for w in range(0, W - pw + 1, stride_w):
                patch = image[..., d:d+pd, h:h+ph, w:w+pw]
                with torch.no_grad():
                    patch_pred = model(patch)
                output_logits[..., d:d+pd, h:h+ph, w:w+pw] += patch_pred
                count_map[..., d:d+pd, h:h+ph, w:w+pw] += 1
    
    output_logits /= count_map + 1e-8
    return output_logits

def test_model(model, dataloader, device, num_classes, ignore_index, patch_size, output_dir):
    model.eval()
    all_metrics = {'dice': [], 'iou': [], 'accuracy': [], 'sensitivity': [], 'specificity': []}
    
    ensure_dir(output_dir)

    with torch.no_grad():
        for images, masks, paths in tqdm(dataloader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            
            output_logits = sliding_window_inference(model, images, patch_size, overlap=0.5, device=device, num_classes=num_classes)
            preds = torch.argmax(output_logits, dim=1)
            
            for i in range(images.size(0)):
                metrics = calculate_metrics(preds[i].cpu(), masks[i].cpu(), num_classes, ignore_index)
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                
                original_path = paths[i]
                filename = os.path.basename(original_path)
                pred_mask_np = preds[i].cpu().numpy().astype(np.uint8)
                original_nib = nib.load(original_path)
                pred_nib = nib.Nifti1Image(pred_mask_np, original_nib.affine, original_nib.header)
                nib.save(pred_nib, os.path.join(output_dir, f"pred_{filename}"))

    print(f"\n--- 최종 테스트 결과 ({len(dataloader.dataset)}개 샘플) ---")
    for key, values in all_metrics.items():
        mean_value = np.mean(values)
        print(f"평균 {key.capitalize():<12}: {mean_value:.4f}")

if __name__ == '__main__':
    IMAGE_DIR = '/workspace/eye2brain/data/img'
    MASK_DIR = '/workspace/eye2brain/data/lbl'
    CHECKPOINT_PATH = "checkpoints_unet3d/best_model.pth"
    OUTPUT_DIR = "test_results"

    BATCH_SIZE = 1
    NUM_CLASSES = 4
    INPUT_CHANNELS = 1
    BASE_CHANNELS = 16
    IGNORE_INDEX = 0
    PATCH_SIZE = (64, 64, 64)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = Compose([CTNormalize(), ToTensor()])
    
    test_target_folders = ['q1_F', 'q1_M']
    image_paths, mask_paths = load_data_paths(IMAGE_DIR, MASK_DIR, target_folders=test_target_folders)
    
    num_test_samples = 20
    image_paths = image_paths[:num_test_samples]
    mask_paths = mask_paths[:num_test_samples]
    print(f"테스트를 {len(image_paths)}개의 샘플로 제한합니다.")
    test_dataset = MedicalImageDataset3D(image_paths, mask_paths, phase='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # --------------------
    
    model = UNet3D(in_channels=INPUT_CHANNELS, out_channels=NUM_CLASSES, base_channels=BASE_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("Best model loaded successfully.")

    test_model(model, test_loader, DEVICE, NUM_CLASSES, IGNORE_INDEX, PATCH_SIZE, OUTPUT_DIR)
