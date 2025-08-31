import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label, binary_fill_holes

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def postprocess_segmentation(input_path, output_path):
    try:
        nib_img = nib.load(input_path)
        pred_mask = nib_img.get_fdata().astype(np.uint8)
        clean_mask = np.zeros_like(pred_mask)
        for class_label in range(1, np.max(pred_mask) + 1):
            class_mask = (pred_mask == class_label)
            if not np.any(class_mask):
                continue
                
            labeled_mask, num_features = label(class_mask)
            if num_features == 0:
                continue

            component_sizes = np.bincount(labeled_mask.ravel())
            largest_component_label = component_sizes[1:].argmax() + 1
            largest_component_mask = (labeled_mask == largest_component_label)
            largest_component_mask = binary_fill_holes(largest_component_mask)
            clean_mask[largest_component_mask] = class_label
        clean_nib = nib.Nifti1Image(clean_mask.astype(np.uint8), nib_img.affine, nib_img.header)
        nib.save(clean_nib, output_path)

    except Exception as e:
        print(f"파일 처리 중 오류 발생: {input_path}, 오류: {e}")

if __name__ == '__main__':
    INPUT_DIR = "test_results"
    OUTPUT_DIR = "postprocessed_results"
    
    ensure_dir(OUTPUT_DIR)

    pred_files = glob.glob(os.path.join(INPUT_DIR, 'pred_*.nii.gz'))
    
    if not pred_files:
        print(f"'{INPUT_DIR}' 폴더에서 예측 파일을 찾을 수 없습니다. 'test.py'를 먼저 실행했는지 확인하세요.")
    else:
        print(f"총 {len(pred_files)}개의 파일에 대해 후처리를 시작합니다...")
        for file_path in tqdm(pred_files, desc="Post-processing"):
            filename = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_DIR, filename)
            postprocess_segmentation(file_path, output_path)
        print("후처리가 완료되었습니다.")
