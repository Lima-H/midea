from pathlib import Path
import cv2
import numpy as np
from utils.detections import detectar_furos_combinado

INPUT_DIRS = [
    Path(__file__).resolve().parent.parent / 'imagens_iphone',
    Path(__file__).resolve().parent.parent / 'Imagens'
]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def run_test_on_folder(folder: Path):
    files = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
    for img_path in files:
        print('Processing', img_path)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print('Failed to read', img_path)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_result, qtd, qtd_fora, debug = detectar_furos_combinado(img_rgb, debug=True)
        print('Detected holes:', qtd, 'Outside tolerance:', qtd_fora)
        # Save results
        out_path = OUTPUT_DIR / (img_path.stem + '_result.jpg')
        out_mask = OUTPUT_DIR / (img_path.stem + '_mask.jpg')
        out_bh = OUTPUT_DIR / (img_path.stem + '_bh.jpg')
        out_th = OUTPUT_DIR / (img_path.stem + '_th.jpg')
        cv2.imwrite(str(out_path), cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))
        if 'combined_mask' in debug and debug['combined_mask'] is not None:
            cv2.imwrite(str(out_mask), debug['combined_mask'])
        if 'blackhat' in debug and debug['blackhat'] is not None:
            bh = debug['blackhat']
            if isinstance(bh, dict):
                cv2.imwrite(str(out_bh), bh['blackhat'])
        if 'threshold' in debug and debug['threshold'] is not None:
            th = debug['threshold']
            if isinstance(th, dict):
                cv2.imwrite(str(out_th), th['thresh_clean'])


if __name__ == '__main__':
    for d in INPUT_DIRS:
        if d.exists():
            print('Scanning', d)
            run_test_on_folder(d)
        else:
            print('Skipping non-existent', d)
