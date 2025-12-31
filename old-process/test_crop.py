"""Test crop detection on a single page."""
from pathlib import Path
import cv2
import numpy as np
import sys

# Force flush
sys.stdout.reconfigure(line_buffering=True)

from src.processors.image_cropper import ImageCropper
from src.processors.base import ProcessingContext
from src.config import Config

def main():
    output = []
    
    # Setup
    config = Config()
    context = ProcessingContext(config=config)
    extracted_dir = Path('extracted/2025-EROLLGEN-S22-114-FinalRoll-Revision1-TAM-1-WI')
    context.setup_paths_from_extracted(extracted_dir)

    # Create cropper with debug
    cropper = ImageCropper(context, diagram_filter='auto')

    # Process just page 16 to debug
    img_path = context.images_dir / 'page-016.png'
    output.append(f'Testing: {img_path}')
    output.append(f'Exists: {img_path.exists()}')

    if img_path.exists():
        # Read image
        img_orig = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        H0, W0 = img_orig.shape[:2]
        output.append(f'Image size: {W0}x{H0}')
        
        # Detect boxes at canonical size
        img_canon = cv2.resize(img_orig, (1187, 1679), interpolation=cv2.INTER_AREA)
        boxes_canon = cropper._detect_boxes(img_canon)
        output.append(f'Boxes detected at canonical: {len(boxes_canon)}')
        
        # Scale back
        sx = W0 / 1187.0
        sy = H0 / 1679.0
        boxes_orig = cropper._scale_boxes(boxes_canon, sx, sy)
        output.append(f'Boxes at original scale: {len(boxes_orig)}')
        
        # Classify with diagram_filter=off to see all boxes
        cropper_off = ImageCropper(context, diagram_filter='off')
        voter_all, _, _ = cropper_off._classify_boxes(img_orig, boxes_orig, W0, H0, 'page-016')
        output.append(f'All boxes (filter off): {len(voter_all)}')
        
        # Classify with diagram_filter=auto
        voter_candidates, skipped, rejected_metrics = cropper._classify_boxes(img_orig, boxes_orig, W0, H0, 'page-016')
        output.append(f'Accepted (filter auto): {len(voter_candidates)}, Rejected: {skipped}')
        
        # Show rejection reasons
        output.append(f'\nFirst 5 rejection reasons:')
        for i, m in enumerate(rejected_metrics[:5]):
            reasons = getattr(m, '_rejection_reasons', [])
            output.append(f'  Rejection {i+1}: ink={m.ink_frac:.4f}, line_ratio={m.line_ratio:.4f}, small_comp={m.small_components}, largest_cc={m.largest_cc_ratio:.4f}, edge={m.edge_frac:.4f}')
            output.append(f'    Reasons: {reasons}')
    else:
        output.append('Image not found!')
    
    # Write to file
    with open('test_crop_output.txt', 'w') as f:
        f.write('\n'.join(output))
    print('\n'.join(output))

if __name__ == '__main__':
    main()
