"""Fine-tune the bundled nine-class segmentation checkpoint with the tenth class."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default=None, help='Examples: 0, mps, cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model = YOLO(str(root / 'weights' / 'best.pt'))
    model.train(
        data=str(root / 'data.yaml'), epochs=args.epochs, batch=args.batch,
        imgsz=800, device=args.device, workers=0, resume=False, cls_remap=True,
        project=str(root / 'runs'), name='rice10_finetune', seed=42,
    )


if __name__ == '__main__':
    main()
