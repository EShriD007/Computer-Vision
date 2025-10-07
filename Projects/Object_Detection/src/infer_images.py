import argparse, json, cv2, numpy as np, pathlib, os
from models_tfhub import TFHubDetector

BASE = pathlib.Path(__file__).resolve().parents[1]
VAL_DIR = BASE/"data"/"coco"/"val"
IMG_DIR = VAL_DIR/"data"
ANN = json.load(open(VAL_DIR/"labels.json"))
OUT_DIR = BASE/"outputs"/"phase2_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: a tiny COCO id->name map (keeps labels readable). If missing, we'll print raw class ids.
COCO80 = {
    1:"person", 44:"bottle", 47:"cup", 48:"fork", 49:"knife", 50:"spoon", 51:"bowl",
    62:"chair", 63:"couch", 64:"potted plant", 67:"dining table", 73:"laptop",
    74:"mouse", 75:"remote", 76:"keyboard", 77:"cell phone", 78:"microwave",
    79:"oven", 80:"toaster", 81:"sink", 82:"refrigerator", 84:"book", 86:"vase",
    # ... add more if you need to display label rather than the id
}

def draw(frame, boxes, scores, classes, score_thresh=0.4):
    h, w = frame.shape[:2]
    for b, s, c in zip(boxes, scores, classes):
        if s < score_thresh: 
            continue
        y1,x1,y2,x2 = b
        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{COCO80.get(int(c), int(c))}:{s:.2f}"
        cv2.putText(frame, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def main(model_name, limit, score):
    det = TFHubDetector(model_name=model_name, score_thresh=score)
    imgs = [img["file_name"] for img in ANN["images"]][:limit]
    for i, fname in enumerate(imgs, 1):
        path = str(IMG_DIR/fname)
        im = cv2.imread(path)
        boxes, scores, classes, dt = det(im)
        vis = im.copy()
        draw(vis, boxes, scores, classes, score_thresh=score)
        out = str(OUT_DIR/f"{model_name}_{i:03d}.jpg")
        cv2.imwrite(out, vis)
        print(f"[{i}/{len(imgs)}] {fname}  dets={int((scores>=score).sum())}  dt={dt:.3f}s  -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ssd_mobilenet_v2",
                    choices=["ssd_mobilenet_v2","efficientdet_d0","efficientdet_d1"])
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--score", type=float, default=0.4)
    a = ap.parse_args()
    main(a.model, a.limit, a.score)
