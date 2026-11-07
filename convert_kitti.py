
from pathlib import Path
import sys
import cv2

src = Path(sys.argv[1])
dst = Path("data/traffic.mp4")
frames = sorted(src.glob("*.png"))

if not frames:
    raise SystemExit(f"No PNG images found in {src}")

im = cv2.imread(str(frames[0]))
if im is None:
    raise SystemExit("Could not read the first image")

h, w = im.shape[:2]
dst.parent.mkdir(parents=True, exist_ok=True)

out = cv2.VideoWriter(
    str(dst),
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,
    (w, h)
)

if not out.isOpened():
    raise SystemExit("Could not create output video")

try:
    for f in frames:
        im = cv2.imread(str(f))
        if im is not None:
            out.write(im)
finally:
    out.release()

print(f"Created {dst} from {len(frames)} frames")
