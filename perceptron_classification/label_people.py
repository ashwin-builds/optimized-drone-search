"""
Label every person in an image using Perceptron's detect().

Setup:
    pip install "perceptron>=0.4.0" pillow
    export PERCEPTRON_API_KEY=sk_live_...

Usage:
    python label_people.py input.jpg            # writes input_labeled.jpg
    python label_people.py input.jpg out.png
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from perceptron import detect, image


def label_people(in_path: str, out_path: str | None = None) -> str:
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path.with_name(f"{in_path.stem}_labeled{in_path.suffix}")

    # Ask the model for boxes around people. Coordinates come back on a 0-1000 grid.
    result = detect(image(str(in_path)), classes=["person"])

    img = Image.open(in_path).convert("RGB")
    width, height = img.size

    # Convert normalized 0-1000 boxes to pixel coordinates.
    boxes = result.boxes_to_pixels(width, height) or []
    print(f"Found {len(boxes)} people")

    draw = ImageDraw.Draw(img)
    line_w = max(2, width // 300)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, width // 50))
    except OSError:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes, start=1):
        x1, y1 = box.top_left.x, box.top_left.y
        x2, y2 = box.bottom_right.x, box.bottom_right.y
        label = f"Person {i}"

        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=line_w)

        # Filled tag above the box (or inside it if the box touches the top edge).
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx2 - tx1, ty2 - ty1
        tag_y = y1 - th - 6 if y1 - th - 6 >= 0 else y1
        draw.rectangle([x1, tag_y, x1 + tw + 8, tag_y + th + 6], fill=(255, 60, 60))
        draw.text((x1 + 4, tag_y + 3 - ty1), label, fill="white", font=font)

        print(f"{label}: ({x1}, {y1}) -> ({x2}, {y2})  [{box.mention}]")

    img.save(out_path)
    print(f"Saved {out_path}")
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python label_people.py <image> [output]")
    label_people(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
