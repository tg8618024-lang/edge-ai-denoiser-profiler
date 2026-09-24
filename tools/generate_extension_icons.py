"""Generate valid pixel-perfect PNG icons for the Chrome Extension.

Creates:
- extensions/chrome-edge-denoiser/icons/icon-16.png (16x16)
- extensions/chrome-edge-denoiser/icons/icon-48.png (48x48)
- extensions/chrome-edge-denoiser/icons/icon-128.png (128x128)
"""

import os
from PIL import Image, ImageDraw

def generate_icon(size: int, output_path: str) -> None:
    """Draw an NVIDIA-green audio waveform / microphone badge icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rectangle or circle
    bg_color = (18, 22, 28, 255)  # Dark cyber slate
    border_color = (118, 185, 0, 255)  # NVIDIA Green (#76B900)
    
    pad = max(1, size // 16)
    draw.rounded_rectangle(
        [pad, pad, size - pad - 1, size - pad - 1],
        radius=size // 4,
        fill=bg_color,
        outline=border_color,
        width=max(1, size // 24),
    )

    # Audio waveform bars in the center
    num_bars = 5
    bar_width = max(1, size // 10)
    spacing = max(1, size // 14)
    total_w = num_bars * bar_width + (num_bars - 1) * spacing
    start_x = (size - total_w) // 2
    center_y = size // 2

    # Heights relative to size
    height_factors = [0.35, 0.65, 0.90, 0.65, 0.35]
    max_h = int(size * 0.55)

    for i, factor in enumerate(height_factors):
        h = max(2, int(max_h * factor))
        x0 = start_x + i * (bar_width + spacing)
        y0 = center_y - h // 2
        x1 = x0 + bar_width - 1
        y1 = center_y + h // 2
        # Center bar electric cyan (0, 229, 255), side bars NVIDIA green
        color = (0, 229, 255, 255) if i == 2 else (118, 185, 0, 255)
        draw.rectangle([x0, y0, x1, y1], fill=color)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Generated {size}x{size} icon: {output_path}")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extensions", "chrome-edge-denoiser", "icons"))
    for s in [16, 48, 128]:
        out = os.path.join(base_dir, f"icon-{s}.png")
        generate_icon(s, out)


if __name__ == "__main__":
    main()
