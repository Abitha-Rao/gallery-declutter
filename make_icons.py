"""One-off script: generate simple PWA app icons (camera glyph on rounded square)."""
from PIL import Image, ImageDraw

BG = (91, 140, 255)  # matches --accent in the app
GLYPH = (255, 255, 255)


def make_icon(size, path):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = int(size * 0.22)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=BG)

    # simple camera body
    body_w, body_h = int(size * 0.56), int(size * 0.38)
    body_x = (size - body_w) // 2
    body_y = int(size * 0.36)
    draw.rounded_rectangle(
        [body_x, body_y, body_x + body_w, body_y + body_h],
        radius=int(size * 0.05), fill=GLYPH,
    )
    # viewfinder bump
    bump_w, bump_h = int(size * 0.18), int(size * 0.08)
    bump_x = body_x + int(body_w * 0.15)
    bump_y = body_y - bump_h + int(size * 0.02)
    draw.rounded_rectangle(
        [bump_x, bump_y, bump_x + bump_w, bump_y + bump_h],
        radius=int(size * 0.02), fill=GLYPH,
    )
    # lens
    lens_r = int(size * 0.13)
    cx, cy = size // 2, body_y + body_h // 2
    draw.ellipse([cx - lens_r, cy - lens_r, cx + lens_r, cy + lens_r], fill=BG)
    inner_r = int(lens_r * 0.55)
    draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], fill=GLYPH)

    img.save(path)


if __name__ == "__main__":
    make_icon(192, "static/icon-192.png")
    make_icon(512, "static/icon-512.png")
    print("Icons written to static/")
