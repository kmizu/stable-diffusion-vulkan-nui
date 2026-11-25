#!/usr/bin/env python3
"""Generate the SDV-NUI faceted V for the app, Windows launcher and website."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
SITE_ASSETS = REPO / "site" / "assets"
UI_ICONS = REPO / "ui" / "src" / "main" / "resources" / "sdfx" / "ui" / "icons"

# Use the same polygons for vector and raster artwork, including 16px favicons.
FACETS = (
    ("#fff2da", ((11, 16), (22, 16), (34, 38), (32, 51))),
    ("#ff9e46", ((42, 16), (53, 16), (32, 51), (28, 44))),
)
BACKGROUND = "#672536"


def svg_artwork() -> str:
    facets = []
    for color, points in FACETS:
        coords = " ".join(f"{x},{y}" for x, y in points)
        facets.append(f'  <polygon fill="{color}" points="{coords}"/>')
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">',
        f'  <rect x="2" y="2" width="60" height="60" rx="13" fill="{BACKGROUND}"/>',
        *facets,
        '</svg>',
    ])


def render_master(size: int) -> Image.Image:
    scale = size / 64
    image = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2*scale, 2*scale, 62*scale-1, 62*scale-1),
                           radius=13*scale, fill=BACKGROUND)
    for color, points in FACETS:
        draw.polygon([(x*scale, y*scale) for x, y in points], fill=color)
    return image


def main() -> None:
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    UI_ICONS.mkdir(parents=True, exist_ok=True)

    # 1. Vector SVG (LF line endings)
    (SITE_ASSETS / "favicon.svg").write_text(svg_artwork() + "\n", encoding="utf-8", newline="\n")

    # 2. High-res master renders
    master = render_master(1024)

    def scaled(sz: int) -> Image.Image:
        return master.resize((sz, sz), Image.Resampling.LANCZOS)

    # 3. JavaFX PNG sizes
    ui_sizes = [16, 24, 32, 48, 64, 128, 256]
    ui_imgs = {}
    for sz in ui_sizes:
        img = scaled(sz)
        ui_imgs[sz] = img
        img.save(UI_ICONS / f"icon-{sz}.png", format="PNG")

    # 4. Windows multi-size ICO (for jpackage and Win32 taskbar)
    ico_sizes = [256, 128, 64, 48, 32, 24, 16]
    ico_frames = [ui_imgs[sz] for sz in ico_sizes]
    ico_frames[0].save(
        UI_ICONS / "sdnui.ico",
        format="ICO",
        sizes=[(sz, sz) for sz in ico_sizes],
        append_images=ico_frames[1:],
    )

    # 5. Web favicons & touch/PWA icons
    web_ico_sizes = [48, 32, 24, 16]
    web_ico_frames = [ui_imgs[sz] for sz in web_ico_sizes]
    web_ico_frames[0].save(
        SITE_ASSETS / "favicon.ico",
        format="ICO",
        sizes=[(sz, sz) for sz in web_ico_sizes],
        append_images=web_ico_frames[1:],
    )
    scaled(180).save(SITE_ASSETS / "apple-touch-icon.png", format="PNG")
    scaled(192).save(SITE_ASSETS / "icon-192.png", format="PNG")
    scaled(512).save(SITE_ASSETS / "icon-512.png", format="PNG")

    # Verify ICO headers and embedded frames
    for ico_path, expected in [(UI_ICONS / "sdnui.ico", ico_sizes), (SITE_ASSETS / "favicon.ico", web_ico_sizes)]:
        raw = ico_path.read_bytes()
        assert raw[:4] == b"\x00\x00\x01\x00", f"Unexpected ICO header in {ico_path}"
        with Image.open(ico_path) as opened:
            actual_sizes = sorted(opened.info.get("sizes", set()))
            assert actual_sizes == sorted((s, s) for s in expected), f"ICO size mismatch in {ico_path}: {actual_sizes}"
        print(f"Verified {ico_path}: sizes={expected}")


if __name__ == "__main__":
    main()
