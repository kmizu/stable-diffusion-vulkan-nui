#!/usr/bin/env python3
"""Generate cohesive SDNUI brand icons across web (SVG/ICO/PNG), JavaFX UI, and Windows packaging.

Renders a geometric "N" diffusion-prism monogram inside a midnight-sapphire-cyan squircle.
Uses supersampled rendering (1024x1024 -> LANCZOS) with Pillow so raster assets match
site/assets/favicon.svg faithfully across 16x16 favicons through 512x512 app icons.
"""

from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parent.parent
SITE_ASSETS = REPO / "site" / "assets"
UI_ICONS = REPO / "ui" / "src" / "main" / "resources" / "sdfx" / "ui" / "icons"

SVG_CONTENT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="bg" x1="8" y1="6" x2="58" y2="58" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0b1120"/>
      <stop offset="0.52" stop-color="#1e40af"/>
      <stop offset="1" stop-color="#0284c7"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="48%" r="48%">
      <stop offset="0" stop-color="#7dd3fc" stop-opacity="0.44"/>
      <stop offset="0.58" stop-color="#38bdf8" stop-opacity="0.12"/>
      <stop offset="1" stop-color="#0284c7" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="rim" x1="10" y1="6" x2="54" y2="58" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.38"/>
      <stop offset="0.45" stop-color="#93c5fd" stop-opacity="0.16"/>
      <stop offset="1" stop-color="#38bdf8" stop-opacity="0.06"/>
    </linearGradient>
    <linearGradient id="prism" x1="16" y1="18" x2="46" y2="48" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="0.55" stop-color="#e0f2fe"/>
      <stop offset="1" stop-color="#7dd3fc"/>
    </linearGradient>
  </defs>
  <rect x="4" y="4" width="56" height="56" rx="14" fill="url(#bg)"/>
  <circle cx="32" cy="31" r="22" fill="url(#glow)"/>
  <rect x="5" y="5" width="54" height="54" rx="13" fill="none" stroke="url(#rim)" stroke-width="1.6"/>
  <path d="M18.5 44.5 V22 L41.5 44.5 V27" fill="none" stroke="url(#prism)" stroke-width="7.4" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M18.5 44.5 V22" fill="none" stroke="#ffffff" stroke-width="7.4" stroke-linecap="round"/>
  <path d="M46 11.5 L47.3 15.7 L51.5 17 L47.3 18.3 L46 22.5 L44.7 18.3 L40.5 17 L44.7 15.7 Z" fill="#ffffff"/>
</svg>
"""


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(round(_lerp(c0[0], c1[0], t))),
        int(round(_lerp(c0[1], c1[1], t))),
        int(round(_lerp(c0[2], c1[2], t))),
    )


def _rounded_rect_mask(size: int, pad: float, radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([pad, pad, size - 1 - pad, size - 1 - pad], radius=radius, fill=255)
    return mask


def _draw_capsule(mask: Image.Image, x0: float, y0: float, x1: float, y1: float, r: float) -> None:
    """Draw a geometrically exact rounded-end segment (stadium capsule) with zero corner artifacts."""
    d = ImageDraw.Draw(mask)
    d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=255)
    d.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=255)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length > 1e-6:
        nx, ny = -dy / length * r, dx / length * r
        d.polygon(
            [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)],
            fill=255,
        )


def _four_point_spark_polygon(cx: float, cy: float, outer_r: float) -> list[tuple[float, float]]:
    """Build a crisp 4-point diffusion spark diamond."""
    inner_r = outer_r * 0.25
    return [
        (cx, cy - outer_r),
        (cx + inner_r, cy - inner_r),
        (cx + outer_r, cy),
        (cx + inner_r, cy + inner_r),
        (cx, cy + outer_r),
        (cx - inner_r, cy + inner_r),
        (cx - outer_r, cy),
        (cx - inner_r, cy - inner_r),
    ]


def _diagonal_gradient(size: int, c0: tuple[int, int, int], c1: tuple[int, int, int], c2: tuple[int, int, int], mid: float) -> Image.Image:
    grid_n = 256
    grad = Image.new("RGBA", (grid_n, grid_n), (0, 0, 0, 255))
    px = grad.load()
    denom = float(2 * (grid_n - 1))
    for y in range(grid_n):
        for x in range(grid_n):
            t = (x + y) / denom
            if t < mid:
                rgb = _lerp_rgb(c0, c1, t / mid)
            else:
                rgb = _lerp_rgb(c1, c2, (t - mid) / (1.0 - mid))
            px[x, y] = (rgb[0], rgb[1], rgb[2], 255)
    return grad.resize((size, size), Image.Resampling.BILINEAR)


def render_master(size: int = 1024, simplify_small: bool = False) -> Image.Image:
    """Render high-res RGBA icon.

    When simplify_small is True (for <= 24px sizes), strokes and pillar geometry are bolded
    and corner padding is tightened so 16x16 favicons stay crisp and legible.
    """
    pad = size * (0.042 if simplify_small else 0.0625)
    radius = size * 0.22

    bg_full = _diagonal_gradient(
        size,
        (11, 17, 32),   # #0b1120 midnight indigo
        (30, 64, 175),  # #1e40af royal sapphire
        (2, 132, 199),  # #0284c7 vibrant sky cyan
        0.52,
    )

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    squircle_mask = _rounded_rect_mask(size, pad, radius)
    canvas.paste(bg_full, (0, 0), squircle_mask)

    # Central cyan aura clipped inside squircle
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx = cy = size / 2.0
    r_glow = size * 0.34
    gd.ellipse([cx - r_glow, cy - r_glow * 1.05, cx + r_glow, cy + r_glow * 0.95], fill=(125, 211, 252, 92))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=size * 0.115))
    clipped_glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    clipped_glow.paste(glow, (0, 0), squircle_mask)
    canvas.alpha_composite(clipped_glow)

    # Subtle glass specular inner rim
    rim_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rim_layer)
    rim_inset = pad + size * 0.015
    rim_w = max(2, int(round(size * 0.025)))
    rd.rounded_rectangle(
        [rim_inset, rim_inset, size - 1 - rim_inset, size - 1 - rim_inset],
        radius=radius * 0.93,
        outline=(224, 242, 254, 82),
        width=rim_w,
    )
    canvas.alpha_composite(rim_layer)

    # Geometric "N" diffusion-prism monogram: 3 exact shared-endpoint stadium capsules.
    # Right pillar starts cleanly below the top-right spark so the spark reads clearly with breathing room.
    r = size * (0.072 if simplify_small else 0.058)
    lx = size * (0.265 if simplify_small else 0.289)
    rx = size * (0.680 if simplify_small else 0.648)
    y_top_l = size * (0.310 if simplify_small else 0.344)
    y_top_r = size * (0.415 if simplify_small else 0.422)
    y_bot = size * (0.720 if simplify_small else 0.695)

    mono_mask = Image.new("L", (size, size), 0)
    _draw_capsule(mono_mask, lx, y_bot, lx, y_top_l, r)
    _draw_capsule(mono_mask, lx, y_top_l, rx, y_bot, r)
    _draw_capsule(mono_mask, rx, y_bot, rx, y_top_r, r)

    # Soft cyan bloom behind monogram
    halo = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    halo.paste((125, 211, 252, 120), (0, 0), mono_mask)
    halo = halo.filter(ImageFilter.GaussianBlur(radius=size * 0.030))
    canvas.alpha_composite(halo)

    # Luminous top-left white to icy sky-cyan prism gradient across diagonal + right arm
    prism_fill = _diagonal_gradient(
        size,
        (255, 255, 255),
        (232, 246, 255),
        (147, 224, 254),
        0.55,
    )
    fg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    fg.paste(prism_fill, (0, 0), mono_mask)

    # Keep left pillar solid pure white for crisp contrast
    left_mask = Image.new("L", (size, size), 0)
    _draw_capsule(left_mask, lx, y_bot, lx, y_top_l, r)
    fg_left = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    fg_left.paste((255, 255, 255, 255), (0, 0), left_mask)
    canvas.alpha_composite(fg)
    canvas.alpha_composite(fg_left)

    # Crisp 4-point diffusion spark positioned clearly above the right pillar apex
    spark_cx = size * (0.725 if simplify_small else 0.719)
    spark_cy = size * (0.250 if simplify_small else 0.266)
    spark_r = size * (0.100 if simplify_small else 0.086)
    spark_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spark_layer)
    sd.polygon(_four_point_spark_polygon(spark_cx, spark_cy, spark_r), fill=(255, 255, 255, 255))
    canvas.alpha_composite(spark_layer)

    return canvas


def main() -> None:
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    UI_ICONS.mkdir(parents=True, exist_ok=True)

    # 1. Vector SVG (LF line endings)
    (SITE_ASSETS / "favicon.svg").write_text(SVG_CONTENT.rstrip() + "\n", encoding="utf-8", newline="\n")

    # 2. High-res master renders
    master_std = render_master(1024, simplify_small=False)
    master_small = render_master(512, simplify_small=True)

    def scaled(sz: int) -> Image.Image:
        src = master_small if sz <= 24 else master_std
        return src.resize((sz, sz), Image.Resampling.LANCZOS)

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
