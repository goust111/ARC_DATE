from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, scale_sprite


MAX_GENERATION_ATTEMPTS = 400
MIN_PAIR_COUNT = 2
MAX_PAIR_COUNT = 4
MAX_GRID_SIZE = 30

SPRITE_TEMPLATES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)


def _component_first_key(component: dict) -> tuple[int, int]:
    coords = np.asarray(component["coords"], dtype=int)
    top_row = int(coords[:, 0].min())
    left_col = int(coords[coords[:, 0] == top_row][:, 1].min())
    return top_row, left_col


def _is_hollow_rectangle(grid: np.ndarray, component: dict) -> bool:
    bg = int(get_bg(grid))
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    patch = crop_bbox(grid, component["bbox"])
    expected = np.full((h, w), bg, dtype=int)
    expected[0, :] = color
    expected[-1, :] = color
    expected[:, 0] = color
    expected[:, -1] = color
    return np.array_equal(patch, expected)


def _classify_components(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    frames: list[dict] = []
    sprites: list[dict] = []
    for component in components:
        if _is_hollow_rectangle(g, component):
            frames.append(component)
        else:
            sprites.append(component)

    assert frames, "expected at least one frame"
    assert len(frames) == len(sprites), "expected the same number of frames and sprites"
    frames.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    sprites.sort(key=_component_first_key)
    return bg, frames, sprites


def _scaled_recolored_sprite(
    grid: np.ndarray,
    sprite: dict,
    frame: dict,
) -> tuple[np.ndarray, tuple[int, int]]:
    bg = int(get_bg(grid))
    sr0, sc0, sr1, sc1 = map(int, sprite["bbox"])
    fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
    sprite_color = int(sprite["color"])
    frame_color = int(frame["color"])

    sprite_patch = crop_bbox(grid == sprite_color, (sr0, sc0, sr1, sc1))
    sprite_h, sprite_w = map(int, sprite_patch.shape)
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1

    assert inner_h >= 1 and inner_w >= 1, "frame must have a non-empty interior"
    assert inner_h % sprite_h == 0 and inner_w % sprite_w == 0, "frame interior must be an integer multiple"
    factor_h = inner_h // sprite_h
    factor_w = inner_w // sprite_w
    assert factor_h == factor_w, "frame interior must encode a single integer scale factor"
    factor = int(factor_h)
    assert factor >= 1, "scale factor must be positive"

    scaled_mask = scale_sprite(sprite_patch.astype(int), factor).astype(bool)
    assert scaled_mask.shape == (inner_h, inner_w), "scaled sprite must exactly fill the frame interior"
    patch = np.full((inner_h, inner_w), bg, dtype=int)
    patch[scaled_mask] = frame_color
    return patch, (fr0 + 1, fc0 + 1)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Pair sprites with frames by position order, not by color.

    Frames are matched left-to-right. Sprites are matched by the row-major position
    of their first pixel. Each sprite is block-scaled by the unique integer implied
    by the matched frame interior, recolored to the frame color, and inserted into
    that interior. The original sprite instances are removed.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, frames, sprites = _classify_components(g)

    out = g.copy()
    for sprite in sprites:
        coords = np.asarray(sprite["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg

    for frame, sprite in zip(frames, sprites, strict=True):
        patch, (r0, c0) = _scaled_recolored_sprite(g, sprite, frame)
        paste(out, patch, r0, c0, transparent=bg)

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _random_derangement(values: list[int], rng: np.random.Generator) -> list[int]:
    arr = list(values)
    while True:
        perm = arr.copy()
        rng.shuffle(perm)
        if all(a != b for a, b in zip(arr, perm, strict=True)):
            return perm


def _sample_layout(
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int], list[int], list[tuple[int, int, int, int]], tuple[int, int]]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair_count = int(rng.integers(MIN_PAIR_COUNT, MAX_PAIR_COUNT + 1))
        templates = [SPRITE_TEMPLATES[int(rng.integers(0, len(SPRITE_TEMPLATES)))] for _ in range(pair_count)]
        scales = [int(rng.integers(2, 5)) for _ in range(pair_count)]

        widths = [int(template.shape[1] * scale + 2) for template, scale in zip(templates, scales, strict=True)]
        heights = [int(template.shape[0] * scale + 2) for template, scale in zip(templates, scales, strict=True)]
        gaps = [int(rng.integers(1, 4)) for _ in range(pair_count - 1)]
        total_span = sum(widths) + sum(gaps)
        if total_span > 26:
            continue
        if max(heights) > 14:
            continue

        canvas_w = int(total_span + rng.integers(4, min(10, MAX_GRID_SIZE - total_span) + 1))
        max_sprite_h = max(int(template.shape[0]) for template in templates)
        canvas_h = max(18, max(heights) + max_sprite_h + 9)
        canvas_h = min(canvas_h, MAX_GRID_SIZE)
        if canvas_w > MAX_GRID_SIZE or canvas_h > MAX_GRID_SIZE:
            continue

        left_margin_max = canvas_w - total_span - 1
        if left_margin_max < 1:
            continue
        left = int(rng.integers(1, left_margin_max + 1))
        top_max = canvas_h - max(heights) - 3
        top = int(rng.integers(2, min(6, top_max) + 1))

        boxes: list[tuple[int, int, int, int]] = []
        c = left
        for idx, (height, width) in enumerate(zip(heights, widths, strict=True)):
            boxes.append((top, c, top + height - 1, c + width - 1))
            if idx + 1 < pair_count:
                c += width + gaps[idx]
        return templates, scales, widths, boxes, (canvas_h, canvas_w)

    raise RuntimeError("failed to sample a valid frame layout")


def _place_sprites(
    grid: np.ndarray,
    templates: list[np.ndarray],
    colors: list[int],
    frame_boxes: list[tuple[int, int, int, int]],
    bg: int,
    rng: np.random.Generator,
) -> bool:
    h, w = map(int, grid.shape)
    reserved = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    prev_key = (-1, -1)
    for template, color in zip(templates, colors, strict=True):
        ph, pw = map(int, template.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                if reserved[r0 : r0 + ph, c0 : c0 + pw].any():
                    continue
                key = (r0, c0)
                if key <= prev_key:
                    continue
                candidates.append(key)
        if not candidates:
            return False

        choice = candidates[int(rng.integers(0, len(candidates)))]
        r0, c0 = choice
        patch = np.full((ph, pw), int(bg), dtype=int)
        patch[template] = int(color)
        paste(grid, patch, r0, c0, transparent=bg)

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r0 + ph)
        cc1 = min(w - 1, c0 + pw)
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        prev_key = key

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = list(Color.ALL_COLORS)

    for _ in range(MAX_GENERATION_ATTEMPTS):
        bg = int(palette[int(rng.integers(0, len(palette)))])
        available = [color for color in palette if color != bg]

        templates, scales, _widths, frame_boxes, shape = _sample_layout(rng)
        pair_count = len(frame_boxes)
        frame_colors = rng.choice(available, size=pair_count, replace=False).astype(int).tolist()
        sprite_colors = _random_derangement(frame_colors, rng)

        grid = np.full(shape, bg, dtype=int)
        for color, bbox in zip(frame_colors, frame_boxes, strict=True):
            _draw_frame(grid, bbox, int(color))

        if not _place_sprites(grid, templates, sprite_colors, frame_boxes, bg, rng):
            continue
        if int(get_bg(grid)) != bg:
            continue

        solved = main(grid)
        if solved.shape != grid.shape:
            continue
        if solved.shape[0] > MAX_GRID_SIZE or solved.shape[1] > MAX_GRID_SIZE:
            continue

        # Regenerate until the intended order-based pairing is actually non-color-based.
        _, frames, sprites = _classify_components(grid)
        if any(int(frame["color"]) == int(sprite["color"]) for frame, sprite in zip(frames, sprites, strict=True)):
            continue

        for frame, sprite, scale in zip(frames, sprites, scales, strict=True):
            patch, (r0, c0) = _scaled_recolored_sprite(grid, sprite, frame)
            if patch.shape != (crop_bbox(grid, frame["bbox"]).shape[0] - 2, crop_bbox(grid, frame["bbox"]).shape[1] - 2):
                break
            expected_inner_h = int(sprite["bbox"][2] - sprite["bbox"][0] + 1) * int(scale)
            expected_inner_w = int(sprite["bbox"][3] - sprite["bbox"][1] + 1) * int(scale)
            if patch.shape != (expected_inner_h, expected_inner_w):
                break
            interior = solved[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
            if not np.array_equal(interior, patch):
                break
        else:
            return grid

    raise RuntimeError("failed to generate a valid input grid")
