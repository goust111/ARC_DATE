from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
LEGEND_SIZE_CHOICES = (3, 4)
LEGEND_COUNT_RANGE = (2, 4)
FENCE_COUNT_RANGE = (1, 4)
LEGEND_GAP_RANGE = (2, 4)
FENCE_INNER_H_RANGE = (3, 7)
FENCE_INNER_W_RANGE = (3, 8)

TEXTURE_TEMPLATES = {
    3: (
        ((1, 1, 0), (0, 1, 1), (1, 1, 0)),
        ((1, 0, 1), (1, 1, 1), (0, 1, 0)),
        ((1, 1, 0), (1, 1, 1), (0, 1, 1)),
        ((0, 1, 1), (1, 1, 0), (1, 1, 1)),
    ),
    4: (
        ((1, 1, 0, 1), (0, 1, 1, 1), (1, 1, 0, 1), (0, 1, 1, 0)),
        ((1, 0, 1, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 1, 0, 1)),
        ((1, 1, 0, 1), (1, 0, 1, 1), (1, 1, 1, 0), (0, 1, 1, 1)),
        ((0, 1, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0), (1, 0, 1, 1)),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(coords) != expected:
        return False
    cells = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    return True


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == BG:
            continue
        for comp in _components(grid == color):
            if not _is_rect_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "cells": set(comp),
                }
            )
    return frames


def _extract_legends(grid: np.ndarray, frames: list[dict[str, object]]) -> dict[int, np.ndarray]:
    frame_sets = {frozenset(frame["cells"]) for frame in frames}
    legends: dict[int, np.ndarray] = {}
    for comp in _components(grid != BG):
        comp_key = frozenset(comp)
        if comp_key in frame_sets or len(comp) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width not in LEGEND_SIZE_CHOICES or height != width + 1:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        non_bg = sub != BG
        if np.any(non_bg[:-1].sum(axis=1) == 0) or np.any(non_bg[:-1].sum(axis=0) == 0):
            continue

        bottom = np.flatnonzero(non_bg[-1])
        if len(bottom) != 1:
            continue
        seed_col = int(bottom[0])
        centers = [width // 2] if width % 2 == 1 else [width // 2 - 1, width // 2]
        if seed_col not in centers:
            continue
        if not non_bg[-2, seed_col]:
            continue

        seed_color = int(sub[-1, seed_col])
        texture = sub[:-1].copy()
        legends[seed_color] = texture

    assert legends, "expected at least one legend component"
    return legends


def _tile_texture(texture: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = texture.shape
    reps_r = (h + th - 1) // th
    reps_c = (w + tw - 1) // tw
    return np.tile(texture, (reps_r, reps_c))[:h, :w]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each fenced interior with the tiled legend texture keyed by its seed color.

    Input structure:
    - The background is black.
    - Along the top are several legend components: each has a 3x3 or 4x4 texture patch and one seed pixel directly below a center column of that patch.
    - Lower in the grid are one-pixel-thick closed rectangular gray fences whose interiors contain exactly one colored seed pixel on an otherwise black background.

    Rule:
    - Detect each legend and build a mapping from its seed color to the legend's texture patch.
    - For every closed fence, read the unique seed color inside the fence, look up the matching texture, and tile that texture across the whole fence interior from the interior's top-left corner.
    - Keep the fence borders and the top legend area unchanged; the original interior seed pixel is overwritten by the tiled texture.

    Key insight:
    - The isolated colored pixel inside each fence is a lookup key, not content to preserve: it selects which top legend texture must be repeated inside that fence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _detect_frames(g)
    assert frames, "expected at least one closed rectangular fence"
    legends = _extract_legends(g, frames)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere(inner != BG)
        assert len(seeds) == 1, "each fence interior must contain exactly one seed pixel"
        sr, sc = map(int, seeds[0])
        seed_color = int(inner[sr, sc])
        assert seed_color in legends, "fence seed color must exist in the legend area"
        texture = legends[seed_color]
        tiled = _tile_texture(texture, inner.shape)
        out[r0 + 1 : r1, c0 + 1 : c1] = tiled

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = _components(mask.astype(bool))
    return len(pts) == 1


def _sample_texture(size: int) -> tuple[np.ndarray, int]:
    middle_cols = [size // 2] if size % 2 == 1 else [size // 2 - 1, size // 2]
    for _ in range(80):
        mask = np.array(random.choice(TEXTURE_TEMPLATES[size]), dtype=int)
        mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if not _mask_connected(mask.astype(bool)):
            continue
        if np.array_equal(mask.astype(bool), _frame_mask((size, size))):
            continue
        anchors = [c for c in middle_cols if mask[-1, c] == 1]
        if not anchors:
            continue
        return mask.astype(int), random.choice(anchors)
    raise RuntimeError(f"failed to sample a valid {size}x{size} texture")


def _place_legends(width: int, sizes: list[int]) -> list[int] | None:
    gaps = [random.randint(*LEGEND_GAP_RANGE) for _ in range(len(sizes) - 1)]
    total = sum(sizes) + sum(gaps)
    if total > width:
        return None
    left_margin = random.randint(0, width - total)
    starts: list[int] = []
    col = left_margin
    for i, size in enumerate(sizes):
        starts.append(col)
        if i < len(gaps):
            col += size + gaps[i]
    return starts


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_fences(height: int, width: int, lower_top: int, count: int) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    occupied[:lower_top, :] = True
    placements: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(160):
            inner_h = random.randint(*FENCE_INNER_H_RANGE)
            inner_w = random.randint(*FENCE_INNER_W_RANGE)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h > height - lower_top or outer_w > width:
                continue
            r0 = random.randint(lower_top, height - outer_h)
            c0 = random.randint(0, width - outer_w)
            if not _can_place(occupied, r0, c0, outer_h, outer_w):
                continue
            occupied[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
            placements.append((r0, c0, inner_h, inner_w))
            placed = True
            break
        if not placed:
            return None
    return placements


def _render_pair(
    height: int,
    width: int,
    legend_top: int,
    legend_specs: list[dict[str, object]],
    fence_specs: list[dict[str, object]],
    fence_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((height, width), dtype=int)
    for spec in legend_specs:
        top = int(spec["top"])
        left = int(spec["left"])
        texture = spec["texture"]
        seed_color = int(spec["seed_color"])
        seed_col = int(spec["seed_col"])
        size = int(texture.shape[0])
        colored = np.where(texture != 0, int(spec["texture_color"]), 0)
        inp[top : top + size, left : left + size] = colored
        inp[top + size, left + seed_col] = seed_color

    out = inp.copy()
    for spec in fence_specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        seed_r = int(spec["seed_r"])
        seed_c = int(spec["seed_c"])
        seed_color = int(spec["seed_color"])
        texture = spec["texture"]

        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1
        inp[r0 : r1 + 1, c0] = fence_color
        inp[r0 : r1 + 1, c1] = fence_color
        inp[r0, c0 : c1 + 1] = fence_color
        inp[r1, c0 : c1 + 1] = fence_color
        inp[seed_r, seed_c] = seed_color

        out[r0 : r1 + 1, c0] = fence_color
        out[r0 : r1 + 1, c1] = fence_color
        out[r0, c0 : c1 + 1] = fence_color
        out[r1, c0 : c1 + 1] = fence_color
        tiled = _tile_texture(texture, (inner_h, inner_w))
        out[r0 + 1 : r1, c0 + 1 : c1] = tiled

    _ = legend_top
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.randint(*LEGEND_COUNT_RANGE)
    fence_count = random.randint(*FENCE_COUNT_RANGE)
    legend_sizes = [random.choice(LEGEND_SIZE_CHOICES) for _ in range(legend_count)]
    min_width = max(18, sum(legend_sizes) + 2 * (legend_count - 1))
    if min_width > 30:
        return None

    width = random.randint(min_width, 30)
    height = random.randint(18, 30)
    legend_top = random.randint(0, 1)
    legend_lefts = _place_legends(width, legend_sizes)
    if legend_lefts is None:
        return None

    lower_top = legend_top + max(legend_sizes) + 3
    if height - lower_top < 5:
        return None

    required_colors = 2 * legend_count + 1
    if required_colors > len(COLORS):
        return None
    sampled = random.sample(COLORS, required_colors)
    fence_color = sampled[0]
    texture_colors = sampled[1 : 1 + legend_count]
    seed_colors = sampled[1 + legend_count :]

    legend_specs: list[dict[str, object]] = []
    legend_lookup: dict[int, np.ndarray] = {}
    for size, left, texture_color, seed_color in zip(
        legend_sizes, legend_lefts, texture_colors, seed_colors, strict=False
    ):
        mask, seed_col = _sample_texture(size)
        legend_specs.append(
            {
                "top": legend_top,
                "left": left,
                "texture": mask,
                "texture_color": texture_color,
                "seed_color": seed_color,
                "seed_col": seed_col,
            }
        )
        legend_lookup[seed_color] = np.where(mask != 0, texture_color, 0)

    fence_boxes = _place_fences(height, width, lower_top, fence_count)
    if fence_boxes is None:
        return None

    fence_specs: list[dict[str, object]] = []
    for r0, c0, inner_h, inner_w in fence_boxes:
        seed_color = random.choice(seed_colors)
        sr = random.randint(r0 + 1, r0 + inner_h)
        sc = random.randint(c0 + 1, c0 + inner_w)
        fence_specs.append(
            {
                "r0": r0,
                "c0": c0,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_r": sr,
                "seed_c": sc,
                "seed_color": seed_color,
                "texture": legend_lookup[seed_color],
            }
        )

    inp, expected = _render_pair(height, width, legend_top, legend_specs, fence_specs, fence_color)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
