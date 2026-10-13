from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    crop[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return crop


def _strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _parse_input(grid: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    infos = []
    for coords in _components(g != BLACK):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        crop = _crop_mask(coords, box)
        values = sorted({int(g[r, c]) for r, c in coords})
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        infos.append(
            {
                "coords": coords,
                "bbox": box,
                "mask": crop,
                "colors": values,
                "h": h,
                "w": w,
                "area": int(coords.shape[0]),
            }
        )

    assert infos, "expected non-black objects"

    palette_candidates = [
        info
        for info in infos
        if min(info["h"], info["w"]) == 1
        and info["area"] == max(info["h"], info["w"])
        and len(info["colors"]) >= 2
    ]
    assert palette_candidates, "expected a multi-color palette strip"

    palette = max(
        palette_candidates,
        key=lambda info: (info["bbox"][1] + info["bbox"][3], len(info["colors"]), -info["bbox"][0]),
    )

    pr0, pc0, pr1, pc1 = palette["bbox"]
    if palette["h"] == 1:
        palette_colors = [int(g[pr0, c]) for c in range(pc0, pc1 + 1)]
    else:
        palette_colors = [int(g[r, pc0]) for r in range(pr0, pr1 + 1)]

    layers = [info for info in infos if info is not palette]
    assert len(layers) == len(palette_colors), "palette length must match layer count"

    contain_counts = []
    for i, info in enumerate(layers):
        count = sum(
            1
            for j, other in enumerate(layers)
            if i != j and _strict_contains(info["bbox"], other["bbox"])
        )
        contain_counts.append(count)

    assert sorted(contain_counts, reverse=True) == list(range(len(layers) - 1, -1, -1)), "layers must be strictly nested"

    ordered_layers = [
        info
        for _, info in sorted(
            zip(contain_counts, layers),
            key=lambda pair: (-pair[0], pair[1]["bbox"][0], pair[1]["bbox"][1]),
        )
    ]

    layer_masks = [info["mask"] for info in ordered_layers]
    return layer_masks, palette_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a nested frame stack into side-by-side layers and recolor them using the palette strip order.
    """
    layer_masks, palette_colors = _parse_input(np.asarray(grid, dtype=int))

    heights = [mask.shape[0] for mask in layer_masks]
    widths = [mask.shape[1] for mask in layer_masks]
    out_h = max(heights)
    out_w = sum(widths) + max(0, len(widths) - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for mask, color in zip(layer_masks, palette_colors):
        h, w = mask.shape
        out[:h, col : col + w][mask] = int(color)
        col += w + 1

    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _sample_inner_size(layer_count: int, inner_kind: str) -> tuple[int, int]:
    if inner_kind == "pixel":
        return 1, 1

    if layer_count == 4:
        max_side = 2
    elif layer_count == 3:
        max_side = 3
    else:
        max_side = 4

    return random.randint(2, max_side), random.randint(2, max_side)


def _build_candidate() -> np.ndarray:
    layer_count = random.choice((2, 3, 4))
    inner_kind = random.choice(("pixel", "solid"))
    palette_orientation = random.choice(("horizontal", "vertical"))

    inner_h, inner_w = _sample_inner_size(layer_count, inner_kind)
    outer_h = inner_h + 4 * (layer_count - 1)
    outer_w = inner_w + 4 * (layer_count - 1)

    colors = random.sample(COLORS, 2 * layer_count)
    layer_colors = colors[:layer_count]
    palette_colors = colors[layer_count:]

    palette_h, palette_w = (1, layer_count) if palette_orientation == "horizontal" else (layer_count, 1)

    min_h = max(outer_h, palette_h) + 4
    min_w = max(2 * outer_w + 4, outer_w + palette_w + 8, 2 * palette_w + 4)
    grid_h = random.randint(min_h, min_h + 7)
    grid_w = random.randint(min_w, min_w + 9)
    mid = grid_w // 2

    max_left = mid - outer_w - 1
    min_palette_left = mid + 1
    max_palette_left = grid_w - palette_w - 1
    if max_left < 1 or min_palette_left > max_palette_left:
        raise ValueError("layout too tight")

    outer_top = random.randint(1, grid_h - outer_h - 1)
    outer_left = random.randint(1, max_left)
    palette_top = random.randint(1, grid_h - palette_h - 1)
    palette_left = random.randint(min_palette_left, max_palette_left)

    grid = np.zeros((grid_h, grid_w), dtype=int)

    for depth in range(layer_count - 1):
        top = outer_top + 2 * depth
        left = outer_left + 2 * depth
        height = outer_h - 4 * depth
        width = outer_w - 4 * depth
        _draw_frame(grid, top, left, height, width, layer_colors[depth])

    inner_top = outer_top + 2 * (layer_count - 1)
    inner_left = outer_left + 2 * (layer_count - 1)
    grid[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = layer_colors[-1]

    if palette_orientation == "horizontal":
        for offset, color in enumerate(palette_colors):
            grid[palette_top, palette_left + offset] = color
    else:
        for offset, color in enumerate(palette_colors):
            grid[palette_top + offset, palette_left] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        try:
            layer_masks, palette_colors = _parse_input(grid)
        except AssertionError:
            continue

        if len(layer_masks) != len(palette_colors):
            continue
        if any(len(set(np.argwhere(mask)[:, 0])) == 1 and len(set(np.argwhere(mask)[:, 1])) > 1 for mask in layer_masks[-1:]):
            continue
        return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
