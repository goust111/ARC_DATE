from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
GENERATION_RETRIES = 500
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_unit_frame(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(sub, expected)


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _render_output(
    sizes: list[tuple[int, int]],
    frame_colors: list[int],
    fill_colors: list[int],
) -> np.ndarray:
    out_h = max(h for h, _ in sizes)
    out_w = sum(w for _, w in sizes) + len(sizes) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)

    col = 0
    for (height, width), border_color, fill_color in zip(sizes, frame_colors, fill_colors):
        out[:height, col : col + width] = fill_color
        _draw_frame(out, 0, col, height, width, border_color)
        col += width + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll a nested frame stack into separate horizontal frames whose fills come
    from the palette blocks on the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    sep_cols = []
    for c in range(w):
        color = int(g[0, c])
        if color == BLACK:
            continue
        if not np.all(g[:, c] == color):
            continue
        has_left = c > 0 and bool(np.any(g[:, :c] != BLACK))
        has_right = c + 1 < w and bool(np.any(g[:, c + 1 :] != BLACK))
        if has_left and has_right:
            sep_cols.append(c)
    assert len(sep_cols) == 1, "expected exactly one full-height separator column"
    sep_col = sep_cols[0]

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    frame_infos = []
    for coords in _components(left != BLACK):
        bbox = _bbox(coords)
        comp_mask = np.zeros_like(left, dtype=bool)
        comp_mask[tuple(coords.T)] = True
        assert _is_unit_frame(comp_mask, bbox), "left-side object must be a 1-pixel frame"
        r0, c0, r1, c1 = bbox
        frame_infos.append(
            {
                "bbox": bbox,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "color": int(left[coords[0, 0], coords[0, 1]]),
            }
        )
    assert 2 <= len(frame_infos) <= 4, "expected 2-4 nested frames"
    frame_infos.sort(key=lambda item: item["height"] * item["width"], reverse=True)

    for outer, inner in zip(frame_infos, frame_infos[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1, "frames must be strictly nested"

    palette_infos = []
    for coords in _components(right != BLACK):
        r0, c0, r1, c1 = bbox = _bbox(coords)
        block = right[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(block == block[0, 0]), "palette block must be solid"
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        assert bh == bw and bh in (1, 2), "palette block must be 1x1 or 2x2"
        palette_infos.append({"top": r0, "left": c0, "color": int(block[0, 0])})
    palette_infos.sort(key=lambda item: (item["top"], item["left"]))
    assert len(palette_infos) == len(frame_infos), "palette length must match frame depth"

    sizes = [(int(info["height"]), int(info["width"])) for info in frame_infos]
    frame_colors = [int(info["color"]) for info in frame_infos]
    fill_colors = [int(info["color"]) for info in palette_infos]
    return _render_output(sizes, frame_colors, fill_colors)


def _gen_attempt() -> np.ndarray | None:
    depth = random.randint(2, 4)
    color_order = random.sample(list(USABLE_COLORS), 2 * depth)
    frame_colors = color_order[:depth]
    fill_colors = color_order[depth:]

    inner_h = random.randint(3, 6)
    inner_w = random.randint(3, 6)

    layers = [{"r": 0, "c": 0, "h": inner_h, "w": inner_w}]
    cur_h, cur_w = inner_h, inner_w
    for _ in range(depth - 1):
        top_gap = random.randint(1, 3)
        bottom_gap = random.randint(1, 3)
        left_gap = random.randint(1, 3)
        right_gap = random.randint(1, 3)
        shift_r = 1 + top_gap
        shift_c = 1 + left_gap
        for layer in layers:
            layer["r"] += shift_r
            layer["c"] += shift_c
        cur_h = cur_h + top_gap + bottom_gap + 2
        cur_w = cur_w + left_gap + right_gap + 2
        layers.insert(0, {"r": 0, "c": 0, "h": cur_h, "w": cur_w})

    outer_h = int(layers[0]["h"])
    outer_w = int(layers[0]["w"])

    palette_sides = [random.choice((1, 2)) for _ in range(depth)]
    gap_values = [random.randint(1, 3) for _ in range(depth - 1)]
    palette_height = sum(palette_sides) + sum(gap_values)
    palette_width = max(palette_sides)

    grid_h = max(outer_h, palette_height) + random.randint(3, 7)
    left_pad_top = random.randint(1, grid_h - outer_h - 1)
    right_pad_top = random.randint(1, grid_h - palette_height - 1)

    left_margin = random.randint(1, 4)
    left_right_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    right_right_margin = random.randint(1, 4)
    left_region_w = left_margin + outer_w + left_right_margin
    right_region_w = right_margin + palette_width + right_right_margin

    grid = np.full((grid_h, left_region_w + 1 + right_region_w), BLACK, dtype=int)
    grid[:, left_region_w] = GREY

    for layer, color in zip(layers, frame_colors):
        _draw_frame(
            grid,
            left_pad_top + int(layer["r"]),
            left_margin + int(layer["c"]),
            int(layer["h"]),
            int(layer["w"]),
            int(color),
        )

    palette_col = left_region_w + 1 + right_margin
    cursor = right_pad_top
    for idx, (side, color) in enumerate(zip(palette_sides, fill_colors)):
        grid[cursor : cursor + side, palette_col : palette_col + side] = color
        if idx + 1 < depth:
            cursor += side + gap_values[idx]

    expected = _render_output(
        [(int(layer["h"]), int(layer["w"])) for layer in layers],
        frame_colors,
        fill_colors,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
