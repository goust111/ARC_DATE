from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
FRAME = 5
OUTER_LEGEND_W = 4
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _decode_legend(inner: np.ndarray, bg: int) -> Dict[int, int]:
    """Decode stacked two-column legend entries into color -> length."""
    if inner.ndim != 2 or inner.shape[1] != 2 or inner.shape[0] < 1:
        raise ValueError("legend interior must be Hx2")

    mapping: Dict[int, int] = {}
    row = 0
    height = int(inner.shape[0])
    while row < height:
        color = int(inner[row, 0])
        if color == bg:
            raise ValueError("each entry must start with a visible key color")
        if int(inner[row, 1]) != color:
            raise ValueError("entry bar must start beside its key")
        if color in mapping:
            raise ValueError("legend colors must be unique")

        length = 0
        while row + length < height and int(inner[row + length, 1]) == color:
            left_val = int(inner[row + length, 0])
            if left_val not in (bg, color):
                raise ValueError("entry body may only contain background or its own color")
            length += 1

        if length < 1 or length > 5:
            raise ValueError("legend lengths must be between 1 and 5")
        mapping[color] = length
        row += length

    if len(mapping) < 2:
        raise ValueError("legend should contain at least two colors")
    return mapping


def _find_legend(grid: np.ndarray, bg: int) -> Tuple[Tuple[int, int, int, int], Dict[int, int]]:
    """Find the top-left framed 2-column legend and decode it."""
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    best: Tuple[Tuple[int, int, int, int], Dict[int, int]] | None = None

    for r0 in range(h - 2):
        for c0 in range(w - OUTER_LEGEND_W + 1):
            frame = int(g[r0, c0])
            if frame == bg:
                continue
            c1 = c0 + OUTER_LEGEND_W - 1
            for r1 in range(r0 + 2, h):
                if not np.all(g[r0, c0 : c1 + 1] == frame):
                    break
                if not np.all(g[r1, c0 : c1 + 1] == frame):
                    continue
                if not np.all(g[r0 : r1 + 1, c0] == frame):
                    continue
                if not np.all(g[r0 : r1 + 1, c1] == frame):
                    continue

                inner = g[r0 + 1 : r1, c0 + 1 : c1]
                if inner.shape[1] != 2 or inner.shape[0] < 1:
                    continue
                if np.any(inner == frame):
                    continue

                try:
                    mapping = _decode_legend(inner, bg)
                except ValueError:
                    continue

                bbox = (int(r0), int(c0), int(r1), int(c1))
                if best is None or bbox[:2] < best[0][:2] or (bbox[:2] == best[0][:2] and bbox[2] > best[0][2]):
                    best = (bbox, mapping)

    if best is None:
        raise ValueError("failed to locate framed legend")
    return best


def _seed_direction(r: int, c: int, h: int, w: int) -> Tuple[int, int]:
    if r == 0:
        return 1, 0
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    if r == h - 1:
        return -1, 0
    raise ValueError("seed is not on the boundary")


def _band_cells(r: int, c: int, dr: int, dc: int, length: int, h: int, w: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for step in range(length):
        rr = int(r + dr * step)
        cc = int(c + dc * step)
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a framed top-left legend and grow the matching boundary seeds inward.

    Input structure:
    - The background is black.
    - Near the top-left corner there is a gray framed legend whose 2-column interior is stacked top-to-bottom.
      Each entry starts with a color key in the left inner column, and the same color forms a contiguous vertical
      run in the right inner column.
    - Outside the legend, single-pixel colored seeds lie on the outer border; top/bottom seeds indicate columns,
      and left/right seeds indicate rows.

    Rule:
    - Read each legend entry as a color-to-length mapping, where the right-column run height is the band length.
    - Scan all boundary cells in row-major order; whenever a boundary seed color appears in the legend, draw a
      1-pixel-wide band of that color starting at the seed and extending inward perpendicular to that edge.
    - Clip bands at the grid boundary, and let later bands overwrite earlier ones when they share cells.
    - Clear the whole legend box to background in the output.

    Key insight:
    - The legend's right-column run heights are the hidden numeric code; solving the task requires first decoding
      those visible heights into lengths, then applying edge-dependent inward growth with row-major overwrite order.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    h, w = map(int, g.shape)
    bg = BG

    (r0, c0, r1, c1), mapping = _find_legend(g, bg)
    out = np.full_like(g, bg)

    seeds: List[Tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            color = int(g[r, c])
            if color == bg or color not in mapping:
                continue
            seeds.append((r, c, color))

    for r, c, color in seeds:
        dr, dc = _seed_direction(r, c, h, w)
        for rr, cc in _band_cells(r, c, dr, dc, mapping[color], h, w):
            out[rr, cc] = color

    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return out


def _draw_legend(grid: np.ndarray, r0: int, c0: int, colors: Sequence[int], lengths: Sequence[int]) -> Tuple[int, int, int, int]:
    inner_h = int(sum(lengths))
    r1 = int(r0 + inner_h + 1)
    c1 = int(c0 + OUTER_LEGEND_W - 1)
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME

    row = r0 + 1
    for color, length in zip(colors, lengths, strict=True):
        grid[row, c0 + 1] = int(color)
        grid[row : row + length, c0 + 2] = int(color)
        row += int(length)
    return r0, c0, r1, c1


def _available_slots(h: int, w: int, legend_bbox: Tuple[int, int, int, int]) -> Dict[str, List[int]]:
    r0, c0, r1, c1 = legend_bbox
    legend_rows = set(range(r0, r1 + 1))
    legend_cols = set(range(c0, c1 + 1))
    return {
        "top": [c for c in range(1, w - 1) if c not in legend_cols],
        "bottom": [c for c in range(1, w - 1) if c not in legend_cols],
        "left": [r for r in range(1, h - 1) if r not in legend_rows],
        "right": [r for r in range(1, h - 1) if r not in legend_rows],
    }


def _place_seed(
    grid: np.ndarray,
    used: set[Tuple[int, int]],
    edge: str,
    idx: int,
    color: int,
) -> Tuple[int, int, str, int]:
    h, w = map(int, grid.shape)
    if edge == "top":
        pos = (0, idx)
    elif edge == "bottom":
        pos = (h - 1, idx)
    elif edge == "left":
        pos = (idx, 0)
    elif edge == "right":
        pos = (idx, w - 1)
    else:
        raise ValueError(edge)
    if pos in used:
        raise ValueError("duplicate seed position")
    used.add(pos)
    grid[pos] = int(color)
    return pos[0], pos[1], edge, int(color)


def _seed_band_from_spec(seed: Tuple[int, int, str, int], mapping: Dict[int, int], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    r, c, edge, color = seed
    h, w = shape
    if edge == "top":
        dr, dc = 1, 0
    elif edge == "bottom":
        dr, dc = -1, 0
    elif edge == "left":
        dr, dc = 0, 1
    else:
        dr, dc = 0, -1
    return _band_cells(r, c, dr, dc, mapping[color], h, w)


def _has_overlap(
    seeds: Sequence[Tuple[int, int, str, int]],
    mapping: Dict[int, int],
    shape: Tuple[int, int],
    legend_bbox: Tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = legend_bbox
    bands = []
    for seed in seeds:
        cells = {
            cell
            for cell in _seed_band_from_spec(seed, mapping, shape)
            if not (r0 <= cell[0] <= r1 and c0 <= cell[1] <= c1)
        }
        bands.append(cells)
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            if bands[i] & bands[j]:
                return True
    return False


def _has_truncation(seeds: Sequence[Tuple[int, int, str, int]], mapping: Dict[int, int], shape: Tuple[int, int]) -> bool:
    for seed in seeds:
        cells = _seed_band_from_spec(seed, mapping, shape)
        if len(cells) < mapping[seed[3]]:
            return True
    return False


def _make_cross_case() -> np.ndarray | None:
    colors = random.sample(PALETTE, random.randint(3, 5))
    lengths = [random.randint(1, 5) for _ in colors]
    long_idxs = [i for i, length in enumerate(lengths) if length >= 2]
    if len(long_idxs) < 2:
        need = 2 - len(long_idxs)
        for idx in random.sample(range(len(lengths)), need):
            lengths[idx] = max(lengths[idx], 2)

    r0 = random.randint(0, 1)
    c0 = random.randint(0, 1)
    inner_h = int(sum(lengths))
    outer_h = inner_h + 2
    min_h = r0 + outer_h + 2
    min_w = 6 if c0 == 0 else 7
    if min_h > 30 or min_w > 30:
        return None
    h = random.randint(min_h, min(30, min_h + 10))
    w = random.randint(min_w, min(30, min_w + 10))

    grid = np.full((h, w), BG, dtype=int)
    legend_bbox = _draw_legend(grid, r0, c0, colors, lengths)
    mapping = dict(zip(colors, lengths, strict=True))
    slots = _available_slots(h, w, legend_bbox)
    if not slots["bottom"] or not slots["right"]:
        return None

    used: set[Tuple[int, int]] = set()
    usable_colors = [color for color, length in mapping.items() if length >= 2]
    if len(usable_colors) < 2:
        return None
    vcolor, hcolor = random.sample(usable_colors, 2)
    lv = mapping[vcolor]
    lh = mapping[hcolor]

    row_candidates = [r for r in slots["right"] if (h - 1 - r) < lv]
    col_candidates = [c for c in slots["bottom"] if (w - 1 - c) < lh]
    if not row_candidates or not col_candidates:
        return None

    seeds: List[Tuple[int, int, str, int]] = []
    seeds.append(_place_seed(grid, used, "bottom", random.choice(col_candidates), vcolor))
    seeds.append(_place_seed(grid, used, "right", random.choice(row_candidates), hcolor))

    for edge in ("top", "left"):
        if slots[edge] and random.random() < 0.7:
            idx = random.choice(slots[edge])
            color = random.choice(colors)
            if edge == "top" and (0, idx) not in used:
                seeds.append(_place_seed(grid, used, edge, idx, color))
            if edge == "left" and (idx, 0) not in used:
                seeds.append(_place_seed(grid, used, edge, idx, color))

    extra_count = random.randint(0, 4)
    all_edges = ["top", "bottom", "left", "right"]
    for _ in range(extra_count):
        edge = random.choice(all_edges)
        choices = [idx for idx in slots[edge] if ((0, idx) if edge == "top" else (h - 1, idx) if edge == "bottom" else (idx, 0) if edge == "left" else (idx, w - 1)) not in used]
        if not choices:
            continue
        seeds.append(_place_seed(grid, used, edge, random.choice(choices), random.choice(colors)))

    if not _has_overlap(seeds, mapping, grid.shape, legend_bbox):
        return None
    return grid


def _make_narrow_case() -> np.ndarray | None:
    colors = random.sample(PALETTE, random.randint(2, 4))
    lengths = [random.randint(1, 5) for _ in colors]
    long_idx = random.randrange(len(lengths))
    lengths[long_idx] = 5

    r0 = random.randint(0, 1)
    c0 = 0
    inner_h = int(sum(lengths))
    outer_h = inner_h + 2
    min_h = r0 + outer_h + 2
    if min_h > 30:
        return None

    h = random.randint(min_h, min(30, min_h + 8))
    w = 4
    grid = np.full((h, w), BG, dtype=int)
    legend_bbox = _draw_legend(grid, r0, c0, colors, lengths)
    mapping = dict(zip(colors, lengths, strict=True))
    slots = _available_slots(h, w, legend_bbox)
    if not slots["left"] or not slots["right"]:
        return None

    used: set[Tuple[int, int]] = set()
    seeds: List[Tuple[int, int, str, int]] = []
    row = random.choice(slots["left"])

    left_color = colors[long_idx]
    right_choices = [color for color in colors if color != left_color and mapping[color] >= 2]
    if not right_choices:
        right_choices = [left_color]
    right_color = random.choice(right_choices)

    seeds.append(_place_seed(grid, used, "left", row, left_color))
    seeds.append(_place_seed(grid, used, "right", row, right_color))

    for _ in range(random.randint(0, 3)):
        edge = random.choice(("left", "right"))
        choices = [idx for idx in slots[edge] if (idx, 0) not in used] if edge == "left" else [idx for idx in slots[edge] if (idx, w - 1) not in used]
        if not choices:
            continue
        seeds.append(_place_seed(grid, used, edge, random.choice(choices), random.choice(colors)))

    if not _has_truncation(seeds, mapping, grid.shape):
        return None
    if not _has_overlap(seeds, mapping, grid.shape, legend_bbox):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        mode = "narrow" if random.random() < 0.35 else "cross"
        grid = _make_narrow_case() if mode == "narrow" else _make_cross_case()
        if grid is None:
            continue
        out = main(grid)
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
