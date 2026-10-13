from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np


BG = 0
BORDER = 5

LEGEND_H = 5
LEGEND_W = 8
LEGEND_DIVIDER = 4

MIN_GRID_H = 24
MAX_GRID_H = 30
MAX_GRID_W = 30
GEN_RETRIES = 300

SHAPE_LIBRARY = (
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
)


def _rotate_cw(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(mask, -quarter_turns)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("cannot normalize an empty mask")
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _components4(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _grey_components(grid: np.ndarray) -> List[Dict[str, object]]:
    components: List[Dict[str, object]] = []
    for coords in _components4(grid == BORDER):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local[r - r0, c - c0] = True
        components.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": local,
            }
        )
    return components


def _legend_divider(local_mask: np.ndarray) -> int | None:
    hh, ww = local_mask.shape
    if hh < 5 or ww < 7:
        return None
    frame = _frame_mask(hh, ww)
    if not np.all(local_mask[frame]):
        return None
    divider_cols = [c for c in range(1, ww - 1) if np.all(local_mask[:, c])]
    if len(divider_cols) != 1:
        return None
    divider = divider_cols[0]
    expected = frame.copy()
    expected[:, divider] = True
    if not np.array_equal(local_mask, expected):
        return None
    if divider - 1 < 3 or ww - divider - 2 < 1 or hh - 2 < 3:
        return None
    return divider


def _is_plain_frame(local_mask: np.ndarray) -> bool:
    hh, ww = local_mask.shape
    if hh < 3 or ww < 3:
        return False
    return np.array_equal(local_mask, _frame_mask(hh, ww))


def _find_lookup_colors(grid: np.ndarray) -> Dict[int, int]:
    h, w = grid.shape
    candidates: List[Tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            vals = [int(v) for v in block.ravel()]
            if BG in vals or BORDER in vals or len(set(vals)) != 4:
                continue
            rr0, rr1 = max(0, r - 1), min(h, r + 3)
            cc0, cc1 = max(0, c - 1), min(w, c + 3)
            ring = grid[rr0:rr1, cc0:cc1]
            ring_mask = np.ones(ring.shape, dtype=bool)
            ring_mask[r - rr0 : r - rr0 + 2, c - cc0 : c - cc0 + 2] = False
            if np.any(ring[ring_mask] != BG):
                continue
            candidates.append((r, c))
    assert len(candidates) == 1, "expected exactly one isolated 2x2 lookup table"
    r, c = candidates[0]
    block = grid[r : r + 2, c : c + 2]
    return {
        0: int(block[0, 0]),
        1: int(block[0, 1]),
        2: int(block[1, 1]),
        3: int(block[1, 0]),
    }


def _find_legends(grid: np.ndarray) -> Dict[int, Tuple[np.ndarray, int]]:
    legends: Dict[int, Tuple[np.ndarray, int]] = {}
    for comp in _grey_components(grid):
        divider = _legend_divider(comp["mask"])  # type: ignore[arg-type]
        if divider is None:
            continue
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[assignment]
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        left_area = sub[1:-1, 1:divider]
        right_area = sub[1:-1, divider + 1 : -1]

        seed_colors = {int(v) for v in np.unique(left_area) if v not in (BG, BORDER)}
        fill_colors = {int(v) for v in np.unique(right_area) if v not in (BG, BORDER)}

        if len(seed_colors) != 1 or len(fill_colors) != 1:
            continue
        seed_color = seed_colors.pop()
        fill_color = fill_colors.pop()
        if fill_color == seed_color:
            continue
        if np.any((left_area != BG) & (left_area != seed_color)):
            continue
        if not np.all(right_area == fill_color):
            continue

        seed_mask = left_area == seed_color
        if len(_components4(seed_mask)) != 1:
            continue
        legends[seed_color] = (_normalize_mask(seed_mask), fill_color)
    assert legends, "expected at least one legend cell"
    return legends


def _find_region_frames(grid: np.ndarray) -> List[Tuple[int, int, int, int]]:
    frames: List[Tuple[int, int, int, int]] = []
    for comp in _grey_components(grid):
        local_mask = comp["mask"]  # type: ignore[assignment]
        if _is_plain_frame(local_mask):
            frames.append(comp["bbox"])  # type: ignore[arg-type]
    assert frames, "expected at least one closed grey frame"
    return frames


def _infer_rotation(template_mask: np.ndarray, observed_mask: np.ndarray) -> int:
    observed_norm = _normalize_mask(observed_mask)
    matches = [
        quarter_turns
        for quarter_turns in range(4)
        if np.array_equal(_normalize_mask(_rotate_cw(template_mask, quarter_turns)), observed_norm)
    ]
    assert len(matches) == 1, "seed orientation must match the legend under exactly one rotation"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region using the legend entry with the same seed color, then
    choose corner-marker colors from the top-right 2x2 rotation lookup table.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    lookup_colors = _find_lookup_colors(g)
    legends = _find_legends(g)
    frames = _find_region_frames(g)

    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors_inside = {int(v) for v in np.unique(interior) if v not in (BG, BORDER)}
        assert len(colors_inside) == 1, "each framed region must contain exactly one seed color"
        seed_color = colors_inside.pop()
        assert seed_color in legends, "every seed color must appear in the bottom legend"

        template_mask, fill_color = legends[seed_color]
        seed_mask = interior == seed_color
        assert len(_components4(seed_mask)) == 1, "seed shape must be a single 4-connected component"
        rotation = _infer_rotation(template_mask, seed_mask)
        marker_color = lookup_colors[rotation]

        target = out[r0 + 1 : r1, c0 + 1 : c1]
        target[interior == BG] = fill_color

        for rr, cc in ((r0 + 1, c0 + 1), (r0 + 1, c1 - 1), (r1 - 1, c1 - 1), (r1 - 1, c0 + 1)):
            if g[rr, cc] == BG:
                out[rr, cc] = marker_color

    return out


def _draw_rect_frame(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top, left : left + width] = BORDER
    grid[top + height - 1, left : left + width] = BORDER
    grid[top : top + height, left] = BORDER
    grid[top : top + height, left + width - 1] = BORDER


def _reserve(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    mask[r0:r1, c0:c1] = True


def _place_lookup_table(grid: np.ndarray, colors: List[int]) -> Tuple[int, int]:
    _, w = grid.shape
    top = 1
    left = w - 3
    grid[top, left] = colors[0]
    grid[top, left + 1] = colors[1]
    grid[top + 1, left + 1] = colors[2]
    grid[top + 1, left] = colors[3]
    return top, left


def _place_legend(
    grid: np.ndarray,
    top: int,
    left: int,
    seed_color: int,
    template_mask: np.ndarray,
    fill_color: int,
) -> None:
    _draw_rect_frame(grid, top, left, LEGEND_H, LEGEND_W)
    grid[top : top + LEGEND_H, left + LEGEND_DIVIDER] = BORDER

    left_patch = grid[top + 1 : top + 4, left + 1 : left + 4]
    left_patch[template_mask] = seed_color
    grid[top + 1 : top + 4, left + LEGEND_DIVIDER + 1 : left + LEGEND_W - 1] = fill_color


def _place_region(
    grid: np.ndarray,
    rng: random.Random,
    top: int,
    left: int,
    height: int,
    width: int,
    seed_color: int,
    template_mask: np.ndarray,
    rotation: int,
) -> None:
    _draw_rect_frame(grid, top, left, height, width)

    interior_h = height - 2
    interior_w = width - 2
    seed_top = rng.randint(1, interior_h - 4)
    seed_left = rng.randint(1, interior_w - 4)
    patch = grid[
        top + 1 + seed_top : top + 1 + seed_top + 3,
        left + 1 + seed_left : left + 1 + seed_left + 3,
    ]
    patch[_rotate_cw(template_mask, rotation)] = seed_color


def _sample_colors(rng: random.Random, n_regions: int) -> Tuple[List[int], List[int], List[int], int]:
    usable = [1, 2, 3, 4, 6, 7, 8, 9]
    seed_colors = rng.sample(usable, n_regions)
    remaining = [c for c in usable if c not in seed_colors]
    lookup_colors = rng.sample(remaining, 4)
    fill_shift = rng.choice((1, 2, 3))
    return seed_colors, lookup_colors, remaining, fill_shift


def _region_size_limits(n_regions: int) -> Tuple[int, int]:
    if n_regions == 2:
        return 7, 9
    return 7, 8


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(2, 3)
    gaps = [rng.randint(1, 2) for _ in range(n_regions - 1)]
    legend_span = n_regions * LEGEND_W + sum(gaps)
    min_w = max(24, legend_span + 2)
    if min_w > MAX_GRID_W:
        return None

    h = rng.randint(MIN_GRID_H, MAX_GRID_H)
    w = rng.randint(min_w, MAX_GRID_W)
    grid = np.full((h, w), BG, dtype=int)

    seed_colors, lookup_colors, _remaining, fill_shift = _sample_colors(rng, n_regions)
    rotations = rng.sample(range(4), n_regions)
    fill_colors = [lookup_colors[(rotation + fill_shift) % 4] for rotation in rotations]

    shape_indices = rng.sample(range(len(SHAPE_LIBRARY)), n_regions)
    template_masks = [SHAPE_LIBRARY[idx] for idx in shape_indices]

    table_top, table_left = _place_lookup_table(grid, lookup_colors)
    occupied = np.zeros((h, w), dtype=bool)
    _reserve(occupied, table_top, table_left, 2, 2, pad=1)

    legend_top = h - LEGEND_H - 1
    start_col = rng.randint(1, w - legend_span - 1)
    legend_lefts: List[int] = []
    cursor = start_col
    for idx in range(n_regions):
        legend_lefts.append(cursor)
        cursor += LEGEND_W
        if idx < len(gaps):
            cursor += gaps[idx]

    for seed_color, template_mask, fill_color, legend_left in zip(
        seed_colors, template_masks, fill_colors, legend_lefts
    ):
        _place_legend(grid, legend_top, legend_left, seed_color, template_mask, fill_color)
        _reserve(occupied, legend_top, legend_left, LEGEND_H, LEGEND_W, pad=1)

    min_size, max_size = _region_size_limits(n_regions)
    region_bottom = legend_top - 2
    if region_bottom - 1 < min_size:
        return None

    for seed_color, template_mask, rotation in zip(seed_colors, template_masks, rotations):
        placed = False
        for _ in range(200):
            height = rng.randint(min_size, max_size)
            width = rng.randint(min_size, max_size + 1)
            if region_bottom - height + 1 < 1 or w - width - 1 < 1:
                continue
            top = rng.randint(1, region_bottom - height + 1)
            left = rng.randint(1, w - width - 1)
            if np.any(occupied[top - 1 : top + height + 1, left - 1 : left + width + 1]):
                continue
            _place_region(grid, rng, top, left, height, width, seed_color, template_mask, rotation)
            _reserve(occupied, top, left, height, width, pad=1)
            placed = True
            break
        if not placed:
            return None

    legends = _find_legends(grid)
    frames = _find_region_frames(grid)
    if len(legends) != n_regions or len(frames) != n_regions:
        return None
    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate random, non-overlapping ARC inputs for the seed-rotation fill task."""
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")


for _shape in SHAPE_LIBRARY:
    rotation_sigs = {
        _normalize_mask(_rotate_cw(_shape, quarter_turns)).astype(np.uint8).tobytes()
        for quarter_turns in range(4)
    }
    assert len(rotation_sigs) == 4, "shape library must contain only fully orientation-sensitive templates"
