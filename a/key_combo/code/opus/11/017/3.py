from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 2
SOURCE_COLORS = (1, 3, 7)
FILL_COLORS = (4, 6, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array in (r,c) order."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    comps: list[np.ndarray] = []
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _legend_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    h, w = grid.shape
    assert w >= 2
    for r in range(h):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left == BG and right == BG:
            break
        assert left in SOURCE_COLORS and right in FILL_COLORS
        pairs.append((left, right))
    assert 2 <= len(pairs) <= 3
    assert len({src for src, _ in pairs}) == len(pairs)
    return pairs


def _marker_rectangles(grid: np.ndarray, pairs: list[tuple[int, int]]) -> list[tuple[int, int, int, int, int, int]]:
    legend_rows = len(pairs)
    rects: list[tuple[int, int, int, int, int, int]] = []
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[:legend_rows, :2] = True
    for priority, (source, fill) in enumerate(pairs):
        coords = np.argwhere((grid == source) & ~legend_mask)
        assert coords.shape == (2, 2)
        r0, c0, r1, c1 = bbox_coords(coords)
        assert r1 - r0 >= 4 and c1 - c0 >= 4
        rects.append((priority, source, fill, r0, c0, r1, c1))
    return rects


def _enclosed_mask_for_rect(grid: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    background = sub == BG
    comps = _components(background)
    out = np.zeros_like(grid, dtype=bool)
    if len(comps) <= 1:
        return out

    sizes = [int(comp.shape[0]) for comp in comps]
    max_size = max(sizes)
    for comp, size in zip(comps, sizes):
        if size == max_size:
            continue
        rr = comp[:, 0] + r0
        cc = comp[:, 1] + c0
        out[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the closed background chambers defined by diagonal marker rectangles using the legend colors.

    Input structure:
    - A black background grid has a 2-column legend in the upper-left corner; each legend row maps one
      source color to one fill color, and legend order defines overlap priority.
    - Outside the legend, each source color appears as exactly two same-color diagonal markers that define
      one axis-aligned rectangle, while red pixels act as one-cell-thick wall segments.
    - Rectangles may overlap, and the red walls may carve several background chambers inside a rectangle.

    Rule:
    - Read the legend rows from top to bottom and keep that row order as the priority order.
    - For each source color, use its two non-legend marker pixels as opposite corners of a rectangle.
    - Inside that rectangle, treat red wall pixels together with the rectangle boundary as separators of the
      black background into connected chambers; keep the unique largest background chamber unchanged and mark
      every smaller chamber as enclosed.
    - Fill each enclosed chamber with the rectangle's legend-mapped fill color, painting legend rows from
      bottom to top so earlier legend rows overwrite lower-priority overlaps.

    Key insight:
    - The same red wall pattern must be interpreted relative to a marker-defined rectangle; once that local
      rectangle is known, the smaller black chambers inside it are exactly the areas that should be colored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pairs = _legend_pairs(g)
    rects = _marker_rectangles(g, pairs)

    out = g.copy()
    for _priority, _source, fill, r0, c0, r1, c1 in reversed(rects):
        enclosed = _enclosed_mask_for_rect(g, (r0, c0, r1, c1))
        out[enclosed] = fill
    return out


def _marker_cells(rect: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    if rect["diag"] == "tlbr":
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _chamber_bounds(rect: dict, corner: str, depth_r: int, depth_c: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    if corner == "tl":
        return r0 + 1, c0 + 1, r0 + depth_r - 1, c0 + depth_c - 1
    if corner == "tr":
        return r0 + 1, c1 - depth_c + 1, r0 + depth_r - 1, c1 - 1
    if corner == "bl":
        return r1 - depth_r + 1, c0 + 1, r1 - 1, c0 + depth_c - 1
    return r1 - depth_r + 1, c1 - depth_c + 1, r1 - 1, c1 - 1


def _l_wall_cells(rect: dict, corner: str, depth_r: int, depth_c: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    cells: list[tuple[int, int]] = []
    if corner == "tl":
        cells.extend((r, c0 + depth_c) for r in range(r0, r0 + depth_r + 1))
        cells.extend((r0 + depth_r, c) for c in range(c0, c0 + depth_c + 1))
    elif corner == "tr":
        cells.extend((r, c1 - depth_c) for r in range(r0, r0 + depth_r + 1))
        cells.extend((r0 + depth_r, c) for c in range(c1 - depth_c, c1 + 1))
    elif corner == "bl":
        cells.extend((r, c0 + depth_c) for r in range(r1 - depth_r, r1 + 1))
        cells.extend((r1 - depth_r, c) for c in range(c0, c0 + depth_c + 1))
    else:
        cells.extend((r, c1 - depth_c) for r in range(r1 - depth_r, r1 + 1))
        cells.extend((r1 - depth_r, c) for c in range(c1 - depth_c, c1 + 1))
    return sorted(set(cells))


def _rect_random(
    rng: random.Random,
    h: int,
    w: int,
    top_min: int,
    left_min: int,
    height_range: tuple[int, int] = (7, 10),
    width_range: tuple[int, int] = (7, 10),
) -> dict | None:
    rh = rng.randint(*height_range)
    rw = rng.randint(*width_range)
    if top_min > h - rh or left_min > w - rw:
        return None
    r0 = rng.randint(top_min, h - rh)
    c0 = rng.randint(left_min, w - rw)
    return {"bbox": (r0, c0, r0 + rh - 1, c0 + rw - 1), "diag": rng.choice(("tlbr", "trbl"))}


def _overlap_rect_pair(rng: random.Random, h: int, w: int, top_min: int, left_min: int) -> tuple[dict, dict, int, int] | None:
    for _ in range(80):
        h1 = rng.randint(8, 10)
        w1 = rng.randint(8, 10)
        h2 = rng.randint(8, 10)
        w2 = rng.randint(8, 10)
        overlap_h = rng.randint(3, min(4, h1 - 3, h2 - 3))
        overlap_w = rng.randint(3, min(4, w1 - 3, w2 - 3))
        max_top = h - h2 - h1 + overlap_h
        max_left = w - w2 - w1 + overlap_w
        if max_top < top_min or max_left < left_min:
            continue
        r0 = rng.randint(top_min, max_top)
        c0 = rng.randint(left_min, max_left)
        rect1 = {"bbox": (r0, c0, r0 + h1 - 1, c0 + w1 - 1), "diag": "trbl"}
        rect2 = {
            "bbox": (rect1["bbox"][2] - overlap_h, rect1["bbox"][3] - overlap_w, rect1["bbox"][2] - overlap_h + h2 - 1, rect1["bbox"][3] - overlap_w + w2 - 1),
            "diag": "trbl",
        }
        return rect1, rect2, overlap_h, overlap_w
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _pad_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _strictly_inside(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _attach_pocket(rect: dict, corner: str, depth_r: int, depth_c: int) -> None:
    rect["pocket"] = (corner, depth_r, depth_c)


def _place_rectangles(rng: random.Random, h: int, w: int, legend_rows: int, pair_count: int) -> tuple[list[dict], bool] | None:
    top_min = legend_rows + 2
    left_min = 4
    force_overlap = pair_count >= 2 and rng.random() < 0.6
    rects: list[dict] = []

    if force_overlap:
        pair = _overlap_rect_pair(rng, h, w, top_min, left_min)
        if pair is None:
            return None
        rect1, rect2, overlap_h, overlap_w = pair
        _attach_pocket(rect1, "br", overlap_h, overlap_w)
        _attach_pocket(rect2, "tl", overlap_h, overlap_w)
        rects.extend([rect1, rect2])

    while len(rects) < pair_count:
        rect = None
        for _ in range(120):
            candidate = _rect_random(rng, h, w, top_min, left_min)
            if candidate is None:
                return None
            r0, c0, r1, c1 = candidate["bbox"]
            max_depth_r = min(4, r1 - r0 - 2)
            max_depth_c = min(4, c1 - c0 - 2)
            if max_depth_r < 3 or max_depth_c < 3:
                continue
            _attach_pocket(
                candidate,
                rng.choice(("tl", "tr", "bl", "br")),
                rng.randint(3, max_depth_r),
                rng.randint(3, max_depth_c),
            )
            if any(_boxes_overlap(_pad_box(candidate["bbox"], 1), _pad_box(other["bbox"], 1)) for other in rects):
                continue
            rect = candidate
            break
        if rect is None:
            return None
        rects.append(rect)
    return rects, force_overlap


def _write_grid(h: int, w: int, pairs: list[tuple[int, int]], rects: list[dict]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    legend_rows = len(pairs)
    for r, (source, fill) in enumerate(pairs):
        grid[r, 0] = source
        grid[r, 1] = fill

    marker_cells = set()
    for rect, (source, _fill) in zip(rects, pairs):
        rect["source"] = source
        m0, m1 = _marker_cells(rect)
        if m0 in marker_cells or m1 in marker_cells:
            return None
        marker_cells.add(m0)
        marker_cells.add(m1)

    for i, rect in enumerate(rects):
        markers = _marker_cells(rect)
        for j, other in enumerate(rects):
            if i == j:
                continue
            if any(_strictly_inside(marker, other["bbox"]) for marker in markers):
                return None

    for rect in rects:
        corner, depth_r, depth_c = rect["pocket"]
        for cell in _l_wall_cells(rect, corner, depth_r, depth_c):
            if cell in marker_cells:
                return None
            r, c = cell
            if grid[r, c] not in (BG, WALL):
                return None
            grid[r, c] = WALL

    for rect, (source, _fill) in zip(rects, pairs):
        for r, c in _marker_cells(rect):
            if grid[r, c] != BG:
                return None
            grid[r, c] = source
    assert np.all(grid[legend_rows:, :2] == BG)
    return grid


def _compute_masks(grid: np.ndarray, pairs: list[tuple[int, int]]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for _priority, _source, _fill, r0, c0, r1, c1 in _marker_rectangles(grid, pairs):
        masks.append(_enclosed_mask_for_rect(grid, (r0, c0, r1, c1)))
    return masks


def _validate_grid(grid: np.ndarray, pairs: list[tuple[int, int]], require_overlap: bool) -> bool:
    out = main(grid)
    if np.array_equal(out, grid):
        return False

    masks = _compute_masks(grid, pairs)
    if any(not mask.any() for mask in masks):
        return False

    rects = _marker_rectangles(grid, pairs)
    for (_priority, _source, _fill, r0, c0, r1, c1), mask in zip(rects, masks):
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == BG
        comps = _components(sub)
        if len(comps) < 2:
            return False
        sizes = sorted((int(comp.shape[0]) for comp in comps), reverse=True)
        if len(sizes) < 2 or sizes[0] == sizes[1]:
            return False
        if np.any(mask & (grid != BG)):
            return False

    if require_overlap:
        overlap_seen = False
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if np.any(masks[i] & masks[j]):
                    overlap_seen = True
                    break
            if overlap_seen:
                break
        if not overlap_seen:
            return False
        earlier_fill = pairs[0][1]
        if not np.any(out == earlier_fill):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 23)
    w = rng.randint(16, 23)
    pair_count = rng.choice((2, 3))
    sources = rng.sample(list(SOURCE_COLORS), pair_count)
    fills = rng.sample(list(FILL_COLORS), pair_count)
    pairs = list(zip(sources, fills))

    placed = _place_rectangles(rng, h, w, len(pairs), pair_count)
    if placed is None:
        return None
    rects, require_overlap = placed

    grid = _write_grid(h, w, pairs, rects)
    if grid is None:
        return None
    if not _validate_grid(grid, pairs, require_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
