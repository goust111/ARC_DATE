from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BACKGROUND = 0
MARKER_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
PAIR_COUNT_CHOICES = (2, 3, 4)
ALLOWED_RECTANGLES = ((2, 3), (3, 3), (4, 3), (2, 5))
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250

_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _placed(mask: np.ndarray, top: int, left: int, out_shape: tuple[int, int]) -> np.ndarray:
    board = np.zeros(out_shape, dtype=bool)
    coords = np.argwhere(mask)
    board[top + coords[:, 0], left + coords[:, 1]] = True
    return board


def _can_fill_rectangle(mask_a: np.ndarray, mask_b: np.ndarray, out_h: int, out_w: int) -> bool:
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    for ra in range(out_h - ha + 1):
        for ca in range(out_w - wa + 1):
            board_a = _placed(mask_a, ra, ca, (out_h, out_w))
            for rb in range(out_h - hb + 1):
                for cb in range(out_w - wb + 1):
                    board_b = _placed(mask_b, rb, cb, (out_h, out_w))
                    if (board_a & board_b).any():
                        continue
                    if bool((board_a | board_b).all()):
                        return True
    return False


def _rectangle_dims_for_oriented_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> set[tuple[int, int]]:
    total = int(mask_a.sum() + mask_b.sum())
    dims: set[tuple[int, int]] = set()
    min_h = max(mask_a.shape[0], mask_b.shape[0])
    min_w = max(mask_a.shape[1], mask_b.shape[1])
    for out_h in range(min_h, total + 1):
        if total % out_h != 0:
            continue
        out_w = total // out_h
        if out_w < min_w:
            continue
        if _can_fill_rectangle(mask_a, mask_b, out_h, out_w):
            dims.add((int(out_h), int(out_w)))
    return dims


def _pair_rectangle_shape(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    dims = set()
    dims |= _rectangle_dims_for_oriented_pair(mask_a, np.rot90(mask_b, 2))
    dims |= _rectangle_dims_for_oriented_pair(np.rot90(mask_a, 2), mask_b)
    assert len(dims) == 1, f"ambiguous complementary rectangle: {sorted(dims)}"
    return next(iter(dims))


def _find_markers(grid: np.ndarray, bg: int) -> tuple[int, list[int], int]:
    rows = np.where(np.any(grid != bg, axis=1))[0]
    assert len(rows) > 0, "grid must contain markers and fragments"
    marker_row = int(rows[-1])
    marker_cols = [int(c) for c in np.where(grid[marker_row] != bg)[0]]
    assert marker_cols, "missing bottom marker row"
    marker_colors = {int(grid[marker_row, c]) for c in marker_cols}
    assert len(marker_colors) == 1, "markers must share one color"
    if len(marker_cols) >= 2:
        gaps = np.diff(marker_cols)
        assert np.all(gaps == gaps[0]), "markers must be equally spaced"
    row = grid[marker_row].copy()
    row[marker_cols] = bg
    assert bool(np.all(row == bg)), "marker row must contain only marker dots"
    return marker_row, marker_cols, int(next(iter(marker_colors)))


def _find_fragments(
    grid: np.ndarray,
    bg: int,
    marker_row: int,
    marker_cols: list[int],
) -> list[dict]:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[marker_row, marker_cols] = True
    visited = np.zeros((h, w), dtype=bool)
    fragments: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or blocked[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or blocked[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            fragments.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "mask": _mask_from_coords(arr),
                }
            )

    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """Pair each same-colored complementary fragment set, rebuild its rectangle, and restack the rectangles above the bottom markers.

    Input structure:
    - The background is black.
    - Several small monochrome 4-connected fragments are scattered in the upper part of the grid.
    - Every non-marker color appears in exactly two fragments, and one fragment becomes rectangle-complementary to the other after a 180-degree rotation.
    - The bottom marker row contains equally spaced singleton dots of one color; the number of dots equals the number of fragment pairs.

    Rule:
    - Ignore the bottom marker dots and extract all other monochrome connected components.
    - Pair fragments by color; for each pair, rotate one fragment by 180 degrees and find the unique solid rectangle that both fragments tile together.
    - Order the color pairs by the row-major position of that color's first input pixel.
    - Erase the original fragments, keep the marker dots, and place the rebuilt solid rectangles directly above the markers from left to right, with each rectangle bottom-aligned to the row above its marker and horizontally centered on that marker.

    Key insight:
    - The task has two independent steps: infer the hidden rectangle size from complementary same-color polyominoes, then sort those reconstructed rectangles by where each color first appears before relocating them to the marker row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    marker_row, marker_cols, marker_color = _find_markers(g, bg)
    fragments = _find_fragments(g, bg, marker_row, marker_cols)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frag in fragments:
        by_color[int(frag["color"])].append(frag)

    assert len(by_color) == len(marker_cols), "marker count must match fragment-pair count"
    pairs: list[dict] = []
    for color, comps in by_color.items():
        assert len(comps) == 2, f"color {color} must appear in exactly two fragments"
        rect_h, rect_w = _pair_rectangle_shape(comps[0]["mask"], comps[1]["mask"])
        first_pixel = min(tuple(map(int, xy)) for comp in comps for xy in comp["coords"])
        pairs.append(
            {
                "color": int(color),
                "rect_shape": (int(rect_h), int(rect_w)),
                "first_pixel": first_pixel,
            }
        )

    pairs.sort(key=lambda item: item["first_pixel"])

    out = np.full_like(g, bg)
    out[marker_row, marker_cols] = marker_color
    for marker_col, pair in zip(marker_cols, pairs):
        rect_h, rect_w = pair["rect_shape"]
        top = marker_row - rect_h
        left = marker_col - rect_w // 2
        assert top >= 0 and left >= 0 and left + rect_w <= g.shape[1]
        out[top:marker_row, left : left + rect_w] = int(pair["color"])
    return out


def _build_rectangle_catalog() -> list[dict]:
    catalog: list[dict] = []
    for rect_h, rect_w in ALLOWED_RECTANGLES:
        area = rect_h * rect_w
        cells = list(range(area))
        for area_a in range(2, min(6, area - 2) + 1):
            area_b = area - area_a
            if not 2 <= area_b <= 6:
                continue
            for combo in combinations(cells[1:], area_a - 1):
                chosen = (0,) + combo
                mask_a = np.zeros((rect_h, rect_w), dtype=bool)
                flat = mask_a.ravel()
                flat[list(chosen)] = True
                mask_b = ~mask_a
                if not _is_connected(mask_a) or not _is_connected(mask_b):
                    continue
                if _is_solid_rectangle(mask_a) and _is_solid_rectangle(mask_b):
                    continue
                input_a = mask_a
                input_b = np.rot90(mask_b, 2)
                if _pair_rectangle_shape(input_a, input_b) != (rect_h, rect_w):
                    continue
                catalog.append(
                    {
                        "rect_shape": (int(rect_h), int(rect_w)),
                        "rect_a": mask_a.copy(),
                        "rect_b": mask_b.copy(),
                    }
                )
    assert catalog, "empty fragment catalog"
    return catalog


_RECTANGLE_CATALOG = _build_rectangle_catalog()


def _sample_pair_specs() -> list[dict]:
    pair_count = int(_RNG.choice(PAIR_COUNT_CHOICES))
    colors = [int(c) for c in _RNG.choice(COLOR_POOL, size=pair_count, replace=False)]
    picked = [dict(_RECTANGLE_CATALOG[int(i)]) for i in _RNG.integers(0, len(_RECTANGLE_CATALOG), size=pair_count)]

    pair_specs: list[dict] = []
    for color, spec in zip(colors, picked):
        if bool(_RNG.integers(0, 2)):
            masks = [np.rot90(spec["rect_a"], 2), spec["rect_b"]]
        else:
            masks = [spec["rect_a"], np.rot90(spec["rect_b"], 2)]
        if bool(_RNG.integers(0, 2)):
            masks.reverse()
        pair_specs.append(
            {
                "color": int(color),
                "rect_shape": tuple(map(int, spec["rect_shape"])),
                "masks": [m.copy() for m in masks],
            }
        )
    return pair_specs


def _layout_from_pairs(pair_specs: list[dict]) -> tuple[int, int, np.ndarray] | None:
    max_rect_h = max(spec["rect_shape"][0] for spec in pair_specs)
    max_rect_w = max(spec["rect_shape"][1] for spec in pair_specs)
    pair_count = len(pair_specs)

    stride = max_rect_w + int(_RNG.integers(1, 3))
    left_margin = max_rect_w // 2 + 1 + int(_RNG.integers(0, 2))
    marker_cols = np.array([left_margin + i * stride for i in range(pair_count)], dtype=int)
    right_slack = 1 + int(_RNG.integers(1, 3))
    width = int(marker_cols[-1] + max_rect_w // 2 + 1 + right_slack)
    height = int(max_rect_h + _RNG.integers(14, 20))
    if height > 30 or width > 30:
        return None
    return height, width, marker_cols


def _try_place_piece(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
    usable_rows: int,
) -> np.ndarray | None:
    h, w = mask.shape
    row_min, col_min = 1, 1
    row_max = usable_rows - h - 1
    col_max = grid.shape[1] - w - 1
    if row_max < row_min or col_max < col_min:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        top = int(_RNG.integers(row_min, row_max + 1))
        left = int(_RNG.integers(col_min, col_max + 1))
        pad_r0 = max(0, top - 1)
        pad_c0 = max(0, left - 1)
        pad_r1 = min(grid.shape[0], top + h + 1)
        pad_c1 = min(grid.shape[1], left + w + 1)
        if bool(reserved[pad_r0:pad_r1, pad_c0:pad_c1].any()):
            continue
        patch = grid[top : top + h, left : left + w]
        patch[mask] = int(color)
        reserved[pad_r0:pad_r1, pad_c0:pad_c1] = True
        coords = np.argwhere(mask) + np.array([top, left], dtype=int)
        return coords
    return None


def _build_expected(
    shape: tuple[int, int],
    marker_row: int,
    marker_cols: np.ndarray,
    pair_infos: list[dict],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    out[marker_row, marker_cols] = MARKER_COLOR
    ordered = sorted(pair_infos, key=lambda item: item["first_pixel"])
    for marker_col, info in zip(marker_cols, ordered):
        rect_h, rect_w = info["rect_shape"]
        top = marker_row - rect_h
        left = int(marker_col) - rect_w // 2
        out[top:marker_row, left : left + rect_w] = int(info["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_specs = _sample_pair_specs()
    layout = _layout_from_pairs(pair_specs)
    if layout is None:
        return None
    height, width, marker_cols = layout
    marker_row = height - 1
    max_rect_h = max(spec["rect_shape"][0] for spec in pair_specs)
    usable_rows = marker_row - max_rect_h - 1
    if usable_rows < 6:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[marker_row, marker_cols] = MARKER_COLOR
    reserved = np.zeros((height, width), dtype=bool)
    reserved[usable_rows:, :] = True

    pair_infos = [
        {"color": int(spec["color"]), "rect_shape": spec["rect_shape"], "coords": []}
        for spec in pair_specs
    ]
    jobs = [
        {"pair_idx": pair_idx, "color": int(spec["color"]), "mask": mask}
        for pair_idx, spec in enumerate(pair_specs)
        for mask in spec["masks"]
    ]
    _RNG.shuffle(jobs)

    for job in jobs:
        coords = _try_place_piece(grid, reserved, job["mask"], job["color"], usable_rows)
        if coords is None:
            return None
        pair_infos[int(job["pair_idx"])]["coords"].append(coords)

    first_rows = []
    for info in pair_infos:
        first_pixel = min(tuple(map(int, xy)) for coords in info["coords"] for xy in coords)
        info["first_pixel"] = first_pixel
        first_rows.append(int(first_pixel[0]))
    if len(set(first_rows)) != len(first_rows):
        return None

    expected = _build_expected(grid.shape, marker_row, marker_cols, pair_infos)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment ARC input")
