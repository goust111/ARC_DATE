import math
from collections import defaultdict, deque

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
RNG = np.random.default_rng()

MIN_OBJECTS = 3
MAX_OBJECTS = 5
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 8)
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400
TIE_PROBABILITY = 0.5
PADDING = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
            out.append(coords)
    return out


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _colors_in_scan_order(grid: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color != bg and color not in seen:
                seen.add(color)
                colors.append(color)
    return colors


def _first_appearance(grid: np.ndarray, color: int) -> tuple[int, int]:
    rows, cols = np.where(grid == color)
    first_row = int(rows.min())
    first_col = int(cols[rows == first_row].min())
    return first_row, first_col


def _straight_cut_score(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    h, w = mask_a.shape
    for cut in range(1, w):
        if not mask_a[:, cut:].any() and not mask_b[:, :cut].any():
            return 2
        if not mask_b[:, cut:].any() and not mask_a[:, :cut].any():
            return 2
    for cut in range(1, h):
        if not mask_a[cut:, :].any() and not mask_b[:cut, :].any():
            return 2
        if not mask_b[cut:, :].any() and not mask_a[:cut, :].any():
            return 2
    return 1


def _reconstruct_rectangle(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    area_total = int(mask_a.sum() + mask_b.sum())
    h_a, w_a = mask_a.shape
    h_b, w_b = mask_b.shape

    candidates: dict[tuple[int, int], int] = {}
    seen_dims: set[tuple[int, int]] = set()
    for h in range(1, int(math.isqrt(area_total)) + 1):
        if area_total % h != 0:
            continue
        for dims in ((h, area_total // h), (area_total // h, h)):
            if dims in seen_dims:
                continue
            seen_dims.add(dims)
            out_h, out_w = dims
            if out_h < max(h_a, h_b) or out_w < max(w_a, w_b):
                continue
            for r0 in range(out_h - h_a + 1):
                for c0 in range(out_w - w_a + 1):
                    placed_a = np.zeros((out_h, out_w), dtype=bool)
                    placed_a[r0 : r0 + h_a, c0 : c0 + w_a] = mask_a
                    placed_b = ~placed_a
                    if placed_b.sum() != mask_b.sum():
                        continue
                    if np.array_equal(_normalize_mask(placed_b), mask_b):
                        score = _straight_cut_score(placed_a, placed_b)
                        candidates[dims] = max(candidates.get(dims, 0), score)

    if not candidates:
        raise AssertionError("failed to reconstruct a complete rectangle")

    best_dims = min(
        (dims for dims, score in candidates.items() if score == max(candidates.values())),
        key=lambda dims: (dims[0], dims[1]),
    )
    return best_dims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, reconstruct their full rectangles, then stack the rectangles.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())

    rectangles: list[tuple[int, int, int, int, int, int, int]] = []
    for scan_index, color in enumerate(_colors_in_scan_order(g, bg)):
        comps = _components(g == color)
        assert len(comps) == 2
        mask_a, _ = _coords_to_mask(comps[0])
        mask_b, _ = _coords_to_mask(comps[1])
        rect_h, rect_w = _reconstruct_rectangle(mask_a, mask_b)
        first_row, first_col = _first_appearance(g, int(color))
        rectangles.append(
            (
                rect_h * rect_w,
                first_row,
                first_col,
                scan_index,
                int(color),
                rect_h,
                rect_w,
            )
        )

    rectangles.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    out_h = sum(item[5] for item in rectangles)
    out_w = max(item[6] for item in rectangles)
    out = np.full((out_h, out_w), bg, dtype=int)

    row = 0
    for _area, _first_row, _first_col, _scan_index, color, rect_h, rect_w in rectangles:
        out[row : row + rect_h, :rect_w] = color
        row += rect_h
    return out


def _all_rectangle_dims() -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(RECT_H_RANGE[0], RECT_H_RANGE[1] + 1):
        for w in range(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1):
            if h >= 3 or w >= 3:
                dims.append((h, w))
    return dims


def _sample_specs() -> list[dict]:
    dims_by_area: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for dims in _all_rectangle_dims():
        dims_by_area[dims[0] * dims[1]].append(dims)

    count = int(RNG.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    use_tie = bool(RNG.random() < TIE_PROBABILITY) if count >= 3 else False
    specs: list[dict] = []
    used_areas: set[int] = set()

    if use_tie:
        tie_areas = [area for area, dims in dims_by_area.items() if len(dims) >= 2]
        tie_area = int(RNG.choice(tie_areas))
        tie_dims = dims_by_area[tie_area].copy()
        RNG.shuffle(tie_dims)
        for rect_h, rect_w in tie_dims[:2]:
            specs.append({"h": rect_h, "w": rect_w, "area": tie_area})
        used_areas.add(tie_area)

    all_dims = _all_rectangle_dims()
    while len(specs) < count:
        rect_h, rect_w = all_dims[int(RNG.integers(0, len(all_dims)))]
        area = rect_h * rect_w
        if area in used_areas:
            continue
        specs.append({"h": rect_h, "w": rect_w, "area": area})
        used_areas.add(area)

    colors = list(RNG.choice(COLOR_POOL, size=count, replace=False))
    RNG.shuffle(colors)
    RNG.shuffle(specs)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)
    return specs


def _split_rectangle(rect_h: int, rect_w: int, color: int) -> list[np.ndarray]:
    orientations = []
    if rect_w >= 3:
        orientations.append("vertical")
    if rect_h >= 3:
        orientations.append("horizontal")
    orientation = orientations[int(RNG.integers(0, len(orientations)))]

    if orientation == "vertical":
        splits = [cut for cut in range(1, rect_w) if cut != rect_w - cut]
        cut = int(RNG.choice(splits))
        parts = [
            np.full((rect_h, cut), color, dtype=int),
            np.full((rect_h, rect_w - cut), color, dtype=int),
        ]
    else:
        splits = [cut for cut in range(1, rect_h) if cut != rect_h - cut]
        cut = int(RNG.choice(splits))
        parts = [
            np.full((cut, rect_w), color, dtype=int),
            np.full((rect_h - cut, rect_w), color, dtype=int),
        ]

    if bool(RNG.integers(0, 2)):
        parts.reverse()
    return parts


def _can_place(occupied: np.ndarray, top: int, left: int, sprite: np.ndarray) -> bool:
    h, w = sprite.shape
    r0 = max(0, top - PADDING)
    c0 = max(0, left - PADDING)
    r1 = min(occupied.shape[0], top + h + PADDING)
    c1 = min(occupied.shape[1], left + w + PADDING)
    return not occupied[r0:r1, c0:c1].any()


def _place_pieces(specs: list[dict]) -> np.ndarray | None:
    grid_h = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    grid_w = int(RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    pieces: list[tuple[int, np.ndarray]] = []
    for spec in specs:
        for part in _split_rectangle(spec["h"], spec["w"], spec["color"]):
            pieces.append((spec["color"], part))

    RNG.shuffle(pieces)
    pieces.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    for color, sprite in pieces:
        h, w = sprite.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = int(RNG.integers(0, grid_h - h + 1))
            left = int(RNG.integers(0, grid_w - w + 1))
            if not _can_place(occupied, top, left, sprite):
                continue
            grid[top : top + h, left : left + w] = color
            occupied[top : top + h, left : left + w] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray | None:
    grouped_rows: dict[int, list[int]] = defaultdict(list)
    ordered = []
    for scan_index, spec in enumerate(specs):
        first_row, first_col = _first_appearance(grid, spec["color"])
        grouped_rows[spec["area"]].append(first_row)
        ordered.append(
            (
                spec["area"],
                first_row,
                first_col,
                scan_index,
                spec["color"],
                spec["h"],
                spec["w"],
            )
        )

    for rows in grouped_rows.values():
        if len(rows) != len(set(rows)):
            return None

    ordered.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    out_h = sum(item[5] for item in ordered)
    out_w = max(item[6] for item in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)

    row = 0
    for _area, _first_row, _first_col, _scan_index, color, rect_h, rect_w in ordered:
        out[row : row + rect_h, :rect_w] = color
        row += rect_h
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    grid = _place_pieces(specs)
    if grid is None:
        return None

    colors = _colors_in_scan_order(grid, BG)
    if len(colors) != len(specs):
        return None
    for color in colors:
        if len(_components(grid == color)) != 2:
            return None

    expected = _expected_output(grid, specs)
    if expected is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
