import random
from collections import defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

# Canvas.
GRID_H_RANGE = (20, 31)  # inclusive-low, exclusive-high
GRID_W_RANGE = (20, 31)
PAIR_COUNT_CHOICES = [2, 3, 4]
COLOR_POOL = list(range(1, 10))

# Piece geometry.
BOARD_H_RANGE = (6, 12)
BOARD_W_RANGE = (6, 12)
MIN_FILLER_AREA = 4
SHAPE_RETRIES = 80
PAIR_RETRIES = 120

# Placement.
BORDER_PAD = 1
OBJECT_PAD = 2
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _count_holes(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    bg = ~padded
    h, w = bg.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque([(0, 0)])
    outside[0, 0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    holes = bg & (~outside)
    return len(_components(holes[1:-1, 1:-1]))


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments by hole count, keep the hole-rich base plate,
    and fill its bounding box to obtain the completed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in _monochrome_components(g):
        by_color[int(comp["color"])].append(comp)

    for color, comps in by_color.items():
        assert len(comps) == 2, "each color should appear in exactly two components"
        hole_counts = [_count_holes(comp["mask"]) for comp in comps]
        assert hole_counts[0] != hole_counts[1], "the base plate must be uniquely determined by hole count"
        board = comps[0] if hole_counts[0] > hole_counts[1] else comps[1]
        r0, c0, r1, c1 = board["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return out


def _choose_weighted(candidates: list[tuple[int, int]], weights: list[float]) -> tuple[int, int]:
    probs = np.asarray(weights, dtype=float)
    probs /= probs.sum()
    idx = int(RNG.choice(len(candidates), p=probs))
    return candidates[idx]


def _grow_shape(canvas_h: int, canvas_w: int, target_area: int, mode: int) -> np.ndarray:
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    start = (int(RNG.integers(0, canvas_h)), int(RNG.integers(0, canvas_w)))
    mask[start] = True
    last = start
    axis = int(RNG.integers(0, 2))
    sign = random.choice([-1, 1])

    while int(mask.sum()) < target_area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < canvas_h and 0 <= nc < canvas_w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break

        candidates = sorted(frontier)
        weights: list[float] = []
        for nr, nc in candidates:
            adj = 0
            for dr, dc in DIR4:
                rr, cc = nr + dr, nc + dc
                if 0 <= rr < canvas_h and 0 <= cc < canvas_w and mask[rr, cc]:
                    adj += 1

            if mode == 0:
                weight = 1.0 + 3.0 * adj
            elif mode == 1:
                weight = 1.0
                if abs(nr - last[0]) + abs(nc - last[1]) == 1:
                    weight += 5.0
                if adj == 1:
                    weight += 3.0
                if adj >= 3:
                    weight *= 0.35
            else:
                progress = nr if axis == 0 else nc
                if sign < 0:
                    progress = (canvas_h - 1 - nr) if axis == 0 else (canvas_w - 1 - nc)
                weight = 1.0 + adj + progress
            weights.append(weight)

        chosen = _choose_weighted(candidates, weights)
        mask[chosen] = True
        last = chosen

    return mask


def _random_filler_mask(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(SHAPE_RETRIES):
        canvas_h = int(RNG.integers(2, max_h + 1))
        canvas_w = int(RNG.integers(2, max_w + 1))
        canvas_area = canvas_h * canvas_w
        area_hi = max(MIN_FILLER_AREA + 1, int(canvas_area * RNG.uniform(0.42, 0.72)))
        target_area = int(RNG.integers(MIN_FILLER_AREA, min(canvas_area, area_hi) + 1))
        mode = int(RNG.integers(0, 3))
        mask = _trim_mask(_grow_shape(canvas_h, canvas_w, target_area, mode))

        if mask.size == 0 or mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        if int(mask.sum()) < MIN_FILLER_AREA:
            continue
        if min(mask.shape) < 2:
            continue
        if mask.all():
            continue
        if len(_components(mask)) != 1:
            continue
        if _count_holes(mask) != 0:
            continue
        return mask

    return None


def _build_pair(color: int) -> dict | None:
    for _ in range(PAIR_RETRIES):
        board_h = int(RNG.integers(BOARD_H_RANGE[0], BOARD_H_RANGE[1] + 1))
        board_w = int(RNG.integers(BOARD_W_RANGE[0], BOARD_W_RANGE[1] + 1))
        filler_mask = _random_filler_mask(board_h - 2, board_w - 2)
        if filler_mask is None:
            continue

        fill_h, fill_w = map(int, filler_mask.shape)
        hole_r = int(RNG.integers(1, board_h - fill_h))
        hole_c = int(RNG.integers(1, board_w - fill_w))

        board = np.full((board_h, board_w), int(color), dtype=int)
        hole_view = board[hole_r : hole_r + fill_h, hole_c : hole_c + fill_w]
        hole_view[filler_mask] = 0

        board_mask = board != 0
        if len(_components(board_mask)) != 1:
            continue
        if _count_holes(board_mask) != 1:
            continue

        filler = np.zeros((fill_h, fill_w), dtype=int)
        filler[filler_mask] = int(color)

        return {
            "color": int(color),
            "board": board,
            "filler": filler,
        }

    return None


def _find_location(blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = patch.shape
    row_lo = BORDER_PAD
    row_hi = h - ph - BORDER_PAD
    col_lo = BORDER_PAD
    col_hi = w - pw - BORDER_PAD
    if row_hi < row_lo or col_hi < col_lo:
        return None

    candidates: list[tuple[int, int]] = []
    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(col_lo, col_hi + 1):
            if not blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                candidates.append((r0, c0))

    if not candidates:
        return None
    return candidates[int(RNG.integers(0, len(candidates)))]


def _stamp_blocked(blocked: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = blocked.shape
    ph, pw = patch.shape
    r0 = max(0, top - OBJECT_PAD)
    c0 = max(0, left - OBJECT_PAD)
    r1 = min(h, top + ph + OBJECT_PAD)
    c1 = min(w, left + pw + OBJECT_PAD)
    blocked[r0:r1, c0:c1] = True


def _paint(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    region = grid[top : top + ph, left : left + pw]
    mask = patch != 0
    region[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    w = int(RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    pair_count = int(RNG.choice(PAIR_COUNT_CHOICES, p=[0.3, 0.45, 0.25]))
    colors = [int(x) for x in RNG.choice(COLOR_POOL, size=pair_count, replace=False)]

    pairs: list[dict] = []
    for color in colors:
        pair = _build_pair(color)
        if pair is None:
            return None
        pairs.append(pair)

    bbox_area = sum(int(pair["board"].shape[0] * pair["board"].shape[1] + pair["filler"].size) for pair in pairs)
    if bbox_area > int(h * w * 0.52):
        return None

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros_like(grid)
    blocked = np.zeros((h, w), dtype=bool)

    board_order = sorted(range(pair_count), key=lambda i: int(np.prod(pairs[i]["board"].shape)), reverse=True)
    filler_order = sorted(range(pair_count), key=lambda i: int(np.prod(pairs[i]["filler"].shape)), reverse=True)

    for idx in board_order:
        patch = pairs[idx]["board"]
        loc = _find_location(blocked, patch)
        if loc is None:
            return None
        top, left = loc
        pairs[idx]["board_pos"] = (top, left)
        _paint(grid, patch, top, left)
        _stamp_blocked(blocked, top, left, patch)
        expected[top : top + patch.shape[0], left : left + patch.shape[1]] = int(pairs[idx]["color"])

    for idx in filler_order:
        patch = pairs[idx]["filler"]
        loc = _find_location(blocked, patch)
        if loc is None:
            return None
        top, left = loc
        _paint(grid, patch, top, left)
        _stamp_blocked(blocked, top, left, patch)

    comps = _monochrome_components(grid)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    if len(by_color) != pair_count:
        return None
    for color in colors:
        color_comps = by_color.get(int(color), [])
        if len(color_comps) != 2:
            return None
        holes = sorted(_count_holes(comp["mask"]) for comp in color_comps)
        if holes != [0, 1]:
            return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
