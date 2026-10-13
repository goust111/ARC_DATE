from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


DIAG_DIRS = ((-1, 1), (1, -1))
PALETTE = [1, 2, 3, 4]
QUADRANTS = ("nw", "ne", "sw", "se")
GEN_ATTEMPTS = 800


def _square_center(square: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = square
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _dist2_to_point(rc: tuple[int, int], center: tuple[float, float]) -> float:
    r, c = rc
    cr, cc = center
    return (r - cr) ** 2 + (c - cc) ** 2


def _all_bg_squares(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    h, w = grid.shape
    for size in (2, 3):
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                patch = grid[r0 : r0 + size, c0 : c0 + size]
                if np.all(patch == bg):
                    out.append((r0, c0, r0 + size - 1, c0 + size - 1))
    return out


def _find_center_gap(grid: np.ndarray, bg: int, objects: list[dict]) -> tuple[int, int, int, int]:
    h, w = grid.shape
    board_center = ((h - 1) / 2.0, (w - 1) / 2.0)
    scored = []
    for square in _all_bg_squares(grid, bg):
        gap_center = _square_center(square)
        obj_d2 = []
        for obj in objects:
            coords = obj["coords"]
            d2 = min(_dist2_to_point((int(r), int(c)), gap_center) for r, c in coords)
            obj_d2.append(d2)
        size = square[2] - square[0] + 1
        centerity = _dist2_to_point((gap_center[0], gap_center[1]), board_center)
        scored.append((sum(obj_d2), max(obj_d2), -size, centerity, square))

    assert scored, "expected at least one 2x2 or 3x3 background square"
    scored.sort(key=lambda item: item[:4])
    best_key = scored[0][:4]
    assert sum(item[:4] == best_key for item in scored) == 1, "expected a unique central gap square"
    return scored[0][4]


def _sum_counts(coords: np.ndarray) -> Counter[int]:
    return Counter(int(r) + int(c) for r, c in coords)


def _diag_degree(coord: tuple[int, int], coord_set: set[tuple[int, int]]) -> int:
    r, c = coord
    return int((r - 1, c + 1) in coord_set) + int((r + 1, c - 1) in coord_set)


def _choose_shrink_pixel(obj: dict, gap_square: tuple[int, int, int, int]) -> tuple[int, int]:
    coords = obj["coords"]
    center = _square_center(gap_square)
    sums = _sum_counts(coords)
    coord_list = [tuple(map(int, rc)) for rc in coords]
    coord_set = set(coord_list)

    records = []
    for r, c in coord_list:
        d2 = _dist2_to_point((r, c), center)
        supported = sums[r + c] >= 2
        degree = _diag_degree((r, c), coord_set) if supported else 2
        records.append((d2, 0 if supported else 1, degree, r, c))

    records.sort()
    best = records[0]
    assert sum(item[:3] == best[:3] for item in records) == 1, "expected a unique shrink pixel"
    return int(best[3]), int(best[4])


def _choose_growth_pivot(obj: dict, gap_square: tuple[int, int, int, int]) -> tuple[int, int]:
    coords = obj["coords"]
    center = _square_center(gap_square)
    sums = _sum_counts(coords)
    coord_list = [tuple(map(int, rc)) for rc in coords]
    coord_set = set(coord_list)

    supported = [rc for rc in coord_list if sums[rc[0] + rc[1]] >= 2]
    assert supported, "winner must contain at least one repeated anti-diagonal"

    farthest_all = max(_dist2_to_point(rc, center) for rc in coord_list)
    candidates = [rc for rc in supported if _dist2_to_point(rc, center) == farthest_all]
    if not candidates:
        farthest_supported = max(_dist2_to_point(rc, center) for rc in supported)
        candidates = [rc for rc in supported if _dist2_to_point(rc, center) == farthest_supported]

    records = []
    for r, c in candidates:
        degree = _diag_degree((r, c), coord_set)
        records.append((degree, -_dist2_to_point((r, c), center), -r, c, (r, c)))

    records.sort()
    best = records[0]
    assert sum(item[:2] == best[:2] for item in records) == 1, "expected a unique growth pivot"
    return best[-1]


def _growth_dir(
    obj: dict, pivot: tuple[int, int], gap_square: tuple[int, int, int, int]
) -> tuple[int, int]:
    coord_set = {tuple(map(int, rc)) for rc in obj["coords"]}
    pr, pc = pivot
    for dr, dc in DIAG_DIRS:
        has_before = (pr - dr, pc - dc) in coord_set
        has_after = (pr + dr, pc + dc) in coord_set
        if has_before and not has_after:
            return dr, dc
        if has_after and not has_before:
            return -dr, -dc

    center = _square_center(gap_square)
    options = []
    for dr, dc in DIAG_DIRS:
        nxt = (pr + dr, pc + dc)
        options.append((-_dist2_to_point(nxt, center), dr, dc))
    options.sort()
    return int(options[0][1]), int(options[0][2])


def _winner_growth_cells(obj: dict, gap_square: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    pivot = _choose_growth_pivot(obj, gap_square)
    dr, dc = _growth_dir(obj, pivot, gap_square)
    pr, pc = pivot
    return [(pr + dr, pc + dc), (pr + 2 * dr, pc + 2 * dc)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink the three smaller diagonal objects and extend the unique largest one along its anti-diagonal.

    Input structure:
    - The background is black and the foreground consists of exactly four 8-connected monochrome objects.
    - A 2x2 or 3x3 background square sits near the middle as the shared central gap, with one object in each diagonal quadrant around it.
    - Every object contains at least one repeated anti-diagonal (cells with the same r+c), and exactly one object has the largest area.

    Rule:
    - Find the central 2x2 or 3x3 background square whose center has the smallest aggregate distance to the four objects.
    - For each non-largest object, erase its closest pixel to the gap center; if multiple pixels tie, choose one that lies on a repeated anti-diagonal.
    - For the largest object, choose its farthest repeated-anti-diagonal pixel from the gap center and extend that anti-diagonal by two same-colored pixels on the outward side.

    Key insight:
    - Center distance decides which pixel retracts or leads, but the actual edit direction is constrained by the object's anti-diagonal support rather than by a simple radial move away from the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert len(objects) == 4, "expected four foreground objects"

    gap_square = _find_center_gap(g, bg, objects)
    areas = [int(obj["area"]) for obj in objects]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "expected a unique largest object"
    winner = objects[areas.index(max_area)]

    out = g.copy()
    for obj in objects:
        if obj is winner:
            continue
        r, c = _choose_shrink_pixel(obj, gap_square)
        out[r, c] = bg

    for r, c in _winner_growth_cells(winner, gap_square):
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
            assert int(g[r, c]) == bg, "growth cells must land on background"
            out[r, c] = int(winner["color"])

    return out


def _corner_anchor(square: tuple[int, int, int, int], quadrant: str) -> tuple[int, int]:
    r0, c0, r1, c1 = square
    if quadrant == "nw":
        return r0, c0
    if quadrant == "ne":
        return r0, c1
    if quadrant == "sw":
        return r1, c0
    if quadrant == "se":
        return r1, c1
    raise ValueError(quadrant)


def _in_quadrant(rc: tuple[int, int], square: tuple[int, int, int, int], quadrant: str) -> bool:
    r, c = rc
    r0, c0, r1, c1 = square
    if quadrant == "nw":
        return r < r0 and c < c0
    if quadrant == "ne":
        return r < r0 and c > c1
    if quadrant == "sw":
        return r > r1 and c < c0
    if quadrant == "se":
        return r > r1 and c > c1
    return False


def _connected8(cells: set[tuple[int, int]]) -> bool:
    todo = [next(iter(cells))]
    seen = {todo[0]}
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
    return seen == cells


def _regular_bud_candidates(
    cells: set[tuple[int, int]],
    square: tuple[int, int, int, int],
    quadrant: str,
    center: tuple[float, float],
    inner: tuple[int, int],
    far: tuple[int, int],
) -> list[tuple[int, int]]:
    inner_d2 = _dist2_to_point(inner, center)
    far_d2 = _dist2_to_point(far, center)
    candidates: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells or not _in_quadrant(nxt, square, quadrant):
                continue
            d2 = _dist2_to_point(nxt, center)
            if not (inner_d2 < d2 < far_d2):
                continue
            candidates.add(nxt)
    return list(candidates)


def _special_far_candidates(
    cells: set[tuple[int, int]],
    square: tuple[int, int, int, int],
    quadrant: str,
    center: tuple[float, float],
    far: tuple[int, int],
    growth_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    far_d2 = _dist2_to_point(far, center)
    candidates = []
    blocked = {far, (far[0] + growth_dir[0], far[1] + growth_dir[1]), (far[0] + 2 * growth_dir[0], far[1] + 2 * growth_dir[1])}
    for dr, dc in (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, -1),
    ):
        nxt = (far[0] + dr, far[1] + dc)
        if nxt in blocked or nxt in cells or not _in_quadrant(nxt, square, quadrant):
            continue
        if _dist2_to_point(nxt, center) <= far_d2:
            continue
        candidates.append(nxt)
    return candidates


def _build_object(
    h: int,
    w: int,
    square: tuple[int, int, int, int],
    quadrant: str,
    target_area: int,
    rng: random.Random,
    *,
    allow_far_special: bool,
) -> set[tuple[int, int]] | None:
    center = _square_center(square)
    for _ in range(200):
        base_len_max = min(5, target_area)
        base_len_min = max(3, target_area - 3)
        if allow_far_special:
            base_len_max = min(base_len_max, target_area - 1)
        if base_len_min > base_len_max:
            continue

        base_len = rng.randint(base_len_min, base_len_max)
        dr, dc = DIAG_DIRS[rng.randrange(2)]
        anchor_r, anchor_c = _corner_anchor(square, quadrant)

        row_gap = rng.randint(base_len, base_len + 2)
        col_gap = rng.randint(base_len, base_len + 2)
        if abs(row_gap - col_gap) > 1:
            if row_gap < col_gap:
                row_gap = col_gap - 1
            else:
                col_gap = row_gap - 1

        if quadrant == "nw":
            inner = (anchor_r - row_gap, anchor_c - col_gap)
        elif quadrant == "ne":
            inner = (anchor_r - row_gap, anchor_c + col_gap)
        elif quadrant == "sw":
            inner = (anchor_r + row_gap, anchor_c - col_gap)
        else:
            inner = (anchor_r + row_gap, anchor_c + col_gap)

        segment = [(inner[0] + k * dr, inner[1] + k * dc) for k in range(base_len)]
        far = segment[-1]
        if not all(0 <= r < h and 0 <= c < w for r, c in segment):
            continue
        if not all(_in_quadrant(rc, square, quadrant) for rc in segment):
            continue

        cells = set(segment)
        growth_dir = (dr, dc)
        special_far = allow_far_special and target_area >= base_len + 1 and rng.random() < 0.45
        if special_far:
            far_options = _special_far_candidates(cells, square, quadrant, center, far, growth_dir)
            rng.shuffle(far_options)
            if far_options:
                cells.add(far_options[0])

        remaining = target_area - len(cells)
        if remaining < 0:
            continue

        regular = _regular_bud_candidates(cells, square, quadrant, center, inner, far)
        rng.shuffle(regular)
        for rc in regular:
            if remaining <= 0:
                break
            cells.add(rc)
            remaining -= 1

        if remaining != 0:
            continue
        if not _connected8(cells):
            continue

        sums = Counter(r + c for r, c in cells)
        if max(sums.values()) < 2:
            continue
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            continue

        return cells
    return None


def _paint_object(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        bg = get_bg(grid)
        objects = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
        if len(objects) != 4:
            return False

        areas = [int(obj["area"]) for obj in objects]
        if areas.count(max(areas)) != 1:
            return False

        gap_square = _find_center_gap(grid, bg, objects)
        winner = objects[areas.index(max(areas))]

        for obj in objects:
            if max(_sum_counts(obj["coords"]).values()) < 2:
                return False
            if obj is winner:
                continue
            _choose_shrink_pixel(obj, gap_square)

        growth_cells = _winner_growth_cells(winner, gap_square)
        if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in growth_cells):
            return False
        if any(int(grid[r, c]) != bg for r, c in growth_cells):
            return False

        out = main(grid)
        changed = np.argwhere(out != grid)
        if len(changed) != 5:
            return False

        winner_color = int(winner["color"])
        added = [(int(r), int(c)) for r, c in changed if int(grid[r, c]) == bg and int(out[r, c]) == winner_color]
        removed = [(int(r), int(c)) for r, c in changed if int(grid[r, c]) != bg and int(out[r, c]) == bg]
        if len(added) != 2 or len(removed) != 3:
            return False

        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(13, 18)
    w = rng.randint(13, 18)
    grid = np.zeros((h, w), dtype=int)

    gap_size = 3 if rng.random() < 0.55 else 2
    min_margin = 4
    r0 = rng.randint(min_margin, h - gap_size - min_margin)
    c0 = rng.randint(min_margin, w - gap_size - min_margin)
    gap_square = (r0, c0, r0 + gap_size - 1, c0 + gap_size - 1)

    winner_quadrant = rng.choice(QUADRANTS)
    winner_area = rng.randint(5, 8)
    loser_areas = [rng.randint(3, winner_area - 1) for _ in range(3)]

    color_order = rng.sample(PALETTE, 4)
    objects: dict[str, set[tuple[int, int]]] = {}
    areas_by_quadrant = {
        winner_quadrant: winner_area,
        **{q: loser_areas[i] for i, q in enumerate(q for q in QUADRANTS if q != winner_quadrant)},
    }
    colors_by_quadrant = {q: color_order[i] for i, q in enumerate(QUADRANTS)}

    for quadrant in QUADRANTS:
        obj = _build_object(
            h,
            w,
            gap_square,
            quadrant,
            areas_by_quadrant[quadrant],
            rng,
            allow_far_special=(quadrant == winner_quadrant),
        )
        if obj is None:
            return None
        objects[quadrant] = obj

    all_cells = set()
    for quadrant, obj in objects.items():
        if all_cells & obj:
            return None
        for r, c in obj:
            if grid[r, c] != 0:
                return None
            if not _in_quadrant((r, c), gap_square, quadrant):
                return None
        all_cells |= obj

    for quadrant in QUADRANTS:
        _paint_object(grid, objects[quadrant], colors_by_quadrant[quadrant])

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 12")
