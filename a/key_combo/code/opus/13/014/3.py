from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
MAGENTA = 6

GEN_RETRIES = 500
GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (18, 25)


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_blue_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLUE:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == BLUE:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            bbox = (
                int(arr[:, 0].min()),
                int(arr[:, 1].min()),
                int(arr[:, 0].max()),
                int(arr[:, 1].max()),
            )
            objects.append({"coords": arr, "bbox": bbox})
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def _rounded_centroid(coords: np.ndarray) -> tuple[int, int]:
    return (
        _round_half_up(float(coords[:, 0].mean())),
        _round_half_up(float(coords[:, 1].mean())),
    )


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 or abs(dr) != abs(dc):
        return []
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    steps = abs(dr)
    return [(r0 + k * sr, c0 + k * sc) for k in range(steps + 1)]


def _touches_existing(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> bool:
    for r, c in coords:
        if (r, c) in cells:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift vertically ordered blue objects, connect new diagonal centroid alignments, then split beams from crossed magenta triggers.

    Input structure:
    - The grid uses black background with 5-7 blue 4-connected objects and a few isolated magenta trigger pixels.
    - Blue objects are distinguishable by their bounding-box top rows, giving a unique top-to-bottom order.
    - Trigger pixels are static markers; only the blue objects move.

    Rule:
    - Sort blue objects by bounding-box top row. Shift rank 1, 3, 5, ... one column left and rank 2, 4, 6, ... one column right, dropping shifted pixels that leave the grid.
    - For each shifted object, compute its centroid from its shifted pixels and round each coordinate to the nearest integer.
    - Whenever two shifted centroids lie on the same 45-degree diagonal, draw a green diagonal segment between them. Blue cells stay blue and magenta triggers stay magenta.
    - Every magenta trigger crossed by a green diagonal emits magenta rays in the perpendicular diagonal direction; each ray paints only background cells and stops at the first non-background cell.

    Key insight:
    - The decisive diagonal pairings do not come from the original layout; they emerge only after applying the alternating left/right shift determined by vertical rank parity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _find_blue_objects(g)
    trigger_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == MAGENTA)}

    out = g.copy()
    out[g == BLUE] = BG

    shifted_objects: list[np.ndarray] = []
    for rank, obj in enumerate(objects, start=1):
        dc = -1 if rank % 2 == 1 else 1
        shifted = []
        for r, c in obj["coords"]:
            nr, nc = int(r), int(c + dc)
            if _in_bounds((h, w), nr, nc):
                shifted.append((nr, nc))
        shifted_arr = np.asarray(shifted, dtype=int) if shifted else np.zeros((0, 2), dtype=int)
        shifted_objects.append(shifted_arr)
        if shifted:
            out[shifted_arr[:, 0], shifted_arr[:, 1]] = BLUE

    centers: list[tuple[int, int] | None] = []
    for coords in shifted_objects:
        centers.append(_rounded_centroid(coords) if len(coords) else None)

    trigger_dirs: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for i in range(len(centers)):
        if centers[i] is None:
            continue
        for j in range(i + 1, len(centers)):
            if centers[j] is None:
                continue
            p = centers[i]
            q = centers[j]
            assert p is not None and q is not None
            if p == q or abs(p[0] - q[0]) != abs(p[1] - q[1]):
                continue
            path = _diag_path(p, q)
            dr = 1 if q[0] > p[0] else -1
            dc = 1 if q[1] > p[1] else -1
            perp_dirs = {(dr, -dc), (-dr, dc)}
            for r, c in path:
                if (r, c) in trigger_cells:
                    trigger_dirs.setdefault((r, c), set()).update(perp_dirs)
                if int(out[r, c]) == BG:
                    out[r, c] = GREEN

    for (r0, c0), dirs in trigger_dirs.items():
        for dr, dc in dirs:
            r, c = r0 + dr, c0 + dc
            while _in_bounds((h, w), r, c):
                if int(out[r, c]) != BG:
                    break
                out[r, c] = MAGENTA
                r += dr
                c += dc

    return out


TEMPLATES = [
    [(-1, 0), (0, -1), (0, 0), (0, 1)],
    [(-1, -1), (-1, 0), (-1, 1), (0, 0)],
    [(-1, 0), (0, 0), (1, 0)],
    [(0, -1), (0, 0), (0, 1)],
    [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
    [(-1, 0), (0, 0), (1, 0), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)],
]


def _template_centroid_ok(template: list[tuple[int, int]]) -> bool:
    coords = np.asarray(template, dtype=int)
    return _rounded_centroid(coords) == (0, 0)


TEMPLATES = [tpl for tpl in TEMPLATES if _template_centroid_ok(tpl)]


def _shifted_pair_count(centers: list[tuple[int, int]]) -> int:
    count = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if abs(centers[i][0] - centers[j][0]) == abs(centers[i][1] - centers[j][1]):
                count += 1
    return count


def _reverse_shift_center(rank: int, center: tuple[int, int]) -> tuple[int, int]:
    r, c = center
    return (r, c + 1) if rank % 2 == 1 else (r, c - 1)


def _line_cells(centers: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    paths = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            p, q = centers[i], centers[j]
            if abs(p[0] - q[0]) == abs(p[1] - q[1]) and p != q:
                paths.append(_diag_path(p, q))
    return paths


def _sample_rows(count: int, h: int) -> list[int] | None:
    gaps = [random.choice((2, 3)) for _ in range(count - 1)]
    span = sum(gaps)
    min_start = 2
    max_start = h - 3 - span
    if max_start < min_start:
        return None
    start = random.randint(min_start, max_start)
    rows = [start]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    return rows


def _build_object_cells(center: tuple[int, int], template: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in template]


def _valid_center(center: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = center
    h, w = shape
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _available_trigger_cells(
    paths: list[list[tuple[int, int]]],
    output_blue: set[tuple[int, int]],
    input_blue: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    for path in paths:
        for cell in path[1:-1]:
            if cell in output_blue or cell in input_blue:
                continue
            if any((nr, nc) in input_blue for nr, nc in _neighbors4(*cell)):
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    dedup = []
    seen = set()
    for cell in candidates:
        if cell not in seen:
            dedup.append(cell)
            seen.add(cell)
    return dedup


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = random.choice((2, 3))
    object_count = random.randint(6, 7) if pair_count == 3 else random.randint(5, 7)

    rows = _sample_rows(object_count, h)
    if rows is None:
        return None

    pair_slots = [(2 * i + 1, 2 * i + 2) for i in range(pair_count)]
    centers: list[tuple[int, int]] = [(-1, -1)] * object_count

    for left_rank, right_rank in pair_slots:
        r1 = rows[left_rank - 1]
        r2 = rows[right_rank - 1]
        diff = r2 - r1
        placed = False
        for _ in range(80):
            sign = random.choice((-1, 1))
            c1 = random.randint(3, w - 4)
            c2 = c1 + sign * diff
            if not (2 <= c2 <= w - 3):
                continue
            trial = centers[:]
            trial[left_rank - 1] = (r1, c1)
            trial[right_rank - 1] = (r2, c2)
            known = [c for c in trial if c != (-1, -1)]
            if _shifted_pair_count(known) != len([c for c in pair_slots if trial[c[0] - 1] != (-1, -1)]):
                continue
            placed = True
            centers[left_rank - 1] = (r1, c1)
            centers[right_rank - 1] = (r2, c2)
            break
        if not placed:
            return None

    for idx in range(object_count):
        if centers[idx] != (-1, -1):
            continue
        r = rows[idx]
        placed = False
        for _ in range(100):
            c = random.randint(2, w - 3)
            trial = centers[:]
            trial[idx] = (r, c)
            known = [cc for cc in trial if cc != (-1, -1)]
            if _shifted_pair_count(known) != pair_count:
                continue
            placed = True
            centers[idx] = (r, c)
            break
        if not placed:
            return None

    if _shifted_pair_count(centers) != pair_count:
        return None

    aligned_pairs = []
    for i in range(object_count):
        for j in range(i + 1, object_count):
            if abs(centers[i][0] - centers[j][0]) == abs(centers[i][1] - centers[j][1]):
                if (i % 2) == (j % 2):
                    return None
                aligned_pairs.append((i, j))
    if len(aligned_pairs) != pair_count:
        return None

    templates = random.sample(TEMPLATES * ((object_count // len(TEMPLATES)) + 1), object_count)
    output_blue: set[tuple[int, int]] = set()
    input_blue: set[tuple[int, int]] = set()
    output_objects: list[list[tuple[int, int]]] = []
    input_objects: list[list[tuple[int, int]]] = []
    output_top_rows = []

    for rank, (center, template) in enumerate(zip(centers, templates), start=1):
        out_cells = _build_object_cells(center, template)
        in_center = _reverse_shift_center(rank, center)
        in_cells = _build_object_cells(in_center, template)
        if not all(_in_bounds((h, w), r, c) for r, c in out_cells + in_cells):
            return None
        if _touches_existing(output_blue, out_cells) or _touches_existing(input_blue, in_cells):
            return None
        output_blue.update(out_cells)
        input_blue.update(in_cells)
        output_objects.append(out_cells)
        input_objects.append(in_cells)
        output_top_rows.append(min(r for r, _ in out_cells))

    if len(set(output_top_rows)) != object_count:
        return None

    for rank, center in enumerate(centers, start=1):
        input_center = _reverse_shift_center(rank, center)
        dr = abs(center[0] - input_center[0])
        dc = abs(center[1] - input_center[1])
        if dr == dc:
            return None

    output_grid = np.zeros((h, w), dtype=int)
    input_grid = np.zeros((h, w), dtype=int)
    for cells in output_objects:
        for r, c in cells:
            output_grid[r, c] = BLUE
    for cells in input_objects:
        for r, c in cells:
            input_grid[r, c] = BLUE

    paths = _line_cells(centers)
    trigger_candidates = _available_trigger_cells(paths, output_blue, input_blue)
    if not trigger_candidates:
        return None

    trigger_count = random.randint(2, 4)
    occupied_input = set(input_blue)
    occupied_output_lines = {cell for path in paths for cell in path}
    trigger_cells: list[tuple[int, int]] = []

    for cell in trigger_candidates:
        if len(trigger_cells) >= min(trigger_count, 2):
            break
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in trigger_cells):
            continue
        trigger_cells.append(cell)
        if len(trigger_cells) == 1 and (trigger_count == 2 or random.random() < 0.5):
            break
    if not trigger_cells:
        return None

    middle_rows = range(h // 4, h - h // 4)
    middle_cols = range(w // 4, w - w // 4)
    off_candidates = [
        (r, c)
        for r in middle_rows
        for c in middle_cols
        if (r, c) not in occupied_input and (r, c) not in occupied_output_lines
    ]
    random.shuffle(off_candidates)

    for cell in off_candidates:
        if len(trigger_cells) >= trigger_count:
            break
        if cell in trigger_cells:
            continue
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in trigger_cells):
            continue
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in occupied_input):
            continue
        trigger_cells.append(cell)
    if len(trigger_cells) != trigger_count:
        return None

    for r, c in trigger_cells:
        input_grid[r, c] = MAGENTA

    solved = main(input_grid)
    if np.count_nonzero(solved == MAGENTA) <= len(trigger_cells):
        return None

    if np.array_equal(solved, input_grid):
        return None

    solved_objects = _find_blue_objects(solved)
    if len(solved_objects) == 0:
        return None

    shifted_centers = [_rounded_centroid(obj["coords"]) for obj in solved_objects]
    if _shifted_pair_count(shifted_centers) < pair_count:
        return None

    input_objects_found = _find_blue_objects(input_grid)
    if len(input_objects_found) != object_count:
        return None
    shifted_by_rule = []
    for rank, obj in enumerate(input_objects_found, start=1):
        dc = -1 if rank % 2 == 1 else 1
        coords = [(int(r), int(c + dc)) for r, c in obj["coords"] if _in_bounds((h, w), int(r), int(c + dc))]
        shifted_by_rule.append(np.asarray(coords, dtype=int))
    shifted_centers_from_input = [_rounded_centroid(coords) for coords in shifted_by_rule]
    if _shifted_pair_count(shifted_centers_from_input) != pair_count:
        return None

    reverse_centers = [_reverse_shift_center(rank, center) for rank, center in enumerate(shifted_centers_from_input, start=1)]
    if any(abs(a[0] - b[0]) == abs(a[1] - b[1]) for i, a in enumerate(reverse_centers) for b in reverse_centers[i + 1 :]):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
