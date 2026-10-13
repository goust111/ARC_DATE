import random
from collections import deque

import numpy as np


BG_COLOR = 0
RIVER_COLOR = 8
PIER_COLOR = 9
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 7)

GENERATION_RETRIES = 500
GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (12, 20)
STEP_OPTIONS = (-1, 0, 1)
STEP_WEIGHTS = {0: 4, -1: 1, 1: 1}
MIN_PAIRS = 2
MAX_PAIRS = 4
PAIR_PLACEMENT_ATTEMPTS = 250


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            labels[r, c] = comp_id
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        queue.append((nr, nc))
    return labels, comp_id


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("line endpoints must share a row or column")


def _row_pivot_path(a: tuple[int, int], b: tuple[int, int], pivot_row: int) -> list[tuple[int, int]]:
    seg1 = _line_cells(a, (pivot_row, a[1]))
    seg2 = _line_cells((pivot_row, a[1]), (pivot_row, b[1]))
    seg3 = _line_cells((pivot_row, b[1]), b)
    return seg1 + seg2[1:] + seg3[1:]


def _col_pivot_path(a: tuple[int, int], b: tuple[int, int], pivot_col: int) -> list[tuple[int, int]]:
    seg1 = _line_cells(a, (a[0], pivot_col))
    seg2 = _line_cells((a[0], pivot_col), (b[0], pivot_col))
    seg3 = _line_cells((b[0], pivot_col), b)
    return seg1 + seg2[1:] + seg3[1:]


def _river_orientation(river_mask: np.ndarray, cell: tuple[int, int]) -> str | None:
    h, w = river_mask.shape
    r, c = cell
    up = r > 0 and river_mask[r - 1, c]
    down = r + 1 < h and river_mask[r + 1, c]
    left = c > 0 and river_mask[r, c - 1]
    right = c + 1 < w and river_mask[r, c + 1]
    if up and down and not left and not right:
        return "vertical"
    if left and right and not up and not down:
        return "horizontal"
    return None


def _side_labels(bank_labels: np.ndarray, pier: tuple[int, int], orient: str) -> tuple[int, int]:
    r, c = pier
    if orient == "vertical":
        return int(bank_labels[r, c - 1]), int(bank_labels[r, c + 1])
    return int(bank_labels[r - 1, c]), int(bank_labels[r + 1, c])


def _candidate_path(
    a: tuple[int, int], b: tuple[int, int], pier: tuple[int, int], orient: str
) -> list[tuple[int, int]] | None:
    if orient == "vertical":
        if not (min(a[1], b[1]) <= pier[1] <= max(a[1], b[1])):
            return None
        return _row_pivot_path(a, b, pier[0])
    if not (min(a[0], b[0]) <= pier[0] <= max(a[0], b[0])):
        return None
    return _col_pivot_path(a, b, pier[1])


def _path_is_valid(
    grid: np.ndarray,
    river_mask: np.ndarray,
    bank_labels: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    pier: tuple[int, int],
    orient: str,
    path: list[tuple[int, int]],
    bg: int,
) -> bool:
    label_a = int(bank_labels[a])
    label_b = int(bank_labels[b])
    if label_a == 0 or label_b == 0 or label_a == label_b:
        return False

    left_label, right_label = _side_labels(bank_labels, pier, orient)
    if left_label == 0 or right_label == 0 or left_label == right_label:
        return False
    if {label_a, label_b} != {left_label, right_label}:
        return False

    river_hits = {cell for cell in path if river_mask[cell]}
    if river_hits != {pier}:
        return False

    for cell in path:
        if cell == a or cell == b:
            continue
        value = int(grid[cell])
        if cell == pier:
            if value != PIER_COLOR:
                return False
        elif value != bg:
            return False
    return True


def _solve_pair(
    grid: np.ndarray,
    river_mask: np.ndarray,
    bank_labels: np.ndarray,
    piers: list[tuple[int, int]],
    used_piers: set[tuple[int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
    bg: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for pier in piers:
        if pier in used_piers:
            continue
        orient = _river_orientation(river_mask, pier)
        if orient is None:
            continue
        path = _candidate_path(a, b, pier, orient)
        if path is None:
            continue
        if _path_is_valid(grid, river_mask, bank_labels, a, b, pier, orient, path, bg):
            candidates.append((pier, path))

    assert len(candidates) == 1, "each color pair must match exactly one bridge pier"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color dots across the river using the bridge piers.

    The river is the connected azure/maroon path. Each maroon pixel marks the
    unique crossing point for one color pair. The bridge direction is determined
    by the river's local straight orientation at that pier.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    river_mask = np.isin(g, (RIVER_COLOR, PIER_COLOR))
    assert int(river_mask.sum()) > 0, "river must be present"

    river_labels, river_components = _label_components(river_mask)
    assert river_components == 1, "river must be a single connected path"

    bank_labels, bank_components = _label_components(~river_mask)
    assert bank_components == 2, "river must split the grid into exactly two banks"

    piers = [tuple(map(int, cell)) for cell in np.argwhere(g == PIER_COLOR)]
    assert piers, "expected at least one bridge pier"

    pair_items: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in map(int, np.unique(g)):
        if color in {bg, RIVER_COLOR, PIER_COLOR}:
            continue
        pts = np.argwhere(g == color)
        assert pts.shape == (2, 2), "each non-river color must appear exactly twice"
        a = tuple(map(int, pts[0]))
        b = tuple(map(int, pts[1]))
        pair_items.append((color, a, b))

    assert len(pair_items) == len(piers), "every bridge pier must correspond to exactly one color pair"

    out = g.copy()
    used_piers: set[tuple[int, int]] = set()
    for color, a, b in sorted(pair_items, key=lambda item: item[0]):
        pier, path = _solve_pair(g, river_mask, bank_labels, piers, used_piers, a, b, bg)
        used_piers.add(pier)
        for cell in path:
            out[cell] = color
    return out


def _random_walk_positions(length: int, low: int, high: int) -> list[int]:
    positions = [random.randint(low, high)]
    for _ in range(length - 1):
        current = positions[-1]
        deltas = [delta for delta in STEP_OPTIONS if low <= current + delta <= high]
        weights = [STEP_WEIGHTS[delta] for delta in deltas]
        positions.append(current + random.choices(deltas, weights=weights, k=1)[0])
    return positions


def _vertical_river_mask(h: int, w: int) -> tuple[np.ndarray, list[int]]:
    cols = _random_walk_positions(h, 2, w - 3)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, cols[0]] = True
    for r in range(h - 1):
        c0 = cols[r]
        c1 = cols[r + 1]
        for c in range(min(c0, c1), max(c0, c1) + 1):
            mask[r, c] = True
        mask[r + 1, c1] = True
    return mask, cols


def _horizontal_river_mask(h: int, w: int) -> tuple[np.ndarray, list[int]]:
    rows = _random_walk_positions(w, 2, h - 3)
    mask = np.zeros((h, w), dtype=bool)
    mask[rows[0], 0] = True
    for c in range(w - 1):
        r0 = rows[c]
        r1 = rows[c + 1]
        for r in range(min(r0, r1), max(r0, r1) + 1):
            mask[r, c] = True
        mask[r1, c + 1] = True
    return mask, rows


def _candidate_piers(river_mask: np.ndarray, global_orient: str) -> list[tuple[int, int]]:
    h, w = river_mask.shape
    row_counts = river_mask.sum(axis=1)
    col_counts = river_mask.sum(axis=0)
    piers: list[tuple[int, int]] = []
    for r, c in map(tuple, np.argwhere(river_mask)):
        orient = _river_orientation(river_mask, (int(r), int(c)))
        if global_orient == "vertical":
            if orient == "vertical" and row_counts[r] == 1 and 1 <= r <= h - 2 and 3 <= c <= w - 4:
                piers.append((int(r), int(c)))
        else:
            if orient == "horizontal" and col_counts[c] == 1 and 3 <= r <= h - 4 and 1 <= c <= w - 2:
                piers.append((int(r), int(c)))
    return piers


def _branch_mix(count: int) -> list[str]:
    branches = ["L"] * ((count + 1) // 2) + ["Z"] * (count // 2)
    random.shuffle(branches)
    return branches


def _path_clear_for_generation(
    inp: np.ndarray,
    out: np.ndarray,
    river_mask: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    pier: tuple[int, int],
    path: list[tuple[int, int]],
) -> bool:
    if int(inp[a]) != BG_COLOR or int(inp[b]) != BG_COLOR:
        return False
    if int(out[a]) != BG_COLOR or int(out[b]) != BG_COLOR:
        return False
    river_hits = {cell for cell in path if river_mask[cell]}
    if river_hits != {pier}:
        return False

    endpoints = {a, b}
    for cell in path:
        if cell in endpoints or cell == pier:
            continue
        if int(out[cell]) != BG_COLOR:
            return False
    return True


def _place_pair_at_pier(
    inp: np.ndarray,
    out: np.ndarray,
    river_mask: np.ndarray,
    bank_labels: np.ndarray,
    pier: tuple[int, int],
    branch: str,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    h, w = inp.shape
    orient = _river_orientation(river_mask, pier)
    if orient is None:
        return None

    if orient == "vertical":
        rp, cp = pier
        left_label, right_label = _side_labels(bank_labels, pier, orient)
        left_cols = list(range(1, cp - 1))
        right_cols = list(range(cp + 2, w - 1))
        inner_rows = list(range(1, h - 1))
        if not left_cols or not right_cols:
            return None

        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            c_left = random.choice(left_cols)
            c_right = random.choice(right_cols)
            if branch == "L":
                rows = [row for row in inner_rows if row != rp]
                if not rows:
                    return None
                if random.random() < 0.5:
                    left = (rp, c_left)
                    right = (random.choice(rows), c_right)
                else:
                    left = (random.choice(rows), c_left)
                    right = (rp, c_right)
            else:
                rows = [row for row in inner_rows if row != rp]
                if not rows:
                    return None
                left = (random.choice(rows), c_left)
                right = (random.choice(rows), c_right)

            if int(bank_labels[left]) != left_label or int(bank_labels[right]) != right_label:
                continue
            path = _row_pivot_path(left, right, rp)
            if _path_clear_for_generation(inp, out, river_mask, left, right, pier, path):
                return left, right, path
        return None

    rp, cp = pier
    top_label, bottom_label = _side_labels(bank_labels, pier, orient)
    top_rows = list(range(1, rp - 1))
    bottom_rows = list(range(rp + 2, h - 1))
    inner_cols = list(range(1, w - 1))
    if not top_rows or not bottom_rows:
        return None

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        r_top = random.choice(top_rows)
        r_bottom = random.choice(bottom_rows)
        if branch == "L":
            cols = [col for col in inner_cols if col != cp]
            if not cols:
                return None
            if random.random() < 0.5:
                top = (r_top, cp)
                bottom = (r_bottom, random.choice(cols))
            else:
                top = (r_top, random.choice(cols))
                bottom = (r_bottom, cp)
        else:
            cols = [col for col in inner_cols if col != cp]
            if not cols:
                return None
            top = (r_top, random.choice(cols))
            bottom = (r_bottom, random.choice(cols))

        if int(bank_labels[top]) != top_label or int(bank_labels[bottom]) != bottom_label:
            continue
        path = _col_pivot_path(top, bottom, cp)
        if _path_clear_for_generation(inp, out, river_mask, top, bottom, pier, path):
            return top, bottom, path
    return None


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    global_orient = random.choice(("vertical", "horizontal"))

    if global_orient == "vertical":
        river_mask, positions = _vertical_river_mask(h, w)
        turn_count = sum(positions[i] != positions[i + 1] for i in range(h - 1))
    else:
        river_mask, positions = _horizontal_river_mask(h, w)
        turn_count = sum(positions[i] != positions[i + 1] for i in range(w - 1))

    if turn_count == 0 and random.random() < 0.7:
        return None

    bank_labels, bank_components = _label_components(~river_mask)
    if bank_components != 2:
        return None

    piers = _candidate_piers(river_mask, global_orient)
    max_pairs = min(MAX_PAIRS, len(piers), len(PAIR_COLORS))
    if max_pairs < MIN_PAIRS:
        return None
    pair_count = random.randint(MIN_PAIRS, max_pairs)

    inp = np.full((h, w), BG_COLOR, dtype=int)
    out = np.full((h, w), BG_COLOR, dtype=int)
    inp[river_mask] = RIVER_COLOR
    out[river_mask] = RIVER_COLOR

    branches = _branch_mix(pair_count)
    random.shuffle(piers)
    colors = random.sample(PAIR_COLORS, pair_count)
    chosen_piers: list[tuple[int, int]] = []

    for color, branch in zip(colors, branches):
        placed = False
        for pier in piers:
            if pier in chosen_piers:
                continue
            candidate = _place_pair_at_pier(inp, out, river_mask, bank_labels, pier, branch)
            if candidate is None:
                continue
            a, b, path = candidate
            trial_inp = inp.copy()
            trial_out = out.copy()
            trial_inp[a] = color
            trial_inp[b] = color
            trial_inp[pier] = PIER_COLOR
            trial_out[a] = color
            trial_out[b] = color
            for cell in path:
                trial_out[cell] = color
            try:
                if not np.array_equal(main(trial_inp), trial_out):
                    continue
            except AssertionError:
                continue
            inp = trial_inp
            out = trial_out
            chosen_piers.append(pier)
            placed = True
            break
        if not placed:
            return None

    if len(set(chosen_piers)) != pair_count:
        return None
    if not any(branch == "L" for branch in branches) or not any(branch == "Z" for branch in branches):
        return None

    color_values = {int(v) for v in np.unique(inp)}
    if not color_values.issubset(set(PAIR_COLORS) | {BG_COLOR, RIVER_COLOR, PIER_COLOR}):
        return None
    if int((inp == BG_COLOR).sum()) <= int(river_mask.sum()):
        return None

    predicted = main(inp)
    if not np.array_equal(predicted, out):
        return None
    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _build_scene()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("failed to generate a valid river-bridge ARC input")
