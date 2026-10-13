import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SEED_COLORS = (BLUE, RED, YELLOW)
MIX_COLORS = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}


def _intersection_mask(track_mask: np.ndarray) -> np.ndarray:
    h, w = track_mask.shape
    mask = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return mask
    mask[1:-1, 1:-1] = (
        track_mask[1:-1, 1:-1]
        & track_mask[:-2, 1:-1]
        & track_mask[2:, 1:-1]
        & track_mask[1:-1, :-2]
        & track_mask[1:-1, 2:]
    )
    return mask


def _label_black_components(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[bool]]:
    h, w = grid.shape
    labels = -np.ones((h, w), dtype=int)
    cells_by_id: list[list[tuple[int, int]]] = []
    touches_border: list[bool] = []
    comp_id = 0

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK or int(labels[r, c]) != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = comp_id
            cells: list[tuple[int, int]] = []
            touch = False
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touch = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if int(grid[nr, nc]) != BLACK or int(labels[nr, nc]) != -1:
                        continue
                    labels[nr, nc] = comp_id
                    q.append((int(nr), int(nc)))
            cells_by_id.append(cells)
            touches_border.append(bool(touch))
            comp_id += 1
    return labels, cells_by_id, touches_border


def _mix_colors(a: int, b: int) -> int:
    if int(a) == int(b):
        return int(a)
    return int(MIX_COLORS[frozenset((int(a), int(b)))])


def _seed_target(track_mask: np.ndarray, intersections: np.ndarray, r: int, c: int) -> tuple[tuple[int, int], str]:
    h, w = track_mask.shape
    has_lr = 0 < c < w - 1 and bool(track_mask[r, c - 1]) and bool(track_mask[r, c + 1])
    has_ud = 0 < r < h - 1 and bool(track_mask[r - 1, c]) and bool(track_mask[r + 1, c])

    if has_lr and not has_ud:
        choices: list[tuple[int, int]] = []
        cc = c - 1
        while cc >= 0 and bool(track_mask[r, cc]):
            if bool(intersections[r, cc]):
                choices.append((int(c - cc), int(cc)))
                break
            cc -= 1
        cc = c + 1
        while cc < w and bool(track_mask[r, cc]):
            if bool(intersections[r, cc]):
                choices.append((int(cc - c), int(cc)))
                break
            cc += 1
        if choices:
            choices.sort(key=lambda item: (int(item[0]), int(item[1])))
            return (int(r), int(choices[0][1])), "h"
        return (int(r), int(c)), "h"

    if has_ud and not has_lr:
        choices = []
        rr = r - 1
        while rr >= 0 and bool(track_mask[rr, c]):
            if bool(intersections[rr, c]):
                choices.append((int(r - rr), int(rr)))
                break
            rr -= 1
        rr = r + 1
        while rr < h and bool(track_mask[rr, c]):
            if bool(intersections[rr, c]):
                choices.append((int(rr - r), int(rr)))
                break
            rr += 1
        if choices:
            choices.sort(key=lambda item: (int(item[0]), int(item[1])))
            return (int(choices[0][1]), int(c)), "v"
        return (int(r), int(c)), "v"

    return (int(r), int(c)), "x"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    track_mask = g != BLACK
    intersections = _intersection_mask(track_mask)
    labels, cells_by_id, touches_border = _label_black_components(g)

    seeds = np.argwhere((g != BLACK) & (g != GREY))
    arrivals: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)

    for r, c in seeds:
        rr, cc = int(r), int(c)
        color = int(g[rr, cc])
        target, orientation = _seed_target(track_mask, intersections, rr, cc)
        out[rr, cc] = GREY
        arrivals[target].append((orientation, color))

    for (r, c), items in arrivals.items():
        if len(items) == 2 and {ori for ori, _color in items} == {"h", "v"}:
            color = _mix_colors(items[0][1], items[1][1])
            out[r, c] = color
            seen_components: set[int] = set()
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                        continue
                    if int(g[nr, nc]) != BLACK:
                        continue
                    comp_id = int(labels[nr, nc])
                    if comp_id < 0 or touches_border[comp_id] or comp_id in seen_components:
                        continue
                    for cr, cc in cells_by_id[comp_id]:
                        out[cr, cc] = color
                    seen_components.add(comp_id)
        elif items:
            out[r, c] = int(items[0][1])

    return out


def _make_line_positions(count: int, gap_lo: int, gap_hi: int) -> list[int]:
    start = random.randint(1, 3)
    pos = [int(start)]
    for _ in range(count - 1):
        pos.append(int(pos[-1] + random.randint(gap_lo, gap_hi)))
    return pos


def _build_track_grid(row_lines: list[int], col_lines: list[int]) -> np.ndarray:
    height = int(row_lines[-1] + random.randint(2, 4))
    width = int(col_lines[-1] + random.randint(2, 4))
    grid = np.full((height, width), BLACK, dtype=int)
    r0, r1 = int(row_lines[0]), int(row_lines[-1])
    c0, c1 = int(col_lines[0]), int(col_lines[-1])
    for r in row_lines:
        grid[int(r), c0 : c1 + 1] = GREY
    for c in col_lines:
        grid[r0 : r1 + 1, int(c)] = GREY
    return grid


def _select_mixed_targets(interior_rows: list[int], interior_cols: list[int], count: int) -> list[tuple[int, int]] | None:
    pairs = [(int(r), int(c)) for r in interior_rows for c in interior_cols]
    random.shuffle(pairs)
    chosen: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for r, c in pairs:
        if r in used_rows or c in used_cols:
            continue
        if any(abs(r - rr) == 1 and abs(c - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append((int(r), int(c)))
        used_rows.add(int(r))
        used_cols.add(int(c))
        if len(chosen) == count:
            return chosen
    return None


def _place_horizontal_seed(grid: np.ndarray, row_lines: list[int], col_lines: list[int], row_idx: int, col_idx: int, color: int) -> bool:
    r = int(row_lines[row_idx])
    target_c = int(col_lines[col_idx])
    options: list[int] = []

    left = int(col_lines[col_idx - 1])
    options.extend([c for c in range(left + 1, target_c) if 2 * c > left + target_c])

    right = int(col_lines[col_idx + 1])
    options.extend([c for c in range(target_c + 1, right) if 2 * c < target_c + right])

    if not options:
        return False
    c = int(random.choice(options))
    if int(grid[r, c]) != GREY:
        return False
    grid[r, c] = int(color)
    return True


def _place_vertical_seed(grid: np.ndarray, row_lines: list[int], col_lines: list[int], row_idx: int, col_idx: int, color: int) -> bool:
    c = int(col_lines[col_idx])
    target_r = int(row_lines[row_idx])
    options: list[int] = []

    up = int(row_lines[row_idx - 1])
    options.extend([r for r in range(up + 1, target_r) if 2 * r > up + target_r])

    down = int(row_lines[row_idx + 1])
    options.extend([r for r in range(target_r + 1, down) if 2 * r < target_r + down])

    if not options:
        return False
    r = int(random.choice(options))
    if int(grid[r, c]) != GREY:
        return False
    grid[r, c] = int(color)
    return True


def _choose_mixed_seed_colors() -> tuple[int, int]:
    if random.random() < 0.5:
        color = int(random.choice(SEED_COLORS))
        return color, color
    a, b = random.sample(SEED_COLORS, 2)
    return int(a), int(b)


def _add_single_targets(
    occupied_targets: set[tuple[int, int]],
    remaining_rows: list[int],
    remaining_cols: list[int],
    all_rows: list[int],
    all_cols: list[int],
    single_count: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    singles_h: list[tuple[int, int]] = []
    singles_v: list[tuple[int, int]] = []
    seed_choices = [("h", int(r)) for r in remaining_rows] + [("v", int(c)) for c in remaining_cols]
    random.shuffle(seed_choices)

    for kind, idx in seed_choices:
        if len(singles_h) + len(singles_v) >= single_count:
            break
        if kind == "h":
            col_candidates = list(all_cols)
            random.shuffle(col_candidates)
            for c in col_candidates:
                target = (int(idx), int(c))
                if target in occupied_targets:
                    continue
                singles_h.append(target)
                occupied_targets.add(target)
                break
        else:
            row_candidates = list(all_rows)
            random.shuffle(row_candidates)
            for r in row_candidates:
                target = (int(r), int(idx))
                if target in occupied_targets:
                    continue
                singles_v.append(target)
                occupied_targets.add(target)
                break

    if len(singles_h) + len(singles_v) != single_count:
        return None
    return singles_h, singles_v


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("mixed_only", "single_only", "both"))
    row_count = random.randint(3, 5)
    col_count = random.randint(3, 5)
    if mode == "both" and (row_count < 4 or col_count < 4):
        return None

    row_lines = _make_line_positions(row_count, 3, 6)
    col_lines = _make_line_positions(col_count, 3, 6)
    grid = _build_track_grid(row_lines, col_lines)

    interior_rows = list(range(1, row_count - 1))
    interior_cols = list(range(1, col_count - 1))
    if not interior_rows or not interior_cols:
        return None

    max_mixed = min(len(interior_rows), len(interior_cols), 2)
    mixed_count = 0
    single_count = 0
    if mode == "mixed_only":
        mixed_count = random.randint(1, max_mixed)
    elif mode == "single_only":
        single_count = random.randint(1, min(4, len(interior_rows) + len(interior_cols)))
    else:
        mixed_count = random.randint(1, max_mixed)
        single_cap = len(interior_rows) + len(interior_cols) - 2 * mixed_count
        if single_cap <= 0:
            return None
        single_count = random.randint(1, min(3, single_cap))

    mixed_targets: list[tuple[int, int]] = []
    if mixed_count:
        mixed_targets = _select_mixed_targets(interior_rows, interior_cols, mixed_count)
        if mixed_targets is None:
            return None

    used_rows = {r for r, _c in mixed_targets}
    used_cols = {c for _r, c in mixed_targets}
    remaining_rows = [r for r in interior_rows if r not in used_rows]
    remaining_cols = [c for c in interior_cols if c not in used_cols]

    occupied_targets = set(mixed_targets)
    singles_h: list[tuple[int, int]] = []
    singles_v: list[tuple[int, int]] = []
    if single_count:
        result = _add_single_targets(
            occupied_targets=occupied_targets,
            remaining_rows=remaining_rows,
            remaining_cols=remaining_cols,
            all_rows=interior_rows,
            all_cols=interior_cols,
            single_count=single_count,
        )
        if result is None:
            return None
        singles_h, singles_v = result

    for row_idx, col_idx in mixed_targets:
        color_h, color_v = _choose_mixed_seed_colors()
        if not _place_horizontal_seed(grid, row_lines, col_lines, row_idx, col_idx, color_h):
            return None
        if not _place_vertical_seed(grid, row_lines, col_lines, row_idx, col_idx, color_v):
            return None

    for row_idx, col_idx in singles_h:
        color = int(random.choice(SEED_COLORS))
        if not _place_horizontal_seed(grid, row_lines, col_lines, row_idx, col_idx, color):
            return None

    for row_idx, col_idx in singles_v:
        color = int(random.choice(SEED_COLORS))
        if not _place_vertical_seed(grid, row_lines, col_lines, row_idx, col_idx, color):
            return None

    seed_count = int(np.sum((grid != BLACK) & (grid != GREY)))
    if seed_count != 2 * len(mixed_targets) + len(singles_h) + len(singles_v):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
