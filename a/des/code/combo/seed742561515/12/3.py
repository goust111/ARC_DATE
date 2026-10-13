from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (14, 24)
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _simulate_growth(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY

    frontiers: dict[int, set[tuple[int, int]]] = {}
    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        color = int(g[r, c])
        out[r, c] = color
        frontiers.setdefault(color, set()).add((int(r), int(c)))

    while frontiers:
        claims: dict[tuple[int, int], set[int]] = {}
        for color, frontier in frontiers.items():
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c, h, w):
                    if out[nr, nc] != BLACK:
                        continue
                    claims.setdefault((nr, nc), set()).add(color)

        if not claims:
            break

        next_frontiers: dict[int, set[tuple[int, int]]] = {}
        for cell, colors in claims.items():
            r, c = cell
            if len(colors) == 1:
                color = next(iter(colors))
                out[r, c] = color
                next_frontiers.setdefault(color, set()).add(cell)
            else:
                out[r, c] = MAROON
        frontiers = next_frontiers

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow all seed colors synchronously through black cells while grey walls block motion.

    Cells simultaneously reached by different colors become maroon and stop further growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _simulate_growth(g)


def _sample_nonadjacent_positions(size: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(2, size - 2))
    if len(candidates) < count:
        return None
    for _ in range(80):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= 3 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _carve_doors(length: int, door_count: int) -> set[int]:
    doors: set[int] = set()
    if door_count <= 0:
        return doors
    for _ in range(80):
        doors.clear()
        ok = True
        spans: list[tuple[int, int]] = []
        for _door in range(door_count):
            door_len = random.randint(1, 2)
            start = random.randint(1, length - door_len - 2)
            end = start + door_len - 1
            if any(not (end + 1 < s or e + 1 < start) for s, e in spans):
                ok = False
                break
            spans.append((start, end))
        if ok:
            for s, e in spans:
                for idx in range(s, e + 1):
                    doors.add(idx)
            return doors
    return doors


def _draw_vertical_barrier(grid: np.ndarray, col: int, door_count: int) -> None:
    h = grid.shape[0]
    grid[:, col] = GREY
    for r in _carve_doors(h, door_count):
        grid[r, col] = BLACK


def _draw_horizontal_barrier(grid: np.ndarray, row: int, door_count: int) -> None:
    w = grid.shape[1]
    grid[row, :] = GREY
    for c in _carve_doors(w, door_count):
        grid[row, c] = BLACK


def _try_add_short_segment(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(60):
        if random.random() < 0.5:
            seg_len = random.randint(3, min(7, w - 2))
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - seg_len - 1)
            cells = [(r, c) for c in range(c0, c0 + seg_len)]
        else:
            seg_len = random.randint(3, min(7, h - 2))
            c = random.randint(1, w - 2)
            r0 = random.randint(1, h - seg_len - 1)
            cells = [(r, c) for r in range(r0, r0 + seg_len)]
        if all(grid[r, c] == BLACK for r, c in cells):
            for r, c in cells:
                grid[r, c] = GREY
            return True
    return False


def _ring_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    bottom = top + height - 1
    right = left + width - 1
    for c in range(left, right + 1):
        cells.append((top, c))
        cells.append((bottom, c))
    for r in range(top + 1, bottom):
        cells.append((r, left))
        cells.append((r, right))
    return cells


def _try_place_ring(grid: np.ndarray, *, sealed: bool) -> bool:
    h, w = grid.shape
    max_h = min(8, h - 2)
    max_w = min(8, w - 2)
    if max_h < 4 or max_w < 4:
        return False

    for _ in range(80):
        box_h = random.randint(4, max_h)
        box_w = random.randint(4, max_w)
        top = random.randint(1, h - box_h - 1)
        left = random.randint(1, w - box_w - 1)
        bottom = top + box_h - 1
        right = left + box_w - 1

        pad_top = max(0, top - 1)
        pad_left = max(0, left - 1)
        pad_bottom = min(h - 1, bottom + 1)
        pad_right = min(w - 1, right + 1)
        if np.any(grid[pad_top : pad_bottom + 1, pad_left : pad_right + 1] != BLACK):
            continue

        for r, c in _ring_cells(top, left, box_h, box_w):
            grid[r, c] = GREY

        if not sealed:
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                c = random.randint(left + 1, right - 1)
                grid[top, c] = BLACK
            elif side == "bottom":
                c = random.randint(left + 1, right - 1)
                grid[bottom, c] = BLACK
            elif side == "left":
                r = random.randint(top + 1, bottom - 1)
                grid[r, left] = BLACK
            else:
                r = random.randint(top + 1, bottom - 1)
                grid[r, right] = BLACK
        return True
    return False


def _build_base_grid(height: int, width: int, mode: str) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)

    if mode == "collision":
        pattern = random.choice(("vertical", "horizontal", "cross"))
        if pattern in ("vertical", "cross"):
            cols = _sample_nonadjacent_positions(width, 1)
            if cols is None:
                return None
            _draw_vertical_barrier(grid, cols[0], random.randint(1, 3))
        if pattern in ("horizontal", "cross"):
            rows = _sample_nonadjacent_positions(height, 1)
            if rows is None:
                return None
            _draw_horizontal_barrier(grid, rows[0], random.randint(1, 3))
        if random.random() < 0.45:
            _try_place_ring(grid, sealed=False)
        for _ in range(random.randint(0, 2)):
            _try_add_short_segment(grid)

    elif mode == "separate_fill":
        pattern = random.choice(("single_v", "single_h", "double_v", "double_h", "cross"))
        if pattern == "single_v":
            cols = _sample_nonadjacent_positions(width, 1)
            if cols is None:
                return None
            _draw_vertical_barrier(grid, cols[0], 0)
        elif pattern == "single_h":
            rows = _sample_nonadjacent_positions(height, 1)
            if rows is None:
                return None
            _draw_horizontal_barrier(grid, rows[0], 0)
        elif pattern == "double_v":
            cols = _sample_nonadjacent_positions(width, 2)
            if cols is None:
                return None
            for col in cols:
                _draw_vertical_barrier(grid, col, 0)
        elif pattern == "double_h":
            rows = _sample_nonadjacent_positions(height, 2)
            if rows is None:
                return None
            for row in rows:
                _draw_horizontal_barrier(grid, row, 0)
        else:
            cols = _sample_nonadjacent_positions(width, 1)
            rows = _sample_nonadjacent_positions(height, 1)
            if cols is None or rows is None:
                return None
            _draw_vertical_barrier(grid, cols[0], 0)
            _draw_horizontal_barrier(grid, rows[0], 0)

    else:
        assert mode == "blocked"
        pattern = random.choice(("single_v", "single_h", "cross"))
        if pattern in ("single_v", "cross"):
            cols = _sample_nonadjacent_positions(width, 1)
            if cols is None:
                return None
            _draw_vertical_barrier(grid, cols[0], 0)
        if pattern in ("single_h", "cross"):
            rows = _sample_nonadjacent_positions(height, 1)
            if rows is None:
                return None
            _draw_horizontal_barrier(grid, rows[0], 0)
        if not _try_place_ring(grid, sealed=True):
            return None

    grey_count = int(np.count_nonzero(grid == GREY))
    if grey_count < 6 or grey_count > (height * width) // 3:
        return None
    return grid


def _pick_component_seed(component: list[tuple[int, int]], comp_set: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    cr = sum(r for r, _ in component) / len(component)
    cc = sum(c for _, c in component) / len(component)
    best_score = None
    best_cells: list[tuple[int, int]] = []
    for r, c in component:
        degree = sum((nr, nc) in comp_set for nr, nc in _neighbors4(r, c, h, w))
        dist2 = (r - cr) ** 2 + (c - cc) ** 2
        score = (degree, -dist2)
        if best_score is None or score > best_score:
            best_score = score
            best_cells = [(r, c)]
        elif score == best_score:
            best_cells.append((r, c))
    return random.choice(best_cells)


def _find_straight_conflict_pair(grid: np.ndarray, component: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    row_groups: dict[int, list[int]] = {}
    col_groups: dict[int, list[int]] = {}
    for r, c in component:
        row_groups.setdefault(r, []).append(c)
        col_groups.setdefault(c, []).append(r)

    best_pair = None
    best_gap = -1

    def consider_line(fixed: int, coords: list[int], *, horizontal: bool) -> None:
        nonlocal best_pair, best_gap
        coords.sort()
        start = 0
        while start < len(coords):
            end = start
            while end + 1 < len(coords) and coords[end + 1] == coords[end] + 1:
                end += 1
            run = coords[start : end + 1]
            run_len = len(run)
            if run_len >= 3:
                if run_len % 2 == 1:
                    a = run[0]
                    b = run[-1]
                else:
                    if random.random() < 0.5:
                        a = run[0]
                        b = run[-2]
                    else:
                        a = run[1]
                        b = run[-1]
                gap = b - a
                if gap >= 2 and gap > best_gap:
                    if horizontal:
                        best_pair = ((fixed, a), (fixed, b))
                    else:
                        best_pair = ((a, fixed), (b, fixed))
                    best_gap = gap
            start = end + 1

    for row, cols in row_groups.items():
        consider_line(row, cols, horizontal=True)
    for col, rows in col_groups.items():
        consider_line(col, rows, horizontal=False)

    if best_pair is None:
        return None
    a, b = best_pair
    if grid[a] != BLACK or grid[b] != BLACK:
        return None
    return a, b


def _seed_grid(base: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    inp = base.copy()
    colors = random.sample(SEED_COLORS, len(positions))
    random.shuffle(positions)
    for color, (r, c) in zip(colors, positions):
        inp[r, c] = color
    return inp


def _place_collision_seeds(base: np.ndarray, components: list[list[tuple[int, int]]]) -> np.ndarray | None:
    contest_components = [comp for comp in components if len(comp) >= 8]
    random.shuffle(contest_components)
    for contest in contest_components:
        pair = _find_straight_conflict_pair(base, contest)
        if pair is None:
            continue
        positions = [pair[0], pair[1]]
        remaining_components = [comp for comp in components if comp is not contest]
        random.shuffle(remaining_components)
        extra_count = random.randint(0, min(2, len(remaining_components)))
        for comp in remaining_components[:extra_count]:
            comp_set = set(comp)
            positions.append(_pick_component_seed(comp, comp_set, base.shape))
        return _seed_grid(base, positions)
    return None


def _place_separate_fill_seeds(base: np.ndarray, components: list[list[tuple[int, int]]]) -> np.ndarray | None:
    if not (2 <= len(components) <= len(SEED_COLORS)):
        return None
    positions = []
    for comp in components:
        comp_set = set(comp)
        positions.append(_pick_component_seed(comp, comp_set, base.shape))
    return _seed_grid(base, positions)


def _place_blocked_seeds(base: np.ndarray, components: list[list[tuple[int, int]]]) -> np.ndarray | None:
    if not (3 <= len(components) <= len(SEED_COLORS)):
        return None
    ordered = sorted(components, key=len, reverse=True)
    seed_count = random.randint(2, min(4, len(ordered) - 1))
    positions = []
    for comp in ordered[:seed_count]:
        comp_set = set(comp)
        positions.append(_pick_component_seed(comp, comp_set, base.shape))
    return _seed_grid(base, positions)


def _accept_sample(inp: np.ndarray, out: np.ndarray, mode: str) -> bool:
    if len(np.unique(inp)) < 3:
        return False
    if np.count_nonzero(inp == GREY) < 6:
        return False

    maroon_count = int(np.count_nonzero(out == MAROON))
    black_count = int(np.count_nonzero(out == BLACK))

    if mode == "collision":
        return maroon_count > 0
    if mode == "separate_fill":
        return maroon_count == 0 and black_count == 0
    return maroon_count == 0 and black_count > 0


def generate_input() -> np.ndarray:
    modes = ("collision", "separate_fill", "blocked")
    for _ in range(GENERATION_RETRIES):
        mode = random.choice(modes)
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        base = _build_base_grid(height, width, mode)
        if base is None:
            continue

        components = _components(base == BLACK)
        if not components:
            continue

        if mode == "collision":
            inp = _place_collision_seeds(base, components)
        elif mode == "separate_fill":
            inp = _place_separate_fill_seeds(base, components)
        else:
            inp = _place_blocked_seeds(base, components)
        if inp is None:
            continue

        out = main(inp)
        if _accept_sample(inp, out, mode):
            return inp

    raise AssertionError("failed to generate a valid sample")
