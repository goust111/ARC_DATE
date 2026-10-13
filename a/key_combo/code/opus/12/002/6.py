from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

PAIR_COLORS = (RED, GREEN, YELLOW, ORANGE)
ODD_SIZES = (15, 17, 19)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _classify_walls(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    structural = np.zeros((h, w), dtype=bool)
    decorative = np.zeros((h, w), dtype=bool)
    for comp in _components(grid == BLUE):
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        target = structural if touches_border else decorative
        for r, c in comp:
            target[r, c] = True
    return structural, decorative


def _find_pairs(grid: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in PAIR_COLORS:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if not coords:
            continue
        assert len(coords) == 2, f"color {color} must appear exactly twice"
        a, b = coords
        if a[0] == b[0] and {a[1], b[1]} == {0, w - 1}:
            p1, p2 = sorted((a, b), key=lambda p: p[1])
        elif a[1] == b[1] and {a[0], b[0]} == {0, h - 1}:
            p1, p2 = sorted((a, b), key=lambda p: p[0])
        else:
            raise AssertionError(f"color {color} is not an opposite-edge pair")
        pairs.append((color, p1, p2))
    assert 2 <= len(pairs) <= 4, "expected 2-4 edge pairs"
    return pairs


def _line_cells(p1: tuple[int, int], p2: tuple[int, int]) -> list[tuple[int, int]]:
    if p1[0] == p2[0]:
        r = p1[0]
        return [(r, c) for c in range(p1[1], p2[1] + 1)]
    c = p1[1]
    return [(r, c) for r in range(p1[0], p2[0] + 1)]


def _trace_pair(
    out: np.ndarray,
    structural: np.ndarray,
    color: int,
    p1: tuple[int, int],
    p2: tuple[int, int],
) -> tuple[bool, tuple[int, int]]:
    cells = _line_cells(p1, p2)
    wall_ids = [idx for idx, (r, c) in enumerate(cells) if structural[r, c]]
    if not wall_ids:
        assert len(cells) % 2 == 1, "unblocked pairs must have a unique meeting cell"
        mid = len(cells) // 2
        for r, c in cells:
            out[r, c] = color
        out[cells[mid]] = GREY
        return False, cells[mid]

    left = wall_ids[0]
    right = wall_ids[-1]
    assert wall_ids == list(range(left, right + 1)), "wall crossing must be a single contiguous segment"
    free_left = left
    free_right = len(cells) - 1 - right
    assert free_left > 0 and free_right > 0, "wall must lie between endpoints"
    assert free_left != free_right, "blocked meeting point must be uniquely offset"

    for idx, (r, c) in enumerate(cells):
        if left <= idx <= right:
            continue
        out[r, c] = color

    grey_idx = right + 1 if free_left < free_right else left - 1
    out[cells[grey_idx]] = GREY
    return True, cells[grey_idx]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    structural, decorative = _classify_walls(g)
    out = g.copy()
    out[decorative] = BLACK
    for color, p1, p2 in _find_pairs(g):
        _trace_pair(out, structural, color, p1, p2)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase decorative blue fragments, then let opposite edge pairs leave inward trails until structural walls shift the gray meeting point.

    Input structure:
    - The background is black, while blue pixels form several 4-connected wall components.
    - Border-touching blue components are structural walls; interior blue components are decorative fragments.
    - Two to four same-color endpoint pairs sit on opposite grid edges, each pair sharing one row or one column.

    Rule:
    - Classify every blue component by border connectivity, keeping only the border-touching ones as real walls.
    - Erase all interior blue fragments to black.
    - For each same-color opposite-edge pair, march inward along its shared row or column, leaving a trail of that color.
    - If no structural wall lies on the path, color the unique midpoint gray.
    - If one contiguous structural wall segment lies on the path, it blocks the nearer side; the gray meeting mark moves to the wall-adjacent cell on the farther side, while the structural wall itself stays blue.

    Key insight:
    - Border connectivity decides which blue pixels are actual obstacles, and that obstacle classification directly changes where the inward trail system can place its gray meeting mark.
    """
    return _solve(grid)


def _rotate_shape(shape: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    cells = shape
    for _ in range(turns % 4):
        cells = [(-c, r) for r, c in cells]
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _place_blue_shape(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        if grid[r, c] != BLACK:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLUE:
                return False
    for r, c in cells:
        grid[r, c] = BLUE
    return True


def _add_random_decorations(
    grid: np.ndarray,
    clear_rows: list[int],
    blocked_row: int,
    wall_span: tuple[int, int],
) -> None:
    h, w = grid.shape
    decor_count = random.randint(1, 3)

    if clear_rows and random.random() < 0.8:
        row = random.choice(clear_rows)
        length = random.randint(2, 3)
        for _ in range(30):
            start = random.randint(2, w - length - 3)
            cells = [(row, c) for c in range(start, start + length)]
            if _place_blue_shape(grid, cells):
                decor_count -= 1
                break

    base_shapes = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]
    forbidden_rows = {blocked_row}
    left_wall, right_wall = wall_span

    for _ in range(decor_count):
        placed = False
        for _ in range(80):
            shape = _rotate_shape(random.choice(base_shapes), random.randint(0, 3))
            shape_h = 1 + max(r for r, _ in shape)
            shape_w = 1 + max(c for _, c in shape)
            top = random.randint(1, h - shape_h - 2)
            left = random.randint(1, w - shape_w - 2)
            cells = [(top + r, left + c) for r, c in shape]
            if any(r in forbidden_rows and left_wall - 1 <= c <= right_wall + 1 for r, c in cells):
                continue
            if _place_blue_shape(grid, cells):
                placed = True
                break
        if not placed:
            return


def _add_extra_structural(grid: np.ndarray, used_rows: set[int], stem_col: int) -> None:
    h, w = grid.shape
    if random.random() >= 0.35:
        return
    candidates = [r for r in range(1, h - 1) if r not in used_rows]
    random.shuffle(candidates)
    for row in candidates:
        side_left = random.random() < 0.5
        length = random.randint(2, 4)
        if side_left:
            cells = [(row, c) for c in range(length)]
        else:
            cells = [(row, c) for c in range(w - length, w)]
        ok = True
        for r, c in cells:
            if grid[r, c] != BLACK:
                ok = False
                break
        if not ok:
            continue
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLUE and (nr, nc) not in cells:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for r, c in cells:
                grid[r, c] = BLUE
            return


def _build_horizontal_instance() -> np.ndarray:
    h = random.choice(ODD_SIZES)
    w = random.choice(ODD_SIZES)
    grid = np.zeros((h, w), dtype=int)

    pair_count = random.randint(2, 4)
    colors = random.sample(PAIR_COLORS, pair_count)
    anchor_top = random.random() < 0.5
    usable_rows = list(range(2, h - 2, 2))

    for _ in range(100):
        if anchor_top:
            blocked_candidates = [r for r in usable_rows if 4 <= r <= h // 2]
        else:
            blocked_candidates = [r for r in usable_rows if h // 2 <= r <= h - 5]
        if not blocked_candidates:
            anchor_top = not anchor_top
            continue
        blocked_row = random.choice(blocked_candidates)
        if anchor_top:
            clear_candidates = [r for r in usable_rows if r >= blocked_row + 2]
        else:
            clear_candidates = [r for r in usable_rows if r <= blocked_row - 2]
        if len(clear_candidates) >= pair_count - 1:
            break
        anchor_top = not anchor_top
    else:
        raise RuntimeError("failed to choose pair rows")

    clear_rows = sorted(random.sample(clear_candidates, pair_count - 1))
    pair_rows = [blocked_row, *clear_rows]
    random.shuffle(pair_rows)
    used_rows = set(pair_rows)

    for color, row in zip(colors, pair_rows):
        grid[row, 0] = color
        grid[row, w - 1] = color

    arm_len = random.randint(2, 4)
    starts = [
        s
        for s in range(3, w - arm_len - 2)
        if min(s, w - s - arm_len) >= 4 and abs(s - (w - s - arm_len)) >= 2
    ]
    if not starts:
        raise RuntimeError("failed to place wall segment")
    start = random.choice(starts)
    end = start + arm_len - 1
    stem_col = start if random.random() < 0.5 else end

    if anchor_top:
        grid[: blocked_row + 1, stem_col] = BLUE
    else:
        grid[blocked_row:, stem_col] = BLUE
    grid[blocked_row, start : end + 1] = BLUE

    _add_extra_structural(grid, used_rows, stem_col)
    _add_random_decorations(grid, clear_rows, blocked_row, (start, end))
    return grid


def _expected_pair_info(grid: np.ndarray):
    structural, decorative = _classify_walls(grid)
    pairs = _find_pairs(grid)
    blocked_count = 0
    info = []
    for color, p1, p2 in pairs:
        cells = _line_cells(p1, p2)
        wall_ids = [idx for idx, (r, c) in enumerate(cells) if structural[r, c]]
        if wall_ids:
            blocked_count += 1
            left = wall_ids[0]
            right = wall_ids[-1]
            assert wall_ids == list(range(left, right + 1))
            assert right - left + 1 >= 2
            free_left = left
            free_right = len(cells) - 1 - right
            assert free_left >= 4 and free_right >= 4
            assert free_left != free_right
            grey = cells[right + 1] if free_left < free_right else cells[left - 1]
            info.append(("blocked", color, grey))
        else:
            assert len(cells) % 2 == 1
            grey = cells[len(cells) // 2]
            info.append(("clear", color, grey))
    return structural, decorative, info, blocked_count


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        structural, decorative, info, blocked_count = _expected_pair_info(grid)
        if blocked_count != 1:
            return False
        if not decorative.any():
            return False
        out = _solve(grid)
        if np.array_equal(out, grid):
            return False
        if np.count_nonzero(out == GREY) != len(info):
            return False
        if np.any(out[structural] != BLUE):
            return False
        if np.any(out[decorative] != BLACK):
            return False
        kinds = {kind for kind, _color, _grey in info}
        if kinds != {"blocked", "clear"}:
            return False
        for kind, _color, grey in info:
            if out[grey] != GREY:
                return False
            if kind == "blocked":
                mid_r, mid_c = grey
                if mid_r == 0 or mid_c == 0 or mid_r == out.shape[0] - 1 or mid_c == out.shape[1] - 1:
                    return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_horizontal_instance()
        if random.random() < 0.5:
            grid = grid.T.copy()
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
