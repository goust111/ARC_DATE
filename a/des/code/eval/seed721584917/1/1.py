from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 5
COLLISION = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400

NW = (-1, -1)
N = (-1, 0)
NE = (-1, 1)
W = (0, -1)
E = (0, 1)
SW = (1, -1)
S = (1, 0)
SE = (1, 1)
DIR_ORDER = (NW, N, NE, W, E, SW, S, SE)


def _canon_pattern(dirs: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = []
    for direction in DIR_ORDER:
        if direction in dirs:
            ordered.append(direction)
    return tuple(ordered)


PRIMARY_PATTERNS = {
    E: [
        _canon_pattern((E,)),
        _canon_pattern((E, NE)),
        _canon_pattern((E, SE)),
        _canon_pattern((E, NE, SE)),
    ],
    W: [
        _canon_pattern((W,)),
        _canon_pattern((W, NW)),
        _canon_pattern((W, SW)),
        _canon_pattern((W, NW, SW)),
    ],
    N: [
        _canon_pattern((N,)),
        _canon_pattern((N, NW)),
        _canon_pattern((N, NE)),
        _canon_pattern((N, NW, NE)),
    ],
    S: [
        _canon_pattern((S,)),
        _canon_pattern((S, SW)),
        _canon_pattern((S, SE)),
        _canon_pattern((S, SW, SE)),
    ],
    NE: [
        _canon_pattern((NE,)),
        _canon_pattern((N, NE)),
        _canon_pattern((E, NE)),
        _canon_pattern((N, E, NE)),
    ],
    NW: [
        _canon_pattern((NW,)),
        _canon_pattern((N, NW)),
        _canon_pattern((W, NW)),
        _canon_pattern((N, W, NW)),
    ],
    SE: [
        _canon_pattern((SE,)),
        _canon_pattern((S, SE)),
        _canon_pattern((E, SE)),
        _canon_pattern((S, E, SE)),
    ],
    SW: [
        _canon_pattern((SW,)),
        _canon_pattern((S, SW)),
        _canon_pattern((W, SW)),
        _canon_pattern((S, W, SW)),
    ],
}

GENERAL_PATTERNS = tuple(
    {
        *PRIMARY_PATTERNS[E],
        *PRIMARY_PATTERNS[W],
        *PRIMARY_PATTERNS[N],
        *PRIMARY_PATTERNS[S],
        *PRIMARY_PATTERNS[NE],
        *PRIMARY_PATTERNS[NW],
        *PRIMARY_PATTERNS[SE],
        *PRIMARY_PATTERNS[SW],
        _canon_pattern((N, S)),
        _canon_pattern((E, W)),
        _canon_pattern((NE, SW)),
        _canon_pattern((NW, SE)),
        _canon_pattern((N, E, S)),
        _canon_pattern((N, W, S)),
        _canon_pattern((E, NE, SE)),
        _canon_pattern((W, NW, SW)),
    }
)


def _parse_left_legend(left: np.ndarray, wall_color: int) -> dict[int, tuple[tuple[int, int], ...]] | None:
    colors = [int(v) for v in np.unique(left) if int(v) not in (BG, wall_color)]
    if len(colors) < 2:
        return None

    used = np.zeros_like(left, dtype=bool)
    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    h, w = left.shape
    for color in colors:
        coords = np.argwhere(left == color)
        if coords.shape[0] != 1:
            return None
        r, c = map(int, coords[0])
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return None
        if used[r - 1 : r + 2, c - 1 : c + 2].any():
            return None

        patch = left[r - 1 : r + 2, c - 1 : c + 2]
        if patch.shape != (3, 3):
            return None
        if int(patch[1, 1]) != color:
            return None
        if int(np.count_nonzero(patch == color)) != 1:
            return None

        dirs: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                value = int(patch[dr + 1, dc + 1])
                if value == wall_color:
                    dirs.append((dr, dc))
                elif value != BG:
                    return None
        if not dirs:
            return None

        used[r - 1 : r + 2, c - 1 : c + 2] = True
        templates[color] = _canon_pattern(dirs)

    if not np.all((left == BG) | used):
        return None
    return templates


def _find_divider_and_templates(grid: np.ndarray) -> tuple[int, int, dict[int, tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best: tuple[int, int, dict[int, tuple[tuple[int, int], ...]]] | None = None

    for divider in range(2, w - 3):
        column = g[:, divider]
        if not np.all(column == column[0]):
            continue
        wall_color = int(column[0])
        if wall_color == BG:
            continue

        left = g[:, :divider]
        right = g[:, divider + 1 :]
        if left.shape[1] < 3 or right.shape[1] < 5:
            continue

        templates = _parse_left_legend(left, wall_color)
        if templates is None:
            continue

        seed_colors = {int(v) for v in np.unique(right) if int(v) not in (BG, wall_color)}
        if seed_colors != set(templates):
            continue
        if not np.any(right == wall_color):
            continue

        valid = True
        for color in seed_colors:
            if int(np.count_nonzero(right == color)) != 1:
                valid = False
                break
        if not valid:
            continue

        score = (len(templates), -divider)
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], templates)
            best_divider = divider
            best_wall = wall_color

    assert best is not None, "expected a left legend and a vertical divider"
    return int(best_divider), int(best_wall), best[2]


def _solve_main_area(
    main_area: np.ndarray, wall_color: int, templates: dict[int, tuple[tuple[int, int], ...]]
) -> np.ndarray:
    area = np.asarray(main_area, dtype=int)
    out = area.copy()
    h, w = out.shape

    frontier: dict[int, set[tuple[int, int]]] = {}
    for color in templates:
        coords = np.argwhere(out == color)
        assert coords.shape[0] == 1, "each seed color must appear exactly once in the main area"
        frontier[color] = {tuple(map(int, coords[0]))}

    while frontier:
        proposals: dict[tuple[int, int], set[int]] = {}
        collisions_on_existing: set[tuple[int, int]] = set()

        for color, cells in frontier.items():
            directions = templates[color]
            for r, c in cells:
                if int(out[r, c]) != color:
                    continue
                for dr, dc in directions:
                    nr = r + dr
                    nc = c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    target = int(out[nr, nc])
                    if target in (wall_color, COLLISION):
                        continue
                    if target == BG:
                        proposals.setdefault((nr, nc), set()).add(color)
                    elif target != color:
                        collisions_on_existing.add((nr, nc))

        next_frontier: dict[int, set[tuple[int, int]]] = {color: set() for color in templates}
        changed = False

        for cell, colors in proposals.items():
            if len(colors) == 1 and cell not in collisions_on_existing:
                color = next(iter(colors))
                if int(out[cell]) == BG:
                    out[cell] = color
                    next_frontier[color].add(cell)
                    changed = True
            else:
                if int(out[cell]) != COLLISION:
                    out[cell] = COLLISION
                    changed = True

        for cell in collisions_on_existing:
            if int(out[cell]) != COLLISION:
                out[cell] = COLLISION
                changed = True

        frontier = {color: cells for color, cells in next_frontier.items() if cells}
        if not changed:
            break

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read direction templates from the left legend, grow each seed inside the
    right-side main area, fuse collisions into maroon, and drop the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider, wall_color, templates = _find_divider_and_templates(g)
    main_area = g[:, divider + 1 :]
    return _solve_main_area(main_area, wall_color, templates)


def _sample_pattern(
    used_patterns: set[tuple[tuple[int, int], ...]], primary: tuple[int, int] | None = None
) -> tuple[tuple[int, int], ...]:
    choices = list(PRIMARY_PATTERNS[primary] if primary is not None else GENERAL_PATTERNS)
    random.shuffle(choices)
    for pattern in choices:
        if pattern not in used_patterns:
            used_patterns.add(pattern)
            return pattern
    fallback = random.choice(choices)
    used_patterns.add(fallback)
    return fallback


def _sample_pair(axis: str, h: int, w: int, row_offset: int = 0, col_offset: int = 0):
    if axis == "h":
        if w < 7 or h < 3:
            return None
        gap = random.randint(4, w - 3)
        c1 = random.randint(1, w - gap - 2)
        r = random.randint(1, h - 2)
        return (row_offset + r, col_offset + c1), (row_offset + r, col_offset + c1 + gap), E, W
    if axis == "v":
        if h < 7 or w < 3:
            return None
        gap = random.randint(4, h - 3)
        r1 = random.randint(1, h - gap - 2)
        c = random.randint(1, w - 2)
        return (row_offset + r1, col_offset + c), (row_offset + r1 + gap, col_offset + c), S, N
    if axis == "d1":
        steps = min(h, w) - 3
        if steps < 4:
            return None
        gap = random.randint(4, steps)
        r1 = random.randint(1, h - gap - 2)
        c1 = random.randint(1, w - gap - 2)
        return (row_offset + r1, col_offset + c1), (row_offset + r1 + gap, col_offset + c1 + gap), SE, NW

    steps = min(h, w) - 3
    if steps < 4:
        return None
    gap = random.randint(4, steps)
    r1 = random.randint(gap + 1, h - 2)
    c1 = random.randint(1, w - gap - 2)
    return (row_offset + r1, col_offset + c1), (row_offset + r1 - gap, col_offset + c1 + gap), NE, SW


def _corridor_cells(start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    dr = 0 if end[0] == start[0] else (1 if end[0] > start[0] else -1)
    dc = 0 if end[1] == start[1] else (1 if end[1] > start[1] else -1)
    r, c = start
    cells = {(r, c)}
    while (r, c) != end:
        r += dr
        c += dc
        cells.add((r, c))

    padded = set()
    for rr, cc in cells:
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                padded.add((nr, nc))
    return padded


def _free_for_seed(
    area: np.ndarray,
    pos: tuple[int, int],
    placed: dict[int, tuple[int, int]],
    keep_clear_of_walls: bool,
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = area.shape
    r, c = pos
    if r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1:
        return False
    if area[r, c] != BG or (r, c) in forbidden:
        return False
    for rr, cc in placed.values():
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    if keep_clear_of_walls:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and area[nr, nc] == WALL:
                    return False
    return True


def _random_seed_pos(
    area: np.ndarray,
    placed: dict[int, tuple[int, int]],
    forbidden: set[tuple[int, int]],
    rows: range | None = None,
    cols: range | None = None,
    keep_clear_of_walls: bool = True,
) -> tuple[int, int] | None:
    h, w = area.shape
    row_values = list(rows if rows is not None else range(1, h - 1))
    col_values = list(cols if cols is not None else range(1, w - 1))
    candidates = [(r, c) for r in row_values for c in col_values]
    random.shuffle(candidates)
    for pos in candidates:
        if _free_for_seed(area, pos, placed, keep_clear_of_walls, forbidden):
            return pos
    return None


def _segment_cells(h: int, w: int, orientation: str) -> list[tuple[int, int]] | None:
    if orientation == "h":
        if w < 4:
            return None
        r = random.randint(0, h - 1)
        c0 = random.randint(0, w - 3)
        c1 = random.randint(c0 + 2, w - 1)
        return [(r, c) for c in range(c0, c1 + 1)]
    if h < 4:
        return None
    c = random.randint(0, w - 1)
    r0 = random.randint(0, h - 3)
    r1 = random.randint(r0 + 2, h - 1)
    return [(r, c) for r in range(r0, r1 + 1)]


def _scatter_random_walls(
    area: np.ndarray,
    protected: set[tuple[int, int]],
    min_segments: int,
    max_segments: int,
) -> None:
    target = random.randint(min_segments, max_segments)
    h, w = area.shape
    placed = 0
    for _ in range(80):
        if placed >= target:
            break
        orientation = random.choice(("h", "v"))
        cells = _segment_cells(h, w, orientation)
        if cells is None:
            continue
        if any(cell in protected for cell in cells):
            continue
        if not any(area[r, c] != WALL for r, c in cells):
            continue
        for r, c in cells:
            area[r, c] = WALL
        placed += 1


def _build_head_on(colors: list[int], h: int, w: int):
    area = np.zeros((h, w), dtype=int)
    used_patterns: set[tuple[tuple[int, int], ...]] = set()
    seeds: dict[int, tuple[int, int]] = {}
    templates: dict[int, tuple[tuple[int, int], ...]] = {}

    pair = _sample_pair(random.choice(("h", "v", "d1", "d2")), h, w)
    if pair is None:
        return None
    p1, p2, d1, d2 = pair
    templates[colors[0]] = _sample_pattern(used_patterns, d1)
    templates[colors[1]] = _sample_pattern(used_patterns, d2)
    seeds[colors[0]] = p1
    seeds[colors[1]] = p2

    protected = _corridor_cells(p1, p2) | {p1, p2}
    _scatter_random_walls(area, protected, min_segments=1, max_segments=3)

    for color in colors[2:]:
        pos = _random_seed_pos(area, seeds, protected)
        if pos is None:
            return None
        seeds[color] = pos
        templates[color] = _sample_pattern(used_patterns)

    for color, pos in seeds.items():
        area[pos] = color
    return area, templates


def _build_separate(colors: list[int], h: int, w: int):
    area = np.zeros((h, w), dtype=int)
    used_patterns: set[tuple[tuple[int, int], ...]] = set()
    seeds: dict[int, tuple[int, int]] = {}
    templates: dict[int, tuple[tuple[int, int], ...]] = {}

    split_vertical = random.choice((True, False))
    if split_vertical:
        if w < 9:
            return None
        split = random.randint(3, w - 4)
        area[:, split] = WALL

        r = random.randint(1, h - 2)
        left_c = random.randint(1, split - 2)
        right_c = random.randint(split + 1, w - 2)
        seeds[colors[0]] = (r, left_c)
        seeds[colors[1]] = (r, right_c)
        templates[colors[0]] = _sample_pattern(used_patterns, E)
        templates[colors[1]] = _sample_pattern(used_patterns, W)

        if random.random() < 0.8:
            branch_row = random.randint(1, h - 2)
            if random.choice((True, False)):
                area[branch_row, :split] = WALL
            else:
                area[branch_row, split + 1 :] = WALL

        left_rows = range(1, h - 1)
        left_cols = range(1, split - 1)
        right_rows = range(1, h - 1)
        right_cols = range(split + 2, w - 1)
        safe_left = (W, NW, SW, N, S)
        safe_right = (E, NE, SE, N, S)
        for color in colors[2:]:
            if random.choice((True, False)):
                pos = _random_seed_pos(area, seeds, set(), left_rows, left_cols)
                primary = random.choice(safe_left)
            else:
                pos = _random_seed_pos(area, seeds, set(), right_rows, right_cols)
                primary = random.choice(safe_right)
            if pos is None:
                return None
            seeds[color] = pos
            templates[color] = _sample_pattern(used_patterns, primary)
    else:
        if h < 9:
            return None
        split = random.randint(3, h - 4)
        area[split, :] = WALL

        c = random.randint(1, w - 2)
        top_r = random.randint(1, split - 2)
        bot_r = random.randint(split + 1, h - 2)
        seeds[colors[0]] = (top_r, c)
        seeds[colors[1]] = (bot_r, c)
        templates[colors[0]] = _sample_pattern(used_patterns, S)
        templates[colors[1]] = _sample_pattern(used_patterns, N)

        if random.random() < 0.8:
            branch_col = random.randint(1, w - 2)
            if random.choice((True, False)):
                area[:split, branch_col] = WALL
            else:
                area[split + 1 :, branch_col] = WALL

        top_rows = range(1, split - 1)
        bot_rows = range(split + 2, h - 1)
        cols = range(1, w - 1)
        safe_top = (N, NE, NW, E, W)
        safe_bottom = (S, SE, SW, E, W)
        for color in colors[2:]:
            if random.choice((True, False)):
                pos = _random_seed_pos(area, seeds, set(), top_rows, cols)
                primary = random.choice(safe_top)
            else:
                pos = _random_seed_pos(area, seeds, set(), bot_rows, cols)
                primary = random.choice(safe_bottom)
            if pos is None:
                return None
            seeds[color] = pos
            templates[color] = _sample_pattern(used_patterns, primary)

    for color, pos in seeds.items():
        if area[pos] != BG:
            return None
        area[pos] = color
    return area, templates


def _build_mixed(colors: list[int], h: int, w: int):
    area = np.zeros((h, w), dtype=int)
    used_patterns: set[tuple[tuple[int, int], ...]] = set()
    seeds: dict[int, tuple[int, int]] = {}
    templates: dict[int, tuple[tuple[int, int], ...]] = {}

    vertical = random.choice((True, False))
    if vertical:
        if w < 10:
            return None
        split = random.randint(4, w - 5)
        area[:, split] = WALL

        left_pair = _sample_pair(random.choice(("h", "v")), h, split)
        if left_pair is None:
            return None
        p1, p2, d1, d2 = left_pair
        seeds[colors[0]] = p1
        seeds[colors[1]] = p2
        templates[colors[0]] = _sample_pattern(used_patterns, d1)
        templates[colors[1]] = _sample_pattern(used_patterns, d2)

        blocked_row = random.randint(1, h - 2)
        blocked_col = random.randint(split + 2, w - 2)
        seeds[colors[2]] = (blocked_row, blocked_col)
        templates[colors[2]] = _sample_pattern(used_patterns, W)

        if split + 2 >= w:
            return None
        branch_row = random.randint(1, h - 2)
        area[branch_row, split + 1 :] = WALL
        protected = _corridor_cells(p1, p2) | {p1, p2}
    else:
        if h < 10:
            return None
        split = random.randint(4, h - 5)
        area[split, :] = WALL

        top_pair = _sample_pair(random.choice(("h", "d1", "d2")), split, w)
        if top_pair is None:
            return None
        p1, p2, d1, d2 = top_pair
        seeds[colors[0]] = p1
        seeds[colors[1]] = p2
        templates[colors[0]] = _sample_pattern(used_patterns, d1)
        templates[colors[1]] = _sample_pattern(used_patterns, d2)

        blocked_row = random.randint(split + 2, h - 2)
        blocked_col = random.randint(1, w - 2)
        seeds[colors[2]] = (blocked_row, blocked_col)
        templates[colors[2]] = _sample_pattern(used_patterns, N)

        branch_col = random.randint(1, w - 2)
        area[split + 1 :, branch_col] = WALL
        protected = _corridor_cells(p1, p2) | {p1, p2}

    _scatter_random_walls(area, protected | set(seeds.values()), min_segments=0, max_segments=2)

    for color in colors[3:]:
        pos = _random_seed_pos(area, seeds, protected)
        if pos is None:
            return None
        seeds[color] = pos
        templates[color] = _sample_pattern(used_patterns)

    for color, pos in seeds.items():
        if area[pos] != BG:
            return None
        area[pos] = color
    return area, templates


def _render_input(
    main_area: np.ndarray, templates: dict[int, tuple[tuple[int, int], ...]]
) -> np.ndarray | None:
    h, main_w = main_area.shape
    left_w = random.randint(5, 7)
    colors = list(templates)
    random.shuffle(colors)

    gaps = [random.randint(1, 2) for _ in range(len(colors) - 1)]
    legend_rows = 3 * len(colors) + sum(gaps)
    if h < legend_rows:
        return None

    spare = h - legend_rows
    top = random.randint(0, spare)

    full = np.zeros((h, left_w + 1 + main_w), dtype=int)
    full[:, left_w] = WALL
    full[:, left_w + 1 :] = main_area

    row = top
    for idx, color in enumerate(colors):
        c0 = random.randint(0, left_w - 3)
        patch = np.zeros((3, 3), dtype=int)
        patch[1, 1] = color
        for dr, dc in templates[color]:
            patch[dr + 1, dc + 1] = WALL
        full[row : row + 3, c0 : c0 + 3] = patch
        row += 3
        if idx < len(gaps):
            row += gaps[idx]

    return full


def _attempt() -> np.ndarray | None:
    mode = random.choice(("head_on", "separate", "mixed"))
    if mode == "mixed":
        n_colors = random.randint(3, 4)
    else:
        n_colors = random.randint(2, 4)

    colors = random.sample(list(SEED_COLORS), n_colors)
    min_h = 3 * n_colors + (n_colors - 1)
    h = random.randint(max(10, min_h), max(14, min_h + 6))
    w = random.randint(12, 20)

    if mode == "head_on":
        built = _build_head_on(colors, h, w)
    elif mode == "separate":
        built = _build_separate(colors, h, w)
    else:
        built = _build_mixed(colors, h, w)
    if built is None:
        return None

    main_area, templates = built
    if int(np.count_nonzero(main_area == WALL)) < 3:
        return None

    full = _render_input(main_area, templates)
    if full is None:
        return None

    expected = _solve_main_area(main_area, WALL, templates)
    try:
        actual = main(full)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    changed = int(np.count_nonzero(actual != main_area))
    maroons = int(np.count_nonzero(actual == COLLISION))
    if changed < max(6, 2 * n_colors):
        return None
    if mode == "head_on" and maroons == 0:
        return None
    if mode == "separate" and maroons != 0:
        return None
    if mode == "mixed" and maroons == 0:
        return None

    return full


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
