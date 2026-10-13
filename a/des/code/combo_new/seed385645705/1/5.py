from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
MAROON = 9
DIRS = ("up", "down", "left", "right")
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _valid_pattern_at(g: np.ndarray, bg: int, r0: int, c0: int) -> np.ndarray | None:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 2 > h or c0 + 2 > w:
        return None
    patch = g[r0 : r0 + 2, c0 : c0 + 2]
    if np.any(patch == bg) or np.any(patch == GREY) or np.any(patch == MAROON):
        return None
    return patch.copy()


def _legend_mapping(g: np.ndarray, bg: int) -> dict[str, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}

    for comp in _components(g == MAROON):
        if len(comp) != 2:
            continue
        (r1, c1), (r2, c2) = sorted(comp)
        candidates: list[tuple[str, np.ndarray]] = []

        if c1 == c2 and abs(r1 - r2) == 1:
            top_r, bot_r, c = min(r1, r2), max(r1, r2), c1
            for c0 in (c - 1, c):
                pat = _valid_pattern_at(g, bg, top_r - 2, c0)
                if pat is not None:
                    candidates.append(("up", pat))
                pat = _valid_pattern_at(g, bg, bot_r + 1, c0)
                if pat is not None:
                    candidates.append(("down", pat))
        elif r1 == r2 and abs(c1 - c2) == 1:
            r, left_c, right_c = r1, min(c1, c2), max(c1, c2)
            for r0 in (r - 1, r):
                pat = _valid_pattern_at(g, bg, r0, left_c - 2)
                if pat is not None:
                    candidates.append(("left", pat))
                pat = _valid_pattern_at(g, bg, r0, right_c + 1)
                if pat is not None:
                    candidates.append(("right", pat))

        if len(candidates) != 1:
            continue
        direction, pattern = candidates[0]
        assert direction not in mapping, "each legend direction must appear exactly once"
        mapping[direction] = pattern

    assert set(mapping) == set(DIRS), "expected one maroon arrow legend for each direction"
    return mapping


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _room_infos(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, str]]:
    rooms: list[tuple[int, int, int, int, str]] = []
    for comp in _components(g == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        height, width = r1 - r0 + 1, c1 - c0 + 1
        if height < 4 or width < 4:
            continue

        border = _border_cells(r0, c0, r1, c1)
        actual = set(comp)
        if not actual <= border:
            continue
        missing = sorted(border - actual)
        if len(missing) != 2:
            continue
        if any(int(g[r, c]) != bg for r, c in missing):
            continue

        (ar, ac), (br, bc) = missing
        direction = None
        if ar == br and abs(ac - bc) == 1:
            if ar == r0:
                direction = "up"
            elif ar == r1:
                direction = "down"
        elif ac == bc and abs(ar - br) == 1:
            if ac == c0:
                direction = "left"
            elif ac == c1:
                direction = "right"
        if direction is None:
            continue

        assert len(actual) == len(border) - 2, "room border must be a rectangle with exactly one two-cell gap"
        rooms.append((int(r0), int(c0), int(r1), int(c1), direction))

    assert rooms, "expected at least one grey room"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each grey room with the 2x2 pattern selected by the side of its entrance gap.

    Input structure:
    - The grid has a dominant background color, grey one-pixel rectangular room borders, and maroon two-pixel arrows in a side legend.
    - Each room border is a rectangle with exactly one adjacent two-cell background gap on its top, bottom, left, or right side.
    - The legend has four maroon arrow pairs; each arrow points toward its adjacent 2x2 non-background color pattern, giving one pattern for each direction.

    Rule:
    - Read the four arrow-pattern legend entries to build a direction -> 2x2 pattern mapping.
    - For every grey room, use the side containing its two-cell gap as the room's entrance direction.
    - Fill background cells inside the room by tiling the matched 2x2 pattern from the room interior's top-left corner, leaving grey borders, entrance gaps, and the legend unchanged.

    Key insight:
    The entrance side of a room selects a pattern, but the side-to-pattern correspondence must first be inferred from the maroon arrow legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    bg = _background_color(g)
    patterns = _legend_mapping(g, bg)

    out = g.copy()
    for r0, c0, r1, c1, direction in _room_infos(g, bg):
        pat = patterns[direction]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(out[r, c]) == bg:
                    out[r, c] = int(pat[(r - r0 - 1) % 2, (c - c0 - 1) % 2])
    return out


def _make_patterns(bg: int) -> dict[str, np.ndarray]:
    palette = [c for c in range(10) if c not in {bg, GREY, MAROON}]
    patterns: dict[str, np.ndarray] = {}
    used: set[tuple[int, ...]] = set()
    for direction in DIRS:
        for _ in range(200):
            vals = [random.choice(palette) for _ in range(4)]
            if len(set(vals)) < 2:
                continue
            key = tuple(vals)
            if key in used:
                continue
            used.add(key)
            patterns[direction] = np.array(vals, dtype=int).reshape(2, 2)
            break
        else:
            raise RuntimeError("failed to sample a unique 2x2 pattern")
    return patterns


def _legend_patch(direction: str, pattern: np.ndarray, bg: int) -> np.ndarray:
    if direction == "up":
        patch = np.full((4, 4), bg, dtype=int)
        patch[0:2, 0:2] = pattern
        patch[2:4, 0] = MAROON
        return patch
    if direction == "down":
        patch = np.full((4, 4), bg, dtype=int)
        patch[0:2, 0] = MAROON
        patch[2:4, 0:2] = pattern
        return patch
    if direction == "left":
        patch = np.full((2, 4), bg, dtype=int)
        patch[0:2, 0:2] = pattern
        patch[0, 2:4] = MAROON
        return patch
    if direction == "right":
        patch = np.full((2, 4), bg, dtype=int)
        patch[0, 0:2] = MAROON
        patch[0:2, 2:4] = pattern
        return patch
    raise ValueError(direction)


def _reserve_box(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    mask[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _fits(mask: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = mask.shape
    r1, c1 = r0 + height - 1, c0 + width - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return not bool(mask[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)].any())


def _draw_room(grid: np.ndarray, r0: int, c0: int, height: int, width: int, direction: str) -> tuple[int, int, int, int, str]:
    r1, c1 = r0 + height - 1, c0 + width - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    if direction in ("up", "down"):
        start = random.randint(c0 + 1, c1 - 2)
        row = r0 if direction == "up" else r1
        grid[row, start : start + 2] = int(_background_color(grid))
    else:
        start = random.randint(r0 + 1, r1 - 2)
        col = c0 if direction == "left" else c1
        grid[start : start + 2, col] = int(_background_color(grid))
    return int(r0), int(c0), int(r1), int(c1), direction


def _draw_legend(grid: np.ndarray, patterns: dict[str, np.ndarray], bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    order = list(DIRS)
    random.shuffle(order)
    patches = [(direction, _legend_patch(direction, patterns[direction], bg)) for direction in order]
    total_h = sum(p.shape[0] for _, p in patches) + len(patches) - 1
    top = random.randint(1, h - total_h - 1)
    left = w - 4
    r = top
    for _, patch in patches:
        ph, pw = patch.shape
        grid[r : r + ph, left : left + pw] = patch
        r += ph + 1
    return int(top), int(left), int(top + total_h - 1), int(w - 1)


def _expected_output(inp: np.ndarray, rooms: list[tuple[int, int, int, int, str]], patterns: dict[str, np.ndarray], bg: int) -> np.ndarray:
    out = inp.copy()
    for r0, c0, r1, c1, direction in rooms:
        pat = patterns[direction]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(out[r, c]) == bg:
                    out[r, c] = int(pat[(r - r0 - 1) % 2, (c - c0 - 1) % 2])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(21, 30)
    w = random.randint(22, 30)
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8])
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    patterns = _make_patterns(bg)
    legend_box = _draw_legend(grid, patterns, bg)
    _reserve_box(blocked, legend_box, pad=1)

    n_rooms = random.randint(4, 6)
    room_dirs = list(DIRS) + [random.choice(DIRS) for _ in range(n_rooms - 4)]
    random.shuffle(room_dirs)

    rooms: list[tuple[int, int, int, int, str]] = []
    max_room_col = w - 6
    for direction in room_dirs:
        placed = False
        for _ in range(250):
            rh = random.randint(5, min(9, h - 3))
            rw = random.randint(5, min(9, max_room_col - 1))
            if max_room_col - rw < 1 or h - rh - 1 < 1:
                continue
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, max_room_col - rw)
            if not _fits(blocked, r0, c0, rh, rw, pad=1):
                continue
            rooms.append(_draw_room(grid, r0, c0, rh, rw, direction))
            _reserve_box(blocked, (r0, c0, r0 + rh - 1, c0 + rw - 1), pad=1)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, rooms, patterns, bg)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-pattern tiling input")
