from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
GRAY = 5
SOURCE_COLORS = (1, 2, 3, 6, 7, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

DIR_FROM_EDGE = {
    "top": DOWN,
    "bottom": UP,
    "left": RIGHT,
    "right": LEFT,
}

REFLECT = {
    "\\": {
        RIGHT: DOWN,
        DOWN: RIGHT,
        LEFT: UP,
        UP: LEFT,
    },
    "/": {
        RIGHT: UP,
        UP: RIGHT,
        LEFT: DOWN,
        DOWN: LEFT,
    },
}

ROTATION_LIMIT = 4
GEN_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_mirrors(grid: np.ndarray) -> tuple[list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]], dict[tuple[int, int], int]]:
    gray_cells = {tuple(map(int, cell)) for cell in np.argwhere(grid == GRAY)}
    visited: set[tuple[int, int]] = set()
    mirrors: list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]] = []
    cell_to_mirror: dict[tuple[int, int], int] = {}

    for cell in gray_cells:
        if cell in visited:
            continue
        stack = [cell]
        visited.add(cell)
        comp: list[tuple[int, int]] = []

        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nxt = (r + dr, c + dc)
                    if nxt in gray_cells and nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)

        assert len(comp) == 2, "each mirror must be an 8-connected pair"
        (r0, c0), (r1, c1) = sorted(comp)
        dr = r1 - r0
        dc = c1 - c0
        assert abs(dr) == 1 and abs(dc) == 1, "mirror cells must be diagonally adjacent"
        orient = "\\" if dr == dc else "/"

        mirror_id = len(mirrors)
        mirrors.append((orient, ((r0, c0), (r1, c1))))
        for coord in comp:
            cell_to_mirror[coord] = mirror_id

    return mirrors, cell_to_mirror


def _find_sources(grid: np.ndarray) -> list[tuple[int, int, tuple[int, int], int]]:
    h, w = grid.shape
    sources: list[tuple[int, int, tuple[int, int], int]] = []
    seen_colors: set[int] = set()

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GRAY):
                continue
            assert r in (0, h - 1) or c in (0, w - 1), "non-mirror colored cells must be on the border"

            sides = []
            if r == 0:
                sides.append("top")
            if r == h - 1:
                sides.append("bottom")
            if c == 0:
                sides.append("left")
            if c == w - 1:
                sides.append("right")
            assert len(sides) == 1, "corner sources are ambiguous"
            assert color not in seen_colors, "source colors must be distinct"
            seen_colors.add(color)
            sources.append((r, c, DIR_FROM_EDGE[sides[0]], color))

    assert sources, "expected at least one border source"
    return sources


def _simulate(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mirrors, mirror_by_cell = _find_mirrors(g)
    sources = _find_sources(g)
    out = g.copy()

    active = [(r, c, dr, dc, color, False) for r, c, (dr, dc), color in sources]
    max_steps = g.shape[0] * g.shape[1] * max(1, len(active))
    steps = 0
    stats = {
        "sources": len(sources),
        "mirrors": len(mirrors),
        "reflections": 0,
        "collisions": 0,
        "painted": 0,
    }

    while active:
        steps += 1
        assert steps <= max_steps, "beam simulation exceeded the expected bound"
        proposals: dict[tuple[int, int], list[tuple[int, int, int, int, int, bool, bool]]] = {}

        for r, c, dr, dc, color, used_reflection in active:
            nr = r + dr
            nc = c + dc
            if not _in_bounds(g.shape, nr, nc):
                continue

            reflected_now = False
            next_dr, next_dc = dr, dc

            if (nr, nc) in mirror_by_cell:
                if used_reflection:
                    continue
                mirror_id = mirror_by_cell[(nr, nc)]
                orient = mirrors[mirror_id][0]
                next_dr, next_dc = REFLECT[orient][(dr, dc)]
                nr = r + next_dr
                nc = c + next_dc
                if not _in_bounds(g.shape, nr, nc):
                    continue
                if (nr, nc) in mirror_by_cell:
                    continue
                reflected_now = True

            if int(out[nr, nc]) != BLACK:
                continue

            proposals.setdefault((nr, nc), []).append(
                (nr, nc, next_dr, next_dc, color, used_reflection or reflected_now, reflected_now)
            )

        next_active: list[tuple[int, int, int, int, int, bool]] = []
        for target, arrivals in proposals.items():
            if len(arrivals) >= 2:
                out[target] = YELLOW
                stats["collisions"] += 1
                stats["painted"] += 1
                stats["reflections"] += sum(1 for item in arrivals if item[-1])
                continue

            nr, nc, dr, dc, color, used_reflection, reflected_now = arrivals[0]
            out[nr, nc] = color
            stats["painted"] += 1
            if reflected_now:
                stats["reflections"] += 1
            next_active.append((nr, nc, dr, dc, color, used_reflection))

        active = next_active

    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each border light source using the interior diagonal gray mirrors, coloring every traversed black pixel.

    Input structure
    - Black background.
    - Distinct single-pixel light sources on the border. Their border side gives the initial direction.
    - Interior gray mirrors, each drawn as a 2-pixel diagonal pair.

    Rule
    - All sources emit simultaneously.
    - Each beam advances one pixel at a time and colors only black cells with its own source color.
    - If the next forward pixel is part of a mirror, the beam reflects according to the mirror orientation and continues from the corner-adjacent pixel instead of covering the mirror itself.
    - A beam stops when it would leave the grid, enter a non-black pixel, or hit a second mirror after already reflecting once.
    - If multiple beams reach the same black pixel on the same step, that pixel becomes yellow.
    - Mirrors remain gray.
    """
    return _simulate(grid, return_stats=False)


def _place_source(grid: np.ndarray, side: str, index: int, color: int) -> None:
    h, w = grid.shape
    if side == "top":
        grid[0, index] = color
    elif side == "bottom":
        grid[h - 1, index] = color
    elif side == "left":
        grid[index, 0] = color
    else:
        grid[index, w - 1] = color


def _candidate_mirror_cells(anchor_r: int, anchor_c: int, orient: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if orient == "\\":
        return (anchor_r, anchor_c), (anchor_r + 1, anchor_c + 1)
    return (anchor_r + 1, anchor_c), (anchor_r, anchor_c + 1)


def _can_place_mirror(grid: np.ndarray, cells: Iterable[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_list = list(cells)
    existing_gray = {tuple(map(int, cell)) for cell in np.argwhere(grid == GRAY)}

    for r, c in cell_list:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if int(grid[r, c]) != BLACK:
            return False

    for er, ec in existing_gray:
        for r, c in cell_list:
            if max(abs(er - r), abs(ec - c)) <= 1:
                return False
    return True


def _place_mirror(grid: np.ndarray, anchor_r: int, anchor_c: int, orient: str) -> bool:
    cells = _candidate_mirror_cells(anchor_r, anchor_c, orient)
    if not _can_place_mirror(grid, cells):
        return False
    for r, c in cells:
        grid[r, c] = GRAY
    return True


def _add_distractor_mirrors(grid: np.ndarray, rng: random.Random, *, preferred_corner: str, count: int) -> None:
    h, w = grid.shape
    row_values = range(max(1, h // 2 + 1), h - 2)
    col_values = range(max(1, w // 2 + 1), w - 2)
    if preferred_corner == "top_left":
        row_values = range(1, max(2, h // 2 - 1))
        col_values = range(1, max(2, w // 2 - 1))

    candidates = [(r, c, orient) for r in row_values for c in col_values for orient in ("\\", "/")]
    rng.shuffle(candidates)
    placed = 0
    for r, c, orient in candidates:
        if _place_mirror(grid, r, c, orient):
            placed += 1
            if placed == count:
                return


def _maybe_add_far_source(grid: np.ndarray, rng: random.Random) -> None:
    h, w = grid.shape
    used_colors = {int(v) for v in grid.ravel() if int(v) not in (BLACK, GRAY)}
    free_colors = [c for c in SOURCE_COLORS if c not in used_colors]
    if not free_colors:
        return

    candidates: list[tuple[str, int]] = []
    for c in range(max(2, w // 2), w - 1):
        candidates.append(("top", c))
    for r in range(max(2, h // 2), h - 1):
        candidates.append(("left", r))
    rng.shuffle(candidates)

    for side, index in candidates:
        snapshot = grid.copy()
        _place_source(snapshot, side, index, rng.choice(free_colors))
        try:
            _simulate(snapshot)
        except AssertionError:
            continue
        grid[:, :] = snapshot
        return


def _build_direct_collision(rng: random.Random) -> np.ndarray:
    h = rng.randint(10, 16)
    w = rng.randint(10, 16)
    k = rng.randint(2, min(5, h - 4, w - 4))
    grid = np.full((h, w), BLACK, dtype=int)
    c1, c2 = rng.sample(SOURCE_COLORS, 2)
    _place_source(grid, "left", k, c1)
    _place_source(grid, "top", k, c2)
    _add_distractor_mirrors(grid, rng, preferred_corner="bottom_right", count=rng.randint(0, 2))
    return grid


def _build_reflection_collision(rng: random.Random) -> np.ndarray:
    h = rng.randint(10, 16)
    turn_col = rng.randint(4, 7)
    w = 2 * turn_col + 2
    row = rng.randint(2, h - 4)
    grid = np.full((h, w), BLACK, dtype=int)
    c1, c2 = rng.sample(SOURCE_COLORS, 2)
    _place_source(grid, "left", row, c1)
    _place_source(grid, "right", row + 1, c2)
    placed = _place_mirror(grid, row - 1, turn_col, "\\")
    assert placed, "active mirror placement must succeed"
    return grid


def _build_reflect_no_collision(rng: random.Random) -> np.ndarray:
    h = rng.randint(10, 16)
    w = rng.randint(10, 16)
    k = rng.randint(2, min(4, h - 4, w - 6))
    grid = np.full((h, w), BLACK, dtype=int)
    c1, c2 = rng.sample(SOURCE_COLORS, 2)
    _place_source(grid, "left", k, c1)
    placed = _place_mirror(grid, k, k + 1, "\\")
    assert placed, "active mirror placement must succeed"
    col2 = rng.randint(max(k + 4, w // 2), w - 2)
    _place_source(grid, "top", col2, c2)
    _add_distractor_mirrors(grid, rng, preferred_corner="bottom_right", count=rng.randint(0, 2))
    if rng.random() < 0.35:
        _maybe_add_far_source(grid, rng)
    return grid


def _build_straight_no_collision(rng: random.Random) -> np.ndarray:
    h = rng.randint(10, 16)
    w = rng.randint(10, 16)
    r1 = rng.randint(2, min(5, h - 3))
    r2 = rng.randint(max(r1 + 2, h // 2), h - 2)
    grid = np.full((h, w), BLACK, dtype=int)
    c1, c2 = rng.sample(SOURCE_COLORS, 2)
    _place_source(grid, "left", r1, c1)
    _place_source(grid, "right", r2, c2)
    _add_distractor_mirrors(grid, rng, preferred_corner="bottom_right", count=rng.randint(0, 1))
    return grid


def _random_transform(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = grid
    if rng.random() < 0.5:
        out = np.fliplr(out)
    turns = rng.randrange(ROTATION_LIMIT)
    if turns:
        out = np.rot90(out, turns)
    return np.array(out, dtype=int, copy=True)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    builders = (
        ("direct_collision", _build_direct_collision),
        ("reflection_collision", _build_reflection_collision),
        ("reflect_no_collision", _build_reflect_no_collision),
        ("straight_no_collision", _build_straight_no_collision),
    )
    mode, builder = rng.choice(builders)
    grid = _random_transform(builder(rng), rng)

    try:
        out, stats = _simulate(grid, return_stats=True)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    if stats["painted"] < 5:
        return None

    if mode == "direct_collision" and not (stats["collisions"] >= 1 and stats["reflections"] == 0):
        return None
    if mode == "reflection_collision" and not (stats["collisions"] >= 1 and stats["reflections"] >= 1):
        return None
    if mode == "reflect_no_collision" and not (stats["collisions"] == 0 and stats["reflections"] >= 1):
        return None
    if mode == "straight_no_collision" and not (stats["collisions"] == 0 and stats["reflections"] == 0):
        return None

    if stats["sources"] < 1 or stats["sources"] > 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
