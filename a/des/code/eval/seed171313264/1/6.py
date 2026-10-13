from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
GREY = 5
MIRROR_COLORS = (1, 3, 6, 7, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
SIDES = ("top", "bottom", "left", "right")
INWARD_DIR = {"top": DOWN, "bottom": UP, "left": RIGHT, "right": LEFT}

REFLECT = {
    ("\\", LEFT): DOWN,
    ("\\", RIGHT): UP,
    ("\\", UP): RIGHT,
    ("\\", DOWN): LEFT,
    ("/", LEFT): UP,
    ("/", RIGHT): DOWN,
    ("/", UP): LEFT,
    ("/", DOWN): RIGHT,
}

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
GEN_ATTEMPTS = 900


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_edge(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _steps_to_edge(pos: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = pos
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _advance(pos: tuple[int, int], direction: tuple[int, int], steps: int) -> tuple[int, int]:
    return (pos[0] + direction[0] * steps, pos[1] + direction[1] * steps)


def _cells_between(
    start: tuple[int, int], direction: tuple[int, int], steps: int
) -> list[tuple[int, int]]:
    return [_advance(start, direction, k) for k in range(1, steps + 1)]


def _partner_options(cell: tuple[int, int], mirror_kind: str) -> tuple[tuple[int, int], ...]:
    r, c = cell
    if mirror_kind == "\\":
        return ((r - 1, c - 1), (r + 1, c + 1))
    return ((r - 1, c + 1), (r + 1, c - 1))


def _target_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert pts, "expected one grey target strip"
    if len(pts) == 1:
        r, c = pts[0]
        h, w = grid.shape
        assert _is_edge(r, c, h, w), "single-cell target must lie on the edge"
        return pts
    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    h, w = grid.shape

    if len(rows) == 1:
        row = next(iter(rows))
        assert row in (0, h - 1), "horizontal target strip must lie on an edge"
        ordered_cols = sorted(c for _, c in pts)
        assert ordered_cols == list(range(ordered_cols[0], ordered_cols[-1] + 1)), (
            "horizontal target strip must be contiguous"
        )
        return [(row, c) for c in ordered_cols]

    assert len(cols) == 1, "target strip must be horizontal or vertical"
    col = next(iter(cols))
    assert col in (0, w - 1), "vertical target strip must lie on an edge"
    ordered_rows = sorted(r for r, _ in pts)
    assert ordered_rows == list(range(ordered_rows[0], ordered_rows[-1] + 1)), (
        "vertical target strip must be contiguous"
    )
    return [(r, col) for r in ordered_rows]


def _mirror_lookup(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, str]]:
    lookup: dict[tuple[int, int], tuple[int, str]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, RED, YELLOW, GREY):
            continue
        pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(pts) == 2, "each mirror color must appear exactly twice"
        (r1, c1), (r2, c2) = pts
        assert abs(r1 - r2) == 1 and abs(c1 - c2) == 1, "mirror pixels must be diagonal neighbors"
        mirror_kind = "\\" if (r1 - r2) * (c1 - c2) > 0 else "/"
        lookup[(r1, c1)] = (color, mirror_kind)
        lookup[(r2, c2)] = (color, mirror_kind)
    return lookup


def _start_state(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    emitters = np.argwhere(grid == YELLOW)
    assert emitters.shape == (1, 2), "expected one yellow emitter"
    r, c = map(int, emitters[0])
    h, w = grid.shape
    assert _is_edge(r, c, h, w), "emitter must be on the grid edge"

    directions = []
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if _in_bounds(rr, cc, h, w) and int(grid[rr, cc]) == RED:
            directions.append((dr, dc))
    assert len(directions) == 1, "expected one adjacent red direction marker"
    return (r, c), directions[0]


def _trace_mirror_sequence(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mirrors = _mirror_lookup(g)
    pos, direction = _start_state(g)
    sequence: list[int] = []
    seen: set[tuple[int, int, int, int]] = set()

    for _ in range(h * w * 4):
        state = (pos[0], pos[1], direction[0], direction[1])
        if state in seen:
            break
        seen.add(state)

        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        if not _in_bounds(nr, nc, h, w):
            break

        pos = (nr, nc)
        hit = mirrors.get(pos)
        if hit is not None:
            color, mirror_kind = hit
            sequence.append(color)
            direction = REFLECT[(mirror_kind, direction)]

        if _is_edge(pos[0], pos[1], h, w):
            break

    return sequence


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the yellow emitter's ray, record mirror colors in encounter order,
    and write that sequence into the grey target strip.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    target = _target_cells(g)
    sequence = _trace_mirror_sequence(g)

    for idx, cell in enumerate(target):
        out[cell] = sequence[idx] if idx < len(sequence) else GREY
    return out


def _edge_anchor(side: str, h: int, w: int, rng: random.Random) -> tuple[int, int]:
    if side == "top":
        return (0, rng.randint(1, w - 2))
    if side == "bottom":
        return (h - 1, rng.randint(1, w - 2))
    if side == "left":
        return (rng.randint(1, h - 2), 0)
    return (rng.randint(1, h - 2), w - 1)


def _candidate_mirror_placements(
    pos: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    path_used: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    h: int,
    w: int,
    remaining_mirrors: int,
) -> list[tuple[str, int, tuple[int, int], tuple[int, int], tuple[int, int]]]:
    placements: list[tuple[str, int, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    max_steps = _steps_to_edge(pos, direction, h, w)

    for dist in range(1, max_steps):
        segment = _cells_between(pos, direction, dist)
        prefix = segment[:-1]
        hit = segment[-1]
        if any(cell in occupied or cell in path_used for cell in prefix):
            break
        if hit in occupied or hit in path_used:
            break
        if hit in reserved:
            continue
        for mirror_kind in ("\\", "/"):
            next_direction = REFLECT[(mirror_kind, direction)]
            next_steps = _steps_to_edge(hit, next_direction, h, w)
            need = 2 if remaining_mirrors > 0 else 1
            if next_steps < need:
                continue
            for partner in _partner_options(hit, mirror_kind):
                if not _in_bounds(partner[0], partner[1], h, w):
                    continue
                if _is_edge(partner[0], partner[1], h, w):
                    continue
                if partner in occupied or partner in path_used or partner in reserved:
                    continue
                placements.append((mirror_kind, dist, hit, partner, next_direction))

    return placements


def _choose_target_strip(
    h: int,
    w: int,
    length: int,
    emitter: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    preferred_sides = list(SIDES)
    rng.shuffle(preferred_sides)
    ex_r, ex_c = emitter

    for side in preferred_sides:
        candidates: list[list[tuple[int, int]]] = []
        if side in ("top", "bottom"):
            row = 0 if side == "top" else h - 1
            for start in range(0, w - length + 1):
                cells = [(row, c) for c in range(start, start + length)]
                if row == ex_r and any(abs(c - ex_c) <= 1 for _, c in cells):
                    continue
                candidates.append(cells)
        else:
            col = 0 if side == "left" else w - 1
            for start in range(0, h - length + 1):
                cells = [(r, col) for r in range(start, start + length)]
                if col == ex_c and any(abs(r - ex_r) <= 1 for r, _ in cells):
                    continue
                candidates.append(cells)
        if candidates:
            return rng.choice(candidates)
    return None


def _build_grid(rng: random.Random) -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(*GRID_H_RANGE)
        w = rng.randint(*GRID_W_RANGE)
        mirror_count = rng.randint(1, 5 if min(h, w) >= 14 else 4)
        side = rng.choice(SIDES)
        emitter = _edge_anchor(side, h, w, rng)
        direction = INWARD_DIR[side]
        red = _advance(emitter, direction, 1)

        occupied = {emitter}
        path_used = {emitter}
        reserved = {emitter, red}
        mirrors: list[tuple[int, tuple[int, int], tuple[int, int], str]] = []
        mirror_colors = rng.sample(MIRROR_COLORS, mirror_count)

        pos = emitter
        cur_dir = direction
        ok = True

        for idx, color in enumerate(mirror_colors):
            remaining = mirror_count - idx - 1
            placements = _candidate_mirror_placements(
                pos=pos,
                direction=cur_dir,
                occupied=occupied,
                path_used=path_used,
                reserved=reserved,
                h=h,
                w=w,
                remaining_mirrors=remaining,
            )
            if not placements:
                ok = False
                break

            mirror_kind, dist, hit, partner, next_direction = rng.choice(placements)
            for cell in _cells_between(pos, cur_dir, dist):
                path_used.add(cell)
            occupied.add(hit)
            occupied.add(partner)
            mirrors.append((color, hit, partner, mirror_kind))
            pos = hit
            cur_dir = next_direction

        if not ok:
            continue

        final_steps = _steps_to_edge(pos, cur_dir, h, w)
        final_segment = _cells_between(pos, cur_dir, final_steps)
        if not final_segment:
            continue
        if any(cell in occupied or cell in path_used for cell in final_segment):
            continue
        path_used.update(final_segment)

        edge_len = w if rng.random() < 0.5 else h
        min_target = max(2, mirror_count)
        max_target = min(edge_len - 1, mirror_count + 3)
        if max_target < min_target:
            continue
        if rng.random() < 0.5:
            target_len = min_target
        else:
            target_len = rng.randint(min_target, max_target)
        target = _choose_target_strip(h, w, target_len, emitter, rng)
        if target is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        grid[emitter] = YELLOW
        grid[red] = RED
        for color, hit, partner, _mirror_kind in mirrors:
            grid[hit] = color
            grid[partner] = color
        for cell in target:
            grid[cell] = GREY

        traced = _trace_mirror_sequence(grid)
        if traced != mirror_colors:
            continue
        if len(_target_cells(grid)) < len(traced):
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _build_grid(random.Random())
