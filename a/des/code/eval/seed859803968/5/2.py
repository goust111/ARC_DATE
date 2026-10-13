from __future__ import annotations

from collections import deque

import numpy as np


# ARC colors.
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

ENTRY_COLORS = (BLUE, RED, YELLOW)
PIPE_DRAW_COLORS = (GREY, ORANGE)
MIX_COLORS = {
    frozenset({BLUE, RED}): MAGENTA,
    frozenset({RED, YELLOW}): ORANGE,
    frozenset({BLUE, YELLOW}): GREEN,
    frozenset({BLUE, RED, YELLOW}): MAROON,
}

# --------------- solver ---------------


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _pipe_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == GREY) | (grid == ORANGE)


def _adjacency(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    h, w = mask.shape
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        nbrs = []
        for nr, nc in _neighbors4(rr, cc):
            if _in_bounds((h, w), nr, nc) and bool(mask[nr, nc]):
                nbrs.append((nr, nc))
        adj[(rr, cc)] = nbrs
    return adj


def _boundary_markers(
    grid: np.ndarray, mask: np.ndarray
) -> tuple[list[tuple[int, int, int, tuple[int, int]]], list[tuple[int, int]]]:
    h, w = grid.shape
    entries: list[tuple[int, int, int, tuple[int, int]]] = []
    exits: list[tuple[int, int]] = []
    seen_markers: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color == BLACK or color in PIPE_DRAW_COLORS:
                continue
            nbrs = []
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and bool(mask[nr, nc]):
                    nbrs.append((nr, nc))
            assert len(nbrs) == 1, "each boundary marker must touch exactly one pipe endpoint"
            if color in ENTRY_COLORS:
                entries.append((r, c, color, nbrs[0]))
            elif color == GREEN:
                exits.append((r, c))
            else:
                raise AssertionError("unexpected non-pipe boundary color")
            seen_markers.add((r, c))

    assert entries, "expected at least one entry marker"
    assert exits, "expected at least one exit marker"
    return entries, exits


def _distance_without_valve(
    start: tuple[int, int],
    valve: tuple[int, int],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    entry_cells: set[tuple[int, int]],
) -> int | None:
    q = deque([(start, 0)])
    seen = {valve, start}
    if start in entry_cells:
        return 0
    while q:
        cell, dist = q.popleft()
        for nxt in adj[cell]:
            if nxt in seen:
                continue
            if nxt in entry_cells:
                return dist + 1
            seen.add(nxt)
            q.append((nxt, dist + 1))
    return None


def _infer_valves(
    grid: np.ndarray,
    mask: np.ndarray,
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    entry_cells: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    valves: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for r, c in np.argwhere(grid == ORANGE):
        valve = (int(r), int(c))
        nbrs = adj[valve]
        assert len(nbrs) == 2, "each valve must connect to exactly two pipe neighbors"
        d0 = _distance_without_valve(nbrs[0], valve, adj, entry_cells)
        d1 = _distance_without_valve(nbrs[1], valve, adj, entry_cells)
        assert d0 is not None or d1 is not None, "a valve must connect to an entry on at least one side"
        if d0 is None:
            far, near = nbrs[0], nbrs[1]
        elif d1 is None:
            far, near = nbrs[1], nbrs[0]
        else:
            assert d0 != d1, "valve direction must be topologically distinguishable"
            if d0 > d1:
                far, near = nbrs[0], nbrs[1]
            else:
                far, near = nbrs[1], nbrs[0]
        valves[valve] = (far, near)
    return valves


def _can_step(
    src: tuple[int, int],
    dst: tuple[int, int],
    valves: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    if dst in valves:
        far, _near = valves[dst]
        return src == far
    if src in valves:
        _far, near = valves[src]
        return dst == near
    return True


def _mix_color(colors: set[int]) -> int:
    if not colors:
        return GREY
    if len(colors) == 1:
        return next(iter(colors))
    mixed = MIX_COLORS.get(frozenset(colors))
    assert mixed is not None, "unexpected color combination"
    return mixed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate entry colors through a grey pipe network, respecting one-way valves
    inferred from entry-side topology, then color each pipe cell by the set of
    source colors that can reach it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pipe = _pipe_mask(g)
    assert np.any(pipe), "expected at least one pipe"
    adj = _adjacency(pipe)
    entries, _exits = _boundary_markers(g, pipe)
    entry_cells = {start for _mr, _mc, _color, start in entries}
    valves = _infer_valves(g, pipe, adj, entry_cells)

    reached: dict[tuple[int, int], set[int]] = {cell: set() for cell in adj}
    starts_by_color: dict[int, set[tuple[int, int]]] = {}
    for _mr, _mc, color, start in entries:
        starts_by_color.setdefault(color, set()).add(start)

    for color, starts in starts_by_color.items():
        q = deque(starts)
        seen = set(starts)
        while q:
            cell = q.popleft()
            reached[cell].add(color)
            for nxt in adj[cell]:
                if nxt in seen or not _can_step(cell, nxt, valves):
                    continue
                seen.add(nxt)
                q.append(nxt)

    out = g.copy()
    for cell, colors in reached.items():
        out[cell] = _mix_color(colors)
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
SIZE_RANGE = (11, 18)
FAMILIES = ("plus", "elbow")
MODES = ("pair_only", "single_pair", "pair_triple", "triple_only")
TRANSFORMS = (
    lambda g: g,
    lambda g: np.rot90(g, 1).copy(),
    lambda g: np.rot90(g, 2).copy(),
    lambda g: np.rot90(g, 3).copy(),
    lambda g: np.fliplr(g).copy(),
    lambda g: np.flipud(g).copy(),
    lambda g: np.fliplr(np.rot90(g, 1)).copy(),
    lambda g: np.flipud(np.rot90(g, 1)).copy(),
)


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _endpoint_marker(shape: tuple[int, int], endpoint: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = endpoint
    if r == 1:
        return (0, c)
    if r == h - 2:
        return (h - 1, c)
    if c == 1:
        return (r, 0)
    if c == w - 2:
        return (r, w - 1)
    raise AssertionError("endpoint must be boundary-adjacent")


def _choose_source_colors(rng: np.random.Generator, count: int) -> list[int]:
    if count == 3:
        return [BLUE, RED, YELLOW]
    colors = [BLUE, RED, YELLOW]
    rng.shuffle(colors)
    return colors[:count]


def _pick_valve_index(rng: np.random.Generator, path: list[tuple[int, int]]) -> int:
    limit = min(3, len(path) - 2)
    assert limit >= 1, "need room for a valve between endpoint and junction"
    return int(rng.integers(1, limit + 1))


def _build_plus(
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    h = int(rng.integers(SIZE_RANGE[0], SIZE_RANGE[1]))
    w = int(rng.integers(SIZE_RANGE[0], SIZE_RANGE[1]))
    hr = int(rng.integers(4, h - 4))
    hc = int(rng.integers(4, w - 4))

    arms = {
        "top": _line((1, hc), (hr, hc)),
        "bottom": _line((h - 2, hc), (hr, hc)),
        "left": _line((hr, 1), (hr, hc)),
        "right": _line((hr, w - 2), (hr, hc)),
    }

    g = np.full((h, w), BLACK, dtype=int)
    cells = {cell for path in arms.values() for cell in path}
    for r, c in cells:
        g[r, c] = GREY
    return g, arms


def _build_elbow(
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    h = int(rng.integers(12, 19))
    w = int(rng.integers(12, 19))
    r1 = int(rng.integers(3, h - 6))
    c1 = int(rng.integers(3, w - 6))
    r2 = int(rng.integers(r1 + 3, h - 2))
    c2 = int(rng.integers(c1 + 3, w - 2))

    arms = {
        "left": _line((r1, 1), (r1, c1)),
        "top": _line((1, c1), (r1, c1)),
        "bottom": _line((h - 2, c2), (r2, c2)),
        "right": _line((r2, w - 2), (r2, c2)),
    }
    backbone = set(_line((r1, c1), (r1, c2)) + _line((r1, c2), (r2, c2)))

    g = np.full((h, w), BLACK, dtype=int)
    cells = backbone | {cell for path in arms.values() for cell in path}
    for r, c in cells:
        g[r, c] = GREY
    return g, arms


def _decorate_case(
    rng: np.random.Generator,
    grid: np.ndarray,
    arms: dict[str, list[tuple[int, int]]],
    mode: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    g = grid.copy()
    arm_names = list(arms)
    source_count = 2 if mode in ("pair_only", "single_pair") else 3
    blocked_count = 1 if mode in ("single_pair", "pair_triple") else 0

    rng.shuffle(arm_names)
    source_arms = arm_names[:source_count]
    exit_arms = arm_names[source_count:]
    blocked_arms = set(source_arms[:blocked_count])
    source_colors = _choose_source_colors(rng, source_count)
    color_by_arm = dict(zip(source_arms, source_colors))

    for arm in blocked_arms:
        idx = _pick_valve_index(rng, arms[arm])
        vr, vc = arms[arm][idx]
        g[vr, vc] = ORANGE

    for arm in source_arms:
        mr, mc = _endpoint_marker(g.shape, arms[arm][0])
        g[mr, mc] = color_by_arm[arm]
    for arm in exit_arms:
        mr, mc = _endpoint_marker(g.shape, arms[arm][0])
        g[mr, mc] = GREEN

    return g, source_colors, [color_by_arm[arm] for arm in blocked_arms]


def _expected_pipe_palette(
    mode: str, source_colors: list[int], blocked_colors: list[int]
) -> set[int]:
    if mode == "pair_only":
        return {MIX_COLORS[frozenset(source_colors)]}
    if mode == "triple_only":
        return {MAROON}

    blocked = blocked_colors[0]
    active = [color for color in source_colors if color != blocked]
    active_mix = _mix_color(set(active))
    if mode == "single_pair":
        return {active_mix, blocked, MIX_COLORS[frozenset(source_colors)]} if len(active) == 1 else set()
    assert mode == "pair_triple"
    return {active_mix, MAROON}


def _mode_ok(
    grid: np.ndarray,
    out: np.ndarray,
    mode: str,
    source_colors: list[int],
    blocked_colors: list[int],
) -> bool:
    pipe_colors = {int(v) for v in out[_pipe_mask(grid)]}
    if mode == "pair_only":
        return pipe_colors == {MIX_COLORS[frozenset(source_colors)]}
    if mode == "triple_only":
        return pipe_colors == {MAROON}

    blocked = blocked_colors[0]
    active = [color for color in source_colors if color != blocked]
    if mode == "single_pair":
        assert len(active) == 1
        return pipe_colors == {active[0], MIX_COLORS[frozenset(source_colors)]}
    assert mode == "pair_triple"
    return pipe_colors == {_mix_color(set(active)), MAROON}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    family = str(rng.choice(FAMILIES))
    mode = str(rng.choice(MODES))
    try:
        if family == "plus":
            grid, arms = _build_plus(rng)
        else:
            grid, arms = _build_elbow(rng)
        grid, source_colors, blocked_colors = _decorate_case(rng, grid, arms, mode)
        grid = TRANSFORMS[int(rng.integers(len(TRANSFORMS)))](grid)
        out = main(grid)
    except AssertionError:
        return None

    if not _mode_ok(grid, out, mode, source_colors, blocked_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network puzzle input")
