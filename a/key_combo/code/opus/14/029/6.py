from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = (BLUE, RED, GREEN)
SUBCELL_OFFSETS = {
    0: (0, 0),  # top-left
    1: (0, 1),  # top-right
    2: (1, 0),  # bottom-left
    3: (1, 1),  # bottom-right
}
TILE_PATTERN = np.array(
    [
        [GREY, GREY, GREY],
        [GREY, GREY, GREY],
        [GREY, GREY, BLACK],
    ],
    dtype=int,
)

GRID_SHAPE = (17, 17)
SEPARATOR_INDEXES = (0, 4, 8, 12, 16)
TILE_STARTS = (1, 5, 9, 13)
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class Endpoint:
    color: int
    subcell: int
    tile_r: int
    tile_c: int
    r: int
    c: int


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _build_base_grid() -> np.ndarray:
    grid = np.full(GRID_SHAPE, ORANGE, dtype=int)
    for tile_r, r0 in enumerate(TILE_STARTS):
        for tile_c, c0 in enumerate(TILE_STARTS):
            grid[r0 : r0 + 3, c0 : c0 + 3] = TILE_PATTERN
    return grid


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    indices: list[int] = []
    length = grid.shape[axis]
    for idx in range(length):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        values = set(map(int, np.unique(line)))
        if values <= {ORANGE, YELLOW}:
            indices.append(idx)
    return indices


def _tile_starts_from_grid(grid: np.ndarray) -> tuple[list[int], list[int]]:
    sep_rows = _separator_indices(grid, axis=0)
    sep_cols = _separator_indices(grid, axis=1)
    assert len(sep_rows) >= 2 and len(sep_cols) >= 2, "expected a separator lattice"
    row_starts = [sep_rows[i] + 1 for i in range(len(sep_rows) - 1) if sep_rows[i + 1] - sep_rows[i] == 4]
    col_starts = [sep_cols[i] + 1 for i in range(len(sep_cols) - 1) if sep_cols[i + 1] - sep_cols[i] == 4]
    assert row_starts and col_starts, "failed to recover 3x3 tile starts"
    return row_starts, col_starts


def _collect_endpoints(grid: np.ndarray) -> list[Endpoint]:
    row_starts, col_starts = _tile_starts_from_grid(grid)
    endpoints: list[Endpoint] = []
    for tile_r, r0 in enumerate(row_starts):
        for tile_c, c0 in enumerate(col_starts):
            for subcell, (dr, dc) in SUBCELL_OFFSETS.items():
                r = r0 + dr
                c = c0 + dc
                color = int(grid[r, c])
                if color in MARKER_COLORS:
                    endpoints.append(Endpoint(color, subcell, tile_r, tile_c, r, c))
    return endpoints


def _collect_controls(grid: np.ndarray) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != YELLOW:
                continue
            if r in (0, H - 1) or c in (0, W - 1):
                coords.append((r, c))
    return coords


def _blocked_mask(grid: np.ndarray, controls: list[tuple[int, int]]) -> np.ndarray:
    H, W = grid.shape
    blocked = np.zeros((H, W), dtype=bool)
    for r, c in controls:
        if r in (0, H - 1):
            blocked[:, c] = True
        else:
            blocked[r, :] = True
    return blocked


def _group_line_pixels(points: list[Endpoint]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = {(p.r, p.c) for p in points}
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for p in points:
        by_row[p.r].append(p.c)
        by_col[p.c].append(p.r)

    for r, cols in by_row.items():
        if len(cols) < 2:
            continue
        c0, c1 = min(cols), max(cols)
        for c in range(c0, c1 + 1):
            pixels.add((r, c))

    for c, rows in by_col.items():
        if len(rows) < 2:
            continue
        r0, r1 = min(rows), max(rows)
        for r in range(r0, r1 + 1):
            pixels.add((r, c))

    return pixels


def _components_from_pixels(pixels: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(pixels)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    endpoints = _collect_endpoints(g)
    controls = _collect_controls(g)
    blocked = _blocked_mask(g, controls)

    groups: dict[tuple[int, int], list[Endpoint]] = defaultdict(list)
    for ep in endpoints:
        groups[(ep.color, ep.subcell)].append(ep)

    kept_by_group: dict[tuple[int, int], set[tuple[int, int]]] = {}
    raw_by_group: dict[tuple[int, int], set[tuple[int, int]]] = {}
    surviving_by_group: dict[tuple[int, int], set[tuple[int, int]]] = {}

    out = g.copy()
    for key, points in groups.items():
        raw = _group_line_pixels(points)
        endpoint_pixels = {(p.r, p.c) for p in points}
        surviving = {p for p in raw if p in endpoint_pixels or not blocked[p]}
        kept: set[tuple[int, int]] = set()
        for comp in _components_from_pixels(surviving):
            if comp & endpoint_pixels:
                kept |= comp
        raw_by_group[key] = raw
        surviving_by_group[key] = surviving
        kept_by_group[key] = kept
        color = key[0]
        for r, c in kept:
            out[r, c] = color

    return {
        "output": out,
        "endpoints": endpoints,
        "controls": controls,
        "blocked": blocked,
        "raw_by_group": raw_by_group,
        "surviving_by_group": surviving_by_group,
        "kept_by_group": kept_by_group,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned same-color tile markers, cut those connections by border rays, and erase detached remnants.

    Input structure:
    - Orange separator rows and columns partition the grid into a 4x4 lattice of 3x3 tiles.
    - Each tile has a grey interior pattern with one optional blue, red, or green endpoint marker in one of four 2x2 subcell positions.
    - Yellow control pixels sit on the outer border and define full-row or full-column blocker rays.

    Rule:
    - Group endpoint markers by both marker color and relative 2x2 subcell.
    - Within each group, if two or more endpoints share a grid row, draw a horizontal connection across the full span between the leftmost and rightmost endpoint; likewise draw vertical connections for shared columns.
    - Treat every yellow control pixel as a blocker ray running straight across its row or column; connection pixels hit by a blocker are removed, except endpoint markers themselves remain.
    - After blocking, keep only those residual connection components that still contain at least one same-color endpoint marker, and erase detached components back to the underlying tile/separator background.

    Key insight:
    - The answer is not just “draw the lines” or “apply the rays”: rays first sever the line network, then any severed fragment that no longer touches an endpoint must be removed.
    """
    return np.asarray(_solve_details(np.asarray(grid, dtype=int))["output"], dtype=int)


def _endpoint_position(tile_r: int, tile_c: int, subcell: int) -> tuple[int, int]:
    dr, dc = SUBCELL_OFFSETS[subcell]
    return TILE_STARTS[tile_r] + dr, TILE_STARTS[tile_c] + dc


def _line_span_pixels(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    endpoints = [Endpoint(0, 0, 0, 0, r, c) for r, c in points]
    return _group_line_pixels(endpoints)


def _span_hits_blocked(points: list[tuple[int, int]], blocked_lines: set[int], axis: str) -> bool:
    for r, c in _line_span_pixels(points):
        if axis == "col" and c in blocked_lines:
            return True
        if axis == "row" and r in blocked_lines:
            return True
    return False


def _sample_primary_group(rng: random.Random, orientation: str, subcell: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if orientation == "H":
        tile_r = rng.randrange(4)
        left_tile = rng.randrange(0, 2)
        right_tile = rng.randrange(left_tile + 2, 4)
        endpoints = [
            _endpoint_position(tile_r, left_tile, subcell),
            _endpoint_position(tile_r, right_tile, subcell),
        ]
        span_cols = list(range(min(c for _, c in endpoints) + 2, max(c for _, c in endpoints) - 1))
        blocker_cols = []
        for c in span_cols:
            if c not in {endpoints[0][1], endpoints[1][1]}:
                blocker_cols.append(c)
        if len(blocker_cols) < 2:
            raise ValueError("not enough blocker columns")
        c1, c2 = sorted(rng.sample(blocker_cols, 2))
        if c2 - c1 < 2:
            raise ValueError("blockers leave no middle segment")
        controls = [(0 if rng.random() < 0.5 else 16, c1), (0 if rng.random() < 0.5 else 16, c2)]
        return endpoints, controls

    tile_c = rng.randrange(4)
    top_tile = rng.randrange(0, 2)
    bottom_tile = rng.randrange(top_tile + 2, 4)
    endpoints = [
        _endpoint_position(top_tile, tile_c, subcell),
        _endpoint_position(bottom_tile, tile_c, subcell),
    ]
    span_rows = list(range(min(r for r, _ in endpoints) + 2, max(r for r, _ in endpoints) - 1))
    blocker_rows = []
    for r in span_rows:
        if r not in {endpoints[0][0], endpoints[1][0]}:
            blocker_rows.append(r)
    if len(blocker_rows) < 2:
        raise ValueError("not enough blocker rows")
    r1, r2 = sorted(rng.sample(blocker_rows, 2))
    if r2 - r1 < 2:
        raise ValueError("blockers leave no middle segment")
    controls = [(r1, 0 if rng.random() < 0.5 else 16), (r2, 0 if rng.random() < 0.5 else 16)]
    return endpoints, controls


def _sample_aux_group(
    rng: random.Random,
    subcell: int,
    preferred_orientation: str,
    blocked_lines: set[int],
    blocked_axis: str,
    used_tiles: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    for _ in range(80):
        size = 3 if rng.random() < 0.45 else 2
        if size == 3 and rng.random() < 0.5:
            kind = "L"
        else:
            kind = preferred_orientation

        if kind == "H":
            tile_r = rng.randrange(4)
            cols = sorted(rng.sample(range(4), size))
            if any((tile_r, c) in used_tiles for c in cols):
                continue
            pts = [_endpoint_position(tile_r, c, subcell) for c in cols]
            if _span_hits_blocked(pts, blocked_lines, blocked_axis):
                continue
            return pts

        if kind == "V":
            tile_c = rng.randrange(4)
            rows = sorted(rng.sample(range(4), size))
            if any((r, tile_c) in used_tiles for r in rows):
                continue
            pts = [_endpoint_position(r, tile_c, subcell) for r in rows]
            if _span_hits_blocked(pts, blocked_lines, blocked_axis):
                continue
            return pts

        corner_r = rng.randrange(4)
        corner_c = rng.randrange(4)
        other_r_choices = [r for r in range(4) if r != corner_r]
        other_c_choices = [c for c in range(4) if c != corner_c]
        other_r = rng.choice(other_r_choices)
        other_c = rng.choice(other_c_choices)
        tiles = {(corner_r, corner_c), (corner_r, other_c), (other_r, corner_c)}
        if tiles & used_tiles:
            continue
        pts = [_endpoint_position(r, c, subcell) for r, c in tiles]
        if _span_hits_blocked(pts, blocked_lines, blocked_axis):
            continue
        return pts

    raise ValueError("failed to place auxiliary group")


def _build_grid_from_spec(
    group_specs: list[tuple[int, int, list[tuple[int, int]]]],
    controls: list[tuple[int, int]],
) -> np.ndarray:
    grid = _build_base_grid()
    for color, subcell, points in group_specs:
        del subcell
        for r, c in points:
            grid[r, c] = color
    for r, c in controls:
        grid[r, c] = YELLOW
    return grid


def _has_group_conflicts(group_specs: list[tuple[int, int, list[tuple[int, int]]]]) -> bool:
    seen_tiles: set[tuple[int, int]] = set()
    occupancy: dict[tuple[int, int], int] = {}
    for color, subcell, points in group_specs:
        tiles = set()
        for r, c in points:
            tile_r = TILE_STARTS.index(r - SUBCELL_OFFSETS[subcell][0])
            tile_c = TILE_STARTS.index(c - SUBCELL_OFFSETS[subcell][1])
            tiles.add((tile_r, tile_c))
        if seen_tiles & tiles:
            return True
        seen_tiles |= tiles

        for p in _line_span_pixels(points):
            prev = occupancy.get(p)
            if prev is not None and prev != color:
                return True
            occupancy[p] = color
    return False


def _validate_generated_grid(grid: np.ndarray) -> bool:
    endpoints = _collect_endpoints(grid)
    controls = _collect_controls(grid)
    if len(controls) != 2:
        return False

    by_color: dict[int, list[Endpoint]] = defaultdict(list)
    for ep in endpoints:
        by_color[ep.color].append(ep)
    if len(by_color) < 2:
        return False
    for color, pts in by_color.items():
        if len(pts) not in (2, 3):
            return False
        if len({p.subcell for p in pts}) != 1:
            return False

    details = _solve_details(grid)
    blocked = details["blocked"]
    raw_by_group = details["raw_by_group"]
    surviving_by_group = details["surviving_by_group"]
    kept_by_group = details["kept_by_group"]

    has_blocked_intersection = False
    has_dead_erasure = False
    has_kept_line = False
    for key in raw_by_group:
        raw = raw_by_group[key]
        surviving = surviving_by_group[key]
        kept = kept_by_group[key]
        endpoint_pixels = {(ep.r, ep.c) for ep in by_color[key[0]]}
        if any(blocked[r, c] and (r, c) not in endpoint_pixels for r, c in raw):
            has_blocked_intersection = True
        dead = surviving - kept
        if dead:
            has_dead_erasure = True
        if len(kept - endpoint_pixels) >= 2:
            has_kept_line = True

    if not (has_blocked_intersection and has_dead_erasure and has_kept_line):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = list(rng.sample(MARKER_COLORS, rng.choice((2, 3))))
    rng.shuffle(colors)
    available_subcells = list(SUBCELL_OFFSETS)
    rng.shuffle(available_subcells)

    primary_color = colors[0]
    primary_subcell = available_subcells[0]
    orientation = rng.choice(("H", "V"))

    try:
        primary_points, controls = _sample_primary_group(rng, orientation, primary_subcell)
    except ValueError:
        return None

    group_specs: list[tuple[int, int, list[tuple[int, int]]]] = [
        (primary_color, primary_subcell, primary_points)
    ]
    used_tiles = {
        (
            (r - SUBCELL_OFFSETS[primary_subcell][0] - 1) // 4,
            (c - SUBCELL_OFFSETS[primary_subcell][1] - 1) // 4,
        )
        for r, c in primary_points
    }

    if orientation == "H":
        blocked_lines = {c for _, c in controls}
        blocked_axis = "col"
        preferred_orientation = "V"
    else:
        blocked_lines = {r for r, _ in controls}
        blocked_axis = "row"
        preferred_orientation = "H"

    for idx, color in enumerate(colors[1:], start=1):
        subcell = available_subcells[idx]
        try:
            points = _sample_aux_group(rng, subcell, preferred_orientation, blocked_lines, blocked_axis, used_tiles)
        except ValueError:
            return None
        group_specs.append((color, subcell, points))
        for r, c in points:
            tile_r = (r - SUBCELL_OFFSETS[subcell][0] - 1) // 4
            tile_c = (c - SUBCELL_OFFSETS[subcell][1] - 1) // 4
            used_tiles.add((tile_r, tile_c))

    if _has_group_conflicts(group_specs):
        return None

    grid = _build_grid_from_spec(group_specs, controls)
    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for seed in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random(random.getrandbits(64) ^ seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
