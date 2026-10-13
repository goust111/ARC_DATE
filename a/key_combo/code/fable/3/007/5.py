from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEAL = 8

MAX_GENERATION_ATTEMPTS = 800
PIECE_KINDS = ("square", "hbar", "vbar")


def _piece_cells(kind: str, r: int, c: int) -> tuple[tuple[int, int], ...]:
    r = int(r)
    c = int(c)
    if kind == "square":
        cells = ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
    elif kind == "hbar":
        cells = ((r, c), (r, c + 1), (r, c + 2))
    elif kind == "vbar":
        cells = ((r, c), (r + 1, c), (r + 2, c))
    else:
        raise ValueError(f"unknown primitive kind: {kind}")
    return tuple(sorted(cells))


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= int(r) < h and 0 <= int(c) < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _candidate_tiles(cells: frozenset[tuple[int, int]]) -> list[tuple[str, tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)

    candidates: list[tuple[str, tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]] = []
    for kind in PIECE_KINDS:
        for r in range(r_min - 2, r_max + 1):
            for c in range(c_min - 2, c_max + 1):
                tile = _piece_cells(kind, r, c)
                tile_set = frozenset(tile)
                if tile_set <= cells:
                    candidates.append((kind, tile, tile_set))

    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates


def _solve_component(coords: np.ndarray) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    cells = frozenset((int(r), int(c)) for r, c in coords)
    candidates = _candidate_tiles(cells)
    by_cell: dict[tuple[int, int], list[tuple[str, tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]]] = {
        cell: [] for cell in cells
    }
    for candidate in candidates:
        for cell in candidate[2]:
            by_cell[cell].append(candidate)

    solutions: list[tuple[tuple[str, tuple[tuple[int, int], ...]], ...]] = []

    def search(
        remaining: frozenset[tuple[int, int]],
        chosen: list[tuple[str, tuple[tuple[int, int], ...]]],
    ) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(tuple(chosen))
            return

        best_cell: tuple[int, int] | None = None
        best_options: list[tuple[str, tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]] | None = None
        for cell in sorted(remaining):
            options = [tile for tile in by_cell[cell] if tile[2] <= remaining]
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_cell = cell
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_cell is not None and best_options is not None
        for kind, tile, tile_set in best_options:
            chosen.append((kind, tile))
            search(remaining - tile_set, chosen)
            chosen.pop()
            if len(solutions) > 1:
                return

    search(cells, [])
    assert len(solutions) == 1, "grey component must have one primitive decomposition"
    return list(solutions[0])


def _decompose(grid: np.ndarray) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    grey = grid == GREY
    assert np.any(grey), "expected at least one grey primitive"
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for comp in _components(grey):
        pieces.extend(_solve_component(comp))
    pieces.sort(key=lambda item: (item[1], item[0]))
    return pieces


def _bar_rays(kind: str, cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    if kind == "hbar":
        r = rows[0]
        c0, c1 = min(cols), max(cols)
        assert len(set(rows)) == 1 and c1 - c0 == 2
        return [((r, c0 - 1), (0, -1)), ((r, c1 + 1), (0, 1))]
    if kind == "vbar":
        c = cols[0]
        r0, r1 = min(rows), max(rows)
        assert len(set(cols)) == 1 and r1 - r0 == 2
        return [((r0 - 1, c), (-1, 0)), ((r1 + 1, c), (1, 0))]
    return []


def _render(shape: tuple[int, int], pieces: list[tuple[str, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    occupied = {cell for _kind, cells in pieces for cell in cells}

    for kind, cells in pieces:
        color = TEAL if kind == "square" else GREY
        for r, c in cells:
            out[r, c] = color

    h, w = shape
    for kind, cells in pieces:
        if kind == "square":
            continue
        for (r, c), (dr, dc) in _bar_rays(kind, cells):
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in occupied:
                    break
                out[r, c] = RED
                r += dr
                c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decompose gray composite blobs into behavior primitives, recolor squares, and cast bar rays.

    Input structure:
    - A black background contains one or more gray 4-connected blobs.
    - Every gray blob has exactly one non-overlapping decomposition into 2x2 squares
      and length-three bars, either 1x3 or 3x1.
    - Some bars share rows or columns with other hidden primitives, so their emitted
      rays may be shortened by squares or by other bars.

    Rule:
    - Recover the unique primitive decomposition of every gray blob.
    - Recolor each 2x2 square in place to teal, while every length-three bar remains gray.
    - From both ends of each bar, extend a red ray along the bar's long axis through
      background cells only; stop before the grid edge or before the first cell occupied
      by any primitive.

    Key insight:
    - The visible gray blobs are composites: the hidden primitive type determines whether
      a part is a passive teal blocker or a gray ray emitter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "expected only black and grey input cells"
    pieces = _decompose(g)
    return _render(g.shape, pieces)


# --------------- generator ---------------


def _group_cells(group: list[tuple[str, tuple[tuple[int, int], ...]]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for _kind, piece_cells in group:
        for cell in piece_cells:
            if cell in cells:
                return set()
            cells.add(cell)
    return cells


def _can_place_group(
    shape: tuple[int, int],
    group: list[tuple[str, tuple[tuple[int, int], ...]]],
    occupied: set[tuple[int, int]],
    padding: int,
) -> bool:
    cells = _group_cells(group)
    if not cells:
        return False
    if any(not _in_bounds(shape, cell) for cell in cells):
        return False
    if any(cell in occupied for cell in cells):
        return False

    pad = int(padding)
    if pad > 0:
        for r, c in cells:
            for er, ec in occupied:
                if max(abs(r - er), abs(c - ec)) <= pad:
                    return False
    return True


def _commit_group(
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
    occupied: set[tuple[int, int]],
    group: list[tuple[str, tuple[tuple[int, int], ...]]],
) -> None:
    pieces.extend(group)
    occupied.update(_group_cells(group))


def _place_square_blocked_hbar(
    rng: np.random.Generator,
    shape: tuple[int, int],
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(160):
        gap = int(rng.integers(1, 5))
        r = int(rng.integers(1, h - 1))
        square_r = r - int(rng.integers(0, 2))
        to_right = bool(rng.integers(0, 2))

        if to_right:
            max_bar_c = w - gap - 5
            if max_bar_c < 0:
                continue
            bar_c = int(rng.integers(0, max_bar_c + 1))
            square_c = bar_c + 3 + gap
            ray_gap = [(r, c) for c in range(bar_c + 3, square_c)]
        else:
            min_bar_c = gap + 2
            max_bar_c = w - 3
            if min_bar_c > max_bar_c:
                continue
            bar_c = int(rng.integers(min_bar_c, max_bar_c + 1))
            square_c = bar_c - gap - 2
            ray_gap = [(r, c) for c in range(square_c + 2, bar_c)]

        group = [
            ("hbar", _piece_cells("hbar", r, bar_c)),
            ("square", _piece_cells("square", square_r, square_c)),
        ]
        if any(cell in occupied for cell in ray_gap):
            continue
        if _can_place_group(shape, group, occupied, padding=1):
            _commit_group(pieces, occupied, group)
            return True
    return False


def _place_bar_blocked_vbar(
    rng: np.random.Generator,
    shape: tuple[int, int],
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = shape
    if w < 3:
        return False
    for _ in range(200):
        gap = int(rng.integers(1, 5))
        c = int(rng.integers(1, w - 1))
        downward = bool(rng.integers(0, 2))

        if downward:
            max_bar_r = h - gap - 4
            if max_bar_r < 0:
                continue
            bar_r = int(rng.integers(0, max_bar_r + 1))
            block_r = bar_r + 3 + gap
            ray_gap = [(r, c) for r in range(bar_r + 3, block_r)]
        else:
            min_bar_r = gap + 1
            max_bar_r = h - 3
            if min_bar_r > max_bar_r:
                continue
            bar_r = int(rng.integers(min_bar_r, max_bar_r + 1))
            block_r = bar_r - gap - 1
            ray_gap = [(r, c) for r in range(block_r + 1, bar_r)]

        group = [
            ("vbar", _piece_cells("vbar", bar_r, c)),
            ("hbar", _piece_cells("hbar", block_r, c - 1)),
        ]
        if any(cell in occupied for cell in ray_gap):
            continue
        if _can_place_group(shape, group, occupied, padding=1):
            _commit_group(pieces, occupied, group)
            return True
    return False


def _place_composite_cluster(
    rng: np.random.Generator,
    shape: tuple[int, int],
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(200):
        square_r = int(rng.integers(0, h - 1))
        square_c = int(rng.integers(0, w - 1))
        side = random.choice(("right", "left", "down", "up"))
        offset = int(rng.integers(0, 2))

        if side == "right":
            bar = ("hbar", _piece_cells("hbar", square_r + offset, square_c + 2))
        elif side == "left":
            bar = ("hbar", _piece_cells("hbar", square_r + offset, square_c - 3))
        elif side == "down":
            bar = ("vbar", _piece_cells("vbar", square_r + 2, square_c + offset))
        else:
            bar = ("vbar", _piece_cells("vbar", square_r - 3, square_c + offset))

        group = [("square", _piece_cells("square", square_r, square_c)), bar]
        if _can_place_group(shape, group, occupied, padding=1):
            _commit_group(pieces, occupied, group)
            return True
    return False


def _place_extra_piece(
    rng: np.random.Generator,
    shape: tuple[int, int],
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for _ in range(80):
        kind = random.choice(PIECE_KINDS)
        max_r = h - (2 if kind == "square" else 1 if kind == "hbar" else 3)
        max_c = w - (2 if kind == "square" else 3 if kind == "hbar" else 1)
        if max_r < 0 or max_c < 0:
            continue
        r = int(rng.integers(0, max_r + 1))
        c = int(rng.integers(0, max_c + 1))
        group = [(kind, _piece_cells(kind, r, c))]
        if _can_place_group(shape, group, occupied, padding=1):
            _commit_group(pieces, occupied, group)
            return True
    return False


def _input_from_pieces(
    shape: tuple[int, int],
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for _kind, cells in pieces:
        for r, c in cells:
            grid[r, c] = GREY
    return grid


def _piece_component_counts(
    shape: tuple[int, int],
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
) -> list[int]:
    owner: dict[tuple[int, int], int] = {}
    mask = np.zeros(shape, dtype=bool)
    for i, (_kind, cells) in enumerate(pieces):
        for r, c in cells:
            owner[(r, c)] = i
            mask[r, c] = True

    counts: list[int] = []
    for comp in _components(mask):
        ids = {owner[(int(r), int(c))] for r, c in comp}
        counts.append(len(ids))
    return counts


def _ray_events(
    shape: tuple[int, int],
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
) -> dict[str, int]:
    cell_owner: dict[tuple[int, int], str] = {}
    for kind, cells in pieces:
        for cell in cells:
            cell_owner[cell] = kind

    h, w = shape
    events = {"square_hit": 0, "bar_hit": 0, "edge": 0, "red_cells": 0}
    for kind, cells in pieces:
        if kind == "square":
            continue
        for (r, c), (dr, dc) in _bar_rays(kind, cells):
            length = 0
            hit_kind: str | None = None
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in cell_owner:
                    hit_kind = cell_owner[(r, c)]
                    break
                length += 1
                r += dr
                c += dc

            events["red_cells"] += length
            if hit_kind is None:
                if length > 0:
                    events["edge"] += 1
            elif length > 0:
                if hit_kind == "square":
                    events["square_hit"] += 1
                else:
                    events["bar_hit"] += 1
    return events


def _semantic_check(
    shape: tuple[int, int],
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]],
) -> bool:
    kinds = [kind for kind, _cells in pieces]
    if "square" not in kinds or "hbar" not in kinds or "vbar" not in kinds:
        return False
    if max(_piece_component_counts(shape, pieces), default=0) < 2:
        return False

    events = _ray_events(shape, pieces)
    if not (events["square_hit"] and events["bar_hit"] and events["edge"] and events["red_cells"] >= 8):
        return False

    inp = _input_from_pieces(shape, pieces)
    expected = _render(shape, pieces)
    try:
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if int(np.count_nonzero(actual == TEAL)) != 4 * kinds.count("square"):
        return False
    if int(np.count_nonzero(actual == RED)) < 6:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(16, 25))
    w = int(rng.integers(16, 25))
    shape = (h, w)
    pieces: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    occupied: set[tuple[int, int]] = set()

    builders = [_place_square_blocked_hbar, _place_bar_blocked_vbar, _place_composite_cluster]
    random.shuffle(builders)
    for builder in builders:
        if not builder(rng, shape, pieces, occupied):
            return None

    for _ in range(int(rng.integers(0, 4))):
        _place_extra_piece(rng, shape, pieces, occupied)

    if not _semantic_check(shape, pieces):
        return None
    return _input_from_pieces(shape, pieces)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid primitive-behavior input")
