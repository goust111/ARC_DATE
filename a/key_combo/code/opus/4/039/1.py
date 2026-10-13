from __future__ import annotations

from collections import deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4


Coord = tuple[int, int]

GEN_RETRIES = 400
GRID_SIZES = (13, 15, 17, 19)
LAYOUTS = ("vertical", "horizontal", "cross", "double_vertical", "double_horizontal")


def _reflection_orbits(shape: tuple[int, int]) -> list[tuple[Coord, ...]]:
    height, width = map(int, shape)
    seen: set[frozenset[Coord]] = set()
    orbits: list[tuple[Coord, ...]] = []
    for r in range(height):
        for c in range(width):
            orbit = frozenset(
                {
                    (r, c),
                    (height - 1 - r, c),
                    (r, width - 1 - c),
                    (height - 1 - r, width - 1 - c),
                }
            )
            if orbit in seen:
                continue
            seen.add(orbit)
            orbits.append(tuple(sorted(orbit)))
    return orbits


def _orbit_key(shape: tuple[int, int], cell: Coord) -> frozenset[Coord]:
    r, c = map(int, cell)
    h, w = map(int, shape)
    return frozenset(
        {
            (r, c),
            (h - 1 - r, c),
            (r, w - 1 - c),
            (h - 1 - r, w - 1 - c),
        }
    )


def _find_marker_color(grid: np.ndarray, orbits: list[tuple[Coord, ...]]) -> int:
    candidates: list[int] = []
    for color in sorted(map(int, np.unique(grid))):
        has_marker_orbit = False
        valid = True
        for orbit in orbits:
            values = [int(grid[r, c]) for r, c in orbit]
            if color not in values:
                continue
            known = {v for v in values if v != color}
            if len(known) != 1:
                valid = False
                break
            has_marker_orbit = True
        if valid and has_marker_orbit:
            candidates.append(color)
    assert len(candidates) == 1, "expected exactly one reflection-marker color"
    return candidates[0]


def _repair_barrier(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    orbits = _reflection_orbits(g.shape)
    marker = _find_marker_color(g, orbits)
    out = g.copy()
    for orbit in orbits:
        values = [int(g[r, c]) for r, c in orbit]
        if marker not in values:
            continue
        known = {v for v in values if v != marker}
        assert len(known) == 1
        fill = known.pop()
        for r, c in orbit:
            if int(g[r, c]) == marker:
                out[r, c] = fill
    return out


def _exterior_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[Coord] = deque()

    def push(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not seen[r, c] and int(g[r, c]) == Color.BLACK:
            seen[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            push(nr, nc)
    return seen


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _classify_background(grid: np.ndarray) -> np.ndarray:
    repaired = np.asarray(grid, dtype=int)
    exterior = _exterior_mask(repaired)
    out = repaired.copy()
    out[(repaired == Color.BLACK) & exterior] = Color.GREEN
    out[(repaired == Color.BLACK) & ~exterior] = Color.RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a reflection-broken wall pattern, then color exterior and enclosed background regions differently.

    Input structure:
    - Black background fills the canvas.
    - Blue pixels form an almost biaxially symmetric internal wall network made of rectangle borders and dividers.
    - Yellow pixels sit on wall orbits as missing markers; each marker orbit has one yellow cell and the other reflected partners stay blue.

    Rule:
    - Group cells into reflection orbits under horizontal mirror, vertical mirror, and 180-degree rotation.
    - Infer the unique marker color whose orbit-mates always agree on one other color, and replace each marker cell by that orbit color to restore the full symmetric wall.
    - Treat the repaired non-black pixels as impassable walls and flood-fill black background from the grid border using 4-connectivity.
    - Recolor border-reachable background green, recolor unreached enclosed background red, and keep the repaired blue wall unchanged.

    Key insight:
    - The yellow cells are not final objects; they are wall gaps whose reflected blue partners reveal the missing barrier needed before inside/outside classification becomes correct.
    """
    repaired = _repair_barrier(np.asarray(grid, dtype=int))
    return _classify_background(repaired)


def _draw_rect_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = Color.BLUE
    grid[r1, c0 : c1 + 1] = Color.BLUE
    grid[r0 : r1 + 1, c0] = Color.BLUE
    grid[r0 : r1 + 1, c1] = Color.BLUE


def _wall_candidates(layout: str, h: int, w: int) -> tuple[np.ndarray, dict[str, object]] | None:
    cr, cc = h // 2, w // 2
    hh_choices = range(3, min(6, cr - 1) + 1)
    hw_choices = range(4, min(6, cc - 1) + 1)
    trials = [(hh, hw) for hh in hh_choices for hw in hw_choices]
    random.shuffle(trials)

    for hh, hw in trials:
        if layout == "double_vertical" and hw < 5:
            continue
        if layout == "double_horizontal" and hh < 5:
            continue

        grid = np.zeros((h, w), dtype=int)
        r0, r1 = cr - hh, cr + hh
        c0, c1 = cc - hw, cc + hw
        _draw_rect_border(grid, r0, c0, r1, c1)

        divider_cells: list[Coord] = []
        if layout in {"vertical", "cross"}:
            for r in range(r0 + 1, r1):
                grid[r, cc] = Color.BLUE
                divider_cells.append((r, cc))
        if layout in {"horizontal", "cross"}:
            for c in range(c0 + 1, c1):
                grid[cr, c] = Color.BLUE
                divider_cells.append((cr, c))
        if layout == "double_vertical":
            d_choices = [d for d in range(2, hw - 1) if hw - d - 1 >= 2]
            if not d_choices:
                continue
            d = random.choice(d_choices)
            for col in (cc - d, cc + d):
                for r in range(r0 + 1, r1):
                    grid[r, col] = Color.BLUE
                    divider_cells.append((r, col))
        elif layout == "double_horizontal":
            d_choices = [d for d in range(2, hh - 1) if hh - d - 1 >= 2]
            if not d_choices:
                continue
            d = random.choice(d_choices)
            for row in (cr - d, cr + d):
                for c in range(c0 + 1, c1):
                    grid[row, c] = Color.BLUE
                    divider_cells.append((row, c))

        wall_pixels = int(np.count_nonzero(grid == Color.BLUE))
        if not (24 <= wall_pixels <= 40):
            continue
        enclosed = _count_components((grid == Color.BLACK) & ~_exterior_mask(grid))
        if enclosed < 2:
            continue

        return grid, {
            "layout": layout,
            "bounds": (r0, c0, r1, c1),
            "center": (cr, cc),
            "divider_cells": divider_cells,
        }
    return None


def _pick_gap_cells(shape: tuple[int, int], meta: dict[str, object]) -> list[Coord]:
    h, w = shape
    r0, c0, r1, c1 = meta["bounds"]  # type: ignore[index]
    cr, cc = meta["center"]  # type: ignore[index]
    layout = meta["layout"]  # type: ignore[index]

    if layout in {"vertical", "cross"}:
        outer = (r0, c0 + 1 + ((r1 - r0) % max(1, cc - c0 - 1)))
        inner = (r0 + 2 + ((c1 - c0) % max(1, cr - r0 - 2)), cc)
        cells = [outer, inner]
        if layout == "cross":
            cells.append((cr, c0 + 2 + ((r1 - r0) % max(1, cc - c0 - 2))))
    elif layout == "horizontal":
        outer = (r0 + 1 + ((c1 - c0) % max(1, cr - r0 - 1)), c0)
        inner = (cr, c0 + 2 + ((r1 - r0) % max(1, cc - c0 - 2)))
        cells = [outer, inner]
    elif layout == "double_vertical":
        divider_cells = sorted(cell for cell in meta["divider_cells"] if cell[1] < cc)  # type: ignore[index]
        outer = (r0, c0 + 1 + ((h + w) % max(1, cc - c0 - 1)))
        inner = divider_cells[len(divider_cells) // 3]
        cells = [outer, inner]
    else:
        divider_cells = sorted(cell for cell in meta["divider_cells"] if cell[0] < cr)  # type: ignore[index]
        outer = (r0 + 1 + ((h + w) % max(1, cr - r0 - 1)), c0)
        inner = divider_cells[len(divider_cells) // 3]
        cells = [outer, inner]

    cleaned: list[Coord] = []
    seen: set[frozenset[Coord]] = set()
    for r, c in cells:
        r = min(max(r0 + 1 if c in (c0, c1) else r0, r), r1 - 1 if c in (c0, c1) else r1)
        c = min(max(c0 + 1 if r in (r0, r1) else c0, c), c1 - 1 if r in (r0, r1) else c1)
        if r in (r0, r1) and c in (c0, c1):
            c = c0 + 1
        if r == cr and c == cc:
            c += 1
        key = _orbit_key(shape, (r, c))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((r, c))
    return cleaned


def _validate_generated(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not set(map(int, np.unique(inp))) <= {Color.BLACK, Color.BLUE, Color.YELLOW}:
        return False

    try:
        repaired = _repair_barrier(inp)
    except AssertionError:
        return False

    if not set(map(int, np.unique(repaired))) <= {Color.BLACK, Color.BLUE}:
        return False

    orbits = _reflection_orbits(inp.shape)
    if any(len({int(repaired[r, c]) for r, c in orbit}) != 1 for orbit in orbits):
        return False

    exterior = _exterior_mask(repaired)
    closed_mask = (repaired == Color.BLACK) & ~exterior
    closed_count = _count_components(closed_mask)
    if closed_count < 2 or not np.any(exterior):
        return False

    incomplete = inp.copy()
    incomplete[incomplete == Color.YELLOW] = Color.BLACK
    incomplete_out = _classify_background(incomplete)
    repaired_out = _classify_background(repaired)
    if np.array_equal(incomplete_out, repaired_out):
        return False
    if _count_components((incomplete == Color.BLACK) & ~_exterior_mask(incomplete)) >= closed_count:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_SIZES)
    w = random.choice(GRID_SIZES)
    layout = random.choice(LAYOUTS)
    built = _wall_candidates(layout, h, w)
    if built is None:
        return None
    full_grid, meta = built
    inp = full_grid.copy()

    gap_cells = _pick_gap_cells(inp.shape, meta)
    if not gap_cells:
        return None

    for r, c in gap_cells:
        if int(inp[r, c]) != Color.BLUE:
            return None
        inp[r, c] = Color.YELLOW

    if not _validate_generated(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
