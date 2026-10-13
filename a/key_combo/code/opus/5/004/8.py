from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import find_components, get_bg, in_bounds, neighbors8
except Exception:  # pragma: no cover
    from collections import Counter, deque

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = map(int, shape)
        return 0 <= int(r) < h and 0 <= int(c) < w

    def neighbors8(r: int, c: int):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                yield int(r) + dr, int(c) + dc

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
    ) -> list[dict]:
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        seen = np.zeros(grid.shape, dtype=bool)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        if colors is None:
            valid = set(int(v) for v in np.unique(grid) if int(v) != bg)
        else:
            valid = {int(v) for v in colors if int(v) != bg}
        out = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                color = int(grid[r, c])
                if seen[r, c] or color == bg or color not in valid:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords = []
                while q:
                    cr, cc = q.popleft()
                    coords.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if not in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                            continue
                        same = int(grid[nr, nc]) == color if monochromatic else int(grid[nr, nc]) != bg
                        if same:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                arr = np.asarray(coords, dtype=int)
                rs = arr[:, 0]
                cs = arr[:, 1]
                out.append(
                    {
                        "color": color if monochromatic else None,
                        "coords": arr,
                        "bbox": (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


BG = 0
INPUT_FG = 8
GREEN = 3
RED = 2
ODD_LENGTHS = (3, 5, 7)
GRID_SIZE_RANGE = (11, 15)
CORE_PAIR_COUNT_OPTIONS = (2, 2, 3)
EXTRA_BAR_MAX = 2
GENERATE_ATTEMPTS = 500


@dataclass(frozen=True)
class Bar:
    orient: str
    r: int
    c: int
    length: int

    @property
    def half(self) -> int:
        return self.length // 2


def _bar_cells(bar: Bar) -> set[tuple[int, int]]:
    if bar.orient == "H":
        return {(bar.r, bar.c + d) for d in range(-bar.half, bar.half + 1)}
    if bar.orient == "V":
        return {(bar.r + d, bar.c) for d in range(-bar.half, bar.half + 1)}
    return {(bar.r, bar.c)}


def _rotated_bar(bar: Bar) -> Bar:
    if bar.orient == "H":
        return Bar("V", bar.r, bar.c, bar.length)
    if bar.orient == "V":
        return Bar("H", bar.r, bar.c, bar.length)
    return bar


def _rotated_cells(bar: Bar) -> set[tuple[int, int]]:
    return _bar_cells(_rotated_bar(bar))


def _extract_bars(grid: np.ndarray) -> list[Bar]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(int(v) for v in np.unique(g) if int(v) != bg)
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    bars: list[Bar] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=fg_colors):
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(coords.shape[0])
        if h == 1 and w == 1:
            bars.append(Bar("P", r0, c0, 1))
            continue
        assert h == 1 or w == 1, "each component must be a straight bar"
        length = max(h, w)
        assert area == length, "bars must be one cell thick and solid"
        assert length % 2 == 1, "bar length must be odd"
        if h == 1:
            bars.append(Bar("H", r0, c0 + length // 2, length))
        else:
            bars.append(Bar("V", r0 + length // 2, c0, length))
    assert bars, "expected at least one bar"
    return bars


def _coverage(shape: tuple[int, int], bars: list[Bar]) -> tuple[np.ndarray, dict[tuple[int, int], list[int]]]:
    counts = np.zeros(shape, dtype=int)
    owners: dict[tuple[int, int], list[int]] = {}
    for idx, bar in enumerate(bars):
        for cell in _rotated_cells(bar):
            r, c = cell
            assert in_bounds(shape, r, c), "rotated bar left the grid"
            counts[r, c] += 1
            owners.setdefault(cell, []).append(idx)
    return counts, owners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every thin bar 90 degrees around its center, then recolor single coverage green and double coverage red.

    Input structure:
    - The background is black and all foreground pixels share one cyan-like color.
    - Foreground components are disjoint one-cell-thick straight bars with odd lengths.
    - Each bar is horizontal, vertical, or a single pixel whose center is its own midpoint.

    Rule:
    - Clear the canvas to the background color.
    - Rotate every foreground bar 90 degrees about its center, preserving its midpoint and length while swapping horizontal and vertical orientation.
    - Count how many rotated bars cover each output cell.
    - Recolor cells covered once to green, cells covered exactly twice to red, and leave uncovered cells as background.

    Key insight:
    - The red cells are not marked in the input; they appear only after all bars are rotated together and their new footprints are counted.
    """
    g = np.asarray(grid, dtype=int)
    bars = _extract_bars(g)
    bg = int(get_bg(g))
    counts, _owners = _coverage(g.shape, bars)
    assert int(counts.max(initial=0)) <= 2, "puzzle inputs should not create triple overlaps"

    out = np.full(g.shape, bg, dtype=int)
    out[counts == 1] = GREEN
    out[counts == 2] = RED
    return out


def _bar_fits(shape: tuple[int, int], bar: Bar) -> bool:
    if bar.length not in ODD_LENGTHS:
        return False
    return all(in_bounds(shape, r, c) for r, c in _bar_cells(bar) | _rotated_cells(bar))


def _input_separated(shape: tuple[int, int], bars: list[Bar]) -> bool:
    occupied: dict[tuple[int, int], int] = {}
    forbidden: set[tuple[int, int]] = set()
    for idx, bar in enumerate(bars):
        cells = _bar_cells(bar)
        if not cells or not all(in_bounds(shape, r, c) for r, c in cells):
            return False
        for cell in cells:
            if cell in occupied or cell in forbidden:
                return False
        for cell in cells:
            occupied[cell] = idx
        for r, c in cells:
            forbidden.add((r, c))
            for nr, nc in neighbors8(r, c):
                if in_bounds(shape, nr, nc):
                    forbidden.add((nr, nc))
    return True


def _rotated_same_orientation_disjoint(bars: list[Bar]) -> bool:
    h_centers = [bar.c for bar in bars if bar.orient == "H"]
    v_centers = [bar.r for bar in bars if bar.orient == "V"]
    return len(h_centers) == len(set(h_centers)) and len(v_centers) == len(set(v_centers))


def _crosses(hbar: Bar, vbar: Bar) -> bool:
    assert hbar.orient == "H" and vbar.orient == "V"
    return abs(vbar.r - hbar.r) <= hbar.half and abs(hbar.c - vbar.c) <= vbar.half


def _partial_match_pair(bars: list[Bar]) -> bool:
    horizontals = [bar for bar in bars if bar.orient == "H"]
    verticals = [bar for bar in bars if bar.orient == "V"]
    for hbar in horizontals:
        for vbar in verticals:
            row_ok = abs(vbar.r - hbar.r) <= hbar.half
            col_ok = abs(hbar.c - vbar.c) <= vbar.half
            if row_ok ^ col_ok:
                return True
    return False


def _semantic_ok(shape: tuple[int, int], bars: list[Bar]) -> bool:
    if sum(bar.orient == "H" for bar in bars) < 2:
        return False
    if sum(bar.orient == "V" for bar in bars) < 2:
        return False
    if not _input_separated(shape, bars):
        return False
    if not _rotated_same_orientation_disjoint(bars):
        return False

    counts, owners = _coverage(shape, bars)
    if int(counts.max(initial=0)) > 2:
        return False

    red_cells = [tuple(map(int, cell)) for cell in np.argwhere(counts == 2)]
    green_cells = int(np.count_nonzero(counts == 1))
    if len(red_cells) < 2 or green_cells == 0:
        return False

    input_cells = set().union(*(_bar_cells(bar) for bar in bars))
    for cell in red_cells:
        if cell in input_cells:
            return False
        pair = owners[cell]
        if len(pair) != 2:
            return False
        orients = {bars[idx].orient for idx in pair}
        if orients != {"H", "V"}:
            return False

    if not _partial_match_pair(bars):
        return False

    grid = np.full(shape, BG, dtype=int)
    for bar in bars:
        for r, c in _bar_cells(bar):
            grid[r, c] = INPUT_FG
    out = main(grid)
    if not np.any(out == RED) or not np.any(out == GREEN):
        return False
    if np.array_equal(grid, out):
        return False
    return True


def _add_bar_if_ok(shape: tuple[int, int], bars: list[Bar], bar: Bar) -> bool:
    if not _bar_fits(shape, bar):
        return False
    trial = bars + [bar]
    if not _input_separated(shape, trial):
        return False
    if not _rotated_same_orientation_disjoint(trial):
        return False
    counts, _owners = _coverage(shape, trial)
    return int(counts.max(initial=0)) <= 2


def _sample_pair(shape: tuple[int, int], kind: str) -> tuple[Bar, Bar] | None:
    h, w = shape
    for _ in range(100):
        if kind == "A":
            sign = random.choice((-1, 1))
            dr = random.choice((-1, 0, 1))
            rh = random.randint(3, h - 4)
            rv = rh + dr
            if not (3 <= rv <= h - 4):
                continue
            if sign == 1:
                ch = random.randint(1, w - 7)
                cv = ch + 3
            else:
                ch = random.randint(6, w - 2)
                cv = ch - 3
            bars = (Bar("H", rh, ch, 3), Bar("V", rv, cv, 7))
        else:
            sign = random.choice((-1, 1))
            dc = random.choice((-1, 0, 1))
            rv = random.randint(1, h - 2)
            rh = rv + sign * 3
            if not (3 <= rh <= h - 4):
                continue
            ch = random.randint(3, w - 4)
            cv = ch + dc
            if not (1 <= cv <= w - 2):
                continue
            bars = (Bar("H", rh, ch, 7), Bar("V", rv, cv, 3))
        if _input_separated(shape, list(bars)) and _crosses(bars[0], bars[1]):
            return bars
    return None


def _sample_extra_bar(shape: tuple[int, int], bars: list[Bar]) -> Bar | None:
    h, w = shape
    for _ in range(120):
        orient = random.choice(("H", "V"))
        length = random.choices(ODD_LENGTHS, weights=(2, 5, 2), k=1)[0]
        half = length // 2
        r = random.randint(half, h - 1 - half)
        c = random.randint(half, w - 1 - half)
        bar = Bar(orient, r, c, length)
        if _add_bar_if_ok(shape, bars, bar):
            return bar
    return None


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIZE_RANGE),
        random.randint(*GRID_SIZE_RANGE),
    )
    bars: list[Bar] = []

    pair_kinds = [random.choice(("A", "B")) for _ in range(random.choice(CORE_PAIR_COUNT_OPTIONS))]
    for kind in pair_kinds:
        placed = False
        for _ in range(120):
            pair = _sample_pair(shape, kind)
            if pair is None:
                continue
            trial = bars + list(pair)
            if _input_separated(shape, trial) and _rotated_same_orientation_disjoint(trial):
                counts, _owners = _coverage(shape, trial)
                if int(counts.max(initial=0)) <= 2:
                    bars = trial
                    placed = True
                    break
        if not placed:
            return None

    extra_target = random.randint(0, EXTRA_BAR_MAX)
    for _ in range(extra_target):
        bar = _sample_extra_bar(shape, bars)
        if bar is not None:
            bars.append(bar)

    if not _semantic_ok(shape, bars):
        return None

    grid = np.full(shape, BG, dtype=int)
    for bar in bars:
        for r, c in _bar_cells(bar):
            grid[r, c] = INPUT_FG
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotating-bar overlap puzzle")
