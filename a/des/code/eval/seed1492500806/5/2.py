from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


_BG = 0
_WALL = 5
_LIQUID_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

_SYSTEM_COUNT_OPTIONS = (2, 3)
_BAR_WIDTH_RANGE = (1, 3)
_BAND_HEIGHT_RANGE_2 = (6, 9)
_BAND_HEIGHT_RANGE_3 = (6, 7)
_BAND_GAP_RANGE = (2, 3)
_VERTICAL_MARGIN_RANGE = (1, 2)
_CHAMBER_COUNT_RANGE = (1, 4)
_CHAMBER_WIDTH_RANGE = (2, 4)
_OPPOSITE_MARGIN_RANGE = (2, 5)
_GRID_EXTRA_WIDTH_RANGE = (0, 3)
_GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _border_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = np.concatenate([grid[0, :], grid[h - 1, :], grid[1:-1, 0], grid[1:-1, w - 1]])
    counts = Counter(int(v) for v in border.tolist())
    return int(counts.most_common(1)[0][0])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel().tolist() if int(v) != bg)
    assert counts, "expected a wall color and at least one colored marker"
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _read_markers(grid: np.ndarray, bg: int, wall: int) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    rulers: dict[int, int] = {}
    inlets: dict[int, tuple[int, int]] = {}

    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, wall)):
        for comp in _components(grid == color):
            touches_edge = any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp)
            if touches_edge:
                rows = {r for r, _c in comp}
                assert len(rows) == 1, "ruler mark must be a 1-row segment"
                row = int(next(iter(rows)))
                assert color not in rulers, "each liquid color must have exactly one ruler mark"
                rulers[color] = row
            else:
                assert len(comp) == 1, "inlet marker must be a singleton"
                r, c = comp[0]
                assert color not in inlets, "each liquid color must have exactly one inlet"
                inlets[color] = (int(r), int(c))

    assert rulers and inlets, "expected at least one ruler and one inlet"
    assert set(rulers) == set(inlets), "ruler colors and inlet colors must match one-to-one"
    return rulers, inlets


def _reachable_background(grid: np.ndarray, bg: int, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    sr, sc = start
    reach = np.zeros((h, w), dtype=bool)
    q = deque([(int(sr), int(sc))])
    reach[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and int(grid[nr, nc]) == bg:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-bounded communicating vessel from its colored inlet up to the matching edge ruler.

    The solver:
    1. Reads colored edge segments as water-level rulers.
    2. Reads colored singleton markers on wall tops as inlets.
    3. Floods the background region reachable below each inlet.
    4. Colors the reachable cells at or below the ruler row, keeping walls and markers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _border_color(g)
    wall = _wall_color(g, bg)
    rulers, inlets = _read_markers(g, bg, wall)

    out = g.copy()
    filled_any = np.zeros((h, w), dtype=bool)
    row_ids = np.arange(h, dtype=int)[:, None]

    for color, (r, c) in sorted(inlets.items()):
        assert r + 1 < h and int(g[r + 1, c]) == bg, "every inlet must pour directly into background"
        reach = _reachable_background(g, bg, (r + 1, c))
        fill = reach & (row_ids >= int(rulers[color]))
        assert np.any(fill), "each liquid must fill at least one cell"
        assert not np.any(filled_any & fill), "different liquids must not target the same cavity cells"
        out[fill] = int(color)
        filled_any |= fill

    return out


@dataclass
class _SystemSpec:
    color: int
    bar_width: int
    height: int
    chamber_widths: list[int]
    divider_open: list[bool]
    gap_rows_local: list[int]
    inlet_chamber: int
    waterline_local: int
    top: int = 0

    @property
    def chamber_count(self) -> int:
        return len(self.chamber_widths)

    @property
    def total_width(self) -> int:
        return 2 + sum(self.chamber_widths) + max(0, self.chamber_count - 1)


def _rand_range(bounds: tuple[int, int]) -> int:
    return random.randint(int(bounds[0]), int(bounds[1]))


def _sample_dividers(chamber_count: int, mode: str) -> list[bool]:
    if chamber_count == 1:
        return []

    while True:
        flags = [random.random() < 0.65 for _ in range(chamber_count - 1)]
        if mode == "communicating" and not any(flags):
            continue
        if mode == "mixed" and (not any(flags) or all(flags)):
            continue
        return flags


def _reachable_chambers(inlet: int, divider_open: list[bool], chamber_count: int) -> set[int]:
    vis = {int(inlet)}
    q = deque([int(inlet)])
    while q:
        idx = q.popleft()
        if idx > 0 and divider_open[idx - 1] and idx - 1 not in vis:
            vis.add(idx - 1)
            q.append(idx - 1)
        if idx + 1 < chamber_count and divider_open[idx] and idx + 1 not in vis:
            vis.add(idx + 1)
            q.append(idx + 1)
    return vis


def _sample_system(color: int, mode: str, n_systems: int) -> _SystemSpec:
    if mode == "mixed":
        chamber_count = random.randint(3, _CHAMBER_COUNT_RANGE[1])
    elif mode == "communicating":
        chamber_count = random.randint(2, _CHAMBER_COUNT_RANGE[1])
    else:
        chamber_count = random.randint(*_CHAMBER_COUNT_RANGE)

    bar_width = _rand_range(_BAR_WIDTH_RANGE)
    height_bounds = _BAND_HEIGHT_RANGE_2 if n_systems == 2 else _BAND_HEIGHT_RANGE_3
    height = _rand_range(height_bounds)
    chamber_widths = [_rand_range(_CHAMBER_WIDTH_RANGE) for _ in range(chamber_count)]
    divider_open = _sample_dividers(chamber_count, mode)
    inlet_chamber = random.randrange(chamber_count)

    gap_rows_local: list[int] = []
    for is_open in divider_open:
        if is_open:
            gap_rows_local.append(random.randint(2, height - 3))
        else:
            gap_rows_local.append(-1)

    reachable = _reachable_chambers(inlet_chamber, divider_open, chamber_count)
    used_gaps = [
        gap_rows_local[j]
        for j in range(chamber_count - 1)
        if divider_open[j] and j in reachable and (j + 1) in reachable
    ]
    level_high = min([height - 2] + used_gaps)
    waterline_local = random.randint(1, level_high)

    return _SystemSpec(
        color=int(color),
        bar_width=int(bar_width),
        height=int(height),
        chamber_widths=[int(v) for v in chamber_widths],
        divider_open=[bool(v) for v in divider_open],
        gap_rows_local=[int(v) for v in gap_rows_local],
        inlet_chamber=int(inlet_chamber),
        waterline_local=int(waterline_local),
    )


def _assign_tops(specs: list[_SystemSpec]) -> int:
    top_margin = _rand_range(_VERTICAL_MARGIN_RANGE)
    bottom_margin = _rand_range(_VERTICAL_MARGIN_RANGE)
    gaps = [_rand_range(_BAND_GAP_RANGE) for _ in range(max(0, len(specs) - 1))]

    cur = top_margin
    for i, spec in enumerate(specs):
        spec.top = int(cur)
        cur += spec.height
        if i < len(gaps):
            cur += gaps[i]
    return int(cur + bottom_margin)


def _grid_width(specs: list[_SystemSpec]) -> int:
    need = max(spec.bar_width + spec.total_width for spec in specs)
    return int(need + _rand_range(_OPPOSITE_MARGIN_RANGE) + _rand_range(_GRID_EXTRA_WIDTH_RANGE))


def _chamber_spans(left: int, spec: _SystemSpec) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    dividers: list[int] = []
    cur = int(left) + 1
    for idx, width in enumerate(spec.chamber_widths):
        spans.append((int(cur), int(cur + width - 1)))
        cur += width
        if idx + 1 < spec.chamber_count:
            dividers.append(int(cur))
            cur += 1
    return spans, dividers


def _render_system(grid: np.ndarray, spec: _SystemSpec, side: str) -> None:
    h, w = grid.shape
    top = int(spec.top)
    bottom = top + spec.height - 1

    if side == "left":
        left = spec.bar_width
        right = left + spec.total_width - 1
        bar_slice = (slice(spec.top + spec.waterline_local, spec.top + spec.waterline_local + 1), slice(0, left))
    else:
        right = w - 1 - spec.bar_width
        left = right - spec.total_width + 1
        bar_slice = (
            slice(spec.top + spec.waterline_local, spec.top + spec.waterline_local + 1),
            slice(right + 1, w),
        )

    assert 0 <= left < right < w
    assert 0 <= top < bottom < h

    grid[top, left : right + 1] = _WALL
    grid[bottom, left : right + 1] = _WALL
    grid[top : bottom + 1, left] = _WALL
    grid[top : bottom + 1, right] = _WALL

    spans, divider_cols = _chamber_spans(left, spec)
    for idx, col in enumerate(divider_cols):
        grid[top : bottom + 1, col] = _WALL
        if spec.divider_open[idx]:
            grid[top + spec.gap_rows_local[idx], col] = _BG

    inlet_left, inlet_right = spans[spec.inlet_chamber]
    inlet_col = random.randint(inlet_left, inlet_right)
    grid[top, inlet_col] = spec.color
    grid[bar_slice] = spec.color


def _make_attempt() -> np.ndarray | None:
    n_systems = random.choice(_SYSTEM_COUNT_OPTIONS)
    colors = list(_LIQUID_COLORS)
    random.shuffle(colors)
    colors = colors[:n_systems]

    feature_idx = random.randrange(n_systems)
    partial_idx = None
    if n_systems >= 2 and random.random() < 0.5:
        choices = [idx for idx in range(n_systems) if idx != feature_idx]
        if choices:
            partial_idx = random.choice(choices)

    specs: list[_SystemSpec] = []
    for idx, color in enumerate(colors):
        mode = "free"
        if idx == feature_idx:
            mode = "communicating"
        if partial_idx is not None and idx == partial_idx:
            mode = "mixed"
        specs.append(_sample_system(color, mode, n_systems))

    side = random.choice(("left", "right"))
    height = _assign_tops(specs)
    width = _grid_width(specs)
    grid = np.full((height, width), _BG, dtype=int)

    for spec in specs:
        _render_system(grid, spec, side)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    for spec in specs:
        color = int(spec.color)
        if int(np.count_nonzero(grid == color)) < 2:
            return None
        if int(np.count_nonzero(out == color)) <= int(np.count_nonzero(grid == color)):
            return None

    unchanged_non_bg = grid != _BG
    if not np.array_equal(out[unchanged_non_bg], grid[unchanged_non_bg]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid communicating-vessels puzzle")
