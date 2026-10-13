from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np


BLACK = 0
MAROON = 9

Band = tuple[str, int, int, int, int]


def _runs_of_value(line: np.ndarray, value: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(line.shape[0])
    while i < n:
        if int(line[i]) != int(value):
            i += 1
            continue
        start = i
        while i + 1 < n and int(line[i + 1]) == int(value):
            i += 1
        runs.append((start, i))
        i += 1
    return runs


def _single_internal_gap(line: np.ndarray, color: int) -> tuple[int, int] | None:
    vals = set(map(int, line))
    if not vals <= {BLACK, int(color)}:
        return None
    gaps = _runs_of_value(line, BLACK)
    if len(gaps) != 1:
        return None
    g0, g1 = gaps[0]
    if g0 == 0 or g1 == int(line.shape[0]) - 1:
        return None
    return int(g0), int(g1)


def _find_bands(g: np.ndarray) -> list[Band]:
    h, w = map(int, g.shape)
    bands: list[Band] = []

    for r in range(h):
        colors = set(map(int, g[r])) - {BLACK, MAROON}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        gap = _single_internal_gap(g[r], color)
        if gap is not None:
            bands.append(("h", int(r), int(color), gap[0], gap[1]))

    for c in range(w):
        colors = set(map(int, g[:, c])) - {BLACK, MAROON}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        gap = _single_internal_gap(g[:, c], color)
        if gap is not None:
            bands.append(("v", int(c), int(color), gap[0], gap[1]))

    assert bands, "expected at least one full color band with one window gap"
    band_colors = [band[2] for band in bands]
    assert len(set(band_colors)) == len(band_colors), "band colors must be distinct"
    return bands


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        pts: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            pts.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc, h, w):
                if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _component_side(band: Band, comp: np.ndarray) -> int | None:
    kind, line, _color, _g0, _g1 = band
    if kind == "h":
        rows = comp[:, 0]
        if bool(np.all(rows < line)) and int(rows.max()) == line - 1:
            return -1
        if bool(np.all(rows > line)) and int(rows.min()) == line + 1:
            return 1
        return None

    cols = comp[:, 1]
    if bool(np.all(cols < line)) and int(cols.max()) == line - 1:
        return -1
    if bool(np.all(cols > line)) and int(cols.min()) == line + 1:
        return 1
    return None


def _project(out: np.ndarray, band: Band, comp: np.ndarray) -> None:
    kind, line, color, g0, g1 = band
    h, w = map(int, out.shape)
    for r_raw, c_raw in comp:
        r = int(r_raw)
        c = int(c_raw)
        if kind == "h":
            if not (g0 <= c <= g1):
                continue
            rr, cc = 2 * line - r, c
        else:
            if not (g0 <= r <= g1):
                continue
            rr, cc = r, 2 * line - c

        if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == BLACK:
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each maroon pattern through the gap in its adjacent color band.

    Input structure:
    - The background is black.
    - Several distinct non-maroon colors each form one full horizontal or vertical width-1 band.
    - Every band has exactly one internal contiguous black window gap.
    - For every band there is one 4-connected maroon pattern on exactly one side, touching the cells one step away from that band.

    Rule:
    - Preserve the original bands, black gaps, background, and maroon patterns.
    - For each band, mirror the maroon pattern across the band line to the opposite side.
    - Keep only mirrored pixels whose tangential coordinate lies inside that band's window gap.
    - Paint those visible mirrored pixels with the band color, not with maroon.

    Key insight:
    The gap is a clipping window: the maroon source shape is mirrored to the far side of the band, but only the portion visible through the window appears and it takes the band color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bands = _find_bands(g)
    components = _components4(g == MAROON)
    assert len(components) == len(bands), "expected one maroon component per band"

    out = g.copy()
    used_components: set[int] = set()
    for band in bands:
        candidates: list[tuple[int, np.ndarray]] = []
        for i, comp in enumerate(components):
            if i in used_components:
                continue
            if _component_side(band, comp) is not None:
                candidates.append((i, comp))
        assert len(candidates) == 1, "each band must have one adjacent maroon pattern"
        comp_i, comp = candidates[0]
        used_components.add(comp_i)
        _project(out, band, comp)

    return out


# --------------- generator ---------------

_GEN_MAX_ATTEMPTS = 500
_GEN_SIZE_RANGE = (15, 28)
_GEN_LINE_MARGIN = 4
_GEN_MIN_LINE_SEP = 5
_GEN_MAX_BANDS = 4
_GEN_BAND_COLORS = tuple(range(1, 9))
_GEN_GAP_LEN_RANGE = (3, 8)
_GEN_DEPTH_RANGE = (2, 3)
_GEN_PATTERN_SIZE_RANGE = (4, 9)
_GEN_LOCAL_ATTEMPTS = 80
_GEN_PATTERN_ATTEMPTS = 120


def _max_line_count(axis_len: int) -> int:
    slots = max(0, int(axis_len) - 2 * _GEN_LINE_MARGIN)
    if slots == 0:
        return 0
    return min(_GEN_MAX_BANDS, 1 + (slots - 1) // _GEN_MIN_LINE_SEP)


def _sample_lines(axis_len: int, count: int) -> list[int] | None:
    choices = list(range(_GEN_LINE_MARGIN, int(axis_len) - _GEN_LINE_MARGIN))
    for _ in range(200):
        picked = sorted(random.sample(choices, int(count)))
        if all(b - a >= _GEN_MIN_LINE_SEP for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _sample_gap(axis_len: int) -> tuple[int, int] | None:
    max_len = min(_GEN_GAP_LEN_RANGE[1], int(axis_len) - 6)
    if max_len < _GEN_GAP_LEN_RANGE[0]:
        return None
    gap_len = random.randint(_GEN_GAP_LEN_RANGE[0], max_len)
    start = random.randint(2, int(axis_len) - gap_len - 3)
    return int(start), int(start + gap_len - 1)


def _draw_band(grid: np.ndarray, band: Band) -> None:
    kind, line, color, g0, g1 = band
    if kind == "h":
        grid[line, :] = int(color)
        grid[line, g0 : g1 + 1] = BLACK
    else:
        grid[:, line] = int(color)
        grid[g0 : g1 + 1, line] = BLACK


def _local_neighbors(cell: tuple[int, int], depth: int, t0: int, t1: int) -> list[tuple[int, int]]:
    d, t = cell
    out: list[tuple[int, int]] = []
    for nd, nt in ((d - 1, t), (d + 1, t), (d, t - 1), (d, t + 1)):
        if 1 <= nd <= int(depth) and int(t0) <= nt <= int(t1):
            out.append((int(nd), int(nt)))
    return out


def _grow_local_pattern(g0: int, g1: int, axis_len: int) -> set[tuple[int, int]] | None:
    depth = random.randint(*_GEN_DEPTH_RANGE)
    left_extra = random.randint(1, 2)
    right_extra = random.randint(1, 2)
    t0 = max(0, int(g0) - left_extra)
    t1 = min(int(axis_len) - 1, int(g1) + right_extra)
    area = int(depth) * (int(t1) - int(t0) + 1)
    size = random.randint(_GEN_PATTERN_SIZE_RANGE[0], min(_GEN_PATTERN_SIZE_RANGE[1], area))

    for _ in range(_GEN_LOCAL_ATTEMPTS):
        seed = (1, random.randint(int(g0), int(g1)))
        selected: set[tuple[int, int]] = {seed}
        frontier = set(_local_neighbors(seed, depth, t0, t1))
        while len(selected) < size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            selected.add(cell)
            for nxt in _local_neighbors(cell, depth, t0, t1):
                if nxt not in selected:
                    frontier.add(nxt)

        visible = [(d, t) for d, t in selected if int(g0) <= t <= int(g1)]
        clipped = [(d, t) for d, t in selected if not (int(g0) <= t <= int(g1))]
        if len(visible) >= 2 and clipped:
            return selected
    return None


def _map_source(kind: str, line: int, side: int, d: int, t: int) -> tuple[int, int]:
    if kind == "h":
        return int(line + side * d), int(t)
    return int(t), int(line + side * d)


def _map_target(kind: str, line: int, side: int, d: int, t: int) -> tuple[int, int]:
    if kind == "h":
        return int(line - side * d), int(t)
    return int(t), int(line - side * d)


def _has_external_maroon_neighbor(
    grid: np.ndarray, source_cells: set[tuple[int, int]], r: int, c: int
) -> bool:
    h, w = map(int, grid.shape)
    for nr, nc in _neighbors4(int(r), int(c), h, w):
        if (nr, nc) not in source_cells and int(grid[nr, nc]) == MAROON:
            return True
    return False


def _try_add_pattern(
    grid: np.ndarray, band: Band, reserved_targets: np.ndarray
) -> tuple[int, list[tuple[int, int]]] | None:
    kind, line, _color, g0, g1 = band
    h, w = map(int, grid.shape)
    axis_len = w if kind == "h" else h

    for _ in range(_GEN_PATTERN_ATTEMPTS):
        side = random.choice([-1, 1])
        local = _grow_local_pattern(g0, g1, axis_len)
        if local is None:
            continue

        source_cells = {
            _map_source(kind, line, side, d, t)
            for d, t in local
        }
        target_cells = [
            _map_target(kind, line, side, d, t)
            for d, t in local
            if int(g0) <= int(t) <= int(g1)
        ]

        all_cells = list(source_cells) + target_cells
        if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
            continue
        if any(int(grid[r, c]) != BLACK or bool(reserved_targets[r, c]) for r, c in source_cells):
            continue
        if any(int(grid[r, c]) != BLACK or bool(reserved_targets[r, c]) for r, c in target_cells):
            continue
        if any(_has_external_maroon_neighbor(grid, source_cells, r, c) for r, c in source_cells):
            continue

        for r, c in source_cells:
            grid[r, c] = MAROON
        for r, c in target_cells:
            reserved_targets[r, c] = True
        return side, target_cells
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GEN_SIZE_RANGE)
    width = random.randint(*_GEN_SIZE_RANGE)
    kind = random.choice(["h", "v"])
    line_axis = height if kind == "h" else width
    tangent_axis = width if kind == "h" else height

    max_count = _max_line_count(line_axis)
    if max_count < 2:
        return None
    count = random.randint(2, max_count)
    lines = _sample_lines(line_axis, count)
    if lines is None:
        return None

    colors = random.sample(_GEN_BAND_COLORS, count)
    bands: list[Band] = []
    for line, color in zip(lines, colors):
        gap = _sample_gap(tangent_axis)
        if gap is None:
            return None
        bands.append((kind, int(line), int(color), gap[0], gap[1]))

    grid = np.full((height, width), BLACK, dtype=int)
    for band in bands:
        _draw_band(grid, band)

    reserved_targets = np.zeros_like(grid, dtype=bool)
    targets_by_color: dict[int, list[tuple[int, int]]] = {}
    for band in bands:
        added = _try_add_pattern(grid, band, reserved_targets)
        if added is None:
            return None
        targets_by_color[int(band[2])] = added[1]

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    diff = out != grid
    if int(np.count_nonzero(diff)) < count:
        return None
    for color, targets in targets_by_color.items():
        if not targets:
            return None
        if not any(bool(diff[r, c]) and int(out[r, c]) == int(color) for r, c in targets):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid window projection input")
