from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 4
BAR_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    bars = []
    for comp in _components((grid != bg) & (grid != MARKER)):
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        color = int(grid[comp[0]])
        assert all(int(grid[r, c]) == color for r, c in comp), "bars must be monochrome"
        if len(rows) == 1:
            row = rows[0]
            assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal bar must be contiguous"
            bars.append(
                {
                    "orientation": "h",
                    "color": color,
                    "row": int(row),
                    "c0": int(cols[0]),
                    "c1": int(cols[-1]),
                }
            )
        elif len(cols) == 1:
            col = cols[0]
            assert rows == list(range(rows[0], rows[-1] + 1)), "vertical bar must be contiguous"
            bars.append(
                {
                    "orientation": "v",
                    "color": color,
                    "col": int(col),
                    "r0": int(rows[0]),
                    "r1": int(rows[-1]),
                }
            )
        else:
            raise AssertionError("every foreground component must be a straight bar")
    assert bars, "expected at least one bar"
    return bars


def _segment_for_marker(marker: tuple[int, int], bars: list[dict]) -> dict:
    mr, mc = marker
    candidates = []
    for bar in bars:
        if bar["orientation"] == "h":
            if bar["c0"] <= mc <= bar["c1"] and mr != bar["row"]:
                radius = min(mc - bar["c0"], bar["c1"] - mc)
                candidates.append(
                    {
                        "bar": bar,
                        "radius": int(radius),
                        "center": (int(bar["row"]), int(mc)),
                    }
                )
        else:
            if bar["r0"] <= mr <= bar["r1"] and mc != bar["col"]:
                radius = min(mr - bar["r0"], bar["r1"] - mr)
                candidates.append(
                    {
                        "bar": bar,
                        "radius": int(radius),
                        "center": (int(mr), int(bar["col"])),
                    }
                )
    assert len(candidates) == 1, "each marker must project onto exactly one bar"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a centered piece out of each marker-addressed bar and replant that piece after a 90-degree turn at the marker.

    Input structure:
    - The background is black.
    - The foreground contains several one-cell-thick monochrome horizontal or vertical bars.
    - Yellow singleton markers sit off the bars; each marker's row/column crosshair intersects exactly one bar and therefore fixes a projection point on that bar.

    Rule:
    - For each yellow marker, find the unique bar whose span contains the marker's orthogonal coordinate and project the marker onto that bar.
    - The projected point determines a centered source segment: its half-length is the distance from that point to the nearer bar endpoint.
    - Erase every such source segment from the original bar, erase the markers, then redraw each removed segment after rotating it 90 degrees so that the marker becomes the new center.
    - Bars with no associated marker stay unchanged.

    Key insight:
    - The marker does not encode a fixed cut size; its projected position inside the bar determines how much of the bar can be symmetrically removed before the nearer endpoint is reached.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    bars = _find_bars(g, bg)
    markers = [(int(r), int(c)) for r, c in np.argwhere(g == MARKER)]
    assert markers, "expected at least one yellow marker"

    clears: set[tuple[int, int]] = set(markers)
    paints: list[tuple[int, int, int]] = []

    for marker in markers:
        info = _segment_for_marker(marker, bars)
        bar = info["bar"]
        radius = int(info["radius"])
        mr, mc = marker
        if bar["orientation"] == "h":
            br = int(bar["row"])
            for dc in range(-radius, radius + 1):
                clears.add((br, int(mc + dc)))
                tr = int(mr + dc)
                assert 0 <= tr < h, "rotated segment must fit"
                paints.append((tr, int(mc), int(bar["color"])))
        else:
            bc = int(bar["col"])
            for dr in range(-radius, radius + 1):
                clears.add((int(mr + dr), bc))
                tc = int(mc + dr)
                assert 0 <= tc < w, "rotated segment must fit"
                paints.append((int(mr), tc, int(bar["color"])))

    out = g.copy()
    for r, c in clears:
        out[r, c] = bg
    for r, c, color in paints:
        out[r, c] = color
    return out


def _intervals(lo: int, hi: int, count: int, min_len: int, max_len: int, gap: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        picked: list[tuple[int, int]] = []
        ok = True
        for _ in range(count):
            choices = []
            for length in range(min_len, max_len + 1):
                for start in range(lo, hi - length + 2):
                    end = start + length - 1
                    if all(end + gap < s or e + gap < start for s, e in picked):
                        choices.append((start, end))
            if not choices:
                ok = False
                break
            picked.append(random.choice(choices))
        if ok:
            return sorted(picked)
    return None


def _render_layout(shape: tuple[int, int], bars: list[dict], markers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for bar in bars:
        if bar["orientation"] == "h":
            grid[int(bar["row"]), int(bar["c0"]) : int(bar["c1"]) + 1] = int(bar["color"])
        else:
            grid[int(bar["r0"]) : int(bar["r1"]) + 1, int(bar["col"])] = int(bar["color"])
    for r, c in markers:
        assert grid[r, c] == BG
        grid[r, c] = MARKER
    return grid


def _choose_horizontal_group(
    h: int,
    w: int,
    row_lo: int,
    row_hi: int,
    active_count: int,
    passive_count: int,
    colors: list[int],
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    total = active_count + passive_count
    if total == 0:
        return [], []
    region_height = row_hi - row_lo + 1
    if region_height < 5:
        return None
    bar_rows = list(range(row_lo, row_hi + 1))
    if len(bar_rows) < total:
        return None
    intervals = _intervals(1, w - 2, total, 5, min(12, w - 2), 2)
    if intervals is None:
        return None
    random.shuffle(bar_rows)
    active_flags = [True] * active_count + [False] * passive_count
    random.shuffle(active_flags)
    bars: list[dict] = []
    markers: list[tuple[int, int]] = []
    used_marker_rows: set[int] = set()

    for (c0, c1), row, active, color in zip(intervals, bar_rows, active_flags, colors):
        bar = {"orientation": "h", "row": int(row), "c0": int(c0), "c1": int(c1), "color": int(color)}
        if not active:
            bars.append(bar)
            continue

        max_radius = min(4, (c1 - c0) // 2)
        radii = list(range(1, max_radius + 1))
        random.shuffle(radii)
        placed = False
        for radius in radii:
            proj_cols = [c0 + radius, c1 - radius]
            random.shuffle(proj_cols)
            for proj_col in proj_cols:
                marker_rows = [
                    mr
                    for mr in range(row_lo, row_hi + 1)
                    if mr != row
                    and mr not in used_marker_rows
                    and mr - radius >= row_lo
                    and mr + radius <= row_hi
                    and abs(mr - row) >= radius + 2
                ]
                if not marker_rows:
                    continue
                marker_row = random.choice(marker_rows)
                bars.append(bar)
                markers.append((int(marker_row), int(proj_col)))
                used_marker_rows.add(int(marker_row))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return bars, markers


def _choose_vertical_group(
    h: int,
    w: int,
    row_lo: int,
    row_hi: int,
    active_count: int,
    passive_count: int,
    colors: list[int],
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    total = active_count + passive_count
    if total == 0:
        return [], []
    region_height = row_hi - row_lo + 1
    if region_height < 5:
        return None
    if w < total + 2:
        return None
    intervals = _intervals(row_lo, row_hi, total, 5, min(12, region_height), 1)
    if intervals is None:
        return None
    cols = list(range(1, w - 1))
    if len(cols) < total:
        return None
    random.shuffle(cols)
    active_flags = [True] * active_count + [False] * passive_count
    random.shuffle(active_flags)
    bars: list[dict] = []
    markers: list[tuple[int, int]] = []
    used_marker_cols: set[int] = set()

    for (r0, r1), col, active, color in zip(intervals, cols, active_flags, colors):
        bar = {"orientation": "v", "col": int(col), "r0": int(r0), "r1": int(r1), "color": int(color)}
        if not active:
            bars.append(bar)
            continue

        max_radius = min(4, (r1 - r0) // 2)
        radii = list(range(1, max_radius + 1))
        random.shuffle(radii)
        placed = False
        for radius in radii:
            proj_rows = [r0 + radius, r1 - radius]
            random.shuffle(proj_rows)
            for proj_row in proj_rows:
                marker_cols = [
                    mc
                    for mc in range(0, w)
                    if mc != col
                    and mc not in used_marker_cols
                    and mc - radius >= 0
                    and mc + radius < w
                    and abs(mc - col) >= radius + 2
                ]
                if not marker_cols:
                    continue
                marker_col = random.choice(marker_cols)
                bars.append(bar)
                markers.append((int(proj_row), int(marker_col)))
                used_marker_cols.add(int(marker_col))
                placed = True
                break
            if placed:
                break
        if placed:
            continue
        return None
    return bars, markers


def _sample_layout() -> np.ndarray | None:
    h = random.randint(14, 28)
    w = random.randint(14, 28)
    mode = random.choice(("h_only", "v_only", "mixed"))

    if mode == "mixed":
        split = random.randint(6, h - 7)
        h_rows = (1, split - 1)
        v_rows = (split + 1, h - 2)
        h_active = random.randint(1, 2)
        v_active = random.randint(1, 2)
        h_passive = random.randint(0, 1)
        v_passive = random.randint(0, 1)
        total = h_active + v_active + h_passive + v_passive
    elif mode == "h_only":
        h_rows = (1, h - 2)
        v_rows = None
        h_active = random.randint(1, 3)
        v_active = 0
        h_passive = random.randint(0, 2)
        v_passive = 0
        total = h_active + h_passive
    else:
        h_rows = None
        v_rows = (1, h - 2)
        h_active = 0
        v_active = random.randint(1, 3)
        h_passive = 0
        v_passive = random.randint(0, 2)
        total = v_active + v_passive

    if total > len(BAR_COLORS):
        return None
    colors = random.sample(BAR_COLORS, total)
    idx = 0

    all_bars: list[dict] = []
    all_markers: list[tuple[int, int]] = []

    if h_rows is not None:
        h_total = h_active + h_passive
        placed = _choose_horizontal_group(h, w, h_rows[0], h_rows[1], h_active, h_passive, colors[idx : idx + h_total])
        if placed is None:
            return None
        bars, markers = placed
        all_bars.extend(bars)
        all_markers.extend(markers)
        idx += h_total

    if v_rows is not None:
        v_total = v_active + v_passive
        placed = _choose_vertical_group(h, w, v_rows[0], v_rows[1], v_active, v_passive, colors[idx : idx + v_total])
        if placed is None:
            return None
        bars, markers = placed
        all_bars.extend(bars)
        all_markers.extend(markers)

    if not all_markers:
        return None

    grid = _render_layout((h, w), all_bars, all_markers)
    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
