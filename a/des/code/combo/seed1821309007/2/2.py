from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 8
PALETTE = list(range(10))
GEN_RETRIES = 400


def _round_half_up(values: list[int]) -> int:
    return int(np.floor(float(sum(values)) / len(values) + 0.5))


def _full_columns(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    cols: list[tuple[int, int]] = []
    h, w = g.shape
    for c in range(w):
        color = int(g[0, c])
        if color == bg:
            continue
        if bool(np.all(g[:, c] == color)):
            cols.append((c, color))
    return cols


def _full_rows(g: np.ndarray, bg: int, col_positions: set[int]) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    h, w = g.shape
    free_cols = [c for c in range(w) if c not in col_positions]
    if not free_cols:
        return rows
    for r in range(h):
        vals = [int(g[r, c]) for c in free_cols]
        color = vals[0]
        if color == bg:
            continue
        if all(v == color for v in vals):
            rows.append((r, color))
    return rows


def _anchor_before(index: int, anchors: list[tuple[int, int]]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for pos, color in anchors:
        if pos < index:
            best = (pos, color)
        else:
            break
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered marker pixels by the color pair induced by the separator row
    above them and the separator column left of them, then replace each group by
    a counted cross centered at the group's rounded centroid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = int(Counter(map(int, g.reshape(-1))).most_common(1)[0][0])

    cols = _full_columns(g, bg)
    col_positions = {c for c, _color in cols}
    rows = _full_rows(g, bg, col_positions)
    row_positions = {r for r, _color in rows}

    assert cols and rows

    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    marker_cells: list[tuple[int, int]] = []
    for r in range(h):
        if r in row_positions:
            continue
        for c in range(w):
            if c in col_positions:
                continue
            value = int(g[r, c])
            if value == bg:
                continue
            row_anchor = _anchor_before(r, rows)
            col_anchor = _anchor_before(c, cols)
            assert row_anchor is not None and col_anchor is not None
            groups[(row_anchor[1], col_anchor[1])].append((r, c))
            marker_cells.append((r, c))

    out = g.copy()
    for r, c in marker_cells:
        out[r, c] = bg

    for (row_color, col_color), pts in groups.items():
        arm = len(pts)
        cr = _round_half_up([r for r, _c in pts])
        cc = _round_half_up([c for _r, c in pts])

        for dc in range(-arm, arm + 1):
            nc = cc + dc
            if 0 <= nc < w:
                out[cr, nc] = row_color
        for dr in range(-arm, arm + 1):
            nr = cr + dr
            if 0 <= nr < h:
                out[nr, cc] = col_color

    return out


def _pick_positions(size: int, count: int, low_margin: int, high_margin: int, min_gap: int) -> list[int] | None:
    choices = list(range(low_margin, size - high_margin))
    random.shuffle(choices)
    picked: list[int] = []
    for pos in sorted(choices):
        if all(abs(pos - prev) >= min_gap for prev in picked):
            picked.append(pos)
            if len(picked) == count:
                return sorted(picked)
    return None


def _band_ranges(anchors: list[int], size: int) -> list[tuple[int, int, int]]:
    bands: list[tuple[int, int, int]] = []
    for i, start in enumerate(anchors):
        end = anchors[i + 1] if i + 1 < len(anchors) else size
        if end - start >= 3:
            bands.append((i, start + 1, end - 1))
    return bands


def _random_symmetric_offsets(max_dr: int, max_dc: int) -> list[tuple[int, int]]:
    offsets = []
    for dr in range(-max_dr, max_dr + 1):
        for dc in range(-max_dc, max_dc + 1):
            if dr == 0 and dc == 0:
                continue
            if dr > 0 or (dr == 0 and dc > 0):
                offsets.append((dr, dc))
    random.shuffle(offsets)
    return offsets


def _make_group_points(
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    count: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    margin = count
    if r1 - r0 + 1 < 2 * margin + 1 or c1 - c0 + 1 < 2 * margin + 1:
        return None

    cr = random.randint(r0 + margin, r1 - margin)
    cc = random.randint(c0 + margin, c1 - margin)
    pts = [(cr, cc)] if count % 2 == 1 else []

    need_pairs = count // 2
    offsets = _random_symmetric_offsets(
        min(cr - r0, r1 - cr),
        min(cc - c0, c1 - cc),
    )
    if len(offsets) < need_pairs:
        return None

    used = set(pts)
    for dr, dc in offsets:
        a = (cr + dr, cc + dc)
        b = (cr - dr, cc - dc)
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        pts.extend([a, b])
        if len(pts) == count:
            break

    if len(pts) != count:
        return None
    return pts, (cr, cc)


def _draw_input(
    h: int,
    w: int,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
    groups: list[tuple[int, int, list[tuple[int, int]], int]],
    marker_colors: list[int],
) -> np.ndarray:
    g = np.full((h, w), BG, dtype=int)
    for r, color in zip(row_positions, row_colors):
        g[r, :] = color
    for c, color in zip(col_positions, col_colors):
        g[:, c] = color
    for _ri, _ci, pts, idx in groups:
        mcolor = marker_colors[idx % len(marker_colors)]
        for r, c in pts:
            g[r, c] = mcolor
    return g


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)

    n_rows = random.randint(2, 3)
    n_cols = random.randint(2, 3)
    row_positions = _pick_positions(h, n_rows, 2, 3, 6)
    col_positions = _pick_positions(w, n_cols, 2, 3, 6)
    if row_positions is None or col_positions is None:
        return None

    row_bands = _band_ranges(row_positions, h)
    col_bands = _band_ranges(col_positions, w)
    if not row_bands or not col_bands:
        return None

    usable_pairs = [(ri, ci, rr0, rr1, cc0, cc1) for ri, rr0, rr1 in row_bands for ci, cc0, cc1 in col_bands]
    random.shuffle(usable_pairs)
    max_groups = min(len(usable_pairs), 4)
    n_groups = random.randint(2, max_groups)
    chosen_pairs = usable_pairs[:n_groups]

    needed_colors = n_rows + n_cols
    if needed_colors > 8:
        return None
    anchor_colors = random.sample([c for c in PALETTE if c != BG], needed_colors)
    row_colors = anchor_colors[:n_rows]
    col_colors = anchor_colors[n_rows:]
    remaining = [c for c in PALETTE if c not in anchor_colors and c != BG]
    marker_colors = remaining if remaining else [row_colors[0]]

    groups: list[tuple[int, int, list[tuple[int, int]], int]] = []
    expected_centers: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()

    for idx, (ri, ci, rr0, rr1, cc0, cc1) in enumerate(chosen_pairs):
        count_choices = [n for n in (1, 2, 3) if rr1 - rr0 + 1 >= 2 * n + 1 and cc1 - cc0 + 1 >= 2 * n + 1]
        if not count_choices:
            return None
        count = random.choice(count_choices)
        built = _make_group_points(rr0, rr1, cc0, cc1, count)
        if built is None:
            return None
        pts, center = built
        if any(p in occupied for p in pts):
            return None
        occupied.update(pts)
        groups.append((ri, ci, pts, idx))
        expected_centers.append((center[0], center[1], count))

    inp = _draw_input(h, w, row_positions, row_colors, col_positions, col_colors, groups, marker_colors)

    try:
        out = main(inp)
    except Exception:
        return None

    # Validate that each group's cross is present and stays inside its rectangle.
    for (ri, ci, pts, idx), (cr, cc, count) in zip(groups, expected_centers):
        row_color = row_colors[ri]
        col_color = col_colors[ci]
        for dc in range(-count, count + 1):
            if int(out[cr, cc + dc]) != row_color and dc != 0:
                return None
        for dr in range(-count, count + 1):
            if int(out[cr + dr, cc]) != col_color:
                return None
        if int(out[cr, cc]) != col_color:
            return None

    # Require visible diversity: at least two different color-pair groups and at least one multi-pixel group often.
    if len({(row_colors[ri], col_colors[ci]) for ri, ci, _pts, _idx in groups}) < 2:
        return None
    if not any(len(pts) > 1 for _ri, _ci, pts, _idx in groups) and random.random() < 0.7:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
