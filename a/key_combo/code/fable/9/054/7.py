from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = list(range(10))
GEN_ATTEMPTS = 400


def _mode_color(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    return Counter(flat).most_common(1)[0][0]


def _divider_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    row_divs = [r for r in range(g.shape[0]) if np.all(g[r, :] == g[r, 0])]
    col_divs = [c for c in range(g.shape[1]) if np.all(g[:, c] == g[0, c])]
    assert len(row_divs) == 2 and len(col_divs) == 2, "expected a 3x3 panel grid with two divider rows and columns"

    divider_colors = {int(g[r, 0]) for r in row_divs} | {int(g[0, c]) for c in col_divs}
    assert len(divider_colors) == 1, "divider rows and columns must share one color"
    divider = divider_colors.pop()

    row_spans = [
        (0, row_divs[0]),
        (row_divs[0] + 1, row_divs[1]),
        (row_divs[1] + 1, g.shape[0]),
    ]
    col_spans = [
        (0, col_divs[0]),
        (col_divs[0] + 1, col_divs[1]),
        (col_divs[1] + 1, g.shape[1]),
    ]
    assert all(r1 > r0 for r0, r1 in row_spans), "panel rows must be non-empty"
    assert all(c1 > c0 for c0, c1 in col_spans), "panel cols must be non-empty"
    return divider, row_spans, col_spans


def _panel_bounds(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], pr: int, pc: int) -> tuple[int, int, int, int]:
    r0, r1 = row_spans[pr]
    c0, c1 = col_spans[pc]
    return r0, r1, c0, c1


def _extract_center_sprite(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], bg: int) -> dict[int, tuple[int, int]]:
    r0, r1, c0, c1 = _panel_bounds(row_spans, col_spans, 1, 1)
    center = np.asarray(grid[r0:r1, c0:c1], dtype=int)
    sprite_map: dict[int, tuple[int, int]] = {}
    for rr, cc in np.argwhere(center != bg):
        color = int(center[rr, cc])
        assert color not in sprite_map, "each sprite color must appear exactly once"
        sprite_map[color] = (int(rr), int(cc))
    assert len(sprite_map) >= 3, "center sprite must contain at least three colored cells"
    return sprite_map


def _stamp_panel(
    out: np.ndarray,
    sprite_map: dict[int, tuple[int, int]],
    panel_origin: tuple[int, int],
    panel_shape: tuple[int, int],
    anchor_local: tuple[int, int],
    anchor_color: int,
) -> None:
    pr0, pc0 = panel_origin
    ph, pw = panel_shape
    ar, ac = anchor_local
    sr, sc = sprite_map[anchor_color]
    dr = ar - sr
    dc = ac - sc

    for color, (rr, cc) in sprite_map.items():
        tr = rr + dr
        tc = cc + dc
        if 0 <= tr < ph and 0 <= tc < pw:
            out[pr0 + tr, pc0 + tc] = int(color)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    divider, row_spans, col_spans = _divider_structure(g)
    bg = _mode_color(g[g != divider])
    sprite_map = _extract_center_sprite(g, row_spans, col_spans, bg)

    out = g.copy()
    for pr in range(3):
        for pc in range(3):
            if pr == 1 and pc == 1:
                continue
            r0, r1, c0, c1 = _panel_bounds(row_spans, col_spans, pr, pc)
            panel = g[r0:r1, c0:c1]
            coords = np.argwhere(panel != bg)
            if coords.size == 0:
                continue
            assert coords.shape[0] == 1, "each outer panel must be empty or contain one anchor pixel"
            ar, ac = map(int, coords[0])
            anchor_color = int(panel[ar, ac])
            assert anchor_color in sprite_map, "anchor color must match a color used by the center sprite"
            _stamp_panel(out, sprite_map, (r0, c0), panel.shape, (ar, ac), anchor_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the center sprite into anchor-marked outer panels, aligning by the matching anchor color.

    Input structure:
    - Divider strips split the canvas into a 3x3 panel layout.
    - The center panel contains one multicolor sprite, with each sprite color occupying exactly one cell.
    - Some outer panels contain one isolated anchor pixel whose color appears in the center sprite, while the other outer panels are empty.

    Rule:
    - For each anchored outer panel, translate the whole center sprite so that the sprite cell with the same color as the anchor lands exactly on the anchor pixel.
    - Keep only the translated sprite cells that stay inside that panel; cells shifted beyond the panel border are clipped away.
    - Empty panels stay empty, and the center panel plus divider strips remain unchanged.

    Key insight:
    - The anchor pixel is not an object to extend by itself; its color selects one specific sprite cell, and that color match determines the sprite offset for that panel.
    """
    return _transform(grid)


def _grow_connected_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    cells = [(random.randrange(height), random.randrange(width))]
    used = {cells[0]}
    while len(cells) < count:
        base_r, base_c = random.choice(cells)
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = base_r + dr, base_c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in used:
                nbrs.append((nr, nc))
        if not nbrs:
            continue
        cell = random.choice(nbrs)
        cells.append(cell)
        used.add(cell)
    return cells


def _place_center_sprite(panel_h: int, panel_w: int, n_cells: int, bg: int, divider: int) -> dict[int, tuple[int, int]]:
    box_h = random.randint(2, min(5, panel_h))
    box_w = random.randint(2, min(5, panel_w))
    count = min(n_cells, box_h * box_w)
    coords = _grow_connected_cells(box_h, box_w, count)

    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1
    top = random.randint(0, panel_h - bbox_h)
    left = random.randint(0, panel_w - bbox_w)

    sprite_colors = random.sample([c for c in PALETTE if c not in {bg, divider}], count)
    sprite_map = {
        color: (top + r - min_r, left + c - min_c)
        for color, (r, c) in zip(sprite_colors, coords)
    }
    return sprite_map


def _valid_shift_ranges(sprite_map: dict[int, tuple[int, int]], panel_h: int, panel_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    rows = [r for r, _ in sprite_map.values()]
    cols = [c for _, c in sprite_map.values()]
    return (-min(rows), panel_h - 1 - max(rows)), (-min(cols), panel_w - 1 - max(cols))


def _choose_shift(
    sprite_map: dict[int, tuple[int, int]],
    panel_h: int,
    panel_w: int,
    anchor_color: int,
    must_clip: bool,
) -> tuple[int, int] | None:
    sr, sc = sprite_map[anchor_color]
    row_fit, col_fit = _valid_shift_ranges(sprite_map, panel_h, panel_w)
    row_anchor = (-sr, panel_h - 1 - sr)
    col_anchor = (-sc, panel_w - 1 - sc)
    candidates = []
    for dr in range(row_anchor[0], row_anchor[1] + 1):
        for dc in range(col_anchor[0], col_anchor[1] + 1):
            clipped = not (row_fit[0] <= dr <= row_fit[1] and col_fit[0] <= dc <= col_fit[1])
            if clipped != must_clip:
                continue
            visible = sum(0 <= rr + dr < panel_h and 0 <= cc + dc < panel_w for rr, cc in sprite_map.values())
            if visible >= 2:
                candidates.append((dr, dc))
    if not candidates:
        return None
    return random.choice(candidates)


def _render_input(
    panel_h: int,
    panel_w: int,
    bg: int,
    divider: int,
    sprite_map: dict[int, tuple[int, int]],
    anchors: list[tuple[int, int, int, tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((panel_h * 3 + 2, panel_w * 3 + 2), bg, dtype=int)
    grid[panel_h, :] = divider
    grid[panel_h * 2 + 1, :] = divider
    grid[:, panel_w] = divider
    grid[:, panel_w * 2 + 1] = divider

    center_r0 = panel_h + 1
    center_c0 = panel_w + 1
    for color, (r, c) in sprite_map.items():
        grid[center_r0 + r, center_c0 + c] = int(color)

    for pr, pc, color, (ar, ac) in anchors:
        r0, c0 = _panel_origin(panel_h, panel_w, pr, pc)
        grid[r0 + ar, c0 + ac] = int(color)
    return grid


def _expected_from_spec(
    panel_h: int,
    panel_w: int,
    bg: int,
    divider: int,
    sprite_map: dict[int, tuple[int, int]],
    anchors: list[tuple[int, int, int, tuple[int, int]]],
) -> np.ndarray:
    out = _render_input(panel_h, panel_w, bg, divider, sprite_map, anchors)
    for pr, pc, color, anchor_local in anchors:
        r0, c0 = _panel_origin(panel_h, panel_w, pr, pc)
        _stamp_panel(out, sprite_map, (r0, c0), (panel_h, panel_w), anchor_local, color)
    return out


def _panel_origin(panel_h: int, panel_w: int, pr: int, pc: int) -> tuple[int, int]:
    rows = [(0, panel_h), (panel_h + 1, panel_h * 2 + 1), (panel_h * 2 + 2, panel_h * 3 + 2)]
    cols = [(0, panel_w), (panel_w + 1, panel_w * 2 + 1), (panel_w * 2 + 2, panel_w * 3 + 2)]
    return rows[pr][0], cols[pc][0]


def _build_anchor_spec(
    sprite_map: dict[int, tuple[int, int]],
    panel_h: int,
    panel_w: int,
) -> list[tuple[int, int, int, tuple[int, int]]] | None:
    outer_panels = [(pr, pc) for pr in range(3) for pc in range(3) if (pr, pc) != (1, 1)]
    random.shuffle(outer_panels)
    n_anchors = random.randint(2, 6)
    anchored_panels = outer_panels[:n_anchors]

    sprite_colors = list(sprite_map)
    if len(sprite_colors) < 2:
        return None

    anchor_colors = [random.choice(sprite_colors) for _ in range(n_anchors)]
    if len(set(anchor_colors)) < 2:
        anchor_colors[random.randrange(n_anchors)] = random.choice([c for c in sprite_colors if c != anchor_colors[0]])

    clip_flags = [False, True] + [random.choice([False, True]) for _ in range(max(0, n_anchors - 2))]
    random.shuffle(clip_flags)

    anchors = []
    shifts = []
    clipped_seen = set()
    for (pr, pc), color, must_clip in zip(anchored_panels, anchor_colors, clip_flags):
        shift = _choose_shift(sprite_map, panel_h, panel_w, color, must_clip)
        if shift is None:
            return None
        sr, sc = sprite_map[color]
        dr, dc = shift
        anchor_local = (sr + dr, sc + dc)
        assert 0 <= anchor_local[0] < panel_h and 0 <= anchor_local[1] < panel_w
        anchors.append((pr, pc, color, anchor_local))
        shifts.append(shift)
        clipped_seen.add(must_clip)

    if len(set(shifts)) < 2 or clipped_seen != {False, True}:
        return None
    return anchors


def _check_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        divider, row_spans, col_spans = _divider_structure(inp)
    except AssertionError:
        return False
    bg = _mode_color(inp[inp != divider])
    try:
        sprite_map = _extract_center_sprite(inp, row_spans, col_spans, bg)
    except AssertionError:
        return False

    offsets = []
    clipped = set()
    anchor_patterns = set()
    empty_panels = 0
    for pr in range(3):
        for pc in range(3):
            if (pr, pc) == (1, 1):
                continue
            r0, r1, c0, c1 = _panel_bounds(row_spans, col_spans, pr, pc)
            panel_in = inp[r0:r1, c0:c1]
            panel_out = out[r0:r1, c0:c1]
            coords = np.argwhere(panel_in != bg)
            if coords.size == 0:
                empty_panels += 1
                if np.any(panel_out != bg):
                    return False
                continue
            if coords.shape[0] != 1:
                return False
            ar, ac = map(int, coords[0])
            color = int(panel_in[ar, ac])
            if color not in sprite_map:
                return False
            sr, sc = sprite_map[color]
            dr, dc = ar - sr, ac - sc
            offsets.append((dr, dc))

            expected_panel = np.full(panel_in.shape, bg, dtype=int)
            _stamp_panel(expected_panel, sprite_map, (0, 0), panel_in.shape, (ar, ac), color)
            anchor_patterns.add(tuple(expected_panel.ravel().tolist()))
            if not np.array_equal(panel_out, expected_panel):
                return False

            full_fit = True
            for rr, cc in sprite_map.values():
                tr, tc = rr + dr, cc + dc
                if not (0 <= tr < panel_in.shape[0] and 0 <= tc < panel_in.shape[1]):
                    full_fit = False
                    break
            clipped.add(not full_fit)

    return empty_panels >= 1 and len(set(offsets)) >= 2 and clipped == {False, True} and len(anchor_patterns) >= 2


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(6, 9)
    panel_w = random.randint(6, 9)
    bg, divider = random.sample(PALETTE, 2)

    n_cells = random.randint(3, min(6, panel_h * panel_w))
    sprite_map = _place_center_sprite(panel_h, panel_w, n_cells, bg, divider)

    anchors = _build_anchor_spec(sprite_map, panel_h, panel_w)
    if anchors is None:
        return None

    inp = _render_input(panel_h, panel_w, bg, divider, sprite_map, anchors)
    expected = _expected_from_spec(panel_h, panel_w, bg, divider, sprite_map, anchors)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _check_semantics(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
