import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
MAROON = 9

RING_COLORS = (1, 2, 3, 4, 6, 8)
GAP_SIDES = ("top", "right", "bottom", "left")

MIN_RINGS = 2
MAX_RINGS = 4
RING_H_RANGE = (4, 8)
RING_W_RANGE = (4, 9)
PANEL_MARGIN_RANGE = (1, 3)
ROW_GAP_RANGE = (1, 3)
ITEM_GAP_RANGE = (1, 3)
EXTRA_PANEL_SLACK = 3
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(1, grid.shape[0] - 1) if bool(np.all(grid[r] == GREY))]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _boundary_cycle(top: int, left: int, bottom: int, right: int) -> list[tuple[int, int]]:
    if top > bottom or left > right:
        return []
    if top == bottom:
        return [(int(top), int(c)) for c in range(left, right + 1)]
    if left == right:
        return [(int(r), int(left)) for r in range(top, bottom + 1)]
    cycle: list[tuple[int, int]] = []
    cycle.extend((int(top), int(c)) for c in range(left, right + 1))
    cycle.extend((int(r), int(right)) for r in range(top + 1, bottom + 1))
    cycle.extend((int(bottom), int(c)) for c in range(right - 1, left - 1, -1))
    cycle.extend((int(r), int(left)) for r in range(bottom - 1, top, -1))
    return cycle


def _spiral_order(height: int, width: int, side: str, offset: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    layer = 0
    while True:
        top = int(layer)
        left = int(layer)
        bottom = int(height - 1 - layer)
        right = int(width - 1 - layer)
        if top > bottom or left > right:
            break

        cycle = _boundary_cycle(top, left, bottom, right)
        if side == "top":
            start = (top, min(max(int(offset), left), right))
        elif side == "bottom":
            start = (bottom, min(max(int(offset), left), right))
        elif side == "left":
            start = (min(max(int(offset), top), bottom), left)
        elif side == "right":
            start = (min(max(int(offset), top), bottom), right)
        else:
            raise AssertionError(f"unknown gap side: {side}")

        idx = cycle.index(start)
        order.extend(cycle[idx:] + cycle[:idx])
        layer += 1

    assert len(order) == height * width
    return order


def _extract_rings(panel: np.ndarray) -> dict[int, dict]:
    rings: dict[int, dict] = {}
    for comp in _components(panel != BLACK):
        colors = {int(panel[r, c]) for r, c in comp}
        assert len(colors) == 1, "each top-panel object must be monochrome"
        color = colors.pop()

        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 3 and c1 - c0 >= 3, "rings must have a nontrivial interior"

        boundary = {
            (int(r), int(c))
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        comp_set = {(int(r), int(c)) for r, c in comp}
        missing = sorted(boundary - comp_set)
        assert len(missing) == 1, "ring boundary must contain exactly one gap"
        gap_r, gap_c = missing[0]
        assert not (gap_r in (r0, r1) and gap_c in (c0, c1)), "gap must lie on a side, not a corner"
        assert comp_set == boundary - {missing[0]}, "ring pixels must be the rectangle boundary minus the gap"
        assert bool(np.all(panel[r0 + 1 : r1, c0 + 1 : c1] == BLACK)), "ring interior must start as black"

        if gap_r == r0:
            side = "top"
            offset = int(gap_c - (c0 + 1))
            span = int(c1 - c0 - 1)
        elif gap_r == r1:
            side = "bottom"
            offset = int(gap_c - (c0 + 1))
            span = int(c1 - c0 - 1)
        elif gap_c == c0:
            side = "left"
            offset = int(gap_r - (r0 + 1))
            span = int(r1 - r0 - 1)
        else:
            side = "right"
            offset = int(gap_r - (r0 + 1))
            span = int(r1 - r0 - 1)

        assert 0 <= offset < span
        assert color not in rings, "ring colors must be unique"
        rings[color] = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "side": side,
            "offset": int(offset),
        }

    assert rings, "expected at least one ring"
    return rings


def _extract_diagrams(panel: np.ndarray) -> dict[int, str]:
    diagrams: dict[int, str] = {}
    for comp in _components(panel != BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each bottom-panel diagram must occupy a 3x3 box"
        sub = panel[r0 : r1 + 1, c0 : c1 + 1]
        border_vals = {
            int(sub[r, c])
            for r in range(3)
            for c in range(3)
            if r in (0, 2) or c in (0, 2)
        }
        assert len(border_vals) == 1, "diagram border must have exactly one color"
        color = border_vals.pop()
        assert color not in (BLACK, GREY, ORANGE, MAROON)
        center = int(sub[1, 1])
        assert center in (BLACK, MAROON), "diagram center must be black or maroon"
        assert color not in diagrams, "diagram colors must be unique"
        diagrams[color] = "spiral" if center == MAROON else "rows"

    assert diagrams, "expected at least one diagram"
    return diagrams


def _fill_rows(out: np.ndarray, ring: dict) -> None:
    r0, c0, r1, c1 = ring["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = ORANGE


def _fill_spiral(out: np.ndarray, ring: dict) -> None:
    r0, c0, r1, c1 = ring["bbox"]
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    for rr, cc in _spiral_order(inner_h, inner_w, ring["side"], ring["offset"]):
        out[r0 + 1 + rr, c0 + 1 + cc] = ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the interior of each top-panel ring according to its color-matched 3x3 diagram.

    The input is split by a full-width grey row. Top-panel objects are hollow rectangular
    rings with exactly one side gap. Bottom-panel objects are 3x3 diagrams whose border
    color identifies the matching ring; a maroon center selects spiral fill, and a black
    center selects row-wise fill. Both modes paint the ring interior orange while leaving
    the colored border and its gap unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_row(g)
    rings = _extract_rings(g[:sep])
    diagrams = _extract_diagrams(g[sep + 1 :])
    assert set(rings) == set(diagrams), "ring and diagram colors must match one-to-one"

    out = g.copy()
    for color, ring in rings.items():
        if diagrams[color] == "spiral":
            _fill_spiral(out, ring)
        else:
            _fill_rows(out, ring)
    return out


def _draw_ring(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int, side: str, gap_idx: int) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color

    if side == "top":
        grid[top, left + gap_idx] = BLACK
    elif side == "bottom":
        grid[top + height - 1, left + gap_idx] = BLACK
    elif side == "left":
        grid[top + gap_idx, left] = BLACK
    elif side == "right":
        grid[top + gap_idx, left + width - 1] = BLACK
    else:
        raise AssertionError(f"unknown gap side: {side}")


def _draw_diagram(grid: np.ndarray, top: int, left: int, color: int, mode: str) -> None:
    grid[top : top + 3, left : left + 3] = color
    grid[top + 1, left + 1] = MAROON if mode == "spiral" else BLACK


def _pack_rectangles(sizes: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    count = len(sizes)
    if count == 0:
        return None

    row_count = 1 if count == 1 else random.randint(1, min(2, count))
    order = list(range(count))
    random.shuffle(order)
    if row_count == 1:
        rows = [order]
    else:
        split = random.randint(1, count - 1)
        rows = [order[:split], order[split:]]

    row_heights = [max(sizes[idx][0] for idx in row) for row in rows]
    top_margin = random.randint(*PANEL_MARGIN_RANGE)
    bottom_margin = random.randint(*PANEL_MARGIN_RANGE)
    left_margin = random.randint(*PANEL_MARGIN_RANGE)
    right_margin = random.randint(*PANEL_MARGIN_RANGE)
    vertical_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(max(0, row_count - 1))]

    row_widths: list[int] = []
    row_gaps: list[list[int]] = []
    for row in rows:
        gaps = [random.randint(*ITEM_GAP_RANGE) for _ in range(max(0, len(row) - 1))]
        width = sum(sizes[idx][1] for idx in row) + sum(gaps)
        row_widths.append(int(width))
        row_gaps.append(gaps)

    content_w = max(row_widths)
    panel_w = left_margin + content_w + right_margin + random.randint(0, EXTRA_PANEL_SLACK)
    panel_h = top_margin + sum(row_heights) + sum(vertical_gaps) + bottom_margin
    if panel_w > MAX_GRID_SIZE or panel_h > MAX_GRID_SIZE:
        return None

    positions: list[tuple[int, int] | None] = [None] * count
    row_top = top_margin
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_w = row_widths[row_idx]
        slack = panel_w - left_margin - right_margin - row_w
        if slack < 0:
            return None
        row_left = left_margin + (random.randint(0, slack) if slack else 0)
        x = row_left
        for item_idx, spec_idx in enumerate(row):
            item_h, item_w = sizes[spec_idx]
            dy = random.randint(0, row_h - item_h)
            positions[spec_idx] = (int(row_top + dy), int(x))
            x += item_w
            if item_idx < len(row) - 1:
                x += row_gaps[row_idx][item_idx]
        if row_idx < row_count - 1:
            row_top += row_h + vertical_gaps[row_idx]

    if any(pos is None for pos in positions):
        return None
    return int(panel_h), int(panel_w), [(int(r), int(c)) for r, c in positions]


def _sample_modes(count: int) -> list[str]:
    n_spiral = random.randint(1, count - 1)
    modes = ["spiral"] * n_spiral + ["rows"] * (count - n_spiral)
    random.shuffle(modes)
    return modes


def _build_expected(inp: np.ndarray, ring_specs: list[dict], ring_positions: list[tuple[int, int]], top_shift: int) -> np.ndarray:
    out = inp.copy()
    for spec, (top, left) in zip(ring_specs, ring_positions):
        r0 = int(top)
        c0 = int(top_shift + left)
        r1 = int(r0 + spec["height"] - 1)
        c1 = int(c0 + spec["width"] - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = ORANGE
    return out


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.randint(MIN_RINGS, MAX_RINGS)
    colors = random.sample(RING_COLORS, ring_count)
    modes = _sample_modes(ring_count)

    ring_specs: list[dict] = []
    for color, mode in zip(colors, modes):
        height = random.randint(*RING_H_RANGE)
        width = random.randint(*RING_W_RANGE)
        side = random.choice(GAP_SIDES)
        if side in ("top", "bottom"):
            gap_idx = random.randint(1, width - 2)
        else:
            gap_idx = random.randint(1, height - 2)
        ring_specs.append(
            {
                "color": int(color),
                "mode": mode,
                "height": int(height),
                "width": int(width),
                "side": side,
                "gap_idx": int(gap_idx),
            }
        )

    top_layout = _pack_rectangles([(spec["height"], spec["width"]) for spec in ring_specs])
    if top_layout is None:
        return None
    top_h, top_w, ring_positions = top_layout

    diagrams = ring_specs[:]
    random.shuffle(diagrams)
    bottom_layout = _pack_rectangles([(3, 3) for _ in diagrams])
    if bottom_layout is None:
        return None
    bottom_h, bottom_w, diagram_positions = bottom_layout

    grid_w = max(top_w, bottom_w) + random.randint(0, EXTRA_PANEL_SLACK)
    grid_h = top_h + 1 + bottom_h
    if grid_w > MAX_GRID_SIZE or grid_h > MAX_GRID_SIZE:
        return None

    top_shift = random.randint(0, grid_w - top_w)
    bottom_shift = random.randint(0, grid_w - bottom_w)
    inp = np.zeros((grid_h, grid_w), dtype=int)
    inp[top_h, :] = GREY

    for spec, (top, left) in zip(ring_specs, ring_positions):
        _draw_ring(inp, top, top_shift + left, spec["height"], spec["width"], spec["color"], spec["side"], spec["gap_idx"])

    for spec, (top, left) in zip(diagrams, diagram_positions):
        _draw_diagram(inp, top_h + 1 + top, bottom_shift + left, spec["color"], spec["mode"])

    expected = _build_expected(inp, ring_specs, ring_positions, top_shift)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
