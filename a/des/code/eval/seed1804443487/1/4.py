import random
from collections import Counter, deque

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    solid_cols = []
    for c in range(w):
        col = grid[:, c]
        if int(col[0]) != bg and np.all(col == col[0]):
            solid_cols.append(c)
    assert len(solid_cols) == 1, "expected exactly one solid non-background divider column"
    divider_col = solid_cols[0]
    divider_color = int(grid[0, divider_col])
    assert 0 < divider_col < w - 1, "divider must split the grid into two non-empty panels"
    assert divider_color != bg
    return divider_col, divider_color


def _classify_ring(comp: dict) -> dict:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 4 and w >= 4, "ring bbox must allow a hollow interior"

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    rr, cc = np.indices((h, w))
    perimeter = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
    assert np.all(mask <= perimeter), "ring pixels must lie on their rectangle perimeter"
    assert mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1], "gap may not consume corners"

    side_gaps = []
    top = [c for c in range(1, w - 1) if not mask[0, c]]
    bottom = [c for c in range(1, w - 1) if not mask[h - 1, c]]
    left = [r for r in range(1, h - 1) if not mask[r, 0]]
    right = [r for r in range(1, h - 1) if not mask[r, w - 1]]
    if top:
        side_gaps.append(("top", top))
    if bottom:
        side_gaps.append(("bottom", bottom))
    if left:
        side_gaps.append(("left", left))
    if right:
        side_gaps.append(("right", right))
    assert len(side_gaps) == 1, "each ring must have exactly one missing side segment"

    side, gap = side_gaps[0]
    assert gap == list(range(gap[0], gap[-1] + 1)), "gap must be contiguous"

    expected = perimeter.copy()
    if side == "top":
        expected[0, gap[0] : gap[-1] + 1] = False
    elif side == "bottom":
        expected[h - 1, gap[0] : gap[-1] + 1] = False
    elif side == "left":
        expected[gap[0] : gap[-1] + 1, 0] = False
    else:
        expected[gap[0] : gap[-1] + 1, w - 1] = False
    assert np.array_equal(mask, expected), "ring must be a rectangle frame with one side gap"

    return {
        "ring_color": int(comp["color"]),
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_start": int(gap[0]),
        "gap_end": int(gap[-1]),
        "gap_length": int(len(gap)),
    }


def _fill_gap(out: np.ndarray, ring: dict, color: int) -> None:
    r0, c0, r1, c1 = ring["bbox"]
    start = ring["gap_start"]
    end = ring["gap_end"]
    side = ring["side"]
    if side == "top":
        out[r0, c0 + start : c0 + end + 1] = color
    elif side == "bottom":
        out[r1, c0 + start : c0 + end + 1] = color
    elif side == "left":
        out[r0 + start : r0 + end + 1, c0] = color
    else:
        out[r0 + start : r0 + end + 1, c1] = color


def _segment_lengths(panel: np.ndarray, bg: int) -> list[int]:
    lengths: list[int] = []
    for comp in _monochrome_components(panel, bg):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert min(h, w) == 1, "right-panel filler must be one cell thick"
        assert comp["area"] == max(h, w), "filler must be a solid straight segment"
        lengths.append(max(h, w))
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each left-panel ring by matching its gap length to a right-panel filler,
    but use the interior marker color as the fill color. The output is only the left panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    divider_col, divider_color = _find_divider(g, bg)
    assert divider_color != bg

    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]
    assert left.size > 0 and right.size > 0

    components = _monochrome_components(left, bg)
    rings: list[dict] = []
    markers: list[dict] = []
    for comp in components:
        if comp["area"] == 1:
            markers.append(comp)
        else:
            rings.append(_classify_ring(comp))
    assert rings, "left panel must contain at least one ring"
    assert len(markers) == len(rings), "expected one marker per ring"

    assigned_markers: set[int] = set()
    out = left.copy()
    gap_lengths = []
    for ring_idx, ring in enumerate(rings):
        r0, c0, r1, c1 = ring["bbox"]
        contained = []
        for marker_idx, comp in enumerate(markers):
            mr, mc = map(int, comp["coords"][0])
            if r0 < mr < r1 and c0 < mc < c1:
                contained.append((marker_idx, comp))
        assert len(contained) == 1, "each ring must contain exactly one interior marker"
        marker_idx, marker = contained[0]
        assert marker_idx not in assigned_markers, "markers may not belong to multiple rings"
        assigned_markers.add(marker_idx)

        marker_color = int(marker["color"])
        assert marker_color != ring["ring_color"], "marker color must differ from its ring color"
        mr, mc = map(int, marker["coords"][0])
        out[mr, mc] = bg
        _fill_gap(out, ring, marker_color)
        gap_lengths.append(ring["gap_length"])

    assert len(assigned_markers) == len(markers), "all marker pixels must be used"

    filler_lengths = _segment_lengths(right, bg)
    assert len(filler_lengths) == len(gap_lengths), "right-panel filler count must match ring count"
    assert len(set(filler_lengths)) == len(filler_lengths), "filler lengths must be unique"
    assert len(set(gap_lengths)) == len(gap_lengths), "gap lengths must be unique"
    assert Counter(filler_lengths) == Counter(gap_lengths), "filler lengths must match ring gaps"

    return out


def _try_place(boxes: list[tuple[int, int]], height: int, width: int, attempts: int = 200) -> list[tuple[int, int]] | None:
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True)
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(boxes)
    for idx in order:
        box_h, box_w = boxes[idx]
        placed = False
        for _ in range(attempts):
            if box_h + 2 > height or box_w + 2 > width:
                return None
            r0 = random.randint(1, height - box_h - 1)
            c0 = random.randint(1, width - box_w - 1)
            if occupied[r0 - 1 : r0 + box_h + 1, c0 - 1 : c0 + box_w + 1].any():
                continue
            occupied[r0 - 1 : r0 + box_h + 1, c0 - 1 : c0 + box_w + 1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return [p for p in placements if p is not None]


def _ring_spec(n_rings: int) -> list[dict]:
    gap_lengths = random.sample(list(range(2, 8)), n_rings)
    ring_colors = random.sample(PALETTE, n_rings)
    marker_pool = [c for c in PALETTE if c not in ring_colors]
    specs: list[dict] = []
    for ring_color, gap_length in zip(ring_colors, gap_lengths):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            outer_h = random.randint(4, 7)
            outer_w = gap_length + 2 + random.randint(0, 2)
        else:
            outer_h = gap_length + 2 + random.randint(0, 2)
            outer_w = random.randint(4, 7)
        marker_color = random.choice([c for c in marker_pool if c != ring_color])
        specs.append(
            {
                "ring_color": ring_color,
                "marker_color": marker_color,
                "gap_length": gap_length,
                "side": side,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
    return specs


def _draw_ring(panel: np.ndarray, spec: dict, top: int, left: int) -> dict:
    ring_color = int(spec["ring_color"])
    marker_color = int(spec["marker_color"])
    h = int(spec["outer_h"])
    w = int(spec["outer_w"])
    side = spec["side"]
    gap_length = int(spec["gap_length"])

    panel[top, left : left + w] = ring_color
    panel[top + h - 1, left : left + w] = ring_color
    panel[top : top + h, left] = ring_color
    panel[top : top + h, left + w - 1] = ring_color

    if side in ("top", "bottom"):
        span = w - 2
        start = random.randint(1, span - gap_length + 1)
        if side == "top":
            panel[top, left + start : left + start + gap_length] = BG
        else:
            panel[top + h - 1, left + start : left + start + gap_length] = BG
    else:
        span = h - 2
        start = random.randint(1, span - gap_length + 1)
        if side == "left":
            panel[top + start : top + start + gap_length, left] = BG
        else:
            panel[top + start : top + start + gap_length, left + w - 1] = BG

    marker_r = random.randint(top + 1, top + h - 2)
    marker_c = random.randint(left + 1, left + w - 2)
    panel[marker_r, marker_c] = marker_color

    return {
        "bbox": (top, left, top + h - 1, left + w - 1),
        "marker": (marker_r, marker_c),
        "marker_color": marker_color,
        "side": side,
        "gap_start": start,
        "gap_end": start + gap_length - 1,
        "gap_length": gap_length,
    }


def _draw_segment(panel: np.ndarray, top: int, left: int, length: int, color: int, horizontal: bool) -> None:
    if horizontal:
        panel[top, left : left + length] = color
    else:
        panel[top : top + length, left] = color


def _gen_attempt() -> np.ndarray | None:
    n_rings = random.choice([2, 3, 4])
    ring_specs = _ring_spec(n_rings)
    filler_colors = random.sample(PALETTE, n_rings)
    filler_specs = []
    max_vertical = 0
    max_horizontal = 0
    for spec, color in zip(ring_specs, filler_colors):
        horizontal = random.choice([False, True])
        filler_specs.append({"length": spec["gap_length"], "color": color, "horizontal": horizontal})
        if horizontal:
            max_horizontal = max(max_horizontal, spec["gap_length"])
        else:
            max_vertical = max(max_vertical, spec["gap_length"])

    max_ring_h = max(spec["outer_h"] for spec in ring_specs)
    max_ring_w = max(spec["outer_w"] for spec in ring_specs)
    height_min = max(12, max_ring_h + 2, max_vertical + 2)
    height_max = max(height_min, 22)
    height = random.randint(height_min, height_max)

    left_min = max(14, max_ring_w + 2)
    left_max = max(left_min, 22)
    left_w = random.randint(left_min, left_max)

    right_min = max(6, max_horizontal + 2)
    right_max = max(right_min, 10)
    right_w = random.randint(right_min, right_max)

    left_panel = np.full((height, left_w), BG, dtype=int)
    right_panel = np.full((height, right_w), BG, dtype=int)

    ring_boxes = [(spec["outer_h"], spec["outer_w"]) for spec in ring_specs]
    ring_places = _try_place(ring_boxes, height, left_w)
    if ring_places is None:
        return None

    segment_boxes = [(1, item["length"]) if item["horizontal"] else (item["length"], 1) for item in filler_specs]
    segment_places = _try_place(segment_boxes, height, right_w)
    if segment_places is None:
        return None

    ring_outputs = []
    for spec, (top, left) in zip(ring_specs, ring_places):
        ring_outputs.append(_draw_ring(left_panel, spec, top, left))

    for item, (top, left) in zip(filler_specs, segment_places):
        _draw_segment(right_panel, top, left, item["length"], item["color"], item["horizontal"])

    grid = np.full((height, left_w + 1 + right_w), BG, dtype=int)
    grid[:, :left_w] = left_panel
    grid[:, left_w] = DIVIDER
    grid[:, left_w + 1 :] = right_panel

    expected = left_panel.copy()
    for ring in ring_outputs:
        mr, mc = ring["marker"]
        expected[mr, mc] = BG
        _fill_gap(expected, ring, int(ring["marker_color"]))

    try:
        solved = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
