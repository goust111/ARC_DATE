import random
from collections import defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _segment_info(coords: list[tuple[int, int]], color: int) -> dict:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    length = len(coords)
    assert 2 <= length <= 4, "segment length must be 2..4"

    if len(rows) == 1:
        assert len(cols) == length, "horizontal segment must be one cell thick"
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal segment must be contiguous"
        return {
            "color": int(color),
            "orientation": "H",
            "fixed": int(rows[0]),
            "start": int(cols[0]),
            "end": int(cols[-1]),
            "length": int(length),
        }

    assert len(cols) == 1, "segment must be horizontal or vertical"
    assert len(rows) == length, "vertical segment must be one cell thick"
    assert rows == list(range(rows[0], rows[-1] + 1)), "vertical segment must be contiguous"
    return {
        "color": int(color),
        "orientation": "V",
        "fixed": int(cols[0]),
        "start": int(rows[0]),
        "end": int(rows[-1]),
        "length": int(length),
    }


def _bridge_span(center: int, length: int) -> tuple[int, int]:
    # For even lengths, the chosen midpoint is the upper/left middle cell.
    start = center - (length - 1) // 2
    end = center + length // 2
    return int(start), int(end)


def _pair_and_bridge(segments: list[dict]) -> tuple[int, list[tuple[int, int]]]:
    assert len(segments) == 2, "each color must appear in exactly two segments"
    first, second = segments
    assert first["orientation"] == second["orientation"], "paired segments must share orientation"

    if first["orientation"] == "H":
        assert first["fixed"] == second["fixed"], "horizontal pair must lie on one row"
        ordered = sorted(segments, key=lambda d: (d["start"], d["end"]))
        first, second = ordered
        gap = int(second["start"] - first["end"] - 1)
        assert gap >= 1, "paired segments must have a positive gap"
        mid_col = int(first["end"] + 1 + (gap - 1) // 2)
        row = int(first["fixed"])
        r0, r1 = _bridge_span(row, gap)
        return gap, [(int(r), mid_col) for r in range(r0, r1 + 1)]

    assert first["fixed"] == second["fixed"], "vertical pair must lie on one column"
    ordered = sorted(segments, key=lambda d: (d["start"], d["end"]))
    first, second = ordered
    gap = int(second["start"] - first["end"] - 1)
    assert gap >= 1, "paired segments must have a positive gap"
    mid_row = int(first["end"] + 1 + (gap - 1) // 2)
    col = int(first["fixed"])
    c0, c1 = _bridge_span(col, gap)
    return gap, [(mid_row, int(c)) for c in range(c0, c1 + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a perpendicular bridge at the midpoint gap of each same-color segment pair.

    Input structure:
    - The background is black.
    - Each non-background object is a 4-connected, one-cell-thick horizontal or vertical segment of length 2 to 4.
    - Every non-background color appears in exactly two disconnected segments, and the two segments of one color are strictly collinear with a black gap between them.

    Rule:
    - Group segments by color and pair the two segments of each color.
    - Sort each pair along its shared row or column, then count the black gap length N between them.
    - Take the midpoint gap cell; when N is even, use the earlier/top-left of the two middle gap cells.
    - Keep both original segments unchanged and draw a same-color bridge perpendicular to them whose length is N.
    - For the discrete even-length bridge, treat the chosen midpoint as the upper/left middle cell of that bridge.

    Key insight:
    - The empty distance between paired collinear segments is not filled along the original line; it is rotated into a perpendicular bridge whose length equals the gap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]

    by_color: dict[int, list[dict]] = defaultdict(list)
    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each non-background color must have exactly two components"
        for comp in comps:
            by_color[color].append(_segment_info(comp, color))

    out = g.copy()
    for color, segments in by_color.items():
        _gap, bridge = _pair_and_bridge(segments)
        for r, c in bridge:
            assert 0 <= r < h and 0 <= c < w, "bridge must stay inside the grid"
            assert int(out[r, c]) in (bg, color), "bridges must not overwrite other colors"
            out[r, c] = int(color)

    return out


# --------------- generator ---------------

BG_COLOR = 0
GRID_MIN_SIDE = 20
GRID_MAX_SIDE = 30
MIN_PAIR_COUNT = 2
MAX_PAIR_COUNT_SMALL = 3
MAX_PAIR_COUNT_LARGE = 4
SEGMENT_LENGTHS = (2, 3, 4)
ODD_GAPS = (3, 5)
EVEN_GAPS = (2, 4, 6)
PLACEMENT_PAD = 1
MAX_GENERATION_ATTEMPTS = 400


def _sample_specs(pair_count: int) -> list[dict]:
    if pair_count >= 2 and random.random() < 0.7:
        orientations = ["H", "V"] + [random.choice(("H", "V")) for _ in range(pair_count - 2)]
        random.shuffle(orientations)
    else:
        orientations = [random.choice(("H", "V")) for _ in range(pair_count)]

    if pair_count >= 2 and random.random() < 0.7:
        parities = ["odd", "even"] + [random.choice(("odd", "even")) for _ in range(pair_count - 2)]
        random.shuffle(parities)
    else:
        parities = [random.choice(("odd", "even")) for _ in range(pair_count)]

    specs = []
    for orientation, parity in zip(orientations, parities):
        specs.append(
            {
                "orientation": orientation,
                "gap": random.choice(ODD_GAPS if parity == "odd" else EVEN_GAPS),
                "len1": random.choice(SEGMENT_LENGTHS),
                "len2": random.choice(SEGMENT_LENGTHS),
            }
        )
    return specs


def _build_pair_cells(spec: dict, fixed: int, start: int) -> dict:
    orientation = spec["orientation"]
    gap = int(spec["gap"])
    len1 = int(spec["len1"])
    len2 = int(spec["len2"])

    if orientation == "H":
        row = int(fixed)
        seg1 = [(row, int(c)) for c in range(start, start + len1)]
        seg2_start = int(start + len1 + gap)
        seg2 = [(row, int(c)) for c in range(seg2_start, seg2_start + len2)]
        mid_col = int(start + len1 + (gap - 1) // 2)
        r0, r1 = _bridge_span(row, gap)
        bridge = [(int(r), mid_col) for r in range(r0, r1 + 1)]
        return {"segments": seg1 + seg2, "bridge": bridge, "line_key": row}

    col = int(fixed)
    seg1 = [(int(r), col) for r in range(start, start + len1)]
    seg2_start = int(start + len1 + gap)
    seg2 = [(int(r), col) for r in range(seg2_start, seg2_start + len2)]
    mid_row = int(start + len1 + (gap - 1) // 2)
    c0, c1 = _bridge_span(col, gap)
    bridge = [(mid_row, int(c)) for c in range(c0, c1 + 1)]
    return {"segments": seg1 + seg2, "bridge": bridge, "line_key": col}


def _expanded_bbox(cells: list[tuple[int, int]], h: int, w: int, pad: int) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (
        max(0, min(rows) - pad),
        max(0, min(cols) - pad),
        min(h - 1, max(rows) + pad),
        min(w - 1, max(cols) + pad),
    )


def _place_pair(
    h: int,
    w: int,
    spec: dict,
    color: int,
    blocked: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
) -> dict | None:
    gap = int(spec["gap"])
    len1 = int(spec["len1"])
    len2 = int(spec["len2"])
    orientation = spec["orientation"]

    if orientation == "H":
        span = len1 + gap + len2
        r0_min = (gap - 1) // 2
        r0_max = h - 1 - gap // 2
        if r0_min > r0_max or span > w:
            return None
        candidates = [(r, c) for r in range(r0_min, r0_max + 1) if r not in used_rows for c in range(w - span + 1)]
    else:
        span = len1 + gap + len2
        c0_min = (gap - 1) // 2
        c0_max = w - 1 - gap // 2
        if c0_min > c0_max or span > h:
            return None
        candidates = [(c, r) for c in range(c0_min, c0_max + 1) if c not in used_cols for r in range(h - span + 1)]

    random.shuffle(candidates)
    for fixed, start in candidates:
        pair_cells = _build_pair_cells(spec, fixed, start)
        all_cells = pair_cells["segments"] + pair_cells["bridge"]
        bbox = _expanded_bbox(all_cells, h, w, PLACEMENT_PAD)
        r0, c0, r1, c1 = bbox
        if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        pair_cells["color"] = int(color)
        pair_cells["bbox"] = bbox
        return pair_cells
    return None


def _render_input_output(h: int, w: int, placed_pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG_COLOR, dtype=int)
    out = np.full((h, w), BG_COLOR, dtype=int)

    for pair in placed_pairs:
        color = int(pair["color"])
        for r, c in pair["segments"]:
            inp[r, c] = color
            out[r, c] = color
        for r, c in pair["bridge"]:
            out[r, c] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    w = random.randint(GRID_MIN_SIDE, GRID_MAX_SIDE)
    max_pairs = MAX_PAIR_COUNT_LARGE if h * w >= 600 else MAX_PAIR_COUNT_SMALL
    pair_count = random.randint(MIN_PAIR_COUNT, max_pairs)
    colors = random.sample(range(1, 10), pair_count)
    specs = _sample_specs(pair_count)

    order = sorted(
        range(pair_count),
        key=lambda i: specs[i]["gap"] + specs[i]["len1"] + specs[i]["len2"],
        reverse=True,
    )

    blocked = np.zeros((h, w), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    placed_pairs: list[dict] = []

    for idx in order:
        placed = _place_pair(h, w, specs[idx], colors[idx], blocked, used_rows, used_cols)
        if placed is None:
            return None
        r0, c0, r1, c1 = placed["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
        if specs[idx]["orientation"] == "H":
            used_rows.add(int(placed["line_key"]))
        else:
            used_cols.add(int(placed["line_key"]))
        placed_pairs.append(placed)

    inp, expected = _render_input_output(h, w, placed_pairs)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
