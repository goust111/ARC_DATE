import numpy as np


BG = 0
RED = 2
GREEN = 3
ORANGE = 7
TEAL = 8

MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_bg(out: np.ndarray, rows: slice | int, cols: slice | int, color: int) -> None:
    region = out[rows, cols]
    region[region == BG] = int(color)


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BG, RED, GREEN, ORANGE, TEAL}), "unexpected colors"

    rect_mask = g == ORANGE
    rect_bbox = _bbox(rect_mask)
    assert rect_bbox is not None, "expected one orange rectangle"
    r0, c0, r1, c1 = rect_bbox
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    assert rect_h >= 3 and rect_w >= 4, "rectangle too small"
    assert int(rect_mask.sum()) == rect_h * rect_w, "orange object must be a solid rectangle"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == ORANGE), "orange bbox must be filled"

    red_coords = np.argwhere(g == RED)
    teal_coords = np.argwhere(g == TEAL)
    assert len(red_coords) == 1 and len(teal_coords) == 1, "expected exactly one red and one teal marker"
    red = tuple(map(int, red_coords[0]))
    teal = tuple(map(int, teal_coords[0]))

    marker_row = r0 - 1
    assert marker_row >= 0, "markers must fit above the rectangle"
    assert red[0] == teal[0] == marker_row, "markers must share the row directly above the rectangle"
    assert c0 <= red[1] <= c1 and c0 <= teal[1] <= c1, "markers must lie above rectangle columns"
    assert abs(red[1] - teal[1]) >= 2, "markers must be non-adjacent"

    controls = {"top": set(), "bottom": set(), "left": set(), "right": set()}
    control_coords = []
    for r, c in np.argwhere(g == GREEN):
        rr, cc = int(r), int(c)
        hits = 0
        if rr < r0 and c0 <= cc <= c1:
            controls["top"].add(cc)
            hits += 1
        if rr > r1 and c0 <= cc <= c1:
            controls["bottom"].add(cc)
            hits += 1
        if cc < c0 and r0 <= rr <= r1:
            controls["left"].add(rr)
            hits += 1
        if cc > c1 and r0 <= rr <= r1:
            controls["right"].add(rr)
            hits += 1
        assert hits == 1, "each green control must align with exactly one side strip"
        control_coords.append((rr, cc))

    excluded = "left" if red[1] < teal[1] else "right"
    return {
        "bbox": rect_bbox,
        "red": red,
        "teal": teal,
        "controls": controls,
        "control_coords": tuple(control_coords),
        "excluded": excluded,
    }


def _apply_rule(grid: np.ndarray, spec: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = spec["bbox"]
    red_r, red_c = spec["red"]
    teal_r, teal_c = spec["teal"]
    controls = spec["controls"]
    excluded = spec["excluded"]

    out[red_r, red_c] = BG
    out[teal_r, teal_c] = RED

    for c in range(c0, c1 + 1):
        if c not in controls["top"]:
            _paint_bg(out, slice(0, r0), c, ORANGE)
        if c not in controls["bottom"]:
            _paint_bg(out, slice(r1 + 1, out.shape[0]), c, ORANGE)

    for r in range(r0, r1 + 1):
        if excluded != "left" and r not in controls["left"]:
            _paint_bg(out, r, slice(0, c0), ORANGE)
        if excluded != "right" and r not in controls["right"]:
            _paint_bg(out, r, slice(c1 + 1, out.shape[1]), ORANGE)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move the red marker onto teal and extend the rectangle on every non-excluded side.

    Input structure:
    - The grid has a black background, one solid orange rectangle, one red marker, one teal marker, and optional green control pixels.
    - The red and teal markers sit on the same row directly above the rectangle, separated horizontally within the rectangle's column span.
    - Each green control pixel lies in exactly one side strip aligned with the rectangle, outside the rectangle interior.

    Rule:
    - Read the red-to-teal horizontal order: red left of teal excludes the left side; red right of teal excludes the right side.
    - Move the red marker onto the teal cell and erase the original red cell to background.
    - Keep the rectangle and green controls unchanged, then extend orange stripes from the rectangle toward the grid boundary on the top, bottom, and the non-excluded horizontal side.
    - For each active side, skip the projected row or column whose strip contains a green control pixel on that side; only background cells are filled.

    Key insight:
    - The same two markers serve two roles at once: their order encodes which horizontal side is globally disabled, while the moved red marker stays embedded inside the top-side extension pattern.
    """
    spec = _parse_input(np.asarray(grid, dtype=int))
    return _apply_rule(grid, spec)


def _choose_distinct_indices(rng: np.random.Generator, length: int, max_count: int) -> list[int]:
    count = int(rng.integers(0, max_count + 1))
    if count == 0:
        return []
    vals = rng.choice(length, size=count, replace=False)
    return sorted(int(v) for v in vals)


def _choose_control_indices(
    rng: np.random.Generator,
    rect_h: int,
    rect_w: int,
    excluded: str,
) -> dict[str, list[int]]:
    max_tb = min(2, rect_w)
    max_lr = min(2, rect_h)
    side_max = {"top": max_tb, "bottom": max_tb, "left": max_lr, "right": max_lr}

    mode = rng.choice(["none", "excluded_only", "mixed"], p=[0.18, 0.24, 0.58])
    chosen = {side: [] for side in ("top", "bottom", "left", "right")}
    if mode == "none":
        return chosen

    if mode == "excluded_only":
        side = excluded
        count = int(rng.integers(1, side_max[side] + 1))
        chosen[side] = sorted(int(v) for v in rng.choice(rect_h if side in ("left", "right") else rect_w, size=count, replace=False))
        return chosen

    for side in ("top", "bottom", "left", "right"):
        length = rect_h if side in ("left", "right") else rect_w
        chosen[side] = _choose_distinct_indices(rng, length, side_max[side])

    active_sides = [side for side in ("top", "bottom", "left", "right") if side != excluded]
    if not any(chosen[side] for side in active_sides):
        side = active_sides[int(rng.integers(0, len(active_sides)))]
        length = rect_h if side in ("left", "right") else rect_w
        chosen[side] = [int(rng.integers(0, length))]
    return chosen


def _choose_marker_columns(
    rng: np.random.Generator,
    c0: int,
    rect_w: int,
    excluded: str,
) -> tuple[int, int]:
    pairs = []
    cols = list(range(c0, c0 + rect_w))
    for i, left_col in enumerate(cols):
        for right_col in cols[i + 1 :]:
            if right_col - left_col >= 2:
                pairs.append((left_col, right_col))
    assert pairs, "rectangle must be wide enough for non-adjacent markers"
    left_col, right_col = pairs[int(rng.integers(0, len(pairs)))]
    if excluded == "left":
        return left_col, right_col
    return right_col, left_col


def _place_controls(
    rng: np.random.Generator,
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    chosen: dict[str, list[int]],
) -> tuple[list[tuple[int, int]], dict[str, set[int]]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    coords = []
    blocked = {side: set() for side in ("top", "bottom", "left", "right")}

    for idx in chosen["top"]:
        rr = int(rng.integers(0, r0 - 1))
        cc = c0 + idx
        coords.append((rr, cc))
        blocked["top"].add(cc)
    for idx in chosen["bottom"]:
        rr = int(rng.integers(r1 + 1, h))
        cc = c0 + idx
        coords.append((rr, cc))
        blocked["bottom"].add(cc)
    for idx in chosen["left"]:
        rr = r0 + idx
        cc = int(rng.integers(0, c0))
        coords.append((rr, cc))
        blocked["left"].add(rr)
    for idx in chosen["right"]:
        rr = r0 + idx
        cc = int(rng.integers(c1 + 1, w))
        coords.append((rr, cc))
        blocked["right"].add(rr)

    return coords, blocked


def _render_input(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    red: tuple[int, int],
    teal: tuple[int, int],
    controls: list[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    grid = np.zeros((h, w), dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
    grid[red] = RED
    grid[teal] = TEAL
    for rc in controls:
        grid[rc] = GREEN
    return grid


def _verify_generated(inp: np.ndarray, expected: np.ndarray, spec: dict) -> bool:
    if not np.array_equal(expected, _apply_rule(inp, spec)):
        return False

    r0, c0, r1, c1 = spec["bbox"]
    red = spec["red"]
    teal = spec["teal"]
    blocked = spec["controls"]
    excluded = spec["excluded"]

    if np.count_nonzero(inp == RED) != 1 or np.count_nonzero(inp == TEAL) != 1:
        return False
    if np.count_nonzero(expected == TEAL) != 0 or np.count_nonzero(expected == RED) != 1:
        return False
    if expected[teal] != RED:
        return False

    if excluded == "left" and not np.array_equal(inp[r0 : r1 + 1, :c0], expected[r0 : r1 + 1, :c0]):
        return False
    if excluded == "right" and not np.array_equal(inp[r0 : r1 + 1, c1 + 1 :], expected[r0 : r1 + 1, c1 + 1 :]):
        return False

    if red[1] not in blocked["top"] and expected[red] not in {ORANGE, BG}:
        return False
    if expected[teal] != RED:
        return False

    for rr, cc in spec["control_coords"]:
        if inp[rr, cc] != GREEN or expected[rr, cc] != GREEN:
            return False
    return not np.array_equal(inp, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 19))
    w = int(rng.integers(12, 19))
    rect_h = int(rng.integers(3, 6))
    rect_w = int(rng.integers(4, 6))

    if h < rect_h + 6 or w < rect_w + 4:
        return None

    r0 = int(rng.integers(3, h - rect_h - 2))
    c0 = int(rng.integers(2, w - rect_w - 1))
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    if r0 < 3 or h - 1 - r1 < 2 or c0 < 2 or w - 1 - c1 < 2:
        return None

    excluded = str(rng.choice(["left", "right"]))
    red_c, teal_c = _choose_marker_columns(rng, c0, rect_w, excluded)
    marker_r = r0 - 1
    red = (marker_r, red_c)
    teal = (marker_r, teal_c)

    chosen = _choose_control_indices(rng, rect_h, rect_w, excluded)
    control_coords, blocked = _place_controls(rng, (h, w), (r0, c0, r1, c1), chosen)
    inp = _render_input((h, w), (r0, c0, r1, c1), red, teal, control_coords)

    spec = {
        "bbox": (r0, c0, r1, c1),
        "red": red,
        "teal": teal,
        "controls": blocked,
        "control_coords": tuple(control_coords),
        "excluded": excluded,
    }
    expected = _apply_rule(inp, spec)

    if not _verify_generated(inp, expected, spec):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
