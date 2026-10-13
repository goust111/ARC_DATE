from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
ANCHOR = Color.GREY
PALETTE = tuple(color for color in range(1, 10) if color != ANCHOR)

MIN_MARGIN = 1
MIN_OBJECTS = 2
MAX_OBJECTS = 3
HEIGHT_RANGE = (2, 6)
WIDTH_RANGE = (2, 6)
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (16, 29)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 140

RIGHT_LEAN = 1
LEFT_LEAN = -1

_RELATION_PATTERNS = [
    ("above", "below"),
    ("across", "above"),
    ("across", "below"),
    ("above", "below", "across"),
    ("across", "across", "below"),
    ("above", "below", "below"),
]
_pattern_queue: list[tuple[str, ...]] = []


def _neighbors8(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                out.append((rr, cc))
    return out


def _component_row_spans(coords: np.ndarray) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    rows = sorted(set(int(r) for r in coords[:, 0]))
    for r in rows:
        cols = np.sort(coords[coords[:, 0] == r, 1])
        left = int(cols[0])
        right = int(cols[-1])
        assert np.array_equal(cols, np.arange(left, right + 1)), "each object row must be one solid segment"
        spans.append((r, left, right))
    return spans


def _analyze_component(comp: dict, axis_row: int) -> dict:
    coords = np.asarray(comp["coords"], dtype=int)
    spans = _component_row_spans(coords)
    rows = [r for r, _c0, _c1 in spans]
    assert len(rows) >= 2, "each leaning object must span at least two rows"
    assert rows == list(range(rows[0], rows[-1] + 1)), "object rows must be contiguous"

    widths = [right - left + 1 for _r, left, right in spans]
    assert len(set(widths)) == 1, "all object rows must have equal length"
    width = int(widths[0])

    left_edges = [left for _r, left, _right in spans]
    diffs = [left_edges[i + 1] - left_edges[i] for i in range(len(left_edges) - 1)]
    assert len(set(diffs)) == 1, "adjacent rows must shift by one consistent offset"
    lean = int(diffs[0])
    assert lean in (LEFT_LEAN, RIGHT_LEAN), "adjacent rows must shift by exactly one column"

    axis_lefts = [left - lean * (r - axis_row) for r, left, _right in spans]
    assert len(set(axis_lefts)) == 1, "global axis intersection must be consistent across rows"

    return {
        "color": int(comp["color"]),
        "coords": coords,
        "top": int(rows[0]),
        "bottom": int(rows[-1]),
        "height": int(len(rows)),
        "width": width,
        "lean": lean,
        "axis_left": int(axis_lefts[0]),
        "spans": spans,
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, tuple[int, int], int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    assert bg != ANCHOR, "grey is reserved for the anchor pixel"

    anchor_cells = np.argwhere(g == ANCHOR)
    assert anchor_cells.shape == (1, 2), "expected exactly one grey anchor pixel"
    axis_row, anchor_col = map(int, anchor_cells[0])
    for rr, cc in _neighbors8(g.shape, axis_row, anchor_col):
        if (rr, cc) != (axis_row, anchor_col):
            assert int(g[rr, cc]) == bg, "anchor pixel must be isolated from every object"

    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, ANCHOR)]
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=colors,
        return_format="dict",
    )
    assert comps, "expected at least one leaning object"

    objects = [_analyze_component(comp, axis_row) for comp in comps]
    return bg, (axis_row, anchor_col), axis_row, objects


def _relation(spec: dict, axis_row: int) -> str:
    if spec["bottom"] < axis_row:
        return "above"
    if spec["top"] > axis_row:
        return "below"
    return "across"


def _paint_rect_row(out: np.ndarray, row: int, left: int, width: int, color: int) -> None:
    right = int(left) + int(width)
    assert 0 <= row < out.shape[0]
    assert 0 <= left and right <= out.shape[1]
    span = out[row, int(left) : right]
    assert np.all(span == BG), "deskewed objects must not overlap"
    out[row, int(left) : right] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten every leaning row-segment object by undoing its unit shear around the grey anchor row.

    Input structure:
    - A dominant background contains one isolated grey anchor pixel and one or more separate colored objects.
    - Each object is a solid parallelogram made of equal-length horizontal row segments.
    - Consecutive object rows are contiguous in row index and shift by exactly one column left or right.

    Rule:
    - Let the anchor pixel's row be the global axis row and keep the grey pixel fixed.
    - For each object, detect whether its rows drift left or right as the row index increases.
    - Move every object row horizontally by the signed amount needed to cancel that drift relative to the axis row: rows farther from the axis move more, and rows on opposite sides move in opposite directions.
    - Paint the shifted rows on the same canvas, producing a solid vertical rectangle for each object while leaving background unchanged.

    Key insight:
    - The grey pixel defines one shared invisible shear axis for all objects, so the correction is not anchored to each object's own top or bottom row; the whole object slides according to its distance from that external row.
    """
    bg, anchor, axis_row, objects = _extract_scene(grid)
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, bg, dtype=int)
    out[anchor] = ANCHOR

    for obj in sorted(objects, key=lambda item: (item["top"], item["axis_left"], item["color"])):
        for r, left, _right in obj["spans"]:
            shifted_left = int(left) - int(obj["lean"]) * (int(r) - axis_row)
            _paint_rect_row(out, int(r), shifted_left, int(obj["width"]), int(obj["color"]))

    return out


def _next_pattern() -> tuple[str, ...]:
    global _pattern_queue
    if not _pattern_queue:
        _pattern_queue = list(_RELATION_PATTERNS)
        random.shuffle(_pattern_queue)
    return _pattern_queue.pop()


def _coords_from_spec(spec: dict, axis_row: int, corrected: bool) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for r in range(int(spec["top"]), int(spec["top"]) + int(spec["height"])):
        if corrected:
            left = int(spec["axis_left"])
        else:
            left = int(spec["axis_left"]) + int(spec["lean"]) * (r - axis_row)
        for c in range(left, left + int(spec["width"])):
            coords.add((int(r), int(c)))
    return coords


def _expanded(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        blocked.update(_neighbors8(shape, r, c))
    return blocked


def _top_options(height: int, axis_row: int, grid_h: int, relation_name: str) -> list[int]:
    top_min = MIN_MARGIN
    top_max = grid_h - height - MIN_MARGIN
    if relation_name == "above":
        top_max = min(top_max, axis_row - height)
    elif relation_name == "below":
        top_min = max(top_min, axis_row + 1)
    elif relation_name == "across":
        top_min = max(top_min, axis_row - height + 1)
        top_max = min(top_max, axis_row)
    else:
        raise ValueError(f"unknown relation: {relation_name}")
    if top_min > top_max:
        return []
    return list(range(int(top_min), int(top_max) + 1))


def _axis_left_range(top: int, height: int, width: int, lean: int, axis_row: int, grid_w: int) -> tuple[int, int]:
    rows = range(int(top), int(top) + int(height))
    lo = MIN_MARGIN
    hi = grid_w - width - MIN_MARGIN
    for r in rows:
        offset = int(lean) * (int(r) - axis_row)
        lo = max(lo, MIN_MARGIN - offset)
        hi = min(hi, grid_w - width - MIN_MARGIN - offset)
    return int(lo), int(hi)


def _place_object(
    shape: tuple[int, int],
    axis_row: int,
    relation_name: str,
    lean: int,
    color: int,
    blocked_input: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
) -> dict | None:
    grid_h, grid_w = shape
    for _ in range(PLACEMENT_RETRIES):
        height = random.randint(*HEIGHT_RANGE)
        width = random.randint(*WIDTH_RANGE)
        tops = _top_options(height, axis_row, grid_h, relation_name)
        if not tops:
            continue
        top = random.choice(tops)
        left_min, left_max = _axis_left_range(top, height, width, lean, axis_row, grid_w)
        if left_min > left_max:
            continue
        axis_left = random.randint(left_min, left_max)
        spec = {
            "color": int(color),
            "top": int(top),
            "height": int(height),
            "width": int(width),
            "lean": int(lean),
            "axis_left": int(axis_left),
        }
        input_coords = _coords_from_spec(spec, axis_row, corrected=False)
        output_coords = _coords_from_spec(spec, axis_row, corrected=True)
        if input_coords & blocked_input:
            continue
        if output_coords & blocked_output:
            continue
        spec["bottom"] = int(top + height - 1)
        return spec
    return None


def _render_input(shape: tuple[int, int], specs: list[dict], axis_row: int, anchor_col: int) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        for r, c in _coords_from_spec(spec, axis_row, corrected=False):
            grid[r, c] = int(spec["color"])
    grid[int(axis_row), int(anchor_col)] = ANCHOR
    return grid


def _render_correct_output(shape: tuple[int, int], specs: list[dict], axis_row: int, anchor_col: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        for r, c in _coords_from_spec(spec, axis_row, corrected=True):
            out[r, c] = int(spec["color"])
    out[int(axis_row), int(anchor_col)] = ANCHOR
    return out


def _render_wrong_output(shape: tuple[int, int], specs: list[dict], axis_row: int, anchor_col: int, mode: str) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        bottom = int(spec["top"]) + int(spec["height"]) - 1
        for r in range(int(spec["top"]), bottom + 1):
            input_left = int(spec["axis_left"]) + int(spec["lean"]) * (r - axis_row)
            if mode == "top_anchor":
                left = input_left - int(spec["lean"]) * (r - int(spec["top"]))
            elif mode == "bottom_anchor":
                left = input_left - int(spec["lean"]) * (r - bottom)
            elif mode == "abs_distance":
                left = input_left - int(spec["lean"]) * abs(r - axis_row)
            else:
                raise ValueError(f"unknown wrong mode: {mode}")
            out[r, left : left + int(spec["width"])] = int(spec["color"])
    out[int(axis_row), int(anchor_col)] = ANCHOR
    return out


def _choose_anchor_col(inp: np.ndarray, out: np.ndarray, axis_row: int) -> int | None:
    candidates: list[int] = []
    for c in range(MIN_MARGIN, inp.shape[1] - MIN_MARGIN):
        if int(inp[axis_row, c]) != BG or int(out[axis_row, c]) != BG:
            continue
        input_clear = all(int(inp[rr, cc]) == BG for rr, cc in _neighbors8(inp.shape, axis_row, c) if (rr, cc) != (axis_row, c))
        output_clear = all(int(out[rr, cc]) == BG for rr, cc in _neighbors8(out.shape, axis_row, c) if (rr, cc) != (axis_row, c))
        if input_clear and output_clear:
            candidates.append(int(c))
    return random.choice(candidates) if candidates else None


def _spec_signature(spec: dict) -> tuple[int, int, int, int, int, int]:
    return (
        int(spec["color"]),
        int(spec["top"]),
        int(spec["height"]),
        int(spec["width"]),
        int(spec["lean"]),
        int(spec["axis_left"]),
    )


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, specs: list[dict], axis_row: int, anchor_col: int, pattern: tuple[str, ...]) -> bool:
    try:
        bg, anchor, parsed_axis, parsed_objects = _extract_scene(inp)
    except AssertionError:
        return False

    if bg != BG or anchor != (axis_row, anchor_col) or parsed_axis != axis_row:
        return False

    parsed_sigs = sorted(_spec_signature(obj) for obj in parsed_objects)
    target_sigs = sorted(_spec_signature(spec) for spec in specs)
    if parsed_sigs != target_sigs:
        return False

    relations = sorted(_relation(spec, axis_row) for spec in specs)
    if relations != sorted(pattern):
        return False

    if len({int(spec["lean"]) for spec in specs}) != 2:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    wrong_top = _render_wrong_output(inp.shape, specs, axis_row, anchor_col, "top_anchor")
    wrong_bottom = _render_wrong_output(inp.shape, specs, axis_row, anchor_col, "bottom_anchor")
    wrong_abs = _render_wrong_output(inp.shape, specs, axis_row, anchor_col, "abs_distance")
    if np.array_equal(expected, wrong_top):
        return False
    if np.array_equal(expected, wrong_bottom):
        return False
    if np.array_equal(expected, wrong_abs):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    pattern = list(_next_pattern())
    if len(pattern) < MAX_OBJECTS and random.random() < 0.45:
        pattern.append(random.choice(("above", "below", "across")))

    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    axis_row = random.randint(3, grid_h - 4)

    object_count = len(pattern)
    if object_count < MIN_OBJECTS or object_count > MAX_OBJECTS:
        return None

    leans = [LEFT_LEAN, RIGHT_LEAN]
    while len(leans) < object_count:
        leans.append(random.choice((LEFT_LEAN, RIGHT_LEAN)))
    random.shuffle(leans)

    colors = random.sample(PALETTE, object_count)
    order = list(range(object_count))
    order.sort(key=lambda idx: (pattern[idx] != "across", pattern[idx] == "below"))

    specs: list[dict] = []
    blocked_input: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()

    for idx in order:
        spec = _place_object(
            (grid_h, grid_w),
            axis_row,
            pattern[idx],
            leans[idx],
            colors[idx],
            blocked_input,
            blocked_output,
        )
        if spec is None:
            return None
        specs.append(spec)
        blocked_input.update(_expanded(_coords_from_spec(spec, axis_row, corrected=False), (grid_h, grid_w)))
        blocked_output.update(_expanded(_coords_from_spec(spec, axis_row, corrected=True), (grid_h, grid_w)))

    provisional_inp = np.full((grid_h, grid_w), BG, dtype=int)
    provisional_out = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        for r, c in _coords_from_spec(spec, axis_row, corrected=False):
            provisional_inp[r, c] = int(spec["color"])
        for r, c in _coords_from_spec(spec, axis_row, corrected=True):
            provisional_out[r, c] = int(spec["color"])

    anchor_col = _choose_anchor_col(provisional_inp, provisional_out, axis_row)
    if anchor_col is None:
        return None

    inp = _render_input((grid_h, grid_w), specs, axis_row, anchor_col)
    expected = _render_correct_output((grid_h, grid_w), specs, axis_row, anchor_col)
    if not _validate_semantics(inp, expected, specs, axis_row, anchor_col, tuple(pattern)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid leaning-tower input")
