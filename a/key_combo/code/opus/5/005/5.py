from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


MARKER_COLOR = 4
BG_COLOR = 0
PAIR_COLORS = [1, 2, 3, 6, 7, 8, 9]
RECT_SHAPES = [(3, 4), (4, 3)]
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
GAP_ODD = [3, 5, 7]
GAP_EVEN = [4, 6]
MARGIN = 2
PAIR_PADDING = 2
MAX_ATTEMPTS = 400


def _is_solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_center(comp: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = comp["bbox"]
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _same_shape(a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    return (ar1 - ar0, ac1 - ac0) == (br1 - br0, bc1 - bc0)


def _group_pairs(grid: np.ndarray) -> list[tuple[dict, dict]]:
    bg = int(get_bg(grid))
    rects = [
        comp
        for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True)
        if comp["color"] != MARKER_COLOR and _is_solid_rect(comp)
    ]
    groups: dict[tuple[int, int, int], list[dict]] = {}
    for comp in rects:
        r0, c0, r1, c1 = comp["bbox"]
        key = (int(comp["color"]), r1 - r0 + 1, c1 - c0 + 1)
        groups.setdefault(key, []).append(comp)
    pairs = []
    for comps in groups.values():
        if len(comps) != 2:
            raise AssertionError("each rectangle color/size signature must appear exactly twice")
        a, b = comps
        assert _same_shape(a, b)
        pairs.append((a, b))
    if len(pairs) != 2:
        raise AssertionError("expected exactly two rectangle pairs")
    return pairs


def _pair_orientation(a: dict, b: dict) -> str:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    same_rows = (ar0, ar1) == (br0, br1)
    same_cols = (ac0, ac1) == (bc0, bc1)
    if same_rows == same_cols:
        raise AssertionError("rectangles must align along exactly one axis")
    return "H" if same_rows else "V"


def _marker_for_pair(grid: np.ndarray, pair: tuple[dict, dict], orientation: str) -> tuple[int, int]:
    H, W = grid.shape
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER_COLOR)]
    if len(markers) != 2:
        raise AssertionError("expected exactly two boundary markers")
    boxes = [comp["bbox"] for comp in pair]
    r0 = min(b[0] for b in boxes)
    c0 = min(b[1] for b in boxes)
    r1 = max(b[2] for b in boxes)
    c1 = max(b[3] for b in boxes)
    if orientation == "H":
        candidates = [(r, c) for r, c in markers if r in (0, H - 1) and c0 <= c <= c1]
    else:
        candidates = [(r, c) for r, c in markers if c in (0, W - 1) and r0 <= r <= r1]
    if len(candidates) != 1:
        raise AssertionError("pair-marker association must be unique")
    return candidates[0]


def _near_component(pair: tuple[dict, dict], orientation: str, marker: tuple[int, int]) -> dict:
    a, b = pair
    mr, mc = marker
    if orientation == "H":
        da = abs(mc - _rect_center(a)[1])
        db = abs(mc - _rect_center(b)[1])
    else:
        da = abs(mr - _rect_center(a)[0])
        db = abs(mr - _rect_center(b)[0])
    if da == db:
        raise AssertionError("marker bias must break ties")
    return a if da < db else b


def _fill_pair(out: np.ndarray, pair: tuple[dict, dict], orientation: str, near: dict) -> None:
    a, b = pair
    if orientation == "H":
        left, right = sorted((a, b), key=lambda comp: comp["bbox"][1])
        gap = right["bbox"][1] - left["bbox"][3] - 1
        if gap < 1:
            raise AssertionError("horizontal gap must be positive")
        near_span = gap // 2 if gap % 2 == 0 else (gap + 1) // 2
        far_span = gap // 2
        left_span = near_span if near is left else far_span
        right_span = near_span if near is right else far_span
        lr0, _lc0, lr1, lc1 = left["bbox"]
        _rr0, rc0, _rr1, _rc1 = right["bbox"]
        if left_span:
            out[lr0 : lr1 + 1, lc1 + 1 : lc1 + 1 + left_span] = left["color"]
        if right_span:
            out[lr0 : lr1 + 1, rc0 - right_span : rc0] = right["color"]
    else:
        top, bottom = sorted((a, b), key=lambda comp: comp["bbox"][0])
        gap = bottom["bbox"][0] - top["bbox"][2] - 1
        if gap < 1:
            raise AssertionError("vertical gap must be positive")
        near_span = gap // 2 if gap % 2 == 0 else (gap + 1) // 2
        far_span = gap // 2
        top_span = near_span if near is top else far_span
        bottom_span = near_span if near is bottom else far_span
        tr0, tc0, tr1, tc1 = top["bbox"]
        br0, _bc0, _br1, _bc1 = bottom["bbox"]
        if top_span:
            out[tr1 + 1 : tr1 + 1 + top_span, tc0 : tc1 + 1] = top["color"]
        if bottom_span:
            out[br0 - bottom_span : br0, tc0 : tc1 + 1] = bottom["color"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each aligned rectangle gap in half, with the boundary marker giving the odd middle line to the nearer rectangle.

    Input structure:
    - Black background contains exactly two color-coded pairs of solid rectangles.
    - Each pair has two same-color, same-size rectangles aligned either side-by-side or top-to-bottom, separated by a background gap.
    - Each pair owns one yellow boundary marker on the outer border perpendicular to the pair axis: top/bottom for horizontal pairs, left/right for vertical pairs.

    Rule:
    - Group rectangles into same-color same-size pairs and determine whether each pair is horizontal or vertical.
    - Measure the face-to-face gap between the paired rectangles and associate the unique aligned yellow border marker with that pair.
    - Use the marker bias to choose the near rectangle: for horizontal pairs compare the marker column to the two rectangle centers; for vertical pairs compare the marker row to the two rectangle centers.
    - Fill the gap across the full rectangle height or width. Even gaps are split evenly. Odd gaps give the extra middle row or column to the near rectangle. Keep the original rectangles and both yellow markers unchanged.

    Key insight:
    - The yellow marker does not paint anything directly; it only breaks the symmetry of odd-width corridor splitting by deciding which rectangle is visually nearer.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for pair in _group_pairs(g):
        orientation = _pair_orientation(*pair)
        marker = _marker_for_pair(g, pair, orientation)
        near = _near_component(pair, orientation, marker)
        _fill_pair(out, pair, orientation, near)
    return out


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _horizontal_spec(H: int, W: int, gap: int, color: int) -> dict | None:
    h, w = random.choice(RECT_SHAPES)
    span_w = 2 * w + gap
    if H < h + 2 * MARGIN or W < span_w + 2 * MARGIN:
        return None
    r0 = random.randint(MARGIN, H - MARGIN - h)
    c0 = random.randint(MARGIN, W - MARGIN - span_w)
    marker_side = random.choice(["top", "bottom"])
    near_side = random.choice(["left", "right"])
    left_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
    right_c0 = c0 + w + gap
    right_bbox = (r0, right_c0, r0 + h - 1, right_c0 + w - 1)
    marker_col_choices = range(left_bbox[1], left_bbox[3] + 1) if near_side == "left" else range(right_bbox[1], right_bbox[3] + 1)
    marker = (0 if marker_side == "top" else H - 1, random.choice(list(marker_col_choices)))
    return {
        "orientation": "H",
        "color": color,
        "gap": gap,
        "near_side": near_side,
        "marker_side": marker_side,
        "marker": marker,
        "rects": [left_bbox, right_bbox],
        "corridor_bbox": (r0, c0, r0 + h - 1, right_bbox[3]),
    }


def _vertical_spec(H: int, W: int, gap: int, color: int) -> dict | None:
    h, w = random.choice(RECT_SHAPES)
    span_h = 2 * h + gap
    if H < span_h + 2 * MARGIN or W < w + 2 * MARGIN:
        return None
    r0 = random.randint(MARGIN, H - MARGIN - span_h)
    c0 = random.randint(MARGIN, W - MARGIN - w)
    marker_side = random.choice(["left", "right"])
    near_side = random.choice(["top", "bottom"])
    top_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
    bottom_r0 = r0 + h + gap
    bottom_bbox = (bottom_r0, c0, bottom_r0 + h - 1, c0 + w - 1)
    marker_row_choices = range(top_bbox[0], top_bbox[2] + 1) if near_side == "top" else range(bottom_bbox[0], bottom_bbox[2] + 1)
    marker = (random.choice(list(marker_row_choices)), 0 if marker_side == "left" else W - 1)
    return {
        "orientation": "V",
        "color": color,
        "gap": gap,
        "near_side": near_side,
        "marker_side": marker_side,
        "marker": marker,
        "rects": [top_bbox, bottom_bbox],
        "corridor_bbox": (r0, c0, bottom_bbox[2], c0 + w - 1),
    }


def _render_input_from_specs(H: int, W: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((H, W), BG_COLOR, dtype=int)
    for spec in specs:
        for r0, c0, r1, c1 in spec["rects"]:
            grid[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]
        mr, mc = spec["marker"]
        if grid[mr, mc] != BG_COLOR:
            raise AssertionError("marker must stay isolated on background")
        grid[mr, mc] = MARKER_COLOR
    return grid


def _expected_from_specs(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        color = spec["color"]
        gap = spec["gap"]
        near_span = gap // 2 if gap % 2 == 0 else (gap + 1) // 2
        far_span = gap // 2
        if spec["orientation"] == "H":
            left_bbox, right_bbox = spec["rects"]
            left_span = near_span if spec["near_side"] == "left" else far_span
            right_span = near_span if spec["near_side"] == "right" else far_span
            r0, _c0, r1, lc1 = left_bbox
            _rr0, rc0, _rr1, _rc1 = right_bbox
            if left_span:
                out[r0 : r1 + 1, lc1 + 1 : lc1 + 1 + left_span] = color
            if right_span:
                out[r0 : r1 + 1, rc0 - right_span : rc0] = color
        else:
            top_bbox, bottom_bbox = spec["rects"]
            top_span = near_span if spec["near_side"] == "top" else far_span
            bottom_span = near_span if spec["near_side"] == "bottom" else far_span
            tr0, tc0, tr1, tc1 = top_bbox
            br0, _bc0, _br1, _bc1 = bottom_bbox
            if top_span:
                out[tr1 + 1 : tr1 + 1 + top_span, tc0 : tc1 + 1] = color
            if bottom_span:
                out[br0 - bottom_span : br0, tc0 : tc1 + 1] = color
    return out


def _validate_structure(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(np.sum(grid == MARKER_COLOR)) != 2:
        return False
    try:
        pairs = _group_pairs(grid)
    except AssertionError:
        return False
    orientations = []
    parities = []
    for pair in pairs:
        try:
            orientation = _pair_orientation(*pair)
            marker = _marker_for_pair(grid, pair, orientation)
            _near_component(pair, orientation, marker)
        except AssertionError:
            return False
        orientations.append(orientation)
        if orientation == "H":
            left, right = sorted(pair, key=lambda comp: comp["bbox"][1])
            gap = right["bbox"][1] - left["bbox"][3] - 1
        else:
            top, bottom = sorted(pair, key=lambda comp: comp["bbox"][0])
            gap = bottom["bbox"][0] - top["bbox"][2] - 1
        if not 3 <= gap <= 7:
            return False
        parities.append(gap % 2)
    if sorted(orientations) != ["H", "V"]:
        return False
    if sorted(parities) != [0, 1]:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    odd_for_horizontal = random.choice([True, False])
    gap_h = random.choice(GAP_ODD if odd_for_horizontal else GAP_EVEN)
    gap_v = random.choice(GAP_EVEN if odd_for_horizontal else GAP_ODD)
    color_h, color_v = random.sample(PAIR_COLORS, 2)

    for _ in range(80):
        hspec = _horizontal_spec(H, W, gap_h, color_h)
        vspec = _vertical_spec(H, W, gap_v, color_v)
        if hspec is None or vspec is None:
            return None
        if _boxes_intersect(hspec["corridor_bbox"], vspec["corridor_bbox"], pad=PAIR_PADDING):
            continue
        inp = _render_input_from_specs(H, W, [hspec, vspec])
        expected = _expected_from_specs(inp, [hspec, vspec])
        if np.array_equal(inp, expected):
            continue
        if not _validate_structure(inp):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
