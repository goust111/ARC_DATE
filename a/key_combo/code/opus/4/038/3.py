from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, bbox_coords
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        PINK = 6
        ORANGE = 7

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


BACKGROUND = Color.BLACK
FRAME_COLOR = Color.BLUE
MIX_COLOR = Color.PINK
GROUP_COLORS = (Color.RED, Color.GREEN, Color.ORANGE)

GRID_HEIGHT_BOUNDS = (15, 23)
GRID_WIDTH_BOUNDS = (16, 24)
GROUP_COUNT_OPTIONS = (2, 3)
BBOX_HEIGHT_BOUNDS = (4, 8)
BBOX_WIDTH_BOUNDS = (4, 9)
MAX_SCENE_ATTEMPTS = 600
MAX_GROUP_ATTEMPTS = 160
MIX_PROBABILITY = 0.8
MIX_VISIBLE_RANGE = (1, 8)

ROW_TOP = "row_top"
ROW_BOTTOM = "row_bottom"
COL_LEFT = "col_left"
COL_RIGHT = "col_right"
BOTH_TL = "both_tl"
BOTH_TR = "both_tr"
BOTH_BL = "both_bl"
BOTH_BR = "both_br"

GROUP_MODES = (
    ROW_TOP,
    ROW_BOTTOM,
    COL_LEFT,
    COL_RIGHT,
    BOTH_TL,
    BOTH_TR,
    BOTH_BL,
    BOTH_BR,
)


def _row_side(mode: str) -> str | None:
    if mode in (ROW_TOP, BOTH_TL, BOTH_TR):
        return "top"
    if mode in (ROW_BOTTOM, BOTH_BL, BOTH_BR):
        return "bottom"
    return None


def _col_side(mode: str) -> str | None:
    if mode in (COL_LEFT, BOTH_TL, BOTH_BL):
        return "left"
    if mode in (COL_RIGHT, BOTH_TR, BOTH_BR):
        return "right"
    return None


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 <= ir1 <= or1 and oc0 <= ic0 <= ic1 <= oc1


def _bbox_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _draw_border(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _group_from_markers(color: int, points: list[tuple[int, int]]) -> dict:
    coords = np.array(points, dtype=int)
    bbox = bbox_coords(coords)
    row_counts = Counter(int(r) for r, _ in points)
    col_counts = Counter(int(c) for _, c in points)
    shared_rows = sorted(row for row, count in row_counts.items() if count >= 2)
    shared_cols = sorted(col for col, count in col_counts.items() if count >= 2)
    return {
        "color": int(color),
        "markers": [tuple(map(int, point)) for point in points],
        "bbox": tuple(map(int, bbox)),
        "shared_rows": shared_rows,
        "shared_cols": shared_cols,
    }


def _collect_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    groups = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND):
        coords = np.argwhere(g == color)
        assert color in GROUP_COLORS, "unexpected input color"
        assert len(coords) == 3, "each color must have exactly three markers"
        markers = [tuple(map(int, rc)) for rc in coords]
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r0 < r1 and c0 < c1, "markers must define a non-degenerate rectangle"
        for r, c in markers:
            assert r in (r0, r1) or c in (c0, c1), "markers must lie on the bbox boundary"
        assert len({r for r, _ in markers}) > 1 and len({c for _, c in markers}) > 1
        group = _group_from_markers(color, markers)
        assert group["shared_rows"] or group["shared_cols"], "each group must induce at least one stripe"
        groups.append(group)

    assert 2 <= len(groups) <= 3, "expected two or three color groups"
    return groups


def _render_output_from_groups(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    fills: dict[tuple[int, int], set[int]] = defaultdict(set)

    for group in groups:
        _draw_border(out, group["bbox"], FRAME_COLOR)

    for group in groups:
        r0, c0, r1, c1 = group["bbox"]
        color = group["color"]
        for row in group["shared_rows"]:
            for c in range(c0, c1 + 1):
                fills[(row, c)].add(color)
        for col in group["shared_cols"]:
            for r in range(r0, r1 + 1):
                fills[(r, col)].add(color)

    for (r, c), colors in fills.items():
        out[r, c] = next(iter(colors)) if len(colors) == 1 else MIX_COLOR

    for group in groups:
        for r, c in group["markers"]:
            out[r, c] = group["color"]

    return out


def _render_input_from_groups(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for group in groups:
        for r, c in group["markers"]:
            out[r, c] = group["color"]
    return out


def _spec_fill_cells(spec: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    cells: set[tuple[int, int]] = set()
    for row in spec["shared_rows"]:
        cells.update((row, c) for c in range(c0, c1 + 1))
    for col in spec["shared_cols"]:
        cells.update((r, col) for r in range(r0, r1 + 1))
    return cells


def _build_spec_from_bbox(
    color: int,
    mode: str,
    bbox: tuple[int, int, int, int],
    *,
    avoid_rows: set[int] | None = None,
    avoid_cols: set[int] | None = None,
) -> dict | None:
    avoid_rows = set() if avoid_rows is None else {int(v) for v in avoid_rows}
    avoid_cols = set() if avoid_cols is None else {int(v) for v in avoid_cols}
    r0, c0, r1, c1 = bbox

    if mode == ROW_TOP:
        choices = [c for c in range(c0 + 1, c1) if c not in avoid_cols]
        if not choices:
            return None
        c_mid = random.choice(choices)
        markers = [(r0, c0), (r0, c1), (r1, c_mid)]
    elif mode == ROW_BOTTOM:
        choices = [c for c in range(c0 + 1, c1) if c not in avoid_cols]
        if not choices:
            return None
        c_mid = random.choice(choices)
        markers = [(r1, c0), (r1, c1), (r0, c_mid)]
    elif mode == COL_LEFT:
        choices = [r for r in range(r0 + 1, r1) if r not in avoid_rows]
        if not choices:
            return None
        r_mid = random.choice(choices)
        markers = [(r0, c0), (r1, c0), (r_mid, c1)]
    elif mode == COL_RIGHT:
        choices = [r for r in range(r0 + 1, r1) if r not in avoid_rows]
        if not choices:
            return None
        r_mid = random.choice(choices)
        markers = [(r0, c1), (r1, c1), (r_mid, c0)]
    elif mode == BOTH_TL:
        markers = [(r0, c0), (r0, c1), (r1, c0)]
    elif mode == BOTH_TR:
        markers = [(r0, c0), (r0, c1), (r1, c1)]
    elif mode == BOTH_BL:
        markers = [(r0, c0), (r1, c0), (r1, c1)]
    elif mode == BOTH_BR:
        markers = [(r0, c1), (r1, c0), (r1, c1)]
    else:
        raise ValueError(f"unknown mode: {mode}")

    spec = _group_from_markers(color, markers)
    assert spec["bbox"] == bbox
    spec["mode"] = mode
    return spec


def _sample_spec(height: int, width: int, color: int, mode: str) -> dict:
    box_h = random.randint(*BBOX_HEIGHT_BOUNDS)
    box_w = random.randint(*BBOX_WIDTH_BOUNDS)
    r0 = random.randint(0, height - box_h)
    c0 = random.randint(0, width - box_w)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    spec = _build_spec_from_bbox(color, mode, (r0, c0, r1, c1))
    assert spec is not None
    return spec


def _has_row(spec: dict) -> bool:
    return bool(spec["shared_rows"])


def _has_col(spec: dict) -> bool:
    return bool(spec["shared_cols"])


def _pairwise_spatially_related(specs: list[dict]) -> bool:
    for i, spec_a in enumerate(specs):
        ar0, ac0, ar1, ac1 = spec_a["bbox"]
        for spec_b in specs[i + 1 :]:
            br0, bc0, br1, bc1 = spec_b["bbox"]
            if not (
                _intervals_overlap(ar0, ar1, br0, br1)
                or _intervals_overlap(ac0, ac1, bc0, bc1)
            ):
                return False
    return True


def _markers_clear_of_other_bboxes(specs: list[dict]) -> bool:
    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1 :]:
            if any(_point_in_bbox(point, spec_b["bbox"]) for point in spec_a["markers"]):
                return False
            if any(_point_in_bbox(point, spec_a["bbox"]) for point in spec_b["markers"]):
                return False
    return True


def _valid_bbox_relationships(specs: list[dict]) -> bool:
    for i, spec_a in enumerate(specs):
        box_a = spec_a["bbox"]
        area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        for spec_b in specs[i + 1 :]:
            box_b = spec_b["bbox"]
            area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
            if _bbox_contains(box_a, box_b) or _bbox_contains(box_b, box_a):
                return False
            overlap = _bbox_intersection_area(box_a, box_b)
            if overlap > 0 and overlap * 3 > min(area_a, area_b) * 2:
                return False
    return True


def _visible_marker_set(specs: list[dict]) -> set[tuple[int, int]]:
    return {point for spec in specs for point in spec["markers"]}


def _scene_has_required_modes(specs: list[dict], require_mix: bool) -> bool:
    has_row = any(_has_row(spec) for spec in specs)
    has_col = any(_has_col(spec) for spec in specs)
    if not (has_row and has_col):
        return False
    if require_mix:
        return has_row and has_col
    return True


def _validate_scene(height: int, width: int, specs: list[dict], require_mix: bool) -> bool:
    if not _valid_bbox_relationships(specs):
        return False
    if not _pairwise_spatially_related(specs):
        return False
    if not _markers_clear_of_other_bboxes(specs):
        return False
    if not _scene_has_required_modes(specs, require_mix):
        return False

    output = _render_output_from_groups((height, width), specs)
    marker_cells = _visible_marker_set(specs)
    mix_visible = [
        (r, c)
        for r, c in np.argwhere(output == MIX_COLOR)
        if (int(r), int(c)) not in marker_cells
    ]
    if require_mix:
        if not (MIX_VISIBLE_RANGE[0] <= len(mix_visible) <= MIX_VISIBLE_RANGE[1]):
            return False
    elif mix_visible:
        return False

    if np.count_nonzero(output == FRAME_COLOR) < 6:
        return False

    for spec in specs:
        color = spec["color"]
        fill_cells = _spec_fill_cells(spec) - set(spec["markers"])
        if not fill_cells:
            return False
        visible_fill = sum(output[r, c] == color for r, c in fill_cells)
        if visible_fill == 0:
            return False

        r0, c0, r1, c1 = spec["bbox"]
        border_cells = {
            (r0, c) for c in range(c0, c1 + 1)
        } | {
            (r1, c) for c in range(c0, c1 + 1)
        } | {
            (r, c0) for r in range(r0, r1 + 1)
        } | {
            (r, c1) for r in range(r0, r1 + 1)
        }
        visible_blue = any(output[r, c] == FRAME_COLOR for r, c in border_cells)
        if not visible_blue:
            return False

    return True


def _sample_modes(group_count: int, require_mix: bool) -> list[str]:
    for _ in range(80):
        modes = [random.choice(GROUP_MODES) for _ in range(group_count)]
        if not any(mode.startswith("row") or mode.startswith("both") for mode in modes):
            continue
        if not any(mode.startswith("col") or mode.startswith("both") for mode in modes):
            continue
        if require_mix and not any(mode.startswith("row") or mode.startswith("both") for mode in modes):
            continue
        return modes
    if group_count == 2:
        return [random.choice((ROW_TOP, BOTH_TL)), random.choice((COL_LEFT, BOTH_BR))]
    return [ROW_TOP, COL_LEFT, random.choice((BOTH_TR, BOTH_BL))]


def _build_mixing_pair(
    height: int,
    width: int,
    row_color: int,
    row_mode: str,
    col_color: int,
    col_mode: str,
) -> tuple[dict, dict] | None:
    row_side = _row_side(row_mode)
    col_side = _col_side(col_mode)
    if row_side is None or col_side is None:
        return None

    for _ in range(160):
        row_h = random.randint(4, 6)
        row_w = random.randint(6, min(BBOX_WIDTH_BOUNDS[1], width - 2))
        if height - row_h - 2 < 1:
            continue
        row_r0 = random.randint(1, height - row_h - 2)
        row_c0 = random.randint(0, width - row_w)
        row_box = (row_r0, row_c0, row_r0 + row_h - 1, row_c0 + row_w - 1)
        cross_row = row_box[0] if row_side == "top" else row_box[2]

        interior_cols = list(range(row_box[1] + 1, row_box[3]))
        if len(interior_cols) < 3:
            continue
        cross_col = random.choice(interior_cols[1:-1])

        min_col_w = 4
        if col_side == "left":
            max_col_w = min(BBOX_WIDTH_BOUNDS[1], width - cross_col)
            if max_col_w < min_col_w:
                continue
            col_w = random.randint(min_col_w, max_col_w)
            col_box = (0, cross_col, 0, cross_col + col_w - 1)
        else:
            max_col_w = min(BBOX_WIDTH_BOUNDS[1], cross_col - row_box[1] + 1)
            if max_col_w < min_col_w:
                continue
            col_w = random.randint(min_col_w, max_col_w)
            col_box = (0, cross_col - col_w + 1, 0, cross_col)

        if col_box[1] <= row_box[1] or col_box[3] >= row_box[3]:
            continue

        top_choices = list(range(0, row_box[0]))
        bottom_choices = list(range(row_box[2] + 1, height))
        if not top_choices or not bottom_choices:
            continue
        col_r0 = random.choice(top_choices)
        col_r1 = random.choice(bottom_choices)
        if not (BBOX_HEIGHT_BOUNDS[0] <= col_r1 - col_r0 + 1 <= BBOX_HEIGHT_BOUNDS[1]):
            continue
        col_box = (col_r0, col_box[1], col_r1, col_box[3])

        row_spec = _build_spec_from_bbox(
            row_color,
            row_mode,
            row_box,
            avoid_cols=set(range(col_box[1], col_box[3] + 1)),
        )
        col_spec = _build_spec_from_bbox(
            col_color,
            col_mode,
            col_box,
            avoid_rows=set(range(row_box[0], row_box[2] + 1)),
        )
        if row_spec is None or col_spec is None:
            continue
        if not _validate_scene(height, width, [row_spec, col_spec], require_mix=True):
            continue
        return row_spec, col_spec

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_SCENE_ATTEMPTS):
        height = random.randint(*GRID_HEIGHT_BOUNDS)
        width = random.randint(*GRID_WIDTH_BOUNDS)
        group_count = random.choice(GROUP_COUNT_OPTIONS)
        require_mix = random.random() < MIX_PROBABILITY
        modes = _sample_modes(group_count, require_mix)
        random.shuffle(modes)
        colors = list(GROUP_COLORS[:group_count])
        specs: list[dict] = []

        if require_mix:
            row_candidates = [(idx, mode) for idx, mode in enumerate(modes) if _row_side(mode) is not None]
            col_candidates = [(idx, mode) for idx, mode in enumerate(modes) if _col_side(mode) is not None]
            if row_candidates and col_candidates:
                row_idx, row_mode = random.choice(row_candidates)
                remaining_cols = [(idx, mode) for idx, mode in col_candidates if idx != row_idx]
                if remaining_cols:
                    col_idx, col_mode = random.choice(remaining_cols)
                    pair = _build_mixing_pair(
                        height,
                        width,
                        colors[row_idx],
                        row_mode,
                        colors[col_idx],
                        col_mode,
                    )
                    if pair is None:
                        continue
                    pair_specs = {row_idx: pair[0], col_idx: pair[1]}
                    specs = [pair_specs[idx] for idx in sorted(pair_specs)]

        for idx, (color, mode) in enumerate(zip(colors, modes)):
            if any(spec["color"] == color for spec in specs):
                continue
            placed = False
            for _ in range(MAX_GROUP_ATTEMPTS):
                spec = _sample_spec(height, width, color, mode)
                trial_specs = specs + [spec]
                if not _valid_bbox_relationships(trial_specs):
                    continue
                if not _markers_clear_of_other_bboxes(trial_specs):
                    continue
                placed = True
                specs.append(spec)
                break
            if not placed:
                specs = []
                break

        if len(specs) != group_count:
            continue

        if not _validate_scene(height, width, specs, require_mix):
            continue

        inp = _render_input_from_groups((height, width), specs)
        out = main(inp)
        expected = _render_output_from_groups((height, width), specs)
        assert np.array_equal(out, expected)
        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")


def main(grid: np.ndarray) -> np.ndarray:
    """Complete blue rectangular frames from three same-color markers, extend edge stripes, and blend overlapping stripe crossings.

    Input structure:
    - A black background contains two or three color groups.
    - Each group is exactly three isolated same-color markers: red, green, and optionally orange.
    - The three markers of a group lie on the boundary of their own axis-aligned bounding box and uniquely determine that rectangle.

    Rule:
    - For each color group, take the markers' minimal bounding box and draw its 1-pixel blue rectangular border.
    - If at least two markers in the group share a row, fill that entire bounding-box row from the left border to the right border in the marker color.
    - If at least two markers in the group share a column, fill that entire bounding-box column from the top border to the bottom border in the marker color.
    - Stripe colors overwrite blue border pixels, pixels reached by stripes of different groups become magenta, and the original marker pixels stay in their own color on the top layer.

    Key insight:
    - The same three markers serve two roles at once: their bounding box reconstructs the missing rectangle, while their repeated row and repeated column determine which completed frame edges must be recolored into directional stripes.
    """
    groups = _collect_groups(np.asarray(grid, dtype=int))
    return _render_output_from_groups(np.asarray(grid, dtype=int).shape, groups)
