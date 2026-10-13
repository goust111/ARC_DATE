from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    """Load the shared ARC helper module by absolute path so SourceFileLoader still works."""
    module_name = "_arc_utils_opus4_014_8"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, "failed to locate arc_utils.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_arc_utils = _load_arc_utils()
Color = _arc_utils.Color
bbox = _arc_utils.bbox
get_bg = _arc_utils.get_bg
paste = _arc_utils.paste


GATE_CYCLE = (Color.BLUE, Color.RED, Color.GREEN)
RECT_SHAPES = ((2, 3), (3, 2))
BACKGROUND_CHOICES = (Color.YELLOW, Color.GREY, Color.PINK, Color.TEAL)
BRANCH_PROGRAM = (
    ("horizontal", 1, 4),
    ("vertical", -1, 3),
    ("horizontal", -1, 2),
    ("vertical", 1, 4),
)
GENERATION_ATTEMPTS = 300
_GEN_CALL_COUNT = 0


def _solid_rect_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = grid == int(color)
    box = bbox(mask)
    assert box is not None, f"expected color {color} to appear"
    r0, c0, r1, c1 = box
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(np.count_nonzero(mask)) == area, "colored object must be one solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)), "rectangle bbox must be solid"
    return int(r0), int(c0), int(r1), int(c1)


def _ordered_gate_positions(
    grid: np.ndarray,
    bg: int,
    source: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> tuple[str, list[int]]:
    sr0, sc0, sr1, sc1 = source
    tr0, tc0, tr1, tc1 = target
    same_rows = (sr0, sr1) == (tr0, tr1)
    same_cols = (sc0, sc1) == (tc0, tc1)
    assert same_rows != same_cols, "rectangles must align on exactly one axis"

    gates: list[int] = []
    h, w = grid.shape

    if same_rows:
        assert sr0 > 0 and sr1 + 1 < h, "horizontal gate corridor needs rows above and below"
        if sc0 < tc0:
            corridor = range(sc1 + 1, tc0)
        else:
            corridor = range(sc0 - 1, tc1, -1)
        for c in corridor:
            flanked = grid[sr0 - 1, c] == Color.BLACK and grid[sr1 + 1, c] == Color.BLACK
            if flanked:
                assert np.all(grid[sr0 : sr1 + 1, c] == bg), "gate opening must be background"
                gates.append(int(c))
        return "horizontal", gates

    assert sc0 > 0 and sc1 + 1 < w, "vertical gate corridor needs columns on both sides"
    if sr0 < tr0:
        corridor = range(sr1 + 1, tr0)
    else:
        corridor = range(sr0 - 1, tr1, -1)
    for r in corridor:
        flanked = grid[r, sc0 - 1] == Color.BLACK and grid[r, sc1 + 1] == Color.BLACK
        if flanked:
            assert np.all(grid[r, sc0 : sc1 + 1] == bg), "gate opening must be background"
            gates.append(int(r))
    return "vertical", gates


def _paint_gate_opening(
    grid: np.ndarray,
    orientation: str,
    cross0: int,
    cross1: int,
    axis_pos: int,
    color: int,
) -> None:
    if orientation == "horizontal":
        grid[cross0 : cross1 + 1, axis_pos] = int(color)
    else:
        grid[axis_pos, cross0 : cross1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the orange rectangle onto the maroon rectangle and color each crossed gate opening by its order from the source side.

    Input structure:
    - The grid has one dominant uniform background plus visible black obstacle segments.
    - There is one solid orange rectangle and one solid maroon rectangle of the same size.
    - The two rectangles share either their row span or their column span, so the move is straight and axis-aligned.

    Rule:
    - Determine the movement direction from the orange rectangle toward the maroon rectangle.
    - Erase the original orange rectangle to the background, then replace the maroon rectangle with orange.
    - Scan only the corridor cells between the two rectangles, in source-to-target order.
    - Whenever the moving rectangle's opening is flanked by black obstacle segments perpendicular to the motion, color that opening with the repeating cycle blue, red, green.
    - Leave all non-gate background cells, one-sided stubs, and off-path gates unchanged.

    Key insight:
    - The gate colors encode sequence position along the actual move path, not absolute coordinates; reversing the move reverses which gate gets which color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    source = _solid_rect_bbox(g, Color.ORANGE)
    target = _solid_rect_bbox(g, Color.MAROON)
    sr0, sc0, sr1, sc1 = source
    tr0, tc0, tr1, tc1 = target

    assert (sr1 - sr0, sc1 - sc0) == (tr1 - tr0, tc1 - tc0), "rectangles must have the same size"

    orientation, gates = _ordered_gate_positions(g, bg, source, target)

    out = g.copy()
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = Color.ORANGE

    if orientation == "horizontal":
        cross0, cross1 = sr0, sr1
    else:
        cross0, cross1 = sc0, sc1

    for idx, axis_pos in enumerate(gates):
        _paint_gate_opening(out, orientation, cross0, cross1, axis_pos, GATE_CYCLE[idx % 3])

    return out


def _next_branch() -> tuple[str, int, int]:
    global _GEN_CALL_COUNT
    if _GEN_CALL_COUNT < len(BRANCH_PROGRAM):
        branch = BRANCH_PROGRAM[_GEN_CALL_COUNT]
    else:
        branch = (
            random.choice(("horizontal", "vertical")),
            random.choice((-1, 1)),
            random.choice((2, 3, 4)),
        )
    _GEN_CALL_COUNT += 1
    return branch


def _sample_spaced_positions(candidates: list[int], count: int, min_gap: int = 2) -> list[int] | None:
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(200):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _pick_sparse_subset(candidates: list[int], max_count: int) -> list[int]:
    if not candidates or max_count <= 0:
        return []
    out: list[int] = []
    shuffled = candidates[:]
    random.shuffle(shuffled)
    target = random.randint(0, min(max_count, len(candidates)))
    for pos in shuffled:
        if len(out) >= target:
            break
        if all(abs(pos - prev) >= 2 for prev in out):
            out.append(int(pos))
    return sorted(out)


def _sample_segment_len(available: int) -> int:
    return random.randint(1, min(4, int(available)))


def _draw_gate_pair(
    grid: np.ndarray,
    orientation: str,
    cross0: int,
    cross1: int,
    axis_pos: int,
) -> None:
    if orientation == "horizontal":
        top_len = _sample_segment_len(cross0)
        bottom_len = _sample_segment_len(grid.shape[0] - (cross1 + 1))
        grid[cross0 - top_len : cross0, axis_pos] = Color.BLACK
        grid[cross1 + 1 : cross1 + 1 + bottom_len, axis_pos] = Color.BLACK
        return

    left_len = _sample_segment_len(cross0)
    right_len = _sample_segment_len(grid.shape[1] - (cross1 + 1))
    grid[axis_pos, cross0 - left_len : cross0] = Color.BLACK
    grid[axis_pos, cross1 + 1 : cross1 + 1 + right_len] = Color.BLACK


def _draw_stub(
    grid: np.ndarray,
    orientation: str,
    cross0: int,
    cross1: int,
    axis_pos: int,
    side: str,
) -> None:
    if orientation == "horizontal":
        if side == "near":
            top_len = _sample_segment_len(cross0)
            grid[cross0 - top_len : cross0, axis_pos] = Color.BLACK
        else:
            bottom_len = _sample_segment_len(grid.shape[0] - (cross1 + 1))
            grid[cross1 + 1 : cross1 + 1 + bottom_len, axis_pos] = Color.BLACK
        return

    if side == "near":
        left_len = _sample_segment_len(cross0)
        grid[axis_pos, cross0 - left_len : cross0] = Color.BLACK
    else:
        right_len = _sample_segment_len(grid.shape[1] - (cross1 + 1))
        grid[axis_pos, cross1 + 1 : cross1 + 1 + right_len] = Color.BLACK


def _render_layout(layout: dict) -> tuple[np.ndarray, np.ndarray]:
    bg = int(layout["bg"])
    h, w = layout["shape"]
    grid = np.full((h, w), bg, dtype=int)

    orientation = str(layout["orientation"])
    cross0 = int(layout["cross0"])
    cross1 = int(layout["cross1"])

    for pos in layout["extra_gate_positions"]:
        _draw_gate_pair(grid, orientation, cross0, cross1, int(pos))
    for pos in layout["true_gate_positions"]:
        _draw_gate_pair(grid, orientation, cross0, cross1, int(pos))
    for pos, side in layout["stub_positions"]:
        _draw_stub(grid, orientation, cross0, cross1, int(pos), str(side))

    rect_h = int(layout["rect_h"])
    rect_w = int(layout["rect_w"])
    sr0, sc0, *_ = layout["source_bbox"]
    tr0, tc0, *_ = layout["target_bbox"]
    paste(grid, np.full((rect_h, rect_w), Color.ORANGE, dtype=int), sr0, sc0)
    paste(grid, np.full((rect_h, rect_w), Color.MAROON, dtype=int), tr0, tc0)

    expected = grid.copy()
    sr0, sc0, sr1, sc1 = layout["source_bbox"]
    tr0, tc0, tr1, tc1 = layout["target_bbox"]
    expected[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg
    expected[tr0 : tr1 + 1, tc0 : tc1 + 1] = Color.ORANGE
    for idx, pos in enumerate(layout["true_gate_order"]):
        _paint_gate_opening(expected, orientation, cross0, cross1, int(pos), GATE_CYCLE[idx % 3])

    return grid, expected


def _strip_values(grid: np.ndarray, orientation: str, cross0: int, cross1: int, axis_pos: int) -> np.ndarray:
    if orientation == "horizontal":
        return grid[cross0 : cross1 + 1, axis_pos]
    return grid[axis_pos, cross0 : cross1 + 1]


def _validate_layout_semantics(inp: np.ndarray, out: np.ndarray, expected: np.ndarray, layout: dict) -> bool:
    if not np.array_equal(out, expected):
        return False

    orientation = str(layout["orientation"])
    cross0 = int(layout["cross0"])
    cross1 = int(layout["cross1"])
    bg = int(layout["bg"])
    sr0, sc0, sr1, sc1 = layout["source_bbox"]
    tr0, tc0, tr1, tc1 = layout["target_bbox"]

    if not np.all(out[sr0 : sr1 + 1, sc0 : sc1 + 1] == bg):
        return False
    if not np.all(out[tr0 : tr1 + 1, tc0 : tc1 + 1] == Color.ORANGE):
        return False

    for idx, pos in enumerate(layout["true_gate_order"]):
        if not np.all(_strip_values(out, orientation, cross0, cross1, int(pos)) == GATE_CYCLE[idx % 3]):
            return False
        if not np.all(_strip_values(inp, orientation, cross0, cross1, int(pos)) == bg):
            return False

    untouched = set(layout["gap_positions"]) - set(layout["true_gate_positions"])
    untouched |= set(layout["extra_gate_positions"])
    untouched |= {int(pos) for pos, _side in layout["stub_positions"]}
    for pos in untouched:
        if not np.all(_strip_values(out, orientation, cross0, cross1, int(pos)) == bg):
            return False

    return True


def _build_layout(branch: tuple[str, int, int]) -> dict | None:
    orientation, direction, gate_count = branch
    rect_h, rect_w = random.choice(RECT_SHAPES)
    bg = random.choice(BACKGROUND_CHOICES)

    if orientation == "horizontal":
        move_len = rect_w
        cross_len = rect_h
    else:
        move_len = rect_h
        cross_len = rect_w

    gap_len = random.randint(2 * gate_count + 1, 2 * gate_count + 4)
    outer_before = random.randint(2, 5)
    outer_after = random.randint(2, 5)
    cross_before = random.randint(2, 5)
    cross_after = random.randint(2, 5)

    axis_len = outer_before + move_len + gap_len + move_len + outer_after
    cross_axis_len = cross_before + cross_len + cross_after
    if axis_len > 30 or cross_axis_len > 30:
        return None

    if orientation == "horizontal":
        h, w = cross_axis_len, axis_len
        cross0 = cross_before
        cross1 = cross0 + rect_h - 1
        source_col = outer_before + (move_len + gap_len if direction < 0 else 0)
        target_col = outer_before + (0 if direction < 0 else move_len + gap_len)
        source_bbox = (cross0, source_col, cross1, source_col + rect_w - 1)
        target_bbox = (cross0, target_col, cross1, target_col + rect_w - 1)
        path_start = outer_before + move_len
    else:
        h, w = axis_len, cross_axis_len
        cross0 = cross_before
        cross1 = cross0 + rect_w - 1
        source_row = outer_before + (move_len + gap_len if direction < 0 else 0)
        target_row = outer_before + (0 if direction < 0 else move_len + gap_len)
        source_bbox = (source_row, cross0, source_row + rect_h - 1, cross1)
        target_bbox = (target_row, cross0, target_row + rect_h - 1, cross1)
        path_start = outer_before + move_len

    gap_positions = list(range(path_start, path_start + gap_len))
    gate_candidates = gap_positions[1:-1]
    true_gates = _sample_spaced_positions(gate_candidates, gate_count, min_gap=2)
    if true_gates is None:
        return None

    if direction < 0:
        true_gate_order = sorted(true_gates, reverse=True)
    else:
        true_gate_order = sorted(true_gates)

    left_candidates = list(range(0, outer_before))
    right_candidates = list(range(path_start + gap_len + move_len, axis_len))
    extra_gates = _pick_sparse_subset(left_candidates, 1) + _pick_sparse_subset(right_candidates, 1)

    stub_candidates = [pos for pos in gap_positions if pos not in true_gates]
    random.shuffle(stub_candidates)
    stub_count = 0
    if stub_candidates:
        stub_count = 1 if len(stub_candidates) == 1 else random.randint(1, min(2, len(stub_candidates)))
    stub_positions = sorted(stub_candidates[:stub_count])
    stub_sides = [random.choice(("near", "far")) for _ in stub_positions]

    return {
        "orientation": orientation,
        "direction": direction,
        "bg": bg,
        "shape": (h, w),
        "rect_h": rect_h,
        "rect_w": rect_w,
        "cross0": cross0,
        "cross1": cross1,
        "source_bbox": source_bbox,
        "target_bbox": target_bbox,
        "gap_positions": gap_positions,
        "true_gate_positions": true_gates,
        "true_gate_order": true_gate_order,
        "extra_gate_positions": extra_gates,
        "stub_positions": list(zip(stub_positions, stub_sides)),
    }


def _gen_attempt(branch: tuple[str, int, int]) -> np.ndarray | None:
    layout = _build_layout(branch)
    if layout is None:
        return None

    inp, expected = _render_layout(layout)
    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not _validate_layout_semantics(inp, actual, expected, layout):
        return None
    return inp


def generate_input() -> np.ndarray:
    branch = _next_branch()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input for branch {branch}")
