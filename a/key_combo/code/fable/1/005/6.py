from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, in_bounds, neighbors4  # noqa: E402


BG = 0
MARKER = 1
RED = 2
RECT_COLOR = 4
PINK = 6
ORANGE = 7
CYAN = 8

DIR_ORDER = ("up", "right", "down", "left")
DIR_VECTORS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

GRID_SIZE_MIN = 13
GRID_SIZE_MAX = 24
RECT_SIZE_MIN = 5
RECT_SIZE_MAX = 11
MARKER_COUNT_MIN = 4
MARKER_COUNT_MAX = 6
MAX_RETRIES = 500

PATTERN_SPECS = (
    {"fronts": ("up", "down", "left", "right"), "n_clipped": 0},
    {"fronts": ("up", "left", "right"), "n_clipped": 1},
    {"fronts": ("down", "left", "right"), "n_clipped": 1},
    {"fronts": ("up", "down", "left"), "n_clipped": 1},
)
_pattern_bag: list[int] = []


def _take_pattern() -> dict[str, object]:
    if not _pattern_bag:
        _pattern_bag.extend(range(len(PATTERN_SPECS)))
        random.shuffle(_pattern_bag)
    return PATTERN_SPECS[_pattern_bag.pop()]


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [v for v in range(int(lo), int(hi) + 1) if v % 2 == 1]


def _rectangle_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, bbox)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _is_hollow_rectangle(grid: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return False
    border_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != border_area:
        return False

    color = int(comp["color"])
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(region[border] == color) and np.all(region[~border] == int(bg)))


def _parse_instance(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert comps, "expected foreground objects"

    rect_candidates = [comp for comp in comps if _is_hollow_rectangle(g, comp, bg)]
    assert len(rect_candidates) == 1, "expected exactly one hollow odd-sized rectangle"
    rect = rect_candidates[0]

    rect_mask = np.zeros(g.shape, dtype=bool)
    rect_coords = rect["coords"]
    rect_mask[rect_coords[:, 0], rect_coords[:, 1]] = True
    r0, c0, r1, c1 = map(int, rect["bbox"])
    center = ((r0 + r1) // 2, (c0 + c1) // 2)

    markers: list[tuple[int, int]] = []
    for comp in comps:
        if comp is rect:
            continue
        assert int(comp["area"]) == 1, "all non-rectangle objects must be single pixels"
        assert int(comp["color"]) == MARKER, "all markers must be blue single pixels"
        mr, mc = map(int, comp["coords"][0])
        dr = center[0] - mr
        dc = center[1] - mc
        assert (dr, dc) != (0, 0), "marker cannot sit at the rectangle center"
        assert abs(dr) != abs(dc), "marker cannot lie on a center diagonal"
        markers.append((mr, mc))

    assert markers, "expected at least one marker"
    markers.sort()
    return {
        "grid": g,
        "rect_bbox": (r0, c0, r1, c1),
        "rect_mask": rect_mask,
        "center": center,
        "markers": markers,
    }


def _front_direction(
    marker: tuple[int, int],
    ref: tuple[int, int],
    *,
    axis_mode: str = "major",
    row_if_tie: bool = False,
) -> str:
    mr, mc = marker
    rr, rc = ref
    dr = int(rr) - int(mr)
    dc = int(rc) - int(mc)
    assert (dr, dc) != (0, 0), "reference must differ from marker"

    abs_dr = abs(dr)
    abs_dc = abs(dc)
    if axis_mode == "major":
        use_row = abs_dr > abs_dc or (row_if_tie and abs_dr == abs_dc)
    elif axis_mode == "minor":
        use_row = abs_dr < abs_dc or (row_if_tie and abs_dr == abs_dc)
    else:
        raise ValueError(f"unknown axis mode: {axis_mode}")

    if use_row:
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _stamped_arm_colors(front: str, *, swap_fb: bool = False, swap_handed: bool = False) -> list[tuple[str, int]]:
    idx = DIR_ORDER.index(front)
    front_color = CYAN if swap_fb else RED
    back_color = RED if swap_fb else CYAN
    cw_color = PINK if swap_handed else ORANGE
    ccw_color = ORANGE if swap_handed else PINK
    return [
        (DIR_ORDER[idx], front_color),
        (DIR_ORDER[(idx + 2) % 4], back_color),
        (DIR_ORDER[(idx + 1) % 4], cw_color),
        (DIR_ORDER[(idx - 1) % 4], ccw_color),
    ]


def _render_output(
    grid: np.ndarray,
    center: tuple[int, int],
    markers: list[tuple[int, int]],
    rect_mask: np.ndarray,
    *,
    ref: tuple[int, int] | None = None,
    axis_mode: str = "major",
    row_if_tie: bool = False,
    absolute_front: str | None = None,
    swap_fb: bool = False,
    swap_handed: bool = False,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    structural = rect_mask.copy()
    for mr, mc in markers:
        structural[mr, mc] = True

    pending: dict[tuple[int, int], int] = {}
    ref_point = center if ref is None else ref

    for marker in markers:
        if absolute_front is None:
            front = _front_direction(marker, ref_point, axis_mode=axis_mode, row_if_tie=row_if_tie)
        else:
            front = absolute_front

        for direction, color in _stamped_arm_colors(front, swap_fb=swap_fb, swap_handed=swap_handed):
            dr, dc = DIR_VECTORS[direction]
            rr = marker[0] + dr
            cc = marker[1] + dc
            if not in_bounds(out.shape, rr, cc):
                continue
            if structural[rr, cc]:
                continue
            cell = (int(rr), int(cc))
            if cell in pending:
                assert pending[cell] == int(color), "overlapping stamps cannot disagree on color"
            else:
                pending[cell] = int(color)

    for (rr, cc), color in pending.items():
        out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a center-pointing colored cross around every blue marker, using the unique hollow rectangle's center as the hidden compass reference.

    Input structure:
    - Black background with exactly one hollow odd-sized rectangle whose geometric center is a single cell.
    - Several isolated blue marker pixels are scattered away from one another and do not lie on a center diagonal.
    - The rectangle itself stays visible as an outline and is not explicitly labeled as the reference point.

    Rule:
    - For each marker, compare its row and column offsets to the rectangle center; the larger absolute offset chooses the main direction from the marker toward that center.
    - Paint the four orthogonal neighbor cells as a clipped cross: the arm pointing toward the center is red, the opposite arm is cyan, the clockwise side from the red arm is orange, and the counterclockwise side is pink.
    - Keep the marker pixel and the hollow rectangle unchanged.

    Key insight:
    - The color layout is not tied to absolute north/south/east/west; each cross is the same directional stamp rotated so its red arm faces the implicit center hidden inside the hollow rectangle.
    """
    parsed = _parse_instance(np.asarray(grid, dtype=int))
    return _render_output(
        parsed["grid"],
        parsed["center"],
        parsed["markers"],
        parsed["rect_mask"],
    )


def _marker_reserved_cells(shape: tuple[int, int], marker: tuple[int, int]) -> set[tuple[int, int]]:
    cells = {marker}
    for nr, nc in neighbors4(*marker):
        nr = int(nr)
        nc = int(nc)
        if in_bounds(shape, nr, nc):
            cells.add((nr, nc))
    return cells


def _valid_marker_candidates(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    rect_mask: np.ndarray,
    occupied: set[tuple[int, int]],
    front: str,
    clipped: bool,
) -> list[tuple[tuple[int, int], set[tuple[int, int]]]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

    for mr in range(h):
        for mc in range(w):
            if r0 <= mr <= r1 and c0 <= mc <= c1:
                continue
            marker = (mr, mc)
            if marker in occupied:
                continue
            if abs(center[0] - mr) == abs(center[1] - mc):
                continue
            if _front_direction(marker, center) != front:
                continue

            reserved = _marker_reserved_cells(shape, marker)
            if reserved & occupied:
                continue
            if any(rect_mask[rr, cc] for rr, cc in reserved if in_bounds(shape, rr, cc)):
                continue

            touches_edge = len(reserved) < 5
            if clipped != touches_edge:
                continue
            candidates.append((marker, reserved))

    random.shuffle(candidates)
    return candidates


def _place_markers(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    rect_mask: np.ndarray,
    requirements: list[tuple[str, bool]],
) -> list[tuple[int, int]] | None:
    base_occupied = {
        (int(r), int(c))
        for r, c in np.argwhere(rect_mask)
    }
    reqs = requirements[:]
    random.shuffle(reqs)
    reqs.sort(key=lambda item: item[1], reverse=True)

    for _ in range(80):
        used = set(base_occupied)
        current: list[tuple[int, int]] = []
        ok = True
        for front, clipped in reqs:
            candidates = _valid_marker_candidates(shape, bbox, center, rect_mask, used, front, clipped)
            if not candidates:
                ok = False
                break
            marker, reserved = random.choice(candidates[:40])
            current.append(marker)
            used |= reserved
        if ok:
            return current
    return None


def _build_input(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    rect_mask = _rectangle_border_mask(shape, bbox)
    grid[rect_mask] = RECT_COLOR
    for mr, mc in markers:
        grid[mr, mc] = MARKER
    return grid


def _wrong_heuristics_distinct(inp: np.ndarray, parsed: dict[str, object], expected: np.ndarray) -> bool:
    r0, c0, _r1, _c1 = parsed["rect_bbox"]
    center = parsed["center"]
    markers = parsed["markers"]
    rect_mask = parsed["rect_mask"]

    wrong_outputs = [
        _render_output(inp, center, markers, rect_mask, absolute_front="up"),
        _render_output(inp, center, markers, rect_mask, ref=(r0, c0), row_if_tie=True),
        _render_output(inp, center, markers, rect_mask, axis_mode="minor"),
        _render_output(inp, center, markers, rect_mask, swap_fb=True),
        _render_output(inp, center, markers, rect_mask, swap_handed=True),
    ]
    return all(not np.array_equal(expected, wrong) for wrong in wrong_outputs)


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    requirements: list[tuple[str, bool]],
) -> bool:
    try:
        parsed = _parse_instance(inp)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    fronts = {_front_direction(marker, parsed["center"]) for marker in parsed["markers"]}
    if not (fronts & {"up", "down"} and fronts & {"left", "right"}):
        return False

    required_fronts = {front for front, _clipped in requirements}
    if not required_fronts.issubset(fronts):
        return False

    clipped_required = sum(1 for _front, clipped in requirements if clipped)
    clipped_seen = sum(1 for marker in parsed["markers"] if len(_marker_reserved_cells(inp.shape, marker)) < 5)
    if clipped_seen != clipped_required:
        return False

    changed_cells = int(np.count_nonzero(inp != expected))
    if changed_cells < 2 * len(parsed["markers"]):
        return False

    if not _wrong_heuristics_distinct(inp, parsed, expected):
        return False

    return True


def _gen_attempt(pattern: dict[str, object]) -> np.ndarray | None:
    h = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    w = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)

    rect_h_choices = _odd_choices(RECT_SIZE_MIN, min(RECT_SIZE_MAX, h - 4))
    rect_w_choices = _odd_choices(RECT_SIZE_MIN, min(RECT_SIZE_MAX, w - 4))
    if not rect_h_choices or not rect_w_choices:
        return None

    rect_h = random.choice(rect_h_choices)
    rect_w = random.choice(rect_w_choices)
    r0 = random.randint(2, h - rect_h - 2)
    c0 = random.randint(2, w - rect_w - 2)
    bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    rect_mask = _rectangle_border_mask((h, w), bbox)

    base_fronts = list(pattern["fronts"])
    target_count = random.randint(max(MARKER_COUNT_MIN, len(base_fronts)), MARKER_COUNT_MAX)
    extras = [random.choice(DIR_ORDER) for _ in range(target_count - len(base_fronts))]
    fronts = base_fronts + extras

    clipped_count = int(pattern["n_clipped"])
    clipped_flags = [False] * len(fronts)
    if clipped_count:
        for idx in random.sample(range(len(fronts)), clipped_count):
            clipped_flags[idx] = True

    requirements = list(zip(fronts, clipped_flags))
    markers = _place_markers((h, w), bbox, center, rect_mask, requirements)
    if markers is None:
        return None

    inp = _build_input((h, w), bbox, markers)
    expected = _render_output(inp, center, markers, rect_mask)
    if _validate_instance(inp, expected, requirements):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        pattern = _take_pattern()
        grid = _gen_attempt(pattern)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compass-cross input")
