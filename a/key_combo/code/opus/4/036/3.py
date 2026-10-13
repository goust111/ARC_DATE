from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


BG = Color.BLACK
MARKER = Color.GREY
BRIDGE = Color.TEAL
PAIR_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON]

Rect = tuple[int, int, int, int]


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    if lo > hi:
        return None
    return lo, hi


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _solid_rect_from_component(grid: np.ndarray, component: dict) -> Rect:
    rect = tuple(map(int, component["bbox"]))
    r0, c0, r1, c1 = rect
    color = int(component["color"])
    assert int(component["area"]) == _rect_area(rect), "every foreground object must be a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be uniformly filled"
    return rect


def _classify_pair(rect_a: Rect, rect_b: Rect) -> tuple[str, Rect, Rect, tuple[int, int]]:
    row_overlap = _interval_overlap(rect_a[0], rect_a[2], rect_b[0], rect_b[2])
    col_overlap = _interval_overlap(rect_a[1], rect_a[3], rect_b[1], rect_b[3])

    horizontal = (
        row_overlap is not None
        and row_overlap[1] - row_overlap[0] + 1 >= 3
        and (rect_a[3] < rect_b[1] or rect_b[3] < rect_a[1])
    )
    vertical = (
        col_overlap is not None
        and col_overlap[1] - col_overlap[0] + 1 >= 3
        and (rect_a[2] < rect_b[0] or rect_b[2] < rect_a[0])
    )
    assert horizontal ^ vertical, "same-color rectangles must face in exactly one axis"

    if horizontal:
        left, right = (rect_a, rect_b) if rect_a[1] < rect_b[1] else (rect_b, rect_a)
        assert row_overlap is not None
        return "H", left, right, row_overlap

    top, bottom = (rect_a, rect_b) if rect_a[0] < rect_b[0] else (rect_b, rect_a)
    assert col_overlap is not None
    return "V", top, bottom, col_overlap


def _marker_for_pair(grid: np.ndarray, orientation: str, rect_a: Rect, rect_b: Rect, shared: tuple[int, int]) -> tuple[int, int]:
    if orientation == "H":
        r0, r1 = shared
        mr = r0 + (r1 - r0) // 2
        gap = grid[r0 : r1 + 1, rect_a[3] + 1 : rect_b[1]]
        coords = np.argwhere(gap == MARKER)
        assert coords.shape[0] == 1, "each horizontal gap must contain exactly one marker"
        rr, cc = map(int, coords[0])
        rr += r0
        cc += rect_a[3] + 1
        assert rr == mr, "horizontal marker must lie on the shared-span middle row"
        return rr, cc

    c0, c1 = shared
    mc = c0 + (c1 - c0) // 2
    gap = grid[rect_a[2] + 1 : rect_b[0], c0 : c1 + 1]
    coords = np.argwhere(gap == MARKER)
    assert coords.shape[0] == 1, "each vertical gap must contain exactly one marker"
    rr, cc = map(int, coords[0])
    rr += rect_a[2] + 1
    cc += c0
    assert cc == mc, "vertical marker must lie on the shared-span middle column"
    return rr, cc


def _slide_pair(rect_a: Rect, rect_b: Rect, orientation: str, marker: tuple[int, int]) -> tuple[Rect, Rect, int]:
    mr, mc = marker
    if orientation == "H":
        left_steps = int(mc - rect_a[3])
        right_steps = int(rect_b[1] - mc)
        assert left_steps >= 1 and right_steps >= 1
        steps = min(left_steps, right_steps)
        moved_a = (rect_a[0], rect_a[1] + steps, rect_a[2], rect_a[3] + steps)
        moved_b = (rect_b[0], rect_b[1] - steps, rect_b[2], rect_b[3] - steps)
        return moved_a, moved_b, steps

    top_steps = int(mr - rect_a[2])
    bottom_steps = int(rect_b[0] - mr)
    assert top_steps >= 1 and bottom_steps >= 1
    steps = min(top_steps, bottom_steps)
    moved_a = (rect_a[0] + steps, rect_a[1], rect_a[2] + steps, rect_a[3])
    moved_b = (rect_b[0] - steps, rect_b[1], rect_b[2] - steps, rect_b[3])
    return moved_a, moved_b, steps


def _paint_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _paint_bridge(grid: np.ndarray, orientation: str, rect_a: Rect, rect_b: Rect, color: int) -> None:
    if orientation == "H":
        shared = _interval_overlap(rect_a[0], rect_a[2], rect_b[0], rect_b[2])
        assert shared is not None
        gap = rect_b[1] - rect_a[3] - 1
        if gap <= 0:
            return
        r0 = shared[0] + 1
        r1 = shared[1] - 1
        if r0 <= r1:
            grid[r0 : r1 + 1, rect_a[3] + 1 : rect_b[1]] = int(color)
        return

    shared = _interval_overlap(rect_a[1], rect_a[3], rect_b[1], rect_b[3])
    assert shared is not None
    gap = rect_b[0] - rect_a[2] - 1
    if gap <= 0:
        return
    c0 = shared[0] + 1
    c1 = shared[1] - 1
    if c0 <= c1:
        grid[rect_a[2] + 1 : rect_b[0], c0 : c1 + 1] = int(color)


def _parse_pairs(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, list[Rect]] = {}

    for comp in comps:
        color = int(comp["color"])
        if color in {BG, MARKER, BRIDGE}:
            continue
        by_color.setdefault(color, []).append(_solid_rect_from_component(grid, comp))

    assert 2 <= len(by_color) <= 3, "expected 2 or 3 same-color rectangle pairs"
    pairs: list[dict] = []
    for color, rects in sorted(by_color.items()):
        assert len(rects) == 2, "each foreground color must appear in exactly two rectangles"
        orientation, rect_a, rect_b, shared = _classify_pair(rects[0], rects[1])
        marker = _marker_for_pair(grid, orientation, rect_a, rect_b, shared)
        pairs.append(
            {
                "color": color,
                "orientation": orientation,
                "rect_a": rect_a,
                "rect_b": rect_b,
                "shared": shared,
                "marker": marker,
            }
        )
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-color rectangle pair toward its grey brake marker, consume the marker, then bridge the remaining gap.

    Input structure:
    - The background is black.
    - There are 2 or 3 colors, and each color appears as exactly two solid axis-aligned rectangles.
    - Each same-color pair faces horizontally or vertically across a black gap that contains exactly one grey marker on the shared-span midline.

    Rule:
    - For each same-color pair, detect whether the rectangles face horizontally or vertically.
    - Slide both rectangles simultaneously toward each other, one cell per step, until the first front reaches the grey marker cell.
    - Remove the grey marker and keep the rectangles at those slid positions.
    - If a background gap still remains between the slid rectangles, fill that gap with a teal bridge using the shared perpendicular span eroded by one cell at both ends.

    Key insight:
    - The grey marker is a brake: it determines how many simultaneous slide steps happen before the pair freezes, so the bridge depends on the post-slide gap rather than the original spacing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    out = np.full_like(g, BG)

    for pair in _parse_pairs(g):
        moved_a, moved_b, _steps = _slide_pair(pair["rect_a"], pair["rect_b"], pair["orientation"], pair["marker"])
        _paint_rect(out, moved_a, pair["color"])
        _paint_rect(out, moved_b, pair["color"])
        _paint_bridge(out, pair["orientation"], moved_a, moved_b, BRIDGE)

    return out


def _rects_overlap(a: Rect, b: Rect, padding: int = 0) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _shape_range(low: int, high: int) -> int:
    return random.randint(int(low), int(high))


def _sample_horizontal_pair(color: int) -> dict:
    h1 = _shape_range(3, 4)
    h2 = _shape_range(3, 4)
    w1 = _shape_range(2, 5)
    w2 = _shape_range(2, 5)
    overlap = random.randint(3, min(h1, h2))

    extra1 = h1 - overlap
    extra2 = h2 - overlap
    top1 = random.randint(0, extra1)
    top2 = random.randint(0, extra2)
    bottom1 = extra1 - top1
    bottom2 = extra2 - top2

    shared_r0 = max(top1, top2)
    r0_1 = shared_r0 - top1
    r0_2 = shared_r0 - top2
    r1_1 = r0_1 + h1 - 1
    r1_2 = r0_2 + h2 - 1

    near = random.randint(1, 3)
    far = near + random.randint(2, 5)
    if random.random() < 0.5:
        left_d, right_d = near, far
    else:
        left_d, right_d = far, near

    c0_1 = 0
    c1_1 = c0_1 + w1 - 1
    marker_c = c1_1 + left_d
    c0_2 = marker_c + right_d
    c1_2 = c0_2 + w2 - 1
    marker_r = shared_r0 + (overlap - 1) // 2

    rect_a = (r0_1, c0_1, r1_1, c1_1)
    rect_b = (r0_2, c0_2, r1_2, c1_2)
    return {
        "color": color,
        "orientation": "H",
        "rect_a": rect_a,
        "rect_b": rect_b,
        "marker": (marker_r, marker_c),
        "bbox": (0, 0, max(r1_1, r1_2), c1_2),
    }


def _sample_vertical_pair(color: int) -> dict:
    h1 = _shape_range(2, 5)
    h2 = _shape_range(2, 5)
    w1 = _shape_range(3, 4)
    w2 = _shape_range(3, 4)
    overlap = random.randint(3, min(w1, w2))

    extra1 = w1 - overlap
    extra2 = w2 - overlap
    left1 = random.randint(0, extra1)
    left2 = random.randint(0, extra2)
    right1 = extra1 - left1
    right2 = extra2 - left2

    shared_c0 = max(left1, left2)
    c0_1 = shared_c0 - left1
    c0_2 = shared_c0 - left2
    c1_1 = c0_1 + w1 - 1
    c1_2 = c0_2 + w2 - 1

    near = random.randint(1, 3)
    far = near + random.randint(2, 5)
    if random.random() < 0.5:
        top_d, bottom_d = near, far
    else:
        top_d, bottom_d = far, near

    r0_1 = 0
    r1_1 = r0_1 + h1 - 1
    marker_r = r1_1 + top_d
    r0_2 = marker_r + bottom_d
    r1_2 = r0_2 + h2 - 1
    marker_c = shared_c0 + (overlap - 1) // 2

    rect_a = (r0_1, c0_1, r1_1, c1_1)
    rect_b = (r0_2, c0_2, r1_2, c1_2)
    return {
        "color": color,
        "orientation": "V",
        "rect_a": rect_a,
        "rect_b": rect_b,
        "marker": (marker_r, marker_c),
        "bbox": (0, 0, r1_2, max(c1_1, c1_2)),
    }


def _shift_pair(pair: dict, dr: int, dc: int) -> dict:
    def shift(rect: Rect) -> Rect:
        return rect[0] + dr, rect[1] + dc, rect[2] + dr, rect[3] + dc

    bbox = pair["bbox"]
    return {
        "color": pair["color"],
        "orientation": pair["orientation"],
        "rect_a": shift(pair["rect_a"]),
        "rect_b": shift(pair["rect_b"]),
        "marker": (pair["marker"][0] + dr, pair["marker"][1] + dc),
        "bbox": (bbox[0] + dr, bbox[1] + dc, bbox[2] + dr, bbox[3] + dc),
    }


def _all_pair_cells(pair: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rect in (pair["rect_a"], pair["rect_b"]):
        for r in range(rect[0], rect[2] + 1):
            for c in range(rect[1], rect[3] + 1):
                cells.add((r, c))
    cells.add(tuple(map(int, pair["marker"])))
    return cells


def _place_pairs(shape: tuple[int, int], pairs: list[dict]) -> list[dict] | None:
    H, W = shape
    placed: list[dict] = []
    occupied: list[Rect] = []

    for base in pairs:
        b0, b1, b2, b3 = base["bbox"]
        h = b2 - b0 + 1
        w = b3 - b1 + 1
        candidates = []
        for _ in range(120):
            dr = random.randint(0, H - h)
            dc = random.randint(0, W - w)
            shifted = _shift_pair(base, dr - b0, dc - b1)
            sb = shifted["bbox"]
            if any(_rects_overlap(sb, other, padding=1) for other in occupied):
                continue
            candidates.append(shifted)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        placed.append(chosen)
        occupied.append(chosen["bbox"])

    all_cells = set()
    for pair in placed:
        cells = _all_pair_cells(pair)
        if all_cells & cells:
            return None
        all_cells |= cells
    return placed


def _build_input(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for pair in pairs:
        _paint_rect(grid, pair["rect_a"], pair["color"])
        _paint_rect(grid, pair["rect_b"], pair["color"])
        mr, mc = pair["marker"]
        grid[mr, mc] = MARKER
    return grid


def _build_expected_output(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pairs:
        moved_a, moved_b, _steps = _slide_pair(pair["rect_a"], pair["rect_b"], pair["orientation"], pair["marker"])
        _paint_rect(out, moved_a, pair["color"])
        _paint_rect(out, moved_b, pair["color"])
        _paint_bridge(out, pair["orientation"], moved_a, moved_b, BRIDGE)
    return out


def _build_no_slide_bridge(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pairs:
        _paint_rect(out, pair["rect_a"], pair["color"])
        _paint_rect(out, pair["rect_b"], pair["color"])
        _paint_bridge(out, pair["orientation"], pair["rect_a"], pair["rect_b"], BRIDGE)
    return out


def _build_adjacent_slide(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for pair in pairs:
        if pair["orientation"] == "H":
            gap = pair["rect_b"][1] - pair["rect_a"][3] - 1
            steps = gap // 2
            moved_a = (pair["rect_a"][0], pair["rect_a"][1] + steps, pair["rect_a"][2], pair["rect_a"][3] + steps)
            moved_b = (pair["rect_b"][0], pair["rect_b"][1] - steps, pair["rect_b"][2], pair["rect_b"][3] - steps)
        else:
            gap = pair["rect_b"][0] - pair["rect_a"][2] - 1
            steps = gap // 2
            moved_a = (pair["rect_a"][0] + steps, pair["rect_a"][1], pair["rect_a"][2] + steps, pair["rect_a"][3])
            moved_b = (pair["rect_b"][0] - steps, pair["rect_b"][1], pair["rect_b"][2] - steps, pair["rect_b"][3])
        _paint_rect(out, moved_a, pair["color"])
        _paint_rect(out, moved_b, pair["color"])
        _paint_bridge(out, pair["orientation"], moved_a, moved_b, BRIDGE)
    return out


def _validate_case(grid: np.ndarray, pairs: list[dict]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    expected = _build_expected_output(grid.shape, pairs)
    if not np.array_equal(out, expected):
        return False

    if np.any(out == MARKER):
        return False

    moved_any = False
    for pair in pairs:
        moved_a, moved_b, steps = _slide_pair(pair["rect_a"], pair["rect_b"], pair["orientation"], pair["marker"])
        if steps <= 0:
            return False
        moved_any |= moved_a != pair["rect_a"] or moved_b != pair["rect_b"]
        if pair["orientation"] == "H":
            gap0 = pair["rect_b"][1] - pair["rect_a"][3] - 1
            gap1 = moved_b[1] - moved_a[3] - 1
        else:
            gap0 = pair["rect_b"][0] - pair["rect_a"][2] - 1
            gap1 = moved_b[0] - moved_a[2] - 1
        if gap1 <= 0 or gap1 >= gap0:
            return False
    if not moved_any:
        return False

    if np.array_equal(out, _build_no_slide_bridge(grid.shape, pairs)):
        return False
    if np.array_equal(out, _build_adjacent_slide(grid.shape, pairs)):
        return False

    reparsed = _parse_pairs(grid)
    if len(reparsed) != len(pairs):
        return False
    return True


def _generate_case() -> tuple[np.ndarray, list[dict]]:
    H = random.randint(15, 24)
    W = random.randint(18, 28)
    n_pairs = random.randint(2, 3)
    colors = random.sample(PAIR_COLORS, n_pairs)

    if n_pairs == 2:
        orientations = random.choice([["H", "H"], ["H", "V"], ["V", "V"]])
    else:
        orientations = random.choice(
            [["H", "H", "V"], ["H", "V", "V"], ["H", "H", "H"], ["V", "V", "V"]]
        )
    random.shuffle(orientations)

    base_pairs = []
    for color, orientation in zip(colors, orientations):
        pair = _sample_horizontal_pair(color) if orientation == "H" else _sample_vertical_pair(color)
        base_pairs.append(pair)

    placed = _place_pairs((H, W), base_pairs)
    if placed is None:
        raise ValueError("placement failed")

    grid = _build_input((H, W), placed)
    return grid, placed


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, pairs = _generate_case()
        except ValueError:
            continue
        if _validate_case(grid, pairs):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
