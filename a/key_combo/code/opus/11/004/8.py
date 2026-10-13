from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
MAROON = 9

INPUT_COLORS = (BLUE, RED, GREEN)
CORNER_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
ADJACENT_CORNER_PAIRS = (
    ("top_left", "top_right"),
    ("top_left", "bottom_left"),
    ("top_right", "bottom_right"),
    ("bottom_left", "bottom_right"),
)

# Missing local cell -> target outer corner.
PRESENT_TO_CORNER = {
    frozenset({(0, 1), (1, 0), (1, 1)}): "top_left",
    frozenset({(0, 0), (1, 0), (1, 1)}): "top_right",
    frozenset({(0, 0), (0, 1), (1, 1)}): "bottom_left",
    frozenset({(0, 0), (0, 1), (1, 0)}): "bottom_right",
}
CORNER_TO_PRESENT = {corner: coords for coords, corner in PRESENT_TO_CORNER.items()}

GRID_SIZES = (11, 13, 15, 17)
COUNT_CHOICES = (1, 2, 2, 3, 3)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
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
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _corner_box(n: int, corner: str) -> tuple[int, int, int, int]:
    arm = (n + 1) // 2
    if corner == "top_left":
        return 0, 0, arm - 1, arm - 1
    if corner == "top_right":
        return 0, n - arm, arm - 1, n - 1
    if corner == "bottom_left":
        return n - arm, 0, n - 1, arm - 1
    assert corner == "bottom_right"
    return n - arm, n - arm, n - 1, n - 1


def _paint_role_box(out: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) and c in (c0, c1):
                out[r, c] = ORANGE
            elif r in (r0, r1) or c in (c0, c1):
                out[r, c] = GREY
            else:
                out[r, c] = MAROON


def _paint_corner_arm(out: np.ndarray, box: tuple[int, int, int, int], corner: str, color: int) -> None:
    r0, c0, r1, c1 = box
    if corner == "top_left":
        out[r0, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
    elif corner == "top_right":
        out[r0, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c1] = color
    elif corner == "bottom_left":
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
    else:
        assert corner == "bottom_right"
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c1] = color


def _solve_from_corner_colors(n: int, corner_to_color: dict[str, int], order: tuple[str, ...] = CORNER_ORDER) -> np.ndarray:
    out = np.zeros((n, n), dtype=int)
    for corner in order:
        color = corner_to_color.get(corner)
        if color is None:
            continue
        box = _corner_box(n, corner)
        _paint_role_box(out, box)
        _paint_corner_arm(out, box, corner, int(color))
    return out


def _render_input(n: int, placements: dict[str, tuple[int, int]], corner_to_color: dict[str, int]) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    for corner, (r0, c0) in placements.items():
        color = int(corner_to_color[corner])
        for dr, dc in CORNER_TO_PRESENT[corner]:
            grid[r0 + dr, c0 + dc] = color
    return grid


def _extract_markers(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be 2D"
    h, w = g.shape
    assert h == w and h % 2 == 1 and 11 <= h <= 17, "input must be an odd square of side 11..17"
    assert set(map(int, np.unique(g))).issubset({BLACK, *INPUT_COLORS}), "unexpected colors"

    comps = _find_components(g, bg=BLACK)
    assert 1 <= len(comps) <= 3, "expected 1..3 markers"

    corner_to_color: dict[str, int] = {}
    seen_colors: set[int] = set()
    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        assert color in INPUT_COLORS and color not in seen_colors, "marker colors must be distinct red/green/blue"
        seen_colors.add(color)
        assert comp["area"] == 3, "each marker must contain exactly 3 pixels"
        assert r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1, "markers must stay away from the border"
        assert (r1 - r0, c1 - c0) == (1, 1), "each marker must fit a 2x2 box"

        local = frozenset((r - r0, c - c0) for r, c in coords)
        assert local in PRESENT_TO_CORNER, "marker must be a 3-pixel L"
        corner = PRESENT_TO_CORNER[local]
        assert corner not in corner_to_color, "missing-corner directions must be distinct"
        corner_to_color[corner] = color

    return corner_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each 3-pixel L marker to its missing-corner outer box, classify that box by role, then overwrite the corner arm in the marker color.

    Input structure:
    - The grid is an odd black square of side 11..17.
    - It contains 1..3 monochrome 3-pixel L markers in red, green, or blue.
    - Each marker fits inside a 2x2 box with exactly one missing local corner, and no two markers share the same missing-corner direction.

    Rule:
    - Read each marker's missing local corner and map it to the matching outer grid corner.
    - Clear the whole grid to black.
    - For each selected outer corner, take the HxH corner box where H = (side + 1) / 2, color its four box corners orange, its non-corner border gray, and its strict interior maroon.
    - Then overwrite that box's corner L arm with the marker's original color.
    - If several corner boxes overlap, process them in fixed order: top-left, top-right, bottom-left, bottom-right; later painting overwrites earlier painting.

    Key insight:
    - The decisive cue is the missing cell of each tiny L: it chooses which large corner box appears, while the marker's own color only survives on the final overwritten arm pixels.
    """
    g = np.asarray(grid, dtype=int)
    corner_to_color = _extract_markers(g)
    return _solve_from_corner_colors(g.shape[0], corner_to_color)


def _corners_for_count(count: int, rng: random.Random) -> list[str]:
    if count == 1:
        return [rng.choice(CORNER_ORDER)]
    if count == 2:
        pair = rng.choice(ADJACENT_CORNER_PAIRS)
        return list(pair)
    assert count == 3
    return rng.sample(list(CORNER_ORDER), 3)


def _boxes_have_padding(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    return ar0 + 2 < br0 - 1 or br0 + 2 < ar0 - 1 or ac0 + 2 < bc0 - 1 or bc0 + 2 < ac0 - 1


def _target_box_contains_marker(n: int, corner: str, pos: tuple[int, int]) -> bool:
    r0, c0 = pos
    br0, bc0, br1, bc1 = _corner_box(n, corner)
    return br0 <= r0 and r0 + 1 <= br1 and bc0 <= c0 and c0 + 1 <= bc1


def _choose_positions(n: int, corners: list[str], rng: random.Random) -> dict[str, tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, n - 2) for c in range(1, n - 2)]
    rng.shuffle(candidates)
    corners = list(corners)
    rng.shuffle(corners)

    placements: dict[str, tuple[int, int]] = {}
    need_misleading = 1
    for index, corner in enumerate(corners):
        remaining = len(corners) - index - 1
        good: list[tuple[int, int]] = []
        fallback: list[tuple[int, int]] = []
        for pos in candidates:
            if any(not _boxes_have_padding(pos, other) for other in placements.values()):
                continue
            if not _target_box_contains_marker(n, corner, pos):
                good.append(pos)
            else:
                fallback.append(pos)
        if need_misleading > remaining:
            pool = good
        else:
            pool = good + fallback
        if not pool:
            return None
        pos = rng.choice(pool)
        placements[corner] = pos
        if pos in good:
            need_misleading = 0
        candidates = [cand for cand in candidates if cand != pos]

    if need_misleading:
        return None
    return placements


def _direct_expected(n: int, corner_to_color: dict[str, int], order: tuple[str, ...]) -> np.ndarray:
    out = np.zeros((n, n), dtype=int)
    arm = (n + 1) // 2
    for corner in order:
        if corner not in corner_to_color:
            continue
        color = int(corner_to_color[corner])
        if corner == "top_left":
            rr = range(0, arm)
            cc = range(0, arm)
            top_row = 0
            side_col = 0
        elif corner == "top_right":
            rr = range(0, arm)
            cc = range(n - arm, n)
            top_row = 0
            side_col = n - 1
        elif corner == "bottom_left":
            rr = range(n - arm, n)
            cc = range(0, arm)
            top_row = n - 1
            side_col = 0
        else:
            rr = range(n - arm, n)
            cc = range(n - arm, n)
            top_row = n - 1
            side_col = n - 1

        r_list = list(rr)
        c_list = list(cc)
        for r in r_list:
            for c in c_list:
                if r in (r_list[0], r_list[-1]) and c in (c_list[0], c_list[-1]):
                    out[r, c] = ORANGE
                elif r in (r_list[0], r_list[-1]) or c in (c_list[0], c_list[-1]):
                    out[r, c] = GREY
                else:
                    out[r, c] = MAROON
        out[top_row, c_list[0] : c_list[-1] + 1] = color
        out[r_list[0] : r_list[-1] + 1, side_col] = color
    return out


def _has_all_role_colors(out: np.ndarray) -> bool:
    present = set(map(int, np.unique(out)))
    return {ORANGE, GREY, MAROON}.issubset(present)


def _input_semantics_hold(inp: np.ndarray, corners: list[str], corner_to_color: dict[str, int]) -> bool:
    try:
        parsed = _extract_markers(inp)
    except AssertionError:
        return False
    return parsed == {corner: int(corner_to_color[corner]) for corner in corners}


def _order_is_observable(n: int, corner_to_color: dict[str, int]) -> bool:
    chosen_order = tuple(corner for corner in CORNER_ORDER if corner in corner_to_color)
    if len(chosen_order) <= 1:
        return True
    forward = _direct_expected(n, corner_to_color, chosen_order)
    reverse = _direct_expected(n, corner_to_color, tuple(reversed(chosen_order)))
    return not np.array_equal(forward, reverse)


def _overwrite_is_necessary(n: int, corner_to_color: dict[str, int], out: np.ndarray) -> bool:
    role_only = np.zeros_like(out)
    for corner in CORNER_ORDER:
        if corner in corner_to_color:
            _paint_role_box(role_only, _corner_box(n, corner))
    return not np.array_equal(role_only, out)


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(GRID_SIZES)
    count = rng.choice(COUNT_CHOICES)
    corners = _corners_for_count(int(count), rng)

    colors = rng.sample(list(INPUT_COLORS), len(corners))
    rng.shuffle(corners)
    corner_to_color = {corner: color for corner, color in zip(corners, colors)}

    placements = _choose_positions(n, corners, rng)
    if placements is None:
        return None

    inp = _render_input(n, placements, corner_to_color)
    expected = _direct_expected(n, corner_to_color, CORNER_ORDER)

    if not _input_semantics_hold(inp, corners, corner_to_color):
        return None
    if not _has_all_role_colors(expected):
        return None
    if not _order_is_observable(n, corner_to_color):
        return None
    if not _overwrite_is_necessary(n, corner_to_color, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
