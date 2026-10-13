from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


BLACK = 0
BLUE = 1
BADGE_ORDER = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
INPUT_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
MAX_TARGET_BOUNCES = 5
MAX_GENERATION_ATTEMPTS = 600

_GEN_COUNTER = 0
_GEN_TARGET_ORDER = list(range(MAX_TARGET_BOUNCES + 1))
random.shuffle(_GEN_TARGET_ORDER)


def _is_arrow_component(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 3:
        return False
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1


def _arrow_start_and_direction(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    assert _is_arrow_component(coords), "expected a 3-cell 2x2-corner arrow"
    pts = set(coords)
    r0 = min(r for r, _c in coords)
    r1 = max(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    c1 = max(c for _r, c in coords)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    missing = [corner for corner in corners if corner not in pts]
    assert len(missing) == 1
    miss_r, miss_c = missing[0]
    dr = 1 if miss_r == r0 else -1
    dc = 1 if miss_c == c0 else -1
    start = (r1 if miss_r == r0 else r0, c1 if miss_c == c0 else c0)
    return start, (dr, dc)


def _arrow_cells(start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r, c), (r - dr, c), (r, c - dc)]


def _parse_instance(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int, int]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    fg_comps = [comp for comp in comps if int(comp["color"]) != bg]
    assert len(fg_comps) == 2, "expected exactly one arrow component and one absorber"

    arrow_comp = None
    absorber_comp = None
    for comp in fg_comps:
        area = int(comp["area"])
        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        if area == 3 and _is_arrow_component(coords):
            assert arrow_comp is None, "expected a unique arrow"
            arrow_comp = comp
        elif area == 1:
            assert absorber_comp is None, "expected a unique absorber"
            absorber_comp = comp
        else:
            raise AssertionError("unexpected foreground component")

    assert arrow_comp is not None and absorber_comp is not None
    assert int(arrow_comp["color"]) != int(absorber_comp["color"]), "absorber must be a different color"

    arrow_coords = [tuple(map(int, rc)) for rc in arrow_comp["coords"]]
    absorber = tuple(map(int, absorber_comp["coords"][0]))
    start, direction = _arrow_start_and_direction(arrow_coords)
    return start, direction, absorber, int(arrow_comp["color"]), int(absorber_comp["color"])


def _first_hit_counts(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> dict[int, list[tuple[int, int]]]:
    h, w = map(int, shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    bounce_count = 0
    seen_states = {(r, c, dr, dc)}
    first_hits: dict[tuple[int, int], int] = {}

    for _ in range(4 * h * w + 8):
        hit_row = not (0 <= r + dr < h)
        hit_col = not (0 <= c + dc < w)
        if hit_row or hit_col:
            if hit_row:
                dr = -dr
            if hit_col:
                dc = -dc
            bounce_count += 1

        r += dr
        c += dc
        first_hits.setdefault((r, c), bounce_count)

        state = (r, c, dr, dc)
        if state in seen_states:
            break
        seen_states.add(state)

    by_count: dict[int, list[tuple[int, int]]] = {}
    for key, bounce_count in first_hits.items():
        cell = (int(key[0]), int(key[1]))
        by_count.setdefault(int(bounce_count), []).append(cell)
    return by_count


def _count_bounces(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    absorber: tuple[int, int],
) -> int:
    h, w = map(int, shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    bounce_count = 0
    seen_states = {(r, c, dr, dc)}

    for _ in range(4 * h * w + 8):
        hit_row = not (0 <= r + dr < h)
        hit_col = not (0 <= c + dc < w)
        if hit_row or hit_col:
            if hit_row:
                dr = -dr
            if hit_col:
                dc = -dc
            bounce_count += 1

        r += dr
        c += dc
        if (r, c) == absorber:
            return bounce_count

        state = (r, c, dr, dc)
        if state in seen_states:
            break
        seen_states.add(state)

    raise AssertionError("absorber is not on the arrow's reflected path")


def _encode_badge(bounce_count: int) -> np.ndarray:
    assert 0 <= int(bounce_count) <= len(BADGE_ORDER)
    out = np.zeros((3, 3), dtype=int)
    for r, c in BADGE_ORDER[: int(bounce_count)]:
        out[r, c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode the number of billiard-style boundary rebounds before a diagonal arrow ray reaches its absorber.

    Input structure:
    - The input is a black-background grid containing one 3-pixel L-shaped arrow and one singleton absorber pixel of a different color.
    - The arrow occupies a 2x2 box with one missing corner; that missing corner determines the diagonal firing direction.

    Rule:
    - Start the ray from the arrow's right-angle corner and fire it diagonally away from the missing corner.
    - Whenever the next diagonal step would leave the grid, reflect on the touched boundary; touching a corner flips both axes and counts as one bounce event.
    - Stop when the ray reaches the absorber pixel, count the boundary bounces before that hit, and fill that many blue cells in a 3x3 black badge using the fixed order: upper-left, upper-right, center, lower-left, lower-right.

    Key insight:
    - The visible input only specifies the start state and the absorber; the output discards the whole path shape and keeps only the implicit rebound count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start, direction, absorber, _arrow_color, _absorber_color = _parse_instance(g)
    bounce_count = _count_bounces(g.shape, start, direction, absorber)
    assert bounce_count <= MAX_TARGET_BOUNCES, "puzzle instances only allow up to five bounces"
    return _encode_badge(bounce_count)


def _next_target_bounces() -> int:
    global _GEN_COUNTER
    idx = _GEN_COUNTER % len(_GEN_TARGET_ORDER)
    if idx == 0 and _GEN_COUNTER > 0:
        random.shuffle(_GEN_TARGET_ORDER)
    target = int(_GEN_TARGET_ORDER[idx])
    _GEN_COUNTER += 1
    return target


def _sample_shape(target_bounces: int) -> tuple[int, int]:
    min_side = 6 if target_bounces <= 2 else 8
    max_side = 18 if target_bounces <= 2 else 26
    return random.randint(min_side, max_side), random.randint(min_side, max_side)


def _sample_start(shape: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    dr, dc = direction
    row_lo = 1 if dr == 1 else 0
    row_hi = h - 1 if dr == 1 else h - 2
    col_lo = 1 if dc == 1 else 0
    col_hi = w - 1 if dc == 1 else w - 2
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _build_grid(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    absorber: tuple[int, int],
    arrow_color: int,
    absorber_color: int,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in _arrow_cells(start, direction):
        grid[r, c] = int(arrow_color)
    grid[absorber] = int(absorber_color)
    return grid


def _validate_generated(grid: np.ndarray, expected_bounces: int) -> bool:
    try:
        start, direction, absorber, arrow_color, absorber_color = _parse_instance(grid)
        if arrow_color == absorber_color:
            return False

        arrow_cells = set(_arrow_cells(start, direction))
        if absorber in arrow_cells:
            return False

        reachable = _first_hit_counts(grid.shape, start, direction)
        if absorber not in reachable.get(expected_bounces, []):
            return False
        if _count_bounces(grid.shape, start, direction, absorber) != expected_bounces:
            return False

        out = main(grid)
        if out.shape != (3, 3):
            return False
        if int(np.count_nonzero(out == BLUE)) != expected_bounces:
            return False
        if set(map(int, np.unique(out))) - {BLACK, BLUE}:
            return False
        if len(reachable) < 2:
            return False
    except Exception:
        return False
    return True


def _gen_attempt(target_bounces: int) -> np.ndarray | None:
    shape = _sample_shape(target_bounces)
    direction = random.choice(DIAGONALS)
    start = _sample_start(shape, direction)
    arrow_cells = set(_arrow_cells(start, direction))

    by_count = _first_hit_counts(shape, start, direction)
    candidates = [cell for cell in by_count.get(target_bounces, []) if cell not in arrow_cells]
    if len(candidates) < 2:
        return None

    absorber = random.choice(candidates)
    arrow_color, absorber_color = random.sample(INPUT_COLORS, 2)
    grid = _build_grid(shape, start, direction, absorber, arrow_color, absorber_color)
    if not _validate_generated(grid, target_bounces):
        return None
    return grid


def generate_input() -> np.ndarray:
    target_bounces = _next_target_bounces()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(target_bounces)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input with {target_bounces} bounces")
