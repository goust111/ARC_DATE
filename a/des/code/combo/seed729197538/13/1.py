from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
INNER_SHAPES = tuple((h, w) for h in range(3, 8) for w in range(3, 8))

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (1, 4)
MIN_TRUNCATED_LEN = 2
GEN_RETRIES = 400
PLACEMENT_RETRIES_PER_SHAPE = 16


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _marker_cell(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    offset: int,
) -> tuple[int, int]:
    if side == "top":
        return r0 + 1, c0 + 1 + offset
    if side == "bottom":
        return r0 + inner_h, c0 + 1 + offset
    if side == "left":
        return r0 + 1 + offset, c0 + 1
    assert side == "right"
    return r0 + 1 + offset, c0 + inner_w


def _ray_dir_and_start(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    marker_r: int,
    marker_c: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (-1, 0), (r0 - 1, marker_c)
    if side == "bottom":
        return (1, 0), (r0 + inner_h + 2, marker_c)
    if side == "left":
        return (0, -1), (marker_r, c0 - 1)
    assert side == "right"
    return (0, 1), (marker_r, c0 + inner_w + 2)


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    rectangles: list[dict[str, object]] = []
    marker_cells: set[tuple[int, int]] = set()

    grey_components = _components(g == GREY)
    assert grey_components, "expected at least one grey rectangle"

    for comp in grey_components:
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "rectangle border must enclose an interior"

        expected_border = set(_border_coords(r0, c0, r1, c1))
        assert set(comp) == expected_border, "grey component must be a 1-cell-thick rectangle border"

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.size > 0, "interior must be non-empty"
        assert not np.any(inner == GREY), "interior must not contain grey pixels"

        marker_local = np.argwhere((inner != BLACK) & (inner != GREY))
        assert marker_local.shape[0] == 1, "each rectangle must contain exactly one colored marker"
        ir, ic = map(int, marker_local[0])
        marker_r = r0 + 1 + ir
        marker_c = c0 + 1 + ic

        sides: list[str] = []
        if marker_r == r0 + 1:
            sides.append("top")
        if marker_r == r1 - 1:
            sides.append("bottom")
        if marker_c == c0 + 1:
            sides.append("left")
        if marker_c == c1 - 1:
            sides.append("right")
        assert len(sides) == 1, "marker must indicate exactly one wall"
        side = sides[0]

        inner_h, inner_w = inner.shape
        area = int(inner_h * inner_w)
        color = int(g[marker_r, marker_c])
        ray_dir, ray_start = _ray_dir_and_start(r0, c0, inner_h, inner_w, side, marker_r, marker_c)

        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_shape": (inner_h, inner_w),
                "area": area,
                "color": color,
                "marker": (marker_r, marker_c),
                "ray_dir": ray_dir,
                "ray_start": ray_start,
            }
        )
        marker_cells.add((marker_r, marker_c))

    all_colored = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BLACK) & (g != GREY))
    }
    assert marker_cells == all_colored, "all non-black non-grey cells must be markers inside rectangles"

    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a colored ray outward from each marked rectangle wall, using the interior area as length.

    Input structure
    - The background is black.
    - Grey pixels form one-cell-thick closed rectangular borders.
    - Each rectangle interior contains exactly one colored marker adjacent to one inner wall.

    Rule
    - The touched inner wall determines the outward ray direction.
    - The interior area determines the ray length.
    - Rays use the marker color and are clipped by the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = sorted(_find_rectangles(g), key=lambda item: item["bbox"])  # deterministic paint order
    out = g.copy()
    h, w = g.shape

    for rect in rectangles:
        color = int(rect["color"])
        area = int(rect["area"])
        start_r, start_c = rect["ray_start"]
        dr, dc = rect["ray_dir"]

        rr = int(start_r)
        cc = int(start_c)
        for _ in range(area):
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if int(out[rr, cc]) == BLACK:
                out[rr, cc] = color
            rr += int(dr)
            cc += int(dc)

    return out


def _placement_ranges(
    height: int,
    width: int,
    inner_h: int,
    inner_w: int,
    side: str,
    want_full: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    r_lo = 1
    r_hi = height - outer_h - 1
    c_lo = 1
    c_hi = width - outer_w - 1
    if r_lo > r_hi or c_lo > c_hi:
        return None

    area = inner_h * inner_w
    if side == "top":
        if want_full:
            r_lo = max(r_lo, area)
        else:
            r_lo = max(r_lo, MIN_TRUNCATED_LEN)
            r_hi = min(r_hi, area - 1)
    elif side == "bottom":
        if want_full:
            r_hi = min(r_hi, height - outer_h - area)
        else:
            r_lo = max(r_lo, height - outer_h - area + 1)
            r_hi = min(r_hi, height - outer_h - MIN_TRUNCATED_LEN)
    elif side == "left":
        if want_full:
            c_lo = max(c_lo, area)
        else:
            c_lo = max(c_lo, MIN_TRUNCATED_LEN)
            c_hi = min(c_hi, area - 1)
    else:
        assert side == "right"
        if want_full:
            c_hi = min(c_hi, width - outer_w - area)
        else:
            c_lo = max(c_lo, width - outer_w - area + 1)
            c_hi = min(c_hi, width - outer_w - MIN_TRUNCATED_LEN)

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (r_lo, r_hi), (c_lo, c_hi)


def _ray_cells(
    height: int,
    width: int,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    offset: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    marker_r, marker_c = _marker_cell(r0, c0, inner_h, inner_w, side, offset)
    (dr, dc), (rr, cc) = _ray_dir_and_start(r0, c0, inner_h, inner_w, side, marker_r, marker_c)

    cells: list[tuple[int, int]] = []
    for _ in range(inner_h * inner_w):
        if not (0 <= rr < height and 0 <= cc < width):
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return (marker_r, marker_c), cells


def _reservation_mask(
    height: int,
    width: int,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    ray_cells: list[tuple[int, int]],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    rr0 = max(0, r0 - 1)
    rr1 = min(height, r0 + outer_h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(width, c0 + outer_w + 1)
    mask[rr0:rr1, cc0:cc1] = True

    for r, c in ray_cells:
        rr0 = max(0, r - 1)
        rr1 = min(height, r + 2)
        cc0 = max(0, c - 1)
        cc1 = min(width, c + 2)
        mask[rr0:rr1, cc0:cc1] = True

    return mask


def _draw_object(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    offset: int,
    color: int,
) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    marker_r, marker_c = _marker_cell(r0, c0, inner_h, inner_w, side, offset)
    grid[marker_r, marker_c] = int(color)


def _place_one_object(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    side: str,
    want_full: bool,
    used_areas: set[int],
) -> dict[str, object] | None:
    height, width = grid.shape
    shapes = [shape for shape in INNER_SHAPES if shape[0] * shape[1] not in used_areas]
    rng.shuffle(shapes)

    for inner_h, inner_w in shapes:
        bounds = _placement_ranges(height, width, inner_h, inner_w, side, want_full)
        if bounds is None:
            continue

        (r_lo, r_hi), (c_lo, c_hi) = bounds
        offset_hi = (inner_w - 2) if side in ("top", "bottom") else (inner_h - 2)

        for _ in range(PLACEMENT_RETRIES_PER_SHAPE):
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            offset = rng.randint(1, offset_hi)
            marker, ray_cells = _ray_cells(height, width, r0, c0, inner_h, inner_w, side, offset)
            if want_full != (len(ray_cells) == inner_h * inner_w):
                continue

            candidate_reserved = _reservation_mask(height, width, r0, c0, inner_h, inner_w, ray_cells)
            if np.any(candidate_reserved & reserved):
                continue

            _draw_object(grid, r0, c0, inner_h, inner_w, side, offset, color)
            reserved |= candidate_reserved
            used_areas.add(inner_h * inner_w)
            return {
                "bbox": (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1),
                "color": color,
                "side": side,
                "full": want_full,
                "area": inner_h * inner_w,
                "marker": marker,
                "ray_len": len(ray_cells),
            }

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)

    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    used_areas: set[int] = set()

    colors = rng.sample(list(MARKER_COLORS), k=n_objects)
    sides = rng.sample(list(SIDES), k=n_objects)
    if n_objects == 1:
        full_flags = [rng.choice([False, True])]
    else:
        n_full = rng.randint(1, n_objects - 1)
        full_flags = [True] * n_full + [False] * (n_objects - n_full)
        rng.shuffle(full_flags)

    specs = list(zip(colors, sides, full_flags))
    rng.shuffle(specs)
    specs.sort(key=lambda item: int(not item[2]))

    placed: list[dict[str, object]] = []
    for color, side, want_full in specs:
        obj = _place_one_object(rng, grid, reserved, color, side, want_full, used_areas)
        if obj is None:
            return None
        placed.append(obj)

    if n_objects >= 2:
        assert {bool(obj["full"]) for obj in placed} == {False, True}

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
