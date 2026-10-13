import random
from collections import deque

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


# --------------- shared helpers ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_local_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0)


def _enclosed_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    bg_mask = ~mask
    h, w = mask.shape
    reached = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(bg_mask[r, c]) and not bool(reached[r, c]):
                reached[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(bg_mask[r, c]) and not bool(reached[r, c]):
                reached[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(bg_mask[nr, nc]) and not bool(reached[nr, nc]):
                reached[nr, nc] = True
                q.append((nr, nc))

    enclosed = bg_mask & ~reached
    return _components(enclosed)


def _hole_orientation(coords: list[tuple[int, int]]) -> str | None:
    local_mask, _ = _coords_to_local_mask(coords)
    holes = _enclosed_regions(local_mask)
    if len(holes) != 1:
        return None

    hr0, hc0, hr1, hc1 = _bbox(holes[0])
    hole_h = hr1 - hr0 + 1
    hole_w = hc1 - hc0 + 1
    if hole_h == 1 and hole_w >= 2:
        return "horizontal"
    if hole_w == 1 and hole_h >= 2:
        return "vertical"
    return None


def _center(coords: list[tuple[int, int]]) -> tuple[float, float]:
    r0, c0, r1, c1 = _bbox(coords)
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _shift_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _in_bounds(coords: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _move_until_adjacent(
    mover: list[tuple[int, int]],
    static: list[tuple[int, int]],
    dr: int,
    dc: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    static_set = set(static)
    for step in range(max(shape) * 2 + 1):
        shifted = _shift_coords(mover, dr * step, dc * step)
        if not _in_bounds(shifted, shape):
            break
        if shifted & static_set:
            raise AssertionError("moved object overlapped the stationary object")
        if _touches(shifted, static_set):
            return shifted
    raise AssertionError("failed to find an adjacent landing position")


def _mask_to_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                cells.add((top + r, left + c))
    return cells


def _pad_cells(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((nr, nc))
    return padded


def _make_object_mask(height: int, width: int, hole_orientation: str | None) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if hole_orientation == "horizontal":
        hole_row = random.randint(1, height - 2)
        mask[hole_row, 1:-1] = False
    elif hole_orientation == "vertical":
        hole_col = random.randint(1, width - 2)
        mask[1:-1, hole_col] = False
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each holed object onto its same-colored partner along the axis perpendicular to the hole.

    Input structure:
    - The background is black.
    - Each non-background color appears as exactly two separate 4-connected objects.
    - Within each same-colored pair, one object is solid while the other contains one enclosed single-row or single-column black slit.
    - The two objects of a pair are separated along one axis but remain aligned on the perpendicular axis.

    Rule:
    - For each color pair, identify the object containing the enclosed slit.
    - If the slit is horizontal, move that object vertically; if the slit is vertical, move it horizontally.
    - Move it toward the same-colored solid object until the two objects first become 4-neighbor adjacent without overlapping.
    - Clear the holed object's original position to black and leave the solid object fixed.

    Key insight:
    - The hole is not meant to be filled; its orientation encodes the perpendicular translation axis of the moving object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    moves: list[tuple[int, set[tuple[int, int]], set[tuple[int, int]]]] = []
    colors = [int(color) for color in np.unique(g) if int(color) != BG]

    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each color must form exactly one pair of objects"

        infos = []
        for comp in comps:
            infos.append({"coords": comp, "hole_orientation": _hole_orientation(comp)})

        movers = [info for info in infos if info["hole_orientation"] is not None]
        statics = [info for info in infos if info["hole_orientation"] is None]
        assert len(movers) == 1 and len(statics) == 1, "each pair needs exactly one holed object"

        mover = movers[0]["coords"]
        static = statics[0]["coords"]
        hole_orientation = movers[0]["hole_orientation"]
        mover_center = _center(mover)
        static_center = _center(static)

        if hole_orientation == "horizontal":
            dr = 1 if static_center[0] > mover_center[0] else -1
            dc = 0
        else:
            dr = 0
            dc = 1 if static_center[1] > mover_center[1] else -1

        mover_cells = set(mover)
        landed = _move_until_adjacent(mover, static, dr, dc, g.shape)
        assert all(g[r, c] == BG or (r, c) in mover_cells for r, c in landed), (
            "landing area may only cover background or the mover's former cells"
        )
        moves.append((color, mover_cells, landed))

    out = g.copy()
    for _color, old_cells, _new_cells in moves:
        for r, c in old_cells:
            out[r, c] = BG
    for color, _old_cells, new_cells in moves:
        for r, c in new_cells:
            out[r, c] = color
    return out


# --------------- generator ---------------


GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
GROUP_COUNT_OPTIONS = (1, 2, 3)
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
OBJ_H_RANGE = (5, 8)
OBJ_W_RANGE = (5, 8)
GAP_RANGE = (1, 5)
GRID_MIN_BY_GROUPS = {1: 14, 2: 18, 3: 24}
GRID_MAX = 30
MARGIN = 1


def _place_group(
    rng: random.Random,
    rows: int,
    cols: int,
    color: int,
    blocked: set[tuple[int, int]],
    inp: np.ndarray,
    out: np.ndarray,
) -> bool:
    hole_orientation = rng.choice(("horizontal", "vertical"))
    move_axis = "vertical" if hole_orientation == "horizontal" else "horizontal"
    side = rng.choice(("left", "right")) if move_axis == "horizontal" else rng.choice(("top", "bottom"))
    height = rng.randint(*OBJ_H_RANGE)
    width = rng.randint(*OBJ_W_RANGE)
    gap = rng.randint(*GAP_RANGE)

    holed_mask = _make_object_mask(height, width, hole_orientation)
    solid_mask = _make_object_mask(height, width, None)

    for _ in range(PLACEMENT_ATTEMPTS):
        if move_axis == "horizontal":
            top_lo = MARGIN
            top_hi = rows - MARGIN - height
            if side == "left":
                left_lo = MARGIN + gap
                left_hi = cols - MARGIN - 2 * width
            else:
                left_lo = MARGIN
                left_hi = cols - MARGIN - (2 * width + gap)
            if top_lo > top_hi or left_lo > left_hi:
                return False
            pair_top = rng.randint(top_lo, top_hi)
            pair_left = rng.randint(left_lo, left_hi)

            if side == "left":
                final_top, final_left = pair_top, pair_left
                initial_top, initial_left = pair_top, pair_left - gap
                solid_top, solid_left = pair_top, pair_left + width
            else:
                solid_top, solid_left = pair_top, pair_left
                final_top, final_left = pair_top, pair_left + width
                initial_top, initial_left = pair_top, pair_left + width + gap
        else:
            left_lo = MARGIN
            left_hi = cols - MARGIN - width
            if side == "top":
                top_lo = MARGIN + gap
                top_hi = rows - MARGIN - 2 * height
            else:
                top_lo = MARGIN
                top_hi = rows - MARGIN - (2 * height + gap)
            if top_lo > top_hi or left_lo > left_hi:
                return False
            pair_top = rng.randint(top_lo, top_hi)
            pair_left = rng.randint(left_lo, left_hi)

            if side == "top":
                final_top, final_left = pair_top, pair_left
                initial_top, initial_left = pair_top - gap, pair_left
                solid_top, solid_left = pair_top + height, pair_left
            else:
                solid_top, solid_left = pair_top, pair_left
                final_top, final_left = pair_top + height, pair_left
                initial_top, initial_left = pair_top + height + gap, pair_left

        solid_cells = _mask_to_cells(solid_mask, solid_top, solid_left)
        initial_cells = _mask_to_cells(holed_mask, initial_top, initial_left)
        final_cells = _mask_to_cells(holed_mask, final_top, final_left)

        path_dr = 0 if initial_top == final_top else (1 if initial_top > final_top else -1)
        path_dc = 0 if initial_left == final_left else (1 if initial_left > final_left else -1)
        path_cells: set[tuple[int, int]] = set()
        for step in range(gap + 1):
            path_cells |= _mask_to_cells(holed_mask, final_top + path_dr * step, final_left + path_dc * step)

        reserve = solid_cells | path_cells
        padded = _pad_cells(reserve, (rows, cols), pad=1)

        if blocked & padded:
            continue
        if any(inp[r, c] != BG for r, c in solid_cells | initial_cells):
            continue
        if any(out[r, c] != BG for r, c in solid_cells | final_cells):
            continue

        for r, c in solid_cells | initial_cells:
            inp[r, c] = color
        for r, c in solid_cells | final_cells:
            out[r, c] = color
        blocked |= padded
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    group_count = rng.choice(GROUP_COUNT_OPTIONS)
    min_side = GRID_MIN_BY_GROUPS[group_count]
    rows = rng.randint(min_side, GRID_MAX)
    cols = rng.randint(min_side, GRID_MAX)

    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()
    colors = rng.sample(COLOR_POOL, group_count)

    for color in colors:
        if not _place_group(rng, rows, cols, color, blocked, inp, out):
            return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
