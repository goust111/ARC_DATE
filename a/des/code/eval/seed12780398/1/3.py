from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")
MODES = ("full", "truncate")

GRID_H_RANGE = (16, 26)
GRID_W_RANGE = (16, 26)
OBJECT_COUNT_RANGE = (2, 5)
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 6)

GEN_RETRY_LIMIT = 400
OBJECT_RETRY_LIMIT = 200


def _components(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append((color, np.array(coords, dtype=int)))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _direction_vector(side: str) -> tuple[int, int]:
    if side == "up":
        return -1, 0
    if side == "down":
        return 1, 0
    if side == "left":
        return 0, -1
    if side == "right":
        return 0, 1
    raise AssertionError(f"unknown side: {side}")


def _gap_info(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, str]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    missing = np.argwhere(patch != color)
    assert missing.shape[0] == 1, "expected a full rectangle with exactly one missing border cell"
    assert int(np.count_nonzero(patch == color)) == int(coords.shape[0]), "component must fill its bbox except the gap"

    mr, mc = map(int, missing[0])
    sides = []
    if mr == 0:
        sides.append("up")
    if mr == patch.shape[0] - 1:
        sides.append("down")
    if mc == 0:
        sides.append("left")
    if mc == patch.shape[1] - 1:
        sides.append("right")
    assert len(sides) == 1, "gap must lie on exactly one non-corner side"

    return r0 + mr, c0 + mc, sides[0]


def _ray_cells(
    height: int,
    width: int,
    gap_r: int,
    gap_c: int,
    side: str,
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = _direction_vector(side)
    r = gap_r + dr
    c = gap_c + dc
    cells: list[tuple[int, int]] = []
    for _ in range(int(length)):
        if not (0 <= r < height and 0 <= c < width):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the missing side of each notched rectangle, then emit an outward ray whose
    length equals the rectangle's current area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    components = []
    for color, coords in _components(g):
        r0, c0, _r1, _c1 = _bbox(coords)
        components.append((r0, c0, color, coords))
    components.sort(key=lambda item: (item[0], item[1], item[2]))

    for _r0, _c0, color, coords in components:
        gap_r, gap_c, side = _gap_info(g, color, coords)
        for r, c in _ray_cells(g.shape[0], g.shape[1], gap_r, gap_c, side, int(coords.shape[0])):
            if int(out[r, c]) != BACKGROUND:
                break
            out[r, c] = color

    return out


def _balanced_choices(options: tuple[str, ...], count: int) -> list[str]:
    chosen: list[str] = []
    while len(chosen) < count:
        batch = list(options)
        random.shuffle(batch)
        chosen.extend(batch)
    return chosen[:count]


def _position_ranges(
    height: int,
    width: int,
    rect_h: int,
    rect_w: int,
    side: str,
    mode: str,
) -> tuple[int, int, int, int] | None:
    r_lo = 1
    r_hi = height - rect_h - 1
    c_lo = 1
    c_hi = width - rect_w - 1
    if r_lo > r_hi or c_lo > c_hi:
        return None

    area = rect_h * rect_w - 1
    if mode == "full":
        if side == "up":
            r_lo = max(r_lo, area)
        elif side == "down":
            r_hi = min(r_hi, height - rect_h - area)
        elif side == "left":
            c_lo = max(c_lo, area)
        else:
            c_hi = min(c_hi, width - rect_w - area)
    elif mode == "truncate":
        if side == "up":
            r_hi = min(r_hi, area - 1)
        elif side == "down":
            r_lo = max(r_lo, height - rect_h - area + 1)
        elif side == "left":
            c_hi = min(c_hi, area - 1)
        else:
            c_lo = max(c_lo, width - rect_w - area + 1)

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return r_lo, r_hi, c_lo, c_hi


def _size_candidates(height: int, width: int, side: str, mode: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    h_max = min(RECT_H_RANGE[1], height - 2)
    w_max = min(RECT_W_RANGE[1], width - 2)
    for rect_h in range(RECT_H_RANGE[0], h_max + 1):
        for rect_w in range(RECT_W_RANGE[0], w_max + 1):
            if side in ("up", "down") and rect_w < 3:
                continue
            if side in ("left", "right") and rect_h < 3:
                continue
            if _position_ranges(height, width, rect_h, rect_w, side, mode) is not None:
                candidates.append((rect_h, rect_w))
    return candidates


def _sample_gap(r0: int, c0: int, rect_h: int, rect_w: int, side: str) -> tuple[int, int]:
    if side == "up":
        return r0, c0 + random.randint(1, rect_w - 2)
    if side == "down":
        return r0 + rect_h - 1, c0 + random.randint(1, rect_w - 2)
    if side == "left":
        return r0 + random.randint(1, rect_h - 2), c0
    return r0 + random.randint(1, rect_h - 2), c0 + rect_w - 1


def _place_one_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    side: str,
    mode: str,
) -> tuple[str, bool] | None:
    height, width = grid.shape
    for trial_mode in (mode, "any"):
        candidates = _size_candidates(height, width, side, trial_mode)
        if not candidates:
            continue

        for _ in range(OBJECT_RETRY_LIMIT):
            rect_h, rect_w = random.choice(candidates)
            ranges = _position_ranges(height, width, rect_h, rect_w, side, trial_mode)
            assert ranges is not None
            r_lo, r_hi, c_lo, c_hi = ranges
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)

            er0 = max(0, r0 - 1)
            ec0 = max(0, c0 - 1)
            er1 = min(height - 1, r0 + rect_h)
            ec1 = min(width - 1, c0 + rect_w)
            if blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue

            gap_r, gap_c = _sample_gap(r0, c0, rect_h, rect_w, side)
            area = rect_h * rect_w - 1
            ray = _ray_cells(height, width, gap_r, gap_c, side, area)
            if any(blocked[r, c] for r, c in ray):
                continue

            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
            grid[gap_r, gap_c] = BACKGROUND
            blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
            for r, c in ray:
                blocked[r, c] = True

            return side, len(ray) == area

    return None


def _is_valid(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    components = _components(grid)
    if not (OBJECT_COUNT_RANGE[0] <= len(components) <= OBJECT_COUNT_RANGE[1]):
        return False

    sides: set[str] = set()
    truncation_flags: set[bool] = set()
    for color, coords in components:
        gap_r, gap_c, side = _gap_info(grid, color, coords)
        sides.add(side)
        ray = _ray_cells(grid.shape[0], grid.shape[1], gap_r, gap_c, side, int(coords.shape[0]))
        truncation_flags.add(len(ray) == int(coords.shape[0]))

    if len(sides) < min(2, len(components)):
        return False
    if len(components) >= 2 and len(truncation_flags) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        count = random.randint(*OBJECT_COUNT_RANGE)
        colors = random.sample(PALETTE, count)
        sides = _balanced_choices(DIRECTIONS, count)
        modes = _balanced_choices(MODES, count)

        grid = np.zeros((height, width), dtype=int)
        blocked = np.zeros((height, width), dtype=bool)

        placements_ok = True
        for color, side, mode in zip(colors, sides, modes):
            result = _place_one_object(grid, blocked, color, side, mode)
            if result is None:
                placements_ok = False
                break

        if placements_ok and _is_valid(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
