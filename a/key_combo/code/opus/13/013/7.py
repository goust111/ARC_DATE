from __future__ import annotations

import random

import numpy as np


BLACK = 0
TOP_COLOR = 1
RIGHT_COLOR = 2
BOTTOM_COLOR = 3
LEFT_COLOR = 4

FRAME_COLORS = {
    "top": TOP_COLOR,
    "right": RIGHT_COLOR,
    "bottom": BOTTOM_COLOR,
    "left": LEFT_COLOR,
}
FRAME_SIDES = ("top", "right", "bottom", "left")
TILE_COLORS = (TOP_COLOR, RIGHT_COLOR, BOTTOM_COLOR, LEFT_COLOR)

GEN_ATTEMPTS = 800
GEN_INTERIOR_RANGE = (8, 11)
GEN_MARGIN_RANGE = (1, 3)
GEN_PERIODS = (2, 3)
GEN_HOLE_RATIO_RANGE = (0.32, 0.48)


def _bbox_nonzero(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    assert coords.size > 0, "expected non-black structure"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _constant_color(segment: np.ndarray, name: str) -> int:
    vals = np.unique(segment)
    assert len(vals) == 1, f"expected a monochrome {name}"
    color = int(vals[0])
    assert color != BLACK, f"expected a non-black {name}"
    return color


def _find_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    r0, c0, r1, c1 = _bbox_nonzero(grid)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame bbox too small"
    assert int(grid[r0, c0]) == BLACK
    assert int(grid[r0, c1]) == BLACK
    assert int(grid[r1, c0]) == BLACK
    assert int(grid[r1, c1]) == BLACK

    colors = {
        "top": _constant_color(grid[r0, c0 + 1 : c1], "top side"),
        "right": _constant_color(grid[r0 + 1 : r1, c1], "right side"),
        "bottom": _constant_color(grid[r1, c0 + 1 : c1], "bottom side"),
        "left": _constant_color(grid[r0 + 1 : r1, c0], "left side"),
    }
    assert len(set(colors.values())) == 4, "frame side colors must be distinct"
    return (r0, c0, r1, c1), colors


def _nearest_side(rr: int, cc: int, height: int, width: int) -> str | None:
    distances = {
        "top": rr + 1,
        "right": width - cc,
        "bottom": height - rr,
        "left": cc + 1,
    }
    nearest = min(distances.values())
    best = [side for side, dist in distances.items() if dist == nearest]
    return best[0] if len(best) == 1 else None


def _infer_tile(interior: np.ndarray) -> np.ndarray:
    observed = interior != BLACK
    assert observed.any(), "expected visible tile pixels"
    h, w = map(int, interior.shape)

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = np.full((row_period, col_period), -1, dtype=int)
            conflict = False
            for rr, cc in np.argwhere(observed):
                color = int(interior[rr, cc])
                tr = int(rr % row_period)
                tc = int(cc % col_period)
                known = int(tile[tr, tc])
                if known == -1:
                    tile[tr, tc] = color
                elif known != color:
                    conflict = True
                    break
            if not conflict and np.all(tile != -1):
                candidates.append((row_period * col_period, row_period, col_period, tile.copy()))

    assert candidates, "expected an inferable repeating tile"
    best_area = min(area for area, _, _, _ in candidates)
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "expected a unique smallest tile"
    return best[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the interior holes whose repeated-tile phase agrees with their uniquely nearest frame side.

    Input structure:
    - The grid has a black background and exactly one hollow rectangular frame whose four sides are monochrome and whose corners stay black.
    - Inside the frame, the non-black pixels are visible cells from a repeated tile that uses only the same four colors as the frame sides.
    - Some interior tile cells are missing as black holes, while the frame border itself remains intact.

    Rule:
    - Detect the frame and read the four side colors.
    - Infer the unique smallest repeated tile that matches every visible non-black interior pixel by row/column phase.
    - For each black interior hole, compute both its tile-phase color and its uniquely nearest frame side by Manhattan distance to the top, right, bottom, and left borders.
    - Fill the hole only when those two colors are equal; otherwise leave it black, and keep every original non-black pixel unchanged.

    Key insight:
    - The output is the intersection of two independent constraints: periodic tile completion says what color a hole could be, while nearest-side ownership says which border color is allowed to claim that hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    (r0, c0, r1, c1), side_colors = _find_frame(g)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    tile = _infer_tile(interior)
    tile_h, tile_w = map(int, tile.shape)

    out = g.copy()
    inner_h, inner_w = map(int, interior.shape)
    for rr in range(inner_h):
        for cc in range(inner_w):
            if int(interior[rr, cc]) != BLACK:
                continue
            predicted = int(tile[rr % tile_h, cc % tile_w])
            nearest = _nearest_side(rr, cc, inner_h, inner_w)
            if nearest is not None and predicted == side_colors[nearest]:
                out[r0 + 1 + rr, c0 + 1 + cc] = predicted

    return out


def _row_period(tile: np.ndarray) -> int:
    h, _ = map(int, tile.shape)
    for period in range(1, h + 1):
        ok = True
        for r in range(h):
            if not np.array_equal(tile[r], tile[r % period]):
                ok = False
                break
        if ok:
            return period
    return h


def _col_period(tile: np.ndarray) -> int:
    _, w = map(int, tile.shape)
    for period in range(1, w + 1):
        ok = True
        for c in range(w):
            if not np.array_equal(tile[:, c], tile[:, c % period]):
                ok = False
                break
        if ok:
            return period
    return w


def _generate_tile(period: int) -> np.ndarray:
    for _ in range(400):
        tile = np.array(
            random.choices(TILE_COLORS, k=period * period),
            dtype=int,
        ).reshape(period, period)
        if set(map(int, tile.ravel())) != set(TILE_COLORS):
            continue
        if _row_period(tile) != period or _col_period(tile) != period:
            continue
        return tile
    raise RuntimeError("failed to generate a non-degenerate tile")


def _tile_canvas(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rep_r = height // tile.shape[0] + 1
    rep_c = width // tile.shape[1] + 1
    return np.tile(tile, (rep_r, rep_c))[:height, :width]


def _phase_representatives(height: int, width: int, period: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    phases: dict[tuple[int, int], list[tuple[int, int]]] = {
        (pr, pc): [] for pr in range(period) for pc in range(period)
    }
    for rr in range(height):
        for cc in range(width):
            phases[(rr % period, cc % period)].append((rr, cc))
    return phases


def _expected_output(grid: np.ndarray, tile: np.ndarray) -> np.ndarray:
    out = grid.copy()
    (r0, c0, r1, c1), side_colors = _find_frame(grid)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    for rr in range(inner_h):
        for cc in range(inner_w):
            if int(grid[r0 + 1 + rr, c0 + 1 + cc]) != BLACK:
                continue
            predicted = int(tile[rr % tile.shape[0], cc % tile.shape[1]])
            nearest = _nearest_side(rr, cc, inner_h, inner_w)
            if nearest is not None and predicted == side_colors[nearest]:
                out[r0 + 1 + rr, c0 + 1 + cc] = predicted
    return out


def _choose_holes(full_interior: np.ndarray, period: int) -> set[tuple[int, int]] | None:
    height, width = map(int, full_interior.shape)
    phase_cells = _phase_representatives(height, width, period)
    keep_visible = {random.choice(cells) for cells in phase_cells.values()}

    eligible: list[tuple[int, int]] = []
    match: list[tuple[int, int]] = []
    mismatch: list[tuple[int, int]] = []
    side_of: dict[tuple[int, int], str] = {}
    for rr in range(height):
        for cc in range(width):
            if (rr, cc) in keep_visible:
                continue
            nearest = _nearest_side(rr, cc, height, width)
            if nearest is None:
                continue
            side_of[(rr, cc)] = nearest
            eligible.append((rr, cc))
            color = int(full_interior[rr, cc])
            if color == FRAME_COLORS[nearest]:
                match.append((rr, cc))
            else:
                mismatch.append((rr, cc))

    if len(match) < 4 or len(mismatch) < 4:
        return None

    min_holes = int(np.ceil(height * width * GEN_HOLE_RATIO_RANGE[0]))
    max_holes = int(np.floor(height * width * GEN_HOLE_RATIO_RANGE[1]))
    max_holes = min(max_holes, len(eligible))
    if min_holes > max_holes:
        return None

    for _ in range(300):
        target = random.randint(min_holes, max_holes)
        anchors = [random.choice(match), random.choice(mismatch)]
        candidate_sides = list({side_of[cell] for cell in eligible})
        if len(candidate_sides) >= 3:
            for side in random.sample(candidate_sides, 3):
                pool = [cell for cell in eligible if side_of[cell] == side and cell not in anchors]
                if pool:
                    anchors.append(random.choice(pool))
        chosen = list(dict.fromkeys(anchors))
        if len(chosen) > target:
            continue
        pool = [cell for cell in eligible if cell not in chosen]
        chosen.extend(random.sample(pool, target - len(chosen)))
        holes = set(chosen)

        filled = {cell for cell in holes if int(full_interior[cell]) == FRAME_COLORS[side_of[cell]]}
        unfilled = holes - filled
        if len(filled) < 4 or len(unfilled) < 4:
            continue
        if len({side_of[cell] for cell in holes}) < 3:
            continue
        if len({side_of[cell] for cell in filled}) < 2:
            continue
        if len({side_of[cell] for cell in unfilled}) < 2:
            continue
        return holes

    return None


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(*GEN_INTERIOR_RANGE)
    inner_w = random.randint(*GEN_INTERIOR_RANGE)
    period = random.choice(GEN_PERIODS)
    tile = _generate_tile(period)
    full_interior = _tile_canvas(tile, inner_h, inner_w)
    holes = _choose_holes(full_interior, period)
    if holes is None:
        return None

    top_margin = random.randint(*GEN_MARGIN_RANGE)
    bottom_margin = random.randint(*GEN_MARGIN_RANGE)
    left_margin = random.randint(*GEN_MARGIN_RANGE)
    right_margin = random.randint(*GEN_MARGIN_RANGE)
    height = top_margin + inner_h + 2 + bottom_margin
    width = left_margin + inner_w + 2 + right_margin
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    grid[r0, c0 + 1 : c1] = TOP_COLOR
    grid[r1, c0 + 1 : c1] = BOTTOM_COLOR
    grid[r0 + 1 : r1, c0] = LEFT_COLOR
    grid[r0 + 1 : r1, c1] = RIGHT_COLOR
    grid[r0 + 1 : r1, c0 + 1 : c1] = full_interior
    for rr, cc in holes:
        grid[r0 + 1 + rr, c0 + 1 + cc] = BLACK

    masked_interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    try:
        inferred = _infer_tile(masked_interior)
    except AssertionError:
        return None
    if inferred.shape != tile.shape or not np.array_equal(inferred, tile):
        return None

    expected = _expected_output(grid, tile)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None

    changed = int(np.count_nonzero(produced != grid))
    if changed < 4:
        return None
    if not np.any(expected[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
