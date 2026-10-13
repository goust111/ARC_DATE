from __future__ import annotations

import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import Color, in_bounds, paste


BG = Color.BLACK
START = Color.YELLOW
STOP = Color.RED
BODY = Color.GREY
ANCHOR = Color.ORANGE
UP = (-1, 0)

PALETTE = [Color.BLUE, Color.GREEN, Color.PINK, Color.TEAL, Color.MAROON]
SPRITE_SHAPES = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)),
)


def _left(dr: int, dc: int) -> tuple[int, int]:
    return -int(dc), int(dr)


def _right(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), -int(dr)


def _bfs_component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _legend_info(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int]], set[tuple[int, int]]]:
    cols = [int(c) for c in np.flatnonzero(grid[0] != BG)]
    assert len(cols) in (3, 4), f"legend length must be 3 or 4, got {len(cols)}"
    assert cols == list(range(cols[0], cols[0] + len(cols))), "legend must be contiguous on the top row"
    colors = [int(grid[0, c]) for c in cols]
    coords = [(0, int(c)) for c in cols]
    return colors, coords, set(coords)


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == START)
    assert pts.shape == (1, 2), "expected a unique yellow start pixel"
    return int(pts[0, 0]), int(pts[0, 1])


def _stop_info(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    H, W = grid.shape
    found: list[list[tuple[int, int]]] = []
    for r in range(H - 1):
        for c in range(W - 1):
            if bool(np.all(grid[r : r + 2, c : c + 2] == STOP)):
                found.append([(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)])
    uniq = []
    seen = set()
    for block in found:
        key = tuple(sorted(block))
        if key not in seen:
            seen.add(key)
            uniq.append(block)
    assert len(uniq) == 1, f"expected one red 2x2 stop block, got {len(uniq)}"
    return uniq[0], set(uniq[0])


def _sprite_info(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], set[tuple[int, int]]]:
    mask = np.isin(grid, [BODY, ANCHOR])
    coords = np.argwhere(mask)
    assert 3 <= coords.shape[0] <= 5, "sprite must have 3-5 pixels"
    anchor_pts = np.argwhere(grid == ANCHOR)
    assert anchor_pts.shape == (1, 2), "expected one orange source anchor"
    ar, ac = map(int, anchor_pts[0])
    comp = _bfs_component(mask, (ar, ac))
    assert len(comp) == int(coords.shape[0]), "sprite must be a single 4-connected component"

    coords_arr = np.array(sorted(comp), dtype=int)
    r0, c0 = coords_arr.min(axis=0)
    r1, c1 = coords_arr.max(axis=0)
    patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
    for r, c in comp:
        patch[int(r - r0), int(c - c0)] = int(grid[r, c])
    return patch, (int(ar - r0), int(ac - c0)), comp


def _simulate(
    grid: np.ndarray,
    start: tuple[int, int],
    legend_cells: set[tuple[int, int]],
    sprite_cells: set[tuple[int, int]],
    stop_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[str]]:
    H, W = grid.shape
    solid = grid != BG
    stop_mask = np.zeros_like(grid, dtype=bool)
    for r, c in stop_cells:
        stop_mask[r, c] = True

    r, c = map(int, start)
    dr, dc = UP
    turns: list[tuple[int, int]] = []
    kinds: list[str] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    for _ in range(H * W * 8):
        state = (int(r), int(c), int(dr), int(dc), int(len(turns) % 2))
        assert state not in seen, "path entered a loop"
        seen.add(state)

        nr, nc = int(r + dr), int(c + dc)
        if in_bounds(grid.shape, nr, nc) and bool(stop_mask[nr, nc]):
            return turns, kinds

        blocked = (not in_bounds(grid.shape, nr, nc)) or bool(solid[nr, nc])
        if not blocked:
            r, c = nr, nc
            continue

        if not in_bounds(grid.shape, nr, nc):
            kind = "boundary"
        elif (nr, nc) in legend_cells:
            kind = "legend"
        elif (nr, nc) in sprite_cells:
            kind = "sprite"
        else:
            kind = "other"

        k = len(turns)
        turns.append((int(r), int(c)))
        kinds.append(kind)

        preferred = _left(dr, dc) if k % 2 == 0 else _right(dr, dc)
        alternate = _right(dr, dc) if k % 2 == 0 else _left(dr, dc)
        chosen = None
        for cand_dr, cand_dc in (preferred, alternate):
            tr, tc = int(r + cand_dr), int(c + cand_dc)
            if not in_bounds(grid.shape, tr, tc):
                continue
            if bool(stop_mask[tr, tc]) or not bool(solid[tr, tc]):
                chosen = (int(cand_dr), int(cand_dc))
                break
        assert chosen is not None, "collision must admit one in-bounds perpendicular continuation"
        dr, dc = chosen

    raise AssertionError("path did not terminate")


def _stamp_cells(
    patch: np.ndarray,
    anchor: tuple[int, int],
    turn: tuple[int, int],
) -> set[tuple[int, int]]:
    ar, ac = map(int, anchor)
    tr, tc = map(int, turn)
    coords = set()
    for r, c in np.argwhere(patch != BG):
        coords.add((int(tr + r - ar), int(tc + c - ac)))
    return coords


def _build_output(
    grid: np.ndarray,
    turn_points: list[tuple[int, int]],
    legend_colors: list[int],
    sprite_patch: np.ndarray,
    anchor: tuple[int, int],
    protected: set[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    mask = sprite_patch != BG
    ar, ac = map(int, anchor)
    for k, (r, c) in enumerate(turn_points):
        color = int(legend_colors[k % len(legend_colors)])
        stamp = np.where(mask, color, BG).astype(int)
        paste(out, stamp, int(r - ar), int(c - ac), transparent=BG)
    for r, c in protected:
        out[r, c] = int(grid[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored copies of a source sprite at the alternating-turn anchors of an invisible path.

    Input structure:
    - A black background contains one yellow start pixel, one red 2x2 stop block, a short top-row color legend, and one small gray sprite with an orange source anchor.
    - The legend gives the recoloring cycle from left to right.
    - The sprite is a single 4-connected 3-5-pixel component whose orange pixel marks the alignment anchor.

    Rule:
    - Launch a particle from the yellow start pixel moving upward through background cells only.
    - If the next cell is the red stop block, stop immediately.
    - If the next step would leave the grid or hit a fixed non-black object, record the current cell as the next turn anchor, then alternate 90-degree turns left, right, left, right; if the preferred perpendicular direction is blocked, use the other one.
    - At each recorded anchor, translate the source sprite so its orange anchor lands on that cell, recolor the whole copy with the next legend color, and overlay later stamps over earlier ones.
    - Keep the original legend, start pixel, stop block, and source sprite unchanged.

    Key insight:
    - The output depends on a four-link chain: simulate the path, extract the turn anchors, align the sprite by its anchor at each turn, and recolor each copy by the legend cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_colors, legend_coords, legend_cells = _legend_info(g)
    start = _find_start(g)
    stop_coords, stop_cells = _stop_info(g)
    sprite_patch, anchor, sprite_cells = _sprite_info(g)

    turn_points, _kinds = _simulate(g, start, legend_cells, sprite_cells, stop_cells)
    assert turn_points, "expected at least one turn"

    protected = set(legend_coords) | {start} | stop_cells | sprite_cells
    return _build_output(g, turn_points, legend_colors, sprite_patch, anchor, protected)


def _render_sprite(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = BODY
    patch[0, 0] = ANCHOR
    return patch


def _stamp_sets_for_validation(
    turns: list[tuple[int, int]],
    patch: np.ndarray,
    anchor: tuple[int, int],
) -> list[set[tuple[int, int]]]:
    return [_stamp_cells(patch, anchor, turn) for turn in turns]


def _validate_instance(grid: np.ndarray) -> None:
    legend_colors, legend_coords, legend_cells = _legend_info(grid)
    start = _find_start(grid)
    _stop_coords, stop_cells = _stop_info(grid)
    sprite_patch, anchor, sprite_cells = _sprite_info(grid)
    turns, kinds = _simulate(grid, start, legend_cells, sprite_cells, stop_cells)

    assert kinds == ["legend", "boundary", "sprite"], kinds
    assert len(turns) == 3, turns

    stamp_sets = _stamp_sets_for_validation(turns, sprite_patch, anchor)
    protected = set(legend_coords) | {start} | stop_cells
    for cells in stamp_sets:
        assert cells.isdisjoint(protected), "stamps must not cover legend/start/stop"
        for r, c in cells:
            assert in_bounds(grid.shape, r, c), "stamps must stay inside the grid"

    for i in range(len(stamp_sets)):
        for j in range(i + 1, len(stamp_sets)):
            assert stamp_sets[i].isdisjoint(stamp_sets[j]), "stamp copies must not overlap"

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "output must change the grid"
    for k, cells in enumerate(stamp_sets):
        color = int(legend_colors[k % len(legend_colors)])
        visible = [cell for cell in cells if cell not in sprite_cells]
        assert visible, "each stamp should contribute visible pixels"
        for r, c in visible:
            assert int(out[r, c]) == color, "visible stamp pixels must use the cycle color"


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 24)
    W = random.randint(18, 24)

    shape = random.choice(SPRITE_SHAPES)
    sprite_patch = _render_sprite(shape)
    sprite_h, sprite_w = sprite_patch.shape

    legend_len = random.choice((3, 4))
    if sprite_w > legend_len:
        return None

    legend_colors = random.sample(PALETTE, legend_len)
    start_col = random.randint(sprite_w, legend_len - 1)
    sprite_r = random.randint(6, H - sprite_h - 3)
    stop_c = random.randint(sprite_w + 3, W - 3)

    grid = np.zeros((H, W), dtype=int)
    for c, color in enumerate(legend_colors):
        grid[0, c] = int(color)
    grid[H - 1, start_col] = START
    grid[sprite_r - 2 : sprite_r, stop_c : stop_c + 2] = STOP
    for r, c in np.argwhere(sprite_patch != BG):
        grid[int(sprite_r + r), int(c)] = int(sprite_patch[int(r), int(c)])

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
