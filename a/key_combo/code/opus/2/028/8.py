from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_028_8",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
GREY = Color.GREY

MOVABLE_COLORS = {BLUE, GREEN, YELLOW}
PANEL_HEIGHT_RANGE = (7, 9)
PANEL_WIDTH_RANGE = (6, 8)
SHARED_HOLE_RANGE = (3, 6)
SIDE_HOLE_RANGE = (5, 10)
GENERATION_RETRIES = 500
ORTHOGONAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    if len(separator_rows) == 1:
        sep = int(separator_rows[0])
        top = g[:sep]
        bottom = g[sep + 1 :]
        assert top.shape == bottom.shape and top.size > 0, "separator must split equal panels"
        assert not np.any(top == GREY) and not np.any(bottom == GREY), "grey may only be the divider"
        left, right = top, bottom
    else:
        assert len(separator_rows) == 0, "expected at most one full grey separator row"
        H, W = map(int, g.shape)
        assert W % 2 == 0, "side-by-side inputs must have equal-width panels"
        mid = W // 2
        left = g[:, :mid]
        right = g[:, mid:]
        assert left.shape == right.shape and left.size > 0

    assert set(map(int, np.unique(left))) <= {BG, BLUE}, "left/top panel must use only black and blue"
    assert set(map(int, np.unique(right))) <= {BG, GREEN}, "right/bottom panel must use only black and green"
    return left, right


def _visible_overlay(left_fg: np.ndarray, right_fg: np.ndarray) -> np.ndarray:
    out = np.full(left_fg.shape, RED, dtype=int)
    out[left_fg & ~right_fg] = BLUE
    out[~left_fg & right_fg] = GREEN
    out[left_fg & right_fg] = YELLOW
    return out


def _settle_layer(layer: np.ndarray, blockers: np.ndarray) -> np.ndarray:
    H, W = map(int, layer.shape)
    settled = np.zeros_like(layer, dtype=bool)

    for c in range(W):
        next_free = H - 1
        for r in range(H - 1, -1, -1):
            if blockers[r, c]:
                next_free = r - 1
                continue
            if not layer[r, c]:
                continue

            while next_free >= 0 and (blockers[next_free, c] or settled[next_free, c]):
                next_free -= 1
            assert next_free >= 0, "layer pixels must fit above blockers"
            settled[next_free, c] = True
            next_free -= 1

    return settled


def _compose_layers(blue_layer: np.ndarray, green_layer: np.ndarray, blockers: np.ndarray) -> np.ndarray:
    out = np.zeros(blue_layer.shape, dtype=int)
    out[blockers] = RED
    out[blue_layer & ~green_layer] = BLUE
    out[~blue_layer & green_layer] = GREEN
    out[blue_layer & green_layer] = YELLOW
    return out


def _solve_from_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_fg = left == BLUE
    right_fg = right == GREEN
    blockers = ~left_fg & ~right_fg

    settled_left = _settle_layer(left_fg, blockers)
    settled_right = _settle_layer(right_fg, blockers)
    return _compose_layers(settled_left, settled_right, blockers)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay two aligned color panels, keep their joint-empty cells fixed, and let the two color layers fall.

    Input structure:
    - The input contains two equal blue/green panels, either side by side or stacked
      with a single full grey separator row.
    - The left/top panel uses only black background and blue foreground.
    - The right/bottom panel uses only black background and green foreground.

    Rule:
    - Align the two panels cell by cell.
    - Any aligned position that is black in both panels becomes a fixed red blocker.
    - Treat the blue occupancy and the green occupancy as two transparent movable
      layers. In each column, blue pixels fall downward through black space until
      blocked by a red cell or an already settled blue pixel; green pixels do the
      same independently with respect to green.
    - Compose the settled layers into one output: blue-only stays blue, green-only
      stays green, cells occupied by both layers become yellow, red blockers stay
      fixed, and all vacated cells become black.

    Key insight:
    - The aligned comparison creates two different kinds of state at once: fixed red
      blockers from joint emptiness, and two falling occupancy layers whose final
      overlap must be recolored yellow.
    """
    left, right = _split_panels(np.asarray(grid, dtype=int))
    return _solve_from_panels(left, right)


def _choice(seq):
    return seq[random.randrange(len(seq))]


def _grow_cells(
    shape: tuple[int, int],
    size: int,
    *,
    forbidden: set[tuple[int, int]] | None = None,
    must_include: tuple[int, int] | None = None,
) -> set[tuple[int, int]] | None:
    H, W = shape
    forbidden = set() if forbidden is None else set(forbidden)
    all_cells = [(r, c) for r in range(H) for c in range(W) if (r, c) not in forbidden]
    if len(all_cells) < size:
        return None

    if must_include is None:
        start = _choice(all_cells)
    else:
        if must_include in forbidden:
            return None
        start = must_include

    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start, shape):
        if (nr, nc) not in forbidden:
            frontier.add((nr, nc))

    while len(cells) < size:
        if not frontier:
            return None
        nxt = _choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        for nr, nc in _neighbors4(*nxt, shape):
            if (nr, nc) not in forbidden and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    H, W = shape
    for dr, dc in ORTHOGONAL_STEPS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < H and 0 <= nc < W:
            yield nr, nc


def _component_count(mask: np.ndarray) -> int:
    grid = np.where(mask, BLUE, BG)
    return len(find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[BLUE]))


def _place_holes(shape: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    H, W = shape
    shared_count = random.randint(*SHARED_HOLE_RANGE)
    left_count = random.randint(*SIDE_HOLE_RANGE)
    right_count = random.randint(*SIDE_HOLE_RANGE)

    block_row = random.randint(1, H - 2)
    block_col = random.randrange(W)
    shared = _grow_cells(shape, shared_count, must_include=(block_row, block_col))
    if shared is None:
        return None

    left_only = _grow_cells(shape, left_count, forbidden=shared)
    if left_only is None:
        return None

    right_only = _grow_cells(shape, right_count, forbidden=shared | left_only)
    if right_only is None:
        return None

    return shared, left_only, right_only


def _build_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    holes = _place_holes(shape)
    if holes is None:
        return None
    shared, left_only, right_only = holes

    left = np.ones(shape, dtype=bool)
    right = np.ones(shape, dtype=bool)
    for r, c in shared | left_only:
        left[r, c] = False
    for r, c in shared | right_only:
        right[r, c] = False
    return left, right


def _render_input(left_mask: np.ndarray, right_mask: np.ndarray, orientation: str) -> np.ndarray:
    H, W = map(int, left_mask.shape)
    if orientation == "horizontal":
        grid = np.zeros((H, 2 * W), dtype=int)
        left_panel = np.zeros((H, W), dtype=int)
        right_panel = np.zeros((H, W), dtype=int)
        left_panel[left_mask] = BLUE
        right_panel[right_mask] = GREEN
        paste(grid, left_panel, 0, 0)
        paste(grid, right_panel, 0, W)
        return grid

    assert orientation == "vertical"
    grid = np.zeros((2 * H + 1, W), dtype=int)
    top_panel = np.zeros((H, W), dtype=int)
    bottom_panel = np.zeros((H, W), dtype=int)
    top_panel[left_mask] = BLUE
    bottom_panel[right_mask] = GREEN
    paste(grid, top_panel, 0, 0)
    grid[H, :] = GREY
    paste(grid, bottom_panel, H + 1, 0)
    return grid


def _semantic_checks(left_mask: np.ndarray, right_mask: np.ndarray) -> bool:
    blockers = ~left_mask & ~right_mask
    overlay = _visible_overlay(left_mask, right_mask)
    out = _compose_layers(
        _settle_layer(left_mask, blockers),
        _settle_layer(right_mask, blockers),
        blockers,
    )

    if int(blockers.sum()) < SHARED_HOLE_RANGE[0] or int(blockers.sum()) > SHARED_HOLE_RANGE[1]:
        return False
    if np.all(np.argwhere(blockers)[:, 0] == left_mask.shape[0] - 1):
        return False
    if _component_count(left_mask) not in (1, 2):
        return False
    if _component_count(right_mask) not in (1, 2):
        return False
    if not np.any(left_mask & ~right_mask):
        return False
    if not np.any(~left_mask & right_mask):
        return False
    if not np.any(left_mask & right_mask):
        return False
    if np.array_equal(out, overlay):
        return False

    no_block_left = _settle_layer(left_mask, np.zeros_like(blockers, dtype=bool))
    no_block_right = _settle_layer(right_mask, np.zeros_like(blockers, dtype=bool))
    if np.array_equal(no_block_left, _settle_layer(left_mask, blockers)) and np.array_equal(
        no_block_right, _settle_layer(right_mask, blockers)
    ):
        return False

    out_colors = set(map(int, np.unique(out)))
    if out_colors != {BG, BLUE, RED, GREEN, YELLOW}:
        return False

    moved = not np.array_equal(_settle_layer(left_mask, blockers), left_mask) or not np.array_equal(
        _settle_layer(right_mask, blockers), right_mask
    )
    if not moved:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*PANEL_HEIGHT_RANGE)
    W = random.randint(*PANEL_WIDTH_RANGE)
    masks = _build_masks((H, W))
    if masks is None:
        return None
    left_mask, right_mask = masks

    if not _semantic_checks(left_mask, right_mask):
        return None

    orientation = random.choice(("horizontal", "vertical"))
    grid = _render_input(left_mask, right_mask, orientation)
    expected = _solve_from_panels(*_split_panels(grid))
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-overlay gravity input")
