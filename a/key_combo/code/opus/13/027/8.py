from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, components_mask, get_bg  # noqa: E402


BACKGROUND = 0
WALL = 5
OBSTACLE = 7
SEPARATOR = 8
SEED_COLORS = (1, 2, 3, 4, 6, 9)
GEN_RETRIES = 400
_RNG = np.random.default_rng()


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_col(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != bg and np.all(col == col[0]):
            candidates.append((c, int(col[0])))
    assert candidates, "expected a full-height separator column"
    mid = grid.shape[1] / 2.0
    return min(candidates, key=lambda item: abs(item[0] - mid))


def _fill_offsets(
    grid_shape: tuple[int, int],
    anchor: tuple[int, int],
    opening: str,
    inner_h: int,
    inner_w: int,
    obstacles: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = map(int, grid_shape)
    ar, ac = map(int, anchor)
    by_row: dict[int, list[int]] = {}
    for r, c in obstacles:
        by_row.setdefault(int(r), []).append(int(c))

    out: set[tuple[int, int]] = set()
    for r in range(ar, ar + inner_h):
        row_obstacles = by_row.get(r, [])
        if opening == "left":
            blocker = max(row_obstacles) if row_obstacles else None
            c0 = 0 if blocker is None else blocker + 1
            c1 = ac + inner_w - 1
        else:
            blocker = min(row_obstacles) if row_obstacles else None
            c0 = ac
            c1 = (w - 1) if blocker is None else blocker - 1
        for c in range(c0, c1 + 1):
            if 0 <= r < h and 0 <= c < w:
                out.add((r - ar, c - ac))
    return out


def _analyze_panel(
    grid: np.ndarray,
    wall_coords: np.ndarray,
    sep_col: int,
    sep_color: int,
    wall_color: int,
    bg: int,
) -> dict:
    r0, c0, r1, c1 = _bbox(wall_coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[wall_coords[:, 0] - r0, wall_coords[:, 1] - c0] = True

    left_edge = int(patch[:, 0].sum())
    right_edge = int(patch[:, -1].sum())
    opening = "left" if left_edge < right_edge else "right"

    inner_h = r1 - r0 - 1
    inner_w = c1 - c0
    assert inner_h >= 2 and inner_w >= 3, "wall interior too small"

    anchor = (r0 + 1, c0 if opening == "left" else c0 + 1)
    ar, ac = anchor

    inside = np.zeros_like(grid, dtype=bool)
    inside[ar : ar + inner_h, ac : ac + inner_w] = True
    seed_mask = inside & (grid != bg) & (grid != wall_color) & (grid != sep_color)
    seed_coords = np.argwhere(seed_mask)
    assert seed_coords.shape[0] == 1, "expected exactly one seed inside each enclosure"
    seed_r, seed_c = map(int, seed_coords[0])

    side_mask = np.zeros_like(grid, dtype=bool)
    if opening == "left":
        side_mask[:, :sep_col] = True
    else:
        side_mask[:, sep_col + 1 :] = True
    other_mask = side_mask & (grid != bg) & (grid != wall_color) & (grid != sep_color)
    other_mask[seed_r, seed_c] = False
    obstacles = [tuple(map(int, rc)) for rc in np.argwhere(other_mask)]

    offsets = _fill_offsets(grid.shape, anchor, opening, inner_h, inner_w, obstacles)
    return {
        "bbox": (r0, c0, r1, c1),
        "opening": opening,
        "anchor": anchor,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "seed": (seed_r, seed_c),
        "seed_color": int(grid[seed_r, seed_c]),
        "obstacles": obstacles,
        "offsets": offsets,
    }


def _panel_infos(grid: np.ndarray) -> tuple[int, int, dict, dict]:
    bg = get_bg(grid)
    sep_col, sep_color = _find_separator_col(grid, bg)
    non_bg = colors_present(grid, exclude={bg, sep_color})
    assert non_bg, "expected walls and seeds"
    wall_color = max(non_bg, key=lambda color: int((grid == color).sum()))

    comps = components_mask(grid == wall_color, connectivity=4)
    assert len(comps) == 2, "expected exactly two wall components"
    left_comp = min(comps, key=lambda coords: _bbox(coords)[1])
    right_comp = max(comps, key=lambda coords: _bbox(coords)[1])

    left = _analyze_panel(grid, left_comp, sep_col, sep_color, wall_color, bg)
    right = _analyze_panel(grid, right_comp, sep_col, sep_color, wall_color, bg)
    assert left["opening"] == "left" and right["opening"] == "right", "expected outward-facing C-walls"
    assert (left["inner_h"], left["inner_w"]) == (right["inner_h"], right["inner_w"]), "enclosure sizes must match"
    return bg, wall_color, left, right


def _stamp_offsets(
    out: np.ndarray,
    bg: int,
    anchor: tuple[int, int],
    offsets: Iterable[tuple[int, int]],
    color: int,
) -> None:
    ar, ac = map(int, anchor)
    h, w = out.shape
    for dr, dc in offsets:
        r = ar + int(dr)
        c = ac + int(dc)
        if 0 <= r < h and 0 <= c < w and int(out[r, c]) == bg:
            out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the flood-filled masks carved by two outward-facing half-enclosures while recoloring them with the local seed colors.

    Input structure:
    - A black background is split by one full-height vertical separator line into left and right panels.
    - Each panel contains one gray C/U-shaped wall with the same interior size, opening outward toward the nearest side edge.
    - Inside each enclosure sits one seed pixel, and optional obstacle pixels lie on the outward radiation path.
    Rule:
    - For each panel, flood-fill the enclosure interior from its seed, then continue horizontally through the opening toward the outer edge.
    - Obstacles are preserved and cast a straight shadow in the radiation direction: the obstacle cell and everything behind it on that row stay unfilled.
    - Record each side's full fill mask relative to the enclosure's interior top-left anchor, then stamp the right mask into the left panel using the left seed color and the left mask into the right panel using the right seed color.
    Key insight:
    - The final shape shown in each panel is determined by the opposite panel's obstacle layout, while its color still comes from the local seed.
    """
    g = np.asarray(grid, dtype=int)
    bg, _wall_color, left, right = _panel_infos(g)

    out = g.copy()
    _stamp_offsets(out, bg, left["anchor"], right["offsets"], left["seed_color"])
    _stamp_offsets(out, bg, right["anchor"], left["offsets"], right["seed_color"])
    return out


def _draw_left_wall(grid: np.ndarray, top: int, anchor_c: int, inner_h: int, inner_w: int) -> None:
    bottom = top + inner_h + 1
    right = anchor_c + inner_w
    grid[top, anchor_c : right + 1] = WALL
    grid[bottom, anchor_c : right + 1] = WALL
    grid[top : bottom + 1, right] = WALL


def _draw_right_wall(grid: np.ndarray, top: int, anchor_c: int, inner_h: int, inner_w: int) -> None:
    bottom = top + inner_h + 1
    left = anchor_c - 1
    grid[top, left:anchor_c + inner_w] = WALL
    grid[bottom, left:anchor_c + inner_w] = WALL
    grid[top : bottom + 1, left] = WALL


def _place_seed(grid: np.ndarray, top: int, anchor_c: int, inner_h: int, inner_w: int, opening: str, color: int) -> None:
    seed_r = int(_RNG.integers(top + 1, top + inner_h + 1))
    if opening == "left":
        lo = anchor_c + max(1, inner_w - 2)
        hi = anchor_c + inner_w
    else:
        lo = anchor_c
        hi = anchor_c + min(2, inner_w)
    seed_c = int(_RNG.integers(lo, hi))
    grid[seed_r, seed_c] = int(color)


def _maybe_place_obstacle(grid: np.ndarray, top: int, anchor_c: int, inner_h: int, inner_w: int, opening: str) -> tuple[int, int] | None:
    row = int(_RNG.integers(top + 1, top + inner_h + 1))
    if opening == "left":
        if anchor_c <= 0:
            return None
        col = int(_RNG.integers(0, anchor_c))
    else:
        start = anchor_c + inner_w
        if start >= grid.shape[1]:
            return None
        col = int(_RNG.integers(start, grid.shape[1]))
    grid[row, col] = OBSTACLE
    return row, col


def _gen_attempt() -> np.ndarray | None:
    inner_h = int(_RNG.integers(3, 6))
    inner_w = int(_RNG.integers(4, 7))
    min_w = 2 * inner_w + 8
    width = int(_RNG.integers(min_w, 25))
    height = int(_RNG.integers(inner_h + 4, 19))

    sep_lo = inner_w + 4
    sep_hi = width - inner_w - 5
    if sep_lo > sep_hi:
        return None
    sep_col = int(_RNG.integers(sep_lo, sep_hi + 1))

    left_anchor_max = sep_col - inner_w - 2
    right_anchor_min = sep_col + 2
    right_anchor_max = width - inner_w - 1
    if left_anchor_max < 2 or right_anchor_min > right_anchor_max:
        return None

    left_anchor = int(_RNG.integers(2, left_anchor_max + 1))
    right_anchor = int(_RNG.integers(right_anchor_min, right_anchor_max + 1))

    wall_h = inner_h + 2
    if height - wall_h - 1 <= 0:
        return None
    left_top = int(_RNG.integers(1, height - wall_h))
    right_top = int(_RNG.integers(1, height - wall_h))

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:, sep_col] = SEPARATOR
    _draw_left_wall(grid, left_top, left_anchor, inner_h, inner_w)
    _draw_right_wall(grid, right_top, right_anchor, inner_h, inner_w)

    left_color, right_color = _RNG.choice(SEED_COLORS, size=2, replace=False)
    _place_seed(grid, left_top, left_anchor, inner_h, inner_w, "left", int(left_color))
    _place_seed(grid, right_top, right_anchor, inner_h, inner_w, "right", int(right_color))

    patterns = ((1, 0), (0, 1), (1, 1), (0, 0))
    weights = np.array((0.3, 0.3, 0.3, 0.1))
    add_left, add_right = patterns[int(_RNG.choice(len(patterns), p=weights))]
    if add_left:
        _maybe_place_obstacle(grid, left_top, left_anchor, inner_h, inner_w, "left")
    if add_right:
        _maybe_place_obstacle(grid, right_top, right_anchor, inner_h, inner_w, "right")

    try:
        bg, _wall, left, right = _panel_infos(grid)
        out = main(grid)
    except AssertionError:
        return None

    if left["offsets"] == right["offsets"]:
        return None
    if int((out == left["seed_color"]).sum()) < 6 or int((out == right["seed_color"]).sum()) < 6:
        return None
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if left["seed_color"] == right["seed_color"] or bg != BACKGROUND:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
