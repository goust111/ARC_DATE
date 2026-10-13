from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds, neighbors4


def _bool_mask(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=bool)


ARROW_SPECS: dict[str, dict[str, object]] = {
    "up": {
        "mask": _bool_mask(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
            ]
        ),
        "tip": (0, 1),
        "vec": (-1, 0),
    },
    "down": {
        "mask": _bool_mask(
            [
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ]
        ),
        "tip": (3, 1),
        "vec": (1, 0),
    },
    "left": {
        "mask": _bool_mask(
            [
                [0, 1, 0, 0],
                [1, 1, 1, 1],
                [0, 1, 0, 0],
            ]
        ),
        "tip": (1, 0),
        "vec": (0, -1),
    },
    "right": {
        "mask": _bool_mask(
            [
                [0, 0, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 0],
            ]
        ),
        "tip": (1, 3),
        "vec": (0, 1),
    },
}

for _spec in ARROW_SPECS.values():
    _spec["coords"] = np.argwhere(_spec["mask"])


GEN_MIN_SIDE = 14
GEN_MAX_SIDE = 30
GEN_ARROW_COUNTS = (2, 3, 4)
GEN_MIN_WALL_LEN = 3
GEN_MIN_REGION_AREA = 8
GEN_MIN_REGION_GAIN = 6
GEN_MAX_ATTEMPTS = 400

PALETTE = [c for c in Color.NOT_BLACK if c != Color.YELLOW]


def _outer_ring_colors(grid: np.ndarray) -> set[int]:
    return set(
        map(
            int,
            np.concatenate(
                [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
            ),
        )
    )


def _tight_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _match_arrow(mask: np.ndarray) -> str:
    for direction, spec in ARROW_SPECS.items():
        if mask.shape == spec["mask"].shape and np.array_equal(mask, spec["mask"]):
            return direction
    raise AssertionError("unrecognized arrow sprite")


def _ray_cells(shape: tuple[int, int], tip: tuple[int, int], vec: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    dr, dc = map(int, vec)
    r, c = int(tip[0] + dr), int(tip[1] + dc)
    cells: list[tuple[int, int]] = []
    while 0 < r < height - 1 and 0 < c < width - 1:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _extract_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    interior = grid[1:-1, 1:-1]
    arrow_mask = (interior != Color.BLACK) & (interior != Color.YELLOW)
    comps = find_components(
        arrow_mask.astype(int),
        bg=0,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert comps, "expected at least one arrow"

    arrows: list[dict[str, object]] = []
    seen_dirs: set[str] = set()
    for comp in comps:
        assert int(comp["area"]) == 6, "each arrow must use six pixels"
        tight = _tight_mask(comp["coords"], comp["bbox"])
        direction = _match_arrow(tight)
        assert direction not in seen_dirs, "arrow directions must be unique"
        seen_dirs.add(direction)

        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        tip_r = 1 + r0 + int(ARROW_SPECS[direction]["tip"][0])
        tip_c = 1 + c0 + int(ARROW_SPECS[direction]["tip"][1])
        coords = comp["coords"] + 1
        arrows.append(
            {
                "direction": direction,
                "tip": (tip_r, tip_c),
                "vec": ARROW_SPECS[direction]["vec"],
                "coords": coords,
            }
        )
    return arrows


def _build_blocked(shape: tuple[int, int], arrows: list[dict[str, object]]) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    for arrow in arrows:
        coords = arrow["coords"]
        blocked[coords[:, 0], coords[:, 1]] = True
        for r, c in _ray_cells(shape, arrow["tip"], arrow["vec"]):
            blocked[r, c] = True

    return blocked


def _flood(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    reachable = np.zeros_like(passable, dtype=bool)
    sr, sc = map(int, start)
    if not passable[sr, sc]:
        return reachable

    q: deque[tuple[int, int]] = deque([(sr, sc)])
    reachable[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(passable.shape, nr, nc):
                continue
            if reachable[nr, nc] or not passable[nr, nc]:
                continue
            reachable[nr, nc] = True
            q.append((nr, nc))
    return reachable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend invisible walls from each arrow tip, then flood the seed's enclosed chamber yellow.

    Input structure:
    - The grid has a one-cell rectangular frame on the outer border and a black interior.
    - Inside the frame are several six-pixel monochrome asymmetric T arrows plus exactly one yellow seed pixel.
    - The arrows do not touch each other, and each direction appears at most once.

    Rule:
    - Decode each arrow's pointing direction from which side of the four-cell stem the three-cell crossbar sits closer to.
    - From the arrow tip, extend an invisible straight wall in that direction until the outer frame.
    - Flood-fill four-connected black cells starting from the yellow seed, but do not cross the frame, arrow bodies, or invisible walls; recolor every reachable black cell yellow.

    Key insight:
    - The visible arrows are only wall emitters: the filled region is determined by hidden rays that must be reconstructed before doing the flood fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert 7 <= height <= 30 and 7 <= width <= 30

    frame_colors = _outer_ring_colors(g)
    assert len(frame_colors) == 1, "frame must be a single color"
    frame_color = next(iter(frame_colors))
    assert frame_color not in (Color.BLACK, Color.YELLOW), "frame color must be visible and non-yellow"

    seed_coords = np.argwhere(g == Color.YELLOW)
    assert len(seed_coords) == 1, "expected exactly one yellow seed"
    seed = tuple(map(int, seed_coords[0]))
    assert 0 < seed[0] < height - 1 and 0 < seed[1] < width - 1

    arrows = _extract_arrows(g)
    blocked = _build_blocked(g.shape, arrows)
    assert not blocked[seed], "seed must not lie on a blocker"

    seed_mask = np.zeros(g.shape, dtype=bool)
    seed_mask[seed] = True
    passable = ((g == Color.BLACK) | seed_mask) & ~blocked
    reachable = _flood(passable, seed)

    out = g.copy()
    out[reachable & (g == Color.BLACK)] = Color.YELLOW
    return out


def _arrow_patch(direction: str, color: int) -> np.ndarray:
    mask = ARROW_SPECS[direction]["mask"]
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _candidate_positions(shape: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    mask = ARROW_SPECS[direction]["mask"]
    tip_r, tip_c = map(int, ARROW_SPECS[direction]["tip"])
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(1, height - h):
        for left in range(1, width - w):
            abs_tip = (top + tip_r, left + tip_c)
            wall_len = len(_ray_cells(shape, abs_tip, ARROW_SPECS[direction]["vec"]))
            if wall_len >= GEN_MIN_WALL_LEN:
                candidates.append((top, left))
    random.shuffle(candidates)
    return candidates


def _cell_set(direction: str, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (top + int(r), left + int(c))
        for r, c in ARROW_SPECS[direction]["coords"]
    }


def _touches_forbidden(cells: set[tuple[int, int]], forbidden: np.ndarray) -> bool:
    return any(forbidden[r, c] for r, c in cells)


def _mark_padding(mask: np.ndarray, cells: set[tuple[int, int]], radius: int = 1) -> None:
    height, width = mask.shape
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    mask[nr, nc] = True


def _place_arrows(shape: tuple[int, int], directions: list[str], colors: list[int]) -> list[dict[str, object]] | None:
    body_forbidden = np.zeros(shape, dtype=bool)
    arrows: list[dict[str, object]] = []

    for direction, color in zip(directions, colors):
        placed = None
        for top, left in _candidate_positions(shape, direction):
            cells = _cell_set(direction, top, left)
            if _touches_forbidden(cells, body_forbidden):
                continue

            tip_r = top + int(ARROW_SPECS[direction]["tip"][0])
            tip_c = left + int(ARROW_SPECS[direction]["tip"][1])
            wall = _ray_cells(shape, (tip_r, tip_c), ARROW_SPECS[direction]["vec"])
            if any(w in other["cells"] for other in arrows for w in wall):
                continue

            placed = {
                "direction": direction,
                "color": int(color),
                "top": int(top),
                "left": int(left),
                "tip": (int(tip_r), int(tip_c)),
                "vec": ARROW_SPECS[direction]["vec"],
                "cells": cells,
                "wall": wall,
            }
            arrows.append(placed)
            _mark_padding(body_forbidden, cells, radius=1)
            break
        if placed is None:
            return None

    return arrows


def _build_input_grid(shape: tuple[int, int], frame_color: int, arrows: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[0, :] = frame_color
    grid[-1, :] = frame_color
    grid[:, 0] = frame_color
    grid[:, -1] = frame_color
    for arrow in arrows:
        patch = _arrow_patch(arrow["direction"], arrow["color"])
        mask = ARROW_SPECS[arrow["direction"]]["mask"]
        top = int(arrow["top"])
        left = int(arrow["left"])
        view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        view[mask] = patch[mask]
    return grid


def _blocked_from_layout(shape: tuple[int, int], arrows: list[dict[str, object]]) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    for arrow in arrows:
        for r, c in arrow["cells"]:
            blocked[r, c] = True
        for r, c in arrow["wall"]:
            blocked[r, c] = True
    return blocked


def _naive_blocked(shape: tuple[int, int], arrows: list[dict[str, object]]) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    for arrow in arrows:
        for r, c in arrow["cells"]:
            blocked[r, c] = True
    return blocked


def _seed_options(grid: np.ndarray, arrows: list[dict[str, object]]) -> list[tuple[tuple[int, int], np.ndarray]]:
    blocked = _blocked_from_layout(grid.shape, arrows)
    naive = _naive_blocked(grid.shape, arrows)
    passable = (grid == Color.BLACK) & ~blocked
    comps = find_components(
        passable.astype(int),
        bg=0,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    options: list[tuple[tuple[int, int], np.ndarray]] = []
    for comp in comps:
        area = int(comp["area"])
        if area < GEN_MIN_REGION_AREA:
            continue

        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        preferred = []
        for r, c in coords:
            degree = sum(
                1
                for nr, nc in neighbors4(r, c)
                if in_bounds(passable.shape, nr, nc) and passable[nr, nc]
            )
            if degree >= 2:
                preferred.append((r, c))
        candidates = preferred or coords
        seed = random.choice(candidates)

        actual = _flood(passable, seed)
        naive_fill = _flood((grid == Color.BLACK) & ~naive, seed)
        gain = int(naive_fill.sum()) - int(actual.sum())
        if gain < GEN_MIN_REGION_GAIN:
            continue
        options.append((seed, actual))
    return options


def _sample_directions(count: int) -> list[str]:
    while True:
        directions = random.sample(list(ARROW_SPECS), count)
        if any(d in ("up", "down") for d in directions) and any(d in ("left", "right") for d in directions):
            return directions


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)
    width = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)
    count = random.choice(GEN_ARROW_COUNTS)
    directions = _sample_directions(count)

    colors = random.sample(PALETTE, count + 1)
    frame_color = colors[0]
    arrow_colors = colors[1:]

    arrows = _place_arrows((height, width), directions, arrow_colors)
    if arrows is None:
        return None

    grid = _build_input_grid((height, width), frame_color, arrows)
    options = _seed_options(grid, arrows)
    if not options:
        return None

    seed, filled = random.choice(options)
    grid[seed] = Color.YELLOW

    expected = grid.copy()
    expected[filled & (grid == Color.BLACK)] = Color.YELLOW

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    seed_mask = np.zeros(grid.shape, dtype=bool)
    seed_mask[seed] = True
    no_wall = _flood(
        ((grid == Color.BLACK) | seed_mask) & ~_naive_blocked(grid.shape, arrows),
        seed,
    )
    if int(no_wall.sum()) <= int(filled.sum()):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid invisible-wall chamber")
