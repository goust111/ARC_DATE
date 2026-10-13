from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


BG = 0
OUTLINE_COLOR = 3
INTERSECTION_COLOR = 4
INPUT_COLORS = (1, 2, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expanded_outline(
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = box
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    cells = set()
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                cells.add((r, c))
    return cells


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    signal_infos: dict[int, dict] = {}
    noise_colors: list[int] = []
    color_components: dict[int, list[np.ndarray]] = {}
    signal_pixels: set[tuple[int, int]] = set()

    for color in colors:
        comps = _components_from_mask(g == color)
        color_components[color] = comps
        if len(comps) == 1:
            coords = comps[0]
            info = {
                "color": int(color),
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": _bbox(coords),
            }
            signal_infos[color] = info
            signal_pixels.update((int(r), int(c)) for r, c in coords)
        else:
            noise_colors.append(int(color))

    return {
        "color_components": color_components,
        "signal_infos": signal_infos,
        "noise_colors": noise_colors,
        "signal_pixels": signal_pixels,
    }


def _solve(grid: np.ndarray, *, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    info = _analyze_grid(g)
    out = g.copy()

    for color in info["noise_colors"]:
        out[g == color] = BG

    outline_cells: set[tuple[int, int]] = set()
    for color in sorted(info["signal_infos"]):
        sig = info["signal_infos"][color]
        if sig["area"] != 2:
            continue
        for r, c in _expanded_outline(sig["bbox"], g.shape):
            if out[r, c] == BG:
                out[r, c] = OUTLINE_COLOR
                outline_cells.add((r, c))

    ray_hits_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    ray_paths: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    first_blockers: dict[tuple[int, int, int], tuple[int, int] | None] = {}
    signal_pixels = info["signal_pixels"]

    for color in sorted(info["signal_infos"]):
        sig = info["signal_infos"][color]
        if sig["area"] != 1:
            continue
        origin = tuple(map(int, sig["coords"][0]))
        for dr, dc in DIRS4:
            path: list[tuple[int, int]] = []
            r, c = origin
            blocker = None
            while True:
                r += dr
                c += dc
                if not _in_bounds(g.shape, r, c):
                    break
                if (r, c) in signal_pixels:
                    blocker = (r, c)
                    break
                path.append((r, c))
                ray_hits_by_color[color].add((r, c))
                out[r, c] = color
            ray_paths[(color, dr, dc)] = path
            first_blockers[(color, dr, dc)] = blocker

    crossings: set[tuple[int, int]] = set()
    for cells in ray_hits_by_color.values():
        crossings |= cells & outline_cells
    for r, c in crossings:
        out[r, c] = INTERSECTION_COLOR

    if not with_meta:
        return out
    return out, {
        **info,
        "outline_cells": outline_cells,
        "ray_hits_by_color": ray_hits_by_color,
        "ray_paths": ray_paths,
        "first_blockers": first_blockers,
        "crossings": crossings,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter single-component colors, outline signal dominoes, extend singleton rays, and recolor ray/outline crossings yellow.

    Input structure:
    - The grid is a black background with several small monochrome 4-connected objects using multiple non-black colors.
    - Some colors appear in exactly one 4-connected component; those colors are the signal colors.
    - Other colors are fragmented into multiple disconnected components and act as noise.

    Rule:
    - For each non-black color, count its 4-connected components. Keep colors with exactly one component and erase all pixels of colors with more than one component.
    - For every remaining signal object of area 2, expand its bounding box by one cell in every direction (clipped to the grid) and draw a green border on the perimeter wherever the current cell is black.
    - For every remaining signal object of area 1, cast rays upward, downward, leftward, and rightward in the object's own color until the grid edge or the original pixel of another signal object; the blocking signal pixel is not included.
    - After all outlines and rays are drawn, recolor every cell that belongs to both a ray and a green border to yellow.

    Key insight:
    - Connectivity decides which colors are meaningful, object area decides which transformation branch they follow, and yellow cells only emerge where the singleton-ray branch later overlaps the domino-outline branch.
    """
    return _solve(grid, with_meta=False)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _can_place(grid: np.ndarray, coords: Iterable[tuple[int, int]]) -> bool:
    h, w = grid.shape
    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in coord_set:
        if not _in_bounds((h, w), r, c) or grid[r, c] != BG:
            return False
    for r, c in coord_set:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not _in_bounds((h, w), nr, nc) or (nr, nc) in coord_set:
                    continue
                if grid[nr, nc] != BG:
                    return False
    return True


def _paint(grid: np.ndarray, coords: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


BLOCKER_SHAPES = {
    "up": (
        ((0, 0), (0, -1), (0, 1)),
        ((0, 0), (-1, 0), (0, -1)),
        ((0, 0), (-1, 0), (0, 1)),
        ((0, 0), (0, -1), (-1, -1), (-1, 0)),
        ((0, 0), (0, 1), (-1, 0), (-1, 1)),
    ),
    "down": (
        ((0, 0), (0, -1), (0, 1)),
        ((0, 0), (1, 0), (0, -1)),
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (0, -1), (1, -1), (1, 0)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ),
    "left": (
        ((0, 0), (-1, 0), (1, 0)),
        ((0, 0), (-1, 0), (0, -1)),
        ((0, 0), (1, 0), (0, -1)),
        ((0, 0), (-1, 0), (-1, -1), (0, -1)),
        ((0, 0), (1, 0), (1, -1), (0, -1)),
    ),
    "right": (
        ((0, 0), (-1, 0), (1, 0)),
        ((0, 0), (-1, 0), (0, 1)),
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (-1, 0), (-1, 1), (0, 1)),
        ((0, 0), (1, 0), (1, 1), (0, 1)),
    ),
}


def _try_place_blocker(
    rng: np.random.Generator,
    grid: np.ndarray,
    single_rc: tuple[int, int],
    color: int,
    directions: tuple[str, ...],
) -> list[tuple[int, int]] | None:
    sr, sc = single_rc
    dir_order = list(directions)
    rng.shuffle(dir_order)
    for side in dir_order:
        distances = [3, 4, 5]
        rng.shuffle(distances)
        shapes = list(BLOCKER_SHAPES[side])
        rng.shuffle(shapes)
        for dist in distances:
            if side == "up":
                anchor = (sr - dist, sc)
            elif side == "down":
                anchor = (sr + dist, sc)
            elif side == "left":
                anchor = (sr, sc - dist)
            else:
                anchor = (sr, sc + dist)
            for offsets in shapes:
                coords = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets]
                if _can_place(grid, coords):
                    _paint(grid, coords, color)
                    return coords
    return None


def _random_noise_shape(rng: np.random.Generator, require_domino: bool) -> tuple[tuple[int, int], ...]:
    if require_domino:
        return ((0, 0), (0, 1)) if bool(rng.integers(0, 2)) else ((0, 0), (1, 0))
    shapes = [
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
    ]
    return shapes[_randint(rng, 0, len(shapes) - 1)]


def _random_place_noise_component(
    rng: np.random.Generator,
    grid: np.ndarray,
    color: int,
    *,
    require_domino: bool,
    forbidden_rows: set[int],
    forbidden_cols: set[int],
    forbidden_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    for _ in range(200):
        offsets = _random_noise_shape(rng, require_domino=require_domino)
        rs = [dr for dr, _ in offsets]
        cs = [dc for _, dc in offsets]
        r0 = _randint(rng, -min(rs), h - 1 - max(rs))
        c0 = _randint(rng, -min(cs), w - 1 - max(cs))
        coords = [(r0 + dr, c0 + dc) for dr, dc in offsets]
        if any(r in forbidden_rows or c in forbidden_cols for r, c in coords):
            continue
        if any((r, c) in forbidden_cells for r, c in coords):
            continue
        if _can_place(grid, coords):
            _paint(grid, coords, color)
            return coords
    return None


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, 13, 18)
    w = _randint(rng, 13, 18)
    grid = np.zeros((h, w), dtype=int)
    single_color, domino_color, large_color, noise_color = map(
        int, rng.choice(np.array(INPUT_COLORS, dtype=int), size=4, replace=False)
    )

    orient = "h" if bool(rng.integers(0, 2)) else "v"
    gap = _randint(rng, 5, 6)

    if orient == "h":
        row = _randint(rng, 2, h - 3)
        single_left = bool(rng.integers(0, 2))
        if single_left:
            sc = _randint(rng, 2, w - gap - 4)
            dc = sc + gap
        else:
            sc = _randint(rng, gap + 2, w - 3)
            dc = sc - gap
        single = [(row, sc)]
        domino = [(row, dc), (row, dc + 1)]
        corridor_cols = (
            list(range(sc + 2, dc - 1)) if sc < dc else list(range(dc + 3, sc - 1))
        )
        noise_corridor = [(row, corridor_cols[_randint(rng, 0, len(corridor_cols) - 1)])]
        if not (_can_place(grid, single) and _can_place(grid, domino) and _can_place(grid, noise_corridor)):
            return None
        _paint(grid, single, single_color)
        _paint(grid, domino, domino_color)
        _paint(grid, noise_corridor, noise_color)
        blocker = _try_place_blocker(rng, grid, single[0], large_color, ("up", "down"))
    else:
        col = _randint(rng, 2, w - 3)
        single_top = bool(rng.integers(0, 2))
        if single_top:
            sr = _randint(rng, 2, h - gap - 4)
            dr = sr + gap
        else:
            sr = _randint(rng, gap + 2, h - 3)
            dr = sr - gap
        single = [(sr, col)]
        domino = [(dr, col), (dr + 1, col)]
        corridor_rows = (
            list(range(sr + 2, dr - 1)) if sr < dr else list(range(dr + 3, sr - 1))
        )
        noise_corridor = [(corridor_rows[_randint(rng, 0, len(corridor_rows) - 1)], col)]
        if not (_can_place(grid, single) and _can_place(grid, domino) and _can_place(grid, noise_corridor)):
            return None
        _paint(grid, single, single_color)
        _paint(grid, domino, domino_color)
        _paint(grid, noise_corridor, noise_color)
        blocker = _try_place_blocker(rng, grid, single[0], large_color, ("left", "right"))

    if blocker is None:
        return None

    domino_outline_zone = _expanded_outline(_bbox(np.array(domino, dtype=int)), grid.shape)
    forbidden_rows = {single[0][0]}
    forbidden_cols = {single[0][1]}

    remote_noise = _random_place_noise_component(
        rng,
        grid,
        noise_color,
        require_domino=True,
        forbidden_rows=forbidden_rows,
        forbidden_cols=forbidden_cols,
        forbidden_cells=domino_outline_zone,
    )
    if remote_noise is None:
        return None

    if bool(rng.integers(0, 100) < 65):
        extra_noise = _random_place_noise_component(
            rng,
            grid,
            noise_color,
            require_domino=False,
            forbidden_rows=forbidden_rows,
            forbidden_cols=forbidden_cols,
            forbidden_cells=domino_outline_zone,
        )
        if extra_noise is None:
            return None

    if not _validate_instance(grid):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve(grid, with_meta=True)
    except AssertionError:
        return False

    signal_infos = list(meta["signal_infos"].values())
    if len(signal_infos) != 3:
        return False

    areas = sorted(info["area"] for info in signal_infos)
    if areas[0] != 1 or areas[1] != 2 or areas[2] < 3:
        return False

    if len(meta["noise_colors"]) != 1:
        return False

    noise_color = meta["noise_colors"][0]
    if len(meta["color_components"][noise_color]) < 2:
        return False

    if noise_color in np.unique(out):
        return False

    single_info = [info for info in signal_infos if info["area"] == 1][0]
    domino_info = [info for info in signal_infos if info["area"] == 2][0]
    large_info = [info for info in signal_infos if info["area"] >= 3][0]

    signal_pixels = meta["signal_pixels"]
    for r, c in signal_pixels:
        if out[r, c] != grid[r, c]:
            return False

    if not meta["outline_cells"] or not meta["crossings"]:
        return False

    expected_green = meta["outline_cells"] - meta["crossings"]
    actual_green = {tuple(map(int, rc)) for rc in np.argwhere(out == OUTLINE_COLOR)}
    if actual_green != expected_green:
        return False

    actual_yellow = {tuple(map(int, rc)) for rc in np.argwhere(out == INTERSECTION_COLOR)}
    if actual_yellow != meta["crossings"]:
        return False

    ray_union = set()
    for color, cells in meta["ray_hits_by_color"].items():
        if color != single_info["color"]:
            return False
        ray_union |= cells
    if not ray_union:
        return False

    for cell in ray_union:
        if cell in meta["crossings"]:
            if out[cell] != INTERSECTION_COLOR:
                return False
        elif out[cell] != single_info["color"]:
            return False

    interaction_dirs = 0
    blocked_dirs = 0
    noise_on_interaction = False
    blocked_by_large = False
    large_pixels = {tuple(map(int, rc)) for rc in large_info["coords"]}
    for key, path in meta["ray_paths"].items():
        color, _dr, _dc = key
        if color != single_info["color"]:
            return False
        blocker = meta["first_blockers"][key]
        crossed = [cell for cell in path if cell in meta["outline_cells"]]
        if crossed:
            interaction_dirs += 1
            first_cross_idx = path.index(crossed[0])
            if any(grid[r, c] == noise_color for r, c in path[: first_cross_idx + 1]):
                noise_on_interaction = True
        if blocker is not None:
            blocked_dirs += 1
            if blocker in large_pixels:
                blocked_by_large = True

    if interaction_dirs < 1 or blocked_dirs < 2:
        return False
    if not noise_on_interaction or not blocked_by_large:
        return False

    domino_outline = _expanded_outline(domino_info["bbox"], grid.shape)
    if not meta["crossings"].issubset(domino_outline):
        return False

    if np.array_equal(grid, out):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connectivity-filtered ray/outline input")
