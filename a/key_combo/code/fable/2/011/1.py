from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    for parent in here.parents:
        new_root = parent / "new"
        utils_path = new_root / "solvers" / "arc_utils.py"
        if utils_path.exists():
            if str(new_root) not in sys.path:
                sys.path.insert(0, str(new_root))
            from solvers.arc_utils import Color, bbox_coords, components_mask  # type: ignore

            return Color, bbox_coords, components_mask
    raise ImportError("Could not locate new/solvers/arc_utils.py from this file")


Color, bbox_coords, components_mask = _load_arc_utils()


BG = Color.BLACK
PALETTE = [c for c in Color.NOT_BLACK]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_separator_row(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for r in range(h):
        row = grid[r]
        if not np.all(row == row[0]):
            continue
        color = int(row[0])
        if color == BG:
            continue
        if int(np.count_nonzero(grid == color)) == w:
            candidates.append(r)
    assert len(candidates) == 1, "expected exactly one unique separator row"
    return int(candidates[0])


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _find_separator_row(g)
    top = g[:sep].copy()
    bottom = g[sep + 1 :].copy()
    assert top.shape == bottom.shape and top.shape[0] > 0

    top_colors = sorted(set(map(int, np.unique(top))) - {BG})
    terrain_colors = sorted(set(map(int, np.unique(bottom))) - {BG})
    assert len(top_colors) == 1, "expected one sprite color in the top panel"
    assert len(terrain_colors) == 1, "expected one terrain color in the bottom panel"

    sprite_color = int(top_colors[0])
    terrain_color = int(terrain_colors[0])
    terrain_mask = bottom == terrain_color
    sprites = components_mask(top == sprite_color, connectivity=4)
    assert sprites, "expected at least one sprite"

    for coords in sprites:
        assert not bool(terrain_mask[coords[:, 0], coords[:, 1]].any()), (
            "sprites must start in empty cells of the terrain panel"
        )

    return top, bottom, sprite_color, terrain_color, terrain_mask, sprites


def _drop_distance(coords: np.ndarray, occupied: np.ndarray) -> int:
    h = occupied.shape[0]
    dr = 0
    while True:
        shifted = coords.copy()
        shifted[:, 0] += dr + 1
        if bool((shifted[:, 0] >= h).any()):
            return dr
        if bool(occupied[shifted[:, 0], shifted[:, 1]].any()):
            return dr
        dr += 1


def _touches_terrain(coords: np.ndarray, terrain_mask: np.ndarray) -> bool:
    h, w = terrain_mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and bool(terrain_mask[nr, nc]):
                return True
    return False


def _support_kinds(coords: np.ndarray, terrain_mask: np.ndarray, settled_mask: np.ndarray) -> set[str]:
    h, w = terrain_mask.shape
    kinds: set[str] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr == h - 1:
            kinds.add("bottom")
        else:
            if bool(terrain_mask[rr + 1, cc]):
                kinds.add("terrain")
            if bool(settled_mask[rr + 1, cc]):
                kinds.add("sprite")
        for nr, nc in ((rr - 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and bool(terrain_mask[nr, nc]):
                kinds.add("terrain_side")
    return kinds


def _solve_with_info(grid: np.ndarray):
    _top, bottom, sprite_color, terrain_color, terrain_mask, sprites = _parse_input(grid)
    h, w = bottom.shape
    occupied = terrain_mask.copy()
    out = bottom.copy()
    placed_mask = np.zeros((h, w), dtype=bool)

    order = sorted(
        sprites,
        key=lambda coords: (
            int(bbox_coords(coords)[2]),
            int(bbox_coords(coords)[0]),
            -int(bbox_coords(coords)[1]),
        ),
        reverse=True,
    )

    placements: list[dict] = []
    for coords in order:
        dr = _drop_distance(coords, occupied)
        final_coords = coords.copy()
        final_coords[:, 0] += dr
        assert not bool(occupied[final_coords[:, 0], final_coords[:, 1]].any())

        touches = _touches_terrain(final_coords, terrain_mask)
        support = _support_kinds(final_coords, terrain_mask, placed_mask)
        color = terrain_color if touches else sprite_color

        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        placed_mask[final_coords[:, 0], final_coords[:, 1]] = True
        out[final_coords[:, 0], final_coords[:, 1]] = color
        placements.append(
            {
                "initial": coords.copy(),
                "final": final_coords,
                "touches_terrain": touches,
                "support": support,
                "color": color,
            }
        )

    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop rigid sprites from the upper panel into the lower terrain panel, then recolor only the sprites that touch the original terrain.

    Input structure:
    - A single-color separator row splits the grid into two equal-height panels, and that separator color appears nowhere else.
    - The upper panel is a black background with several 4-connected sprites of one shared color.
    - The lower panel is a black background with one terrain color forming bottom-grown solid columns.

    Rule:
    - Remove the separator row and reinterpret the upper panel as a moving layer aligned to the lower panel's coordinates.
    - Drop each sprite as a rigid body straight downward until the next step would hit the original terrain, a previously settled sprite, or the bottom edge; lower starting sprites settle first when columns compete.
    - Paint the settled sprites onto the terrain panel, but recolor an entire sprite to the terrain color only if any of its final pixels is 4-neighbor adjacent to the original terrain.

    Key insight:
    - The two panels are not merged as patterns; they are aligned motion and terrain layers, and the final color tells whether a settled sprite ever touches the original terrain layer.
    """
    out, _placements = _solve_with_info(grid)
    return out


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(1, 4))
    w = int(rng.integers(1, 5))
    max_area = min(h * w, 6)
    min_area = min(2, max_area)
    target = int(rng.integers(min_area, max_area + 1))

    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    frontier = [start]
    while len(cells) < target:
        base = frontier[int(rng.integers(0, len(frontier)))]
        candidates = []
        for nr, nc in _neighbors4(*base):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                candidates.append((nr, nc))
        if not candidates:
            frontier = list(cells)
            continue
        new_cell = candidates[int(rng.integers(0, len(candidates)))]
        cells.add(new_cell)
        frontier.append(new_cell)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        shape[r - r0, c - c0] = 1

    if bool(rng.integers(0, 2)):
        shape = np.fliplr(shape)
    if bool(rng.integers(0, 2)):
        shape = np.flipud(shape)
    return shape


def _sample_distinct_colors(rng: np.random.Generator) -> tuple[int, int, int]:
    colors = rng.choice(PALETTE, size=3, replace=False)
    return int(colors[0]), int(colors[1]), int(colors[2])


def _paint_terrain(heights: list[int], color: int, h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for c, height in enumerate(heights):
        if height <= 0:
            continue
        grid[h - height :, c] = color
    return grid


def _max_safe_row(shape: np.ndarray, heights: list[int], start_col: int, panel_h: int) -> int:
    limit = panel_h - 1
    for r, c in np.argwhere(shape == 1):
        col = start_col + int(c)
        height = int(heights[col])
        terrain_top = panel_h if height == 0 else panel_h - height
        limit = min(limit, terrain_top - int(r) - 1)
    return int(limit - shape.shape[0] + 1 + shape.shape[0] - 1)  # equivalent to limit for top-left rows


def _place_shape(panel: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(shape == 1)
    for r, c in coords:
        panel[top + int(r), left + int(c)] = color


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(mask.tolist() + [False]):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            runs.append((start, idx))
            start = None
    return runs


def _drop_on_empty_support(shape: np.ndarray, left: int, panel_h: int) -> set[tuple[int, int]]:
    coords = np.argwhere(shape == 1)
    max_r = int(coords[:, 0].max())
    dr = panel_h - 1 - max_r
    return {(int(r + dr), left + int(c)) for r, c in coords}


def _reject_if_touching_terrain(final_cells: set[tuple[int, int]], terrain_mask: np.ndarray) -> bool:
    h, w = terrain_mask.shape
    for r, c in final_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(terrain_mask[nr, nc]):
                return True
    return False


def _build_heights(
    rng: np.random.Generator,
    panel_h: int,
    panel_w: int,
    shaft_start: int,
    shaft_w: int,
    hill_start: int,
    hill_w: int,
) -> list[int]:
    max_hill = panel_h - 4
    plateau = int(rng.integers(2, max_hill + 1))
    heights = [0] * panel_w

    for c in range(hill_start, hill_start + hill_w):
        heights[c] = plateau

    for c in range(panel_w):
        if shaft_start <= c < shaft_start + shaft_w:
            heights[c] = 0
            continue
        if hill_start <= c < hill_start + hill_w:
            if rng.random() < 0.35 and hill_w > 2 and c not in (hill_start, hill_start + hill_w - 1):
                heights[c] = max(plateau - 1, 1)
            continue
        if rng.random() < 0.35:
            heights[c] = int(rng.integers(1, max(2, plateau + 1)))

    return heights


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sprite_color, terrain_color, separator_color = _sample_distinct_colors(rng)

    lower_shape = _random_shape(rng)
    upper_shape = _random_shape(rng)
    ground_shape = _random_shape(rng)
    extra_shape = _random_shape(rng) if bool(rng.integers(0, 2)) else None

    stack_inner_w = max(lower_shape.shape[1], upper_shape.shape[1])
    shaft_w = stack_inner_w + 2 + int(rng.integers(0, 3))
    hill_w = ground_shape.shape[1] + 1 + int(rng.integers(0, 3))
    extra_w = 0 if extra_shape is None else extra_shape.shape[1] + 1

    min_w = max(10, shaft_w + hill_w + extra_w + 3)
    max_w = 18
    if min_w > max_w:
        return None

    panel_h = int(rng.integers(8, 14))
    panel_w = int(rng.integers(min_w, max_w + 1))

    regions = []
    cursor = 0
    order = [("shaft", shaft_w), ("hill", hill_w)]
    if extra_shape is not None:
        order.append(("extra", extra_shape.shape[1] + 1 + int(rng.integers(0, 2))))
    rng.shuffle(order)
    for name, width in order:
        gap = int(rng.integers(1, 3 if regions else 2))
        cursor += gap
        if cursor + width > panel_w:
            return None
        slack = int(rng.integers(0, max(1, panel_w - cursor - width - sum(w for _, w in order[len(regions) + 1 :]) - (len(order) - len(regions) - 1))))
        start = cursor + slack
        regions.append((name, start, width))
        cursor = start + width

    region_map = {name: (start, width) for name, start, width in regions}
    shaft_start, shaft_w = region_map["shaft"]
    hill_start, hill_w = region_map["hill"]
    extra_region = region_map.get("extra")

    heights = _build_heights(rng, panel_h, panel_w, shaft_start, shaft_w, hill_start, hill_w)
    terrain = _paint_terrain(heights, terrain_color, panel_h, panel_w)
    terrain_mask = terrain == terrain_color

    top = np.zeros((panel_h, panel_w), dtype=int)

    shaft_inner_l = shaft_start + 1
    shaft_inner_r = shaft_start + shaft_w - 1
    if shaft_inner_r - shaft_inner_l < stack_inner_w:
        return None

    lower_left_min = shaft_inner_l
    lower_left_max = shaft_inner_r - lower_shape.shape[1]
    lower_left = int(rng.integers(lower_left_min, lower_left_max + 1))

    overlap_l = max(shaft_inner_l, lower_left - upper_shape.shape[1] + 1)
    overlap_r = min(shaft_inner_r - upper_shape.shape[1], lower_left + lower_shape.shape[1] - 1)
    if overlap_l > overlap_r:
        return None
    upper_left = int(rng.integers(overlap_l, overlap_r + 1))

    lower_top_max = max(2, panel_h // 2 - lower_shape.shape[0])
    lower_top = int(rng.integers(2, lower_top_max + 1))
    upper_top_max = lower_top - upper_shape.shape[0] - 1
    if upper_top_max < 0:
        return None
    upper_top = int(rng.integers(0, upper_top_max + 1))

    hill_left_min = hill_start
    hill_left_max = hill_start + hill_w - ground_shape.shape[1]
    if hill_left_min > hill_left_max:
        return None
    ground_left = int(rng.integers(hill_left_min, hill_left_max + 1))

    ground_max_row = panel_h - heights[ground_left] - ground_shape.shape[0] - 1
    if ground_max_row < 0:
        return None
    for local_r, local_c in np.argwhere(ground_shape == 1):
        col = ground_left + int(local_c)
        column_limit = panel_h - heights[col] - int(local_r) - 1
        ground_max_row = min(ground_max_row, column_limit)
    if ground_max_row < 0:
        return None
    ground_top = int(rng.integers(0, min(ground_max_row, panel_h // 2) + 1))

    _place_shape(top, lower_shape, lower_top, lower_left, sprite_color)
    _place_shape(top, upper_shape, upper_top, upper_left, sprite_color)
    _place_shape(top, ground_shape, ground_top, ground_left, sprite_color)

    if extra_shape is not None and extra_region is not None:
        extra_start, extra_w_total = extra_region
        extra_left_min = extra_start
        extra_left_max = extra_start + extra_w_total - extra_shape.shape[1]
        if extra_left_min <= extra_left_max:
            extra_left = int(rng.integers(extra_left_min, extra_left_max + 1))
            extra_kind = "terrain"
            if all(h == 0 for h in heights[extra_start : extra_start + extra_w_total]):
                extra_kind = "empty"
            if extra_kind == "terrain":
                extra_max_row = panel_h - extra_shape.shape[0] - 1
                for local_r, local_c in np.argwhere(extra_shape == 1):
                    col = extra_left + int(local_c)
                    extra_max_row = min(extra_max_row, panel_h - heights[col] - int(local_r) - 1)
                if extra_max_row >= 0:
                    extra_top = int(rng.integers(0, min(extra_max_row, panel_h // 2) + 1))
                    _place_shape(top, extra_shape, extra_top, extra_left, sprite_color)
            else:
                extra_top = int(rng.integers(0, max(1, panel_h // 2)))
                _place_shape(top, extra_shape, extra_top, extra_left, sprite_color)

    if not np.all((top == 0) | (terrain == 0)):
        return None

    full = np.vstack(
        [
            top,
            np.full((1, panel_w), separator_color, dtype=int),
            terrain,
        ]
    )

    try:
        out, placements = _solve_with_info(full)
    except AssertionError:
        return None

    if out.shape != terrain.shape:
        return None
    if len(placements) < 3:
        return None
    if not any(bool(p["touches_terrain"]) for p in placements):
        return None
    if not any(not bool(p["touches_terrain"]) for p in placements):
        return None
    if not any("sprite" in p["support"] for p in placements):
        return None
    if not any("bottom" in p["support"] for p in placements):
        return None

    sprite_outputs = components_mask(out == sprite_color, connectivity=4)
    terrain_outputs = components_mask(out == terrain_color, connectivity=4)
    if not sprite_outputs or not terrain_outputs:
        return None

    if _reject_if_touching_terrain(_drop_on_empty_support(lower_shape, lower_left, panel_h), terrain_mask):
        return None

    return full


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-panel gravity case")
