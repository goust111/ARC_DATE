from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
ACTION_COLORS = (1, 2, 3, 4)  # blue, red, green, yellow
SATELLITE_COLORS = (5, 6, 7, 9)
SIDE_ORDER = ("top", "right", "bottom", "left")
PUSH_VECTORS = {
    1: (-1, 0),  # blue -> up
    2: (0, 1),   # red -> right
    3: (1, 0),   # green -> down
}

GEN_RETRIES = 400
PLACE_RETRIES = 80
SHAPE_LIBRARY = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, *, monochromatic: bool) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BLACK:
                continue
            seed_color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) == BLACK:
                        continue
                    if monochromatic and int(grid[nr, nc]) != seed_color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "coords": coords_arr,
                    "bbox": _bbox_from_coords(coords_arr),
                    "colors": colors,
                    "color": seed_color if len(colors) == 1 else None,
                }
            )
    return components


def _central_and_satellites(grid: np.ndarray) -> tuple[dict, list[dict]]:
    comps = _find_components(grid, monochromatic=False)
    central = [comp for comp in comps if len(comp["colors"]) >= 2]
    assert len(central) == 1, "expected exactly one multicolor central component"
    satellites = [comp for comp in comps if comp is not central[0]]
    assert len(satellites) == 4, "expected exactly four satellite components"
    assert all(len(comp["colors"]) == 1 for comp in satellites)
    return central[0], satellites


def _stretch_border(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    out = grid.copy()
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    assert 0 < r0 < r1 < h - 1 and 0 < c0 < c1 < w - 1

    out[r0 - 1, c0 : c1 + 1] = grid[r0, c0 : c1 + 1]
    out[r1 + 1, c0 : c1 + 1] = grid[r1, c0 : c1 + 1]
    out[r0 : r1 + 1, c0 - 1] = grid[r0 : r1 + 1, c0]
    out[r0 : r1 + 1, c1 + 1] = grid[r0 : r1 + 1, c1]
    out[r0 - 1, c0 - 1] = BLACK
    out[r0 - 1, c1 + 1] = BLACK
    out[r1 + 1, c0 - 1] = BLACK
    out[r1 + 1, c1 + 1] = BLACK

    border_cells: list[tuple[int, int, int]] = []
    for c in range(c0, c1 + 1):
        top_color = int(grid[r0, c])
        bottom_color = int(grid[r1, c])
        if top_color != BLACK:
            border_cells.append((r0 - 1, c, top_color))
        if bottom_color != BLACK:
            border_cells.append((r1 + 1, c, bottom_color))
    for r in range(r0, r1 + 1):
        left_color = int(grid[r, c0])
        right_color = int(grid[r, c1])
        if left_color != BLACK:
            border_cells.append((r, c0 - 1, left_color))
        if right_color != BLACK:
            border_cells.append((r, c1 + 1, right_color))

    return out, border_cells


def _nearest_border_info(
    coords: np.ndarray,
    border_cells: list[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], bool]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    scored = [
        (abs(center_r - r) + abs(center_c - c), r, c, color)
        for r, c, color in border_cells
    ]
    best = min(scored, key=lambda item: (item[0], item[1], item[2]))
    tie_count = sum(
        1
        for dist, r, c, _ in scored
        if dist == best[0] and r == best[1] and c == best[2]
    )
    min_count = sum(1 for dist, *_ in scored if dist == best[0])
    return (int(best[1]), int(best[2]), int(best[3])), min_count == 1 and tie_count == 1


def _mirror_horizontally(coords: np.ndarray) -> np.ndarray:
    r0, c0, _r1, c1 = _bbox_from_coords(coords)
    mirrored = coords.copy()
    mirrored[:, 1] = c0 + c1 - mirrored[:, 1]
    return mirrored


def _push_to_edge(coords: np.ndarray, action_color: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    shifted = coords.copy()
    if action_color == 1:
        steps = int(coords[:, 0].min())
        shifted[:, 0] -= steps
    elif action_color == 2:
        steps = int(w - 1 - coords[:, 1].max())
        shifted[:, 1] += steps
    elif action_color == 3:
        steps = int(h - 1 - coords[:, 0].max())
        shifted[:, 0] += steps
    else:
        raise AssertionError("yellow does not push")
    return shifted


def _swept_cells(coords: np.ndarray, action_color: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    if action_color == 4:
        return set()
    h, w = shape
    if action_color == 1:
        steps, dr, dc = int(coords[:, 0].min()), -1, 0
    elif action_color == 2:
        steps, dr, dc = int(w - 1 - coords[:, 1].max()), 0, 1
    else:
        steps, dr, dc = int(h - 1 - coords[:, 0].max()), 1, 0
    own = {tuple(cell) for cell in coords.tolist()}
    swept: set[tuple[int, int]] = set()
    for step in range(1, steps + 1):
        for r, c in coords:
            loc = (int(r + dr * step), int(c + dc * step))
            if loc not in own:
                swept.add(loc)
    return swept


def _apply_satellite_action(coords: np.ndarray, action_color: int, shape: tuple[int, int]) -> np.ndarray:
    if action_color == 4:
        return _mirror_horizontally(coords)
    return _push_to_edge(coords, action_color, shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the central multicolor pattern into a one-cell command border, then move or mirror each satellite using the nearest border color.

    Input structure:
    - The grid is mostly black.
    - One connected multicolor central pattern sits away from the canvas edge.
    - Four small monochrome satellite objects sit outside the central gap on different sides.

    Rule:
    - Find the unique multicolor central component and its bounding box.
    - Add a one-cell stretched border around that box by copying the top row, bottom row, left column, and right column outward while keeping the four new border corners black.
    - For each satellite, measure Manhattan distance from its bounding-box center to every non-black stretched-border cell; use the closest cell, breaking ties by smaller row then smaller column.
    - Blue means push the whole satellite upward to row 0, red means push it rightward to the last column, green means push it downward to the last row, and yellow means mirror it horizontally inside its own bounding box without moving that box.
    - Clear pushed satellites from their old locations, keep mirrored satellites in place, and leave the central pattern plus stretched border unchanged.

    Key insight:
    - The stretched border is a spatial instruction map: it converts the central pattern's edge colors into action commands for the outer satellites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    central, satellites = _central_and_satellites(g)
    out, border_cells = _stretch_border(g, central["bbox"])

    for sat in satellites:
        out[sat["coords"][:, 0], sat["coords"][:, 1]] = BLACK

    for sat in satellites:
        nearest, _is_unique = _nearest_border_info(sat["coords"], border_cells)
        action_color = nearest[2]
        assert action_color in ACTION_COLORS
        moved = _apply_satellite_action(sat["coords"], action_color, g.shape)
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < g.shape[0]))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < g.shape[1]))
        assert np.all(out[moved[:, 0], moved[:, 1]] == BLACK)
        out[moved[:, 0], moved[:, 1]] = int(sat["color"])

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _rotated_shape(mask: np.ndarray, k: int) -> np.ndarray:
    return _normalize_mask(np.rot90(mask, k))


def _shape_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _shape_is_lr_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _random_palette() -> tuple[int, ...]:
    if random.random() < 0.7:
        palette = tuple(sorted(random.sample(ACTION_COLORS, 3)))
        if 4 not in palette and random.random() < 0.6:
            other = [color for color in ACTION_COLORS if color != 4]
            keep = random.sample(other, 2)
            palette = tuple(sorted((*keep, 4)))
    else:
        pairs = [
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), (3, 4),
        ]
        palette = random.choice(pairs)
    return tuple(int(color) for color in palette)


def _make_side_commands(palette: tuple[int, ...]) -> dict[str, int]:
    commands = list(palette)
    while len(commands) < 4:
        commands.append(random.choice(palette))
    random.shuffle(commands)
    return {side: int(color) for side, color in zip(SIDE_ORDER, commands)}


def _edge_targets(
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> dict[str, tuple[int, int, tuple[int, int]]]:
    top_idx = random.randint(1, max(1, w // 2))
    right_idx = random.randint(1, max(1, h // 2))
    bottom_idx = random.randint(max(1, w // 2), w - 2)
    left_idx = random.randint(max(1, h // 2), h - 2)
    return {
        "top": (r0 - 1, c0 + top_idx, (0, top_idx)),
        "right": (r0 + right_idx, c0 + w, (right_idx, w - 1)),
        "bottom": (r0 + h, c0 + bottom_idx, (h - 1, bottom_idx)),
        "left": (r0 + left_idx, c0 - 1, (left_idx, 0)),
    }


def _make_central_sprite(
    h: int,
    w: int,
    palette: tuple[int, ...],
    side_commands: dict[str, int],
    target_info: dict[str, tuple[int, int, tuple[int, int]]],
) -> np.ndarray:
    sprite = np.random.choice(np.array(palette, dtype=int), size=(h, w))
    sprite[0, :] = np.random.choice(np.array(palette, dtype=int), size=w)
    sprite[-1, :] = np.random.choice(np.array(palette, dtype=int), size=w)
    sprite[:, 0] = np.random.choice(np.array(palette, dtype=int), size=h)
    sprite[:, -1] = np.random.choice(np.array(palette, dtype=int), size=h)
    for side, color in side_commands.items():
        rr, cc = target_info[side][2]
        sprite[rr, cc] = int(color)
    missing = [color for color in palette if color not in set(map(int, sprite.flat))]
    for color in missing:
        rr = random.randint(1, h - 2)
        cc = random.randint(1, w - 2)
        sprite[rr, cc] = int(color)
    return sprite.astype(int)


def _all_cells_free(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if cells & occupied:
        return False
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _candidate_anchors(
    side: str,
    shape_h: int,
    shape_w: int,
    target_rc: tuple[int, int],
    central_bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = central_bbox
    target_r, target_c = target_rc
    anchors: list[tuple[int, int]] = []

    if side == "top":
        max_row = r0 - 3 - shape_h + 1
        col_lo = max(0, target_c - shape_w - 2)
        col_hi = min(w - shape_w, target_c + 2)
        for rr in range(max_row + 1):
            for cc in range(col_lo, col_hi + 1):
                center_c = cc + (shape_w - 1) / 2.0
                if center_c < w / 2.0:
                    anchors.append((rr, cc))
    elif side == "right":
        row_lo = max(0, target_r - shape_h)
        row_hi = min(h - shape_h, target_r + 1)
        col_lo = c1 + 3
        for rr in range(row_lo, row_hi + 1):
            for cc in range(col_lo, w - shape_w + 1):
                center_r = rr + (shape_h - 1) / 2.0
                if center_r < h / 2.0:
                    anchors.append((rr, cc))
    elif side == "bottom":
        row_lo = r1 + 3
        col_lo = max(0, target_c - shape_w - 1)
        col_hi = min(w - shape_w, target_c + 1)
        for rr in range(row_lo, h - shape_h + 1):
            for cc in range(col_lo, col_hi + 1):
                center_c = cc + (shape_w - 1) / 2.0
                if center_c > w / 2.0:
                    anchors.append((rr, cc))
    else:
        row_lo = max(0, target_r - shape_h)
        row_hi = min(h - shape_h, target_r + 1)
        max_col = c0 - 3 - shape_w + 1
        for rr in range(row_lo, row_hi + 1):
            for cc in range(max_col + 1):
                center_r = rr + (shape_h - 1) / 2.0
                if center_r > h / 2.0:
                    anchors.append((rr, cc))
    random.shuffle(anchors)
    return anchors


def _render_satellite(coords: np.ndarray, color: int, grid: np.ndarray) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _place_satellites(
    grid_shape: tuple[int, int],
    central_bbox: tuple[int, int, int, int],
    border_cells: list[tuple[int, int, int]],
    target_info: dict[str, tuple[int, int, tuple[int, int]]],
    central_coords: np.ndarray,
    side_commands: dict[str, int],
) -> list[dict] | None:
    side_colors = list(SATELLITE_COLORS)
    random.shuffle(side_colors)
    fixed_output = set(map(tuple, central_coords.tolist()))
    r0, c0, r1, c1 = central_bbox
    for c in range(c0, c1 + 1):
        fixed_output.add((r0 - 1, c))
        fixed_output.add((r1 + 1, c))
    for r in range(r0, r1 + 1):
        fixed_output.add((r, c0 - 1))
        fixed_output.add((r, c1 + 1))

    occupied_input = set(map(tuple, central_coords.tolist()))
    occupied_output = set(fixed_output)
    blocked_sweeps: set[tuple[int, int]] = set()
    placements: list[dict] = []

    for side, sat_color in zip(SIDE_ORDER, side_colors):
        command = side_commands[side]
        target_rc = target_info[side][:2]
        placed = None
        for _ in range(PLACE_RETRIES):
            mask = _rotated_shape(random.choice(SHAPE_LIBRARY), random.randint(0, 3))
            if command == 4 and _shape_is_lr_symmetric(mask):
                continue
            shape_coords = _shape_coords(mask)
            shape_h, shape_w = mask.shape
            for anchor_r, anchor_c in _candidate_anchors(
                side,
                shape_h,
                shape_w,
                target_rc,
                central_bbox,
                grid_shape,
            ):
                abs_coords = shape_coords + np.array([anchor_r, anchor_c])
                cells = {tuple(map(int, cell)) for cell in abs_coords.tolist()}
                if not _all_cells_free(cells, occupied_input):
                    continue
                if cells & blocked_sweeps:
                    continue
                nearest, is_unique = _nearest_border_info(abs_coords, border_cells)
                if (nearest[0], nearest[1]) != target_rc or nearest[2] != command or not is_unique:
                    continue
                moved = _apply_satellite_action(abs_coords, command, grid_shape)
                moved_cells = {tuple(map(int, cell)) for cell in moved.tolist()}
                if command != 4 and moved_cells == cells:
                    continue
                if command == 4 and moved_cells == cells:
                    continue
                if moved_cells & occupied_output:
                    continue
                swept = _swept_cells(abs_coords, command, grid_shape)
                if swept & occupied_input:
                    continue
                placed = {
                    "side": side,
                    "color": int(sat_color),
                    "command": int(command),
                    "coords": abs_coords.astype(int),
                    "moved": moved.astype(int),
                    "swept": swept,
                }
                break
            if placed is not None:
                break
        if placed is None:
            return None

        occupied_input |= {tuple(map(int, cell)) for cell in placed["coords"].tolist()}
        occupied_output |= {tuple(map(int, cell)) for cell in placed["moved"].tolist()}
        blocked_sweeps |= placed["swept"]
        placements.append(placed)
    return placements


def _validate_generated(grid: np.ndarray, side_commands: dict[str, int]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if np.array_equal(grid, out):
        return False

    central, satellites = _central_and_satellites(grid)
    if len({tuple(sorted(comp["colors"])) for comp in satellites}) != 4:
        return False
    stretched, border_cells = _stretch_border(grid, central["bbox"])
    if np.array_equal(stretched, grid):
        return False

    commands_seen: list[int] = []
    mirror_changed = False
    for sat in satellites:
        nearest, is_unique = _nearest_border_info(sat["coords"], border_cells)
        if not is_unique or nearest[2] not in ACTION_COLORS:
            return False
        commands_seen.append(nearest[2])
        transformed = _apply_satellite_action(sat["coords"], nearest[2], grid.shape)
        if nearest[2] == 4 and {tuple(x) for x in transformed.tolist()} != {tuple(x) for x in sat["coords"].tolist()}:
            mirror_changed = True
    if len(set(commands_seen)) < 2:
        return False
    if 4 in side_commands.values() and not mirror_changed:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        center_h = random.randint(5, 7)
        center_w = random.randint(5, 7)
        top_margin = random.randint(4, 6)
        bottom_margin = random.randint(4, 6)
        left_margin = random.randint(4, 6)
        right_margin = random.randint(4, 6)
        h = center_h + top_margin + bottom_margin
        w = center_w + left_margin + right_margin
        if max(h, w) > 30:
            continue

        grid = np.zeros((h, w), dtype=int)
        r0, c0 = top_margin, left_margin
        central_bbox = (r0, c0, r0 + center_h - 1, c0 + center_w - 1)
        palette = _random_palette()
        side_commands = _make_side_commands(palette)
        target_info = _edge_targets(r0, c0, center_h, center_w)
        sprite = _make_central_sprite(center_h, center_w, palette, side_commands, target_info)
        grid[r0 : r0 + center_h, c0 : c0 + center_w] = sprite

        central_coords = np.argwhere(grid != BLACK)
        _stretched, border_cells = _stretch_border(grid, central_bbox)
        placements = _place_satellites(
            grid.shape,
            central_bbox,
            border_cells,
            target_info,
            central_coords,
            side_commands,
        )
        if placements is None:
            continue

        for sat in placements:
            _render_satellite(sat["coords"], sat["color"], grid)

        if _validate_generated(grid, side_commands):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 8")
