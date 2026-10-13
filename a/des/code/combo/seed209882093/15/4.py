from __future__ import annotations

import random

import numpy as np


BG = 0
CARDINALS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ROTATION_TURNS = {"up": 0, "left": 1, "down": 2, "right": 3}
BASE_ARROW_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (1, 1), (2, 1), (3, 0), (3, 1), (3, 2)),
)

GRID_SHAPE_RANGE = (12, 18)
ARROW_COUNT_RANGE = (2, 4)
ARROW_SHIFT_RANGE = (2, 6)
OBSTACLE_COUNT_RANGE = (0, 4)
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    rows = coords[:, 0]
    cols = coords[:, 1]
    mask = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=bool)
    mask[rows, cols] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _sorted_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    arr = _sorted_coords(coords)
    arr = arr - np.array([int(arr[:, 0].min()), int(arr[:, 1].min())], dtype=int)
    return _sorted_coords(arr)


def _rotate_shape(base: tuple[tuple[int, int], ...], direction: str) -> np.ndarray:
    mask = _coords_to_mask(np.array(base, dtype=int))
    return _normalize_coords(_mask_to_coords(np.rot90(mask, k=ROTATION_TURNS[direction])))


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.asarray(coords, dtype=int) + np.array([int(dr), int(dc)], dtype=int)


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in np.asarray(coords, dtype=int)}


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    h, w = map(int, shape)
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(_sorted_coords(np.array(cells, dtype=int)))
    return components


def _match_direction(coords: np.ndarray, direction: str) -> tuple[int, int] | None:
    dr, dc = CARDINALS[direction]
    axis = 0 if dr != 0 else 1
    sign = dr if dr != 0 else dc
    vals = coords[:, axis]
    layers = sorted(int(v) for v in np.unique(vals))
    if layers != list(range(layers[0], layers[-1] + 1)):
        return None

    ordered_layers = layers if sign < 0 else layers[::-1]
    counts = [int(np.sum(vals == layer)) for layer in ordered_layers]
    if counts[0] != 1 or counts[-1] < 2:
        return None
    if any(a > b for a, b in zip(counts, counts[1:])):
        return None
    if max(counts) != counts[-1]:
        return None
    if len(counts) >= 2 and counts[-1] == counts[-2]:
        return None

    tip_layer = ordered_layers[0]
    tip_cells = coords[vals == tip_layer]
    if len(tip_cells) != 1:
        return None
    return tuple(map(int, tip_cells[0]))


def _infer_direction(coords: np.ndarray) -> tuple[str, tuple[int, int]]:
    raw = _sorted_coords(np.asarray(coords, dtype=int))
    offset = np.array([int(raw[:, 0].min()), int(raw[:, 1].min())], dtype=int)
    arr = raw - offset
    matches = []
    for direction in CARDINALS:
        tip = _match_direction(arr, direction)
        if tip is not None:
            tip_abs = (int(tip[0] + offset[0]), int(tip[1] + offset[1]))
            matches.append((direction, tip_abs))
    assert len(matches) == 1, f"expected one arrow direction, got {matches}"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each arrow-shaped object along its implicit direction until its tip is
    directly adjacent to the same-colored anchor. Different-colored single-pixel
    obstacles are ignored during the slide and remain visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    arrows: list[dict[str, object]] = []
    for color in colors:
        comps = _color_components(g, color)
        big = [comp for comp in comps if len(comp) > 1]
        small = [comp for comp in comps if len(comp) == 1]
        if not big:
            continue
        assert len(big) == 1 and len(small) == 1, f"color {color} must have one arrow and one anchor"
        arrow = big[0]
        assert 3 <= len(arrow) <= 6, f"unexpected arrow size for color {color}"
        direction, tip = _infer_direction(arrow)
        anchor = tuple(map(int, small[0][0]))
        arrows.append(
            {
                "color": color,
                "coords": arrow,
                "coord_set": _coords_set(arrow),
                "direction": direction,
                "tip": tip,
                "anchor": anchor,
            }
        )

    out = g.copy()
    for arrow in arrows:
        coords = np.asarray(arrow["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = BG

    moved_occupied: set[tuple[int, int]] = set()
    for arrow in arrows:
        color = int(arrow["color"])
        coords = np.asarray(arrow["coords"], dtype=int)
        coord_set = set(arrow["coord_set"])
        direction = str(arrow["direction"])
        dr, dc = CARDINALS[direction]
        tip_r, tip_c = map(int, arrow["tip"])

        distance = 1
        stop_distance = None
        while True:
            probe = (tip_r + distance * dr, tip_c + distance * dc)
            if not _in_bounds(g.shape, *probe):
                break
            if probe not in coord_set and int(g[probe]) == color:
                stop_distance = distance
                break
            distance += 1

        assert stop_distance is not None and stop_distance >= 1, f"color {color} must see a same-color stop pixel"
        shift = stop_distance - 1
        shifted = _translate(coords, shift * dr, shift * dc)

        for cell in shifted:
            rc = tuple(map(int, cell))
            assert rc not in moved_occupied, "moved arrows must not overlap each other"
            moved_occupied.add(rc)

        for r, c in shifted:
            rr, cc = int(r), int(c)
            if out[rr, cc] == BG:
                out[rr, cc] = color

    return out


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    arr = np.asarray(coords, dtype=int)
    grid[arr[:, 0], arr[:, 1]] = int(color)


def _free_cells(
    shape: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    blocked_input: set[tuple[int, int]],
    blocked_output: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    candidates = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in occupied_input or cell in occupied_output:
                continue
            if cell in blocked_input or cell in blocked_output:
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _build_layout(shape: tuple[int, int], arrow_colors: list[int]) -> list[dict[str, object]] | None:
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    blocked_input: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []
    h, w = map(int, shape)

    for color in arrow_colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            direction = random.choice(list(CARDINALS))
            shape_coords = _rotate_shape(random.choice(BASE_ARROW_SHAPES), direction)
            _dir_check, tip_rel = _infer_direction(shape_coords)
            sh = int(shape_coords[:, 0].max()) + 1
            sw = int(shape_coords[:, 1].max()) + 1

            final_r = random.randint(0, h - sh)
            final_c = random.randint(0, w - sw)
            dr, dc = CARDINALS[direction]
            anchor = (final_r + tip_rel[0] + dr, final_c + tip_rel[1] + dc)
            if not _in_bounds(shape, *anchor):
                continue

            if dr < 0:
                max_shift = h - sh - final_r
            elif dr > 0:
                max_shift = final_r
            elif dc < 0:
                max_shift = w - sw - final_c
            else:
                max_shift = final_c

            if max_shift < ARROW_SHIFT_RANGE[0]:
                continue

            shift_lo = ARROW_SHIFT_RANGE[0]
            shift_hi = min(ARROW_SHIFT_RANGE[1], max_shift)
            if shift_hi >= 3:
                shift_lo = max(shift_lo, 3)
            if shift_lo > shift_hi:
                continue
            shift = random.randint(shift_lo, shift_hi)

            init_r = final_r - shift * dr
            init_c = final_c - shift * dc
            initial_coords = _translate(shape_coords, init_r, init_c)
            final_coords = _translate(shape_coords, final_r, final_c)

            initial_cells = _coords_set(initial_coords)
            final_cells = _coords_set(final_coords)
            input_cells = set(initial_cells)
            input_cells.add(anchor)
            output_cells = set(final_cells)
            output_cells.add(anchor)

            if input_cells & occupied_input or output_cells & occupied_output:
                continue
            if input_cells & blocked_input or output_cells & blocked_output:
                continue

            path_cells: set[tuple[int, int]] = set()
            for step in range(1, shift):
                path_cells |= _coords_set(_translate(initial_coords, step * dr, step * dc))
            path_cells -= initial_cells
            path_cells -= final_cells
            path_cells.discard(anchor)

            placements.append(
                {
                    "color": color,
                    "direction": direction,
                    "initial": initial_coords,
                    "final": final_coords,
                    "anchor": anchor,
                    "path": path_cells,
                }
            )
            occupied_input |= input_cells
            occupied_output |= output_cells
            blocked_input |= _halo(input_cells, shape)
            blocked_output |= _halo(output_cells, shape)
            placed = True
            break
        if not placed:
            return None

    return placements


def _place_obstacles(
    shape: tuple[int, int],
    placements: list[dict[str, object]],
    obstacle_colors: list[int],
) -> list[tuple[int, tuple[int, int]]]:
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    blocked_input: set[tuple[int, int]] = set()
    blocked_output: set[tuple[int, int]] = set()
    for placement in placements:
        initial_cells = _coords_set(np.asarray(placement["initial"], dtype=int))
        final_cells = _coords_set(np.asarray(placement["final"], dtype=int))
        anchor = tuple(map(int, placement["anchor"]))
        input_cells = set(initial_cells)
        input_cells.add(anchor)
        output_cells = set(final_cells)
        output_cells.add(anchor)
        occupied_input |= input_cells
        occupied_output |= output_cells
        blocked_input |= _halo(input_cells, shape)
        blocked_output |= _halo(output_cells, shape)

    path_candidates: list[tuple[int, int]] = []
    for placement in placements:
        for cell in placement["path"]:
            if cell in occupied_input or cell in occupied_output:
                continue
            if cell in blocked_input or cell in blocked_output:
                continue
            path_candidates.append(cell)
    path_candidates = list(dict.fromkeys(path_candidates))
    random.shuffle(path_candidates)

    obstacles: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()

    if path_candidates and obstacle_colors and random.random() < 0.85:
        n_path = random.randint(1, min(2, len(path_candidates), len(obstacle_colors)))
        for cell, color in zip(path_candidates[:n_path], obstacle_colors[:n_path], strict=False):
            obstacles.append((int(color), cell))
            used.add(cell)
        obstacle_colors = obstacle_colors[n_path:]

    if not obstacle_colors:
        return obstacles

    free_cells = _free_cells(shape, occupied_input | used, occupied_output | used, blocked_input, blocked_output)
    if free_cells:
        n_extra = min(random.randint(*OBSTACLE_COUNT_RANGE), len(free_cells), len(obstacle_colors))
        for cell, color in zip(free_cells[:n_extra], obstacle_colors[:n_extra], strict=False):
            obstacles.append((int(color), cell))
            used.add(cell)

    return obstacles


def _expected_output(inp: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for placement in placements:
        initial = np.asarray(placement["initial"], dtype=int)
        out[initial[:, 0], initial[:, 1]] = BG

    seen: set[tuple[int, int]] = set()
    for placement in placements:
        color = int(placement["color"])
        final = np.asarray(placement["final"], dtype=int)
        for r, c in final:
            cell = (int(r), int(c))
            assert cell not in seen
            seen.add(cell)
            if out[cell] == BG:
                out[cell] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        h = random.randint(*GRID_SHAPE_RANGE)
        w = random.randint(*GRID_SHAPE_RANGE)
        n_arrows = random.randint(*ARROW_COUNT_RANGE)
        palette = list(range(1, 10))
        random.shuffle(palette)
        arrow_colors = palette[:n_arrows]
        obstacle_palette = palette[n_arrows:]

        placements = _build_layout((h, w), arrow_colors)
        if placements is None:
            continue

        obstacle_cap = min(len(obstacle_palette), OBSTACLE_COUNT_RANGE[1] + 2)
        obstacle_colors = obstacle_palette[:obstacle_cap]
        obstacles = _place_obstacles((h, w), placements, obstacle_colors)

        grid = np.zeros((h, w), dtype=int)
        for placement in placements:
            _paint(grid, np.asarray(placement["initial"], dtype=int), int(placement["color"]))
            anchor = tuple(map(int, placement["anchor"]))
            grid[anchor] = int(placement["color"])
        for color, cell in obstacles:
            grid[cell] = int(color)

        expected = _expected_output(grid, placements)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid ARC instance")
