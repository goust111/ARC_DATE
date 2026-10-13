import random
from collections import Counter

import numpy as np


BG = 0
RECEIVER = 5
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _cross_pixels(center):
    r, c = center
    return {
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def _infer_towers_and_receivers(grid):
    g = np.asarray(grid, dtype=int)
    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    towers = []
    tower_pixels = set()
    non_tower_coords = []

    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        coord_set = set(coords)
        center = None
        if len(coords) == 5:
            for r, c in coords:
                if _cross_pixels((r, c)) == coord_set:
                    center = (r, c)
                    break
        if center is not None:
            towers.append({"color": color, "center": center, "pixels": coord_set})
            tower_pixels |= coord_set
        else:
            non_tower_coords.extend((color, rc) for rc in coords)

    assert towers, "expected at least one signal tower"
    towers.sort(key=lambda tower: tower["center"])

    receiver_colors = sorted({color for color, _ in non_tower_coords})
    if non_tower_coords:
        assert receiver_colors == [RECEIVER], "expected only grey receiver markers outside towers"

    receivers = {rc for _color, rc in non_tower_coords}
    for r, c in receivers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in receivers:
                    raise AssertionError("receiver markers must stay as isolated single pixels")
    assert not (receivers & tower_pixels), "receivers must not overlap towers"

    return towers, receivers


def _tower_pixel_map(towers):
    mapping = {}
    for idx, tower in enumerate(towers):
        for pixel in tower["pixels"]:
            assert pixel not in mapping, "towers must not overlap"
            mapping[pixel] = idx
    return mapping


def _cast_ray(shape, towers, tower_by_pixel, current_idx, direction, visit):
    h, w = shape
    dr, dc = direction
    tower = towers[current_idx]
    current = tower["center"]

    while True:
        nr = current[0] + dr
        nc = current[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return
        nxt = (nr, nc)
        if nxt in tower["pixels"]:
            current = nxt
            continue
        hit_idx = tower_by_pixel.get(nxt)
        if hit_idx is not None:
            _cast_ray(shape, towers, tower_by_pixel, hit_idx, direction, visit)
            return
        visit(nxt, tower["color"])
        current = nxt


def _paint_all_rays(out, towers):
    tower_by_pixel = _tower_pixel_map(towers)
    shape = out.shape
    for tower_idx, _tower in enumerate(towers):
        for direction in DIRECTIONS:
            _cast_ray(
                shape,
                towers,
                tower_by_pixel,
                tower_idx,
                direction,
                lambda cell, color: out.__setitem__(cell, color),
            )
    return out


def _source_coverages(shape, towers):
    tower_by_pixel = _tower_pixel_map(towers)
    coverages = []
    for tower_idx, _tower in enumerate(towers):
        visited = set()
        for direction in DIRECTIONS:
            _cast_ray(shape, towers, tower_by_pixel, tower_idx, direction, lambda cell, _color: visited.add(cell))
        coverages.append(visited)
    return coverages


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast each tower color along cardinal rays, relaying through later towers until all reachable receivers are colored.

    Input structure:
    - The background is black.
    - Signal towers are monochrome 3x3 plus signs; each tower color appears only in its own five-pixel cross.
    - Receivers are isolated grey single pixels placed on the background.

    Rule:
    - Keep every tower cross unchanged.
    - Process towers in row-major order by their centers; for each tower, emit rays upward, downward, leftward, and rightward.
    - A ray passes through its own tower pixels, paints every non-tower cell it traverses, colors any receiver it reaches, and continues onward.
    - If the ray would enter another tower pixel, it stops before that tower and immediately relays from the hit tower's center in the same direction using the hit tower's color.

    Key insight:
    - Rays are not blocked permanently by towers; each encountered tower becomes a same-direction relay, so the final color at a receiver can depend on a whole chain of tower hand-offs.
    """
    g = np.asarray(grid, dtype=int)
    towers, _receivers = _infer_towers_and_receivers(g)
    out = g.copy()
    return _paint_all_rays(out, towers)


def _build_chain_centers(orientation, chain_len, allow_offsets, rng):
    axial = 0
    ortho = 0
    centers = []
    shifts = []
    for idx in range(chain_len):
        if orientation == "horizontal":
            centers.append((ortho, axial))
        else:
            centers.append((axial, ortho))
        if idx == chain_len - 1:
            break
        axial += rng.randint(4, 6)
        shift = rng.choice((-1, 0, 1)) if allow_offsets else 0
        shifts.append(shift)
        ortho += shift
    if allow_offsets and shifts and all(shift == 0 for shift in shifts):
        pos = rng.randrange(len(shifts))
        shifts[pos] = rng.choice((-1, 1))
        axial = 0
        ortho = 0
        centers = []
        for idx in range(chain_len):
            if orientation == "horizontal":
                centers.append((ortho, axial))
            else:
                centers.append((axial, ortho))
            if idx < chain_len - 1:
                axial += rng.randint(4, 6)
                ortho += shifts[idx]
    return centers


def _layout_towers(rng):
    orientation = rng.choice(("horizontal", "vertical"))
    chain_len = rng.randint(3, 4)
    allow_offsets = rng.random() < 0.5
    overwrite_receiver = rng.random() < 0.5
    centers = _build_chain_centers(orientation, chain_len, allow_offsets, rng)

    if orientation == "horizontal":
        last_r, last_c = centers[-1]
        primary_receiver = (last_r, last_c + rng.randint(3, 5))
        if overwrite_receiver:
            centers.append((primary_receiver[0] + rng.randint(4, 6), primary_receiver[1]))
    else:
        last_r, last_c = centers[-1]
        primary_receiver = (last_r + rng.randint(3, 5), last_c)
        if overwrite_receiver:
            centers.append((primary_receiver[0], primary_receiver[1] + rng.randint(4, 6)))

    tower_pixels = Counter()
    for center in centers:
        for pixel in _cross_pixels(center):
            tower_pixels[pixel] += 1
    if tower_pixels and max(tower_pixels.values()) > 1:
        return None

    min_r = min(r for r, _c in tower_pixels)
    max_r = max(r for r, _c in tower_pixels)
    min_c = min(c for _r, c in tower_pixels)
    max_c = max(c for _r, c in tower_pixels)

    top_pad = rng.randint(2, 4)
    left_pad = rng.randint(2, 4)
    row_shift = top_pad - min_r
    col_shift = left_pad - min_c
    shifted_centers = [(r + row_shift, c + col_shift) for r, c in centers]
    shifted_receiver = (primary_receiver[0] + row_shift, primary_receiver[1] + col_shift)

    max_row = max(max_r + row_shift, shifted_receiver[0])
    max_col = max(max_c + col_shift, shifted_receiver[1])
    h = max_row + rng.randint(3, 5)
    w = max_col + rng.randint(3, 5)
    if max(h, w) > 30:
        return None

    towers = []
    colors = rng.sample(TOWER_COLORS, len(shifted_centers))
    for color, center in zip(colors, shifted_centers):
        towers.append({"color": color, "center": center, "pixels": _cross_pixels(center)})

    towers.sort(key=lambda tower: tower["center"])
    return {
        "shape": (h, w),
        "towers": towers,
        "primary_receiver": shifted_receiver,
        "overwrite_receiver": overwrite_receiver,
        "orientation": orientation,
        "allow_offsets": allow_offsets,
    }


def _choose_receivers(shape, towers, primary_receiver, overwrite_receiver, rng):
    tower_pixels = set().union(*(tower["pixels"] for tower in towers))
    coverages = _source_coverages(shape, towers)
    coverage_map = {}
    color_map = {}
    for src_idx, cells in enumerate(coverages):
        for cell in cells:
            coverage_map.setdefault(cell, set()).add(src_idx)
            color_map.setdefault(cell, set()).add(towers[src_idx]["color"])

    def clear_cell(cell):
        if cell in tower_pixels or cell not in coverage_map:
            return False
        r, c = cell
        if r <= 0 or c <= 0 or r >= shape[0] - 1 or c >= shape[1] - 1:
            return False
        for tr, tc in tower_pixels:
            if max(abs(r - tr), abs(c - tc)) <= 1:
                return False
        return True

    if not clear_cell(primary_receiver):
        return None

    receivers = [primary_receiver]
    if overwrite_receiver and len(color_map.get(primary_receiver, set())) < 2:
        return None

    candidates = [cell for cell in coverage_map if clear_cell(cell) and cell != primary_receiver]
    rng.shuffle(candidates)
    multi_candidates = [cell for cell in candidates if len(color_map[cell]) >= 2]
    unique_candidates = [cell for cell in candidates if len(color_map[cell]) == 1]

    target_count = rng.randint(2, 5)
    if overwrite_receiver:
        target_count = max(target_count, 3)

    if rng.random() < 0.5 and multi_candidates:
        candidates = multi_candidates + [cell for cell in candidates if cell not in set(multi_candidates)]
    else:
        candidates = unique_candidates + [cell for cell in candidates if cell not in set(unique_candidates)]

    for cell in candidates:
        if len(receivers) >= target_count:
            break
        if any(max(abs(cell[0] - rr), abs(cell[1] - rc)) <= 1 for rr, rc in receivers):
            continue
        receivers.append(cell)

    if len(receivers) < 2:
        return None
    return receivers


def _render_input(shape, towers, receivers):
    grid = np.full(shape, BG, dtype=int)
    for tower in towers:
        for r, c in tower["pixels"]:
            grid[r, c] = tower["color"]
    for r, c in receivers:
        assert grid[r, c] == BG, "receivers must stay off tower pixels"
        grid[r, c] = RECEIVER
    return grid


def _gen_attempt():
    rng = random.Random()
    layout = _layout_towers(rng)
    if layout is None:
        return None

    shape = layout["shape"]
    towers = layout["towers"]
    receivers = _choose_receivers(
        shape,
        towers,
        layout["primary_receiver"],
        layout["overwrite_receiver"],
        rng,
    )
    if receivers is None:
        return None

    grid = _render_input(shape, towers, receivers)
    out = main(grid)

    if np.array_equal(grid, out):
        return None
    if layout["overwrite_receiver"]:
        primary = layout["primary_receiver"]
        if int(out[primary]) == RECEIVER:
            return None
        later_color = towers[-1]["color"]
        if int(out[primary]) != later_color:
            return None

    path_pixels = int(np.count_nonzero((out != grid) & (grid != RECEIVER)))
    if path_pixels < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-tower input")
