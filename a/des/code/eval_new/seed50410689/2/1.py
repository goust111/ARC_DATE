from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

GEN_ATTEMPTS = 400
REGION_SHAPE_ATTEMPTS = 200
REGION_PLACE_ATTEMPTS = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _touches_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a same-colored ray outside each grey-walled room using the room seed as the key.

    Input structure:
    - Black background surrounds several irregular rooms enclosed by 1-pixel-thick grey walls.
    - Each enclosed room contains exactly one single-pixel seed whose color is neither black nor grey.
    - Outside each room, one same-colored single-pixel exit marker sits directly beyond one wall side.

    Rule:
    - Find every 4-connected non-grey component that does not touch the grid edge; each is one enclosed room plus its seed.
    - For each room, read the seed color C, count its black interior cells, and locate the other pixel of color C outside the room as the exit marker.
    - The exit marker's adjacent grey wall gives the outward direction; draw color C from the cell beyond the marker for as many steps as the room has black interior cells, stopping when the next cell is out of bounds or already non-black in the input.
    - Keep all original seeds, exit markers, walls, and unrelated pixels unchanged.

    Key insight:
    - Seed color pairs each closed room with exactly one outside marker, while marker side encodes direction and room area encodes ray length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    color_positions: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        color_positions.setdefault(int(g[r, c]), []).append((int(r), int(c)))

    for comp in _components(g != GREY):
        if _touches_edge(comp, (h, w)):
            continue

        seed_pixels = [(r, c) for r, c in comp if int(g[r, c]) not in (BLACK, GREY)]
        assert len(seed_pixels) == 1, "each room must contain exactly one seed"
        seed_r, seed_c = seed_pixels[0]
        color = int(g[seed_r, seed_c])

        positions = color_positions.get(color, [])
        assert len(positions) == 2, "seed color must appear exactly twice"
        marker = positions[0] if positions[1] == (seed_r, seed_c) else positions[1]

        grey_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(*marker)
            if _in_bounds((h, w), nr, nc) and int(g[nr, nc]) == GREY
        ]
        assert len(grey_neighbors) == 1, "exit marker direction must be unambiguous"
        wall_r, wall_c = grey_neighbors[0]
        dr = int(marker[0] - wall_r)
        dc = int(marker[1] - wall_c)

        ray_len = sum(1 for r, c in comp if int(g[r, c]) == BLACK)
        rr, cc = marker
        for _ in range(ray_len):
            rr += dr
            cc += dc
            if not _in_bounds((h, w), rr, cc):
                break
            if int(g[rr, cc]) != BLACK:
                break
            out[rr, cc] = color

    return out


def _distance_to_border(pos: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = pos
    h, w = shape
    if direction == UP:
        return r
    if direction == DOWN:
        return h - 1 - r
    if direction == LEFT:
        return c
    return w - 1 - c


def _sample_interior_mask() -> np.ndarray | None:
    for _ in range(REGION_SHAPE_ATTEMPTS):
        box_h = random.randint(2, 5)
        box_w = random.randint(2, 5)
        max_area = min(9, box_h * box_w - 1)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < box_h and 0 <= nc < box_w
        }
        while len(cells) < area and frontier:
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            cells.add(nxt)
            for nr, nc in _neighbors4(*nxt):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        crop_h = r1 - r0 + 1
        crop_w = c1 - c0 + 1
        if crop_h < 2 or crop_w < 2:
            continue

        mask = np.zeros((crop_h, crop_w), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        if bool(mask.all()):
            continue
        if int(mask.sum()) < 3:
            continue
        return mask
    return None


def _region_template(mask: np.ndarray) -> dict | None:
    ih, iw = mask.shape
    h, w = ih + 4, iw + 4
    interior = np.zeros((h, w), dtype=bool)
    interior[2 : 2 + ih, 2 : 2 + iw] = mask

    wall = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(interior):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c):
            if not interior[nr, nc]:
                wall[nr, nc] = True

    candidates = {d: set() for d in DIRS}
    for r, c in np.argwhere(wall):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS:
            ir, ic = r - dr, c - dc
            mr, mc = r + dr, c + dc
            if not _in_bounds((h, w), mr, mc):
                continue
            if not interior[ir, ic]:
                continue
            if interior[mr, mc] or wall[mr, mc]:
                continue
            if dr == -1 and mr != 0:
                continue
            if dr == 1 and mr != h - 1:
                continue
            if dc == -1 and mc != 0:
                continue
            if dc == 1 and mc != w - 1:
                continue
            grey_neighbors = 0
            for nr, nc in _neighbors4(mr, mc):
                if _in_bounds((h, w), nr, nc) and wall[nr, nc]:
                    grey_neighbors += 1
            if grey_neighbors == 1:
                candidates[(dr, dc)].add((mr, mc))

    if not all(candidates[d] for d in DIRS):
        return None
    return {
        "interior": interior,
        "wall": wall,
        "candidates": {d: sorted(v) for d, v in candidates.items()},
    }


def _sample_region_spec(color: int, direction: tuple[int, int]) -> dict | None:
    for _ in range(REGION_SHAPE_ATTEMPTS):
        mask = _sample_interior_mask()
        if mask is None:
            continue
        template = _region_template(mask)
        if template is None or not template["candidates"][direction]:
            continue

        interior = template["interior"]
        wall = template["wall"]
        seed = random.choice([(int(r), int(c)) for r, c in np.argwhere(interior)])
        marker = random.choice(template["candidates"][direction])

        patch = np.zeros(interior.shape, dtype=int)
        patch[wall] = GREY
        patch[seed] = color
        patch[marker] = color

        return {
            "patch": patch,
            "nonblack": [(int(r), int(c)) for r, c in np.argwhere(patch != BLACK)],
            "marker": marker,
            "direction": direction,
            "length": int(interior.sum()) - 1,
            "color": color,
        }
    return None


def _offset_ranges(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    direction: tuple[int, int],
    ray_len: int,
    clipped: bool,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = grid_shape
    ph, pw = patch_shape
    row_min, row_max = 0, h - ph
    col_min, col_max = 0, w - pw

    if direction == UP:
        row_min = 1 if clipped else ray_len
        row_max = min(h - ph, ray_len - 1) if clipped else h - ph
    elif direction == DOWN:
        row_min = max(0, h - ph - (ray_len - 1)) if clipped else 0
        row_max = h - ph - 1 if clipped else h - ph - ray_len
    elif direction == LEFT:
        col_min = 1 if clipped else ray_len
        col_max = min(w - pw, ray_len - 1) if clipped else w - pw
    else:
        col_min = max(0, w - pw - (ray_len - 1)) if clipped else 0
        col_max = w - pw - 1 if clipped else w - pw - ray_len

    if row_min > row_max or col_min > col_max:
        return None
    return (row_min, row_max), (col_min, col_max)


def _can_place_nonblack(
    cells: list[tuple[int, int]],
    occupied: np.ndarray,
    reserved: np.ndarray,
) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c] or reserved[r, c]:
            return False
        rs = slice(max(0, r - 1), min(h, r + 2))
        cs = slice(max(0, c - 1), min(w, c + 2))
        if bool(occupied[rs, cs].any()):
            return False
    return True


def _place_region(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved: np.ndarray,
    spec: dict,
    clipped: bool,
) -> dict | None:
    h, w = grid.shape
    patch = spec["patch"]
    ph, pw = patch.shape
    ray_len = int(spec["length"])
    direction = spec["direction"]
    ranges = _offset_ranges((h, w), (ph, pw), direction, ray_len, clipped)
    if ranges is None:
        return None
    (row_min, row_max), (col_min, col_max) = ranges

    local_cells = spec["nonblack"]
    for _ in range(REGION_PLACE_ATTEMPTS):
        r0 = random.randint(row_min, row_max)
        c0 = random.randint(col_min, col_max)
        global_cells = [(r0 + r, c0 + c) for r, c in local_cells]
        if not _can_place_nonblack(global_cells, occupied, reserved):
            continue

        marker = (r0 + spec["marker"][0], c0 + spec["marker"][1])
        avail = _distance_to_border(marker, direction, (h, w))
        steps = min(ray_len, avail)
        corridor: list[tuple[int, int]] = []
        rr, cc = marker
        ok = True
        for _ in range(steps):
            rr += direction[0]
            cc += direction[1]
            if occupied[rr, cc] or reserved[rr, cc]:
                ok = False
                break
            corridor.append((rr, cc))
        if not ok:
            continue

        for (gr, gc), (lr, lc) in zip(global_cells, local_cells):
            grid[gr, gc] = int(patch[lr, lc])
            occupied[gr, gc] = True
        for rr, cc in corridor:
            reserved[rr, cc] = True

        return {
            "color": int(spec["color"]),
            "marker": marker,
            "direction": direction,
            "length": ray_len,
        }
    return None


def _manual_output(inp: np.ndarray, placed_regions: list[dict]) -> np.ndarray:
    out = inp.copy()
    h, w = inp.shape
    for region in placed_regions:
        rr, cc = region["marker"]
        dr, dc = region["direction"]
        color = int(region["color"])
        for _ in range(int(region["length"])):
            rr += dr
            cc += dc
            if not _in_bounds((h, w), rr, cc):
                break
            if int(inp[rr, cc]) != BLACK:
                break
            out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    region_count = random.choice([1, 2, 2, 3, 3, 4])
    min_dim = min(30, 12 + 3 * region_count)
    if min_dim > 30:
        return None
    h = random.randint(min_dim, 30)
    w = random.randint(min_dim, 30)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)

    colors = random.sample(PALETTE, region_count)
    directions = random.sample(list(DIRS), region_count)
    target_clipped = random.randint(0, region_count)

    placed_regions: list[dict] = []
    clipped_used = 0
    for idx, (color, direction) in enumerate(zip(colors, directions)):
        remaining = region_count - idx
        need_clip = clipped_used < target_clipped and target_clipped - clipped_used >= remaining

        placed = None
        for _ in range(REGION_SHAPE_ATTEMPTS):
            spec = _sample_region_spec(color, direction)
            if spec is None:
                continue

            clip_possible = _offset_ranges((h, w), spec["patch"].shape, direction, spec["length"], True) is not None
            full_possible = _offset_ranges((h, w), spec["patch"].shape, direction, spec["length"], False) is not None
            modes: list[bool] = []
            if need_clip and clip_possible:
                modes = [True, False] if full_possible else [True]
            else:
                if clip_possible:
                    modes.append(True)
                if full_possible:
                    modes.append(False)
                random.shuffle(modes)
            for clipped in modes:
                maybe = _place_region(grid, occupied, reserved, spec, clipped)
                if maybe is None:
                    continue
                placed = maybe
                if clipped:
                    clipped_used += 1
                break
            if placed is not None:
                break
        if placed is None:
            return None
        placed_regions.append(placed)

    expected = _manual_output(grid, placed_regions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
