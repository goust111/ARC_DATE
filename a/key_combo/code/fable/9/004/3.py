from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


BG = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("left", "right", "top", "bottom")
DIR_STEP = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _component_bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _component_bbox(coords),
                }
            )

    return components


def _is_block_component(component: dict[str, object]) -> bool:
    coords = list(component["coords"])
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = component["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return set(coords) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _direction_from_border(shape: tuple[int, int], cell: Coord) -> tuple[int, int]:
    h, w = shape
    r, c = cell
    dirs = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    assert len(dirs) == 1, "edge emitters must lie on exactly one border side"
    return dirs[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay each edge emitter's color through ordered swap pixels and paint the hit 2x2 block with the final carried color.

    Input structure:
    - The background is black.
    - Several singleton colored emitters sit on the outer border and each points inward along its row or column.
    - Along each invisible ray lie zero or more background cells, several singleton colored swap pixels, and exactly one filled 2x2 grey block at the end.

    Rule:
    - Each border emitter launches one invisible ray inward carrying the emitter's color.
    - Whenever the ray reaches a singleton swap pixel, the pixel takes the carried color and the ray takes the pixel's previous color.
    - When the ray first reaches its 2x2 grey block, stop and recolor the whole block with the ray's current carried color; emitters stay fixed and empty path cells remain background.

    Key insight:
    - The colored dots on the path are not simple paint targets: they form an ordered color-exchange relay, so both the intermediate dots and the terminal block depend on the ray's traversal order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected foreground objects"

    block_components = [comp for comp in components if _is_block_component(comp)]
    assert block_components, "expected at least one 2x2 block"
    assert len({int(comp["color"]) for comp in block_components}) == 1, "all terminal blocks must share one color"

    block_cells: dict[Coord, int] = {}
    block_bboxes: list[BBox] = []
    for idx, comp in enumerate(block_components):
        bbox = tuple(map(int, comp["bbox"]))
        block_bboxes.append(bbox)
        for cell in comp["coords"]:
            block_cells[cell] = idx

    emitters: list[Coord] = []
    for comp in components:
        if comp in block_components:
            continue
        coords = list(comp["coords"])
        assert len(coords) == 1, "all non-block foreground objects must be singleton emitters or swap pixels"
        r, c = coords[0]
        if r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1):
            emitters.append((r, c))

    assert emitters, "expected at least one border emitter"

    out = g.copy()
    hits_per_block = [0] * len(block_bboxes)
    for emitter in emitters:
        dr, dc = _direction_from_border(g.shape, emitter)
        carried = int(out[emitter])
        rr, cc = emitter[0] + dr, emitter[1] + dc
        hit_block = False

        while _in_bounds(g.shape, rr, cc):
            if (rr, cc) in block_cells:
                idx = block_cells[(rr, cc)]
                hits_per_block[idx] += 1
                r0, c0, r1, c1 = block_bboxes[idx]
                out[r0 : r1 + 1, c0 : c1 + 1] = carried
                hit_block = True
                break

            if int(g[rr, cc]) != bg:
                previous = int(out[rr, cc])
                out[rr, cc] = carried
                carried = previous

            rr += dr
            cc += dc

        assert hit_block, "each emitter ray must terminate at a block"

    assert all(count == 1 for count in hits_per_block), "each block must be hit by exactly one ray"
    return out


def _pick_swap_indices(path_len: int, swap_count: int) -> list[int] | None:
    if path_len < 2 * swap_count + 1:
        return None
    upper = path_len - swap_count - 1
    if upper < swap_count:
        return None
    base = sorted(random.sample(range(1, upper + 1), swap_count))
    return [value + i for i, value in enumerate(base)]


def _candidate_path(direction: str, h: int, w: int, block_top_left: Coord, hit_offset: int) -> tuple[Coord, list[Coord]]:
    br, bc = block_top_left
    if direction == "left":
        row = br + hit_offset
        emitter = (row, 0)
        path = [(row, c) for c in range(1, bc)]
    elif direction == "right":
        row = br + hit_offset
        emitter = (row, w - 1)
        path = [(row, c) for c in range(w - 2, bc + 1, -1)]
    elif direction == "top":
        col = bc + hit_offset
        emitter = (0, col)
        path = [(r, col) for r in range(1, br)]
    else:
        col = bc + hit_offset
        emitter = (h - 1, col)
        path = [(r, col) for r in range(h - 2, br + 1, -1)]
    return emitter, path


def _nonbg_well_spaced(existing: set[Coord], new_cells: list[Coord], shape: tuple[int, int]) -> bool:
    for r, c in new_cells:
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(shape, nr, nc) and (nr, nc) in existing:
                return False
    return True


def _gen_direction_sequence(ray_count: int) -> list[str]:
    dirs = [random.choice(("left", "right")), random.choice(("top", "bottom"))]
    while len(dirs) < ray_count:
        dirs.append(random.choice(DIRECTIONS))
    random.shuffle(dirs)
    return dirs


def _choose_colors_for_ray(swap_count: int) -> tuple[int, list[int]]:
    emitter_color = random.choice(PALETTE)
    carried = emitter_color
    swaps: list[int] = []
    for _ in range(swap_count):
        choices = [color for color in PALETTE if color != carried]
        color = random.choice(choices)
        swaps.append(color)
        carried = color
    return emitter_color, swaps


def _place_ray(
    grid: np.ndarray,
    direction: str,
    swap_count: int,
    occupied_nonbg: set[Coord],
    reserved_path: set[Coord],
) -> bool:
    h, w = grid.shape
    required_len = 2 * swap_count + 1

    for _ in range(250):
        hit_offset = random.randint(0, 1)

        if direction == "left":
            bc_min = required_len + 1
            if bc_min > w - 3:
                return False
            br = random.randint(1, h - 3)
            bc = random.randint(bc_min, w - 3)
        elif direction == "right":
            bc_max = w - required_len - 3
            if bc_max < 1:
                return False
            br = random.randint(1, h - 3)
            bc = random.randint(1, bc_max)
        elif direction == "top":
            br_min = required_len + 1
            if br_min > h - 3:
                return False
            br = random.randint(br_min, h - 3)
            bc = random.randint(1, w - 3)
        else:
            br_max = h - required_len - 3
            if br_max < 1:
                return False
            br = random.randint(1, br_max)
            bc = random.randint(1, w - 3)

        emitter, path = _candidate_path(direction, h, w, (br, bc), hit_offset)
        swap_indices = _pick_swap_indices(len(path), swap_count)
        if swap_indices is None:
            continue

        block_cells = [(br + dr, bc + dc) for dr in (0, 1) for dc in (0, 1)]
        swap_cells = [path[idx] for idx in swap_indices]
        new_nonbg = [emitter, *swap_cells, *block_cells]

        if any(cell in occupied_nonbg for cell in new_nonbg):
            continue
        if any(cell in reserved_path for cell in path + block_cells + [emitter]):
            continue
        if not _nonbg_well_spaced(occupied_nonbg, new_nonbg, grid.shape):
            continue

        emitter_color, swap_colors = _choose_colors_for_ray(swap_count)
        grid[emitter] = emitter_color
        for cell, color in zip(swap_cells, swap_colors):
            grid[cell] = color
        for cell in block_cells:
            grid[cell] = GRAY

        occupied_nonbg.update(new_nonbg)
        reserved_path.update(path)
        reserved_path.update(block_cells)
        reserved_path.add(emitter)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    grid = np.full((h, w), BG, dtype=int)

    ray_count = random.randint(2, 5)
    directions = _gen_direction_sequence(ray_count)
    complex_ray = random.randrange(ray_count)

    occupied_nonbg: set[Coord] = set()
    reserved_path: set[Coord] = set()
    swap_counts: list[int] = []

    for i, direction in enumerate(directions):
        if i == complex_ray:
            swap_count = random.randint(2, 4)
        else:
            swap_count = random.randint(1, 4)
        if not _place_ray(grid, direction, swap_count, occupied_nonbg, reserved_path):
            return None
        swap_counts.append(swap_count)

    if len(set(directions)) < 2:
        return None
    if not any(direction in ("left", "right") for direction in directions):
        return None
    if not any(direction in ("top", "bottom") for direction in directions):
        return None
    if max(swap_counts) < 2:
        return None

    non_bg_colors = {int(color) for color in np.unique(grid) if int(color) != BG}
    if len(non_bg_colors - {GRAY}) < 3:
        return None

    out = main(grid)
    block_mask = grid == GRAY
    swap_changed = np.count_nonzero((grid != out) & (grid != BG) & (grid != GRAY))
    block_changed = np.any(out[block_mask] != GRAY)
    if swap_changed == 0 or not block_changed:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid relay color-swap ray input")
