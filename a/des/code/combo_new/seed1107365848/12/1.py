import random

import numpy as np


GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (1, 3)
OBJECT_HEIGHT_RANGE = (7, 13)
OBJECT_WIDTH_RANGE = (7, 13)
PLACEMENT_PADDING = 2
PLACEMENT_TRIES = 120
SHAPE_TRIES = 150
SEED_PICK_TRIES = 80


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                out[r, c] = True
                break
    return out


def _is_ring_like(mask: np.ndarray) -> bool:
    if int(mask.sum()) < 4:
        return False
    rows, cols = np.where(mask)
    if rows.size == 0:
        return False
    if int(rows.max() - rows.min()) < 1 or int(cols.max() - cols.min()) < 1:
        return False
    if not _is_connected(mask):
        return False
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    if np.any((row_counts > 0) & (row_counts < 2)):
        return False
    if np.any((col_counts > 0) & (col_counts < 2)):
        return False
    return True


def _extract_layers(region: np.ndarray) -> list[np.ndarray]:
    current = region.copy()
    layers: list[np.ndarray] = []
    while np.any(current):
        layer = _boundary(current)
        if not _is_ring_like(layer):
            break
        layers.append(layer)
        current = current & (~layer)
    return layers


def _enclosed_region(component_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    h, w = component_mask.shape
    rows, cols = np.where(component_mask)
    if rows.size == 0:
        return None
    r0 = max(0, int(rows.min()) - 1)
    r1 = min(h - 1, int(rows.max()) + 1)
    c0 = max(0, int(cols.min()) - 1)
    c1 = min(w - 1, int(cols.max()) + 1)

    block = component_mask[r0 : r1 + 1, c0 : c1 + 1]
    open_mask = ~block
    hh, ww = open_mask.shape
    outside = np.zeros((hh, ww), dtype=bool)
    stack: list[tuple[int, int]] = []

    for rr in range(hh):
        for cc in (0, ww - 1):
            if open_mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    for cc in range(ww):
        for rr in (0, hh - 1):
            if open_mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))

    while stack:
        rr, cc = stack.pop()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < hh and 0 <= nc < ww and open_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    enclosed = open_mask & (~outside)
    if not np.any(enclosed):
        return None
    if len(_components(enclosed)) != 1:
        return None
    return enclosed, (r0, c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint inward-shrinking copies of each closed fence, using the enclosed seed colors as ordered layer labels.

    Input structure:
    - The background is black.
    - One or more single-color, one-pixel-wide closed fences appear as separate 4-connected loops.
    - Each fence encloses several colored seed pixels that are not the fence color, not the background, and not adjacent to the fence.

    Rule:
    - For each closed fence, find the cells enclosed by that loop and collect its seed pixels in row-major order.
    - Repeatedly take the current enclosed region's inner boundary as the next shrunken fence layer.
    - Color the first layer with the first seed color, the second layer with the second seed color, and so on, stopping early if no complete inner ring can be formed.

    Key insight:
    - The number of seeds controls how many inward layers are attempted, while the visible top-left-to-bottom-right order of the seeds controls the color sequence of those layers.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        color_mask = g == color
        for comp in _components(color_mask):
            if len(comp) < 4:
                continue
            comp_mask = np.zeros_like(color_mask, dtype=bool)
            for r, c in comp:
                comp_mask[r, c] = True
            enclosed_info = _enclosed_region(comp_mask)
            if enclosed_info is None:
                continue
            enclosed_local, (r0, c0) = enclosed_info
            local_grid = g[r0 : r0 + enclosed_local.shape[0], c0 : c0 + enclosed_local.shape[1]]
            seeds: list[tuple[int, int, int]] = []
            for rr, cc in np.argwhere(enclosed_local):
                rr = int(rr)
                cc = int(cc)
                value = int(local_grid[rr, cc])
                if value != bg and value != color:
                    seeds.append((r0 + rr, c0 + cc, value))
            if not seeds:
                continue
            seeds.sort(key=lambda item: (item[0], item[1]))

            region = np.zeros_like(g, dtype=bool)
            region[r0 : r0 + enclosed_local.shape[0], c0 : c0 + enclosed_local.shape[1]] = enclosed_local
            layers = _extract_layers(region)
            for layer_mask, (_, _, seed_color) in zip(layers, seeds):
                out[layer_mask] = seed_color

    return out


def _trim_corner(mask: np.ndarray, size: int, corner: str) -> None:
    if size <= 0:
        return
    h, w = mask.shape
    for dr in range(size):
        width = size - dr
        for dc in range(width):
            if corner == "tl":
                mask[dr, dc] = False
            elif corner == "tr":
                mask[dr, w - 1 - dc] = False
            elif corner == "bl":
                mask[h - 1 - dr, dc] = False
            else:
                mask[h - 1 - dr, w - 1 - dc] = False


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_TRIES):
        h = random.randint(*OBJECT_HEIGHT_RANGE)
        w = random.randint(*OBJECT_WIDTH_RANGE)
        mask = np.ones((h, w), dtype=bool)
        family = random.choice(("rect", "trimmed"))
        if family == "trimmed":
            max_cut = min(2, (min(h, w) - 5) // 2)
            if max_cut <= 0:
                continue
            cuts = [random.randint(0, max_cut) for _ in range(4)]
            if not any(cuts):
                cuts[random.randrange(4)] = 1
            _trim_corner(mask, cuts[0], "tl")
            _trim_corner(mask, cuts[1], "tr")
            _trim_corner(mask, cuts[2], "bl")
            _trim_corner(mask, cuts[3], "br")

        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        if not _is_connected(mask):
            continue

        fence = _boundary(mask)
        if not _is_ring_like(fence):
            continue

        interior = mask & (~fence)
        if not np.any(interior):
            continue

        layers = _extract_layers(interior)
        if not layers:
            continue

        seed_zone = interior & (~layers[0])
        if int(seed_zone.sum()) == 0:
            continue

        if not _is_connected(interior):
            continue

        return mask
    return None


def _pick_spaced_positions(mask: np.ndarray, n: int) -> list[tuple[int, int]] | None:
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    for _ in range(SEED_PICK_TRIES):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        blocked: set[tuple[int, int]] = set()
        for r, c in cells:
            if (r, c) in blocked:
                continue
            chosen.append((r, c))
            for nr, nc in _neighbors8(r, c):
                blocked.add((nr, nc))
            blocked.add((r, c))
            if len(chosen) == n:
                return chosen
    return None


def _max_spaced_count(mask: np.ndarray) -> int:
    best = 0
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    for _ in range(SEED_PICK_TRIES):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        blocked: set[tuple[int, int]] = set()
        for r, c in cells:
            if (r, c) in blocked:
                continue
            chosen.append((r, c))
            for nr, nc in _neighbors8(r, c):
                blocked.add((nr, nc))
            blocked.add((r, c))
        best = max(best, len(chosen))
    return best


def _make_object(fence_color: int) -> dict[str, object] | None:
    shape = _random_shape()
    if shape is None:
        return None

    fence = _boundary(shape)
    interior = shape & (~fence)
    layers = _extract_layers(interior)
    seed_zone = interior & (~layers[0])

    max_seeds = min(8, _max_spaced_count(seed_zone))
    if max_seeds <= 0:
        return None

    layer_count = len(layers)
    overflow_ok = max_seeds > layer_count
    use_overflow = overflow_ok and random.choice([False, True])

    if use_overflow:
        seed_count = random.randint(layer_count + 1, max_seeds)
    else:
        seed_count = random.randint(1, min(layer_count, max_seeds))

    positions = _pick_spaced_positions(seed_zone, seed_count)
    if positions is None:
        return None

    bg = 0
    palette = [c for c in range(1, 10) if c != fence_color]
    if seed_count > len(palette):
        return None
    seed_colors = random.sample(palette, seed_count)

    patch = np.zeros(shape.shape, dtype=int)
    patch[fence] = fence_color
    for (r, c), color in zip(positions, seed_colors):
        patch[r, c] = int(color)

    return {
        "patch": patch,
        "height": int(patch.shape[0]),
        "width": int(patch.shape[1]),
    }


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < 1 or left < 1 or top + height >= h or left + width >= w:
        return False
    r0 = max(0, top - PLACEMENT_PADDING)
    r1 = min(h, top + height + PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    c1 = min(w, left + width + PLACEMENT_PADDING)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(PLACEMENT_TRIES):
        top = random.randint(1, h - ph - 1)
        left = random.randint(1, w - pw - 1)
        if not _can_place(occupied, top, left, ph, pw):
            continue
        mask = patch != 0
        grid[top : top + ph, left : left + pw][mask] = patch[mask]
        occupied[top : top + ph, left : left + pw] = True
        return True
    return False


def _generate_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    fence_color = random.randint(1, 9)
    objects: list[dict[str, object]] = []
    for _ in range(object_count):
        obj = _make_object(fence_color)
        if obj is None:
            return None
        objects.append(obj)

    max_obj_h = max(int(obj["height"]) for obj in objects)
    max_obj_w = max(int(obj["width"]) for obj in objects)
    min_h = max(GRID_MIN, max_obj_h + 2)
    min_w = max(GRID_MIN, max_obj_w + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = random.randint(min_h, GRID_MAX)
    grid_w = random.randint(min_w, GRID_MAX)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    objects.sort(key=lambda item: int(item["height"]) * int(item["width"]), reverse=True)
    for obj in objects:
        if not _place_patch(grid, occupied, np.asarray(obj["patch"], dtype=int)):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
