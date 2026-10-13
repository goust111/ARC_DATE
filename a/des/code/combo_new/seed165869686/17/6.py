from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    height, width = map(int, mask.shape)
    count = 0
    for comp in _components(~mask):
        if any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in comp):
            continue
        count += 1
    return count


def _extract_objects(grid: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    objects: list[tuple[int, int, int, np.ndarray]] = []
    for cells in _components(grid != 0):
        colors = {int(grid[r, c]) for r, c in cells}
        assert len(colors) == 1, "each object must be monochromatic"
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(cells)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        hole_count = _count_holes(crop == color)
        objects.append((hole_count, int(c0), int(r0), crop))
    return objects


def _pack_objects(objects: list[tuple[int, np.ndarray]]) -> np.ndarray:
    ordered = sorted(objects, key=lambda item: item[0])
    out_h = max(sprite.shape[0] for _, sprite in ordered)
    out_w = sum(sprite.shape[1] for _, sprite in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for _hole_count, sprite in ordered:
        sh, sw = map(int, sprite.shape)
        out[out_h - sh : out_h, col : col + sw] = sprite
        col += sw + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the colored objects by how many enclosed black holes they contain, then repack them along the bottom.

    Input structure:
    - The background is black.
    - Several separated 4-connected colored objects appear on the grid.
    - Each object is monochromatic and may contain zero or more fully enclosed black holes.
    - Different objects do not touch each other.

    Rule:
    - Find every colored object and count the number of enclosed black regions inside its bounding box.
    - Sort the objects from fewer holes to more holes.
    - Build a black output whose height is the maximum object bounding-box height and whose width is the sum of all object bounding-box widths plus one black separator column between neighbors.
    - Copy each object's original cropped shape into the output from left to right, bottom-aligned, without changing its color or geometry.

    Key insight:
    - The discriminating feature is topological rather than geometric: only black regions fully sealed inside an object count as holes, so the output order depends on enclosed cavities, not on color, size, or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    extracted = _extract_objects(g)
    assert extracted, "expected at least one object"
    objects = [(hole_count, sprite) for hole_count, _c0, _r0, sprite in extracted]
    return _pack_objects(objects)


# --------------- generator ---------------

ARC_MAX = 30
BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
OBJECT_COUNT_RANGE = (3, 5)
HOLE_COUNT_POOL = (1, 2, 3, 4)
GEN_RETRY_LIMIT = 400
PLACEMENT_TRIES = 200


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _add_protrusions(mask: np.ndarray, max_arms: int, max_length: int) -> np.ndarray:
    canvas = np.pad(mask, max_length, constant_values=False)
    height, width = map(int, canvas.shape)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(random.randint(0, max_arms)):
        dr, dc = random.choice(directions)
        candidates = []
        for r in range(height):
            for c in range(width):
                if not bool(canvas[r, c]):
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and bool(canvas[nr, nc]):
                    continue
                candidates.append((r, c))
        if not candidates:
            continue
        r, c = random.choice(candidates)
        length = random.randint(1, max_length)
        for step in range(1, length + 1):
            rr = r + dr * step
            cc = c + dc * step
            if 0 <= rr < height and 0 <= cc < width:
                canvas[rr, cc] = True

    return _tight_crop_mask(canvas)


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], height, width))

    while int(mask.sum()) < area:
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c, height, width):
            if not bool(mask[nr, nc]):
                frontier.add((int(nr), int(nc)))

    return _tight_crop_mask(mask)


def _make_zero_hole_mask() -> np.ndarray:
    for _ in range(100):
        height = random.randint(2, 6)
        width = random.randint(2, 6)
        area = random.randint(max(4, (height + width) // 2), min(height * width, 11))
        mask = _grow_connected_mask(height, width, area)
        if mask is None or _count_holes(mask) != 0:
            continue
        if random.random() < 0.35:
            scale = 2
            mask = np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)
        mask = _add_protrusions(mask, max_arms=2, max_length=2)
        mask = _random_transform(mask)
        if _count_holes(mask) == 0:
            return mask

    fallback = np.ones((random.randint(2, 4), random.randint(3, 5)), dtype=bool)
    return _random_transform(fallback)


def _hole_layouts(hole_count: int) -> list[tuple[int, int]]:
    layouts = []
    for rows in range(1, hole_count + 1):
        if hole_count % rows == 0:
            layouts.append((rows, hole_count // rows))
    layouts.sort(key=lambda rc: abs(rc[0] - rc[1]))
    return layouts


def _make_holey_mask(hole_count: int) -> np.ndarray:
    assert hole_count > 0
    for _ in range(100):
        layouts = _hole_layouts(hole_count)
        if len(layouts) > 1 and random.random() < 0.7:
            rows, cols = random.choice(layouts[:2])
        else:
            rows, cols = random.choice(layouts)

        base = np.ones((2 * rows + 1, 2 * cols + 1), dtype=bool)
        for rr in range(rows):
            for cc in range(cols):
                base[2 * rr + 1, 2 * cc + 1] = False

        scale = 1
        if hole_count == 1 and random.random() < 0.6:
            scale = 2
        elif hole_count == 2 and random.random() < 0.35:
            scale = 2
        if scale > 1:
            base = np.kron(base.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)

        base = _add_protrusions(base, max_arms=3, max_length=2)
        base = _random_transform(base)
        if _count_holes(base) == hole_count:
            return base

    raise RuntimeError(f"failed to generate a {hole_count}-hole object")


def _make_sprite(hole_count: int, color: int) -> np.ndarray:
    if hole_count == 0:
        mask = _make_zero_hole_mask()
    else:
        mask = _make_holey_mask(hole_count)
    return np.where(mask, int(color), BACKGROUND).astype(int)


def _place_sprites(sprites: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    order = sorted(sprites, key=lambda sprite: sprite.shape[0] * sprite.shape[1], reverse=True)
    for sprite in order:
        sh, sw = map(int, sprite.shape)
        if sh + 2 > height or sw + 2 > width:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r = random.randint(1, height - sh - 1)
            c = random.randint(1, width - sw - 1)
            if reserved[r : r + sh, c : c + sw].any():
                continue

            grid[r : r + sh, c : c + sw][sprite != BACKGROUND] = sprite[sprite != BACKGROUND]
            reserved[max(0, r - 1) : min(height, r + sh + 1), max(0, c - 1) : min(width, c + sw + 1)] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _expected_output(specs: list[tuple[int, np.ndarray]]) -> np.ndarray:
    return _pack_objects(specs)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    hole_counts = [0] + random.sample(list(HOLE_COUNT_POOL), n_objects - 1)
    colors = random.sample(list(COLOR_POOL), n_objects)
    specs: list[tuple[int, np.ndarray]] = []

    for hole_count, color in zip(hole_counts, colors, strict=False):
        sprite = _make_sprite(hole_count, color)
        specs.append((int(hole_count), sprite))

    expected = _expected_output(specs)
    if max(expected.shape) > ARC_MAX:
        return None

    max_h = max(sprite.shape[0] for _, sprite in specs)
    max_w = max(sprite.shape[1] for _, sprite in specs)
    out_w = expected.shape[1]

    min_h = max(max_h + 2, 10)
    max_h_grid = min(ARC_MAX, max_h + 10)
    min_w = max(out_w, max_w + 2, 12)
    max_w_grid = min(ARC_MAX, out_w + 8)
    if min_h > max_h_grid or min_w > max_w_grid:
        return None

    grid_h = random.randint(min_h, max_h_grid)
    grid_w = random.randint(min_w, max_w_grid)
    grid = _place_sprites([sprite for _, sprite in specs], grid_h, grid_w)
    if grid is None:
        return None

    extracted = _extract_objects(grid)
    if len(extracted) != n_objects:
        return None

    actual_counts = sorted(hole_count for hole_count, _c0, _r0, _sprite in extracted)
    if actual_counts != sorted(hole_counts):
        return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRY_LIMIT} attempts")
