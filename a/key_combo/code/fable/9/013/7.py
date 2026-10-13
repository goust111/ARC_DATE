from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _solid_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    solid_rows = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if np.all(grid[r, :] == int(grid[r, 0]))
    ]
    solid_cols = [
        (c, int(grid[0, c]))
        for c in range(w)
        if np.all(grid[:, c] == int(grid[0, c]))
    ]
    crosses = [
        (r, c, color)
        for r, color in solid_rows
        for c, c_color in solid_cols
        if color == c_color
    ]
    assert len(crosses) == 1, "expected exactly one separator cross"
    r, c, color = crosses[0]
    assert 0 < r < h - 1 and 0 < c < w - 1, "separator must split four nonempty quadrants"
    return r, c, color


def _quadrant_slices(cross_r: int, cross_c: int, height: int, width: int):
    return [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, width)),
        (slice(cross_r + 1, height), slice(0, cross_c)),
        (slice(cross_r + 1, height), slice(cross_c + 1, width)),
    ]


def _analyze_quadrant(region: np.ndarray, bg: int) -> dict:
    colors = [int(c) for c in np.unique(region) if int(c) != int(bg)]
    sprites = []
    for color in colors:
        for coords in _components_of_color(region, color):
            bbox = _bbox_coords(coords)
            mask = _crop_mask(coords, bbox)
            sprites.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "mask": mask,
                    "shape": tuple(mask.shape),
                    "signature": (tuple(mask.shape), mask.tobytes()),
                }
            )

    assert len(sprites) >= 4, "each quadrant must contain several sprites"
    sprite_shapes = {sprite["shape"] for sprite in sprites}
    assert len(sprite_shapes) == 1, "all sprites in a quadrant must share one bbox size"

    shape_counts = Counter(sprite["signature"] for sprite in sprites)
    unique = [sprite for sprite in sprites if shape_counts[sprite["signature"]] == 1]
    assert len(unique) == 1, "expected exactly one unique shape per quadrant"

    color_counts = Counter(sprite["color"] for sprite in sprites)
    ranked = color_counts.most_common()
    assert len(ranked) >= 2 and ranked[0][1] > ranked[1][1], "quadrant majority color must be unique"

    return {
        "sprites": sprites,
        "sprite_shape": next(iter(sprite_shapes)),
        "unique_sprite": unique[0],
        "majority_color": int(ranked[0][0]),
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    cross_r, cross_c, separator = _solid_cross(g)

    non_separator = g[g != int(separator)]
    assert non_separator.size > 0, "grid must contain non-separator cells"
    bg = int(Counter(map(int, non_separator.reshape(-1))).most_common(1)[0][0])

    height, width = g.shape
    quads = []
    sprite_shapes = set()
    for rs, cs in _quadrant_slices(cross_r, cross_c, height, width):
        info = _analyze_quadrant(g[rs, cs], bg)
        quads.append(info)
        sprite_shapes.add(info["sprite_shape"])

    assert len(sprite_shapes) == 1, "all quadrants must use the same sprite size"
    return {
        "separator": int(separator),
        "bg": int(bg),
        "sprite_shape": next(iter(sprite_shapes)),
        "quadrants": quads,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique-shape sprite in each quadrant, recolor it from the diagonal quadrant, and compress the board.

    Input structure:
    - One solid full row and one solid full column of the same color form a separator cross.
    - The cross splits the canvas into four quadrants filled with background and several same-size monochrome sprites.
    - In each quadrant exactly one sprite shape appears once, while every other shape in that quadrant is duplicated or triplicated.

    Rule:
    - In each quadrant, identify the sprite whose binary shape occurs exactly once within that quadrant.
    - Recolor that sprite to the unique majority sprite color of the diagonally opposite quadrant.
    - Output a reduced four-cell board with the same separator color: each quadrant cell has one sprite-sized box containing the recolored unique sprite in its original orientation.

    Key insight:
    - Shape selection is local to each quadrant, but the final color comes from the opposite diagonal quadrant's color frequency, so neither local color counts nor original colors are sufficient.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    sprite_h, sprite_w = info["sprite_shape"]
    out = np.full((sprite_h * 2 + 1, sprite_w * 2 + 1), info["bg"], dtype=int)
    out[sprite_h, :] = info["separator"]
    out[:, sprite_w] = info["separator"]

    diagonal = [3, 2, 1, 0]
    anchors = [
        (0, 0),
        (0, sprite_w + 1),
        (sprite_h + 1, 0),
        (sprite_h + 1, sprite_w + 1),
    ]
    for i, (r0, c0) in enumerate(anchors):
        mask = info["quadrants"][i]["unique_sprite"]["mask"]
        color = info["quadrants"][diagonal[i]]["majority_color"]
        patch = np.where(mask, color, info["bg"]).astype(int)
        out[r0 : r0 + sprite_h, c0 : c0 + sprite_w] = patch
    return out


SPRITE_SIZE_OPTIONS = ((3, 3), (3, 4), (4, 3), (4, 4))
INPUT_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400


def _shape_area_bounds(h: int, w: int) -> tuple[int, int]:
    low = max(h + w - 2, int(np.ceil(0.45 * h * w)))
    high = max(low, h * w - 1)
    return low, high


def _connect_points(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    rr, cc = end
    if random.random() < 0.5:
        while r != rr:
            mask[r, c] = True
            r += 1 if rr > r else -1
        while c != cc:
            mask[r, c] = True
            c += 1 if cc > c else -1
    else:
        while c != cc:
            mask[r, c] = True
            c += 1 if cc > c else -1
        while r != rr:
            mask[r, c] = True
            r += 1 if rr > r else -1
    mask[rr, cc] = True


def _random_shape_mask(h: int, w: int) -> np.ndarray:
    low, high = _shape_area_bounds(h, w)
    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        border_points = [
            (0, random.randrange(w)),
            (random.randrange(h), w - 1),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
        ]
        order = border_points[:]
        random.shuffle(order)
        for a, b in zip(order, order[1:]):
            _connect_points(mask, a, b)
        target = random.randint(low, high)

        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            rr, cc = cell
            mask[rr, cc] = True
            for nr, nc in _neighbors4(rr, cc, h, w):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) < low:
            continue
        if int(mask.sum()) == h * w:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    raise RuntimeError("failed to generate a sprite shape")


def _shape_pool(h: int, w: int, count: int) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    seen = set()
    while len(pool) < count:
        mask = _random_shape_mask(h, w)
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        pool.append(mask)
    return pool


def _quadrant_partition(n_sprites: int) -> list[int]:
    if n_sprites == 4:
        return [3]
    if n_sprites == 5:
        return [2, 2]
    raise ValueError("unsupported sprite count")


def _quadrant_layout(n_sprites: int, sprite_h: int, sprite_w: int) -> tuple[int, int, list[tuple[int, int]], tuple[int, int]]:
    if n_sprites == 4:
        rows, cols = 2, 2
        slots = [(r, c) for r in range(rows) for c in range(cols)]
    else:
        rows, cols = 2, 3
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        slots.pop(random.randrange(len(slots)))

    coords = [(r * (sprite_h + 1), c * (sprite_w + 1)) for r, c in slots]
    height = rows * sprite_h + (rows - 1)
    width = cols * sprite_w + (cols - 1)
    return height, width, coords, (rows, cols)


def _assign_colors(
    n_sprites: int,
    majority_color: int,
    aux_colors: list[int],
    diagonal_majority: int,
) -> list[int]:
    low = (n_sprites // 2) + 1
    high = max(low, n_sprites - 2)
    majority_count = random.randint(low, high)
    branch_choices = ["majority"]
    if n_sprites - majority_count >= 2:
        branch_choices.append("secondary")
    branch = random.choice(branch_choices)
    secondary = aux_colors[0]
    tertiary = aux_colors[1]
    assert majority_color != diagonal_majority and secondary != diagonal_majority

    if branch == "majority":
        others = [majority_color] * (majority_count - 1)
        others.extend([secondary] * (n_sprites - majority_count))
        random.shuffle(others)
        colors = [majority_color] + others
    else:
        others = [majority_color] * majority_count
        others.append(secondary)
        remainder = n_sprites - majority_count - 2
        if remainder > 0:
            others.extend([tertiary] * remainder)
        random.shuffle(others)
        colors = [secondary] + others

    assert len(colors) == n_sprites
    return colors


def _place_mask(arr: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = arr[top : top + h, left : left + w]
    patch[mask] = int(color)


def _build_quadrant(
    width: int,
    height: int,
    sprite_h: int,
    sprite_w: int,
    bg: int,
    majority_color: int,
    diagonal_majority: int,
    aux_colors: list[int],
) -> tuple[np.ndarray, dict]:
    n_sprites = random.choice((4, 5))
    qh_need, qw_need, slot_tops, _ = _quadrant_layout(n_sprites, sprite_h, sprite_w)
    assert qh_need == height
    left_margin = random.randint(0, width - qw_need)

    pool = _shape_pool(sprite_h, sprite_w, 5)
    repeated_sizes = _quadrant_partition(n_sprites)
    unique_shape = pool[0]
    instance_masks = [unique_shape]
    for i, repeat_count in enumerate(repeated_sizes, start=1):
        instance_masks.extend([pool[i]] * repeat_count)

    colors = _assign_colors(n_sprites, majority_color, aux_colors, diagonal_majority)
    unique_color = colors[0]
    if unique_color == diagonal_majority:
        return None, {}

    if Counter(colors)[unique_color] < 2:
        return None, {}

    order = list(range(n_sprites))
    random.shuffle(order)
    grid = np.full((height, width), bg, dtype=int)
    placements = []
    for slot, idx in zip(slot_tops, order):
        top, left = slot
        top = int(top)
        left = int(left + left_margin)
        mask = instance_masks[idx]
        color = colors[idx]
        _place_mask(grid, mask, top, left, color)
        placements.append(
            {
                "top": top,
                "left": left,
                "mask": mask.copy(),
                "color": int(color),
                "is_unique": idx == 0,
            }
        )

    color_counts = Counter(p["color"] for p in placements)
    ranked = color_counts.most_common()
    if len(ranked) < 2 or ranked[0][1] <= ranked[1][1]:
        return None, {}
    if ranked[0][0] != majority_color:
        return None, {}

    unique_entries = [p for p in placements if p["is_unique"]]
    if len(unique_entries) != 1:
        return None, {}
    if unique_entries[0]["color"] == diagonal_majority:
        return None, {}

    return grid, {
        "unique_mask": unique_entries[0]["mask"],
        "unique_color": int(unique_entries[0]["color"]),
        "majority_color": int(majority_color),
        "n_sprites": int(n_sprites),
    }


def _expected_output(meta: list[dict], sprite_h: int, sprite_w: int, bg: int, separator: int) -> np.ndarray:
    out = np.full((sprite_h * 2 + 1, sprite_w * 2 + 1), bg, dtype=int)
    out[sprite_h, :] = separator
    out[:, sprite_w] = separator
    diagonal = [3, 2, 1, 0]
    anchors = [
        (0, 0),
        (0, sprite_w + 1),
        (sprite_h + 1, 0),
        (sprite_h + 1, sprite_w + 1),
    ]
    for i, (r0, c0) in enumerate(anchors):
        color = meta[diagonal[i]]["majority_color"]
        out[r0 : r0 + sprite_h, c0 : c0 + sprite_w][meta[i]["unique_mask"]] = color
    return out


def _semantic_checks(grid: np.ndarray, meta: list[dict], expected: np.ndarray) -> bool:
    try:
        analyzed = _analyze_grid(grid)
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if analyzed["sprite_shape"] != meta[0]["unique_mask"].shape:
        return False

    diagonal = [3, 2, 1, 0]
    for i, quad in enumerate(analyzed["quadrants"]):
        if quad["majority_color"] != meta[i]["majority_color"]:
            return False
        if not np.array_equal(quad["unique_sprite"]["mask"], meta[i]["unique_mask"]):
            return False
        if quad["unique_sprite"]["color"] != meta[i]["unique_color"]:
            return False
        if quad["unique_sprite"]["color"] == analyzed["quadrants"][diagonal[i]]["majority_color"]:
            return False
        local_color_counts = Counter(sprite["color"] for sprite in quad["sprites"])
        if local_color_counts[quad["unique_sprite"]["color"]] < 2:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    bg, separator = random.sample(INPUT_COLORS, 2)
    remaining = [c for c in INPUT_COLORS if c not in (bg, separator)]
    majority_colors = random.sample(remaining, 4)
    aux_palette = [c for c in remaining if c not in majority_colors]
    random.shuffle(aux_palette)

    sprite_h, sprite_w = random.choice(SPRITE_SIZE_OPTIONS)
    quad_height = 2 * sprite_h + 1
    width_need_5 = 3 * sprite_w + 2
    left_width = width_need_5
    right_width = width_need_5

    grid = np.full((quad_height * 2 + 1, left_width + right_width + 1), bg, dtype=int)
    cross_r = quad_height
    cross_c = left_width
    grid[cross_r, :] = separator
    grid[:, cross_c] = separator

    diagonal = [3, 2, 1, 0]
    quad_origins = [
        (0, 0, left_width),
        (0, left_width + 1, right_width),
        (quad_height + 1, 0, left_width),
        (quad_height + 1, left_width + 1, right_width),
    ]
    meta = [None] * 4

    for i, (r0, c0, qwidth) in enumerate(quad_origins):
        aux = aux_palette[:]
        random.shuffle(aux)
        quad, info = _build_quadrant(
            qwidth,
            quad_height,
            sprite_h,
            sprite_w,
            bg,
            majority_colors[i],
            majority_colors[diagonal[i]],
            aux,
        )
        if quad is None:
            return None
        grid[r0 : r0 + quad_height, c0 : c0 + qwidth] = quad
        meta[i] = info

    expected = _expected_output(meta, sprite_h, sprite_w, bg, separator)
    if not _semantic_checks(grid, meta, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
