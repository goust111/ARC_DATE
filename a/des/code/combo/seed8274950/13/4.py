from __future__ import annotations

import random
from collections import deque

import numpy as np


MAX_GEN_ATTEMPTS = 300
OUTPUT_SIDE_MAX = 28


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Keep only the external contour by filling enclosed voids."""
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    seen = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1]:
                if padded[nr, nc] or seen[nr, nc]:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))

    filled = padded.copy()
    filled[(~padded) & (~seen)] = True
    return filled[1:-1, 1:-1]


def count_outline_corners(mask: np.ndarray) -> int:
    """
    Count 90-degree turns on the object's outer orthogonal contour.

    In a padded 2x2 window view, every contour turn appears as a 2x2 block
    containing exactly one filled cell (convex corner) or three filled cells
    (concave corner).
    """
    solid = fill_holes(mask)
    padded = np.pad(solid.astype(np.uint8), 1, constant_values=0)
    corners = 0
    for r in range(padded.shape[0] - 1):
        for c in range(padded.shape[1] - 1):
            total = int(padded[r : r + 2, c : c + 2].sum())
            if total == 1 or total == 3:
                corners += 1
    return corners


def scale_for_corners(corners: int) -> int:
    if corners == 4:
        return 1
    if corners >= 12:
        return 3
    return 2


def find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects: list[dict[str, object]] = []

    for color_value in np.unique(g):
        color = int(color_value)
        if color == bg:
            continue
        mask = g == color
        seen = np.zeros_like(mask, dtype=bool)

        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue

                stack = [(r, c)]
                seen[r, c] = True
                cells: list[tuple[int, int]] = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for nr, nc in neighbors4(cr, cc):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

                rows = [rr for rr, _ in cells]
                cols = [cc for _, cc in cells]
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)
                sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
                for rr, cc in cells:
                    sprite[rr - r0, cc - c0] = color

                objects.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "sprite": sprite,
                        "mask": sprite == color,
                    }
                )

    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    return objects


def paste(out: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = sprite.shape
    window = out[top : top + h, left : left + w]
    mask = sprite != bg
    window[mask] = sprite[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each object by outer-corner count, scale it, then pack left-to-right.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_objects(g, bg)
    assert objects, "expected at least one foreground object"

    scaled_sprites: list[np.ndarray] = []
    for obj in objects:
        corners = count_outline_corners(obj["mask"])
        scale = scale_for_corners(corners)
        sprite = np.repeat(np.repeat(obj["sprite"], scale, axis=0), scale, axis=1)
        scaled_sprites.append(sprite)

    out_h = max(sprite.shape[0] for sprite in scaled_sprites)
    out_w = sum(sprite.shape[1] for sprite in scaled_sprites) + len(scaled_sprites) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for sprite in scaled_sprites:
        top = out_h - sprite.shape[0]
        paste(out, sprite, top, col, bg)
        col += sprite.shape[1] + 1

    return out


def _inflate(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask.copy()
    block = np.ones((factor, factor), dtype=np.uint8)
    return np.kron(mask.astype(np.uint8), block).astype(bool)


def _random_orient(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _make_rectangle(rng: random.Random) -> np.ndarray:
    return np.ones((rng.randint(1, 4), rng.randint(1, 5)), dtype=bool)


def _make_t(rng: random.Random) -> np.ndarray:
    thickness = rng.randint(1, 2)
    left_arm = rng.randint(1, 2)
    right_arm = rng.randint(1, 2)
    stem_depth = rng.randint(1, 3)
    h = thickness + stem_depth
    w = left_arm + thickness + right_arm
    mask = np.zeros((h, w), dtype=bool)
    mask[:thickness, :] = True
    mask[:, left_arm : left_arm + thickness] = True
    return mask


def _make_u(rng: random.Random) -> np.ndarray:
    thickness = rng.randint(1, 2)
    inner_gap = rng.randint(1, 2)
    leg_height = rng.randint(1, 3)
    h = leg_height + thickness
    w = 2 * thickness + inner_gap
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thickness] = True
    mask[:, -thickness:] = True
    mask[-thickness:, :] = True
    return mask


def _make_plus(rng: random.Random) -> np.ndarray:
    thickness = rng.randint(1, 2)
    up = rng.randint(1, 2)
    down = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = rng.randint(1, 2)
    h = up + thickness + down
    w = left + thickness + right
    mask = np.zeros((h, w), dtype=bool)
    mask[:, left : left + thickness] = True
    mask[up : up + thickness, :] = True
    return mask


def _make_h(rng: random.Random) -> np.ndarray:
    thickness = rng.randint(1, 2)
    gap = rng.randint(1, 2)
    top = rng.randint(1, 2)
    bottom = rng.randint(1, 2)
    h = top + thickness + bottom
    w = 2 * thickness + gap
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thickness] = True
    mask[:, -thickness:] = True
    mask[top : top + thickness, :] = True
    return mask


def _sample_shape(category: int, rng: random.Random) -> tuple[np.ndarray, int] | None:
    builders = {
        0: [_make_rectangle],
        1: [_make_t, _make_u],
        2: [_make_plus, _make_h],
    }[category]

    for _ in range(40):
        mask = builders[rng.randrange(len(builders))](rng)
        if rng.random() < 0.45:
            mask = _inflate(mask, 2)
        mask = _random_orient(mask, rng)
        corners = count_outline_corners(mask)
        scale = scale_for_corners(corners)
        if category == 0 and scale == 1:
            return mask, corners
        if category == 1 and scale == 2 and 4 < corners < 12:
            return mask, corners
        if category == 2 and scale == 3 and corners >= 12:
            return mask, corners

    return None


def _choose_categories(n_objects: int, rng: random.Random) -> list[int]:
    categories = [0, 1, 2]
    rng.shuffle(categories)
    chosen = categories[: min(n_objects, 3)]
    while len(chosen) < n_objects:
        chosen.append(rng.randrange(3))
    rng.shuffle(chosen)
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    n_objects = rng.choice([2, 3])
    categories = _choose_categories(n_objects, rng)
    colors = rng.sample([c for c in range(10) if c != bg], n_objects)

    objects: list[tuple[np.ndarray, int, int]] = []
    for color, category in zip(colors, categories):
        sampled = _sample_shape(category, rng)
        if sampled is None:
            return None
        mask, corners = sampled
        objects.append((mask, color, corners))

    out_h = max(mask.shape[0] * scale_for_corners(corners) for mask, _color, corners in objects)
    out_w = sum(mask.shape[1] * scale_for_corners(corners) for mask, _color, corners in objects) + (n_objects - 1)
    if out_h > OUTPUT_SIDE_MAX or out_w > OUTPUT_SIDE_MAX:
        return None

    widths = [mask.shape[1] for mask, _color, _corners in objects]
    heights = [mask.shape[0] for mask, _color, _corners in objects]
    gaps = [rng.randint(2, 4) for _ in range(n_objects - 1)]
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 4)
    grid_w = left_margin + sum(widths) + sum(gaps) + right_margin
    grid_h = max(heights) + rng.randint(2, 7)
    if grid_w > 28 or grid_h > 28:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    col = left_margin
    for i, (mask, color, _corners) in enumerate(objects):
        h, w = mask.shape
        top = rng.randint(1, grid_h - h - 1)
        grid[top : top + h, col : col + w][mask] = color
        if i < n_objects - 1:
            col += w + gaps[i]

    if get_bg(grid) != bg:
        return None
    if len(find_objects(grid, bg)) != n_objects:
        return None

    out = main(grid)
    if out.shape[0] > OUTPUT_SIDE_MAX or out.shape[1] > OUTPUT_SIDE_MAX:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
