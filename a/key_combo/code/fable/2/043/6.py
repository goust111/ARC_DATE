from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tight_crop(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    box = _bbox(grid != bg)
    if box is None:
        return np.empty((0, 0), dtype=int)
    return _crop_bbox(grid, box)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _identity(arr: np.ndarray) -> np.ndarray:
    return arr.copy()


TRANSFORMS = {
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "flip_lr": np.fliplr,
    "flip_ud": np.flipud,
    "transpose": np.transpose,
    "anti_transpose": lambda a: np.rot90(a, 2).T,
}


ALL_TRANSFORMS = {"identity": _identity, **TRANSFORMS}


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(ALL_TRANSFORMS[name](arr), dtype=int)


def _apply_color_map(arr: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in color_map.items():
        out[arr == int(src)] = int(dst)
    return out


def _split_sections(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    sep_rows = [
        r for r in range(h)
        if int(g[r, 0]) != bg and bool(np.all(g[r, :] == g[r, 0]))
    ]
    if len(sep_rows) == 2:
        r0, r1 = map(int, sep_rows)
        return [g[:r0, :], g[r0 + 1 : r1, :], g[r1 + 1 :, :]]

    sep_cols = [
        c for c in range(w)
        if int(g[0, c]) != bg and bool(np.all(g[:, c] == g[0, c]))
    ]
    if len(sep_cols) == 2:
        c0, c1 = map(int, sep_cols)
        return [g[:, :c0], g[:, c0 + 1 : c1], g[:, c1 + 1 :]]

    raise ValueError("expected exactly two full-length separator lines")


def _extract_sprites(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    sections = _split_sections(grid)
    bg = _get_bg(grid)
    if len(sections) != 3:
        raise ValueError("expected three sections")
    sprites = []
    for section in sections:
        sprite = _tight_crop(section, bg)
        if sprite.size == 0:
            raise ValueError("section is empty")
        sprites.append(sprite)
    return sprites[0], sprites[1], sprites[2], bg


def _valid_demo_candidates(before: np.ndarray, after: np.ndarray, bg: int) -> list[tuple[str, dict[int, int]]]:
    after_mask = after != bg
    candidates: list[tuple[str, dict[int, int]]] = []
    for name in TRANSFORMS:
        transformed = _apply_transform(before, name)
        if transformed.shape != after.shape:
            continue
        transformed_mask = transformed != bg
        if not np.array_equal(transformed_mask, after_mask):
            continue

        color_map: dict[int, int] = {}
        ok = True
        for src in map(int, np.unique(transformed[transformed_mask])):
            dst_vals = np.unique(after[(transformed == src) & transformed_mask])
            if len(dst_vals) != 1:
                ok = False
                break
            color_map[int(src)] = int(dst_vals[0])
        if not ok:
            continue
        if np.array_equal(_apply_color_map(transformed, color_map), after):
            candidates.append((name, color_map))
    return candidates


def _solve_rule(before: np.ndarray, after: np.ndarray, bg: int) -> tuple[str, dict[int, int]]:
    candidates = _valid_demo_candidates(before, after, bg)
    if len(candidates) != 1:
        raise ValueError("demo pair does not define a unique transform")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a transform-and-recolor rule from the demo pair, then apply it to the target sprite.

    Input structure:
    - Two full-length separator lines split the grid into three sections in a row or a column.
    - Section 1 contains a multicolor demo-before sprite on a plain background.
    - Section 2 contains the corresponding demo-after sprite; section 3 contains a different target sprite.

    Rule:
    - Crop each section to its non-background bounding box.
    - Compare the two demo sprites to recover the unique rigid transform
      (a 90/180/270 rotation or one mirror reflection) and the per-color replacement table.
    - Apply that same rigid transform to the target sprite, recolor it with the recovered table,
      and return the transformed target cropped to its minimal bounding box.

    Key insight:
    - The concrete transform and recoloring are not fixed across examples; they must be learned
      from the visible demo pair inside the current input before being applied to the target.
    """
    g = np.asarray(grid, dtype=int)
    demo_before, demo_after, target, bg = _extract_sprites(g)
    transform_name, color_map = _solve_rule(demo_before, demo_after, bg)
    transformed = _apply_transform(target, transform_name)
    recolored = _apply_color_map(transformed, color_map)
    return _tight_crop(recolored, bg)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(r, c)]
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _random_connected_mask() -> np.ndarray | None:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)

    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    random.shuffle(anchors)
    coords = {anchors[0]}
    for anchor in anchors[1:]:
        start = random.choice(list(coords))
        coords.update(_random_path(start, anchor))

    min_area = max(len(coords) + 1, 5, (h * w) // 3)
    max_area = h * w - 1
    if min_area > max_area:
        return None
    target_area = random.randint(min_area, max_area)

    frontier = set()
    for r, c in coords:
        frontier.update(_neighbors4(r, c, h, w))
    frontier -= coords
    while len(coords) < target_area and frontier:
        cell = random.choice(list(frontier))
        coords.add(cell)
        frontier.discard(cell)
        frontier.update(_neighbors4(cell[0], cell[1], h, w))
        frontier -= coords

    if len(coords) != target_area:
        return None

    for r, c in coords:
        mask[r, c] = True
    if bool(mask.all()):
        return None
    return mask


def _shape_key(mask: np.ndarray) -> bytes:
    parts = []
    for name in ALL_TRANSFORMS:
        transformed = _apply_transform(mask.astype(int), name).astype(np.uint8)
        parts.append(transformed.tobytes() + str(transformed.shape).encode())
    return min(parts)


def _all_non_identity_shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [_apply_transform(mask.astype(int), name).astype(bool) for name in TRANSFORMS]


def _is_transform_unique_shape(mask: np.ndarray) -> bool:
    base = mask.astype(bool)
    for variant in _all_non_identity_shape_variants(base):
        if variant.shape == base.shape and np.array_equal(variant, base):
            return False
    return True


def _paint_mask(mask: np.ndarray, colors: list[int]) -> np.ndarray | None:
    coords = [tuple(x) for x in np.argwhere(mask)]
    if len(coords) < len(colors):
        return None
    random.shuffle(coords)
    sprite = np.zeros(mask.shape, dtype=int)
    for color, coord in zip(colors, coords):
        sprite[coord] = int(color)
    for coord in coords[len(colors):]:
        sprite[coord] = int(random.choice(colors))
    return sprite


def _random_derangement(colors: list[int]) -> dict[int, int]:
    perm = colors[:]
    while True:
        random.shuffle(perm)
        if all(a != b for a, b in zip(colors, perm)):
            return {int(a): int(b) for a, b in zip(colors, perm)}


def _embed_sprite(sprite: np.ndarray, h: int, w: int, bg: int) -> np.ndarray:
    canvas = np.full((h, w), bg, dtype=int)
    top_margin = h - sprite.shape[0]
    left_margin = w - sprite.shape[1]
    r0 = random.randint(1, top_margin - 1) if top_margin >= 2 else 0
    c0 = random.randint(1, left_margin - 1) if left_margin >= 2 else 0
    _paste(canvas, sprite, r0, c0)
    return canvas


def _compose_input(demo_before: np.ndarray, demo_after: np.ndarray, target: np.ndarray, sep_color: int) -> np.ndarray | None:
    sprites = [demo_before, demo_after, target]
    orientation = random.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        full_w = max(sprite.shape[1] for sprite in sprites) + random.randint(2, 4)
        bands = []
        total_h = 2
        for sprite in sprites:
            band_h = sprite.shape[0] + random.randint(2, 4)
            total_h += band_h
            bands.append(_embed_sprite(sprite, band_h, full_w, BG))
        if total_h > 30 or full_w > 30:
            return None
        sep = np.full((1, full_w), sep_color, dtype=int)
        return np.vstack([bands[0], sep, bands[1], sep, bands[2]])

    full_h = max(sprite.shape[0] for sprite in sprites) + random.randint(2, 4)
    strips = []
    total_w = 2
    for sprite in sprites:
        strip_w = sprite.shape[1] + random.randint(2, 4)
        total_w += strip_w
        strips.append(_embed_sprite(sprite, full_h, strip_w, BG))
    if total_w > 30 or full_h > 30:
        return None
    sep = np.full((full_h, 1), sep_color, dtype=int)
    return np.hstack([strips[0], sep, strips[1], sep, strips[2]])


def _semantic_checks(
    demo_before: np.ndarray,
    demo_after: np.ndarray,
    target: np.ndarray,
    expected: np.ndarray,
    transform_name: str,
    color_map: dict[int, int],
) -> bool:
    if _shape_key(demo_before != BG) == _shape_key(target != BG):
        return False

    candidates = _valid_demo_candidates(demo_before, demo_after, BG)
    if len(candidates) != 1:
        return False
    if candidates[0][0] != transform_name or candidates[0][1] != color_map:
        return False

    transform_only = _tight_crop(_apply_transform(target, transform_name), BG)
    recolor_only = _tight_crop(_apply_color_map(target, color_map), BG)
    if np.array_equal(transform_only, expected):
        return False
    if np.array_equal(recolor_only, expected):
        return False

    for alt_name in ALL_TRANSFORMS:
        if alt_name == transform_name:
            continue
        alt = _tight_crop(_apply_color_map(_apply_transform(target, alt_name), color_map), BG)
        if np.array_equal(alt, expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(2, 4)
    chosen_colors = random.sample(PALETTE, color_count + 1)
    sep_color = chosen_colors[-1]
    source_colors = chosen_colors[:-1]
    color_map = _random_derangement(source_colors)
    transform_name = random.choice(list(TRANSFORMS))

    demo_mask = None
    target_mask = None
    for _ in range(200):
        candidate = _random_connected_mask()
        if candidate is None or not _is_transform_unique_shape(candidate):
            continue
        demo_mask = candidate
        break
    if demo_mask is None:
        return None

    demo_before = _paint_mask(demo_mask, source_colors)
    if demo_before is None:
        return None
    demo_after = _apply_color_map(_apply_transform(demo_before, transform_name), color_map)

    for _ in range(200):
        candidate = _random_connected_mask()
        if candidate is None:
            continue
        if _shape_key(candidate) == _shape_key(demo_mask):
            continue
        expected = _tight_crop(_apply_color_map(_apply_transform(_paint_mask(candidate, source_colors), transform_name), color_map), BG)
        if expected.size == 0:
            continue
        target_mask = candidate
        break
    if target_mask is None:
        return None

    target = _paint_mask(target_mask, source_colors)
    if target is None:
        return None
    expected = _tight_crop(_apply_color_map(_apply_transform(target, transform_name), color_map), BG)

    if not _semantic_checks(demo_before, demo_after, target, expected, transform_name, color_map):
        return None

    grid = _compose_input(demo_before, demo_after, target, sep_color)
    if grid is None:
        return None
    if max(grid.shape) > 30:
        return None
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
