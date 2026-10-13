from __future__ import annotations

import hashlib
import random
from collections import Counter

import numpy as np


FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 600
UNIQUE_KIND_RANGE = (3, 5)
GRID_SIDE_RANGE = (18, 24)
SPRITE_SIDE_RANGE = (2, 5)
MAX_FG_CELLS = 8
PALETTE = tuple(range(1, 10))
_SEEN_OUTPUT_SIGS: set[str] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(H):
        for c0 in range(W):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox_from_coords(arr), "area": int(arr.shape[0])})
    return components


def _sprite_from_component(grid: np.ndarray, component: dict[str, object], bg: int) -> np.ndarray:
    coords = np.asarray(component["coords"], dtype=int)
    bbox = component["bbox"]
    r0, c0, _r1, _c1 = bbox
    sprite = np.full(_crop_bbox(grid, bbox).shape, bg, dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return sprite


def _sprite_key(sprite: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, sprite.shape)), tuple(int(v) for v in sprite.ravel())


def _mask_key(sprite: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    mask = (np.asarray(sprite) != 0).astype(int)
    return tuple(map(int, mask.shape)), tuple(int(v) for v in mask.ravel())


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    return np.repeat(np.repeat(np.asarray(sprite, dtype=int), k, axis=0), k, axis=1)


def _analyze_grid(grid: np.ndarray) -> tuple[dict, dict, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    sprites: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    counts: Counter = Counter()
    mask_counts: Counter = Counter()
    for component in components:
        sprite = _sprite_from_component(g, component, bg)
        key = _sprite_key(sprite)
        sprites[key] = sprite
        counts[key] += 1
        mask_counts[_mask_key(sprite)] += 1
    return sprites, counts, mask_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the only exactly repeated colored sprite crop, then scale it by the number of unique crop patterns.

    Input structure:
    - Black is the background.
    - Several separated 4-connected multicolor sprites are scattered on the grid.
    - Exactly one full colored bounding-box crop appears twice; other crop patterns appear once.

    Rule:
    - Crop every sprite to its tight bounding box, preserving black cells inside the box.
    - Compare those full colored crops and find the unique pattern that occurs exactly twice.
    - Count the number of distinct crop patterns N, then replace every pixel of the repeated crop, including black pixels, with an N x N block of the same color.

    Key insight:
    - One relation chooses which sprite to keep, while the overall variety of distinct sprite crops encodes how much that chosen crop must be enlarged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    sprites, counts, _mask_counts = _analyze_grid(g)
    duplicate_keys = [key for key, count in counts.items() if int(count) == 2]
    assert len(duplicate_keys) == 1, "expected exactly one repeated colored crop"
    factor = len(counts)
    sprite = sprites[duplicate_keys[0]]
    assert max(sprite.shape) * factor <= 30, "scaled output must stay within ARC limits"
    return _scale_sprite(sprite, factor)


def _random_connected_mask(shape: tuple[int, int], area: int) -> np.ndarray | None:
    h, w = shape
    if area < max(h, w) or area >= h * w:
        return None

    for _ in range(150):
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    return None


def _paint_mask(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    for idx, color in enumerate(colors):
        r, c = coords[idx]
        sprite[r, c] = int(color)
    for r, c in coords[len(colors) :]:
        sprite[r, c] = int(random.choice(colors))
    return sprite


def _random_sprite(
    *,
    fixed_shape: tuple[int, int] | None = None,
    fixed_area: int | None = None,
    forbidden: set[tuple[tuple[int, int], tuple[int, ...]]] | None = None,
) -> np.ndarray | None:
    forbidden = forbidden or set()
    for _ in range(200):
        if fixed_shape is None:
            h = random.randint(*SPRITE_SIDE_RANGE)
            w = random.randint(*SPRITE_SIDE_RANGE)
        else:
            h, w = fixed_shape

        if max(h, w) > 5:
            return None

        min_area = max(h, w)
        max_area = min(MAX_FG_CELLS, h * w - 1)
        if fixed_area is not None:
            if not (min_area <= fixed_area <= max_area):
                continue
            area = fixed_area
        else:
            if min_area > max_area:
                continue
            area = random.randint(min_area, max_area)

        mask = _random_connected_mask((h, w), area)
        if mask is None:
            continue

        n_colors = random.randint(2, min(3, area))
        colors = random.sample(PALETTE, n_colors)
        sprite = _paint_mask(mask, colors)
        key = _sprite_key(sprite)
        if key in forbidden:
            continue
        return sprite
    return None


def _recolor_same_mask(sprite: np.ndarray, mode: str) -> np.ndarray | None:
    mask = sprite != 0
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    base_values = [int(sprite[r, c]) for r, c in coords]
    base_unique = sorted(set(base_values))
    for _ in range(150):
        out = np.zeros_like(sprite)
        if mode == "permute_same_colors":
            permuted = base_values[:]
            random.shuffle(permuted)
            if permuted == base_values:
                continue
            for (r, c), value in zip(coords, permuted):
                out[r, c] = int(value)
        else:
            n_colors = random.randint(2, min(3, len(coords)))
            if set(base_unique) == set(PALETTE) and n_colors == len(base_unique):
                continue
            colors = random.sample(PALETTE, n_colors)
            if set(colors) == set(base_unique):
                continue
            out = _paint_mask(mask, colors)

        if _sprite_key(out) != _sprite_key(sprite):
            return out
    return None


def _sprite_area(sprite: np.ndarray) -> int:
    return int(np.count_nonzero(sprite))


def _build_sprite_set(n_unique: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    duplicate = _random_sprite()
    if duplicate is None:
        return None

    seen = {_sprite_key(duplicate)}
    distractors: list[np.ndarray] = []

    recolor_mode = random.choice(("permute_same_colors", "new_colors"))
    same_mask = _recolor_same_mask(duplicate, recolor_mode)
    if same_mask is None or _sprite_key(same_mask) in seen:
        return None
    distractors.append(same_mask)
    seen.add(_sprite_key(same_mask))

    while len(distractors) < n_unique - 1:
        mode = random.choice(("same_area", "same_shape", "free"))
        kwargs = {"forbidden": seen}
        if mode == "same_area":
            kwargs["fixed_area"] = _sprite_area(duplicate)
        elif mode == "same_shape":
            kwargs["fixed_shape"] = tuple(map(int, duplicate.shape))
        sprite = _random_sprite(**kwargs)
        if sprite is None:
            return None
        key = _sprite_key(sprite)
        if key in seen:
            continue
        distractors.append(sprite)
        seen.add(key)

    return duplicate, distractors


def _can_place_bbox(blocked: np.ndarray, shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = shape
    return not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())


def _mark_blocked(blocked: np.ndarray, shape: tuple[int, int], r0: int, c0: int) -> None:
    h, w = shape
    H, W = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_sprites(grid_shape: tuple[int, int], sprites: list[np.ndarray]) -> np.ndarray | None:
    H, W = grid_shape
    grid = np.zeros((H, W), dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    order = list(range(len(sprites)))
    random.shuffle(order)
    order.sort(key=lambda idx: sprites[idx].shape[0] * sprites[idx].shape[1], reverse=True)

    placed: dict[int, tuple[int, int]] = {}
    for idx in order:
        sprite = sprites[idx]
        h, w = sprite.shape
        candidates = [
            (r0, c0)
            for r0 in range(H - h + 1)
            for c0 in range(W - w + 1)
            if _can_place_bbox(blocked, sprite.shape, r0, c0)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        patch_mask = sprite != 0
        grid[r0 : r0 + h, c0 : c0 + w][patch_mask] = sprite[patch_mask]
        _mark_blocked(blocked, sprite.shape, r0, c0)
        placed[idx] = (r0, c0)

    assert len(placed) == len(sprites)
    return grid


def _semantic_checks(grid: np.ndarray, duplicate: np.ndarray, n_unique: int) -> bool:
    sprites, counts, mask_counts = _analyze_grid(grid)
    duplicate_key = _sprite_key(duplicate)

    if len(counts) != n_unique:
        return False
    if counts.get(duplicate_key, 0) != 2:
        return False
    if sum(1 for value in counts.values() if int(value) == 2) != 1:
        return False
    if mask_counts[_mask_key(duplicate)] < 3:
        return False

    other_keys = [key for key in counts if key != duplicate_key]
    shared_surface_cue = False
    duplicate_area = _sprite_area(duplicate)
    duplicate_shape = tuple(map(int, duplicate.shape))
    duplicate_colors = frozenset(int(v) for v in duplicate.ravel() if int(v) != 0)
    for key in other_keys:
        sprite = sprites[key]
        if (
            tuple(map(int, sprite.shape)) == duplicate_shape
            or _sprite_area(sprite) == duplicate_area
            or frozenset(int(v) for v in sprite.ravel() if int(v) != 0) == duplicate_colors
            or _mask_key(sprite) == _mask_key(duplicate)
        ):
            shared_surface_cue = True
            break
    if not shared_surface_cue:
        return False

    expected = _scale_sprite(duplicate, n_unique)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if max(actual.shape) > 30:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n_unique = random.randint(*UNIQUE_KIND_RANGE)
    built = _build_sprite_set(n_unique)
    if built is None:
        return None
    duplicate, distractors = built
    if max(duplicate.shape) * n_unique > 30:
        return None

    sprites = [duplicate.copy(), duplicate.copy(), *[sprite.copy() for sprite in distractors]]
    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    grid = _place_sprites((H, W), sprites)
    if grid is None:
        return None

    if not _semantic_checks(grid, duplicate, n_unique):
        return None

    output = main(grid)
    output_sig = hashlib.md5(output.tobytes() + str(output.shape).encode()).hexdigest()
    if output_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
