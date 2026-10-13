from __future__ import annotations

import itertools

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SPRITE_SIZE = 3
STACK_GAP = 1
COLUMN_GAP = 2
BACKGROUND = 0
PALETTE = tuple(color for color in range(10) if color != BACKGROUND)
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _tight_bbox(mask)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_mask(mask)
    return tuple(map(int, tight.shape)), tight.astype(np.uint8).tobytes()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _has_full_bbox(mask: np.ndarray) -> bool:
    return bool(np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)))


def _rotation_index(mask: np.ndarray) -> int:
    variants = [np.rot90(mask, k) for k in range(4)]
    keys = [_mask_key(variant) for variant in variants]
    best = min(keys)
    indices = [i for i, key in enumerate(keys) if key == best]
    assert len(indices) == 1, "sprite orientation must be unambiguous"
    return int(indices[0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            coords = [(r, c)]
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
                    coords.append((nr, nc))

            coord_arr = np.asarray(coords, dtype=int)
            r0 = int(coord_arr[:, 0].min())
            c0 = int(coord_arr[:, 1].min())
            r1 = int(coord_arr[:, 0].max())
            c1 = int(coord_arr[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coord_arr[:, 0] - r0, coord_arr[:, 1] - c0] = True

            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return components


def _build_output_from_components(components: list[dict[str, object]], bg: int) -> np.ndarray:
    ordered = sorted(components, key=lambda comp: tuple(comp["bbox"][:2]))
    groups = {orientation: [] for orientation in range(4)}

    for comp in ordered:
        mask = np.asarray(comp["mask"], dtype=bool)
        assert mask.shape == (SPRITE_SIZE, SPRITE_SIZE)
        assert _has_full_bbox(mask)
        assert _is_connected(mask)
        groups[_rotation_index(mask)].append(comp)

    present = [orientation for orientation in range(4) if groups[orientation]]
    assert present

    out_h = max(len(groups[orientation]) * SPRITE_SIZE + (len(groups[orientation]) - 1) * STACK_GAP for orientation in present)
    out_w = len(present) * SPRITE_SIZE + (len(present) - 1) * COLUMN_GAP
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for col_index, orientation in enumerate(present):
        x0 = col_index * (SPRITE_SIZE + COLUMN_GAP)
        for row_index, comp in enumerate(groups[orientation]):
            y0 = row_index * (SPRITE_SIZE + STACK_GAP)
            mask = np.asarray(comp["mask"], dtype=bool)
            color = int(comp["color"])
            patch = out[y0 : y0 + SPRITE_SIZE, x0 : x0 + SPRITE_SIZE]
            assert not bool((patch != bg)[mask].any())
            patch[mask] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Group 3x3 sprites by rotational orientation and stack each orientation in its own column.

    Input structure:
    - The background is black/dominant and the foreground consists of separate monochrome sprites.
    - Every sprite is a single 4-connected component whose tight bounding box is 3x3.
    - Each 3x3 sprite is rotationally asymmetric, so its orientation is uniquely determined.

    Rule:
    - Extract every sprite as a binary 3x3 mask and compare its four quarter-turn rotations.
    - The unique lexicographically smallest rotation is the sprite family's reference "up" form; the number of turns needed to reach that reference gives the orientation class in the fixed order up, right, down, left.
    - Group sprites by that orientation, preserve source reading order within each group, and place each group in one 3-cell-wide column with one blank row between stacked sprites and two blank columns between neighboring groups.
    - Skip any orientation whose group is empty, and keep each sprite's original shape and color.

    Key insight:
    - Color and position are distractions; the only stable grouping key is the rotation class of each sprite's normalized shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components
    return _build_output_from_components(components, bg)


def _enumerate_sprite_families() -> tuple[np.ndarray, ...]:
    families: list[np.ndarray] = []
    for bits in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        for index in range(SPRITE_SIZE * SPRITE_SIZE):
            if bits & (1 << index):
                mask[index // SPRITE_SIZE, index % SPRITE_SIZE] = True

        if not _has_full_bbox(mask):
            continue
        if not _is_connected(mask):
            continue
        if not (5 <= int(mask.sum()) <= 7):
            continue

        variants = [np.rot90(mask, k) for k in range(4)]
        keys = [_mask_key(variant) for variant in variants]
        if len(set(keys)) != 4:
            continue
        if keys[0] != min(keys):
            continue
        families.append(mask)

    assert families
    return tuple(families)


SPRITE_FAMILIES = _enumerate_sprite_families()
_RNG = np.random.default_rng()


def _oriented_mask(family: np.ndarray, orientation: int) -> np.ndarray:
    return np.rot90(family, -int(orientation))


def _random_counts(total: int, groups: int) -> list[int]:
    counts = [1] * groups
    for _ in range(total - groups):
        counts[int(_RNG.integers(groups))] += 1
    return counts


def _place_sprites(h: int, w: int, sprites: list[dict[str, object]]) -> np.ndarray | None:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    positions = list(itertools.product(range(h - SPRITE_SIZE + 1), range(w - SPRITE_SIZE + 1)))

    for sprite in sprites:
        found = False
        for index in _RNG.permutation(len(positions)):
            r0, c0 = positions[int(index)]
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + SPRITE_SIZE + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + SPRITE_SIZE + 1)
            if bool(occupied[rr0:rr1, cc0:cc1].any()):
                continue

            mask = np.asarray(sprite["mask"], dtype=bool)
            color = int(sprite["color"])
            patch = grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE]
            patch[mask] = color
            occupied[rr0:rr1, cc0:cc1] = True
            sprite["bbox"] = (int(r0), int(c0), int(r0 + SPRITE_SIZE - 1), int(c0 + SPRITE_SIZE - 1))
            found = True
            break

        if not found:
            return None

    return grid


def _expected_output(sprites: list[dict[str, object]]) -> np.ndarray:
    components = [
        {
            "color": int(sprite["color"]),
            "bbox": tuple(sprite["bbox"]),
            "mask": np.asarray(sprite["mask"], dtype=bool),
        }
        for sprite in sprites
    ]
    return _build_output_from_components(components, BACKGROUND)


def _gen_attempt() -> np.ndarray | None:
    group_count = int(_RNG.integers(2, 5))
    present_orientations = list(map(int, _RNG.choice(4, size=group_count, replace=False)))
    total_sprites = int(_RNG.integers(max(4, group_count), 8))
    counts = _random_counts(total_sprites, group_count)

    orientations: list[int] = []
    for orientation, count in zip(present_orientations, counts):
        orientations.extend([int(orientation)] * int(count))
    _RNG.shuffle(orientations)

    family_indices = list(map(int, _RNG.choice(len(SPRITE_FAMILIES), size=total_sprites, replace=False)))
    colors = list(map(int, _RNG.choice(PALETTE, size=total_sprites, replace=False)))
    sprites = [
        {
            "color": colors[i],
            "orientation": orientations[i],
            "mask": _oriented_mask(SPRITE_FAMILIES[family_indices[i]], orientations[i]),
        }
        for i in range(total_sprites)
    ]
    _RNG.shuffle(sprites)

    min_side = max(12, 4 * int(np.ceil(np.sqrt(total_sprites))) + 3)
    h = int(_RNG.integers(min_side, 27))
    w = int(_RNG.integers(min_side, 27))
    grid = _place_sprites(h, w, sprites)
    if grid is None:
        return None

    components = _find_components(grid, BACKGROUND)
    if len(components) != total_sprites:
        return None
    if not all(np.asarray(comp["mask"]).shape == (SPRITE_SIZE, SPRITE_SIZE) for comp in components):
        return None

    expected = _expected_output(sprites)
    if max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sprite-orientation sample")
