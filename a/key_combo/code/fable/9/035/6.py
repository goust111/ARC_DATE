from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty object"
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == bg:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch,
                    "mask": patch == color,
                }
            )

    return components


def _variant_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _rotation_class_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_variant_signature(np.rot90(mask, k)) for k in range(4))


def _d4_orbit_size(mask: np.ndarray) -> int:
    variants = []
    mirrored = np.fliplr(mask)
    for k in range(4):
        variants.append(_variant_signature(np.rot90(mask, k)))
        variants.append(_variant_signature(np.rot90(mirrored, k)))
    return len(set(variants))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the uniquely mirrored sprite from a 3x3 gallery of rotated copies.

    Input structure:
    - The grid contains nine separated monochrome sprites arranged in a 3x3 layout
      with background gaps between neighboring blocks.
    - All nine sprites share one base shape with no rotational or mirror symmetry,
      but each copy has its own color and orientation.
    - Eight copies are pure rotations of the same shape; the remaining copy is a
      reflected version, possibly rotated afterward.

    Rule:
    - Treat each sprite by its occupied shape only, ignoring its color and position.
    - Group sprites by rotation equivalence: two sprites match iff one can be rotated
      onto the other without any reflection.
    - Find the single sprite whose shape is not rotation-equivalent to the other eight,
      then crop its tight bounding box and return it with its original color.

    Key insight:
    - Orientation and color are decoys; the only stable discriminator is chirality,
      so the answer is the unique mirror-class outlier under rotation-only matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(np.bincount(g.ravel()).argmax())
    components = _find_components(g, bg)
    assert len(components) == 9, "expected exactly nine separated sprites"

    signatures = [_rotation_class_signature(component["mask"]) for component in components]
    counts = Counter(signatures)
    assert sorted(counts.values()) == [1, 8], "expected one mirror outlier against eight rotational matches"

    outlier_signature = next(sig for sig, count in counts.items() if count == 1)
    outlier = components[signatures.index(outlier_signature)]
    return _crop_bbox(outlier["patch"], outlier["mask"])


_BG = 0
_PALETTE = list(range(1, 10))
_GEN_ATTEMPTS = 800
_SPRITE_SIZE_RANGE = range(3, 7)
_MIN_DENSITY = 0.42
_MAX_DENSITY = 0.72
_CELL_EXTRA = 3


def _grow_connected_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    low = max(height + width - 1, int(np.ceil(area * _MIN_DENSITY)))
    high = min(area - 1, int(np.floor(area * _MAX_DENSITY)))
    if low > high:
        raise ValueError("invalid sprite size range")

    target = random.randint(low, high)
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        chosen = random.choice(tuple(frontier))
        mask[chosen] = True

    return mask


def _generate_base_mask() -> np.ndarray | None:
    height = random.choice(tuple(_SPRITE_SIZE_RANGE))
    width = random.choice(tuple(_SPRITE_SIZE_RANGE))
    mask = _grow_connected_mask(height, width)

    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return None
    if int(mask.sum()) < 5:
        return None
    if _d4_orbit_size(mask) != 8:
        return None
    return mask


def _render_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, color, _BG).astype(int)


def _semantic_signature_counts(grid: np.ndarray) -> list[int]:
    components = _find_components(grid, _BG)
    signatures = [_rotation_class_signature(component["mask"]) for component in components]
    return sorted(Counter(signatures).values())


def _build_scene(base_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cell_size = max(base_mask.shape) + _CELL_EXTRA
    grid = np.zeros((3 * cell_size + 2, 3 * cell_size + 2), dtype=int)

    colors = _PALETTE[:]
    random.shuffle(colors)
    outlier_index = random.randrange(9)
    mirrored_base = np.fliplr(base_mask)

    normal_rotations = [0, 1, 2, 3] + [random.randrange(4) for _ in range(4)]
    random.shuffle(normal_rotations)
    outlier_rotation = random.randrange(4)

    expected: np.ndarray | None = None
    normal_cursor = 0

    for index in range(9):
        block_r, block_c = divmod(index, 3)
        top = block_r * (cell_size + 1)
        left = block_c * (cell_size + 1)

        if index == outlier_index:
            shape = np.rot90(mirrored_base, outlier_rotation)
        else:
            shape = np.rot90(base_mask, normal_rotations[normal_cursor])
            normal_cursor += 1

        sprite = _render_sprite(shape, colors[index])
        sh, sw = sprite.shape
        max_r = cell_size - sh
        max_c = cell_size - sw

        if index == outlier_index:
            local_r = random.randint(1, max_r - 1)
            local_c = random.randint(1, max_c - 1)
        else:
            local_r = random.randint(0, max_r)
            local_c = random.randint(0, max_c)

        grid[top + local_r : top + local_r + sh, left + local_c : left + local_c + sw] = sprite
        if index == outlier_index:
            expected = sprite.copy()

    assert expected is not None
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    base_mask = _generate_base_mask()
    if base_mask is None:
        return None

    grid, expected = _build_scene(base_mask)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if _semantic_signature_counts(grid) != [1, 8]:
        return None
    if expected.shape[0] >= max(base_mask.shape) + _CELL_EXTRA and expected.shape[1] >= max(base_mask.shape) + _CELL_EXTRA:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chirality-outlier scene")
