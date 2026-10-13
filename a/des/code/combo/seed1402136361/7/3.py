from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


RED = 2
GREEN = 3
GREY = 5
BG_CHOICES = (0, 1, 4, 6, 7, 8, 9)

OBJECT_SIZES = (2, 3, 4, 5, 6)
OBJECT_COUNT_RANGE = (2, 5)
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 24
PLACEMENT_PAD = 1
GENERATION_RETRIES = 400

FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(np.asarray(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_local_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_90_symmetric(mask: np.ndarray) -> bool:
    return bool(mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.rot90(mask, -1)))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor 90-degree-rotation-symmetric objects, box-fill the rest."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    out = np.full_like(g, bg)
    components = _find_components(g != bg)

    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        mask = _coords_to_local_mask(coords)
        if _is_90_symmetric(mask):
            out[coords[:, 0], coords[:, 1]] = GREEN
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = RED

    return out


def _rot90_cell(r: int, c: int, size: int) -> tuple[int, int]:
    return c, size - 1 - r


def _rotation_orbits(size: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []

    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit: set[tuple[int, int]] = set()
            rr, cc = r, c
            for _ in range(4):
                orbit.add((rr, cc))
                rr, cc = _rot90_cell(rr, cc, size)
            ordered = tuple(sorted(orbit))
            seen.update(ordered)
            orbits.append(ordered)

    return orbits


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_find_components(mask)) == 1


def _is_valid_object_mask(mask: np.ndarray) -> bool:
    return bool(
        mask.ndim == 2
        and mask.shape[0] == mask.shape[1]
        and mask.any()
        and _touches_all_sides(mask)
        and _is_connected(mask)
    )


def _enumerate_symmetric_masks(size: int) -> list[np.ndarray]:
    orbits = _rotation_orbits(size)
    masks: list[np.ndarray] = []

    for subset in range(1, 1 << len(orbits)):
        mask = np.zeros((size, size), dtype=bool)
        for bit, orbit in enumerate(orbits):
            if not (subset & (1 << bit)):
                continue
            for r, c in orbit:
                mask[r, c] = True
        if _is_valid_object_mask(mask):
            masks.append(mask.copy())

    return masks


def _enumerate_asymmetric_masks(sym_masks: list[np.ndarray]) -> list[np.ndarray]:
    seen: set[bytes] = set()
    masks: list[np.ndarray] = []

    for base in sym_masks:
        size = base.shape[0]
        for r in range(size):
            for c in range(size):
                mask = base.copy()
                mask[r, c] = not bool(mask[r, c])
                if not _is_valid_object_mask(mask):
                    continue
                if _is_90_symmetric(mask):
                    continue
                key = mask.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                masks.append(mask.copy())

    return masks


def _precompute_mask_pools() -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    symmetric: dict[int, list[np.ndarray]] = {}
    asymmetric: dict[int, list[np.ndarray]] = {}

    for size in OBJECT_SIZES:
        sym_masks = _enumerate_symmetric_masks(size)
        asym_masks = _enumerate_asymmetric_masks(sym_masks)
        assert sym_masks and asym_masks
        symmetric[size] = sym_masks
        asymmetric[size] = asym_masks

    return symmetric, asymmetric


SYMMETRIC_MASKS, ASYMMETRIC_MASKS = _precompute_mask_pools()


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _balanced_object_types(count: int) -> list[bool]:
    sym_count = count // 2 + random.randint(0, count % 2)
    kinds = [True] * sym_count + [False] * (count - sym_count)
    random.shuffle(kinds)
    return kinds


def _sample_object_mask(is_symmetric: bool) -> np.ndarray:
    size = random.choice(OBJECT_SIZES)
    pool = SYMMETRIC_MASKS[size] if is_symmetric else ASYMMETRIC_MASKS[size]
    return _random_dihedral(random.choice(pool))


def _sample_objects() -> list[np.ndarray]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    kinds = _balanced_object_types(count)
    objects: list[np.ndarray] = []
    used: set[tuple[bool, bytes]] = set()

    for is_symmetric in kinds:
        for _ in range(40):
            mask = _sample_object_mask(is_symmetric)
            key = (is_symmetric, mask.tobytes())
            if key not in used or len(used) > 10:
                used.add(key)
                objects.append(mask)
                break
        else:
            objects.append(_sample_object_mask(is_symmetric))

    return objects


def _grid_bounds(objects: list[np.ndarray]) -> tuple[int, int]:
    sizes = [mask.shape[0] for mask in objects]
    padded_area = sum((size + PLACEMENT_PAD) ** 2 for size in sizes)
    min_side = max(max(sizes) + 2, int(np.ceil(np.sqrt(padded_area))) + 2, GRID_SIDE_MIN)
    max_side = min(GRID_SIDE_MAX, min_side + 8)
    return min_side, max_side


def _place_objects(height: int, width: int, objects: list[np.ndarray]) -> list[tuple[int, int, np.ndarray]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int, np.ndarray]] = []

    for mask in sorted(objects, key=lambda arr: arr.shape[0], reverse=True):
        size = mask.shape[0]
        positions = [(r, c) for r in range(height - size + 1) for c in range(width - size + 1)]
        random.shuffle(positions)

        for r, c in positions:
            r0 = max(0, r - PLACEMENT_PAD)
            c0 = max(0, c - PLACEMENT_PAD)
            r1 = min(height, r + size + PLACEMENT_PAD)
            c1 = min(width, c + size + PLACEMENT_PAD)
            if bool(occupied[r0:r1, c0:c1].any()):
                continue
            occupied[r0:r1, c0:c1] = True
            placements.append((r, c, mask))
            break
        else:
            return None

    return placements


def _render_input(height: int, width: int, placements: list[tuple[int, int, np.ndarray]], bg: int) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for r, c, mask in placements:
        size = mask.shape[0]
        patch = grid[r : r + size, c : c + size]
        patch[mask] = GREY
    return grid


def _validate_generated_grid(grid: np.ndarray) -> None:
    bg = _background_color(grid)
    assert bg != GREY

    colors = Counter(map(int, grid.flat))
    assert len(colors) == 2
    assert GREY in colors

    components = _find_components(grid == GREY)
    assert len(components) >= 2

    has_symmetric = False
    has_asymmetric = False
    for coords in components:
        mask = _coords_to_local_mask(coords)
        assert mask.shape[0] == mask.shape[1]
        if _is_90_symmetric(mask):
            has_symmetric = True
        else:
            has_asymmetric = True

    assert has_symmetric and has_asymmetric

    out = main(grid)
    out_colors = set(map(int, np.unique(out)))
    assert out_colors <= {bg, RED, GREEN}
    assert RED in out_colors and GREEN in out_colors


def _gen_attempt() -> np.ndarray | None:
    objects = _sample_objects()
    min_side, max_side = _grid_bounds(objects)
    if min_side > GRID_SIDE_MAX:
        return None

    height = random.randint(min_side, max_side)
    width = random.randint(min_side, max_side)
    placements = _place_objects(height, width, objects)
    if placements is None:
        return None

    bg = random.choice(BG_CHOICES)
    grid = _render_input(height, width, placements, bg)

    try:
        _validate_generated_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
