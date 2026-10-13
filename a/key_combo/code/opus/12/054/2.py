from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
SPRITE_SIDE = 5
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 400

QUADRANT_ORDER = ("tl", "tr", "bl", "br")
QUADRANT_SLICES = {
    "tl": (slice(0, 3), slice(0, 3)),
    "tr": (slice(0, 3), slice(3, 5)),
    "bl": (slice(3, 5), slice(0, 3)),
    "br": (slice(3, 5), slice(3, 5)),
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _connected_components(mask: np.ndarray, *, connectivity: int) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray, *, connectivity: int) -> bool:
    if not np.any(mask):
        return False
    return len(_connected_components(mask, connectivity=connectivity)) == 1


def _find_sprites(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    comps = _connected_components(grid != BG, connectivity=8)
    sprites: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        box = (int(r0), int(c0), int(r1), int(c1))
        r0, c0, r1, c1 = box
        assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIDE, SPRITE_SIDE), "each sprite must occupy a 5x5 bbox"
        sprites.append((box, grid[r0 : r1 + 1, c0 : c1 + 1].copy()))
    sprites.sort(key=lambda item: (item[0][0], item[0][1]))
    return sprites


def _rotation_orbits(side: int = SPRITE_SIDE) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(side):
        for c in range(side):
            cells = []
            rr, cc = r, c
            for _ in range(4):
                cells.append((rr, cc))
                rr, cc = cc, side - 1 - rr
            orbit = tuple(sorted(set(cells)))
            if orbit in seen:
                continue
            seen.add(orbit)
            orbits.append(orbit)
    orbits.sort(key=lambda orb: (len(orb), orb))
    return orbits


ORBITS = _rotation_orbits()
CENTER_ORBIT = next(orbit for orbit in ORBITS if len(orbit) == 1)
FOUR_ORBITS = [orbit for orbit in ORBITS if len(orbit) == 4]


def _quadrant_mask(name: str) -> np.ndarray:
    mask = np.zeros((SPRITE_SIDE, SPRITE_SIDE), dtype=bool)
    rows, cols = QUADRANT_SLICES[name]
    mask[rows, cols] = True
    return mask


QUADRANT_MASKS = {name: _quadrant_mask(name) for name in QUADRANT_ORDER}
QUADRANT_COORDS = {
    name: [tuple(pos) for pos in np.argwhere(mask)]
    for name, mask in QUADRANT_MASKS.items()
}


def _template_is_valid(mask: np.ndarray) -> bool:
    if not np.array_equal(mask, np.rot90(mask)):
        return False
    box = _bbox(mask)
    if box != (0, 0, SPRITE_SIDE - 1, SPRITE_SIDE - 1):
        return False
    if not _is_connected(mask, connectivity=8):
        return False
    counts = [int(np.count_nonzero(mask & QUADRANT_MASKS[name])) for name in QUADRANT_ORDER]
    return min(counts) >= 2


def _random_template() -> np.ndarray | None:
    orbit_pool = FOUR_ORBITS.copy()
    random.shuffle(orbit_pool)
    orbit_count = random.randint(3, 5)
    chosen = orbit_pool[:orbit_count]
    if random.random() < 0.7:
        chosen = [CENTER_ORBIT, *chosen]

    mask = np.zeros((SPRITE_SIDE, SPRITE_SIDE), dtype=bool)
    for orbit in chosen:
        for r, c in orbit:
            mask[r, c] = True

    if not _template_is_valid(mask):
        return None
    sprite = np.zeros((SPRITE_SIDE, SPRITE_SIDE), dtype=int)
    sprite[mask] = BLUE
    return sprite


def _connected_subsets(coords: list[tuple[int, int]]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    n = len(coords)
    for bits in range(1, 1 << n):
        subset = np.zeros((SPRITE_SIDE, SPRITE_SIDE), dtype=bool)
        for i, (r, c) in enumerate(coords):
            if bits >> i & 1:
                subset[r, c] = True
        local = subset.copy()
        cells = np.argwhere(local)
        if cells.size == 0:
            continue
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        local = local[r0 : r1 + 1, c0 : c1 + 1]
        if _is_connected(local, connectivity=4):
            out.append(subset)
    return out


CONNECTED_DAMAGE_MASKS = {
    name: _connected_subsets(QUADRANT_COORDS[name]) for name in QUADRANT_ORDER
}


def _damage_candidates(template: np.ndarray, quadrant: str) -> list[np.ndarray]:
    qmask = QUADRANT_MASKS[quadrant]
    template_mask = template == BLUE
    target_binary = template_mask & qmask
    candidates: list[np.ndarray] = []
    for red_mask in CONNECTED_DAMAGE_MASKS[quadrant]:
        if np.array_equal(red_mask, target_binary):
            continue
        sprite = template.copy()
        sprite[qmask] = BG
        sprite[red_mask] = RED
        if not _is_connected(sprite != BG, connectivity=8):
            continue
        if _bbox(sprite != BG) != (0, 0, SPRITE_SIDE - 1, SPRITE_SIDE - 1):
            continue
        if np.array_equal(sprite, np.rot90(sprite)):
            continue
        mismatch = [name for name in QUADRANT_ORDER if not np.array_equal(sprite[QUADRANT_SLICES[name]], template[QUADRANT_SLICES[name]])]
        if mismatch != [quadrant]:
            continue
        if np.any((sprite == BLUE) & qmask):
            continue
        candidates.append(sprite)
    return candidates


def _repair_sprite(sprite: np.ndarray, template: np.ndarray) -> np.ndarray:
    mismatch = [name for name in QUADRANT_ORDER if not np.array_equal(sprite[QUADRANT_SLICES[name]], template[QUADRANT_SLICES[name]])]
    assert len(mismatch) == 1, "each damaged sprite must differ from the template in exactly one quadrant"
    quadrant = mismatch[0]
    rows, cols = QUADRANT_SLICES[quadrant]
    repaired = sprite.copy()
    repaired[rows, cols] = template[rows, cols]
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair every damaged 5x5 sprite by finding the unique quarter-turn-symmetric template.

    Input structure:
    - Black background contains 4-6 separated sprites, each occupying a 5x5 bounding box.
    - All intact template pixels are blue, and damaged copies contain red pixels only inside one quadrant-sized region.
    - Exactly one sprite is invariant under 90-degree rotation of its full 5x5 box; that sprite is the template.

    Rule:
    - Detect all 5x5 sprites and identify the unique one whose full 5x5 pattern matches its 90/180/270 degree rotations.
    - For every other sprite, compare each quadrant region against the template and locate the single mismatching quadrant.
    - Replace that mismatching quadrant with the template's blue cells from the same 5x5 reference positions, leaving the rest of the grid unchanged.

    Key insight:
    - The template is not given by position; it is the only sprite certified by quarter-turn rotational symmetry, and that template then tells which quadrant of each red-damaged copy must be overwritten.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = _find_sprites(g)
    assert len(sprites) >= 2, "expected multiple sprites"

    template_ids = [i for i, (_box, sprite) in enumerate(sprites) if np.array_equal(sprite, np.rot90(sprite))]
    assert len(template_ids) == 1, "expected exactly one rotationally symmetric template"
    template = sprites[template_ids[0]][1]

    out = g.copy()
    for i, (box, sprite) in enumerate(sprites):
        if i == template_ids[0]:
            continue
        repaired = _repair_sprite(sprite, template)
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = repaired

    return out


def _placeable(top_lefts: list[tuple[int, int]], r: int, c: int, gap: int) -> bool:
    for rr, cc in top_lefts:
        if not (r + SPRITE_SIDE + gap <= rr or rr + SPRITE_SIDE + gap <= r or c + SPRITE_SIDE + gap <= cc or cc + SPRITE_SIDE + gap <= c):
            return False
    return True


def _random_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - SPRITE_SIDE) for c in range(1, width - SPRITE_SIDE)]
    for _ in range(PLACEMENT_ATTEMPTS):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        gap = random.choice([1, 1, 2])
        for r, c in candidates:
            if _placeable(chosen, r, c, gap):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, template: np.ndarray, quadrants: list[str]) -> bool:
    try:
        sprites = _find_sprites(inp)
    except Exception:
        return False
    if len(sprites) != len(quadrants) + 1:
        return False

    symmetric = [i for i, (_box, sprite) in enumerate(sprites) if np.array_equal(sprite, np.rot90(sprite))]
    if len(symmetric) != 1:
        return False
    template_index = symmetric[0]
    if not np.array_equal(sprites[template_index][1], template):
        return False

    seen_quadrants: list[str] = []
    for i, (_box, sprite) in enumerate(sprites):
        if i == template_index:
            continue
        mismatch = [name for name in QUADRANT_ORDER if not np.array_equal(sprite[QUADRANT_SLICES[name]], template[QUADRANT_SLICES[name]])]
        if len(mismatch) != 1:
            return False
        quadrant = mismatch[0]
        rows, cols = QUADRANT_SLICES[quadrant]
        if not np.any(sprite[rows, cols] == RED):
            return False
        if np.any(sprite[rows, cols] == BLUE):
            return False
        for other in QUADRANT_ORDER:
            if other == quadrant:
                continue
            if not np.array_equal(sprite[QUADRANT_SLICES[other]], template[QUADRANT_SLICES[other]]):
                return False
        seen_quadrants.append(quadrant)

    if len(seen_quadrants) != len(set(seen_quadrants)):
        return False
    if sorted(seen_quadrants) != sorted(quadrants):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    return True


def _generate_case() -> np.ndarray | None:
    template = None
    for _ in range(80):
        template = _random_template()
        if template is not None:
            break
    if template is None:
        return None

    total_sprites = random.choices([4, 5], weights=[2, 5])[0]
    damaged_quadrants = random.sample(QUADRANT_ORDER, total_sprites - 1)
    damaged_sprites = []
    for quadrant in damaged_quadrants:
        candidates = _damage_candidates(template, quadrant)
        if not candidates:
            return None
        damaged_sprites.append(random.choice(candidates))

    sprite_order = [template.copy(), *damaged_sprites]
    random.shuffle(sprite_order)

    if total_sprites == 5:
        height = random.randint(20, 24)
        width = random.randint(20, 24)
    else:
        height = random.randint(17, 24)
        width = random.randint(17, 24)

    positions = _random_positions(height, width, total_sprites)
    if positions is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    for (r0, c0), sprite in zip(positions, sprite_order):
        inp[r0 : r0 + SPRITE_SIDE, c0 : c0 + SPRITE_SIDE] = sprite
        expected[r0 : r0 + SPRITE_SIDE, c0 : c0 + SPRITE_SIDE] = template

    if not _semantic_checks(inp, expected, template, damaged_quadrants):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotational-template repair input")
