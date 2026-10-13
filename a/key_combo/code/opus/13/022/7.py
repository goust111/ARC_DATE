from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
SPRITE_SIZE = 5
OUTPUT_SIZE = 11
OUTPUT_SLOTS = ((0, 0), (0, 6), (6, 0), (6, 6))
GRID_MIN = 19
GRID_MAX = 24
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = _neighbors4 if connectivity == 4 else _neighbors8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _gray_sprites(grid: np.ndarray) -> list[dict]:
    comps = _components_from_mask(grid == GREY, connectivity=8)
    sprites = []
    for coords in comps:
        box = _bbox(coords)
        patch = _crop(grid, box)
        sprites.append({"coords": coords, "bbox": box, "patch": patch})
    return sprites


def _enclosed_region_and_red_count(patch: np.ndarray) -> tuple[np.ndarray, int]:
    comps = _components_from_mask(patch != GREY, connectivity=4)
    enclosed = [coords for coords in comps if not _touches_edge(coords, patch.shape)]
    assert len(enclosed) == 1, "expected exactly one enclosed non-grey cavity"
    cavity = enclosed[0]
    red_count = int(sum(int(patch[r, c] == RED) for r, c in cavity))
    return cavity, red_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the four hollow grey sprites by how many red pixels lie inside each enclosed cavity, then pack them into a 2x2 atlas.

    Input structure:
    - Black background contains exactly four separated 8-connected grey sprites.
    - Each grey sprite occupies a 5x5 bounding box and encloses one internal cavity made of non-grey cells that does not touch the box edge.
    - The cavity may contain black cells and 0-3 red pixels.

    Rule:
    - Find the four grey sprites and crop each 5x5 bounding box unchanged.
    - Inside each crop, locate the unique enclosed 4-connected non-grey cavity and count its red pixels.
    - Sort the crops by red count ascending, breaking ties by the sprite's top-left input position, then place them into a 2x2 11x11 atlas with one black separator row and column.

    Key insight:
    - The output order is driven by hidden content inside each sprite's hole rather than by the sprites' external positions in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprites = _gray_sprites(g)
    assert len(sprites) == 4, "expected exactly four grey sprites"

    ranked = []
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE)
        patch = sprite["patch"]
        _cavity, red_count = _enclosed_region_and_red_count(patch)
        ranked.append((red_count, r0, c0, patch))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))

    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BG, dtype=int)
    for (_count, _r0, _c0, patch), (slot_r, slot_c) in zip(ranked, OUTPUT_SLOTS):
        _paste(out, patch, slot_r, slot_c)
    return out


_CAVITY_BANK: dict[int, list[np.ndarray]] | None = None


def _build_cavity_bank() -> dict[int, list[np.ndarray]]:
    bank: dict[int, list[np.ndarray]] = {}
    for bits in range(1, 1 << 9):
        cavity = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(cavity.sum())
        if area < 4:
            continue
        if len(_components_from_mask(cavity, connectivity=4)) != 1:
            continue

        patch = np.full((SPRITE_SIZE, SPRITE_SIZE), GREY, dtype=int)
        patch[1:4, 1:4] = np.where(cavity, BG, GREY)
        if len(_components_from_mask(patch == GREY, connectivity=8)) != 1:
            continue

        bank.setdefault(area, []).append(cavity.copy())
    return bank


def _cavity_bank() -> dict[int, list[np.ndarray]]:
    global _CAVITY_BANK
    if _CAVITY_BANK is None:
        _CAVITY_BANK = _build_cavity_bank()
    return _CAVITY_BANK


def _make_sprite(red_count: int) -> np.ndarray:
    valid_areas = [area for area in _cavity_bank() if area >= red_count + 1]
    area = random.choice(valid_areas)
    cavity = random.choice(_cavity_bank()[area]).copy()

    patch = np.full((SPRITE_SIZE, SPRITE_SIZE), GREY, dtype=int)
    patch[1:4, 1:4] = np.where(cavity, BG, GREY)
    cells = [(r + 1, c + 1) for r, c in np.argwhere(cavity)]
    for r, c in random.sample(cells, red_count):
        patch[r, c] = RED
    return patch


def _boxes_too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + SPRITE_SIZE - 1, ac0 + SPRITE_SIZE - 1
    br1, bc1 = br0 + SPRITE_SIZE - 1, bc0 + SPRITE_SIZE - 1
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _random_positions(height: int, width: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(height - SPRITE_SIZE + 1) for c in range(width - SPRITE_SIZE + 1)]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(_boxes_too_close((r, c), other) for other in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == 4:
            return chosen
    return None


def _row_major_order(positions: list[tuple[int, int]]) -> list[int]:
    return sorted(range(len(positions)), key=lambda idx: (positions[idx][0], positions[idx][1]))


def _expected_output(sprites: list[np.ndarray], counts: list[int]) -> np.ndarray:
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BG, dtype=int)
    order = sorted(range(4), key=lambda idx: counts[idx])
    for idx, (slot_r, slot_c) in zip(order, OUTPUT_SLOTS):
        _paste(out, sprites[idx], slot_r, slot_c)
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    counts = [0, 1, 2, 3]
    random.shuffle(counts)
    sprites = [_make_sprite(count) for count in counts]
    positions = _random_positions(height, width)
    if positions is None:
        return None

    if _row_major_order(positions) == sorted(range(4), key=lambda idx: counts[idx]):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for patch, (r0, c0) in zip(sprites, positions):
        _paste(grid, patch, r0, c0)

    expected = _expected_output(sprites, counts)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
