from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

ROTATION_TO_COLOR = {0: RED, 1: GREEN, 2: BLUE, 3: YELLOW}
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    mask[rr, cc] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.astype(np.uint8).tobytes()


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), -turns)


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(_rotate_cw(mask, k)) for k in range(4))


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_holes(left: np.ndarray) -> list[np.ndarray]:
    holes = []
    for coords in _components(left == BACKGROUND):
        if not _touches_border(coords, left.shape):
            holes.append(coords)
    return holes


def _extract_objects(right: np.ndarray) -> list[np.ndarray]:
    return _components(right == GREY)


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    separator_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert separator_cols.size == 1, "expected one full-height grey separator"
    sep = int(separator_cols[0])
    assert 1 < sep < g.shape[1] - 2

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    holes = _extract_holes(left)
    objects = _extract_objects(right)
    assert 3 <= len(holes) <= 4
    assert len(holes) == len(objects)

    hole_masks = [_coords_to_mask(coords) for coords in holes]
    object_masks = [_coords_to_mask(coords) for coords in objects]

    matches: list[tuple[int, int]] = []
    used_objects: set[int] = set()
    for hole_mask in hole_masks:
        local = []
        for obj_idx, obj_mask in enumerate(object_masks):
            for turns in range(4):
                if np.array_equal(_rotate_cw(hole_mask, turns), obj_mask):
                    local.append((obj_idx, turns))
        assert len(local) == 1, "each hole must have one unique rotation-equivalent object"
        obj_idx, turns = local[0]
        assert obj_idx not in used_objects, "objects must match bijectively"
        used_objects.add(obj_idx)
        matches.append((obj_idx, turns))

    assert len(used_objects) == len(objects)
    return left, holes, matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each left-panel hole with a color that encodes how many clockwise quarter-turns align it to a right-panel shape.

    Input structure:
    - A full-height grey separator splits the grid into a left template panel and a right candidate panel.
    - The left panel contains one orange connected template with 3 or 4 enclosed black hole-regions.
    - The right panel contains the same number of grey 4-connected shapes on a black background.

    Rule:
    - Extract every enclosed black hole from the left panel and normalize each one to its tight bounding box.
    - Extract every grey right-panel object and normalize it to its tight bounding box.
    - For each hole, find the unique right-panel object that matches after 0, 1, 2, or 3 clockwise 90-degree rotations.
    - Fill the hole in the left panel with the rotation-code color: 0 turns -> red, 1 -> green, 2 -> blue, 3 -> yellow.

    Key insight:
    - The right-panel shapes act as rotation clues rather than color clues: the number of turns needed to align the hole to its partner determines the fill color.
    """
    left, holes, matches = _analyze(np.asarray(grid, dtype=int))
    out = left.copy()
    for coords, (_, turns) in zip(holes, matches):
        out[coords[:, 0], coords[:, 1]] = ROTATION_TO_COLOR[turns]
    return out


@lru_cache(None)
def _shape_catalog() -> tuple[np.ndarray, ...]:
    catalog: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for h in range(1, 5):
        for w in range(1, 5):
            n = h * w
            for bits in range(1, 1 << n):
                area = bits.bit_count()
                if not (3 <= area <= 6):
                    continue
                mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool).reshape(h, w)
                if len(_components(mask)) != 1:
                    continue
                coords = np.argwhere(mask)
                norm = _coords_to_mask(coords)
                if norm.shape != mask.shape:
                    continue
                if len({_mask_key(_rotate_cw(norm, k)) for k in range(4)}) != 4:
                    continue
                catalog.setdefault(_canonical_rotation_key(norm), norm)
    return tuple(catalog.values())


def _placement_ok(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, border: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < border or c0 < border or r0 + mh > h - border or c0 + mw > w - border:
        return False
    rr, cc = np.where(mask)
    for dr, dc in zip(rr, cc):
        ar = r0 + int(dr)
        ac = c0 + int(dc)
        for nr in range(max(0, ar - 1), min(h, ar + 2)):
            for nc in range(max(0, ac - 1), min(w, ac + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _place_masks(shape: tuple[int, int], masks: list[np.ndarray], border: int = 1) -> list[tuple[int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        candidates = []
        for r0 in range(border, shape[0] - mh - border + 1):
            for c0 in range(border, shape[1] - mw - border + 1):
                if _placement_ok(occupied, mask, r0, c0, border):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        rr, cc = np.where(mask)
        occupied[r0 + rr, c0 + cc] = True
    return [(int(r0), int(c0)) for r0, c0 in placements]  # type: ignore[misc]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rr, cc = np.where(mask)
    grid[r0 + rr, c0 + cc] = color


def _choose_rotations(count: int) -> list[int]:
    if count == 4:
        turns = [0, 1, 2, 3]
        random.shuffle(turns)
        return turns
    omitted = random.randrange(4)
    turns = [k for k in range(4) if k != omitted]
    random.shuffle(turns)
    return turns


def _semantic_validation(inp: np.ndarray, expected: np.ndarray, rotations: list[int]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    left, holes, matches = _analyze(inp)
    if not np.array_equal(left.shape, expected.shape):
        return False

    seen_turns = sorted(turns for _, turns in matches)
    if seen_turns != sorted(rotations):
        return False

    hole_masks = [_coords_to_mask(coords) for coords in holes]
    if len({_canonical_rotation_key(mask) for mask in hole_masks}) != len(hole_masks):
        return False
    if any(len({_mask_key(_rotate_cw(mask, k)) for k in range(4)}) != 4 for mask in hole_masks):
        return False

    colors = [int(expected[coords[0, 0], coords[0, 1]]) for coords in holes]
    if sorted(colors) != sorted(ROTATION_TO_COLOR[t] for t in rotations):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    hole_count = 4 if random.random() < 0.65 else 3
    rotations = _choose_rotations(hole_count)
    hole_masks = [mask.copy() for mask in random.sample(list(_shape_catalog()), hole_count)]
    random.shuffle(hole_masks)

    h = random.randint(12, 16)
    left_w = random.randint(10, 13)
    right_w = random.randint(9, 13)

    hole_positions = _place_masks((h, left_w), hole_masks, border=1)
    if hole_positions is None:
        return None

    right_masks = [_rotate_cw(mask, turns) for mask, turns in zip(hole_masks, rotations)]
    right_positions = _place_masks((h, right_w), right_masks, border=1)
    if right_positions is None:
        return None

    left_in = np.full((h, left_w), ORANGE, dtype=int)
    left_out = left_in.copy()
    for mask, (r0, c0), turns in zip(hole_masks, hole_positions, rotations):
        _paint_mask(left_in, mask, r0, c0, BACKGROUND)
        _paint_mask(left_out, mask, r0, c0, ROTATION_TO_COLOR[turns])

    right = np.zeros((h, right_w), dtype=int)
    for mask, (r0, c0) in zip(right_masks, right_positions):
        _paint_mask(right, mask, r0, c0, GREY)

    inp = np.concatenate([left_in, np.full((h, 1), GREY, dtype=int), right], axis=1)

    if not _semantic_validation(inp, left_out, rotations):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-coded hole-filling input")
