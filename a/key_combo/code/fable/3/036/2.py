from __future__ import annotations

import numpy as np


MASK_COLOR = 3
GENERATION_RETRIES = 400
QUAD_SIZE_MIN = 7
QUAD_SIZE_MAX = 15
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 5
NON_MASK_COLORS = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
SHAPE_KINDS = ("rect", "el", "tee", "frame")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _split_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    side = int(grid.shape[0] // 2)
    return [
        grid[:side, :side],
        grid[:side, side:],
        grid[side:, :side],
        grid[side:, side:],
    ]


def _rotation_matches(src: np.ndarray, dst: np.ndarray) -> list[int]:
    return [k for k in range(4) if np.array_equal(np.rot90(src, k), dst)]


def main(grid: np.ndarray) -> np.ndarray:
    """Use three complete rotated quadrants to recover the masked crop from the fourth.

    Input structure
    - The grid is an even-sized square split into four equal quadrants with no divider.
    - The four quadrants are the four 90-degree rotations of one asymmetric base image,
      but their placement in the 2x2 layout is shuffled.
    - Exactly one quadrant contains a solid green rectangle masking part of that rotated image.

    Rule
    - Isolate the masked quadrant and its rectangular green bbox.
    - Compare the three visible quadrants by rotation, treating their rotation labels as unknown,
      and infer which relative quarter-turn is missing.
    - Rotate one visible quadrant by that missing amount to reconstruct the hidden quadrant,
      then return only the cells inside the green rectangle's bbox.

    Key insight
    - The quadrant layout itself carries no fixed rotation meaning; only the mutual rotation
      agreement among the three intact quadrants identifies the hidden orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert height == width and height % 2 == 0

    quadrants = _split_quadrants(g)
    masked_indices = [idx for idx, quad in enumerate(quadrants) if np.any(quad == MASK_COLOR)]
    assert len(masked_indices) == 1
    masked_index = masked_indices[0]
    masked_quad = quadrants[masked_index]

    mask = masked_quad == MASK_COLOR
    mask_box = _bbox(mask)
    assert mask_box is not None
    r0, c0, r1, c1 = mask_box
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1])

    visible_quads = [quad for idx, quad in enumerate(quadrants) if idx != masked_index]
    ref = visible_quads[0]
    seen_turns = {0}
    for quad in visible_quads[1:]:
        matches = _rotation_matches(ref, quad)
        assert len(matches) == 1
        seen_turns.add(matches[0])
    assert len(seen_turns) == 3

    missing_turn = next(iter({0, 1, 2, 3} - seen_turns))
    restored_quad = np.rot90(ref, missing_turn)
    assert np.array_equal(masked_quad[~mask], restored_quad[~mask])
    return restored_quad[r0 : r1 + 1, c0 : c1 + 1].copy()


def _sample_shape_mask(quad_size: int) -> np.ndarray:
    max_h = min(5, quad_size - 1)
    max_w = min(5, quad_size - 1)
    kind = SHAPE_KINDS[int(np.random.randint(len(SHAPE_KINDS)))]

    if kind == "rect":
        h = int(np.random.randint(1, max_h + 1))
        w = int(np.random.randint(1, max_w + 1))
        if h == 1 and w == 1:
            w = min(2, max_w)
        mask = np.ones((h, w), dtype=bool)
    elif kind == "el":
        h = int(np.random.randint(3, max_h + 1))
        w = int(np.random.randint(3, max_w + 1))
        thickness = int(np.random.randint(1, min(2, h - 1, w - 1) + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :thickness] = True
        mask[-thickness:, :] = True
    elif kind == "tee":
        h = int(np.random.randint(3, max_h + 1))
        w = int(np.random.randint(3, max_w + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        center = int(np.random.randint(w))
        mask[:, center] = True
    else:
        h = int(np.random.randint(3, max_h + 1))
        w = int(np.random.randint(3, max_w + 1))
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

    if int(np.random.randint(2)) == 1:
        mask = np.flipud(mask)
    if int(np.random.randint(2)) == 1:
        mask = np.fliplr(mask)
    return mask


def _find_valid_positions(forbidden: np.ndarray, shape_mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = map(int, forbidden.shape)
    shape_h, shape_w = map(int, shape_mask.shape)
    coords = np.argwhere(shape_mask)
    positions: list[tuple[int, int]] = []
    for r0 in range(height - shape_h + 1):
        for c0 in range(width - shape_w + 1):
            if all(not forbidden[r0 + dr, c0 + dc] for dr, dc in coords):
                positions.append((r0, c0))
    return positions


def _stamp_shape(
    base: np.ndarray,
    occupied: np.ndarray,
    forbidden: np.ndarray,
    shape_mask: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    shape_h, shape_w = map(int, shape_mask.shape)
    for dr in range(shape_h):
        for dc in range(shape_w):
            if not shape_mask[dr, dc]:
                continue
            rr = top + dr
            cc = left + dc
            base[rr, cc] = int(color)
            occupied[rr, cc] = True
            for nr in range(max(0, rr - 1), min(base.shape[0], rr + 2)):
                for nc in range(max(0, cc - 1), min(base.shape[1], cc + 2)):
                    forbidden[nr, nc] = True


def _rotations_are_distinct(base: np.ndarray) -> bool:
    return all(not np.array_equal(base, np.rot90(base, k)) for k in (1, 2, 3))


def _generate_base() -> tuple[np.ndarray, int] | None:
    quad_size = int(np.random.randint(QUAD_SIZE_MIN, QUAD_SIZE_MAX + 1))
    bg = int(NON_MASK_COLORS[int(np.random.randint(len(NON_MASK_COLORS)))])
    palette = [int(c) for c in NON_MASK_COLORS if int(c) != bg]
    np.random.shuffle(palette)
    object_count = int(np.random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1))
    colors = palette[:object_count]

    base = np.full((quad_size, quad_size), bg, dtype=int)
    occupied = np.zeros((quad_size, quad_size), dtype=bool)
    forbidden = np.zeros((quad_size, quad_size), dtype=bool)

    for color in colors:
        placed = False
        for _ in range(50):
            shape_mask = _sample_shape_mask(quad_size)
            positions = _find_valid_positions(forbidden, shape_mask)
            if not positions:
                continue
            top, left = positions[int(np.random.randint(len(positions)))]
            _stamp_shape(base, occupied, forbidden, shape_mask, top, left, color)
            placed = True
            break
        if not placed:
            return None

    fg_mask = base != bg
    if int(fg_mask.sum()) < quad_size:
        return None
    if int(fg_mask.sum()) > (quad_size * quad_size) // 2:
        return None
    if len(np.unique(base[fg_mask])) < 3:
        return None
    if not _rotations_are_distinct(base):
        return None
    return base, bg


def _choose_mask_bbox(rotated_quad: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    side = int(rotated_quad.shape[0])
    max_span = min(6, side - 1)
    area_cap = max(6, (side * side) // 3)
    candidates: list[tuple[int, int, int, int, int]] = []
    for h in range(2, max_span + 1):
        for w in range(2, max_span + 1):
            if h * w > area_cap:
                continue
            for r0 in range(side - h + 1):
                for c0 in range(side - w + 1):
                    patch = rotated_quad[r0 : r0 + h, c0 : c0 + w]
                    fg_count = int(np.count_nonzero(patch != bg))
                    if fg_count == 0:
                        continue
                    unique_colors = int(len(np.unique(patch)))
                    score = 100 * int(unique_colors >= 2) + 10 * fg_count - h * w
                    candidates.append((score, r0, c0, h, w))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    top = candidates[: min(40, len(candidates))]
    _score, r0, c0, h, w = top[int(np.random.randint(len(top)))]
    return int(r0), int(c0), int(h), int(w)


def _assemble_grid(base: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    quad_size = int(base.shape[0])
    rotations = np.random.permutation(4).astype(int).tolist()
    quadrants = [np.rot90(base, turns).copy() for turns in rotations]

    masked_index = int(np.random.randint(4))
    masked_quad = quadrants[masked_index].copy()
    mask_choice = _choose_mask_bbox(masked_quad, bg)
    if mask_choice is None:
        return None
    r0, c0, mask_h, mask_w = mask_choice
    expected = masked_quad[r0 : r0 + mask_h, c0 : c0 + mask_w].copy()
    masked_quad[r0 : r0 + mask_h, c0 : c0 + mask_w] = MASK_COLOR
    quadrants[masked_index] = masked_quad

    grid = np.empty((quad_size * 2, quad_size * 2), dtype=int)
    grid[:quad_size, :quad_size] = quadrants[0]
    grid[:quad_size, quad_size:] = quadrants[1]
    grid[quad_size:, :quad_size] = quadrants[2]
    grid[quad_size:, quad_size:] = quadrants[3]
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    generated = _generate_base()
    if generated is None:
        return None
    base, bg = generated
    assembled = _assemble_grid(base, bg)
    if assembled is None:
        return None
    grid, expected = assembled

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 036/2")
