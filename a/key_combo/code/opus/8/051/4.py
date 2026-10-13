from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

MIN_PIXELS = 3
MAX_PIXELS = 7
MIN_QUAD = 6
MAX_QUAD = 9
GEN_ATTEMPTS = 1200
LOCAL_SHORTCUT_FREE_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    remaining = set(coords)
    out: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                nxt = (nr, nc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
        out.append(comp)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    sprite = _trim(mask)
    variants = []
    for k in range(4):
        rot = np.rot90(sprite, k)
        variants.append(_mask_key(rot))
        variants.append(_mask_key(np.fliplr(rot)))
    return min(variants)


def _pad_top_left(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    out[:h, :w] = mask
    return out


def _full_union_candidates(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for dst in range(len(masks)):
        others = [idx for idx in range(len(masks)) if idx != dst]
        for src_a, src_b in itertools.combinations(others, 2):
            a = masks[src_a]
            b = masks[src_b]
            c = masks[dst]
            if not np.array_equal(a | b, c):
                continue
            if not (np.any(a & ~b) and np.any(b & ~a)):
                continue
            candidates.append((src_a, src_b, dst))
    return sorted(candidates)


def _normalized_union_candidates(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    trimmed = [_trim(mask) for mask in masks]
    for dst in range(len(trimmed)):
        others = [idx for idx in range(len(trimmed)) if idx != dst]
        for src_a, src_b in itertools.combinations(others, 2):
            a = trimmed[src_a]
            b = trimmed[src_b]
            c = trimmed[dst]
            h = max(a.shape[0], b.shape[0], c.shape[0])
            w = max(a.shape[1], b.shape[1], c.shape[1])
            pa = _pad_top_left(a, (h, w))
            pb = _pad_top_left(b, (h, w))
            pc = _pad_top_left(c, (h, w))
            if not np.array_equal(pa | pb, pc):
                continue
            if not (np.any(pa & ~pb) and np.any(pb & ~pa)):
                continue
            candidates.append((src_a, src_b, dst))
    return sorted(candidates)


def _extract_single_sprite_mask(patch: np.ndarray) -> np.ndarray:
    values = set(map(int, np.unique(patch)))
    assert values <= {BLACK, RED}, "quadrants may only contain black background and one red sprite"
    mask = patch == RED
    assert MIN_PIXELS <= int(mask.sum()) <= MAX_PIXELS, "sprite area must be 3-7 pixels"
    assert _is_connected(mask), "sprite must be 4-connected"
    assert len(_components(mask)) == 1, "each quadrant must contain exactly one sprite"
    return mask


def _quadrant_masks(grid: np.ndarray) -> list[np.ndarray]:
    grey_rows = np.where(np.all(grid == GREY, axis=1))[0]
    grey_cols = np.where(np.all(grid == GREY, axis=0))[0]
    assert len(grey_rows) == 1 and len(grey_cols) == 1, "expected one full grey separator row and column"
    sep_r = int(grey_rows[0])
    sep_c = int(grey_cols[0])

    top_h = sep_r
    bottom_h = grid.shape[0] - sep_r - 1
    left_w = sep_c
    right_w = grid.shape[1] - sep_c - 1
    assert top_h == bottom_h and left_w == right_w, "separator must split the grid into equal quadrants"

    quadrants = [
        grid[:sep_r, :sep_c],
        grid[:sep_r, sep_c + 1 :],
        grid[sep_r + 1 :, :sep_c],
        grid[sep_r + 1 :, sep_c + 1 :],
    ]
    shapes = {patch.shape for patch in quadrants}
    assert len(shapes) == 1, "all quadrants must have the same size"
    return [_extract_single_sprite_mask(patch) for patch in quadrants]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the quadrant sprites linked by a local-coordinate pixel union and compress their roles to 2x2.

    Input structure:
    - The grid has black background, one full grey separator row, and one full grey separator column.
    - Those separators split the canvas into four equal quadrants.
    - Each quadrant contains exactly one 4-connected red sprite mask with 3-7 pixels.

    Rule:
    - Convert each quadrant to a boolean sprite mask in that quadrant's own full local coordinates.
    - Find the unique triple (A, B, C) for which A OR B equals C, with both operands contributing at least one pixel not supplied by the other.
    - Output a 2x2 grid aligned with the four quadrants: blue for operand quadrants, green for the result quadrant, and black for the unrelated quadrant.

    Key insight:
    - The relation is invisible until the sprites are compared in the common local frame defined by each quadrant's top-left corner; cropping each sprite independently is the wrong reference system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    masks = _quadrant_masks(g)
    candidates = _full_union_candidates(masks)
    assert len(candidates) == 1, "expected exactly one valid union triple"

    src_a, src_b, dst = candidates[0]
    out = np.zeros((2, 2), dtype=int)
    for idx, color in ((src_a, BLUE), (src_b, BLUE), (dst, GREEN)):
        out[idx // 2, idx % 2] = color
    return out


def _random_connected_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    area: int,
) -> np.ndarray:
    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        start = (int(rng.integers(height)), int(rng.integers(width)))
        mask[start] = True
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < height and 0 <= nc < width
        }
        while int(mask.sum()) < area and frontier:
            choices = sorted(frontier)
            pick = choices[int(rng.integers(len(choices)))]
            frontier.remove(pick)
            mask[pick] = True
            for nr, nc in _neighbors4(*pick):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if int(mask.sum()) == area and _is_connected(mask):
            return mask
    raise RuntimeError("failed to generate a connected mask")


def _connected_subsets(mask: np.ndarray) -> list[np.ndarray]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    out: list[np.ndarray] = []
    for bits in range(1, 1 << len(coords)):
        area = bits.bit_count()
        if area < MIN_PIXELS or area >= len(coords):
            continue
        sub = np.zeros_like(mask, dtype=bool)
        for idx, rc in enumerate(coords):
            if bits & (1 << idx):
                sub[rc] = True
        if _is_connected(sub):
            out.append(sub)
    return out


def _operand_pairs_for_result(result: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    subsets = _connected_subsets(result)
    triples: list[tuple[np.ndarray, np.ndarray]] = []
    result_key = _shape_key(result)
    for i, a in enumerate(subsets):
        key_a = _shape_key(a)
        if key_a == result_key:
            continue
        for b in subsets[i + 1 :]:
            key_b = _shape_key(b)
            if len({key_a, key_b, result_key}) < 3:
                continue
            if not np.array_equal(a | b, result):
                continue
            if not (np.any(a & ~b) and np.any(b & ~a)):
                continue
            if _normalized_union_candidates([a, b, result]):
                continue
            triples.append((a.copy(), b.copy()))
    return triples


def _distractor_area_ok(mode: int, area_d: int, area_a: int, area_b: int, area_c: int) -> bool:
    if mode == 0:
        return area_d >= area_c
    if mode == 1:
        return area_d in {area_a, area_b}
    return area_d < min(area_a, area_b)


def _valid_mask_set(masks: list[np.ndarray], area_mode: int) -> bool:
    if len(masks) != 4:
        return False
    areas = [int(mask.sum()) for mask in masks]
    if any(area < MIN_PIXELS or area > MAX_PIXELS for area in areas):
        return False
    if any(not _is_connected(mask) for mask in masks):
        return False
    if len({_shape_key(mask) for mask in masks}) != 4:
        return False
    candidates = _full_union_candidates(masks)
    if candidates != [(0, 1, 2)]:
        return False
    if _normalized_union_candidates(masks):
        return False
    area_a, area_b, area_c, area_d = areas
    if not _distractor_area_ok(area_mode, area_d, area_a, area_b, area_c):
        return False
    return True


def _render_case(masks_by_role: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = list(map(int, rng.permutation(4)))
    positioned_masks = [masks_by_role[idx] for idx in order]
    qh, qw = positioned_masks[0].shape
    grid = np.zeros((2 * qh + 1, 2 * qw + 1), dtype=int)
    grid[qh, :] = GREY
    grid[:, qw] = GREY

    slices = [
        (slice(0, qh), slice(0, qw)),
        (slice(0, qh), slice(qw + 1, 2 * qw + 1)),
        (slice(qh + 1, 2 * qh + 1), slice(0, qw)),
        (slice(qh + 1, 2 * qh + 1), slice(qw + 1, 2 * qw + 1)),
    ]
    for patch_slice, mask in zip(slices, positioned_masks):
        patch = grid[patch_slice]
        patch[mask] = RED

    expected = np.zeros((2, 2), dtype=int)
    for pos, role_idx in enumerate(order):
        if role_idx in (0, 1):
            expected[pos // 2, pos % 2] = BLUE
        elif role_idx == 2:
            expected[pos // 2, pos % 2] = GREEN
    return grid, expected


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    qh = int(rng.integers(MIN_QUAD, MAX_QUAD + 1))
    qw = int(rng.integers(MIN_QUAD, MAX_QUAD + 1))

    for _ in range(LOCAL_SHORTCUT_FREE_ATTEMPTS):
        c_area = int(rng.integers(4, MAX_PIXELS + 1))
        result = _random_connected_mask(rng, qh, qw, c_area)
        pairs = _operand_pairs_for_result(result)
        if not pairs:
            continue
        a, b = pairs[int(rng.integers(len(pairs)))]
        area_mode = int(rng.integers(3))
        area_a = int(a.sum())
        area_b = int(b.sum())
        area_c = int(result.sum())

        for _ in range(120):
            if area_mode == 0:
                choices = list(range(area_c, MAX_PIXELS + 1))
            elif area_mode == 1:
                choices = sorted({area_a, area_b})
            else:
                choices = list(range(MIN_PIXELS, min(area_a, area_b)))
            if not choices:
                break
            d_area = int(rng.choice(choices))
            distractor = _random_connected_mask(rng, qh, qw, d_area)
            masks_by_role = [a, b, result, distractor]
            if not _valid_mask_set(masks_by_role, area_mode):
                continue
            grid, expected = _render_case(masks_by_role, rng)
            if not np.array_equal(main(grid), expected):
                continue
            return grid, expected
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid quadrant-union puzzle")
