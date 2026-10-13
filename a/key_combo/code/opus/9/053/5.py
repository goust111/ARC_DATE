from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
FOREGROUND = 3
OUTPUT_SIZE = 9
SPRITE_SIZE = 3
OBJECT_COUNTS = (3, 4, 5)
GRID_MIN = 15
GRID_MAX = 23
MIN_BBOX_GAP = 1
MAX_GENERATION_ATTEMPTS = 500

_RNG = np.random.default_rng()
_SEEN_OUTPUT_SIGNATURES: set[bytes] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    local = coords - np.array([[r0, c0]], dtype=int)
    crop[local[:, 0], local[:, 1]] = grid[coords[:, 0], coords[:, 1]]
    return crop


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _stamp_output(control_mask: np.ndarray, template_crop: np.ndarray, bg: int) -> np.ndarray:
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), int(bg), dtype=int)
    for r, c in np.argwhere(control_mask):
        rr = int(r) * SPRITE_SIZE
        cc = int(c) * SPRITE_SIZE
        out[rr : rr + SPRITE_SIZE, cc : cc + SPRITE_SIZE] = template_crop
    return out


def _analyze_input(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    fg_colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])

    comps = _components_mask(g != bg, connectivity=8)
    assert 3 <= len(comps) <= 5, "expected 3 to 5 foreground sprites"

    template_crops: list[np.ndarray] = []
    control_masks: list[np.ndarray] = []
    for coords in comps:
        crop = _crop_component(g, coords, bg)
        assert crop.shape == (SPRITE_SIZE, SPRITE_SIZE), "expected every sprite bbox to be 3x3"
        mask = crop != bg
        if _is_lr_symmetric(mask):
            template_crops.append(crop)
        else:
            control_masks.append(mask)

    assert len(control_masks) == 1, "expected exactly one asymmetric controller sprite"
    assert len(template_crops) >= 2, "expected at least two symmetric template sprites"
    template_mask = template_crops[0] != bg
    for crop in template_crops[1:]:
        assert np.array_equal(crop != bg, template_mask), "all symmetric sprites must share one mask"

    return bg, fg, control_masks[0], template_crops[0], len(template_crops)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique left-right-asymmetric sprite as a 3x3 stamp controller for the repeated symmetric sprite.

    Input structure:
    - A black background contains 3 to 5 same-color 8-connected sprites.
    - Every sprite has a tight 3x3 bounding box, and the sprites are separated from each other.
    - Exactly one sprite is not mirror-symmetric left-to-right; all remaining sprites are left-right symmetric and share the same 3x3 mask.

    Rule:
    - Split the foreground into 8-connected sprites and crop each one to its tight 3x3 box.
    - Identify the unique left-right-asymmetric crop as the controller mask C.
    - Use any symmetric crop as the template stamp T.
    - Build a 9x9 output divided into 3x3 blocks; for every foreground pixel of C at (i, j), copy T into block (i, j), and leave all other blocks black.

    Key insight:
    - The odd sprite is not the answer to extract; it is a placement map telling where copies of the repeated symmetric sprite must be stamped.
    """
    bg, _fg, control_mask, template_crop, _template_count = _analyze_input(grid)
    return _stamp_output(control_mask, template_crop, bg)


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask, connectivity=8)
    return len(comps) == 1 and int(comps[0].shape[0]) == int(mask.sum())


def _has_exact_3x3_bbox(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return False
    return (
        int(rows.min()) == 0
        and int(rows.max()) == SPRITE_SIZE - 1
        and int(cols.min()) == 0
        and int(cols.max()) == SPRITE_SIZE - 1
    )


def _enumerate_mask_pools() -> tuple[list[np.ndarray], list[np.ndarray]]:
    template_masks: list[np.ndarray] = []
    control_masks: list[np.ndarray] = []
    for bits in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        flat = np.array([(bits >> k) & 1 for k in range(SPRITE_SIZE * SPRITE_SIZE)], dtype=bool)
        mask = flat.reshape(SPRITE_SIZE, SPRITE_SIZE)
        area = int(mask.sum())
        if not _has_exact_3x3_bbox(mask) or not _is_connected(mask):
            continue
        if _is_lr_symmetric(mask):
            if 4 <= area <= 7 and area != SPRITE_SIZE * SPRITE_SIZE:
                template_masks.append(mask.copy())
        else:
            if 3 <= area <= 6:
                control_masks.append(mask.copy())
    assert template_masks and control_masks
    return template_masks, control_masks


TEMPLATE_MASKS, CONTROL_MASKS = _enumerate_mask_pools()


def _mask_to_patch(mask: np.ndarray, fg: int = FOREGROUND, bg: int = BACKGROUND) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(fg)
    return patch


def _sample_spaced_starts(limit: int, count: int) -> list[int]:
    max_start = int(limit) - SPRITE_SIZE
    baseline_last = (SPRITE_SIZE + MIN_BBOX_GAP) * (count - 1)
    slack = max_start - baseline_last
    assert slack >= 0
    gaps = _RNG.multinomial(slack, np.full(count + 1, 1.0 / (count + 1)))
    starts: list[int] = []
    cur = int(gaps[0])
    for i in range(count):
        starts.append(cur)
        if i < count - 1:
            cur += SPRITE_SIZE + MIN_BBOX_GAP + int(gaps[i + 1])
    return starts


def _sample_positions(height: int, width: int, count: int) -> list[tuple[int, int]]:
    rows = _sample_spaced_starts(height, count)
    cols = _sample_spaced_starts(width, count)
    permuted_cols = list(map(int, _RNG.permutation(cols)))
    return [(int(r), int(c)) for r, c in zip(rows, permuted_cols)]


def _output_signature(grid: np.ndarray) -> bytes:
    return grid.tobytes() + str(tuple(map(int, grid.shape))).encode()


def _validate_generated_case(grid: np.ndarray) -> np.ndarray:
    bg, fg, control_mask, template_crop, template_count = _analyze_input(grid)
    assert bg == BACKGROUND, "generator must keep the dominant background black"
    assert fg == FOREGROUND, "generator uses a single visible foreground color"
    assert template_count >= 2

    template_mask = template_crop != bg
    control_area = int(control_mask.sum())
    template_area = int(template_mask.sum())
    assert 3 <= control_area <= 6
    assert 4 <= template_area <= 7
    assert not np.array_equal(control_mask, template_mask), "controller and template roles must differ"

    out = main(grid)
    expected = _stamp_output(control_mask, template_crop, bg)
    assert np.array_equal(out, expected)

    solid_blocks = np.kron(control_mask.astype(int), np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=int)) * fg
    assert not np.array_equal(out, solid_blocks), "template must matter; reject solid-block degeneracy"

    filled_blocks = int(control_mask.sum())
    assert 3 <= filled_blocks <= 6
    assert filled_blocks < SPRITE_SIZE * SPRITE_SIZE, "controller must leave some output blocks empty"

    comps = _components_mask(grid != bg, connectivity=8)
    bboxes = [_bbox_coords(coords) for coords in comps]
    for i in range(len(bboxes)):
        r0a, c0a, r1a, c1a = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            r0b, c0b, r1b, c1b = bboxes[j]
            row_gap = max(r0b - r1a - 1, r0a - r1b - 1)
            col_gap = max(c0b - c1a - 1, c0a - c1b - 1)
            assert row_gap >= MIN_BBOX_GAP and col_gap >= MIN_BBOX_GAP, "sprite bboxes must leave a row and a column between them"

    return out


def _generate_candidate() -> np.ndarray:
    object_count = int(_RNG.choice(OBJECT_COUNTS))
    min_dim = max(GRID_MIN, (SPRITE_SIZE + MIN_BBOX_GAP) * (object_count - 1) + SPRITE_SIZE)
    height = int(_RNG.integers(min_dim, GRID_MAX + 1))
    width = int(_RNG.integers(min_dim, GRID_MAX + 1))

    template_mask = TEMPLATE_MASKS[int(_RNG.integers(len(TEMPLATE_MASKS)))]
    control_mask = CONTROL_MASKS[int(_RNG.integers(len(CONTROL_MASKS)))]
    template_patch = _mask_to_patch(template_mask)
    control_patch = _mask_to_patch(control_mask)

    patches = [template_patch.copy() for _ in range(object_count - 1)] + [control_patch]
    _RNG.shuffle(patches)
    positions = _sample_positions(height, width, object_count)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for patch, (r0, c0) in zip(patches, positions):
        grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = patch
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        out = _validate_generated_case(grid)
        sig = _output_signature(out)
        if sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(sig)
        return grid
    raise RuntimeError("failed to generate a fresh valid asymmetric-controller stamp puzzle")
