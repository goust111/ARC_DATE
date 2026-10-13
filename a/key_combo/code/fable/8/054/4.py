from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
NON_BLACK = tuple(range(1, 10))
TRANSFORMS = (
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
)
MAX_GENERATION_ATTEMPTS = 600


def _apply_transform(arr: np.ndarray, quarter_turns: int, mirror: bool) -> np.ndarray:
    out = np.rot90(np.asarray(arr, dtype=arr.dtype), k=int(quarter_turns))
    if mirror:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _invert_transform(arr: np.ndarray, quarter_turns: int, mirror: bool) -> np.ndarray:
    out = np.asarray(arr, dtype=arr.dtype)
    if mirror:
        out = np.fliplr(out)
    out = np.rot90(out, k=-int(quarter_turns))
    return np.array(out, copy=True)


def _bbox_nonzero(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(np.asarray(grid) != BLACK)
    assert coords.size > 0, "expected at least one non-black cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_nonzero(grid)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _spans(bits: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(bits.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(bits) - 1))
    return spans


def _extract_two_objects(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    occ = g != BLACK
    row_spans = _spans(occ.any(axis=1))
    col_spans = _spans(occ.any(axis=0))

    pieces: list[np.ndarray] = []
    if len(col_spans) == 2 and len(row_spans) == 1:
        for c0, c1 in col_spans:
            slab = g[:, c0 : c1 + 1]
            pieces.append(_crop_nonzero(slab))
    elif len(row_spans) == 2 and len(col_spans) == 1:
        for r0, r1 in row_spans:
            slab = g[r0 : r1 + 1, :]
            pieces.append(_crop_nonzero(slab))
    else:
        raise AssertionError("expected two rectangular objects separated by a blank band")

    assert len(pieces) == 2
    a, b = pieces
    if a.shape[0] * a.shape[1] <= b.shape[0] * b.shape[1]:
        return a, b
    return b, a


def _block_majority(block: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(block, dtype=int), return_counts=True)
    best = int(np.argmax(counts))
    assert int(np.sum(counts == counts[best])) == 1, "block majority must be unique"
    return int(values[best])


def _majority_grid(big: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, int] | None:
    big = np.asarray(big, dtype=int)
    target_h, target_w = map(int, target_shape)
    if big.shape[0] % target_h != 0 or big.shape[1] % target_w != 0:
        return None
    scale_h = big.shape[0] // target_h
    scale_w = big.shape[1] // target_w
    if scale_h != scale_w:
        return None
    scale = int(scale_h)
    out = np.empty((target_h, target_w), dtype=int)
    for r in range(target_h):
        for c in range(target_w):
            block = big[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            out[r, c] = _block_majority(block)
    return out, scale


def _orientation_candidates(small: np.ndarray, big: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for transform_id, (quarter_turns, mirror) in enumerate(TRANSFORMS):
        oriented_small = _apply_transform(small, quarter_turns, mirror)
        majority_info = _majority_grid(big, oriented_small.shape)
        if majority_info is None:
            continue
        majority, scale = majority_info
        known = oriented_small != BLACK
        if not known.any():
            continue
        mismatches = int(np.sum(majority[known] != oriented_small[known]))
        candidates.append(
            {
                "transform_id": transform_id,
                "quarter_turns": quarter_turns,
                "mirror": mirror,
                "majority": majority,
                "scale": scale,
                "mismatches": mismatches,
            }
        )
    return candidates


def _solve_details(grid: np.ndarray) -> dict:
    small, big = _extract_two_objects(grid)
    candidates = _orientation_candidates(small, big)
    assert candidates, "expected at least one scale/orientation candidate"
    best_mismatch = min(candidate["mismatches"] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate["mismatches"] == best_mismatch]
    assert len(best) == 1, "orientation must be uniquely determined"
    chosen = dict(best[0])
    recovered_full = _invert_transform(
        chosen["majority"], chosen["quarter_turns"], chosen["mirror"]
    )
    out = np.array(small, copy=True)
    out[out == BLACK] = recovered_full[out == BLACK]
    chosen["small"] = small
    chosen["big"] = big
    chosen["output"] = out
    chosen["recovered_full"] = recovered_full
    chosen["best_mismatch"] = best_mismatch
    chosen["candidate_count"] = len(candidates)
    chosen["known_count"] = int(np.sum(small != BLACK))
    chosen["missing_count"] = int(np.sum(small == BLACK))
    chosen["candidates"] = candidates
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the missing cells of a small multicolor rectangle by cross-checking it against a rotated/reflected scaled copy.

    Input structure:
    - The grid has a black background and exactly two separated non-black rectangular objects.
    - The smaller rectangle is a multicolor picture with some cells missing as black holes.
    - The larger rectangle is the complete picture after one unknown rotation/reflection and then an integer scale-up, with blockwise noise inside its bounding box.

    Rule:
    - Identify the small rectangle and the large scaled copy, then infer the integer scale from their bounding-box side ratios.
    - Among the eight dihedral orientations, choose the unique one whose transformed known small cells best agree with the large rectangle's per-block majority colors.
    - Inverse-transform that majority grid back to the small rectangle's frame, keep every known small cell unchanged, and fill only the black missing cells.

    Key insight:
    - The large copy supplies colors only after block-majority downsampling, while the small copy anchors the correct orientation and protects the few heavily corrupted blocks by keeping known cells fixed.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _random_full_pattern(height: int, width: int, palette: tuple[int, ...]) -> np.ndarray:
    while True:
        arr = np.random.choice(palette, size=(height, width))
        if len(np.unique(arr)) < 3:
            continue
        return arr.astype(int)


def _sample_missing_mask(height: int, width: int, heavy_mask: np.ndarray) -> np.ndarray:
    total = height * width
    upper = max(1, min(total - 2, total // 3))
    while True:
        mask = np.zeros((height, width), dtype=bool)
        count = random.randint(1, upper)
        coords = [(r, c) for r in range(height) for c in range(width) if not heavy_mask[r, c]]
        random.shuffle(coords)
        for r, c in coords[:count]:
            mask[r, c] = True
        if not mask.any():
            continue
        if np.any(mask & heavy_mask):
            continue
        if np.any(mask.all(axis=0)) or np.any(mask.all(axis=1)):
            continue
        return mask


def _orient_mask(mask: np.ndarray, quarter_turns: int, mirror: bool) -> np.ndarray:
    return _apply_transform(mask.astype(int), quarter_turns, mirror).astype(bool)


def _scaled_noisy_copy(
    oriented: np.ndarray, scale: int, heavy_mask_oriented: np.ndarray
) -> tuple[np.ndarray, int]:
    h, w = oriented.shape
    big = np.empty((h * scale, w * scale), dtype=int)
    total_noise_pixels = 0
    normal_noise_cap = (scale * scale - 1) // 2
    heavy_noise_floor = scale * scale // 2 + 1

    noisy_blocks: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(oriented[r, c])
            wrong_choices = [candidate for candidate in NON_BLACK if candidate != color]
            wrong = random.choice(wrong_choices)
            block = np.full((scale, scale), color, dtype=int)
            if heavy_mask_oriented[r, c]:
                noise_pixels = random.randint(heavy_noise_floor, scale * scale)
            else:
                if random.random() < 0.7:
                    noise_pixels = random.randint(1, max(1, normal_noise_cap))
                else:
                    noise_pixels = 0
            if noise_pixels > 0:
                indices = random.sample(range(scale * scale), noise_pixels)
                for index in indices:
                    br = index // scale
                    bc = index % scale
                    block[br, bc] = wrong
                total_noise_pixels += noise_pixels
                noisy_blocks.append((r, c))
            big[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = block

    if total_noise_pixels == 0:
        r, c = random.choice([(r, c) for r in range(h) for c in range(w) if not heavy_mask_oriented[r, c]])
        color = int(oriented[r, c])
        wrong = random.choice([candidate for candidate in NON_BLACK if candidate != color])
        big[r * scale, c * scale] = wrong
        total_noise_pixels = 1
    return big, total_noise_pixels


def _place_objects(small: np.ndarray, big: np.ndarray) -> np.ndarray:
    small_h, small_w = small.shape
    big_h, big_w = big.shape
    gap = random.randint(2, 4)
    margin_a = random.randint(0, 2)
    margin_b = random.randint(0, 2)
    margin_c = random.randint(0, 2)
    margin_d = random.randint(0, 2)

    horizontal = random.random() < 0.5
    if horizontal:
        grid_h = max(small_h, big_h) + margin_a + margin_b
        grid_w = small_w + big_w + gap + margin_c + margin_d
        grid = np.zeros((grid_h, grid_w), dtype=int)
        top = random.randint(0, grid_h - max(small_h, big_h))
        if random.random() < 0.5:
            left_small = margin_c
            left_big = left_small + small_w + gap
            grid[top : top + small_h, left_small : left_small + small_w] = small
            grid[top : top + big_h, left_big : left_big + big_w] = big
        else:
            left_big = margin_c
            left_small = left_big + big_w + gap
            grid[top : top + big_h, left_big : left_big + big_w] = big
            grid[top : top + small_h, left_small : left_small + small_w] = small
    else:
        grid_h = small_h + big_h + gap + margin_a + margin_b
        grid_w = max(small_w, big_w) + margin_c + margin_d
        grid = np.zeros((grid_h, grid_w), dtype=int)
        left = random.randint(0, grid_w - max(small_w, big_w))
        if random.random() < 0.5:
            top_small = margin_a
            top_big = top_small + small_h + gap
            grid[top_small : top_small + small_h, left : left + small_w] = small
            grid[top_big : top_big + big_h, left : left + big_w] = big
        else:
            top_big = margin_a
            top_small = top_big + big_h + gap
            grid[top_big : top_big + big_h, left : left + big_w] = big
            grid[top_small : top_small + small_h, left : left + small_w] = small
    return grid


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    quarter_turns: int,
    mirror: bool,
    heavy_mask: np.ndarray,
) -> bool:
    try:
        details = _solve_details(inp)
    except AssertionError:
        return False

    if not np.array_equal(details["output"], expected):
        return False
    if details["missing_count"] <= 0:
        return False
    if details["transform_id"] == 0:
        return False
    if details["best_mismatch"] != int(heavy_mask.sum()):
        return False

    chosen_oriented_small = _apply_transform(details["small"], quarter_turns, mirror)
    chosen_heavy = _orient_mask(heavy_mask, quarter_turns, mirror)
    known = chosen_oriented_small != BLACK
    mismatch_mask = (details["majority"] != chosen_oriented_small) & known
    if not np.array_equal(mismatch_mask, chosen_heavy):
        return False

    if np.array_equal(details["recovered_full"], expected):
        return False

    same_frame_majority = details["majority"]
    if same_frame_majority.shape == expected.shape and np.array_equal(same_frame_majority, expected):
        return False

    candidates = details["candidates"]
    best = details["best_mismatch"]
    if sum(candidate["mismatches"] == best for candidate in candidates) != 1:
        return False
    if any(candidate["mismatches"] <= best for candidate in candidates if candidate["transform_id"] != details["transform_id"]):
        return False
    return True


def _generation_attempt() -> np.ndarray | None:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    palette_size = random.randint(3, 5)
    palette = tuple(random.sample(NON_BLACK, palette_size))
    full_small = _random_full_pattern(height, width, palette)

    transform_id = random.randint(1, len(TRANSFORMS) - 1)
    quarter_turns, mirror = TRANSFORMS[transform_id]
    if np.array_equal(_apply_transform(full_small, quarter_turns, mirror), full_small):
        return None

    all_coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(all_coords)
    heavy_count = 1 if height * width < 12 or random.random() < 0.5 else 2
    heavy_mask = np.zeros((height, width), dtype=bool)
    for r, c in all_coords[:heavy_count]:
        heavy_mask[r, c] = True

    missing_mask = _sample_missing_mask(height, width, heavy_mask)
    if int(np.sum(~missing_mask)) <= int(np.sum(heavy_mask)):
        return None

    input_small = np.array(full_small, copy=True)
    input_small[missing_mask] = BLACK

    scale = random.randint(2, 3)
    oriented_full = _apply_transform(full_small, quarter_turns, mirror)
    heavy_mask_oriented = _orient_mask(heavy_mask, quarter_turns, mirror)
    big, noise_pixels = _scaled_noisy_copy(oriented_full, scale, heavy_mask_oriented)
    if noise_pixels < 1:
        return None

    inp = _place_objects(input_small, big)
    if max(inp.shape) > 30:
        return None

    occ = inp != BLACK
    row_spans = _spans(occ.any(axis=1))
    col_spans = _spans(occ.any(axis=0))
    if not ((len(row_spans) == 2 and len(col_spans) == 1) or (len(row_spans) == 1 and len(col_spans) == 2)):
        return None

    if not _semantic_ok(inp, full_small, quarter_turns, mirror, heavy_mask):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _generation_attempt()
        if attempt is not None:
            return attempt
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
