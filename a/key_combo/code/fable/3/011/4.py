from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

PALETTE = tuple(range(10))
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    """Inclusive bbox for an (N,2) coords array in (r,c) order."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop a grid to an inclusive bbox."""
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components of a boolean mask, as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in neigh:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))

    return out


def _bbox_shape(bbox: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _find_scaled_patterns(grid: np.ndarray) -> tuple[int, BBox, BBox, int]:
    bg = get_bg(grid)
    comps = _components(grid != bg)
    assert len(comps) == 2, f"expected exactly two separated patterns, got {len(comps)}"

    infos: list[tuple[int, int, BBox]] = []
    for coords in comps:
        bbox = bbox_coords(coords)
        h, w = _bbox_shape(bbox)
        infos.append((h * w, int(coords.shape[0]), bbox))

    infos.sort(key=lambda item: (item[0], item[1]))
    small_bbox = infos[0][2]
    big_bbox = infos[1][2]

    sh, sw = _bbox_shape(small_bbox)
    bh, bw = _bbox_shape(big_bbox)
    assert bh % sh == 0 and bw % sw == 0, "bbox sizes must be exact integer multiples"
    scale_r = bh // sh
    scale_c = bw // sw
    assert scale_r == scale_c and scale_r > 1, "expected a unique integer upscale factor"
    return bg, small_bbox, big_bbox, int(scale_r)


def _is_uniform(block: np.ndarray) -> bool:
    flat = block.reshape(-1)
    return bool(np.all(flat == flat[0]))


def _upscale(patch: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(patch, dtype=int), scale, axis=0), scale, axis=1)


def _repair_pair(small: np.ndarray, big: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    repaired_small = np.array(small, dtype=int, copy=True)
    repaired_big = np.array(big, dtype=int, copy=True)

    for r in range(small.shape[0]):
        for c in range(small.shape[1]):
            block = big[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            if _is_uniform(block):
                repaired_small[r, c] = int(block[0, 0])

    for r in range(small.shape[0]):
        for c in range(small.shape[1]):
            block = big[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            if not _is_uniform(block):
                repaired_big[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = repaired_small[r, c]

    return repaired_small, repaired_big


def main(grid: np.ndarray) -> np.ndarray:
    """Repair both scaled copies by trusting uniform large blocks and using the small pixels where large blocks are mixed.

    Input structure:
    - A background grid contains exactly two separated multicolor patterns, one small and one large.
    - Their bounding boxes have the same aspect ratio, and the large bbox dimensions are an integer scale multiple of the small one.
    - The large pattern should be the exact pixel-upscaled image of the small pattern, but some small pixels are recolored and some large scale-blocks are partially recolored into mixed-color blocks.

    Rule:
    - Infer the unique scale factor from the two bounding-box sizes and align the large pattern into s-by-s blocks over the small pattern's pixels.
    - If a large block is uniform, treat it as trustworthy and overwrite the corresponding small pixel with that block color.
    - If a large block is mixed, treat the block as damaged and repaint the whole block with the corresponding small pixel's color.
    - Leave the background and every other pixel unchanged.

    Key insight:
    - Reliability belongs to each corresponding pair, not to one whole image: uniform enlarged blocks certify the small pixel, while mixed enlarged blocks must be restored from the untouched small counterpart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, small_bbox, big_bbox, scale = _find_scaled_patterns(g)
    small = crop_bbox(g, small_bbox)
    big = crop_bbox(g, big_bbox)
    assert big.shape == (small.shape[0] * scale, small.shape[1] * scale)

    repaired_small, repaired_big = _repair_pair(small, big, scale)

    out = np.array(g, copy=True)
    _paste(out, repaired_small, small_bbox[0], small_bbox[1])
    _paste(out, repaired_big, big_bbox[0], big_bbox[1])
    return out


def _choose_color(colors: Iterable[int], banned: Iterable[int]) -> int:
    banned_set = {int(c) for c in banned}
    options = [int(c) for c in colors if int(c) not in banned_set]
    assert options
    return int(random.choice(options))


def _make_pattern(h: int, w: int, colors: list[int]) -> np.ndarray:
    for _ in range(100):
        patch = np.full((h, w), random.choice(colors), dtype=int)
        op_count = random.randint(4, 7)
        for _ in range(op_count):
            color = random.choice(colors)
            mode = random.choice(("rect", "row", "col", "point"))
            if mode == "rect":
                r0 = random.randint(0, h - 1)
                r1 = random.randint(r0, h - 1)
                c0 = random.randint(0, w - 1)
                c1 = random.randint(c0, w - 1)
                patch[r0 : r1 + 1, c0 : c1 + 1] = color
            elif mode == "row":
                r = random.randint(0, h - 1)
                c0 = random.randint(0, w - 1)
                c1 = random.randint(c0, w - 1)
                patch[r, c0 : c1 + 1] = color
            elif mode == "col":
                c = random.randint(0, w - 1)
                r0 = random.randint(0, h - 1)
                r1 = random.randint(r0, h - 1)
                patch[r0 : r1 + 1, c] = color
            else:
                patch[random.randint(0, h - 1), random.randint(0, w - 1)] = color

        if len(np.unique(patch)) < 2:
            continue
        transitions = int(np.count_nonzero(patch[:, 1:] != patch[:, :-1]))
        transitions += int(np.count_nonzero(patch[1:, :] != patch[:-1, :]))
        if transitions < max(3, (h + w) // 2):
            continue
        return patch

    raise RuntimeError("failed to make a non-degenerate pattern")


def _position_scene(
    small_clean: np.ndarray,
    small_input: np.ndarray,
    big_clean: np.ndarray,
    big_input: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    sh, sw = small_clean.shape
    bh, bw = big_clean.shape

    for _ in range(80):
        gap = random.randint(2, 5)
        border = random.randint(1, 3)
        orientation = random.choice(("horizontal", "vertical"))

        if orientation == "horizontal":
            min_h = max(sh, bh) + 2 * border
            min_w = sw + bw + gap + 2 * border
            if min_h > 30 or min_w > 30:
                continue
            h = random.randint(min_h, min(30, min_h + 6))
            w = random.randint(min_w, min(30, min_w + 6))
            left_big = random.choice((True, False))
            if left_big:
                bc = random.randint(border, w - border - (bw + gap + sw))
                sc = random.randint(bc + bw + gap, w - border - sw)
            else:
                sc = random.randint(border, w - border - (sw + gap + bw))
                bc = random.randint(sc + sw + gap, w - border - bw)
            br = random.randint(border, h - border - bh)
            sr = random.randint(border, h - border - sh)
        else:
            min_h = sh + bh + gap + 2 * border
            min_w = max(sw, bw) + 2 * border
            if min_h > 30 or min_w > 30:
                continue
            h = random.randint(min_h, min(30, min_h + 6))
            w = random.randint(min_w, min(30, min_w + 6))
            top_big = random.choice((True, False))
            if top_big:
                br = random.randint(border, h - border - (bh + gap + sh))
                sr = random.randint(br + bh + gap, h - border - sh)
            else:
                sr = random.randint(border, h - border - (sh + gap + bh))
                br = random.randint(sr + sh + gap, h - border - bh)
            bc = random.randint(border, w - border - bw)
            sc = random.randint(border, w - border - sw)

        clean = np.full((h, w), bg, dtype=int)
        inp = np.full((h, w), bg, dtype=int)
        _paste(clean, big_clean, br, bc)
        _paste(clean, small_clean, sr, sc)
        _paste(inp, big_input, br, bc)
        _paste(inp, small_input, sr, sc)

        counts = Counter(map(int, inp.ravel()))
        clean_counts = Counter(map(int, clean.ravel()))
        if counts[bg] <= max(count for color, count in counts.items() if color != bg):
            continue
        if clean_counts[bg] <= max(count for color, count in clean_counts.items() if color != bg):
            continue
        return inp, clean

    raise RuntimeError("failed to place the two patterns without degeneracy")


def _corrupt_case(clean_small: np.ndarray, scale: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = clean_small.shape
    clean_big = _upscale(clean_small, scale)
    input_small = np.array(clean_small, copy=True)
    input_big = np.array(clean_big, copy=True)

    positions = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(positions)

    max_branch = min(3, max(1, (h * w) // 3))
    small_count = random.randint(1, max_branch)
    big_count = random.randint(1, max_branch)
    while small_count + big_count > len(positions) - 1:
        if big_count > 1:
            big_count -= 1
        elif small_count > 1:
            small_count -= 1
        else:
            break

    small_positions = positions[:small_count]
    big_positions = positions[small_count : small_count + big_count]

    for r, c in small_positions:
        input_small[r, c] = _choose_color(colors, [clean_small[r, c]])

    majority_index = random.randrange(len(big_positions))
    for idx, (r, c) in enumerate(big_positions):
        block = input_big[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
        true_color = int(clean_small[r, c])
        wrong_total = scale * scale
        if idx == majority_index:
            wrong_count = random.randint(wrong_total // 2 + 1, wrong_total - 1)
        else:
            wrong_count = random.randint(1, wrong_total - 1)

        cells = [(rr, cc) for rr in range(scale) for cc in range(scale)]
        bad_cells = random.sample(cells, wrong_count)
        wrong_colors = [int(col) for col in colors if int(col) != true_color]
        if len(wrong_colors) >= 2 and wrong_count >= 2 and random.random() < 0.35:
            split = random.randint(1, wrong_count - 1)
            color_a, color_b = random.sample(wrong_colors, 2)
            for rr, cc in bad_cells[:split]:
                block[rr, cc] = color_a
            for rr, cc in bad_cells[split:]:
                block[rr, cc] = color_b
        else:
            wrong_color = random.choice(wrong_colors)
            for rr, cc in bad_cells:
                block[rr, cc] = wrong_color

    assert any(input_small[r, c] != clean_small[r, c] for r, c in small_positions)
    assert any(not _is_uniform(input_big[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]) for r, c in big_positions)
    return input_small, input_big


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    scale = random.choice((2, 2, 2, 3, 3, 4))
    if scale == 2:
        h = random.randint(3, 7)
        w = random.randint(3, 7)
    elif scale == 3:
        h = random.randint(3, 6)
        w = random.randint(3, 6)
    else:
        h = random.randint(3, 5)
        w = random.randint(3, 5)

    color_count = random.randint(3, min(5, len(PALETTE) - 1))
    colors = random.sample([c for c in PALETTE if c != bg], color_count)
    clean_small = _make_pattern(h, w, colors)
    input_small, input_big = _corrupt_case(clean_small, scale, colors)
    clean_big = _upscale(clean_small, scale)

    try:
        inp, clean = _position_scene(clean_small, input_small, clean_big, input_big, bg)
    except RuntimeError:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, clean):
        return None
    if not np.array_equal(main(clean), clean):
        return None

    bg2, small_bbox, big_bbox, inferred_scale = _find_scaled_patterns(inp)
    if bg2 != bg or inferred_scale != scale:
        return None
    small_in = crop_bbox(inp, small_bbox)
    big_in = crop_bbox(inp, big_bbox)
    small_out = crop_bbox(solved, small_bbox)
    big_out = crop_bbox(solved, big_bbox)

    seen_uniform_mismatch = False
    seen_mixed_block = False
    seen_majority_trap = False
    for r in range(small_in.shape[0]):
        for c in range(small_in.shape[1]):
            block_in = big_in[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            block_out = big_out[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            if _is_uniform(block_in):
                if int(small_in[r, c]) != int(block_in[0, 0]):
                    seen_uniform_mismatch = True
                if int(small_out[r, c]) != int(block_in[0, 0]):
                    return None
            else:
                seen_mixed_block = True
                if not _is_uniform(block_out):
                    return None
                repaired_color = int(block_out[0, 0])
                if repaired_color != int(small_out[r, c]):
                    return None
                matching = int(np.count_nonzero(block_in == repaired_color))
                if matching < (scale * scale) / 2:
                    seen_majority_trap = True

    if not (seen_uniform_mismatch and seen_mixed_block and seen_majority_trap):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
