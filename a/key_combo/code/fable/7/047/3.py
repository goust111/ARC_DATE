from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
RULER_COLORS = {GREY, RED, BLUE}
MOTIF_PALETTE = [3, 4, 6, 7, 8, 9]

GEN_MAX_ATTEMPTS = 300
GEN_TILE_H_RANGE = (2, 4)
GEN_TILE_W_RANGE = (2, 4)
GEN_SEGMENT_RANGE = (3, 6)
GEN_TOP_MARGIN_RANGE = (1, 5)
GEN_LEFT_MARGIN_RANGE = (1, 4)
GEN_RIGHT_MARGIN_RANGE = (1, 4)
GEN_BOTTOM_MARGIN_RANGE = (1, 5)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _iter_runs(mask: np.ndarray):
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            yield start, idx - 1
            start = None
    if start is not None:
        yield start, len(mask) - 1


def _flip_tile(tile: np.ndarray, vflip: bool, hflip: bool) -> np.ndarray:
    out = tile
    if vflip:
        out = np.flipud(out)
    if hflip:
        out = np.fliplr(out)
    return out


def _extract_motif(
    grid: np.ndarray, bar_r: int, c0: int, c1: int, bg: int
) -> tuple[int, int, np.ndarray] | None:
    top = int(bar_r) + 1
    if top >= grid.shape[0]:
        return None
    first = int(grid[top, c0])
    if first == bg or first in RULER_COLORS:
        return None

    motif_w = 0
    while c0 + motif_w < grid.shape[1]:
        value = int(grid[top, c0 + motif_w])
        if value == bg or value in RULER_COLORS:
            break
        motif_w += 1
    if motif_w == 0:
        return None

    bar_w = int(c1) - int(c0) + 1
    if bar_w % motif_w != 0:
        return None

    motif_h = 0
    while top + motif_h < grid.shape[0]:
        patch_row = grid[top + motif_h, c0 : c0 + motif_w]
        if patch_row.shape[0] != motif_w:
            break
        row_vals = set(map(int, patch_row))
        if bg in row_vals or row_vals & RULER_COLORS:
            break
        motif_h += 1
    if motif_h == 0:
        return None

    motif = np.array(grid[top : top + motif_h, c0 : c0 + motif_w], dtype=int)
    below_rest = grid[top : top + motif_h, c0 + motif_w : c1 + 1]
    if below_rest.size and np.any(below_rest != bg):
        return None
    return motif_h, motif_w, motif


def _find_ruler_and_motif(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    candidates = []
    ruler_mask_values = np.isin(g, list(RULER_COLORS))

    for r in range(g.shape[0] - 1):
        for c0, c1 in _iter_runs(ruler_mask_values[r]):
            if c1 - c0 + 1 < 2:
                continue
            motif_info = _extract_motif(g, r, c0, c1, bg)
            if motif_info is None:
                continue
            motif_h, motif_w, motif = motif_info
            candidates.append((r, c0, c1, motif_h, motif_w, motif))

    assert len(candidates) == 1, "expected exactly one horizontal ruler with one attached left motif"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the left motif under the full grey ruler while red and blue notches persistently toggle vertical and horizontal flips.

    Input structure:
    - A black background contains one horizontal grey ruler bar whose cells may be replaced by red or blue notch pixels.
    - A colored rectangular motif is attached directly below the ruler's leftmost segment, and the rest of the space below the ruler is empty.
    - The ruler length is an integer multiple of the motif width, with at most one notch pixel inside each segment.

    Rule:
    - Split the ruler into equal-width segments matching the attached motif width, then fill the rectangle below the ruler segment by segment.
    - Maintain two persistent state bits, both initially off: a red notch toggles vertical flip from its segment onward, and a blue notch toggles horizontal mirror from its segment onward.
    - Paste the motif in each segment using the current combination of flips, while leaving the ruler and notch pixels unchanged.

    Key insight:
    - The colored notches are cumulative state switches rather than local decorators, and the two flip axes evolve independently as the tiling proceeds left to right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bar_r, c0, c1, motif_h, motif_w, motif = _find_ruler_and_motif(g)
    bar_w = int(c1) - int(c0) + 1
    n_segments = bar_w // motif_w
    top = int(bar_r) + 1

    out = g.copy()
    vflip = False
    hflip = False
    for seg_idx in range(n_segments):
        seg_left = int(c0) + seg_idx * motif_w
        seg_right = seg_left + motif_w
        ruler_slice = g[bar_r, seg_left:seg_right]
        assert np.all(np.isin(ruler_slice, list(RULER_COLORS))), "ruler contains unexpected colors"

        notch_count = int(np.count_nonzero(ruler_slice != GREY))
        assert notch_count <= 1, "each ruler segment may contain at most one notch"
        if np.any(ruler_slice == RED):
            vflip = not vflip
        if np.any(ruler_slice == BLUE):
            hflip = not hflip

        out[top : top + motif_h, seg_left:seg_right] = _flip_tile(motif, vflip, hflip)

    return out


def _is_good_motif(motif: np.ndarray) -> bool:
    variants = {
        motif.tobytes(),
        np.flipud(motif).tobytes(),
        np.fliplr(motif).tobytes(),
        np.flipud(np.fliplr(motif)).tobytes(),
    }
    return len(variants) == 4 and len(set(map(int, motif.ravel()))) >= 2


def _generate_motif(height: int, width: int) -> np.ndarray:
    for _ in range(200):
        color_count = random.randint(2, min(4, height * width))
        colors = random.sample(MOTIF_PALETTE, color_count)
        motif = np.array(random.choices(colors, k=height * width), dtype=int).reshape(height, width)
        forced = random.sample(range(height * width), color_count)
        for idx, color in zip(forced, colors):
            motif.flat[idx] = int(color)
        if _is_good_motif(motif):
            return motif
    raise RuntimeError("failed to generate an asymmetric motif")


def _sample_notches(n_segments: int) -> list[str]:
    modes = ["both", "red_only", "blue_only"]
    weights = [0.6, 0.2, 0.2]
    for _ in range(200):
        mode = random.choices(modes, weights=weights, k=1)[0]
        if mode == "both":
            choices = random.choices(["none", "red", "blue"], weights=[0.4, 0.3, 0.3], k=n_segments)
            if "red" not in choices or "blue" not in choices:
                continue
        elif mode == "red_only":
            choices = random.choices(["none", "red"], weights=[0.45, 0.55], k=n_segments)
            if "red" not in choices:
                continue
        else:
            choices = random.choices(["none", "blue"], weights=[0.45, 0.55], k=n_segments)
            if "blue" not in choices:
                continue

        if "none" not in choices:
            choices[random.randrange(n_segments)] = "none"
        if all(kind == "none" for kind in choices):
            continue
        if not any(choices[i] != "none" and "none" in choices[i + 1 :] for i in range(n_segments - 1)):
            continue
        if random.random() < 0.35 and choices[0] == "none":
            first_kind = "red" if mode == "red_only" else "blue" if mode == "blue_only" else random.choice(["red", "blue"])
            swap_idx = next((i for i in range(1, n_segments) if choices[i] == first_kind), None)
            if swap_idx is not None:
                choices[0], choices[swap_idx] = choices[swap_idx], choices[0]
        return choices
    raise RuntimeError("failed to sample notch pattern")


def _render_expected(inp: np.ndarray, motif: np.ndarray, bar_r: int, left: int, notches: list[str]) -> np.ndarray:
    tile_h, tile_w = map(int, motif.shape)
    out = np.array(inp, copy=True)
    vflip = False
    hflip = False
    for seg_idx, notch in enumerate(notches):
        if notch == "red":
            vflip = not vflip
        elif notch == "blue":
            hflip = not hflip
        seg_left = int(left) + seg_idx * tile_w
        out[bar_r + 1 : bar_r + 1 + tile_h, seg_left : seg_left + tile_w] = _flip_tile(motif, vflip, hflip)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    tile_h = random.randint(*GEN_TILE_H_RANGE)
    tile_w = random.randint(*GEN_TILE_W_RANGE)
    top_margin = random.randint(*GEN_TOP_MARGIN_RANGE)
    left_margin = random.randint(*GEN_LEFT_MARGIN_RANGE)
    right_margin = random.randint(*GEN_RIGHT_MARGIN_RANGE)
    bottom_margin = random.randint(*GEN_BOTTOM_MARGIN_RANGE)

    width_budget = 30 - left_margin - right_margin
    max_segments = min(GEN_SEGMENT_RANGE[1], width_budget // tile_w)
    assert max_segments >= GEN_SEGMENT_RANGE[0]
    n_segments = random.randint(GEN_SEGMENT_RANGE[0], max_segments)
    bar_w = tile_w * n_segments

    height = top_margin + 1 + tile_h + bottom_margin
    width = left_margin + bar_w + right_margin
    assert height <= 30 and width <= 30

    bar_r = top_margin
    left = left_margin
    motif = _generate_motif(tile_h, tile_w)
    notches = _sample_notches(n_segments)

    inp = np.full((height, width), BG, dtype=int)
    inp[bar_r, left : left + bar_w] = GREY

    for seg_idx, notch in enumerate(notches):
        if notch == "none":
            continue
        seg_left = left + seg_idx * tile_w
        notch_c = seg_left + random.randrange(tile_w)
        inp[bar_r, notch_c] = RED if notch == "red" else BLUE

    inp[bar_r + 1 : bar_r + 1 + tile_h, left : left + tile_w] = motif
    expected = _render_expected(inp, motif, bar_r, left, notches)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp, expected = _generate_case()
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate valid input")
