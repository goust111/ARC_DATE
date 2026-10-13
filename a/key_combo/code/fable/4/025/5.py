from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


BG = 0
GEN_RETRIES = 400


@dataclass(frozen=True)
class ObjectInfo:
    color: int
    bbox: tuple[int, int, int, int]
    crop_mask: np.ndarray
    orientation: str
    seam: int
    symmetric: bool


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0, "expected a non-empty object"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _colors_present(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BG]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    height, width = mask.shape

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _analyze_object(grid: np.ndarray, color: int) -> ObjectInfo:
    mask = (grid == int(color))
    r0, c0, r1, c1 = _bbox(mask)
    crop_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    height, width = crop_mask.shape

    empty_rows = [r for r in range(1, height - 1) if not bool(crop_mask[r].any())]
    empty_cols = [c for c in range(1, width - 1) if not bool(crop_mask[:, c].any())]
    assert (len(empty_rows), len(empty_cols)) in {(1, 0), (0, 1)}, "expected exactly one internal blank seam"

    if empty_rows:
        seam = int(empty_rows[0])
        top = crop_mask[:seam, :]
        bottom = crop_mask[seam + 1 :, :]
        assert top.shape == bottom.shape, "expected a balanced horizontal seam"
        symmetric = bool(np.array_equal(top, bottom[::-1, :]))
        orientation = "h"
    else:
        seam = int(empty_cols[0])
        left = crop_mask[:, :seam]
        right = crop_mask[:, seam + 1 :]
        assert left.shape == right.shape, "expected a balanced vertical seam"
        symmetric = bool(np.array_equal(left, right[:, ::-1]))
        orientation = "v"

    return ObjectInfo(
        color=int(color),
        bbox=(r0, c0, r1, c1),
        crop_mask=crop_mask.copy(),
        orientation=orientation,
        seam=seam,
        symmetric=symmetric,
    )


def _erase_object(out: np.ndarray, info: ObjectInfo) -> None:
    r0, c0, r1, c1 = info.bbox
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch[info.crop_mask] = BG


def _complete_object(out: np.ndarray, info: ObjectInfo, donor_color: int) -> None:
    r0, c0, r1, c1 = info.bbox
    crop = info.crop_mask
    height, width = crop.shape

    if info.orientation == "h":
        seam = info.seam
        for r in range(seam):
            mr = height - 1 - r
            for c in range(width):
                top_has = bool(crop[r, c])
                bottom_has = bool(crop[mr, c])
                if top_has and not bottom_has:
                    out[r0 + mr, c0 + c] = int(donor_color)
                elif bottom_has and not top_has:
                    out[r0 + r, c0 + c] = int(donor_color)
    else:
        seam = info.seam
        for c in range(seam):
            mc = width - 1 - c
            for r in range(height):
                left_has = bool(crop[r, c])
                right_has = bool(crop[r, mc])
                if left_has and not right_has:
                    out[r0 + r, c0 + mc] = int(donor_color)
                elif right_has and not left_has:
                    out[r0 + r, c0 + c] = int(donor_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete split monochrome objects by routing missing mirror pixels to the donor color of the same seam orientation.

    Input structure:
    - Black background with several non-touching monochrome objects, all in distinct colors.
    - Each color's bounding box contains exactly one internal blank row or one internal blank column that splits the object into two halves.
    - Among the horizontal-seam objects and among the vertical-seam objects, exactly one object is perfectly mirror-symmetric about its own blank seam.

    Rule:
    - For each color, detect its bounding box, locate the unique blank seam, and classify it as horizontal or vertical.
    - The unique symmetric object of each seam orientation is that orientation's color donor; remove both donors from the output.
    - For every remaining object, compare mirrored positions across its seam and fill any empty counterpart with the donor color of the same orientation, while keeping the seam itself black.

    Key insight:
    - The two symmetric objects are not completed; they act as orientation-specific palettes, so the seam direction determines which donor color should repair each asymmetric object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    infos = [_analyze_object(g, color) for color in _colors_present(g)]
    donors: dict[str, ObjectInfo] = {}
    for orientation in ("h", "v"):
        matches = [info for info in infos if info.orientation == orientation and info.symmetric]
        assert len(matches) == 1, f"expected one symmetric donor for orientation {orientation}"
        donors[orientation] = matches[0]

    out = g.copy()
    donor_colors = {info.color for info in donors.values()}
    for info in donors.values():
        _erase_object(out, info)

    for info in infos:
        if info.color in donor_colors:
            continue
        _complete_object(out, info, donors[info.orientation].color)

    return out


def _intervals_to_mask(intervals: Sequence[tuple[int, int]], width: int) -> np.ndarray:
    mask = np.zeros((len(intervals), width), dtype=bool)
    for r, (start, end) in enumerate(intervals):
        mask[r, int(start) : int(end) + 1] = True
    return mask


def _sample_full_half(half_h: int, width: int, rng: np.random.Generator) -> np.ndarray:
    min_len = 2 if width > 2 else 1
    intervals: list[list[int]] = []

    length = int(rng.integers(min_len, width + 1))
    start = int(rng.integers(0, width - length + 1))
    intervals.append([start, start + length - 1])

    for _ in range(1, half_h):
        prev_start, prev_end = intervals[-1]
        overlap = int(rng.integers(prev_start, prev_end + 1))
        length = int(rng.integers(min_len, width + 1))
        lo = max(0, overlap - length + 1)
        hi = min(overlap, width - length)
        start = int(rng.integers(lo, hi + 1))
        intervals.append([start, start + length - 1])

    intervals[int(rng.integers(half_h))][0] = 0
    intervals[int(rng.integers(half_h))][1] = width - 1

    for row in range(half_h):
        if rng.random() < 0.5:
            intervals[row][0] = int(rng.integers(0, intervals[row][0] + 1))
        if rng.random() < 0.5:
            intervals[row][1] = int(rng.integers(intervals[row][1], width))

    mask = _intervals_to_mask([(start, end) for start, end in intervals], width)
    assert np.all(mask.any(axis=1)), "every row must contain color"
    assert np.all(mask.any(axis=0)), "the half shape must span every column"
    assert _is_connected(mask), "expected the full half to be connected"
    return mask


def _sample_subset_half(full_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    half_h, width = full_mask.shape
    intervals: list[tuple[int, int]] = []
    candidates: list[int] = []
    for r in range(half_h):
        cols = np.flatnonzero(full_mask[r])
        start, end = int(cols[0]), int(cols[-1])
        intervals.append((start, end))
        if end > start:
            candidates.append(r)

    assert candidates, "expected at least one row that can be shrunk"

    for _ in range(80):
        subset = np.zeros((half_h, width), dtype=bool)
        strict = False
        for r, (start, end) in enumerate(intervals):
            full_len = end - start + 1
            sub_len = int(rng.integers(1, full_len + 1))
            sub_start = int(rng.integers(start, end - sub_len + 2))
            subset[r, sub_start : sub_start + sub_len] = True
            if sub_len < full_len:
                strict = True

        if not strict:
            row = int(candidates[int(rng.integers(len(candidates)))])
            cols = np.flatnonzero(subset[row])
            if rng.random() < 0.5:
                subset[row, int(cols[0])] = False
            else:
                subset[row, int(cols[-1])] = False

        if _is_connected(subset):
            return subset

    raise RuntimeError("failed to sample a connected asymmetric half")


def _build_horizontal_object(
    rng: np.random.Generator,
    color: int,
    donor_color: int | None,
    *,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray]:
    half_h = int(rng.integers(2, 5))
    width = int(rng.integers(3, 8))
    full_half = _sample_full_half(half_h, width, rng)

    height = 2 * half_h + 1
    inp = np.zeros((height, width), dtype=int)

    if symmetric:
        top_half = full_half
        bottom_half = full_half
    else:
        subset_half = _sample_subset_half(full_half, rng)
        if bool(rng.integers(0, 2)):
            top_half = full_half
            bottom_half = subset_half
            fill_top_side = False
        else:
            top_half = subset_half
            bottom_half = full_half
            fill_top_side = True

    for r in range(half_h):
        inp[r, top_half[r]] = int(color)
        inp[height - 1 - r, bottom_half[r]] = int(color)

    if symmetric:
        return inp, np.zeros_like(inp)

    assert donor_color is not None
    out = inp.copy()
    missing = full_half & ~subset_half
    if fill_top_side:
        for r in range(half_h):
            out[r, missing[r]] = int(donor_color)
    else:
        for r in range(half_h):
            out[height - 1 - r, missing[r]] = int(donor_color)
    return inp, out


def _build_vertical_object(
    rng: np.random.Generator,
    color: int,
    donor_color: int | None,
    *,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray]:
    inp, out = _build_horizontal_object(rng, color, donor_color, symmetric=symmetric)
    return inp.T.copy(), out.T.copy()


def _choose_canvas_size(
    patches: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    padded_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in patches)
    min_h = max(patch.shape[0] + 2 for patch in patches)
    min_w = max(patch.shape[1] + 2 for patch in patches)
    side_floor = max(min_h, min_w, int(math.ceil(math.sqrt(padded_area))) - 1, 14)
    if side_floor > 30:
        return None
    height = int(rng.integers(side_floor, 31))
    width = int(rng.integers(side_floor, 31))
    return height, width


def _place_patches(
    input_patches: Sequence[np.ndarray],
    output_patches: Sequence[np.ndarray],
    height: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    order = sorted(
        range(len(input_patches)),
        key=lambda idx: input_patches[idx].shape[0] * input_patches[idx].shape[1],
        reverse=True,
    )

    for idx in order:
        input_patch = input_patches[idx]
        output_patch = output_patches[idx]
        patch_h, patch_w = input_patch.shape
        candidates: list[tuple[int, int]] = []

        for r0 in range(height - patch_h + 1):
            for c0 in range(width - patch_w + 1):
                rs = max(0, r0 - 1)
                re = min(height, r0 + patch_h + 1)
                cs = max(0, c0 - 1)
                ce = min(width, c0 + patch_w + 1)
                if bool(blocked[rs:re, cs:ce].any()):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        input_mask = input_patch != BG
        output_mask = output_patch != BG
        inp[r0 : r0 + patch_h, c0 : c0 + patch_w][input_mask] = input_patch[input_mask]
        out[r0 : r0 + patch_h, c0 : c0 + patch_w][output_mask] = output_patch[output_mask]
        blocked[max(0, r0 - 1) : min(height, r0 + patch_h + 1), max(0, c0 - 1) : min(width, c0 + patch_w + 1)] = True

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    horizontal_count = int(rng.integers(2, 5))
    vertical_count = int(rng.integers(2, 5))
    total_count = horizontal_count + vertical_count
    colors = [int(color) for color in rng.choice(np.arange(1, 10), size=total_count, replace=False)]

    donor_h_color = colors[0]
    donor_v_color = colors[1]
    cursor = 2

    input_patches: list[np.ndarray] = []
    output_patches: list[np.ndarray] = []

    donor_in, donor_out = _build_horizontal_object(rng, donor_h_color, None, symmetric=True)
    input_patches.append(donor_in)
    output_patches.append(donor_out)
    for _ in range(horizontal_count - 1):
        color = colors[cursor]
        cursor += 1
        inp, out = _build_horizontal_object(rng, color, donor_h_color, symmetric=False)
        input_patches.append(inp)
        output_patches.append(out)

    donor_in, donor_out = _build_vertical_object(rng, donor_v_color, None, symmetric=True)
    input_patches.append(donor_in)
    output_patches.append(donor_out)
    for _ in range(vertical_count - 1):
        color = colors[cursor]
        cursor += 1
        inp, out = _build_vertical_object(rng, color, donor_v_color, symmetric=False)
        input_patches.append(inp)
        output_patches.append(out)

    order = rng.permutation(len(input_patches))
    input_patches = [input_patches[int(idx)] for idx in order]
    output_patches = [output_patches[int(idx)] for idx in order]

    size = _choose_canvas_size(input_patches, rng)
    if size is None:
        return None
    height, width = size

    placed = _place_patches(input_patches, output_patches, height, width, rng)
    if placed is None:
        return None
    inp, expected = placed

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
