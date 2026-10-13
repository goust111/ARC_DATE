import hashlib
from typing import Iterable

import numpy as np


BG = 0
FRAME = 2
PATTERN_COLORS = tuple(c for c in range(1, 10) if c != FRAME)
MAX_ATTEMPTS = 800

_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale2(sprite: np.ndarray) -> np.ndarray:
    return np.kron(sprite, np.ones((2, 2), dtype=int))


def _mirror_quadrants(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return np.block(
        [
            [np.rot90(g, 2), np.flipud(g)],
            [np.fliplr(g), g],
        ]
    )


def _frame_bbox_from_mosaic(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == FRAME
    box = _bbox(mask)
    assert box is not None, "expected red frame pixels after mirroring"
    r0, c0, r1, c1 = box
    frame = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(frame, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    assert np.array_equal(frame, expected), "red pixels must form one hollow rectangle"
    assert int(mask.sum()) == int(expected.sum()), "no extra red pixels may lie outside the frame bbox"
    return box


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-stitch a truncated lower-right frame fragment, crop to the recovered frame, and center a 2x enlargement of the enclosed pattern.

    Input structure:
    - A black background contains only the lower-right fragment of a red rectangular border.
    - The visible red fragment is an L-shape: a horizontal segment and a vertical segment that meet at the fragment's corner.
    - A small non-black, non-red seed pattern lies inside that visible corner region.

    Rule:
    - Treat the input as the lower-right quadrant of a 2x2 mirror mosaic.
    - Fill the other quadrants with, respectively, the 180-degree rotation, the up-down reflection, and the left-right reflection of the input.
    - In the stitched mosaic, find the complete red rectangular frame and crop to its bounding box.
    - Inside the crop, take the tight bounding box of all non-black, non-red pixels, enlarge it by a factor of 2, clear the frame interior to black, and place the enlarged pattern at the center while preserving the red border.

    Key insight:
    - The red frame and interior pattern are both incomplete in the input; only mirror stitching reveals the closed frame and the full four-way-symmetric seed that must then be cropped and enlarged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mosaic = _mirror_quadrants(g)
    frame_box = _frame_bbox_from_mosaic(mosaic)
    cropped = _crop_bbox(mosaic, frame_box)

    pattern_mask = (cropped != BG) & (cropped != FRAME)
    pattern_box = _bbox(pattern_mask)
    assert pattern_box is not None, "expected a colored interior pattern"
    pattern = _crop_bbox(cropped, pattern_box)
    scaled = _scale2(pattern)

    out = cropped.copy()
    out[1:-1, 1:-1] = BG

    inner_h = out.shape[0] - 2
    inner_w = out.shape[1] - 2
    scaled_h, scaled_w = scaled.shape
    assert scaled_h <= inner_h and scaled_w <= inner_w, "scaled pattern must fit inside the frame"
    assert (inner_h - scaled_h) % 2 == 0 and (inner_w - scaled_w) % 2 == 0, "pattern must center unambiguously"

    top = 1 + (inner_h - scaled_h) // 2
    left = 1 + (inner_w - scaled_w) // 2
    out[top : top + scaled_h, left : left + scaled_w] = scaled
    return out


def _make_seed_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    cells = {(height - 1, width - 1)}
    choices = [(r, c) for r in range(height) for c in range(width) if (r, c) != (height - 1, width - 1)]
    np.random.shuffle(choices)
    for cell in choices[: count - 1]:
        cells.add(cell)
    return sorted(cells)


def _validate_generated_input(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    assert 6 <= h <= 10 and 6 <= w <= 10

    red = {tuple(map(int, rc)) for rc in np.argwhere(grid == FRAME)}
    rb = h - 2
    cb = w - 2
    expected_red = {(rb, c) for c in range(cb + 1)} | {(r, cb) for r in range(rb + 1)}
    assert red == expected_red, "input must expose exactly one truncated L-shaped frame fragment"

    seed_mask = (grid != BG) & (grid != FRAME)
    seed_count = int(seed_mask.sum())
    assert 1 <= seed_count <= 3, "seed pattern must use 1 to 3 visible pixels"

    seed_box = _bbox(seed_mask)
    assert seed_box is not None
    sr0, sc0, sr1, sc1 = seed_box
    assert sr1 < rb and sc1 < cb, "seed pixels must stay inside the visible frame interior"

    mosaic = _mirror_quadrants(grid)
    frame_box = _frame_bbox_from_mosaic(mosaic)
    assert frame_box == (1, 1, 2 * h - 2, 2 * w - 2), "cropping must remove the mirrored black margin"

    cropped = _crop_bbox(mosaic, frame_box)
    pattern_mask = (cropped != BG) & (cropped != FRAME)
    pattern_box = _bbox(pattern_mask)
    assert pattern_box is not None
    pattern = _crop_bbox(cropped, pattern_box)
    scaled = _scale2(pattern)
    inner_h = cropped.shape[0] - 2
    inner_w = cropped.shape[1] - 2
    assert scaled.shape[0] <= inner_h and scaled.shape[1] <= inner_w
    assert scaled.shape != (inner_h, inner_w), "scaled pattern must not simply replace the whole interior"
    assert (inner_h - scaled.shape[0]) % 2 == 0 and (inner_w - scaled.shape[1]) % 2 == 0

    expected = cropped.copy()
    expected[1:-1, 1:-1] = BG
    top = 1 + (inner_h - scaled.shape[0]) // 2
    left = 1 + (inner_w - scaled.shape[1]) // 2
    expected[top : top + scaled.shape[0], left : left + scaled.shape[1]] = scaled

    out = main(grid)
    assert np.array_equal(out, expected), "solver output must match the independently reconstructed semantics"
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(6, 11))
    w = int(np.random.randint(6, 11))
    rb = h - 2
    cb = w - 2

    max_total_h = rb // 2
    max_total_w = cb // 2
    if max_total_h < 1 or max_total_w < 1:
        return None

    total_h = int(np.random.randint(1, max_total_h + 1))
    total_w = int(np.random.randint(1, max_total_w + 1))
    if total_h == max_total_h and total_w == max_total_w:
        return None

    qh = int(np.random.randint(1, min(3, total_h) + 1))
    qw = int(np.random.randint(1, min(3, total_w) + 1))
    offset_r = total_h - qh
    offset_c = total_w - qw

    cell_cap = min(3, qh * qw)
    if cell_cap < 1:
        return None
    count = int(np.random.randint(1, cell_cap + 1))
    seed_cells = _make_seed_cells(qh, qw, count)

    grid = np.zeros((h, w), dtype=int)
    grid[rb, : cb + 1] = FRAME
    grid[: rb + 1, cb] = FRAME

    color = int(np.random.choice(PATTERN_COLORS))
    for dr, dc in seed_cells:
        grid[offset_r + dr, offset_c + dc] = color

    try:
        out = _validate_generated_input(grid)
    except AssertionError:
        return None

    input_sig = _signature(grid)
    output_sig = _signature(out)
    if input_sig in _SEEN_INPUTS or output_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(input_sig)
    _SEEN_OUTPUTS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
