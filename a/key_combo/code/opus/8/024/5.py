from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


GREY = Color.GREY
CONTROL_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)


def _solid_3x3_blocks(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    comps = components_mask(grid == GREY, connectivity=4)
    blocks: list[tuple[int, int]] = []
    for coords in comps:
        if int(coords.shape[0]) != 9:
            raise AssertionError("every grey component must be a solid 3x3 block")
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r1 - r0), int(c1 - c0)) != (2, 2):
            raise AssertionError("every grey component must be exactly 3x3")
        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        if not np.all(patch == GREY):
            raise AssertionError("grey generator blocks must be solid")
        blocks.append((int(r0) + 1, int(c0) + 1))
    if not 2 <= len(blocks) <= 4:
        raise AssertionError("expected 2 to 4 grey generator blocks")
    return sorted(blocks)


def _controls(grid: np.ndarray, bg: int) -> dict[int, int]:
    H, W = grid.shape
    controls: dict[int, int] = {}
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color in (bg, GREY):
                continue
            if r != 0:
                raise AssertionError("control pixels must appear only in the top row")
            if color not in CONTROL_COLORS:
                raise AssertionError("unexpected control color")
            if c in controls:
                raise AssertionError("expected one control pixel per column")
            controls[c] = color
    if len(controls) < 2:
        raise AssertionError("expected multiple top-row control pixels")
    if len(set(controls.values())) != len(controls):
        raise AssertionError("control colors must be unique")
    return controls


def _block_to_color(blocks: list[tuple[int, int]], controls: dict[int, int]) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for center_r, center_c in blocks:
        matches = [
            color
            for col, color in controls.items()
            if center_c - 1 <= int(col) <= center_c + 1
        ]
        if len(matches) != 1:
            raise AssertionError("each block span must contain exactly one control column")
        mapping[(center_r, center_c)] = int(matches[0])
    if len(mapping) != len(controls):
        raise AssertionError("expected one control for each block")
    return mapping


def _cross_intersections(blocks: list[tuple[int, int]]) -> set[tuple[int, int]]:
    hits: set[tuple[int, int]] = set()
    for i, (row_i, _col_i) in enumerate(blocks):
        for j, (_row_j, col_j) in enumerate(blocks):
            if i != j:
                hits.add((int(row_i), int(col_j)))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each grey 3x3 block into a cross, clear cross-to-cross intersections, then recolor each cross by the top-row control column inside that block's original 3-column span.

    Input structure:
    - The background is black.
    - There are 2 to 4 grey solid 3x3 blocks whose center rows are all distinct and whose center columns are all distinct.
    - The top row contains unique single-pixel control markers colored blue, red, green, or yellow.
    - Each block's original column span [center_col-1, center_col+1] contains exactly one control marker column.

    Rule:
    - Copy the input grid.
    - For each grey 3x3 block, draw a full-width horizontal arm on the block center row and a full-height vertical arm on the block center column.
    - Any pixel that is the center-row/center-column crossing of two different blocks is reset to background.
    - Recolor each block and all of its generated arm pixels with the color of the unique top-row control marker whose column lies inside that block's original 3-column span.
    - Preserve the top-row control pixels themselves.

    Key insight:
    - The control color is matched to each cross by containment in the source block's narrow original 3-column span, not by the expanded full-width cross.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    bg = int(get_bg(g))
    if bg != Color.BLACK:
        raise AssertionError("expected black background")

    blocks = _solid_3x3_blocks(g, bg)
    rows = [r for r, _c in blocks]
    cols = [c for _r, c in blocks]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        raise AssertionError("block centers must use distinct rows and columns")
    for i, (r0, c0) in enumerate(blocks):
        for r1, c1 in blocks[i + 1 :]:
            if abs(r0 - r1) < 4 or abs(c0 - c1) < 4:
                raise AssertionError("blocks must be sufficiently separated")

    controls = _controls(g, bg)
    if len(controls) != len(blocks):
        raise AssertionError("expected one unique control per block")
    block_color = _block_to_color(blocks, controls)
    intersections = _cross_intersections(blocks)

    out = g.copy()
    for center_r, center_c in blocks:
        out[center_r, :] = GREY
        out[:, center_c] = GREY

    for row, col in intersections:
        out[row, col] = bg

    for center_r, center_c in blocks:
        color = block_color[(center_r, center_c)]
        out[center_r, :] = color
        out[:, center_c] = color
        out[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = color

    for row, col in intersections:
        out[row, col] = bg

    for col, color in controls.items():
        out[0, col] = color

    return out


GEN_HEIGHT_RANGE = (16, 22)
GEN_WIDTH_RANGE = (16, 22)
GEN_BLOCK_COUNTS = (2, 3, 4)
GEN_BLOCK_COUNT_WEIGHTS = (2, 3, 3)
GEN_MIN_CENTER_SEP = 4
GEN_ATTEMPTS = 400


def _choose_centers(length: int, count: int) -> list[int] | None:
    candidates = list(range(2, length - 2))
    random.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - prev) >= GEN_MIN_CENTER_SEP for prev in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                random.shuffle(chosen)
                return chosen
    return None


def _choose_offsets(count: int) -> list[int]:
    offsets = [random.choice((-1, 0, 1)) for _ in range(count)]
    if all(offset == 0 for offset in offsets):
        offsets[random.randrange(count)] = random.choice((-1, 1))
    return offsets


def _render_input(
    height: int,
    width: int,
    blocks: list[tuple[int, int]],
    controls: dict[int, int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for center_r, center_c in blocks:
        grid[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = GREY
    for col, color in controls.items():
        grid[0, col] = int(color)
    return grid


def _independent_expected(
    inp: np.ndarray,
    blocks: list[tuple[int, int]],
    controls: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    mapping = _block_to_color(blocks, controls)
    intersections = _cross_intersections(blocks)

    for center_r, center_c in blocks:
        out[center_r, :] = GREY
        out[:, center_c] = GREY
    for row, col in intersections:
        out[row, col] = 0

    for center_r, center_c in blocks:
        color = mapping[(center_r, center_c)]
        for c in range(out.shape[1]):
            if (center_r, c) not in intersections:
                out[center_r, c] = color
        for r in range(out.shape[0]):
            if (r, center_c) not in intersections:
                out[r, center_c] = color
        out[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = color
    for row, col in intersections:
        out[row, col] = 0
    for col, color in controls.items():
        out[0, col] = color
    return out


def _validate_semantics(inp: np.ndarray) -> bool:
    bg = int(get_bg(inp))
    blocks = _solid_3x3_blocks(inp, bg)
    controls = _controls(inp, bg)
    if len(blocks) != len(controls):
        return False

    offsets = []
    for _r, c in blocks:
        matched = [col for col in controls if c - 1 <= col <= c + 1]
        if len(matched) != 1:
            return False
        offsets.append(int(matched[0] - c))
    if all(offset == 0 for offset in offsets):
        return False

    out = main(inp)
    expected = _independent_expected(inp, blocks, controls)
    if not np.array_equal(out, expected):
        return False

    intersections = _cross_intersections(blocks)
    if len(intersections) != len(blocks) * (len(blocks) - 1):
        return False
    if any(int(out[r, c]) != 0 for r, c in intersections):
        return False

    mapping = _block_to_color(blocks, controls)
    used_colors = set(mapping.values())
    if len(used_colors) != len(blocks):
        return False
    for center_r, center_c in blocks:
        color = mapping[(center_r, center_c)]
        if not np.all(out[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] == color):
            return False
        row_pixels = [
            int(out[center_r, c])
            for c in range(out.shape[1])
            if (center_r, c) not in intersections and c not in controls
        ]
        col_pixels = [
            int(out[r, center_c])
            for r in range(out.shape[0])
            if (r, center_c) not in intersections and r != 0
        ]
        if not row_pixels or not col_pixels:
            return False
        if any(pixel != color for pixel in row_pixels):
            return False
        if any(pixel != color for pixel in col_pixels):
            return False

    for col, color in controls.items():
        if int(out[0, col]) != color:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    count = random.choices(GEN_BLOCK_COUNTS, weights=GEN_BLOCK_COUNT_WEIGHTS, k=1)[0]

    rows = _choose_centers(height, count)
    cols = _choose_centers(width, count)
    if rows is None or cols is None:
        return None

    blocks = sorted(zip(rows, cols))
    offsets = _choose_offsets(count)
    control_colors = random.sample(CONTROL_COLORS, count)
    controls = {center_c + offset: color for (_r, center_c), offset, color in zip(blocks, offsets, control_colors)}
    if len(controls) != count:
        return None

    spans = [(c - 1, c + 1) for _r, c in blocks]
    for idx, (lo, hi) in enumerate(spans):
        inside = [col for col in controls if lo <= col <= hi]
        if len(inside) != 1:
            return None
        for j, (other_lo, other_hi) in enumerate(spans):
            if idx != j and inside[0] >= other_lo and inside[0] <= other_hi:
                return None

    inp = _render_input(height, width, blocks, controls)
    if not _validate_semantics(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
