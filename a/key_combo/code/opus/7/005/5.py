from __future__ import annotations

import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = spec_from_file_location("_arc_utils_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste
is_contiguous = _ARC_UTILS.is_contiguous


FRAME_COLOR = Color.GREY
CODE_COLOR = Color.RED
PATTERN_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.TEAL, Color.MAROON)
GEN_RETRY_LIMIT = 500


def _border_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(width)]
    coords.extend((r, width - 1) for r in range(1, height - 1))
    coords.extend((height - 1, c) for c in range(width - 1, -1, -1))
    coords.extend((r, 0) for r in range(height - 2, 0, -1))
    return coords


def _run_lengths(seq: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for value in seq:
        color = int(value)
        if runs and runs[-1][0] == color:
            runs[-1] = (color, runs[-1][1] + 1)
        else:
            runs.append((color, 1))
    return runs


def _decode_axis_from_border(grid: np.ndarray) -> str:
    border = [int(grid[r, c]) for r, c in _border_coords(*grid.shape)]
    runs = _run_lengths(border)
    red_runs = [length for color, length in runs if color == CODE_COLOR]
    grey_runs = [length for color, length in runs if color == FRAME_COLOR]
    assert red_runs and grey_runs, "border must contain both red segments and grey gaps"
    seg_len = max(red_runs)
    gap_len = max(grey_runs)
    assert seg_len != gap_len, "segment length and gap length must differ"
    return "vertical" if seg_len > gap_len else "horizontal"


def _mirror_fill(grid: np.ndarray, axis: str) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    inner = out[1:-1, 1:-1]
    rows, cols = inner.shape
    source = np.asarray(grid, dtype=int)[1:-1, 1:-1]
    for r, c in np.argwhere(source != Color.BLACK):
        rr = int(r)
        cc = int(c)
        if axis == "vertical":
            out[1 + rr, 1 + (cols - 1 - cc)] = int(source[rr, cc])
        else:
            out[1 + (rows - 1 - rr), 1 + cc] = int(source[rr, cc])
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decode a border rhythm to choose the mirror axis and complete the interior pattern.

    Input structure:
    - A one-cell grey rectangular frame surrounds the grid.
    - Red border segments repeat clockwise around that frame with equal red lengths and equal grey gaps, possibly with a truncated first or last partial run.
    - Inside the frame, only one half contains a multi-color pattern while the rest is black.

    Rule:
    - Read the border clockwise as a 1D sequence and measure the repeated red-segment length n and grey-gap length g.
    - If n > g, reflect the interior non-black pixels across the vertical center line; if g > n, reflect them across the horizontal center line.
    - Keep the border unchanged and copy the reflected pixels with their original colors.

    Key insight:
    - The border decoration is an instruction channel: the segment/gap ratio, not the visible pattern position, determines which mirror axis completes the picture.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2 and min(grid.shape) >= 5
    axis = _decode_axis_from_border(grid)
    return _mirror_fill(grid, axis)


def _build_border_sequence(perimeter: int, seg_len: int, gap_len: int, start_phase: int, starts_red: bool) -> list[int]:
    period = seg_len + gap_len
    seq: list[int] = []
    for idx in range(perimeter):
        phase = (start_phase + idx) % period
        if starts_red:
            seq.append(CODE_COLOR if phase < seg_len else FRAME_COLOR)
        else:
            seq.append(FRAME_COLOR if phase < gap_len else CODE_COLOR)
    return seq


def _random_patch(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None

    patch_h = random.randint(2, min(5, max_h))
    patch_w = random.randint(2, min(5, max_w))
    area_cap = patch_h * patch_w
    target = random.randint(4, min(area_cap, 8))

    coords = {(random.randrange(patch_h), random.randrange(patch_w))}
    while len(coords) < target:
        frontier = set()
        for r, c in coords:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < patch_h and 0 <= cc < patch_w and (rr, cc) not in coords:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        coords.add(random.choice(tuple(frontier)))

    mask = np.zeros((patch_h, patch_w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    if not is_contiguous(mask, connectivity=4):
        return None

    used = np.argwhere(mask)
    r0 = int(used[:, 0].min())
    c0 = int(used[:, 1].min())
    r1 = int(used[:, 0].max())
    c1 = int(used[:, 1].max())
    if r0 == r1 or c0 == c1:
        return None

    mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    if int(mask.sum()) < 4:
        return None

    patch = np.zeros(mask.shape, dtype=int)
    palette_size = random.randint(2, min(4, len(PATTERN_COLORS)))
    palette = random.sample(PATTERN_COLORS, palette_size)
    cells = [tuple(cell) for cell in np.argwhere(mask)]
    random.shuffle(cells)
    for idx, (r, c) in enumerate(cells):
        patch[r, c] = palette[idx % palette_size]
    return patch


def _source_box_limits(inner_h: int, inner_w: int, axis: str, mode: str, patch: np.ndarray) -> tuple[int, int, int, int] | None:
    patch_h, patch_w = patch.shape
    row_lo = 1
    row_hi = inner_h - patch_h - 1
    col_lo = 1
    col_hi = inner_w - patch_w - 1
    if axis == "vertical":
        col_hi = min(col_hi, inner_w // 2 - patch_w)
        if mode == "corner":
            row_hi = min(row_hi, inner_h // 2 - patch_h)
    else:
        row_hi = min(row_hi, inner_h // 2 - patch_h)
        if mode == "corner":
            col_hi = min(col_hi, inner_w // 2 - patch_w)
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return row_lo, row_hi, col_lo, col_hi


def _place_patch(grid: np.ndarray, patch: np.ndarray, axis: str, mode: str) -> bool:
    inner_h = grid.shape[0] - 2
    inner_w = grid.shape[1] - 2
    limits = _source_box_limits(inner_h, inner_w, axis, mode, patch)
    if limits is None:
        return False
    row_lo, row_hi, col_lo, col_hi = limits
    top = random.randint(row_lo, row_hi)
    left = random.randint(col_lo, col_hi)
    paste(grid, patch, 1 + top, 1 + left, transparent=Color.BLACK)
    return True


def _expected_output(inp: np.ndarray, axis: str) -> np.ndarray:
    return _mirror_fill(inp, axis)


def _wrong_axis(axis: str) -> str:
    return "horizontal" if axis == "vertical" else "vertical"


def _border_candidates(perimeter: int) -> list[tuple[int, int]]:
    out = []
    for seg_len in range(1, 6):
        for gap_len in range(1, 6):
            if seg_len == gap_len:
                continue
            period = seg_len + gap_len
            if perimeter >= period * 4:
                out.append((seg_len, gap_len))
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(11, 18)
    width = random.randint(11, 18)
    perimeter = 2 * (height + width) - 4
    border_options = _border_candidates(perimeter)
    if not border_options:
        return None

    axis = random.choice(("vertical", "horizontal"))
    mode = random.choices(("corner", "free"), weights=(3, 2), k=1)[0]
    patch = _random_patch(max_h=(height - 2) // 2 - 1 if axis == "horizontal" else height - 4,
                          max_w=(width - 2) // 2 - 1 if axis == "vertical" else width - 4)
    if patch is None:
        return None

    if axis == "vertical":
        valid_border = [(n, g) for n, g in border_options if n > g]
    else:
        valid_border = [(n, g) for n, g in border_options if g > n]
    if not valid_border:
        return None
    seg_len, gap_len = random.choice(valid_border)
    starts_red = bool(random.getrandbits(1))
    start_phase = random.randrange(seg_len + gap_len)
    border_seq = _build_border_sequence(perimeter, seg_len, gap_len, start_phase, starts_red)
    runs = _run_lengths(border_seq)
    if max(length for color, length in runs if color == CODE_COLOR) != seg_len:
        return None
    if max(length for color, length in runs if color == FRAME_COLOR) != gap_len:
        return None

    grid = np.full((height, width), Color.BLACK, dtype=int)
    for (r, c), color in zip(_border_coords(height, width), border_seq):
        grid[r, c] = color

    if not _place_patch(grid, patch, axis, mode):
        return None

    expected = _expected_output(grid, axis)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if _decode_axis_from_border(grid) != axis:
        return None
    if np.array_equal(_mirror_fill(grid, _wrong_axis(axis)), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid border-code mirror puzzle")
