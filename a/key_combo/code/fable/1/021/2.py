import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_fable_1_021_2", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)
colors_present = _ARC_UTILS.colors_present


PALETTE = tuple(range(10))
ANSWER_DIM_MIN = 2
ANSWER_DIM_MAX = 6
STRETCH_CHOICES = (1, 2, 3, 4)
STRETCH_WEIGHTS = (1, 3, 3, 2)
BLANK_MODES = ("noise", "checker", "solid")
MAX_GENERATION_ATTEMPTS = 400


def _compatible(template: np.ndarray, line: np.ndarray, blank: int) -> bool:
    both_visible = (template != blank) & (line != blank)
    return bool(np.all(template[both_visible] == line[both_visible]))


def _merge_template(template: np.ndarray, line: np.ndarray, blank: int) -> np.ndarray:
    visible = line != blank
    conflict = visible & (template != blank) & (template != line)
    assert not np.any(conflict), "incompatible rows/columns cannot share one segment"
    merged = template.copy()
    fill = (merged == blank) & visible
    merged[fill] = line[fill]
    return merged


def _collapse_lines(lines: np.ndarray, blank: int) -> tuple[np.ndarray, tuple[int, ...]]:
    lines = np.asarray(lines, dtype=int)
    assert lines.ndim == 2 and lines.shape[0] >= 1

    merged_lines: list[np.ndarray] = []
    lengths: list[int] = []

    template = lines[0].copy()
    seg_len = 1
    for idx in range(1, lines.shape[0]):
        line = lines[idx]
        if _compatible(template, line, blank):
            template = _merge_template(template, line, blank)
            seg_len += 1
        else:
            assert not np.any(template == blank), "each segment position must stay visible somewhere"
            merged_lines.append(template)
            lengths.append(seg_len)
            template = line.copy()
            seg_len = 1

    assert not np.any(template == blank), "each segment position must stay visible somewhere"
    merged_lines.append(template)
    lengths.append(seg_len)
    return np.stack(merged_lines), tuple(lengths)


def _check_segment_coverage(lines: np.ndarray, blank: int, lengths: tuple[int, ...]) -> None:
    start = 0
    for length in lengths:
        segment = lines[start : start + length]
        assert np.all(np.any(segment != blank, axis=0)), "segment contains an all-blank position"
        start += length


def _check_segment_boundaries(lines: np.ndarray, blank: int, lengths: tuple[int, ...]) -> None:
    start = 0
    for length in lengths[:-1]:
        upper = lines[start + length - 1]
        lower = lines[start + length]
        both_visible = (upper != blank) & (lower != blank)
        assert np.any(both_visible & (upper != lower)), "adjacent segments need a visible differing witness"
        start += length


def _collapse_axis(grid: np.ndarray, blank: int, axis: int) -> tuple[np.ndarray, tuple[int, ...]]:
    assert axis in (0, 1)
    lines = np.asarray(grid, dtype=int) if axis == 0 else np.asarray(grid, dtype=int).T
    collapsed_lines, lengths = _collapse_lines(lines, blank)
    _check_segment_coverage(lines, blank, lengths)
    _check_segment_boundaries(lines, blank, lengths)
    collapsed = collapsed_lines if axis == 0 else collapsed_lines.T
    return collapsed, lengths


def _stretch(answer: np.ndarray, row_lengths: tuple[int, ...], col_lengths: tuple[int, ...]) -> np.ndarray:
    return np.repeat(np.repeat(answer, row_lengths, axis=0), col_lengths, axis=1)


def _solve_for_blank(grid: np.ndarray, blank: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    row_first, row_lengths = _collapse_axis(grid, blank, axis=0)
    rc_out, col_lengths = _collapse_axis(row_first, blank, axis=1)

    col_first, col_lengths_2 = _collapse_axis(grid, blank, axis=1)
    cr_out, row_lengths_2 = _collapse_axis(col_first, blank, axis=0)

    assert row_lengths == row_lengths_2, "row segmentation must be order-independent"
    assert col_lengths == col_lengths_2, "column segmentation must be order-independent"
    assert np.array_equal(rc_out, cr_out), "row/column collapse order must agree"
    assert not np.any(rc_out == blank), "recovered matrix cannot contain the wildcard color"

    rebuilt = _stretch(rc_out, row_lengths, col_lengths)
    assert rebuilt.shape == grid.shape
    blank_mask = grid == blank
    assert np.any(blank_mask), "input must contain at least one wildcard cell"
    assert np.array_equal(grid[~blank_mask], rebuilt[~blank_mask]), "non-wildcard cells must match the rebuild"
    return rc_out, row_lengths, col_lengths


def _valid_candidates(grid: np.ndarray) -> list[tuple[int, np.ndarray, tuple[int, ...], tuple[int, ...]]]:
    valid = []
    for blank in colors_present(grid):
        try:
            out, row_lengths, col_lengths = _solve_for_blank(grid, int(blank))
        except AssertionError:
            continue
        valid.append((int(blank), out, row_lengths, col_lengths))
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse a stretched matrix by treating the erased color as a wildcard and filling every hole.

    Input structure:
    - The input is a row-and-column stretched copy of a smaller color matrix.
    - One extra color marks erased cells; that wildcard color does not belong to the small matrix.
    - Within each true row segment and column segment, every position remains visible somewhere.

    Rule:
    - Try each present color as the wildcard.
    - Merge adjacent rows when all positions that are visible in both rows agree, then fill each merged row position
      with the unique visible color from that segment; do the same for columns.
    - Keep only the unique candidate whose row-first and column-first collapses agree and whose rebuilt stretch
      explains every non-wildcard input cell.

    Key insight:
    - Row deduplication and hole filling certify each other: wildcard-aware merging reveals the hidden small matrix,
      and the recovered segment values are exactly what each erased cell must have been.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 1, "expected a non-empty 2D grid"

    valid = _valid_candidates(g)
    assert len(valid) == 1, "expected a unique wildcard color"
    _blank, out, _row_lengths, _col_lengths = valid[0]
    return out


def _sample_answer(blank: int, h: int, w: int) -> np.ndarray:
    palette = [color for color in PALETTE if color != blank]
    for _ in range(300):
        n_colors = random.randint(2, min(6, len(palette), h * w))
        chosen = random.sample(palette, n_colors)
        answer = np.array([random.choice(chosen) for _ in range(h * w)], dtype=int).reshape(h, w)
        if len(np.unique(answer)) < 2:
            continue
        if any(np.all(answer[r] == answer[r + 1]) for r in range(h - 1)):
            continue
        if any(np.all(answer[:, c] == answer[:, c + 1]) for c in range(w - 1)):
            continue
        return answer
    raise RuntimeError("failed to sample a valid answer matrix")


def _sample_stretches(n_segments: int) -> tuple[int, ...]:
    for _ in range(200):
        stretches = tuple(int(v) for v in random.choices(STRETCH_CHOICES, weights=STRETCH_WEIGHTS, k=n_segments))
        if max(stretches) >= 3 and any(v > 1 for v in stretches):
            return stretches
    raise RuntimeError("failed to sample stretch factors")


def _segment_bounds(lengths: tuple[int, ...]) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for length in lengths:
        bounds.append((start, start + length))
        start += length
    return bounds


def _blank_mask_for_block(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if h <= 2 or w <= 2 or random.random() < 0.35:
        return mask

    mode = random.choice(BLANK_MODES)
    if mode == "solid":
        mask[1:-1, 1:-1] = True
    elif mode == "checker":
        parity = random.randint(0, 1)
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                mask[r, c] = ((r + c + parity) % 2) == 0
    else:
        density = random.uniform(0.25, 0.8)
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if random.random() < density:
                    mask[r, c] = True
        if not np.any(mask):
            mask[random.randint(1, h - 2), random.randint(1, w - 2)] = True
    return mask


def _render_input(answer: np.ndarray, row_lengths: tuple[int, ...], col_lengths: tuple[int, ...], blank: int) -> np.ndarray:
    grid = _stretch(answer, row_lengths, col_lengths)
    row_bounds = _segment_bounds(row_lengths)
    col_bounds = _segment_bounds(col_lengths)

    blank_cells = 0
    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            block = grid[r0:r1, c0:c1]
            mask = _blank_mask_for_block(block.shape[0], block.shape[1])
            block[mask] = blank
            blank_cells += int(mask.sum())

    if blank_cells == 0:
        candidates = [(r0, r1, c0, c1) for r0, r1 in row_bounds for c0, c1 in col_bounds if (r1 - r0) >= 3 and (c1 - c0) >= 3]
        assert candidates, "stretch factors must allow at least one interior blank"
        r0, r1, c0, c1 = random.choice(candidates)
        rr = random.randint(r0 + 1, r1 - 2)
        cc = random.randint(c0 + 1, c1 - 2)
        grid[rr, cc] = blank
    return grid


def _gen_attempt() -> np.ndarray | None:
    blank = random.choice(PALETTE)
    answer_h = random.randint(ANSWER_DIM_MIN, ANSWER_DIM_MAX)
    answer_w = random.randint(ANSWER_DIM_MIN, ANSWER_DIM_MAX)

    row_lengths = _sample_stretches(answer_h)
    col_lengths = _sample_stretches(answer_w)
    answer = _sample_answer(blank, answer_h, answer_w)
    inp = _render_input(answer, row_lengths, col_lengths, blank)

    if max(inp.shape) > 30:
        return None

    valid = _valid_candidates(inp)
    if len(valid) != 1:
        return None

    found_blank, solved, found_rows, found_cols = valid[0]
    if found_blank != blank:
        return None
    if not np.array_equal(solved, answer):
        return None
    if found_rows != row_lengths or found_cols != col_lengths:
        return None
    if np.any(solved == blank) or not np.any(inp == blank):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
