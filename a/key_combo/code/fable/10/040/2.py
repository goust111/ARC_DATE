import random
from collections import Counter

import numpy as np


def _unique_majority(arr: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(arr).ravel()))
    color, top = counts.most_common(1)[0]
    tied = [value for value in counts.values() if value == top]
    assert len(tied) == 1, "expected a unique majority color"
    return int(color)


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _parse_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(g[0, 0])
    assert np.all(g[0, :] == bg) and np.all(g[-1, :] == bg), "expected full background top/bottom border"
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "expected full background left/right border"

    full_bg_rows = np.all(g == bg, axis=1)
    full_bg_cols = np.all(g == bg, axis=0)
    row_runs = _runs([r for r, is_bg in enumerate(full_bg_rows) if not is_bg])
    col_runs = _runs([c for c, is_bg in enumerate(full_bg_cols) if not is_bg])

    assert row_runs and col_runs, "expected at least one foreground block"
    assert full_bg_rows[0] and full_bg_rows[-1], "expected separator rows at the outer border"
    assert full_bg_cols[0] and full_bg_cols[-1], "expected separator cols at the outer border"
    return bg, row_runs, col_runs


def _summary_and_match(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg, row_runs, col_runs = _parse_lattice(g)
    rows = len(row_runs)
    cols = len(col_runs)

    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert row_heights == {rows}, "each block height must equal the number of row bands"
    assert col_widths == {cols}, "each block width must equal the number of column bands"

    summary = np.zeros((rows, cols), dtype=int)
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for rr, (r0, r1) in enumerate(row_runs):
        for cc, (c0, c1) in enumerate(col_runs):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(block == bg), "blocks must contain no separator color"
            summary[rr, cc] = _unique_majority(block)
            blocks[(rr, cc)] = block

    matches = [(rr, cc) for (rr, cc), block in blocks.items() if np.array_equal(block, summary)]
    assert len(matches) == 1, "expected exactly one block equal to the global majority summary"
    return summary, matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse the block lattice to its majority-color thumbnail, then locate the one block that literally draws that thumbnail.

    Input structure:
    - A single separator color forms the complete outer border plus full separator rows and columns.
    - The separators partition the foreground into an R-by-C lattice of equal-sized filled blocks.
    - Each block is exactly R rows by C columns, contains no separator cells, and has one unique majority color.

    Rule:
    - Compute the R-by-C thumbnail whose cell at each lattice position is the unique majority color of that block.
    - Find the unique input block whose full R-by-C pixel content matches that thumbnail exactly, cell by cell.
    - Output an R-by-C black grid with only that lattice position marked by the block's majority color.

    Key insight:
    - The same thumbnail serves both as the global compression result and as the template to search for inside one micro-block.
    """
    summary, (match_r, match_c) = _summary_and_match(grid)
    out = np.zeros(summary.shape, dtype=int)
    out[match_r, match_c] = int(summary[match_r, match_c])
    return out


def _sample_composition(total: int, parts: int, cap: int, rng: random.Random) -> list[int]:
    if parts <= 0 or parts > total:
        raise ValueError("invalid composition request")
    values = [1] * parts
    remaining = total - parts
    while remaining > 0:
        choices = [idx for idx, value in enumerate(values) if value < cap]
        if not choices:
            raise ValueError("composition cap is too small")
        idx = rng.choice(choices)
        values[idx] += 1
        remaining -= 1
    rng.shuffle(values)
    return values


def _sample_summary(rows: int, cols: int, palette: list[int], rng: random.Random) -> np.ndarray:
    n_cells = rows * cols
    min_colors = 3 if n_cells <= 6 else 4
    low = max(2, n_cells // 4 + 1)
    high = max(low, min(n_cells - 2, n_cells // 3 + 1))

    for _ in range(200):
        majority_count = rng.randint(low, high)
        majority_color = rng.choice(palette)
        remaining = n_cells - majority_count
        max_colors = min(len(palette), min(6, remaining + 1))
        max_other_parts = min(max_colors - 1, remaining)
        min_other_parts = max(min_colors - 1, (remaining + (majority_count - 2)) // (majority_count - 1))
        if min_other_parts > max_other_parts:
            continue

        n_other = rng.randint(min_other_parts, max_other_parts)
        other_colors = rng.sample([color for color in palette if color != majority_color], n_other)
        other_counts = _sample_composition(remaining, n_other, majority_count - 1, rng)

        flat = [majority_color] * majority_count
        for color, count in zip(other_colors, other_counts, strict=True):
            flat.extend([color] * count)

        for _ in range(100):
            rng.shuffle(flat)
            summary = np.array(flat, dtype=int).reshape(rows, cols)
            if np.any(summary == majority_color) and np.sum(summary == majority_color) >= 2:
                row_uniform = any(len(set(map(int, row))) == 1 for row in summary)
                col_uniform = any(len(set(map(int, col))) == 1 for col in summary.T)
                if not row_uniform and not col_uniform:
                    return summary
    raise RuntimeError("failed to sample a non-degenerate summary")


def _is_unique_mode(flat: np.ndarray, color: int) -> bool:
    counts = Counter(map(int, flat))
    top = max(counts.values())
    return counts[int(color)] == top and sum(value == top for value in counts.values()) == 1


def _recolor_one(flat: np.ndarray, from_color: int, to_color: int, rng: random.Random) -> bool:
    indices = [idx for idx, value in enumerate(flat.tolist()) if int(value) == int(from_color)]
    if not indices:
        return False
    flat[rng.choice(indices)] = int(to_color)
    return True


def _build_block_fallback(
    shape: tuple[int, int],
    target_color: int,
    palette: list[int],
    majority_count: int,
    rng: random.Random,
    summary: np.ndarray,
) -> np.ndarray:
    n_cells = shape[0] * shape[1]
    remaining = n_cells - majority_count
    min_other_parts = max(2, (remaining + (majority_count - 2)) // (majority_count - 1))
    max_other_parts = min(4, remaining, len([color for color in palette if color != int(target_color)]))
    if min_other_parts > max_other_parts:
        min_other_parts = max_other_parts
    if min_other_parts <= 0:
        raise RuntimeError("invalid fallback block composition")

    for _ in range(200):
        n_other = rng.randint(min_other_parts, max_other_parts)
        other_colors = rng.sample([color for color in palette if color != int(target_color)], n_other)
        other_counts = _sample_composition(remaining, n_other, majority_count - 1, rng)
        flat = [int(target_color)] * majority_count
        for color, count in zip(other_colors, other_counts, strict=True):
            flat.extend([int(color)] * count)
        rng.shuffle(flat)
        block = np.array(flat, dtype=int).reshape(shape)
        if _unique_majority(block) == int(target_color) and len(set(map(int, block.ravel()))) >= 3 and not np.array_equal(block, summary):
            return block
    raise RuntimeError("failed to build fallback distractor block")


def _build_distractor_block(
    summary: np.ndarray,
    target_color: int,
    palette: list[int],
    rng: random.Random,
    near_copy: bool,
) -> np.ndarray:
    flat = summary.ravel().copy()
    summary_majority = _unique_majority(summary)
    majority_count = Counter(map(int, summary.ravel()))[summary_majority]

    if near_copy:
        assert int(target_color) == int(summary_majority), "near-copy distractor must keep the summary majority color"
        non_target = [idx for idx, value in enumerate(flat.tolist()) if int(value) != int(target_color)]
        assert non_target, "summary must contain a minority color"
        flat[rng.choice(non_target)] = int(target_color)
    else:
        for _ in range(flat.size * 3):
            if _is_unique_mode(flat, target_color) and not np.array_equal(flat.reshape(summary.shape), summary):
                break
            counts = Counter(map(int, flat))
            donor_colors = [color for color, count in counts.items() if color != int(target_color) and count > 0]
            donor_colors.sort(key=lambda color: (-counts[color], color))
            if not donor_colors:
                break
            if not _recolor_one(flat, donor_colors[0], target_color, rng):
                break
        else:
            raise RuntimeError("failed to create a unique-majority distractor")

        if np.array_equal(flat.reshape(summary.shape), summary):
            donor_candidates = [idx for idx, value in enumerate(flat.tolist()) if int(value) != int(target_color)]
            if donor_candidates:
                flat[rng.choice(donor_candidates)] = int(target_color)

    block = flat.reshape(summary.shape)
    if len(set(map(int, block.ravel()))) < 3:
        spare_colors = [color for color in palette if color != int(target_color)]
        target_cells = [idx for idx, value in enumerate(flat.tolist()) if int(value) == int(target_color)]
        if spare_colors and len(target_cells) >= 2:
            old_value = int(flat[target_cells[0]])
            flat[target_cells[0]] = int(rng.choice(spare_colors))
            if _is_unique_mode(flat, target_color):
                block = flat.reshape(summary.shape)
            else:
                flat[target_cells[0]] = old_value
                block = flat.reshape(summary.shape)

    if _unique_majority(block) != int(target_color) or np.array_equal(block, summary) or len(set(map(int, block.ravel()))) < 3:
        block = _build_block_fallback(summary.shape, target_color, palette, majority_count, rng, summary)

    assert _unique_majority(block) == int(target_color)
    assert not np.array_equal(block, summary)
    assert len(set(map(int, block.ravel()))) >= 3
    return block


def _assemble_grid(summary: np.ndarray, blocks: dict[tuple[int, int], np.ndarray], bg: int) -> np.ndarray:
    rows, cols = summary.shape
    block_h, block_w = rows, cols
    height = rows * block_h + (rows + 1)
    width = cols * block_w + (cols + 1)
    grid = np.full((height, width), int(bg), dtype=int)

    for rr in range(rows):
        r0 = 1 + rr * (block_h + 1)
        for cc in range(cols):
            c0 = 1 + cc * (block_w + 1)
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = blocks[(rr, cc)]
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    summary, match = _summary_and_match(grid)
    match_color = int(summary[match])
    assert _unique_majority(summary) == match_color, "the matching block must inherit the summary's unique majority"
    assert int(np.sum(summary == match_color)) >= 2, "match color must appear in multiple summary cells"

    _, row_runs, col_runs = _parse_lattice(grid)
    similarities = []
    for rr, (r0, r1) in enumerate(row_runs):
        for cc, (c0, c1) in enumerate(col_runs):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            same = int(np.sum(block == summary))
            similarities.append(((rr, cc), same))
    near_miss = max(score for pos, score in similarities if pos != match)
    assert near_miss >= summary.size - 2, "generator should include at least one near-match distractor"

    expected = np.zeros(summary.shape, dtype=int)
    expected[match] = match_color
    assert np.array_equal(main(grid), expected), "solver output must follow the puzzle rule exactly"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    bg = rng.randint(0, 9)
    palette = [color for color in range(1, 10) if color != bg]
    if len(palette) < 4:
        return None

    try:
        summary = _sample_summary(rows, cols, palette, rng)
    except RuntimeError:
        return None

    match_color = _unique_majority(summary)
    match_candidates = [tuple(coord) for coord in np.argwhere(summary == match_color)]
    if len(match_candidates) < 2:
        return None
    match_pos = tuple(rng.choice(match_candidates))

    same_color_distractors = [pos for pos in match_candidates if pos != match_pos]
    near_copy_pos = rng.choice(same_color_distractors) if same_color_distractors else None

    blocks: dict[tuple[int, int], np.ndarray] = {}
    for rr in range(rows):
        for cc in range(cols):
            pos = (rr, cc)
            target_color = int(summary[pos])
            if pos == match_pos:
                block = summary.copy()
            else:
                block = _build_distractor_block(
                    summary,
                    target_color,
                    palette,
                    rng,
                    near_copy=pos == near_copy_pos,
                )
            blocks[pos] = block

    grid = _assemble_grid(summary, blocks, bg)
    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential lattice input")
