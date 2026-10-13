from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

PALETTE = (RED, GREEN, YELLOW, ORANGE)
GEN_RETRIES = 400


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if not indices:
        return runs
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _full_color_rows(grid: np.ndarray, color: int) -> np.ndarray:
    return np.all(grid == int(color), axis=1)


def _full_color_cols(grid: np.ndarray, color: int) -> np.ndarray:
    return np.all(grid == int(color), axis=0)


def _first_occurrence(grid: np.ndarray) -> dict[int, int]:
    first: dict[int, int] = {}
    flat = grid.reshape(-1)
    for idx, value in enumerate(flat):
        color = int(value)
        if color not in first:
            first[color] = idx
    return first


def _majority_with_tiebreak(values: list[int], first_pos: dict[int, int]) -> int:
    counts = Counter(map(int, values))
    assert counts, "expected at least one candidate color"
    best = max(counts.values())
    tied = [color for color, count in counts.items() if count == best]
    if len(tied) == 1:
        return int(tied[0])
    return int(min(tied, key=lambda color: first_pos[int(color)]))


def _legend_mapping(grid: np.ndarray, legend_row_band: tuple[int, int], legend_col_band: tuple[int, int]) -> dict[int, int]:
    r0, r1 = legend_row_band
    c0, c1 = legend_col_band
    mapping: dict[int, int] = {}
    for r in range(r0, r1 + 1):
        colored = [c for c in range(c0, c1 + 1) if int(grid[r, c]) not in (BLACK, GREY)]
        assert len(colored) == 2, "expected each legend row to contain exactly one input/output pair"
        src = int(grid[r, colored[0]])
        dst = int(grid[r, colored[1]])
        mapping[src] = dst
    assert mapping, "expected a non-empty legend mapping"
    return mapping


def _bands_between(separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    for left, right in zip(separators, separators[1:]):
        if right - left > 1:
            bands.append((left + 1, right - 1))
    return bands


def _subcell_bands(block: np.ndarray, sep_color: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    full_rows = np.all(block == int(sep_color), axis=1)
    full_cols = np.all(block == int(sep_color), axis=0)
    row_bands = _runs([r for r, is_sep in enumerate(full_rows) if not is_sep])
    col_bands = _runs([c for c, is_sep in enumerate(full_cols) if not is_sep])
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a 2x2 or 3x3 fine grid"
    assert len(row_bands) == len(col_bands), "expected square fine partitions"
    assert len(row_bands) in (2, 3), "expected 2x2 or 3x3 fine partitions"
    return row_bands, col_bands


def _subcell_representative(grid: np.ndarray, row_band: tuple[int, int], col_band: tuple[int, int], first_pos: dict[int, int]) -> int:
    r0, r1 = row_band
    c0, c1 = col_band
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = [int(value) for value in patch.reshape(-1) if int(value) not in (BLACK, BLUE, GREY)]
    assert colors, "expected a colored subcell interior"
    return _majority_with_tiebreak(colors, first_pos)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse a two-level separator grid by voting inside fine subcells, then remap each coarse winner through the legend.

    Input structure
    - The grid has black background, grey full-row/full-column separators, and a
      top-left legend box bounded by those grey separators.
    - The grey separators isolate a lower-right lattice of coarse blocks.
    - Inside each coarse block, blue full rows and columns split the block into a
      2x2 or 3x3 lattice of colored subcells whose main colors come from the
      legend domain.

    Rule
    - Read the legend in the top-left band as 1x2 color pairs mapping input
      colors to output colors.
    - For each fine subcell, count only non-black, non-blue, non-grey pixels and
      take the majority color; ties use the color whose first appearance in the
      whole input is earliest in row-major order.
    - For each coarse block, majority-vote over its subcell representative colors
      with the same tie rule, then recolor that coarse winner through the legend.
    - Return one output pixel per coarse block, arranged in the same coarse-grid
      layout; the legend does not appear in the output.

    Key insight
    - The grey grid and the blue grid encode two different voting scales: the blue
      splits decide subcell representatives first, and only those representatives
      participate in the coarse vote before the legend remaps the result.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_rows = [r for r, ok in enumerate(_full_color_rows(g, GREY)) if ok]
    grey_cols = [c for c, ok in enumerate(_full_color_cols(g, GREY)) if ok]
    assert len(grey_rows) >= 4 and len(grey_cols) >= 4, "expected grey separators and outer border"

    row_bands = _bands_between(grey_rows)
    col_bands = _bands_between(grey_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected one legend band plus coarse bands"

    legend_row_band = row_bands[0]
    legend_col_band = col_bands[0]
    mapping = _legend_mapping(g, legend_row_band, legend_col_band)
    first_pos = _first_occurrence(g)

    coarse_row_bands = row_bands[1:]
    coarse_col_bands = col_bands[1:]
    out = np.zeros((len(coarse_row_bands), len(coarse_col_bands)), dtype=int)

    for out_r, row_band in enumerate(coarse_row_bands):
        for out_c, col_band in enumerate(coarse_col_bands):
            r0, r1 = row_band
            c0, c1 = col_band
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            fine_rows, fine_cols = _subcell_bands(block, BLUE)

            reps: list[int] = []
            for fr0, fr1 in fine_rows:
                for fc0, fc1 in fine_cols:
                    rep = _subcell_representative(
                        g,
                        (r0 + fr0, r0 + fr1),
                        (c0 + fc0, c0 + fc1),
                        first_pos,
                    )
                    reps.append(rep)

            agg = _majority_with_tiebreak(reps, first_pos)
            assert agg in mapping, "legend must cover every coarse aggregate color"
            out[out_r, out_c] = mapping[agg]

    return out


def _derangement(colors: tuple[int, ...]) -> dict[int, int]:
    colors = tuple(map(int, colors))
    if len(colors) == 2:
        return {colors[0]: colors[1], colors[1]: colors[0]}
    shuffled = list(colors)
    while True:
        random.shuffle(shuffled)
        if all(src != dst for src, dst in zip(colors, shuffled)):
            return {src: int(dst) for src, dst in zip(colors, shuffled)}


def _compositions(total: int, parts: int, min_part: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)] if total >= min_part else []
    out: list[tuple[int, ...]] = []
    max_first = total - min_part * (parts - 1)
    for first in range(min_part, max_first + 1):
        for rest in _compositions(total - first, parts - 1, min_part):
            out.append((first, *rest))
    return out


def _sample_sizes(
    n_blocks: int,
    min_size: int,
    max_size: int,
    max_sum: int,
    require_eight: bool,
    min_big_count: int = 0,
) -> tuple[int, ...] | None:
    candidates = [
        sizes
        for sizes in map(tuple, np.ndindex(*([max_size - min_size + 1] * n_blocks)))
        for sizes in [tuple(min_size + v for v in sizes)]
        if sum(sizes) <= max_sum
        and (not require_eight or max(sizes) >= 8)
        and sum(size >= 8 for size in sizes) >= min_big_count
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _render_legend(rows: list[int], mapping: dict[int, int]) -> np.ndarray:
    height = len(rows) + 2
    legend = np.full((height, 4), GREY, dtype=int)
    for i, color in enumerate(rows, start=1):
        legend[i, 1] = int(color)
        legend[i, 2] = int(mapping[int(color)])
    return legend


def _precedence_from_legend(legend: np.ndarray, colors: tuple[int, ...]) -> list[int]:
    first = _first_occurrence(legend)
    return sorted(colors, key=lambda color: first[int(color)])


def _choose_noise_positions(area: int, allow_noise: bool) -> int:
    if not allow_noise or area < 4:
        return 0
    return random.randint(0, min(area // 4, int(area * 0.3)))


def _render_subcell(height: int, width: int, main_color: int, colors: tuple[int, ...], allow_noise: bool) -> np.ndarray:
    patch = np.full((height, width), int(main_color), dtype=int)
    noise_count = _choose_noise_positions(height * width, allow_noise)
    if noise_count == 0:
        return patch
    cells = random.sample(range(height * width), noise_count)
    others = [color for color in colors if color != main_color]
    for idx in cells:
        patch[idx // width, idx % width] = int(random.choice(others))
    counts = Counter(map(int, patch.reshape(-1)))
    assert counts[int(main_color)] == max(counts.values())
    assert list(counts.values()).count(counts[int(main_color)]) == 1
    return patch


def _block_partitions(total: int, cells_per_axis: int, skew: bool) -> tuple[int, ...]:
    interior = total - (cells_per_axis - 1)
    assert interior >= 2 * cells_per_axis
    if cells_per_axis == 2 and skew:
        large = interior - 2
        return (large, 2)
    return random.choice(_compositions(interior, cells_per_axis, 2))


def _majority_matrix(target: int, colors: tuple[int, ...], size: int) -> np.ndarray:
    reps = np.full((size, size), int(target), dtype=int)
    choices = [(r, c) for r in range(size) for c in range(size)]
    off_count = 1 if size == 2 else random.randint(3, 5)
    off_count = min(off_count, size * size - 1)
    for r, c in random.sample(choices, off_count):
        reps[r, c] = int(random.choice([color for color in colors if color != target]))
    assert np.any(reps != int(target))
    return reps


def _tie_matrix(target: int, loser: int) -> np.ndarray:
    reps = np.array([[target, loser], [loser, target]], dtype=int)
    if random.random() < 0.5:
        reps = reps[:, ::-1]
    if random.random() < 0.5:
        reps = reps[::-1, :]
    return reps


def _anti_shortcut_matrix(target: int, big_color: int) -> np.ndarray:
    reps = np.array([[big_color, target], [target, target]], dtype=int)
    if random.random() < 0.5:
        reps = reps[:, ::-1]
    if random.random() < 0.5:
        reps = reps[::-1, :]
    return reps


def _render_block(
    height: int,
    width: int,
    reps: np.ndarray,
    colors: tuple[int, ...],
    *,
    skew: bool,
    allow_noise: bool,
) -> tuple[np.ndarray, list[int]]:
    size = int(reps.shape[0])
    row_sizes = _block_partitions(height, size, skew)
    col_sizes = _block_partitions(width, size, skew)

    block = np.full((height, width), BLUE, dtype=int)
    rep_list: list[int] = []

    row_starts = []
    pos = 0
    for i, span in enumerate(row_sizes):
        row_starts.append(pos)
        pos += span + (1 if i < size - 1 else 0)

    col_starts = []
    pos = 0
    for i, span in enumerate(col_sizes):
        col_starts.append(pos)
        pos += span + (1 if i < size - 1 else 0)

    for rr, r_span in enumerate(row_sizes):
        for cc, c_span in enumerate(col_sizes):
            color = int(reps[rr, cc])
            rep_list.append(color)
            r0 = row_starts[rr]
            c0 = col_starts[cc]
            patch = _render_subcell(r_span, c_span, color, colors, allow_noise)
            block[r0 : r0 + r_span, c0 : c0 + c_span] = patch

    return block, rep_list


def _direct_block_majority(block: np.ndarray, first_pos: dict[int, int]) -> int:
    colors = [int(value) for value in block.reshape(-1) if int(value) not in (BLACK, BLUE, GREY)]
    assert colors
    return _majority_with_tiebreak(colors, first_pos)


def _oracle_output_from_spec(
    block_reps: list[list[list[int]]],
    mapping: dict[int, int],
    first_pos: dict[int, int],
) -> np.ndarray:
    out = np.zeros((len(block_reps), len(block_reps[0])), dtype=int)
    for r, row in enumerate(block_reps):
        for c, reps in enumerate(row):
            agg = _majority_with_tiebreak(reps, first_pos)
            out[r, c] = int(mapping[agg])
    return out


def _sample_coarse_count() -> int:
    return random.choices([3, 4], weights=[7, 4], k=1)[0]


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_colors = random.choices([2, 3, 4], weights=[2, 3, 3], k=1)[0]
    colors = tuple(random.sample(PALETTE, n_colors))
    mapping = _derangement(colors)
    legend_rows = list(colors)
    random.shuffle(legend_rows)
    legend = _render_legend(legend_rows, mapping)
    precedence = _precedence_from_legend(legend, colors)

    n_blocks = _sample_coarse_count()
    allow_anti = n_blocks == 3
    row_max_sum = 28 - len(colors) - n_blocks
    col_max_sum = 26 - n_blocks

    row_sizes = _sample_sizes(
        n_blocks,
        5,
        8 if n_blocks == 3 else 6,
        row_max_sum,
        require_eight=allow_anti,
        min_big_count=2 if n_blocks == 3 else 0,
    )
    col_sizes = _sample_sizes(
        n_blocks,
        5,
        8 if n_blocks == 3 else 6,
        col_max_sum,
        require_eight=allow_anti,
        min_big_count=2 if n_blocks == 3 else 0,
    )
    if row_sizes is None or col_sizes is None:
        return None

    row_seps = [0, len(colors) + 1]
    pos = len(colors) + 1
    for span in row_sizes:
        pos += span + 1
        row_seps.append(pos)

    col_seps = [0, 3]
    pos = 3
    for span in col_sizes:
        pos += span + 1
        col_seps.append(pos)

    height = row_seps[-1] + 1
    width = col_seps[-1] + 1
    assert max(height, width) <= 30

    grid = np.full((height, width), BLACK, dtype=int)
    grid[row_seps, :] = GREY
    grid[:, col_seps] = GREY
    grid[: legend.shape[0], : legend.shape[1]] = legend

    block_targets = np.array(
        [random.choice(colors) for _ in range(n_blocks * n_blocks)],
        dtype=int,
    ).reshape(n_blocks, n_blocks)
    forced_positions = random.sample(range(n_blocks * n_blocks), len(colors))
    for idx, color in zip(forced_positions, colors):
        block_targets[idx // n_blocks, idx % n_blocks] = int(color)

    anti_index: tuple[int, int] | None = None
    if allow_anti:
        tall = [i for i, span in enumerate(row_sizes) if span >= 8]
        wide = [j for j, span in enumerate(col_sizes) if span >= 8]
        if tall and wide:
            anti_index = (random.choice(tall), random.choice(wide))

    three_index: tuple[int, int] | None = None
    if n_blocks == 3 and random.random() < 0.75:
        eligible_three = [
            (r, c)
            for r in range(n_blocks)
            for c in range(n_blocks)
            if row_sizes[r] >= 8 and col_sizes[c] >= 8 and (r, c) != anti_index
        ]
        if eligible_three:
            three_index = random.choice(eligible_three)

    tie_index: tuple[int, int] | None = None
    two_by_two_candidates = []
    for r in range(n_blocks):
        for c in range(n_blocks):
            if anti_index == (r, c):
                continue
            block_h = row_sizes[r]
            block_w = col_sizes[c]
            if min(block_h, block_w) >= 5:
                two_by_two_candidates.append((r, c))
    if two_by_two_candidates and random.random() < 0.7:
        tie_index = random.choice(two_by_two_candidates)

    block_reps: list[list[list[int]]] = [[[] for _ in range(n_blocks)] for _ in range(n_blocks)]
    saw_noise = False
    saw_three_by_three = False

    for br in range(n_blocks):
        for bc in range(n_blocks):
            block_h = int(row_sizes[br])
            block_w = int(col_sizes[bc])
            target = int(block_targets[br, bc])

            if (br, bc) == anti_index:
                size = 2
                big_choices = [color for color in colors if color != target]
                reps = _anti_shortcut_matrix(target, random.choice(big_choices))
                skew = True
                allow_noise = False
            elif (br, bc) == tie_index:
                size = 2
                later = [color for color in precedence if precedence.index(color) > precedence.index(target)]
                if not later:
                    target = int(precedence[0])
                    block_targets[br, bc] = target
                    later = [color for color in precedence[1:]]
                reps = _tie_matrix(target, random.choice(later))
                skew = False
                allow_noise = True
            elif (br, bc) == three_index:
                size = 3
                reps = _majority_matrix(target, colors, size)
                skew = False
                allow_noise = True
                saw_three_by_three = True
            else:
                can_three = n_blocks == 3 and min(block_h, block_w) >= 8
                size = 3 if can_three and random.random() < 0.45 else 2
                reps = _majority_matrix(target, colors, size)
                skew = size == 2 and random.random() < 0.35
                allow_noise = True
                saw_three_by_three = saw_three_by_three or size == 3

            block, reps_list = _render_block(block_h, block_w, reps, colors, skew=skew, allow_noise=allow_noise)
            if allow_noise and np.any(block != reps.repeat(1, axis=0)[0, 0]):
                saw_noise = saw_noise or np.any((block != BLUE) & (block != reps[0, 0]))
            block_reps[br][bc] = reps_list

            r0 = row_seps[br + 1] + 1
            c0 = col_seps[bc + 1] + 1
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = block

    first_pos = _first_occurrence(grid)
    expected = _oracle_output_from_spec(block_reps, mapping, first_pos)

    if set(map(int, np.unique(expected))) == {mapping[next(iter(mapping))]}:
        return None

    if anti_index is not None:
        ar, ac = anti_index
        block = grid[
            row_seps[ar + 1] + 1 : row_seps[ar + 2],
            col_seps[ac + 1] + 1 : col_seps[ac + 2],
        ]
        aggregate = _majority_with_tiebreak(block_reps[ar][ac], first_pos)
        direct = _direct_block_majority(block, first_pos)
        if aggregate == direct:
            return None

    direct_out = np.zeros_like(expected)
    for br in range(n_blocks):
        for bc in range(n_blocks):
            block = grid[
                row_seps[br + 1] + 1 : row_seps[br + 2],
                col_seps[bc + 1] + 1 : col_seps[bc + 2],
            ]
            direct_out[br, bc] = mapping[_direct_block_majority(block, first_pos)]
    if anti_index is not None and np.array_equal(direct_out, expected):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if set(map(int, expected.reshape(-1))) != {mapping[color] for color in set(map(int, block_targets.reshape(-1)))}:
        return None
    if n_blocks == 3 and not saw_three_by_three and random.random() < 0.4:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid double-majority legend-remap ARC input")
