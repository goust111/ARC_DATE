from __future__ import annotations

import itertools
import random

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates = [
        r for r in range(grid.shape[0])
        if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != int(bg)
    ]
    assert len(candidates) == 1, "expected a unique non-background separator row"
    return int(candidates[0])


def _parse_blocks(row: np.ndarray, bg: int) -> list[dict[str, int]]:
    arr = np.asarray(row, dtype=int)
    blocks: list[dict[str, int]] = []
    c = 0
    width = arr.shape[0]
    while c < width:
        if int(arr[c]) == int(bg):
            c += 1
            continue
        color = int(arr[c])
        start = c
        while c < width and int(arr[c]) == color:
            c += 1
        end = c - 1
        blocks.append(
            {
                "color": color,
                "start": int(start),
                "end": int(end),
                "length": int(end - start + 1),
            }
        )

    assert blocks, "row must contain foreground blocks"
    colors = [block["color"] for block in blocks]
    assert len(colors) == len(set(colors)), "blocks must have distinct colors"
    for left, right in zip(blocks, blocks[1:]):
        assert right["start"] - left["end"] >= 2, "blocks must be separated by background"
    return blocks


def _gap_pattern(blocks: list[dict[str, int]]) -> tuple[int, ...]:
    return tuple(
        int(blocks[i + 1]["start"] - blocks[i]["end"] - 1)
        for i in range(len(blocks) - 1)
    )


def _extract_examples_and_queries(
    grid: np.ndarray, bg: int, separator_row: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    top_rows = [
        np.asarray(grid[r], dtype=int)
        for r in range(separator_row)
        if not np.all(grid[r] == int(bg))
    ]
    assert len(top_rows) >= 2 and len(top_rows) % 2 == 0, "top region must contain example pairs"
    examples = [(top_rows[i], top_rows[i + 1]) for i in range(0, len(top_rows), 2)]

    queries = [
        np.asarray(grid[r], dtype=int)
        for r in range(separator_row + 1, grid.shape[0])
        if not np.all(grid[r] == int(bg))
    ]
    assert queries, "bottom region must contain query rows"
    return examples, queries


def _infer_rule(examples: list[tuple[np.ndarray, np.ndarray]], bg: int) -> tuple[tuple[int, ...], tuple[int, ...], str, int]:
    permutation: tuple[int, ...] | None = None
    out_gaps: tuple[int, ...] | None = None
    block_count: int | None = None
    left_possible = True
    right_possible = True

    for input_row, output_row in examples:
        src = _parse_blocks(input_row, bg)
        dst = _parse_blocks(output_row, bg)
        assert len(src) == len(dst), "example input/output must have the same number of blocks"

        count = len(src)
        if block_count is None:
            block_count = count
        else:
            assert count == block_count, "all examples must use the same block count"

        src_colors = [block["color"] for block in src]
        dst_colors = [block["color"] for block in dst]
        assert sorted(src_colors) == sorted(dst_colors), "example rows must reuse the same block colors"
        current_permutation = tuple(src_colors.index(color) for color in dst_colors)
        if permutation is None:
            permutation = current_permutation
        else:
            assert current_permutation == permutation, "examples must agree on one permutation"

        current_out_gaps = _gap_pattern(dst)
        if out_gaps is None:
            out_gaps = current_out_gaps
        else:
            assert current_out_gaps == out_gaps, "examples must agree on output gaps"

        left_possible &= int(src[0]["start"]) == int(dst[0]["start"])
        right_possible &= int(src[-1]["end"]) == int(dst[-1]["end"])

        for src_idx, dst_block in zip(current_permutation, dst):
            src_block = src[src_idx]
            assert int(src_block["length"]) == int(dst_block["length"]), "block lengths must be preserved"

    assert permutation is not None and out_gaps is not None and block_count is not None
    assert len(out_gaps) == block_count - 1
    assert left_possible or right_possible, "could not infer anchoring"
    anchor = "left" if left_possible and not right_possible else "right"
    return permutation, out_gaps, anchor, block_count


def _paint_blocks(
    width: int,
    bg: int,
    blocks: list[dict[str, int]],
    permutation: tuple[int, ...],
    out_gaps: tuple[int, ...],
    anchor: str,
) -> np.ndarray:
    ordered = [blocks[idx] for idx in permutation]
    total_span = sum(block["length"] for block in ordered) + sum(out_gaps)
    if anchor == "left":
        start = int(blocks[0]["start"])
    else:
        end = int(blocks[-1]["end"])
        start = int(end - total_span + 1)

    assert 0 <= start < width, "transformed row must start inside the grid"
    out = np.full(width, int(bg), dtype=int)
    c = int(start)
    for i, block in enumerate(ordered):
        length = int(block["length"])
        assert c + length <= width, "transformed row must fit inside the grid"
        out[c : c + length] = int(block["color"])
        c += length
        if i < len(out_gaps):
            c += int(out_gaps[i])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a row-wise block-reordering rule from example pairs and apply it to query rows.

    Input structure:
    - One full separator row splits the grid into an upper example region and a lower query region.
    - Each example consists of an input row and its output row, with all-background spacer rows
      allowed between examples.
    - Every active row contains several distinct contiguous color blocks separated by background.

    Rule:
    - Read the example row pairs and infer:
      1. the permutation of block order,
      2. the gap pattern used in the transformed output rows,
      3. whether the transformed rows preserve the left or right boundary.
    - Apply that same row-level rule to every query row below the separator.

    Output:
    - Return only the transformed query rows, stacked without the separator or examples.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode_color(g)
    separator_row = _find_separator_row(g, bg)
    examples, queries = _extract_examples_and_queries(g, bg, separator_row)
    permutation, out_gaps, anchor, block_count = _infer_rule(examples, bg)

    outputs = []
    for row in queries:
        blocks = _parse_blocks(row, bg)
        assert len(blocks) == block_count, "query rows must match the learned block count"
        outputs.append(_paint_blocks(row.shape[0], bg, blocks, permutation, out_gaps, anchor))

    return np.asarray(outputs, dtype=int)


# --------------- generator ---------------

COLORS = tuple(range(10))
WIDTH_RANGE = (18, 30)
EXAMPLE_COUNT_RANGE = (2, 4)
QUERY_COUNT_RANGE = (1, 4)
BLOCK_COUNT_CHOICES = (3, 4)
BLOCK_LENGTH_RANGE = (1, 4)
PRESERVED_MARGIN_RANGE = (1, 5)
EXTRA_WIDTH_RANGE = (0, 4)
MAX_GENERATION_ATTEMPTS = 400


def _uniform_gap_pattern(block_count: int, gap_size: int) -> tuple[int, ...]:
    return tuple([int(gap_size)] * max(0, block_count - 1))


def _span(lengths: list[int], gaps: tuple[int, ...]) -> int:
    return int(sum(lengths) + sum(gaps))


def _build_row(
    width: int,
    bg: int,
    lengths: list[int],
    colors: list[int],
    gaps: tuple[int, ...],
    anchor: str,
    preserved_margin: int,
) -> np.ndarray:
    row = np.full(width, int(bg), dtype=int)
    span = _span(lengths, gaps)
    if anchor == "left":
        c = int(preserved_margin)
    else:
        c = int(width - preserved_margin - span)
    assert c >= 0 and c + span <= width

    for i, (length, color) in enumerate(zip(lengths, colors)):
        row[c : c + length] = int(color)
        c += int(length)
        if i < len(gaps):
            c += int(gaps[i])
    return row


def _oracle_accepts(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _background_is_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(value): int(count) for value, count in zip(values, counts)}
    bg_count = count_map.get(int(bg), 0)
    other_best = max((count for color, count in count_map.items() if color != int(bg)), default=-1)
    return bg_count > other_best


def _sample_row_specs(
    row_count: int,
    block_count: int,
    palette: list[int],
    in_gaps: tuple[int, ...],
    out_gaps: tuple[int, ...],
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for _ in range(row_count):
        lengths = [random.randint(*BLOCK_LENGTH_RANGE) for _ in range(block_count)]
        colors = random.sample(palette, block_count)
        preserved_margin = random.randint(*PRESERVED_MARGIN_RANGE)
        specs.append(
            {
                "lengths": lengths,
                "colors": colors,
                "preserved_margin": preserved_margin,
                "required_width": preserved_margin + max(_span(lengths, in_gaps), _span(lengths, out_gaps)) + 1,
            }
        )
    return specs


def _render_example_and_query_rows(
    width: int,
    bg: int,
    specs: list[dict[str, object]],
    permutation: tuple[int, ...],
    in_gaps: tuple[int, ...],
    out_gaps: tuple[int, ...],
    anchor: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    example_rows: list[np.ndarray] = []
    query_rows: list[np.ndarray] = []
    for spec in specs:
        lengths = list(spec["lengths"])
        colors = list(spec["colors"])
        margin = int(spec["preserved_margin"])

        inp = _build_row(width, bg, lengths, colors, in_gaps, anchor, margin)
        permuted_lengths = [lengths[i] for i in permutation]
        permuted_colors = [colors[i] for i in permutation]
        out = _build_row(width, bg, permuted_lengths, permuted_colors, out_gaps, anchor, margin)

        example_rows.append(inp)
        query_rows.append(out)
    return example_rows, query_rows


def _assemble_grid(
    width: int,
    bg: int,
    sep: int,
    example_inputs: list[np.ndarray],
    example_outputs: list[np.ndarray],
    query_inputs: list[np.ndarray],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for idx, (inp, out) in enumerate(zip(example_inputs, example_outputs)):
        rows.append(inp)
        rows.append(out)
        if idx != len(example_inputs) - 1:
            rows.append(np.full(width, int(bg), dtype=int))

    rows.append(np.full(width, int(sep), dtype=int))
    rows.extend(query_inputs)
    return np.asarray(rows, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    bg, sep = random.sample(COLORS, 2)
    palette = [color for color in COLORS if color not in {bg, sep}]

    block_count = random.choice(BLOCK_COUNT_CHOICES)
    permutations = [
        perm for perm in itertools.permutations(range(block_count))
        if perm != tuple(range(block_count))
    ]
    permutation = tuple(random.choice(permutations))

    in_gap_size, out_gap_size = random.choice([(1, 2), (2, 1)])
    in_gaps = _uniform_gap_pattern(block_count, in_gap_size)
    out_gaps = _uniform_gap_pattern(block_count, out_gap_size)
    anchor = random.choice(("left", "right"))

    example_count = random.randint(*EXAMPLE_COUNT_RANGE)
    query_count = random.randint(*QUERY_COUNT_RANGE)
    all_specs = _sample_row_specs(example_count + query_count, block_count, palette, in_gaps, out_gaps)
    min_width = max(int(spec["required_width"]) for spec in all_specs)
    if min_width > WIDTH_RANGE[1]:
        return None

    extra_width_max = min(EXTRA_WIDTH_RANGE[1], WIDTH_RANGE[1] - min_width)
    width = min_width + random.randint(EXTRA_WIDTH_RANGE[0], max(EXTRA_WIDTH_RANGE[0], extra_width_max))
    if width < WIDTH_RANGE[0]:
        if WIDTH_RANGE[0] > WIDTH_RANGE[1]:
            return None
        width = WIDTH_RANGE[0]
    if width > WIDTH_RANGE[1]:
        return None

    example_specs = all_specs[:example_count]
    query_specs = all_specs[example_count:]

    example_inputs, example_outputs = _render_example_and_query_rows(
        width, bg, example_specs, permutation, in_gaps, out_gaps, anchor
    )
    query_inputs, query_outputs = _render_example_and_query_rows(
        width, bg, query_specs, permutation, in_gaps, out_gaps, anchor
    )

    grid = _assemble_grid(width, bg, sep, example_inputs, example_outputs, query_inputs)
    expected = np.asarray(query_outputs, dtype=int)

    if not _background_is_mode(grid, bg):
        return None
    if not _oracle_accepts(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
