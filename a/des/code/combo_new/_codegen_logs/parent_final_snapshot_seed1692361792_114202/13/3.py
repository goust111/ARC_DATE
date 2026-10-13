from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = tuple(color for color in range(10) if color not in (BLACK, GREY))

REFERENCE_HEIGHT_RANGE = (3, 4)
REFERENCE_WIDTH_RANGE = (12, 26)
BLOCK_COUNT_RANGE = (3, 8)
BLOCK_LEN_RANGE = (3, 5)
HORIZONTAL_MARGIN = 1
HORIZONTAL_GAP = 1
TOP_GAP_BELOW_SEPARATOR = 1
BOTTOM_GAP = 1
ROW_STEP_OPTIONS = (2, 3, 4)
MAX_REFERENCE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _find_separator_row(g: np.ndarray) -> int:
    rows = [
        r
        for r in range(g.shape[0])
        if int(g[r, 0]) != BLACK and np.all(g[r, :] == g[r, 0])
    ]
    assert rows, "expected a full-width separator row"
    return int(rows[0])


def _extract_blocks(lower: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    mask = lower != BLACK
    blocks: list[tuple[int, int, np.ndarray]] = []
    for r in range(lower.shape[0]):
        c = 0
        while c < lower.shape[1]:
            if not mask[r, c]:
                c += 1
                continue
            start = c
            while c < lower.shape[1] and mask[r, c]:
                c += 1
            end = c
            if r > 0:
                assert not np.any(mask[r - 1, start:end]), "block copies must be one row tall"
            if r + 1 < lower.shape[0]:
                assert not np.any(mask[r + 1, start:end]), "block copies must be one row tall"
            blocks.append((int(r), int(start), lower[r, start:end].copy()))
    assert blocks, "expected at least one lookup strip"
    blocks.sort(key=lambda item: (item[0], item[1]))
    return blocks


def _match_start(reference: np.ndarray, block: np.ndarray) -> int:
    width = int(reference.shape[0])
    block_w = int(block.shape[0])
    hits = [
        start
        for start in range(width - block_w)
        if np.array_equal(reference[start : start + block_w], block)
    ]
    assert len(hits) == 1, "each lookup strip must match exactly one reference segment"
    return int(hits[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use scattered strip copies as lookup keys to read the next color from a striped reference band.

    Input structure:
    - The background is black.
    - The top of the grid contains a full-width horizontal reference band, 3-4 rows tall, whose rows are identical vertical color stripes.
    - A single full-width grey separator row sits directly below that band.
    - Below the separator are several scattered 1xN strip copies (N = 3-5), each copied exactly from one contiguous segment of the reference band.

    Rule:
    - Read the reference pattern from any row of the top band.
    - For each scattered strip, find the unique matching segment in the reference band.
    - Take the color immediately to the right of that matched segment in the reference band.
    - Order the strips in normal reading order: top to bottom, and left to right within each row.
    - Output a single-row grid whose cells are those extracted right-neighbor colors in that order.

    Key insight:
    - The scattered strips are only locators; the answer color is not inside the strip itself but in the next reference column after the matched segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row = _find_separator_row(g)
    assert sep_row >= 1, "reference band must be above the separator"
    band = g[:sep_row, :]
    assert np.all(band == band[0:1, :]), "reference band rows must be identical"
    reference = band[0].copy()

    lower = g[sep_row + 1 :, :]
    blocks = _extract_blocks(lower)

    out_colors = []
    for _row, _col, block in blocks:
        start = _match_start(reference, block)
        out_colors.append(int(reference[start + len(block)]))

    return np.asarray([out_colors], dtype=int)


def _positive_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(int(cut - prev))
        prev = cut
    return values


def _random_slack_parts(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _build_reference(width: int) -> np.ndarray | None:
    for _ in range(MAX_REFERENCE_ATTEMPTS):
        seq: list[int] = []
        seen = {length: set() for length in range(BLOCK_LEN_RANGE[0], BLOCK_LEN_RANGE[1] + 1)}
        while len(seq) < width:
            choices = []
            for color in random.sample(PALETTE, len(PALETTE)):
                if seq and color == seq[-1]:
                    continue
                ok = True
                for length in range(BLOCK_LEN_RANGE[0], BLOCK_LEN_RANGE[1] + 1):
                    if len(seq) + 1 < length:
                        continue
                    prefix = seq[-(length - 1) :] if length > 1 else []
                    window = tuple(prefix + [color])
                    if window in seen[length]:
                        ok = False
                        break
                if ok:
                    choices.append(int(color))
            if not choices:
                break
            picked = random.choice(choices)
            seq.append(int(picked))
            for length in range(BLOCK_LEN_RANGE[0], BLOCK_LEN_RANGE[1] + 1):
                if len(seq) >= length:
                    seen[length].add(tuple(seq[-length:]))
        if len(seq) == width:
            return np.asarray(seq, dtype=int)
    return None


def _row_run_width(lengths: list[int]) -> int:
    return sum(lengths) + HORIZONTAL_GAP * (len(lengths) - 1)


def _row_positions(width: int, lengths: list[int]) -> list[int] | None:
    needed = _row_run_width(lengths)
    inner_width = width - 2 * HORIZONTAL_MARGIN
    if needed > inner_width:
        return None
    extras = _random_slack_parts(inner_width - needed, len(lengths) + 1)
    col = HORIZONTAL_MARGIN + extras[0]
    positions = []
    for i, length in enumerate(lengths):
        positions.append(int(col))
        col += length
        if i + 1 < len(lengths):
            col += HORIZONTAL_GAP + extras[i + 1]
    return positions


def _choose_layout(width: int, n_blocks: int) -> tuple[list[int], list[int]] | None:
    row_count = random.randint(1, min(3, n_blocks))
    row_sizes = _positive_composition(n_blocks, row_count)
    lengths = [random.randint(*BLOCK_LEN_RANGE) for _ in range(n_blocks)]

    offset = 0
    for size in row_sizes:
        row_lengths = lengths[offset : offset + size]
        if _row_run_width(row_lengths) > width - 2 * HORIZONTAL_MARGIN:
            return None
        offset += size
    return row_sizes, lengths


def _render_blocks(
    lower: np.ndarray,
    row_offsets: list[int],
    row_sizes: list[int],
    blocks: list[np.ndarray],
) -> list[tuple[int, int]]:
    placements: list[tuple[int, int]] = []
    offset = 0
    for row_index, size in enumerate(row_sizes):
        row_blocks = blocks[offset : offset + size]
        lengths = [int(block.shape[0]) for block in row_blocks]
        cols = _row_positions(lower.shape[1], lengths)
        assert cols is not None
        r = row_offsets[row_index]
        for block, c in zip(row_blocks, cols):
            lower[r, c : c + block.shape[0]] = block
            placements.append((int(r), int(c)))
        offset += size
    return placements


def _expected_output(reference: np.ndarray, specs: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([[int(reference[start + length]) for start, length in specs]], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*REFERENCE_WIDTH_RANGE)
    n_blocks = random.randint(*BLOCK_COUNT_RANGE)
    layout = _choose_layout(width, n_blocks)
    if layout is None:
        return None
    row_sizes, lengths = layout

    reference = _build_reference(width)
    if reference is None:
        return None

    available_by_length = {
        length: [(start, length) for start in range(width - length)]
        for length in range(BLOCK_LEN_RANGE[0], BLOCK_LEN_RANGE[1] + 1)
    }
    chosen_specs: list[tuple[int, int]] = []
    for length in lengths:
        options = available_by_length[length]
        if not options:
            return None
        spec = random.choice(options)
        options.remove(spec)
        chosen_specs.append(spec)

    expected = _expected_output(reference, chosen_specs)
    if len(set(map(int, expected.reshape(-1)))) < 2:
        return None

    band_height = random.randint(*REFERENCE_HEIGHT_RANGE)
    row_offsets = [TOP_GAP_BELOW_SEPARATOR]
    for _ in range(1, len(row_sizes)):
        row_offsets.append(row_offsets[-1] + random.choice(ROW_STEP_OPTIONS))
    lower_height = row_offsets[-1] + 1 + BOTTOM_GAP
    total_height = band_height + 1 + lower_height
    if total_height > 30 or width > 30:
        return None

    grid = np.full((total_height, width), BLACK, dtype=int)
    for r in range(band_height):
        grid[r, :] = reference
    grid[band_height, :] = GREY

    blocks = [reference[start : start + length].copy() for start, length in chosen_specs]
    placements = _render_blocks(grid[band_height + 1 :, :], row_offsets, row_sizes, blocks)

    ordered_specs = [
        spec
        for _placement, spec in sorted(
            zip(placements, chosen_specs), key=lambda item: (item[0][0], item[0][1])
        )
    ]
    oracle = _expected_output(reference, ordered_specs)

    try:
        if not np.array_equal(main(grid), oracle):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-lookup ARC input")
