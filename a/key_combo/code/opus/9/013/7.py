from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 8
BLOCK = 3
MOTIF_BLOCKS = 3
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
EXPANDED_BLOCKS = 5
EXPANDED_SIZE = BLOCK * EXPANDED_BLOCKS
PALETTE = (1, 2, 3, 4, 5, 6)
NON_CENTER_SLOTS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)
EDGE_SLOTS = ((0, 1), (1, 0), (1, 2), (2, 1))
CORNER_SLOTS = ((0, 0), (0, 2), (2, 0), (2, 2))
OUTER_TARGETS = {
    (0, 0): ((4, 4),),
    (0, 1): ((4, 1), (4, 2), (4, 3)),
    (0, 2): ((4, 0),),
    (1, 0): ((1, 4), (2, 4), (3, 4)),
    (1, 2): ((1, 0), (2, 0), (3, 0)),
    (2, 0): ((0, 4),),
    (2, 1): ((0, 1), (0, 2), (0, 3)),
    (2, 2): ((0, 0),),
}
INNER_SAMPLE_ROWS = np.array([2, 4, 6])
INNER_SAMPLE_COLS = np.array([2, 4, 6])
GRID_MIN = 18
GRID_MAX = 26
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _block_colors(block: np.ndarray, bg: int) -> tuple[int, int]:
    colors = sorted(set(map(int, block.reshape(-1))) - {int(bg)})
    assert len(colors) == 2, "each 3x3 block must contain exactly two foreground colors"
    return int(colors[0]), int(colors[1])


def _swap_block_colors(block: np.ndarray, bg: int) -> np.ndarray:
    a, b = _block_colors(block, bg)
    swapped = block.copy()
    swapped[block == a] = b
    swapped[block == b] = a
    return swapped


def _add_counts(counter: Counter[int], patch: np.ndarray, bg: int) -> None:
    values, counts = np.unique(patch[patch != int(bg)], return_counts=True)
    for color, count in zip(values, counts):
        counter[int(color)] += int(count)


def _unique_majority(counter: Counter[int]) -> int:
    ranked = counter.most_common()
    assert ranked, "expected at least one foreground color"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _unique_majority_or_none(counter: Counter[int]) -> int | None:
    ranked = counter.most_common()
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return int(ranked[0][0])


def _extract_motif(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    box = bbox(grid != int(bg))
    assert box is not None, "expected one foreground motif"
    motif = crop_bbox(grid, box)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE), "expected a 9x9 motif crop"
    assert np.all(motif[BLOCK : 2 * BLOCK, BLOCK : 2 * BLOCK] == int(bg)), "expected an empty center block"
    return box, motif


def _extract_blocks(motif: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for br, bc in NON_CENTER_SLOTS:
        block = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK].copy()
        blocks[(br, bc)] = block
    return blocks


def _vote_counter(blocks: dict[tuple[int, int], np.ndarray], bg: int, *, swap_outer: bool) -> Counter[int]:
    counter: Counter[int] = Counter()
    for slot in NON_CENTER_SLOTS:
        src = blocks[slot]
        _add_counts(counter, src, bg)
        outer_patch = _swap_block_colors(src, bg) if swap_outer else src
        for _dst in OUTER_TARGETS[slot]:
            _add_counts(counter, outer_patch, bg)
    return counter


def _center_patch(motif: np.ndarray, center_color: int) -> np.ndarray:
    patch = motif[np.ix_(INNER_SAMPLE_ROWS, INNER_SAMPLE_COLS)].copy()
    patch[1, 1] = int(center_color)
    return patch


def _render_expected(grid: np.ndarray, *, swap_outer: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    (r0, c0, _r1, _c1), motif = _extract_motif(g, bg)
    blocks = _extract_blocks(motif)
    vote_color = _unique_majority(_vote_counter(blocks, bg, swap_outer=swap_outer))

    out = g.copy()
    expanded_r0 = r0 - BLOCK
    expanded_c0 = c0 - BLOCK
    for slot in NON_CENTER_SLOTS:
        br, bc = slot
        src = blocks[slot]
        paste(out, src, r0 + br * BLOCK, c0 + bc * BLOCK)
        outer_patch = _swap_block_colors(src, bg) if swap_outer else src
        for dst_br, dst_bc in OUTER_TARGETS[slot]:
            paste(out, outer_patch, expanded_r0 + dst_br * BLOCK, expanded_c0 + dst_bc * BLOCK)
    paste(out, _center_patch(motif, vote_color), r0 + BLOCK, c0 + BLOCK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the hollow 3x3 block array into a full 5x5 array, swap colors only on the new outer ring, and rebuild the center.

    Input structure:
    - Teal (8) is the background.
    - One 9x9 foreground motif is a 3x3 array of 3x3 blocks with an empty teal center block.
    - Each of the eight visible 3x3 blocks is fully filled and uses exactly two non-teal colors.

    Rule:
    - Keep the original 3x3 block ring in place.
    - Copy each corner block to the opposite outer corner, and copy each edge block across the entire opposite outer side of the implied 5x5 block grid.
    - Swap the two foreground colors inside every copied outer block, but do not alter the inner eight source blocks.
    - Rebuild the missing center block: its eight border cells are read directly from the eight pixels touching the hole, and its middle cell is the unique most frequent foreground color across all 24 non-center blocks after the outer copies have been color-swapped.

    Key insight:
    - The output is determined by two interacting levels: block-level symmetric expansion creates a color-negative outer ring, and that swapped outer ring changes the global vote that fixes the center pixel.
    """
    return _render_expected(grid, swap_outer=True)


def _sample_block(
    colors: tuple[int, int],
    dominant: int,
    dominant_count: int,
    seen: set[tuple[int, ...]] | None = None,
) -> np.ndarray:
    a, b = map(int, colors)
    other = b if int(dominant) == a else a
    dominant_count = int(dominant_count)
    assert 1 <= 9 - dominant_count <= 8

    positions = list(range(9))
    for _ in range(80):
        dominant_cells = set(random.sample(positions, dominant_count))
        flat = [other] * 9
        for idx in dominant_cells:
            flat[idx] = int(dominant)
        key = tuple(flat)
        if seen is not None and key in seen:
            continue
        arr = np.array(flat, dtype=int).reshape(BLOCK, BLOCK)
        if len(set(flat)) != 2:
            continue
        if seen is not None:
            seen.add(key)
        return arr
    raise RuntimeError("failed to sample a distinct two-color block")


def _assemble_input_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for (br, bc), block in blocks.items():
        paste(motif, block, br * BLOCK, bc * BLOCK)
    return motif


def _visible_majority_color(blocks: dict[tuple[int, int], np.ndarray]) -> int:
    counter: Counter[int] = Counter()
    for slot in NON_CENTER_SLOTS:
        _add_counts(counter, blocks[slot], BG)
    winner = _unique_majority_or_none(counter)
    if winner is None:
        raise RuntimeError("visible inner-ring majority should be unique")
    return winner


def _generate_blocks() -> dict[tuple[int, int], np.ndarray]:
    colors = tuple(random.sample(PALETTE, 2))
    edge_dominant = random.choice(colors)
    center_color = colors[1] if edge_dominant == colors[0] else colors[0]
    corner_mode = random.choice(("support_edge", "support_center", "mixed"))

    blocks: dict[tuple[int, int], np.ndarray] = {}
    seen: set[tuple[int, ...]] = set()

    for slot in EDGE_SLOTS:
        dominant_count = random.randint(5, 7)
        blocks[slot] = _sample_block(colors, edge_dominant, dominant_count, seen)

    for slot in CORNER_SLOTS:
        if corner_mode == "support_edge":
            dominant = edge_dominant
            dominant_count = random.randint(5, 7)
        elif corner_mode == "support_center":
            dominant = center_color
            dominant_count = random.randint(6, 8)
        else:
            dominant = random.choice(colors)
            dominant_count = random.randint(5, 8)
        blocks[slot] = _sample_block(colors, dominant, dominant_count, seen)

    swap_winner = _unique_majority_or_none(_vote_counter(blocks, BG, swap_outer=True))
    no_swap_winner = _unique_majority_or_none(_vote_counter(blocks, BG, swap_outer=False))
    if swap_winner is None or no_swap_winner is None:
        raise RuntimeError("vote winner should be unique with and without the outer swap")
    if swap_winner == no_swap_winner:
        raise RuntimeError("outer swap must change the winning vote color")

    if swap_winner != center_color:
        raise RuntimeError("edge blocks must determine the post-swap center color")

    visible_winner = _visible_majority_color(blocks)
    if corner_mode == "support_edge" and visible_winner != edge_dominant:
        raise RuntimeError("support_edge mode should keep the visible majority on the edge-dominant color")
    if corner_mode == "support_center" and visible_winner != center_color:
        raise RuntimeError("support_center mode should flip the visible majority")

    boundary = _assemble_input_motif(blocks)[np.ix_(INNER_SAMPLE_ROWS, INNER_SAMPLE_COLS)]
    if len(set(map(int, boundary.reshape(-1)))) < 2:
        raise RuntimeError("center boundary encoding should use at least two visible colors")

    return blocks


def _validate_input_structure(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[tuple[int, int], np.ndarray]]:
    bg = get_bg(grid)
    assert bg == BG, "generator must use teal as the background"
    box, motif = _extract_motif(grid, bg)
    r0, c0, r1, c1 = box
    assert r0 >= BLOCK and c0 >= BLOCK
    assert r1 + BLOCK < grid.shape[0] and c1 + BLOCK < grid.shape[1]

    blocks = _extract_blocks(motif)
    all_pairs = set()
    for slot in NON_CENTER_SLOTS:
        block = blocks[slot]
        colors = _block_colors(block, bg)
        all_pairs.add(colors)
        assert np.all(block != bg), "source blocks must be fully filled"
        assert not np.array_equal(block, _swap_block_colors(block, bg)), "swapping colors must change the block"
    assert len(all_pairs) == 1, "generator should use one visible color pair so the swap effect is comparable"
    assert _unique_majority(_vote_counter(blocks, bg, swap_outer=True)) != _unique_majority(
        _vote_counter(blocks, bg, swap_outer=False)
    ), "generator must make the outer swap affect the center vote"
    return box, blocks


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape == out.shape
    bg = get_bg(inp)
    (r0, c0, r1, c1), blocks = _validate_input_structure(inp)
    expected = _render_expected(inp, swap_outer=True)
    assert np.array_equal(out, expected), "main output does not match the semantic oracle"

    expanded_r0 = r0 - BLOCK
    expanded_c0 = c0 - BLOCK
    expanded = out[expanded_r0 : expanded_r0 + EXPANDED_SIZE, expanded_c0 : expanded_c0 + EXPANDED_SIZE]
    assert expanded.shape == (EXPANDED_SIZE, EXPANDED_SIZE)
    assert np.all(expanded != bg), "the expanded 15x15 motif should be fully filled"

    for slot in NON_CENTER_SLOTS:
        br, bc = slot
        inner = out[r0 + br * BLOCK : r0 + (br + 1) * BLOCK, c0 + bc * BLOCK : c0 + (bc + 1) * BLOCK]
        assert np.array_equal(inner, blocks[slot]), "inner source blocks must remain unchanged"
        swapped = _swap_block_colors(blocks[slot], bg)
        for dst_br, dst_bc in OUTER_TARGETS[slot]:
            outer = out[
                expanded_r0 + dst_br * BLOCK : expanded_r0 + (dst_br + 1) * BLOCK,
                expanded_c0 + dst_bc * BLOCK : expanded_c0 + (dst_bc + 1) * BLOCK,
            ]
            assert np.array_equal(outer, swapped), "outer ring blocks must be swapped copies"

    motif = crop_bbox(inp, (r0, c0, r1, c1))
    center = out[r0 + BLOCK : r0 + 2 * BLOCK, c0 + BLOCK : c0 + 2 * BLOCK]
    assert np.array_equal(center[[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]], motif[
        [2, 2, 2, 4, 4, 6, 6, 6], [2, 4, 6, 2, 6, 2, 4, 6]
    ]), "center border must come from the visible inner boundary"
    vote_color = _unique_majority(_vote_counter(blocks, bg, swap_outer=True))
    assert int(center[1, 1]) == vote_color, "center pixel must equal the post-swap global majority color"


def _generate_attempt() -> np.ndarray | None:
    try:
        blocks = _generate_blocks()
    except RuntimeError:
        return None

    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)

    max_r0 = height - (MOTIF_SIZE + BLOCK)
    max_c0 = width - (MOTIF_SIZE + BLOCK)
    if max_r0 < BLOCK or max_c0 < BLOCK:
        return None

    r0 = random.randint(BLOCK, max_r0)
    c0 = random.randint(BLOCK, max_c0)
    paste(grid, _assemble_input_motif(blocks), r0, c0)

    try:
        out = main(grid)
        _validate_output_semantics(grid, out)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
