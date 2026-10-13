from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SEP = 5
BLOCK_SIZE = 5
CHAIN_LENGTH = 4
CHAIN_COLORS = (1, 2, 3, 4)
INTERIOR_OFFSETS = (1, 2, 3)
ALL_PERMS = tuple(itertools.permutations(range(CHAIN_LENGTH)))
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_contiguous(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _split_blocks(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    sep_rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == SEP))]
    sep_cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == SEP))]

    if sep_cols and not sep_rows:
        bounds = [-1, *sep_cols, grid.shape[1]]
        blocks = [grid[:, bounds[i] + 1 : bounds[i + 1]].copy() for i in range(len(bounds) - 1)]
        assert len(blocks) == CHAIN_LENGTH, "expected four horizontal blocks"
        assert all(block.shape == (BLOCK_SIZE, BLOCK_SIZE) for block in blocks), "expected 5x5 blocks"
        return "h", blocks

    if sep_rows and not sep_cols:
        bounds = [-1, *sep_rows, grid.shape[0]]
        blocks = [grid[bounds[i] + 1 : bounds[i + 1], :].copy() for i in range(len(bounds) - 1)]
        assert len(blocks) == CHAIN_LENGTH, "expected four vertical blocks"
        assert all(block.shape == (BLOCK_SIZE, BLOCK_SIZE) for block in blocks), "expected 5x5 blocks"
        return "v", blocks

    raise AssertionError("expected a single row or column of four 5x5 blocks separated by grey lines")


def _main_color(block: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(block))) - {BG, SEP})
    assert len(colors) == 1, "each block must contain exactly one foreground color"
    return int(colors[0])


def _edge_offsets(block: np.ndarray, side: str) -> set[int]:
    mask = (block != BG) & (block != SEP)
    if side == "left":
        return {int(r) for r in np.where(mask[:, 0])[0]}
    if side == "right":
        return {int(r) for r in np.where(mask[:, -1])[0]}
    if side == "top":
        return {int(c) for c in np.where(mask[0, :])[0]}
    if side == "bottom":
        return {int(c) for c in np.where(mask[-1, :])[0]}
    raise ValueError(f"unknown side: {side}")


def _connects(left: np.ndarray, right: np.ndarray, orientation: str) -> bool:
    if orientation == "h":
        return bool(_edge_offsets(left, "right") & _edge_offsets(right, "left"))
    return bool(_edge_offsets(left, "bottom") & _edge_offsets(right, "top"))


def _all_chains(blocks: list[np.ndarray], orientation: str) -> list[tuple[int, ...]]:
    chains: list[tuple[int, ...]] = []
    for perm in ALL_PERMS:
        if all(_connects(blocks[perm[i]], blocks[perm[i + 1]], orientation) for i in range(CHAIN_LENGTH - 1)):
            chains.append(perm)
    return chains


def _render_output(blocks: list[np.ndarray], chain: tuple[int, ...]) -> np.ndarray:
    out = np.full((BLOCK_SIZE, CHAIN_LENGTH * BLOCK_SIZE + (CHAIN_LENGTH - 1)), SEP, dtype=int)
    for pos, block_index in enumerate(chain):
        block = blocks[block_index].copy()
        fill_color = _main_color(blocks[chain[(pos + 1) % CHAIN_LENGTH]])
        block[block == BG] = fill_color
        c0 = pos * (BLOCK_SIZE + 1)
        out[:, c0 : c0 + BLOCK_SIZE] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the four sprite blocks into their unique edge-matching chain, then fill each block with its successor's color.

    Input structure:
    - The grid contains four 5x5 sprite blocks on black background, separated by full grey divider lines.
    - The blocks appear in a single horizontal row or a single vertical column.
    - Each block has exactly one non-black, non-grey foreground color and a 4-connected shape with visible black gaps.

    Rule:
    - Detect the four blocks and determine the layout orientation from the grey separator lines.
    - Compare facing edges in that orientation: a block can precede another when their touching-edge foreground offsets overlap in at least one position.
    - Find the unique chain order covering all four blocks, output the blocks in that order as one horizontal strip with grey separators, and fill each block's black cells with the main color of the next block in the chain, wrapping from the last block back to the first.

    Key insight:
    - The fill color is not an intrinsic property of a block; it is induced by the unique successor relation produced by the edge-matching chain order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orientation, blocks = _split_blocks(g)
    chains = _all_chains(blocks, orientation)
    assert len(chains) == 1, "expected a unique valid chain"
    return _render_output(blocks, chains[0])


def _frontier_interior(mask: np.ndarray) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in map(tuple, np.argwhere(mask)):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 1 <= nr <= 3 and 1 <= nc <= 3 and not mask[nr, nc]:
                frontier.add((nr, nc))
    return sorted(frontier)


def _grow_interior(mask: np.ndarray, target_area: int) -> bool:
    while int(mask.sum()) < target_area:
        frontier = _frontier_interior(mask)
        if not frontier:
            return False
        r, c = random.choice(frontier)
        mask[r, c] = True
    return True


def _horizontal_bridge(left_row: int, right_row: int) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    mask[left_row, 0:3] = True
    lo, hi = sorted((left_row, right_row))
    mask[lo : hi + 1, 2] = True
    mask[right_row, 2:5] = True
    return mask


def _horizontal_stub(side: str, edge_row: int, anchor_row: int) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    lo, hi = sorted((edge_row, anchor_row))
    mask[lo : hi + 1, 2] = True
    if side == "left":
        mask[edge_row, 0:3] = True
    else:
        mask[edge_row, 2:5] = True
    return mask


def _vertical_bridge(top_col: int, bottom_col: int) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    mask[0:3, top_col] = True
    lo, hi = sorted((top_col, bottom_col))
    mask[2, lo : hi + 1] = True
    mask[2:5, bottom_col] = True
    return mask


def _vertical_stub(side: str, edge_col: int, anchor_col: int) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    lo, hi = sorted((edge_col, anchor_col))
    mask[2, lo : hi + 1] = True
    if side == "top":
        mask[0:3, edge_col] = True
    else:
        mask[2:5, edge_col] = True
    return mask


def _random_target_area(base_area: int) -> int:
    return random.randint(base_area, min(6, base_area + 2))


def _make_mask(orientation: str, incoming: int | None, outgoing: int | None) -> np.ndarray:
    if orientation == "h":
        if incoming is None:
            anchor = random.choice([incoming_row for incoming_row in INTERIOR_OFFSETS if abs(incoming_row - outgoing) <= 1])
            mask = _horizontal_stub("right", int(outgoing), int(anchor))
        elif outgoing is None:
            anchor = random.choice([outgoing_row for outgoing_row in INTERIOR_OFFSETS if abs(outgoing_row - incoming) <= 1])
            mask = _horizontal_stub("left", int(incoming), int(anchor))
        else:
            mask = _horizontal_bridge(int(incoming), int(outgoing))
    else:
        if incoming is None:
            anchor = random.choice([incoming_col for incoming_col in INTERIOR_OFFSETS if abs(incoming_col - outgoing) <= 1])
            mask = _vertical_stub("bottom", int(outgoing), int(anchor))
        elif outgoing is None:
            anchor = random.choice([outgoing_col for outgoing_col in INTERIOR_OFFSETS if abs(outgoing_col - incoming) <= 1])
            mask = _vertical_stub("top", int(incoming), int(anchor))
        else:
            mask = _vertical_bridge(int(incoming), int(outgoing))

    target_area = _random_target_area(int(mask.sum()))
    assert _grow_interior(mask, target_area)
    assert 3 <= int(mask.sum()) <= 6
    assert _is_contiguous(mask)
    return mask


def _paint_block(mask: np.ndarray, color: int) -> np.ndarray:
    block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    block[mask] = int(color)
    return block


def _assemble_input(blocks: list[np.ndarray], orientation: str) -> np.ndarray:
    if orientation == "h":
        out = np.full((BLOCK_SIZE, CHAIN_LENGTH * BLOCK_SIZE + (CHAIN_LENGTH - 1)), BG, dtype=int)
        for i, block in enumerate(blocks):
            c0 = i * (BLOCK_SIZE + 1)
            out[:, c0 : c0 + BLOCK_SIZE] = block
            if i < CHAIN_LENGTH - 1:
                out[:, c0 + BLOCK_SIZE] = SEP
        return out

    out = np.full((CHAIN_LENGTH * BLOCK_SIZE + (CHAIN_LENGTH - 1), BLOCK_SIZE), BG, dtype=int)
    for i, block in enumerate(blocks):
        r0 = i * (BLOCK_SIZE + 1)
        out[r0 : r0 + BLOCK_SIZE, :] = block
        if i < CHAIN_LENGTH - 1:
            out[r0 + BLOCK_SIZE, :] = SEP
    return out


def _inverse_order(order: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(order.index(i) for i in range(CHAIN_LENGTH))


def _validate_block(block: np.ndarray) -> None:
    colors = sorted(set(map(int, np.unique(block))) - {BG})
    assert len(colors) == 1, "generator must create monochromatic blocks"
    assert int(np.count_nonzero(block == BG)) >= 10, "each block must keep at least 40% black cells"
    mask = block != BG
    assert 3 <= int(mask.sum()) <= 6, "foreground area must stay in the target range"
    assert _is_contiguous(mask), "foreground must be 4-connected"


def _validate_instance(
    grid: np.ndarray,
    orientation: str,
    expected_output: np.ndarray,
    input_order: tuple[int, ...],
) -> bool:
    try:
        parsed_orientation, parsed_blocks = _split_blocks(grid)
        if parsed_orientation != orientation:
            return False
        for block in parsed_blocks:
            _validate_block(block)

        input_colors = [_main_color(block) for block in parsed_blocks]
        if sorted(input_colors) != list(CHAIN_COLORS):
            return False

        chains = _all_chains(parsed_blocks, orientation)
        if len(chains) != 1:
            return False
        if chains[0] != _inverse_order(input_order):
            return False
        if chains[0] == tuple(range(CHAIN_LENGTH)):
            return False

        other_orientation = "v" if orientation == "h" else "h"
        if _all_chains(parsed_blocks, other_orientation):
            return False

        out = main(grid)
        if not np.array_equal(out, expected_output):
            return False

        # Check the intended semantics directly instead of only trusting the oracle round-trip.
        for pos, block_index in enumerate(chains[0]):
            c0 = pos * (BLOCK_SIZE + 1)
            out_block = out[:, c0 : c0 + BLOCK_SIZE]
            src_block = parsed_blocks[block_index]
            own_color = _main_color(src_block)
            next_color = _main_color(parsed_blocks[chains[0][(pos + 1) % CHAIN_LENGTH]])
            fg_mask = src_block != BG

            if own_color == next_color:
                return False
            if np.any(out_block[fg_mask] != own_color):
                return False
            if np.any(out_block[~fg_mask] != next_color):
                return False
        return True
    except AssertionError:
        return False


def _make_chain_blocks(orientation: str) -> list[np.ndarray]:
    tokens = random.choice(((1, 2, 3), (3, 2, 1)))
    colors = random.sample(list(CHAIN_COLORS), CHAIN_LENGTH)
    specs = (
        (None, tokens[0]),
        (tokens[0], tokens[1]),
        (tokens[1], tokens[2]),
        (tokens[2], None),
    )
    blocks = [_paint_block(_make_mask(orientation, incoming, outgoing), color) for (incoming, outgoing), color in zip(specs, colors)]
    for block in blocks:
        _validate_block(block)
    return blocks


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        orientation = random.choice(("h", "v"))
        chain_blocks = _make_chain_blocks(orientation)
        input_order = random.choice([perm for perm in ALL_PERMS if perm != tuple(range(CHAIN_LENGTH))])
        scrambled_blocks = [chain_blocks[i] for i in input_order]
        grid = _assemble_input(scrambled_blocks, orientation)
        expected_output = _render_output(scrambled_blocks, _inverse_order(input_order))
        if _validate_instance(grid, orientation, expected_output, input_order):
            return grid
    raise RuntimeError("failed to generate a valid edge-chain borrowing instance")
