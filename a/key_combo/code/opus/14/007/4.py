from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
SEP = 5
BLOCK = 5
GEN_RETRIES = 400
CHAIN_ORIENTATIONS = ("h", "v")
RUN_LENGTHS = (1, 2, 3)
PALETTE_SIZES = (2, 3)
BLOCK_COUNTS = (3, 4, 5)
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
EDGE_OFFSETS = (1, 2, 3)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _foreground_mask(patch: np.ndarray, bg: int) -> np.ndarray:
    return (patch != int(bg)) & (patch != SEP)


def _is_contiguous(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return len(seen) == int(mask.sum())


def _edge_sets(patch: np.ndarray, bg: int) -> dict[str, set[int]]:
    mask = _foreground_mask(patch, bg)
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bottom": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _count_valid_orders(blocks: list[np.ndarray], orient: str, bg: int) -> tuple[int, tuple[int, ...] | None]:
    edges = [_edge_sets(block, bg) for block in blocks]
    if orient == "h":
        def ok(a: int, b: int) -> bool:
            return bool(edges[a]["right"] & edges[b]["left"])
    else:
        def ok(a: int, b: int) -> bool:
            return bool(edges[a]["bottom"] & edges[b]["top"])

    count = 0
    found: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(int(perm[i]), int(perm[i + 1])) for i in range(len(perm) - 1)):
            count += 1
            if found is None:
                found = tuple(map(int, perm))
    return count, found


def _find_marker_and_palette(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    H, W = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    for c in range(W):
        rows = np.where(grid[:, c] != int(bg))[0]
        if not (3 <= len(rows) <= 6):
            continue
        if not np.array_equal(rows, np.arange(int(rows[0]), int(rows[0]) + len(rows))):
            continue
        values = [int(grid[int(r), int(c)]) for r in rows]
        split = 0
        while split < len(values) and values[split] == SEP:
            split += 1
        if not (1 <= split <= 3 and 2 <= len(values) - split <= 3):
            continue
        palette = values[split:]
        if any(color in (int(bg), SEP) for color in palette):
            continue
        if len(set(palette)) != len(palette):
            continue
        isolated = True
        for r in rows:
            for dc in (-1, 1):
                cc = int(c) + dc
                if 0 <= cc < W and int(grid[int(r), cc]) != int(bg):
                    isolated = False
                    break
            if not isolated:
                break
        if isolated:
            candidates.append((int(split), palette))
    assert len(candidates) == 1, "expected one marker/palette control column"
    return candidates[0]


def _separator_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[int(r)]
        if int(np.sum(row == SEP)) >= BLOCK and bool(np.all((row == int(bg)) | (row == SEP))):
            rows.append(int(r))
    return rows


def _separator_cols(grid: np.ndarray, bg: int) -> list[int]:
    cols = []
    for c in range(grid.shape[1]):
        col = grid[:, int(c)]
        if int(np.sum(col == SEP)) >= BLOCK and bool(np.all((col == int(bg)) | (col == SEP))):
            cols.append(int(c))
    return cols


def _spans_from_separators(
    sep_positions: list[int],
    support_positions: list[int],
) -> list[tuple[int, int]]:
    if sep_positions:
        spans = [(int(sep_positions[0]) - BLOCK, int(sep_positions[0]) - 1)]
        for prev, cur in zip(sep_positions, sep_positions[1:]):
            spans.append((int(prev) + 1, int(cur) - 1))
        spans.append((int(sep_positions[-1]) + 1, int(sep_positions[-1]) + BLOCK))
    else:
        spans = [(int(min(support_positions)), int(max(support_positions)))]
    assert spans and all(b - a + 1 == BLOCK for a, b in spans), "expected 5-cell slots"
    return spans


def _largest_contiguous_run(indices: list[int]) -> list[int]:
    assert indices
    runs: list[list[int]] = []
    current = [int(indices[0])]
    for value in indices[1:]:
        value = int(value)
        if value == current[-1] + 1:
            current.append(value)
        else:
            runs.append(current)
            current = [value]
    runs.append(current)
    return max(runs, key=lambda run: (len(run), -run[0]))


def _extract_blocks(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    sep_rows = _separator_rows(grid, bg)
    sep_cols = _separator_cols(grid, bg)
    assert sep_rows or sep_cols, "expected separator structure"

    if sep_rows:
        row_support = sep_rows
    else:
        assert sep_cols
        row_candidates = np.where(np.all(grid[:, sep_cols] == SEP, axis=1))[0].tolist()
        row_support = _largest_contiguous_run(row_candidates)
    if sep_cols:
        col_support = sep_cols
    else:
        assert sep_rows
        col_candidates = np.where(np.all(grid[sep_rows, :] == SEP, axis=0))[0].tolist()
        col_support = _largest_contiguous_run(col_candidates)

    row_spans = _spans_from_separators(sep_rows, row_support)
    col_spans = _spans_from_separators(sep_cols, col_support)

    blocks: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
            if np.any(_foreground_mask(patch, bg)):
                blocks.append(patch)
    assert blocks, "expected at least one sprite block"
    return blocks


def _recolor_blocks(blocks: list[np.ndarray], palette: list[int], run_len: int, bg: int) -> list[np.ndarray]:
    recolored: list[np.ndarray] = []
    for index, block in enumerate(blocks):
        color = int(palette[(index // int(run_len)) % len(palette)])
        patch = block.copy()
        patch[_foreground_mask(patch, bg)] = color
        recolored.append(patch)
    return recolored


def _render_strip(blocks: list[np.ndarray], bg: int) -> np.ndarray:
    width = len(blocks) * BLOCK + (len(blocks) - 1)
    out = np.full((BLOCK, width), int(bg), dtype=int)
    col = 0
    for idx, block in enumerate(blocks):
        out[:, int(col) : int(col) + BLOCK] = block
        col += BLOCK
        if idx != len(blocks) - 1:
            out[:, int(col)] = SEP
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor chained 5x5 sprite blocks by palette runs and lay them out as a horizontal strip.

    Input structure:
    - The grid has a black background, several 5x5 sprite blocks, and grey separator rows and/or columns that partition the block slots.
    - Each non-empty slot contains one monochrome 4-connected sprite on black inside its 5x5 cell, while all sprites share the same current foreground color.
    - A separate top-left control column contains a grey marker segment of height 1-3, with 2-3 palette pixels stacked directly below it.

    Rule:
    - Recover every non-empty 5x5 block from the separator lattice.
    - Try horizontal edge chaining (right edge to left edge) and vertical edge chaining (bottom edge to top edge), and select the unique orientation that yields exactly one full ordering of all blocks.
    - Let the marker height be the run length H and the palette column order be C1, C2, ....
    - Recolor the first H chained blocks to C1, the next H to C2, cycling through the palette, then output the recolored blocks as one horizontal strip separated by single grey columns.

    Key insight:
    - The edge-overlap chain determines block order, while the grey marker height controls run length over whole blocks rather than over grid cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    run_len, palette = _find_marker_and_palette(g, bg)
    blocks = _extract_blocks(g, bg)

    h_count, h_order = _count_valid_orders(blocks, "h", bg)
    v_count, v_order = _count_valid_orders(blocks, "v", bg)
    if h_count == 1 and v_count == 0:
        ordered = [blocks[int(i)] for i in h_order]
    elif v_count == 1 and h_count == 0:
        ordered = [blocks[int(i)] for i in v_order]
    else:
        raise AssertionError("expected exactly one chaining orientation with a unique full order")

    recolored = _recolor_blocks(ordered, palette, run_len, bg)
    return _render_strip(recolored, bg)


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 1 <= nr <= 3 and 1 <= nc <= 3 and not mask[nr, nc]:
                pts.add((int(nr), int(nc)))
    return list(pts)


def _build_horizontal_mask(left_offset: int | None, right_offset: int | None, area: int) -> np.ndarray | None:
    mask = np.zeros((BLOCK, BLOCK), dtype=bool)
    if left_offset is not None:
        mask[int(left_offset), 0:3] = True
        lo, hi = sorted((int(left_offset), 2))
        mask[lo : hi + 1, 2] = True
    if right_offset is not None:
        mask[int(right_offset), 2:BLOCK] = True
        lo, hi = sorted((int(right_offset), 2))
        mask[lo : hi + 1, 2] = True

    if int(mask.sum()) < 4:
        mask[2, 2] = True
        mask[2, 1] = True

    while int(mask.sum()) < int(area):
        frontier = _frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[int(r), int(c)] = True

    if mask[0].any() or mask[-1].any():
        return None
    left_hits = np.where(mask[:, 0])[0].tolist()
    right_hits = np.where(mask[:, -1])[0].tolist()
    if left_offset is None and left_hits:
        return None
    if right_offset is None and right_hits:
        return None
    if left_offset is not None and left_hits != [int(left_offset)]:
        return None
    if right_offset is not None and right_hits != [int(right_offset)]:
        return None
    if not _is_contiguous(mask):
        return None
    return mask


def _build_block(
    orient: str,
    in_offset: int | None,
    out_offset: int | None,
    color: int,
) -> np.ndarray | None:
    area = random.randint(6, 10)
    if orient == "h":
        mask = _build_horizontal_mask(in_offset, out_offset, area)
    else:
        base = _build_horizontal_mask(in_offset, out_offset, area)
        mask = None if base is None else base.T
    if mask is None:
        return None
    patch = np.full((BLOCK, BLOCK), BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _layout_options(count: int) -> list[tuple[int, int]]:
    if count == 3:
        return [(1, 3), (3, 1)]
    if count == 4:
        return [(2, 2), (1, 4), (4, 1)]
    return [(2, 3), (3, 2), (5, 1)]


def _layout_size(rows: int, cols: int) -> tuple[int, int]:
    return rows * BLOCK + (rows - 1), cols * BLOCK + (cols - 1)


def _separator_row_positions(row_off: int, rows: int) -> set[int]:
    return {int(row_off + 5 + i * 6) for i in range(rows - 1)}


def _assemble_input(
    blocks: list[np.ndarray],
    run_len: int,
    palette: list[int],
    rows: int,
    cols: int,
    empty_slots: set[int],
) -> np.ndarray | None:
    layout_h, layout_w = _layout_size(rows, cols)
    marker_len = int(run_len + len(palette))

    max_col_off = 30 - layout_w
    if max_col_off < 3:
        return None
    max_row_off = 30 - layout_h
    if max_row_off < 0:
        return None

    block_perm = list(range(len(blocks)))
    random.shuffle(block_perm)
    occupied_slots = [idx for idx in range(rows * cols) if idx not in empty_slots]
    if len(occupied_slots) != len(blocks):
        return None
    row_major_perm = [block_perm[occupied_slots.index(idx)] for idx in occupied_slots]
    if row_major_perm == list(range(len(blocks))):
        return None

    candidate_row_offs = list(range(max_row_off + 1))
    random.shuffle(candidate_row_offs)
    row_off = None
    palette_rows = None
    for guess in candidate_row_offs:
        palette_start = random.randint(0, max(0, min(2, 30 - marker_len)))
        candidate_rows = set(range(int(palette_start + run_len), int(palette_start + marker_len)))
        if candidate_rows & _separator_row_positions(int(guess), rows):
            continue
        row_off = int(guess)
        palette_rows = candidate_rows
        marker_start = int(palette_start)
        break
    if row_off is None or palette_rows is None:
        return None

    col_off = random.randint(3, max_col_off)
    min_h = max(int(row_off + layout_h), int(marker_start + marker_len))
    min_w = int(col_off + layout_w)
    if min_h > 30 or min_w > 30:
        return None
    H = random.randint(min_h, min(30, min_h + 2))
    W = random.randint(min_w, min(30, min_w + 2))

    grid = np.full((H, W), BG, dtype=int)

    for r in range(rows - 1):
        rr = int(row_off + 5 + r * 6)
        grid[int(rr), int(col_off) : int(col_off + layout_w)] = SEP
    for c in range(cols - 1):
        cc = int(col_off + 5 + c * 6)
        grid[int(row_off) : int(row_off + layout_h), int(cc)] = SEP

    slot_to_block = dict(zip(occupied_slots, block_perm))
    for slot, block_index in slot_to_block.items():
        sr = int(slot // cols)
        sc = int(slot % cols)
        r0 = int(row_off + sr * 6)
        c0 = int(col_off + sc * 6)
        grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = blocks[int(block_index)]

    marker_col = random.choice((0, 1))
    grid[int(marker_start) : int(marker_start + run_len), int(marker_col)] = SEP
    for idx, color in enumerate(palette):
        grid[int(marker_start + run_len + idx), int(marker_col)] = int(color)
    return grid


def _generate_blocks(count: int, orient: str, color: int) -> list[np.ndarray] | None:
    offsets = [random.choice(EDGE_OFFSETS) for _ in range(count - 1)]
    blocks: list[np.ndarray] = []
    for idx in range(count):
        in_offset = offsets[idx - 1] if idx > 0 else None
        out_offset = offsets[idx] if idx < count - 1 else None
        block = _build_block(orient, in_offset, out_offset, color)
        if block is None:
            return None
        blocks.append(block)

    chosen_count, chosen_order = _count_valid_orders(blocks, orient, BG)
    other = "v" if orient == "h" else "h"
    other_count, _ = _count_valid_orders(blocks, other, BG)
    if chosen_count != 1 or other_count != 0 or chosen_order != tuple(range(count)):
        return None
    return blocks


def _generate_attempt() -> np.ndarray | None:
    count = random.choice(BLOCK_COUNTS)
    orient = random.choice(CHAIN_ORIENTATIONS)
    run_len = random.choice(RUN_LENGTHS)
    palette_size = random.choice(PALETTE_SIZES)

    sprite_color = random.choice(COLOR_POOL)
    palette_choices = [color for color in COLOR_POOL if color != sprite_color]
    palette = random.sample(palette_choices, palette_size)

    blocks = _generate_blocks(count, orient, sprite_color)
    if blocks is None:
        return None

    rows, cols = random.choice(_layout_options(count))
    empty_slots: set[int] = set()
    if rows * cols > count:
        empty_slots = {random.randrange(rows * cols)}

    grid = _assemble_input(blocks, run_len, palette, rows, cols, empty_slots)
    if grid is None:
        return None

    out = main(grid)
    if out.shape[0] > 30 or out.shape[1] > 30:
        return None
    if count * BLOCK + (count - 1) != out.shape[1]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
