from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


BG = Color.BLACK
WATER = Color.BLUE
BLOCK_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK]

SOURCE_COUNT_CHOICES = (2, 3)
BLOCK_COUNT_PATTERNS = (
    (1, 2),
    (2, 1),
    (1, 1, 1),
    (2, 2),
    (1, 2, 1),
    (2, 1, 1),
)
BLOCK_SIZE_CHOICES = (4, 5)
GRID_W_RANGE = (15, 24)
TOP_ROW_MIN = 4
TOP_ROW_MAX = 7
LAYER_GAP_MIN = 3
LAYER_GAP_MAX = 5
MIN_BLOCK_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _solid_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    blocks: list[dict] = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=BLOCK_COLORS,
        return_format="dict",
    ):
        color = int(comp["color"])
        b = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(grid, b)
        if int(comp["area"]) != bbox_h(b) * bbox_w(b):
            continue
        if not bool((patch == color).all()):
            continue
        r0, c0, _r1, _c1 = b
        blocks.append(
            {
                "color": color,
                "bbox": b,
                "r0": r0,
                "c0": c0,
                "h": bbox_h(b),
                "w": bbox_w(b),
            }
        )
    blocks.sort(key=lambda item: (item["r0"], item["c0"]))
    return blocks


def _source_starts(grid: np.ndarray) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=8,
        monochromatic=True,
        colors=[WATER],
        return_format="dict",
    ):
        coords = comp["coords"]
        if not bool((coords[:, 0] == 0).any()):
            continue
        rmax = int(coords[:, 0].max())
        cols = sorted({int(c) for r, c in coords if int(r) == rmax})
        for c in cols:
            sources.append((rmax, c))
    sources.sort()
    return sources


def _pack_tiles(blocks: list[dict], carved_tiles: dict[tuple[int, int], np.ndarray], bg: int) -> np.ndarray:
    row_keys = sorted({int(block["r0"]) for block in blocks})
    col_keys = sorted({int(block["c0"]) for block in blocks})

    row_heights = {
        row: max(int(block["h"]) for block in blocks if int(block["r0"]) == row)
        for row in row_keys
    }
    col_widths = {
        col: max(int(block["w"]) for block in blocks if int(block["c0"]) == col)
        for col in col_keys
    }

    row_offsets: dict[int, int] = {}
    cursor = 1
    for row in row_keys:
        row_offsets[row] = cursor
        cursor += row_heights[row] + 1
    out_h = cursor

    col_offsets: dict[int, int] = {}
    cursor = 1
    for col in col_keys:
        col_offsets[col] = cursor
        cursor += col_widths[col] + 1
    out_w = cursor

    out = np.full((out_h, out_w), int(bg), dtype=int)
    for block in blocks:
        tile = carved_tiles[(int(block["r0"]), int(block["c0"]))]
        paste(out, tile, row_offsets[int(block["r0"])], col_offsets[int(block["c0"])])
    return out


def _solve_structure(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], np.ndarray], dict[int, dict[str, bool]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    blocks = _solid_blocks(g, bg)
    assert 3 <= len(blocks) <= 5, "expected 3-5 solid colored blocks"
    sources = _source_starts(g)
    assert sources, "expected top-edge blue sources"

    block_at = np.full(g.shape, -1, dtype=int)
    for idx, block in enumerate(blocks):
        r0, c0, r1, c1 = map(int, block["bbox"])
        block_at[r0 : r1 + 1, c0 : c1 + 1] = idx

    flags = {idx: {"top": False, "left": False, "right": False} for idx in range(len(blocks))}

    @lru_cache(maxsize=None)
    def trace_branch(start_r: int, col: int) -> bool:
        rr = int(start_r)
        cc = int(col)
        while rr + 1 < g.shape[0]:
            rr += 1
            idx = int(block_at[int(rr), int(cc)])
            if idx < 0:
                continue

            block = blocks[idx]
            flags[idx]["top"] = True
            r0, c0, _r1, c1 = map(int, block["bbox"])

            left_continues = False
            if c0 > 0:
                left_continues = trace_branch(r0 - 1, c0 - 1)
                if left_continues:
                    flags[idx]["left"] = True

            right_continues = False
            if c1 + 1 < g.shape[1]:
                right_continues = trace_branch(r0 - 1, c1 + 1)
                if right_continues:
                    flags[idx]["right"] = True

            return True
        return False

    for start_r, start_c in sources:
        trace_branch(int(start_r), int(start_c))

    carved_tiles: dict[tuple[int, int], np.ndarray] = {}
    for idx, block in enumerate(blocks):
        tile = np.full((int(block["h"]), int(block["w"])), int(block["color"]), dtype=int)
        if flags[idx]["top"]:
            tile[0, :] = int(bg)
        if flags[idx]["left"]:
            tile[:, 0] = int(bg)
        if flags[idx]["right"]:
            tile[:, -1] = int(bg)
        carved_tiles[(int(block["r0"]), int(block["c0"]))] = tile

    return blocks, carved_tiles, flags


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade top-edge water branches through solid blocks and pack the carved results.

    Input structure:
    - The background is black.
    - Two or three blue source clusters touch the top edge.
    - Three to five solid monochrome rectangles of distinct non-blue colors sit lower in the grid, separated by blank space.

    Rule:
    - Treat the blue cells as water sources and let each source column fall straight downward through empty space.
    - When a falling branch first meets a rectangle, that block records a carved top edge, and the water splits to the block's left and right overflow columns just outside the rectangle.
    - An overflow side is carved into the parent block only when that child branch later reaches another lower block; branches that simply fall away do not carve a side.
    - Remove the simulated water and pack all carved rectangles into a compact grid with a one-cell black border and one-cell gaps, preserving the input row/column order.

    Key insight:
    - The carving stencil is generated by the waterfall tree itself: a block's top marks direct contact, while its left/right side cuts encode which overflow branches continue the cascade to later blocks.
    """
    blocks, carved_tiles, _flags = _solve_structure(np.asarray(grid, dtype=int))
    return _pack_tiles(blocks, carved_tiles, BG)


def _sample_source_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(1, width - 1))
    choices: list[list[int]] = []
    for start in cols:
        current = [start]
        last = start
        while len(current) < count:
            candidates = [c for c in cols if last + 3 <= c <= last + 4]
            if not candidates:
                break
            nxt = random.choice(candidates)
            current.append(nxt)
            last = nxt
        if len(current) == count:
            choices.append(current)
    if not choices:
        return None
    return random.choice(choices)


def _candidate_groupings(active_cols: list[int], k: int) -> list[list[tuple[int, ...]]]:
    n = len(active_cols)
    group_pool: list[tuple[int, ...]] = []
    for i in range(n):
        group_pool.append((i,))
        if i + 1 < n and active_cols[i + 1] - active_cols[i] <= 4:
            group_pool.append((i, i + 1))

    valid: list[list[tuple[int, ...]]] = []

    def backtrack(start_idx: int, chosen: list[tuple[int, ...]], used: set[int]) -> None:
        if len(chosen) == k:
            valid.append(chosen.copy())
            return
        for group in group_pool:
            if group[0] < start_idx:
                continue
            if any(idx in used for idx in group):
                continue
            chosen.append(group)
            backtrack(group[-1] + 1, chosen, used | set(group))
            chosen.pop()

    backtrack(0, [], set())
    return valid


def _candidate_block_positions(groups: list[list[int]], active_cols: list[int], width: int) -> list[list[tuple[int, int]]]:
    all_positions: list[list[tuple[int, int]]] = []
    other_cols_sets = []
    for group in groups:
        group_set = set(group)
        other_cols_sets.append([c for c in active_cols if c not in group_set])

    def recurse(idx: int, placed: list[tuple[int, int]]) -> None:
        if idx == len(groups):
            all_positions.append(placed.copy())
            return

        cols = groups[idx]
        others = other_cols_sets[idx]
        min_col = min(cols)
        max_col = max(cols)

        for block_w in random.sample(list(BLOCK_SIZE_CHOICES), k=len(BLOCK_SIZE_CHOICES)):
            lo = max(0, max_col - block_w + 1)
            hi = min(min_col, width - block_w)
            starts = list(range(lo, hi + 1))
            if idx == 0 and 0 in starts and random.random() < 0.35:
                starts.remove(0)
                starts.insert(0, 0)
            if idx == len(groups) - 1:
                edge_start = width - block_w
                if edge_start in starts and random.random() < 0.35:
                    starts.remove(edge_start)
                    starts.insert(0, edge_start)
            random.shuffle(starts)

            for c0 in starts:
                c1 = c0 + block_w - 1
                if any(c0 <= other <= c1 for other in others):
                    continue
                if placed and c0 - placed[-1][1] - 1 < MIN_BLOCK_GAP:
                    continue
                placed.append((c0, c1))
                recurse(idx + 1, placed)
                placed.pop()

    recurse(0, [])
    return all_positions


def _build_grid_from_spec(height: int, width: int, sources: list[tuple[int, int]], blocks: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for source_height, col in sources:
        for r in range(source_height):
            grid[r, col] = WATER
    for block in blocks:
        r0, c0, h, w, color = (
            int(block["r0"]),
            int(block["c0"]),
            int(block["h"]),
            int(block["w"]),
            int(block["color"]),
        )
        grid[r0 : r0 + h, c0 : c0 + w] = color
    return grid


def _naive_top_only(blocks: list[dict], bg: int) -> np.ndarray:
    tiles: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        tile = np.full((int(block["h"]), int(block["w"])), int(block["color"]), dtype=int)
        tile[0, :] = int(bg)
        tiles[(int(block["r0"]), int(block["c0"]))] = tile
    return _pack_tiles(blocks, tiles, bg)


def _naive_available_sides(blocks: list[dict], grid_width: int, bg: int) -> np.ndarray:
    tiles: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        r0, c0, _r1, c1 = map(int, block["bbox"])
        tile = np.full((int(block["h"]), int(block["w"])), int(block["color"]), dtype=int)
        tile[0, :] = int(bg)
        if c0 > 0:
            tile[:, 0] = int(bg)
        if c1 + 1 < grid_width:
            tile[:, -1] = int(bg)
        tiles[(int(block["r0"]), int(block["c0"]))] = tile
    return _pack_tiles(blocks, tiles, bg)


def _semantic_checks(
    grid: np.ndarray,
    blocks_meta: list[dict],
    source_count: int,
    overflow_parent_count: int,
) -> bool:
    try:
        blocks, carved_tiles, flags = _solve_structure(grid)
        actual = _pack_tiles(blocks, carved_tiles, BG)
    except Exception:
        return False

    if len(blocks) != len(blocks_meta):
        return False
    if source_count not in SOURCE_COUNT_CHOICES:
        return False
    if overflow_parent_count < 1:
        return False

    top_only = 0
    sideful = 0
    single_side = 0
    carved_signatures = set()
    for idx, block in enumerate(blocks):
        tile = carved_tiles[(int(block["r0"]), int(block["c0"]))]
        carved_signatures.add(tile.tobytes())
        left = bool(flags[idx]["left"])
        right = bool(flags[idx]["right"])
        if not left and not right:
            top_only += 1
        else:
            sideful += 1
        if left ^ right:
            single_side += 1
    if len(carved_signatures) < 2:
        return False
    if top_only < 1 or sideful < 1 or single_side < 1:
        return False

    if np.array_equal(actual, _naive_top_only(blocks, BG)):
        return False
    if np.array_equal(actual, _naive_available_sides(blocks, grid.shape[1], BG)):
        return False
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*GRID_W_RANGE)
    source_count = random.choice(SOURCE_COUNT_CHOICES)
    source_cols = _sample_source_columns(width, source_count)
    if source_cols is None:
        return None

    pattern_options = [p for p in BLOCK_COUNT_PATTERNS if p[0] <= source_count]
    layer_counts = list(random.choice(pattern_options))
    total_blocks = sum(layer_counts)
    colors = random.sample(BLOCK_COLORS, total_blocks)

    source_heights = [(random.choice((1, 2)), col) for col in source_cols]
    active: dict[int, bool] = {int(col): False for _h, col in source_heights}

    blocks_meta: list[dict] = []
    color_idx = 0
    prev_bottom = max(h for h, _c in source_heights) - 1
    overflow_parent_count = 0

    for layer_idx, block_count in enumerate(layer_counts):
        active_cols = sorted(active)
        if len(active_cols) < block_count:
            return None

        grouping_choices = _candidate_groupings(active_cols, block_count)
        if not grouping_choices:
            return None
        grouping = random.choice(grouping_choices)
        groups = [[active_cols[i] for i in group] for group in grouping]

        position_choices = _candidate_block_positions(groups, active_cols, width)
        if not position_choices:
            return None
        positions = random.choice(position_choices)

        layer_top = random.randint(
            TOP_ROW_MIN if layer_idx == 0 else prev_bottom + LAYER_GAP_MIN,
            TOP_ROW_MAX if layer_idx == 0 else prev_bottom + LAYER_GAP_MAX,
        )

        selected_cols = {col for group in groups for col in group}
        next_active = {col: active[col] for col in active_cols if col not in selected_cols}

        layer_bottom = layer_top
        for group_cols, (c0, c1) in zip(groups, positions):
            h = random.choice(BLOCK_SIZE_CHOICES)
            color = colors[color_idx]
            color_idx += 1
            layer_bottom = max(layer_bottom, layer_top + h - 1)
            parent_from_overflow = any(bool(active[col]) for col in group_cols)
            if parent_from_overflow:
                overflow_parent_count += 1
            blocks_meta.append(
                {
                    "r0": layer_top,
                    "c0": c0,
                    "h": h,
                    "w": c1 - c0 + 1,
                    "color": color,
                    "from_overflow": parent_from_overflow,
                }
            )
            if c0 > 0:
                next_active[c0 - 1] = True or next_active.get(c0 - 1, False)
            if c1 + 1 < width:
                next_active[c1 + 1] = True or next_active.get(c1 + 1, False)

        active = next_active
        prev_bottom = layer_bottom

    height = prev_bottom + random.randint(2, 4)
    if height > 30:
        return None

    grid = _build_grid_from_spec(height, width, source_heights, blocks_meta)

    if not _semantic_checks(grid, blocks_meta, source_count, overflow_parent_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid waterfall-carving input")
