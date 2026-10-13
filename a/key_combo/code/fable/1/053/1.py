from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
STRUCTURE = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
PLATFORM_WIDTH = 3
MIN_BLOCKS = 3
MAX_BLOCKS = 4
MIN_HEIGHT = 16
MAX_HEIGHT = 28
MIN_RIGHT_WIDTH = 5
MAX_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in directions:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    divider_cols = [
        c
        for c in range(grid.shape[1])
        if np.all(grid[:, c] != bg) and np.unique(grid[:, c]).size == 1
    ]
    assert len(divider_cols) == 1, "expected exactly one full-height divider column"
    divider_c = int(divider_cols[0])
    return divider_c, int(grid[0, divider_c])


def _parse_blocks(left: np.ndarray, bg: int, structure: int) -> list[dict]:
    comps = components_mask((left != bg) & (left != structure), connectivity=4)
    assert comps, "expected colored blocks on the left side"
    blocks: list[dict] = []
    for coords in comps:
        assert len(coords) == 1, "each colored block must be a single cell"
        r, c = map(int, coords[0])
        supports = np.flatnonzero(left[r + 1 :, c] == structure)
        assert supports.size > 0, "each block needs a platform below"
        platform_r = r + 1 + int(supports[0])
        land_r = platform_r - 1
        assert left[land_r, c] == bg, "landing cell must be empty"
        blocks.append(
            {
                "row": r,
                "col": c,
                "color": int(left[r, c]),
                "platform_row": platform_r,
                "land_row": land_r,
            }
        )
    land_rows = [block["land_row"] for block in blocks]
    assert len(set(land_rows)) == len(land_rows), "landing heights must be unique"
    return blocks


def _parse_rectangles(right: np.ndarray, structure: int) -> list[dict]:
    comps = components_mask(right == structure, connectivity=4)
    assert comps, "expected gray rectangles on the right side"
    rects: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = right[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == structure), "right-side objects must be solid rectangles"
        rects.append({"bbox": (r0, c0, r1, c1), "coords": coords})
    top_rows = [rect["bbox"][0] for rect in rects]
    assert len(set(top_rows)) == len(top_rows), "right rectangles must have unique top rows"
    return rects


def _recolor_rectangles(
    out: np.ndarray,
    divider_c: int,
    rects: list[dict],
    colors: list[int],
) -> None:
    right_c0 = divider_c + 1
    for rect, color in zip(rects, colors):
        for r, c in rect["coords"]:
            out[int(r), right_c0 + int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the left blocks onto their platforms, then use the landing-height ranking to recolor the right legend stack.

    Input structure:
    - A single full-height vertical divider splits the grid into left and right halves.
    - On the left, a black field contains several gray horizontal platforms and one single-cell colored block above each platform.
    - On the right, the same number of gray solid rectangles are stacked from top to bottom on the black background.

    Rule:
    - Each colored block falls straight down within its own column until it stops on the nearest gray platform below it.
    - Rank the block colors by their final landing rows from low to high, meaning larger row index first because the grid bottom is lower.
    - Recolor the right-side gray rectangles from top to bottom with that ranked color sequence, and keep the divider and platforms unchanged.

    Key insight:
    - The color legend for the right side is not explicitly drawn: it is created by simulating gravity on the left and reading the resulting height order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    divider_c, structure = _find_divider(g, bg)
    left = g[:, :divider_c]
    right = g[:, divider_c + 1 :]

    blocks = _parse_blocks(left, bg, structure)
    rects = _parse_rectangles(right, structure)
    assert len(blocks) == len(rects), "block count must match right-rectangle count"

    out = g.copy()
    for block in blocks:
        out[block["row"], block["col"]] = bg
    for block in blocks:
        out[block["land_row"], block["col"]] = block["color"]

    ranked_colors = [
        block["color"]
        for block in sorted(blocks, key=lambda item: item["land_row"], reverse=True)
    ]
    sorted_rects = sorted(rects, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    _recolor_rectangles(out, divider_c, sorted_rects, ranked_colors)
    return out


def _sample_spaced(count: int, low: int, high: int, min_gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if not values:
        return None
    for _ in range(200):
        random.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - prev) >= min_gap for prev in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _left_layout(n_blocks: int) -> tuple[int, list[tuple[int, int, int]]]:
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gaps = [random.randint(1, 2) for _ in range(n_blocks - 1)]
    x = left_margin
    layout: list[tuple[int, int, int]] = []
    for idx in range(n_blocks):
        c0 = x
        c1 = c0 + PLATFORM_WIDTH - 1
        block_c = random.randint(c0, c1)
        layout.append((c0, c1, block_c))
        x = c1 + 1
        if idx < n_blocks - 1:
            x += gaps[idx]
    left_w = x + right_margin
    return left_w, layout


def _choose_final_order(n_blocks: int) -> tuple[int, ...]:
    identity = tuple(range(n_blocks))
    for _ in range(50):
        order = tuple(random.sample(range(n_blocks), n_blocks))
        if order != identity:
            return order
    raise RuntimeError("failed to choose a nontrivial final order")


def _assign_platform_rows(height: int, final_order: tuple[int, ...]) -> list[int] | None:
    sampled = _sample_spaced(len(final_order), 6, height - 3, 3)
    if sampled is None:
        return None
    platform_rows = [0] * len(final_order)
    for rank, block_idx in enumerate(final_order):
        platform_rows[block_idx] = sampled[::-1][rank]
    return platform_rows


def _assign_initial_rows(
    platform_rows: list[int],
    final_order: tuple[int, ...],
) -> tuple[list[int], tuple[int, ...]] | None:
    n_blocks = len(platform_rows)
    for _ in range(300):
        rows = [None] * n_blocks
        used: set[int] = set()
        order = list(range(n_blocks))
        random.shuffle(order)
        ok = True
        for idx in order:
            choices = [r for r in range(1, platform_rows[idx] - 1) if r not in used]
            if not choices:
                ok = False
                break
            rows[idx] = random.choice(choices)
            used.add(rows[idx])
        if not ok:
            continue
        init_order = tuple(sorted(range(n_blocks), key=lambda i: rows[i], reverse=True))
        if init_order != final_order:
            return [int(r) for r in rows], init_order
    return None


def _right_layout(height: int, right_w: int, n_blocks: int) -> list[tuple[int, int, int, int]] | None:
    heights = [random.choice([2, 3]) for _ in range(n_blocks)]
    gaps = [random.choice([1, 2]) for _ in range(n_blocks - 1)]
    total = sum(heights) + sum(gaps)
    max_start = height - total - 1
    if max_start < 1:
        return None
    start_r = random.randint(1, max_start)
    rects: list[tuple[int, int, int, int]] = []
    r0 = start_r
    for idx, rect_h in enumerate(heights):
        rect_w = random.randint(3, right_w - 2)
        c0 = random.randint(1, right_w - rect_w - 1)
        rects.append((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
        r0 += rect_h
        if idx < n_blocks - 1:
            r0 += gaps[idx]
    return rects


def _expected_output(
    grid: np.ndarray,
    divider_c: int,
    blocks: list[dict],
    rects: list[tuple[int, int, int, int]],
    final_order: tuple[int, ...],
) -> np.ndarray:
    out = grid.copy()
    for block in blocks:
        out[block["row"], block["col"]] = BACKGROUND
        out[block["land_row"], block["col"]] = block["color"]
    ranked_colors = [blocks[idx]["color"] for idx in final_order]
    for color, (r0, c0, r1, c1) in zip(ranked_colors, rects):
        out[r0 : r1 + 1, divider_c + 1 + c0 : divider_c + 1 + c1 + 1] = color
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    blocks: list[dict],
    rects: list[tuple[int, int, int, int]],
    final_order: tuple[int, ...],
    init_order: tuple[int, ...],
) -> bool:
    if np.array_equal(grid, expected):
        return False
    left_to_right = tuple(sorted(range(len(blocks)), key=lambda i: blocks[i]["col"]))
    if final_order == left_to_right or final_order == init_order:
        return False

    final_colors = [blocks[idx]["color"] for idx in final_order]
    left_colors = [blocks[idx]["color"] for idx in left_to_right]
    init_colors = [blocks[idx]["color"] for idx in init_order]
    if final_colors == left_colors or final_colors == init_colors:
        return False

    divider_c, _structure = _find_divider(grid, BACKGROUND)
    wrong_lr = grid.copy()
    wrong_init = grid.copy()
    for color, rect in zip(left_colors, rects):
        r0, c0, r1, c1 = rect
        wrong_lr[r0 : r1 + 1, divider_c + 1 + c0 : divider_c + 1 + c1 + 1] = color
    for color, rect in zip(init_colors, rects):
        r0, c0, r1, c1 = rect
        wrong_init[r0 : r1 + 1, divider_c + 1 + c0 : divider_c + 1 + c1 + 1] = color
    if np.array_equal(wrong_lr, expected) or np.array_equal(wrong_init, expected):
        return False

    landed_rows = [block["land_row"] for block in blocks]
    if len(set(landed_rows)) != len(landed_rows):
        return False
    top_rows = [rect[0] for rect in rects]
    if top_rows != sorted(top_rows):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_blocks = random.randint(MIN_BLOCKS, MAX_BLOCKS)
    left_w, left_layout = _left_layout(n_blocks)
    max_right_w = 29 - left_w
    if max_right_w < MIN_RIGHT_WIDTH:
        return None
    right_w = random.randint(MIN_RIGHT_WIDTH, max_right_w)

    min_h = max(MIN_HEIGHT, 3 * n_blocks + 6)
    height = random.randint(min_h, MAX_HEIGHT)
    rects = _right_layout(height, right_w, n_blocks)
    if rects is None:
        return None

    final_order = _choose_final_order(n_blocks)
    platform_rows = _assign_platform_rows(height, final_order)
    if platform_rows is None:
        return None

    initial = _assign_initial_rows(platform_rows, final_order)
    if initial is None:
        return None
    initial_rows, init_order = initial

    width = left_w + 1 + right_w
    divider_c = left_w
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:, divider_c] = STRUCTURE

    colors = random.sample(COLOR_POOL, n_blocks)
    blocks: list[dict] = []
    for idx, (c0, c1, block_c) in enumerate(left_layout):
        platform_r = platform_rows[idx]
        row = initial_rows[idx]
        color = colors[idx]
        grid[platform_r, c0 : c1 + 1] = STRUCTURE
        grid[row, block_c] = color
        blocks.append(
            {
                "row": row,
                "col": block_c,
                "color": color,
                "platform_row": platform_r,
                "land_row": platform_r - 1,
            }
        )

    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, divider_c + 1 + c0 : divider_c + 1 + c1 + 1] = STRUCTURE

    expected = _expected_output(grid, divider_c, blocks, rects, final_order)
    if not _semantic_checks(grid, expected, blocks, rects, final_order, init_order):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-ranking puzzle")
