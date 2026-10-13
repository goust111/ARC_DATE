from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

ARROW_TEMPLATES: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    RIGHT: ((0, 0), (0, 1), (0, 2), (-1, 2), (1, 2)),
    LEFT: ((0, 0), (0, -1), (0, -2), (-1, -2), (1, -2)),
    DOWN: ((0, 0), (1, 0), (2, 0), (2, -1), (2, 1)),
    UP: ((0, 0), (-1, 0), (-2, 0), (-2, -1), (-2, 1)),
}

# Generator retry budgets.
MAX_GENERATION_ATTEMPTS = 400
PAIR_ATTEMPTS = 250
EXTRA_BLOCK_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _translate(anchor: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(int(ar + dr), int(ac + dc)) for dr, dc in offsets]


def _block_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = top_left
    return [(int(r + dr), int(c + dc)) for dr in (0, 1) for dc in (0, 1)]


def _tail_cells(tail: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    tr, tc = tail
    dr, dc = direction
    return [(int(tr - dr * step), int(tc - dc * step)) for step in range(1, int(length) + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _mark(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _mark_buffer(buffered: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = buffered.shape
    for r, c in cells:
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                buffered[rr, cc] = True


def _cells_clear(cells: list[tuple[int, int]], blocked: np.ndarray, buffered: np.ndarray) -> bool:
    for r, c in cells:
        if blocked[r, c] or buffered[r, c]:
            return False
    return True


def _detect_arrows(grid: np.ndarray) -> list[dict[str, tuple[int, int]]]:
    arrows: list[dict[str, tuple[int, int]]] = []
    for coords in _components(grid == GREY):
        pts = {(int(r), int(c)) for r, c in coords}
        assert len(pts) == 5, "grey objects must be 5-pixel arrows"

        match = None
        for tail in pts:
            for direction, template in ARROW_TEMPLATES.items():
                if pts == set(_translate(tail, template)):
                    tip = (int(tail[0] + 2 * direction[0]), int(tail[1] + 2 * direction[1]))
                    candidate = {"tail": tail, "tip": tip, "dir": direction}
                    assert match is None, "arrow component matched multiple directions"
                    match = candidate

        assert match is not None, "failed to decode a grey arrow"
        arrows.append(match)

    arrows.sort(key=lambda item: (item["tail"][0], item["tail"][1], item["dir"][0], item["dir"][1]))
    return arrows


def _detect_blocks(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int], dict[int, int]]:
    blocks: list[dict[str, object]] = []
    cell_to_block: dict[tuple[int, int], int] = {}
    color_counts: dict[int, int] = defaultdict(int)

    for color in sorted(int(col) for col in np.unique(grid) if int(col) not in (BLACK, GREY)):
        for coords in _components(grid == int(color)):
            assert len(coords) == 4, "each colored object must be a 2x2 block"
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            pts = {(int(r), int(c)) for r, c in coords}
            expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            assert r1 - r0 == 1 and c1 - c0 == 1 and pts == expected, "colored objects must be exact 2x2 blocks"

            block_id = int(len(blocks))
            cells = tuple(sorted(expected))
            blocks.append({"color": int(color), "top_left": (r0, c0), "cells": cells})
            color_counts[int(color)] += 1
            for cell in cells:
                assert cell not in cell_to_block
                cell_to_block[cell] = block_id

    return blocks, cell_to_block, color_counts


def _first_hit_block(
    grid: np.ndarray,
    cell_to_block: dict[tuple[int, int], int],
    tip: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    h, w = grid.shape
    r, c = tip
    dr, dc = direction

    while True:
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "every arrow ray must hit a block"
        if (int(r), int(c)) in cell_to_block:
            return int(cell_to_block[(int(r), int(c))])
        assert int(grid[r, c]) == BLACK, "rays may only cross background before the first block"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a colored tail behind each gray arrow using the first 2x2 block hit by its ray.

    Input structure:
    - The background is black.
    - Each gray object is a 5-pixel arrow: a 3-pixel axis plus two symmetric wing pixels at the tip, pointing up, down, left, or right.
    - The remaining colored objects are isolated monochrome 2x2 blocks whose colors are neither black nor gray.

    Rule:
    - For each gray arrow, cast a ray from its tip in the pointing direction and find the first 2x2 block that the ray enters.
    - Let C be that block's color and let K be the total number of 2x2 blocks of color C anywhere in the grid.
    - Keep all arrows and blocks unchanged, then color K new pixels from the arrow tail backward along the opposite direction using color C.

    Key insight:
    - The arrow determines where to look, the first hit determines the color, and the global multiplicity of that color determines the tail length.
    """
    g = np.asarray(grid, dtype=int)
    blocks, cell_to_block, color_counts = _detect_blocks(g)
    assert blocks, "expected at least one colored block"

    out = g.copy()
    for arrow in _detect_arrows(g):
        block_id = _first_hit_block(g, cell_to_block, arrow["tip"], arrow["dir"])
        color = int(blocks[block_id]["color"])
        length = int(color_counts[color])
        tr, tc = arrow["tail"]
        dr, dc = arrow["dir"]

        for step in range(1, length + 1):
            rr = int(tr - dr * step)
            cc = int(tc - dc * step)
            assert _in_bounds(g.shape, rr, cc), "tail growth must stay inside the grid"
            assert int(g[rr, cc]) == BLACK, "tail growth must extend into empty background"
            out[rr, cc] = int(color)

    return out


def _sample_tail(height: int, width: int, direction: tuple[int, int], length: int) -> tuple[int, int] | None:
    if direction == RIGHT:
        if height < 3 or width < length + 6:
            return None
        return random.randint(1, height - 2), random.randint(length, width - 6)
    if direction == LEFT:
        if height < 3 or width < length + 6:
            return None
        return random.randint(1, height - 2), random.randint(5, width - 1 - length)
    if direction == DOWN:
        if width < 3 or height < length + 6:
            return None
        return random.randint(length, height - 6), random.randint(1, width - 2)
    assert direction == UP
    if width < 3 or height < length + 6:
        return None
    return random.randint(5, height - 1 - length), random.randint(1, width - 2)


def _sample_target_top_left(
    height: int,
    width: int,
    tail: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int] | None:
    tr, tc = tail
    if direction == RIGHT:
        lo, hi = int(tc + 4), int(width - 2)
        if lo > hi:
            return None
        return tr + random.choice((-1, 0)), random.randint(lo, hi)
    if direction == LEFT:
        lo, hi = 0, int(tc - 5)
        if lo > hi:
            return None
        return tr + random.choice((-1, 0)), random.randint(lo, hi)
    if direction == DOWN:
        lo, hi = int(tr + 4), int(height - 2)
        if lo > hi:
            return None
        return random.randint(lo, hi), tc + random.choice((-1, 0))
    assert direction == UP
    lo, hi = 0, int(tr - 5)
    if lo > hi:
        return None
    return random.randint(lo, hi), tc + random.choice((-1, 0))


def _place_block(
    grid: np.ndarray,
    blocked: np.ndarray,
    buffered: np.ndarray,
    color: int,
) -> list[tuple[int, int]] | None:
    height, width = grid.shape
    positions = [(r, c) for r in range(height - 1) for c in range(width - 1)]
    random.shuffle(positions)

    for top_left in positions[:EXTRA_BLOCK_ATTEMPTS]:
        cells = _block_cells(top_left)
        if not _cells_clear(cells, blocked, buffered):
            continue
        for r, c in cells:
            grid[r, c] = int(color)
        _mark(blocked, cells)
        _mark_buffer(buffered, cells)
        return cells
    return None


def _place_arrow_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    buffered: np.ndarray,
    direction: tuple[int, int],
    color: int,
    length: int,
) -> dict[str, object] | None:
    height, width = grid.shape

    for _ in range(PAIR_ATTEMPTS):
        tail = _sample_tail(height, width, direction, length)
        if tail is None:
            return None

        arrow_cells = _translate(tail, ARROW_TEMPLATES[direction])
        if not _cells_clear(arrow_cells, blocked, buffered):
            continue

        top_left = _sample_target_top_left(height, width, tail, direction)
        if top_left is None:
            continue
        target_cells = _block_cells(top_left)
        if not _cells_clear(target_cells, blocked, buffered):
            continue

        future_tail = _tail_cells(tail, direction, length)
        if not _cells_clear(future_tail, blocked, buffered):
            continue

        tip = (int(tail[0] + 2 * direction[0]), int(tail[1] + 2 * direction[1]))
        ray_cells: list[tuple[int, int]] = []
        target_set = set(target_cells)
        rr, cc = tip
        valid = True
        while True:
            rr += direction[0]
            cc += direction[1]
            if not _in_bounds(grid.shape, rr, cc):
                valid = False
                break
            if (rr, cc) in target_set:
                break
            if blocked[rr, cc] or buffered[rr, cc]:
                valid = False
                break
            ray_cells.append((int(rr), int(cc)))

        if not valid or not ray_cells:
            continue

        for r, c in arrow_cells:
            grid[r, c] = GREY
        for r, c in target_cells:
            grid[r, c] = int(color)

        _mark(blocked, arrow_cells + target_cells + future_tail + ray_cells)
        _mark_buffer(buffered, arrow_cells + target_cells + future_tail)

        return {
            "direction": direction,
            "color": int(color),
            "length": int(length),
            "tail": tail,
            "tip": tip,
            "arrow_cells": arrow_cells,
            "target_cells": target_cells,
            "tail_cells": future_tail,
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    buffered = np.zeros_like(grid, dtype=bool)

    arrow_count = random.randint(2, 4)
    directions = random.sample([UP, DOWN, LEFT, RIGHT], arrow_count)
    lengths = random.sample([1, 2, 3, 4], arrow_count)
    colors = random.sample(PALETTE, arrow_count)

    specs = [
        {"direction": direction, "length": int(length), "color": int(color)}
        for direction, length, color in zip(directions, lengths, colors)
    ]
    random.shuffle(specs)
    specs.sort(key=lambda item: -int(item["length"]))

    placed_specs: list[dict[str, object]] = []
    for spec in specs:
        placed = _place_arrow_pair(
            grid=grid,
            blocked=blocked,
            buffered=buffered,
            direction=spec["direction"],
            color=spec["color"],
            length=spec["length"],
        )
        if placed is None:
            return None
        placed_specs.append(placed)

    extra_colors: list[int] = []
    for spec in placed_specs:
        extra_colors.extend([int(spec["color"])] * (int(spec["length"]) - 1))
    random.shuffle(extra_colors)

    for color in extra_colors:
        if _place_block(grid, blocked, buffered, color) is None:
            return None

    expected = grid.copy()
    for spec in placed_specs:
        for r, c in spec["tail_cells"]:
            expected[r, c] = int(spec["color"])

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray ARC input")
