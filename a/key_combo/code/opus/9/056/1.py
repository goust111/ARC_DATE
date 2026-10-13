from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
FG_COLORS = (BLUE, RED, GREEN)

CARDINAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_STEPS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ALL8_STEPS = CARDINAL_STEPS + DIAGONAL_STEPS

GRID_MIN = 10
GRID_MAX = 14
GEN_ATTEMPTS = 500
INPUT_ASSIGNMENT_ATTEMPTS = 300


def _neighbors(shape: tuple[int, int], r: int, c: int, steps: Iterable[tuple[int, int]]):
    height, width = shape
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components(mask: np.ndarray, *, connectivity: int) -> list[list[tuple[int, int]]]:
    steps = CARDINAL_STEPS if connectivity == 4 else ALL8_STEPS
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors(mask.shape, cr, cc, steps):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(comp)

    return parts


def _has_four_neighbor_pair(coords: Iterable[tuple[int, int]]) -> bool:
    pts = set(coords)
    return any((r + 1, c) in pts or (r, c + 1) in pts for r, c in pts)


def _settle_columns(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    out = np.zeros_like(grid)
    for col in range(width):
        values = grid[:, col][grid[:, col] != BLACK]
        out[height - values.size :, col] = values
    return out


def _analyze_groups(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {BLACK}):
        for comp in _components(grid == color, connectivity=8):
            has4 = _has_four_neighbor_pair(comp)
            infos.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "size": len(comp),
                    "has4": has4,
                    "pure_diagonal": len(comp) >= 2 and not has4,
                }
            )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """Settle each column, then recolor pure same-color diagonal chains to grey.

    Input structure:
    - The grid uses black background and 2-3 foreground colors chosen from blue,
      red, and green.
    - Foreground cells are sparse scattered single pixels before motion; columns
      may contain multiple colors stacked in top-to-bottom order.

    Rule:
    - For each column, read the non-black cells from top to bottom, clear the
      column, then write those colors back from the bottom upward in the same
      order.
    - After this independent column gravity settles every column, process each
      foreground color separately and split its cells into 8-connected groups.
    - Recolor an entire group grey exactly when the settled group has at least
      two cells and contains no 4-neighbor pair anywhere inside it; otherwise
      keep the original color.

    Key insight:
    - The decisive topology appears only after gravity: column order fixes where
      colors land, and only the settled same-color 8-components must be tested
      for the absence of any edge-sharing contact.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    settled = _settle_columns(g)
    out = settled.copy()
    for info in _analyze_groups(settled):
        if info["pure_diagonal"]:
            for r, c in info["coords"]:
                out[r, c] = GREY
    return out


def _intervals_separated(a0: int, a1: int, b0: int, b1: int, gap: int = 1) -> bool:
    return a1 + gap < b0 or b1 + gap < a0


def _place_layout(width: int, *, three_colors: bool):
    chain_len = 4
    block_len = 3 if three_colors else 2

    for _ in range(200):
        chain_start = random.randint(0, width - chain_len)
        block_start = random.randint(0, width - block_len)
        if not _intervals_separated(chain_start, chain_start + chain_len - 1, block_start, block_start + block_len - 1):
            continue

        if not three_colors:
            return {
                "chain_start": chain_start,
                "block_start": block_start,
                "block_kind": "pair",
                "iso_col": None,
            }

        block_kind = random.choice(("triple", "mixed_left", "mixed_right"))
        free_cols = [
            col
            for col in range(width)
            if _intervals_separated(col, col, chain_start, chain_start + chain_len - 1)
            and _intervals_separated(col, col, block_start, block_start + block_len - 1)
        ]
        if not free_cols:
            continue
        return {
            "chain_start": chain_start,
            "block_start": block_start,
            "block_kind": block_kind,
            "iso_col": random.choice(free_cols),
        }

    return None


def _build_settled_grid(height: int, width: int) -> np.ndarray | None:
    three_colors = random.random() < 0.65
    chosen = random.sample(FG_COLORS, 3 if three_colors else 2)
    chain_color = chosen[0]
    if three_colors:
        block_color = chosen[1]
        filler_color = chosen[2]
    else:
        block_color = chosen[1]
        filler_color = chosen[1]

    layout = _place_layout(width, three_colors=three_colors)
    if layout is None:
        return None

    settled = np.zeros((height, width), dtype=int)

    chain_heights = [1, 2, 3, 4]
    if random.random() < 0.5:
        chain_heights.reverse()
    for idx, stack_height in enumerate(chain_heights):
        col = layout["chain_start"] + idx
        top_row = height - stack_height
        settled[top_row, col] = chain_color
        if stack_height > 1:
            settled[top_row + 1 :, col] = filler_color

    if three_colors:
        block_kind = layout["block_kind"]
        for idx in range(3):
            col = layout["block_start"] + idx
            if block_kind == "triple":
                settled[height - 1, col] = block_color
            elif block_kind == "mixed_left":
                stack_height = (2, 1, 1)[idx]
                top_row = height - stack_height
                settled[top_row, col] = block_color
                if stack_height == 2:
                    settled[height - 1, col] = filler_color
            else:
                stack_height = (1, 1, 2)[idx]
                top_row = height - stack_height
                settled[top_row, col] = block_color
                if stack_height == 2:
                    settled[height - 1, col] = filler_color
        settled[height - 1, int(layout["iso_col"])] = block_color
    else:
        for idx in range(2):
            col = layout["block_start"] + idx
            settled[height - 1, col] = block_color

    return settled


def _row_patterns(height: int, count: int, prev_rows: set[int], *, require_unsettled: bool) -> list[tuple[int, ...]]:
    settled_rows = tuple(range(height - count, height))
    patterns: list[tuple[int, ...]] = []

    def rec(start: int, acc: list[int]) -> None:
        if len(acc) == count:
            rows = tuple(acc)
            if require_unsettled and rows == settled_rows:
                return
            patterns.append(rows)
            return

        remaining = count - len(acc)
        last_possible = height - 1 - 2 * (remaining - 1)
        for row in range(start, last_possible + 1):
            if row in prev_rows:
                continue
            acc.append(row)
            rec(row + 2, acc)
            acc.pop()

    rec(0, [])
    return patterns


def _sample_input_from_settled(settled: np.ndarray) -> np.ndarray | None:
    height, width = settled.shape
    columns = [list(map(int, settled[:, c][settled[:, c] != BLACK])) for c in range(width)]

    for _ in range(INPUT_ASSIGNMENT_ATTEMPTS):
        inp = np.zeros_like(settled)
        prev_rows: set[int] = set()
        ok = True
        unsettled_columns = 0

        for col, values in enumerate(columns):
            count = len(values)
            if count == 0:
                prev_rows = set()
                continue

            patterns = _row_patterns(height, count, prev_rows, require_unsettled=True)
            if not patterns:
                ok = False
                break

            rows = random.choice(patterns)
            if rows != tuple(range(height - count, height)):
                unsettled_columns += 1
            for row, value in zip(rows, values):
                inp[row, col] = value
            prev_rows = set(rows)

        if not ok or unsettled_columns < 2:
            continue

        if not np.array_equal(_settle_columns(inp), settled):
            continue

        if any(info["pure_diagonal"] and info["size"] >= 3 for info in _analyze_groups(inp)):
            continue

        return inp

    return None


def _is_valid_input(inp: np.ndarray) -> bool:
    colors = sorted(set(map(int, inp.ravel())) - {BLACK})
    if not (2 <= len(colors) <= 3):
        return False

    for color in colors:
        count = int(np.count_nonzero(inp == color))
        if not (4 <= count <= 8):
            return False

    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if inp[r, c] == BLACK:
                continue
            for nr, nc in _neighbors(inp.shape, r, c, CARDINAL_STEPS):
                if inp[nr, nc] != BLACK:
                    return False
    return True


def _semantic_checks(inp: np.ndarray) -> bool:
    settled = _settle_columns(inp)
    out = main(inp)
    infos = _analyze_groups(settled)

    target_groups = [info for info in infos if info["pure_diagonal"]]
    if not target_groups:
        return False
    if max(info["size"] for info in target_groups) < 4:
        return False
    if not any(info["has4"] for info in infos):
        return False

    recolored = int(np.count_nonzero(out == GREY))
    expected_recolored = sum(info["size"] for info in target_groups)
    if recolored != expected_recolored:
        return False

    if np.array_equal(inp, settled) or np.array_equal(out, settled):
        return False

    input_infos = _analyze_groups(inp)
    if any(info["pure_diagonal"] and info["size"] >= 3 for info in input_infos):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)

        settled = _build_settled_grid(height, width)
        if settled is None:
            continue

        inp = _sample_input_from_settled(settled)
        if inp is None:
            continue

        if _is_valid_input(inp) and _semantic_checks(inp):
            return inp

    raise RuntimeError("failed to generate a valid gravity-diagonal-chain input")
