from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np


BACKGROUND = 0
GREEN = 3
ORANGE = 7
BLOCK_COLORS = (1, 2, 4, 6)
MAX_ATTEMPTS = 600


Block = tuple[int, int, int, tuple[int, int, int, int]]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solid_3x3_blocks(grid: np.ndarray, bg: int) -> list[Block]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    blocks: list[Block] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if (
                        in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = bbox_coords(arr)
            if (
                len(coords) == 9
                and r1 - r0 == 2
                and c1 - c0 == 2
                and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            ):
                blocks.append((r0 + 1, c0 + 1, color, (r0, c0, r1, c1)))
    return sorted(blocks)


def _clear_neighborhood_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = map(int, grid.shape)
    clear = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            clear[r, c] = bool(np.all(grid[r0:r1, c0:c1] == bg))
    return clear


def _maximal_true_rectangles(
    mask: np.ndarray,
) -> list[tuple[int, int, int, int, int, int, int]]:
    h, w = map(int, mask.shape)
    rects: set[tuple[int, int, int, int, int, int, int]] = set()
    for r0 in range(h):
        valid_cols = np.ones(w, dtype=bool)
        for r1 in range(r0, h):
            valid_cols &= mask[r1]
            c = 0
            while c < w:
                if not valid_cols[c]:
                    c += 1
                    continue
                c0 = c
                while c < w and valid_cols[c]:
                    c += 1
                c1 = c - 1
                can_extend_up = r0 > 0 and np.all(mask[r0 - 1, c0 : c1 + 1])
                can_extend_down = r1 + 1 < h and np.all(mask[r1 + 1, c0 : c1 + 1])
                can_extend_left = c0 > 0 and np.all(mask[r0 : r1 + 1, c0 - 1])
                can_extend_right = c1 + 1 < w and np.all(mask[r0 : r1 + 1, c1 + 1])
                if can_extend_up or can_extend_down or can_extend_left or can_extend_right:
                    continue
                rh = r1 - r0 + 1
                rw = c1 - c0 + 1
                rects.add((rh * rw, rh, rw, r0, c0, r1, c1))
    return sorted(rects, key=lambda x: (-x[0], -max(x[1], x[2]), x[3], x[4]))


def _detect_corridors(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    clear = _clear_neighborhood_mask(g, BACKGROUND)
    min_major = max(2, min(g.shape) // 2)
    corridors = np.zeros_like(clear, dtype=bool)
    for _area, rh, rw, r0, c0, r1, c1 in _maximal_true_rectangles(clear):
        if max(rh, rw) >= min_major:
            corridors[r0 : r1 + 1, c0 : c1 + 1] = True
    return corridors


def _draw_rays(grid: np.ndarray, corridors: np.ndarray, blocks: list[Block]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = map(int, out.shape)
    for r, c, color, _bbox in blocks:
        for cc in range(c, -1, -1):
            if corridors[r, cc]:
                break
            out[r, cc] = color
        for cc in range(c + 1, w):
            if corridors[r, cc]:
                break
            out[r, cc] = color
        for rr in range(r - 1, -1, -1):
            if corridors[rr, c]:
                break
            out[rr, c] = color
        for rr in range(r + 1, h):
            if corridors[rr, c]:
                break
            out[rr, c] = color
    return out


def _reset_crossings(grid: np.ndarray, corridors: np.ndarray, blocks: list[Block]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for i, (r, _c, _color, _bbox) in enumerate(blocks):
        for j, (_rr, c, _ccolor, _bb) in enumerate(blocks):
            if i == j or corridors[r, c]:
                continue
            out[r, c] = BACKGROUND
    return out


def _apply_rule_with_known_corridors(
    grid: np.ndarray, corridors: np.ndarray, blocks: list[Block]
) -> np.ndarray:
    out = _draw_rays(grid, corridors, blocks)
    out = _reset_crossings(out, corridors, blocks)
    out[corridors] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each 3x3 color block into a cross whose rays stop at hidden blank corridors, then reveal those corridors in green.

    Input structure:
    - The background is black, with orange clutter scattered across most non-corridor cells.
    - Hidden corridors are long black rectangular bars embedded in that clutter; they are detectable only as clear-background rectangles surviving a 3x3 all-black test.
    - Two to four solid 3x3 blocks with distinct colors act as cross generators, and their center rows and center columns are pairwise distinct.

    Rule:
    - Detect corridor cells by keeping only background pixels whose full in-grid 3x3 neighborhood is background, then take the maximal rectangles in that mask whose major axis spans at least half a grid side.
    - For every solid 3x3 block, extend its center row left and right and its center column up and down in the block color, stopping immediately before any detected corridor cell.
    - For every pair of different blocks, reset the crossing of one block's center row and the other's center column to black unless that crossing lies inside a corridor.
    - Finally recolor every detected corridor cell green.

    Key insight:
    - The hidden negative-space corridors are not themselves the objects to expand; instead they must be found first because they physically cap how far each colored cross can grow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    blocks = _solid_3x3_blocks(g, BACKGROUND)
    assert 2 <= len(blocks) <= 4, "expected 2-4 solid 3x3 blocks"
    rows = [r for r, _c, _color, _bbox in blocks]
    cols = [c for _r, c, _color, _bbox in blocks]
    assert len(set(rows)) == len(rows), "block center rows must be unique"
    assert len(set(cols)) == len(cols), "block center cols must be unique"
    corridors = _detect_corridors(g)
    assert np.any(corridors), "expected at least one hidden corridor"
    return _apply_rule_with_known_corridors(g, corridors, blocks)


def _rect_clear(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _expand_rect(rect: tuple[int, int, int, int], margin: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _paint_support_and_clutter(
    shape: tuple[int, int],
    supports: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.zeros((h, w), dtype=int)
    protected = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in supports:
        protected[r0 : r1 + 1, c0 : c1 + 1] = True
    outside = ~protected
    density = float(rng.uniform(0.70, 0.84))
    grid[outside & (rng.random((h, w)) < density)] = ORANGE
    return grid


def _choose_corridors(
    h: int, w: int, rng: np.random.Generator
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]] | None:
    min_major = max(2, min(h, w) // 2)
    target_count = int(rng.integers(1, 4))
    desired = list(rng.choice(np.array(["h", "v"], dtype=object), size=target_count))
    if target_count >= 2 and len(set(desired)) == 1 and rng.random() < 0.6:
        desired[0] = "h"
        desired[-1] = "v"

    corridors: list[tuple[int, int, int, int]] = []
    supports: list[tuple[int, int, int, int]] = []
    for orient in desired:
        placed = False
        for _ in range(200):
            if orient == "h":
                max_h = min(4, h - 4)
                max_w = w - 4
                if max_h < 2 or max_w < min_major:
                    continue
                height = int(rng.integers(2, max_h + 1))
                width = int(rng.integers(min_major, max_w + 1))
                r0 = int(rng.integers(2, h - height - 1))
                c0 = int(rng.integers(2, w - width - 1))
            else:
                max_w = min(4, w - 4)
                max_h = h - 4
                if max_w < 2 or max_h < min_major:
                    continue
                width = int(rng.integers(2, max_w + 1))
                height = int(rng.integers(min_major, max_h + 1))
                r0 = int(rng.integers(2, h - height - 1))
                c0 = int(rng.integers(2, w - width - 1))
            rect = (r0, c0, r0 + height - 1, c0 + width - 1)
            support = _expand_rect(rect, 1)
            if support[0] < 0 or support[1] < 0 or support[2] >= h or support[3] >= w:
                continue
            if not _rect_clear(support, supports, gap=1):
                continue
            corridors.append(rect)
            supports.append(support)
            placed = True
            break
        if not placed:
            return None
    return corridors, supports


def _ray_info(corridors: np.ndarray, r: int, c: int) -> dict[str, tuple[int, bool]]:
    h, w = map(int, corridors.shape)

    def walk(dr: int, dc: int) -> tuple[int, bool]:
        rr, cc = r + dr, c + dc
        steps = 0
        while 0 <= rr < h and 0 <= cc < w:
            if corridors[rr, cc]:
                return steps, True
            steps += 1
            rr += dr
            cc += dc
        return steps, False

    return {
        "left": walk(0, -1),
        "right": walk(0, 1),
        "up": walk(-1, 0),
        "down": walk(1, 0),
    }


def _row_col_reach(corridors: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    if corridors[r1, c1]:
        return False
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            if corridors[r0, c]:
                return False
        return True
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            if corridors[r, c0]:
                return False
        return True
    raise ValueError("reachability must stay on one row or one column")


def _candidate_centers(
    grid: np.ndarray, corridors: np.ndarray, supports: list[tuple[int, int, int, int]]
) -> list[dict]:
    h, w = map(int, grid.shape)
    support_mask = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in supports:
        support_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    candidates: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if np.any(support_mask[r - 1 : r + 2, c - 1 : c + 2]):
                continue
            info = _ray_info(corridors, r, c)
            row_trunc = info["left"][1] or info["right"][1]
            col_trunc = info["up"][1] or info["down"][1]
            longest = max(info["left"][0], info["right"][0], info["up"][0], info["down"][0])
            if not (row_trunc or col_trunc) or longest < 3:
                continue
            candidates.append(
                {
                    "r": r,
                    "c": c,
                    "row_trunc": row_trunc,
                    "col_trunc": col_trunc,
                    "info": info,
                }
            )
    return candidates


def _choose_blocks(
    grid: np.ndarray,
    corridors: np.ndarray,
    supports: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> list[Block] | None:
    candidates = _candidate_centers(grid, corridors, supports)
    if len(candidates) < 2:
        return None

    support_mask = np.zeros_like(corridors, dtype=bool)
    for r0, c0, r1, c1 in supports:
        support_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    for _ in range(400):
        target = int(rng.integers(2, min(4, len(BLOCK_COLORS)) + 1))
        pool = list(rng.permutation(len(candidates)))
        chosen: list[dict] = []
        for idx in pool:
            cand = candidates[int(idx)]
            if any(abs(cand["r"] - prev["r"]) <= 3 or abs(cand["c"] - prev["c"]) <= 3 for prev in chosen):
                continue
            if any(support_mask[cand["r"], prev["c"]] or support_mask[prev["r"], cand["c"]] for prev in chosen):
                continue
            chosen.append(cand)
            if len(chosen) == target:
                break
        if len(chosen) < 2:
            continue
        if not any(c["row_trunc"] for c in chosen):
            continue
        if not any(c["col_trunc"] for c in chosen):
            continue

        reachable_crossings = 0
        for a, b in itertools.permutations(chosen, 2):
            if support_mask[a["r"], b["c"]]:
                continue
            if _row_col_reach(corridors, a["r"], a["c"], a["r"], b["c"]) and _row_col_reach(
                corridors, b["r"], b["c"], a["r"], b["c"]
            ):
                reachable_crossings += 1
        if reachable_crossings == 0:
            continue

        colors = list(rng.choice(np.array(BLOCK_COLORS, dtype=int), size=len(chosen), replace=False))
        blocks: list[Block] = []
        for cand, color in zip(chosen, colors):
            r = int(cand["r"])
            c = int(cand["c"])
            blocks.append((r, c, int(color), (r - 1, c - 1, r + 1, c + 1)))
        return sorted(blocks)
    return None


def _stamp_blocks(grid: np.ndarray, blocks: list[Block]) -> np.ndarray:
    out = grid.copy()
    for r, c, color, _bbox in blocks:
        out[r - 1 : r + 2, c - 1 : c + 2] = color
    return out


def _semantic_ok(
    grid: np.ndarray,
    intended_corridors: np.ndarray,
    blocks: list[Block],
) -> bool:
    detected = _detect_corridors(grid)
    if not np.array_equal(detected, intended_corridors):
        return False

    detected_blocks = _solid_3x3_blocks(grid, BACKGROUND)
    if [(r, c, color) for r, c, color, _ in detected_blocks] != [
        (r, c, color) for r, c, color, _ in blocks
    ]:
        return False

    pre_reset = _draw_rays(grid, intended_corridors, blocks)
    post_reset = _reset_crossings(pre_reset, intended_corridors, blocks)
    final = post_reset.copy()
    final[intended_corridors] = GREEN

    if not np.any(pre_reset != post_reset):
        return False
    if not np.any((pre_reset != _draw_rays(grid, np.zeros_like(intended_corridors), blocks)) & ~intended_corridors):
        return False
    if not np.any(intended_corridors):
        return False
    if np.array_equal(post_reset, final):
        return False
    if np.array_equal(final, grid):
        return False

    for r, c, _color, _bbox in blocks:
        info = _ray_info(intended_corridors, r, c)
        if not (info["left"][1] or info["right"][1] or info["up"][1] or info["down"][1]):
            return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(19, 25))
    w = int(rng.integers(19, 25))
    chosen = _choose_corridors(h, w, rng)
    if chosen is None:
        return None
    rects, supports = chosen
    grid = _paint_support_and_clutter((h, w), supports, rng)

    intended_corridors = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in rects:
        intended_corridors[r0 : r1 + 1, c0 : c1 + 1] = True

    blocks = _choose_blocks(grid, intended_corridors, supports, rng)
    if blocks is None:
        return None
    inp = _stamp_blocks(grid, blocks)

    if not _semantic_ok(inp, intended_corridors, blocks):
        return None

    expected = _apply_rule_with_known_corridors(inp, intended_corridors, blocks)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid hidden-corridor cross input")
