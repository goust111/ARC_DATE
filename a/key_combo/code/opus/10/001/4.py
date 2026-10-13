from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7
BG = 0
BLOCK_SIZE = 3
MAX_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _rows_overlap(r0: int, r1: int) -> bool:
    return abs(int(r0) - int(r1)) < BLOCK_SIZE


def _rects_touch_or_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) < BLOCK_SIZE + 1 and abs(int(a[1]) - int(b[1])) < BLOCK_SIZE + 1


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blocks: list[dict] = []
    for color in (BLUE, RED):
        for comp in _component_coords(g == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(comp) == BLOCK_SIZE * BLOCK_SIZE, "each block must be a solid 3x3 square"
            assert r1 - r0 + 1 == BLOCK_SIZE and c1 - c0 + 1 == BLOCK_SIZE, "components must be 3x3"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "block interior must be solid"
            blocks.append({"color": color, "r0": r0, "c0": c0})
    assert 4 <= len(blocks) <= 6, "expected 4 to 6 blocks"
    return blocks


def _mark_candidates(block: dict) -> list[tuple[int, int, int]]:
    r0 = int(block["r0"])
    c0 = int(block["final_c"])
    if int(block["distance"]) % 2 == 1:
        return [
            (r0 - 1, c0 - 1, YELLOW),
            (r0 - 1, c0 + BLOCK_SIZE, YELLOW),
            (r0 + BLOCK_SIZE, c0 - 1, YELLOW),
            (r0 + BLOCK_SIZE, c0 + BLOCK_SIZE, YELLOW),
        ]
    return [
        (r0 - 1, c0 + 1, ORANGE),
        (r0 + BLOCK_SIZE, c0 + 1, ORANGE),
        (r0 + 1, c0 - 1, ORANGE),
        (r0 + 1, c0 + BLOCK_SIZE, ORANGE),
    ]


def _solve_blocks(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[tuple[int, int], set[int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocks = _extract_blocks(g)

    out = np.zeros_like(g)
    final_blocks: list[dict] = []

    for color in (BLUE, RED):
        side_blocks = [b.copy() for b in blocks if int(b["color"]) == color]
        if color == BLUE:
            side_blocks.sort(key=lambda b: int(b["c0"]))
            edge_c = 0
            inner_c = BLOCK_SIZE
        else:
            side_blocks.sort(key=lambda b: -int(b["c0"]))
            edge_c = w - BLOCK_SIZE
            inner_c = w - 2 * BLOCK_SIZE
        assert inner_c >= 0, "grid too narrow for inward placement"

        placed_rows: list[int] = []
        for block in side_blocks:
            overlap = any(_rows_overlap(int(block["r0"]), prev_r0) for prev_r0 in placed_rows)
            final_c = inner_c if overlap else edge_c
            dist = abs(int(final_c) - int(block["c0"]))
            placed_rows.append(int(block["r0"]))
            final = {
                "color": int(color),
                "r0": int(block["r0"]),
                "c0": int(block["c0"]),
                "final_c": int(final_c),
                "distance": int(dist),
                "inward": bool(overlap),
            }
            final_blocks.append(final)
            out[final["r0"] : final["r0"] + BLOCK_SIZE, final["final_c"] : final["final_c"] + BLOCK_SIZE] = final["color"]

    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    for block in final_blocks:
        for r, c, color in _mark_candidates(block):
            if not _in_bounds(g.shape, r, c):
                continue
            if out[r, c] != BG:
                continue
            claims[(r, c)].add(int(color))

    for (r, c), colors in claims.items():
        if len(colors) == 1 and out[r, c] == BG:
            out[r, c] = next(iter(colors))

    return out, final_blocks, claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide blue and red 3x3 blocks to opposite edges, then mark different neighborhoods based on each block's true travel distance.

    Input structure:
    - The grid is black background with 4 to 6 solid 3x3 blocks.
    - Every block is monochrome blue or red, and the blocks are separated in the input.
    - No other foreground objects appear; rows stay meaningful because blocks keep their original top row after sliding.

    Rule:
    - Blue blocks slide left and red blocks slide right, each keeping its original row.
    - For each color separately, process blocks nearest their target edge first; if a block's row range overlaps any already placed same-color block, place it one block-width inward instead.
    - Compute the block's actual horizontal travel distance from its original top-left column to its final top-left column.
    - Odd travel distance marks the four outside diagonal corner-neighbor cells yellow; even travel distance marks the four outside orthogonal side-neighbor cells orange; only paint currently black cells and never overwrite a block.

    Key insight:
    - The overlap-driven inward shift changes the final position and also flips the parity of the travel distance, so the visible neighborhood type can only be determined by simulating the ordered slide first.
    """
    out, _final_blocks, _claims = _solve_blocks(grid)
    return out


def _sample_rows_for_side(height: int, count: int, has_inward: bool, rng: random.Random) -> list[tuple[int, bool]]:
    rows = list(range(height - BLOCK_SIZE + 1))
    for _ in range(200):
        rng.shuffle(rows)
        if has_inward:
            edge_rows: list[int] = []
            for r in rows:
                if all(not _rows_overlap(r, prev) for prev in edge_rows):
                    edge_rows.append(r)
                    if len(edge_rows) == count - 1:
                        break
            if len(edge_rows) != count - 1:
                continue

            preferred = [r for r in rows if any(_rows_overlap(r, e) for e in edge_rows) and all(r != e for e in edge_rows)]
            fallback = [r for r in rows if any(_rows_overlap(r, e) for e in edge_rows)]
            inward_choices = preferred if preferred else fallback
            if not inward_choices:
                continue
            inward_row = rng.choice(inward_choices)
            placed = [(r, False) for r in edge_rows] + [(inward_row, True)]
            rng.shuffle(placed[:-1])
            return placed

        chosen: list[int] = []
        for r in rows:
            if all(not _rows_overlap(r, prev) for prev in chosen):
                chosen.append(r)
                if len(chosen) == count:
                    break
        if len(chosen) == count:
            return [(r, False) for r in chosen]
    raise RuntimeError("failed to sample rows")


def _search_side_columns(
    ordered_blocks: list[dict],
    domain: list[int],
    ascending: bool,
    rng: random.Random,
) -> bool:
    domain = list(domain)
    if ascending:
        domain.sort()
    else:
        domain.sort(reverse=True)

    def valid(block: dict, c: int, assigned: list[int]) -> bool:
        final_c = int(block["final_c"])
        want_odd = bool(block["odd"])
        if c == final_c:
            return False
        if (abs(c - final_c) % 2 == 1) != want_odd:
            return False
        if assigned:
            prev_c = assigned[-1]
            if ascending and c <= prev_c:
                return False
            if not ascending and c >= prev_c:
                return False
        for prev_block in ordered_blocks[: len(assigned)]:
            if _rects_touch_or_overlap((int(block["r0"]), c), (int(prev_block["r0"]), int(prev_block["c0"]))):
                return False
        return True

    assigned_cols: list[int] = []

    def backtrack(idx: int) -> bool:
        if idx == len(ordered_blocks):
            return True
        block = ordered_blocks[idx]
        candidates = [c for c in domain if valid(block, c, assigned_cols)]
        rng.shuffle(candidates)
        for c in candidates:
            assigned_cols.append(c)
            block["c0"] = int(c)
            if backtrack(idx + 1):
                return True
            assigned_cols.pop()
            block.pop("c0", None)
        return False

    return backtrack(0)


def _build_blocks(height: int, width: int, rng: random.Random) -> list[dict] | None:
    n_blue = rng.randint(2, 3)
    n_red = rng.randint(2, 3)
    if n_blue + n_red > 6:
        return None

    overlap_color = rng.choice([BLUE, RED])
    red_has_inward = overlap_color == RED or (n_red == 3 and rng.random() < 0.45)
    blue_has_inward = overlap_color == BLUE or (n_blue == 3 and rng.random() < 0.45)

    blue_rows = _sample_rows_for_side(height, n_blue, blue_has_inward, rng)
    red_rows = _sample_rows_for_side(height, n_red, red_has_inward, rng)

    blocks: list[dict] = []
    for color, rows in ((BLUE, blue_rows), (RED, red_rows)):
        final_edge = 0 if color == BLUE else width - BLOCK_SIZE
        final_inner = BLOCK_SIZE if color == BLUE else width - 2 * BLOCK_SIZE
        side_blocks = []
        for r0, inward in rows:
            side_blocks.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "final_c": int(final_inner if inward else final_edge),
                    "inward": bool(inward),
                }
            )

        base_odd = bool(rng.randint(0, 1))
        seen_non_inward = False
        for block in side_blocks:
            if block["inward"]:
                block["odd"] = not base_odd
            else:
                block["odd"] = base_odd if not seen_non_inward else bool(rng.randint(0, 1))
                seen_non_inward = True
        rng.shuffle(side_blocks)
        blocks.extend(side_blocks)

    if not any(b["inward"] for b in blocks):
        return None
    return blocks


def _assign_input_columns(blocks: list[dict], width: int, rng: random.Random) -> bool:
    blue_order = [b for b in blocks if int(b["color"]) == BLUE]
    red_order = [b for b in blocks if int(b["color"]) == RED]

    blue_order.sort(key=lambda b: (int(b["inward"]), int(b["r0"])))
    red_order.sort(key=lambda b: (int(b["inward"]), int(b["r0"])))

    blue_domain = list(range(0, min(6, width - BLOCK_SIZE + 1)))
    red_domain = list(range(9, width - BLOCK_SIZE + 1))
    if len(blue_domain) < len(blue_order) or len(red_domain) < len(red_order):
        return False

    if not _search_side_columns(blue_order, blue_domain, ascending=True, rng=rng):
        return False
    if not _search_side_columns(red_order, red_domain, ascending=False, rng=rng):
        return False
    return True


def _render_input(shape: tuple[int, int], blocks: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for block in blocks:
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        color = int(block["color"])
        grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = color
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        inp_blocks = _extract_blocks(grid)
        out, final_blocks, claims = _solve_blocks(grid)
    except AssertionError:
        return False

    if any(_rects_touch_or_overlap((a["r0"], a["c0"]), (b["r0"], b["c0"])) for i, a in enumerate(inp_blocks) for b in inp_blocks[i + 1 :]):
        return False
    if any(
        abs(int(a["r0"]) - int(b["r0"])) < BLOCK_SIZE and abs(int(a["final_c"]) - int(b["final_c"])) < BLOCK_SIZE
        for i, a in enumerate(final_blocks)
        for b in final_blocks[i + 1 :]
    ):
        return False
    if not any(a["color"] == b["color"] and _rows_overlap(a["r0"], b["r0"]) for i, a in enumerate(inp_blocks) for b in inp_blocks[i + 1 :]):
        return False
    if not any(bool(block["inward"]) for block in final_blocks):
        return False

    parity_by_color: dict[int, set[int]] = defaultdict(set)
    for block in final_blocks:
        parity_by_color[int(block["color"])].add(int(block["distance"]) % 2)
        if block["distance"] <= 0:
            return False
    if len({int(block["distance"]) % 2 for block in final_blocks}) != 2:
        return False
    if not any(len(vals) == 2 for vals in parity_by_color.values()):
        return False

    claim_count = 0
    visible_by_block = []
    occupied = np.zeros_like(out, dtype=bool)
    for block in final_blocks:
        occupied[block["r0"] : block["r0"] + BLOCK_SIZE, block["final_c"] : block["final_c"] + BLOCK_SIZE] = True
    all_claim_positions: set[tuple[int, int]] = set()
    for block in final_blocks:
        visible = 0
        for r, c, color in _mark_candidates(block):
            if not _in_bounds(out.shape, r, c):
                continue
            if occupied[r, c]:
                continue
            all_claim_positions.add((r, c))
            if out[r, c] == color:
                visible += 1
        visible_by_block.append(visible)
    for colors in claims.values():
        if len(colors) != 1:
            return False
        claim_count += 1
    if claim_count != len(all_claim_positions):
        return False
    if not all(v >= 1 for v in visible_by_block):
        return False
    if not np.any(out == YELLOW) or not np.any(out == ORANGE):
        return False

    moved = any(block["final_c"] != (0 if block["color"] == BLUE else out.shape[1] - BLOCK_SIZE) for block in final_blocks)
    if not moved:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) + attempt)
        height = rng.randint(13, 18)
        width = rng.randint(16, 19)
        blocks = _build_blocks(height, width, rng)
        if blocks is None:
            continue
        if not _assign_input_columns(blocks, width, rng):
            continue
        grid = _render_input((height, width), blocks)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
