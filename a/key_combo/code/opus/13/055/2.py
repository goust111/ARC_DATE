from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
RED = 2
ORANGE = 7
TEAL = 8
BLOCK = 2
MAX_GENERATION_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 200
INPUT_ATTEMPTS = 120


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_blocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blocks: list[dict] = []
    for color in (RED, TEAL):
        for coords in _mask_components(g == color):
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            assert coords.shape[0] == 4, "each block must have area 4"
            assert (r1 - r0 + 1, c1 - c0 + 1) == (BLOCK, BLOCK), "each block must be 2x2"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "each block must be a solid 2x2 square"
            blocks.append({"r": r0, "c": c0, "color": color})

    blocks.sort(key=lambda block: (int(block["r"]), int(block["c"]), int(block["color"])))
    red_count = sum(int(block["color"]) == RED for block in blocks)
    teal_count = sum(int(block["color"]) == TEAL for block in blocks)
    assert red_count == 1, "expected exactly one red block"
    assert 4 <= teal_count <= 6, "expected 4-6 teal blocks"
    return blocks


def _render_blocks(blocks: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.zeros((h, w), dtype=int)
    for block in blocks:
        r = int(block["r"])
        c = int(block["c"])
        color = int(block["color"])
        assert 0 <= r <= h - BLOCK and 0 <= c <= w - BLOCK, "block out of bounds"
        patch = grid[r : r + BLOCK, c : c + BLOCK]
        assert np.all(patch == BLACK), "blocks must not overlap"
        patch[:] = color
    return grid


def _copy_blocks(blocks: list[dict]) -> list[dict]:
    return [dict(block) for block in blocks]


def _settle_blocks(blocks: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = map(int, shape)
    settled = _copy_blocks(blocks)
    for _ in range(w + 2):
        occ = np.full((h, w), -1, dtype=int)
        for idx, block in enumerate(settled):
            r = int(block["r"])
            c = int(block["c"])
            occ[r : r + BLOCK, c : c + BLOCK] = idx

        movable: list[int] = []
        for idx, block in enumerate(settled):
            r = int(block["r"])
            c = int(block["c"])
            if c + BLOCK >= w:
                continue
            can_move = True
            for rr in range(r, r + BLOCK):
                for cc in range(c + 1, c + BLOCK + 1):
                    owner = int(occ[rr, cc])
                    if owner not in (-1, idx):
                        can_move = False
                        break
                if not can_move:
                    break
            if can_move:
                movable.append(idx)

        if not movable:
            return settled

        for idx in movable:
            settled[idx]["c"] = int(settled[idx]["c"]) + 1

    raise AssertionError("gravity did not settle within the grid width")


def _visible_step(grid: np.ndarray, a: dict, b: dict) -> tuple[str, int] | None:
    ar, ac = int(a["r"]), int(a["c"])
    br, bc = int(b["r"]), int(b["c"])

    if ar == br:
        c0 = min(ac, bc) + BLOCK
        c1 = max(ac, bc) - 1
        if c0 <= c1 and not np.all(grid[ar : ar + BLOCK, c0 : c1 + 1] == BLACK):
            return None
        return "h", abs(ac - bc)

    if ac == bc:
        r0 = min(ar, br) + BLOCK
        r1 = max(ar, br) - 1
        if r0 <= r1 and not np.all(grid[r0 : r1 + 1, ac : ac + BLOCK] == BLACK):
            return None
        return "v", abs(ar - br)

    return None


def _paint_corridor(out: np.ndarray, a: dict, b: dict, axis: str) -> None:
    ar, ac = int(a["r"]), int(a["c"])
    br, bc = int(b["r"]), int(b["c"])
    if axis == "h":
        c0 = min(ac, bc) + BLOCK
        c1 = max(ac, bc) - 1
        if c0 <= c1:
            out[ar : ar + BLOCK, c0 : c1 + 1] = ORANGE
        return

    r0 = min(ar, br) + BLOCK
    r1 = max(ar, br) - 1
    if r0 <= r1:
        out[r0 : r1 + 1, ac : ac + BLOCK] = ORANGE


def _greedy_steps(blocks: list[dict], shape: tuple[int, int]) -> tuple[list[dict], set[int]]:
    base = _render_blocks(blocks, shape)
    start = next(idx for idx, block in enumerate(blocks) if int(block["color"]) == RED)
    unvisited = {idx for idx, block in enumerate(blocks) if int(block["color"]) == TEAL}
    current = start
    steps: list[dict] = []

    while True:
        candidates = []
        for idx in sorted(unvisited):
            step = _visible_step(base, blocks[current], blocks[idx])
            if step is None:
                continue
            axis, distance = step
            candidates.append((distance, int(blocks[idx]["r"]), int(blocks[idx]["c"]), idx, axis))

        if not candidates:
            break

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        min_distance = int(candidates[0][0])
        distance_tie = sum(int(item[0]) == min_distance for item in candidates) > 1
        distance, _row, _col, idx, axis = candidates[0]
        steps.append(
            {
                "from": current,
                "to": idx,
                "axis": axis,
                "distance": int(distance),
                "distance_tie": bool(distance_tie),
            }
        )
        unvisited.remove(idx)
        current = idx

    return steps, unvisited


def _apply_walk(blocks: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, list[dict], set[int]]:
    out = _render_blocks(blocks, shape)
    steps, remaining = _greedy_steps(blocks, shape)
    for step in steps:
        _paint_corridor(out, blocks[step["from"]], blocks[step["to"]], str(step["axis"]))
    return out, steps, remaining


def _step_orange_area(blocks: list[dict], step: dict) -> int:
    a = blocks[int(step["from"])]
    b = blocks[int(step["to"])]
    if str(step["axis"]) == "h":
        return max(0, abs(int(a["c"]) - int(b["c"])) - BLOCK) * BLOCK
    return max(0, abs(int(a["r"]) - int(b["r"])) - BLOCK) * BLOCK


def _step_signature(steps: list[dict]) -> tuple[tuple[str, int, int], ...]:
    return tuple((str(step["axis"]), int(step["from"]), int(step["to"])) for step in steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle all 2x2 blocks to the right, then paint the greedy visible walk from the red block through teal blocks.

    Input structure:
    - Black cells are background.
    - The foreground contains exactly one red 2x2 start block and 4-6 teal 2x2 target blocks.
    - Blocks are rigid squares; only red and teal blocks participate in the gravity phase.

    Rule:
    - Move every 2x2 block one column right at a time whenever the shifted square stays in bounds and does not collide with another block doing the same synchronous step.
    - Repeat until no block can move.
    - On the settled layout, start from the red block and repeatedly choose the nearest unvisited teal block that is visible in the same 2-row band or 2-column band; when distances tie, prefer the teal block with the smaller row.
    - Paint orange only on background cells in the corridor between consecutive chosen blocks, keep all red and teal pixels unchanged, and stop when no unvisited visible teal block remains.

    Key insight:
    - The traversal must be planned on the post-gravity geometry: settling changes which blocks line up, which corridors stay clear, and which teal block is the greedy next choice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    blocks = _find_blocks(g)
    settled = _settle_blocks(blocks, g.shape)
    out, _steps, _remaining = _apply_walk(settled, g.shape)
    return out


def _compositions(total: int, parts: int, max_part: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, cur: list[int]) -> None:
        if slots == 0:
            if remaining == 0:
                out.append(tuple(cur))
            return
        low = 1
        high = min(max_part, remaining - slots + 1)
        for value in range(low, high + 1):
            cur.append(value)
            rec(remaining - value, slots - 1, cur)
            cur.pop()

    rec(total, parts, [])
    return out


GEN_COMPOSITIONS = tuple(
    comp
    for total in range(5, 8)
    for parts in range(3, 5)
    for comp in _compositions(total, parts, max_part=2)
    if any(value == 2 for value in comp)
)


def _sample_rows(counts: tuple[int, ...], mode: str, rng: random.Random) -> tuple[list[int], int]:
    top_margin = rng.randint(1, 2)
    if mode == "tie":
        gaps = [rng.randint(2, 4) for _ in range(len(counts) - 1)]
        if gaps:
            gaps[rng.randrange(len(gaps))] = 2
        if not any(gap >= 3 for gap in gaps):
            gaps[rng.randrange(len(gaps))] = 3
    else:
        gaps = [rng.randint(3, 5) for _ in range(len(counts) - 1)]

    rows = [top_margin]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    height = rows[-1] + BLOCK + rng.randint(1, 2)
    return rows, height


def _build_settled_candidate(
    counts: tuple[int, ...], rows: list[int], width: int, red_band: int
) -> list[dict]:
    blocks: list[dict] = []
    for band, count in enumerate(counts):
        for slot in range(count):
            blocks.append(
                {
                    "r": int(rows[band]),
                    "c": int(width - BLOCK * (slot + 1)),
                    "color": RED if band == red_band and slot == 1 else TEAL,
                    "band": int(band),
                    "slot": int(slot),
                }
            )
    return blocks


def _settled_layout_ok(mode: str, blocks: list[dict], steps: list[dict]) -> bool:
    if len(steps) < 3:
        return False
    axes = [str(step["axis"]) for step in steps]
    if steps[0]["axis"] != "h":
        return False
    if "v" not in axes:
        return False
    if sum(axis == "h" for axis in axes) > 2:
        return False
    if not any(_step_orange_area(blocks, step) > 0 for step in steps):
        return False
    has_tie = any(bool(step["distance_tie"]) for step in steps)
    return has_tie if mode == "tie" else not has_tie


def _sample_input_blocks(settled: list[dict], shape: tuple[int, int], rng: random.Random) -> list[dict] | None:
    _h, w = map(int, shape)
    used_cols: set[int] = set()
    input_blocks: list[dict] = []

    for band in sorted({int(block["band"]) for block in settled}):
        band_blocks = sorted((block for block in settled if int(block["band"]) == band), key=lambda block: int(block["slot"]))
        if len(band_blocks) == 1:
            candidates = [col for col in range(1, w - 2) if col not in used_cols]
            if not candidates:
                return None
            col = rng.choice(candidates)
            used_cols.add(col)
            settled_block = band_blocks[0]
            input_blocks.append(
                {
                    "r": int(settled_block["r"]),
                    "c": int(col),
                    "color": int(settled_block["color"]),
                    "band": int(settled_block["band"]),
                    "slot": int(settled_block["slot"]),
                }
            )
            continue

        assert len(band_blocks) == 2, "generator only uses 1- or 2-block bands"
        right_block, left_block = band_blocks[0], band_blocks[1]
        same_color_gap = 3 if int(right_block["color"]) == int(left_block["color"]) else 2
        right_candidates = [col for col in range(1 + same_color_gap, w - 2) if col not in used_cols]
        if not right_candidates:
            return None
        right_col = rng.choice(right_candidates)
        left_candidates = [col for col in range(1, right_col - same_color_gap + 1) if col not in used_cols]
        if not left_candidates:
            return None
        left_col = rng.choice(left_candidates)
        used_cols.add(right_col)
        used_cols.add(left_col)
        for settled_block, col in ((right_block, right_col), (left_block, left_col)):
            input_blocks.append(
                {
                    "r": int(settled_block["r"]),
                    "c": int(col),
                    "color": int(settled_block["color"]),
                    "band": int(settled_block["band"]),
                    "slot": int(settled_block["slot"]),
                }
            )

    if len(used_cols) != len(input_blocks):
        return None
    if max(used_cols) >= w - 2:
        return None
    return sorted(input_blocks, key=lambda block: (int(block["r"]), int(block["c"]), int(block["color"])))


def _validate_generated_instance(grid: np.ndarray, mode: str) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        blocks = _find_blocks(grid)
    except AssertionError:
        return False
    if len({int(block["c"]) for block in blocks}) != len(blocks):
        return False

    settled = _settle_blocks(blocks, grid.shape)
    settled_grid = _render_blocks(settled, grid.shape)
    out, settled_steps, _remaining = _apply_walk(settled, grid.shape)
    input_steps, _input_remaining = _greedy_steps(blocks, grid.shape)

    if not _settled_layout_ok(mode, settled, settled_steps):
        return False
    if not input_steps:
        return False
    if any(str(step["axis"]) == "v" for step in input_steps):
        return False
    if len(input_steps) >= len(settled_steps):
        return False
    if _step_signature(input_steps) == _step_signature(settled_steps):
        return False
    if np.array_equal(settled_grid, grid):
        return False
    if int(np.count_nonzero(out == ORANGE)) == 0:
        return False
    if not np.array_equal(main(grid), out):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = "tie" if rng.random() < 0.28 else "standard"

    for _ in range(LAYOUT_ATTEMPTS):
        counts = rng.choice(GEN_COMPOSITIONS)
        red_bands = [band for band, count in enumerate(counts) if count == 2]
        if not red_bands:
            continue

        rows, height = _sample_rows(counts, mode, rng)
        width = rng.randint(max(15, 2 * max(counts) + 10), 24)
        red_band = rng.choice(red_bands)
        settled = _build_settled_candidate(counts, rows, width, red_band)
        settled_steps, _remaining = _greedy_steps(settled, (height, width))
        if not _settled_layout_ok(mode, settled, settled_steps):
            continue

        for _ in range(INPUT_ATTEMPTS):
            candidate = _sample_input_blocks(settled, (height, width), rng)
            if candidate is None:
                continue
            if sorted((int(b["r"]), int(b["c"]), int(b["color"])) for b in _settle_blocks(candidate, (height, width))) != sorted(
                (int(b["r"]), int(b["c"]), int(b["color"])) for b in settled
            ):
                continue
            grid = _render_blocks(candidate, (height, width))
            if _validate_generated_instance(grid, mode):
                return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-greedy-path input")
