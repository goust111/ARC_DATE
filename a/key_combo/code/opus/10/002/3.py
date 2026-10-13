from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 2000


def _find_blocks(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    blocks: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            side_r = r1 - r0 + 1
            side_c = c1 - c0 + 1
            assert side_r == side_c, "foreground components must be squares"
            assert side_r in (2, 3), "square side must be 2 or 3"
            assert len(coords) == side_r * side_r, "square must be solid"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "square bbox must be filled"
            blocks.append(
                {
                    "color": color,
                    "side": side_r,
                    "area": side_r * side_r,
                    "r0": r0,
                    "c0": c0,
                    "center2": 2 * c0 + side_r - 1,
                }
            )

    assert 2 <= len(blocks) <= 3, "expected 2 or 3 blocks"
    colors = [block["color"] for block in blocks]
    assert len(set(colors)) == len(colors), "blocks must have distinct colors"
    blocks.sort(key=lambda block: block["c0"])
    for left, right in zip(blocks, blocks[1:]):
        assert left["c0"] + left["side"] - 1 < right["c0"], "column ranges must be disjoint"
    return blocks


def _winner(blocks: list[dict], claimants: list[int], mode: str) -> int:
    uniq = sorted(set(claimants))
    if mode == "correct":
        return max(uniq, key=lambda idx: (blocks[idx]["area"], -blocks[idx]["center2"]))
    if mode == "leftmost":
        return min(uniq, key=lambda idx: blocks[idx]["center2"])
    if mode == "rightmost":
        return max(uniq, key=lambda idx: blocks[idx]["center2"])
    raise ValueError(f"unknown mode {mode}")


def _render(blocks: list[dict], shape: tuple[int, int], *, use_input_rows: bool, mode: str) -> tuple[np.ndarray, dict, list[dict], list[set]]:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    body_mask = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    for block in blocks:
        side = int(block["side"])
        top = int(block["r0"]) if use_input_rows else h - side
        c0 = int(block["c0"])
        out[top : top + side, c0 : c0 + side] = int(block["color"])
        body_mask[top : top + side, c0 : c0 + side] = True
        placed.append({**block, "top": top})

    claims: dict[tuple[int, int], list[int]] = {}
    ray_pixels: list[set[tuple[int, int]]] = [set() for _ in placed]
    for idx, block in enumerate(placed):
        top = int(block["top"])
        c0 = int(block["c0"])
        side = int(block["side"])
        starts = [(top, c0, -1, -1), (top, c0 + side - 1, -1, 1)]
        for r, c, dr, dc in starts:
            rr = r
            cc = c
            while 0 <= rr < h and 0 <= cc < w:
                if not body_mask[rr, cc]:
                    claims.setdefault((rr, cc), []).append(idx)
                    ray_pixels[idx].add((rr, cc))
                rr += dr
                cc += dc

    conflicts: list[dict] = []
    for cell, claimants in claims.items():
        winner = _winner(placed, claimants, mode)
        out[cell] = int(placed[winner]["color"])
        uniq = sorted(set(claimants))
        if len(uniq) >= 2:
            conflicts.append({"cell": cell, "claimants": uniq, "winner": winner})
    return out, claims, conflicts, ray_pixels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each floating square to the floor, then emit upward diagonal rays from its landed top corners.

    Input structure:
    - The grid has black background and 2-3 disconnected solid square blocks.
    - Every block is monochromatic, has side length 2 or 3, and each block occupies a disjoint column range.
    - The blocks float above the bottom edge, so their landed positions are not already present in the input.

    Rule:
    - Move every square straight downward until its bottom edge touches the grid bottom.
    - From the landed square's top-left corner emit an up-left diagonal ray, and from its top-right corner emit an up-right diagonal ray.
    - Rays only paint background pixels; square body pixels stay unchanged. If rays from different blocks claim the same pixel, the larger-area block wins, and equal areas break toward the block with the smaller center column.

    Key insight:
    - The visible blocks are only launch templates: you must first infer their landed positions, because those hidden landing corners determine the ray origins and therefore the later area-based conflict outcomes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"
    assert 0 in set(map(int, np.unique(g))), "expected black background"
    blocks = _find_blocks(g)
    out, _claims, _conflicts, _ray_pixels = _render(blocks, g.shape, use_input_rows=False, mode="correct")
    return out


def _distribute_starts(width: int, sizes: list[int]) -> list[int] | None:
    min_gap = 1
    need = sum(sizes) + min_gap * (len(sizes) - 1)
    if need > width:
        return None
    extras = width - need
    slots = [0] * (len(sizes) + 1)
    for _ in range(extras):
        slots[random.randrange(len(slots))] += 1
    starts: list[int] = []
    c = slots[0]
    for i, side in enumerate(sizes):
        starts.append(c)
        c += side
        if i + 1 < len(sizes):
            c += min_gap + slots[i + 1]
    return starts


def _build_input(blocks: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for block in blocks:
        side = int(block["side"])
        landed_top = h - side
        if landed_top <= 0:
            return None
        top = random.randint(0, landed_top - 1)
        block["r0"] = top
        c0 = int(block["c0"])
        grid[top : top + side, c0 : c0 + side] = int(block["color"])
    return grid


def _has_target_conflict(blocks: list[dict], conflicts: list[dict], target: str) -> bool:
    if target == "area":
        for conflict in conflicts:
            claimants = conflict["claimants"]
            areas = {blocks[idx]["area"] for idx in claimants}
            if len(areas) >= 2:
                winner = conflict["winner"]
                best_area = max(blocks[idx]["area"] for idx in claimants)
                if blocks[winner]["area"] == best_area:
                    return True
        return False
    if target == "tie":
        for conflict in conflicts:
            claimants = conflict["claimants"]
            if len(claimants) != 2:
                continue
            a, b = claimants
            if blocks[a]["area"] != blocks[b]["area"]:
                continue
            left = a if blocks[a]["center2"] < blocks[b]["center2"] else b
            if conflict["winner"] == left:
                return True
        return False
    raise ValueError(f"unknown target {target}")


def _semantic_checks(inp: np.ndarray, blocks: list[dict], scenario: str) -> bool:
    correct_out, _claims, conflicts, ray_pixels = _render(blocks, inp.shape, use_input_rows=False, mode="correct")
    if len(conflicts) == 0:
        return False
    if any(len(pixels) == 0 for pixels in ray_pixels):
        return False
    if not np.array_equal(main(inp), correct_out):
        return False

    input_row_out, _c2, _f2, _p2 = _render(blocks, inp.shape, use_input_rows=True, mode="correct")
    if np.array_equal(input_row_out, correct_out):
        return False

    left_out, _c3, _f3, _p3 = _render(blocks, inp.shape, use_input_rows=False, mode="leftmost")
    right_out, _c4, _f4, _p4 = _render(blocks, inp.shape, use_input_rows=False, mode="rightmost")

    if scenario in {"area_left", "area_right"}:
        if not _has_target_conflict(blocks, conflicts, "area"):
            return False
        if scenario == "area_left" and np.array_equal(correct_out, right_out):
            return False
        if scenario == "area_right" and np.array_equal(correct_out, left_out):
            return False
    elif scenario == "tie":
        if not _has_target_conflict(blocks, conflicts, "tie"):
            return False
        if np.array_equal(correct_out, right_out):
            return False
    else:
        return False

    colored = int(np.count_nonzero(correct_out))
    if colored <= sum(block["area"] for block in blocks):
        return False
    if colored >= (inp.shape[0] * inp.shape[1]) // 2:
        return False
    return True


def _sample_blocks() -> tuple[tuple[int, int], list[dict], str] | None:
    scenario = random.choices(
        ["area_left", "area_right", "tie"],
        weights=[3, 3, 2],
        k=1,
    )[0]
    height = random.randint(13, 18)

    if scenario == "area_left":
        sizes = [3, 2]
    elif scenario == "area_right":
        sizes = [2, 3]
    else:
        sizes = random.choice([[2, 2, 3], [3, 2, 2], [3, 3, 2], [2, 3, 3]])

    min_width = sum(sizes) + (len(sizes) - 1)
    width = random.randint(max(12, min_width), 20)
    starts = _distribute_starts(width, sizes)
    if starts is None:
        return None

    colors = random.sample(list(range(1, 10)), len(sizes))
    blocks = [
        {
            "color": color,
            "side": side,
            "area": side * side,
            "r0": 0,
            "c0": c0,
            "center2": 2 * c0 + side - 1,
        }
        for color, side, c0 in zip(colors, sizes, starts)
    ]
    return (height, width), blocks, scenario


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_blocks()
    if sampled is None:
        return None
    shape, blocks, scenario = sampled
    inp = _build_input(blocks, shape)
    if inp is None:
        return None
    if not _semantic_checks(inp, blocks, scenario):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-ray input")
