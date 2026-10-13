from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
BLOCK_SHAPES = [(2, 2), (2, 3), (3, 2)]
MAX_GENERATION_ATTEMPTS = 400

_BRANCH_BAG: list[tuple[str, int, int]] = []


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_axis(grid: np.ndarray) -> tuple[str, int]:
    h, w = grid.shape
    rows = [r for r in range(h) if np.all(grid[r, :] == YELLOW)]
    cols = [c for c in range(w) if np.all(grid[:, c] == YELLOW)]
    candidates = [("h", r) for r in rows] + [("v", c) for c in cols]
    assert len(candidates) == 1, "expected exactly one full yellow axis"
    orient, axis = candidates[0]
    if orient == "h":
        assert h == 2 * axis + 1, "horizontal axis must split the grid into equal halves"
    else:
        assert w == 2 * axis + 1, "vertical axis must split the grid into equal halves"
    return orient, int(axis)


def _mirror_coord(orient: str, axis: int, coord: tuple[int, int]) -> tuple[int, int]:
    r, c = coord
    if orient == "h":
        return 2 * axis - r, c
    return r, 2 * axis - c


def _sign_candidates(grid: np.ndarray, orient: str, axis: int, bg: int) -> list[list[tuple[int, int]]]:
    mask = (grid != bg) & (grid != YELLOW)
    candidates: list[list[tuple[int, int]]] = []
    for comp in _components(mask):
        coords = [tuple(map(int, rc)) for rc in comp]
        if len(coords) != 2:
            continue
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(coords)
        if orient == "h":
            if not (c0 == c1 and abs(r0 - r1) == 1):
                continue
        else:
            if not (r0 == r1 and abs(c0 - c1) == 1):
                continue
        mirrors = [_mirror_coord(orient, axis, coord) for coord in coords]
        if any(int(grid[mr, mc]) != bg for mr, mc in mirrors):
            continue
        candidates.append(coords)
    return candidates


def _extract_sign(grid: np.ndarray, orient: str, axis: int, bg: int) -> tuple[set[tuple[int, int]], set[int], int]:
    candidates = _sign_candidates(grid, orient, axis, bg)
    assert len(candidates) == 1, "expected exactly one detached two-color adjudication sign"
    coords = candidates[0]
    if orient == "h":
        near = min(coords, key=lambda rc: abs(rc[0] - axis))
    else:
        near = min(coords, key=lambda rc: abs(rc[1] - axis))
    sign_colors = {int(grid[r, c]) for r, c in coords}
    winner = int(grid[near])
    return set(coords), sign_colors, winner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve mirror mismatches across the yellow axis by filling missing cells and adjudicating color conflicts from the detached two-color sign.

    Input structure:
    - The background is black, and one full yellow row or column is the unique symmetry axis.
    - Away from the axis, the two halves contain small foreground fragments in two non-yellow colors and are almost mirror images.
    - One detached 1x2 two-color sign, oriented perpendicular to the axis, sits on one side only and has no mirrored partner.

    Rule:
    - Keep the yellow axis and the detached sign unchanged.
    - For every non-sign pair of cells mirrored across the axis: if one side is foreground and the other is background, copy the foreground color onto the background side.
    - If the mirrored pair contains the two sign colors in conflict, recolor both cells to the sign-end color that is nearer to the yellow axis.

    Key insight:
    - The small two-cell sign is not another fragment to complete; its inner-vs-outer order is the only rule that decides every true color conflict.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    orient, axis = _find_axis(g)
    sign_coords, sign_colors, winner = _extract_sign(g, orient, axis, bg)
    out = g.copy()

    if orient == "h":
        for delta in range(1, axis + 1):
            r0 = axis - delta
            r1 = axis + delta
            for c in range(g.shape[1]):
                if (r0, c) in sign_coords or (r1, c) in sign_coords:
                    continue
                a = int(g[r0, c])
                b = int(g[r1, c])
                if a == bg and b == bg:
                    continue
                if a == bg:
                    out[r0, c] = b
                elif b == bg:
                    out[r1, c] = a
                elif a != b:
                    assert {a, b} == sign_colors, "conflicts must use the sign colors"
                    out[r0, c] = winner
                    out[r1, c] = winner
    else:
        for delta in range(1, axis + 1):
            c0 = axis - delta
            c1 = axis + delta
            for r in range(g.shape[0]):
                if (r, c0) in sign_coords or (r, c1) in sign_coords:
                    continue
                a = int(g[r, c0])
                b = int(g[r, c1])
                if a == bg and b == bg:
                    continue
                if a == bg:
                    out[r, c0] = b
                elif b == bg:
                    out[r, c1] = a
                elif a != b:
                    assert {a, b} == sign_colors, "conflicts must use the sign colors"
                    out[r, c0] = winner
                    out[r, c1] = winner

    return out


def _next_branch() -> tuple[str, int, int]:
    global _BRANCH_BAG
    if not _BRANCH_BAG:
        _BRANCH_BAG = [
            (orient, winner_idx, sign_side)
            for orient in ("h", "v")
            for winner_idx in (0, 1)
            for sign_side in (-1, 1)
        ]
        random.shuffle(_BRANCH_BAG)
    return _BRANCH_BAG.pop()


def _reserve_negative_for_sign(
    orient: str,
    axis: int,
    sign_coords: list[tuple[int, int]],
    neg_h: int,
    neg_w: int,
) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for coord in sign_coords:
        cand = coord if (coord[0] < axis if orient == "h" else coord[1] < axis) else _mirror_coord(orient, axis, coord)
        reserved.add(cand)
        r, c = cand
        for nr in range(max(0, r - 1), min(neg_h, r + 2)):
            for nc in range(max(0, c - 1), min(neg_w, c + 2)):
                reserved.add((nr, nc))
    return reserved


def _can_place_block(
    occ: np.ndarray,
    reserved: set[tuple[int, int]],
    top: int,
    left: int,
    height: int,
    width: int,
) -> bool:
    h, w = occ.shape
    if top + height > h or left + width > w:
        return False
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    if occ[r0:r1, c0:c1].any():
        return False
    for r in range(top, top + height):
        for c in range(left, left + width):
            if (r, c) in reserved:
                return False
    return True


def _place_blocks(
    neg_h: int,
    neg_w: int,
    reserved: set[tuple[int, int]],
    block_colors: list[int],
) -> list[dict] | None:
    occ = np.zeros((neg_h, neg_w), dtype=bool)
    blocks: list[dict] = []
    for color in block_colors:
        for _ in range(80):
            height, width = random.choice(BLOCK_SHAPES)
            candidates = []
            for top in range(neg_h - height + 1):
                for left in range(neg_w - width + 1):
                    if _can_place_block(occ, reserved, top, left, height, width):
                        candidates.append((top, left))
            if not candidates:
                continue
            top, left = random.choice(candidates)
            cells = [(r, c) for r in range(top, top + height) for c in range(left, left + width)]
            for r, c in cells:
                occ[r, c] = True
            blocks.append({"color": color, "cells": cells})
            break
        else:
            return None
    return blocks


def _paint_output(
    orient: str,
    axis: int,
    grid_shape: tuple[int, int],
    sign_coords: list[tuple[int, int]],
    sign_colors: tuple[int, int],
    winner: int,
    blocks: list[dict],
) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    if orient == "h":
        out[axis, :] = YELLOW
    else:
        out[:, axis] = YELLOW

    loser = sign_colors[0] if sign_colors[1] == winner else sign_colors[1]
    assert winner != loser

    for block in blocks:
        for cell in block["cells"]:
            mirror = _mirror_coord(orient, axis, cell)
            out[cell] = block["color"]
            out[mirror] = block["color"]

    near = None
    if orient == "h":
        near = min(sign_coords, key=lambda rc: abs(rc[0] - axis))
    else:
        near = min(sign_coords, key=lambda rc: abs(rc[1] - axis))
    far = next(coord for coord in sign_coords if coord != near)
    out[near] = winner
    out[far] = loser
    return out


def _pair_value(grid: np.ndarray, orient: str, axis: int, cell: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return cell, _mirror_coord(orient, axis, cell)


def _apply_corruptions(
    out: np.ndarray,
    orient: str,
    axis: int,
    winner: int,
    loser: int,
    blocks: list[dict],
) -> tuple[np.ndarray, dict]:
    inp = out.copy()
    winner_blocks = [idx for idx, block in enumerate(blocks) if block["color"] == winner]
    other_blocks = [idx for idx, block in enumerate(blocks) if block["color"] != winner]
    if len(winner_blocks) < 2 or len(other_blocks) < 2:
        raise ValueError("need at least two winner blocks and two other blocks")

    conflict_cells = []
    used_blocks = set()
    random.shuffle(winner_blocks)
    for keep_negative, block_idx in zip((True, False), winner_blocks[:2]):
        cell = random.choice(blocks[block_idx]["cells"])
        used_blocks.add(block_idx)
        a, b = _pair_value(inp, orient, axis, cell)
        if keep_negative:
            inp[a] = winner
            inp[b] = loser
        else:
            inp[a] = loser
            inp[b] = winner
        conflict_cells.append({"cell": cell, "keep_negative": keep_negative, "block": block_idx})

    remaining_blocks = [idx for idx in range(len(blocks)) if idx not in used_blocks]
    random.shuffle(remaining_blocks)
    missing_count = 2 if len(remaining_blocks) < 3 else random.choice((2, 3))
    missing_cells = []
    for idx, block_idx in enumerate(remaining_blocks[:missing_count]):
        cell = random.choice(blocks[block_idx]["cells"])
        remove_negative = (idx % 2 == 0)
        a, b = _pair_value(inp, orient, axis, cell)
        if remove_negative:
            inp[a] = BG
        else:
            inp[b] = BG
        missing_cells.append({"cell": cell, "remove_negative": remove_negative, "block": block_idx})

    return inp, {"conflicts": conflict_cells, "missing": missing_cells}


def _body_color_counts(grid: np.ndarray, sign_coords: set[tuple[int, int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) in sign_coords or int(grid[r, c]) in (BG, YELLOW):
                continue
            counts[int(grid[r, c])] = counts.get(int(grid[r, c]), 0) + 1
    return counts


def _validate_attempt(inp: np.ndarray, meta: dict) -> bool:
    out = meta["out"]
    orient = meta["orient"]
    axis = meta["axis"]
    winner = meta["winner"]
    loser = meta["loser"]
    sign_coords = set(meta["sign_coords"])

    if not np.array_equal(main(inp), out):
        return False

    candidates = _sign_candidates(inp, orient, axis, BG)
    if len(candidates) != 1 or set(candidates[0]) != sign_coords:
        return False

    counts = _body_color_counts(inp, sign_coords)
    if counts.get(loser, 0) <= counts.get(winner, 0):
        return False

    if len(meta["mods"]["conflicts"]) < 2 or len(meta["mods"]["missing"]) < 2:
        return False

    keep_sides = {entry["keep_negative"] for entry in meta["mods"]["conflicts"]}
    if keep_sides != {True, False}:
        return False

    conflict_map = {entry["cell"]: entry for entry in meta["mods"]["conflicts"]}
    missing_map = {entry["cell"]: entry for entry in meta["mods"]["missing"]}

    if orient == "h":
        neg_cells = [(r, c) for r in range(axis) for c in range(inp.shape[1])]
    else:
        neg_cells = [(r, c) for r in range(inp.shape[0]) for c in range(axis)]

    for cell in neg_cells:
        pair = _mirror_coord(orient, axis, cell)
        if cell in sign_coords or pair in sign_coords:
            continue
        in_a = int(inp[cell])
        in_b = int(inp[pair])
        out_a = int(out[cell])
        out_b = int(out[pair])
        if cell in conflict_map:
            keep_negative = conflict_map[cell]["keep_negative"]
            if out_a != winner or out_b != winner:
                return False
            if {in_a, in_b} != {winner, loser}:
                return False
            if keep_negative and in_a != winner:
                return False
            if (not keep_negative) and in_b != winner:
                return False
        elif cell in missing_map:
            remove_negative = missing_map[cell]["remove_negative"]
            if out_a != out_b or out_a == BG:
                return False
            if remove_negative:
                if not (in_a == BG and in_b == out_b):
                    return False
            else:
                if not (in_b == BG and in_a == out_a):
                    return False
        else:
            if in_a != out_a or in_b != out_b:
                return False

    return True


def _generate_attempt() -> tuple[np.ndarray, dict] | None:
    orient, winner_idx, sign_side = _next_branch()
    half_depth = random.randint(5, 9)
    cross = random.randint(13, 20)
    color_a, color_b = random.sample(PALETTE, 2)
    sign_colors = (color_a, color_b)
    winner = sign_colors[winner_idx]
    loser = sign_colors[1 - winner_idx]

    if orient == "h":
        shape = (2 * half_depth + 1, cross)
        axis = half_depth
        near_dist = random.randint(1, min(3, half_depth - 1))
        col = random.randint(1, cross - 2)
        if sign_side < 0:
            sign_coords = [(axis - near_dist, col), (axis - near_dist - 1, col)]
        else:
            sign_coords = [(axis + near_dist, col), (axis + near_dist + 1, col)]
        neg_h, neg_w = axis, cross
    else:
        shape = (cross, 2 * half_depth + 1)
        axis = half_depth
        near_dist = random.randint(1, min(3, half_depth - 1))
        row = random.randint(1, cross - 2)
        if sign_side < 0:
            sign_coords = [(row, axis - near_dist), (row, axis - near_dist - 1)]
        else:
            sign_coords = [(row, axis + near_dist), (row, axis + near_dist + 1)]
        neg_h, neg_w = cross, axis

    reserved = _reserve_negative_for_sign(orient, axis, sign_coords, neg_h, neg_w)
    block_count = random.choice((5, 6))
    block_colors = [winner, winner] + [loser] * (block_count - 2)
    random.shuffle(block_colors)
    blocks = _place_blocks(neg_h, neg_w, reserved, block_colors)
    if blocks is None:
        return None

    winner_area = sum(len(block["cells"]) for block in blocks if block["color"] == winner)
    loser_area = sum(len(block["cells"]) for block in blocks if block["color"] == loser)
    if loser_area <= winner_area:
        return None

    out = _paint_output(orient, axis, shape, sign_coords, sign_colors, winner, blocks)
    inp, mods = _apply_corruptions(out, orient, axis, winner, loser, blocks)
    meta = {
        "orient": orient,
        "axis": axis,
        "winner": winner,
        "loser": loser,
        "sign_coords": sign_coords,
        "out": out,
        "mods": mods,
    }
    if not _validate_attempt(inp, meta):
        return None
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _generate_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid axis-adjudication puzzle input")
