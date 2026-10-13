from __future__ import annotations

from collections import Counter
import random

import numpy as np


SLOT_COLOR = 0
BACKGROUND_COLOR = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
GENERATION_RETRIES = 400

CORNER_INFO = {
    "tl": {
        "missing3": (0, 0),
        "missing2": (0, 0),
        "rows": slice(0, 2),
        "cols": slice(0, 2),
    },
    "tr": {
        "missing3": (0, 2),
        "missing2": (0, 1),
        "rows": slice(0, 2),
        "cols": lambda w: slice(w - 2, w),
    },
    "bl": {
        "missing3": (2, 0),
        "missing2": (1, 0),
        "rows": lambda h: slice(h - 2, h),
        "cols": slice(0, 2),
    },
    "br": {
        "missing3": (2, 2),
        "missing2": (1, 1),
        "rows": lambda h: slice(h - 2, h),
        "cols": lambda w: slice(w - 2, w),
    },
}


def _resolve_slice(spec, extent: int) -> slice:
    return spec(extent) if callable(spec) else spec


def _slot_box(name: str, h: int, w: int) -> tuple[int, int, int, int]:
    info = CORNER_INFO[name]
    rows = _resolve_slice(info["rows"], h)
    cols = _resolve_slice(info["cols"], w)
    return int(rows.start), int(cols.start), int(rows.stop - 1), int(cols.stop - 1)


def _infer_slot_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _infer_bg(grid: np.ndarray, slot_color: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    counts.pop(int(slot_color), None)
    assert counts, "grid must contain a background distinct from the slot color"
    return int(counts.most_common(1)[0][0])


def _candidate_blocks(grid: np.ndarray, bg: int, slot_color: int) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    seen: set[tuple[frozenset[tuple[int, int]], str, int]] = set()

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            for name, info in CORNER_INFO.items():
                mr, mc = info["missing3"]
                if int(patch[mr, mc]) != int(bg):
                    continue

                color = None
                cells: list[tuple[int, int]] = []
                ok = True
                for dr in range(3):
                    for dc in range(3):
                        if (dr, dc) == (mr, mc):
                            continue
                        val = int(patch[dr, dc])
                        if val in (int(bg), int(slot_color)):
                            ok = False
                            break
                        if color is None:
                            color = int(val)
                        elif int(val) != int(color):
                            ok = False
                            break
                        cells.append((int(r0 + dr), int(c0 + dc)))
                    if not ok:
                        break

                if not ok or color is None:
                    continue

                key = (frozenset(cells), str(name), int(color))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"corner": str(name), "color": int(color), "cells": frozenset(cells)})

    return candidates


def _solve_cover(candidates: list[dict], foreground: frozenset[tuple[int, int]]) -> list[dict]:
    if not foreground:
        return []

    cell_to_candidates: dict[tuple[int, int], list[int]] = {cell: [] for cell in foreground}
    for idx, cand in enumerate(candidates):
        if not cand["cells"] <= foreground:
            continue
        for cell in cand["cells"]:
            cell_to_candidates[cell].append(int(idx))

    for cell, options in cell_to_candidates.items():
        assert options, f"foreground cell {cell} is not covered by any 3x3 L candidate"

    def search(remaining: frozenset[tuple[int, int]], used_corners: frozenset[str]) -> list[dict] | None:
        if not remaining:
            return []

        anchor = min(
            remaining,
            key=lambda cell: sum(
                1
                for idx in cell_to_candidates[cell]
                if candidates[idx]["cells"] <= remaining and candidates[idx]["corner"] not in used_corners
            ),
        )
        options = [
            int(idx)
            for idx in cell_to_candidates[anchor]
            if candidates[idx]["cells"] <= remaining and candidates[idx]["corner"] not in used_corners
        ]
        if not options:
            return None

        for idx in options:
            cand = candidates[idx]
            tail = search(remaining - cand["cells"], used_corners | {cand["corner"]})
            if tail is not None:
                return [cand, *tail]
        return None

    solution = search(foreground, frozenset())
    assert solution is not None, "could not partition the foreground into distinct 3x3 L blocks"
    return solution


def _find_blocks(grid: np.ndarray) -> tuple[int, int, list[dict], frozenset[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    slot_color = _infer_slot_color(g)
    bg = _infer_bg(g, slot_color)
    foreground = frozenset(map(tuple, np.argwhere((g != int(bg)) & (g != int(slot_color)))))
    candidates = _candidate_blocks(g, bg, slot_color)
    blocks = _solve_cover(candidates, foreground)
    return int(bg), int(slot_color), blocks, foreground


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome 3x3 L block into the corner slot named by its missing corner.

    Input structure:
    - A uniform background fills the canvas, and each grid corner contains a 2x2 slot of one shared color.
    - Away from the corners are several non-overlapping 3x3 monochrome regions with exactly one background-colored corner cell missing.
    - Every such 3x3 region is otherwise filled by a single non-slot color, forming an L shape.

    Rule:
    - For each 3x3 L block, detect which corner of its 3x3 box is the missing background cell.
    - Erase the source block back to the background.
    - Send the block to the matching grid corner, equivalent to rotating it until the missing corner faces outward and shrinking it to a 2x2 slot.
    - In that 2x2 slot, keep the outward missing-corner cell in the slot color and paint the other three cells with the L block's color.

    Key insight:
    - The single missing corner simultaneously encodes both the destination corner and the one slot cell that must stay unfilled.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, slot_color, blocks, foreground = _find_blocks(g)

    out = g.copy()
    for r, c in foreground:
        out[int(r), int(c)] = int(bg)

    for name in CORNER_INFO:
        rows = _resolve_slice(CORNER_INFO[name]["rows"], h)
        cols = _resolve_slice(CORNER_INFO[name]["cols"], w)
        out[rows, cols] = int(slot_color)

    for block in blocks:
        info = CORNER_INFO[block["corner"]]
        rows = _resolve_slice(info["rows"], h)
        cols = _resolve_slice(info["cols"], w)
        patch = np.full((2, 2), int(slot_color), dtype=int)
        miss = info["missing2"]
        for dr in range(2):
            for dc in range(2):
                if (dr, dc) != miss:
                    patch[dr, dc] = int(block["color"])
        out[rows, cols] = patch

    return out


def _box_with_margin(top: int, left: int, size: int = 3, margin: int = 1) -> tuple[int, int, int, int]:
    return int(top - margin), int(left - margin), int(top + size - 1 + margin), int(left + size - 1 + margin)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (
        int(a[2]) < int(b[0])
        or int(b[2]) < int(a[0])
        or int(a[3]) < int(b[1])
        or int(b[3]) < int(a[1])
    )


def _can_place(top: int, left: int, h: int, w: int, taken: list[tuple[int, int, int, int]]) -> bool:
    box = _box_with_margin(top, left)
    return all(not _boxes_overlap(box, other) for other in taken)


def _place_l_patch(grid: np.ndarray, top: int, left: int, corner: str, color: int) -> None:
    patch = np.full((3, 3), int(color), dtype=int)
    mr, mc = CORNER_INFO[corner]["missing3"]
    patch[mr, mc] = int(BACKGROUND_COLOR)
    grid[top : top + 3, left : left + 3] = patch


def _random_positions(h: int, w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    taken = [_box_with_margin(*_slot_box(name, h, w)[:2], size=2, margin=1) for name in CORNER_INFO]
    choices = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    rng.shuffle(choices)

    positions: list[tuple[int, int]] = []
    for top, left in choices:
        if not _can_place(int(top), int(left), h, w, taken):
            continue
        positions.append((int(top), int(left)))
        taken.append(_box_with_margin(int(top), int(left)))
        if len(positions) == int(count):
            return positions
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 24)
    w = rng.randint(10, 24)
    count = rng.randint(1, 4)
    corners = rng.sample(tuple(CORNER_INFO.keys()), count)
    colors = rng.sample(PALETTE, count)
    positions = _random_positions(h, w, count, rng)
    if positions is None:
        return None

    grid = np.full((h, w), int(BACKGROUND_COLOR), dtype=int)
    for name in CORNER_INFO:
        r0, c0, r1, c1 = _slot_box(name, h, w)
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(SLOT_COLOR)

    rng.shuffle(positions)
    triples = list(zip(corners, colors, positions, strict=True))
    rng.shuffle(triples)
    for corner, color, (top, left) in triples:
        _place_l_patch(grid, int(top), int(left), str(corner), int(color))

    try:
        bg, slot_color, blocks, _foreground = _find_blocks(grid)
    except AssertionError:
        return None

    if int(bg) != int(BACKGROUND_COLOR) or int(slot_color) != int(SLOT_COLOR):
        return None
    if len(blocks) != int(count):
        return None

    found = {(str(block["corner"]), int(block["color"])) for block in blocks}
    expected = {(str(corner), int(color)) for corner, color, _pos in triples}
    if found != expected:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
