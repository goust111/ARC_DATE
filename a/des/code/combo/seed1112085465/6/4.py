import random
from collections import Counter

import numpy as np


BG_COLOR = 7       # orange
BAR_COLOR = 9      # maroon
ARROW_COLORS = (1, 2, 3, 4, 5, 6, 8)
GENERATE_ATTEMPTS = 300

MISSING_TO_CELLS = {
    "tl": ((0, 1), (1, 0), (1, 1)),
    "tr": ((0, 0), (1, 0), (1, 1)),
    "bl": ((0, 0), (0, 1), (1, 1)),
    "br": ((0, 0), (0, 1), (1, 0)),
}

MISSING_TO_SIDES = {
    "tl": ("up", "left"),
    "tr": ("up", "right"),
    "bl": ("down", "left"),
    "br": ("down", "right"),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_arrows(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _components_of_color(grid == color):
            assert len(comp) == 3, "each arrow must have exactly three cells"
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert r1 == r0 + 1 and c1 == c0 + 1, "arrow must fit a 2x2 bounding box"
            bbox = np.zeros((2, 2), dtype=bool)
            for r, c in comp:
                bbox[r - r0, c - c0] = True
            missing = [(rr, cc) for rr in range(2) for cc in range(2) if not bbox[rr, cc]]
            assert len(missing) == 1, "arrow must miss exactly one corner"
            miss_r, miss_c = missing[0]
            key = ("t" if miss_r == 0 else "b") + ("l" if miss_c == 0 else "r")
            arrows.append(
                {
                    "color": color,
                    "cells": comp,
                    "bbox": (r0, c0, r1, c1),
                    "missing": key,
                }
            )
    assert arrows, "expected at least one arrow sprite"
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize L-shaped arrow sprites as maroon count bars on the grid edges.

    Each arrow occupies three cells of a 2x2 block. The missing corner encodes a
    diagonal direction, so it contributes to the two matching cardinal edges:
    top-left -> up and left, top-right -> up and right, bottom-left -> down and
    left, bottom-right -> down and right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 4

    bg = _most_common_color(g)
    arrows = _detect_arrows(g, bg)

    counts = Counter({"up": 0, "down": 0, "left": 0, "right": 0})
    for arrow in arrows:
        for side in MISSING_TO_SIDES[str(arrow["missing"])]:
            counts[side] += 1

    assert counts["up"] <= w and counts["down"] <= w
    assert counts["left"] <= h and counts["right"] <= h

    out = g.copy()
    if counts["up"] > 0:
        out[0, : counts["up"]] = BAR_COLOR
    if counts["down"] > 0:
        out[h - 1, : counts["down"]] = BAR_COLOR
    if counts["left"] > 0:
        out[: counts["left"], 0] = BAR_COLOR
    if counts["right"] > 0:
        out[: counts["right"], w - 1] = BAR_COLOR
    return out


def _can_place(occupied: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + 3)
    c1 = min(w, left + 3)
    return not occupied[r0:r1, c0:c1].any()


def _paint_arrow(grid: np.ndarray, occupied: np.ndarray, top: int, left: int, missing: str, color: int) -> None:
    for dr, dc in MISSING_TO_CELLS[missing]:
        grid[top + dr, left + dc] = color
    occupied[top : top + 2, left : left + 2] = True


def _sample_missing_sequence(rng: random.Random, n_arrows: int) -> list[str]:
    keys = list(MISSING_TO_CELLS)
    while True:
        seq = [rng.choice(keys) for _ in range(n_arrows)]
        if len(set(seq)) >= min(2, n_arrows):
            return seq


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(8, 15)
    w = rng.randint(8, 15)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    max_by_size = min(len(ARROW_COLORS), h, w, ((h - 2) * (w - 2)) // 6)
    if max_by_size <= 0:
        return None
    min_arrows = 2 if max_by_size >= 2 else 1
    n_arrows = rng.randint(min_arrows, max_by_size)
    missing_seq = _sample_missing_sequence(rng, n_arrows)

    side_counts = Counter({"up": 0, "down": 0, "left": 0, "right": 0})
    for missing in missing_seq:
        for side in MISSING_TO_SIDES[missing]:
            side_counts[side] += 1
    if max(side_counts.values()) > max(h, w):
        return None
    if side_counts["up"] > w or side_counts["down"] > w or side_counts["left"] > h or side_counts["right"] > h:
        return None

    colors = rng.sample(list(ARROW_COLORS), n_arrows)
    placements = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    rng.shuffle(placements)

    placed = 0
    for color, missing in zip(colors, missing_seq):
        for idx, (top, left) in enumerate(placements):
            if _can_place(occupied, top, left):
                _paint_arrow(grid, occupied, top, left, missing, color)
                placements.pop(idx)
                placed += 1
                break
        else:
            return None

    if placed != n_arrows:
        return None

    # Reject overly one-sided cases so all edge-summary branches appear regularly.
    active_sides = sum(int(side_counts[side] > 0) for side in ("up", "down", "left", "right"))
    if active_sides < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
