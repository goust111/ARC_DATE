from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PERP = {
    (-1, 0): ((0, -1), (0, 1)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (0, 1): ((-1, 0), (1, 0)),
}

GRID_MIN = 10
GRID_MAX = 24
MAX_BODY_LEN = 7
PLACEMENT_ATTEMPTS = 400
GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_by_color(grid: np.ndarray) -> list[tuple[int, set[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, set[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                coords.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(g.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, coords))
    return comps


def _decode_arrow(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], int]:
    degrees = {
        cell: sum((nbr in coords) for nbr in _neighbors4(*cell))
        for cell in coords
    }
    junctions = [cell for cell, degree in degrees.items() if degree == 3]
    leaves = [cell for cell, degree in degrees.items() if degree == 1]
    middles = [cell for cell, degree in degrees.items() if degree == 2]

    assert len(junctions) == 1, "each arrow must have one tip junction"
    assert len(leaves) == 3, "each arrow must have three leaf pixels"
    assert len(middles) + len(leaves) + len(junctions) == len(coords), "arrow degrees must be 1/2/3 only"

    tip = junctions[0]
    present = {
        (nr - tip[0], nc - tip[1])
        for nr, nc in _neighbors4(*tip)
        if (nr, nc) in coords
    }
    direction = next((dr, dc) for dr, dc in DIRS if (dr, dc) not in present)
    body_step = (-direction[0], -direction[1])

    body = [tip]
    cur = tip
    while True:
        nxt = (cur[0] + body_step[0], cur[1] + body_step[1])
        if nxt not in coords:
            break
        body.append(nxt)
        cur = nxt

    side_dirs = PERP[direction]
    expected = set(body)
    for dr, dc in side_dirs:
        expected.add((tip[0] + dr, tip[1] + dc))

    assert len(body) >= 2, "arrow body must have length at least 2"
    assert expected == coords, "component must be a straight arrow with a 3-pixel tip"
    return tip, direction, len(body)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each monochrome arrow by a same-color ray whose length matches the arrow body.

    Input structure:
    - The background is black.
    - Each foreground object is a monochrome 4-connected arrow component.
    - Every arrow consists of a straight horizontal or vertical body plus a
      3-pixel tip whose two side pixels reveal the pointing direction.
    - Different arrows are separated from one another on the canvas.

    Rule:
    - Keep every original arrow unchanged.
    - For each arrow, identify the tip junction and count the body length N,
      excluding only the two side pixels of the tip.
    - Starting one step beyond the tip, draw up to N pixels in the arrow's
      pointing direction using the arrow's color.
    - If the grid edge is reached before N pixels are drawn, truncate the ray.

    Key insight:
    - The missing side around the tip junction reveals the arrow direction,
      while the body itself encodes the ray length to paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for color, coords in _components_by_color(g):
        tip, direction, body_len = _decode_arrow(coords)
        rr, cc = tip
        dr, dc = direction
        for _ in range(body_len):
            rr += dr
            cc += dc
            if not _in_bounds(g.shape, rr, cc):
                break
            out[rr, cc] = int(color)
    return out


def _steps_to_edge(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]) -> int:
    rr, cc = start
    dr, dc = step
    count = 0
    while True:
        rr += dr
        cc += dc
        if not _in_bounds(shape, rr, cc):
            return count
        count += 1


def _arrow_cells(
    tip: tuple[int, int],
    direction: tuple[int, int],
    body_len: int,
) -> set[tuple[int, int]]:
    body_step = (-direction[0], -direction[1])
    cells = {
        (tip[0] + body_step[0] * k, tip[1] + body_step[1] * k)
        for k in range(body_len)
    }
    for dr, dc in PERP[direction]:
        cells.add((tip[0] + dr, tip[1] + dc))
    return cells


def _ray_cells(
    shape: tuple[int, int],
    tip: tuple[int, int],
    direction: tuple[int, int],
    body_len: int,
) -> list[tuple[int, int]]:
    rr, cc = tip
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for _ in range(body_len):
        rr += dr
        cc += dc
        if not _in_bounds(shape, rr, cc):
            break
        cells.append((rr, cc))
    return cells


def _reserve_region(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    reserved = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    reserved.add((nr, nc))
    return reserved


def _enumerate_candidates(shape: tuple[int, int]) -> list[dict[str, object]]:
    h, w = shape
    candidates: list[dict[str, object]] = []
    for direction in DIRS:
        for r in range(h):
            for c in range(w):
                tip = (r, c)
                behind = _steps_to_edge(shape, tip, (-direction[0], -direction[1]))
                ahead = _steps_to_edge(shape, tip, direction)
                max_len = min(MAX_BODY_LEN, behind + 1)
                for body_len in range(2, max_len + 1):
                    cells = _arrow_cells(tip, direction, body_len)
                    if any(not _in_bounds(shape, rr, cc) for rr, cc in cells):
                        continue
                    if ahead >= body_len:
                        mode = "full"
                    elif 1 <= ahead < body_len:
                        mode = "truncated"
                    else:
                        continue
                    ray = _ray_cells(shape, tip, direction, body_len)
                    footprint = set(cells) | set(ray)
                    candidates.append(
                        {
                            "tip": tip,
                            "direction": direction,
                            "body_len": body_len,
                            "cells": cells,
                            "ray": ray,
                            "mode": mode,
                            "reserve": _reserve_region(shape, footprint),
                        }
                    )
    return candidates


def _desired_modes(count: int) -> list[str]:
    modes = ["full" if i % 2 == 0 else "truncated" for i in range(count)]
    if count % 2 == 1:
        modes[-1] = random.choice(["full", "truncated"])
    random.shuffle(modes)
    return modes


def _desired_dirs(count: int) -> list[tuple[int, int]]:
    base = list(DIRS)
    random.shuffle(base)
    dirs = [base[i % len(base)] for i in range(count)]
    random.shuffle(dirs)
    return dirs


def _pick_candidate(
    candidates: list[dict[str, object]],
    occupied: set[tuple[int, int]],
    mode: str,
    direction: tuple[int, int],
) -> dict[str, object] | None:
    ranked_filters = (
        lambda cand: cand["mode"] == mode and cand["direction"] == direction,
        lambda cand: cand["mode"] == mode,
        lambda cand: cand["direction"] == direction,
        lambda cand: True,
    )

    for keep in ranked_filters:
        valid = [
            cand for cand in candidates
            if keep(cand) and not (cand["reserve"] & occupied)
        ]
        if valid:
            return random.choice(valid)
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)
    candidates = _enumerate_candidates(shape)
    if not candidates:
        return None

    full_count = sum(cand["mode"] == "full" for cand in candidates)
    trunc_count = sum(cand["mode"] == "truncated" for cand in candidates)
    if full_count == 0 or trunc_count == 0:
        return None

    max_arrows = min(5, len(COLORS), max(1, (h * w) // 70))
    min_arrows = 1 if max_arrows == 1 else 2
    arrow_count = random.randint(min_arrows, max_arrows)

    modes = _desired_modes(arrow_count)
    dirs = _desired_dirs(arrow_count)
    colors = random.sample(COLORS, arrow_count)
    occupied: set[tuple[int, int]] = set()
    placements: list[tuple[int, dict[str, object]]] = []

    for color, mode, direction in zip(colors, modes, dirs):
        candidate = _pick_candidate(candidates, occupied, mode, direction)
        if candidate is None:
            return None
        placements.append((color, candidate))
        occupied |= candidate["reserve"]

    grid = np.full(shape, BG, dtype=int)
    placed_modes = []
    for color, candidate in placements:
        placed_modes.append(str(candidate["mode"]))
        for r, c in candidate["cells"]:
            grid[r, c] = color

    if len(placements) >= 2 and len(set(placed_modes)) < 2:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow scene")
