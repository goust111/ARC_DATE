from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


GREY = Color.GREY
BG = Color.BLACK
SEED_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _corners(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _is_corner(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    return cell in set(_corners(shape))


def _trace_billiard(shape: tuple[int, int], start: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = map(int, shape)
    r, c = map(int, start)
    dr = 1 if r == 0 else -1
    dc = 1 if c == 0 else -1
    path = [(r, c, 0)]
    bounce_count = 0

    step_cap = 4 * max(1, (h - 1) * (w - 1) + 1)
    for _ in range(step_cap):
        row_flip = not (0 <= r + dr < h)
        col_flip = not (0 <= c + dc < w)
        if row_flip:
            dr *= -1
        if col_flip:
            dc *= -1
        bounce_count += int(row_flip) + int(col_flip)
        r += dr
        c += dc
        path.append((r, c, bounce_count))
        if (r in (0, h - 1)) and (c in (0, w - 1)):
            return path
    raise AssertionError("billiard path failed to terminate at a corner")


def _path_cells(path: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(r, c) for r, c, _b in path}


def _interior_points(path: list[tuple[int, int, int]], shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = map(int, shape)
    return [(r, c, b) for r, c, b in path[1:-1] if 0 < r < h - 1 and 0 < c < w - 1]


def _seed_infos(shape: tuple[int, int]) -> dict[tuple[int, int], dict]:
    infos: dict[tuple[int, int], dict] = {}
    for corner in _corners(shape):
        path = _trace_billiard(shape, corner)
        interior = _interior_points(path, shape)
        infos[corner] = {
            "path": path,
            "cells": _path_cells(path),
            "interior": interior,
            "after_bounce": [(r, c) for r, c, b in interior if b > 0],
            "pre_bounce": [(r, c) for r, c, b in interior if b == 0],
        }
    return infos


def _build_shape_bank() -> list[dict]:
    bank = []
    for h in range(6, 19):
        for w in range(6, 19):
            infos = _seed_infos((h, w))
            valid_pairs = []
            direct_pairs = []
            bouncy_pairs = []
            corner_list = _corners((h, w))
            for i, a in enumerate(corner_list):
                for b in corner_list[i + 1 :]:
                    if infos[a]["cells"].isdisjoint(infos[b]["cells"]):
                        valid_pairs.append((a, b))
                        if not infos[a]["after_bounce"] and not infos[b]["after_bounce"]:
                            direct_pairs.append((a, b))
                        if infos[a]["after_bounce"] or infos[b]["after_bounce"]:
                            bouncy_pairs.append((a, b))

            direct_singles = [corner for corner, info in infos.items() if not info["after_bounce"] and len(info["interior"]) >= 2]
            bouncy_singles = [corner for corner, info in infos.items() if info["after_bounce"] and len(info["interior"]) >= 3]
            all_singles = [corner for corner, info in infos.items() if len(info["interior"]) >= 2]
            if all_singles:
                bank.append(
                    {
                        "shape": (h, w),
                        "infos": infos,
                        "all_singles": all_singles,
                        "direct_singles": direct_singles,
                        "bouncy_singles": bouncy_singles,
                        "valid_pairs": valid_pairs,
                        "direct_pairs": direct_pairs,
                        "bouncy_pairs": bouncy_pairs,
                    }
                )
    return bank


SHAPE_BANK = _build_shape_bank()


def _neighbor4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in CARDINALS]


def _pick_obstacles(
    shape: tuple[int, int],
    chosen: list[tuple[tuple[int, int], int]],
    infos: dict[tuple[int, int], dict],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], set[tuple[int, int]]]]:
    h, w = map(int, shape)
    used: set[tuple[int, int]] = set()
    hits_by_seed: dict[tuple[int, int], set[tuple[int, int]]] = {corner: set() for corner, _color in chosen}

    def can_place(cell: tuple[int, int]) -> bool:
        r, c = cell
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        if cell in used:
            return False
        return all(nbr not in used for nbr in _neighbor4(cell))

    def place_from(candidates: list[tuple[int, int]], need: int) -> list[tuple[int, int]]:
        random.shuffle(candidates)
        placed = []
        for cell in candidates:
            if len(placed) >= need:
                break
            if can_place(cell):
                used.add(cell)
                placed.append(cell)
        return placed

    for corner, _color in chosen:
        info = infos[corner]
        mandatory_after = 1 if info["after_bounce"] else 0
        placed_after = place_from(info["after_bounce"][:], mandatory_after)
        hits_by_seed[corner].update(placed_after)

        remaining = [cell[:2] for cell in info["interior"] if (cell[0], cell[1]) not in hits_by_seed[corner]]
        need_extra = random.randint(0, min(3, max(0, len(remaining) - 1)))
        extra = place_from(remaining, 1 + need_extra if not hits_by_seed[corner] else need_extra)
        hits_by_seed[corner].update(extra)

        if not hits_by_seed[corner]:
            raise AssertionError("each seed must recolor at least one obstacle")

    path_union = set()
    for corner, _color in chosen:
        path_union |= {cell[:2] for cell in infos[corner]["interior"]}

    near_path = set()
    for cell in path_union:
        for nbr in _neighbor4(cell):
            r, c = nbr
            if 0 < r < h - 1 and 0 < c < w - 1 and nbr not in path_union:
                near_path.add(nbr)

    far_pool = {
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in path_union
    }

    lure_target = random.randint(1, min(6, max(1, len(far_pool) // 5)))
    lure_count = 0
    for pool in (list(near_path), list(far_pool - near_path)):
        random.shuffle(pool)
        for cell in pool:
            if lure_count >= lure_target:
                break
            if can_place(cell):
                used.add(cell)
                lure_count += 1
        if lure_count >= lure_target:
            break

    if lure_count == 0:
        raise AssertionError("need at least one untouched grey lure")
    return used, hits_by_seed


def _render_instance(
    shape: tuple[int, int],
    chosen: list[tuple[tuple[int, int], int]],
    obstacles: set[tuple[int, int]],
    hits_by_seed: dict[tuple[int, int], set[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full(shape, BG, dtype=int)
    for corner, color in chosen:
        grid[corner] = color
    for r, c in obstacles:
        grid[r, c] = GREY

    out = grid.copy()
    for corner, color in chosen:
        for r, c in hits_by_seed[corner]:
            out[r, c] = color
    return grid, out


def _sample_layout() -> tuple[tuple[int, int], list[tuple[tuple[int, int], int]], dict]:
    want_two = random.random() < 0.58
    want_direct = random.random() < 0.22

    candidate_entries = []
    for entry in SHAPE_BANK:
        if want_two:
            if want_direct:
                pairs = entry["direct_pairs"]
            else:
                pairs = entry["bouncy_pairs"] or entry["valid_pairs"]
            if pairs:
                candidate_entries.append((entry, pairs, 2))
        else:
            if want_direct:
                singles = entry["direct_singles"]
            else:
                singles = entry["bouncy_singles"] or entry["all_singles"]
            if singles:
                candidate_entries.append((entry, singles, 1))
    if not candidate_entries:
        raise AssertionError("shape bank unexpectedly empty")

    entry, options, seed_count = random.choice(candidate_entries)
    colors = random.sample(SEED_COLORS, seed_count)
    infos = entry["infos"]

    if seed_count == 1:
        corner = random.choice(options)
        chosen = [(corner, colors[0])]
    else:
        pair = random.choice(options)
        chosen = [(pair[0], colors[0]), (pair[1], colors[1])]
    return entry["shape"], chosen, infos


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = map(int, g.shape)
    seeds = []
    for corner in _corners((h, w)):
        color = int(g[corner])
        if color not in (BG, GREY):
            seeds.append((corner, color))

    for corner, color in seeds:
        for r, c, _b in _trace_billiard((h, w), corner)[1:-1]:
            if out[r, c] == GREY:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor gray obstacle pixels that lie on each corner seed's invisible diagonal billiard path.

    Input structure
    - The grid is a black canvas with one or two colored seed pixels located at corners.
    - Remaining non-black foreground cells are gray singleton obstacle pixels scattered away from the corners.
    - The chosen seeds' diagonal reflection paths do not overlap.

    Rule
    - From each seed corner, launch a 45-degree path toward the interior and reflect it off the grid boundary like a mirror until it reaches a corner.
    - Do not draw the path itself; recolor every gray obstacle cell visited by that path to the seed's color.
    - Leave unvisited gray obstacles gray and preserve the original seed pixels.

    Key insight
    - The only visible clues are the corner seeds and gray dots, so the solver must mentally trace the hidden diagonal bounce path to know which obstacles change color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    bg = get_bg(g)
    assert bg == BG, "expected a black background"

    h, w = map(int, g.shape)
    corner_set = set(_corners((h, w)))
    seeds: list[tuple[tuple[int, int], int]] = []
    for corner in corner_set:
        color = int(g[corner])
        if color not in (bg, GREY):
            seeds.append((corner, color))
    assert 1 <= len(seeds) <= 2, "expected one or two colored corner seeds"

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if (r, c) in corner_set:
                if color == GREY:
                    raise AssertionError("corner cells cannot be gray obstacles")
            elif color not in (bg, GREY):
                raise AssertionError("non-corner colored pixels are not allowed")

    grey_components = find_components(
        g,
        bg=bg,
        colors=[GREY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert all(int(comp["area"]) == 1 for comp in grey_components), "gray obstacles must be singleton pixels"

    traced = []
    for corner, color in seeds:
        path = _trace_billiard((h, w), corner)
        traced.append((corner, color, path))
    for i, (_corner_a, _color_a, path_a) in enumerate(traced):
        cells_a = _path_cells(path_a)
        for _corner_b, _color_b, path_b in traced[i + 1 :]:
            assert cells_a.isdisjoint(_path_cells(path_b)), "seed trajectories must not overlap"

    out = g.copy()
    for _corner, color, path in traced:
        for r, c, _b in path[1:-1]:
            if out[r, c] == GREY:
                out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        shape, chosen, infos = _sample_layout()
        try:
            obstacles, hits_by_seed = _pick_obstacles(shape, chosen, infos)
        except AssertionError:
            continue
        grid, expected = _render_instance(shape, chosen, obstacles, hits_by_seed)

        if np.array_equal(grid, expected):
            continue
        if not np.array_equal(_oracle_output(grid), expected):
            continue
        try:
            solved = main(grid)
        except Exception:
            continue
        if not np.array_equal(solved, expected):
            continue

        seed_count = sum(int(grid[corner] not in (BG, GREY)) for corner in _corners(shape))
        if seed_count != len(chosen):
            continue

        recolored = {
            (r, c)
            for r in range(shape[0])
            for c in range(shape[1])
            if grid[r, c] == GREY and solved[r, c] != GREY
        }
        if not recolored:
            continue
        if any(not hits_by_seed[corner] for corner, _color in chosen):
            continue

        return grid
    raise RuntimeError("failed to generate a valid invisible-billiard ARC input")
