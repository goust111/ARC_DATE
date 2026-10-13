from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
PLATFORM = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_sprite_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    fg = (grid != BG) & (grid != PLATFORM)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                assert int(grid[cr, cc]) == color, "each sprite must be monochromatic"
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": _component_mask(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _parse_platforms(grid: np.ndarray) -> list[dict]:
    platforms: list[dict] = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] == PLATFORM)
        if cols.size == 0:
            continue
        c0 = int(cols.min())
        c1 = int(cols.max())
        holes: list[dict] = []
        c = c0
        while c <= c1:
            if grid[r, c] != BG:
                c += 1
                continue
            start = c
            while c <= c1 and grid[r, c] == BG:
                c += 1
            end = c - 1
            cells = tuple((r, cc) for cc in range(start, end + 1))
            holes.append(
                {
                    "row": r,
                    "c0": start,
                    "c1": end,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "length": end - start + 1,
                }
            )
        platforms.append({"row": r, "c0": c0, "c1": c1, "holes": holes})
    return platforms


def _nearest_platform_distance(sprite: dict, platforms: list[dict]) -> int:
    _r0, c0, r1, c1 = sprite["bbox"]
    cols = range(c0, c1 + 1)
    best: int | None = None
    for platform in platforms:
        if platform["row"] <= r1:
            continue
        if any(platform["c0"] <= c <= platform["c1"] for c in cols):
            dist = int(platform["row"] - r1 - 1)
            if best is None or dist < best:
                best = dist
    if best is None:
        return 10**6
    return best


def _shape_signature(mask: np.ndarray) -> tuple[int, tuple[tuple[int, ...], ...]]:
    return int(mask.sum()), tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _simulate(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    platforms = _parse_platforms(g)
    assert platforms, "expected at least one platform"
    sprites = _find_sprite_components(g)
    assert sprites, "expected at least one sprite"

    out = g.copy()
    out[(out != BG) & (out != PLATFORM)] = BG
    occupied = out != BG

    holes: list[dict] = []
    for platform in platforms:
        for hole in platform["holes"]:
            holes.append({**hole, "filled": False})

    order = sorted(
        sprites,
        key=lambda comp: (
            _nearest_platform_distance(comp, platforms),
            int(comp["bbox"][1]),
            int(comp["bbox"][0]),
        ),
    )

    trace: dict[int, dict] = {}
    for idx, sprite in enumerate(order):
        coords = sprite["coords"].copy()
        color = int(sprite["color"])
        event = "bottom"
        embedded_hole: frozenset[tuple[int, int]] | None = None

        while True:
            if int(coords[:, 0].max()) == h - 1:
                break

            next_coords = coords + np.array([1, 0], dtype=int)

            matched_hole = None
            next_cells = frozenset((int(r), int(c)) for r, c in next_coords)
            if len({int(r) for r, _c in next_cells}) == 1:
                for hole in holes:
                    if hole["filled"]:
                        continue
                    if next_cells == hole["cell_set"]:
                        matched_hole = hole
                        break

            if matched_hole is not None:
                out[next_coords[:, 0], next_coords[:, 1]] = color
                occupied[next_coords[:, 0], next_coords[:, 1]] = True
                matched_hole["filled"] = True
                event = "embed"
                embedded_hole = matched_hole["cell_set"]
                coords = next_coords
                break

            if np.any(next_coords[:, 0] >= h) or np.any(occupied[next_coords[:, 0], next_coords[:, 1]]):
                if np.any(occupied[next_coords[:, 0], next_coords[:, 1]]):
                    event = "rest"
                break

            coords = next_coords

        out[coords[:, 0], coords[:, 1]] = color
        occupied[coords[:, 0], coords[:, 1]] = True
        trace[color] = {
            "event": event,
            "order_index": idx,
            "final_coords": tuple(sorted((int(r), int(c)) for r, c in coords)),
            "hole": embedded_hole,
        }

    if return_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let colored sprites fall downward so exact horizontal-line matches embed into platform holes while other sprites strand or keep falling.

    Input structure:
    - A black background contains several gray one-row horizontal platforms at different heights.
    - Each platform spans one continuous left-right interval, but some cells inside that interval are black holes.
    - Several monochrome 4-connected colored sprites sit above the platforms, with distinct shapes and colors.

    Rule:
    - Clear the sprites from their starting cells, keep the gray platforms fixed, and process the sprites in order of increasing distance from each sprite's bottom edge to the nearest lower platform interval; break ties by leftmost column.
    - During its turn, a sprite drops rigidly one row at a time.
    - If the next one-row move would place the whole sprite exactly into one still-empty platform hole, embed it there in the sprite's own color; that hole then becomes solid terrain for later sprites.
    - Otherwise, if the next move would hit gray platform cells, an already embedded hole-fill, or a previously stranded sprite, the sprite strands in its current position.
    - If no obstacle stops it first, the sprite settles on the grid bottom.

    Key insight:
    - A hole is both a pass-through gap and a possible exact-fit receptacle, so an earlier embedding can block a later sprite that would otherwise fall through to a lower level.
    """
    return _simulate(grid)


def _shape_hline(length: int) -> list[tuple[int, int]]:
    return [(0, c) for c in range(length)]


def _shape_vline(length: int) -> list[tuple[int, int]]:
    return [(r, 0) for r in range(length)]


NONMATCH_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
)


def _platform_cells(width: int, holes: list[tuple[int, int]]) -> list[int]:
    mask = [PLATFORM] * width
    for offset, length in holes:
        for c in range(offset, offset + length):
            mask[c] = BG
    return mask


def _draw_platform(grid: np.ndarray, row: int, c0: int, width: int, holes: list[tuple[int, int]]) -> dict:
    cells = _platform_cells(width, holes)
    for offset, value in enumerate(cells):
        grid[row, c0 + offset] = value
    return {
        "row": row,
        "c0": c0,
        "c1": c0 + width - 1,
        "width": width,
        "holes": [(c0 + off, c0 + off + length - 1) for off, length in holes],
    }


def _shape_dims(shape: Iterable[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _sprite_cells(shape: Iterable[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _can_place_sprite(
    shape: Iterable[tuple[int, int]],
    top: int,
    left: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> bool:
    cells = _sprite_cells(shape, top, left)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _paste_sprite(
    grid: np.ndarray,
    shape: Iterable[tuple[int, int]],
    top: int,
    left: int,
    color: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = _sprite_cells(shape, top, left)
    for r, c in cells:
        grid[r, c] = color
        occupied.add((r, c))
    return cells


def _pick_top_holes() -> tuple[int, int, bool]:
    key_len = random.choice((2, 3))
    other_len = 5 - key_len
    key_is_left = bool(random.getrandbits(1))
    return key_len, other_len, key_is_left


def _len1_holes(width: int, count: int, forbidden: set[int]) -> list[tuple[int, int]]:
    candidates = [c for c in range(1, width - 1) if c not in forbidden]
    for target in (count, 1):
        random.shuffle(candidates)
        holes: list[int] = []
        for c in candidates:
            if all(abs(c - prev) >= 2 for prev in holes):
                holes.append(c)
            if len(holes) == target:
                break
        if len(holes) == target:
            return sorted((c, 1) for c in holes)
    raise ValueError("failed to place len-1 holes")


def _input_hole_signatures(grid: np.ndarray) -> list[tuple[int, tuple[tuple[int, ...], ...]]]:
    sigs = []
    for platform in _parse_platforms(grid):
        for hole in platform["holes"]:
            width = hole["c1"] - hole["c0"] + 1
            mask = np.ones((1, width), dtype=bool)
            sigs.append(_shape_signature(mask))
    return sigs


def _validate_unique_hole_matches(grid: np.ndarray) -> bool:
    hole_sigs = _input_hole_signatures(grid)
    sprites = _find_sprite_components(grid)
    for sprite in sprites:
        sig = _shape_signature(sprite["mask"])
        if sum(1 for hole_sig in hole_sigs if hole_sig == sig) > 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 20)
    w = random.randint(16, 19)
    grid = np.zeros((h, w), dtype=int)

    top_row = random.randint(5, 7)
    gap1 = random.randint(3, 4)
    gap2 = random.randint(3, 4)
    row2 = top_row + gap1
    row3 = row2 + gap2
    if row3 >= h - 2:
        return None
    use_row4 = row3 + 3 < h and random.random() < 0.65
    row4 = row3 + random.randint(3, 4) if use_row4 else None
    if row4 is not None and row4 >= h:
        return None

    top_c0 = 1
    top_width = 8
    key_len, other_len, key_is_left = _pick_top_holes()
    if key_is_left:
        top_holes = [(1, key_len), (key_len + 2, other_len)]
        key_hole = (top_c0 + 1, top_c0 + key_len)
        other_hole = (top_c0 + key_len + 2, top_c0 + key_len + other_len + 1)
    else:
        top_holes = [(1, other_len), (other_len + 2, key_len)]
        other_hole = (top_c0 + 1, top_c0 + other_len)
        key_hole = (top_c0 + other_len + 2, top_c0 + other_len + key_len + 1)
    top_platform = _draw_platform(grid, top_row, top_c0, top_width, top_holes)

    second_width = random.choice((7, 8))
    second_c0 = max(0, min(key_hole[0] - 2, 2))
    second_count = random.choice((1, 2))
    key_col_for_b = random.randint(key_hole[0], key_hole[1])
    second_holes = _len1_holes(second_width, second_count, {key_col_for_b - second_c0})
    second_platform = _draw_platform(grid, row2, second_c0, second_width, second_holes)

    third_width = 6
    third_c0 = w - third_width
    if third_c0 <= top_platform["c1"] + 1:
        return None
    third_count = random.choice((1, 2))
    third_holes = _len1_holes(third_width, third_count, set())
    third_platform = _draw_platform(grid, row3, third_c0, third_width, third_holes)
    third_hole_col = third_c0 + third_holes[0][0]

    fourth_platform = None
    if row4 is not None:
        fourth_width = random.choice((6, 7))
        fourth_c0 = min(max(third_c0 - random.choice((0, 1)), 0), w - fourth_width)
        if not (fourth_c0 <= third_hole_col <= fourth_c0 + fourth_width - 1):
            return None
        fourth_forbidden = {third_hole_col - fourth_c0}
        fourth_holes = _len1_holes(fourth_width, random.choice((1, 2)), fourth_forbidden)
        fourth_platform = _draw_platform(grid, row4, fourth_c0, fourth_width, fourth_holes)

    sprite_count = 5 + int(random.random() < 0.45)
    colors = random.sample(SPRITE_COLORS, sprite_count)
    a_color, b_color, c_color, d_color, e_color = colors[:5]
    f_color = colors[5] if sprite_count == 6 else None

    occupied: set[tuple[int, int]] = set()

    a_shape = _shape_hline(key_len)
    a_rows = [top_row - 3, top_row - 2]
    random.shuffle(a_rows)
    a_top = next((r for r in a_rows if _can_place_sprite(a_shape, r, key_hole[0], h, w, occupied)), None)
    if a_top is None:
        return None
    _paste_sprite(grid, a_shape, a_top, key_hole[0], a_color, occupied)

    b_shape = _shape_vline(2)
    b_rows = [0, 1, 2]
    random.shuffle(b_rows)
    b_top = next((r for r in b_rows if _can_place_sprite(b_shape, r, key_col_for_b, h, w, occupied)), None)
    if b_top is None or b_top >= a_top - 1:
        return None
    _paste_sprite(grid, b_shape, b_top, key_col_for_b, b_color, occupied)

    c_shape = _shape_hline(other_len)
    c_rows = [top_row - 4, top_row - 3, top_row - 2]
    random.shuffle(c_rows)
    c_top = next((r for r in c_rows if r >= 0 and _can_place_sprite(c_shape, r, other_hole[0], h, w, occupied)), None)
    if c_top is None:
        return None
    _paste_sprite(grid, c_shape, c_top, other_hole[0], c_color, occupied)

    top_solid_cols = [c for c in range(top_platform["c0"], top_platform["c1"] + 1) if not (key_hole[0] <= c <= key_hole[1]) and not (other_hole[0] <= c <= other_hole[1])]
    d_col = random.choice(top_solid_cols)
    d_shape = _shape_vline(random.choice((2, 3)))
    d_rows = [0, 1, 2, top_row - 5]
    random.shuffle(d_rows)
    d_top = next((r for r in d_rows if r >= 0 and _can_place_sprite(d_shape, r, d_col, h, w, occupied)), None)
    if d_top is None:
        return None
    _paste_sprite(grid, d_shape, d_top, d_col, d_color, occupied)

    e_shape = random.choice((_shape_vline(2), _shape_vline(3)))
    e_rows = [0, 1, 2]
    random.shuffle(e_rows)
    e_top = next((r for r in e_rows if _can_place_sprite(e_shape, r, third_hole_col, h, w, occupied)), None)
    if e_top is None:
        return None
    _paste_sprite(grid, e_shape, e_top, third_hole_col, e_color, occupied)

    if f_color is not None:
        f_shape = list(random.choice(NONMATCH_SHAPES))
        fh, fw = _shape_dims(f_shape)
        third_solid_offsets = [
            c - third_c0
            for c in range(third_c0, third_c0 + third_width)
            if c != third_hole_col and all(not (start <= c <= end) for start, end in third_platform["holes"])
        ]
        if not third_solid_offsets:
            return None
        f_left_candidates = [third_c0 + max(0, off - fw // 2) for off in third_solid_offsets]
        random.shuffle(f_left_candidates)
        f_rows = [0, 1, 2]
        placed = False
        for left in f_left_candidates:
            for top in f_rows:
                if _can_place_sprite(f_shape, top, left, h, w, occupied):
                    _paste_sprite(grid, f_shape, top, left, f_color, occupied)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    out, trace = _simulate(grid, return_trace=True)

    if trace[a_color]["event"] != "embed":
        return None
    if trace[c_color]["event"] != "embed":
        return None
    if trace[d_color]["event"] != "rest":
        return None
    if trace[e_color]["event"] not in {"rest", "bottom"}:
        return None
    if sum(1 for item in trace.values() if item["event"] == "embed") < 2:
        return None
    if sum(1 for item in trace.values() if item["event"] == "rest") < 1:
        return None

    grid_without_a = grid.copy()
    grid_without_a[grid_without_a == a_color] = BG
    _, trace_without_a = _simulate(grid_without_a, return_trace=True)
    if b_color not in trace_without_a:
        return None
    b_rows_with = [r for r, _c in trace[b_color]["final_coords"]]
    b_rows_without = [r for r, _c in trace_without_a[b_color]["final_coords"]]
    if min(b_rows_without) <= top_row or trace[b_color]["final_coords"] == trace_without_a[b_color]["final_coords"]:
        return None

    components = _find_sprite_components(grid)
    if not (4 <= len(components) <= 6):
        return None
    if not all(2 <= comp["area"] <= 4 for comp in components):
        return None
    if not _validate_unique_hole_matches(grid):
        return None
    if any(_nearest_platform_distance(comp, _parse_platforms(grid)) >= 10**6 for comp in components):
        return None

    assert grid.shape == out.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-hole ARC input")
