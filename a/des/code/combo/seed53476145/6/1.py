from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MISSING_CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
MOVE_BY_MISSING = {
    (0, 0): (1, 1),
    (0, 2): (1, -1),
    (2, 0): (-1, 1),
    (2, 2): (-1, -1),
}

# --------------- solver ---------------


def _filled_offsets(missing: tuple[int, int]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(3)
        for c in range(3)
        if (r, c) != tuple(missing)
    ]


def _find_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append({"color": color, "cells": cells})
    return comps


def _parse_sprite(g: np.ndarray, comp: dict) -> dict:
    cells = comp["cells"]
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "each sprite bbox must be 3x3"
    assert len(cells) == 8, "each sprite must contain exactly 8 foreground cells"

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape == (3, 3)
    missing = [corner for corner in MISSING_CORNERS if int(patch[corner]) == BG]
    assert len(missing) == 1, "each sprite must miss exactly one corner"
    missing_corner = missing[0]

    color = int(comp["color"])
    for rr, cc in _filled_offsets(missing_corner):
        assert int(patch[rr, cc]) == color, "sprite must be monochromatic except its gap"

    return {"top": r0, "left": c0, "color": color, "missing": missing_corner}


def main(grid: np.ndarray) -> np.ndarray:
    """Infer diagonal motion from each 3x3 sprite's missing corner and paint its grey trail."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = [_parse_sprite(g, comp) for comp in _find_components(g)]
    assert sprites, "expected at least one sprite"
    sprites.sort(key=lambda item: (int(item["top"]), int(item["left"])))

    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    for sprite in sprites:
        for rr, cc in _filled_offsets(sprite["missing"]):
            out[int(sprite["top"]) + rr, int(sprite["left"]) + cc] = GREY

    for sprite in sprites:
        dr, dc = MOVE_BY_MISSING[tuple(sprite["missing"])]
        base_r = int(sprite["top"]) + 3 * dr
        base_c = int(sprite["left"]) + 3 * dc
        for rr, cc in _filled_offsets(sprite["missing"]):
            nr, nc = base_r + rr, base_c + cc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = int(sprite["color"])

    return out


# --------------- generator ---------------

GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 20
SPRITE_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 400
MAX_CLIPPED_SPRITES = 2
MIN_CLIPPED_VISIBLE_CELLS = 3


def _bbox_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r in range(3) for c in range(3)}


def _shifted_bbox_cells(
    top: int,
    left: int,
    missing: tuple[int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    dr, dc = MOVE_BY_MISSING[tuple(missing)]
    base_r = top + 3 * dr
    base_c = left + 3 * dc
    cells: set[tuple[int, int]] = set()
    for rr in range(3):
        for cc in range(3):
            nr, nc = base_r + rr, base_c + cc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
    return cells


def _shifted_paint_cells(
    top: int,
    left: int,
    missing: tuple[int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    dr, dc = MOVE_BY_MISSING[tuple(missing)]
    base_r = top + 3 * dr
    base_c = left + 3 * dc
    cells: set[tuple[int, int]] = set()
    for rr, cc in _filled_offsets(missing):
        nr, nc = base_r + rr, base_c + cc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def _guard(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    guarded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    guarded.add((nr, nc))
    return guarded


def _all_top_lefts(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h - 2) for c in range(w - 2)]


def _place_sprite(
    shape: tuple[int, int],
    missing: tuple[int, int],
    require_clip: bool,
    used_input_guard: set[tuple[int, int]],
    used_output_guard: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    candidates: list[dict] = []
    for top, left in _all_top_lefts(h, w):
        input_bbox = _bbox_cells(top, left)
        if input_bbox & used_input_guard:
            continue

        moved_bbox = _shifted_bbox_cells(top, left, missing, shape)
        moved_paint = _shifted_paint_cells(top, left, missing, shape)
        visible = len(moved_paint)
        if require_clip:
            if not (MIN_CLIPPED_VISIBLE_CELLS <= visible < 8):
                continue
        else:
            if visible != 8:
                continue

        output_bbox = input_bbox | moved_bbox
        if output_bbox & used_output_guard:
            continue

        candidates.append(
            {
                "top": top,
                "left": left,
                "input_bbox": input_bbox,
                "output_bbox": output_bbox,
            }
        )

    return random.choice(candidates) if candidates else None


def _direction_list(n_sprites: int) -> list[tuple[int, int]]:
    dirs = list(MISSING_CORNERS)
    random.shuffle(dirs)
    while len(dirs) < n_sprites:
        dirs.append(random.choice(MISSING_CORNERS))
    random.shuffle(dirs)
    return dirs[:n_sprites]


def _clip_flags(n_sprites: int) -> list[bool]:
    flags = [False] * n_sprites
    if random.random() < 0.5:
        clipped = random.randint(1, min(MAX_CLIPPED_SPRITES, n_sprites))
        for idx in random.sample(range(n_sprites), clipped):
            flags[idx] = True
    return flags


def _draw_sprite(grid: np.ndarray, top: int, left: int, color: int, missing: tuple[int, int]) -> None:
    patch = np.full((3, 3), int(color), dtype=int)
    patch[missing] = BG
    grid[top : top + 3, left : left + 3] = patch


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_ATTEMPTS):
        n_sprites = random.randint(*SPRITE_COUNT_RANGE)
        min_side = GRID_SIDE_MIN + max(0, n_sprites - SPRITE_COUNT_RANGE[0])
        h = random.randint(min_side, GRID_SIDE_MAX)
        w = random.randint(min_side, GRID_SIDE_MAX)
        shape = (h, w)

        colors = random.sample(SPRITE_COLORS, n_sprites)
        missings = _direction_list(n_sprites)
        clip_flags = _clip_flags(n_sprites)

        order = list(range(n_sprites))
        random.shuffle(order)
        order.sort(key=lambda idx: not clip_flags[idx])

        placements: list[dict | None] = [None] * n_sprites
        used_input_guard: set[tuple[int, int]] = set()
        used_output_guard: set[tuple[int, int]] = set()
        ok = True

        for idx in order:
            placement = _place_sprite(
                shape,
                missings[idx],
                clip_flags[idx],
                used_input_guard,
                used_output_guard,
            )
            if placement is None:
                ok = False
                break

            placements[idx] = placement
            used_input_guard |= _guard(placement["input_bbox"], shape)
            used_output_guard |= _guard(placement["output_bbox"], shape)

        if not ok:
            continue

        grid = np.zeros(shape, dtype=int)
        for idx, placement in enumerate(placements):
            assert placement is not None
            _draw_sprite(
                grid,
                int(placement["top"]),
                int(placement["left"]),
                int(colors[idx]),
                tuple(missings[idx]),
            )

        # Generator-side validation: the produced input should satisfy solver assumptions.
        sprites = [_parse_sprite(grid, comp) for comp in _find_components(grid)]
        if len(sprites) != n_sprites:
            continue
        if sorted(tuple(sprite["missing"]) for sprite in sprites).count((0, 0)) == 0:
            continue
        if sorted(tuple(sprite["missing"]) for sprite in sprites).count((0, 2)) == 0:
            continue
        if sorted(tuple(sprite["missing"]) for sprite in sprites).count((2, 0)) == 0:
            continue
        if sorted(tuple(sprite["missing"]) for sprite in sprites).count((2, 2)) == 0:
            continue

        out = main(grid)
        assert isinstance(out, np.ndarray)
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
