from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, neighbors4, paste


BG = Color.BLACK
RECT_COLOR = Color.BLUE
MARKER_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]
BODY_COLORS = [Color.GREY, Color.TEAL, Color.MAROON, Color.PINK]
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 18
GRID_MAX = 24
RECT_SIZES = (4, 5)
SPRITE_LEG_SIZES = (2, 3)
PLACEMENT_PADDING = 1
EDGE_MARGIN = 2
RECT_CLEARANCE = 3
GEN_ATTEMPTS = 400

_SIDE_COMBO_BAG: list[tuple[str, ...]] = []


def _find_rectangle(g: np.ndarray) -> dict:
    bg = get_bg(g)
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[RECT_COLOR],
        return_format="dict",
    )
    candidates = []
    center = np.array([(g.shape[0] - 1) / 2.0, (g.shape[1] - 1) / 2.0])
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        dist = abs((r0 + r1) / 2.0 - center[0]) + abs((c0 + c1) / 2.0 - center[1])
        candidates.append((comp["area"], -dist, comp))
    assert candidates, "expected one solid blue rectangle"
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _l_bend(body_coords: np.ndarray) -> tuple[int, int]:
    coords = [tuple(map(int, rc)) for rc in body_coords]
    coord_set = set(coords)
    bends = []
    for r, c in coords:
        nbrs = [(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in coord_set]
        has_vertical = any(nc == c for nr, nc in nbrs)
        has_horizontal = any(nr == r for nr, nc in nbrs)
        if has_vertical and has_horizontal:
            bends.append((r, c))
    assert len(bends) == 1, f"expected one L bend, got {bends}"
    return bends[0]


def _sprite_info(g: np.ndarray, comp: dict) -> dict:
    coords = comp["coords"]
    counts = Counter(int(g[r, c]) for r, c in coords)
    marker_colors = [color for color, count in counts.items() if count == 1]
    body_colors = [color for color, count in counts.items() if count > 1]
    assert len(counts) == 2, f"sprite must use exactly two colors, got {counts}"
    assert len(marker_colors) == 1 and len(body_colors) == 1, f"invalid sprite colors {counts}"

    body_color = int(body_colors[0])
    marker_color = int(marker_colors[0])
    body_coords = np.array(
        [(int(r), int(c)) for r, c in coords if int(g[r, c]) == body_color],
        dtype=int,
    )
    bend = _l_bend(body_coords)
    marker = next((int(r), int(c)) for r, c in coords if int(g[r, c]) == marker_color)
    cells = [(int(r), int(c), int(g[r, c])) for r, c in coords]
    return {
        "cells": cells,
        "bend": bend,
        "marker": marker,
        "marker_color": marker_color,
    }


def _parse_scene(g: np.ndarray, rect: dict) -> tuple[list[dict], dict[int, list[tuple[int, int]]]]:
    bg = get_bg(g)
    scene = g.copy()
    r0, c0, r1, c1 = rect["bbox"]
    scene[r0 : r1 + 1, c0 : c1 + 1] = bg

    controls: dict[int, list[tuple[int, int]]] = {}
    sprites = []
    for comp in find_components(scene, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
        if comp["area"] == 1:
            r, c = map(int, comp["coords"][0])
            controls.setdefault(int(scene[r, c]), []).append((r, c))
        else:
            sprites.append(_sprite_info(g, comp))
    return sprites, controls


def _translate_cells(cells: list[tuple[int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int]]:
    return [(int(r) + int(dr), int(c) + int(dc), int(color)) for r, c, color in cells]


def _excluded_lines(
    final_sprites: list[list[tuple[int, int, int]]],
    rect_bbox: tuple[int, int, int, int],
) -> tuple[set[int], set[int], set[int], set[int]]:
    r0, c0, r1, c1 = rect_bbox
    top_cols: set[int] = set()
    bottom_cols: set[int] = set()
    left_rows: set[int] = set()
    right_rows: set[int] = set()

    for sprite in final_sprites:
        for r, c, _color in sprite:
            if 0 <= r < r0 and c0 <= c <= c1:
                top_cols.add(int(c))
            if r1 < r < 10**9 and c0 <= c <= c1:
                bottom_cols.add(int(c))
            if r0 <= r <= r1 and 0 <= c < c0:
                left_rows.add(int(r))
            if r0 <= r <= r1 and c1 < c < 10**9:
                right_rows.add(int(r))
    return top_cols, bottom_cols, left_rows, right_rows


def _paint_stripes(out: np.ndarray, rect_bbox: tuple[int, int, int, int], final_sprites: list[list[tuple[int, int, int]]], bg: int) -> None:
    r0, c0, r1, c1 = rect_bbox
    top_cols, bottom_cols, left_rows, right_rows = _excluded_lines(final_sprites, rect_bbox)

    for c in range(c0, c1 + 1):
        if c not in top_cols:
            for r in range(0, r0):
                if int(out[r, c]) == bg:
                    out[r, c] = RECT_COLOR
        if c not in bottom_cols:
            for r in range(r1 + 1, out.shape[0]):
                if int(out[r, c]) == bg:
                    out[r, c] = RECT_COLOR

    for r in range(r0, r1 + 1):
        if r not in left_rows:
            for c in range(0, c0):
                if int(out[r, c]) == bg:
                    out[r, c] = RECT_COLOR
        if r not in right_rows:
            for c in range(c1 + 1, out.shape[1]):
                if int(out[r, c]) == bg:
                    out[r, c] = RECT_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marked L-sprite onto its color-matched side control, then extend the central rectangle only along side-band rows and columns not occupied by the moved sprites.

    Input structure:
    - The background is black, with one central solid blue rectangle.
    - Several multicolor sprites each contain one repeated body color in an L shape and one embedded marker pixel of a different color.
    - Matching singleton control pixels of those marker colors sit in the rectangle's top, bottom, left, or right side bands, separated from the rectangle and the sprites.

    Rule:
    - For each sprite, find the body L-bend and translate the whole sprite so that bend lands on the singleton control pixel of the same marker color; clear the sprite's old location and the consumed control.
    - After all moves, treat the rectangle's top and bottom side bands as column extensions and its left and right side bands as row extensions.
    - A column in the top or bottom band is excluded if any translated sprite pixel occupies that column inside the band; a row in the left or right band is excluded if any translated sprite pixel occupies that row inside the band.
    - Fill every non-excluded side-band background cell with blue, while leaving the moved sprites on top of the stripes.

    Key insight:
    - The sprite translation is also the exclusion signal: the moved L-shapes are repositioned precisely to decide which rectangle-aligned rows and columns may still emit blue stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = get_bg(g)
    rect = _find_rectangle(g)
    sprites, controls = _parse_scene(g, rect)

    moved_sources: set[tuple[int, int]] = set()
    used_controls: set[tuple[int, int]] = set()
    translated_sprites: list[list[tuple[int, int, int]]] = []
    final_sprites: list[list[tuple[int, int, int]]] = []

    for sprite in sprites:
        targets = controls.get(int(sprite["marker_color"]), [])
        if len(targets) != 1:
            final_sprites.append(sprite["cells"])
            continue

        target = targets[0]
        dr = int(target[0]) - int(sprite["bend"][0])
        dc = int(target[1]) - int(sprite["bend"][1])
        translated = _translate_cells(sprite["cells"], dr, dc)
        for r, c, _color in translated:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "sprite move must stay in bounds"
        translated_sprites.append(translated)
        final_sprites.append(translated)
        moved_sources.update((int(r), int(c)) for r, c, _color in sprite["cells"])
        used_controls.add(target)

    stationary_sprites = [sprite["cells"] for sprite in sprites if sprite["cells"] not in final_sprites]
    final_sprites.extend(stationary_sprites)

    out = g.copy()
    for r, c in moved_sources | used_controls:
        out[r, c] = bg

    occupied: set[tuple[int, int]] = set()
    for sprite in translated_sprites:
        for r, c, _color in sprite:
            assert int(out[r, c]) == bg, "translated sprite must land on cleared background"
            assert (r, c) not in occupied, "translated sprites must not overlap"
            occupied.add((r, c))
        for r, c, color in sprite:
            out[r, c] = int(color)

    _paint_stripes(out, rect["bbox"], final_sprites, bg)
    return out


def _draw_side_combo() -> tuple[str, ...]:
    global _SIDE_COMBO_BAG
    if not _SIDE_COMBO_BAG:
        combos = []
        for size in (2, 3, 4):
            combos.extend(combinations(SIDES, size))
        random.shuffle(combos)
        _SIDE_COMBO_BAG = [tuple(combo) for combo in combos]
    return _SIDE_COMBO_BAG.pop()


def _make_sprite_patch(body_color: int, marker_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    v_len = random.choice(SPRITE_LEG_SIZES)
    h_len = random.choice(SPRITE_LEG_SIZES)
    orient = random.choice(("NW", "NE", "SW", "SE"))
    patch = np.zeros((v_len, h_len), dtype=int)

    bend_r = 0 if orient in ("NW", "NE") else v_len - 1
    bend_c = 0 if orient in ("NW", "SW") else h_len - 1
    patch[bend_r, :] = body_color
    patch[:, bend_c] = body_color
    marker_r = bend_r + (1 if bend_r == 0 else -1)
    marker_c = bend_c + (1 if bend_c == 0 else -1)
    patch[marker_r, marker_c] = marker_color
    return patch, (bend_r, bend_c)


def _band_bbox(side: str, rect_bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect_bbox
    rows, cols = shape
    if side == "top":
        return 0, c0, r0 - 1, c1
    if side == "bottom":
        return r1 + 1, c0, rows - 1, c1
    if side == "left":
        return r0, 0, r1, c0 - 1
    assert side == "right"
    return r0, c1 + 1, r1, cols - 1


def _control_ok(side: str, anchor: tuple[int, int], rect_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect_bbox
    r, c = anchor
    if side == "top":
        return r <= r0 - 2
    if side == "bottom":
        return r >= r1 + 2
    if side == "left":
        return c <= c0 - 2
    return c >= c1 + 2


def _side_positions(
    side: str,
    patch: np.ndarray,
    bend: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    br0, bc0, br1, bc1 = _band_bbox(side, rect_bbox, shape)
    h, w = patch.shape
    if br1 - br0 + 1 < h or bc1 - bc0 + 1 < w:
        return []

    spots = []
    for r in range(br0, br1 - h + 2):
        for c in range(bc0, bc1 - w + 2):
            anchor = (r + bend[0], c + bend[1])
            if _control_ok(side, anchor, rect_bbox):
                spots.append((r, c))
    random.shuffle(spots)
    return spots


def _bbox_is_clear(canvas: np.ndarray, r: int, c: int, h: int, w: int, padding: int) -> bool:
    r0 = max(0, int(r) - padding)
    c0 = max(0, int(c) - padding)
    r1 = min(canvas.shape[0], int(r) + int(h) + padding)
    c1 = min(canvas.shape[1], int(c) + int(w) + padding)
    return not np.any(canvas[r0:r1, c0:c1] != BG)


def _far_from_rect(
    top: int,
    left: int,
    h: int,
    w: int,
    rect_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> bool:
    rows, cols = shape
    bottom = top + h - 1
    right = left + w - 1
    r0, c0, r1, c1 = rect_bbox
    near_edge = (
        top <= EDGE_MARGIN
        or left <= EDGE_MARGIN
        or bottom >= rows - 1 - EDGE_MARGIN
        or right >= cols - 1 - EDGE_MARGIN
    )
    clear_of_rect = (
        bottom < r0 - RECT_CLEARANCE
        or top > r1 + RECT_CLEARANCE
        or right < c0 - RECT_CLEARANCE
        or left > c1 + RECT_CLEARANCE
    )
    return near_edge and clear_of_rect


def _initial_positions(
    canvas: np.ndarray,
    patch: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    rows, cols = canvas.shape
    h, w = patch.shape
    spots = []
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            if not _far_from_rect(r, c, h, w, rect_bbox, canvas.shape):
                continue
            if _bbox_is_clear(canvas, r, c, h, w, PLACEMENT_PADDING):
                spots.append((r, c))
    random.shuffle(spots)
    return spots


def _choose_rect_origin(rows: int, cols: int, rect_h: int, rect_w: int) -> tuple[int, int]:
    row_choices = [
        r
        for r in range(4, rows - rect_h - 3)
        if abs((r + (rect_h - 1) / 2.0) - (rows - 1) / 2.0) <= 2.5
    ]
    col_choices = [
        c
        for c in range(4, cols - rect_w - 3)
        if abs((c + (rect_w - 1) / 2.0) - (cols - 1) / 2.0) <= 2.5
    ]
    return random.choice(row_choices), random.choice(col_choices)


def _gen_case() -> np.ndarray | None:
    rows = random.randint(GRID_MIN, GRID_MAX)
    cols = random.randint(GRID_MIN, GRID_MAX)
    rect_h = random.choice(RECT_SIZES)
    rect_w = random.choice(RECT_SIZES)
    rect_r0, rect_c0 = _choose_rect_origin(rows, cols, rect_h, rect_w)
    rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

    sides = list(_draw_side_combo())
    if rect_h <= 3 and any(side in ("left", "right") for side in sides):
        return None
    if rect_w <= 3 and any(side in ("top", "bottom") for side in sides):
        return None

    marker_colors = random.sample(MARKER_COLORS, len(sides))
    body_colors = random.sample(BODY_COLORS, len(sides))

    planned = []
    for side, body_color, marker_color in zip(sides, body_colors, marker_colors):
        patch, bend = _make_sprite_patch(body_color, marker_color)
        if side in ("top", "bottom") and patch.shape[1] >= rect_w:
            return None
        if side in ("left", "right") and patch.shape[0] >= rect_h:
            return None
        final_spots = _side_positions(side, patch, bend, rect_bbox, (rows, cols))
        if not final_spots:
            return None
        final_r0, final_c0 = final_spots[0]
        control = (final_r0 + bend[0], final_c0 + bend[1])
        planned.append(
            {
                "side": side,
                "patch": patch,
                "marker_color": marker_color,
                "control": control,
            }
        )

    inp = np.zeros((rows, cols), dtype=int)
    r0, c0, r1, c1 = rect_bbox
    inp[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR

    for spec in planned:
        cr, cc = spec["control"]
        if int(inp[cr, cc]) != BG:
            return None
        inp[cr, cc] = int(spec["marker_color"])

    for spec in planned:
        positions = _initial_positions(inp, spec["patch"], rect_bbox)
        if not positions:
            return None
        start_r0, start_c0 = positions[0]
        paste(inp, spec["patch"], start_r0, start_c0, transparent=BG)

    out = main(inp)
    added_blue = np.count_nonzero((out == RECT_COLOR) & (inp != RECT_COLOR))
    if added_blue == 0:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_case()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid sprite-translation stripe-exclusion puzzle")
