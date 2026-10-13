from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import bbox, crop_bbox, find_components, paste


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

ANCHOR_COLORS = (YELLOW, ORANGE)
BODY_COLORS = (6, 8, 9)
SQUARE_SIZES = (11, 13, 15)
OCCLUDER_SIZES = (3, 4)
GEN_RETRIES = 400

SPRITE_SHAPES = [
    ("111", "010"),
    ("110", "011", "010"),
    ("111", "001"),
    ("011", "110", "010"),
    ("110", "010", "011"),
    ("111", "101"),
    ("1100", "0111"),
    ("1110", "0011"),
]


@dataclass
class Sprite:
    patch: np.ndarray
    mask: np.ndarray
    anchor_color: int
    anchor: tuple[int, int]


def _shape_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _make_sprite(anchor_color: int, body_color: int) -> Sprite:
    mask = _random_transform(_shape_from_rows(tuple(random.choice(SPRITE_SHAPES))))
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    anchor = random.choice(coords)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(body_color)
    patch[anchor] = int(anchor_color)
    return Sprite(patch=patch, mask=mask, anchor_color=int(anchor_color), anchor=anchor)


def _build_source_panel(sprites: list[Sprite]) -> tuple[np.ndarray, dict[int, Sprite]]:
    ordered = sprites[:]
    random.shuffle(ordered)
    inner_w = max(int(sprite.patch.shape[1]) for sprite in ordered) + random.randint(1, 2)
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    gap = random.randint(1, 2)

    row = top_margin
    placements: list[tuple[int, int, Sprite]] = []
    for sprite in ordered:
        left = random.randint(0, inner_w - int(sprite.patch.shape[1]))
        placements.append((row, left, sprite))
        row += int(sprite.patch.shape[0]) + gap
    inner_h = row - gap + bottom_margin

    panel = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    panel[0, :] = GREY
    panel[-1, :] = GREY
    panel[:, 0] = GREY
    panel[:, -1] = GREY
    for r0, c0, sprite in placements:
        paste(panel, sprite.patch, r0 + 1, c0 + 1, mask=sprite.mask)
    return panel, {sprite.anchor_color: sprite for sprite in sprites}


def _base_pattern(n: int) -> np.ndarray:
    mode = random.randrange(4)
    center = n // 2
    out = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            a = max(abs(r - center), abs(c - center))
            b = min(abs(r - center), abs(c - center))
            if mode == 0:
                out[r, c] = BLUE if (a + b) % 2 == 0 else RED
            elif mode == 1:
                out[r, c] = BLUE if a % 2 == 0 else RED
            elif mode == 2:
                out[r, c] = BLUE if (a - b) % 2 == 0 else RED
            else:
                out[r, c] = BLUE if ((a + b) % 3 == 0 or a == b) else RED
    if random.random() < 0.5:
        out = np.where(out == BLUE, RED, BLUE)
    return out


def _d4_orbit(n: int, r: int, c: int) -> list[tuple[int, int]]:
    pts = {
        (r, c),
        (c, n - 1 - r),
        (n - 1 - r, n - 1 - c),
        (n - 1 - c, r),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (c, r),
        (n - 1 - c, n - 1 - r),
    }
    return sorted((int(rr), int(cc)) for rr, cc in pts)


def _stamp_cells(sprite: Sprite, anchor_pos: tuple[int, int]) -> tuple[int, int, set[tuple[int, int]]]:
    ar, ac = sprite.anchor
    top = int(anchor_pos[0]) - int(ar)
    left = int(anchor_pos[1]) - int(ac)
    cells = {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(sprite.mask)
    }
    return top, left, cells


def _candidate_positions(
    occ_mask: np.ndarray,
    sprite: Sprite,
) -> list[dict]:
    n = int(occ_mask.shape[0])
    occ_set = {tuple(map(int, rc)) for rc in np.argwhere(occ_mask)}
    h, w = map(int, sprite.patch.shape)
    candidates = []
    for r, c in occ_set:
        orbit = _d4_orbit(n, r, c)
        if len(orbit) == 1:
            continue
        if sum((pt in occ_set) for pt in orbit) != 1:
            continue
        top = int(r) - int(sprite.anchor[0])
        left = int(c) - int(sprite.anchor[1])
        if top < 0 or left < 0 or top + h > n or left + w > n:
            continue
        _, _, cells = _stamp_cells(sprite, (r, c))
        if all(cell in occ_set for cell in cells):
            continue
        candidates.append(
            {
                "anchor": (int(r), int(c)),
                "orbit": tuple(orbit),
                "cells": cells,
                "top": int(top),
                "left": int(left),
            }
        )
    random.shuffle(candidates)
    return candidates


def _choose_occluder_and_targets(
    n: int,
    sprite_by_color: dict[int, Sprite],
) -> tuple[np.ndarray, dict[int, dict]] | None:
    rects = []
    center = n // 2
    for oh in OCCLUDER_SIZES:
        for ow in OCCLUDER_SIZES:
            for r0 in range(1, n - oh):
                for c0 in range(1, n - ow):
                    if r0 <= center < r0 + oh and c0 <= center < c0 + ow:
                        continue
                    rects.append((r0, c0, oh, ow))
    random.shuffle(rects)

    for r0, c0, oh, ow in rects:
        occ_mask = np.zeros((n, n), dtype=bool)
        occ_mask[r0 : r0 + oh, c0 : c0 + ow] = True
        by_color = {
            color: _candidate_positions(occ_mask, sprite_by_color[color])
            for color in ANCHOR_COLORS
        }
        if not by_color[YELLOW] or not by_color[ORANGE]:
            continue
        for cand_a in by_color[YELLOW]:
            for cand_b in by_color[ORANGE]:
                if cand_a["orbit"] == cand_b["orbit"]:
                    continue
                if cand_a["cells"] & cand_b["cells"]:
                    continue
                return occ_mask, {YELLOW: cand_a, ORANGE: cand_b}
    return None


def _extract_templates(panel: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    inner = panel[1:-1, 1:-1]
    comps = find_components(inner, bg=BG, connectivity=4, monochromatic=False)
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = inner[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = np.zeros(patch.shape, dtype=bool)
        for rr, cc in comp["coords"]:
            mask[int(rr) - r0, int(cc) - c0] = True
        patch[~mask] = BG
        colors = Counter(int(patch[rr, cc]) for rr, cc in zip(*np.where(mask)))
        anchor_colors = [color for color, count in colors.items() if count == 1 and color in ANCHOR_COLORS]
        assert len(anchor_colors) == 1
        anchor_color = int(anchor_colors[0])
        anchor = tuple(map(int, np.argwhere((patch == anchor_color) & mask)[0]))
        templates[anchor_color] = (patch, mask, anchor)
    return templates


def _restore_square(square: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occ_mask = square == GREEN
    occ_box = bbox(occ_mask)
    assert occ_box is not None
    r0, c0, r1, c1 = occ_box
    assert bool(occ_mask[r0 : r1 + 1, c0 : c1 + 1].all())
    n = int(square.shape[0])
    restored = square.copy()
    for r, c in np.argwhere(occ_mask):
        colors = {
            int(square[rr, cc])
            for rr, cc in _d4_orbit(n, int(r), int(c))
            if not occ_mask[rr, cc]
        }
        assert len(colors) == 1
        restored[int(r), int(c)] = colors.pop()
    return restored, occ_mask


def _stamp_hidden_anchors(
    square: np.ndarray,
    occ_mask: np.ndarray,
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
) -> np.ndarray:
    out = square.copy()
    hidden = sorted(
        tuple(map(int, rc))
        for rc in np.argwhere(occ_mask & np.isin(square, list(ANCHOR_COLORS)))
    )
    for r, c in hidden:
        color = int(square[r, c])
        patch, mask, (ar, ac) = templates[color]
        paste(out, patch, int(r) - int(ar), int(c) - int(ac), mask=mask)
    return out


def _find_panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = bbox(grid == GREY)
    assert box is not None
    r0, c0, r1, c1 = box
    panel = crop_bbox(grid, box)
    assert bool((panel[0, :] == GREY).all() and (panel[-1, :] == GREY).all())
    assert bool((panel[:, 0] == GREY).all() and (panel[:, -1] == GREY).all())
    return int(r0), int(c0), int(r1), int(c1)


def _find_square_bbox(grid: np.ndarray, panel_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    mask = grid != BG
    r0, c0, r1, c1 = panel_box
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    box = bbox(mask)
    assert box is not None
    sr0, sc0, sr1, sc1 = map(int, box)
    assert sr1 - sr0 == sc1 - sc0
    return sr0, sc0, sr1, sc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden symmetric anchor pixels, then stamp the matching source sprites onto them and crop the target square.

    Input structure:
    - Black background with a grey bordered source panel on the left and one colored square pattern on the right.
    - The source panel contains two 4-connected multicolor sprites, each with one unique anchor pixel in yellow or orange.
    - The right square is D4-symmetric except for a solid green occluder rectangle hiding part of the pattern.

    Rule:
    - Identify the green rectangle as an occluder and restore every covered cell from its unoccluded D4-symmetric equivalents.
    - Inside the recovered occluder, find the yellow and orange anchor pixels.
    - In the source panel, match each anchor color to the sprite containing that unique anchor pixel and translate the sprite so the anchor lands on the recovered target pixel.
    - Return only the restored-and-stamped right square.

    Key insight:
    - The stamp locations are not directly visible: symmetry completion is required first, because only the recovered hidden anchor pixels reveal where each sprite must be placed.
    """
    g = np.asarray(grid, dtype=int)
    panel_box = _find_panel_bbox(g)
    panel = crop_bbox(g, panel_box)
    templates = _extract_templates(panel)
    square_box = _find_square_bbox(g, panel_box)
    square = crop_bbox(g, square_box)
    restored, occ_mask = _restore_square(square)
    return _stamp_hidden_anchors(restored, occ_mask, templates)


def _compose_canvas(panel: np.ndarray, square: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    panel_h, panel_w = map(int, panel.shape)
    square_n = int(square.shape[0])
    grid_h = max(panel_h, square_n) + random.randint(1, 4)
    left_margin = random.randint(0, 1)
    gap = random.randint(2, 4)
    right_margin = random.randint(0, 2)
    grid_w = left_margin + panel_w + gap + square_n + right_margin
    if grid_h > 30 or grid_w > 30:
        raise ValueError("canvas too large")

    grid = np.zeros((grid_h, grid_w), dtype=int)
    panel_top = random.randint(0, grid_h - panel_h)
    square_top = random.randint(0, grid_h - square_n)
    panel_left = left_margin
    square_left = panel_left + panel_w + gap

    paste(grid, panel, panel_top, panel_left)
    paste(grid, square, square_top, square_left)
    panel_box = (panel_top, panel_left, panel_top + panel_h - 1, panel_left + panel_w - 1)
    square_box = (square_top, square_left, square_top + square_n - 1, square_left + square_n - 1)
    return grid, panel_box, square_box


def _render_expected(full_square: np.ndarray, targets: dict[int, dict], sprite_by_color: dict[int, Sprite]) -> np.ndarray:
    out = full_square.copy()
    for color in ANCHOR_COLORS:
        sprite = sprite_by_color[color]
        top = int(targets[color]["top"])
        left = int(targets[color]["left"])
        paste(out, sprite.patch, top, left, mask=sprite.mask)
    return out


def _full_square_is_d4_symmetric(square: np.ndarray) -> bool:
    n = int(square.shape[0])
    for r in range(n):
        for c in range(n):
            colors = {int(square[rr, cc]) for rr, cc in _d4_orbit(n, r, c)}
            if len(colors) != 1:
                return False
    return True


def _validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    panel_box: tuple[int, int, int, int],
    full_square: np.ndarray,
    occ_mask: np.ndarray,
    targets: dict[int, dict],
) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert expected.shape[0] == expected.shape[1]
    assert _full_square_is_d4_symmetric(full_square)

    panel = crop_bbox(inp, panel_box)
    assert bool((panel[0, :] == GREY).all() and (panel[-1, :] == GREY).all())
    assert bool((panel[:, 0] == GREY).all() and (panel[:, -1] == GREY).all())
    inner = panel[1:-1, 1:-1]
    comps = find_components(inner, bg=BG, connectivity=4, monochromatic=False)
    assert len(comps) == 2

    comp_anchor_colors = []
    for comp in comps:
        colors = Counter(int(inner[int(r), int(c)]) for r, c in comp["coords"])
        anchor_candidates = [color for color, count in colors.items() if count == 1 and color in ANCHOR_COLORS]
        assert len(anchor_candidates) == 1
        comp_anchor_colors.append(int(anchor_candidates[0]))
    assert set(comp_anchor_colors) == set(ANCHOR_COLORS)

    hidden_anchor_cells = np.argwhere(occ_mask & np.isin(full_square, list(ANCHOR_COLORS)))
    assert hidden_anchor_cells.shape[0] == 2
    assert {int(full_square[int(r), int(c)]) for r, c in hidden_anchor_cells} == set(ANCHOR_COLORS)

    for color in ANCHOR_COLORS:
        anchor = targets[color]["anchor"]
        orbit = _d4_orbit(int(full_square.shape[0]), int(anchor[0]), int(anchor[1]))
        assert sum(bool(occ_mask[pt]) for pt in orbit) == 1
        assert sum(not bool(occ_mask[pt]) for pt in orbit) >= 1

    restored_only = full_square.copy()
    diff = np.argwhere(expected != restored_only)
    assert diff.shape[0] >= 4
    assert any(not bool(occ_mask[int(r), int(c)]) for r, c in diff)

    solved = main(inp)
    assert np.array_equal(solved, expected)


def _generate_scene() -> tuple[np.ndarray, np.ndarray]:
    n = random.choice(SQUARE_SIZES)
    body_colors = random.sample(BODY_COLORS, 2)
    sprite_by_color = {
        YELLOW: _make_sprite(YELLOW, body_colors[0]),
        ORANGE: _make_sprite(ORANGE, body_colors[1]),
    }
    panel, sprite_by_color = _build_source_panel([sprite_by_color[YELLOW], sprite_by_color[ORANGE]])

    choice = _choose_occluder_and_targets(n, sprite_by_color)
    if choice is None:
        raise ValueError("no valid occluder placement")
    occ_mask, targets = choice

    full_square = _base_pattern(n)
    for color in ANCHOR_COLORS:
        for r, c in targets[color]["orbit"]:
            full_square[int(r), int(c)] = int(color)

    input_square = full_square.copy()
    input_square[occ_mask] = GREEN
    expected = _render_expected(full_square, targets, sprite_by_color)

    inp, panel_box, _square_box = _compose_canvas(panel, input_square)
    _validate_scene(inp, expected, panel_box, full_square, occ_mask, targets)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, _expected = _generate_scene()
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate valid puzzle input")
