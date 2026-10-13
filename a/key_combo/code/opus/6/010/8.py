import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import colors_present, draw_line, find_components, get_bg, in_bounds


BODY_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
NON_BG_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATE_RETRIES = 400


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(cells):
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in DIRS.values():
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return seen == cells


def _tail_cell(direction, offset):
    if direction == "U":
        return (-1, offset)
    if direction == "D":
        return (2, offset)
    if direction == "L":
        return (offset, -1)
    return (offset, 2)


def _sprite_cells(top_left, direction, offset):
    r0, c0 = top_left
    cells = {(r0 + dr, c0 + dc) for dr, dc in BODY_CELLS}
    tr, tc = _tail_cell(direction, offset)
    cells.add((r0 + tr, c0 + tc))
    return cells


def _tail_shrinks_bbox(cells, reduced, direction):
    r0, c0, r1, c1 = _bbox(cells)
    rr0, rc0, rr1, rc1 = _bbox(reduced)
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return rr0 == r0 + 1 and rc0 == c0 and rr1 == r1 and rc1 == c1
    if (dr, dc) == (1, 0):
        return rr0 == r0 and rc0 == c0 and rr1 == r1 - 1 and rc1 == c1
    if (dr, dc) == (0, -1):
        return rr0 == r0 and rc0 == c0 + 1 and rr1 == r1 and rc1 == c1
    return rr0 == r0 and rc0 == c0 and rr1 == r1 and rc1 == c1 - 1


def _identify_tail(cells):
    candidates = []
    for cell in sorted(cells):
        neighbors = []
        r, c = cell
        for dr, dc in DIRS.values():
            nxt = (r + dr, c + dc)
            if nxt in cells:
                neighbors.append(nxt)
        if len(neighbors) != 1:
            continue
        nr, nc = neighbors[0]
        direction = (r - nr, c - nc)
        reduced = cells - {cell}
        if not _is_connected(reduced):
            continue
        if not _tail_shrinks_bbox(cells, reduced, direction):
            continue
        candidates.append((cell, direction, reduced))
    assert len(candidates) == 1, "expected exactly one bbox-extending tail spike"
    return candidates[0]


def _ray_hit(grid, start, direction, bg):
    r, c = start
    dr, dc = direction
    length = 0
    while True:
        r += dr
        c += dc
        assert in_bounds(grid.shape, r, c), "ray must hit before leaving the grid"
        length += 1
        if int(grid[r, c]) != bg:
            return (r, c), length


def _analyze(grid):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    sprites = []
    owner = {}
    for comp in comps:
        cells = {tuple(map(int, rc)) for rc in comp["coords"]}
        tail, direction, body = _identify_tail(cells)
        sprite = {
            "color": int(comp["color"]),
            "cells": cells,
            "body": body,
            "tail": tail,
            "direction": direction,
        }
        idx = len(sprites)
        sprites.append(sprite)
        for cell in cells:
            owner[cell] = idx
    assert 4 <= len(sprites) <= 5, "expected 4-5 sprites"
    lengths = []
    for idx, sprite in enumerate(sprites):
        hit, length = _ray_hit(g, sprite["tail"], sprite["direction"], bg)
        target = owner[hit]
        assert target != idx, "ray must hit another sprite"
        assert hit in sprites[target]["body"], "ray must hit a body pixel, not another tail"
        sprite["hit"] = hit
        sprite["target"] = target
        sprite["length"] = int(length)
        lengths.append(int(length))
    assert len(set(lengths)) == len(lengths), "ray lengths must be unique"
    return sprites, bg


def _simulate(grid, *, sort_by_length=True, cascade=True):
    sprites, bg = _analyze(grid)
    out = np.array(grid, dtype=int).copy()
    current = [int(sprite["color"]) for sprite in sprites]
    order = list(range(len(sprites)))
    if sort_by_length:
        order.sort(key=lambda idx: sprites[idx]["length"])
    changed_before_emit = 0
    for idx in order:
        color = current[idx] if cascade else int(sprites[idx]["color"])
        if color != int(sprites[idx]["color"]):
            changed_before_emit += 1
        tr, tc = sprites[idx]["tail"]
        hr, hc = sprites[idx]["hit"]
        draw_line(out, tr, tc, end_r=hr, end_c=hc, color=color)
        target = sprites[idx]["target"]
        for r, c in sprites[target]["cells"]:
            out[r, c] = color
        current[target] = color
    return out, changed_before_emit, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect spiked sprites by ray-hit segments, then recolor hit sprites in ray-length order to create a cascade.

    Input structure:
    - Black background with 4-5 separated small monochrome sprites.
    - Each sprite is a 4-connected body plus one 1-pixel tail spike protruding from its bbox.
    - Extending straight outward from every tail eventually reaches exactly one other sprite's body.

    Rule:
    - For each sprite, identify the spike tip and its outward direction.
    - Ray-cast from each spike tip to the first non-background cell, recording the hit target sprite and ray length.
    - Sort sprites by increasing ray length.
    - In that order, draw the full segment from the spike tip to the hit cell using the source sprite's current color, then recolor the entire hit sprite (body and tail) to that current color.
    - Recolored sprites later emit using their updated color; already drawn line segments stay as drawn.

    Key insight:
    - The ray graph alone is not enough: earlier recolors change the colors used by later emitters, so the sorted processing order creates the final cascade.
    """
    out, _changed, _sprites = _simulate(grid, sort_by_length=True, cascade=True)
    return out


def _transform_grid(grid, op):
    if op == "rot90":
        return np.rot90(grid, 1)
    if op == "rot180":
        return np.rot90(grid, 2)
    if op == "rot270":
        return np.rot90(grid, 3)
    if op == "flipud":
        return np.flipud(grid)
    if op == "fliplr":
        return np.fliplr(grid)
    if op == "flipud_rot90":
        return np.rot90(np.flipud(grid), 1)
    if op == "fliplr_rot90":
        return np.rot90(np.fliplr(grid), 1)
    return grid


def _build_patch():
    n = random.choice([4, 5])
    colors = random.sample(NON_BG_COLORS, n)

    valid_pairs = []
    for len_a in range(4, 9):
        for len_b in range(2, 6):
            lengths = {len_a, len_b, len_a + 2, len_b + 1}
            if len(lengths) == 4:
                valid_pairs.append((len_a, len_b))
    len_a, len_b = random.choice(valid_pairs)

    len_e = None
    if n == 5:
        choices = [
            cand
            for cand in range(2, 10)
            if cand not in {len_a, len_b, len_a + 2, len_b + 1}
        ]
        len_e = random.choice(choices)

    specs = [
        {"top": (0, 1), "direction": "R", "offset": 0, "color": colors[0]},
        {"top": (0, len_a + 3), "direction": "D", "offset": 1, "color": colors[1]},
        {"top": (len_b + 2, len_a + 4), "direction": "L", "offset": 1, "color": colors[2]},
        {"top": (len_b + 3, 0), "direction": "U", "offset": 1, "color": colors[3]},
    ]
    if n == 5:
        specs.append(
            {
                "top": (len_b + 2, len_a + 6 + len_e),
                "direction": "L",
                "offset": 0,
                "color": colors[4],
            }
        )

    all_cells = set()
    for spec in specs:
        cells = _sprite_cells(spec["top"], spec["direction"], spec["offset"])
        spec["cells"] = cells
        assert not (all_cells & cells), "sprites must not overlap"
        all_cells |= cells

    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for spec in specs:
        for r, c in spec["cells"]:
            patch[r - r0, c - c0] = int(spec["color"])
    return patch


def _embed_patch(patch):
    op = random.choice(
        ["id", "rot90", "rot180", "rot270", "flipud", "fliplr", "flipud_rot90", "fliplr_rot90"]
    )
    oriented = _transform_grid(patch, op)
    ph, pw = oriented.shape
    H = random.randint(max(ph + 2, 14), min(30, ph + 10))
    W = random.randint(max(pw + 2, 14), min(30, pw + 10))
    top = random.randint(1, H - ph - 1)
    left = random.randint(1, W - pw - 1)
    grid = np.zeros((H, W), dtype=int)
    grid[top : top + ph, left : left + pw] = oriented
    return grid


def _semantic_checks(inp):
    out = main(inp)
    alt_order, changed_before_emit, sprites = _simulate(inp, sort_by_length=False, cascade=True)
    no_cascade, _dummy, _dummy2 = _simulate(inp, sort_by_length=True, cascade=False)

    assert changed_before_emit >= 1, "at least one sprite must emit after being recolored"
    assert not np.array_equal(out, alt_order), "processing order must matter"
    assert not np.array_equal(out, no_cascade), "cascade recoloring must matter"
    assert len(colors_present(out, exclude={0})) >= 2, "output must preserve nontrivial color structure"

    lengths = [sprite["length"] for sprite in sprites]
    assert len(set(lengths)) == len(lengths)
    for idx, sprite in enumerate(sprites):
        target = sprite["target"]
        assert sprites[target]["target"] != idx, "mutual hits are not allowed"


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        patch = _build_patch()
        inp = _embed_patch(patch)
        try:
            _semantic_checks(inp)
        except AssertionError:
            continue
        return inp
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
