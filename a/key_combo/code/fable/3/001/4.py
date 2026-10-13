from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, get_bg


TEAL = int(Color.TEAL)


def _object_plans(g: np.ndarray, bg: int) -> list[dict]:
    marker_coords = {tuple(map(int, p)) for p in np.argwhere(g == TEAL)}
    assert marker_coords, "expected at least one teal control pixel"

    colors = colors_present(g, exclude={bg, TEAL})
    assert colors, "expected at least one colored object"
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=colors)
    assert objects, "expected monochromatic foreground objects"

    plans: list[dict] = []
    used_markers: set[tuple[int, int]] = set()
    for obj in objects:
        color = int(obj["color"])
        coords = np.asarray(obj["coords"], dtype=int)
        r0, c0, r1, c1 = map(int, obj["bbox"])

        candidates: list[tuple[int, int, tuple[int, int]]] = []
        if c0 > 0:
            for r in range(r0, r1 + 1):
                if int(g[r, c0 - 1]) == TEAL:
                    candidates.append((r, -1, (r, c0 - 1)))
        if c1 + 1 < g.shape[1]:
            for r in range(r0, r1 + 1):
                if int(g[r, c1 + 1]) == TEAL:
                    candidates.append((r, 1, (r, c1 + 1)))

        assert len(candidates) == 1, "each object must have exactly one side control pixel"
        anchor, direction, marker = candidates[0]
        assert marker not in used_markers, "a control pixel cannot belong to two objects"
        assert np.any(coords[:, 0] == anchor), "control pixel row must cross the object"

        used_markers.add(marker)
        plans.append(
            {
                "color": color,
                "coords": coords,
                "anchor": int(anchor),
                "direction": int(direction),
            }
        )

    assert used_markers == marker_coords, "every teal pixel must control exactly one object"
    return plans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shear each foreground object sideways from the row marked by its teal control pixel.

    Input structure:
    - A dominant background surrounds several separated monochromatic 4-connected objects.
    - Each object has exactly one teal control pixel immediately beside the left or right
      side of its bounding box, on a row that crosses that object.

    Rule:
    - Clear all foreground objects and teal control pixels to the background.
    - For each object, use the teal pixel's row as the anchor row and its side as the
      horizontal direction.
    - Repaint every object pixel (r, c) at (r, c +/- abs(r - anchor_row)); anchor-row
      pixels stay in place, rows farther from the anchor move farther, and off-grid
      pixels are clipped.

    Key insight:
    - The shear is not tied to a fixed bottom row or fixed left direction; the one-use
      teal pixel supplies both the fold row and the side toward which all rows shear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    assert bg != TEAL, "teal is reserved for control pixels"
    assert int(np.count_nonzero(g == bg)) > g.size // 2, "background must be dominant"

    plans = _object_plans(g, bg)
    out = np.full_like(g, bg)
    occupied = np.zeros_like(g, dtype=bool)
    height, width = map(int, g.shape)

    for plan in plans:
        color = int(plan["color"])
        anchor = int(plan["anchor"])
        direction = int(plan["direction"])
        for r, c in plan["coords"]:
            rr = int(r)
            cc = int(c) + direction * abs(int(r) - anchor)
            if 0 <= rr < height and 0 <= cc < width:
                assert not occupied[rr, cc], "sheared objects must not overlap"
                out[rr, cc] = color
                occupied[rr, cc] = True

    return out


# --------------- generator ---------------

BACKGROUND = int(Color.BLACK)
OBJECT_COLORS = tuple(c for c in Color.ALL_COLORS if c not in {BACKGROUND, TEAL})
GRID_H_RANGE = (17, 28)
GRID_W_RANGE = (19, 30)
OBJECT_COUNT_RANGE = (3, 4)
SPRITE_H_RANGE = (4, 7)
SPRITE_W_RANGE = (3, 6)
MAX_LOST_PIXELS = 8
GENERATION_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 80
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _paint_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = map(int, start)
    tr, tc = map(int, end)
    mask[r, c] = True
    axes = [0, 1]
    random.shuffle(axes)
    for axis in axes:
        if axis == 0:
            while r != tr:
                r += 1 if tr > r else -1
                mask[r, c] = True
        else:
            while c != tc:
                c += 1 if tc > c else -1
                mask[r, c] = True


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for dr, dc in ORTHOGONAL_STEPS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _grow_connected(mask: np.ndarray, target: int) -> None:
    while int(mask.sum()) < target:
        cells = _frontier(mask)
        if not cells:
            return
        r, c = random.choice(cells)
        mask[r, c] = True


def _make_sprite_mask(h: int, w: int, side: int, anchor: int) -> np.ndarray:
    side_col = 0 if side == -1 else w - 1
    other_col = w - 1 - side_col
    mask = np.zeros((h, w), dtype=bool)
    seed = (anchor, side_col)

    targets = [
        (0, side_col),
        (h - 1, side_col),
        (random.randrange(h), other_col),
        (random.choice((0, h - 1)), random.randrange(w)),
    ]
    random.shuffle(targets)
    for target in targets:
        _paint_path(mask, seed, target)

    area = h * w
    lo = int(mask.sum())
    hi = max(lo, min(area, int(np.ceil(area * 0.78))))
    _grow_connected(mask, random.randint(lo, hi))

    assert mask[anchor, side_col]
    assert np.any(mask[0, :]) and np.any(mask[-1, :])
    assert np.any(mask[:, 0]) and np.any(mask[:, -1])
    return mask


def _make_specs() -> tuple[list[dict], int]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(list(OBJECT_COLORS), count)

    sides = [-1, 1]
    while len(sides) < count:
        sides.append(random.choice((-1, 1)))
    random.shuffle(sides)

    clip_index = random.randrange(count)
    second_interior = random.randrange(count)
    specs: list[dict] = []
    for i in range(count):
        h = random.randint(*SPRITE_H_RANGE)
        w = random.randint(*SPRITE_W_RANGE)
        if i in {clip_index, second_interior} or random.random() < 0.75:
            anchor = random.randint(1, h - 2)
        else:
            anchor = random.choice((0, h - 1))
        side = int(sides[i])
        mask = _make_sprite_mask(h, w, side, anchor)
        specs.append(
            {
                "mask": mask,
                "h": h,
                "w": w,
                "side": side,
                "anchor": anchor,
                "color": int(colors[i]),
                "want_clip": i == clip_index,
            }
        )
    return specs, clip_index


def _input_cells(spec: dict, r0: int, c0: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    coords = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(spec["mask"])]
    marker_c = c0 - 1 if int(spec["side"]) == -1 else c0 + int(spec["w"])
    marker = (r0 + int(spec["anchor"]), marker_c)
    return coords, marker


def _shifted_cells(
    spec: dict, r0: int, c0: int, shape: tuple[int, int], *, anchor_mode: str = "correct", direction_mode: str = "correct"
) -> tuple[list[tuple[int, int]], int, bool]:
    height, width = map(int, shape)
    anchor = int(spec["anchor"]) if anchor_mode == "correct" else int(spec["h"]) - 1
    direction = int(spec["side"]) if direction_mode == "correct" else -1

    cells: list[tuple[int, int]] = []
    lost = 0
    shifted_visible = False
    for rr, cc in np.argwhere(spec["mask"]):
        rr = int(rr)
        cc = int(cc)
        out_r = r0 + rr
        offset = abs(rr - anchor)
        out_c = c0 + cc + direction * offset
        if 0 <= out_r < height and 0 <= out_c < width:
            cells.append((out_r, out_c))
            if offset:
                shifted_visible = True
        else:
            lost += 1
    return cells, lost, shifted_visible


def _clear(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not bool(blocked[r, c]) for r, c in cells)


def _mark_padded(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    height, width = map(int, blocked.shape)
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    blocked[nr, nc] = True


def _candidate_positions(shape: tuple[int, int], spec: dict) -> list[tuple[int, int, list[tuple[int, int]], tuple[int, int], list[tuple[int, int]], int]]:
    height, width = map(int, shape)
    h = int(spec["h"])
    w = int(spec["w"])
    want_clip = bool(spec["want_clip"])

    candidates = []
    for r0 in range(1, height - h):
        for c0 in range(1, width - w):
            object_cells, marker = _input_cells(spec, r0, c0)
            mr, mc = marker
            if not (0 <= mr < height and 0 <= mc < width):
                continue
            shifted, lost, shifted_visible = _shifted_cells(spec, r0, c0, shape)
            if not shifted or not shifted_visible:
                continue
            if want_clip:
                if not (0 < lost <= MAX_LOST_PIXELS):
                    continue
            elif lost != 0:
                continue
            if len(set(shifted)) != len(shifted):
                continue
            candidates.append((r0, c0, object_cells, marker, shifted, lost))

    random.shuffle(candidates)
    return candidates


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    order = sorted(
        range(len(specs)),
        key=lambda i: (not bool(specs[i]["want_clip"]), -int(specs[i]["mask"].sum())),
    )
    candidates_by_index = [_candidate_positions(shape, spec) for spec in specs]
    if any(not candidates for candidates in candidates_by_index):
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        input_blocked = np.zeros(shape, dtype=bool)
        output_blocked = np.zeros(shape, dtype=bool)
        placed: list[dict | None] = [None] * len(specs)
        ok = True

        for idx in order:
            choices = candidates_by_index[idx][:]
            random.shuffle(choices)
            chosen = None
            for r0, c0, object_cells, marker, shifted, lost in choices:
                package = object_cells + [marker]
                if not _clear(input_blocked, package):
                    continue
                if not _clear(output_blocked, shifted):
                    continue
                chosen = {
                    **specs[idx],
                    "r0": r0,
                    "c0": c0,
                    "object_cells": object_cells,
                    "marker": marker,
                    "shifted": shifted,
                    "lost": lost,
                }
                _mark_padded(input_blocked, package, pad=1)
                _mark_padded(output_blocked, shifted, pad=1)
                break

            if chosen is None:
                ok = False
                break
            placed[idx] = chosen

        if ok and all(p is not None for p in placed):
            return [p for p in placed if p is not None]
    return None


def _render_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in placed:
        color = int(spec["color"])
        for r, c in spec["object_cells"]:
            grid[r, c] = color
        mr, mc = spec["marker"]
        grid[mr, mc] = TEAL
    return grid


def _render_variant(
    shape: tuple[int, int],
    placed: list[dict],
    *,
    anchor_mode: str = "correct",
    direction_mode: str = "correct",
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    occupied: set[tuple[int, int]] = set()
    for spec in placed:
        shifted, _lost, _shifted_visible = _shifted_cells(
            spec,
            int(spec["r0"]),
            int(spec["c0"]),
            shape,
            anchor_mode=anchor_mode,
            direction_mode=direction_mode,
        )
        for r, c in shifted:
            if anchor_mode == "correct" and direction_mode == "correct":
                assert (r, c) not in occupied
            occupied.add((r, c))
            out[r, c] = int(spec["color"])
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, placed: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if int(np.count_nonzero(inp == BACKGROUND)) <= inp.size // 2:
        return False
    if set(int(spec["side"]) for spec in placed) != {-1, 1}:
        return False
    if not any(0 < int(spec["anchor"]) < int(spec["h"]) - 1 for spec in placed):
        return False
    if not (any(int(spec["lost"]) > 0 for spec in placed) and any(int(spec["lost"]) == 0 for spec in placed)):
        return False
    if not all(np.any(np.argwhere(spec["mask"])[:, 0] < int(spec["anchor"])) and np.any(np.argwhere(spec["mask"])[:, 0] > int(spec["anchor"])) for spec in placed if 0 < int(spec["anchor"]) < int(spec["h"]) - 1):
        return False
    if np.any(expected == TEAL):
        return False
    for spec in placed:
        if not np.any(expected == int(spec["color"])):
            return False

    wrong_bottom_anchor = _render_variant(inp.shape, placed, anchor_mode="bottom", direction_mode="correct")
    wrong_fixed_left = _render_variant(inp.shape, placed, anchor_mode="correct", direction_mode="left")
    wrong_bottom_left = _render_variant(inp.shape, placed, anchor_mode="bottom", direction_mode="left")
    if np.array_equal(expected, wrong_bottom_anchor):
        return False
    if np.array_equal(expected, wrong_fixed_left):
        return False
    if np.array_equal(expected, wrong_bottom_left):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    shape = (height, width)

    specs, _clip_index = _make_specs()
    placed = _place_specs(shape, specs)
    if placed is None:
        return None

    inp = _render_input(shape, placed)
    expected = _render_variant(shape, placed)
    if not _semantic_checks(inp, expected, placed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid control-anchored shear input")
