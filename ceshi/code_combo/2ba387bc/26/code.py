# Combo variant of 2ba387bc (index 26)
# combo_spec: D2/3+D8/1
# Name: 两轮排序+颜色排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for the fused variant
# ====================================
# a) Concrete behavior trace
#    Original task:
#    - detect all scattered 4x4 objects;
#    - classify them into frames vs solids;
#    - inside each class, sort by spatial order `(row, col)`;
#    - stack frames in the left 4 columns and solids in the right 4 columns.
#    Fused task:
#    - round 1 is unchanged: classify by type into left/right columns;
#    - round 2 replaces spatial sorting with visible color-sequence sorting.
#      Example using the palette order red -> orange -> yellow -> green -> teal -> blue:
#      if the frame colors appear top-to-bottom as [blue, red, yellow], the original output column is
#      [blue, red, yellow], but the fused output column must be [red, yellow, blue].
#      The same second-round reorder happens independently inside the solid column.
#
# b) Complexity calibration
#    Hidden numeric color ids are not human-visible, so the solver does not sort by raw integer value.
#    Instead it uses a fixed visible spectrum-like order over a small palette:
#    red(2), orange(7), yellow(4), green(3), teal(8), blue(1).
#    The generator only samples from this palette and uses 2-3 objects per type so each example still
#    looks clean while giving enough evidence for the color-order rule.
#
# c) Implicit assumptions audit
#    - `get_bg` assumes background is dominant: generator keeps the canvas sparse and background=0.
#    - `find_components(... monochromatic=False)` assumes each 4x4 object is disconnected from others:
#      generator places every object with a 1-cell moat.
#    - `bbox_h/bbox_w == 4` assumes no merged or partial objects: generator only places intact 4x4 sprites.
#    - Frame detection assumes a single-color border plus bg-filled 2x2 interior; solid detection assumes
#      one non-bg color fills the 4x4 block. Generator constructs exactly those two object families.
#    - Color sorting assumes every object's main color is in the visible palette and ties are avoided.
#      Generator samples distinct colors within each type, so ordering is strict and legible.
#
# d) Placement legality
#    `random_free_location` with `border=1` and `padding=1` guarantees every 4x4 sprite stays in-bounds,
#    does not overlap others, and does not 4-touch/8-touch in a way that would merge components.
#
# e) Visual observability / f) cleanliness / g) inferability
#    The only new cue is the order of already-salient block colors. No auxiliary markers are added.
#    The palette is small and visually ordered; each sample contains multiple objects per type; and the
#    generator rejects degenerate placements where color order accidentally matches spatial order everywhere.
#    This keeps the fusion visible from 2-4 examples without clutter.
#
# 9) New assumptions introduced by the fused solver
#    The new solver depends on a fixed visible palette order. The generator enforces that by drawing colors
#    only from `COLOR_ORDER` and by requiring a non-zero `sort_disagreement` score, meaning at least one
#    column's color-sorted order differs from the original spatial order on the same input.

COLOR_ORDER = [2, 7, 4, 3, 8, 1]  # red -> orange -> yellow -> green -> teal -> blue
COLOR_RANK = {color: idx for idx, color in enumerate(COLOR_ORDER)}


def _visible_color_rank(color: int) -> tuple[int, int]:
    color = int(color)
    return int(COLOR_RANK.get(color, len(COLOR_ORDER) + color)), color


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _classify_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[dict] = []
    solids: list[dict] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        border_colors = {int(x) for x in border.flatten()}
        fill_colors = {int(x) for x in sub.flatten()}

        if len(border_colors) == 1 and bool((sub[1:3, 1:3] == bg).all()):
            color = int(next(iter(border_colors)))
            frames.append({"r0": int(r0), "c0": int(c0), "sub": sub, "color": color})
        elif len(fill_colors) == 1 and bg not in fill_colors:
            color = int(next(iter(fill_colors)))
            solids.append({"r0": int(r0), "c0": int(c0), "sub": sub, "color": color})

    return bg, frames, solids


def _position_key(obj: dict) -> tuple[int, int]:
    return int(obj["r0"]), int(obj["c0"])


def _color_key(obj: dict) -> tuple[int, int, int, int]:
    rank_major, rank_minor = _visible_color_rank(int(obj["color"]))
    return rank_major, rank_minor, int(obj["r0"]), int(obj["c0"])


def sort_disagreement(grid: np.ndarray) -> int:
    """
    Quantify how strongly the fused rule differs from the original row-based rule on this input.

    Score = number of positions where the per-type color sequence differs between
    spatial ordering `(row, col)` and visible-color ordering.
    """
    _bg, frames, solids = _classify_objects(grid)
    score = 0
    for group in (frames, solids):
        by_pos = [int(obj["color"]) for obj in sorted(group, key=_position_key)]
        by_col = [int(obj["color"]) for obj in sorted(group, key=_color_key)]
        score += sum(int(a != b) for a, b in zip(by_pos, by_col))
    return int(score)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Find all scattered 4x4 objects and classify them into frames vs solids.
    2. Build the output as two columns: frames on the left, solids on the right.
    3. Inside each column, reorder objects by the visible palette order
       red -> orange -> yellow -> green -> teal -> blue, not by their input positions.
    """
    bg, frames, solids = _classify_objects(grid)

    frames.sort(key=_color_key)
    solids.sort(key=_color_key)

    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), bg, dtype=int)

    for i, obj in enumerate(frames):
        paste(out, obj["sub"], i * 4, 0)
    for i, obj in enumerate(solids):
        paste(out, obj["sub"], i * 4, 4)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 30))
    w = int(np.random.randint(20, 30))
    grid = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(2, 4))
    n_solids = int(np.random.randint(2, 4))

    total = n_frames + n_solids
    if total > len(COLOR_ORDER):
        return None

    chosen = list(np.random.permutation(COLOR_ORDER)[:total])
    frame_colors = chosen[:n_frames]
    solid_colors = chosen[n_frames:]

    try:
        for color in frame_colors:
            patch = _make_frame(int(color), bg)
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(grid, patch, r0, c0, transparent=bg)

        for color in solid_colors:
            patch = _make_solid(int(color))
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(grid, patch, r0, c0, transparent=bg)
    except Exception:
        return None

    _, frames, solids = _classify_objects(grid)
    if len(frames) != n_frames or len(solids) != n_solids:
        return None

    # Reject samples that accidentally preserve the original row-order behavior.
    if sort_disagreement(grid) <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
