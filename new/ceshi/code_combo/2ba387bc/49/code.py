# Combo variant of 2ba387bc (index 49)
# combo_spec: D5/1+D7/2+D14/0
# Name: 颜色互换+反向堆叠+分隔列
#
# Design analysis for this combo variant
# -------------------------------------
# a) Concrete behavior trace:
# - Original rule: detect every 4x4 object, classify it as frame or solid, sort
#   each class by input row, then stack both classes from top to bottom with no
#   gap between the two columns.
# - Fused rule:
#   * D5/1 inserts a 1-cell background separator, so the output width becomes 9
#     and the right column starts at col 5 instead of col 4.
#   * D14/0 keeps frames stacked from the top, but solids are assigned from the
#     bottom upward. With three solids S1,S2,S3 sorted by input row, their output
#     row slots become bottom/middle/top respectively.
#   * D7/2 swaps colors across matching output row slots, not across raw sorted
#     indices. That point is crucial: because the right column is reversed, the
#     frame in slot 0 swaps with the solid that ends up in slot 0, which is the
#     bottom-most input solid when counts match.
# - Example with frames F1,F2,F3 and solids S1,S2,S3 (both sorted top->bottom):
#   old output rows:   [F1|S1], [F2|S2], [F3|S3]
#   new row slots:     [F1|S3], [F2|S2], [F3|S1]
#   after color swap:  [frame(F1 shape, S3 color) | solid(S3 shape, F1 color)]
#                      [frame(F2 shape, S2 color) | solid(S2 shape, F2 color)]
#                      [frame(F3 shape, S1 color) | solid(S1 shape, F3 color)]
#   plus a background separator column in the middle.
#
# b) Complexity calibration:
# - The generator uses 2..4 frame/solid pairs with globally unique non-bg colors.
#   This keeps the swap visible: every output row exposes a one-to-one color move.
# - It deliberately avoids same-row ties within the frame group or the solid group.
#   Otherwise the hidden `(row, col)` tie-break from the solver would matter more
#   than the human-visible "top to bottom" cue.
# - Equal counts are favored in generated samples so every row slot participates in
#   the cross-column swap. That makes the fusion dimensions reinforce each other
#   instead of leaving unmatched rows that weaken the evidence.
#
# c) Implicit-assumption audit against the original solver:
# - `outW = 8` is no longer valid; the separator changes the layout width to 9.
# - `paste(..., col=4)` for the right column is no longer valid; the right column
#   starts at col 5.
# - `for i, solid in enumerate(solids): row = i * 4` is no longer valid; solids
#   must be mapped into row slots from the bottom upward.
# - "same index means corresponding objects" is no longer valid once the right
#   column is reversed. The solver now builds explicit row-slot arrays first and
#   swaps colors only after both columns have been assigned to final slots.
# - The original broad "else => solid" classification is tightened. The fused
#   solver assumes valid 4x4 monochrome frames and 4x4 monochrome solids; the
#   generator guarantees exactly those object types.
#
# d) Placement legality:
# - The generator scatters 4x4 sprites with `random_free_location(..., padding=1)`,
#   so objects do not overlap, do not touch orthogonally, and stay in-bounds.
# - A post-check re-extracts components and rejects any sample that would create
#   row-order ties or lose an object.
#
# e) Visual observability:
# - The separator is directly visible as an empty middle column.
# - Reverse stacking is visible because the right-column solid sequence is the
#   opposite of its input top-to-bottom order.
# - Color swap is visible because each frame takes the color of the solid on the
#   same output row slot, and vice versa.
# - The generator rejects degenerate samples whose output matches either of two
#   same-width ablations: "no color swap" or "no reverse-right stacking".

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for _utils_dir in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    if _utils_dir.exists() and str(_utils_dir) not in sys.path:
        sys.path.append(str(_utils_dir))

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste, random_free_location


Obj = tuple[int, int, np.ndarray]


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    return len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    colors = set(map(int, sub.flatten()))
    return len(colors) == 1 and int(bg) not in colors


def _obj_color(sub: np.ndarray, bg: int) -> int:
    colors = list(set(map(int, sub.flatten())) - {int(bg)})
    if len(colors) != 1:
        raise ValueError("object must contain exactly one non-background color")
    return int(colors[0])


def _recolor(sub: np.ndarray, bg: int, new_color: int) -> np.ndarray:
    out = np.array(sub, dtype=int, copy=True)
    out[out != int(bg)] = int(new_color)
    return out


def _extract_objects(grid: np.ndarray) -> tuple[int, list[Obj], list[Obj]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[Obj] = []
    solids: list[Obj] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, r1, c1 = map(int, b)
        sub = np.array(g[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)

        if _is_frame(sub, bg):
            frames.append((r0, c0, sub))
        elif _is_solid(sub, bg):
            solids.append((r0, c0, sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, frames, solids


def _solve(
    grid: np.ndarray,
    *,
    separator: bool,
    swap_colors: bool,
    reverse_right: bool,
) -> np.ndarray:
    bg, frames, solids = _extract_objects(grid)
    n_slots = max(len(frames), len(solids))
    if n_slots == 0:
        return np.zeros((0, 9 if separator else 8), dtype=int)

    out_h = 4 * int(n_slots)
    out_w = 9 if separator else 8
    left_c0 = 0
    right_c0 = 5 if separator else 4
    out = np.full((out_h, out_w), int(bg), dtype=int)

    left_slots: list[Obj | None] = [None] * n_slots
    right_slots: list[Obj | None] = [None] * n_slots

    for i, obj in enumerate(frames):
        left_slots[int(i)] = obj

    if reverse_right:
        for i, obj in enumerate(solids):
            right_slots[n_slots - 1 - int(i)] = obj
    else:
        for i, obj in enumerate(solids):
            right_slots[int(i)] = obj

    for slot in range(n_slots):
        row0 = 4 * int(slot)
        left_obj = left_slots[slot]
        right_obj = right_slots[slot]

        left_patch = None if left_obj is None else left_obj[2]
        right_patch = None if right_obj is None else right_obj[2]

        if swap_colors and left_obj is not None and right_obj is not None:
            left_color = _obj_color(left_obj[2], bg)
            right_color = _obj_color(right_obj[2], bg)
            left_patch = _recolor(left_obj[2], bg, right_color)
            right_patch = _recolor(right_obj[2], bg, left_color)

        if left_patch is not None:
            paste(out, left_patch, row0, left_c0)
        if right_patch is not None:
            paste(out, right_patch, row0, right_c0)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects into two separated columns with asymmetric stacking.

    - Frames are stacked on the left from top to bottom.
    - Solids are stacked on the right from bottom to top.
    - The columns are separated by one background column.
    - After slot assignment, objects on the same output row slot swap colors.
    """
    return _solve(grid, separator=True, swap_colors=True, reverse_right=True)


def _make_frame(color: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(color), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(color: int) -> np.ndarray:
    return np.full((4, 4), int(color), dtype=int)


def _row_values(objs: list[Obj]) -> list[int]:
    return [int(r0) for r0, _c0, _sub in objs]


def _has_distinct_rows(objs: list[Obj]) -> bool:
    rows = _row_values(objs)
    return len(rows) == len(set(rows))


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n_pairs = int(np.random.randint(2, 5))
    H = int(np.random.randint(22, 31))
    W = int(np.random.randint(22, 31))
    g = np.full((H, W), int(bg), dtype=int)

    colors = np.random.choice(np.arange(1, 10), size=2 * n_pairs, replace=False)
    frame_colors = list(map(int, colors[:n_pairs]))
    solid_colors = list(map(int, colors[n_pairs:]))

    try:
        for color in frame_colors:
            patch = _make_frame(color, bg)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        for color in solid_colors:
            patch = _make_solid(color)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)
    except Exception:
        return None

    _bg, frames, solids = _extract_objects(g)
    if len(frames) != n_pairs or len(solids) != n_pairs:
        return None
    if not _has_distinct_rows(frames) or not _has_distinct_rows(solids):
        return None

    fused = main(g)
    no_swap = _solve(g, separator=True, swap_colors=False, reverse_right=True)
    no_reverse = _solve(g, separator=True, swap_colors=True, reverse_right=False)
    if np.array_equal(fused, no_swap) or np.array_equal(fused, no_reverse):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
