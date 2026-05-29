# Combo variant of 221dfab4 (index 14)
# combo_spec: D3/0+D12/0
# Name: YG交替+线段颜色编码条纹颜色
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the combo variant D3/0 + D12/0
# ---------------------------------------------------
# a) Concrete behavior and trace
#    Geometry stays the same as 221dfab4: a single border segment provides
#    orientation, span, and marching direction; stripes advance every 2 cells and
#    the in-between cells are reset to background inside the source span.
#    What changes is the stripe palette logic:
#    - Original task: the source must be yellow, and stripe colors repeat
#      yellow, yellow, green.
#    - D12/0 alone: the source color becomes semantic, so a blue segment might mean
#      blue, blue, red.
#    - This combo: keep the source-color semantics, but simplify the cycle to a
#      strict two-step alternation source, partner, source, partner, ...
#      Example: top-edge source color 1 spanning columns 5..7.
#        Original rows 0,2,4,6 -> 4,4,3,4
#        D12-only rows 0,2,4,6 -> 1,1,2,1
#        Combo rows 0,2,4,6 -> 1,2,1,2
#      Partner stripes still preserve the original "flood" behavior: row 2 and row
#      6 recolor all non-background pixels across the full row to the partner color,
#      while source-color stripes only paint inside columns 5..7.
# b) Complexity calibration
#    The mapping is intentionally small and categorical:
#      blue<->red, pink<->orange, teal<->maroon.
#    That gives a human a few reusable pairings to notice across examples without
#    exploding the hypothesis space. Interior objects use only neutral colors 0/5 so
#    both stripe colors stand out. The generator rejects weak samples unless partner
#    stripes visibly recolor off-span object cells and unless the combo output differs
#    from both the original yellow/green logic and the D12-only three-cycle logic.
# c) Hidden assumptions audit against the original solver
#    - `smask = (g == 4) & border` is no longer valid. We must detect a border
#      segment of any allowed semantic color.
#    - `Y = 4`, `G = 3`, and `cycle = [4,4,3]` no longer hold. They become
#      `source_color`, `partner_color`, and an alternating 2-cycle.
#    - The branch "green stripes recolor the whole row/column" remains valid, but
#      is now keyed off `partner_color`, not color id 3.
#    - Counting foreground as `(canvas != bg) & (canvas != 4)` is invalid because
#      the source is not fixed yellow. The generator must explicitly track source
#      cells with a mask.
#    - The solver still assumes the chosen border component is a single straight,
#      contiguous, monochrome segment; the generator enforces that and keeps blobs
#      one cell away from borders so no competing border objects appear.
# d) Placement legality
#    The generator uses `random_free_location(..., border=1, padding=1)` so blobs do
#    not overlap or merge. The source segment never touches corners. Interior blobs
#    stay away from the source border row/column, which keeps the border cue unique.
#    Additional rejection checks ensure at least several partner stripes intersect
#    off-span foreground so the flood behavior is actually present.
# e) Visual observability
#    From a few train pairs, a human should be able to see two coupled changes:
#    the source is no longer always yellow, and every other stripe is now the mapped
#    partner color rather than every third stripe. Because generated sources never
#    use yellow/green and because multiple partner stripes visibly repaint interior
#    blobs, the combo cannot silently look like the original task.
#
# Post-change assumption audit
#    - `_detect_source` assumes exactly one usable border segment with a color from
#      `SOURCE_TO_PARTNER`; generator construction guarantees this.
#    - `_apply_pattern` assumes marching by 2 cells from that border reaches all
#      stripe positions and that gap cells can be overwritten safely inside the span;
#      this matches the task geometry.
#    - The partner-flood metric assumes at least some off-span foreground lies on
#      partner rows/columns; generator rejection enforces visibility.
#    - The "difference vs D12-only three-cycle" metric assumes enough stripe depth
#      that alternating and every-third-partner behaviors diverge; the chosen board
#      sizes guarantee several stripe steps.


SOURCE_TO_PARTNER = {
    1: 2,  # blue -> red
    2: 1,  # red -> blue
    6: 7,  # pink -> orange
    7: 6,  # orange -> pink
    8: 9,  # teal -> maroon
    9: 8,  # maroon -> teal
}
SOURCE_COLORS = tuple(SOURCE_TO_PARTNER.keys())
NEUTRALS = (0, 5)


def _detect_source(g: np.ndarray, bg: int | None = None) -> dict | None:
    g = np.array(g, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g) if bg is None else bg)

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = border & (g != bg)
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best = None
    best_size = -1

    for cid in range(1, int(n) + 1):
        mask = lab == int(cid)
        pts = np.argwhere(mask)
        if pts.size == 0:
            continue

        colors = np.unique(g[mask])
        if len(colors) != 1:
            continue
        source_color = int(colors[0])
        if source_color not in SOURCE_TO_PARTNER:
            continue

        rs = pts[:, 0]
        cs = pts[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())

        if r0 == r1:
            span = list(range(c0, c1 + 1))
            if len(span) != int(mask[r0, :].sum()):
                continue
            spec = {
                "orient": "h",
                "start_pos": r0,
                "span": span,
                "dir1": 1 if r0 == 0 else -1,
                "limit": H,
            }
        elif c0 == c1:
            span = list(range(r0, r1 + 1))
            if len(span) != int(mask[:, c0].sum()):
                continue
            spec = {
                "orient": "v",
                "start_pos": c0,
                "span": span,
                "dir1": 1 if c0 == 0 else -1,
                "limit": W,
            }
        else:
            continue

        size = int(len(pts))
        if size > best_size:
            best_size = size
            spec["source_color"] = source_color
            spec["partner_color"] = int(SOURCE_TO_PARTNER[source_color])
            best = spec

    return best


def _stripe_steps(spec: dict, mode: str = "combo") -> list[tuple[int, int, int]]:
    steps: list[tuple[int, int, int]] = []
    main_color = int(spec["source_color"])
    partner_color = int(spec["partner_color"])
    step = 0

    while True:
        pos = int(spec["start_pos"]) + int(spec["dir1"]) * (2 * step)
        if not (0 <= pos < int(spec["limit"])):
            break

        if mode == "combo":
            color = partner_color if step % 2 == 1 else main_color
        elif mode == "d12_three_cycle":
            color = partner_color if step % 3 == 2 else main_color
        elif mode == "original":
            color = 3 if step % 3 == 2 else 4
        else:
            raise ValueError(f"Unknown stripe mode: {mode}")

        steps.append((step, int(pos), int(color)))
        step += 1

    return steps


def _apply_pattern(grid: np.ndarray, spec: dict, bg: int, mode: str = "combo") -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    orient = str(spec["orient"])
    span = np.array(spec["span"], dtype=int)
    partner_color = int(spec["partner_color"])
    H, W = out.shape

    for step, pos, color in _stripe_steps(spec, mode=mode):
        color = int(color)
        if orient == "h":
            if mode != "original" and color == partner_color:
                out[pos, :] = np.where(out[pos, :] != bg, partner_color, out[pos, :])
            elif mode == "original" and color == 3:
                out[pos, :] = np.where(out[pos, :] != bg, 3, out[pos, :])

            out[pos, span] = color

            gap = int(spec["start_pos"]) + int(spec["dir1"]) * (2 * step + 1)
            if 0 <= gap < H:
                out[gap, span] = bg
        else:
            if mode != "original" and color == partner_color:
                out[:, pos] = np.where(out[:, pos] != bg, partner_color, out[:, pos])
            elif mode == "original" and color == 3:
                out[:, pos] = np.where(out[:, pos] != bg, 3, out[:, pos])

            out[span, pos] = color

            gap = int(spec["start_pos"]) + int(spec["dir1"]) * (2 * step + 1)
            if 0 <= gap < W:
                out[span, gap] = bg

    return out


def _original_like_output(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = border & (g == 4)
    if not bool(smask.any()):
        return g.copy()

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return g.copy()

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))
    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        spec = {
            "orient": "h",
            "start_pos": r0,
            "span": list(range(c0, c1 + 1)),
            "dir1": 1 if r0 == 0 else -1,
            "limit": H,
            "source_color": 4,
            "partner_color": 3,
        }
    elif c0 == c1:
        spec = {
            "orient": "v",
            "start_pos": c0,
            "span": list(range(r0, r1 + 1)),
            "dir1": 1 if c0 == 0 else -1,
            "limit": W,
            "source_color": 4,
            "partner_color": 3,
        }
    else:
        return g.copy()

    return _apply_pattern(g, spec, bg, mode="original")


def _behavior_metrics(inp: np.ndarray) -> dict[str, int] | None:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    spec = _detect_source(g, bg)
    if spec is None:
        return None

    out_combo = _apply_pattern(g, spec, bg, mode="combo")
    out_d12 = _apply_pattern(g, spec, bg, mode="d12_three_cycle")
    out_orig = _original_like_output(g, bg)
    partner = int(spec["partner_color"])
    span = set(int(x) for x in spec["span"])

    partner_steps = 0
    visible_lines = 0
    visible_cells = 0

    for _step, pos, color in _stripe_steps(spec, mode="combo"):
        if int(color) != partner:
            continue
        partner_steps += 1

        if spec["orient"] == "h":
            keep = np.ones(g.shape[1], dtype=bool)
            for c in span:
                keep[int(c)] = False
            hits = keep & (g[pos, :] != bg) & (out_combo[pos, :] == partner)
        else:
            keep = np.ones(g.shape[0], dtype=bool)
            for r in span:
                keep[int(r)] = False
            hits = keep & (g[:, pos] != bg) & (out_combo[:, pos] == partner)

        count = int(hits.sum())
        if count > 0:
            visible_lines += 1
            visible_cells += count

    return {
        "partner_steps": int(partner_steps),
        "visible_partner_lines": int(visible_lines),
        "visible_partner_cells": int(visible_cells),
        "delta_vs_d12_three_cycle": int(np.sum(out_combo != out_d12)),
        "delta_vs_original": int(np.sum(out_combo != out_orig)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate stripes from a border segment toward the opposite side.

    Combo rule:
    - Detect the unique semantic border segment; its color is the primary stripe color.
    - Infer a partner color from `SOURCE_TO_PARTNER`.
    - March inward in steps of 2 cells, alternating source, partner, source, partner.
    - Source stripes paint only within the segment span.
    - Partner stripes also recolor every non-background pixel on that full row/column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    spec = _detect_source(g, bg)
    if spec is None:
        return g.copy()
    return _apply_pattern(g, spec, bg, mode="combo")


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    source_color = int(rng.choice(SOURCE_COLORS))
    bg, fg = rng.sample(list(NEUTRALS), 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 8)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        edge_row = 0 if edge == "top" else H - 1
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1

    canvas = np.full((H, W), bg, dtype=int)
    source_mask = np.zeros((H, W), dtype=bool)

    if orient == "h":
        for c in span:
            canvas[edge_row, c] = source_color
            source_mask[edge_row, c] = True
        forbidden_rows = {edge_row}
        forbidden_cols: set[int] = set()
    else:
        for r in span:
            canvas[r, edge_col] = source_color
            source_mask[r, edge_col] = True
        forbidden_rows = set()
        forbidden_cols = {edge_col}

    target_blobs = rng.randint(7, 10)
    placed = 0

    for _ in range(target_blobs * 7):
        if placed >= target_blobs:
            break

        bh = rng.randint(2, 10)
        bw = rng.randint(2, 10)
        density = rng.uniform(0.4, 0.85)

        try:
            sprite = random_sprite(
                bh,
                bw,
                density=density,
                symmetry="not_symmetric",
                color_palette=[fg],
                connectivity=4,
                background=bg,
            )
        except (AssertionError, RecursionError):
            continue

        mask = sprite != bg
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[r0 : r1 + 1, c0 : c1 + 1]

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != bg))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        sprite_mask = sprite != bg
        blob_rows = {int(pr + dr) for dr in range(sh) if sprite_mask[dr].any()}
        blob_cols = {int(pc + dc) for dc in range(sw) if sprite_mask[:, dc].any()}

        if blob_rows & forbidden_rows:
            continue
        if blob_cols & forbidden_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

        placed += 1

    fg_mask = (canvas != bg) & (~source_mask)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    bg2 = int(get_bg(inp))
    if bg2 != bg:
        return None

    spec = _detect_source(inp, bg)
    if spec is None:
        return None

    out = main(inp)
    if not np.array_equal(out, _apply_pattern(inp, spec, bg, mode="combo")):
        return None

    metrics = _behavior_metrics(inp)
    if metrics is None:
        return None

    if metrics["partner_steps"] < 4:
        return None
    if metrics["visible_partner_lines"] < 3:
        return None
    if metrics["visible_partner_cells"] < 10:
        return None
    if metrics["delta_vs_d12_three_cycle"] < max(8, len(spec["span"]) + 2):
        return None
    if metrics["delta_vs_original"] < max(12, len(spec["span"]) * 2):
        return None

    return inp


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
