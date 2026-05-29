# Variant of 221dfab4
# Dimension: D12 — 符号编码——线段属性的语义 (symbol_instructions)
# Change: 黄色线段仅编码位置和跨度信息，颜色固定为黄(4)和绿(3)... → 线段颜色编码条纹颜色
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the D12 symbol-instructions variant
# -------------------------------------------------------
# a) Concrete behavior and trace
#    The border segment is still the only directional cue, but its color now carries
#    semantics. A blue source means the stripe cycle is blue, blue, red; a pink source
#    means pink, pink, orange; a teal source means teal, teal, maroon; a green source
#    means green, green, yellow. The third stripe keeps the original "flood" role:
#    it recolors all non-background pixels along the whole row/column, not just the
#    source span.
#    Small trace: if a top-edge source of color 1 spans columns 5..7, then the old
#    task would paint rows 0,2,4 as 4,4,3. This variant paints them as 1,1,2. Row 4
#    also repaints any off-span foreground pixels on that row to 2. The geometry is
#    unchanged; only the source color semantics change.
# b) Complexity calibration
#    Using several possible source colors makes the change visible across examples.
#    The generator restricts backgrounds/objects to neutral colors (0 and 5), so the
#    new semantic colors stand out clearly. It also rejects cases where the partner
#    flood stripes do not visibly recolor off-span object pixels.
# c) Hidden assumptions from the original solver that must change
#    - `smask = (g == 4) & border` assumed the source is always yellow. That is no
#      longer valid; now we must detect the unique non-background border segment first.
#    - `Y = 4`, `G = 3`, and `cycle = [4,4,3]` hard-coded the stripe palette. Those
#      constants are replaced by `source_color` and a mapped `partner_color`.
#    - Excluding color 4 in the generator (`canvas != 4`) only worked because the
#      source color was fixed. The variant must exclude the actual detected source
#      cells instead.
#    - Unconditional stripe painting is still safe because the source segment is the
#      only border non-background object, and gap cells remain on the alternating
#      parity positions just as in the original task.
# d) Placement legality
#    The generator keeps the source as one straight border segment that does not touch
#    corners, keeps blobs at least one cell away from the border, and uses
#    `random_free_location(..., padding=1)` to avoid overlap/merging between blobs.
#    Solver assumptions about a single straight border source are therefore enforced.
# e) Visual observability
#    A human looking at several input/output pairs should notice that the border color
#    itself becomes the local stripe color, while every third stripe switches to a
#    paired partner color with the same flood behavior. Because the source never uses
#    yellow here, samples cannot silently collapse back to the original 4,4,3 task.
#
# Post-change assumption audit
#    - `_detect_source` assumes exactly one straight, contiguous, monochrome border
#      segment whose color is in `SOURCE_TO_PARTNER`; the generator guarantees this.
#    - The main loop assumes marching by steps of two from the source border reaches
#      all stripe positions; this matches the construction exactly.
#    - Flood recoloring assumes interior blobs can intersect at least some partner
#      rows/cols outside the source span; generator rejection enforces that visibility.

SOURCE_TO_PARTNER = {
    1: 2,  # blue -> red
    3: 4,  # green -> yellow
    6: 7,  # pink -> orange
    8: 9,  # teal -> maroon
}
SOURCE_COLORS = tuple(SOURCE_TO_PARTNER.keys())
NEUTRALS = (0, 5)


def _detect_source(g: np.ndarray, bg: int | None = None):
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


def _stripe_steps(spec: dict) -> list[tuple[int, int, int]]:
    steps: list[tuple[int, int, int]] = []
    main_color = int(spec["source_color"])
    partner_color = int(spec["partner_color"])
    step = 0
    while True:
        pos = int(spec["start_pos"]) + int(spec["dir1"]) * (2 * step)
        if not (0 <= pos < int(spec["limit"])):
            break
        color = partner_color if step % 3 == 2 else main_color
        steps.append((step, int(pos), int(color)))
        step += 1
    return steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate stripes from a border segment toward the opposite side.

    Variant rule:
    - The border segment's own color is the primary stripe color.
    - Every third stripe uses a partner color inferred from the source color via
      `SOURCE_TO_PARTNER`.
    - Primary stripes only paint within the source span.
    - Partner stripes additionally recolor all non-background pixels on the full
      row/column, preserving the original flood behavior of 221dfab4.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    spec = _detect_source(g, bg)
    if spec is None:
        return out

    orient = str(spec["orient"])
    span = list(spec["span"])
    partner_color = int(spec["partner_color"])
    H, W = g.shape

    for step, pos, color in _stripe_steps(spec):
        if orient == "h":
            if color == partner_color:
                out[pos, :] = np.where(out[pos, :] != bg, partner_color, out[pos, :])
            for c in span:
                out[pos, int(c)] = int(color)

            gap = int(spec["start_pos"]) + int(spec["dir1"]) * (2 * step + 1)
            if 0 <= gap < H:
                for c in span:
                    out[int(gap), int(c)] = bg
        else:
            if color == partner_color:
                out[:, pos] = np.where(out[:, pos] != bg, partner_color, out[:, pos])
            for r in span:
                out[int(r), pos] = int(color)

            gap = int(spec["start_pos"]) + int(spec["dir1"]) * (2 * step + 1)
            if 0 <= gap < W:
                for r in span:
                    out[int(r), int(gap)] = bg

    return out


def _partner_flood_visibility(inp: np.ndarray, out: np.ndarray, spec: dict, bg: int) -> tuple[int, int]:
    span = set(int(x) for x in spec["span"])
    partner = int(spec["partner_color"])
    visible_lines = 0
    visible_cells = 0

    for _step, pos, color in _stripe_steps(spec):
        if color != partner:
            continue

        if spec["orient"] == "h":
            keep = np.ones(inp.shape[1], dtype=bool)
            for c in span:
                keep[int(c)] = False
            hits = keep & (inp[pos, :] != bg) & (out[pos, :] == partner)
        else:
            keep = np.ones(inp.shape[0], dtype=bool)
            for r in span:
                keep[int(r)] = False
            hits = keep & (inp[:, pos] != bg) & (out[:, pos] == partner)

        count = int(hits.sum())
        if count > 0:
            visible_lines += 1
            visible_cells += count

    return visible_lines, visible_cells


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    source_color = int(rng.choice(SOURCE_COLORS))
    partner_color = int(SOURCE_TO_PARTNER[source_color])
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

    target_blobs = rng.randint(6, 10)
    placed = 0

    for _ in range(target_blobs * 5):
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
    out = main(inp)
    spec = _detect_source(inp, bg)
    if spec is None:
        return None

    visible_lines, visible_cells = _partner_flood_visibility(inp, out, spec, bg)
    if visible_lines < 2 or visible_cells < 8:
        return None

    if int(get_bg(inp)) != int(bg):
        return None

    if not np.array_equal(main(inp), out):
        return None

    # Prevent silent collapse back to the original fixed yellow/green semantics.
    if spec["source_color"] == 4 or spec["partner_color"] == 3:
        return None

    return inp


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
