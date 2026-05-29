# Variant of 221dfab4
# Dimension: D4 — 绿色条纹的全行/列重着色范围 (trajectory)
# Change: 绿色条纹将全行/列的所有非背景像素重着色为绿色（全局 flood 效果）
#         → 黄色条纹执行全局 flood，绿色条纹仅限跨度
#
# Design analysis for this variant
# - Concrete behavior:
#   The edge yellow segment still launches stripes every 2 cells with cycle Y,Y,G.
#   The geometric trajectory is unchanged. The only semantic inversion is which stripe
#   owns the "full row/column recolor" privilege:
#   1) old task: yellow = span-only, green = full trajectory flood
#   2) new task: yellow = full trajectory flood, green = span-only
#   Example for a top-edge horizontal segment spanning cols 3..5:
#     input row 2 has a blob at cols 0..1, row 4 has a blob at cols 7..8.
#     old output: row 2 blob stays original color because Y only paints cols 3..5;
#                 row 4 blob turns green because G floods the whole row.
#     new output: row 2 blob turns yellow because Y floods the whole row;
#                 row 4 blob stays original color because G only paints cols 3..5.
#   This makes the D4 inversion visible specifically on off-span foreground pixels
#   that lie on stripe rows/cols.
# - Complexity calibration:
#   If all foreground happened to sit only inside the segment span, the new task would
#   look identical to the old one. The generator therefore forces multiple witness blobs
#   to intersect stripe rows/cols outside the span, for both yellow and green stripe
#   types, so the inversion is visually obvious and non-degenerate.
# - Implicit-assumption audit of the solver:
#   The original solver depended on "green owns flood" at exactly two branches
#   (`if col == G` for horizontal/vertical). Everything else only depends on the edge
#   segment being straight, on the Y,Y,G cadence, and on gap clearing staying limited
#   to the span. Those assumptions still hold, so only the flood condition flips to Y.
#   The overwrite order also remains safe: flood happens first, then the span stripe is
#   painted with the same stripe color, so there is no conflicting write.
# - Placement legality:
#   Witness blobs are axis-aligned rectangles placed strictly inside the board, outside
#   the segment span, with a 1-cell margin from existing non-background cells. Random
#   blobs are then added with `random_free_location`, so components do not overlap or
#   merge accidentally, and the solver still sees one clear yellow border segment.
# - Visual observability:
#   The generator rejects any sample whose metric reports zero yellow-flood evidence or
#   zero green-span evidence. Concretely, it requires:
#   1) at least one off-span foreground cell on a yellow stripe row/col recolors to 4
#   2) at least one off-span foreground cell on a green stripe row/col stays unchanged
#   This keeps degeneracy to the original behavior at 0% by construction.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg, label_mask, random_free_location, random_sprite
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "arc2_solver_select"))
    from arc_utils import get_bg, label_mask, random_free_location, random_sprite


Y = 4
G = 3


def _find_border_segment(g: np.ndarray) -> tuple[str, int, list[int], int, int] | None:
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(Y)) & border
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        start_pos = int(r0)
        cols = sorted(int(c) for c in cs)
        span = list(range(cols[0], cols[-1] + 1))
        dir1 = 1 if start_pos == 0 else -1
        return "h", start_pos, span, dir1, H

    if c0 == c1:
        start_pos = int(c0)
        rows = sorted(int(r) for r in rs)
        span = list(range(rows[0], rows[-1] + 1))
        dir1 = 1 if start_pos == 0 else -1
        return "v", start_pos, span, dir1, W

    return None


def _trajectory(start_pos: int, dir1: int, limit: int) -> list[tuple[int, int, int]]:
    cycle = [int(Y), int(Y), int(G)]
    out: list[tuple[int, int, int]] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        out.append((int(step), int(pos), int(cycle[int(step) % 3])))
        step += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate stripes from the yellow edge segment toward the opposite side.

    Variant rule:
    - Stripe colors still cycle Y, Y, G with a 1-cell bg gap between stripes.
    - Yellow stripes now own the full-row/full-column recolor effect.
    - Green stripes are now limited to the segment span only.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    seg = _find_border_segment(g)
    if seg is None:
        return out

    orient, start_pos, span, dir1, limit = seg

    for step, pos, col in _trajectory(start_pos, dir1, limit):
        if orient == "h":
            if col == int(Y):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(Y), out[pos, :])
            for c in span:
                out[pos, int(c)] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < H:
                for c in span:
                    out[int(gap), int(c)] = int(bg)
        else:
            if col == int(Y):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(Y), out[:, pos])
            for r in span:
                out[int(r), pos] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < W:
                for r in span:
                    out[int(r), int(gap)] = int(bg)

    return out


def _paint_rect(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)] = int(color)


def _region_is_clear(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, bg: int, padding: int = 1) -> bool:
    H, W = canvas.shape
    rr0 = max(0, int(r0) - int(padding))
    cc0 = max(0, int(c0) - int(padding))
    rr1 = min(H, int(r0) + int(h) + int(padding))
    cc1 = min(W, int(c0) + int(w) + int(padding))
    return bool(np.all(canvas[rr0:rr1, cc0:cc1] == int(bg)))


def _place_targeted_witness(
    canvas: np.ndarray,
    orient: str,
    target_pos: int,
    span: list[int],
    color: int,
    bg: int,
    rng: random.Random,
) -> bool:
    H, W = canvas.shape

    if orient == "h":
        h, w = 2, 4
        row_starts = [r for r in (target_pos - 1, target_pos) if 1 <= r <= H - h - 1]

        left_starts = list(range(1, span[0] - w + 1))
        right_starts = list(range(span[-1] + 1, W - w))

        side_lists = [left_starts, right_starts]
        rng.shuffle(side_lists)
        rng.shuffle(row_starts)

        for cols in side_lists:
            cols = cols[:]
            rng.shuffle(cols)
            for r0 in row_starts:
                for c0 in cols:
                    if _region_is_clear(canvas, r0, c0, h, w, bg, padding=1):
                        _paint_rect(canvas, r0, c0, h, w, color)
                        return True
        return False

    h, w = 4, 2
    col_starts = [c for c in (target_pos - 1, target_pos) if 1 <= c <= W - w - 1]

    top_starts = list(range(1, span[0] - h + 1))
    bottom_starts = list(range(span[-1] + 1, H - h))

    side_lists = [top_starts, bottom_starts]
    rng.shuffle(side_lists)
    rng.shuffle(col_starts)

    for rows in side_lists:
        rows = rows[:]
        rng.shuffle(rows)
        for c0 in col_starts:
            for r0 in rows:
                if _region_is_clear(canvas, r0, c0, h, w, bg, padding=1):
                    _paint_rect(canvas, r0, c0, h, w, color)
                    return True
    return False


def _variant_metric(inp: np.ndarray, out: np.ndarray, bg: int) -> tuple[int, int]:
    seg = _find_border_segment(inp)
    if seg is None:
        return 0, 0

    orient, start_pos, span, dir1, limit = seg
    yellow_flood_hits = 0
    green_span_hits = 0
    span_set = set(int(x) for x in span)

    for _step, pos, col in _trajectory(start_pos, dir1, limit):
        if orient == "h":
            off_span = [c for c in range(inp.shape[1]) if c not in span_set]
            if col == int(Y):
                yellow_flood_hits += sum(
                    int(inp[pos, c] != int(bg) and out[pos, c] == int(Y)) for c in off_span
                )
            else:
                green_span_hits += sum(
                    int(inp[pos, c] != int(bg) and out[pos, c] == inp[pos, c]) for c in off_span
                )
        else:
            off_span = [r for r in range(inp.shape[0]) if r not in span_set]
            if col == int(Y):
                yellow_flood_hits += sum(
                    int(inp[r, pos] != int(bg) and out[r, pos] == int(Y)) for r in off_span
                )
            else:
                green_span_hits += sum(
                    int(inp[r, pos] != int(bg) and out[r, pos] == inp[r, pos]) for r in off_span
                )

    return int(yellow_flood_hits), int(green_span_hits)


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (int(Y), int(G))]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 8)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        start_pos = 0 if edge == "top" else H - 1
        dir1 = 1 if edge == "top" else -1
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        start_pos = 0 if edge == "left" else W - 1
        dir1 = 1 if edge == "left" else -1

    grid = np.full((H, W), bg, dtype=int)
    if orient == "h":
        for c in span:
            grid[start_pos, c] = int(Y)
    else:
        for r in span:
            grid[r, start_pos] = int(Y)

    traj = _trajectory(start_pos, dir1, H if orient == "h" else W)
    yellow_positions = [pos for step, pos, col in traj if col == int(Y) and step > 0]
    green_positions = [pos for _step, pos, col in traj if col == int(G)]
    if len(yellow_positions) < 3 or len(green_positions) < 2:
        return None

    canvas = grid.copy()

    # Guaranteed witness blobs so the swapped flood role is visible and non-degenerate.
    witness_targets = [
        yellow_positions[0],
        green_positions[0],
        yellow_positions[1],
        green_positions[1],
        yellow_positions[2],
    ]
    for pos in witness_targets:
        if not _place_targeted_witness(canvas, orient, pos, span, fg, bg, rng):
            return None

    # Add a few random amorphous blobs to preserve the original task's look.
    for _ in range(rng.randint(2, 5)):
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

        pix = int(np.sum(sprite != bg))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        sh, sw = sprite.shape
        sprite_mask = sprite != bg
        blob_rows = set(int(pr + dr) for dr in range(sh) if sprite_mask[dr].any())
        blob_cols = set(int(pc + dc) for dc in range(sw) if sprite_mask[:, dc].any())

        # Keep the original edge segment isolated and obvious.
        if orient == "h" and start_pos in blob_rows:
            continue
        if orient == "v" and start_pos in blob_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    fg_mask = (canvas != bg) & (canvas != int(Y))
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
        return None

    yellow_hits, green_hits = _variant_metric(inp, out, bg)
    if yellow_hits <= 0 or green_hits <= 0:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
