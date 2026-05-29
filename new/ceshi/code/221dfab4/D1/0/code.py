# Variant of 221dfab4
# Dimension: D1 — 推进方向 (path)
# Change: 从黄色线段所在边缘沿直线向对侧边缘推进（straight_ray）改为沿 45 度对角线推进。

from __future__ import annotations

import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the D1 variant:
# a) Concretization + trace:
#    Original rule for a top-edge segment on cols 4..7 is:
#    rows 0,2,4,... reuse the same cols 4..7, with rows 1,3,5,... cleared to bg.
#    Variant keeps the same Y,Y,G cycle and the same "green recolors whole row/col"
#    behavior, but each stripe anchor moves diagonally.
#    Example for a top-edge segment on cols 4..7:
#      original stripes: (row 0, cols 4..7), (row 2, cols 4..7), (row 4, cols 4..7)
#      variant stripes:  (row 0, cols 4..7), (row 2, cols 6..9), (row 4, cols 8..11)
#    The gap segments also move diagonally:
#      gap at row 1 cols 5..8, then row 3 cols 7..10, ...
#    So the only conceptual change is the propagation path: the copied segment keeps
#    its horizontal/vertical shape, but its anchor advances by (+/-1, +/-1) each unit.
# b) Complexity calibration:
#    If the generator allowed only one or two stripes, the task would visually collapse
#    back into "a source segment plus noise". The generator therefore guarantees at least
#    six visible stripes, so the diagonal drift is obvious across several Y/Y/G cycles.
# c) Hidden assumptions in the original solver:
#    - `pos` used to be a single row or col index; that relied on zero lateral drift.
#      The variant breaks that, so every stripe now needs both a primary coordinate
#      (row/col) and a shifted span.
#    - `span` used to be reused verbatim at every step; now it must be translated by
#      the diagonal column/row offset before painting or clearing the gap.
#    - Green recoloring of a whole row/col still works, but only after the diagonally
#      shifted row/col has been identified correctly.
# d) Placement legality:
#    The generator picks the source segment so the entire stripe span stays in-bounds for
#    at least six stripe steps. Blobs are placed strictly inside the border and verified
#    via the solver oracle.
# e) Visual observability:
#    The quantitative signature of the variant is non-zero lateral drift:
#    `max(abs(shifted_span_start - source_span_start)) >= 10` in generated samples.
#    That makes the output form a staircase/ladder rather than the original straight stack.
# Additional post-change assumptions checked by the generator:
#    - the yellow source is a single straight border component;
#    - enough diagonal steps exist to show at least one green stripe;
#    - green recoloring affects interior foreground pixels, so the green-row/green-column
#      behavior remains visible instead of being a no-op.

Y = 4
G = 3
CYCLE = (Y, Y, G)
MIN_STRIPES = 6


def _find_source_segment(g: np.ndarray):
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == Y) & border
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
        edge = "top" if r0 == 0 else "bottom"
        return {
            "orient": "h",
            "fixed": int(r0),
            "span": list(range(int(c0), int(c1) + 1)),
            "dr": 1 if edge == "top" else -1,
            "dc": 1 if edge == "top" else -1,
            "edge": edge,
        }
    if c0 == c1:
        edge = "left" if c0 == 0 else "right"
        return {
            "orient": "v",
            "fixed": int(c0),
            "span": list(range(int(r0), int(r1) + 1)),
            "dr": 1 if edge == "left" else -1,
            "dc": 1 if edge == "left" else -1,
            "edge": edge,
        }
    return None


def _shifted(values: list[int], offset: int, direction: int) -> list[int]:
    return [int(v + direction * offset) for v in values]


def _stripe_cells(info: dict, shift: int):
    if info["orient"] == "h":
        row = int(info["fixed"] + info["dr"] * shift)
        cols = _shifted(info["span"], shift, info["dc"])
        return row, cols
    col = int(info["fixed"] + info["dc"] * shift)
    rows = _shifted(info["span"], shift, info["dr"])
    return col, rows


def _stripe_in_bounds(info: dict, shift: int, H: int, W: int) -> bool:
    if info["orient"] == "h":
        row, cols = _stripe_cells(info, shift)
        return 0 <= int(row) < H and all(0 <= int(c) < W for c in cols)
    col, rows = _stripe_cells(info, shift)
    return 0 <= int(col) < W and all(0 <= int(r) < H for r in rows)


def _count_stripes(info: dict, H: int, W: int) -> int:
    count = 0
    while _stripe_in_bounds(info, 2 * count, H, W):
        count += 1
    return int(count)


def _lateral_drift(info: dict, H: int, W: int) -> int:
    return max(0, 2 * (_count_stripes(info, H, W) - 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diagonal-path variant of 221dfab4.

    The source is still a yellow border segment, the color cycle is still Y,Y,G,
    yellow stripes still paint only the shifted span, and green stripes still recolor
    all non-background pixels on the touched row/column. The only changed mechanism is
    the path: every stripe and every gap segment shifts by one row and one column per
    unit step instead of moving along a single axis.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    info = _find_source_segment(g)
    if info is None:
        return out

    step = 0
    while True:
        shift = 2 * step
        if not _stripe_in_bounds(info, shift, H, W):
            break

        color = int(CYCLE[step % 3])

        if info["orient"] == "h":
            row, cols = _stripe_cells(info, shift)
            if color == G:
                out[int(row), :] = np.where(out[int(row), :] != bg, G, out[int(row), :])
            out[int(row), np.array(cols, dtype=int)] = color

            gap_row, gap_cols = _stripe_cells(info, shift + 1)
            if 0 <= int(gap_row) < H:
                valid_gap_cols = [int(c) for c in gap_cols if 0 <= int(c) < W]
                if valid_gap_cols:
                    out[int(gap_row), np.array(valid_gap_cols, dtype=int)] = bg
        else:
            col, rows = _stripe_cells(info, shift)
            if color == G:
                out[:, int(col)] = np.where(out[:, int(col)] != bg, G, out[:, int(col)])
            out[np.array(rows, dtype=int), int(col)] = color

            gap_col, gap_rows = _stripe_cells(info, shift + 1)
            if 0 <= int(gap_col) < W:
                valid_gap_rows = [int(r) for r in gap_rows if 0 <= int(r) < H]
                if valid_gap_rows:
                    out[np.array(valid_gap_rows, dtype=int), int(gap_col)] = bg

        step += 1

    return out


def _choose_segment(H: int, W: int, rng: random.Random):
    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(4, 8)
    runway = 2 * (MIN_STRIPES - 1)

    if edge == "top":
        upper = W - seg_len - runway
        if upper < 1:
            return None
        start = rng.randint(1, upper)
        return {"edge": edge, "orient": "h", "fixed": 0, "span": list(range(start, start + seg_len))}

    if edge == "bottom":
        lower = max(1, runway)
        upper = W - seg_len - 1
        if lower > upper:
            return None
        start = rng.randint(lower, upper)
        return {"edge": edge, "orient": "h", "fixed": H - 1, "span": list(range(start, start + seg_len))}

    if edge == "left":
        upper = H - seg_len - runway
        if upper < 1:
            return None
        start = rng.randint(1, upper)
        return {"edge": edge, "orient": "v", "fixed": 0, "span": list(range(start, start + seg_len))}

    lower = max(1, runway)
    upper = H - seg_len - 1
    if lower > upper:
        return None
    start = rng.randint(lower, upper)
    return {"edge": edge, "orient": "v", "fixed": W - 1, "span": list(range(start, start + seg_len))}


def _paste_segment(grid: np.ndarray, seg: dict):
    if seg["orient"] == "h":
        grid[int(seg["fixed"]), np.array(seg["span"], dtype=int)] = Y
    else:
        grid[np.array(seg["span"], dtype=int), int(seg["fixed"])] = Y


def _gen_attempt():
    rng = random.Random()
    H = rng.randint(26, 32)
    W = rng.randint(26, 32)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    seg = _choose_segment(H, W, rng)
    if seg is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    _paste_segment(grid, seg)

    if seg["orient"] == "h":
        forbidden_rows = {int(seg["fixed"])}
        forbidden_cols = set()
    else:
        forbidden_rows = set()
        forbidden_cols = {int(seg["fixed"])}

    canvas = grid.copy()
    target_blobs = rng.randint(6, 10)
    placed = 0

    for _ in range(target_blobs * 6):
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
                if int(sprite[dr, dc]) != bg:
                    canvas[pr + dr, pc + dc] = int(sprite[dr, dc])
        placed += 1
        if placed >= target_blobs:
            break

    fg_mask = (canvas != bg) & (canvas != Y)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if int(get_bg(canvas)) != bg:
        return None

    info = _find_source_segment(canvas)
    if info is None:
        return None
    stripe_count = _count_stripes(info, H, W)
    if stripe_count < MIN_STRIPES:
        return None
    if _lateral_drift(info, H, W) < 10:
        return None

    out = main(canvas)

    changed_to_green = int(np.sum((canvas != bg) & (canvas != Y) & (out == G)))
    new_yellow = int(np.sum((out == Y) & (canvas != Y)))
    if changed_to_green < 8:
        return None
    if new_yellow < len(info["span"]) * 2:
        return None

    if not np.array_equal(main(canvas), out):
        return None

    return canvas


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
