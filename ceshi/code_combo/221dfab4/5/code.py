# Combo variant of 221dfab4 (index 5)
# combo_spec: D3/0+D4/0+D5/1
# Name: YG交替+取消flood+间隙第三色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_UTILS_DIR = _ROOT / "arc2_solver_select"
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


Y = 4
R = 2
G = 3


def _find_segment(g: np.ndarray, yellow: int = Y) -> tuple[str, int, list[int], int] | None:
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(yellow)) & border
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
        cols = sorted(int(c) for c in cs)
        return "h", int(r0), list(range(int(cols[0]), int(cols[-1]) + 1)), H
    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        return "v", int(c0), list(range(int(rows[0]), int(rows[-1]) + 1)), W
    return None


def _trajectory_positions(start_pos: int, limit: int, direction: int) -> tuple[list[int], list[int]]:
    stripes: list[int] = []
    gaps: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(direction) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        stripes.append(int(pos))
        gap = int(start_pos) + int(direction) * (2 * int(step) + 1)
        if 0 <= gap < int(limit):
            gaps.append(int(gap))
        step += 1
    return stripes, gaps


def _apply_pattern(
    grid: np.ndarray,
    *,
    cycle: list[int],
    flood_green: bool,
    gap_color: int | None,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    segment = _find_segment(g, Y)
    if segment is None:
        return out

    orient, start_pos, span, limit = segment
    direction = 1 if start_pos == 0 else -1

    step = 0
    while True:
        pos = int(start_pos) + int(direction) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        stripe_color = int(cycle[int(step) % len(cycle)])

        if orient == "h":
            if stripe_color == int(G) and bool(flood_green):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            for c in span:
                out[pos, int(c)] = stripe_color

            gap = int(start_pos) + int(direction) * (2 * int(step) + 1)
            if gap_color is not None and 0 <= gap < out.shape[0]:
                for c in span:
                    out[int(gap), int(c)] = int(gap_color)
        else:
            if stripe_color == int(G) and bool(flood_green):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            for r in span:
                out[int(r), pos] = stripe_color

            gap = int(start_pos) + int(direction) * (2 * int(step) + 1)
            if gap_color is not None and 0 <= gap < out.shape[1]:
                for r in span:
                    out[int(r), int(gap)] = int(gap_color)

        step += 1

    return out


def _stripe_mask(
    shape: tuple[int, int],
    orient: str,
    span: list[int],
    stripes: list[int],
) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if orient == "h":
        for pos in stripes:
            mask[int(pos), span] = True
    else:
        for pos in stripes:
            mask[span, int(pos)] = True
    return mask


def _gap_mask(
    shape: tuple[int, int],
    orient: str,
    span: list[int],
    gaps: list[int],
) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if orient == "h":
        for pos in gaps:
            mask[int(pos), span] = True
    else:
        for pos in gaps:
            mask[span, int(pos)] = True
    return mask


def _green_flood_witness_mask(
    shape: tuple[int, int],
    orient: str,
    span: list[int],
    green_positions: list[int],
) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if orient == "h":
        cols_outside = [c for c in range(W) if c not in span]
        for pos in green_positions:
            mask[int(pos), cols_outside] = True
    else:
        rows_outside = [r for r in range(H) if r not in span]
        for pos in green_positions:
            mask[rows_outside, int(pos)] = True
    return mask


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    segment = _find_segment(g, Y)
    if segment is None:
        return {
            "red_gap_cells": 0,
            "gap_area": 0,
            "cycle_diff_cells": 0,
            "noflood_diff_cells": 0,
            "combo_vs_original": 0,
        }

    orient, start_pos, span, limit = segment
    direction = 1 if start_pos == 0 else -1
    stripes, gaps = _trajectory_positions(start_pos, limit, direction)

    combo = _apply_pattern(g, cycle=[Y, G], flood_green=False, gap_color=R)
    d3_baseline = _apply_pattern(g, cycle=[Y, Y, G], flood_green=False, gap_color=R)
    d4_baseline = _apply_pattern(g, cycle=[Y, G], flood_green=True, gap_color=R)
    original = _apply_pattern(g, cycle=[Y, Y, G], flood_green=True, gap_color=get_bg(g))

    stripe_mask = _stripe_mask(g.shape, orient, span, stripes)
    gap_mask = _gap_mask(g.shape, orient, span, gaps)
    combo_green_positions = [int(pos) for idx, pos in enumerate(stripes) if idx % 2 == 1]
    outside_green_mask = _green_flood_witness_mask(g.shape, orient, span, combo_green_positions)

    return {
        "red_gap_cells": int(np.sum(combo[gap_mask] == int(R))),
        "gap_area": int(np.sum(gap_mask)),
        "cycle_diff_cells": int(np.sum((combo != d3_baseline) & stripe_mask)),
        "noflood_diff_cells": int(
            np.sum((combo != d4_baseline) & outside_green_mask & (d4_baseline == int(G)))
        ),
        "combo_vs_original": int(np.sum(combo != original)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo analysis for D3/0 + D4/0 + D5/1.

    Concrete behavior trace on a top-edge segment spanning cols 5..7:
    - Original rows over the span were:
      row 0 = YYY, row 1 = bgbgbg, row 2 = YYY, row 3 = bgbgbg, row 4 = GGG, ...
      and the green row also recolored every non-bg pixel on the full row.
    - This combo changes all three moving parts together:
      row 0 = YYY, row 1 = RRR, row 2 = GGG, row 3 = RRR, row 4 = YYY, row 5 = RRR, ...
      Green is now only painted inside the original segment span, so cells outside cols 5..7
      keep their old colors even on green rows.
    - The visible result is a neat column of alternating yellow / red / green / red stripes.

    Complexity calibration:
    - The rule would become visually weak if red already existed in the input, so the generator
      excludes 2/3/4 from input colors.
    - To keep the D4 change observable, the generator forces at least one foreground blob to cross
      a combo-green row/column outside the span. That creates a direct "old solver would flood this,
      combo solver does not" witness.
    - The grid remains large enough that multiple stripe/gap repetitions appear, so Y-G alternation
      is supported by repeated evidence rather than a single ambiguous local pattern.

    Implicit assumption audit against the original solver:
    - Segment parsing still assumes one straight yellow border component; the generator keeps the
      marker contiguous and away from corners.
    - The old code relied on green flood using `out != bg`; that assumption is intentionally removed
      here because green no longer needs whole-row/column stateful recoloring.
    - Gap writing used to assume "erase to bg is harmless". That is false in this combo because gap
      cells carry red information, so they must be explicitly painted.
    - Stripe stepping still assumes every active line is at offset 0,2,4,... from the border line.
      The generator respects that geometry and never introduces competing border markers.

    Placement legality and new assumptions:
    - All blobs stay interior, non-overlapping, and one cell apart.
    - The no-flood witness blob is also kept interior and disjoint, so solver behavior does not
      depend on accidental collisions.
    - New solver assumption: at least one in-grid gap and at least one combo-green stripe exist if
      the sample is meant to teach the rule. The generator enforces that with metric checks.

    Visual observability:
    - A human sees three repeating colors inside the same narrow span: yellow stripe, red separator,
      green stripe, red separator, ...
    - The green rows/cols are visibly "quiet" outside the span, which contrasts with the original
      task's flood behavior when foreground blobs happen to touch those lines.
    """
    return _apply_pattern(np.array(grid, dtype=int), cycle=[Y, G], flood_green=False, gap_color=R)


def _can_place_rect(
    canvas: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    *,
    bg: int,
    border: int = 1,
    padding: int = 1,
) -> bool:
    H, W = canvas.shape
    if top < border or left < border:
        return False
    if top + height > H - border or left + width > W - border:
        return False

    r0 = max(0, int(top) - int(padding))
    c0 = max(0, int(left) - int(padding))
    r1 = min(H, int(top) + int(height) + int(padding))
    c1 = min(W, int(left) + int(width) + int(padding))
    return bool(np.all(canvas[r0:r1, c0:c1] == int(bg)))


def _place_rect(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[int(top) : int(top) + int(height), int(left) : int(left) + int(width)] = int(color)


def _place_noflood_witness(
    canvas: np.ndarray,
    *,
    orient: str,
    span: list[int],
    green_positions: list[int],
    fg: int,
    bg: int,
) -> bool:
    H, W = canvas.shape
    if orient == "h":
        left_options: list[int] = []
        if span[0] >= 3:
            left_options.append(int(span[0]) - 2)
        if span[-1] + 2 <= W - 2:
            left_options.append(int(span[-1]) + 1)
        for row in green_positions:
            top_options = [int(row), int(row) - 1]
            for top in top_options:
                for left in left_options:
                    if _can_place_rect(canvas, top, left, 2, 2, bg=bg):
                        _place_rect(canvas, top, left, 2, 2, fg)
                        return True
    else:
        top_options_span: list[int] = []
        if span[0] >= 3:
            top_options_span.append(int(span[0]) - 2)
        if span[-1] + 2 <= H - 2:
            top_options_span.append(int(span[-1]) + 1)
        for col in green_positions:
            left_options = [int(col), int(col) - 1]
            for top in top_options_span:
                for left in left_options:
                    if _can_place_rect(canvas, top, left, 2, 2, bg=bg):
                        _place_rect(canvas, top, left, 2, 2, fg)
                        return True
    return False


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(24, 30)
    W = rng.randint(24, 30)

    available = [c for c in range(10) if c not in (R, G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(4, 8)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        edge_row = 0 if edge == "top" else H - 1
        start_pos = int(edge_row)
        limit = H
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1
        start_pos = int(edge_col)
        limit = W

    direction = 1 if start_pos == 0 else -1
    stripes, gaps = _trajectory_positions(start_pos, limit, direction)
    green_positions = [int(pos) for idx, pos in enumerate(stripes) if idx % 2 == 1]

    if not gaps or not green_positions:
        return None

    canvas = np.full((H, W), bg, dtype=int)
    if orient == "h":
        canvas[int(start_pos), span] = int(Y)
        forbidden_rows = {int(start_pos)}
        forbidden_cols: set[int] = set()
    else:
        canvas[span, int(start_pos)] = int(Y)
        forbidden_rows = set()
        forbidden_cols = {int(start_pos)}

    if not _place_noflood_witness(
        canvas,
        orient=orient,
        span=span,
        green_positions=green_positions,
        fg=fg,
        bg=bg,
    ):
        return None

    n_blobs = rng.randint(5, 9)
    for _ in range(n_blobs):
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

        if orient == "h" and blob_rows & forbidden_rows:
            continue
        if orient == "v" and blob_cols & forbidden_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    fg_mask = (canvas != bg) & (canvas != Y)
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
        return None

    metrics = _behavior_metrics(inp)
    if metrics["red_gap_cells"] != metrics["gap_area"]:
        return None
    if metrics["cycle_diff_cells"] <= 0:
        return None
    if metrics["noflood_diff_cells"] <= 0:
        return None
    if metrics["combo_vs_original"] <= metrics["gap_area"]:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
