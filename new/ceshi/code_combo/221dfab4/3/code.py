# Combo variant of 221dfab4 (index 3)
# combo_spec: D4/1+D10/2
# Name: 黄色flood+绿色加宽
#
# Design analysis for this combo
# - Concrete behavior and trace:
#   Start from the original 221dfab4 rule: a yellow border segment launches stripes
#   every 2 cells with cadence Y, Y, G, and the interleaved gap is reset to bg only
#   inside the original segment span. Originally, yellow stripes are narrow span-only
#   marks, while green stripes are narrow but also flood the full row/column.
#   In this combo both affected stages are changed together:
#   1) D4/1 flips the flood privilege: yellow stripes recolor every non-bg cell on
#      their whole row/column to yellow, green stripes lose that privilege.
#   2) D10/2 widens only green local painting: green stripes now paint a span of
#      width 2 * segment_length by extending half a segment to each side, while
#      yellow stripes keep the original span width.
#   Small horizontal example: suppose a top-edge segment covers cols 8..11 (len 4).
#   The widened green span becomes cols 6..13. On yellow stripe row 2, a blob at
#   cols 1..3 is recolored to yellow even though it is far outside the segment span.
#   On green stripe row 4, a blob at cols 6..7 turns green because it lies in the
#   green shoulder, but a blob at cols 15..16 stays its original color because green
#   no longer floods the whole row. Compared to the original task, the "strong"
#   effect visibly moves from green to yellow, while green becomes wider but local.
# - Complexity calibration:
#   If the generator only placed blobs inside the yellow span, or only used green
#   rows without shoulder witnesses, the combo could look identical to a simpler
#   variant. To avoid that, the generator always creates three witness types:
#   1) off-green-span foreground on yellow stripe lines, so yellow flood is visible
#   2) shoulder foreground on green stripe lines, so widened green painting is visible
#   3) off-green-span foreground on green stripe lines, so the lack of green flood is
#      also visible. This keeps both dimensions active in the examples.
# - Implicit-assumption audit of main():
#   The original solver assumed one shared local span for both colors and assumed
#   `col == G` was the only branch that ever performed full-line recoloring. Both are
#   false here. The combo solver therefore tracks two spans (`yellow_*`, `green_*`)
#   and a parameterized flood owner. The write order remains "flood first, then local
#   stripe", which is still safe because both writes use the same stripe color on that
#   step. Gap clearing must still use only the yellow/original span; if it cleared the
#   widened green shoulders, the D10 effect would erase itself on the next gap row/col.
# - Placement legality:
#   The generator picks segment starts with enough margin for both widened shoulders
#   and off-green-span witness blobs. All targeted witnesses are axis-aligned interior
#   rectangles written only after `_region_is_clear(..., padding=1)` succeeds, so they
#   do not overlap or merge. Extra random blobs use `random_free_location`, and the
#   border segment is kept isolated on a single edge.
# - Visual observability:
#   A human comparing 2-4 samples should repeatedly see the same pattern:
#   yellow lines recolor distant objects across the full line, green lines are wider
#   than yellow but only locally green, and some objects on green lines remain their
#   original color when they sit beyond the widened band. The generator rejects any
#   sample lacking these witnesses, so the combo should not collapse back to D4-only,
#   D10-only, or the original rule.
# - New solver assumptions and generator guarantees:
#   `_find_segment()` assumes the yellow border seed is the dominant yellow border
#   component and is a straight segment; the generator creates exactly one such seed.
#   `_solve()` assumes the widened green span fits in-bounds; `_choose_segment()`
#   enforces the necessary margins. Witness metrics assume there are usable yellow and
#   green trajectory positions beyond the edge seed; the chosen board sizes and segment
#   lengths guarantee multiple Y and G steps.

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


def _find_segment(g: np.ndarray) -> dict[str, int] | None:
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
        seg_len = int(c1 - c0 + 1)
        extra_lo = seg_len // 2
        extra_hi = seg_len - extra_lo
        return {
            "orient": 0,  # horizontal
            "start_pos": int(r0),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": int(H),
            "yellow_lo": int(c0),
            "yellow_hi": int(c1),
            "green_lo": max(0, int(c0) - extra_lo),
            "green_hi": min(W - 1, int(c1) + extra_hi),
            "seg_len": int(seg_len),
        }

    if c0 == c1:
        seg_len = int(r1 - r0 + 1)
        extra_lo = seg_len // 2
        extra_hi = seg_len - extra_lo
        return {
            "orient": 1,  # vertical
            "start_pos": int(c0),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": int(W),
            "yellow_lo": int(r0),
            "yellow_hi": int(r1),
            "green_lo": max(0, int(r0) - extra_lo),
            "green_hi": min(H - 1, int(r1) + extra_hi),
            "seg_len": int(seg_len),
        }

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


def _solve(grid: np.ndarray, *, yellow_flood: bool, wide_green: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    info = _find_segment(g)
    if info is None:
        return out

    flood_owner = int(Y) if yellow_flood else int(G)

    for step, pos, col in _trajectory(int(info["start_pos"]), int(info["dir1"]), int(info["limit"])):
        lo = int(info["green_lo"]) if wide_green and col == int(G) else int(info["yellow_lo"])
        hi = int(info["green_hi"]) if wide_green and col == int(G) else int(info["yellow_hi"])

        if int(info["orient"]) == 0:
            if col == flood_owner:
                out[pos, :] = np.where(out[pos, :] != int(bg), int(col), out[pos, :])
            out[pos, lo : hi + 1] = int(col)

            gap = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < g.shape[0]:
                out[gap, int(info["yellow_lo"]) : int(info["yellow_hi"]) + 1] = int(bg)
        else:
            if col == flood_owner:
                out[:, pos] = np.where(out[:, pos] != int(bg), int(col), out[:, pos])
            out[lo : hi + 1, pos] = int(col)

            gap = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < g.shape[1]:
                out[int(info["yellow_lo"]) : int(info["yellow_hi"]) + 1, gap] = int(bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - cadence stays Y, Y, G with a 1-cell bg gap between stripes
    - yellow stripes keep the original narrow segment span but flood the full row/col
    - green stripes lose flood behavior but use a doubled local span
    """
    return _solve(grid, yellow_flood=True, wide_green=True)


def _paint_rect(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)] = int(color)


def _region_is_clear(
    canvas: np.ndarray, r0: int, c0: int, h: int, w: int, bg: int, *, padding: int = 1
) -> bool:
    H, W = canvas.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False
    rr0 = max(0, int(r0) - int(padding))
    cc0 = max(0, int(c0) - int(padding))
    rr1 = min(H, int(r0) + int(h) + int(padding))
    cc1 = min(W, int(c0) + int(w) + int(padding))
    return bool(np.all(canvas[rr0:rr1, cc0:cc1] == int(bg)))


def _shuffle_copy(rng: random.Random, values: list[int]) -> list[int]:
    out = values[:]
    rng.shuffle(out)
    return out


def _place_targeted_rect(
    canvas: np.ndarray,
    orient: int,
    target_pos: int,
    intervals: list[tuple[int, int]],
    color: int,
    bg: int,
    rng: random.Random,
    *,
    desired_long: int = 4,
) -> bool:
    H, W = canvas.shape

    valid_intervals = [(int(lo), int(hi)) for lo, hi in intervals if int(lo) <= int(hi)]
    if not valid_intervals:
        return False

    if orient == 0:
        row_starts = [r for r in (target_pos - 1, target_pos) if 1 <= r <= H - 3]
        row_starts = _shuffle_copy(rng, row_starts)
        for lo, hi in _shuffle_copy(rng, valid_intervals):
            max_w = int(hi) - int(lo) + 1
            widths = list(range(min(int(desired_long), max_w), 0, -1))
            for w in widths:
                starts = list(range(int(lo), int(hi) - int(w) + 2))
                starts = _shuffle_copy(rng, starts)
                for r0 in row_starts:
                    for c0 in starts:
                        if _region_is_clear(canvas, r0, c0, 2, int(w), bg, padding=1):
                            _paint_rect(canvas, r0, c0, 2, int(w), color)
                            return True
        return False

    col_starts = [c for c in (target_pos - 1, target_pos) if 1 <= c <= W - 3]
    col_starts = _shuffle_copy(rng, col_starts)
    for lo, hi in _shuffle_copy(rng, valid_intervals):
        max_h = int(hi) - int(lo) + 1
        heights = list(range(min(int(desired_long), max_h), 0, -1))
        for h in heights:
            starts = list(range(int(lo), int(hi) - int(h) + 2))
            starts = _shuffle_copy(rng, starts)
            for c0 in col_starts:
                for r0 in starts:
                    if _region_is_clear(canvas, r0, c0, int(h), 2, bg, padding=1):
                        _paint_rect(canvas, r0, c0, int(h), 2, color)
                        return True
    return False


def _outside_intervals(info: dict[str, int], extent_limit: int) -> list[tuple[int, int]]:
    return [
        (1, int(info["green_lo"]) - 1),
        (int(info["green_hi"]) + 1, int(extent_limit) - 2),
    ]


def _shoulder_intervals(info: dict[str, int]) -> list[tuple[int, int]]:
    return [
        (int(info["green_lo"]), int(info["yellow_lo"]) - 1),
        (int(info["yellow_hi"]) + 1, int(info["green_hi"])),
    ]


def _metric_bundle(inp: np.ndarray) -> dict[str, int]:
    info = _find_segment(inp)
    if info is None:
        return {
            "yellow_flood_hits": 0,
            "green_widen_hits": 0,
            "green_no_flood_hits": 0,
            "diff_vs_original": 0,
            "diff_vs_d4": 0,
            "diff_vs_d10": 0,
        }

    out = main(inp)
    bg = int(get_bg(inp))
    yellow_flood_hits = 0
    green_widen_hits = 0
    green_no_flood_hits = 0

    for _step, pos, col in _trajectory(int(info["start_pos"]), int(info["dir1"]), int(info["limit"])):
        if int(info["orient"]) == 0:
            if col == int(Y):
                for c in range(inp.shape[1]):
                    if int(info["yellow_lo"]) <= c <= int(info["yellow_hi"]):
                        continue
                    if inp[pos, c] != int(bg) and out[pos, c] == int(Y):
                        yellow_flood_hits += 1
            else:
                for c in range(inp.shape[1]):
                    if int(info["green_lo"]) <= c <= int(info["green_hi"]):
                        if not (int(info["yellow_lo"]) <= c <= int(info["yellow_hi"])):
                            if inp[pos, c] != int(bg) and out[pos, c] == int(G):
                                green_widen_hits += 1
                    else:
                        if inp[pos, c] != int(bg) and out[pos, c] == inp[pos, c]:
                            green_no_flood_hits += 1
        else:
            if col == int(Y):
                for r in range(inp.shape[0]):
                    if int(info["yellow_lo"]) <= r <= int(info["yellow_hi"]):
                        continue
                    if inp[r, pos] != int(bg) and out[r, pos] == int(Y):
                        yellow_flood_hits += 1
            else:
                for r in range(inp.shape[0]):
                    if int(info["green_lo"]) <= r <= int(info["green_hi"]):
                        if not (int(info["yellow_lo"]) <= r <= int(info["yellow_hi"])):
                            if inp[r, pos] != int(bg) and out[r, pos] == int(G):
                                green_widen_hits += 1
                    else:
                        if inp[r, pos] != int(bg) and out[r, pos] == inp[r, pos]:
                            green_no_flood_hits += 1

    original = _solve(inp, yellow_flood=False, wide_green=False)
    d4_only = _solve(inp, yellow_flood=True, wide_green=False)
    d10_only = _solve(inp, yellow_flood=False, wide_green=True)

    return {
        "yellow_flood_hits": int(yellow_flood_hits),
        "green_widen_hits": int(green_widen_hits),
        "green_no_flood_hits": int(green_no_flood_hits),
        "diff_vs_original": int(np.sum(out != original)),
        "diff_vs_d4": int(np.sum(out != d4_only)),
        "diff_vs_d10": int(np.sum(out != d10_only)),
    }


def _choose_segment(rng: random.Random, H: int, W: int) -> dict[str, object] | None:
    edge = rng.choice(["top", "bottom", "left", "right"])
    margin = 3

    if edge in ("top", "bottom"):
        max_len = min(8, W // 2)
        if max_len < 4:
            return None
        seg_len = rng.randint(4, max_len)
        extra_lo = seg_len // 2
        extra_hi = seg_len - extra_lo
        start_lo = max(1, margin + extra_lo)
        start_hi = min(W - 1 - seg_len, W - seg_len - extra_hi - margin)
        if start_hi < start_lo:
            return None
        seg_start = rng.randint(start_lo, start_hi)
        return {
            "orient": 0,
            "edge": edge,
            "start_pos": 0 if edge == "top" else H - 1,
            "yellow_span": list(range(seg_start, seg_start + seg_len)),
            "green_span": list(range(seg_start - extra_lo, seg_start + seg_len + extra_hi)),
        }

    max_len = min(8, H // 2)
    if max_len < 4:
        return None
    seg_len = rng.randint(4, max_len)
    extra_lo = seg_len // 2
    extra_hi = seg_len - extra_lo
    start_lo = max(1, margin + extra_lo)
    start_hi = min(H - 1 - seg_len, H - seg_len - extra_hi - margin)
    if start_hi < start_lo:
        return None
    seg_start = rng.randint(start_lo, start_hi)
    return {
        "orient": 1,
        "edge": edge,
        "start_pos": 0 if edge == "left" else W - 1,
        "yellow_span": list(range(seg_start, seg_start + seg_len)),
        "green_span": list(range(seg_start - extra_lo, seg_start + seg_len + extra_hi)),
    }


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (int(Y), int(G))]
    bg, fg = rng.sample(available, 2)

    seg = _choose_segment(rng, H, W)
    if seg is None:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    orient = int(seg["orient"])
    start_pos = int(seg["start_pos"])
    yellow_span = [int(x) for x in seg["yellow_span"]]
    green_span = [int(x) for x in seg["green_span"]]

    if orient == 0:
        for c in yellow_span:
            grid[start_pos, c] = int(Y)
        limit = H
    else:
        for r in yellow_span:
            grid[r, start_pos] = int(Y)
        limit = W

    info = _find_segment(grid)
    if info is None:
        return None

    traj = _trajectory(int(info["start_pos"]), int(info["dir1"]), int(info["limit"]))
    yellow_positions = [pos for step, pos, col in traj if col == int(Y) and step > 0]
    green_positions = [pos for _step, pos, col in traj if col == int(G)]
    if len(yellow_positions) < 2 or len(green_positions) < 2:
        return None

    canvas = grid.copy()
    extent_limit = W if orient == 0 else H

    witness_specs = [
        ("yellow_flood", yellow_positions[0], _outside_intervals(info, extent_limit), 4),
        ("green_shoulder", green_positions[0], _shoulder_intervals(info), 3),
        ("green_no_flood", green_positions[1], _outside_intervals(info, extent_limit), 4),
        ("yellow_flood", yellow_positions[1], _outside_intervals(info, extent_limit), 4),
        ("green_shoulder", green_positions[0], _shoulder_intervals(info), 2),
    ]

    for _kind, target_pos, intervals, desired_long in witness_specs:
        if not _place_targeted_rect(
            canvas,
            orient,
            int(target_pos),
            intervals,
            int(fg),
            int(bg),
            rng,
            desired_long=int(desired_long),
        ):
            return None

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
                color_palette=[int(fg)],
                connectivity=4,
                background=int(bg),
            )
        except (AssertionError, RecursionError):
            continue

        mask = sprite != int(bg)
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[r0 : r1 + 1, c0 : c1 + 1]

        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=int(bg), border=1, padding=1)
        except ValueError:
            continue

        sh, sw = sprite.shape
        sprite_mask = sprite != int(bg)
        blob_rows = {int(pr + dr) for dr in range(sh) if sprite_mask[dr].any()}
        blob_cols = {int(pc + dc) for dc in range(sw) if sprite_mask[:, dc].any()}

        if orient == 0 and start_pos in blob_rows:
            continue
        if orient == 1 and start_pos in blob_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != int(bg):
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    fg_mask = (canvas != int(bg)) & (canvas != int(Y))
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)
    metrics = _metric_bundle(inp)

    if int(get_bg(inp)) != int(bg):
        return None
    if metrics["yellow_flood_hits"] <= 0:
        return None
    if metrics["green_widen_hits"] <= 0:
        return None
    if metrics["green_no_flood_hits"] <= 0:
        return None
    if metrics["diff_vs_d4"] <= 0 or metrics["diff_vs_d10"] <= 0:
        return None
    if metrics["diff_vs_original"] <= 0:
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
