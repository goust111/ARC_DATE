# Combo variant of 221dfab4 (index 9)
# combo_spec: D5/0+D8/1+D9/0
# Name: 保留间隙+色块遮挡+裁剪输出
#
# Design analysis for this combo
# a) Concrete behavior:
#    The yellow border segment still defines an orientation, a finite span, and a Y-Y-G stripe
#    cycle every two cells as it marches toward the opposite side. The fused change is best viewed
#    as a two-layer construction:
#    - Bottom layer: place the propagated stripes only inside the segment span.
#    - Top layer: keep the original interior content wherever it already existed, and keep every
#      gap cell exactly as it was instead of forcibly clearing it.
#    - Final step: crop to the minimal rectangle covered by the stripe trajectory.
#    Example, horizontal segment on the top edge with span columns 3..6:
#      input rows 0..5 inside that span:
#        r0: Y Y Y Y
#        r1: . A . .
#        r2: . . B .
#        r3: . C . .
#        r4: . . . .
#        r5: . . . .
#      original task:
#        r1 is forced to background, so A disappears;
#        r2 becomes yellow across the whole span, so B is overwritten by yellow;
#        r3 is forced to background, so C disappears;
#        r4 becomes green across the span and also recolors non-background cells on the whole row.
#      combo output:
#        r1 stays ". A . ." because gap cells are preserved;
#        r2 becomes "Y Y B Y" because stripes sit behind the source block;
#        r3 stays ". C . ." for the same gap-preservation reason;
#        r4 becomes green only on background cells in the span.
#      Then the output is cropped to rows 0..last_stripe and columns 3..6.
# b) Complexity calibration:
#    If blobs never collide with stripe cells or gap cells, the sample degenerates into a simple
#    crop of the original task. The generator therefore forces three distinct collision types:
#    yellow-stripe collisions, green-stripe collisions, and gap collisions. A few extra blobs are
#    added afterward so the preserved colors repeat across multiple examples instead of looking like
#    one-off noise.
# c) Implicit assumptions audit:
#    - Original line `out[pos, :] = np.where(..., G, ...)` assumed green may freely recolor every
#      non-background pixel on that row/column. That no longer holds because source blobs now sit on
#      top of the stripe layer, and the final crop only keeps the span rectangle anyway.
#    - Original direct writes `out[pos, span] = col` assumed overwriting source blobs was valid.
#      The combo must instead paint only cells that were background in the input.
#    - Original gap clearing `out[gap, span] = bg` assumed every gap cell should become background.
#      D5 explicitly breaks that assumption, so the combo performs no gap write at all.
#    - Using `out != bg` as an obstacle mask would be wrong, because previous stripe paints would
#      incorrectly block later logic. Every protection check therefore consults the original input
#      `g`, not the evolving output.
# d) Placement legality:
#    The generator keeps blobs at least one cell away from the grid border and one cell away from
#    each other. Targeted blobs are anchored to legal stripe/gap cells inside the eventual crop
#    rectangle, so no placement goes out of bounds and every collision category is guaranteed.
# e) Visual observability:
#    A human should be able to infer three things from 2-4 examples:
#    - some colored blobs survive on rows/columns where stripes pass, creating visible stripe holes;
#    - some colored blobs survive in the nominal 1-cell gaps, proving the gaps are not cleared;
#    - the output frame shrinks tightly to the stripe trajectory rectangle, removing unrelated
#      background around it. The generator rejects samples unless all three cues are quantitatively
#      visible.
# f) New implicit assumptions introduced by the combo solver:
#    - The largest yellow border component is the intended segment and is a straight line.
#      The generator creates exactly one such component.
#    - The crop box is defined only by the stripe trajectory rectangle, not by arbitrary blobs.
#      The generator may place extra blobs outside that rectangle, but they are intentionally
#      discarded by the crop.
#    - Stripe and gap visibility rely on collisions with existing foreground. The generator
#      guarantees those collisions and rejects low-signal cases with metrics.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[5] / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


Y = 4
G = 3


def _segment_info(g: np.ndarray) -> tuple[str, int, list[int], int, int] | None:
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
        span = list(range(int(cols[0]), int(cols[-1]) + 1))
        dir1 = 1 if start_pos == 0 else -1
        return "h", start_pos, span, dir1, H

    if c0 == c1:
        start_pos = int(c0)
        rows = sorted(int(r) for r in rs)
        span = list(range(int(rows[0]), int(rows[-1]) + 1))
        dir1 = 1 if start_pos == 0 else -1
        return "v", start_pos, span, dir1, W

    return None


def _stripe_steps(start_pos: int, dir1: int, limit: int):
    cycle = [int(Y), int(Y), int(G)]
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
        yield int(step), int(pos), int(cycle[int(step) % 3]), int(gap)
        step += 1


def _trajectory_meta(info: tuple[str, int, list[int], int, int], shape: tuple[int, int]) -> dict[str, object]:
    orient, start_pos, span, dir1, limit = info
    stripe_positions: list[int] = []
    gap_positions: list[int] = []
    stripe_colors: list[int] = []

    for _step, pos, col, gap in _stripe_steps(start_pos, dir1, limit):
        stripe_positions.append(int(pos))
        stripe_colors.append(int(col))
        if 0 <= gap < limit:
            gap_positions.append(int(gap))

    if orient == "h":
        r0 = min(stripe_positions)
        r1 = max(stripe_positions)
        c0 = min(span)
        c1 = max(span)
    else:
        r0 = min(span)
        r1 = max(span)
        c0 = min(stripe_positions)
        c1 = max(stripe_positions)

    return {
        "orient": orient,
        "start_pos": int(start_pos),
        "span": list(int(x) for x in span),
        "dir1": int(dir1),
        "limit": int(limit),
        "stripe_positions": stripe_positions,
        "gap_positions": gap_positions,
        "stripe_colors": stripe_colors,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "shape": tuple(int(x) for x in shape),
    }


def _crop_to_trajectory(arr: np.ndarray, meta: dict[str, object]) -> np.ndarray:
    r0, c0, r1, c1 = meta["bbox"]
    return np.array(arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Propagate the same Y-Y-G stripe cycle from the yellow border segment.
    2. Paint stripes only on cells that were background in the input.
    3. Do not clear the 1-cell gaps at all; those cells keep their original content.
    4. Crop the final result to the trajectory rectangle.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    info = _segment_info(g)
    if info is None:
        return out

    meta = _trajectory_meta(info, g.shape)
    orient = meta["orient"]
    span_idx = np.array(meta["span"], dtype=int)

    for pos, col in zip(meta["stripe_positions"], meta["stripe_colors"]):
        pos = int(pos)
        col = int(col)
        if orient == "h":
            paintable = g[pos, span_idx] == int(bg)
            if bool(paintable.any()):
                out[pos, span_idx[paintable]] = int(col)
        else:
            paintable = g[span_idx, pos] == int(bg)
            if bool(paintable.any()):
                out[span_idx[paintable], pos] = int(col)

    return _crop_to_trajectory(out, meta)


def _original_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    info = _segment_info(g)
    if info is None:
        return out

    orient, start_pos, span, dir1, limit = info

    for _step, pos, col, gap in _stripe_steps(start_pos, dir1, limit):
        if orient == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            for c in span:
                out[pos, int(c)] = int(col)
            if 0 <= gap < g.shape[0]:
                for c in span:
                    out[int(gap), int(c)] = int(bg)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            for r in span:
                out[int(r), pos] = int(col)
            if 0 <= gap < g.shape[1]:
                for r in span:
                    out[int(r), int(gap)] = int(bg)

    return out


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    if not bool(mask.any()):
        return None

    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    trimmed = sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    pix = int(np.sum(trimmed != int(bg)))
    if pix < 5 or pix > 67:
        return None
    return trimmed


def _can_place(
    canvas: np.ndarray,
    sprite: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    *,
    border: int = 1,
    padding: int = 1,
) -> bool:
    H, W = canvas.shape
    h, w = sprite.shape

    if r0 < border or c0 < border or r0 + h > H - border or c0 + w > W - border:
        return False

    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    if bool((canvas[rr0:rr1, cc0:cc1] != int(bg)).any()):
        return False

    return True


def _paste(canvas: np.ndarray, sprite: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = sprite.shape
    for dr in range(h):
        for dc in range(w):
            if int(sprite[dr, dc]) != int(bg):
                canvas[int(r0 + dr), int(c0 + dc)] = int(sprite[dr, dc])


def _random_blob(rng: random.Random, bg: int, color: int) -> np.ndarray | None:
    for _ in range(30):
        bh = rng.randint(2, 9)
        bw = rng.randint(2, 9)
        density = rng.uniform(0.45, 0.85)
        try:
            sprite = random_sprite(
                bh,
                bw,
                density=density,
                symmetry="not_symmetric",
                color_palette=[int(color)],
                connectivity=4,
                background=int(bg),
            )
        except (AssertionError, RecursionError):
            continue

        trimmed = _trim_sprite(sprite, bg)
        if trimmed is not None:
            return trimmed

    return None


def _place_targeted_blob(
    canvas: np.ndarray,
    rng: random.Random,
    bg: int,
    color: int,
    targets: list[tuple[int, int]],
) -> bool:
    if not targets:
        return False

    for _ in range(160):
        sprite = _random_blob(rng, bg, color)
        if sprite is None:
            continue

        occ = np.argwhere(sprite != int(bg))
        if occ.size == 0:
            continue

        tr, tc = targets[rng.randrange(len(targets))]
        rr, cc = occ[rng.randrange(len(occ))]
        r0 = int(tr) - int(rr)
        c0 = int(tc) - int(cc)
        if not _can_place(canvas, sprite, r0, c0, bg, border=1, padding=1):
            continue

        _paste(canvas, sprite, r0, c0, bg)
        return True

    return False


def _place_random_blob(canvas: np.ndarray, rng: random.Random, bg: int, color: int) -> bool:
    sprite = _random_blob(rng, bg, color)
    if sprite is None:
        return False

    try:
        pr, pc = random_free_location(canvas, sprite, transparent=int(bg), border=1, padding=1)
    except ValueError:
        return False

    _paste(canvas, sprite, int(pr), int(pc), bg)
    return True


def _targets_from_meta(meta: dict[str, object]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    orient = meta["orient"]
    start_pos = int(meta["start_pos"])
    span = [int(x) for x in meta["span"]]
    stripe_positions = [int(x) for x in meta["stripe_positions"]]
    stripe_colors = [int(x) for x in meta["stripe_colors"]]
    gap_positions = [int(x) for x in meta["gap_positions"]]

    yellow_targets: list[tuple[int, int]] = []
    green_targets: list[tuple[int, int]] = []
    gap_targets: list[tuple[int, int]] = []

    if orient == "h":
        for pos, col in zip(stripe_positions, stripe_colors):
            if pos == int(start_pos):
                continue
            for c in span:
                if col == int(Y):
                    yellow_targets.append((int(pos), int(c)))
                else:
                    green_targets.append((int(pos), int(c)))
        for gap in gap_positions:
            for c in span:
                gap_targets.append((int(gap), int(c)))
    else:
        for pos, col in zip(stripe_positions, stripe_colors):
            if pos == int(start_pos):
                continue
            for r in span:
                if col == int(Y):
                    yellow_targets.append((int(r), int(pos)))
                else:
                    green_targets.append((int(r), int(pos)))
        for gap in gap_positions:
            for r in span:
                gap_targets.append((int(r), int(gap)))

    return yellow_targets, green_targets, gap_targets


def _cropped_original(grid: np.ndarray) -> np.ndarray:
    info = _segment_info(grid)
    if info is None:
        return np.array(grid, dtype=int)
    meta = _trajectory_meta(info, np.array(grid).shape)
    return _crop_to_trajectory(_original_main(grid), meta)


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    info = _segment_info(g)
    if info is None:
        return {
            "stripe_preserved": 0,
            "gap_preserved": 0,
            "painted_background": 0,
            "crop_smaller": 0,
            "components": 0,
        }

    meta = _trajectory_meta(info, g.shape)
    combo = main(g)
    orig_crop = _cropped_original(g)
    inp_crop = _crop_to_trajectory(g, meta)

    h, w = combo.shape
    stripe_mask = np.zeros((h, w), dtype=bool)
    gap_mask = np.zeros((h, w), dtype=bool)

    r0, c0, _r1, _c1 = meta["bbox"]
    orient = meta["orient"]
    if orient == "h":
        for pos in meta["stripe_positions"]:
            stripe_mask[int(pos) - int(r0), :] = True
        for gap in meta["gap_positions"]:
            if int(r0) <= int(gap) <= int(r0) + h - 1:
                gap_mask[int(gap) - int(r0), :] = True
    else:
        for pos in meta["stripe_positions"]:
            stripe_mask[:, int(pos) - int(c0)] = True
        for gap in meta["gap_positions"]:
            if int(c0) <= int(gap) <= int(c0) + w - 1:
                gap_mask[:, int(gap) - int(c0)] = True

    protected = (inp_crop != int(bg)) & (combo == inp_crop) & (orig_crop != inp_crop)
    painted_background = (inp_crop == int(bg)) & (combo != int(bg)) & stripe_mask

    fg_mask = (g != int(bg)) & (g != int(Y))
    _lab, n_components = label_mask(fg_mask, connectivity=4)

    return {
        "stripe_preserved": int(np.count_nonzero(protected & stripe_mask)),
        "gap_preserved": int(np.count_nonzero(protected & gap_mask)),
        "painted_background": int(np.count_nonzero(painted_background)),
        "crop_smaller": int(combo.shape != g.shape),
        "components": int(n_components),
    }


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(24, 30)
    W = rng.randint(24, 30)

    bg = rng.choice([c for c in range(10) if c not in (int(Y), int(G))])
    fg_colors = rng.sample([c for c in range(10) if c not in (int(Y), int(G), int(bg))], 4)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(4, 8)

    grid = np.full((H, W), int(bg), dtype=int)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_row = 0 if edge == "top" else H - 1
        for c in span:
            grid[int(edge_row), int(c)] = int(Y)
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_col = 0 if edge == "left" else W - 1
        for r in span:
            grid[int(r), int(edge_col)] = int(Y)

    info = _segment_info(grid)
    if info is None:
        return None

    meta = _trajectory_meta(info, grid.shape)
    yellow_targets, green_targets, gap_targets = _targets_from_meta(meta)
    if not yellow_targets or not green_targets or not gap_targets:
        return None

    canvas = grid.copy()

    must_place = [
        (yellow_targets, fg_colors[0]),
        (green_targets, fg_colors[1]),
        (gap_targets, fg_colors[2]),
        (gap_targets, fg_colors[3]),
        (yellow_targets + green_targets, fg_colors[0]),
    ]
    for targets, color in must_place:
        if not _place_targeted_blob(canvas, rng, bg, int(color), targets):
            return None

    extra_blobs = rng.randint(1, 3)
    for i in range(extra_blobs):
        if not _place_random_blob(canvas, rng, bg, int(fg_colors[i % len(fg_colors)])):
            return None

    if int(get_bg(canvas)) != int(bg):
        return None

    metrics = _combo_metrics(canvas)
    if metrics["components"] < 5:
        return None
    if metrics["stripe_preserved"] < 4:
        return None
    if metrics["gap_preserved"] < 4:
        return None
    if metrics["painted_background"] < 8:
        return None
    if metrics["crop_smaller"] != 1:
        return None

    combo_out = main(canvas)
    orig_crop = _cropped_original(canvas)
    if np.array_equal(combo_out, orig_crop):
        return None

    return canvas


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
