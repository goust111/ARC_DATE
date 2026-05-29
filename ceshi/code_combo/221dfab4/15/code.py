# Combo variant of 221dfab4 (index 15)
# combo_spec: D2/0+D5/2+D14/0
# Name: 步长3+间隙周期变化+仅水平
#
# Design analysis for the fused variant
# ------------------------------------
# 1. Concrete behavior and execution trace:
#    The original task accepts a yellow border segment on any edge and advances with uniform
#    stripe spacing 2: stripe, 1-cell gap, stripe, 1-cell gap, ... while cycling Y, Y, G.
#    This fusion keeps the stripe width = 1 and keeps the Y, Y, G recolor semantics, but:
#    - only top/bottom edge seeds are legal, so propagation is always vertical in terms of rows;
#    - the trajectory is horizontal-stripe-only, so left/right edge cases are removed;
#    - spacing is no longer uniform: gap widths alternate 1, 2, 1, 2, ... so stripe rows are
#      separated by 2, 3, 2, 3, ... .
#    Example with a top seed spanning columns 4..7:
#      Original rows: stripe@0(Y), gap@1, stripe@2(Y), gap@3, stripe@4(G), gap@5, stripe@6(Y)
#      Fused rows:    stripe@0(Y), gap@1, stripe@2(Y), gaps@3,4, stripe@5(G), gap@6, stripe@7(Y)
#    Difference:
#      - row 4 was a green stripe in the original, but is now the second cleared gap row;
#      - row 5 is now the first green stripe, so any non-bg pixels anywhere on row 5 turn green;
#      - the segment itself visibly shows 1 blank row, then 2 blank rows, then 1, then 2.
#
# 2. Complexity calibration:
#    A pure stride-3 uniform pattern would be visible but too regular, while a pure D5 alternating
#    gap without the horizontal-only restriction would keep extra orientation freedom. The fusion is
#    still human-solvable because only top/bottom seeds appear, yet it is not degenerate:
#    several stripe intervals in a 26-30 row grid show the repeating 2,3,2,3 spacing, and green
#    flood rows move away from the original even-row schedule. This creates repeated evidence while
#    keeping the number of free parameters small.
#
# 3. Hidden assumptions review against the original `main()`:
#    - The border mask originally allowed all four sides. That assumption is invalid now; we only
#      inspect yellow pixels on row 0 or row H-1.
#    - The original `pos = start + dir * (2 * step)` assumed a fixed 1-cell gap. That is invalid;
#      positions must be generated cumulatively from the alternating 1,2 gap cycle.
#    - The original vertical branch assumed left/right edge seeds may exist. That branch is removed,
#      and non-horizontal yellow components are treated as out-of-spec input and left unchanged.
#    - The green recolor step (`row`-wide non-bg -> green) still holds once stripe rows are updated.
#    - Gap clearing is still limited to the segment span, but now it must clear one or two rows after
#      each stripe depending on the cycle phase.
#
# 4. Placement legality in the generator:
#    - The generator emits only top/bottom yellow segments and keeps them away from corners.
#    - Foreground blobs are placed with border and padding constraints so they do not overlap, do not
#      leave the grid, and do not touch the yellow seed row.
#    - One targeted blob is placed on a fusion-only green row outside the yellow span so the green
#      full-row recolor is always observable.
#
# 5. Visual observability thought experiment:
#    From 2-4 examples, a solver sees the same horizontal seed mechanism each time, so orientation
#    inference becomes easy. What remains to infer is the nonuniform vertical rhythm: one blank row,
#    then two blank rows, repeating. Because at least one interior blob lies on a fusion-only green
#    row outside the span, the green flood effect is visible independently of the yellow segment.
#
# 6. Degeneracy check used after implementation:
#    We compare the fused output to the original uniform-spacing solver on the same generated input.
#    The key metric is the Hamming distance between those outputs. If it were always zero, samples
#    would have degenerated. The generator also guarantees at least one foreground pixel on a fused
#    green row that is not an original green row, so the fusion changes more than just stripe spacing.
#
# 7. New solver assumptions and how the generator satisfies them:
#    - Assumption: the main yellow component is a straight horizontal segment on row 0 or row H-1.
#      Guarantee: generator creates exactly one such segment.
#    - Assumption: there are enough rows to display several 2/3-spaced stripes.
#      Guarantee: H is chosen in [26, 30] and candidates with too few stripe rows are rejected.
#    - Assumption: at least one fused green row intersects interior foreground outside the span so the
#      row-wide recolor is visible.
#      Guarantee: generator deliberately places a blob on a fused-only green row outside the span.

from __future__ import annotations

import random

import numpy as np

from arc2_solver_select.arc_utils import get_bg, label_mask, random_free_location, random_sprite


Y = 4
G = 3
COLOR_CYCLE = (Y, Y, G)
GAP_CYCLE = (1, 2)


def _iter_combo_trajectory(start_row: int, dir1: int, height: int):
    """Yield `(step, stripe_row, cleared_gap_rows)` for the alternating 1,2,1,2 gap schedule."""
    row = int(start_row)
    step = 0
    while 0 <= row < int(height):
        gap_w = int(GAP_CYCLE[step % len(GAP_CYCLE)])
        gap_rows = []
        for off in range(1, gap_w + 1):
            rr = row + int(dir1) * off
            if 0 <= rr < int(height):
                gap_rows.append(int(rr))
        yield int(step), int(row), gap_rows
        row += int(dir1) * (1 + gap_w)
        step += 1


def _iter_original_trajectory(start_row: int, dir1: int, height: int):
    """Original uniform spacing: every stripe is followed by exactly one cleared row."""
    row = int(start_row)
    step = 0
    while 0 <= row < int(height):
        gap = row + int(dir1)
        gaps = [int(gap)] if 0 <= gap < int(height) else []
        yield int(step), int(row), gaps
        row += int(dir1) * 2
        step += 1


def _green_rows_from_iter(iterator) -> list[int]:
    return [row for step, row, _gaps in iterator if step % len(COLOR_CYCLE) == 2]


def _find_seed_segment(grid: np.ndarray):
    """Return `(start_row, dir1, span_cols)` for the main top/bottom yellow segment, else None."""
    g = np.array(grid, dtype=int)
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True

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

    if r0 != r1:
        return None
    if r0 not in (0, H - 1):
        return None

    span_cols = list(range(int(c0), int(c1) + 1))
    dir1 = 1 if r0 == 0 else -1
    return int(r0), int(dir1), span_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - only horizontal yellow seeds on the top/bottom border are recognized;
    - stripes progress away from the seed row with spacing 2,3,2,3,...;
    - colors cycle Y, Y, G;
    - yellow paints only inside the seed span;
    - green paints the whole non-background row green, then paints the span itself green.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    seed = _find_seed_segment(g)
    if seed is None:
        return out

    start_row, dir1, span_cols = seed

    for step, row, gap_rows in _iter_combo_trajectory(start_row, dir1, H):
        color = int(COLOR_CYCLE[step % len(COLOR_CYCLE)])
        if color == int(G):
            out[row, :] = np.where(out[row, :] != int(bg), int(G), out[row, :])
        for c in span_cols:
            out[row, int(c)] = color
        for gap_row in gap_rows:
            for c in span_cols:
                out[int(gap_row), int(c)] = int(bg)

    return out


def _fits_with_padding(canvas: np.ndarray, r0: int, c0: int, patch: np.ndarray, bg: int, padding: int = 1) -> bool:
    H, W = canvas.shape
    h, w = patch.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False
    rr0 = max(0, int(r0) - int(padding))
    rr1 = min(H, int(r0) + h + int(padding))
    cc0 = max(0, int(c0) - int(padding))
    cc1 = min(W, int(c0) + w + int(padding))
    return not bool((canvas[rr0:rr1, cc0:cc1] != int(bg)).any())


def _paste_patch(canvas: np.ndarray, r0: int, c0: int, patch: np.ndarray, bg: int) -> None:
    h, w = patch.shape
    for dr in range(h):
        for dc in range(w):
            if int(patch[dr, dc]) != int(bg):
                canvas[int(r0) + dr, int(c0) + dc] = int(patch[dr, dc])


def _target_columns(width: int, patch_w: int, span_cols: list[int]) -> list[int]:
    left = []
    right = []
    span_lo = int(span_cols[0])
    span_hi = int(span_cols[-1])
    left_max = span_lo - patch_w - 1
    if left_max >= 1:
        left = list(range(1, left_max + 1))
    right_min = span_hi + 2
    right_max = width - patch_w - 1
    if right_min <= right_max:
        right = list(range(right_min, right_max + 1))
    return left + right


def _place_target_blob(canvas: np.ndarray, target_row: int, fg: int, bg: int, span_cols: list[int], rng: random.Random) -> bool:
    """Place a small rectangle touching `target_row` and outside the yellow span."""
    patch = np.full((2, 3), int(fg), dtype=int)
    H, W = canvas.shape

    candidate_rows = []
    for r0 in (int(target_row) - 1, int(target_row)):
        if 1 <= r0 <= H - patch.shape[0] - 1 and r0 <= int(target_row) < r0 + patch.shape[0]:
            candidate_rows.append(int(r0))
    if not candidate_rows:
        return False

    candidate_cols = _target_columns(W, patch.shape[1], span_cols)
    rng.shuffle(candidate_rows)
    rng.shuffle(candidate_cols)

    for r0 in candidate_rows:
        for c0 in candidate_cols:
            if _fits_with_padding(canvas, r0, c0, patch, bg, padding=1):
                _paste_patch(canvas, r0, c0, patch, bg)
                return True
    return False


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not rows_any.any():
        return None
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[r0 : r1 + 1, c0 : c1 + 1]


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    """Quantify that the fused behavior differs from the original uniform-spacing rule."""
    seed = _find_seed_segment(inp)
    if seed is None:
        return {"delta_vs_original": 0, "fused_only_green_fg": 0, "wide_gaps": 0}

    start_row, dir1, _span_cols = seed
    fused_out = main(inp)
    original_out = _apply_uniform_original(inp)

    fused_green = set(_green_rows_from_iter(_iter_combo_trajectory(start_row, dir1, inp.shape[0])))
    original_green = set(_green_rows_from_iter(_iter_original_trajectory(start_row, dir1, inp.shape[0])))
    fg_mask = (inp != int(get_bg(inp))) & (inp != int(Y))

    fused_only_green_fg = sum(int(np.any(fg_mask[r, :])) for r in sorted(fused_green - original_green))
    wide_gaps = sum(1 for _step, _row, gaps in _iter_combo_trajectory(start_row, dir1, inp.shape[0]) if len(gaps) == 2)
    delta = int(np.sum(fused_out != original_out))
    return {
        "delta_vs_original": delta,
        "fused_only_green_fg": int(fused_only_green_fg),
        "wide_gaps": int(wide_gaps),
    }


def _apply_uniform_original(grid: np.ndarray) -> np.ndarray:
    """Reference implementation of the original top/bottom-uniform rule, used only for degeneracy checks."""
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    seed = _find_seed_segment(g)
    if seed is None:
        return out
    start_row, dir1, span_cols = seed

    for step, row, gap_rows in _iter_original_trajectory(start_row, dir1, H):
        color = int(COLOR_CYCLE[step % len(COLOR_CYCLE)])
        if color == int(G):
            out[row, :] = np.where(out[row, :] != int(bg), int(G), out[row, :])
        for c in span_cols:
            out[row, int(c)] = color
        for gap_row in gap_rows:
            for c in span_cols:
                out[int(gap_row), int(c)] = int(bg)
    return out


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(26, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom"])
    seg_len = rng.randint(4, 8)
    max_start = W - 1 - seg_len
    if max_start < 1:
        return None

    seg_start = rng.randint(1, max_start)
    span_cols = list(range(seg_start, seg_start + seg_len))
    seed_row = 0 if edge == "top" else H - 1
    dir1 = 1 if edge == "top" else -1

    fused_rows = [row for _step, row, _gaps in _iter_combo_trajectory(seed_row, dir1, H)]
    if len(fused_rows) < 8:
        return None

    fused_green_rows = _green_rows_from_iter(_iter_combo_trajectory(seed_row, dir1, H))
    original_green_rows = _green_rows_from_iter(_iter_original_trajectory(seed_row, dir1, H))
    fusion_only_green = [r for r in fused_green_rows if r not in original_green_rows and 1 <= r < H - 1]
    if not fusion_only_green:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    for c in span_cols:
        grid[seed_row, int(c)] = int(Y)

    canvas = grid.copy()

    target_row = rng.choice(fusion_only_green)
    if not _place_target_blob(canvas, target_row, fg, bg, span_cols, rng):
        return None

    n_blobs = rng.randint(4, 8)
    for _ in range(n_blobs):
        bh = rng.randint(2, 9)
        bw = rng.randint(2, 9)
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

        sprite = _trim_sprite(sprite, bg)
        if sprite is None:
            continue

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=int(bg), border=1, padding=1)
        except ValueError:
            continue

        if seed_row in range(int(pr), int(pr) + sh):
            continue
        _paste_patch(canvas, int(pr), int(pc), sprite, bg)

    fg_mask = (canvas != int(bg)) & (canvas != int(Y))
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if not any(np.any(fg_mask[r, :]) for r in fused_green_rows):
        return None

    inp = canvas.copy()
    if int(get_bg(inp)) != int(bg):
        return None

    metrics = _variant_metrics(inp)
    if metrics["delta_vs_original"] <= 0:
        return None
    if metrics["fused_only_green_fg"] <= 0:
        return None
    if metrics["wide_gaps"] < 2:
        return None

    _ = main(inp)
    return inp


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
