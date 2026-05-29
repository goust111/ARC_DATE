# Combo variant of 221dfab4 (index 17)
# combo_spec: D1/0+D2/2+D3/1
# Name: 对角推进+递增步长+四步周期

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

HELPER_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(HELPER_DIR) not in sys.path:
    sys.path.append(str(HELPER_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the combo variant:
# a) Concretization + trace:
#    Take a top-edge seed on cols 4..7. The original task used straight positions
#    row 0, 2, 4, 6, ... with colors Y, Y, G, Y, ...
#    This combo instead uses diagonal cumulative shifts s = 0, 2, 5, 9, 14, ...
#    so the stripes become:
#      s=0  -> row 0,  cols 4..7,   Y
#      s=2  -> row 2,  cols 6..9,   Y
#      s=5  -> row 5,  cols 9..12,  Y
#      s=9  -> row 9,  cols 13..16, G
#    and every intermediate diagonal copy of the span is cleared to bg:
#      s=1, then s=3..4, then s=6..8, ...
#    Compared with the original, three things now interact instead of acting
#    independently: the stripe anchor drifts diagonally, the gap lengths grow as
#    1,2,3,..., and the green flood appears only every fourth stripe (Y,Y,Y,G).
# b) Complexity calibration:
#    The generator enforces at least seven visible stripes, which exposes five
#    different gap lengths and a large diagonal drift. That keeps the output from
#    collapsing back into a near-straight stack and makes the "dense near the seed,
#    sparse later" fan readable by eye.
# c) Hidden-assumption audit on the original solver:
#    - `pos = start + 2*step` assumed constant stride; replaced by cumulative shifts.
#    - Reusing a fixed `span` assumed no lateral drift; now the span is translated
#      by the same scalar shift used for the primary direction.
#    - Clearing a single gap cell assumed unit spacing; now every diagonal copy
#      between two consecutive stripes must be reset to bg.
#    - Cycling by 3 assumed green every third stripe; replaced with a 4-step cycle.
#    - Flooding a whole row/col still works, but only after computing the diagonally
#      shifted row/col for the current stripe.
# d) Placement legality:
#    The generator chooses the seed so the whole shifted span stays in-bounds for the
#    required number of stripes, keeps the seed away from corners, and places blobs
#    away from the seed row/col so the yellow source stays visually isolated.
#    A forced blob is added on the first green row/col (but outside the stripe span)
#    so the rare green flood is guaranteed to have a visible effect.
# e) Visual observability:
#    A solver seeing a few examples should be able to read:
#    - the anchor walks on a 45-degree diagonal instead of straight across;
#    - the empty corridors widen from 1 to 2 to 3 to ... cells;
#    - green full-row/full-column recolors are sparse, appearing after three yellow
#      stripes rather than every other yellow pair.
# Additional new assumptions and how the generator satisfies them:
#    - `main()` assumes exactly one dominant yellow border segment; the generator uses
#      no other yellow pixels in the input.
#    - `main()` assumes the seed is straight and border-aligned; the generator builds
#      only horizontal or vertical edge segments.
#    - `main()` assumes at least one green stripe can exist; the generator requires
#      enough runway for at least seven stripes, so the first green stripe is present.
#    - The forced green-hit blob assumes some free space exists on one side of the
#      first green span; the generator sizes the board and seed runway to leave room.

Y = 4
G = 3
CYCLE = (Y, Y, Y, G)
MIN_STRIPES = 7
FIRST_GREEN_STEP = 3
MIN_GREEN_RECOLOR = 6
MIN_NEW_YELLOW = 18
MIN_DIAGONAL_DRIFT = 20


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


def _shift_values(values: list[int], shift: int, direction: int) -> list[int]:
    return [int(v + direction * shift) for v in values]


def _stripe_cells(info: dict, shift: int):
    if info["orient"] == "h":
        row = int(info["fixed"] + info["dr"] * shift)
        cols = _shift_values(info["span"], shift, info["dc"])
        return row, cols
    col = int(info["fixed"] + info["dc"] * shift)
    rows = _shift_values(info["span"], shift, info["dr"])
    return col, rows


def _stripe_in_bounds(info: dict, shift: int, H: int, W: int) -> bool:
    if info["orient"] == "h":
        row, cols = _stripe_cells(info, shift)
        return 0 <= int(row) < H and all(0 <= int(c) < W for c in cols)
    col, rows = _stripe_cells(info, shift)
    return 0 <= int(col) < W and all(0 <= int(r) < H for r in rows)


def _stripe_shifts(info: dict, H: int, W: int) -> list[int]:
    shifts: list[int] = []
    shift = 0
    step = 0
    while _stripe_in_bounds(info, shift, H, W):
        shifts.append(int(shift))
        shift += int(step) + 2
        step += 1
    return shifts


def _gap_lengths(shifts: list[int]) -> list[int]:
    return [int(shifts[i + 1] - shifts[i] - 1) for i in range(len(shifts) - 1)]


def _last_required_shift(num_stripes: int) -> int:
    shift = 0
    step = 0
    seen = 0
    while seen < num_stripes:
        current = shift
        shift += int(step) + 2
        step += 1
        seen += 1
    return int(current)


def _paste_segment(grid: np.ndarray, seg: dict):
    if seg["orient"] == "h":
        grid[int(seg["fixed"]), np.array(seg["span"], dtype=int)] = Y
    else:
        grid[np.array(seg["span"], dtype=int), int(seg["fixed"])] = Y


def _paint_gap(out: np.ndarray, info: dict, shift: int, bg: int):
    H, W = out.shape
    if not _stripe_in_bounds(info, shift, H, W):
        return

    if info["orient"] == "h":
        row, cols = _stripe_cells(info, shift)
        out[int(row), np.array(cols, dtype=int)] = bg
    else:
        col, rows = _stripe_cells(info, shift)
        out[np.array(rows, dtype=int), int(col)] = bg


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    info = _find_source_segment(g)
    if info is None:
        return out

    shifts = _stripe_shifts(info, H, W)
    for step, shift in enumerate(shifts):
        color = int(CYCLE[step % len(CYCLE)])

        if info["orient"] == "h":
            row, cols = _stripe_cells(info, shift)
            cols_arr = np.array(cols, dtype=int)
            if color == G:
                out[int(row), :] = np.where(out[int(row), :] != bg, G, out[int(row), :])
            out[int(row), cols_arr] = color
        else:
            col, rows = _stripe_cells(info, shift)
            rows_arr = np.array(rows, dtype=int)
            if color == G:
                out[:, int(col)] = np.where(out[:, int(col)] != bg, G, out[:, int(col)])
            out[rows_arr, int(col)] = color

        if step + 1 < len(shifts):
            next_shift = int(shifts[step + 1])
            for gap_shift in range(int(shift) + 1, next_shift):
                _paint_gap(out, info, gap_shift, bg)

    return out


def _choose_segment(H: int, W: int, rng: random.Random):
    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(4, 7)
    runway = _last_required_shift(MIN_STRIPES)

    if edge == "top":
        upper = W - seg_len - runway
        if upper < 1 or runway >= H:
            return None
        start = rng.randint(1, upper)
        return {"edge": edge, "orient": "h", "fixed": 0, "span": list(range(start, start + seg_len))}

    if edge == "bottom":
        lower = max(1, runway)
        upper = W - seg_len - 1
        if lower > upper or runway >= H:
            return None
        start = rng.randint(lower, upper)
        return {"edge": edge, "orient": "h", "fixed": H - 1, "span": list(range(start, start + seg_len))}

    if edge == "left":
        upper = H - seg_len - runway
        if upper < 1 or runway >= W:
            return None
        start = rng.randint(1, upper)
        return {"edge": edge, "orient": "v", "fixed": 0, "span": list(range(start, start + seg_len))}

    lower = max(1, runway)
    upper = H - seg_len - 1
    if lower > upper or runway >= W:
        return None
    start = rng.randint(lower, upper)
    return {"edge": edge, "orient": "v", "fixed": W - 1, "span": list(range(start, start + seg_len))}


def _can_place_rect(canvas: np.ndarray, top: int, left: int, height: int, width: int, bg: int) -> bool:
    H, W = canvas.shape
    if top < 1 or left < 1 or top + height > H - 1 or left + width > W - 1:
        return False
    return bool(np.all(canvas[top : top + height, left : left + width] == bg))


def _place_rect(canvas: np.ndarray, top: int, left: int, height: int, width: int, fg: int):
    canvas[top : top + height, left : left + width] = int(fg)


def _place_forced_green_blob(canvas: np.ndarray, info: dict, shifts: list[int], fg: int, bg: int) -> bool:
    if len(shifts) <= FIRST_GREEN_STEP:
        return False

    H, W = canvas.shape
    green_shift = int(shifts[FIRST_GREEN_STEP])

    if info["orient"] == "h":
        target_row, cols = _stripe_cells(info, green_shift)
        stripe_lo, stripe_hi = min(cols), max(cols)
        rect_h, rect_w = 3, 6
        candidate_rows = [int(target_row) - 1, int(target_row) - 2, int(target_row)]
        left_cols = list(range(1, max(1, stripe_lo - rect_w - 1) + 1))
        right_start = stripe_hi + 2
        right_cols = list(range(right_start, W - rect_w))
        for top in candidate_rows:
            for left in left_cols + right_cols:
                if _can_place_rect(canvas, top, left, rect_h, rect_w, bg):
                    _place_rect(canvas, top, left, rect_h, rect_w, fg)
                    return True
        return False

    target_col, rows = _stripe_cells(info, green_shift)
    stripe_lo, stripe_hi = min(rows), max(rows)
    rect_h, rect_w = 6, 3
    candidate_cols = [int(target_col) - 1, int(target_col) - 2, int(target_col)]
    upper_rows = list(range(1, max(1, stripe_lo - rect_h - 1) + 1))
    lower_start = stripe_hi + 2
    lower_rows = list(range(lower_start, H - rect_h))
    for left in candidate_cols:
        for top in upper_rows + lower_rows:
            if _can_place_rect(canvas, top, left, rect_h, rect_w, bg):
                _place_rect(canvas, top, left, rect_h, rect_w, fg)
                return True
    return False


def _gen_attempt():
    rng = random.Random()
    H = rng.randint(36, 44)
    W = rng.randint(36, 44)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    seg = _choose_segment(H, W, rng)
    if seg is None:
        return None

    canvas = np.full((H, W), bg, dtype=int)
    _paste_segment(canvas, seg)

    info = _find_source_segment(canvas)
    if info is None:
        return None
    shifts = _stripe_shifts(info, H, W)
    if len(shifts) < MIN_STRIPES:
        return None

    if not _place_forced_green_blob(canvas, info, shifts, fg, bg):
        return None

    if seg["orient"] == "h":
        forbidden_rows = {int(seg["fixed"])}
        forbidden_cols = set()
    else:
        forbidden_rows = set()
        forbidden_cols = {int(seg["fixed"])}

    target_blobs = rng.randint(6, 10)
    placed = 1

    for _ in range(target_blobs * 8):
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

    gaps = _gap_lengths(shifts)
    if len(gaps) < 5 or tuple(gaps[:5]) != (1, 2, 3, 4, 5):
        return None
    if int(shifts[-1]) < MIN_DIAGONAL_DRIFT:
        return None

    out = main(canvas)
    changed_to_green = int(np.sum((canvas != bg) & (canvas != Y) & (out == G)))
    new_yellow = int(np.sum((out == Y) & (canvas != Y)))
    if changed_to_green < MIN_GREEN_RECOLOR:
        return None
    if new_yellow < MIN_NEW_YELLOW:
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
