# Combo variant of 221dfab4 (index 1)
# combo_spec: D2/2+D3/2
# Name: 递增步长+线段长度编码周期

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the fused D2/2 + D3/2 variant:
# a) Concrete behavior and trace:
#    Start from the same single yellow border segment as the source task, but fuse
#    both changed dimensions into one trajectory rule. Example: a top-edge segment
#    spans cols 6..9 and has length L=4. The original task paints rows
#    0, 2, 4, 6, 8, ... with colors Y, Y, G, Y, Y, G, ...
#    The fused variant paints rows 0, 2, 5, 9, 14, 20, ... because the motion
#    increments are +2, +3, +4, +5, +6, ... so the gaps are 1, 2, 3, 4, 5, ...
#    The color rule also changes: for L=4 the cycle is Y, Y, Y, Y, G and then
#    repeats. So in the example rows 0,2,5,9 are yellow and row 14 is the first
#    green flood row. Relative to the source task, both "where is the next stripe"
#    and "when does green happen" change, and they interact: the first green row is
#    pushed much farther away because it is delayed both by a longer yellow cycle
#    and by growing stripe spacing.
# b) Complexity calibration:
#    The generator keeps segment length in {3,4,5}. This avoids the degenerate
#    L=2 case that would recreate the original Y,Y,G timing, while keeping only a
#    few reusable cycle lengths for human inference. Grid sizes are chosen so the
#    output exposes at least three distinct gaps and at least one green stripe.
#    A witness blob is forced onto the first green row/column outside the segment
#    span, so the "green floods the whole row/column" effect stays visible even
#    when later random blobs miss the green trajectories.
# c) Hidden-assumption audit:
#    - The original solver assumed constant stride via `start + 2*step`. That is no
#      longer valid; we replace it with cumulative increments 2,3,4,...
#    - The original gap logic cleared exactly one cell. With increasing stride, all
#      cells between two successive stripe positions inside the segment span must be
#      reset to background.
#    - Measuring the yellow span by bbox is no longer enough unless the border
#      component is a straight contiguous run, because its exact length L drives the
#      color period. `_find_segment_info()` now rejects non-contiguous runs.
#    - The green-flood branch assumes colors 3 and 4 are reserved for generated
#      stripes, not pre-existing blobs. The generator enforces that by choosing
#      foreground colors outside {3,4}.
#    - `_apply_pattern()` assumes consecutive stripe positions are distinct and at
#      least two cells apart, so gap clearing does not erase a later stripe. The
#      increasing-step rule guarantees that because every increment is >= 2.
# d) Placement legality:
#    The yellow seed stays on one border and avoids corners. Random blobs are placed
#    with `border=1` and `padding=1`, so they do not overlap each other or the seed.
#    The witness blob is placed entirely inside the grid, outside the seed span, and
#    with a one-cell empty moat so later random placements cannot merge into it.
# e) Visual observability:
#    A solver looking at 2-4 examples can read both axes directly from geometry:
#    stripe gaps expand 1,2,3,... instead of staying uniform, and the first green
#    event occurs only after counting the seed length L yellow stripes. Because the
#    witness sits on a green trajectory but outside the yellow span, the output makes
#    the delayed whole-row/column recoloring explicit rather than leaving it hidden.
# New solver assumptions guaranteed by `generate_input()`:
#    the seed is the unique straight yellow border run, the motion axis is long
#    enough to show three distinct gaps and one green stripe, and the fused output
#    differs from the source task and from each single-dimension variant.

Y = 4
G = 3


def _find_segment_info(g: np.ndarray):
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
    if pts.size == 0:
        return None

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        cols = sorted(int(c) for c in cs)
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span0": int(cols[0]),
            "span1": int(cols[-1]),
            "seg_len": int(len(cols)),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": int(H),
        }

    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span0": int(rows[0]),
            "span1": int(rows[-1]),
            "seg_len": int(len(rows)),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": int(W),
        }

    return None


def _stripe_positions(start_pos: int, dir1: int, limit: int, increasing_stride: bool) -> list[int]:
    positions: list[int] = []
    pos = int(start_pos)
    step = 0
    while 0 <= pos < int(limit):
        positions.append(int(pos))
        inc = int(step) + 2 if increasing_stride else 2
        pos += int(dir1) * int(inc)
        step += 1
    return positions


def _stripe_color(step: int, seg_len: int, length_coded_cycle: bool) -> int:
    if length_coded_cycle:
        return G if int(step) % (int(seg_len) + 1) == int(seg_len) else Y
    return [Y, Y, G][int(step) % 3]


def _apply_pattern(
    g: np.ndarray,
    info: dict[str, int | str],
    bg: int,
    *,
    increasing_stride: bool,
    length_coded_cycle: bool,
) -> np.ndarray:
    out = g.copy()

    orient = str(info["orient"])
    start_pos = int(info["start_pos"])
    span0 = int(info["span0"])
    span1 = int(info["span1"])
    seg_len = int(info["seg_len"])
    dir1 = int(info["dir1"])
    limit = int(info["limit"])

    pos = int(start_pos)
    step = 0
    while 0 <= pos < limit:
        col = int(_stripe_color(step, seg_len, length_coded_cycle))
        inc = int(step) + 2 if increasing_stride else 2
        next_pos = int(pos) + int(dir1) * int(inc)

        if orient == "h":
            if col == G:
                out[pos, :] = np.where(out[pos, :] != int(bg), G, out[pos, :])
            out[pos, span0 : span1 + 1] = col
            if 0 <= next_pos < limit:
                lo, hi = sorted((int(pos), int(next_pos)))
                if hi - lo > 1:
                    out[lo + 1 : hi, span0 : span1 + 1] = int(bg)
        else:
            if col == G:
                out[:, pos] = np.where(out[:, pos] != int(bg), G, out[:, pos])
            out[span0 : span1 + 1, pos] = col
            if 0 <= next_pos < limit:
                lo, hi = sorted((int(pos), int(next_pos)))
                if hi - lo > 1:
                    out[span0 : span1 + 1, lo + 1 : hi] = int(bg)

        pos = int(next_pos)
        step += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    info = _find_segment_info(g)
    if info is None:
        return g.copy()

    return _apply_pattern(
        g,
        info,
        bg,
        increasing_stride=True,
        length_coded_cycle=True,
    )


def _solve_original_style(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    info = _find_segment_info(g)
    if info is None:
        return g.copy()
    return _apply_pattern(
        g,
        info,
        bg,
        increasing_stride=False,
        length_coded_cycle=False,
    )


def _solve_d2_only(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    info = _find_segment_info(g)
    if info is None:
        return g.copy()
    return _apply_pattern(
        g,
        info,
        bg,
        increasing_stride=True,
        length_coded_cycle=False,
    )


def _solve_d3_only(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    info = _find_segment_info(g)
    if info is None:
        return g.copy()
    return _apply_pattern(
        g,
        info,
        bg,
        increasing_stride=False,
        length_coded_cycle=True,
    )


def _visible_gap_lengths(limit: int) -> tuple[int, ...]:
    positions = _stripe_positions(0, 1, limit, True)
    return tuple(int(positions[i + 1] - positions[i] - 1) for i in range(len(positions) - 1))


def _green_positions(info: dict[str, int | str]) -> list[int]:
    positions = _stripe_positions(int(info["start_pos"]), int(info["dir1"]), int(info["limit"]), True)
    seg_len = int(info["seg_len"])
    return [int(pos) for step, pos in enumerate(positions) if _stripe_color(step, seg_len, True) == G]


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not rows_any.any():
        return None
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[r0 : r1 + 1, c0 : c1 + 1]


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    for dr in range(h):
        for dc in range(w):
            if int(patch[dr, dc]) != int(bg):
                canvas[int(top) + dr, int(left) + dc] = int(patch[dr, dc])


def _line_has_offspan_fg(canvas: np.ndarray, orient: str, span0: int, span1: int, pos: int, bg: int) -> bool:
    if orient == "h":
        mask = canvas[int(pos), :] != int(bg)
        mask[int(span0) : int(span1) + 1] = False
        return bool(mask.any())
    mask = canvas[:, int(pos)] != int(bg)
    mask[int(span0) : int(span1) + 1] = False
    return bool(mask.any())


def _add_green_witness(
    canvas: np.ndarray,
    info: dict[str, int | str],
    green_pos: int,
    bg: int,
    fg: int,
    rng: random.Random,
) -> bool:
    patch = np.full((2, 2), int(fg), dtype=int)
    H, W = canvas.shape
    orient = str(info["orient"])
    span0 = int(info["span0"])
    span1 = int(info["span1"])

    if orient == "h":
        row_tops = [int(green_pos) - 1, int(green_pos)]
        left_choices = list(range(1, max(1, int(span0) - 1))) + list(range(int(span1) + 1, W - 2))
        rng.shuffle(left_choices)
        for top in row_tops:
            if not (1 <= top and top + 1 < H - 1):
                continue
            for left in left_choices:
                region = canvas[max(0, top - 1) : min(H, top + 3), max(0, left - 1) : min(W, left + 3)]
                if np.any(region != int(bg)):
                    continue
                _paste_patch(canvas, patch, top, left, bg)
                return True
        return False

    col_lefts = [int(green_pos) - 1, int(green_pos)]
    top_choices = list(range(1, max(1, int(span0) - 1))) + list(range(int(span1) + 1, H - 2))
    rng.shuffle(top_choices)
    for left in col_lefts:
        if not (1 <= left and left + 1 < W - 1):
            continue
        for top in top_choices:
            region = canvas[max(0, top - 1) : min(H, top + 3), max(0, left - 1) : min(W, left + 3)]
            if np.any(region != int(bg)):
                continue
            _paste_patch(canvas, patch, top, left, bg)
            return True
    return False


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(27, 33)
    W = rng.randint(27, 33)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 5)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        info = {
            "orient": "h",
            "start_pos": 0 if edge == "top" else H - 1,
            "span0": int(seg_start),
            "span1": int(seg_start + seg_len - 1),
            "seg_len": int(seg_len),
            "dir1": 1 if edge == "top" else -1,
            "limit": int(H),
        }
        forbidden_rows = {int(info["start_pos"])}
        forbidden_cols: set[int] = set()
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        info = {
            "orient": "v",
            "start_pos": 0 if edge == "left" else W - 1,
            "span0": int(seg_start),
            "span1": int(seg_start + seg_len - 1),
            "seg_len": int(seg_len),
            "dir1": 1 if edge == "left" else -1,
            "limit": int(W),
        }
        forbidden_rows = set()
        forbidden_cols = {int(info["start_pos"])}

    visible_gaps = _visible_gap_lengths(int(info["limit"]))
    if len(visible_gaps) < 3 or len(set(visible_gaps[:3])) < 3:
        return None

    green_positions = _green_positions(info)
    if not green_positions:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    if str(info["orient"]) == "h":
        grid[int(info["start_pos"]), int(info["span0"]) : int(info["span1"]) + 1] = Y
    else:
        grid[int(info["span0"]) : int(info["span1"]) + 1, int(info["start_pos"])] = Y

    canvas = grid.copy()
    if not _add_green_witness(canvas, info, int(green_positions[0]), int(bg), int(fg), rng):
        return None

    target_blobs = rng.randint(5, 8)
    attempts = 0
    while attempts < 120:
        attempts += 1
        bh = rng.randint(2, 9)
        bw = rng.randint(2, 9)
        density = rng.uniform(0.4, 0.82)

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

        sprite = _trim_sprite(sprite, int(bg))
        if sprite is None:
            continue

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 50:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=int(bg), border=1, padding=1)
        except ValueError:
            continue

        sprite_mask = sprite != int(bg)
        blob_rows = {int(pr + dr) for dr in range(sh) if sprite_mask[dr].any()}
        blob_cols = {int(pc + dc) for dc in range(sw) if sprite_mask[:, dc].any()}

        if blob_rows & forbidden_rows:
            continue
        if blob_cols & forbidden_cols:
            continue

        _paste_patch(canvas, sprite, int(pr), int(pc), int(bg))

        fg_mask = (canvas != int(bg)) & (canvas != Y)
        _lab, n_placed = label_mask(fg_mask, connectivity=4)
        if int(n_placed) >= int(target_blobs):
            break

    fg_mask = (canvas != int(bg)) & (canvas != Y)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if not _line_has_offspan_fg(
        canvas,
        str(info["orient"]),
        int(info["span0"]),
        int(info["span1"]),
        int(green_positions[0]),
        int(bg),
    ):
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
        return None

    if np.array_equal(out, _solve_original_style(inp)):
        return None
    if np.array_equal(out, _solve_d2_only(inp)):
        return None
    if np.array_equal(out, _solve_d3_only(inp)):
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
