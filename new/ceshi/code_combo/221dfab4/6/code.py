# Combo variant of 221dfab4 (index 6)
# combo_spec: D1/0+D7/1
# Name: 对角推进+距离编码终止

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
UTILS_DIR = ROOT / "arc2_solver_select"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location


# Design analysis for the fused variant:
# 3a) Concrete behavior and trace
# - Original 221dfab4 keeps the yellow segment's span fixed and marches straight inward:
#   a top-edge segment on cols 4..7 paints rows 0,2,4,... on those same cols, clears
#   rows 1,3,5,... on those same cols, and every third stripe recolors the whole row green.
# - Fused rule changes both the travel geometry and the stop rule, but on the same distance axis:
#   the copied segment now moves diagonally, so shift `s` means row and column both move by `s`;
#   stripe shifts are still even (0,2,4,...) and gap shifts are odd (1,3,5,...). The stripe train
#   stops when the diagonal shift exceeds `segment_length * 3`.
# - Small trace: top-edge segment on cols 4..7 has length 4, so encoded stop = 12.
#   Original base task: stripes at (row 0, cols 4..7), (2, 4..7), (4, 4..7), ... to the far edge.
#   D1-only task:      stripes at (row 0, cols 4..7), (2, 6..9), (4, 8..11), ... to the far edge.
#   Fused task here:   same diagonal positions, but only shifts 0,2,4,6,8,10,12 are painted;
#                      gaps are at shifts 1,3,5,7,9,11. The first legacy-only stripe would be
#                      shift 14 at (row 14, cols 18..21), and that stripe is omitted.
# - Difference summary:
#   straight -> diagonal changes where every copied span lands;
#   to-edge -> length-coded stop changes how many diagonal copies exist.
#   They are coupled because both use the same diagonal shift coordinate.
# 3b) Complexity calibration
# - The distance factor is fixed at 3, keeping only one numeric mapping for a human to infer.
# - Segment lengths are sampled from a small range, but the generator forces enough runway so
#   several diagonal stripes are visible before the stop and at least one additional legacy stripe
#   would still fit afterward. This prevents collapse back to either the original straight version
#   or the diagonal-to-edge single variant.
# - The generator also places a mandatory "future blob" on the first post-stop legacy stripe and a
#   mandatory "green-hit blob" on a reachable green row/column outside the stripe span, so both
#   fused dimensions create visible evidence rather than acting only on empty background.
# 3c) Implicit-assumption audit of the original solver
# - Original `pos` was a single row or col index. That assumed no lateral drift. The fused solver
#   must instead recover both the primary coordinate and the shifted span for every stripe/gap.
# - Original `span` could be reused verbatim forever. That is invalid once the pattern moves
#   diagonally; every shift needs a translated span.
# - Original loop stopped only when falling off-grid. That fails after D7/1 because a valid in-bounds
#   stripe can still be forbidden once `shift > seg_len * 3`.
# - Original gap clearing could always fire for the next row/column. In the fused rule, the gap must
#   also obey the encoded stop, otherwise cells beyond the frontier would be erased incorrectly.
# - Original green recolor assumes the affected line is obvious from `pos`. In the fused rule the
#   touched row/column must be derived from the diagonal anchor first, then the whole row/column is
#   recolored as before.
# 3d) Placement legality
# - The source remains exactly one straight 4-connected yellow border segment that never touches a
#   corner. Generator sampling and solver detection share the same edge-direction convention.
# - The chosen segment always leaves border-safe space for all stripes up to the encoded stop and for
#   the first omitted legacy stripe beyond the stop. This keeps all intended placements in bounds.
# - Blobs are placed with one-cell padding, so input components do not overlap each other or the
#   source segment. The two mandatory blobs are placed by explicit candidate search, then extra blobs
#   are scattered only into free space.
# 3e) Visual observability
# - A human sees a staircase of Y/Y/G bands rather than a straight stack, which exposes D1/0.
# - The staircase terminates early even though more diagonal runway is available, and the future blob
#   remains unchanged exactly where an additional diagonal stripe would have landed, which exposes D7/1.
# - The green-hit blob turns partly green on a whole row/column outside the stripe span, preserving the
#   original task's row/column flood cue while keeping the diagonal drift visible.
# 9) New solver assumptions and generator enforcement
# - Assumption: the yellow source is the largest border-yellow component and is perfectly straight.
#   Enforced by generating exactly one such segment.
# - Assumption: diagonal direction is determined only by the touched border:
#   top/left -> down-right, bottom/right -> up-left. Generator uses the same convention.
# - Assumption: the encoded stop is measured in unit diagonal shifts, with painted stripes only on
#   even shifts. Generator samples sizes so multiple even shifts fit and the next even shift also fits.
# - Assumption: at least one green stripe intersects foreground and at least one post-stop legacy stripe
#   would intersect foreground. Enforced by the two mandatory blobs plus output comparisons.

YELLOW = 4
GREEN = 3
DIST_FACTOR = 3
CYCLE = (YELLOW, YELLOW, GREEN)


def _find_source_segment(g: np.ndarray) -> dict[str, object] | None:
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == YELLOW) & border
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
        edge = "top" if int(r0) == 0 else "bottom"
        return {
            "orient": "h",
            "fixed": int(r0),
            "span": list(range(int(c0), int(c1) + 1)),
            "dr": 1 if edge == "top" else -1,
            "dc": 1 if edge == "top" else -1,
            "edge": edge,
            "seg_len": int(c1 - c0 + 1),
        }

    if c0 == c1:
        edge = "left" if int(c0) == 0 else "right"
        return {
            "orient": "v",
            "fixed": int(c0),
            "span": list(range(int(r0), int(r1) + 1)),
            "dr": 1 if edge == "left" else -1,
            "dc": 1 if edge == "left" else -1,
            "edge": edge,
            "seg_len": int(r1 - r0 + 1),
        }

    return None


def _shifted(values: list[int], shift: int, direction: int) -> list[int]:
    return [int(v + direction * shift) for v in values]


def _diagonal_cells(info: dict[str, object], shift: int) -> tuple[int, list[int]]:
    if str(info["orient"]) == "h":
        row = int(info["fixed"]) + int(info["dr"]) * int(shift)
        cols = _shifted(list(info["span"]), int(shift), int(info["dc"]))
        return int(row), cols

    col = int(info["fixed"]) + int(info["dc"]) * int(shift)
    rows = _shifted(list(info["span"]), int(shift), int(info["dr"]))
    return int(col), rows


def _straight_cells(info: dict[str, object], shift: int) -> tuple[int, list[int]]:
    if str(info["orient"]) == "h":
        row = int(info["fixed"]) + int(info["dr"]) * int(shift)
        return int(row), list(info["span"])

    col = int(info["fixed"]) + int(info["dc"]) * int(shift)
    return int(col), list(info["span"])


def _cells_in_bounds(info: dict[str, object], shift: int, H: int, W: int, *, diagonal: bool) -> bool:
    coord, values = (_diagonal_cells(info, shift) if diagonal else _straight_cells(info, shift))
    if str(info["orient"]) == "h":
        return 0 <= int(coord) < H and all(0 <= int(c) < W for c in values)
    return 0 <= int(coord) < W and all(0 <= int(r) < H for r in values)


def _paint_pattern(
    g: np.ndarray,
    *,
    info: dict[str, object] | None = None,
    max_shift: int | None,
    diagonal: bool,
) -> np.ndarray:
    out = np.array(g, dtype=int).copy()
    H, W = out.shape
    bg = int(get_bg(out))
    info = _find_source_segment(out) if info is None else info
    if info is None:
        return out

    step = 0
    while True:
        shift = 2 * step
        if max_shift is not None and int(shift) > int(max_shift):
            break
        if not _cells_in_bounds(info, shift, H, W, diagonal=diagonal):
            break

        color = int(CYCLE[step % len(CYCLE)])
        coord, values = (_diagonal_cells(info, shift) if diagonal else _straight_cells(info, shift))

        if str(info["orient"]) == "h":
            row = int(coord)
            cols = np.array(values, dtype=int)
            if color == GREEN:
                out[row, :] = np.where(out[row, :] != bg, GREEN, out[row, :])
            out[row, cols] = color

            gap_shift = int(shift + 1)
            if max_shift is None or gap_shift <= int(max_shift):
                gap_row, gap_cols = (_diagonal_cells(info, gap_shift) if diagonal else _straight_cells(info, gap_shift))
                if 0 <= int(gap_row) < H:
                    valid_gap_cols = [int(c) for c in gap_cols if 0 <= int(c) < W]
                    if valid_gap_cols:
                        out[int(gap_row), np.array(valid_gap_cols, dtype=int)] = bg
        else:
            col = int(coord)
            rows = np.array(values, dtype=int)
            if color == GREEN:
                out[:, col] = np.where(out[:, col] != bg, GREEN, out[:, col])
            out[rows, col] = color

            gap_shift = int(shift + 1)
            if max_shift is None or gap_shift <= int(max_shift):
                gap_col, gap_rows = (_diagonal_cells(info, gap_shift) if diagonal else _straight_cells(info, gap_shift))
                if 0 <= int(gap_col) < W:
                    valid_gap_rows = [int(r) for r in gap_rows if 0 <= int(r) < H]
                    if valid_gap_rows:
                        out[np.array(valid_gap_rows, dtype=int), int(gap_col)] = bg

        step += 1

    return out


def _next_stripe_shift_after(stop_shift: int) -> int:
    return 2 * (int(stop_shift) // 2 + 1)


def _reachable_green_shifts(info: dict[str, object], max_shift: int, H: int, W: int) -> list[int]:
    shifts: list[int] = []
    step = 0
    while True:
        shift = 2 * step
        if int(shift) > int(max_shift):
            break
        if not _cells_in_bounds(info, shift, H, W, diagonal=True):
            break
        if step % len(CYCLE) == 2:
            shifts.append(int(shift))
        step += 1
    return shifts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    info = _find_source_segment(g)
    if info is None:
        return g.copy()
    stop_shift = int(info["seg_len"]) * DIST_FACTOR
    return _paint_pattern(g, info=info, max_shift=stop_shift, diagonal=True)


def _legacy_diagonal_to_edge(grid: np.ndarray) -> np.ndarray:
    return _paint_pattern(np.array(grid, dtype=int), max_shift=None, diagonal=True)


def _straight_distance_stop(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    info = _find_source_segment(g)
    if info is None:
        return g.copy()
    stop_shift = int(info["seg_len"]) * DIST_FACTOR
    return _paint_pattern(g, info=info, max_shift=stop_shift, diagonal=False)


def _box_fits(canvas: np.ndarray, bg: int, r0: int, c0: int, h: int, w: int, *, padding: int = 1) -> bool:
    H, W = canvas.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False
    if np.any(canvas[int(r0) : int(r0 + h), int(c0) : int(c0 + w)] != int(bg)):
        return False

    rr0 = max(0, int(r0) - int(padding))
    cc0 = max(0, int(c0) - int(padding))
    rr1 = min(H, int(r0 + h) + int(padding))
    cc1 = min(W, int(c0 + w) + int(padding))
    if np.any(canvas[rr0:rr1, cc0:cc1] != int(bg)):
        return False
    return True


def _paste_box(canvas: np.ndarray, fg: int, r0: int, c0: int, h: int, w: int) -> None:
    canvas[int(r0) : int(r0 + h), int(c0) : int(c0 + w)] = int(fg)


def _place_green_hit_blob(
    canvas: np.ndarray,
    info: dict[str, object],
    green_shift: int,
    fg: int,
    bg: int,
    rng: random.Random,
) -> bool:
    H, W = canvas.shape
    coord, values = _diagonal_cells(info, green_shift)
    target = set(int(v) for v in values)
    candidates: list[tuple[int, int]] = []

    if str(info["orient"]) == "h":
        row = int(coord)
        r0 = int(row - 1)
        if 1 <= r0 <= H - 4:
            for c0 in range(1, W - 3):
                cols = {int(c0), int(c0 + 1), int(c0 + 2)}
                if cols & target:
                    continue
                if _box_fits(canvas, bg, r0, c0, 3, 3):
                    candidates.append((int(r0), int(c0)))
    else:
        col = int(coord)
        c0 = int(col - 1)
        if 1 <= c0 <= W - 4:
            for r0 in range(1, H - 3):
                rows = {int(r0), int(r0 + 1), int(r0 + 2)}
                if rows & target:
                    continue
                if _box_fits(canvas, bg, r0, c0, 3, 3):
                    candidates.append((int(r0), int(c0)))

    if not candidates:
        return False
    r0, c0 = rng.choice(candidates)
    _paste_box(canvas, fg, r0, c0, 3, 3)
    return True


def _place_future_blob(
    canvas: np.ndarray,
    info: dict[str, object],
    future_shift: int,
    fg: int,
    bg: int,
    rng: random.Random,
) -> bool:
    H, W = canvas.shape
    coord, values = _diagonal_cells(info, future_shift)
    lane = [int(v) for v in values]
    candidates: list[tuple[int, int]] = []

    if str(info["orient"]) == "h":
        row = int(coord)
        row_opts = []
        if 1 <= row <= H - 3:
            row_opts.append(int(row))
        if 1 <= row - 1 <= H - 3:
            row_opts.append(int(row - 1))
        for r0 in row_opts:
            for c0 in lane:
                if c0 < 1 or c0 + 1 >= W - 0:
                    continue
                if int(c0 + 1) not in lane:
                    continue
                if _box_fits(canvas, bg, r0, c0, 2, 2):
                    candidates.append((int(r0), int(c0)))
    else:
        col = int(coord)
        col_opts = []
        if 1 <= col <= W - 3:
            col_opts.append(int(col))
        if 1 <= col - 1 <= W - 3:
            col_opts.append(int(col - 1))
        for c0 in col_opts:
            for r0 in lane:
                if r0 < 1 or r0 + 1 >= H - 0:
                    continue
                if int(r0 + 1) not in lane:
                    continue
                if _box_fits(canvas, bg, r0, c0, 2, 2):
                    candidates.append((int(r0), int(c0)))

    if not candidates:
        return False
    r0, c0 = rng.choice(candidates)
    _paste_box(canvas, fg, r0, c0, 2, 2)
    return True


def _scatter_extra_boxes(canvas: np.ndarray, bg: int, fg: int, rng: random.Random, target_count: int) -> int:
    placed = 0
    for _ in range(target_count * 8):
        if placed >= target_count:
            break
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        patch = np.full((h, w), int(fg), dtype=int)
        try:
            r0, c0 = random_free_location(canvas, patch, transparent=bg, border=1, padding=1)
        except ValueError:
            continue
        _paste_box(canvas, fg, r0, c0, h, w)
        placed += 1
    return int(placed)


def _choose_segment(H: int, W: int, rng: random.Random) -> dict[str, object] | None:
    candidates: list[tuple[str, int, list[int]]] = []
    for edge in ("top", "bottom", "left", "right"):
        for seg_len in range(3, 7):
            stop_shift = int(seg_len) * DIST_FACTOR
            future_shift = _next_stripe_shift_after(stop_shift)

            if edge == "top":
                max_start = W - seg_len - future_shift - 1
                if H >= future_shift + 2 and max_start >= 1:
                    candidates.append((edge, seg_len, list(range(1, int(max_start) + 1))))
            elif edge == "bottom":
                min_start = future_shift + 1
                max_start = W - seg_len - 1
                if H >= future_shift + 2 and min_start <= max_start:
                    candidates.append((edge, seg_len, list(range(int(min_start), int(max_start) + 1))))
            elif edge == "left":
                max_start = H - seg_len - future_shift - 1
                if W >= future_shift + 2 and max_start >= 1:
                    candidates.append((edge, seg_len, list(range(1, int(max_start) + 1))))
            else:
                min_start = future_shift + 1
                max_start = H - seg_len - 1
                if W >= future_shift + 2 and min_start <= max_start:
                    candidates.append((edge, seg_len, list(range(int(min_start), int(max_start) + 1))))

    if not candidates:
        return None

    edge, seg_len, starts = rng.choice(candidates)
    start = rng.choice(starts)

    if edge in ("top", "bottom"):
        return {
            "edge": edge,
            "orient": "h",
            "fixed": 0 if edge == "top" else H - 1,
            "span": list(range(int(start), int(start + seg_len))),
        }

    return {
        "edge": edge,
        "orient": "v",
        "fixed": 0 if edge == "left" else W - 1,
        "span": list(range(int(start), int(start + seg_len))),
    }


def _paste_segment(grid: np.ndarray, seg: dict[str, object]) -> None:
    if str(seg["orient"]) == "h":
        grid[int(seg["fixed"]), np.array(list(seg["span"]), dtype=int)] = YELLOW
    else:
        grid[np.array(list(seg["span"]), dtype=int), int(seg["fixed"])] = YELLOW


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    H = rng.randint(28, 34)
    W = rng.randint(28, 34)

    available = [c for c in range(10) if c not in (GREEN, YELLOW)]
    bg, fg = rng.sample(available, 2)

    seg = _choose_segment(H, W, rng)
    if seg is None:
        return None

    canvas = np.full((H, W), bg, dtype=int)
    _paste_segment(canvas, seg)

    info = _find_source_segment(canvas)
    if info is None:
        return None

    stop_shift = int(info["seg_len"]) * DIST_FACTOR
    future_shift = _next_stripe_shift_after(stop_shift)
    if not _cells_in_bounds(info, future_shift, H, W, diagonal=True):
        return None

    green_shifts = _reachable_green_shifts(info, stop_shift, H, W)
    if not green_shifts:
        return None

    if not _place_green_hit_blob(canvas, info, green_shifts[-1], fg, bg, rng):
        return None
    if not _place_future_blob(canvas, info, future_shift, fg, bg, rng):
        return None

    extra_needed = rng.randint(3, 6)
    if _scatter_extra_boxes(canvas, bg, fg, rng, extra_needed) < 3:
        return None

    fg_mask = (canvas != bg) & (canvas != YELLOW)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if int(get_bg(canvas)) != int(bg):
        return None

    combo_out = main(canvas)
    diag_to_edge = _legacy_diagonal_to_edge(canvas)
    straight_stop = _straight_distance_stop(canvas)

    if np.array_equal(combo_out, diag_to_edge):
        return None
    if np.array_equal(combo_out, straight_stop):
        return None

    green_recolored = int(np.sum((canvas != bg) & (canvas != YELLOW) & (combo_out == GREEN)))
    stop_visible = int(np.sum((canvas != bg) & (combo_out != diag_to_edge)))
    diagonal_visible = int(np.sum(combo_out != straight_stop))
    if green_recolored < 4:
        return None
    if stop_visible < 2:
        return None
    if diagonal_visible < max(8, int(info["seg_len"]) * 2):
        return None

    if not np.array_equal(main(canvas), combo_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
