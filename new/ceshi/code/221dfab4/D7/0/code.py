# Variant of 221dfab4
# Dimension: D7 — 终止条件 (dst)
# Change: 推进至对侧边缘——条纹一直画到网格的另一边... → 碰到前景色块即停止

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_sprite

# Design analysis for the D7 variant:
# a) Concrete behavior:
#    The yellow border segment is still the seed. Stripes still advance every 2 cells
#    with the Y-Y-G cycle and still clear the 1-cell gap inside the segment span.
#    The change is that the advancing front now treats existing foreground blobs as
#    obstacles. Before drawing a stripe, the solver checks the original input line that
#    the front is about to traverse. Yellow stripes and gap clearing only care about the
#    seed span; green stripes still inspect the full row/column because the original task
#    interacted with full rows/columns on green turns. If the inspected line already has
#    foreground, propagation stops immediately and that obstacle stays untouched.
#    Example: a top-edge segment at cols 10..13 with a blob on row 6. Original logic would
#    continue through row 6 and recolor that row on the green turn; the variant paints the
#    earlier stripes, then halts before row 6, so row 6 and everything beyond remain as-is.
# b) Complexity calibration:
#    If blobs were allowed to miss the trajectory, the task would visually collapse back to
#    the original "paint to the far edge" behavior. The generator therefore forces a blocker
#    onto a reachable trajectory line several steps away from the seed and also places more
#    blobs beyond that blocker so the truncation is obvious.
# c) Implicit assumptions audited:
#    - Obstacle detection must read from the original grid, not the evolving output, or the
#      freshly painted stripes would incorrectly block later steps.
#    - Gap rows/cols can no longer be blanked unconditionally; we must test them first so an
#      obstacle sitting in the gap survives and halts the process.
#    - The original green "recolor full row/column" assignment is only safe after the full
#      line was verified obstacle-free; otherwise it would wrongly paint through a blocker.
# d) Placement legality:
#    The generator picks one specific stop step, pastes a blocker blob that intersects that
#    stop line, and forbids every earlier protected line that the solver checks. Extra blobs
#    are kept past the blocker and separated by padding so components stay distinct.
# e) Visual observability:
#    Each generated sample has multiple painted stripes before the stop and at least one more
#    potential stripe position beyond the blocker. Across a few examples, the human-visible
#    cue is that the stripe train truncates at the first blob instead of reaching the far edge.

Y = 4
G = 3
CYCLE = [Y, Y, G]


def _extract_seed(g: np.ndarray):
    H, W = g.shape
    bg = int(get_bg(g))

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
        orient = "h"
        start_pos = int(r0)
        span = list(range(int(cs.min()), int(cs.max()) + 1))
        dir1 = 1 if start_pos == 0 else -1
        limit = H
    elif c0 == c1:
        orient = "v"
        start_pos = int(c0)
        span = list(range(int(rs.min()), int(rs.max()) + 1))
        dir1 = 1 if start_pos == 0 else -1
        limit = W
    else:
        return None

    positions: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        positions.append(int(pos))
        step += 1

    return {
        "bg": bg,
        "smask": smask,
        "orient": orient,
        "start_pos": int(start_pos),
        "span": span,
        "dir1": int(dir1),
        "limit": int(limit),
        "positions": positions,
    }


def _line_has_obstacle(
    obstacle_mask: np.ndarray,
    orient: str,
    pos: int,
    span: list[int],
    *,
    full_line: bool,
) -> bool:
    if orient == "h":
        return bool(obstacle_mask[int(pos), :].any()) if full_line else bool(obstacle_mask[int(pos), span].any())
    return bool(obstacle_mask[:, int(pos)].any()) if full_line else bool(obstacle_mask[span, int(pos)].any())


def _simulate_variant(g: np.ndarray):
    info = _extract_seed(g)
    out = np.array(g, dtype=int).copy()
    if info is None:
        return out, {"stop_kind": "no_seed", "painted_steps": 0, "total_steps": 0}

    bg = int(info["bg"])
    orient = str(info["orient"])
    start_pos = int(info["start_pos"])
    span = list(info["span"])
    dir1 = int(info["dir1"])
    limit = int(info["limit"])
    total_steps = len(info["positions"])
    obstacle_mask = (g != bg) & (~info["smask"])

    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            return out, {
                "stop_kind": "edge",
                "stop_step": int(step),
                "blocked_line": None,
                "painted_steps": int(step),
                "total_steps": int(total_steps),
            }

        col = int(CYCLE[int(step) % 3])
        stripe_is_full = col == int(G)
        if _line_has_obstacle(obstacle_mask, orient, pos, span, full_line=stripe_is_full):
            return out, {
                "stop_kind": "stripe_obstacle",
                "stop_step": int(step),
                "blocked_line": int(pos),
                "painted_steps": int(step),
                "total_steps": int(total_steps),
            }

        if orient == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            out[pos, span] = int(col)
        else:
            if col == int(G):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            out[span, pos] = int(col)

        gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
        if 0 <= gap < int(limit):
            if _line_has_obstacle(obstacle_mask, orient, gap, span, full_line=False):
                return out, {
                    "stop_kind": "gap_obstacle",
                    "stop_step": int(step),
                    "blocked_line": int(gap),
                    "painted_steps": int(step) + 1,
                    "total_steps": int(total_steps),
                }
            if orient == "h":
                out[gap, span] = int(bg)
            else:
                out[span, gap] = int(bg)

        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out, _ = _simulate_variant(g)
    return out


def _random_blob_sprite(rng: random.Random, bg: int, fg: int) -> np.ndarray | None:
    for _ in range(50):
        bh = rng.randint(2, 8)
        bw = rng.randint(2, 8)
        density = rng.uniform(0.45, 0.85)
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

        mask = (sprite != bg)
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[r0 : r1 + 1, c0 : c1 + 1]

        pix = int(np.sum(sprite != bg))
        if 5 <= pix <= 45:
            return sprite
    return None


def _candidate_locations(H: int, W: int, sh: int, sw: int, border: int = 1) -> list[tuple[int, int]]:
    return [(r, c) for r in range(border, H - border - sh + 1) for c in range(border, W - border - sw + 1)]


def _touches_forbidden_lines(
    pr: int,
    pc: int,
    sprite_mask: np.ndarray,
    orient: str,
    span_set: set[int],
    full_lines: set[int],
    span_lines: set[int],
) -> bool:
    sh, sw = sprite_mask.shape
    if orient == "h":
        for dr in range(sh):
            abs_r = int(pr + dr)
            row_mask = sprite_mask[dr]
            if not bool(row_mask.any()):
                continue
            if abs_r in full_lines:
                return True
            if abs_r in span_lines:
                cols = np.where(row_mask)[0]
                for dc in cols:
                    if int(pc + dc) in span_set:
                        return True
    else:
        for dc in range(sw):
            abs_c = int(pc + dc)
            col_mask = sprite_mask[:, dc]
            if not bool(col_mask.any()):
                continue
            if abs_c in full_lines:
                return True
            if abs_c in span_lines:
                rows = np.where(col_mask)[0]
                for dr in rows:
                    if int(pr + dr) in span_set:
                        return True
    return False


def _place_sprite(
    rng: random.Random,
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
    allow_fn,
) -> bool:
    H, W = canvas.shape
    sh, sw = sprite.shape
    if sh > H - 2 or sw > W - 2:
        return False

    sprite_mask = (sprite != bg)
    occ = (canvas != bg)
    occ_pad = ndimage.binary_dilation(occ, structure=np.ones((3, 3), dtype=bool), iterations=1)

    candidates = _candidate_locations(H, W, sh, sw, border=1)
    rng.shuffle(candidates)

    for pr, pc in candidates:
        if bool((occ_pad[pr : pr + sh, pc : pc + sw] & sprite_mask).any()):
            continue
        if not allow_fn(pr, pc, sprite_mask):
            continue
        patch = canvas[pr : pr + sh, pc : pc + sw]
        patch[sprite_mask] = sprite[sprite_mask]
        return True
    return False


def _line_sets_before_stop(info: dict, stop_step: int) -> tuple[set[int], set[int]]:
    full_lines: set[int] = set()
    span_lines: set[int] = set()

    start_pos = int(info["start_pos"])
    dir1 = int(info["dir1"])
    limit = int(info["limit"])
    for step in range(int(stop_step)):
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if int(CYCLE[int(step) % 3]) == int(G):
            full_lines.add(int(pos))
        else:
            span_lines.add(int(pos))

        gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
        if 0 <= gap < int(limit):
            span_lines.add(int(gap))

    return full_lines, span_lines


def _is_past_barrier(index: int, blocked_line: int, dir1: int) -> bool:
    return int(dir1) * (int(index) - int(blocked_line)) > 0


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (3, 4)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 7)

    grid = np.full((H, W), bg, dtype=int)
    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_row = 0 if edge == "top" else H - 1
        grid[edge_row, span] = int(Y)
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_col = 0 if edge == "left" else W - 1
        grid[span, edge_col] = int(Y)

    info = _extract_seed(grid)
    if info is None:
        return None

    total_steps = len(info["positions"])
    if total_steps < 6:
        return None

    stop_step = rng.randint(2, total_steps - 2)
    stop_line = int(info["positions"][int(stop_step)])
    stop_color = int(CYCLE[int(stop_step) % 3])
    orient = str(info["orient"])
    dir1 = int(info["dir1"])
    span_set = set(int(x) for x in info["span"])
    full_lines, span_lines = _line_sets_before_stop(info, stop_step)

    canvas = grid.copy()

    blocker = _random_blob_sprite(rng, bg, fg)
    if blocker is None:
        return None

    def allow_blocker(pr: int, pc: int, sprite_mask: np.ndarray) -> bool:
        if _touches_forbidden_lines(pr, pc, sprite_mask, orient, span_set, full_lines, span_lines):
            return False

        hits_stop = False
        sh, sw = sprite_mask.shape
        if orient == "h":
            for dr in range(sh):
                abs_r = int(pr + dr)
                if abs_r != int(stop_line):
                    continue
                cols = np.where(sprite_mask[dr])[0]
                for dc in cols:
                    abs_c = int(pc + dc)
                    if int(abs_c) in span_set:
                        hits_stop = True
                        break
                if hits_stop:
                    break
        else:
            for dc in range(sw):
                abs_c = int(pc + dc)
                if abs_c != int(stop_line):
                    continue
                rows = np.where(sprite_mask[:, dc])[0]
                for dr in rows:
                    abs_r = int(pr + dr)
                    if int(abs_r) in span_set:
                        hits_stop = True
                        break
                if hits_stop:
                    break

        if not hits_stop:
            return False

        if orient == "h":
            rows = {int(pr + dr) for dr in range(sh) if bool(sprite_mask[dr].any())}
            return all(abs_r == int(stop_line) or _is_past_barrier(abs_r, stop_line, dir1) for abs_r in rows)
        cols = {int(pc + dc) for dc in range(sw) if bool(sprite_mask[:, dc].any())}
        return all(abs_c == int(stop_line) or _is_past_barrier(abs_c, stop_line, dir1) for abs_c in cols)

    if not _place_sprite(rng, canvas, blocker, bg, allow_blocker):
        return None

    target_blobs = rng.randint(5, 8)
    placed = 1
    for _ in range(40):
        if placed >= target_blobs:
            break
        sprite = _random_blob_sprite(rng, bg, fg)
        if sprite is None:
            continue

        def allow_extra(pr: int, pc: int, sprite_mask: np.ndarray) -> bool:
            if _touches_forbidden_lines(pr, pc, sprite_mask, orient, span_set, full_lines, span_lines):
                return False
            sh, sw = sprite_mask.shape
            if orient == "h":
                rows = {int(pr + dr) for dr in range(sh) if bool(sprite_mask[dr].any())}
                return rows and all(_is_past_barrier(abs_r, stop_line, dir1) for abs_r in rows)
            cols = {int(pc + dc) for dc in range(sw) if bool(sprite_mask[:, dc].any())}
            return cols and all(_is_past_barrier(abs_c, stop_line, dir1) for abs_c in cols)

        if _place_sprite(rng, canvas, sprite, bg, allow_extra):
            placed += 1

    fg_mask = (canvas != bg) & (canvas != int(Y))
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if int(get_bg(canvas)) != int(bg):
        return None

    out, meta = _simulate_variant(canvas)
    if meta["stop_kind"] == "edge":
        return None
    if int(meta["stop_step"]) != int(stop_step):
        return None
    if int(meta["total_steps"]) - int(meta["painted_steps"]) < 1:
        return None

    if stop_color == int(G):
        if orient == "h" and not bool(fg_mask[int(stop_line), :].any()):
            return None
        if orient == "v" and not bool(fg_mask[:, int(stop_line)].any()):
            return None

    if not np.array_equal(main(canvas), out):
        return None

    return canvas


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
