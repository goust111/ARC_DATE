# Variant of 221dfab4
# Dimension: D11 — 多源与推进顺序 (multi_step)
# Change: 单个源从一侧向对侧单次推进，无迭代或链式步骤... → 反射推进（到对侧后反弹继续）
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


# Design analysis for the D11 reflection variant
# ----------------------------------------------
# a) Concrete behavior and trace
#    Keep the original source detection, stripe width/span, and Y-Y-G cycle. The only
#    change is the travel order of stripe positions along the normal direction:
#    first march outward exactly as before, then mirror those stripe positions back
#    toward the source side while continuing the same color cycle.
#    Example with a top-edge source on rows [0, 2, 4, 6]:
#      original: 0(Y), 2(Y), 4(G), 6(Y)
#      variant : 0(Y), 2(Y), 4(G), 6(Y), 4(Y), 2(G), 0(Y)
#    The returning pass repaints earlier stripe rows/cols, while any earlier green
#    full-row/full-column recoloring remains visible off the source span. This creates
#    the intended double-layer "reflection" look without changing object semantics.
# b) Complexity calibration
#    The variant must be visibly different from the original one-way march. Large
#    grids (25-30 by 22-30) already provide many stripe positions, so the return pass
#    changes multiple previously painted rows/cols. Generator rejection also enforces
#    a nontrivial difference against the original single-pass output.
# c) Hidden assumptions in the original solver
#    - `pos = start + dir * (2 * step)` assumes strictly monotone travel; that fails
#      once the pattern bounces.
#    - `gap = start + dir * (2 * step + 1)` also assumes monotone travel; after
#      reflection the correct gap is determined by each pair of consecutive stripe
#      positions, plus one optional far-edge terminal gap before the bounce.
#    - Unconditional gap clearing remains safe because stripe cells stay on one parity
#      and gap cells stay on the opposite parity, so a reflected pass only revisits
#      earlier gaps, never overwrites a stripe cell with background.
#    - Green repainting must still run on the evolving output, not the input, so that
#      reflected green passes can recolor earlier yellow stripes and blobs.
# d) Placement legality
#    Generator keeps the yellow segment as the only border-yellow component, keeps
#    blobs away from the border, and uses `random_free_location` for non-overlap.
#    The solver only assumes one straight border segment; generator enforces that.
# e) Visual observability
#    A human should see that the output is not just a one-way sequence because rows or
#    columns near the source are repainted late in the cycle. Some of those rows/cols
#    retain green off-span object recoloring even when the final local stripe is yellow.
#    Generator rejects samples whose reflected output is too close to the original
#    single-pass output, preventing invisible degeneracies.

Y = 4
G = 3
CYCLE = [Y, Y, G]


def _detect_source(g: np.ndarray):
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
        cols = sorted(int(c) for c in cs)
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span": list(range(int(cols[0]), int(cols[-1]) + 1)),
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": H,
        }
    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span": list(range(int(rows[0]), int(rows[-1]) + 1)),
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
        }
    return None


def _forward_positions(start_pos: int, dir1: int, limit: int) -> tuple[list[int], int | None]:
    positions: list[int] = []
    pos = int(start_pos)
    while 0 <= pos < int(limit):
        positions.append(int(pos))
        pos += 2 * int(dir1)

    if not positions:
        return positions, None

    terminal_gap = positions[-1] + int(dir1)
    if not (0 <= terminal_gap < int(limit)):
        terminal_gap = None
    return positions, terminal_gap


def _trajectory(start_pos: int, dir1: int, limit: int, *, reflect: bool) -> tuple[list[int], list[int], int | None]:
    forward, terminal_gap = _forward_positions(start_pos, dir1, limit)
    if reflect and len(forward) >= 2:
        positions = forward + forward[-2::-1]
    else:
        positions = forward
    gaps = [int((positions[i] + positions[i + 1]) // 2) for i in range(len(positions) - 1)]
    return positions, gaps, terminal_gap


def _apply_stripe(out: np.ndarray, bg: int, orient: str, span: list[int], pos: int, col: int) -> None:
    if orient == "h":
        if int(col) == int(G):
            out[int(pos), :] = np.where(out[int(pos), :] != int(bg), int(G), out[int(pos), :])
        out[int(pos), span] = int(col)
    else:
        if int(col) == int(G):
            out[:, int(pos)] = np.where(out[:, int(pos)] != int(bg), int(G), out[:, int(pos)])
        out[span, int(pos)] = int(col)


def _clear_gap(out: np.ndarray, bg: int, orient: str, span: list[int], gap: int | None) -> None:
    if gap is None:
        return
    if orient == "h":
        out[int(gap), span] = int(bg)
    else:
        out[span, int(gap)] = int(bg)


def _solve(grid: np.ndarray, *, reflect: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    src = _detect_source(g)
    if src is None:
        return out

    positions, gaps, terminal_gap = _trajectory(
        src["start_pos"], src["dir1"], src["limit"], reflect=reflect
    )
    for step, pos in enumerate(positions):
        col = int(CYCLE[int(step) % len(CYCLE)])
        _apply_stripe(out, bg, src["orient"], src["span"], int(pos), int(col))
        if step < len(gaps):
            _clear_gap(out, bg, src["orient"], src["span"], int(gaps[step]))

    _clear_gap(out, bg, src["orient"], src["span"], terminal_gap)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflecting variant of 221dfab4.

    First propagate the original Y-Y-G stripe sequence from the yellow border
    segment toward the opposite side. After the outward pass is exhausted, mirror
    the stripe positions back toward the source side and keep advancing the same
    color cycle through that return leg.
    """
    return _solve(grid, reflect=True)


def _reflection_impact(inp: np.ndarray) -> int:
    reflected = _solve(inp, reflect=True)
    single_pass = _solve(inp, reflect=False)
    return int(np.count_nonzero(reflected != single_pass))


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
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
        edge_row = 0 if edge == "top" else H - 1
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1

    grid = np.full((H, W), bg, dtype=int)
    if orient == "h":
        grid[edge_row, span] = int(Y)
    else:
        grid[span, edge_col] = int(Y)

    canvas = grid.copy()
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

        if orient == "h" and edge_row in blob_rows:
            continue
        if orient == "v" and edge_col in blob_cols:
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

    if get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None

    impact = _reflection_impact(inp)
    if impact < max(2 * len(span), 8):
        return None

    return inp


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
