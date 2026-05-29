# Variant of 221dfab4
# Dimension: D12 — 符号编码——线段属性的语义 (symbol_instructions)
# Change: 黄色线段仅编码位置和跨度信息，颜色固定为黄(4)和绿(3)... → 线段中嵌入绿色像素编码周期

from __future__ import annotations

from pathlib import Path
import sys
import random
import itertools

import numpy as np

HELPER_DIR = Path(__file__).resolve().parents[5] / "arc2_solver_select"
if str(HELPER_DIR) not in sys.path:
    sys.path.append(str(HELPER_DIR))

from arc_utils import get_bg, label_mask, random_sprite, random_free_location


# Design analysis for the D12 variant
# a) Concrete behavior:
#    The border segment stays straight and contiguous, but it is now a 3/4 code string instead of solid yellow.
#    Reading the segment left->right or top->bottom gives the stripe cycle directly: e.g. 4 4 3 4 4 4 means
#    Y,Y,G,Y,Y,Y, so stripes 0..5 use Y,Y,G,Y,Y,Y and then repeat. Original logic would have used Y,Y,G,Y,Y,G.
#    The affected step is the cycle lookup inside the stripe march; span detection and green-row/column flood rules stay.
# b) Complexity calibration:
#    A single embedded green in a long segment is already visible because the cycle length changes from fixed-3 to
#    segment-length. The generator still rejects near-baseline cycles so the variant does not collapse back to Y,Y,G.
# c) Hidden assumptions audited:
#    Old code assumed the border clue was all yellow, so `(g == 4)` both located the segment and safely filled it.
#    That breaks here because the clue itself contains green. The new solver therefore searches the largest border
#    component over colors {3,4}, verifies it is a gap-free straight run, and decodes each cell's color as cycle data.
#    Another old assumption was that green on the border can only come from output-time recoloring; now green exists in
#    input, so the generator keeps all other objects away from colors 3/4 to avoid ambiguity.
# d) Placement legality:
#    The generator places one contiguous border code that never touches corners, keeps interior blobs off the starting
#    edge row/column, and uses `random_free_location` so blobs do not overlap or leave the grid.
# e) Visual observability:
#    Samples are rejected unless the decoded cycle disagrees with the original fixed Y,Y,G baseline on several stripe
#    steps, and unless at least two decoded green stripes visibly flood pixels outside the narrow span. This makes the
#    D12 change recoverable from a few IO examples instead of being a hidden implementation detail.

Y = 4
G = 3


def _locate_border_code(g: np.ndarray):
    H, W = g.shape

    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = border & np.isin(g, [Y, G])
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
        if cols != list(range(int(cols[0]), int(cols[-1]) + 1)):
            return None
        cycle = [int(g[int(r0), int(c)]) for c in cols]
        if any(col not in (Y, G) for col in cycle):
            return None
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span": cols,
            "cycle": cycle,
            "dir1": 1 if int(r0) == 0 else -1,
            "limit": H,
        }

    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        if rows != list(range(int(rows[0]), int(rows[-1]) + 1)):
            return None
        cycle = [int(g[int(r), int(c0)]) for r in rows]
        if any(col not in (Y, G) for col in cycle):
            return None
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span": rows,
            "cycle": cycle,
            "dir1": 1 if int(c0) == 0 else -1,
            "limit": W,
        }

    return None


def _stripe_positions(start_pos: int, dir1: int, limit: int) -> list[int]:
    positions: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        positions.append(int(pos))
        step += 1
    return positions


def _baseline_mismatch_count(cycle: list[int], n_steps: int) -> int:
    baseline = [Y, Y, G]
    return sum(int(cycle[i % len(cycle)] != baseline[i % len(baseline)]) for i in range(int(n_steps)))


def _visible_green_floods(inp: np.ndarray, info: dict, bg: int) -> int:
    H, W = inp.shape
    span = set(int(x) for x in info["span"])
    count = 0
    for step, pos in enumerate(_stripe_positions(info["start_pos"], info["dir1"], info["limit"])):
        if int(info["cycle"][step % len(info["cycle"])]) != G:
            continue
        if info["orient"] == "h":
            if any(int(inp[int(pos), c]) != int(bg) for c in range(W) if c not in span):
                count += 1
        else:
            if any(int(inp[r, int(pos)]) != int(bg) for r in range(H) if r not in span):
                count += 1
    return int(count)


def _build_code_cycle(seg_len: int, n_steps: int) -> list[int] | None:
    interior = list(range(1, int(seg_len) - 1))
    if len(interior) < 1:
        return None

    choices: list[list[int]] = []
    for n_green in (1, 2):
        if n_green > len(interior):
            continue
        for combo in itertools.combinations(interior, n_green):
            if any((b - a) <= 1 for a, b in zip(combo, combo[1:])):
                continue
            cycle = [Y] * int(seg_len)
            for idx in combo:
                cycle[int(idx)] = G
            green_steps = sum(int(cycle[i % len(cycle)] == G) for i in range(int(n_steps)))
            mismatch = _baseline_mismatch_count(cycle, n_steps)
            if green_steps >= 2 and mismatch >= 3:
                choices.append(cycle)

    if not choices:
        return None
    return random.choice(choices)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    info = _locate_border_code(g)
    if info is None:
        return out

    span = np.array(info["span"], dtype=int)
    positions = _stripe_positions(info["start_pos"], info["dir1"], info["limit"])

    for step, pos in enumerate(positions):
        col = int(info["cycle"][step % len(info["cycle"])])

        if info["orient"] == "h":
            if col == G:
                out[int(pos), :] = np.where(out[int(pos), :] != int(bg), G, out[int(pos), :])
            out[int(pos), span] = col
            gap = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < H:
                out[int(gap), span] = int(bg)
        else:
            if col == G:
                out[:, int(pos)] = np.where(out[:, int(pos)] != int(bg), G, out[:, int(pos)])
            out[span, int(pos)] = col
            gap = int(info["start_pos"]) + int(info["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < W:
                out[span, int(gap)] = int(bg)

    return out


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(5, 7)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "h"
        edge_row = 0 if edge == "top" else H - 1
        n_steps = len(_stripe_positions(edge_row, 1 if edge_row == 0 else -1, H))
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1
        n_steps = len(_stripe_positions(edge_col, 1 if edge_col == 0 else -1, W))

    cycle = _build_code_cycle(seg_len, n_steps)
    if cycle is None:
        return None

    grid = np.full((H, W), bg, dtype=int)

    if orient == "h":
        for offset, c in enumerate(span):
            grid[edge_row, int(c)] = int(cycle[offset])
        forbidden_rows = {int(edge_row)}
        forbidden_cols = set()
    else:
        for offset, r in enumerate(span):
            grid[int(r), edge_col] = int(cycle[offset])
        forbidden_rows = set()
        forbidden_cols = {int(edge_col)}

    n_blobs = rng.randint(5, 9)
    canvas = grid.copy()

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
        blob_rows = set(int(pr + dr) for dr in range(sh) if sprite_mask[dr].any())
        blob_cols = set(int(pc + dc) for dc in range(sw) if sprite_mask[:, dc].any())

        if blob_rows & forbidden_rows:
            continue
        if blob_cols & forbidden_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    fg_mask = (canvas != bg) & ~np.isin(canvas, [Y, G])
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    info = _locate_border_code(inp)
    if info is None:
        return None

    mismatch = _baseline_mismatch_count(info["cycle"], len(_stripe_positions(info["start_pos"], info["dir1"], info["limit"])))
    if mismatch < 3:
        return None

    if _visible_green_floods(inp, info, bg) < 2:
        return None

    out = main(inp)

    if get_bg(inp) != bg:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
