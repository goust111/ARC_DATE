# Variant of 221dfab4
# Dimension: D14 — 边缘选择与旋转对称性 (meta)
# Change: 黄色线段可位于四条边缘中的任意一条，推进方向自动朝对侧... → 推进方向由额外标记决定

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite

Y = 4
G = 3

# Design analysis required by the task:
# a) Concrete behavior:
#    The input still has one yellow border segment plus scattered interior blobs, but now a
#    single corner marker pixel is added. If the marker sits in the corner associated with a
#    clockwise 90-degree turn from the original inward propagation, the striped pattern turns
#    instead of marching toward the opposite edge. Example: with a top-edge segment at
#    columns 6..10 and a marker at the top-left corner, the old code would paint rows
#    0,2,4,... downward; the variant keeps the seed segment on the top edge and paints
#    vertical stripes at columns 6,4,2,0 with the same Y,Y,G cycle, so the green "flood"
#    happens on full columns rather than full rows.
# b) Complexity calibration:
#    The generator forces enough room for at least three turned stripes, so a green stripe is
#    visible. The marker uses a dedicated color distinct from blob/stripe colors. The output
#    therefore differs from the old task by both geometry (turned axis) and recolor direction.
# c) Old implicit assumptions that break:
#    The original `dir1 = 1 if start_pos == 0 else -1` assumes edge location alone determines
#    the propagation direction. That is no longer true because the corner marker can switch the
#    solver into the turned mode. Likewise, the old code assumes stripe orientation always
#    matches the border segment orientation; in the turned mode the copied stripes rotate by
#    90 degrees, so stripe orientation and edge-segment orientation must be tracked separately.
# d) Placement legality:
#    The generator keeps blobs off the border, keeps the yellow segment away from corners, and
#    chooses segment positions with at least 4 cells of room in the turned direction so the
#    third stripe (green) always exists. Marker and blobs never overlap because blobs are placed
#    with `border=1`.
# e) Visual observability:
#    A human can now see that the corner marker predicts which adjacent side receives the stripe
#    train: the stripes hug a corner instead of heading across the grid, and green flood lines
#    run along the turned axis. The generator rejects cases where green stripes miss all interior
#    foreground pixels, so the changed flood direction remains visible in the examples.
# Post-edit assumption review:
#    `main()` now assumes the largest yellow border component is the seed segment and that at
#    most one informative corner marker exists. `generate_input()` enforces exactly that setup
#    and also guarantees the marker is placed in the clockwise-turn corner, so the new branch is
#    actually exercised rather than silently falling back to the original behavior.


def _find_seed_segment(g: np.ndarray) -> dict | None:
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

    if r0 == r1 and (r0 == 0 or r0 == H - 1):
        cols = sorted(int(c) for c in cs)
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        return {
            "edge": "top" if r0 == 0 else "bottom",
            "seg_orient": "h",
            "span": list(range(cols[0], cols[-1] + 1)),
            "length": int(cols[-1] - cols[0] + 1),
        }

    if c0 == c1 and (c0 == 0 or c0 == W - 1):
        rows = sorted(int(r) for r in rs)
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return {
            "edge": "left" if c0 == 0 else "right",
            "seg_orient": "v",
            "span": list(range(rows[0], rows[-1] + 1)),
            "length": int(rows[-1] - rows[0] + 1),
        }

    return None


def _find_marker_corner(g: np.ndarray, bg: int) -> str | None:
    H, W = g.shape
    corners = {
        "tl": (0, 0),
        "tr": (0, W - 1),
        "bl": (H - 1, 0),
        "br": (H - 1, W - 1),
    }
    present = []
    for name, (r, c) in corners.items():
        col = int(g[r, c])
        if col != int(bg) and col != int(Y):
            present.append(name)
    if len(present) == 1:
        return present[0]
    return None


def _layout(seed: dict, shape: tuple[int, int], rotated: bool) -> tuple[str, list[int], list[int], list[int]]:
    H, W = shape
    edge = seed["edge"]
    span = list(seed["span"])
    length = int(seed["length"])

    if not rotated:
        if edge == "top":
            stripe_orient = "h"
            stripe_span = span
            start_pos = 0
            dir1 = 1
            limit = H
        elif edge == "bottom":
            stripe_orient = "h"
            stripe_span = span
            start_pos = H - 1
            dir1 = -1
            limit = H
        elif edge == "left":
            stripe_orient = "v"
            stripe_span = span
            start_pos = 0
            dir1 = 1
            limit = W
        else:
            stripe_orient = "v"
            stripe_span = span
            start_pos = W - 1
            dir1 = -1
            limit = W
    else:
        if edge == "top":
            stripe_orient = "v"
            stripe_span = list(range(0, length))
            start_pos = int(span[0])
            dir1 = -1
            limit = W
        elif edge == "bottom":
            stripe_orient = "v"
            stripe_span = list(range(H - length, H))
            start_pos = int(span[-1])
            dir1 = 1
            limit = W
        elif edge == "left":
            stripe_orient = "h"
            stripe_span = list(range(0, length))
            start_pos = int(span[-1])
            dir1 = 1
            limit = H
        else:
            stripe_orient = "h"
            stripe_span = list(range(W - length, W))
            start_pos = int(span[0])
            dir1 = -1
            limit = H

    positions: list[int] = []
    gaps: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        positions.append(int(pos))
        gaps.append(int(start_pos) + int(dir1) * (2 * int(step) + 1))
        step += 1

    return stripe_orient, stripe_span, positions, gaps


def _paint_pattern(g: np.ndarray, bg: int, layout: tuple[str, list[int], list[int], list[int]]) -> np.ndarray:
    H, W = g.shape
    stripe_orient, stripe_span, positions, gaps = layout
    out = g.copy()
    cycle = [Y, Y, G]

    for step, pos in enumerate(positions):
        col = int(cycle[int(step) % 3])
        if stripe_orient == "h":
            if col == G:
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            for c in stripe_span:
                out[pos, int(c)] = col
            gap = int(gaps[step])
            if 0 <= gap < H:
                for c in stripe_span:
                    out[gap, int(c)] = int(bg)
        else:
            if col == G:
                out[:, pos] = np.where(out[:, pos] != int(bg), int(G), out[:, pos])
            for r in stripe_span:
                out[int(r), pos] = col
            gap = int(gaps[step])
            if 0 <= gap < W:
                for r in stripe_span:
                    out[int(r), int(gap)] = int(bg)

    return out


def _solve(g: np.ndarray) -> np.ndarray:
    bg = int(get_bg(g))
    seed = _find_seed_segment(g)
    if seed is None:
        return g.copy()

    marker_corner = _find_marker_corner(g, bg)
    clockwise_corner = {
        "top": "tl",
        "bottom": "br",
        "left": "bl",
        "right": "tr",
    }[seed["edge"]]
    rotated = marker_corner == clockwise_corner

    return _paint_pattern(g, bg, _layout(seed, g.shape, rotated=rotated))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    return _solve(g)


def _green_hit_count(grid: np.ndarray, seed: dict, fg: int, rotated: bool) -> int:
    stripe_orient, _span, positions, _gaps = _layout(seed, grid.shape, rotated=rotated)
    hits = 0
    for step, pos in enumerate(positions):
        if step % 3 != 2:
            continue
        if stripe_orient == "h":
            hits += int(np.count_nonzero(grid[pos, :] == int(fg)))
        else:
            hits += int(np.count_nonzero(grid[:, pos] == int(fg)))
    return int(hits)


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg, marker = rng.sample(available, 3)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 8)

    if edge == "top":
        min_start = 4
        max_start = W - seg_len - 1
        if min_start > max_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        seg_span = list(range(seg_start, seg_start + seg_len))
        marker_rc = (0, 0)
    elif edge == "bottom":
        min_start = 1
        max_start = W - seg_len - 4
        if min_start > max_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        seg_span = list(range(seg_start, seg_start + seg_len))
        marker_rc = (H - 1, W - 1)
    elif edge == "left":
        min_start = 1
        max_start = H - seg_len - 4
        if min_start > max_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        seg_span = list(range(seg_start, seg_start + seg_len))
        marker_rc = (H - 1, 0)
    else:
        min_start = 4
        max_start = H - seg_len - 1
        if min_start > max_start:
            return None
        seg_start = rng.randint(min_start, max_start)
        seg_span = list(range(seg_start, seg_start + seg_len))
        marker_rc = (0, W - 1)

    grid = np.full((H, W), bg, dtype=int)

    if edge in ("top", "bottom"):
        row = 0 if edge == "top" else H - 1
        for c in seg_span:
            grid[row, int(c)] = int(Y)
    else:
        col = 0 if edge == "left" else W - 1
        for r in seg_span:
            grid[int(r), col] = int(Y)

    grid[marker_rc] = int(marker)
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

        pix = int(np.sum(sprite != bg))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        sh, sw = sprite.shape
        for dr in range(sh):
            for dc in range(sw):
                if sprite[dr, dc] != bg:
                    canvas[pr + dr, pc + dc] = sprite[dr, dc]

    fg_mask = canvas == fg
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    seed = _find_seed_segment(canvas)
    if seed is None:
        return None

    if _green_hit_count(canvas, seed, fg=fg, rotated=True) < 3:
        return None

    inp = canvas.copy()
    out = main(inp)
    old_out = _paint_pattern(inp, int(bg), _layout(seed, inp.shape, rotated=False))

    if get_bg(inp) != bg:
        return None
    if np.array_equal(out, old_out):
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
