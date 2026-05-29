# Variant of 221dfab4
# Dimension: D8 — 前景色块碰撞变换 (transform)
# Change: 绿色条纹将全行/列非背景像素重着色为绿色(3)；黄色条纹覆盖跨度内的色块像素... → 色块保持原色不变（条纹被色块遮挡）
#
# Design analysis for this variant
# a) Concrete behavior:
#    The yellow border segment still defines an axis, a finite span, and a Y-Y-G stripe cycle
#    every two cells. The D8 change is a layer swap: interior blobs are now "in front" of the
#    propagated pattern. Any stripe or forced gap cell that lands on an original non-background
#    pixel must leave that pixel unchanged. Concretely, yellow stripes become yellow only on
#    background cells inside the span; green stripes likewise paint only background cells inside
#    the span; and the 1-cell gap is cleared only where the input was background. Example
#    (horizontal case, span columns 3..5): if row 4 has an object at (4,4), old logic writes 4/3/bg
#    over it depending on cycle position, while the variant leaves that object color intact and the
#    stripe/gap appears as "... stripe, hole, stripe ...".
# b) Complexity calibration:
#    A naive generator could accidentally produce inputs where no blob intersects any stripe/gap,
#    making the task visually indistinguishable from the original. To avoid that, the generator
#    explicitly places blobs on yellow-stripe cells, green-stripe cells, and green rows/cols
#    outside the span where the original solver would recolor them.
# c) Implicit assumptions audit:
#    In the original solver, three writes depended on "overwriting blobs is harmless":
#    - `out[pos, :] = np.where(out[pos, :] != bg, G, out[pos, :])` for green full-line recolor
#    - `out[pos, span] = col` for direct stripe painting
#    - `out[gap, span] = bg` for clearing the 1-cell gap
#    After D8, all three must instead consult the original input occupancy (`g != bg`) so that
#    only original background is paintable. Using `out != bg` would be wrong because it would let
#    later stripe writes treat earlier painted cells as obstacles, which is not part of the task.
# d) Placement legality:
#    All blobs are kept at least one cell away from the border and from each other. Targeted blobs
#    are anchored to cells that guarantee stripe collisions while still using bounded random sprites,
#    so the resulting inputs remain non-overlapping, in-bounds, and solvable by the same segment
#    detection logic as the original task.
# e) Visual observability:
#    The generator rejects samples unless several original foreground pixels would have been modified
#    by the old solver. This guarantees visible yellow/green stripe holes and preserved foreground on
#    green lines, so a human can infer the changed layering from a few examples.
#
# Reference-task note:
# The prompt cites 53fb4810 as the keep_source inspiration. Under the allowed local resources, its
# evaluation examples are available but a solver file is not present in the workspace, so the
# "keep source pixels unchanged while pattern continues around them" idea is used as the baseline.

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the same Y-Y-G pattern as 221dfab4, but treat original foreground blobs as opaque.

    D8 change:
    - Stripe writes only affect cells that were background in the input.
    - The 1-cell gap clearing also skips original foreground.
    - Foreground blobs therefore keep their source color and punch holes in the pattern.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    info = _segment_info(g)
    if info is None:
        return out

    orient, start_pos, span, dir1, limit = info
    span_idx = np.array(span, dtype=int)

    for _step, pos, col, gap in _stripe_steps(start_pos, dir1, limit):
        if orient == "h":
            paintable = (g[pos, span_idx] == int(bg))
            if bool(paintable.any()):
                out[pos, span_idx[paintable]] = int(col)

            if 0 <= gap < g.shape[0]:
                clearable = (g[gap, span_idx] == int(bg))
                if bool(clearable.any()):
                    out[gap, span_idx[clearable]] = int(bg)
        else:
            paintable = (g[span_idx, pos] == int(bg))
            if bool(paintable.any()):
                out[span_idx[paintable], pos] = int(col)

            if 0 <= gap < g.shape[1]:
                clearable = (g[span_idx, gap] == int(bg))
                if bool(clearable.any()):
                    out[span_idx[clearable], gap] = int(bg)

    return out


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
    mask = (sprite != int(bg))
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


def _can_place(canvas: np.ndarray, sprite: np.ndarray, r0: int, c0: int, bg: int, *, border: int = 1, padding: int = 1) -> bool:
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

    for _ in range(120):
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


def _line_targets(
    H: int,
    W: int,
    orient: str,
    span: list[int],
    stripe_positions: list[int],
    green_positions: list[int],
    start_pos: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    span_set = set(int(x) for x in span)
    yellow_lines = [int(pos) for pos in stripe_positions if int(pos) != int(start_pos) and int(pos) not in set(green_positions)]
    green_lines = [int(pos) for pos in green_positions if int(pos) != int(start_pos)]

    yellow_span_targets: list[tuple[int, int]] = []
    green_span_targets: list[tuple[int, int]] = []
    green_outer_targets: list[tuple[int, int]] = []

    if orient == "h":
        for r in yellow_lines:
            for c in span:
                yellow_span_targets.append((int(r), int(c)))
        for r in green_lines:
            for c in span:
                green_span_targets.append((int(r), int(c)))
            for c in range(1, W - 1):
                if int(c) not in span_set:
                    green_outer_targets.append((int(r), int(c)))
    else:
        for c in yellow_lines:
            for r in span:
                yellow_span_targets.append((int(r), int(c)))
        for c in green_lines:
            for r in span:
                green_span_targets.append((int(r), int(c)))
            for r in range(1, H - 1):
                if int(r) not in span_set:
                    green_outer_targets.append((int(r), int(c)))

    return yellow_span_targets, green_span_targets, green_outer_targets


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    new_out = main(inp)
    old_out = _original_main(inp)
    preserved = (inp != int(bg)) & (new_out == inp) & (old_out != inp)

    info = _segment_info(inp)
    if info is None:
        return {
            "protected_pixels": 0,
            "span_protected_pixels": 0,
            "green_outer_protected_pixels": 0,
        }

    orient, start_pos, span, dir1, limit = info
    span_set = set(int(x) for x in span)
    span_protected = 0
    green_outer_protected = 0

    for _step, pos, col, _gap in _stripe_steps(start_pos, dir1, limit):
        if orient == "h":
            if col in (int(Y), int(G)):
                for c in span:
                    if bool(preserved[int(pos), int(c)]):
                        span_protected += 1
            if col == int(G):
                for c in range(inp.shape[1]):
                    if int(c) not in span_set and bool(preserved[int(pos), int(c)]):
                        green_outer_protected += 1
        else:
            if col in (int(Y), int(G)):
                for r in span:
                    if bool(preserved[int(r), int(pos)]):
                        span_protected += 1
            if col == int(G):
                for r in range(inp.shape[0]):
                    if int(r) not in span_set and bool(preserved[int(r), int(pos)]):
                        green_outer_protected += 1

    return {
        "protected_pixels": int(np.count_nonzero(preserved)),
        "span_protected_pixels": int(span_protected),
        "green_outer_protected_pixels": int(green_outer_protected),
    }


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = list(range(10))
    bg = rng.choice([c for c in available if c not in (int(Y), int(G))])
    fg_colors = rng.sample([c for c in available if c not in (int(Y), int(G), int(bg))], rng.randint(2, 3))

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

    orient, start_pos, span, dir1, limit = info
    stripe_positions = []
    green_positions = []
    for _step, pos, col, _gap in _stripe_steps(start_pos, dir1, limit):
        stripe_positions.append(int(pos))
        if int(col) == int(G):
            green_positions.append(int(pos))

    yellow_targets, green_span_targets, green_outer_targets = _line_targets(
        H, W, orient, span, stripe_positions, green_positions, start_pos
    )

    canvas = grid.copy()

    must_place = [
        (yellow_targets, fg_colors[0]),
        (green_span_targets, fg_colors[1 % len(fg_colors)]),
        (green_outer_targets, fg_colors[2 % len(fg_colors)]),
    ]
    for targets, color in must_place:
        if not _place_targeted_blob(canvas, rng, bg, int(color), targets):
            return None

    extra_blobs = rng.randint(3, 6)
    for i in range(extra_blobs):
        color = fg_colors[i % len(fg_colors)]
        _place_random_blob(canvas, rng, bg, int(color))

    fg_mask = (canvas != int(bg)) & (canvas != int(Y))
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    if int(get_bg(canvas)) != int(bg):
        return None

    metrics = _variant_metrics(canvas)
    if metrics["protected_pixels"] < 6:
        return None
    if metrics["span_protected_pixels"] < 3:
        return None
    if metrics["green_outer_protected_pixels"] < 2:
        return None

    if np.array_equal(main(canvas), _original_main(canvas)):
        return None

    return canvas


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
