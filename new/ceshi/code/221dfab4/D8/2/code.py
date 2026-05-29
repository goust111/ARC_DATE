# Variant of 221dfab4
# Dimension: D8 — 前景色块碰撞变换 (transform)
# Change: 绿色条纹将全行/列非背景像素重着色为绿色(3)；黄色条纹覆盖跨度内的色块像素... → 碰撞检测标记
#
# Design analysis for this variant
# a) Concrete behavior:
#    - The border yellow segment still launches stripes inward every 2 cells with cycle Y,Y,G.
#    - Stripe cells are still drawn along the original segment span, and the 1-cell gaps are still cleared to bg.
#    - The changed part is object interaction: foreground blobs are no longer recolored by the stripe itself.
#      Instead, if a stripe "hits" a blob, the whole connected blob receives a one-cell 8-neighborhood outer
#      outline in a third color (7), matching the surround idea from 71e489b6.
#    - Hit semantics stay close to the original task:
#        * yellow stripe: only checks collisions inside the segment span;
#        * green stripe: checks collisions on the full row/column, because the original green rule also acted
#          on the whole row/column.
#    - Small example for horizontal stripes:
#        original: a green stripe on row r recolored every non-bg cell on row r to 3.
#        variant: the same row r only marks any touched component(s) for outlining; the blob body keeps its
#        original fg color, while nearby exterior cells become 7.
#
# b) Complexity calibration:
#    - If the generator allowed random placement, many samples would accidentally show only the old-looking
#      stripe pattern and no collision feedback. To make D8 visible, the generator guarantees both a yellow-hit
#      blob and a green-hit blob, plus untouched blobs for contrast.
#    - Outline color 7 is excluded from bg/fg choices so the surround signal is visually distinct.
#
# c) Old assumptions that break under D8:
#    - `out[pos, :] = np.where(..., 3, ...)` and unconditional `out[pos, c] = col` relied on the old rule that
#      overwriting foreground pixels was correct. Under D8 that would destroy the blob before outlining it, so
#      stripe painting is now restricted to non-object cells.
#    - Gap clearing previously overwrote the span unconditionally. Under D8 that could erase collision markers,
#      so outlines are applied after stripe painting, and gap clearing only targets non-object cells.
#
# d) Placement legality:
#    - Blobs are placed with border>=2 and padding>=2 so they do not overlap, do not touch the frame, and have
#      room for a full one-cell surround ring.
#    - Forced-hit placements are validated against the exact collision rule used by `main()`, so generator and
#      solver stay aligned.
#
# e) Visual observability:
#    - From a few examples, a human can still infer the stripe propagation, but the touched blobs now remain fg
#      and gain a ring of 7 instead of being repainted into the stripe color. Because the generator always
#      includes multiple outlined blobs, the D8 change is directly visible rather than a rare side effect.
#
# Reference-task note:
# - A solver file for 71e489b6 was not present in the provided workspace after repository-wide search by task id.
#   I therefore calibrated "surround" from the official 71e489b6 I/O examples only, using the standard
#   8-neighborhood outer ring in color 7.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[5]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_sprite

Y = 4
G = 3
S = 7
RING8 = np.ones((3, 3), dtype=bool)


def _find_border_segment(g: np.ndarray):
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
        return {
            "orient": "h",
            "start_pos": start_pos,
            "span": span,
            "dir1": 1 if start_pos == 0 else -1,
            "limit": H,
            "smask": smask,
        }

    if c0 == c1:
        start_pos = int(c0)
        rows = sorted(int(r) for r in rs)
        span = list(range(int(rows[0]), int(rows[-1]) + 1))
        return {
            "orient": "v",
            "start_pos": start_pos,
            "span": span,
            "dir1": 1 if start_pos == 0 else -1,
            "limit": W,
            "smask": smask,
        }

    return None


def _stripe_steps(start_pos: int, dir1: int, limit: int):
    cycle = [int(Y), int(Y), int(G)]
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        yield int(step), int(pos), int(cycle[int(step) % 3])
        step += 1


def _component_ring(mask: np.ndarray) -> np.ndarray:
    dil = ndimage.binary_dilation(mask, structure=RING8, iterations=1)
    return dil & (~mask)


def _collision_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    seg = _find_border_segment(g)
    if seg is None:
        return {
            "yellow_hit_components": 0,
            "green_hit_components": 0,
            "outlined_components": 0,
            "ring_pixels": 0,
        }

    obj_mask = (g != int(bg)) & (~seg["smask"])
    obj_lab, n_obj = label_mask(obj_mask, connectivity=4)
    if int(n_obj) <= 0:
        return {
            "yellow_hit_components": 0,
            "green_hit_components": 0,
            "outlined_components": 0,
            "ring_pixels": 0,
        }

    y_hits: set[int] = set()
    g_hits: set[int] = set()

    for _step, pos, col in _stripe_steps(seg["start_pos"], seg["dir1"], seg["limit"]):
        if seg["orient"] == "h":
            seen = obj_lab[pos, seg["span"]] if col == int(Y) else obj_lab[pos, :]
        else:
            seen = obj_lab[seg["span"], pos] if col == int(Y) else obj_lab[:, pos]

        ids = {int(v) for v in np.unique(seen) if int(v) > 0}
        if col == int(Y):
            y_hits |= ids
        else:
            g_hits |= ids

    ring_pixels = 0
    for cid in sorted(y_hits | g_hits):
        ring = _component_ring(obj_lab == int(cid))
        paintable = ring & (~obj_mask) & (~seg["smask"])
        ring_pixels += int(np.sum(paintable))

    return {
        "yellow_hit_components": len(y_hits),
        "green_hit_components": len(g_hits),
        "outlined_components": len(y_hits | g_hits),
        "ring_pixels": int(ring_pixels),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    seg = _find_border_segment(g)
    if seg is None:
        return out

    obj_mask = (g != int(bg)) & (~seg["smask"])
    obj_lab, n_obj = label_mask(obj_mask, connectivity=4)

    touched: set[int] = set()

    for step, pos, col in _stripe_steps(seg["start_pos"], seg["dir1"], seg["limit"]):
        if seg["orient"] == "h":
            if col == int(Y):
                hit_view = obj_lab[pos, seg["span"]]
            else:
                hit_view = obj_lab[pos, :]

            for v in np.unique(hit_view):
                if int(v) > 0:
                    touched.add(int(v))

            for c in seg["span"]:
                if not obj_mask[pos, int(c)] and out[pos, int(c)] == int(bg):
                    out[pos, int(c)] = int(col)

            gap = int(seg["start_pos"]) + int(seg["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < H:
                for c in seg["span"]:
                    if not obj_mask[int(gap), int(c)] and not seg["smask"][int(gap), int(c)]:
                        out[int(gap), int(c)] = int(bg)
        else:
            if col == int(Y):
                hit_view = obj_lab[seg["span"], pos]
            else:
                hit_view = obj_lab[:, pos]

            for v in np.unique(hit_view):
                if int(v) > 0:
                    touched.add(int(v))

            for r in seg["span"]:
                if not obj_mask[int(r), pos] and out[int(r), pos] == int(bg):
                    out[int(r), pos] = int(col)

            gap = int(seg["start_pos"]) + int(seg["dir1"]) * (2 * int(step) + 1)
            if 0 <= gap < W:
                for r in seg["span"]:
                    if not obj_mask[int(r), int(gap)] and not seg["smask"][int(r), int(gap)]:
                        out[int(r), int(gap)] = int(bg)

    if int(n_obj) > 0:
        for cid in sorted(touched):
            mask = obj_lab == int(cid)
            ring = _component_ring(mask)
            paintable = ring & (~obj_mask) & (~seg["smask"])
            out[paintable] = int(S)

    return out


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    if not bool(mask.any()):
        return None
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    trimmed = sprite[r0 : r1 + 1, c0 : c1 + 1]
    if np.sum(trimmed != int(bg)) < 5:
        return None
    return trimmed


def _sample_sprite(rng: random.Random, bg: int, fg: int) -> np.ndarray | None:
    for _ in range(80):
        bh = rng.randint(2, 6)
        bw = rng.randint(2, 7)
        density = rng.uniform(0.45, 0.82)
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

        sprite = _trim_sprite(sprite, bg)
        if sprite is None:
            continue

        h, w = sprite.shape
        pix = int(np.sum(sprite != int(bg)))
        if 2 <= h <= 7 and 2 <= w <= 8 and 5 <= pix <= 32:
            return sprite
    return None


def _placement_candidates(
    canvas: np.ndarray,
    sprite: np.ndarray,
    bg: int,
    *,
    border: int = 2,
    padding: int = 2,
):
    H, W = canvas.shape
    h, w = sprite.shape
    mask = sprite != int(bg)
    if not bool(mask.any()):
        return []
    if H - h - border < border or W - w - border < border:
        return []

    occ = canvas != int(bg)
    if padding > 0 and bool(occ.any()):
        occ = ndimage.binary_dilation(occ, structure=RING8, iterations=padding)

    coords = np.argwhere(mask)
    out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for r0 in range(border, H - h - border + 1):
        for c0 in range(border, W - w - border + 1):
            if bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                continue
            rows = r0 + coords[:, 0]
            cols = c0 + coords[:, 1]
            out.append((int(r0), int(c0), rows, cols))
    return out


def _paste_sprite(canvas: np.ndarray, sprite: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = sprite.shape
    for dr in range(h):
        for dc in range(w):
            if sprite[dr, dc] != int(bg):
                canvas[int(r0) + dr, int(c0) + dc] = int(sprite[dr, dc])


def _place_forced_blob(
    rng: random.Random,
    canvas: np.ndarray,
    bg: int,
    fg: int,
    *,
    orient: str,
    span: list[int],
    yellow_targets: list[int],
    green_targets: list[int],
    mode: str,
) -> bool:
    span_set = set(int(v) for v in span)
    green_set = set(int(v) for v in green_targets)

    if mode == "yellow_hit":
        targets = yellow_targets
    elif mode == "green_hit":
        targets = green_targets
    else:
        targets = []

    for _ in range(120):
        sprite = _sample_sprite(rng, bg, fg)
        if sprite is None:
            continue
        candidates = _placement_candidates(canvas, sprite, bg, border=2, padding=2)
        if not candidates:
            continue

        rng.shuffle(candidates)
        for r0, c0, rows, cols in candidates:
            row_set = {int(v) for v in rows}
            col_set = {int(v) for v in cols}

            ok = False
            if mode == "yellow_hit":
                if orient == "h":
                    ok = any(int(r) in targets for r in row_set) and bool(col_set & span_set)
                else:
                    ok = any(int(c) in targets for c in col_set) and bool(row_set & span_set)
            elif mode == "green_hit":
                if orient == "h":
                    ok = any(int(r) in targets for r in row_set) and not bool(col_set & span_set)
                else:
                    ok = any(int(c) in targets for c in col_set) and not bool(row_set & span_set)
            elif mode == "untouched":
                if orient == "h":
                    ok = not bool(row_set & green_set) and not bool(col_set & span_set)
                else:
                    ok = not bool(col_set & green_set) and not bool(row_set & span_set)

            if not ok:
                continue

            _paste_sprite(canvas, sprite, r0, c0, bg)
            return True

    return False


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (3, 4, 7)]
    bg, fg = rng.sample(available, 2)

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
            grid[edge_row, int(c)] = int(Y)
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_col = 0 if edge == "left" else W - 1
        for r in span:
            grid[int(r), edge_col] = int(Y)

    seg = _find_border_segment(grid)
    if seg is None:
        return None

    yellow_targets = [
        int(pos)
        for _step, pos, col in _stripe_steps(seg["start_pos"], seg["dir1"], seg["limit"])
        if col == int(Y) and int(pos) != int(seg["start_pos"]) and 2 <= int(pos) <= int(seg["limit"]) - 3
    ]
    green_targets = [
        int(pos)
        for _step, pos, col in _stripe_steps(seg["start_pos"], seg["dir1"], seg["limit"])
        if col == int(G) and 2 <= int(pos) <= int(seg["limit"]) - 3
    ]

    if not yellow_targets or not green_targets:
        return None

    canvas = grid.copy()

    # New solver assumptions:
    # - There is at least one component hit by a yellow stripe and one by a green stripe.
    # - Components have one-cell clearance for the surround ring.
    # - Some components remain untouched so the surround behavior is inferable by contrast.
    placements = ["yellow_hit", "green_hit", "untouched", "untouched", "untouched"]
    tail = placements[2:]
    rng.shuffle(tail)
    placements[2:] = tail

    for mode in placements:
        ok = _place_forced_blob(
            rng,
            canvas,
            bg,
            fg,
            orient=seg["orient"],
            span=seg["span"],
            yellow_targets=yellow_targets,
            green_targets=green_targets,
            mode=mode,
        )
        if not ok:
            return None

    fg_mask = (canvas != int(bg)) & (~seg["smask"])
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    metrics = _collision_metrics(canvas)
    if metrics["yellow_hit_components"] < 1 or metrics["green_hit_components"] < 1:
        return None
    if metrics["outlined_components"] < 2 or metrics["ring_pixels"] < 12:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
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
