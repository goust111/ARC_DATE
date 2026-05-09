# Variant of 221dfab4
# Dimension: D6 — 源对象类型 (src)
# Change: 位于网格边缘的短黄色(4)线段，长度3-8格，距角至少1格... → 单像素控制点

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite

"""
Design analysis for the D6 variant.

a) Concrete behavior and trace
   Original 221dfab4 uses a yellow border segment as both the direction cue and the width cue.
   This variant keeps the Y,Y,G stripe propagation rule unchanged, but replaces the segment with:
   - one yellow control pixel on a border cell, not on a corner
   - one interior measuring bar lying on that control pixel's inward ray

   Example (top edge):
   - Original input might have row 0 columns 7..11 colored yellow. The solver reads span=7..11 directly.
   - Variant input has only grid[0, 9] = yellow. Looking downward along column 9, the first non-bg
     component is a short horizontal rectangle occupying columns 7..11. The solver now infers
     span=7..11 from that component, then performs the same stripe march as before.
   - Difference: source recognition becomes easier (one border pixel), but span recovery now needs a
     second step: ray-following plus component-span extraction, similar in spirit to 4a21e3da.

b) Complexity calibration
   A single border pixel alone would over-simplify the task because the stripe width would be 1 and the
   visual result would collapse. The measuring bar prevents that degeneration: outputs still contain wide
   stripes, but width is no longer explicitly drawn on the border. The generator keeps the measuring bar
   3+ cells away from the border so the new cue is visually distinct from the original border segment.

c) Implicit assumptions review
   Original code assumptions that break under D6:
   - Label the yellow border object and infer orientation from "same row" / "same column": invalid when
     the source is one pixel, because both row-span and col-span are length 1.
   - Use the yellow object's extent as the stripe span: invalid, since the border source no longer stores
     width information.
   The updated solver therefore:
   - infers orientation from which border the control pixel lies on
   - scans inward along the control ray until the first non-bg, non-yellow component is hit
   - uses that component's orthogonal extent as the stripe span

   New assumptions introduced by the variant:
   - there is exactly one salient yellow border control pixel
   - its inward ray reaches a unique first foreground component
   - that component is isolated enough that its min/max rows or columns are the intended span
   The generator explicitly enforces all three properties.

d) Placement legality
   The generator places:
   - one border yellow control pixel
   - one measuring bar intersecting the control ray but not touching the border
   - additional foreground blobs with padding, and never crossing the control ray
   This guarantees no overlap, no out-of-bounds placement, and prevents an earlier accidental blob from
   stealing the role of the measuring component in the solver.

e) Visual observability
   Across examples, a human can see that:
   - the border source shrank from a yellow segment to a single yellow dot
   - the produced stripe width still matches a wider interior bar aligned with that dot
   - the Y,Y,G cycle and the green-row/green-column recoloring behavior remain unchanged
   Because the measuring bar is rectangular while the other blobs are amorphous, its role is easier to
   infer from a few IO pairs.
"""


YELLOW = 4
GREEN = 3


def _pick_control_pixel(g: np.ndarray) -> tuple[int, int] | None:
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == int(YELLOW)) & border
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    comp_ids = list(range(1, int(n) + 1))
    singleton_ids = [cid for cid in comp_ids if int((lab == int(cid)).sum()) == 1]
    best_id = int(singleton_ids[0] if singleton_ids else max(comp_ids, key=lambda cid: int((lab == int(cid)).sum())))
    pts = np.argwhere(lab == int(best_id))
    if pts.shape[0] == 0:
        return None
    r, c = map(int, pts[0])
    return r, c


def _control_geometry(
    g: np.ndarray, ctrl: tuple[int, int]
) -> tuple[str, int, int, int, tuple[int, int]]:
    H, W = g.shape
    r, c = ctrl

    if r == 0:
        return "h", 0, 1, H, (1, 0)
    if r == H - 1:
        return "h", H - 1, -1, H, (-1, 0)
    if c == 0:
        return "v", 0, 1, W, (0, 1)
    if c == W - 1:
        return "v", W - 1, -1, W, (0, -1)
    raise ValueError("control pixel is not on the border")


def _infer_span(g: np.ndarray, bg: int, ctrl: tuple[int, int]) -> tuple[str, int, int, int, list[int]] | None:
    H, W = g.shape
    orient, start_pos, dir1, limit, (dr, dc) = _control_geometry(g, ctrl)
    r, c = ctrl

    rr = int(r + dr)
    cc = int(c + dc)
    fg_mask = (g != int(bg)) & (g != int(YELLOW))
    lab, _ = label_mask(fg_mask, connectivity=4)

    while 0 <= rr < H and 0 <= cc < W:
        if bool(fg_mask[int(rr), int(cc)]):
            cid = int(lab[int(rr), int(cc)])
            if cid <= 0:
                return None
            pts = np.argwhere(lab == int(cid))
            if pts.shape[0] == 0:
                return None
            if orient == "h":
                c0 = int(pts[:, 1].min())
                c1 = int(pts[:, 1].max())
                span = list(range(int(c0), int(c1) + 1))
            else:
                r0 = int(pts[:, 0].min())
                r1 = int(pts[:, 0].max())
                span = list(range(int(r0), int(r1) + 1))
            return orient, int(start_pos), int(dir1), int(limit), span
        rr += int(dr)
        cc += int(dc)

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the same Y,Y,G stripe pattern as 221dfab4, but infer stripe width from a
    measuring bar reached by a single yellow border control pixel.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    ctrl = _pick_control_pixel(g)
    if ctrl is None:
        return out

    inferred = _infer_span(g, bg, ctrl)
    if inferred is None:
        return out

    orient, start_pos, dir1, limit, span = inferred
    if not span:
        return out

    cycle = [int(YELLOW), int(YELLOW), int(GREEN)]

    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        col = int(cycle[int(step) % 3])

        if orient == "h":
            if col == int(GREEN):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(GREEN), out[pos, :])
            for span_c in span:
                out[pos, int(span_c)] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < H:
                for span_c in span:
                    out[int(gap), int(span_c)] = int(bg)
        else:
            if col == int(GREEN):
                out[:, pos] = np.where(out[:, pos] != int(bg), int(GREEN), out[:, pos])
            for span_r in span:
                out[int(span_r), pos] = int(col)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < W:
                for span_r in span:
                    out[int(span_r), int(gap)] = int(bg)

        step += 1

    return out


def _build_measuring_bar(
    rng: random.Random,
    grid: np.ndarray,
    fg: int,
    edge: str,
) -> tuple[tuple[int, int], dict[str, int]]:
    H, W = grid.shape
    span_len = rng.randint(3, 8)
    thickness = rng.randint(2, 3)

    if edge in ("top", "bottom"):
        c0 = rng.randint(2, W - span_len - 2)
        c1 = c0 + span_len - 1
        ctrl_c = rng.randint(c0, c1)
        r0 = rng.randint(3, H - thickness - 4)
        r1 = r0 + thickness - 1
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(fg)
        ctrl = (0 if edge == "top" else H - 1, ctrl_c)
        meta = {"orient_h": 1, "span_start": c0, "span_end": c1, "ray_hit": r0}
    else:
        r0 = rng.randint(2, H - span_len - 2)
        r1 = r0 + span_len - 1
        ctrl_r = rng.randint(r0, r1)
        c0 = rng.randint(3, W - thickness - 4)
        c1 = c0 + thickness - 1
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(fg)
        ctrl = (ctrl_r, 0 if edge == "left" else W - 1)
        meta = {"orient_h": 0, "span_start": r0, "span_end": r1, "ray_hit": c0}

    grid[int(ctrl[0]), int(ctrl[1])] = int(YELLOW)
    return ctrl, meta


def _count_recolored_objects(inp: np.ndarray, out: np.ndarray, bg: int) -> int:
    return int(np.sum((inp != int(bg)) & (inp != int(YELLOW)) & (out == int(GREEN))))


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (int(GREEN), int(YELLOW))]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])

    canvas = np.full((H, W), int(bg), dtype=int)
    ctrl, meta = _build_measuring_bar(rng, canvas, int(fg), edge)
    ctrl_r, ctrl_c = map(int, ctrl)

    n_blobs_target = rng.randint(5, 8)
    added = 0
    attempts = 0

    while added < n_blobs_target and attempts < 80:
        attempts += 1
        bh = rng.randint(2, 10)
        bw = rng.randint(2, 10)
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

        mask = sprite != int(bg)
        if not bool(mask.any()):
            continue
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=int(bg), border=1, padding=1)
        except ValueError:
            continue

        sprite_mask = sprite != int(bg)
        blob_rows = {int(pr + dr) for dr in range(sh) if bool(sprite_mask[dr].any())}
        blob_cols = {int(pc + dc) for dc in range(sw) if bool(sprite_mask[:, dc].any())}

        # Keep the control ray unique so the solver's first-hit component is always the measuring bar.
        if edge in ("top", "bottom") and int(ctrl_c) in blob_cols:
            continue
        if edge in ("left", "right") and int(ctrl_r) in blob_rows:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if int(sprite[dr, dc]) != int(bg):
                    canvas[int(pr + dr), int(pc + dc)] = int(sprite[dr, dc])
        added += 1

    fg_mask = (canvas != int(bg)) & (canvas != int(YELLOW))
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 6:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if _count_recolored_objects(inp, out, int(bg)) < 6:
        return None

    # Anti-degeneracy: the yellow source must be a single border cell and the measuring bar must sit
    # strictly inside the grid, otherwise the example would drift back toward the original border segment.
    if edge in ("top", "bottom"):
        if int(np.sum(inp[0 if edge == "top" else H - 1, :] == int(YELLOW))) != 1:
            return None
        if int(meta["ray_hit"]) <= 2:
            return None
    else:
        if int(np.sum(inp[:, 0 if edge == "left" else W - 1] == int(YELLOW))) != 1:
            return None
        if int(meta["ray_hit"]) <= 2:
            return None

    return inp


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
