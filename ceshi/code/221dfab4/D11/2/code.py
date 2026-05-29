# Variant of 221dfab4
# Dimension: D11 — 多源与推进顺序 (multi_step)
# Change: 单个源从一侧向对侧单次推进，无迭代或链式步骤... → 按色块距离排序的多步处理

from __future__ import annotations

"""
Design analysis for the D11 variant

- Reference-task note:
  The allowed local resources contain `1ae2feb7.json` and a mechanism summary
  ("process multiple bars by distance from separator"), but no standalone
  `1ae2feb7.py` solver file was present in the repository search. The design
  below therefore uses that distance-ordered multi-step idea as the baseline,
  while keeping the visible grammar of 221dfab4.

- a) Concrete behavior / trace:
  Original 221dfab4 has one yellow edge segment and a single sweep of stripes:
  positions 0,2,4,... along the travel axis are painted with colors Y,Y,G,Y,Y,G...
  Yellow stripes only paint the segment span; green stripes recolor the whole
  row/column's non-background cells to green.
  This variant keeps those stripes, but also tracks every interior foreground
  component. Components are ordered by axial distance from the yellow segment.
  When a component is first hit by a green stripe, it undergoes one extra step:
  its own bounding box is filled into a solid rectangle of the component color,
  with the striking green row/column preserved in green.
  Example for a top-edge source:
  - Original: an irregular blob crossing green row 10 only gets the cells on row
    10 recolored to green.
  - Variant: that same blob is processed when the sweep reaches row 10; after
    the green hit, the whole blob bbox turns into a solid rectangle, split by
    the green row. Farther blobs do not transform until their own later green
    hit arrives.
  The key visible difference is therefore not just "row 10 becomes green", but
  "the nearest hit blob changes shape first, then the next, then the next".

- b) Complexity calibration:
  If the generator kept arbitrary scattered blobs, many would never land on a
  green stripe and the D11 change would be visually weak. To avoid that
  simplification, the generator deliberately places 3-5 blobs on distinct green
  stripe rows/columns. Their input shapes are irregular and asymmetric, so the
  rectangle-after-hit transformation is obvious and does not collapse back to the
  original task.

- c) Old implicit assumptions that break under D11:
  - The original solver assumes the stripe sweep is the only state change.
    Under D11, green-hit components need their own per-object state.
  - The original unconditional `out[pos, :] = ...green...` is still valid for
    the stripe itself, but no longer describes the full effect of a green step:
    extra object transformation must happen exactly once per hit component.
  - The original code never needs object ordering. Under D11, ordering must be
    explicit and stable; ties are broken by orthogonal anchor coordinate.
  - Writing object transformations directly onto `out` is only safe if object
    bboxes are disjoint. The new generator enforces that.

- d) Placement legality:
  The generator builds each object inside its own bbox with one-cell border and
  one-cell padding between bboxes. Objects are placed away from the yellow span,
  so yellow stripes and forced bg gaps do not destroy them before the intended
  green hit. Each chosen object intersects exactly one intended green stripe at
  creation time, so `main()` can process it deterministically.

- e) Visual observability:
  In a few training examples, a human should see: same yellow border segment,
  same Y/Y/G stripe cadence, but now several blobs on successive green steps
  change from jagged shapes into solid rectangles. Because the rectangles appear
  at multiple increasing distances from the source, the "near-to-far multi-step"
  behavior is visible from the I/O pairs instead of being a hidden implementation
  detail.

- 9) New assumptions introduced by the solver and guaranteed by the generator:
  - Components exclude yellow and background colors.
  - Component bboxes do not overlap.
  - Every intended transformed component intersects a unique earliest green
    stripe.
  - Objects do not intersect the yellow segment span, so their special
    transformation does not need to preserve yellow-overpainted cells.
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_sprite


Y = 4
G = 3


def _find_yellow_segment(g: np.ndarray):
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
        cols = sorted(int(c) for c in cs)
        start_pos = int(r0)
        return {
            "orient": "h",
            "start_pos": start_pos,
            "span": list(range(int(cols[0]), int(cols[-1]) + 1)),
            "dir1": 1 if start_pos == 0 else -1,
            "limit": H,
        }
    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        start_pos = int(c0)
        return {
            "orient": "v",
            "start_pos": start_pos,
            "span": list(range(int(rows[0]), int(rows[-1]) + 1)),
            "dir1": 1 if start_pos == 0 else -1,
            "limit": W,
        }
    return None


def _stripe_step_from_axis(axis_pos: int, start_pos: int, dir1: int) -> int | None:
    delta = int(axis_pos - start_pos) * int(dir1)
    if delta < 0 or delta % 2 != 0:
        return None
    return delta // 2


def _component_infos(g: np.ndarray, bg: int, meta: dict) -> list[dict]:
    mask = (g != int(bg)) & (g != Y)
    lab, n = label_mask(mask, connectivity=4)
    infos: list[dict] = []
    orient = meta["orient"]
    start_pos = int(meta["start_pos"])
    dir1 = int(meta["dir1"])

    for cid in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(cid))
        if coords.size == 0:
            continue

        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())

        colors, counts = np.unique(g[lab == int(cid)], return_counts=True)
        pairs = [(int(col), int(cnt)) for col, cnt in zip(colors, counts) if int(col) not in (int(bg), Y)]
        if not pairs:
            continue
        color = max(pairs, key=lambda t: t[1])[0]

        axis_vals = rs if orient == "h" else cs
        orth_anchor = int(cs.min()) if orient == "h" else int(rs.min())
        axial_dists = [int(v - start_pos) * dir1 for v in axis_vals]
        dist = min(d for d in axial_dists if d >= 0)

        first_green_step = None
        first_green_pos = None
        for axis_pos in sorted(set(int(v) for v in axis_vals), key=lambda v: (int(v - start_pos) * dir1, v)):
            step = _stripe_step_from_axis(axis_pos, start_pos, dir1)
            if step is not None and step % 3 == 2:
                first_green_step = int(step)
                first_green_pos = int(axis_pos)
                break

        infos.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "color": int(color),
                "dist": int(dist),
                "orth_anchor": int(orth_anchor),
                "first_green_step": first_green_step,
                "first_green_pos": first_green_pos,
            }
        )

    infos.sort(
        key=lambda info: (
            int(info["dist"]),
            int(info["first_green_step"]) if info["first_green_step"] is not None else 10**9,
            int(info["orth_anchor"]),
        )
    )
    return infos


def _solidify_bbox(out: np.ndarray, comp: dict, strike_pos: int, orient: str) -> None:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    if orient == "h" and r0 <= int(strike_pos) <= r1:
        out[int(strike_pos), c0 : c1 + 1] = G
    if orient == "v" and c0 <= int(strike_pos) <= c1:
        out[r0 : r1 + 1, int(strike_pos)] = G


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    meta = _find_yellow_segment(g)
    if meta is None:
        return out

    orient = meta["orient"]
    start_pos = int(meta["start_pos"])
    span = list(meta["span"])
    dir1 = int(meta["dir1"])
    limit = int(meta["limit"])

    comps = _component_infos(g, bg, meta)
    comps_by_green_pos: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        if comp["first_green_pos"] is not None:
            comps_by_green_pos[int(comp["first_green_pos"])].append(comp)

    cycle = [Y, Y, G]

    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break

        col = int(cycle[int(step) % 3])
        if orient == "h":
            if col == G:
                out[pos, :] = np.where(out[pos, :] != int(bg), G, out[pos, :])
            for c in span:
                out[pos, int(c)] = int(col)
            if col == G:
                for comp in comps_by_green_pos.get(int(pos), []):
                    _solidify_bbox(out, comp, pos, orient)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < H:
                for c in span:
                    out[int(gap), int(c)] = int(bg)
        else:
            if col == G:
                out[:, pos] = np.where(out[:, pos] != int(bg), G, out[:, pos])
            for r in span:
                out[int(r), pos] = int(col)
            if col == G:
                for comp in comps_by_green_pos.get(int(pos), []):
                    _solidify_bbox(out, comp, pos, orient)
            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < W:
                for r in span:
                    out[int(r), int(gap)] = int(bg)

        step += 1

    return out


def _green_positions(meta: dict) -> list[int]:
    out: list[int] = []
    step = 0
    start_pos = int(meta["start_pos"])
    dir1 = int(meta["dir1"])
    limit = int(meta["limit"])
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        if step % 3 == 2:
            out.append(int(pos))
        step += 1
    return out


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray:
    mask = (sprite != int(bg))
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _make_irregular_blob(rng: random.Random, h: int, w: int, fg: int, bg: int, orient: str) -> np.ndarray | None:
    anchor = (h // 2) if orient == "h" else (w // 2)
    for _ in range(200):
        try:
            sprite = random_sprite(
                h,
                w,
                density=rng.uniform(0.45, 0.72),
                symmetry="not_symmetric",
                color_palette=[int(fg)],
                connectivity=4,
                background=int(bg),
            )
        except (AssertionError, RecursionError):
            continue

        sprite = _trim_sprite(np.array(sprite, dtype=int), bg)
        sh, sw = sprite.shape
        if sh != h or sw != w:
            continue

        mask = (sprite != int(bg))
        if int(mask.sum()) < 5 or int(mask.sum()) >= int(h * w):
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        if orient == "h" and not bool(mask[int(anchor), :].any()):
            continue
        if orient == "v" and not bool(mask[:, int(anchor)].any()):
            continue
        return sprite
    return None


def _bbox_separated(bbox: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], padding: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in others:
        if not (r1 + padding < or0 or or1 + padding < r0 or c1 + padding < oc0 or oc1 + padding < c0):
            return False
    return True


def _place_hit_blob(
    rng: random.Random,
    canvas: np.ndarray,
    meta: dict,
    gpos: int,
    fg: int,
    bg: int,
    used_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    H, W = canvas.shape
    span0 = int(min(meta["span"]))
    span1 = int(max(meta["span"]))
    orient = meta["orient"]

    for _ in range(200):
        h = rng.randint(3, 5)
        w = rng.randint(3, 6)
        sprite = _make_irregular_blob(rng, h, w, fg, bg, orient)
        if sprite is None:
            continue

        if orient == "h":
            anchor = h // 2
            r0 = int(gpos) - int(anchor)
            if r0 < 1 or r0 + h > H - 1:
                continue
            candidates = []
            left_max = span0 - w
            right_min = span1 + 1
            if left_max >= 1:
                candidates.extend(range(1, left_max + 1))
            if right_min + w - 1 <= W - 2:
                candidates.extend(range(right_min, W - w))
            rng.shuffle(candidates)
            for c0 in candidates:
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                if not _bbox_separated(bbox, used_bboxes, padding=1):
                    continue
                mask = (sprite != int(bg))
                canvas[r0 : r0 + h, c0 : c0 + w][mask] = int(fg)
                used_bboxes.append(bbox)
                return True
        else:
            anchor = w // 2
            c0 = int(gpos) - int(anchor)
            if c0 < 1 or c0 + w > W - 1:
                continue
            candidates = []
            top_max = span0 - h
            bot_min = span1 + 1
            if top_max >= 1:
                candidates.extend(range(1, top_max + 1))
            if bot_min + h - 1 <= H - 2:
                candidates.extend(range(bot_min, H - h))
            rng.shuffle(candidates)
            for r0 in candidates:
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                if not _bbox_separated(bbox, used_bboxes, padding=1):
                    continue
                mask = (sprite != int(bg))
                canvas[r0 : r0 + h, c0 : c0 + w][mask] = int(fg)
                used_bboxes.append(bbox)
                return True

    return False


def _gen_attempt():
    rng = random.Random()
    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 8)

    grid = np.full((H, W), int(bg), dtype=int)
    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_row = 0 if edge == "top" else H - 1
        for c in span:
            grid[int(edge_row), int(c)] = Y
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        edge_col = 0 if edge == "left" else W - 1
        for r in span:
            grid[int(r), int(edge_col)] = Y

    meta = _find_yellow_segment(grid)
    if meta is None:
        return None

    green_positions = [p for p in _green_positions(meta) if 1 <= p < int(meta["limit"]) - 1]
    if len(green_positions) < 3:
        return None

    n_targets = rng.randint(3, min(5, len(green_positions)))
    chosen_green_positions = sorted(rng.sample(green_positions, n_targets))

    canvas = grid.copy()
    used_bboxes: list[tuple[int, int, int, int]] = []
    for gpos in chosen_green_positions:
        ok = _place_hit_blob(rng, canvas, meta, int(gpos), int(fg), int(bg), used_bboxes)
        if not ok:
            return None

    fg_mask = (canvas != int(bg)) & (canvas != Y)
    lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 3:
        return None

    infos = _component_infos(canvas, int(bg), meta)
    transformed = [info for info in infos if info["first_green_pos"] is not None]
    if len(transformed) < 3:
        return None
    transformed_dists = {int(info["dist"]) for info in transformed}
    if len(transformed_dists) < 3:
        return None

    if get_bg(canvas) != int(bg):
        return None

    _ = main(canvas)
    return canvas


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
