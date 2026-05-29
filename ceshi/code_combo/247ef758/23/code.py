# Combo variant of 247ef758 (index 23)
# combo_spec: D4/3+D11/0
# Name: 实心化印刻+对角目标

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
The input scene stays the same as the source task: left sprite strip, one uniform divider column,
and a right-side frame whose opposite edges carry mirrored color markers. Each mover color still
gets row marks from the frame's left/right edges and column marks from the top/bottom edges.

Source behavior for one mover:
- sparse sprite bbox = 3x4, occupied pixels are not a full rectangle
- row marks = [3, 8], col marks = [10, 14]
- source targets = Cartesian product:
  (3,10), (3,14), (8,10), (8,14)
- source paint at each target = translate only occupied sprite pixels

Combo behavior:
- targets are diagonal/index-wise pairs only:
  (3,10), (8,14)
- paint at each target = translated sprite bbox becomes a solid 3x4 rectangle

So two pipeline stages change together:
1. target construction shrinks from all row x col crossings to sorted row/col pairings;
2. rendering changes from sparse stamp to solid bbox fill.
The visible result is "fewer copies, each copy blockier".

b) Complexity calibration
-------------------------
The combo would become too weak if either changed dimension were invisible:
- if every mover had only one row mark and one column mark, diagonal pairing would match the source;
- if mover sprites were already full rectangles, bbox filling would match the source.

The generator therefore enforces:
- at least one mover has exactly two row marks and two column marks;
- mover sprites are sparse within their bbox, usually with at least two empty bbox cells;
- row/column centers are spaced using bbox spans, so solid rectangles stay readable.

c) Implicit assumption audit against main()
-------------------------------------------
Frame detection:
- `largest_enclosed_component_bbox` assumes the frame interior is the largest background component
  not touching the border. The generator preserves a single clean framed cavity.

Divider detection:
- the divider finder assumes a near-full-height uniform non-bg column left of the frame.
  The generator still uses exactly one such full-height divider column.

Sprite extraction:
- left-strip components are assumed single-color and 8-connected. The generator builds movers and
  distractors that satisfy this.

Marker matching:
- the solver assumes only mover colors appear mirrored on both relevant frame edges. Distractors
  never receive mirrored marks.

Diagonal pairing:
- unlike the source task, target construction now assumes row marks and column marks can be ordered
  and paired. The solver uses sorted scan order; the generator guarantees equal counts per mover.

Solid fill:
- rendering no longer depends on foreground pixels after the erase step; it depends on each sprite's
  bbox. The generator guarantees movers are sparse so this change is visually observable.

Cover order:
- bottom-to-top overwrite still uses visible vertical position only, not color ids. This remains
  human-visible and compatible with solid rectangles.

d) Placement legality
---------------------
For each mover, marker rows/cols are chosen only from centers where the full bbox fits inside the
frame interior. That makes clipping unnecessary for generated samples and keeps the combo readable.
Global row/column clearance uses bbox height/width so rectangles from different movers do not merge
into ambiguous blobs. Left-strip sprites are stacked without overlap and never cross the divider.

e) Visual observability
-----------------------
From 2-4 examples a human can see:
- the frame advertises multiple row and column marks for a color, but only the ordered diagonal
  pairings receive copies;
- copied shapes are solid rectangles whose sizes match the movers' bounding boxes, not the movers'
  sparse silhouettes.
Both cues are pixel-visible and repeated across samples.

f) Visual cleanliness
---------------------
The scene vocabulary is unchanged from the source task. The generator avoids reusing row/column marks
 across colors, keeps mover count low (2-3), and reserves clearance based on bbox span so stamped
rectangles remain separable. This prevents the frame interior from turning into a noisy mosaic.

g) Change inferability
----------------------
No color remapping is introduced. Colors still identify which mover matches which edge markers.
Only geometry changes: "which marked crossings are used" and "what footprint is painted". That keeps
the rule inferable from structure rather than hidden metadata.

Post-change main()/generator contract
-------------------------------------
New solver assumptions introduced by the combo:
- every mover used by the frame has at least one row mark and one column mark;
- per mover, row mark count equals column mark count so diagonal pairing is well-defined;
- at least one mover has two pairs, otherwise diagonal would collapse to the source;
- movers are sparse enough that solid fill differs from sparse stamping.

`generate_input()` explicitly checks these assumptions by comparing the combo output against:
- diagonal sparse stamping (isolates D4/3),
- Cartesian solid stamping (isolates D11/0).
Samples are rejected unless both comparisons differ by a meaningful number of pixels.
"""

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """Return the bbox of the largest connected component not touching the grid border."""
    h, w = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))
    return best_bbox


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((h, w), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(w)[None, :] < int(fc0)

    sprites: list[dict[str, object]] = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0): int(r1) + 1, int(c0): int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                }
            )

    move_colors = {int(s["color"]) for s in sprites}
    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))

    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    return {
        "g": g,
        "bg": int(bg),
        "ir0": int(ir0),
        "ic0": int(ic0),
        "ir1": int(ir1),
        "ic1": int(ic1),
        "sprites": sprites,
        "row_marks": row_marks,
        "col_marks": col_marks,
    }


def _build_targets(
    row_marks: dict[int, list[int]],
    col_marks: dict[int, list[int]],
    *,
    mode: str,
) -> dict[int, list[tuple[int, int]]]:
    targets: dict[int, list[tuple[int, int]]] = {}
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        rows = sorted(map(int, row_marks[int(col)]))
        cols = sorted(map(int, col_marks[int(col)]))
        if mode == "diagonal":
            pts = [(int(r), int(c)) for r, c in zip(rows, cols)]
        elif mode == "cartesian":
            pts = sorted({(int(r), int(c)) for r in rows for c in cols})
        else:
            raise ValueError(f"unknown target mode: {mode}")
        if pts:
            targets[int(col)] = pts
    return targets


def _render(scene: dict[str, object], targets: dict[int, list[tuple[int, int]]], *, fill_bbox: bool) -> np.ndarray:
    g = np.array(scene["g"], dtype=int)
    out = g.copy()
    bg = int(scene["bg"])
    ir0 = int(scene["ir0"])
    ic0 = int(scene["ic0"])
    ir1 = int(scene["ir1"])
    ic1 = int(scene["ic1"])
    sprites = list(scene["sprites"])

    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(np.array(s["mask"], dtype=bool)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        r0, c0, r1, c1 = map(int, (s["r0"], s["c0"], s["r1"], s["c1"]))
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        pix = [(r0 + int(rr), c0 + int(cc2)) for rr, cc2 in np.argwhere(np.array(s["mask"], dtype=bool))]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            if fill_bbox:
                rr0 = max(int(ir0), int(r0 + dr))
                rr1 = min(int(ir1), int(r1 + dr))
                cc0 = max(int(ic0), int(c0 + dc))
                cc1 = min(int(ic1), int(c1 + dc))
                if rr0 <= rr1 and cc0 <= cc1:
                    out[int(rr0): int(rr1) + 1, int(cc0): int(cc1) + 1] = int(col)
            else:
                for r, c in pix:
                    rr = int(r + dr)
                    cc3 = int(c + dc)
                    if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                        out[rr, cc3] = int(col)
    return out


def _solve(grid: np.ndarray, *, target_mode: str, fill_bbox: bool) -> np.ndarray:
    scene = _extract_scene(grid)
    targets = _build_targets(scene["row_marks"], scene["col_marks"], mode=target_mode)
    return _render(scene, targets, fill_bbox=fill_bbox)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - use only sorted row/column diagonal pairings as targets;
    - erase movers from the left strip;
    - stamp a solid bbox rectangle at each target center;
    - preserve bottom-to-top overwrite order.
    """
    return _solve(grid, target_mode="diagonal", fill_bbox=True)


def _diagonal_sparse_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, target_mode="diagonal", fill_bbox=False)


def _cartesian_solid_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, target_mode="cartesian", fill_bbox=True)


def _pick_positions_with_clearance(
    candidates: list[int],
    n: int,
    span: int,
    occupied: list[tuple[int, int]],
) -> list[int] | None:
    """Pick exactly n centers while respecting span-based clearance against prior centers."""
    if not candidates:
        return None
    order = candidates[:]
    random.shuffle(order)
    picked: list[int] = []
    local_occupied = occupied[:]
    for v in order:
        if all(abs(int(v) - int(p)) >= max(int(span), int(prev_span)) for p, prev_span in local_occupied):
            picked.append(int(v))
            local_occupied.append((int(v), int(span)))
            if len(picked) == int(n):
                return sorted(picked)
    return None


def _make_sprite(color: int, max_h: int, max_w: int, *, require_sparse_bbox: bool) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bbox."""
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(80):
        sh = min(random.choice([3, 3, 4, 4, 5]), max_h)
        sw = min(random.choice([3, 3, 4, 4, 5]), max_w)
        if sh < 2 or sw < 2:
            continue

        min_pixels = 4
        max_missing = 2 if require_sparse_bbox and sh * sw >= 6 else 1
        max_pixels = min(9, sh * sw - max_missing)
        if max_pixels < min_pixels:
            continue

        n_pixels = random.randint(min_pixels, max_pixels)
        pixels = {(sh // 2, sw // 2)}
        r, c = sh // 2, sw // 2
        for _ in range(n_pixels * 40):
            if len(pixels) >= n_pixels:
                break
            dr, dc = random.choice(
                [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            )
            nr, nc = r + dr, c + dc
            if 0 <= nr < sh and 0 <= nc < sw:
                pixels.add((nr, nc))
                r, c = nr, nc

        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        sprite_h = max_r - min_r + 1
        sprite_w = max_c - min_c + 1
        if sprite_h < 2 or sprite_w < 2:
            continue

        sprite = np.zeros((sprite_h, sprite_w), dtype=int)
        for pr, pc in pixels:
            sprite[pr - min_r, pc - min_c] = int(color)

        bbox_area = int(sprite_h * sprite_w)
        filled = int(np.count_nonzero(sprite))
        missing = bbox_area - filled
        if require_sparse_bbox and missing < (2 if bbox_area >= 6 else 1):
            continue
        return sprite
    return None


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether both combo dimensions are active on a generated sample.
    - paired_colors: movers with at least two row marks and two col marks.
    - extra_cartesian_targets: targets removed by diagonal pairing.
    - fill_gain_pixels: combo output vs diagonal sparse output.
    - diagonal_omission_pixels: combo output vs Cartesian solid output.
    """
    scene = _extract_scene(grid)
    row_marks = scene["row_marks"]
    col_marks = scene["col_marks"]

    paired_colors = 0
    extra_cartesian_targets = 0
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        nr = len(row_marks[int(col)])
        nc = len(col_marks[int(col)])
        if nr >= 2 and nc >= 2:
            paired_colors += 1
        extra_cartesian_targets += max(0, nr * nc - min(nr, nc))

    combo = main(grid)
    diagonal_sparse = _diagonal_sparse_output(grid)
    cartesian_solid = _cartesian_solid_output(grid)

    return {
        "paired_colors": int(paired_colors),
        "extra_cartesian_targets": int(extra_cartesian_targets),
        "fill_gain_pixels": int(np.count_nonzero(combo != diagonal_sparse)),
        "diagonal_omission_pixels": int(np.count_nonzero(combo != cartesian_solid)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    h = interior_h + 2
    if not 10 <= h <= 15:
        return None

    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    w = sprite_area_w + 1 + frame_w
    if not 14 <= w <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, w - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = h - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx: idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx: idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_sprite(mc, 5, sprite_area_w, require_sparse_bbox=True)
        if sp is None:
            return None
        sprites.append(sp)

    forced_double = random.randrange(n_movers)
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: list[tuple[int, int]] = []
    occupied_cols: list[tuple[int, int]] = []

    for i, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None

        n_pairs = 2 if i == forced_double else random.choice([1, 2])
        row_marks = _pick_positions_with_clearance(list(range(min_row, max_row + 1)), n_pairs, sh, occupied_rows)
        col_marks = _pick_positions_with_clearance(list(range(min_col, max_col + 1)), n_pairs, sw, occupied_cols)
        if row_marks is None or col_marks is None:
            return None

        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.extend((int(r), int(sh)) for r in row_marks)
        occupied_cols.extend((int(c), int(sw)) for c in col_marks)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0: fc0 + frame_w] = top_edge
    inp[h - 1, fc0: fc0 + frame_w] = top_edge

    for r in range(1, h - 1):
        val = int(frame_base)
        for mc, (rmarks, _) in mover_targets.items():
            if int(r) in rmarks:
                val = int(mc)
                break
        inp[int(r), int(fc0)] = val
        inp[int(r), int(fc1)] = val

    cur_row = 0
    all_items: list[np.ndarray] = sprites[:]
    for dc in distractor_colors:
        ds = _make_sprite(dc, min(5, h - cur_row), sprite_area_w, require_sparse_bbox=False)
        if ds is None:
            return None
        all_items.append(ds)

    for sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[dr, dc])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        combo = main(inp)
        diagonal_sparse = _diagonal_sparse_output(inp)
        cartesian_solid = _cartesian_solid_output(inp)
        metrics = _variant_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, combo):
        return None
    if np.array_equal(combo, diagonal_sparse):
        return None
    if np.array_equal(combo, cartesian_solid):
        return None
    if metrics["paired_colors"] < 1:
        return None
    if metrics["extra_cartesian_targets"] < 2:
        return None
    if metrics["fill_gain_pixels"] < 4:
        return None
    if metrics["diagonal_omission_pixels"] < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
