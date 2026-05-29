# Variant of 247ef758
# Dimension: D11 — 笛卡尔积目标模式 (meta)
# Change: 每个精灵的目标位置是其行标记×列标记的完整笛卡尔积（如2行×2列=4个目标位置）... → 仅对角目标（非完整笛卡尔积）

from __future__ import annotations

# Design analysis for the D11 variant
# ----------------------------------
# a) Concrete behavior and trace
#    Input geometry stays the same: a left sprite strip, one solid divider column, and a framed
#    rectangle on the right whose opposite edges carry mirrored color markers. The original task
#    interprets all row markers of a color and all column markers of the same color as a full
#    Cartesian product. Example: rows [3, 8] and cols [10, 14] produce four targets
#    (3,10), (3,14), (8,10), (8,14). This variant keeps the same markers but changes only the
#    target-selection rule: the sorted row list is paired with the sorted column list index-wise,
#    so the same markers produce only two targets: (3,10) and (8,14). The frame still visually
#    advertises two row hits and two column hits, but only the "matched-order diagonal" gets used.
#    The change happens exactly where the solver turns marker positions into target coordinates.
#
# b) Complexity calibration
#    If every mover had only one row mark and one column mark, diagonal pairing would collapse back
#    to the original task. The generator therefore forces at least one mover per sample to have two
#    row marks and two column marks. Other movers may have one or two pairs. This keeps the variant
#    visible while preserving the original object vocabulary and difficulty.
#
# c) Old assumptions that no longer hold
#    - Original `targets` construction assumed every row mark combines with every column mark.
#      That is the exact line that must change.
#    - Rendering, erase logic, divider detection, frame detection, and bottom-to-top overwrite order
#      do not depend on Cartesian expansion and remain valid.
#    - A subtle new requirement appears: diagonal pairing is only unambiguous if row marks and
#      column marks can be ordered. The solver therefore uses scan order / sorted order, and the
#      generator guarantees equal counts for rows and cols per mover so no "leftover" markers exist.
#
# d) Placement legality
#    Generator keeps each mover's centered copies inside the frame by choosing marker rows/cols from
#    sprite-dependent valid ranges. Markers are never reused across colors, so no two movers compete
#    for the same center point. Inputs are accepted only if the new solver runs successfully and the
#    new output differs from the original Cartesian-product output on the same input.
#
# e) Visual observability
#    A 2-row × 2-col mover now yields two stamped copies instead of four. Because omitted
#    off-diagonal centers remain empty, the difference is directly visible from the input/output pair.
#    The generator rejects degenerate samples where diagonal and Cartesian outputs accidentally match.
#
# f) Visual cleanliness
#    The frame, divider, and left sprites are unchanged from the source task, so the scene remains
#    readable. Marker reuse across colors is forbidden, which keeps the edge coding clean. Stamps are
#    still clipped to the frame interior only.
#
# g) Color inferability
#    No new color mapping is introduced. Colors continue to identify which sprite matches which edge
#    markers; only the geometry of the chosen targets changes.

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask` that does not touch the border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    """
    Parse the common scene structure shared by the source task and this D11 variant.

    New solver assumptions introduced by D11 and satisfied by the generator:
    - For each mover color, row marks and column marks are both present.
    - Their intended pairing is determined by ascending order on each axis.
    - Counts are equal in generated samples, though the solver safely zips to min(len(rows), len(cols)).
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

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
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m}
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
        "H": int(H),
        "W": int(W),
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


def _render_targets(scene: dict[str, object], targets: dict[int, list[tuple[int, int]]]) -> np.ndarray:
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
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(np.array(s["mask"], dtype=bool))]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def _cartesian_output(grid: np.ndarray) -> np.ndarray:
    scene = _extract_scene(grid)
    targets = _build_targets(scene["row_marks"], scene["col_marks"], mode="cartesian")
    return _render_targets(scene, targets)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantifies whether D11 is actually active on a generated sample.
    - `paired_colors`: movers whose row/col marker count is >= 2 on both axes.
    - `extra_cartesian_targets`: number of targets that the source task would add but D11 omits.
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
    return {
        "paired_colors": int(paired_colors),
        "extra_cartesian_targets": int(extra_cartesian_targets),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    D11 variant of 247ef758.

    Same parsing and stamping as the source task, except target centers are not the full Cartesian
    product. Instead, for each sprite color, the sorted row markers are paired with the sorted
    column markers one-by-one, producing diagonal/order-matched targets only.
    """
    scene = _extract_scene(grid)
    targets = _build_targets(scene["row_marks"], scene["col_marks"], mode="diagonal")
    return _render_targets(scene, targets)


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return sorted(picked)
    return sorted(picked) if len(picked) == n else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    sh = min(random.choice([3, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
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
    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for pr, pc in pixels:
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    forced_double = random.randrange(n_movers)
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()

    for i, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None

        n_pairs = 2 if i == forced_double else random.choice([1, 2])
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        row_marks = _pick_spaced(avail_rows, n_pairs, sh)
        col_marks = _pick_spaced(avail_cols, n_pairs, sw)
        if row_marks is None or col_marks is None:
            return None

        mover_targets[mc] = (row_marks, col_marks)
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(mc, sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds))

    for _, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
        cart_output = _cartesian_output(inp)
        metrics = _variant_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, output):
        return None
    if np.array_equal(output, cart_output):
        return None
    if metrics["paired_colors"] < 1:
        return None
    if metrics["extra_cartesian_targets"] < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
