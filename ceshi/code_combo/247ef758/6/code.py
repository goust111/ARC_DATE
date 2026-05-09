# Combo variant of 247ef758 (index 6)
# combo_spec: D3/0+D11/0
# Name: 内部标记对角配对

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    Source task:
#    - A mover color X appears as a sprite in the left strip.
#    - The right frame encodes X on both vertical edges at selected rows and on
#      both horizontal edges at selected columns.
#    - Every selected row combines with every selected column, so rows [r1, r2]
#      and cols [c1, c2] produce four centers:
#      (r1, c1), (r1, c2), (r2, c1), (r2, c2).
#    Combo variant:
#    - The border becomes fully uniform again. Informative marks move inside the
#      frame onto one guide column and one guide row.
#    - A row marker for X is now a small internal glyph centered at
#      (target_row, guide_col); a column marker is centered at
#      (guide_row, target_col). Glyphs are dots or tiny plus-shapes, but their
#      center cell always lies on the guide lane, so the solver can read the
#      same center coordinate regardless of style.
#    - Targets are no longer the full Cartesian product. Sorted row marks pair
#      with sorted column marks index-wise, so [r1, r2] and [c1, c2] now produce
#      only (r1, c1) and (r2, c2).
#    - Example delta:
#         source: edge rows/cols -> 4 copies
#         combo : internal rows/cols -> 2 diagonal/order-matched copies
#      The solver must therefore both look inside the frame and infer that only
#      the order-matched intersections are used.
#
# b) Complexity calibration
#    - Keep 2-3 mover colors and 0-1 distractors from the source task so color
#      matching remains repeated and structural rather than one-off.
#    - Force at least one mover to have two row marks and two column marks. This
#      makes the diagonal-vs-Cartesian difference visually observable.
#    - Keep row/column marks color-coded and unique across movers so pairing is
#      inferable without introducing extra symbolic machinery.
#
# c) Implicit-assumption audit
#    - `largest_enclosed_component_bbox(g == bg)` still works because internal
#      dots/pluses are small isolated obstacles inside the frame background and
#      do not touch the outer grid border.
#    - Divider detection and left-strip sprite extraction are unchanged because
#      they never depended on edge markers.
#    - The original row/col extraction loops depended on mirrored colored frame
#      edges; that assumption is broken. In the combo solver they instead read
#      the internal guide column/row.
#    - The original target builder assumed full Cartesian expansion; that is
#      replaced by sorted zip-pairing.
#    - Rendering order, clipping, and erase logic remain valid once target
#      centers are computed.
#
# d) Placement legality
#    - The generator reserves one internal guide row and one internal guide
#      column for markers.
#    - Target centers are sampled with extra clearance from those guide lanes so
#      moved sprites do not overwrite the internal marker glyphs.
#    - Marker rows/cols are never reused across colors, and counts per color are
#      equal on both axes, making diagonal pairing well-defined.
#
# e) Visual observability
#    - Compared with the source task, the border is now uniformly one color; the
#      evidence moved visibly inside the frame.
#    - Compared with a pure D3-style internal-marker variant, at least one mover
#      with a 2x2 marker set now produces only two outputs instead of four. The
#      omitted off-diagonal centers are directly observable across examples.
#
# f) Visual cleanliness
#    - Marks live only on one guide row/column near the top-left of the frame.
#      The rest of the frame stays uncluttered until stamping.
#    - Glyphs are tiny and sparse, and the generator enforces spacing so adjacent
#      marks do not merge into ambiguous blobs.
#
# g) Change inferability
#    - No color remapping is introduced. Colors still identify which sprite a
#      marker belongs to.
#    - The only new reasoning burden is geometric: internal search for marks and
#      diagonal pairing by order rather than full combination.
#
# New solver assumptions introduced by the combo and guaranteed by the generator
# - The first usable internal guide row/column stay mostly background except for
#   mover-colored marker centers and optional one-cell plus arms.
# - Each mover color has equally many row and column marks.
# - At least one mover has two marks on both axes, and the diagonal output
#   differs from the internal Cartesian interpretation on generated samples.

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does not touch the outer grid boundary.
    """
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
        if bool(
            (rs == 0).any()
            or (rs == int(h - 1)).any()
            or (cs == 0).any()
            or (cs == int(w - 1)).any()
        ):
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
    Parse the shared scene structure for the combo variant.

    The frame is found exactly as in the source task via the largest enclosed
    background component. Internal markers are then read from one guide column
    and one guide row inside that frame.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fc0 = int(ic0 - 1)
    fc1 = int(ic1 + 1)
    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    assert guide_row <= int(ir1) and guide_col <= int(ic1)

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
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
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
        val = int(g[int(r), int(guide_col)])
        if val != int(bg) and val in move_colors:
            row_marks[int(val)].append(int(r))

    for c in range(int(ic0), int(ic1) + 1):
        val = int(g[int(guide_row), int(c)])
        if val != int(bg) and val in move_colors:
            col_marks[int(val)].append(int(c))

    return {
        "g": g,
        "h": int(h),
        "w": int(w),
        "bg": int(bg),
        "ir0": int(ir0),
        "ic0": int(ic0),
        "ir1": int(ir1),
        "ic1": int(ic1),
        "fc0": int(fc0),
        "fc1": int(fc1),
        "guide_row": int(guide_row),
        "guide_col": int(guide_col),
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
            raise ValueError(f"unknown mode: {mode}")
        if pts:
            targets[int(col)] = pts
    return targets


def _render_targets(
    scene: dict[str, object], targets: dict[int, list[tuple[int, int]]]
) -> np.ndarray:
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
        pix = [
            (int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2))
            for rr, cc2 in np.argwhere(np.array(s["mask"], dtype=bool))
        ]

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


def _frame_border_values(scene: dict[str, object]) -> np.ndarray:
    g = np.array(scene["g"], dtype=int)
    ir0 = int(scene["ir0"])
    ic0 = int(scene["ic0"])
    ir1 = int(scene["ir1"])
    ic1 = int(scene["ic1"])
    fr0 = int(ir0 - 1)
    fr1 = int(ir1 + 1)
    fc0 = int(ic0 - 1)
    fc1 = int(ic1 + 1)
    vals = [
        *map(int, g[int(fr0), int(fc0) : int(fc1) + 1]),
        *map(int, g[int(fr1), int(fc0) : int(fc1) + 1]),
        *map(int, g[int(fr0) + 1 : int(fr1), int(fc0)]),
        *map(int, g[int(fr0) + 1 : int(fr1), int(fc1)]),
    ]
    return np.array(vals, dtype=int)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the combo's two key behaviors are both active.

    Metrics:
    - `uniform_border`: 1 iff the frame border carries no informative colors.
    - `internal_marks`: total number of row+column marker centers found.
    - `paired_colors`: mover colors with at least two marks on both axes.
    - `extra_cartesian_targets`: targets the internal Cartesian rule would add.
    - `diff_vs_cartesian`: output pixel count that differs from internal
      Cartesian stamping, confirming diagonal pairing matters.
    """
    scene = _extract_scene(grid)
    border_vals = _frame_border_values(scene)
    bg = int(scene["bg"])
    row_marks = scene["row_marks"]
    col_marks = scene["col_marks"]
    diag = main(grid)
    cart = _cartesian_output(grid)

    paired_colors = 0
    extra_cartesian_targets = 0
    internal_marks = 0
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        nr = len(row_marks[int(col)])
        nc = len(col_marks[int(col)])
        internal_marks += nr + nc
        if nr >= 2 and nc >= 2:
            paired_colors += 1
        extra_cartesian_targets += max(0, nr * nc - min(nr, nc))

    uniform_border = int(len(set(map(int, border_vals))) == 1 and int(border_vals[0]) != bg)
    diff_vs_cartesian = int(np.count_nonzero(diag != cart))
    return {
        "uniform_border": int(uniform_border),
        "internal_marks": int(internal_marks),
        "paired_colors": int(paired_colors),
        "extra_cartesian_targets": int(extra_cartesian_targets),
        "diff_vs_cartesian": int(diff_vs_cartesian),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Read row/column markers from internal guide lanes inside the frame.
    - Pair sorted row marks and sorted column marks index-wise per color.
    - Erase mover sprites from the left strip and stamp them at those diagonal
      pair targets, preserving the original bottom-over-top overwrite order.
    """
    scene = _extract_scene(grid)
    targets = _build_targets(scene["row_marks"], scene["col_marks"], mode="diagonal")
    return _render_targets(scene, targets)


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick exactly n values from candidates with at least min_spacing separation."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return sorted(picked)
    return None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bbox."""
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


def _paint_marker(
    grid: np.ndarray,
    center_r: int,
    center_c: int,
    color: int,
    *,
    style: str,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
) -> None:
    """Paint a dot or tiny plus marker inside the frame interior."""
    pts = [(int(center_r), int(center_c))]
    if style == "plus":
        pts.extend(
            [
                (int(center_r - 1), int(center_c)),
                (int(center_r + 1), int(center_c)),
                (int(center_r), int(center_c - 1)),
                (int(center_r), int(center_c + 1)),
            ]
        )
    elif style != "dot":
        raise ValueError(f"unknown marker style: {style}")

    for r, c in pts:
        if int(ir0) <= r <= int(ir1) and int(ic0) <= c <= int(ic1):
            grid[int(r), int(c)] = int(color)


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
    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    if guide_row > ir1 or guide_col > ic1:
        return None

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.choice([2, 2, 3])
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
        top_extent = sh // 2
        bottom_extent = sh - 1 - top_extent
        left_extent = sw // 2
        right_extent = sw - 1 - left_extent

        min_row = guide_row + top_extent + 2
        max_row = ir1 - bottom_extent
        min_col = guide_col + left_extent + 2
        max_col = ic1 - right_extent
        if min_row > max_row or min_col > max_col:
            return None

        n_pairs = 2 if i == forced_double else random.choice([1, 2])
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        row_marks = _pick_spaced(avail_rows, n_pairs, max(3, sh))
        col_marks = _pick_spaced(avail_cols, n_pairs, max(3, sw))
        if row_marks is None or col_marks is None:
            return None

        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = divider_color

    inp[0, fc0 : fc1 + 1] = frame_base
    inp[h - 1, fc0 : fc1 + 1] = frame_base
    inp[:, fc0] = frame_base
    inp[:, fc1] = frame_base

    row_styles = {mc: random.choice(["dot", "plus"]) for mc in mover_colors}
    col_styles = {mc: random.choice(["dot", "plus"]) for mc in mover_colors}
    if all(style == "dot" for style in list(row_styles.values()) + list(col_styles.values())):
        forced_color = random.choice(mover_colors)
        row_styles[int(forced_color)] = "plus"

    for mc, (rmarks, cmarks) in mover_targets.items():
        for r in rmarks:
            _paint_marker(
                inp,
                int(r),
                int(guide_col),
                int(mc),
                style=row_styles[int(mc)],
                ir0=int(ir0),
                ic0=int(ic0),
                ir1=int(ir1),
                ic1=int(ic1),
            )
        for c in cmarks:
            _paint_marker(
                inp,
                int(guide_row),
                int(c),
                int(mc),
                style=col_styles[int(mc)],
                ir0=int(ir0),
                ic0=int(ic0),
                ir1=int(ir1),
                ic1=int(ic1),
            )

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(mc, sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds))

    for _color, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
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
    if metrics["uniform_border"] != 1:
        return None
    if metrics["internal_marks"] < 4:
        return None
    if metrics["paired_colors"] < 1:
        return None
    if metrics["extra_cartesian_targets"] < 2:
        return None
    if metrics["diff_vs_cartesian"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
