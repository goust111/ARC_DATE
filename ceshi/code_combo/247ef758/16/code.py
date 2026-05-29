# Combo variant of 247ef758 (index 16)
# combo_spec: D3/0+D11/0+D12/0
# Name: 内部标记+对角目标+左上对齐

from __future__ import annotations

# Design analysis for the fused combo
# ----------------------------------
# a) Concrete behavior and trace
# - Base task: a left sprite strip sits to the left of a full-height rectangular
#   frame. In the source task, mirrored edge colors on the frame border encode
#   target rows and target columns. For one sprite color X with rows [r1, r2]
#   and cols [c1, c2], the source task stamps X at the full Cartesian product
#   {(r1, c1), (r1, c2), (r2, c1), (r2, c2)} and aligns the sprite center to
#   each target.
# - Fused combo:
#   1. D3/0 moves the markers off the border and into the frame interior. The
#      first inner guide column contains row-marker dots, and the first inner
#      guide row contains column-marker dots. A row marker for color X is a dot
#      at (r, guide_col); a column marker is a dot at (guide_row, c).
#   2. D11/0 changes target construction from Cartesian product to diagonal
#      pairing. If color X has sorted rows [r1, r2] and sorted cols [c1, c2],
#      the used targets are only [(r1, c1), (r2, c2)].
#   3. D12/0 changes stamping from center alignment to top-left alignment. If a
#      sprite has bbox shape 3x4 and a target is (r1, c1), the sprite occupies
#      rows r1..r1+2 and cols c1..c1+3.
# - Small trace: suppose a blue sprite has two internal row dots at rows 4 and 8
#   on the guide column, and two internal column dots at cols 10 and 14 on the
#   guide row. The source task would have read border marks, produced four
#   center-aligned targets, and filled a 2x2 grid. The combo reads the two
#   internal dot lists instead, pairs them as (4,10) and (8,14), and places the
#   blue sprite with its top-left corner anchored exactly at those two points.
#   The whole "where to look -> how to pair -> how to anchor" pipeline changes.
#
# b) Complexity calibration
# - The scene vocabulary stays close to the source task: 2-3 movers, optional
#   distractor, one divider column, one frame, and same-color sprites/markers.
# - To keep the diagonal rule visible, the generator forces at least one mover to
#   have two row markers and two column markers, so off-diagonal Cartesian
#   positions are visibly omitted.
# - Internal markers are single-pixel dots on fixed guide lanes. That keeps the
#   frame readable while still making the solver search inside the box rather
#   than on the border.
#
# c) Implicit-assumption audit
# - `largest_enclosed_component_bbox(g == bg)` assumes the frame interior is the
#   largest enclosed background component. Internal dots do not block that under
#   4-connectivity, so the bbox logic remains valid.
# - Divider detection and left-strip sprite extraction never depended on border
#   markers, so they can be reused directly.
# - The source row/column scans assumed mirrored border marks. In the combo,
#   those lines must read from the internal guide column / guide row instead.
# - The source target builder assumed Cartesian expansion. In the combo, rows and
#   cols must be sorted and zipped one-by-one.
# - The source renderer assumed target = center. In the combo, target = bbox
#   top-left, so the base placement offset logic must change.
# - New combined assumption: guide-row / guide-col markers must survive in the
#   output so humans can observe the rule. The generator enforces target rows
#   strictly below the guide row and target cols strictly right of the guide
#   col, so top-left anchored stamps never overwrite the guide-lane dots.
#
# d) Placement legality
# - For a sprite of shape (sh, sw), valid top-left targets are restricted to
#   `guide_row + 1 .. ir1 - sh + 1` and `guide_col + 1 .. ic1 - sw + 1`.
#   This guarantees every stamp fits fully inside the frame and stays clear of
#   the marker guide lanes.
# - Different mover colors never share row markers or column markers, so the
#   marker decoding remains unambiguous.
# - Within one mover, row marks are spaced by at least the sprite height and
#   column marks by at least the sprite width. That prevents same-color diagonal
#   copies from collapsing into one merged blob.
#
# e) Visual observability
# - Humans can see three independent but coordinated differences across examples:
#   the border is uniform, the informative dots are inside the frame, a 2x2 mark
#   set yields only two copies instead of four, and each copy starts at the
#   marked point rather than being centered around it.
# - The generator rejects samples where the fused output matches either the
#   internal-marker Cartesian version or the internal-marker centered version.
#   That guarantees D11 and D12 are both visually active.
#
# f) Visual cleanliness
# - Only one internal guide row and one internal guide column carry dots; the
#   rest of the frame stays sparse.
# - Because all top-left anchors are strictly below/right of the guide lanes, the
#   marker dots remain visible in the output instead of being buried under the
#   copies.
# - A little inter-color overlap is still allowed and explained by the preserved
#   bottom-to-top overwrite rule, but the generator avoids same-color self-chaos.
#
# g) Color inferability
# - No recoloring is introduced. A marker color still refers to the sprite of the
#   same color. The only changed information is marker location, target pairing,
#   and anchor choice.
#
# h) New solver assumptions checked against the generator
# - Solver assumption: a fixed internal guide row/col exists at `ir0+1` / `ic0+1`.
#   Generator guarantee: frame interior is always at least 8x8, so those lanes
#   exist and are reserved for marker dots.
# - Solver assumption: intended diagonal pairing is the sorted row list zipped
#   with the sorted column list. Generator guarantee: each mover uses equal row
#   and column counts.
# - Solver assumption: target coordinates can serve as sprite top-left anchors.
#   Generator guarantee: target sampling uses top-left-fit bounds, not center-fit
#   bounds.

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Return the bbox of the largest connected component in `mask`
    that does not touch the outer grid border.
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
    Parse the common scene structure for the fused combo.

    This parser assumes:
    - the right frame encloses the largest non-border-touching bg component,
    - the left strip contains 8-connected single-color sprites,
    - internal markers live on one guide row and one guide column.
    """
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
            cropped = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": cropped,
                }
            )

    move_colors = {int(s["color"]) for s in sprites}
    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    assert guide_row <= int(ir1) and guide_col <= int(ic1)

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
        "fr0": int(fr0),
        "fc0": int(fc0),
        "fr1": int(fr1),
        "fc1": int(fc1),
        "ir0": int(ir0),
        "ic0": int(ic0),
        "ir1": int(ir1),
        "ic1": int(ic1),
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
    target_mode: str,
) -> dict[int, list[tuple[int, int]]]:
    targets: dict[int, list[tuple[int, int]]] = {}
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        rows = sorted(map(int, row_marks[int(col)]))
        cols = sorted(map(int, col_marks[int(col)]))
        if target_mode == "diagonal":
            pts = [(int(r), int(c)) for r, c in zip(rows, cols)]
        elif target_mode == "cartesian":
            pts = sorted({(int(r), int(c)) for r in rows for c in cols})
        else:
            raise ValueError(f"unknown target_mode: {target_mode}")
        if pts:
            targets[int(col)] = pts
    return targets


def _render_targets(
    scene: dict[str, object],
    targets: dict[int, list[tuple[int, int]]],
    *,
    anchor_mode: str,
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

    movers.sort(
        key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"]))
    )

    if anchor_mode not in {"center", "topleft"}:
        raise ValueError(f"unknown anchor_mode: {anchor_mode}")

    for s in movers:
        col = int(s["color"])
        mask = np.array(s["mask"], dtype=bool)
        sh, sw = mask.shape
        if anchor_mode == "center":
            anchor_r = (int(sh) - 1) // 2
            anchor_c = (int(sw) - 1) // 2
        else:
            anchor_r = 0
            anchor_c = 0

        for tr, tc in targets[int(col)]:
            base_r = int(tr - anchor_r)
            base_c = int(tc - anchor_c)
            for rr, cc in np.argwhere(mask):
                pr = int(base_r + int(rr))
                pc = int(base_c + int(cc))
                if int(ir0) <= pr <= int(ir1) and int(ic0) <= pc <= int(ic1):
                    out[pr, pc] = int(col)

    return out


def _solve_variant(
    grid: np.ndarray,
    *,
    target_mode: str,
    anchor_mode: str,
) -> np.ndarray:
    scene = _extract_scene(grid)
    targets = _build_targets(
        scene["row_marks"],
        scene["col_marks"],
        target_mode=target_mode,
    )
    return _render_targets(scene, targets, anchor_mode=anchor_mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Read row/column marker dots from fixed internal guide lanes.
    - Pair same-color row/column marks diagonally by sorted order.
    - Stamp each sprite with its bbox top-left corner anchored at each paired target.
    - Lower input sprites stamp first; higher input sprites overwrite them.
    """
    return _solve_variant(grid, target_mode="diagonal", anchor_mode="topleft")


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick exactly n values from candidates with pairwise spacing >= min_spacing."""
    if len(candidates) < n:
        return None
    if n == 1:
        return [random.choice(candidates)]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return sorted(picked)
    return None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a cropped 8-connected single-color sprite."""
    sh = min(random.choice([3, 3, 3, 4, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None

    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
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
        sprite[int(pr - min_r), int(pc - min_c)] = int(color)
    return sprite


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the combo is actually active on a sample.

    Metrics:
    - `internal_marker_pixels`: mover-colored dots on the internal guide lanes.
    - `edge_mover_marker_pixels`: mover-colored pixels on the frame border.
      This should be zero in this combo.
    - `paired_colors`: mover colors with at least 2 row marks and 2 col marks.
    - `omitted_cartesian_targets`: target count removed by diagonal pairing.
    - `diff_vs_cartesian`: output pixel diffs between diagonal+topleft and
      cartesian+topleft on the same internal-marker input.
    - `diff_vs_center`: output pixel diffs between diagonal+topleft and
      diagonal+center on the same internal-marker input.
    """
    scene = _extract_scene(grid)
    g = np.array(scene["g"], dtype=int)
    move_colors = {int(s["color"]) for s in scene["sprites"]}
    fr0 = int(scene["fr0"])
    fc0 = int(scene["fc0"])
    fr1 = int(scene["fr1"])
    fc1 = int(scene["fc1"])
    ir0 = int(scene["ir0"])
    ic0 = int(scene["ic0"])
    ir1 = int(scene["ir1"])
    ic1 = int(scene["ic1"])
    guide_row = int(scene["guide_row"])
    guide_col = int(scene["guide_col"])
    row_marks = scene["row_marks"]
    col_marks = scene["col_marks"]

    internal_marker_pixels = 0
    for r in range(int(ir0), int(ir1) + 1):
        if int(g[int(r), int(guide_col)]) in move_colors:
            internal_marker_pixels += 1
    for c in range(int(ic0), int(ic1) + 1):
        if int(g[int(guide_row), int(c)]) in move_colors:
            internal_marker_pixels += 1

    edge_positions = []
    for c in range(int(fc0), int(fc1) + 1):
        edge_positions.append((int(fr0), int(c)))
        edge_positions.append((int(fr1), int(c)))
    for r in range(int(fr0), int(fr1) + 1):
        edge_positions.append((int(r), int(fc0)))
        edge_positions.append((int(r), int(fc1)))
    edge_mover_marker_pixels = sum(
        1 for r, c in edge_positions if int(g[int(r), int(c)]) in move_colors
    )

    paired_colors = 0
    omitted_cartesian_targets = 0
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        nr = len(row_marks[int(col)])
        nc = len(col_marks[int(col)])
        if nr >= 2 and nc >= 2:
            paired_colors += 1
        omitted_cartesian_targets += max(0, nr * nc - min(nr, nc))

    combo_out = _solve_variant(grid, target_mode="diagonal", anchor_mode="topleft")
    cart_out = _solve_variant(grid, target_mode="cartesian", anchor_mode="topleft")
    center_out = _solve_variant(grid, target_mode="diagonal", anchor_mode="center")

    return {
        "internal_marker_pixels": int(internal_marker_pixels),
        "edge_mover_marker_pixels": int(edge_mover_marker_pixels),
        "paired_colors": int(paired_colors),
        "omitted_cartesian_targets": int(omitted_cartesian_targets),
        "diff_vs_cartesian": int(np.count_nonzero(combo_out != cart_out)),
        "diff_vs_center": int(np.count_nonzero(combo_out != center_out)),
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
    guide_row = ir0 + 1
    guide_col = ic0 + 1
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
        sprite = _make_sprite(mc, 5, sprite_area_w)
        if sprite is None:
            return None
        sprites.append(sprite)

    forced_double = random.randrange(n_movers)
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()

    for i, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        sh, sw = sp.shape
        min_row = guide_row + 1
        max_row = ir1 - int(sh) + 1
        min_col = guide_col + 1
        max_col = ic1 - int(sw) + 1
        if min_row > max_row or min_col > max_col:
            return None

        n_pairs = 2 if i == forced_double else random.choice([1, 2])
        avail_rows = [
            r for r in range(min_row, max_row + 1) if int(r) not in occupied_rows
        ]
        avail_cols = [
            c for c in range(min_col, max_col + 1) if int(c) not in occupied_cols
        ]
        row_marks = _pick_spaced(avail_rows, n_pairs, max(1, int(sh)))
        col_marks = _pick_spaced(avail_cols, n_pairs, max(1, int(sw)))
        if row_marks is None or col_marks is None:
            return None

        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.update(map(int, row_marks))
        occupied_cols.update(map(int, col_marks))

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    inp[0, fc0 : fc1 + 1] = int(frame_base)
    inp[h - 1, fc0 : fc1 + 1] = int(frame_base)
    inp[:, int(fc0)] = int(frame_base)
    inp[:, int(fc1)] = int(frame_base)

    for mc, (row_marks, col_marks) in mover_targets.items():
        for r in row_marks:
            inp[int(r), int(guide_col)] = int(mc)
        for c in col_marks:
            inp[int(guide_row), int(c)] = int(mc)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [
        (int(mc), sp) for mc, sp in zip(mover_colors, sprites)
    ]
    for dc_color in distractor_colors:
        sprite = _make_sprite(int(dc_color), min(5, h - cur_row), sprite_area_w)
        if sprite is None:
            return None
        all_items.append((int(dc_color), sprite))

    for _color, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[int(dr), int(dc)])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        output = main(inp)
        metrics = _variant_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, output):
        return None
    if metrics["edge_mover_marker_pixels"] != 0:
        return None
    if metrics["internal_marker_pixels"] < 4:
        return None
    if metrics["paired_colors"] < 1:
        return None
    if metrics["omitted_cartesian_targets"] < 2:
        return None
    if metrics["diff_vs_cartesian"] <= 0:
        return None
    if metrics["diff_vs_center"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
