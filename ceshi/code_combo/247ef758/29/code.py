# Combo variant of 247ef758 (index 29)
# combo_spec: D5/0+D11/0
# Name: 轨迹连线+对角目标

from __future__ import annotations

# Design analysis
# ===============
# a) Concrete behavior and trace
# - Shared scene layout stays the same as the source task: left sprite strip | one full-height divider
#   column | framed rectangle on the right whose opposite edges mirror color markers.
# - The fusion changes two linked stages:
#   1. target selection uses only diagonal/order-matched row/column pairs instead of the full Cartesian
#      product
#   2. mover rendering preserves the source sprite and draws a same-color straight trail from the source
#      center to each chosen target before stamping the translated sprite
# - Small trace example:
#   suppose a red sprite has row markers [3, 8] and column markers [10, 14].
#   Source task: targets = {(3,10), (3,14), (8,10), (8,14)}, erase the red sprite on the left, stamp
#   four copies on those centers.
#   Fused task: targets = {(3,10), (8,14)} only, keep the red sprite on the left, draw one red line from
#   the source center to (3,10) and another to (8,14), then stamp two copies at those endpoints.
# - The visible difference from the original is therefore dual:
#   omitted off-diagonal target copies and newly visible source-to-target trails.
#
# b) Complexity calibration
# - The frame marker language remains unchanged, which preserves the original inference scaffold.
# - To make the diagonal rule visible, the generator forces at least one mover to have exactly two row
#   marks and two column marks; this creates a visible 2x2-vs-2-target contrast.
# - To make the trail rule visible, the generator also requires enough added non-bg cells relative to a
#   no-trail diagonal baseline. This avoids samples where the new rule is technically active but visually
#   weak.
# - Colors remain low-cardinality and repeated structurally: each mover color appears on the source
#   sprite, the frame markers, the trail, and the target stamps.
#
# c) Implicit assumptions broken by the fusion
# - The source solver assumes movers disappear; the "erase originals" loop is invalid and must not run.
# - The source solver assumes every row mark combines with every column mark; target construction must be
#   replaced by sorted row/column pairing.
# - Once trails are added, painter order now affects both trail overlaps and stamp overlaps. We therefore
#   keep the source task's bottom-first / top-last order for the entire rendering phase.
# - Preserving sources introduces a new constraint: drawing a trail must not overwrite any pre-existing
#   left-strip sprite pixel, otherwise the source-retention cue becomes unreadable. The renderer protects
#   all non-bg, non-divider cells in the left strip.
#
# d) Placement legality
# - Generator still chooses marker rows/cols from sprite-size-dependent valid ranges so each centered
#   stamp fits inside the frame interior.
# - Marker rows and columns remain unique across colors, so movers do not compete for the same target
#   centers.
# - Each mover is generated as one 8-connected component inside the left strip, and sprites are stacked
#   without overlap.
# - Equal row/column counts are generated per mover, so diagonal pairing is unambiguous and leaves no
#   leftover markers.
#
# e) Visual observability
# - A solver looking at 2-4 examples can see that mover colors now have three roles at once:
#   left source blob remains, lines emanate rightward, and only some marker intersections are used.
# - The diagonal rule is observable because a 2-row + 2-column color leaves the off-diagonal locations
#   empty despite having all four edge markers present.
# - No invisible feature such as numeric color ordering is used; only spatial order on the frame edges
#   determines the row/column pairing.
#
# f) Visual cleanliness
# - The scene still separates into a left source region and a right target region, with the divider and
#   frame keeping those roles legible.
# - Bottom-first painter order means overlaps follow one consistent convention instead of introducing a
#   second arbitrary rule for trails.
# - The generator keeps only 2-3 movers and at most one distractor, limiting clutter from multiple lines.
#
# g) Color inferability
# - No color remapping is introduced. A mover keeps one color across source, markers, trail, and target.
# - Divider and frame-base colors remain distinct from mover colors, preserving the original partitioning
#   cues.
#
# h) Quantitative anti-degeneracy targets used by the generator
# - D11 activity: at least one mover has two diagonal pairs, and the fused output differs from a
#   cartesian-target trail baseline.
# - D5 activity: the fused output differs from a diagonal no-trail baseline, with both preserved source
#   pixels and extra trail-only pixels.
#
# i) New solver assumptions introduced by the fusion and how the generator satisfies them
# - `_build_targets(..., mode="diagonal")` assumes rows and columns can be paired in sorted order;
#   generator creates equal counts per mover.
# - Trail drawing assumes left-strip semantic pixels should stay untouched; generator places all source
#   semantics exclusively in that strip, so protecting it is correct.
# - Metrics compare the fused output to single-dimension baselines; generator explicitly computes those
#   baselines and rejects accidental matches.

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


def _discrete_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer grid line including both endpoints."""
    r, c = int(r0), int(c0)
    r1, c1 = int(r1), int(c1)
    pts = [(r, c)]
    dr = abs(int(r1 - r))
    dc = abs(int(c1 - c))
    sr = 0 if r == r1 else (1 if r1 > r else -1)
    sc = 0 if c == c1 else (1 if c1 > c else -1)

    if dc >= dr:
        err = dc // 2
        while c != c1:
            c += sc
            err -= dr
            if err < 0:
                r += sr
                err += dc
            pts.append((int(r), int(c)))
    else:
        err = dr // 2
        while r != r1:
            r += sr
            err -= dc
            if err < 0:
                c += sc
                err += dr
            pts.append((int(r), int(c)))
    return pts


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    """
    Parse the shared source-task scene structure.

    Assumptions required by the fused solver:
    - The frame interior is the largest enclosed bg component.
    - Left-strip sprites are the only non-divider components left of the frame.
    - Any mover color intended for rendering appears both in the left strip and on matching opposite
      frame edges.
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
    protected_left = left_cols & (~divider_mask) & (g != int(bg))

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
        "protected_left": protected_left,
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


def _render(
    scene: dict[str, object],
    targets: dict[int, list[tuple[int, int]]],
    *,
    keep_sources: bool,
    draw_trails: bool,
) -> np.ndarray:
    g = np.array(scene["g"], dtype=int)
    out = g.copy()
    H = int(scene["H"])
    W = int(scene["W"])
    bg = int(scene["bg"])
    ir0 = int(scene["ir0"])
    ic0 = int(scene["ic0"])
    ir1 = int(scene["ir1"])
    ic1 = int(scene["ic1"])
    protected_left = np.array(scene["protected_left"], dtype=bool)
    sprites = list(scene["sprites"])

    movers = [s for s in sprites if int(s["color"]) in targets]

    if not keep_sources:
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
            if draw_trails:
                for rr, cc2 in _discrete_line(int(cr), int(cc), int(tr), int(tc)):
                    if not (0 <= int(rr) < int(H) and 0 <= int(cc2) < int(W)):
                        continue
                    if bool(protected_left[int(rr), int(cc2)]):
                        continue
                    out[int(rr), int(cc2)] = int(col)

            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def _cartesian_trail_output(grid: np.ndarray) -> np.ndarray:
    scene = _extract_scene(grid)
    targets = _build_targets(scene["row_marks"], scene["col_marks"], mode="cartesian")
    return _render(scene, targets, keep_sources=True, draw_trails=True)


def _diagonal_disappear_output(grid: np.ndarray) -> np.ndarray:
    scene = _extract_scene(grid)
    targets = _build_targets(scene["row_marks"], scene["col_marks"], mode="diagonal")
    return _render(scene, targets, keep_sources=False, draw_trails=False)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Metrics used both for generator rejection and for post-hoc validation.

    - `paired_colors`: mover colors with at least two row marks and two column marks.
    - `extra_cartesian_targets`: how many extra target centers the full Cartesian rule would add.
    - `preserved_source_pixels`: mover source pixels present in fused output but erased in a diagonal
      disappear baseline.
    - `trail_only_pixels`: non-source extra pixels added by trails relative to the same baseline.
    """
    scene = _extract_scene(grid)
    bg = int(scene["bg"])
    row_marks = scene["row_marks"]
    col_marks = scene["col_marks"]
    fused_targets = _build_targets(row_marks, col_marks, mode="diagonal")
    fused_out = _render(scene, fused_targets, keep_sources=True, draw_trails=True)
    diag_disappear = _render(scene, fused_targets, keep_sources=False, draw_trails=False)

    mover_source_mask = np.zeros_like(np.array(scene["g"], dtype=int), dtype=bool)
    for s in scene["sprites"]:
        if int(s["color"]) not in fused_targets:
            continue
        r0 = int(s["r0"])
        c0 = int(s["c0"])
        mask = np.array(s["mask"], dtype=bool)
        mover_source_mask[int(r0) : int(r0) + mask.shape[0], int(c0) : int(c0) + mask.shape[1]] |= mask

    paired_colors = 0
    extra_cartesian_targets = 0
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        nr = len(row_marks[int(col)])
        nc = len(col_marks[int(col)])
        if nr >= 2 and nc >= 2:
            paired_colors += 1
        extra_cartesian_targets += max(0, nr * nc - min(nr, nc))

    gained_vs_diag = (diag_disappear == int(bg)) & (fused_out != int(bg))
    preserved_source_pixels = int(np.count_nonzero(gained_vs_diag & mover_source_mask))
    trail_only_pixels = int(np.count_nonzero(gained_vs_diag & (~mover_source_mask)))

    return {
        "paired_colors": int(paired_colors),
        "extra_cartesian_targets": int(extra_cartesian_targets),
        "preserved_source_pixels": int(preserved_source_pixels),
        "trail_only_pixels": int(trail_only_pixels),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused D5/0 + D11/0 solver for 247ef758.

    For each mover color:
    - detect matching frame row markers and matching frame column markers
    - pair them in sorted order to get diagonal targets only
    - keep the source sprite in the left strip
    - draw same-color straight trails from the source center to each target
    - stamp the sprite centered at those targets
    """
    scene = _extract_scene(grid)
    targets = _build_targets(scene["row_marks"], scene["col_marks"], mode="diagonal")
    return _render(scene, targets, keep_sources=True, draw_trails=True)


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick exactly n values from candidates with at least min_spacing between any two."""
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
    return None


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
    total_diagonal_targets = 0

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

        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)
        total_diagonal_targets += int(n_pairs)

    if total_diagonal_targets < 3:
        return None

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
                val = int(mc)
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
        fused_out = main(inp)
        cartesian_trail = _cartesian_trail_output(inp)
        diagonal_disappear = _diagonal_disappear_output(inp)
        metrics = _fusion_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, fused_out):
        return None
    if np.array_equal(fused_out, cartesian_trail):
        return None
    if np.array_equal(fused_out, diagonal_disappear):
        return None
    if metrics["paired_colors"] < 1:
        return None
    if metrics["extra_cartesian_targets"] < 2:
        return None
    if metrics["preserved_source_pixels"] < 3:
        return None
    if metrics["trail_only_pixels"] < 6:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
