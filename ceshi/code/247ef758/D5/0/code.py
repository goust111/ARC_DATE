# Variant of 247ef758
# Dimension: D5 — 轨迹/源保留行为 (trajectory)
# Change: 精灵从左侧条带被擦除（disappear），仅在目标位置出现
#   -> 保留源并添加连线轨迹：原始精灵保留在左侧条带，同时从精灵中心到每个目标位置
#      画一条同色直线作为连接轨迹。
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - Input remains the original layout: left sprite strip | full-height divider | framed area on the right.
# - The frame edges still encode targets exactly as before:
#   matching left/right edge markers choose interior rows, matching top/bottom edge markers choose interior cols,
#   and each mover color stamps at the Cartesian product of those row/col marks.
# - D5 changes only what happens to movers after targets are found.
#   Original local behavior for one mover:
#   1. collect mover pixels in the left strip
#   2. erase those source pixels to bg
#   3. translate and stamp the sprite at every target
#   Variant local behavior:
#   1. collect mover pixels in the left strip
#   2. keep source pixels unchanged
#   3. for every target, draw a same-color discrete straight line from the source bbox center to the target center
#   4. stamp the translated sprite at that target, so the endpoint is visibly occupied by the sprite rather than
#      only by a one-pixel line tip
# - Small trace example:
#   if a red sprite centered at (2,1) has targets (4,9) and (7,11), the original output removes the red blob from
#   the left and only shows red copies at those two destinations. The variant leaves the blob in place and adds two
#   red Bresenham lines from (2,1) to (4,9) and (7,11), then stamps the two red copies. The difference is localized
#   to the mover application stage; target discovery is unchanged.
#
# b) Complexity calibration
# - We keep 2-3 movers and 0-1 distractors, so colors still repeat structurally: each mover color appears on source,
#   frame markers, trail, and target stamps.
# - Trails are single-color and deterministic. No per-color alternative path styles are introduced.
# - The original row/col Cartesian-product freedom remains, so some movers create multiple targets and therefore
#   multiple trails, making the new mechanism visually obvious without adding a separate matching system.
#
# c) Implicit assumptions in the original solver that D5 breaks
# - `out = g.copy()` still works, but the old "erase originals of moved sprites" loop relies on the disappear rule.
#   In D5 that loop is invalid and must be removed.
# - The old stamping loop assumes only translated copies matter. After D5, ordering also affects trail overlaps,
#   so trail drawing must participate in the same bottom-first / top-last painter order.
# - The old code freely writes translated pixels because the source strip has already been cleared. In D5 the source
#   strip must stay readable, so trail drawing cannot overwrite existing left-strip sprite pixels.
# - The original center computation from bbox integer midpoint is still valid because trails and translation both need
#   a single anchor cell. Generator guarantees each target is chosen so the centered stamp fits the frame interior.
#
# d) Placement legality
# - Generator still enforces each mover target center to be far enough from the frame interior border so the centered
#   stamp fits after clipping. Row/col marks remain unique across mover colors, reducing ambiguity in target matching.
# - Left-strip sprites remain non-overlapping connected components. Trails are only added in the output, so input
#   legality is unaffected.
# - Solver protects all existing left-strip sprite pixels while drawing trails, ensuring preserved sources and
#   distractors are not damaged by another mover's path.
#
# e) Visual observability
# - Across only 2-4 examples, a human can now see three repeated cues for each mover color:
#   source blob still present, same-color line(s) extending rightward, and same-color copies at frame-defined targets.
# - No invisible numeric property is used. The same visible color links source, marker, trail, and target.
# - The rule stays single-branch: every mover both keeps its source and emits trails. There is no hidden per-sample
#   switch that would require more examples to infer.
#
# f) Visual cleanliness
# - We keep the original frame and marker encoding, so the scene still decomposes into left source region and right
#   target region.
# - Bottom-to-top painter order is reused for both trail and stamp drawing. Later, higher movers can overwrite earlier
#   lower movers, exactly mirroring the original cover rule rather than introducing a second overlap convention.
# - Generator keeps only 2-3 movers and at most one distractor, which limits path clutter.
#
# g) Color inference
# - No color remapping is introduced. Each mover color stays constant from source to trail to target.
# - Divider color and frame base color remain distinct from mover colors, preserving the original structural cues.
#
# h) Quantitative anti-degeneracy target
# - This variant should differ from the original disappear baseline in two measurable ways on generated samples:
#   1. mover source pixels remain colored in the output instead of becoming bg
#   2. there are extra colored pixels on line paths that were bg in both the input and the disappear baseline
# - Post-implementation tests below measure both counts over random samples.
#
# i) New assumptions introduced by the variant and generator support
# - Main assumes left-strip sprites are detectable exactly as before; generator keeps a single divider column and uses
#   distinct mover colors so each mover remains one 8-connected component.
# - Main assumes source-strip preservation matters only for original sprite pixels, not for empty left-strip bg cells;
#   generator does not place any other semantic objects there.
# - Main assumes row/col target centers exist for each mover color it processes; generator continues to create markers
#   only for actual mover colors and validates `main(generate_input())` on every accepted sample.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve colored sprites in the left strip, draw same-color trails to marker-derived targets,
    and stamp copies of those sprites centered at the targets inside the framed area.

    Key insight: target discovery is unchanged from the original task; only the mover application
    stage changes from "erase and move" to "keep, connect, and replicate".
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- frame interior: largest bg component not touching boundary ---
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    # --- divider columns (fixed) on the left side ---
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)
    protected_left = left_cols & (~divider_mask) & (g != int(bg))

    # --- sprites: per-color 8-connected components in left area (excluding divider) ---
    sprites = []
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
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

    move_colors = {int(s["color"]) for s in sprites}

    # --- marker-derived targets (restricted to colors that actually appear as sprites) ---
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

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    # Bottom-to-top ordering: lower sprites first, upper sprites last (covering).
    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in targets[int(col)]:
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
                    out[int(rr), int(cc3)] = int(col)

    return out


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
                return picked
    return picked if picked else None


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


def _target_count_ok(targets_by_color: dict[int, tuple[list[int], list[int]]]) -> bool:
    total = sum(len(rows) * len(cols) for rows, cols in targets_by_color.values())
    return total >= 3


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    # Dimensions — frame spans full grid height
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

    # Key positions
    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    # Assign distinct colors
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

    # Generate mover sprites
    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    # Pick marker positions per mover (no two movers share a row or col marker)
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    multi_target_mover = False
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        n_row_marks = random.randint(1, 2)
        n_col_marks = random.randint(1, 2)
        row_marks = _pick_spaced(avail_rows, n_row_marks, sh)
        col_marks = _pick_spaced(avail_cols, n_col_marks, sw)
        if not row_marks or not col_marks:
            return None
        if len(row_marks) * len(col_marks) >= 2:
            multi_target_mover = True
        mover_targets[mc] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    if not _target_count_ok(mover_targets):
        return None
    if not multi_target_mover:
        return None

    # Build input grid: bg fill, then divider, frame, sprites
    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    # Frame top/bottom edges (mirrored)
    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[cc - fc0] = mc
    inp[0, fc0: fc0 + frame_w] = top_edge
    inp[H - 1, fc0: fc0 + frame_w] = top_edge

    # Frame left/right edges (mirrored)
    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    # Stack sprites in left strip: movers top-to-bottom, then distractors
    cur_row = 0
    all_items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for _, sp, _ in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    # Validate: bg must be most common color so solver detects it correctly
    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None

    # D5 must visibly differ from input: kept sources plus nontrivial rightward trails/stamps.
    if np.array_equal(inp, output):
        return None
    trail_like_gain = int(np.count_nonzero((inp == bg) & (output != bg)))
    if trail_like_gain < 6:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
