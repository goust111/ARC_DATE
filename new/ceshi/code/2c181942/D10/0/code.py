# Variant of 2c181942
# Dimension: D10 — hub选择与定位策略 (meta)
# Change: hub放置在网格中心附近，solver按照距中心最近的候选选取hub... → 多个hub竞争选择

from __future__ import annotations

import random
import sys
from typing import Any

import numpy as np
from scipy.ndimage import label as _label

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D10 variant
# ---------------------------------------------------------------------------
#
# variant_analysis:
#   concrete_input:
#     - The input now contains 2-3 hollow 4x4 squares instead of a single central one.
#     - Exactly one square is the real hub. Its four edge colors are all distinct and the
#       scattered monochrome sprites use a subset of those edge colors.
#     - A strong distractor hub is also present. It is intentionally placed closer to the
#       grid center than the true hub, and it also looks structurally valid as a hollow
#       square. The difference is semantic: its edge colors do not match any sprite color.
#     - An optional second distractor hub may appear; it keeps the same square geometry but
#       repeats at least one edge color, so it also fails the "real hub" role.
#     - Output still keeps only the true hub plus the attached sprites. All distractor hubs
#       disappear.
#   behavior_trace:
#     - Original `_find_hub()`:
#       1. enumerate hollow-square candidates
#       2. choose the one closest to the grid center
#       3. attach sprites by color
#     - Variant:
#       1. enumerate all hollow-square candidates
#       2. exclude all candidate hub boxes from sprite search, so hub edges are never
#          mistaken for sprites
#       3. for each candidate, try to attach matching-color sprites side by side using the
#          original rotation/symmetry rules
#       4. choose the candidate with the most successful attachments; break ties by favoring
#          all-distinct edge colors, then by the old center heuristic
#     - Example:
#       * center decoy edges = {1,2,3,4}, true off-center hub edges = {5,6,7,9}
#       * scattered sprites use colors {5,7,9}
#       * old solver would lock onto the center decoy and produce only the decoy square
#       * new solver gives the decoy score 0, the true hub score 3, and attaches 3 sprites
#   complexity_calibration:
#     - The visual change is obvious because at least one wrong hub is present and it is
#       usually the closest one to the center.
#     - The decisive feature is low-arity and visible: matching sprite colors vs non-matching
#       colors. No invisible ordering by raw color id is used.
#     - The generator keeps the original sprite logic unchanged, so D10 remains the dominant
#       new source of difficulty.
#   implicit_assumption_audit:
#     - Old assumption: "closest hollow square is the hub" no longer holds. Replaced with
#       candidate scoring based on successful sprite attachments.
#     - Old code searched `(g == color) & (~hub_mask)` for each side. With multiple hubs this
#       would treat other hub edges as potential sprites. The variant excludes the union of
#       all candidate hub boxes from sprite search.
#     - Old loop could safely reuse the same color once because true hub colors were unique in
#       practice. The new helper tracks used sprite components, so even a repeated-color decoy
#       cannot count the same sprite twice.
#   placement_legality:
#     - The generator reserves full 4x4 hub bounding boxes, not only colored edge pixels.
#       This prevents scattered sprites from landing inside decoy interiors and preserves the
#       hollow-square pattern for the solver.
#     - One distractor hub is chosen to be closer to the center than the true hub whenever
#       the grid allows it, ensuring the D10 behavior is actually exercised.
#   visual_observability:
#     - Across examples, the solver must ignore a hub-like square that is visually salient but
#       color-incompatible with the sprite set. This is inferable from 2-4 examples because
#       the surviving hub is the one whose edge colors reappear in the detached pieces.
#   neatness:
#     - Hubs are separated by bbox padding and sprite scattering respects additional padding,
#       so objects do not visually merge.
#   color_inference:
#     - Color use stays structured: the true hub contributes 4 colors, sprites reuse 2-4 of
#       them, and the main decoy uses four fresh colors. The relation is readable by eye.
#
# new_main_assumptions:
#   - True sprites live outside every candidate hub bbox.
#   - Among candidate hubs, the true hub has strictly more attachable sprite matches than any
#     distractor hub. The generator enforces this by giving the main distractor fresh colors.
#   - Sprite geometry remains from the original task: 8-connected, monochrome, and uniquely
#     attachable after rotation via symmetry + contact-edge length tests.


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _edge_segment(mask: np.ndarray, side: str) -> tuple[str, int, list[int]] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _hub_candidates(g: np.ndarray, bg: int) -> list[dict[str, Any]]:
    H, W = g.shape
    out: list[dict[str, Any]] = []
    for n in range(3, 7):
        for r in range(0, int(H - n + 1)):
            for c in range(0, int(W - n + 1)):
                if (
                    int(g[int(r), int(c)]) != int(bg)
                    or int(g[int(r), int(c + n - 1)]) != int(bg)
                    or int(g[int(r + n - 1), int(c)]) != int(bg)
                    or int(g[int(r + n - 1), int(c + n - 1)]) != int(bg)
                ):
                    continue
                inner = g[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[int(r), int(c + 1) : int(c + n - 1)]
                bot = g[int(r + n - 1), int(c + 1) : int(c + n - 1)]
                left = g[int(r + 1) : int(r + n - 1), int(c)]
                right = g[int(r + 1) : int(r + n - 1), int(c + n - 1)]
                if (
                    bool((top == int(bg)).any())
                    or bool((bot == int(bg)).any())
                    or bool((left == int(bg)).any())
                    or bool((right == int(bg)).any())
                ):
                    continue
                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                ):
                    continue

                colors = (int(top[0]), int(bot[0]), int(left[0]), int(right[0]))
                cr = float(r + (n - 1) / 2)
                cc = float(c + (n - 1) / 2)
                center_score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                out.append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "n": int(n),
                        "colors": colors,
                        "center_score": float(center_score),
                        "all_distinct": len(set(colors)) == 4,
                    }
                )
    return out


def _all_hub_boxes_mask(shape: tuple[int, int], candidates: list[dict[str, Any]]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    for cand in candidates:
        r = int(cand["r"])
        c = int(cand["c"])
        n = int(cand["n"])
        mask[r : r + n, c : c + n] = True
    return mask


def _side_specs(cand: dict[str, Any]) -> dict[str, tuple[int, tuple[str, int, list[int]]]]:
    r = int(cand["r"])
    c = int(cand["c"])
    n = int(cand["n"])
    top_col, bot_col, left_col, right_col = map(int, cand["colors"])
    return {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }


def _try_attach_component(
    g: np.ndarray,
    bg: int,
    comp_mask_full: np.ndarray,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
    out: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray] | None:
    H, W = g.shape
    spr_arr = g.copy()
    spr_arr[~comp_mask_full] = int(bg)
    spr_arr, spr_mask = _crop_to_mask(spr_arr, comp_mask_full)

    for krot in range(4):
        a = np.rot90(spr_arr, int(krot))
        msk = np.rot90(spr_mask, int(krot))
        a, msk = _crop_to_mask(a, msk)

        if need_edge in ("left", "right"):
            if not bool(np.array_equal(msk, np.flipud(msk))):
                continue
        else:
            if not bool(np.array_equal(msk, np.fliplr(msk))):
                continue

        seg = _edge_segment(msk, need_edge)
        if seg is None:
            continue

        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg
            if len(rows) != int(target) or not _is_contig(rows):
                continue
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg
            if len(cols) != int(target) or not _is_contig(cols):
                continue
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
        if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
            continue

        if out is not None:
            ok = True
            for R, C in placed:
                val = int(a[int(R - dr), int(C - dc)])
                if val == int(bg):
                    continue
                if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != val:
                    ok = False
                    break
            if not ok:
                continue

        return a, msk, int(dr), int(dc), comp_mask_full

    return None


def _find_matching_sprite(
    g: np.ndarray,
    bg: int,
    all_hubs_mask: np.ndarray,
    used_sprite_mask: np.ndarray,
    col: int,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
    out: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray] | None:
    m = (g == int(col)) & (~all_hubs_mask) & (~used_sprite_mask)
    if not bool(m.any()):
        return None

    labc, nc = label_mask(m, connectivity=8)
    comp_ids = sorted(
        range(1, int(nc) + 1),
        key=lambda cid: int((labc == int(cid)).sum()),
        reverse=True,
    )
    for cid in comp_ids:
        comp_mask = labc == int(cid)
        found = _try_attach_component(g, bg, comp_mask, need_edge, fixed, span, target, out)
        if found is not None:
            return found
    return None


def _select_hub(g: np.ndarray, bg: int, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    all_hubs_mask = _all_hub_boxes_mask(g.shape, candidates)
    best_cand = None
    best_score = None

    for cand in candidates:
        r = int(cand["r"])
        c = int(cand["c"])
        n = int(cand["n"])
        target = int(n - 2)
        out = np.full_like(g, int(bg))
        out[r : r + n, c : c + n] = g[r : r + n, c : c + n]
        used_sprite_mask = np.zeros_like(g, dtype=bool)
        matched = 0
        total_pixels = 0
        side_matches: list[str] = []

        for side, (col, (need_edge, fixed, span)) in _side_specs(cand).items():
            found = _find_matching_sprite(
                g,
                bg,
                all_hubs_mask,
                used_sprite_mask,
                int(col),
                need_edge,
                fixed,
                span,
                target,
                out,
            )
            if found is None:
                continue
            a, msk, dr, dc, comp_mask = found
            paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)
            used_sprite_mask |= comp_mask
            matched += 1
            total_pixels += int(comp_mask.sum())
            side_matches.append(side)

        score = (
            int(matched),
            int(cand["all_distinct"]),
            int(total_pixels),
            -float(cand["center_score"]),
        )
        cand["matched_count"] = int(matched)
        cand["matched_sides"] = tuple(side_matches)
        if best_score is None or score > best_score:
            best_score = score
            best_cand = cand

    if best_cand is None:
        raise AssertionError("no hub candidates found")
    return best_cand, all_hubs_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Multiple hollow squares may appear.
    - The true hub is the candidate whose edge colors can actually absorb the scattered
      sprites under the original rotation/symmetry/contact rules.
    - Distractor hubs are ignored and removed from the output.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    candidates = _hub_candidates(g, bg)
    assert candidates
    hub, all_hubs_mask = _select_hub(g, bg, candidates)

    r = int(hub["r"])
    c = int(hub["c"])
    n = int(hub["n"])
    out = np.full((H, W), int(bg), dtype=int)
    out[r : r + n, c : c + n] = g[r : r + n, c : c + n]
    used_sprite_mask = np.zeros_like(g, dtype=bool)
    target = int(n - 2)

    for _side, (col, (need_edge, fixed, span)) in _side_specs(hub).items():
        found = _find_matching_sprite(
            g,
            bg,
            all_hubs_mask,
            used_sprite_mask,
            int(col),
            need_edge,
            fixed,
            span,
            target,
            out,
        )
        if found is None:
            continue
        a, msk, dr, dc, comp_mask = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)
        used_sprite_mask |= comp_mask

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Constructively build a sprite mask in the attached orientation for `side`.
    The mask is bool, 8-connected, has the correct symmetry and contact-edge length,
    and exactly 1 rotation (out of 4) is valid for the intended side.
    """
    e = edge_len
    h = 4
    w = rng.randint(2, 4)
    half_h = (h + 1) // 2

    for _attempt in range(100):
        grid = np.zeros((h, w), dtype=bool)
        for ri in range(half_h):
            mirror_ri = h - 1 - ri
            for ci in range(w):
                val = rng.random() < 0.5
                grid[ri, ci] = val
                grid[mirror_ri, ci] = val

        right_col = grid[:, -1]
        runs = []
        count = 0
        for v in right_col:
            if v:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        if len(runs) != 1 or runs[0] != e:
            continue

        rows_any = grid.any(axis=1)
        cols_any = grid.any(axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        grid = grid[r0 : r1 + 1, c0 : c1 + 1]

        if grid.shape[0] < 2 or grid.shape[1] < 2:
            continue
        if not np.array_equal(grid, np.flipud(grid)):
            continue

        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if int(num) != 1:
            continue
        if np.array_equal(grid, np.fliplr(grid)):
            continue

        px = int(grid.sum())
        if px < 4 or px > 12:
            continue

        valid_for_left = 0
        for k in range(4):
            rot = np.rot90(grid, k)
            ra = rot.any(axis=1)
            ca = rot.any(axis=0)
            rr0, rr1 = np.where(ra)[0][[0, -1]]
            cc0, cc1 = np.where(ca)[0][[0, -1]]
            cr = rot[rr0 : rr1 + 1, cc0 : cc1 + 1]
            if not np.array_equal(cr, np.flipud(cr)):
                continue
            rc = cr[:, -1]
            runs2 = []
            cnt2 = 0
            for v in rc:
                if v:
                    cnt2 += 1
                else:
                    if cnt2 > 0:
                        runs2.append(cnt2)
                    cnt2 = 0
            if cnt2 > 0:
                runs2.append(cnt2)
            if len(runs2) == 1 and runs2[0] == e:
                valid_for_left += 1

        if valid_for_left != 1:
            continue

        if side == "left":
            mask = grid.copy()
        elif side == "right":
            mask = np.fliplr(grid)
        elif side == "top":
            mask = np.rot90(grid, -1)
        else:
            mask = np.rot90(grid, 1)
        return mask

    return None


def _build_hub_patch(n: int, colors: tuple[int, int, int, int], bg: int) -> np.ndarray:
    top_col, bot_col, left_col, right_col = map(int, colors)
    patch = np.full((n, n), int(bg), dtype=int)
    patch[0, 1 : n - 1] = int(top_col)
    patch[n - 1, 1 : n - 1] = int(bot_col)
    patch[1 : n - 1, 0] = int(left_col)
    patch[1 : n - 1, n - 1] = int(right_col)
    return patch


def _center_score_pos(r: int, c: int, n: int, H: int, W: int) -> float:
    cr = float(r + (n - 1) / 2)
    cc = float(c + (n - 1) / 2)
    return abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))


def _list_bbox_positions(
    blocked: np.ndarray,
    h: int,
    w: int,
    *,
    padding: int = 0,
    r_min: int = 0,
    r_max: int | None = None,
    c_min: int = 0,
    c_max: int | None = None,
) -> list[tuple[int, int]]:
    H, W = blocked.shape
    if r_max is None:
        r_max = H - h
    if c_max is None:
        c_max = W - w
    out: list[tuple[int, int]] = []
    for r in range(int(r_min), int(r_max) + 1):
        for c in range(int(c_min), int(c_max) + 1):
            rr0 = max(0, int(r - padding))
            cc0 = max(0, int(c - padding))
            rr1 = min(H, int(r + h + padding))
            cc1 = min(W, int(c + w + padding))
            if not bool(blocked[rr0:rr1, cc0:cc1].any()):
                out.append((int(r), int(c)))
    return out


def _reserve_bbox(canvas: np.ndarray, r: int, c: int, h: int, w: int, mark: int) -> None:
    canvas[int(r) : int(r + h), int(c) : int(c + w)] = int(mark)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    bg = 8
    N = 4
    edge_len = N - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    true_colors = tuple(all_colors[:4])
    spare_colors = all_colors[4:]
    if len(spare_colors) < 4:
        return None

    sides = ["top", "bot", "left", "right"]
    side_colors = dict(zip(sides, true_colors))

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)
    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = max_extent + 2

    blocked = np.zeros((H, W), dtype=bool)

    true_positions = _list_bbox_positions(
        blocked,
        N,
        N,
        padding=0,
        r_min=margin,
        r_max=H - N - margin,
        c_min=margin,
        c_max=W - N - margin,
    )
    if not true_positions:
        return None
    true_positions_sorted = sorted(
        true_positions,
        key=lambda rc: _center_score_pos(rc[0], rc[1], N, H, W),
        reverse=True,
    )
    far_positions = true_positions_sorted[: max(1, len(true_positions_sorted) // 2)]
    hub_r, hub_c = rng.choice(far_positions)
    true_center_score = _center_score_pos(hub_r, hub_c, N, H, W)
    blocked[hub_r : hub_r + N, hub_c : hub_c + N] = True

    decoy_main_colors = tuple(spare_colors[:4])
    decoy_patch_main = _build_hub_patch(N, decoy_main_colors, bg)
    decoy_positions = [
        rc
        for rc in _list_bbox_positions(blocked, N, N, padding=1)
        if _center_score_pos(rc[0], rc[1], N, H, W) < true_center_score
    ]
    if not decoy_positions:
        decoy_positions = sorted(
            _list_bbox_positions(blocked, N, N, padding=0),
            key=lambda rc: _center_score_pos(rc[0], rc[1], N, H, W),
        )
    if not decoy_positions:
        return None
    decoy_positions.sort(key=lambda rc: _center_score_pos(rc[0], rc[1], N, H, W))
    decoy_r, decoy_c = decoy_positions[0]
    decoy_center_score = _center_score_pos(decoy_r, decoy_c, N, H, W)
    blocked[decoy_r : decoy_r + N, decoy_c : decoy_c + N] = True

    extra_decoys: list[tuple[int, int, np.ndarray]] = []
    if rng.random() < 0.6:
        a, b, c = spare_colors[0], spare_colors[1], spare_colors[2]
        repeated_colors = (a, b, a, c)
        extra_patch = _build_hub_patch(N, repeated_colors, bg)
        extra_positions = _list_bbox_positions(blocked, N, N, padding=1)
        if extra_positions:
            extra_r, extra_c = rng.choice(extra_positions)
            blocked[extra_r : extra_r + N, extra_c : extra_c + N] = True
            extra_decoys.append((extra_r, extra_c, extra_patch))

    out = np.full((H, W), bg, dtype=int)
    true_hub_patch = _build_hub_patch(N, true_colors, bg)
    paste(out, true_hub_patch, hub_r, hub_c)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        sh, sw = mask.shape
        color = int(side_colors[side])
        arr = np.where(mask, color, bg).astype(int)

        if side == "top":
            filled = np.where(mask[-1])[0]
            sc = hub_c + 1 - int(filled[0])
            sr = hub_r - sh
        elif side == "bot":
            filled = np.where(mask[0])[0]
            sc = hub_c + 1 - int(filled[0])
            sr = hub_r + N
        elif side == "left":
            filled = np.where(mask[:, -1])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c - sw
        else:
            filled = np.where(mask[:, 0])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c + N

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if mask[rr, cc] and int(out[sr + rr, sc + cc]) != int(bg):
                    return None
        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask, sr, sc)

    inp = np.full((H, W), bg, dtype=int)
    paste(inp, true_hub_patch, hub_r, hub_c)
    paste(inp, decoy_patch_main, decoy_r, decoy_c)
    for extra_r, extra_c, extra_patch in extra_decoys:
        paste(inp, extra_patch, extra_r, extra_c)

    occupancy = np.full((H, W), bg, dtype=int)
    occ_mark = 0 if bg != 0 else 1
    _reserve_bbox(occupancy, decoy_r, decoy_c, N, N, occ_mark)
    _reserve_bbox(occupancy, hub_r, hub_c, N, N, occ_mark)
    for extra_r, extra_c, _extra_patch in extra_decoys:
        _reserve_bbox(occupancy, extra_r, extra_c, N, N, occ_mark)

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, k)
        rotated_mask = np.rot90(mask, k)
        rows_any = rotated_mask.any(axis=1)
        cols_any = rotated_mask.any(axis=0)
        r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
        c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
        cropped_arr = rotated_arr[r0_idx : r1_idx + 1, c0_idx : c1_idx + 1]
        cropped_mask = rotated_mask[r0_idx : r1_idx + 1, c0_idx : c1_idx + 1]
        patch = np.where(cropped_mask, cropped_arr, bg).astype(int)

        try:
            pr, pc = random_free_location(
                occupancy,
                patch,
                transparent=bg,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.3:
        used_colors = set(map(int, true_colors)) | set(map(int, decoy_main_colors))
        for _extra_r, _extra_c, extra_patch in extra_decoys:
            used_colors |= set(int(x) for x in np.unique(extra_patch) if int(x) != int(bg))
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = rng.choice(distractor_colors)
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dist_arr = np.where(dist_mask, dcol, bg).astype(int)
                try:
                    pr, pc = random_free_location(
                        occupancy,
                        dist_arr,
                        transparent=bg,
                        padding=2,
                        padding_connectivity=8,
                    )
                    paste(occupancy, dist_arr, pr, pc, transparent=bg)
                    paste(inp, dist_arr, pr, pc, transparent=bg)
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    true_center_score = _center_score_pos(hub_r, hub_c, N, H, W)
    if not (decoy_center_score < true_center_score):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
