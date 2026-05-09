# Combo variant of 2c181942 (index 13)
# combo_spec: D6/1+D10/0
# Name: 同色干扰+多hub竞争

from __future__ import annotations

import random
import sys
from typing import Any

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior trace
#    - Base task: one hollow 4x4 square sits near the center; its four sides have four
#      distinct colors. For each shown side color there is one detached monochrome sprite.
#      The old solver chooses the hub closest to center, then for each hub color takes the
#      largest same-color component and tries rotations until one satisfies the original
#      symmetry + contact-edge constraints.
#    - Combo input now contains:
#        * one true hub: hollow 4x4, four pairwise-distinct side colors
#        * one decoy hub: also hollow 4x4, placed closer to the center, but with repeated
#          side colors so it is not a true hub
#        * 2-4 real sprites matching a subset of the true hub colors
#        * one same-color distractor: it reuses one real sprite color but its geometry is
#          invalid for that side in all four rotations
#    - Small trace:
#        * old hub selection would pick the closer repeated-color decoy
#        * old same-color selection would also be unsafe because the distractor is larger
#        * combo solver must first enumerate hub candidates, prefer the all-distinct one,
#          and then, for each side color, test every same-color component geometrically
#        * only the unique valid component is rotated and attached; decoy hub and invalid
#          same-color distractor both disappear in the output
#    - Difference from the original is visible in two separate places at once: multiple
#      hollow squares compete for "hub" status, and one sprite color appears on two pieces
#      but only one has a legal attaching edge.
#
# b) Complexity calibration
#    - The freedom remains controlled:
#        * hub size stays fixed at 4x4
#        * real sprite count stays 2-4
#        * exactly one same-color distractor is added
#        * the main decoy hub always exists and is closer to center than the real hub
#    - This is enough to visibly change the puzzle without making the inference diffuse.
#      The learner sees repeated support: "true hub = the only all-distinct square" and
#      "same color alone is insufficient; geometry still matters."
#
# c) Implicit assumption audit of the original main()
#    - Original assumption "closest hollow square is the hub" breaks. The combo solver now
#      scores all hollow-square candidates and prefers all-distinct hubs, then attachment
#      evidence, then the old center heuristic only as a late tie-breaker.
#    - Original assumption "largest same-color component is the sprite" breaks. The combo
#      solver now labels every same-color component, checks each one for exactly one legal
#      rotation, and only then chooses a component to place.
#    - Original search mask `(g == color) & (~hub_mask)` also breaks with multiple hubs,
#      because other hub edges would be mistaken for sprites. The combo solver excludes the
#      union of all hub candidate boxes from sprite search.
#    - Tracking used components becomes important even though the true hub has unique side
#      colors, because decoy hubs can repeat colors and we must not count the same sprite
#      twice while scoring hub candidates.
#
# d) Placement legality
#    - Generator stays output-first. It builds the true output, then scatters rotated copies
#      of the real sprites, then places the decoy hub and the same-color distractor.
#    - Full hub bounding boxes are reserved during scattering so no sprite can land inside a
#      hollow square interior. `random_free_location(..., padding=2, padding_connectivity=8)`
#      keeps all detached components visually separated and prevents component merges.
#    - The generator rejects any sample where the closest hub is accidentally the true one,
#      or where the target same-color pool does not consist of exactly two components with
#      exactly one geometrically valid attachment.
#
# e) Visual observability
#    - A human can infer the rule from examples because both changes are directly visible:
#        * there is an extra hollow square that vanishes in the output
#        * there are two detached pieces of one color but only one gets used
#    - No hidden ordering by raw color id or by object creation order is used. All decisions
#      come from visible structure: hub-side color diversity plus sprite geometry.
#
# f) Visual tidiness
#    - Objects remain monochrome, isolated, and non-overlapping.
#    - Only one same-color conflict is introduced, so the scene stays readable instead of
#      becoming a dense field of ambiguous copies.
#    - The decoy hub is a clean hollow square rather than a noisy annotation, so the grid
#      still looks ARC-like and sparse.
#
# g) Change inferability
#    - Color semantics remain structured. The true hub contributes four side colors, the
#      real sprites reuse 2-4 of them, and exactly one of those colors appears on two
#      detached components. This gives the learner repeated evidence that color narrows the
#      candidate set while shape decides the final match.
#
# new_main_assumptions guaranteed by generate_input():
#   - There is at least one hollow-square candidate and at least two in normal cases.
#   - Exactly one candidate hub has four distinct side colors.
#   - For one true side color there are exactly two detached same-color components outside
#     all hub boxes, and exactly one of them has a unique legal attachment for that side.
#   - The closest-to-center hub candidate is not the true hub.


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.where(mask, 1, 0).astype(int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


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


def _need_edge_for_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _valid_rotations_for_side(mask: np.ndarray, need_edge: str, target: int) -> list[tuple[np.ndarray, tuple[str, int, list[int]]]]:
    valid: list[tuple[np.ndarray, tuple[str, int, list[int]]]] = []
    for krot in range(4):
        rot = _crop_mask(np.rot90(mask, int(krot)))
        if need_edge in ("left", "right"):
            if not bool(np.array_equal(rot, np.flipud(rot))):
                continue
        else:
            if not bool(np.array_equal(rot, np.fliplr(rot))):
                continue
        seg = _edge_segment(rot, need_edge)
        if seg is None:
            continue
        if need_edge in ("left", "right"):
            _typ, _edge_c, rows = seg
            if len(rows) != int(target) or not _is_contig(rows):
                continue
        else:
            _typ, _edge_r, cols = seg
            if len(cols) != int(target) or not _is_contig(cols):
                continue
        valid.append((rot, seg))
    return valid


def _aligned_translation(mask: np.ndarray, need_edge: str, fixed: int, span: list[int], target: int) -> tuple[int, int] | None:
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None
    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if len(rows) != int(target) or not _is_contig(rows):
            return None
        return int(span[0] - rows[0]), int(fixed - edge_c)
    _typ, edge_r, cols = seg
    if len(cols) != int(target) or not _is_contig(cols):
        return None
    return int(fixed - edge_r), int(span[0] - cols[0])


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
                out.append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "n": int(n),
                        "colors": colors,
                        "center_score": abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2)),
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


def _component_attachment_candidate(
    g: np.ndarray,
    bg: int,
    comp_mask_full: np.ndarray,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
    out: np.ndarray | None,
) -> tuple[int, np.ndarray, np.ndarray, int, int, np.ndarray] | None:
    H, W = g.shape
    comp_arr = np.where(comp_mask_full, g, int(bg)).astype(int)
    comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask_full)

    valid: list[tuple[int, np.ndarray, np.ndarray, int, int, np.ndarray]] = []
    for krot in range(4):
        a = np.rot90(comp_arr, int(krot))
        msk = np.rot90(comp_mask, int(krot))
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

        valid.append((int(msk.sum()), a, msk, int(dr), int(dc), comp_mask_full))

    if len(valid) != 1:
        return None
    return valid[0]


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
) -> tuple[int, np.ndarray, np.ndarray, int, int, np.ndarray] | None:
    m = (g == int(col)) & (~all_hubs_mask) & (~used_sprite_mask)
    if not bool(m.any()):
        return None

    labc, nc = label_mask(m, connectivity=8)
    candidates: list[tuple[int, np.ndarray, np.ndarray, int, int, np.ndarray]] = []
    for cid in range(1, int(nc) + 1):
        comp_mask = labc == int(cid)
        found = _component_attachment_candidate(g, bg, comp_mask, need_edge, fixed, span, target, out)
        if found is not None:
            candidates.append(found)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0]


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

        for _side, (col, (need_edge, fixed, span)) in _side_specs(cand).items():
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
            area, a, msk, dr, dc, comp_mask = found
            paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)
            used_sprite_mask |= comp_mask
            matched += 1
            total_pixels += int(area)

        score = (
            int(cand["all_distinct"]),
            int(matched),
            int(total_pixels),
            -float(cand["center_score"]),
        )
        cand["matched_count"] = int(matched)
        if best_score is None or score > best_score:
            best_score = score
            best_cand = cand

    if best_cand is None:
        raise AssertionError("no hub candidate selected")
    return best_cand, all_hubs_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - multiple hollow squares may appear
    - the true hub is the all-distinct candidate that can absorb the detached sprites
    - one sprite color may have a same-color distractor, so geometry must be checked per
      component rather than choosing the largest component blindly
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
        _area, a, msk, dr, dc, comp_mask = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)
        used_sprite_mask |= comp_mask

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Build a true sprite from a small canonical family.

    Canonical frame:
    - height 4
    - contact edge on the right
    - flipud symmetric
    - right edge has exactly `edge_len` contiguous cells
    - not fliplr symmetric
    - exactly one valid rotation for left-side attachment
    """
    if int(edge_len) != 2:
        return None

    left_attach_caps = [
        np.array([1, 0], dtype=bool),
        np.array([1, 0, 0], dtype=bool),
        np.array([1, 1, 0], dtype=bool),
        np.array([1, 0, 0, 0], dtype=bool),
    ]
    rng.shuffle(left_attach_caps)

    for cap in left_attach_caps:
        w = int(cap.size)
        grid = np.zeros((4, w), dtype=bool)
        grid[0] = cap
        grid[1] = True
        grid[2] = True
        grid[3] = cap
        grid = _crop_mask(grid)
        if int(grid.sum()) > 10:
            continue
        if len(_valid_rotations_for_side(grid, "right", edge_len)) != 1:
            continue

        if side == "left":
            return grid.copy()
        if side == "right":
            return np.fliplr(grid)
        if side == "top":
            return np.rot90(grid, -1)
        return np.rot90(grid, 1)

    return None


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    sr = rng.randrange(h)
    sc = rng.randrange(w)
    mask[sr, sc] = True

    while int(mask.sum()) < int(area):
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = rng.choice(list(frontier))
        mask[int(nr), int(nc)] = True

    return _crop_mask(mask)


def _make_same_color_distractor_mask(edge_len: int, side: str, min_area: int, rng: random.Random) -> np.ndarray | None:
    need_edge = _need_edge_for_side(side)
    lo = max(int(min_area), 11)
    hi = 12
    if lo > hi:
        lo = hi

    for _ in range(250):
        h = rng.randint(4, 5)
        w = rng.randint(4, 5)
        area = rng.randint(lo, hi)
        if area >= h * w:
            area = h * w - 1
        mask = _grow_connected_mask(h, w, area, rng)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < int(lo):
            continue
        _lab, ncomp = label_mask(mask, connectivity=8)
        if int(ncomp) != 1:
            continue
        if _valid_rotations_for_side(mask, need_edge, edge_len):
            continue
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


def _attach_mask_to_hub(
    out: np.ndarray,
    hub: dict[str, Any],
    side: str,
    mask: np.ndarray,
    color: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    arr = np.where(mask, int(color), int(bg)).astype(int)
    target = int(hub["n"]) - 2
    _col, (need_edge, fixed, span) = _side_specs(hub)[side]
    shift = _aligned_translation(mask, need_edge, fixed, span, target)
    if shift is None:
        return None
    sr, sc = shift
    sh, sw = mask.shape
    H, W = out.shape
    if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
        return None
    for rr, cc in np.argwhere(mask):
        if int(out[int(sr + rr), int(sc + cc)]) != int(bg):
            return None
    paste(out, arr, int(sr), int(sc), transparent=int(bg))
    return arr, mask, int(sr), int(sc)


def _count_valid_components_for_side(
    g: np.ndarray,
    all_hubs_mask: np.ndarray,
    color: int,
    side: str,
    target: int,
) -> tuple[int, int]:
    m = (g == int(color)) & (~all_hubs_mask)
    if not bool(m.any()):
        return 0, 0
    labc, nc = label_mask(m, connectivity=8)
    valid = 0
    need_edge = _need_edge_for_side(side)
    for cid in range(1, int(nc) + 1):
        comp_mask = labc == int(cid)
        if len(_valid_rotations_for_side(_crop_mask(comp_mask), need_edge, target)) == 1:
            valid += 1
    return int(nc), int(valid)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2
    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    true_colors = tuple(all_colors[:4])
    spare_colors = all_colors[4:]
    if len(spare_colors) < 3:
        return None

    sides = ["top", "bot", "left", "right"]
    side_colors = dict(zip(sides, true_colors))
    sprite_sides = rng.sample(sides, rng.randint(2, 4))

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_true_extent = max(max(mask.shape) for mask in sprite_masks.values())
    margin = max(max_true_extent, 5) + 2

    blocked = np.zeros((H, W), dtype=bool)
    true_positions = _list_bbox_positions(
        blocked,
        n,
        n,
        r_min=margin,
        r_max=H - n - margin,
        c_min=margin,
        c_max=W - n - margin,
    )
    if not true_positions:
        return None

    far_positions = sorted(
        true_positions,
        key=lambda rc: _center_score_pos(rc[0], rc[1], n, H, W),
        reverse=True,
    )
    far_positions = far_positions[: max(1, len(far_positions) // 2)]
    hub_r, hub_c = rng.choice(far_positions)
    true_center_score = _center_score_pos(hub_r, hub_c, n, H, W)
    blocked[hub_r : hub_r + n, hub_c : hub_c + n] = True

    decoy_colors = (int(spare_colors[0]), int(spare_colors[1]), int(spare_colors[0]), int(spare_colors[2]))
    decoy_positions = [
        rc
        for rc in _list_bbox_positions(blocked, n, n, padding=1)
        if _center_score_pos(rc[0], rc[1], n, H, W) < true_center_score
    ]
    if not decoy_positions:
        return None
    decoy_positions.sort(key=lambda rc: _center_score_pos(rc[0], rc[1], n, H, W))
    decoy_r, decoy_c = decoy_positions[0]
    decoy_center_score = _center_score_pos(decoy_r, decoy_c, n, H, W)
    blocked[decoy_r : decoy_r + n, decoy_c : decoy_c + n] = True

    true_hub = {"r": int(hub_r), "c": int(hub_c), "n": int(n), "colors": tuple(map(int, true_colors))}
    true_patch = _build_hub_patch(n, true_colors, bg)
    decoy_patch = _build_hub_patch(n, decoy_colors, bg)

    out = np.full((H, W), int(bg), dtype=int)
    paste(out, true_patch, int(hub_r), int(hub_c))

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        attached = _attach_mask_to_hub(out, true_hub, side, sprite_masks[side], int(side_colors[side]), int(bg))
        if attached is None:
            return None
        attached_sprites[side] = attached

    inp = np.full((H, W), int(bg), dtype=int)
    paste(inp, true_patch, int(hub_r), int(hub_c))
    paste(inp, decoy_patch, int(decoy_r), int(decoy_c))

    occupancy = np.full((H, W), int(bg), dtype=int)
    occ_mark = 0 if bg != 0 else 1
    _reserve_bbox(occupancy, int(hub_r), int(hub_c), int(n), int(n), int(occ_mark))
    _reserve_bbox(occupancy, int(decoy_r), int(decoy_c), int(n), int(n), int(occ_mark))

    for side in sprite_sides:
        _arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_mask = _crop_mask(np.rot90(mask, int(k)))
        patch = np.where(rotated_mask, int(side_colors[side]), int(bg)).astype(int)
        try:
            pr, pc = random_free_location(
                occupancy,
                patch,
                transparent=int(bg),
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    distractor_side = rng.choice(sprite_sides)
    real_area = int(sprite_masks[distractor_side].sum())
    dist_mask = _make_same_color_distractor_mask(edge_len, distractor_side, real_area + 1, rng)
    if dist_mask is None:
        return None
    dist_mask = _crop_mask(np.rot90(dist_mask, rng.choice([0, 1, 2, 3])))
    dist_arr = np.where(dist_mask, int(side_colors[distractor_side]), int(bg)).astype(int)
    try:
        pr, pc = random_free_location(
            occupancy,
            dist_arr,
            transparent=int(bg),
            padding=2,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(occupancy, dist_arr, int(pr), int(pc), transparent=int(bg))
    paste(inp, dist_arr, int(pr), int(pc), transparent=int(bg))

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    candidates = _hub_candidates(inp, bg)
    if len(candidates) < 2:
        return None
    if sum(1 for cand in candidates if bool(cand["all_distinct"])) != 1:
        return None
    closest = min(candidates, key=lambda cand: float(cand["center_score"]))
    if (int(closest["r"]), int(closest["c"]), int(closest["n"])) == (int(hub_r), int(hub_c), int(n)):
        return None

    all_hubs_mask = _all_hub_boxes_mask(inp.shape, candidates)
    comp_count, valid_count = _count_valid_components_for_side(
        inp,
        all_hubs_mask,
        int(side_colors[distractor_side]),
        distractor_side,
        edge_len,
    )
    if comp_count != 2 or valid_count != 1:
        return None

    meta = {
        "hub_count": int(len(candidates)),
        "sprite_count": int(len(sprite_sides)),
        "distractor_side": str(distractor_side),
        "same_color_component_count": int(comp_count),
        "valid_component_count": int(valid_count),
        "true_center_score": float(true_center_score),
        "closest_decoy_center_score": float(decoy_center_score),
        "closest_is_true": False,
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            inp, _out, _meta = result
            return inp
    raise RuntimeError("generate_input: failed after 500 attempts")
