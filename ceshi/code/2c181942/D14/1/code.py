# Variant of 2c181942
# Dimension: D14 — 多步处理顺序 (multi_step)
# Change: 单步处理——所有精灵独立匹配和贴附，顺序无关（不存在精灵间遮挡冲突）... → 链式贴附：先贴附的精灵成为后续贴附的锚点

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D14 variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior / trace
#    - Keep the original hub: a hollow NxN square with bg corners + interior and
#      four solid-colored edges.
#    - The variant changes only what happens to matching sprites after the first
#      attachment. One hub side color may now have 2-3 scattered monochrome
#      sprites of that same color, not just one.
#    - Output is no longer "hub side + one sprite". Instead, for one chosen hub
#      side the matching same-color sprites form a straight 2-link chain growing
#      away from the hub. The first sprite touches the hub. The second sprite
#      touches the exposed outer edge of the first.
#    - Order is visible because each link has an incoming contact-edge length and
#      an outgoing exposed-edge length. Example on a top-side chain:
#        old logic: color 2 component -> rotate -> attach directly to hub top
#        new logic:
#          step 1: require bottom edge length = hub edge length (say 3)
#          step 2: read the placed link's top edge length (say 4)
#          step 3: among remaining color-2 components, choose the one whose
#                  bottom edge length is 4, place it above step-1
#          step 4: continue from its exposed top edge length, until no match
#      The core difference is that the "target side" is no longer fixed by the
#      hub after step 1; it is recursively updated from the previously placed
#      sprite.
#
# b) Complexity calibration
#    - To make chaining observable, the generator always creates at least one
#      color with exactly 2 components. In the original task each hub color had at
#      most one relevant component; here one color intentionally has multiple.
#    - The chain is restricted to a single straight ray from one hub side. This
#      keeps the rule learnable from 2-4 examples and avoids branching or turns.
#    - Incoming / outgoing edge lengths come from a small visible set
#      (1,2,3,4). This is enough to encode order while keeping the space of
#      cases small.
#
# c) Implicit-assumption audit of the original main()
#    - Original assumption: each hub color has at most one relevant sprite, so
#      taking the largest 8-connected component is safe. This is false now;
#      same-color sprites can all be relevant chain links.
#    - Original assumption: each sprite is only judged against a fixed hub side.
#      This is false after the first placement; later links must be judged
#      against the previous link's exposed edge.
#    - Original assumption: mirror symmetry around the contact axis is the main
#      rotation disambiguator. This becomes brittle for chained links because a
#      link needs distinct "incoming" and "outgoing" sides. The new solver
#      therefore identifies a rotation by an ordered pair:
#         (incoming side segment length, opposite outgoing side segment length)
#      The generator guarantees that the intended side has exactly one valid
#      rotation under this predicate.
#
# d) Placement legality
#    - The generator still uses output-first construction. It attaches the chain
#      onto the hub in output space, checking bounds and overlap at every link.
#    - Input sprites are then detached, randomly rotated, and scattered with
#      padding so same-color links stay as separate 8-connected components.
#    - Hub placement reserves margin for the full chain depth, not just one
#      sprite as in the original generator.
#
# e) Visual observability
#    - The new rule is visible directly in the inputs: one hub edge color now
#      appears on multiple separated components. In the outputs those same-color
#      components collapse into a single outward-growing chain.
#    - The ordering signal is geometric and visible: the outer edge width of one
#      placed link matches the inner edge width of the next.
#
# f) Visual cleanliness
#    - Only one hub side is allowed to chain, which prevents dense tangles.
#    - The chain grows in a straight line away from the hub, so the resulting
#      output stays readable rather than meandering around the hub.
#
# g) Color reasoning
#    - No hidden color ordering is used. Colors keep the original role:
#      matching a hub side to its relevant sprite set.
#    - The new ordering information is carried by visible edge lengths, not by
#      color ids.
#
# New solver assumptions introduced by the variant and enforced by generator
#    - For each hub color, at most one straight chain exists.
#    - For every chain step, among remaining same-color components there is at
#      most one component/rotation whose incoming edge matches the current
#      anchor length.
#    - Every link has a contiguous incoming edge on the chain-facing side and a
#      contiguous outgoing edge on the opposite side.


OPPOSITE = {"top": "bot", "bot": "top", "left": "right", "right": "left"}
SIDE_TO_NEED_EDGE = {"top": "bot", "bot": "top", "left": "right", "right": "left"}


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _segment_len(mask: np.ndarray, side: str) -> int | None:
    seg = _edge_segment(mask, side)
    if seg is None:
        return None
    if side in ("left", "right"):
        _typ, _edge_c, rows = seg
        return len(rows) if _is_contig(rows) else None
    _typ, _edge_r, cols = seg
    return len(cols) if _is_contig(cols) else None


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None  # (score, (r,c,n), (top,bot,left,right))
    for n in range(3, 7):
        for r in range(0, int(H - n + 1)):
            for c in range(0, int(W - n + 1)):
                if (
                    int(g[r, c]) != int(bg)
                    or int(g[r, c + n - 1]) != int(bg)
                    or int(g[r + n - 1, c]) != int(bg)
                    or int(g[r + n - 1, c + n - 1]) != int(bg)
                ):
                    continue
                inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[r, c + 1 : c + n - 1]
                bot = g[r + n - 1, c + 1 : c + n - 1]
                left = g[r + 1 : r + n - 1, c]
                right = g[r + 1 : r + n - 1, c + n - 1]

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

                cr = float(r + (n - 1) / 2)
                cc = float(c + (n - 1) / 2)
                score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                cand = (
                    score,
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
    return best


def _side_specs(r: int, c: int, n: int, colors: tuple[int, int, int, int]):
    top_col, bot_col, left_col, right_col = colors
    return {
        "top": {
            "color": int(top_col),
            "need_edge": "bot",
            "fixed": int(r - 1),
            "span": list(range(int(c + 1), int(c + n - 1))),
        },
        "bot": {
            "color": int(bot_col),
            "need_edge": "top",
            "fixed": int(r + n),
            "span": list(range(int(c + 1), int(c + n - 1))),
        },
        "left": {
            "color": int(left_col),
            "need_edge": "right",
            "fixed": int(c - 1),
            "span": list(range(int(r + 1), int(r + n - 1))),
        },
        "right": {
            "color": int(right_col),
            "need_edge": "left",
            "fixed": int(c + n),
            "span": list(range(int(r + 1), int(r + n - 1))),
        },
    }


def _extract_components(g: np.ndarray, hub_mask: np.ndarray, bg: int, color: int):
    m = (g == int(color)) & (~hub_mask)
    labc, nc = label_mask(m, connectivity=8)
    comps = []
    for cid in range(1, int(nc) + 1):
        comp_mask = labc == int(cid)
        if not bool(comp_mask.any()):
            continue
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = int(color)
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        comps.append((arr, comp_mask))
    return comps


def _match_and_place_candidate(
    arr0: np.ndarray,
    mask0: np.ndarray,
    *,
    need_edge: str,
    required_len: int,
    fixed: int,
    span: list[int],
    out: np.ndarray,
    bg: int,
):
    H, W = out.shape
    out_edge = OPPOSITE[need_edge]

    for krot in range(4):
        a = np.rot90(arr0, int(krot))
        msk = np.rot90(mask0, int(krot))
        a, msk = _crop_to_mask(a, msk)

        seg_in = _edge_segment(msk, need_edge)
        seg_out = _edge_segment(msk, out_edge)
        if seg_in is None or seg_out is None:
            continue

        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg_in
            if len(rows) != int(required_len) or not _is_contig(rows):
                continue
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg_in
            if len(cols) != int(required_len) or not _is_contig(cols):
                continue
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
        if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
            continue
        if any(int(out[R, C]) != int(bg) for R, C in placed):
            continue

        out_len = _segment_len(msk, out_edge)
        if out_len is None or int(out_len) <= 0:
            continue

        if out_edge in ("left", "right"):
            _typ2, out_c, out_rows = seg_out
            next_span = [int(x + dr) for x in out_rows]
            global_out_c = int(out_c + dc)
            next_fixed = int(global_out_c - 1) if need_edge == "right" else int(global_out_c + 1)
        else:
            _typ2, out_r, out_cols = seg_out
            next_span = [int(x + dc) for x in out_cols]
            global_out_r = int(out_r + dr)
            next_fixed = int(global_out_r - 1) if need_edge == "bot" else int(global_out_r + 1)

        return {
            "arr": a,
            "mask": msk,
            "dr": int(dr),
            "dc": int(dc),
            "krot": int(krot),
            "out_len": int(out_len),
            "next_fixed": int(next_fixed),
            "next_span": next_span,
            "pixels": int(msk.sum()),
        }

    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), colors = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    out = np.full_like(g, int(bg))
    out[r : r + n, c : c + n] = g[r : r + n, c : c + n]

    target = int(n - 2)
    for _side, info in _side_specs(r, c, n, colors).items():
        comps = _extract_components(g, hub_mask, bg, info["color"])
        need_edge = str(info["need_edge"])
        required_len = int(target)
        fixed = int(info["fixed"])
        span = list(info["span"])

        while comps:
            matches = []
            for idx, (arr0, mask0) in enumerate(comps):
                cand = _match_and_place_candidate(
                    arr0,
                    mask0,
                    need_edge=need_edge,
                    required_len=required_len,
                    fixed=fixed,
                    span=span,
                    out=out,
                    bg=bg,
                )
                if cand is not None:
                    matches.append((idx, cand))

            if not matches:
                break

            matches.sort(key=lambda item: (item[1]["krot"], -item[1]["pixels"], item[0]))
            idx, chosen = matches[0]
            paste(out, chosen["arr"], chosen["dr"], chosen["dc"], mask=chosen["mask"], overwrite=True, clip=False)
            fixed = int(chosen["next_fixed"])
            span = list(chosen["next_span"])
            required_len = int(chosen["out_len"])
            comps.pop(idx)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _rot_to_side(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def _count_valid_rotations(mask: np.ndarray, side: str, in_len: int, out_len: int) -> int:
    need_edge = SIDE_TO_NEED_EDGE[side]
    out_edge = OPPOSITE[need_edge]
    cnt = 0
    for k in range(4):
        rot = np.rot90(mask, k)
        rot, rot_mask = _crop_to_mask(rot.astype(int), rot)
        rot_mask = rot_mask.astype(bool)
        if _segment_len(rot_mask, need_edge) == int(in_len) and _segment_len(rot_mask, out_edge) == int(out_len):
            cnt += 1
    return int(cnt)


def _build_horizontal_link_mask(in_len: int, out_len: int, rng: random.Random) -> np.ndarray | None:
    for _attempt in range(400):
        w = rng.randint(3, 5)
        h = max(int(in_len), int(out_len)) + rng.randint(0, 3)
        if h < max(int(in_len), int(out_len)):
            continue

        left_top = rng.randint(0, h - int(out_len))
        right_top = rng.randint(0, h - int(in_len))
        left_bot = left_top + int(out_len) - 1
        right_bot = right_top + int(in_len) - 1

        grid = np.zeros((h, w), dtype=bool)
        for x in range(w):
            top = int(round((left_top * (w - 1 - x) + right_top * x) / max(1, w - 1)))
            bot = int(round((left_bot * (w - 1 - x) + right_bot * x) / max(1, w - 1)))
            if top > bot:
                top, bot = bot, top
            grid[top : bot + 1, x] = True

        if w >= 4 and rng.random() < 0.7:
            x = rng.randint(1, w - 2)
            filled = np.where(grid[:, x])[0]
            if len(filled) > 0:
                if rng.random() < 0.5 and int(filled[0]) > 0:
                    grid[int(filled[0]) - 1, x] = True
                elif int(filled[-1]) < h - 1:
                    grid[int(filled[-1]) + 1, x] = True

        grid, grid_mask = _crop_to_mask(grid.astype(int), grid)
        grid = grid_mask.astype(bool)

        if _segment_len(grid, "right") != int(in_len):
            continue
        if _segment_len(grid, "left") != int(out_len):
            continue

        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if int(num) != 1:
            continue

        px = int(grid.sum())
        if px < 4 or px > 16:
            continue

        return grid

    return None


def _make_link_mask(in_len: int, out_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    for _attempt in range(500):
        base = _build_horizontal_link_mask(int(in_len), int(out_len), rng)
        if base is None:
            continue
        mask = _rot_to_side(base, side)
        mask, mask_crop = _crop_to_mask(mask.astype(int), mask)
        mask = mask_crop.astype(bool)
        if _count_valid_rotations(mask, side, int(in_len), int(out_len)) != 1:
            continue
        return mask
    return None


def _choose_chain_lengths(start_len: int, rng: random.Random) -> list[int]:
    templates = {
        2: [2, 4, 1],
        3: [3, 2, 1],
        4: [4, 2, 1],
    }
    return list(templates[int(start_len)])


def _place_attached_link(
    out: np.ndarray,
    *,
    bg: int,
    color: int,
    mask: np.ndarray,
    need_edge: str,
    fixed: int,
    span: list[int],
):
    arr = np.where(mask, int(color), int(bg)).astype(int)
    seg_in = _edge_segment(mask, need_edge)
    assert seg_in is not None

    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg_in
        sr = int(span[0] - rows[0])
        sc = int(fixed - edge_c)
    else:
        _typ, edge_r, cols = seg_in
        sr = int(fixed - edge_r)
        sc = int(span[0] - cols[0])

    h, w = mask.shape
    H, W = out.shape
    if sr < 0 or sc < 0 or sr + h > H or sc + w > W:
        return None

    for rr in range(h):
        for cc in range(w):
            if bool(mask[rr, cc]) and int(out[sr + rr, sc + cc]) != int(bg):
                return None

    paste(out, arr, sr, sc, transparent=bg)

    out_edge = OPPOSITE[need_edge]
    seg_out = _edge_segment(mask, out_edge)
    assert seg_out is not None
    out_len = _segment_len(mask, out_edge)
    assert out_len is not None

    if out_edge in ("left", "right"):
        _typ2, out_c, out_rows = seg_out
        next_span = [int(sr + x) for x in out_rows]
        global_out_c = int(sc + out_c)
        next_fixed = int(global_out_c - 1) if need_edge == "right" else int(global_out_c + 1)
    else:
        _typ2, out_r, out_cols = seg_out
        next_span = [int(sc + x) for x in out_cols]
        global_out_r = int(sr + out_r)
        next_fixed = int(global_out_r - 1) if need_edge == "bot" else int(global_out_r + 1)

    return {
        "arr": arr,
        "mask": mask,
        "sr": int(sr),
        "sc": int(sc),
        "next_fixed": int(next_fixed),
        "next_span": next_span,
        "out_len": int(out_len),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    N = rng.choice([4, 5, 6])
    edge_len = N - 2

    H = rng.randint(22, 26)
    W = rng.randint(22, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    chain_side = rng.choice(["top", "bot", "left", "right"])
    chain_color = int(side_colors[chain_side])
    chain_lengths = _choose_chain_lengths(int(edge_len), rng)
    link_pairs = [(int(chain_lengths[i]), int(chain_lengths[i + 1])) for i in range(len(chain_lengths) - 1)]

    masks: list[np.ndarray] = []
    for in_len, out_len in link_pairs:
        mask = _make_link_mask(int(in_len), int(out_len), chain_side, rng)
        if mask is None:
            return None
        masks.append(mask)

    ext_top = ext_bot = ext_left = ext_right = 0
    axial = sum(int(m.shape[0] if chain_side in ("top", "bot") else m.shape[1]) for m in masks) + 1
    lateral = max(max(m.shape) for m in masks) + 2
    if chain_side == "top":
        ext_top = axial
    elif chain_side == "bot":
        ext_bot = axial
    elif chain_side == "left":
        ext_left = axial
    else:
        ext_right = axial

    hub_r_min = lateral + ext_top
    hub_r_max = H - N - lateral - ext_bot
    hub_c_min = lateral + ext_left
    hub_c_max = W - N - lateral - ext_right
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    for ci in range(1, N - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + N - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, N - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + N - 1] = int(right_col)

    attached = []
    specs = _side_specs(hub_r, hub_c, N, (top_col, bot_col, left_col, right_col))
    info = specs[chain_side]
    need_edge = str(info["need_edge"])
    fixed = int(info["fixed"])
    span = list(info["span"])
    required_len = int(edge_len)

    for mask in masks:
        if _segment_len(mask, need_edge) != int(required_len):
            return None
        placed = _place_attached_link(
            out,
            bg=bg,
            color=chain_color,
            mask=mask,
            need_edge=need_edge,
            fixed=fixed,
            span=span,
        )
        if placed is None:
            return None
        attached.append(placed)
        fixed = int(placed["next_fixed"])
        span = list(placed["next_span"])
        required_len = int(placed["out_len"])

    inp = np.full((H, W), bg, dtype=int)
    inp[hub_r : hub_r + N, hub_c : hub_c + N] = out[hub_r : hub_r + N, hub_c : hub_c + N]
    canvas = inp.copy()

    for placed in attached:
        arr = placed["arr"]
        mask = placed["mask"]
        k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, k)
        rot_mask = np.rot90(mask, k)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        patch = np.where(rot_mask, rot_arr, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.2:
        free_colors = [c for c in range(10) if c != bg and c not in set(side_colors.values())]
        if free_colors:
            dcol = rng.choice(free_colors)
            dmask = _make_link_mask(rng.choice([2, 3]), 1, rng.choice(["top", "bot", "left", "right"]), rng)
            if dmask is not None:
                darr = np.where(dmask, int(dcol), int(bg)).astype(int)
                try:
                    pr, pc = random_free_location(canvas, darr, transparent=bg, padding=2, padding_connectivity=8)
                    paste(canvas, darr, pr, pc, transparent=bg)
                    paste(inp, darr, pr, pc, transparent=bg)
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
