# Combo variant of 446ef5d2 (index 17)
# combo_spec: D7/3+D15/2
# Name: 填充率编码顺序配合链式拼接

from __future__ import annotations

"""
Design analysis for the fused variant
=====================================

Source understanding
--------------------
- Baseline 446ef5d2:
  detect the 3-cell L marker, infer which rectangle corner it names, extract all
  scattered rectangular pieces, and backtrack over tilings using wildcard edge
  matching (`rect_col` matches anything on a seam).
- This combo removes the old "search until one works" multi-step assumption.
  Assembly must now be deterministic:
  1. compute each piece's fill ratio = non-`rect_col` pixels / piece area,
  2. sort pieces from low fill to high fill,
  3. place them greedily in that order with no backtracking.

a) Concrete behavior trace
--------------------------
- Old behavior:
  pieces could be tried in area order and a recursive search would backtrack if
  an early placement was wrong.
- New fused behavior:
  fill order is the only legal order, and every next piece extends the current
  partial rectangle in exactly one way.
- Small concrete trace in canonical TL orientation:
  piece 1: 2x3, fill 1/6 -> placed at top-left corner.
  piece 2: 2x2, fill 2/4 -> same height as current rectangle, so it can only
           extend to the right, yielding a 2x5 rectangle.
  piece 3: 3x5, fill 4/15 -> same width as current rectangle, so it can only
           extend downward, yielding a 5x5 rectangle.
  piece 4: 2x5, fill 6/10 -> only downward again, yielding the full 7x5 rectangle.
  Difference from baseline: the solver never asks "which unused piece fits this
  cell after trying alternatives"; it only asks "does the next higher-fill piece
  extend the current rectangle rightward or downward, and is exactly one option valid?"

b) Complexity calibration
-------------------------
- If fill order existed but geometry still allowed many placements, the rule would
  be invisible and the task would collapse back to ordinary tiling.
- The generator therefore constructs pieces by rectangle growth from one corner.
  After sorting by fill, each next piece must match either the current width or
  the current height, and generated samples guarantee that exactly one of those
  two growth moves works.
- Fill ratios are strictly increasing with visible gaps, not tiny floating-point
  accidents. Wrong greedy orders (notably the original area-descending order) are
  rejected during generation if they also succeed.

c) Implicit assumption audit against the original main()
-------------------------------------------------------
- `marker -> corner` still holds. The L marker is unchanged.
- `extract rectangles from scattered occupancy` still holds because pieces remain
  solid axis-aligned rectangles with >=1 background cell of padding.
- `rect_col is the most common piece color` still holds, but the old search logic
  is invalid: the original `next_empty + backtracking` assumes order can be chosen
  dynamically. Here order is externally encoded by fill ratio and must be obeyed.
- The old solver could accept any tiling consistent with seam compat. That premise
  is intentionally broken. The new solver assumes that, after canonicalizing the
  corner to TL, the sorted pieces form a monotone rectangle-growth chain.
- New assumption introduced here:
  every fill ratio is distinct, and in fill order every step has exactly one legal
  growth move. The generator enforces both conditions.

d) Placement legality
---------------------
- Generator is still output-first: it first builds the assembled rectangle, then
  scatters its rectangular pieces onto background cells with padding.
- Marker cells are checked to stay in bounds and never overlap scattered pieces.
- All scattered pieces are placed by `random_free_location(..., padding=1)`, so
  rectangles do not overlap or touch.

e) Visual observability / cleanliness / inferability
----------------------------------------------------
- Humans can see that the pieces are rectangular, that colored content density is
  clearly increasing from one piece to the next, and that the output assembles the
  rectangle by successively larger "growth" pieces rather than by arbitrary search.
- No hidden numeric color ordering is used. Only visible fill density drives order.
- Interior colors are limited to 2 repeated colors so the grid stays readable.

Post-change solver assumptions, and generator guarantees
--------------------------------------------------------
- Assumption: fill ratios are strictly increasing. Guaranteed by discrete count
  search with a minimum visible gap.
- Assumption: after canonical corner-flipping to TL, the pieces in fill order form
  one monotone growth chain. Guaranteed by construction.
- Assumption: the wrong greedy order is genuinely wrong. Guaranteed by rejecting
  samples where area-descending greedy assembly also succeeds.
"""

from collections import Counter
import math
import random

import numpy as np

from arc_utils import paste, random_free_location


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, grid.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) != 1:
            continue
        missing = (int(miss[0][0]), int(miss[0][1]))
        if missing == (r0, c0):
            corner = "BR"
        elif missing == (r0, c1):
            corner = "BL"
        elif missing == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), missing, corner
    raise AssertionError("marker not found")


def _extract_rectangles(grid: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    occ = (grid != int(bg)) & (grid != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    gh, gw = grid.shape
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < gw and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < gh and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, grid[rr : rr + h, cc : cc + w].copy()))
    return rects


def _top_left_from_corner(corner: str, missing: tuple[int, int], H: int, W: int) -> tuple[int, int]:
    mr, mc = map(int, missing)
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _candidate_dims(total_area: int, missing: tuple[int, int], corner: str, gh: int, gw: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for hh in range(1, int(math.isqrt(total_area)) + 1):
        if total_area % hh != 0:
            continue
        for H, W in ((hh, total_area // hh), (total_area // hh, hh)):
            tr, tc = _top_left_from_corner(corner, missing, int(H), int(W))
            if 0 <= tr and 0 <= tc and tr + int(H) <= gh and tc + int(W) <= gw:
                dims.append((int(H), int(W)))
    return sorted(set(dims), key=lambda t: (abs(t[0] - t[1]), -t[0], -t[1]))


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.asarray(v1, dtype=int)
    b = np.asarray(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _flip_piece(piece: np.ndarray, corner: str) -> np.ndarray:
    out = np.array(piece, dtype=int)
    if corner in ("TR", "BR"):
        out = np.fliplr(out)
    if corner in ("BL", "BR"):
        out = np.flipud(out)
    return out


def _unflip_rect(rect: np.ndarray, corner: str) -> np.ndarray:
    out = np.array(rect, dtype=int)
    if corner in ("BL", "BR"):
        out = np.flipud(out)
    if corner in ("TR", "BR"):
        out = np.fliplr(out)
    return out


def _piece_stats(piece: np.ndarray, rect_col: int) -> tuple[int, int, float]:
    area = int(piece.shape[0] * piece.shape[1])
    fill = int(np.sum(piece != int(rect_col)))
    return fill, area, fill / area


def _fill_order(pieces: list[np.ndarray], rect_col: int) -> list[int]:
    return sorted(
        range(len(pieces)),
        key=lambda i: (_piece_stats(pieces[i], rect_col)[2], _piece_stats(pieces[i], rect_col)[0], i),
    )


def _strictly_increasing_fill(order: list[int], pieces: list[np.ndarray], rect_col: int) -> bool:
    prev_num = -1
    prev_den = 1
    for i in order:
        num, den, _ = _piece_stats(pieces[i], rect_col)
        if prev_num >= 0 and num * prev_den <= prev_num * den:
            return False
        prev_num, prev_den = num, den
    return True


def _growth_assemble_tl(
    pieces: list[np.ndarray],
    order: list[int],
    H: int,
    W: int,
    rect_col: int,
) -> np.ndarray | None:
    if not order:
        return None

    occ = np.full((H, W), -1, dtype=int)
    first = np.array(pieces[order[0]], dtype=int)
    fh, fw = map(int, first.shape)
    if fh > H or fw > W:
        return None
    if not np.all(first[0, :] == int(rect_col)):
        return None
    if not np.all(first[:, 0] == int(rect_col)):
        return None
    occ[:fh, :fw] = first
    cur_h, cur_w = fh, fw

    for idx in order[1:]:
        arr = np.array(pieces[idx], dtype=int)
        h, w = map(int, arr.shape)
        options: list[tuple[str, int, int]] = []

        if h == cur_h and cur_w + w <= W:
            if np.all(arr[0, :] == int(rect_col)) and _compat(occ[:cur_h, cur_w - 1], arr[:, 0], rect_col):
                options.append(("right", 0, cur_w))

        if w == cur_w and cur_h + h <= H:
            if np.all(arr[:, 0] == int(rect_col)) and _compat(occ[cur_h - 1, :cur_w], arr[0, :], rect_col):
                options.append(("down", cur_h, 0))

        if len(options) != 1:
            return None

        mode, r0, c0 = options[0]
        occ[r0 : r0 + h, c0 : c0 + w] = arr
        if mode == "right":
            cur_w += w
        else:
            cur_h += h

    if cur_h != H or cur_w != W:
        return None
    if not np.all(occ[:, -1] == int(rect_col)):
        return None
    if not np.all(occ[-1, :] == int(rect_col)):
        return None
    return occ


def _area_desc_order(pieces: list[np.ndarray]) -> list[int]:
    return sorted(range(len(pieces)), key=lambda i: (-(pieces[i].shape[0] * pieces[i].shape[1]), i))


def analyze_variant(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    gh, gw = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner = _find_marker(g, bg)
    rects = _extract_rectangles(g, bg, mark)
    allpix: list[int] = []
    for _r, _c, _h, _w, arr in rects:
        allpix.extend(map(int, arr.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [_flip_piece(arr, corner) for _r, _c, _h, _w, arr in rects]
    fill_order = _fill_order(pieces, rect_col)
    fills = [_piece_stats(pieces[i], rect_col)[2] for i in fill_order]
    gaps = [round(fills[i + 1] - fills[i], 4) for i in range(len(fills) - 1)]
    dims = _candidate_dims(int(sum(p.size for p in pieces)), missing, corner, gh, gw)

    fill_success = False
    area_success = False
    used_dims = None
    for H, W in dims:
        fill_rect = _growth_assemble_tl(pieces, fill_order, H, W, rect_col)
        if fill_rect is not None:
            fill_success = True
            used_dims = (H, W)
            area_success = _growth_assemble_tl(pieces, _area_desc_order(pieces), H, W, rect_col) is not None
            break

    return {
        "piece_count": len(pieces),
        "fill_ratios": [round(x, 4) for x in fills],
        "fill_gaps": gaps,
        "strict_increase": bool(_strictly_increasing_fill(fill_order, pieces, rect_col)),
        "fill_order_succeeds": bool(fill_success),
        "area_order_succeeds": bool(area_success),
        "used_dims": used_dims,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the pieces by:
    1. detecting the marker corner,
    2. sorting pieces by increasing fill ratio,
    3. canonicalizing the named corner to TL via flips,
    4. greedily growing a rectangle in that fixed order with no backtracking.
    """
    g = np.array(grid, dtype=int)
    gh, gw = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner = _find_marker(g, bg)
    rects = _extract_rectangles(g, bg, mark)
    assert rects

    allpix: list[int] = []
    for _r, _c, _h, _w, arr in rects:
        allpix.extend(map(int, arr.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [_flip_piece(arr, corner) for _r, _c, _h, _w, arr in rects]
    total_area = int(sum(int(p.shape[0] * p.shape[1]) for p in pieces))
    dims = _candidate_dims(total_area, missing, corner, gh, gw)
    assert dims

    order = _fill_order(pieces, rect_col)
    assert _strictly_increasing_fill(order, pieces, rect_col)

    assembled_tl = None
    used_dims = None
    for H, W in dims:
        assembled_tl = _growth_assemble_tl(pieces, order, int(H), int(W), rect_col)
        if assembled_tl is not None:
            used_dims = (int(H), int(W))
            break

    assert assembled_tl is not None and used_dims is not None
    H, W = used_dims
    assembled = _unflip_rect(assembled_tl, corner)
    tr, tc = _top_left_from_corner(corner, missing, H, W)

    out = np.full((gh, gw), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = assembled
    return out


def _build_growth_plan(rng: random.Random) -> tuple[int, int, list[tuple[int, int, int, int]], list[tuple[str, int, int]]] | None:
    seed_choices = [(2, 3), (3, 2), (3, 3)]
    seed_h, seed_w = rng.choice(seed_choices)
    n_pieces = rng.choice([4, 4, 5])

    cur_h, cur_w = seed_h, seed_w
    pieces = [(0, 0, seed_h, seed_w)]
    steps: list[tuple[str, int, int]] = []
    seen_orients = set()

    for _ in range(n_pieces - 1):
        opts = []
        if cur_w + 2 <= 8:
            opts.append("H")
        if cur_h + 2 <= 8:
            opts.append("V")
        if not opts:
            return None

        rng.shuffle(opts)
        orient = opts[0]
        deltas = [d for d in (2, 3) if (cur_w + d <= 8 if orient == "H" else cur_h + d <= 8)]
        if not deltas:
            return None
        delta = rng.choice(deltas)

        if orient == "H":
            pieces.append((0, cur_w, cur_h, delta))
            steps.append(("H", cur_h, cur_w))
            cur_w += delta
        else:
            pieces.append((cur_h, 0, delta, cur_w))
            steps.append(("V", cur_h, cur_w))
            cur_h += delta
        seen_orients.add(orient)

    if cur_h < 6 or cur_w < 6:
        return None
    if len(seen_orients) < 2:
        return None
    return cur_h, cur_w, pieces, steps


def _transform_bbox(bbox: tuple[int, int, int, int], H: int, W: int, corner: str) -> tuple[int, int, int, int]:
    r, c, h, w = bbox
    nr, nc = int(r), int(c)
    if corner in ("TR", "BR"):
        nc = int(W - c - w)
    if corner in ("BL", "BR"):
        nr = int(H - r - h)
    return nr, nc, int(h), int(w)


def _outer_border_mask(piece_bbox: tuple[int, int, int, int], H: int, W: int) -> np.ndarray:
    r0, c0, h, w = piece_bbox
    mask = np.zeros((h, w), dtype=bool)
    for rr in range(h):
        for cc in range(w):
            gr = r0 + rr
            gc = c0 + cc
            if gr == 0 or gr == H - 1 or gc == 0 or gc == W - 1:
                mask[rr, cc] = True
    return mask


def _choose_fill_targets(
    pieces: list[tuple[int, int, int, int]],
    rect: np.ndarray,
    rect_col: int,
    H: int,
    W: int,
    rng: random.Random,
) -> list[int] | None:
    areas = [int(h * w) for _r, _c, h, w in pieces]
    base = []
    caps = []
    for bbox in pieces:
        r0, c0, h, w = bbox
        sub = rect[r0 : r0 + h, c0 : c0 + w]
        outer = _outer_border_mask(bbox, H, W)
        base.append(int(np.sum(sub != int(rect_col))))
        caps.append(int(np.sum(~outer)))

    target_low = rng.uniform(0.10, 0.16)
    target_high = rng.uniform(0.30, 0.42)
    pref = [
        target_low if len(pieces) == 1 else target_low + i * (target_high - target_low) / (len(pieces) - 1)
        for i in range(len(pieces))
    ]

    def dfs(i: int, prev_ratio: float) -> list[int] | None:
        if i == len(pieces):
            return []
        candidates = []
        lo = max(1, base[i])
        hi = min(caps[i], max(base[i], int(math.floor(0.45 * areas[i]))))
        for cnt in range(lo, hi + 1):
            ratio = cnt / areas[i]
            if ratio <= prev_ratio + 0.03:
                continue
            candidates.append((abs(ratio - pref[i]), ratio, cnt))
        candidates.sort()
        for _dist, ratio, cnt in candidates:
            rest = dfs(i + 1, ratio)
            if rest is not None:
                return [cnt] + rest
        return None

    return dfs(0, -1.0)


def _add_fillers(
    rect: np.ndarray,
    pieces: list[tuple[int, int, int, int]],
    rect_col: int,
    H: int,
    W: int,
    targets: list[int],
    interior_cols: list[int],
) -> None:
    for i, bbox in enumerate(pieces):
        r0, c0, h, w = bbox
        sub = rect[r0 : r0 + h, c0 : c0 + w]
        outer = _outer_border_mask(bbox, H, W)
        current = int(np.sum(sub != int(rect_col)))
        extra = int(targets[i] - current)
        if extra <= 0:
            continue
        candidates = [
            (rr, cc)
            for rr in range(h)
            for cc in range(w)
            if (not outer[rr, cc]) and int(sub[rr, cc]) == int(rect_col)
        ]
        candidates.sort(key=lambda p: (p[0] + p[1], p[0], p[1]))
        if len(candidates) < extra:
            raise ValueError("not enough filler cells")
        color = int(interior_cols[i % len(interior_cols)])
        for rr, cc in candidates[:extra]:
            sub[rr, cc] = color


def _assign_seam_clues(
    rect: np.ndarray,
    pieces: list[tuple[int, int, int, int]],
    steps: list[tuple[str, int, int]],
    interior_cols: list[int],
    rng: random.Random,
) -> bool:
    H, W = rect.shape
    for i, (orient, prev_h, prev_w) in enumerate(steps, start=1):
        color = int(interior_cols[(i - 1) % len(interior_cols)])
        if orient == "H":
            valid = [r for r in range(prev_h) if 0 < r < H - 1]
            if not valid:
                return False
            k = 1 + int(len(valid) >= 4 and rng.random() < 0.5)
            use = rng.sample(valid, k=min(k, len(valid)))
            for r in use:
                rect[r, prev_w - 1] = color
                rect[r, prev_w] = color
        else:
            valid = [c for c in range(prev_w) if 0 < c < W - 1]
            if not valid:
                return False
            k = 1 + int(len(valid) >= 4 and rng.random() < 0.5)
            use = rng.sample(valid, k=min(k, len(valid)))
            for c in use:
                rect[prev_h - 1, c] = color
                rect[prev_h, c] = color
    return True


def _make_canonical_rectangle(
    rng: random.Random,
    rect_col: int,
    interior_cols: list[int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    plan = _build_growth_plan(rng)
    if plan is None:
        return None
    H, W, pieces, steps = plan

    rect = np.full((H, W), int(rect_col), dtype=int)
    if not _assign_seam_clues(rect, pieces, steps, interior_cols, rng):
        return None

    targets = _choose_fill_targets(pieces, rect, rect_col, H, W, rng)
    if targets is None:
        return None
    _add_fillers(rect, pieces, rect_col, H, W, targets, interior_cols)

    ratios = []
    for bbox in pieces:
        r0, c0, h, w = bbox
        sub = rect[r0 : r0 + h, c0 : c0 + w]
        fill = int(np.sum(sub != int(rect_col)))
        ratios.append(fill / (h * w))
    if any(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)):
        return None
    if min(ratios[i + 1] - ratios[i] for i in range(len(ratios) - 1)) < 0.03:
        return None

    for bbox in pieces:
        r0, c0, h, w = bbox
        sub = rect[r0 : r0 + h, c0 : c0 + w]
        fill = int(np.sum(sub != int(rect_col))) / (h * w)
        if fill < 0.10 or fill > 0.45:
            return None
        has_non_rc_edge = any(
            np.any(edge != int(rect_col))
            for edge in (sub[0, :], sub[-1, :], sub[:, 0], sub[:, -1])
        )
        if not has_non_rc_edge:
            return None

    return rect, pieces


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _area_order_fails(canon_rect: np.ndarray, canon_boxes: list[tuple[int, int, int, int]], rect_col: int) -> bool:
    pieces = [canon_rect[r : r + h, c : c + w].copy() for r, c, h, w in canon_boxes]
    fill_order = _fill_order(pieces, rect_col)
    if fill_order == _area_desc_order(pieces):
        return False
    H, W = canon_rect.shape
    if _growth_assemble_tl(pieces, fill_order, H, W, rect_col) is None:
        return False
    return _growth_assemble_tl(pieces, _area_desc_order(pieces), H, W, rect_col) is None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    gh = rng.randint(12, 18)
    gw = rng.randint(12, 18)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    interior_cols = [int(palette[3]), int(palette[4])]
    corner = rng.choice(["TL", "TR", "BL", "BR"])

    built = _make_canonical_rectangle(rng, rect_col, interior_cols)
    if built is None:
        return None
    canon_rect, canon_boxes = built
    RH, RW = map(int, canon_rect.shape)

    if not _area_order_fails(canon_rect, canon_boxes, rect_col):
        return None

    transformed_rect = np.array(canon_rect, dtype=int)
    if corner in ("TR", "BR"):
        transformed_rect = np.fliplr(transformed_rect)
    if corner in ("BL", "BR"):
        transformed_rect = np.flipud(transformed_rect)
    boxes = [_transform_bbox(b, RH, RW, corner) for b in canon_boxes]

    if corner == "TL":
        mr_min, mr_max = 2, gh - RH
        mc_min, mc_max = 2, gw - RW
    elif corner == "TR":
        mr_min, mr_max = 2, gh - RH
        mc_min, mc_max = RW - 1, gw - 3
    elif corner == "BL":
        mr_min, mr_max = RH - 1, gh - 3
        mc_min, mc_max = 2, gw - RW
    else:
        mr_min, mr_max = RH - 1, gh - 3
        mc_min, mc_max = RW - 1, gw - 3

    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = rng.randint(mr_min, mr_max)
    mc = rng.randint(mc_min, mc_max)
    markers = _marker_cells(corner, mr, mc)
    if any(not (0 <= r < gh and 0 <= c < gw) for r, c in markers):
        return None

    piece_arrays = [transformed_rect[r : r + h, c : c + w].copy() for r, c, h, w in boxes]

    canvas = np.full((gh, gw), int(bg), dtype=int)
    for r, c in markers:
        canvas[r, c] = int(marker_col)

    order = list(range(len(piece_arrays)))
    rng.shuffle(order)
    try:
        for idx in order:
            arr = piece_arrays[idx]
            r0, c0 = random_free_location(canvas, arr, transparent=bg, padding=1)
            paste(canvas, arr, r0, c0)
    except Exception:
        return None

    cnt = Counter(map(int, canvas.flatten()))
    for col, n in cnt.items():
        if int(col) != int(marker_col) and int(col) != int(bg) and int(n) == 3:
            return None

    tr, tc = _top_left_from_corner(corner, (mr, mc), RH, RW)
    expected = np.full((gh, gw), int(bg), dtype=int)
    expected[tr : tr + RH, tc : tc + RW] = transformed_rect

    if Counter(map(int, canvas.flatten())).most_common(1)[0][0] != int(bg):
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    metrics = analyze_variant(canvas)
    if (not metrics["strict_increase"]) or (not metrics["fill_order_succeeds"]) or metrics["area_order_succeeds"]:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 600 attempts")
