from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
ANCHOR_COLORS = (1, 2, 3, 4, 6, 7)
BODY_SIZES = ((2, 2), (2, 3))
SIDES = ("left", "right", "top", "bottom")
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _is_filled_bbox(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _largest_solid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    best_area = 0
    best: list[tuple[int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0]:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if not mask[r0 : r1 + 1, c0 : c1 + 1].all():
                        continue
                    if area > best_area:
                        best_area = area
                        best = []
                    best.append((r0, c0, r1, c1))
    assert best_area > 0
    assert len(best) == 1, "slider body must be the unique largest solid rectangle"
    return best[0]


def _body_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _parse_stem(
    stem_mask: np.ndarray, body_box: tuple[int, int, int, int]
) -> tuple[int, int, int, tuple[int, int]]:
    r0, c0, r1, c1 = body_box
    rows, cols = np.where(stem_mask)
    assert rows.size > 0, "slider must have a connector stem"
    unique_rows = sorted(set(map(int, rows)))
    unique_cols = sorted(set(map(int, cols)))

    if len(unique_rows) == 1:
        row = unique_rows[0]
        cs = sorted(map(int, cols))
        assert cs == list(range(cs[0], cs[-1] + 1))
        assert r0 <= row <= r1
        if cs[-1] < c0:
            return 0, -1, c0 - cs[0], (row, cs[0])
        if c1 < cs[0]:
            return 0, 1, cs[-1] - c1, (row, cs[-1])
        raise AssertionError("horizontal stem must lie outside the body")

    if len(unique_cols) == 1:
        col = unique_cols[0]
        rs = sorted(map(int, rows))
        assert rs == list(range(rs[0], rs[-1] + 1))
        assert c0 <= col <= c1
        if rs[-1] < r0:
            return -1, 0, r0 - rs[0], (rs[0], col)
        if r1 < rs[0]:
            return 1, 0, rs[-1] - r1, (rs[-1], col)
        raise AssertionError("vertical stem must lie outside the body")

    raise AssertionError("stem must be a straight one-cell-wide segment")


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    parts = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    colors = np.unique(boundary)
    assert colors.size == 1, "each layer boundary must be monochrome"
    return int(colors[0])


def _layer_order(rect: np.ndarray) -> list[int]:
    top, left = 0, 0
    bottom, right = map(int, rect.shape)
    order: list[int] = []
    while top < bottom and left < right:
        color = _boundary_color(rect, top, bottom, left, right)
        order.append(color)
        previous = (top, bottom, left, right)
        while top < bottom and np.all(rect[top, left:right] == color):
            top += 1
        while top < bottom and np.all(rect[bottom - 1, left:right] == color):
            bottom -= 1
        while left < right and top < bottom and np.all(rect[top:bottom, left] == color):
            left += 1
        while left < right and top < bottom and np.all(rect[top:bottom, right - 1] == color):
            right -= 1
        assert (top, bottom, left, right) != previous
    assert len(order) >= 2
    assert len(order) == len(set(order))
    return order


def _reverse_layers(rect: np.ndarray) -> np.ndarray:
    order = _layer_order(rect)
    mapping = dict(zip(order, reversed(order)))
    out = rect.copy()
    for src, dst in mapping.items():
        out[rect == src] = dst
    return out


def _touches_mask(coords: np.ndarray, mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in coords:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _anchor_data(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    mask = (grid != BLACK) & (grid != GREY)
    comps = _components(mask)
    assert len(comps) == 3, "expected exactly three anchors"
    anchor_id = np.full(grid.shape, -1, dtype=int)
    anchors = []
    for idx, coords in enumerate(comps):
        box = _bbox_coords(coords)
        assert _is_filled_bbox(mask, box), "anchor must fill its bounding box"
        r0, c0, r1, c1 = box
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        order = _layer_order(patch)
        for r, c in coords:
            anchor_id[int(r), int(c)] = idx
        anchors.append(
            {
                "bbox": box,
                "coords": coords,
                "patch": patch,
                "order": order,
                "mask": anchor_id == idx,
            }
        )
    return anchors, anchor_id


def _slider_data(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == GREY)
    assert len(comps) == 2, "expected exactly two sliders"
    sliders = []
    for coords in comps:
        comp_mask = np.zeros(grid.shape, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        body_box = _largest_solid_rectangle(comp_mask)
        body = _body_mask(grid.shape, body_box)
        stem = comp_mask & ~body
        dr, dc, dist, far_end = _parse_stem(stem, body_box)
        sliders.append(
            {
                "coords": coords,
                "comp_mask": comp_mask,
                "body_box": body_box,
                "body_coords": np.argwhere(body),
                "stem_mask": stem,
                "direction": (dr, dc),
                "distance": dist,
                "far_end": far_end,
            }
        )
    return sliders


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each grey body along its stem to trigger layer reversal only on the touched nested anchor.

    Input structure:
    - A black background contains exactly three filled rectangular anchors made of 2-3 nested color layers.
    - Two separate grey slider objects each consist of one small solid rectangular body plus one straight 1-cell-wide stem.
    - Every stem points toward a unique anchor, and the stem end already sits immediately next to that anchor.

    Rule:
    - Remove both grey stems and translate only each grey body by exactly its stem length in the stem direction.
    - A moved body must end 4-neighbor adjacent to the anchor that its stem was pointing to.
    - For every anchor that gets touched this way, read its layer colors from outermost to innermost and repaint the layers in reversed order.
    - Anchors never touched by a moved grey body stay unchanged.

    Key insight:
    - The stem is not part of the final object; it simultaneously encodes the slide vector and selects which anchors are allowed to reverse.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert _most_common_color(g) == BLACK, "background must be black"

    anchors, anchor_id = _anchor_data(g)
    sliders = _slider_data(g)

    moves = []
    touched: set[int] = set()
    for slider in sliders:
        dr, dc = slider["direction"]
        fr, fc = slider["far_end"]
        probe = (fr + dr, fc + dc)
        pr, pc = probe
        assert 0 <= pr < g.shape[0] and 0 <= pc < g.shape[1], "stem must point to an in-bounds anchor"
        idx = int(anchor_id[pr, pc])
        assert idx >= 0, "stem end must be adjacent to exactly one anchor"
        touched.add(idx)

        body_coords = slider["body_coords"]
        moved = body_coords + np.array([dr * slider["distance"], dc * slider["distance"]], dtype=int)
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < g.shape[0]))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < g.shape[1]))
        assert _touches_mask(moved, anchors[idx]["mask"]), "moved body must touch its target anchor"
        moves.append((idx, moved))

    assert len(touched) == 2, "the two sliders must trigger two different anchors"

    out = g.copy()
    out[g == GREY] = BLACK

    for idx in touched:
        r0, c0, r1, c1 = anchors[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _reverse_layers(anchors[idx]["patch"])

    occupied = out != BLACK
    for idx, moved in moves:
        assert np.all(~occupied[moved[:, 0], moved[:, 1]]), "moved body must land on background"
        out[moved[:, 0], moved[:, 1]] = GREY
        assert _touches_mask(moved, anchors[idx]["mask"])
        occupied[moved[:, 0], moved[:, 1]] = True

    return out


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _build_anchor_patch(h: int, w: int, layers: int) -> np.ndarray | None:
    assert layers in (2, 3)
    options: list[tuple[int, ...]] = []

    def rec(cur_h: int, cur_w: int, remaining_rings: int, acc: list[int]):
        if remaining_rings == 0:
            options.append(tuple(acc))
            return
        for thickness in (1, 2):
            nh = cur_h - 2 * thickness
            nw = cur_w - 2 * thickness
            if nh < 1 or nw < 1:
                continue
            if remaining_rings > 1 and min(nh, nw) < 3:
                continue
            acc.append(thickness)
            rec(nh, nw, remaining_rings - 1, acc)
            acc.pop()

    rec(h, w, layers - 1, [])
    if not options:
        return None

    thicks = random.choice(options)
    colors = random.sample(ANCHOR_COLORS, layers)
    patch = np.zeros((h, w), dtype=int)
    top, left = 0, 0
    bottom, right = h, w
    for i, color in enumerate(colors):
        patch[top:bottom, left:right] = color
        if i < len(thicks):
            t = thicks[i]
            top += t
            left += t
            bottom -= t
            right -= t
    return patch


def _render_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _bbox_clear(
    box: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    padding: int,
) -> bool:
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    for rr in range(r0 - padding, r1 + padding + 1):
        for cc in range(c0 - padding, c1 + padding + 1):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) in occupied:
                return False
    return True


def _patch_signature(patch: np.ndarray) -> bytes:
    return patch.tobytes() + str(patch.shape).encode()


def _place_anchors(h: int, w: int) -> list[dict] | None:
    layer_counts = [2, 3, random.choice((2, 3))]
    random.shuffle(layer_counts)
    dims = [(random.randint(4, 6), random.randint(4, 6)) for _ in range(3)]
    patches = []
    seen_sigs = set()
    for (ah, aw), layers in zip(dims, layer_counts):
        patch = _build_anchor_patch(ah, aw, layers)
        if patch is None:
            return None
        sig = _patch_signature(patch)
        if sig in seen_sigs:
            return None
        seen_sigs.add(sig)
        patches.append(patch)

    anchors: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    order = list(range(3))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ah, aw = patch.shape
        placed = False
        positions = [(r, c) for r in range(1, h - ah) for c in range(1, w - aw)]
        random.shuffle(positions)
        for r0, c0 in positions:
            box = (r0, c0, r0 + ah - 1, c0 + aw - 1)
            if not _bbox_clear(box, occupied, h, w, padding=1):
                continue
            coords = {(r, c) for r in range(r0, r0 + ah) for c in range(c0, c0 + aw)}
            anchors.append(
                {
                    "bbox": box,
                    "patch": patch.copy(),
                    "coords_set": coords,
                    "layers": _layer_order(patch),
                }
            )
            occupied.update(coords)
            placed = True
            break
        if not placed:
            return None
    return anchors


def _gray_adjacent_forbidden(
    cells: set[tuple[int, int]],
    existing_gray: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in existing_gray:
                return True
    return False


def _anchor_adjacencies(
    cells: set[tuple[int, int]], anchors: list[dict]
) -> dict[int, set[tuple[int, int]]]:
    out: dict[int, set[tuple[int, int]]] = {}
    for idx, anchor in enumerate(anchors):
        touched = set()
        aset = anchor["coords_set"]
        for r, c in cells:
            for nbr in _neighbors4(r, c):
                if nbr in aset:
                    touched.add((r, c))
        if touched:
            out[idx] = touched
    return out


def _slider_candidate(
    anchor_idx: int,
    anchor: dict,
    anchors: list[dict],
    input_gray: set[tuple[int, int]],
    output_gray: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[dict]:
    r0, c0, r1, c1 = anchor["bbox"]
    candidates = []
    for bh, bw in BODY_SIZES:
        for length in range(2, 6):
            for side in SIDES:
                if side == "left":
                    out_c = c0 - bw
                    in_c = out_c - length
                    row_lo = max(0, r0 - bh + 1)
                    row_hi = min(h - bh, r1)
                    if in_c < 0 or out_c < 0:
                        continue
                    starts = list(range(row_lo, row_hi + 1))
                    random.shuffle(starts)
                    for out_r in starts:
                        overlap0 = max(out_r, r0)
                        overlap1 = min(out_r + bh - 1, r1)
                        attach_rows = list(range(overlap0, overlap1 + 1))
                        random.shuffle(attach_rows)
                        for attach_r in attach_rows:
                            body_in = {(rr, cc) for rr in range(out_r, out_r + bh) for cc in range(in_c, in_c + bw)}
                            body_out = {(rr, cc) for rr in range(out_r, out_r + bh) for cc in range(out_c, out_c + bw)}
                            stem = {(attach_r, cc) for cc in range(in_c + bw, in_c + bw + length)}
                            far_end = (attach_r, in_c + bw + length - 1)
                            candidates.append(
                                {
                                    "side": side,
                                    "length": length,
                                    "body_in": body_in,
                                    "body_out": body_out,
                                    "stem": stem,
                                    "far_end": far_end,
                                }
                            )
                elif side == "right":
                    out_c = c1 + 1
                    in_c = out_c + length
                    row_lo = max(0, r0 - bh + 1)
                    row_hi = min(h - bh, r1)
                    if in_c + bw > w or out_c + bw > w:
                        continue
                    starts = list(range(row_lo, row_hi + 1))
                    random.shuffle(starts)
                    for out_r in starts:
                        overlap0 = max(out_r, r0)
                        overlap1 = min(out_r + bh - 1, r1)
                        attach_rows = list(range(overlap0, overlap1 + 1))
                        random.shuffle(attach_rows)
                        for attach_r in attach_rows:
                            body_in = {(rr, cc) for rr in range(out_r, out_r + bh) for cc in range(in_c, in_c + bw)}
                            body_out = {(rr, cc) for rr in range(out_r, out_r + bh) for cc in range(out_c, out_c + bw)}
                            stem = {(attach_r, cc) for cc in range(out_c, out_c + length)}
                            far_end = (attach_r, out_c)
                            candidates.append(
                                {
                                    "side": side,
                                    "length": length,
                                    "body_in": body_in,
                                    "body_out": body_out,
                                    "stem": stem,
                                    "far_end": far_end,
                                }
                            )
                elif side == "top":
                    out_r = r0 - bh
                    in_r = out_r - length
                    col_lo = max(0, c0 - bw + 1)
                    col_hi = min(w - bw, c1)
                    if in_r < 0 or out_r < 0:
                        continue
                    starts = list(range(col_lo, col_hi + 1))
                    random.shuffle(starts)
                    for out_c in starts:
                        overlap0 = max(out_c, c0)
                        overlap1 = min(out_c + bw - 1, c1)
                        attach_cols = list(range(overlap0, overlap1 + 1))
                        random.shuffle(attach_cols)
                        for attach_c in attach_cols:
                            body_in = {(rr, cc) for rr in range(in_r, in_r + bh) for cc in range(out_c, out_c + bw)}
                            body_out = {(rr, cc) for rr in range(out_r, out_r + bh) for cc in range(out_c, out_c + bw)}
                            stem = {(rr, attach_c) for rr in range(in_r + bh, in_r + bh + length)}
                            far_end = (in_r + bh + length - 1, attach_c)
                            candidates.append(
                                {
                                    "side": side,
                                    "length": length,
                                    "body_in": body_in,
                                    "body_out": body_out,
                                    "stem": stem,
                                    "far_end": far_end,
                                }
                            )
                else:
                    out_r = r1 + 1
                    in_r = out_r + length
                    col_lo = max(0, c0 - bw + 1)
                    col_hi = min(w - bw, c1)
                    if in_r + bh > h or out_r + bh > h:
                        continue
                    starts = list(range(col_lo, col_hi + 1))
                    random.shuffle(starts)
                    for out_c in starts:
                        overlap0 = max(out_c, c0)
                        overlap1 = min(out_c + bw - 1, c1)
                        attach_cols = list(range(overlap0, overlap1 + 1))
                        random.shuffle(attach_cols)
                        for attach_c in attach_cols:
                            body_in = {(rr, cc) for rr in range(in_r, in_r + bh) for cc in range(out_c, out_c + bw)}
                            body_out = {(rr, cc) for rr in range(out_r, out_r + bh) for cc in range(out_c, out_c + bw)}
                            stem = {(rr, attach_c) for rr in range(out_r, out_r + length)}
                            far_end = (out_r, attach_c)
                            candidates.append(
                                {
                                    "side": side,
                                    "length": length,
                                    "body_in": body_in,
                                    "body_out": body_out,
                                    "stem": stem,
                                    "far_end": far_end,
                                }
                            )

    valid = []
    all_anchor_cells = set().union(*(a["coords_set"] for a in anchors))
    for cand in candidates:
        body_in = cand["body_in"]
        body_out = cand["body_out"]
        stem = cand["stem"]
        comp = body_in | stem
        if body_in & all_anchor_cells or stem & all_anchor_cells:
            continue
        if comp & input_gray:
            continue
        if body_out & all_anchor_cells:
            continue
        if body_out & output_gray:
            continue
        if _gray_adjacent_forbidden(comp, input_gray, h, w):
            continue
        if _gray_adjacent_forbidden(body_out, output_gray, h, w):
            continue

        input_adj = _anchor_adjacencies(comp, anchors)
        target_adj = input_adj.get(anchor_idx, set())
        if target_adj != {cand["far_end"]}:
            continue
        if set(input_adj) != {anchor_idx}:
            continue

        output_adj = _anchor_adjacencies(body_out, anchors)
        if anchor_idx not in output_adj:
            continue
        if set(output_adj) != {anchor_idx}:
            continue

        non_target_touch = False
        for r, c in body_out:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in output_gray:
                    non_target_touch = True
                    break
            if non_target_touch:
                break
        if non_target_touch:
            continue

        cand = cand.copy()
        cand["anchor_idx"] = anchor_idx
        valid.append(cand)
    return valid


def _render_input(h: int, w: int, anchors: list[dict], sliders: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for anchor in anchors:
        r0, c0, r1, c1 = anchor["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = anchor["patch"]
    for slider in sliders:
        for r, c in slider["body_in"] | slider["stem"]:
            grid[r, c] = GREY
    return grid


def _validate_generated(inp: np.ndarray, anchors: list[dict], sliders: list[dict], touched_ids: set[int]) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    if np.array_equal(inp, out):
        return False

    changed = set()
    unchanged = set()
    for idx, anchor in enumerate(anchors):
        r0, c0, r1, c1 = anchor["bbox"]
        in_patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if np.array_equal(in_patch, out_patch):
            unchanged.add(idx)
        elif np.array_equal(out_patch, _reverse_layers(anchor["patch"])):
            changed.add(idx)
        else:
            return False

    if changed != touched_ids:
        return False
    if len(unchanged) != 1:
        return False

    output_gray = _components(out == GREY)
    if len(output_gray) != 2:
        return False

    anchor_masks = []
    for anchor in anchors:
        mask = np.zeros(inp.shape, dtype=bool)
        for r, c in anchor["coords_set"]:
            mask[r, c] = True
        anchor_masks.append(mask)

    touch_counts = Counter()
    for comp in output_gray:
        touched_here = [idx for idx, mask in enumerate(anchor_masks) if _touches_mask(comp, mask)]
        if len(touched_here) != 1:
            return False
        touch_counts[touched_here[0]] += 1

    return set(touch_counts) == touched_ids and all(v == 1 for v in touch_counts.values())


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        h = random.randint(17, 22)
        w = random.randint(17, 22)
        anchors = _place_anchors(h, w)
        if anchors is None:
            continue

        touched = set(random.sample(range(3), 2))
        input_gray: set[tuple[int, int]] = set()
        output_gray: set[tuple[int, int]] = set()
        sliders = []
        ok = True

        order = list(touched)
        random.shuffle(order)
        for anchor_idx in order:
            cands = _slider_candidate(anchor_idx, anchors[anchor_idx], anchors, input_gray, output_gray, h, w)
            if not cands:
                ok = False
                break
            cand = random.choice(cands)
            sliders.append(cand)
            input_gray.update(cand["body_in"] | cand["stem"])
            output_gray.update(cand["body_out"])

        if not ok:
            continue

        inp = _render_input(h, w, anchors, sliders)
        if _validate_generated(inp, anchors, sliders, touched):
            return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
