# Combo variant of 16b78196 (index 25)
# combo_spec: D4/2+D11/0+D16/1
# Name: 颜色排序+有序散布+同链同色

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import ARC_RGB, components_mask, find_components, get_bg

# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original input scatters all pieces independently, so solver must search both chain membership
#    and within-chain order. Here each chain becomes a same-color group, and the pieces of a group
#    are laid out on one side of the bar from near to far in assembly order. Because D16/1 removes
#    per-piece color variation inside a chain, D4/2 is lifted to the chain level: the darker chain
#    is always placed on the negative side of the bar (top for horizontal, left for vertical) and
#    the brighter chain on the positive side. Example: with a horizontal bar, all blue pieces above
#    the bar may form one chain ordered by increasing distance upward, while all yellow pieces below
#    the bar form the other chain ordered downward. Output assembly still uses exact shape/cavity
#    matching, so the cues only narrow the search.
# b) Complexity calibration:
#    The puzzle is easier than the base task, but not trivialized. Color groups reveal chain
#    membership; monotone distance reveals within-chain order; shape matching against notches and
#    cavities still decides where each ordered piece actually fits. Keeping exactly two chains with
#    repeated same-color evidence makes the rule visually legible across 2-4 examples.
# c) Hidden assumptions audited:
#    - The largest monochrome component is still the bar.
#    - Pieces remain unrotated in the input and output.
#    - Each color group lies wholly on one side of the bar.
#    - The nearest piece of a group is the root candidate for that chain.
#    - The darker color group maps to top/left and the brighter group to bottom/right.
#    Generator construction enforces all of these assumptions explicitly.
# d) Placement legality:
#    Input placement uses ordered distance bands and collision checks, so pieces never overlap the
#    bar, each other, or the grid boundary.
# e) Visual observability:
#    The extra clues are visible in raw pixels: same-color grouping, one-sided ordered layout, and
#    a consistent dark-vs-bright side convention. No hidden use of numeric color ids is required;
#    brightness is derived from the visible ARC palette RGB values.
# f) Visual cleanliness:
#    Each chain occupies a different side region of the bar, and pieces are separated by at least
#    one blank cell, so the layout stays readable rather than cluttered.
# g) Inferability of color behavior:
#    Only two chain colors are used, chosen from dark and bright subsets with a large luminance
#    gap. This keeps the color rule simple enough to infer from a few examples.


BRIGHTNESS = {int(c): float(np.dot(ARC_RGB[int(c)], np.array([0.299, 0.587, 0.114]))) for c in range(len(ARC_RGB))}
VISIBLE_ORDER = [9, 1, 2, 3, 6, 5, 7, 8, 4]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _brightness(color: int) -> float:
    return float(BRIGHTNESS[int(color)])


def _shape_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, object]:
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {
        "rel": rel,
        "rel_set": set(rel),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
    }


def _pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top) + int(lr), int(left) + int(lc)) for lr, lc in shape["rel"]}


def _bar_notches(bar_mask: np.ndarray, bar_r0: int, bar_c0: int) -> list[dict[str, object]]:
    empty = ~bar_mask
    h, w = map(int, empty.shape)
    notches: list[dict[str, object]] = []
    for pts in components_mask(empty, connectivity=4):
        r = pts[:, 0]
        c = pts[:, 1]
        top = int((r == 0).sum())
        bottom = int((r == h - 1).sum())
        left = int((c == 0).sum())
        right = int((c == w - 1).sum())
        if top + bottom + left + right == 0:
            continue
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        hole_gl = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}
        notches.append({"side": side, "axis": axis, "hole": hole_gl, "mouth_n": int(mouth_n)})
    return notches


def _placement_pts(
    shape: dict[str, object],
    top: int,
    left: int,
    H: int,
    W: int,
    occ: set[tuple[int, int]],
    *,
    must_cover: set[tuple[int, int]] | None = None,
    strip: tuple[str, int, int] | None = None,
) -> set[tuple[int, int]] | None:
    top = int(top)
    left = int(left)
    if top < 0 or left < 0 or top + int(shape["h"]) > H or left + int(shape["w"]) > W:
        return None
    if strip is not None:
        axis, A, B = strip
        if axis == "cols":
            if left != int(A) or int(shape["w"]) != int(B) - int(A) + 1:
                return None
        else:
            if top != int(A) or int(shape["h"]) != int(B) - int(A) + 1:
                return None
    pts = _pixels(shape, top, left)
    if pts & occ:
        return None
    if must_cover is not None and not (must_cover <= pts):
        return None
    return pts


def _aligned_positions(axis: str, A: int, cover: set[tuple[int, int]], shape: dict[str, object]):
    if axis == "cols":
        left = int(A)
        tops: set[int] = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if int(cN) == left + int(lc):
                    tops.add(int(rN) - int(lr))
        for top in tops:
            yield int(top), left
    else:
        top = int(A)
        lefts: set[int] = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if int(rN) == top + int(lr):
                    lefts.add(int(cN) - int(lc))
        for left in lefts:
            yield top, int(left)


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _hole_positions(hole: set[tuple[int, int]], shape: dict[str, object]) -> set[tuple[int, int]]:
    pos: set[tuple[int, int]] = set()
    for rH, cH in hole:
        for lr, lc in shape["rel"]:
            pos.add((int(rH) - int(lr), int(cH) - int(lc)))
    return pos


def _orientation(bar_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bar_bbox)
    return (r1 - r0) <= (c1 - c0)


def _piece_side(bbox: tuple[int, int, int, int], bar_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = map(int, bbox)
    br0, bc0, br1, bc1 = map(int, bar_bbox)
    if r1 < br0:
        return "top"
    if r0 > br1:
        return "bottom"
    if c1 < bc0:
        return "left"
    if c0 > bc1:
        return "right"
    raise AssertionError("piece overlaps bar side bands")


def _gap_from_bar(bbox: tuple[int, int, int, int], bar_bbox: tuple[int, int, int, int], side: str) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    br0, bc0, br1, bc1 = map(int, bar_bbox)
    if side == "top":
        return int(br0 - r1 - 1)
    if side == "bottom":
        return int(r0 - br1 - 1)
    if side == "left":
        return int(bc0 - c1 - 1)
    if side == "right":
        return int(c0 - bc1 - 1)
    raise ValueError(side)


def _root_extends_outward(side: str, top: int, left: int, shape: dict[str, object], bar_bbox: tuple[int, int, int, int]) -> bool:
    br0, bc0, br1, bc1 = map(int, bar_bbox)
    bottom = int(top) + int(shape["h"]) - 1
    right = int(left) + int(shape["w"]) - 1
    if side == "top":
        return int(top) < br0
    if side == "bottom":
        return bottom > br1
    if side == "left":
        return int(left) < bc0
    if side == "right":
        return right > bc1
    return False


def _extends_farther(
    side: str,
    prev: dict[str, object],
    new_top: int,
    new_left: int,
    new_shape: dict[str, object],
) -> bool:
    prev_top = int(prev["top"])
    prev_left = int(prev["left"])
    prev_bottom = prev_top + int(prev["shape"]["h"]) - 1
    prev_right = prev_left + int(prev["shape"]["w"]) - 1
    new_bottom = int(new_top) + int(new_shape["h"]) - 1
    new_right = int(new_left) + int(new_shape["w"]) - 1
    if side == "top":
        return int(new_top) < prev_top
    if side == "bottom":
        return new_bottom > prev_bottom
    if side == "left":
        return int(new_left) < prev_left
    if side == "right":
        return new_right > prev_right
    return False


def _extension_score(side: str, top: int, left: int, shape: dict[str, object], bar_bbox: tuple[int, int, int, int]) -> int:
    bbox = (int(top), int(left), int(top) + int(shape["h"]) - 1, int(left) + int(shape["w"]) - 1)
    return _gap_from_bar(bbox, bar_bbox, side)


def _cavity_cells(out_grid: np.ndarray, bg: int, head: dict[str, object]) -> set[tuple[int, int]]:
    top = int(head["top"])
    left = int(head["left"])
    h = int(head["shape"]["h"])
    w = int(head["shape"]["w"])
    return {
        (r, c)
        for r in range(top, top + h)
        for c in range(left, left + w)
        if int(out_grid[r, c]) == int(bg)
    }


def _ordered_notch_candidates(
    notches: list[dict[str, object]],
    side_order: list[str],
    group_side: str,
    preferred_side: str | None,
) -> list[dict[str, object]]:
    ordered: list[dict[str, object]] = []
    wanted: list[str] = []
    for side in [preferred_side, group_side, *side_order]:
        if side is not None and side not in wanted:
            wanted.append(side)
    for side in wanted:
        chunk = [n for n in notches if n["side"] == side]
        chunk.sort(key=lambda n: (-int(n["mouth_n"]), len(n["hole"])))
        ordered.extend(chunk)
    return ordered


def _apply_chain(
    out_grid: np.ndarray,
    occ: set[tuple[int, int]],
    bg: int,
    bar_bbox: tuple[int, int, int, int],
    notches: list[dict[str, object]],
    group: dict[str, object],
    side_order: list[str],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    pieces = group["ordered"]
    root_piece = pieces[0]
    ordered_notches = _ordered_notch_candidates(notches, side_order, group["side"], group.get("preferred_side"))
    H, W = out_grid.shape
    for notch in ordered_notches:
        axis = str(notch["axis"])
        root_shape = root_piece["shape"]
        for top, left in _hole_positions(notch["hole"], root_shape):
            pts = _placement_pts(root_shape, top, left, H, W, occ, must_cover=notch["hole"])
            if pts is None:
                continue
            if not _root_extends_outward(str(notch["side"]), top, left, root_shape, bar_bbox):
                continue
            A, B = (int(left), int(left) + int(root_shape["w"]) - 1) if axis == "cols" else (int(top), int(top) + int(root_shape["h"]) - 1)
            temp_grid = out_grid.copy()
            temp_occ = set(occ)
            for r, c in pts:
                temp_grid[r, c] = int(group["color"])
            temp_occ |= pts
            head = {"top": int(top), "left": int(left), "shape": root_shape}
            ok = True
            for piece in pieces[1:]:
                cavity = _cavity_cells(temp_grid, bg, head)
                if not cavity:
                    ok = False
                    break
                moves: list[tuple[int, int, set[tuple[int, int]], dict[str, object]]] = []
                shape = piece["shape"]
                head_pts = _pixels(head["shape"], int(head["top"]), int(head["left"]))
                for nxt_top, nxt_left in _aligned_positions(axis, A, cavity, shape):
                    nxt_pts = _placement_pts(shape, nxt_top, nxt_left, H, W, temp_occ, must_cover=cavity, strip=(axis, A, B))
                    if nxt_pts is None:
                        continue
                    if not _touches(nxt_pts, head_pts):
                        continue
                    if not _extends_farther(str(notch["side"]), head, nxt_top, nxt_left, shape):
                        continue
                    score = _extension_score(str(notch["side"]), nxt_top, nxt_left, shape, bar_bbox)
                    moves.append((int(score), int(nxt_top), nxt_left, nxt_pts, shape))
                if not moves:
                    ok = False
                    break
                moves.sort(key=lambda item: (-item[0], item[1], item[2]))
                _score, nxt_top, nxt_left, nxt_pts, shape = moves[0]
                for r, c in nxt_pts:
                    temp_grid[r, c] = int(group["color"])
                temp_occ |= nxt_pts
                head = {"top": int(nxt_top), "left": int(nxt_left), "shape": shape}
            if ok and not _cavity_cells(temp_grid, bg, head):
                return temp_grid, temp_occ
    raise AssertionError(f"failed to place chain color={group['color']}")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -int(d["area"]))
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]
    bar_bbox = tuple(map(int, bar["bbox"]))
    br0, bc0, br1, bc1 = bar_bbox
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
    for r, c in bar_occ:
        bar_mask[int(r) - br0, int(c) - bc0] = True

    out = np.full((H, W), bg, dtype=int)
    for r, c in bar_occ:
        out[r, c] = int(bar["color"])

    notches = _bar_notches(bar_mask, br0, bc0)
    is_horizontal = _orientation(bar_bbox)
    side_order = ["top", "bottom"] if is_horizontal else ["left", "right"]

    pieces: list[dict[str, object]] = []
    for p in pieces_raw:
        bbox = tuple(map(int, p["bbox"]))
        pieces.append(
            {
                "color": int(p["color"]),
                "bbox": bbox,
                "shape": _shape_from_coords(p["coords"], bbox),
                "side": _piece_side(bbox, bar_bbox),
            }
        )

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for piece in pieces:
        by_color[int(piece["color"])].append(piece)

    groups: list[dict[str, object]] = []
    for color, chunk in by_color.items():
        sides = {str(piece["side"]) for piece in chunk}
        assert len(sides) == 1
        side = next(iter(sides))
        ordered = sorted(chunk, key=lambda piece: _gap_from_bar(tuple(piece["bbox"]), bar_bbox, side))
        groups.append({"color": int(color), "side": side, "ordered": ordered})

    groups.sort(key=lambda grp: (_brightness(int(grp["color"])), int(grp["color"])))
    for idx, group in enumerate(groups):
        if idx < len(side_order):
            group["preferred_side"] = side_order[idx]
        else:
            group["preferred_side"] = None

    occ = set(bar_occ)
    for group in groups:
        out, occ = _apply_chain(out, occ, bg, bar_bbox, notches, group, side_order)
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _rot90_cw(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=-1)


def _rot90_ccw(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=1)


def _rot180(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=2)


def _trim_piece(piece: np.ndarray) -> np.ndarray | None:
    rows = np.any(piece, axis=1)
    cols = np.any(piece, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return piece[rmin : rmax + 1, cmin : cmax + 1].copy()


def _is_connected_piece(piece: np.ndarray) -> bool:
    if int(np.sum(piece)) == 0:
        return False
    h, w = piece.shape
    seen = np.zeros_like(piece, dtype=bool)
    start = None
    for r in range(h):
        for c in range(w):
            if int(piece[r, c]) == 1:
                start = (r, c)
                break
        if start is not None:
            break
    if start is None:
        return False
    stack = [start]
    count = 0
    while stack:
        r, c = stack.pop()
        if not (0 <= r < h and 0 <= c < w):
            continue
        if seen[r, c] or int(piece[r, c]) == 0:
            continue
        seen[r, c] = True
        count += 1
        stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    return count == int(np.sum(piece))


def _is_rectangular_piece(piece: np.ndarray) -> bool:
    rows = np.any(piece, axis=1)
    cols = np.any(piece, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int((rmax - rmin + 1) * (cmax - cmin + 1)) == int(np.sum(piece))


def _validate_piece(piece: np.ndarray, original_width: int) -> bool:
    if piece is None:
        return False
    area = int(np.sum(piece))
    if area < 8 or area > 15:
        return False
    if not _is_connected_piece(piece):
        return False
    if _is_rectangular_piece(piece):
        return False
    if area <= int(original_width):
        return False
    return True


def _can_place_piece(placed_mask: np.ndarray, piece: np.ndarray, y: int, x: int) -> bool:
    gh, gw = placed_mask.shape
    ph, pw = piece.shape
    for py in range(ph):
        for px in range(pw):
            if int(piece[py, px]) != 1:
                continue
            gy, gx = int(y) + py, int(x) + px
            if not (0 <= gy < gh and 0 <= gx < gw):
                return False
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < gh and 0 <= nx < gw and bool(placed_mask[ny, nx]):
                        return False
    return True


def _holes_too_close_1d(start_a: int, size_a: int, start_b: int, size_b: int, min_gap: int) -> bool:
    end_a = int(start_a) + int(size_a) - 1
    end_b = int(start_b) + int(size_b) - 1
    if end_a < int(start_b):
        gap = int(start_b) - end_a - 1
    elif end_b < int(start_a):
        gap = int(start_a) - end_b - 1
    else:
        gap = -1
    return gap < int(min_gap)


def _is_similar_to_protrusions(h: int, w: int, protrusions: list[np.ndarray]) -> bool:
    return any(int(h) == int(prot.shape[0]) and int(w) == int(prot.shape[1]) for prot in protrusions)


def _generate_different_protrusion_types() -> list[tuple[int, int, int]]:
    possible = [(1, 2, 2), (1, 3, 3), (2, 2, 1), (2, 3, 1), (2, 3, 2)]
    indices = np.random.choice(len(possible), size=2, replace=False)
    return [possible[int(i)] for i in indices]


def _add_protrusion_with_shape(piece: np.ndarray, protrusion_type: tuple[int, int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if piece is None:
        return None, None
    h, w = piece.shape
    prot_h, prot_bottom_w, prot_top_w = protrusion_type
    top_row_idx = 0
    for y in range(h):
        if np.any(piece[y, :]):
            top_row_idx = y
            break
    top_cols = np.where(piece[top_row_idx, :] == 1)[0]
    if len(top_cols) < prot_bottom_w:
        prot_bottom_w = len(top_cols)
        prot_top_w = min(prot_top_w, prot_bottom_w)
    if prot_bottom_w < 1:
        return None, None

    new_piece = np.zeros((h + prot_h, w), dtype=np.int8)
    new_piece[prot_h:, :] = piece
    protrusion_shape = np.zeros((prot_h, prot_bottom_w), dtype=np.int8)

    max_start = max(0, len(top_cols) - prot_bottom_w)
    start_idx = int(np.random.randint(0, max_start + 1))
    base_start = int(top_cols[start_idx])
    for row in range(prot_h):
        if row == prot_h - 1:
            row_width = prot_bottom_w
            start = base_start
        else:
            row_width = prot_top_w
            start = base_start + (prot_bottom_w - prot_top_w) // 2
        end = min(w, start + row_width)
        if end > start:
            new_piece[row, start:end] = 1
            protrusion_shape[row, max(0, start - base_start) : min(prot_bottom_w, end - base_start)] = 1
    return _trim_piece(new_piece), protrusion_shape


def _create_matching_hole_horizontal(grid: np.ndarray, protrusion: np.ndarray, edge_y: int, x_start: int, direction: str) -> None:
    h, w = protrusion.shape
    for py in range(h):
        for px in range(w):
            if int(protrusion[py, px]) != 1:
                continue
            if direction == "top":
                y = int(edge_y) + (h - 1 - py)
            else:
                y = int(edge_y) - (h - 1 - py)
            grid[y, int(x_start) + px] = 0


def _add_distractor_holes_horizontal(
    grid: np.ndarray,
    top_y: int,
    bottom_y: int,
    grid_size: int,
    existing_holes: list[tuple[str, int, int]],
    num_distractors: int,
    protrusions: list[np.ndarray],
) -> None:
    min_hole_gap = 4
    for _ in range(int(num_distractors)):
        side = str(np.random.choice(["top", "bottom"]))
        dist_h = int(np.random.randint(1, 3))
        dist_w = int(np.random.randint(1, 4))
        while _is_similar_to_protrusions(dist_h, dist_w, protrusions):
            dist_h = int(np.random.randint(1, 3))
            dist_w = int(np.random.randint(1, 4))
        for _attempt in range(60):
            x = int(np.random.randint(2, grid_size - dist_w - 2))
            valid = True
            for hole_side, hole_x, hole_w in existing_holes:
                if hole_side == side and _holes_too_close_1d(x, dist_w, hole_x, hole_w, min_hole_gap):
                    valid = False
                    break
            if not valid:
                continue
            if side == "top":
                for py in range(dist_h):
                    for px in range(dist_w):
                        if int(top_y) + py < int(bottom_y):
                            grid[int(top_y) + py, x + px] = 0
            else:
                for py in range(dist_h):
                    for px in range(dist_w):
                        if int(bottom_y) - py > int(top_y):
                            grid[int(bottom_y) - py, x + px] = 0
            existing_holes.append((side, x, dist_w))
            break


def _create_holes_horizontal_with_sides(
    grid: np.ndarray,
    protrusions: list[np.ndarray],
    top_y: int,
    bottom_y: int,
    grid_size: int,
    hole_sides: list[str],
) -> list[tuple[str, int, int]]:
    min_hole_gap = 4
    hole_positions: list[tuple[str, int, int]] = []
    for prot, side in zip(protrusions, hole_sides):
        prot_for_hole = _rot180(np.flipud(prot)) if side == "top" else prot
        _h, w = prot_for_hole.shape
        for _attempt in range(60):
            x_pos = int(np.random.randint(3, grid_size - w - 3))
            valid = True
            for existing_side, existing_x, existing_w in hole_positions:
                if existing_side == side and _holes_too_close_1d(x_pos, w, existing_x, existing_w, min_hole_gap):
                    valid = False
                    break
            if valid:
                break
        else:
            return []
        if side == "top":
            _create_matching_hole_horizontal(grid, prot_for_hole, top_y, x_pos, direction="top")
        else:
            _create_matching_hole_horizontal(grid, prot_for_hole, bottom_y, x_pos, direction="bottom")
        hole_positions.append((side, x_pos, w))

    num_distractors = int(np.random.randint(4, 8))
    _add_distractor_holes_horizontal(grid, top_y, bottom_y, grid_size, hole_positions, num_distractors, protrusions)
    return hole_positions


def _generate_irregular_split_line_constrained(width: int, base_y: int) -> np.ndarray:
    split_line = np.full(width, int(base_y), dtype=np.int8)
    num_features = int(np.random.randint(1, min(3, width - 1)))
    available = list(range(1, width - 1))
    for _ in range(num_features):
        if not available:
            break
        pos = int(np.random.choice(available))
        split_line[pos] = int(base_y) + int(np.random.choice([-1, 1]))
        available = [p for p in available if abs(p - pos) > 1]
    return split_line


def _extract_pieces_from_splits_constrained(width: int, split_lines: list[np.ndarray], length: int, max_height: int) -> list[np.ndarray] | None:
    pieces: list[np.ndarray] = []
    all_lines = [np.zeros(width, dtype=np.int8)] + split_lines + [np.full(width, int(length), dtype=np.int8)]
    for i in range(len(all_lines) - 1):
        top_line = all_lines[i]
        bottom_line = all_lines[i + 1]
        if int(np.min(bottom_line)) - int(np.max(top_line)) < 1:
            return None
        min_y = max(0, int(np.min(top_line)))
        max_y = min(int(length), int(np.max(bottom_line)))
        piece_h = max_y - min_y
        if piece_h > int(max_height):
            return None
        piece = np.zeros((piece_h, width), dtype=np.int8)
        for x in range(width):
            y_start = int(top_line[x]) - min_y
            y_end = int(bottom_line[x]) - min_y
            piece[max(0, y_start) : min(piece_h, y_end), x] = 1
        trimmed = _trim_piece(piece)
        if trimmed is None or int(np.sum(trimmed)) == 0 or int(trimmed.shape[0]) > int(max_height):
            return None
        pieces.append(trimmed)
    return pieces if len(pieces) >= 2 else None


def _generate_split_pieces(width: int, length: int, num_pieces: int) -> list[np.ndarray] | None:
    max_piece_height = 6
    min_piece_base = 2
    max_piece_base = 3
    if int(length) < int(num_pieces) * int(min_piece_base):
        return None
    split_positions: list[int] = []
    remaining = int(length)
    current_y = 0
    for i in range(num_pieces - 1):
        pieces_left = num_pieces - i
        reserved = (pieces_left - 1) * int(min_piece_base)
        available = remaining - reserved
        max_for_this = min(int(max_piece_base), available)
        if max_for_this < int(min_piece_base):
            return None
        piece_height = int(np.random.randint(int(min_piece_base), max_for_this + 1))
        split_y = current_y + piece_height
        split_positions.append(int(split_y))
        current_y = int(split_y)
        remaining = int(length) - current_y
    split_lines = [_generate_irregular_split_line_constrained(width, split_y) for split_y in split_positions]
    pieces = _extract_pieces_from_splits_constrained(width, split_lines, length, max_piece_height)
    if pieces is None:
        return None
    for piece in pieces:
        full_rows = sum(int(np.sum(piece[row, :])) == int(piece.shape[1]) for row in range(piece.shape[0]))
        if full_rows < 1 or full_rows > 2:
            return None
    return pieces


def _choose_chain_colors(bar_color: int) -> list[int]:
    order = [c for c in VISIBLE_ORDER if int(c) != int(bar_color)]
    dark_choices = order[:3]
    light_choices = order[-3:]
    dark = int(np.random.choice(dark_choices))
    light_pool = [c for c in light_choices if int(c) != dark]
    light = int(np.random.choice(light_pool))
    return [dark, light]


def _shape_from_piece_array(piece: np.ndarray) -> dict[str, object]:
    coords = np.argwhere(piece == 1)
    return _shape_from_coords(coords, (0, 0, int(piece.shape[0]) - 1, int(piece.shape[1]) - 1))


def _place_chain_ordered(
    grid: np.ndarray,
    placed_mask: np.ndarray,
    pieces: list[np.ndarray],
    color: int,
    side: str,
    bar_bbox: tuple[int, int, int, int],
) -> bool:
    H, W = grid.shape
    br0, bc0, br1, bc1 = map(int, bar_bbox)
    if side == "top":
        side_space = br0
        dims = [int(piece.shape[0]) for piece in pieces]
    elif side == "bottom":
        side_space = H - br1 - 1
        dims = [int(piece.shape[0]) for piece in pieces]
    elif side == "left":
        side_space = bc0
        dims = [int(piece.shape[1]) for piece in pieces]
    else:
        side_space = W - bc1 - 1
        dims = [int(piece.shape[1]) for piece in pieces]

    for _ in range(250):
        gaps: list[int] = []
        ok = True
        prev_gap = 0
        for i, dim in enumerate(dims):
            max_gap = int(side_space) - int(dim) - 1
            min_gap = 2 if i == 0 else prev_gap + 2
            reserve = 2 * (len(dims) - i - 1)
            hi = max_gap - reserve
            if hi < min_gap:
                ok = False
                break
            gap = int(np.random.randint(min_gap, hi + 1))
            gaps.append(gap)
            prev_gap = gap
        if not ok:
            continue

        temp_mask = placed_mask.copy()
        placements: list[tuple[np.ndarray, int, int]] = []
        for piece, gap in zip(pieces, gaps):
            ph, pw = piece.shape
            placed = False
            for _attempt in range(250):
                if side == "top":
                    y = br0 - gap - ph
                    x = int(np.random.randint(1, W - pw - 1))
                elif side == "bottom":
                    y = br1 + 1 + gap
                    x = int(np.random.randint(1, W - pw - 1))
                elif side == "left":
                    x = bc0 - gap - pw
                    y = int(np.random.randint(1, H - ph - 1))
                else:
                    x = bc1 + 1 + gap
                    y = int(np.random.randint(1, H - ph - 1))
                if _can_place_piece(temp_mask, piece, y, x):
                    for py in range(ph):
                        for px in range(pw):
                            if int(piece[py, px]) == 1:
                                temp_mask[y + py, x + px] = True
                    placements.append((piece, int(y), int(x)))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        for piece, y, x in placements:
            ph, pw = piece.shape
            for py in range(ph):
                for px in range(pw):
                    if int(piece[py, px]) == 1:
                        grid[y + py, x + px] = int(color)
                        placed_mask[y + py, x + px] = True
        return True
    return False


def _build_output_scene(seed: int) -> tuple[np.ndarray, list[list[np.ndarray]], list[int], list[str], tuple[int, int, int, int]] | None:
    np.random.seed(seed)
    grid_size = 30
    is_vertical = bool(np.random.choice([True, False]))
    bar_width = int(np.random.randint(5, 8))
    all_colors = list(range(1, 10))
    np.random.shuffle(all_colors)
    bar_color = int(all_colors[0])
    chain_colors = _choose_chain_colors(bar_color)

    protrusion_types = _generate_different_protrusion_types()
    chain_pieces: list[list[np.ndarray]] = []
    protrusions: list[np.ndarray] = []

    for rect_idx in range(2):
        rect_width = int(np.random.randint(4, 6))
        rect_length = int(np.random.randint(7, 10))
        num_pieces = int(np.random.randint(2, 4))
        pieces = _generate_split_pieces(rect_width, rect_length, num_pieces)
        if pieces is None:
            return None
        pieces[0], protrusion_shape = _add_protrusion_with_shape(pieces[0], protrusion_types[rect_idx])
        if pieces[0] is None or protrusion_shape is None:
            return None
        if any(not _validate_piece(piece, rect_width) for piece in pieces):
            return None
        chain_pieces.append(pieces)
        protrusions.append(protrusion_shape)

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    if is_vertical:
        temp = np.zeros((grid_size, grid_size), dtype=np.int8)
        bar_y_start = (grid_size - bar_width) // 2
        bar_y_end = bar_y_start + bar_width
        temp[bar_y_start:bar_y_end, :] = int(bar_color)
        hole_sides = ["bottom", "top"]
        if not _create_holes_horizontal_with_sides(temp, protrusions, bar_y_start, bar_y_end - 1, grid_size, hole_sides):
            return None
        grid = _rot90_cw(temp)
        rotated_hole_sides = ["right" if side == "top" else "left" for side in hole_sides]
        piece_rotations = ["cw90" if side == "left" else "ccw90" for side in rotated_hole_sides]
        sides_for_input = rotated_hole_sides
        bar_x_start = (grid_size - bar_width) // 2
        bar_x_end = bar_x_start + bar_width - 1
        bar_bbox = (0, bar_x_start, grid_size - 1, bar_x_end)
    else:
        bar_y_start = (grid_size - bar_width) // 2
        bar_y_end = bar_y_start + bar_width
        grid[bar_y_start:bar_y_end, :] = int(bar_color)
        hole_sides = ["top", "bottom"]
        if not _create_holes_horizontal_with_sides(grid, protrusions, bar_y_start, bar_y_end - 1, grid_size, hole_sides):
            return None
        piece_rotations = ["none" if side == "bottom" else "180" for side in hole_sides]
        sides_for_input = hole_sides
        bar_bbox = (bar_y_start, 0, bar_y_end - 1, grid_size - 1)

    for chain_idx, pieces in enumerate(chain_pieces):
        rotation = piece_rotations[chain_idx]
        for piece_idx in range(len(pieces)):
            if rotation == "cw90":
                pieces[piece_idx] = _rot90_cw(pieces[piece_idx])
            elif rotation == "ccw90":
                pieces[piece_idx] = _rot90_ccw(pieces[piece_idx])
            elif rotation == "180":
                pieces[piece_idx] = _rot180(pieces[piece_idx])
    bar_only = np.array(grid, dtype=int)
    comps = find_components(bar_only, bg=0, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -int(d["area"]))
    if not comps:
        return None
    bar = comps[0]
    bar_bbox = tuple(map(int, bar["bbox"]))
    br0, bc0, br1, bc1 = bar_bbox
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
    for r, c in bar_occ:
        bar_mask[int(r) - br0, int(c) - bc0] = True
    notches = _bar_notches(bar_mask, br0, bc0)
    side_order = ["top", "bottom"] if _orientation(bar_bbox) else ["left", "right"]

    out = np.zeros_like(bar_only, dtype=int)
    for r, c in bar_occ:
        out[r, c] = int(bar_color)
    occ = set(bar_occ)
    for pieces, color, side in zip(chain_pieces, chain_colors, sides_for_input):
        group = {
            "color": int(color),
            "side": side,
            "preferred_side": side,
            "ordered": [{"shape": _shape_from_piece_array(piece)} for piece in pieces],
        }
        out, occ = _apply_chain(out, occ, 0, bar_bbox, notches, group, side_order)
    return out.astype(int), chain_pieces, chain_colors, sides_for_input, bar_bbox


def _try_generate_input_grid(seed: int) -> np.ndarray | None:
    built = _build_output_scene(seed)
    if built is None:
        return None
    out, chain_pieces, chain_colors, sides_for_input, bar_bbox = built
    grid = np.zeros_like(out, dtype=np.int8)

    comps = find_components(out, bg=0, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -int(d["area"]))
    if not comps:
        return None
    bar = comps[0]
    bar_color = int(bar["color"])
    for r, c in bar["coords"]:
        grid[int(r), int(c)] = bar_color
    placed_mask = grid != 0

    for pieces, color, side in zip(chain_pieces, chain_colors, sides_for_input):
        if not _place_chain_ordered(grid, placed_mask, pieces, int(color), side, bar_bbox):
            return None

    fusion = inspect_fusion_features(grid.astype(int))
    if int(fusion["group_count"]) != len(chain_pieces):
        return None
    if not bool(fusion["brightness_side_order_ok"]) or not bool(fusion["monotone_gap_ok"]):
        return None
    if any(int(size) < 2 for size in fusion["same_color_group_sizes"]):
        return None
    if np.array_equal(grid, out):
        return None
    if not np.array_equal(main(grid.astype(int)), out.astype(int)):
        return None
    return grid.astype(int)


def generate_input() -> np.ndarray:
    # _build_output_scene(seed) reseeds numpy's global RNG for deterministic scene synthesis.
    # Isolate each candidate attempt so repeated generate_input() calls do not fall into
    # cross-call bad cycles that intermittently exhaust all 800 attempts.
    seed_rng = np.random.default_rng(int(np.random.randint(1, 2**31 - 1)))
    max_attempts = 1600
    for attempt in range(max_attempts):
        attempt_seed = int(seed_rng.integers(1, 2**31 - 1)) + attempt * 1009
        saved_state = np.random.get_state()
        try:
            grid = _try_generate_input_grid(attempt_seed)
        finally:
            np.random.set_state(saved_state)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {max_attempts} attempts")


def inspect_fusion_features(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -int(d["area"]))
    bar = comps[0]
    bar_bbox = tuple(map(int, bar["bbox"]))
    is_horizontal = _orientation(bar_bbox)
    side_order = ["top", "bottom"] if is_horizontal else ["left", "right"]
    groups: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for comp in comps[1:]:
        groups[int(comp["color"])].append(tuple(map(int, comp["bbox"])))

    data = []
    for color, bboxes in sorted(groups.items()):
        sides = [_piece_side(bbox, bar_bbox) for bbox in bboxes]
        gaps = [_gap_from_bar(bbox, bar_bbox, sides[i]) for i, bbox in enumerate(bboxes)]
        data.append(
            {
                "color": int(color),
                "brightness": float(_brightness(color)),
                "sides": sides,
                "gaps": sorted(int(g) for g in gaps),
            }
        )
    data.sort(key=lambda item: item["brightness"])
    brightness_side_order_ok = len(data) == 2 and all(
        len(set(item["sides"])) == 1 and item["sides"][0] == side_order[idx] for idx, item in enumerate(data)
    )
    same_color_group_sizes = [len(item["gaps"]) for item in data]
    monotone_gap_ok = all(item["gaps"] == sorted(item["gaps"]) and len(set(item["gaps"])) == len(item["gaps"]) for item in data)
    return {
        "group_count": len(data),
        "same_color_group_sizes": same_color_group_sizes,
        "brightness_side_order_ok": bool(brightness_side_order_ok),
        "monotone_gap_ok": bool(monotone_gap_ok),
        "groups": data,
    }
