# Combo variant of 16b78196 (index 22)
# combo_spec: D6/2+D9/2
# Name: 源位置杆色填充+短杆
#
# Design analysis for this fused variant:
# - Concrete behavior:
#   The original task assembles scattered colored sprites into notch holes of a central bar.
#   Here the bar is short (about 50-70% of the full span), and after assembly the source
#   locations of every scattered sprite are recolored with the bar color. So the output has
#   bar-colored cells in two roles: the real connected bar, and detached source-location marks.
# - Small example:
#   In the base task, a top-side sprite scattered above a horizontal bar disappears from its
#   source location and reappears inside a notch-chain on the bar. In this fused version the
#   same sprite reappears in the chain, but its exact source pixels are also repainted with the
#   bar color. If that source sprite sat left of the short bar's endpoint, the output now shows
#   an isolated bar-colored fragment outside the bar span.
# - Complexity calibration:
#   The detached bar-colored fragments are deliberately repeated across multiple pieces, and the
#   generator forces at least one non-root piece outside the bar's endpoint projection so the
#   D9/2 effect is visible. At the same time the number of chains stays small (2 chains with
#   2-3 pieces each) so the search space remains tractable.
# - Implicit assumptions checked against the original solver:
#   1. "The largest monochrome component is the bar" still holds in the input, because source
#      filling only appears in the output and piece colors never equal the bar color.
#   2. "The bar spans the full grid" is removed. The solver only really needs the bar's own bbox
#      and holes opening to that bbox, so the short-bar geometry is safe.
#   3. "All bar-colored output cells belong to the true bar" is no longer valid. To avoid mixing
#      bar structure with source markers during cavity search and backtracking, the solver first
#      reconstructs only the true assembly, then adds source-location bar-color markers as a
#      final post-process.
# - Placement legality:
#   The generator keeps every scattered piece non-overlapping, inside bounds, and separated from
#   the bar and from each other by at least one cell in the 8-neighborhood.
# - Visual observability:
#   Human-solvable structure comes from repeated cues: the real bar is the only large connected
#   bar-colored object in the input, while every output preserves exact source sprite silhouettes
#   as detached bar-colored marks. Because some of those marks lie beyond the short bar ends, the
#   "bar color means both bar and source trace" rule is visually obvious across samples.

from __future__ import annotations

from itertools import combinations

import numpy as np

from arc_utils import components_mask, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble colored sprites into notch chains on the bar, then repaint each source sprite
    location with the bar color.

    The search phase intentionally ignores the future source-location markers; otherwise the
    detached bar-colored fragments would pollute cavity detection and bar restoration.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        return {"rel": rel, "rel_set": set(rel), "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append(
            {
                "color": int(p["color"]),
                "shape": shape_from_coords(p["coords"], p["bbox"]),
                "orig": (r0, c0),
                "orig_coords": {(int(r), int(c)) for r, c in p["coords"]},
            }
        )

    def bar_notches(mask: np.ndarray, top: int, left: int):
        empty = ~mask
        h, w = empty.shape
        notches = []
        for pts in components_mask(empty, connectivity=4):
            rr = pts[:, 0]
            cc = pts[:, 1]
            top_n = int((rr == 0).sum())
            bottom_n = int((rr == h - 1).sum())
            left_n = int((cc == 0).sum())
            right_n = int((cc == w - 1).sum())
            if top_n + bottom_n + left_n + right_n == 0:
                continue
            hole_gl = {(top + int(r), left + int(c)) for r, c in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top_n), ("bottom", bottom_n), ("left", left_n), ("right", right_n)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(v, top, left):
        return {(top + dr, left + dc) for dr, dc in v["rel"]}

    def placement_pts(v, top, left, occ, must_cover=None, strip=None):
        if top < 0 or left < 0 or top + v["h"] > H or left + v["w"] > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or v["w"] != B - A + 1:
                    return None
            else:
                if top != A or v["h"] != B - A + 1:
                    return None
        pts = pixels(v, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis, A, cover, v):
        if axis == "cols":
            left = A
            tops = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if rN == top + lr:
                        lefts.add(cN - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, v):
        pos = set()
        for rH, cH in hole:
            for lr, lc in v["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    root_cands = []
    for ni, n in enumerate(notches):
        axis = n["axis"]
        for pi, p in enumerate(pieces):
            v = p["shape"]
            for top, left in hole_positions(n["hole"], v):
                pts = placement_pts(v, top, left, bar_occ, must_cover=n["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + v["w"] - 1)
                else:
                    A, B = int(top), int(top + v["h"] - 1)
                mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(n["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": v},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:20]]

    best_out = out.copy()
    best_key = None

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def gen_extend_moves(chain, remaining_set):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = pixels(head["shape"], head["top"], head["left"])
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])

            cavity = {
                (r, c)
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if out_grid[r, c] == bg
            }
            if not cavity:
                return []

            moves = []
            for pi in remaining_set:
                v = pieces[pi]["shape"]
                for top, left in aligned_positions(axis, A, cavity, v):
                    pts = placement_pts(v, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    moves.append((pi, top, left, pts, v))
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, v in moves[:8]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": v}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[r, c] = bar_color

            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = pixels(root["shape"], root["top"], root["left"])
                if pts & occ0:
                    ok = False
                    break
                used_p.add(root["pi"])
                used_n.add(root["ni"])
                for r, c in pts:
                    out0[r, c] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                chains.append({"axis": root["axis"], "A": root["A"], "B": root["B"], "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]}})

            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    final = best_out.copy()
    for p in pieces:
        for r, c in p["orig_coords"]:
            final[r, c] = bar_color
    return final


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
    if np.sum(piece) == 0:
        return False
    h, w = piece.shape
    seen = np.zeros_like(piece, dtype=bool)
    start = None
    for r in range(h):
        for c in range(w):
            if piece[r, c] == 1:
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
        if seen[r, c] or piece[r, c] == 0:
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
    return int(np.sum(piece)) == int((rmax - rmin + 1) * (cmax - cmin + 1))


def _validate_piece(piece: np.ndarray, original_width: int) -> bool:
    if piece is None:
        return False
    area = int(np.sum(piece))
    if area < original_width + 1 or area > 15:
        return False
    if not _is_connected_piece(piece):
        return False
    if _is_rectangular_piece(piece):
        return False
    return True


def _holes_too_close_1d(start_a: int, size_a: int, start_b: int, size_b: int, min_gap: int) -> bool:
    end_a = start_a + size_a - 1
    end_b = start_b + size_b - 1
    if end_a < start_b:
        gap = start_b - end_a - 1
    elif end_b < start_a:
        gap = start_a - end_b - 1
    else:
        gap = -1
    return gap < min_gap


def _is_similar_to_protrusions(h: int, w: int, protrusions: list[np.ndarray]) -> bool:
    for prot in protrusions:
        ph, pw = prot.shape
        if h == ph and w == pw:
            return True
    return False


def _generate_different_protrusion_types() -> list[tuple[int, int, int]]:
    possible_types = [(1, 2, 2), (1, 3, 3), (2, 2, 1), (2, 3, 1), (2, 3, 2)]
    indices = np.random.choice(len(possible_types), size=2, replace=False)
    return [possible_types[int(i)] for i in indices]


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
        start = max(0, start)
        if end > start:
            new_piece[row, start:end] = 1
        shape_start = max(0, start - base_start)
        shape_end = min(prot_bottom_w, end - base_start)
        if shape_end > shape_start:
            protrusion_shape[row, shape_start:shape_end] = 1

    return _trim_piece(new_piece), protrusion_shape


def _create_matching_hole_horizontal_local(grid: np.ndarray, protrusion: np.ndarray, edge_y: int, x_start: int, direction: str) -> set[tuple[int, int]]:
    h, w = protrusion.shape
    coords: set[tuple[int, int]] = set()
    if direction == "top":
        for py in range(h):
            for px in range(w):
                if protrusion[py, px] != 1:
                    continue
                y = edge_y + (h - 1 - py)
                x = x_start + px
                grid[y, x] = 0
                coords.add((int(y), int(x)))
    else:
        for py in range(h):
            for px in range(w):
                if protrusion[py, px] != 1:
                    continue
                y = edge_y - (h - 1 - py)
                x = x_start + px
                grid[y, x] = 0
                coords.add((int(y), int(x)))
    return coords


def _add_distractor_holes_horizontal_local(
    grid: np.ndarray,
    top_y: int,
    bottom_y: int,
    existing_holes: list[tuple[str, int, int]],
    protrusions: list[np.ndarray],
) -> None:
    min_hole_gap = 3
    span_len = grid.shape[1]
    num_distractors = int(np.random.randint(2, 5))
    for _ in range(num_distractors):
        side = str(np.random.choice(["top", "bottom"]))
        dist_h = int(np.random.randint(1, 3))
        dist_w = int(np.random.randint(1, 4))
        while _is_similar_to_protrusions(dist_h, dist_w, protrusions):
            dist_h = int(np.random.randint(1, 3))
            dist_w = int(np.random.randint(1, 4))
        if span_len - dist_w - 4 <= 2:
            return
        attempts = 0
        while attempts < 50:
            x = int(np.random.randint(2, span_len - dist_w - 2))
            valid = True
            for hole_side, hole_x, hole_w in existing_holes:
                if hole_side == side and _holes_too_close_1d(x, dist_w, hole_x, hole_w, min_hole_gap):
                    valid = False
                    break
            if valid:
                if side == "top":
                    for py in range(dist_h):
                        for px in range(dist_w):
                            if top_y + py < bottom_y:
                                grid[top_y + py, x + px] = 0
                else:
                    for py in range(dist_h):
                        for px in range(dist_w):
                            if bottom_y - py > top_y:
                                grid[bottom_y - py, x + px] = 0
                existing_holes.append((side, x, dist_w))
                break
            attempts += 1


def _create_holes_horizontal_local_with_sides(
    grid: np.ndarray,
    protrusions: list[np.ndarray],
    top_y: int,
    bottom_y: int,
    hole_sides: list[str],
) -> list[dict]:
    min_hole_gap = 3
    span_len = grid.shape[1]
    hole_positions: list[tuple[str, int, int]] = []
    used_holes: list[dict] = []
    for prot, side in zip(protrusions, hole_sides):
        prot_for_hole = _rot180(np.flipud(prot)) if side == "top" else prot
        _h, w = prot_for_hole.shape
        if span_len - w - 6 <= 3:
            raise ValueError("bar too short for used hole placement")
        x_pos = int(np.random.randint(3, span_len - w - 3))
        attempts = 0
        while attempts < 50:
            valid = True
            for existing_side, existing_x, existing_w in hole_positions:
                if existing_side == side and _holes_too_close_1d(x_pos, w, existing_x, existing_w, min_hole_gap):
                    valid = False
                    break
            if valid:
                break
            x_pos = int(np.random.randint(3, span_len - w - 3))
            attempts += 1
        coords = _create_matching_hole_horizontal_local(grid, prot_for_hole, top_y if side == "top" else bottom_y, x_pos, direction=side)
        hole_positions.append((side, x_pos, w))
        used_holes.append({"side": side, "x": int(x_pos), "w": int(w), "coords": coords})

    _add_distractor_holes_horizontal_local(grid, top_y, bottom_y, hole_positions, protrusions)
    return used_holes


def _generate_irregular_split_line_constrained(width: int, base_y: int) -> np.ndarray:
    split_line = np.full(width, base_y, dtype=np.int8)
    num_features = int(np.random.randint(1, min(3, width - 1)))
    available_positions = list(range(1, width - 1))
    feature_positions: list[int] = []
    for _ in range(num_features):
        if not available_positions:
            break
        pos = int(np.random.choice(available_positions))
        feature_positions.append(pos)
        available_positions = [p for p in available_positions if abs(p - pos) > 1]
    for x_pos in feature_positions:
        split_line[x_pos] = base_y + int(np.random.choice([-1, 1]))
    return split_line


def _extract_pieces_from_splits_constrained(width: int, split_lines: list[np.ndarray], length: int, max_height: int) -> list[np.ndarray] | None:
    pieces: list[np.ndarray] = []
    all_lines = [np.zeros(width, dtype=np.int8)] + split_lines + [np.full(width, length, dtype=np.int8)]
    for i in range(len(all_lines) - 1):
        top_line = all_lines[i]
        bottom_line = all_lines[i + 1]
        full_row_start = int(np.max(top_line))
        full_row_end = int(np.min(bottom_line))
        if full_row_end - full_row_start < 1:
            return None

        min_y = max(0, int(np.min(top_line)))
        max_y = min(length, int(np.max(bottom_line)))
        piece_h = max_y - min_y
        if piece_h > max_height:
            return None
        piece = np.zeros((piece_h, width), dtype=np.int8)
        for x in range(width):
            y_start = int(top_line[x]) - min_y
            y_end = int(bottom_line[x]) - min_y
            for y in range(max(0, y_start), min(piece_h, y_end)):
                piece[y, x] = 1

        trimmed = _trim_piece(piece)
        if trimmed is None or np.sum(trimmed) == 0 or trimmed.shape[0] > max_height:
            return None
        pieces.append(trimmed)
    return pieces if len(pieces) >= 2 else None


def _generate_split_pieces(width: int, length: int, num_pieces: int) -> list[np.ndarray] | None:
    max_piece_height = 5
    min_piece_base = 1
    max_piece_base = 2
    if length < num_pieces * min_piece_base:
        return None

    split_positions = []
    remaining = length
    current_y = 0
    for i in range(num_pieces - 1):
        pieces_left = num_pieces - i
        reserved = (pieces_left - 1) * min_piece_base
        available = remaining - reserved
        max_for_this = min(max_piece_base, available)
        if max_for_this < min_piece_base:
            return None
        piece_height = int(np.random.randint(min_piece_base, max_for_this + 1))
        split_y = current_y + piece_height
        split_positions.append(split_y)
        current_y = split_y
        remaining = length - current_y

    split_lines = [_generate_irregular_split_line_constrained(width, split_y) for split_y in split_positions]
    pieces = _extract_pieces_from_splits_constrained(width, split_lines, length, max_piece_height)
    if pieces is None:
        return None
    for piece in pieces:
        full_rows = sum(1 for row in range(piece.shape[0]) if int(np.sum(piece[row, :])) == int(piece.shape[1]))
        if full_rows < 1 or full_rows > 2:
            return None
    return pieces


def _can_place_piece_with_gap(placed_mask: np.ndarray, piece: np.ndarray, y: int, x: int) -> bool:
    gh, gw = placed_mask.shape
    ph, pw = piece.shape
    for py in range(ph):
        for px in range(pw):
            if piece[py, px] != 1:
                continue
            gy, gx = y + py, x + px
            if not (0 <= gy < gh and 0 <= gx < gw):
                return False
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < gh and 0 <= nx < gw and placed_mask[ny, nx]:
                        return False
    return True


def _place_piece_at(grid: np.ndarray, placed_mask: np.ndarray, piece: np.ndarray, color: int, y: int, x: int) -> None:
    ph, pw = piece.shape
    for py in range(ph):
        for px in range(pw):
            if piece[py, px] == 1:
                grid[y + py, x + px] = int(color)
                placed_mask[y + py, x + px] = True


def _place_piece_in_regions(
    grid: np.ndarray,
    placed_mask: np.ndarray,
    piece: np.ndarray,
    color: int,
    regions: list[tuple[int, int, int, int]],
    *,
    attempts_per_region: int = 250,
) -> tuple[int, int] | None:
    ph, pw = piece.shape
    valid_regions = []
    for r0, c0, r1, c1 in regions:
        if r0 <= r1 and c0 <= c1 and r1 - r0 + 1 >= ph and c1 - c0 + 1 >= pw:
            valid_regions.append((int(r0), int(c0), int(r1), int(c1)))
    if not valid_regions:
        return None

    order = list(np.random.permutation(len(valid_regions)))
    for idx in order:
        r0, c0, r1, c1 = valid_regions[idx]
        max_y = r1 - ph + 1
        max_x = c1 - pw + 1
        for _ in range(attempts_per_region):
            y = int(np.random.randint(r0, max_y + 1))
            x = int(np.random.randint(c0, max_x + 1))
            if _can_place_piece_with_gap(placed_mask, piece, y, x):
                _place_piece_at(grid, placed_mask, piece, color, y, x)
                return y, x
        for y in range(r0, max_y + 1):
            for x in range(c0, max_x + 1):
                if _can_place_piece_with_gap(placed_mask, piece, y, x):
                    _place_piece_at(grid, placed_mask, piece, color, y, x)
                    return y, x
    return None


def _measure_fusion_behavior(inp: np.ndarray, out: np.ndarray) -> dict[str, int | str]:
    bg = get_bg(inp)
    comps = find_components(inp, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    bar = comps[0]
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_color = int(bar["color"])
    r0, c0, r1, c1 = map(int, bar["bbox"])
    orientation = "horizontal" if (c1 - c0) >= (r1 - r0) else "vertical"

    source_cells = {(int(r), int(c)) for r, c in np.argwhere(out == bar_color) if (int(r), int(c)) not in bar_occ}
    if orientation == "horizontal":
        outside = {(r, c) for r, c in source_cells if c < c0 or c > c1}
    else:
        outside = {(r, c) for r, c in source_cells if r < r0 or r > r1}
    return {
        "orientation": orientation,
        "bar_span": int((c1 - c0 + 1) if orientation == "horizontal" else (r1 - r0 + 1)),
        "source_cells": int(len(source_cells)),
        "outside_source_cells": int(len(outside)),
    }


def _rotate_local_patch_and_coords(
    patch: np.ndarray, coords_sets: list[set[tuple[int, int]]]
) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    h, w = patch.shape
    rotated = _rot90_cw(patch)
    rotated_sets: list[set[tuple[int, int]]] = []
    for coords in coords_sets:
        mapped = {(int(c), int(h - 1 - r)) for r, c in coords}
        rotated_sets.append(mapped)
    return rotated, rotated_sets


def _try_generate_case(seed: int) -> tuple[np.ndarray, dict[str, int | str]] | None:
    np.random.seed(seed)
    grid_size = 30
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)

    is_vertical = bool(np.random.choice([True, False]))
    bar_width = int(np.random.randint(6, 9))
    span_len = int(np.random.randint(15, 22))

    all_colors = list(range(1, 10))
    np.random.shuffle(all_colors)
    bar_color = int(all_colors[0])
    piece_colors = all_colors[1:]

    protrusion_types = _generate_different_protrusion_types()
    all_pieces: list[np.ndarray] = []
    all_protrusions: list[np.ndarray] = []
    rect_piece_ranges: list[tuple[int, int]] = []
    piece_count = 0

    for rect_idx in range(2):
        rect_width = int(np.random.randint(4, 6))
        rect_length = int(np.random.randint(5, 8))
        num_pieces = int(np.random.randint(2, 4))
        pieces = _generate_split_pieces(rect_width, rect_length, num_pieces)
        if pieces is None:
            return None

        pieces[0], protrusion_shape = _add_protrusion_with_shape(pieces[0], protrusion_types[rect_idx])
        if pieces[0] is None or protrusion_shape is None:
            return None
        if any(not _validate_piece(piece, rect_width) for piece in pieces):
            return None

        rect_piece_ranges.append((piece_count, piece_count + len(pieces)))
        all_pieces.extend(pieces)
        all_protrusions.append(protrusion_shape)
        piece_count += len(pieces)

    if len(all_pieces) > len(piece_colors):
        return None
    colors = piece_colors[: len(all_pieces)]

    hole_sides = [str(np.random.choice(["top", "bottom"])) for _ in range(2)]
    bar_local = np.full((bar_width, span_len), bar_color, dtype=np.int8)
    used_holes = _create_holes_horizontal_local_with_sides(bar_local, all_protrusions, 0, bar_width - 1, hole_sides)
    local_hole_coords = [h["coords"] for h in used_holes]

    if is_vertical:
        bar_patch, rotated_holes = _rotate_local_patch_and_coords(bar_local, local_hole_coords)
        for rect_idx, (start_idx, end_idx) in enumerate(rect_piece_ranges):
            out_side = "right" if hole_sides[rect_idx] == "top" else "left"
            for piece_idx in range(start_idx, end_idx):
                if out_side == "left":
                    all_pieces[piece_idx] = _rot90_cw(all_pieces[piece_idx])
                else:
                    all_pieces[piece_idx] = _rot90_ccw(all_pieces[piece_idx])
        chain_sides = ["right" if side == "top" else "left" for side in hole_sides]
        bar_r0 = (grid_size - span_len) // 2
        bar_c0 = (grid_size - bar_width) // 2
        for idx, coords in enumerate(rotated_holes):
            used_holes[idx]["coords"] = {(bar_r0 + r, bar_c0 + c) for r, c in coords}
        grid[bar_r0 : bar_r0 + bar_patch.shape[0], bar_c0 : bar_c0 + bar_patch.shape[1]] = bar_patch
        bar_r1 = bar_r0 + bar_patch.shape[0] - 1
        bar_c1 = bar_c0 + bar_patch.shape[1] - 1
    else:
        for rect_idx, (start_idx, end_idx) in enumerate(rect_piece_ranges):
            if hole_sides[rect_idx] == "top":
                for piece_idx in range(start_idx, end_idx):
                    all_pieces[piece_idx] = _rot180(all_pieces[piece_idx])
        chain_sides = list(hole_sides)
        bar_r0 = (grid_size - bar_width) // 2
        bar_c0 = (grid_size - span_len) // 2
        for idx, coords in enumerate(local_hole_coords):
            used_holes[idx]["coords"] = {(bar_r0 + r, bar_c0 + c) for r, c in coords}
        grid[bar_r0 : bar_r0 + bar_local.shape[0], bar_c0 : bar_c0 + bar_local.shape[1]] = bar_local
        bar_r1 = bar_r0 + bar_local.shape[0] - 1
        bar_c1 = bar_c0 + bar_local.shape[1] - 1

    def base_region(side: str) -> tuple[int, int, int, int] | None:
        if side == "top":
            r0, r1 = 0, bar_r0 - 2
            c0, c1 = 0, grid_size - 1
        elif side == "bottom":
            r0, r1 = bar_r1 + 2, grid_size - 1
            c0, c1 = 0, grid_size - 1
        elif side == "left":
            r0, r1 = 0, grid_size - 1
            c0, c1 = 0, bar_c0 - 2
        else:
            r0, r1 = 0, grid_size - 1
            c0, c1 = bar_c1 + 2, grid_size - 1
        if r0 > r1 or c0 > c1:
            return None
        return (r0, c0, r1, c1)

    def near_region(side: str) -> tuple[int, int, int, int] | None:
        reg = base_region(side)
        if reg is None:
            return None
        r0, c0, r1, c1 = reg
        if side in ("top", "bottom"):
            c0 = max(c0, bar_c0 - 3)
            c1 = min(c1, bar_c1 + 3)
        else:
            r0 = max(r0, bar_r0 - 3)
            r1 = min(r1, bar_r1 + 3)
        if r0 > r1 or c0 > c1:
            return None
        return (r0, c0, r1, c1)

    def far_regions(side: str) -> list[tuple[int, int, int, int]]:
        regs = []
        reg = base_region(side)
        if reg is None:
            return regs
        r0, c0, r1, c1 = reg
        if side in ("top", "bottom"):
            if c0 <= bar_c0 - 2:
                regs.append((r0, c0, r1, bar_c0 - 2))
            if bar_c1 + 2 <= c1:
                regs.append((r0, bar_c1 + 2, r1, c1))
        else:
            if r0 <= bar_r0 - 2:
                regs.append((r0, c0, bar_r0 - 2, c1))
            if bar_r1 + 2 <= r1:
                regs.append((bar_r1 + 2, c0, r1, c1))
        return regs

    placed_mask = grid != 0
    far_success = False

    for rect_idx, (start_idx, end_idx) in enumerate(rect_piece_ranges):
        side = chain_sides[rect_idx]
        indices = list(range(start_idx, end_idx))
        if len(indices) < 2:
            return None

        root_idx = indices[0]
        tail_idx = indices[-1]
        middle_indices = indices[1:-1]

        root_regions = [r for r in [near_region(side), base_region(side)] if r is not None]
        pos = _place_piece_in_regions(grid, placed_mask, all_pieces[root_idx], int(colors[root_idx]), root_regions)
        if pos is None:
            return None

        far_pos = _place_piece_in_regions(grid, placed_mask, all_pieces[tail_idx], int(colors[tail_idx]), far_regions(side))
        if far_pos is None:
            fallback_regions = [r for r in [base_region(side)] if r is not None]
            far_pos = _place_piece_in_regions(grid, placed_mask, all_pieces[tail_idx], int(colors[tail_idx]), fallback_regions)
        else:
            far_success = True
        if far_pos is None:
            return None

        for idx in middle_indices:
            pos = _place_piece_in_regions(grid, placed_mask, all_pieces[idx], int(colors[idx]), [r for r in [base_region(side)] if r is not None])
            if pos is None:
                return None

    if not far_success:
        return None

    inp = grid.astype(int)
    out = main(inp)
    metrics = _measure_fusion_behavior(inp, out)
    span_frac = float(metrics["bar_span"]) / grid_size
    if not (0.5 <= span_frac <= 0.7):
        return None
    if int(metrics["outside_source_cells"]) < 4:
        return None
    if int(metrics["source_cells"]) < 16:
        return None
    return inp, metrics


def generate_input() -> np.ndarray:
    for attempt in range(400):
        seed = attempt * 1000 + int(np.random.randint(1, 10_000))
        case = _try_generate_case(seed)
        if case is not None:
            return case[0]
    raise RuntimeError("generate_input: failed after 400 attempts")
