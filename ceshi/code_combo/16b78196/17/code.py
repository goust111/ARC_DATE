# Combo variant of 16b78196 (index 17)
# combo_spec: D5/2+D16/1
# Name: 渐变着色+同链同色
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - Base geometry is unchanged: a central bar has true notches, root pieces plug those notches, and later
#   pieces keep the same strip width while filling the cavity inside the previous piece bbox.
# - D16/1 changes the input evidence: pieces from the same chain now share one input color, while pieces
#   from different chains use different colors. Color therefore indicates chain membership, not depth.
# - D5/2 changes the output rendering: once a chain is assembled, depth 0 keeps the chain color, depth 1
#   is recolored to the next color in a visible legend, and depth 2 to the second-next color. Example:
#   if the legend is [red, teal, yellow, green, blue, pink] and a chain's input color is teal, then its
#   assembled colors become [teal, yellow, green].
# - Small trace:
#   original: pieces A(red), B(green), C(blue) are independent colors; placement writes the same colors.
#   combo: chain-1 uses A/B both red in input, chain-2 uses C/D both teal in input; solver groups by
#   color first, then determines order inside each color-group by notch fitting and cavity filling, and
#   finally writes [red, next(red)] and [teal, next(teal)] in output.
#
# b) Complexity calibration
# - The chain geometry still carries the hard reasoning. Same-color grouping only tells "which chain",
#   not "what order". Order is still recovered from shape/cavity compatibility.
# - Each sample uses exactly 2 chains of length 2-3, so the same-color clue repeats and the gradient rule
#   repeats. This avoids degenerate one-off transformations.
# - A visible 6-color legend is kept from D5/2, because raw ARC color ids are not human-visible order.
#
# c) Implicit-assumption audit of the original main()
# - Original assumption `placed_color == piece.color` is broken. Output color now depends on both chain
#   color and depth, so the search state tracks chain depth and applies `legend_shift(chain_color, depth)`.
# - Original assumption "all remaining pieces are interchangeable except shape" is broken. In this combo,
#   a chain may only consume pieces from its own input-color group. Search is therefore color-partitioned.
# - Original assumption "every non-bar component is a movable piece" is broken because the legend adds tiny
#   guide components. The solver filters movable pieces by the original area band 8..13.
# - Undo/restoration must preserve both bar cells and legend cells, because both are immutable obstacles.
#
# d) Placement legality
# - Generation is still output-first. The bar, used notches, chain strips, and decoy notches are built in
#   the assembled output, then pieces are scattered into the input with no overlap and no 4-neighbor touch.
# - Legend cells and a small buffer remain blocked during scattering so guides stay visually separate.
# - If a chain grows on the same side as the legend, its strip is prevented from overlapping the legend span.
#
# e) Visual observability
# - Human-visible cues are:
#   1. same input color => same chain,
#   2. within that chain, the output colors march forward along the visible legend.
# - No hidden ordering by numeric color id is used; the only color order is the spatial order of the legend.
#
# f) Visual cleanliness
# - Guide dominoes have area 2; movable sprites stay in area 8..13. This prevents guide/piece confusion.
# - Chains are kept to opposite sides of the bar, so the assembled result reads as two separate structures.
#
# g) Change inferability
# - Every generated sample has two monochrome input groups of size 2-3.
# - Every chain changes color in output for at least one non-root piece, so the fusion does not collapse
#   back to the original task where output color equals input color everywhere.
#
# 9) New assumptions introduced by the combo and how the generator guarantees them
# - Assumption: each movable input color corresponds to exactly one chain.
#   Guarantee: generator uses exactly two chain base colors and reuses each only within one chain.
# - Assumption: each chain base color can be shifted far enough for its full length.
#   Guarantee: base colors are assigned from legend slots with room for depth 0..len(chain)-1.
# - Assumption: color groups are not merged in input.
#   Guarantee: scattered pieces never 4-touch each other, even when they share a color.

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, find_components, get_bg

D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEGEND_LEN = 6
PIECE_MIN_AREA = 8
PIECE_MAX_AREA = 13


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _pixels(shape, top, left):
    return {(top + lr, left + lc) for lr, lc in shape["rel"]}


def _connected(cells):
    if len(cells) <= 1:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in D4:
            nb = (r + dr, c + dc)
            if nb in cells_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells_set)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [
                nb
                for r, c in cells
                for dr, dc in D4
                if (nb := (r + dr, c + dc)) in available and nb not in cells
            ]
            if not frontier:
                break
            if must_span_axis is not None and span_range is not None:
                A, B = span_range
                if must_span_axis == "cols":
                    cmin = min((c for _, c in cells), default=B)
                    cmax = max((c for _, c in cells), default=A)
                    if cmin > A:
                        cand = [nb for nb in frontier if nb[1] < cmin]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cmax < B:
                        cand = [nb for nb in frontier if nb[1] > cmax]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                else:
                    rmin = min((r for r, _ in cells), default=B)
                    rmax = max((r for r, _ in cells), default=A)
                    if rmin > A:
                        cand = [nb for nb in frontier if nb[0] < rmin]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if rmax < B:
                        cand = [nb for nb in frontier if nb[0] > rmax]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
            cells.add(rng.choice(frontier))
        if len(cells) < target:
            continue
        while len(cells) > target:
            removable = []
            for rc in cells:
                if rc in seeds:
                    continue
                if must_span_axis is not None and span_range is not None:
                    A, B = span_range
                    if must_span_axis == "cols":
                        if rc[1] == A and sum(1 for _, c2 in cells if c2 == A) <= 1:
                            continue
                        if rc[1] == B and sum(1 for _, c2 in cells if c2 == B) <= 1:
                            continue
                    else:
                        if rc[0] == A and sum(1 for r2, _ in cells if r2 == A) <= 1:
                            continue
                        if rc[0] == B and sum(1 for r2, _ in cells if r2 == B) <= 1:
                            continue
                if _connected(cells - {rc}):
                    removable.append(rc)
            if not removable:
                break
            cells.remove(rng.choice(removable))
        if len(cells) != target:
            continue
        if must_span_axis is not None and span_range is not None:
            A, B = span_range
            if must_span_axis == "cols":
                if min(c for _, c in cells) != A or max(c for _, c in cells) != B:
                    continue
            else:
                if min(r for r, _ in cells) != A or max(r for r, _ in cells) != B:
                    continue
        if _connected(cells):
            return cells
    return None


def _legend_components(comps):
    guides = [c for c in comps if int(c["area"]) <= 2]
    if len(guides) < 3:
        return guides, []
    centers = []
    for guide in guides:
        r0, c0, r1, c1 = map(int, guide["bbox"])
        centers.append(((r0 + r1) / 2.0, (c0 + c1) / 2.0))
    row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
    col_span = max(c for _, c in centers) - min(c for _, c in centers)
    if col_span >= row_span:
        guides.sort(key=lambda d: (int(d["bbox"][1]), int(d["bbox"][0])))
    else:
        guides.sort(key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1])))
    palette = [int(g["color"]) for g in guides]
    return guides, palette


def _legend_layout(H, W, orientation):
    if orientation == "horizontal":
        row = 1
        dominoes = []
        for i in range(LEGEND_LEN):
            c0 = 1 + 3 * i
            dominoes.append({(row, c0), (row, c0 + 1)})
        r0, c0, r1, c1 = 0, 0, row + 1, 1 + 3 * (LEGEND_LEN - 1) + 2
        side = "top"
        span = (1, 1 + 3 * (LEGEND_LEN - 1) + 1)
    else:
        col = 1
        dominoes = []
        for i in range(LEGEND_LEN):
            r0d = 1 + 3 * i
            dominoes.append({(r0d, col), (r0d + 1, col)})
        r0, c0, r1, c1 = 0, 0, 1 + 3 * (LEGEND_LEN - 1) + 2, col + 1
        side = "left"
        span = (1, 1 + 3 * (LEGEND_LEN - 1) + 1)
    occ = set().union(*dominoes)
    buffer = {
        (r, c)
        for r in range(max(0, r0), min(H - 1, r1) + 1)
        for c in range(max(0, c0), min(W - 1, c1) + 1)
    }
    return {"dominoes": dominoes, "occ": occ, "buffer": buffer, "side": side, "span": span}


def _intervals_overlap(a0, a1, b0, b1):
    return not (a1 < b0 or b1 < a0)


def _assign_chain_bases(rng, chain_lens, palette_len):
    order = sorted(range(len(chain_lens)), key=lambda i: -int(chain_lens[i]))
    used_inputs = set()
    used_outputs = set()
    base_idx = {}

    def backtrack(k):
        if k == len(order):
            return True
        ci = order[k]
        need = int(chain_lens[ci])
        candidates = [
            b
            for b in range(palette_len - need + 1)
            if b not in used_inputs and all((b + d) not in used_outputs for d in range(need))
        ]
        rng.shuffle(candidates)
        for b in candidates:
            used_inputs.add(b)
            for d in range(need):
                used_outputs.add(b + d)
            base_idx[ci] = b
            if backtrack(k + 1):
                return True
            used_inputs.remove(b)
            for d in range(need):
                used_outputs.remove(b + d)
            del base_idx[ci]
        return False

    if not backtrack(0):
        return None
    return [base_idx[i] for i in range(len(chain_lens))]


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
    expected_area = int((rmax - rmin + 1) * (cmax - cmin + 1))
    return expected_area == int(np.sum(piece))


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


def _can_place_piece(placed_mask: np.ndarray, piece: np.ndarray, y: int, x: int) -> bool:
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
        end = start + row_width
        start = max(0, start)
        end = min(w, end)
        if end > start:
            new_piece[row, start:end] = 1
        shape_start = max(0, start - base_start)
        shape_end = min(prot_bottom_w, end - base_start)
        if shape_end > shape_start:
            protrusion_shape[row, shape_start:shape_end] = 1
    return _trim_piece(new_piece), protrusion_shape


def _create_matching_hole_horizontal(grid: np.ndarray, protrusion: np.ndarray, edge_y: int, x_start: int, direction: str) -> None:
    h, w = protrusion.shape
    if direction == "top":
        for py in range(h):
            for px in range(w):
                if protrusion[py, px] == 1:
                    y = edge_y + (h - 1 - py)
                    x = x_start + px
                    grid[y, x] = 0
    else:
        for py in range(h):
            for px in range(w):
                if protrusion[py, px] == 1:
                    y = edge_y - (h - 1 - py)
                    x = x_start + px
                    grid[y, x] = 0


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
    for _ in range(num_distractors):
        side = str(np.random.choice(["top", "bottom"]))
        dist_h = int(np.random.randint(1, 3))
        dist_w = int(np.random.randint(1, 4))
        while _is_similar_to_protrusions(dist_h, dist_w, protrusions):
            dist_h = int(np.random.randint(1, 3))
            dist_w = int(np.random.randint(1, 4))

        attempts = 0
        while attempts < 50:
            x = int(np.random.randint(2, grid_size - dist_w - 2))
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
        x_pos = int(np.random.randint(3, grid_size - w - 3))
        attempts = 0
        while attempts < 50:
            valid = True
            for existing_side, existing_x, existing_w in hole_positions:
                if existing_side == side and _holes_too_close_1d(x_pos, w, existing_x, existing_w, min_hole_gap):
                    valid = False
                    break
            if valid:
                break
            x_pos = int(np.random.randint(3, grid_size - w - 3))
            attempts += 1
        if side == "top":
            _create_matching_hole_horizontal(grid, prot_for_hole, top_y, x_pos, direction="top")
        else:
            _create_matching_hole_horizontal(grid, prot_for_hole, bottom_y, x_pos, direction="bottom")
        hole_positions.append((side, x_pos, w))

    num_distractors = int(np.random.randint(2, 5))
    _add_distractor_holes_horizontal(grid, top_y, bottom_y, grid_size, hole_positions, num_distractors, protrusions)
    return hole_positions


def _generate_irregular_split_line_constrained(width: int, base_y: int) -> np.ndarray:
    split_line = np.full(width, base_y, dtype=np.int8)
    num_features = int(np.random.randint(1, min(3, width - 2)))
    available_positions = list(range(1, width - 1))
    feature_positions: list[int] = []
    for _ in range(num_features):
        if not available_positions:
            break
        pos = int(np.random.choice(available_positions))
        feature_positions.append(pos)
        available_positions = [p for p in available_positions if abs(p - pos) > 1]
    for x_pos in feature_positions:
        direction = int(np.random.choice([-1, 1]))
        split_line[x_pos] = base_y + direction
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
        if trimmed is None or np.sum(trimmed) == 0:
            return None
        if trimmed.shape[0] > max_height:
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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -int(d["area"]))
    assert comps

    bar = comps[0]
    guides, palette = _legend_components(comps[1:])
    assert len(palette) >= 4
    palette_index = {color: idx for idx, color in enumerate(palette)}

    pieces_raw = [c for c in comps[1:] if PIECE_MIN_AREA <= int(c["area"]) <= PIECE_MAX_AREA]
    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": _shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    color_groups: dict[int, list[int]] = {}
    for pi, piece in enumerate(pieces):
        color_groups.setdefault(int(piece["color"]), []).append(pi)
    group_colors = sorted(color_groups)

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    guide_occ = {(int(r), int(c)) for guide in guides for r, c in guide["coords"]}
    base_occ = set(bar_occ) | set(guide_occ)
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color
    for guide in guides:
        for r, c in guide["coords"]:
            out[int(r), int(c)] = int(guide["color"])

    def transform_color(chain_color, depth):
        idx = palette_index[int(chain_color)]
        if idx + depth >= len(palette):
            return None
        return int(palette[idx + depth])

    def bar_notches(mask_local, top0, left0):
        empty = ~mask_local
        h, w = map(int, empty.shape)
        notches = []
        for pts in components_mask(empty, connectivity=4):
            r = pts[:, 0]
            c = pts[:, 1]
            top = int((r == 0).sum())
            bottom = int((r == h - 1).sum())
            left = int((c == 0).sum())
            right = int((c == w - 1).sum())
            if top + bottom + left + right == 0:
                continue
            hole_gl = {(top0 + int(rr), left0 + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    def placement_pts(shape, top, left, occ, must_cover=None, strip=None):
        if top < 0 or left < 0 or top + shape["h"] > H or left + shape["w"] > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or shape["w"] != B - A + 1:
                    return None
            else:
                if top != A or shape["h"] != B - A + 1:
                    return None
        pts = _pixels(shape, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis, A, cover, shape):
        if axis == "cols":
            left = A
            tops = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if rN == top + lr:
                        lefts.add(cN - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in D4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, shape):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    notches = bar_notches(bar_mask, bar_r0, bar_c0)
    root_cands_by_color: dict[int, list[dict]] = {}
    for color in group_colors:
        if color not in palette_index or palette_index[color] + len(color_groups[color]) - 1 >= len(palette):
            continue
        cands = []
        for ni, notch in enumerate(notches):
            axis = notch["axis"]
            for pi in color_groups[color]:
                shape = pieces[pi]["shape"]
                for top, left in hole_positions(notch["hole"], shape):
                    pts = placement_pts(shape, top, left, base_occ, must_cover=notch["hole"])
                    if pts is None:
                        continue
                    if axis == "cols":
                        A, B = int(left), int(left + shape["w"] - 1)
                    else:
                        A, B = int(top), int(top + shape["h"] - 1)
                    mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                    cands.append(
                        (
                            (int(mv), -int(notch["mouth_n"])),
                            {
                                "color": color,
                                "ni": ni,
                                "pi": pi,
                                "top": int(top),
                                "left": int(left),
                                "axis": axis,
                                "A": A,
                                "B": B,
                                "shape": shape,
                            },
                        )
                    )
        cands.sort(key=lambda x: x[0])
        root_cands_by_color[color] = [item for _, item in cands[:12]]

    best_out = out.copy()
    best_key = None

    def placed_count(chains):
        return sum(1 + chain["depth"] for chain in chains)

    def search(out_grid, occ_set, chains):
        nonlocal best_out, best_key
        unfinished = [chain for chain in chains if chain["remaining"]]
        key = (placed_count(chains), -len(unfinished), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not unfinished:
            return True

        options = []
        for chain in unfinished:
            head = chain["head"]
            head_pts = _pixels(head["shape"], head["top"], head["left"])
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])
            cavity = {
                (r, c)
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if out_grid[r, c] == bg
            }
            if not cavity:
                return False
            if transform_color(chain["color"], int(chain["depth"]) + 1) is None:
                return False

            moves = []
            for pi in sorted(chain["remaining"]):
                shape = pieces[pi]["shape"]
                for top, left in aligned_positions(chain["axis"], chain["A"], cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(chain["axis"], chain["A"], chain["B"]))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    move_key = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                    moves.append((move_key, pi, int(top), int(left), pts, shape))
            if not moves:
                return False
            moves.sort(key=lambda x: x[0])
            options.append((len(moves), chain, moves[:8]))

        options.sort(key=lambda item: item[0])
        _, chain, moves = options[0]
        for _, pi, top, left, pts, shape in moves:
            new_depth = int(chain["depth"]) + 1
            write_color = transform_color(chain["color"], new_depth)
            if write_color is None:
                continue
            for r, c in pts:
                out_grid[r, c] = int(write_color)
            old_head = chain["head"]
            old_depth = int(chain["depth"])
            old_remaining = set(chain["remaining"])
            chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
            chain["depth"] = new_depth
            chain["remaining"].remove(pi)
            occ2 = set(occ_set) | set(pts)
            if search(out_grid, occ2, chains):
                return True
            chain["head"] = old_head
            chain["depth"] = old_depth
            chain["remaining"] = old_remaining
            for r, c in pts:
                out_grid[r, c] = int(bg)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color
            for guide in guides:
                for r, c in guide["coords"]:
                    out_grid[int(r), int(c)] = int(guide["color"])
        return False

    if not group_colors:
        return best_out
    candidate_lists = [root_cands_by_color.get(color, []) for color in group_colors]
    if any(not lst for lst in candidate_lists):
        return best_out

    for roots_sel in product(*candidate_lists):
        used_notches = set()
        occ0 = set(base_occ)
        out0 = out.copy()
        chains = []
        ok = True
        for root in roots_sel:
            if root["ni"] in used_notches:
                ok = False
                break
            pts = _pixels(root["shape"], root["top"], root["left"])
            if pts & occ0:
                ok = False
                break
            chain_color = int(root["color"])
            write_color = transform_color(chain_color, 0)
            if write_color is None:
                ok = False
                break
            for r, c in pts:
                out0[r, c] = int(write_color)
            occ0 |= pts
            used_notches.add(root["ni"])
            chains.append(
                {
                    "color": chain_color,
                    "axis": root["axis"],
                    "A": root["A"],
                    "B": root["B"],
                    "depth": 0,
                    "remaining": set(color_groups[chain_color]) - {root["pi"]},
                    "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                }
            )
        if not ok:
            continue
        if search(out0, occ0, chains):
            return out0
    return best_out


def _gen_attempt():
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.randint(1, 9)
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = bar_r0 + thickness - 1
        if bar_r1 >= H:
            bar_r1 = H - 1
            bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = bar_r0, H - 1 - bar_r1
        chain_sides = ["top", "bottom"]
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = bar_c0 + thickness - 1
        if bar_c1 >= W:
            bar_c1 = W - 1
            bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        space_before, space_after = bar_c0, W - 1 - bar_c1
        chain_sides = ["left", "right"]

    legend = _legend_layout(H, W, orientation)
    legend_colors_all = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(legend_colors_all)
    palette = legend_colors_all[:LEGEND_LEN]

    chain_lens = list(rng.choice([(2, 2), (2, 3), (3, 2)]))
    base_indices = _assign_chain_bases(rng, chain_lens, len(palette))
    if base_indices is None:
        return None

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    all_chain_pieces = []
    all_chains = []
    notch_holes = []
    occ_set = set(bar_cells) | set(legend["occ"])

    for chain_idx, side in enumerate(chain_sides):
        chain_len = int(chain_lens[chain_idx])
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            if (space_before if side == "top" else space_after) < chain_len * 2 + 3:
                return None
            for _ in range(50):
                sc0 = rng.randint(1, W - strip_width - 1)
                sc1 = sc0 + strip_width - 1
                if side == legend["side"] and _intervals_overlap(sc0, sc1, legend["span"][0] - 1, legend["span"][1] + 1):
                    continue
                break
            else:
                return None
            sA, sB = sc0, sc1
        else:
            if (space_before if side == "left" else space_after) < chain_len * 2 + 3:
                return None
            for _ in range(50):
                sr0 = rng.randint(1, H - strip_width - 1)
                sr1 = sr0 + strip_width - 1
                if side == legend["side"] and _intervals_overlap(sr0, sr1, legend["span"][0] - 1, legend["span"][1] + 1):
                    continue
                break
            else:
                return None
            sA, sB = sr0, sr1

        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sc0, sc1 + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(r, c) for r in range(sA, sB + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in D4}
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            piece_area = rng.randint(PIECE_MIN_AREA, PIECE_MAX_AREA)
            chain_dim = rng.randint(2, 3)

            if piece_idx == 0:
                if orientation == "horizontal":
                    if side == "top":
                        pr0, pr1 = max(0, bar_r0 - chain_dim), max(r for r, _ in notch)
                    else:
                        pr0, pr1 = min(r for r, _ in notch), min(H - 1, bar_r1 + chain_dim)
                    pc0, pc1 = sA, sB
                else:
                    if side == "left":
                        pc0, pc1 = max(0, bar_c0 - chain_dim), max(c for _, c in notch)
                    else:
                        pc0, pc1 = min(c for _, c in notch), min(W - 1, bar_c1 + chain_dim)
                    pr0, pr1 = sA, sB

                bbox_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
                target = max(max(PIECE_MIN_AREA, int(bbox_area * 0.54)), min(piece_area, int(bbox_area * 0.80)))
                if target > bbox_area or target < PIECE_MIN_AREA:
                    return None
                avail = {
                    (r, c)
                    for r in range(pr0, pr1 + 1)
                    for c in range(pc0, pc1 + 1)
                    if (r, c) in notch or (r, c) not in occ_set
                }
                piece_cells = _grow_connected(
                    rng,
                    avail,
                    target,
                    notch,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=5,
                )
                if piece_cells is None:
                    return None
                if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in piece_cells):
                    return None
            else:
                prev_coords = chain_pieces[-1]["coords"]
                prs = [r for r, _ in prev_coords]
                pcs = [c for _, c in prev_coords]
                prev_r0, prev_r1 = min(prs), max(prs)
                prev_c0, prev_c1 = min(pcs), max(pcs)

                if orientation == "horizontal":
                    cavity = {
                        (r, c)
                        for r in range(prev_r0, prev_r1 + 1)
                        for c in range(sA, sB + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                else:
                    cavity = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                if not cavity:
                    return None

                if orientation == "horizontal":
                    es = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {
                        (r, c)
                        for r in range(es[0], es[1] + 1)
                        for c in range(sA, sB + 1)
                        if (r, c) not in occ_set
                    } if es[0] <= es[1] else set()
                else:
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(es[0], es[1] + 1)
                        if (r, c) not in occ_set
                    } if es[0] <= es[1] else set()

                all_avail = cavity | ext
                target = min(PIECE_MAX_AREA, max(len(cavity), min(piece_area, len(all_avail))))
                if len(cavity) > PIECE_MAX_AREA or target < PIECE_MIN_AREA:
                    return None
                is_tail = piece_idx == chain_len - 1
                if is_tail:
                    target = min(PIECE_MAX_AREA, len(all_avail))

                piece_cells = _grow_connected(
                    rng,
                    all_avail,
                    target,
                    cavity,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=5,
                )
                if piece_cells is None or not cavity.issubset(piece_cells):
                    return None

                if is_tail:
                    prs2 = [r for r, _ in piece_cells]
                    pcs2 = [c for _, c in piece_cells]
                    t_r0, t_r1 = min(prs2), max(prs2)
                    t_c0, t_c1 = min(pcs2), max(pcs2)
                    if orientation == "horizontal":
                        remaining = {
                            (r, c)
                            for r in range(t_r0, t_r1 + 1)
                            for c in range(sA, sB + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    else:
                        remaining = {
                            (r, c)
                            for r in range(sA, sB + 1)
                            for c in range(t_c0, t_c1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    if remaining:
                        return None

            if not (PIECE_MIN_AREA <= len(piece_cells) <= PIECE_MAX_AREA):
                return None
            chain_pieces.append({"coords": piece_cells, "depth": piece_idx})
            occ_set |= piece_cells

        all_chains.append({"pieces": chain_pieces, "side": side, "A": sA, "B": sB})
        all_chain_pieces.extend(chain_pieces)

    if not (4 <= len(all_chain_pieces) <= 6):
        return None

    num_decoys = rng.randint(4, 7)
    decoys = 0
    for _ in range(500):
        if decoys >= num_decoys:
            break
        depth = rng.randint(1, 2)
        area = rng.randint(1, min(6, depth * 5))
        if orientation == "horizontal":
            ds = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if ds == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if ds == "top" else bar_r1
            dc0 = rng.randint(bar_c0, bar_c1)
            dc1 = min(bar_c1, dc0 + rng.randint(0, 4))
        else:
            ds = rng.choice(["left", "right"])
            dc0 = bar_c0 if ds == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if ds == "left" else bar_c1
            dr0 = rng.randint(bar_r0, bar_r1)
            dr1 = min(bar_r1, dr0 + rng.randint(0, 4))

        d_avail = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_holes = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in D4}
        if len(d_avail) < 1:
            continue
        area = min(area, len(d_avail))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None:
            continue
        if not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(dn)
        decoys += 1
    if decoys < 4:
        return None

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for color, dom in zip(palette, legend["dominoes"]):
        for r, c in dom:
            output[r, c] = int(color)

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color
    for color, dom in zip(palette, legend["dominoes"]):
        for r, c in dom:
            input_grid[r, c] = int(color)

    scatter_specs = []
    metrics = {"chain_lengths": list(chain_lens), "input_group_sizes": [], "color_span_sum": 0, "chains": []}
    for chain_idx, chain_info in enumerate(all_chains):
        chain = chain_info["pieces"]
        base_idx = int(base_indices[chain_idx])
        input_color = int(palette[base_idx])
        out_colors = []
        for piece in chain:
            out_color = int(palette[base_idx + int(piece["depth"])])
            out_colors.append(out_color)
            for r, c in piece["coords"]:
                output[r, c] = out_color
            scatter_specs.append(
                {
                    "coords": piece["coords"],
                    "color": input_color,
                    "side": chain_info["side"],
                    "A": int(chain_info["A"]),
                    "B": int(chain_info["B"]),
                }
            )
        metrics["input_group_sizes"].append(len(chain))
        metrics["color_span_sum"] += len(set(out_colors)) - 1
        metrics["chains"].append({"input_color": input_color, "output_colors": out_colors})

    placed_occ = set(bar_cells) | set(legend["buffer"])
    rng.shuffle(scatter_specs)

    for piece in scatter_specs:
        coords = piece["coords"]
        color = int(piece["color"])
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        p_r0, p_c0 = min(rs), min(cs)
        rel = [(r - p_r0, c - p_c0) for r, c in coords]
        p_h, p_w = max(rs) - p_r0 + 1, max(cs) - p_c0 + 1

        placed = False
        for _ in range(500):
            gap = rng.randint(2, 7)
            if orientation == "horizontal":
                top = (bar_r0 - gap - p_h) if piece["side"] == "top" else (bar_r1 + gap + 1)
                left_min = max(0, int(piece["A"]) - 2)
                left_max = min(W - p_w, int(piece["A"]) + 2)
                left = rng.randint(left_min, left_max)
            else:
                left = (bar_c0 - gap - p_w) if piece["side"] == "left" else (bar_c1 + gap + 1)
                top_min = max(0, int(piece["A"]) - 2)
                top_max = min(H - p_h, int(piece["A"]) + 2)
                top = rng.randint(top_min, top_max)
            if top < 0 or top + p_h > H or left < 0 or left + p_w > W:
                continue
            pts = {(top + dr, left + dc) for dr, dc in rel}
            if pts & placed_occ:
                continue
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in D4):
                continue
            for r, c in pts:
                input_grid[r, c] = color
            placed_occ |= pts
            placed = True
            break
        if not placed:
            return None

    return input_grid, output, metrics


def generate_input():
    for _ in range(5000):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out, _meta = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 5000 attempts")
