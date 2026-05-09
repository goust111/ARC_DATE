# Combo variant of 16b78196 (index 15)
# combo_spec: D3/3+D12/2
# Name: 深缺口+无延伸填充
#
# Design analysis for the fused variant
# a) Pixel-level behavior
# - Original root placement only needed to cover a bar notch and could keep
#   growing outward through later pieces. Here the used notch is generated from
#   the root piece's exact overlap with the bar, so the root can sink 1..floor(T/2)
#   cells into the bar and the solver must match that exact deep overlap.
# - Original continuation used: next_piece covers cavity AND may extend outside
#   the previous bbox. Here continuation is stricter: next_piece must equal the
#   current cavity exactly. A concrete chain can be:
#     root = big U-shape touching the bar slot
#     next = the exact hole inside that U
#     tail = the exact hole inside next
#   so the chain becomes a nested fill process rather than outward growth.
# - Small example: if the root bbox is 6x5 and the root occupies every cell
#   except a 3x2 shape near its outer end, then the next piece is exactly that
#   3x2 shape at the same absolute cavity position. If that next piece is itself
#   a hollow U around a 2x1 cavity, the tail is exactly that 2x1 cavity. The
#   termination condition is therefore "current cavity is empty".
#
# b) Complexity calibration
# - The variant keeps 1-2 chains and several distractor notches so the solver
#   still has ambiguity to resolve.
# - Deep slots are visually obvious because the generator biases bar thickness to
#   6-7 and forces at least one used notch to depth=floor(thickness/2), which is
#   beyond the original shallow 1-2 depth regime.
# - The nested cavity rule is visible because later pieces shrink to the exact
#   cavity instead of preserving strip width and extending.
#
# c) Original implicit assumptions that no longer hold
# - "All pieces in a chain share the same strip width" is no longer reliable.
#   Once a later piece must equal the cavity exactly, its bbox may shrink in the
#   perpendicular direction. The fused solver therefore matches exact cavity
#   shape, not fixed-width strip placement.
# - "Tail piece is non-rectangular" also breaks. Under D12/2 the final cavity is
#   empty, so the tail can legitimately be a solid rectangle.
# - Root matching cannot rely on shallow notch depth. The solver now requires the
#   piece's overlap with the bar bbox to equal the notch exactly.
#
# d) Placement legality
# - Output-first generation places roots first, carves the matching used holes in
#   the bar, then scatters trimmed piece masks back into the input with an
#   8-neighborhood exclusion rule. This prevents overlap, touching, and OOB.
#
# e) Visual observability
# - A human can see the deep notch because the root visibly plugs far into the
#   bar. A human can also see the no-extension rule because each later sprite is
#   exactly the blank region inside the previous sprite, with no extra outward
#   cells.
#
# f) Visual cleanliness
# - Later pieces only occupy cavities carved by earlier ones, so each chain stays
#   in a compact local region. Distractor notches are shallow rectangles so they
#   do not visually compete with the deep used notches.
#
# g) New solver assumptions and how the generator guarantees them
# - Used notches are one-sided boundary holes: guaranteed by carving them from
#   root∩bar_bbox.
# - Each non-root cavity is a single trimmed monochrome shape: guaranteed by the
#   recursive "parent = full rectangle minus child flush to the outer side"
#   construction.
# - At least one root uses a genuinely deep slot: guaranteed by depth=thickness//2
#   for the first chain.

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_coords(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_from_coords(coords: set[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    rel = sorted((r - r0, c - c0) for r, c in coords)
    return {
        "rel": rel,
        "rel_set": set(rel),
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "area": len(rel),
        "bbox": (r0, c0, r1, c1),
    }


def _shape_from_component(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    r0, c0, r1, c1 = map(int, bbox)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return {
        "rel": rel,
        "rel_set": set(rel),
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "area": len(rel),
        "bbox": (r0, c0, r1, c1),
    }


def _pixels(shape: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape["rel"]}


def _shape_signature(shape: dict) -> tuple[int, int, frozenset[tuple[int, int]]]:
    return int(shape["h"]), int(shape["w"]), frozenset(shape["rel_set"])


def _set_to_mask(coords: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in coords:
        mask[r, c] = 1
    return mask


def _trim_mask(mask: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int]]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None, (0, 0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    r0, r1 = int(r_idx[0]), int(r_idx[-1])
    c0, c1 = int(c_idx[0]), int(c_idx[-1])
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask == 1)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] == 1 and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _rect_coords(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _shift_coords(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _transform_coords(coords: set[tuple[int, int]], shape: tuple[int, int], side: str) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    h, w = shape
    if side == "bottom":
        return set(coords), (h, w)
    if side == "top":
        return {(h - 1 - r, c) for r, c in coords}, (h, w)
    if side == "right":
        return {(w - 1 - c, r) for r, c in coords}, (w, h)
    if side == "left":
        return {(c, h - 1 - r) for r, c in coords}, (w, h)
    raise ValueError(side)


def _placement_ok(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    gh, gw = occupied.shape
    ph, pw = mask.shape
    if top < 0 or left < 0 or top + ph > gh or left + pw > gw:
        return False
    for r in range(ph):
        for c in range(pw):
            if mask[r, c] == 0:
                continue
            gr, gc = top + r, left + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < gh and 0 <= nc < gw and occupied[nr, nc]:
                        return False
    return True


def _paint(grid: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _bar_notches(bar_mask: np.ndarray, bar_r0: int, bar_c0: int) -> list[dict]:
    empty = ~bar_mask
    h, w = empty.shape
    notches: list[dict] = []
    for pts in components_mask(empty, connectivity=4):
        r = pts[:, 0]
        c = pts[:, 1]
        top = int((r == 0).sum())
        bottom = int((r == h - 1).sum())
        left = int((c == 0).sum())
        right = int((c == w - 1).sum())
        if top + bottom + left + right == 0:
            continue
        side = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )[0]
        if side == "top":
            depth = int(r.max()) + 1
        elif side == "bottom":
            depth = int(h - r.min())
        elif side == "left":
            depth = int(c.max()) + 1
        else:
            depth = int(w - c.min())
        hole = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}
        notches.append({"side": side, "depth": depth, "hole": hole, "area": len(hole)})
    return notches


def _piece_matches_side(bbox: tuple[int, int, int, int], side: str, bar_bbox: tuple[int, int, int, int]) -> bool:
    pr0, pc0, pr1, pc1 = map(int, bbox)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar_bbox)
    if side == "top":
        return pr1 < bar_r0
    if side == "bottom":
        return pr0 > bar_r1
    if side == "left":
        return pc1 < bar_c0
    return pc0 > bar_c1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -int(d["area"]))
    if not comps:
        return g.copy()

    bar = comps[0]
    pieces_raw = comps[1:]

    out = np.full((h, w), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_bbox = (bar_r0, bar_c0, bar_r1, bar_c1)
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_box = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    pieces = []
    sig_to_piece_ids: dict[tuple[int, int, frozenset[tuple[int, int]]], list[int]] = {}
    for idx, p in enumerate(pieces_raw):
        shape = _shape_from_component(p["coords"], p["bbox"])
        entry = {"id": idx, "color": int(p["color"]), "shape": shape, "orig": tuple(map(int, p["bbox"][:2]))}
        pieces.append(entry)
        sig_to_piece_ids.setdefault(_shape_signature(shape), []).append(idx)

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)

    def root_positions(hole: set[tuple[int, int]], shape: dict) -> set[tuple[int, int]]:
        pos = set()
        for rh, ch in hole:
            for lr, lc in shape["rel"]:
                pos.add((rh - lr, ch - lc))
        return pos

    def root_outward(pts: set[tuple[int, int]], side: str) -> bool:
        if side == "top":
            return min(r for r, _ in pts) < bar_r0
        if side == "bottom":
            return max(r for r, _ in pts) > bar_r1
        if side == "left":
            return min(c for _, c in pts) < bar_c0
        return max(c for _, c in pts) > bar_c1

    root_cands: list[dict] = []
    for ni, notch in enumerate(notches):
        for pi, piece in enumerate(pieces):
            if not _piece_matches_side(piece["shape"]["bbox"], notch["side"], bar_bbox):
                continue
            seen = set()
            for top, left in root_positions(notch["hole"], piece["shape"]):
                if (top, left) in seen:
                    continue
                seen.add((top, left))
                if top < 0 or left < 0 or top + piece["shape"]["h"] > h or left + piece["shape"]["w"] > w:
                    continue
                pts = _pixels(piece["shape"], top, left)
                if pts & bar_occ:
                    continue
                if (pts & bar_box) != notch["hole"]:
                    continue
                if not root_outward(pts, notch["side"]):
                    continue
                move = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    {
                        "ni": ni,
                        "pi": pi,
                        "top": int(top),
                        "left": int(left),
                        "pts": pts,
                        "side": notch["side"],
                        "depth": int(notch["depth"]),
                        "hole_area": int(notch["area"]),
                        "score": (-int(notch["depth"]), -int(notch["area"]), -int(piece["shape"]["area"]), int(move)),
                    }
                )

    root_cands.sort(key=lambda d: d["score"])
    balanced_root_cands: list[dict] = []
    per_notch_kept: dict[int, int] = {}
    for cand in root_cands:
        ni = int(cand["ni"])
        if per_notch_kept.get(ni, 0) >= 12:
            continue
        balanced_root_cands.append(cand)
        per_notch_kept[ni] = per_notch_kept.get(ni, 0) + 1
        if len(balanced_root_cands) >= 96:
            break
    root_cands = balanced_root_cands

    best_out = out.copy()
    best_key = (-1, -1)

    def cavity_of(head: dict, out_grid: np.ndarray) -> tuple[set[tuple[int, int]], dict | None]:
        top, left, shape = int(head["top"]), int(head["left"]), head["shape"]
        cavity = {
            (r, c)
            for r in range(top, top + int(shape["h"]))
            for c in range(left, left + int(shape["w"]))
            if out_grid[r, c] == bg
        }
        if not cavity:
            return set(), None
        return cavity, _shape_from_coords(cavity)

    def search(out_grid: np.ndarray, occ_set: set[tuple[int, int]], chains: list[dict], remaining: set[int]) -> bool:
        nonlocal best_out, best_key
        placed = len(pieces) - len(remaining)
        key = (placed, -len(remaining))
        if key > best_key:
            best_key = key
            best_out = out_grid.copy()

        active = []
        for ci, chain in enumerate(chains):
            cavity, cav_shape = cavity_of(chain["head"], out_grid)
            if not cavity:
                continue
            sig = _shape_signature(cav_shape)
            moves = []
            for pi in sig_to_piece_ids.get(sig, []):
                if pi not in remaining:
                    continue
                if not _piece_matches_side(pieces[pi]["shape"]["bbox"], chain["side"], bar_bbox):
                    continue
                top, left = cav_shape["bbox"][:2]
                pts = _pixels(pieces[pi]["shape"], top, left)
                if pts != cavity:
                    continue
                if pts & occ_set:
                    continue
                moves.append((pi, top, left, pts))
            if not moves:
                return False
            active.append((len(moves), ci, moves))

        if not active:
            return not remaining

        _, ci, moves = min(active, key=lambda x: x[0])
        for pi, top, left, pts in moves:
            out2 = out_grid.copy()
            _paint(out2, pts, pieces[pi]["color"])
            chains2 = []
            for j, ch in enumerate(chains):
                if j == ci:
                    head = {"top": int(top), "left": int(left), "shape": pieces[pi]["shape"]}
                else:
                    head = {"top": int(ch["head"]["top"]), "left": int(ch["head"]["left"]), "shape": ch["head"]["shape"]}
                chains2.append({"side": ch["side"], "head": head})
            rem2 = set(remaining)
            rem2.remove(pi)
            if search(out2, occ_set | pts, chains2, rem2):
                return True
        return False

    for k in range(1, min(2, len(root_cands)) + 1):
        for sel in combinations(range(len(root_cands)), k):
            used_notches = set()
            used_pieces = set()
            occ = set(bar_occ)
            out0 = out.copy()
            chains = []
            ok = True
            for idx in sel:
                root = root_cands[idx]
                if root["ni"] in used_notches or root["pi"] in used_pieces or root["pts"] & occ:
                    ok = False
                    break
                used_notches.add(root["ni"])
                used_pieces.add(root["pi"])
                _paint(out0, root["pts"], pieces[root["pi"]]["color"])
                occ |= root["pts"]
                chains.append(
                    {
                        "side": root["side"],
                        "head": {
                            "top": root["top"],
                            "left": root["left"],
                        "shape": pieces[root["pi"]]["shape"],
                        },
                    }
                )
            if not ok:
                continue
            if search(out0, occ, chains, set(range(len(pieces))) - used_pieces):
                return best_out

    return best_out


def _build_parent(pieces: list[set[tuple[int, int]]], cur_h: int, cur_w: int, top_margin: int, left_margin: int, right_margin: int) -> tuple[list[set[tuple[int, int]]], int, int]:
    new_h = cur_h + top_margin
    new_w = cur_w + left_margin + right_margin
    shifted = [_shift_coords(coords, top_margin, left_margin) for coords in pieces]
    parent = _rect_coords(new_h, new_w) - shifted[0]
    return [parent] + shifted, new_h, new_w


def _generate_canonical_chain(rng: np.random.Generator, bar_thickness: int, chain_len: int, root_depth: int) -> tuple[list[set[tuple[int, int]]], tuple[int, int]]:
    tail_h = int(rng.integers(2, 4))
    tail_w = int(rng.integers(2, 4))
    pieces = [_rect_coords(tail_h, tail_w)]
    cur_h, cur_w = tail_h, tail_w

    if chain_len == 3:
        top_margin = int(rng.integers(1, 3))
        left_margin = int(rng.integers(0, 2))
        right_margin = 1 if left_margin == 0 else int(rng.integers(0, 2))
        pieces, cur_h, cur_w = _build_parent(pieces, cur_h, cur_w, top_margin, left_margin, right_margin)
        assert _mask_connected(_set_to_mask(pieces[0], cur_h, cur_w))

    extra_top = int(rng.integers(0, 2))
    top_margin = int(root_depth + extra_top)
    left_margin = 1 if bool(rng.integers(0, 2)) else 0
    right_margin = 1 if left_margin == 0 else int(rng.integers(0, 2))
    pieces, cur_h, cur_w = _build_parent(pieces, cur_h, cur_w, top_margin, left_margin, right_margin)
    assert _mask_connected(_set_to_mask(pieces[0], cur_h, cur_w))
    return pieces, (cur_h, cur_w)


def _orient_sides(is_vertical: bool, chain_count: int, rng: np.random.Generator) -> list[str]:
    if not is_vertical:
        if chain_count == 2:
            return ["top", "bottom"]
        return [str(rng.choice(["top", "bottom"]))]
    if chain_count == 2:
        return ["left", "right"]
    return [str(rng.choice(["left", "right"]))]


def _place_chain_absolute(
    pieces_local: list[set[tuple[int, int]]],
    local_shape: tuple[int, int],
    side: str,
    root_depth: int,
    span_start: int,
    bar_r0: int,
    bar_r1: int,
    bar_c0: int,
    bar_c1: int,
) -> tuple[list[set[tuple[int, int]]], tuple[int, int], tuple[int, int]]:
    transformed = []
    new_shape = local_shape
    for coords in pieces_local:
        coords_t, new_shape = _transform_coords(coords, local_shape, side)
        transformed.append(coords_t)
    root_h, root_w = new_shape

    if side == "bottom":
        top = bar_r1 - root_depth + 1
        left = span_start
    elif side == "top":
        top = bar_r0 - (root_h - root_depth)
        left = span_start
    elif side == "right":
        top = span_start
        left = bar_c1 - root_depth + 1
    else:
        top = span_start
        left = bar_c0 - (root_w - root_depth)

    abs_pieces = [{(top + r, left + c) for r, c in coords} for coords in transformed]
    return abs_pieces, (root_h, root_w), (top, left)


def _carve_rect_notch(mask: np.ndarray, side: str, start: int, width: int, depth: int) -> None:
    h, w = mask.shape
    if side == "top":
        mask[:depth, start : start + width] = False
    elif side == "bottom":
        mask[h - depth :, start : start + width] = False
    elif side == "left":
        mask[start : start + width, :depth] = False
    else:
        mask[start : start + width, w - depth :] = False


def _intervals_conflict(a0: int, a1: int, b0: int, b1: int, gap: int) -> bool:
    return not (a1 + gap < b0 or b1 + gap < a0)


def _scatter_piece(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    piece_mask: np.ndarray,
    color: int,
    regions: list[tuple[int, int, int, int]],
) -> bool:
    ph, pw = piece_mask.shape
    order = rng.permutation(len(regions))
    for idx in order:
        r0, c0, r1, c1 = regions[int(idx)]
        if r1 - r0 < ph or c1 - c0 < pw:
            continue
        for _ in range(150):
            top = int(rng.integers(r0, r1 - ph + 1))
            left = int(rng.integers(c0, c1 - pw + 1))
            if not _placement_ok(occupied, piece_mask, top, left):
                continue
            for r in range(ph):
                for c in range(pw):
                    if piece_mask[r, c]:
                        grid[top + r, left + c] = int(color)
                        occupied[top + r, left + c] = True
            return True
    return False


def _generate_case(seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    grid_size = 30
    bg = 0
    is_vertical = bool(rng.integers(0, 2))
    bar_thickness = int(rng.integers(6, 8))
    chain_count = int(rng.integers(1, 3))
    bar_color = int(rng.choice(np.arange(1, 10)))
    piece_colors = [int(c) for c in rng.permutation([c for c in range(1, 10) if c != bar_color])]

    if not is_vertical:
        bar_r0 = (grid_size - bar_thickness) // 2
        bar_r1 = bar_r0 + bar_thickness - 1
        bar_c0, bar_c1 = 0, grid_size - 1
    else:
        bar_c0 = (grid_size - bar_thickness) // 2
        bar_c1 = bar_c0 + bar_thickness - 1
        bar_r0, bar_r1 = 0, grid_size - 1

    sides = _orient_sides(is_vertical, chain_count, rng)
    root_depths = [bar_thickness // 2]
    for _ in range(1, chain_count):
        root_depths.append(int(rng.integers(1, bar_thickness // 2 + 1)))

    chain_specs = []
    used_intervals: dict[str, list[tuple[int, int]]] = {}
    used_deep_intervals: dict[str, list[tuple[int, int, int]]] = {}
    opposite_side = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    for idx, side in enumerate(sides):
        chain_len = 2
        pieces_local, local_shape = _generate_canonical_chain(rng, bar_thickness, chain_len, root_depths[idx])
        root_h, root_w = local_shape if side in ("top", "bottom") else (local_shape[1], local_shape[0])
        span_len = root_w if side in ("top", "bottom") else root_h
        span_limit = grid_size - span_len - 2
        for _ in range(80):
            span_start = int(rng.integers(2, span_limit + 1))
            same_side = used_intervals.get(side, [])
            same_side_ok = all(not _intervals_conflict(span_start, span_start + span_len - 1, a, b, 2) for a, b in same_side)
            opposite_ok = True
            for a, b, other_depth in used_deep_intervals.get(opposite_side[side], []):
                if root_depths[idx] + other_depth >= bar_thickness and _intervals_conflict(
                    span_start, span_start + span_len - 1, a, b, 1
                ):
                    opposite_ok = False
                    break
            if same_side_ok and opposite_ok:
                used_intervals.setdefault(side, []).append((span_start, span_start + span_len - 1))
                used_deep_intervals.setdefault(side, []).append((span_start, span_start + span_len - 1, root_depths[idx]))
                break
        else:
            raise RuntimeError("failed to place chain interval")

        abs_pieces, root_shape, root_anchor = _place_chain_absolute(
            pieces_local, local_shape, side, root_depths[idx], span_start, bar_r0, bar_r1, bar_c0, bar_c1
        )
        chain_specs.append(
            {
                "side": side,
                "depth": root_depths[idx],
                "pieces": abs_pieces,
                "root_shape": root_shape,
                "root_anchor": root_anchor,
            }
        )

    bar_mask = np.zeros((grid_size, grid_size), dtype=bool)
    if not is_vertical:
        bar_mask[bar_r0 : bar_r1 + 1, :] = True
    else:
        bar_mask[:, bar_c0 : bar_c1 + 1] = True

    all_piece_coords: list[tuple[set[tuple[int, int]], int, str]] = []
    metrics = {"deep_depths": [], "exact_extensions": []}
    color_idx = 0
    for spec in chain_specs:
        root = spec["pieces"][0]
        used_hole = {
            (r, c)
            for r, c in root
            if bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1
        }
        for r, c in used_hole:
            bar_mask[r, c] = False
        metrics["deep_depths"].append(int(spec["depth"]))

        prev = root
        for piece in spec["pieces"]:
            all_piece_coords.append((piece, piece_colors[color_idx], spec["side"]))
            color_idx += 1
        for child in spec["pieces"][1:]:
            cavity = {
                (r, c)
                for r in range(min(rr for rr, _ in prev), max(rr for rr, _ in prev) + 1)
                for c in range(min(cc for _, cc in prev), max(cc for _, cc in prev) + 1)
                if (r, c) not in prev
            }
            metrics["exact_extensions"].append(len(child ^ cavity))
            prev = child

    distractor_total = int(rng.integers(4, 7))
    span_by_side = used_intervals
    for _ in range(distractor_total):
        side = str(rng.choice(["left", "right"] if is_vertical else ["top", "bottom"]))
        width = int(rng.integers(1, 4))
        depth = int(rng.integers(1, 3))
        limit = grid_size - width - 2
        for _ in range(100):
            start = int(rng.integers(2, limit + 1))
            same_side = span_by_side.get(side, [])
            if all(not _intervals_conflict(start, start + width - 1, a, b, 2) for a, b in same_side):
                span_by_side.setdefault(side, []).append((start, start + width - 1))
                _carve_rect_notch(bar_mask, side, start, width, depth)
                break

    output = np.full((grid_size, grid_size), bg, dtype=int)
    output[bar_mask] = bar_color
    for coords, color, _side in all_piece_coords:
        _paint(output, coords, color)

    input_grid = np.full((grid_size, grid_size), bg, dtype=int)
    input_grid[bar_mask] = bar_color
    occupied = input_grid != bg
    if not is_vertical:
        top_regions = [(1, 1, max(2, bar_r0 - 1), grid_size - 1)]
        bot_regions = [(bar_r1 + 2, 1, grid_size - 1, grid_size - 1)]
        regions_by_side = {"top": top_regions, "bottom": bot_regions}
    else:
        left_regions = [(1, 1, grid_size - 1, max(2, bar_c0 - 1))]
        right_regions = [(1, bar_c1 + 2, grid_size - 1, grid_size - 1)]
        regions_by_side = {"left": left_regions, "right": right_regions}

    for coords, color, side in sorted(all_piece_coords, key=lambda x: -len(x[0])):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        piece_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
        for r, c in coords:
            piece_mask[r - r0, c - c0] = 1
        if not _scatter_piece(rng, input_grid, occupied, piece_mask, color, regions_by_side[side]):
            raise RuntimeError("failed to scatter piece")

    return input_grid.astype(int), output.astype(int), metrics


def generate_input() -> np.ndarray:
    for attempt in range(300):
        seed = int(np.random.randint(1, 1_000_000_000))
        try:
            inp, out, _meta = _generate_case(seed)
        except Exception:
            continue
        try:
            got = main(inp)
        except Exception:
            continue
        if np.array_equal(got, out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
