# Combo variant of 16b78196 (index 23)
# combo_spec: D1/1+D8/0+D13/0
# Name: 旋转匹配+扇形扩展+边界终止

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

for _UTILS_DIR in (
    Path("/home/lixinyu/snap/new/solvers"),
    Path("/home/lixinyu/snap/new/arc2_solver_select"),
):
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, find_components, get_bg


# Design analysis for combo D1/1 + D8/0 + D13/0
# ------------------------------------------------
# a) Concrete behavior and trace
#    Source task: a root piece plugs a bar notch, later pieces keep the same strip width, and
#    growth stops once the current bbox has no bg cavity left.
#    Combo behavior here:
#    - D1/1: scattered sprites may be rotated/flipped before matching.
#    - D8/0: widths grow one step at a time, 4 -> 5 -> 6, always on the same lateral side.
#    - D13/0: growth stops only when the tail touches the outer grid boundary.
#    We materialize that with one clear 3-piece chain:
#      root (width 4): notch-covering wedge whose front row leaves a 3-cell cavity;
#      middle (width 5): fills that cavity and has zero *bg* cavity in its own bbox because
#        its single missing back-row cell is occupied by the previous root, not by bg;
#      tail (width 6): no longer depends on cavity filling, attaches flush farther outward and
#        reaches the boundary using a full-width lip plus a one-cell spine.
#    Small top-growing example:
#      root rows: depth 0 has ###. on the bar edge, depth 1 has ####, depth 2 has ...#
#      middle rows: depth 2 has ##### except the old anchor cell, depth 3 has #####
#      tail rows: depth 4 has ######, depths 5..8 keep only the new outer anchor cell
#    Compared with the original:
#      - the solver must search dihedral variants instead of one fixed orientation;
#      - strip interval [A,B] is mutable and widens by exactly one on each extension;
#      - after the middle piece the cavity is empty, but the chain must still continue because
#        the head is not at the boundary yet.
# b) Complexity calibration
#    The generator keeps the ladder tiny and repeated: exactly three pieces with widths 4,5,6.
#    That is visually obvious in 2-3 examples, avoids combinatorial strip ambiguity, and still
#    forces the solver to integrate the three changed dimensions. The bar still has 5-7 extra
#    decoy notches so root matching is not trivial.
# c) Hidden assumptions broken from the original main()
#    - "each piece has one immutable shape" is invalid under D1/1, so pieces store unique
#      dihedral variants and every matching loop iterates over them.
#    - "strip width is fixed after the root" is invalid under D8/0, so chain state carries
#      mutable [A,B] plus the chosen fan_dir.
#    - "if head cavity is empty, chain cannot extend" is invalid under D13/0, so extension
#      generation falls back to all placements on the next widened strip whenever cavity is empty.
#    - "tail should have no remaining cavity" is no longer a useful termination predicate; the
#      solver now stops when the head touches the outward boundary on its growth side.
# d) Placement legality
#    Output is constructed first. Every chain cell is explicitly generated in-bounds, non-
#    overlapping, 4-connected, and monotone in the outward direction. Decoy notches are carved
#    only from boundary-adjacent bar cells and are rejected if they disconnect the bar.
#    Input scattering uses transformed variants but still forbids overlap and 4-neighbor contact.
# e) Visual observability
#    A human can see three coupled signals:
#    - the input sprites do not already face the notch correctly, so reorientation is necessary;
#    - the assembled chain widens from 4 to 5 to 6 on one lateral side only;
#    - the middle piece is already cavity-free, yet the output continues one more step until the
#      boundary is reached.
# f) Visual cleanliness
#    Only one chain is used, which avoids clutter while leaving enough room for the boundary-touch
#    tail to be legible. The bar remains the single dominant object; decoys stay on the bar and do
#    not overlap the piece field.
# g) Color inference
#    No color remapping is introduced. Colors remain piece identifiers only; the reasoning burden
#    is entirely geometric and therefore visible from the examples.
# h) New solver assumptions and generator guarantees
#    - Assumption: the true root has width 4 along the strip axis.
#      Guarantee: generator always uses a 4->5->6 ladder.
#    - Assumption: every used piece has at least one non-identity unique transform.
#      Guarantee: the three templates are asymmetric and generation rejects accidental symmetry.
#    - Assumption: there is exactly one used chain.
#      Guarantee: generator builds exactly one chain and several decoy notches.
#    - Assumption: after the first extension, the chosen fan direction stays fixed.
#      Guarantee: generator widens only on one side for the whole chain.
# i) Degeneracy guard
#    This construction intentionally blocks collapse back to the source task:
#    - the root is scattered with a non-identity transform;
#    - width changes happen in every sample;
#    - the middle piece has zero bg cavity but is not boundary-touching, so original "stop on
#      empty cavity" logic would miss the true tail.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_LAST_GEN_INFO: dict[str, object] = {}


def _shape_from_coords(coords: np.ndarray, bbox) -> dict[str, object]:
    r0, c0, r1, c1 = map(int, bbox)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return {
        "rel": rel,
        "rel_set": set(rel),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
    }


def _shape_from_cells(cells) -> dict[str, object]:
    coords = np.array(sorted((int(r), int(c)) for r, c in cells), dtype=int)
    rs = coords[:, 0]
    cs = coords[:, 1]
    bbox = (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))
    return _shape_from_coords(coords, bbox)


def _shape_variants(shape: dict[str, object]) -> tuple[list[dict[str, object]], tuple[tuple[int, int], ...]]:
    mask = np.zeros((int(shape["h"]), int(shape["w"])), dtype=bool)
    for r, c in shape["rel"]:
        mask[int(r), int(c)] = True

    base_key = tuple(shape["rel"])
    variants: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(mask, int(k))
        for flip in (False, True):
            arr = np.fliplr(rot) if flip else rot
            rel = tuple((int(r), int(c)) for r, c in np.argwhere(arr))
            if rel in seen:
                continue
            seen.add(rel)
            variants.append(
                {
                    "rel": list(rel),
                    "rel_set": set(rel),
                    "h": int(arr.shape[0]),
                    "w": int(arr.shape[1]),
                    "rot": int(k),
                    "flip": bool(flip),
                    "is_base": rel == base_key,
                }
            )
    return variants, base_key


def _transform_name(rot: int, flip: bool) -> str:
    return ("f" if flip else "r") + str(int(rot) * 90)


def _pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in shape["rel"]}


def _span_len(shape: dict[str, object], axis: str) -> int:
    return int(shape["w"] if axis == "cols" else shape["h"])


def _span_range(axis: str, top: int, left: int, shape: dict[str, object]) -> tuple[int, int]:
    if axis == "cols":
        return int(left), int(left + int(shape["w"]) - 1)
    return int(top), int(top + int(shape["h"]) - 1)


def _placement_pts(shape: dict[str, object], top: int, left: int, H: int, W: int, occ, must_cover=None, strip=None):
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


def _aligned_positions(axis: str, A: int, cover, shape: dict[str, object]):
    if axis == "cols":
        left = int(A)
        tops = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if int(cN) == left + int(lc):
                    tops.add(int(rN - lr))
        for top in tops:
            yield int(top), int(left)
    else:
        top = int(A)
        lefts = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if int(rN) == top + int(lr):
                    lefts.add(int(cN - lc))
        for left in lefts:
            yield int(top), int(left)


def _all_positions_in_strip(axis: str, A: int, shape: dict[str, object], H: int, W: int):
    if axis == "cols":
        left = int(A)
        for top in range(H - int(shape["h"]) + 1):
            yield int(top), int(left)
    else:
        top = int(A)
        for left in range(W - int(shape["w"]) + 1):
            yield int(top), int(left)


def _touches(pts, other) -> bool:
    for r, c in pts:
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in other:
                return True
    return False


def _hole_positions(hole, shape: dict[str, object]):
    pos = set()
    for rH, cH in hole:
        for lr, lc in shape["rel"]:
            pos.add((int(rH - lr), int(cH - lc)))
    return pos


def _head_bbox(head: dict[str, object]) -> tuple[int, int, int, int]:
    top = int(head["top"])
    left = int(head["left"])
    return top, left, top + int(head["shape"]["h"]) - 1, left + int(head["shape"]["w"]) - 1


def _head_cavity(head: dict[str, object], out_grid: np.ndarray, bg: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _head_bbox(head)
    return {(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(out_grid[r, c]) == int(bg)}


def _touches_growth_boundary(chain: dict[str, object], head: dict[str, object], H: int, W: int) -> bool:
    pts = _pixels(head["shape"], int(head["top"]), int(head["left"]))
    if chain["axis"] == "cols":
        row_needed = 0 if int(chain["dir"]) < 0 else H - 1
        return any(int(r) == row_needed for r, _ in pts)
    col_needed = 0 if int(chain["dir"]) < 0 else W - 1
    return any(int(c) == col_needed for _, c in pts)


def _extends_outward(chain: dict[str, object], head: dict[str, object], pts) -> bool:
    r0, c0, r1, c1 = _head_bbox(head)
    pr = [int(r) for r, _ in pts]
    pc = [int(c) for _, c in pts]
    if chain["axis"] == "cols":
        p0, p1 = min(pr), max(pr)
        if int(chain["dir"]) < 0:
            return p0 < r0 and p1 <= r1
        return p1 > r1 and p0 >= r0
    p0, p1 = min(pc), max(pc)
    if int(chain["dir"]) < 0:
        return p0 < c0 and p1 <= c1
    return p1 > c1 and p0 >= c0


def _interval_after_step(A: int, B: int, fan_dir: int) -> tuple[int, int]:
    return (int(A - 1), int(B)) if int(fan_dir) < 0 else (int(A), int(B + 1))


def _frontier_uses_fan_side(chain: dict[str, object], pts, fan_dir: int, A: int, B: int) -> bool:
    if chain["axis"] == "cols":
        frontier_depth = min(r for r, _ in pts) if int(chain["dir"]) < 0 else max(r for r, _ in pts)
        frontier_span = {int(c) for r, c in pts if int(r) == int(frontier_depth)}
    else:
        frontier_depth = min(c for _, c in pts) if int(chain["dir"]) < 0 else max(c for _, c in pts)
        frontier_span = {int(r) for r, c in pts if int(c) == int(frontier_depth)}
    expected = int(A) if int(fan_dir) < 0 else int(B)
    return frontier_span == {expected}


def _bar_notches(bar_mask: np.ndarray, bar_r0: int, bar_c0: int) -> list[dict[str, object]]:
    empty = ~bar_mask
    h, w = map(int, empty.shape)
    notches = []
    for pts in components_mask(empty, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        top = int((rr == 0).sum())
        bottom = int((rr == h - 1).sum())
        left = int((cc == 0).sum())
        right = int((cc == w - 1).sum())
        sides = [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0]
        if not sides:
            continue
        side, mouth_n = max(sides, key=lambda x: x[1])
        axis = "cols" if side in ("top", "bottom") else "rows"
        grow_dir = -1 if side in ("top", "left") else 1
        hole_gl = {(int(bar_r0 + r), int(bar_c0 + c)) for r, c in pts}
        notches.append({"axis": axis, "side": side, "dir": int(grow_dir), "hole": hole_gl, "mouth_n": int(mouth_n)})
    return notches


def _connected(cells) -> bool:
    if len(cells) <= 1:
        return True
    cells_set = {(int(r), int(c)) for r, c in cells}
    start = next(iter(cells_set))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (int(r + dr), int(c + dc))
            if nb in cells_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells_set)


def _grow_connected(rng: random.Random, available, target: int, seeds, retries: int = 6):
    for _ in range(retries):
        cells = {(int(r), int(c)) for r, c in seeds}
        available2 = {(int(r), int(c)) for r, c in available}
        if not cells.issubset(available2):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [
                nb
                for r, c in cells
                for dr, dc in DIR4
                if (nb := (int(r + dr), int(c + dc))) in available2 and nb not in cells
            ]
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        if _connected(cells):
            return cells
    return None


def _project_chain_cell(side: str, boundary: int, strip_coord: int, depth: int) -> tuple[int, int]:
    if side == "top":
        return int(boundary - depth), int(strip_coord)
    if side == "bottom":
        return int(boundary + depth), int(strip_coord)
    if side == "left":
        return int(strip_coord), int(boundary - depth)
    return int(strip_coord), int(boundary + depth)


def _build_root_piece(side: str, boundary: int, interval: tuple[int, int], anchor: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    A, B = map(int, interval)
    cells = set()
    notch = set()
    for s in range(A, B + 1):
        if s != int(anchor):
            cells.add(_project_chain_cell(side, boundary, s, 0))
            notch.add(_project_chain_cell(side, boundary, s, 0))
        cells.add(_project_chain_cell(side, boundary, s, 1))
    cells.add(_project_chain_cell(side, boundary, int(anchor), 2))
    return cells, notch


def _build_middle_piece(side: str, boundary: int, interval: tuple[int, int], blocked_anchor: int) -> set[tuple[int, int]]:
    A, B = map(int, interval)
    cells = set()
    for s in range(A, B + 1):
        if s != int(blocked_anchor):
            cells.add(_project_chain_cell(side, boundary, s, 2))
        cells.add(_project_chain_cell(side, boundary, s, 3))
    return cells


def _build_tail_piece(side: str, boundary: int, interval: tuple[int, int], anchor: int, boundary_depth: int) -> set[tuple[int, int]]:
    A, B = map(int, interval)
    cells = {_project_chain_cell(side, boundary, s, 4) for s in range(A, B + 1)}
    for depth in range(5, int(boundary_depth) + 1):
        cells.add(_project_chain_cell(side, boundary, int(anchor), depth))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
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

    pieces = []
    for p in pieces_raw:
        shape = _shape_from_coords(p["coords"], p["bbox"])
        variants, _ = _shape_variants(shape)
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "orig": (int(r0), int(c0)), "variants": variants})

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = str(notch["axis"])
        for pi, piece in enumerate(pieces):
            local = []
            for variant in piece["variants"]:
                if _span_len(variant, axis) != 4:
                    continue
                for top, left in _hole_positions(notch["hole"], variant):
                    pts = _placement_pts(variant, top, left, H, W, bar_occ, must_cover=notch["hole"])
                    if pts is None or not _touches(pts, bar_occ):
                        continue
                    A, B = _span_range(axis, top, left, variant)
                    mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                    local.append(
                        (
                            (int(mv), -int(notch["mouth_n"])),
                            {
                                "ni": int(ni),
                                "pi": int(pi),
                                "top": int(top),
                                "left": int(left),
                                "axis": axis,
                                "A": int(A),
                                "B": int(B),
                                "dir": int(notch["dir"]),
                                "shape": variant,
                            },
                        )
                    )
            local.sort(key=lambda x: x[0])
            root_cands.extend(local[:5])

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [item[1] for item in root_cands[:40]]

    best_out = out.copy()
    best_key = None

    def search(out_grid: np.ndarray, occ_set, chains, remaining) -> bool:
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (int(placed_cnt), -int(len(remaining)), -int(len(occ_set)))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def candidate_strips(chain):
            A, B = int(chain["A"]), int(chain["B"])
            dirs = [-1, 1] if chain["fan_dir"] is None else [int(chain["fan_dir"])]
            out_ranges = []
            for d in dirs:
                nA, nB = _interval_after_step(A, B, int(d))
                if chain["axis"] == "cols":
                    if 0 <= nA and nB < W:
                        out_ranges.append((int(d), int(nA), int(nB)))
                else:
                    if 0 <= nA and nB < H:
                        out_ranges.append((int(d), int(nA), int(nB)))
            return out_ranges

        def gen_extend_moves(chain, remaining_set):
            head = chain["head"]
            if _touches_growth_boundary(chain, head, H, W):
                return []
            axis = str(chain["axis"])
            head_pts = _pixels(head["shape"], int(head["top"]), int(head["left"]))
            cavity = _head_cavity(head, out_grid, int(bg))
            expected_width = int(chain["B"]) - int(chain["A"]) + 2
            moves = []
            for fan_dir, nA, nB in candidate_strips(chain):
                for pi in remaining_set:
                    for variant in pieces[pi]["variants"]:
                        if _span_len(variant, axis) != expected_width:
                            continue
                        pos_iter = (
                            _aligned_positions(axis, nA, cavity, variant)
                            if cavity
                            else _all_positions_in_strip(axis, nA, variant, H, W)
                        )
                        for top, left in pos_iter:
                            pts = _placement_pts(
                                variant,
                                top,
                                left,
                                H,
                                W,
                                occ_set,
                                must_cover=cavity if cavity else None,
                                strip=(axis, nA, nB),
                            )
                            if pts is None or not _touches(pts, head_pts) or not _extends_outward(chain, head, pts):
                                continue
                            if not cavity and not _frontier_uses_fan_side(chain, pts, int(fan_dir), int(nA), int(nB)):
                                continue
                            if axis == "cols":
                                progress = min(r for r, _ in pts) if int(chain["dir"]) < 0 else -max(r for r, _ in pts)
                                boundary_bonus = 0 if not _touches_growth_boundary({"axis": axis, "dir": chain["dir"]}, {"top": top, "left": left, "shape": variant}, H, W) else -100
                            else:
                                progress = min(c for _, c in pts) if int(chain["dir"]) < 0 else -max(c for _, c in pts)
                                boundary_bonus = 0 if not _touches_growth_boundary({"axis": axis, "dir": chain["dir"]}, {"top": top, "left": left, "shape": variant}, H, W) else -100
                            mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                            cavity_rank = 0 if cavity else -50
                            moves.append(
                                (
                                    int(boundary_bonus),
                                    int(progress),
                                    int(cavity_rank),
                                    int(mv),
                                    int(pi),
                                    int(top),
                                    int(left),
                                    pts,
                                    variant,
                                    int(fan_dir),
                                    int(nA),
                                    int(nB),
                                )
                            )
            moves.sort(key=lambda x: x[:4])
            return [m[4:] for m in moves[:16]]

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, variant, fan_dir, nA, nB in moves:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_state = (ch["A"], ch["B"], ch["fan_dir"], ch["head"])
                ch["A"], ch["B"], ch["fan_dir"] = int(nA), int(nB), int(fan_dir)
                ch["head"] = {"top": int(top), "left": int(left), "shape": variant}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(int(pi))
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["A"], ch["B"], ch["fan_dir"], ch["head"] = old_state
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    for root in cand_roots:
        occ0 = set(bar_occ)
        out0 = np.full((H, W), int(bg), dtype=int)
        for r, c in bar_occ:
            out0[r, c] = bar_color

        pts = _pixels(root["shape"], int(root["top"]), int(root["left"]))
        if pts & occ0:
            continue
        for r, c in pts:
            out0[r, c] = int(pieces[root["pi"]]["color"])
        occ0 |= pts
        chains = [
            {
                "axis": root["axis"],
                "A": int(root["A"]),
                "B": int(root["B"]),
                "dir": int(root["dir"]),
                "fan_dir": None,
                "head": {"top": int(root["top"]), "left": int(root["left"]), "shape": root["shape"]},
            }
        ]
        remaining = set(range(len(pieces))) - {int(root["pi"])}
        if search(out0, occ0, chains, remaining):
            return best_out

    return best_out


def _gen_attempt():
    rng = random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    fan_dir = rng.choice([-1, 1])
    thickness = rng.choice([6, 7])
    bar_color = rng.randint(1, 9)

    def choose_strip_start(dim: int, fan_dir: int) -> int:
        if int(fan_dir) > 0:
            cands = [s for s in (0, 1) if s + 5 < dim]
        else:
            cands = [s for s in (dim - 5, dim - 4) if 2 <= s <= dim - 4]
        return int(rng.choice(cands))

    if orientation == "horizontal":
        side = "top"
        boundary_depth = 8
        bar_r0 = boundary_depth
        bar_r1 = bar_r0 + thickness - 1
        bar_c0, bar_c1 = 0, W - 1
        strip_start = choose_strip_start(W, int(fan_dir))
        boundary = bar_r0
    else:
        side = "left"
        boundary_depth = 8
        bar_c0 = boundary_depth
        bar_c1 = bar_c0 + thickness - 1
        bar_r0, bar_r1 = 0, H - 1
        strip_start = choose_strip_start(H, int(fan_dir))
        boundary = bar_c0

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(int(r), int(c)) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    A0, B0 = int(strip_start), int(strip_start + 3)
    A1, B1 = _interval_after_step(A0, B0, int(fan_dir))
    A2, B2 = _interval_after_step(A1, B1, int(fan_dir))
    anchor0 = int(A0 if fan_dir < 0 else B0)
    anchor2 = int(A2 if fan_dir < 0 else B2)

    root_cells, notch = _build_root_piece(side, boundary, (A0, B0), anchor0)
    middle_cells = _build_middle_piece(side, boundary, (A1, B1), anchor0)
    tail_cells = _build_tail_piece(side, boundary, (A2, B2), anchor2, boundary_depth)
    piece_cells_seq = [root_cells, middle_cells, tail_cells]

    if any(len(cells) < 8 or len(cells) > 13 for cells in piece_cells_seq):
        return None
    if any(not _connected(cells) for cells in piece_cells_seq):
        return None
    if notch - bar_cells:
        return None

    occ_set = set(bar_cells) - set(notch)
    for cells in piece_cells_seq:
        if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
            return None
        if cells & occ_set:
            return None
        occ_set |= cells

    middle_bbox = (
        min(r for r, _ in middle_cells),
        min(c for _, c in middle_cells),
        max(r for r, _ in middle_cells),
        max(c for _, c in middle_cells),
    )
    middle_cavity = {
        (int(r), int(c))
        for r in range(middle_bbox[0], middle_bbox[2] + 1)
        for c in range(middle_bbox[1], middle_bbox[3] + 1)
        if (r, c) not in middle_cells and (r, c) not in root_cells and (r, c) not in tail_cells
    }
    if middle_cavity:
        return None

    if side == "top":
        if not any(r == 0 for r, _ in tail_cells):
            return None
    elif side == "left":
        if not any(c == 0 for _, c in tail_cells):
            return None

    bar_cells -= notch

    notch_holes = [set(notch)]
    decoys_needed = rng.randint(5, 7)
    decoys = 0
    for _ in range(900):
        if decoys >= decoys_needed:
            break
        depth = rng.randint(1, 2)
        # Keep the true notch as the only 3-cell mouth; otherwise the solver (and a human)
        # can legitimately attach the whole fan chain to a decoy boundary opening.
        span = rng.randint(1, 2)
        if orientation == "horizontal":
            ds = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if ds == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if ds == "top" else bar_r1
            dc0 = rng.randint(bar_c0, bar_c1 - span + 1)
            dc1 = dc0 + span - 1
        else:
            ds = rng.choice(["left", "right"])
            dc0 = bar_c0 if ds == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if ds == "left" else bar_c1
            dr0 = rng.randint(bar_r0, bar_r1 - span + 1)
            dr1 = dr0 + span - 1

        d_avail = {(int(r), int(c)) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_holes = set().union(*notch_holes)
        halo = {(int(r + dr), int(c + dc)) for r, c in all_holes for dr, dc in DIR4}
        d_avail -= all_holes | halo
        if not d_avail:
            continue
        area = rng.randint(1, min(len(d_avail), depth * span))
        dn = _grow_connected(rng, d_avail, int(area), {rng.choice(list(d_avail))})
        if dn is None:
            continue
        if not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(set(dn))
        decoys += 1
    if decoys < 5:
        return None

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    piece_infos = []
    for idx, cells in enumerate(piece_cells_seq):
        shape = _shape_from_cells(cells)
        variants, base_key = _shape_variants(shape)
        non_base = [v for v in variants if tuple(v["rel"]) != base_key]
        if not non_base:
            return None
        piece_infos.append(
            {
                "coords": set(cells),
                "color": int(avail_colors[idx]),
                "shape": shape,
                "variants": variants,
                "base_key": base_key,
                "non_base": non_base,
                "is_root": bool(idx == 0),
            }
        )

    output = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        output[r, c] = int(bar_color)
    for p in piece_infos:
        for r, c in p["coords"]:
            output[r, c] = int(p["color"])

    input_grid = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = int(bar_color)

    placed_occ = set(bar_cells)
    order = list(range(len(piece_infos)))
    rng.shuffle(order)
    transform_hist: dict[str, int] = {}

    for idx in order:
        piece = piece_infos[idx]
        candidate_vars = list(piece["non_base"])
        rng.shuffle(candidate_vars)
        placed = False
        for variant in candidate_vars:
            p_h, p_w = int(variant["h"]), int(variant["w"])
            for _ in range(400):
                gap = rng.randint(2, 7)
                if orientation == "horizontal":
                    top = (bar_r0 - gap - p_h) if rng.random() < 0.5 else (bar_r1 + gap + 1)
                    left = rng.randint(0, max(0, W - p_w))
                else:
                    left = (bar_c0 - gap - p_w) if rng.random() < 0.5 else (bar_c1 + gap + 1)
                    top = rng.randint(0, max(0, H - p_h))
                if top < 0 or left < 0 or top + p_h > H or left + p_w > W:
                    continue
                pts = {(int(top + dr), int(left + dc)) for dr, dc in variant["rel"]}
                if pts & placed_occ:
                    continue
                if any((int(r + dr), int(c + dc)) in placed_occ for r, c in pts for dr, dc in DIR4):
                    continue
                for r, c in pts:
                    input_grid[r, c] = int(piece["color"])
                placed_occ |= pts
                key = _transform_name(int(variant["rot"]), bool(variant["flip"]))
                transform_hist[key] = transform_hist.get(key, 0) + 1
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    info = {
        "orientation": orientation,
        "used_side": side,
        "fan_dir": int(fan_dir),
        "widths": [4, 5, 6],
        "piece_count": 3,
        "decoy_count": int(decoys),
        "transformed_piece_count": 3,
        "root_transformed": True,
        "middle_cavity_zero": True,
        "tail_touches_boundary": True,
        "transform_hist": dict(transform_hist),
        "bar_thickness": int(thickness),
    }
    return input_grid.astype(int), output.astype(int), info


def generate_input() -> np.ndarray:
    global _LAST_GEN_INFO
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out, info = result
        if np.array_equal(main(inp), out):
            _LAST_GEN_INFO = dict(info)
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
