# Combo variant of 16b78196 (index 10)
# combo_spec: D6/1+D7/0
# Name: 移动轨迹+紧凑裁剪
#
# Design analysis for this combo variant
# --------------------------------------
# concrete_behavior:
#   - Base assembly logic is still the original 16b78196 rule: a full-span bar has a
#     few real notches; each used notch is plugged by a root sprite, then later sprites
#     in the same chain fill the previous sprite bbox cavity while keeping the same strip
#     width.
#   - D6/1 changes only the rendering of motion. After we infer each sprite's final
#     assembled placement, we draw a 1-pixel straight trail in that sprite's own color
#     from the center of its scattered source bbox to the center of its target bbox.
#     The source sprite itself does not remain; only the trail and the final sprite do.
#   - D7/0 is then applied to the fused picture, not to the bare assembly. So the crop
#     must enclose the bar, all assembled chains, and every trail cell. The correct
#     bounding box therefore depends on reconstructing motion first.
#   - Example trace:
#       horizontal bar at rows 12..17, one blue root finally at rows 9..13 / cols 8..12.
#       Input blue sprite sits at rows 2..6 / cols 9..13. Original task returns only the
#       bar plus placed chains, and pure D7 would crop to roughly rows 8..22. In this
#       combo we also draw a blue line between source-center (4,11) and target-center
#       (11,10), then crop the whole result; now the top boundary moves up near row 4.
#
# complexity_calibration:
#   - The trail uses the sprite's own color and a single deterministic geometry
#     (center-to-center Bresenham line). That is visually strong but still low-branching:
#     there is no color remapping, no alternate path family, and no per-piece mode choice.
#   - The generator keeps pieces on the same side of the bar as their final chain and
#     limits side-to-side drift, so trails read as plausible movement traces rather than
#     arbitrary scribbles. At the same time, sources are pushed far enough away that the
#     crop visibly grows beyond the pure assembly crop.
#
# implicit_assumption_audit:
#   - Original search assumes a fixed 30x30 workspace, absolute scattered piece origins,
#     and full-size occupancy checks. Those assumptions still hold because cropping is
#     postponed until after the full assembled scene and every motion path are known.
#   - Original `best_out` was enough for D7. Here that is insufficient: trail rendering
#     needs the piece-to-target mapping, so the solver must retain final placements, not
#     only the assembled pixels.
#   - The original code implicitly assumed the output depended only on final occupancy.
#     That is false now. Rendering is split into two layers: trails first, then bar and
#     final sprites on top so the chain remains readable even if a line crosses it.
#   - New solver assumption: every intended piece participates in exactly one final
#     placement. The generator guarantees this by constructing complete valid chains and
#     validating `main(input) == expected_output`.
#
# placement_legality_and_visual_cleanliness:
#   - Input placement still enforces non-overlap and no 4-neighbor touching with the bar
#     or other scattered pieces.
#   - Output trails are thin single-pixel lines. The generator rejects candidates whose
#     visible trail cells overlap previous visible trails, preventing an unreadable bundle
#     of same-side lines.
#   - Final sprites are painted after the trails, so the causal story remains legible:
#     scattered source disappears, a colored path shows motion, and the assembled sprite
#     is crisp at the destination.
#
# visual_observability_and_inferability:
#   - Across 2-4 examples, humans can spot two stable cues: the bar/chains are assembled
#     exactly as before, and every moved sprite leaves a same-colored line back toward its
#     input location. The crop always hugs those lines, so the frame is often much larger
#     than the naked assembly crop.
#   - No hidden ordering by color value is introduced. Trails reuse already meaningful
#     sprite colors, and the crop rule depends only on visible occupied cells.
#
# behavior_metric_used_after_testing:
#   - `bbox_expand`: how much farther the combo crop extends than the pure assembly crop.
#   - `visible_trail_cells`: union of trail cells that remain visible after the final bar
#     and sprites are repainted on top.
#   - Samples are accepted only when `bbox_expand >= 4` and `visible_trail_cells >= 8`,
#     which prevents regression to a pure D7 crop or a mostly hidden trail.

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _non_bg_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    return tuple(map(int, bbox_coords(nz)))


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return (int((r0 + r1) // 2), int((c0 + c1) // 2))


def _shape_from_coords(coords, bbox=None):
    arr = np.array(list(coords), dtype=int)
    if bbox is None:
        bbox = tuple(map(int, bbox_coords(arr)))
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in arr]
    return {"rel": rel, "rel_set": set(rel), "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}


def _pixels(shape: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + lr), int(left + lc)) for lr, lc in shape["rel"]}


def _placed_bbox(top: int, left: int, shape: dict) -> tuple[int, int, int, int]:
    return (int(top), int(left), int(top + shape["h"] - 1), int(left + shape["w"] - 1))


def _bresenham_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    pts: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    while True:
        pts.append((int(r0), int(c0)))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return pts


def _bar_notches(bar_mask: np.ndarray, bar_r0: int, bar_c0: int):
    empty = ~bar_mask
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
        hole_gl = {(int(bar_r0 + rr), int(bar_c0 + cc)) for rr, cc in pts}
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        notches.append({"axis": axis, "hole": hole_gl, "mouth_n": int(mouth_n)})
    return notches


def _solve_assembly(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]

    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = int(bar_r1 - bar_r0 + 1), int(bar_c1 - bar_c0 + 1)
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    bar_grid = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_occ:
        bar_mask[int(r - bar_r0), int(c - bar_c0)] = True
        bar_grid[int(r), int(c)] = int(bar_color)

    pieces = []
    for p in pieces_raw:
        pr0, pc0, pr1, pc1 = map(int, p["bbox"])
        pieces.append(
            {
                "color": int(p["color"]),
                "shape": _shape_from_coords(p["coords"], p["bbox"]),
                "orig": (int(pr0), int(pc0)),
                "orig_bbox": (int(pr0), int(pc0), int(pr1), int(pc1)),
            }
        )

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)

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
        pts = _pixels(v, top, left)
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
                        tops.add(int(rN - lr))
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if rN == top + lr:
                        lefts.add(int(cN - lc))
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (int(r + dr), int(c + dc)) in other:
                    return True
        return False

    def hole_positions(hole, v):
        pos = set()
        for rH, cH in hole:
            for lr, lc in v["rel"]:
                pos.add((int(rH - lr), int(cH - lc)))
        return pos

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            v = piece["shape"]
            for top, left in hole_positions(notch["hole"], v):
                pts = placement_pts(v, top, left, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + v["w"] - 1)
                else:
                    A, B = int(top), int(top + v["h"] - 1)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(notch["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": v},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:20]]

    best_state = {"grid": bar_grid.copy(), "placements": {}}
    best_key = None

    def search(out_grid, occ_set, chains, remaining, placements):
        nonlocal best_state, best_key
        placed_cnt = len(placements)
        key = (int(placed_cnt), -int(len(remaining)), -int(len(occ_set)))
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                "grid": out_grid.copy(),
                "placements": {int(k): {"top": int(v["top"]), "left": int(v["left"]), "shape": v["shape"]} for k, v in placements.items()},
            }
        if not remaining:
            return True

        def gen_extend_moves(chain, remaining_set):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], head["top"], head["left"])
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])
            cavity = {
                (int(r), int(c))
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if out_grid[int(r), int(c)] == int(bg)
            }
            if not cavity:
                return []

            moves = []
            for pi in remaining_set:
                v = pieces[pi]["shape"]
                for top, left in aligned_positions(axis, A, cavity, v):
                    pts = placement_pts(v, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None or not touches(pts, head_pts):
                        continue
                    mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                    moves.append((int(mv), int(pi), int(top), int(left), pts, v))
            moves.sort(key=lambda x: x[0])
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for _mv, pi, top, left, pts, v in moves[:8]:
                for r, c in pts:
                    out_grid[int(r), int(c)] = int(pieces[pi]["color"])
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": v}
                placements[int(pi)] = {"top": int(top), "left": int(left), "shape": v}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(int(pi))
                if search(out_grid, occ2, chains, rem2, placements):
                    return True
                del placements[int(pi)]
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[int(r), int(c)] = int(bg)
                for r, c in bar_occ:
                    out_grid[int(r), int(c)] = int(bar_color)
        return False

    from itertools import combinations

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[int(r), int(c)] = int(bar_color)

            chains = []
            placements = {}
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = _pixels(root["shape"], root["top"], root["left"])
                if pts & occ0:
                    ok = False
                    break
                used_p.add(int(root["pi"]))
                used_n.add(int(root["ni"]))
                placements[int(root["pi"])] = {"top": int(root["top"]), "left": int(root["left"]), "shape": root["shape"]}
                for r, c in pts:
                    out0[int(r), int(c)] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": int(root["A"]),
                        "B": int(root["B"]),
                        "head": {"top": int(root["top"]), "left": int(root["left"]), "shape": root["shape"]},
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining, placements)

    return {
        "H": int(H),
        "W": int(W),
        "bg": int(bg),
        "bar_occ": set(bar_occ),
        "bar_color": int(bar_color),
        "pieces": pieces,
        "placements": best_state["placements"],
        "assembled_grid": best_state["grid"],
    }


def _solution_trails(solution: dict) -> dict[int, list[tuple[int, int]]]:
    trails = {}
    for pi, place in solution["placements"].items():
        orig_bbox = solution["pieces"][int(pi)]["orig_bbox"]
        target_bbox = _placed_bbox(place["top"], place["left"], place["shape"])
        trails[int(pi)] = _bresenham_line(_bbox_center(orig_bbox), _bbox_center(target_bbox))
    return trails


def _render_solution(solution: dict) -> np.ndarray:
    H, W, bg = int(solution["H"]), int(solution["W"]), int(solution["bg"])
    out = np.full((H, W), int(bg), dtype=int)
    for pi, trail_pts in _solution_trails(solution).items():
        color = int(solution["pieces"][int(pi)]["color"])
        for r, c in trail_pts:
            if 0 <= int(r) < H and 0 <= int(c) < W:
                out[int(r), int(c)] = int(color)
    for r, c in solution["bar_occ"]:
        out[int(r), int(c)] = int(solution["bar_color"])
    for pi, place in solution["placements"].items():
        for r, c in _pixels(place["shape"], place["top"], place["left"]):
            out[int(r), int(c)] = int(solution["pieces"][int(pi)]["color"])
    return out


def _solution_metrics(solution: dict) -> dict:
    H, W, bg = int(solution["H"]), int(solution["W"]), int(solution["bg"])
    assembled = np.full((H, W), int(bg), dtype=int)
    for r, c in solution["bar_occ"]:
        assembled[int(r), int(c)] = int(solution["bar_color"])
    final_cover = set(solution["bar_occ"])
    for pi, place in solution["placements"].items():
        pts = _pixels(place["shape"], place["top"], place["left"])
        final_cover |= pts
        for r, c in pts:
            assembled[int(r), int(c)] = int(solution["pieces"][int(pi)]["color"])
    assembly_bbox = _non_bg_bbox(assembled, bg)
    combo_full = _render_solution(solution)
    combo_bbox = _non_bg_bbox(combo_full, bg)
    visible_trails = set()
    max_piece_visible = 0
    for trail_pts in _solution_trails(solution).values():
        visible = {pt for pt in trail_pts if pt not in final_cover}
        visible_trails |= visible
        max_piece_visible = max(int(max_piece_visible), int(len(visible)))
    bbox_expand = 0
    if assembly_bbox is not None and combo_bbox is not None:
        bbox_expand = (
            max(0, int(assembly_bbox[0] - combo_bbox[0]))
            + max(0, int(combo_bbox[2] - assembly_bbox[2]))
            + max(0, int(assembly_bbox[1] - combo_bbox[1]))
            + max(0, int(combo_bbox[3] - assembly_bbox[3]))
        )
    return {
        "assembly_bbox": assembly_bbox,
        "combo_bbox": combo_bbox,
        "bbox_expand": int(bbox_expand),
        "visible_trail_cells": int(len(visible_trails)),
        "max_piece_visible_trail": int(max_piece_visible),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the chains exactly as in 16b78196, then draw each piece's same-colored
    straight motion trail from its scattered source bbox to its assembled bbox, and
    finally tight-crop around the whole rendered result.
    """
    solution = _solve_assembly(grid)
    full = _render_solution(solution)
    return _tight_crop_non_bg(full, int(solution["bg"]))


def _connected(cells):
    if len(cells) <= 1:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (int(r + dr), int(c + dc))
            if nb in cells_set and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(cells_set)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [nb for r, c in cells for dr, dc in DIR4 if (nb := (int(r + dr), int(c + dc))) in available and nb not in cells]
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cm = min((c for _, c in cells), default=B)
                    cx = max((c for _, c in cells), default=A)
                    if cm > A:
                        p = [nb for nb in frontier if nb[1] < cm]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                    if cx < B:
                        p = [nb for nb in frontier if nb[1] > cx]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                else:
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
                    if cm > A:
                        p = [nb for nb in frontier if nb[0] < cm]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                    if cx < B:
                        p = [nb for nb in frontier if nb[0] > cx]
                        if p:
                            cells.add(rng.choice(p))
                            continue
            cells.add(rng.choice(frontier))
        if len(cells) < target:
            continue
        while len(cells) > target:
            removable = []
            for rc in cells:
                if rc in seeds:
                    continue
                if must_span_axis and span_range:
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
        if must_span_axis and span_range:
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


def _touches_or_overlaps(pts: set[tuple[int, int]], occ: set[tuple[int, int]]) -> bool:
    for r, c in pts:
        if (int(r), int(c)) in occ:
            return True
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in occ:
                return True
    return False


def _piece_bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(list(coords), dtype=int)
    return tuple(map(int, bbox_coords(arr)))


def _scatter_piece(
    rng: random.Random,
    piece: dict,
    placed_occ: set[tuple[int, int]],
    cover_final: set[tuple[int, int]],
    used_visible_trails: set[tuple[int, int]],
    orientation: str,
    bar_r0: int,
    bar_r1: int,
    bar_c0: int,
    bar_c1: int,
    H: int,
    W: int,
):
    target_bbox = _piece_bbox(piece["coords"])
    shape = _shape_from_coords(piece["coords"], target_bbox)
    h, w = int(shape["h"]), int(shape["w"])
    rel = list(shape["rel"])
    side = piece["side"]

    for _ in range(500):
        gap = rng.randint(3, 8)
        if orientation == "horizontal":
            left = min(max(0, int(target_bbox[1] + rng.randint(-4, 4))), int(W - w))
            if side == "top":
                max_top = int(bar_r0 - gap - h)
                if max_top < 0:
                    continue
                top = rng.randint(0, max_top)
            else:
                min_top = int(bar_r1 + gap + 1)
                if min_top > H - h:
                    continue
                top = rng.randint(min_top, H - h)
        else:
            top = min(max(0, int(target_bbox[0] + rng.randint(-4, 4))), int(H - h))
            if side == "left":
                max_left = int(bar_c0 - gap - w)
                if max_left < 0:
                    continue
                left = rng.randint(0, max_left)
            else:
                min_left = int(bar_c1 + gap + 1)
                if min_left > W - w:
                    continue
                left = rng.randint(min_left, W - w)

        pts = {(int(top + dr), int(left + dc)) for dr, dc in rel}
        if _touches_or_overlaps(pts, placed_occ):
            continue
        src_bbox = (int(top), int(left), int(top + h - 1), int(left + w - 1))
        trail_pts = set(_bresenham_line(_bbox_center(src_bbox), _bbox_center(target_bbox)))
        visible = trail_pts - cover_final
        if len(visible) < 4:
            continue
        if visible & used_visible_trails:
            continue
        return {"pts": pts, "src_bbox": src_bbox, "visible_trail": visible}
    return None


def _gen_attempt():
    rng = random.Random()
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
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = bar_c0 + thickness - 1
        if bar_c1 >= W:
            bar_c1 = W - 1
            bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        space_before, space_after = bar_c0, W - 1 - bar_c1

    bar_h, bar_w = int(bar_r1 - bar_r0 + 1), int(bar_c1 - bar_c0 + 1)
    bar_cells = {(int(r), int(c)) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = rng.randint(1, 2)
    chain_lens = [3] if num_chains == 1 else [rng.randint(2, 3) for _ in range(2)]
    if num_chains == 2:
        while not (3 <= sum(chain_lens) <= 6):
            chain_lens = [rng.randint(2, 3) for _ in range(2)]

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    notch_holes: list[set[tuple[int, int]]] = []
    occ_set = set(bar_cells)

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3), None)
            if side is None:
                return None
            sc0 = rng.randint(1, W - strip_width - 1)
            sc1 = sc0 + strip_width - 1
            sA, sB = sc0, sc1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 3), None)
            if side is None:
                return None
            sr0 = rng.randint(1, H - strip_width - 1)
            sr1 = sr0 + strip_width - 1
            sA, sB = sr0, sr1

        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(int(r), int(c)) for r in range(nr0, nr1 + 1) for c in range(sc0, sc1 + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(int(r), int(c)) for r in range(sA, sB + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        n_avail -= all_holes | {(int(r + dr), int(c + dc)) for r, c in all_holes for dr, dc in DIR4}
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            if color_idx >= len(avail_colors):
                return None
            piece_color = int(avail_colors[color_idx])
            color_idx += 1
            piece_area = rng.randint(8, 13)
            chain_dim = rng.randint(2, 3)

            if piece_idx == 0:
                if orientation == "horizontal":
                    if side == "top":
                        pr0, pr1 = max(0, bar_r0 - chain_dim), max(r for r, _ in notch)
                    else:
                        pr0, pr1 = min(r for r, _ in notch), min(H - 1, bar_r1 + chain_dim)
                    pc0, pc1 = sc0, sc1
                else:
                    if side == "left":
                        pc0, pc1 = max(0, bar_c0 - chain_dim), max(c for _, c in notch)
                    else:
                        pc0, pc1 = min(c for _, c in notch), min(W - 1, bar_c1 + chain_dim)
                    pr0, pr1 = sA, sB

                bbox_area = int((pr1 - pr0 + 1) * (pc1 - pc0 + 1))
                target = max(max(8, int(bbox_area * 0.54)), min(piece_area, int(bbox_area * 0.80)))
                if target > bbox_area or target < 8:
                    return None

                avail = {(int(r), int(c)) for r in range(pr0, pr1 + 1) for c in range(pc0, pc1 + 1) if (r, c) in notch or (r, c) not in occ_set}
                root = _grow_connected(
                    rng,
                    avail,
                    target,
                    notch,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=5,
                )
                if root is None:
                    return None
                if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in root):
                    return None
                piece_cells = root
            else:
                prev_coords = chain_pieces[-1]["coords"]
                prs = [r for r, _ in prev_coords]
                pcs = [c for _, c in prev_coords]
                prev_r0, prev_r1 = min(prs), max(prs)
                prev_c0, prev_c1 = min(pcs), max(pcs)

                if orientation == "horizontal":
                    cavity = {
                        (int(r), int(c))
                        for r in range(prev_r0, prev_r1 + 1)
                        for c in range(sc0, sc1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                else:
                    cavity = {
                        (int(r), int(c))
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                if not cavity:
                    return None

                if orientation == "horizontal":
                    es = (max(0, prev_r0 - chain_dim), prev_r0 - 1) if side == "top" else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    ext = {(int(r), int(c)) for r in range(es[0], es[1] + 1) for c in range(sc0, sc1 + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()
                else:
                    es = (max(0, prev_c0 - chain_dim), prev_c0 - 1) if side == "left" else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    ext = {(int(r), int(c)) for r in range(sA, sB + 1) for c in range(es[0], es[1] + 1) if (r, c) not in occ_set} if es[0] <= es[1] else set()

                all_avail = cavity | ext
                target = min(13, max(len(cavity), min(piece_area, len(all_avail))))
                if len(cavity) > 13 or target < 8:
                    return None

                is_tail = piece_idx == chain_len - 1
                if is_tail:
                    target = min(13, len(all_avail))

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
                        remaining = {(int(r), int(c)) for r in range(t_r0, t_r1 + 1) for c in range(sc0, sc1 + 1) if (r, c) not in occ_set and (r, c) not in piece_cells}
                    else:
                        remaining = {(int(r), int(c)) for r in range(sA, sB + 1) for c in range(t_c0, t_c1 + 1) if (r, c) not in occ_set and (r, c) not in piece_cells}
                    if remaining:
                        return None

            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append({"color": piece_color, "coords": piece_cells, "side": side, "orientation": orientation, "chain_idx": chain_idx})
            occ_set |= piece_cells

        if len(chain_pieces) < chain_len:
            return None
        all_chain_pieces.extend(chain_pieces)

    if not (3 <= len(all_chain_pieces) <= 6):
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

        d_avail = {(int(r), int(c)) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_h = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_h | {(int(r + dr), int(c + dc)) for r, c in all_h for dr, dc in DIR4}
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
        bar_mask[int(r - bar_r0), int(c - bar_c0)] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    input_grid = np.full((H, W), int(bg), dtype=int)
    for r, c in bar_cells:
        input_grid[int(r), int(c)] = int(bar_color)

    cover_final = set(bar_cells)
    for piece in all_chain_pieces:
        cover_final |= set(piece["coords"])

    placed_occ = set(bar_cells)
    used_visible_trails: set[tuple[int, int]] = set()
    source_bboxes: dict[int, tuple[int, int, int, int]] = {}

    order = list(range(len(all_chain_pieces)))
    rng.shuffle(order)
    order.sort(key=lambda idx: -len(all_chain_pieces[idx]["coords"]))
    for idx in order:
        placement = _scatter_piece(
            rng,
            all_chain_pieces[idx],
            placed_occ,
            cover_final,
            used_visible_trails,
            orientation,
            bar_r0,
            bar_r1,
            bar_c0,
            bar_c1,
            H,
            W,
        )
        if placement is None:
            return None
        source_bboxes[int(idx)] = placement["src_bbox"]
        used_visible_trails |= placement["visible_trail"]
        for r, c in placement["pts"]:
            input_grid[int(r), int(c)] = int(all_chain_pieces[idx]["color"])
        placed_occ |= placement["pts"]

    gen_pieces = []
    placements = {}
    for idx, piece in enumerate(all_chain_pieces):
        target_bbox = _piece_bbox(piece["coords"])
        shape = _shape_from_coords(piece["coords"], target_bbox)
        gen_pieces.append({"color": int(piece["color"]), "orig_bbox": source_bboxes[idx], "shape": shape})
        placements[int(idx)] = {"top": int(target_bbox[0]), "left": int(target_bbox[1]), "shape": shape}

    solution = {
        "H": int(H),
        "W": int(W),
        "bg": int(bg),
        "bar_occ": set(bar_cells),
        "bar_color": int(bar_color),
        "pieces": gen_pieces,
        "placements": placements,
    }
    metrics = _solution_metrics(solution)
    if metrics["bbox_expand"] < 4:
        return None
    if metrics["visible_trail_cells"] < 8:
        return None
    if metrics["max_piece_visible_trail"] < 3:
        return None
    if metrics["assembly_bbox"] == metrics["combo_bbox"]:
        return None

    output = _tight_crop_non_bg(_render_solution(solution), bg)
    return input_grid, output


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 500 attempts")
