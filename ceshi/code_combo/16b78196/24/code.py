# Combo variant of 16b78196 (index 24)
# combo_spec: D3/2+D4/2
# Name: 嵌套框架+颜色排序链
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Original task: a single horizontal/vertical bar contains one-layer edge notches;
#    a root piece plugs one bar notch, then later pieces fill the previous piece's bbox
#    cavity while staying on the same strip.
#    Fused task: the scaffold is now two concentric rectangular frames with the same
#    color. Each chain starts at an outer-frame hole, then the next piece must also
#    cover a matched inner-frame hole on the same side, and only then can the last
#    piece continue inward. Example on the top side:
#      - root plugs the top hole of the outer frame
#      - middle fills the root cavity and simultaneously bridges the top hole of the
#        inner frame
#      - tail fills the middle cavity inside the inner frame
#    The difference from the original is not just "bar -> frame": the target now has
#    a visible outer/inner hierarchy and the second step is constrained by a second
#    layer-specific hole.
#
# b) Complexity calibration:
#    Two chains are always present, on opposite sides of the nested frames. Their strip
#    widths are 4 and 5, which keeps them visually distinct and reduces accidental
#    cross-matching. Each chain still uses the cavity-fill mechanic, so the puzzle is
#    not reduced to simple placement into two rectangular slots.
#
# c) Hidden assumptions broken from the original solver:
#    - "largest component = one full-span bar" is false; now the scaffold is two
#      disconnected ring components of the same color.
#    - "all notches live on one object/layer" is false; there is an outer-hole stage
#      and an inner-hole stage.
#    - "any fitting extension is valid" is false; the chain must be brightness-ordered
#      from outer to inner.
#    The new solver therefore detects two frame components, reconstructs their ideal
#    rectangular-ring masks, pairs holes by side, and only accepts root/middle/tail
#    triples with strictly increasing visible brightness.
#
# d) Placement legality:
#    The generator constructs complete frame rectangles first, then removes exactly the
#    cells that will be covered by the outer-hole plug and by the rectangular inner
#    hole. Pieces are scattered only in the exterior side bands around the outer frame,
#    with 8-neighborhood exclusion, so nothing overlaps or touches illegally.
#
# e) Visual observability:
#    The ordering cue is based on visible palette brightness, not on hidden numeric ids.
#    Only a few fixed dark->bright triples are used, such as maroon->orange->yellow or
#    blue->grey->teal, and the same order repeats on both chains inside one sample.
#
# f) Visual cleanliness:
#    The frames stay centered and axis-aligned, holes appear on only two opposite sides,
#    and scattered pieces live in clean side bands outside the outer frame. This keeps
#    the nested scaffold readable.
#
# g) Inferability of the fused change:
#    Across samples, a human sees two concentric frames, two opposite-side chains, and
#    the same outer->inner brightness trend repeated twice. That provides both the
#    hierarchical clue (outer frame before inner frame) and the ordering clue.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import ARC_RGB, components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_THICKNESS = 2
OUTER_BBOX = (6, 6, 23, 23)
INNER_BBOX = (10, 10, 19, 19)
SIDE_ORDER = ["top", "bottom", "left", "right"]
BRIGHTNESS_TRIPLES = [
    (9, 7, 4),  # maroon -> orange -> yellow
    (1, 5, 8),  # blue -> grey -> teal
    (2, 3, 4),  # red -> green -> yellow
    (9, 5, 8),  # maroon -> grey -> teal
]


def _brightness(color: int) -> float:
    rgb = ARC_RGB[int(color)]
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


def _parse_piece(lines: list[str]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in lines], dtype=np.int8)


BASE_TEMPLATES = {
    4: [
        _parse_piece(
            [
                "#...",
                "##..",
                "####",
                "##.#",
            ]
        ),
        _parse_piece(
            [
                "..#.",
                "####",
                "####",
                ".#..",
            ]
        ),
        _parse_piece(
            [
                "#.##",
                "####",
            ]
        ),
    ],
    5: [
        _parse_piece(
            [
                "#....",
                "##...",
                "#####",
                "#.###",
            ]
        ),
        _parse_piece(
            [
                ".#...",
                "#####",
                "#####",
                ".#...",
            ]
        ),
        _parse_piece(
            [
                "#.###",
                "#####",
            ]
        ),
    ],
}


def _rot90_cw(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=-1)


def _rot90_ccw(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=1)


def _rot180(piece: np.ndarray) -> np.ndarray:
    return np.rot90(piece, k=2)


def _rotate_piece_for_side(piece: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return piece.copy()
    if side == "bottom":
        return _rot180(piece)
    if side == "left":
        return _rot90_ccw(piece)
    if side == "right":
        return _rot90_cw(piece)
    raise ValueError(side)


def _shape_from_piece(piece: np.ndarray) -> dict[str, object]:
    coords = np.argwhere(piece == 1)
    h, w = map(int, piece.shape)
    rel = [(int(r), int(c)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _shape_from_coords(coords: np.ndarray, bbox) -> dict[str, object]:
    r0, c0, r1, c1 = map(int, bbox)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}


def _pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + lr), int(left + lc)) for lr, lc in shape["rel"]}  # type: ignore[index]


def _axis_for_side(side: str) -> str:
    return "cols" if side in ("top", "bottom") else "rows"


def _strip_bounds(axis: str, shape: dict[str, object], top: int, left: int) -> tuple[int, int]:
    if axis == "cols":
        return int(left), int(left + int(shape["w"]) - 1)
    return int(top), int(top + int(shape["h"]) - 1)


def _placement_pts(
    shape: dict[str, object],
    top: int,
    left: int,
    H: int,
    W: int,
    occ: set[tuple[int, int]],
    must_cover: set[tuple[int, int]] | None = None,
    strip: tuple[str, int, int] | None = None,
) -> set[tuple[int, int]] | None:
    h = int(shape["h"])
    w = int(shape["w"])
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return None
    if strip is not None:
        axis, A, B = strip
        if axis == "cols":
            if left != A or w != B - A + 1:
                return None
        else:
            if top != A or h != B - A + 1:
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
        for r_need, c_need in cover:
            for lr, lc in shape["rel"]:  # type: ignore[index]
                if c_need == left + int(lc):
                    tops.add(int(r_need - lr))
        for top in sorted(tops):
            yield top, left
    else:
        top = int(A)
        lefts: set[int] = set()
        for r_need, c_need in cover:
            for lr, lc in shape["rel"]:  # type: ignore[index]
                if r_need == top + int(lr):
                    lefts.add(int(c_need - lc))
        for left in sorted(lefts):
            yield top, left


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _bbox_cavity(
    top: int,
    left: int,
    shape: dict[str, object],
    occ: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h = int(shape["h"])
    w = int(shape["w"])
    return {
        (r, c)
        for r in range(top, top + h)
        for c in range(left, left + w)
        if (r, c) not in occ
    }


def _hole_positions(hole: set[tuple[int, int]], shape: dict[str, object]):
    pos: set[tuple[int, int]] = set()
    for r_hole, c_hole in hole:
        for lr, lc in shape["rel"]:  # type: ignore[index]
            pos.add((int(r_hole - lr), int(c_hole - lc)))
    return pos


def _bbox_from_shape(top: int, left: int, shape: dict[str, object]) -> tuple[int, int, int, int]:
    return int(top), int(left), int(top + int(shape["h"]) - 1), int(left + int(shape["w"]) - 1)


def _progress_ok(side: str, prev_bbox, next_bbox) -> bool:
    pr0, pc0, pr1, pc1 = prev_bbox
    nr0, nc0, nr1, nc1 = next_bbox
    if side == "top":
        return nr0 >= pr0 and nr1 > pr1
    if side == "bottom":
        return nr0 < pr0 and nr1 <= pr1
    if side == "left":
        return nc0 >= pc0 and nc1 > pc1
    if side == "right":
        return nc0 < pc0 and nc1 <= pc1
    return False


def _frame_mask(H: int, W: int, bbox, thickness: int = FRAME_THICKNESS) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros((H, W), dtype=bool)
    mask[r0 : r0 + thickness, c0 : c1 + 1] = True
    mask[r1 - thickness + 1 : r1 + 1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0 : c0 + thickness] = True
    mask[r0 : r1 + 1, c1 - thickness + 1 : c1 + 1] = True
    return mask


def _frame_cells(H: int, W: int, bbox, thickness: int = FRAME_THICKNESS) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(_frame_mask(H, W, bbox, thickness))}


def _rect_hole(side: str, bbox, start: int, width: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    if side == "top":
        return {(r, c) for r in range(r0, r0 + FRAME_THICKNESS) for c in range(start, start + width)}
    if side == "bottom":
        return {(r, c) for r in range(r1 - FRAME_THICKNESS + 1, r1 + 1) for c in range(start, start + width)}
    if side == "left":
        return {(r, c) for r in range(start, start + width) for c in range(c0, c0 + FRAME_THICKNESS)}
    if side == "right":
        return {(r, c) for r in range(start, start + width) for c in range(c1 - FRAME_THICKNESS + 1, c1 + 1)}
    raise ValueError(side)


def _root_anchor(side: str, bbox, shape: dict[str, object], start: int) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    h = int(shape["h"])
    w = int(shape["w"])
    if side == "top":
        return int(r0), int(start)
    if side == "bottom":
        return int(r1 - h + 1), int(start)
    if side == "left":
        return int(start), int(c0)
    if side == "right":
        return int(start), int(c1 - w + 1)
    raise ValueError(side)


def _detect_frame_components(grid: np.ndarray, bg: int):
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)
    color_rank = sorted(
        ((sum(int(comp["area"]) for comp in color_comps), int(color)) for color, color_comps in by_color.items()),
        reverse=True,
    )
    if len(color_rank) < 2:
        return None
    frame_colors = [color_rank[0][1], color_rank[1][1]]

    frame_groups = []
    for color in frame_colors:
        color_comps = by_color[int(color)]
        coords = np.vstack([comp["coords"] for comp in color_comps])
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        frame_groups.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": int(coords.shape[0]),
            }
        )

    frame_groups.sort(key=lambda d: -((d["bbox"][2] - d["bbox"][0] + 1) * (d["bbox"][3] - d["bbox"][1] + 1)))
    pieces_raw = [comp for comp in comps if int(comp["color"]) not in set(frame_colors)]
    return frame_groups, pieces_raw


def _detect_frame_holes(frame_comp: dict, H: int, W: int):
    bbox = frame_comp["bbox"]
    r0, c0, r1, c1 = map(int, bbox)
    local_h = r1 - r0 + 1
    local_w = c1 - c0 + 1
    actual = np.zeros((local_h, local_w), dtype=bool)
    for r, c in frame_comp["coords"]:
        actual[int(r - r0), int(c - c0)] = True
    ideal = _frame_mask(local_h, local_w, (0, 0, local_h - 1, local_w - 1))
    missing = ideal & ~actual
    holes = []
    for pts in components_mask(missing, connectivity=4):
        gl = {(int(r0 + rr), int(c0 + cc)) for rr, cc in pts}
        rr = pts[:, 0]
        cc = pts[:, 1]
        counts = {
            "top": int((rr == 0).sum()),
            "bottom": int((rr == local_h - 1).sum()),
            "left": int((cc == 0).sum()),
            "right": int((cc == local_w - 1).sum()),
        }
        side = max((name for name in SIDE_ORDER if counts[name] > 0), key=lambda name: counts[name], default=None)
        if side is None:
            continue
        holes.append({"side": side, "hole": gl, "size": len(gl)})
    return holes


def _build_piece_records(pieces_raw: list[dict]):
    pieces = []
    for comp in pieces_raw:
        r0, c0 = map(int, comp["bbox"][:2])
        color = int(comp["color"])
        pieces.append(
            {
                "color": color,
                "brightness": _brightness(color),
                "shape": _shape_from_coords(comp["coords"], comp["bbox"]),
                "orig": (r0, c0),
            }
        )
    return pieces


def _extract_scene(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    detected = _detect_frame_components(g, bg)
    if detected is None:
        return None
    frame_comps, pieces_raw = detected
    outer_comp, inner_comp = frame_comps

    outer_holes = {h["side"]: h["hole"] for h in _detect_frame_holes(outer_comp, H, W)}
    inner_holes = {h["side"]: h["hole"] for h in _detect_frame_holes(inner_comp, H, W)}
    shared_sides = [side for side in SIDE_ORDER if side in outer_holes and side in inner_holes]
    if not shared_sides:
        return None

    frame_grid = np.full_like(g, bg)
    frame_occ: set[tuple[int, int]] = set()
    for comp in frame_comps:
        color = int(comp["color"])
        for r, c in comp["coords"]:
            rr, cc = int(r), int(c)
            frame_grid[rr, cc] = color
            frame_occ.add((rr, cc))

    chain_specs = [
        {
            "side": side,
            "axis": _axis_for_side(side),
            "outer_hole": set(outer_holes[side]),
            "inner_hole": set(inner_holes[side]),
        }
        for side in shared_sides
    ]
    pieces = _build_piece_records(pieces_raw)
    return {
        "grid": g,
        "bg": bg,
        "frame_colors": tuple(int(comp["color"]) for comp in frame_comps),
        "frame_grid": frame_grid,
        "frame_occ": frame_occ,
        "chain_specs": chain_specs,
        "pieces": pieces,
    }


def _chain_solutions_for_spec(scene: dict, spec: dict):
    H, W = scene["grid"].shape
    bg = int(scene["bg"])
    frame_occ = set(scene["frame_occ"])
    pieces = scene["pieces"]
    side = str(spec["side"])
    axis = str(spec["axis"])
    outer_hole = set(spec["outer_hole"])
    inner_hole = set(spec["inner_hole"])
    sols = []

    for i, root in enumerate(pieces):
        root_shape = root["shape"]
        for top0, left0 in sorted(_hole_positions(outer_hole, root_shape)):
            pts0 = _placement_pts(root_shape, top0, left0, H, W, frame_occ, must_cover=outer_hole)
            if pts0 is None:
                continue
            A, B = _strip_bounds(axis, root_shape, top0, left0)
            occ1 = frame_occ | pts0
            cavity0 = _bbox_cavity(top0, left0, root_shape, occ1)
            if not cavity0:
                continue
            bbox0 = _bbox_from_shape(top0, left0, root_shape)

            for j, mid in enumerate(pieces):
                if j == i or float(mid["brightness"]) <= float(root["brightness"]):
                    continue
                mid_shape = mid["shape"]
                cover1 = set(cavity0) | set(inner_hole)
                for top1, left1 in _aligned_positions(axis, A, cover1, mid_shape):
                    pts1 = _placement_pts(mid_shape, top1, left1, H, W, occ1, must_cover=cover1, strip=(axis, A, B))
                    if pts1 is None or not _touches(pts1, pts0):
                        continue
                    bbox1 = _bbox_from_shape(top1, left1, mid_shape)
                    if not _progress_ok(side, bbox0, bbox1):
                        continue
                    occ2 = occ1 | pts1
                    cavity1 = _bbox_cavity(top1, left1, mid_shape, occ2)
                    if not cavity1:
                        continue

                    for k, tail in enumerate(pieces):
                        if k in {i, j} or float(tail["brightness"]) <= float(mid["brightness"]):
                            continue
                        tail_shape = tail["shape"]
                        for top2, left2 in _aligned_positions(axis, A, cavity1, tail_shape):
                            pts2 = _placement_pts(tail_shape, top2, left2, H, W, occ2, must_cover=cavity1, strip=(axis, A, B))
                            if pts2 is None or not _touches(pts2, pts1):
                                continue
                            bbox2 = _bbox_from_shape(top2, left2, tail_shape)
                            if not _progress_ok(side, bbox1, bbox2):
                                continue
                            move = (
                                abs(top0 - int(root["orig"][0])) + abs(left0 - int(root["orig"][1]))
                                + abs(top1 - int(mid["orig"][0])) + abs(left1 - int(mid["orig"][1]))
                                + abs(top2 - int(tail["orig"][0])) + abs(left2 - int(tail["orig"][1]))
                            )
                            sols.append(
                                {
                                    "side": side,
                                    "pieces": (i, j, k),
                                    "placements": (
                                        (i, top0, left0, pts0),
                                        (j, top1, left1, pts1),
                                        (k, top2, left2, pts2),
                                    ),
                                    "score": (
                                        -move,
                                        len(outer_hole),
                                        len(inner_hole),
                                        -(A + B),
                                    ),
                                }
                            )
    sols.sort(key=lambda s: s["score"], reverse=True)
    return sols


def _solve_scene(scene: dict) -> np.ndarray | None:
    chain_specs = scene["chain_specs"]
    solutions_by_chain = [_chain_solutions_for_spec(scene, spec) for spec in chain_specs]
    if any(not sols for sols in solutions_by_chain):
        return None

    best = None

    def rec(idx: int, used: set[int], chosen: list[dict]):
        nonlocal best
        if idx == len(solutions_by_chain):
            best = list(chosen)
            return True
        for sol in solutions_by_chain[idx]:
            piece_ids = set(sol["pieces"])
            if piece_ids & used:
                continue
            chosen.append(sol)
            if rec(idx + 1, used | piece_ids, chosen):
                return True
            chosen.pop()
        return False

    rec(0, set(), [])
    if best is None:
        return None

    out = np.array(scene["frame_grid"], dtype=int)
    pieces = scene["pieces"]
    for sol in best:
        for piece_idx, _top, _left, pts in sol["placements"]:
            color = int(pieces[piece_idx]["color"])
            for r, c in pts:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    scene = _extract_scene(np.array(grid, dtype=int))
    if scene is None:
        return np.array(grid, dtype=int)
    solved = _solve_scene(scene)
    if solved is None:
        return np.array(scene["frame_grid"], dtype=int)
    return solved


def fusion_signature(grid: np.ndarray) -> dict[str, object]:
    scene = _extract_scene(np.array(grid, dtype=int))
    if scene is None:
        return {"layers": 0, "chains": 0, "ordered_chains": 0, "sides": (), "widths": ()}
    chain_specs = scene["chain_specs"]
    solved = _solve_scene(scene)
    ordered = 0
    widths = []
    if solved is not None:
        solutions_by_chain = [_chain_solutions_for_spec(scene, spec) for spec in chain_specs]
        chosen: list[dict] = []
        used: set[int] = set()
        for sols in solutions_by_chain:
            for sol in sols:
                if set(sol["pieces"]) & used:
                    continue
                chosen.append(sol)
                used |= set(sol["pieces"])
                break
        for sol in chosen:
            cols = [int(scene["pieces"][idx]["color"]) for idx in sol["pieces"]]
            br = [_brightness(c) for c in cols]
            if br[0] < br[1] < br[2]:
                ordered += 1
            first = sol["placements"][0]
            shape = scene["pieces"][first[0]]["shape"]
            axis = _axis_for_side(str(sol["side"]))
            A, B = _strip_bounds(axis, shape, int(first[1]), int(first[2]))
            widths.append(int(B - A + 1))
    return {
        "layers": 2,
        "chains": len(chain_specs),
        "ordered_chains": ordered,
        "sides": tuple(str(spec["side"]) for spec in chain_specs),
        "widths": tuple(sorted(widths)),
    }


def _side_band_regions():
    return [
        (0, 0, 6, 30),
        (24, 0, 30, 30),
        (6, 0, 24, 6),
        (6, 24, 24, 30),
    ]


def _can_place_piece(mask: np.ndarray, piece: np.ndarray, top: int, left: int) -> bool:
    H, W = mask.shape
    h, w = piece.shape
    for pr in range(h):
        for pc in range(w):
            if piece[pr, pc] != 1:
                continue
            rr, cc = top + pr, left + pc
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                        return False
    return True


def _place_piece_in_region(
    grid: np.ndarray,
    placed_mask: np.ndarray,
    piece: np.ndarray,
    color: int,
    region: tuple[int, int, int, int],
) -> bool:
    y0, x0, y1, x1 = map(int, region)
    h, w = piece.shape
    max_top = y1 - h
    max_left = x1 - w
    if max_top < y0 or max_left < x0:
        return False
    candidates = [(top, left) for top in range(y0, max_top + 1) for left in range(x0, max_left + 1)]
    np.random.shuffle(candidates)
    for top, left in candidates:
        if not _can_place_piece(placed_mask, piece, int(top), int(left)):
            continue
        for pr in range(h):
            for pc in range(w):
                if piece[pr, pc] == 1:
                    grid[top + pr, left + pc] = int(color)
                    placed_mask[top + pr, left + pc] = True
        return True
    return False


def _pick_colors() -> tuple[tuple[int, int], tuple[int, int, int]]:
    triple_options = list(BRIGHTNESS_TRIPLES)
    np.random.shuffle(triple_options)
    for triple in triple_options:
        remaining = [c for c in range(1, 10) if c not in triple]
        if len(remaining) >= 2:
            picked = np.random.choice(remaining, size=2, replace=False)
            frame_colors = tuple(int(c) for c in picked)
            return frame_colors, tuple(int(c) for c in triple)
    raise RuntimeError("no color triple available")


def _build_chain(side: str, width: int, start: int, H: int, W: int):
    outer_cells = _frame_cells(H, W, OUTER_BBOX)
    inner_cells = _frame_cells(H, W, INNER_BBOX)
    frame_full = outer_cells | inner_cells

    pieces = [_rotate_piece_for_side(piece, side) for piece in BASE_TEMPLATES[width]]
    shapes = [_shape_from_piece(piece) for piece in pieces]
    root_top, root_left = _root_anchor(side, OUTER_BBOX, shapes[0], start)
    root_pts = _pixels(shapes[0], root_top, root_left)
    outer_hole = root_pts & outer_cells
    inner_hole = _rect_hole(side, INNER_BBOX, start, width)
    occ0 = frame_full - outer_hole - inner_hole
    if root_pts & occ0:
        return None
    occ1 = occ0 | root_pts
    axis = _axis_for_side(side)
    A, B = _strip_bounds(axis, shapes[0], root_top, root_left)
    cavity0 = _bbox_cavity(root_top, root_left, shapes[0], occ1)

    mids = []
    for top1, left1 in _aligned_positions(axis, A, cavity0 | inner_hole, shapes[1]):
        pts1 = _placement_pts(shapes[1], top1, left1, H, W, occ1, must_cover=cavity0 | inner_hole, strip=(axis, A, B))
        if pts1 is None or not _touches(pts1, root_pts):
            continue
        mids.append((top1, left1, pts1))

    for top1, left1, pts1 in mids:
        occ2 = occ1 | pts1
        cavity1 = _bbox_cavity(top1, left1, shapes[1], occ2)
        for top2, left2 in _aligned_positions(axis, A, cavity1, shapes[2]):
            pts2 = _placement_pts(shapes[2], top2, left2, H, W, occ2, must_cover=cavity1, strip=(axis, A, B))
            if pts2 is None or not _touches(pts2, pts1):
                continue
            return {
                "side": side,
                "width": width,
                "start": start,
                "pieces": pieces,
                "placements": [(root_top, root_left, root_pts), (top1, left1, pts1), (top2, left2, pts2)],
                "outer_hole": outer_hole,
                "inner_hole": inner_hole,
            }
    return None


def _render_frame_only(H: int, W: int, outer_color: int, inner_color: int, chains: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    outer_mask = _frame_mask(H, W, OUTER_BBOX)
    inner_mask = _frame_mask(H, W, INNER_BBOX)
    grid[outer_mask] = int(outer_color)
    grid[inner_mask] = int(inner_color)
    for chain in chains:
        for r, c in chain["outer_hole"] | chain["inner_hole"]:
            grid[int(r), int(c)] = 0
    return grid


def _render_output(frame_only: np.ndarray, chains: list[dict], chain_colors: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.array(frame_only, dtype=int)
    for chain, colors in zip(chains, chain_colors):
        for (top, left, pts), color in zip(chain["placements"], colors):
            _ = top
            _ = left
            for r, c in pts:
                out[int(r), int(c)] = int(color)
    return out


def _try_generate_input() -> tuple[np.ndarray, np.ndarray] | None:
    H = W = 30
    side_pairs = [
        ["top", "bottom"],
        ["left", "right"],
    ]
    sides = side_pairs[int(np.random.randint(0, len(side_pairs)))]

    widths = [4, 5]
    if bool(np.random.randint(0, 2)):
        widths.reverse()
    starts = [12, 13]

    (outer_color, inner_color), triple = _pick_colors()
    chain_colors = [triple, triple]
    chains = []
    for side, width, start in zip(sides, widths, starts):
        chain = _build_chain(side, int(width), int(start), H, W)
        if chain is None:
            return None
        chains.append(chain)

    frame_only = _render_frame_only(H, W, outer_color, inner_color, chains)
    out = _render_output(frame_only, chains, chain_colors)

    inp = np.array(frame_only, dtype=int)
    placed_mask = inp != 0
    pieces_to_scatter: list[tuple[np.ndarray, int]] = []
    for chain, colors in zip(chains, chain_colors):
        for piece, color in zip(chain["pieces"], colors):
            pieces_to_scatter.append((np.array(piece, dtype=np.int8), int(color)))
    np.random.shuffle(pieces_to_scatter)

    regions = _side_band_regions()
    region_order = list(range(len(regions)))
    np.random.shuffle(region_order)
    for idx, (piece, color) in enumerate(pieces_to_scatter):
        placed = False
        for offset in range(len(regions)):
            region = regions[region_order[(idx + offset) % len(regions)]]
            if _place_piece_in_region(inp, placed_mask, piece, color, region):
                placed = True
                break
        if not placed:
            return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    sig = fusion_signature(inp)
    if not (sig["layers"] == 2 and sig["chains"] == 2 and sig["ordered_chains"] == 2):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(200):
        result = _try_generate_input()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("generate_input: failed after 200 attempts")
