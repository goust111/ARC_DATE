# Combo variant of 67e490f4 (index 10)
# combo_spec: D1/4+D10/0
# Name: 连接器配对+完整网格输出

# Design analysis for this combo variant
# a) Concrete behavior and trace
#    - Original task: crop to the frame bbox, find outside sprites by shape, then fill every same-shaped
#      hole with the majority sprite color for that shape.
#    - Combo behavior: every hole instance has its own colored connector path. The path starts on a
#      frame cell adjacent to the hole, tunnels through the frame, continues across the outer background,
#      and ends next to exactly one outside sprite. The output keeps the full 30x30 board: sprites and
#      connectors vanish, frame tunnels heal back to frame color, and only the hole cells get filled.
#    - Small trace: if two 2x2 holes have the same shape but one red sprite is linked to the left hole
#      and one green sprite is linked to the right hole, the original solver would collapse them by
#      shape; the combo solver instead follows each connector separately and fills the two identical
#      holes with different colors.
# b) Complexity calibration
#    - The task is still object-centric: a large frame, several small holes, several outside sprites.
#    - Complexity stays visible because at least one canonical hole shape is repeated with different
#      paired sprite colors, so connector tracing is necessary and the puzzle does not degenerate into
#      simple per-shape voting.
#    - Freedom is controlled: connectors are single-color, 4-connected, non-branching paths; sprites
#      remain monochrome and separated; hole count stays modest so the board remains readable.
# c) Hidden-assumption review of the original main()
#    - "Hole = any non-frame component inside the bbox" is no longer sufficient because each hole is
#      attached to a connector tail. The new solver treats each bbox component as "hole + tail" and
#      uses only the background-colored subset as the true fill mask.
#    - "Outside sprites can be read by blanking the frame bbox and grouping by shape" is false now
#      because sprites are fused to connector lines. The new solver traces a connector color component
#      from the frame tail out to the sprite instead.
#    - "Output canvas equals the frame crop" is false because D10 keeps the entire 30x30 board.
#    - "All same-shaped holes share one chosen color" is false because D1/4 introduces instance-level
#      pairing, so same-shaped holes may legitimately receive different fills.
# d) Placement legality
#    - Holes are placed strictly inside the frame border and remain 4-disconnected.
#    - Each connector path is routed with BFS through frame cells and outer background cells while
#      avoiding other holes, existing connectors, and existing sprites.
#    - Sprites are placed fully outside the frame bbox with padding so distinct connector-sprite
#      systems do not touch or merge.
# e) Visual observability
#    - The full-grid output makes the "sprites disappear, lines disappear, frame stays" transform
#      directly visible.
#    - Connectors are human-visible colored paths, so the pairing cue is explicit in 2-3 examples.
#    - Repeated hole shapes with different linked sprite colors make the connector logic observable
#      and distinguishable from the original shape-majority rule.
# f) Visual cleanliness
#    - Each hole gets exactly one thin connector, no branches.
#    - Components are spaced apart; no overlapping overlays are used.
#    - The frame region remains the dominant visual structure, so the added lines read as annotations
#      rather than clutter.
# g) Color inferability
#    - Sprite color is the fill color. Connector color only indicates the path; it is never reused as
#      an output fill for that same pair.
#    - The generator deliberately creates at least one repeated-shape / multi-color situation, so the
#      solver cannot succeed by relying on hidden color ordering or silent tie-breakers.
# h) Variant metric used during validation
#    - Count repeated canonical hole shapes whose linked sprites use more than one fill color. A sample
#      is considered connector-dependent when this count is positive, because a per-shape fill rule
#      would disagree.
# i) New solver assumptions introduced by the combo
#    - Each hole+tail component inside the frame contains exactly one connector color and at least one
#      background hole cell.
#    - Following that connector color yields exactly one 4-connected path whose external neighborhood
#      exposes exactly one sprite color.
#    - The generator guarantees these assumptions by routing one path per hole and keeping components
#      separated.

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (9, 13)
FRAME_W_RANGE = (10, 18)
FRAME_MARGIN = 3

HOLE_COUNT_RANGE = (4, 5)
CANONICAL_SHAPE_RANGE = (2, 4)
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.68, 0.9)
HOLE_PLACEMENT_RETRIES = 120
SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 300
MAX_HOLE_BORDER_DISTANCE = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m.copy())
    return result


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 0]],
        [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
    ]
    seen, dedup = set(), []
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        if int(np.sum(mask)) > SHAPE_MAX_AREA:
            continue
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()
SIDES = ("top", "bottom", "left", "right")


def _find_frame(g: np.ndarray, bg: int) -> dict:
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    candidates = []
    for comp in comps:
        bb = comp["bbox"]
        sub = crop_bbox(g, bb)
        bbox_area = int(sub.shape[0] * sub.shape[1])
        density = float(comp["area"]) / float(bbox_area)
        if not np.any(sub == int(bg)):
            continue
        candidates.append((density, int(comp["area"]), comp))
    assert candidates
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _trace_color_component(g: np.ndarray, seeds: list[tuple[int, int]], color: int) -> set[tuple[int, int]]:
    H, W = g.shape
    target = int(color)
    todo = deque()
    seen: set[tuple[int, int]] = set()
    for r, c in seeds:
        rr, cc = int(r), int(c)
        assert 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == target
        todo.append((rr, cc))
        seen.add((rr, cc))

    while todo:
        r, c = todo.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and int(g[nr, nc]) == target:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return seen


def _parse_hole_components(g: np.ndarray):
    """Extract frame-local hole+tunnel components and recover their pure hole masks."""
    bg = int(get_bg(g))
    frame = _find_frame(g, bg)
    frame_col = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])
    sub = crop_bbox(g, frame["bbox"])
    comps = components_mask(sub != frame_col, connectivity=4)

    holes = []
    for comp_coords in comps:
        vals = [int(v) for v in sub[comp_coords[:, 0], comp_coords[:, 1]]]
        hole_coords = comp_coords[np.array(vals) == bg]
        assert hole_coords.size > 0
        line_colors = sorted({int(v) for v in vals if int(v) != bg})
        assert len(line_colors) == 1
        line_color = int(line_colors[0])

        hr0, hc0, hr1, hc1 = bbox_coords(hole_coords)
        hh = int(hr1 - hr0 + 1)
        hw = int(hc1 - hc0 + 1)
        rel = hole_coords - np.array([hr0, hc0])
        hole_mask = np.zeros((hh, hw), dtype=bool)
        hole_mask[rel[:, 0], rel[:, 1]] = True

        line_seeds_local = [
            (int(r), int(c))
            for r, c in comp_coords
            if int(sub[int(r), int(c)]) == line_color
        ]
        assert line_seeds_local

        holes.append(
            {
                "hole_bbox": (int(hr0), int(hc0), int(hr1), int(hc1)),
                "hole_mask": hole_mask,
                "canonical_key": _canonical_key(hole_mask),
                "line_color": line_color,
                "line_seeds_global": [(r0 + lr, c0 + lc) for lr, lc in line_seeds_local],
            }
        )

    return {
        "bg": bg,
        "frame_col": frame_col,
        "frame_bbox": (r0, c0, r1, c1),
        "frame_shape": sub.shape,
        "holes": holes,
    }


def _extract_pairings(g: np.ndarray):
    """Recover frame info plus one fill color for every hole instance."""
    info = _parse_hole_components(g)
    bg = int(info["bg"])
    frame_col = int(info["frame_col"])
    H, W = g.shape
    pairings = []

    for hole in info["holes"]:
        path = _trace_color_component(g, hole["line_seeds_global"], hole["line_color"])
        adj_colors = set()
        for r, c in path:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < H and 0 <= nc < W:
                    val = int(g[nr, nc])
                    if val not in {bg, frame_col, int(hole["line_color"])}:
                        adj_colors.add(val)
        assert len(adj_colors) == 1
        pairings.append({**hole, "fill_color": next(iter(adj_colors)), "line_len": len(path)})

    info["pairings"] = pairings
    return info


def _solve_full_grid(g: np.ndarray) -> np.ndarray:
    info = _extract_pairings(g)
    bg = int(info["bg"])
    frame_col = int(info["frame_col"])
    r0, c0, _r1, _c1 = info["frame_bbox"]
    th, tw = map(int, info["frame_shape"])

    solved_frame = np.full((th, tw), frame_col, dtype=int)
    for pair in info["pairings"]:
        hr0, hc0, _hr1, _hc1 = pair["hole_bbox"]
        patch = np.full(pair["hole_mask"].shape, int(pair["fill_color"]), dtype=int)
        paste(solved_frame, patch, int(hr0), int(hc0), mask=pair["hole_mask"])

    out = np.full(g.shape, bg, dtype=int)
    paste(out, solved_frame, r0, c0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill frame holes by tracing connector-linked sprites, then keep the entire canvas.

    Input structure:
    - The most common color is the background on a 30x30 board.
    - One large monochrome object is the frame; its true holes remain background-colored cells inside the frame bbox.
    - Each hole is attached to a thin colored connector tail that leaves the frame and reaches exactly one outside sprite.

    Rule:
    - Treat every non-frame component inside the frame bbox as a hole plus its connector tail.
    - Use only the background-colored subset of that component as the fill mask, then follow the tail's color path across the whole grid to find the linked sprite color.
    - Return a full-size grid: restore the frame in place, fill each hole with its linked sprite color, and erase all outside sprites and connector paths.

    Key insight:
    - Shape repetition is no longer enough: identical holes can receive different colors, because pairing happens instance-by-instance through explicit connector paths.
    """
    g = np.array(grid, dtype=int)
    return _solve_full_grid(g)


def _can_place_hole(occ: np.ndarray, shape: np.ndarray, r: int, c: int) -> bool:
    th, tw = occ.shape
    sh, sw = shape.shape
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return False

    new_cells = set()
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                new_cells.add((int(r + dr), int(c + dc)))

    for rr, cc in new_cells:
        if not bool(occ[rr, cc]):
            return False
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < th and 0 <= nc < tw and not bool(occ[nr, nc]) and (nr, nc) not in new_cells:
                return False
    return True


def _shape_cells(shape: np.ndarray, r: int, c: int) -> set[tuple[int, int]]:
    sh, sw = shape.shape
    out = set()
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                out.add((int(r + dr), int(c + dc)))
    return out


def _carve_hole(occ: np.ndarray, shape: np.ndarray, r: int, c: int) -> None:
    sh, sw = shape.shape
    for dr in range(sh):
        for dc in range(sw):
            if shape[dr, dc]:
                    occ[r + dr, c + dc] = False


def _edge_cells(shape: np.ndarray, side: str) -> list[tuple[int, int]]:
    sh, sw = shape.shape
    if side == "top":
        return [(0, c) for c in range(sw) if shape[0, c]]
    if side == "bottom":
        return [(sh - 1, c) for c in range(sw) if shape[sh - 1, c]]
    if side == "left":
        return [(r, 0) for r in range(sh) if shape[r, 0]]
    if side == "right":
        return [(r, sw - 1) for r in range(sh) if shape[r, sw - 1]]
    raise ValueError(side)


def _opposite_side(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side]


def _corridor_cells(side: str, r: int, c: int, shape: np.ndarray, anchor: tuple[int, int], th: int, tw: int):
    ar, ac = map(int, anchor)
    if side == "top":
        lane = c + ac
        cells = [(rr, lane) for rr in range(0, r)]
    elif side == "bottom":
        lane = c + ac
        start = r + ar + 1
        cells = [(rr, lane) for rr in range(start, th)]
    elif side == "left":
        lane = r + ar
        cells = [(lane, cc) for cc in range(0, c)]
    elif side == "right":
        lane = r + ar
        start = c + ac + 1
        cells = [(lane, cc) for cc in range(start, tw)]
    else:
        raise ValueError(side)
    return [(int(rr), int(cc)) for rr, cc in cells]


def _build_hole_layout(th: int, tw: int):
    candidates = [
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= th - 2 and s.shape[1] <= tw - 2 and int(np.sum(s)) <= SHAPE_MAX_AREA
    ]
    if len(candidates) < CANONICAL_SHAPE_RANGE[0]:
        return None

    n_holes = random.randint(*HOLE_COUNT_RANGE)
    max_canon = min(CANONICAL_SHAPE_RANGE[1], n_holes - 1, len(candidates))
    n_canon = random.randint(CANONICAL_SHAPE_RANGE[0], max_canon)
    chosen = random.sample(candidates, n_canon)

    counts = [1] * n_canon
    for _ in range(n_holes - n_canon):
        counts[random.randrange(n_canon)] += 1
    if max(counts) < 2:
        return None

    specs = []
    for base, count in zip(chosen, counts):
        for _ in range(count):
            specs.append(
                {
                    "canonical_key": _canonical_key(base),
                    "orientations": _all_orientations(base),
                    "area": int(np.sum(base)),
                }
            )
    random.shuffle(specs)
    specs.sort(key=lambda item: item["area"], reverse=True)

    occ = np.ones((th, tw), dtype=bool)
    tunnel_reserved = np.zeros((th, tw), dtype=bool)
    holes = []
    side_counts = {side: 0 for side in SIDES}
    for spec in specs:
        placed = False
        for _ in range(HOLE_PLACEMENT_RETRIES):
            side_choices = [side for side in SIDES if side_counts[side] < 2]
            if not side_choices:
                break
            side = random.choice(side_choices)
            shape = random.choice(spec["orientations"])
            sh, sw = shape.shape
            anchors = _edge_cells(shape, side)
            if not anchors:
                continue
            anchor = random.choice(anchors)

            if side in {"top", "bottom"}:
                lane = random.randint(1, tw - 2)
                c = int(lane - anchor[1])
                if c < 1 or c + sw > tw - 1:
                    continue
                depth = random.randint(1, MAX_HOLE_BORDER_DISTANCE)
                r = int(depth if side == "top" else th - sh - depth)
            else:
                lane = random.randint(1, th - 2)
                r = int(lane - anchor[0])
                if r < 1 or r + sh > th - 1:
                    continue
                depth = random.randint(1, MAX_HOLE_BORDER_DISTANCE)
                c = int(depth if side == "left" else tw - sw - depth)

            if not _can_place_hole(occ, shape, r, c):
                continue

            new_cells = _shape_cells(shape, r, c)
            tunnel_block = _dilate4(tunnel_reserved)
            if any(tunnel_block[rr, cc] for rr, cc in new_cells):
                continue

            corridor = _corridor_cells(side, r, c, shape, anchor, th, tw)
            if not corridor:
                continue
            ok_corridor = True
            for rr, cc in corridor:
                if not bool(occ[rr, cc]) or bool(tunnel_block[rr, cc]):
                    ok_corridor = False
                    break
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < th and 0 <= nc < tw and not bool(occ[nr, nc]) and (nr, nc) not in new_cells:
                        ok_corridor = False
                        break
                if not ok_corridor:
                    break
            if not ok_corridor:
                continue

            _carve_hole(occ, shape, r, c)
            for rr, cc in corridor:
                tunnel_reserved[rr, cc] = True
            holes.append(
                {
                    "canonical_key": spec["canonical_key"],
                    "shape": shape.copy(),
                    "r": int(r),
                    "c": int(c),
                    "side": side,
                    "anchor": (int(anchor[0]), int(anchor[1])),
                    "corridor": corridor,
                }
            )
            side_counts[side] += 1
            placed = True
            break
        if not placed:
            return None

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    groups: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, hole in enumerate(holes):
        groups.setdefault(hole["canonical_key"], []).append(idx)
    if max(len(ids) for ids in groups.values()) < 2:
        return None
    return occ, holes


def _build_solved_frame(occ: np.ndarray, holes: list[dict], fill_colors: list[int], frame_col: int) -> np.ndarray:
    solved = np.full(occ.shape, frame_col, dtype=int)
    for hole, fill in zip(holes, fill_colors):
        patch = np.full(hole["shape"].shape, int(fill), dtype=int)
        paste(solved, patch, int(hole["r"]), int(hole["c"]), mask=hole["shape"])
    return solved


def _choose_colors(holes: list[dict], bg: int, frame_col: int):
    palette = [c for c in range(NUM_COLORS) if c not in {bg, frame_col}]
    fill_colors = [random.choice(palette) for _ in holes]

    groups: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, hole in enumerate(holes):
        groups.setdefault(hole["canonical_key"], []).append(idx)

    hetero_groups = 0
    for ids in groups.values():
        if len(ids) < 2:
            continue
        if len({fill_colors[i] for i in ids}) == 1:
            pivot = random.choice(ids)
            choices = [c for c in palette if c != fill_colors[pivot]]
            fill_colors[pivot] = random.choice(choices)
        if len({fill_colors[i] for i in ids}) > 1:
            hetero_groups += 1
    if hetero_groups == 0:
        return None

    line_colors = []
    for fill in fill_colors:
        choices = [c for c in palette if c != fill]
        line_colors.append(random.choice(choices))
    return fill_colors, line_colors


def _place_straight_pair(
    inp: np.ndarray,
    hole: dict,
    fill_color: int,
    line_color: int,
    fr: int,
    fc: int,
    frame_bbox: tuple[int, int, int, int],
    component_reserved: np.ndarray,
    soft_block: np.ndarray,
) -> bool:
    bg = int(get_bg(inp))
    side = str(hole["side"])
    global_corridor = [(fr + rr, fc + cc) for rr, cc in hole["corridor"]]
    for rr, cc in global_corridor:
        if component_reserved[rr, cc] or soft_block[rr, cc]:
            return False

    lane = fc + hole["c"] + hole["anchor"][1] if side in {"top", "bottom"} else fr + hole["r"] + hole["anchor"][0]
    r0, c0, r1, c1 = map(int, frame_bbox)
    orientations = _all_orientations(hole["shape"])
    random.shuffle(orientations)

    for sprite_mask in orientations:
        patch = np.full(sprite_mask.shape, bg, dtype=int)
        patch[sprite_mask] = int(fill_color)
        attach_cells = _edge_cells(sprite_mask, _opposite_side(side))
        random.shuffle(attach_cells)
        sh, sw = sprite_mask.shape

        for ar, ac in attach_cells:
            if side == "top":
                sc = lane - ac
                if sc < 0 or sc + sw > GRID_SIZE:
                    continue
                sr_values = list(range(0, r0 - sh))
            elif side == "bottom":
                sc = lane - ac
                if sc < 0 or sc + sw > GRID_SIZE:
                    continue
                sr_values = list(range(r1 + 1, GRID_SIZE - sh + 1))
            elif side == "left":
                sr = lane - ar
                if sr < 0 or sr + sh > GRID_SIZE:
                    continue
                sc_values = list(range(0, c0 - sw))
            else:
                sr = lane - ar
                if sr < 0 or sr + sh > GRID_SIZE:
                    continue
                sc_values = list(range(c1 + 1, GRID_SIZE - sw + 1))

            if side in {"top", "bottom"}:
                random.shuffle(sr_values)
                for sr in sr_values:
                    sprite_cells = np.zeros(inp.shape, dtype=bool)
                    for dr in range(sh):
                        for dc in range(sw):
                            if sprite_mask[dr, dc]:
                                sprite_cells[sr + dr, sc + dc] = True
                    if np.any(sprite_cells & (component_reserved | soft_block)):
                        continue
                    if side == "top":
                        outside = [(rr, lane) for rr in range(sr + ar + 1, r0)]
                    else:
                        outside = [(rr, lane) for rr in range(r1 + 1, sr + ar)]
                    if any(component_reserved[rr, cc] or soft_block[rr, cc] for rr, cc in outside):
                        continue
                    for rr, cc in global_corridor + outside:
                        inp[rr, cc] = int(line_color)
                        component_reserved[rr, cc] = True
                    paste(inp, patch, sr, sc, transparent=bg)
                    component_reserved |= sprite_cells
                    soft_block[:] = _dilate4(component_reserved)
                    return True
            else:
                random.shuffle(sc_values)
                for sc in sc_values:
                    sprite_cells = np.zeros(inp.shape, dtype=bool)
                    for dr in range(sh):
                        for dc in range(sw):
                            if sprite_mask[dr, dc]:
                                sprite_cells[sr + dr, sc + dc] = True
                    if np.any(sprite_cells & (component_reserved | soft_block)):
                        continue
                    if side == "left":
                        outside = [(lane, cc) for cc in range(sc + ac + 1, c0)]
                    else:
                        outside = [(lane, cc) for cc in range(c1 + 1, sc + ac)]
                    if any(component_reserved[rr, cc] or soft_block[rr, cc] for rr, cc in outside):
                        continue
                    for rr, cc in global_corridor + outside:
                        inp[rr, cc] = int(line_color)
                        component_reserved[rr, cc] = True
                    paste(inp, patch, sr, sc, transparent=bg)
                    component_reserved |= sprite_cells
                    soft_block[:] = _dilate4(component_reserved)
                    return True
    return False


def _dilate4(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = True
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, holes = layout

    colors = _choose_colors(holes, bg, frame_col)
    if colors is None:
        return None
    fill_colors, line_colors = colors

    solved_frame = _build_solved_frame(occ, holes, fill_colors, frame_col)
    frame_patch = np.where(occ, frame_col, bg).astype(int)

    if GRID_SIZE - th - FRAME_MARGIN < FRAME_MARGIN or GRID_SIZE - tw - FRAME_MARGIN < FRAME_MARGIN:
        return None
    fr = random.randint(FRAME_MARGIN, GRID_SIZE - th - FRAME_MARGIN)
    fc = random.randint(FRAME_MARGIN, GRID_SIZE - tw - FRAME_MARGIN)
    frame_bbox = (fr, fc, fr + th - 1, fc + tw - 1)

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, frame_patch, fr, fc)

    expected = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(expected, solved_frame, fr, fc)

    component_reserved = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    soft_block = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    order = list(range(len(holes)))
    order.sort(
        key=lambda idx: min(
            holes[idx]["r"],
            holes[idx]["c"],
            th - 1 - (holes[idx]["r"] + holes[idx]["shape"].shape[0] - 1),
            tw - 1 - (holes[idx]["c"] + holes[idx]["shape"].shape[1] - 1),
        ),
        reverse=True,
    )
    for idx in order:
        if not _place_straight_pair(
            inp,
            holes[idx],
            int(fill_colors[idx]),
            int(line_colors[idx]),
            fr,
            fc,
            frame_bbox,
            component_reserved,
            soft_block,
        ):
            return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
