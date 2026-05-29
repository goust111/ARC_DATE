# Variant of 8b9c3697
# Dimension: D1 — 匹配条件：行列对齐射线投射 (match_cond)
# Change: 从结构内部插槽（socket）向四个基本方向发射射线，命中红色色块即确定匹配关系和滑入方向（s.col/row == d... → 连接器匹配

from __future__ import annotations

from collections import Counter
import random
import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D1 connector-matching variant
# ---------------------------------------------------------------------------
# variant_design:
#   concretization:
#     - Keep the original large comb/bracket structure and the original motion:
#       a rectangular red block still slides straight into the interior socket and
#       paints a black trail.
#     - Replace "which red does this structure dock with?" from ray-hit matching
#       to connector matching. Each structure has one tiny attached connector at
#       the opening lip. Each red candidate may have one tiny detached connector
#       nearby. A match requires exact equality of connector color and normalized
#       2-3-cell shape.
#     - Structure-side connectors are attached to the big structure, so they are
#       part of the large non-red component. Red-side connectors are detached tiny
#       components close to their red rectangle; they disappear together with the
#       candidate reds in the output.
#   behavior_trace:
#     - Old logic: a structure scans outward from interior sockets and assigns the
#       red component that is most visible / closest on those rays.
#     - New logic: the structure first reads its attached connector signature,
#       finds the red whose nearby connector has the same signature, and only then
#       performs the same straight slide into the socket.
#     - To make this visible, the generator deliberately inserts at least one
#       misleading red on the same opening axis but with a wrong connector. The
#       old solver would prefer that closer red; the new solver must ignore it.
#   complexity_calibration:
#     - Connector vocabulary is small and repeated: 2 colors x 2 shapes = 4
#       possible signatures. Each grid uses 2-3 distinct structure signatures,
#       while wrong reds use only signatures absent from structures.
#     - This is enough to force local pattern recognition without introducing an
#       unlearnable many-to-many color mapping.
#   implicit_assumptions_audit:
#     - Original `main()` assumed every non-bg/non-red component is a structure.
#       That no longer holds because detached connectors are also non-bg/non-red,
#       so the new solver splits large components (structures) from tiny ones
#       (free connectors) by area.
#     - Original docking choice depended on red visibility from socket rays. The
#       new solver cannot use those rays for identity, so it must:
#         1. read structure connector signature from the large component,
#         2. associate each tiny connector with a nearby red,
#         3. use connector equality for assignment,
#         4. recover the slide direction from the attached connector side.
#     - Original output clearing removed only red. In the variant, detached red
#       connectors must also be cleared, or unmatched markers would remain.
#   placement_legality:
#     - Structure connectors are attached outside the body bbox on the opening
#       side and never occupy the sliding corridor.
#     - Red connectors are one background cell away from their red, so the solver
#       can still extract pure red connected components.
#     - Structure placement uses a union footprint of input/output content, so
#       hidden input-only connectors or decoys cannot overlap other patches.
#   visual_observability:
#     - The cue is visible in 2-4 examples because the same tiny colored pattern
#       appears once on a structure and once near the correct red, while closer
#       wrong reds display a different tiny pattern.
#     - No hidden ordering by color id is used; color only participates as a
#       visible connector attribute.
#   cleanliness:
#     - Only one attached connector is placed per structure, always near the lip.
#       Red connectors are offset to a side of the red rectangle, so the black
#       trail and the connector clue do not visually overwrite each other.


RED = int(Color.RED)
BLACK = int(Color.BLACK)

GRID_H_RANGE = (19, 23)
GRID_W_RANGE = (22, 27)
BG_COLORS = (3, 4)
STRUCT_COLORS = (1, 4, 8)

STRUCT_WIDTH_TEETH = ((7, 1), (8, 2), (9, 3))
STRUCT_COUNT_RANGE = (2, 3)
UNMATCHED_PROB = 0.35
SLIDE_RANGE = (5, 8)
STRUCT_PADDING = 3

TOTAL_RED_MIN = 5
TOTAL_RED_MAX = 11
MAX_RETRIES = 400

STRUCT_MIN_AREA = 8
FREE_CONNECTOR_MAX_AREA = 3

CONNECTOR_SHAPES = {
    "domino": ((0, 0), (0, 1)),
    "ell": ((0, 0), (1, 0), (1, 1)),
}
CONNECTOR_TYPES = (
    (6, "domino"),
    (6, "ell"),
    (7, "domino"),
    (7, "ell"),
)

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _normalize_shape(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0, c0 = min(rows), min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _rot_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = [(-int(c), int(r)) for r, c in shape]
    return _normalize_shape(pts)


ALLOWED_CONNECTOR_SHAPES: set[tuple[tuple[int, int], ...]] = set()
for _shape in CONNECTOR_SHAPES.values():
    cur = _normalize_shape(list(_shape))
    for _ in range(4):
        ALLOWED_CONNECTOR_SHAPES.add(cur)
        cur = _rot_shape(cur)


def _component_signature(grid: np.ndarray, coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    pts = [(int(r), int(c)) for r, c in coords]
    colors = {int(grid[r, c]) for r, c in pts}
    if len(colors) != 1:
        return None
    return int(next(iter(colors))), _normalize_shape(pts)


def _min_manhattan(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = 10**9
    for ar, ac in a:
        for br, bc in b:
            best = min(best, abs(int(ar) - int(br)) + abs(int(ac) - int(bc)))
    return int(best)


def _structure_direction(connector_pts: list[tuple[int, int]], body_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, body_bbox)
    rows = [int(r) for r, _ in connector_pts]
    cols = [int(c) for _, c in connector_pts]
    if max(rows) > r1:
        return (1, 0)
    if min(rows) < r0:
        return (-1, 0)
    if max(cols) > c1:
        return (0, 1)
    if min(cols) < c0:
        return (0, -1)
    raise AssertionError("connector must sit outside the structure body bbox")


def _split_points4(points: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = {tuple(map(int, p)) for p in points}
    groups: list[list[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        group = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    group.append(nxt)
        groups.append(group)
    return groups


def _socket_cells(body_pts: list[tuple[int, int]], body_bbox: tuple[int, int, int, int],
                  direction: tuple[int, int], grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, body_bbox)
    dr, dc = map(int, direction)
    out: set[tuple[int, int]] = set()
    for r, c in body_pts:
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            continue
        if int(grid[nr, nc]) != int(bg):
            continue
        if not (r0 < nr < r1 and c0 < nc < c1):
            continue
        out.add((int(nr), int(nc)))
    return sorted(out)


def _red_on_open_side(red_bbox: tuple[int, int, int, int], body_bbox: tuple[int, int, int, int],
                      direction: tuple[int, int]) -> bool:
    rr0, rc0, rr1, rc1 = map(int, red_bbox)
    sr0, sc0, sr1, sc1 = map(int, body_bbox)
    dr, dc = map(int, direction)
    if (dr, dc) == (1, 0):
        return rr0 > sr1
    if (dr, dc) == (-1, 0):
        return rr1 < sr0
    if (dr, dc) == (0, 1):
        return rc0 > sc1
    return rc1 < sc0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert red components into interior sockets of large structures by connector match, leaving a black trail.

    Input structure:
    - Background = most common color. Large comb-like structures are non-bg and non-red.
    - Each structure carries a tiny attached connector at its opening lip.
    - Red rectangles outside the structures may carry their own tiny nearby connector.

    Rule:
    - Read each structure's connector signature (connector color + normalized 2-3-cell shape).
    - Read each red candidate's nearby connector signature; a red matches only if the signatures are equal.
    - Move the matched red straight into the interior socket on that structure's opening side and paint the traversed cells black.
    - Unmatched reds and all detached red-side connectors disappear.

    Key insight: The docking target is no longer chosen by line-of-sight to the nearest red; it is selected by matching the small connector pattern attached to the structure with the small connector pattern placed near the red.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    assert int(RED) in set(map(int, np.unique(g)))

    red_comps = components_mask(g == int(RED), connectivity=4)
    assert red_comps

    aux_mask = (g != int(bg)) & (g != int(RED)) & (g != int(BLACK))
    aux_comps = components_mask(aux_mask, connectivity=4)
    assert aux_comps

    struct_comps = [comp for comp in aux_comps if int(comp.shape[0]) >= int(STRUCT_MIN_AREA)]
    free_conn_comps = [comp for comp in aux_comps if int(comp.shape[0]) <= int(FREE_CONNECTOR_MAX_AREA)]
    assert struct_comps, "expected at least one structure component"

    free_connectors = []
    for comp in free_conn_comps:
        pts = [(int(r), int(c)) for r, c in comp]
        sig = _component_signature(g, pts)
        if sig is None:
            continue
        free_connectors.append({"coords": pts, "sig": sig})

    red_infos = []
    for rid, comp in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in comp]
        bbox = tuple(map(int, bbox_coords(comp)))
        nearby = []
        for cid, info in enumerate(free_connectors):
            dist = _min_manhattan(pts, info["coords"])
            if int(dist) <= 3:
                nearby.append((int(dist), int(cid)))
        nearby.sort()
        sig = None
        if nearby and (len(nearby) == 1 or int(nearby[0][0]) < int(nearby[1][0])):
            sig = free_connectors[int(nearby[0][1])]["sig"]
        red_infos.append({
            "rid": int(rid),
            "coords": pts,
            "bbox": bbox,
            "sig": sig,
        })

    sig_to_reds: dict[tuple[int, tuple[tuple[int, int], ...]], list[int]] = {}
    for info in red_infos:
        if info["sig"] is not None:
            sig_to_reds.setdefault(info["sig"], []).append(int(info["rid"]))

    struct_infos = []
    for sid, comp in enumerate(struct_comps):
        pts = [(int(r), int(c)) for r, c in comp]
        color_counts = Counter(int(g[r, c]) for r, c in pts)
        body_color = int(color_counts.most_common(1)[0][0])
        body_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == int(body_color)]
        raw_conn_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) != int(body_color)]
        assert raw_conn_pts, "structure must carry an attached connector"
        body_bbox = tuple(map(int, bbox_coords(np.array(body_pts, dtype=int))))
        body_set = set(body_pts)
        conn_groups = []
        for color in sorted({int(g[r, c]) for r, c in raw_conn_pts}):
            color_pts = [(int(r), int(c)) for r, c in raw_conn_pts if int(g[r, c]) == int(color)]
            conn_groups.extend(_split_points4(color_pts))
        touching = [
            grp for grp in conn_groups
            if any((int(r + dr), int(c + dc)) in body_set for r, c in grp for dr, dc in DIRS)
        ]
        touching = [
            grp for grp in touching
            if len(grp) in (2, 3) and _normalize_shape(grp) in ALLOWED_CONNECTOR_SHAPES
        ]
        assert touching, "expected a legal attached connector group"
        outer = [
            grp for grp in touching
            if max(r for r, _ in grp) > body_bbox[2]
            or min(r for r, _ in grp) < body_bbox[0]
            or max(c for _, c in grp) > body_bbox[3]
            or min(c for _, c in grp) < body_bbox[1]
        ]
        conn_pts = max(outer or touching, key=len)
        conn_sig = _component_signature(g, conn_pts)
        assert conn_sig is not None
        direction = _structure_direction(conn_pts, body_bbox)
        sockets = _socket_cells(body_pts, body_bbox, direction, g, bg)
        assert sockets, "expected at least one interior socket cell"
        struct_infos.append({
            "sid": int(sid),
            "bbox": body_bbox,
            "sig": conn_sig,
            "dir": direction,
            "anchor": sockets[0],
        })

    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    used_red: set[int] = set()
    for sinfo in struct_infos:
        candidate_rids = []
        for rid in sig_to_reds.get(sinfo["sig"], []):
            rinfo = red_infos[int(rid)]
            if _red_on_open_side(rinfo["bbox"], sinfo["bbox"], sinfo["dir"]):
                candidate_rids.append(int(rid))
        if len(candidate_rids) != 1:
            continue
        rid = int(candidate_rids[0])
        if rid in used_red:
            continue
        used_red.add(rid)
        assignments.append((rid, tuple(map(int, sinfo["dir"])), tuple(map(int, sinfo["anchor"]))))

    out = g.copy()
    out[out == int(RED)] = int(bg)
    for info in free_connectors:
        for r, c in info["coords"]:
            out[int(r), int(c)] = int(bg)

    for rid, (dr, dc), (sr, sc) in assignments:
        coords = red_infos[int(rid)]["coords"]
        r0, c0, r1, c1 = map(int, red_infos[int(rid)]["bbox"])
        tr, tc = int(sr), int(sc)

        if (dr, dc) == (0, 1):
            shift_r, shift_c = 0, int(tc - c0)
        elif (dr, dc) == (0, -1):
            shift_r, shift_c = 0, int(tc - c1)
        elif (dr, dc) == (1, 0):
            shift_r, shift_c = int(tr - r0), 0
        else:
            shift_r, shift_c = int(tr - r1), 0

        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = 0 if int(shift_r) == 0 else (1 if int(shift_r) > 0 else -1)
        step_dc = 0 if int(shift_c) == 0 else (1 if int(shift_c) > 0 else -1)

        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(BLACK)

        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(RED)

    return out


def _structure_array(width: int, teeth: int, bg: int, sc: int) -> np.ndarray:
    g = np.full((4, width), bg, dtype=int)
    start = (width - teeth) // 2
    g[0, :] = int(sc)
    g[1, 1] = int(sc)
    g[1, width - 2] = int(sc)
    g[2:4, 1] = int(sc)
    g[2:4, width - 2] = int(sc)
    g[1, start:start + teeth] = int(sc)
    return g


def _stamp_connector(patch: np.ndarray, anchor_r: int, anchor_c: int,
                     conn_type: tuple[int, str], bg: int) -> bool:
    color, shape_name = conn_type
    for dr, dc in CONNECTOR_SHAPES[str(shape_name)]:
        rr = int(anchor_r + dr)
        cc = int(anchor_c + dc)
        if not (0 <= rr < patch.shape[0] and 0 <= cc < patch.shape[1]):
            return False
        if int(patch[rr, cc]) != int(bg):
            return False
    for dr, dc in CONNECTOR_SHAPES[str(shape_name)]:
        rr = int(anchor_r + dr)
        cc = int(anchor_c + dc)
        patch[rr, cc] = int(color)
    return True


def _connector_dims(conn_type: tuple[int, str]) -> tuple[int, int]:
    offs = CONNECTOR_SHAPES[str(conn_type[1])]
    return max(dr for dr, _ in offs) + 1, max(dc for _, dc in offs) + 1


def _matched_patches(width: int, teeth: int, slide: int, rot_k: int, bg: int, sc: int,
                     conn_type: tuple[int, str], wrong_pool: list[tuple[int, str]],
                     force_decoy: bool) -> dict | None:
    red_h = min(int(teeth), 2)
    base = _structure_array(width, teeth, bg, sc)
    patch_h = int(2 + slide + red_h)
    structure = np.full((patch_h, width), bg, dtype=int)
    paste(structure, base, 0, 0, transparent=bg)

    if not _stamp_connector(structure, 4, width - 2, conn_type, bg):
        return None

    input_patch = structure.copy()
    output_patch = structure.copy()

    start = (width - teeth) // 2
    for off in range(1, slide + 1):
        output_patch[2 + off:2 + off + red_h, start:start + teeth] = int(BLACK)
    output_patch[2:2 + red_h, start:start + teeth] = int(RED)

    true_top = int(2 + slide)
    input_patch[true_top:true_top + red_h, start:start + teeth] = int(RED)
    if not _stamp_connector(input_patch, true_top, 0, conn_type, bg):
        return None

    has_decoy = bool(force_decoy or random.random() < 0.55)
    if has_decoy:
        decoy_slide = random.randint(2, slide - 3)
        decoy_top = int(2 + decoy_slide)
        input_patch[decoy_top:decoy_top + 1, start:start + teeth] = int(RED)
        wrong_type = random.choice(wrong_pool)
        if not _stamp_connector(input_patch, decoy_top, 0, wrong_type, bg):
            return None

    footprint = np.full_like(input_patch, bg)
    footprint[(input_patch != int(bg)) | (output_patch != int(bg))] = int(sc)

    return {
        "structure": np.rot90(structure, k=rot_k),
        "input": np.rot90(input_patch, k=rot_k),
        "output": np.rot90(output_patch, k=rot_k),
        "footprint": np.rot90(footprint, k=rot_k),
        "has_decoy": has_decoy,
    }


def _unmatched_patches(width: int, teeth: int, rot_k: int, bg: int, sc: int,
                       conn_type: tuple[int, str]) -> dict | None:
    structure = np.full((6, width), bg, dtype=int)
    paste(structure, _structure_array(width, teeth, bg, sc), 0, 0, transparent=bg)
    if not _stamp_connector(structure, 4, width - 2, conn_type, bg):
        return None
    rotated = np.rot90(structure, k=rot_k)
    return {
        "structure": rotated,
        "input": rotated.copy(),
        "output": rotated.copy(),
        "footprint": rotated.copy(),
        "has_decoy": False,
    }


def _place_patch(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int) -> tuple[int, int] | None:
    try:
        return random_free_location(canvas, patch, transparent=bg, padding=padding)
    except ValueError:
        return None


def _place_structures(specs: list[dict], h: int, w: int, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    footprint_canvas = np.full((h, w), bg, dtype=int)

    for spec in specs:
        loc = _place_patch(footprint_canvas, spec["footprint"], bg, padding=STRUCT_PADDING)
        if loc is None:
            return None
        r0, c0 = loc
        paste(footprint_canvas, spec["footprint"], r0, c0, transparent=bg)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(out, spec["output"], r0, c0, transparent=bg)

    return inp, out


def _red_pair_patch(rect_h: int, rect_w: int, bg: int, conn_type: tuple[int, str], side: str) -> np.ndarray | None:
    conn_h, conn_w = _connector_dims(conn_type)
    gap = 1
    h = max(int(rect_h), int(conn_h))
    w = int(rect_w + conn_w + gap)
    patch = np.full((h, w), bg, dtype=int)

    if side == "right":
        patch[0:rect_h, 0:rect_w] = int(RED)
        if not _stamp_connector(patch, 0, rect_w + gap, conn_type, bg):
            return None
    else:
        patch[0:rect_h, conn_w + gap:conn_w + gap + rect_w] = int(RED)
        if not _stamp_connector(patch, 0, 0, conn_type, bg):
            return None
    return patch


def _place_extra_distractors(inp: np.ndarray, bg: int, wrong_pool: list[tuple[int, str]], count: int) -> bool:
    shapes = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    for _ in range(count):
        random.shuffle(shapes)
        random.shuffle(wrong_pool)
        placed = False
        for rh, rw in shapes:
            for conn_type in wrong_pool:
                side = random.choice(["left", "right"])
                patch = _red_pair_patch(rh, rw, bg, conn_type, side)
                if patch is None:
                    continue
                loc = _place_patch(inp, patch, bg, padding=1)
                if loc is None:
                    continue
                paste(inp, patch, loc[0], loc[1], transparent=bg)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_COLORS)
    sc = random.choice([c for c in STRUCT_COLORS if c not in (bg, int(RED), int(BLACK))])
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    n_structs = random.randint(*STRUCT_COUNT_RANGE)
    unmatched = 1 if random.random() < UNMATCHED_PROB else 0
    matched = n_structs - unmatched
    if matched <= 0:
        return None

    struct_conn_types = random.sample(list(CONNECTOR_TYPES), k=n_structs)
    wrong_pool = [t for t in CONNECTOR_TYPES if t not in struct_conn_types]
    if not wrong_pool:
        return None

    specs = []
    for idx in range(n_structs):
        width, teeth = random.choice(STRUCT_WIDTH_TEETH)
        rot_k = random.randrange(4)
        conn_type = struct_conn_types[idx]
        if idx < matched:
            slide = random.randint(*SLIDE_RANGE)
            spec = _matched_patches(width, teeth, slide, rot_k, bg, sc, conn_type, wrong_pool, force_decoy=(idx == 0))
        else:
            spec = _unmatched_patches(width, teeth, rot_k, bg, sc, conn_type)
        if spec is None:
            return None
        specs.append(spec)

    random.shuffle(specs)
    specs.sort(key=lambda s: np.count_nonzero(s["footprint"] != bg), reverse=True)

    placed = _place_structures(specs, h, w, bg)
    if placed is None:
        return None
    inp, out = placed

    current_reds = len(components_mask(inp == int(RED), connectivity=4))
    target_total = random.randint(max(TOTAL_RED_MIN, current_reds), TOTAL_RED_MAX)
    extra = target_total - current_reds
    if extra < 0:
        return None
    if not _place_extra_distractors(inp, bg, wrong_pool, extra):
        return None

    if not any(spec["has_decoy"] for spec in specs):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 D1 connector variant")
