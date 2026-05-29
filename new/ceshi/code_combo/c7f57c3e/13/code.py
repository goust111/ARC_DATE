# Combo variant of c7f57c3e (index 13)
# combo_spec: D13/2+D15/0
# Name: 边框分隔结构 + 图例区域

"""
Design analysis for the D13/2 + D15/0 combo variant.

a) Concrete behavior and trace
   - The sprite rule still comes from c7f57c3e: every object is a scaled copy of one of
     two related multicolor structures, and the output swaps each sprite to the opposite
     structure while preserving its scale and anchor alignment.
   - D15 adds an edge legend. The legend is a compact black box on the top edge with two
     active rows, each encoding:
         [source_color_1, source_color_2] -> [target_color_1, target_color_2]
     A sprite family must read its target colors from the legend instead of copying the
     other visible family.
   - D13 adds a non-sprite separator structure. Here it is a monochrome interior frame
     plus a vertical divider, splitting the lower playfield into left/right rooms. The
     divider stays unchanged from input to output and only organizes the scene.
   - Small trace:
       original logic: detect the two smallest visible sprites, infer the two prototypes,
       and replace each sprite with the opposite visible prototype colors.
       combo logic: first parse the black top legend, then remove both the legend and the
       monochrome room divider from consideration. After that, recover the two normalized
       structures from the smallest sprites, swap a sprite to the opposite structure, and
       recolor its variable regions using the legend row matched by that sprite's source
       color pair.

b) Complexity calibration
   - There are exactly two legend rows because there are still exactly two sprite families.
     This keeps the symbol-decoding step bounded and learnable.
   - The separator is always one connected monochrome structure with a rectangular frame
     and a full-height interior divider. That makes the region split visually obvious
     without introducing many structural cases.
   - Source pairs remain disjoint as sets, and target pairs are forced to differ from both
     visible source pairs, so the legend remains necessary and unambiguous.

c) Implicit-assumption audit against the original solver
   - Original assumption: every non-background component is a sprite. Broken twice here:
     one component is the legend and another is the separator frame/divider.
   - Original assumption: smallest colored components can be compared directly in their
     visible palettes. Broken by the legend because output colors are no longer visible on
     the input sprites. The solver must normalize sprites back to labels {1,2,3}.
   - D15-only assumption: the only non-sprite component is the legend. Broken here by the
     monochrome separator, so the solver must explicitly classify and exclude both kinds of
     structures before sprite recovery.
   - Matching/scaling logic from the original task still works once non-sprite structures
     are removed, because the anchor color-1 sub-bbox is preserved.

d) Placement legality
   - The generator uses output-first placement exactly as in the original task, but reserves
     three occupied areas before placing sprites: the legend, the separator frame, and the
     separator divider.
   - Sprites are placed by room, inside the left or right chamber of the divider, so they
     cannot overlap the frame, the divider, the legend, or each other.
   - The room geometry is derived from the largest union bbox needed for input/output anchor
     alignment, so both the input sprite and its swapped output fit within the same room.

e) Visual observability
   - Humans can see a black edge box that looks instruction-like and a separate large
     monochrome architectural structure that looks like a frame/divider rather than a
     transformable sprite.
   - Because target colors are sampled to introduce at least one legend-only color, output
     recoloring cannot be inferred from visible sprite palettes alone.
   - Because sprites appear in both left/right rooms, the divider is not a per-room rewrite
     rule; it is visibly a structural partition that must be ignored as an object.

f) Visual cleanliness
   - No symbol is drawn on top of sprites. The legend sits on the outer edge; the separator
     lives in the lower scene; sprites remain inside room interiors with padding.
   - The separator is single-color and connected, so it reads as one piece of scaffolding
     instead of many distracting marks.

g) Color-change inferability
   - Only two family mappings exist, and each family is identified by exactly two source
     colors. The legend shows the full ordered mapping table directly.
   - The separator color is allowed to overlap with sprite/legend colors, so geometry rather
     than raw color identity distinguishes structure from sprite. This strengthens the fusion
     while keeping the rule human-visible.

h) New solver assumptions and generator guarantees
   - The legend is the unique edge-touching black-containing component whose non-empty rows
     each contain exactly four non-black swatches.
   - Separator structures are monochrome, do not contain color 1, and are isolated from
     sprites by at least the same placement padding used between sprites.
   - Every sprite still contains color 1 and exactly two variable colors, and the generator
     guarantees at least one scale-1 instance of each structure.

i) Fusion-degeneracy check target
   - The generated sample must contain both a valid legend and a valid separator-defined
     two-room layout. A D15-only solver that ignores only the legend should fail on these
     samples because it would still try to treat the separator as a sprite.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


BLACK = 0


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of color-1 pixels; every sprite prototype is guaranteed to have one."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    raise AssertionError("sprite array must contain color-1 anchor pixels")


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _colorize_labels(labels: np.ndarray, bg: int, pair: tuple[int, int]) -> np.ndarray:
    """Convert canonical labels {0,1,2,3} into a colored sprite."""
    out = np.full(labels.shape, int(bg), dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = int(pair[0])
    out[labels == 3] = int(pair[1])
    return out


def _bbox_touches_edge(bbox: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _parse_legend_map(box: np.ndarray) -> dict[frozenset[int], tuple[tuple[int, int], tuple[int, int]]] | None:
    """Decode the legend rows if this bbox is a valid legend box; otherwise return None."""
    mapping_rows: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for rr in range(box.shape[0]):
        vals = [int(v) for v in box[rr].tolist() if int(v) != BLACK]
        if not vals:
            continue
        if len(vals) != 4:
            return None
        mapping_rows.append(((vals[0], vals[1]), (vals[2], vals[3])))
    if len(mapping_rows) != 2:
        return None

    legend_map: dict[frozenset[int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for src, dst in mapping_rows:
        key = frozenset(src)
        if len(key) != 2 or key in legend_map:
            return None
        legend_map[key] = (src, dst)
    return legend_map


def _extract_legend(
    g: np.ndarray, bg: int
) -> tuple[tuple[int, int, int, int], dict[frozenset[int], tuple[tuple[int, int], tuple[int, int]]]]:
    """Return the legend bbox and decoded source-pair -> (source,target) map."""
    h, w = g.shape
    comps = components_mask(g != bg, connectivity=8)
    candidates = []
    for pts in comps:
        bbox = bbox_coords(pts)
        if not _bbox_touches_edge(bbox, h, w):
            continue
        r0, c0, r1, c1 = bbox
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(box == BLACK):
            continue
        legend_map = _parse_legend_map(box)
        if legend_map is not None:
            candidates.append((bbox, legend_map))
    assert len(candidates) == 1, "expected exactly one valid legend component"
    return candidates[0]


def _is_separator_component(vals: np.ndarray, bg: int) -> bool:
    """Separator/frame structures are monochrome non-anchor components."""
    colors = [int(v) for v in np.unique(vals) if int(v) != bg]
    return len(colors) == 1 and colors[0] != 1


def _sprite_mapping(
    arr: np.ndarray,
    bg: int,
    legend_map: dict[frozenset[int], tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Match a sprite's current variable colors to one legend row."""
    colors = [int(v) for v in np.unique(arr) if int(v) not in (bg, 1)]
    assert len(colors) == 2, "each sprite must use exactly two variable colors"
    key = frozenset(colors)
    assert key in legend_map, "sprite colors must match one legend source pair"
    return legend_map[key]


def _normalize_sprite(arr: np.ndarray, bg: int, source_pair: tuple[int, int]) -> np.ndarray:
    """Map a colored sprite back to canonical labels {0,1,2,3}."""
    allowed = {int(bg), 1, int(source_pair[0]), int(source_pair[1])}
    vals = set(map(int, np.unique(arr)))
    assert vals <= allowed, "sprite contains colors outside its decoded palette"

    out = np.full(arr.shape, int(bg), dtype=int)
    out[arr == 1] = 1
    out[arr == int(source_pair[0])] = 2
    out[arr == int(source_pair[1])] = 3
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the edge legend, ignore the room-divider structure, then swap each sprite to the opposite structure.

    Input structure:
    - The background is the most common color. A compact black legend box sits on the top edge.
    - A large monochrome frame-plus-divider structure partitions the lower scene into left/right rooms.
    - Inside those rooms are exactly 3 sprite instances at integer scales. Every sprite contains color 1 and
      exactly two family-specific source colors.

    Rule:
    - Read the two legend rows as ordered source-pair -> target-pair mappings.
    - Exclude both kinds of non-sprite structure: the legend box and the monochrome separator frame/divider.
    - Recover the two base sprite structures from the smallest normalized sprites. For each sprite, identify
      its source pair from the legend, normalize it to canonical labels, swap it to the opposite structure
      at the same scale, and recolor the two variable regions with the legend-specified target pair.

    Key insight: The input now contains two different fixed structures that are not sprites. Solving requires
    separating architectural scaffolding from symbolic instructions first, then applying the original anchor-
    aligned structure swap with colors taken from the decoded legend rather than from other visible sprites.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    legend_bbox, legend_map = _extract_legend(g, bg)
    body = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    body[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

    # Remove separator structures before extracting sprites.
    for pts in components_mask(body != bg, connectivity=8):
        vals = body[pts[:, 0], pts[:, 1]]
        if _is_separator_component(vals, bg):
            body[pts[:, 0], pts[:, 1]] = bg

    comps = components_mask(body != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least two sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = body[r0 : r1 + 1, c0 : c1 + 1].copy()
        src_pair, dst_pair = _sprite_mapping(arr, bg, legend_map)
        norm = _normalize_sprite(arr, bg, src_pair)
        sprites.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "norm": norm,
                "dst_pair": dst_pair,
            }
        )

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["norm"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["norm"]
    assert len(uniq) == 2, "smallest sprites must reveal exactly two normalized structures"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h = anc0[2] - anc0[0] + 1
    base_anc_w = anc0[3] - anc0[1] + 1
    assert base_anc_h == anc1[2] - anc1[0] + 1 and base_anc_w == anc1[3] - anc1[1] + 1

    out = g.copy()

    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["norm"]
        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))

        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(p, bg)
            pr0 *= scale
            pc0 *= scale
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

            rr0 = max(0, top)
            cc0 = max(0, left)
            rr1 = min(h, top + sp.shape[0])
            cc1 = min(w, left + sp.shape[1])
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            sub_g = body[rr0:rr1, cc0:cc1]
            sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
            mask = sub_p != bg
            if np.any(mask):
                score = int(np.sum(sub_g[mask] == sub_p[mask]))
                if score > best_score:
                    best_score = score
                    best_idx = idx

        repl = _colorize_labels(protos[1 - best_idx], bg, s["dst_pair"])

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        sp = _scale_nn(repl, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(repl, bg)
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + sp.shape[0])
        cc1 = min(w, left + sp.shape[1])
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[mask] = sub_p[mask]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 24
GRID_SIZE_MAX = 28

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3]
BIG_SCALES = [2]

# --- CFG: placement ---
PLACEMENT_PADDING = 1
PLACEMENT_CONNECTIVITY = 8

# --- CFG: legend ---
LEGEND_SHAPE = (5, 9)
LEGEND_SOURCE_COLS = (1, 2)
LEGEND_TARGET_COLS = (6, 7)
LEGEND_ROWS = (1, 3)

# --- CFG: retries ---
MAX_RETRIES = 500

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _make_labels() -> np.ndarray:
    """Build a canonical label grid with roles 1 (anchor), 2, 3 and background 0."""
    h, w = random.choice(PROTO_SHAPES)

    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3
    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    weights_by_role = {
        1: ANCHOR_ROW_WEIGHTS,
        2: VARIABLE_ROW_WEIGHTS,
        3: VARIABLE_ROW_WEIGHTS,
    }
    masks = []
    for role in roles:
        kinds, wts = zip(*weights_by_role[role])
        masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[i] = _row_mask("outer", w)
        else:
            masks[i] = _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_structures() -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    return labels, np.flipud(labels)


def _sample_pairs(bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Choose two disjoint source pairs and two distinct target pairs."""
    pool = [c for c in range(2, 10) if c != bg]
    vals = random.sample(pool, 5)
    src_pairs = [(vals[0], vals[1]), (vals[2], vals[3])]

    candidates = [
        (vals[4], vals[0]),
        (vals[0], vals[4]),
        (vals[4], vals[1]),
        (vals[1], vals[4]),
        (vals[4], vals[2]),
        (vals[2], vals[4]),
        (vals[4], vals[3]),
        (vals[3], vals[4]),
    ]
    random.shuffle(candidates)
    dst_pairs: list[tuple[int, int]] = []
    for cand in candidates:
        if cand in src_pairs or cand in dst_pairs:
            continue
        dst_pairs.append(cand)
        if len(dst_pairs) == 2:
            break
    assert len(dst_pairs) == 2
    return src_pairs, dst_pairs


def _build_legend(
    src_pairs: list[tuple[int, int]],
    dst_pairs: list[tuple[int, int]],
) -> np.ndarray:
    legend = np.full(LEGEND_SHAPE, BLACK, dtype=int)
    row_order = [0, 1]
    mapping_order = [0, 1]
    random.shuffle(row_order)
    random.shuffle(mapping_order)
    for row_idx, mapping_idx in zip(row_order, mapping_order, strict=False):
        rr = LEGEND_ROWS[row_idx]
        src = src_pairs[mapping_idx]
        dst = dst_pairs[mapping_idx]
        legend[rr, LEGEND_SOURCE_COLS[0]] = int(src[0])
        legend[rr, LEGEND_SOURCE_COLS[1]] = int(src[1])
        legend[rr, LEGEND_TARGET_COLS[0]] = int(dst[0])
        legend[rr, LEGEND_TARGET_COLS[1]] = int(dst[1])
    return legend


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    """Compute (dr, dc, union_h, union_w) for anchor-aligned placement of two patches."""
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h = max(patch_a.shape[0], patch_b.shape[0])
    w = max(patch_a.shape[1], patch_b.shape[1])
    return dr, dc, h + abs(dr), w + abs(dc)


def _room_free_location(
    reserve: np.ndarray,
    marker: np.ndarray,
    room: tuple[int, int, int, int],
    bg: int,
) -> tuple[int, int]:
    """Place a marker inside one room interior using the shared reserve grid."""
    r0, c0, r1, c1 = room
    sub = reserve[r0 : r1 + 1, c0 : c1 + 1]
    sr, sc = random_free_location(
        sub,
        marker,
        transparent=bg,
        border=0,
        padding=PLACEMENT_PADDING,
        padding_connectivity=PLACEMENT_CONNECTIVITY,
    )
    return r0 + sr, c0 + sc


def _draw_separator(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    divider_c: int,
    color: int,
) -> None:
    """Draw a rectangular frame plus a vertical divider."""
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)
    grid[top : bottom + 1, divider_c] = int(color)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    structures = list(_make_structures())
    src_pairs, dst_pairs = _sample_pairs(bg)
    legend = _build_legend(src_pairs, dst_pairs)

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances: list[tuple[int, int, str]] = [
        (0, 1, "left"),
        (1, 1, "right"),
        (random.randrange(2), big_scale, random.choice(["left", "right"])),
    ]
    if count == 4:
        instances.append((random.randrange(2), 1, random.choice(["left", "right"])))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    union_widths = []
    union_heights = []
    for curr_type, scale, _room_name in instances:
        input_patch = _scale_nn(_colorize_labels(structures[curr_type], bg, src_pairs[curr_type]), scale)
        output_patch = _scale_nn(_colorize_labels(structures[1 - curr_type], bg, dst_pairs[curr_type]), scale)
        _dr, _dc, uh, uw = _union_bbox(output_patch, input_patch)
        union_heights.append(uh)
        union_widths.append(uw)
    need_room_w = max(union_widths)
    need_room_h = max(union_heights)

    top = LEGEND_SHAPE[0] + 1
    min_n = max(GRID_SIZE_MIN, LEGEND_SHAPE[1] + 2, need_room_h + top + 5, 2 * need_room_w + 9)
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    left = 1
    right = n - 2
    bottom = n - 2
    divider_min = need_room_w + 4
    divider_max = n - 5 - need_room_w
    if divider_min > divider_max:
        return None
    divider_c = random.randint(divider_min, divider_max)

    left_room = (top + 2, left + 2, bottom - 2, divider_c - 2)
    right_room = (top + 2, divider_c + 2, bottom - 2, right - 2)
    if left_room[0] > left_room[2] or left_room[1] > left_room[3]:
        return None
    if right_room[0] > right_room[2] or right_room[1] > right_room[3]:
        return None

    sep_color = random.choice([c for c in range(2, 10) if c != bg])

    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    legend_c = random.randint(0, n - LEGEND_SHAPE[1])
    paste(inp, legend, 0, legend_c)
    paste(out, legend, 0, legend_c)
    paste(reserve, legend, 0, legend_c)

    _draw_separator(inp, top, bottom, left, right, divider_c, sep_color)
    _draw_separator(out, top, bottom, left, right, divider_c, sep_color)
    _draw_separator(reserve, top, bottom, left, right, divider_c, sep_color)

    rooms = {"left": left_room, "right": right_room}
    room_usage = {"left": 0, "right": 0}
    for curr_type, scale, room_name in instances:
        input_patch = _scale_nn(_colorize_labels(structures[curr_type], bg, src_pairs[curr_type]), scale)
        output_patch = _scale_nn(_colorize_labels(structures[1 - curr_type], bg, dst_pairs[curr_type]), scale)
        dr, dc, uh, uw = _union_bbox(output_patch, input_patch)
        marker = np.full((uh, uw), 1, dtype=int)
        try:
            ur, uc = _room_free_location(reserve, marker, rooms[room_name], bg)
        except ValueError:
            return None

        out_r = ur + max(0, -dr)
        out_c = uc + max(0, -dc)
        in_r = out_r + dr
        in_c = out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)
        room_usage[room_name] += 1

    if min(room_usage.values()) < 1:
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
