# Variant of c7f57c3e
# Dimension: D15 — 符号编码/图例 (symbol_instructions)
# Change: 无符号编码，颜色模式直接从精灵颜色推断分组... → 图例区域指示颜色映射

"""
Design analysis for the D15 variant.

a) Concrete behavior and trace
   - The main scene still contains 3-4 scaled multicolor sprites built from two related
     structures. Structure B is the vertical flip of structure A, and every sprite keeps
     using color 1 as its anchor region.
   - A new legend is pasted against the top edge. It is a compact black rectangle whose
     two non-empty rows each show four swatches:
         [source_color_1, source_color_2] -> [target_color_1, target_color_2]
   - Each source pair belongs to exactly one current sprite family. The solver must first
     read the two legend rows, then identify a sprite's current source pair, swap the
     sprite to the opposite structure, and recolor its two variable regions with the
     target pair from the matching legend row.
   - Small trace:
       original logic: a scale-1 sprite with structure A and colors (6,8) is replaced by
       the visible opposite prototype already present elsewhere in colors (3,7).
       D15 logic: the same sprite now consults legend row (6,8)->(9,6), so it becomes
       structure B colored (9,6). The target pair is no longer recoverable from another
       sprite alone.

b) Complexity calibration
   - The legend has exactly two rows, matching the two sprite families. This keeps the
     new decoding step visible but bounded.
   - Source pairs use four distinct colors, so every input sprite matches exactly one
     legend row without ambiguity.
   - Target pairs are chosen so neither row simply reproduces either visible source pair.
     This prevents degeneration back to the original "read the opposite sprite colors"
     behavior.

c) Implicit-assumption audit against the original solver
   - Original code assumed every non-background component was a sprite. That fails once a
     legend component exists, so the new solver must detect and exclude the legend before
     sprite extraction.
   - Original prototype identification used raw colored arrays from the smallest sprites.
     That also fails because source palettes and target palettes are now separated by the
     legend. The new solver normalizes sprites to canonical labels {1,2,3} using the
     decoded source pair before deduplicating structures.
   - Original replacement copied the other visible prototype colors. The new variant must
     colorize the swapped structure from the legend target pair instead.

d) Placement legality
   - The legend is pasted first, touching the top edge. Sprite placement uses the same
     non-overlap reserve canvas as the original generator, so sprites cannot collide with
     the legend or each other.
   - The generator still guarantees at least one scale-1 sprite of each structure and
     exactly one scale-2/3 sprite, allowing the solver to recover both base structures.

e) Visual observability
   - The legend is a single black block distinct from the scene background, so humans can
     immediately see it as an instruction area.
   - Output sprite colors always include at least one color that appears only in the
     legend on input, making the legend functionally necessary and visually testable.

f) Visual cleanliness
   - No markers are drawn on top of sprites. The legend is isolated at the edge, while
     sprites remain separated by padding, so the scene stays readable.

g) Color-change inferability
   - Color changes are limited to two explicit pair mappings. The structural swap remains
     the same across all examples, and the legend provides the full mapping table needed
     to infer the recoloring rule from few examples.

h) New solver assumptions and generator guarantees
   - The legend is the only component containing color 0; the generator excludes color 0
     from sprite palettes and excludes background 0.
   - Every sprite has exactly two non-background, non-anchor colors; the label generator
     guarantees both variable roles are present.
   - The two source pairs are disjoint as sets; the palette sampler enforces that.
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
    """Convert a canonical label grid {0,1,2,3} into a colored sprite."""
    out = np.full(labels.shape, int(bg), dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = int(pair[0])
    out[labels == 3] = int(pair[1])
    return out


def _extract_legend(
    g: np.ndarray, bg: int
) -> tuple[tuple[int, int, int, int], dict[frozenset[int], tuple[tuple[int, int], tuple[int, int]]]]:
    """Return the legend bbox and a map source-pair-set -> (ordered source pair, ordered target pair)."""
    comps = components_mask(g != bg, connectivity=8)
    legend_boxes = []
    for pts in comps:
        vals = g[pts[:, 0], pts[:, 1]]
        if np.any(vals == BLACK):
            legend_boxes.append(bbox_coords(pts))
    assert len(legend_boxes) == 1, "expected exactly one legend component"

    r0, c0, r1, c1 = legend_boxes[0]
    box = g[r0 : r1 + 1, c0 : c1 + 1]
    mapping_rows = []
    for rr in range(box.shape[0]):
        vals = [int(v) for v in box[rr].tolist() if int(v) != BLACK]
        if vals:
            assert len(vals) == 4, "each legend row must contain exactly four colored swatches"
            src = (vals[0], vals[1])
            dst = (vals[2], vals[3])
            mapping_rows.append((src, dst))
    assert len(mapping_rows) == 2, "legend must contain exactly two mapping rows"

    legend_map: dict[frozenset[int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for src, dst in mapping_rows:
        key = frozenset(src)
        assert len(key) == 2 and key not in legend_map, "source pairs must be unique two-color sets"
        legend_map[key] = (src, dst)
    return legend_boxes[0], legend_map


def _sprite_mapping(
    arr: np.ndarray,
    bg: int,
    legend_map: dict[frozenset[int], tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Match a sprite's current two variable colors to one legend row."""
    colors = [int(v) for v in np.unique(arr) if int(v) not in (bg, 1)]
    assert len(colors) == 2, "each sprite must use exactly two variable colors"
    key = frozenset(colors)
    assert key in legend_map, "sprite colors must match one legend source pair"
    return legend_map[key]


def _normalize_sprite(arr: np.ndarray, bg: int, source_pair: tuple[int, int]) -> np.ndarray:
    """Map sprite colors back to canonical labels {0,1,2,3} using the legend source pair order."""
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
    Decode a legend, then swap every sprite to the opposite structure with legend-specified colors.

    Input structure:
    - The background is the most common color. A small black legend box sits on the grid edge.
    - Outside the legend, 3-4 sprite instances appear at integer scales. Each sprite is built from
      one of two related structures and uses color 1 plus exactly two family-specific source colors.

    Rule:
    - Read the two legend rows as ordered source-pair -> target-pair mappings.
    - Ignore the legend while recovering the two base sprite structures from the smallest sprites.
    - For each sprite, identify its current source pair from the legend, normalize it to canonical
      labels, swap to the other structure at the same scale, and recolor its two variable regions
      with the target pair from the matching legend row.

    Key insight: The geometric part of the task is unchanged, but the replacement colors are no
    longer inferable from the visible sprites alone; the black edge legend is the instruction table
    that disambiguates how each sprite family should be recolored after the structure swap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    legend_bbox, legend_map = _extract_legend(g, bg)
    body = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    body[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

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
                "arr": arr,
                "norm": norm,
                "src_pair": src_pair,
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
        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, ar1, ac1 = anc
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
            m = sub_p != bg
            if np.any(m):
                score = int(np.sum(sub_g[m] == sub_p[m]))
                if score > best_score:
                    best_score = score
                    best_idx = idx

        repl_labels = protos[1 - best_idx]
        repl = _colorize_labels(repl_labels, bg, s["dst_pair"])

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
        m = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[m] = sub_p[m]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 7

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: legend ---
LEGEND_SHAPE = (5, 9)
LEGEND_SOURCE_COLS = (1, 2)
LEGEND_TARGET_COLS = (6, 7)
LEGEND_ROWS = (1, 3)

# --- CFG: generation retries ---
MAX_RETRIES = 400

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


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    return _scale_nn(arr, k)


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
        chosen_kind = random.choices(kinds, weights=wts)[0]
        masks.append(_row_mask(chosen_kind, w))

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
    """Choose two distinct source pairs and two legend-only target pairs."""
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
    rows = list(range(2))
    random.shuffle(rows)
    mapping_order = [0, 1]
    random.shuffle(mapping_order)
    for row_idx, mapping_idx in zip(rows, mapping_order, strict=False):
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


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    struct_a, struct_b = _make_structures()
    structs = [struct_a, struct_b]
    src_pairs, dst_pairs = _sample_pairs(bg)
    legend = _build_legend(src_pairs, dst_pairs)

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(max(structs[typ].shape) * scale for typ, scale in instances)
    min_n = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES, LEGEND_SHAPE[1] + 2)
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    legend_c = random.randint(0, n - LEGEND_SHAPE[1])
    paste(inp, legend, 0, legend_c)
    paste(out, legend, 0, legend_c)
    paste(reserve, legend, 0, legend_c)

    for curr_type, scale in instances:
        input_patch = _scale(_colorize_labels(structs[curr_type], bg, src_pairs[curr_type]), scale)
        output_patch = _scale(_colorize_labels(structs[1 - curr_type], bg, dst_pairs[curr_type]), scale)
        dr, dc, uh, uw = _union_bbox(output_patch, input_patch)
        marker = np.full((uh, uw), 1, dtype=int)
        try:
            ur, uc = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        out_r = ur + max(0, -dr)
        out_c = uc + max(0, -dc)
        in_r = out_r + dr
        in_c = out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D15 input")
