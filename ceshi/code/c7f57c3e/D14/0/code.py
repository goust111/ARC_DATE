# Variant of c7f57c3e
# Dimension: D14 — 交换后的轨迹/残留 (trajectory)
# Change: 无轨迹，颜色模式直接替换，原始颜色完全消失... → 保留原始颜色的轮廓痕迹

"""
Design analysis for the D14 variant.

- Concrete behavior:
  Input stays the same as the base task: 3-4 disconnected multicolor sprites, each a
  scaled copy of one of two prototypes. Output still swaps every sprite to the other
  prototype at the same scale and anchor alignment, but the source sprite does not
  vanish completely. A 1-pixel-wide contour of the source instance is painted back on
  top using the source colors. The contour is computed on the scaled instance pixels,
  so a scale-3 sprite gets a true 1-pixel rim instead of a 3-pixel-thick rim.
- Behavior trace on one sprite:
  Base task: clear the source sprite -> place the opposite prototype.
  Variant: clear the source sprite -> place the opposite prototype -> repaint only the
  source boundary pixels at their original coordinates and colors.
  The difference is visible exactly on the old sprite edge cells.
- Complexity calibration:
  The rule keeps the original two-prototype matching problem intact and adds one extra
  layer with a single binary choice per pixel: boundary or non-boundary. This avoids a
  proliferation of cases while producing a clear visual difference from direct replace.
- Hidden-assumption audit for main():
  Prototype detection from the smallest sprites still works because the input format is
  unchanged. Scale inference still depends on color-1 anchor bboxes being present and
  equally sized across the two prototypes; the generator keeps that invariant. The new
  edge repaint must not use prototype-space edges, or large sprites would get thick
  bands, so the solver computes edges from the observed scaled instance array itself.
  Repainting also assumes different sprites do not share the same swap footprint; the
  generator already reserves the aligned input/output union for every placed instance.
- Placement legality:
  The generator still works output-first. For each intended output sprite, it builds the
  full swapped patch (target sprite plus source contour), then reserves that aligned
  union footprint before placing both the output patch and the source input patch.
  This prevents overlap, clipping, and cross-sprite interference.
- Visual observability:
  The contour is visible in the output only, which matches the "trajectory/residue"
  interpretation. To make the new layer easy to notice, the generator biases the single
  large sprite toward scale 3 and lowers cross-prototype color sharing, so the old rim
  usually contrasts with the new interior.
- Visual cleanliness:
  Only one extra layer is added, and it lives on the boundary of the source sprite.
  There is no extra marker object and no global bookkeeping, so the output remains a
  set of separated sprites rather than a cluttered scene.
- Color inference:
  The recolored contour always copies literal source colors from the input sprite at the
  same cells. No hidden ordering by color id is used.
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of the color-1 anchor block; it is guaranteed to exist."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _edge_mask(arr: np.ndarray, bg: int) -> np.ndarray:
    """1-pixel contour of the observed sprite instance, using 4-neighborhood erosion."""
    mask = arr != bg
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    up = np.zeros_like(mask, dtype=bool)
    down = np.zeros_like(mask, dtype=bool)
    left = np.zeros_like(mask, dtype=bool)
    right = np.zeros_like(mask, dtype=bool)
    up[1:, :] = mask[:-1, :]
    down[:-1, :] = mask[1:, :]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]

    interior = mask & up & down & left & right
    return mask & ~interior


def _aligned_union_layout(
    source_patch: np.ndarray, target_patch: np.ndarray, bg: int
) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    """Return union size and top-left offsets when anchor-aligning target to source."""
    sa0, sb0, _sa1, _sb1 = _pick_anchor_bbox(source_patch, bg)
    ta0, tb0, _ta1, _tb1 = _pick_anchor_bbox(target_patch, bg)
    dr = int(sa0 - ta0)
    dc = int(sb0 - tb0)

    sh, sw = map(int, source_patch.shape)
    th, tw = map(int, target_patch.shape)
    top = min(0, dr)
    left = min(0, dc)
    bottom = max(sh, dr + th)
    right = max(sw, dc + tw)

    union_h = int(bottom - top)
    union_w = int(right - left)
    source_pos = (int(-top), int(-left))
    target_pos = (int(dr - top), int(dc - left))
    return union_h, union_w, source_pos, target_pos


def _compose_swapped_patch(source_patch: np.ndarray, target_patch: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build the aligned output patch: swapped target plus source-colored contour."""
    union_h, union_w, source_pos, target_pos = _aligned_union_layout(source_patch, target_patch, bg)
    out = np.full((union_h, union_w), bg, dtype=int)

    paste(out, target_patch, target_pos[0], target_pos[1], transparent=bg)

    edge = _edge_mask(source_patch, bg)
    sr, sc = source_pos
    out_view = out[sr : sr + source_patch.shape[0], sc : sc + source_patch.shape[1]]
    out_view[edge] = source_patch[edge]
    out[sr : sr + source_patch.shape[0], sc : sc + source_patch.shape[1]] = out_view
    return out, source_pos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each sprite to the opposite prototype while keeping a source-colored rim.

    Input structure:
    - The background is the most frequent color. Foreground pixels form 3-4 isolated sprites.
    - Every sprite is an integer-scaled copy of one of two small multicolor prototypes, and
      both prototypes contain a color-1 anchor region that survives all scaling.

    Rule:
    - Recover the two prototypes from the smallest sprites, infer each sprite's scale, and
      decide which prototype it currently matches.
    - Replace that sprite with the other prototype at the same scale and anchor alignment.
    - After placing the swapped sprite, repaint the 1-pixel boundary of the original instance
      at its original cells using its original colors.

    Key insight:
    - The anchor still solves the placement problem exactly as in the base task; the only new
      step is that the source instance is not discarded completely. Its scaled edge pixels are
      overlaid at the end, creating a colored trace of the old sprite around the new one.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h0 = anc0[2] - anc0[0] + 1
    base_anc_w0 = anc0[3] - anc0[1] + 1
    base_anc_h1 = anc1[2] - anc1[0] + 1
    base_anc_w1 = anc1[3] - anc1[1] + 1
    assert (base_anc_h0, base_anc_w0) == (base_anc_h1, base_anc_w1), "prototype anchors must share the same size"

    out = g.copy()

    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, ar1, ac1 = anc
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        scale_h = max(1, inst_anc_h // max(1, base_anc_h0))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w0))
        scale = int(max(scale_h, scale_w))

        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            panc = _pick_anchor_bbox(p, bg)
            pr0, pc0 = int(panc[0] * scale), int(panc[1] * scale)
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

            score = 0
            ph, pw = sp.shape
            rr0 = max(0, top)
            cc0 = max(0, left)
            rr1 = min(h, top + ph)
            cc1 = min(w, left + pw)
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            sub_g = g[rr0:rr1, cc0:cc1]
            sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
            m = sub_p != bg
            if np.any(m):
                score = int(np.sum(sub_g[m] == sub_p[m]))
            if score > best_score:
                best_score = score
                best_idx = idx

        repl_idx = 1 - best_idx

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        panc = _pick_anchor_bbox(p, bg)
        pr0, pc0 = int(panc[0] * scale), int(panc[1] * scale)
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[m] = sub_p[m]
        out[rr0:rr1, cc0:cc1] = out_slice

        edge = _edge_mask(arr, bg)
        edge_slice = out[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]]
        edge_slice[edge] = arr[edge]
        out[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]] = edge_slice

    return out


# --------------- generator (o5x, D14 variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.2

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

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
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _make_labels() -> np.ndarray:
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
        entries = weights_by_role[role]
        kinds, wts = zip(*entries)
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


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
    a_cols = random.sample(pool, 2)
    remaining = [c for c in pool if c not in a_cols]
    if random.random() < COLOR_SHARE_PROB:
        b_cols = [random.choice(a_cols), random.choice(remaining)]
        random.shuffle(b_cols)
    else:
        b_cols = random.sample(remaining, 2)

    def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = cols[0]
        proto[lab == 3] = cols[1]
        return proto

    return _colorize(labels, a_cols), _colorize(np.flipud(labels), b_cols)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))

    records = []
    for out_type, scale in instances:
        target_patch = _scale(protos[out_type], scale)
        source_patch = _scale(protos[1 - out_type], scale)
        output_patch, source_pos = _compose_swapped_patch(source_patch, target_patch, bg)
        records.append(
            {
                "source_patch": source_patch,
                "output_patch": output_patch,
                "source_pos": source_pos,
                "size_key": max(output_patch.shape),
            }
        )
    random.shuffle(records)
    records.sort(key=lambda item: item["size_key"], reverse=True)

    max_dim = max(max(rec["output_patch"].shape) for rec in records)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for rec in records:
        marker = np.full(rec["output_patch"].shape, 1, dtype=int)
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

        paste(reserve, marker, ur, uc)
        paste(out, rec["output_patch"], ur, uc, transparent=bg)
        src_r, src_c = rec["source_pos"]
        paste(inp, rec["source_patch"], ur + src_r, uc + src_c, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D14 input")
