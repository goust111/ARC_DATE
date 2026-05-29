# Variant of c7f57c3e
# Dimension: D1 — 分组匹配条件 (match_cond)
# Change: 按颜色模式（如 teal/green 组 vs red/green 组）将精灵分为两组，同组精灵共享相同颜色组合... → 按空间位置分组

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D1 variant
# ----------------------------------
# spec_variant:
#   idea:
#     - Keep the original world model: exactly two multicolor prototypes, integer scaling,
#       anchor-color 1 for alignment, and prototype replacement at fixed positions.
#     - Change only the grouping rule. The original task groups by "which prototype does this
#       sprite currently look like?" and toggles every sprite individually.
#     - This variant groups by spatial rows instead: sprites are laid out as two upper sprites
#       and two lower sprites, paired by left-to-right order. Each upper/lower pair exchanges
#       prototype identity across the row boundary.
#   concrete_behavior:
#     - Input contains four disconnected sprites in a 2x2 layout.
#     - Sort sprites into an upper row and a lower row. Inside each row, sort by x-position.
#     - Pair upper[i] with lower[i]. The output at upper[i] uses lower[i]'s prototype;
#       the output at lower[i] uses upper[i]'s prototype.
#     - Each slot keeps its own scale and anchor position, so only the prototype pattern changes.
#   why_this_is_visibly_different_from_the_original:
#     - The generator enforces one mixed pair (top/bottom use different prototypes) and one
#       same-type pair (top/bottom use the same prototype).
#     - Mixed pairs visibly swap. Same-type pairs stay unchanged.
#     - Therefore the old "every sprite toggles by its own color pattern" rule is wrong on
#       every generated sample, while the new spatial-pair rule is supported by repetition.
#   affected_solver_assumptions:
#     - The original per-sprite replacement index `1 - best_idx` is no longer valid, because
#       a sprite may now remain unchanged when its vertical partner has the same prototype.
#     - Prototype recovery from the smallest sprites still works, but the generator must ensure
#       that the scale-1 sprites include both prototype types. We guarantee this by putting the
#       only enlarged sprite inside the same-type pair, leaving the mixed pair at scale 1.
#     - Row grouping now assumes a clear vertical gap between the upper and lower bands; the
#       generator constructs a 2x2 slot layout with explicit row spacing.
#   placement_legality:
#     - Each slot reserves the union bbox of its input patch and output patch, so swapping to
#       the partner prototype cannot push any sprite out of bounds.
#     - Row/column blocks are separated by fixed gaps, preventing overlaps and keeping the four
#       sprites visually tidy and easy to parse as two rows.
#   observability_and_color_reasoning:
#     - Only two prototype identities exist, repeated across the four sprites.
#     - The controlling signal is visible geometry: "same column, opposite row".
#     - No hidden ordering by color id or numeric value is used anywhere.


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick a stable anchor bbox inside the sprite array (color-1 pixels; always present)."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _match_proto_index(
    grid: np.ndarray,
    arr: np.ndarray,
    bbox: tuple[int, int, int, int],
    protos: list[np.ndarray],
    scale: int,
    bg: int,
) -> int:
    """Return the prototype index whose scaled patch best matches the sprite at this location."""
    h, w = grid.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)

    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        sp = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        top = (r0 + ar0) - pr0 * scale
        left = (c0 + ac0) - pc0 * scale

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            continue

        sub_g = grid[rr0:rr1, cc0:cc1]
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _place_scaled_proto(
    out: np.ndarray,
    proto: np.ndarray,
    scale: int,
    anchor_rc: tuple[int, int],
    bg: int,
) -> None:
    """Paste a scaled prototype so its anchor top-left lands on `anchor_rc`."""
    sp = _scale_nn(proto, scale)
    pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
    top = int(anchor_rc[0] - pr0 * scale)
    left = int(anchor_rc[1] - pc0 * scale)
    paste(out, sp, top, left, transparent=bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange prototype identity between vertically aligned sprite pairs.

    Input structure:
    - bg = most common color. The grid contains four disconnected sprite instances built from
      exactly two prototype patterns, possibly at two scales.
    - The sprites form two horizontal bands: two upper sprites and two lower sprites. Within a
      band, left-to-right order determines which upper sprite pairs with which lower sprite.

    Rule:
    - Recover the two prototype patterns from the smallest sprites, then determine each sprite's
      scale and current prototype identity.
    - Split the sprites into an upper row and a lower row, sort both rows by column, and pair
      them by index.
    - Replace each sprite with the prototype used by its vertical partner, while keeping the
      sprite's own scale and anchor-aligned position.

    Key insight:
    - The grouping cue is spatial rather than chromatic: prototype exchange happens across the
      row boundary for corresponding columns. Mixed pairs swap, while same-type pairs stay the same.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) == 4, "grid must contain exactly 4 sprite components"

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

    protos = list(uniq.values())[:2]
    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h = anc0[2] - anc0[0] + 1
    base_anc_w = anc0[3] - anc0[1] + 1
    assert (anc1[2] - anc1[0] + 1, anc1[3] - anc1[1] + 1) == (base_anc_h, base_anc_w)

    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(s["arr"], bg)
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))
        s["scale"] = scale
        s["anchor_rc"] = (int(r0 + ar0), int(c0 + ac0))
        s["proto_idx"] = _match_proto_index(g, s["arr"], s["bbox"], protos, scale, bg)
        s["center_r"] = (r0 + r1) / 2.0
        s["center_c"] = (c0 + c1) / 2.0

    by_row = sorted(sprites, key=lambda s: s["center_r"])
    gaps = [by_row[i + 1]["center_r"] - by_row[i]["center_r"] for i in range(len(by_row) - 1)]
    split = int(np.argmax(gaps)) + 1
    upper = by_row[:split]
    lower = by_row[split:]
    assert len(upper) == len(lower) == 2, "sprites must form two rows of two"
    assert max(s["center_r"] for s in upper) < min(s["center_r"] for s in lower)

    upper = sorted(upper, key=lambda s: s["center_c"])
    lower = sorted(lower, key=lambda s: s["center_c"])

    out = np.full_like(g, bg)
    for top_s, bot_s in zip(upper, lower, strict=True):
        _place_scaled_proto(out, protos[bot_s["proto_idx"]], top_s["scale"], top_s["anchor_rc"], bg)
        _place_scaled_proto(out, protos[top_s["proto_idx"]], bot_s["scale"], bot_s["anchor_rc"], bg)

    return out


# --------------- generator ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 20
GRID_SIZE_MAX = 28
PLACEMENT_BORDER = 1
ROW_GAP_RANGE = (2, 4)
COL_GAP_RANGE = (3, 5)
MAX_RETRIES = 300

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
BIG_SCALES = [2]

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
        kinds, wts = zip(*weights_by_role[role])
        chosen = random.choices(kinds, weights=wts)[0]
        masks.append(_row_mask(chosen, w))

    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        masks[i] = _row_mask("outer", w) if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB else _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks, strict=True)):
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


def _aligned_union_bbox(output_patch: np.ndarray, input_patch: np.ndarray) -> tuple[int, int, int, int]:
    oa = _anchor_bbox(output_patch)
    ia = _anchor_bbox(input_patch)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]

    top = min(0, dr)
    left = min(0, dc)
    bottom = max(output_patch.shape[0], dr + input_patch.shape[0])
    right = max(output_patch.shape[1], dc + input_patch.shape[1])
    return int(dr), int(dc), int(bottom - top), int(right - left)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    same_col = random.randrange(2)
    same_type = random.randrange(2)
    diff_top_type = random.randrange(2)
    diff_bottom_type = 1 - diff_top_type

    input_types: dict[tuple[int, int], int] = {}
    for col in range(2):
        if col == same_col:
            input_types[(0, col)] = same_type
            input_types[(1, col)] = same_type
        else:
            input_types[(0, col)] = diff_top_type
            input_types[(1, col)] = diff_bottom_type

    big_row = random.randrange(2)
    big_scale = random.choice(BIG_SCALES)
    scales = {(row, col): 1 for row in range(2) for col in range(2)}
    scales[(big_row, same_col)] = big_scale

    slot_data = {}
    for row in range(2):
        for col in range(2):
            in_type = input_types[(row, col)]
            out_type = input_types[(1 - row, col)]
            scale = scales[(row, col)]
            output_patch = _scale(protos[out_type], scale)
            input_patch = _scale(protos[in_type], scale)
            dr, dc, uh, uw = _aligned_union_bbox(output_patch, input_patch)
            slot_data[(row, col)] = {
                "input_patch": input_patch,
                "output_patch": output_patch,
                "dr": dr,
                "dc": dc,
                "uh": uh,
                "uw": uw,
            }

    row_heights = [max(slot_data[(row, col)]["uh"] for col in range(2)) for row in range(2)]
    col_widths = [max(slot_data[(row, col)]["uw"] for row in range(2)) for col in range(2)]
    row_gap = random.randint(*ROW_GAP_RANGE)
    col_gap = random.randint(*COL_GAP_RANGE)

    base_h = row_heights[0] + row_heights[1] + row_gap
    base_w = col_widths[0] + col_widths[1] + col_gap
    min_n = max(GRID_SIZE_MIN, base_h + 2 * PLACEMENT_BORDER, base_w + 2 * PLACEMENT_BORDER)
    if min_n > GRID_SIZE_MAX:
        return None

    n = random.randint(min_n, GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)

    extra_h = n - (base_h + 2 * PLACEMENT_BORDER)
    extra_w = n - (base_w + 2 * PLACEMENT_BORDER)
    top_extra = random.randint(0, extra_h)
    left_extra = random.randint(0, extra_w)

    row_starts = [
        PLACEMENT_BORDER + top_extra,
        PLACEMENT_BORDER + top_extra + row_heights[0] + row_gap,
    ]
    col_starts = [
        PLACEMENT_BORDER + left_extra,
        PLACEMENT_BORDER + left_extra + col_widths[0] + col_gap,
    ]

    for row in range(2):
        for col in range(2):
            info = slot_data[(row, col)]
            ur = row_starts[row] + random.randint(0, row_heights[row] - info["uh"])
            uc = col_starts[col] + random.randint(0, col_widths[col] - info["uw"])

            out_r = ur + max(0, -info["dr"])
            out_c = uc + max(0, -info["dc"])
            in_r = out_r + info["dr"]
            in_c = out_c + info["dc"]

            paste(out, info["output_patch"], out_r, out_c, transparent=bg)
            paste(inp, info["input_patch"], in_r, in_c, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D1 variant input")
