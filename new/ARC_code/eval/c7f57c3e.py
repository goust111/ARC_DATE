import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location
import random


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype while preserving scale and position.

    Input structure:
    - bg = most common color. Two distinct small sprite patterns (prototypes) appear at various scales.
    - Each sprite instance is a scaled copy of one of the two prototypes, placed on the grid.

    Rule:
    - Identify the two prototype patterns from the smallest-area sprites.
    - For each sprite, determine which prototype it matches and its integer scale factor.
    - Replace it with the other prototype at the same scale, aligned by an internal anchor region (color-1 pixels).

    Key insight: Every sprite toggles to the alternate prototype; the anchor sub-region inside each sprite pins the alignment so swapped sprites land in the correct position regardless of scale.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    # Extract sprite records.
    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    # Determine two base prototypes from the smallest bbox area sprites.
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
    a0r0, a0c0, a0r1, a0c1 = anc0
    base_anc_h = a0r1 - a0r0 + 1
    base_anc_w = a0c1 - a0c0 + 1

    out = g.copy()

    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        arr = s["arr"]
        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, ar1, ac1 = anc
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        # infer integer scale
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))

        # score which prototype matches this sprite when aligned by core bbox
        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            panc = _pick_anchor_bbox(p, bg)
            pr0, pc0, pr1, pc1 = panc
            pr0 *= scale
            pc0 *= scale
            # align anchor top-left to instance anchor top-left (global coords)
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

            # overlap score: count matching non-bg pixels where prototype expects non-bg
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

        # Clear current sprite pixels.
        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        # Place replacement.
        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        panc = _pick_anchor_bbox(p, bg)
        pr0, pc0, pr1, pc1 = panc
        pr0 *= scale
        pc0 *= scale
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

    return out

# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18                  # square grid lower bound
GRID_SIZE_MAX = 28                  # square grid upper bound
GRID_MARGIN_FOR_SPRITES = 6         # min_n = max_sprite_dim + this

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]  # (h, w) choices for label grid
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9] # bg color pool (excludes 1)
COLOR_SHARE_PROB = 0.65             # prob that proto_B shares a color with proto_A

# --- CFG: row-mask weights (anchor role vs variable role) ---
#   each entry: (mask_type, weight)  where mask_type in {"center", "outer", "full"}
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7             # when fixing no-edge row, prob of using outer vs full

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]             # number of sprite instances per grid
BIG_SCALES = [2, 3]                # scale factor for the one large sprite

# --- CFG: placement ---
PLACEMENT_BORDER = 1               # min distance from grid edge
PLACEMENT_PADDING = 2              # min gap between sprite bboxes
PLACEMENT_CONNECTIVITY = 8         # connectivity for padding check

# --- CFG: generation retries ---
MAX_RETRIES = 300

# --- Row-mask patterns indexed by width ---
_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer":  {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full":   {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    """Return a row occupancy mask of the given kind and width."""
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of color-1 (anchor) pixels: (r0, c0, r1, c1)."""
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    """Nearest-neighbor upscale by integer factor k."""
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _make_labels() -> np.ndarray:
    """Build an (h, w) label grid with roles 1 (anchor), 2, 3 and bg holes (0)."""
    h, w = random.choice(PROTO_SHAPES)

    # Assign row counts per role (each role gets at least 1 row)
    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3
    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    # Pick row masks via weighted sampling
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

    # Fix-up: ensure at least one row touches the left/right edge
    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[i] = _row_mask("outer", w)
        else:
            masks[i] = _row_mask("full", w)

    # Fix-up: ensure at least one row has bg holes
    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    # Assemble label grid
    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Create proto_A and proto_B (flipud + recolored) from a shared label grid."""
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


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    """Compute (dr, dc, union_h, union_w) for anchor-aligned placement of two patches.

    dr, dc: offset from the input patch origin to the output patch origin.
    union_h, union_w: bounding box that encloses both patches when anchor-aligned.
    """
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h, w = patch_a.shape
    return dr, dc, h + abs(dr), w + abs(dc)


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid; returns None on placement failure or oracle mismatch."""
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    # Build instance list: 1 of each proto at scale 1, plus 1 at big scale
    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)  # place largest first

    max_dim = max(max(protos[typ].shape) * scale for typ, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_type, scale in instances:
        output_patch = _scale(protos[out_type], scale)
        input_patch = _scale(protos[1 - out_type], scale)
        dr, dc, uh, uw = _union_bbox(output_patch, input_patch)
        marker = np.full((uh, uw), 1, dtype=int)
        try:
            ur, uc = random_free_location(
                reserve, marker, transparent=bg,
                border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        out_r = ur + max(0, -dr)
        out_c = uc + max(0, -dc)
        in_r, in_c = out_r + dr, out_c + dc

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
    raise RuntimeError("failed to generate a valid c7f57c3e input")
