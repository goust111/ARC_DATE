# Combo variant of c7f57c3e (index 10)
# combo_spec: D11/0+D13/1
# Name: 局部背景参与交换 + 背景噪声像素

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the fused variant:
# a) Concrete behavior.
#    - Original: only the non-bg motif pixels swapped; holes inside the motif stayed global bg.
#    - Here each instance is an opaque rectangular local patch. Inside that rectangle, color-1
#      and the two variable colors form the motif, while the former "holes" are filled by a
#      prototype-specific local background color. Swapping therefore changes both foreground
#      and local background. Separately, a few isolated single pixels are scattered on the
#      global background and never participate in the swap.
#    - Small example: if proto A is a 4x3 patch with local bg 6 and proto B is the flipped
#      counterpart with local bg 8, then an A-instance turns into the aligned B-instance and
#      all cells that were 6 inside its patch become 8. A far-away singleton noise pixel of
#      color 7 remains 7 in the output.
# b) Complexity calibration.
#    - The local patch stays rectangular and repeats at 3-4 scales/instances, so the task is
#      visibly different from the seed but still structured enough to infer from 2-4 examples.
#    - Noise is kept to a small count and forced to stay isolated, so it is noticeable without
#      visually drowning out the real objects.
# c) Assumption audit against the seed solver.
#    - `components_mask(g != bg)` no longer means "all components are sprites": singleton
#      components are noise and must be ignored.
#    - Clearing only the sprite foreground is no longer valid: the whole local patch must be
#      replaced because its interior background color also changes.
#    - Matching on non-bg foreground pixels only is no longer the right abstraction. We match
#      full rectangular patches, then still use the color-1 anchor box for aligned placement.
# d) Placement legality.
#    - The generator reserves the union rectangle needed by both the input patch and its swapped
#      output patch, so neither direction clips or overlaps another instance.
#    - Noise is sampled only from cells outside those reserved rectangles and outside each
#      other's 8-neighborhood, which preserves their singleton status.
# e) Visual observability and cleanliness.
#    - Humans can distinguish four pixel roles: global bg, isolated noise singletons, opaque
#      local patch background, and the motif colors inside each patch.
#    - Local background colors are distinct across the two prototype groups and occupy repeated
#      positions, so the "background also swaps" rule is directly visible instead of hidden in
#      a tiny number of cells.
# f) New solver assumptions and generator guarantees.
#    - Every real instance is a full rectangle with area > 1 and contains color 1.
#    - The smallest-area real instances are the scale-1 prototypes and yield exactly two unique
#      patterns.
#    - The most common color is still the global background because the grid size is chosen to
#      leave more global bg cells than the total reserved patch area.


def _pick_anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(arr == 1)
    assert ys.size > 0, "every patch must contain color-1 anchor cells"
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _local_patch_color(arr: np.ndarray) -> int:
    vals2, counts2 = np.unique(arr[arr != 1], return_counts=True)
    assert vals2.size > 0
    best = int(vals2[int(np.argmax(counts2))])
    if np.sum(counts2 == counts2.max()) == 1:
        return best
    corners = [int(arr[0, 0]), int(arr[0, -1]), int(arr[-1, 0]), int(arr[-1, -1])]
    vals, counts = np.unique(np.array(corners, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _extract_patch_records(g: np.ndarray, bg: int) -> tuple[list[dict], list[np.ndarray]]:
    comps = components_mask(g != bg, connectivity=8)
    patches: list[dict] = []
    noise: list[np.ndarray] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        area = int(pts.shape[0])
        bbox = (r0, c0, r1, c1)
        if area == 1:
            noise.append(pts)
            continue
        if area != _bbox_area(bbox):
            continue
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        if not np.any(arr == 1):
            continue
        patches.append({"pts": pts, "bbox": bbox, "arr": arr})
    return patches, noise


def _identify_prototypes(patches: list[dict]) -> list[np.ndarray]:
    assert len(patches) >= 2, "grid must contain at least 2 real patch instances"
    areas = [_bbox_area(s["bbox"]) for s in patches]
    min_area = min(areas)
    base = [s for s, area in zip(patches, areas, strict=False) if area == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest patch instances must yield exactly 2 prototypes"
    return [uniq[key] for key in list(uniq.keys())[:2]]


def _classify_patch(arr: np.ndarray, protos: list[np.ndarray]) -> tuple[int, int]:
    base_h, base_w = protos[0].shape
    scale_h, rem_h = divmod(arr.shape[0], base_h)
    scale_w, rem_w = divmod(arr.shape[1], base_w)
    assert rem_h == 0 and rem_w == 0 and scale_h == scale_w and scale_h >= 1
    scale = int(scale_h)

    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        sp = _scale_nn(proto, scale)
        if sp.shape != arr.shape:
            continue
        score = int(np.sum(sp == arr))
        if score > best_score:
            best_score = score
            best_idx = idx
    assert best_score == arr.size, "each real patch must be an exact scaled prototype"
    return best_idx, scale


def _solve_core(grid: np.ndarray, *, swap_local_bg: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    patches, _noise = _extract_patch_records(g, bg)
    protos = _identify_prototypes(patches)

    out = g.copy()
    for patch in patches:
        arr = patch["arr"]
        r0, c0, r1, c1 = patch["bbox"]
        src_idx, scale = _classify_patch(arr, protos)
        dst_idx = 1 - src_idx

        src_anchor = _pick_anchor_bbox(arr)
        dst_proto = protos[dst_idx]
        dst_patch = _scale_nn(dst_proto, scale)
        dst_anchor = _pick_anchor_bbox(dst_proto)

        if not swap_local_bg:
            dst_patch = dst_patch.copy()
            src_local = _local_patch_color(arr)
            dst_local = _local_patch_color(dst_proto)
            dst_patch[dst_patch == dst_local] = src_local

        top = (r0 + src_anchor[0]) - dst_anchor[0] * scale
        left = (c0 + src_anchor[1]) - dst_anchor[1] * scale

        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        paste(out, dst_patch, top, left)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each local patch to the opposite prototype while preserving scale and anchor placement.

    Input structure:
    - The most common color is the global background. On top of it sit 3-4 opaque rectangular
      local patches built from two prototype families at integer scales.
    - Inside each patch, color 1 marks an anchor sub-bbox, two other colors encode the motif,
      and the remaining cells of the rectangle use a prototype-specific local background color.
      A few isolated non-bg singleton pixels elsewhere are background noise, not sprites.

    Rule:
    - Ignore singleton noise components. Recover the two scale-1 local patch prototypes from the
      smallest real rectangles, identify which prototype each larger patch instantiates, and keep
      its scale unchanged.
    - Replace every real patch with the opposite prototype, aligning by the color-1 anchor box.
      The swap applies to the entire rectangular patch, so the local background color changes
      together with the foreground motif. Global background and singleton noise stay unchanged.

    Key insight: the meaningful object is no longer just the foreground motif but the whole local
    patch around it. Real instances are full rectangles repeated at different scales, whereas the
    distracting noise is only area-1, so robust parsing comes from separating those two classes
    before doing the same anchor-based prototype swap as in the seed task.
    """
    return _solve_core(grid, swap_local_bg=True)


# --------------- generator (combo) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 19
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(3, 3), (4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.85

# --- CFG: row masks ---
ANCHOR_ROW_WEIGHTS = [("center", 0.42), ("outer", 0.33), ("full", 0.25)]
VARIABLE_ROW_WEIGHTS = [("center", 0.7), ("outer", 0.1), ("full", 0.2)]
OUTER_FIXUP_PROB = 0.75

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: noise ---
NOISE_COUNTS = [2, 3, 4]

# --- CFG: generation retries ---
MAX_RETRIES = 400

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _labels_are_valid(labels: np.ndarray) -> bool:
    if len(components_mask(labels > 0, connectivity=8)) != 1:
        return False
    zero_count = int(np.sum(labels == 0))
    if zero_count < 2:
        return False
    role_counts = [int(np.sum(labels == role)) for role in (1, 2, 3)]
    if zero_count <= max(role_counts):
        return False
    corners = [int(labels[0, 0]), int(labels[0, -1]), int(labels[-1, 0]), int(labels[-1, -1])]
    if sum(v == 0 for v in corners) < 2:
        return False
    return True


def _make_labels() -> np.ndarray:
    for _ in range(300):
        h, w = random.choice(PROTO_SHAPES)
        counts = {1: 1, 2: 1, 3: 1}
        if h > 3:
            counts[random.choice([1, 2, 3])] += h - 3

        roles = []
        for role in random.sample([1, 2, 3], 3):
            roles.extend([role] * counts[role])

        masks = []
        weights_by_role = {
            1: ANCHOR_ROW_WEIGHTS,
            2: VARIABLE_ROW_WEIGHTS,
            3: VARIABLE_ROW_WEIGHTS,
        }
        for role in roles:
            kinds, weights = zip(*weights_by_role[role])
            masks.append(_row_mask(random.choices(kinds, weights=weights)[0], w))

        if not any(mask[0] or mask[-1] for mask in masks):
            i = random.randrange(h)
            if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
                masks[i] = _row_mask("outer", w)
            else:
                masks[i] = _row_mask("full", w)

        if all(all(mask) for mask in masks):
            masks[random.randrange(h)] = _row_mask("center", w)

        labels = np.zeros((h, w), dtype=int)
        for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
            for c, bit in enumerate(mask):
                if bit:
                    labels[r, c] = role

        if _labels_are_valid(labels):
            return labels

    raise RuntimeError("failed to build valid local-patch labels")


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    pool = [color for color in range(10) if color not in (bg, 1)]
    local_a, local_b = random.sample(pool, 2)

    remaining = [color for color in pool if color not in (local_a, local_b)]
    a_cols = random.sample(remaining, 2)
    remaining = [color for color in remaining if color not in a_cols]
    if random.random() < COLOR_SHARE_PROB:
        b_cols = [random.choice(a_cols), random.choice(remaining)]
        random.shuffle(b_cols)
    else:
        b_cols = random.sample(remaining, 2)

    def _colorize(lab: np.ndarray, local_bg: int, cols: list[int]) -> np.ndarray:
        proto = np.full(lab.shape, local_bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = cols[0]
        proto[lab == 3] = cols[1]
        return proto

    return _colorize(labels, local_a, a_cols), _colorize(np.flipud(labels), local_b, b_cols)


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    anchor_a = _pick_anchor_bbox(patch_a)
    anchor_b = _pick_anchor_bbox(patch_b)
    dr = anchor_a[0] - anchor_b[0]
    dc = anchor_a[1] - anchor_b[1]
    h, w = patch_a.shape
    return dr, dc, h + abs(dr), w + abs(dc)


def _noise_candidates(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    if mask[r, c]:
        return False
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if mask[rr, cc]:
                return False
    return True


def _scatter_noise(inp: np.ndarray, out: np.ndarray, reserved: np.ndarray, palette: list[int], bg: int) -> int:
    taken = reserved != bg
    placed = 0
    target = random.choice(NOISE_COUNTS)

    candidates = [(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if _noise_candidates(taken, r, c)]
    random.shuffle(candidates)
    for r, c in candidates:
        if placed >= target:
            break
        if not _noise_candidates(taken, r, c):
            continue
        color = random.choice(palette)
        inp[r, c] = color
        out[r, c] = color
        taken[r, c] = True
        placed += 1

    if placed != target:
        raise ValueError("failed to place the requested singleton noise")
    return placed


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    patch_areas = [int(protos[idx].shape[0] * scale * protos[idx].shape[1] * scale) for idx, scale in instances]
    total_patch_area = sum(patch_areas)
    max_dim = max(max(protos[idx].shape) * scale for idx, scale in instances)
    min_side = int(math.ceil(math.sqrt(2 * total_patch_area + 1)))
    n_min = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES, min_side)
    if n_min > GRID_SIZE_MAX:
        return None
    n = random.randint(n_min, GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_type, scale in instances:
        output_patch = _scale_nn(protos[out_type], scale)
        input_patch = _scale_nn(protos[1 - out_type], scale)
        dr, dc, union_h, union_w = _union_bbox(output_patch, input_patch)
        marker = np.full((union_h, union_w), 1, dtype=int)
        try:
            union_r, union_c = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        out_r = union_r + max(0, -dr)
        out_c = union_c + max(0, -dc)
        in_r = out_r + dr
        in_c = out_c + dc

        paste(reserve, marker, union_r, union_c)
        paste(out, output_patch, out_r, out_c)
        paste(inp, input_patch, in_r, in_c)

    palette = sorted({int(v) for v in np.concatenate([proto_a.ravel(), proto_b.ravel()]) if int(v) not in (bg, 1)})
    try:
        noise_count = _scatter_noise(inp, out, reserve, palette, bg)
    except ValueError:
        return None

    if noise_count <= 0:
        return None
    _patches, noise = _extract_patch_records(inp, bg)
    if len(noise) != noise_count:
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
