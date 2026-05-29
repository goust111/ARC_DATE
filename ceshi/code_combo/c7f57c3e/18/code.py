# Combo variant of c7f57c3e (index 18)
# combo_spec: D1/3+D4/0
# Name: 按连通区域分组 + 单色精灵加标记
#
# Design analysis for the combo variant
# -------------------------------------
# spec:
#   grid:
#     size: 18-28 square
#     bg: one uniform background color, distinct from body/marker colors
#   objects:
#     unit of transformation: a connected cluster, not a lone sprite
#     each cluster contains two monochrome sprite bodies (color 1) joined by a color-1 bridge
#     each body carries one attached marker block:
#       state 0 -> red marker above the body
#       state 1 -> green marker below the body
#   prototypes:
#     count: exactly 2 cluster prototypes per sample
#     shared structure:
#       same body shape, same two body-slot positions, same bridge, same body-color bbox
#     difference:
#       the two prototypes differ only in how the two slot markers are assigned
#       pattern A = (top-red, top-red)
#       pattern B = (bottom-green, top-red)
#     non-degeneracy:
#       B is deliberately not the slotwise opposite of A, so "flip each marked body independently"
#       does not match the intended output
#   clusters:
#     count: 3-4
#     scale: integer nearest-neighbor scales 1-3, with exactly one scale 2 or 3 cluster
#     balance: both cluster prototypes appear at scale 1
#     placement: non-overlapping under the union bbox of input/output placements, padding >= 2
#   rule:
#     each connected cluster swaps to the other cluster prototype at the same scale
#     alignment is by the bbox of all color-1 pixels, so both bodies and the bridge stay fixed
#
# a) Concrete behavior trace
#   Old task:
#     every connected component was a single polychrome sprite, and the sprite's own colors told
#     you which opposite prototype to paste back.
#   New fused task:
#     every connected component is a two-body cluster. The bodies are monochrome, the only varying
#     cue is the attached marker pattern across the two slots.
#     Example at scale 1:
#       input cluster:  [red-top | red-top]
#       output cluster: [green-bottom | red-top]
#     The body pair and the bridge remain fixed; only the cluster-level marker pattern changes.
#   Difference:
#     the solver must reason at the connected-cluster level. Treating each marked body as an
#     independent D4 sprite would produce the wrong output because the chosen pattern pair is not
#     slotwise complementary.
#
# b) Complexity calibration
#   This stays readable because there are still only two repeated prototype families and only one
#   large instance per grid. It is not over-simplified because the cluster, not the single body, is
#   the transformation unit, and the two prototypes differ in a visible but nontrivial marker layout.
#
# c) Hidden-assumption audit against the original main()
#   - Original component extraction assumed "one component == one sprite". Here one component is a
#     cluster of two linked bodies, so replacement must operate on the whole cluster at once.
#   - Original anchor logic assumed color 1 exists. That still holds, but color 1 now denotes the
#     entire cluster skeleton: both bodies plus their bridge.
#   - Original prototype matching used full-array overlap after anchor alignment. That remains valid
#     because the two cluster prototypes share the same color-1 scaffold and differ only by markers.
#   - Original scale inference relied on a stable anchor bbox. That is preserved by keeping the body
#     scaffold identical across both cluster prototypes.
#   - A new risk is accidental reliance on per-body marker flipping. The generator avoids that by
#     using a non-complementary prototype pair, so independent body toggling disagrees with the true
#     cluster swap.
#
# d) Placement legality
#   The generator is still output-first. For each cluster instance it reserves the union rectangle of
#   the input and output placements after body-bbox alignment, so swapping never collides with other
#   clusters and never leaves the canvas.
#
# e) Visual observability
#   A human sees two repeated connected clusters, each made of two same-colored bodies tied together.
#   Across examples, one cluster pattern turns into the other whole-cluster pattern while the body
#   scaffold stays fixed. That makes the connected-region grouping visible without relying on hidden
#   numeric ordering.
#
# f) Visual cleanliness
#   There are only three active colors besides the background: body blue (1), red markers, and green
#   markers. The bridge is the same body color as the bodies, so the grouped structure is easy to see.
#   Padding keeps clusters separated from one another.
#
# g) Change inferability
#   Marker colors change with a structural cue: which slot inside the connected cluster owns the green
#   bottom marker. Because there are only two cluster patterns and both repeat within a sample set,
#   humans can infer the swap from a few examples.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


BODY_COLOR = 1
TOP_MARKER_COLOR = 2
BOTTOM_MARKER_COLOR = 3
SLOT_GAP = 1
CLUSTER_PATTERNS = ((0, 0), (1, 0))


def _pick_body_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of all color-1 pixels, which form the cluster scaffold."""
    if np.any(arr == BODY_COLOR):
        ys, xs = np.where(arr == BODY_COLOR)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "cluster array has no body pixels of color 1"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _marker_col(body: np.ndarray, which: str) -> int:
    row = body[0] if which == "top" else body[-1]
    cols = np.where(row == 1)[0]
    return int(cols[len(cols) // 2])


def _build_cluster_proto(body: np.ndarray, states: tuple[int, int], bg: int) -> np.ndarray:
    """Create one connected two-slot cluster prototype from a shared body shape and slot states."""
    h, w = body.shape
    proto = np.full((h + 2, 2 * w + SLOT_GAP), bg, dtype=int)
    slot_cols = [0, w + SLOT_GAP]
    body_top = 1

    for slot_idx, slot_c0 in enumerate(slot_cols):
        body_slice = proto[body_top : body_top + h, slot_c0 : slot_c0 + w]
        body_slice[body == 1] = BODY_COLOR
        proto[body_top : body_top + h, slot_c0 : slot_c0 + w] = body_slice

        state = int(states[slot_idx])
        if state == 0:
            proto[0, slot_c0 + _marker_col(body, "top")] = TOP_MARKER_COLOR
        else:
            proto[h + 1, slot_c0 + _marker_col(body, "bottom")] = BOTTOM_MARKER_COLOR

    bridge_body_row = h // 2
    bridge_r = body_top + bridge_body_row
    left_edge = int(np.where(body[bridge_body_row] == 1)[0].max())
    right_edge = int(np.where(body[bridge_body_row] == 1)[0].min())
    start_c = slot_cols[0] + left_edge + 1
    end_c = slot_cols[1] + right_edge - 1
    if start_c <= end_c:
        proto[bridge_r, start_c : end_c + 1] = BODY_COLOR

    tail_c = (slot_cols[0] + w + slot_cols[1] - 1) // 2
    proto[bridge_r : h + 2, tail_c] = BODY_COLOR

    return proto


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    body = random.choice(BODY_SHAPES).copy()
    if random.random() < HFLIP_PROB:
        body = np.fliplr(body)
    return (
        _build_cluster_proto(body, CLUSTER_PATTERNS[0], bg),
        _build_cluster_proto(body, CLUSTER_PATTERNS[1], bg),
    )


def _aligned_union_layout(
    patch_a: np.ndarray, patch_b: np.ndarray, bg: int
) -> tuple[int, int, int, int, int, int]:
    """Return union placement offsets for two body-aligned cluster patches."""
    a_body = _pick_body_bbox(patch_a, bg)
    b_body = _pick_body_bbox(patch_b, bg)
    dr = a_body[0] - b_body[0]
    dc = a_body[1] - b_body[1]

    rmin = min(0, dr)
    cmin = min(0, dc)
    rmax = max(patch_a.shape[0], dr + patch_b.shape[0])
    cmax = max(patch_a.shape[1], dc + patch_b.shape[1])

    out_r = -rmin
    out_c = -cmin
    in_r = dr - rmin
    in_c = dc - cmin
    return int(out_r), int(out_c), int(in_r), int(in_c), int(rmax - rmin), int(cmax - cmin)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every connected marked cluster to the other cluster prototype while preserving scale and scaffold placement.

    Input structure:
    - bg = most common color. Two small connected cluster prototypes appear at various scales.
    - Every cluster is a pair of monochrome bodies joined by a color-1 bridge, with per-body marker pixels attached above or below.

    Rule:
    - Recover the two scale-1 cluster prototypes from the smallest connected clusters.
    - For each cluster, infer its integer scale from the color-1 scaffold bbox and determine which prototype it matches.
    - Replace the whole cluster with the other prototype at the same scale, aligned by the bbox of all color-1 pixels.

    Key insight: The transform acts on a connected region made of multiple marked bodies. The body scaffold is the anchor, but the deciding signal is the cluster-level marker arrangement, not an independent flip of each body.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 connected clusters"

    clusters = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        clusters.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((c["bbox"][2] - c["bbox"][0] + 1) * (c["bbox"][3] - c["bbox"][1] + 1)) for c in clusters]
    min_area = min(areas)
    base = [c for c, area in zip(clusters, areas, strict=False) if area == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for cluster in base:
        key = tuple(cluster["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = cluster["arr"]
    assert len(uniq) == 2, "smallest clusters must yield exactly 2 distinct cluster prototypes"

    protos = list(uniq.values())[:2]
    body0 = _pick_body_bbox(protos[0], bg)
    body1 = _pick_body_bbox(protos[1], bg)
    base_body_h = body0[2] - body0[0] + 1
    base_body_w = body0[3] - body0[1] + 1
    assert base_body_h == body1[2] - body1[0] + 1
    assert base_body_w == body1[3] - body1[1] + 1

    out = g.copy()

    for cluster in clusters:
        r0, c0, _r1, _c1 = cluster["bbox"]
        arr = cluster["arr"]
        br0, bc0, br1, bc1 = _pick_body_bbox(arr, bg)
        inst_body_h = br1 - br0 + 1
        inst_body_w = bc1 - bc0 + 1
        scale_h = max(1, inst_body_h // max(1, base_body_h))
        scale_w = max(1, inst_body_w // max(1, base_body_w))
        scale = int(max(scale_h, scale_w))

        best_idx = 0
        best_score = -1
        for idx, proto in enumerate(protos):
            scaled = _scale_nn(proto, scale)
            pbr0, pbc0, _pbr1, _pbc1 = _pick_body_bbox(proto, bg)
            top = (r0 + br0) - pbr0 * scale
            left = (c0 + bc0) - pbc0 * scale

            ph, pw = scaled.shape
            rr0 = max(0, top)
            cc0 = max(0, left)
            rr1 = min(h, top + ph)
            cc1 = min(w, left + pw)
            if rr0 >= rr1 or cc0 >= cc1:
                continue

            sub_g = g[rr0:rr1, cc0:cc1]
            sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
            m = sub_p != bg
            if np.any(m):
                score = int(np.sum(sub_g[m] == sub_p[m]))
                if score > best_score:
                    best_score = score
                    best_idx = idx

        repl_idx = 1 - best_idx

        for rr, cc in cluster["pts"]:
            out[rr, cc] = bg

        proto = protos[repl_idx]
        scaled = _scale_nn(proto, scale)
        pbr0, pbc0, _pbr1, _pbc1 = _pick_body_bbox(proto, bg)
        top = (r0 + br0) - pbr0 * scale
        left = (c0 + bc0) - pbc0 * scale

        ph, pw = scaled.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[m] = sub_p[m]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 1

# --- CFG: prototypes ---
BG_CHOICES = [0, 4, 5, 6, 7, 8, 9]
BODY_SHAPES = [
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=int,
    ),
]
HFLIP_PROB = 0.5

# --- CFG: cluster instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid; return None on placement failure or oracle mismatch."""
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

    max_dim = max(max(protos[typ].shape) * scale for typ, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_type, scale in instances:
        output_patch = _scale_nn(protos[out_type], scale)
        input_patch = _scale_nn(protos[1 - out_type], scale)
        out_r, out_c, in_r, in_c, uh, uw = _aligned_union_layout(output_patch, input_patch, bg)
        marker = np.full((uh, uw), BODY_COLOR, dtype=int)
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
        paste(out, output_patch, ur + out_r, uc + out_c, transparent=bg)
        paste(inp, input_patch, ur + in_r, uc + in_c, transparent=bg)

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
