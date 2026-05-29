# Variant of c7f57c3e
# Dimension: D4 — 源对象特征 (src)
# Change: 多色精灵（polychrome sprites），每个精灵使用不同的颜色组合构成可识别的颜色模式... → 单色精灵加标记像素
#
# Design analysis for the D4 variant
# ----------------------------------
# spec:
#   grid:
#     size: 18-28 square
#     bg: one uniform background color, chosen away from body/marker colors
#   prototypes:
#     count: exactly 2
#     body: one shared monochrome body shape (color 1), 8-connected, with small bg holes allowed
#     marker:
#       proto 0: one red marker block (color 2) attached above the body
#       proto 1: one green marker block (color 3) attached below the body
#     relation:
#       the body shape is the same in both prototypes
#       only the marker color and the marker's vertical side differ
#     anchor:
#       the bbox of the color-1 body is the alignment anchor; it ignores the marker
#   sprites:
#     count: 3-4
#     scale: integer nearest-neighbor scales 1-3, with exactly one sprite at scale 2 or 3
#     balance: both marker groups appear at scale 1
#     placement: non-overlapping under the union bbox of input/output placements, with padding >= 2
#   rule:
#     every sprite switches to the opposite marker prototype at the same scale
#     alignment is by the monochrome body bbox, so the body stays fixed and only the marker side/color flips
#
# a) Concrete behavior trace
#   Old task:
#     a sprite's internal multicolor pattern identifies which prototype it is, then the whole pattern
#     swaps to the other prototype while a color-1 anchor region preserves placement.
#   New task:
#     the body no longer carries the grouping clue; every body is monochrome color 1.
#     Example at scale 1:
#       input:  red marker above + body
#       output: green marker below + same body
#     The body bbox is the anchor, so after replacement the body overlaps maximally and only the
#     attached marker moves from top to bottom or bottom to top.
#   Difference:
#     grouping now comes from an explicit marker pixel color, not from a full polychrome body pattern.
#
# b) Complexity calibration
#   The variant avoids over-simplifying into a trivial recolor-by-id task by keeping:
#   - multiple sprites at mixed scales
#   - two repeated groups in each sample
#   - marker motion across the body, not just a color change in place
#   The design also avoids excess freedom: there are only two marker groups and one shared body shape
#   per sample, so humans can infer the rule from 2-3 examples.
#
# c) Hidden-assumption audit against the original main()
#   - Original `_pick_anchor_bbox(arr, bg)` assumed color 1 exists and can anchor replacement.
#     This remains valid, but color 1 now means the entire monochrome body, not one region inside
#     a multicolor sprite.
#   - Original prototype matching used full-array overlap scoring. In this variant that would make
#     marker reasoning optional, because the bodies are intentionally the same. The new solver instead
#     reads the unique non-bg, non-1 marker color from each sprite and uses that as the group id.
#   - Original code inferred scale from the anchor bbox size. This still works because the body bbox
#     scales cleanly and is identical across the two prototypes.
#   - Clearing only the current component before pasting the replacement is still safe because the
#     generator reserves the union bbox needed by either marker side, so no other sprite can occupy it.
#
# d) Placement legality
#   The generator is still output-first. For each sprite instance it computes the union rectangle that
#   can contain both the current marker-side version and the swapped version when body-aligned. That
#   union rectangle is what gets spaced apart, so replacement never collides with neighbors and never
#   leaves the canvas.
#
# e) Visual observability
#   A human sees that all objects share the same single-color body, while the only meaningful group cue
#   is a red top marker versus a green bottom marker. Across examples, the output consistently flips
#   each marker to the opposite side/color while leaving the body fixed.
#
# f) Visual cleanliness
#   Markers are attached to the body and become scaled blocks under enlargement, so they remain easy to
#   spot. Objects are separated by padding, and no extra decorative colors are introduced.
#
# g) Color-change inference
#   Only two non-body colors participate in the rule: red and green markers. Because their positions
#   are also systematic (above vs below), the color change is not arbitrary and can be inferred from a
#   small number of examples.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


BODY_COLOR = 1
MARKER_COLORS = (2, 3)


def _pick_body_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of the monochrome sprite body (color 1), which is the placement anchor."""
    if np.any(arr == BODY_COLOR):
        ys, xs = np.where(arr == BODY_COLOR)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no body pixels of color 1"


def _marker_info(arr: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """Return the unique marker color and its bbox inside one sprite array."""
    colors = sorted(int(c) for c in np.unique(arr) if int(c) not in (bg, BODY_COLOR))
    assert len(colors) == 1, "each sprite must contain exactly one marker color"
    marker_color = colors[0]
    pts = np.argwhere(arr == marker_color)
    return marker_color, bbox_coords(pts)


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every marked monochrome sprite to the opposite marker prototype while preserving scale and body placement.

    Input structure:
    - bg = most common color. Two small prototype variants appear at various scales.
    - Every sprite has the same monochrome body (color 1), plus one attached marker block whose color identifies the group.

    Rule:
    - Recover the two scale-1 prototypes from the smallest sprites and read their marker colors.
    - For each sprite, infer its scale from the body bbox and decide its group from the marker color.
    - Replace it with the other prototype at the same scale, aligned by the monochrome body bbox rather than by the full sprite bbox.

    Key insight: The body is not the clue anymore — it is the stable anchor. The only group signal is the attached marker, so solving requires separating marker identity from body alignment.
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
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct marker prototypes"

    protos = list(uniq.values())[:2]
    marker_to_idx: dict[int, int] = {}
    for idx, proto in enumerate(protos):
        marker_color, _ = _marker_info(proto, bg)
        assert marker_color not in marker_to_idx, "prototype marker colors must be distinct"
        marker_to_idx[marker_color] = idx

    body0 = _pick_body_bbox(protos[0], bg)
    body1 = _pick_body_bbox(protos[1], bg)
    base_body_h = body0[2] - body0[0] + 1
    base_body_w = body0[3] - body0[1] + 1
    assert base_body_h == body1[2] - body1[0] + 1
    assert base_body_w == body1[3] - body1[1] + 1

    out = g.copy()

    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        br0, bc0, br1, bc1 = _pick_body_bbox(arr, bg)
        inst_body_h = br1 - br0 + 1
        inst_body_w = bc1 - bc0 + 1
        scale_h = max(1, inst_body_h // max(1, base_body_h))
        scale_w = max(1, inst_body_w // max(1, base_body_w))
        scale = int(max(scale_h, scale_w))

        marker_color, _ = _marker_info(arr, bg)
        assert marker_color in marker_to_idx, "sprite marker color must match one of the two prototype groups"
        repl_idx = 1 - marker_to_idx[marker_color]

        for rr, cc in s["pts"]:
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
GRID_MARGIN_FOR_SPRITES = 6

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

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300


def _build_proto(body: np.ndarray, marker_color: int, side: str, bg: int) -> np.ndarray:
    """Embed a shared body in a slightly taller array and attach one marker above or below it."""
    h, w = body.shape
    proto = np.full((h + 1, w), bg, dtype=int)
    if side == "top":
        proto[1: 1 + h][body == 1] = BODY_COLOR
        cols = np.where(body[0] == 1)[0]
        proto[0, int(cols[len(cols) // 2])] = marker_color
        return proto
    assert side == "bottom"
    proto[:h][body == 1] = BODY_COLOR
    cols = np.where(body[-1] == 1)[0]
    proto[h, int(cols[len(cols) // 2])] = marker_color
    return proto


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Create two prototypes that share one monochrome body and differ only by marker side/color."""
    body = random.choice(BODY_SHAPES).copy()
    if random.random() < HFLIP_PROB:
        body = np.fliplr(body)
    return (
        _build_proto(body, MARKER_COLORS[0], "top", bg),
        _build_proto(body, MARKER_COLORS[1], "bottom", bg),
    )


def _aligned_union_layout(
    patch_a: np.ndarray, patch_b: np.ndarray, bg: int
) -> tuple[int, int, int, int, int, int]:
    """Return union placement offsets for two body-aligned patches."""
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
    raise RuntimeError("failed to generate a valid c7f57c3e D4 input")
