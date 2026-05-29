# Combo variant of c7f57c3e (index 25)
# combo_spec: D4/0+D13/0+D15/1
# Name: 单色精灵 + 干扰精灵 + 标记像素分组
#
# Design analysis for the combo variant
# -------------------------------------
# spec adaptation:
#   grid:
#     size: 18-28 square
#     bg: one uniform color, never equal to body or marker colors
#   active prototypes:
#     count: exactly 2
#     body: both prototypes share one monochrome color-1 body shape
#     marker:
#       proto 0: one red marker pixel/block attached on the left side
#       proto 1: one green marker pixel/block attached on the right side
#     anchor:
#       the color-1 body bbox is the alignment anchor; marker pixels do not define placement
#   distractors:
#     count: 1-2
#     form: compact monochrome sprites with no marker and no color-1 body pixels
#     role: they never swap and remain unchanged
#   sprites:
#     active count: 3-4 with scales 1-3, including both groups at scale 1 and exactly one active sprite at scale 2 or 3
#     placement: all objects are non-overlapping under the union bbox reserved for each active swap
#   rule:
#     only marked sprites participate; their marker color determines which prototype family they belong to
#     each participating sprite swaps to the opposite prototype at the same scale and aligned by the body bbox
#
# a) Concrete behavior trace
#   Old task:
#     every non-bg component was a real sprite, and the sprite's full multicolor pattern identified
#     which prototype to swap to.
#   New task:
#     the body is monochrome in both families, so the body itself no longer separates the groups.
#     Example at scale 1:
#       input active A: red marker on the left + color-1 body
#       input active B: green marker on the right + same color-1 body
#       output: each marked sprite flips to the opposite side/color marker while its body stays fixed
#       distractor: a small unmarked orange shape remains exactly unchanged
#   Difference:
#     marker presence now decides "participates or distractor", and marker color decides the group.
#     The same visible cue serves both D13 and D15 at once.
#
# b) Complexity calibration
#   - The task is not reduced to arbitrary recoloring: the marker changes both color and side.
#   - There are only two marker colors, and both repeat within each sample family, keeping the mapping learnable.
#   - Distractors are whole compact sprites, not noise, and they are deliberately smaller than the real prototypes,
#     so the old "smallest component defines prototype" shortcut becomes invalid in an observable way.
#
# c) Hidden-assumption audit against the original main()
#   - `components_mask(g != bg)` no longer means every component should be transformed.
#   - `_pick_anchor_bbox()` on every component is no longer safe because distractors have no color-1 body.
#   - Global smallest-area components are no longer guaranteed to be true prototypes because distractors are smaller.
#   - Full-pattern matching is no longer the right discriminator because both active families share the same body.
#   - Fix: classify components first, keep only marked body-bearing sprites as active, infer scale from the body bbox,
#     and use the marker color as the group id.
#
# d) Placement legality
#   - The generator is output-first for active sprites and reserves the union rectangle that can contain either
#     left-marked or right-marked version when body-aligned.
#   - Distractors are placed through the same reserve canvas with the same border and padding rules, so neither the
#     swap itself nor the unchanged distractors can overlap or spill outside the canvas.
#
# e) Visual observability
#   - Across examples, a human can see that marked color-1 bodies always change, while unmarked sprites never do.
#   - Among the marked bodies, red-left always turns into green-right and vice versa, making the grouping clue explicit.
#
# f) Visual cleanliness
#   - Bodies, markers, and distractors are all compact and separated by padding.
#   - Marker pixels are attached orthogonally to the body and upscale into clean blocks, so group cues remain visible.
#
# g) Change inferability
#   - Only two rule-bearing colors exist: red and green markers.
#   - Distractor colors do not participate in any mapping and appear only on unmarked components, so the solver can
#     infer that color change is conditioned on marker presence and marker color, not on arbitrary palette remapping.
#
# 9) New solver assumptions and generator guarantees
#   - Every active sprite contains color 1 plus exactly one marker color from {2, 3}; distractors do not.
#   - Among active sprites, the smallest bbox area still yields exactly two distinct scale-1 prototypes.
#   - Both prototypes share the same body mask and differ only in marker color/side; the generator constructs them
#     from one body template so scale inference by body bbox is stable.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


BODY_COLOR = 1
MARKER_COLORS = (2, 3)

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

# --- CFG: distractors ---
DISTRACTOR_COUNTS = [1, 2]
DISTRACTOR_SHAPES = [
    np.array(
        [
            [1, 1],
            [0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
]

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400


def _pick_body_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 body pixels."""
    if np.any(arr == BODY_COLOR):
        ys, xs = np.where(arr == BODY_COLOR)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "active sprite has no body pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _extract_marker_color(arr: np.ndarray, bg: int) -> int | None:
    """Return the unique valid marker color for one active sprite, else None."""
    colors = sorted(int(c) for c in np.unique(arr) if int(c) != bg)
    if BODY_COLOR not in colors:
        return None
    other = [c for c in colors if c != BODY_COLOR]
    if len(other) != 1:
        return None
    if other[0] not in MARKER_COLORS:
        return None
    return other[0]


def _sprite_record(grid: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    marker_color = _extract_marker_color(arr, bg)
    has_body = bool(np.any(arr == BODY_COLOR))
    return {
        "pts": pts,
        "bbox": (r0, c0, r1, c1),
        "arr": arr,
        "marker_color": marker_color,
        "has_body": has_body,
        "is_active": marker_color is not None,
    }


def _body_signature(arr: np.ndarray, bg: int) -> tuple[tuple[int, ...], ...]:
    r0, c0, r1, c1 = _pick_body_bbox(arr, bg)
    mask = (arr[r0 : r1 + 1, c0 : c1 + 1] == BODY_COLOR).astype(int)
    return tuple(tuple(int(v) for v in row) for row in mask)


def _build_proto(body: np.ndarray, marker_color: int, side: str, bg: int) -> np.ndarray:
    """Attach a one-pixel marker on the left or right side of a shared monochrome body."""
    h, w = body.shape
    proto = np.full((h, w + 1), bg, dtype=int)

    if side == "left":
        proto[:, 1:][body == 1] = BODY_COLOR
        rows = np.where(body[:, 0] == 1)[0]
        proto[int(rows[len(rows) // 2]), 0] = marker_color
        return proto

    assert side == "right"
    proto[:, :w][body == 1] = BODY_COLOR
    rows = np.where(body[:, w - 1] == 1)[0]
    proto[int(rows[len(rows) // 2]), w] = marker_color
    return proto


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    body = random.choice(BODY_SHAPES).copy()
    if random.random() < HFLIP_PROB:
        body = np.fliplr(body)
    return (
        _build_proto(body, MARKER_COLORS[0], "left", bg),
        _build_proto(body, MARKER_COLORS[1], "right", bg),
    )


def _aligned_union_layout(
    patch_a: np.ndarray, patch_b: np.ndarray, bg: int
) -> tuple[int, int, int, int, int, int]:
    """Return origin offsets and union size for two body-aligned patches."""
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


def _make_distractor_patch(bg: int, color: int) -> np.ndarray:
    shape = random.choice(DISTRACTOR_SHAPES)
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape == 1] = color
    return patch


def _grid_lower_bound(items: list[np.ndarray]) -> int:
    max_dim = max(max(item.shape) for item in items)
    total_area = sum(int(item.shape[0] * item.shape[1]) for item in items)
    area_bound = int(np.ceil(np.sqrt(total_area))) + 6
    return max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES, area_bound)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap only the marked monochrome sprites to the opposite marker prototype and leave unmarked distractors untouched.

    Input structure:
    - bg is the dominant color. The grid mixes 3-4 marked sprites with 1-2 extra unmarked distractor sprites.
    - Every active sprite has the same color-1 body plus exactly one attached marker color (red or green); distractors do not carry that marker/body combination.

    Rule:
    - Filter components by the marker-bearing monochrome-body pattern, ignoring unmarked distractors.
    - Recover the two scale-1 active prototypes from the smallest active sprites, infer each active sprite's scale from the body bbox, and use marker color to decide which group it belongs to.
    - Replace each active sprite with the opposite prototype at the same scale, aligned by the color-1 body bbox, while copying distractors through unchanged.

    Key insight: Marker pixels serve two jobs at once. Their presence says the component participates at all, and their color says which family it belongs to; the body is only the placement anchor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 4, "grid must contain active sprites and distractors"

    records = [_sprite_record(g, pts, bg) for pts in comps]
    sprites = [rec for rec in records if rec["is_active"]]
    assert len(sprites) >= 3, "grid must contain at least 3 marked sprites"
    assert any(not rec["is_active"] for rec in records), "grid must contain at least one distractor"

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest active sprites must yield exactly 2 prototypes"

    protos = list(uniq.values())[:2]
    marker_to_idx: dict[int, int] = {}
    body_sigs = []
    for idx, proto in enumerate(protos):
        marker_color = _extract_marker_color(proto, bg)
        assert marker_color is not None
        assert marker_color not in marker_to_idx, "prototype marker colors must differ"
        marker_to_idx[marker_color] = idx
        body_sigs.append(_body_signature(proto, bg))
    assert body_sigs[0] == body_sigs[1], "active prototypes must share the same body mask"

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

        marker_color = s["marker_color"]
        assert marker_color in marker_to_idx, "active sprite marker must match one of the prototypes"
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

    distractor_count = random.choice(DISTRACTOR_COUNTS)
    distractor_pool = [c for c in range(10) if c not in (bg, BODY_COLOR, *MARKER_COLORS)]
    if len(distractor_pool) < distractor_count:
        return None
    distractor_colors = random.sample(distractor_pool, distractor_count)
    distractors = [_make_distractor_patch(bg, color) for color in distractor_colors]

    placement_items = []
    for out_type, scale in instances:
        output_patch = _scale_nn(protos[out_type], scale)
        input_patch = _scale_nn(protos[1 - out_type], scale)
        _out_r, _out_c, _in_r, _in_c, uh, uw = _aligned_union_layout(output_patch, input_patch, bg)
        placement_items.append(np.full((uh, uw), BODY_COLOR, dtype=int))
    placement_items.extend(distractors)

    lower = _grid_lower_bound(placement_items)
    if lower > GRID_SIZE_MAX:
        return None

    n = random.randint(lower, GRID_SIZE_MAX)
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

    for patch in distractors:
        try:
            rr, cc = random_free_location(
                reserve,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(reserve, patch, rr, cc, transparent=bg)
        paste(out, patch, rr, cc, transparent=bg)
        paste(inp, patch, rr, cc, transparent=bg)

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
