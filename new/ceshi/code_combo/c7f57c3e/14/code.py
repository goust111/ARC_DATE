# Combo variant of c7f57c3e (index 14)
# combo_spec: D4/3+D6/2
# Name: 对角线条纹精灵 + 颜色模式混合

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task fully swapped each sprite to the other prototype. Here each prototype is a
#    5x5 striped sprite with the same footprint and anchor cell but a different diagonal direction:
#    type A uses "\" stripes, type B uses "/" stripes. Example at scale 1:
#      A:  . a b a .
#          a b a b a
#          b a 1 a b
#          a b a b a
#          . a b a .
#      B:  . c d c .
#          d c d c d
#          c d 1 d c
#          d c d c d
#          . c d c .
#    The fused output for an A-sprite is not plain B. It becomes B-geometry colored with
#    {keep A's first stripe color a, take B's second stripe color d}. A B-sprite analogously becomes
#    A-geometry colored with {keep c, take b}. This visibly mixes both groups.
# b) Complexity calibration:
#    Only two prototype groups exist, one stripe direction per group, and each prototype has exactly
#    two non-anchor colors. That keeps the hypothesis space small enough for ARC while still making
#    the mixed-color output visually distinct from both "leave as-is" and "full swap".
# c) Assumption audit:
#    - Solver assumption: the smallest sprites expose both base prototypes. Generator guarantees one
#      scale-1 instance of each type.
#    - Solver assumption: the retained color can be identified without reading numeric color ids.
#      We enforce a fixed stripe phase so the row-major first non-anchor cell always belongs to the
#      retained stripe family.
#    - Solver assumption: mixed output differs from the original full-swap rule. Generator uses four
#      distinct variable colors, so the kept color and borrowed color never collapse accidentally.
# d) Placement legality:
#    All sprites are pasted fully in-bounds with bbox-level reservation and >=2-cell padding, so the
#    transformed sprites remain non-overlapping because the two prototype footprints are identical.
# e/f/g) Visual observability and neatness:
#    The diagonal stripes are wide enough to see at scale 1 and obvious at scale 2. There are only
#    two directions and exactly one mixed color per output sprite, so a human can compare repeated
#    A/B instances and infer "direction toggles; one stripe family keeps its own color, the other
#    stripe family adopts the counterpart group's other color" from a small number of examples.


GRID_SIZE_MIN = 20
GRID_SIZE_MAX = 30
GRID_MARGIN_FOR_SPRITES = 8

BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2]

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

MAX_RETRIES = 400

_FOOTPRINT = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype=bool,
)
_ANCHOR_RC = (2, 2)


def _pick_anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(arr == 1)
    assert ys.size > 0, "sprite array must contain color-1 anchor pixels"
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _make_label_proto(orientation: str) -> np.ndarray:
    labels = np.zeros(_FOOTPRINT.shape, dtype=int)
    for r in range(_FOOTPRINT.shape[0]):
        for c in range(_FOOTPRINT.shape[1]):
            if not _FOOTPRINT[r, c]:
                continue
            if (r, c) == _ANCHOR_RC:
                labels[r, c] = 1
            elif orientation == "backslash":
                labels[r, c] = 2 if ((r - c + 1) % 2 == 0) else 3
            else:
                labels[r, c] = 2 if ((r + c + 1) % 2 == 0) else 3
    return labels


def _colorize(labels: np.ndarray, bg: int, primary: int, secondary: int) -> np.ndarray:
    out = np.full(labels.shape, bg, dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = primary
    out[labels == 3] = secondary
    return out


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(grid != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least two sprite components"
    sprites: list[dict] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    return sprites


def _recover_prototypes(sprites: list[dict]) -> list[np.ndarray]:
    areas = [_bbox_area(s["bbox"]) for s in sprites]
    min_area = min(areas)
    base = [s for s, area in zip(sprites, areas, strict=False) if area == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(int(v) for v in s["arr"].ravel())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest sprites must expose exactly two base prototypes"
    return list(uniq.values())


def _first_variable_color(arr: np.ndarray, bg: int) -> int:
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            val = int(arr[r, c])
            if val not in (bg, 1):
                return val
    raise AssertionError("sprite must contain a non-anchor stripe color")


def _role_colors(arr: np.ndarray, bg: int) -> tuple[int, int]:
    primary = _first_variable_color(arr, bg)
    others = sorted({int(v) for v in arr.ravel() if int(v) not in (bg, 1)})
    assert len(others) == 2, "prototype must contain exactly two stripe colors"
    secondary = others[0] if others[1] == primary else others[1]
    return primary, secondary


def _infer_scale(arr: np.ndarray, base_anchor_bbox: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr)
    inst_h = ar1 - ar0 + 1
    inst_w = ac1 - ac0 + 1
    br0, bc0, br1, bc1 = base_anchor_bbox
    base_h = br1 - br0 + 1
    base_w = bc1 - bc0 + 1
    scale_h = max(1, inst_h // base_h)
    scale_w = max(1, inst_w // base_w)
    scale = int(max(scale_h, scale_w))
    assert scale >= 1, "scale must be positive"
    return scale


def _match_sprite(
    grid: np.ndarray,
    sprite: dict,
    prototypes: list[np.ndarray],
    bg: int,
    scale: int,
) -> int:
    h, w = grid.shape
    r0, c0, _r1, _c1 = sprite["bbox"]
    arr = sprite["arr"]
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr)
    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(prototypes):
        scaled = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto)
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + scaled.shape[0])
        cc1 = min(w, left + scaled.shape[1])
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        sub_g = grid[rr0:rr1, cc0:cc1]
        sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub_p != bg
        if not np.any(mask):
            continue
        score = int(np.sum(sub_g[mask] == sub_p[mask]))
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _mixed_partner_proto(src_proto: np.ndarray, dst_proto: np.ndarray, bg: int) -> np.ndarray:
    src_primary, _src_secondary = _role_colors(src_proto, bg)
    dst_primary, dst_secondary = _role_colors(dst_proto, bg)
    mixed = np.full(dst_proto.shape, bg, dtype=int)
    mixed[dst_proto == 1] = 1
    mixed[dst_proto == dst_primary] = src_primary
    mixed[dst_proto == dst_secondary] = dst_secondary
    return mixed


def _full_swap_partner_proto(dst_proto: np.ndarray) -> np.ndarray:
    return dst_proto.copy()


def _transform(grid: np.ndarray, *, mixed: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    sprites = _extract_sprites(g, bg)
    prototypes = _recover_prototypes(sprites)
    base_anchor_bbox = _pick_anchor_bbox(prototypes[0])

    out = g.copy()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        arr = sprite["arr"]
        ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr)
        scale = _infer_scale(arr, base_anchor_bbox)
        src_idx = _match_sprite(g, sprite, prototypes, bg, scale)
        dst_idx = 1 - src_idx

        for rr, cc in sprite["pts"]:
            out[rr, cc] = bg

        if mixed:
            replacement = _mixed_partner_proto(prototypes[src_idx], prototypes[dst_idx], bg)
        else:
            replacement = _full_swap_partner_proto(prototypes[dst_idx])
        scaled = _scale_nn(replacement, scale)

        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(replacement)
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + scaled.shape[0])
        cc1 = min(w, left + scaled.shape[1])
        sub = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[mask] = sub[mask]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild every striped sprite in the opposite diagonal orientation while mixing its colors.

    Input structure:
    - The background is the most common color. All foreground components are scaled copies of one of
      two 5x5 diagonal-stripe prototypes with the same footprint and the same color-1 anchor cell.
    - One prototype uses "\" stripes and the other uses "/" stripes. Each prototype has exactly two
      stripe colors besides the anchor.

    Rule:
    - Recover the two base prototypes from the smallest sprites, then determine each sprite's type and
      integer scale from its anchor block.
    - Replace each sprite with the opposite stripe direction at the same scale, but do not fully copy
      the partner colors: keep the sprite's own primary stripe color and use the partner prototype's
      secondary stripe color for the other stripe family.

    Key insight:
    - The stripe phase is fixed so the row-major first non-anchor cell always identifies the stripe
      family that survives the transformation, while the color-1 anchor keeps scale and placement
      aligned when the diagonal direction flips.
    """
    return _transform(grid, mixed=True)


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels_a = _make_label_proto("backslash")
    labels_b = _make_label_proto("slash")
    pool = [c for c in range(10) if c not in (bg, 1)]
    a_primary, a_secondary, b_primary, b_secondary = random.sample(pool, 4)
    proto_a = _colorize(labels_a, bg, a_primary, a_secondary)
    proto_b = _colorize(labels_b, bg, b_primary, b_secondary)
    return proto_a, proto_b


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    prototypes = [proto_a, proto_b]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(prototypes[idx].shape[0] * scale for idx, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    grid = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for proto_idx, scale in instances:
        patch = _scale_nn(prototypes[proto_idx], scale)
        marker = np.full(patch.shape, 1, dtype=int)
        try:
            r0, c0 = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(reserve, marker, r0, c0)
        paste(grid, patch, r0, c0, transparent=bg)

    try:
        out = main(grid)
        full_swap = _transform(grid, mixed=False)
    except Exception:
        return None

    if np.array_equal(out, full_swap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused c7f57c3e input")
