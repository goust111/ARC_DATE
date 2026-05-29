# Variant of c7f57c3e
# Dimension: D4 — 源对象特征 (src)
# Change: 多色精灵（polychrome sprites），每个精灵使用不同的颜色组合构成可识别的颜色模式... → 嵌套精灵

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D4 nested-sprite variant
# ------------------------------------------------
# variant_spec:
#   grid:
#     shape: square, 18-28 on each side
#     background: one uniform bg color
#   objects:
#     count: 3-4 connected components
#     base_sprite:
#       size: 5x5 before scaling
#       outer_layer: a solid 1-cell border ring, all one color
#       inner_layer: a 3x3 asymmetric multicolor micro-sprite with bg holes
#       nesting: the inner 3x3 sits inside the border ring, so the object reads as
#         "large shell containing a small sprite" instead of one flat color pattern
#     groups:
#       exactly 2 groups, distinguished only by outer border color
#       each group owns one inner prototype in the input
#     scale:
#       integer nearest-neighbor scale 1-3
#       both groups appear at scale 1
#       exactly 1 larger instance at scale 2 or 3
#   rule:
#     keep each sprite's outer border, bbox, and scale unchanged
#     swap only the inner 3x3 prototype between the two outer-color groups
#
# behavior_trace:
#   original_logic:
#     the whole component was one prototype, so replacing a sprite meant swapping the
#     complete pattern and re-aligning via a color-1 anchor sub-bbox
#   variant_logic:
#     the complete component now contains two semantic layers:
#       1. outer shell = group identity and spatial frame
#       2. inner micro-sprite = payload that gets exchanged
#     example at scale 1:
#       input  group A = [outer A] + [inner A]
#       input  group B = [outer B] + [inner B]
#       output group A = [outer A] + [inner B]
#       output group B = [outer B] + [inner A]
#     difference:
#       the old solver could treat every sprite as an indivisible flat prototype;
#       the new solver must parse the bbox into border vs interior and recombine them
#
# complexity_calibration:
#   why not simplify to pure recoloring:
#     a simple recolor of the whole object would collapse back toward the old task and
#     would not force hierarchical recognition
#   why not add many prototype families:
#     more than 2 outer groups or many independent inner templates would make the
#     mapping hard to infer from 2-4 examples
#   chosen balance:
#     2 repeated outer groups + 2 repeated inner prototypes keep the swap visible,
#     support repeated evidence within one grid, and avoid ambiguous matching
#
# implicit_assumptions_audit:
#   original_main_dependency_1:
#     picking a color-1 anchor bbox from the whole sprite assumed the whole object was
#     the thing being replaced; that is no longer true because only the interior changes
#   original_main_dependency_2:
#     prototype matching by overlap score assumed a sprite must become the other entire
#     prototype; here that would wrongly overwrite the outer shell
#   original_main_dependency_3:
#     anchor-based offset placement assumed replacement and source might have different
#     occupied extents; in this variant the outer shell is preserved, so bbox alignment
#     is exact and offsets should stay zero
#   original_main_dependency_4:
#     smallest-area sprites are still useful, but now they serve as exemplars for
#     "outer border color + owned inner prototype", not as complete replacement targets
#
# placement_legality:
#   generator uses output-first construction with same-bbox input/output patches
#   because swapping only the interior never changes sprite extents
#   each scaled 5x5 sprite is placed with:
#     border >= 1 from canvas edge
#     bbox gap >= 2 between objects
#   this guarantees no overlap, no clipping, and solver-visible clean components
#
# observability_and_cleanliness:
#   visible cue:
#     outer border colors stay fixed while the contained mini-patterns trade places
#   readability:
#     the solid ring separates each object from the background and frames the change
#   anti-degeneration:
#     disjoint outer colors ensure grouping is visible
#     asymmetric inner patterns with holes ensure the inner swap is not mistaken for
#     a trivial recolor of a symmetric block
#
# color_change_inference:
#   the rule does not depend on numeric color ids or sorting by color value
#   humans can infer it from repeated pairings:
#     every outer-A shell receives the inner layout seen inside outer-B shells, and
#     vice versa
#   colors are limited to:
#     bg + 2 outer colors + shared color 1 + 4 inner variable colors
#
# new_solver_assumptions_to_enforce_in_generator:
#   assumption_1:
#     both base groups appear among the smallest scale-1 sprites, so the solver can
#     recover two distinct base exemplars
#   assumption_2:
#     both base exemplars share the same 5x5 shape with a solid 1-cell border
#   assumption_3:
#     each sprite bbox is an exact integer scale of that base shape
#   assumption_4:
#     the top-left cell of every sprite is part of the preserved outer border, so the
#     solver can identify the group by that visible border color


BASE_SHAPE = (5, 5)
INNER_SLICE = (slice(1, 4), slice(1, 4))

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 8

# --- CFG: colors ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

INNER_LABEL_TEMPLATES = [
    np.array(
        [
            [1, 2, 0],
            [1, 2, 3],
            [0, 0, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 2],
            [3, 1, 2],
            [3, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [2, 1, 0],
            [0, 1, 3],
            [0, 3, 3],
        ],
        dtype=int,
    ),
]


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _make_nested_sprite(bg: int, outer_color: int, inner_arr: np.ndarray) -> np.ndarray:
    arr = np.full(BASE_SHAPE, bg, dtype=int)
    arr[0, :] = outer_color
    arr[-1, :] = outer_color
    arr[:, 0] = outer_color
    arr[:, -1] = outer_color
    arr[INNER_SLICE] = inner_arr
    return arr


def _swap_inner(keep_outer: np.ndarray, take_inner: np.ndarray) -> np.ndarray:
    out = keep_outer.copy()
    out[INNER_SLICE] = take_inner[INNER_SLICE]
    return out


def _infer_scale(arr: np.ndarray, base_shape: tuple[int, int]) -> int:
    bh, bw = base_shape
    ah, aw = arr.shape
    assert ah % bh == 0 and aw % bw == 0, "sprite bbox must be an integer scale of the base shape"
    scale_h = ah // bh
    scale_w = aw // bw
    assert scale_h == scale_w and scale_h >= 1, "sprite scale must be isotropic"
    return int(scale_h)


def _outer_color(arr: np.ndarray) -> int:
    return int(arr[0, 0])


def _build_base_swap_catalog(sprites: list[dict], bg: int) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    areas = [int(s["arr"].shape[0] * s["arr"].shape[1]) for s in sprites]
    min_area = min(areas)
    base = [s for s, area in zip(sprites, areas, strict=False) if area == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct nested base sprites"

    protos = list(uniq.values())
    assert protos[0].shape == protos[1].shape == BASE_SHAPE, "base sprites must be unscaled 5x5 nested exemplars"

    outer_colors = [_outer_color(p) for p in protos]
    assert outer_colors[0] != outer_colors[1], "the two groups must be distinguished by border color"

    swaps = [
        _swap_inner(protos[0], protos[1]),
        _swap_inner(protos[1], protos[0]),
    ]
    return protos, swaps, outer_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the inner micro-sprite between the two outer-shell groups while preserving each shell.

    Input structure:
    - The background is the most common color. Each foreground component is a scaled 5x5 nested sprite.
    - Every sprite has a solid outer border ring (group identity) and a 3x3 asymmetric inner pattern with holes.

    Rule:
    - Recover the two scale-1 nested exemplars from the smallest sprites and use their border colors to define the two groups.
    - For each sprite, keep its outer border color, position, and scale unchanged, but replace its inner 3x3 pattern with the other group's inner pattern.

    Key insight: The object is no longer a flat prototype. Its outer border is the stable frame, and only the contained inner motif is exchanged, so solving the task requires splitting each sprite into preserved outer structure and swapped inner payload.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 nested sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    _protos, swaps, outer_colors = _build_base_swap_catalog(sprites, bg)

    out = g.copy()
    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        scale = _infer_scale(arr, BASE_SHAPE)
        group_idx = next((i for i, col in enumerate(outer_colors) if col == _outer_color(arr)), None)
        assert group_idx is not None, "every sprite border color must match one of the two recovered groups"

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        replacement = _scale_nn(swaps[group_idx], scale)
        paste(out, replacement, r0, c0, transparent=bg)

    return out


def _make_inner_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = random.choice(INNER_LABEL_TEMPLATES)
    pool = [c for c in range(10) if c not in (bg, 1)]
    inner_a_cols = random.sample(pool, 2)
    remaining = [c for c in pool if c not in inner_a_cols]
    inner_b_cols = random.sample(remaining, 2)

    def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
        out = np.full((3, 3), bg, dtype=int)
        out[lab == 1] = 1
        out[lab == 2] = cols[0]
        out[lab == 3] = cols[1]
        return out

    return _colorize(labels, inner_a_cols), _colorize(np.flipud(labels), inner_b_cols)


def _make_base_prototypes(bg: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inner_a, inner_b = _make_inner_prototypes(bg)

    pool = [c for c in range(10) if c not in set([bg, 1] + sorted(set(inner_a.ravel()).union(set(inner_b.ravel()))))]
    outer_colors = random.sample(pool, 2)

    input_protos = [
        _make_nested_sprite(bg, outer_colors[0], inner_a),
        _make_nested_sprite(bg, outer_colors[1], inner_b),
    ]
    output_protos = [
        _make_nested_sprite(bg, outer_colors[0], inner_b),
        _make_nested_sprite(bg, outer_colors[1], inner_a),
    ]
    return input_protos, output_protos


def _instance_plan() -> list[tuple[int, int]]:
    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)
    return instances


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    input_protos, output_protos = _make_base_prototypes(bg)
    instances = _instance_plan()

    max_dim = max(BASE_SHAPE[0] * scale for _group, scale in instances)
    min_n = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES)
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for group_idx, scale in instances:
        input_patch = _scale_nn(input_protos[group_idx], scale)
        output_patch = _scale_nn(output_protos[group_idx], scale)
        marker = np.full(input_patch.shape, 1, dtype=int)

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
        paste(inp, input_patch, r0, c0, transparent=bg)
        paste(out, output_patch, r0, c0, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D4 nested-sprite input")
