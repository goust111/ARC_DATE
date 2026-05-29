# Combo variant of c7f57c3e (index 12)
# combo_spec: D4/2+D14/0
# Name: 嵌套精灵 + 保留原始轮廓痕迹

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# ------------------------------------
# fusion_spec:
#   object model:
#     each foreground component is a scaled 6x6 nested sprite
#     outer layer: 1-cell border ring, one solid group color
#     inner layer: 4x4 multicolor micro-sprite with bg holes and a shared color-1 anchor core
#   grouping:
#     exactly 2 groups, determined only by the outer border color
#     each group owns one inner prototype in the input
#   transform:
#     preserve the outer shell, bbox, and scale
#     swap only the inner 4x4 prototype between the two outer-color groups
#     then repaint a 1-pixel source-colored contour for each semantic layer:
#       1. shell layer contour
#       2. inner micro-sprite contour
#
# behavior_trace:
#   original_logic:
#     whole flat prototype A -> whole flat prototype B, aligned by a color-1 anchor
#   D4/2 alone:
#     [shell A + inner A] -> [shell A + inner B]
#     [shell B + inner B] -> [shell B + inner A]
#   fused_logic here:
#     first do the nested inner swap as above
#     then repaint only the source layer contours:
#       [shell A + inner A]
#         -> [shell A + inner B]
#         -> [shell A contour stays A] + [inner A contour sits on top of inner B]
#     because the shell is preserved, its contour is mostly semantically important;
#     the visibly new residue appears mainly around the swapped inner motif, producing
#     a nested "frame + old inner rim + new inner core" effect
#
# complexity_calibration:
#   why not reuse the 5x5 shell / 3x3 inner layout from the single D4 variant:
#     under D14-style contour repaint, a scale-1 3x3 inner sprite would have almost no
#     non-boundary variable pixels; since the surviving center is color 1 in both groups,
#     the contour would erase nearly all visible evidence of the swap and the combo would
#     degenerate toward "unchanged small sprites"
#   chosen fix:
#     enlarge only enough to a 6x6 shell with a 4x4 inner chamber; this keeps scale-1
#     sprites with visible interior variable pixels after contour repaint while still
#     fitting 3-4 objects on a 28x28 canvas with the required spacing
#   why keep exactly 2 shell groups and 2 inner prototypes:
#     repeated pairings are necessary for humans to infer the swap; more families would
#     create unnecessary correspondence ambiguity in 2-4 training examples
#
# implicit_assumptions_audit:
#   original_main_dependency_1:
#     the whole component used to be the replaceable unit
#     combo status: false; only the inner layer is swapped, so solver must parse shell vs inner
#   original_main_dependency_2:
#     the original residue variant could compute one contour on the whole object mask
#     combo status: false; that would miss the inner-layer boundary because shell and inner
#     are connected, so the fused solver must compute contours per semantic layer
#   original_main_dependency_3:
#     anchor-based offset placement handled prototype bbox differences
#     combo status: unnecessary; every sprite is an exact scaled 6x6 box and the shell is
#     preserved, so replacement stays at the same bbox origin
#   original_main_dependency_4:
#     smallest-area components can still recover the base exemplars
#     combo status: true, but generator must ensure both shell-color groups appear at scale 1
#   new_combo_dependency_1:
#     every sprite bbox must be an integer isotropic scale of the 6x6 base shape
#   new_combo_dependency_2:
#     shell colors must be distinct and readable from visible border cells such as arr[0, 0]
#   new_combo_dependency_3:
#     the inner templates must contain at least one variable-colored non-boundary pixel so
#     the swap remains visible after contour repaint; generator enforces this structurally
#
# placement_legality:
#   input and output patches share the same bbox because the shell is preserved and the
#   contour never extends beyond the source sprite footprint
#   generator therefore uses output-first placement with one reserved rectangle per sprite
#   and places both input and output at the same top-left
#   placement constraints:
#     border >= 1 from canvas edge
#     bbox gap >= 2 between sprites
#   this prevents overlap, clipping, and cross-sprite contour interference
#
# observability_and_cleanliness:
#   visible cue:
#     shell colors stay with their groups, but the content inside each shell changes to the
#     other group's motif while keeping a source-colored inner rim
#   why the rule is inferable:
#     humans can match shells by outer color, see that the inner motif came from the other
#     shell group, and also notice that only the inner edge preserves old colors
#   anti-clutter choice:
#     no extra markers are added outside the sprites; all residue stays on existing layer
#     boundaries, so the grid remains a set of separated nested objects rather than a collage
#
# color_change_inference:
#   no sorting by numeric color ids is used anywhere
#   shell colors define the two groups perceptually
#   residue colors are literal source colors copied back at the same cells
#   inner swap is supported by repeated structure, not by arbitrary color remapping
#
# generator_guarantees_for_main:
#   assumption_1:
#     the smallest-scale sprites include one exemplar from each shell-color group
#   assumption_2:
#     all sprites are exact integer scales of the same 6x6 nested base shape
#   assumption_3:
#     output differs from both the input and the pure D4 nested-swap output on most samples:
#       input difference confirms the inner swap is still visible
#       pure-swap difference confirms the contour residue is genuinely active


BASE_SHAPE = (6, 6)
INNER_SLICE = (slice(1, 5), slice(1, 5))

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

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
MAX_RETRIES = 400

INNER_LABEL_TEMPLATES = [
    np.array(
        [
            [0, 2, 2, 0],
            [3, 2, 1, 2],
            [3, 1, 3, 2],
            [0, 3, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 2, 3, 0],
            [2, 1, 2, 3],
            [3, 2, 3, 1],
            [0, 3, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 2, 0],
            [3, 2, 1, 2],
            [2, 3, 2, 1],
            [0, 3, 3, 0],
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


def _infer_scale(arr: np.ndarray) -> int:
    bh, bw = BASE_SHAPE
    ah, aw = arr.shape
    assert ah % bh == 0 and aw % bw == 0, "sprite bbox must be an integer scale of the base shape"
    scale_h = ah // bh
    scale_w = aw // bw
    assert scale_h == scale_w and scale_h >= 1, "sprite scale must be isotropic"
    return int(scale_h)


def _outer_color(arr: np.ndarray) -> int:
    return int(arr[0, 0])


def _edge_mask(arr: np.ndarray, bg: int) -> np.ndarray:
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


def _layer_trace_mask(arr: np.ndarray, scale: int, bg: int) -> np.ndarray:
    """Contour union of the shell layer and the inner layer, each measured separately."""
    h, w = arr.shape
    assert (h, w) == (BASE_SHAPE[0] * scale, BASE_SHAPE[1] * scale), "unexpected sprite shape"

    shell = np.full_like(arr, bg)
    shell[:scale, :] = arr[:scale, :]
    shell[-scale:, :] = arr[-scale:, :]
    shell[:, :scale] = arr[:, :scale]
    shell[:, -scale:] = arr[:, -scale:]

    inner = np.full_like(arr, bg)
    inner[scale : 5 * scale, scale : 5 * scale] = arr[scale : 5 * scale, scale : 5 * scale]

    return _edge_mask(shell, bg) | _edge_mask(inner, bg)


def _build_base_swap_catalog(sprites: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
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
    assert protos[0].shape == protos[1].shape == BASE_SHAPE, "base sprites must be unscaled 6x6 nested exemplars"

    outer_colors = [_outer_color(p) for p in protos]
    assert outer_colors[0] != outer_colors[1], "the two groups must be distinguished by border color"

    swaps = [
        _swap_inner(protos[0], protos[1]),
        _swap_inner(protos[1], protos[0]),
    ]
    return protos, swaps, outer_colors


def _compose_combo_patch(source_patch: np.ndarray, swapped_patch: np.ndarray, scale: int, bg: int) -> np.ndarray:
    out = swapped_patch.copy()
    trace = _layer_trace_mask(source_patch, scale, bg)
    out[trace] = source_patch[trace]
    return out


def _pure_nested_swap(grid: np.ndarray) -> np.ndarray:
    """Reference D4-only behavior, used by generator-side anti-degeneration checks."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=8)
    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": g[r0 : r1 + 1, c0 : c1 + 1].copy()})

    _protos, swaps, outer_colors = _build_base_swap_catalog(sprites)
    out = g.copy()
    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        scale = _infer_scale(arr)
        group_idx = next((i for i, col in enumerate(outer_colors) if col == _outer_color(arr)), None)
        assert group_idx is not None

        for rr, cc in s["pts"]:
            out[rr, cc] = bg
        replacement = _scale_nn(swaps[group_idx], scale)
        paste(out, replacement, r0, c0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the inner nested motif between the two shell groups and keep source-colored layer rims.

    Input structure:
    - The background is the most common color. Each foreground component is a scaled 6x6 nested sprite.
    - Every sprite has a solid outer border ring that identifies its group and a 4x4 inner micro-sprite with color 1 and bg holes.

    Rule:
    - Recover the two unscaled nested exemplars from the smallest sprites and use their border colors to define the two groups.
    - For each sprite, keep its outer shell, bbox, and scale fixed, but replace the inner 4x4 pattern with the other group's inner pattern.
    - After the inner swap, repaint the 1-pixel contour of the source shell layer and the 1-pixel contour of the source inner layer at their original cells using their original colors.

    Key insight:
    - The fused task cannot be solved by treating the sprite as one flat blob. The solver must split each object into semantic layers, swap only the contained inner motif, and compute residue on each layer separately so the old inner rim survives inside the unchanged shell.
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

    _protos, swaps, outer_colors = _build_base_swap_catalog(sprites)

    out = g.copy()
    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        scale = _infer_scale(arr)
        group_idx = next((i for i, col in enumerate(outer_colors) if col == _outer_color(arr)), None)
        assert group_idx is not None, "every sprite border color must match one of the two recovered groups"

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        swapped_patch = _scale_nn(swaps[group_idx], scale)
        combo_patch = _compose_combo_patch(arr, swapped_patch, scale, bg)
        paste(out, combo_patch, r0, c0, transparent=bg)

    return out


def _make_inner_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = random.choice(INNER_LABEL_TEMPLATES)
    pool = [c for c in range(10) if c not in (bg, 1)]
    inner_a_cols = random.sample(pool, 2)
    remaining = [c for c in pool if c not in inner_a_cols]
    inner_b_cols = random.sample(remaining, 2)

    def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
        out = np.full((4, 4), bg, dtype=int)
        out[lab == 1] = 1
        out[lab == 2] = cols[0]
        out[lab == 3] = cols[1]
        return out

    inner_a = _colorize(labels, inner_a_cols)
    inner_b = _colorize(np.flipud(labels), inner_b_cols)
    return inner_a, inner_b


def _make_base_prototypes(bg: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    inner_a, inner_b = _make_inner_prototypes(bg)

    used = set(map(int, inner_a.ravel())).union(set(map(int, inner_b.ravel())))
    pool = [c for c in range(10) if c not in used and c != bg]
    outer_colors = random.sample(pool, 2)

    input_protos = [
        _make_nested_sprite(bg, outer_colors[0], inner_a),
        _make_nested_sprite(bg, outer_colors[1], inner_b),
    ]
    swapped_protos = [
        _swap_inner(input_protos[0], input_protos[1]),
        _swap_inner(input_protos[1], input_protos[0]),
    ]
    return input_protos, swapped_protos


def _instance_plan() -> list[tuple[int, int]]:
    count = random.choice(SPRITE_COUNTS)
    if count == 4:
        big_scale = 2
    else:
        big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)
    return instances


def _trace_metrics(inp: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    pure = _pure_nested_swap(inp)
    trace_pixels = int(np.sum(out != pure))
    visible_swap_pixels = int(np.sum(out != inp))
    return trace_pixels, visible_swap_pixels


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    input_protos, swapped_protos = _make_base_prototypes(bg)
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
        swapped_patch = _scale_nn(swapped_protos[group_idx], scale)
        output_patch = _compose_combo_patch(input_patch, swapped_patch, scale, bg)
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
        solved = main(inp)
        if not np.array_equal(solved, out):
            return None
        trace_pixels, visible_swap_pixels = _trace_metrics(inp, out)
        if trace_pixels <= 0 or visible_swap_pixels <= 0:
            return None
        return inp
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
