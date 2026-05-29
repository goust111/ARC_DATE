# Variant of c7f57c3e
# Dimension: D1 — 分组匹配条件 (match_cond)
# Change: 按颜色模式（如 teal/green 组 vs red/green 组）将精灵分为两组，同组精灵共享相同颜色组合... → 按形状分组而非颜色模式
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     original:
#       The two smallest prototypes are treated as the two group identities themselves, so a sprite is
#       classified by its exact colored pattern and then replaced by the opposite full prototype.
#     variant:
#       Two groups are instead defined by silhouette+internal-region layout. Each group owns one ordered
#       two-color palette in the input. The output keeps every sprite's silhouette, bbox, scale and anchor
#       pixels fixed, but swaps the ordered palette between the two silhouette groups.
#   concrete_pixels:
#     - Every sprite is a 4x4 label template scaled by 1..3 with bg holes, a color-1 anchor spine,
#       and two non-anchor regions.
#     - Two different templates appear in each grid; they are the visible "shape groups".
#     - Input: all copies of shape A use palette A, all copies of shape B use palette B.
#     - Output: shape A is recolored with palette B, shape B is recolored with palette A.
#   behavior_trace:
#     example:
#       - Suppose the smallest sprites are an L-like template colored (orange, teal) and a T-like template
#         colored (red, green).
#       - Original logic would map every L-like sprite to the full T-like prototype, changing the outline.
#       - Variant logic only borrows the T group's ordered palette (red, green) and paints it onto the
#         L-like template, so the outline stays L-like while colors change.
#       - This is the crucial D1 shift: grouping comes from repeated shape class, not repeated color class.
#   complexity_calibration:
#     - Keep exactly two shape families so the rule stays inferable from 2-4 examples.
#     - Keep exactly two ordered variable-color slots per family; this is enough structure to make the
#       palette swap visible without exploding the hypothesis space.
#     - Reuse the original task's 3-4 instances and one enlarged sprite so each family repeats and the
#       shape cue remains visible at multiple scales.
#   implicit_assumption_audit:
#     original_dependencies_to_remove:
#       - Exact colored-array identity can no longer be used as the grouping key.
#       - Replacing a sprite with the opposite full prototype would incorrectly change the silhouette.
#     new_solver_assumptions:
#       - The two scale-1 sprites share the same 4x4 bbox size and expose exactly two distinct canonical
#         shape templates.
#       - Each template uses color 1 plus exactly two ordered non-bg colors.
#       - Nearest-neighbor scaling preserves row-major first appearance of the two variable regions, so a
#         canonical relabeling by first-seen color is stable across scales.
#     generator_guarantees:
#       - Both shape families appear at scale 1, optionally with one duplicate small sprite.
#       - Both templates are 8-connected, distinct after removing palette values, and use the same bbox size.
#       - Placement reserves full bboxes, leaving clean separation and preventing accidental component merges.
#   placement_legality:
#     - Because input/output keep the same bbox for each sprite, generator placement is simpler than the
#       original union-bbox placement and cannot create anchor-alignment drift.
#     - Full-bbox reservation with padding >= 2 keeps sprites visually isolated and solver-safe.
#   visual_observability:
#     - The human-visible invariant is the outline family: every L-like object changes to the T group's
#       colors while remaining L-like, and vice versa.
#     - Shared-color probability is retained so examples are not trivially separable by "all colors change";
#       the shape cue still has to drive the grouping.
#   neatness:
#     - No overlays, no touching sprites, and only two shape families per grid.
#     - The output is a clean recoloring of the same connected components, making the rule easy to compare.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _ordered_variable_colors(arr: np.ndarray, bg: int) -> list[int]:
    seen: list[int] = []
    for value in arr.ravel().tolist():
        v = int(value)
        if v in (bg, 1):
            continue
        if v not in seen:
            seen.append(v)
    assert len(seen) == 2, "each sprite must expose exactly two ordered variable colors"
    return seen


def _canonicalize(arr: np.ndarray, bg: int) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=int)
    out[arr == 1] = 1
    ordered = _ordered_variable_colors(arr, bg)
    out[arr == ordered[0]] = 2
    out[arr == ordered[1]] = 3
    return out


def _paint_from_canonical(labels: np.ndarray, bg: int, palette: list[int]) -> np.ndarray:
    out = np.full(labels.shape, bg, dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = int(palette[0])
    out[labels == 3] = int(palette[1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the ordered two-color palette between the two shape families while preserving every sprite's outline.

    Input structure:
    - bg = most common color. Two distinct small silhouette templates appear at various scales.
    - Each sprite is a nearest-neighbor scaled copy of one template, with color 1 as a fixed anchor region and two variable-color regions.

    Rule:
    - Use the smallest sprites to recover the two canonical shape templates and the ordered two-color palette attached to each shape family.
    - For every sprite, identify its shape by ignoring the concrete palette and matching the canonicalized template at the correct scale.
    - Repaint that same canonical template with the other shape family's ordered palette, keeping bbox, scale, and occupied cells unchanged.

    Key insight: The repeated cue is the silhouette, not the literal colors; once the two shape families are recognized, each family simply borrows the other family's palette while staying in exactly the same place.
    """
    g = np.asarray(grid, dtype=int)
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
    base = [s for s, a in zip(sprites, areas) if a == min_area]

    uniq: dict[tuple[int, ...], dict[str, object]] = {}
    for s in base:
        canon = _canonicalize(s["arr"], bg)
        key = tuple(canon.ravel().tolist())
        if key not in uniq:
            uniq[key] = {
                "canon": canon,
                "palette": _ordered_variable_colors(s["arr"], bg),
            }
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct shape templates"

    families = list(uniq.values())
    base_h, base_w = families[0]["canon"].shape

    out = g.copy()

    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        arr = s["arr"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h % base_h == 0 and w % base_w == 0, "sprite bbox must be an integer scale of the base template"
        scale_h = h // base_h
        scale_w = w // base_w
        assert scale_h == scale_w, "scaling must be isotropic"
        scale = int(scale_h)

        canon_arr = _canonicalize(arr, bg)
        best_idx = None
        best_score = -1
        for idx, fam in enumerate(families):
            scaled = _scale_nn(fam["canon"], scale)
            if scaled.shape != canon_arr.shape:
                continue
            score = int(np.sum(scaled == canon_arr))
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None

        replacement = _paint_from_canonical(canon_arr, bg, families[1 - best_idx]["palette"])

        for rr, cc in s["pts"]:
            out[rr, cc] = bg
        paste(out, replacement, r0, c0, transparent=bg)

    return out


# --------------- generator (D1 shape-match variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

# 4x4 label templates. 0 = bg hole, 1 = anchor, 2/3 = ordered variable regions.
SHAPE_LIBRARY = [
    np.array(
        [
            [1, 0, 0, 2],
            [1, 2, 2, 2],
            [1, 0, 3, 0],
            [1, 3, 3, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [2, 2, 1, 2],
            [0, 2, 1, 0],
            [3, 3, 1, 3],
            [0, 3, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [2, 1, 2, 0],
            [2, 1, 2, 3],
            [0, 1, 2, 3],
            [3, 1, 3, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 2, 1, 3],
            [2, 2, 1, 0],
            [0, 3, 1, 3],
            [2, 3, 1, 0],
        ],
        dtype=int,
    ),
]


def _colorize(labels: np.ndarray, bg: int, palette: list[int]) -> np.ndarray:
    return _paint_from_canonical(labels, bg, palette)


def _pick_palettes(bg: int) -> tuple[list[int], list[int]]:
    pool = [c for c in range(10) if c not in (bg, 1)]
    pal_a = random.sample(pool, 2)
    remaining = [c for c in pool if c not in pal_a]
    if random.random() < COLOR_SHARE_PROB:
        shared = random.choice(pal_a)
        fresh = random.choice(remaining)
        pal_b = [shared, fresh] if random.random() < 0.5 else [fresh, shared]
    else:
        pal_b = random.sample(remaining, 2)
    assert pal_a != pal_b
    return pal_a, pal_b


def _pick_shape_pair() -> tuple[np.ndarray, np.ndarray]:
    idxs = random.sample(range(len(SHAPE_LIBRARY)), 2)
    return SHAPE_LIBRARY[idxs[0]].copy(), SHAPE_LIBRARY[idxs[1]].copy()


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    shape_a, shape_b = _pick_shape_pair()
    pal_a, pal_b = _pick_palettes(bg)

    in_protos = [
        _colorize(shape_a, bg, pal_a),
        _colorize(shape_b, bg, pal_b),
    ]
    out_protos = [
        _colorize(shape_a, bg, pal_b),
        _colorize(shape_b, bg, pal_a),
    ]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(4 * scale for _shape_idx, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for shape_idx, scale in instances:
        input_patch = _scale_nn(in_protos[shape_idx], scale)
        output_patch = _scale_nn(out_protos[shape_idx], scale)
        marker = np.full(input_patch.shape, 1, dtype=int)
        try:
            top, left = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, marker, top, left, transparent=bg)
        paste(inp, input_patch, top, left, transparent=bg)
        paste(out, output_patch, top, left, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D1 input")
