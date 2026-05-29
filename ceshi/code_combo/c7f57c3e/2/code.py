# Combo variant of c7f57c3e (index 2)
# combo_spec: D1/0+D15/0
# Name: 按形状分组 + 图例指示映射
#
# Design analysis
# ---------------
# combo_variant:
#   summary:
#     - Keep the D1 shift: sprites are grouped by repeated shape family, not by raw colored-pattern identity.
#     - Add the D15 shift: a black edge legend explicitly encodes color-pair mappings.
#     - Fuse them so the visible sprite shape decides "which family am I?", while the legend decides
#       "which recoloring scheme belongs to that family?". The output then swaps the legend-decoded
#       target schemes between the two shape families.
#   concrete_behavior_trace:
#     - Suppose shape family A is an L-like 4x4 template and family B is a cross-like 4x4 template.
#     - Input sprites of A all use ordered source pair (orange, teal); input sprites of B use the same
#       two colors but in reversed order (teal, orange). The legend shows:
#           (orange, teal) -> (red, green)
#           (teal, orange) -> (yellow, maroon)
#     - Original task: every A sprite would become the full B prototype and vice versa.
#     - Fused task: shape stays fixed. First recover whether a sprite is family A or B from its
#       canonicalized shape. Then read that family's own legend row to learn its target pair. Finally,
#       swap target pairs across shape families: A receives B's legend target (yellow, maroon), and B
#       receives A's legend target (red, green).
#     - Difference from original: geometry no longer flips to the other prototype. Difference from pure
#       D1: the borrowed palette is not read off another visible sprite; it is decoded from the legend.
#   complexity_calibration:
#     - Exactly two shape families are kept, with 3-4 total sprites and exactly one enlarged sprite, so
#       the repeated grouping cue remains strong and comparable to the base task.
#     - The legend still has exactly two rows, preventing instruction overload.
#     - The two families intentionally reuse the same unordered input color set in opposite order. This
#       makes the legend's ordered pairs meaningful and avoids a degenerate "just use unordered colors"
#       shortcut.
#     - Target pairs use fresh legend-only colors, so the legend is visibly necessary.
#   implicit_assumption_audit:
#     - Original assumption "every component is a sprite" breaks because the black legend is also a
#       component; the solver must isolate and remove it before sprite parsing.
#     - Original assumption "smallest colored arrays are the two prototypes" also breaks because family
#       identity is now shape-only; the solver must canonicalize each sprite by background/anchor/slot
#       roles before deduplicating.
#     - A new assumption appears: row-major ordered variable colors must be stable under scaling. The
#       generator guarantees nearest-neighbor scaling and shape templates with both variable slots present.
#     - Another new assumption: each smallest sprite family must appear at scale 1 and keep a unique
#       canonical template. The generator forces one scale-1 instance of each family.
#   placement_legality:
#     - Legend is pasted first on the top edge and reserved as occupied area.
#     - Because the fused rule preserves each sprite's own bbox, placement can reserve each full bbox
#       directly; no anchor-aligned union box is needed.
#     - Padding >= 2 and border >= 1 keep all components disjoint and solver-safe.
#   visual_observability:
#     - Humans can see two repeated shape families and a separate black instruction panel.
#     - Outputs keep every outline in place, so the salient change is "same shape, new family palette".
#     - Legend-only output colors make it visually obvious that the edge panel is functionally active.
#   visual_cleanliness:
#     - No overlays are drawn on sprites.
#     - The legend is isolated at the edge, and sprite bboxes never touch each other.
#   inferability_check:
#     - Across examples, the stable invariants are: same-shape grouping, legend decoding, and swapping of
#       decoded target schemes between the two shape groups.
#     - The generator avoids cases where the fused output accidentally equals the original prototype-swap
#       output by keeping geometry fixed and using fresh legend-only target palettes.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


BLACK = 0


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _ordered_variable_colors(arr: np.ndarray, bg: int) -> tuple[int, int]:
    seen: list[int] = []
    for value in arr.ravel().tolist():
        v = int(value)
        if v in (bg, 1):
            continue
        if v not in seen:
            seen.append(v)
    assert len(seen) == 2, "each sprite must expose exactly two ordered variable colors"
    return int(seen[0]), int(seen[1])


def _canonicalize(arr: np.ndarray, bg: int) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=int)
    out[arr == 1] = 1
    c0, c1 = _ordered_variable_colors(arr, bg)
    out[arr == c0] = 2
    out[arr == c1] = 3
    return out


def _paint_from_canonical(labels: np.ndarray, bg: int, palette: tuple[int, int]) -> np.ndarray:
    out = np.full(labels.shape, int(bg), dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = int(palette[0])
    out[labels == 3] = int(palette[1])
    return out


def _extract_legend(
    g: np.ndarray, bg: int
) -> tuple[tuple[int, int, int, int], dict[tuple[int, int], tuple[int, int]]]:
    comps = components_mask(g != bg, connectivity=8)
    legend_boxes = []
    for pts in comps:
        vals = g[pts[:, 0], pts[:, 1]]
        if np.any(vals == BLACK):
            legend_boxes.append(bbox_coords(pts))
    assert len(legend_boxes) == 1, "expected exactly one legend component"

    r0, c0, r1, c1 = legend_boxes[0]
    box = g[r0 : r1 + 1, c0 : c1 + 1]
    legend_map: dict[tuple[int, int], tuple[int, int]] = {}
    for rr in range(box.shape[0]):
        vals = [int(v) for v in box[rr].tolist() if int(v) != BLACK]
        if not vals:
            continue
        assert len(vals) == 4, "each legend row must contain exactly four swatches"
        src = (vals[0], vals[1])
        dst = (vals[2], vals[3])
        assert src not in legend_map, "legend source rows must be unique ordered pairs"
        legend_map[src] = dst
    assert len(legend_map) == 2, "legend must expose exactly two ordered mappings"
    return legend_boxes[0], legend_map


def _sprite_area(sprite: dict[str, object]) -> int:
    r0, c0, r1, c1 = sprite["bbox"]
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep every sprite's own shape, but swap the legend-decoded color schemes between the two shape families.

    Input structure:
    - The most common color is the background. A compact black legend box touches the top edge.
    - Away from the legend, 3-4 sprite instances appear at integer scales. There are exactly two repeated
      shape families; each sprite uses color 1 plus two ordered variable colors.

    Rule:
    - Read the legend as two ordered source-pair -> target-pair mappings, then ignore the legend area itself.
    - Recover the two sprite families from the smallest scale-1 examples by canonicalizing each sprite into
      background / anchor / first-variable-slot / second-variable-slot labels.
    - For every sprite, identify which canonical shape family it belongs to, look up that family's own legend
      target pair, and repaint the sprite with the other family's target pair while preserving its shape, scale,
      bbox, and occupied cells.

    Key insight: The shape tells you which repeated family a sprite belongs to, but the replacement colors do not
    come from any visible sprite. They come from the legend, and then those legend-decoded color schemes are
    exchanged between the two shape families.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legend_bbox, legend_map = _extract_legend(g, bg)
    body = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    body[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

    comps = components_mask(body != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least two sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = body[r0 : r1 + 1, c0 : c1 + 1].copy()
        ordered = _ordered_variable_colors(arr, bg)
        assert ordered in legend_map, "sprite ordered colors must match one legend row"
        sprites.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "arr": arr,
                "ordered": ordered,
                "canon": _canonicalize(arr, bg),
            }
        )

    areas = [_sprite_area(s) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]

    families_by_key: dict[tuple[int, ...], dict[str, object]] = {}
    for s in base:
        key = tuple(np.asarray(s["canon"]).ravel().tolist())
        if key not in families_by_key:
            ordered = s["ordered"]
            families_by_key[key] = {
                "canon": np.asarray(s["canon"]).copy(),
                "src_pair": ordered,
                "dst_pair": legend_map[ordered],
            }
    assert len(families_by_key) == 2, "smallest sprites must reveal exactly two shape families"

    families = list(families_by_key.values())
    base_h, base_w = families[0]["canon"].shape
    assert all(f["canon"].shape == (base_h, base_w) for f in families), "families must share one base bbox size"

    out = g.copy()

    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        canon_arr = np.asarray(s["canon"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h % base_h == 0 and w % base_w == 0, "sprite bbox must be an integer scale of the family template"
        scale_h = h // base_h
        scale_w = w // base_w
        assert scale_h == scale_w, "sprite scaling must be isotropic"
        scale = int(scale_h)

        best_idx = None
        best_score = -1
        for idx, fam in enumerate(families):
            scaled = _scale_nn(np.asarray(fam["canon"]), scale)
            if scaled.shape != canon_arr.shape:
                continue
            score = int(np.sum(scaled == canon_arr))
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None

        repl_palette = families[1 - best_idx]["dst_pair"]
        replacement = _paint_from_canonical(canon_arr, bg, repl_palette)

        for rr, cc in s["pts"]:
            out[rr, cc] = bg
        paste(out, replacement, r0, c0, transparent=bg)

    return out


# --------------- generator (combo D1/0 + D15/0) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 7

# --- CFG: palettes ---
BG_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9]

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: legend ---
LEGEND_SHAPE = (5, 9)
LEGEND_SOURCE_COLS = (1, 2)
LEGEND_TARGET_COLS = (6, 7)
LEGEND_ROWS = (1, 3)

# --- CFG: generation retries ---
MAX_RETRIES = 400

# 4x4 canonical family templates. 0 = bg hole, 1 = anchor, 2/3 = ordered variable slots.
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


def _pick_shape_pair() -> tuple[np.ndarray, np.ndarray]:
    idx_a, idx_b = random.sample(range(len(SHAPE_LIBRARY)), 2)
    return SHAPE_LIBRARY[idx_a].copy(), SHAPE_LIBRARY[idx_b].copy()


def _sample_pairs(bg: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    pool = [c for c in range(2, 10) if c != bg]
    src_colors = random.sample(pool, 2)
    src_a = (int(src_colors[0]), int(src_colors[1]))
    src_b = (int(src_colors[1]), int(src_colors[0]))

    remaining = [c for c in pool if c not in src_colors]
    fresh = random.sample(remaining, 4)
    dst_a = (int(fresh[0]), int(fresh[1]))
    dst_b = (int(fresh[2]), int(fresh[3]))
    return src_a, src_b, dst_a, dst_b


def _build_legend(mappings: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    legend = np.full(LEGEND_SHAPE, BLACK, dtype=int)
    rows = list(LEGEND_ROWS)
    random.shuffle(rows)
    shuffled = mappings[:]
    random.shuffle(shuffled)
    for rr, (src, dst) in zip(rows, shuffled, strict=False):
        legend[rr, LEGEND_SOURCE_COLS[0]] = int(src[0])
        legend[rr, LEGEND_SOURCE_COLS[1]] = int(src[1])
        legend[rr, LEGEND_TARGET_COLS[0]] = int(dst[0])
        legend[rr, LEGEND_TARGET_COLS[1]] = int(dst[1])
    return legend


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    shape_a, shape_b = _pick_shape_pair()
    src_a, src_b, dst_a, dst_b = _sample_pairs(bg)
    legend = _build_legend([(src_a, dst_a), (src_b, dst_b)])

    in_protos = [
        _paint_from_canonical(shape_a, bg, src_a),
        _paint_from_canonical(shape_b, bg, src_b),
    ]
    out_protos = [
        _paint_from_canonical(shape_a, bg, dst_b),
        _paint_from_canonical(shape_b, bg, dst_a),
    ]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(4 * scale for _shape_idx, scale in instances)
    min_n = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES, LEGEND_SHAPE[1] + 2)
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    legend_c = random.randint(0, n - LEGEND_SHAPE[1])
    paste(inp, legend, 0, legend_c)
    paste(out, legend, 0, legend_c)
    paste(reserve, legend, 0, legend_c)

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
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
