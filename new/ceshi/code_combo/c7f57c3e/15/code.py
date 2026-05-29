# Combo variant of c7f57c3e (index 15)
# combo_spec: D2/3+D14/0
# Name: 镜像对称配对 + 保留原始轮廓

"""
Design analysis for the combo variant.

- Concrete behavior and trace:
  The grid contains exactly two mirror families, each with two disconnected multicolor
  sprites. Within one family, the two sprites have the same role-labeled structure up
  to one mirror transform, but they use different two-color palettes and may use
  different scales. For one sprite, the base task would have tried to classify it as
  one of two global prototypes and replace its whole shape. This combo instead does:
  recover the sprite's scale from the color-1 anchor block -> compress to an unscaled
  label grid -> normalize the two non-anchor roles by region size -> group it with its
  mirror mate -> recolor its interior to the mate's palette -> repaint the source
  sprite's own 1-pixel contour in the source colors. The visible difference from both
  the base task and plain D2 is that the output keeps the same geometry while showing a
  partner-colored interior wrapped by a source-colored rim.
- Complexity calibration:
  The scene is fixed to two mirror families / four sprites so every example repeats the
  pairing rule twice. One family is forced to be cross-scale (2 vs 3) so mirror pairing
  cannot collapse into a same-size heuristic. All sprites use scale >= 2: this is
  deliberate because a scale-1 thin sprite would let the contour overwrite the whole
  recolor and hide the fusion effect. Within each family the two palettes are disjoint,
  which makes both the interior swap and the preserved contour immediately visible.
- Hidden-assumption audit:
  The original solver assumed "two smallest sprites reveal the two global prototypes"
  and "replacement target is determined by prototype class"; both assumptions are false
  here. The new solver instead assumes:
  1. every sprite contains exactly one color-1 anchor cell, so its scaled anchor bbox
     is a square of side == scale;
  2. the two variable roles have different areas in base-template space, so they can be
     normalized without using color ids;
  3. each mirror family appears exactly twice, and the paired templates are related by
     one supported mirror kind;
  4. the contour must be computed on the observed scaled sprite, not in template space,
     otherwise the retained rim would become scale-thick.
  The generator constructs and verifies all four properties.
- Placement legality:
  Input and output use the same bbox for every sprite because this combo only recolors
  plus overlays a source contour. Generator placement therefore only needs non-overlap
  for the observed input patches. It reserves each sprite bbox with border >= 1 and
  padding >= 1, so components stay disconnected and contour repainting cannot collide
  with another sprite.
- Visual observability:
  The mirror relation is visible from shape alone because the template pool is
  asymmetric and excludes self-mirrors for the chosen transform. The preserved contour
  is also visible because it keeps the literal source colors on the edge while the
  interior switches to the partner palette. Seeing two such families in one grid gives
  enough support for a human to infer both the pairing rule and the residue rule.
- Visual cleanliness and color inference:
  There are no extra markers outside the sprites. The only extra layer is a 1-pixel rim
  inside the existing sprite footprint, so the output stays sparse and readable. Color
  transfer is inferable from structure: role areas determine which partner color goes to
  which region, while the contour copies source colors at the same edge cells.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


MIRROR_KINDS = ("h", "v", "d", "ad")
PAIR_COUNT = 2


def _anti_diag(arr: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(arr)).T


def _mirror(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "id":
        return arr.copy()
    if kind == "h":
        return np.flipud(arr)
    if kind == "v":
        return np.fliplr(arr)
    if kind == "d":
        return arr.T
    if kind == "ad":
        return _anti_diag(arr)
    raise ValueError(f"unknown mirror kind: {kind}")


def _mirror_variants(arr: np.ndarray) -> list[np.ndarray]:
    return [arr.copy(), np.flipud(arr), np.fliplr(arr), arr.T, _anti_diag(arr)]


def _all_symmetries(arr: np.ndarray) -> list[np.ndarray]:
    return [
        arr.copy(),
        np.rot90(arr, 1),
        np.rot90(arr, 2),
        np.rot90(arr, 3),
        np.flipud(arr),
        np.fliplr(arr),
        arr.T,
        _anti_diag(arr),
    ]


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _pick_anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(arr == 1)
    assert len(ys) > 0, "sprite must contain anchor color 1"
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _infer_scale(arr: np.ndarray) -> int:
    r0, c0, r1, c1 = _pick_anchor_bbox(arr)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w, "scaled anchor must remain a square block"
    return int(h)


def _compress_scaled(arr: np.ndarray, scale: int) -> np.ndarray:
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "bbox must be divisible by scale"
    out = np.zeros((h // scale, w // scale), dtype=int)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = arr[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            vals = np.unique(block)
            assert len(vals) == 1, "sprite is not a nearest-neighbor scaled template"
            out[r, c] = int(vals[0])
    return out


def _normalize_roles(arr: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    colors = [int(c) for c in np.unique(arr) if c not in (bg, 1)]
    assert len(colors) == 2, "each sprite must contain exactly two non-anchor colors"
    counts = sorted((int(np.sum(arr == c)), c) for c in colors)
    assert counts[0][0] != counts[1][0], "role areas must be distinct"
    role2_color = counts[0][1]
    role3_color = counts[1][1]

    norm = np.zeros(arr.shape, dtype=int)
    norm[arr == 1] = 1
    norm[arr == role2_color] = 2
    norm[arr == role3_color] = 3
    return norm, (role2_color, role3_color)


def _family_key(norm: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    candidates = []
    for variant in _all_symmetries(norm):
        candidates.append((tuple(map(int, variant.shape)), tuple(int(x) for x in variant.ravel())))
    return min(candidates)


def _extract_sprite_record(grid: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    scale = _infer_scale(arr)
    base = _compress_scaled(arr, scale)
    norm, role_colors = _normalize_roles(base, bg)
    return {
        "pts": pts,
        "bbox": (r0, c0, r1, c1),
        "arr": arr,
        "scale": scale,
        "base": base,
        "norm": norm,
        "role_colors": role_colors,
        "key": _family_key(norm),
    }


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


def _recolor_sprite(arr: np.ndarray, src_colors: tuple[int, int], dst_colors: tuple[int, int]) -> np.ndarray:
    out = arr.copy()
    out[arr == src_colors[0]] = dst_colors[0]
    out[arr == src_colors[1]] = dst_colors[1]
    return out


def _compose_combo_patch(
    source_patch: np.ndarray, bg: int, src_colors: tuple[int, int], dst_colors: tuple[int, int]
) -> np.ndarray:
    out = _recolor_sprite(source_patch, src_colors, dst_colors)
    edge = _edge_mask(source_patch, bg)
    out[edge] = source_patch[edge]
    return out


def _detect_pair_transform(a: np.ndarray, b: np.ndarray) -> str:
    for kind in MIRROR_KINDS:
        if np.array_equal(_mirror(a, kind), b):
            return kind
    raise AssertionError("paired sprites are not related by a supported mirror")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each sprite's interior palette with its mirror mate while preserving a source-colored rim.

    Input structure:
    - The background is the most frequent color. Foreground pixels form exactly four isolated sprites.
    - Every sprite is a nearest-neighbor scaled copy of one asymmetric label template, with one color-1 anchor block and exactly two other colors.
    - The four sprites partition into two mirror families: inside each family, the unscaled label patterns differ by one horizontal, vertical, or diagonal reflection, and the two members may have different scales.

    Rule:
    - Recover each sprite's integer scale from its anchor block, compress it back to an unscaled label pattern, and normalize the two variable regions by their visible areas.
    - Group sprites by mirror-family equivalence. Within each matched pair, recolor each sprite's two variable regions using the partner sprite's colors, while keeping its own geometry, scale, anchor, and position.
    - After recoloring, repaint the observed 1-pixel contour of the source sprite in its original colors so the old outline remains visible.

    Key insight:
    - Mirror pairing has to be done on a scale-free, color-normalized representation; once the pair is known, the transformation is local to the original sprite footprint. The final output is "partner colors inside, self colors on the rim", not a geometric replacement.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) == 2 * PAIR_COUNT, "grid must contain exactly four sprites"

    sprites = [_extract_sprite_record(g, pts, bg) for pts in comps]

    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = {}
    for sprite in sprites:
        groups.setdefault(sprite["key"], []).append(sprite)

    assert len(groups) == PAIR_COUNT, "the grid must contain exactly two mirror families"

    out = g.copy()
    for family in groups.values():
        assert len(family) == 2, "each mirror family must contain exactly two sprites"
        left, right = family
        _detect_pair_transform(left["norm"], right["norm"])

        for sprite, partner in ((left, right), (right, left)):
            patch = _compose_combo_patch(sprite["arr"], bg, sprite["role_colors"], partner["role_colors"])
            r0, c0, _r1, _c1 = sprite["bbox"]
            paste(out, patch, r0, c0, transparent=bg)

    return out


# --------------- generator ---------------

GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
SMALL_SCALE = 2
BIG_SCALE = 3

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
PLACEMENT_CONNECTIVITY = 8

MAX_RETRIES = 500

PROTO_TEMPLATES = (
    np.array(
        [
            [0, 1, 0, 0],
            [2, 2, 3, 0],
            [0, 3, 3, 0],
            [0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 2, 0, 3, 0],
            [0, 2, 3, 3, 0],
            [0, 0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0],
            [0, 2, 2, 0],
            [3, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 3, 3, 0],
            [0, 0, 0, 3, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 3, 3, 0],
            [0, 0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 2, 3, 0],
            [0, 0, 3, 3],
            [0, 0, 0, 3],
        ],
        dtype=int,
    ),
)


def _template_pool() -> tuple[np.ndarray, ...]:
    seen: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for template in PROTO_TEMPLATES:
        seen.setdefault(_family_key(template), template)
    return tuple(seen.values())


TEMPLATE_POOL = _template_pool()


def _colorize(labels: np.ndarray, palette: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(labels.shape, bg, dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = palette[0]
    out[labels == 3] = palette[1]
    return out


def _valid_mirror_kinds(labels: np.ndarray) -> list[str]:
    return [kind for kind in MIRROR_KINDS if not np.array_equal(_mirror(labels, kind), labels)]


def _changed_inner_pixels(source_patch: np.ndarray, output_patch: np.ndarray, bg: int) -> int:
    edge = _edge_mask(source_patch, bg)
    inner = (source_patch != bg) & ~edge
    return int(np.sum(output_patch[inner] != source_patch[inner]))


def _pair_specs(bg: int) -> list[dict]:
    templates = random.sample(list(TEMPLATE_POOL), PAIR_COUNT)
    big_pair_idx = random.randrange(PAIR_COUNT)
    pool = [c for c in range(10) if c not in (bg, 1)]

    specs = []
    for idx, template in enumerate(templates):
        colors = random.sample(pool, 4)
        pal_a = (colors[0], colors[1])
        pal_b = (colors[2], colors[3])
        if random.random() < 0.5:
            pal_a = (pal_a[1], pal_a[0])
        if random.random() < 0.5:
            pal_b = (pal_b[1], pal_b[0])

        mirror_kind = random.choice(_valid_mirror_kinds(template))
        labels_a = template.copy()
        labels_b = _mirror(template, mirror_kind)
        if random.random() < 0.5:
            labels_a, labels_b = labels_b, labels_a
            pal_a, pal_b = pal_b, pal_a

        scales = [SMALL_SCALE, SMALL_SCALE]
        if idx == big_pair_idx:
            scales[random.randrange(2)] = BIG_SCALE

        specs.append(
            {
                "labels": (labels_a, labels_b),
                "palettes": (pal_a, pal_b),
                "scales": tuple(scales),
            }
        )
    return specs


def _build_sprite_records(bg: int) -> list[dict]:
    records = []
    for pair_idx, spec in enumerate(_pair_specs(bg)):
        labels_a, labels_b = spec["labels"]
        pal_a, pal_b = spec["palettes"]
        scale_a, scale_b = spec["scales"]

        in_a = _scale_nn(_colorize(labels_a, pal_a, bg), scale_a)
        in_b = _scale_nn(_colorize(labels_b, pal_b, bg), scale_b)
        out_a = _compose_combo_patch(in_a, bg, pal_a, pal_b)
        out_b = _compose_combo_patch(in_b, bg, pal_b, pal_a)

        assert _changed_inner_pixels(in_a, out_a, bg) > 0, "combo effect vanished on sprite A"
        assert _changed_inner_pixels(in_b, out_b, bg) > 0, "combo effect vanished on sprite B"

        records.extend(
            [
                {"input_patch": in_a, "output_patch": out_a, "pair": pair_idx, "size_key": max(in_a.shape)},
                {"input_patch": in_b, "output_patch": out_b, "pair": pair_idx, "size_key": max(in_b.shape)},
            ]
        )
    return records


def _place_sprites(records: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_dim = max(max(rec["input_patch"].shape) for rec in records)
    min_n = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES)
    if min_n > GRID_SIZE_MAX:
        return None
    n = GRID_SIZE_MAX

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    order = sorted(range(len(records)), key=lambda idx: records[idx]["size_key"], reverse=True)
    for idx in order:
        rec = records[idx]
        patch = rec["input_patch"]
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
        paste(inp, rec["input_patch"], r0, c0, transparent=bg)
        paste(out, rec["output_patch"], r0, c0, transparent=bg)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    records = _build_sprite_records(bg)
    placed = _place_sprites(records, bg)
    if placed is None:
        return None
    inp, expected = placed

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
