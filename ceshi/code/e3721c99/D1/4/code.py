# Variant of e3721c99
# Dimension: D1 — 匹配条件：洞数匹配 vs 其他匹配逻辑 (match_cond)
# Change: s.hole_count == d.hole_count — 灰色精灵的洞数与图例形状的洞数一致时匹配... → 旋转/翻转后形状匹配

"""
Design analysis for the D1 variant
==================================

variant_spec:
  preserved_backbone:
    grid: 30x30, bg=0
    divider: still a full row / full column / L-shaped separator
    legend_panel: still parked near an edge or corner behind the divider
    workspace: still contains scattered grey sprites
    output: matched grey sprites recolor to the legend color; distractors erase to bg
  changed_core:
    old_match: compare grey-sprite hole count against the legend component's hole count
    new_match: compare the sprite's exact binary silhouette against the legend silhouette
      after searching all 8 dihedral transforms (4 rotations x optional reflection)
  supporting_adjustment:
    reason: hole-count matching becomes invisible if legends stay as the old symmetric
      "bar with holes" motifs, because many rotations/flips look equivalent
    fix: legends and sprites now come from a small library of asymmetric polyomino-like
      shapes, all with the same hole count (0), so geometry is the only usable cue

a) Concrete behavior and trace
  - In the input, each legend color owns one small asymmetric colored component inside the
    panel. Grey workspace sprites reuse those same silhouettes, but often in a different
    orientation; extra grey sprites use library shapes that are not present in the panel.
  - Example trace:
      original rule:
        legend red = 0 holes, legend green = 1 hole
        grey blob with 1 hole -> turns green
      new rule:
        legend red = skewed F-shape, legend green = bent L-shape
        grey sprite is a flipped F-shape -> turns red
        grey sprite is a different 0-hole shape -> erased
    The pipeline change happens exactly at the legend lookup step:
      old solver computed `hole_count(submask)`;
      new solver computes a canonical dihedral shape key.

b) Complexity calibration
  - The shape library is small and reused across samples, so the human can compare repeated
    silhouettes instead of facing arbitrary noise.
  - Each sample uses only 2-4 legend shapes and 8-11 total sprites, preserving the original
    workload.
  - Every shape in the library is asymmetric and pairwise non-equivalent under dihedral
    transforms, preventing ambiguous matches.

c) Implicit-assumption audit of the original main()
  - `_legend_mapping(...)` originally assumed "one hole count -> one color". That becomes
    invalid because multiple legends intentionally share hole count 0. The replacement map is
    "canonical transformed silhouette -> color".
  - The grey-sprite loop originally assumed topology alone identifies the target color. That
    no longer holds, so each component is normalized to a tight binary mask and matched by
    shape equivalence under rotation/reflection.
  - The old generator's organic blobs relied on hole seeds and validation by hole count.
    Those writes are now irrelevant and would actively hide the D1 change, so sprite
    generation switches to explicit template placement while keeping the same placement rules.

d) Placement legality
  - Legend components are still entirely inside the legend panel, with a one-cell inset from
    edges and divider.
  - Grey sprites are still placed only in the workspace, with one-cell separation enforced by
    dilating the occupied mask before trying another location.
  - The generator keeps output-first validation: it builds the expected output and asserts
    `main(input) == expected`.

e) Visual observability
  - All selected legend shapes have hole count 0, so the old cue is unusable.
  - For every legend, at least one matching grey sprite is forced to use a different displayed
    orientation than the legend itself. This makes mental rotation or reflection visibly
    necessary in every generated task.

f) Visual cleanliness
  - Legends remain grouped in a clean strip/corner panel.
  - Workspace sprites never overlap or touch, and the templates are compact (at most 5x5),
    so silhouettes stay legible.

g) Color-change inferability
  - Color is still tied to the nearby legend component, not to any hidden numeric rule.
  - The number of active colors stays small (2-4 legend colors), and repeated matches per
    legend supply enough evidence for the mapping.

New solver assumptions introduced by the variant
  - Every legend shape must be unique up to rotation/reflection.
  - Every grey match sprite must be drawn from the same template library as its target legend.
  - The generator guarantees both by sampling distinct legend templates and constructing
    matches directly from those templates before placement.
"""

import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _mask(rows: list[str]) -> np.ndarray:
    arr = np.array([[ch == "X" for ch in row] for row in rows], dtype=bool)
    pts = np.argwhere(arr)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _shape_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    arr = mask.astype(np.uint8)
    return int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            key = _shape_signature(cand)
            if key in seen:
                continue
            seen.add(key)
            variants.append(cand.copy())
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_shape_signature(v) for v in _dihedral_variants(mask))


def _component_mask(pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[int(r - r0), int(c - c0)] = True
    return sub


# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
PANEL_SIZE_RANGE = (7, 9)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2

# -- generation --
GEN_RETRIES = 300
_STRUCT8 = generate_binary_structure(2, 2)


_BASE_SHAPES = [
    _mask([
        ".XX",
        "XX.",
        ".X.",
    ]),
    _mask([
        "X.",
        "X.",
        "XX",
        "X.",
    ]),
    _mask([
        ".XX",
        "XX.",
        "X..",
    ]),
    _mask([
        "XX",
        "XX",
        "X.",
    ]),
    _mask([
        "XX.",
        "X..",
        "XXX",
    ]),
    _mask([
        "XXX",
        ".X.",
        "XX.",
    ]),
    _mask([
        "X..",
        "XXX",
        ".XX",
    ]),
    _mask([
        ".XX",
        "XX.",
        ".X.",
        ".X.",
    ]),
    _mask([
        ".XX",
        "XXX",
        "..X",
    ]),
    _mask([
        "X.X",
        "XXX",
        ".X.",
    ]),
]
_SHAPE_VARIANTS = [_dihedral_variants(mask) for mask in _BASE_SHAPES]
_SHAPE_KEYS = [_canonical_shape_key(mask) for mask in _BASE_SHAPES]
assert len(set(_SHAPE_KEYS)) == len(_SHAPE_KEYS)
assert all(len(vs) > 1 for vs in _SHAPE_VARIANTS)


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[tuple[int, int, bytes], int]:
    """
    Pick one legend component per non-(bg,grey) color:
      - component bbox is at least 2x2,
      - component does not touch the outer grid border (filters dividers).
    Map canonical dihedral shape -> legend color.
    """
    h, w = g.shape
    mapping: dict[tuple[int, int, bytes], int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = components_mask(g == c, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append(pts)
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {c} must have exactly one legend component"
        key = _canonical_shape_key(_component_mask(candidates[0]))
        assert key not in mapping, "legend shapes must be unique up to rotation/reflection"
        mapping[key] = c
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by matching them to a rotated or mirrored legend shape.

    Input structure:
    - bg = most common color; grey (5) sprites are scattered through the workspace.
    - Colored legend components (non-bg, non-grey, at least 2x2 bbox, not touching the outer border)
      sit inside a panel behind the divider, and each color owns one asymmetric silhouette.

    Rule:
    - Each legend component defines a mapping: canonical shape under rotation/reflection -> color.
    - For every grey sprite, extract its tight binary mask and compare it against all rotated/flipped
      versions of the legend silhouettes.
    - Recolor the grey sprite to the matching legend color; erase grey distractors whose shape does not
      match any legend under those spatial transforms.

    Key insight: The deciding feature is exact geometric shape after dihedral normalization, not topology.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        key = _canonical_shape_key(_component_mask(pts))
        newc = int(legend[key]) if key in legend else bg
        for r, c in pts:
            out[int(r), int(c)] = newc

    return out


def _render_shape(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _legend_line_span(legends: list[dict], arrangement: str) -> int:
    if arrangement == "horizontal":
        return sum(int(item["patch"].shape[1]) for item in legends) + len(legends) - 1
    return sum(int(item["patch"].shape[0]) for item in legends) + len(legends) - 1


def _legend_cross_dim(legends: list[dict], arrangement: str) -> int:
    if arrangement == "horizontal":
        return max(int(item["patch"].shape[0]) for item in legends)
    return max(int(item["patch"].shape[1]) for item in legends)


def _choose_legends(arrangement: str, max_span: int) -> list[dict]:
    for _ in range(200):
        count = random.randint(*LEGEND_COUNT_RANGE)
        shape_ids = random.sample(range(len(_BASE_SHAPES)), count)
        colors = random.sample(LEGEND_COLORS, count)
        legends = []
        for shape_id, color in zip(shape_ids, colors):
            variants = _SHAPE_VARIANTS[shape_id]
            display_index = random.randrange(len(variants))
            mask = variants[display_index]
            legends.append(
                {
                    "shape_id": int(shape_id),
                    "shape_key": _SHAPE_KEYS[shape_id],
                    "display_index": int(display_index),
                    "color": int(color),
                    "patch": _render_shape(mask, int(color)),
                }
            )
        if _legend_line_span(legends, arrangement) <= max_span:
            return legends
    raise RuntimeError("failed to choose legend set that fits")


def _place_legend_line(grid: np.ndarray, legends: list[dict], arrangement: str, r: int, c: int) -> None:
    rr, cc = int(r), int(c)
    for item in legends:
        patch = item["patch"]
        paste(grid, patch, rr, cc, transparent=BG)
        if arrangement == "horizontal":
            cc += int(patch.shape[1]) + 1
        else:
            rr += int(patch.shape[0]) + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _legend_line_span(legends, "horizontal")
    cross = _legend_cross_dim(legends, "horizontal")
    panel_h = random.randint(max(PANEL_SIZE_RANGE[0], cross + 2), PANEL_SIZE_RANGE[1])
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - cross - 1)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - cross - 1)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _legend_line_span(legends, "vertical")
    cross = _legend_cross_dim(legends, "vertical")
    panel_w = random.randint(max(PANEL_SIZE_RANGE[0], cross + 2), PANEL_SIZE_RANGE[1])
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - cross - 1)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - cross - 1)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, line_h: int, line_w: int
) -> tuple[int, int, int, int, int, int]:
    n = GRID_SIZE
    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - line_h - 1
        c_min, c_max = 1, panel_w - line_w - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - line_h - 1
        c_min, c_max = dc + 2, n - 1 - line_w
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - 1 - line_h
        c_min, c_max = 1, panel_w - line_w - 1
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - line_h
        c_min, c_max = dc + 2, n - 1 - line_w
    return int(dr), int(dc), int(r_min), int(r_max), int(c_min), int(c_max)


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _legend_line_span(legends, arrangement)
    cross = _legend_cross_dim(legends, arrangement)
    cross_panel = random.randint(max(PANEL_SIZE_RANGE[0], cross + 2), PANEL_SIZE_RANGE[1])
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    if arrangement == "horizontal":
        panel_h, panel_w = cross_panel, along
        line_h, line_w = cross, total
    else:
        panel_h, panel_w = along, cross_panel
        line_h, line_w = total, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, line_h, line_w
    )

    if corner == "tl":
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "tr":
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_plan(legends: list[dict]) -> list[dict]:
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, item in enumerate(legends):
        plan.append(
            {
                "shape_id": int(item["shape_id"]),
                "color": int(item["color"]),
                "avoid_display_index": int(item["display_index"]),
            }
        )
        for _ in range(extras[i]):
            plan.append(
                {
                    "shape_id": int(item["shape_id"]),
                    "color": int(item["color"]),
                    "avoid_display_index": None,
                }
            )

    distractor_ids = [i for i in range(len(_BASE_SHAPES)) if i not in {int(item["shape_id"]) for item in legends}]
    for _ in range(n_distractors):
        plan.append(
            {
                "shape_id": int(random.choice(distractor_ids)),
                "color": None,
                "avoid_display_index": None,
            }
        )
    random.shuffle(plan)
    return plan


def _make_sprite(shape_id: int, avoid_display_index: int | None) -> np.ndarray:
    variants = _SHAPE_VARIANTS[int(shape_id)]
    choices = list(range(len(variants)))
    if avoid_display_index is not None and len(choices) > 1:
        choices = [i for i in choices if i != int(avoid_display_index)]
    vidx = random.choice(choices)
    return _render_shape(variants[vidx], GREY)


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    n = GRID_SIZE
    candidates = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(n, r + h + 1), min(n, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[dict]) -> bool:
    items = []
    for spec in _sprite_plan(legends):
        sprite = _make_sprite(spec["shape_id"], spec["avoid_display_index"])
        items.append((int(sprite.size), sprite, spec["color"]))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, color in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if color is not None:
            out_patch = np.where(sprite == GREY, int(color), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    n = GRID_SIZE
    grid = np.zeros((n, n), dtype=int)
    expected = np.zeros((n, n), dtype=int)
    blocked = np.zeros((n, n), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 D1 variant input")
