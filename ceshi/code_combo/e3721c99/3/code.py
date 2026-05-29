# Combo variant of e3721c99 (index 3)
# combo_spec: D1/4+D2/0
# Name: 旋转翻转匹配+多色精灵

"""
Design analysis for the combo variant
=====================================

variant_spec:
  preserved_backbone:
    grid: 30x30, bg=0
    divider: still a full row / full column / L-shaped separator touching the border
    legend_panel: still occupies the smaller side behind the divider
    workspace: still holds 8-11 separated source sprites
    output: matched workspace sprites recolor to legend colors; unmatched ones erase
  fused_changes:
    D1/4:
      old_match: compare source hole count with legend hole count
      new_match: compare exact silhouette after searching all dihedral transforms
    D2/0:
      old_source_detection: identify all source objects by fixed grey color 5
      new_source_detection: identify source objects by being on the workspace side of the divider;
        workspace sprites are single-color but use multiple colors across the sample
  supporting_generator_shift:
    reason:
      - If legends kept the original symmetric hole-bars, rotation/flip matching would be weak or
        ambiguous.
      - If workspace sprites stayed grey, the D2 change would be invisible.
    fix:
      - Legends and sprites now come from a small library of asymmetric zero-hole silhouettes that
        are unique up to rotation/reflection.
      - Workspace sprites are painted from a small overlapping source palette, with at least three
        distinct input colors per sample.

a) Concrete behavior and trace
  - Pixel layout:
      a border-touching divider splits the grid into a small legend panel and a larger workspace.
      The panel contains 2-4 colored asymmetric legend shapes. The workspace contains scattered
      source sprites with mixed input colors, plus distractor shapes from the same library.
  - Example trace:
      original task:
        grey blob with 2 holes -> compare hole count -> recolor to the legend whose bar shows 2 holes
      fused task:
        panel has red skewed-F and green bent-L
        workspace sprite A is yellow and is a flipped skewed-F -> recolor to red
        workspace sprite B is teal and is a shape absent from the panel -> erase
    The changed pipeline is:
      1. old: find `g == 5` components directly
         new: find the unique border-touching divider, split the grid into panel/workspace regions,
         and treat only workspace components as sources
      2. old: compute hole counts
         new: normalize each component by canonical dihedral shape key
      3. old: hole_count -> color
         new: canonical_shape -> color

b) Complexity calibration
  - The silhouette library is limited to 10 compact shapes, so humans can compare repeated motifs
    across 2-4 examples without the task turning into arbitrary visual search.
  - Each sample still uses only 2-4 legend entries and 8-11 total workspace sprites, preserving the
    original workload.
  - All library shapes share hole count 0, which removes the old topological shortcut entirely and
    makes the rotation/flip cue necessary instead of optional.
  - Every legend contributes at least one forced non-identical orientation match, and workspace input
    colors span at least three values, so both dimensions stay visually active.

c) Implicit-assumption audit of the original main()
  - `for pts in components_mask(g == grey)` no longer works: source sprites are no longer defined by
    one color, so role detection must move from color-based to region-based.
  - `_legend_mapping(..., grey)` assumed all non-grey foreground belonged to the legend. That breaks
    because workspace sprites can reuse legend colors. The new mapping only reads components on the
    panel side of the divider.
  - The original `hole_count -> color` map assumed legend keys were topological and unique. In the
    combo, all legend shapes deliberately share the same hole count, so the key must become the
    canonical binary silhouette under rotation/reflection.
  - Recolor writes are only safe if legend components are excluded from the source loop. The new
    solver therefore checks each component's side label before recoloring.

d) Placement legality
  - The generator keeps the divider as the only border-touching foreground component.
  - Legend shapes are inset from the grid edge and divider by at least one background cell.
  - Workspace sprites are placed only inside the larger side, never on the border, and are separated
    by a one-cell moat enforced through 8-neighborhood dilation of the occupied mask.
  - Output-first validation remains in place: generated inputs are accepted only when
    `main(input) == expected`.

e) Visual observability
  - Since every active shape has zero holes, the old legend-bar interpretation cannot explain the
    recoloring. The solver must infer geometry rather than topology.
  - Because source sprites use multiple colors and overlap the legend palette, the viewer cannot use
    "grey means source" or "this color only appears in legends" as a shortcut.
  - Forced orientation changes make it visible that matching is by rotated/flipped silhouette rather
    than literal pose equality.

f) Visual cleanliness
  - Legends stay grouped in a small dedicated panel, keeping the matching examples easy to scan.
  - Source sprites are compact (mostly 3x3 to 4x4) and never touch each other, so the workspace
    remains legible despite the extra color variety.
  - Divider, legend panel, and workspace continue to occupy distinct spatial roles, preventing the
    fusion from looking like an unstructured collage.

g) Change inferability
  - Output colors are still anchored by visible legend exemplars, not by hidden numeric properties.
  - Repeated matches per legend provide support for the color mapping, while distractor shapes show
    that not every workspace object survives.
  - Input colors of the sources have no fixed relationship to output colors, but that ambiguity is
    constrained: there are only 2-4 target colors and a 3-5 color source palette, so the human can
    infer "shape after dihedral normalization + workspace role" from a few samples.

New solver assumptions introduced by the combo
  - There is exactly one border-touching divider component, and removing it splits the board into
    exactly two regions.
  - Legend silhouettes are unique up to rotation/reflection.
  - Workspace matches are drawn from the same silhouette library as the legends, while distractors are
    drawn from unused library entries.
  - The generator explicitly guarantees all of these assumptions.
"""

import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, paste


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


def _touches_border(pts: np.ndarray, h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _divider_and_regions(g: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Find the unique border-touching divider and split the board into panel/workspace regions.

    Returns:
      divider_mask, side_labels, panel_label, workspace_label
    """
    h, w = g.shape
    comps = components_mask(g != bg, connectivity=4)
    border_comps = [pts for pts in comps if _touches_border(pts, h, w)]
    assert len(border_comps) == 1, "grid must contain exactly one border-touching divider component"

    divider_mask = np.zeros_like(g, dtype=bool)
    for r, c in border_comps[0]:
        divider_mask[int(r), int(c)] = True

    side_labels, n_side = label_mask(~divider_mask, connectivity=4)
    assert n_side == 2, "divider must split the grid into exactly two regions"

    sizes = {lab: int(np.sum(side_labels == lab)) for lab in range(1, n_side + 1)}
    workspace_label = max(sizes, key=sizes.get)
    panel_label = min(sizes, key=sizes.get)
    return divider_mask, side_labels, int(panel_label), int(workspace_label)


def _legend_mapping(
    g: np.ndarray, bg: int, divider_mask: np.ndarray, side_labels: np.ndarray, panel_label: int
) -> dict[tuple[int, int, bytes], int]:
    mapping: dict[tuple[int, int, bytes], int] = {}
    non_divider_fg = (g != bg) & ~divider_mask
    for pts in components_mask(non_divider_fg, connectivity=4):
        r, c = pts[0]
        if int(side_labels[int(r), int(c)]) != panel_label:
            continue
        key = _canonical_shape_key(_component_mask(pts))
        assert key not in mapping, "legend shapes must be unique up to rotation/reflection"
        mapping[key] = int(g[int(r), int(c)])
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor multicolored workspace sprites by matching rotated or mirrored silhouettes to the legend.

    Input structure:
    - bg is the most common color, and one border-touching divider isolates a smaller legend region
      from a larger workspace region.
    - The legend region contains 2-4 colored asymmetric shapes, each representing one target color.
    - The workspace contains separated source sprites with multiple input colors; these colors overlap
      the general palette and are not sufficient to identify which objects should change.

    Rule:
    - Identify the divider, split the board into legend-side and workspace-side, and build a mapping
      from each legend's canonical dihedral silhouette to its color.
    - For every non-divider component in the workspace, extract its tight binary mask and normalize it
      across all rotations and flips.
    - Recolor the whole sprite to the matching legend color; erase workspace distractors whose shape is
      absent from the legend even after those spatial transforms.

    Key insight:
    - Two cues must be combined: spatial partition tells you which objects are sources, and dihedral
      shape equivalence tells you which legend color each source should become.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    divider_mask, side_labels, panel_label, workspace_label = _divider_and_regions(g, bg)
    legend = _legend_mapping(g, bg, divider_mask, side_labels, panel_label)
    assert legend, "legend mapping must be non-empty: legend-side must contain colored shapes"

    out = g.copy()
    non_divider_fg = (g != bg) & ~divider_mask
    for pts in components_mask(non_divider_fg, connectivity=4):
        r, c = pts[0]
        if int(side_labels[int(r), int(c)]) != workspace_label:
            continue
        key = _canonical_shape_key(_component_mask(pts))
        newc = int(legend[key]) if key in legend else bg
        for rr, cc in pts:
            out[int(rr), int(cc)] = newc

    return out


BG = 0
DIV = 1
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SOURCE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
PANEL_SIZE_RANGE = (7, 9)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2

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
assert all(len(variants) > 1 for variants in _SHAPE_VARIANTS)


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
            variants = _SHAPE_VARIANTS[int(shape_id)]
            display_index = random.randrange(len(variants))
            legends.append(
                {
                    "shape_id": int(shape_id),
                    "shape_key": _SHAPE_KEYS[int(shape_id)],
                    "display_index": int(display_index),
                    "color": int(color),
                    "patch": _render_shape(variants[int(display_index)], int(color)),
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


def _choose_source_palette() -> list[int]:
    size = random.randint(3, min(5, len(SOURCE_COLORS)))
    return random.sample(SOURCE_COLORS, size)


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

    base_items: list[dict] = []
    for i, item in enumerate(legends):
        base_items.append(
            {
                "shape_id": int(item["shape_id"]),
                "target_color": int(item["color"]),
                "avoid_display_index": int(item["display_index"]),
            }
        )
        for _ in range(extras[i]):
            base_items.append(
                {
                    "shape_id": int(item["shape_id"]),
                    "target_color": int(item["color"]),
                    "avoid_display_index": None,
                }
            )

    used_shape_ids = {int(item["shape_id"]) for item in legends}
    distractor_ids = [i for i in range(len(_BASE_SHAPES)) if i not in used_shape_ids]
    for _ in range(n_distractors):
        base_items.append(
            {
                "shape_id": int(random.choice(distractor_ids)),
                "target_color": None,
                "avoid_display_index": None,
            }
        )
    random.shuffle(base_items)

    palette = _choose_source_palette()
    need_distinct = min(3, len(base_items), len(palette))
    for _ in range(20):
        plan: list[dict] = []
        used_colors: list[int] = []
        reserved = palette.copy()
        random.shuffle(reserved)
        ok = True
        for i, item in enumerate(base_items):
            target_color = item["target_color"]
            choices = [c for c in palette if target_color is None or c != target_color]
            assert choices, "source palette must provide a color distinct from the target"
            if i < need_distinct:
                distinct_choices = [c for c in reserved if c in choices]
                if not distinct_choices:
                    ok = False
                    break
                source_color = distinct_choices[0]
                reserved.remove(source_color)
            else:
                source_color = random.choice(choices)
            plan.append({**item, "source_color": int(source_color)})
            used_colors.append(int(source_color))
        if ok and len(set(used_colors)) >= 3:
            return plan
    raise AssertionError("source sprites must use at least three colors")


def _make_sprite(shape_id: int, avoid_display_index: int | None) -> np.ndarray:
    variants = _SHAPE_VARIANTS[int(shape_id)]
    choices = list(range(len(variants)))
    if avoid_display_index is not None and len(choices) > 1:
        choices = [i for i in choices if i != int(avoid_display_index)]
    vidx = random.choice(choices)
    return variants[int(vidx)]


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite_mask: np.ndarray
) -> tuple[int, int] | None:
    h, w = sprite_mask.shape
    occupied_pad = _dilate_mask(occupied)
    n = GRID_SIZE
    candidates = [(r, c) for r in range(n - h + 1) for c in range(n - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if r == 0 or c == 0 or r + h == n or c + w == n:
            continue
        rr0, cc0 = max(0, r - 1), max(0, c - 1)
        rr1, cc1 = min(n, r + h + 1), min(n, c + w + 1)
        if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _paint_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _place_sprites(grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[dict]) -> bool:
    items = []
    for spec in _sprite_plan(legends):
        sprite_mask = _make_sprite(spec["shape_id"], spec["avoid_display_index"])
        items.append((int(sprite_mask.size), sprite_mask, spec["target_color"], spec["source_color"]))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid, dtype=bool)
    for _, sprite_mask, target_color, source_color in items:
        loc = _random_sprite_location(blocked, occupied, sprite_mask)
        if loc is None:
            return False
        r, c = loc
        in_patch = _paint_sprite(sprite_mask, int(source_color))
        paste(grid, in_patch, r, c, transparent=BG)
        occupied[r : r + sprite_mask.shape[0], c : c + sprite_mask.shape[1]] |= sprite_mask
        if target_color is not None:
            out_patch = _paint_sprite(sprite_mask, int(target_color))
            paste(expected, out_patch, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid != BG)):
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
    raise RuntimeError("failed to generate e3721c99 combo variant input")
