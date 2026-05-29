# Combo variant of e3721c99 (index 25)
# combo_spec: D1/4+D4/1+D8/2
# Name: 旋转匹配+洞内填色+凹陷计数

"""
Design analysis for the combo variant
=====================================

variant_spec:
  preserved_backbone:
    grid: still 30x30 with bg=0
    divider: still a full row or full column separator
    legend_panel: still placed behind the divider near an edge or corner
    workspace: still contains scattered grey components
  fused_changes:
    old_match: compare enclosed-hole count of a grey sprite against the legend
    new_match: compare open-concavity count instead; orientation may differ, so the
      count must be read after mentally rotating / flipping the displayed shape
    old_transform: recolor the full grey body
    new_transform: keep the grey body and fill only the sprite's enclosed holes
      with the matched legend color
    key_separation: the feature used for matching (open concavities) is different
      from the feature that receives the color (closed holes)

a) Concrete behavior and trace
  - The colored legend is no longer the old "bar with enclosed holes". Instead each
    legend entry is a compact comb-like block with 1-4 deep open notches and zero
    enclosed holes. The notch count is the legend key.
  - A matching grey sprite reuses the same notch count but also contains one or two
    true enclosed holes elsewhere in its body. The sprite can appear in a different
    orientation from the legend.
  - Example trace:
      original rule:
        legend red has 1 enclosed hole
        grey sprite with 1 enclosed hole -> whole body turns red
      combo rule:
        legend red has 2 open concavities and 0 enclosed holes
        grey sprite has 2 open concavities, 1 enclosed hole, and is rotated leftward
        -> sprite stays grey, its 1 enclosed hole becomes red
    The decisive difference is that the legend's and sprite's enclosed-hole counts
    can disagree completely while the open-concavity counts agree.

b) Complexity calibration
  - Concavity values are limited to 1..3 and each sample uses exactly 2 legend entries.
    That keeps the mapping learnable while still allowing distractors with unused
    concavity counts.
  - Every matched sprite has at least one enclosed hole, so the D4-style fill is
    always visible.
  - The legend and sprite families are regular enough for humans to count notches by
    eye, but they are not identical to the original hole bars, so the variant does
    not collapse back into hole-count matching.

c) Implicit-assumption audit of the original main()
  - Original `_legend_mapping(...)` assumed "one enclosed-hole count -> one color".
    That fails here because legend entries intentionally have zero enclosed holes.
    The replacement map is "one concavity count -> one color".
  - Original grey-sprite processing assumed the matched feature and the transformed
    pixels were the same object body. That is no longer true: matching inspects open
    background intrusions, while coloring targets closed background regions.
  - Original generator assumed organic hole blobs were enough. Under the combo rule
    that would hide the concavity signal, so generation switches to explicit templates
    whose concavity count and hole count are both controlled.

d) Placement legality
  - Legend patches stay fully inside the legend panel with a one-cell inset from the
    divider and from the outer border.
  - Grey sprites stay in the workspace, remain 4-connected, and keep a one-cell gap
    from each other.
  - Sprite bounding boxes are also kept disjoint so a filled hole can never overlap a
    different sprite nested inside it.

e) Visual observability
  - The legend's visible cue is the number of deep open notches, not the number of
    closed cavities.
  - Every legend gets at least one matched sprite whose displayed orientation differs
    from the legend's panel orientation, so the sample set visibly requires reading
    the same concavity count across rotations.
  - Because matched sprites carry 1 or 2 enclosed holes, the output visibly says
    "keep shell, color interior" rather than "recolor body".

f) Visual cleanliness
  - The templates are regular comb/block silhouettes rather than noisy blobs, so the
    notch count and the enclosed holes remain easy to parse.
  - The output in any local area has only background, a grey shell, and one fill
    color; no extra markers or overlays are introduced.

g) Color-change inferability
  - Only 2 legend colors are active per sample.
  - Each color is supported by repeated matches, and the copied color appears exactly
    inside holes of sprites whose open-notch count matches the corresponding legend.

New solver assumptions introduced by the combo variant
  - Every legend color corresponds to exactly one concavity count.
  - Grey matches must have at least one enclosed hole if the color fill is to be
    visible.
  - Concavity counting is performed on the tight component mask and is intended to be
    invariant under rotation/reflection.
  - `generate_input()` guarantees all three by constructing legends from no-hole
    concavity templates, matched sprites from hole-bearing templates with the same
    concavity count, and validating every sample with `main(input) == expected`.
"""

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "X" for ch in row] for row in rows], dtype=bool)


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


def _component_mask(pts: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[int(r - r0), int(c - c0)] = True
    return sub


def _hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return enclosed background components inside a boolean mask's bbox."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            pts = []
            while st:
                r, c = st.pop()
                pts.append((int(r), int(c)))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes.append(pts)
    return holes


def _concavity_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Count open concavities as exterior-connected background intrusions whose interior
    portion reaches depth >= 2 inside the component's tight bounding box.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    exterior_cells: list[tuple[int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            pts = []
            while st:
                r, c = st.pop()
                pts.append((int(r), int(c)))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if touches:
                exterior_cells.extend(pts)

    inner = [(r, c) for r, c in exterior_cells if 0 < r < h - 1 and 0 < c < w - 1]
    inner_set = set(inner)
    seen = set()
    regions: list[list[tuple[int, int]]] = []
    for p in inner:
        if p in seen:
            continue
        st = [p]
        seen.add(p)
        comp = []
        while st:
            r, c = st.pop()
            comp.append((int(r), int(c)))
            for rr, cc in neighbors4(r, c):
                if (rr, cc) in inner_set and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    st.append((rr, cc))
        depth = max(min(r, c, h - 1 - r, w - 1 - c) for r, c in comp)
        if depth >= 2:
            regions.append(comp)
    return regions


def _concavity_count(mask: np.ndarray) -> int:
    return len(_concavity_regions(mask))


def _orientation_invariant_concavity_count(mask: np.ndarray) -> int:
    counts = {_concavity_count(v) for v in _dihedral_variants(mask)}
    assert len(counts) == 1, "concavity count should be invariant under dihedral transforms"
    return next(iter(counts))


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one legend component per non-(bg,grey) color:
      - component bbox is at least 2x2,
      - component does not touch the outer grid border (filters dividers).
    Map concavity_count -> legend_color.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
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
        cc = _orientation_invariant_concavity_count(_component_mask(candidates[0]))
        assert cc not in mapping, f"legend concavity count {cc} must map to exactly one color"
        mapping[cc] = int(c)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match grey sprites to legend colors by open-concavity count, then fill only the
    matched sprites' enclosed holes.

    Input structure:
    - bg is the most common color; grey (5) components occupy the workspace.
    - Colored legend components live inside a divider-separated panel. Each legend is a
      rotated or flipped no-hole shape whose visible clue is its number of deep open
      concavities, not its enclosed-hole count.

    Rule:
    - Build a legend map from concavity count -> color, where concavity count is read
      from the component's tight mask and treated as invariant under rotation/reflection.
    - For every grey sprite, compute its concavity count for matching and separately
      collect its enclosed holes.
    - If the sprite's concavity count appears in the legend, keep the grey shell and
      paint every enclosed hole with the matched legend color. Otherwise erase the grey
      sprite to background.

    Key insight: The matching cue is an open geometric feature (deep concavities) while
    the transformed target is a different topological feature (closed holes).
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[int(r - r0), int(c - c0)] = True
        concavity_count = _orientation_invariant_concavity_count(sub)
        holes = _hole_regions(sub)
        if concavity_count in legend:
            fillc = int(legend[concavity_count])
            for hole in holes:
                for dr, dc in hole:
                    out[r0 + dr, c0 + dc] = fillc
        else:
            for r, c in pts:
                out[int(r), int(c)] = bg
    return out


# --------------- generator (template-based concavity/hole fusion) ---------------

# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 2)
CONCAVITY_VALUES = [1, 2, 3]
PANEL_SIZE_RANGE = (8, 10)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (5, 5)
MAX_DISTRACTORS = 2
MAX_EXTRA_PER_LEGEND = 1

# -- generation --
GEN_RETRIES = 400
_STRUCT8 = generate_binary_structure(2, 2)

_BASE_WIDTH = {1: 5, 2: 7, 3: 10, 4: 13}
_SLOT_STARTS = {
    1: [1],
    2: [1, 4],
    3: [1, 4, 7],
    4: [1, 4, 7, 10],
}


def _comb_shape(concavity_count: int, height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    for start in _SLOT_STARTS[int(concavity_count)]:
        mask[:4, start : start + 2] = False
    return mask


def _legend_base(concavity_count: int) -> np.ndarray:
    return _comb_shape(int(concavity_count), height=7, width=_BASE_WIDTH[int(concavity_count)])


def _sprite_base(concavity_count: int, hole_count: int) -> np.ndarray:
    q = int(concavity_count)
    hole_count = int(hole_count)
    width = _BASE_WIDTH[q] if hole_count == 1 else _BASE_WIDTH[q] + 2
    mask = _comb_shape(q, height=7, width=width)
    if hole_count == 1:
        mask[5, width // 2] = False
    else:
        mask[5, 2] = False
        mask[5, width - 3] = False
    return mask


def _rotation_variant(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, int(k)).copy()


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
    count_choices = list(range(LEGEND_COUNT_RANGE[0], LEGEND_COUNT_RANGE[1] + 1))
    random.shuffle(count_choices)
    for count in count_choices:
        combos = []
        for counts in itertools.combinations(CONCAVITY_VALUES, count):
            if 1 not in counts:
                continue
            line_span = sum(
                _rotation_variant(_legend_base(q), 0 if arrangement == "horizontal" else 1).shape[
                    1 if arrangement == "horizontal" else 0
                ]
                for q in counts
            ) + count - 1
            if line_span <= max_span:
                combos.append(tuple(int(x) for x in counts))
        if not combos:
            continue
        selected = list(random.choice(combos))
        random.shuffle(selected)
        colors = random.sample(LEGEND_COLORS, len(selected))
        legends = []
        rot_choices = [0, 2] if arrangement == "horizontal" else [1, 3]
        for concavity_count, color in zip(selected, colors):
            rot = random.choice(rot_choices)
            patch = _render_shape(_rotation_variant(_legend_base(concavity_count), rot), int(color))
            legends.append(
                {
                    "concavity_count": int(concavity_count),
                    "color": int(color),
                    "rot": int(rot),
                    "patch": patch,
                }
            )
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
    kind = random.choice(["horizontal", "vertical"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    return _layout_full_vertical(grid, blocked)


def _sprite_plan(legends: list[dict]) -> list[dict]:
    candidate_totals = []
    for total in range(TOTAL_SPRITES_RANGE[0], TOTAL_SPRITES_RANGE[1] + 1):
        remaining = total - len(legends)
        max_d = min(MAX_DISTRACTORS, remaining)
        valid_d = [
            d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
        ]
        if valid_d:
            candidate_totals.append((total, valid_d))
    assert candidate_totals, "sprite count range must allow a feasible match/distractor split"
    total, valid_d = random.choice(candidate_totals)
    remaining = total - len(legends)
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, item in enumerate(legends):
        opposite_rotations = [1, 3] if item["rot"] in (0, 2) else [0, 2]
        plan.append(
            {
                "concavity_count": int(item["concavity_count"]),
                "color": int(item["color"]),
                "required_rotations": opposite_rotations,
            }
        )
        for _ in range(extras[i]):
            plan.append(
                {
                    "concavity_count": int(item["concavity_count"]),
                    "color": int(item["color"]),
                    "required_rotations": None,
                }
            )

    distractor_counts = [q for q in CONCAVITY_VALUES if q not in {int(item["concavity_count"]) for item in legends}]
    for _ in range(n_distractors):
        plan.append(
            {
                "concavity_count": int(random.choice(distractor_counts)),
                "color": None,
                "required_rotations": None,
            }
        )
    random.shuffle(plan)
    return plan


def _make_sprite(concavity_count: int, required_rotations: list[int] | None) -> np.ndarray:
    allowed_holes = [1, 2] if int(concavity_count) == 1 else [1]
    hole_count = random.choice(allowed_holes)
    base = _sprite_base(int(concavity_count), hole_count)
    rotations = list(required_rotations) if required_rotations else [0, 1, 2, 3]
    random.shuffle(rotations)
    rot = rotations[0]
    return _render_shape(_rotation_variant(base, rot), GREY)


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


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[dict]
) -> bool:
    items = []
    for spec in _sprite_plan(legends):
        sprite = _make_sprite(spec["concavity_count"], spec["required_rotations"])
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
            paste(expected, sprite, r, c, transparent=BG)
            for hole in _hole_regions(sprite == GREY):
                for dr, dc in hole:
                    expected[r + dr, c + dc] = int(color)
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
    raise RuntimeError("failed to generate e3721c99 combo input")
