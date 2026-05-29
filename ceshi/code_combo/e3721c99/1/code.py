# Combo variant of e3721c99 (index 1)
# combo_spec: D1/1+D8/1
# Name: 面积匹配+面积加权洞计数
#
# Design analysis for the fusion:
# - Concretization / behavior trace:
#   The original task maps `raw hole count -> legend color`. This variant keeps the
#   divider + legend-panel staging, but changes the mapping to `colored-pixel area -> legend color`.
#   Area means the number of foreground pixels in the component itself. Hole cells stay bg and do
#   not contribute. Example: a 3x3 ring has bbox area 9 but pixel area 8; here it matches an
#   8-pixel legend, not a 9-pixel solid block. Weighted hole counting is still present as a
#   secondary structural dimension: 1-cell pinholes are visually present but count as 0 effective
#   holes, whereas 2-cell cavities count as 1. The generator therefore creates sprites where
#   raw-hole perception, weighted-hole perception, and pixel area can disagree.
# - Complexity calibration:
#   Legend shapes still come from a compact 3-thick strip family so humans can count their pixels.
#   Their areas are distinct (9, 8, 13, 18, 23, 28 for hole-values 0..5), avoiding ambiguous
#   many-to-one matches. Grey sprites are varied by hole profile, but the number of area classes
#   remains small (2-4 legends), so the rule is visible rather than combinatorially noisy.
# - Implicit assumptions audit:
#   Original `_legend_mapping` assumed "one non-border component per legend color" and used raw
#   hole count as the key. In this variant, that assumption still holds, but the key changes to
#   component pixel area. Any logic that implicitly treated holes as part of area would fail on
#   sprites like a 3x3 ring: the solver must count only `True` pixels in the component mask.
#   The generator explicitly guarantees unique legend areas and at least one sprite whose area-match
#   output differs from the legacy hole-match output.
# - Placement legality:
#   Legend placement reuses the original divider/panel layouts. Sprite placement keeps all grey
#   shapes inside the workspace, separated from legends/dividers and from one another by at least
#   one bg cell via 8-neighborhood padding checks.
# - Visual observability:
#   Samples include counterexamples to "match by number of visible holes": matched sprites often
#   have a different raw hole count from the legend patch they match by area, and some matched
#   sprites contain holes so bbox area and pixel area visibly differ.
# - Visual cleanliness:
#   The panel stays isolated behind a divider, legend colors stay unique, and sprites are spaced
#   apart. Holey sprites are compact and connected, so the relevant cues remain legible.
# - Inference support:
#   The generator rejects degenerate instances where the new area-based output would coincide with
#   the legacy hole-count output. It also enforces the presence of pinholes and larger cavities
#   across matched sprites so the "weighted hole" dimension is actually instantiated in the input.

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


def _hole_component_sizes(mask: np.ndarray) -> list[int]:
    """
    Return enclosed hole sizes in a binary component mask using 4-connectivity.

    A hole is a bg component inside the mask's tight bbox that does not touch the bbox border.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            size = 0
            while st:
                r, c = st.pop()
                size += 1
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                sizes.append(size)
    sizes.sort()
    return sizes


def _hole_counts(mask: np.ndarray) -> tuple[int, int]:
    """Return (raw_holes, weighted_holes) where only holes of area >= 2 count in the weighted total."""
    sizes = _hole_component_sizes(mask)
    return len(sizes), sum(1 for s in sizes if s >= 2)


def _component_mask(pts: np.ndarray) -> np.ndarray:
    """Tight boolean mask for one component represented as an (N,2) coord array."""
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _legend_mapping_by_area(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one legend component per non-(bg, grey) color and map its pixel area to that color.
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
        area = int(len(candidates[0]))
        assert area not in mapping, f"legend area {area} must map to exactly one color"
        mapping[area] = int(c)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by matching their true pixel area to a color legend.

    Input structure:
    - bg is the dominant background color; grey (5) sprites are scattered in the workspace.
    - Near one edge or corner, a divider isolates a legend panel containing colored prototype shapes.
      Each legend color appears once as a compact non-border component.

    Rule:
    - Each legend shape defines a mapping from component area (number of colored pixels) to color.
    - For every grey sprite, measure only its actual foreground pixels; enclosed bg holes do not add area.
    - Recolor the sprite if its area matches a legend area; otherwise erase it to bg.

    Key insight:
    - The relevant invariant is geometric area, not raw hole count. Visible cavities may still appear,
      and 1-cell pinholes are especially misleading: they change the silhouette without contributing
      an effective weighted hole, but they still remove one foreground pixel from the area.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _legend_mapping_by_area(g, bg, grey)
    assert legend, "legend mapping must be non-empty: grid must contain colored legend shapes"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        area = int(len(pts))
        newc = int(legend[area]) if area in legend else bg
        for r, c in pts:
            out[r, c] = newc
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_VALUE = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

TOTAL_SPRITES_RANGE = (8, 11)
MAX_DISTRACTORS = 7
MAX_EXTRA_PER_LEGEND = 2
GEN_RETRIES = 300
SPRITE_RETRIES = 40
SPRITE_CANVAS = 16
SPRITE_PAD = 4
SPRITE_MAX_H = 12
SPRITE_MAX_W = 18
SPRITE_MAX_ASPECT = 2.5
MAX_AREA_DELTA_FROM_BASE = 10

_STRUCT8 = generate_binary_structure(2, 2)
_PROFILE_POOL = [
    (),
    (1,),
    (2,),
    (1, 1),
    (1, 2),
    (2, 2),
    (1, 1, 1),
    (1, 1, 2),
]


def _legend_span(hole_value: int) -> int:
    """Width/height of a single legend strip along its long axis."""
    return 3 if hole_value == 0 else 2 * hole_value + 1


def _legend_patch(hole_value: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xSpan or Spanx3 legend patch using the original strip vocabulary."""
    span = _legend_span(hole_value)
    if arrangement == "horizontal":
        patch = np.full((3, span), int(color), dtype=int)
        if hole_value:
            patch[1, 1::2] = BG
    else:
        patch = np.full((span, 3), int(color), dtype=int)
        if hole_value:
            patch[1::2, 1] = BG
    return patch


def _legend_area(hole_value: int) -> int:
    """Pixel area of one legend patch; this is the new matching key."""
    return 9 if hole_value == 0 else 5 * hole_value + 3


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total long-axis span including 1-cell gaps."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Choose 2-4 distinct legend hole-values whose patches fit the panel."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes for holes in itertools.combinations(range(MAX_HOLE_VALUE), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    holes = list(random.choice(combos))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray,
    legends: list[tuple[int, int]],
    arrangement: str,
    r: int,
    c: int,
) -> None:
    """Paste legend patches sequentially with 1-cell spacing."""
    rr, cc = int(r), int(c)
    for hole_value, color in legends:
        patch = _legend_patch(hole_value, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - 4)
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "horizontal", r0, c0)
    return legends


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - 4)
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - 4)
    _place_legend_line(grid, legends, "vertical", r0, c0)
    return legends


def _l_corner_geometry(
    corner: str,
    panel_h: int,
    panel_w: int,
    arrangement: str,
    total: int,
) -> tuple[int, int, int, int, int, int]:
    """Return divider pivot and admissible legend origin ranges for an L-layout."""
    n = GRID_SIZE
    cross_dim = 3 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 3

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, n - 1 - along_dim
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim
    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(
        corner, panel_h, panel_w, arrangement, total
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


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _candidate_hole_shapes(h: int, w: int, size: int) -> list[tuple[tuple[int, int], ...]]:
    """Hole templates inside an h x w filled rectangle, expressed in interior coordinates."""
    shapes: list[tuple[tuple[int, int], ...]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if size == 1:
                shapes.append(((r, c),))
            elif size == 2:
                if c + 1 < w - 1:
                    shapes.append(((r, c), (r, c + 1)))
                if r + 1 < h - 1:
                    shapes.append(((r, c), (r + 1, c)))
    return shapes


def _holes_are_separate(existing: list[set[tuple[int, int]]], cand: tuple[tuple[int, int], ...]) -> bool:
    """Different hole components may touch diagonally, but never orthogonally."""
    cand_set = set(cand)
    for other in existing:
        if cand_set & other:
            return False
        for r, c in cand_set:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in other:
                    return False
    return True


def _place_holes_in_rect(h: int, w: int, hole_areas: tuple[int, ...]) -> list[set[tuple[int, int]]] | None:
    """Backtracking placement of 1-cell and 2-cell holes inside a filled rectangle."""
    hole_areas = tuple(sorted(hole_areas, reverse=True))
    pools = {k: _candidate_hole_shapes(h, w, k) for k in set(hole_areas)}
    for pool in pools.values():
        random.shuffle(pool)

    placed: list[set[tuple[int, int]]] = []

    def backtrack(i: int) -> bool:
        if i == len(hole_areas):
            return True
        for cand in pools[hole_areas[i]]:
            if not _holes_are_separate(placed, cand):
                continue
            placed.append(set(cand))
            if backtrack(i + 1):
                return True
            placed.pop()
        return False

    if backtrack(0):
        return [set(x) for x in placed]
    return None


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _valid_bbox(mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_of_mask(mask)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return (
        h <= SPRITE_MAX_H
        and w <= SPRITE_MAX_W
        and max(h, w) <= SPRITE_MAX_ASPECT * min(h, w)
    )


def _frontier_cells(mask: np.ndarray, forbidden: np.ndarray) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                if not mask[nr, nc] and not forbidden[nr, nc]:
                    frontier.add((int(nr), int(nc)))
    return list(frontier)


def _candidate_score(mask: np.ndarray, cell: tuple[int, int]) -> tuple[int, int]:
    """Prefer cells that keep the bbox compact and stay close to the current shape."""
    r0, c0, r1, c1 = _bbox_of_mask(mask)
    r, c = cell
    nr0, nc0, nr1, nc1 = min(r0, r), min(c0, c), max(r1, r), max(c1, c)
    expand = (nr1 - nr0 + 1) * (nc1 - nc0 + 1) - (r1 - r0 + 1) * (c1 - c0 + 1)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dist = abs(r - center_r) + abs(c - center_c)
    return int(expand), int(dist * 10)


def _grow_exterior(mask: np.ndarray, forbidden: np.ndarray, target_area: int) -> bool:
    """Add connected exterior pixels until the mask reaches target_area."""
    while int(mask.sum()) < target_area:
        frontier = _frontier_cells(mask, forbidden)
        if not frontier:
            return False
        frontier.sort(key=lambda cell: (_candidate_score(mask, cell), random.random()))
        placed = False
        for r, c in frontier[:12]:
            mask[r, c] = True
            if _valid_bbox(mask):
                placed = True
                break
            mask[r, c] = False
        if not placed:
            return False
    return True


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _make_sprite(target_area: int, hole_areas: tuple[int, ...]) -> np.ndarray | None:
    """
    Build a compact connected grey sprite with exact pixel area `target_area`.

    Hole areas are explicit: 1 = pinhole, 2 = effective cavity. This keeps the weighted-hole
    notion visible even though matching itself is by area.
    """
    hole_sum = sum(hole_areas)
    for _ in range(SPRITE_RETRIES):
        candidates: list[tuple[int, int, list[set[tuple[int, int]]], int]] = []
        for h in range(2 if not hole_areas else 3, 7):
            for w in range(2 if not hole_areas else 3, 9):
                if hole_areas:
                    holes = _place_holes_in_rect(h, w, hole_areas)
                    if holes is None:
                        continue
                else:
                    holes = []
                base_area = h * w - hole_sum
                delta = target_area - base_area
                if delta < 0 or delta > MAX_AREA_DELTA_FROM_BASE:
                    continue
                candidates.append((h, w, holes, delta))
        if not candidates:
            return None

        h, w, holes, _delta = random.choice(candidates)
        canvas = np.zeros((SPRITE_CANVAS, SPRITE_CANVAS), dtype=bool)
        forbidden = np.zeros_like(canvas, dtype=bool)
        r0 = SPRITE_PAD
        c0 = SPRITE_PAD
        canvas[r0 : r0 + h, c0 : c0 + w] = True
        for hole in holes:
            for hr, hc in hole:
                canvas[r0 + hr, c0 + hc] = False
                forbidden[r0 + hr, c0 + hc] = True

        if not _valid_bbox(canvas):
            continue
        if not _grow_exterior(canvas, forbidden, target_area):
            continue
        if int(canvas.sum()) != target_area:
            continue

        cropped = canvas.copy()
        cropped = _random_transform(cropped)
        rr0, cc0, rr1, cc1 = _bbox_of_mask(cropped)
        cropped = cropped[rr0 : rr1 + 1, cc0 : cc1 + 1]
        if not _valid_bbox(cropped):
            continue
        raw_holes, weighted_holes = _hole_counts(cropped)
        if raw_holes != len(hole_areas):
            continue
        if weighted_holes != sum(1 for x in hole_areas if x >= 2):
            continue
        if int(cropped.sum()) != target_area:
            continue
        return np.where(cropped, GREY, BG).astype(int)
    return None


def _raw_hole_count(mask: np.ndarray) -> int:
    return len(_hole_component_sizes(mask))


def _legacy_legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """Original task behavior: map raw hole count -> color."""
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
        assert len(candidates) == 1
        raw = _raw_hole_count(_component_mask(candidates[0]))
        mapping[raw] = int(c)
    return mapping


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Legacy oracle used only to reject degenerate samples."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend = _legacy_legend_mapping(g, bg, GREY)
    out = g.copy()
    for pts in components_mask(g == GREY, connectivity=4):
        raw = _raw_hole_count(_component_mask(pts))
        newc = int(legend[raw]) if raw in legend else bg
        for r, c in pts:
            out[r, c] = newc
    return out


def _sprite_stats(sprite: np.ndarray) -> dict[str, int]:
    mask = sprite == GREY
    r0, c0, r1, c1 = _bbox_of_mask(mask)
    raw_holes, weighted_holes = _hole_counts(mask)
    return {
        "area": int(mask.sum()),
        "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
        "raw_holes": int(raw_holes),
        "weighted_holes": int(weighted_holes),
    }


def _candidate_profiles_for_area(area: int, legend_hole_value: int | None) -> list[tuple[int, ...]]:
    """Profiles are ordered so counterexamples to raw-hole matching are attempted first."""
    pool = list(_PROFILE_POOL)
    random.shuffle(pool)
    if legend_hole_value is None:
        return pool
    pool.sort(key=lambda prof: (len(prof) == legend_hole_value, sum(x >= 2 for x in prof) == legend_hole_value))
    return pool


def _build_sprite_for_item(
    target_area: int,
    legend_hole_value: int | None,
    *,
    want_micro: bool = False,
    want_effective: bool = False,
    want_counterexample: bool = False,
) -> tuple[np.ndarray, dict[str, int], tuple[int, ...]] | None:
    """
    Try different hole profiles until one satisfies the requested behavioral constraints.
    """
    for profile in _candidate_profiles_for_area(target_area, legend_hole_value):
        for _ in range(3):
            sprite = _make_sprite(target_area, profile)
            if sprite is None:
                continue
            stats = _sprite_stats(sprite)
            if want_micro and not (stats["raw_holes"] > stats["weighted_holes"]):
                continue
            if want_effective and not (stats["weighted_holes"] > 0):
                continue
            if want_counterexample and legend_hole_value is not None and stats["raw_holes"] == legend_hole_value:
                continue
            return sprite, stats, profile
    return None


def _sprite_plan(
    legends: list[tuple[int, int]]
) -> list[dict[str, int | None | bool]]:
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d for d in range(1, max_d + 1)
        if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    legend_areas = [_legend_area(h) for h, _ in legends]
    distractor_area_pool = [a for a in range(6, 29) if a not in legend_areas]

    plan: list[dict[str, int | None | bool]] = []
    need_micro = True
    need_effective = True
    for i, (hole_value, color) in enumerate(legends):
        total_for_legend = 1 + extras[i]
        for j in range(total_for_legend):
            plan.append(
                {
                    "target_area": _legend_area(hole_value),
                    "color": int(color),
                    "legend_hole_value": int(hole_value),
                    "want_counterexample": bool(j == 0),
                    "want_micro": bool(need_micro),
                    "want_effective": bool(need_effective and (i > 0 or total_for_legend > 1)),
                }
            )
            if need_micro:
                need_micro = False
            elif need_effective:
                need_effective = False

    for _ in range(n_distractors):
        plan.append(
            {
                "target_area": int(random.choice(distractor_area_pool)),
                "color": None,
                "legend_hole_value": None,
                "want_counterexample": False,
                "want_micro": False,
                "want_effective": False,
            }
        )
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray,
    occupied: np.ndarray,
    sprite: np.ndarray,
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
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    legends: list[tuple[int, int]],
) -> tuple[bool, list[dict[str, int]]]:
    plan = _sprite_plan(legends)
    items: list[tuple[int, np.ndarray, int | None, dict[str, int]]] = []
    stats_records: list[dict[str, int]] = []
    matched_records: list[dict[str, int]] = []

    for item in plan:
        built = _build_sprite_for_item(
            int(item["target_area"]),
            item["legend_hole_value"],
            want_micro=bool(item["want_micro"]),
            want_effective=bool(item["want_effective"]),
            want_counterexample=bool(item["want_counterexample"]),
        )
        if built is None:
            return False, []
        sprite, stats, _profile = built
        record = {
            "target_area": int(item["target_area"]),
            "color": -1 if item["color"] is None else int(item["color"]),
            "legend_hole_value": -1 if item["legend_hole_value"] is None else int(item["legend_hole_value"]),
            "raw_holes": int(stats["raw_holes"]),
            "weighted_holes": int(stats["weighted_holes"]),
            "bbox_area": int(stats["bbox_area"]),
            "area": int(stats["area"]),
        }
        stats_records.append(record)
        if item["color"] is not None:
            matched_records.append(record)
        items.append((sprite.shape[0] * sprite.shape[1], sprite, item["color"], stats))

    if not matched_records:
        return False, []
    if not any(rec["raw_holes"] != rec["legend_hole_value"] for rec in matched_records):
        return False, []
    if not any(rec["raw_holes"] > rec["weighted_holes"] for rec in matched_records):
        return False, []
    if not any(rec["weighted_holes"] > 0 for rec in matched_records):
        return False, []
    if not any(rec["bbox_area"] > rec["area"] for rec in matched_records):
        return False, []

    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite, color, _stats in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False, []
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
        if color is not None:
            out_patch = np.where(sprite == GREY, int(color), BG)
            paste(expected, out_patch, r, c, transparent=BG)
    return True, stats_records


def _gen_attempt() -> tuple[np.ndarray, list[dict[str, int]]] | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    placed, stats_records = _place_sprites(grid, expected, blocked, legends)
    if not placed:
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(_legacy_main(grid), expected):
        return None
    return grid, stats_records


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            attempt = _gen_attempt()
        except Exception:
            attempt = None
        if attempt is not None:
            grid, _stats = attempt
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
