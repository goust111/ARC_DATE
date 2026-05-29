# Combo variant of e3721c99 (index 21)
# combo_spec: D9/1+D11/1
# Name: 迭代消除+一对多冲突解决

# Design analysis for the fused variant
# a) Concrete behavior:
#    - Base task: each grey sprite with h holes maps directly to the unique legend
#      that also has h holes.
#    - Fused task: a grey sprite with h holes may use either the h-hole legend or
#      the (h+1)-hole legend. Sprites are processed by increasing h. In round h,
#      every h-hole sprite takes the smallest still-available matching legend
#      (prefer h, otherwise h+1), then that legend is erased from the legend panel.
#    - Example: legends {2:red, 3:blue}; grey sprites with 1 hole and 2 holes.
#      Round 1: 1-hole sprites can only use 2, so they turn red and the 2-hole
#      legend disappears. Round 2: 2-hole sprites originally had two candidates
#      (2 or 3), but 2 is gone, so they turn blue. This is impossible in the base
#      task and makes the order visible in the final grid.
# b) Complexity calibration:
#    - Allowing only {h, h+1} as the conflict window keeps the ambiguity visible
#      without making every sprite compatible with every legend.
#    - The generator enforces at least one chain where round h-1 consumes legend h,
#      forcing round h to switch from legend h to legend h+1.
# c) Broken implicit assumptions from the original solver:
#    - "hole_count -> color" is no longer a static dictionary.
#    - A later sprite group cannot be solved from the input alone; it depends on
#      which lower-hole legend was consumed earlier.
#    - The legend panel is no longer inert: used legends must be erased in output.
# d) Placement legality:
#    - The generator still uses the original blocked-panel layout and padded sprite
#      placement, so legends and grey sprites never overlap or cross the divider.
# e) Visual observability:
#    - The only new moving part is the disappearance of used legends plus adjacent
#      hole-count conflicts. Both are visible with ordinary counting; no color-id
#      ordering or hidden metadata is used.
# f) Visual cleanliness:
#    - Legends remain isolated 3-cell-thick patches in a dedicated panel, while
#      sprites stay in the workspace with at least one background cell of padding.
# g) Inferrability:
#    - Repeated ambiguous rounds are generated, and the generator rejects samples
#      whose final output matches the original exact-match solver. This prevents a
#      silent fallback to the base task.

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import (
    bbox_coords,
    components_mask,
    get_bg,
    label_mask,
    neighbors4,
    paste,
)


def _hole_count(mask: np.ndarray) -> int:
    """
    Count holes in a binary mask using 4-connectivity.
    A hole is a background component that does not touch the mask's bounding-box border.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            stack = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        stack.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _component_box_mask(pts: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    """Convert a coords list into a tight bbox plus local boolean mask."""
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[int(r - r0), int(c - c0)] = True
    return int(r0), int(c0), int(r1), int(c1), sub


def _legend_entries(g: np.ndarray, bg: int, grey: int) -> list[dict[str, object]]:
    """
    Extract legend components:
      - non-bg and non-grey,
      - at least 2x2,
      - not touching the outer border (filters divider lines).
    """
    h, w = g.shape
    entries: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(g))) - {int(bg), int(grey)}):
        for pts in components_mask(g == int(color), connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            rr0, cc0, rr1, cc1, sub = _component_box_mask(pts)
            entries.append(
                {
                    "color": int(color),
                    "hole_count": int(_hole_count(sub)),
                    "bbox": (rr0, cc0, rr1, cc1),
                    "mask": sub,
                }
            )
    entries.sort(key=lambda e: (int(e["hole_count"]), int(e["bbox"][0]), int(e["bbox"][1]), int(e["color"])))
    return entries


def _grey_entries(g: np.ndarray, grey: int) -> list[dict[str, object]]:
    """Extract grey sprites as bbox/mask records with their hole counts."""
    items: list[dict[str, object]] = []
    for pts in components_mask(g == int(grey), connectivity=4):
        r0, c0, r1, c1, sub = _component_box_mask(pts)
        items.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": sub,
                "hole_count": int(_hole_count(sub)),
            }
        )
    items.sort(key=lambda e: (int(e["hole_count"]), int(e["bbox"][0]), int(e["bbox"][1])))
    return items


def _iterative_round_choices(
    legend_holes: set[int], sprite_holes: list[int]
) -> tuple[dict[int, int | None], list[int], set[int]]:
    """
    Resolve the fused rule at the level of hole-count groups.

    Each sprite group h may use legend h or h+1. Prefer h if still available,
    otherwise use h+1. After finishing round h, remove the chosen legend hole count.
    """
    available = set(map(int, legend_holes))
    choices: dict[int, int | None] = {}
    removed: list[int] = []
    for h in sorted(set(map(int, sprite_holes))):
        if h in available:
            pick = int(h)
        elif (h + 1) in available:
            pick = int(h + 1)
        else:
            pick = None
        choices[int(h)] = pick
        if pick is not None:
            available.remove(int(pick))
            removed.append(int(pick))
    return choices, removed, available


def _combo_metrics_from_holes(legend_holes: list[int], sprite_holes: list[int]) -> dict[str, int]:
    """Quantify whether D9/1 and D11/1 are both active for a sample."""
    legend_set = set(map(int, legend_holes))
    round_choices, _removed, _left = _iterative_round_choices(legend_set, sprite_holes)
    ambiguous_rounds = 0
    shifted_rounds = 0
    changed_vs_exact = 0
    for h in sorted(set(map(int, sprite_holes))):
        exact = int(h) if h in legend_set else None
        if h in legend_set and (h + 1) in legend_set:
            ambiguous_rounds += 1
        if round_choices[int(h)] == h + 1 and h in legend_set and (h + 1) in legend_set:
            shifted_rounds += 1
        if round_choices[int(h)] != exact:
            changed_vs_exact += 1
    return {
        "ambiguous_rounds": int(ambiguous_rounds),
        "shifted_rounds": int(shifted_rounds),
        "changed_vs_exact_rounds": int(changed_vs_exact),
    }


def analyze_combo_behavior(grid: np.ndarray) -> dict[str, int]:
    """Public helper for post-generation checks."""
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    grey = 5
    legends = _legend_entries(g, bg, grey)
    sprites = _grey_entries(g, grey)
    return _combo_metrics_from_holes(
        [int(e["hole_count"]) for e in legends],
        [int(e["hole_count"]) for e in sprites],
    )


def _baseline_exact_output(grid: np.ndarray) -> np.ndarray:
    """Original exact-match behavior, used only to reject degenerate generated samples."""
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    grey = 5
    legend = {int(e["hole_count"]): int(e["color"]) for e in _legend_entries(g, bg, grey)}
    out = g.copy()
    for sprite in _grey_entries(g, grey):
        color = int(legend[int(sprite["hole_count"])]) if int(sprite["hole_count"]) in legend else int(bg)
        r0, c0, _r1, _c1 = sprite["bbox"]
        mask = np.asarray(sprite["mask"], dtype=bool)
        out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by adjacent-hole legend matching, while consumed legends disappear.

    Input structure:
    - Background is the most common color. Grey sprites (color 5) sit in the workspace.
    - A legend panel near one edge contains small colored shapes whose hole counts can be read.
    - Legend counts remain distinct, but a sprite may match two adjacent legend counts: its own and one higher.

    Rule:
    - Group grey sprites by hole count and process those groups from the fewest holes upward.
    - In round h, all h-hole sprites use the smallest still-available matching legend: prefer hole count h,
      otherwise use hole count h+1. If neither is available, erase that sprite group to background.
    - After a legend is used for a round, erase that legend shape from the legend panel before later rounds.

    Key insight:
    - Matching is no longer a static hole_count -> color lookup. Some sprite groups can use two adjacent legends,
      so earlier low-hole rounds consume one option and force later rounds onto the next surviving legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    grey = 5

    legends = _legend_entries(g, bg, grey)
    assert legends, "grid must contain at least one legend component"
    sprites = _grey_entries(g, grey)

    legend_by_hole: dict[int, dict[str, object]] = {}
    for entry in legends:
        legend_by_hole.setdefault(int(entry["hole_count"]), entry)

    round_choices, _removed, _left = _iterative_round_choices(
        set(legend_by_hole.keys()),
        [int(sprite["hole_count"]) for sprite in sprites],
    )

    out = g.copy()

    sprites_by_hole: dict[int, list[dict[str, object]]] = {}
    for sprite in sprites:
        sprites_by_hole.setdefault(int(sprite["hole_count"]), []).append(sprite)

    for hole_count in sorted(sprites_by_hole):
        chosen_hole = round_choices[int(hole_count)]
        if chosen_hole is None:
            fill = int(bg)
        else:
            legend = legend_by_hole[int(chosen_hole)]
            fill = int(legend["color"])
            r0, c0, _r1, _c1 = legend["bbox"]
            mask = np.asarray(legend["mask"], dtype=bool)
            out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(bg)

        for sprite in sprites_by_hole[int(hole_count)]:
            r0, c0, _r1, _c1 = sprite["bbox"]
            mask = np.asarray(sprite["mask"], dtype=bool)
            out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(fill)

    return out


# --------------- generator (holes-first organic blob, combo-aware) ---------------

BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (8, 11)
SPRITE_RETRIES = 30
SPRITE_AREA_0 = (5, 25)
SPRITE_AREA_LOW_MIN = 8
CANVAS_SCALE = 1.2
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 12
SPRITE_MAX_W = 18
SPRITE_MAX_ASPECT = 2.5

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


def _legend_span(hole_count: int) -> int:
    """Width (or height) of a single legend patch along the arrangement axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a 3xSpan (horizontal) or Spanx3 (vertical) legend patch."""
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        patch = np.full((3, span), int(color), dtype=int)
        if hole_count:
            patch[1, 1::2] = BG
    else:
        patch = np.full((span, 3), int(color), dtype=int)
        if hole_count:
            patch[1::2, 1] = BG
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of legend patches plus 1-cell gaps between them."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> tuple[list[tuple[int, int]], int]:
    """
    Pick 2-4 distinct legend hole counts with an enforced conflict chain:
      - h and h+1 both exist,
      - h-1 does not exist,
    so round h-1 can consume legend h and force round h to shift to h+1.
    """
    del arrangement  # span only depends on hole counts
    configs: list[tuple[tuple[int, ...], int]] = []
    for count in range(LEGEND_COUNT_RANGE[0], LEGEND_COUNT_RANGE[1] + 1):
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count):
            if sum(_legend_span(h) for h in holes) + count - 1 > max_span:
                continue
            hs = set(holes)
            for conflict_h in range(1, MAX_HOLE_COUNT - 2):
                if conflict_h in hs and (conflict_h + 1) in hs and (conflict_h - 1) not in hs:
                    configs.append((holes, int(conflict_h)))
    assert configs, "failed to find a combo-valid legend configuration"

    holes, conflict_h = random.choice(configs)
    colors = random.sample(LEGEND_COLORS, len(holes))
    hole_order = list(holes)
    random.shuffle(hole_order)
    legends = list(zip(hole_order, colors))
    return legends, int(conflict_h)


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> None:
    """Paste legend patches sequentially along the arrangement axis."""
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1


def _layout_full_horizontal(
    grid: np.ndarray, blocked: np.ndarray
) -> tuple[list[tuple[int, int]], int]:
    """Place a full-row horizontal divider with legend panel above or below."""
    legends, conflict_h = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
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
    return legends, int(conflict_h)


def _layout_full_vertical(
    grid: np.ndarray, blocked: np.ndarray
) -> tuple[list[tuple[int, int]], int]:
    """Place a full-column vertical divider with legend panel left or right."""
    legends, conflict_h = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
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
    return legends, int(conflict_h)


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
    """Compute divider pivot and legend placement bounds for an L-corner."""
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

    return int(dr), int(dc), int(r_min), int(r_max), int(c_min), int(c_max)


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    """Place an L-shaped divider with legend panel in one corner."""
    arrangement = random.choice(["horizontal", "vertical"])
    legends, conflict_h = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(corner, panel_h, panel_w, arrangement, total)

    n = GRID_SIZE
    if corner == "tl":
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "tr":
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    _place_legend_line(grid, legends, arrangement, r0, c0)
    return legends, int(conflict_h)


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    """Randomly choose and apply a divider layout."""
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_area(hole_count: int) -> int:
    """Sample a random sprite area based on hole count."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Find well-separated seed positions for holes inside the canvas."""
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area ** 0.5))
    positions = [
        (r, c)
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
        if margin <= r < csz - margin and margin <= c < csz - margin
    ]
    random.shuffle(positions)
    seeds: list[tuple[int, int]] = []
    for r, c in positions:
        if all(max(abs(r - sr), abs(c - sc)) >= HOLE_SEED_MIN_SEP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == hole_count:
                return seeds
    return None


def _grow_hole_blobs(
    seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int
) -> set[tuple[int, int]]:
    """Grow irregular blob regions around each hole seed."""
    margin = HOLE_SEED_MARGIN
    max_hole = max(2, area // (hole_count + 2)) if hole_count <= 2 else max(1, area // (hole_count * 2 + 1))
    forbidden: set[tuple[int, int]] = set()
    for sr, sc in seeds:
        forbidden.add((sr, sc))
        blob = {(sr, sc)}
        target = random.randint(1, max(1, max_hole))
        while len(blob) < target:
            cands: list[tuple[int, int]] = []
            for r, c in blob:
                for nr, nc in neighbors4(r, c):
                    if not (margin <= nr < csz - margin and margin <= nc < csz - margin):
                        continue
                    if (nr, nc) in forbidden:
                        continue
                    if any((nnr, nnc) in forbidden and (nnr, nnc) not in blob for nnr, nnc in neighbors4(nr, nc)):
                        continue
                    cands.append((nr, nc))
            if not cands:
                break
            chosen = random.choice(cands)
            forbidden.add(chosen)
            blob.add(chosen)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    """Find the nearest non-forbidden cell to the canvas center."""
    if (center, center) not in forbidden:
        return center, center
    for d in range(1, csz):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
                nr, nc = center + dr, center + dc
                if 0 <= nr < csz and 0 <= nc < csz and (nr, nc) not in forbidden:
                    return nr, nc
    return center, center


def _grow_mask(
    csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]
) -> np.ndarray:
    """Grow a random connected blob of target area, avoiding forbidden cells."""
    mask = np.zeros((csz, csz), dtype=bool)
    sr, sc = start
    mask[sr, sc] = True
    frontier = list(neighbors4(sr, sc))
    filled = 1
    while filled < area and frontier:
        idx = random.randrange(len(frontier))
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        r, c = frontier.pop()
        if not (0 <= r < csz and 0 <= c < csz) or mask[r, c] or (r, c) in forbidden:
            continue
        mask[r, c] = True
        filled += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc]:
                frontier.append((nr, nc))
    return mask


def _seal_and_connect(mask: np.ndarray, forbidden: set[tuple[int, int]]) -> np.ndarray:
    """Seal gaps around forbidden cells and keep only the largest connected component."""
    csz = mask.shape[0]
    for _ in range(SEAL_ITERS):
        sealed = False
        for r, c in sorted(forbidden):
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc] and (nr, nc) not in forbidden:
                    mask[nr, nc] = True
                    sealed = True
        if not sealed:
            break
    if np.any(mask):
        labeled, n_comp = label_mask(mask)
        if n_comp > 1:
            best = max(range(1, n_comp + 1), key=lambda lab: int(np.sum(labeled == lab)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop mask to tight bbox (including forbidden cells), validate shape and holes."""
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)
    for r, c in forbidden:
        rmin, cmin = min(rmin, r), min(cmin, c)
        rmax, cmax = max(rmax, r), max(cmax, c)
    h, w = rmax - rmin + 1, cmax - cmin + 1
    if h > SPRITE_MAX_H or w > SPRITE_MAX_W or max(h, w) > SPRITE_MAX_ASPECT * min(h, w):
        return None
    cropped = mask[rmin : rmax + 1, cmin : cmax + 1].copy()
    if _hole_count(cropped) != hole_count:
        return None
    return np.where(cropped, GREY, BG).astype(int)


def _make_sprite(hole_count: int) -> np.ndarray | None:
    """Generate a random amorphous grey sprite with the specified number of holes."""
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area ** 0.5 * CANVAS_SCALE) + CANVAS_PAD)
        center = csz // 2

        forbidden: set[tuple[int, int]] = set()
        if hole_count > 0:
            seeds = _place_hole_seeds(hole_count, center, csz, area)
            if seeds is None:
                continue
            forbidden = _grow_hole_blobs(seeds, hole_count, area, csz)

        start = _find_start(center, csz, forbidden)
        mask = _grow_mask(csz, area, start, forbidden)
        mask = _seal_and_connect(mask, forbidden)

        result = _crop_and_validate(mask, forbidden, hole_count)
        if result is not None:
            return result
    return None


def _sprite_plan(legends: list[tuple[int, int]], conflict_h: int) -> list[int]:
    """Build a combo-aware list of sprite hole counts."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    legend_holes = {h for h, _ in legends}
    extra_legend_holes = [h for h in legend_holes if h not in {conflict_h, conflict_h + 1}]
    distractor_pool = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]

    plan: list[int] = []
    plan.extend([conflict_h - 1] * random.randint(1, 2))
    plan.extend([conflict_h] * random.randint(2, 3))
    if random.random() < 0.7:
        plan.extend([conflict_h + 1] * random.randint(1, 2))

    for h in extra_legend_holes:
        if random.random() < 0.8:
            plan.extend([h] * random.randint(1, 2))

    weighted: list[int] = [conflict_h - 1] * 3 + [conflict_h] * 4 + [conflict_h + 1] * 2
    for h in extra_legend_holes:
        weighted.extend([h] * 2)
    for h in distractor_pool:
        weighted.append(h)

    while len(plan) < total:
        plan.append(random.choice(weighted))

    random.shuffle(plan)
    return plan[:total]


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """8-connectivity dilation of a boolean mask."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(
    static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray
) -> tuple[int, int] | None:
    """Find a random valid placement for sprite, avoiding blocked and occupied areas."""
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


def _place_sprites(grid: np.ndarray, blocked: np.ndarray, sprite_holes: list[int]) -> bool:
    """Generate and place all grey sprites onto the grid."""
    items: list[tuple[int, np.ndarray]] = []
    for hole_count in sprite_holes:
        sprite = _make_sprite(int(hole_count))
        if sprite is None:
            return False
        items.append((int(sprite.size), sprite))
    items.sort(key=lambda x: x[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _size, sprite in items:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends, conflict_h = _add_layout(grid, blocked)
    sprite_holes = _sprite_plan(legends, conflict_h)
    plan_metrics = _combo_metrics_from_holes([h for h, _ in legends], sprite_holes)
    if plan_metrics["ambiguous_rounds"] < 1 or plan_metrics["shifted_rounds"] < 1:
        return None

    if not _place_sprites(grid, blocked, sprite_holes):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None

    combo_metrics = analyze_combo_behavior(grid)
    if combo_metrics["ambiguous_rounds"] < 1 or combo_metrics["shifted_rounds"] < 1:
        return None

    combo_out = main(grid)
    if np.array_equal(combo_out, _baseline_exact_output(grid)):
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
