# Combo variant of e3721c99 (index 8)
# combo_spec: D3/1+D13/1
# Name: 图例嵌入内部+精灵沿边缘排列

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis for the fused variant:
# a) Concrete behavior shift:
#    Original samples show a border-side legend panel and grey sprites scattered in the
#    workspace. Here the roles are reversed: a central "legend chamber" sits inside the
#    grid, while every grey target sprite is flush against one border. A small example is:
#    original = [legend strip at top] + [grey blobs in middle];
#    fused    = [colored frame + legend bars in middle] + [grey blobs on top/left/right/bottom].
#    The recoloring rule still uses hole count, but the spatial parsing step changes first.
# b) Complexity calibration:
#    The chamber contains only 2-4 legend bars, so the legend-to-color mapping is still
#    readable. Edge sprites remain 8-10 total, with repeated matches per legend plus a few
#    distractors. This keeps the topology-matching idea intact without visual overload.
# c) Implicit assumption audit:
#    The original solver assumes every internal colored component is a legend candidate.
#    That assumption is intentionally broken here by adding a non-legend colored chamber
#    frame inside the grid. The new solver therefore accepts only exact 3-thick legend-bar
#    templates and ignores other internal colored components. It also treats border-touching
#    grey components as the actionable sprites for this reversed layout.
# d) Placement legality:
#    Generation reserves a central chamber, then places each grey sprite with its bbox flush
#    to one side of the grid. A 1-cell 8-neighborhood padding check prevents overlap and
#    accidental touching between sprites or the chamber.
# e) Visual observability:
#    Humans can see a compact colored structure in the center and a ring of grey objects on
#    the perimeter. This makes the "center is reference, edge is target" inversion visible
#    from a few examples without relying on color ids or hidden ordering.
# f) Visual cleanliness:
#    The chamber is separated from the border by at least four cells, and edge sprites are
#    spaced apart. No object layers are stacked in the same region, so legends, chamber, and
#    sprites remain visually distinct.
# g) Non-degeneration guard:
#    The chamber frame has one enclosed hole, and generation always includes a true 1-hole
#    legend bar. A legacy "all internal colored components are legends" solver therefore
#    fails or misreads the mapping, so samples do not collapse back to the original task.


BG = 0
FRAME = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]
MAX_HOLE_COUNT = 6
LEGEND_COUNT_RANGE = (2, 4)
TOTAL_SPRITES_RANGE = (8, 10)
MAX_DISTRACTORS = 6
MAX_EXTRA_PER_LEGEND = 2
GEN_RETRIES = 300
SPRITE_RETRIES = 36
SPRITE_AREA_0 = (4, 18)
SPRITE_AREA_LOW_MIN = 7
CANVAS_SCALE = 1.2
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 10
SPRITE_MAX_W = 14
SPRITE_MAX_ASPECT = 2.5
CENTER_MARGIN = 4

_STRUCT8 = generate_binary_structure(2, 2)
_SIDES = ["top", "bottom", "left", "right"]


def _hole_count(mask: np.ndarray) -> int:
    """Count enclosed background regions of a boolean mask using 4-connectivity."""
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _component_mask(pts: np.ndarray) -> np.ndarray:
    """Convert component coordinates into a tight boolean mask."""
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _touches_border(pts: np.ndarray, h: int, w: int) -> bool:
    """Return True iff any component cell lies on the outer border."""
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _legend_span(hole_count: int) -> int:
    """Width/height of a legend bar along its long axis."""
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    """Build a monochrome legend patch whose holes encode the count."""
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


def _legend_hole_count_if_patch(mask: np.ndarray) -> int | None:
    """Return the encoded hole count if the mask matches the legend template family."""
    for hole_count in range(MAX_HOLE_COUNT):
        for arrangement in ("horizontal", "vertical"):
            proto = _legend_patch(hole_count, 1, arrangement) == 1
            if mask.shape == proto.shape and np.array_equal(mask, proto):
                return hole_count
    return None


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Read only exact legend-template components from the interior.
    Non-template colored components (the chamber frame) are ignored.
    """
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        for pts in components_mask(g == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            mask = _component_mask(pts)
            hole_count = _legend_hole_count_if_patch(mask)
            if hole_count is None:
                continue
            assert hole_count not in mapping, f"duplicate legend for hole count {hole_count}"
            mapping[hole_count] = int(color)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor border-touching grey sprites by matching their hole counts to the embedded legend bars.

    Input structure:
    - A colored chamber sits inside the grid, away from the outer border.
    - Inside that chamber are 2-4 compact colored legend bars, each a 3-thick template whose holes
      encode a distinct count; the chamber frame itself is colored but is not a legend.
    - Grey sprites are no longer scattered through the interior: each target sprite is attached to one
      grid edge and may extend inward from that border.

    Rule:
    - Identify only the exact legend-bar templates among the internal colored components and read
      their hole counts to obtain a hole_count -> color mapping.
    - For each grey component that touches the outer border, count its enclosed holes within its own
      bounding box.
    - Recolor matching border sprites to the mapped legend color, and erase border sprites whose hole
      counts are absent from the legend set.

    Key insight:
    - The center is now the reference zone and the perimeter is the target zone, so the solver must
      first separate true legend bars from other colored internal structure before applying the same
      topology-based recoloring rule.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY
    h, w = g.shape

    legend = _legend_mapping(g, bg, grey)
    assert legend, "embedded legend mapping must be non-empty"

    out = g.copy()
    for pts in components_mask(g == grey, connectivity=4):
        if not _touches_border(pts, h, w):
            continue
        mask = _component_mask(pts)
        hole_count = _hole_count(mask)
        new_color = int(legend[hole_count]) if hole_count in legend else bg
        for r, c in pts:
            out[r, c] = new_color
    return out


def _legacy_legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """Original broader legend reader, kept only for non-degeneration checks."""
    h, w = g.shape
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        assert len(candidates) == 1, f"legacy solver found multiple candidates for color {color}"
        r0, c0, r1, c1 = candidates[0]
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
        hole_count = _hole_count(sub)
        assert hole_count not in mapping, f"legacy solver duplicate hole count {hole_count}"
        mapping[hole_count] = int(color)
    return mapping


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Reference of the original task logic, used to reject degenerate samples."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend = _legacy_legend_mapping(g, bg, GREY)
    assert legend
    out = g.copy()
    for pts in components_mask(g == GREY, connectivity=4):
        mask = _component_mask(pts)
        hole_count = _hole_count(mask)
        new_color = int(legend[hole_count]) if hole_count in legend else bg
        for r, c in pts:
            out[r, c] = new_color
    return out


def _line_span(legends: list[tuple[int, int]]) -> int:
    """Total span of a legend line including 1-cell gaps."""
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    """Choose 2-4 legend entries, always including a 1-hole legend for anti-degeneration."""
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if 1 in holes and sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    holes = list(random.choice(combos))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _build_chamber(legends: list[tuple[int, int]], arrangement: str) -> np.ndarray:
    """Create the internal chamber frame and place legend bars inside it."""
    total = _line_span(legends)
    line_h, line_w = (3, total) if arrangement == "horizontal" else (total, 3)
    patch = np.zeros((line_h + 2, line_w + 2), dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    rr, cc = 1, 1
    for hole_count, color in legends:
        piece = _legend_patch(hole_count, color, arrangement)
        paste(patch, piece, rr, cc)
        if arrangement == "horizontal":
            cc += piece.shape[1] + 1
        else:
            rr += piece.shape[0] + 1
    return patch


def _place_chamber(grid: np.ndarray) -> list[tuple[int, int]]:
    """Place a legend chamber away from the outer border and return the legend mapping list."""
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, 21)
    chamber = _build_chamber(legends, arrangement)
    h, w = chamber.shape
    base_r = (GRID_SIZE - h) // 2
    base_c = (GRID_SIZE - w) // 2
    r_lo = CENTER_MARGIN
    c_lo = CENTER_MARGIN
    r_hi = GRID_SIZE - h - CENTER_MARGIN
    c_hi = GRID_SIZE - w - CENTER_MARGIN
    r0 = min(max(base_r + random.randint(-2, 2), r_lo), r_hi)
    c0 = min(max(base_c + random.randint(-2, 2), c_lo), c_hi)
    paste(grid, chamber, r0, c0)
    return legends


def _sprite_area(hole_count: int) -> int:
    """Sample a sprite area based on hole complexity."""
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 22 + 8 * hole_count)
    return random.randint(10 + 2 * hole_count, 18 + 4 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
    """Choose separated seed cells for hole growth."""
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
    """Grow irregular forbidden blobs that become the sprite holes."""
    margin = HOLE_SEED_MARGIN
    max_hole = max(1, area // max(3, hole_count + 2))
    forbidden: set[tuple[int, int]] = set()
    for sr, sc in seeds:
        forbidden.add((sr, sc))
        blob = {(sr, sc)}
        target = random.randint(1, max_hole)
        while len(blob) < target:
            cands = []
            for r, c in blob:
                for nr, nc in neighbors4(r, c):
                    if not (margin <= nr < csz - margin and margin <= nc < csz - margin):
                        continue
                    if (nr, nc) in forbidden:
                        continue
                    if any(
                        (nnr, nnc) in forbidden and (nnr, nnc) not in blob
                        for nnr, nnc in neighbors4(nr, nc)
                    ):
                        continue
                    cands.append((nr, nc))
            if not cands:
                break
            pick = random.choice(cands)
            forbidden.add(pick)
            blob.add(pick)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    """Find the nearest fill start cell that is not reserved for holes."""
    if (center, center) not in forbidden:
        return center, center
    for dist in range(1, csz):
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                nr, nc = center + dr, center + dc
                if 0 <= nr < csz and 0 <= nc < csz and (nr, nc) not in forbidden:
                    return nr, nc
    return center, center


def _grow_mask(
    csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]
) -> np.ndarray:
    """Grow a connected blob to the target area while avoiding hole cells."""
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
    """Seal thin leaks around holes and keep the largest component."""
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
            best = max(range(1, n_comp + 1), key=lambda k: int(np.sum(labeled == k)))
            mask = labeled == best
    return mask


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
    """Crop a sprite to its tight bbox and verify size, aspect, and hole count."""
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
    """Generate a random connected grey sprite with the requested number of holes."""
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


def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    """Create a shuffled list of sprite requests: matches plus distractors."""
    total = random.randint(*TOTAL_SPRITES_RANGE)
    remaining = total - len(legends)
    max_d = min(MAX_DISTRACTORS, remaining)
    valid_d = [
        d for d in range(1, max_d + 1) if remaining - d <= MAX_EXTRA_PER_LEGEND * len(legends)
    ]
    n_distractors = random.choice(valid_d)
    extra_matches = remaining - n_distractors
    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, n in enumerate(extras) if n < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan: list[tuple[int, int | None]] = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, color))
    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), None))
    random.shuffle(plan)
    return plan


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """1-cell 8-neighborhood dilation."""
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _side_plan(count: int) -> list[str]:
    """Distribute sprites across 3 or 4 sides so the edge arrangement is obvious."""
    used_count = 4 if count >= 9 or random.random() < 0.6 else 3
    used_sides = random.sample(_SIDES, used_count)
    counts = [1] * used_count
    for _ in range(count - used_count):
        counts[random.randrange(used_count)] += 1
    sides: list[str] = []
    for side, n in zip(used_sides, counts):
        sides.extend([side] * n)
    random.shuffle(sides)
    return sides


def _unique_variants(sprite: np.ndarray) -> list[np.ndarray]:
    """Return unique rotations/reflections of a sprite."""
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(sprite, k)
        for cand in (rot, np.fliplr(rot)):
            key = (tuple(cand.shape), cand.tobytes())
            if key not in seen:
                variants.append(cand.copy())
                seen.add(key)
    random.shuffle(variants)
    return variants


def _side_positions(side: str, h: int, w: int) -> list[tuple[int, int]]:
    """All flush-to-edge positions for a bbox on the chosen side, avoiding corners."""
    if side == "top":
        return [(0, c) for c in range(1, GRID_SIZE - w)]
    if side == "bottom":
        return [(GRID_SIZE - h, c) for c in range(1, GRID_SIZE - w)]
    if side == "left":
        return [(r, 0) for r in range(1, GRID_SIZE - h)]
    return [(r, GRID_SIZE - w) for r in range(1, GRID_SIZE - h)]


def _can_place(occupied: np.ndarray, sprite: np.ndarray, r: int, c: int) -> bool:
    """Check that sprite cells do not overlap or touch existing occupied cells."""
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied)
    if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
        return False
    return True


def _place_edge_sprite(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    sprite: np.ndarray,
    color: int | None,
    preferred_side: str,
) -> bool:
    """Place one sprite along an edge, trying the preferred side first."""
    side_order = [preferred_side] + random.sample([s for s in _SIDES if s != preferred_side], 3)
    for side in side_order:
        for variant in _unique_variants(sprite):
            h, w = variant.shape
            positions = _side_positions(side, h, w)
            random.shuffle(positions)
            for r, c in positions:
                if not _can_place(occupied, variant, r, c):
                    continue
                paste(grid, variant, r, c, transparent=BG)
                occupied[r : r + h, c : c + w] |= variant == GREY
                if color is not None:
                    out_patch = np.where(variant == GREY, int(color), BG)
                    paste(expected, out_patch, r, c, transparent=BG)
                return True
    return False


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, legends: list[tuple[int, int]]
) -> bool:
    """Generate sprites and place them around the outer border."""
    requests = _sprite_plan(legends)
    side_targets = _side_plan(len(requests))
    items = []
    for (hole_count, color), side in zip(requests, side_targets):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite, color, side))
    items.sort(key=lambda item: item[0], reverse=True)

    occupied = grid != BG
    for _, sprite, color, side in items:
        if not _place_edge_sprite(grid, expected, occupied, sprite, color, side):
            return False
    return True


def _border_sprite_metrics(grid: np.ndarray) -> tuple[int, int, int]:
    """Return (#grey comps, #border-touching grey comps, #occupied border sides)."""
    h, w = grid.shape
    comps = components_mask(grid == GREY, connectivity=4)
    border_count = sum(_touches_border(pts, h, w) for pts in comps)
    sides = 0
    sides += int(np.any(grid[0, :] == GREY))
    sides += int(np.any(grid[-1, :] == GREY))
    sides += int(np.any(grid[:, 0] == GREY))
    sides += int(np.any(grid[:, -1] == GREY))
    return len(comps), border_count, sides


def _gen_attempt() -> np.ndarray | None:
    """Construct one input instance and reject samples that collapse to the legacy task."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    legends = _place_chamber(grid)
    expected = grid.copy()

    if not _place_sprites(grid, expected, legends):
        return None
    total_grey, border_grey, used_sides = _border_sprite_metrics(grid)
    if total_grey == 0 or total_grey != border_grey or used_sides < 3:
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    try:
        legacy = _legacy_main(grid)
    except Exception:
        legacy = None
    if legacy is not None and np.array_equal(legacy, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a fused D3/1 + D13/1 input grid."""
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate fused e3721c99 input")
