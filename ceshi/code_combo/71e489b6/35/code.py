# Combo variant of 71e489b6 (index 35)
# combo_spec: D2/1+D8/2
# Name: 基于形状检测+更大缺陷

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for combo D2/1 + D8/2
# a) Concrete behavior trace:
#    - Source task after spur absorption:
#        1 1 1 1      1 7 7 1
#        1 0 0 1  ->  1 0 0 1
#        1 1 1 1      1 7 7 1
#      A 1x2 black bar is size<=3, so the original solver marks it.
#    - Fused task on the same local pattern:
#        1 1 1 1      1 1 1 1
#        1 0 0 1  ->  1 0 0 1
#        1 1 1 1      1 1 1 1
#      The bar is a filled rectangle, so it is ignored despite being small.
#    - Source task on a 4-cell L defect:
#        1 0 1        7 0 7
#        1 0 0   ->   7 0 0
#        1 1 1        1 7 7
#      Only the leaf pixels of the size>3 black component are targets.
#    - Fused task on the same 4-cell L defect:
#        1 0 1        7 0 7
#        1 0 0   ->   7 0 0
#        1 7 7        7 7 7
#      The whole non-rectangular component is the defect, so the halo is built
#      around all four black cells rather than only its tips.
# b) Complexity calibration:
#    - The change must stay visible in 2-4 examples, so accepted boards always
#      include both classes: at least one larger non-rectangular defect (size
#      4-6) that is marked, and at least one small rectangular black component
#      (size 1-3) that remains unmarked.
#    - This avoids a degenerate regime where the new rule accidentally behaves
#      like the source rule on every sample.
# c) Hidden assumptions audit:
#    - Original STEP 1 assumed "size<=3 => mark whole component". That premise
#      is invalid here and must be removed.
#    - Original large-component handling assumed "only degree<=1 pixels matter".
#      That is also invalid: a non-rectangular 6-cell T should be marked as one
#      object, not reduced to endpoints.
#    - STEP 0 remains unchanged because D2/1 and D8/2 only alter black-defect
#      geometry, not the blue-spur cleanup predicate.
#    - `main()` now assumes meaningful defects are black 4-connected components
#      strictly enclosed by blue on their 4-neighborhood. The generator enforces
#      that every placed anomaly is fully interior to a blue region.
# d) Placement legality:
#    - Every planned defect cell must be in bounds, currently blue, and every
#      non-component 4-neighbor must also be in-bounds blue. This prevents
#      overlaps, border contact, and accidental attachment to large black areas.
#    - Components are separated by a Chebyshev buffer so halos remain readable.
# e) Visual observability:
#    - A human can infer the new rule from repeated contrast: bent L/T/zigzag
#      black islands get orange halos, while equally small straight bars and
#      rectangles do not.
#    - Because the generator includes both marked and ignored black islands
#      inside the same blue regions, the relevant cue is shape rather than size.
# f) Visual cleanliness:
#    - The palette stays {0,1,7}. Large partition regions remain simple stripes,
#      insets, or L-shapes. Only a handful of interior anomalies are injected,
#      and they are spaced apart to avoid one giant merged orange cloud.
# g) Inferability and degeneration check:
#    - The rule depends only on visible geometry: "fills its bounding box" vs
#      "leaves empty bbox cells". No hidden numeric ordering is used.
#    - Generation rejects samples unless the fused solver output differs from a
#      reference implementation of the source rule on the same input.
# h) New solver/generator contract:
#    - `main()` assumes all black components that are blue-enclosed and meant to
#      be ignored are true rectangles under bbox filling. `generate_input()`
#      creates ignored distractors only from 1xk, kx1, 2x2, and 2x3 rectangles,
#      so that assumption is exact rather than heuristic.


BLACK = 0
BLUE = 1
ORANGE = 7

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 30
MIN_BLACK_REGION = 20
PARTITION_STYLES = ("h_stripes", "rect_inset", "l_shape")
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
TARGET_COUNT_RANGE = (2, 4)
RECT_COUNT_RANGE = (2, 4)
SPUR_COUNT_RANGE = (2, 6)
MIN_TARGET_COMPONENTS = 2
MIN_RECT_COMPONENTS = 2
PLACEMENT_CLEARANCE = 3
PLACEMENT_TRIES = 200
MAX_GEN_ATTEMPTS = 800


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(norm)


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    pts = list(cells)
    for flip in (1, -1):
        for rot in range(4):
            transformed = []
            for r, c in pts:
                rr, cc = r, c * flip
                for _ in range(rot):
                    rr, cc = cc, -rr
                transformed.append((rr, cc))
            variants.add(_normalize_shape(tuple(transformed)))
    return sorted(variants)


def _rectangle_shape(height: int, width: int) -> tuple[tuple[int, int], ...]:
    return tuple((r, c) for r in range(height) for c in range(width))


BASE_NONRECT_SHAPES = (
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)),
)

NONRECT_TEMPLATES = sorted(
    {variant for shape in BASE_NONRECT_SHAPES for variant in _shape_variants(shape)},
    key=lambda cells: (len(cells), cells[-1][0], cells[-1][1], cells),
)
NONRECT_TEMPLATES_LARGE = [cells for cells in NONRECT_TEMPLATES if 4 <= len(cells) <= 6]

RECT_TEMPLATES = [
    _normalize_shape(_rectangle_shape(h, w))
    for h, w in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (1, 4), (4, 1), (1, 5),
                 (5, 1), (2, 3), (3, 2), (1, 6), (6, 1))
]
RECT_TEMPLATES_SMALL = [cells for cells in RECT_TEMPLATES if len(cells) <= 3]


def _nearby(r: int, c: int, dist: int, coords: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in coords:
                return True
    return False


def _effective_black_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    H, W = grid.shape
    black_n = 0
    oob = 0
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < H and 0 <= cc < W):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return black_n


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Mirror the source task's row-major blue-spur absorption pass."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            if _effective_black_neighbors(base, r, c) >= 3:
                base[r, c] = BLACK
    return base


def _component_bbox(comp_set: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in comp_set]
    cs = [c for _, c in comp_set]
    return min(rs), min(cs), max(rs), max(cs)


def _is_rectangle_component(comp_set: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _component_bbox(comp_set)
    return len(comp_set) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_blue_enclosed_component(grid: np.ndarray, comp_set: set[tuple[int, int]]) -> bool:
    H, W = grid.shape
    for r, c in comp_set:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in comp_set:
                continue
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if int(grid[rr, cc]) != BLUE:
                return False
    return True


def _classify_blue_enclosed_components(
    grid: np.ndarray,
) -> tuple[list[set[tuple[int, int]]], list[set[tuple[int, int]]]]:
    targets: list[set[tuple[int, int]]] = []
    ignored_rects: list[set[tuple[int, int]]] = []
    for comp in components_mask(grid == BLACK, connectivity=4):
        comp_set = {(int(r), int(c)) for r, c in comp}
        if not _is_blue_enclosed_component(grid, comp_set):
            continue
        if _is_rectangle_component(comp_set):
            ignored_rects.append(comp_set)
        else:
            targets.append(comp_set)
    return targets, ignored_rects


def _paint_halo(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int)
    H, W = out.shape
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _legacy_collect_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    """Reference source-task detector used only for degeneration checks."""
    H, W = grid.shape
    preserve: set[tuple[int, int]] = set()
    for comp in components_mask(grid == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in comp]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) in comp_set:
                    black_n += 1
            if black_n <= 1:
                preserve.add((r, c))
    return preserve


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    base = _absorb_blue_spurs(grid)
    preserve = _legacy_collect_targets(base)
    return _paint_halo(base, preserve)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Surround only non-rectangular black islands inside blue regions with orange halos.

    Input structure:
    - Large blue (1) and black (0) partition regions fill the board, and some black regions may also contain thin blue spur pixels that should collapse away.
    - Inside blue-region interiors there are several isolated black 4-connected islands ranging from 1 to 6 cells.
    - Some of those islands are filled rectangles such as 1x2 bars or 2x2 blocks, while others are bent shapes such as L, T, or stair-step forms.

    Rule:
    - First absorb any blue pixel that has at least three effective black 4-neighbor sides, using the same corner treatment as in the source task.
    - Then inspect black 4-connected components that are strictly enclosed by blue on their 4-neighborhood.
    - Mark only the non-rectangular enclosed components: paint orange (7) on every Chebyshev-1 neighbor of the component while keeping the component's own pixels black. Enclosed rectangular black components are left unchanged.

    Key insight:
    - Size is no longer the deciding cue. The visible distinction is whether a black island completely fills its own bounding box; bent shapes are highlighted, compact rectangles are not.
    """
    base = _absorb_blue_spurs(np.asarray(grid, dtype=int))
    target_components, _ = _classify_blue_enclosed_components(base)
    preserve = {cell for comp in target_components for cell in comp}
    return _paint_halo(base, preserve)


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a large-scale blue/black partition similar to the source task."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_choices = list(range(CUT_MARGIN, H - CUT_MARGIN))
        cuts = sorted(random.sample(cut_choices, min(n_cuts, len(cut_choices))))
        boundaries = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for idx in range(len(boundaries) - 1):
            grid[boundaries[idx]:boundaries[idx + 1], :] = (first_color + idx) % 2
        return grid

    if style == "rect_inset":
        grid[:, :] = BLACK
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0:r1 + 1, c0:c1 + 1] = BLUE
        return grid

    grid[:, :] = BLACK
    vc = random.randint(W // 3, 2 * W // 3)
    hc = random.randint(H // 3, 2 * H // 3)
    if random.random() < 0.5:
        grid[:, :vc] = BLUE
        grid[:hc, vc:] = BLUE
    else:
        grid[:, vc:] = BLUE
        grid[hc:, :vc] = BLUE
    return grid


def _region_sizes_ok(grid: np.ndarray) -> bool:
    for comp in components_mask(grid == BLUE, connectivity=4):
        if len(comp) < MIN_BLUE_REGION:
            return False
    for comp in components_mask(grid == BLACK, connectivity=4):
        if len(comp) < MIN_BLACK_REGION:
            return False
    return True


def _can_place_component(
    grid: np.ndarray,
    coords: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    for r, c in coords:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != BLUE:
            return False
        if _nearby(r, c, PLACEMENT_CLEARANCE, occupied):
            return False
    for r, c in coords:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in coords:
                continue
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if int(grid[rr, cc]) != BLUE:
                return False
    return True


def _place_component_from_templates(
    grid: np.ndarray,
    templates: list[tuple[tuple[int, int], ...]],
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    for _ in range(PLACEMENT_TRIES):
        template = random.choice(templates)
        max_r = max(r for r, _ in template)
        max_c = max(c for _, c in template)
        if H - max_r <= 0 or W - max_c <= 0:
            continue
        anchor_r = random.randint(0, H - max_r - 1)
        anchor_c = random.randint(0, W - max_c - 1)
        coords = {(anchor_r + dr, anchor_c + dc) for dr, dc in template}
        if not _can_place_component(grid, coords, occupied):
            continue
        for r, c in coords:
            grid[r, c] = BLACK
        occupied.update(coords)
        return coords
    return None


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find black pixels that the solver's spur-absorption pass would delete if made blue."""
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BLACK or (r, c) in exclude:
                continue
            if _effective_black_neighbors(grid, r, c) >= 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def analyze_input(grid: np.ndarray) -> dict[str, int | bool]:
    """Quantify the fused behavior on a generated input."""
    base = _absorb_blue_spurs(np.asarray(grid, dtype=int))
    target_components, rect_components = _classify_blue_enclosed_components(base)
    metrics: dict[str, int | bool] = {
        "target_components": len(target_components),
        "rect_components": len(rect_components),
        "large_target_components": sum(1 for comp in target_components if len(comp) >= 4),
        "small_rect_components": sum(1 for comp in rect_components if len(comp) <= 3),
        "legacy_differs": not np.array_equal(main(grid), _legacy_main(grid)),
    }
    return metrics


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)
    if not _region_sizes_ok(grid):
        return None

    occupied: set[tuple[int, int]] = set()
    target_components: list[set[tuple[int, int]]] = []
    rect_components: list[set[tuple[int, int]]] = []

    mandatory_target = _place_component_from_templates(grid, NONRECT_TEMPLATES_LARGE, occupied)
    if mandatory_target is None:
        return None
    target_components.append(mandatory_target)

    mandatory_rect = _place_component_from_templates(grid, RECT_TEMPLATES_SMALL, occupied)
    if mandatory_rect is None:
        return None
    rect_components.append(mandatory_rect)

    extra_targets = random.randint(max(0, TARGET_COUNT_RANGE[0] - 1), TARGET_COUNT_RANGE[1] - 1)
    extra_rects = random.randint(max(0, RECT_COUNT_RANGE[0] - 1), RECT_COUNT_RANGE[1] - 1)

    for _ in range(extra_targets):
        comp = _place_component_from_templates(grid, NONRECT_TEMPLATES, occupied)
        if comp is not None:
            target_components.append(comp)

    for _ in range(extra_rects):
        comp = _place_component_from_templates(grid, RECT_TEMPLATES, occupied)
        if comp is not None:
            rect_components.append(comp)

    clean = np.array(grid, dtype=int)
    target_sets, rect_sets = _classify_blue_enclosed_components(clean)
    if len(target_sets) < MIN_TARGET_COMPONENTS or len(rect_sets) < MIN_RECT_COMPONENTS:
        return None
    if not any(len(comp) >= 4 for comp in target_sets):
        return None
    if not any(len(comp) <= 3 for comp in rect_sets):
        return None

    preserve = {cell for comp in target_sets for cell in comp}
    output = _paint_halo(clean, preserve)

    inp = clean.copy()
    n_spurs = random.randint(*SPUR_COUNT_RANGE)
    spur_candidates = _find_spur_candidates(inp, preserve | {cell for comp in rect_sets for cell in comp})
    used_spurs = spur_candidates[:n_spurs]
    if len(used_spurs) < min(2, n_spurs):
        return None
    for r, c in used_spurs:
        inp[r, c] = BLUE

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    metrics = analyze_input(inp)
    if int(metrics["target_components"]) < MIN_TARGET_COMPONENTS:
        return None
    if int(metrics["rect_components"]) < MIN_RECT_COMPONENTS:
        return None
    if int(metrics["large_target_components"]) < 1:
        return None
    if int(metrics["small_rect_components"]) < 1:
        return None
    if not bool(metrics["legacy_differs"]):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
