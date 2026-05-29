# Combo variant of 71e489b6 (index 31)
# combo_spec: D5/2+D10/0
# Name: 区域位置分色光晕+多色前景

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask  # noqa: E402


# Design analysis for combo D5/2 + D10/0
# 1. Concrete behavior:
#    - Original 71e489b6 has one blue foreground color and one orange halo color.
#      Defect detection is structural: absorb colored spurs in black, then mark
#      tiny black components and leaf pixels of large black components.
#    - This combo keeps the structural detector but replaces the single foreground
#      with 2-3 large foreground regions of distinct colors. The regions are still
#      large, rectangular, and separated by black so the visual grammar remains
#      close to the source task.
#    - D5/2 says the halo depends on region identity; D10/0 makes region identity
#      visible by giving each position-ranked region its own foreground color.
#      The generator sorts regions by reading order and assigns:
#      first region -> blue(1) with orange halo(7),
#      second region -> green(3) with pink halo(6),
#      third region -> red(2) with yellow halo(4).
#      Solver-side this becomes a stable foreground-color -> halo-color mapping.
# 2. Small example trace:
#    - Suppose a green rectangle contains a 1-cell black hole. The original task
#      would paint an orange 3x3 halo. Here the same detector still finds that
#      1-cell black component, but its surrounding foreground color is green, so
#      the halo becomes pink instead of orange.
#    - Suppose a black background protrudes by one pixel into the red rectangle.
#      The original task would mark the leaf orange. Here the pixel is still a
#      leaf of a large black component, but because its non-black neighbors are
#      red, the halo becomes yellow.
# 3. Complexity calibration:
#    - Too many colors would make the mapping hard to infer from 2-4 examples, so
#      the palette is capped at three foreground colors and three halo colors.
#    - Every generated foreground region gets at least one defect, so each color
#      repeats and its halo color is supported inside the same sample.
# 4. Broken original assumptions and fixes:
#    - Original `main()` assumes every non-black pixel is blue. Now spur removal
#      must treat any foreground color as removable if it is surrounded by black.
#    - Original halo painting uses one global halo color. Now each target must be
#      assigned an owner region color first, then painted with that region's halo.
#    - Original small black components can be labeled without context. Now the
#      solver must inspect neighboring foreground colors to infer which region owns
#      the defect or leaf.
# 5. Placement legality / cleanliness:
#    - Foreground regions are large rectangles or stripes with black gaps, so
#      defects fit strictly inside one color region and leaves touch exactly one
#      black side.
#    - Targets are kept at Chebyshev distance >= 3 from targets in other regions,
#      preventing cross-color halo overlap and keeping outputs readable.
# 6. Observability / anti-degeneracy:
#    - Samples always contain at least two foreground colors and at least two halo
#      colors. The generator also rejects cases whose output matches a hypothetical
#      single-orange version, so the combo cannot silently collapse back to the
#      original behavior.


FOREGROUND_TO_HALO = {1: 7, 3: 6, 2: 4}
REGION_COLOR_ORDER = (1, 3, 2)
HALO_COLORS = frozenset(FOREGROUND_TO_HALO.values())

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MAX_GEN_ATTEMPTS = 400

REGION_COUNT_CHOICES = (2, 3)
LAYOUT_STYLES = ("h_bands", "v_bands", "blocks")
MIN_REGION_HEIGHT = 3
MIN_REGION_WIDTH = 3
REGION_GAP = 2
OUTER_MARGIN = 1
MIN_REGION_AREA = 30

DEFECT_SIZE_WEIGHTS = [6, 2, 1]
MIN_DEFECTS_PER_REGION = 1
EXTRA_DEFECT_RANGE = (0, 3)
LEAF_PROB = 0.6
SPUR_COUNT_RANGE = (2, 6)
TARGET_CLEARANCE = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _nearby(component: set[tuple[int, int]], ref: set[tuple[int, int]], dist: int) -> bool:
    for r, c in component:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                if (r + dr, c + dc) in ref:
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
        if int(grid[rr, cc]) == 0:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return black_n


def _embedded_component_owner(grid: np.ndarray, pixels: set[tuple[int, int]]) -> int | None:
    colors: set[int] = set()
    H, W = grid.shape
    for r, c in pixels:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in pixels:
                continue
            if not (0 <= rr < H and 0 <= cc < W):
                return None
            color = int(grid[rr, cc])
            if color not in FOREGROUND_TO_HALO:
                return None
            colors.add(color)
    if len(colors) != 1:
        return None
    return next(iter(colors))


def _leaf_owner_color(grid: np.ndarray, r: int, c: int) -> int | None:
    H, W = grid.shape
    black_n = 0
    colors: list[int] = []
    for rr, cc in _neighbors4(r, c):
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        color = int(grid[rr, cc])
        if color == 0:
            black_n += 1
            continue
        if color not in FOREGROUND_TO_HALO:
            return None
        colors.append(color)
    if black_n != 1 or len(colors) != 3 or len(set(colors)) != 1:
        return None
    return colors[0]


def _collect_targets(grid: np.ndarray) -> tuple[list[tuple[set[tuple[int, int]], int]], set[tuple[int, int]]]:
    H, W = grid.shape
    preserve: set[tuple[int, int]] = set()
    targets: list[tuple[set[tuple[int, int]], int]] = []

    comps = []
    for coords in components_mask(grid == 0, connectivity=4):
        comps.append([(int(r), int(c)) for r, c in coords])

    for pts_list in comps:
        pts = set(pts_list)
        if len(pts_list) <= 3:
            owner = _embedded_component_owner(grid, pts)
            if owner is None:
                continue
            preserve.update(pts)
            targets.append((pts, owner))
            continue
        for r, c in pts_list:
            owner = _leaf_owner_color(grid, r, c)
            if owner is not None:
                preserve.add((r, c))
                targets.append(({(r, c)}, owner))
    return targets, preserve


def _paint_targets(base: np.ndarray, targets: list[tuple[set[tuple[int, int]], int]], *, force_halo: int | None = None) -> np.ndarray:
    out = np.array(base, dtype=int)
    preserve = set()
    for pixels, _ in targets:
        preserve.update(pixels)
    for pixels, owner_color in targets:
        halo_color = force_halo if force_halo is not None else FOREGROUND_TO_HALO[owner_color]
        for r, c in pixels:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and (rr, cc) not in preserve:
                        out[rr, cc] = halo_color
    for r, c in preserve:
        out[r, c] = 0
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight black defects in multi-colored foreground regions with region-specific halo colors.

    Input structure:
    - Black (0) background plus 2-3 large foreground regions in distinct colors such as blue (1),
      green (3), and red (2).
    - Small black holes may sit strictly inside a foreground region, and large black regions may send
      one-pixel leaf intrusions into those colored regions.
    - Tiny foreground-colored specks can also appear inside black areas; they are removable spurs.

    Rule:
    - First absorb any foreground-colored spur that is surrounded by black on at least three effective
      4-neighbor sides, using the original corner-aware rule.
    - Then detect every small black 4-component (size <= 3) and every degree-0/1 leaf pixel of each
      larger black component.
    - For each detected black target, inspect which foreground color surrounds it and paint a
      Chebyshev-1 halo in that region's paired marker color:
      blue->orange, green->pink, red->yellow. The target pixel itself stays black.

    Key insight: The structural detector is unchanged; only the ownership cue changes. A black anomaly
    inherits its halo color from the colored region it belongs to, so the solver must recover
    region membership before painting.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    base = g.copy()
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) not in FOREGROUND_TO_HALO:
                continue
            if _effective_black_neighbors(base, r, c) >= 3:
                base[r, c] = 0

    targets, preserve = _collect_targets(base)
    out = _paint_targets(base, targets)
    for r, c in preserve:
        out[r, c] = 0
    return out


def _split_sizes(total: int, parts: int, min_size: int) -> list[int] | None:
    if total < parts * min_size:
        return None
    sizes = [min_size] * parts
    extra = total - parts * min_size
    for idx in _split_sizes.random_order[:parts]:
        if extra <= 0:
            break
        add = random.randint(0, extra)
        sizes[idx] += add
        extra -= add
    if extra > 0:
        sizes[-1] += extra
    random.shuffle(sizes)
    return sizes


_split_sizes.random_order = [0, 1, 2, 3, 4]


def _sorted_regions(bboxes: list[tuple[int, int, int, int]]) -> list[dict]:
    ordered = sorted(bboxes, key=lambda box: (box[0], box[1]))
    regions = []
    for idx, bbox in enumerate(ordered):
        color = REGION_COLOR_ORDER[idx]
        regions.append({"bbox": bbox, "color": color, "index": idx})
    return regions


def _build_band_regions(H: int, W: int, *, horizontal: bool) -> list[dict] | None:
    n_regions = random.choice(REGION_COUNT_CHOICES)
    usable = (H if horizontal else W) - 2 * OUTER_MARGIN - REGION_GAP * (n_regions - 1)
    min_size = MIN_REGION_HEIGHT if horizontal else MIN_REGION_WIDTH
    sizes = _split_sizes(usable, n_regions, min_size)
    if sizes is None:
        return None

    bboxes = []
    pos = OUTER_MARGIN
    for size in sizes:
        if horizontal:
            bbox = (pos, 0, pos + size - 1, W - 1)
        else:
            bbox = (0, pos, H - 1, pos + size - 1)
        bboxes.append(bbox)
        pos += size + REGION_GAP
    return _sorted_regions(bboxes)


def _jitter_interval(start: int, end: int, lo: int, hi: int, slack: int) -> tuple[int, int]:
    left = random.randint(max(lo, start - slack), start)
    right = random.randint(end, min(hi, end + slack))
    return left, right


def _build_block_regions(H: int, W: int) -> list[dict] | None:
    n_regions = random.choice(REGION_COUNT_CHOICES)
    bboxes: list[tuple[int, int, int, int]] = []
    if n_regions == 2:
        top = 1
        bottom = H - 2
        left_end = W // 2 - 2
        right_start = W // 2 + 1
        r0, r1 = _jitter_interval(top + 1, bottom - 1, 1, H - 2, 1)
        bboxes.append((r0 - 1, 1, r1, left_end))
        r0b, r1b = _jitter_interval(top + 1, bottom - 1, 1, H - 2, 1)
        bboxes.append((r0b - 1, right_start, r1b, W - 2))
    else:
        top_h = max(5, H // 3 + 1)
        bboxes.append((1, 1, top_h, W // 2 - 2))
        bboxes.append((1, W // 2 + 1, top_h, W - 2))
        bboxes.append((top_h + 2, W // 4, H - 2, W - W // 4 - 1))
    regions = _sorted_regions(bboxes)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        if r1 - r0 + 1 < MIN_REGION_HEIGHT or c1 - c0 + 1 < MIN_REGION_WIDTH:
            return None
        if (r1 - r0 + 1) * (c1 - c0 + 1) < MIN_REGION_AREA:
            return None
    return regions


def _build_regions(H: int, W: int) -> list[dict] | None:
    style = random.choice(LAYOUT_STYLES)
    if style == "h_bands":
        return _build_band_regions(H, W, horizontal=True)
    if style == "v_bands":
        return _build_band_regions(H, W, horizontal=False)
    return _build_block_regions(H, W)


def _fill_regions(H: int, W: int, regions: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = region["color"]
    return grid


def _valid_defect_shape(grid: np.ndarray, region: dict, cells: list[tuple[int, int]]) -> bool:
    region_color = region["color"]
    cell_set = set(cells)
    for r, c in cells:
        if int(grid[r, c]) != region_color:
            return False
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in cell_set:
                continue
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                return False
            if int(grid[rr, cc]) != region_color:
                return False
    return True


def _place_region_defect(
    grid: np.ndarray,
    region: dict,
    preserve: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int] | None:
    r0, c0, r1, c1 = region["bbox"]
    seeds = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    random.shuffle(seeds)
    shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
    ]
    defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
    shape_choices = [shape for shape in shapes if len(shape) == defect_size]
    if not shape_choices:
        return None

    for seed_r, seed_c in seeds:
        random.shuffle(shape_choices)
        for offsets in shape_choices:
            cells = [(seed_r + dr, seed_c + dc) for dr, dc in offsets]
            if not all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in cells):
                continue
            cell_set = set(cells)
            if _nearby(cell_set, preserve, TARGET_CLEARANCE):
                continue
            if not _valid_defect_shape(grid, region, cells):
                continue
            for r, c in cells:
                grid[r, c] = 0
            return cell_set, region["color"]
    return None


def _leaf_candidates(grid: np.ndarray, region: dict, preserve: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region["bbox"]
    color = region["color"]
    candidates = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) != color:
                continue
            if r in {r0, r1} and c in {c0, c1}:
                continue
            cell = {(r, c)}
            if _nearby(cell, preserve, TARGET_CLEARANCE):
                continue
            black_n = 0
            same_n = 0
            ok = True
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                    ok = False
                    break
                if int(grid[rr, cc]) == 0:
                    black_n += 1
                elif int(grid[rr, cc]) == color:
                    same_n += 1
                else:
                    ok = False
                    break
            if ok and black_n == 1 and same_n == 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_leaf(grid: np.ndarray, region: dict, preserve: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int] | None:
    for r, c in _leaf_candidates(grid, region, preserve):
        grid[r, c] = 0
        return {(r, c)}, region["color"]
    return None


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != 0 or (r, c) in exclude:
                continue
            if _effective_black_neighbors(grid, r, c) >= 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _generate_clean_case() -> tuple[np.ndarray, list[dict], list[tuple[set[tuple[int, int]], int]]] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    regions = _build_regions(H, W)
    if regions is None or len(regions) < 2:
        return None

    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        if (r1 - r0 + 1) * (c1 - c0 + 1) < MIN_REGION_AREA:
            return None

    grid = _fill_regions(H, W, regions)
    preserve: set[tuple[int, int]] = set()
    targets: list[tuple[set[tuple[int, int]], int]] = []

    for region in regions:
        placed = _place_region_defect(grid, region, preserve)
        if placed is None:
            return None
        pixels, owner = placed
        preserve.update(pixels)
        targets.append((pixels, owner))

    extra_defects = random.randint(*EXTRA_DEFECT_RANGE)
    for _ in range(extra_defects):
        region = random.choice(regions)
        placed = _place_region_defect(grid, region, preserve)
        if placed is None:
            continue
        pixels, owner = placed
        preserve.update(pixels)
        targets.append((pixels, owner))

    for region in regions:
        if random.random() >= LEAF_PROB:
            continue
        placed = _place_leaf(grid, region, preserve)
        if placed is None:
            continue
        pixels, owner = placed
        preserve.update(pixels)
        targets.append((pixels, owner))

    halo_colors_used = {FOREGROUND_TO_HALO[owner] for _, owner in targets}
    if len(halo_colors_used) < 2:
        return None
    return grid, regions, targets


def _single_halo_output(grid: np.ndarray) -> np.ndarray:
    base = np.array(grid, dtype=int)
    for r in range(base.shape[0]):
        for c in range(base.shape[1]):
            if int(base[r, c]) in FOREGROUND_TO_HALO and _effective_black_neighbors(base, r, c) >= 3:
                base[r, c] = 0
    targets, preserve = _collect_targets(base)
    out = _paint_targets(base, targets, force_halo=7)
    for r, c in preserve:
        out[r, c] = 0
    return out


def _gen_attempt() -> np.ndarray | None:
    built = _generate_clean_case()
    if built is None:
        return None
    clean, regions, targets = built
    output = _paint_targets(clean, targets)

    preserve = set()
    for pixels, _ in targets:
        preserve.update(pixels)

    inp = clean.copy()
    present_colors = [region["color"] for region in regions]
    spur_candidates = _find_spur_candidates(inp, preserve)
    n_spurs = min(random.randint(*SPUR_COUNT_RANGE), len(spur_candidates))
    if n_spurs < 2:
        return None
    for r, c in spur_candidates[:n_spurs]:
        inp[r, c] = random.choice(present_colors)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    if np.array_equal(output, _single_halo_output(clean)):
        return None
    if len(set(int(v) for v in np.unique(inp)) & set(FOREGROUND_TO_HALO)) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
