import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
AREA_TO_DIMS = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
}
SEED_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)
FRAME_COLOR = 7
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = DIR4
    else:
        steps = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in steps:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _outline_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_rectangle_frame(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return False
    expected = _outline_cells(r0, c0, r1, c1)
    return set(coords) == expected


def _find_frame_color(grid: np.ndarray) -> int:
    candidates: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        rect_count = 0
        rect_pixels = 0
        for comp in _components(grid == color, connectivity=4):
            if _is_rectangle_frame(comp):
                rect_count += 1
                rect_pixels += len(comp)
        if rect_count:
            candidates.append((rect_count, rect_pixels, color))
    if not candidates:
        raise ValueError("no rectangle frame color found")
    candidates.sort()
    return int(candidates[-1][2])


def _detect_checkerboard_pair(grid: np.ndarray, frame_color: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(grid) if int(c) != int(frame_color)]
    rr, cc = np.indices(grid.shape)
    parity = (rr + cc) & 1
    usable = grid != int(frame_color)
    best: tuple[int, int, int] | None = None
    for even_color in colors:
        for odd_color in colors:
            if even_color == odd_color:
                continue
            match = usable & (
                ((parity == 0) & (grid == int(even_color)))
                | ((parity == 1) & (grid == int(odd_color)))
            )
            score = int(np.count_nonzero(match))
            cand = (score, int(even_color), int(odd_color))
            if best is None or cand > best:
                best = cand
    if best is None:
        raise ValueError("no checkerboard colors found")
    return int(best[1]), int(best[2])


def _paint_checkerboard(shape: tuple[int, int], bg_pair: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    parity = (rr + cc) & 1
    out = np.full((h, w), int(bg_pair[0]), dtype=int)
    out[parity == 1] = int(bg_pair[1])
    return out


def _extract_regions(
    grid: np.ndarray, frame_color: int, bg_pair: tuple[int, int]
) -> list[dict]:
    bg_colors = {int(bg_pair[0]), int(bg_pair[1])}
    regions: list[dict] = []
    for comp in _components(grid == int(frame_color), connectivity=4):
        if not _is_rectangle_frame(comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h <= 0 or inner_w <= 0:
            continue
        seed_cells = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) not in bg_colors and int(grid[r, c]) != int(frame_color)
        ]
        if not seed_cells:
            continue
        inner_mask = np.zeros((inner_h, inner_w), dtype=bool)
        for r, c in seed_cells:
            inner_mask[r - (r0 + 1), c - (c0 + 1)] = True
        comps = _components(inner_mask, connectivity=4)
        if len(comps) != 1:
            continue
        seed_colors = {int(grid[r, c]) for r, c in seed_cells}
        if len(seed_colors) != 1:
            continue
        regions.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(inner_h * inner_w),
                "seed": [(int(r), int(c)) for r, c in seed_cells],
                "color": int(next(iter(seed_colors))),
            }
        )
    regions.sort(key=lambda reg: (reg["bbox"][0], reg["bbox"][1]))
    return regions


def _expanded_positions(
    seed_cells: list[tuple[int, int]], area: int, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    h, w = shape
    claimed = {(int(r), int(c)) for r, c in seed_cells}
    for dr, dc in DIR4:
        for step in range(1, int(area) + 1):
            for r, c in seed_cells:
                rr = int(r + dr * step)
                cc = int(c + dc * step)
                if 0 <= rr < h and 0 <= cc < w:
                    claimed.add((rr, cc))
    return claimed


def _collect_claims(
    regions: list[dict], shape: tuple[int, int], frame_mask: np.ndarray
) -> tuple[dict[tuple[int, int], list[tuple[int, int, int]]], list[set[tuple[int, int]]]]:
    claims: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    per_region: list[set[tuple[int, int]]] = []
    for idx, region in enumerate(regions):
        positions = {
            pos
            for pos in _expanded_positions(region["seed"], int(region["area"]), shape)
            if not frame_mask[pos]
        }
        per_region.append(positions)
        for pos in positions:
            claims[pos].append((int(region["area"]), int(idx), int(region["color"])))
    return claims, per_region


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame_color = _find_frame_color(g)
    bg_pair = _detect_checkerboard_pair(g, frame_color)
    regions = _extract_regions(g, frame_color, bg_pair)
    if not regions:
        return g.copy()

    out = _paint_checkerboard(g.shape, bg_pair)
    frame_mask = g == int(frame_color)
    out[frame_mask] = int(frame_color)

    claims, _ = _collect_claims(regions, g.shape, frame_mask)
    for (r, c), contenders in claims.items():
        area, neg_idx, color = max(
            (area, -idx, color) for area, idx, color in contenders
        )
        out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def _choose_shape(
    inner_h: int, inner_w: int, force_single: bool = False
) -> tuple[tuple[int, int], ...]:
    if force_single:
        return SEED_SHAPES[0]
    fitting = []
    for shape in SEED_SHAPES:
        sh, sw = _shape_dims(shape)
        if sh <= inner_h and sw <= inner_w:
            fitting.append(shape)
    if not fitting:
        return SEED_SHAPES[0]
    non_single = [shape for shape in fitting if len(shape) > 1]
    if non_single and random.random() < 0.7:
        return random.choice(non_single)
    return random.choice(fitting)


def _rect_too_close(
    rect: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    margin: int = 1,
) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if not (
            r1 < or0 - margin
            or or1 < r0 - margin
            or c1 < oc0 - margin
            or oc1 < c0 - margin
        ):
            return True
    return False


def _region_from_seed(
    seed_r: int,
    seed_c: int,
    inner_h: int,
    inner_w: int,
    offset_r: int,
    offset_c: int,
    color: int,
) -> dict:
    r0 = int(seed_r - 1 - offset_r)
    c0 = int(seed_c - 1 - offset_c)
    r1 = int(r0 + inner_h + 1)
    c1 = int(c0 + inner_w + 1)
    return {
        "bbox": (r0, c0, r1, c1),
        "area": int(inner_h * inner_w),
        "seed": [(int(seed_r), int(seed_c))],
        "color": int(color),
    }


def _place_conflict_pair(
    mode: str,
    grid_h: int,
    grid_w: int,
    left_spec: tuple[int, int, int],
    right_spec: tuple[int, int, int],
    left_color: int,
    right_color: int,
) -> list[dict] | None:
    ih1, iw1, area1 = map(int, left_spec)
    ih2, iw2, area2 = map(int, right_spec)
    for _ in range(300):
        off_r1 = random.randrange(ih1)
        off_c1 = random.randrange(iw1)
        off_r2 = random.randrange(ih2)
        off_c2 = random.randrange(iw2)
        if mode == "horizontal":
            row_lo = max(1 + off_r1, 1 + off_r2)
            row_hi = min(grid_h - 1 - ih1 + off_r1, grid_h - 1 - ih2 + off_r2)
            if row_lo > row_hi:
                continue
            seed_r = random.randint(row_lo, row_hi)
            gap_lo = max(area1, area2) + 1
            gap_hi = area1 + area2
            if gap_lo > gap_hi:
                continue
            gap = random.randint(gap_lo, gap_hi)
            c1_lo = max(1 + off_c1, 1 + off_c2 - gap)
            c1_hi = min(grid_w - 1 - iw1 + off_c1, grid_w - 1 - iw2 + off_c2 - gap)
            if c1_lo > c1_hi:
                continue
            seed_c1 = random.randint(c1_lo, c1_hi)
            seed_c2 = seed_c1 + gap
            region1 = _region_from_seed(seed_r, seed_c1, ih1, iw1, off_r1, off_c1, left_color)
            region2 = _region_from_seed(seed_r, seed_c2, ih2, iw2, off_r2, off_c2, right_color)
        else:
            col_lo = max(1 + off_c1, 1 + off_c2)
            col_hi = min(grid_w - 1 - iw1 + off_c1, grid_w - 1 - iw2 + off_c2)
            if col_lo > col_hi:
                continue
            seed_c = random.randint(col_lo, col_hi)
            gap_lo = max(area1, area2) + 1
            gap_hi = area1 + area2
            if gap_lo > gap_hi:
                continue
            gap = random.randint(gap_lo, gap_hi)
            r1_lo = max(1 + off_r1, 1 + off_r2 - gap)
            r1_hi = min(grid_h - 1 - ih1 + off_r1, grid_h - 1 - ih2 + off_r2 - gap)
            if r1_lo > r1_hi:
                continue
            seed_r1 = random.randint(r1_lo, r1_hi)
            seed_r2 = seed_r1 + gap
            region1 = _region_from_seed(seed_r1, seed_c, ih1, iw1, off_r1, off_c1, left_color)
            region2 = _region_from_seed(seed_r2, seed_c, ih2, iw2, off_r2, off_c2, right_color)
        boxes = [region1["bbox"]]
        if _rect_too_close(region2["bbox"], boxes, margin=1):
            continue
        return [region1, region2]
    return None


def _place_random_region(
    grid_h: int,
    grid_w: int,
    inner_h: int,
    inner_w: int,
    color: int,
    boxes: list[tuple[int, int, int, int]],
) -> dict | None:
    shape = _choose_shape(inner_h, inner_w, force_single=False)
    sh, sw = _shape_dims(shape)
    for _ in range(300):
        r0 = random.randint(0, grid_h - (inner_h + 2))
        c0 = random.randint(0, grid_w - (inner_w + 2))
        rect = (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1)
        if _rect_too_close(rect, boxes, margin=1):
            continue
        sr = random.randint(0, inner_h - sh)
        sc = random.randint(0, inner_w - sw)
        seed = [
            (int(r0 + 1 + sr + dr), int(c0 + 1 + sc + dc))
            for dr, dc in shape
        ]
        return {
            "bbox": rect,
            "area": int(inner_h * inner_w),
            "seed": seed,
            "color": int(color),
        }
    return None


def _render_input(
    shape: tuple[int, int], bg_pair: tuple[int, int], regions: list[dict]
) -> np.ndarray:
    grid = _paint_checkerboard(shape, bg_pair)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        for r, c in _outline_cells(r0, c0, r1, c1):
            grid[r, c] = FRAME_COLOR
        for r, c in region["seed"]:
            grid[r, c] = int(region["color"])
    return grid


def _frame_mask_from_regions(shape: tuple[int, int], regions: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        for r, c in _outline_cells(r0, c0, r1, c1):
            mask[r, c] = True
    return mask


def _valid_regions(regions: list[dict], shape: tuple[int, int], mode: str) -> bool:
    frame_mask = _frame_mask_from_regions(shape, regions)
    claims, _ = _collect_claims(regions, shape, frame_mask)
    overlap_pixels = [pos for pos, vals in claims.items() if len(vals) > 1]
    seed_cells = {cell for region in regions for cell in region["seed"]}
    if any(pos in seed_cells for pos in overlap_pixels):
        return False
    priority_overlap = any(
        len({area for area, _idx, _color in vals}) > 1
        for vals in claims.values()
        if len(vals) > 1
    )
    if mode == "none":
        return not overlap_pixels
    if not priority_overlap:
        return False
    return len(overlap_pixels) <= 20


def _random_spec(area: int | None = None) -> tuple[int, int, int]:
    if area is None:
        area = random.choice(sorted(AREA_TO_DIMS))
    inner_h, inner_w = random.choice(AREA_TO_DIMS[int(area)])
    return int(inner_h), int(inner_w), int(area)


def _gen_attempt(mode: str) -> np.ndarray | None:
    n_regions = random.randint(2, 4)
    if mode == "none":
        grid_h = random.randint(17, 24)
        grid_w = random.randint(17, 24)
    else:
        grid_h = random.randint(18, 24)
        grid_w = random.randint(18, 24)

    palette = [c for c in range(10) if c != FRAME_COLOR]
    bg_pair = tuple(random.sample(palette, 2))
    seed_palette = [c for c in palette if c not in bg_pair]
    if len(seed_palette) < n_regions:
        return None
    seed_colors = random.sample(seed_palette, n_regions)

    regions: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []

    if mode in {"horizontal", "vertical"}:
        primary_areas = random.sample([4, 6, 8, 9], 2)
        pair = _place_conflict_pair(
            mode,
            grid_h,
            grid_w,
            _random_spec(primary_areas[0]),
            _random_spec(primary_areas[1]),
            seed_colors[0],
            seed_colors[1],
        )
        if pair is None:
            return None
        regions.extend(pair)
        boxes.extend(region["bbox"] for region in pair)
        start_idx = 2
    else:
        start_idx = 0

    while len(regions) < n_regions:
        spec = _random_spec()
        region = _place_random_region(
            grid_h, grid_w, spec[0], spec[1], seed_colors[start_idx], boxes
        )
        if region is None:
            return None
        regions.append(region)
        boxes.append(region["bbox"])
        start_idx += 1

    if not _valid_regions(regions, (grid_h, grid_w), mode):
        return None

    grid = _render_input((grid_h, grid_w), bg_pair, regions)

    parsed = _extract_regions(grid, FRAME_COLOR, bg_pair)
    if len(parsed) != len(regions):
        return None
    if sorted(region["area"] for region in parsed) != sorted(region["area"] for region in regions):
        return None
    if not _valid_regions(parsed, grid.shape, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    mode = random.choice(["horizontal", "vertical", "none"])
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
