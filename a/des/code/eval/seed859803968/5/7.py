import random
from collections import deque

import numpy as np


BACKGROUND = 8  # azure
CONTAINER_BORDER = 5  # grey
LEGEND_BORDER = 9  # maroon
TYPE_TO_MARKER = {"lr": 4, "ud": 3, "none": 2}  # yellow, green, red
PATTERN_COLORS = (0, 1, 6, 7)
OUTER_SIZE_CHOICES = ((5, 5), (5, 5), (5, 7), (7, 5), (7, 7))
GRID_MIN = 24
GRID_MAX = 30
LEGEND_INNER_H = 11
LEGEND_INNER_W_RANGE = (7, 9)
CONTAINER_COUNT_RANGE = (3, 5)
GEN_RETRIES = 400
MASK_RETRIES = 300
PLACEMENT_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray, exclude: set[int]) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in exclude):
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_patch(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return expected == {tuple(map(int, p)) for p in comp["coords"]}


def _is_legend_l(comp: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    if r1 != h - 1 or c1 != w - 1:
        return False
    if r1 - r0 + 1 < 8 or c1 - c0 + 1 < 6:
        return False
    expected = {(r0, c) for c in range(c0, c1 + 1)}
    expected |= {(r, c0) for r in range(r0, r1 + 1)}
    return expected == {tuple(map(int, p)) for p in comp["coords"]}


def _classify_symmetry(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return "none"
    # "lr" means left and right halves match across the vertical midline.
    if np.array_equal(mask, np.fliplr(mask)):
        return "lr"
    if np.array_equal(mask, np.flipud(mask)):
        return "ud"
    return "none"


def _extract_legend_mapping(grid: np.ndarray, bg: int) -> dict[str, int]:
    all_comps = _find_monochrome_components(grid, exclude={bg})
    legend_candidates = [comp for comp in all_comps if _is_legend_l(comp, grid.shape)]
    assert len(legend_candidates) == 1, "expected a unique legend L border"

    legend = legend_candidates[0]
    border_color = int(legend["color"])
    r0, c0, r1, c1 = legend["bbox"]
    inner = grid[r0 + 1 : r1 + 1, c0 + 1 : c1 + 1]

    inner_comps = _find_monochrome_components(inner, exclude={bg, border_color})
    patterns = [comp for comp in inner_comps if comp["area"] > 1]
    markers = [comp for comp in inner_comps if comp["area"] == 1]
    assert len(patterns) == 3 and len(markers) == 3, "legend must contain three demo patterns and three markers"

    patterns.sort(key=lambda comp: ((comp["bbox"][0] + comp["bbox"][2]) / 2.0, comp["bbox"][1]))
    markers.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    mapping: dict[str, int] = {}
    for pattern, marker in zip(patterns, markers):
        patch = _component_patch(pattern["coords"], pattern["bbox"])
        sym_type = _classify_symmetry(patch)
        mapping[sym_type] = int(marker["color"])

    assert set(mapping) == {"lr", "ud", "none"}, "legend mapping must cover all symmetry types"
    return mapping


def _find_container_bboxes(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for comp in _find_monochrome_components(grid, exclude={bg}):
        if _is_legend_l(comp, grid.shape):
            continue
        if _is_rect_frame(comp):
            bboxes.append(tuple(map(int, comp["bbox"])))
    assert bboxes, "expected at least one hollow rectangular container"
    return sorted(bboxes)


def _marker_positions(sym_type: str, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if sym_type == "lr":
        return [(mid_r, c0 - 1), (mid_r, c1 + 1)]
    if sym_type == "ud":
        return [(r0 - 1, mid_c), (r1 + 1, mid_c)]
    return [
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    marker_map = _extract_legend_mapping(g, bg)

    out = g.copy()
    h, w = g.shape
    for bbox in _find_container_bboxes(g, bg):
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        mask = (interior != bg) & (interior != g[r0, c0])
        sym_type = _classify_symmetry(mask)
        marker_color = marker_map[sym_type]
        for rr, cc in _marker_positions(sym_type, bbox):
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == bg:
                out[rr, cc] = marker_color
    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _random_bool_matrix(rng: random.Random, h: int, w: int, p: float) -> np.ndarray:
    return np.array([[rng.random() < p for _ in range(w)] for _ in range(h)], dtype=bool)


def _generate_lr_mask(rng: random.Random, h: int, w: int, pure: bool) -> np.ndarray:
    total = h * w
    min_area = max(2, total // 5)
    half_w = (w + 1) // 2
    for _ in range(MASK_RETRIES):
        base = _random_bool_matrix(rng, h, half_w, rng.uniform(0.28, 0.60))
        if not np.any(base[:, -1]):
            base[rng.randrange(h), -1] = True
        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(half_w):
                if base[r, c]:
                    mask[r, c] = True
                    mask[r, w - 1 - c] = True
        area = int(mask.sum())
        if area < min_area or area >= total:
            continue
        if not _mask_is_connected(mask):
            continue
        if _classify_symmetry(mask) != "lr":
            continue
        if pure and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate left-right symmetric mask")


def _generate_ud_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    total = h * w
    min_area = max(2, total // 5)
    half_h = (h + 1) // 2
    for _ in range(MASK_RETRIES):
        base = _random_bool_matrix(rng, half_h, w, rng.uniform(0.28, 0.60))
        if not np.any(base[-1, :]):
            base[-1, rng.randrange(w)] = True
        mask = np.zeros((h, w), dtype=bool)
        for r in range(half_h):
            for c in range(w):
                if base[r, c]:
                    mask[r, c] = True
                    mask[h - 1 - r, c] = True
        area = int(mask.sum())
        if area < min_area or area >= total:
            continue
        if not _mask_is_connected(mask):
            continue
        if _classify_symmetry(mask) == "ud":
            return mask
    raise RuntimeError("failed to generate up-down symmetric mask")


def _generate_none_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    total = h * w
    min_area = max(2, total // 5)
    for _ in range(MASK_RETRIES):
        mask = _random_bool_matrix(rng, h, w, rng.uniform(0.25, 0.55))
        area = int(mask.sum())
        if area < min_area or area >= total:
            continue
        if not _mask_is_connected(mask):
            continue
        if _classify_symmetry(mask) == "none":
            return mask
    raise RuntimeError("failed to generate asymmetric mask")


def _random_pattern_mask(rng: random.Random, h: int, w: int, sym_type: str, pure: bool = False) -> np.ndarray:
    if sym_type == "lr":
        return _generate_lr_mask(rng, h, w, pure=pure)
    if sym_type == "ud":
        return _generate_ud_mask(rng, h, w)
    return _generate_none_mask(rng, h, w)


def _random_legend_pattern(rng: random.Random, sym_type: str) -> np.ndarray:
    for _ in range(MASK_RETRIES):
        mask = _random_pattern_mask(rng, 3, 3, sym_type, pure=(sym_type == "lr"))
        coords = np.argwhere(mask)
        tight = _component_patch(coords, _bbox(coords))
        if _classify_symmetry(tight) == sym_type:
            return mask
    raise RuntimeError("failed to generate a stable legend demo pattern")


def _paint_container(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, mask: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = CONTAINER_BORDER
    grid[r1, c0 : c1 + 1] = CONTAINER_BORDER
    grid[r0 : r1 + 1, c0] = CONTAINER_BORDER
    grid[r0 : r1 + 1, c1] = CONTAINER_BORDER
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[mask] = color


def _draw_legend(grid: np.ndarray, rng: random.Random, top: int, left: int, inner_w: int) -> None:
    h, w = grid.shape
    grid[top, left:w] = LEGEND_BORDER
    grid[top:h, left] = LEGEND_BORDER

    row_starts = (0, 4, 8)
    demo_colors = list(PATTERN_COLORS)
    rng.shuffle(demo_colors)
    for idx, sym_type in enumerate(("lr", "ud", "none")):
        row0 = top + 1 + row_starts[idx]
        pattern_left = rng.randint(0, inner_w - 5)
        marker_col = rng.randint(pattern_left + 4, inner_w - 1)
        demo_color = demo_colors[idx % len(demo_colors)]
        mask = _random_legend_pattern(rng, sym_type)
        for dr in range(3):
            for dc in range(3):
                if mask[dr, dc]:
                    grid[row0 + dr, left + 1 + pattern_left + dc] = demo_color
        grid[row0 + 1, left + 1 + marker_col] = TYPE_TO_MARKER[sym_type]


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _render_expected(grid: np.ndarray, containers: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in containers:
        for r, c in _marker_positions(spec["type"], spec["bbox"]):
            out[r, c] = TYPE_TO_MARKER[spec["type"]]
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    inner_w = rng.randint(*LEGEND_INNER_W_RANGE)
    legend_h = LEGEND_INNER_H + 1
    legend_w = inner_w + 1
    legend_top = h - legend_h
    legend_left = w - legend_w

    grid = np.full((h, w), BACKGROUND, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    _reserve(occupied, max(0, legend_top - 1), max(0, legend_left - 1), h - 1, w - 1)

    count_lo, count_hi = CONTAINER_COUNT_RANGE
    if min(h, w) <= 25:
        count_hi = min(count_hi, 4)
    if min(h, w) <= 24:
        count_hi = min(count_hi, 3)
    container_count = rng.randint(count_lo, count_hi)
    types = ["lr", "ud", "none"]
    while len(types) < container_count:
        types.append(rng.choice(("lr", "ud", "none")))
    rng.shuffle(types)

    colors = list(PATTERN_COLORS)
    rng.shuffle(colors)
    containers: list[dict] = []

    specs = []
    for idx, sym_type in enumerate(types):
        oh, ow = rng.choice(OUTER_SIZE_CHOICES)
        color = colors[idx % len(colors)]
        specs.append((oh, ow, sym_type, color))
    specs.sort(key=lambda item: item[0] * item[1], reverse=True)

    for oh, ow, sym_type, color in specs:
        placed = False
        if h - oh - 2 < 1 or w - ow - 2 < 1:
            raise RuntimeError("grid too small for requested container")
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, h - oh - 2)
            c0 = rng.randint(1, w - ow - 2)
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            rr0, cc0 = r0 - 1, c0 - 1
            rr1, cc1 = r1 + 1, c1 + 1
            if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue
            pure = sym_type == "lr" and rng.random() < 0.6
            mask = _random_pattern_mask(rng, oh - 2, ow - 2, sym_type, pure=pure)
            containers.append({"bbox": (r0, c0, r1, c1), "type": sym_type, "color": color, "mask": mask})
            _reserve(occupied, rr0, cc0, rr1, cc1)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place all containers")

    for spec in containers:
        _paint_container(grid, spec["bbox"], spec["color"], spec["mask"])
    _draw_legend(grid, rng, legend_top, legend_left, inner_w)

    expected = _render_expected(grid, containers)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid, _ = _generate_case(rng)
        except RuntimeError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
