import random
from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
REGION_BORDER = 5
TEMPLATE_FILL = 9
TEMPLATE_BORDER_POOL = (1, 2, 3, 4, 6, 7, 8)
BASE_SLOT_SHAPE = (2, 3)
SLOT_STRIDE = 3
BLOCK_SIZE = 2
MIN_PAIRS = 2
MAX_PAIRS = 3
PLACEMENT_PAD = 1
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    pts = set(coords)
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return pts == expected


def _frame_records(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            if not _is_rect_frame(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = set(int(v) for v in np.unique(inner) if int(v) != bg)
            frames.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "inner_mask": inner != bg,
                    "inner_colors": inner_colors,
                }
            )
    frames.sort(key=lambda rec: rec["bbox"])
    return frames


def _rotation_to_match(target: np.ndarray, source: np.ndarray) -> int | None:
    for k in range(4):
        rotated = np.rot90(source, k)
        if rotated.shape == target.shape and np.array_equal(rotated, target):
            return k
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey-framed region by matching its interior block layout to a rotated external template.

    Input structure:
    - The background is black.
    - Several grey 1-cell-thick rectangular frames enclose target regions whose interiors contain 2 or 3 colored 2x2 blocks.
    - Separate framed templates sit on the grid edge or in corners; each template has a unique border color and contains the same kind of 2x2 blocks, all in one template color.

    Rule:
    - Detect every monochrome rectangular frame and split them into templates versus target regions using the interior colors: templates use one interior color, target regions use multiple interior colors.
    - For each target region, ignore block colors and compare only the occupied interior cells against every template interior under 0, 90, 180, and 270 degree rotation.
    - Find the unique template whose rotated occupancy mask matches exactly, then repaint the target region's grey border with that template's border color.
    - Leave all interior blocks and all template objects unchanged.

    Key insight:
    - The only transferable attribute is the template border color; the match is determined purely by the topology of occupied cells inside the frame, up to rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frames = _frame_records(g, bg)
    templates = [rec for rec in frames if rec["inner_mask"].any() and len(rec["inner_colors"]) == 1]
    regions = [rec for rec in frames if rec["inner_mask"].any() and len(rec["inner_colors"]) >= 2]

    for region in regions:
        matches = [
            template
            for template in templates
            if _rotation_to_match(region["inner_mask"], template["inner_mask"]) is not None
        ]
        assert len(matches) == 1, "each region must match exactly one template"
        for r, c in region["coords"]:
            out[r, c] = matches[0]["color"]

    return out


def _mask_key(mask: np.ndarray) -> tuple:
    return mask.shape, tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _canonical_pattern(mask: np.ndarray) -> tuple:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _all_base_patterns(block_count: int) -> list[np.ndarray]:
    rows, cols = BASE_SLOT_SHAPE
    patterns: list[np.ndarray] = []
    for combo in combinations(range(rows * cols), block_count):
        mask = np.zeros((rows, cols), dtype=bool)
        for idx in combo:
            r, c = divmod(idx, cols)
            mask[r, c] = True
        patterns.append(mask)
    return patterns


BASE_PATTERNS = {
    2: _all_base_patterns(2),
    3: _all_base_patterns(3),
}


def _choose_patterns(rng: random.Random, pair_count: int) -> list[np.ndarray]:
    used: set[tuple] = set()
    chosen: list[np.ndarray] = []
    for _ in range(pair_count):
        block_count = rng.choice((2, 3))
        candidates = [
            pattern
            for pattern in BASE_PATTERNS[block_count]
            if _canonical_pattern(pattern) not in used
        ]
        if not candidates:
            return []
        pattern = candidates[rng.randrange(len(candidates))].copy()
        used.add(_canonical_pattern(pattern))
        chosen.append(pattern)
    return chosen


def _slot_mask_to_pixel_mask(slot_mask: np.ndarray) -> np.ndarray:
    rows, cols = slot_mask.shape
    inner = np.zeros((SLOT_STRIDE * rows - 1, SLOT_STRIDE * cols - 1), dtype=bool)
    for r, c in np.argwhere(slot_mask):
        rr = SLOT_STRIDE * int(r)
        cc = SLOT_STRIDE * int(c)
        inner[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = True
    return inner


def _render_object(slot_mask: np.ndarray, border_color: int, block_colors: list[int]) -> np.ndarray:
    inner_mask = _slot_mask_to_pixel_mask(slot_mask)
    h, w = inner_mask.shape
    obj = np.full((h + 2, w + 2), BG, dtype=int)
    obj[0, :] = border_color
    obj[-1, :] = border_color
    obj[:, 0] = border_color
    obj[:, -1] = border_color

    slots = [tuple(map(int, rc)) for rc in np.argwhere(slot_mask)]
    assert len(slots) == len(block_colors)
    for (r, c), color in zip(slots, block_colors):
        rr = 1 + SLOT_STRIDE * r
        cc = 1 + SLOT_STRIDE * c
        obj[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = int(color)
    return obj


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    r0 = max(0, top - PLACEMENT_PAD)
    c0 = max(0, left - PLACEMENT_PAD)
    r1 = min(H, top + h + PLACEMENT_PAD)
    c1 = min(W, left + w + PLACEMENT_PAD)
    return not occupied[r0:r1, c0:c1].any()


def _place_object(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    obj: np.ndarray,
    edge_only: bool,
) -> tuple[int, int] | None:
    H, W = grid.shape
    h, w = obj.shape
    candidates: list[tuple[int, int]] = []
    if edge_only:
        seen = set()
        for c in range(W - w + 1):
            seen.add((0, c))
            seen.add((H - h, c))
        for r in range(H - h + 1):
            seen.add((r, 0))
            seen.add((r, W - w))
        candidates = list(seen)
    else:
        for r in range(1, H - h):
            for c in range(1, W - w):
                candidates.append((r, c))
    rng.shuffle(candidates)
    for top, left in candidates:
        if _can_place(occupied, top, left, h, w):
            grid[top : top + h, left : left + w] = obj
            occupied[top : top + h, left : left + w] = True
            return top, left
    return None


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(MIN_PAIRS, MAX_PAIRS)
    patterns = _choose_patterns(rng, pair_count)
    if len(patterns) != pair_count:
        return None

    border_colors = rng.sample(TEMPLATE_BORDER_POOL, pair_count)
    region_color_pool = [color for color in TEMPLATE_BORDER_POOL if color not in border_colors]

    object_specs: list[tuple[str, np.ndarray]] = []
    for idx, base_pattern in enumerate(patterns):
        template_turn = rng.randrange(4)
        relative_turn = rng.randrange(4)
        region_turn = (template_turn + relative_turn) % 4

        template_mask = np.rot90(base_pattern, template_turn)
        region_mask = np.rot90(base_pattern, region_turn)
        block_count = int(template_mask.sum())

        region_colors = rng.sample(region_color_pool, block_count)
        rng.shuffle(region_colors)

        template_obj = _render_object(template_mask, border_colors[idx], [TEMPLATE_FILL] * block_count)
        region_obj = _render_object(region_mask, REGION_BORDER, region_colors)
        object_specs.append(("template", template_obj))
        object_specs.append(("region", region_obj))

    total_area = sum(obj.shape[0] * obj.shape[1] for _, obj in object_specs)
    min_side = max(22, int(np.ceil(np.sqrt(total_area * 1.7))))
    if pair_count == 3:
        min_side = max(min_side, 24)
    if min_side > 30:
        return None

    H = rng.randint(min_side, 30)
    W = rng.randint(min_side, 30)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    templates = [obj for kind, obj in object_specs if kind == "template"]
    regions = [obj for kind, obj in object_specs if kind == "region"]

    rng.shuffle(templates)
    rng.shuffle(regions)
    templates.sort(key=lambda obj: obj.shape[0] * obj.shape[1], reverse=True)
    regions.sort(key=lambda obj: obj.shape[0] * obj.shape[1], reverse=True)

    for obj in templates:
        if _place_object(rng, grid, occupied, obj, edge_only=True) is None:
            return None
    for obj in regions:
        if _place_object(rng, grid, occupied, obj, edge_only=False) is None:
            return None

    frames = _frame_records(grid, BG)
    templates_found = [rec for rec in frames if rec["inner_mask"].any() and len(rec["inner_colors"]) == 1]
    regions_found = [rec for rec in frames if rec["inner_mask"].any() and len(rec["inner_colors"]) >= 2]
    if len(templates_found) != pair_count or len(regions_found) != pair_count:
        return None

    for region in regions_found:
        match_count = sum(
            _rotation_to_match(region["inner_mask"], template["inner_mask"]) is not None
            for template in templates_found
        )
        if match_count != 1:
            return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
