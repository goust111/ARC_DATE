import random
from functools import lru_cache

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False
    if int(comp["area"]) != 2 * (height + width) - 4:
        return False

    mask = np.zeros((height, width), dtype=bool)
    rows = comp["coords"][:, 0] - r0
    cols = comp["coords"][:, 1] - c0
    mask[rows, cols] = True

    expected = np.zeros((height, width), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _frame_contains(outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    return ir0 >= or0 + 2 and ic0 >= oc0 + 2 and ir1 <= or1 - 2 and ic1 <= oc1 - 2


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _select_frame_chain(candidates: list[dict]) -> list[dict]:
    assert candidates, "expected at least one rectangular frame candidate"
    n = len(candidates)
    contains: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _frame_contains(candidates[i]["bbox"], candidates[j]["bbox"]):
                contains[i].append(j)

    @lru_cache(maxsize=None)
    def best_from(idx: int) -> tuple[int, ...]:
        best_chain = (idx,)
        for child in contains[idx]:
            chain = (idx,) + best_from(child)
            if len(chain) > len(best_chain):
                best_chain = chain
        return best_chain

    chains = [best_from(i) for i in range(n)]
    max_len = max(len(chain) for chain in chains)
    best_chains = [chain for chain in chains if len(chain) == max_len]
    best_idx_chain = max(best_chains, key=lambda chain: (_bbox_area(candidates[chain[0]]["bbox"]), -_bbox_area(candidates[chain[-1]]["bbox"])))
    return [candidates[i] for i in best_idx_chain]


def _detect_case(grid: np.ndarray) -> tuple[int, list[dict], tuple[int, int, int, int], np.ndarray]:
    bg = _get_bg(grid)
    comps = _find_mono_components(grid, bg)
    frame_candidates = [comp for comp in comps if _is_rect_frame(comp)]
    chain = _select_frame_chain(frame_candidates)
    assert len(chain) >= 2, "expected a nested frame chain"

    frame_mask = np.zeros(grid.shape, dtype=bool)
    for comp in chain:
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        frame_mask[rows, cols] = True

    seed_mask = (grid != bg) & (~frame_mask)
    assert np.any(seed_mask), "expected a seed pattern outside the frames"

    rows, cols = np.where(seed_mask)
    sr0 = int(rows.min())
    sc0 = int(cols.min())
    sr1 = int(rows.max())
    sc1 = int(cols.max())

    seed_patch = grid[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    local_mask = seed_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]
    seed_patch[~local_mask] = bg
    return bg, chain, (sr0, sc0, sr1, sc1), seed_patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the nested rectangular frames, rotate the separate seed pattern clockwise that many times,
    then stamp it into the innermost cavity aligned to the cavity's top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, chain, seed_bbox, seed_patch = _detect_case(g)
    n_layers = len(chain)
    rotated = np.rot90(seed_patch, -n_layers)

    r0, c0, r1, c1 = chain[-1]["bbox"]
    cavity_r = int(r0 + 1)
    cavity_c = int(c0 + 1)
    cavity_h = int(r1 - r0 - 1)
    cavity_w = int(c1 - c0 - 1)
    rh, rw = rotated.shape
    assert rh <= cavity_h and rw <= cavity_w, "rotated seed must fit inside the innermost cavity"

    sr0, sc0, sr1, sc1 = seed_bbox
    out = g.copy()
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg
    out[cavity_r : cavity_r + rh, cavity_c : cavity_c + rw] = rotated
    return out


# --------------- generator ---------------

BG = 0
COLOR_POOL = tuple(range(1, 10))
FRAME_COUNT_CHOICES = (2, 3, 4)
PATTERN_BBOX_RANGE = (2, 4)
PATTERN_AREA_RANGE = (3, 8)
FRAME_GAP_RANGE = (1, 2)
CAVITY_EXTRA_RANGE = (0, 2)
OUTER_MARGIN_RANGE = (1, 4)
OBJECT_GAP_RANGE = (2, 5)
PLACEMENT_SLACK_RANGE = (0, 4)
MAX_GRID_SIDE = 34
MAX_GEN_ATTEMPTS = 300


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()
    for dr, dc in DIRS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while int(mask.sum()) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        rr, cc = cell
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    return _crop_mask(mask)


def _rotationally_distinct(patch: np.ndarray) -> bool:
    return all(not np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3))


def _sample_seed_patch() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        height = random.randint(*PATTERN_BBOX_RANGE)
        width = random.randint(*PATTERN_BBOX_RANGE)
        max_area = min(PATTERN_AREA_RANGE[1], height * width)
        area = random.randint(PATTERN_AREA_RANGE[0], max_area)
        mask = _grow_connected_mask(height, width, area)
        if min(mask.shape) < 2:
            continue

        coords = list(zip(*np.where(mask)))
        color_count = random.randint(2, min(4, len(coords)))
        palette = random.sample(COLOR_POOL, color_count)
        values = palette[:] + [random.choice(palette) for _ in range(len(coords) - color_count)]
        random.shuffle(values)

        patch = np.zeros(mask.shape, dtype=int)
        for (r, c), color in zip(coords, values):
            patch[int(r), int(c)] = int(color)

        if _rotationally_distinct(patch):
            return patch

    raise RuntimeError("failed to sample a non-symmetric seed patch")


def _expand_outward(inner_bbox: tuple[int, int, int, int], gap: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    top_gap, left_gap, bottom_gap, right_gap = gap
    r0, c0, r1, c1 = inner_bbox
    return (r0 - top_gap - 1, c0 - left_gap - 1, r1 + bottom_gap + 1, c1 + right_gap + 1)


def _build_frame_bboxes(n_layers: int, cavity_h: int, cavity_w: int) -> list[tuple[int, int, int, int]]:
    boxes = [(0, 0, cavity_h + 1, cavity_w + 1)]
    while len(boxes) < n_layers:
        gap = tuple(random.randint(*FRAME_GAP_RANGE) for _ in range(4))
        boxes.insert(0, _expand_outward(boxes[0], gap))

    min_r = min(bbox[0] for bbox in boxes)
    min_c = min(bbox[1] for bbox in boxes)
    shifted = [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for (r0, c0, r1, c1) in boxes]
    return shifted


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _layout_boxes(struct_h: int, struct_w: int, seed_h: int, seed_w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    left_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)
    gap = random.randint(*OBJECT_GAP_RANGE)
    extra_h = random.randint(*PLACEMENT_SLACK_RANGE)
    extra_w = random.randint(*PLACEMENT_SLACK_RANGE)

    if random.random() < 0.5:
        usable_h = max(struct_h, seed_h) + extra_h
        usable_w = struct_w + seed_w + gap + extra_w
        grid_h = top_margin + usable_h + bottom_margin
        grid_w = left_margin + usable_w + right_margin
        if random.random() < 0.5:
            seed_c = left_margin + random.randint(0, extra_w)
            struct_c = seed_c + seed_w + gap
        else:
            struct_c = left_margin + random.randint(0, extra_w)
            seed_c = struct_c + struct_w + gap
        struct_r = top_margin + random.randint(0, usable_h - struct_h)
        seed_r = top_margin + random.randint(0, usable_h - seed_h)
    else:
        usable_h = struct_h + seed_h + gap + extra_h
        usable_w = max(struct_w, seed_w) + extra_w
        grid_h = top_margin + usable_h + bottom_margin
        grid_w = left_margin + usable_w + right_margin
        if random.random() < 0.5:
            seed_r = top_margin + random.randint(0, extra_h)
            struct_r = seed_r + seed_h + gap
        else:
            struct_r = top_margin + random.randint(0, extra_h)
            seed_r = struct_r + struct_h + gap
        struct_c = left_margin + random.randint(0, usable_w - struct_w)
        seed_c = left_margin + random.randint(0, usable_w - seed_w)

    return (grid_h, grid_w), (struct_r, struct_c), (seed_r, seed_c)


def _expected_from_latent(
    grid: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    rotated_patch: np.ndarray,
    cavity_top_left: tuple[int, int],
) -> np.ndarray:
    sr0, sc0, sr1, sc1 = seed_bbox
    cr, cc = cavity_top_left
    out = grid.copy()
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
    rh, rw = rotated_patch.shape
    out[cr : cr + rh, cc : cc + rw] = rotated_patch
    return out


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_layers = random.choice(FRAME_COUNT_CHOICES)
    seed_patch = _sample_seed_patch()
    seed_colors = {int(v) for v in np.unique(seed_patch) if int(v) != BG}
    frame_colors = random.sample([c for c in COLOR_POOL if c not in seed_colors], n_layers)

    rotated = np.rot90(seed_patch, -n_layers)
    cavity_h = int(rotated.shape[0] + random.randint(*CAVITY_EXTRA_RANGE))
    cavity_w = int(rotated.shape[1] + random.randint(*CAVITY_EXTRA_RANGE))
    boxes = _build_frame_bboxes(n_layers, cavity_h, cavity_w)
    struct_h = int(boxes[0][2] + 1)
    struct_w = int(boxes[0][3] + 1)
    seed_h, seed_w = map(int, seed_patch.shape)

    (grid_h, grid_w), (struct_r, struct_c), (seed_r, seed_c) = _layout_boxes(struct_h, struct_w, seed_h, seed_w)
    if max(grid_h, grid_w) > MAX_GRID_SIDE:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for bbox, color in zip(boxes, frame_colors):
        shifted = (bbox[0] + struct_r, bbox[1] + struct_c, bbox[2] + struct_r, bbox[3] + struct_c)
        _draw_frame(grid, shifted, int(color))

    grid[seed_r : seed_r + seed_h, seed_c : seed_c + seed_w] = seed_patch

    inner = boxes[-1]
    cavity_top_left = (inner[0] + struct_r + 1, inner[1] + struct_c + 1)
    seed_bbox = (seed_r, seed_c, seed_r + seed_h - 1, seed_c + seed_w - 1)
    expected = _expected_from_latent(grid, seed_bbox, rotated, cavity_top_left)

    detected_bg, chain, detected_seed_bbox, detected_patch = _detect_case(grid)
    if detected_bg != BG or len(chain) != n_layers:
        return None
    if detected_seed_bbox != seed_bbox or not np.array_equal(detected_patch, seed_patch):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        case = _make_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid nested-frame rotation task")
