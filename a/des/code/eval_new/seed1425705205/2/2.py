import random
from collections import Counter, deque

import numpy as np


N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_RETRIES = 400
MASK_RETRIES = 200
BOX_MIN = 5
BOX_MAX_FOR_2_PAIRS = 8
BOX_MAX_FOR_3_PAIRS = 7
PAIR_COUNT_OPTIONS = [2, 3]
_RNG = random.Random()


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in N8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(coords)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _bbox_and_mask(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return (r0, c0, r1 - r0 + 1, c1 - c0 + 1), mask


def _extract_fragments(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    fragments = []
    for color in sorted({int(x) for x in np.unique(grid)} - {int(bg)}):
        mask = grid == int(color)
        visited = np.zeros((h, w), dtype=bool)
        seeds = np.argwhere(mask)
        for sr, sc in seeds:
            sr, sc = int(sr), int(sc)
            if visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in N8:
                    rr, cc = r + dr, c + dc
                    if (
                        0 <= rr < h
                        and 0 <= cc < w
                        and bool(mask[rr, cc])
                        and not bool(visited[rr, cc])
                    ):
                        visited[rr, cc] = True
                        q.append((rr, cc))
            bbox, tight = _bbox_and_mask(coords)
            fragments.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "mask": tight,
                    "area": int(len(coords)),
                }
            )
    return fragments


def _complementary_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return mask_a.shape == mask_b.shape and bool(np.all(np.logical_xor(mask_a, mask_b)))


def _matching_graph(fragments: list[dict]) -> list[list[int]]:
    n = len(fragments)
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _complementary_masks(fragments[i]["mask"], fragments[j]["mask"]):
                graph[i].append(j)
                graph[j].append(i)
    return graph


def _count_perfect_matchings(graph: list[list[int]], limit: int = 2) -> int:
    n = len(graph)
    used = [False] * n

    def rec() -> int:
        try:
            i = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            return 1
        used[i] = True
        total = 0
        for j in graph[i]:
            if used[j]:
                continue
            used[j] = True
            total += rec()
            used[j] = False
            if total >= limit:
                break
        used[i] = False
        return total

    return rec()


def _find_one_matching(graph: list[list[int]]) -> list[tuple[int, int]]:
    n = len(graph)
    used = [False] * n
    result: list[tuple[int, int]] = []

    def rec() -> bool:
        try:
            i = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            return True
        used[i] = True
        for j in graph[i]:
            if used[j]:
                continue
            used[j] = True
            result.append((i, j))
            if rec():
                return True
            result.pop()
            used[j] = False
        used[i] = False
        return False

    ok = rec()
    assert ok, "expected a perfect matching"
    return result


def _solve(grid: np.ndarray) -> np.ndarray:
    bg = _background_color(grid)
    fragments = _extract_fragments(grid, bg)
    assert fragments and len(fragments) % 2 == 0, "grid must contain an even number of fragments"

    graph = _matching_graph(fragments)
    assert all(graph), "every fragment must have a complementary partner"
    assert _count_perfect_matchings(graph) == 1, "complementary pairing must be unique"
    pairs = _find_one_matching(graph)

    out = grid.copy()
    to_clear = []
    to_fill = []
    for i, j in pairs:
        a, b = fragments[i], fragments[j]
        assert a["bbox"][2:] == b["bbox"][2:], "paired fragments must share bbox dimensions"
        assert a["area"] != b["area"], "paired fragments must have different areas"
        bigger, smaller = (a, b) if a["area"] > b["area"] else (b, a)
        to_clear.extend(smaller["coords"])
        to_fill.append((bigger["bbox"], int(bigger["color"])))

    for r, c in to_clear:
        out[r, c] = int(bg)
    for (r0, c0, hh, ww), color in to_fill:
        out[r0 : r0 + hh, c0 : c0 + ww] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments into solid rectangles colored by the larger piece.

    Input structure:
    - The background is the most common color.
    - Several scattered single-color fragments of distinct colors appear on the grid.
    - Each valid pair has the same tight bounding-box dimensions, and within that box their occupied cells are exact complements.
    - In every pair, one fragment has strictly more pixels than the other.

    Rule:
    - Extract every fragment as a tight binary mask inside its own bounding box.
    - Match fragments whose masks have equal shape and tile the full bounding box with no overlap.
    - For each matched pair, translate the smaller fragment conceptually into the larger fragment's missing cells, then repaint the whole completed rectangle with the larger fragment's original color.
    - Keep that completed rectangle at the larger fragment's original position and erase the smaller fragment's original cells to background.

    Key insight:
    - Pairing depends on complementary occupancy inside equal-size bounding boxes, and only after that area comparison decides both the surviving location and the final color.
    """
    return _solve(np.asarray(grid, dtype=int))


def _valid_fragment_mask(mask: np.ndarray) -> bool:
    comp = ~mask
    total = int(mask.size)
    area = int(mask.sum())
    smaller = min(area, total - area)
    min_smaller = max(5, total // 5)
    return (
        area != total - area
        and smaller >= min_smaller
        and _touches_all_sides(mask)
        and _touches_all_sides(comp)
        and _mask_connected(mask)
        and _mask_connected(comp)
    )


def _base_corner_band(h: int, w: int, rng: random.Random) -> np.ndarray:
    mirror = bool(rng.randrange(2))
    c = 0 if not mirror else w - 1
    target = w - 1 if not mirror else 0
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        width = 1 + int(rng.random() < 0.45)
        for t in range(width):
            cc = c - t if mirror else c + t
            if 0 <= cc < w:
                mask[r, cc] = True
        if r == h - 1:
            break
        remaining_rows = h - 1 - r
        if remaining_rows <= abs(target - c):
            step = -1 if mirror else 1
        else:
            steps = [0]
            if c != target:
                steps.append(-1 if mirror else 1)
            if 0 < c < w - 1 and rng.random() < 0.3:
                steps.append(1 if mirror else -1)
            step = rng.choice(steps)
        c = max(0, min(w - 1, c + step))
    mask[h - 1, target] = True
    return mask


def _generate_mask_pair(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(MASK_RETRIES):
        mask = _base_corner_band(h, w, rng)
        growth_budget = rng.randint(0, max(1, (h * w) // 5))
        for _ in range(growth_budget):
            frontier = set()
            for r, c in np.argwhere(mask):
                r, c = int(r), int(c)
                for dr, dc in N4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not bool(mask[rr, cc]):
                        frontier.add((rr, cc))
            if not frontier:
                break
            candidates = list(frontier)
            rng.shuffle(candidates)
            added = False
            for r, c in candidates:
                mask[r, c] = True
                if _valid_fragment_mask(mask):
                    added = True
                    break
                mask[r, c] = False
            if not added:
                break
        if _valid_fragment_mask(mask):
            if rng.random() < 0.5:
                return mask, ~mask
            return ~mask, mask
    return None


def _choose_pair_dims(pair_count: int, rng: random.Random) -> list[tuple[int, int]]:
    box_max = BOX_MAX_FOR_3_PAIRS if pair_count == 3 else BOX_MAX_FOR_2_PAIRS
    pool = [(h, w) for h in range(BOX_MIN, box_max + 1) for w in range(BOX_MIN, box_max + 1)]
    repeat_dims = rng.random() < 0.5
    if pair_count == 2:
        if repeat_dims:
            d0 = rng.choice(pool)
            return [d0, d0]
        return rng.sample(pool, 2)

    if repeat_dims:
        d0 = rng.choice(pool)
        d1 = rng.choice([d for d in pool if d != d0])
        dims = [d0, d0, d1]
        rng.shuffle(dims)
        return dims
    return rng.sample(pool, 3)


def _mask_matching_count(masks: list[np.ndarray]) -> int:
    fragments = [{"mask": mask} for mask in masks]
    return _count_perfect_matchings(_matching_graph(fragments), limit=3)


def _layout_objects(objects: list[dict], rng: random.Random) -> tuple[int, int, list[tuple[int, int]]] | None:
    layouts = [(2, 2)] if len(objects) == 4 else [(2, 3), (3, 2)]
    rng.shuffle(layouts)
    for rows, cols in layouts:
        slots = list(range(rows * cols))
        rng.shuffle(slots)
        slots = slots[: len(objects)]
        row_heights = [0] * rows
        col_widths = [0] * cols
        for obj, slot in zip(objects, slots):
            rr, cc = divmod(slot, cols)
            h, w = obj["mask"].shape
            row_heights[rr] = max(row_heights[rr], int(h))
            col_widths[cc] = max(col_widths[cc], int(w))

        gap = 2
        margin = 1
        h_total = margin * 2 + sum(row_heights) + gap * (rows - 1)
        w_total = margin * 2 + sum(col_widths) + gap * (cols - 1)
        if h_total > 30 or w_total > 30:
            continue

        row_starts = []
        cur = margin
        for height in row_heights:
            row_starts.append(cur)
            cur += height + gap

        col_starts = []
        cur = margin
        for width in col_widths:
            col_starts.append(cur)
            cur += width + gap

        positions = []
        for obj, slot in zip(objects, slots):
            rr, cc = divmod(slot, cols)
            h, w = obj["mask"].shape
            r0 = row_starts[rr] + rng.randint(0, row_heights[rr] - int(h))
            c0 = col_starts[cc] + rng.randint(0, col_widths[cc] - int(w))
            positions.append((r0, c0))
        return int(h_total), int(w_total), positions
    return None


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_OPTIONS)
    dims = _choose_pair_dims(pair_count, rng)

    pair_masks = []
    for h, w in dims:
        masks = _generate_mask_pair(int(h), int(w), rng)
        if masks is None:
            return None
        pair_masks.extend(masks)

    if _mask_matching_count(pair_masks) != 1:
        return None

    bg = rng.randrange(10)
    colors = rng.sample([c for c in range(10) if c != bg], len(pair_masks))
    objects = [{"mask": mask, "color": color} for mask, color in zip(pair_masks, colors)]
    rng.shuffle(objects)

    layout = _layout_objects(objects, rng)
    if layout is None:
        return None
    h_total, w_total, positions = layout

    grid = np.full((h_total, w_total), int(bg), dtype=int)
    for obj, (r0, c0) in zip(objects, positions):
        h, w = obj["mask"].shape
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[obj["mask"]] = int(obj["color"])

    counts = Counter(map(int, grid.ravel()))
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    fragments = _extract_fragments(grid, bg)
    if len(fragments) != len(objects):
        return None

    try:
        _solve(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
