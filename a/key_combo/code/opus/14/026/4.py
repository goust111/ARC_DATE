from __future__ import annotations

from collections import deque
import random

import numpy as np


RECT_COLORS = (1, 2, 3, 4)
NOISE_COLORS = (5, 6, 7, 8, 9)
DEPTH_TO_COLOR = {0: 1, 1: 2, 2: 3, 3: 4}

GRID_ROWS = (12, 16)
GRID_COLS = (12, 16)
RECT_COUNT_OPTIONS = (3, 4)
RECT_HEIGHTS = (3, 4, 5, 6)
RECT_WIDTHS = (4, 5, 6, 7)
NOISE_COUNT = (2, 6)
GEN_ATTEMPTS = 600


def _bbox_of_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), ((1, 0), (1, 0)))
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _overlap_box(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _rect_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _cells_in_bbox(bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _find_runs(mask_1d: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask_1d.tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                runs.append((start, idx - 1))
            start = None
    return runs


def _full_visible_sides(
    bbox: tuple[int, int, int, int], occupied_above: np.ndarray
) -> list[str]:
    r0, c0, r1, c1 = bbox
    sides = []
    if not bool(occupied_above[r0, c0 : c1 + 1].any()):
        sides.append("top")
    if not bool(occupied_above[r1, c0 : c1 + 1].any()):
        sides.append("bottom")
    if not bool(occupied_above[r0 : r1 + 1, c0].any()):
        sides.append("left")
    if not bool(occupied_above[r0 : r1 + 1, c1].any()):
        sides.append("right")
    return sides


def _visible_mask_for_rect(
    current_grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros_like(current_grid, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = current_grid[r0 : r1 + 1, c0 : c1 + 1] == color
    return mask


def _enumerate_candidates(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int, int]]:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return []

    h, w = grid.shape
    min_r, min_c, max_r, max_c = _bbox_of_coords(coords)
    nonblack_prefix = _prefix_sum(grid != 0)
    candidates: set[tuple[int, int, int, int, int]] = set()

    def area_is_full(r0: int, c0: int, r1: int, c1: int) -> bool:
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        return _rect_sum(nonblack_prefix, r0, c0, r1, c1) == area

    for r in range(h):
        for c0, c1 in _find_runs(grid[r] == color, min_len=4):
            if min_c < c0 or max_c > c1:
                continue
            if r <= min_r:
                for r1 in range(max(max_r, r + 2), h):
                    if area_is_full(r, c0, r1, c1):
                        candidates.add((color, r, c0, r1, c1))
            if r >= max_r:
                for r0 in range(0, min(min_r, r - 2) + 1):
                    if area_is_full(r0, c0, r, c1):
                        candidates.add((color, r0, c0, r, c1))

    for c in range(w):
        for r0, r1 in _find_runs(grid[:, c] == color, min_len=3):
            if min_r < r0 or max_r > r1:
                continue
            if c <= min_c:
                for c1 in range(max(max_c, c + 3), w):
                    if area_is_full(r0, c, r1, c1):
                        candidates.add((color, r0, c, r1, c1))
            if c >= max_c:
                for c0 in range(0, min(min_c, c - 3) + 1):
                    if area_is_full(r0, c0, r1, c):
                        candidates.add((color, r0, c0, r1, c))

    return sorted(candidates)


def _solution_key(rects: list[dict], depths: list[int]) -> tuple[tuple[int, int, int, int, int, int], ...]:
    items = []
    for rect, depth in zip(rects, depths):
        r0, c0, r1, c1 = rect["bbox"]
        items.append((rect["color"], r0, c0, r1, c1, depth))
    return tuple(sorted(items))


def _topological_order(edges: set[tuple[int, int]], n: int) -> tuple[list[int], np.ndarray] | None:
    reach = np.zeros((n, n), dtype=bool)
    indeg = [0] * n
    children = [[] for _ in range(n)]
    for i, j in edges:
        if reach[j, i]:
            return None
        children[i].append(j)
        indeg[j] += 1
        reach[i, j] = True

    order: list[int] = []
    queue = deque(sorted(i for i in range(n) if indeg[i] == 0))
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in children[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if len(order) != n:
        return None

    for k in range(n):
        for i in range(n):
            if not reach[i, k]:
                continue
            for j in range(n):
                if reach[k, j]:
                    reach[i, j] = True
    return order, reach


def _validate_rects(
    grid: np.ndarray, rects: list[dict]
) -> tuple[list[int], list[int], tuple[tuple[int, int, int, int, int, int], ...]] | None:
    n = len(rects)
    edges: set[tuple[int, int]] = set()
    overlaps: list[set[int]] = [set() for _ in range(n)]
    rect_colors = {rect["color"] for rect in rects}

    for i in range(n):
        for j in range(i + 1, n):
            box = _overlap_box(rects[i]["bbox"], rects[j]["bbox"])
            if box is None:
                continue
            overlaps[i].add(j)
            overlaps[j].add(i)
            seen_i = False
            seen_j = False
            r0, c0, r1, c1 = box
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    value = int(grid[r, c])
                    if value == rects[i]["color"]:
                        seen_i = True
                    elif value == rects[j]["color"]:
                        seen_j = True
            if seen_i and seen_j:
                return None
            if not seen_i and not seen_j:
                return None
            if seen_i:
                edges.add((i, j))
            else:
                edges.add((j, i))

    topo = _topological_order(edges, n)
    if topo is None:
        return None
    order, reach = topo

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            covering = [idx for idx, rect in enumerate(rects) if rect["bbox"][0] <= r <= rect["bbox"][2] and rect["bbox"][1] <= c <= rect["bbox"][3]]
            value = int(grid[r, c])
            if not covering:
                if value in rect_colors:
                    return None
                continue
            if value == 0 or value not in rect_colors:
                return None
            top_candidates = []
            for idx in covering:
                if not any(reach[other, idx] for other in covering if other != idx):
                    top_candidates.append(idx)
            if len(top_candidates) != 1:
                return None
            if rects[top_candidates[0]]["color"] != value:
                return None

    depths = []
    for idx in range(n):
        depth = sum(1 for other in overlaps[idx] if reach[other, idx])
        if depth not in DEPTH_TO_COLOR:
            return None
        depths.append(depth)
    return order, depths, _solution_key(rects, depths)


def _render_solution(shape: tuple[int, int], rects: list[dict], order: list[int], depths: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx in reversed(order):
        r0, c0, r1, c1 = rects[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = DEPTH_TO_COLOR[depths[idx]]
    return out


def _solve_all(grid: np.ndarray, max_solutions: int = 3) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [color for color in RECT_COLORS if np.any(g == color)]
    if len(colors) not in (3, 4):
        return []

    candidate_map = {color: _enumerate_candidates(g, color) for color in colors}
    if any(not candidate_map[color] for color in colors):
        return []

    ordered_colors = sorted(colors, key=lambda color: len(candidate_map[color]))
    solutions: list[dict] = []
    seen_keys: set[tuple[tuple[int, int, int, int, int, int], ...]] = set()

    def backtrack(pos: int, chosen: list[dict]) -> None:
        if len(solutions) >= max_solutions:
            return
        if pos == len(ordered_colors):
            valid = _validate_rects(g, chosen)
            if valid is None:
                return
            order, depths, key = valid
            if key in seen_keys:
                return
            seen_keys.add(key)
            solutions.append(
                {
                    "rects": [dict(rect) for rect in chosen],
                    "order": order,
                    "depths": depths,
                    "output": _render_solution(g.shape, chosen, order, depths),
                    "key": key,
                }
            )
            return

        color = ordered_colors[pos]
        for cand in candidate_map[color]:
            _, r0, c0, r1, c1 = cand
            chosen.append({"color": color, "bbox": (r0, c0, r1, c1)})
            backtrack(pos + 1, chosen)
            chosen.pop()

    backtrack(0, [])
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild overlapping colored rectangles, infer how many higher overlapping rectangles cover each one, then recolor by that depth.

    Input structure:
    - Black background contains 3-4 solid axis-aligned rectangles whose visible surfaces use distinct colors from blue, red, green, and yellow.
    - Rectangles can overlap, so only the topmost rectangle color is visible inside each overlap.
    - A few isolated non-black single-pixel noises may appear on background cells.

    Rule:
    - Ignore the isolated noise pixels and recover each rectangle's full bounding box from its fully visible edge and the non-black coverage.
    - For every pair of recovered rectangles that overlap, use the visible color inside their overlap to determine which one is above the other.
    - For each rectangle, count how many overlapping rectangles lie above it: depth 0 -> blue, depth 1 -> red, depth 2 -> green, depth 3 -> yellow.
    - Render the recovered rectangles with those depth colors, keeping black background and showing the topmost recolored rectangle inside overlaps.

    Key insight:
    - The original rectangle colors are only clues for decomposing the occluded rectangles and their stacking order; the output palette depends solely on overlap depth.
    """
    solutions = _solve_all(np.asarray(grid, dtype=int), max_solutions=2)
    assert len(solutions) == 1, "puzzle input should admit exactly one rectangle decomposition"
    return solutions[0]["output"]


def _visible_overlap_cells(
    current_grid: np.ndarray, rect: tuple[int, int, int, int], color: int, candidate: tuple[int, int, int, int]
) -> int:
    box = _overlap_box(rect, candidate)
    if box is None:
        return 0
    r0, c0, r1, c1 = box
    return int(np.count_nonzero(current_grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _sample_rect(
    shape: tuple[int, int],
    current_grid: np.ndarray,
    placed: list[dict],
    overlap_target: int,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    occupied = current_grid != 0
    candidates = []
    for rect_h in RECT_HEIGHTS:
        for rect_w in RECT_WIDTHS:
            if rect_h > h or rect_w > w:
                continue
            for r0 in range(0, h - rect_h + 1):
                for c0 in range(0, w - rect_w + 1):
                    bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                    overlaps = []
                    okay = True
                    visible_overlap_count = 0
                    for prev in placed:
                        box = _overlap_box(prev["bbox"], bbox)
                        if box is None:
                            continue
                        overlap_area = _rect_area(box)
                        if overlap_area < 2 or overlap_area > min(_rect_area(prev["bbox"]), _rect_area(bbox)) // 2 + 2:
                            okay = False
                            break
                        visible_cells = _visible_overlap_cells(current_grid, prev["bbox"], prev["color"], bbox)
                        if visible_cells <= 0:
                            okay = False
                            break
                        overlaps.append(prev)
                        visible_overlap_count += visible_cells
                    if not okay:
                        continue
                    if placed and len(overlaps) != overlap_target:
                        continue
                    visible_sides = _full_visible_sides(bbox, occupied)
                    if not visible_sides:
                        continue
                    visible_area = _rect_area(bbox) - int(np.count_nonzero(occupied[r0 : bbox[2] + 1, c0 : bbox[3] + 1]))
                    if visible_area < max(rect_h, rect_w) + 2:
                        continue
                    if placed and visible_overlap_count < len(overlaps):
                        continue
                    candidates.append(bbox)
    if not candidates:
        return None
    return random.choice(candidates)


def _render_input(shape: tuple[int, int], rects_top_to_bottom: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in reversed(rects_top_to_bottom):
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]
    return grid


def _expected_output(shape: tuple[int, int], rects_top_to_bottom: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx in range(len(rects_top_to_bottom) - 1, -1, -1):
        rect = rects_top_to_bottom[idx]
        depth = sum(
            1
            for higher in rects_top_to_bottom[:idx]
            if _overlap_box(higher["bbox"], rect["bbox"]) is not None
        )
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = DEPTH_TO_COLOR[depth]
    return out


def _add_noise(grid: np.ndarray) -> np.ndarray | None:
    out = grid.copy()
    h, w = out.shape
    count = random.randint(*NOISE_COUNT)
    cells = [(r, c) for r in range(h) for c in range(w) if out[r, c] == 0]
    random.shuffle(cells)
    placed = []
    for r, c in cells:
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in placed):
            continue
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in np.argwhere(out != 0)):
            continue
        out[r, c] = random.choice(NOISE_COLORS)
        placed.append((r, c))
        if len(placed) == count:
            return out
    return None


def _generate_rectangles(shape: tuple[int, int], n_rects: int) -> list[dict] | None:
    colors = random.sample(list(RECT_COLORS), n_rects)
    rects: list[dict] = []
    current_grid = np.zeros(shape, dtype=int)

    for idx, color in enumerate(colors):
        overlap_choices = [0] if idx == 0 else list(range(1, min(idx, 3) + 1))
        random.shuffle(overlap_choices)
        bbox = None
        for overlap_target in overlap_choices:
            bbox = _sample_rect(shape, current_grid, rects, overlap_target)
            if bbox is not None:
                break
        if bbox is None:
            return None
        rects.append({"color": color, "bbox": bbox})
        current_grid = _render_input(shape, rects)

    return rects


def _count_color_changes(rects_top_to_bottom: list[dict]) -> int:
    changes = 0
    for idx, rect in enumerate(rects_top_to_bottom):
        depth = sum(
            1
            for higher in rects_top_to_bottom[:idx]
            if _overlap_box(higher["bbox"], rect["bbox"]) is not None
        )
        if rect["color"] != DEPTH_TO_COLOR[depth]:
            changes += 1
    return changes


def _valid_generated_instance(grid: np.ndarray, expected: np.ndarray, truth: list[dict]) -> bool:
    solutions = _solve_all(grid, max_solutions=3)
    if len(solutions) != 1:
        return False
    if not np.array_equal(solutions[0]["output"], expected):
        return False
    truth_key = _solution_key(
        [{"color": rect["color"], "bbox": rect["bbox"]} for rect in truth],
        [
            sum(1 for higher in truth[:idx] if _overlap_box(higher["bbox"], rect["bbox"]) is not None)
            for idx, rect in enumerate(truth)
        ],
    )
    if solutions[0]["key"] != truth_key:
        return False
    if np.array_equal(grid, expected):
        return False
    if _count_color_changes(truth) < 2:
        return False
    overlap_edges = sum(
        1
        for i in range(len(truth))
        for j in range(i + 1, len(truth))
        if _overlap_box(truth[i]["bbox"], truth[j]["bbox"]) is not None
    )
    if overlap_edges < len(truth) - 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shape = (
            random.randint(*GRID_ROWS),
            random.randint(*GRID_COLS),
        )
        n_rects = random.choice(RECT_COUNT_OPTIONS)
        rects = _generate_rectangles(shape, n_rects)
        if rects is None:
            continue
        clean = _render_input(shape, rects)
        noisy = _add_noise(clean)
        if noisy is None:
            continue
        expected = _expected_output(shape, rects)
        if not _valid_generated_instance(noisy, expected, rects):
            continue
        return noisy
    raise RuntimeError("failed to generate a valid overlapping-rectangle ARC input")
