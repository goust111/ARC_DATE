import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 500


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _find_divider_col(grid: np.ndarray) -> int:
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(divider_cols) == 1, "expected one full-height grey divider"
    return divider_cols[0]


def _extract_left_bars(left: np.ndarray) -> list[dict]:
    h, w = left.shape
    bars = []
    for c in range(w):
        rows = np.flatnonzero(left[:, c] != BG)
        if rows.size == 0:
            continue
        colors = set(map(int, np.unique(left[rows, c])))
        assert len(colors) == 1, "each bar must be monochrome"
        color = next(iter(colors))
        assert color in {BLUE, RED}, "left half must contain only blue/red bars"
        assert rows[-1] == h - 1, "bars must be bottom-aligned"
        assert np.array_equal(rows, np.arange(rows[0], h)), "bars must be contiguous vertical segments"
        height = int(rows.size)
        bars.append({"color": color, "col": c, "height": height})

    assert bars, "expected at least one bar"
    assert any(bar["color"] == BLUE for bar in bars), "expected at least one blue starter"
    return sorted(bars, key=lambda bar: bar["col"])


def _chain_lengths(bars: list[dict]) -> list[int]:
    by_col = {bar["col"]: bar for bar in bars}
    assert len(by_col) == len(bars), "bars must occupy distinct columns"

    lengths = []
    for bar in bars:
        if bar["color"] != BLUE:
            continue
        length = 1
        seen_cols = {bar["col"]}
        current = bar
        while True:
            next_col = current["col"] + current["height"]
            nxt = by_col.get(next_col)
            if nxt is None or nxt["color"] != RED:
                break
            assert nxt["col"] not in seen_cols, "chain should not cycle"
            seen_cols.add(nxt["col"])
            length += 1
            current = nxt
        lengths.append(length)

    assert lengths, "expected at least one blue-started chain"
    return lengths


def _extract_right_rectangles(right: np.ndarray) -> list[dict]:
    rects = []
    for coords in _components(right != BG, connectivity=4):
        bbox = _bbox_from_coords(coords)
        patch = _crop_bbox(right, bbox)
        colors = set(map(int, np.unique(patch)))
        assert colors <= {GREEN, YELLOW}, "right half must contain only green rectangles with yellow markers"
        assert np.all(patch != BG), "rectangle bbox must be fully filled"
        marker_count = int(np.count_nonzero(patch == YELLOW))
        assert marker_count > 0, "each rectangle needs yellow markers"
        assert np.any(patch == GREEN), "each rectangle must retain green fill"
        rects.append({"bbox": bbox, "patch": patch, "markers": marker_count})

    assert 3 <= len(rects) <= 4, "expected 3-4 right-side rectangles"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique right-side rectangle whose yellow-marker count matches the longest left-side jump chain.

    Input structure:
    - A full-height grey divider splits the grid into independent left and right regions.
    - Left of the divider are bottom-aligned one-cell-wide vertical bars on black: blue bars start chains and red bars can continue them.
    - Right of the divider are 3-4 non-overlapping solid green rectangles on black, each containing some yellow marker pixels.

    Rule:
    - For every blue bar, repeatedly jump to the red bar whose column equals the current bar's column plus the current bar's height; count how many bars belong to that chain, including the starter.
    - Let K be the longest such chain length.
    - Count the yellow marker pixels inside each right-side rectangle and find the unique rectangle whose count equals K.
    - Output that rectangle as an exact crop, preserving both green fill and yellow markers.

    Key insight:
    - The left side encodes a number geometrically via height-based column jumps, and the right side encodes numbers by marker counts; only matching those two independent encodings identifies the output crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= MAX_SIDE, "grid exceeds ARC size limit"

    divider_col = _find_divider_col(g)
    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]
    assert left.size > 0 and right.size > 0, "divider must leave content on both sides"

    bars = _extract_left_bars(left)
    lengths = _chain_lengths(bars)
    k = max(lengths)
    assert lengths.count(k) == 1, "longest chain must be unique"

    rects = _extract_right_rectangles(right)
    winners = [rect for rect in rects if rect["markers"] == k]
    assert len(winners) == 1, "yellow-marker count must match exactly one rectangle"
    return winners[0]["patch"]


def _sample_chain_lengths() -> tuple[list[int], int, int] | None:
    for _ in range(80):
        n_blue = random.choice([2, 3])
        k = random.randint(max(2, n_blue), 5)
        lower = list(range(1, k))
        want_singleton = n_blue >= 2 and 1 in lower and random.random() < 0.6
        forced = [1] if want_singleton else []
        pool = [x for x in lower if x not in forced]
        need = n_blue - 1 - len(forced)
        if need < 0 or len(pool) < need:
            continue
        lengths = [k] + forced + random.sample(pool, need)
        random.shuffle(lengths)
        chained_total = sum(lengths)
        min_extra_red = max(0, 6 - chained_total)
        max_extra_red = min(2, 10 - chained_total)
        if min_extra_red <= max_extra_red:
            extra_red = random.randint(min_extra_red, max_extra_red)
            return lengths, k, extra_red
    return None


def _sample_bar_height() -> int:
    return random.choice([2, 2, 2, 3, 3, 4, 5])


def _build_left_panel(height: int) -> tuple[np.ndarray, int] | None:
    sampled = _sample_chain_lengths()
    if sampled is None:
        return None
    chain_lengths, k, extra_red = sampled

    groups = []
    for chain_len in chain_lengths:
        bars = [{"color": BLUE if i == 0 else RED, "height": _sample_bar_height()} for i in range(chain_len)]
        groups.append({"kind": "chain", "bars": bars, "chain_len": chain_len})
    for _ in range(extra_red):
        groups.append({"kind": "distractor", "bars": [{"color": RED, "height": _sample_bar_height()}], "chain_len": 0})
    random.shuffle(groups)

    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    gaps = [random.randint(1, 3) for _ in range(max(0, len(groups) - 1))]

    col = left_margin
    placed = []
    group_index = 0
    for group in groups:
        for idx, bar in enumerate(group["bars"]):
            placed.append(
                {
                    "color": bar["color"],
                    "col": col,
                    "height": bar["height"],
                    "group_index": group_index,
                    "is_starter": idx == 0 and bar["color"] == BLUE,
                    "declared_chain_len": group["chain_len"],
                }
            )
            col += bar["height"]
        if group_index < len(groups) - 1:
            col += gaps[group_index]
        group_index += 1

    width = col + right_margin
    if not (8 <= width <= 21):
        return None

    panel = np.zeros((height, width), dtype=int)
    for bar in placed:
        if bar["height"] > height:
            return None
        panel[height - bar["height"] :, bar["col"]] = bar["color"]

    parsed = _extract_left_bars(panel)
    lengths = _chain_lengths(parsed)
    if len(lengths) != len(chain_lengths) or sorted(lengths) != sorted(chain_lengths):
        return None
    if lengths.count(max(lengths)) != 1 or max(lengths) != k:
        return None

    starter_cols = [bar["col"] for bar in parsed if bar["color"] == BLUE]
    if starter_cols == sorted(starter_cols) and len(starter_cols) >= 2 and random.random() < 0.2:
        return None
    return panel, k


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _sample_right_boxes(height: int, max_width: int, n_rects: int) -> tuple[int, list[tuple[int, int, int, int]]] | None:
    width_choices = []
    if max_width >= 6:
        width_choices.extend(range(6, min(max_width, 8) + 1))
    if max_width >= 8:
        width_choices.extend(range(8, min(max_width, 12) + 1))
    if not width_choices:
        return None

    for _ in range(80):
        right_width = random.choice(width_choices)
        dims = [(random.randint(3, 5), random.randint(3, 5)) for _ in range(n_rects)]
        dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

        boxes: list[tuple[int, int, int, int]] = []
        for rect_h, rect_w in dims:
            slots = []
            for r0 in range(0, height - rect_h + 1):
                for c0 in range(0, right_width - rect_w + 1):
                    box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                    if all(not _boxes_too_close(box, other, gap=1) for other in boxes):
                        slots.append(box)
            if not slots:
                boxes = []
                break
            boxes.append(random.choice(slots))
        if len(boxes) == n_rects:
            boxes.sort()
            used_width = max(box[3] for box in boxes) + 1
            final_width = min(right_width, used_width + random.randint(0, min(2, right_width - used_width)))
            return final_width, boxes
    return None


def _assign_marker_counts(areas: list[int], k: int) -> tuple[list[int], int] | None:
    n_rects = len(areas)
    target_candidates = [
        i
        for i, area in enumerate(areas)
        if any(other < area for j, other in enumerate(areas) if j != i)
        and any(other > area for j, other in enumerate(areas) if j != i)
    ]
    if not target_candidates:
        return None
    target_idx = random.choice(target_candidates)

    lower_pool = list(range(1, k))
    upper_pool = list(range(k + 1, 9))
    feasible_ranks = [r for r in range(n_rects) if len(lower_pool) >= r and len(upper_pool) >= n_rects - 1 - r]
    if not feasible_ranks:
        return None

    inner_ranks = [r for r in feasible_ranks if 0 < r < n_rects - 1]
    if inner_ranks and random.random() < 0.7:
        target_rank = random.choice(inner_ranks)
    else:
        target_rank = random.choice(feasible_ranks)

    smaller = random.sample(lower_pool, target_rank)
    larger = random.sample(upper_pool, n_rects - 1 - target_rank)
    other_counts = smaller + larger
    random.shuffle(other_counts)

    counts = [0] * n_rects
    counts[target_idx] = k
    cursor = 0
    for i in range(n_rects):
        if i == target_idx:
            continue
        counts[i] = other_counts[cursor]
        cursor += 1

    if len(set(counts)) != n_rects:
        return None
    if any(count >= area for count, area in zip(counts, areas)):
        return None
    target_area = areas[target_idx]
    if not any(area < target_area for i, area in enumerate(areas) if i != target_idx):
        return None
    if not any(area > target_area for i, area in enumerate(areas) if i != target_idx):
        return None
    return counts, target_idx


def _render_right_panel(
    height: int,
    max_width: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    n_rects = random.randint(3, 4)
    sampled = _sample_right_boxes(height, max_width, n_rects)
    if sampled is None:
        return None
    right_width, boxes = sampled

    areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in boxes]
    assigned = _assign_marker_counts(areas, k)
    if assigned is None:
        return None
    counts, target_idx = assigned

    panel = np.zeros((height, right_width), dtype=int)
    target_patch = None
    for idx, ((r0, c0, r1, c1), marker_count) in enumerate(zip(boxes, counts)):
        patch_h = r1 - r0 + 1
        patch_w = c1 - c0 + 1
        patch = np.full((patch_h, patch_w), GREEN, dtype=int)
        cells = random.sample(range(patch_h * patch_w), marker_count)
        patch.reshape(-1)[cells] = YELLOW
        panel[r0 : r1 + 1, c0 : c1 + 1] = patch
        if idx == target_idx:
            target_patch = patch.copy()

    assert target_patch is not None

    rects = _extract_right_rectangles(panel)
    marker_counts = [rect["markers"] for rect in rects]
    if len(set(marker_counts)) != len(marker_counts):
        return None
    if marker_counts.count(k) != 1:
        return None

    winner = next(rect["patch"] for rect in rects if rect["markers"] == k)
    if not np.array_equal(winner, target_patch):
        return None
    return panel, target_patch


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(16, 22)
    left_built = _build_left_panel(height)
    if left_built is None:
        return None
    left_panel, k = left_built

    max_right_width = MAX_SIDE - left_panel.shape[1] - 1
    if max_right_width < 6:
        return None
    right_built = _render_right_panel(height, max_right_width, k)
    if right_built is None:
        return None
    right_panel, expected = right_built

    grid = np.zeros((height, left_panel.shape[1] + 1 + right_panel.shape[1]), dtype=int)
    grid[:, : left_panel.shape[1]] = left_panel
    grid[:, left_panel.shape[1]] = GREY
    grid[:, left_panel.shape[1] + 1 :] = right_panel

    if max(grid.shape) > MAX_SIDE:
        return None

    divider = _find_divider_col(grid)
    chain_lengths = _chain_lengths(_extract_left_bars(grid[:, :divider]))
    if len(set(chain_lengths)) != len(chain_lengths):
        return None
    rects = _extract_right_rectangles(grid[:, divider + 1 :])
    marker_counts = [rect["markers"] for rect in rects]
    if len(set(marker_counts)) != len(marker_counts):
        return None
    if sum(count < k for count in marker_counts) + sum(count > k for count in marker_counts) != len(marker_counts) - 1:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
