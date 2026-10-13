from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))

# Generator controls.
GRID_MIN_SIZE = 20
GRID_MAX_SIZE = 30
GROUP_PADDING = 1
INDICATOR_BUFFER_ROWS = 2
INDICATOR_BUFFER_COLS = 4
GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_distances(height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return np.minimum(np.minimum(rows, height - 1 - rows), np.minimum(cols, width - 1 - cols))


def _find_indicator(grid: np.ndarray) -> tuple[int, int, list[int]]:
    candidates: list[tuple[int, int, list[int]]] = []
    for coords in _components(grid != BLACK):
        if len(coords) != 3:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != r1 or c1 - c0 != 2:
            continue
        colors = [int(grid[r0, c]) for c in range(c0, c1 + 1)]
        if len(set(colors)) != 3:
            continue
        candidates.append((r0, c0, colors))

    assert candidates, "expected one 1x3 color-cycle indicator"
    row, col, colors = max(candidates, key=lambda item: (item[0], item[1]))
    return int(row), int(col), colors


def _validate_group(sub: np.ndarray) -> int:
    mask = sub != BLACK
    assert np.any(mask), "group must be nonempty"
    dist = _ring_distances(*sub.shape)
    max_depth = int(dist[mask].max())
    assert max_depth >= 2 and max_depth % 2 == 0, "expected 2-4 concentric frames"
    assert np.array_equal(mask, dist <= max_depth), "group must be a solid stack of rectangular rings"
    for depth in range(max_depth + 1):
        ring_colors = np.unique(sub[dist == depth])
        assert ring_colors.size == 1 and int(ring_colors[0]) != BLACK, "each ring must be monochrome"
    return max_depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor odd-indexed gap rings inside nested concentric rectangles according to the 1x3 cycle indicator,
    then remove the indicator.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    ind_r, ind_c, cycle = _find_indicator(g)
    mapping = {cycle[0]: cycle[1], cycle[1]: cycle[2], cycle[2]: cycle[0]}

    group_mask = g != BLACK
    group_mask[ind_r, ind_c : ind_c + 3] = False

    out = g.copy()
    out[ind_r, ind_c : ind_c + 3] = BLACK

    groups = _components(group_mask)
    assert groups, "expected at least one nested-rectangle group"
    for coords in groups:
        r0, c0, r1, c1 = _bbox(coords)
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        max_depth = _validate_group(sub)
        dist = _ring_distances(*sub.shape)
        for gap_index in range(1, max_depth // 2 + 1):
            if gap_index % 2 == 0:
                continue
            ring_mask = dist == (2 * gap_index - 1)
            color = int(sub[ring_mask][0])
            sub[ring_mask] = mapping.get(color, color)

    return out


def _sample_frame_counts(group_count: int) -> list[int]:
    counts = [random.choice((3, 4))]
    counts.extend(random.choice((2, 3, 4)) for _ in range(group_count - 1))
    random.shuffle(counts)
    return counts


def _sample_groups(group_count: int, cycle_colors: list[int]) -> list[dict]:
    groups: list[dict] = []
    odd_slots: list[tuple[int, int]] = []
    even_slots: list[tuple[int, int]] = []
    size_bonus = {1: 7, 2: 4, 3: 2}[group_count]

    for group_index, frame_count in enumerate(_sample_frame_counts(group_count)):
        min_outer = 4 * frame_count - 1
        outer_h = min_outer + random.randint(0, size_bonus + 1)
        outer_w = min_outer + random.randint(0, size_bonus + 3)

        border_pool = [color for color in COLORS if color not in cycle_colors]
        border_colors = random.sample(border_pool, frame_count)
        plain_gap_pool = [color for color in border_pool if color not in border_colors]

        group = {
            "frames": frame_count,
            "height": outer_h,
            "width": outer_w,
            "borders": border_colors,
            "gaps": [None] * (frame_count - 1),
            "plain_gap_pool": plain_gap_pool,
        }
        groups.append(group)

        for gap_index in range(1, frame_count):
            slot = (group_index, gap_index - 1)
            if gap_index % 2 == 1:
                odd_slots.append(slot)
            else:
                even_slots.append(slot)

    must_cycle = random.choice(odd_slots)
    must_plain = None
    if len(odd_slots) > 1:
        plain_candidates = [slot for slot in odd_slots if slot != must_cycle]
        must_plain = random.choice(plain_candidates)

    def assign(slot: tuple[int, int], use_cycle: bool) -> None:
        group_index, gap_pos = slot
        pool = cycle_colors if use_cycle else groups[group_index]["plain_gap_pool"]
        groups[group_index]["gaps"][gap_pos] = random.choice(pool)

    for slot in odd_slots:
        if slot == must_cycle:
            assign(slot, True)
        elif slot == must_plain:
            assign(slot, False)
        else:
            assign(slot, bool(random.getrandbits(1)))

    for slot in even_slots:
        assign(slot, bool(random.getrandbits(1)))

    return groups


def _sample_grid_shape(groups: list[dict]) -> tuple[int, int] | None:
    max_h = max(group["height"] for group in groups)
    max_w = max(group["width"] for group in groups)
    padded_area = sum((group["height"] + 2 * GROUP_PADDING) * (group["width"] + 2 * GROUP_PADDING) for group in groups)
    side_floor = int(np.ceil(np.sqrt(padded_area))) + 2

    min_h = max(GRID_MIN_SIZE, max_h + 2, side_floor)
    min_w = max(GRID_MIN_SIZE, max_w + 2, side_floor)
    if min_h > GRID_MAX_SIZE or min_w > GRID_MAX_SIZE:
        return None
    return random.randint(min_h, GRID_MAX_SIZE), random.randint(min_w, GRID_MAX_SIZE)


def _place_groups(groups: list[dict], height: int, width: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    occupied[height - INDICATOR_BUFFER_ROWS : height, width - INDICATOR_BUFFER_COLS : width] = True

    placements: list[tuple[int, int] | None] = [None] * len(groups)
    order = sorted(range(len(groups)), key=lambda idx: groups[idx]["height"] * groups[idx]["width"], reverse=True)
    for idx in order:
        group = groups[idx]
        gh = int(group["height"])
        gw = int(group["width"])
        row_starts = list(range(1, height - gh))
        col_starts = list(range(1, width - gw))
        random.shuffle(row_starts)
        random.shuffle(col_starts)

        placed = False
        for top in row_starts:
            for left in col_starts:
                r0 = max(0, top - GROUP_PADDING)
                r1 = min(height, top + gh + GROUP_PADDING)
                c0 = max(0, left - GROUP_PADDING)
                c1 = min(width, left + gw + GROUP_PADDING)
                if np.any(occupied[r0:r1, c0:c1]):
                    continue
                occupied[r0:r1, c0:c1] = True
                placements[idx] = (top, left)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    assert all(pos is not None for pos in placements)
    return [(int(pos[0]), int(pos[1])) for pos in placements]


def _render_group(grid: np.ndarray, group: dict) -> None:
    top = int(group["top"])
    left = int(group["left"])
    height = int(group["height"])
    width = int(group["width"])
    dist = _ring_distances(height, width)
    sub = grid[top : top + height, left : left + width]
    max_depth = 2 * int(group["frames"]) - 2

    for depth in range(max_depth + 1):
        if depth % 2 == 0:
            color = int(group["borders"][depth // 2])
        else:
            color = int(group["gaps"][depth // 2])
        sub[dist == depth] = color


def _render_input(groups: list[dict], cycle_colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    grid = np.zeros((height, width), dtype=int)
    for group in groups:
        _render_group(grid, group)
    grid[height - 1, width - 3 : width] = np.array(cycle_colors, dtype=int)
    return grid


def _expected_output(inp: np.ndarray, groups: list[dict], cycle_colors: list[int]) -> np.ndarray:
    out = inp.copy()
    mapping = {cycle_colors[0]: cycle_colors[1], cycle_colors[1]: cycle_colors[2], cycle_colors[2]: cycle_colors[0]}

    for group in groups:
        top = int(group["top"])
        left = int(group["left"])
        height = int(group["height"])
        width = int(group["width"])
        dist = _ring_distances(height, width)
        sub = out[top : top + height, left : left + width]
        for gap_index, color in enumerate(group["gaps"], start=1):
            if gap_index % 2 == 1:
                sub[dist == (2 * gap_index - 1)] = mapping.get(int(color), int(color))

    out[-1, -3:] = BLACK
    return out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(1, 3)
    cycle_colors = random.sample(list(COLORS), 3)
    groups = _sample_groups(group_count, cycle_colors)

    shape = _sample_grid_shape(groups)
    if shape is None:
        return None
    height, width = shape

    placements = _place_groups(groups, height, width)
    if placements is None or len(placements) != len(groups):
        return None

    for group, (top, left) in zip(groups, placements, strict=False):
        group["top"] = top
        group["left"] = left

    inp = _render_input(groups, cycle_colors, shape)
    out = _expected_output(inp, groups, cycle_colors)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle ARC sample")
