from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEAL = 8

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, object]] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox(arr), "area": int(arr.shape[0])})
    return components


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _bboxes_share_row_or_col(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _merge_fragment_groups(components: list[dict[str, object]]) -> list[dict[str, object]]:
    parent = list(range(len(components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _bboxes_share_row_or_col(
                components[i]["bbox"],  # type: ignore[arg-type]
                components[j]["bbox"],  # type: ignore[arg-type]
            ):
                union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(len(components)):
        buckets.setdefault(find(i), []).append(i)

    groups: list[dict[str, object]] = []
    for indices in buckets.values():
        fragment_coords = [components[i]["coords"] for i in indices]  # type: ignore[index]
        coords = np.vstack(fragment_coords)
        groups.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
                "fragment_count": len(indices),
                "fragment_areas": [int(components[i]["area"]) for i in indices],
            }
        )
    return groups


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _segment_start(center: int, length: int, limit: int) -> int:
    start = int(center) - int(length) // 2
    stop = start + int(length)
    assert 0 <= start < stop <= limit, "compressed segment must fit inside the grid"
    return start


def _guide_info(grid: np.ndarray) -> tuple[str, tuple[int, int], int]:
    h, w = grid.shape
    full_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    full_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]

    if len(full_rows) == 2 and len(full_cols) == 1:
        return "H", (int(full_rows[0]), int(full_rows[1])), int(full_cols[0])
    if len(full_rows) == 1 and len(full_cols) == 2:
        return "V", (int(full_cols[0]), int(full_cols[1])), int(full_rows[0])
    raise AssertionError("expected exactly two parallel grey bars and one perpendicular center bar")


def _analyze_scene(grid: np.ndarray) -> tuple[str, tuple[int, int], int, list[dict[str, object]]]:
    orientation, pair, center = _guide_info(grid)
    fragments = _find_components(grid == TEAL)
    assert fragments, "expected at least one teal fragment"
    groups = _merge_fragment_groups(fragments)
    return orientation, pair, center, groups


def main(grid: np.ndarray) -> np.ndarray:
    """Merge teal fragment groups, then compress each group into a centered red count segment.

    Input structure:
    - Black background contains a grey guide made of exactly two full parallel bars and one full perpendicular center bar.
    - Teal foreground appears only between the parallel bars and is split into several 4-connected fragments.
    - Fragments belonging to the same hidden object are not necessarily touching, but their row spans or column spans overlap transitively.

    Rule:
    - Find all 4-connected teal fragments and merge them transitively whenever fragment bounding boxes overlap in row span or in column span.
    - For each merged group, count its teal pixels and compute the rounded mean coordinate on the axis perpendicular to the guide-bar direction.
    - Erase the teal fragments and draw a red 1-cell-thick segment of that pixel count, centered on the grey center bar and placed on the group's rounded mean row or column.

    Key insight:
    - The puzzle is not about each connected fragment separately: the real objects are fragment groups recovered from bbox-span overlap, and only the guide bars determine the compression axis and center anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orientation, _pair, center, groups = _analyze_scene(g)
    out = g.copy()
    out[out == TEAL] = BLACK

    for group in groups:
        coords = group["coords"]  # type: ignore[assignment]
        area = int(group["area"])
        if orientation == "H":
            row = _round_half_up(float(np.mean(coords[:, 0])))
            start = _segment_start(center, area, g.shape[1])
            out[row, start : start + area] = RED
        else:
            col = _round_half_up(float(np.mean(coords[:, 1])))
            start = _segment_start(center, area, g.shape[0])
            out[start : start + area, col] = RED
    return out


def _random_partition(total: int, count: int, lo: int, hi: int) -> list[int] | None:
    for _ in range(200):
        values = [lo] * count
        remaining = total - lo * count
        while remaining > 0:
            choices = [i for i, value in enumerate(values) if value < hi]
            if not choices:
                break
            idx = random.choice(choices)
            add = random.randint(1, min(hi - values[idx], remaining))
            values[idx] += add
            remaining -= add
        if remaining == 0:
            random.shuffle(values)
            return values
    return None


def _grow_row_fragment(slot_w: int, area: int) -> np.ndarray | None:
    if area < 2 or area > min(8, 2 * slot_w):
        return None
    cells = {(0, random.randrange(slot_w)), (1, random.randrange(slot_w))}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < 2 and 0 <= nc < slot_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    patch = np.zeros((2, slot_w), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    comp_count = len(_find_components(patch))
    if comp_count != 1:
        return None
    return patch


def _grow_col_fragment(slot_h: int, area: int) -> np.ndarray | None:
    if area < 2 or area > min(8, 2 * slot_h):
        return None
    cells = {(random.randrange(slot_h), 0), (random.randrange(slot_h), 1)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < slot_h and 0 <= nc < 2 and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    patch = np.zeros((slot_h, 2), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    comp_count = len(_find_components(patch))
    if comp_count != 1:
        return None
    return patch


def _build_row_chain(max_width: int, area_cap: int) -> np.ndarray | None:
    for _ in range(200):
        fragment_count = random.randint(2, 4)
        min_total = 2 * fragment_count
        if min_total > area_cap:
            continue
        total = random.randint(min_total, min(area_cap, 8 * fragment_count))
        areas = _random_partition(total, fragment_count, 2, 8)
        if areas is None:
            continue

        widths = [max(1, math.ceil(area / 2) + random.randint(0, 1)) for area in areas]
        chain_width = sum(widths) + fragment_count - 1
        if chain_width > max_width:
            continue

        patch = np.zeros((2, chain_width), dtype=bool)
        x = 0
        ok = True
        for area, width in zip(areas, widths):
            frag = _grow_row_fragment(width, area)
            if frag is None:
                ok = False
                break
            patch[:, x : x + width] = frag
            x += width + 1
        if not ok:
            continue

        components = _find_components(patch)
        if len(components) != fragment_count:
            continue
        groups = _merge_fragment_groups(components)
        if len(groups) != 1:
            continue
        return patch
    return None


def _build_col_chain(max_height: int, area_cap: int) -> np.ndarray | None:
    for _ in range(200):
        fragment_count = random.randint(2, 4)
        min_total = 2 * fragment_count
        if min_total > area_cap:
            continue
        total = random.randint(min_total, min(area_cap, 8 * fragment_count))
        areas = _random_partition(total, fragment_count, 2, 8)
        if areas is None:
            continue

        heights = [max(1, math.ceil(area / 2) + random.randint(0, 1)) for area in areas]
        chain_height = sum(heights) + fragment_count - 1
        if chain_height > max_height:
            continue

        patch = np.zeros((chain_height, 2), dtype=bool)
        y = 0
        ok = True
        for area, height in zip(areas, heights):
            frag = _grow_col_fragment(height, area)
            if frag is None:
                ok = False
                break
            patch[y : y + height, :] = frag
            y += height + 1
        if not ok:
            continue

        components = _find_components(patch)
        if len(components) != fragment_count:
            continue
        groups = _merge_fragment_groups(components)
        if len(groups) != 1:
            continue
        return patch
    return None


def _paint_patch(grid: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    rows, cols = np.where(patch)
    grid[top + rows, left + cols] = TEAL


def _attempt_horizontal_scene() -> np.ndarray | None:
    h = random.randint(17, 24)
    w = random.randint(18, 26)
    if max(h, w) > 30:
        return None

    interior_height = random.randint(8, min(12, h - 5))
    top_bar = random.randint(2, h - interior_height - 3)
    bottom_bar = top_bar + interior_height + 1
    center_col = random.randint(8, w - 9)

    left_start, left_end = 1, center_col - 2
    right_start, right_end = center_col + 2, w - 2
    left_width = left_end - left_start + 1
    right_width = right_end - right_start + 1
    if min(left_width, right_width) < 5:
        return None

    top_row = top_bar + 1 + random.randint(0, 1)
    bottom_row = bottom_bar - 2 - random.randint(0, 1)
    if top_row + 1 >= bottom_row:
        return None

    left_patch = _build_row_chain(left_width, interior_height)
    right_patch = _build_row_chain(right_width, interior_height)
    if left_patch is None or right_patch is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[top_bar, :] = GREY
    grid[bottom_bar, :] = GREY
    grid[:, center_col] = GREY

    swap = random.choice([False, True])
    if swap:
        top_patch, bottom_patch = right_patch, left_patch
        top_left_range = (right_start, right_end - right_patch.shape[1] + 1)
        bottom_left_range = (left_start, left_end - left_patch.shape[1] + 1)
    else:
        top_patch, bottom_patch = left_patch, right_patch
        top_left_range = (left_start, left_end - left_patch.shape[1] + 1)
        bottom_left_range = (right_start, right_end - right_patch.shape[1] + 1)

    if top_left_range[0] > top_left_range[1] or bottom_left_range[0] > bottom_left_range[1]:
        return None

    top_left = random.randint(top_left_range[0], top_left_range[1])
    bottom_left = random.randint(bottom_left_range[0], bottom_left_range[1])
    _paint_patch(grid, top_row, top_left, top_patch)
    _paint_patch(grid, bottom_row, bottom_left, bottom_patch)
    return grid


def _attempt_vertical_scene() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(17, 24)
    if max(h, w) > 30:
        return None

    interior_width = random.randint(8, min(12, w - 5))
    left_bar = random.randint(2, w - interior_width - 3)
    right_bar = left_bar + interior_width + 1
    center_row = random.randint(8, h - 9)

    top_start, top_end = 1, center_row - 2
    bottom_start, bottom_end = center_row + 2, h - 2
    top_height = top_end - top_start + 1
    bottom_height = bottom_end - bottom_start + 1
    if min(top_height, bottom_height) < 5:
        return None

    left_col = left_bar + 1 + random.randint(0, 1)
    right_col = right_bar - 2 - random.randint(0, 1)
    if left_col + 1 >= right_col:
        return None

    top_patch = _build_col_chain(top_height, interior_width)
    bottom_patch = _build_col_chain(bottom_height, interior_width)
    if top_patch is None or bottom_patch is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[:, left_bar] = GREY
    grid[:, right_bar] = GREY
    grid[center_row, :] = GREY

    swap = random.choice([False, True])
    if swap:
        upper_patch, lower_patch = bottom_patch, top_patch
        upper_top_range = (top_start, top_end - bottom_patch.shape[0] + 1)
        lower_top_range = (bottom_start, bottom_end - top_patch.shape[0] + 1)
        upper_col, lower_col = right_col, left_col
    else:
        upper_patch, lower_patch = top_patch, bottom_patch
        upper_top_range = (top_start, top_end - top_patch.shape[0] + 1)
        lower_top_range = (bottom_start, bottom_end - bottom_patch.shape[0] + 1)
        upper_col, lower_col = left_col, right_col

    if upper_top_range[0] > upper_top_range[1] or lower_top_range[0] > lower_top_range[1]:
        return None

    upper_top = random.randint(upper_top_range[0], upper_top_range[1])
    lower_top = random.randint(lower_top_range[0], lower_top_range[1])
    _paint_patch(grid, upper_top, upper_col, upper_patch)
    _paint_patch(grid, lower_top, lower_col, lower_patch)
    return grid


def _valid_generated_scene(grid: np.ndarray) -> bool:
    try:
        orientation, pair, center, groups = _analyze_scene(grid)
    except AssertionError:
        return False

    if len(groups) != 2:
        return False

    interior_span = abs(pair[1] - pair[0]) - 1
    coords_rounded: list[int] = []
    for group in groups:
        if not (2 <= int(group["fragment_count"]) <= 4):
            return False
        fragment_areas = group["fragment_areas"]  # type: ignore[assignment]
        if any(area < 2 or area > 8 for area in fragment_areas):
            return False
        if int(group["area"]) > interior_span:
            return False

        coords = group["coords"]  # type: ignore[assignment]
        if orientation == "H":
            coords_rounded.append(_round_half_up(float(np.mean(coords[:, 0]))))
        else:
            coords_rounded.append(_round_half_up(float(np.mean(coords[:, 1]))))

    if len(set(coords_rounded)) != len(coords_rounded):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_horizontal_scene() if random.choice([False, True]) else _attempt_vertical_scene()
        if grid is not None and _valid_generated_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
