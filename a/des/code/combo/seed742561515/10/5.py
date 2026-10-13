from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


AZURE = 8
BLACK = 0
MAROON = 9
ORANGE = 7
YELLOW = 4

FLAG_ORDER = (MAROON, ORANGE, YELLOW)
FLAG_RANK = {color: rank for rank, color in enumerate(FLAG_ORDER)}
BODY_COLORS = (1, 2, 3, 5, 6)

GRID_WIDTH_LIMIT = 30
GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as arrays of (row, col)."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or bool(seen[row, col]):
                continue
            q = deque([(row, col)])
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((int(r), int(c)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]) and not bool(seen[rr, cc]):
                        seen[rr, cc] = True
                        q.append((int(rr), int(cc)))
            comps.append(np.array(cells, dtype=int))

    return comps


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected at least one color"
    return int(counts.most_common(1)[0][0])


def _slice_row_bounds(height: int, parts: int) -> list[tuple[int, int]]:
    assert 1 <= parts <= height
    base = height // parts
    remainder = height % parts
    bounds: list[tuple[int, int]] = []
    start = 0
    for idx in range(parts):
        extra = remainder if idx == parts - 1 else 0
        stop = start + base + extra
        bounds.append((start, stop))
        start = stop
    assert bounds[-1][1] == height
    return bounds


def _extract_buildings(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _components(grid != AZURE)
    buildings: list[dict[str, object]] = []

    for coords in comps:
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        height, width = map(int, patch.shape)

        assert height >= 3 and width >= 4, "building bbox too small"
        assert not bool((patch == AZURE).any()), "each building must fill its bbox"

        black_cols: list[tuple[int, int]] = []
        for local_col in range(width):
            black_rows = np.flatnonzero(patch[:, local_col] == BLACK)
            if black_rows.size == 0:
                continue
            expected = np.arange(height - black_rows.size, height)
            assert np.array_equal(black_rows, expected), "shaft must be a contiguous run from the bottom"
            assert int(black_rows[0]) > 0, "shaft must not reach the roof"
            black_cols.append((int(local_col), int(black_rows.size)))

        assert len(black_cols) == 1, "each building must contain exactly one shaft column"
        shaft_col, shaft_height = black_cols[0]

        body_color = _mode_color(patch[patch != BLACK])
        flag_candidates = [
            local_col
            for local_col in range(width)
            if int(patch[0, local_col]) in FLAG_RANK and int(patch[0, local_col]) != body_color
        ]
        if len(flag_candidates) == 1:
            flag_col = int(flag_candidates[0])
        else:
            assert int(patch[0, shaft_col]) in FLAG_RANK and int(patch[0, shaft_col]) != body_color
            flag_col = int(shaft_col)
        flag_color = int(patch[0, flag_col])

        clean = patch.copy()
        clean[clean == BLACK] = body_color
        clean[0, flag_col] = body_color

        slices = [
            clean[row_start:row_stop, :].copy()
            for row_start, row_stop in _slice_row_bounds(height, shaft_height)
        ]
        assert all(piece.size > 0 for piece in slices)

        buildings.append(
            {
                "left": c0,
                "flag": flag_color,
                "slices": slices,
            }
        )

    buildings.sort(key=lambda item: int(item["left"]))
    assert buildings, "expected at least one building"
    return buildings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut each building into horizontal slices using its shaft height, then reorder
    all slices by roof-flag color group: maroon, then orange, then yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    buildings = _extract_buildings(g)
    ordered_slices: list[np.ndarray] = []
    for flag_color in FLAG_ORDER:
        for building in buildings:
            if int(building["flag"]) == int(flag_color):
                ordered_slices.extend(building["slices"])

    assert ordered_slices, "expected at least one output slice"
    max_slice_height = max(int(piece.shape[0]) for piece in ordered_slices)
    out_width = sum(int(piece.shape[1]) for piece in ordered_slices) + len(ordered_slices) - 1
    out = np.full((max_slice_height, out_width), AZURE, dtype=int)

    col = 0
    for piece in ordered_slices:
        height, width = map(int, piece.shape)
        out[:height, col : col + width] = piece
        col += width + 1

    return out


def _sample_building_specs() -> list[dict[str, int]] | None:
    n_buildings = random.randint(3, 4)

    possible_group_counts = [count for count in (2, 3) if count <= n_buildings]
    active_group_count = random.choice(possible_group_counts)
    active_flags = random.sample(list(FLAG_ORDER), active_group_count)

    flags = active_flags[:]
    while len(flags) < n_buildings:
        flags.append(random.choice(active_flags))

    for _ in range(20):
        random.shuffle(flags)
        ranks = [FLAG_RANK[color] for color in flags]
        if any(ranks[i] > ranks[j] for i in range(n_buildings) for j in range(i + 1, n_buildings)):
            break

    possible_shaft_counts = [count for count in (2, 3) if count <= n_buildings]
    active_shaft_count = random.choice(possible_shaft_counts)
    active_shafts = random.sample([1, 2, 3], active_shaft_count)
    shaft_heights = active_shafts[:]
    while len(shaft_heights) < n_buildings:
        shaft_heights.append(random.choice(active_shafts))
    random.shuffle(shaft_heights)

    body_colors = random.sample(list(BODY_COLORS), n_buildings)
    specs: list[dict[str, int]] = []
    total_width = 0
    for idx in range(n_buildings):
        shaft_height = int(shaft_heights[idx])
        height = random.randint(max(3, shaft_height + 1), 6)
        width = random.randint(4, 8)
        gap_after = random.randint(1, 2) if idx < n_buildings - 1 else 0
        total_width += width + gap_after
        specs.append(
            {
                "height": height,
                "width": width,
                "shaft_height": shaft_height,
                "flag": int(flags[idx]),
                "body": int(body_colors[idx]),
                "gap_after": gap_after,
            }
        )

    if all(spec["height"] % spec["shaft_height"] == 0 for spec in specs):
        adjustable = [spec for spec in specs if spec["shaft_height"] > 1]
        if not adjustable:
            return None
        chosen = random.choice(adjustable)
        candidates = [
            height
            for height in range(max(3, chosen["shaft_height"] + 1), 7)
            if height % chosen["shaft_height"] != 0
        ]
        if not candidates:
            return None
        chosen["height"] = random.choice(candidates)

    side_margins = random.randint(0, 2) + random.randint(0, 2)
    if total_width + side_margins > GRID_WIDTH_LIMIT:
        return None
    return specs


def _render_input(specs: list[dict[str, int]]) -> np.ndarray:
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    base_width = sum(spec["width"] + spec["gap_after"] for spec in specs)
    spare_width = GRID_WIDTH_LIMIT - base_width
    assert spare_width >= 0
    left_margin = random.randint(0, min(2, spare_width))
    right_margin = random.randint(0, min(2, spare_width - left_margin))

    max_height = max(spec["height"] for spec in specs)
    grid_height = top_margin + max_height + bottom_margin
    grid_width = left_margin + base_width + right_margin
    assert grid_width <= GRID_WIDTH_LIMIT

    grid = np.full((grid_height, grid_width), AZURE, dtype=int)
    baseline = top_margin + max_height - 1

    col = left_margin
    for spec in specs:
        height = int(spec["height"])
        width = int(spec["width"])
        shaft_height = int(spec["shaft_height"])
        body = int(spec["body"])
        flag = int(spec["flag"])

        patch = np.full((height, width), body, dtype=int)
        shaft_col = width // 2
        patch[height - shaft_height :, shaft_col] = BLACK
        patch[0, shaft_col] = flag

        top = baseline - height + 1
        grid[top : top + height, col : col + width] = patch
        col += width + int(spec["gap_after"])

    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        buildings = _extract_buildings(grid)
        output = main(grid)
    except Exception:
        return False

    flags_present = {int(building["flag"]) for building in buildings}
    shaft_heights = {len(building["slices"]) for building in buildings}
    if not flags_present:
        return False
    if not shaft_heights.issubset({1, 2, 3}):
        return False
    if output.shape[0] < 1 or output.shape[1] < 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        specs = _sample_building_specs()
        if specs is None:
            continue
        grid = _render_input(specs)
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
