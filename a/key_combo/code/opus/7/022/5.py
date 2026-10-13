from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
LINE_COLOR = 1
FILL_COLOR = 2
SPRITE_COLOR_POOL = (2, 3, 4, 7, 8, 9)
MAX_ATTEMPTS = 500


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _find_monochrome_components(grid: np.ndarray, forbidden: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in forbidden or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "patch": _extract_patch(grid, coords),
                }
            )
    return components


def _fill_line_row(row: np.ndarray) -> np.ndarray:
    filled = row.copy()
    if row.size >= 3:
        mask = (row[1:-1] == BG) & (row[:-2] == LINE_COLOR) & (row[2:] == LINE_COLOR)
        filled[1:-1][mask] = FILL_COLOR
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each dashed blue line, read its full length, and reorder the aligned sprites by that length.

    Input structure:
    - A black background contains 3-4 blue horizontal dashed lines in the upper part and the same number of small monochrome sprites below.
    - Each dashed line is made of blue runs separated only by single black gaps on one row, and the completed line lengths are all different.
    - Every sprite is 4-connected, non-touching, and its bbox horizontal center is exactly aligned with one dashed line center.

    Rule:
    - For every dashed-line row, fill each black cell whose immediate left and right neighbors are blue with red.
    - Measure the completed contiguous line length using both the original blue cells and the inserted red cells.
    - Associate each sprite with the uniquely center-aligned line, sort sprites by their associated completed lengths from short to long, crop each sprite to its bbox, and place the cropped sprites left-to-right with one black column between them.
    - The output is the minimal top-aligned strip containing only the reordered sprites; the dashed lines and red fills disappear.

    Key insight:
    - The sprites do not encode their order directly: the hidden key is the completed length of the dashed line sharing the same horizontal center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    line_rows = [r for r in range(h) if np.any(g[r] == LINE_COLOR)]
    assert 3 <= len(line_rows) <= 4, "expected 3-4 dashed lines"

    line_info: list[dict] = []
    for r in line_rows:
        row = g[r]
        assert set(map(int, np.unique(row))).issubset({BG, LINE_COLOR}), "dashed rows may only contain black and blue"
        filled = _fill_line_row(row)
        cols = np.flatnonzero(filled != BG)
        assert cols.size >= 4, "completed line should be at least length 4"
        start = int(cols[0])
        end = int(cols[-1])
        assert np.all(filled[start : end + 1] != BG), "filled dashed line must become contiguous"
        line_info.append(
            {
                "row": int(r),
                "center2": int(start + end),
                "length": int(end - start + 1),
            }
        )

    lengths = [info["length"] for info in line_info]
    assert len(lengths) == len(set(lengths)), "completed line lengths must be unique"
    line_by_center = {info["center2"]: info for info in line_info}
    assert len(line_by_center) == len(line_info), "line centers must be unique"

    sprites = _find_monochrome_components(g, forbidden={BG, LINE_COLOR})
    assert len(sprites) == len(line_info), "expected one sprite per dashed line"

    paired: list[tuple[int, np.ndarray]] = []
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        center2 = int(c0 + c1)
        assert center2 in line_by_center, "sprite must align to exactly one dashed line center"
        paired.append((line_by_center[center2]["length"], sprite["patch"]))

    paired.sort(key=lambda item: item[0])
    out_h = max(int(patch.shape[0]) for _, patch in paired)
    out_w = sum(int(patch.shape[1]) for _, patch in paired) + len(paired) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for _, patch in paired:
        ph, pw = patch.shape
        mask = patch != BG
        out[:ph, c : c + pw][mask] = patch[mask]
        c += pw + 1

    return out


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _random_sprite_patch(width: int) -> np.ndarray:
    height = random.choice((2, 3))
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    max_cells = min(6, height * width)
    for _ in range(200):
        target_cells = random.randint(3, max_cells)
        coords = {random.choice(all_cells)}
        while len(coords) < target_cells:
            frontier = set()
            for r, c in coords:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            coords.add(random.choice(tuple(frontier)))
        if not _is_connected(coords):
            continue
        if {c for _, c in coords} != set(range(width)):
            continue
        if {r for r, _ in coords} != set(range(height)):
            continue
        patch = np.full((height, width), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = 1
        return patch
    raise RuntimeError("failed to build a connected sprite patch")


def _feasible_blue_counts(length: int) -> list[int]:
    values = []
    for blue_count in range((length + 1) // 2, length):
        runs = length - blue_count + 1
        if runs <= blue_count <= 3 * runs:
            values.append(blue_count)
    return values


def _compose_runs(total: int, count: int) -> tuple[int, ...]:
    choices: list[tuple[int, ...]] = []
    for parts in itertools.product((1, 2, 3), repeat=count):
        if sum(parts) == total:
            choices.append(parts)
    if not choices:
        raise RuntimeError("no valid run composition")
    return random.choice(choices)


def _choose_lengths(n: int) -> list[int]:
    options = [
        combo
        for combo in itertools.combinations(range(4, 13), n)
        if any(length % 2 == 0 for length in combo) and any(length % 2 == 1 for length in combo)
    ]
    return list(random.choice(options))


def _order_signature(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=lambda i: values[i]))


def _assign_blue_counts(lengths: list[int]) -> list[int] | None:
    options_per_line = [_feasible_blue_counts(length) for length in lengths]
    for _ in range(200):
        blue_counts = [random.choice(opts) for opts in options_per_line]
        if len(set(blue_counts)) != len(blue_counts):
            continue
        if _order_signature(blue_counts) == _order_signature(lengths):
            continue
        return blue_counts
    return None


def _place_sprite_columns(lengths: list[int], widths: list[int], width: int) -> list[int] | None:
    order = list(range(len(lengths)))
    random.shuffle(order)

    c0_min = [(length - sprite_w) // 2 for length, sprite_w in zip(lengths, widths)]
    c0_max = [width - sprite_w - (length - sprite_w) // 2 for length, sprite_w in zip(lengths, widths)]
    if any(lo > hi for lo, hi in zip(c0_min, c0_max)):
        return None

    positions = [None] * len(lengths)

    def backtrack(k: int, next_min: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        candidates = list(range(max(c0_min[idx], next_min), c0_max[idx] + 1))
        random.shuffle(candidates)
        for c0 in candidates:
            positions[idx] = c0
            if backtrack(k + 1, c0 + widths[idx] + 1):
                return True
        positions[idx] = None
        return False

    if not backtrack(0, 0):
        return None
    return [int(c0) for c0 in positions]


def _build_expected(sprite_specs: list[dict]) -> np.ndarray:
    ordered = sorted(sprite_specs, key=lambda spec: spec["length"])
    out_h = max(spec["patch"].shape[0] for spec in ordered)
    out_w = sum(spec["patch"].shape[1] for spec in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for spec in ordered:
        patch = spec["patch"]
        mask = patch != BG
        h, w = patch.shape
        out[:h, col : col + w][mask] = patch[mask]
        col += w + 1
    return out


def _validate_nontrivial_orders(line_specs: list[dict], sprite_specs: list[dict]) -> bool:
    length_order = tuple(spec["id"] for spec in sorted(line_specs, key=lambda spec: spec["length"]))
    row_order = tuple(spec["id"] for spec in sorted(line_specs, key=lambda spec: spec["row"]))
    x_order = tuple(spec["id"] for spec in sorted(line_specs, key=lambda spec: spec["center2"]))
    blue_order = tuple(spec["id"] for spec in sorted(line_specs, key=lambda spec: spec["blue_count"]))
    color_order = tuple(spec["id"] for spec in sorted(sprite_specs, key=lambda spec: spec["color"]))
    if length_order in (row_order, tuple(reversed(row_order))):
        return False
    if length_order in (x_order, tuple(reversed(x_order))):
        return False
    if length_order == blue_order:
        return False
    if length_order == color_order:
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n = random.choice((3, 4))
    lengths = _choose_lengths(n)
    blue_counts = _assign_blue_counts(lengths)
    if blue_counts is None:
        raise RuntimeError("failed to decorrelate visible blue counts from final lengths")

    widths = [3 if length % 2 == 1 else 2 for length in lengths]
    min_grid_w = max(14, max(lengths) + 2, sum(widths) + (n - 1) + 2)
    grid_w = random.randint(min_grid_w, min(24, min_grid_w + 4))
    sprite_c0 = _place_sprite_columns(lengths, widths, grid_w)
    if sprite_c0 is None:
        raise RuntimeError("failed to place sprite columns")

    grid_h = random.randint(13, 17)
    line_rows = sorted(random.sample(range(0, 6), n))
    row_assignment = list(range(n))
    random.shuffle(row_assignment)
    sprite_top = random.randint(8, grid_h - 3)

    colors = random.sample(SPRITE_COLOR_POOL, n)
    sprite_specs: list[dict] = []
    line_specs: list[dict] = []

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    max_sprite_bottom = sprite_top
    for idx in range(n):
        length = lengths[idx]
        blue_count = blue_counts[idx]
        width = widths[idx]
        c0 = sprite_c0[idx]
        line_start = c0 - (length - width) // 2
        runs = _compose_runs(blue_count, length - blue_count + 1)
        line_row = line_rows[row_assignment[idx]]
        color = colors[idx]

        col = line_start
        for run_i, run_len in enumerate(runs):
            grid[line_row, col : col + run_len] = LINE_COLOR
            col += run_len
            if run_i + 1 < len(runs):
                col += 1
        assert col - line_start == length

        center2 = 2 * line_start + length - 1
        patch = _random_sprite_patch(width)
        patch = np.where(patch != BG, color, BG)
        ph, pw = patch.shape
        assert pw == width
        grid[sprite_top : sprite_top + ph, c0 : c0 + pw][patch != BG] = patch[patch != BG]
        max_sprite_bottom = max(max_sprite_bottom, sprite_top + ph)

        sprite_specs.append(
            {
                "id": idx,
                "color": color,
                "patch": patch,
                "bbox": (sprite_top, c0, sprite_top + ph - 1, c0 + pw - 1),
                "length": length,
            }
        )
        line_specs.append(
            {
                "id": idx,
                "row": line_row,
                "start": line_start,
                "center2": center2,
                "length": length,
                "blue_count": blue_count,
            }
        )

    assert max_sprite_bottom <= grid_h
    expected = _build_expected(sprite_specs)
    if not _validate_nontrivial_orders(line_specs, sprite_specs):
        raise RuntimeError("sample allows a shortcut ordering")
    if not np.array_equal(main(grid), expected):
        raise RuntimeError("solver mismatch on generated sample")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, _expected = _generate_case()
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC sample")
