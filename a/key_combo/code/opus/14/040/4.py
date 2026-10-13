from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
CONTROL_COLORS = (1, 2, 3, 7)
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _extract_instance(grid: np.ndarray) -> tuple[list[int], list[int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h > 0 and w > 0

    control_rows = [r for r in range(h) if int(g[r, 0]) not in (BLACK, GREY)]
    assert control_rows, "expected at least one control row"
    row_colors = [int(g[r, 0]) for r in control_rows]
    assert len(set(row_colors)) == len(row_colors), "control colors must be distinct"

    grey_rows = {int(r) for r in np.where(np.any(g == GREY, axis=1))[0]}
    assert grey_rows.issubset(set(control_rows)), "grey placeholders must lie on control rows"

    sprite_infos: list[dict] = []
    for color in row_colors:
        comps = _components_from_mask(g == color)
        control_comps = [comp for comp in comps if len(comp) == 1 and comp[0][1] == 0]
        sprite_comps = [comp for comp in comps if all(c > 0 for _r, c in comp)]
        assert len(control_comps) == 1, "each color needs one column-0 control pixel"
        assert len(sprite_comps) == 1, "each control color needs exactly one sprite"
        sprite = sprite_comps[0]
        r0, c0, r1, c1 = _bbox(sprite)
        sprite_infos.append(
            {
                "color": int(color),
                "coords": sprite,
                "area": len(sprite),
                "bbox": (r0, c0, r1, c1),
                "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "shape": _shape_signature(sprite),
            }
        )
    areas = [info["area"] for info in sprite_infos]
    assert len(set(areas)) == len(areas), "sprite areas must be distinct"
    return control_rows, row_colors, sprite_infos


def _build_output(
    grid: np.ndarray,
    control_rows: list[int],
    assigned_colors: list[int],
    sprite_infos: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for info in sprite_infos:
        for r, c in info["coords"]:
            out[r, c] = BLACK
    for row, color in zip(control_rows, assigned_colors):
        out[row, 0] = int(color)
        out[row, grid[row] == GREY] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign each control row by sprite-area rank, then recolor that row's grey placeholders.

    Input structure:
    - The background is black, and some rows have one colored control pixel in column 0.
    - Those control rows also contain grey placeholder pixels, while the right side contains one
      monochrome 4-connected sprite for each control color.
    - Each sprite color matches exactly one control pixel color, and sprite pixel areas are all distinct.

    Rule:
    - Measure every right-side sprite's pixel area and sort the sprites from smallest to largest.
    - Order the control rows from top to bottom.
    - Replace the control color of the k-th control row by the color of the k-th smallest sprite.
    - Recolor all grey pixels in each control row to that row's new control color, then erase all sprites.

    Key insight:
    - The original control colors are decoys; only the ranking of sprite areas determines the final row colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    control_rows, _row_colors, sprite_infos = _extract_instance(g)
    ranked_colors = [info["color"] for info in sorted(sprite_infos, key=lambda info: info["area"])]
    return _build_output(g, control_rows, ranked_colors, sprite_infos)


def _all_groupings(total: int, max_part: int = 3) -> list[tuple[int, ...]]:
    if total == 0:
        return [()]
    out: list[tuple[int, ...]] = []
    for first in range(1, min(max_part, total) + 1):
        for tail in _all_groupings(total - first, max_part):
            out.append((first,) + tail)
    return out


def _scatter_extra(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _choose_row_groups(n_rows: int, group_sizes: tuple[int, ...]) -> list[list[int]] | None:
    min_required = 1 + sum(group_sizes) + (len(group_sizes) - 1) + 1
    if min_required > n_rows:
        return None

    extra = n_rows - min_required
    slots = _scatter_extra(extra, len(group_sizes) + 1)
    current = 1 + slots[0]
    groups: list[list[int]] = []
    for idx, size in enumerate(group_sizes):
        rows = list(range(current, current + size))
        groups.append(rows)
        current += size
        if idx < len(group_sizes) - 1:
            current += 1 + slots[idx + 1]
    return groups


def _interval_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _make_group_intervals(rows: list[int], c_min: int, c_max: int) -> list[tuple[int, int]] | None:
    max_len = min(6, c_max - c_min + 1)
    if max_len < 2:
        return None

    start = random.randint(c_min, c_max - 1)
    end = random.randint(start + 1, min(c_max, start + max_len - 1))
    intervals = [(start, end)]

    for _ in rows[1:]:
        prev = intervals[-1]
        candidates: list[tuple[int, int]] = []
        for length in range(2, max_len + 1):
            for c0 in range(c_min, c_max - length + 2):
                cand = (c0, c0 + length - 1)
                if _interval_overlap(prev, cand):
                    candidates.append(cand)
        if not candidates:
            return None
        intervals.append(random.choice(candidates))
    return intervals


def _grow_polyomino(area: int, require_irregular: bool) -> list[tuple[int, int]] | None:
    board_h = 6
    board_w = 6
    for _ in range(200):
        cells = {(board_h // 2, board_w // 2)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < board_h and 0 <= nc < board_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            options = list(frontier)
            random.shuffle(options)
            placed = False
            for cand in options:
                trial = list(cells | {cand})
                r0, c0, r1, c1 = _bbox(trial)
                if r1 - r0 + 1 <= 4 and c1 - c0 + 1 <= 4:
                    cells.add(cand)
                    placed = True
                    break
            if not placed:
                break
        if len(cells) != area:
            continue
        coords = list(cells)
        r0, c0, r1, c1 = _bbox(coords)
        norm = sorted((r - r0, c - c0) for r, c in coords)
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if require_irregular and bbox_area == area:
            continue
        return norm
    return None


def _can_place(
    coords: list[tuple[int, int]],
    top: int,
    left: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    min_col: int,
) -> bool:
    for dr, dc in coords:
        r = top + dr
        c = left + dc
        if not (1 <= r < h - 1 and min_col <= c < w - 1):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _place_sprites(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color_to_area: dict[int, int],
    sprite_start_col: int,
) -> dict[int, dict] | None:
    items = list(color_to_area.items())
    random.shuffle(items)
    infos: dict[int, dict] = {}
    irregular_color = random.choice([color for color, _area in items])

    for color, area in items:
        shape = _grow_polyomino(area, require_irregular=color == irregular_color)
        if shape is None:
            return None
        bbox_h = max(r for r, _c in shape) + 1
        bbox_w = max(c for _r, c in shape) + 1

        positions = [
            (top, left)
            for top in range(1, grid.shape[0] - bbox_h)
            for left in range(sprite_start_col, grid.shape[1] - bbox_w)
        ]
        random.shuffle(positions)

        placed = None
        for top, left in positions:
            if _can_place(shape, top, left, grid.shape[0], grid.shape[1], occupied, sprite_start_col):
                placed = [(top + dr, left + dc) for dr, dc in shape]
                break
        if placed is None:
            return None

        _paint_coords(grid, placed, color)
        occupied.update(placed)
        r0, c0, r1, c1 = _bbox(placed)
        infos[color] = {
            "color": color,
            "coords": placed,
            "area": area,
            "bbox": (r0, c0, r1, c1),
            "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
            "height": r1 - r0 + 1,
            "width": c1 - c0 + 1,
        }
    return infos


def _orders_match(a: list[int], b: list[int]) -> bool:
    return tuple(a) == tuple(b)


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        control_rows, row_colors, sprite_infos = _extract_instance(grid)
        out = main(grid)
    except AssertionError:
        return False

    n = len(control_rows)
    if n < 2 or n > 4:
        return False
    if any(not np.any(grid[row] == GREY) for row in control_rows):
        return False
    if np.any(grid[:, 0] == GREY):
        return False

    ranked = sorted(sprite_infos, key=lambda info: info["area"])
    ranked_colors = [info["color"] for info in ranked]
    if _orders_match(ranked_colors, row_colors):
        return False
    if _orders_match(ranked_colors, sorted(row_colors)):
        return False

    bbox_areas = [info["bbox_area"] for info in sprite_infos]
    if len(set(bbox_areas)) != len(bbox_areas):
        return False
    bbox_ranked_colors = [info["color"] for info in sorted(sprite_infos, key=lambda info: info["bbox_area"])]
    if _orders_match(ranked_colors, bbox_ranked_colors):
        return False

    if not any(info["bbox_area"] > info["area"] for info in sprite_infos):
        return False

    naive_original = _build_output(grid, control_rows, row_colors, sprite_infos)
    if np.array_equal(out, naive_original):
        return False
    naive_numeric = _build_output(grid, control_rows, sorted(row_colors), sprite_infos)
    if np.array_equal(out, naive_numeric):
        return False
    naive_bbox = _build_output(grid, control_rows, bbox_ranked_colors, sprite_infos)
    if np.array_equal(out, naive_bbox):
        return False

    for row, color in zip(control_rows, ranked_colors):
        if int(out[row, 0]) != color:
            return False
        grey_cols = np.where(grid[row] == GREY)[0]
        if grey_cols.size == 0:
            return False
        if not np.all(out[row, grey_cols] == color):
            return False
    for info in sprite_infos:
        for r, c in info["coords"]:
            if int(out[r, c]) != BLACK:
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n = random.choice((2, 3, 4))
    colors = random.sample(list(CONTROL_COLORS), n)
    row_colors = colors[:]
    random.shuffle(row_colors)

    areas = random.sample(range(3, 9), n)
    color_to_area = {color: area for color, area in zip(colors, areas)}
    ranked_colors = [color for color, _area in sorted(color_to_area.items(), key=lambda item: item[1])]
    if _orders_match(ranked_colors, row_colors) or _orders_match(ranked_colors, sorted(row_colors)):
        return None

    groupings = [g for g in _all_groupings(n) if not (n >= 3 and all(part == 1 for part in g))]
    group_sizes = random.choice(groupings)
    h = random.randint(max(12, 1 + n + len(group_sizes) + 1), 18)
    w = random.randint(15, 18)
    row_groups = _choose_row_groups(h, group_sizes)
    if row_groups is None:
        return None

    left_end = max(6, w // 2 - 1)
    sprite_start = left_end + 3
    if sprite_start >= w - 2:
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()

    intervals_by_row: dict[int, tuple[int, int]] = {}
    for rows in row_groups:
        intervals = _make_group_intervals(rows, 2, left_end)
        if intervals is None:
            return None
        for row, (c0, c1) in zip(rows, intervals):
            intervals_by_row[row] = (c0, c1)

    control_rows = [row for group in row_groups for row in group]
    if len(control_rows) != n:
        return None
    if len(row_colors) != len(control_rows):
        return None

    for row, color in zip(control_rows, row_colors):
        grid[row, 0] = int(color)
        occupied.add((row, 0))
        c0, c1 = intervals_by_row[row]
        grid[row, c0 : c1 + 1] = GREY
        occupied.update((row, c) for c in range(c0, c1 + 1))

    sprite_infos = _place_sprites(grid, occupied, color_to_area, sprite_start)
    if sprite_infos is None:
        return None

    if max(grid.shape) > 30 or not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
