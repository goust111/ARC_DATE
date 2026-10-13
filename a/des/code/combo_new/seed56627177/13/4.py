from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
ICON = 1
GREY = 5
AZURE = 8

LEGEND_COLORS = (2, 3, 4, 6, 7, 9)
ALL_CATEGORIES = ("lr", "ud", "rot", "none")
CELL_SIZE = 7
INNER_SIZE = 5
CELL_GAP = 1
ITEM_SIZE = 5
ITEM_GAP = 1
GEN_RETRIES = 300
PATTERN_SEARCH_RETRIES = 5000

ICON_OFFSETS = {
    "lr": {(-1, 3), (0, 3), (1, 2), (1, 3), (1, 4), (2, 3), (3, 3)},
    "ud": {(-1, 3), (1, 2), (1, 3), (1, 4), (3, 3)},
    "rot": {(0, 2), (1, 3), (2, 4)},
    "none": {(0, 2)},
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _symmetry_flags(mask: np.ndarray) -> dict[str, bool]:
    return {
        "lr": np.array_equal(mask, np.fliplr(mask)),
        "ud": np.array_equal(mask, np.flipud(mask)),
        "rot": np.array_equal(mask, np.rot90(mask, 2)),
    }


def _classify_pattern(mask: np.ndarray, order: list[str]) -> str | None:
    flags = _symmetry_flags(mask)
    no_sym = not any(flags.values())
    for category in order:
        if category == "none":
            if no_sym:
                return "none"
        elif flags[category]:
            return category
    return None


def _is_rectangular_frame(grid: np.ndarray, points: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(
        len(points) == int(border.sum())
        and np.all(sub[border] == GREY)
        and np.all(sub[~border] != GREY)
    )


def _find_regions(grid: np.ndarray) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for pts in _components(grid == GREY):
        if not _is_rectangular_frame(grid, pts):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if not np.any(interior == AZURE):
            continue
        regions.append({"bbox": (r0, c0, r1, c1), "border": pts})
    assert regions, "expected at least one grey-framed region"
    dims = {(int(r1 - r0 + 1), int(c1 - c0 + 1)) for r0, c0, r1, c1 in (region["bbox"] for region in regions)}
    assert len(dims) == 1, "all regions must have the same size"
    return regions


def _parse_legend_square(grid: np.ndarray, sr: int, sc: int, square_color: int) -> tuple[str, int] | None:
    h, w = grid.shape
    row_lo, row_hi = max(0, sr - 1), min(h - 1, sr + 3)
    col_lo, col_hi = max(0, sc), min(w - 1, sc + 4)
    icon_rel = set()
    for rr in range(row_lo, row_hi + 1):
        for cc in range(col_lo, col_hi + 1):
            if int(grid[rr, cc]) == ICON:
                icon_rel.add((rr - sr, cc - sc))
    for category, offsets in ICON_OFFSETS.items():
        if icon_rel == offsets:
            return category, square_color
    return None


def _legend_mapping(grid: np.ndarray) -> list[tuple[str, int]]:
    h, w = grid.shape
    items: list[tuple[str, int, tuple[int, int]]] = []
    for sr in range(h - 1):
        for sc in range(w - 1):
            color = int(grid[sr, sc])
            if color in (BLACK, GREY, AZURE, ICON):
                continue
            if not np.all(grid[sr : sr + 2, sc : sc + 2] == color):
                continue
            if sr > 0 and np.any(grid[sr - 1, sc : sc + 2] == color):
                continue
            if sr + 2 < h and np.any(grid[sr + 2, sc : sc + 2] == color):
                continue
            if sc > 0 and np.any(grid[sr : sr + 2, sc - 1] == color):
                continue
            if sc + 2 < w and np.any(grid[sr : sr + 2, sc + 2] == color):
                continue
            parsed = _parse_legend_square(grid, sr, sc, color)
            if parsed is None:
                continue
            category, square_color = parsed
            items.append((category, square_color, (sr, sc)))
    assert 3 <= len(items) <= 4, "legend must contain three or four items"
    rows = [pos[0] for _cat, _color, pos in items]
    cols = [pos[1] for _cat, _color, pos in items]
    horizontal = (max(cols) - min(cols)) >= (max(rows) - min(rows))
    items.sort(key=lambda item: (item[2][1], item[2][0]) if horizontal else (item[2][0], item[2][1]))
    categories = [category for category, _color, _pos in items]
    assert len(categories) == len(set(categories)), "legend categories must be unique"
    return [(category, color) for category, color, _pos in items]


def _extract_pattern(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1] == AZURE
    pts = np.argwhere(interior)
    assert len(pts) > 0, "each region must contain an azure pattern"
    return _crop(interior)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each framed region according to the symmetry type of its internal azure pattern.

    Input structure:
    - The background is black.
    - Several equal-sized rectangular regions are outlined by one-cell-thick grey borders.
    - Each region contains one small azure pattern on black cells.
    - A corner legend contains three or four colored squares, each paired with a nearby icon that denotes one symmetry type.

    Rule:
    - Decode the legend by reading each icon as one of four symmetry types: left-right mirror, up-down mirror, 180-degree rotational symmetry, or no symmetry.
    - For every region, crop its azure pattern tightly and test which symmetry types it satisfies.
    - If multiple legend-listed types apply, choose the earliest one in the legend order; otherwise choose the legend's no-symmetry entry.
    - Recolor the grey border of that region to the legend color for the chosen type, while leaving the azure pattern unchanged.

    Key insight:
    - The corner legend is a symbolic lookup table from icon shape to color, and its item order is also the tie-breaker when a pattern has multiple symmetries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend = _legend_mapping(g)
    order = [category for category, _color in legend]
    color_map = {category: color for category, color in legend}

    out = g.copy()
    for region in _find_regions(g):
        pattern = _extract_pattern(g, region["bbox"])
        category = _classify_pattern(pattern, order)
        assert category is not None, "every region must match one legend entry"
        color = color_map[category]
        pts = region["border"]
        out[pts[:, 0], pts[:, 1]] = color
    return out


def _render_item(category: str, color: int) -> np.ndarray:
    patch = np.zeros((ITEM_SIZE, ITEM_SIZE), dtype=int)
    patch[1:3, 0:2] = color
    for dr, dc in ICON_OFFSETS[category]:
        patch[1 + dr, dc] = ICON
    return patch


def _random_mask_family(family: str, size: int) -> np.ndarray:
    fill_prob = random.uniform(0.25, 0.65)
    if family == "lr":
        left = np.random.random((size, (size + 1) // 2)) < fill_prob
        mirror = np.fliplr(left[:, : size // 2])
        mask = np.concatenate([left, mirror], axis=1)
    elif family == "ud":
        top = np.random.random(((size + 1) // 2, size)) < fill_prob
        mirror = np.flipud(top[: size // 2, :])
        mask = np.concatenate([top, mirror], axis=0)
    elif family == "rot":
        mask = np.zeros((size, size), dtype=bool)
        for r in range(size):
            for c in range(size):
                rr, cc = size - 1 - r, size - 1 - c
                if (r, c) > (rr, cc):
                    continue
                on = random.random() < fill_prob
                mask[r, c] = on
                mask[rr, cc] = on
    elif family == "all":
        quarter = np.random.random(((size + 1) // 2, (size + 1) // 2)) < fill_prob
        top = np.concatenate([quarter, np.fliplr(quarter[:, : size // 2])], axis=1)
        mask = np.concatenate([top, np.flipud(top[: size // 2, :])], axis=0)
    else:
        mask = np.random.random((size, size)) < fill_prob

    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4))
    if not np.any(mask):
        return mask
    mask = _crop(mask)
    return mask


def _make_candidate(order: list[str]) -> dict[str, object] | None:
    family = random.choice(("lr", "ud", "rot", "all", "none"))
    size = random.randint(3, 5)
    mask = _random_mask_family(family, size)
    if not np.any(mask):
        return None
    h, w = mask.shape
    area = int(mask.sum())
    if min(h, w) < 3 or max(h, w) > 5:
        return None
    if area < 3 or area > 15:
        return None
    category = _classify_pattern(mask, order)
    if category is None:
        return None
    ambiguous = sum(_symmetry_flags(mask).values()) >= 2
    return {"mask": mask, "category": category, "ambiguous": ambiguous}


def _collect_candidates(order: list[str], per_category: int) -> dict[str, list[dict[str, object]]] | None:
    found: dict[str, list[dict[str, object]]] = {category: [] for category in order}
    seen = {category: set() for category in order}
    for _ in range(PATTERN_SEARCH_RETRIES):
        candidate = _make_candidate(order)
        if candidate is None:
            continue
        category = str(candidate["category"])
        if category not in found:
            continue
        key = _mask_key(candidate["mask"])
        if key in seen[category]:
            continue
        seen[category].add(key)
        found[category].append(candidate)
        if all(len(found[category]) >= per_category for category in order):
            return found
    return None


def _balanced_targets(order: list[str], count: int) -> list[str]:
    full_cycles, rem = divmod(count, len(order))
    targets = order * full_cycles + random.sample(order, rem)
    random.shuffle(targets)
    return targets


def _choose_patterns(order: list[str], region_count: int) -> list[np.ndarray] | None:
    need = max(4, (region_count + len(order) - 1) // len(order) + 2)
    candidates = _collect_candidates(order, need)
    if candidates is None:
        return None

    targets = _balanced_targets(order, region_count)
    used: set[tuple[str, ...]] = set()
    chosen: list[np.ndarray] = []

    ambiguous_slots = [
        idx
        for idx, category in enumerate(targets)
        if any(bool(item["ambiguous"]) for item in candidates[category])
    ]
    force_ambiguous = set()
    if ambiguous_slots and random.random() < 0.7:
        force_ambiguous.add(random.choice(ambiguous_slots))

    for idx, category in enumerate(targets):
        pool = candidates[category]
        if idx in force_ambiguous:
            pool = [item for item in pool if bool(item["ambiguous"])]
        elif random.random() < 0.3:
            ambiguous_pool = [item for item in pool if bool(item["ambiguous"])]
            if ambiguous_pool:
                pool = ambiguous_pool
        options = [item for item in pool if _mask_key(item["mask"]) not in used]
        if not options:
            options = [item for item in candidates[category] if _mask_key(item["mask"]) not in used]
        if not options:
            return None
        choice = random.choice(options)
        used.add(_mask_key(choice["mask"]))
        chosen.append(choice["mask"])
    return chosen


def _board_dimensions(rows: int, cols: int) -> tuple[int, int]:
    return rows * CELL_SIZE + (rows - 1) * CELL_GAP, cols * CELL_SIZE + (cols - 1) * CELL_GAP


def _legend_dimensions(count: int, orientation: str) -> tuple[int, int]:
    if orientation == "h":
        return ITEM_SIZE, count * ITEM_SIZE + (count - 1) * ITEM_GAP
    return count * ITEM_SIZE + (count - 1) * ITEM_GAP, ITEM_SIZE


def _layout_positions(rows: int, cols: int, legend_count: int, orientation: str, corner: str) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    board_h, board_w = _board_dimensions(rows, cols)
    legend_h, legend_w = _legend_dimensions(legend_count, orientation)

    if orientation == "h":
        canvas_h = legend_h + ITEM_GAP + board_h
        canvas_w = max(board_w, legend_w)
        if "T" in corner:
            legend_r = 0
            board_r = legend_h + ITEM_GAP
        else:
            board_r = 0
            legend_r = board_h + ITEM_GAP
        if "L" in corner:
            legend_c = 0
            board_c = 0
        else:
            legend_c = canvas_w - legend_w
            board_c = canvas_w - board_w

        starts = []
        for idx in range(legend_count):
            starts.append((legend_r, legend_c + idx * (ITEM_SIZE + ITEM_GAP)))
        return (board_r, board_c), starts, (canvas_h, canvas_w)

    canvas_h = max(board_h, legend_h)
    canvas_w = legend_w + ITEM_GAP + board_w
    if "L" in corner:
        legend_c = 0
        board_c = legend_w + ITEM_GAP
    else:
        board_c = 0
        legend_c = board_w + ITEM_GAP
    if "T" in corner:
        legend_r = 0
        board_r = 0
    else:
        legend_r = canvas_h - legend_h
        board_r = canvas_h - board_h

    starts = []
    for idx in range(legend_count):
        starts.append((legend_r + idx * (ITEM_SIZE + ITEM_GAP), legend_c))
    return (board_r, board_c), starts, (canvas_h, canvas_w)


def _place_board(canvas: np.ndarray, board_origin: tuple[int, int], rows: int, cols: int, patterns: list[np.ndarray]) -> None:
    br, bc = board_origin
    idx = 0
    for row in range(rows):
        for col in range(cols):
            top = br + row * (CELL_SIZE + CELL_GAP)
            left = bc + col * (CELL_SIZE + CELL_GAP)
            canvas[top, left : left + CELL_SIZE] = GREY
            canvas[top + CELL_SIZE - 1, left : left + CELL_SIZE] = GREY
            canvas[top : top + CELL_SIZE, left] = GREY
            canvas[top : top + CELL_SIZE, left + CELL_SIZE - 1] = GREY

            pattern = patterns[idx]
            idx += 1
            ph, pw = pattern.shape
            off_r = random.randint(0, INNER_SIZE - ph)
            off_c = random.randint(0, INNER_SIZE - pw)
            region = canvas[top + 1 : top + 1 + INNER_SIZE, left + 1 : left + 1 + INNER_SIZE]
            region[off_r : off_r + ph, off_c : off_c + pw][pattern] = AZURE


def _place_legend(canvas: np.ndarray, starts: list[tuple[int, int]], entries: list[tuple[str, int]]) -> None:
    for (r0, c0), (category, color) in zip(starts, entries):
        patch = _render_item(category, color)
        canvas[r0 : r0 + ITEM_SIZE, c0 : c0 + ITEM_SIZE] = np.maximum(
            canvas[r0 : r0 + ITEM_SIZE, c0 : c0 + ITEM_SIZE],
            patch,
        )


def _generate_case() -> np.ndarray | None:
    rows, cols = random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
    region_count = rows * cols

    legend_count = 4 if random.random() < 0.5 else 3
    legend_categories = random.sample(list(ALL_CATEGORIES), legend_count)
    legend_colors = random.sample(LEGEND_COLORS, legend_count)
    entries = list(zip(legend_categories, legend_colors))

    patterns = _choose_patterns([category for category, _color in entries], region_count)
    if patterns is None:
        return None

    orientation = random.choice(("h", "v"))
    corner = random.choice(("TL", "TR", "BL", "BR"))
    board_origin, legend_starts, (height, width) = _layout_positions(
        rows,
        cols,
        legend_count,
        orientation,
        corner,
    )
    if height > 30 or width > 30:
        return None

    canvas = np.zeros((height, width), dtype=int)
    _place_board(canvas, board_origin, rows, cols, patterns)
    _place_legend(canvas, legend_starts, entries)

    solved = main(canvas)
    if np.array_equal(solved, canvas):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid puzzle input")
