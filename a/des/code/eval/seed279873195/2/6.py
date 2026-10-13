from __future__ import annotations

import random

import numpy as np


BG = 8
BASELINE = 5
MARKER = 9
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7)

PAIR_COUNT_RANGE = (2, 4)
RECT_DIM_OPTIONS = tuple(
    (h, w)
    for h in range(2, 7)
    for w in (3, 5, 7)
    if 6 <= h * w <= 30
)
MAX_GENERATION_ATTEMPTS = 500
PATCH_PLACEMENT_ATTEMPTS = 200
PARTITION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _normalize_coords(
    coords: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[frozenset[tuple[int, int]], int, int]:
    cells = list(coords)
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    norm = frozenset((r - r0, c - c0) for r, c in cells)
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _bbox_area(coords: set[tuple[int, int]]) -> int:
    _norm, h, w = _normalize_coords(coords)
    return int(h * w)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    remaining = set(coords)
    stack = [next(iter(remaining))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in remaining and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(coords)


def _find_components(
    grid: np.ndarray, ignore_colors: set[int]
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignore_colors:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _find_markers(grid: np.ndarray, bg: int, baseline_color: int) -> tuple[int, list[int]]:
    row = grid[-2]
    colors = [int(v) for v in np.unique(row) if int(v) not in {bg, baseline_color}]
    assert len(colors) == 1
    marker_color = int(colors[0])
    cols = [int(c) for c in np.where(row == marker_color)[0]]
    assert len(cols) >= 2
    steps = {b - a for a, b in zip(cols, cols[1:])}
    assert len(steps) == 1
    return marker_color, cols


def _rectangle_dim_solutions(
    first: tuple[tuple[int, int], ...], second: tuple[tuple[int, int], ...]
) -> set[tuple[int, int]]:
    shape_a, h_a, w_a = _normalize_coords(first)
    shape_b, h_b, w_b = _normalize_coords(second)
    total_area = len(shape_a) + len(shape_b)
    solutions: set[tuple[int, int]] = set()

    for rect_h in range(1, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        rect_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}

        for base_shape, base_h, base_w, other_shape, other_h, other_w in (
            (shape_a, h_a, w_a, shape_b, h_b, w_b),
            (shape_b, h_b, w_b, shape_a, h_a, w_a),
        ):
            if rect_h < base_h or rect_w < base_w:
                continue
            if rect_h < other_h or rect_w < other_w:
                continue

            for r0 in range(rect_h - base_h + 1):
                for c0 in range(rect_w - base_w + 1):
                    placed = {(r + r0, c + c0) for r, c in base_shape}
                    complement = rect_cells - placed
                    if len(complement) != len(other_shape):
                        continue
                    norm_other, other_fit_h, other_fit_w = _normalize_coords(complement)
                    if (
                        other_fit_h == other_h
                        and other_fit_w == other_w
                        and norm_other == other_shape
                    ):
                        solutions.add((int(rect_h), int(rect_w)))

    return solutions


def _find_rectangle_dims(
    first: tuple[tuple[int, int], ...], second: tuple[tuple[int, int], ...]
) -> tuple[int, int]:
    solutions = _rectangle_dim_solutions(first, second)
    assert len(solutions) == 1
    return next(iter(solutions))


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble same-color complementary fragments into rectangles and place them by area order."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _dominant_color(g)

    baseline_color = int(g[-1, 0])
    assert np.all(g[-1] == baseline_color)
    marker_color, marker_cols = _find_markers(g, bg, baseline_color)

    work = g.copy()
    work[-1, :] = bg
    work[-2, work[-2] == marker_color] = bg
    components = _find_components(work, ignore_colors={bg, baseline_color, marker_color})

    groups: dict[int, list[dict[str, object]]] = {}
    for comp in components:
        groups.setdefault(int(comp["color"]), []).append(comp)

    assert groups
    assert len(groups) == len(marker_cols)

    rectangles: list[dict[str, int | tuple[int, int]]] = []
    for color, pair in groups.items():
        assert len(pair) == 2
        first = tuple(pair[0]["coords"])
        second = tuple(pair[1]["coords"])
        rect_h, rect_w = _find_rectangle_dims(first, second)
        area = int(pair[0]["area"]) + int(pair[1]["area"])
        assert area == rect_h * rect_w
        anchor = min(
            (int(pair[0]["bbox"][0]), int(pair[0]["bbox"][1])),
            (int(pair[1]["bbox"][0]), int(pair[1]["bbox"][1])),
        )
        rectangles.append(
            {
                "color": int(color),
                "h": int(rect_h),
                "w": int(rect_w),
                "area": int(area),
                "anchor": anchor,
            }
        )

    rectangles.sort(key=lambda item: (int(item["area"]), item["anchor"]))

    out = np.full((h, w), bg, dtype=int)
    out[-1, :] = baseline_color
    out[-2, marker_cols] = marker_color

    for marker_col, rect in zip(marker_cols, rectangles):
        rect_h = int(rect["h"])
        rect_w = int(rect["w"])
        r0 = h - 1 - rect_h
        c0 = marker_col - rect_w // 2
        assert 0 <= r0 < h - 1
        assert 0 <= c0 and c0 + rect_w <= w
        out[r0 : h - 1, c0 : c0 + rect_w] = int(rect["color"])

    return out


def _random_connected_subset(
    rect_h: int, rect_w: int, target_size: int, rng: random.Random
) -> set[tuple[int, int]]:
    start = (rng.randrange(rect_h), rng.randrange(rect_w))
    chosen = {start}
    frontier = {
        nbr for nbr in _neighbors4(start[0], start[1], rect_h, rect_w) if nbr not in chosen
    }

    while len(chosen) < target_size and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nbr in _neighbors4(cell[0], cell[1], rect_h, rect_w):
            if nbr not in chosen:
                frontier.add(nbr)

    return chosen


def _sample_partition(
    rect_h: int, rect_w: int, rng: random.Random
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    area = rect_h * rect_w
    all_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
    min_piece = max(3, area // 3)
    max_piece = min(area - 3, area - min_piece)

    for _ in range(PARTITION_ATTEMPTS):
        target = rng.randint(min_piece, max_piece)
        first = _random_connected_subset(rect_h, rect_w, target, rng)
        if len(first) != target:
            continue
        second = all_cells - first
        if not _is_connected(first) or not _is_connected(second):
            continue
        if _bbox_area(first) == len(first) or _bbox_area(second) == len(second):
            continue
        norm_first, _fh, _fw = _normalize_coords(first)
        norm_second, _sh, _sw = _normalize_coords(second)
        solutions = _rectangle_dim_solutions(tuple(norm_first), tuple(norm_second))
        if solutions != {(rect_h, rect_w)}:
            continue
        return norm_first, norm_second

    return None


def _shape_to_patch(shape: frozenset[tuple[int, int]], color: int) -> np.ndarray:
    _norm, h, w = _normalize_coords(shape)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _can_place_patch(
    canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int
) -> bool:
    ph, pw = patch.shape
    h, w = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False

    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(h, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(w, cc + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != BG):
            return False
    return True


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for dr, dc in np.argwhere(patch != BG):
        canvas[int(r0 + dr), int(c0 + dc)] = int(patch[int(dr), int(dc)])


def _choose_rectangle_dims(
    pair_count: int, rng: random.Random
) -> list[tuple[int, int]]:
    max_width = 5 if pair_count == 4 else 7
    by_area: dict[int, list[tuple[int, int]]] = {}
    for rect_h, rect_w in RECT_DIM_OPTIONS:
        if rect_w > max_width:
            continue
        by_area.setdefault(rect_h * rect_w, []).append((rect_h, rect_w))

    areas = rng.sample(sorted(by_area), pair_count)
    dims = [rng.choice(by_area[area]) for area in areas]
    rng.shuffle(dims)
    return dims


def _build_expected(
    grid_h: int,
    grid_w: int,
    marker_cols: list[int],
    rectangles: list[dict[str, int]],
) -> np.ndarray:
    out = np.full((grid_h, grid_w), BG, dtype=int)
    out[-1, :] = BASELINE
    out[-2, marker_cols] = MARKER

    for marker_col, rect in zip(
        marker_cols, sorted(rectangles, key=lambda item: int(item["area"]))
    ):
        rect_h = int(rect["h"])
        rect_w = int(rect["w"])
        c0 = marker_col - rect_w // 2
        out[grid_h - 1 - rect_h : grid_h - 1, c0 : c0 + rect_w] = int(rect["color"])

    return out


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    dims = _choose_rectangle_dims(pair_count, rng)
    colors = rng.sample(FRAGMENT_COLORS, pair_count)

    max_width = max(rect_w for _rect_h, rect_w in dims)
    max_height = max(rect_h for rect_h, _rect_w in dims)
    slot_stride = max_width + rng.choice((2, 3))
    margin = max_width // 2 + rng.randint(1, 2)
    marker_cols = [margin + i * slot_stride for i in range(pair_count)]
    grid_w = marker_cols[-1] + max_width // 2 + 1 + rng.randint(1, 3)
    grid_h = rng.randint(max_height + 11, max_height + 16)

    rectangles: list[dict[str, int]] = []
    patches: list[np.ndarray] = []
    for color, (rect_h, rect_w) in zip(colors, dims):
        partition = _sample_partition(rect_h, rect_w, rng)
        if partition is None:
            return None
        first, second = partition
        rectangles.append(
            {"color": int(color), "h": int(rect_h), "w": int(rect_w), "area": rect_h * rect_w}
        )
        patches.append(_shape_to_patch(first, color))
        patches.append(_shape_to_patch(second, color))

    expected = _build_expected(grid_h, grid_w, marker_cols, rectangles)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[-1, :] = BASELINE
    inp[-2, marker_cols] = MARKER

    rng.shuffle(patches)
    for patch in patches:
        ph, _pw = patch.shape
        max_r0 = grid_h - 4 - ph
        if max_r0 < 0:
            return None

        placed = False
        for _ in range(PATCH_PLACEMENT_ATTEMPTS):
            r0 = rng.randint(0, max_r0)
            c0 = rng.randint(0, grid_w - patch.shape[1])
            if not _can_place_patch(inp, patch, r0, c0, padding=1):
                continue
            _place_patch(inp, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None

    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC input")
