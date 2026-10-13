from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEAL = 8

ANCHOR_COLORS = (TEAL, RED)
DIRECTIONS = ("NW", "NE", "SW", "SE")
RAY_INFO = {
    "NW": {"step": (-1, -1), "corner_offset": (0, 0)},
    "NE": {"step": (-1, 1), "corner_offset": (0, 1)},
    "SW": {"step": (1, -1), "corner_offset": (1, 0)},
    "SE": {"step": (1, 1), "corner_offset": (1, 1)},
}

GRID_MIN = 14
GRID_MAX = 18
GRAY_COUNT_MIN = 6
GRAY_COUNT_MAX = 10
MAX_GENERATION_ATTEMPTS = 600


def main(grid: np.ndarray) -> np.ndarray:
    """Assign each gray diagonal pixel to its aligned 2x2 anchor, compact it along that diagonal, and recolor it with the anchor's color.

    Input structure:
    - The background is black.
    - There are exactly two solid 2x2 anchor blocks, one teal and one red, in different quadrants.
    - The remaining foreground cells are isolated gray pixels, each lying on exactly one outward diagonal ray of exactly one anchor.

    Rule:
    - For every gray pixel, determine which anchor ray it belongs to by checking the four outward diagonals of each 2x2 anchor: NW from the top-left corner, NE from the top-right, SW from the bottom-left, and SE from the bottom-right.
    - Erase the gray pixels from their original positions.
    - For each anchor and diagonal direction separately, sort that ray's gray pixels by increasing distance from the anchor and pack them into consecutive cells starting adjacent to the anchor along the same diagonal.
    - Recolor each packed pixel with its anchor's color while leaving both anchor blocks unchanged.

    Key insight:
    - Ownership is determined by diagonal alignment to a specific anchor corner, not by proximity, so a pixel can be closer to the other anchor but still compact toward the diagonally aligned one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    anchors = [_find_anchor(g, color) for color in ANCHOR_COLORS]
    gray_cells = [tuple(map(int, cell)) for cell in np.argwhere(g == GREY)]
    assert GRAY_COUNT_MIN <= len(gray_cells) <= GRAY_COUNT_MAX

    groups: dict[tuple[int, str], list[tuple[int, int, int]]] = defaultdict(list)
    for cell in gray_cells:
        anchor_idx, direction, distance = _classify_gray(cell, anchors)
        groups[(anchor_idx, direction)].append((distance, cell[0], cell[1]))

    assert sum(len(cells) for cells in groups.values()) == len(gray_cells)

    out = np.full_like(g, BLACK)
    occupied = set()

    for anchor in anchors:
        for r, c in _anchor_cells(anchor):
            assert (r, c) not in occupied
            occupied.add((r, c))
            out[r, c] = anchor["color"]

    for (anchor_idx, direction), cells in groups.items():
        anchor = anchors[anchor_idx]
        corner = _corner(anchor, direction)
        step = RAY_INFO[direction]["step"]
        for slot, (distance, _r, _c) in enumerate(sorted(cells), start=1):
            rr = corner[0] + step[0] * slot
            cc = corner[1] + step[1] * slot
            assert _in_bounds(g.shape, rr, cc)
            assert (rr, cc) not in occupied
            occupied.add((rr, cc))
            out[rr, cc] = anchor["color"]
            assert distance >= slot

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = (
            int(rng.integers(GRID_MIN, GRID_MAX + 1)),
            int(rng.integers(GRID_MIN, GRID_MAX + 1)),
        )
        anchors = _sample_anchors(shape, rng)
        candidates = _enumerate_candidates(shape, anchors)
        selected = _sample_gray_configuration(shape, anchors, candidates, rng)
        if selected is None:
            continue
        grid = _build_input(shape, anchors, selected)
        if _validate_generated_instance(grid, anchors, selected):
            return grid
    raise RuntimeError("failed to generate valid input for diagonal-anchor compaction")


def _find_anchor(grid: np.ndarray, color: int) -> dict[str, object]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (4, 2), f"expected one 2x2 anchor of color {color}"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    assert (r1 - r0, c1 - c0) == (1, 1)
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert {tuple(map(int, cell)) for cell in coords} == expected
    return {"color": color, "top_left": (r0, c0)}


def _anchor_cells(anchor: dict[str, object]) -> list[tuple[int, int]]:
    r0, c0 = anchor["top_left"]
    return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]


def _corner(anchor: dict[str, object], direction: str) -> tuple[int, int]:
    r0, c0 = anchor["top_left"]
    dr, dc = RAY_INFO[direction]["corner_offset"]
    return r0 + dr, c0 + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _manhattan_to_anchor(cell: tuple[int, int], anchor: dict[str, object]) -> int:
    return min(abs(cell[0] - r) + abs(cell[1] - c) for r, c in _anchor_cells(anchor))


def _classify_relative_to_anchor(
    cell: tuple[int, int], anchor: dict[str, object]
) -> tuple[str, int] | None:
    r, c = cell
    r0, c0 = anchor["top_left"]
    r1, c1 = r0 + 1, c0 + 1
    if r < r0 and c < c0 and (r0 - r) == (c0 - c):
        return "NW", r0 - r
    if r < r0 and c > c1 and (r0 - r) == (c - c1):
        return "NE", r0 - r
    if r > r1 and c < c0 and (r - r1) == (c0 - c):
        return "SW", r - r1
    if r > r1 and c > c1 and (r - r1) == (c - c1):
        return "SE", r - r1
    return None


def _classify_gray(
    cell: tuple[int, int], anchors: list[dict[str, object]]
) -> tuple[int, str, int]:
    matches = []
    for idx, anchor in enumerate(anchors):
        rel = _classify_relative_to_anchor(cell, anchor)
        if rel is not None:
            direction, distance = rel
            matches.append((idx, direction, distance))
    assert len(matches) == 1, f"gray cell {cell} must align with exactly one anchor ray"
    return matches[0]


def _max_distance(shape: tuple[int, int], anchor: dict[str, object], direction: str) -> int:
    corner_r, corner_c = _corner(anchor, direction)
    step_r, step_c = RAY_INFO[direction]["step"]
    row_cap = corner_r if step_r < 0 else (shape[0] - 1 - corner_r)
    col_cap = corner_c if step_c < 0 else (shape[1] - 1 - corner_c)
    return int(min(row_cap, col_cap))


def _sample_anchors(
    shape: tuple[int, int], rng: np.random.Generator
) -> list[dict[str, object]]:
    h, w = shape
    layout = str(rng.choice(["tl_br", "tr_bl"]))
    color_order = list(ANCHOR_COLORS)
    if rng.random() < 0.5:
        color_order.reverse()

    if layout == "tl_br":
        quads = ("tl", "br")
    else:
        quads = ("tr", "bl")

    anchors = []
    for color, quadrant in zip(color_order, quads):
        r0, c0 = _sample_anchor_top_left(shape, quadrant, rng)
        anchors.append({"color": color, "top_left": (r0, c0), "quadrant": quadrant})
    return anchors


def _sample_anchor_top_left(
    shape: tuple[int, int], quadrant: str, rng: np.random.Generator
) -> tuple[int, int]:
    h, w = shape
    top_rows = (2, h // 2 - 3)
    bottom_rows = (h // 2 + 1, h - 4)
    left_cols = (2, w // 2 - 3)
    right_cols = (w // 2 + 1, w - 4)

    row_range = top_rows if quadrant in {"tl", "tr"} else bottom_rows
    col_range = left_cols if quadrant in {"tl", "bl"} else right_cols
    assert row_range[0] <= row_range[1] and col_range[0] <= col_range[1]
    r0 = int(rng.integers(row_range[0], row_range[1] + 1))
    c0 = int(rng.integers(col_range[0], col_range[1] + 1))
    return r0, c0


def _enumerate_candidates(
    shape: tuple[int, int], anchors: list[dict[str, object]]
) -> dict[tuple[int, str], list[dict[str, object]]]:
    candidates: dict[tuple[int, str], list[dict[str, object]]] = {}
    for anchor_idx, anchor in enumerate(anchors):
        for direction in DIRECTIONS:
            ray = (anchor_idx, direction)
            cells = []
            max_dist = _max_distance(shape, anchor, direction)
            for distance in range(2, max_dist + 1):
                corner_r, corner_c = _corner(anchor, direction)
                step_r, step_c = RAY_INFO[direction]["step"]
                cell = (corner_r + step_r * distance, corner_c + step_c * distance)
                try:
                    owner_idx, owner_dir, _ = _classify_gray(cell, anchors)
                except AssertionError:
                    continue
                if (owner_idx, owner_dir) != ray:
                    continue
                owner_dist = _manhattan_to_anchor(cell, anchors[anchor_idx])
                other_dist = _manhattan_to_anchor(cell, anchors[1 - anchor_idx])
                cells.append(
                    {
                        "cell": cell,
                        "anchor_idx": anchor_idx,
                        "direction": direction,
                        "distance": distance,
                        "special": other_dist < owner_dist,
                    }
                )
            candidates[ray] = cells
    return candidates


def _ray_supports_two(cands: list[dict[str, object]]) -> bool:
    for i, first in enumerate(cands):
        for second in cands[i + 1 :]:
            if _chebyshev(first["cell"], second["cell"]) > 1:
                return True
    return False


def _sample_gray_configuration(
    shape: tuple[int, int],
    anchors: list[dict[str, object]],
    candidates: dict[tuple[int, str], list[dict[str, object]]],
    rng: np.random.Generator,
) -> list[dict[str, object]] | None:
    ray_keys = [ray for ray, cells in candidates.items() if cells]
    if len(ray_keys) < 4:
        return None

    special_rays = [ray for ray in ray_keys if any(c["special"] for c in candidates[ray])]
    if not special_rays:
        return None

    doubleable = [ray for ray in ray_keys if _ray_supports_two(candidates[ray])]
    if not doubleable:
        return None

    for _ in range(250):
        active_count = int(rng.integers(4, min(6, len(ray_keys)) + 1))
        active_rays = _sample_without_replacement(ray_keys, active_count, rng)
        active_special = [ray for ray in active_rays if ray in special_rays]
        if not active_special:
            continue

        target_total = int(rng.integers(GRAY_COUNT_MIN, GRAY_COUNT_MAX + 1))
        target_total = max(target_total, active_count + 1 if rng.random() < 0.7 else active_count)
        if target_total > GRAY_COUNT_MAX:
            target_total = GRAY_COUNT_MAX
        extras = target_total - active_count

        active_doubleable = [ray for ray in active_rays if ray in doubleable]
        if extras > len(active_doubleable):
            continue

        counts = {ray: 1 for ray in active_rays}
        if extras > 0:
            for ray in _sample_without_replacement(active_doubleable, extras, rng):
                counts[ray] += 1

        anchor_totals = Counter(ray[0] for ray in active_rays)
        pixel_totals = Counter()
        for ray, count in counts.items():
            pixel_totals[ray[0]] += count
        if pixel_totals[0] < 2 or pixel_totals[1] < 2:
            continue

        packed = _packed_output_positions(shape, anchors, counts)
        if packed is None:
            continue

        special_ray = active_special[int(rng.integers(0, len(active_special)))]

        options_by_ray = {}
        failed = False
        for ray in active_rays:
            force_special = ray == special_ray
            options = _ray_options(candidates[ray], counts[ray], force_special, rng)
            if not options:
                failed = True
                break
            options_by_ray[ray] = options
        if failed:
            continue

        ordered_rays = sorted(active_rays, key=lambda ray: len(options_by_ray[ray]))
        chosen = _choose_options_backtracking(ordered_rays, options_by_ray, [])
        if chosen is None:
            continue
        if not any(item["special"] for item in chosen):
            continue
        if not _has_distance_counterexample(chosen, anchors):
            continue
        if not _counts_are_visually_nontrivial(counts):
            continue
        return chosen
    return None


def _packed_output_positions(
    shape: tuple[int, int],
    anchors: list[dict[str, object]],
    counts: dict[tuple[int, str], int],
) -> set[tuple[int, int]] | None:
    occupied = set()
    for anchor in anchors:
        for cell in _anchor_cells(anchor):
            occupied.add(cell)

    for (anchor_idx, direction), count in counts.items():
        corner_r, corner_c = _corner(anchors[anchor_idx], direction)
        step_r, step_c = RAY_INFO[direction]["step"]
        for slot in range(1, count + 1):
            cell = (corner_r + step_r * slot, corner_c + step_c * slot)
            if not _in_bounds(shape, *cell) or cell in occupied:
                return None
            occupied.add(cell)
    return occupied


def _ray_options(
    candidates: list[dict[str, object]],
    count: int,
    force_special: bool,
    rng: np.random.Generator,
) -> list[list[dict[str, object]]]:
    options: list[list[dict[str, object]]] = []
    if count == 1:
        for cand in candidates:
            if force_special and not cand["special"]:
                continue
            options.append([cand])
    else:
        for i, first in enumerate(candidates):
            for second in candidates[i + 1 :]:
                if _chebyshev(first["cell"], second["cell"]) <= 1:
                    continue
                if force_special and not (first["special"] or second["special"]):
                    continue
                options.append(sorted([first, second], key=lambda item: item["distance"]))
    rng.shuffle(options)
    return options


def _choose_options_backtracking(
    ordered_rays: list[tuple[int, str]],
    options_by_ray: dict[tuple[int, str], list[list[dict[str, object]]]],
    chosen: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    if not ordered_rays:
        return list(chosen)

    ray = ordered_rays[0]
    for option in options_by_ray[ray]:
        if _option_conflicts(option, chosen):
            continue
        chosen.extend(option)
        result = _choose_options_backtracking(ordered_rays[1:], options_by_ray, chosen)
        if result is not None:
            return result
        del chosen[-len(option) :]
    return None


def _option_conflicts(option: list[dict[str, object]], chosen: list[dict[str, object]]) -> bool:
    for cand in option:
        for prev in chosen:
            if _chebyshev(cand["cell"], prev["cell"]) <= 1:
                return True
    return False


def _has_distance_counterexample(
    selected: list[dict[str, object]], anchors: list[dict[str, object]]
) -> bool:
    for item in selected:
        owner = anchors[item["anchor_idx"]]
        other = anchors[1 - item["anchor_idx"]]
        if _manhattan_to_anchor(item["cell"], other) < _manhattan_to_anchor(item["cell"], owner):
            return True
    return False


def _counts_are_visually_nontrivial(counts: dict[tuple[int, str], int]) -> bool:
    active = [count for count in counts.values() if count > 0]
    return len(active) >= 4 and (max(active) > 1 or len(set(active)) > 1)


def _build_input(
    shape: tuple[int, int],
    anchors: list[dict[str, object]],
    selected: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for anchor in anchors:
        for r, c in _anchor_cells(anchor):
            grid[r, c] = anchor["color"]
    for item in selected:
        r, c = item["cell"]
        grid[r, c] = GREY
    return grid


def _sample_without_replacement(
    items: list[tuple[int, str]], count: int, rng: np.random.Generator
) -> list[tuple[int, str]]:
    indices = rng.permutation(len(items))[:count]
    return [items[int(i)] for i in indices]


def _expected_output_from_metadata(
    shape: tuple[int, int],
    anchors: list[dict[str, object]],
    selected: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    occupied = set()
    for anchor in anchors:
        for cell in _anchor_cells(anchor):
            occupied.add(cell)
            out[cell] = anchor["color"]

    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for item in selected:
        groups[(item["anchor_idx"], item["direction"])].append(item)

    for (anchor_idx, direction), items in groups.items():
        corner_r, corner_c = _corner(anchors[anchor_idx], direction)
        step_r, step_c = RAY_INFO[direction]["step"]
        for slot, item in enumerate(sorted(items, key=lambda x: x["distance"]), start=1):
            cell = (corner_r + step_r * slot, corner_c + step_c * slot)
            assert _in_bounds(shape, *cell)
            assert cell not in occupied
            occupied.add(cell)
            out[cell] = anchors[anchor_idx]["color"]
            assert item["distance"] > slot - 1
    return out


def _validate_generated_instance(
    grid: np.ndarray,
    anchors: list[dict[str, object]],
    selected: list[dict[str, object]],
) -> bool:
    gray_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == GREY)]
    if len(gray_cells) != len(selected):
        return False
    if not (GRAY_COUNT_MIN <= len(gray_cells) <= GRAY_COUNT_MAX):
        return False

    expected_anchor_cells = {
        (cell, anchor["color"])
        for anchor in anchors
        for cell in _anchor_cells(anchor)
    }
    for (r, c), color in expected_anchor_cells:
        if int(grid[r, c]) != int(color):
            return False

    selected_map = {item["cell"]: item for item in selected}
    if len(selected_map) != len(selected):
        return False

    for item in selected:
        if grid[item["cell"]] != GREY:
            return False
        try:
            anchor_idx, direction, distance = _classify_gray(item["cell"], anchors)
        except AssertionError:
            return False
        if anchor_idx != item["anchor_idx"] or direction != item["direction"] or distance != item["distance"]:
            return False

    for i, cell in enumerate(gray_cells):
        for other in gray_cells[i + 1 :]:
            if _chebyshev(cell, other) <= 1:
                return False

    per_anchor = Counter(item["anchor_idx"] for item in selected)
    if per_anchor[0] < 2 or per_anchor[1] < 2:
        return False

    if not _has_distance_counterexample(selected, anchors):
        return False

    expected = _expected_output_from_metadata(grid.shape, anchors, selected)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if len({tuple(item["cell"]) for item in selected}) != len(selected):
        return False
    return True
