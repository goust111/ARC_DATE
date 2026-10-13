from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SEPARATOR = 5
CONTOUR = 9
SAMPLE_PALETTE = [1, 2, 3, 4, 6, 7, 8]

_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = _NEIGHBORS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _randint(low: int, high: int) -> int:
    return int(_RNG.integers(low, high + 1))


def _components(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    deltas = _NEIGHBORS4 if connectivity == 4 else _NEIGHBORS8
    comps: list[dict[str, object]] = []

    next_id = 0
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or labels[r0, c0] != 0:
                continue
            next_id += 1
            labels[r0, c0] = next_id
            queue = deque([(r0, c0)])
            coords: list[tuple[int, int]] = []
            min_r = max_r = r0
            min_c = max_c = c0
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c
                for dr, dc in deltas:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and labels[rr, cc] == 0:
                        labels[rr, cc] = next_id
                        queue.append((rr, cc))
            comps.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min_r, min_c, max_r, max_c),
                    "area": len(coords),
                }
            )
    return labels, comps


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = grid.shape
    row_candidates = [r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])]
    col_candidates = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    assert bool(row_candidates) ^ bool(col_candidates), "expected exactly one separator orientation"
    if row_candidates:
        assert len(row_candidates) == 1
        return "h", int(row_candidates[0])
    assert len(col_candidates) == 1
    return "v", int(col_candidates[0])


def _extract_sample_pairs(sample: np.ndarray, bg: int, axis: str) -> list[tuple[int, int]]:
    labels, comps = _components(sample != bg, connectivity=4)
    assert labels.size > 0

    pairs: list[tuple[int, int, int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "sample block must be 2x2"
        patch = sample[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg), "sample block must be solid"
        left = int(patch[0, 0])
        right = int(patch[0, 1])
        assert left != right, "sample pair colors must differ"
        assert np.all(patch[:, 0] == left)
        assert np.all(patch[:, 1] == right)
        pairs.append((r0, c0, left, right))

    if axis == "v":
        pairs.sort(key=lambda item: (item[0], item[1]))
    else:
        pairs.sort(key=lambda item: (item[1], item[0]))
    return [(left, right) for _r, _c, left, right in pairs]


def _analyze_main(main: np.ndarray, bg: int) -> tuple[list[dict[str, object]], dict[int, int], dict[int, int], list[int]]:
    contour_colors = [int(v) for v in np.unique(main) if int(v) != bg]
    assert len(contour_colors) == 1, "main area must contain exactly one contour color"
    contour = contour_colors[0]

    black_labels, black_comps = _components(main == bg, connectivity=4)
    assert black_comps, "expected background regions in main area"

    boundary_ids = set(int(v) for v in black_labels[0, :])
    boundary_ids.update(int(v) for v in black_labels[-1, :])
    boundary_ids.update(int(v) for v in black_labels[:, 0])
    boundary_ids.update(int(v) for v in black_labels[:, -1])
    boundary_ids.discard(0)
    assert len(boundary_ids) == 1, "outer background must be unique"
    root = next(iter(boundary_ids))

    _contour_labels, contour_comps = _components(main == contour, connectivity=8)
    graph = {idx: set() for idx in range(1, len(black_comps) + 1)}
    for comp in contour_comps:
        adjacent: set[int] = set()
        for r, c in comp["coords"]:
            for dr, dc in _NEIGHBORS4:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < main.shape[0] and 0 <= cc < main.shape[1]:
                    region_id = int(black_labels[rr, cc])
                    if region_id > 0:
                        adjacent.add(region_id)
        assert len(adjacent) == 2, "each contour must separate exactly two black regions"
        a, b = sorted(adjacent)
        graph[a].add(b)
        graph[b].add(a)

    depth = {root: 0}
    top_group = {root: 0}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for nxt in sorted(graph[current]):
            if nxt in depth:
                continue
            depth[nxt] = depth[current] + 1
            top_group[nxt] = nxt if current == root else top_group[current]
            queue.append(nxt)

    assert len(depth) == len(black_comps), "all black regions must belong to a nesting graph"
    group_ids = [region_id for region_id, d in depth.items() if d == 1]
    assert group_ids, "expected at least one enclosed structure"
    group_ids.sort(
        key=lambda region_id: (
            -int(black_comps[region_id - 1]["area"]),
            int(black_comps[region_id - 1]["bbox"][0]),
            int(black_comps[region_id - 1]["bbox"][1]),
        )
    )
    return black_comps, depth, top_group, group_ids


def main(grid: np.ndarray) -> np.ndarray:
    """Fill nested enclosed regions using sample pairs matched by outer-ring area."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    axis, sep = _find_separator(g, bg)

    if axis == "v":
        main_area = g[:, :sep]
        sample_area = g[:, sep + 1 :]
        sample_axis = "v"
    else:
        main_area = g[:sep, :]
        sample_area = g[sep + 1 :, :]
        sample_axis = "h"

    sample_pairs = _extract_sample_pairs(sample_area, bg, sample_axis)
    black_comps, depth, top_group, group_ids = _analyze_main(main_area, bg)
    assert len(sample_pairs) == len(group_ids), "sample count must match structure count"

    pair_for_group = {group_id: sample_pairs[i] for i, group_id in enumerate(group_ids)}

    out = g.copy()
    for region_id, comp in enumerate(black_comps, start=1):
        d = depth[region_id]
        if d == 0:
            continue
        x, y = pair_for_group[top_group[region_id]]
        fill_color = x if d % 2 == 1 else y
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = fill_color
    return out


def _paint_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _generate_bboxes(level_count: int, group_count: int) -> list[tuple[int, int, int, int]] | None:
    min_outer = 3 + 4 * (level_count - 1)
    extra = 3 if group_count == 3 else 6
    outer_h = _randint(min_outer + 1, min_outer + extra)
    outer_w = _randint(min_outer + 1, min_outer + extra + 1)

    bboxes = [(0, 0, outer_h - 1, outer_w - 1)]
    current = bboxes[0]
    for level in range(1, level_count):
        remain = level_count - level - 1
        min_child_h = 3 + 4 * remain
        min_child_w = 3 + 4 * remain

        cur_h = current[2] - current[0] + 1
        cur_w = current[3] - current[1] + 1
        max_child_h = cur_h - 4
        max_child_w = cur_w - 4
        if max_child_h < min_child_h or max_child_w < min_child_w:
            return None

        child_h = _randint(min_child_h, max_child_h)
        child_w = _randint(min_child_w, max_child_w)
        top = _randint(2, cur_h - child_h - 2)
        left = _randint(2, cur_w - child_w - 2)

        child = (
            current[0] + top,
            current[1] + left,
            current[0] + top + child_h - 1,
            current[1] + left + child_w - 1,
        )
        bboxes.append(child)
        current = child
    return bboxes


def _render_structure(
    bboxes: list[tuple[int, int, int, int]],
    contour_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    h = bboxes[0][2] + 1
    w = bboxes[0][3] + 1
    patch = np.full((h, w), BG, dtype=int)
    depth_map = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1 in bboxes:
        _paint_border(patch, (r0, c0, r1, c1), contour_color)
        depth_map[r0 + 1 : r1, c0 + 1 : c1] += 1
    depth_map[patch != BG] = 0
    return patch, depth_map


def _filled_patch(base: np.ndarray, depth_map: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    x, y = pair
    out = base.copy()
    odd = depth_map % 2 == 1
    even = depth_map > 0
    even &= ~odd
    out[odd] = x
    out[even] = y
    return out


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_structures(
    structures: list[dict[str, object]],
    main_h: int,
    main_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    input_main = np.full((main_h, main_w), BG, dtype=int)
    output_main = np.full((main_h, main_w), BG, dtype=int)
    occupied = np.zeros((main_h, main_w), dtype=bool)

    order = sorted(
        range(len(structures)),
        key=lambda idx: int(np.prod(structures[idx]["input_patch"].shape)),
        reverse=True,
    )

    def search(depth: int) -> bool:
        if depth == len(order):
            return True

        idx = order[depth]
        input_patch = structures[idx]["input_patch"]
        output_patch = structures[idx]["output_patch"]
        h, w = input_patch.shape
        if main_h < h + 2 or main_w < w + 2:
            return False

        candidates = [(r0, c0) for r0 in range(1, main_h - h) for c0 in range(1, main_w - w)]
        if not candidates:
            return False

        for order_idx in _RNG.permutation(len(candidates)):
            r0, c0 = candidates[int(order_idx)]
            if not _can_place(occupied, r0, c0, h, w, pad=1):
                continue

            _paste(input_main, input_patch, r0, c0)
            _paste(output_main, output_patch, r0, c0)
            occupied[r0 : r0 + h, c0 : c0 + w] = True

            if search(depth + 1):
                return True

            occupied[r0 : r0 + h, c0 : c0 + w] = False
            input_main[r0 : r0 + h, c0 : c0 + w] = BG
            output_main[r0 : r0 + h, c0 : c0 + w] = BG
        return False

    if not search(0):
        return None
    return input_main, output_main


def _sample_starts(total: int, count: int) -> list[int] | None:
    if total < 5:
        return None
    choices = np.arange(1, total - 2)
    if len(choices) < count:
        return None
    for _ in range(200):
        starts = np.sort(_RNG.choice(choices, size=count, replace=False))
        if all(int(starts[i + 1]) - int(starts[i]) >= 3 for i in range(count - 1)):
            return [int(v) for v in starts]
    return None


def _build_sample_area(
    pair_order: list[tuple[int, int]],
    main_h: int,
    main_w: int,
    orientation: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    count = len(pair_order)
    if orientation == "v":
        sample_w = _randint(4, 5)
        sample = np.full((main_h, sample_w), BG, dtype=int)
        starts = _sample_starts(main_h, count)
        if starts is None:
            return None
        for row, (x, y) in zip(starts, pair_order):
            col = _randint(1, sample_w - 3)
            sample[row : row + 2, col] = x
            sample[row : row + 2, col + 1] = y
        separator = np.full((main_h, 1), SEPARATOR, dtype=int)
        return separator, sample

    sample_h = _randint(4, 5)
    sample = np.full((sample_h, main_w), BG, dtype=int)
    starts = _sample_starts(main_w, count)
    if starts is None:
        return None
    for col, (x, y) in zip(starts, pair_order):
        row = _randint(1, sample_h - 3)
        sample[row : row + 2, col] = x
        sample[row : row + 2, col + 1] = y
    separator = np.full((1, main_w), SEPARATOR, dtype=int)
    return separator, sample


def _assemble_grid(
    main_input: np.ndarray,
    main_output: np.ndarray,
    sample_pack: tuple[np.ndarray, np.ndarray],
    orientation: str,
) -> tuple[np.ndarray, np.ndarray]:
    separator, sample = sample_pack
    if orientation == "v":
        grid = np.concatenate([main_input, separator, sample], axis=1)
        expected = np.concatenate([main_output, separator, sample], axis=1)
    else:
        grid = np.concatenate([main_input, separator, sample], axis=0)
        expected = np.concatenate([main_output, separator, sample], axis=0)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    orientation = "v" if bool(_RNG.integers(0, 2)) else "h"
    group_count = _randint(2, 3)
    level_counts = [_randint(2, 3) for _ in range(group_count)]

    structures: list[dict[str, object]] = []
    for level_count in level_counts:
        bboxes = _generate_bboxes(level_count, group_count)
        if bboxes is None:
            return None
        input_patch, depth_map = _render_structure(bboxes, CONTOUR)
        outer_area = int((depth_map == 1).sum())
        structures.append(
            {
                "bboxes": bboxes,
                "input_patch": input_patch,
                "depth_map": depth_map,
                "outer_area": outer_area,
            }
        )

    outer_areas = sorted(int(s["outer_area"]) for s in structures)
    if len(set(outer_areas)) != len(outer_areas):
        return None
    if any(outer_areas[i + 1] - outer_areas[i] < 4 for i in range(len(outer_areas) - 1)):
        return None

    sorted_ids = sorted(range(group_count), key=lambda idx: -int(structures[idx]["outer_area"]))
    sample_colors = list(_RNG.choice(SAMPLE_PALETTE, size=2 * group_count, replace=False))
    pair_order: list[tuple[int, int]] = []
    for i in range(group_count):
        x = int(sample_colors[2 * i])
        y = int(sample_colors[2 * i + 1])
        if bool(_RNG.integers(0, 2)):
            x, y = y, x
        pair_order.append((x, y))

    for rank, idx in enumerate(sorted_ids):
        pair = pair_order[rank]
        structures[idx]["pair"] = pair
        structures[idx]["output_patch"] = _filled_patch(
            structures[idx]["input_patch"],
            structures[idx]["depth_map"],
            pair,
        )

    max_h = max(int(s["input_patch"].shape[0]) for s in structures)
    max_w = max(int(s["input_patch"].shape[1]) for s in structures)
    total_patch_area = sum(int(np.prod(s["input_patch"].shape)) for s in structures)

    if orientation == "v":
        max_main_h = 28
        max_main_w = 24
    else:
        max_main_h = 24
        max_main_w = 28

    min_main_h = max(18, max_h + 2, int(np.ceil(np.sqrt(total_patch_area * 0.9))))
    min_main_w = max(18, max_w + 2, int(np.ceil(np.sqrt(total_patch_area * 0.9))))
    if min_main_h > max_main_h or min_main_w > max_main_w:
        return None

    main_h = _randint(max(min_main_h, max_main_h - 5), max_main_h)
    main_w = _randint(max(min_main_w, max_main_w - 5), max_main_w)
    placed = _place_structures(structures, main_h, main_w)
    if placed is None:
        return None
    main_input, main_output = placed

    sample_pack = _build_sample_area(pair_order, main_h, main_w, orientation)
    if sample_pack is None:
        return None
    grid, expected = _assemble_grid(main_input, main_output, sample_pack, orientation)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
