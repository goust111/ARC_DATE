from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
GRAY = 5
SEPARATOR = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
RUN_LENGTH_CHOICES = np.arange(3, 8, dtype=int)
LEFTOVER_AREA_CHOICES = tuple(range(2, 12))
MAX_GENERATION_ATTEMPTS = 800
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[np.ndarray] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            cells.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        parts.append(np.array(cells, dtype=int))
    return parts


def _scan_code_row(code_row: np.ndarray) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    longest: dict[int, int] = {}
    totals: dict[int, int] = {}
    first_start: dict[int, int] = {}
    c = 0
    width = int(code_row.shape[0])
    while c < width:
        color = int(code_row[c])
        start = c
        while c < width and int(code_row[c]) == color:
            c += 1
        run_len = c - start
        longest[color] = max(longest.get(color, 0), run_len)
        totals[color] = totals.get(color, 0) + run_len
        first_start.setdefault(color, start)
    return longest, totals, first_start


def _analyze_input(grid: np.ndarray) -> tuple[int, np.ndarray, dict[int, int], dict[int, int], dict[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    bg = _get_bg(g)

    separator_rows = [r for r in range(h) if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg]
    assert len(separator_rows) == 1
    separator_row = int(separator_rows[0])
    assert separator_row < h - 1

    code_row = g[separator_row + 1, :].copy()
    longest, totals, first_start = _scan_code_row(code_row)
    assert len(longest) >= 2
    assert len(set(longest.values())) == len(longest)

    top = g[:separator_row, :]
    parts = _components(top == GRAY)
    objects: list[dict] = []
    for coords in parts:
        box = _bbox(coords)
        objects.append(
            {
                "coords": coords,
                "bbox": box,
                "area": int(coords.shape[0]),
                "center_c": float(coords[:, 1].mean()),
                "center_r": float(coords[:, 0].mean()),
            }
        )

    assert len(objects) == len(longest)
    assert len({obj["area"] for obj in objects}) == len(objects)
    return separator_row, code_row, longest, totals, first_start, objects


def _assignment_from_values(objects: list[dict], color_to_value: dict[int, int]) -> dict[int, int] | None:
    if len(set(color_to_value.values())) != len(color_to_value):
        return None

    value_to_color = {value: color for color, value in color_to_value.items()}
    assignment: dict[int, int] = {}
    matched_colors: set[int] = set()
    leftovers: list[int] = []

    for idx, obj in enumerate(objects):
        area = int(obj["area"])
        color = value_to_color.get(area)
        if color is None or color in matched_colors:
            leftovers.append(idx)
            continue
        assignment[idx] = color
        matched_colors.add(color)

    if len(leftovers) != 1:
        return None

    remaining_colors = [color for color in color_to_value if color not in matched_colors]
    if len(remaining_colors) != 1:
        return None

    assignment[leftovers[0]] = int(remaining_colors[0])
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each gray object by the matching longest code-row run length, with one leftover object assigned by elimination.

    Input structure:
    - A solid full-width separator row splits the grid into an upper object zone and a lower code row.
    - The row below the separator contains several colors, and each color may appear in multiple separated runs.
    - Above the separator are disconnected gray objects whose pixel counts are all distinct.

    Rule:
    - For each color in the code row, measure that color's longest contiguous horizontal run.
    - Recolor every gray object whose area equals one of those longest-run lengths using the corresponding color.
    - Exactly one object area matches no longest run and exactly one code color remains unused; recolor that object with the leftover color.

    Key insight:
    - The persistent key is the per-color longest contiguous segment length, while the final unmatched object-color pair is resolved only by simultaneous elimination on both sides.
    """
    g = np.asarray(grid, dtype=int)
    separator_row, _code_row, longest, _totals, _first_start, objects = _analyze_input(g)

    assignment = _assignment_from_values(objects, longest)
    assert assignment is not None

    direct_matches = sum(obj["area"] in set(longest.values()) for obj in objects)
    assert direct_matches == len(objects) - 1

    out = g.copy()
    for idx, color in assignment.items():
        coords = objects[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

    assert np.array_equal(out[separator_row + 1 :, :], g[separator_row + 1 :, :])
    return out


def _crop_nonzero(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_shape(area: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    pos = (r + dr, c + dc)
                    if pos not in cells:
                        frontier.add(pos)
            frontier_list = list(frontier)
            cells.add(frontier_list[int(rng.integers(len(frontier_list)))])

        coords = np.array(list(cells), dtype=int)
        coords -= coords.min(axis=0)
        h, w = coords.max(axis=0) + 1
        if area >= 4 and (int(h) == 1 or int(w) == 1):
            continue
        if int(h) > 6 or int(w) > 6:
            continue

        mask = np.zeros((int(h), int(w)), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        mask = np.rot90(mask, int(rng.integers(4)))
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = _crop_nonzero(mask)
        h, w = mask.shape
        if area >= 5 and h * w > area * 2 + 2:
            continue

        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = GRAY
        return patch
    return None


def _all_positions(blocked: np.ndarray, patch_mask: np.ndarray, border: int = 1) -> list[tuple[int, int]]:
    top_h, width = blocked.shape
    h, w = patch_mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(border, top_h - h - border + 1):
        for c0 in range(border, width - w - border + 1):
            if not np.any(blocked[r0 : r0 + h, c0 : c0 + w] & patch_mask):
                positions.append((r0, c0))
    return positions


def _stamp_with_padding(canvas: np.ndarray, blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    mask = patch == GRAY
    canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = GRAY
    for rr, cc in np.argwhere(mask):
        ar = r0 + int(rr)
        ac = c0 + int(cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = ar + dr
                nc = ac + dc
                if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                    blocked[nr, nc] = True
    abs_coords = np.argwhere(mask) + np.array([[r0, c0]])
    return abs_coords.astype(int)


def _place_objects(areas: list[int], top_h: int, width: int, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]] | None:
    canvas = np.full((top_h, width), BG, dtype=int)
    blocked = np.zeros((top_h, width), dtype=bool)
    placements: list[np.ndarray | None] = [None] * len(areas)

    for idx in np.argsort([-area for area in areas]):
        placed = False
        for _ in range(80):
            patch = _random_shape(int(areas[idx]), rng)
            if patch is None:
                continue
            positions = _all_positions(blocked, patch == GRAY, border=1)
            if not positions:
                return None
            r0, c0 = positions[int(rng.integers(len(positions)))]
            placements[idx] = _stamp_with_padding(canvas, blocked, patch, int(r0), int(c0))
            placed = True
            break
        if not placed:
            return None

    return canvas, [coords for coords in placements if coords is not None]


def _ordered_segments(
    segments: list[tuple[int, int]], rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    count = len(segments)
    for _ in range(400):
        order = [segments[i] for i in rng.permutation(count)]
        if all(order[i][0] != order[i + 1][0] for i in range(count - 1)):
            return order
    return None


def _sample_code_spec(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], np.ndarray, dict[int, int], dict[int, int], dict[int, int]] | None:
    n_colors = int(rng.integers(3, 6))
    colors = rng.choice(np.array(COLOR_POOL, dtype=int), size=n_colors, replace=False).astype(int).tolist()
    run_lengths = rng.choice(RUN_LENGTH_CHOICES, size=n_colors, replace=False).astype(int).tolist()

    segments: list[tuple[int, int]] = []
    width = 0
    for color, longest in zip(colors, run_lengths):
        extra = int(rng.integers(1, longest))
        segments.append((int(color), int(longest)))
        segments.append((int(color), extra))
        width += int(longest) + extra

    if width > 30:
        return None

    spare = 30 - width
    for idx in rng.permutation(n_colors):
        if spare <= 0:
            break
        if rng.random() >= 0.55:
            continue
        longest = int(run_lengths[int(idx)])
        upper = min(longest - 1, spare)
        if upper < 1:
            continue
        extra = int(rng.integers(1, upper + 1))
        segments.append((int(colors[int(idx)]), extra))
        spare -= extra

    ordered = _ordered_segments(segments, rng)
    if ordered is None:
        return None

    row_values: list[int] = []
    for color, seg_len in ordered:
        row_values.extend([int(color)] * int(seg_len))
    code_row = np.array(row_values, dtype=int)

    longest, totals, first_start = _scan_code_row(code_row)
    if len(longest) != n_colors:
        return None
    if any(totals[color] == longest[color] for color in colors):
        return None
    if [longest[color] for color in colors] != run_lengths:
        return None
    return colors, run_lengths, code_row, longest, totals, first_start


def _leftover_options(run_lengths: list[int]) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    sorted_runs = sorted(run_lengths)
    for idx, run_len in enumerate(run_lengths):
        target_rank = sorted_runs.index(run_len)
        for area in LEFTOVER_AREA_CHOICES:
            if area in run_lengths:
                continue
            areas = run_lengths.copy()
            areas[idx] = int(area)
            if len(set(areas)) != len(areas):
                continue
            rank = sorted(areas).index(int(area))
            if rank != target_rank:
                options.append((idx, int(area)))
    return options


def _true_assignment(objects: list[dict], expected: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for idx, obj in enumerate(objects):
        r, c = map(int, obj["coords"][0])
        mapping[idx] = int(expected[r, c])
    return mapping


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    separator_row, _code_row, longest, totals, first_start, objects = _analyze_input(inp)
    if separator_row != inp.shape[0] - 2:
        return False

    truth = _true_assignment(objects, expected)
    longest_assignment = _assignment_from_values(objects, longest)
    if longest_assignment != truth:
        return False

    direct_longest = sum(obj["area"] in set(longest.values()) for obj in objects)
    if direct_longest != len(objects) - 1:
        return False

    total_assignment = _assignment_from_values(objects, totals)
    if total_assignment == truth:
        return False

    objects_by_area = sorted(range(len(objects)), key=lambda idx: int(objects[idx]["area"]))
    colors_by_longest = [color for color, _length in sorted(longest.items(), key=lambda item: item[1])]
    rank_assignment = {obj_idx: colors_by_longest[pos] for pos, obj_idx in enumerate(objects_by_area)}
    if rank_assignment == truth:
        return False

    objects_by_left = sorted(range(len(objects)), key=lambda idx: (objects[idx]["center_c"], objects[idx]["center_r"]))
    colors_by_first = [color for color, _col in sorted(first_start.items(), key=lambda item: item[1])]
    left_assignment = {obj_idx: colors_by_first[pos] for pos, obj_idx in enumerate(objects_by_left)}
    if left_assignment == truth:
        return False

    return np.array_equal(main(inp), expected)


def _sample_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    sampled = _sample_code_spec(rng)
    if sampled is None:
        return None
    colors, run_lengths, code_row, _longest, _totals, _first_start = sampled

    options = _leftover_options(run_lengths)
    if not options:
        return None
    leftover_idx, leftover_area = options[int(rng.integers(len(options)))]

    areas = run_lengths.copy()
    areas[leftover_idx] = int(leftover_area)

    width = int(code_row.shape[0])
    top_h = int(rng.integers(8, 16))
    placed = _place_objects(areas, top_h, width, rng)
    if placed is None:
        return None
    top, placements = placed
    if len(placements) != len(colors):
        return None

    inp = np.full((top_h + 2, width), BG, dtype=int)
    inp[:top_h, :] = top
    inp[top_h, :] = SEPARATOR
    inp[top_h + 1, :] = code_row

    expected = inp.copy()
    for idx, coords in enumerate(placements):
        expected[coords[:, 0], coords[:, 1]] = int(colors[idx])

    if not _validate_semantics(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _sample_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
