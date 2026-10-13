from __future__ import annotations

import random

import numpy as np


BG = 0
DEFAULT_GREY = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _is_rectangle(cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return (r1 - r0 + 1) * (c1 - c0 + 1) == len(cells)


def _piece_from_cells(cells: set[tuple[int, int]], color: int) -> dict:
    rel = _normalize(cells)
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    return {
        "color": int(color),
        "rel": rel,
        "rel_set": set(rel),
        "h": int(h),
        "w": int(w),
        "area": int(len(rel)),
    }


def _patch_for_piece(piece: dict) -> np.ndarray:
    patch = np.full((piece["h"], piece["w"]), BG, dtype=int)
    for r, c in piece["rel"]:
        patch[r, c] = int(piece["color"])
    return patch


def _find_divider(grid: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != BG:
            candidates.append((int(r), int(row[0])))
    assert len(candidates) == 1
    return candidates[0]


def _extract_slots(grid: np.ndarray, divider_row: int, grey: int) -> list[dict]:
    mask = np.zeros_like(grid, dtype=bool)
    mask[divider_row + 1 :, :] = grid[divider_row + 1 :, :] == int(grey)
    slots: list[dict] = []
    for coords in _connected_components(mask):
        cells = set(coords)
        r0, c0, r1, c1 = _bbox(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(sub[border] == int(grey))
        assert np.all(sub[~border] == BG)
        assert len(cells) == int(border.sum())
        slots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_r": r0 + 1,
                "inner_c": c0 + 1,
                "inner_h": h - 2,
                "inner_w": w - 2,
                "area": (h - 2) * (w - 2),
            }
        )
    slots.sort(key=lambda slot: (slot["bbox"][1], slot["bbox"][0]))
    assert slots
    areas = [slot["area"] for slot in slots]
    assert len(set(areas)) == len(areas)
    assert areas == sorted(areas)
    return slots


def _extract_pieces(grid: np.ndarray, divider_row: int, grey: int) -> list[dict]:
    top_mask = (grid[:divider_row, :] != BG) & (grid[:divider_row, :] != int(grey))
    pieces: list[dict] = []
    for coords in _connected_components(top_mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1
        color = colors.pop()
        piece = _piece_from_cells(set(coords), color)
        r0, c0, r1, c1 = _bbox(coords)
        piece["bbox"] = (r0, c0, r1, c1)
        pieces.append(piece)
    pieces.sort(key=lambda piece: (piece["bbox"][0], piece["bbox"][1], piece["color"]))
    return pieces


def _valid_assemblies(piece_a: dict, piece_b: dict) -> list[dict]:
    a = piece_a["rel_set"]
    b = piece_b["rel_set"]
    seen = set()
    variants: list[dict] = []
    for dr in range(-piece_b["h"], piece_a["h"] + 1):
        for dc in range(-piece_b["w"], piece_a["w"] + 1):
            shifted_b = {(r + dr, c + dc) for r, c in b}
            if a & shifted_b:
                continue
            union = a | shifted_b
            r0, c0, r1, c1 = _bbox(union)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h * w != len(union):
                continue
            a_norm = tuple(sorted((r - r0, c - c0) for r, c in a))
            b_norm = tuple(sorted((r - r0, c - c0) for r, c in shifted_b))
            key = (h, w, a_norm, b_norm)
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "area": int(len(union)),
                    "h": int(h),
                    "w": int(w),
                    "colored_cells": (
                        (int(piece_a["color"]), a_norm),
                        (int(piece_b["color"]), b_norm),
                    ),
                }
            )
    return variants


def _solve_pairing(pieces: list[dict], slots: list[dict]) -> list[tuple[int, int, dict]]:
    assert len(pieces) == 2 * len(slots)
    slot_by_area = {slot["area"]: slot for slot in slots}
    options: dict[tuple[int, int], list[dict]] = {}
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            valid = []
            for variant in _valid_assemblies(pieces[i], pieces[j]):
                slot = slot_by_area.get(variant["area"])
                if slot is None:
                    continue
                if variant["h"] <= slot["inner_h"] and variant["w"] <= slot["inner_w"]:
                    valid.append(variant)
            if valid:
                options[(i, j)] = valid

    used = [False] * len(pieces)
    remaining_areas = set(slot_by_area)
    solutions: list[list[tuple[int, int, dict]]] = []

    def search(chosen: list[tuple[int, int, dict]]) -> None:
        if len(solutions) > 1:
            return
        try:
            i = next(idx for idx, taken in enumerate(used) if not taken)
        except StopIteration:
            if not remaining_areas:
                solutions.append(list(chosen))
            return

        candidates = []
        for j in range(i + 1, len(pieces)):
            if used[j]:
                continue
            variants = [v for v in options.get((i, j), []) if v["area"] in remaining_areas]
            if variants:
                candidates.append((j, variants))
        candidates.sort(key=lambda item: (len(item[1]), pieces[item[0]]["bbox"][0], pieces[item[0]]["bbox"][1]))

        used[i] = True
        for j, variants in candidates:
            used[j] = True
            for variant in variants:
                area = variant["area"]
                if area not in remaining_areas:
                    continue
                remaining_areas.remove(area)
                chosen.append((i, j, variant))
                search(chosen)
                chosen.pop()
                remaining_areas.add(area)
            used[j] = False
        used[i] = False

    search([])
    assert len(solutions) == 1
    return sorted(solutions[0], key=lambda item: item[2]["area"])


def _render_output(grid: np.ndarray, divider_row: int, grey: int, slots: list[dict], solution: list[tuple[int, int, dict]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    clear_mask = np.zeros_like(out, dtype=bool)
    clear_mask[:divider_row, :] = (out[:divider_row, :] != BG) & (out[:divider_row, :] != int(grey))
    out[clear_mask] = BG
    for slot in slots:
        r0, c0, r1, c1 = slot["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
    slot_by_area = {slot["area"]: slot for slot in slots}
    for _i, _j, variant in solution:
        slot = slot_by_area[variant["area"]]
        for color, coords in variant["colored_cells"]:
            for r, c in coords:
                out[slot["inner_r"] + r, slot["inner_c"] + c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary scattered fragments into rectangles, then place the assembled rectangles into the lower slots by area order.

    Input structure:
    - The background is black, and a single grey horizontal divider splits the grid into an upper fragment area and a lower slot area.
    - Above the divider are several separated monochrome connected components, with a different color for each component.
    - Below the divider are empty grey rectangular frames whose interiors are black and whose left-to-right order already follows increasing interior area.

    Rule:
    - Find the unique pairing of upper components such that each pair can form a solid rectangle when one piece is translated without rotation or overlap.
    - Measure the area of each completed rectangle and sort the pairs from smallest area to largest area.
    - Clear the original upper fragments, then place each assembled colored rectangle into the correspondingly ordered lower slot with the rectangle's top-left corner aligned to the slot interior's top-left corner.

    Key insight:
    - The hard part is not color matching but shape complementarity: each correct pair is defined by becoming a perfect rectangle after translation, and only after recovering those hidden rectangles does the lower slot ordering make sense.
    """
    g = np.asarray(grid, dtype=int)
    divider_row, grey = _find_divider(g)
    slots = _extract_slots(g, divider_row, grey)
    pieces = _extract_pieces(g, divider_row, grey)
    solution = _solve_pairing(pieces, slots)
    return _render_output(g, divider_row, grey, slots, solution)


def _grow_connected_subset(h: int, w: int, size: int) -> set[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while len(cells) < size:
        if not frontier:
            return set()
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _make_pair(h: int, w: int, color_a: int, color_b: int) -> tuple[dict, dict, dict] | None:
    full = {(r, c) for r in range(h) for c in range(w)}
    area = h * w
    for _ in range(PAIR_ATTEMPTS):
        target = random.randint(2, area - 2)
        a_cells = _grow_connected_subset(h, w, target)
        if not a_cells:
            continue
        b_cells = full - a_cells
        if not _is_connected(b_cells):
            continue
        if _is_rectangle(a_cells) or _is_rectangle(b_cells):
            continue
        piece_a = _piece_from_cells(a_cells, color_a)
        piece_b = _piece_from_cells(b_cells, color_b)
        variants = _valid_assemblies(piece_a, piece_b)
        if len(variants) != 1:
            continue
        return (
            piece_a,
            piece_b,
            {
                "area": area,
                "h": h,
                "w": w,
                "colored_cells": (
                    (int(color_a), tuple(sorted(a_cells))),
                    (int(color_b), tuple(sorted(b_cells))),
                ),
            },
        )
    return None


def _candidate_rectangles(pair_count: int) -> list[tuple[int, int]]:
    if pair_count == 2:
        heights = range(2, 7)
        widths = range(3, 8)
        max_area = 24
    elif pair_count == 3:
        heights = range(2, 6)
        widths = range(3, 7)
        max_area = 20
    else:
        heights = range(2, 6)
        widths = range(2, 6)
        max_area = 16
    return [
        (h, w)
        for h in heights
        for w in widths
        if 6 <= h * w <= max_area
    ]


def _pick_rectangles(pair_count: int) -> list[tuple[int, int]] | None:
    candidates = _candidate_rectangles(pair_count)
    for _ in range(100):
        picked = random.sample(candidates, pair_count)
        areas = [h * w for h, w in picked]
        if len(set(areas)) != pair_count:
            continue
        picked.sort(key=lambda hw: hw[0] * hw[1])
        gaps = [random.randint(1, 2) for _ in range(pair_count - 1)]
        outer_width = sum(w + 2 for _h, w in picked) + sum(gaps)
        if outer_width > 28:
            continue
        return picked
    return None


def _place_slots(grid: np.ndarray, divider_row: int, grey: int, rectangles: list[dict]) -> list[dict] | None:
    outer_heights = [rect["h"] + 2 for rect in rectangles]
    outer_widths = [rect["w"] + 2 for rect in rectangles]
    gaps = [random.randint(1, 2) for _ in range(len(rectangles) - 1)]
    total_width = sum(outer_widths) + sum(gaps)
    if total_width > grid.shape[1] - 2:
        return None
    bottom_h = grid.shape[0] - divider_row - 1
    max_outer_h = max(outer_heights)
    if max_outer_h > bottom_h - 1:
        return None
    left = random.randint(1, grid.shape[1] - total_width - 1)
    top = divider_row + 1 + random.randint(0, bottom_h - max_outer_h)
    slots: list[dict] = []
    cursor = left
    for rect, outer_h, outer_w in zip(rectangles, outer_heights, outer_widths):
        r0 = top
        c0 = cursor
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0, c0 : c1 + 1] = grey
        grid[r1, c0 : c1 + 1] = grey
        grid[r0 : r1 + 1, c0] = grey
        grid[r0 : r1 + 1, c1] = grey
        slots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_r": r0 + 1,
                "inner_c": c0 + 1,
                "inner_h": rect["h"],
                "inner_w": rect["w"],
                "area": rect["area"],
            }
        )
        cursor = c1 + 1
        if len(slots) - 1 < len(gaps):
            cursor += gaps[len(slots) - 1]
    return slots


def _place_top_pieces(grid: np.ndarray, top_h: int, pieces: list[dict]) -> bool:
    occupied = np.zeros((top_h, grid.shape[1]), dtype=bool)
    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        ph, pw = piece["h"], piece["w"]
        patch = _patch_for_piece(piece)
        positions: list[tuple[int, int]] = []
        for r in range(1, top_h - ph):
            for c in range(1, grid.shape[1] - pw):
                rr0 = max(0, r - 1)
                rr1 = min(top_h, r + ph + 1)
                cc0 = max(0, c - 1)
                cc1 = min(grid.shape[1], c + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                positions.append((r, c))
        if not positions:
            return False
        r, c = random.choice(positions)
        occupied[r : r + ph, c : c + pw] = True
        mask = patch != BG
        grid[r : r + ph, c : c + pw][mask] = patch[mask]
    return True


def _expected_from_design(inp: np.ndarray, divider_row: int, grey: int, slots: list[dict], assembled: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    clear_mask = np.zeros_like(out, dtype=bool)
    clear_mask[:divider_row, :] = (out[:divider_row, :] != BG) & (out[:divider_row, :] != int(grey))
    out[clear_mask] = BG
    for slot in slots:
        r0, c0, r1, c1 = slot["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
    for slot, rect in zip(slots, assembled):
        for color, coords in rect["colored_cells"]:
            for r, c in coords:
                out[slot["inner_r"] + r, slot["inner_c"] + c] = int(color)
    return out


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3, 4))
    rectangles_hw = _pick_rectangles(pair_count)
    if rectangles_hw is None:
        return None

    colors = random.sample(PIECE_COLORS, 2 * pair_count)
    rectangles: list[dict] = []
    pieces: list[dict] = []
    for idx, (h, w) in enumerate(rectangles_hw):
        pair = _make_pair(h, w, colors[2 * idx], colors[2 * idx + 1])
        if pair is None:
            return None
        piece_a, piece_b, assembled = pair
        rectangles.append(assembled)
        pieces.extend([piece_a, piece_b])

    min_bottom_h = max(rect["h"] + 2 for rect in rectangles)
    top_h_min = max(10, max(piece["h"] for piece in pieces) + 4)
    top_h_max = 30 - 1 - min_bottom_h
    if top_h_min > top_h_max:
        return None
    top_h = random.randint(top_h_min, top_h_max)
    height = top_h + 1 + random.randint(min_bottom_h, 30 - top_h - 1)

    min_width = max(
        sum(rect["w"] + 2 for rect in rectangles) + (pair_count - 1) + 2,
        max(piece["w"] for piece in pieces) + 4,
        14,
    )
    if min_width > 30:
        return None
    width = random.randint(min_width, 30)

    grid = np.full((height, width), BG, dtype=int)
    divider_row = top_h
    grey = DEFAULT_GREY
    grid[divider_row, :] = grey

    slots = _place_slots(grid, divider_row, grey, rectangles)
    if slots is None:
        return None
    if not _place_top_pieces(grid, top_h, pieces):
        return None

    expected = _expected_from_design(grid, divider_row, grey, slots, rectangles)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    extracted_slots = _extract_slots(grid, divider_row, grey)
    extracted_pieces = _extract_pieces(grid, divider_row, grey)
    try:
        solution = _solve_pairing(extracted_pieces, extracted_slots)
    except AssertionError:
        return None
    if len(solution) != pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
