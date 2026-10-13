from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
DIVIDER = 5

PANEL_HEIGHT_RANGE = (8, 14)
PANEL_WIDTH_RANGE = (6, 14)
RECT_COUNT_CHOICES = (2, 3, 4)
SLOT_COUNT_CHOICES = (1, 2, 3)
GAP = 1
GENERATION_ATTEMPTS = 400
VALIDATION_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

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
            components.append(np.array(coords, dtype=int))
    return components


def _find_divider(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [
        c
        for c in range(w)
        if np.all(grid[:, c] == DIVIDER) and c > 0 and (w - c - 1) == c
    ]
    assert len(candidates) == 1, "expected one solid gray divider splitting equal-width panels"
    return int(candidates[0])


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    divider = _find_divider(grid)
    left = grid[:, :divider].copy()
    right = grid[:, divider + 1 :].copy()
    assert left.shape == right.shape and left.size > 0
    assert set(np.unique(left)).issubset({BLACK, BLUE})
    assert set(np.unique(right)).issubset({BLACK, RED})
    return left, right


def _extract_rectangles(left: np.ndarray) -> list[tuple[int, int, int, int]]:
    components = _components_mask(left == BLUE)
    assert 2 <= len(components) <= 4, "expected 2 to 4 blue rectangles"

    rects: list[tuple[int, int, int, int]] = []
    covered = np.zeros(left.shape, dtype=bool)
    for coords in components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "rectangles must be at least 3x3"
        assert int(coords.shape[0]) == height * width, "each blue component must be a solid rectangle"
        assert np.all(left[r0 : r1 + 1, c0 : c1 + 1] == BLUE)
        rects.append((r0, c0, r1, c1))
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

    assert np.all(left[~covered] == BLACK)
    rects.sort()
    return rects


def _render_expected(
    left: np.ndarray,
    right: np.ndarray,
    rects: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[bool]]:
    out = left.copy()
    hit_flags: list[bool] = []

    for r0, c0, r1, c1 in rects:
        hit = bool(np.any(right[r0 : r1 + 1, c0 : c1 + 1] == RED))
        hit_flags.append(hit)
        if hit:
            out[r0 : r1 + 1, c0 : c1 + 1] = RED
        else:
            out[r0 + 1 : r1, c0 + 1 : c1] = GREEN

    return out, hit_flags


def _render_global_all_red(
    left: np.ndarray,
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = left.copy()
    for r0, c0, r1, c1 in rects:
        out[r0 : r1 + 1, c0 : c1 + 1] = RED
    return out


def _render_all_interior_green(
    left: np.ndarray,
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = left.copy()
    for r0, c0, r1, c1 in rects:
        out[r0 + 1 : r1, c0 + 1 : c1] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the divider and recolor each left-panel rectangle according to whether the right-panel mask overlaps that rectangle.

    Input structure:
    - The grid has a black background and one solid gray divider column splitting two equal-width panels.
    - The left panel contains 2 to 4 separated blue solid rectangles, each at least 3x3.
    - The right panel is an aligned red foreground mask on black background.

    Rule:
    - Remove the gray divider and use the left panel as the output canvas.
    - For each blue rectangle in the left panel, inspect the same coordinate range in the right panel.
    - If any red mask pixel falls inside that rectangle's aligned range, recolor the entire rectangle red.
    - Otherwise keep the blue border and recolor only the strict interior green; background stays black.

    Key insight:
    - The overlap test is per rectangle, not global: one touched rectangle turns fully red while untouched rectangles in the same grid still get only an interior recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    left, right = _extract_panels(g)
    rects = _extract_rectangles(left)
    out, _hit_flags = _render_expected(left, right, rects)
    return out


def _sample_axis(total_range: tuple[int, int], slot_count: int) -> list[int] | None:
    total_min, total_max = total_range
    feasible_totals = [
        total
        for total in range(total_min, total_max + 1)
        if total - GAP * (slot_count - 1) >= 3 * slot_count
    ]
    if not feasible_totals:
        return None

    total = random.choice(feasible_totals)
    payload = total - GAP * (slot_count - 1)
    parts = [3] * slot_count
    for _ in range(payload - 3 * slot_count):
        parts[random.randrange(slot_count)] += 1
    random.shuffle(parts)
    return parts


def _sample_layout(rect_count: int) -> tuple[int, int, list[int], list[int]] | None:
    layouts = [
        (slot_rows, slot_cols)
        for slot_rows in SLOT_COUNT_CHOICES
        for slot_cols in SLOT_COUNT_CHOICES
        if slot_rows * slot_cols >= rect_count
    ]
    random.shuffle(layouts)

    for slot_rows, slot_cols in layouts:
        row_sizes = _sample_axis(PANEL_HEIGHT_RANGE, slot_rows)
        col_sizes = _sample_axis(PANEL_WIDTH_RANGE, slot_cols)
        if row_sizes is None or col_sizes is None:
            continue
        panel_h = sum(row_sizes) + GAP * (slot_rows - 1)
        panel_w = sum(col_sizes) + GAP * (slot_cols - 1)
        return panel_h, panel_w, row_sizes, col_sizes
    return None


def _slot_boxes(row_sizes: list[int], col_sizes: list[int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    r0 = 0
    for height in row_sizes:
        c0 = 0
        for width in col_sizes:
            boxes.append((r0, c0, r0 + height - 1, c0 + width - 1))
            c0 += width + GAP
        r0 += height + GAP
    return boxes


def _sample_rectangles(
    boxes: list[tuple[int, int, int, int]],
    rect_count: int,
) -> list[tuple[int, int, int, int]]:
    chosen_boxes = random.sample(boxes, rect_count)
    rects: list[tuple[int, int, int, int]] = []

    for br0, bc0, br1, bc1 in chosen_boxes:
        slot_h = br1 - br0 + 1
        slot_w = bc1 - bc0 + 1
        rect_h = random.randint(3, slot_h)
        rect_w = random.randint(3, slot_w)
        r0 = random.randint(br0, br1 - rect_h + 1)
        c0 = random.randint(bc0, bc1 - rect_w + 1)
        rects.append((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))

    rects.sort()
    return rects


def _paint_rectangles(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    panel = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        panel[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    return panel


def _rect_mask(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rects:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _connected_subset(cells: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    allowed = set(cells)
    start = random.choice(cells)
    chosen = {start}
    frontier = [nb for nb in _neighbors4(*start) if nb in allowed]

    while len(chosen) < size:
        frontier = [cell for cell in frontier if cell not in chosen]
        if not frontier:
            break
        cell = random.choice(frontier)
        chosen.add(cell)
        for nb in _neighbors4(*cell):
            if nb in allowed and nb not in chosen:
                frontier.append(nb)

    return sorted(chosen)


def _paint_fragment(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = RED


def _background_anchor_cells(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    rect_mask = _rect_mask(shape, rects)
    anchors: set[tuple[int, int]] = set()

    for r0, c0, r1, c1 in rects:
        for r in range(max(0, r0 - 1), min(h, r1 + 2)):
            for c in range(max(0, c0 - 1), min(w, c1 + 2)):
                if not rect_mask[r, c]:
                    anchors.add((r, c))

    if not anchors:
        anchors = {(r, c) for r in range(h) for c in range(w) if not rect_mask[r, c]}
    return sorted(anchors)


def _grow_background_fragment(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    start: tuple[int, int],
    size: int,
) -> list[tuple[int, int]]:
    rect_mask = _rect_mask(shape, rects)
    h, w = shape
    chosen = {start}
    frontier = [nb for nb in _neighbors4(*start) if 0 <= nb[0] < h and 0 <= nb[1] < w and not rect_mask[nb]]

    while len(chosen) < size:
        frontier = [cell for cell in frontier if cell not in chosen]
        if not frontier:
            break
        cell = random.choice(frontier)
        chosen.add(cell)
        for nb in _neighbors4(*cell):
            if 0 <= nb[0] < h and 0 <= nb[1] < w and not rect_mask[nb] and nb not in chosen:
                frontier.append(nb)

    return sorted(chosen)


def _build_mask(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    hit_flags: list[bool],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=int)

    for (r0, c0, r1, c1), hit in zip(rects, hit_flags):
        if not hit:
            continue
        rect_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        max_size = min(4, max(1, len(rect_cells) // 3))
        fragment_size = random.randint(1, max_size)
        _paint_fragment(mask, _connected_subset(rect_cells, fragment_size))

    background_cells = _background_anchor_cells(shape, rects)
    extra_fragments = random.randint(1, 3)
    for _ in range(extra_fragments):
        if not background_cells:
            break
        start = random.choice(background_cells)
        fragment_size = random.randint(1, 3)
        _paint_fragment(mask, _grow_background_fragment(shape, rects, start, fragment_size))

    return mask


def _choose_hit_flags(rect_count: int) -> list[bool]:
    hit_count = random.randint(1, rect_count - 1)
    hit_indices = set(random.sample(range(rect_count), hit_count))
    return [idx in hit_indices for idx in range(rect_count)]


def _assemble_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = left.shape
    grid = np.zeros((h, 2 * w + 1), dtype=int)
    grid[:, :w] = left
    grid[:, w] = DIVIDER
    grid[:, w + 1 :] = right
    return grid


def _validate_case(
    inp: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    hit_flags: list[bool],
) -> bool:
    left, right = _extract_panels(inp)
    parsed_rects = _extract_rectangles(left)
    expected, parsed_hits = _render_expected(left, right, parsed_rects)
    if parsed_rects != rects or parsed_hits != hit_flags:
        return False
    if not any(parsed_hits) or all(parsed_hits):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(expected, _render_global_all_red(left, parsed_rects)):
        return False
    if np.array_equal(expected, _render_all_interior_green(left, parsed_rects)):
        return False

    for (r0, c0, r1, c1), hit in zip(parsed_rects, parsed_hits):
        patch = actual[r0 : r1 + 1, c0 : c1 + 1]
        if hit:
            if not np.all(patch == RED):
                return False
        else:
            border_ok = (
                np.all(patch[0, :] == BLUE)
                and np.all(patch[-1, :] == BLUE)
                and np.all(patch[:, 0] == BLUE)
                and np.all(patch[:, -1] == BLUE)
            )
            interior_ok = np.all(patch[1:-1, 1:-1] == GREEN)
            if not (border_ok and interior_ok):
                return False

    return True


def _generate_case() -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[bool]]:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    layout = _sample_layout(rect_count)
    if layout is None:
        raise RuntimeError("failed to sample a feasible panel layout")

    panel_h, panel_w, row_sizes, col_sizes = layout
    boxes = _slot_boxes(row_sizes, col_sizes)
    rects = _sample_rectangles(boxes, rect_count)
    hit_flags = _choose_hit_flags(rect_count)

    left = _paint_rectangles((panel_h, panel_w), rects)
    right = _build_mask((panel_h, panel_w), rects, hit_flags)
    return _assemble_input(left, right), rects, hit_flags


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp, rects, hit_flags = _generate_case()
        if _validate_case(inp, rects, hit_flags):
            return inp
    raise RuntimeError("failed to generate a valid overlap-driven panel input")


if __name__ == "__main__":
    for _ in range(VALIDATION_ATTEMPTS):
        sample = generate_input()
        result = main(sample)
        assert sample.ndim == 2 and result.ndim == 2
        assert max(sample.shape) <= 30 and max(result.shape) <= 30
