from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ORIENTATIONS = ("h", "v", "d1", "d2")

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
RECT_COUNT_CHOICES = (2, 3)
RECT_OUTER_SIDE_RANGE = range(5, 9)
MARKER_EXTENSION_RANGE = (1, 3)
PLACEMENT_MARGIN = 0
PLACEMENT_PADDING = 1
GENERATION_RETRIES = 500


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bbox(cell: Coord, bbox: BBox) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _is_boundary(cell: Coord, bbox: BBox) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return _in_bbox(cell, bbox) and (r in (r0, r1) or c in (c0, c1))


def _frame_bbox(coords: np.ndarray, shape: tuple[int, int]) -> BBox | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], ring):
        return None
    return r0, c0, r1, c1


def _find_rectangles(grid: np.ndarray) -> list[BBox]:
    rects: list[BBox] = []
    for coords in _components(grid == BLUE):
        bbox = _frame_bbox(coords, grid.shape)
        assert bbox is not None, "blue components must be 1-cell-thick hollow rectangles"
        rects.append(bbox)
    rects.sort()
    assert rects, "expected at least one blue rectangle"
    return rects


def _line_cells(start: Coord, end: Coord) -> list[Coord] | None:
    r0, c0 = start
    r1, c1 = end
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc == 0:
        return [start]
    if dr == 0:
        step = (0, 1 if dc > 0 else -1)
        length = abs(dc)
    elif dc == 0:
        step = (1 if dr > 0 else -1, 0)
        length = abs(dr)
    elif abs(dr) == abs(dc):
        step = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
        length = abs(dr)
    else:
        return None
    return [(r0 + k * step[0], c0 + k * step[1]) for k in range(length + 1)]


def _split_regions(bbox: BBox, barrier: set[Coord]) -> list[list[Coord]]:
    r0, c0, r1, c1 = bbox
    open_mask = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=bool)
    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            if (rr, cc) not in barrier:
                open_mask[rr - r0 - 1, cc - c0 - 1] = True

    comps: list[list[Coord]] = []
    seen = np.zeros_like(open_mask, dtype=bool)
    h, w = open_mask.shape
    for r in range(h):
        for c in range(w):
            if not open_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr + r0 + 1, cc + c0 + 1))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            cells.sort()
            comps.append(cells)
    comps.sort(key=lambda cells: cells[0])
    return comps


def _path_info_for_rectangle(rect: BBox, markers: tuple[Coord, Coord], rects: list[BBox]) -> dict[str, object] | None:
    start, end = markers
    line = _line_cells(start, end)
    if line is None:
        return None

    if _in_bbox(start, rect) or _in_bbox(end, rect):
        return None

    inside_indices = [idx for idx, cell in enumerate(line) if _in_bbox(cell, rect)]
    if not inside_indices:
        return None
    first_idx = inside_indices[0]
    last_idx = inside_indices[-1]
    if inside_indices != list(range(first_idx, last_idx + 1)):
        return None
    rect_line = line[first_idx : last_idx + 1]
    if len(rect_line) < 3:
        return None
    if not _is_boundary(rect_line[0], rect) or not _is_boundary(rect_line[-1], rect):
        return None
    if any(_is_boundary(cell, rect) for cell in rect_line[1:-1]):
        return None

    for other in rects:
        if other == rect:
            continue
        if any(_in_bbox(cell, other) for cell in line):
            return None

    regions = _split_regions(rect, set(rect_line))
    if len(regions) != 2:
        return None
    return {
        "markers": frozenset((start, end)),
        "line": tuple(line),
        "rect_line": tuple(rect_line),
        "regions": tuple(tuple(region) for region in regions),
    }


def _pair_rectangles_with_markers(rects: list[BBox], markers: list[Coord]) -> list[dict[str, object]]:
    assert len(markers) == 2 * len(rects), "expected exactly one red marker pair per rectangle"
    marker_indices = range(len(markers))
    candidates: dict[int, list[tuple[frozenset[int], dict[str, object]]]] = {idx: [] for idx in range(len(rects))}

    for rect_idx, rect in enumerate(rects):
        for i in marker_indices:
            for j in range(i + 1, len(markers)):
                info = _path_info_for_rectangle(rect, (markers[i], markers[j]), rects)
                if info is not None:
                    candidates[rect_idx].append((frozenset((i, j)), info))
        assert candidates[rect_idx], "each rectangle must admit at least one marker pair"

    order = sorted(range(len(rects)), key=lambda idx: len(candidates[idx]))
    solutions: list[dict[int, dict[str, object]]] = []

    def search(pos: int, used: frozenset[int], chosen: dict[int, dict[str, object]]) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            if len(used) == len(markers):
                solutions.append(chosen.copy())
            return

        rect_idx = order[pos]
        for marker_ids, info in candidates[rect_idx]:
            if marker_ids & used:
                continue
            chosen[rect_idx] = info
            search(pos + 1, used | marker_ids, chosen)
            chosen.pop(rect_idx, None)

    search(0, frozenset(), {})
    assert len(solutions) == 1, "marker pairing must be unique"
    return [solutions[0][idx] for idx in range(len(rects))]


def _analyze_input(grid: np.ndarray) -> tuple[list[BBox], list[dict[str, object]]]:
    rects = _find_rectangles(grid)
    red_coords = np.argwhere(grid == RED)
    markers = [tuple(map(int, coord)) for coord in red_coords]
    assert markers, "expected red markers"
    for coords in _components(grid == RED):
        assert len(coords) == 1, "red markers must be isolated single pixels"
    pairing = _pair_rectangles_with_markers(rects, markers)
    return rects, pairing


def _paint_path(out: np.ndarray, line: tuple[Coord, ...]) -> None:
    for r, c in line:
        value = int(out[r, c])
        if value == BLUE:
            out[r, c] = GREEN
        elif value in (BG, RED):
            out[r, c] = RED
        elif value == GREEN:
            continue
        else:
            raise AssertionError("path may only traverse background, markers, or blue borders")


def _main_region_first(regions: list[list[Coord]]) -> tuple[list[Coord], list[Coord]]:
    ranked = sorted(regions, key=lambda cells: (-len(cells), cells[0][0], cells[0][1]))
    return ranked[0], ranked[1]


def main(grid: np.ndarray) -> np.ndarray:
    """Draw each red marker pair through its rectangle, then fill the larger split side with gray checkerboard and the smaller side yellow.

    Input structure:
    - Black background contains 2-3 blue 1-cell-thick hollow rectangles whose interiors are empty.
    - Each rectangle has exactly one pair of isolated red marker pixels outside it.
    - The two markers of a pair lie on one horizontal, vertical, or 45-degree diagonal line whose segment passes through that rectangle and no other rectangle.

    Rule:
    - For every red marker pair, draw the full endpoint-inclusive queen-move line segment between them.
    - On that segment, blue border cells become green, black cells become red, and red marker cells stay red.
    - After all lines are drawn, the red cells inside each rectangle split its interior into exactly two 4-connected regions.
    - The larger region is the main region; if the areas tied, the region containing the lexicographically smallest cell would be main.
    - Fill the main region with a gray checkerboard anchored at its smallest cell, and fill the other region solid yellow.
    - Leave rectangle borders, path cells, and everything outside the rectangles otherwise unchanged.

    Key insight:
    - The red path is both a recoloring action and a topological cut: only after drawing it can the solver determine the two new interior regions and which one gets checkerboard versus solid fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rects, pairing = _analyze_input(g)

    out = g.copy()
    for info in pairing:
        _paint_path(out, info["line"])

    for rect, info in zip(rects, pairing):
        regions = _split_regions(rect, set(info["rect_line"]))
        assert len(regions) == 2, "each rectangle interior must split into two regions"
        main_region, secondary = _main_region_first(regions)
        for r, c in secondary:
            out[r, c] = YELLOW
        start_r, start_c = main_region[0]
        for r, c in main_region:
            if (r - start_r + c - start_c) % 2 == 0:
                out[r, c] = GRAY
            else:
                assert out[r, c] == BG

    return out


def _draw_frame(patch: np.ndarray, bbox: BBox) -> None:
    r0, c0, r1, c1 = bbox
    patch[r0, c0 : c1 + 1] = BLUE
    patch[r1, c0 : c1 + 1] = BLUE
    patch[r0 : r1 + 1, c0] = BLUE
    patch[r0 : r1 + 1, c1] = BLUE


def _local_candidates(height: int, width: int, orientation: str) -> list[dict[str, object]]:
    bbox = (0, 0, height - 1, width - 1)
    raw_lines: list[list[Coord]] = []

    if orientation == "h":
        for r in range(1, height - 1):
            raw_lines.append([(r, c) for c in range(width)])
    elif orientation == "v":
        for c in range(1, width - 1):
            raw_lines.append([(r, c) for r in range(height)])
    elif orientation == "d1":
        for diff in range(-(width - 1), height):
            cells = [(r, c) for r in range(height) for c in range(width) if r - c == diff]
            cells.sort()
            raw_lines.append(cells)
    elif orientation == "d2":
        for total in range(height + width - 1):
            cells = [(r, total - r) for r in range(height) if 0 <= total - r < width]
            cells.sort()
            raw_lines.append(cells)
    else:
        raise AssertionError(f"unknown orientation {orientation}")

    candidates: list[dict[str, object]] = []
    for line in raw_lines:
        if len(line) < 3:
            continue
        if not _is_boundary(line[0], bbox) or not _is_boundary(line[-1], bbox):
            continue
        if any(_is_boundary(cell, bbox) for cell in line[1:-1]):
            continue
        regions = _split_regions(bbox, set(line))
        if len(regions) != 2:
            continue
        sizes = sorted(len(region) for region in regions)
        if sizes[0] == sizes[1]:
            continue
        candidates.append({"line": tuple(line), "regions": tuple(tuple(region) for region in regions)})
    return candidates


def _build_patch(orientation: str) -> dict[str, object] | None:
    dims: list[tuple[int, int]] = []
    for height in RECT_OUTER_SIDE_RANGE:
        for width in RECT_OUTER_SIDE_RANGE:
            if orientation == "h" and height < 6:
                continue
            if orientation == "v" and width < 6:
                continue
            dims.append((height, width))
    random.shuffle(dims)

    for height, width in dims:
        candidates = _local_candidates(height, width, orientation)
        if not candidates:
            continue
        choice = random.choice(candidates)
        line = choice["line"]
        step = (line[1][0] - line[0][0], line[1][1] - line[0][1])
        before = random.randint(*MARKER_EXTENSION_RANGE)
        after = random.randint(*MARKER_EXTENSION_RANGE)
        marker_a = (line[0][0] - before * step[0], line[0][1] - before * step[1])
        marker_b = (line[-1][0] + after * step[0], line[-1][1] + after * step[1])

        min_r = min(0, marker_a[0], marker_b[0])
        min_c = min(0, marker_a[1], marker_b[1])
        max_r = max(height - 1, marker_a[0], marker_b[0])
        max_c = max(width - 1, marker_a[1], marker_b[1])

        patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
        shift_r = -min_r
        shift_c = -min_c
        rect_bbox = (shift_r, shift_c, shift_r + height - 1, shift_c + width - 1)
        _draw_frame(patch, rect_bbox)

        marker_a_shifted = (marker_a[0] + shift_r, marker_a[1] + shift_c)
        marker_b_shifted = (marker_b[0] + shift_r, marker_b[1] + shift_c)
        patch[marker_a_shifted] = RED
        patch[marker_b_shifted] = RED

        rect_line_shifted = tuple((r + shift_r, c + shift_c) for r, c in line)
        marker_pair = tuple(sorted((marker_a_shifted, marker_b_shifted)))
        full_line_shifted = tuple(_line_cells(marker_pair[0], marker_pair[1]))
        return {
            "patch": patch,
            "bbox": rect_bbox,
            "markers": frozenset((marker_a_shifted, marker_b_shifted)),
            "line": rect_line_shifted,
            "path": full_line_shifted,
            "orientation": orientation,
        }
    return None


def _boxes_separated(a: BBox, b: BBox, pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_orientations(count: int) -> list[str]:
    if count == 2 and random.random() < 0.7:
        return random.sample(list(ORIENTATIONS), 2)
    if count == 3 and random.random() < 0.8:
        chosen = [random.choice(("h", "v")), random.choice(("d1", "d2")), random.choice(ORIENTATIONS)]
        random.shuffle(chosen)
        return chosen
    return [random.choice(ORIENTATIONS) for _ in range(count)]


def _place_patches(height: int, width: int, items: list[dict[str, object]]) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(items)
    placed_boxes: list[BBox] = []
    order = sorted(range(len(items)), key=lambda idx: int(items[idx]["patch"].shape[0]) * int(items[idx]["patch"].shape[1]), reverse=True)

    for idx in order:
        patch = items[idx]["patch"]
        ph, pw = map(int, patch.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(PLACEMENT_MARGIN, height - ph - PLACEMENT_MARGIN + 1):
            for c0 in range(PLACEMENT_MARGIN, width - pw - PLACEMENT_MARGIN + 1):
                bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if all(_boxes_separated(bbox, other, PLACEMENT_PADDING) for other in placed_boxes):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        positions[idx] = pos
        placed_boxes.append((pos[0], pos[1], pos[0] + ph - 1, pos[1] + pw - 1))

    return [pos for pos in positions if pos is not None]


def _shift_bbox(bbox: BBox, dr: int, dc: int) -> BBox:
    return bbox[0] + dr, bbox[1] + dc, bbox[2] + dr, bbox[3] + dc


def _validate_against_metadata(grid: np.ndarray, items: list[dict[str, object]], positions: list[tuple[int, int]]) -> bool:
    expected: dict[BBox, tuple[frozenset[Coord], frozenset[Coord]]] = {}
    for item, (r0, c0) in zip(items, positions):
        rect_bbox = _shift_bbox(item["bbox"], r0, c0)
        markers = frozenset((mr + r0, mc + c0) for mr, mc in item["markers"])
        line = frozenset((lr + r0, lc + c0) for lr, lc in item["path"])
        expected[rect_bbox] = (markers, line)

    try:
        rects, pairing = _analyze_input(grid)
    except AssertionError:
        return False

    if set(rects) != set(expected):
        return False

    actual: dict[BBox, tuple[frozenset[Coord], frozenset[Coord]]] = {}
    for rect, info in zip(rects, pairing):
        actual[rect] = (
            info["markers"],
            frozenset(info["line"]),
        )

    return actual == expected


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(RECT_COUNT_CHOICES)
    items: list[dict[str, object]] = []
    for orientation in _sample_orientations(count):
        item = _build_patch(orientation)
        if item is None:
            return None
        items.append(item)

    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    positions = _place_patches(height, width, items)
    if positions is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for item, (r0, c0) in zip(items, positions):
        patch = item["patch"]
        ph, pw = patch.shape
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        assert np.all(region == BG)
        region[patch != BG] = patch[patch != BG]

    if not _validate_against_metadata(grid, items, positions):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-split rectangle instance")
