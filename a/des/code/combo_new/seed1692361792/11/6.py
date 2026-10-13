from __future__ import annotations

import random

import numpy as np


_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_OBJECT_COUNT_CHOICES = (2, 3, 4)
_RECT_DIM_CHOICES = (2, 3, 4)
_DISTANCE_CHOICES = (1, 2, 3, 4)
_MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_filled_rectangle(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _expand_box(
    box: tuple[int, int, int, int], distance: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - distance),
        max(0, c0 - distance),
        min(h - 1, r1 + distance),
        min(w - 1, c1 + distance),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not _boxes_overlap(a, b):
        return 0
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _touches_edge(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = box
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != _BG]
    objects: list[dict[str, object]] = []
    for color in colors:
        comps = _components(grid == color)
        rects = [cells for cells in comps if len(cells) > 1 and _is_filled_rectangle(cells)]
        singles = [cells[0] for cells in comps if len(cells) == 1]
        assert len(rects) == 1, "each color must have exactly one solid rectangle"
        assert len(singles) == 2, "each color must have exactly two singleton markers"
        assert len(comps) == 3, "same-color pixels should split into one rectangle and two markers"

        p0, p1 = singles
        if p0[0] == p1[0]:
            distance = abs(p0[1] - p1[1]) - 1
        else:
            assert p0[1] == p1[1], "marker pair must lie on one row or one column"
            distance = abs(p0[0] - p1[0]) - 1
        assert 1 <= distance <= 4, "marker distance must be between 1 and 4"

        box = _bbox(rects[0])
        objects.append(
            {
                "color": color,
                "box": box,
                "distance": int(distance),
                "center_row_key": int(box[0] + box[2]),
                "center_col_key": int(box[1] + box[3]),
            }
        )
    objects.sort(key=lambda obj: (int(obj["center_row_key"]), int(obj["center_col_key"])))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each same-color singleton pair as a distance ruler, then erase the markers and expand the matching rectangle.

    Input structure:
    - The background is black.
    - Each foreground color appears in exactly three 4-connected components: one small solid rectangle and two isolated singleton markers.
    - For a given color, the two markers lie on the same row or the same column, with 1 to 4 black cells between them.

    Rule:
    - Remove every singleton marker.
    - For each color, read D as the number of black cells between its two markers.
    - Expand that color's rectangle by D cells upward, downward, leftward, and rightward, clipping to the grid boundary.
    - If expanded rectangles overlap, keep the color painted earlier by sorting rectangles from smaller center-row coordinate to larger one.

    Key insight:
    - The scattered same-color singleton pair is not noise: its gap directly encodes how far the matching rectangle must grow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    out = np.full(g.shape, _BG, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["box"]
        distance = int(obj["distance"])
        color = int(obj["color"])
        er0, ec0, er1, ec1 = _expand_box((r0, c0, r1, c1), distance, g.shape)
        patch = out[er0 : er1 + 1, ec0 : ec1 + 1]
        patch[patch == _BG] = color
    return out


def _mark_box(mask: np.ndarray, box: tuple[int, int, int, int], margin: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_shape(n_objects: int) -> tuple[int, int]:
    min_side = {2: 14, 3: 17, 4: 20}[n_objects]
    return random.randint(min_side, 30), random.randint(min_side, 30)


def _sample_descriptors(n_objects: int, force_overlap: bool) -> tuple[list[dict[str, object]], int | None]:
    colors = random.sample(_COLOR_POOL, n_objects)
    overlap_index = random.randint(1, n_objects - 1) if force_overlap else None
    specs: list[dict[str, object]] = []
    for idx, color in enumerate(colors):
        distance = random.choice(_DISTANCE_CHOICES)
        if force_overlap and overlap_index is not None and idx in {overlap_index - 1, overlap_index} and random.random() < 0.75:
            distance = random.choice((2, 3, 4))
        specs.append(
            {
                "color": int(color),
                "rect_h": random.choice(_RECT_DIM_CHOICES),
                "rect_w": random.choice(_RECT_DIM_CHOICES),
                "distance": int(distance),
                "orientation": random.choice(("h", "v")),
            }
        )
    return specs, overlap_index


def _choose_rectangle_layout(
    shape: tuple[int, int], specs: list[dict[str, object]], overlap_index: int | None
) -> list[dict[str, object]] | None:
    h, w = shape
    buffer_mask = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []
    last_center_row_key = -1

    for idx, spec in enumerate(specs):
        rect_h = int(spec["rect_h"])
        rect_w = int(spec["rect_w"])
        distance = int(spec["distance"])
        candidates: list[dict[str, object]] = []

        for r0 in range(h - rect_h + 1):
            r1 = r0 + rect_h - 1
            center_row_key = r0 + r1
            if center_row_key <= last_center_row_key:
                continue
            for c0 in range(w - rect_w + 1):
                c1 = c0 + rect_w - 1
                if buffer_mask[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)].any():
                    continue

                box = (r0, c0, r1, c1)
                expanded = _expand_box(box, distance, shape)
                if overlap_index is not None and idx == overlap_index:
                    if _intersection_area(expanded, placed[-1]["expanded_box"]) < 2:
                        continue
                    if any(_boxes_overlap(expanded, prev["expanded_box"]) for prev in placed[:-1]):
                        continue
                else:
                    if any(_boxes_overlap(expanded, prev["expanded_box"]) for prev in placed):
                        continue

                candidates.append(
                    {
                        **spec,
                        "box": box,
                        "expanded_box": expanded,
                        "center_row_key": int(center_row_key),
                        "center_col_key": int(c0 + c1),
                    }
                )

        if not candidates:
            return None

        edge_candidates = [cand for cand in candidates if _touches_edge(cand["expanded_box"], shape)]
        if edge_candidates and random.random() < 0.3:
            chosen = random.choice(edge_candidates)
        else:
            chosen = random.choice(candidates)

        _mark_box(buffer_mask, chosen["box"], margin=1)
        placed.append(chosen)
        last_center_row_key = int(chosen["center_row_key"])

    return placed


def _marker_candidates(
    shape: tuple[int, int], distance: int, orientation: str, buffer_mask: np.ndarray
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]]:
    h, w = shape
    candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]] = []
    if orientation == "h":
        span = distance + 2
        for r in range(h):
            for c0 in range(w - span + 1):
                c1 = c0 + distance + 1
                rr0 = max(0, r - 1)
                rr1 = min(h - 1, r + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w - 1, c1 + 1)
                if buffer_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                candidates.append(((r, c0), (r, c1), (rr0, cc0, rr1, cc1)))
    else:
        span = distance + 2
        for r0 in range(h - span + 1):
            r1 = r0 + distance + 1
            for c in range(w):
                rr0 = max(0, r0 - 1)
                rr1 = min(h - 1, r1 + 1)
                cc0 = max(0, c - 1)
                cc1 = min(w - 1, c + 1)
                if buffer_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                candidates.append(((r0, c), (r1, c), (rr0, cc0, rr1, cc1)))
    return candidates


def _place_markers(
    shape: tuple[int, int], specs: list[dict[str, object]]
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    h, w = shape
    buffer_mask = np.zeros((h, w), dtype=bool)
    for spec in specs:
        _mark_box(buffer_mask, spec["box"], margin=1)

    markers: list[tuple[tuple[int, int], tuple[int, int]] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: int(specs[idx]["distance"]), reverse=True)
    for idx in order:
        spec = specs[idx]
        candidates = _marker_candidates(shape, int(spec["distance"]), str(spec["orientation"]), buffer_mask)
        if not candidates:
            return None
        pair_a, pair_b, reserve = random.choice(candidates)
        rr0, cc0, rr1, cc1 = reserve
        buffer_mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        markers[idx] = (pair_a, pair_b)

    assert all(pair is not None for pair in markers)
    return [pair for pair in markers if pair is not None]


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]], markers: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for spec, pair in zip(specs, markers, strict=True):
        color = int(spec["color"])
        r0, c0, r1, c1 = spec["box"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        (pr0, pc0), (pr1, pc1) = pair
        grid[pr0, pc0] = color
        grid[pr1, pc1] = color
    return grid


def _try_generate() -> np.ndarray | None:
    n_objects = random.choice(_OBJECT_COUNT_CHOICES)
    force_overlap = random.choice((False, True))
    shape = _sample_shape(n_objects)
    specs, overlap_index = _sample_descriptors(n_objects, force_overlap)

    placed = _choose_rectangle_layout(shape, specs, overlap_index)
    if placed is None:
        return None

    markers = _place_markers(shape, placed)
    if markers is None:
        return None

    grid = _render_input(shape, placed, markers)

    parsed = _extract_objects(grid)
    if len(parsed) != n_objects:
        return None
    if force_overlap:
        overlaps = 0
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                overlaps += int(_boxes_overlap(placed[i]["expanded_box"], placed[j]["expanded_box"]))
        if overlaps != 1:
            return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
