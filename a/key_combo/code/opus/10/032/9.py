from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
PATTERN_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND):
        for comp in _components_of_color(grid, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue

            border = set()
            for cc in range(c0, c1 + 1):
                border.add((r0, cc))
                border.add((r1, cc))
            for rr in range(r0, r1 + 1):
                border.add((rr, c0))
                border.add((rr, c1))

            comp_set = set(comp)
            if comp_set != border:
                continue

            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "interior_area": (height - 2) * (width - 2),
                    "interior_shape": (height - 2, width - 2),
                }
            )
    return frames


def _largest_frame_crop(grid: np.ndarray) -> np.ndarray:
    frames = _find_frames(grid)
    assert frames, "expected at least one complete hollow rectangle"
    largest_area = max(int(frame["interior_area"]) for frame in frames)
    largest = [frame for frame in frames if int(frame["interior_area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest frame by interior area"
    r0, c0, r1, c1 = largest[0]["bbox"]
    return grid[r0 + 1 : r1, c0 + 1 : c1].copy()


def _scale_pixels(grid: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(grid, np.ones((factor, factor), dtype=grid.dtype))


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the interior of the largest hollow frame, then scale that crop by its own foreground count.

    Input structure:
    - A black background contains 2-3 complete 1-cell-thick monochrome rectangular frames.
    - Each frame encloses a small grey pattern on black interior cells, and extra same-colored noise may exist outside the frames.
    - The largest frame is defined by interior area, not by position or color id.

    Rule:
    - Detect every complete hollow monochrome rectangle and choose the unique one with maximum interior area.
    - Crop its strict interior to obtain pattern grid P.
    - Count the non-black cells in P, then scale every pixel of P into an N-by-N block where N is that count.

    Key insight:
    - The same cropped interior provides both the image content and the scale parameter, so the frame must be identified before the magnification factor is even knowable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    patch = _largest_frame_crop(g)
    scale = int(np.count_nonzero(patch != BACKGROUND))
    assert scale > 0, "expected foreground inside the largest frame"
    return _scale_pixels(patch, scale)


def _random_connected_pattern(height: int, width: int, size: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(40):
        start = random.choice(cells)
        chosen = {start}
        frontier = {nbr for nbr in _neighbors4(*start) if 0 <= nbr[0] < height and 0 <= nbr[1] < width}
        while len(chosen) < size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in chosen:
                continue
            chosen.add(cell)
            for nbr in _neighbors4(*cell):
                if 0 <= nbr[0] < height and 0 <= nbr[1] < width and nbr not in chosen:
                    frontier.add(nbr)
        if len(chosen) != size:
            continue
        patch = np.zeros((height, width), dtype=int)
        for r, c in chosen:
            patch[r, c] = PATTERN_COLOR
        return patch
    return None


def _sample_frame_specs(frame_count: int) -> list[dict[str, object]] | None:
    all_dims = [(h, w) for h in range(2, 7) for w in range(2, 7)]
    largest_fg = random.randint(2, 4)
    largest_candidates = [
        (h, w)
        for h, w in all_dims
        if h * w >= max(6, largest_fg) and h * largest_fg <= 30 and w * largest_fg <= 30
    ]
    if not largest_candidates:
        return None
    largest_h, largest_w = random.choice(largest_candidates)
    largest_area = largest_h * largest_w

    smaller_candidates = [(h, w) for h, w in all_dims if h * w < largest_area]
    area_buckets: dict[int, list[tuple[int, int]]] = {}
    for h, w in smaller_candidates:
        area_buckets.setdefault(h * w, []).append((h, w))
    if len(area_buckets) < frame_count - 1:
        return None

    chosen_small_areas = random.sample(sorted(area_buckets), frame_count - 1)
    specs: list[dict[str, object]] = [
        {
            "interior_h": largest_h,
            "interior_w": largest_w,
            "outer_h": largest_h + 2,
            "outer_w": largest_w + 2,
            "fg_count": largest_fg,
            "is_largest": True,
        }
    ]
    for area in chosen_small_areas:
        h, w = random.choice(area_buckets[area])
        specs.append(
            {
                "interior_h": h,
                "interior_w": w,
                "outer_h": h + 2,
                "outer_w": w + 2,
                "fg_count": random.randint(2, min(4, h * w)),
                "is_largest": False,
            }
        )
    random.shuffle(specs)
    return specs


def _expand_bbox(bbox: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - margin), max(0, c0 - margin), min(h - 1, r1 + margin), min(w - 1, c1 + margin)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_frames(specs: list[dict[str, object]], grid_shape: tuple[int, int]) -> list[dict[str, object]] | None:
    h, w = grid_shape
    placed: list[dict[str, object]] = []
    colors = random.sample(FRAME_COLORS, len(specs))
    order = list(range(len(specs)))
    random.shuffle(order)
    for index in order:
        spec = specs[index]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        placed_here = False
        for _ in range(80):
            r0 = random.randint(0, h - outer_h)
            c0 = random.randint(0, w - outer_w)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            grown = _expand_bbox(bbox, 1, grid_shape)
            if any(_boxes_overlap(grown, _expand_bbox(prev["bbox"], 1, grid_shape)) for prev in placed):
                continue
            spec = dict(spec)
            spec["bbox"] = bbox
            spec["color"] = colors[index]
            placed.append(spec)
            placed_here = True
            break
        if not placed_here:
            return None
    return placed


def _paint_frame(grid: np.ndarray, spec: dict[str, object], interior: np.ndarray) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    color = int(spec["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0 + 1 : r1, c0 + 1 : c1] = interior


def _available_noise_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    for bbox in boxes:
        r0, c0, r1, c1 = _expand_bbox(bbox, 1, shape)
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _try_place_noise_cells(
    grid: np.ndarray,
    available: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
) -> bool:
    h, w = grid.shape
    for base_r in range(h):
        for base_c in range(w):
            placed = []
            ok = True
            for dr, dc in cells:
                rr, cc = base_r + dr, base_c + dc
                if not (0 <= rr < h and 0 <= cc < w and available[rr, cc]):
                    ok = False
                    break
                placed.append((rr, cc))
            if not ok:
                continue
            if any(int(grid[rr, cc]) != BACKGROUND for rr, cc in placed):
                continue
            for rr, cc in placed:
                grid[rr, cc] = color
                available[rr, cc] = False
            return True
    return False


def _add_noise(grid: np.ndarray, frame_colors: list[int], boxes: list[tuple[int, int, int, int]]) -> None:
    available = _available_noise_mask(grid.shape, boxes)
    placements = []
    mode = random.choice(("singletons", "segments", "mixed"))
    if mode in ("segments", "mixed"):
        placements.extend(
            random.sample(
                [
                    [(0, 0), (0, 1)],
                    [(0, 0), (1, 0)],
                    [(0, 0), (0, 1), (0, 2)],
                    [(0, 0), (1, 0), (2, 0)],
                    [(0, 0), (0, 1), (1, 0)],
                    [(0, 0), (1, 0), (1, 1)],
                ],
                k=random.randint(1, 2),
            )
        )
    if mode in ("singletons", "mixed"):
        placements.extend([[(0, 0)] for _ in range(random.randint(1, 4))])
    random.shuffle(placements)
    for shape in placements:
        color = random.choice(frame_colors)
        _try_place_noise_cells(grid, available, shape, color)


def _expected_output_from_spec(interior: np.ndarray) -> np.ndarray:
    scale = int(np.count_nonzero(interior != BACKGROUND))
    return _scale_pixels(interior, scale)


def _validate_semantics(
    grid: np.ndarray,
    expected_frames: list[tuple[int, int, int, int]],
    expected_largest_bbox: tuple[int, int, int, int],
    expected_interior: np.ndarray,
) -> bool:
    frames = _find_frames(grid)
    if len(frames) != len(expected_frames):
        return False
    found_boxes = {tuple(frame["bbox"]) for frame in frames}
    if found_boxes != set(expected_frames):
        return False
    areas = [int(frame["interior_area"]) for frame in frames]
    if areas.count(max(areas)) != 1:
        return False
    actual = max(frames, key=lambda frame: int(frame["interior_area"]))
    if tuple(actual["bbox"]) != expected_largest_bbox:
        return False
    r0, c0, r1, c1 = expected_largest_bbox
    cropped = grid[r0 + 1 : r1, c0 + 1 : c1]
    if not np.array_equal(cropped, expected_interior):
        return False
    scale = int(np.count_nonzero(cropped != BACKGROUND))
    if scale < 2 or scale > 4:
        return False
    expected = _expected_output_from_spec(expected_interior)
    if max(expected.shape) > 30:
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((2, 3))
    specs = _sample_frame_specs(frame_count)
    if specs is None:
        return None

    grid_h = random.randint(14, 20)
    grid_w = random.randint(14, 20)
    placed = _place_frames(specs, (grid_h, grid_w))
    if placed is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected_largest_bbox = None
    expected_interior = None
    frame_boxes: list[tuple[int, int, int, int]] = []

    for spec in placed:
        interior_h = int(spec["interior_h"])
        interior_w = int(spec["interior_w"])
        fg_count = int(spec["fg_count"])
        interior = _random_connected_pattern(interior_h, interior_w, fg_count)
        if interior is None:
            return None
        _paint_frame(grid, spec, interior)
        bbox = tuple(spec["bbox"])
        frame_boxes.append(bbox)
        if bool(spec["is_largest"]):
            expected_largest_bbox = bbox
            expected_interior = interior.copy()

    assert expected_largest_bbox is not None
    assert expected_interior is not None

    _add_noise(grid, [int(spec["color"]) for spec in placed], frame_boxes)

    if not _validate_semantics(grid, frame_boxes, expected_largest_bbox, expected_interior):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
