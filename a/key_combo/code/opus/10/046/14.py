import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

REGION_COUNT_CHOICES = (3, 4, 5, 5)
INTERIOR_HEIGHT_RANGE = (8, 13)
INTERIOR_WIDTH_RANGE = (8, 13)
MARGIN_RANGE = (1, 3)
MIN_REGION_SIDE = 2
MAX_ATTEMPTS = 600
MISLEADING_MOTIF_PROB = 0.6


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
            comps.append(np.array(coords, dtype=int))
    return comps


def _enclosed_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions = []
    for coords in _components(g != GREY):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        box = _bbox(coords)
        regions.append({"coords": coords, "bbox": box, "area": int(coords.shape[0])})
    return regions


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _diagonal_tile(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    out = np.zeros((2 * h, 2 * w), dtype=int)
    by_diag: dict[int, int] = {}
    for r, c in np.argwhere(patch != BLACK):
        color = int(patch[r, c])
        diag = int(r - c)
        prev = by_diag.get(diag)
        assert prev is None or prev == color, "same diagonal must not encode conflicting colors"
        by_diag[diag] = color
        for k in range(min(2 * h - int(r), 2 * w - int(c))):
            rr = int(r) + k
            cc = int(c) + k
            target = int(out[rr, cc])
            assert target in (BLACK, color), "diagonal copies must remain unambiguous"
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely smallest gray-enclosed region and diagonally tile its interior motif on a doubled canvas.

    Input structure:
    - The grid has black background and gray orthogonal line segments that form several closed rectangular regions.
    - Each closed region is a 4-connected non-gray interior containing black cells plus a small same-color motif.
    - Region interiors are separated only by gray boundaries; the exterior background touches the grid border.

    Rule:
    - Detect every 4-connected non-gray component that does not touch the grid border; these are the enclosed regions.
    - Compute each enclosed region's area as its total count of non-gray interior cells, including black background and colored motif cells.
    - Select the unique minimum-area region, crop its interior bounding box as pattern block P, then create a black canvas of size (2*P.height, 2*P.width).
    - For every non-black pixel of P at (r, c), copy that color to all output positions (r+k, c+k) while they stay in bounds.

    Key insight:
    - The gray lines first define a candidate pool of enclosed regions, and only the smallest region survives to drive the later diagonal-copy step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = _enclosed_regions(g)
    assert regions, "expected at least one enclosed region"

    min_area = min(region["area"] for region in regions)
    chosen = [region for region in regions if region["area"] == min_area]
    assert len(chosen) == 1, "smallest enclosed region must be unique"

    patch = _crop_bbox(g, chosen[0]["bbox"])
    assert not np.any(patch == GREY), "cropped region should contain only interior cells"
    return _diagonal_tile(patch)


def _draw_rect_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _split_options(region: tuple[int, int, int, int]) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = region
    out: list[tuple[str, int]] = []
    if r1 - r0 + 1 >= 2 * MIN_REGION_SIDE + 1:
        for row in range(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE + 1):
            out.append(("h", row))
    if c1 - c0 + 1 >= 2 * MIN_REGION_SIDE + 1:
        for col in range(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE + 1):
            out.append(("v", col))
    return out


def _apply_split(
    region: tuple[int, int, int, int], orientation: str, pos: int
) -> tuple[list[tuple[int, int, int, int]], tuple]:
    r0, c0, r1, c1 = region
    if orientation == "h":
        return [(r0, c0, pos - 1, c1), (pos + 1, c0, r1, c1)], ("h", pos, c0, c1)
    return [(r0, c0, r1, pos - 1), (r0, pos + 1, r1, c1)], ("v", pos, r0, r1)


def _choose_split(
    regions: list[tuple[int, int, int, int]],
    orientations_used: set[str],
    splits_left: int,
) -> tuple[int, str, int] | None:
    options: list[tuple[int, str, int]] = []
    for idx, region in enumerate(regions):
        for orientation, pos in _split_options(region):
            options.append((idx, orientation, pos))
    if not options:
        return None

    if splits_left == 1 and len(orientations_used) == 1:
        missing = "v" if "h" in orientations_used else "h"
        forced = [option for option in options if option[1] == missing]
        if forced:
            options = forced

    return random.choice(options)


def _make_connected_motif(height: int, width: int, n_cells: int) -> list[tuple[int, int]]:
    for _ in range(80):
        row_choices = list(range(1, height - 1)) if height >= 3 else list(range(height))
        col_choices = list(range(1, width - 1)) if width >= 3 else list(range(width))
        cells = {(random.choice(row_choices), random.choice(col_choices))}
        while len(cells) < n_cells:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) == n_cells:
            return sorted(cells)
    raise RuntimeError("failed to build motif")


def _motif_sizes(regions: list[tuple[int, int, int, int]]) -> list[int]:
    sizes = [random.randint(2, min(4, (r1 - r0 + 1) * (c1 - c0 + 1))) for r0, c0, r1, c1 in regions]
    if random.random() < MISLEADING_MOTIF_PROB:
        areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in regions]
        min_idx = int(np.argmin(areas))
        sizes[min_idx] = min(4, areas[min_idx])
        distractors = [i for i, area in enumerate(areas) if i != min_idx and area >= 2]
        if distractors:
            sizes[random.choice(distractors)] = 2
    return sizes


def _place_motifs(grid: np.ndarray, regions: list[tuple[int, int, int, int]]) -> None:
    colors = random.sample(PALETTE, len(regions))
    sizes = _motif_sizes(regions)
    for (r0, c0, r1, c1), color, n_cells in zip(regions, colors, sizes):
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        motif = _make_connected_motif(height, width, n_cells)
        for rr, cc in motif:
            grid[r0 + rr, c0 + cc] = int(color)


def _validate_generated(grid: np.ndarray) -> bool:
    regions = _enclosed_regions(grid)
    if not 3 <= len(regions) <= 5:
        return False

    areas = [region["area"] for region in regions]
    if len(set(areas)) != len(areas):
        return False

    for region in regions:
        patch = _crop_bbox(grid, region["bbox"])
        if np.any(patch == GREY):
            return False
        if region["area"] != int(patch.shape[0] * patch.shape[1]):
            return False
        colors = sorted({int(v) for v in patch.flatten()} - {BLACK})
        if len(colors) != 1:
            return False
        if not 2 <= int(np.count_nonzero(patch != BLACK)) <= 4:
            return False

    min_idx = int(np.argmin(areas))
    expected = _diagonal_tile(_crop_bbox(grid, regions[min_idx]["bbox"]))
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    region_target = random.choice(REGION_COUNT_CHOICES)
    interior_h = random.randint(*INTERIOR_HEIGHT_RANGE)
    interior_w = random.randint(*INTERIOR_WIDTH_RANGE)
    margin_t = random.randint(*MARGIN_RANGE)
    margin_b = random.randint(*MARGIN_RANGE)
    margin_l = random.randint(*MARGIN_RANGE)
    margin_r = random.randint(*MARGIN_RANGE)

    height = margin_t + interior_h + 2 + margin_b
    width = margin_l + interior_w + 2 + margin_r
    if max(height, width) > 30:
        return None

    frame_r0 = margin_t
    frame_c0 = margin_l
    frame_r1 = frame_r0 + interior_h + 1
    frame_c1 = frame_c0 + interior_w + 1

    regions = [(frame_r0 + 1, frame_c0 + 1, frame_r1 - 1, frame_c1 - 1)]
    segments: list[tuple] = []
    orientations_used: set[str] = set()

    for split_idx in range(region_target - 1):
        chosen = _choose_split(regions, orientations_used, region_target - 2 - split_idx)
        if chosen is None:
            return None
        region_idx, orientation, pos = chosen
        region = regions.pop(region_idx)
        children, segment = _apply_split(region, orientation, pos)
        regions.extend(children)
        segments.append(segment)
        orientations_used.add(orientation)

    areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in regions]
    if len(set(areas)) != len(areas):
        return None

    grid = np.zeros((height, width), dtype=int)
    _draw_rect_border(grid, frame_r0, frame_c0, frame_r1, frame_c1)
    for segment in segments:
        if segment[0] == "h":
            _, row, c0, c1 = segment
            grid[row, c0 : c1 + 1] = GREY
        else:
            _, col, r0, r1 = segment
            grid[r0 : r1 + 1, col] = GREY

    _place_motifs(grid, regions)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
