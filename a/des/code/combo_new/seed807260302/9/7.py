from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_SIZE_RANGE = (20, 30)
RECT_SIDE_RANGE = (7, 9)
PIPE_GAP_RANGE = (2, 5)
GENERATION_ATTEMPTS = 500


STAMP_PATTERNS: dict[int, tuple[np.ndarray, ...]] = {
    3: (
        np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    ),
    4: (
        np.array(
            [[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]],
            dtype=bool,
        ),
        np.array(
            [[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]],
            dtype=bool,
        ),
        np.array(
            [[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1]],
            dtype=bool,
        ),
        np.array(
            [[1, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 1]],
            dtype=bool,
        ),
        np.array(
            [[1, 0, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 0, 1]],
            dtype=bool,
        ),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _infer_stamp(patch: np.ndarray, color: int) -> np.ndarray:
    assert patch.shape[0] >= 3 and patch.shape[1] >= 3, "region must fit a stamp"
    assert np.all((patch == color) | (patch == BG)), "region interior may contain only fill color and black"
    for size in (3, 4):
        if patch.shape[0] < size or patch.shape[1] < size:
            continue
        stamp_zone = np.zeros(patch.shape, dtype=bool)
        stamp_zone[:size, :size] = True
        if np.any((patch == BG) & ~stamp_zone):
            continue
        stamp = patch[:size, :size]
        if np.any(stamp == BG) and np.any(stamp == color):
            return stamp == color
    raise AssertionError("could not identify a 3x3 or 4x4 top-left stamp")


def _find_regions(g: np.ndarray) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for color in [int(c) for c in np.unique(g) if int(c) not in (BG, GREY)]:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        stamp = _infer_stamp(patch, color)
        regions.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "stamp": stamp,
            }
        )
    assert regions, "expected at least one colored region"
    return regions


def _pipe_pairs(g: np.ndarray, regions: list[dict[str, object]]) -> list[tuple[int, int]]:
    color_to_region = {int(region["color"]): i for i, region in enumerate(regions)}
    pairs: list[tuple[int, int]] = []
    used_regions: set[int] = set()

    for comp in _components(g == GREY):
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        straight_horizontal = len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1))
        straight_vertical = len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1))
        assert straight_horizontal or straight_vertical, "each grey pipe must be one straight segment"

        adjacent: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    color = int(g[nr, nc])
                    if color in color_to_region:
                        adjacent.add(color_to_region[color])
        assert len(adjacent) == 2, "each pipe must connect exactly two colored regions"
        a, b = tuple(adjacent)
        assert a not in used_regions and b not in used_regions, "each region may use at most one pipe"
        used_regions.add(a)
        used_regions.add(b)
        pairs.append((a, b))

    return pairs


def _draw_stamp(out: np.ndarray, target: dict[str, object], source_stamp: np.ndarray, clear_size: int) -> None:
    color = int(target["color"])
    r0, c0, r1, c1 = target["bbox"]  # type: ignore[misc]
    assert r0 + clear_size - 1 <= r1 and c0 + clear_size - 1 <= c1, "incoming stamp must fit target"
    out[r0 : r0 + clear_size, c0 : c0 + clear_size] = color
    size = int(source_stamp.shape[0])
    out[r0 : r0 + size, c0 : c0 + size] = np.where(source_stamp, color, BG)


def _swap_connected_stamps(
    grid: np.ndarray,
    regions: list[dict[str, object]],
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for a, b in pairs:
        stamp_a = regions[a]["stamp"]  # type: ignore[assignment]
        stamp_b = regions[b]["stamp"]  # type: ignore[assignment]
        assert isinstance(stamp_a, np.ndarray) and isinstance(stamp_b, np.ndarray)
        clear_size = max(int(stamp_a.shape[0]), int(stamp_b.shape[0]))
        _draw_stamp(out, regions[a], stamp_b, clear_size)
        _draw_stamp(out, regions[b], stamp_a, clear_size)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange each connected pair of rectangular regions' hidden top-left stamp patterns.

    Input structure:
    - Black is the background and grey forms one-pixel straight pipes.
    - Several differently colored near-solid rectangular regions sit on the background.
    - Each region has a 3x3 or 4x4 stamp in its interior upper-left corner, made only from black and that region's fill color.
    - Every grey pipe is horizontal or vertical and touches exactly two different regions; each region touches at most one pipe.

    Rule:
    - Identify each colored rectangular region and read its upper-left black/fill stamp as a binary pattern.
    - For every pair of regions joined by a grey pipe, draw each region's stamp into the other region's upper-left corner.
    - Recolor the incoming stamp's filled cells with the target region color, clear its black cells to black, and preserve all grey pipes.
    - Regions without a grey pipe keep their original stamps.

    Key insight:
    - The small black/fill corner pattern is a transferable stamp, and the grey pipe is the visible cue that selects the two regions whose stamps must be exchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    regions = _find_regions(g)
    pairs = _pipe_pairs(g, regions)
    return _swap_connected_stamps(g, regions, pairs)


def _rect_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _can_place(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or occupied[r, c]:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _mark(cells: list[tuple[int, int]], occupied: np.ndarray) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _sample_stamp() -> np.ndarray:
    size = random.choice((3, 4))
    return random.choice(STAMP_PATTERNS[size]).copy()


def _sample_different_stamp(stamp: np.ndarray) -> np.ndarray:
    for _ in range(30):
        other = _sample_stamp()
        if other.shape != stamp.shape or not np.array_equal(other, stamp):
            return other
    other_size = 4 if stamp.shape[0] == 3 else 3
    return STAMP_PATTERNS[other_size][0].copy()


def _paint_region(grid: np.ndarray, region: dict[str, object]) -> None:
    color = int(region["color"])
    r0, c0, r1, c1 = region["bbox"]  # type: ignore[misc]
    stamp = region["stamp"]
    assert isinstance(stamp, np.ndarray)
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    k = int(stamp.shape[0])
    grid[r0 : r0 + k, c0 : c0 + k] = np.where(stamp, color, BG)


def _place_pair(
    occupied: np.ndarray,
    color_a: int,
    color_b: int,
) -> tuple[dict[str, object], dict[str, object], list[tuple[int, int]]] | None:
    h_grid, w_grid = occupied.shape
    for _ in range(120):
        orient = random.choice(("horizontal", "vertical"))
        gap = random.randint(*PIPE_GAP_RANGE)
        if orient == "horizontal":
            height = random.randint(*RECT_SIDE_RANGE)
            width_a = random.randint(*RECT_SIDE_RANGE)
            width_b = random.randint(*RECT_SIDE_RANGE)
            if width_a + gap + width_b + 2 > w_grid or height + 2 > h_grid:
                continue
            top = random.randint(1, h_grid - height - 1)
            left = random.randint(1, w_grid - width_a - gap - width_b - 1)
            box_a = (top, left, top + height - 1, left + width_a - 1)
            box_b = (
                top,
                left + width_a + gap,
                top + height - 1,
                left + width_a + gap + width_b - 1,
            )
            pipe_r = top + height - 2
            pipe = [(pipe_r, c) for c in range(box_a[3] + 1, box_b[1])]
        else:
            width = random.randint(*RECT_SIDE_RANGE)
            height_a = random.randint(*RECT_SIDE_RANGE)
            height_b = random.randint(*RECT_SIDE_RANGE)
            if height_a + gap + height_b + 2 > h_grid or width + 2 > w_grid:
                continue
            top = random.randint(1, h_grid - height_a - gap - height_b - 1)
            left = random.randint(1, w_grid - width - 1)
            box_a = (top, left, top + height_a - 1, left + width - 1)
            box_b = (
                top + height_a + gap,
                left,
                top + height_a + gap + height_b - 1,
                left + width - 1,
            )
            pipe_c = left + width - 2
            pipe = [(r, pipe_c) for r in range(box_a[2] + 1, box_b[0])]

        footprint = _rect_cells(box_a) + _rect_cells(box_b) + pipe
        if not _can_place(footprint, occupied):
            continue
        stamp_a = _sample_stamp()
        stamp_b = _sample_different_stamp(stamp_a)
        region_a = {"color": color_a, "bbox": box_a, "stamp": stamp_a}
        region_b = {"color": color_b, "bbox": box_b, "stamp": stamp_b}
        _mark(footprint, occupied)
        return region_a, region_b, pipe
    return None


def _place_unpaired(occupied: np.ndarray, color: int) -> dict[str, object] | None:
    h_grid, w_grid = occupied.shape
    for _ in range(120):
        height = random.randint(*RECT_SIDE_RANGE)
        width = random.randint(*RECT_SIDE_RANGE)
        if height + 2 > h_grid or width + 2 > w_grid:
            continue
        top = random.randint(1, h_grid - height - 1)
        left = random.randint(1, w_grid - width - 1)
        box = (top, left, top + height - 1, left + width - 1)
        footprint = _rect_cells(box)
        if not _can_place(footprint, occupied):
            continue
        _mark(footprint, occupied)
        return {"color": color, "bbox": box, "stamp": _sample_stamp()}
    return None


def _sample_counts() -> tuple[int, int]:
    pair_count = random.choice((1, 1, 2))
    unpaired_count = random.choice((0, 1, 2))
    if 2 * pair_count + unpaired_count > len(COLORS):
        unpaired_count = len(COLORS) - 2 * pair_count
    return pair_count, unpaired_count


def _gen_attempt() -> np.ndarray | None:
    pair_count, unpaired_count = _sample_counts()
    total_regions = 2 * pair_count + unpaired_count
    min_size = 24 if total_regions >= 5 else GRID_SIZE_RANGE[0]
    height = random.randint(min_size, GRID_SIZE_RANGE[1])
    width = random.randint(min_size, GRID_SIZE_RANGE[1])
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    colors = random.sample(COLORS, total_regions)

    regions: list[dict[str, object]] = []
    pairs: list[tuple[int, int]] = []
    pipes: list[list[tuple[int, int]]] = []
    color_idx = 0

    for _ in range(pair_count):
        placed = _place_pair(occupied, colors[color_idx], colors[color_idx + 1])
        if placed is None:
            return None
        region_a, region_b, pipe = placed
        idx_a = len(regions)
        idx_b = idx_a + 1
        regions.extend([region_a, region_b])
        pairs.append((idx_a, idx_b))
        pipes.append(pipe)
        color_idx += 2

    for _ in range(unpaired_count):
        region = _place_unpaired(occupied, colors[color_idx])
        if region is None:
            return None
        regions.append(region)
        color_idx += 1

    order = list(range(len(regions)))
    random.shuffle(order)
    for idx in order:
        _paint_region(grid, regions[idx])
    for pipe in pipes:
        for r, c in pipe:
            grid[r, c] = GREY

    expected = _swap_connected_stamps(grid, regions, pairs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stamp-pipe exchange input")
