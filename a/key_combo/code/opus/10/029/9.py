from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
TEAL = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and grid[nr, nc] == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _green_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components_of_color(grid, GREEN)
    assert len(comps) == 2, "expected exactly two green components"
    boxes = []
    for comp in comps:
        assert _is_solid_rectangle(comp), "green components must be solid rectangles"
        boxes.append(_bbox(comp))
    return boxes


def _bridge_info(grid: np.ndarray) -> tuple[str, list[tuple[int, int]], tuple[int, int, int, int], tuple[int, int, int, int]]:
    b1, b2 = _green_rectangles(grid)

    horizontal = None
    bridge: list[tuple[int, int]] = []

    if min(b1[2], b2[2]) >= max(b1[0], b2[0]):
        left, right = sorted([b1, b2], key=lambda b: (b[1], b[0]))
        gap_c0 = left[3] + 1
        gap_c1 = right[1] - 1
        row0 = max(left[0], right[0])
        row1 = min(left[2], right[2])
        if gap_c0 <= gap_c1 and row0 <= row1:
            candidate = [(r, c) for r in range(row0, row1 + 1) for c in range(gap_c0, gap_c1 + 1)]
            if candidate:
                horizontal = True
                bridge = candidate
                b1, b2 = left, right

    if horizontal is None and min(b1[3], b2[3]) >= max(b1[1], b2[1]):
        top, bottom = sorted([b1, b2], key=lambda b: (b[0], b[1]))
        gap_r0 = top[2] + 1
        gap_r1 = bottom[0] - 1
        col0 = max(top[1], bottom[1])
        col1 = min(top[3], bottom[3])
        if gap_r0 <= gap_r1 and col0 <= col1:
            candidate = [(r, c) for r in range(gap_r0, gap_r1 + 1) for c in range(col0, col1 + 1)]
            if candidate:
                horizontal = False
                bridge = candidate
                b1, b2 = top, bottom

    assert horizontal is not None and bridge, "rectangles must be horizontally or vertically aligned with a gap"
    assert all(grid[r, c] == BLACK for r, c in bridge), "bridge region must be black in the input"
    return ("horizontal" if horizontal else "vertical"), bridge, b1, b2


def _flood_after_bridge(grid: np.ndarray, bridge: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in bridge:
        out[r, c] = TEAL

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(out == BLUE)]
    assert seeds, "expected at least one blue seed"

    seen = np.zeros(out.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for seed in seeds:
        seen[seed] = True
        q.append(seed)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(out.shape, nr, nc) or seen[nr, nc]:
                continue
            if out[nr, nc] not in (BLACK, TEAL):
                continue
            seen[nr, nc] = True
            if out[nr, nc] == BLACK:
                out[nr, nc] = BLUE
            q.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bridge the gap between two aligned green rectangles, then flood-fill through the opened bottleneck.

    Input structure:
    - The grid has a black background, exactly two solid green rectangles, scattered red obstacle pixels,
      and one or more blue seed pixels near one side of the rectangle pair.
    - The two green rectangles are aligned either horizontally or vertically, with a black gap of width 1-3
      cells between them along their shared overlap band.
    - Red pixels and green rectangles are blockers; the bridge gap itself is initially black.

    Rule:
    - Detect the two green rectangle components and compute the black cells geometrically between them:
      for a horizontal pair, use the intersection of their row ranges and the empty columns between them;
      for a vertical pair, use the intersection of their column ranges and the empty rows between them.
    - Recolor every such gap cell teal to form the bridge.
    - Starting from the original blue seed pixels, flood-fill with 4-connectivity through black and teal cells only.
      Reached black cells become blue; teal bridge cells stay teal; green rectangles, red obstacles, and original blue
      seeds keep their colors.

    Key insight:
    - The crucial signal is the geometric gap between the two rectangles: it marks the bottleneck corridor that must be
      highlighted first and then used as part of a constrained 4-neighbor flood.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _orientation, bridge, _box1, _box2 = _bridge_info(g)
    return _flood_after_bridge(g, bridge)


def _reachable_mask(
    grid: np.ndarray,
    *,
    blocked: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    blocked = blocked or set()
    mask = np.zeros(grid.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(grid == BLUE):
        r = int(r)
        c = int(c)
        mask[r, c] = True
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc) or mask[nr, nc]:
                continue
            if (nr, nc) in blocked or grid[nr, nc] != BLACK:
                continue
            mask[nr, nc] = True
            q.append((nr, nc))
    return mask


def _paint_bar(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    region: tuple[int, int, int, int],
    orientation: str,
) -> bool:
    r0, c0, r1, c1 = region
    if r0 > r1 or c0 > c1:
        return False
    if orientation == "horizontal":
        length = random.randint(1, min(3, c1 - c0 + 1))
        r = random.randint(r0, r1)
        c = random.randint(c0, c1 - length + 1)
        cells = [(r, cc) for cc in range(c, c + length)]
    else:
        length = random.randint(1, min(3, r1 - r0 + 1))
        r = random.randint(r0, r1 - length + 1)
        c = random.randint(c0, c1)
        cells = [(rr, c) for rr in range(r, r + length)]
    if any(cell in reserved or grid[cell] != BLACK for cell in cells):
        return False
    for cell in cells:
        grid[cell] = RED
        reserved.add(cell)
    return True


def _scatter_local_obstacles(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    regions: list[tuple[int, int, int, int]],
) -> None:
    for region in regions:
        n_bars = random.randint(1, 3)
        tries = 0
        placed = 0
        while tries < 20 and placed < n_bars:
            tries += 1
            if _paint_bar(grid, reserved, region, random.choice(["horizontal", "vertical"])):
                placed += 1


def _build_horizontal_case() -> tuple[np.ndarray, dict[str, object]]:
    rect_h = random.randint(3, 5)
    rect_w1 = random.randint(3, 5)
    rect_w2 = random.randint(3, 5)
    gap_w = random.randint(1, 3)
    total_w = rect_w1 + gap_w + rect_w2
    min_h = max(12, rect_h + 6)
    min_w = max(13, total_w + 6)
    assert min_h <= 18 and min_w <= 19

    h = random.randint(min_h, 18)
    w = random.randint(min_w, 19)
    grid = np.full((h, w), BLACK, dtype=int)

    r0 = random.randint(2, h - rect_h - 3)
    c0 = random.randint(2, w - total_w - 3)
    c1 = c0 + rect_w1
    c2 = c1 + gap_w

    grid[r0 : r0 + rect_h, c0:c1] = GREEN
    grid[r0 : r0 + rect_h, c2 : c2 + rect_w2] = GREEN

    if c0 > 0:
        grid[r0 : r0 + rect_h, :c0] = RED
    if c2 + rect_w2 < w:
        grid[r0 : r0 + rect_h, c2 + rect_w2 :] = RED

    bridge = [(r, c) for r in range(r0, r0 + rect_h) for c in range(c1, c2)]
    seed_side = random.choice(["top", "bottom"])
    seed_r = r0 - 1 if seed_side == "top" else r0 + rect_h
    seed_c = random.randint(c1, c2 - 1)
    grid[seed_r, seed_c] = BLUE

    reserved = set(bridge)
    reserved.update((r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c1))
    reserved.update((r, c) for r in range(r0, r0 + rect_h) for c in range(c2, c2 + rect_w2))
    reserved.add((seed_r, seed_c))
    reserved.update((r, seed_c) for r in range(h))

    top_region = (0, 0, max(0, r0 - 2), w - 1)
    bottom_region = (min(h - 1, r0 + rect_h + 1), 0, h - 1, w - 1)
    _scatter_local_obstacles(grid, reserved, [top_region, bottom_region])

    meta = {
        "orientation": "horizontal",
        "bridge": set(bridge),
        "target_side": "bottom" if seed_side == "top" else "top",
        "target_region": bottom_region if seed_side == "top" else top_region,
    }
    return grid, meta


def _build_vertical_case() -> tuple[np.ndarray, dict[str, object]]:
    rect_w = random.randint(3, 5)
    rect_h1 = random.randint(3, 5)
    rect_h2 = random.randint(3, 5)
    gap_h = random.randint(1, 3)
    total_h = rect_h1 + gap_h + rect_h2
    min_h = max(13, total_h + 6)
    min_w = max(12, rect_w + 6)
    assert min_h <= 19 and min_w <= 18

    h = random.randint(min_h, 19)
    w = random.randint(min_w, 18)
    grid = np.full((h, w), BLACK, dtype=int)

    c0 = random.randint(2, w - rect_w - 3)
    r0 = random.randint(2, h - total_h - 3)
    r1 = r0 + rect_h1
    r2 = r1 + gap_h

    grid[r0:r1, c0 : c0 + rect_w] = GREEN
    grid[r2 : r2 + rect_h2, c0 : c0 + rect_w] = GREEN

    if r0 > 0:
        grid[:r0, c0 : c0 + rect_w] = RED
    if r2 + rect_h2 < h:
        grid[r2 + rect_h2 :, c0 : c0 + rect_w] = RED

    bridge = [(r, c) for r in range(r1, r2) for c in range(c0, c0 + rect_w)]
    seed_side = random.choice(["left", "right"])
    seed_r = random.randint(r1, r2 - 1)
    seed_c = c0 - 1 if seed_side == "left" else c0 + rect_w
    grid[seed_r, seed_c] = BLUE

    reserved = set(bridge)
    reserved.update((r, c) for r in range(r0, r1) for c in range(c0, c0 + rect_w))
    reserved.update((r, c) for r in range(r2, r2 + rect_h2) for c in range(c0, c0 + rect_w))
    reserved.add((seed_r, seed_c))
    reserved.update((seed_r, c) for c in range(w))

    left_region = (0, 0, h - 1, max(0, c0 - 2))
    right_region = (0, min(w - 1, c0 + rect_w + 1), h - 1, w - 1)
    _scatter_local_obstacles(grid, reserved, [left_region, right_region])

    meta = {
        "orientation": "vertical",
        "bridge": set(bridge),
        "target_side": "right" if seed_side == "left" else "left",
        "target_region": right_region if seed_side == "left" else left_region,
    }
    return grid, meta


def _target_cells(grid: np.ndarray, region: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = region
    cells = set()
    if r0 > r1 or c0 > c1:
        return cells
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r, c] == BLACK:
                cells.add((r, c))
    return cells


def _validate_instance(grid: np.ndarray, meta: dict[str, object]) -> bool:
    try:
        orientation, bridge, _box1, _box2 = _bridge_info(grid)
    except AssertionError:
        return False

    if orientation != meta["orientation"]:
        return False

    bridge_set = set(bridge)
    if bridge_set != meta["bridge"]:
        return False

    if len(np.argwhere(grid == BLUE)) != 1:
        return False

    target_cells = _target_cells(grid, meta["target_region"])
    if len(target_cells) < 6:
        return False

    no_bridge = _reachable_mask(grid, blocked=bridge_set)
    with_bridge = _reachable_mask(grid)
    if not any(with_bridge[r, c] for r, c in bridge_set):
        return False
    if any(no_bridge[r, c] for r, c in target_cells):
        return False
    if not any(with_bridge[r, c] for r, c in target_cells):
        return False

    out = main(grid)
    for r, c in bridge:
        if out[r, c] != TEAL:
            return False
    if np.count_nonzero(out == BLUE) <= np.count_nonzero(grid == BLUE):
        return False
    if not any(out[r, c] == BLUE for r, c in target_cells):
        return False
    if not np.array_equal(out[grid == GREEN], grid[grid == GREEN]):
        return False
    if not np.array_equal(out[grid == RED], grid[grid == RED]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        builder = _build_horizontal_case if random.random() < 0.5 else _build_vertical_case
        grid, meta = builder()
        if _validate_instance(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid bridge-and-flood instance")
