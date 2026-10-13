import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
LINE_COLORS = (1, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _exterior_black_mask(grid: np.ndarray) -> np.ndarray:
    """Mark black cells that are 4-connected to the grid border."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _region_metadata(grid: np.ndarray) -> list[dict]:
    outside = _exterior_black_mask(grid)
    enclosed_mask = (grid == BLACK) & ~outside
    regions = []
    for coords in _components(enclosed_mask):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width > height:
            kind = "wide"
        elif height > width:
            kind = "tall"
        else:
            kind = "square"
        regions.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
                "kind": kind,
                "area": int(coords.shape[0]),
            }
        )
    return regions


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    outside = _exterior_black_mask(g)
    out[outside] = GREEN

    for region in _region_metadata(g):
        r0, c0, _r1, _c1 = region["bbox"]
        height = region["height"]
        width = region["width"]
        if width == height:
            color_by_cell = lambda _r, _c: RED
        elif width > height:
            color_by_cell = lambda r, _c: RED if (r - r0) % 2 == 0 else YELLOW
        else:
            color_by_cell = lambda _r, c: RED if (c - c0) % 2 == 0 else YELLOW

        for r, c in region["coords"]:
            out[int(r), int(c)] = int(color_by_cell(int(r), int(c)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color the exterior green and fill each enclosed wall chamber with stripes chosen from its bbox ratio.

    Input structure:
    - The grid has a black background and one 4-connected non-black orthogonal wall object.
    - The wall acts as a barrier that leaves some black cells border-connected and traps others inside enclosed chambers.

    Rule:
    - Preserve every wall pixel.
    - Flood-fill all border-connected black cells to green.
    - Split the remaining black cells into 4-connected enclosed regions and compute each region's tight bounding box.
    - If a region's bbox is wider than tall, fill only that region with horizontal red/yellow stripes starting with red on the bbox's top row.
    - If a region's bbox is taller than wide, fill only that region with vertical red/yellow stripes starting with red on the bbox's left column.
    - If the bbox is square, fill the whole region solid red.

    Key insight:
    - The wall first determines topology, separating exterior background from enclosed chambers; only then can each chamber's own bbox geometry choose the stripe orientation and local phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    wall_colors = sorted(set(map(int, g.flatten())) - {BLACK})
    assert len(wall_colors) == 1
    assert len(_components(g == wall_colors[0])) == 1
    return _solve(g)


def _draw_rect_outline(grid: np.ndarray, top: int, left: int, bottom: int, right: int, color: int) -> None:
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _build_layout() -> np.ndarray:
    room_height = random.randint(4, 7)
    widths = {
        "tall": random.randint(2, room_height - 1),
        "square": room_height,
        "wide": random.randint(room_height + 1, room_height + 3),
    }
    order = random.sample(["tall", "square", "wide"], 3)
    room_widths = [widths[name] for name in order]

    outer_height = room_height + 2
    outer_width = sum(room_widths) + 4
    grid_height = outer_height + random.randint(2, min(10, 30 - outer_height))
    grid_width = outer_width + random.randint(2, min(10, 30 - outer_width))
    top = random.randint(1, grid_height - outer_height - 1)
    left = random.randint(1, grid_width - outer_width - 1)
    bottom = top + outer_height - 1
    right = left + outer_width - 1
    line_color = random.choice(LINE_COLORS)

    grid = np.zeros((grid_height, grid_width), dtype=int)
    _draw_rect_outline(grid, top, left, bottom, right, line_color)

    wall_cols = [left]
    cursor = left + 1
    for width in room_widths[:-1]:
        cursor += width
        wall_cols.append(cursor)
        cursor += 1
    wall_cols.append(right)

    for col in wall_cols[1:-1]:
        grid[top : bottom + 1, col] = line_color

    if room_height >= 5 and random.random() < 0.65:
        wide_idx = order.index("wide")
        divider_row = random.randint(top + 3, bottom - 3)
        grid[divider_row, wall_cols[wide_idx] : wall_cols[wide_idx + 1] + 1] = line_color

    if random.random() < 0.5:
        grid = grid.T.copy()
    if random.random() < 0.5:
        grid = np.flipud(grid).copy()
    if random.random() < 0.5:
        grid = np.fliplr(grid).copy()
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    colors = sorted(set(map(int, grid.flatten())) - {BLACK})
    if len(colors) != 1:
        return False
    if len(_components(grid == colors[0])) != 1:
        return False

    outside = _exterior_black_mask(grid)
    if not outside.any():
        return False
    regions = _region_metadata(grid)
    if len(regions) < 3:
        return False
    if min(region["area"] for region in regions) < 6:
        return False

    kinds = {region["kind"] for region in regions}
    if kinds != {"wide", "tall", "square"}:
        return False

    out = _solve(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[grid != BLACK], grid[grid != BLACK]):
        return False
    if not {RED, GREEN, YELLOW}.issubset(set(map(int, out.flatten()))):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_layout()
        if _valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
