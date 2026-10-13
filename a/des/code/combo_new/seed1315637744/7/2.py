import random
from collections import Counter

import numpy as np


ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = {"top", "bottom"}
VERTICAL_SIDES = {"left", "right"}
BG = 0
GEN_RETRIES = 400


def _majority_color(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    return int(Counter(flat).most_common(1)[0][0])


def _edge_colors(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    left_vals = g[1:-1, 0] if h > 2 else g[:, 0]
    right_vals = g[1:-1, -1] if h > 2 else g[:, -1]
    return {
        "top": _majority_color(g[0, :]),
        "bottom": _majority_color(g[-1, :]),
        "left": _majority_color(left_vals),
        "right": _majority_color(right_vals),
    }


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ORTHOGONALS:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _object_components(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    inner = np.zeros_like(g, dtype=bool)
    if g.shape[0] > 2 and g.shape[1] > 2:
        inner[1:-1, 1:-1] = g[1:-1, 1:-1] != BG
    return _connected_components(inner)


def _touching_side(coords: np.ndarray, shape: tuple[int, int]) -> str:
    h, w = shape
    touched: list[str] = []
    if bool(np.any(coords[:, 0] == 1)):
        touched.append("top")
    if bool(np.any(coords[:, 0] == h - 2)):
        touched.append("bottom")
    if bool(np.any(coords[:, 1] == 1)):
        touched.append("left")
    if bool(np.any(coords[:, 1] == w - 2)):
        touched.append("right")
    assert len(touched) == 1, f"object must touch exactly one border, got {touched}"
    return touched[0]


def _mirror_coords(coords: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    if side in HORIZONTAL_SIDES:
        rows = r0 + r1 - coords[:, 0]
        cols = coords[:, 1].copy()
    else:
        rows = coords[:, 0].copy()
        cols = c0 + c1 - coords[:, 1]
    return rows.astype(int), cols.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each interior object according to the border line it touches, then recolor it with that border color.

    Input structure:
    - The background is black.
    - The grid edges contain four different one-cell-thick colored border lines.
    - The interior contains several 4-connected monochrome objects, and each object is orthogonally adjacent to exactly one border line.

    Rule:
    - Identify which edge border line each interior object touches.
    - If an object touches the top or bottom border, reflect it across its own horizontal midpoint.
    - If an object touches the left or right border, reflect it across its own vertical midpoint.
    - Replace the original object with the reflected copy recolored to the touched border color, while keeping the border lines unchanged.

    Key insight: Border adjacency determines both the mirror axis and the output color, so the solver must first classify each object by the side it touches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    edge_colors = _edge_colors(g)
    assert len(set(edge_colors.values())) == 4

    out = g.copy()
    for coords in _object_components(g):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        side = _touching_side(coords, g.shape)
        out[coords[:, 0], coords[:, 1]] = BG
        new_rows, new_cols = _mirror_coords(coords, side)
        out[new_rows, new_cols] = int(edge_colors[side])
    return out


def _paint_borders(grid: np.ndarray, edge_colors: dict[str, int]) -> None:
    grid[0, :] = int(edge_colors["top"])
    grid[-1, :] = int(edge_colors["bottom"])
    grid[1:-1, 0] = int(edge_colors["left"])
    grid[1:-1, -1] = int(edge_colors["right"])


def _sample_sides() -> list[str]:
    count = random.randint(1, 4)
    if count == 1:
        return [random.choice(SIDES)]
    if random.random() < 0.75:
        horiz = random.choice(tuple(HORIZONTAL_SIDES))
        vert = random.choice(tuple(VERTICAL_SIDES))
        chosen = [horiz, vert]
        remaining = [side for side in SIDES if side not in chosen]
        chosen.extend(random.sample(remaining, count - 2))
        random.shuffle(chosen)
        return chosen
    return random.sample(SIDES, count)


def _neighbor_cells(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr, dc in ORTHOGONALS:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.append((nr, nc))
    return cells


def _cropped_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _sample_shape(height: int, width: int, side: str) -> np.ndarray | None:
    if height < 2 or width < 2:
        return None
    boundary: list[tuple[int, int]]
    if side == "top":
        boundary = [(0, c) for c in range(width)]
    elif side == "bottom":
        boundary = [(height - 1, c) for c in range(width)]
    elif side == "left":
        boundary = [(r, 0) for r in range(height)]
    else:
        boundary = [(r, width - 1) for r in range(height)]

    min_area = max(3, (height + width) // 2)
    max_area = max(min_area, height * width - 1)
    for _ in range(80):
        target_area = random.randint(min_area, max_area)
        mask = np.zeros((height, width), dtype=bool)
        seed_r, seed_c = random.choice(boundary)
        mask[seed_r, seed_c] = True
        frontier = set(_neighbor_cells(seed_r, seed_c, height, width))
        while int(mask.sum()) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            rr, cc = cell
            mask[rr, cc] = True
            for nr, nc in _neighbor_cells(rr, cc, height, width):
                if not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        if int(mask.sum()) != target_area:
            continue
        pts = np.argwhere(mask)
        if pts.size == 0:
            continue
        if int(pts[:, 0].max()) == int(pts[:, 0].min()) or int(pts[:, 1].max()) == int(pts[:, 1].min()):
            continue
        cropped = _cropped_mask(mask)
        if side in HORIZONTAL_SIDES and bool(np.array_equal(cropped, np.flipud(cropped))):
            continue
        if side in VERTICAL_SIDES and bool(np.array_equal(cropped, np.fliplr(cropped))):
            continue
        return mask
    return None


def _sample_box(side: str, h: int, w: int) -> tuple[int, int, int, int] | None:
    if side in HORIZONTAL_SIDES:
        box_h_max = min(6, h - 4)
        box_w_max = min(8, w - 4)
        if box_h_max < 2 or box_w_max < 3:
            return None
        box_h = random.randint(2, box_h_max)
        box_w = random.randint(3, box_w_max)
        c0 = random.randint(2, w - box_w - 2)
        r0 = 1 if side == "top" else h - 1 - box_h
        return r0, c0, box_h, box_w

    box_h_max = min(8, h - 4)
    box_w_max = min(6, w - 4)
    if box_h_max < 3 or box_w_max < 2:
        return None
    box_h = random.randint(3, box_h_max)
    box_w = random.randint(2, box_w_max)
    r0 = random.randint(2, h - box_h - 2)
    c0 = 1 if side == "left" else w - 1 - box_w
    return r0, c0, box_h, box_w


def _reserve_slice(start: int, length: int, limit: int) -> tuple[int, int]:
    return max(1, start - 1), min(limit - 1, start + length + 1)


def _place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    side: str,
    color: int,
) -> np.ndarray | None:
    h, w = grid.shape
    for _ in range(120):
        box = _sample_box(side, h, w)
        if box is None:
            return None
        r0, c0, box_h, box_w = box
        rr0, rr1 = _reserve_slice(r0, box_h, h)
        cc0, cc1 = _reserve_slice(c0, box_w, w)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            continue

        mask = _sample_shape(box_h, box_w, side)
        if mask is None:
            continue
        coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
        if bool(np.any(grid[coords[:, 0], coords[:, 1]] != BG)):
            continue

        grid[coords[:, 0], coords[:, 1]] = int(color)
        occupied[rr0:rr1, cc0:cc1] = True
        return coords.astype(int)
    return None


def _expected_output(grid: np.ndarray, placements: list[dict[str, object]], edge_colors: dict[str, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for placement in placements:
        coords = np.asarray(placement["coords"], dtype=int)
        side = str(placement["side"])
        out[coords[:, 0], coords[:, 1]] = BG
        new_rows, new_cols = _mirror_coords(coords, side)
        out[new_rows, new_cols] = int(edge_colors[side])
    return out


def _valid_scene(grid: np.ndarray, expected: np.ndarray, placements: list[dict[str, object]]) -> bool:
    comps = _object_components(grid)
    if len(comps) != len(placements):
        return False
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            return False
        try:
            _touching_side(coords, grid.shape)
        except AssertionError:
            return False
    return bool(np.array_equal(main(grid), expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 28)
    w = random.randint(12, 28)
    edge_values = random.sample(list(range(1, 10)), 4)
    edge_colors = dict(zip(SIDES, edge_values))
    if len(set(edge_colors.values())) != 4:
        return None

    sides = _sample_sides()
    free_colors = [color for color in range(1, 10) if color not in edge_colors.values()]
    object_colors = random.sample(free_colors, len(sides))

    grid = np.full((h, w), BG, dtype=int)
    _paint_borders(grid, edge_colors)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[dict[str, object]] = []

    for side, color in zip(sides, object_colors):
        coords = _place_object(grid, occupied, side, color)
        if coords is None:
            return None
        placements.append({"side": side, "coords": coords})

    expected = _expected_output(grid, placements, edge_colors)
    if np.array_equal(grid, expected):
        return None
    if not _valid_scene(grid, expected, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
