import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
PHASE_TO_COLOR = {0: 2, 1: 3, 2: 1}

GEN_ATTEMPTS = 800
LAYOUTS = ((2, 2), (2, 3), (3, 2))
ROW_CHOICES = (2, 3, 4, 5, 6)
COL_CHOICES = (2, 3, 4, 5, 6)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _erode8(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.copy()
    padded = np.pad(mask, 1, constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out &= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return out


def _inner_border(mask: np.ndarray, width: int) -> np.ndarray:
    interior = mask.copy()
    for _ in range(width):
        interior = _erode8(interior)
        if not interior.any():
            break
    return mask & ~interior


def _extract_regions(grid: np.ndarray) -> list[dict]:
    regions: list[dict] = []
    for coords in _component_coords(grid == BG):
        if _touches_border(coords, grid.shape):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        r0, c0, r1, c1 = _bbox(coords)
        area = int(coords.shape[0])
        width = 1 + area // 12
        border = _inner_border(mask, width)
        regions.append(
            {
                "coords": coords,
                "mask": mask,
                "bbox": (r0, c0, r1, c1),
                "area": area,
                "width": width,
                "phase": (r0 + c0) % 3,
                "border": border,
            }
        )
    regions.sort(key=lambda item: item["bbox"])
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each enclosed room's inner border using area-dependent thickness and diagonal-phase color.

    Input structure:
    - The grid uses black background cells and grey wall cells.
    - Grey walls form one 4-connected border-touching partition that encloses several black 4-connected rooms.
    - Each room is fully surrounded by grey walls and is identified by its own area and bounding box.

    Rule:
    - Find every black 4-connected component that does not touch the grid boundary.
    - For each room, compute border width = 1 + floor(area / 12).
    - Recolor the room cells whose Chebyshev distance to the surrounding grey wall is at most that width.
    - Choose the border color from the room bbox top-left coordinate: (row + col) mod 3 gives red(2), green(3), blue(1) for residues 0, 1, 2 respectively.

    Key insight:
    - Thickness depends only on room area, while color depends only on the room's global diagonal phase; both cues must be applied independently to the same enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for region in _extract_regions(g):
        out[region["border"]] = PHASE_TO_COLOR[region["phase"]]
    return out


def _cuts(lengths: tuple[int, ...]) -> list[int]:
    cuts = [0]
    pos = 0
    for length in lengths:
        pos += length + 1
        cuts.append(pos)
    return cuts


def _build_grid(row_heights: tuple[int, ...], col_widths: tuple[int, ...]) -> np.ndarray:
    row_cuts = _cuts(row_heights)
    col_cuts = _cuts(col_widths)
    h = row_cuts[-1] + 1
    w = col_cuts[-1] + 1
    grid = np.full((h, w), BG, dtype=int)
    grid[row_cuts, :] = WALL
    grid[:, col_cuts] = WALL
    return grid


def _room_metadata(row_heights: tuple[int, ...], col_widths: tuple[int, ...]) -> list[dict]:
    row_cuts = _cuts(row_heights)
    col_cuts = _cuts(col_widths)
    rooms = []
    for i, rh in enumerate(row_heights):
        for j, cw in enumerate(col_widths):
            r0 = row_cuts[i] + 1
            c0 = col_cuts[j] + 1
            rooms.append(
                {
                    "bbox": (r0, c0, r0 + rh - 1, c0 + cw - 1),
                    "area": rh * cw,
                    "width": 1 + (rh * cw) // 12,
                    "phase": (r0 + c0) % 3,
                    "h": rh,
                    "w": cw,
                }
            )
    return rooms


def _has_independent_examples(rooms: list[dict]) -> bool:
    same_phase_diff_width = False
    same_width_diff_phase = False
    for i, a in enumerate(rooms):
        for b in rooms[i + 1 :]:
            if a["phase"] == b["phase"] and a["width"] != b["width"]:
                same_phase_diff_width = True
            if a["width"] == b["width"] and a["phase"] != b["phase"]:
                same_width_diff_phase = True
    return same_phase_diff_width and same_width_diff_phase


def _valid_layout(row_heights: tuple[int, ...], col_widths: tuple[int, ...]) -> bool:
    rooms = _room_metadata(row_heights, col_widths)
    areas = [room["area"] for room in rooms]
    if any(area < 6 or area > 30 for area in areas):
        return False
    if len(set(areas)) != len(areas):
        return False

    widths = {room["width"] for room in rooms}
    if widths != {1, 2, 3}:
        return False

    phases = {room["phase"] for room in rooms}
    if phases != {0, 1, 2}:
        return False

    if not any(6 <= area <= 11 for area in areas):
        return False
    if not any(13 <= area <= 20 for area in areas):
        return False
    if not any(24 <= area <= 30 for area in areas):
        return False

    if not _has_independent_examples(rooms):
        return False

    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    if max(h, w) > 30:
        return False
    if min(h, w) < 8:
        return False
    if h > 20 or w > 20:
        return False

    grid = _build_grid(row_heights, col_widths)
    regions = _extract_regions(grid)
    if len(regions) != len(rooms):
        return False
    fully_colored = sum(int(region["border"].sum()) == region["area"] for region in regions)
    partially_colored = sum(int(region["border"].sum()) < region["area"] for region in regions)
    if fully_colored < 1 or partially_colored < 1:
        return False

    return True


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(np.unique(grid)).issubset({BG, WALL})

    wall_comps = _component_coords(grid == WALL)
    assert len(wall_comps) == 1
    assert _touches_border(wall_comps[0], grid.shape)

    black_border_comps = [comp for comp in _component_coords(grid == BG) if _touches_border(comp, grid.shape)]
    assert not black_border_comps

    regions = _extract_regions(grid)
    assert 4 <= len(regions) <= 6

    areas = [region["area"] for region in regions]
    widths = [region["width"] for region in regions]
    phases = [region["phase"] for region in regions]
    assert len(set(areas)) == len(areas)
    assert all(6 <= area <= 30 for area in areas)
    assert set(widths) == {1, 2, 3}
    assert set(phases) == {0, 1, 2}
    assert any(6 <= area <= 11 for area in areas)
    assert any(13 <= area <= 20 for area in areas)
    assert any(24 <= area <= 30 for area in areas)
    assert _has_independent_examples(regions)

    fully_colored = 0
    partially_colored = 0
    for region in regions:
        border_area = int(region["border"].sum())
        if border_area == region["area"]:
            fully_colored += 1
        elif border_area < region["area"]:
            partially_colored += 1
    assert fully_colored >= 1
    assert partially_colored >= 1

    out = main(grid)
    assert out.shape == grid.shape
    assert np.all(out[grid == WALL] == WALL)
    assert {1, 2, 3}.issubset(set(np.unique(out)))

    for region in regions:
        color = PHASE_TO_COLOR[region["phase"]]
        assert np.all(out[region["border"]] == color)
        assert np.all(out[region["mask"] & ~region["border"]] == BG)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows, n_cols = rng.choice(LAYOUTS)
    for _ in range(120):
        row_heights = tuple(rng.choice(ROW_CHOICES) for _ in range(n_rows))
        col_widths = tuple(rng.choice(COL_CHOICES) for _ in range(n_cols))
        if not _valid_layout(row_heights, col_widths):
            continue
        grid = _build_grid(row_heights, col_widths)
        _validate_grid(grid)
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
