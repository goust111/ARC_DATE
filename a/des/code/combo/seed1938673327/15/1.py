from __future__ import annotations

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _mode(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _nonbg_components(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            area = 0

            while stack:
                rr, cc = stack.pop()
                area += 1
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append((r0, c0, r1, c1, area))

    return comps


def _frame_patch(height: int, width: int, color: int, bg: int, thickness: int) -> np.ndarray:
    patch = np.full((height, width), color, dtype=int)
    if thickness <= 0:
        return patch
    if height > 2 * thickness and width > 2 * thickness:
        patch[thickness : height - thickness, thickness : width - thickness] = bg
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid rectangle plus its internal marker pixels with a same-color frame.

    The frame thickness equals the number of non-rectangle marker pixels inside that
    rectangle. When the requested thickness would leave no interior, the rectangle stays
    solid.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for r0, c0, r1, c1, area in _nonbg_components(g, bg):
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if area != height * width:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        rect_color = _mode(sub[border])
        thickness = int(np.count_nonzero(sub != rect_color))
        out[r0 : r1 + 1, c0 : c1 + 1] = _frame_patch(height, width, rect_color, bg, thickness)

    return out


# --------------- generator ---------------

BG_COLOR = 0
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 4)
HOLLOW_SIZE_RANGE = (3, 8)
SOLID_SIZE_RANGE = (4, 8)
PLACEMENT_MARGIN = 1
PLACEMENT_PADDING = 1
MAX_PLACEMENT_TRIES = 80
MAX_GENERATION_ATTEMPTS = 400


def _randint(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _sample_geometry(rng: np.random.Generator, solid_branch: bool) -> tuple[int, int, int]:
    for _ in range(200):
        bounds = SOLID_SIZE_RANGE if solid_branch else HOLLOW_SIZE_RANGE
        height = _randint(rng, bounds)
        width = _randint(rng, bounds)
        interior_area = (height - 2) * (width - 2)
        if interior_area <= 0:
            continue

        short_side = min(height, width)
        if solid_branch:
            min_k = (short_side + 1) // 2
            if interior_area < min_k:
                continue
            max_k = min(interior_area, min_k + 4)
            thickness = int(rng.integers(min_k, max_k + 1))
            return height, width, thickness

        max_k = min((height - 1) // 2, (width - 1) // 2, interior_area)
        if max_k < 1:
            continue
        thickness = int(rng.integers(1, max_k + 1))
        return height, width, thickness

    raise RuntimeError("failed to sample rectangle geometry")


def _sample_markers(
    rng: np.random.Generator, height: int, width: int, count: int
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    picks = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in np.atleast_1d(picks)]


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    rr1 = min(h, r0 + height + PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    cc1 = min(w, c0 + width + PLACEMENT_PADDING)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_specs(
    rng: np.random.Generator, height: int, width: int, specs: list[dict[str, object]]
) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["h"] * specs[idx]["w"], reverse=True)

    for idx in order:
        obj_h = int(specs[idx]["h"])
        obj_w = int(specs[idx]["w"])
        if obj_h + 2 * PLACEMENT_MARGIN > height or obj_w + 2 * PLACEMENT_MARGIN > width:
            return False

        placed = False
        for _ in range(MAX_PLACEMENT_TRIES):
            r0 = int(rng.integers(PLACEMENT_MARGIN, height - obj_h - PLACEMENT_MARGIN + 1))
            c0 = int(rng.integers(PLACEMENT_MARGIN, width - obj_w - PLACEMENT_MARGIN + 1))
            if not _can_place(occupied, r0, c0, obj_h, obj_w):
                continue
            occupied[r0 : r0 + obj_h, c0 : c0 + obj_w] = True
            specs[idx]["r0"] = r0
            specs[idx]["c0"] = c0
            placed = True
            break

        if not placed:
            return False

    return True


def _build_grids(
    shape: tuple[int, int], marker_color: int, specs: list[dict[str, object]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG_COLOR, dtype=int)
    out = np.full(shape, BG_COLOR, dtype=int)

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        height = int(spec["h"])
        width = int(spec["w"])
        rect_color = int(spec["color"])
        thickness = int(spec["k"])
        markers = spec["markers"]

        inp[r0 : r0 + height, c0 : c0 + width] = rect_color
        out[r0 : r0 + height, c0 : c0 + width] = _frame_patch(
            height, width, rect_color, BG_COLOR, thickness
        )

        for mr, mc in markers:
            inp[r0 + int(mr), c0 + int(mc)] = marker_color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_objects = _randint(rng, OBJECT_COUNT_RANGE)

    branch_flags = rng.integers(0, 2, size=n_objects)
    if np.all(branch_flags == branch_flags[0]):
        branch_flags[int(rng.integers(0, n_objects))] = 1 - int(branch_flags[0])

    colors = rng.choice(np.arange(1, 10), size=n_objects + 1, replace=False)
    marker_color = int(colors[0])
    rect_colors = [int(color) for color in colors[1:]]

    specs: list[dict[str, object]] = []
    total_area = 0
    max_h = 0
    max_w = 0
    for solid_branch, rect_color in zip(branch_flags, rect_colors):
        obj_h, obj_w, thickness = _sample_geometry(rng, bool(solid_branch))
        markers = _sample_markers(rng, obj_h, obj_w, thickness)
        specs.append(
            {
                "h": obj_h,
                "w": obj_w,
                "k": thickness,
                "color": rect_color,
                "markers": markers,
            }
        )
        total_area += obj_h * obj_w
        max_h = max(max_h, obj_h)
        max_w = max(max_w, obj_w)

    grid_h = _randint(rng, (max(18, max_h + 2), GRID_HEIGHT_RANGE[1]))
    grid_w = _randint(rng, (max(18, max_w + 2), GRID_WIDTH_RANGE[1]))
    if total_area > int(0.45 * grid_h * grid_w):
        return None

    if not _place_specs(rng, grid_h, grid_w, specs):
        return None

    inp, expected = _build_grids((grid_h, grid_w), marker_color, specs)
    if np.count_nonzero(inp == BG_COLOR) <= np.count_nonzero(inp != BG_COLOR):
        return None
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
