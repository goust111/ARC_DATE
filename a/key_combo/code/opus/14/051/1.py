from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MARKER_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _marker_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(
        (grid != BLACK) & (grid != YELLOW) & (grid != GREY)
    )
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    colors = [color for _, _, color in markers]
    assert len(markers) in {3, 4}, "expected 3 or 4 colored singleton markers"
    assert len(colors) == len(set(colors)), "marker colors must be distinct"
    for r, c, color in markers:
        same_color_neighbors = 0
        for rr, cc in _neighbors4(r, c):
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == color:
                same_color_neighbors += 1
        assert same_color_neighbors == 0, "markers must stay isolated singletons"
    return markers


def _build_ray_grid(grid: np.ndarray) -> np.ndarray:
    rays = np.zeros_like(grid, dtype=int)
    for r0, c0, color in _marker_cells(grid):
        for dr, dc in DIRS4:
            r, c = r0, c0
            while True:
                claimed = int(rays[r, c])
                assert claimed in (0, color), "conflicting ray colors are ambiguous"
                rays[r, c] = color

                nr, nc = r + dr, c + dc
                if not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == GREY:
                    break

                cell = int(grid[nr, nc])
                assert cell in (BLACK, color), "valid inputs must keep ray paths free of other objects"
                r, c = nr, nc
    return rays


def _flood_regions(grid: np.ndarray, rays: np.ndarray) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    regions: list[dict] = []
    for seed_r, seed_c in np.argwhere(grid == YELLOW):
        seed = (int(seed_r), int(seed_c))
        q: deque[tuple[int, int]] = deque()
        coords: list[tuple[int, int]] = []
        for rr, cc in _neighbors4(*seed):
            if not _in_bounds(grid.shape, rr, cc):
                continue
            if seen[rr, cc] or int(grid[rr, cc]) != BLACK or int(rays[rr, cc]) != 0:
                continue
            seen[rr, cc] = True
            q.append((rr, cc))

        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if not _in_bounds(grid.shape, rr, cc):
                    continue
                if seen[rr, cc] or int(grid[rr, cc]) != BLACK or int(rays[rr, cc]) != 0:
                    continue
                seen[rr, cc] = True
                q.append((rr, cc))

        contact_colors = set()
        for r, c in coords:
            for rr, cc in _neighbors4(r, c):
                if _in_bounds(grid.shape, rr, cc) and int(rays[rr, cc]) != 0:
                    contact_colors.add(int(rays[rr, cc]))

        regions.append({"seed": seed, "coords": coords, "contacts": contact_colors})
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """Extend colored rays from singleton markers, flood seeded black regions, and recolor each region from the ray colors it touches.

    Input structure:
    - The grid has black background, 3 or 4 distinct non-black non-grey singleton markers, several grey wall segments, and 2 or 3 yellow seed pixels.
    - Grey walls are blockers; the markers are the only colored ray sources; seeds sit inside closed black regions.

    Rule:
    - From each colored marker, extend same-colored rays up, down, left, and right until the next step would hit a grey wall or leave the grid.
    - From each yellow seed, flood-fill only black cells in 4-neighbor fashion; the fill cannot cross rays, grey walls, or the grid boundary.
    - For each flooded region, inspect the set of ray colors touching its boundary: one touching color recolors the whole region to that color, while multiple touching colors recolor it to grey.
    - Preserve the original colors of markers, rays, grey walls, and yellow seeds; leave black cells outside all seeded flood regions unchanged.

    Key insight:
    - The rays are both visible output lines and invisible flood barriers, so region color is determined only after the ray-made partitions exist and the touched ray colors are read back from the region boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rays = _build_ray_grid(g)
    out = g.copy()
    out[(g == BLACK) & (rays != 0)] = rays[(g == BLACK) & (rays != 0)]

    for region in _flood_regions(g, rays):
        if not region["coords"]:
            continue
        contacts = region["contacts"]
        assert contacts, "every seeded region must touch at least one ray"
        fill_color = next(iter(contacts)) if len(contacts) == 1 else GREY
        for r, c in region["coords"]:
            out[r, c] = fill_color

    out[g == YELLOW] = YELLOW
    out[g == GREY] = GREY
    return out


def _random_room_shape(rng: np.random.Generator) -> tuple[int, int]:
    shapes = ((1, 2), (2, 1), (2, 2))
    return shapes[int(rng.integers(len(shapes)))]


def _seed_one_cell(module: np.ndarray, inner_r0: int, inner_r1: int, inner_c0: int, inner_c1: int, rng: np.random.Generator) -> None:
    cells = [(r, c) for r in range(inner_r0, inner_r1) for c in range(inner_c0, inner_c1)]
    sr, sc = cells[int(rng.integers(len(cells)))]
    module[sr, sc] = YELLOW


def _make_mono_module(color: int, inner_h: int, inner_w: int, rng: np.random.Generator) -> np.ndarray:
    h = inner_h + 4
    w = inner_w + 4
    mod = np.zeros((h, w), dtype=int)

    mod[1, 1] = color
    mod[0, 1:3] = GREY
    mod[1:3, 0] = GREY
    mod[1 : inner_h + 3, inner_w + 2] = GREY
    mod[inner_h + 2, 1 : inner_w + 3] = GREY

    _seed_one_cell(mod, 2, inner_h + 2, 2, inner_w + 2, rng)
    return mod


def _make_duo_module(color_a: int, color_b: int, inner_h: int, inner_w: int, rng: np.random.Generator) -> np.ndarray:
    h = inner_h + 5
    w = inner_w + 5
    mod = np.zeros((h, w), dtype=int)

    br = inner_h + 2
    bc = inner_w + 2
    mod[1, 1] = color_a
    mod[br, bc] = color_b

    mod[0, 1:3] = GREY
    mod[1:3, 0] = GREY
    mod[1, bc : bc + 2] = GREY
    mod[br : br + 2, 1] = GREY
    mod[br - 1 : br + 1, bc + 1] = GREY
    mod[br + 1, bc - 1 : bc + 1] = GREY

    _seed_one_cell(mod, 2, inner_h + 2, 2, inner_w + 2, rng)
    return mod


def _rotated(module: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.rot90(module, k=int(rng.integers(4)))


def _can_stamp(grid: np.ndarray, blocked: np.ndarray, module: np.ndarray, r0: int, c0: int) -> bool:
    h, w = module.shape
    if r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    mask = module != BLACK
    return not blocked[r0 : r0 + h, c0 : c0 + w][mask].any()


def _stamp(grid: np.ndarray, blocked: np.ndarray, module: np.ndarray, r0: int, c0: int) -> None:
    h, w = module.shape
    mask = module != BLACK
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = module[mask]
    pad_r0 = max(0, r0 - 1)
    pad_c0 = max(0, c0 - 1)
    pad_r1 = min(grid.shape[0], r0 + h + 1)
    pad_c1 = min(grid.shape[1], c0 + w + 1)
    blocked[pad_r0:pad_r1, pad_c0:pad_c1] = True


def _add_extra_walls(grid: np.ndarray, blocked: np.ndarray, rng: np.random.Generator) -> None:
    for _ in range(int(rng.integers(0, 3))):
        horizontal = bool(rng.integers(2))
        length = int(rng.integers(2, 5))
        placed = False
        for _ in range(40):
            if horizontal:
                r = int(rng.integers(0, grid.shape[0]))
                c = int(rng.integers(0, grid.shape[1] - length + 1))
                if blocked[r, c : c + length].any():
                    continue
                grid[r, c : c + length] = GREY
                blocked[max(0, r - 1) : min(grid.shape[0], r + 2), max(0, c - 1) : min(grid.shape[1], c + length + 1)] = True
            else:
                r = int(rng.integers(0, grid.shape[0] - length + 1))
                c = int(rng.integers(0, grid.shape[1]))
                if blocked[r : r + length, c].any():
                    continue
                grid[r : r + length, c] = GREY
                blocked[max(0, r - 1) : min(grid.shape[0], r + length + 1), max(0, c - 1) : min(grid.shape[1], c + 2)] = True
            placed = True
            break
        if not placed:
            continue


def _analyze_generated(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    rays = _build_ray_grid(grid)
    regions = _flood_regions(grid, rays)
    assert 2 <= len(regions) <= 3, "expected 2 or 3 seeds"
    contact_sizes = [len(region["contacts"]) for region in regions if region["coords"]]
    assert contact_sizes, "seeded regions must contain fillable black cells"
    assert any(size == 1 for size in contact_sizes), "need a single-color region"
    assert any(size > 1 for size in contact_sizes), "need a multi-color-to-grey region"

    colored_regions = []
    for region in regions:
        assert region["coords"], "every seed must have at least one black cell to flood"
        assert region["contacts"], "every flooded region must touch a ray"
        fill_color = next(iter(region["contacts"])) if len(region["contacts"]) == 1 else GREY
        colored_regions.append(fill_color)
    assert len(set(colored_regions)) >= 2, "seeded regions should not all recolor the same way"
    return rays, regions


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    module_count = 2 if rng.random() < 0.45 else 3
    colors = list(rng.choice(np.array(MARKER_COLORS, dtype=int), size=module_count + 1, replace=False))

    modules = []
    duo_h, duo_w = _random_room_shape(rng)
    duo = _make_duo_module(colors[0], colors[1], duo_h, duo_w, rng)
    modules.append(_rotated(duo, rng))

    mono1_h, mono1_w = _random_room_shape(rng)
    mono1 = _make_mono_module(colors[2], mono1_h, mono1_w, rng)
    modules.append(_rotated(mono1, rng))

    if module_count == 3:
        mono2_h, mono2_w = _random_room_shape(rng)
        mono2 = _make_mono_module(colors[3], mono2_h, mono2_w, rng)
        modules.append(_rotated(mono2, rng))

    max_h = max(module.shape[0] for module in modules)
    max_w = max(module.shape[1] for module in modules)
    h = int(rng.integers(max(15, max_h + 4), 21))
    w = int(rng.integers(max(15, max_w + 4), 21))
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = list(range(len(modules)))
    rng.shuffle(order)
    for idx in order:
        module = modules[idx]
        placed = False
        mh, mw = module.shape
        for _ in range(120):
            r0 = int(rng.integers(0, h - mh + 1))
            c0 = int(rng.integers(0, w - mw + 1))
            if not _can_stamp(grid, blocked, module, r0, c0):
                continue
            _stamp(grid, blocked, module, r0, c0)
            placed = True
            break
        if not placed:
            return None

    _add_extra_walls(grid, blocked, rng)

    try:
        _, regions = _analyze_generated(grid)
        out = main(grid)
    except Exception:
        return None

    if not (out == BLACK).any():
        return None
    if len(_marker_cells(grid)) not in {3, 4}:
        return None
    if not (2 <= len(regions) <= 3):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition flood-fill input")
