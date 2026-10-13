from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
RING_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
BRANCHES = ("zero", "partial", "erase")
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _outside_background(ring: np.ndarray) -> np.ndarray:
    rows, cols = ring.shape
    outside = np.zeros_like(ring, dtype=bool)
    queue = deque()

    for r in range(rows):
        for c in (0, cols - 1):
            if not ring[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if not ring[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < rows and 0 <= nc < cols and not ring[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return outside


def _peel_outer_layers(ring: np.ndarray, layers: int) -> np.ndarray:
    current = ring.copy()
    for _ in range(max(0, layers)):
        if not current.any():
            break
        outside = _outside_background(current)
        peel = np.zeros_like(current, dtype=bool)
        for r, c in np.argwhere(current):
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < current.shape[0] and 0 <= nc < current.shape[1] and outside[nr, nc]:
                    peel[rr, cc] = True
                    break
        current[peel] = False
    return current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each hollow ring by the number of red markers inside it.

    Input structure:
    - bg = black.
    - The grid contains several disjoint single-color closed rectangular rings.
    - Each ring encloses a black cavity that may contain 0 to 3 red single-pixel markers.
    - Different rings use different colors, and the red markers appear only inside ring cavities.

    Rule:
    - For each ring, count the red marker pixels inside its enclosed cavity.
    - Remove that many outer layers from the ring, where one layer means every ring pixel
      that is 4-neighbor-adjacent to the outside black background.
    - Repeat the peeling independently for each ring.
    - Remove all red markers; keep only the shrunken ring remnants on black.

    Key insight:
    - The red pixels are not output objects; they are counters that say how many times
      the surrounding ring must be peeled from the outside inward.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)
    ring_mask = (g != BLACK) & (g != RED)

    for comp in _components(ring_mask):
        color = int(g[comp[0][0], comp[0][1]])
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
        for r, c in comp:
            local[r - r0 + 1, c - c0 + 1] = True

        marker_count = int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] == RED))
        peeled = _peel_outer_layers(local, marker_count)
        for rr, cc in np.argwhere(peeled):
            out[r0 + int(rr) - 1, c0 + int(cc) - 1] = color

    return out


def _branch_sequence(n_rings: int) -> list[str]:
    branches = random.sample(BRANCHES, min(n_rings, len(BRANCHES)))
    while len(branches) < n_rings:
        branches.append(random.choice(BRANCHES))
    random.shuffle(branches)
    return branches


def _sample_thickness_and_markers(branch: str) -> tuple[int, int]:
    if branch == "zero":
        thickness = random.randint(1, 3)
        return thickness, 0
    if branch == "partial":
        thickness = random.randint(2, 3)
        return thickness, random.randint(1, thickness - 1)
    thickness = random.randint(1, 3)
    return thickness, random.randint(thickness, 3)


def _size_limit(n_rings: int) -> int:
    return {2: 10, 3: 8, 4: 6}[n_rings]


def _grid_min_size(n_rings: int) -> int:
    return {2: 16, 3: 20, 4: 24}[n_rings]


def _placement_candidates(occupied: np.ndarray, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    rows, cols = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, rows - outer_h):
        for c0 in range(1, cols - outer_w):
            if occupied[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1].any():
                continue
            candidates.append((r0, c0))
    return candidates


def _paint_ring(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    thickness: int,
    color: int,
) -> None:
    outer_h = inner_h + 2 * thickness
    outer_w = inner_w + 2 * thickness
    grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = color
    grid[r0 + thickness : r0 + thickness + inner_h, c0 + thickness : c0 + thickness + inner_w] = BLACK


def _marker_cells(spec: dict[str, int]) -> list[tuple[int, int]]:
    top = spec["r0"] + spec["thickness"]
    left = spec["c0"] + spec["thickness"]
    cells = [
        (r, c)
        for r in range(top, top + spec["inner_h"])
        for c in range(left, left + spec["inner_w"])
    ]
    return random.sample(cells, spec["markers"])


def _paint_shrunken_ring(grid: np.ndarray, spec: dict[str, int]) -> None:
    if spec["markers"] >= spec["thickness"]:
        return
    _paint_ring(
        grid,
        spec["r0"] + spec["markers"],
        spec["c0"] + spec["markers"],
        spec["inner_h"],
        spec["inner_w"],
        spec["thickness"] - spec["markers"],
        spec["color"],
    )


def _build_specs(n_rings: int) -> list[dict[str, int]]:
    branches = _branch_sequence(n_rings)
    colors = random.sample(RING_COLORS, n_rings)
    limit = _size_limit(n_rings)
    specs: list[dict[str, int]] = []
    for branch, color in zip(branches, colors):
        thickness, markers = _sample_thickness_and_markers(branch)
        inner_h = random.randint(2, limit)
        inner_w = random.randint(2, limit + 1)
        outer_h = inner_h + 2 * thickness
        outer_w = inner_w + 2 * thickness
        specs.append(
            {
                "branch": branch,
                "color": color,
                "thickness": thickness,
                "markers": markers,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    return specs


def _place_specs(rows: int, cols: int, specs: list[dict[str, int]]) -> bool:
    occupied = np.zeros((rows, cols), dtype=bool)
    for spec in specs:
        candidates = _placement_candidates(occupied, spec["outer_h"], spec["outer_w"])
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        spec["r0"] = r0
        spec["c0"] = c0
        occupied[r0 : r0 + spec["outer_h"], c0 : c0 + spec["outer_w"]] = True
    return True


def _render_pair(rows: int, cols: int, specs: list[dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), BLACK, dtype=int)
    out = np.full((rows, cols), BLACK, dtype=int)
    for spec in specs:
        _paint_ring(
            inp,
            spec["r0"],
            spec["c0"],
            spec["inner_h"],
            spec["inner_w"],
            spec["thickness"],
            spec["color"],
        )
        for r, c in _marker_cells(spec):
            inp[r, c] = RED
        _paint_shrunken_ring(out, spec)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_rings = random.randint(2, 4)
    rows = random.randint(_grid_min_size(n_rings), 30)
    cols = random.randint(_grid_min_size(n_rings), 30)
    specs = _build_specs(n_rings)

    if not _place_specs(rows, cols, specs):
        return None

    inp, expected = _render_pair(rows, cols, specs)
    if not np.count_nonzero(inp == RED):
        return None
    if not np.count_nonzero(expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
