from __future__ import annotations

import numpy as np


BG = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

# Generator configuration.
GRID_HEIGHT_RANGE = (12, 22)
GRID_WIDTH_RANGE = (18, 30)
PAIR_COUNT_OPTIONS = (1, 2, 3, 4)
UNPAIRED_COUNT_OPTIONS = (0, 1, 2)
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
SHAPE_AREA_RANGE = (3, 6)
SHAPE_BOX_HEIGHT_MAX = 4
SHAPE_BOX_WIDTH_MAX = 4
PAIR_GAP_RANGE = (2, 7)
HALO = 1
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 160
GENERATION_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _collect_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                }
            )
    return components


def _normalized_shape(component: dict[str, object]) -> tuple[int, int, frozenset[tuple[int, int]]]:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = component["bbox"]
    rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return int(r1 - r0 + 1), int(c1 - c0 + 1), rel


def _mirror_shape(shape: frozenset[tuple[int, int]], width: int) -> frozenset[tuple[int, int]]:
    return frozenset((r, width - 1 - c) for r, c in shape)


def _pair_record(left: dict[str, object], right: dict[str, object]) -> dict[str, object] | None:
    if int(left["color"]) != int(right["color"]):
        return None

    lr0, lc0, lr1, lc1 = left["bbox"]
    rr0, rc0, rr1, rc1 = right["bbox"]

    if lc1 >= rc0:
        return None
    if (lr0, lr1) != (rr0, rr1):
        return None

    left_h, left_w, left_shape = _normalized_shape(left)
    right_h, right_w, right_shape = _normalized_shape(right)
    if left_h != right_h or left_w != right_w:
        return None
    if _mirror_shape(left_shape, left_w) != right_shape:
        return None

    shift = int(lc1 + 1 - rc0)
    dest = np.asarray(right["coords"], dtype=int).copy()
    dest[:, 1] += shift

    left_cells = {tuple(x) for x in np.asarray(left["coords"], dtype=int).tolist()}
    for cell in map(tuple, dest.tolist()):
        if cell in left_cells:
            return None

    return {
        "color": int(left["color"]),
        "left": left,
        "right": right,
        "dest": dest,
    }


def _find_transformable_pairs(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    components = _collect_components(grid, bg=bg)
    by_color: dict[int, list[dict[str, object]]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    pairs: list[dict[str, object]] = []
    for comps in by_color.values():
        if len(comps) != 2:
            continue
        ordered = sorted(comps, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
        pair = _pair_record(ordered[0], ordered[1])
        if pair is None:
            continue

        right_cells = {tuple(x) for x in np.asarray(pair["right"]["coords"], dtype=int).tolist()}
        clear = True
        for r, c in map(tuple, np.asarray(pair["dest"], dtype=int).tolist()):
            if int(grid[r, c]) != bg and (r, c) not in right_cells:
                clear = False
                break
        if clear:
            pairs.append(pair)

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Merge each same-color horizontal mirror pair into one adjacent symmetric pattern."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for pair in _find_transformable_pairs(g, bg=BG):
        right = np.asarray(pair["right"]["coords"], dtype=int)
        for r, c in right:
            out[r, c] = BG
        for r, c in np.asarray(pair["dest"], dtype=int):
            out[r, c] = int(pair["color"])

    return out


def _tight_shape(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    rel = sorted((r - r0, c - c0) for r, c in coords)
    return np.array(rel, dtype=int)


def _random_connected_shape(require_non_symmetric: bool) -> np.ndarray:
    area_lo, area_hi = SHAPE_AREA_RANGE
    for _ in range(SHAPE_RETRIES):
        area = int(RNG.integers(area_lo, area_hi + 1))
        box_h = int(RNG.integers(1, min(SHAPE_BOX_HEIGHT_MAX, area) + 1))
        box_w = int(RNG.integers(2, min(SHAPE_BOX_WIDTH_MAX, area) + 1))
        if box_h * box_w < area:
            continue

        start = (int(RNG.integers(0, box_h)), int(RNG.integers(0, box_w)))
        coords = {start}
        while len(coords) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in coords:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(frontier[int(RNG.integers(0, len(frontier)))])
        if len(coords) != area:
            continue

        shape = _tight_shape(coords)
        h = int(shape[:, 0].max()) + 1
        w = int(shape[:, 1].max()) + 1
        if w < 2:
            continue

        shape_set = frozenset(map(tuple, shape.tolist()))
        if require_non_symmetric and shape_set == _mirror_shape(shape_set, w):
            continue
        return shape

    raise RuntimeError("failed to sample a valid shape")


def _shape_dims(shape: np.ndarray) -> tuple[int, int]:
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _mirror_relative_coords(shape: np.ndarray) -> np.ndarray:
    h, w = _shape_dims(shape)
    mirrored = np.array([(int(r), w - 1 - int(c)) for r, c in shape], dtype=int)
    rows = mirrored[:, 0]
    cols = mirrored[:, 1]
    mirrored[:, 0] -= int(rows.min())
    mirrored[:, 1] -= int(cols.min())
    assert _shape_dims(mirrored) == (h, w)
    return mirrored


def _can_reserve(blocked: np.ndarray, bbox: tuple[int, int, int, int], halo: int = HALO) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        return False
    rr0 = max(0, r0 - halo)
    cc0 = max(0, c0 - halo)
    rr1 = min(h - 1, r1 + halo)
    cc1 = min(w - 1, c1 + halo)
    return not np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _reserve(blocked: np.ndarray, bbox: tuple[int, int, int, int], halo: int = HALO) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - halo)
    cc0 = max(0, c0 - halo)
    rr1 = min(h - 1, r1 + halo)
    cc1 = min(w - 1, c1 + halo)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in shape:
        grid[top + int(r), left + int(c)] = int(color)


def _fit_choices(count_options: tuple[int, ...], limit: int) -> list[int]:
    return [x for x in count_options if x <= limit]


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    w = int(RNG.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))

    max_pairs = max(1, min(max(PAIR_COUNT_OPTIONS), h // 4))
    pair_choices = _fit_choices(PAIR_COUNT_OPTIONS, max_pairs)
    n_pairs = int(RNG.choice(pair_choices))

    max_unpaired = min(max(UNPAIRED_COUNT_OPTIONS), max(0, (h - 2 * n_pairs) // 5))
    unpaired_choices = _fit_choices(UNPAIRED_COUNT_OPTIONS, max_unpaired)
    n_unpaired = int(RNG.choice(unpaired_choices if unpaired_choices else [0]))

    total_colors = n_pairs + n_unpaired
    if total_colors > len(COLOR_POOL):
        return None

    colors = RNG.choice(COLOR_POOL, size=total_colors, replace=False).tolist()
    pair_colors = [int(x) for x in colors[:n_pairs]]
    single_colors = [int(x) for x in colors[n_pairs:]]

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for color in pair_colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            shape = _random_connected_shape(require_non_symmetric=True)
            shape_r = _mirror_relative_coords(shape)
            shape_h, shape_w = _shape_dims(shape)
            min_gap, max_gap = PAIR_GAP_RANGE
            gap_hi = min(max_gap, w - 2 * shape_w)
            if gap_hi < min_gap:
                continue
            gap = int(RNG.integers(min_gap, gap_hi + 1))
            span_w = 2 * shape_w + gap
            top = int(RNG.integers(0, h - shape_h + 1))
            left = int(RNG.integers(0, w - span_w + 1))
            bbox = (top, left, top + shape_h - 1, left + span_w - 1)
            if not _can_reserve(blocked, bbox):
                continue

            _paint_shape(grid, shape, top, left, color)
            _paint_shape(grid, shape_r, top, left + shape_w + gap, color)
            _reserve(blocked, bbox)
            placed = True
            break
        if not placed:
            return None

    for color in single_colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            shape = _random_connected_shape(require_non_symmetric=False)
            shape_h, shape_w = _shape_dims(shape)
            top = int(RNG.integers(0, h - shape_h + 1))
            left = int(RNG.integers(0, w - shape_w + 1))
            bbox = (top, left, top + shape_h - 1, left + shape_w - 1)
            if not _can_reserve(blocked, bbox):
                continue

            _paint_shape(grid, shape, top, left, color)
            _reserve(blocked, bbox)
            placed = True
            break
        if not placed:
            return None

    pairs = _find_transformable_pairs(grid, bg=BG)
    if len(pairs) != n_pairs:
        return None
    if {int(pair["color"]) for pair in pairs} != set(pair_colors):
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
