from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) != bg:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            colors = np.unique(grid[arr[:, 0], arr[:, 1]])
            assert colors.size == 1, "each foreground component must be monochromatic"
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            components.append(
                {
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "color": int(colors[0]),
                    "anchor": (r0, c0),
                }
            )

    assert components, "expected at least one foreground component"
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-swap component colors by area rank."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    ranked = sorted(components, key=lambda comp: (int(comp["area"]), tuple(comp["anchor"])))

    out = g.copy()
    for idx, comp in enumerate(ranked):
        new_color = int(ranked[-1 - idx]["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = new_color
    return out


# --------------- generator ---------------

PALETTE = tuple(range(10))
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 24
AREA_MIN = 2
AREA_MAX = 12
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 80


def _sample_component_count() -> int:
    if random.random() < 0.5:
        return random.choice((3, 5))
    return random.choice((4, 6))


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    coords[:, 0] -= r0
    coords[:, 1] -= c0
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    out = np.zeros((h, w), dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                frontier.add((r + dr, c + dc))
        frontier.difference_update(cells)
        cells.add(random.choice(tuple(frontier)))

    coords = np.asarray(list(cells), dtype=int)
    coords[:, 0] -= int(coords[:, 0].min())
    coords[:, 1] -= int(coords[:, 1].min())
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return _normalize_mask(mask)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        blocked[r, c] = True
        for dr, dc in DIRS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_objects(
    h: int,
    w: int,
    objects: list[dict[str, object]],
    bg: int,
) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = sorted(
        range(len(objects)),
        key=lambda idx: (
            int(objects[idx]["area"]),
            int(np.prod(np.asarray(objects[idx]["mask"]).shape)),
        ),
        reverse=True,
    )

    for idx in order:
        mask = np.asarray(objects[idx]["mask"], dtype=bool)
        color = int(objects[idx]["color"])
        mh, mw = mask.shape
        if mh > h or mw > w:
            return None

        candidates = [(r0, c0) for r0 in range(h - mh + 1) for c0 in range(w - mw + 1)]
        random.shuffle(candidates)
        placed = False

        for r0, c0 in candidates[:PLACEMENT_ATTEMPTS]:
            patch = blocked[r0 : r0 + mh, c0 : c0 + mw]
            if bool((patch & mask).any()):
                continue

            grid_slice = grid[r0 : r0 + mh, c0 : c0 + mw]
            grid_slice[mask] = color

            coords = np.argwhere(mask).astype(int)
            coords[:, 0] += int(r0)
            coords[:, 1] += int(c0)
            _mark_blocked(blocked, coords)
            placed = True
            break

        if not placed:
            return None

    return grid


def _valid_input(grid: np.ndarray, expected_count: int) -> bool:
    bg = _get_bg(grid)
    components = _find_components(grid, bg)
    if len(components) != expected_count:
        return False

    areas = [int(comp["area"]) for comp in components]
    colors = [int(comp["color"]) for comp in components]
    if len(set(areas)) != expected_count:
        return False
    if len(set(colors)) != expected_count:
        return False

    bg_count = int(np.count_nonzero(grid == bg))
    return bg_count > max(areas)


def _gen_attempt() -> np.ndarray | None:
    count = _sample_component_count()
    bg = random.choice(PALETTE)
    fg_colors = random.sample([color for color in PALETTE if color != bg], count)
    areas = random.sample(range(AREA_MIN, AREA_MAX + 1), count)

    objects = []
    for area, color in zip(areas, fg_colors):
        objects.append({"area": int(area), "color": int(color), "mask": _random_polyomino(int(area))})

    max_h = max(int(np.asarray(obj["mask"]).shape[0]) for obj in objects)
    max_w = max(int(np.asarray(obj["mask"]).shape[1]) for obj in objects)
    total_area = sum(int(obj["area"]) for obj in objects)

    min_h = max(GRID_SIDE_MIN, max_h + 2, int(np.ceil(np.sqrt(total_area * 3.0))))
    min_w = max(GRID_SIDE_MIN, max_w + 2, int(np.ceil(np.sqrt(total_area * 3.0))))
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    h = random.randint(min_h, GRID_SIDE_MAX)
    w = random.randint(min_w, GRID_SIDE_MAX)
    grid = _place_objects(h, w, objects, bg)
    if grid is None or not _valid_input(grid, count):
        return None

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
