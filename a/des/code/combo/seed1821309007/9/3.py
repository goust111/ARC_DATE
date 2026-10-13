import random
from collections import deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_RETRIES = 400


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors = set()

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rel = sorted((rr - r0, cc - c0) for rr, cc in coords)
            components.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "color": int(grid[coords[0]]),
                    "colors": colors,
                    "shape_key": (r1 - r0 + 1, c1 - c0 + 1, tuple(rel)),
                }
            )
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate colors among same-shape objects ordered by bounding-box centers."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = _find_components(g)

    out = g.copy()
    groups: dict[tuple, list[dict]] = {}
    for comp in components:
        assert len(comp["colors"]) == 1
        groups.setdefault(comp["shape_key"], []).append(comp)

    for group in groups.values():
        if len(group) <= 1:
            continue
        ordered = sorted(
            group,
            key=lambda comp: (
                comp["bbox"][0] + comp["bbox"][2],
                comp["bbox"][1] + comp["bbox"][3],
                comp["bbox"][0],
                comp["bbox"][1],
            ),
        )
        colors = [int(comp["color"]) for comp in ordered]
        for idx, comp in enumerate(ordered):
            new_color = colors[(idx + 1) % len(colors)]
            for r, c in comp["coords"]:
                out[r, c] = new_color

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    rel = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return (r1 - r0 + 1, c1 - c0 + 1, rel)


def _random_shape(rng: random.Random, existing: set[tuple]) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(1, 4)
        w = rng.randint(1, 4)
        if h * w < 2:
            continue
        area_target = rng.randint(2, min(h * w, 7))
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in NEIGHBORS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < area_target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in NEIGHBORS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        key = _normalize_shape(cells)
        if key in existing:
            continue
        hh, ww, rel = key
        if len(rel) < 2:
            continue
        mask = np.zeros((hh, ww), dtype=bool)
        for r, c in rel:
            mask[r, c] = True
        if hh * ww == len(rel):
            continue
        return mask
    raise RuntimeError("failed to sample unique shape")


def _sample_group_sizes(rng: random.Random) -> list[int]:
    include_singleton = rng.random() < 0.5
    rotating_groups = rng.randint(1, 3)
    sizes = [rng.randint(2, 4) for _ in range(rotating_groups)]
    if include_singleton:
        singleton_count = rng.randint(1, min(2, 4 - len(sizes)))
        sizes.extend([1] * singleton_count)
    while len(sizes) < 2:
        sizes.append(1)
    while sum(sizes) > 8:
        idx = max(range(len(sizes)), key=lambda i: sizes[i])
        if sizes[idx] > 2:
            sizes[idx] -= 1
        elif sizes[idx] == 1 and len([s for s in sizes if s == 1]) > 1:
            sizes.pop(idx)
        else:
            break
    rng.shuffle(sizes)
    return sizes


def _can_place(grid: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            rr, cc = top + r, left + c
            if grid[rr, cc] != BG:
                return False
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < gh and 0 <= nc < gw and grid[nr, nc] != BG:
                    return False
    return True


def _place_object(grid: np.ndarray, mask: np.ndarray, color: int, rng: random.Random) -> tuple[int, int] | None:
    gh, gw = grid.shape
    h, w = mask.shape
    anchors = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    rng.shuffle(anchors)
    for top, left in anchors:
        if not _can_place(grid, mask, top, left):
            continue
        for r, c in np.argwhere(mask):
            grid[top + int(r), left + int(c)] = color
        return (top, left)
    return None


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(12, 22)
    grid_w = rng.randint(12, 22)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    group_sizes = _sample_group_sizes(rng)
    total_objects = sum(group_sizes)
    if total_objects > len(PALETTE):
        return None

    existing_shapes: set[tuple] = set()
    group_masks: list[np.ndarray] = []
    for _ in group_sizes:
        group_masks.append(_random_shape(rng, existing_shapes))
        existing_shapes.add(_normalize_shape(set(map(tuple, np.argwhere(group_masks[-1])))))

    colors = list(PALETTE)
    rng.shuffle(colors)
    cursor = 0
    placements: list[dict] = []

    for mask, size in zip(group_masks, group_sizes):
        group_colors = colors[cursor : cursor + size]
        cursor += size
        rng.shuffle(group_colors)
        group_entries = []
        for color in group_colors:
            placed = _place_object(grid, mask, color, rng)
            if placed is None:
                return None
            group_entries.append({"mask": mask, "color": color, "top_left": placed})
        placements.append({"size": size, "entries": group_entries})

    if not any(group["size"] >= 2 for group in placements):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
