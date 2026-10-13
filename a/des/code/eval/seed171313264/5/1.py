from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ORIENTATIONS = ("TL", "TR", "BL", "BR")
GEN_ATTEMPTS = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (24, 34)
REGION_COUNT_RANGE = (1, 4)
LEGEND_WIDTH = 6
LEGEND_ENTRY_HEIGHT = 3
LEGEND_GAP_CHOICES = (1, 2)
PLAY_GAP = 2
PLACEMENT_ATTEMPTS = 200

LEGEND_MARKERS = {
    "TL": np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool),
    "TR": np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool),
    "BL": np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    "BR": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=bool),
}

TEMPLATE_LIBRARY = [
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mode(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_connected(points: list[tuple[int, int]]) -> bool:
    if not points:
        return False
    point_set = set(points)
    seen = {points[0]}
    queue = deque([points[0]])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in point_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(point_set)


def _anchor_key(points: list[tuple[int, int]]) -> str | None:
    if not 3 <= len(points) <= 4:
        return None
    r0, c0, r1, c1 = _bbox(points)
    norm = [(r - r0, c - c0) for r, c in points]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    corners = {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }
    matches: list[str] = []
    for key, (ar, ac) in corners.items():
        if (ar, ac) not in norm:
            continue
        if not any(r == ar and c != ac for r, c in norm):
            continue
        if not any(c == ac and r != ar for r, c in norm):
            continue
        if all(r == ar or c == ac for r, c in norm):
            matches.append(key)
    if len(matches) == 1:
        return matches[0]
    return None


def _extract_seed(region: list[tuple[int, int]], grid: np.ndarray, bg: int, wall: int) -> tuple[str, int] | None:
    colored = [(r, c) for r, c in region if int(grid[r, c]) not in {bg, wall}]
    if not colored or not _is_connected(colored):
        return None
    colors = {int(grid[r, c]) for r, c in colored}
    if len(colors) != 1:
        return None
    key = _anchor_key(colored)
    if key is None:
        return None
    return key, colors.pop()


def _find_wall_and_regions(
    grid: np.ndarray, bg: int
) -> tuple[int, np.ndarray, list[tuple[list[tuple[int, int]], str, int]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    best_score: tuple[int, int, int] | None = None
    best_wall = -1
    best_enclosed: np.ndarray | None = None
    best_regions: list[tuple[list[tuple[int, int]], str, int]] = []
    for wall in colors:
        passable = grid != wall
        enclosed = passable & ~_flood_from_border(passable)
        regions: list[tuple[list[tuple[int, int]], str, int]] = []
        for comp in _components(enclosed):
            seed = _extract_seed(comp, grid, bg, wall)
            if seed is None:
                continue
            key, color = seed
            regions.append((comp, key, color))
        score = (len(regions), sum(len(comp) for comp, _, _ in regions), int(np.sum(grid == wall)))
        if best_score is None or score > best_score:
            best_score = score
            best_wall = wall
            best_enclosed = enclosed
            best_regions = regions
    assert best_score is not None and best_score[0] > 0, "expected a wall color producing enclosed seeded regions"
    assert best_enclosed is not None
    return best_wall, best_enclosed, best_regions


def _split_intervals(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    intervals: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != prev + 1:
            intervals.append((start, prev))
            start = value
        prev = value
    intervals.append((start, prev))
    return intervals


def _legend_mapping(grid: np.ndarray, bg: int, wall: int, enclosed: np.ndarray) -> dict[str, np.ndarray]:
    legend_mask = (grid != bg) & (grid != wall) & ~enclosed
    rows = np.where(np.any(legend_mask, axis=1))[0]
    cols = np.where(np.any(legend_mask, axis=0))[0]
    assert rows.size > 0 and cols.size > 0, "expected a visible legend outside the rooms"
    intervals = _split_intervals(rows)
    assert len(intervals) == 4, "expected four legend entries"
    c0, c1 = int(cols[0]), int(cols[-1])
    assert c1 - c0 + 1 == LEGEND_WIDTH, "legend width must match 3x3 marker+template layout"

    mapping: dict[str, np.ndarray] = {}
    for r0, r1 in intervals:
        assert r1 - r0 + 1 == LEGEND_ENTRY_HEIGHT, "legend entry must occupy exactly three rows"
        entry = legend_mask[r0 : r1 + 1, c0 : c1 + 1]
        marker_points = [(int(r), int(c)) for r, c in np.argwhere(entry[:, :3])]
        key = _anchor_key(marker_points)
        assert key is not None, "legend marker must be an L-shape with one of four orientations"
        template = entry[:, 3:6].copy()
        assert template.shape == (3, 3) and template.any(), "legend template must be a nonempty 3x3 mask"
        mapping[key] = template

    assert set(mapping) == set(ORIENTATIONS), "legend must define all four orientations"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed-cluster orientation inside each enclosed region to look up a right-side legend
    texture, then tile that 3x3 texture across the whole region in the seed color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_mode(g)
    wall, enclosed, regions = _find_wall_and_regions(g, bg)
    templates = _legend_mapping(g, bg, wall, enclosed)

    out = g.copy()
    for region, key, color in regions:
        template = templates[key]
        r0, c0, _, _ = _bbox(region)
        for r, c in region:
            out[r, c] = color if template[(r - r0) % 3, (c - c0) % 3] else bg
    return out


def _seed_variants(key: str) -> list[list[tuple[int, int]]]:
    variants = {
        "TL": [
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (0, 1), (0, 2), (1, 0)],
            [(0, 0), (0, 1), (1, 0), (2, 0)],
        ],
        "TR": [
            [(0, 0), (0, 1), (1, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 1)],
        ],
        "BL": [
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (1, 0), (1, 1), (1, 2)],
            [(0, 0), (1, 0), (2, 0), (2, 1)],
        ],
        "BR": [
            [(0, 1), (1, 0), (1, 1)],
            [(0, 2), (1, 0), (1, 1), (1, 2)],
            [(0, 1), (1, 1), (2, 0), (2, 1)],
        ],
    }
    return variants[key]


def _rectangle_mask(max_dim: int) -> np.ndarray:
    height = random.randint(4, max_dim)
    width = random.randint(4, max_dim)
    return np.ones((height, width), dtype=bool)


def _l_region_mask(max_dim: int) -> np.ndarray:
    height = random.randint(5, max_dim)
    width = random.randint(5, max_dim)
    arm_v = random.randint(2, width - 2)
    arm_h = random.randint(2, height - 2)
    anchor = random.choice(ORIENTATIONS)
    mask = np.zeros((height, width), dtype=bool)
    if anchor == "TL":
        mask[:, :arm_v] = True
        mask[:arm_h, :] = True
    elif anchor == "TR":
        mask[:, width - arm_v :] = True
        mask[:arm_h, :] = True
    elif anchor == "BL":
        mask[:, :arm_v] = True
        mask[height - arm_h :, :] = True
    else:
        mask[:, width - arm_v :] = True
        mask[height - arm_h :, :] = True
    return mask


def _room_interior(max_dim: int) -> np.ndarray:
    if random.random() < 0.5:
        return _rectangle_mask(max_dim)
    return _l_region_mask(max_dim)


def _place_seed(interior: np.ndarray, key: str, color: int) -> np.ndarray | None:
    positions = [(r, c) for r in range(interior.shape[0]) for c in range(interior.shape[1])]
    random.shuffle(positions)
    variants = _seed_variants(key)
    random.shuffle(variants)
    for cells in variants:
        h = 1 + max(r for r, _ in cells)
        w = 1 + max(c for _, c in cells)
        for top, left in positions:
            if top + h > interior.shape[0] or left + w > interior.shape[1]:
                continue
            coords = [(top + r, left + c) for r, c in cells]
            if all(interior[r, c] for r, c in coords):
                seed = np.zeros_like(interior, dtype=int)
                for r, c in coords:
                    seed[r, c] = color
                return seed
    return None


def _room_grid(interior: np.ndarray, seed: np.ndarray, wall_color: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = interior.shape
    inner = np.zeros((h + 2, w + 2), dtype=bool)
    inner[1 : h + 1, 1 : w + 1] = interior
    wall = np.zeros_like(inner)
    for r, c in np.argwhere(inner):
        for nr, nc in _neighbors4(int(r), int(c)):
            if not inner[nr, nc]:
                wall[nr, nc] = True

    grid = np.zeros_like(inner, dtype=int)
    grid[wall] = wall_color
    for r, c in np.argwhere(seed):
        grid[int(r) + 1, int(c) + 1] = int(seed[r, c])
    return grid, inner | wall


def _draw_legend(
    grid: np.ndarray,
    legend_color: int,
    template_map: dict[str, np.ndarray],
    legend_order: list[str],
    gaps: list[int],
    top: int,
) -> None:
    left = grid.shape[1] - LEGEND_WIDTH
    row = top
    for idx, key in enumerate(legend_order):
        marker = LEGEND_MARKERS[key]
        template = template_map[key]
        grid[row : row + 3, left : left + 3][marker] = legend_color
        grid[row : row + 3, left + 3 : left + 6][template] = legend_color
        row += LEGEND_ENTRY_HEIGHT
        if idx < 3:
            row += gaps[idx]


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.randint(*REGION_COUNT_RANGE)
    max_dim = 8 if n_regions <= 2 else 6

    gaps = [random.choice(LEGEND_GAP_CHOICES) for _ in range(3)]
    total_h = 4 * LEGEND_ENTRY_HEIGHT + sum(gaps)
    min_height = max(GRID_H_RANGE[0], total_h + 2)
    height = random.randint(min_height, GRID_H_RANGE[1])
    width = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    play_width = width - LEGEND_WIDTH - PLAY_GAP
    if play_width < 10:
        return None

    palette = random.sample(range(1, 10), 2 + n_regions)
    wall_color = palette[0]
    legend_color = palette[1]
    seed_colors = palette[2:]

    template_map = dict(zip(ORIENTATIONS, random.sample(TEMPLATE_LIBRARY, 4)))
    grid = np.zeros((height, width), dtype=int)
    legend_order = random.sample(list(ORIENTATIONS), k=4)
    top = random.randint(1, height - total_h - 1)
    _draw_legend(grid, legend_color, template_map, legend_order, gaps, top)

    rooms = []
    for idx in range(n_regions):
        key = random.choice(ORIENTATIONS)
        for _ in range(50):
            interior = _room_interior(max_dim)
            seed = _place_seed(interior, key, seed_colors[idx])
            if seed is None:
                continue
            room_grid, footprint = _room_grid(interior, seed, wall_color)
            rooms.append((room_grid, footprint))
            break
        else:
            return None

    rooms.sort(key=lambda item: int(np.sum(item[1])), reverse=True)
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:, play_width:] = True
    for room_grid, footprint in rooms:
        rh, rw = room_grid.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, height - rh - 1)
            left = random.randint(1, play_width - rw - 1)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(height, top + rh + 1)
            c1 = min(width, left + rw + 1)
            if np.any(occupied[r0:r1, c0:c1]):
                continue
            patch = room_grid != BG
            grid[top : top + rh, left : left + rw][patch] = room_grid[patch]
            occupied[top : top + rh, left : left + rw] |= footprint
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the textured-room task")
