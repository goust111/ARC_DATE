import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
EDGES = ("top", "bottom", "left", "right")
ACTIVE_EDGE_GROUPS = {
    ("odd", True): [
        ("top",),
        ("left",),
        ("top", "bottom", "right"),
        ("left", "bottom", "right"),
        ("top", "left", "bottom"),
        ("top", "left", "right"),
    ],
    ("odd", False): [
        ("bottom",),
        ("right",),
    ],
    ("even", True): [
        ("top", "left"),
        ("top", "bottom"),
        ("top", "right"),
        ("left", "bottom"),
        ("left", "right"),
        ("top", "bottom", "left", "right"),
    ],
    ("even", False): [
        ("bottom", "right"),
    ],
}

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (18, 32)
EMPTY_COUNT_RANGE = (2, 4)
SAMPLE_H_RANGE = (4, 7)
SAMPLE_W_RANGE = (4, 7)
EMPTY_H_RANGE = (4, 9)
EMPTY_W_RANGE = (4, 10)
OUTER_MARGIN = 1
PLACEMENT_PAD = 1
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_containers(grid: np.ndarray, bg: int) -> list[dict]:
    containers: list[dict] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        if np.any(patch[border] == bg):
            continue
        if h <= 2 or w <= 2:
            continue
        interior = patch[1:-1, 1:-1]
        interior_empty = bool(np.all(interior == bg))
        interior_filled = bool(np.all(interior != bg))
        if not (interior_empty or interior_filled):
            continue
        colors = set(map(int, patch[patch != bg].reshape(-1)))
        if len(colors) != 2:
            continue
        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "colors": colors,
                "interior_empty": interior_empty,
            }
        )
    return containers


def _infer_marker_color(containers: list[dict]) -> int:
    shared = set(containers[0]["colors"])
    for container in containers[1:]:
        shared &= set(container["colors"])
    if len(shared) == 1:
        return int(next(iter(shared)))

    occurrences: Counter[int] = Counter()
    for container in containers:
        for color in container["colors"]:
            occurrences[int(color)] += 1
    best = max(occurrences.values())
    candidates = [color for color, count in occurrences.items() if count == best]
    assert len(candidates) == 1, "marker color must be uniquely inferable"
    return int(candidates[0])


def _active_edges(patch: np.ndarray, marker: int) -> tuple[str, ...]:
    active = []
    if np.any(patch[0, 1:-1] == marker):
        active.append("top")
    if np.any(patch[-1, 1:-1] == marker):
        active.append("bottom")
    if np.any(patch[1:-1, 0] == marker):
        active.append("left")
    if np.any(patch[1:-1, -1] == marker):
        active.append("right")
    return tuple(active)


def _filled_interior(
    height: int,
    width: int,
    border_color: int,
    marker_color: int,
    active_edges: tuple[str, ...],
) -> np.ndarray:
    fill = np.empty((height - 2, width - 2), dtype=int)
    start_with_marker = "top" in active_edges or "left" in active_edges
    checkerboard = len(active_edges) % 2 == 1
    for r in range(height - 2):
        for c in range(width - 2):
            if checkerboard:
                same_phase = (r + c) % 2 == 0
            else:
                same_phase = r % 2 == 0
            use_marker = start_with_marker if same_phase else not start_with_marker
            fill[r, c] = marker_color if use_marker else border_color
    return fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill empty rectangular containers using shared border markers.

    Each container uses a unique border color plus one shared marker color.
    Marker positions on the border determine the interior pattern:
    - top/left active -> top-left interior cell is marker color, else border color
    - odd number of active sides -> checkerboard
    - even number of active sides -> horizontal stripes
    Filled sample containers stay unchanged; only empty containers are filled.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    containers = _find_containers(g, bg)
    assert len(containers) >= 2, "expected at least a sample container and one empty container"

    marker = _infer_marker_color(containers)
    out = g.copy()

    for container in containers:
        if not container["interior_empty"]:
            continue
        r0, c0, r1, c1 = container["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        active = _active_edges(patch, marker)
        assert 1 <= len(active) <= 4, "each container must expose at least one active side"
        border_color = next(color for color in container["colors"] if color != marker)
        out[r0 + 1 : r1, c0 + 1 : c1] = _filled_interior(
            r1 - r0 + 1, c1 - c0 + 1, border_color, marker, active
        )

    return out


def _choose_active_edges(rng: random.Random, groups: list[tuple[str, bool]]) -> list[tuple[str, ...]]:
    return [rng.choice(ACTIVE_EDGE_GROUPS[group]) for group in groups]


def _place_rectangles(
    rng: random.Random, height: int, width: int, shapes: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)

    for idx in order:
        h, w = shapes[idx]
        row_hi = height - h - OUTER_MARGIN
        col_hi = width - w - OUTER_MARGIN
        if row_hi < OUTER_MARGIN or col_hi < OUTER_MARGIN:
            return None
        placed = False
        for _ in range(200):
            r0 = rng.randint(OUTER_MARGIN, row_hi)
            c0 = rng.randint(OUTER_MARGIN, col_hi)
            rr0 = max(0, r0 - PLACEMENT_PAD)
            cc0 = max(0, c0 - PLACEMENT_PAD)
            rr1 = min(height, r0 + h + PLACEMENT_PAD)
            cc1 = min(width, c0 + w + PLACEMENT_PAD)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _edge_position(rng: random.Random, edge: str, h: int, w: int) -> tuple[int, int]:
    if edge == "top":
        return 0, rng.randint(1, w - 2)
    if edge == "bottom":
        return h - 1, rng.randint(1, w - 2)
    if edge == "left":
        return rng.randint(1, h - 2), 0
    assert edge == "right"
    return rng.randint(1, h - 2), w - 1


def _draw_container(
    grid: np.ndarray,
    rng: random.Random,
    r0: int,
    c0: int,
    h: int,
    w: int,
    border_color: int,
    marker_color: int,
    active_edges: tuple[str, ...],
    filled: bool,
) -> None:
    grid[r0, c0 : c0 + w] = border_color
    grid[r0 + h - 1, c0 : c0 + w] = border_color
    grid[r0 : r0 + h, c0] = border_color
    grid[r0 : r0 + h, c0 + w - 1] = border_color

    for edge in active_edges:
        rr, cc = _edge_position(rng, edge, h, w)
        grid[r0 + rr, c0 + cc] = marker_color

    if filled:
        grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = _filled_interior(
            h, w, border_color, marker_color, active_edges
        )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_empty = rng.randint(*EMPTY_COUNT_RANGE)
    n_containers = n_empty + 1

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    marker = palette[1]
    border_colors = palette[2 : 2 + n_containers]
    if len(border_colors) != n_containers:
        return None

    if rng.random() < 0.5:
        empty_groups = [("odd", True), ("even", False)]
    else:
        empty_groups = [("odd", False), ("even", True)]
    all_group_keys = list(ACTIVE_EDGE_GROUPS)
    while len(empty_groups) < n_empty:
        empty_groups.append(rng.choice(all_group_keys))
    rng.shuffle(empty_groups)
    empty_edges = _choose_active_edges(rng, empty_groups)
    sample_edges = rng.choice(ACTIVE_EDGE_GROUPS[rng.choice(all_group_keys)])

    specs = []
    sample_index = rng.randrange(n_containers)
    empty_idx = 0
    for idx in range(n_containers):
        if idx == sample_index:
            specs.append(
                {
                    "filled": True,
                    "h": rng.randint(*SAMPLE_H_RANGE),
                    "w": rng.randint(*SAMPLE_W_RANGE),
                    "edges": sample_edges,
                }
            )
        else:
            specs.append(
                {
                    "filled": False,
                    "h": rng.randint(*EMPTY_H_RANGE),
                    "w": rng.randint(*EMPTY_W_RANGE),
                    "edges": empty_edges[empty_idx],
                }
            )
            empty_idx += 1

    min_h = max(GRID_H_RANGE[0], 6 + 4 * n_containers)
    min_w = max(GRID_W_RANGE[0], 8 + 4 * n_containers)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    height = rng.randint(min_h, GRID_H_RANGE[1])
    width = rng.randint(min_w, GRID_W_RANGE[1])

    shapes = [(spec["h"], spec["w"]) for spec in specs]
    positions = _place_rectangles(rng, height, width, shapes)
    if positions is None or len(positions) != n_containers:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for idx, (r0, c0) in enumerate(positions):
        spec = specs[idx]
        _draw_container(
            grid,
            rng,
            r0,
            c0,
            spec["h"],
            spec["w"],
            border_colors[idx],
            marker,
            spec["edges"],
            spec["filled"],
        )

    counts = Counter(map(int, grid.reshape(-1)))
    non_bg_counts = [count for color, count in counts.items() if color != bg]
    if not non_bg_counts or counts[bg] <= max(non_bg_counts):
        return None

    try:
        containers = _find_containers(grid, bg)
        if len(containers) != n_containers:
            return None
        if sum(not container["interior_empty"] for container in containers) != 1:
            return None
        marker_guess = _infer_marker_color(containers)
        if marker_guess != marker:
            return None
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
