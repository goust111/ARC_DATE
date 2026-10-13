from __future__ import annotations

import random

import numpy as np


BG = 0
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

LEGEND_COLORS = (GREEN, YELLOW, ORANGE, MAGENTA)
CONTAINER_COLOR_POOL = (1, 2, 5, 8, 9)

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
INTERIOR_H_RANGE = (2, 4)
INTERIOR_W_RANGE = (2, 4)
GEN_ATTEMPTS = 400

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = DIRS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

_RNG = random.Random()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    dirs = DIRS8 if connectivity == 8 else DIRS4
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr in (r0, r1) or cc in (c0, c1):
                cells.append((rr, cc))
    return cells


def _is_hollow_rectangle(coords: list[tuple[int, int]]) -> bool:
    coord_set = set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return False
    if len(coord_set) != 2 * height + 2 * width - 4:
        return False
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if ((rr, cc) in coord_set) != on_border:
                return False
    return True


def _detect_containers(
    grid: np.ndarray, bg: int, legend_colors: set[int]
) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for value in np.unique(grid):
        color = int(value)
        if color == bg or color in legend_colors:
            continue
        comps = _components(grid == color, connectivity=8)
        assert comps, f"missing component for color {color}"
        comps.sort(key=len, reverse=True)
        border = comps[0]
        assert _is_hollow_rectangle(border), f"color {color} must contain one hollow rectangle"
        scatter: list[tuple[int, int]] = []
        for comp in comps[1:]:
            assert len(comp) == 1, f"extra color-{color} pieces must be singleton scatter pixels"
            scatter.extend(comp)
        containers.append(
            {
                "color": color,
                "bbox": _bbox(border),
                "border": set(border),
                "scatter": scatter,
            }
        )
    containers.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))  # stable scan order
    assert containers, "expected at least one container"
    return containers


def _quadrant_index(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    upper = 2 * r <= r0 + r1
    left = 2 * c <= c0 + c1
    if upper and left:
        return 0
    if upper and not left:
        return 1
    if not upper and left:
        return 2
    return 3


def _legend_from_grid(grid: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert same-colored scatter pixels around hollow containers into quadrant-coded fills inside those containers.

    Input structure:
    - The background is the dominant color.
    - The four grid corners contain one-pixel legend markers: top-left, top-right, bottom-left, and bottom-right encode the four quadrant fill colors.
    - Elsewhere there are 2 or 3 hollow axis-aligned rectangular containers with distinct border colors, plus isolated single pixels whose color matches one container border.

    Rule:
    - For each container, collect all non-border pixels of its border color and treat them as scatter pixels belonging to that container.
    - Classify each scatter pixel by its position relative to the container center: ties on the center row count as upper, and ties on the center column count as left.
    - Count scatter pixels in quadrant order top-left, top-right, bottom-left, bottom-right, then fill the container interior in row-major order with the corresponding corner-legend colors; truncate if the interior runs out of cells.
    - Keep the container borders and corner legend unchanged, and erase the scatter pixels back to the background color.

    Key insight:
    - Color identifies which container owns each scatter pixel, while the corner markers act as a visible quadrant legend that converts spatial relation into the fill color written inside the matching container.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    legend = _legend_from_grid(g)
    assert len(set(legend)) == 4, "corner legend colors must be distinct"

    out = g.copy()
    containers = _detect_containers(g, bg, set(legend))
    for container in containers:
        bbox = container["bbox"]
        scatter = container["scatter"]
        counts = [0, 0, 0, 0]
        for r, c in scatter:
            counts[_quadrant_index(int(r), int(c), bbox)] += 1
            out[int(r), int(c)] = bg

        fill_values: list[int] = []
        for idx, count in enumerate(counts):
            fill_values.extend([int(legend[idx])] * int(count))

        r0, c0, r1, c1 = bbox
        fill_index = 0
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if fill_index < len(fill_values):
                    out[rr, cc] = fill_values[fill_index]
                    fill_index += 1
                else:
                    out[rr, cc] = bg

    return out


def _mark_halo(mask: np.ndarray, r: int, c: int, radius: int = 1) -> None:
    h, w = mask.shape
    for rr in range(max(0, r - radius), min(h, r + radius + 1)):
        for cc in range(max(0, c - radius), min(w, c + radius + 1)):
            mask[rr, cc] = True


def _mark_box(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    for rr in range(max(0, r0 - pad), min(h, r1 + pad + 1)):
        for cc in range(max(0, c0 - pad), min(w, c1 + pad + 1)):
            mask[rr, cc] = True


def _draw_container(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _place_containers(
    rng: random.Random, sizes: list[tuple[int, int]], height: int, width: int
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    for corner in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
        _mark_halo(blocked, corner[0], corner[1], radius=1)

    placements: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    for idx in order:
        box_h, box_w = sizes[idx]
        row_hi = height - box_h - 1
        col_hi = width - box_w - 1
        if row_hi < 1 or col_hi < 1:
            return None

        candidates = [(r0, c0) for r0 in range(1, row_hi + 1) for c0 in range(1, col_hi + 1)]
        rng.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            bbox = (r0, c0, r1, c1)
            placements[idx] = bbox
            _mark_box(blocked, bbox, pad=1)
            placed = True
            break
        if not placed:
            return None

    assert all(bbox is not None for bbox in placements)
    return [bbox for bbox in placements if bbox is not None]


def _split_total_across_quadrants(rng: random.Random, total: int) -> list[int]:
    counts = [0, 0, 0, 0]
    if total >= 4 and rng.random() < 0.4:
        seeds = [0, 1, 2, 3]
    elif total >= 3 and rng.random() < 0.7:
        seeds = rng.sample(range(4), 3)
    elif total >= 2:
        seeds = rng.sample(range(4), 2)
    else:
        seeds = [rng.randrange(4)]

    for quadrant in seeds:
        counts[int(quadrant)] += 1
    for _ in range(total - len(seeds)):
        counts[rng.randrange(4)] += 1
    return counts


def _sample_counts(rng: random.Random, capacities: list[int]) -> list[list[int]]:
    overflow_idx = rng.randrange(len(capacities)) if rng.random() < 0.5 else None
    counts_by_container: list[list[int]] = []
    for idx, capacity in enumerate(capacities):
        if idx == overflow_idx:
            extra = rng.randint(1, min(4, capacity))
            total = capacity + extra
        else:
            total = rng.randint(1, min(capacity, 7))
        counts_by_container.append(_split_total_across_quadrants(rng, total))
    return counts_by_container


def _special_request(
    rng: random.Random, bbox: tuple[int, int, int, int], counts: list[int]
) -> tuple[int, str] | None:
    r0, c0, r1, c1 = bbox
    options: list[tuple[int, str]] = []
    if (r0 + r1) % 2 == 0:
        if counts[0] > 0:
            options.append((0, "row"))
        if counts[1] > 0:
            options.append((1, "row"))
    if (c0 + c1) % 2 == 0:
        if counts[0] > 0:
            options.append((0, "col"))
        if counts[2] > 0:
            options.append((2, "col"))
    if not options or rng.random() >= 0.5:
        return None
    return rng.choice(options)


def _scatter_requests(
    rng: random.Random, bbox: tuple[int, int, int, int], counts: list[int]
) -> list[tuple[int, str | None]]:
    remaining = list(counts)
    requests: list[tuple[int, str | None]] = []
    special = _special_request(rng, bbox, remaining)
    if special is not None:
        quadrant, axis = special
        requests.append((quadrant, axis))
        remaining[quadrant] -= 1
    for quadrant, count in enumerate(remaining):
        requests.extend((quadrant, None) for _ in range(count))
    head = requests[:1]
    tail = requests[1:]
    rng.shuffle(tail)
    return head + tail


def _scatter_candidates(
    blocked: np.ndarray,
    bbox: tuple[int, int, int, int],
    quadrant: int,
    axis: str | None,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    row_center = None
    col_center = None
    r0, c0, r1, c1 = bbox
    if (r0 + r1) % 2 == 0:
        row_center = (r0 + r1) // 2
    if (c0 + c1) % 2 == 0:
        col_center = (c0 + c1) // 2

    candidates: list[tuple[int, int]] = []
    for rr in range(h):
        for cc in range(w):
            if blocked[rr, cc]:
                continue
            if axis == "row" and rr != row_center:
                continue
            if axis == "col" and cc != col_center:
                continue
            if _quadrant_index(rr, cc, bbox) != quadrant:
                continue
            candidates.append((rr, cc))
    return candidates


def _validate_instance(
    grid: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    colors: list[int],
    counts_by_container: list[list[int]],
    scatter_by_container: list[list[tuple[int, int]]],
) -> None:
    out = main(grid)
    assert out.shape == grid.shape

    legend = _legend_from_grid(grid)
    h, w = grid.shape
    assert legend == LEGEND_COLORS
    assert out[0, 0] == GREEN and out[0, w - 1] == YELLOW
    assert out[h - 1, 0] == ORANGE and out[h - 1, w - 1] == MAGENTA

    for bbox, color, counts, scatter in zip(bboxes, colors, counts_by_container, scatter_by_container):
        expected_border = set(_border_cells(bbox))
        actual_color_cells = {tuple(map(int, pos)) for pos in np.argwhere(out == color)}
        assert actual_color_cells == expected_border
        for r, c in scatter:
            assert out[r, c] == BG

        fill_values: list[int] = []
        for idx, count in enumerate(counts):
            fill_values.extend([int(legend[idx])] * int(count))
        r0, c0, r1, c1 = bbox
        interior = out[r0 + 1 : r1, c0 + 1 : c1].ravel()
        prefix = np.array(fill_values[: interior.size], dtype=int)
        if prefix.size:
            assert np.array_equal(interior[: prefix.size], prefix)
        if prefix.size < interior.size:
            assert np.all(interior[prefix.size :] == BG)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    n_containers = 2 if rng.random() < 0.5 else 3
    colors = list(rng.sample(CONTAINER_COLOR_POOL, n_containers))

    outer_sizes: list[tuple[int, int]] = []
    capacities: list[int] = []
    for _ in range(n_containers):
        inner_h = rng.randint(*INTERIOR_H_RANGE)
        inner_w = rng.randint(*INTERIOR_W_RANGE)
        outer_sizes.append((inner_h + 2, inner_w + 2))
        capacities.append(inner_h * inner_w)

    bboxes = _place_containers(rng, outer_sizes, height, width)
    if bboxes is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[0, 0] = GREEN
    grid[0, width - 1] = YELLOW
    grid[height - 1, 0] = ORANGE
    grid[height - 1, width - 1] = MAGENTA

    for bbox, color in zip(bboxes, colors):
        _draw_container(grid, bbox, color)

    counts_by_container = _sample_counts(rng, capacities)
    requests_by_container = [
        _scatter_requests(rng, bbox, counts) for bbox, counts in zip(bboxes, counts_by_container)
    ]

    blocked = np.zeros((height, width), dtype=bool)
    for corner in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
        _mark_halo(blocked, corner[0], corner[1], radius=1)
    for bbox in bboxes:
        _mark_box(blocked, bbox, pad=1)

    scatter_by_container: list[list[tuple[int, int]]] = [[] for _ in range(n_containers)]
    order = sorted(range(n_containers), key=lambda idx: len(requests_by_container[idx]), reverse=True)
    for idx in order:
        pending = list(requests_by_container[idx])
        while pending:
            best_index = -1
            best_candidates: list[tuple[int, int]] | None = None
            for req_index, (quadrant, axis) in enumerate(pending):
                candidates = _scatter_candidates(blocked, bboxes[idx], quadrant, axis)
                if not candidates:
                    return None
                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_index = req_index
                    best_candidates = candidates

            assert best_candidates is not None
            quadrant, axis = pending.pop(best_index)
            _ = quadrant, axis
            r, c = rng.choice(best_candidates)
            grid[r, c] = colors[idx]
            scatter_by_container[idx].append((r, c))
            _mark_halo(blocked, r, c, radius=1)

    try:
        _validate_instance(grid, bboxes, colors, counts_by_container, scatter_by_container)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-quadrant puzzle")
