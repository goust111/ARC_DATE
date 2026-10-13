from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

TEMPLATE_SIZE = 3
OUTER_RING_SIZE = 7
MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 17
GRID_MAX = 20
RING_COUNT_OPTIONS = (2, 3)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _label_regions(traversable: np.ndarray) -> tuple[np.ndarray, list[bool], list[np.ndarray]]:
    h, w = traversable.shape
    labels = np.full((h, w), -1, dtype=int)
    touches_edge: list[bool] = []
    components: list[np.ndarray] = []
    next_label = 0

    for r in range(h):
        for c in range(w):
            if not traversable[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            coords: list[tuple[int, int]] = []
            touches = False
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and traversable[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            touches_edge.append(touches)
            components.append(np.array(coords, dtype=int))
            next_label += 1

    return labels, touches_edge, components


def _stamp_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _in_template(rr: int, cc: int) -> bool:
    return 0 <= rr < TEMPLATE_SIZE and 0 <= cc < TEMPLATE_SIZE


def _apply_stamp(out: np.ndarray, template: np.ndarray, center: tuple[int, int]) -> None:
    h, w = out.shape
    r, c = center
    for tr in range(TEMPLATE_SIZE):
        for tc in range(TEMPLATE_SIZE):
            rr = r + tr - 1
            cc = c + tc - 1
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(template[tr, tc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gate template stamps by whether each green marker lies in a topologically enclosed region.

    Input structure:
    - The top-left 3x3 block is a red/yellow template tile that must be copied as a whole.
    - Grey cells are wall-like obstacles; black cells are background; green singletons are markers placed on background regions.
    - Background regions are defined by 4-connectivity through black-or-green cells, while grey walls and the template remain fixed obstacles.

    Rule:
    - Read the top-left 3x3 block as the stamp template.
    - Split the black background into 4-connected regions, treating each green marker as occupying a background cell rather than as a barrier.
    - If a marker's region reaches any grid edge, erase that marker to black.
    - If a marker's region is fully enclosed, stamp the 3x3 template centered on that marker.

    Key insight:
    - The green pixel is only a query point; whether it survives depends entirely on the topology of its containing background region, not on the marker's local neighborhood.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= TEMPLATE_SIZE

    template = g[:TEMPLATE_SIZE, :TEMPLATE_SIZE].copy()
    traversable = (g == BLACK) | (g == GREEN)
    labels, touches_edge, _components = _label_regions(traversable)

    out = g.copy()
    out[g == GREEN] = BLACK

    for marker in np.argwhere(g == GREEN):
        r, c = map(int, marker)
        region_id = int(labels[r, c])
        assert region_id >= 0, "marker must lie in a traversable region"
        if not touches_edge[region_id]:
            _apply_stamp(out, template, (r, c))

    out[g == GREY] = GREY
    out[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = template
    return out


def _random_template() -> np.ndarray:
    while True:
        template = np.random.choice([RED, YELLOW], size=(TEMPLATE_SIZE, TEMPLATE_SIZE))
        if len(np.unique(template)) == 2:
            return template.astype(int)


def _ring_bbox(slot: str, h: int, w: int) -> tuple[int, int, int, int]:
    if slot == "top_right":
        r0, c0 = 1, w - OUTER_RING_SIZE - 1
    elif slot == "bottom_left":
        r0, c0 = h - OUTER_RING_SIZE - 1, 1
    elif slot == "bottom_right":
        r0, c0 = h - OUTER_RING_SIZE - 1, w - OUTER_RING_SIZE - 1
    else:
        raise ValueError(f"unknown slot: {slot}")
    return r0, c0, r0 + OUTER_RING_SIZE - 1, c0 + OUTER_RING_SIZE - 1


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _draw_line(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = GREY
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = GREY
        return
    raise ValueError("line must be axis-aligned")


def _connector_start(bbox: tuple[int, int, int, int], hub: tuple[int, int]) -> tuple[tuple[int, int], str]:
    r0, c0, r1, c1 = bbox
    hr, hc = hub
    choices: list[tuple[int, tuple[int, int], str]] = []
    if hc < c0:
        start = (r0 + OUTER_RING_SIZE // 2, c0)
        choices.append((abs(hr - start[0]) + abs(hc - start[1]), start, "horizontal"))
    if hc > c1:
        start = (r0 + OUTER_RING_SIZE // 2, c1)
        choices.append((abs(hr - start[0]) + abs(hc - start[1]), start, "horizontal"))
    if hr < r0:
        start = (r0, c0 + OUTER_RING_SIZE // 2)
        choices.append((abs(hr - start[0]) + abs(hc - start[1]), start, "vertical"))
    if hr > r1:
        start = (r1, c0 + OUTER_RING_SIZE // 2)
        choices.append((abs(hr - start[0]) + abs(hc - start[1]), start, "vertical"))
    assert choices, "hub must lie outside each ring"
    _dist, start, orientation = min(choices, key=lambda item: item[0])
    return start, orientation


def _connect_ring_to_hub(grid: np.ndarray, bbox: tuple[int, int, int, int], hub: tuple[int, int]) -> None:
    start, orientation = _connector_start(bbox, hub)
    if orientation == "horizontal":
        elbow = (start[0], hub[1])
    else:
        elbow = (hub[0], start[1])
    _draw_line(grid, start, elbow)
    _draw_line(grid, elbow, hub)


def _maybe_add_spur(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    if random.random() < 0.35:
        return
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    options = [
        ((r0 + OUTER_RING_SIZE // 2, c0), 0, -1),
        ((r0 + OUTER_RING_SIZE // 2, c1), 0, 1),
        ((r0, c0 + OUTER_RING_SIZE // 2), -1, 0),
        ((r1, c0 + OUTER_RING_SIZE // 2), 1, 0),
    ]
    random.shuffle(options)
    for anchor, dr, dc in options:
        length = random.randint(1, 2)
        cells = []
        for step in range(1, length + 1):
            rr = anchor[0] + dr * step
            cc = anchor[1] + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                cells = []
                break
            if _in_template(rr, cc) or grid[rr, cc] != BLACK:
                cells = []
                break
            cells.append((rr, cc))
        if cells:
            for rr, cc in cells:
                grid[rr, cc] = GREY
            return


def _pick_closed_marker(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = bbox
    return random.randint(r0 + 2, r0 + 4), random.randint(c0 + 2, c0 + 4)


def _stamp_is_locally_clear(grid: np.ndarray, center: tuple[int, int]) -> bool:
    h, w = grid.shape
    for rr, cc in _stamp_cells(center):
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if _in_template(rr, cc):
            return False
        if grid[rr, cc] != BLACK:
            return False
    return True


def _choose_open_marker(grid: np.ndarray, forbidden: set[tuple[int, int]]) -> tuple[int, int] | None:
    labels, touches_edge, _components = _label_regions(grid == BLACK)
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BLACK:
                continue
            region_id = int(labels[r, c])
            if region_id < 0 or not touches_edge[region_id]:
                continue
            if not _stamp_is_locally_clear(grid, (r, c)):
                continue
            footprint = set(_stamp_cells((r, c)))
            if footprint & forbidden:
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _naive_local_gate_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    template = g[:TEMPLATE_SIZE, :TEMPLATE_SIZE].copy()
    out = g.copy()
    out[g == GREEN] = BLACK
    for marker in np.argwhere(g == GREEN):
        center = tuple(map(int, marker))
        if _stamp_is_locally_clear(np.where(g == GREEN, BLACK, g), center):
            _apply_stamp(out, template, center)
    out[g == GREY] = GREY
    out[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = template
    return out


def _validate_generated_case(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.any(grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE] == GREY):
        return False

    markers = [tuple(map(int, p)) for p in np.argwhere(grid == GREEN)]
    if not 3 <= len(markers) <= 6:
        return False

    labels, touches_edge, _components = _label_regions((grid == BLACK) | (grid == GREEN))
    seen_regions: set[int] = set()
    open_markers: list[tuple[int, int]] = []
    closed_markers: list[tuple[int, int]] = []
    occupied_footprints: set[tuple[int, int]] = set()

    for marker in markers:
        region_id = int(labels[marker])
        if region_id < 0 or region_id in seen_regions:
            return False
        seen_regions.add(region_id)
        footprint = set(_stamp_cells(marker))
        if touches_edge[region_id]:
            open_markers.append(marker)
        else:
            closed_markers.append(marker)
            if not _stamp_is_locally_clear(np.where(grid == GREEN, BLACK, grid), marker):
                return False
            if footprint & occupied_footprints:
                return False
            occupied_footprints |= footprint

    if len(open_markers) != 1 or len(closed_markers) < 2:
        return False
    if not _stamp_is_locally_clear(np.where(grid == GREEN, BLACK, grid), open_markers[0]):
        return False
    if set(_stamp_cells(open_markers[0])) & occupied_footprints:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[:TEMPLATE_SIZE, :TEMPLATE_SIZE], grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE]):
        return False
    if np.any(out[grid == GREY] != GREY):
        return False
    if out[open_markers[0]] != BLACK:
        return False
    if np.array_equal(out, _naive_local_gate_output(grid)):
        return False

    template_colors = set(map(int, np.unique(grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE])))
    stamped_mask = np.zeros_like(grid, dtype=bool)
    stamped_mask[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = True
    visible_stamp = np.isin(out, list(template_colors)) & ~stamped_mask
    return bool(np.any(visible_stamp))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    grid[:TEMPLATE_SIZE, :TEMPLATE_SIZE] = _random_template()

    chosen_slots = random.sample(["top_right", "bottom_left", "bottom_right"], k=random.choice(RING_COUNT_OPTIONS))
    bboxes = [_ring_bbox(slot, h, w) for slot in chosen_slots]

    for bbox in bboxes:
        _draw_ring(grid, bbox)

    hub = (h // 2, w // 2)
    if _in_template(*hub):
        hub = (h // 2, w // 2 + 1)
    for bbox in bboxes:
        _connect_ring_to_hub(grid, bbox, hub)
    for bbox in bboxes:
        _maybe_add_spur(grid, bbox)

    closed_markers = [_pick_closed_marker(bbox) for bbox in bboxes]
    forbidden: set[tuple[int, int]] = set()
    for marker in closed_markers:
        forbidden |= set(_stamp_cells(marker))

    open_marker = _choose_open_marker(grid, forbidden)
    if open_marker is None:
        return None

    for r, c in closed_markers + [open_marker]:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = GREEN

    if not _validate_generated_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-region stamp task")
