from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BOX_OUTER = 5
BOX_INNER = 3
DIRS = ("top", "bottom", "left", "right")
TARGET_ANCHORS = {
    "top": (0, 1),
    "bottom": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}

GRID_RANGES = {
    2: ((18, 24), (18, 24)),
    3: ((22, 28), (22, 28)),
    4: ((24, 30), (24, 30)),
}
BOX_COUNT_RANGE = (2, 4)
BOX_EDGE_MARGIN = 3
BOX_PADDING = 2
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_box_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 != BOX_OUTER or c1 - c0 + 1 != BOX_OUTER:
        return False
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    actual = {tuple(map(int, cell)) for cell in coords}
    return actual == expected


def _dominant_direction(coords: np.ndarray, box_center: tuple[float, float]) -> str:
    center = coords.mean(axis=0)
    dr = float(center[0] - box_center[0])
    dc = float(center[1] - box_center[1])
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    return "bottom" if dr > 0 else "top"


def _normalize(coords: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _is_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    todo = [coords[0]]
    seen = {coords[0]}
    cells = set(coords)
    while todo:
        r, c = todo.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == len(coords)


def _round_half_up(num: int, den: int) -> int:
    return int((num + den // 2) // den)


def _shape_anchor(local_coords: tuple[tuple[int, int], ...], direction: str, exact: bool = False) -> tuple[int, int] | None:
    rows = [r for r, _ in local_coords]
    cols = [c for _, c in local_coords]
    if direction == "top":
        edge_r = min(rows)
        edge_vals = [c for r, c in local_coords if r == edge_r]
        if exact and sum(edge_vals) % len(edge_vals) != 0:
            return None
        return edge_r, _round_half_up(sum(edge_vals), len(edge_vals))
    if direction == "bottom":
        edge_r = max(rows)
        edge_vals = [c for r, c in local_coords if r == edge_r]
        if exact and sum(edge_vals) % len(edge_vals) != 0:
            return None
        return edge_r, _round_half_up(sum(edge_vals), len(edge_vals))
    if direction == "left":
        edge_c = min(cols)
        edge_vals = [r for r, c in local_coords if c == edge_c]
        if exact and sum(edge_vals) % len(edge_vals) != 0:
            return None
        return _round_half_up(sum(edge_vals), len(edge_vals)), edge_c
    edge_c = max(cols)
    edge_vals = [r for r, c in local_coords if c == edge_c]
    if exact and sum(edge_vals) % len(edge_vals) != 0:
        return None
    return _round_half_up(sum(edge_vals), len(edge_vals)), edge_c


def _place_local_shape(
    local_coords: tuple[tuple[int, int], ...], direction: str, exact_anchor: bool = False
) -> tuple[tuple[int, int], ...] | None:
    anchor = _shape_anchor(local_coords, direction, exact=exact_anchor)
    if anchor is None:
        return None
    tr, tc = TARGET_ANCHORS[direction]
    ar, ac = anchor
    shifted = tuple(sorted((r + tr - ar, c + tc - ac) for r, c in local_coords))
    if all(0 <= r < BOX_INNER and 0 <= c < BOX_INNER for r, c in shifted):
        return shifted
    return None


def _draw_box(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    r1 = r0 + BOX_OUTER - 1
    c1 = c0 + BOX_OUTER - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the 3x3 interior of the box with the same color.

    The fragment's color selects the target box, and the fragment's coarse direction
    relative to that box selects which side-center anchor inside the box is used.
    The fragment shape is translated without rotation or reflection.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    out = g.copy()

    for color in colors:
        comps = _find_components(g == color)
        boxes = [coords for coords in comps if _is_box_component(coords)]
        assert len(boxes) == 1, "each box color must have exactly one 5x5 frame"
        box = boxes[0]
        fragments = [coords for coords in comps if not _is_box_component(coords)]
        assert 1 <= len(fragments) <= 4, "each box must have 1-4 matching fragments"

        r0, c0, r1, c1 = _bbox(box)
        box_center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        used_dirs: set[str] = set()
        used_cells: set[tuple[int, int]] = set()

        for coords in fragments:
            direction = _dominant_direction(coords, box_center)
            assert direction not in used_dirs, "same-color fragments must occupy distinct directions"
            used_dirs.add(direction)

            fr0, fc0, _fr1, _fc1 = _bbox(coords)
            local = tuple(sorted((int(r - fr0), int(c - fc0)) for r, c in coords))
            placed = _place_local_shape(local, direction)
            assert placed is not None, "fragment shape must fit when anchored in the 3x3 box interior"

            for r, c in coords:
                out[int(r), int(c)] = bg
            for ir, ic in placed:
                cell = (r0 + 1 + ir, c0 + 1 + ic)
                assert cell not in used_cells, "placed fragments inside a box must not overlap"
                used_cells.add(cell)
                out[cell] = color

    return out


def _enumerate_shape_options() -> dict[str, list[dict[str, object]]]:
    cells = [(r, c) for r in range(BOX_INNER) for c in range(BOX_INNER)]
    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[tuple[tuple[int, int], ...]] = []
    for size in (1, 2, 3):
        for subset in combinations(cells, size):
            norm = _normalize(subset)
            if norm in seen or not _is_connected(norm):
                continue
            seen.add(norm)
            shapes.append(norm)

    by_dir: dict[str, list[dict[str, object]]] = {direction: [] for direction in DIRS}
    for shape in shapes:
        for direction in DIRS:
            placed = _place_local_shape(shape, direction, exact_anchor=True)
            if placed is None:
                continue
            by_dir[direction].append(
                {
                    "shape": shape,
                    "placed": frozenset(placed),
                    "area": len(shape),
                }
            )
    return by_dir


SHAPE_OPTIONS = _enumerate_shape_options()


def _choose_box_shapes(directions: list[str]) -> dict[str, dict[str, object]] | None:
    order = directions[:]
    random.shuffle(order)
    chosen: dict[str, dict[str, object]] = {}
    occupied: set[tuple[int, int]] = set()

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        direction = order[i]
        options = SHAPE_OPTIONS[direction][:]
        random.shuffle(options)
        options.sort(key=lambda item: (random.random(), -int(item["area"])))
        for option in options:
            placed = set(option["placed"])
            if placed & occupied:
                continue
            chosen[direction] = option
            occupied.update(placed)
            if backtrack(i + 1):
                return True
            occupied.difference_update(placed)
            chosen.pop(direction, None)
        return False

    if backtrack(0):
        return chosen
    return None


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    )


def _place_boxes(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    if height < BOX_OUTER + 2 * BOX_EDGE_MARGIN or width < BOX_OUTER + 2 * BOX_EDGE_MARGIN:
        return None
    candidates = [
        (r, c)
        for r in range(BOX_EDGE_MARGIN, height - BOX_OUTER - BOX_EDGE_MARGIN + 1)
        for c in range(BOX_EDGE_MARGIN, width - BOX_OUTER - BOX_EDGE_MARGIN + 1)
    ]
    for _ in range(200):
        random.shuffle(candidates)
        placed: list[tuple[int, int]] = []
        rects: list[tuple[int, int, int, int]] = []
        for r, c in candidates:
            rect = (r, c, r + BOX_OUTER - 1, c + BOX_OUTER - 1)
            if any(_rects_too_close(rect, other, BOX_PADDING) for other in rects):
                continue
            placed.append((r, c))
            rects.append(rect)
            if len(placed) == count:
                return placed
    return None


def _mark_box_areas(mask: np.ndarray, boxes: list[tuple[int, int]]) -> None:
    for r0, c0 in boxes:
        mask[r0 : r0 + BOX_OUTER, c0 : c0 + BOX_OUTER] = True


def _has_clearance(grid: np.ndarray, bg: int, cells: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if grid[r, c] != bg:
            return False
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cells:
                    continue
                if grid[nr, nc] != bg:
                    return False
    return True


def _fragment_candidates(
    grid: np.ndarray,
    bg: int,
    box_rects: list[tuple[int, int, int, int]],
    target_box: tuple[int, int, int, int],
    direction: str,
    shape: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    sr0, sc0, sr1, sc1 = 0, 0, max(r for r, _ in shape), max(c for _, c in shape)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    br0, bc0, br1, bc1 = target_box
    box_center = ((br0 + br1) / 2.0, (bc0 + bc1) / 2.0)
    candidates: list[tuple[int, int]] = []
    box_area_mask = np.zeros((h, w), dtype=bool)
    for rect in box_rects:
        rr0, cc0, rr1, cc1 = rect
        box_area_mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            frag_rect = (top, left, top + sh - 1, left + sw - 1)
            if direction == "top" and frag_rect[2] >= br0:
                continue
            if direction == "bottom" and frag_rect[0] <= br1:
                continue
            if direction == "left" and frag_rect[3] >= bc0:
                continue
            if direction == "right" and frag_rect[1] <= bc1:
                continue

            world_cells = {(top + r, left + c) for r, c in shape}
            if any(box_area_mask[r, c] for r, c in world_cells):
                continue
            if not _has_clearance(grid, bg, world_cells):
                continue
            coords = np.array(sorted(world_cells), dtype=int)
            if _dominant_direction(coords, box_center) != direction:
                continue
            candidates.append((top, left))
    return candidates


def _paint_shape(grid: np.ndarray, top: int, left: int, shape: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in shape:
        grid[top + r, left + c] = color


def _build_expected(
    grid: np.ndarray,
    boxes: list[tuple[int, int]],
    box_colors: list[int],
    box_shapes: list[dict[str, dict[str, object]]],
    fragment_placements: list[tuple[int, int, tuple[tuple[int, int], ...], int]],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for top, left, shape, _color in fragment_placements:
        for r, c in shape:
            out[top + r, left + c] = bg
    for (r0, c0), color, shape_map in zip(boxes, box_colors, box_shapes):
        _draw_box(out, r0, c0, color)
        for option in shape_map.values():
            for ir, ic in option["placed"]:
                out[r0 + 1 + ir, c0 + 1 + ic] = color
    return out


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    box_count = random.randint(*BOX_COUNT_RANGE)
    h_range, w_range = GRID_RANGES[box_count]
    height = random.randint(*h_range)
    width = random.randint(*w_range)
    bg = random.randrange(10)
    available_colors = [c for c in range(10) if c != bg]
    if len(available_colors) < box_count:
        return None
    box_colors = random.sample(available_colors, box_count)

    boxes = _place_boxes(height, width, box_count)
    if boxes is None:
        return None
    box_rects = [(r, c, r + BOX_OUTER - 1, c + BOX_OUTER - 1) for r, c in boxes]

    box_dirs: list[list[str]] = []
    box_shapes: list[dict[str, dict[str, object]]] = []
    for _ in range(box_count):
        count = random.randint(1, 4)
        directions = random.sample(list(DIRS), count)
        shape_map = _choose_box_shapes(directions)
        if shape_map is None:
            return None
        box_dirs.append(directions)
        box_shapes.append(shape_map)

    grid = np.full((height, width), bg, dtype=int)
    for (r0, c0), color in zip(boxes, box_colors):
        _draw_box(grid, r0, c0, color)

    plans: list[tuple[int, int, str, tuple[tuple[int, int], ...], int]] = []
    for box_index, ((r0, c0), color, directions, shape_map) in enumerate(
        zip(boxes, box_colors, box_dirs, box_shapes)
    ):
        for direction in directions:
            plans.append((box_index, color, direction, shape_map[direction]["shape"], 0))

    random.shuffle(plans)
    fragment_placements: list[tuple[int, int, tuple[tuple[int, int], ...], int]] = []
    for box_index, color, direction, shape, _dummy in plans:
        target_box = box_rects[box_index]
        candidates = _fragment_candidates(grid, bg, box_rects, target_box, direction, shape)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _paint_shape(grid, top, left, shape, color)
        fragment_placements.append((top, left, shape, color))

    expected = _build_expected(grid, boxes, box_colors, box_shapes, fragment_placements, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _sample_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
