from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_solid_rectangle(grid: np.ndarray, comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    area = int(comp["area"])
    return area == (r1 - r0 + 1) * (c1 - c0 + 1) and bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _shift_for_component(comp: dict[str, object], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    rect_h = int(r1 - r0 + 1)
    rect_w = int(c1 - c0 + 1)

    touches_top = r0 == 0
    touches_bottom = r1 == h - 1
    touches_left = c0 == 0
    touches_right = c1 == w - 1

    assert touches_top != touches_bottom, "rectangle must touch exactly one horizontal border"
    assert touches_left != touches_right, "rectangle must touch exactly one vertical border"

    dr = rect_h if touches_top else -rect_h
    dc = rect_w if touches_left else -rect_w
    return dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the repeated boundary-color rectangles one rectangle-size inward and keep singleton colors fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components_by_color(g, color):
            assert _is_solid_rectangle(g, comp), "each foreground object must be a solid rectangle"
            by_color[color].append(comp)

    moving_colors = [color for color, comps in by_color.items() if len(comps) > 1]
    assert len(moving_colors) == 1, "expected exactly one repeated foreground color"
    moving_color = int(moving_colors[0])

    out = g.copy()
    for comp in by_color[moving_color]:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for comp in by_color[moving_color]:
        coords = comp["coords"]
        dr, dc = _shift_for_component(comp, (h, w))
        shifted = coords + np.array([dr, dc], dtype=int)
        assert bool(np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < h) & (0 <= shifted[:, 1]) & (shifted[:, 1] < w)))
        target = out[shifted[:, 0], shifted[:, 1]]
        assert bool(np.all((target == bg) | (target == moving_color))), "shifted rectangles may only overlap same color"
        out[shifted[:, 0], shifted[:, 1]] = moving_color

    return out


CORNERS = ("TL", "TR", "BL", "BR")
PALETTE = tuple(range(10))
GRID_H_RANGE = (8, 22)
GRID_W_RANGE = (8, 22)
RECT_H_RANGE = (1, 5)
RECT_W_RANGE = (1, 5)
MOVING_COUNT_CHOICES = (2, 3, 4)
MAX_SINGLETONS = 2
GEN_ATTEMPTS = 400


def _corner_bbox(corner: str, grid_h: int, grid_w: int, rect_h: int, rect_w: int) -> tuple[int, int, int, int]:
    r0 = 0 if corner[0] == "T" else grid_h - rect_h
    c0 = 0 if corner[1] == "L" else grid_w - rect_w
    return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)


def _shifted_bbox(corner: str, grid_h: int, grid_w: int, rect_h: int, rect_w: int) -> tuple[int, int, int, int]:
    r0 = rect_h if corner[0] == "T" else grid_h - 2 * rect_h
    c0 = rect_w if corner[1] == "L" else grid_w - 2 * rect_w
    return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _sample_dimensions(grid_h: int, grid_w: int, compact: bool) -> dict[str, tuple[int, int]]:
    dims: dict[str, tuple[int, int]] = {}
    for corner in CORNERS:
        max_h = min(RECT_H_RANGE[1], (grid_h - 1) // 2)
        max_w = min(RECT_W_RANGE[1], (grid_w - 1) // 2)
        if compact:
            min_h = min(max_h, max(1, min(2, max_h)))
            min_w = min(max_w, max(1, min(2, max_w)))
        else:
            min_h = RECT_H_RANGE[0]
            min_w = RECT_W_RANGE[0]
        rect_h = random.randint(min_h, max_h)
        rect_w = random.randint(min_w, max_w)
        dims[corner] = (rect_h, rect_w)
    return dims


def _valid_input_layout(grid_h: int, grid_w: int, dims: dict[str, tuple[int, int]]) -> bool:
    boxes = {corner: _corner_bbox(corner, grid_h, grid_w, *dims[corner]) for corner in CORNERS}
    for i, a in enumerate(CORNERS):
        for b in CORNERS[i + 1 :]:
            if _boxes_overlap(boxes[a], boxes[b]):
                return False
    return True


def _can_keep_singletons_clear(
    grid_h: int,
    grid_w: int,
    dims: dict[str, tuple[int, int]],
    moving: set[str],
    singletons: list[str],
) -> bool:
    moved_boxes = [_shifted_bbox(corner, grid_h, grid_w, *dims[corner]) for corner in moving]
    for corner in singletons:
        keep_box = _corner_bbox(corner, grid_h, grid_w, *dims[corner])
        for moved_box in moved_boxes:
            if _boxes_overlap(keep_box, moved_box):
                return False
    return True


def _has_shift_overlap(grid_h: int, grid_w: int, dims: dict[str, tuple[int, int]], moving: set[str]) -> bool:
    boxes = [_shifted_bbox(corner, grid_h, grid_w, *dims[corner]) for corner in moving]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i], boxes[j]):
                return True
    return False


def _foreground_area(dims: dict[str, tuple[int, int]]) -> int:
    return sum(rect_h * rect_w for rect_h, rect_w in dims.values())


def _build_grid(
    grid_h: int,
    grid_w: int,
    bg: int,
    moving_color: int,
    moving: set[str],
    singletons: list[str],
    singleton_colors: list[int],
    dims: dict[str, tuple[int, int]],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for corner in moving:
        r0, c0, r1, c1 = _corner_bbox(corner, grid_h, grid_w, *dims[corner])
        grid[r0 : r1 + 1, c0 : c1 + 1] = moving_color
    for corner, color in zip(singletons, singleton_colors):
        r0, c0, r1, c1 = _corner_bbox(corner, grid_h, grid_w, *dims[corner])
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _sample_layout() -> tuple[int, int, dict[str, tuple[int, int]], set[str], list[str]] | None:
    want_overlap = random.random() < 0.5
    moving_count = random.choice(MOVING_COUNT_CHOICES)

    for _ in range(120):
        if want_overlap:
            grid_h = random.randint(8, 13)
            grid_w = random.randint(8, 13)
        else:
            grid_h = random.randint(*GRID_H_RANGE)
            grid_w = random.randint(*GRID_W_RANGE)

        compact = want_overlap and random.random() < 0.8
        dims = _sample_dimensions(grid_h, grid_w, compact=compact)
        if not _valid_input_layout(grid_h, grid_w, dims):
            continue

        moving = set(random.sample(CORNERS, moving_count))
        singletons = [corner for corner in CORNERS if corner not in moving]
        if len(singletons) > MAX_SINGLETONS:
            random.shuffle(singletons)
            singletons = singletons[:MAX_SINGLETONS]
        if not singletons:
            if moving_count == 4:
                singletons = []
            else:
                continue

        if not _can_keep_singletons_clear(grid_h, grid_w, dims, moving, singletons):
            continue

        if _foreground_area({k: dims[k] for k in moving.union(singletons)}) * 2 >= grid_h * grid_w:
            continue

        has_overlap = _has_shift_overlap(grid_h, grid_w, dims, moving)
        if want_overlap != has_overlap:
            continue
        return grid_h, grid_w, dims, moving, singletons
    return None


def _sample_colors(singleton_count: int) -> tuple[int, int, list[int]]:
    bg = random.choice(PALETTE)
    rest = [c for c in PALETTE if c != bg]
    moving_color = random.choice(rest)
    rest.remove(moving_color)
    singleton_colors = random.sample(rest, singleton_count)
    return bg, moving_color, singleton_colors


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    grid_h, grid_w, dims, moving, singletons = layout
    bg, moving_color, singleton_colors = _sample_colors(len(singletons))
    grid = _build_grid(grid_h, grid_w, bg, moving_color, moving, singletons, singleton_colors, dims)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None

    by_color = Counter(map(int, grid.flatten()))
    if by_color[bg] <= max(v for k, v in by_color.items() if k != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
