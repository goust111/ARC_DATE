from __future__ import annotations

import random
from collections import deque

import numpy as np


YELLOW = 4
GEN_RETRIES = 300
GRID_SIDE_RANGE = (24, 30)
OBJECT_COUNT_CHOICES = (6, 7, 8)
OBJECT_SIZE_RANGE = (3, 5)
SYMMETRY_TYPES = ("none", "lr", "ud", "both")
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_points(points: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _classify_mask(mask: np.ndarray) -> str:
    left_right = np.array_equal(mask, mask[:, ::-1])
    up_down = np.array_equal(mask, mask[::-1, :])
    if left_right and up_down:
        return "both"
    if left_right:
        return "lr"
    if up_down:
        return "ud"
    return "none"


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, pts[0])
    queue = deque([(start_r, start_c)])
    seen[start_r, start_c] = True
    visited = 0
    while queue:
        r, c = queue.popleft()
        visited += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return visited == int(mask.sum())


def _paint_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    assert r0 > 0 and c0 > 0
    assert r1 + 1 < grid.shape[0] and c1 + 1 < grid.shape[1]
    grid[r0 - 1, c0 - 1 : c1 + 2] = color
    grid[r1 + 1, c0 - 1 : c1 + 2] = color
    grid[r0 - 1 : r1 + 2, c0 - 1] = color
    grid[r0 - 1 : r1 + 2, c1 + 1] = color


def _parse_legend_component(grid: np.ndarray, comp: list[tuple[int, int]], bg: int) -> tuple[str, int]:
    colors = {int(grid[r, c]) for r, c in comp if int(grid[r, c]) != bg}
    assert YELLOW in colors
    other = colors - {YELLOW}
    assert len(other) == 1
    border_color = int(next(iter(other)))

    yellow_pts = [(r, c) for r, c in comp if int(grid[r, c]) == YELLOW]
    min_r = min(r for r, _ in yellow_pts)
    min_c = min(c for _, c in yellow_pts)
    shape = frozenset((r - min_r, c - min_c) for r, c in yellow_pts)
    kind_by_shape = {
        frozenset({(0, 0)}): "none",
        frozenset({(0, 0), (0, 1)}): "lr",
        frozenset({(0, 0), (1, 0)}): "ud",
        frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}): "both",
    }
    assert shape in kind_by_shape
    return kind_by_shape[shape], border_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)

    comps = _components(g != bg)
    legend_comps: list[list[tuple[int, int]]] = []
    pattern_comps: list[list[tuple[int, int]]] = []
    for comp in comps:
        has_yellow = any(int(g[r, c]) == YELLOW for r, c in comp)
        touches_bottom = max(r for r, _ in comp) >= h - 1
        if has_yellow and touches_bottom:
            legend_comps.append(comp)
        else:
            pattern_comps.append(comp)

    assert legend_comps
    legend_map: dict[str, int] = {}
    for comp in legend_comps:
        kind, border_color = _parse_legend_component(g, comp, bg)
        legend_map[kind] = border_color

    out = g.copy()
    for comp in pattern_comps:
        colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != bg}
        assert len(colors) == 1
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        kind = _classify_mask(mask)
        assert kind in legend_map
        _paint_border(out, r0, c0, r1, c1, legend_map[kind])

    return out


def _random_path(start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = start
    rr, cc = end
    path = [(r, c)]
    while (r, c) != (rr, cc):
        options: list[tuple[int, int]] = []
        if r < rr:
            options.append((r + 1, c))
        elif r > rr:
            options.append((r - 1, c))
        if c < cc:
            options.append((r, c + 1))
        elif c > cc:
            options.append((r, c - 1))
        r, c = rng.choice(options)
        path.append((r, c))
    return path


def _seed_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    hub = (rng.randrange(h), rng.randrange(w))
    anchors = {
        (0, rng.randrange(w)),
        (h - 1, rng.randrange(w)),
        (rng.randrange(h), 0),
        (rng.randrange(h), w - 1),
    }
    cells = {hub}
    for anchor in anchors:
        cells.update(_random_path(anchor, hub, rng))

    max_target = min(h * w - 1, max(len(cells), int(round(h * w * 0.7))))
    min_target = min(max_target, max(len(cells), int(round(h * w * 0.35))))
    target = rng.randint(min_target, max_target)

    frontier = {
        (nr, nc)
        for r, c in cells
        for nr, nc in _neighbors4(r, c)
        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
    }
    while frontier and len(cells) < target:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) == h * w:
        return np.zeros((h, w), dtype=bool)
    return _mask_from_points(list(cells), h, w)


def _reflect_mask(base: np.ndarray, full_h: int, full_w: int, reflect_lr: bool, reflect_ud: bool) -> np.ndarray:
    out = np.zeros((full_h, full_w), dtype=bool)
    for r, c in np.argwhere(base):
        pts = {(int(r), int(c))}
        if reflect_lr:
            pts.add((int(r), full_w - 1 - int(c)))
        if reflect_ud:
            pts.add((full_h - 1 - int(r), int(c)))
        if reflect_lr and reflect_ud:
            pts.add((full_h - 1 - int(r), full_w - 1 - int(c)))
        for rr, cc in pts:
            out[rr, cc] = True
    return out


def _full_bbox(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    rows = pts[:, 0]
    cols = pts[:, 1]
    return (
        int(rows.min()) == 0
        and int(rows.max()) == mask.shape[0] - 1
        and int(cols.min()) == 0
        and int(cols.max()) == mask.shape[1] - 1
    )


def _generate_shape(kind: str, rng: random.Random) -> np.ndarray:
    for _ in range(400):
        h = rng.randint(OBJECT_SIZE_RANGE[0], OBJECT_SIZE_RANGE[1])
        w = rng.randint(OBJECT_SIZE_RANGE[0], OBJECT_SIZE_RANGE[1])

        if kind == "both":
            base = _seed_mask((h + 1) // 2, (w + 1) // 2, rng)
            mask = _reflect_mask(base, h, w, True, True)
        elif kind == "lr":
            base = _seed_mask(h, (w + 1) // 2, rng)
            mask = _reflect_mask(base, h, w, True, False)
        elif kind == "ud":
            base = _seed_mask((h + 1) // 2, w, rng)
            mask = _reflect_mask(base, h, w, False, True)
        else:
            mask = _seed_mask(h, w, rng)

        if not mask.any():
            continue
        if not _is_connected(mask):
            continue
        if not _full_bbox(mask):
            continue
        if _classify_mask(mask) != kind:
            continue
        return mask
    raise RuntimeError(f"failed to generate shape for {kind}")


def _segment_width(kind: str) -> int:
    return 4 if kind in ("lr", "both") else 3


def _draw_legend_segment(grid: np.ndarray, top: int, left: int, kind: str, color: int) -> None:
    if kind == "none":
        grid[top, left] = YELLOW
        grid[top : top + 2, left + 1 : left + 3] = color
    elif kind == "lr":
        grid[top, left : left + 2] = YELLOW
        grid[top : top + 2, left + 2 : left + 4] = color
    elif kind == "ud":
        grid[top : top + 2, left] = YELLOW
        grid[top : top + 2, left + 1 : left + 3] = color
    else:
        grid[top : top + 2, left : left + 2] = YELLOW
        grid[top : top + 2, left + 2 : left + 4] = color


def _place_shapes(
    grid: np.ndarray,
    shapes: list[dict[str, object]],
    legend_top: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)
    reserved[legend_top - 1 :, :] = True

    order = sorted(range(len(shapes)), key=lambda i: -int(np.prod(np.asarray(shapes[i]["mask"]).shape)))
    for idx in order:
        shape = shapes[idx]
        mask = np.asarray(shape["mask"], dtype=bool)
        sh, sw = mask.shape
        row_lo = 2
        row_hi = legend_top - sh - 3
        col_lo = 2
        col_hi = w - sw - 3
        if row_lo > row_hi or col_lo > col_hi:
            return False

        candidates: list[tuple[int, int]] = []
        for r in range(row_lo, row_hi + 1):
            for c in range(col_lo, col_hi + 1):
                if reserved[r - 2 : r + sh + 2, c - 2 : c + sw + 2].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return False

        r0, c0 = rng.choice(candidates)
        color = int(shape["color"])
        for rr, cc in np.argwhere(mask):
            grid[r0 + int(rr), c0 + int(cc)] = color
        reserved[r0 - 2 : r0 + sh + 2, c0 - 2 : c0 + sw + 2] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1])
    width = rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1])
    legend_top = height - 2
    grid = np.zeros((height, width), dtype=int)

    legend_order = list(SYMMETRY_TYPES)
    rng.shuffle(legend_order)
    legend_colors = rng.sample(PALETTE, len(SYMMETRY_TYPES))
    kind_to_border = dict(zip(SYMMETRY_TYPES, legend_colors))

    gaps = [rng.randint(1, 2) for _ in range(len(legend_order) - 1)]
    legend_width = sum(_segment_width(kind) for kind in legend_order) + sum(gaps)
    if legend_width + 4 > width:
        return None
    start_col = rng.randint(2, width - legend_width - 2)
    col = start_col
    for i, kind in enumerate(legend_order):
        _draw_legend_segment(grid, legend_top, col, kind, kind_to_border[kind])
        col += _segment_width(kind)
        if i + 1 < len(legend_order):
            col += gaps[i]

    object_count = rng.choice(OBJECT_COUNT_CHOICES)
    kinds = list(SYMMETRY_TYPES)
    while len(kinds) < object_count:
        kinds.append(rng.choice(SYMMETRY_TYPES))
    rng.shuffle(kinds)

    shape_palette = [color for color in PALETTE if color not in legend_colors]
    if not shape_palette:
        return None

    shapes: list[dict[str, object]] = []
    for kind in kinds:
        mask = _generate_shape(kind, rng)
        color = rng.choice(shape_palette)
        shapes.append({"kind": kind, "mask": mask, "color": color})

    if not _place_shapes(grid, shapes, legend_top, rng):
        return None

    try:
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
    raise RuntimeError("failed to generate valid input")
