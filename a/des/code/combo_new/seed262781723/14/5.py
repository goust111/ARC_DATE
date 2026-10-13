from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
MARKER = 2
DIVIDER = 5
BODY_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_HEIGHTS = tuple(range(17, 30, 2))
GRID_WIDTH_RANGE = (16, 30)
OBJECTS_PER_SIDE = (1, 2)
MIN_DIVIDER_GAP = 1
EDGE_BORDER = 1
PLACEMENT_PAD = 1
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _label_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _rotate_shape(
    coords: set[tuple[int, int]],
    marker: tuple[int, int],
    height: int,
    width: int,
    turns: int,
    flip_lr: bool,
) -> tuple[set[tuple[int, int]], tuple[int, int], int, int]:
    pts = set(coords)
    mr, mc = marker
    h, w = height, width
    for _ in range(turns % 4):
        pts = {(c, h - 1 - r) for r, c in pts}
        mr, mc = mc, h - 1 - mr
        h, w = w, h
    if flip_lr:
        pts = {(r, w - 1 - c) for r, c in pts}
        mc = w - 1 - mc
    return pts, (mr, mc), h, w


def _shape_marker_candidates(coords: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in coords:
        if not (0 < r < height - 1 and 0 < c < width - 1):
            continue
        degree = sum((r + dr, c + dc) in coords for dr, dc in DIRS4)
        if degree >= 2:
            candidates.append((r, c))
    return candidates


def _random_shape(rng: np.random.Generator) -> dict[str, object] | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = int(rng.integers(3, 6))
        width = int(rng.integers(3, 6))
        min_cells = max(5, height + width - 2)
        max_cells = min(height * width - 1, 9)
        if min_cells > max_cells:
            continue

        start = (int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1)))
        target = int(rng.integers(min_cells, max_cells + 1))
        cells = {start}
        frontier = set()

        while len(cells) < target:
            frontier.clear()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            idx = int(rng.integers(len(frontier)))
            cells.add(tuple(list(frontier)[idx]))

        if len(cells) != target:
            continue

        norm = _normalize(cells)
        coords = set(norm)
        real_h = max(r for r, _ in coords) + 1
        real_w = max(c for _, c in coords) + 1
        if real_h < 3 or real_w < 3:
            continue

        candidates = _shape_marker_candidates(coords, real_h, real_w)
        if not candidates:
            continue

        marker = candidates[int(rng.integers(len(candidates)))]
        turns = int(rng.integers(0, 4))
        flip_lr = bool(rng.integers(0, 2))
        coords, marker, real_h, real_w = _rotate_shape(coords, marker, real_h, real_w, turns, flip_lr)

        orig = _normalize(coords)
        mr, mc = marker
        mirror = _normalize({(r, 2 * mc - c) for r, c in coords})
        rotate = _normalize({(2 * mr - r, 2 * mc - c) for r, c in coords})
        if mirror == orig or rotate == orig:
            continue

        return {
            "coords": frozenset(coords),
            "marker": marker,
            "height": real_h,
            "width": real_w,
        }
    return None


def _extract_objects(
    grid: np.ndarray,
    divider_row: int,
    divider_color: int,
    bg: int,
) -> list[dict[str, object]]:
    mask = (grid != bg) & (grid != divider_color)
    comps = _label_components(mask)
    objects: list[dict[str, object]] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        assert max(rows) < divider_row or min(rows) > divider_row, "object may not cross the divider"

        vals = [int(grid[r, c]) for r, c in comp]
        counts = Counter(vals)
        assert len(counts) == 2, "each object must contain one body color and one marker color"
        marker_colors = [color for color, count in counts.items() if count == 1]
        assert len(marker_colors) == 1, "marker color must appear exactly once per object"
        marker_color = marker_colors[0]
        body = [color for color, count in counts.items() if color != marker_color and count > 1]
        assert len(body) == 1, "body color must be unique within each object"

        marker_pos = next((r, c) for r, c in comp if int(grid[r, c]) == marker_color)
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        objects.append(
            {
                "coords": tuple(sorted(comp)),
                "bbox": (r0, c0, r1, c1),
                "marker": marker_pos,
                "side": "top" if r1 < divider_row else "bottom",
            }
        )
    objects.sort(key=lambda obj: (obj["side"], obj["bbox"][0], obj["bbox"][1]))
    return objects


def _copy_pixels(obj: dict[str, object], grid: np.ndarray, divider_row: int) -> list[tuple[int, int, int]]:
    coords: tuple[tuple[int, int], ...] = obj["coords"]  # type: ignore[assignment]
    r0, _c0, r1, _c1 = obj["bbox"]  # type: ignore[misc]
    _mr, mc = obj["marker"]  # type: ignore[misc]
    side = obj["side"]

    copied: list[tuple[int, int, int]] = []
    if side == "top":
        row_shift = 2 * divider_row - (r0 + r1)
        for r, c in coords:
            copied.append((r + row_shift, 2 * mc - c, int(grid[r, c])))
    else:
        for r, c in coords:
            copied.append((2 * divider_row - r, 2 * mc - c, int(grid[r, c])))
    return copied


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each object across the central horizontal divider using a side-dependent symmetry rule.

    Input structure:
    - The background is black, and the exact middle row is a full-width non-black divider line.
    - Above and below the divider sit several separated 4-connected foreground objects.
    - Each object has one body color plus exactly one different marker pixel inside its bounding box.

    Rule:
    - Keep the divider and every original object unchanged.
    - For each object above the divider, mirror the object left-right across the marker pixel's column, then place that mirrored copy below the divider while preserving the object's gap to the divider.
    - For each object below the divider, rotate the object 180 degrees around its marker pixel, then place that rotated copy above the divider while preserving the object's gap to the divider.
    - Copy the marker pixel together with the rest of the object; copied pixels overwrite anything already present.

    Key insight:
    - The divider does two jobs at once: it tells which side receives the copy and also selects the transformation type, while the single marker pixel supplies the mirror axis or rotation pivot for that object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape

    bg = _get_bg(g)
    divider_row = h // 2
    divider_values = set(map(int, g[divider_row, :]))
    assert len(divider_values) == 1, "divider row must be monochrome"
    divider_color = int(g[divider_row, 0])
    assert divider_color != bg, "divider row must differ from background"

    objects = _extract_objects(g, divider_row, divider_color, bg)
    assert objects, "at least one object required"

    out = g.copy()
    for obj in objects:
        for rr, cc, value in _copy_pixels(obj, g, divider_row):
            assert 0 <= rr < h and 0 <= cc < w, "copy must stay inside the grid"
            out[rr, cc] = value
    return out


def _copy_coords(
    spec: dict[str, object],
    base_r: int,
    base_c: int,
    divider_row: int,
    side: str,
) -> list[tuple[int, int]]:
    coords: frozenset[tuple[int, int]] = spec["coords"]  # type: ignore[assignment]
    _mr, mc = spec["marker"]  # type: ignore[misc]
    height = int(spec["height"])
    if side == "top":
        row_shift = 2 * divider_row - (2 * base_r + height - 1)
        return [(base_r + r + row_shift, base_c + 2 * mc - c) for r, c in coords]
    return [(2 * divider_row - (base_r + r), base_c + 2 * mc - c) for r, c in coords]


def _reserve_box(
    reserved: np.ndarray,
    coords: list[tuple[int, int]],
    pad: int,
) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = _bbox(coords)
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(h - 1, r1 + pad)
    c1 = min(w - 1, c1 + pad)
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _box_conflicts(
    reserved: np.ndarray,
    coords: list[tuple[int, int]],
    pad: int,
) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = _bbox(coords)
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(h - 1, r1 + pad)
    c1 = min(w - 1, c1 + pad)
    return bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any())


def _paint_object(
    grid: np.ndarray,
    spec: dict[str, object],
    base_r: int,
    base_c: int,
    color: int,
) -> None:
    coords: frozenset[tuple[int, int]] = spec["coords"]  # type: ignore[assignment]
    mr, mc = spec["marker"]  # type: ignore[misc]
    for r, c in coords:
        grid[base_r + r, base_c + c] = color
    grid[base_r + mr, base_c + mc] = MARKER


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    spec: dict[str, object],
    side: str,
    color: int,
    divider_row: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    height = int(spec["height"])
    width = int(spec["width"])
    _mr, mc = spec["marker"]  # type: ignore[misc]

    col_lo = max(EDGE_BORDER, EDGE_BORDER + (width - 1) - 2 * mc)
    col_hi = min(w - EDGE_BORDER - width, w - EDGE_BORDER - 1 - 2 * mc)
    if side == "top":
        row_lo = EDGE_BORDER
        row_hi = divider_row - MIN_DIVIDER_GAP - height
    else:
        row_lo = divider_row + 1 + MIN_DIVIDER_GAP
        row_hi = h - EDGE_BORDER - height

    if row_lo > row_hi or col_lo > col_hi:
        return False

    candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    rng.shuffle(candidates)
    coords: frozenset[tuple[int, int]] = spec["coords"]  # type: ignore[assignment]

    for base_r, base_c in candidates[:PLACEMENT_ATTEMPTS]:
        src = [(base_r + r, base_c + c) for r, c in coords]
        dst = _copy_coords(spec, base_r, base_c, divider_row, side)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in src + dst):
            continue
        if any(r == divider_row for r, _ in src + dst):
            continue
        if _box_conflicts(reserved, src, PLACEMENT_PAD) or _box_conflicts(reserved, dst, PLACEMENT_PAD):
            continue

        _paint_object(grid, spec, base_r, base_c, color)
        _reserve_box(reserved, src, PLACEMENT_PAD)
        _reserve_box(reserved, dst, PLACEMENT_PAD)
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.choice(GRID_HEIGHTS))
    width = int(rng.integers(*GRID_WIDTH_RANGE))
    divider_row = height // 2

    grid = np.full((height, width), BG, dtype=int)
    grid[divider_row, :] = DIVIDER
    reserved = np.zeros((height, width), dtype=bool)

    per_side = int(rng.integers(OBJECTS_PER_SIDE[0], OBJECTS_PER_SIDE[1] + 1))
    chosen_colors = list(map(int, rng.choice(BODY_COLORS, size=2 * per_side, replace=False)))
    jobs = [("top", chosen_colors[i]) for i in range(per_side)] + [
        ("bottom", chosen_colors[per_side + i]) for i in range(per_side)
    ]
    rng.shuffle(jobs)

    for side, color in jobs:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            spec = _random_shape(rng)
            if spec is None:
                continue
            if _place_object(grid, reserved, spec, side, color, divider_row, rng):
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out[:divider_row, :] != grid[:divider_row, :]):
        return None
    if not np.any(out[divider_row + 1 :, :] != grid[divider_row + 1 :, :]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
