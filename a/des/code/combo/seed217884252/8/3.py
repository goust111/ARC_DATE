from __future__ import annotations

import random

import numpy as np


PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GRID_HEIGHT_RANGE = (14, 22)
GRID_WIDTH_RANGE = (14, 22)
COLOR_COUNT_RANGE = (3, 5)
MARKER_COUNT_POOL = (2, 3, 4, 5, 6)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
KIND_WEIGHTS = (0.55, 0.225, 0.225)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _non_bg_colors(grid: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != bg]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _draw_outline(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _outline_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _outline_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return bool(_outline_cells(a) & _outline_cells(b))


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    )


def _boundary_tags(
    cell: tuple[int, int], bbox: tuple[int, int, int, int]
) -> set[str]:
    r, c = cell
    r0, c0, r1, c1 = bbox
    tags: set[str] = set()
    if r == r0:
        tags.add("top")
    if r == r1:
        tags.add("bottom")
    if c == c0:
        tags.add("left")
    if c == c1:
        tags.add("right")
    return tags


def _sample_bbox(
    kind: str, count: int, height: int, width: int, margin: int, rng: random.Random
) -> tuple[int, int, int, int] | None:
    row_lo = margin
    row_hi = height - 1 - margin
    col_lo = margin
    col_hi = width - 1 - margin
    if row_lo > row_hi or col_lo > col_hi:
        return None

    usable_h = row_hi - row_lo + 1
    usable_w = col_hi - col_lo + 1

    if kind == "row":
        min_w = 2 * count - 1
        if usable_w < min_w:
            return None
        box_w = rng.randint(min_w, min(usable_w, 12))
        row = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi - box_w + 1)
        return (row, c0, row, c0 + box_w - 1)

    if kind == "col":
        min_h = 2 * count - 1
        if usable_h < min_h:
            return None
        box_h = rng.randint(min_h, min(usable_h, 12))
        col = rng.randint(col_lo, col_hi)
        r0 = rng.randint(row_lo, row_hi - box_h + 1)
        return (r0, col, r0 + box_h - 1, col)

    max_h = min(usable_h, 11)
    max_w = min(usable_w, 11)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(100):
        box_h = rng.randint(2, max_h)
        box_w = rng.randint(2, max_w)
        if box_h * box_w < count:
            continue
        if (box_h * box_w + 1) // 2 < count:
            continue
        r0 = rng.randint(row_lo, row_hi - box_h + 1)
        c0 = rng.randint(col_lo, col_hi - box_w + 1)
        return (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
    return None


def _pick_marker_points(
    bbox: tuple[int, int, int, int],
    count: int,
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    candidates = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in occupied
    ]
    if len(candidates) < count:
        return None

    tag_map = {cell: _boundary_tags(cell, bbox) for cell in candidates}
    required = {"top", "bottom", "left", "right"}

    rng.shuffle(candidates)
    candidates.sort(key=lambda cell: -len(tag_map[cell]))
    candidate_set = set(candidates)
    neighbor_map = {
        cell: {
            (cell[0] - 1, cell[1]),
            (cell[0] + 1, cell[1]),
            (cell[0], cell[1] - 1),
            (cell[0], cell[1] + 1),
        }
        & candidate_set
        for cell in candidates
    }

    def dfs(
        start: int,
        chosen: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
        missing: set[str],
    ) -> list[tuple[int, int]] | None:
        need = count - len(chosen)
        if need == 0:
            return chosen if not missing else None

        available = 0
        possible_tags: set[str] = set()
        for idx in range(start, len(candidates)):
            cell = candidates[idx]
            if cell in blocked:
                continue
            available += 1
            possible_tags |= tag_map[cell]
        if available < need or not missing <= possible_tags:
            return None

        for idx in range(start, len(candidates)):
            cell = candidates[idx]
            if cell in blocked:
                continue
            next_blocked = blocked | {cell} | neighbor_map[cell]
            result = dfs(
                idx + 1,
                chosen + [cell],
                next_blocked,
                missing - tag_map[cell],
            )
            if result is not None:
                return result
        return None

    return dfs(0, [], set(), required)


def _place_bboxes(
    counts: list[int],
    kinds: list[str],
    height: int,
    width: int,
    margin: int,
    want_overlap: bool,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    bboxes: list[tuple[int, int, int, int]] = []
    for idx, (count, kind) in enumerate(zip(counts, kinds)):
        for _ in range(PLACEMENT_RETRIES):
            bbox = _sample_bbox(kind, count, height, width, margin, rng)
            if bbox is None or bbox in bboxes:
                continue
            if idx == 0:
                bboxes.append(bbox)
                break
            if want_overlap and idx == 1:
                if any(_outline_overlap(bbox, prev) for prev in bboxes):
                    bboxes.append(bbox)
                    break
                continue
            if any(_boxes_too_close(bbox, prev, pad=1) for prev in bboxes):
                continue
            bboxes.append(bbox)
            break
        else:
            return None
    return bboxes


def _validate_generated(grid: np.ndarray, expect_overlap: bool) -> bool:
    bg = _background_color(grid)
    if bg != 0:
        return False

    colors = _non_bg_colors(grid, bg)
    if not (3 <= len(colors) <= 5):
        return False

    counts: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        count = int(coords.shape[0])
        if not (2 <= count <= 6):
            return False
        points = {tuple(map(int, pt)) for pt in coords}
        for r, c in points:
            if (r - 1, c) in points or (r + 1, c) in points:
                return False
            if (r, c - 1) in points or (r, c + 1) in points:
                return False
        counts.append(count)
        bboxes.append(_bbox(coords))

    if len(set(counts)) != len(counts):
        return False

    overlap_count = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _outline_overlap(bboxes[i], bboxes[j]):
                overlap_count += 1
    if expect_overlap != (overlap_count > 0):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[grid != bg], grid[grid != bg]):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    n_colors = rng.randint(*COLOR_COUNT_RANGE)
    counts = rng.sample(list(MARKER_COUNT_POOL), n_colors)
    colors = rng.sample(list(PALETTE), n_colors)
    want_overlap = rng.random() < 0.5
    margin = 0 if rng.random() < 0.35 else 1

    kinds = [
        rng.choices(("rect", "row", "col"), weights=KIND_WEIGHTS, k=1)[0]
        for _ in range(n_colors)
    ]
    if "rect" not in kinds:
        kinds[rng.randrange(n_colors)] = "rect"

    bboxes = _place_bboxes(counts, kinds, height, width, margin, want_overlap, rng)
    if bboxes is None:
        return None

    occupied: set[tuple[int, int]] = set()
    point_groups: list[list[tuple[int, int]]] = []
    for bbox, count in zip(bboxes, counts):
        points = _pick_marker_points(bbox, count, occupied, rng)
        if points is None:
            return None
        point_groups.append(points)
        occupied.update(points)

    grid = np.zeros((height, width), dtype=int)
    for color, points in zip(colors, point_groups):
        for r, c in points:
            grid[r, c] = color

    if not _validate_generated(grid, expect_overlap=want_overlap):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered single-pixel markers by color, then draw each color's bbox outline.

    For every non-background color:
    - collect all pixels of that color;
    - compute the minimal axis-aligned bounding box;
    - draw the one-pixel-thick rectangular outline in that color.

    Rectangles are drawn in descending order of marker count. If counts tie, the
    earlier marker in row-major order is used as a stable fallback. Original
    marker pixels are restored after all outlines are drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    marker_mask = g != bg
    height, width = map(int, g.shape)

    groups: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for color in _non_bg_colors(g, bg):
        coords = np.argwhere(g == color)
        bbox = _bbox(coords)
        first = min(int(r) * width + int(c) for r, c in coords)
        groups.append((int(coords.shape[0]), first, color, bbox))

    out = g.copy()
    for _count, _first, color, bbox in sorted(groups, key=lambda item: (-item[0], item[1])):
        _draw_outline(out, bbox, color)

    out[marker_mask] = g[marker_mask]
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
