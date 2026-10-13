from __future__ import annotations

import random

import numpy as np


BG = 8
BLACK = 0
GREY = 5
MAROON = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7)
RESERVED_COLORS = {BG, BLACK, GREY, MAROON}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

STYLE_BY_DIRECTION = {
    "horizontal": "solid",
    "vertical": "dashed",
    "diagonal": "double",
}
RING_DEPTH = {"solid": 1, "dashed": 1, "double": 2}

TOP_HEIGHT_RANGE = (14, 18)
GRID_WIDTH_RANGE = (28, 30)
LEGEND_HEIGHT = 8
MIN_OBJECT_MARGIN = 2
GENERATION_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == bg:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _ring_cells(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], dist: int
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    rr0 = r0 - dist
    rr1 = r1 + dist
    cc0 = c0 - dist
    cc1 = c1 + dist
    cells: set[tuple[int, int]] = set()

    for c in range(cc0, cc1 + 1):
        if 0 <= rr0 < h and 0 <= c < w:
            cells.add((rr0, c))
        if 0 <= rr1 < h and 0 <= c < w:
            cells.add((rr1, c))
    for r in range(rr0 + 1, rr1):
        if 0 <= r < h and 0 <= cc0 < w:
            cells.add((r, cc0))
        if 0 <= r < h and 0 <= cc1 < w:
            cells.add((r, cc1))
    return sorted(cells)


def _direction_from_bbox(bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if width > height:
        return "horizontal"
    if height > width:
        return "vertical"
    return "diagonal"


def _paint_border(
    out: np.ndarray,
    blocked: np.ndarray,
    bbox: tuple[int, int, int, int],
    style: str,
) -> None:
    def paint_ring(dist: int, dashed: bool) -> None:
        for r, c in _ring_cells(out.shape, bbox, dist):
            if blocked[r, c]:
                continue
            if dashed:
                out[r, c] = MAROON if (r + c) % 2 == 0 else BLACK
            else:
                out[r, c] = MAROON

    if style == "solid":
        paint_ring(1, dashed=False)
    elif style == "dashed":
        paint_ring(1, dashed=True)
    else:
        paint_ring(2, dashed=False)
        paint_ring(1, dashed=False)


def _legend_mapping(legend: np.ndarray) -> dict[str, str]:
    is_separator = np.all(legend == GREY, axis=0)
    patches: list[np.ndarray] = []
    c = 0
    while c < legend.shape[1]:
        while c < legend.shape[1] and is_separator[c]:
            c += 1
        if c >= legend.shape[1]:
            break
        c0 = c
        while c < legend.shape[1] and not is_separator[c]:
            c += 1
        patch = legend[:, c0:c]
        if np.any(patch != BG):
            patches.append(patch)

    mapping: dict[str, str] = {}
    for patch in patches:
        colors = [int(v) for v in np.unique(patch) if int(v) not in RESERVED_COLORS]
        assert colors
        sample_color = max(colors, key=lambda color: int(np.count_nonzero(patch == color)))
        coords = np.argwhere(patch == sample_color)
        bbox = _bbox_from_coords(coords)
        direction = _direction_from_bbox(bbox)

        ring1 = _ring_cells(patch.shape, bbox, 1)
        ring2 = _ring_cells(patch.shape, bbox, 2)
        ring1_vals = [int(patch[r, c]) for r, c in ring1]
        ring2_vals = [int(patch[r, c]) for r, c in ring2]

        if any(v == MAROON for v in ring2_vals):
            style = "double"
        elif any(v == BLACK for v in ring1_vals):
            style = "dashed"
        else:
            style = "solid"
        mapping[direction] = style

    assert set(mapping) == {"horizontal", "vertical", "diagonal"}
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    separator_rows = np.where(np.all(g == GREY, axis=1))[0]
    assert len(separator_rows) > 0
    sep = int(separator_rows[-1])

    top = g[:sep, :].copy()
    legend = g[sep + 1 :, :]
    mapping = _legend_mapping(legend)

    out = top.copy()
    blocked = top != BG
    comps = _same_color_components(top, BG)
    comps.sort(key=lambda comp: (int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))

    for comp in comps:
        bbox = comp["bbox"]
        direction = _direction_from_bbox(bbox)
        _paint_border(out, blocked, bbox, mapping[direction])

    return out


def _coords_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    norm = {(r - r0, c - c0) for r, c in cells}
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_flip(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = out[:, ::-1]
    if random.random() < 0.5:
        out = out[::-1, :]
    return out.copy()


def _horizontal_mask() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(max(5, h + 2), 8)
    row = random.randint(0, h - 1)
    cells: set[tuple[int, int]] = {(row, 0)}
    rows_by_col = [row]

    for c in range(1, w):
        new_row = min(h - 1, max(0, row + random.choice((-1, 0, 1))))
        if new_row != row:
            cells.add((row, c))
        row = new_row
        cells.add((row, c))
        rows_by_col.append(row)

    top_col = random.randrange(w)
    bottom_col = random.randrange(w)
    for rr in range(min(rows_by_col[top_col], 0), max(rows_by_col[top_col], 0) + 1):
        cells.add((rr, top_col))
    for rr in range(min(rows_by_col[bottom_col], h - 1), max(rows_by_col[bottom_col], h - 1) + 1):
        cells.add((rr, bottom_col))

    frontier: list[tuple[int, int]] = []
    for r, c in list(cells):
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.append((nr, nc))
    random.shuffle(frontier)
    for nr, nc in frontier[: random.randint(0, max(1, w // 3))]:
        cells.add((nr, nc))

    return _random_flip(_coords_to_mask(cells))


def _vertical_mask() -> np.ndarray:
    return _horizontal_mask().T.copy()


def _diagonal_mask() -> np.ndarray:
    n = random.randint(4, 6)
    r = 0
    c = 0
    cells = [(r, c)]
    rights = n - 1
    downs = n - 1

    while rights or downs:
        if rights == 0:
            r += 1
            downs -= 1
        elif downs == 0:
            c += 1
            rights -= 1
        elif r < c:
            r += 1
            downs -= 1
        elif c < r:
            c += 1
            rights -= 1
        else:
            if random.random() < 0.5:
                c += 1
                rights -= 1
            else:
                r += 1
                downs -= 1
        cells.append((r, c))

    occupied = set(cells)
    path = cells[1:-1]
    random.shuffle(path)
    extra_budget = random.randint(1, max(1, n // 2))
    for pr, pc in path[:extra_budget]:
        choices = []
        for dr, dc in NEIGHBORS4:
            nr = pr + dr
            nc = pc + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in occupied:
                choices.append((nr, nc))
        if choices:
            occupied.add(random.choice(choices))

    if random.random() < 0.5:
        occupied = {(rr, n - 1 - cc) for rr, cc in occupied}

    return _random_flip(_coords_to_mask(occupied))


def _make_object_mask(direction: str, used_areas: set[int]) -> np.ndarray:
    for _ in range(80):
        if direction == "horizontal":
            mask = _horizontal_mask()
        elif direction == "vertical":
            mask = _vertical_mask()
        else:
            mask = _diagonal_mask()
        area = int(mask.sum())
        if area not in used_areas:
            return mask
    raise RuntimeError("failed to sample unique-area object mask")


def _expand_bbox(bbox: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _ring_area(shape: tuple[int, int], bbox: tuple[int, int, int, int], style: str) -> set[tuple[int, int]]:
    cells = set(_ring_cells(shape, bbox, 1))
    if style == "double":
        cells.update(_ring_cells(shape, bbox, 2))
    return cells


def _place_masks(
    top_h: int,
    top_w: int,
    masks: list[np.ndarray],
    styles: list[str],
    require_overlap: bool,
) -> list[tuple[int, int]] | None:
    for _ in range(200):
        placements: list[tuple[int, int]] = []
        bboxes: list[tuple[int, int, int, int]] = []

        ok = True
        for mask in masks:
            h, w = mask.shape
            candidates = []
            for _ in range(120):
                r0 = random.randint(MIN_OBJECT_MARGIN, top_h - MIN_OBJECT_MARGIN - h)
                c0 = random.randint(MIN_OBJECT_MARGIN, top_w - MIN_OBJECT_MARGIN - w)
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(_bboxes_overlap(_expand_bbox(bbox, 2), other) or _bboxes_overlap(bbox, _expand_bbox(other, 2)) for other in bboxes):
                    continue
                candidates.append((r0, c0, bbox))
            if not candidates:
                ok = False
                break
            r0, c0, bbox = random.choice(candidates)
            placements.append((r0, c0))
            bboxes.append(bbox)

        if not ok:
            continue

        overlap = False
        for i in range(len(bboxes)):
            ring_i = _ring_area((top_h, top_w), bboxes[i], styles[i])
            for j in range(i + 1, len(bboxes)):
                ring_j = _ring_area((top_h, top_w), bboxes[j], styles[j])
                if ring_i & ring_j:
                    overlap = True
                    break
            if overlap:
                break
        if overlap == require_overlap:
            return placements
    return None


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    patch = canvas[top : top + h, left : left + w]
    patch[mask] = int(color)


def _legend_sample_mask(direction: str) -> np.ndarray:
    if direction == "horizontal":
        return np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)
    if direction == "vertical":
        return np.array([[0, 1], [1, 1], [1, 1], [0, 1]], dtype=bool)
    mask = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    if random.random() < 0.5:
        mask = mask[:, ::-1]
    return mask


def _build_legend(width: int) -> np.ndarray:
    extra = width - 28
    slot_widths = [8, 8, 8]
    for _ in range(extra):
        slot_widths[random.randrange(3)] += 1

    directions = ["horizontal", "vertical", "diagonal"]
    random.shuffle(directions)
    sample_colors = random.sample(OBJECT_COLORS, 3)

    legend = np.full((LEGEND_HEIGHT, width), GREY, dtype=int)
    cursor = 1
    for slot_w, direction, color in zip(slot_widths, directions, sample_colors, strict=False):
        legend[:, cursor : cursor + slot_w] = BG
        slot = np.full((LEGEND_HEIGHT, slot_w), BG, dtype=int)
        local = np.full((LEGEND_HEIGHT, 8), BG, dtype=int)
        mask = _legend_sample_mask(direction)
        if direction == "horizontal":
            top = 3
            left = 2
        elif direction == "vertical":
            top = 2
            left = 3
        else:
            top = 2
            left = 2
        _paint_mask(local, mask, color, top, left)
        bbox = (top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1)
        blocked = local != BG
        _paint_border(local, blocked, bbox, STYLE_BY_DIRECTION[direction])
        offset = random.randint(0, slot_w - 8)
        slot[:, offset : offset + 8] = local
        legend[:, cursor : cursor + slot_w] = slot
        cursor += slot_w + 1
    return legend


def _build_top_canvas(
    top_h: int,
    top_w: int,
    directions: list[str],
    colors: list[int],
    require_overlap: bool,
) -> np.ndarray | None:
    used_areas: set[int] = set()
    masks = []
    for direction in directions:
        mask = _make_object_mask(direction, used_areas)
        used_areas.add(int(mask.sum()))
        masks.append(mask)

    styles = [STYLE_BY_DIRECTION[direction] for direction in directions]
    placements = _place_masks(top_h, top_w, masks, styles, require_overlap)
    if placements is None:
        return None

    top = np.full((top_h, top_w), BG, dtype=int)
    for mask, color, (r0, c0) in zip(masks, colors, placements, strict=False):
        _paint_mask(top, mask, color, r0, c0)
    return top


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        top_h = random.randint(*TOP_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        object_count = random.randint(3, 4)
        directions = ["horizontal", "vertical", "diagonal"]
        while len(directions) < object_count:
            directions.append(random.choice(["horizontal", "vertical", "diagonal"]))
        random.shuffle(directions)

        colors = random.sample(OBJECT_COLORS, object_count)
        require_overlap = random.random() < 0.5
        top = _build_top_canvas(top_h, width, directions, colors, require_overlap)
        if top is None:
            continue

        legend = _build_legend(width)
        separator = np.full((1, width), GREY, dtype=int)
        grid = np.vstack([top, separator, legend])

        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != top.shape:
            continue
        if np.array_equal(out, top):
            continue
        return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
