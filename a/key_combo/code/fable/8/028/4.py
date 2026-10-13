from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
HOLE_POOL = [0, 1, 2, 3, 4, 5]
OBJECT_COUNT_RANGE = (3, 5)
MAX_OUTPUT_HEIGHT = 30
MIN_GAP = 2
LAYOUT_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components_mask(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _count_holes(crop: np.ndarray, color: int) -> int:
    mask = crop != int(color)
    holes = 0
    h, w = crop.shape
    for coords in _components_mask(mask):
        touches_border = np.any(
            (coords[:, 0] == 0) | (coords[:, 0] == h - 1) | (coords[:, 1] == 0) | (coords[:, 1] == w - 1)
        )
        if not touches_border:
            holes += 1
    return holes


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    bg = int(Counter(map(int, grid.ravel())).most_common(1)[0][0])
    objects = _find_components(grid, bg)
    enriched = []
    for obj in objects:
        crop = _crop_bbox(grid, obj["bbox"])
        holes = _count_holes(crop, int(obj["color"]))
        enriched.append(
            {
                "color": int(obj["color"]),
                "bbox": obj["bbox"],
                "coords": obj["coords"],
                "area": int(obj["area"]),
                "holes": int(holes),
            }
        )
    return enriched


def _render_output(objects: list[dict]) -> np.ndarray:
    ordered = sorted(objects, key=lambda obj: obj["holes"])
    height = max(int(obj["area"]) for obj in ordered)
    out = np.zeros((height, len(ordered)), dtype=int)
    for c, obj in enumerate(ordered):
        area = int(obj["area"])
        out[height - area :, c] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored object into a histogram bar whose order is set by hole count and whose height is set by solid area.

    Input structure:
    - The background is black and the foreground consists of several separated monochrome objects.
    - Every object has a distinct color and may contain fully enclosed black holes.
    - The objects' hole counts and solid pixel counts are both pairwise distinct.

    Rule:
    - For each object, count its enclosed black holes and count its own foreground pixels.
    - Sort the objects from left to right by increasing hole count.
    - Output a black grid with one vertical bar per object, using the object's color and a bar height equal to its solid pixel count, with all bars bottom-aligned.

    Key insight:
    - Each object carries two independent signals: holes determine ordering, while solid area determines bar height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _analyze_objects(g)
    assert objects

    holes = [int(obj["holes"]) for obj in objects]
    areas = [int(obj["area"]) for obj in objects]
    colors = [int(obj["color"]) for obj in objects]
    assert len(set(holes)) == len(holes)
    assert len(set(areas)) == len(areas)
    assert len(set(colors)) == len(colors)

    out = _render_output(objects)
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def _checkerboard_capacity(inner_h: int, inner_w: int) -> int:
    return sum(1 for r in range(inner_h) for c in range(inner_w) if (r + c) % 2 == 0)


def _sample_hole_positions(outer_h: int, outer_w: int, hole_count: int) -> list[Coord]:
    if hole_count == 0:
        return []
    options = []
    for parity in (0, 1):
        cells = [
            (r, c)
            for r in range(1, outer_h - 1)
            for c in range(1, outer_w - 1)
            if (r + c) % 2 == parity
        ]
        if len(cells) >= hole_count:
            options.append(cells)
    if not options:
        raise ValueError("not enough interior cells for isolated holes")
    cells = random.choice(options)
    return random.sample(cells, hole_count)


def _grow_appendage(base_mask: np.ndarray, extra_cells: int) -> np.ndarray | None:
    if extra_cells == 0:
        return base_mask.copy()

    h, w = base_mask.shape
    pad = extra_cells + 2
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    r0 = pad
    c0 = pad
    canvas[r0 : r0 + h, c0 : c0 + w] = base_mask
    rect = (r0, c0, r0 + h - 1, c0 + w - 1)

    for _ in range(extra_cells):
        frontier: list[Coord] = []
        for rr, cc in np.argwhere(canvas):
            for nr, nc in _neighbors4(int(rr), int(cc)):
                if not (0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]):
                    continue
                if canvas[nr, nc]:
                    continue
                if rect[0] <= nr <= rect[2] and rect[1] <= nc <= rect[3]:
                    continue
                occupied_neighbors = 0
                for ar, ac in _neighbors4(nr, nc):
                    if 0 <= ar < canvas.shape[0] and 0 <= ac < canvas.shape[1] and canvas[ar, ac]:
                        occupied_neighbors += 1
                if occupied_neighbors == 1:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        canvas[nr, nc] = True

    coords = np.argwhere(canvas)
    rr0, cc0, rr1, cc1 = _bbox_coords(coords)
    return canvas[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _make_object_mask(hole_count: int) -> np.ndarray | None:
    candidates = [
        (outer_h, outer_w)
        for outer_h in range(3, 7)
        for outer_w in range(3, 7)
        if _checkerboard_capacity(outer_h - 2, outer_w - 2) >= hole_count
        and outer_h * outer_w - hole_count <= MAX_OUTPUT_HEIGHT
    ]
    if not candidates:
        return None

    outer_h, outer_w = random.choice(candidates)
    base = np.ones((outer_h, outer_w), dtype=bool)
    for r, c in _sample_hole_positions(outer_h, outer_w, hole_count):
        base[r, c] = False

    base_area = int(base.sum())
    max_extra = min(5, MAX_OUTPUT_HEIGHT - base_area)
    extra_cells = random.randint(0, max_extra)
    grown = _grow_appendage(base, extra_cells)
    if grown is None:
        return None
    return grown.astype(bool)


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _row_layout_width(row: list[dict], gaps: list[int]) -> int:
    return sum(int(obj["patch"].shape[1]) for obj in row) + sum(gaps)


def _place_objects(objects: list[dict]) -> np.ndarray | None:
    for _ in range(LAYOUT_ATTEMPTS):
        shuffled = objects[:]
        random.shuffle(shuffled)
        row_count = 1 if random.random() < 0.4 else 2
        if row_count == 1:
            rows = [shuffled]
        else:
            split = random.randint(1, len(shuffled) - 1)
            rows = [shuffled[:split], shuffled[split:]]

        row_heights = [max(int(obj["patch"].shape[0]) for obj in row) for row in rows]
        row_gaps = [random.randint(MIN_GAP, MIN_GAP + 2) for _ in range(len(rows) - 1)]
        margin_top = random.randint(1, 3)
        margin_bottom = random.randint(1, 3)
        margin_left = random.randint(1, 3)
        margin_right = random.randint(1, 3)

        row_widths = []
        row_internal_gaps: list[list[int]] = []
        for row in rows:
            gaps = [random.randint(MIN_GAP, MIN_GAP + 2) for _ in range(len(row) - 1)]
            row_internal_gaps.append(gaps)
            row_widths.append(_row_layout_width(row, gaps))

        grid_h = margin_top + margin_bottom + sum(row_heights) + sum(row_gaps)
        grid_w = margin_left + margin_right + max(row_widths)
        if grid_h > 30 or grid_w > 30:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        r_cursor = margin_top
        for row_idx, (row, row_h, gaps, row_w) in enumerate(zip(rows, row_heights, row_internal_gaps, row_widths)):
            x_slack = grid_w - margin_left - margin_right - row_w
            c_cursor = margin_left + random.randint(0, x_slack)
            for i, obj in enumerate(row):
                patch = obj["patch"]
                ph, pw = patch.shape
                top = r_cursor + random.randint(0, row_h - ph)
                region = grid[top : top + ph, c_cursor : c_cursor + pw]
                if np.any((region != BG) & (patch != BG)):
                    break
                region[patch != BG] = patch[patch != BG]
                c_cursor += pw
                if i < len(gaps):
                    c_cursor += gaps[i]
            else:
                r_cursor += row_h
                if row_idx < len(rows) - 1:
                    r_cursor += row_gaps[row_idx]
                continue
            break
        else:
            analyzed = _analyze_objects(grid)
            bg_cells = int(np.sum(grid == BG))
            max_area = max(int(obj["area"]) for obj in analyzed) if analyzed else -1
            if len(analyzed) == len(objects) and bg_cells > max_area:
                return grid
    return None


def _generate_candidate() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    if random.random() < 0.45:
        hole_counts = [0] + random.sample(HOLE_POOL[1:], object_count - 1)
        random.shuffle(hole_counts)
    else:
        hole_counts = random.sample(HOLE_POOL, object_count)
    colors = random.sample(PALETTE, object_count)

    objects = []
    for hole_count, color in zip(hole_counts, colors):
        mask = None
        for _ in range(40):
            mask = _make_object_mask(int(hole_count))
            if mask is not None:
                break
        if mask is None:
            return None
        patch = _colorize(mask, int(color))
        objects.append(
            {
                "holes": int(hole_count),
                "area": int(mask.sum()),
                "color": int(color),
                "patch": patch,
            }
        )

    areas = [int(obj["area"]) for obj in objects]
    if len(set(areas)) != len(areas):
        return None

    hole_order = [idx for idx, _ in sorted(enumerate(objects), key=lambda item: item[1]["holes"])]
    area_order = [idx for idx, _ in sorted(enumerate(objects), key=lambda item: item[1]["area"])]
    if hole_order == area_order:
        return None

    grid = _place_objects(objects)
    if grid is None:
        return None

    analyzed = _analyze_objects(grid)
    if len(analyzed) != object_count:
        return None
    if len({obj["color"] for obj in analyzed}) != object_count:
        return None
    if len({obj["holes"] for obj in analyzed}) != object_count:
        return None
    if len({obj["area"] for obj in analyzed}) != object_count:
        return None

    sorted_by_holes = [obj["color"] for obj in sorted(analyzed, key=lambda obj: obj["holes"])]
    sorted_by_area = [obj["color"] for obj in sorted(analyzed, key=lambda obj: obj["area"])]
    if sorted_by_holes == sorted_by_area:
        return None

    expected = _render_output(analyzed)
    if max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
