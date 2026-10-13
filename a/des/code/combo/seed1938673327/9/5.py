from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _row_arrangement(components: list[dict[str, object]]) -> bool:
    row_centers = [int(comp["bbox"][0]) + int(comp["bbox"][2]) for comp in components]
    col_centers = [int(comp["bbox"][1]) + int(comp["bbox"][3]) for comp in components]
    row_span = max(row_centers) - min(row_centers)
    col_span = max(col_centers) - min(col_centers)
    if row_span != col_span:
        return row_span < col_span

    min_r = min(int(comp["bbox"][0]) for comp in components)
    max_r = max(int(comp["bbox"][2]) for comp in components)
    min_c = min(int(comp["bbox"][1]) for comp in components)
    max_c = max(int(comp["bbox"][3]) for comp in components)
    return (max_c - min_c) >= (max_r - min_r)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose a compression axis from the global object alignment, then turn each object
    into a same-area color band anchored at its own bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _dominant_color(g)
    components = _find_components(g, bg)
    assert len(components) >= 2, "expected multiple foreground objects"
    colors = [int(comp["color"]) for comp in components]
    assert len(colors) == len(set(colors)), "expected distinct object colors"

    row_arranged = _row_arrangement(components)
    out = np.full((h, w), bg, dtype=int)
    for comp in sorted(components, key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1]))):
        color = int(comp["color"])
        area = int(comp["area"])
        r0, c0, r1, c1 = map(int, comp["bbox"])

        if row_arranged:
            row = (r0 + r1) // 2
            end = c0 + area
            assert end <= w, "compressed band exceeds grid width"
            assert np.all(out[row, c0:end] == bg), "compressed bands must not overlap"
            out[row, c0:end] = color
        else:
            col = (c0 + c1) // 2
            end = r0 + area
            assert end <= h, "compressed band exceeds grid height"
            assert np.all(out[r0:end, col] == bg), "compressed bands must not overlap"
            out[r0:end, col] = color

    return out


# --------------- generator ---------------

BACKGROUND_COLOR = 0
OBJECT_COUNT_RANGE = (2, 4)
AREA_RANGE = (4, 7)
BBOX_HEIGHT_CHOICES = (3, 5)
MAX_BBOX_WIDTH = 5
SIDE_MARGIN = 1
MIN_GAP = 1
EXTRA_WIDTH_RANGE = (0, 6)
GRID_HEIGHT_RANGE = (10, 16)
ROW_JITTER_CHOICES = (-1, 0, 1)
SHAPE_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400


def _path_cells(a: tuple[int, int], b: tuple[int, int], rng: random.Random) -> set[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = {(r, c)}
    while (r, c) != (tr, tc):
        moves: list[tuple[int, int]] = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = rng.choice(moves)
        cells.add((r, c))
    return cells


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _touches_all_sides(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    return 0 in rows and h - 1 in rows and 0 in cols and w - 1 in cols


def _random_shape(area: int, box_h: int, box_w: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        left = (rng.randrange(box_h), 0)
        right = (rng.randrange(box_h), box_w - 1)
        top = (0, rng.randrange(box_w))
        bottom = (box_h - 1, rng.randrange(box_w))
        hub = (rng.randrange(box_h), rng.randrange(box_w))

        cells = set()
        for anchor in (left, right, top, bottom):
            cells |= _path_cells(anchor, hub, rng)
        if len(cells) > area:
            continue

        while len(cells) < area:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in DIR4
                if 0 <= r + dr < box_h and 0 <= c + dc < box_w and (r + dr, c + dc) not in cells
            }
            if not frontier:
                break
            cells.add(rng.choice(sorted(frontier)))

        if len(cells) != area:
            continue
        if _connected(cells) and _touches_all_sides(cells, box_h, box_w):
            return np.array(sorted(cells), dtype=int)
    return None


def _bbox_candidates(area: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for box_h in BBOX_HEIGHT_CHOICES:
        for box_w in range(2, min(MAX_BBOX_WIDTH, area) + 1):
            if box_h + box_w - 1 <= area <= box_h * box_w:
                candidates.append((box_h, box_w))
    return candidates


def _distribute_extra(total: int, parts: int, rng: random.Random) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[rng.randrange(parts)] += 1
    return values


def _band_intervals(lengths: list[int], width: int, rng: random.Random) -> list[int]:
    base = sum(lengths) + MIN_GAP * (len(lengths) - 1) + 2 * SIDE_MARGIN
    extra = width - base
    assert extra >= 0
    slack = _distribute_extra(extra, len(lengths) + 1, rng)

    starts: list[int] = []
    cursor = SIDE_MARGIN + slack[0]
    for idx, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length
        if idx < len(lengths) - 1:
            cursor += MIN_GAP + slack[idx + 1]
    return starts


def _row_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    areas = [rng.randint(*AREA_RANGE) for _ in range(n_objects)]
    bbox_sizes = []
    for area in areas:
        candidates = _bbox_candidates(area)
        if not candidates:
            return None
        bbox_sizes.append(rng.choice(candidates))

    min_width = sum(areas) + MIN_GAP * (n_objects - 1) + 2 * SIDE_MARGIN
    width = min_width + rng.randint(*EXTRA_WIDTH_RANGE)
    height = rng.randint(*GRID_HEIGHT_RANGE)
    starts = _band_intervals(areas, width, rng)

    max_half = max(box_h // 2 for box_h, _box_w in bbox_sizes)
    base_row_min = max_half + 1
    base_row_max = height - max_half - 2
    if base_row_min > base_row_max:
        return None
    base_row = rng.randint(base_row_min, base_row_max)

    specs = []
    for area, (box_h, box_w), left in zip(areas, bbox_sizes, starts):
        shape = _random_shape(area, box_h, box_w, rng)
        if shape is None:
            return None
        jitter = rng.choice(ROW_JITTER_CHOICES)
        center_row = base_row + jitter
        top = center_row - box_h // 2
        if top < 0 or top + box_h > height:
            return None
        specs.append(
            {
                "area": area,
                "color": -1,
                "left": left,
                "top": top,
                "box_h": box_h,
                "box_w": box_w,
                "shape": shape,
            }
        )

    colors = rng.sample(list(range(1, 10)), n_objects)
    for spec, color in zip(specs, colors):
        spec["color"] = color

    inp = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    out = np.full((height, width), BACKGROUND_COLOR, dtype=int)

    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        color = int(spec["color"])
        area = int(spec["area"])
        box_h = int(spec["box_h"])
        shape = np.asarray(spec["shape"], dtype=int)

        abs_coords = shape + np.array([top, left])
        if np.any(inp[abs_coords[:, 0], abs_coords[:, 1]] != BACKGROUND_COLOR):
            return None
        inp[abs_coords[:, 0], abs_coords[:, 1]] = color

        row = top + box_h // 2
        if np.any(out[row, left : left + area] != BACKGROUND_COLOR):
            return None
        out[row, left : left + area] = color

    return inp, out


def _valid_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = _dominant_color(inp)
    if bg != BACKGROUND_COLOR:
        return False
    components = _find_components(inp, bg)
    if len(components) < 2:
        return False

    colors = [int(comp["color"]) for comp in components]
    if len(colors) != len(set(colors)):
        return False

    row_centers = [int(comp["bbox"][0]) + int(comp["bbox"][2]) for comp in components]
    col_centers = [int(comp["bbox"][1]) + int(comp["bbox"][3]) for comp in components]
    row_span = max(row_centers) - min(row_centers)
    col_span = max(col_centers) - min(col_centers)
    if row_span == col_span:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    row_input, row_output = _row_case(rng) or (None, None)
    if row_input is None or row_output is None:
        return None

    if rng.random() < 0.5:
        inp, expected = row_input, row_output
    else:
        inp, expected = row_input.T.copy(), row_output.T.copy()

    if not _valid_case(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
