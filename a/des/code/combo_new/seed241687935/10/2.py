import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
CLUSTER_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(cells), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _transform_cells(cells: np.ndarray, rotate_k: int, flip: bool) -> np.ndarray:
    arr = cells.copy()
    if flip:
        arr[:, 1] *= -1
    for _ in range(rotate_k % 4):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    return _normalize_cells({tuple(rc) for rc in arr})


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_separator_row(grid: np.ndarray) -> int:
    candidates = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BLACK and np.all(grid[r, :] == grid[r, 0])
    ]
    assert len(candidates) == 1, "expected one full-width separator row"
    return int(candidates[0])


def _monochrome_components(grid: np.ndarray, valid_mask: np.ndarray, row_offset: int = 0) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not valid_mask[r, c] or visited[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr + row_offset, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or not valid_mask[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            bbox = _bbox_from_coords(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": int(len(cells)),
                }
            )
    return components


def _find_shelves(grid: np.ndarray, separator_row: int) -> list[dict]:
    lower = grid[separator_row + 1 :, :]
    components = _monochrome_components(lower, lower == GREY, row_offset=separator_row + 1)
    shelves: list[dict] = []

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        pts = {tuple(rc) for rc in comp["coords"]}
        assert c1 - c0 >= 2 and r1 - r0 >= 1, "shelf must have an interior"
        for c in range(c0, c1 + 1):
            assert (r1, c) in pts, "shelf bottom must be solid"
        for r in range(r0, r1 + 1):
            assert (r, c0) in pts and (r, c1) in pts, "shelf sides must be solid"
        for r, c in pts:
            assert r == r1 or c == c0 or c == c1, "shelf must be a U shape"

        shelves.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_left": c0 + 1,
                "inner_right": c1 - 1,
                "inner_top": r0,
                "inner_bottom": r1 - 1,
                "inner_w": c1 - c0 - 1,
                "inner_h": r1 - r0,
            }
        )

    shelves.sort(key=lambda shelf: shelf["bbox"][1])
    return shelves


def _arranged_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    separator_row = _find_separator_row(g)
    upper = g[:separator_row, :]

    clusters = _monochrome_components(upper, upper != BLACK)
    assert clusters, "expected at least one cluster"
    areas = [comp["area"] for comp in clusters]
    assert len(areas) == len(set(areas)), "cluster sizes must be distinct"

    shelves = _find_shelves(g, separator_row)
    assert len(shelves) == len(clusters), "number of shelves must match number of clusters"

    out = g.copy()
    out[:separator_row, :][out[:separator_row, :] != BLACK] = BLACK

    ordered_clusters = sorted(
        clusters,
        key=lambda comp: (comp["area"], comp["bbox"][0], comp["bbox"][1]),
    )

    for comp, shelf in zip(ordered_clusters, shelves):
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height <= shelf["inner_h"] and width <= shelf["inner_w"]

        rel = comp["coords"] - np.array([r0, c0], dtype=int)
        dest_r0 = shelf["inner_bottom"] - (height - 1)
        dest_c0 = shelf["inner_left"] + (shelf["inner_w"] - width) // 2
        for rr, cc in rel:
            out[dest_r0 + int(rr), dest_c0 + int(cc)] = int(comp["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the scattered clusters by size and place them into the shelves from left to right.

    Input structure:
    - Black background.
    - A single full-width Maroon separator row splits the grid into an upper scatter area and a lower shelf area.
    - Above the separator are several 4-connected monochrome clusters, each in a different color and with a distinct pixel count.
    - Below the separator are equally sized Grey U-shaped shelf slots arranged from left to right.

    Rule:
    - Keep the separator row and every shelf outline unchanged.
    - Clear all cluster pixels from the upper area back to Black.
    - Sort the upper clusters by their pixel counts from smallest to largest.
    - Put the i-th smallest cluster into the i-th shelf from the left, preserving the cluster's shape, bottom-aligning it inside the shelf, and centering it horizontally with left bias when exact centering is impossible.

    Key insight:
    - The shelves do not match clusters by color or original position; the hidden link is the ordering induced by each cluster's size.
    """
    return _arranged_output(grid)


def _random_polyomino(size: int) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            frontier_list = sorted(frontier)
            cells.add(frontier_list[int(RNG.integers(len(frontier_list)))])

        shape = _normalize_cells(cells)
        shape = _transform_cells(shape, int(RNG.integers(4)), bool(RNG.integers(2)))
        bbox = _bbox_from_coords(shape)
        if bbox[2] + 1 <= 4 and bbox[3] + 1 <= 4:
            return shape
    raise RuntimeError(f"failed to generate a compact polyomino of size {size}")


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _place_upper_clusters(
    grid_w: int,
    upper_h: int,
    objects: list[dict],
) -> list[dict] | None:
    occupied = np.zeros((upper_h, grid_w), dtype=bool)
    placed: list[dict] = []

    order = list(range(len(objects)))
    RNG.shuffle(order)

    for idx in order:
        obj = objects[idx]
        shape = obj["shape"]
        h = obj["bbox"][2] + 1
        w = obj["bbox"][3] + 1
        for _ in range(300):
            top = int(RNG.integers(0, upper_h - h + 1))
            left = int(RNG.integers(0, grid_w - w + 1))
            rs = slice(max(0, top - 1), min(upper_h, top + h + 1))
            cs = slice(max(0, left - 1), min(grid_w, left + w + 1))
            if occupied[rs, cs].any():
                continue

            abs_coords = shape + np.array([top, left], dtype=int)
            for r, c in abs_coords:
                occupied[int(r), int(c)] = True

            placed.append(
                {
                    "color": obj["color"],
                    "area": obj["area"],
                    "coords": abs_coords,
                    "bbox": (top, left, top + h - 1, left + w - 1),
                }
            )
            break
        else:
            return None

    placed.sort(key=lambda item: item["bbox"][1])
    x_order_areas = [item["area"] for item in placed]
    if x_order_areas == sorted(x_order_areas):
        return None
    row_starts = {item["bbox"][0] for item in placed}
    if len(row_starts) == 1:
        return None
    return placed


def _draw_shelves(
    grid: np.ndarray,
    shelf_top: int,
    shelf_lefts: list[int],
    inner_h: int,
    inner_w: int,
) -> list[dict]:
    outer_w = inner_w + 2
    shelves: list[dict] = []
    bottom = shelf_top + inner_h

    for left in shelf_lefts:
        right = left + outer_w - 1
        grid[bottom, left : right + 1] = GREY
        grid[shelf_top : bottom + 1, left] = GREY
        grid[shelf_top : bottom + 1, right] = GREY
        shelves.append(
            {
                "bbox": (shelf_top, left, bottom, right),
                "inner_left": left + 1,
                "inner_right": right - 1,
                "inner_top": shelf_top,
                "inner_bottom": bottom - 1,
                "inner_w": inner_w,
                "inner_h": inner_h,
            }
        )
    return shelves


def _build_expected(
    grid_shape: tuple[int, int],
    separator_row: int,
    upper_clusters: list[dict],
    shelves: list[dict],
) -> np.ndarray:
    out = np.full(grid_shape, BLACK, dtype=int)
    out[separator_row, :] = MAROON
    for shelf in shelves:
        r0, c0, r1, c1 = shelf["bbox"]
        out[r1, c0 : c1 + 1] = GREY
        out[r0 : r1 + 1, c0] = GREY
        out[r0 : r1 + 1, c1] = GREY

    ordered = sorted(upper_clusters, key=lambda item: item["area"])
    for cluster, shelf in zip(ordered, shelves):
        coords = cluster["coords"]
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        rel = coords - np.array([r0, c0], dtype=int)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        dest_r0 = shelf["inner_bottom"] - (height - 1)
        dest_c0 = shelf["inner_left"] + (shelf["inner_w"] - width) // 2
        _paint_cells(out, rel + np.array([dest_r0, dest_c0], dtype=int), cluster["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    cluster_count = int(RNG.integers(3, 5))
    sizes = sorted(RNG.choice(np.arange(1, 7), size=cluster_count, replace=False).tolist())
    colors = RNG.choice(CLUSTER_COLORS, size=cluster_count, replace=False).tolist()

    objects: list[dict] = []
    for area, color in zip(sizes, colors):
        shape = _random_polyomino(int(area))
        bbox = _bbox_from_coords(shape)
        objects.append(
            {
                "area": int(area),
                "color": int(color),
                "shape": shape,
                "bbox": bbox,
            }
        )

    max_h = max(obj["bbox"][2] + 1 for obj in objects)
    max_w = max(obj["bbox"][3] + 1 for obj in objects)

    inner_h = int(RNG.integers(max_h, 5))
    inner_w = int(RNG.integers(max(max_w, 3), 5))
    outer_w = inner_w + 2
    outer_h = inner_h + 1

    gap = 1
    left_margin = int(RNG.integers(1, 3))
    right_margin = int(RNG.integers(1, 3))
    grid_w = left_margin + cluster_count * outer_w + (cluster_count - 1) * gap + right_margin
    if grid_w > 30:
        return None

    upper_h = int(RNG.integers(max(max_h + 2, 7), 16))
    lower_gap = int(RNG.integers(1, 3))
    bottom_margin = int(RNG.integers(0, 2))
    grid_h = upper_h + 1 + lower_gap + outer_h + bottom_margin
    if grid_h > 30:
        return None

    placed_clusters = _place_upper_clusters(grid_w, upper_h, objects)
    if placed_clusters is None:
        return None

    separator_row = upper_h
    shelf_top = separator_row + 1 + lower_gap
    shelf_lefts = [left_margin + i * (outer_w + gap) for i in range(cluster_count)]

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for cluster in placed_clusters:
        _paint_cells(grid, cluster["coords"], cluster["color"])
    grid[separator_row, :] = MAROON
    shelves = _draw_shelves(grid, shelf_top, shelf_lefts, inner_h, inner_w)

    expected = _build_expected(grid.shape, separator_row, placed_clusters, shelves)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid shelf-sorting input")
