import numpy as np


BG = 0
MARKER = 4
OBJECT_COLORS = (1, 2, 3, 6, 7)
RNG = np.random.default_rng()

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Minimal 4-connected component finder, keeping arc_utils (r, c) conventions."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for coords in _components_mask(grid == color):
            bbox = _bbox_coords(coords)
            center_col = bbox[1] + (bbox[3] - bbox[1]) // 2
            objects.append(
                {
                    "coords": coords,
                    "color": int(color),
                    "bbox": bbox,
                    "center_col": int(center_col),
                }
            )
    objects.sort(key=lambda obj: obj["center_col"])
    return objects


def _seed_depths(center_cols: list[int]) -> list[int]:
    if len(center_cols) < 2:
        return [0] * len(center_cols)
    depths: list[int] = []
    for i, col in enumerate(center_cols):
        if i == 0:
            depths.append(center_cols[1] - col)
        elif i == len(center_cols) - 1:
            depths.append(col - center_cols[-2])
        else:
            depths.append(min(col - center_cols[i - 1], center_cols[i + 1] - col))
    return depths


def _border_cells(height: int, width: int, seed_col: int, layer: int) -> list[Coord]:
    bottom = height - 1
    top = bottom - 2 * layer
    left = seed_col - layer
    right = seed_col + layer
    cells: list[Coord] = []
    for c in range(left, right + 1):
        if 0 <= top < height and 0 <= c < width:
            cells.append((top, c))
        if 0 <= bottom < height and 0 <= c < width:
            cells.append((bottom, c))
    for r in range(top + 1, bottom):
        if 0 <= r < height and 0 <= left < width:
            cells.append((r, left))
        if 0 <= r < height and 0 <= right < width:
            cells.append((r, right))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each upper object to the bottom row, then grow color-inherited square borders whose layer counts come from seed spacing.

    Input structure:
    - A black background contains 3 to 5 monochrome 4-connected foreground objects in the upper half.
    - The lower half, especially the last row, is empty so each object can be projected downward by bbox center column.

    Rule:
    - For each object, compute its bbox center column and place a yellow marker on the bottom row at that column.
    - Sort the markers by column; each marker grows upward into d nested square borders, where d is the gap to its nearest neighboring marker.
    - Layer k uses the source object's color and forms the border of the odd square whose bottom midpoint is the marker and whose side length is 2k+1.
    - When borders overlap, larger k overrides smaller k; if k ties, the left seed wins. Original source objects stay unchanged, and yellow bottom markers remain visible.

    Key insight:
    - The projected seeds are only intermediate anchors: the real parameter is the spacing pattern between neighboring seed columns, which controls how many border layers each object can grow.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("input must be a 2D grid")
    h, w = g.shape

    objects = _extract_objects(g)
    if len(objects) < 2:
        raise AssertionError("expected at least two foreground objects")

    center_cols = [obj["center_col"] for obj in objects]
    if len(center_cols) != len(set(center_cols)):
        raise AssertionError("object bbox center columns must be distinct")

    depths = _seed_depths(center_cols)
    reserved = g != BG
    best_layer = np.full((h, w), -1, dtype=int)
    best_seed = np.full((h, w), w + 1, dtype=int)
    best_color = np.zeros((h, w), dtype=int)

    for obj, depth in zip(objects, depths):
        seed_col = int(obj["center_col"])
        color = int(obj["color"])
        for layer in range(1, int(depth) + 1):
            for r, c in _border_cells(h, w, seed_col, layer):
                if reserved[r, c]:
                    continue
                if layer > best_layer[r, c] or (layer == best_layer[r, c] and seed_col < best_seed[r, c]):
                    best_layer[r, c] = layer
                    best_seed[r, c] = seed_col
                    best_color[r, c] = color

    out = g.copy()
    mask = best_layer >= 0
    out[mask] = best_color[mask]
    for seed_col in center_cols:
        out[h - 1, seed_col] = MARKER
    return out


def _random_path(a: Coord, b: Coord) -> list[Coord]:
    r, c = a
    dr = int(np.sign(b[0] - r))
    dc = int(np.sign(b[1] - c))
    steps = [0] * abs(b[0] - r) + [1] * abs(b[1] - c)
    RNG.shuffle(steps)
    path = [(r, c)]
    for step in steps:
        if step == 0:
            r += dr
        else:
            c += dc
        path.append((r, c))
    return path


def _make_object_mask(height: int, width: int) -> np.ndarray:
    for _ in range(200):
        anchors = [
            (0, int(RNG.integers(0, width))),
            (height - 1, int(RNG.integers(0, width))),
            (int(RNG.integers(0, height)), 0),
            (int(RNG.integers(0, height)), width - 1),
        ]
        order = list(range(4))
        RNG.shuffle(order)
        cells = {anchors[order[0]]}
        current = anchors[order[0]]
        for idx in order[1:]:
            for cell in _random_path(current, anchors[idx]):
                cells.add(cell)
            current = anchors[idx]

        max_area = min(10, height * width)
        if len(cells) > max_area:
            continue
        target = int(RNG.integers(max(4, len(cells)), max_area + 1))
        while len(cells) < target:
            frontier: list[Coord] = []
            frontier_set = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells and (nr, nc) not in frontier_set:
                        frontier.append((nr, nc))
                        frontier_set.add((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(RNG.integers(0, len(frontier)))])

        mask = np.zeros((height, width), dtype=bool)
        rows, cols = zip(*cells)
        mask[rows, cols] = True
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not (4 <= int(mask.sum()) <= 10):
            continue
        return mask
    raise RuntimeError("failed to build a connected object mask")


def _sample_centers(width: int, count: int) -> list[int] | None:
    for _ in range(200):
        gaps = [int(x) for x in RNG.integers(3, 7, size=count - 1)]
        if len(set(gaps)) == 1 and RNG.random() < 0.7:
            continue
        start_min = gaps[0]
        start_max = width - 1 - gaps[-1] - sum(gaps)
        if start_min > start_max:
            continue
        start = int(RNG.integers(start_min, start_max + 1))
        centers = [start]
        for gap in gaps:
            centers.append(centers[-1] + gap)
        return centers
    return None


def _build_attempt() -> np.ndarray | None:
    height = int(RNG.integers(18, 26))
    width = int(RNG.integers(18, 30))
    count = int(RNG.integers(3, 6))
    centers = _sample_centers(width, count)
    if centers is None:
        return None

    colors = [int(c) for c in RNG.choice(np.array(OBJECT_COLORS, dtype=int), size=count, replace=False)]
    grid = np.zeros((height, width), dtype=int)
    upper_limit = height // 2 - 1

    for center, color in zip(centers, colors):
        width_choices = [2, 3]
        obj_w = int(RNG.choice(np.array(width_choices, dtype=int)))
        obj_h = int(RNG.integers(2, 5))
        c0 = center - (obj_w - 1) // 2
        c1 = c0 + obj_w - 1
        if c0 < 0 or c1 >= width:
            return None
        top_max = upper_limit - obj_h
        if top_max < 0:
            return None
        r0 = int(RNG.integers(0, min(top_max, 8) + 1))
        mask = _make_object_mask(obj_h, obj_w)
        patch = np.zeros((obj_h, obj_w), dtype=int)
        patch[mask] = color
        if np.any(grid[r0 : r0 + obj_h, c0 : c0 + obj_w] != BG):
            return None
        grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = patch

    objects = _extract_objects(grid)
    if len(objects) != count:
        return None
    if [obj["center_col"] for obj in objects] != centers:
        return None
    if any(obj["bbox"][2] >= height // 2 for obj in objects):
        return None
    if np.any(grid[height // 2 :, :] != BG):
        return None

    depths = _seed_depths(centers)
    if min(depths) < 3 or max(depths) > 8:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if any(out[-1, center] != MARKER for center in centers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
