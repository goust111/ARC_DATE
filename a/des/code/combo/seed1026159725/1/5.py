import random
from collections import Counter, deque

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
GRID_MIN_SIZE = 22
GRID_MAX_SIZE = 30
OBJECT_COUNT_MIN = 4
OBJECT_COUNT_MAX = 6
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(values))


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        queue = deque([(rr, cc)])
        seen[rr, cc] = True
        coords = []
        while queue:
            cr, cc = queue.popleft()
            coords.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        rows = [r0 for r0, _ in coords]
        cols = [c0 for _, c0 in coords]
        components.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )
    return components


def _stripe_counts(patch: np.ndarray, axis: int) -> Counter | None:
    lines = patch if axis == 0 else patch.T
    sequence = []
    for line in lines:
        if not np.all(line == line[0]):
            return None
        sequence.append(int(line[0]))
    if len(set(sequence)) != 2:
        return None
    if any(a == b for a, b in zip(sequence, sequence[1:])):
        return None
    return Counter(sequence)


def _packed_width(patches: list[np.ndarray]) -> int:
    if not patches:
        return 0
    return sum(int(p.shape[1]) for p in patches) + len(patches) - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort striped rectangles by stripe direction into upper/lower bands, then
    rotate the ones whose stripe-color counts are imbalanced.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _find_components(g != bg)
    assert components, "expected at least one foreground object"

    objects = []
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.size == len(component["coords"])
        assert np.all(patch != bg), "objects must be solid rectangles"
        colors = np.unique(patch)
        assert len(colors) == 2, "each object must use exactly two non-background colors"

        row_counts = _stripe_counts(patch, axis=0)
        col_counts = _stripe_counts(patch, axis=1)
        assert (row_counts is None) != (col_counts is None), "stripe direction must be unique"

        stripe_counts = row_counts if row_counts is not None else col_counts
        rotate = max(stripe_counts.values()) > min(stripe_counts.values())
        moved = np.rot90(patch, -1) if rotate else patch.copy()
        band = "top" if row_counts is not None else "bottom"
        objects.append({"band": band, "patch": moved, "key": (c0, r0)})

    objects.sort(key=lambda obj: obj["key"])
    top = [obj["patch"] for obj in objects if obj["band"] == "top"]
    bottom = [obj["patch"] for obj in objects if obj["band"] == "bottom"]

    top_h = max((int(p.shape[0]) for p in top), default=0)
    bottom_h = max((int(p.shape[0]) for p in bottom), default=0)
    sep_h = 1 if top and bottom else 0
    out_h = top_h + sep_h + bottom_h
    out_w = max(_packed_width(top), _packed_width(bottom))
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for patch in top:
        ph, pw = patch.shape
        out[:ph, col : col + pw] = patch
        col += pw + 1

    col = 0
    for patch in bottom:
        ph, pw = patch.shape
        row = out_h - ph
        out[row : row + ph, col : col + pw] = patch
        col += pw + 1

    return out


def _make_striped_patch(
    height: int,
    width: int,
    orientation: str,
    colors: tuple[int, int],
    start: int,
) -> np.ndarray:
    patch = np.empty((height, width), dtype=int)
    if orientation == "H":
        for r in range(height):
            patch[r, :] = colors[(start + r) % 2]
    else:
        for c in range(width):
            patch[:, c] = colors[(start + c) % 2]
    return patch


def _sample_spec(orientation: str, rotate: bool, bg: int) -> dict:
    available = [color for color in PALETTE if color != bg]
    color_a, color_b = random.sample(available, 2)
    start = random.randrange(2)

    if orientation == "H":
        height = random.choice((3, 5)) if rotate else random.choice((2, 4, 6))
        width = random.randint(2, 6)
    else:
        height = random.randint(2, 6)
        width = random.choice((3, 5)) if rotate else random.choice((2, 4, 6))

    patch = _make_striped_patch(height, width, orientation, (color_a, color_b), start)
    placed = np.rot90(patch, -1) if rotate else patch.copy()
    return {
        "orientation": orientation,
        "rotate": rotate,
        "input_patch": patch,
        "output_patch": placed,
    }


def _sample_specs(bg: int) -> list[dict] | None:
    count = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    branches = [("H", True), ("H", False), ("V", True), ("V", False)]
    while len(branches) < count:
        branches.append((random.choice(("H", "V")), bool(random.getrandbits(1))))

    specs = [_sample_spec(orientation, rotate, bg) for orientation, rotate in branches]
    top = [spec["output_patch"] for spec in specs if spec["orientation"] == "H"]
    bottom = [spec["output_patch"] for spec in specs if spec["orientation"] == "V"]

    if not top or not bottom:
        return None
    if _packed_width(top) > GRID_MAX_SIZE or _packed_width(bottom) > GRID_MAX_SIZE:
        return None
    if (
        max(int(p.shape[0]) for p in top)
        + 1
        + max(int(p.shape[0]) for p in bottom)
        > GRID_MAX_SIZE
    ):
        return None
    return specs


def _place_specs(specs: list[dict], grid_h: int, grid_w: int, bg: int) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    used_lefts = set()
    order = specs[:]
    random.shuffle(order)

    for spec in order:
        patch = spec["input_patch"]
        ph, pw = patch.shape
        placed = False
        for _ in range(250):
            row = random.randint(1, grid_h - ph - 1)
            col = random.randint(1, grid_w - pw - 1)
            if col in used_lefts:
                continue
            if blocked[row : row + ph, col : col + pw].any():
                continue
            grid[row : row + ph, col : col + pw] = patch
            blocked[
                max(0, row - 1) : min(grid_h, row + ph + 1),
                max(0, col - 1) : min(grid_w, col + pw + 1),
            ] = True
            used_lefts.add(col)
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    specs = _sample_specs(bg)
    if specs is None:
        return None

    grid_h = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    grid_w = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    grid = _place_specs(specs, grid_h, grid_w, bg)
    if grid is None:
        return None
    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return None

    out = main(grid)
    if out.shape[0] > GRID_MAX_SIZE or out.shape[1] > GRID_MAX_SIZE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
