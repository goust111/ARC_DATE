import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
INPUT_COLORS = [1, 2, 3, 4, 6, 7, 9]
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _dense_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for comp in _components(grid != BLACK):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != BLACK), "each foreground component must fill a dense rectangle"
        rects.append({"bbox": (r0, c0, r1, c1), "patch": patch.copy()})
    return rects


def _project(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    assert source.shape == mask.shape
    out = np.full(source.shape, BLACK, dtype=int)
    visible = mask != GREY
    same = visible & (mask == source)
    diff = visible & ~same
    out[same] = AZURE
    out[diff] = mask[diff]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the upper source rectangle through the lower filter mask and crop to the source size.

    Input structure:
    - The background is black.
    - One dense rectangular source pattern sits in the upper part of the grid and uses only non-black, non-grey colors.
    - One dense rectangular filter mask of the same shape sits lower in the grid, separated from the source by at least one black row.
    - Mask cells are either grey blockers or non-black replacement colors.

    Rule:
    - Detect the two non-black rectangles and sort them by row so the upper one is the source and the lower one is the mask.
    - Align the rectangles by their bounding-box coordinates and compare them cell by cell.
    - If a mask cell is grey, output black at that position.
    - If a non-grey mask cell matches the source color at the same position, output azure.
    - Otherwise output the mask color itself, then return only the projected rectangle.

    Key insight:
    - The lower rectangle is both a visibility mask and a recoloring guide, and exact color matches trigger a special azure highlight.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _dense_rectangles(g)
    assert len(rects) == 2, "expected exactly two non-black rectangles"
    rects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    source_info, mask_info = rects

    sr0, sc0, sr1, sc1 = source_info["bbox"]
    mr0, mc0, mr1, mc1 = mask_info["bbox"]
    assert mr0 > sr1, "source and mask must be vertically separated"

    source = source_info["patch"]
    mask = mask_info["patch"]
    assert source.shape == mask.shape, "source and mask must have identical size"
    assert not np.any(source == GREY), "source pattern must not contain grey"
    assert not np.any(mask == BLACK), "mask must not contain black"

    return _project(source, mask)


def _pattern_indices(h: int, w: int, k: int, min_unique: int) -> np.ndarray:
    assert k >= 1
    for _ in range(100):
        mode = random.choice(["random", "rows", "cols", "blocks", "diagonal", "rings"])
        arr = np.zeros((h, w), dtype=int)
        if mode == "random":
            arr = np.array([random.randrange(k) for _ in range(h * w)], dtype=int).reshape(h, w)
        elif mode == "rows":
            for r in range(h):
                arr[r, :] = random.randrange(k)
        elif mode == "cols":
            for c in range(w):
                arr[:, c] = random.randrange(k)
        elif mode == "blocks":
            row_cuts = sorted(random.sample(range(1, h), k=min(2, h - 1)))
            col_cuts = sorted(random.sample(range(1, w), k=min(2, w - 1)))
            row_edges = [0] + row_cuts + [h]
            col_edges = [0] + col_cuts + [w]
            for i in range(len(row_edges) - 1):
                for j in range(len(col_edges) - 1):
                    arr[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]] = random.randrange(k)
        elif mode == "diagonal":
            ar = random.randint(1, max(1, k - 1))
            ac = random.randint(1, max(1, k - 1))
            offset = random.randrange(k)
            for r in range(h):
                for c in range(w):
                    arr[r, c] = (ar * r + ac * c + offset) % k
        else:
            offset = random.randrange(k)
            for r in range(h):
                for c in range(w):
                    dist = min(r, c, h - 1 - r, w - 1 - c)
                    arr[r, c] = (dist + offset) % k

        if len(set(map(int, arr.reshape(-1)))) >= min_unique:
            return arr
    raise RuntimeError("failed to build a diverse pattern")


def _make_source(h: int, w: int) -> np.ndarray:
    palette_size = random.randint(2, min(4, len(INPUT_COLORS)))
    palette = random.sample(INPUT_COLORS, palette_size)
    idx = _pattern_indices(h, w, palette_size, min_unique=2)
    source = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            source[r, c] = palette[int(idx[r, c])]
    return source


def _make_mask(source: np.ndarray) -> np.ndarray:
    h, w = source.shape
    branch_map = _pattern_indices(h, w, 3, min_unique=3)
    mask = np.empty((h, w), dtype=int)
    replacement_palette = random.sample(INPUT_COLORS, random.randint(2, min(4, len(INPUT_COLORS))))

    for r in range(h):
        for c in range(w):
            src = int(source[r, c])
            branch = int(branch_map[r, c])
            if branch == 0:
                mask[r, c] = GREY
            elif branch == 1:
                mask[r, c] = src
            else:
                candidates = [color for color in replacement_palette if color != src]
                if not candidates:
                    candidates = [color for color in INPUT_COLORS if color != src]
                mask[r, c] = random.choice(candidates)

    assert np.any(mask == GREY)
    assert np.any(mask == source)
    assert np.any((mask != GREY) & (mask != source))
    return mask


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(3, 10)
    w = random.randint(3, 10)
    sep = random.randint(1, 4)
    top_margin = random.randint(0, 3)
    bottom_margin = random.randint(0, 3)

    left_cap = min(10, 30 - w)
    source_left = random.randint(0, left_cap)
    mask_left = random.randint(0, left_cap)
    base_width = max(source_left, mask_left) + w
    extra_right = random.randint(0, min(6, 30 - base_width))

    grid_h = top_margin + h + sep + h + bottom_margin
    grid_w = base_width + extra_right
    if grid_h > 30 or grid_w > 30:
        return None

    source = _make_source(h, w)
    mask = _make_mask(source)
    expected = _project(source, mask)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    source_top = top_margin
    mask_top = source_top + h + sep
    grid[source_top : source_top + h, source_left : source_left + w] = source
    grid[mask_top : mask_top + h, mask_left : mask_left + w] = mask

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
