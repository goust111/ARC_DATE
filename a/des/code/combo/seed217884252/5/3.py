import random

import numpy as np


BG = 0
FG_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
MAX_GRID_SIZE = 30
MAX_OUTPUT_WIDTH = 30
MIN_OBJECTS = 2
MAX_OBJECTS = 4


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _object_info(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    colors = {int(grid[r, c]) for r, c in cells}
    assert len(colors) == 1, "each object must be monochromatic"
    return {
        "bbox": (r0, c0, r1, c1),
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
        "area": len(cells),
        "crop": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
    }


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    assert comps, "expected at least one foreground object"
    return [_object_info(g, cells) for cells in comps]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge same-height objects into one horizontal strip ordered by left bbox column.

    Each object's cropped bbox is pasted left-to-right. The gap after an object
    has width equal to that object's pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    heights = {obj["height"] for obj in objects}
    assert len(heights) == 1, "all object bbox heights must match"

    objects = sorted(objects, key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    out_h = objects[0]["height"]
    out_w = sum(obj["width"] for obj in objects) + sum(obj["area"] for obj in objects[:-1])

    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for i, obj in enumerate(objects):
        crop = obj["crop"]
        assert crop.shape[0] == out_h
        w = obj["width"]
        out[:, col : col + w] = crop
        col += w
        if i != len(objects) - 1:
            col += obj["area"]
    assert col == out_w
    return out


def _sample_height(rng: random.Random, n_objects: int) -> int:
    if n_objects == 2:
        return rng.randint(2, 5)
    if n_objects == 3:
        return rng.randint(2, 4)
    return rng.randint(2, 3)


def _sample_width(rng: random.Random, n_objects: int) -> int:
    if n_objects == 2:
        return rng.randint(2, 6)
    if n_objects == 3:
        return rng.randint(2, 5)
    return rng.randint(2, 4)


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    occupied = set(zip(*np.where(mask)))
    for r, c in occupied:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _sample_target_area(rng: random.Random, base_area: int, total_area: int) -> int:
    if base_area == total_area:
        return total_area

    slack = total_area - base_area
    band = rng.choice(("sparse", "medium", "dense", "solid"))
    if band == "sparse":
        lo = base_area
        hi = base_area + max(0, slack // 3)
    elif band == "medium":
        lo = base_area + max(0, slack // 4)
        hi = base_area + max(0, (3 * slack) // 4)
    elif band == "dense":
        lo = base_area + max(0, (2 * slack) // 3)
        hi = total_area
    else:
        lo = total_area
        hi = total_area

    lo = min(lo, total_area)
    hi = min(max(lo, hi), total_area)
    return rng.randint(lo, hi)


def _random_object_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)

    spine_col = rng.randrange(width)
    mask[:, spine_col] = True

    left_row = rng.randrange(height)
    right_row = rng.randrange(height)
    mask[left_row, : spine_col + 1] = True
    mask[right_row, spine_col:] = True

    target_area = _sample_target_area(rng, int(mask.sum()), height * width)
    while int(mask.sum()) < target_area:
        choices = _frontier(mask)
        assert choices
        r, c = rng.choice(choices)
        mask[r, c] = True

    assert mask[0].any() and mask[-1].any()
    assert mask[:, 0].any() and mask[:, -1].any()
    assert len(_components(mask)) == 1
    return mask


def _render_input(
    masks: list[np.ndarray],
    fg: int,
    top_rows: list[int],
    left_cols: list[int],
    canvas_h: int,
    canvas_w: int,
) -> np.ndarray:
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    for mask, top, left in zip(masks, top_rows, left_cols):
        h, w = mask.shape
        grid[top : top + h, left : left + w][mask] = fg
    return grid


def _oracle_output(
    masks: list[np.ndarray],
    fg: int,
    left_cols: list[int],
) -> np.ndarray:
    order = sorted(range(len(masks)), key=lambda i: left_cols[i])
    height = masks[0].shape[0]
    widths = [masks[i].shape[1] for i in order]
    areas = [int(masks[i].sum()) for i in order]
    out_w = sum(widths) + sum(areas[:-1])

    out = np.full((height, out_w), BG, dtype=int)
    col = 0
    for idx, obj_idx in enumerate(order):
        mask = masks[obj_idx]
        h, w = mask.shape
        tile = np.full((h, w), BG, dtype=int)
        tile[mask] = fg
        out[:, col : col + w] = tile
        col += w
        if idx != len(order) - 1:
            col += int(mask.sum())
    return out


def _sample_scene_layout(
    rng: random.Random,
    widths: list[int],
    height: int,
) -> tuple[list[int], list[int], int, int] | None:
    n = len(widths)
    min_content_w = sum(widths) + (n - 1)
    if min_content_w > MAX_GRID_SIZE:
        return None

    margin_left = rng.randint(0, 3)
    margin_right = rng.randint(0, 3)
    gaps = [rng.randint(1, 5) for _ in range(n - 1)]
    left_cols = [margin_left]
    for i in range(1, n):
        left_cols.append(left_cols[-1] + widths[i - 1] + gaps[i - 1])
    canvas_w = left_cols[-1] + widths[-1] + margin_right
    if canvas_w > MAX_GRID_SIZE:
        return None

    extra_top = rng.randint(0, 4)
    extra_bottom = rng.randint(0, 4)
    canvas_h = height + extra_top + extra_bottom
    if canvas_h > MAX_GRID_SIZE:
        return None

    top_rows = [rng.randint(0, canvas_h - height) for _ in range(n)]
    return top_rows, left_cols, canvas_h, canvas_w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
    height = _sample_height(rng, n_objects)
    fg = rng.choice(FG_COLORS)

    masks: list[np.ndarray] = []
    for _ in range(n_objects):
        width = _sample_width(rng, n_objects)
        masks.append(_random_object_mask(height, width, rng))

    output_width = sum(mask.shape[1] for mask in masks) + sum(int(mask.sum()) for mask in masks[:-1])
    if output_width > MAX_OUTPUT_WIDTH:
        return None

    layout = _sample_scene_layout(rng, [mask.shape[1] for mask in masks], height)
    if layout is None:
        return None
    top_rows, left_cols, canvas_h, canvas_w = layout

    grid = _render_input(masks, fg, top_rows, left_cols, canvas_h, canvas_w)
    expected = _oracle_output(masks, fg, left_cols)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
