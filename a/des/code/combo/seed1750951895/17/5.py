import numpy as np


BG = 8  # ARC azure
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400


def _majority_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_same_color(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _shape_from_coords(coords: np.ndarray) -> dict:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    rel = coords - np.array([r0, c0], dtype=int)
    h = int(rel[:, 0].max()) + 1
    w = int(rel[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return {"r0": r0, "c0": c0, "h": h, "w": w, "mask": mask}


def _extract_ordered_objects(scene: np.ndarray, legend: list[int], bg: int) -> list[dict]:
    present = set(map(int, np.unique(scene))) - {bg}
    assert present == set(legend), "scene colors must match legend colors exactly"
    objects = []
    for color in legend:
        comps = _components_same_color(scene == int(color))
        assert len(comps) == 1, "each legend color must correspond to exactly one object"
        obj = _shape_from_coords(comps[0])
        obj["color"] = int(color)
        objects.append(obj)
    return objects


def _pack_objects(objects: list[dict], canvas_shape: tuple[int, int], bg: int) -> np.ndarray:
    h, w = map(int, canvas_shape)
    out = np.full((h, w), bg, dtype=int)
    y = 0
    x = 0
    row_h = 0
    for obj in objects:
        oh = int(obj["h"])
        ow = int(obj["w"])
        if x > 0 and x + ow > w:
            y += row_h + 1
            x = 0
            row_h = 0
        assert y + oh <= h, "packed objects exceed canvas height"
        assert ow <= w, "object exceeds canvas width"
        block = out[y : y + oh, x : x + ow]
        block[obj["mask"]] = int(obj["color"])
        x += ow + 1
        row_h = max(row_h, oh)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder colored objects by the top-to-bottom order of a right-side legend.

    Input structure:
    - Azure background.
    - The rightmost non-background column is a vertical legend: one color per row.
    - Left of the legend is a scene containing exactly one 4-connected object per legend color.

    Rule:
    - Read legend colors from top to bottom.
    - Remove the legend.
    - Pack the matching objects onto the remaining canvas in that order, filling rows
      left-to-right with a one-cell gap between neighboring objects and rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _majority_color(g)

    non_bg_cols = np.flatnonzero(np.any(g != bg, axis=0))
    assert non_bg_cols.size > 0, "expected foreground content"
    legend_col = int(non_bg_cols[-1])

    legend_rows = np.flatnonzero(g[:, legend_col] != bg)
    assert legend_rows.size >= 2, "legend must contain at least two entries"
    legend = [int(g[r, legend_col]) for r in legend_rows]
    assert len(set(legend)) == len(legend), "legend colors must be unique"

    canvas_end = legend_col
    while canvas_end > 0 and np.all(g[:, canvas_end - 1] == bg):
        canvas_end -= 1
    scene = g[:, :canvas_end]
    assert scene.shape[1] > 0, "scene must have positive width"

    objects = _extract_ordered_objects(scene, legend, bg)
    return _pack_objects(objects, scene.shape, bg)


def _random_connected_mask() -> np.ndarray:
    for _ in range(80):
        h = int(RNG.integers(1, 5))
        w = int(RNG.integers(1, 5))
        lo = min(max(2, h + w - 2), h * w)
        hi = min(h * w, 7)
        if lo > hi:
            continue
        target = int(RNG.integers(lo, hi + 1))
        cells = {(int(RNG.integers(h)), int(RNG.integers(w)))}
        frontier = list(cells)
        while len(cells) < target and frontier:
            rr, cc = frontier[int(RNG.integers(len(frontier)))]
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(RNG.integers(4))]
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
        coords = np.array(sorted(cells), dtype=int)
        shape = _shape_from_coords(coords)
        mask = shape["mask"]
        if mask.sum() >= 2:
            return mask
    raise RuntimeError("failed to sample connected mask")


def _can_stamp(occ: np.ndarray, top: int, left: int, mask: np.ndarray) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > occ.shape[0] or left + w > occ.shape[1]:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occ.shape[0], top + h + 1)
    c1 = min(occ.shape[1], left + w + 1)
    return not occ[r0:r1, c0:c1].any()


def _stamp(grid: np.ndarray, occ: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    block = grid[top : top + h, left : left + w]
    block[mask] = int(color)
    occ[top : top + h, left : left + w][mask] = True


def _sample_input() -> np.ndarray | None:
    n = int(RNG.integers(2, 6))
    colors = list(map(int, RNG.choice(COLOR_POOL, size=n, replace=False)))
    legend_colors = colors.copy()
    RNG.shuffle(legend_colors)

    masks = [_random_connected_mask() for _ in range(n)]
    objects = [{"mask": mask, "h": mask.shape[0], "w": mask.shape[1], "color": color} for mask, color in zip(masks, legend_colors)]

    pack_h = int(sum(obj["h"] for obj in objects) + (n - 1))
    pack_w = int(max(obj["w"] for obj in objects))
    scene_h = max(pack_h + int(RNG.integers(0, 5)), int(RNG.integers(8, 16)))
    scene_w = max(pack_w + int(RNG.integers(6, 13)), int(RNG.integers(10, 19)))

    try:
        _pack_objects(objects, (scene_h, scene_w), BG)
    except AssertionError:
        return None

    grid = np.full((scene_h, scene_w + 2), BG, dtype=int)
    scene = grid[:, :scene_w]
    occ = np.zeros_like(scene, dtype=bool)

    legend_rows = sorted(map(int, RNG.choice(np.arange(scene_h), size=n, replace=False)))
    for row, color in zip(legend_rows, legend_colors):
        grid[row, scene_w + 1] = int(color)

    order = list(range(n))
    RNG.shuffle(order)
    for idx in order:
        mask = objects[idx]["mask"]
        h, w = mask.shape
        placed = False
        for _ in range(120):
            top = int(RNG.integers(0, scene_h - h + 1))
            left = int(RNG.integers(0, scene_w - w + 1))
            if _can_stamp(occ, top, left, mask):
                _stamp(scene, occ, top, left, mask, objects[idx]["color"])
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != (scene_h, scene_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _sample_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
