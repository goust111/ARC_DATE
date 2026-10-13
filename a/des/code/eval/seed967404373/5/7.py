import random
from collections import deque

import numpy as np


BG = 0
MAX_LAYERS = 4
GEN_RETRIES = 500


def _components(mask: np.ndarray):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        yield cells


def _ring_from_component(mask: np.ndarray, cells: list[tuple[int, int]]):
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    size = r1 - r0 + 1
    if size != c1 - c0 + 1 or size < 3 or size % 2 == 0:
        return None
    if (size - 3) % 4 != 0:
        return None
    expected = 4 * size - 4
    if len(cells) != expected:
        return None
    local = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        local[r - r0, c - c0] = True
    if not (
        np.all(local[0, :])
        and np.all(local[-1, :])
        and np.all(local[:, 0])
        and np.all(local[:, -1])
        and not np.any(local[1:-1, 1:-1])
    ):
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "size": size,
        "center": (r0 + r1, c0 + c1),
    }


def _find_objects(grid: np.ndarray):
    work = np.asarray(grid, dtype=int)
    if work.shape[0] < 4:
        raise AssertionError("grid too short")
    area = work[:-1, :]
    rings_by_key: dict[tuple[int, int, int], list[dict]] = {}
    for color in sorted(int(c) for c in np.unique(area) if int(c) != BG):
        for cells in _components(area == color):
            ring = _ring_from_component(area == color, cells)
            if ring is None:
                continue
            key = (color, ring["center"][0], ring["center"][1])
            ring["color"] = color
            rings_by_key.setdefault(key, []).append(ring)

    objects = []
    for (color, _cr, _cc), rings in rings_by_key.items():
        rings.sort(key=lambda x: x["size"])
        sizes = [r["size"] for r in rings]
        expected = [3 + 4 * i for i in range(len(rings))]
        if sizes != expected or not (1 <= len(rings) <= MAX_LAYERS):
            continue
        outer = rings[-1]
        r0, c0, r1, c1 = outer["bbox"]
        patch = work[:-1, :][r0 : r1 + 1, c0 : c1 + 1].copy()
        objects.append(
            {
                "layers": len(rings),
                "bbox": outer["bbox"],
                "patch": patch,
                "color": color,
            }
        )
    if not objects:
        raise AssertionError("no nested square objects found")
    return objects


def _legend(grid: np.ndarray) -> list[int]:
    row = [int(x) for x in np.asarray(grid, dtype=int)[-1, :]]
    legend = [x for x in row if x != BG]
    if len(legend) < MAX_LAYERS:
        raise AssertionError("bottom legend must contain at least four colors")
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend = _legend(g)
    objects = _find_objects(g)
    rendered = []
    for obj in objects:
        patch = obj["patch"].copy()
        fill = legend[obj["layers"] - 1]
        center = patch.shape[0] // 2
        patch[center, center] = fill
        rendered.append((obj["layers"], obj["bbox"][1], obj["bbox"][0], patch))

    rendered.sort(key=lambda item: (item[0], item[1], item[2]))
    max_h = max(patch.shape[0] for *_head, patch in rendered)
    total_w = sum(patch.shape[1] for *_head, patch in rendered) + len(rendered) - 1
    out = np.full((max_h, total_w), BG, dtype=int)
    col = 0
    for _layers, _c0, _r0, patch in rendered:
        row = max_h - patch.shape[0]
        out[row : row + patch.shape[0], col : col + patch.shape[1]] = patch
        col += patch.shape[1] + 1
    return out


def _draw_nested(grid: np.ndarray, top: int, left: int, layers: int, color: int) -> None:
    size = 3 + 4 * (layers - 1)
    for layer in range(layers):
        inset = 2 * layer
        r0 = top + inset
        c0 = left + inset
        r1 = top + size - 1 - inset
        c1 = left + size - 1 - inset
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color


def _bbox_free(used: np.ndarray, r: int, c: int, size: int) -> bool:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(used.shape[0], r + size + 1)
    c1 = min(used.shape[1], c + size + 1)
    return not bool(np.any(used[r0:r1, c0:c1]))


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(4, 7)
    layer_values = [1, 2, 3, 4]
    while len(layer_values) < count:
        layer_values.append(random.randint(1, 4))
    random.shuffle(layer_values)

    max_size = max(3 + 4 * (k - 1) for k in layer_values)
    h = random.randint(max_size + 5, max_size + 16)
    w = random.randint(26, 40)
    if h < max_size + 3:
        return None
    grid = np.full((h, w), BG, dtype=int)
    used = np.zeros((h - 1, w), dtype=bool)

    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], count)
    legend = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], MAX_LAYERS)
    offset = random.randint(0, max(0, w - MAX_LAYERS))
    grid[-1, offset : offset + MAX_LAYERS] = legend

    placed = []
    for layers, color in zip(layer_values, colors):
        size = 3 + 4 * (layers - 1)
        ok = False
        for _ in range(100):
            r = random.randint(0, h - 2 - size)
            c = random.randint(0, w - size)
            if _bbox_free(used, r, c, size):
                _draw_nested(grid, r, c, layers, color)
                r0 = max(0, r - 1)
                c0 = max(0, c - 1)
                r1 = min(used.shape[0], r + size + 1)
                c1 = min(used.shape[1], c + size + 1)
                used[r0:r1, c0:c1] = True
                placed.append((layers, c, r))
                ok = True
                break
        if not ok:
            return None

    if len(placed) != count or len({k for k, _c, _r in placed}) < 3:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape[0] != max_size:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
