import random
from collections import Counter

import numpy as np


BG = 7
PALETTE = [0, 1, 2, 3, 4, 5, 6, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
PLACE_RETRIES = 80


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    for comp in _components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if bbox_area > comp["area"]:
            out[r0 : r1 + 1, c0 : c1 + 1] = int(comp["color"])
    return out


def _solid_rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _hollow_rect(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _notched_rect(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        width = rng.randint(1, max(1, w - 1))
        start = rng.randint(0, w - width)
        row = 0 if side == "top" else h - 1
        mask[row, start : start + width] = False
    else:
        height = rng.randint(1, max(1, h - 1))
        start = rng.randint(0, h - height)
        col = 0 if side == "left" else w - 1
        mask[start : start + height, col] = False
    return mask


def _plus_shape(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    return mask


def _l_shape(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _sample_nonsolid_shape(rng: random.Random) -> np.ndarray:
    kind = rng.choice(["hollow", "notch", "plus", "l"])
    if kind == "hollow":
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        return _hollow_rect(h, w)
    if kind == "notch":
        h = rng.randint(2, 6)
        w = rng.randint(2, 6)
        return _notched_rect(h, w, rng)
    if kind == "plus":
        h = rng.choice([3, 5])
        w = rng.choice([3, 5])
        return _plus_shape(h, w)
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)
    return _l_shape(h, w)


def _sample_solid_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(2, 6)
    return _solid_rect(h, w)


def _mask_is_4_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_reserved(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + h + 1)
    cc1 = min(reserved.shape[1], c0 + w + 1)
    return bool(reserved[rr0:rr1, cc0:cc1].any())


def _place_shape(grid: np.ndarray, reserved: np.ndarray, color: int, mask: np.ndarray, rng: random.Random) -> bool:
    gh, gw = grid.shape
    h, w = mask.shape
    if h > gh or w > gw:
        return False
    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(0, gh - h)
        c0 = rng.randint(0, gw - w)
        if _touches_reserved(reserved, r0, c0, h, w):
            continue
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any(sub[mask] != BG):
            continue
        sub[mask] = color
        reserved[r0 : r0 + h, c0 : c0 + w] = True
        return True
    return False


def _classify_components(grid: np.ndarray) -> tuple[int, int]:
    solid = 0
    nonsolid = 0
    for comp in _components(grid, BG):
        r0, c0, r1, c1 = comp["bbox"]
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if bbox_area == comp["area"]:
            solid += 1
        else:
            nonsolid += 1
    return solid, nonsolid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 22)
    w = rng.randint(12, 22)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    num_objects = rng.randint(3, min(6, len(PALETTE)))
    colors = rng.sample(PALETTE, num_objects)

    shapes = [_sample_solid_shape(rng), _sample_nonsolid_shape(rng)]
    while len(shapes) < num_objects:
        if rng.random() < 0.5:
            shapes.append(_sample_solid_shape(rng))
        else:
            shapes.append(_sample_nonsolid_shape(rng))
    rng.shuffle(shapes)

    placed_nonsolid = 0
    placed_solid = 0
    for color, mask in zip(colors, shapes):
        if not _mask_is_4_connected(mask):
            return None
        if not _place_shape(grid, reserved, color, mask, rng):
            return None
        bbox_area = mask.shape[0] * mask.shape[1]
        if bbox_area == int(mask.sum()):
            placed_solid += 1
        else:
            placed_nonsolid += 1

    solid, nonsolid = _classify_components(grid)
    if solid != placed_solid or nonsolid != placed_nonsolid:
        return None
    if solid < 1 or nonsolid < 1:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
