from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
FOREGROUND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = outer
    br0, bc0, br1, bc1 = inner
    return ar0 <= br0 and ac0 <= bc0 and ar1 >= br1 and ac1 >= bc1


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            box = _bbox(pts)
            comps.append(
                {
                    "idx": len(comps),
                    "color": color,
                    "points": pts,
                    "bbox": box,
                    "bbox_area": _bbox_area(box),
                }
            )
    return comps


def _hole_count_after_erasure(out: np.ndarray, obj: dict) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    crop = out[r0 : r1 + 1, c0 : c1 + 1]
    black = crop == BLACK
    h, w = black.shape

    obj_mask = np.zeros((h, w), dtype=bool)
    for r, c in obj["points"]:
        obj_mask[r - r0, c - c0] = True

    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for sr in range(h):
        for sc in range(w):
            if not black[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if touches_edge:
                continue

            enclosed_by_object = True
            for rr, cc in pts:
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not black[nr, nc] and not obj_mask[nr, nc]:
                        enclosed_by_object = False
                        break
                if not enclosed_by_object:
                    break
            if enclosed_by_object:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase objects whose bounding boxes sit inside larger differently colored bounding boxes, then recolor odd-hole containers.

    Input structure:
    - Black (0) is the background.
    - Foreground pixels form one or more 8-connected monochromatic objects per color.
    - Some larger objects have U-, C-, or window-like bounding boxes that contain smaller objects of other colors.
    - Larger objects may also enclose zero or more Black holes.

    Rule:
    - Find every 8-connected monochromatic non-Black object.
    - If object B's bounding box is fully contained in a larger bounding box of object A with a different color, erase all pixels of B to Black.
    - For each non-erased object A that contained at least one erased object, count the remaining enclosed Black regions inside A after erasure.
    - If that count is odd, recolor A to Grey (5); if it is even, keep A's original color.
    - Objects that do not contain any erased object remain unchanged.

    Key insight: Bounding-box containment chooses which object is removed, but the post-removal hole parity of the container controls whether the container itself changes color.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components(g)

    contained_by: dict[int, list[int]] = {obj["idx"]: [] for obj in comps}
    contains: dict[int, list[int]] = {obj["idx"]: [] for obj in comps}
    for outer in comps:
        for inner in comps:
            if outer["idx"] == inner["idx"] or outer["color"] == inner["color"]:
                continue
            if outer["bbox_area"] <= inner["bbox_area"]:
                continue
            if _bbox_contains(outer["bbox"], inner["bbox"]):
                contained_by[inner["idx"]].append(outer["idx"])
                contains[outer["idx"]].append(inner["idx"])

    erased = {idx for idx, outers in contained_by.items() if outers}

    out = g.copy()
    for obj in comps:
        if obj["idx"] not in erased:
            continue
        for r, c in obj["points"]:
            out[r, c] = BLACK

    recolor_to_grey: list[dict] = []
    for obj in comps:
        idx = obj["idx"]
        if idx in erased or not contains[idx]:
            continue
        holes = _hole_count_after_erasure(out, obj)
        if holes % 2 == 1:
            recolor_to_grey.append(obj)

    for obj in recolor_to_grey:
        for r, c in obj["points"]:
            out[r, c] = GREY

    return out


def _closed_patch(hole_count: int, container_color: int, child_color: int, rng: random.Random) -> np.ndarray:
    height = rng.choice([6, 7, 8])
    extra_left = rng.randint(0, 1)
    extra_right = rng.randint(0, 1)
    width = 3 * hole_count + 3 + extra_left + extra_right
    patch = np.full((height, width), container_color, dtype=int)

    hole_r = rng.randint(2, height - 4)
    hole_cols = [2 + extra_left + 3 * i for i in range(hole_count)]
    child_index = rng.randrange(hole_count)
    for i, hole_c in enumerate(hole_cols):
        patch[hole_r : hole_r + 2, hole_c : hole_c + 2] = BLACK
        if i == child_index:
            patch[hole_r : hole_r + 2, hole_c : hole_c + 2] = child_color

    return np.rot90(patch, rng.randrange(4)).copy()


def _open_patch(container_color: int, child_color: int, rng: random.Random) -> np.ndarray:
    height = rng.randint(7, 9)
    width = rng.randint(7, 9)
    thickness = rng.choice([1, 2])
    patch = np.zeros((height, width), dtype=int)

    patch[:, :thickness] = container_color
    patch[:thickness, :] = container_color
    patch[height - thickness :, :] = container_color

    child_h = rng.choice([1, 2])
    child_w = rng.choice([1, 2])
    r_min = thickness
    r_max = height - thickness - child_h
    c_min = thickness
    c_max = width - child_w - 2
    if r_min > r_max or c_min > c_max:
        return _open_patch(container_color, child_color, rng)

    child_r = rng.randint(r_min, r_max)
    child_c = rng.randint(c_min, c_max)
    patch[child_r : child_r + child_h, child_c : child_c + child_w] = child_color

    return np.rot90(patch, rng.randrange(4)).copy()


def _random_sprite(color: int, rng: random.Random) -> np.ndarray:
    target = rng.randint(3, 6)
    cells = {(0, 0)}
    while len(cells) < target:
        r, c = rng.choice(list(cells))
        nr, nc = rng.choice(list(_neighbors4(r, c)))
        cells.add((nr, nc))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in norm:
        patch[r, c] = color
    return patch


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, rng: random.Random) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return None

    candidates = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
    rng.shuffle(candidates)
    for r, c in candidates:
        if np.any(reserved[r - 1 : r + ph + 1, c - 1 : c + pw + 1]):
            continue
        mask = patch != BLACK
        grid[r : r + ph, c : c + pw][mask] = patch[mask]
        reserved[r - 1 : r + ph + 1, c - 1 : c + pw + 1] = True
        return r, c
    return None


def _build_container_specs(rng: random.Random) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    specs.append(("closed", rng.choice([1, 3])))
    if rng.random() < 0.5:
        specs.append(("open", 0))
    else:
        specs.append(("closed", 2))

    if rng.random() < 0.55:
        kind = rng.choice(["open", "closed"])
        if kind == "open":
            specs.append(("open", 0))
        else:
            specs.append(("closed", rng.choice([1, 2, 3])))
    rng.shuffle(specs)
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(20, 30)
    width = rng.randint(20, 30)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    specs = _build_container_specs(rng)
    distractor_count = rng.randint(1, 2)
    need_colors = 2 * len(specs) + distractor_count
    if need_colors > len(FOREGROUND_COLORS):
        return None

    colors = FOREGROUND_COLORS[:]
    rng.shuffle(colors)

    for kind, hole_count in specs:
        container_color = colors.pop()
        child_color = colors.pop()
        if kind == "open":
            patch = _open_patch(container_color, child_color, rng)
        else:
            patch = _closed_patch(hole_count, container_color, child_color, rng)
        if _place_patch(grid, reserved, patch, rng) is None:
            return None

    for _ in range(distractor_count):
        patch = _random_sprite(colors.pop(), rng)
        if _place_patch(grid, reserved, patch, rng) is None:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.any(out == GREY):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid containment-and-hole-parity input")
