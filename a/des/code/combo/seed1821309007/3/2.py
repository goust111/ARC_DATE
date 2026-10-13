import random
from collections import deque
from itertools import permutations

import numpy as np


PATTERN_KINDS = ("checker", "hstripe", "vstripe")
GENERATE_RETRIES = 400
PALETTE = tuple(range(10))


def _pattern_value(r: int, c: int, color0: int, color1: int, kind: str) -> int:
    if kind == "checker":
        return color0 if (r + c) % 2 == 0 else color1
    if kind == "hstripe":
        return color0 if r % 2 == 0 else color1
    if kind == "vstripe":
        return color0 if c % 2 == 0 else color1
    raise ValueError(f"unknown pattern kind: {kind}")


def _pattern_grid(shape: tuple[int, int], color0: int, color1: int, kind: str) -> np.ndarray:
    h, w = shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = _pattern_value(r, c, color0, color1, kind)
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _valid_candidate(
    grid: np.ndarray, bg0: int, bg1: int, kind: str
) -> tuple[np.ndarray, list[dict[str, int]]] | None:
    pattern = _pattern_grid(grid.shape, bg0, bg1, kind)
    mismatch = grid != pattern
    if not mismatch.any():
        return None

    visible = ~mismatch
    if not np.any(visible & (pattern == bg0)) or not np.any(visible & (pattern == bg1)):
        return None

    objects: list[dict[str, int]] = []
    for comp in _components(mismatch):
        r0, c0, r1, c1 = _bbox(comp)
        color = int(grid[comp[0]])
        if color in (bg0, bg1):
            return None
        for r, c in comp:
            if int(grid[r, c]) != color:
                return None
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return None
        if not np.all(mismatch[r0 : r1 + 1, c0 : c1 + 1]):
            return None
        objects.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": color})
    if not objects:
        return None
    return pattern, objects


def _find_solution(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    colors = sorted(set(int(v) for v in grid.ravel()))
    candidates: list[tuple[np.ndarray, list[dict[str, int]], tuple[int, int, str]]] = []
    for bg0, bg1 in permutations(colors, 2):
        for kind in PATTERN_KINDS:
            result = _valid_candidate(grid, int(bg0), int(bg1), kind)
            if result is None:
                continue
            pattern, objects = result
            candidates.append((pattern, objects, (int(bg0), int(bg1), kind)))

    unique: list[tuple[np.ndarray, list[dict[str, int]], tuple[int, int, str]]] = []
    seen = set()
    for pattern, objects, key in candidates:
        obj_key = tuple(sorted((o["r0"], o["c0"], o["r1"], o["c1"], o["color"]) for o in objects))
        full_key = (key, obj_key)
        if full_key not in seen:
            seen.add(full_key)
            unique.append((pattern, objects, key))

    assert len(unique) == 1, f"expected exactly one valid interpretation, found {len(unique)}"
    return unique[0][0], unique[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a hidden two-color periodic background, restore it under rectangular occluders,
    then shrink each occluder to a single center marker of its original color.
    """
    g = np.asarray(grid, dtype=int)
    pattern, objects = _find_solution(g)

    out = pattern.copy()
    for obj in objects:
        cr = (obj["r0"] + obj["r1"]) // 2
        cc = (obj["c0"] + obj["c1"]) // 2
        out[cr, cc] = obj["color"]
    return out


def _place_rectangles(
    h: int, w: int, rng: random.Random, count: int
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    padded = np.zeros((h, w), dtype=bool)
    for _ in range(count):
        placed = False
        for _attempt in range(80):
            rh = rng.randint(1, min(5, h - 2))
            rw = rng.randint(1, min(5, w - 2))
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h - 1, r1 + 1)
            pc1 = min(w - 1, c1 + 1)
            if padded[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue
            padded[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return rects


def _expected_output(
    shape: tuple[int, int],
    bg0: int,
    bg1: int,
    kind: str,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    out = _pattern_grid(shape, bg0, bg1, kind)
    for (r0, c0, r1, c1), color in zip(rects, colors):
        out[(r0 + r1) // 2, (c0 + c1) // 2] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(8, 15)
    w = rng.randint(8, 15)
    kind = rng.choice(PATTERN_KINDS)

    bg0, bg1 = rng.sample(PALETTE, 2)
    grid = _pattern_grid((h, w), bg0, bg1, kind)

    max_objects = min(4, len(PALETTE) - 2, (h * w) // 10)
    if max_objects < 1:
        return None
    object_count = rng.randint(1, max_objects)
    rects = _place_rectangles(h, w, rng, object_count)
    if rects is None:
        return None

    available_colors = [c for c in PALETTE if c not in (bg0, bg1)]
    obj_colors = rng.sample(available_colors, object_count)
    for (r0, c0, r1, c1), color in zip(rects, obj_colors):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    expected = _expected_output((h, w), bg0, bg1, kind, rects, obj_colors)

    try:
        actual = main(grid)
        pattern, objects = _find_solution(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(pattern, _pattern_grid((h, w), bg0, bg1, kind)):
        return None
    if len(objects) != object_count:
        return None

    # Reject underconstrained samples by requiring that the visible background
    # shows both parities of the chosen pattern after occlusion.
    mismatch = grid != pattern
    visible = ~mismatch
    if kind in ("checker", "hstripe"):
        if not np.any(visible[0::2]) or not np.any(visible[1::2]):
            return None
    if kind in ("checker", "vstripe"):
        if not np.any(visible[:, 0::2]) or not np.any(visible[:, 1::2]):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
