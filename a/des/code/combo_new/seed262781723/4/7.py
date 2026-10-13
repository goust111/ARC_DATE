from __future__ import annotations

from collections import Counter
import random

import numpy as np


_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant background"
    return int(ranked[0][0])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _connected_components(grid == int(color))


def _normalize(coords: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    assert coords, "expected at least one coordinate"
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return frozenset((r - min_r, c - min_c) for r, c in coords)


def _candidate_lock(
    grid: np.ndarray,
    bg: int,
    color: int,
    coords: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    if not np.all((patch == int(bg)) | (patch == int(color))):
        return None

    local_missing = [(int(r), int(c)) for r, c in np.argwhere(patch == int(bg))]
    if not local_missing:
        return None

    missing_mask = np.zeros(patch.shape, dtype=bool)
    for r, c in local_missing:
        missing_mask[r, c] = True
    if len(_connected_components(missing_mask)) != 1:
        return None

    h, w = patch.shape
    touches = 0
    touches += int(any(r == 0 for r, _ in local_missing))
    touches += int(any(r == h - 1 for r, _ in local_missing))
    touches += int(any(c == 0 for _, c in local_missing))
    touches += int(any(c == w - 1 for _, c in local_missing))
    if touches != 1:
        return None

    missing_abs = [(r0 + r, c0 + c) for r, c in local_missing]
    return missing_abs, _normalize(local_missing)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock each notched monochrome rectangle by moving its matching same-color key into the missing bite.

    Input structure:
    - The background is black.
    - Each non-background color appears in exactly two 4-connected components: one large lock and one smaller key.
    - A lock is a filled rectangle of one color with one connected notch carved from exactly one side.
    - The key is a separate same-color connected region whose shape matches that notch without rotation or reflection.

    Rule:
    - For each color, identify which component is the lock by checking which one can be completed to a solid rectangle by filling a single edge-touching missing region.
    - Remove the separate key from its original location.
    - Translate that key shape into the lock's missing region so the notch is filled exactly.
    - Leave every other pixel unchanged.

    Key insight:
    The lock is recognized by rectangle completion: its bounding box is almost solid except for one connected edge indentation, and the other same-color component is the exact translated complement of that indentation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _dominant_color(g)
    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one lock/key color"

    for color in colors:
        comps = _color_components(g, color)
        assert len(comps) == 2, "each color must have exactly one lock and one key"

        matches: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
        cand0 = _candidate_lock(g, bg, color, comps[0])
        cand1 = _candidate_lock(g, bg, color, comps[1])

        if cand0 is not None and cand0[1] == _normalize(comps[1]):
            matches.append((cand0[0], comps[1]))
        if cand1 is not None and cand1[1] == _normalize(comps[0]):
            matches.append((cand1[0], comps[0]))

        assert len(matches) == 1, "each color must admit a unique lock/key pairing"
        missing_abs, key_coords = matches[0]

        for r, c in key_coords:
            out[r, c] = bg
        for r, c in missing_abs:
            out[r, c] = color

    return out


_GEN_BG = 0
_GEN_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
_GEN_SIDES = ("top", "bottom", "left", "right")
_GEN_COUNT_RANGE = (1, 4)
_GEN_MIN_SIDE = {1: 12, 2: 15, 3: 19, 4: 23}
_GEN_MAX_SIDE = 30
_GEN_MAX_RECT_H = {1: 9, 2: 8, 3: 7, 4: 7}
_GEN_MAX_RECT_W = {1: 10, 2: 9, 3: 8, 4: 7}
_GEN_PADDING = 1
_GEN_MAX_ATTEMPTS = 400


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != _GEN_BG
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    for dr, dc in np.argwhere(patch != _GEN_BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(grid[r1:r2, c1:c2] != _GEN_BG):
            return False
    return True


def _all_positions(grid_shape: tuple[int, int], patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = grid_shape
    h, w = patch_shape
    return [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]


def _sample_notch(rect_h: int, rect_w: int, side: str) -> frozenset[tuple[int, int]] | None:
    for _ in range(100):
        if side in ("top", "bottom"):
            if rect_w < 3 or rect_h < 3:
                return None
            start = random.randint(1, rect_w - 2)
            end = random.randint(start, rect_w - 2)
            span = list(range(start, end + 1))
            max_depth = min(rect_h - 2, 4)
            depths = [random.randint(1, max_depth) for _ in span]
            cells = []
            for col, depth in zip(span, depths):
                for d in range(depth):
                    row = d if side == "top" else rect_h - 1 - d
                    cells.append((row, col))
        else:
            if rect_h < 3 or rect_w < 3:
                return None
            start = random.randint(1, rect_h - 2)
            end = random.randint(start, rect_h - 2)
            span = list(range(start, end + 1))
            max_depth = min(rect_w - 2, 4)
            depths = [random.randint(1, max_depth) for _ in span]
            cells = []
            for row, depth in zip(span, depths):
                for d in range(depth):
                    col = d if side == "left" else rect_w - 1 - d
                    cells.append((row, col))

        notch = frozenset(cells)
        area = len(notch)
        rect_area = rect_h * rect_w
        if 2 <= area <= max(2, rect_area // 3):
            return notch
    return None


def _render_key_patch(color: int, coords: frozenset[tuple[int, int]]) -> np.ndarray:
    norm = list(_normalize(list(coords)))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.full((h, w), _GEN_BG, dtype=int)
    for r, c in norm:
        patch[r, c] = color
    return patch


def _build_object(color: int, pair_count: int) -> dict | None:
    max_h = _GEN_MAX_RECT_H[pair_count]
    max_w = _GEN_MAX_RECT_W[pair_count]

    for _ in range(100):
        rect_h = random.randint(4, max_h)
        rect_w = random.randint(4, max_w)
        side = random.choice(_GEN_SIDES)
        notch = _sample_notch(rect_h, rect_w, side)
        if notch is None:
            continue

        rect_patch = np.full((rect_h, rect_w), color, dtype=int)
        key_patch = _render_key_patch(color, notch)
        return {
            "color": color,
            "side": side,
            "rect_h": rect_h,
            "rect_w": rect_w,
            "rect_area": rect_h * rect_w,
            "notch": notch,
            "rect_patch": rect_patch,
            "key_patch": key_patch,
        }
    return None


def _place_rectangles(expected: np.ndarray, objects: list[dict]) -> bool:
    order = objects[:]
    random.shuffle(order)
    for obj in order:
        positions = _all_positions(expected.shape, obj["rect_patch"].shape)
        random.shuffle(positions)
        for r0, c0 in positions:
            if _can_place_patch(expected, obj["rect_patch"], r0, c0, _GEN_PADDING):
                _paste_patch(expected, obj["rect_patch"], r0, c0)
                obj["lock_pos"] = (r0, c0)
                break
        else:
            return False
    return True


def _carve_locks(expected: np.ndarray, objects: list[dict]) -> np.ndarray:
    inp = expected.copy()
    for obj in objects:
        r0, c0 = obj["lock_pos"]
        for dr, dc in obj["notch"]:
            inp[r0 + dr, c0 + dc] = _GEN_BG
    return inp


def _place_keys(inp: np.ndarray, objects: list[dict]) -> bool:
    order = objects[:]
    random.shuffle(order)
    for obj in order:
        positions = _all_positions(inp.shape, obj["key_patch"].shape)
        random.shuffle(positions)
        for r0, c0 in positions:
            if _can_place_patch(inp, obj["key_patch"], r0, c0, _GEN_PADDING):
                _paste_patch(inp, obj["key_patch"], r0, c0)
                obj["key_pos"] = (r0, c0)
                break
        else:
            return False
    return True


def _valid_generated_grid(inp: np.ndarray, expected: np.ndarray, objects: list[dict]) -> bool:
    counts = Counter(map(int, inp.reshape(-1)))
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][0] != _GEN_BG:
        return False
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return False

    for obj in objects:
        comps = _color_components(inp, obj["color"])
        if len(comps) != 2:
            return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*_GEN_COUNT_RANGE)
    H = random.randint(_GEN_MIN_SIDE[pair_count], _GEN_MAX_SIDE)
    W = random.randint(_GEN_MIN_SIDE[pair_count], _GEN_MAX_SIDE)
    colors = random.sample(_GEN_COLORS, pair_count)

    objects = []
    for color in colors:
        obj = _build_object(color, pair_count)
        if obj is None:
            return None
        objects.append(obj)

    total_area = sum(obj["rect_area"] for obj in objects)
    if total_area * 2 >= H * W:
        return None

    expected = np.full((H, W), _GEN_BG, dtype=int)
    if not _place_rectangles(expected, objects):
        return None

    inp = _carve_locks(expected, objects)
    if not _place_keys(inp, objects):
        return None

    if not _valid_generated_grid(inp, expected, objects):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
