from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BLACK = 0
ORANGE = 7
MAROON = 9

_DIRECTIONS = ("up", "right", "down", "left")
_MISSINGS = ("ul", "ur", "lr", "ll")
_TARGET_MISSING = {"up": "ul", "right": "ur", "down": "lr", "left": "ll"}
_MISSING_COORD = {"ul": (0, 0), "ur": (0, 2), "lr": (2, 2), "ll": (2, 0)}
_GEN_COLORS = [1, 2, 3, 4, 5, 6, 8]


def _l_mask(missing: str) -> np.ndarray:
    mask = np.ones((3, 3), dtype=bool)
    r, c = _MISSING_COORD[missing]
    mask[r, c] = False
    return mask


def _l_patch(color: int, missing: str) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    patch[_l_mask(missing)] = int(color)
    return patch


def _missing_corner(mask: np.ndarray) -> str | None:
    if tuple(mask.shape) != (3, 3) or int(mask.sum()) != 8:
        return None
    holes = np.argwhere(~mask)
    if int(holes.shape[0]) != 1:
        return None
    hole = (int(holes[0, 0]), int(holes[0, 1]))
    for name, coord in _MISSING_COORD.items():
        if hole == coord:
            return name
    return None


def _arrow_direction(g: np.ndarray) -> str:
    H, W = map(int, g.shape)
    pts = {(int(r), int(c)) for r, c in np.argwhere(g == MAROON)}
    assert len(pts) == 3, "expected exactly one 3-pixel maroon arrow"

    matches: list[str] = []
    for c in range(1, W - 1):
        if pts == {(0, c - 1), (0, c + 1), (1, c)}:
            matches.append("down")
        if pts == {(H - 1, c - 1), (H - 1, c + 1), (H - 2, c)}:
            matches.append("up")
    for r in range(1, H - 1):
        if pts == {(r - 1, 0), (r + 1, 0), (r, 1)}:
            matches.append("right")
        if pts == {(r - 1, W - 1), (r + 1, W - 1), (r, W - 2)}:
            matches.append("left")

    assert len(matches) == 1, "maroon pixels must form one inward edge arrow"
    return matches[0]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    H, W = map(int, mask.shape)
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in nbrs:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_l_shapes(g: np.ndarray) -> list[dict]:
    foreground = (g != BLACK) & (g != ORANGE) & (g != MAROON)
    shapes = []
    for coords in _components(foreground, connectivity=8):
        assert len(coords) == 8, "each movable shape must have eight pixels"
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each shape must have a 3x3 bbox"

        color_values = {int(g[r, c]) for r, c in coords}
        assert len(color_values) == 1, "each shape must be monochrome"
        color = color_values.pop()

        mask = np.zeros((3, 3), dtype=bool)
        for r, c in coords:
            mask[int(r - r0), int(c - c0)] = True
        missing = _missing_corner(mask)
        assert missing is not None, "each shape must be a 3x3 square missing one corner"

        patch = np.zeros((3, 3), dtype=int)
        patch[mask] = int(color)
        shapes.append({"r0": int(r0), "c0": int(c0), "patch": patch, "missing": missing})

    shapes.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    return shapes


def _rotate_to_missing(patch: np.ndarray, target_missing: str) -> np.ndarray:
    mask = patch != BLACK
    for k in range(4):
        rotated_mask = np.rot90(mask, int(k))
        if _missing_corner(rotated_mask) == target_missing:
            return np.rot90(patch, int(k))
    raise AssertionError("no rotation makes the requested missing corner")


def _paste_checked(out: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    H, W = map(int, out.shape)
    h, w = map(int, patch.shape)
    assert 0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W, "placed shape must fit"
    mask = patch != BLACK
    target = out[int(r0) : int(r0 + h), int(c0) : int(c0 + w)]
    assert not bool(np.any((target != BLACK) & mask)), "placed shapes must not overlap"
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate missing-corner 3x3 shapes to the arrow direction and move them to ordered anchor points.

    Input structure:
    - Background is black.
    - One maroon 3-pixel arrow sits on an outer edge and points inward.
    - Several interior monochrome 3x3 shapes each have exactly one corner cell missing.
    - The same number of orange single-pixel anchors are arranged in row-major target positions.

    Rule:
    - Read the maroon arrow direction and map it to the target missing corner:
      up -> upper-left, right -> upper-right, down -> lower-right, left -> lower-left.
    - Sort the original 3x3 shapes by row then column, and sort orange anchors the same way.
    - Rotate each shape by 0, 90, 180, or 270 degrees so its missing corner becomes the target corner.
    - Move each rotated shape to the paired anchor with the 3x3 bbox centered on that anchor.
    - Remove the maroon arrow, all orange anchors, and all original shape locations.

    Key insight:
    The edge arrow defines a global target notch orientation, while source shapes and anchors pair only by spatial row-major order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)

    target_missing = _TARGET_MISSING[_arrow_direction(g)]
    shapes = _extract_l_shapes(g)
    anchors = [(int(r), int(c)) for r, c in np.argwhere(g == ORANGE)]
    anchors.sort()
    assert len(shapes) == len(anchors) and shapes, "shape and anchor counts must match"

    out = np.zeros((H, W), dtype=int)
    for shape, (ar, ac) in zip(shapes, anchors):
        rotated = _rotate_to_missing(shape["patch"], target_missing)
        _paste_checked(out, rotated, int(ar - 1), int(ac - 1))
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 600
GEN_OBJECT_COUNT_RANGE = (2, 6)
GEN_GRID_SIZE_RANGE = (15, 30)
GEN_ANCHOR_GAP = 5
GEN_OBJECT_PAD = 1


def _arrow_coords(direction: str, H: int, W: int) -> list[tuple[int, int]]:
    if direction == "down":
        c = random.randint(2, W - 3)
        return [(0, c - 1), (0, c + 1), (1, c)]
    if direction == "up":
        c = random.randint(2, W - 3)
        return [(H - 1, c - 1), (H - 1, c + 1), (H - 2, c)]
    if direction == "right":
        r = random.randint(2, H - 3)
        return [(r - 1, 0), (r + 1, 0), (r, 1)]
    r = random.randint(2, H - 3)
    return [(r - 1, W - 1), (r + 1, W - 1), (r, W - 2)]


def _reserve_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> None:
    H, W = map(int, occupied.shape)
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(H, int(r0 + h + pad))
    cc1 = min(W, int(c0 + w + pad))
    occupied[rr0:rr1, cc0:cc1] = True


def _reserve_cells(occupied: np.ndarray, coords: list[tuple[int, int]], pad: int = 0) -> None:
    for r, c in coords:
        _reserve_rect(occupied, int(r), int(c), 1, 1, pad=pad)


def _rect_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = map(int, occupied.shape)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    return not bool(occupied[int(r0) : int(r0 + h), int(c0) : int(c0 + w)].any())


def _spaced_values(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(int(lo), int(hi) + 1))
    if len(values) < count:
        return None
    for _ in range(80):
        picked = sorted(random.sample(values, int(count)))
        if all(int(b - a) >= int(min_gap) for a, b in zip(picked, picked[1:])):
            return picked
    choices = [
        list(combo)
        for combo in itertools.combinations(values, int(count))
        if all(int(b - a) >= int(min_gap) for a, b in zip(combo, combo[1:]))
    ]
    if not choices:
        return None
    return random.choice(choices)


def _anchor_layout_shape(n: int) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for rows in range(1, min(3, int(n)) + 1):
        cols = int((n + rows - 1) // rows)
        if cols <= 4:
            choices.append((rows, cols))
    if int(n) >= 4:
        choices = [x for x in choices if x[0] > 1] or choices
    return random.choice(choices)


def _place_anchors(occupied: np.ndarray, n: int) -> list[tuple[int, int]] | None:
    H, W = map(int, occupied.shape)
    layouts = [_anchor_layout_shape(int(n)) for _ in range(8)]
    random.shuffle(layouts)
    for rows_count, cols_count in layouts:
        rows = _spaced_values(2, H - 3, int(rows_count), GEN_ANCHOR_GAP)
        cols = _spaced_values(2, W - 3, int(cols_count), GEN_ANCHOR_GAP)
        if rows is None or cols is None:
            continue
        positions = [(r, c) for r in rows for c in cols][: int(n)]
        if len(positions) != int(n):
            continue
        if all(_rect_free(occupied, int(r - 1), int(c - 1), 3, 3) for r, c in positions):
            for r, c in positions:
                _reserve_rect(occupied, int(r - 1), int(c - 1), 3, 3, pad=GEN_OBJECT_PAD)
            return positions
    return None


def _place_l_shape(grid: np.ndarray, occupied: np.ndarray, color: int, missing: str) -> bool:
    H, W = map(int, grid.shape)
    candidates = [
        (r, c)
        for r in range(1, H - 3)
        for c in range(1, W - 3)
        if _rect_free(occupied, int(r), int(c), 3, 3)
    ]
    random.shuffle(candidates)
    if not candidates:
        return False

    r0, c0 = candidates[0]
    patch = _l_patch(int(color), missing)
    mask = patch != BLACK
    grid[int(r0) : int(r0) + 3, int(c0) : int(c0) + 3][mask] = patch[mask]
    _reserve_rect(occupied, int(r0), int(c0), 3, 3, pad=GEN_OBJECT_PAD)
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_GRID_SIZE_RANGE)
    W = random.randint(*GEN_GRID_SIZE_RANGE)
    n = random.randint(*GEN_OBJECT_COUNT_RANGE)
    direction = random.choice(list(_DIRECTIONS))

    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    arrow = _arrow_coords(direction, H, W)
    for r, c in arrow:
        grid[int(r), int(c)] = MAROON
    _reserve_cells(occupied, arrow, pad=GEN_OBJECT_PAD)

    anchors = _place_anchors(occupied, n)
    if anchors is None:
        return None

    missings = [random.choice(list(_MISSINGS)) for _ in range(n)]
    if n >= 4 and random.random() < 0.65:
        first = random.sample(list(_MISSINGS), 4)
        missings[:4] = first
        random.shuffle(missings)
    target = _TARGET_MISSING[direction]
    if all(m == target for m in missings):
        alternatives = [m for m in _MISSINGS if m != target]
        missings[random.randrange(n)] = random.choice(alternatives)

    for missing in missings:
        color = random.choice(_GEN_COLORS)
        if not _place_l_shape(grid, occupied, color, missing):
            return None

    for r, c in anchors:
        if int(grid[int(r), int(c)]) != BLACK:
            return None
        grid[int(r), int(c)] = ORANGE

    try:
        out = main(grid)
    except Exception:
        return None
    if tuple(out.shape) != (H, W) or max(out.shape) > 30:
        return None
    if int(np.count_nonzero(out)) != int(8 * n):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1155420539/5/4 input")
