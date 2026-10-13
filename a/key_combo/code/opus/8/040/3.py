from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, find_components, get_bg, in_bounds, neighbors4, neighbors8, paste


def _separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(1, w - 1):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != int(bg) and c == w - c - 1:
            candidates.append(int(c))
    assert len(candidates) == 1, f"expected one centered separator, got {candidates}"
    return candidates[0]


def _difference_count(ideal: np.ndarray, actual: np.ndarray, bg: int) -> int:
    diff = ((ideal != int(bg)) | (actual != int(bg))) & (ideal != actual)
    return int(diff.sum())


def _slide_left_objects(left: np.ndarray, distance: int, bg: int) -> np.ndarray:
    out = np.full_like(left, int(bg))
    objects = find_components(left, bg=bg, connectivity=4, monochromatic=True)
    assert objects, "expected at least one foreground object in the left panel"

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        patch = np.full((int(r1) - int(r0) + 1, int(c1) - int(c0) + 1), int(bg), dtype=int)
        coords = obj["coords"]
        patch[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = int(obj["color"])
        target_c = int(c0) + int(distance)
        h, w = patch.shape
        assert target_c + w <= left.shape[1], "shifted object crossed the separator"
        target = out[int(r0) : int(r0) + h, target_c : target_c + w]
        assert target.shape == patch.shape
        assert not np.any((target != int(bg)) & (patch != int(bg))), "shifted objects must not overlap"
        paste(out, patch, int(r0), target_c, transparent=int(bg))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the left panel, count its disagreement with the noisy right panel, and slide the left objects by that count.

    Input structure
    - The grid has a black background and a single full-height separator column between two equal-width panels.
    - The left panel contains 2-3 small single-color 4-connected objects.
    - The right panel contains one same-color counterpart per left object, close to the horizontal mirror of the left panel but with a few noisy pixels.

    Rule
    - Remove the separator conceptually and mirror the left panel horizontally to form the ideal right panel.
    - Count every right-panel cell where the ideal and actual panels differ and at least one side is non-background; call this count D.
    - Shift every left-panel foreground object rigidly D columns to the right, erase the old positions, and return only the left panel.

    Key insight
    - The right panel is only a noisy mirror-based measuring device: its mismatch count encodes the exact slide distance for the left objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    sep = _separator_col(g, bg)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape and left.shape[1] > 0

    ideal_right = np.fliplr(left)
    distance = _difference_count(ideal_right, right, bg)
    assert distance >= 0
    return _slide_left_objects(left, distance, bg)


# --------------- generator (o5x) ---------------

_GEN_HEIGHT = (9, 14)
_GEN_PANEL_WIDTH = (8, 11)
_GEN_OBJECT_COUNTS = (2, 3)
_GEN_OBJECT_AREAS = (3, 6)
_GEN_D_RANGE = (1, 4)
_GEN_MAX_ATTEMPTS = 400
_GEN_COLORS = [c for c in Color.ALL_COLORS if c not in {Color.BLACK, Color.GREY}]


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    arr = np.array([(r - r0, c - c0) for r, c in coords], dtype=int)
    return arr[np.lexsort((arr[:, 1], arr[:, 0]))]


def _random_object_mask(area: int) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in coords:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in coords:
                    frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    arr = _normalize_coords(coords)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True

    for _ in range(random.randint(0, 3)):
        mask = np.rot90(mask)
    if random.choice([False, True]):
        mask = np.fliplr(mask)
    return mask


def _can_place_mask(panel: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = mask.shape
    H, W = panel.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        for nr, nc in neighbors8(rr, cc):
            if in_bounds((H, W), nr, nc) and int(panel[nr, nc]) not in {Color.BLACK, int(color)}:
                return False
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        neighborhood = panel[r1:r2, c1:c2]
        if np.any(neighborhood != Color.BLACK):
            return False
    return True


def _place_left_objects(h: int, panel_w: int, distance: int, n_obj: int) -> np.ndarray | None:
    left = np.zeros((h, panel_w), dtype=int)
    usable_rightmost = panel_w - max(3, distance + 1)
    if usable_rightmost < 1:
        return None

    colors = random.sample(_GEN_COLORS, n_obj)
    for color in colors:
        placed = False
        for _ in range(80):
            area = random.randint(*_GEN_OBJECT_AREAS)
            mask = _random_object_mask(area)
            mh, mw = mask.shape
            max_c0 = usable_rightmost - mw
            max_r0 = h - mh
            if max_c0 < 0 or max_r0 < 0:
                continue
            r0 = random.randint(0, max_r0)
            c0 = random.randint(0, max_c0)
            if not _can_place_mask(left, mask, r0, c0, color):
                continue
            patch = np.zeros_like(mask, dtype=int)
            patch[mask] = int(color)
            paste(left, patch, r0, c0, transparent=Color.BLACK)
            placed = True
            break
        if not placed:
            return None

    comps = find_components(left, bg=Color.BLACK, connectivity=4, monochromatic=True)
    if len(comps) != n_obj:
        return None
    for comp in comps:
        area = int(comp["area"])
        if not (_GEN_OBJECT_AREAS[0] <= area <= _GEN_OBJECT_AREAS[1]):
            return None
        max_col = int(comp["coords"][:, 1].max())
        if panel_w - max_col <= distance:
            return None
    return left


def _single_component(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return len(comps) == 1


def _noise_candidates(actual: np.ndarray, ideal: np.ndarray, color: int) -> list[tuple[str, int, int]]:
    mask = actual == int(color)
    if not mask.any():
        return []

    candidates: list[tuple[str, int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if int(ideal[rr, cc]) != int(color):
            continue
        if int(mask.sum()) <= 3:
            break
        shrunk = mask.copy()
        shrunk[rr, cc] = False
        if shrunk.sum() >= 3 and _single_component(shrunk):
            candidates.append(("remove", rr, cc))

    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(actual.shape, nr, nc) and int(actual[nr, nc]) == Color.BLACK and int(ideal[nr, nc]) == Color.BLACK:
                frontier.add((nr, nc))
    for rr, cc in frontier:
        blocked = False
        for nr, nc in neighbors8(rr, cc):
            if in_bounds(actual.shape, nr, nc) and int(actual[nr, nc]) not in {Color.BLACK, int(color)}:
                blocked = True
                break
        if blocked:
            continue
        grown = mask.copy()
        grown[rr, cc] = True
        if grown.sum() <= 8 and _single_component(grown):
            candidates.append(("add", int(rr), int(cc)))
    return candidates


def _inject_noise(ideal: np.ndarray, distance: int) -> np.ndarray | None:
    actual = ideal.copy()
    colors = [int(c) for c in np.unique(ideal) if int(c) != Color.BLACK]
    edited_per_color = {color: 0 for color in colors}

    for _ in range(distance):
        options: list[tuple[int, str, int, int]] = []
        for color in colors:
            for kind, r, c in _noise_candidates(actual, ideal, color):
                if edited_per_color[color] >= 3:
                    continue
                options.append((color, kind, r, c))
        if not options:
            return None
        weights = [3 if edited_per_color[color] == 0 else 1 for color, _kind, _r, _c in options]
        color, kind, r, c = random.choices(options, weights=weights, k=1)[0]
        if kind == "remove":
            actual[r, c] = Color.BLACK
        else:
            actual[r, c] = int(color)
        edited_per_color[color] += 1

    if _difference_count(ideal, actual, Color.BLACK) != distance:
        return None
    for color in colors:
        mask = actual == int(color)
        if not _single_component(mask):
            return None
    return actual


def _valid_right_panel(right: np.ndarray, n_obj: int) -> bool:
    colors = [int(c) for c in np.unique(right) if int(c) != Color.BLACK]
    if len(colors) != n_obj:
        return False
    comps = find_components(right, bg=Color.BLACK, connectivity=4, monochromatic=True)
    if len(comps) != n_obj:
        return False
    for comp in comps:
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        for r, c in zip(rows, cols):
            rr = int(r)
            cc = int(c)
            for nr, nc in neighbors8(rr, cc):
                if not in_bounds(right.shape, nr, nc):
                    continue
                value = int(right[nr, nc])
                if value not in {Color.BLACK, int(comp["color"])}:
                    return False
    return True


def _expected_output(left: np.ndarray, distance: int) -> np.ndarray:
    return _slide_left_objects(left, distance, Color.BLACK)


def _gen_attempt() -> np.ndarray | None:
    distance = random.randint(*_GEN_D_RANGE)
    n_obj = random.randint(*_GEN_OBJECT_COUNTS)
    panel_w = random.randint(max(_GEN_PANEL_WIDTH[0], distance + 4), _GEN_PANEL_WIDTH[1])
    h = random.randint(*_GEN_HEIGHT)

    left = _place_left_objects(h, panel_w, distance, n_obj)
    if left is None:
        return None

    ideal_right = np.fliplr(left)
    right = _inject_noise(ideal_right, distance)
    if right is None or not _valid_right_panel(right, n_obj):
        return None

    inp = np.concatenate(
        [
            left,
            np.full((h, 1), Color.GREY, dtype=int),
            right,
        ],
        axis=1,
    )

    if _difference_count(np.fliplr(left), right, Color.BLACK) != distance:
        return None
    expected = _expected_output(left, distance)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-noise slide puzzle")
