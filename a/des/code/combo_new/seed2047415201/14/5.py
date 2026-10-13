from __future__ import annotations

import random
from collections import deque

import numpy as np


AZURE = 8
TRIANGLE_SIDE = 4
DIRS = ("TL", "TR", "BL", "BR")

L_CELLS = {
    "TL": ((0, 1), (1, 0), (1, 1)),
    "TR": ((0, 0), (1, 0), (1, 1)),
    "BL": ((0, 0), (0, 1), (1, 1)),
    "BR": ((0, 0), (0, 1), (1, 0)),
}


def _bg_color(g: np.ndarray) -> int:
    if np.any(g == AZURE):
        return AZURE
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _same_color_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _marker_from_component(comp: dict) -> dict | None:
    coords = set(comp["coords"])
    if len(coords) != 3:
        return None

    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None

    full = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    missing = list(full - coords)
    if len(missing) != 1:
        return None

    mr, mc = missing[0]
    if (mr, mc) == (r0, c0):
        corner = "TL"
    elif (mr, mc) == (r0, c1):
        corner = "TR"
    elif (mr, mc) == (r1, c0):
        corner = "BL"
    else:
        corner = "BR"

    return {
        "color": int(comp["color"]),
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "corner": corner,
        "area": len(coords),
    }


def _corner_point(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, int(w - 1)
    if corner == "BL":
        return int(h - 1), 0
    return int(h - 1), int(w - 1)


def _corner_distance(coords: set[tuple[int, int]], shape: tuple[int, int], corner: str) -> int:
    cr, cc = _corner_point(shape, corner)
    return min(abs(int(r) - cr) + abs(int(c) - cc) for r, c in coords)


def _paint_corner_triangle(out: np.ndarray, corner: str, color: int) -> None:
    h, w = out.shape
    side = min(TRIANGLE_SIDE, h, w)
    for r in range(h):
        for c in range(w):
            if corner == "TL":
                d = int(r) + int(c)
            elif corner == "TR":
                d = int(r) + int(w - 1 - c)
            elif corner == "BL":
                d = int(h - 1 - r) + int(c)
            else:
                d = int(h - 1 - r) + int(w - 1 - c)
            if d < side:
                out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace colored L-shaped direction markers with matching solid corner triangles.

    Input structure:
    - The background is Azure.
    - The foreground consists of separated monochrome 3-pixel L markers, each
      occupying three cells of a 2x2 block with one corner missing.
    - The missing corner of each marker is one of TL, TR, BL, or BR, and the
      four 4x4 grid-corner regions begin as Azure.

    Rule:
    - Erase every L marker from its original position.
    - The marker's missing corner names the output corner to fill.
    - Paint a solid right triangle of side 4 in that corner, with the right
      angle at the grid corner and the legs along the two grid edges.
    - If several markers point to the same corner, use the largest marker;
      when sizes tie, use the marker closest to that target corner.

    Key insight:
    - The missing cell in each 2x2 L is a direction code, while the marker
      color is the payload transferred to the corresponding corner triangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w > 0, "expected a non-empty grid"

    bg = _bg_color(g)
    markers = []
    for comp in _same_color_components(g, bg):
        marker = _marker_from_component(comp)
        assert marker is not None, "every foreground component must be a 3-pixel L marker"
        marker["distance"] = _corner_distance(marker["coords"], (h, w), marker["corner"])
        markers.append(marker)
    assert markers, "expected at least one L marker"

    best_by_corner: dict[str, dict] = {}
    for marker in markers:
        corner = marker["corner"]
        cur = best_by_corner.get(corner)
        key = (
            int(marker["area"]),
            -int(marker["distance"]),
            -int(marker["bbox"][0]),
            -int(marker["bbox"][1]),
        )
        if cur is None:
            best_by_corner[corner] = marker
            continue
        cur_key = (
            int(cur["area"]),
            -int(cur["distance"]),
            -int(cur["bbox"][0]),
            -int(cur["bbox"][1]),
        )
        if key > cur_key:
            best_by_corner[corner] = marker

    out = np.full(g.shape, int(bg), dtype=int)
    for corner, marker in best_by_corner.items():
        _paint_corner_triangle(out, corner, int(marker["color"]))
    return out


def _corner_region_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    s = min(TRIANGLE_SIDE, h, w)
    mask[:s, :s] = True
    mask[:s, w - s :] = True
    mask[h - s :, :s] = True
    mask[h - s :, w - s :] = True
    return mask


def _abs_l_cells(r0: int, c0: int, direction: str) -> set[tuple[int, int]]:
    return {(int(r0 + dr), int(c0 + dc)) for dr, dc in L_CELLS[direction]}


def _fits(blocked: np.ndarray, r0: int, c0: int) -> bool:
    h, w = blocked.shape
    if not (1 <= r0 <= h - 3 and 1 <= c0 <= w - 3):
        return False
    for r in range(r0 - 1, r0 + 3):
        for c in range(c0 - 1, c0 + 3):
            if bool(blocked[r, c]):
                return False
    return True


def _reserve(blocked: np.ndarray, r0: int, c0: int) -> None:
    for r in range(r0 - 1, r0 + 3):
        for c in range(c0 - 1, c0 + 3):
            blocked[r, c] = True


def _conflicts_are_unambiguous(markers: list[dict], shape: tuple[int, int]) -> bool:
    by_corner = {corner: [] for corner in DIRS}
    for marker in markers:
        by_corner[marker["corner"]].append(
            _corner_distance(marker["coords"], shape, marker["corner"])
        )
    return all(len(ds) == len(set(ds)) for ds in by_corner.values())


def _directions_for_scene() -> list[str]:
    if random.choice([False, True]):
        directions = list(DIRS)
        directions.extend(random.choices(DIRS, k=random.randint(1, 4)))
    else:
        directions = random.sample(list(DIRS), k=random.randint(2, 4))
    random.shuffle(directions)
    return directions


def _gen_attempt() -> np.ndarray | None:
    directions = _directions_for_scene()
    n = len(directions)
    min_side = 12 if n <= 4 else 14 if n <= 6 else 16
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    colors = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    if n > len(colors):
        return None
    random.shuffle(colors)

    grid = np.full((h, w), AZURE, dtype=int)
    blocked = _corner_region_mask(h, w)
    markers = []

    for direction, color in zip(directions, colors):
        candidates = [
            (r, c)
            for r in range(1, h - 2)
            for c in range(1, w - 2)
            if _fits(blocked, r, c)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        cells = _abs_l_cells(r0, c0, direction)
        for r, c in cells:
            grid[r, c] = int(color)
        _reserve(blocked, r0, c0)
        markers.append({"corner": direction, "coords": cells})

    if not _conflicts_are_unambiguous(markers, (h, w)):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid L-marker corner-triangle input")
