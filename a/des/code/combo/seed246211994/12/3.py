from __future__ import annotations

import random
from collections import deque

import numpy as np


_LOCAL_TO_CORNER = {
    frozenset({(0, 0), (0, 1), (1, 0)}): "top_left",
    frozenset({(0, 0), (0, 1), (1, 1)}): "top_right",
    frozenset({(0, 0), (1, 0), (1, 1)}): "bottom_left",
    frozenset({(0, 1), (1, 0), (1, 1)}): "bottom_right",
}
_CORNERS = tuple(_LOCAL_TO_CORNER.values())


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1, "background must be unique"
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _draw_corner_l(out: np.ndarray, corner: str, color: int) -> None:
    n = int(out.shape[0])
    arm = n // 2

    if corner == "top_left":
        out[0, :arm] = color
        out[:arm, 0] = color
    elif corner == "top_right":
        out[0, n - arm :] = color
        out[:arm, n - 1] = color
    elif corner == "bottom_left":
        out[n - arm :, 0] = color
        out[n - 1, :arm] = color
    else:
        assert corner == "bottom_right"
        out[n - arm :, n - 1] = color
        out[n - 1, n - arm :] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each 3-pixel L marker into a same-color outer-corner L."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1, "grid must be an odd square"

    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected at least one marker"

    corner_to_color: dict[str, int] = {}
    for comp in comps:
        assert int(comp["area"]) == 3, "every object must be a 3-pixel marker"
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "marker bbox must be 2x2"

        local = frozenset((rr - r0, cc - c0) for rr, cc in comp["coords"])
        assert local in _LOCAL_TO_CORNER, "foreground objects must be L markers"
        corner = _LOCAL_TO_CORNER[local]
        color = int(comp["color"])

        if corner in corner_to_color:
            assert corner_to_color[corner] == color, "duplicate orientations must agree in color"
        else:
            corner_to_color[corner] = color

    out = np.full_like(g, bg)
    for corner, color in corner_to_color.items():
        _draw_corner_l(out, corner, color)
    return out


# --------------- generator ---------------

_GEN_GRID_SIZES = {
    1: (5, 7, 9, 11, 13, 15),
    2: (5, 7, 9, 11, 13, 15),
    3: (7, 9, 11, 13, 15),
    4: (7, 9, 11, 13, 15),
}
_GEN_COLORS = tuple(range(10))
_GEN_MAX_ATTEMPTS = 400
_GEN_ALL_DISTINCT_PROB = 0.5
_GEN_MARGIN = 1


def _marker_cells(corner: str) -> tuple[tuple[int, int], ...]:
    for local, mapped_corner in _LOCAL_TO_CORNER.items():
        if mapped_corner == corner:
            return tuple(sorted(local))
    raise AssertionError("unknown corner")


def _make_marker(bg: int, color: int, corner: str) -> np.ndarray:
    patch = np.full((2, 2), bg, dtype=int)
    for dr, dc in _marker_cells(corner):
        patch[dr, dc] = color
    return patch


def _valid_positions(occupied: np.ndarray, margin: int) -> list[tuple[int, int]]:
    n = int(occupied.shape[0])
    locs: list[tuple[int, int]] = []
    for r0 in range(n - 1):
        for c0 in range(n - 1):
            rr0 = max(0, r0 - margin)
            rr1 = min(n, r0 + 2 + margin)
            cc0 = max(0, c0 - margin)
            cc1 = min(n, c0 + 2 + margin)
            if bool(occupied[rr0:rr1, cc0:cc1].any()):
                continue
            locs.append((r0, c0))
    return locs


def _sample_marker_colors(bg: int, count: int) -> list[int]:
    palette = [color for color in _GEN_COLORS if color != bg]
    if count == 1 or random.random() < _GEN_ALL_DISTINCT_PROB:
        return random.sample(palette, count)

    base_count = random.randint(1, min(3, count))
    base = random.sample(palette, base_count)
    colors = [random.choice(base) for _ in range(count)]
    if count > 1 and len(set(colors)) == count:
        colors[random.randrange(count)] = colors[0]
    return colors


def _expected_output(n: int, bg: int, corner_to_color: dict[str, int]) -> np.ndarray:
    out = np.full((n, n), bg, dtype=int)
    for corner, color in corner_to_color.items():
        _draw_corner_l(out, corner, color)
    return out


def _place_markers(inp: np.ndarray, bg: int, corner_to_color: dict[str, int]) -> bool:
    occupied = np.zeros(inp.shape, dtype=bool)
    corners = list(corner_to_color)
    random.shuffle(corners)

    for corner in corners:
        locs = _valid_positions(occupied, _GEN_MARGIN)
        if not locs:
            return False
        r0, c0 = random.choice(locs)
        patch = _make_marker(bg, int(corner_to_color[corner]), corner)
        mask = patch != bg
        inp[r0 : r0 + 2, c0 : c0 + 2][mask] = patch[mask]
        occupied[r0 : r0 + 2, c0 : c0 + 2] |= mask
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(1, 4)
    n = random.choice(_GEN_GRID_SIZES[count])
    bg = random.choice(_GEN_COLORS)
    corners = random.sample(list(_CORNERS), count)
    colors = _sample_marker_colors(bg, count)
    corner_to_color = dict(zip(corners, colors))

    inp = np.full((n, n), bg, dtype=int)
    if not _place_markers(inp, bg, corner_to_color):
        return None

    expected = _expected_output(n, bg, corner_to_color)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
