from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
STAMP_COLORS = (3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS = ("tl", "tr", "bl", "br")

# --------------- shared helpers ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_boundary(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _corner_cells_from_interior(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
        "br": (r1 + 1, c1 + 1),
    }


def _anchor_from_corner(
    r0: int, c0: int, r1: int, c1: int, ph: int, pw: int, corner: str
) -> tuple[int, int]:
    if corner == "tl":
        return r0, c0
    if corner == "tr":
        return r0, c1 - pw + 1
    if corner == "bl":
        return r1 - ph + 1, c0
    if corner == "br":
        return r1 - ph + 1, c1 - pw + 1
    raise ValueError(f"unknown corner {corner}")


def _tile_region(
    out: np.ndarray, r0: int, c0: int, r1: int, c1: int, pattern: np.ndarray, corner: str
) -> None:
    ph, pw = map(int, pattern.shape)
    origin_r, origin_c = _anchor_from_corner(r0, c0, r1, c1, ph, pw, corner)
    row_idx = (np.arange(r0, r1 + 1) - origin_r) % ph
    col_idx = (np.arange(c0, c1 + 1) - origin_c) % pw
    out[r0 : r1 + 1, c0 : c1 + 1] = pattern[row_idx[:, None], col_idx[None, :]]


def _room_interiors(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    interiors: list[tuple[int, int, int, int]] = []
    for coords in _components(grid != BLUE):
        if _touches_boundary(coords, h, w):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        assert int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)
        interiors.append((r0, c0, r1, c1))
    return interiors


def _detect_corner(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> tuple[str, tuple[int, int]]:
    corners = _corner_cells_from_interior(r0, c0, r1, c1)
    found = [(name, rc) for name, rc in corners.items() if int(grid[rc]) == RED]
    assert len(found) == 1, "each room must have exactly one red wall corner"
    return found[0]


# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each room's corner stamp across the full room interior from the red-marked wall corner.

    Input structure:
    - The background is black.
    - Blue one-pixel walls form one or more axis-aligned rectangular rooms.
    - Each room contains a small 2x2 to 3x3 multicolor stamp and exactly one wall-corner pixel is red instead of blue.

    Rule:
    - Find every enclosed room interior surrounded by blue walls.
    - Inside each room, extract the small non-black stamp as a tile template and read the red wall corner as the tiling origin.
    - Repeat the tile across the whole room interior without rotation or reflection, clipping the far-edge tiles as needed, then restore the red wall pixel to blue.

    Key insight:
    - The colored corner patch is not a finished object but a reusable tile, and the unique red wall corner fixes the tiling phase.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _room_interiors(g):
        corner, marker = _detect_corner(g, r0, c0, r1, c1)
        interior = g[r0 : r1 + 1, c0 : c1 + 1]
        stamp_coords = np.argwhere(interior != BLACK)
        assert stamp_coords.size, "each room must contain a visible stamp"
        sr0, sc0, sr1, sc1 = _bbox(stamp_coords)
        pattern = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        assert 2 <= pattern.shape[0] <= 3 and 2 <= pattern.shape[1] <= 3
        _tile_region(out, r0, c0, r1, c1, pattern, corner)
        out[marker] = BLUE

    return out


# --------------- generator ---------------


MIN_ROOMS = 1
MAX_ROOMS = 4
GRID_MIN = 16
GRID_MAX = 30
ROOM_PLACE_TRIES = 250
GEN_RETRIES = 400


def _minimal_row_period(arr: np.ndarray) -> int:
    h = int(arr.shape[0])
    for p in range(1, h):
        if all(np.array_equal(arr[r], arr[r % p]) for r in range(h)):
            return p
    return h


def _minimal_col_period(arr: np.ndarray) -> int:
    w = int(arr.shape[1])
    for p in range(1, w):
        if all(np.array_equal(arr[:, c], arr[:, c % p]) for c in range(w)):
            return p
    return w


def _make_stamp(h: int, w: int) -> np.ndarray:
    for _ in range(200):
        n_colors = random.randint(2, min(4, h * w))
        palette = random.sample(STAMP_COLORS, n_colors)
        stamp = np.array([[random.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(map(int, stamp.flat))) < 2:
            continue
        if _minimal_row_period(stamp) != h:
            continue
        if _minimal_col_period(stamp) != w:
            continue
        return stamp
    raise RuntimeError("failed to build a non-degenerate stamp")


def _sample_span(tile_size: int) -> int:
    repeats = random.randint(2, 4)
    if random.random() < 0.5:
        visible_last = tile_size
    else:
        visible_last = random.randint(1, tile_size - 1)
    return (repeats - 1) * tile_size + visible_last


def _make_room_spec() -> dict[str, object]:
    ph = random.randint(2, 3)
    pw = random.randint(2, 3)
    pattern = _make_stamp(ph, pw)
    inner_h = _sample_span(ph)
    inner_w = _sample_span(pw)
    corner = random.choice(CORNERS)
    return {
        "pattern": pattern,
        "corner": corner,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
    }


def _draw_room(grid: np.ndarray, spec: dict[str, object]) -> None:
    top = int(spec["top"])
    left = int(spec["left"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top, left : right + 1] = BLUE
    grid[bottom, left : right + 1] = BLUE
    grid[top : bottom + 1, left] = BLUE
    grid[top : bottom + 1, right] = BLUE

    inner_r0 = top + 1
    inner_c0 = left + 1
    inner_r1 = bottom - 1
    inner_c1 = right - 1
    pattern = np.asarray(spec["pattern"], dtype=int)
    corner = str(spec["corner"])
    stamp_r, stamp_c = _anchor_from_corner(
        inner_r0, inner_c0, inner_r1, inner_c1, int(pattern.shape[0]), int(pattern.shape[1]), corner
    )
    grid[stamp_r : stamp_r + pattern.shape[0], stamp_c : stamp_c + pattern.shape[1]] = pattern

    marker = _corner_cells_from_interior(inner_r0, inner_c0, inner_r1, inner_c1)[corner]
    grid[marker] = RED


def _build_expected(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        pattern = np.asarray(spec["pattern"], dtype=int)
        corner = str(spec["corner"])
        r0 = top + 1
        c0 = left + 1
        r1 = top + outer_h - 2
        c1 = left + outer_w - 2
        _tile_region(out, r0, c0, r1, c1, pattern, corner)
        out[_corner_cells_from_interior(r0, c0, r1, c1)[corner]] = BLUE
    return out


def _place_rooms(specs: list[dict[str, object]], h: int, w: int) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["outer_h"]) * int(specs[i]["outer_w"]), reverse=True)

    for idx in order:
        spec = specs[idx]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        row_lo = 1 if h - outer_h >= 2 else 0
        row_hi = h - outer_h - 1 if h - outer_h >= 2 else h - outer_h
        col_lo = 1 if w - outer_w >= 2 else 0
        col_hi = w - outer_w - 1 if w - outer_w >= 2 else w - outer_w
        if row_hi < row_lo or col_hi < col_lo:
            return False

        for _ in range(ROOM_PLACE_TRIES):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(h - 1, top + outer_h)
            c1 = min(w - 1, left + outer_w)
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            spec["top"] = top
            spec["left"] = left
            break
        else:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(MIN_ROOMS, MAX_ROOMS)
    specs = [_make_room_spec() for _ in range(room_count)]

    max_outer_h = max(int(spec["outer_h"]) for spec in specs)
    max_outer_w = max(int(spec["outer_w"]) for spec in specs)
    min_h = max(GRID_MIN, max_outer_h + 2)
    min_w = max(GRID_MIN, max_outer_w + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)

    padded_area = sum((int(spec["outer_h"]) + 2) * (int(spec["outer_w"]) + 2) for spec in specs)
    if padded_area > int(h * w * 0.6):
        return None

    if not _place_rooms(specs, h, w):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for spec in specs:
        _draw_room(grid, spec)

    interiors = _room_interiors(grid)
    if len(interiors) != room_count:
        return None

    expected = _build_expected(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiled-room sample")
