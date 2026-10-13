from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAGENTA = 6

CORNER_KEYS = ("tl", "tr", "bl", "br")
COLOR_POOL = [1, 2, 3, 4, 7, 8, 9]
BRANCHES = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
    ("left", "top"),
    ("left", "bottom"),
    ("right", "top"),
    ("right", "bottom"),
]

GEN_HEIGHT_RANGE = (22, 30)
GEN_WIDTH_RANGE = (22, 30)
GEN_BOX_COUNT_RANGE = (2, 4)
GEN_BOX_HEIGHT_RANGE = (5, 8)
GEN_BOX_WIDTH_RANGE = (5, 8)
GEN_MAX_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _corner_colors(grid: np.ndarray) -> dict[str, int]:
    patches = {
        "tl": grid[:2, :2],
        "tr": grid[:2, -2:],
        "bl": grid[-2:, :2],
        "br": grid[-2:, -2:],
    }
    colors: dict[str, int] = {}
    for key, patch in patches.items():
        vals = np.unique(patch)
        assert len(vals) == 1, f"corner palette {key} must be uniform"
        colors[key] = int(vals[0])
    return colors


def _axis_distance(index: int, anchors: tuple[int, int]) -> int:
    return min(abs(int(index) - int(anchor)) for anchor in anchors)


def _horizontal_side(width: int, col: int, *, break_ties: bool) -> str | None:
    left_dist = _axis_distance(int(col), (0, 1))
    right_dist = _axis_distance(int(col), (int(width - 2), int(width - 1)))
    if left_dist < right_dist:
        return "left"
    if right_dist < left_dist:
        return "right"
    return "left" if break_ties else None


def _vertical_side(height: int, row: int, *, break_ties: bool) -> str | None:
    top_dist = _axis_distance(int(row), (0, 1))
    bottom_dist = _axis_distance(int(row), (int(height - 2), int(height - 1)))
    if top_dist < bottom_dist:
        return "top"
    if bottom_dist < top_dist:
        return "bottom"
    return "top" if break_ties else None


def _closure_color(
    corners: dict[str, int], height: int, width: int, side: str, row: int, col: int
) -> int:
    if side == "top":
        return int(corners["tl"] if _horizontal_side(width, col, break_ties=True) == "left" else corners["tr"])
    if side == "bottom":
        return int(corners["bl"] if _horizontal_side(width, col, break_ties=True) == "left" else corners["br"])
    if side == "left":
        return int(corners["tl"] if _vertical_side(height, row, break_ties=True) == "top" else corners["bl"])
    if side == "right":
        return int(corners["tr"] if _vertical_side(height, row, break_ties=True) == "top" else corners["br"])
    raise AssertionError(side)


def _gap_info(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> tuple[str, list[tuple[int, int]]]:
    sides = {
        "top": [(int(r0), int(c)) for c in range(int(c0 + 1), int(c1)) if int(grid[r0, c]) != GREY],
        "bottom": [(int(r1), int(c)) for c in range(int(c0 + 1), int(c1)) if int(grid[r1, c]) != GREY],
        "left": [(int(r), int(c0)) for r in range(int(r0 + 1), int(r1)) if int(grid[r, c0]) != GREY],
        "right": [(int(r), int(c1)) for r in range(int(r0 + 1), int(r1)) if int(grid[r, c1]) != GREY],
    }
    present = [(side, cells) for side, cells in sides.items() if cells]
    assert len(present) == 1, "every frame must have exactly one opening side"
    side, cells = present[0]
    if side in ("top", "bottom"):
        cols = [c for _r, c in cells]
        assert cols == list(range(cols[0], cols[-1] + 1)), "opening must be contiguous"
    else:
        rows = [r for r, _c in cells]
        assert rows == list(range(rows[0], rows[-1] + 1)), "opening must be contiguous"
    return side, cells


def _seed_info(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int, int]:
    seeds: list[tuple[int, int, int]] = []
    for r in range(int(r0 + 1), int(r1)):
        for c in range(int(c0 + 1), int(c1)):
            value = int(grid[r, c])
            if value not in (BLACK, GREY):
                seeds.append((int(r), int(c), int(value)))
    assert len(seeds) == 1, "every frame must contain exactly one seed"
    return seeds[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each seed toward its frame opening and close that opening with the color
    of the nearer same-side corner palette.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    corners = _corner_colors(g)
    frames = _components(g == GREY)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = _bbox(frame)
        assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must have a nontrivial interior"
        side, gap_cells = _gap_info(g, r0, c0, r1, c1)
        seed_r, seed_c, seed_color = _seed_info(g, r0, c0, r1, c1)

        out[seed_r, seed_c] = BLACK
        if side == "top":
            new_r, new_c = r0 + 1, seed_c
        elif side == "bottom":
            new_r, new_c = r1 - 1, seed_c
        elif side == "left":
            new_r, new_c = seed_r, c0 + 1
        else:
            new_r, new_c = seed_r, c1 - 1

        out[new_r, new_c] = seed_color
        close_color = _closure_color(corners, h, w, side, new_r, new_c)
        for gr, gc in gap_cells:
            out[gr, gc] = close_color

    return out


def _expanded_slice(
    height: int, width: int, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int, int, int]:
    return max(0, r0 - 1), max(0, c0 - 1), min(height - 1, r1 + 1), min(width - 1, c1 + 1)


def _preferred_cols(width: int, c0: int, c1: int, prefer: str) -> list[int]:
    cols = []
    for c in range(int(c0 + 2), int(c1 - 1)):
        if _horizontal_side(width, c, break_ties=False) == prefer:
            cols.append(int(c))
    return cols


def _preferred_rows(height: int, r0: int, r1: int, prefer: str) -> list[int]:
    rows = []
    for r in range(int(r0 + 2), int(r1 - 1)):
        if _vertical_side(height, r, break_ties=False) == prefer:
            rows.append(int(r))
    return rows


def _valid_starts(
    height: int,
    width: int,
    box_h: int,
    box_w: int,
    side: str,
    prefer: str,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    max_r0 = int(height - box_h - 3)
    max_c0 = int(width - box_w - 3)
    for r0 in range(3, max_r0 + 1):
        r1 = int(r0 + box_h - 1)
        for c0 in range(3, max_c0 + 1):
            c1 = int(c0 + box_w - 1)
            er0, ec0, er1, ec1 = _expanded_slice(height, width, r0, c0, r1, c1)
            if bool(occupied[er0 : er1 + 1, ec0 : ec1 + 1].any()):
                continue
            if side in ("top", "bottom"):
                if not _preferred_cols(width, c0, c1, prefer):
                    continue
            else:
                if not _preferred_rows(height, r0, r1, prefer):
                    continue
            starts.append((int(r0), int(c0)))
    return starts


def _gap_cells_for_spec(r0: int, c0: int, r1: int, c1: int, side: str) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        positions = list(range(int(c0 + 1), int(c1)))
        max_len = min(4, len(positions) - 1)
        gap_len = random.randint(1, max_len)
        start = random.randint(0, len(positions) - gap_len)
        row = int(r0 if side == "top" else r1)
        return [(row, int(c)) for c in positions[start : start + gap_len]]

    positions = list(range(int(r0 + 1), int(r1)))
    max_len = min(4, len(positions) - 1)
    gap_len = random.randint(1, max_len)
    start = random.randint(0, len(positions) - gap_len)
    col = int(c0 if side == "left" else c1)
    return [(int(r), col) for r in positions[start : start + gap_len]]


def _draw_corner_palettes(grid: np.ndarray, corners: dict[str, int]) -> None:
    grid[:2, :2] = int(corners["tl"])
    grid[:2, -2:] = int(corners["tr"])
    grid[-2:, :2] = int(corners["bl"])
    grid[-2:, -2:] = int(corners["br"])


def _draw_box(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    gap_cells: list[tuple[int, int]],
    seed_r: int,
    seed_c: int,
    seed_color: int,
) -> None:
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    for gr, gc in gap_cells:
        grid[gr, gc] = BLACK
    grid[seed_r, seed_c] = int(seed_color)


def _build_expected(
    inp: np.ndarray, specs: list[dict[str, object]], corners: dict[str, int]
) -> np.ndarray:
    out = inp.copy()
    h, w = out.shape
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = int(spec["r1"])
        c1 = int(spec["c1"])
        side = str(spec["side"])
        seed_r = int(spec["seed_r"])
        seed_c = int(spec["seed_c"])
        seed_color = int(spec["seed_color"])
        gap_cells = list(spec["gap_cells"])

        out[seed_r, seed_c] = BLACK
        if side == "top":
            new_r, new_c = r0 + 1, seed_c
        elif side == "bottom":
            new_r, new_c = r1 - 1, seed_c
        elif side == "left":
            new_r, new_c = seed_r, c0 + 1
        else:
            new_r, new_c = seed_r, c1 - 1

        out[new_r, new_c] = seed_color
        fill = _closure_color(corners, h, w, side, new_r, new_c)
        for gr, gc in gap_cells:
            out[int(gr), int(gc)] = int(fill)
    return out


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    n_boxes = random.randint(*GEN_BOX_COUNT_RANGE)

    corners = dict(zip(CORNER_KEYS, random.sample(COLOR_POOL, 4), strict=False))
    seed_colors = random.sample(COLOR_POOL, n_boxes)
    branches = random.sample(BRANCHES, n_boxes)

    skeletons = []
    for (side, prefer), seed_color in zip(branches, seed_colors, strict=False):
        skeletons.append(
            {
                "side": side,
                "prefer": prefer,
                "seed_color": int(seed_color),
                "box_h": random.randint(*GEN_BOX_HEIGHT_RANGE),
                "box_w": random.randint(*GEN_BOX_WIDTH_RANGE),
            }
        )
    skeletons.sort(key=lambda spec: int(spec["box_h"]) * int(spec["box_w"]), reverse=True)

    occupied = np.zeros((height, width), dtype=bool)
    specs: list[dict[str, object]] = []

    for skeleton in skeletons:
        side = str(skeleton["side"])
        prefer = str(skeleton["prefer"])
        box_h = int(skeleton["box_h"])
        box_w = int(skeleton["box_w"])

        starts = _valid_starts(height, width, box_h, box_w, side, prefer, occupied)
        if not starts:
            return None
        r0, c0 = random.choice(starts)
        r1 = int(r0 + box_h - 1)
        c1 = int(c0 + box_w - 1)

        if side in ("top", "bottom"):
            final_c = random.choice(_preferred_cols(width, c0, c1, prefer))
            seed_r = random.randint(int(r0 + 2), int(r1 - 2))
            seed_c = int(final_c)
        else:
            final_r = random.choice(_preferred_rows(height, r0, r1, prefer))
            seed_r = int(final_r)
            seed_c = random.randint(int(c0 + 2), int(c1 - 2))

        gap_cells = _gap_cells_for_spec(r0, c0, r1, c1, side)
        er0, ec0, er1, ec1 = _expanded_slice(height, width, r0, c0, r1, c1)
        occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True

        specs.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "side": side,
                "seed_r": int(seed_r),
                "seed_c": int(seed_c),
                "seed_color": int(skeleton["seed_color"]),
                "gap_cells": gap_cells,
            }
        )

    inp = np.full((height, width), BLACK, dtype=int)
    _draw_corner_palettes(inp, corners)
    for spec in specs:
        _draw_box(
            inp,
            int(spec["r0"]),
            int(spec["c0"]),
            int(spec["r1"]),
            int(spec["c1"]),
            list(spec["gap_cells"]),
            int(spec["seed_r"]),
            int(spec["seed_c"]),
            int(spec["seed_color"]),
        )

    expected = _build_expected(inp, specs, corners)
    if len(_components(inp == GREY)) != n_boxes:
        return None
    if np.array_equal(inp, expected):
        return None
    if not set(np.unique(inp)).isdisjoint({MAGENTA}):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        candidate = _generate_scene()
        if candidate is not None:
            inp, _expected = candidate
            return inp
    raise RuntimeError("failed to generate a valid input")
