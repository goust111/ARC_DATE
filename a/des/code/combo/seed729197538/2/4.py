from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
BAND_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ("up", "down", "left", "right")
DIR_BY_NAME = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shift_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    steps: int,
) -> list[tuple[int, int]]:
    return [(r + dr * steps, c + dc * steps) for r, c in coords]


def _handle_direction(coords: list[tuple[int, int]]) -> tuple[int, int]:
    pts = set(coords)
    leaf: tuple[int, int] | None = None
    neighbor: tuple[int, int] | None = None
    for r, c in coords:
        nbrs = [(r + dr, c + dc) for dr, dc in DIR4 if (r + dr, c + dc) in pts]
        if len(nbrs) == 1:
            assert leaf is None, "expected exactly one handle pixel"
            leaf = (r, c)
            neighbor = nbrs[0]
    assert leaf is not None and neighbor is not None, "expected one protruding handle"
    return leaf[0] - neighbor[0], leaf[1] - neighbor[1]


def _travel_distance(coords: list[tuple[int, int]], slide: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    if slide == (-1, 0):
        return r0
    if slide == (1, 0):
        return h - 1 - r1
    if slide == (0, -1):
        return c0
    if slide == (0, 1):
        return w - 1 - c1
    raise AssertionError("invalid slide direction")


def _band_color_on_path(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    slide: tuple[int, int],
    steps: int,
) -> int:
    dr, dc = slide
    paint = GREY
    for step in range(1, steps + 1):
        seen = {
            int(grid[r + dr * step, c + dc * step])
            for r, c in coords
            if int(grid[r + dr * step, c + dc * step]) not in (BLACK, GREY)
        }
        if seen:
            assert len(seen) == 1, "a single stamp path should not hit two band colors at once"
            paint = next(iter(seen))
    return paint


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each grey stamp opposite its one-pixel handle until it reaches the grid edge.
    Re-stamp the full shape there, recoloring it with the last crossed band color,
    or keeping it grey if no band is crossed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    h, w = g.shape

    out = g.copy()
    out[g == GREY] = BLACK

    stamps = _components(g == GREY)
    assert stamps, "expected at least one grey stamp"

    for coords in stamps:
        handle = _handle_direction(coords)
        slide = (-handle[0], -handle[1])
        steps = _travel_distance(coords, slide, (h, w))
        paint = _band_color_on_path(g, coords, slide, steps)
        dst = _shift_coords(coords, slide[0], slide[1], steps)
        for r, c in dst:
            assert out[r, c] == BLACK, "stamp destination must stay clear of color bands"
        for r, c in dst:
            out[r, c] = paint
    return out


# --------------- generator ---------------

CORE_SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
)
GRID_H_RANGE = range(13, 19)
GRID_W_RANGE = range(13, 19)
STAMP_COUNT_CHOICES = (2, 3, 4)
MIN_TRAVEL = 2
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 400
PADDING = 1


def _normalize(coords: set[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    shifted = tuple(sorted((r - min_r, c - min_c) for r, c in coords))
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    return shifted, h, w


def _side_cells(core: tuple[tuple[int, int], ...], side: str) -> list[tuple[int, int]]:
    if side == "up":
        edge = min(r for r, _ in core)
        return [(r, c) for r, c in core if r == edge]
    if side == "down":
        edge = max(r for r, _ in core)
        return [(r, c) for r, c in core if r == edge]
    if side == "left":
        edge = min(c for _, c in core)
        return [(r, c) for r, c in core if c == edge]
    if side == "right":
        edge = max(c for _, c in core)
        return [(r, c) for r, c in core if c == edge]
    raise AssertionError("unknown side")


def _make_stamp(core: tuple[tuple[int, int], ...], handle_side: str) -> tuple[tuple[tuple[int, int], ...], int, int]:
    anchor = random.choice(_side_cells(core, handle_side))
    dr, dc = DIR_BY_NAME[handle_side]
    handle = (anchor[0] + dr, anchor[1] + dc)
    assert handle not in core
    return _normalize(set(core) | {handle})


def _travel_for_bbox(r0: int, c0: int, sh: int, sw: int, slide: str, h: int, w: int) -> int:
    if slide == "up":
        return r0
    if slide == "down":
        return h - sh - r0
    if slide == "left":
        return c0
    if slide == "right":
        return w - sw - c0
    raise AssertionError("invalid slide")


def _swept_bbox(r0: int, c0: int, sh: int, sw: int, slide: str, h: int, w: int) -> tuple[int, int, int, int]:
    if slide == "up":
        return 0, c0, r0 + sh - 1, c0 + sw - 1
    if slide == "down":
        return r0, c0, h - 1, c0 + sw - 1
    if slide == "left":
        return r0, 0, r0 + sh - 1, c0 + sw - 1
    if slide == "right":
        return r0, c0, r0 + sh - 1, w - 1
    raise AssertionError("invalid slide")


def _pad_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _box_is_clear(blocked: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not blocked[r0 : r1 + 1, c0 : c1 + 1].any()


def _band_axis_positions(r0: int, c0: int, sh: int, sw: int, slide: str, h: int, w: int) -> list[int]:
    if slide == "left":
        return list(range(sw, c0))
    if slide == "right":
        return list(range(c0 + sw, w - sw))
    if slide == "up":
        return list(range(sh, r0))
    if slide == "down":
        return list(range(r0 + sh, h - sh))
    raise AssertionError("invalid slide")


def _candidate_positions(
    blocked: np.ndarray,
    sh: int,
    sw: int,
    slide: str,
    need_band: bool,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    along = sw if slide in ("left", "right") else sh
    min_travel = along + 1 if need_band else MIN_TRAVEL
    out: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            travel = _travel_for_bbox(r0, c0, sh, sw, slide, h, w)
            if travel < min_travel:
                continue
            sweep = _swept_bbox(r0, c0, sh, sw, slide, h, w)
            padded = _pad_box(sweep, PADDING, h, w)
            if not _box_is_clear(blocked, padded):
                continue
            if need_band and not _band_axis_positions(r0, c0, sh, sw, slide, h, w):
                continue
            out.append((r0, c0))
    return out


def _band_cells(r0: int, c0: int, sh: int, sw: int, slide: str, h: int, w: int) -> list[tuple[int, int]]:
    axis = random.choice(_band_axis_positions(r0, c0, sh, sw, slide, h, w))
    if slide in ("left", "right"):
        return [(r, axis) for r in range(r0, r0 + sh)]
    return [(axis, c) for c in range(c0, c0 + sw)]


def _place_stamp(
    blocked: np.ndarray,
    need_band: bool,
    band_color: int | None,
) -> dict | None:
    h, w = blocked.shape
    for _ in range(PLACEMENT_RETRIES):
        slide = random.choice(DIR_NAMES)
        handle_side = OPPOSITE[slide]
        core = random.choice(CORE_SHAPES)
        shape, sh, sw = _make_stamp(core, handle_side)
        candidates = _candidate_positions(blocked, sh, sw, slide, need_band)
        if not candidates:
            continue
        r0, c0 = random.choice(candidates)
        travel = _travel_for_bbox(r0, c0, sh, sw, slide, h, w)
        dr, dc = DIR_BY_NAME[slide]
        source = [(r0 + r, c0 + c) for r, c in shape]
        dest = _shift_coords(source, dr, dc, travel)
        sweep = _swept_bbox(r0, c0, sh, sw, slide, h, w)
        padded = _pad_box(sweep, PADDING, h, w)
        blocked[padded[0] : padded[2] + 1, padded[1] : padded[3] + 1] = True
        bands = _band_cells(r0, c0, sh, sw, slide, h, w) if need_band else []
        return {
            "source": source,
            "dest": dest,
            "band_cells": bands,
            "band_color": band_color,
            "paint": band_color if band_color is not None else GREY,
        }
    return None


def _render_example(h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    for spec in specs:
        if spec["band_color"] is not None:
            for r, c in spec["band_cells"]:
                inp[r, c] = spec["band_color"]
                out[r, c] = spec["band_color"]
    for spec in specs:
        for r, c in spec["source"]:
            assert inp[r, c] == BLACK
            inp[r, c] = GREY
        for r, c in spec["dest"]:
            assert out[r, c] == BLACK
            out[r, c] = spec["paint"]
    return inp, out


def _dyed_flags(count: int) -> list[bool]:
    flags = [random.random() < 0.5 for _ in range(count)]
    if count > 1 and (all(flags) or not any(flags)) and random.random() < 0.75:
        idx = random.randrange(count)
        flags[idx] = not flags[idx]
    return flags


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(tuple(GRID_H_RANGE))
    w = random.choice(tuple(GRID_W_RANGE))
    count = random.choice(STAMP_COUNT_CHOICES)
    flags = _dyed_flags(count)
    color_pool = random.sample(BAND_COLORS, sum(flags))
    blocked = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []
    color_idx = 0

    for need_band in flags:
        band_color = color_pool[color_idx] if need_band else None
        if need_band:
            color_idx += 1
        spec = _place_stamp(blocked, need_band, band_color)
        if spec is None:
            return None
        specs.append(spec)

    inp, out = _render_example(h, w, specs)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stamp-and-band input")
