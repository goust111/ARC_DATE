from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SEP = 5
OUTPUT_HEIGHT = 10
GAP = 1
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_RECTS = 3
MAX_RECTS = 5
MAX_GEN_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox of an (N,2) coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    allowed_colors: Iterable[int],
    col_range: tuple[int, int] | None = None,
) -> list[dict]:
    """Minimal 4-connected monochrome component finder in arc_utils style."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if col_range is None:
        c0, c1 = 0, w
    else:
        c0, c1 = map(int, col_range)
    colors = {int(color) for color in allowed_colors}
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(c0, c1):
            color = int(g[r, c])
            if seen[r, c] or color not in colors:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not (0 <= nr < h and c0 <= nc < c1):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, "expected exactly one full-height grey separator"
    return int(cols[0])


def _extract_rectangles_and_refs(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    sep_col = _find_separator_col(grid)
    obj_colors = PALETTE

    rects = find_components(grid, allowed_colors=obj_colors, col_range=(0, sep_col))
    rects = [rect for rect in rects if rect["area"] > 1]
    rects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))

    refs = find_components(grid, allowed_colors=obj_colors, col_range=(sep_col + 1, grid.shape[1]))
    refs = [ref for ref in refs if ref["area"] == 1]
    refs.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))

    assert MIN_RECTS <= len(rects) <= MAX_RECTS
    assert len(rects) == len(refs)
    assert len({rect["color"] for rect in rects}) == len(rects), "rectangle colors must be distinct"

    for rect, ref in zip(rects, refs):
        r0, c0, r1, c1 = rect["bbox"]
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        assert rect["area"] == hh * ww, "left objects must be solid rectangles"
        assert hh != ww, "rectangles must be non-square"
        assert c1 < sep_col < ref["bbox"][1], "references must stay right of the separator"
    return rects, refs


def _render(rects: list[dict], refs: list[dict]) -> np.ndarray:
    placed: list[tuple[int, int, int, bool]] = []
    total_w = 0
    for rect, ref in zip(rects, refs):
        r0, c0, r1, c1 = rect["bbox"]
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        short = min(hh, ww)
        long = max(hh, ww)
        horizontal = int(rect["color"]) == int(ref["color"])
        width = long if horizontal else short
        placed.append((int(rect["color"]), short, long, horizontal))
        total_w += width

    total_w += GAP * (len(placed) - 1)
    out = np.full((OUTPUT_HEIGHT, total_w), BG, dtype=int)

    col = 0
    for color, short, long, horizontal in placed:
        if horizontal:
            out[:short, col : col + long] = color
            col += long + GAP
        else:
            out[OUTPUT_HEIGHT - long : OUTPUT_HEIGHT, col : col + short] = color
            col += short + GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use color-matched reference pixels to choose whether each source rectangle is packed wide or tall.

    Input structure:
    - Black background with one full-height grey vertical separator.
    - Left of the separator: 3-5 solid non-square rectangles, each a distinct non-grey color.
    - Right of the separator: one single-cell reference pixel per rectangle, ordered top-to-bottom.

    Rule:
    - Pair rectangles and reference pixels from top to bottom.
    - If a rectangle color matches its paired reference color, normalize it to `min(h,w) x max(h,w)` and top-align it.
    - Otherwise normalize it to `max(h,w) x min(h,w)` and bottom-align it.
    - Pack the normalized rectangles left-to-right in source order on a 10-row black canvas, leaving one black column between neighbors.

    Key insight:
    - Rectangle size alone is insufficient: each object's final orientation is controlled by an external color comparison against its aligned reference pixel.
    """
    g = np.asarray(grid, dtype=int)
    rects, refs = _extract_rectangles_and_refs(g)
    return _render(rects, refs)


def _sample_match_flags(n_rects: int) -> list[bool]:
    flags = [bool(random.getrandbits(1)) for _ in range(n_rects)]
    if all(flags):
        flags[-1] = False
    if not any(flags):
        flags[0] = True
    return flags


def _sample_rectangles(n_rects: int) -> list[tuple[int, int]]:
    for _ in range(300):
        dims: list[tuple[int, int]] = []
        widths_sum = 0
        heights_sum = 0
        for _idx in range(n_rects):
            h = random.choice((1, 3, 5))
            candidates = [w for w in range(h + 1, 7)]
            if not candidates:
                break
            w = random.choice(candidates)
            dims.append((h, w))
            widths_sum += w
            heights_sum += h
        if len(dims) != n_rects:
            continue
        if widths_sum + GAP * (n_rects - 1) > 30:
            continue
        if heights_sum + 2 * (n_rects - 1) + 4 > 30:
            continue
        return dims
    raise RuntimeError("failed to sample rectangle dimensions")


def _layout_rows(dims: list[tuple[int, int]], colors: list[int]) -> tuple[list[int], int]:
    top_margin = 1 + (colors[0] % 2)
    rows: list[int] = []
    cur = top_margin
    for idx, (h, w) in enumerate(dims):
        rows.append(cur)
        if idx < len(dims) - 1:
            gap = 1 + ((h + w + colors[idx]) % 2)
            cur += h + gap
        else:
            cur += h
    bottom_margin = 1 + (colors[-1] % 2)
    return rows, cur + bottom_margin


def _build_input(
    dims: list[tuple[int, int]],
    colors: list[int],
    match_flags: list[bool],
) -> np.ndarray:
    rows, height = _layout_rows(dims, colors)
    max_width = max(w for _, w in dims)
    left_span = max_width + 3 + (sum(colors) % 2)
    sep_col = left_span
    ref_col = sep_col + 2 + (sum(w for _, w in dims) % 2)
    width = ref_col + 2

    assert height <= 30 and width <= 30
    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep_col] = SEP

    for idx, ((h, w), color, row0) in enumerate(zip(dims, colors, rows)):
        left_choices = left_span - w - 1
        assert left_choices >= 1
        col0 = 1 + ((color + h + 2 * idx) % left_choices)
        grid[row0 : row0 + h, col0 : col0 + w] = color

        center_row = row0 + h // 2
        ref_color = color if match_flags[idx] else colors[(idx + 1) % len(colors)]
        grid[center_row, ref_col] = ref_color

    return grid


def _spec_expected(grid: np.ndarray) -> np.ndarray:
    rects, refs = _extract_rectangles_and_refs(grid)
    return _render(rects, refs)


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        rects, refs = _extract_rectangles_and_refs(grid)
    except Exception:
        return False

    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False

    match_count = 0
    mismatch_count = 0
    for rect, ref in zip(rects, refs):
        r0, _c0, r1, _c1 = rect["bbox"]
        h = r1 - r0 + 1
        ref_r = int(ref["bbox"][0])
        if ref_r != r0 + h // 2:
            return False
        if int(rect["color"]) == int(ref["color"]):
            match_count += 1
        else:
            mismatch_count += 1

    if match_count == 0 or mismatch_count == 0:
        return False

    expected = _spec_expected(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if not (actual.shape[0] == OUTPUT_HEIGHT and max(actual.shape) <= 30):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n_rects = random.randint(MIN_RECTS, MAX_RECTS)
    colors = random.sample(PALETTE, n_rects)
    dims = _sample_rectangles(n_rects)
    match_flags = _sample_match_flags(n_rects)
    grid = _build_input(dims, colors, match_flags)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid ARC instance after {MAX_GEN_ATTEMPTS} attempts")
