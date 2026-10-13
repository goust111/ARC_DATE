from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

POINTER_COLORS = (RED, GREEN, BLUE, YELLOW, MAROON, ORANGE)
DIRS = ("up", "down", "left", "right")
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# 2x2 L-shapes, encoded by the missing corner:
# missing TL -> right, missing TR -> down, missing BR -> left, missing BL -> up
DIR_TO_POINTER = {
    "right": ((0, 1), (1, 0), (1, 1)),
    "down": ((0, 0), (1, 0), (1, 1)),
    "left": ((0, 0), (0, 1), (1, 0)),
    "up": ((0, 0), (0, 1), (1, 1)),
}
MISSING_TO_DIR = {
    (0, 0): "right",
    (0, 1): "down",
    (1, 1): "left",
    (1, 0): "up",
}

GEN_RETRIES = 400
LAYOUTS = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
]
OVERLAP_LAYOUTS = [(2, 2), (2, 3), (3, 2)]
SIZE_MIN = 6
SIZE_MAX = 8


def _spans_from_line_indices(lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        if b - a <= 1:
            continue
        spans.append((a + 1, b - 1))
    return spans


def _full_black_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = g.shape
    row_lines = [r for r in range(h) if np.all(g[r, :] == BLACK)]
    col_lines = [c for c in range(w) if np.all(g[:, c] == BLACK)]
    assert row_lines and row_lines[0] == 0 and row_lines[-1] == h - 1
    assert col_lines and col_lines[0] == 0 and col_lines[-1] == w - 1
    return row_lines, col_lines


def _parse_regions(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict]]:
    row_lines, col_lines = _full_black_lines(g)
    row_spans = _spans_from_line_indices(row_lines)
    col_spans = _spans_from_line_indices(col_lines)
    regions: list[dict] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            regions.append(
                {
                    "ri": ri,
                    "ci": ci,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                }
            )
    return row_spans, col_spans, regions


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _pointer_direction(local_coords: np.ndarray) -> str:
    assert len(local_coords) == 3
    r0, c0, r1, c1 = _bbox(local_coords)
    assert r1 - r0 == 1 and c1 - c0 == 1
    present = {(int(r - r0), int(c - c0)) for r, c in local_coords}
    missing = ({(0, 0), (0, 1), (1, 0), (1, 1)} - present)
    assert len(missing) == 1
    return MISSING_TO_DIR[missing.pop()]


def _region_contents(
    g: np.ndarray,
    region: dict,
) -> tuple[tuple[int, int, int, int], int, str]:
    r0, r1, c0, c1 = region["r0"], region["r1"], region["c0"], region["c1"]
    view = g[r0 : r1 + 1, c0 : c1 + 1]

    gray = np.argwhere(view == GRAY)
    assert len(gray) > 0
    gr0, gc0, gr1, gc1 = _bbox(gray)
    assert len(gray) == (gr1 - gr0 + 1) * (gc1 - gc0 + 1)
    assert np.all(view[gr0 : gr1 + 1, gc0 : gc1 + 1] == GRAY)

    colors = [
        int(v)
        for v in np.unique(view)
        if int(v) not in (AZURE, BLACK, GRAY)
    ]
    assert len(colors) == 1
    pointer_color = colors[0]
    pointer = np.argwhere(view == pointer_color)
    assert len(pointer) == 3
    direction = _pointer_direction(pointer)

    return (r0 + gr0, c0 + gc0, r0 + gr1, c0 + gc1), pointer_color, direction


def _destination_bbox(
    gray_bbox: tuple[int, int, int, int],
    region: dict,
    direction: str,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    gr0, gc0, gr1, gc1 = gray_bbox
    height = gr1 - gr0 + 1
    width = gc1 - gc0 + 1
    ri, ci = region["ri"], region["ci"]

    if direction == "up":
        if ri == 0:
            return None
        tr0, tr1 = row_spans[ri - 1]
        new_r0 = tr0
        new_c0 = gc0
    elif direction == "down":
        if ri + 1 >= len(row_spans):
            return None
        tr0, tr1 = row_spans[ri + 1]
        new_r0 = tr1 - height + 1
        new_c0 = gc0
    elif direction == "left":
        if ci == 0:
            return None
        tc0, tc1 = col_spans[ci - 1]
        new_r0 = gr0
        new_c0 = tc0
    else:
        assert direction == "right"
        if ci + 1 >= len(col_spans):
            return None
        tc0, tc1 = col_spans[ci + 1]
        new_r0 = gr0
        new_c0 = tc1 - width + 1

    return (new_r0, new_c0, new_r0 + height - 1, new_c0 + width - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each region's gray block into the adjacent region indicated by its L-shaped pointer and recolor it.

    Input structure:
    - The background is Azure and one-cell-thick Black rows/columns form a full rectangular partition grid.
    - Each partition cell is a rectangular region with Azure interior.
    - Every region contains exactly one solid Gray rectangle and exactly one 3-cell monochrome L-pointer in a non-Gray, non-Black, non-Azure color.

    Rule:
    - Read the pointer's direction from which corner is missing in its 2x2 L-shape: the four rotations encode up/down/left/right.
    - Remove every original Gray block and pointer, keeping only the Black partition lines.
    - For each region in top-to-bottom, left-to-right order, translate its Gray rectangle into the adjacent region in the pointer direction until the rectangle's front edge touches that target region's far Black wall, then recolor the moved rectangle with the pointer color.
    - If multiple translated rectangles overlap, later regions in the scan order overwrite earlier ones.

    Key insight:
    - The 3-cell L-pointer carries two pieces of information at once: its orientation selects the neighboring destination region, and its color becomes the moved block's final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    row_spans, col_spans, regions = _parse_regions(g)

    out = np.full_like(g, AZURE)
    out[g == BLACK] = BLACK

    for region in regions:
        gray_bbox, pointer_color, direction = _region_contents(g, region)
        dst = _destination_bbox(gray_bbox, region, direction, row_spans, col_spans)
        if dst is None:
            continue
        r0, c0, r1, c1 = dst
        out[r0 : r1 + 1, c0 : c1 + 1] = pointer_color

    return out


def _line_indices_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    lines = [0]
    cur = 1
    for size in sizes:
        spans.append((cur, cur + size - 1))
        cur += size
        lines.append(cur)
        cur += 1
    total = cur
    assert lines[-1] == total - 1
    return spans, lines, total


def _destination_region_key(ri: int, ci: int, direction: str) -> tuple[int, int]:
    dr, dc = DIR_TO_DELTA[direction]
    return ri + dr, ci + dc


def _choose_layout(rng: random.Random, want_overlap: bool) -> tuple[int, int]:
    choices = OVERLAP_LAYOUTS if want_overlap else LAYOUTS
    return rng.choice(choices)


def _random_sizes(count: int, rng: random.Random) -> list[int]:
    return [rng.randint(SIZE_MIN, SIZE_MAX) for _ in range(count)]


def _valid_dirs_for_region(ri: int, ci: int, n_rows: int, n_cols: int) -> list[str]:
    dirs: list[str] = []
    if ri > 0:
        dirs.append("up")
    if ri + 1 < n_rows:
        dirs.append("down")
    if ci > 0:
        dirs.append("left")
    if ci + 1 < n_cols:
        dirs.append("right")
    return dirs


def _assign_directions(
    n_rows: int,
    n_cols: int,
    want_overlap: bool,
    rng: random.Random,
) -> dict[tuple[int, int], str] | None:
    region_keys = [(ri, ci) for ri in range(n_rows) for ci in range(n_cols)]
    valid = {
        key: _valid_dirs_for_region(key[0], key[1], n_rows, n_cols)
        for key in region_keys
    }

    for _ in range(200):
        chosen: dict[tuple[int, int], str] = {}

        if want_overlap:
            target = rng.choice(region_keys)
            horiz_sources = []
            vert_sources = []
            ri, ci = target
            if ci > 0:
                horiz_sources.append(((ri, ci - 1), "right"))
            if ci + 1 < n_cols:
                horiz_sources.append(((ri, ci + 1), "left"))
            if ri > 0:
                vert_sources.append(((ri - 1, ci), "down"))
            if ri + 1 < n_rows:
                vert_sources.append(((ri + 1, ci), "up"))
            if not horiz_sources or not vert_sources:
                continue
            src1, dir1 = rng.choice(horiz_sources)
            src2, dir2 = rng.choice(vert_sources)
            chosen[src1] = dir1
            chosen[src2] = dir2

        available_dirs = [d for d in DIRS if any(d in valid[key] for key in region_keys)]
        if len(region_keys) >= len(available_dirs) and rng.random() < 0.75:
            remaining = [key for key in region_keys if key not in chosen]
            ok = True
            for direction in rng.sample(available_dirs, len(available_dirs)):
                if direction in chosen.values():
                    continue
                candidates = [key for key in remaining if direction in valid[key]]
                if not candidates:
                    ok = False
                    break
                picked = rng.choice(candidates)
                chosen[picked] = direction
                remaining.remove(picked)
            if not ok:
                continue

        for key in region_keys:
            if key not in chosen:
                chosen[key] = rng.choice(valid[key])

        if want_overlap:
            incoming: dict[tuple[int, int], list[str]] = {}
            for (ri, ci), direction in chosen.items():
                dst = _destination_region_key(ri, ci, direction)
                incoming.setdefault(dst, []).append(direction)
            if not any(
                any(d in ("left", "right") for d in dirs) and any(d in ("up", "down") for d in dirs)
                for dirs in incoming.values()
            ):
                continue

        return chosen

    return None


def _bboxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_region_objects(
    region_h: int,
    region_w: int,
    direction: str,
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]] | None:
    max_h = min(4, region_h - 3)
    max_w = min(4, region_w - 3)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(200):
        gray_h = rng.randint(2, max_h)
        gray_w = rng.randint(2, max_w)
        gr0 = rng.randint(0, region_h - gray_h)
        gc0 = rng.randint(0, region_w - gray_w)
        gray_bbox = (gr0, gc0, gr0 + gray_h - 1, gc0 + gray_w - 1)

        pr0 = rng.randint(0, region_h - 2)
        pc0 = rng.randint(0, region_w - 2)
        pointer_bbox = (pr0, pc0, pr0 + 1, pc0 + 1)
        if not _bboxes_have_gap(gray_bbox, pointer_bbox, pad=1):
            continue

        return gray_bbox, [(pr0 + dr, pc0 + dc) for dr, dc in DIR_TO_POINTER[direction]]

    return None


def _build_expected(
    inp: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    region_order: list[tuple[int, int]],
    plans: dict[tuple[int, int], dict],
) -> tuple[np.ndarray, int]:
    out = np.full_like(inp, AZURE)
    out[inp == BLACK] = BLACK
    cover = np.zeros_like(inp, dtype=int)

    for key in region_order:
        plan = plans[key]
        gray_bbox = plan["gray_global"]
        dst = _destination_bbox(gray_bbox, plan["region"], plan["direction"], row_spans, col_spans)
        assert dst is not None
        r0, c0, r1, c1 = dst
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
        out[r0 : r1 + 1, c0 : c1 + 1] = plan["color"]

    return out, int(np.sum(cover > 1))


def _paint_region(
    grid: np.ndarray,
    region: dict,
    gray_local: tuple[int, int, int, int],
    pointer_local: Iterable[tuple[int, int]],
    color: int,
) -> tuple[int, int, int, int]:
    r0, c0 = region["r0"], region["c0"]
    gr0, gc0, gr1, gc1 = gray_local
    gray_global = (r0 + gr0, c0 + gc0, r0 + gr1, c0 + gc1)
    grid[gray_global[0] : gray_global[2] + 1, gray_global[1] : gray_global[3] + 1] = GRAY
    for rr, cc in pointer_local:
        grid[r0 + rr, c0 + cc] = color
    return gray_global


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    want_overlap = rng.random() < 0.5
    n_rows, n_cols = _choose_layout(rng, want_overlap)
    row_sizes = _random_sizes(n_rows, rng)
    col_sizes = _random_sizes(n_cols, rng)
    row_spans, row_lines, height = _line_indices_from_sizes(row_sizes)
    col_spans, col_lines, width = _line_indices_from_sizes(col_sizes)
    if height > 30 or width > 30:
        return None

    directions = _assign_directions(n_rows, n_cols, want_overlap, rng)
    if directions is None:
        return None

    region_count = n_rows * n_cols
    colors = rng.sample(list(POINTER_COLORS), region_count)
    grid = np.full((height, width), AZURE, dtype=int)
    grid[row_lines, :] = BLACK
    grid[:, col_lines] = BLACK

    region_order = [(ri, ci) for ri in range(n_rows) for ci in range(n_cols)]
    plans: dict[tuple[int, int], dict] = {}

    for idx, key in enumerate(region_order):
        ri, ci = key
        region = {
            "ri": ri,
            "ci": ci,
            "r0": row_spans[ri][0],
            "r1": row_spans[ri][1],
            "c0": col_spans[ci][0],
            "c1": col_spans[ci][1],
        }
        placement = _place_region_objects(
            region["r1"] - region["r0"] + 1,
            region["c1"] - region["c0"] + 1,
            directions[key],
            rng,
        )
        if placement is None:
            return None
        gray_local, pointer_local = placement
        gray_global = _paint_region(grid, region, gray_local, pointer_local, colors[idx])
        plans[key] = {
            "region": region,
            "direction": directions[key],
            "color": colors[idx],
            "gray_global": gray_global,
        }

    expected, overlap_cells = _build_expected(grid, row_spans, col_spans, region_order, plans)
    if want_overlap != (overlap_cells > 0):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
