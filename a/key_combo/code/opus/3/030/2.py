from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, colors_present, get_bg, paste


BACKGROUND = 0
MARKER_COLOR = 1
FILL_COLOR = 2
BLOCK_COLOR = 3

BLOCK_SHAPE = (2, 2)
GEN_HEIGHT_RANGE = (11, 15)
GEN_WIDTH_RANGE = (11, 15)
GEN_MARKER_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 500
MIN_SOURCE_DISTANCE = 2

# (ring_row_offset, ring_col_offset, outward_dr, outward_dc, kind)
RING_SPECS = (
    (-1, -1, -1, -1, "diag"),
    (-1, 0, -1, 0, "orth"),
    (-1, 1, -1, 0, "orth"),
    (-1, 2, -1, 1, "diag"),
    (0, -1, 0, -1, "orth"),
    (1, -1, 0, -1, "orth"),
    (0, 2, 0, 1, "orth"),
    (1, 2, 0, 1, "orth"),
    (2, -1, 1, -1, "diag"),
    (2, 0, 1, 0, "orth"),
    (2, 1, 1, 0, "orth"),
    (2, 2, 1, 1, "diag"),
)


def _ring_projection(r: int, c: int, block_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = block_bbox
    nr = r0 - 1 if r < r0 else r1 + 1 if r > r1 else r
    nc = c0 - 1 if c < c0 else c1 + 1 if c > c1 else c
    return int(nr), int(nc)


def _trace_cells(start: tuple[int, int], stop: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = stop
    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    return [(r0 + step * dr, c0 + step * dc) for step in range(steps + 1)]


def _furthest_source(dest: tuple[int, int], outward: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    dr, dc = outward
    r, c = dest
    while 0 <= r + dr < h and 0 <= c + dc < w:
        r += dr
        c += dc
    return int(r), int(c)


def _find_anchor_block(g: np.ndarray) -> tuple[int, int, tuple[int, int, int, int], int]:
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    block_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in fg_colors:
        mask = g == int(color)
        box = bbox(mask)
        assert box is not None, "foreground color must be present"
        r0, c0, r1, c1 = box
        if (
            r1 - r0 + 1 == 2
            and c1 - c0 + 1 == 2
            and int(np.count_nonzero(mask)) == 4
            and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color))
        ):
            block_candidates.append((int(color), box))

    assert len(block_candidates) == 1, "expected one filled 2x2 anchor block"
    block_color, block_bbox = block_candidates[0]
    marker_color = [int(color) for color in fg_colors if int(color) != block_color][0]
    return block_color, marker_color, block_bbox, int(bg)


def _slide_markers(
    g: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int, int, int], int, int]:
    block_color, marker_color, block_bbox, bg = _find_anchor_block(g)
    r0, c0, r1, c1 = block_bbox

    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape[0] >= 1, "expected at least one marker"

    out = g.copy()
    out[g == marker_color] = bg

    destinations: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    height, width = map(int, g.shape)
    for rr, cc in marker_coords:
        r = int(rr)
        c = int(cc)
        assert not (r0 <= r <= r1 and c0 <= c <= c1), "marker cannot lie inside the anchor"

        nr, nc = _ring_projection(r, c, block_bbox)
        row_aligned = r0 <= r <= r1
        col_aligned = c0 <= c <= c1
        if not row_aligned and not col_aligned:
            assert abs(r - nr) == abs(c - nc), "diagonal marker must lie on a 45-degree ray"

        assert 0 <= nr < height and 0 <= nc < width, "marker destination left the grid"
        assert int(g[nr, nc]) != block_color, "marker cannot overwrite the anchor"
        assert (nr, nc) not in seen, "marker destinations must be unique"
        seen.add((nr, nc))
        destinations.append((nr, nc))
        out[nr, nc] = marker_color

    return out, destinations, block_bbox, marker_color, bg


def _fill_intersections(grid: np.ndarray, rows: set[int], cols: set[int], bg: int) -> np.ndarray:
    out = grid.copy()
    for r in sorted(rows):
        for c in sorted(cols):
            if int(out[r, c]) == int(bg):
                out[r, c] = FILL_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the edge markers onto the anchor ring, then fill the red intersections defined by their landing rows and columns.

    Input structure:
    - The background is black and there are exactly two foreground colors in the input.
    - One foreground color forms a unique solid 2x2 anchor block.
    - The other foreground color appears as single-pixel markers aligned with that block by row, column, or 45-degree diagonal.

    Rule:
    - Move every marker straight toward the 2x2 anchor along its aligned ray and stop in the corresponding cell of the outer 4x4 ring.
    - Erase the original marker positions and redraw the markers only at those landing cells.
    - Collect the set of landing rows R and the set of landing columns C, then fill every background cell in R x C with red.
    - Never overwrite the blue landing markers or the green anchor block.

    Key insight:
    - The slide destinations are not just decorative ring markers: their row and column indices are the only coordinates that define the red Cartesian-product fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    slid, destinations, _block_bbox, _marker_color, bg = _slide_markers(g)
    rows = {r for r, _c in destinations}
    cols = {c for _r, c in destinations}
    return _fill_intersections(slid, rows, cols, bg)


# --------------- generator ---------------


def _sample_anchor(shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = map(int, shape)
    if h < 9 or w < 9:
        return None

    row_options = list(range(3, h - 4))
    col_options = list(range(3, w - 4))
    if not row_options or not col_options:
        return None

    random.shuffle(row_options)
    random.shuffle(col_options)
    for r0 in row_options:
        for c0 in col_options:
            if h % 2 == 0 and w % 2 == 0 and r0 == h // 2 - 1 and c0 == w // 2 - 1:
                continue
            return int(r0), int(c0)
    return None


def _candidate_marker_specs(
    shape: tuple[int, int],
    anchor: tuple[int, int],
) -> list[dict]:
    h, w = map(int, shape)
    r0, c0 = anchor
    out: list[dict] = []
    for ro, co, dr, dc, kind in RING_SPECS:
        dest = (r0 + ro, c0 + co)
        source = _furthest_source(dest, (dr, dc), (h, w))
        distance = max(abs(source[0] - dest[0]), abs(source[1] - dest[1]))
        if distance < MIN_SOURCE_DISTANCE:
            continue
        out.append(
            {
                "dest": dest,
                "source": source,
                "outward": (dr, dc),
                "kind": kind,
            }
        )
    return out


def _sample_marker_specs(candidates: list[dict]) -> list[dict] | None:
    count = random.randint(*GEN_MARKER_COUNT_RANGE)
    orth = [spec for spec in candidates if spec["kind"] == "orth"]
    diag = [spec for spec in candidates if spec["kind"] == "diag"]

    profile = random.choices(
        ("mixed", "diag_heavy", "orth_only"),
        weights=(0.5, 0.25, 0.25),
        k=1,
    )[0]

    if profile == "orth_only":
        if len(orth) < count:
            return None
        chosen = random.sample(orth, count)
    else:
        if profile == "diag_heavy":
            diag_min = 2
            diag_max = min(3, count - 1, len(diag))
        else:
            diag_min = 1
            diag_max = min(2, count - 1, len(diag))
        if diag_max < diag_min:
            return None
        diag_count = random.randint(diag_min, diag_max)
        orth_count = count - diag_count
        if len(orth) < orth_count:
            return None
        chosen = random.sample(diag, diag_count) + random.sample(orth, orth_count)

    random.shuffle(chosen)
    return chosen


def _render_slide_only(shape: tuple[int, int], anchor: tuple[int, int], destinations: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    paste(out, np.full(BLOCK_SHAPE, BLOCK_COLOR, dtype=int), anchor[0], anchor[1])
    for r, c in destinations:
        out[r, c] = MARKER_COLOR
    return out


def _render_input(shape: tuple[int, int], anchor: tuple[int, int], sources: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    paste(grid, np.full(BLOCK_SHAPE, BLOCK_COLOR, dtype=int), anchor[0], anchor[1])
    for r, c in sources:
        grid[r, c] = MARKER_COLOR
    return grid


def _wrong_fill_from_sources(slid: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    rows = {r for r, _c in sources}
    cols = {c for _r, c in sources}
    return _fill_intersections(slid, rows, cols, BACKGROUND)


def _wrong_fill_full_ring(slid: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    r0, c0 = anchor
    out = slid.copy()
    for r in range(r0 - 1, r0 + 3):
        for c in range(c0 - 1, c0 + 3):
            if int(out[r, c]) == BACKGROUND:
                out[r, c] = FILL_COLOR
    return out


def _validate_sample(
    inp: np.ndarray,
    expected: np.ndarray,
    slide_only: np.ndarray,
    anchor: tuple[int, int],
    chosen: list[dict],
) -> bool:
    marker_sources = [tuple(spec["source"]) for spec in chosen]
    marker_dests = [tuple(spec["dest"]) for spec in chosen]
    rows = {r for r, _c in marker_dests}
    cols = {c for _r, c in marker_dests}

    if len(rows) < 2 or len(cols) < 2:
        return False
    if len(rows) == 4 and len(cols) == 4:
        return False

    red_count = int(np.count_nonzero(expected == FILL_COLOR))
    if red_count < 2:
        return False
    if np.array_equal(expected, slide_only):
        return False
    if np.array_equal(expected, _wrong_fill_from_sources(slide_only, marker_sources)):
        return False
    if np.array_equal(expected, _wrong_fill_full_ring(slide_only, anchor)):
        return False

    for source, dest in zip(marker_sources, marker_dests):
        if source[0] not in {0, inp.shape[0] - 1} and source[1] not in {0, inp.shape[1] - 1}:
            return False
        ray = _trace_cells(source, dest)
        for r, c in ray[1:]:
            if int(inp[r, c]) != BACKGROUND:
                return False

    try:
        actual = main(inp)
    except AssertionError:
        return False

    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GEN_HEIGHT_RANGE),
        random.randint(*GEN_WIDTH_RANGE),
    )
    anchor = _sample_anchor(shape)
    if anchor is None:
        return None

    candidates = _candidate_marker_specs(shape, anchor)
    chosen = _sample_marker_specs(candidates)
    if chosen is None:
        return None

    sources = [tuple(spec["source"]) for spec in chosen]
    destinations = [tuple(spec["dest"]) for spec in chosen]
    if len(set(sources)) != len(sources):
        return None

    inp = _render_input(shape, anchor, sources)
    slide_only = _render_slide_only(shape, anchor, destinations)
    expected = _fill_intersections(
        slide_only,
        {r for r, _c in destinations},
        {c for _r, c in destinations},
        BACKGROUND,
    )

    if not _validate_sample(inp, expected, slide_only, anchor, chosen):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 030/2")
