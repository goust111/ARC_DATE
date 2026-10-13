import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5

# Keep the three bracket colors fixed across all generated examples, but do not
# align them with numeric order. Depth must be recovered from the nesting chain.
DEPTH_TO_BRACKET_COLOR = {
    1: 3,  # green
    2: 1,  # blue
    3: 2,  # red
}

GLYPHS = {
    1: np.array(
        [
            [0, 4, 0],
            [4, 4, 4],
            [0, 4, 0],
        ],
        dtype=int,
    ),
    2: np.array(
        [
            [6, 0, 0],
            [6, 6, 0],
            [6, 0, 0],
        ],
        dtype=int,
    ),
    3: np.array(
        [
            [7, 7, 7],
            [0, 7, 0],
            [7, 0, 7],
        ],
        dtype=int,
    ),
}

DICT_TOPS = (0, 4, 8)
DICT_COL = 0
DICT_MARK_COL = 3
MAIN_START_COL = 8
HEIGHT_RANGE = (14, 19)
WIDTH_RANGE = (28, 30)
INNER_LEN_CHOICES = (3, 4)
MAX_ATTEMPTS = 500


def _extract_glyphs(grid: np.ndarray) -> dict[int, np.ndarray]:
    grey_coords = sorted((int(r), int(c)) for r, c in np.argwhere(grid == GREY))
    assert len(grey_coords) == 3, "expected exactly three grey dictionary markers"

    glyphs: dict[int, np.ndarray] = {}
    for depth, (r, c) in enumerate(grey_coords, start=1):
        assert 1 <= r <= grid.shape[0] - 2, "grey marker must sit on the middle row of a 3x3 glyph"
        assert c >= 3, "grey marker must have a 3x3 glyph immediately to its left"
        glyph = grid[r - 1 : r + 2, c - 3 : c].copy()
        assert glyph.shape == (3, 3)
        assert np.any(glyph != BG), "glyph cannot be empty"
        assert GREY not in glyph, "grey markers are not part of glyphs"
        glyphs[depth] = glyph

    first_rows = {tuple(map(int, glyphs[d][0])) for d in glyphs}
    assert len(first_rows) == 3, "the three glyph top rows must be distinct"
    return glyphs


def _find_bracket_rows(grid: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for r in range(grid.shape[0]):
        cols = [(int(c), int(grid[r, c])) for c in np.flatnonzero(np.isin(grid[r], [1, 2, 3]))]
        if not cols:
            continue
        colors = {color for _, color in cols}
        assert len(cols) == 2 and len(colors) == 1, "each bracket row must contain exactly one same-color pair"
        left, color = cols[0]
        right, _ = cols[1]
        assert left < right - 1, "each bracket pair must enclose at least one cell"
        rows.append(
            {
                "row": int(r),
                "left": int(left),
                "right": int(right),
                "color": int(color),
                "segment": grid[r, left + 1 : right].copy(),
            }
        )

    assert rows, "expected at least one bracket row"
    return rows


def _strictly_contains(outer: dict, inner: dict) -> bool:
    return int(outer["left"]) < int(inner["left"]) and int(outer["right"]) > int(inner["right"])


def _pair_depths(rows: list[dict]) -> tuple[list[int], dict[int, int]]:
    depths: list[int] = []
    color_to_depths: dict[int, set[int]] = defaultdict(set)

    for row in rows:
        containers = sum(_strictly_contains(other, row) for other in rows if other is not row)
        depth = int(containers) + 1
        assert 1 <= depth <= 3, "nesting depth must be between 1 and 3"
        depths.append(depth)
        color_to_depths[int(row["color"])].add(depth)

    color_to_depth: dict[int, int] = {}
    for color, seen in color_to_depths.items():
        assert len(seen) == 1, "one bracket color cannot represent multiple depths"
        color_to_depth[int(color)] = int(next(iter(seen)))

    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i >= j:
                continue
            overlaps = a["left"] < b["right"] and b["left"] < a["right"]
            nested = _strictly_contains(a, b) or _strictly_contains(b, a)
            assert not overlaps or nested, "intervals may only be disjoint or properly nested"

    return depths, color_to_depth


def _render_output(rows: list[dict], glyphs: dict[int, np.ndarray], color_to_depth: dict[int, int]) -> np.ndarray:
    processed: list[tuple[int, int, np.ndarray]] = []
    for row in rows:
        depth = int(color_to_depth[int(row["color"])])
        segment = np.array(row["segment"], dtype=int, copy=True)
        top = glyphs[depth][0]
        use = min(segment.size, 3)
        segment[:use] = top[:use]
        processed.append((depth, int(row["row"]), segment))

    processed.sort(key=lambda item: (item[0], item[1]))
    width = max(int(seg.size) for _, _, seg in processed)
    out = np.zeros((len(processed), width), dtype=int)
    for r, (_, _, seg) in enumerate(processed):
        out[r, : seg.size] = seg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract bracketed row segments, infer their nesting depth, and stamp the depth glyph at each segment start.

    Input structure:
    - Black background with a left dictionary strip: three vertically stacked 3x3 glyphs, each tagged by one grey pixel immediately to its right.
    - The main area contains horizontal rows with exactly one same-color bracket pair; the bracket colors are green, blue, and red.
    - Rows may be nested by horizontal containment: a shallower pair strictly contains a deeper pair.

    Rule:
    - Read the three glyphs from top to bottom as the depth-1, depth-2, and depth-3 dictionary entries.
    - For every bracket row, compute its depth from how many other bracket intervals strictly contain it, and require one color to map to one depth consistently.
    - Extract the cells strictly between the two brackets, replace the first up-to-three cells by the first row of the glyph for that depth, and keep the remaining cells unchanged.
    - Stack the processed segments by increasing depth, breaking ties by original row order, and right-pad shorter rows with black.

    Key insight:
    - The bracket colors do not need numeric ordering; the containment chain reveals which color is outer, middle, and inner, and that recovered depth selects the glyph row to stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    glyphs = _extract_glyphs(g)
    rows = _find_bracket_rows(g)
    _depths, color_to_depth = _pair_depths(rows)
    return _render_output(rows, glyphs, color_to_depth)


def _build_chain(inner_left: int, inner_len: int) -> dict[int, tuple[int, int]]:
    inner_right = inner_left + inner_len + 1
    return {
        1: (inner_left - 2, inner_right + 2),
        2: (inner_left - 1, inner_right + 1),
        3: (inner_left, inner_right),
    }


def _sample_rows(height: int) -> list[int]:
    forbidden = {top + 1 for top in DICT_TOPS}
    choices = [r for r in range(height) if r not in forbidden]
    assert len(choices) >= 6
    rows = sorted(random.sample(choices, 6))
    return rows


def _sample_two_chains(width: int, inner_a: int, inner_b: int) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    outer_width_a = inner_a + 6
    outer_width_b = inner_b + 6
    available = width - MAIN_START_COL
    free = available - outer_width_a - outer_width_b
    assert free >= 0

    pad_left = random.randint(0, free)
    gap = random.randint(0, free - pad_left)
    left_a = MAIN_START_COL + pad_left
    left_b = left_a + outer_width_a + gap

    chain_a = _build_chain(left_a + 2, inner_a)
    chain_b = _build_chain(left_b + 2, inner_b)
    if random.random() < 0.5:
        chain_a, chain_b = chain_b, chain_a
    return chain_a, chain_b


def _metadata_output(rows: list[dict]) -> np.ndarray:
    packed = []
    for row in rows:
        depth = int(row["depth"])
        segment = np.zeros(int(row["right"]) - int(row["left"]) - 1, dtype=int)
        top = GLYPHS[depth][0]
        use = min(segment.size, 3)
        segment[:use] = top[:use]
        packed.append((depth, int(row["row"]), segment))
    packed.sort(key=lambda item: (item[0], item[1]))
    width = max(seg.size for _, _, seg in packed)
    out = np.zeros((len(packed), width), dtype=int)
    for r, (_, _, seg) in enumerate(packed):
        out[r, : seg.size] = seg
    return out


def _validate_case(grid: np.ndarray, rows: list[dict]) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(out, _metadata_output(rows)):
        return False

    # The depth-sorted output must differ from preserving the original row order.
    unsorted = []
    for row in sorted(rows, key=lambda item: int(item["row"])):
        segment = np.zeros(int(row["right"]) - int(row["left"]) - 1, dtype=int)
        segment[: min(segment.size, 3)] = GLYPHS[int(row["depth"])][0][: min(segment.size, 3)]
        unsorted.append(segment)
    width = max(seg.size for seg in unsorted)
    original_order = np.zeros((len(unsorted), width), dtype=int)
    for r, seg in enumerate(unsorted):
        original_order[r, : seg.size] = seg
    if np.array_equal(out, original_order):
        return False

    first_rows = {tuple(map(int, GLYPHS[d][0])) for d in GLYPHS}
    if len(first_rows) != 3:
        return False

    inferred_rows = _find_bracket_rows(grid)
    depths, color_to_depth = _pair_depths(inferred_rows)
    if sorted(depths) != [1, 1, 2, 2, 3, 3]:
        return False
    if sorted(color_to_depth.values()) != [1, 2, 3]:
        return False

    sigs = {(int(row["depth"]), int(row["right"]) - int(row["left"]) - 1) for row in rows}
    return len(sigs) >= 3


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    grid = np.zeros((height, width), dtype=int)

    for depth, top in enumerate(DICT_TOPS, start=1):
        if top + 2 >= height:
            return None
        grid[top : top + 3, DICT_COL : DICT_COL + 3] = GLYPHS[depth]
        grid[top + 1, DICT_MARK_COL] = GREY

    rows = _sample_rows(height)
    inner_a = random.choice(INNER_LEN_CHOICES)
    inner_b = random.choice(INNER_LEN_CHOICES)
    chain_a, chain_b = _sample_two_chains(width, inner_a, inner_b)

    spec_rows = [
        {"row": rows[0], "depth": 1, "left": chain_a[1][0], "right": chain_a[1][1]},
        {"row": rows[1], "depth": 2, "left": chain_a[2][0], "right": chain_a[2][1]},
        {"row": rows[2], "depth": 3, "left": chain_a[3][0], "right": chain_a[3][1]},
        {"row": rows[3], "depth": 1, "left": chain_b[1][0], "right": chain_b[1][1]},
        {"row": rows[4], "depth": 2, "left": chain_b[2][0], "right": chain_b[2][1]},
        {"row": rows[5], "depth": 3, "left": chain_b[3][0], "right": chain_b[3][1]},
    ]

    for row in spec_rows:
        color = DEPTH_TO_BRACKET_COLOR[int(row["depth"])]
        rr = int(row["row"])
        left = int(row["left"])
        right = int(row["right"])
        if not (MAIN_START_COL <= left < right < width):
            return None
        if np.any(grid[rr, left : right + 1] != BG):
            return None
        grid[rr, left] = color
        grid[rr, right] = color

    if not _validate_case(grid, spec_rows):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-bracket ARC input")
