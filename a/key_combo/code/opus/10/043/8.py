from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
GREY = Color.GREY

BLOCK_COUNT_OPTIONS = (5, 7)
BLOCK_TOP_MIN = 2
GEN_RETRIES = 400

_GEN_CALL_COUNT = 0


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _other_path_color(color: int) -> int:
    return BLUE if int(color) == RED else RED


def _find_blocks(grid: np.ndarray) -> list[dict]:
    blocks = find_components(
        grid,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[GREEN],
    )
    good = []
    for block in blocks:
        if int(block["area"]) != 4:
            continue
        if bbox_h(block["bbox"]) != 2 or bbox_w(block["bbox"]) != 2:
            continue
        good.append(block)
    assert good, "expected green 2x2 blocks"
    left_cols = [int(block["bbox"][1]) for block in good]
    assert len(set(left_cols)) == len(left_cols), "block left edges must be unique"
    return sorted(good, key=lambda block: int(block["bbox"][1]))


def _find_markers(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    marker_coords = np.argwhere(grid == GREY)
    assert marker_coords.shape[0] == 2, "expected exactly two grey markers"
    markers = [tuple(map(int, coord)) for coord in marker_coords]
    markers.sort(key=lambda rc: rc[0])
    top_marker, bottom_marker = markers
    assert top_marker[0] == 0, "top marker must sit on the upper edge"
    assert bottom_marker[0] == grid.shape[0] - 1, "bottom marker must sit on the lower edge"
    return top_marker, bottom_marker


def _anchor(block: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = map(int, block["bbox"])
    return int(r0 + 1), int(c0 + 1)


def _walk_vertical_then_horizontal(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = map(int, start)
    tr, tc = map(int, end)
    walked: list[tuple[int, int]] = []

    while r != tr:
        r += _sign(tr - r)
        walked.append((int(r), int(c)))
    while c != tc:
        c += _sign(tc - c)
        walked.append((int(r), int(c)))
    return walked


def _path_segments(
    base: np.ndarray,
    nodes: list[tuple[int, int]],
    *,
    start_color: int = RED,
) -> list[tuple[list[tuple[int, int]], list[int]]]:
    segments: list[tuple[list[tuple[int, int]], list[int]]] = []
    current_start = int(start_color)

    for src, dst in zip(nodes, nodes[1:]):
        walked = _walk_vertical_then_horizontal(src, dst)
        paintable = [(int(r), int(c)) for r, c in walked if int(base[r, c]) == BLACK]
        assert paintable, "every segment must contribute visible path pixels"

        colors = []
        for idx, _ in enumerate(paintable):
            use_first = ((idx // 2) % 2) == 0
            colors.append(int(current_start if use_first else _other_path_color(current_start)))

        current_start = _other_path_color(colors[-1])
        segments.append((paintable, colors))

    return segments


def _render_output_from_parsed(
    grid: np.ndarray,
    sorted_blocks: list[dict],
    top_marker: tuple[int, int],
    bottom_marker: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[list[tuple[int, int]], list[int]]]]:
    out = np.array(grid, dtype=int, copy=True)
    kept_blocks = []
    for rank, block in enumerate(sorted_blocks, start=1):
        coords = block["coords"]
        if rank % 2 == 1:
            kept_blocks.append(block)
        else:
            out[coords[:, 0], coords[:, 1]] = BLACK

    base = out.copy()
    nodes = [top_marker] + [_anchor(block) for block in kept_blocks] + [bottom_marker]
    segments = _path_segments(base, nodes)

    for paintable, colors in segments:
        for (r, c), color in zip(paintable, colors):
            assert int(out[r, c]) == BLACK, "segments may only paint background after erasure"
            out[r, c] = int(color)

    return out, segments


def main(grid: np.ndarray) -> np.ndarray:
    """Erase every second horizontally sorted block, then connect the survivors with phase-linked alternating paths.

    Input structure:
    - Black background with 5-7 green 2x2 blocks whose left edges are all distinct.
    - One grey marker lies on the top edge near the leftmost block; one grey marker lies on the bottom edge near the rightmost block.

    Rule:
    - Sort the green 2x2 blocks by left edge from left to right, with ranks starting at 1.
    - Keep odd-ranked blocks and erase even-ranked blocks to black.
    - Starting from the top grey marker, connect the kept blocks in sorted order and then the bottom grey marker using Manhattan routes that move vertically first and horizontally second.
    - The first visible path segment starts red; later segments start with the opposite color from the previous segment's last visible path pixel, and each segment flips between red and blue every two painted pixels.
    - Grey markers and kept green blocks remain unchanged; only black cells created by the background or by erasing even-ranked blocks can be painted by the path.

    Key insight:
    - Horizontal sorting simultaneously determines which blocks survive, the station order of the route, and the color phase handed from one segment to the next.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(get_bg(g)) == BLACK, "background must be black"

    sorted_blocks = _find_blocks(g)
    assert len(sorted_blocks) in BLOCK_COUNT_OPTIONS and len(sorted_blocks) % 2 == 1

    top_marker, bottom_marker = _find_markers(g)
    return _render_output_from_parsed(g, sorted_blocks, top_marker, bottom_marker)[0]


def _make_input_grid(
    height: int,
    width: int,
    left_cols: list[int],
    top_rows: list[int],
    top_marker_col: int,
    bottom_marker_col: int,
) -> np.ndarray:
    grid = np.full((int(height), int(width)), BLACK, dtype=int)
    for top, left in zip(top_rows, left_cols):
        grid[int(top) : int(top) + 2, int(left) : int(left) + 2] = GREEN
    grid[0, int(top_marker_col)] = GREY
    grid[int(height) - 1, int(bottom_marker_col)] = GREY
    return grid


def _path_pixel_dict(
    segments: list[tuple[list[tuple[int, int]], list[int]]]
) -> dict[tuple[int, int], int]:
    painted: dict[tuple[int, int], int] = {}
    for coords, colors in segments:
        for coord, color in zip(coords, colors):
            assert coord not in painted, "path segments must not overlap on visible cells"
            painted[coord] = int(color)
    return painted


def _validate_semantics(inp: np.ndarray, out: np.ndarray, *, require_cross: bool | None) -> None:
    assert inp.shape == out.shape
    sorted_blocks = _find_blocks(inp)
    top_marker, bottom_marker = _find_markers(inp)
    rebuilt, segments = _render_output_from_parsed(inp, sorted_blocks, top_marker, bottom_marker)
    assert np.array_equal(out, rebuilt), "output must match the parsed rule"

    kept_masks = set()
    erased_masks = set()
    for rank, block in enumerate(sorted_blocks, start=1):
        coords = [tuple(map(int, rc)) for rc in block["coords"]]
        for coord in coords:
            if rank % 2 == 1:
                kept_masks.add(coord)
            else:
                erased_masks.add(coord)

    path_pixels = _path_pixel_dict(segments)
    assert path_pixels, "output must contain a visible path"
    assert RED in path_pixels.values() and BLUE in path_pixels.values(), "both path colors should appear"
    assert any(len(coords) >= 3 for coords, _ in segments), "at least one segment should show internal alternation"

    if require_cross is True:
        assert any(coord in erased_masks for coord in path_pixels), "expected a path-over-erased-block example"
    if require_cross is False:
        assert not any(coord in erased_masks for coord in path_pixels), "expected a clean non-crossing example"

    untouched_erased = [coord for coord in erased_masks if coord not in path_pixels]
    assert untouched_erased, "at least one erased cell should remain visibly erased"

    for coord in kept_masks:
        assert int(out[coord]) == GREEN
    for marker in (top_marker, bottom_marker):
        assert int(out[marker]) == GREY

    allowed = {BLACK, BLUE, RED, GREEN, GREY}
    assert set(map(int, np.unique(out))) <= allowed

    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            coord = (int(r), int(c))
            if coord in kept_masks or coord == top_marker or coord == bottom_marker:
                continue
            if coord in path_pixels:
                assert int(out[coord]) == int(path_pixels[coord])
            else:
                assert int(out[coord]) == BLACK


def _sample_left_columns(n_blocks: int) -> tuple[list[int], int] | None:
    for _ in range(80):
        start_left = int(np.random.randint(2, 5))
        if n_blocks == 5:
            gap_choices = np.array([3, 4, 4, 5], dtype=int)
        else:
            gap_choices = np.array([3, 3, 4, 4], dtype=int)
        gaps = [int(np.random.choice(gap_choices)) for _ in range(n_blocks - 1)]
        left_cols = [start_left]
        for gap in gaps:
            left_cols.append(left_cols[-1] + int(gap))
        width = int(left_cols[-1] + 5)
        if width <= 30:
            return left_cols, width
    return None


def _sample_kept_rows(height: int, kept_count: int, row_mode: int) -> list[int] | None:
    top_max = int(height - 4)
    valid_tops = np.arange(BLOCK_TOP_MIN, top_max + 1, dtype=int)
    if valid_tops.size < kept_count:
        return None

    if row_mode == 0:
        choices = np.random.choice(valid_tops, size=kept_count, replace=False)
        return [int(v) for v in choices]

    anchor_rows = np.arange(BLOCK_TOP_MIN + 1, height - 2, dtype=int)
    if anchor_rows.size < kept_count:
        return None

    if row_mode == 1:
        low = [int(v) for v in anchor_rows if int(v) <= int(height // 2)]
        high = [int(v) for v in anchor_rows if int(v) > int(height // 2)]
        if not low or not high:
            return None
        anchors = []
        for idx in range(kept_count):
            pool = low if idx % 2 == 0 else high
            anchors.append(int(np.random.choice(pool)))
        return [anchor - 1 for anchor in anchors]

    anchor_min = BLOCK_TOP_MIN + 1
    anchor_max = height - 3
    if anchor_max - anchor_min + 1 < kept_count:
        return None
    start = int(np.random.randint(anchor_min, anchor_max - kept_count + 2))
    anchors = list(range(start, start + kept_count))
    if np.random.random() < 0.5:
        anchors.reverse()
    return [int(anchor - 1) for anchor in anchors]


def _choose_even_top(
    height: int,
    target_top: int,
    *,
    force_cross: bool,
) -> int | None:
    top_max = int(height - 4)
    valid = list(range(BLOCK_TOP_MIN, top_max + 1))
    if force_cross:
        cross_candidates = [top for top in (int(target_top), int(target_top + 1)) if BLOCK_TOP_MIN <= top <= top_max]
        if not cross_candidates:
            return None
        return int(np.random.choice(np.array(cross_candidates, dtype=int)))

    safe = [top for top in valid if top not in (int(target_top), int(target_top + 1))]
    if not safe:
        return None
    return int(np.random.choice(np.array(safe, dtype=int)))


def _gen_attempt(branch_id: int) -> np.ndarray | None:
    n_blocks = BLOCK_COUNT_OPTIONS[branch_id % 2]
    cross_mode = (branch_id % 4) in (0, 1)
    row_mode = branch_id % 3
    sampled = _sample_left_columns(n_blocks)
    if sampled is None:
        return None
    left_cols, width = sampled

    height_lo = 16 if n_blocks == 5 else 18
    height = int(np.random.randint(height_lo, 25))
    kept_count = (n_blocks + 1) // 2
    kept_tops = _sample_kept_rows(height, kept_count, row_mode)
    if kept_tops is None:
        return None

    top_rows = [0] * n_blocks
    kept_indices = list(range(0, n_blocks, 2))
    for idx, top in zip(kept_indices, kept_tops):
        top_rows[int(idx)] = int(top)

    chosen_cross_even = int(np.random.randint(0, kept_count - 1)) if cross_mode else -1
    for even_order, even_idx in enumerate(range(1, n_blocks, 2)):
        target_top = int(top_rows[even_idx + 1])
        force_cross = cross_mode and even_order == chosen_cross_even
        even_top = _choose_even_top(height, target_top, force_cross=force_cross)
        if even_top is None:
            return None
        top_rows[int(even_idx)] = int(even_top)

    first_anchor = left_cols[0] + 1
    last_anchor = left_cols[-1] + 1
    top_choices = [offset for offset in (-3, -2) if 0 <= first_anchor + offset < first_anchor]
    bottom_choices = [offset for offset in (1, 2) if last_anchor < last_anchor + offset < width]
    if not top_choices or not bottom_choices:
        return None
    top_offset = int(top_choices[branch_id % len(top_choices)])
    bottom_offset = int(bottom_choices[(branch_id // 3) % len(bottom_choices)])
    top_marker_col = int(first_anchor + top_offset)
    bottom_marker_col = int(last_anchor + bottom_offset)

    inp = _make_input_grid(height, width, left_cols, top_rows, top_marker_col, bottom_marker_col)
    expected = np.array(inp, copy=True)
    parsed_blocks = _find_blocks(inp)
    top_marker, bottom_marker = _find_markers(inp)
    expected, segments = _render_output_from_parsed(expected, parsed_blocks, top_marker, bottom_marker)

    path_pixels = _path_pixel_dict(segments)
    erased_coords = set()
    for rank, block in enumerate(parsed_blocks, start=1):
        if rank % 2 == 0:
            erased_coords.update(tuple(map(int, rc)) for rc in block["coords"])

    if cross_mode and not any(coord in erased_coords for coord in path_pixels):
        return None
    if not cross_mode and any(coord in erased_coords for coord in path_pixels):
        return None

    try:
        _validate_semantics(inp, expected, require_cross=cross_mode)
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch_id = int(_GEN_CALL_COUNT)
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(branch_id)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
