from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, in_bounds, neighbors8


BG = Color.BLACK
GREY = Color.GREY
MIN_SEGMENT_LEN = 2
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GRID_SIZE_RANGE = (16, 30)
RECT_SIZE_RANGE = (4, 8)
RECT_MARGIN = 4
MAX_GENERATION_ATTEMPTS = 800


def _boundary_runs(grid: np.ndarray) -> dict[str, list[dict[str, int]]]:
    height, width = map(int, grid.shape)
    runs: dict[str, list[dict[str, int]]] = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": [],
        "full_h": [],
        "full_v": [],
    }

    for r in range(height):
        color = int(grid[r, 0])
        if color != BG:
            c = 0
            while c < width and int(grid[r, c]) == color:
                c += 1
            if c == width:
                runs["full_h"].append({"row": r, "color": color})
            elif c >= MIN_SEGMENT_LEN and int(grid[r, c]) == BG:
                runs["left"].append({"row": r, "endpoint": c - 1, "boundary": c, "color": color})

        color = int(grid[r, width - 1])
        if color != BG:
            c = width - 1
            while c >= 0 and int(grid[r, c]) == color:
                c -= 1
            start = c + 1
            length = width - start
            if length == width:
                # Already recorded as a full horizontal row from the left scan.
                pass
            elif length >= MIN_SEGMENT_LEN and int(grid[r, c]) == BG:
                runs["right"].append({"row": r, "endpoint": start, "boundary": start - 1, "color": color})

    for c in range(width):
        color = int(grid[0, c])
        if color != BG:
            r = 0
            while r < height and int(grid[r, c]) == color:
                r += 1
            if r == height:
                runs["full_v"].append({"col": c, "color": color})
            elif r >= MIN_SEGMENT_LEN and int(grid[r, c]) == BG:
                runs["top"].append({"col": c, "endpoint": r - 1, "boundary": r, "color": color})

        color = int(grid[height - 1, c])
        if color != BG:
            r = height - 1
            while r >= 0 and int(grid[r, c]) == color:
                r -= 1
            start = r + 1
            length = height - start
            if length == height:
                # Already recorded as a full vertical column from the top scan.
                pass
            elif length >= MIN_SEGMENT_LEN and int(grid[r, c]) == BG:
                runs["bottom"].append({"col": c, "endpoint": start, "boundary": start - 1, "color": color})

    return runs


def _unique_boundary(name: str, runs: list[dict[str, int]]) -> int:
    assert runs, f"expected at least one stopped {name} ray"
    values = {int(run["boundary"]) for run in runs}
    assert len(values) == 1, f"stopped {name} rays must agree on one hidden side"
    return values.pop()


def _infer_hidden_rectangle(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[str, list[dict[str, int]]]]:
    runs = _boundary_runs(grid)
    top = _unique_boundary("top", runs["top"])
    left = _unique_boundary("left", runs["left"])
    bottom = _unique_boundary("bottom", runs["bottom"])
    right = _unique_boundary("right", runs["right"])

    height, width = map(int, grid.shape)
    assert 0 < top <= bottom < height - 1, "hidden rectangle must have interior vertical margins"
    assert 0 < left <= right < width - 1, "hidden rectangle must have interior horizontal margins"

    for run in runs["left"]:
        assert top <= int(run["row"]) <= bottom, "left-stopped ray row must meet the inferred rectangle"
    for run in runs["right"]:
        assert top <= int(run["row"]) <= bottom, "right-stopped ray row must meet the inferred rectangle"
    for run in runs["top"]:
        assert left <= int(run["col"]) <= right, "top-stopped ray column must meet the inferred rectangle"
    for run in runs["bottom"]:
        assert left <= int(run["col"]) <= right, "bottom-stopped ray column must meet the inferred rectangle"

    return (top, left, bottom, right), runs


def _isolated_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    isolated: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color == BG:
                continue
            alone = True
            for rr, cc in neighbors8(r, c):
                if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BG:
                    alone = False
                    break
            if alone:
                isolated.append((r, c, color))
    return isolated


def _ray_to_rectangle(
    r: int, c: int, rect: tuple[int, int, int, int]
) -> list[tuple[int, int]] | None:
    top, left, bottom, right = rect
    if top <= r <= bottom:
        if c < left:
            return [(r, cc) for cc in range(c, left)]
        if c > right:
            return [(r, cc) for cc in range(right + 1, c + 1)]
    if left <= c <= right:
        if r < top:
            return [(rr, c) for rr in range(r, top)]
        if r > bottom:
            return [(rr, c) for rr in range(bottom + 1, r + 1)]
    return None


def _render(grid: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    top, left, bottom, right = rect
    out = grid.copy()
    out[top : bottom + 1, left : right + 1] = GREY

    for r, c, color in _isolated_pixels(grid):
        assert not (top <= r <= bottom and left <= c <= right), "isolated pixels must not sit inside the hidden rectangle"
        cells = _ray_to_rectangle(r, c, rect)
        if cells is None:
            continue
        rr = np.array([cell[0] for cell in cells], dtype=int)
        cc = np.array([cell[1] for cell in cells], dtype=int)
        out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the invisible rectangle from stopped boundary rays, then complete aligned isolated rays toward it.

    Input structure:
    - The grid has a black background.
    - Several single-color horizontal or vertical line residues start on a grid boundary; some reach the opposite boundary and some stop internally.
    - Stopped residues exist from all four directions, and additional isolated colored pixels are scattered away from the line residues.

    Rule:
    - Ignore full-width and full-height residues, because they were not blocked.
    - A stopped left-to-right residue fixes the hidden rectangle's left side one cell after its endpoint; right-to-left, top-to-bottom, and bottom-to-top residues analogously fix the right, top, and bottom sides.
    - Fill the uniquely inferred rectangle with gray, then extend every isolated pixel of its own color straight to the rectangle edge when the pixel lies on a rectangle row or column.

    Key insight:
    - The visible line endpoints are causal traces of an unseen obstacle; after reconstructing that obstacle, the isolated pixels can be treated as rays that have not yet been drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"
    assert int(g[0, 0]) == BG or np.count_nonzero(g == BG) > g.size // 2, "expected black background"

    rect, _runs = _infer_hidden_rectangle(g)
    return _render(g, rect)


def _paint_stopped_segments(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    line_color: int,
    left_rows: list[int],
    right_rows: list[int],
    top_cols: list[int],
    bottom_cols: list[int],
) -> None:
    top, left, bottom, right = rect
    height, width = map(int, grid.shape)
    for r in left_rows:
        grid[r, :left] = line_color
    for r in right_rows:
        grid[r, right + 1 : width] = line_color
    for c in top_cols:
        grid[:top, c] = line_color
    for c in bottom_cols:
        grid[bottom + 1 : height, c] = line_color


def _paint_full_distractor_lines(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    line_color: int,
) -> list[tuple[str, int]]:
    top, left, bottom, right = rect
    height, width = map(int, grid.shape)
    specs: list[tuple[str, int]] = []

    h_candidates = [r for r in range(1, height - 1) if r <= top - 2 or r >= bottom + 2]
    v_candidates = [c for c in range(1, width - 1) if c <= left - 2 or c >= right + 2]

    random.shuffle(h_candidates)
    random.shuffle(v_candidates)
    h_count = random.randint(0, min(2, len(h_candidates)))
    v_count = random.randint(0, min(2, len(v_candidates)))
    if h_count + v_count == 0:
        if h_candidates and (not v_candidates or random.random() < 0.5):
            h_count = 1
        elif v_candidates:
            v_count = 1

    for r in h_candidates[:h_count]:
        grid[r, :] = line_color
        specs.append(("h", r))
    for c in v_candidates[:v_count]:
        grid[:, c] = line_color
        specs.append(("v", c))
    return specs


def _path_clear(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> bool:
    if any((r, c) in reserved_paths for r, c in path):
        return False
    for r, c in path:
        if not in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
            return False
    sr, sc = path[0]
    for rr, cc in neighbors8(sr, sc):
        if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BG:
            return False
        if (rr, cc) in reserved_paths:
            return False
    return True


def _candidate_aligned_paths(
    side: str,
    shape: tuple[int, int],
    rect: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    height, width = shape
    top, left, bottom, right = rect
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []

    if side == "left":
        for r in range(top, bottom + 1):
            for c in range(1, left - 1):
                candidates.append(((r, c), [(r, cc) for cc in range(c, left)]))
    elif side == "right":
        for r in range(top, bottom + 1):
            for c in range(right + 2, width - 1):
                candidates.append(((r, c), [(r, cc) for cc in range(right + 1, c + 1)]))
    elif side == "top":
        for c in range(left, right + 1):
            for r in range(1, top - 1):
                candidates.append(((r, c), [(rr, c) for rr in range(r, top)]))
    elif side == "bottom":
        for c in range(left, right + 1):
            for r in range(bottom + 2, height - 1):
                candidates.append(((r, c), [(rr, c) for rr in range(bottom + 1, r + 1)]))

    random.shuffle(candidates)
    return candidates


def _place_aligned_pixel(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    side: str,
    color: int,
    reserved_paths: set[tuple[int, int]],
) -> tuple[int, int, int, str, list[tuple[int, int]]] | None:
    for (r, c), path in _candidate_aligned_paths(side, grid.shape, rect):
        if not _path_clear(grid, path, reserved_paths):
            continue
        grid[r, c] = int(color)
        reserved_paths.update(path)
        return (r, c, int(color), side, path)
    return None


def _place_distractor_pixel(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    color: int,
    reserved_paths: set[tuple[int, int]],
) -> tuple[int, int, int] | None:
    height, width = map(int, grid.shape)
    top, left, bottom, right = rect
    coords = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if not (top <= r <= bottom)
        and not (left <= c <= right)
        and (r, c) not in reserved_paths
        and int(grid[r, c]) == BG
    ]
    random.shuffle(coords)
    for r, c in coords:
        clear = True
        for rr, cc in neighbors8(r, c):
            if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BG:
                clear = False
                break
            if (rr, cc) in reserved_paths:
                clear = False
                break
        if not clear:
            continue
        grid[r, c] = int(color)
        return (r, c, int(color))
    return None


def _expected_from_metadata(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    aligned: list[tuple[int, int, int, str, list[tuple[int, int]]]],
) -> np.ndarray:
    top, left, bottom, right = rect
    out = grid.copy()
    out[top : bottom + 1, left : right + 1] = GREY
    for _r, _c, color, _side, path in aligned:
        rr = np.array([cell[0] for cell in path], dtype=int)
        cc = np.array([cell[1] for cell in path], dtype=int)
        out[rr, cc] = int(color)
    return out


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    rect: tuple[int, int, int, int],
    aligned: list[tuple[int, int, int, str, list[tuple[int, int]]]],
    distractors: list[tuple[int, int, int]],
) -> bool:
    try:
        inferred, runs = _infer_hidden_rectangle(grid)
        if inferred != rect:
            return False
        if not runs["full_h"] and not runs["full_v"]:
            return False
        if len({side for *_prefix, side, _path in aligned}) < 4:
            return False
        if not distractors:
            return False

        top, left, bottom, right = rect
        if not np.all(grid[top : bottom + 1, left : right + 1] == BG):
            return False
        if not np.all(expected[top : bottom + 1, left : right + 1] == GREY):
            return False

        iso_coords = {(r, c) for r, c, _color in _isolated_pixels(grid)}
        meta_coords = {(r, c) for r, c, _color, _side, _path in aligned}
        meta_coords.update((r, c) for r, c, _color in distractors)
        if iso_coords != meta_coords:
            return False

        for r, c, color, _side, path in aligned:
            cells = _ray_to_rectangle(r, c, rect)
            if cells != path:
                return False
            for rr, cc in path:
                if int(expected[rr, cc]) != int(color):
                    return False
        for r, c, color in distractors:
            if _ray_to_rectangle(r, c, rect) is not None:
                return False
            if int(expected[r, c]) != int(color):
                return False
        return np.array_equal(main(grid), expected)
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    max_rect_h = min(RECT_SIZE_RANGE[1], height - 2 * RECT_MARGIN)
    max_rect_w = min(RECT_SIZE_RANGE[1], width - 2 * RECT_MARGIN)
    if max_rect_h < RECT_SIZE_RANGE[0] or max_rect_w < RECT_SIZE_RANGE[0]:
        return None

    rect_h = random.randint(RECT_SIZE_RANGE[0], max_rect_h)
    rect_w = random.randint(RECT_SIZE_RANGE[0], max_rect_w)
    top = random.randint(RECT_MARGIN, height - rect_h - RECT_MARGIN)
    left = random.randint(RECT_MARGIN, width - rect_w - RECT_MARGIN)
    bottom = top + rect_h - 1
    right = left + rect_w - 1
    rect = (top, left, bottom, right)

    colors = random.sample(PALETTE, len(PALETTE))
    line_color = colors[0]
    pixel_colors = [color for color in colors[1:] if color != GREY]
    if len(pixel_colors) < 4:
        return None

    grid = np.full((height, width), BG, dtype=int)
    rect_rows = list(range(top, bottom + 1))
    rect_cols = list(range(left, right + 1))

    max_row_rays = min(3, len(rect_rows) - 1)
    max_col_rays = min(3, len(rect_cols) - 1)
    left_rows = random.sample(rect_rows, random.randint(1, max_row_rays))
    right_rows = random.sample(rect_rows, random.randint(1, max_row_rays))
    top_cols = random.sample(rect_cols, random.randint(1, max_col_rays))
    bottom_cols = random.sample(rect_cols, random.randint(1, max_col_rays))
    _paint_stopped_segments(grid, rect, line_color, left_rows, right_rows, top_cols, bottom_cols)
    _paint_full_distractor_lines(grid, rect, line_color)

    reserved_paths: set[tuple[int, int]] = set()
    aligned: list[tuple[int, int, int, str, list[tuple[int, int]]]] = []
    required_sides = ["left", "right", "top", "bottom"]
    random.shuffle(required_sides)
    color_index = 0
    for side in required_sides:
        placed = _place_aligned_pixel(
            grid,
            rect,
            side,
            pixel_colors[color_index % len(pixel_colors)],
            reserved_paths,
        )
        color_index += 1
        if placed is None:
            return None
        aligned.append(placed)

    for side in random.sample(["left", "right", "top", "bottom"], random.randint(0, 2)):
        placed = _place_aligned_pixel(
            grid,
            rect,
            side,
            pixel_colors[color_index % len(pixel_colors)],
            reserved_paths,
        )
        color_index += 1
        if placed is not None:
            aligned.append(placed)

    distractors: list[tuple[int, int, int]] = []
    for _ in range(random.randint(1, 3)):
        placed = _place_distractor_pixel(
            grid,
            rect,
            pixel_colors[color_index % len(pixel_colors)],
            reserved_paths,
        )
        color_index += 1
        if placed is None:
            return None
        distractors.append(placed)

    expected = _expected_from_metadata(grid, rect, aligned)
    if not _semantic_ok(grid, expected, rect, aligned, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-rectangle ray input")
