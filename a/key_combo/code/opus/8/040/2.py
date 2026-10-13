from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg


MARKER_COLOR = Color.GREY
BACKGROUND_COLOR = Color.BLACK
CONTOUR_COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

GRID_HEIGHT_RANGE = (15, 19)
GRID_WIDTH_RANGE = (15, 19)
CONTOUR_COUNT_RANGE = (3, 4)
CONTOUR_HEIGHT_RANGE = (4, 8)
ROW_SPAN_RANGE = (3, 7)  # right - left on each occupied row
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
STEP_ATTEMPTS = 30


def _reading_key(coords: np.ndarray) -> tuple[int, int]:
    top_row = int(coords[:, 0].min())
    top_cols = coords[coords[:, 0] == top_row, 1]
    return top_row, int(top_cols.min())


def _extract_contours(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    contours: list[dict] = []
    for color in colors_present(grid, exclude={bg, MARKER_COLOR}):
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0, "expected every contour color to appear"

        row_bounds: dict[int, tuple[int, int]] = {}
        for r in sorted({int(v) for v in coords[:, 0]}):
            cols = sorted(int(v) for v in coords[coords[:, 0] == r, 1])
            assert len(cols) == 2, "expected exactly two boundary pixels per occupied row"
            left, right = cols
            assert right > left + 1, "expected a non-empty interior on each occupied row"
            row_bounds[r] = (left, right)

        contours.append(
            {
                "color": int(color),
                "coords": coords,
                "row_bounds": row_bounds,
                "order_key": _reading_key(coords),
            }
        )

    contours.sort(key=lambda item: item["order_key"])
    return contours


def _interior_cells(contour: dict) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in sorted(contour["row_bounds"]):
        left, right = contour["row_bounds"][r]
        for c in range(int(left) + 1, int(right)):
            cells.append((int(r), int(c)))
    return cells


def _marker_count(grid: np.ndarray, contour: dict) -> int:
    count = 0
    for r in sorted(contour["row_bounds"]):
        left, right = contour["row_bounds"][r]
        count += int(np.count_nonzero(grid[int(r), int(left) + 1 : int(right)] == MARKER_COLOR))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the sparsest outline interior with a diagonal color cycle derived from all contour colors.

    Input structure:
    - The grid has a black background, 3-4 differently colored outline shapes, and optional grey marker pixels inside them.
    - Each outline color occupies several consecutive rows, with exactly two same-colored boundary pixels on every occupied row.
    - Grey marker pixels, when present, appear only strictly between a row's left and right boundary pixels.

    Rule:
    - Count the grey marker pixels inside each outline by scanning the strict interior of every occupied row.
    - Select the unique outline with the fewest markers.
    - Sort all outlines in reading order by their topmost row, then by the leftmost boundary pixel on that row, and use that ordered color list as a cycle.
    - Recolor every interior cell of the selected outline with cycle[(row + col) mod K], where K is the number of outlines, while leaving all boundary pixels and all non-selected outlines unchanged.

    Key insight:
    - The same contour set provides both selection information (marker sparsity) and the diagonal fill palette/order, so the winner is chosen locally but filled by a global periodic rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    contours = _extract_contours(g)
    assert contours, "expected at least one contour"

    marker_counts = [_marker_count(g, contour) for contour in contours]
    min_count = min(marker_counts)
    winners = [idx for idx, count in enumerate(marker_counts) if count == min_count]
    assert len(winners) == 1, "expected a unique sparsest contour"

    target = contours[winners[0]]
    cycle = [int(contour["color"]) for contour in contours]
    period = len(cycle)

    out = g.copy()
    for r, c in _interior_cells(target):
        out[int(r), int(c)] = int(cycle[(int(r) + int(c)) % period])
    return out


def _generate_outline_rows(rng: np.random.Generator) -> tuple[list[tuple[int, int]], int, int] | None:
    height = int(rng.integers(CONTOUR_HEIGHT_RANGE[0], CONTOUR_HEIGHT_RANGE[1] + 1))
    lefts = [0]
    rights = [int(rng.integers(ROW_SPAN_RANGE[0], ROW_SPAN_RANGE[1] + 1))]

    for _ in range(1, height):
        for _attempt in range(STEP_ATTEMPTS):
            new_left = lefts[-1] + int(rng.choice([-1, 0, 1]))
            new_right = rights[-1] + int(rng.choice([-1, 0, 1]))
            if ROW_SPAN_RANGE[0] <= new_right - new_left <= ROW_SPAN_RANGE[1]:
                lefts.append(new_left)
                rights.append(new_right)
                break
        else:
            return None

    if len(set(zip(lefts, rights))) == 1 and rng.random() < 0.6:
        return None

    min_left = min(lefts)
    lefts = [int(v - min_left) for v in lefts]
    rights = [int(v - min_left) for v in rights]
    width = max(rights) + 1
    rows = [(lefts[r], rights[r]) for r in range(height)]
    return rows, height, width


def _boxes_too_close(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _place_outlines(
    rng: np.random.Generator,
    height: int,
    width: int,
    outline_rows: list[tuple[list[tuple[int, int]], int, int]],
    colors: np.ndarray,
) -> list[dict] | None:
    placed: list[dict] = []

    order = list(range(len(outline_rows)))
    rng.shuffle(order)
    for idx in order:
        rows, shape_h, shape_w = outline_rows[idx]
        if shape_h > height or shape_w > width:
            return None

        placed_one = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(0, height - shape_h + 1))
            c0 = int(rng.integers(0, width - shape_w + 1))
            bbox = (r0, c0, r0 + shape_h - 1, c0 + shape_w - 1)
            if any(_boxes_too_close(bbox, item["bbox"]) for item in placed):
                continue

            coords = []
            row_bounds: dict[int, tuple[int, int]] = {}
            for local_r, (left, right) in enumerate(rows):
                gr = r0 + int(local_r)
                gl = c0 + int(left)
                gright = c0 + int(right)
                coords.append((gr, gl))
                coords.append((gr, gright))
                row_bounds[gr] = (gl, gright)

            placed.append(
                {
                    "color": int(colors[idx]),
                    "bbox": bbox,
                    "coords": np.array(coords, dtype=int),
                    "row_bounds": row_bounds,
                }
            )
            placed_one = True
            break

        if not placed_one:
            return None

    return placed


def _residue_coverage(cells: list[tuple[int, int]], period: int) -> int:
    return len({(int(r) + int(c)) % int(period) for r, c in cells})


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    contour_count = int(rng.integers(CONTOUR_COUNT_RANGE[0], CONTOUR_COUNT_RANGE[1] + 1))
    colors = rng.choice(CONTOUR_COLOR_POOL, size=contour_count, replace=False)

    outline_rows: list[tuple[list[tuple[int, int]], int, int]] = []
    for _ in range(contour_count):
        shape = _generate_outline_rows(rng)
        if shape is None:
            return None
        outline_rows.append(shape)

    contours = _place_outlines(rng, height, width, outline_rows, colors)
    if contours is None:
        return None

    ordered = sorted(contours, key=lambda item: _reading_key(item["coords"]))
    target_candidates = []
    for idx, contour in enumerate(ordered):
        cells = _interior_cells(contour)
        if len(cells) < contour_count:
            continue
        if _residue_coverage(cells, contour_count) == contour_count:
            target_candidates.append(idx)
    if not target_candidates:
        return None

    target_idx = int(rng.choice(target_candidates))

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    for contour in ordered:
        grid[contour["coords"][:, 0], contour["coords"][:, 1]] = int(contour["color"])

    non_target_counts = list(range(1, contour_count))
    rng.shuffle(non_target_counts)

    count_cursor = 0
    for idx, contour in enumerate(ordered):
        cells = _interior_cells(contour)
        marker_count = 0 if idx == target_idx else non_target_counts[count_cursor]
        if idx != target_idx:
            count_cursor += 1
        if marker_count > len(cells):
            return None
        if marker_count:
            pick = rng.choice(len(cells), size=marker_count, replace=False)
            for chosen in np.atleast_1d(pick):
                r, c = cells[int(chosen)]
                grid[int(r), int(c)] = MARKER_COLOR

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    target_cells = _interior_cells(ordered[target_idx])
    if len({int(out[r, c]) for r, c in target_cells}) != contour_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-outline input")
