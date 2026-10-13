from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, get_bg
except Exception:
    class Color:
        BLACK = 0
        RED = 2
        GREEN = 3
        YELLOW = 4
        TEAL = 8

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).flat)).most_common(1)[0][0])


BBox = tuple[int, int, int, int]

BLACK = int(Color.BLACK)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
TEAL = int(Color.TEAL)
MARKER_COLORS = (RED, GREEN, YELLOW)

HEIGHT_RANGE = (10, 18)
WIDTH_RANGE = (12, 20)
MARKED_ROW_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 400
ROW_PICK_ATTEMPTS = 120


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_anchor_bbox(grid: np.ndarray) -> BBox:
    coords = np.argwhere(grid == TEAL)
    assert coords.shape[0] == 4, "expected exactly four teal anchor pixels"
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    assert r1 == r0 + 1 and c1 == c0 + 1, "anchor must be a 2x2 square"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == TEAL), "anchor square must be solid teal"
    return box


def _collect_marked_rows(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    marked: dict[int, tuple[int, int]] = {}
    width = int(grid.shape[1])
    for r in range(int(grid.shape[0])):
        left = int(grid[r, 0])
        right = int(grid[r, width - 1])
        if left == bg and right == bg:
            continue
        assert left != bg and right != bg, "every active row must have both boundary markers"
        marked[r] = (left, right)
    assert marked, "expected at least one marked row"
    return marked


def _scene_info(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = int(get_bg(g))
    assert bg == BLACK, "background must be black"

    anchor_bbox = _find_anchor_bbox(g)
    r0, c0, r1, c1 = anchor_bbox
    assert c0 > 0 and c1 < g.shape[1] - 1, "anchor must not touch the left or right boundary"

    marked_rows = _collect_marked_rows(g, bg)
    for left, right in marked_rows.values():
        assert left in MARKER_COLORS and right in MARKER_COLORS, "markers must use red/green/yellow"

    unexpected = g != bg
    unexpected[r0 : r1 + 1, c0 : c1 + 1] = False
    for row in marked_rows:
        unexpected[row, 0] = False
        unexpected[row, g.shape[1] - 1] = False
    assert not np.any(unexpected), "unexpected foreground pixels outside anchor or boundary markers"

    same_rows = [row for row, (left, right) in marked_rows.items() if left == right]
    diff_rows = [row for row, (left, right) in marked_rows.items() if left != right]
    return {
        "bg": bg,
        "anchor_bbox": anchor_bbox,
        "marked_rows": marked_rows,
        "same_rows": same_rows,
        "diff_rows": diff_rows,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each boundary-marked row to either a full-row fill or a pair of rays converging toward the teal anchor.

    Input structure:
    - The background is black.
    - One teal 2x2 square acts as the anchor and does not touch the left or right border.
    - On several rows, the leftmost and rightmost cells contain one colored marker each, using red, green, or yellow.

    Rule:
    - For every row with boundary markers, compare the left and right marker colors.
    - If the two markers have the same color, fill that entire row with that color.
    - If the two markers have different colors, extend the left color rightward up to the column immediately left of the anchor and extend the right color leftward up to the column immediately right of the anchor.
    - Never paint over the teal 2x2 anchor; keep all unmarked rows as background.

    Key insight:
    - The boundary color pairing is not a filter: it is the branch selector that decides whether a row becomes a solid band or two inward-facing rays aimed at the anchor.
    """
    g = np.asarray(grid, dtype=int)
    info = _scene_info(g)
    bg = int(info["bg"])
    r0, c0, r1, c1 = info["anchor_bbox"]
    marked_rows: dict[int, tuple[int, int]] = info["marked_rows"]  # type: ignore[assignment]

    out = np.full_like(g, bg)
    out[r0 : r1 + 1, c0 : c1 + 1] = TEAL

    for row, (left, right) in marked_rows.items():
        if left == right:
            out[row, :] = left
        else:
            out[row, :c0] = left
            out[row, c1 + 1 :] = right

    out[r0 : r1 + 1, c0 : c1 + 1] = TEAL
    return out


def _choose_rows(available_rows: list[int], count: int) -> list[int]:
    assert count <= len(available_rows)
    for min_gap in (2, 1):
        for _ in range(ROW_PICK_ATTEMPTS):
            rows = sorted(random.sample(available_rows, count))
            if all(b - a >= min_gap for a, b in zip(rows, rows[1:])):
                return rows
    return sorted(random.sample(available_rows, count))


def _expected_row(
    width: int,
    row: int,
    anchor_bbox: BBox,
    bg: int,
    marked_rows: dict[int, tuple[int, int]],
) -> np.ndarray:
    r0, c0, r1, c1 = anchor_bbox
    expected = np.full(width, bg, dtype=int)
    if row in marked_rows:
        left, right = marked_rows[row]
        if left == right:
            expected[:] = left
        else:
            expected[:c0] = left
            expected[c1 + 1 :] = right
    if r0 <= row <= r1:
        expected[c0 : c1 + 1] = TEAL
    return expected


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    info = _scene_info(inp)
    bg = int(info["bg"])
    anchor_bbox = info["anchor_bbox"]
    marked_rows: dict[int, tuple[int, int]] = info["marked_rows"]  # type: ignore[assignment]
    same_rows: list[int] = info["same_rows"]  # type: ignore[assignment]
    diff_rows: list[int] = info["diff_rows"]  # type: ignore[assignment]
    r0, c0, r1, c1 = anchor_bbox
    height, width = map(int, inp.shape)

    assert out.shape == inp.shape
    assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == TEAL), "anchor must remain teal"
    for row in range(height):
        expected = _expected_row(width, row, anchor_bbox, bg, marked_rows)
        assert np.array_equal(out[row], expected), "output row violates the branch semantics"

    assert same_rows and diff_rows, "generated instances must expose both branches"
    assert any(np.all(out[row] == marked_rows[row][0]) for row in same_rows), "same-color rows must become solid bands"
    assert any(np.all(out[row, c0 : c1 + 1] == bg) for row in diff_rows), "different-color rows must leave the anchor columns open"


def _make_candidate() -> np.ndarray:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)

    anchor_row = random.randint(2, height - 4)
    anchor_col_min = max(4, width // 2)
    anchor_col_max = width - 4
    anchor_col = random.randint(anchor_col_min, anchor_col_max)
    grid[anchor_row : anchor_row + 2, anchor_col : anchor_col + 2] = TEAL

    available_rows = [row for row in range(height) if row not in (anchor_row, anchor_row + 1)]
    row_count = random.randint(MARKED_ROW_COUNT_RANGE[0], min(MARKED_ROW_COUNT_RANGE[1], len(available_rows)))
    marked_rows = _choose_rows(available_rows, row_count)

    same_count = random.randint(1, row_count - 1)
    diff_count = row_count - same_count
    branch_flags = [True] * same_count + [False] * diff_count
    random.shuffle(branch_flags)

    pairings: dict[int, tuple[int, int]] = {}
    for row, is_same in zip(marked_rows, branch_flags):
        if is_same:
            color = random.choice(MARKER_COLORS)
            pairings[row] = (color, color)
        else:
            left, right = random.sample(MARKER_COLORS, 2)
            pairings[row] = (left, right)

    for row, (left, right) in pairings.items():
        grid[row, 0] = left
        grid[row, width - 1] = right

    info = _scene_info(grid)
    assert 3 <= len(info["marked_rows"]) <= 5, "generator must place 3-5 marked rows"
    assert not {anchor_row, anchor_row + 1}.intersection(info["marked_rows"]), "marked rows may not overlap the anchor rows"
    assert info["same_rows"] and info["diff_rows"], "generator must include both branch types"
    assert anchor_col >= width // 2, "anchor should stay in the center-right half"
    assert len(set(pairings.values())) >= 2, "avoid degenerate repeated pairings"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _make_candidate()
        out = main(inp)
        _validate_output(inp, out)
        return inp
    raise RuntimeError("failed to generate a valid puzzle-6 input")
