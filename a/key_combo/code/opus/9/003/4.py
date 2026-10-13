from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_shared", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()
Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
random_free_location = _ARC_UTILS.random_free_location


FRAME_SIZE = 5
INNER_SIZE = 3
TABLE_COLS = 3

ROW_LABEL_POOL = [Color.BLUE, Color.YELLOW, Color.ORANGE, Color.TEAL, Color.MAROON]
PATTERN_COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.TEAL, Color.MAROON]
FRAME_PATCH = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=int)
GEN_RETRIES = 400

MASK_LIBRARY = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 2), (2, 0), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 2), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
)


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(arr, dtype=int).ravel())


def _render_pattern(mask_coords: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    pattern = np.zeros((INNER_SIZE, INNER_SIZE), dtype=int)
    for r, c in mask_coords:
        pattern[r, c] = int(color)
    return pattern


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == Color.GREY)]
    assert len(cols) == 1, f"expected a unique full-height grey separator, got {cols}"
    return int(cols[0])


def _frame_border_mask() -> np.ndarray:
    mask = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


FRAME_BORDER_MASK = _frame_border_mask()


def _collect_frames(grid: np.ndarray, sep_c: int) -> list[dict]:
    frames = []
    for comp in find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True):
        if int(comp["color"]) != Color.GREY:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if c1 >= sep_c:
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) != (FRAME_SIZE, FRAME_SIZE):
            continue
        patch = crop_bbox(grid, comp["bbox"])
        if not np.array_equal(patch == Color.GREY, FRAME_BORDER_MASK):
            continue
        interior = patch[1:-1, 1:-1]
        assert interior.shape == (INNER_SIZE, INNER_SIZE)
        assert np.any(interior != Color.BLACK), "frame interior pattern must not be empty"
        frames.append({"bbox": comp["bbox"], "pattern": interior.copy()})
    frames.sort(key=lambda item: tuple(int(x) for x in item["bbox"]))
    assert 3 <= len(frames) <= 4, f"expected 3 or 4 left frames, got {len(frames)}"
    return frames


def _parse_reference_table(grid: np.ndarray, sep_c: int) -> tuple[list[int], dict[tuple[int, ...], tuple[int, int]], int]:
    pink_bbox = bbox(grid == Color.PINK)
    assert pink_bbox is not None, "reference table grid lines must be present"
    r0, c0, r1, c1 = pink_bbox
    assert c0 > sep_c, "reference table must lie to the right of the separator"

    table = crop_bbox(grid, pink_bbox)
    row_lines = [r for r in range(table.shape[0]) if np.all(table[r, :] == Color.PINK)]
    col_lines = [c for c in range(table.shape[1]) if np.all(table[:, c] == Color.PINK)]
    assert len(col_lines) == TABLE_COLS + 1, f"expected {TABLE_COLS} table columns"
    assert len(row_lines) in (4, 5), "expected a 3x3 or 4x3 reference table"
    assert all(b - a == 4 for a, b in zip(row_lines, row_lines[1:])), "table row height must be 3 with separator lines"
    assert all(b - a == 4 for a, b in zip(col_lines, col_lines[1:])), "table col width must be 3 with separator lines"

    row_labels: list[int] = []
    lookup: dict[tuple[int, ...], tuple[int, int]] = {}
    for ri in range(len(row_lines) - 1):
        label_band = grid[r0 + row_lines[ri] + 1 : r0 + row_lines[ri + 1], sep_c + 1 : c0]
        label_pixels = np.argwhere((label_band != Color.BLACK) & (label_band != Color.PINK))
        assert label_pixels.shape[0] == 1, f"row {ri} must have exactly one label pixel"
        label_color = int(label_band[tuple(label_pixels[0])])
        row_labels.append(label_color)

        for ci in range(TABLE_COLS):
            rr = row_lines[ri]
            cc = col_lines[ci]
            cell = table[rr + 1 : rr + 4, cc + 1 : cc + 4].copy()
            key = _pattern_key(cell)
            assert key not in lookup, "reference table patterns must be unique"
            lookup[key] = (ri, ci)
    return row_labels, lookup, len(row_lines) - 1


def _is_complete_line(matches: list[tuple[int, int]], nrows: int) -> bool:
    rows = {r for r, _ in matches}
    cols = {c for _, c in matches}
    if len(matches) == TABLE_COLS and len(rows) == 1 and cols == set(range(TABLE_COLS)):
        return True
    if len(matches) == nrows and len(cols) == 1 and rows == set(range(nrows)):
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each left-frame interior pattern, use it as a key into the right reference table, then recolor the frame and globally refill the interiors.

    Input structure:
    - The grid has a black background, a unique full-height grey separator column, 3 or 4 grey hollow 5x5 frames on the left, and a magenta-lined 3x3 or 4x3 reference table on the right.
    - Each left frame contains a 3x3 interior pattern, and each reference-table cell also contains one 3x3 pattern.
    - Every table row has exactly one colored label pixel placed immediately to the left of that row.

    Rule:
    - Extract the strict 3x3 interior of each left frame and match it exactly against the reference-table cells.
    - Recolor that frame's grey border with the label color of the matched table row.
    - If the matched cells are exactly one complete table row or one complete table column, refill every left-frame interior with green; otherwise refill every interior with red.
    - Keep the separator and the entire reference table unchanged.

    Key insight:
    - The cropped 3x3 content is not the answer itself; it is a lookup key whose matched table position drives both a local border recolor and a global all-frames verdict.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_c = _find_separator_col(g)
    frames = _collect_frames(g, sep_c)
    row_labels, lookup, nrows = _parse_reference_table(g, sep_c)

    matches: list[tuple[int, int]] = []
    frame_rows: list[int] = []
    seen = set()
    for frame in frames:
        key = _pattern_key(frame["pattern"])
        assert key in lookup, "every left-frame pattern must appear in the reference table"
        pos = lookup[key]
        assert pos not in seen, "left frames must match distinct reference cells"
        seen.add(pos)
        matches.append(pos)
        frame_rows.append(pos[0])

    fill_color = Color.GREEN if _is_complete_line(matches, nrows) else Color.RED
    out = g.copy()
    for frame, row_idx in zip(frames, frame_rows, strict=False):
        r0, c0, r1, c1 = frame["bbox"]
        border_color = row_labels[row_idx]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[FRAME_BORDER_MASK] = border_color
        patch[1:-1, 1:-1] = fill_color
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _sample_patterns(count: int) -> list[np.ndarray]:
    patterns = [_render_pattern(mask, color) for mask in MASK_LIBRARY for color in PATTERN_COLOR_POOL]
    random.shuffle(patterns)
    chosen = patterns[:count]
    assert len({_pattern_key(p) for p in chosen}) == count
    return chosen


def _sample_match_positions(nrows: int, branch: str) -> list[tuple[int, int]]:
    all_positions = [(r, c) for r in range(nrows) for c in range(TABLE_COLS)]
    if branch == "row":
        row = random.randrange(nrows)
        return [(row, c) for c in range(TABLE_COLS)]
    if branch == "col":
        col = random.randrange(TABLE_COLS)
        return [(r, col) for r in range(nrows)]

    counts = [3, 4] if len(all_positions) >= 4 else [3]
    for _ in range(200):
        positions = random.sample(all_positions, k=random.choice(counts))
        if not _is_complete_line(positions, nrows):
            return positions
    raise RuntimeError("failed to sample a negative branch that is not a full row/column")


def _place_frames(height: int, left_w: int, frame_count: int) -> list[tuple[int, int]] | None:
    occ = np.zeros((height, left_w), dtype=int)
    positions: list[tuple[int, int]] = []
    for _ in range(frame_count):
        try:
            loc = random_free_location(
                occ,
                FRAME_PATCH,
                transparent=0,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except Exception:
            return None
        if loc is None:
            return None
        r0, c0 = loc
        if c0 + FRAME_SIZE > left_w - 1:
            return None
        occ[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = 1
        positions.append((int(r0), int(c0)))
    return positions


def _draw_frame(grid: np.ndarray, r0: int, c0: int, pattern: np.ndarray) -> None:
    grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE][FRAME_BORDER_MASK] = Color.GREY
    grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = pattern


def _build_expected(
    grid: np.ndarray,
    frame_positions: list[tuple[int, int]],
    matched_positions: list[tuple[int, int]],
    row_labels: list[int],
    nrows: int,
) -> np.ndarray:
    out = grid.copy()
    fill_color = Color.GREEN if _is_complete_line(matched_positions, nrows) else Color.RED
    for (r0, c0), (row_idx, _col_idx) in zip(frame_positions, matched_positions, strict=False):
        patch = out[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE]
        patch[FRAME_BORDER_MASK] = row_labels[row_idx]
        patch[1:-1, 1:-1] = fill_color
    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray, dict] | None:
    branch = random.choice(["other", "other", "other", "other", "other", "row", "row", "col", "col"])
    nrows = random.choice([3, 4])
    if branch == "col":
        frame_count = nrows
    elif branch == "row":
        frame_count = TABLE_COLS
    else:
        frame_count = random.choice([3, 4])

    table_h = 4 * nrows + 1
    height = random.randint(max(20, table_h + 2), 29)
    left_w = random.randint(11 if frame_count == 4 else 9, 13)
    right_w = 16
    width = left_w + 1 + right_w
    assert width <= 30

    frame_positions = _place_frames(height, left_w, frame_count)
    if frame_positions is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    sep_c = left_w
    grid[:, sep_c] = Color.GREY

    row_labels = random.sample(ROW_LABEL_POOL, k=nrows)
    patterns = _sample_patterns(nrows * TABLE_COLS)
    table_patterns = np.empty((nrows, TABLE_COLS, INNER_SIZE, INNER_SIZE), dtype=int)
    idx = 0
    for r in range(nrows):
        for c in range(TABLE_COLS):
            table_patterns[r, c] = patterns[idx]
            idx += 1

    top = random.randint(1, height - table_h - 1)
    table_c0 = sep_c + 4
    for rr in range(nrows + 1):
        grid[top + 4 * rr, table_c0 : table_c0 + 13] = Color.PINK
    for cc in range(TABLE_COLS + 1):
        grid[top : top + table_h, table_c0 + 4 * cc] = Color.PINK
    for r in range(nrows):
        grid[top + 4 * r + 2, table_c0 - 2] = row_labels[r]
        for c in range(TABLE_COLS):
            grid[top + 4 * r + 1 : top + 4 * r + 4, table_c0 + 4 * c + 1 : table_c0 + 4 * c + 4] = table_patterns[r, c]

    matched_positions = _sample_match_positions(nrows, branch)
    random.shuffle(matched_positions)
    for (r0, c0), (tr, tc) in zip(frame_positions, matched_positions, strict=False):
        _draw_frame(grid, r0, c0, table_patterns[tr, tc])

    expected = _build_expected(grid, frame_positions, matched_positions, row_labels, nrows)
    meta = {
        "branch": branch,
        "matched_positions": list(matched_positions),
        "row_labels": list(row_labels),
        "nrows": nrows,
    }
    return grid, expected, meta


def _validate_generated(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    sep_c = _find_separator_col(grid)
    frames = _collect_frames(grid, sep_c)
    row_labels, lookup, nrows = _parse_reference_table(grid, sep_c)
    if nrows != meta["nrows"] or row_labels != meta["row_labels"]:
        return False

    parsed_matches = []
    for frame in frames:
        key = _pattern_key(frame["pattern"])
        if key not in lookup:
            return False
        parsed_matches.append(lookup[key])
    if set(parsed_matches) != set(meta["matched_positions"]):
        return False

    green_expected = meta["branch"] in {"row", "col"}
    if _is_complete_line(parsed_matches, nrows) != green_expected:
        return False

    fill_color = Color.GREEN if green_expected else Color.RED
    out = main(grid)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if not np.all(out[r0 + 1 : r1, c0 + 1 : c1] == fill_color):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_instance()
        if built is None:
            continue
        grid, expected, meta = built
        if _validate_generated(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")


if __name__ == "__main__":
    p = Path(__file__)
    mod = SourceFileLoader("arc_new_task_self", str(p)).load_module()
    for _ in range(10):
        sample = mod.generate_input()
        assert np.array_equal(mod.main(sample), mod.main(sample))
