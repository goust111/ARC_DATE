from __future__ import annotations

import numpy as np


BG = 0
MARKER = 9
FRAME_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)

_GEN_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0
_GEN_MAX_ATTEMPTS = 400
_GEN_FRAME_COUNT_PLAN = (3, 4, 3, 4, 3, 4)
_GEN_COUNT_PLAN = (
    (1, 3, 5),
    (1, 2, 4, 5),
    (2, 3, 4),
    (1, 3, 4, 5),
    (1, 2, 5),
    (2, 3, 4, 5),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _staircase_cells(inner_h: int, inner_w: int, levels: int, align: str = "left") -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    visible_rows = min(int(levels), int(inner_h))
    for layer in range(visible_rows):
        row = inner_h - 1 - layer
        width = min(levels - layer, inner_w)
        if width <= 0:
            continue
        if align == "left":
            cols = range(width)
        else:
            cols = range(inner_w - width, inner_w)
        for col in cols:
            cells.append((row, col))
    return cells


def _square_cells(inner_h: int, inner_w: int, levels: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    visible_rows = min(int(levels), int(inner_h))
    width = min(int(levels), int(inner_w))
    for layer in range(visible_rows):
        row = inner_h - 1 - layer
        for col in range(width):
            cells.append((row, col))
    return cells


def _render_frame_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        _render_frame_border(grid, spec["bbox"], spec["color"])
        for r, c in spec["markers"]:
            grid[r, c] = MARKER
    return grid


def _output_from_specs(
    shape: tuple[int, int],
    specs: list[dict],
    *,
    clear_markers: bool = True,
    align: str = "left",
    mode: str = "stair",
) -> np.ndarray:
    out = _render_input(shape, specs)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = int(spec["color"])
        count = int(spec["count"])
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        if clear_markers:
            inner[inner == MARKER] = BG
        if mode == "stair":
            cells = _staircase_cells(inner.shape[0], inner.shape[1], count, align=align)
        else:
            cells = _square_cells(inner.shape[0], inner.shape[1], count)
        for rr, cc in cells:
            inner[rr, cc] = color
    return out


def _find_frames(grid: np.ndarray) -> list[dict]:
    bg = _background_color(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) not in (bg, MARKER)]
    frames: list[dict] = []

    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "frame color must be present"
        r0, c0 = np.min(coords, axis=0)
        r1, c1 = np.max(coords, axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "frame must have a non-empty interior"
        assert int(coords.shape[0]) == 2 * (height + width) - 4, "frame must be a 1-pixel hollow rectangle"

        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(block[0, :] == color) and np.all(block[-1, :] == color)
        assert np.all(block[:, 0] == color) and np.all(block[:, -1] == color)
        assert not np.any(block[1:-1, 1:-1] == color), "frame color cannot appear in its own interior"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "inner_h": height - 2,
                "inner_w": width - 2,
            }
        )

    marker_coords = np.argwhere(grid == MARKER)
    for r_raw, c_raw in marker_coords:
        r = int(r_raw)
        c = int(c_raw)
        containing = [
            idx
            for idx, frame in enumerate(frames)
            if frame["bbox"][0] < r < frame["bbox"][2] and frame["bbox"][1] < c < frame["bbox"][3]
        ]
        assert len(containing) == 1, "each marker must belong to exactly one frame interior"

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the markers inside each hollow frame and replace them with a bottom-left staircase whose height equals that count.

    Input structure:
    - The grid has a dominant black background plus 3-4 non-overlapping hollow rectangular frames.
    - Each frame border is a unique non-black color with 1-pixel thickness.
    - Inside each frame there are black cells and one or more maroon marker pixels of color 9.

    Rule:
    - For each frame independently, count the color-9 markers strictly inside its border.
    - Erase those markers back to background.
    - In the frame interior, draw a left-aligned staircase in the frame's own color: the bottom row gets N cells, the next row up gets N-1, and so on until 1, where N is the marker count.
    - If the requested width exceeds the interior width, clip on the right; if the requested height exceeds the interior height, clip at the top. Leave the frame border and the rest of the background unchanged.

    Key insight:
    - The marker positions are irrelevant after detection: they only encode a count that controls how many stacked staircase layers to build inside the containing frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    bg = _background_color(g)
    out = g.copy()
    frames = _find_frames(g)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        inner_in = g[r0 + 1 : r1, c0 + 1 : c1]
        assert set(map(int, np.unique(inner_in))) <= {bg, MARKER}, "frame interior must contain only background and markers"
        count = int(np.count_nonzero(inner_in == MARKER))
        assert count >= 1, "each frame must contain at least one marker"

        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_out[inner_out == MARKER] = bg
        for rr, cc in _staircase_cells(inner_out.shape[0], inner_out.shape[1], count, align="left"):
            inner_out[rr, cc] = color

    return out


def _is_strictly_monotone(values: list[int]) -> bool:
    return all(a < b for a, b in zip(values, values[1:])) or all(a > b for a, b in zip(values, values[1:]))


def _sample_dimensions(counts: list[int], rng: np.random.Generator) -> list[tuple[int, int, int, int]] | None:
    for _ in range(200):
        dims: list[tuple[int, int, int, int]] = []
        has_exact_w = False
        has_exact_h = False
        has_wider_than_count = False
        has_two_axis_slack = False

        for count in counts:
            extra_h = int(rng.integers(0, 3))
            extra_w = int(rng.integers(0, 3))
            if count == 1 and extra_h == 0 and extra_w == 0:
                extra_w = 1

            inner_h = count + extra_h
            inner_w = count + extra_w
            dims.append((inner_h, inner_w, extra_h, extra_w))

            has_exact_w |= extra_w == 0
            has_exact_h |= extra_h == 0
            has_wider_than_count |= count > 1 and extra_w > 0
            has_two_axis_slack |= extra_h > 0 and extra_w > 0

        if has_exact_w and has_exact_h and has_wider_than_count and has_two_axis_slack:
            return dims
    return None


def _slot_requirements(specs: list[dict]) -> tuple[list[int], list[int]]:
    row_reqs = [0, 0]
    col_reqs = [0, 0]
    for spec in specs:
        slot = int(spec["slot"])
        row = slot // 2
        col = slot % 2
        bbox_h = int(spec["inner_h"]) + 2
        bbox_w = int(spec["inner_w"]) + 2
        row_reqs[row] = max(row_reqs[row], bbox_h)
        col_reqs[col] = max(col_reqs[col], bbox_w)
    return row_reqs, col_reqs


def _place_specs(specs: list[dict], rng: np.random.Generator) -> tuple[int, int] | None:
    row_reqs, col_reqs = _slot_requirements(specs)
    for _ in range(100):
        row_heights = [req + int(rng.integers(0, 3)) for req in row_reqs]
        col_widths = [req + int(rng.integers(0, 3)) for req in col_reqs]
        top = int(rng.integers(1, 3))
        bottom = int(rng.integers(1, 3))
        left = int(rng.integers(1, 3))
        right = int(rng.integers(1, 3))
        gutter_h = int(rng.integers(2, 4))
        gutter_w = int(rng.integers(2, 4))

        height = top + row_heights[0] + gutter_h + row_heights[1] + bottom
        width = left + col_widths[0] + gutter_w + col_widths[1] + right
        if max(height, width) > 30:
            continue

        row_starts = [top, top + row_heights[0] + gutter_h]
        col_starts = [left, left + col_widths[0] + gutter_w]

        for spec in specs:
            slot = int(spec["slot"])
            row = slot // 2
            col = slot % 2
            bbox_h = int(spec["inner_h"]) + 2
            bbox_w = int(spec["inner_w"]) + 2
            slot_h = row_heights[row]
            slot_w = col_widths[col]
            assert slot_h >= bbox_h and slot_w >= bbox_w
            r0 = row_starts[row] + int(rng.integers(0, slot_h - bbox_h + 1))
            c0 = col_starts[col] + int(rng.integers(0, slot_w - bbox_w + 1))
            spec["bbox"] = (r0, c0, r0 + bbox_h - 1, c0 + bbox_w - 1)

        return height, width
    return None


def _sample_markers(spec: dict, rng: np.random.Generator) -> list[tuple[int, int]]:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    count = int(spec["count"])
    r0, c0, _, _ = spec["bbox"]

    all_cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    stair = set(_staircase_cells(inner_h, inner_w, count, align="left"))
    inside = [cell for cell in all_cells if cell in stair]
    outside = [cell for cell in all_cells if cell not in stair]

    chosen: list[tuple[int, int]] = []
    if count == 1 and outside:
        chosen.append(outside[int(rng.integers(len(outside)))])
    elif count > 1:
        chosen.append(inside[int(rng.integers(len(inside)))])
        if outside:
            outside_idx = int(rng.integers(len(outside)))
            chosen.append(outside[outside_idx])

    remaining_pool = [cell for cell in all_cells if cell not in chosen]
    if count - len(chosen) > len(remaining_pool):
        raise AssertionError("not enough interior cells for the requested marker count")

    if count > len(chosen):
        take = rng.choice(len(remaining_pool), size=count - len(chosen), replace=False)
        for idx in np.atleast_1d(take):
            chosen.append(remaining_pool[int(idx)])

    return [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in chosen]


def _validate_specs_against_rendered(grid: np.ndarray, specs: list[dict], expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if grid.shape != expected.shape or max(grid.shape) > 30:
        return False
    if not np.array_equal(grid, _render_input(grid.shape, specs)):
        return False

    counts = [int(spec["count"]) for spec in specs]
    colors = [int(spec["color"]) for spec in specs]
    if len(counts) != len(set(counts)) or len(colors) != len(set(colors)):
        return False

    row_major_counts = [
        int(spec["count"])
        for spec in sorted(specs, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    ]
    if _is_strictly_monotone(row_major_counts):
        return False

    marker_outside_stair = False
    exact_w = False
    exact_h = False
    wider_than_count = False

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = int(spec["color"])
        count = int(spec["count"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])

        if not (0 <= r0 < r1 < grid.shape[0] and 0 <= c0 < c1 < grid.shape[1]):
            return False
        if inner_h < count or inner_w < count:
            return False

        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(block[0, :] == color) or not np.all(block[-1, :] == color):
            return False
        if not np.all(block[:, 0] == color) or not np.all(block[:, -1] == color):
            return False
        if np.any(block[1:-1, 1:-1] == color):
            return False
        if set(map(int, np.unique(block[1:-1, 1:-1]))) - {BG, MARKER}:
            return False

        markers = list(spec["markers"])
        if len(markers) != count or len(markers) != len(set(markers)):
            return False
        if int(np.count_nonzero(block[1:-1, 1:-1] == MARKER)) != count:
            return False

        stair = {
            (r0 + 1 + rr, c0 + 1 + cc)
            for rr, cc in _staircase_cells(inner_h, inner_w, count, align="left")
        }
        marker_outside_stair |= any(marker not in stair for marker in markers)
        exact_w |= inner_w == count
        exact_h |= inner_h == count
        wider_than_count |= count > 1 and inner_w > count

    if not marker_outside_stair or not exact_w or not exact_h or not wider_than_count:
        return False

    wrong_keep_markers = _output_from_specs(grid.shape, specs, clear_markers=False)
    wrong_right_align = _output_from_specs(grid.shape, specs, clear_markers=True, align="right")
    wrong_square = _output_from_specs(grid.shape, specs, clear_markers=True, mode="square")
    if np.array_equal(expected, wrong_keep_markers):
        return False
    if np.array_equal(expected, wrong_right_align):
        return False
    if np.array_equal(expected, wrong_square):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _build_specs_for_call(call_index: int, rng: np.random.Generator) -> tuple[tuple[int, int], list[dict]] | None:
    frame_count = _GEN_FRAME_COUNT_PLAN[call_index % len(_GEN_FRAME_COUNT_PLAN)]
    base_counts = list(_GEN_COUNT_PLAN[call_index % len(_GEN_COUNT_PLAN)])
    assert len(base_counts) == frame_count

    slots = [0, 1, 2, 3]
    if frame_count == 3:
        missing_slot = int((call_index + int(rng.integers(0, 4))) % 4)
        slots.remove(missing_slot)

    dims = _sample_dimensions(base_counts, rng)
    if dims is None:
        return None

    colors = [int(c) for c in rng.choice(FRAME_COLORS, size=frame_count, replace=False)]
    slot_order = np.array(slots, dtype=int)

    for _ in range(80):
        count_perm = rng.permutation(frame_count)
        row_major_counts = [base_counts[idx] for idx in count_perm[np.argsort(slot_order)]]
        if not _is_strictly_monotone(list(row_major_counts)):
            break
    else:
        return None

    specs: list[dict] = []
    for spec_idx, perm_idx_raw in enumerate(count_perm):
        perm_idx = int(perm_idx_raw)
        count = int(base_counts[perm_idx])
        inner_h, inner_w, extra_h, extra_w = dims[perm_idx]
        specs.append(
            {
                "slot": int(slots[spec_idx]),
                "count": count,
                "color": colors[spec_idx],
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "extra_h": int(extra_h),
                "extra_w": int(extra_w),
            }
        )

    shape = _place_specs(specs, rng)
    if shape is None:
        return None

    for spec in specs:
        spec["markers"] = _sample_markers(spec, rng)

    return shape, specs


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    for _ in range(_GEN_MAX_ATTEMPTS):
        built = _build_specs_for_call(call_index, _GEN_RNG)
        if built is None:
            continue
        shape, specs = built
        grid = _render_input(shape, specs)
        expected = _output_from_specs(shape, specs)
        if _validate_specs_against_rendered(grid, specs, expected):
            return grid

    raise RuntimeError("failed to generate a valid staircase-from-marker-count input")
