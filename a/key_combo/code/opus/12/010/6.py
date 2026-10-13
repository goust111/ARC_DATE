from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
MARKER_COLOR = 1
FRAME_COLOR = 5

GRID_ROWS_RANGE = (18, 24)
GRID_COLS_RANGE = (18, 24)
FRAME_COUNT_RANGE = (4, 6)
SMALL_INTERIORS = ((2, 3), (3, 2), (2, 4), (4, 2))
LARGE_INNER_SIDE_RANGE = (3, 7)
PLACEMENT_GAP = 1
PLACEMENT_TRIES_PER_FRAME = 200
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _stamp_cells(inner_h: int, inner_w: int, marker: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    mr, mc = map(int, marker)
    rr0 = max(0, mr - int(radius))
    rr1 = min(int(inner_h) - 1, mr + int(radius))
    cc0 = max(0, mc - int(radius))
    cc1 = min(int(inner_w) - 1, mc + int(radius))
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    grey_comps = _components(grid == FRAME_COLOR)
    assert grey_comps, "expected at least one grey frame"

    frames: list[dict[str, object]] = []
    for comp in grey_comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "frame must enclose a non-empty interior"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert len(comp) == int(border.sum()), "grey component must be a rectangular border"
        assert np.all(patch[border] == FRAME_COLOR), "frame border must be solid grey"

        interior = patch[1:-1, 1:-1]
        allowed = {int(bg), MARKER_COLOR}
        assert set(map(int, interior.ravel())).issubset(allowed), "frame interior may contain only background and one marker"
        markers = np.argwhere(interior == MARKER_COLOR)
        assert markers.shape[0] == 1, "each frame must contain exactly one blue marker"
        mr, mc = map(int, markers[0])

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_shape": (int(interior.shape[0]), int(interior.shape[1])),
                "marker": (mr, mc),
                "area": int(interior.size),
            }
        )

    marker_count = int(np.count_nonzero(grid == MARKER_COLOR))
    assert marker_count == len(frames), "all blue pixels must be frame markers"
    frames.sort(key=lambda rec: tuple(int(v) for v in rec["bbox"]))
    return frames


def _apply_rule(
    grid: np.ndarray,
    bg: int,
    radius_fn,
) -> np.ndarray:
    out = grid.copy()
    for frame in _extract_frames(grid, bg):
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        inner_h, inner_w = map(int, frame["inner_shape"])
        mr, mc = map(int, frame["marker"])
        radius = int(radius_fn(int(frame["area"])))
        for ir, ic in _stamp_cells(inner_h, inner_w, (mr, mc), radius):
            gr = r0 + 1 + ir
            gc = c0 + 1 + ic
            if int(grid[gr, gc]) == int(bg):
                out[gr, gc] = MARKER_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each framed marker with a blue square whose radius is chosen by the frame's interior area.

    Input structure:
    - The grid has a dominant black background and several disjoint one-pixel-thick grey rectangular frames.
    - Each frame encloses a black interior and contains exactly one blue marker pixel inside that interior.
    - Frames do not overlap, and blue pixels occur only as these interior markers.

    Rule:
    - For each frame, compute the interior area as interior_height times interior_width, excluding the grey border.
    - If that area is at most 9, use the blue marker as the center of a 3x3 stamp; otherwise use it as the center of a 5x5 stamp.
    - Apply the chosen stamp only inside the frame interior, clipping any part that would cross the grey border, and recolor only black background cells to blue.
    - Preserve the grey border and preserve the original blue marker pixel.

    Key insight:
    - The marker gives the stamp center, the frame geometry clips the stamp, and the interior area alone decides whether the expansion radius is 1 or 2.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    return _apply_rule(g, bg, lambda area: 1 if int(area) <= 9 else 2)


def _boxes_clear(candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + PLACEMENT_GAP < pr0 or pr1 + PLACEMENT_GAP < r0 or c1 + PLACEMENT_GAP < pc0 or pc1 + PLACEMENT_GAP < c0):
            return False
    return True


def _small_frame_spec() -> dict[str, object]:
    inner_h, inner_w = random.choice(SMALL_INTERIORS)
    gainful = [
        (r, c)
        for r in range(inner_h)
        for c in range(inner_w)
        if _stamp_cells(inner_h, inner_w, (r, c), 1) != _stamp_cells(inner_h, inner_w, (r, c), 2)
    ]
    marker = random.choice(gainful)
    return {
        "inner_shape": (inner_h, inner_w),
        "outer_shape": (inner_h + 2, inner_w + 2),
        "marker": marker,
        "branch": "small",
    }


def _large_frame_spec() -> dict[str, object]:
    while True:
        inner_h = random.randint(*LARGE_INNER_SIDE_RANGE)
        inner_w = random.randint(*LARGE_INNER_SIDE_RANGE)
        if inner_h * inner_w <= 9:
            continue

        preferred = [
            (r, c)
            for r in range(inner_h)
            for c in range(inner_w)
            if (r < 2 or r > inner_h - 3 or c < 2 or c > inner_w - 3)
            and (2 * r != inner_h - 1 or 2 * c != inner_w - 1)
        ]
        if not preferred:
            continue
        marker = random.choice(preferred)
        return {
            "inner_shape": (inner_h, inner_w),
            "outer_shape": (inner_h + 2, inner_w + 2),
            "marker": marker,
            "branch": "large",
        }


def _sample_specs() -> list[dict[str, object]]:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    small_count = 1 if frame_count == 4 else random.choice([1, 2])
    small_count = min(small_count, frame_count - 1)
    large_count = frame_count - small_count

    specs = [_small_frame_spec() for _ in range(small_count)]
    specs.extend(_large_frame_spec() for _ in range(large_count))
    random.shuffle(specs)
    return specs


def _place_specs(rows: int, cols: int, specs: list[dict[str, object]]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(
        range(len(specs)),
        key=lambda i: int(specs[i]["outer_shape"][0]) * int(specs[i]["outer_shape"][1]),
        reverse=True,
    )
    placed: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    used_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        outer_h, outer_w = map(int, specs[idx]["outer_shape"])
        row_choices = list(range(rows - outer_h + 1))
        col_choices = list(range(cols - outer_w + 1))
        if not row_choices or not col_choices:
            return None

        placed_here = False
        for _ in range(PLACEMENT_TRIES_PER_FRAME):
            r0 = random.choice(row_choices)
            c0 = random.choice(col_choices)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _boxes_clear(box, used_boxes):
                continue
            used_boxes.append(box)
            placed[idx] = box
            placed_here = True
            break
        if not placed_here:
            return None

    return [box for box in placed if box is not None]


def _render_input(rows: int, cols: int, specs: list[dict[str, object]], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    for spec, (r0, c0, r1, c1) in zip(specs, boxes):
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        mr, mc = map(int, spec["marker"])
        grid[r0 + 1 + mr, c0 + 1 + mc] = MARKER_COLOR
    return grid


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape:
        return False
    if np.array_equal(inp, out):
        return False
    if _most_common_color(inp) != BACKGROUND:
        return False

    frames = _extract_frames(inp, BACKGROUND)
    if not (FRAME_COUNT_RANGE[0] <= len(frames) <= FRAME_COUNT_RANGE[1]):
        return False

    small_visible = False
    large_clipped = False
    large_off_center = False
    blue_growth = int(np.count_nonzero(out == MARKER_COLOR)) > int(np.count_nonzero(inp == MARKER_COLOR))

    for frame in frames:
        inner_h, inner_w = map(int, frame["inner_shape"])
        marker = tuple(map(int, frame["marker"]))
        area = int(frame["area"])
        cells1 = _stamp_cells(inner_h, inner_w, marker, 1)
        cells2 = _stamp_cells(inner_h, inner_w, marker, 2)
        if area <= 9 and cells1 != cells2:
            small_visible = True
        if area > 9:
            mr, mc = marker
            if mr < 2 or mr > inner_h - 3 or mc < 2 or mc > inner_w - 3:
                large_clipped = True
            if 2 * mr != inner_h - 1 or 2 * mc != inner_w - 1:
                large_off_center = True

        r0, c0, r1, c1 = map(int, frame["bbox"])
        out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(out_patch[border] == FRAME_COLOR):
            return False

    if not (small_visible and large_clipped and large_off_center and blue_growth):
        return False

    all_small = _apply_rule(inp, BACKGROUND, lambda _area: 1)
    all_large = _apply_rule(inp, BACKGROUND, lambda _area: 2)
    if np.array_equal(out, all_small):
        return False
    if np.array_equal(out, all_large):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    specs = _sample_specs()
    boxes = _place_specs(rows, cols, specs)
    if boxes is None:
        return None

    grid = _render_input(rows, cols, specs, boxes)
    expected = _apply_rule(grid, BACKGROUND, lambda area: 1 if int(area) <= 9 else 2)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not _semantic_check(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid framed-area-stamp sample")
