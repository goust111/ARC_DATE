from __future__ import annotations

from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

FILL_BY_CHILDREN = {
    0: GREEN,
    1: YELLOW,
}

Rect = tuple[int, int, int, int]

GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _rect_h(rect: Rect) -> int:
    return rect[2] - rect[0] + 1


def _rect_w(rect: Rect) -> int:
    return rect[3] - rect[1] + 1


def _rect_area(rect: Rect) -> int:
    return _rect_h(rect) * _rect_w(rect)


def _draw_frame(grid: np.ndarray, rect: Rect, color: int = RED) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _grey_line_positions(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_threshold = max(4, int(np.ceil(0.30 * w)))
    col_threshold = max(4, int(np.ceil(0.30 * h)))
    rows = [r for r in range(h) if int(np.count_nonzero(grid[r, :] == GREY)) >= row_threshold]
    cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == GREY)) >= col_threshold]
    assert len(rows) >= 4 and len(cols) >= 4, "expected a regular grey lattice"
    if len(rows) >= 3:
        assert len(set(np.diff(rows))) == 1, "horizontal lattice lines must be regularly spaced"
    if len(cols) >= 3:
        assert len(set(np.diff(cols))) == 1, "vertical lattice lines must be regularly spaced"
    return rows, cols


def _crop_bbox_from_holes(grid: np.ndarray) -> Rect:
    rows, cols = _grey_line_positions(grid)
    holes = [(r, c) for r in rows for c in cols if int(grid[r, c]) == BLACK]
    assert len(holes) == 4, "expected exactly four missing grey intersections"
    hole_rows = sorted({r for r, _ in holes})
    hole_cols = sorted({c for _, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2, "holes must be the four corners of one rectangle"
    r0, r1 = hole_rows
    c0, c1 = hole_cols
    assert set(holes) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return r0, c0, r1, c1


def _rect_from_component(comp: list[tuple[int, int]]) -> Rect:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "red frames must have non-empty interiors"

    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    expected = 2 * (box_h + box_w) - 4
    assert len(comp) == expected, "every red component must be a one-pixel-thick rectangular border"

    local = {(r - r0, c - c0) for r, c in comp}
    border = {
        (rr, cc)
        for rr in range(box_h)
        for cc in range(box_w)
        if rr in (0, box_h - 1) or cc in (0, box_w - 1)
    }
    assert local == border, "red components must be hollow axis-aligned rectangles"
    return r0, c0, r1, c1


def _find_frames(crop: np.ndarray) -> list[Rect]:
    comps = _components4(crop == RED)
    frames = [_rect_from_component(comp) for comp in comps]
    assert frames, "expected at least one red rectangular frame in the cropped region"
    return frames


def _contains(outer: Rect, inner: Rect) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _direct_child_counts(frames: list[Rect]) -> dict[Rect, int]:
    counts: dict[Rect, int] = {}
    for outer in frames:
        children = []
        for inner in frames:
            if inner == outer or not _contains(outer, inner):
                continue
            if any(mid not in (outer, inner) and _contains(outer, mid) and _contains(mid, inner) for mid in frames):
                continue
            children.append(inner)
        counts[outer] = len(children)
    return counts


def _filled_crop(crop: np.ndarray, frames: list[Rect]) -> np.ndarray:
    out = crop.copy()
    child_counts = _direct_child_counts(frames)
    for rect in sorted(frames, key=_rect_area):
        r0, c0, r1, c1 = rect
        fill_color = FILL_BY_CHILDREN.get(child_counts[rect], BLUE)
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the hole-marked lattice region, then fill each red frame by the number of directly nested frames it contains.

    Input structure:
    - A black background is overlaid with a one-pixel-thick regular grey lattice.
    - Exactly four lattice intersections are turned back to black, and those four holes are the corners of one axis-aligned crop rectangle.
    - Inside that hole-bounded region are one or more one-pixel-thick red hollow rectangles, possibly nested; outside the region there may be distracting red rectangles.

    Rule:
    - Find the grey lattice rows and columns, then locate the four black intersection holes and crop the inclusive rectangle that they mark.
    - Inside the cropped region, detect every red hollow rectangular frame.
    - For each frame, count how many other red frames lie directly inside it, ignoring deeper descendants.
    - Fill only the black cells of each frame interior, processing frames from inner to outer: 0 direct children -> green, 1 direct child -> yellow, 2 or more direct children -> blue.

    Key insight:
    - The four black lattice holes do not mark objects to fill; they mark the only region that matters, and the final fill color comes from direct containment structure among the cropped red frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    r0, c0, r1, c1 = _crop_bbox_from_holes(g)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    frames = _find_frames(crop)
    return _filled_crop(crop, frames)


def _random_rect_in_region(
    region: Rect,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    rng: np.random.Generator,
) -> Rect | None:
    r0, c0, r1, c1 = region
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    max_h = min(max_h, region_h)
    max_w = min(max_w, region_w)
    if max_h < min_h or max_w < min_w:
        return None
    h = int(rng.integers(min_h, max_h + 1))
    w = int(rng.integers(min_w, max_w + 1))
    top = int(rng.integers(r0, r1 - h + 2))
    left = int(rng.integers(c0, c1 - w + 2))
    return top, left, top + h - 1, left + w - 1


def _sample_outer_rect(crop_h: int, crop_w: int, min_h: int, min_w: int, rng: np.random.Generator) -> Rect | None:
    return _random_rect_in_region((1, 1, crop_h - 2, crop_w - 2), min_h, crop_h - 2, min_w, crop_w - 2, rng)


def _sample_nested_child(
    parent: Rect,
    min_h: int,
    min_w: int,
    max_h: int,
    max_w: int,
    rng: np.random.Generator,
) -> Rect | None:
    inner_region = (parent[0] + 2, parent[1] + 2, parent[2] - 2, parent[3] - 2)
    return _random_rect_in_region(inner_region, min_h, max_h, min_w, max_w, rng)


def _separated(a: Rect, b: Rect, gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _build_chain(crop_h: int, crop_w: int, levels: int, rng: np.random.Generator) -> list[Rect] | None:
    outer_min = 11 if levels == 3 else 7
    outer = _sample_outer_rect(crop_h, crop_w, outer_min, outer_min, rng)
    if outer is None:
        return None

    if levels == 2:
        child = _sample_nested_child(outer, 3, 3, min(5, _rect_h(outer) - 4), min(5, _rect_w(outer) - 4), rng)
        return None if child is None else [outer, child]

    mid = _sample_nested_child(outer, 7, 7, min(9, _rect_h(outer) - 4), min(9, _rect_w(outer) - 4), rng)
    if mid is None:
        return None
    inner = _sample_nested_child(mid, 3, 3, min(5, _rect_h(mid) - 4), min(5, _rect_w(mid) - 4), rng)
    return None if inner is None else [outer, mid, inner]


def _build_binary(crop_h: int, crop_w: int, rng: np.random.Generator) -> list[Rect] | None:
    shape_options: list[tuple[int, int]] = []
    if crop_h >= 7 and crop_w >= 11:
        shape_options.append((7, 11))
    if crop_h >= 11 and crop_w >= 7:
        shape_options.append((11, 7))
    if crop_h >= 11 and crop_w >= 11:
        shape_options.append((11, 11))
    if not shape_options:
        return None

    min_h, min_w = shape_options[int(rng.integers(0, len(shape_options)))]
    outer = _sample_outer_rect(crop_h, crop_w, min_h, min_w, rng)
    if outer is None:
        return None

    allowed = (outer[0] + 2, outer[1] + 2, outer[2] - 2, outer[3] - 2)
    children: list[Rect] = []
    for _ in range(200):
        first = _random_rect_in_region(allowed, 3, 5, 3, 5, rng)
        second = _random_rect_in_region(allowed, 3, 5, 3, 5, rng)
        if first is None or second is None:
            return None
        if _contains(first, second) or _contains(second, first):
            continue
        if not _separated(first, second, gap=1):
            continue
        children = [first, second]
        break
    return None if not children else [outer, *children]


def _build_mixed(crop_h: int, crop_w: int, rng: np.random.Generator) -> list[Rect] | None:
    outer = _sample_outer_rect(crop_h, crop_w, 11, 11, rng)
    if outer is None:
        return None
    mid = _sample_nested_child(outer, 7, 7, min(9, _rect_h(outer) - 4), min(9, _rect_w(outer) - 4), rng)
    if mid is None:
        return None
    inner = _sample_nested_child(mid, 3, 3, min(5, _rect_h(mid) - 4), min(5, _rect_w(mid) - 4), rng)
    if inner is None:
        return None

    allowed = (outer[0] + 2, outer[1] + 2, outer[2] - 2, outer[3] - 2)
    sibling = None
    for _ in range(200):
        candidate = _random_rect_in_region(allowed, 3, 5, 3, 5, rng)
        if candidate is None:
            return None
        if _contains(mid, candidate) or _contains(candidate, mid):
            continue
        if not _separated(candidate, mid, gap=1):
            continue
        sibling = candidate
        break
    return None if sibling is None else [outer, mid, inner, sibling]


def _frame_layout(crop_h: int, crop_w: int, rng: np.random.Generator) -> list[Rect] | None:
    builders: list[str] = []
    if crop_h >= 11 and crop_w >= 11:
        builders.extend(["mixed", "mixed", "binary", "chain3", "chain2"])
    elif crop_h >= 7 and crop_w >= 7:
        builders.extend(["binary", "chain2", "chain2"])
    else:
        return None

    for _ in range(40):
        kind = builders[int(rng.integers(0, len(builders)))]
        if kind == "mixed":
            frames = _build_mixed(crop_h, crop_w, rng)
        elif kind == "binary":
            frames = _build_binary(crop_h, crop_w, rng)
        elif kind == "chain3":
            frames = _build_chain(crop_h, crop_w, 3, rng)
        else:
            frames = _build_chain(crop_h, crop_w, 2, rng)
        if frames is not None:
            return frames
    return None


def _choose_crop_indices(num_lines: int, cell_size: int, min_dim: int, rng: np.random.Generator) -> tuple[int, int] | None:
    spacing = cell_size + 1
    valid_spans = [span for span in (2, 3) if span < num_lines and span * spacing + 1 >= min_dim]
    if not valid_spans:
        return None
    span = int(valid_spans[int(rng.integers(0, len(valid_spans)))])
    start = int(rng.integers(0, num_lines - span))
    return start, span


def _render_lattice(num_h: int, num_v: int, cell_size: int) -> tuple[np.ndarray, list[int], list[int]]:
    spacing = cell_size + 1
    rows = [cell_size + i * spacing for i in range(num_h)]
    cols = [cell_size + i * spacing for i in range(num_v)]
    h = rows[-1] + cell_size + 1
    w = cols[-1] + cell_size + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[rows, :] = GREY
    grid[:, cols] = GREY
    return grid, rows, cols


def _add_distractors(grid: np.ndarray, crop_bbox: Rect, rng: np.random.Generator) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = crop_bbox
    target = int(rng.integers(0, 3))
    placed: list[Rect] = []
    for _ in range(target):
        done = False
        for _ in range(200):
            rect = _random_rect_in_region((1, 1, h - 2, w - 2), 3, 6, 3, 6, rng)
            if rect is None:
                break
            if not (rect[2] < r0 or rect[0] > r1 or rect[3] < c0 or rect[1] > c1):
                continue
            if any(not _separated(rect, other, gap=0) for other in placed):
                continue
            _draw_frame(grid, rect)
            placed.append(rect)
            done = True
            break
        if not done:
            continue


def _expected_output_from_layout(grid: np.ndarray, crop_bbox: Rect, frames_abs: list[Rect]) -> np.ndarray:
    r0, c0, r1, c1 = crop_bbox
    crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    frames_rel = [(fr0 - r0, fc0 - c0, fr1 - r0, fc1 - c0) for fr0, fc0, fr1, fc1 in frames_abs]
    return _filled_crop(crop, frames_rel)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cell_size = int(rng.choice(np.array([3, 4], dtype=int)))
    num_h = int(rng.choice(np.array([4, 5], dtype=int)))
    num_v = int(rng.choice(np.array([4, 5], dtype=int)))
    lattice, line_rows, line_cols = _render_lattice(num_h, num_v, cell_size)

    crop_row_pick = _choose_crop_indices(num_h, cell_size, 11, rng)
    crop_col_pick = _choose_crop_indices(num_v, cell_size, 11, rng)
    if crop_row_pick is None or crop_col_pick is None:
        return None
    row_start, row_span = crop_row_pick
    col_start, col_span = crop_col_pick

    crop_bbox = (
        line_rows[row_start],
        line_cols[col_start],
        line_rows[row_start + row_span],
        line_cols[col_start + col_span],
    )
    crop_h = crop_bbox[2] - crop_bbox[0] + 1
    crop_w = crop_bbox[3] - crop_bbox[1] + 1

    frames_rel = _frame_layout(crop_h, crop_w, rng)
    if frames_rel is None:
        return None

    grid = lattice.copy()
    frames_abs = []
    for rect in frames_rel:
        abs_rect = (
            crop_bbox[0] + rect[0],
            crop_bbox[1] + rect[1],
            crop_bbox[0] + rect[2],
            crop_bbox[1] + rect[3],
        )
        _draw_frame(grid, abs_rect)
        frames_abs.append(abs_rect)

    for rr in (crop_bbox[0], crop_bbox[2]):
        for cc in (crop_bbox[1], crop_bbox[3]):
            grid[rr, cc] = BLACK

    _add_distractors(grid, crop_bbox, rng)

    expected = _expected_output_from_layout(grid, crop_bbox, frames_abs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    rel_counts = _direct_child_counts(frames_rel)
    branch_values = set(rel_counts.values())
    if 0 not in branch_values:
        return None
    if len(frames_rel) >= 3 and not ({1, 2} & branch_values):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame lattice puzzle")
