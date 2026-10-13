from __future__ import annotations

from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_pair(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _infer_scene(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = _get_bg(grid)
    counts = Counter(map(int, grid.ravel()))
    counts.pop(bg, None)
    assert counts, "expected foreground pixels"

    content_color, content_count = max(counts.items(), key=lambda kv: kv[1])
    assert int(content_count) > 2, "content color must appear more than twice"

    rectangles: list[dict[str, object]] = []
    for color, count in counts.items():
        if int(color) == int(content_color):
            continue
        assert int(count) == 2, "each marker color must appear exactly twice"
        coords = np.argwhere(grid == int(color))
        rectangles.append({"color": int(color), "bbox": _bbox_from_pair(coords)})

    assert len(rectangles) >= 2, "expected at least two marker pairs"
    return int(content_color), rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor only the content pixels selected by the symmetric difference of implicit rectangles.

    Input structure:
    - A dominant background contains one scattered content color and several marker colors.
    - Each marker color appears exactly twice as isolated singleton pixels placed on opposite corners of one axis-aligned rectangle.
    - The rectangles themselves are not drawn; only their two corner generators are visible.

    Rule:
    - Each same-color marker pair defines the inclusive solid rectangle spanning those two corners.
    - For every content pixel, count how many implicit rectangles contain it.
    - Recolor content pixels covered by exactly one rectangle to that rectangle's marker color.
    - Erase content pixels covered by zero or multiple rectangles, and erase all marker pixels.

    Key insight:
    - The visible markers are only generators of hidden regions; the actual selection is the rectangles'
      symmetric difference, applied only to the scattered content pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    content_color, rectangles = _infer_scene(g)

    out = np.full_like(g, bg)
    for r, c in np.argwhere(g == content_color):
        owners: list[int] = []
        rr = int(r)
        cc = int(c)
        for rect in rectangles:
            r0, c0, r1, c1 = rect["bbox"]
            if int(r0) <= rr <= int(r1) and int(c0) <= cc <= int(c1):
                owners.append(int(rect["color"]))
        if len(owners) == 1:
            out[rr, cc] = owners[0]

    return out


_RNG = np.random.default_rng()
_PALETTE = tuple(range(10))
_GRID_MIN = 18
_GRID_MAX = 28
_RECT_MIN = 5
_RECT_MAX = 9
_DISJOINT_GAP = 2
_BORDER = 1
_MAX_ATTEMPTS = 500


def _flip_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    flip_r: bool,
    flip_c: bool,
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if flip_r:
        r0, r1 = h - 1 - r1, h - 1 - r0
    if flip_c:
        c0, c1 = w - 1 - c1, w - 1 - c0
    return int(r0), int(c0), int(r1), int(c1)


def _marker_pair(bbox: tuple[int, int, int, int], anti_diag: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if anti_diag:
        return (int(r0), int(c1)), (int(r1), int(c0))
    return (int(r0), int(c0)), (int(r1), int(c1))


def _gather_marker_cells(rectangles: list[dict[str, object]]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for rect in rectangles:
        for cell in rect["markers"]:
            points.append((int(cell[0]), int(cell[1])))
    return points


def _markers_are_isolated(rectangles: list[dict[str, object]]) -> bool:
    points = _gather_marker_cells(rectangles)
    if len(points) != len(set(points)):
        return False
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _rects_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _build_overlap_pair(shape: tuple[int, int], colors: list[int]) -> list[dict[str, object]] | None:
    h, w = shape
    for _ in range(100):
        h_a = int(_RNG.integers(_RECT_MIN, _RECT_MAX + 1))
        w_a = int(_RNG.integers(_RECT_MIN, _RECT_MAX + 1))
        h_b = int(_RNG.integers(_RECT_MIN, _RECT_MAX + 1))
        w_b = int(_RNG.integers(_RECT_MIN, _RECT_MAX + 1))

        max_overlap_h = min(h_a, h_b) - 2
        max_overlap_w = min(w_a, w_b) - 2
        if max_overlap_h < 2 or max_overlap_w < 2:
            continue

        overlap_h = int(_RNG.integers(2, max_overlap_h + 1))
        overlap_w = int(_RNG.integers(2, max_overlap_w + 1))
        extra_top = h_a - overlap_h
        extra_left = w_b - overlap_w

        total_h = extra_top + h_b
        total_w = extra_left + w_a
        if total_h + 2 * _BORDER > h or total_w + 2 * _BORDER > w:
            continue

        r0_a = int(_RNG.integers(_BORDER, h - total_h - _BORDER + 1))
        c0_a = int(_RNG.integers(_BORDER + extra_left, w - w_a - _BORDER + 1))
        bbox_a = (r0_a, c0_a, r0_a + h_a - 1, c0_a + w_a - 1)
        bbox_b = (
            r0_a + extra_top,
            c0_a - extra_left,
            r0_a + extra_top + h_b - 1,
            c0_a - extra_left + w_b - 1,
        )

        flip_r = bool(int(_RNG.integers(0, 2)))
        flip_c = bool(int(_RNG.integers(0, 2)))
        bbox_a = _flip_bbox(bbox_a, shape, flip_r, flip_c)
        bbox_b = _flip_bbox(bbox_b, shape, flip_r, flip_c)

        rectangles = [
            {
                "color": int(colors[0]),
                "bbox": bbox_a,
                "markers": _marker_pair(bbox_a, anti_diag=bool(int(_RNG.integers(0, 2)))),
            },
            {
                "color": int(colors[1]),
                "bbox": bbox_b,
                "markers": _marker_pair(bbox_b, anti_diag=bool(int(_RNG.integers(0, 2)))),
            },
        ]
        if _markers_are_isolated(rectangles):
            return rectangles
    return None


def _add_disjoint_rectangle(
    shape: tuple[int, int],
    rectangles: list[dict[str, object]],
    color: int,
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(120):
        rect_h = int(_RNG.integers(4, 9))
        rect_w = int(_RNG.integers(4, 9))
        if rect_h + 2 * _BORDER > h or rect_w + 2 * _BORDER > w:
            continue

        r0 = int(_RNG.integers(_BORDER, h - rect_h - _BORDER + 1))
        c0 = int(_RNG.integers(_BORDER, w - rect_w - _BORDER + 1))
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if any(_rects_too_close(bbox, rect["bbox"], gap=_DISJOINT_GAP) for rect in rectangles):
            continue

        new_rect = {
            "color": int(color),
            "bbox": bbox,
            "markers": _marker_pair(bbox, anti_diag=bool(int(_RNG.integers(0, 2)))),
        }
        if _markers_are_isolated(rectangles + [new_rect]):
            return new_rect
    return None


def _coverage_map(shape: tuple[int, int], rectangles: list[dict[str, object]]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        cover[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] += 1
    return cover


def _mark_neighbors(blocked: np.ndarray, cell: tuple[int, int]) -> None:
    r, c = cell
    h, w = blocked.shape
    blocked[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _candidate_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    order = _RNG.permutation(len(coords))
    return [tuple(map(int, coords[i])) for i in order]


def _take_points(
    candidates: list[tuple[int, int]],
    blocked: np.ndarray,
    need_min: int,
    need_max: int,
) -> list[tuple[int, int]] | None:
    need_max = min(int(need_max), len(candidates))
    for want in range(need_max, int(need_min) - 1, -1):
        trial = blocked.copy()
        picked: list[tuple[int, int]] = []
        for r, c in candidates:
            if trial[r, c]:
                continue
            picked.append((int(r), int(c)))
            _mark_neighbors(trial, (int(r), int(c)))
            if len(picked) == want:
                blocked[:, :] = trial
                return picked
    return None


def _render_expected(
    shape: tuple[int, int],
    bg: int,
    rectangles: list[dict[str, object]],
    content_points: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for r, c in content_points:
        owners: list[int] = []
        for rect in rectangles:
            r0, c0, r1, c1 = rect["bbox"]
            if int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1):
                owners.append(int(rect["color"]))
        if len(owners) == 1:
            out[int(r), int(c)] = owners[0]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    w = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))

    bg, content_color, color_a, color_b, color_c = _RNG.choice(_PALETTE, size=5, replace=False).tolist()
    rectangles = _build_overlap_pair((h, w), [color_a, color_b])
    if rectangles is None:
        return None

    if bool(int(_RNG.integers(0, 2))):
        extra = _add_disjoint_rectangle((h, w), rectangles, color_c)
        if extra is not None:
            rectangles.append(extra)

    cover = _coverage_map((h, w), rectangles)
    if int(cover.max()) != 2:
        return None

    marker_mask = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    for cell in _gather_marker_cells(rectangles):
        marker_mask[cell] = True
        _mark_neighbors(blocked, cell)

    content_points: list[tuple[int, int]] = []

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        rect_mask = np.zeros((h, w), dtype=bool)
        rect_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
        candidates = _candidate_coords((cover == 1) & rect_mask & ~marker_mask)
        picked = _take_points(candidates, blocked, 1, 3)
        if picked is None:
            return None
        content_points.extend(picked)

    overlap_candidates = _candidate_coords((cover == 2) & ~marker_mask)
    overlap_points = _take_points(overlap_candidates, blocked, 1, 3)
    if overlap_points is None:
        return None
    content_points.extend(overlap_points)

    outside_candidates = _candidate_coords((cover == 0) & ~marker_mask)
    outside_points = _take_points(outside_candidates, blocked, 1, 4)
    if outside_points is None:
        return None
    content_points.extend(outside_points)

    if len(content_points) <= 2:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    for rect in rectangles:
        for r, c in rect["markers"]:
            grid[int(r), int(c)] = int(rect["color"])
    for r, c in content_points:
        grid[int(r), int(c)] = int(content_color)

    expected = _render_expected((h, w), int(bg), rectangles, content_points)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
