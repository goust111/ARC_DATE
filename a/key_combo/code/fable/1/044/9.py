from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    """Minimal monochromatic 4-connected component finder using (r, c) coordinates."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _frame_border_coords(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _frame_side_len(bbox: tuple[int, int, int, int]) -> int:
    r0, _c0, r1, _c1 = bbox
    return int(r1 - r0 + 1)


def _max_depth_for_bbox(bbox: tuple[int, int, int, int]) -> int:
    return (_frame_side_len(bbox) - 1) // 2


def _find_frame_in_components(components: list[dict]) -> dict:
    candidates = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w or h < 3:
            continue
        border = _frame_border_coords(comp["bbox"])
        actual = {tuple(map(int, rc)) for rc in comp["coords"]}
        if actual == border:
            candidates.append(comp)
    assert len(candidates) == 1, "expected exactly one hollow square frame"
    return candidates[0]


def _marker_side_and_depth(
    marker: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[str, int]:
    r, c = marker
    r0, c0, r1, c1 = bbox
    if r < r0 and c0 <= c <= c1:
        return "top", int(r0 - r - 1)
    if r > r1 and c0 <= c <= c1:
        return "bottom", int(r - r1 - 1)
    if c < c0 and r0 <= r <= r1:
        return "left", int(c0 - c - 1)
    if c > c1 and r0 <= r <= r1:
        return "right", int(c - c1 - 1)
    raise AssertionError("marker must lie in one of the four cardinal side bands")


def _segment_cells(
    bbox: tuple[int, int, int, int], side: str, depth: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    top = r0 + depth
    left = c0 + depth
    bottom = r1 - depth
    right = c1 - depth
    assert top <= bottom and left <= right, "selected depth must correspond to an inner layer"
    if side == "top":
        return [(top, c) for c in range(left, right + 1)]
    if side == "bottom":
        return [(bottom, c) for c in range(left, right + 1)]
    if side == "left":
        return [(r, left) for r in range(top, bottom + 1)]
    assert side == "right"
    return [(r, right) for r in range(top, bottom + 1)]


def _extract_scene(grid: np.ndarray) -> tuple[dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG if np.count_nonzero(g == BG) else int(np.bincount(g.ravel()).argmax())
    components = _find_components(g, bg)
    frame = _find_frame_in_components(components)
    max_depth = _max_depth_for_bbox(frame["bbox"])
    markers: list[dict] = []
    seen_side_depth: dict[tuple[str, int], int] = {}
    for comp in components:
        if comp is frame:
            continue
        assert int(comp["area"]) == 1, "every non-frame foreground component must be a singleton marker"
        r, c = map(int, comp["coords"][0])
        side, depth = _marker_side_and_depth((r, c), frame["bbox"])
        assert 1 <= depth <= max_depth, "marker gap must fold to an existing inner layer"
        key = (side, depth)
        if key in seen_side_depth:
            assert seen_side_depth[key] == int(comp["color"]), "same side/depth markers must agree in color"
        else:
            seen_side_depth[key] = int(comp["color"])
        markers.append({"r": r, "c": c, "side": side, "depth": depth, "color": int(comp["color"])})
    assert markers, "expected at least one external marker"
    return frame, markers


def _paint_segments(
    out: np.ndarray, bbox: tuple[int, int, int, int], markers: list[dict]
) -> np.ndarray:
    painted = out.copy()
    for marker in markers:
        for r, c in _segment_cells(bbox, marker["side"], marker["depth"]):
            current = int(painted[r, c])
            color = int(marker["color"])
            assert current in (BG, color), "different marker colors cannot claim the same segment cell"
            painted[r, c] = color
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each outside marker's gap into the frame and paint only the matching inner side segment.

    Input structure:
    - Black background, one one-cell-thick gray hollow square frame, and several colored singleton markers.
    - Every marker sits strictly outside the frame in one of the four cardinal side bands, separated from the wall by one or more black cells.

    Rule:
    - Partition the frame interior into concentric depth layers: cells adjacent to the frame wall are layer 1, then depth increases inward.
    - For a marker with gap d from the nearest frame wall, select layer d.
    - Use the marker's side relative to the frame to choose the top, bottom, left, or right segment of that layer, and recolor that whole segment to the marker color.
    - Preserve the original frame and all outside markers; leave every unselected interior cell black.

    Key insight:
    - The empty air gap outside the frame is not a translation distance; it is folded inward as a depth index, while the marker's orientation independently chooses which edge of that depth layer is painted.
    """
    g = np.asarray(grid, dtype=int)
    frame, markers = _extract_scene(g)
    return _paint_segments(g.copy(), frame["bbox"], markers)


def _blank_canvas(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in _frame_border_coords(bbox):
        grid[r, c] = FRAME_COLOR
    return grid


def _choose_assignments(side_len: int) -> list[dict]:
    max_depth = (side_len - 1) // 2
    assert max_depth >= 1
    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    color_i = 0
    assignments: list[dict] = []
    used_pairs: set[tuple[str, int]] = set()
    depth_to_color: dict[int, int] = {}

    cluster_depths = [d for d in range(1, max_depth + 1) if side_len - 2 * d >= 2]
    if cluster_depths and random.random() < 0.7:
        depth = random.choice(cluster_depths)
        side_count = random.randint(2, 3)
        sides = random.sample(SIDES, side_count)
        color = colors[color_i]
        color_i += 1
        depth_to_color[depth] = color
        for side in sides:
            assignments.append({"side": side, "depth": depth, "color": color})
            used_pairs.add((side, depth))

    target_total = random.randint(max(2, len(assignments)), 4)
    available = [(side, depth) for side in SIDES for depth in range(1, max_depth + 1) if (side, depth) not in used_pairs]
    random.shuffle(available)
    while len(assignments) < target_total and available:
        side, depth = available.pop()
        if depth in depth_to_color:
            color = depth_to_color[depth]
        else:
            color = colors[color_i]
            color_i += 1
            depth_to_color[depth] = color
        assignments.append({"side": side, "depth": depth, "color": color})
        used_pairs.add((side, depth))

    side_kinds = {item["side"] for item in assignments}
    if len(side_kinds) == 1:
        for side, depth in available:
            if side != next(iter(side_kinds)):
                if depth in depth_to_color:
                    color = depth_to_color[depth]
                else:
                    color = colors[color_i]
                    color_i += 1
                    depth_to_color[depth] = color
                assignments.append({"side": side, "depth": depth, "color": color})
                break

    return assignments


def _marker_candidates(
    bbox: tuple[int, int, int, int], side: str, depth: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        row = r0 - depth - 1
        return [(row, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        row = r1 + depth + 1
        return [(row, c) for c in range(c0, c1 + 1)]
    if side == "left":
        col = c0 - depth - 1
        return [(r, col) for r in range(r0, r1 + 1)]
    assert side == "right"
    col = c1 + depth + 1
    return [(r, col) for r in range(r0, r1 + 1)]


def _pick_marker_positions(
    bbox: tuple[int, int, int, int], assignments: list[dict]
) -> list[dict] | None:
    chosen: list[dict] = []
    used_positions: list[tuple[int, int]] = []
    order = assignments[:]
    random.shuffle(order)
    for item in order:
        candidates = _marker_candidates(bbox, item["side"], item["depth"])
        random.shuffle(candidates)
        pick = None
        for r, c in candidates:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in used_positions):
                pick = (r, c)
                break
        if pick is None:
            return None
        used_positions.append(pick)
        chosen.append({**item, "r": pick[0], "c": pick[1]})
    return chosen


def _build_expected(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], placed_markers: list[dict]
) -> np.ndarray:
    grid = _blank_canvas(shape, bbox)
    for marker in placed_markers:
        grid[marker["r"], marker["c"]] = int(marker["color"])
    return _paint_segments(grid, bbox, placed_markers)


def _signature(markers: list[dict]) -> list[tuple[str, int, int]]:
    return sorted((str(m["side"]), int(m["depth"]), int(m["color"])) for m in markers)


def _gen_attempt() -> np.ndarray | None:
    side_len = random.randint(5, 13)
    max_depth = (side_len - 1) // 2
    margin_lo = max_depth + 1
    margin_hi = max_depth + 2
    top = random.randint(margin_lo, margin_hi)
    bottom = random.randint(margin_lo, margin_hi)
    left = random.randint(margin_lo, margin_hi)
    right = random.randint(margin_lo, margin_hi)
    shape = (side_len + top + bottom, side_len + left + right)
    assert max(shape) <= 30
    bbox = (top, left, top + side_len - 1, left + side_len - 1)

    assignments = _choose_assignments(side_len)
    placed_markers = _pick_marker_positions(bbox, assignments)
    if placed_markers is None:
        return None

    inp = _blank_canvas(shape, bbox)
    for marker in placed_markers:
        inp[marker["r"], marker["c"]] = int(marker["color"])

    expected = _build_expected(shape, bbox, placed_markers)
    parsed_frame, parsed_markers = _extract_scene(inp)
    if parsed_frame["bbox"] != bbox:
        return None
    if _signature(parsed_markers) != _signature(placed_markers):
        return None
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid outer-gap inner-layer frame puzzle")
