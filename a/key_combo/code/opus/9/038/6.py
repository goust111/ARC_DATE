from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ANCHOR_COLORS = (4, 5, 6, 7, 8, 9)

RING_PATTERN = np.array(
    [
        [GREEN, GREEN, GREEN],
        [GREEN, RED, GREEN],
        [GREEN, GREEN, GREEN],
    ],
    dtype=int,
)

GENERATION_RETRIES = 500


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _marker_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if np.array_equal(grid[r - 1 : r + 2, c - 1 : c + 2], RING_PATTERN):
                centers.append((r, c))
    return centers


def _guide_rows(grid: np.ndarray) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        vals = set(map(int, grid[r]))
        if RED in vals and vals <= {RED, GREEN}:
            rows.append(int(r))
    return rows


def _anchor_blocks(grid: np.ndarray) -> list[dict]:
    anchors = []
    for color in sorted(set(map(int, np.unique(grid))) - {BLACK, RED, GREEN}):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0 or int(coords.shape[0]) != 9:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)):
            continue
        anchors.append(
            {
                "color": int(color),
                "center": (int(r0 + 1), int(c0 + 1)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return anchors


def _crossings(guide_rows: list[int], marker_centers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cols = [int(c) for _r, c in marker_centers]
    sources = {(int(r), int(c)) for r, c in marker_centers}
    out = []
    for r in guide_rows:
        for c in cols:
            point = (int(r), int(c))
            if point not in sources:
                out.append(point)
    return sorted(out)


def _nearest_anchor(point: tuple[int, int], anchors: list[dict]) -> dict:
    scored = [(_manhattan(point, anchor["center"]), idx, anchor) for idx, anchor in enumerate(anchors)]
    scored.sort(key=lambda item: (int(item[0]), int(item[1])))
    assert len(scored) >= 2
    assert int(scored[0][0]) != int(scored[1][0]), "nearest anchor must be unique"
    return scored[0][2]


def _parse_scene(grid: np.ndarray) -> dict:
    markers = sorted(_marker_centers(grid))
    guides = sorted(_guide_rows(grid))
    anchors = sorted(_anchor_blocks(grid), key=lambda item: item["color"])

    assert len(markers) == 2, "expected exactly two green marker rings"
    assert len(guides) == 2, "expected exactly two horizontal red guide rows"
    assert len(anchors) == 2, "expected exactly two solid 3x3 anchor blocks"
    assert len({r for r, _c in markers}) == 2, "markers must sit on different guide rows"
    assert len({c for _r, c in markers}) == 2, "markers must use different projection columns"
    assert sorted(r for r, _c in markers) == guides, "marker centers must lie on the red guide rows"

    crossings = _crossings(guides, markers)
    assert len(crossings) == 2
    return {
        "markers": markers,
        "guides": guides,
        "anchors": anchors,
        "crossings": crossings,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project blue guide columns from green rings, then stamp each crossing with the nearest anchor block pattern.

    Input structure:
    - Black background with exactly two full-width horizontal red guide lines.
    - Two green 3x3 hollow rings with red centers sit on different guide lines and different columns.
    - Two solid 3x3 anchor blocks of distinct colors appear away from the red guide lines.

    Rule:
    - Each green ring emits a vertical blue guide line through its center column, painting only original black cells.
    - The non-source intersections of those blue columns with the red guide rows are the stamp centers.
    - For each stamp center, choose the anchor block whose center has the smaller Manhattan distance, then stamp that anchor's 3x3 color pattern there while preserving red guide-line pixels and leaving the green rings and anchor blocks unchanged.

    Key insight:
    - The green rings determine where stamps happen, but the nearby anchor blocks determine which colored 3x3 pattern gets stamped at each crossing.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    out = g.copy()

    for _r, c in scene["markers"]:
        for rr in range(out.shape[0]):
            if int(out[rr, c]) == BLACK:
                out[rr, c] = BLUE

    for center_r, center_c in scene["crossings"]:
        anchor = _nearest_anchor((center_r, center_c), scene["anchors"])
        color = int(anchor["color"])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(center_r + dr)
                cc = int(center_c + dc)
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    if int(out[rr, cc]) not in {RED, GREEN}:
                        out[rr, cc] = color

    return out


def _spaced_pair(low: int, high: int, min_gap: int) -> tuple[int, int]:
    candidates = [
        (a, b)
        for a in range(low, high + 1)
        for b in range(a + min_gap, high + 1)
    ]
    return random.choice(candidates)


def _valid_anchor_center(
    center: tuple[int, int],
    h: int,
    w: int,
    guide_rows: list[int],
    protected_centers: list[tuple[int, int]],
) -> bool:
    r, c = center
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if any(abs(int(r) - int(rr)) <= 1 for rr in guide_rows):
        return False
    if any(_chebyshev(center, other) < 3 for other in protected_centers):
        return False
    return True


def _candidate_centers(
    point: tuple[int, int],
    h: int,
    w: int,
    guide_rows: list[int],
    protected_centers: list[tuple[int, int]],
    preferred_cols: set[int],
) -> list[tuple[int, int]]:
    preferred = []
    fallback = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = (int(r), int(c))
            if not _valid_anchor_center(center, h, w, guide_rows, protected_centers):
                continue
            dist = _manhattan(center, point)
            if not (4 <= dist <= 12):
                continue
            if c in preferred_cols:
                preferred.append(center)
            else:
                fallback.append(center)
    random.shuffle(preferred)
    random.shuffle(fallback)
    return preferred + fallback


def _build_input() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    row0, row1 = _spaced_pair(2, h - 3, 4)
    col0, col1 = _spaced_pair(2, w - 3, 4)
    guides = [int(row0), int(row1)]
    diag_flip = random.choice([False, True])
    if diag_flip:
        markers = [(int(row0), int(col1)), (int(row1), int(col0))]
    else:
        markers = [(int(row0), int(col0)), (int(row1), int(col1))]
    crossings = _crossings(guides, markers)

    colors = random.sample(list(ANCHOR_COLORS), 2)
    swap_mapping = random.choice([False, True])
    assigned_points = [crossings[1], crossings[0]] if swap_mapping else [crossings[0], crossings[1]]

    preferred_col_anchor = random.choice([None, markers[0][1], markers[1][1]])
    protected = list(markers) + list(crossings)

    candidate_lists = []
    for idx, point in enumerate(assigned_points):
        preferred_cols = {preferred_col_anchor} if idx == 0 and preferred_col_anchor is not None else set()
        candidate_lists.append(_candidate_centers(point, h, w, guides, protected, preferred_cols))

    if not candidate_lists[0] or not candidate_lists[1]:
        return None

    anchor_centers: list[tuple[int, int]] | None = None
    for center_a in candidate_lists[0][:80]:
        for center_b in candidate_lists[1][:80]:
            if _chebyshev(center_a, center_b) < 3:
                continue
            anchors = [
                {"color": int(colors[0]), "center": center_a},
                {"color": int(colors[1]), "center": center_b},
            ]
            try:
                chosen = [_nearest_anchor(point, anchors)["color"] for point in crossings]
            except AssertionError:
                continue
            if chosen != colors:
                continue
            margins = []
            for point in crossings:
                d0 = _manhattan(point, anchors[0]["center"])
                d1 = _manhattan(point, anchors[1]["center"])
                margins.append(abs(d0 - d1))
            if min(margins) < 2:
                continue
            anchor_centers = [center_a, center_b]
            break
        if anchor_centers is not None:
            break

    if anchor_centers is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r in guides:
        grid[r, :] = RED

    for r, c in markers:
        grid[r - 1 : r + 2, c - 1 : c + 2] = RING_PATTERN

    for color, (r, c) in zip(colors, anchor_centers):
        grid[r - 1 : r + 2, c - 1 : c + 2] = int(color)

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    scene = _parse_scene(grid)
    assert sorted(anchor["color"] for anchor in scene["anchors"]) == sorted(set(anchor["color"] for anchor in scene["anchors"]))

    chosen_colors = [_nearest_anchor(point, scene["anchors"])["color"] for point in scene["crossings"]]
    assert len(set(chosen_colors)) == 2, "both anchors must affect the output"

    ring_and_stamp_centers = scene["markers"] + scene["crossings"]
    for i, a in enumerate(ring_and_stamp_centers):
        for b in ring_and_stamp_centers[i + 1 :]:
            assert _chebyshev(a, b) >= 3, "rings and stamp regions must not overlap"
    for anchor in scene["anchors"]:
        for center in ring_and_stamp_centers:
            assert _chebyshev(anchor["center"], center) >= 3, "anchors must stay clear of rings and stamps"
    assert _chebyshev(scene["anchors"][0]["center"], scene["anchors"][1]["center"]) >= 3, "anchor blocks must not overlap"

    out = main(grid)
    assert out.shape == grid.shape

    added_stamp_colors = set()
    for point in scene["crossings"]:
        anchor = _nearest_anchor(point, scene["anchors"])
        color = int(anchor["color"])
        r, c = point
        patch = out[r - 1 : r + 2, c - 1 : c + 2]
        assert patch.shape == (3, 3)
        assert np.all(patch[1, :] == RED), "red guide row must survive the stamp"
        assert np.all(patch[[0, 2], :] == color), "solid anchor rows must stamp above and below the red line"
        added_stamp_colors.add(color)

    assert len(added_stamp_colors) == 2
    for _r, c in scene["markers"]:
        black_before = int(np.count_nonzero(grid[:, c] == BLACK))
        blue_after = int(np.count_nonzero(out[:, c] == BLUE))
        assert blue_after > 0 and blue_after <= black_before, "blue projection must come from original black cells"


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_input()
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
