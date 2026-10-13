from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PIVOT = 5
FOREGROUND_COLORS = (1, 2, 3, 4)
CORNER_LABELS = ("TL", "TR", "BL", "BR")
GRID_SIZE_MIN = 10
GRID_SIZE_MAX = 16
MAX_GENERATION_ATTEMPTS = 800
PIVOT_COVER_MODES = ("require", "forbid", "either")
PIVOT_COVER_WEIGHTS = (0.4, 0.35, 0.25)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _corners(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = shape
    return {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }


def _nearest_corner(r: int, c: int, shape: tuple[int, int]) -> tuple[str, tuple[int, int], int] | None:
    corner_map = _corners(shape)
    distances = {
        label: abs(r - corner_r) + abs(c - corner_c)
        for label, (corner_r, corner_c) in corner_map.items()
    }
    best = min(distances.values())
    winners = [label for label, distance in distances.items() if distance == best]
    if len(winners) != 1:
        return None
    label = winners[0]
    return label, corner_map[label], best


def _reflect(r: int, c: int, pr: int, pc: int) -> tuple[int, int]:
    return 2 * pr - r, 2 * pc - c


def _rect_bounds(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = a
    r1, c1 = b
    return min(r0, r1), min(c0, c1), max(r0, r1), max(c0, c1)


def _rect_contains(rect: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _render_from_items(
    shape: tuple[int, int],
    pivot: tuple[int, int],
    items: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    pr, pc = pivot
    for item in sorted(
        items,
        key=lambda entry: (
            -int(entry["distance"]),
            int(entry["target"][0]),
            int(entry["target"][1]),
            int(entry["color"]),
        ),
    ):
        r0, c0, r1, c1 = item["rect"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(item["color"])
    out[pr, pc] = PIVOT
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored singleton through the grey pivot, then fill the rectangle from the reflected point to its nearest corner.

    Input structure:
    - A black background contains exactly one grey pivot pixel.
    - Two to four non-black, non-grey foreground cells are isolated single pixels with distinct colors.
    - Every foreground cell reflects through the pivot to another in-bounds location.

    Rule:
    - For each colored foreground pixel at `(r, c)`, compute its 180-degree reflection through the pivot `(pr, pc)` as `(2*pr-r, 2*pc-c)`.
    - At that reflected location, find the unique nearest grid corner by Manhattan distance.
    - Fill the inclusive axis-aligned rectangle between the reflected location and that nearest corner with the pixel's color.
    - Draw larger reflected-to-corner rectangles first and smaller ones later; the grey pivot remains grey even if a rectangle passes through it.

    Key insight:
    - The half-turn is only an intermediate step: it changes which corner is nearest, and that new corner determines the final filled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    shape = g.shape

    pivots = np.argwhere(g == PIVOT)
    assert len(pivots) == 1, "expected exactly one grey pivot"
    pr, pc = map(int, pivots[0])

    items: list[dict[str, object]] = []
    for r_raw, c_raw in np.argwhere((g != BACKGROUND) & (g != PIVOT)):
        r = int(r_raw)
        c = int(c_raw)
        rr, cc = _reflect(r, c, pr, pc)
        assert _in_bounds(shape, rr, cc), "reflected point must stay in bounds"
        nearest = _nearest_corner(rr, cc, shape)
        assert nearest is not None, "reflected point must have a unique nearest corner"
        _label, corner, distance = nearest
        items.append(
            {
                "target": (rr, cc),
                "rect": _rect_bounds((rr, cc), corner),
                "distance": distance,
                "color": int(g[r, c]),
            }
        )

    assert len(items) >= 1, "expected at least one colored singleton"
    return _render_from_items(shape, (pr, pc), items)


def _pivot_candidates(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for pr in range(1, h - 1):
        for pc in range(1, w - 1):
            nearest = _nearest_corner(pr, pc, shape)
            if nearest is None:
                continue
            distances = [
                abs(pr - corner_r) + abs(pc - corner_c)
                for corner_r, corner_c in _corners(shape).values()
            ]
            if len(set(distances)) != 4:
                continue
            candidates.append((pr, pc))
    return candidates


def _target_candidates(
    shape: tuple[int, int],
    pivot: tuple[int, int],
    label: str,
) -> list[dict[str, object]]:
    h, w = shape
    pr, pc = pivot
    candidates: list[dict[str, object]] = []
    for rr in range(h):
        for cc in range(w):
            if (rr, cc) == pivot:
                continue
            nearest = _nearest_corner(rr, cc, shape)
            if nearest is None or nearest[0] != label:
                continue
            _winner, corner, distance = nearest
            if distance < 2:
                continue
            r, c = _reflect(rr, cc, pr, pc)
            if not _in_bounds(shape, r, c):
                continue
            if (r, c) == pivot or _chebyshev((r, c), pivot) <= 1:
                continue
            rect = _rect_bounds((rr, cc), corner)
            candidates.append(
                {
                    "origin": (r, c),
                    "target": (rr, cc),
                    "rect": rect,
                    "distance": distance,
                    "covers_pivot": _rect_contains(rect, pivot),
                }
            )
    random.shuffle(candidates)
    return candidates


def _pick_corner_set(object_count: int, pivot_corner: str, pivot_cover_mode: str) -> tuple[str, ...]:
    if pivot_cover_mode == "require":
        remaining = [label for label in CORNER_LABELS if label != pivot_corner]
        return tuple([pivot_corner, *random.sample(remaining, object_count - 1)])
    return tuple(random.sample(CORNER_LABELS, object_count))


def _search_selection(
    order: list[str],
    candidates_by_corner: dict[str, list[dict[str, object]]],
    index: int,
    chosen: list[dict[str, object]],
    used_origins: set[tuple[int, int]],
    used_targets: set[tuple[int, int]],
    used_distances: set[int],
) -> list[dict[str, object]] | None:
    if index == len(order):
        return [dict(item) for item in chosen]

    label = order[index]
    for candidate in candidates_by_corner[label]:
        origin = candidate["origin"]
        target = candidate["target"]
        distance = int(candidate["distance"])
        rect = candidate["rect"]
        if origin in used_origins or target in used_targets or distance in used_distances:
            continue
        if any(_chebyshev(origin, other["origin"]) <= 1 for other in chosen):
            continue
        if any(_rects_overlap(rect, other["rect"]) for other in chosen):
            continue

        chosen.append(candidate)
        used_origins.add(origin)
        used_targets.add(target)
        used_distances.add(distance)
        result = _search_selection(
            order,
            candidates_by_corner,
            index + 1,
            chosen,
            used_origins,
            used_targets,
            used_distances,
        )
        if result is not None:
            return result
        chosen.pop()
        used_origins.remove(origin)
        used_targets.remove(target)
        used_distances.remove(distance)
    return None


def _expected_from_selection(
    shape: tuple[int, int],
    pivot: tuple[int, int],
    selection: list[dict[str, object]],
) -> np.ndarray:
    items = [
        {
            "target": candidate["target"],
            "rect": candidate["rect"],
            "distance": candidate["distance"],
            "color": candidate["color"],
        }
        for candidate in selection
    ]
    return _render_from_items(shape, pivot, items)


def _attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX),
        random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX),
    )
    pivot_options = _pivot_candidates(shape)
    if not pivot_options:
        return None

    pivot = random.choice(pivot_options)
    pivot_info = _nearest_corner(pivot[0], pivot[1], shape)
    assert pivot_info is not None
    pivot_corner = pivot_info[0]
    object_count = random.choice((2, 3, 4))
    pivot_cover_mode = random.choices(PIVOT_COVER_MODES, weights=PIVOT_COVER_WEIGHTS, k=1)[0]
    chosen_corners = _pick_corner_set(object_count, pivot_corner, pivot_cover_mode)

    candidates_by_corner = {label: _target_candidates(shape, pivot, label) for label in chosen_corners}
    if any(not options for options in candidates_by_corner.values()):
        return None
    if pivot_cover_mode == "require":
        candidates_by_corner[pivot_corner] = [
            candidate for candidate in candidates_by_corner[pivot_corner] if bool(candidate["covers_pivot"])
        ]
        if not candidates_by_corner[pivot_corner]:
            return None
    elif pivot_cover_mode == "forbid":
        for label in chosen_corners:
            candidates_by_corner[label] = [
                candidate for candidate in candidates_by_corner[label] if not bool(candidate["covers_pivot"])
            ]
            if not candidates_by_corner[label]:
                return None

    order = sorted(chosen_corners, key=lambda label: len(candidates_by_corner[label]))
    selection = _search_selection(order, candidates_by_corner, 0, [], set(), set(), set())
    if selection is None:
        return None

    cover_count = sum(bool(candidate["covers_pivot"]) for candidate in selection)
    if pivot_cover_mode == "require" and cover_count == 0:
        return None
    if pivot_cover_mode == "forbid" and cover_count != 0:
        return None
    if max(int(candidate["distance"]) for candidate in selection) < 4:
        return None

    colors = random.sample(FOREGROUND_COLORS, len(selection))
    for candidate, color in zip(selection, colors):
        candidate["color"] = color

    grid = np.full(shape, BACKGROUND, dtype=int)
    pr, pc = pivot
    grid[pr, pc] = PIVOT
    for candidate in selection:
        r, c = candidate["origin"]
        grid[r, c] = int(candidate["color"])

    expected = _expected_from_selection(shape, pivot, selection)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.count_nonzero((grid != BACKGROUND) & (grid != PIVOT)) != len(selection):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
