from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
REGION_COLORS = (BLUE, RED, GREEN, YELLOW)

RECT_DIMS = tuple(
    (h, w)
    for h in range(2, 6)
    for w in range(2, 6)
    if 6 <= h * w <= 15
)

GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_distance_to_point(bbox: tuple[int, int, int, int], point: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    pr, pc = point
    dr = 0 if r0 <= pr <= r1 else min(abs(pr - r0), abs(pr - r1))
    dc = 0 if c0 <= pc <= c1 else min(abs(pc - c0), abs(pc - c1))
    return int(dr + dc)


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _edge_adjacency_to_mask(
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
) -> int:
    r0, c0, r1, c1 = bbox
    total = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
                    total += 1
    return int(total)


def _trace_spiral(grid: np.ndarray, marker: tuple[int, int]) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r, c = marker
    mask = np.zeros((h, w), dtype=bool)
    coords: list[tuple[int, int]] = []
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    step_len = 1
    dir_idx = 0
    empty_dirs = 0
    skipped_oob = 0
    skipped_blocked = 0
    max_step_len = 3 * (h + w) + 10

    while empty_dirs < 4 and step_len <= max_step_len:
        for _ in range(2):
            dr, dc = dirs[dir_idx % 4]
            drew_in_direction = False
            for _ in range(step_len):
                nr = r + dr
                nc = c + dc
                if _in_bounds(g.shape, nr, nc):
                    if g[nr, nc] == BLACK and not mask[nr, nc]:
                        r, c = nr, nc
                        if not mask[r, c]:
                            mask[r, c] = True
                            coords.append((r, c))
                        drew_in_direction = True
                    else:
                        skipped_blocked += 1
                else:
                    skipped_oob += 1
            empty_dirs = 0 if drew_in_direction else empty_dirs + 1
            dir_idx += 1
            if empty_dirs >= 4:
                break
        step_len += 1

    assert empty_dirs >= 4, "spiral should terminate after four empty directions"
    return {
        "mask": mask,
        "coords": coords,
        "skipped_oob": int(skipped_oob),
        "skipped_blocked": int(skipped_blocked),
    }


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.shape == (1, 2), "expected exactly one grey marker pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, GREY)]
    assert 3 <= len(colors) <= 4, "expected 3-4 colored rectangles"
    assert set(colors) <= set(REGION_COLORS), "unexpected rectangle color"

    rectangles = []
    for color in sorted(colors):
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "region must be a solid rectangle"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be monochrome"
        rectangles.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "area": int(coords.shape[0]),
            }
        )
    return rectangles


def _vote_counts(
    grid: np.ndarray,
    spiral_mask: np.ndarray,
    region_colors: list[int],
) -> dict[int, int]:
    counts = {color: 0 for color in region_colors}
    for r, c in np.argwhere(spiral_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if _in_bounds(grid.shape, nr, nc):
                color = int(grid[nr, nc])
                if color in counts:
                    counts[color] += 1
    return counts


def _winning_color(counts: dict[int, int]) -> int:
    winners = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    assert winners and winners[0][1] > 0, "spiral must touch at least one region"
    if len(winners) >= 2:
        assert winners[0][1] != winners[1][1], "adjacency vote must have a unique winner"
    return int(winners[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the spiral traced from the grey marker with the rectangle color that touches it most.

    Input structure:
    - The background is black.
    - There are 3 or 4 solid colored rectangles using distinct colors from blue, red, green, and yellow.
    - A single grey marker pixel sits on black background between regions.

    Rule:
    - Start from the grey marker and follow a square spiral with directions up, right, down, left and step lengths 1, 1, 2, 2, 3, 3, ...
    - Advance the spiral logically even when a step goes out of bounds or lands on a non-black cell, but only draw spiral pixels on in-bounds black cells.
    - Stop once four consecutive directions add no drawable spiral pixel.
    - Count, for each rectangle color, how many 4-neighbor side contacts exist between spiral pixels and that color's rectangle.
    - Recolor every drawn spiral pixel with the uniquely highest-contact rectangle color, while keeping the rectangles and the original grey marker unchanged.

    Key insight:
    - The grey pixel determines a geometric spiral probe, and the probe's side-contact counts against the surrounding rectangles determine the probe's final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    marker = _find_marker(g)
    rectangles = _extract_rectangles(g)
    spiral = _trace_spiral(g, marker)
    counts = _vote_counts(g, spiral["mask"], [int(rect["color"]) for rect in rectangles])
    winner = _winning_color(counts)

    out = g.copy()
    out[np.asarray(spiral["mask"], dtype=bool)] = winner
    out[marker] = GREY
    return out


def _empty_spiral(shape: tuple[int, int], marker: tuple[int, int]) -> dict[str, object]:
    base = np.zeros(shape, dtype=int)
    base[marker] = GREY
    return _trace_spiral(base, marker)


def _enumerate_candidates(
    shape: tuple[int, int],
    marker: tuple[int, int],
    empty_mask: np.ndarray,
) -> list[dict[str, object]]:
    h, w = shape
    mr, mc = marker
    candidates: list[dict[str, object]] = []
    for rect_h, rect_w in RECT_DIMS:
        for r0 in range(h - rect_h + 1):
            for c0 in range(w - rect_w + 1):
                r1 = r0 + rect_h - 1
                c1 = c0 + rect_w - 1
                bbox = (r0, c0, r1, c1)
                if r0 <= mr <= r1 and c0 <= mc <= c1:
                    continue
                marker_dist = _bbox_distance_to_point(bbox, marker)
                if marker_dist < 2:
                    continue
                overlap = int(empty_mask[r0 : r1 + 1, c0 : c1 + 1].sum())
                adjacency = _edge_adjacency_to_mask(bbox, empty_mask)
                if adjacency < 2 or adjacency > 7:
                    continue
                if overlap > 2:
                    continue
                candidates.append(
                    {
                        "bbox": bbox,
                        "h": rect_h,
                        "w": rect_w,
                        "area": rect_h * rect_w,
                        "empty_adj": adjacency,
                        "overlap": overlap,
                        "marker_dist": marker_dist,
                    }
                )
    random.shuffle(candidates)
    return candidates


def _marker_between_pair(
    marker: tuple[int, int],
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> bool:
    mr, mc = marker
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b

    horizontal = (
        ((ac1 < mc < bc0) or (bc1 < mc < ac0))
        and max(ar0, br0) - 1 <= mr <= min(ar1, br1) + 1
    )
    vertical = (
        ((ar1 < mr < br0) or (br1 < mr < ar0))
        and max(ac0, bc0) - 1 <= mc <= min(ac1, bc1) + 1
    )
    return bool(horizontal or vertical)


def _compatible_with_all(
    candidate: dict[str, object],
    chosen: list[dict[str, object]],
) -> bool:
    bbox = candidate["bbox"]
    return all(_bboxes_separated(bbox, other["bbox"]) for other in chosen)


def _choose_rectangles(
    candidates: list[dict[str, object]],
    marker: tuple[int, int],
    count: int,
    want_blocker: bool,
) -> list[dict[str, object]] | None:
    open_near = [
        cand
        for cand in candidates
        if cand["overlap"] == 0 and 2 <= cand["marker_dist"] <= 6
    ]
    if len(open_near) < 2:
        return None

    pair_indices = list(range(len(open_near)))
    random.shuffle(pair_indices)
    pair_pool: list[tuple[dict[str, object], dict[str, object]]] = []
    for i, j in combinations(pair_indices, 2):
        first = open_near[i]
        second = open_near[j]
        if first["empty_adj"] == second["empty_adj"]:
            continue
        if not _bboxes_separated(first["bbox"], second["bbox"]):
            continue
        if not _marker_between_pair(marker, first["bbox"], second["bbox"]):
            continue
        pair_pool.append((first, second))
    random.shuffle(pair_pool)

    blocker_pool = [cand for cand in candidates if cand["overlap"] > 0]
    open_pool = [cand for cand in candidates if cand["overlap"] == 0]

    for first, second in pair_pool:
        chosen = [first, second]
        used_counts = {int(first["empty_adj"]), int(second["empty_adj"])}
        used_blocker = False

        if want_blocker:
            blocker_choices = blocker_pool[:]
            random.shuffle(blocker_choices)
            blocker = None
            for cand in blocker_choices:
                if int(cand["empty_adj"]) in used_counts:
                    continue
                if not _compatible_with_all(cand, chosen):
                    continue
                blocker = cand
                break
            if blocker is None:
                continue
            chosen.append(blocker)
            used_counts.add(int(blocker["empty_adj"]))
            used_blocker = True

        if len(chosen) > count:
            continue

        extra_choices = open_pool[:]
        random.shuffle(extra_choices)
        for cand in extra_choices:
            if len(chosen) == count:
                break
            if cand in chosen:
                continue
            if int(cand["empty_adj"]) in used_counts:
                continue
            if not _compatible_with_all(cand, chosen):
                continue
            chosen.append(cand)
            used_counts.add(int(cand["empty_adj"]))

        if len(chosen) == count and (not want_blocker or used_blocker):
            return chosen
    return None


def _build_grid(
    shape: tuple[int, int],
    marker: tuple[int, int],
    chosen: list[dict[str, object]],
) -> tuple[np.ndarray, dict[int, int]]:
    grid = np.zeros(shape, dtype=int)
    grid[marker] = GREY
    colors = random.sample(list(REGION_COLORS), len(chosen))
    planned: dict[int, int] = {}
    for cand, color in zip(random.sample(chosen, len(chosen)), colors):
        r0, c0, r1, c1 = cand["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        planned[color] = int(cand["empty_adj"])
    return grid, planned


def _nearest_rectangle_color(
    rectangles: list[dict[str, int | tuple[int, int, int, int]]],
    marker: tuple[int, int],
) -> int | None:
    distances = [
        (int(_bbox_distance_to_point(rect["bbox"], marker)), int(rect["color"]))
        for rect in rectangles
    ]
    distances.sort()
    if len(distances) >= 2 and distances[0][0] == distances[1][0]:
        return None
    return distances[0][1]


def _largest_rectangle_color(
    rectangles: list[dict[str, int | tuple[int, int, int, int]]]
) -> int | None:
    by_area = sorted((int(rect["area"]), int(rect["color"])) for rect in rectangles)
    if len(by_area) >= 2 and by_area[-1][0] == by_area[-2][0]:
        return None
    return by_area[-1][1]


def _validate_generated(
    grid: np.ndarray,
    planned_counts: dict[int, int],
    want_blocker: bool,
) -> bool:
    marker = _find_marker(grid)
    rectangles = _extract_rectangles(grid)
    spiral = _trace_spiral(grid, marker)
    if len(spiral["coords"]) < 12:
        return False
    if int(spiral["skipped_oob"]) <= 0:
        return False
    if want_blocker != (int(spiral["skipped_blocked"]) > 0):
        return False

    counts = _vote_counts(grid, spiral["mask"], [int(rect["color"]) for rect in rectangles])
    if any(count <= 0 for count in counts.values()):
        return False
    if len(set(counts.values())) != len(counts):
        return False

    winner = _winning_color(counts)
    largest = _largest_rectangle_color(rectangles)
    nearest = _nearest_rectangle_color(rectangles, marker)
    if largest is not None and nearest is not None and winner == largest == nearest:
        return False

    if want_blocker and not any(counts[color] != planned_counts[color] for color in counts):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if tuple(np.argwhere(out == GREY)[0]) != marker:
        return False
    return True


def _sample_marker(shape: tuple[int, int], edge_mode: bool) -> tuple[int, int] | None:
    h, w = shape
    if edge_mode:
        positions = [
            (r, c)
            for r in range(2, h - 2)
            for c in range(2, w - 2)
            if min(r, c, h - 1 - r, w - 1 - c) <= 2
            and min(r, c, h - 1 - r, w - 1 - c) >= 1
        ]
    else:
        positions = [
            (r, c)
            for r in range(3, h - 3)
            for c in range(3, w - 3)
        ]
    if not positions:
        return None
    return random.choice(positions)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 15)
    w = random.randint(12, 15)
    edge_mode = random.random() < 0.45
    marker = _sample_marker((h, w), edge_mode)
    if marker is None:
        return None

    empty = _empty_spiral((h, w), marker)
    if len(empty["coords"]) < 16:
        return None

    want_blocker = True
    rect_count = random.choice([3, 4])
    candidates = _enumerate_candidates((h, w), marker, np.asarray(empty["mask"], dtype=bool))
    if len(candidates) < 20:
        return None

    chosen = _choose_rectangles(candidates, marker, rect_count, want_blocker)
    if chosen is None:
        return None

    grid, planned_counts = _build_grid((h, w), marker, chosen)
    if not _validate_generated(grid, planned_counts, want_blocker):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-vote ARC input")
