from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
YELLOW = 4
MAROON = 9

ARM_OFFSETS = (
    (0, -3),
    (0, -2),
    (0, 2),
    (0, 3),
    (-3, 0),
    (-2, 0),
    (2, 0),
    (3, 0),
)
GAP_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_TO_BAR = {
    "left": ((0, -3), (0, -2)),
    "right": ((0, 2), (0, 3)),
    "up": ((-3, 0), (-2, 0)),
    "down": ((2, 0), (3, 0)),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == BLUE):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                nxt = (nr, nc)
                if _in_bounds((h, w), nr, nc) and nxt not in seen and int(grid[nr, nc]) == BLUE:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _bar_orientation(comp: list[tuple[int, int]]) -> str | None:
    if len(comp) != 2:
        return None
    (r0, c0), (r1, c1) = sorted(comp)
    if r0 == r1 and abs(c0 - c1) == 1:
        return "h"
    if c0 == c1 and abs(r0 - r1) == 1:
        return "v"
    return None


def _cross_arm_pixels(r: int, c: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for dr, dc in ARM_OFFSETS}


def _is_cross_center(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    for dr, dc in GAP_OFFSETS:
        if int(grid[r + dr, c + dc]) != BG:
            return False
    for dr, dc in ARM_OFFSETS:
        if int(grid[r + dr, c + dc]) != BLUE:
            return False
    return True


def _detect_crosses(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    h, w = grid.shape
    crosses: list[dict] = []
    participating: set[tuple[int, int]] = set()
    for r in range(3, h - 3):
        for c in range(3, w - 3):
            if not _is_cross_center(grid, r, c):
                continue
            left_cols = [c - 3, c - 2]
            right_cols = [c + 2, c + 3]
            span = max(right_cols) - min(left_cols) + 1
            arms = _cross_arm_pixels(r, c)
            crosses.append({"center": (r, c), "width": span, "arms": arms})
            participating |= arms
    crosses.sort(key=lambda item: item["center"])
    return crosses, participating


def _base_interval(c: int, width: int) -> tuple[int, int]:
    left = c - width // 2
    right = left + width - 1
    return left, right


def _up_rows(center: tuple[int, int], width: int, shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = shape
    r, c = center
    left, right = _base_interval(c, width)
    rows: list[tuple[int, int, int]] = []
    rr = r
    step = 0
    while 0 <= rr < h and 0 <= left and right < w:
        rows.append((rr, left, right))
        if right - left + 1 >= w:
            break
        rr -= 1
        if rr < 0:
            break
        if step % 2 == 0:
            left -= 1
        else:
            right += 1
        step += 1
    return rows


def _down_rows(center: tuple[int, int], width: int, shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = shape
    r, c = center
    left, right = _base_interval(c, width)
    rows: list[tuple[int, int, int]] = []
    rr = r
    step = 0
    while 0 <= rr < h and 0 <= left and right < w and left <= right:
        rows.append((rr, left, right))
        rr += 1
        if rr >= h:
            break
        if step % 2 == 0:
            left += 1
        else:
            right -= 1
        step += 1
    return rows


def _paint_black_only(out: np.ndarray, rr: int, left: int, right: int, color: int) -> None:
    segment = out[rr, left : right + 1]
    segment[segment == BG] = color


def _wedge_claims(center: tuple[int, int], width: int, shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    claims: dict[tuple[int, int], int] = {}
    for rr, left, right in _up_rows(center, width, shape):
        for cc in range(left, right + 1):
            claims[(rr, cc)] = GREEN
    for rr, left, right in _down_rows(center, width, shape):
        for cc in range(left, right + 1):
            claims.setdefault((rr, cc), MAROON)
    return claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect hidden plus centers, grow wedges from them, and erase non-participating bars.

    Input structure:
    - The grid uses black background and blue two-pixel bars, each bar being either horizontal or vertical.
    - Some groups of four bars form an implied plus sign around an empty center, with one black gap cell between the center and every bar.
    - Other blue bars are isolated distractors and do not belong to any complete implied plus.

    Rule:
    - Mark every implied plus center yellow.
    - For each detected center, compute the horizontal span from the left arm's left pixel to the right arm's right pixel and use that as the base width.
    - Paint an upward green wedge from the center row, expanding by one cell per row with left-then-right alternating centering, and paint a downward maroon wedge from the same base, shrinking by one cell per row with the same left-then-right alternation.
    - Blue bars that participate in a detected plus stay blue above the wedges, the yellow center stays above everything, later wedges never overwrite earlier non-black wedge pixels, and any blue bar not used by a detected plus is erased to black.

    Key insight:
    - The same bar layout simultaneously decides what gets built and what gets deleted: complete four-arm pluses create anchors and widths for the wedges, while every remaining bar disappears and becomes a black hole if it lay inside a wedge region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    crosses, participating = _detect_crosses(g)
    out = g.copy()

    for cross in crosses:
        r, c = cross["center"]
        out[r, c] = YELLOW

    for cross in crosses:
        center = cross["center"]
        width = int(cross["width"])
        for rr, left, right in _up_rows(center, width, g.shape):
            _paint_black_only(out, rr, left, right, GREEN)
        for rr, left, right in _down_rows(center, width, g.shape):
            _paint_black_only(out, rr, left, right, MAROON)

    for r, c in np.argwhere(g == BLUE):
        cell = (int(r), int(c))
        if cell not in participating:
            out[cell] = BG

    for cross in crosses:
        r, c = cross["center"]
        out[r, c] = YELLOW

    return out


def _cells_isolated(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    own = set(cells)
    for r, c in cells:
        if not _in_bounds((h, w), r, c) or int(grid[r, c]) != BG:
            return False
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in own:
                continue
            if _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) == BLUE:
                return False
    return True


def _place_cells(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = BLUE


def _remove_cells(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = BG


def _row_col_candidates(height: int, width: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(3, height - 3):
        max_up_steps = min(r, width - 7)
        c_min = 3 + (max_up_steps + 1) // 2
        c_max = width - 4 - max_up_steps // 2
        for c in range(c_min, c_max + 1):
            pairs.append((r, c))
    random.shuffle(pairs)
    return pairs


def _place_crosses(
    grid: np.ndarray,
    cross_count: int,
    want_overlap: bool,
) -> list[tuple[int, int]] | None:
    candidates = _row_col_candidates(*grid.shape)
    if not want_overlap:
        mid = (grid.shape[1] - 1) / 2
        candidates.sort(key=lambda rc: abs(rc[1] - mid), reverse=True)
    centers: list[tuple[int, int]] = []
    placed_set: set[tuple[int, int]] = set()

    def search(start_idx: int) -> bool:
        if len(centers) == cross_count:
            return True
        need = cross_count - len(centers)
        for idx in range(start_idx, len(candidates)):
            if len(candidates) - idx < need:
                return False
            r, c = candidates[idx]
            if any(abs(r - rr) < 5 for rr, _cc in centers):
                continue
            arms = sorted(_cross_arm_pixels(r, c))
            support = arms + [(r, c)] + [(r + dr, c + dc) for dr, dc in GAP_OFFSETS]
            if any(not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != BG for rr, cc in support):
                continue
            if not _cells_isolated(grid, arms):
                continue
            _place_cells(grid, arms)
            test_centers = sorted(placed_set | {(r, c)})
            detected = [item["center"] for item in _detect_crosses(grid)[0]]
            if detected != test_centers:
                _remove_cells(grid, arms)
                continue
            if not want_overlap and _has_visible_overlap(test_centers, grid.shape):
                _remove_cells(grid, arms)
                continue
            centers.append((r, c))
            placed_set.add((r, c))
            if search(idx + 1):
                return True
            centers.pop()
            placed_set.remove((r, c))
            _remove_cells(grid, arms)
        return False

    if search(0):
        return sorted(centers)
    return None


def _all_candidate_bars(cell_set: set[tuple[int, int]], shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = shape
    bars: list[list[tuple[int, int]]] = []
    for r, c in sorted(cell_set):
        if c + 1 < w and (r, c + 1) in cell_set:
            bars.append([(r, c), (r, c + 1)])
        if r + 1 < h and (r + 1, c) in cell_set:
            bars.append([(r, c), (r + 1, c)])
    random.shuffle(bars)
    return bars


def _claims_by_center(
    centers: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[dict[tuple[int, int], dict[tuple[int, int], int]], set[tuple[int, int]]]:
    claims_by_center: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    union: set[tuple[int, int]] = set()
    for center in centers:
        claims = _wedge_claims(center, 7, shape)
        claims_by_center[center] = claims
        union |= set(claims)
    return claims_by_center, union


def _place_decoy_cluster(
    grid: np.ndarray,
    wedge_cells: set[tuple[int, int]],
    expected_centers: list[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    h, w = grid.shape
    directions = ["left", "right", "up", "down"]
    candidates = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
    random.shuffle(candidates)
    expected = expected_centers[:]
    for r, c in candidates:
        if (r, c) in expected_centers:
            continue
        missing = random.choice(directions)
        used_dirs = [name for name in directions if name != missing]
        bars = []
        ok = True
        for name in used_dirs:
            cells = [(r + dr, c + dc) for dr, dc in DIR_TO_BAR[name]]
            if not set(cells) <= wedge_cells or not _cells_isolated(grid, cells):
                ok = False
                break
            bars.append(cells)
        if not ok:
            continue
        for cells in bars:
            _place_cells(grid, cells)
        if [item["center"] for item in _detect_crosses(grid)[0]] != expected:
            for cells in bars:
                _remove_cells(grid, cells)
            continue
        return bars
    return None


def _place_visible_distractors(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    distractor_count: int,
    wedge_cells: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    bars: list[list[tuple[int, int]]] = []
    expected = centers[:]

    if distractor_count >= 3 and random.random() < 0.7:
        decoy = _place_decoy_cluster(grid, wedge_cells, expected)
        if decoy is not None:
            bars.extend(decoy)

    while len(bars) < distractor_count:
        placed = False
        for cells in _all_candidate_bars(wedge_cells, grid.shape):
            if not _cells_isolated(grid, cells):
                continue
            _place_cells(grid, cells)
            if [item["center"] for item in _detect_crosses(grid)[0]] != expected:
                _remove_cells(grid, cells)
                continue
            bars.append(cells)
            placed = True
            break
        if not placed:
            return None
    return bars[:distractor_count]


def _overlap_summary(
    centers: list[tuple[int, int]],
    shape: tuple[int, int],
) -> dict[tuple[int, int], list[tuple[tuple[int, int], int]]]:
    claims_by_center, _ = _claims_by_center(centers, shape)
    overlaps: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = defaultdict(list)
    for center in centers:
        for cell, color in claims_by_center[center].items():
            overlaps[cell].append((center, color))
    return overlaps


def _has_visible_overlap(
    centers: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    overlaps = _overlap_summary(centers, shape)
    arm_pixels = set().union(*(_cross_arm_pixels(r, c) for r, c in centers))
    center_pixels = set(centers)
    for cell, claimants in overlaps.items():
        if len(claimants) < 2 or cell in arm_pixels or cell in center_pixels:
            continue
        colors = {color for _center, color in claimants}
        if len(colors) >= 2:
            return True
    return False


def _sample_case() -> tuple[np.ndarray, dict] | None:
    height = random.choices(
        [19, 20, 21, 22, 23, 24, 25],
        weights=[1, 1, 2, 3, 3, 3, 3],
    )[0]
    width = random.randint(15, 21)
    max_crosses = min(4, 1 + (height - 7) // 5)
    if max_crosses < 2:
        return None

    want_overlap = random.random() < 0.6
    if max_crosses == 2:
        cross_count = 2
    elif want_overlap:
        if max_crosses == 3:
            cross_count = random.choices([2, 3], weights=[1, 2])[0]
        else:
            cross_count = random.choices([3, 4], weights=[1, 2])[0]
    else:
        cross_count = random.choices([2, min(3, max_crosses)], weights=[2, 1])[0]
    total_bars = random.randint(max(18, cross_count * 4 + 4), min(28, cross_count * 4 + 11))
    distractor_count = total_bars - cross_count * 4

    grid = np.zeros((height, width), dtype=int)
    centers = _place_crosses(grid, cross_count, want_overlap)
    if centers is None:
        return None
    if _has_visible_overlap(centers, grid.shape) != want_overlap:
        return None

    claims_by_center, wedge_union = _claims_by_center(centers, grid.shape)
    participating = set().union(*(_cross_arm_pixels(r, c) for r, c in centers))
    wedge_union -= participating
    wedge_union -= set(centers)
    if len(wedge_union) < distractor_count * 2:
        return None

    distractor_bars = _place_visible_distractors(grid, centers, distractor_count, wedge_union)
    if distractor_bars is None:
        return None

    return grid, {
        "centers": centers,
        "participating": participating,
        "distractor_bars": distractor_bars,
        "claims_by_center": claims_by_center,
        "want_overlap": want_overlap,
    }


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    comps = _blue_components(grid)
    if not comps or any(_bar_orientation(comp) is None for comp in comps):
        return False

    detected, participating = _detect_crosses(grid)
    centers = [item["center"] for item in detected]
    if centers != meta["centers"]:
        return False
    if participating != meta["participating"]:
        return False

    out = main(grid)
    for r, c in meta["centers"]:
        if int(out[r, c]) != YELLOW:
            return False

    for r, c in participating:
        if int(out[r, c]) != BLUE:
            return False

    for cells in meta["distractor_bars"]:
        if any(int(out[r, c]) != BG for r, c in cells):
            return False

    for center, claims in meta["claims_by_center"].items():
        for cell, color in claims.items():
            if cell in participating or cell in meta["centers"]:
                continue
            if any(cell in set(bar) for bar in meta["distractor_bars"]):
                if int(out[cell]) != BG:
                    return False

    if meta["want_overlap"]:
        overlaps = _overlap_summary(meta["centers"], grid.shape)
        witnessed = False
        for cell, claimants in overlaps.items():
            if len(claimants) < 2:
                continue
            if cell in participating or cell in meta["centers"]:
                continue
            order = {center: idx for idx, center in enumerate(meta["centers"])}
            claimants.sort(key=lambda item: order[item[0]])
            expected_color = claimants[0][1]
            if int(out[cell]) == expected_color:
                witnessed = True
                break
        if not witnessed:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        sample = _sample_case()
        if sample is None:
            continue
        grid, meta = sample
        if _validate_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid hidden-cross wedge puzzle")
