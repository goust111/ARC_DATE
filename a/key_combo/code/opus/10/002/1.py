from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_NONBLACK = tuple(range(1, 10))

GRID_MIN = 18
GRID_MAX = 24
N_CROSS_CHOICES = (3, 4)  # 5 crosses would require 10 distinct non-black colors.
GAP_CHOICES = (5, 6, 6, 7)
MARKER_COUNT_RANGE = (4, 8)
GEN_ATTEMPTS = 900
MARKER_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr != 0 or dc != 0:
                yield r + dr, c + dc


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTH_DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _cross_input_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _find_crosses_and_markers(grid: np.ndarray):
    comps = _components4(grid != BLACK)
    crosses = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        if len(comp) == 1:
            (r, c) = comp[0]
            markers[(r, c)] = int(grid[r, c])
            continue

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert len(comp) == 5 and (r1 - r0, c1 - c0) == (2, 2), "expected a 3x3 cross"
        center = (r0 + 1, c0 + 1)
        center_color = int(grid[center])
        arm_cells = [(r0, c0 + 1), (r1, c0 + 1), (r0 + 1, c0), (r0 + 1, c1)]
        arm_colors = {int(grid[r, c]) for r, c in arm_cells}
        assert len(arm_colors) == 1, "cross arms must share one color"
        arm_color = arm_colors.pop()
        assert center_color != arm_color and center_color != BLACK and arm_color != BLACK
        assert set(comp) == _cross_input_cells(center), "unexpected non-cross foreground component"
        crosses.append(
            {
                "center": center,
                "arm_color": arm_color,
                "center_color": center_color,
            }
        )

    arm_colors = [cross["arm_color"] for cross in crosses]
    center_colors = [cross["center_color"] for cross in crosses]
    assert len(set(arm_colors)) == len(arm_colors)
    assert len(set(center_colors)) == len(center_colors)
    assert not (set(arm_colors) & set(center_colors))
    return crosses, markers


def _resolve_expansion(crosses, shape: tuple[int, int]):
    claims: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for idx, cross in enumerate(crosses):
        r, c = cross["center"]
        claims.setdefault((r, c), []).append((idx, 1))
        for step in range(1, 4):
            for dr, dc in ORTH_DIRS:
                pos = (r + dr * step, c + dc * step)
                if _in_bounds(shape, pos[0], pos[1]):
                    claims.setdefault(pos, []).append((idx, 2))
        for step in range(1, 3):
            for dr, dc in DIAG_DIRS:
                pos = (r + dr * step, c + dc * step)
                if _in_bounds(shape, pos[0], pos[1]):
                    claims.setdefault(pos, []).append((idx, 1))

    expanded = np.zeros(shape, dtype=int)
    owner = np.full(shape, -1, dtype=int)
    role = np.zeros(shape, dtype=int)  # 0 none, 1 center-role, 2 arm-role
    overlap_cells = 0
    distance_tie_overlap = False

    for (r, c), cell_claims in claims.items():
        if len(cell_claims) > 1:
            overlap_cells += 1
        ranked = []
        for idx, cell_role in cell_claims:
            cr, cc = crosses[idx]["center"]
            ranked.append((abs(r - cr) + abs(c - cc), cr, cc, idx, cell_role))
        ranked.sort()
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            distance_tie_overlap = True
        _dist, _row, _col, idx, cell_role = ranked[0]
        owner[r, c] = idx
        role[r, c] = cell_role
        if cell_role == 1:
            expanded[r, c] = crosses[idx]["center_color"]
        else:
            expanded[r, c] = crosses[idx]["arm_color"]

    return expanded, owner, role, overlap_cells, distance_tie_overlap


def _closed_black_regions(board: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = board.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if board[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in ORTH_DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(board.shape, nr, nc) and board[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                regions.append(coords)
    return regions


def _pick_by_count(counts: Counter[int], crosses) -> int | None:
    if not counts:
        return None
    best_count = max(counts.values())
    candidates = [idx for idx, count in counts.items() if count == best_count]
    return min(candidates, key=lambda idx: (crosses[idx]["center"][0], crosses[idx]["center"][1]))


def _analyze_objects(crosses, markers: dict[tuple[int, int], int], shape: tuple[int, int]):
    expanded, owner, role, overlap_cells, distance_tie_overlap = _resolve_expansion(crosses, shape)
    board = expanded.copy()
    for (r, c), color in markers.items():
        assert board[r, c] == BLACK
        board[r, c] = color

    regions = _closed_black_regions(board)
    center_owner_by_color = {cross["center_color"]: idx for idx, cross in enumerate(crosses)}
    out = board.copy()
    region_infos = []

    for cells in regions:
        touched_center_pixels: set[tuple[int, int]] = set()
        touched_arm_pixels: set[tuple[int, int]] = set()
        touched_markers: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors8(r, c):
                if not _in_bounds(shape, nr, nc):
                    continue
                if role[nr, nc] == 1:
                    touched_center_pixels.add((nr, nc))
                elif role[nr, nc] == 2:
                    touched_arm_pixels.add((nr, nc))
                if (nr, nc) in markers and markers[(nr, nc)] in center_owner_by_color:
                    touched_markers.add((nr, nc))

        center_counts = Counter(owner[r, c] for r, c in touched_center_pixels)
        if not center_counts:
            return None

        best_count = max(center_counts.values())
        tied_winners = [idx for idx, count in center_counts.items() if count == best_count]
        centroid = (
            sum(r for r, _ in cells) // len(cells),
            sum(c for _, c in cells) // len(cells),
        )
        ranked_tied = sorted(
            tied_winners,
            key=lambda idx: (
                abs(crosses[idx]["center"][0] - centroid[0]) + abs(crosses[idx]["center"][1] - centroid[1]),
                crosses[idx]["center"][0],
                crosses[idx]["center"][1],
            ),
        )
        best_distance = abs(crosses[ranked_tied[0]]["center"][0] - centroid[0]) + abs(
            crosses[ranked_tied[0]]["center"][1] - centroid[1]
        )
        distance_tie = sum(
            1
            for idx in tied_winners
            if abs(crosses[idx]["center"][0] - centroid[0]) + abs(crosses[idx]["center"][1] - centroid[1])
            == best_distance
        ) > 1
        winner = ranked_tied[0]

        arm_counts = Counter(owner[r, c] for r, c in touched_arm_pixels)
        arm_winner = _pick_by_count(arm_counts, crosses)
        nearest_owner = min(
            range(len(crosses)),
            key=lambda idx: (
                abs(crosses[idx]["center"][0] - centroid[0]) + abs(crosses[idx]["center"][1] - centroid[1]),
                crosses[idx]["center"][0],
                crosses[idx]["center"][1],
            ),
        )
        wrong_counts = Counter(center_counts)
        for marker_pos in touched_markers:
            wrong_counts[center_owner_by_color[markers[marker_pos]]] += 1
        wrong_winner = _pick_by_count(wrong_counts, crosses)

        fill_color = crosses[winner]["center_color"]
        for r, c in cells:
            out[r, c] = fill_color
        for (r, c), color in markers.items():
            out[r, c] = color

        region_infos.append(
            {
                "cells": cells,
                "size": len(cells),
                "winner": winner,
                "winner_color": fill_color,
                "center_counts": center_counts,
                "tied_vote": len(tied_winners) > 1,
                "distance_tie": distance_tie,
                "centroid": centroid,
                "nearest_owner": nearest_owner,
                "arm_winner": arm_winner,
                "wrong_marker_winner": wrong_winner,
            }
        )

    return {
        "expanded": expanded,
        "board": board,
        "out": out,
        "regions": region_infos,
        "overlap_cells": overlap_cells,
        "distance_tie_overlap": distance_tie_overlap,
    }


def _render_input(crosses, markers: dict[tuple[int, int], int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for cross in crosses:
        r, c = cross["center"]
        grid[r, c] = cross["center_color"]
        grid[r - 1, c] = cross["arm_color"]
        grid[r + 1, c] = cross["arm_color"]
        grid[r, c - 1] = cross["arm_color"]
        grid[r, c + 1] = cross["arm_color"]
    for (r, c), color in markers.items():
        assert grid[r, c] == BLACK
        grid[r, c] = color
    return grid


def _solve_objects(crosses, markers: dict[tuple[int, int], int], shape: tuple[int, int]) -> np.ndarray:
    analysis = _analyze_objects(crosses, markers, shape)
    assert analysis is not None
    return analysis["out"]


def main(grid: np.ndarray) -> np.ndarray:
    """Expand small crosses into a barrier network, then fill each enclosed region by the winning center color around its boundary.

    Input structure:
    - Black is the background.
    - The foreground contains several isolated 3x3 crosses; each cross has one arm color and one different center color.
    - Additional isolated single-pixel markers use only cross-center colors and sit in empty gaps without touching the crosses.

    Rule:
    - Detect every 3x3 cross and expand it from its center: extend the arm color to orthogonal distance 3 and extend the center color two steps along all four diagonals.
    - If multiple expansions claim one cell, keep the cross whose center has smaller Manhattan distance to that cell; break equal distances by smaller center row, then smaller center column.
    - Treat all expanded cross pixels as barriers, keep the original single-pixel markers unchanged, and flood-fill only the remaining black cells into 4-connected enclosed regions that do not touch the grid border.
    - For each enclosed region, count the distinct expanded center-role pixels that touch the region in the 8-neighborhood; fill the whole region with the majority center color, breaking vote ties by the closest cross center to the floored region centroid.

    Key insight:
    - The center color has a double role: its diagonal expansion helps build the barrier topology and the same expanded center-color pixels vote on how each enclosed compartment should be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    crosses, markers = _find_crosses_and_markers(g)
    analysis = _analyze_objects(crosses, markers, g.shape)
    assert analysis is not None
    return analysis["out"]


def _sample_layout(h: int, w: int, n_crosses: int, mode: str):
    for _ in range(250):
        if mode == "overlap_tie":
            if random.random() < 0.5:
                row_gap = 6
                col_gap = random.choice((5, 6, 7))
            else:
                row_gap = random.choice((5, 6, 7))
                col_gap = 6
        else:
            row_gap = random.choice(GAP_CHOICES)
            col_gap = random.choice(GAP_CHOICES)

        if row_gap > h - 8 or col_gap > w - 8:
            continue
        r0 = random.randint(3, h - 4 - row_gap)
        c0 = random.randint(3, w - 4 - col_gap)
        rect = [
            (r0, c0),
            (r0, c0 + col_gap),
            (r0 + row_gap, c0),
            (r0 + row_gap, c0 + col_gap),
        ]
        if n_crosses == 4:
            return rect

        missing_idx = random.randrange(4)
        return [point for idx, point in enumerate(rect) if idx != missing_idx]
    return None


def _marker_candidates(crosses, shape: tuple[int, int], analysis):
    input_cross_cells = set()
    for cross in crosses:
        input_cross_cells |= _cross_input_cells(cross["center"])

    region_id: dict[tuple[int, int], int] = {}
    for idx, info in enumerate(analysis["regions"]):
        for cell in info["cells"]:
            region_id[cell] = idx

    candidates = []
    boundary_candidates: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(analysis["regions"]))}
    for r in range(shape[0]):
        for c in range(shape[1]):
            if analysis["board"][r, c] != BLACK:
                continue
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in input_cross_cells):
                continue
            candidates.append((r, c))
            if (r, c) in region_id:
                if any(
                    _in_bounds(shape, nr, nc) and analysis["expanded"][nr, nc] != BLACK
                    for nr, nc in _neighbors8(r, c)
                ):
                    boundary_candidates[region_id[(r, c)]].append((r, c))
    return candidates, boundary_candidates


def _build_markers(crosses, shape: tuple[int, int], mode: str):
    base = _analyze_objects(crosses, {}, shape)
    if base is None:
        return None
    if len(base["regions"]) < 2 or len(base["regions"]) > 8:
        return None
    if mode == "overlap_tie" and not base["distance_tie_overlap"]:
        return None

    candidates, boundary_candidates = _marker_candidates(crosses, shape, base)
    if len(candidates) < MARKER_COUNT_RANGE[0]:
        return None

    center_colors = [cross["center_color"] for cross in crosses]
    region_ids = list(range(len(base["regions"])))
    misleading_regions = [idx for idx in region_ids if len(boundary_candidates[idx]) >= 2]

    for _ in range(MARKER_ATTEMPTS):
        marker_total = random.randint(*MARKER_COUNT_RANGE)
        markers: dict[tuple[int, int], int] = {}

        if misleading_regions:
            target_region = random.choice(misleading_regions)
            winner = base["regions"][target_region]["winner"]
            alt_owners = [idx for idx in range(len(crosses)) if idx != winner]
            alt_owner = random.choice(alt_owners)
            region_cells = boundary_candidates[target_region][:]
            random.shuffle(region_cells)
            for pos in region_cells:
                if any(max(abs(pos[0] - other[0]), abs(pos[1] - other[1])) <= 1 for other in markers):
                    continue
                markers[pos] = crosses[alt_owner]["center_color"]
                if len(markers) >= 2:
                    break

        pool = candidates[:]
        random.shuffle(pool)
        for pos in pool:
            if len(markers) >= marker_total:
                break
            if pos in markers:
                continue
            if any(max(abs(pos[0] - other[0]), abs(pos[1] - other[1])) <= 1 for other in markers):
                continue
            markers[pos] = random.choice(center_colors)

        if len(markers) < MARKER_COUNT_RANGE[0]:
            continue
        if len(set(markers.values())) < 2:
            continue

        analysis = _analyze_objects(crosses, markers, shape)
        if analysis is None:
            continue
        if len(analysis["regions"]) < 2 or len(analysis["regions"]) > 8:
            continue
        if any(info["distance_tie"] for info in analysis["regions"]):
            continue
        if len({info["winner_color"] for info in analysis["regions"]}) < 2:
            continue
        if not any(info["winner"] != info["nearest_owner"] for info in analysis["regions"]):
            continue
        if not any(info["arm_winner"] is not None and info["winner"] != info["arm_winner"] for info in analysis["regions"]):
            continue
        if not any(
            info["wrong_marker_winner"] is not None and info["wrong_marker_winner"] != info["winner"]
            for info in analysis["regions"]
        ):
            continue
        if mode == "region_tie" and not any(info["tied_vote"] for info in analysis["regions"]):
            continue
        if mode == "overlap_tie" and not analysis["distance_tie_overlap"]:
            continue
        return markers, analysis

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        mode = random.choices(
            population=("plain", "overlap_tie", "region_tie"),
            weights=(6, 2, 1),
            k=1,
        )[0]
        n_crosses = random.choice((4, 4, 4, 3))
        centers = _sample_layout(h, w, n_crosses, mode)
        if centers is None:
            continue

        colors = random.sample(list(ALL_NONBLACK), 2 * n_crosses)
        crosses = [
            {
                "center": centers[idx],
                "arm_color": colors[idx],
                "center_color": colors[n_crosses + idx],
            }
            for idx in range(n_crosses)
        ]

        built = _build_markers(crosses, (h, w), mode)
        if built is None:
            continue
        markers, analysis = built
        grid = _render_input(crosses, markers, (h, w))
        out = main(grid)
        if not np.array_equal(out, analysis["out"]):
            continue
        if np.array_equal(grid, out):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ten-cross-expansion partition example")
