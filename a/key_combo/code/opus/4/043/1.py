from __future__ import annotations

from collections import Counter, deque
from importlib.machinery import SourceFileLoader
import itertools
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
components_mask = _ARC_UTILS.components_mask
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4


BLACK = Color.BLACK
WALL = Color.GREY
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_BOUNDS = (15, 20)
OUTLINE_COUNT_RANGE = (3, 4)
MARKER_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 400
SUBSET_ATTEMPTS = 200
PATCH_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 100
TIE_MODE_PROB = 0.5


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _outline_inside_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    wall = np.zeros((h, w), dtype=bool)
    local = coords - np.array([r0, c0])
    wall[local[:, 0], local[:, 1]] = True

    passable = ~wall
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    out = np.zeros(shape, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = enclosed
    return out


def _extract_contours(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    wall_comps = components_mask(grid == WALL, connectivity=4)
    wall_comps.sort(key=lambda coords: tuple(_bbox_from_coords(coords)))
    label_grid = np.zeros(grid.shape, dtype=int)
    contours: list[dict] = []

    for idx, coords in enumerate(wall_comps, start=1):
        inside_mask = _outline_inside_mask(grid.shape, coords)
        label_grid[coords[:, 0], coords[:, 1]] = idx
        contours.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "inside_mask": inside_mask,
            }
        )
    return contours, label_grid


def _extract_markers(grid: np.ndarray) -> list[dict]:
    markers: list[dict] = []
    for color in MARKER_COLORS:
        for coords in components_mask(grid == color, connectivity=4):
            if coords.shape[0] == 1:
                r, c = map(int, coords[0])
                markers.append({"pos": (r, c), "color": int(color)})
    markers.sort(key=lambda item: (item["pos"][0], item["pos"][1], item["color"]))
    return markers


def _trace_marker_hits(
    shape: tuple[int, int], label_grid: np.ndarray, pos: tuple[int, int]
) -> list[tuple[int, int, int]]:
    hits: list[tuple[int, int, int]] = []
    r0, c0 = pos
    for dr, dc in DIAGONALS:
        r = r0 + dr
        c = c0 + dc
        while in_bounds(shape, r, c):
            contour_id = int(label_grid[r, c])
            if contour_id > 0:
                hits.append((contour_id, int(r), int(c)))
                break
            r += dr
            c += dc
    return hits


def _choose_fill_color(votes: list[tuple[int, int, int, int, int]]) -> int | None:
    if not votes:
        return None
    counts = Counter(color for color, *_rest in votes)
    top = max(counts.values())
    leaders = [color for color, count in counts.items() if count == top]
    if len(leaders) == 1:
        return int(leaders[0])

    best_vote = min(
        (
            hit_r,
            hit_c,
            marker_r,
            marker_c,
            color,
        )
        for color, hit_r, hit_c, marker_r, marker_c in votes
        if color in leaders
    )
    return int(best_vote[-1])


def _classify_votes(votes: list[tuple[int, int, int, int, int]]) -> dict:
    if not votes:
        return {"winner": None, "mode": "none", "colors": set()}

    counts = Counter(color for color, *_rest in votes)
    top = max(counts.values())
    leaders = [color for color, count in counts.items() if count == top]
    colors = set(counts)
    if len(colors) == 1:
        return {"winner": int(leaders[0]), "mode": "single", "colors": colors}
    if len(leaders) == 1:
        return {"winner": int(leaders[0]), "mode": "majority", "colors": colors}

    first_hit_by_color = {
        color: min((hit_r, hit_c, marker_r, marker_c) for col, hit_r, hit_c, marker_r, marker_c in votes if col == color)
        for color in leaders
    }
    best_rc = min((value[0], value[1]) for value in first_hit_by_color.values())
    best_colors = [color for color, value in first_hit_by_color.items() if (value[0], value[1]) == best_rc]
    if len(best_colors) != 1:
        winner = _choose_fill_color(votes)
        return {"winner": int(winner), "mode": "ambiguous", "colors": colors}
    return {"winner": int(best_colors[0]), "mode": "tie", "colors": colors}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed grey outline interior using marker colors delivered by diagonal ray hits, then erase the markers.

    Input structure:
    - The background is black.
    - Grey pixels form several 4-connected closed outline objects whose interiors are black.
    - Separate isolated one-pixel markers use blue, red, green, or yellow and do not touch the outlines orthogonally.

    Rule:
    - From every marker pixel, cast one invisible ray along each diagonal direction.
    - Each ray stops at the first grey outline pixel it reaches, or at the grid boundary if no outline is hit.
    - For each outline, collect the colors of all rays that hit it and fill only its enclosed interior:
      use the unique hit color if only one color appears, otherwise use the most frequent hit color,
      and if frequencies tie use the color whose tied hit set contains the smallest hit-row, then smallest hit-column.
    - Leave unhit outline interiors black, keep grey outline pixels unchanged, and erase all marker pixels to black.

    Key insight:
    - The markers do not draw visible diagonals; they act as remote color senders whose diagonal line of sight votes on which outline interior gets which fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    contours, label_grid = _extract_contours(g)
    markers = _extract_markers(g)
    out = g.copy()

    votes_by_contour: dict[int, list[tuple[int, int, int, int, int]]] = {item["id"]: [] for item in contours}
    for marker in markers:
        marker_r, marker_c = marker["pos"]
        for contour_id, hit_r, hit_c in _trace_marker_hits(g.shape, label_grid, marker["pos"]):
            votes_by_contour[contour_id].append((marker["color"], hit_r, hit_c, marker_r, marker_c))

    for contour in contours:
        fill = _choose_fill_color(votes_by_contour[contour["id"]])
        if fill is not None:
            out[contour["inside_mask"]] = int(fill)
        else:
            out[contour["inside_mask"]] = BLACK
        out[contour["coords"][:, 0], contour["coords"][:, 1]] = WALL

    for marker in markers:
        out[marker["pos"]] = BLACK
    return out


def _add_segment(cells: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        lo, hi = sorted((int(c0), int(c1)))
        for c in range(lo, hi + 1):
            cells.add((int(r0), c))
    else:
        lo, hi = sorted((int(r0), int(r1)))
        for r in range(lo, hi + 1):
            cells.add((r, int(c0)))


def _rect_perimeter_cells(h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(h):
        cells.add((r, 0))
        cells.add((r, w - 1))
    return cells


def _apply_notch(cells: set[tuple[int, int]], h: int, w: int, side: str) -> bool:
    depth = 1
    if side in {"top", "bottom"}:
        if w < 5:
            return False
        span = random.randint(2, min(3, w - 2))
        start = random.randint(1, w - span - 1)
        end = start + span - 1
        if side == "top":
            for c in range(start + 1, end):
                cells.discard((0, c))
            _add_segment(cells, 0, start, depth, start)
            _add_segment(cells, 0, end, depth, end)
            _add_segment(cells, depth, start, depth, end)
        else:
            r = h - 1
            inner = r - depth
            for c in range(start + 1, end):
                cells.discard((r, c))
            _add_segment(cells, r, start, inner, start)
            _add_segment(cells, r, end, inner, end)
            _add_segment(cells, inner, start, inner, end)
        return True

    if h < 5:
        return False
    span = random.randint(2, min(3, h - 2))
    start = random.randint(1, h - span - 1)
    end = start + span - 1
    if side == "left":
        for r in range(start + 1, end):
            cells.discard((r, 0))
        _add_segment(cells, start, 0, start, depth)
        _add_segment(cells, end, 0, end, depth)
        _add_segment(cells, start, depth, end, depth)
    else:
        c = w - 1
        inner = c - depth
        for r in range(start + 1, end):
            cells.discard((r, c))
        _add_segment(cells, start, c, start, inner)
        _add_segment(cells, end, c, end, inner)
        _add_segment(cells, start, inner, end, inner)
    return True


def _make_outline_patch() -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        cells = _rect_perimeter_cells(h, w)
        if random.random() < 0.7:
            _apply_notch(cells, h, w, random.choice(("top", "bottom", "left", "right")))

        wall = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            wall[r, c] = True
        coords = np.argwhere(wall)
        if not (10 <= len(coords) <= 20):
            continue
        if len(components_mask(wall, connectivity=4)) != 1:
            continue

        inside = _outline_inside_mask(wall.shape, coords)
        inside_components = components_mask(inside, connectivity=4)
        if len(inside_components) != 1:
            continue
        if inside_components[0].shape[0] < 2:
            continue

        patch = np.zeros((h, w), dtype=int)
        patch[wall] = WALL
        return patch
    return None


def _expand_cells(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows, cols):
        r0 = max(0, int(r) - radius)
        r1 = min(mask.shape[0], int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(mask.shape[1], int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _place_outlines(grid: np.ndarray, n_outlines: int) -> bool:
    forbidden = np.zeros(grid.shape, dtype=bool)
    for _ in range(n_outlines):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            patch = _make_outline_patch()
            if patch is None:
                continue
            ph, pw = patch.shape
            if ph >= grid.shape[0] or pw >= grid.shape[1]:
                continue
            top = random.randint(0, grid.shape[0] - ph)
            left = random.randint(0, grid.shape[1] - pw)
            local_mask = patch == WALL
            global_mask = np.zeros(grid.shape, dtype=bool)
            global_mask[top : top + ph, left : left + pw] = local_mask
            if np.any(forbidden & global_mask):
                continue
            grid[top : top + ph, left : left + pw][local_mask] = WALL
            forbidden |= _expand_cells(global_mask, radius=2)
            placed = True
            break
        if not placed:
            return False
    return True


def _candidate_marker_positions(grid: np.ndarray, label_grid: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BLACK:
                continue
            if any(in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == WALL for nr, nc in neighbors4(r, c)):
                continue
            hits = _trace_marker_hits(grid.shape, label_grid, (r, c))
            unique_contours = {contour_id for contour_id, *_rest in hits}
            if not hits or len(unique_contours) > 2:
                continue
            candidates.append(
                {
                    "pos": (int(r), int(c)),
                    "hits": [{"contour": contour_id, "hit": (hit_r, hit_c)} for contour_id, hit_r, hit_c in hits],
                    "unique_contours": unique_contours,
                }
            )
    random.shuffle(candidates)
    return candidates


def _markers_compatible(selected: list[dict], candidate: dict) -> bool:
    r, c = candidate["pos"]
    for other in selected:
        rr, cc = other["pos"]
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _subset_contour_stats(selected: list[dict], n_contours: int) -> dict[int, dict]:
    stats = {
        contour_id: {"markers": set(), "ray_count": 0}
        for contour_id in range(1, n_contours + 1)
    }
    for idx, candidate in enumerate(selected):
        for hit in candidate["hits"]:
            contour_id = int(hit["contour"])
            stats[contour_id]["markers"].add(idx)
            stats[contour_id]["ray_count"] += 1
    return stats


def _choose_marker_subset(candidates: list[dict], n_contours: int, desired_mode: str) -> list[dict] | None:
    contour_to_ids = {contour_id: [] for contour_id in range(1, n_contours + 1)}
    for idx, candidate in enumerate(candidates):
        for contour_id in candidate["unique_contours"]:
            contour_to_ids[int(contour_id)].append(idx)

    for _ in range(SUBSET_ATTEMPTS):
        focus_options = [cid for cid, ids in contour_to_ids.items() if len(ids) >= 2]
        if not focus_options:
            return None
        focus = random.choice(focus_options)

        seed_pool = contour_to_ids[focus][:]
        random.shuffle(seed_pool)
        selected_ids: list[int] = []
        seed_size_choices = [2, 3] if desired_mode == "majority" else [2]
        seed_sizes = [size for size in seed_size_choices if size <= len(seed_pool)]
        if not seed_sizes:
            continue
        seed_size = random.choice(seed_sizes)

        for candidate_id in seed_pool:
            candidate = candidates[candidate_id]
            if _markers_compatible([candidates[i] for i in selected_ids], candidate):
                selected_ids.append(candidate_id)
            if len(selected_ids) == seed_size:
                break
        if len(selected_ids) != seed_size:
            continue

        selected = [candidates[i] for i in selected_ids]
        if desired_mode == "majority":
            stats = _subset_contour_stats(selected, n_contours)
            if stats[focus]["ray_count"] < 3:
                extra_ids = contour_to_ids[focus][:]
                random.shuffle(extra_ids)
                for extra_id in extra_ids:
                    if extra_id in selected_ids:
                        continue
                    if _markers_compatible(selected, candidates[extra_id]):
                        selected_ids.append(extra_id)
                        selected.append(candidates[extra_id])
                        break
            stats = _subset_contour_stats(selected, n_contours)
            if stats[focus]["ray_count"] < 3:
                continue

        covered = {hit["contour"] for candidate in selected for hit in candidate["hits"]}
        while len(selected) < MARKER_COUNT_RANGE[1] and covered != set(range(1, n_contours + 1)):
            options = []
            for idx, candidate in enumerate(candidates):
                if idx in selected_ids or not _markers_compatible(selected, candidate):
                    continue
                gain = len({hit["contour"] for hit in candidate["hits"]} - covered)
                if gain > 0:
                    options.append((gain, random.random(), idx))
            if not options:
                break
            _, _, best_id = max(options)
            selected_ids.append(best_id)
            selected.append(candidates[best_id])
            covered |= {hit["contour"] for hit in candidates[best_id]["hits"]}

        if covered != set(range(1, n_contours + 1)):
            continue

        while len(selected) < MARKER_COUNT_RANGE[0]:
            extras = [
                idx
                for idx, candidate in enumerate(candidates)
                if idx not in selected_ids and _markers_compatible(selected, candidate)
            ]
            if not extras:
                break
            best_id = random.choice(extras)
            selected_ids.append(best_id)
            selected.append(candidates[best_id])

        if not (MARKER_COUNT_RANGE[0] <= len(selected) <= MARKER_COUNT_RANGE[1]):
            continue

        stats = _subset_contour_stats(selected, n_contours)
        if not any(len(stats[cid]["markers"]) >= 2 for cid in stats):
            continue
        return selected
    return None


def _iter_color_assignments(n_markers: int) -> itertools.product:
    counts = {color: 0 for color in MARKER_COLORS}
    current = [0] * n_markers

    def rec(i: int):
        if i == n_markers:
            yield tuple(current)
            return
        colors = list(MARKER_COLORS)
        random.shuffle(colors)
        for color in colors:
            if counts[color] >= 2:
                continue
            counts[color] += 1
            current[i] = int(color)
            yield from rec(i + 1)
            counts[color] -= 1

    return rec(0)


def _assignment_vote_map(selected: list[dict], colors: tuple[int, ...], n_contours: int) -> dict[int, list[tuple[int, int, int, int, int]]]:
    votes_by_contour = {contour_id: [] for contour_id in range(1, n_contours + 1)}
    for idx, candidate in enumerate(selected):
        marker_r, marker_c = candidate["pos"]
        for hit in candidate["hits"]:
            hit_r, hit_c = hit["hit"]
            votes_by_contour[int(hit["contour"])].append((int(colors[idx]), int(hit_r), int(hit_c), marker_r, marker_c))
    return votes_by_contour


def _find_color_assignment(selected: list[dict], n_contours: int, desired_mode: str) -> tuple[tuple[int, ...], dict[int, dict]] | None:
    for colors in _iter_color_assignments(len(selected)):
        votes_by_contour = _assignment_vote_map(selected, colors, n_contours)
        summaries = {cid: _classify_votes(votes) for cid, votes in votes_by_contour.items()}
        if any(summary["mode"] in {"none", "ambiguous"} for summary in summaries.values()):
            continue
        if desired_mode not in {summary["mode"] for summary in summaries.values()}:
            continue
        if not any(len(summary["colors"]) >= 2 for summary in summaries.values()):
            continue
        return colors, summaries
    return None


def _build_candidate_grid() -> tuple[np.ndarray, str] | None:
    h = random.randint(*GRID_BOUNDS)
    w = random.randint(*GRID_BOUNDS)
    n_outlines = random.randint(*OUTLINE_COUNT_RANGE)
    desired_mode = "tie" if random.random() < TIE_MODE_PROB else "majority"

    grid = np.zeros((h, w), dtype=int)
    if not _place_outlines(grid, n_outlines):
        return None

    contours, label_grid = _extract_contours(grid)
    candidates = _candidate_marker_positions(grid, label_grid)
    if len(candidates) < MARKER_COUNT_RANGE[0]:
        return None

    selected = _choose_marker_subset(candidates, len(contours), desired_mode)
    if selected is None:
        return None

    assignment = _find_color_assignment(selected, len(contours), desired_mode)
    if assignment is None:
        return None
    colors, _summaries = assignment

    for candidate, color in zip(selected, colors):
        r, c = candidate["pos"]
        grid[r, c] = int(color)

    return grid, desired_mode


def _validate_generated(grid: np.ndarray, expected_mode: str | None = None) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert int(get_bg(g)) == BLACK

    contours, label_grid = _extract_contours(g)
    assert OUTLINE_COUNT_RANGE[0] <= len(contours) <= OUTLINE_COUNT_RANGE[1]
    for contour in contours:
        coords = contour["coords"]
        assert 10 <= coords.shape[0] <= 20
        inside = contour["inside_mask"]
        inside_components = components_mask(inside, connectivity=4)
        assert len(inside_components) == 1
        assert np.all(g[inside] == BLACK)

    markers = _extract_markers(g)
    assert MARKER_COUNT_RANGE[0] <= len(markers) <= MARKER_COUNT_RANGE[1]
    marker_counts = Counter(marker["color"] for marker in markers)
    assert all(count <= 2 for count in marker_counts.values())

    marker_positions = [marker["pos"] for marker in markers]
    assert len(set(marker_positions)) == len(marker_positions)
    for idx, marker in enumerate(markers):
        r, c = marker["pos"]
        assert any(_trace_marker_hits(g.shape, label_grid, marker["pos"]))
        for nr, nc in neighbors4(r, c):
            if in_bounds(g.shape, nr, nc):
                assert int(g[nr, nc]) != WALL
        for other in markers[idx + 1 :]:
            rr, cc = other["pos"]
            assert max(abs(r - rr), abs(c - cc)) > 1

    votes_by_contour = {contour["id"]: [] for contour in contours}
    for marker in markers:
        marker_r, marker_c = marker["pos"]
        for contour_id, hit_r, hit_c in _trace_marker_hits(g.shape, label_grid, marker["pos"]):
            votes_by_contour[contour_id].append((marker["color"], hit_r, hit_c, marker_r, marker_c))

    summaries = {cid: _classify_votes(votes) for cid, votes in votes_by_contour.items()}
    assert all(summary["mode"] not in {"none", "ambiguous"} for summary in summaries.values())
    assert any(len(summary["colors"]) >= 2 for summary in summaries.values())
    if expected_mode is not None:
        assert expected_mode in {summary["mode"] for summary in summaries.values()}

    out = main(g)
    for contour in contours:
        cid = contour["id"]
        winner = summaries[cid]["winner"]
        if winner is None:
            assert np.all(out[contour["inside_mask"]] == BLACK)
        else:
            assert np.all(out[contour["inside_mask"]] == int(winner))
        assert np.all(out[contour["coords"][:, 0], contour["coords"][:, 1]] == WALL)
    for marker in markers:
        assert int(out[marker["pos"]]) == BLACK


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_candidate_grid()
        if built is None:
            continue
        grid, mode = built
        try:
            _validate_generated(grid, expected_mode=mode)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid diagonal-ray flood-fill input")
