from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BODY_COLORS = (RED, GREEN, YELLOW, PINK, TEAL)
CONTROL_COLORS = {BLUE, MAROON}
SPECIAL_COLORS = {BG, GREY, ORANGE}
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("up", "right", "down", "left")

# Canonical templates point east: (0, 0) is the cell immediately beyond the arm end.
EAST_TEMPLATES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (-1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (-1, 0), (1, 0)),
    ((0, 0), (-1, 0), (1, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1), (0, 1)),
    ((0, 0), (-1, 0), (0, 1), (1, 0)),
)
OVERLAP_TEMPLATES = (
    ((0, 0), (-1, 0), (1, 0)),
    ((0, 0), (-1, 0), (1, 0), (0, 1)),
    ((0, 0), (-1, 0), (0, 1), (1, 0)),
)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _transform_template(template: tuple[tuple[int, int], ...], direction: str) -> list[tuple[int, int]]:
    if direction == "right":
        return [(r, c) for r, c in template]
    if direction == "left":
        return [(r, -c) for r, c in template]
    if direction == "up":
        return [(-c, r) for r, c in template]
    if direction == "down":
        return [(c, -r) for r, c in template]
    raise ValueError(direction)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _component_from_start(
    grid: np.ndarray,
    start: tuple[int, int],
    allowed: set[int],
    claimed: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    assert _in_bounds((h, w), sr, sc), "marker seed must stay in bounds"
    assert int(grid[sr, sc]) in allowed, "marker seed must be a marker cell"
    q = deque([start])
    seen = {start}
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc):
                continue
            if (nr, nc) in seen or (nr, nc) in claimed:
                continue
            if int(grid[nr, nc]) not in allowed:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return comp


def _marker_body_and_control(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, int, tuple[int, int]]:
    controls = [(r, c) for r, c in coords if int(grid[r, c]) in CONTROL_COLORS]
    assert len(controls) == 1, "each marker must contain exactly one control pixel"
    control_pos = controls[0]
    control_color = int(grid[control_pos])
    body_colors = {int(grid[r, c]) for r, c in coords if int(grid[r, c]) not in CONTROL_COLORS}
    assert len(body_colors) == 1, "each marker must have one visible body color"
    body_color = next(iter(body_colors))
    return body_color, control_color, control_pos


def _find_crosses(grid: np.ndarray) -> list[dict]:
    crosses: list[dict] = []
    for comp in _components_of_color(grid, GREY):
        pts = set(comp)
        centers = [
            (r, c)
            for r, c in comp
            if all((r + dr, c + dc) in pts for dr, dc in DIRS.values())
        ]
        assert len(centers) == 1, "each grey component must be one plus anchor"
        center = centers[0]
        arms: dict[str, dict] = {}
        for direction in DIR_ORDER:
            dr, dc = DIRS[direction]
            r, c = center
            length = 0
            while (r + dr, c + dc) in pts:
                r += dr
                c += dc
                length += 1
            assert 2 <= length <= 3, "cross arm length must be 2 or 3"
            arms[direction] = {"end": (r, c), "length": length}
        crosses.append({"center": center, "coords": pts, "arms": arms})
    assert 2 <= len(crosses) <= 3, "expected 2 or 3 crosses"
    crosses.sort(key=lambda item: item["center"])
    return crosses


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    crosses = _find_crosses(g)
    claimed: set[tuple[int, int]] = set()
    allowed = set(range(10)) - SPECIAL_COLORS
    markers: list[dict] = []

    for cross_index, cross in enumerate(crosses):
        for direction in DIR_ORDER:
            dr, dc = DIRS[direction]
            er, ec = cross["arms"][direction]["end"]
            seed = (er + dr, ec + dc)
            marker_cells = _component_from_start(g, seed, allowed, claimed)
            assert 2 <= len(marker_cells) <= 4, "marker area must be 2..4"
            for cell in marker_cells:
                claimed.add(cell)
            body_color, control_color, control_pos = _marker_body_and_control(g, marker_cells)
            r0, c0, r1, c1 = _bbox(marker_cells)
            extent = (c1 - c0 + 1) if direction in {"left", "right"} else (r1 - r0 + 1)
            markers.append(
                {
                    "cross_index": cross_index,
                    "center": cross["center"],
                    "direction": direction,
                    "dir_vec": DIRS[direction],
                    "coords": tuple(sorted(marker_cells)),
                    "body_color": body_color,
                    "control_color": control_color,
                    "control_pos": control_pos,
                    "bbox": (r0, c0, r1, c1),
                    "extent": extent,
                }
            )

    assert len(markers) == 4 * len(crosses), "each arm must own one marker"
    for cross_index in range(len(crosses)):
        branch_colors = {m["control_color"] for m in markers if m["cross_index"] == cross_index}
        assert BLUE in branch_colors and MAROON in branch_colors, "each cross must expose both branches"
    return crosses, markers


def _max_gravity_shift(marker: dict, shape: tuple[int, int], obstacles: set[tuple[int, int]]) -> int:
    h, w = shape
    dr, dc = marker["dir_vec"]
    shift = 0
    while True:
        nxt = shift + 1
        blocked = False
        for r, c in marker["coords"]:
            rr = r + dr * nxt
            cc = c + dc * nxt
            if not _in_bounds((h, w), rr, cc) or (rr, cc) in obstacles:
                blocked = True
                break
        if blocked:
            return shift
        shift = nxt


def _collect_candidates(
    shape: tuple[int, int],
    markers: list[dict],
    protected: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], tuple[int, int]], list[tuple[int, int, int, int]]]:
    h, w = shape
    chosen: dict[tuple[int, int], tuple[int, int]] = {}
    proposals: list[tuple[int, int, int, int]] = []

    for marker in markers:
        dr, dc = marker["dir_vec"]
        shifts: list[int]
        if marker["control_color"] == BLUE:
            shifts = []
            step = marker["extent"]
            offset = step
            while True:
                moved = [(r + dr * offset, c + dc * offset) for r, c in marker["coords"]]
                if not all(_in_bounds((h, w), rr, cc) for rr, cc in moved):
                    break
                shifts.append(offset)
                offset += step
        else:
            fall = _max_gravity_shift(marker, shape, obstacles)
            assert fall > 0, "maroon marker must move at least one cell"
            shifts = [fall]

        for offset in shifts:
            for r, c in marker["coords"]:
                rr = r + dr * offset
                cc = c + dc * offset
                if (rr, cc) in protected:
                    continue
                dist = abs(rr - marker["center"][0]) + abs(cc - marker["center"][1])
                proposals.append((rr, cc, dist, marker["body_color"]))
                key = (rr, cc)
                prev = chosen.get(key)
                if prev is None or dist < prev[0]:
                    chosen[key] = (dist, marker["body_color"])
    return chosen, proposals


def _render_from_markers(grid: np.ndarray, markers: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    protected = {
        (r, c)
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if int(g[r, c]) not in {BG, ORANGE}
    }
    obstacles = {(r, c) for r, c in zip(*np.where(g == ORANGE))}
    out = g.copy()
    for marker in markers:
        rr, cc = marker["control_pos"]
        out[rr, cc] = marker["body_color"]
    chosen, _ = _collect_candidates(g.shape, markers, protected, obstacles)
    for (r, c), (_dist, color) in chosen.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each cross-adjacent marker into a periodic-tiling branch or a one-drop branch based on its hidden control pixel.

    Input structure:
    - Black background contains 2 or 3 grey plus anchors, each with four orthogonal arms of length 2 or 3.
    - Immediately beyond every arm end sits one small connected marker block of area 2 to 4.
    - Each marker uses one visible body color plus exactly one hidden control pixel, blue or maroon.
    - Orange obstacle pixels are isolated from the source crosses and markers.

    Rule:
    - For each marker, use the arm it touches to determine the outward extension direction away from the grey cross center.
    - Blue-controlled markers repeat by periodic tiling in that direction, using their size along that direction as the translation step, until the next full copy would leave the grid; source crosses and source markers are never overwritten, while obstacles are ignored and may be covered.
    - Maroon-controlled markers create exactly one translated copy by sliding cell-by-cell in the same direction until the next step would hit an orange obstacle or leave the grid.
    - After all branches are generated, every original or copied control pixel is recolored to the marker body color, so the control color disappears from the output.
    - If generated pixels from different markers compete for the same cell, the winner is the marker whose source cross center has smaller Manhattan distance to that output cell.

    Key insight:
    - The only difference between two visually similar markers is one hidden blue or maroon control pixel, yet that invisible choice switches the arm extension between periodic repetition and single-copy obstacle-stopped motion.
    """
    _crosses, markers = _parse_scene(np.asarray(grid, dtype=int))
    return _render_from_markers(np.asarray(grid, dtype=int), markers)


def _sample_branch_map(rng: random.Random, forced: dict[str, int] | None = None) -> dict[str, int]:
    branches = {direction: rng.choice((BLUE, MAROON)) for direction in DIR_ORDER}
    if forced:
        branches.update(forced)
    if BLUE not in branches.values():
        branches[rng.choice(DIR_ORDER)] = BLUE
    if MAROON not in branches.values():
        options = [d for d in DIR_ORDER if forced is None or d not in forced]
        branches[rng.choice(options)] = MAROON
    return branches


def _source_adjacency_ok(
    object_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    for r, c in object_cells:
        if not _in_bounds((h, w), r, c) or (r, c) in occupied:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (nr, nc) in occupied:
                    return False
    return True


def _make_cross_object(
    center: tuple[int, int],
    arm_len: int,
    branch_map: dict[str, int],
    template_choices: dict[str, tuple[tuple[int, int], ...]],
    body_colors: dict[str, int],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[dict]]:
    cr, cc = center
    cross_cells = {center}
    source_cells = {center}
    markers: list[dict] = []
    for direction in DIR_ORDER:
        dr, dc = DIRS[direction]
        for step in range(1, arm_len + 1):
            cell = (cr + dr * step, cc + dc * step)
            cross_cells.add(cell)
            source_cells.add(cell)
        end = (cr + dr * arm_len, cc + dc * arm_len)
        template = template_choices[direction]
        transformed = _transform_template(template, direction)
        marker_cells = [(end[0] + dr + rr, end[1] + dc + cc2) for rr, cc2 in transformed]
        control_pos = rng.choice(marker_cells)
        marker = {
            "center": center,
            "direction": direction,
            "dir_vec": (dr, dc),
            "coords": tuple(sorted(marker_cells)),
            "body_color": body_colors[direction],
            "control_color": branch_map[direction],
            "control_pos": control_pos,
        }
        markers.append(marker)
        source_cells.update(marker_cells)
    return cross_cells, source_cells, markers


def _axis_extent(coords: tuple[tuple[int, int], ...], direction: str) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    return (c1 - c0 + 1) if direction in {"left", "right"} else (r1 - r0 + 1)


def _max_shift_before_boundary(coords: tuple[tuple[int, int], ...], direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    if direction == "right":
        return min(w - 1 - c for _, c in coords)
    if direction == "left":
        return min(c for _, c in coords)
    if direction == "down":
        return min(h - 1 - r for r, _ in coords)
    return min(r for r, _ in coords)


def _front_cells(coords: tuple[tuple[int, int], ...], direction: str) -> list[tuple[int, int]]:
    if direction == "right":
        best = max(c for _, c in coords)
        return [(r, c) for r, c in coords if c == best]
    if direction == "left":
        best = min(c for _, c in coords)
        return [(r, c) for r, c in coords if c == best]
    if direction == "down":
        best = max(r for r, _ in coords)
        return [(r, c) for r, c in coords if r == best]
    best = min(r for r, _ in coords)
    return [(r, c) for r, c in coords if r == best]


def _path_cells_before(
    marker: dict,
    last_clear_shift: int,
) -> set[tuple[int, int]]:
    dr, dc = marker["dir_vec"]
    path = set()
    for offset in range(1, last_clear_shift + 1):
        for r, c in marker["coords"]:
            path.add((r + dr * offset, c + dc * offset))
    return path


def _choose_obstacle_for_maroon(
    marker: dict,
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[tuple[int, int], int] | None:
    max_shift = _max_shift_before_boundary(marker["coords"], marker["direction"], shape)
    if max_shift < 2:
        return None
    dr, dc = marker["dir_vec"]
    fronts = _front_cells(marker["coords"], marker["direction"])
    shifts = list(range(1, max_shift))
    rng.shuffle(shifts)
    for stop_shift in shifts:
        for fr, fc in rng.sample(fronts, len(fronts)):
            obstacle = (fr + dr * (stop_shift + 1), fc + dc * (stop_shift + 1))
            if obstacle in forbidden:
                continue
            path = _path_cells_before(marker, stop_shift)
            if obstacle in path:
                continue
            return obstacle, stop_shift
    return None


def _paint_scene(
    shape: tuple[int, int],
    crosses: list[dict],
    markers: list[dict],
    obstacles: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for cross in crosses:
        for r, c in cross["coords"]:
            grid[r, c] = GREY
    for marker in markers:
        for r, c in marker["coords"]:
            grid[r, c] = marker["body_color"]
        rr, cc = marker["control_pos"]
        grid[rr, cc] = marker["control_color"]
    for r, c in obstacles:
        grid[r, c] = ORANGE
    return grid


def _metadata_output(shape: tuple[int, int], markers: list[dict], grid: np.ndarray) -> np.ndarray:
    enriched = []
    for marker in markers:
        r0, c0, r1, c1 = _bbox(marker["coords"])
        item = dict(marker)
        item["bbox"] = (r0, c0, r1, c1)
        item["extent"] = _axis_extent(marker["coords"], marker["direction"])
        enriched.append(item)
    return _render_from_markers(grid, enriched)


def _conflict_stats(markers: list[dict], shape: tuple[int, int], grid: np.ndarray) -> tuple[int, bool]:
    enriched = []
    for marker in markers:
        r0, c0, r1, c1 = _bbox(marker["coords"])
        item = dict(marker)
        item["bbox"] = (r0, c0, r1, c1)
        item["extent"] = _axis_extent(marker["coords"], marker["direction"])
        enriched.append(item)
    protected = {
        (r, c)
        for r in range(shape[0])
        for c in range(shape[1])
        if int(grid[r, c]) not in {BG, ORANGE}
    }
    obstacles = {(r, c) for r, c in zip(*np.where(grid == ORANGE))}
    _chosen, proposals = _collect_candidates(shape, enriched, protected, obstacles)
    by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c, dist, color in proposals:
        by_cell.setdefault((r, c), []).append((dist, color))
    overlap = 0
    tie = False
    for vals in by_cell.values():
        if len(vals) >= 2:
            overlap += 1
            dists = sorted(dist for dist, _ in vals)
            if len(dists) >= 2 and dists[0] == dists[1]:
                tie = True
    return overlap, tie


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 26)
    w = rng.randint(20, 26)
    shape = (h, w)
    need_third = rng.random() < 0.45

    occupied: set[tuple[int, int]] = set()
    all_crosses: list[dict] = []
    all_markers: list[dict] = []

    row1 = rng.randint(6, h - 7)
    row2 = row1 + rng.choice((-1, 1))
    if not (6 <= row2 <= h - 7):
        return None
    sep = rng.randint(11, 14)
    col1_min = 6
    col1_max = w - sep - 7
    if col1_min > col1_max:
        return None
    col1 = rng.randint(col1_min, col1_max)
    col2 = col1 + sep

    pair_specs = [((row1, col1), {"right": BLUE}), ((row2, col2), {"left": BLUE})]
    for center, forced in pair_specs:
        arm_len = rng.randint(2, 3)
        branch_map = _sample_branch_map(rng, forced)
        template_choices = {direction: rng.choice(EAST_TEMPLATES) for direction in DIR_ORDER}
        if forced:
            for direction in forced:
                template_choices[direction] = rng.choice(OVERLAP_TEMPLATES)
        colors = rng.sample(BODY_COLORS, 4)
        body_colors = {direction: colors[i] for i, direction in enumerate(DIR_ORDER)}
        cross_cells, source_cells, markers = _make_cross_object(
            center, arm_len, branch_map, template_choices, body_colors, rng
        )
        if not _source_adjacency_ok(source_cells, occupied, shape):
            return None
        occupied.update(source_cells)
        all_crosses.append({"center": center, "coords": cross_cells})
        all_markers.extend(markers)

    if need_third:
        for _ in range(60):
            center = (rng.randint(6, h - 7), rng.randint(6, w - 7))
            arm_len = rng.randint(2, 3)
            branch_map = _sample_branch_map(rng)
            template_choices = {direction: rng.choice(EAST_TEMPLATES) for direction in DIR_ORDER}
            colors = rng.sample(BODY_COLORS, 4)
            body_colors = {direction: colors[i] for i, direction in enumerate(DIR_ORDER)}
            cross_cells, source_cells, markers = _make_cross_object(
                center, arm_len, branch_map, template_choices, body_colors, rng
            )
            if not _source_adjacency_ok(source_cells, occupied, shape):
                continue
            occupied.update(source_cells)
            all_crosses.append({"center": center, "coords": cross_cells})
            all_markers.extend(markers)
            break
        else:
            return None

    # Validate source count after optional third cross.
    if len(all_crosses) not in {2, 3}:
        return None

    source_forbidden = set(occupied)
    for r, c in list(occupied):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                source_forbidden.add((r + dr, c + dc))

    obstacles: set[tuple[int, int]] = set()
    maroon_markers = [m for m in all_markers if m["control_color"] == MAROON]
    maroon_paths: list[set[tuple[int, int]]] = []
    for marker in maroon_markers:
        choice = _choose_obstacle_for_maroon(marker, shape, source_forbidden | obstacles, rng)
        if choice is None:
            return None
        obstacle, stop_shift = choice
        obstacles.add(obstacle)
        source_forbidden.add(obstacle)
        maroon_paths.append(_path_cells_before(marker, stop_shift))

    # Add one ignored obstacle in a blue tiling lane whenever possible.
    blue_markers = [m for m in all_markers if m["control_color"] == BLUE]
    rng.shuffle(blue_markers)
    blue_lane_obstacle = None
    for marker in blue_markers:
        step = _axis_extent(marker["coords"], marker["direction"])
        dr, dc = marker["dir_vec"]
        for mult in (step, step * 2):
            moved = [(r + dr * mult, c + dc * mult) for r, c in marker["coords"]]
            if not all(_in_bounds(shape, rr, cc) for rr, cc in moved):
                continue
            candidate = moved[0]
            if candidate in source_forbidden or candidate in obstacles:
                continue
            obstacles.add(candidate)
            blue_lane_obstacle = candidate
            break
        if blue_lane_obstacle is not None:
            break

    # Scatter a few extra isolated obstacles outside source cells and maroon travel corridors.
    blocked_for_extra = set(source_forbidden) | obstacles
    for path in maroon_paths:
        blocked_for_extra.update(path)
    target_extra = rng.randint(2, 5)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    for r, c in cells:
        if len(obstacles) >= len(maroon_markers) + target_extra + (1 if blue_lane_obstacle is not None else 0):
            break
        if (r, c) in blocked_for_extra:
            continue
        if any((r + dr, c + dc) in occupied for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            continue
        if any((r + dr, c + dc) in obstacles for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            continue
        obstacles.add((r, c))

    grid = _paint_scene(shape, all_crosses, all_markers, obstacles)

    # Semantic validation from construction metadata.
    for cross in {m["center"] for m in all_markers}:
        colors = {m["control_color"] for m in all_markers if m["center"] == cross}
        if BLUE not in colors or MAROON not in colors:
            return None

    for marker in maroon_markers:
        if _max_gravity_shift(
            {
                "coords": marker["coords"],
                "dir_vec": marker["dir_vec"],
            },
            shape,
            obstacles,
        ) <= 0:
            return None

    expected = _metadata_output(shape, all_markers, grid)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None

    overlap, tie = _conflict_stats(all_markers, shape, grid)
    if overlap == 0 or tie:
        return None

    if blue_lane_obstacle is not None and int(actual[blue_lane_obstacle]) == ORANGE:
        return None

    parsed_crosses, parsed_markers = _parse_scene(grid)
    if len(parsed_crosses) != len(all_crosses) or len(parsed_markers) != len(all_markers):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid ARC instance")
