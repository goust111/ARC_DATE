from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
RayHit = tuple[int, Coord, Coord, tuple[Coord, ...]]

BG = 0
OBSTACLE = 1
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400
LOCAL_RETRIES = 200
SLOT_H = 12
SLOT_W = 12
SLOT_GAP = 2
OUTER_MARGIN = 1


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[set[Coord]]:
    g = np.asarray(grid, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
    cells = set(coords)
    comps: list[set[Coord]] = []
    while cells:
        seed = cells.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nbr in _neighbors4(*cur):
                if nbr in cells:
                    cells.remove(nbr)
                    comp.add(nbr)
                    stack.append(nbr)
        comps.append(comp)
    return comps


def _marker_positions(grid: np.ndarray, bg: int) -> dict[int, Coord]:
    g = np.asarray(grid, dtype=int)
    markers: dict[int, Coord] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (int(bg), OBSTACLE)):
        comps = _components_of_color(g, color)
        assert len(comps) == 1, "each marker color must form one component"
        comp = comps[0]
        assert len(comp) == 1, "markers must be single pixels"
        markers[color] = next(iter(comp))
    assert markers, "expected at least one marker"
    return markers


def _coord_to_component(components: list[set[Coord]]) -> dict[Coord, int]:
    lookup: dict[Coord, int] = {}
    for idx, comp in enumerate(components):
        for cell in comp:
            assert cell not in lookup, "components must be disjoint"
            lookup[cell] = idx
    return lookup


def _ray_hits(
    grid: np.ndarray,
    marker: Coord,
    bg: int,
    obstacle_lookup: dict[Coord, int],
) -> list[RayHit]:
    g = np.asarray(grid, dtype=int)
    hits: list[RayHit] = []
    sr, sc = marker
    for dr, dc in DIRS4:
        r = sr + dr
        c = sc + dc
        gap: list[Coord] = []
        while _in_bounds(g.shape, r, c):
            val = int(g[r, c])
            if val == bg:
                gap.append((r, c))
                r += dr
                c += dc
                continue
            if val == OBSTACLE and gap:
                hit = (r, c)
                hits.append((obstacle_lookup[hit], (dr, dc), hit, tuple(gap)))
            break
    return hits


def _adjacent_to_component(cell: Coord, component: set[Coord]) -> bool:
    return any(nbr in component for nbr in _neighbors4(*cell))


def _diagonal_directions(sightline: Coord) -> tuple[Coord, Coord]:
    dr, dc = sightline
    if dr != 0:
        return (dr, -1), (dr, 1)
    return (-1, dc), (1, dc)


def _walk_diagonal_path(
    grid: np.ndarray,
    start: Coord,
    step: Coord,
    component: set[Coord],
    bg: int,
) -> list[Coord]:
    g = np.asarray(grid, dtype=int)
    path: list[Coord] = []
    r, c = start
    dr, dc = step
    while True:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(g.shape, nr, nc):
            break
        if int(g[nr, nc]) != bg:
            break
        if not _adjacent_to_component((nr, nc), component):
            break
        path.append((nr, nc))
        r, c = nr, nc
    return path


def _analyze_instance(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    obstacle_components = _components_of_color(g, OBSTACLE)
    assert 1 <= len(obstacle_components) <= 3, "expected 1-3 obstacle components"
    obstacle_lookup = _coord_to_component(obstacle_components)
    markers = _marker_positions(g, bg)
    assert 1 <= len(markers) <= 3, "expected 1-3 markers"
    assert len(markers) == len(obstacle_components), "markers and obstacles must match one-to-one"

    analysis: list[dict] = []
    used_obstacles: set[int] = set()
    for color, marker in sorted(markers.items()):
        hits = _ray_hits(g, marker, bg, obstacle_lookup)
        assert len(hits) == 1, "each marker must have exactly one clear obstacle sightline"
        obstacle_idx, sightline, hit, gap = hits[0]
        assert obstacle_idx not in used_obstacles, "obstacle matching must be one-to-one"
        used_obstacles.add(obstacle_idx)
        start = gap[-1]
        component = obstacle_components[obstacle_idx]
        candidates = [
            _walk_diagonal_path(g, start, step, component, bg)
            for step in _diagonal_directions(sightline)
        ]
        lengths = [len(path) for path in candidates]
        best_len = max(lengths)
        assert best_len > 0, "at least one contour branch must be walkable"
        assert lengths.count(best_len) == 1, "one contour branch must be strictly longer"
        best_path = candidates[lengths.index(best_len)]
        analysis.append(
            {
                "marker_color": color,
                "marker_pos": marker,
                "obstacle_idx": obstacle_idx,
                "sightline": sightline,
                "touch": hit,
                "gap": tuple(gap),
                "gap_len": len(gap),
                "path": tuple(best_path),
                "path_len": len(best_path),
                "other_path_len": min(lengths),
            }
        )

    assert len(used_obstacles) == len(obstacle_components)
    return analysis


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each marker to the obstacle it sees, walk the longer contour branch, and repaint a tail from the far end.

    Input structure:
    - The background is black and blue pixels form 2-3 separate 4-connected obstacle regions.
    - Each non-blue non-background color is a single marker pixel outside the obstacles.
    - Every marker has exactly one horizontal or vertical background-only sightline to one obstacle, with a nonzero gap.

    Rule:
    - For each marker, follow its unique sightline to the nearest blue touchpoint and measure the number of background cells in that gap.
    - From the last background cell before the obstacle, try the two diagonal contour-following turns and keep the branch that stays 4-adjacent to that same obstacle for longer.
    - Recolor the last min(gap, branch_length) cells of that branch, counting backward from the far endpoint, with the marker color.
    - Keep the marker pixels, blue obstacles, and the sightline gap cells unchanged.

    Key insight:
    - The marker does not move onto the obstacle: its viewing distance encodes how much of the matched contour branch must be repainted from the remote end.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    planned: dict[Coord, int] = {}

    for info in _analyze_instance(g):
        path = list(info["path"])
        tail_len = min(int(info["gap_len"]), len(path))
        for cell in path[-tail_len:]:
            prev = planned.get(cell)
            color = int(info["marker_color"])
            assert prev is None or prev == color, "repaint tails must not conflict"
            planned[cell] = color

    for (r, c), color in planned.items():
        out[r, c] = int(color)
    return out


def _transform_grid(grid: np.ndarray, rotations: int, mirror: bool) -> np.ndarray:
    out = np.rot90(np.asarray(grid, dtype=int), int(rotations))
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _local_left_profile(height: int, start_row: int, start_left: int, upper_len: int, walk_len: int) -> list[int]:
    lefts = [start_left] * height
    for i in range(1, upper_len + 1):
        lefts[start_row - i] = start_left - i
    for r in range(start_row - upper_len - 1, -1, -1):
        lefts[r] = lefts[r + 1]
    for i in range(1, walk_len + 1):
        lefts[start_row + i] = start_left + i
    for r in range(start_row + walk_len + 1, height):
        lefts[r] = lefts[r - 1] + random.choice((0, 0, 1))
    return lefts


def _generate_local_patch(marker_color: int, relation: str) -> np.ndarray | None:
    for _ in range(LOCAL_RETRIES):
        if relation == "gap_shorter":
            walk_len = random.randint(2, 4)
            gap = random.randint(1, walk_len - 1)
        elif relation == "gap_longer":
            walk_len = random.randint(1, 3)
            gap = random.randint(walk_len + 1, min(5, walk_len + 3))
        elif relation == "gap_equal":
            walk_len = random.randint(1, 4)
            gap = walk_len
        else:
            walk_len = random.randint(1, 4)
            gap = random.randint(1, 5)

        width = random.randint(max(4, walk_len + 2), 6)
        max_upper = min(walk_len - 1, 2)
        upper_len = random.randint(0, max_upper)
        min_start_left = max(1, upper_len)
        max_start_left = min(gap, width - walk_len - 1)
        if max_start_left < min_start_left:
            continue
        start_left = random.randint(min_start_left, max_start_left)
        extra_top = random.randint(0, 1)
        extra_bottom = random.randint(0, 2)
        start_row = upper_len + extra_top
        height = start_row + walk_len + 1 + extra_bottom
        if not (4 <= height <= 8):
            continue

        lefts = _local_left_profile(height, start_row, start_left, upper_len, walk_len)
        if min(lefts) < 0 or max(lefts) >= width:
            continue
        area = sum(width - left for left in lefts)
        if not (6 <= area <= 15):
            continue
        if len(set(width - left for left in lefts)) == 1:
            continue

        c0 = gap - start_left + 1
        cols = c0 + width
        patch = np.zeros((height, cols), dtype=int)
        for r, left in enumerate(lefts):
            patch[r, c0 + left : c0 + width] = OBSTACLE
        patch[start_row, 0] = marker_color

        rotations = random.randrange(4)
        mirror = random.random() < 0.5
        patch = _transform_grid(patch, rotations, mirror)
        try:
            info = _analyze_instance(patch)
        except AssertionError:
            continue
        if len(info) != 1:
            continue
        one = info[0]
        if one["marker_color"] != marker_color:
            continue
        if relation == "gap_shorter" and not (one["gap_len"] < one["path_len"]):
            continue
        if relation == "gap_longer" and not (one["gap_len"] > one["path_len"]):
            continue
        if relation == "gap_equal" and not (one["gap_len"] == one["path_len"]):
            continue
        return patch
    return None


def _crop_with_margin(grid: np.ndarray, margin: int) -> np.ndarray:
    coords = np.argwhere(np.asarray(grid, dtype=int) != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = max(0, int(r0) - margin)
    c0 = max(0, int(c0) - margin)
    r1 = min(grid.shape[0] - 1, int(r1) + margin)
    c1 = min(grid.shape[1] - 1, int(c1) + margin)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _make_global_canvas(patches: list[np.ndarray]) -> np.ndarray | None:
    base_h = OUTER_MARGIN * 2 + SLOT_H * 2 + SLOT_GAP
    base_w = OUTER_MARGIN * 2 + SLOT_W * 2 + SLOT_GAP
    canvas = np.zeros((base_h, base_w), dtype=int)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    chosen_slots = random.sample(slots, len(patches))
    random.shuffle(chosen_slots)

    for patch, (sr, sc) in zip(patches, chosen_slots):
        ph, pw = patch.shape
        if ph > SLOT_H or pw > SLOT_W:
            return None
        top0 = OUTER_MARGIN + sr * (SLOT_H + SLOT_GAP)
        left0 = OUTER_MARGIN + sc * (SLOT_W + SLOT_GAP)
        dr = random.randint(0, SLOT_H - ph)
        dc = random.randint(0, SLOT_W - pw)
        region = canvas[top0 + dr : top0 + dr + ph, left0 + dc : left0 + dc + pw]
        if np.any((region != BG) & (patch != BG)):
            return None
        region[patch != BG] = patch[patch != BG]

    cropped = _crop_with_margin(canvas, margin=random.randint(1, 2))
    if max(cropped.shape) > 30:
        return None
    return cropped


def _validate_generated_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if int(np.count_nonzero(g == OBSTACLE)) == 0:
        return False

    try:
        info = _analyze_instance(g)
        out = main(g)
    except AssertionError:
        return False

    if len(info) not in (2, 3):
        return False
    if not np.any(out != g):
        return False
    tail_cells: set[Coord] = set()
    has_shorter = False
    has_longer = False
    for item in info:
        if len(item["path"]) != item["path_len"] or item["other_path_len"] >= item["path_len"]:
            return False
        if item["gap_len"] < item["path_len"]:
            has_shorter = True
        if item["gap_len"] > item["path_len"]:
            has_longer = True
        tail_len = min(item["gap_len"], item["path_len"])
        tail = set(item["path"][-tail_len:])
        if tail_cells & tail:
            return False
        tail_cells |= tail

    return has_shorter and has_longer


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair_count = random.choice((2, 3))
        relations = ["gap_shorter", "gap_longer"]
        if pair_count == 3:
            relations.append(random.choice(("gap_equal", "gap_shorter", "gap_longer", "any")))
        random.shuffle(relations)
        marker_colors = random.sample(MARKER_COLORS, pair_count)

        patches: list[np.ndarray] = []
        ok = True
        for color, relation in zip(marker_colors, relations):
            patch = _generate_local_patch(color, relation)
            if patch is None:
                ok = False
                break
            patches.append(patch)
        if not ok:
            continue

        grid = _make_global_canvas(patches)
        if grid is None:
            continue
        if _validate_generated_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid instance for puzzle 3")
