from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MARKER = 6
WALL_COLORS = (1, 2, 3)
DIR_ORDER = ("left", "right", "up", "down")
DIRS = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

GRID_RANGE = (15, 22)
RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _component_cells(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != color or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _is_wall_component(cells: list[tuple[int, int]]) -> bool:
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    if len(rows) == 1:
        span = list(range(cols[0], cols[-1] + 1))
        return cols == span and 2 <= len(cells) <= 4
    if len(cols) == 1:
        span = list(range(rows[0], rows[-1] + 1))
        return rows == span and 2 <= len(cells) <= 4
    return False


def _extract_templates_and_walls(grid: np.ndarray):
    templates: dict[int, np.ndarray] = {}
    template_cells: set[tuple[int, int]] = set()
    wall_cells: set[tuple[int, int]] = set()
    walls_by_color: dict[int, list[list[tuple[int, int]]]] = {}

    for color in WALL_COLORS:
        comps = _component_cells(grid, color)
        if not comps:
            continue
        template_comp = None
        walls: list[list[tuple[int, int]]] = []
        for comp in comps:
            if _is_wall_component(comp):
                walls.append(comp)
            else:
                template_comp = comp if template_comp is None else None
        assert template_comp is not None, f"missing unique template for color {color}"
        r0, c0, r1, c1 = _bbox(template_comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "template must occupy a 3x3 bbox"
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in template_comp:
            mask[r - r0, c - c0] = True
        templates[color] = mask
        template_cells.update(template_comp)
        wall_cells.update(cell for wall in walls for cell in wall)
        walls_by_color[color] = walls

    return templates, walls_by_color, template_cells, wall_cells


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == MARKER)
    assert len(pts) == 1, "expected a unique magenta marker"
    return tuple(map(int, pts[0]))


def _ray_events(grid: np.ndarray):
    marker = _find_marker(grid)
    events = []
    for direction in DIR_ORDER:
        dr, dc = DIRS[direction]
        r, c = marker
        steps = 0
        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc):
                break
            cell = int(grid[nr, nc])
            if cell not in (BG, MARKER):
                if steps > 0:
                    events.append(
                        {
                            "direction": direction,
                            "orientation": "horizontal" if direction in {"left", "right"} else "vertical",
                            "collision": (r, c),
                            "stop": (nr, nc),
                            "length": steps,
                            "color": cell,
                        }
                    )
                break
            r, c = nr, nc
            steps += 1
    return events


def _stamp_proposals(grid: np.ndarray):
    templates, _walls_by_color, _template_cells, _wall_cells = _extract_templates_and_walls(grid)
    proposals: dict[tuple[int, int], list[dict[str, object]]] = {}
    events = _ray_events(grid)
    H, W = grid.shape
    for idx, event in enumerate(events):
        mask = templates[int(event["color"])]
        base_r = int(event["collision"][0]) - 1
        base_c = int(event["collision"][1]) - 1
        orient_rank = 0 if event["orientation"] == "horizontal" else 1
        for rr, cc in np.argwhere(mask):
            tr = base_r + int(rr)
            tc = base_c + int(cc)
            if not _in_bounds((H, W), tr, tc):
                continue
            if int(grid[tr, tc]) != BG:
                continue
            proposals.setdefault((tr, tc), []).append(
                {
                    "priority": (int(event["length"]), orient_rank, idx),
                    "priority_rule": (int(event["length"]), orient_rank),
                    "color": int(event["color"]),
                    "event": event,
                }
            )
    return events, proposals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire four cardinal rays from the magenta marker and stamp the matched same-color motif at each wall collision point.

    Input structure:
    - A black background contains one magenta marker near the center.
    - Blue, red, and green may appear as short 1-cell-thick wall segments and as separate same-color 3x3 motif templates.
    - Each template is an irregular connected shape; walls and templates remain fixed in place.

    Rule:
    - From the magenta marker, trace rays up, down, left, and right through black cells until the next step would leave the grid or hit a non-black non-magenta cell.
    - If a ray stops because it hits a colored object after traversing at least one black cell, use the last traversed black cell as the collision point and the hit color to select that color's template.
    - Stamp the template with its 3x3 center aligned to the collision point, but only onto black cells; when multiple stamps compete for one black cell, the shorter ray wins, and equal lengths prefer horizontal rays over vertical rays.

    Key insight:
    - The ray determines where to stamp, the struck wall color determines which motif to stamp, and overlap resolution depends on comparing ray lengths rather than object order.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _templates, _walls_by_color, _template_cells, _wall_cells = _extract_templates_and_walls(g)
    _find_marker(g)
    _events, proposals = _stamp_proposals(g)
    for (r, c), entries in proposals.items():
        best = min(entries, key=lambda item: item["priority"])
        out[r, c] = int(best["color"])
    return out


def _build_mask_library() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if area < 3 or area > 5:
            continue
        if not mask[1, 1]:
            continue
        if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        if not _connected(mask):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        masks.append(mask)
    return masks


MASK_LIBRARY = _build_mask_library()


def _direction_subset() -> list[str]:
    if random.random() < 0.5:
        dirs = list(DIR_ORDER)
        random.shuffle(dirs)
        return dirs[:3]
    return list(DIR_ORDER)


def _event_distances(active_dirs: list[str], require_tie: bool) -> dict[str, int]:
    horiz = [d for d in active_dirs if d in {"left", "right"}]
    vert = [d for d in active_dirs if d in {"up", "down"}]
    pair_h = random.choice(horiz)
    pair_v = random.choice(vert)
    if require_tie:
        base = random.randint(1, 2)
        distances = {pair_h: base, pair_v: base}
    else:
        distances = {pair_h: random.randint(1, 2), pair_v: random.randint(1, 2)}
        if distances[pair_h] == distances[pair_v]:
            distances[pair_v] = 1 if distances[pair_h] == 2 else 2

    remaining = [d for d in active_dirs if d not in distances]
    pool = [1, 2, 3, 4]
    random.shuffle(pool)
    for direction in remaining:
        value = pool.pop() if pool else random.randint(1, 4)
        while any(direction in {"left", "right"} and other in {"left", "right"} and value == distances[other] for other in distances):
            value = random.randint(1, 4)
        while any(direction in {"up", "down"} and other in {"up", "down"} and value == distances[other] for other in distances):
            value = random.randint(1, 4)
        distances[direction] = value
    return distances


def _segment_candidates(direction: str, obstacle: tuple[int, int], shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    H, W = shape
    r, c = obstacle
    candidates: list[list[tuple[int, int]]] = []
    for length in range(2, 5):
        if direction in {"left", "right"}:
            for top in range(max(0, r - length + 1), min(r, H - length) + 1):
                candidates.append([(top + k, c) for k in range(length)])
            start = c - length + 1 if direction == "left" else c
            end = c if direction == "left" else c + length - 1
            if 0 <= start <= end < W:
                candidates.append([(r, cc) for cc in range(start, end + 1)])
        else:
            for left in range(max(0, c - length + 1), min(c, W - length) + 1):
                candidates.append([(r, left + k) for k in range(length)])
            start = r - length + 1 if direction == "up" else r
            end = r if direction == "up" else r + length - 1
            if 0 <= start <= end < H:
                candidates.append([(rr, c) for rr in range(start, end + 1)])
    random.shuffle(candidates)
    return candidates


def _separated(new_cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in new_cells:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_walls(
    grid: np.ndarray,
    marker: tuple[int, int],
    active_dirs: list[str],
    distances: dict[str, int],
    color_by_dir: dict[str, int],
) -> bool:
    occupied = {marker}
    for direction in active_dirs:
        dr, dc = DIRS[direction]
        obstacle = (marker[0] + dr * (distances[direction] + 1), marker[1] + dc * (distances[direction] + 1))
        placed = False
        for cells in _segment_candidates(direction, obstacle, grid.shape):
            if obstacle not in cells:
                continue
            if not _separated(cells, occupied):
                continue
            if any(not _in_bounds(grid.shape, r, c) for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = color_by_dir[direction]
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _corner_origins(H: int, W: int) -> list[tuple[int, int]]:
    return [(1, 1), (1, W - 4), (H - 4, 1), (H - 4, W - 4)]


def _place_templates(
    grid: np.ndarray,
    marker: tuple[int, int],
    used_colors: list[int],
    masks: dict[int, np.ndarray],
) -> bool:
    occupied = {tuple(map(int, p)) for p in np.argwhere(grid != BG)}
    corners = _corner_origins(*grid.shape)
    random.shuffle(corners)
    if len(used_colors) > len(corners):
        return False
    for color, (r0, c0) in zip(used_colors, corners):
        mask = masks[color]
        cells = [(r0 + int(rr), c0 + int(cc)) for rr, cc in np.argwhere(mask)]
        if marker[0] in range(r0, r0 + 3) or marker[1] in range(c0, c0 + 3):
            return False
        if not _separated(cells, occupied):
            return False
        for r, c in cells:
            grid[r, c] = color
        occupied.update(cells)
    return True


def _independent_scene_check(grid: np.ndarray, require_tie: bool) -> bool:
    try:
        marker = _find_marker(grid)
        templates, walls_by_color, template_cells, wall_cells = _extract_templates_and_walls(grid)
    except AssertionError:
        return False

    if not all(color in templates for color in templates):
        return False
    if any(not walls for walls in walls_by_color.values()):
        return False

    for color in WALL_COLORS:
        for comp in _component_cells(grid, color):
            if not _is_wall_component(comp):
                for cell in comp:
                    for nr, nc in _neighbors4(*cell):
                        if (nr, nc) != marker and _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BG, color):
                            return False

    events, proposals = _stamp_proposals(grid)
    if len(events) < 3:
        return False

    for event in events:
        if tuple(map(int, event["stop"])) in template_cells:
            return False
        if tuple(map(int, event["stop"])) not in wall_cells:
            return False

    overlap_pixels = [entries for entries in proposals.values() if len(entries) >= 2]
    if not overlap_pixels:
        return False

    tie_overlap = False
    for entries in overlap_pixels:
        seen_rule: dict[tuple[int, int], int] = {}
        for entry in entries:
            rule_key = tuple(entry["priority_rule"])
            color = int(entry["color"])
            if rule_key in seen_rule and seen_rule[rule_key] != color:
                return False
            seen_rule[rule_key] = color
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a = entries[i]["event"]
                b = entries[j]["event"]
                if int(a["length"]) == int(b["length"]) and a["orientation"] != b["orientation"]:
                    tie_overlap = True

    if require_tie and not tie_overlap:
        return False

    if np.array_equal(main(grid), grid):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    H = random.randint(*GRID_RANGE)
    W = random.randint(*GRID_RANGE)
    grid = np.zeros((H, W), dtype=int)
    marker = (
        random.randint(max(4, H // 2 - 1), min(H - 5, H // 2 + 1)),
        random.randint(max(4, W // 2 - 1), min(W - 5, W // 2 + 1)),
    )
    grid[marker] = MARKER

    active_dirs = _direction_subset()
    require_tie = random.random() < 0.45
    distances = _event_distances(active_dirs, require_tie)

    used_colors = list(WALL_COLORS)
    random.shuffle(used_colors)
    color_by_dir = {direction: used_colors[i] for i, direction in enumerate(active_dirs[:3])}
    if len(active_dirs) == 4:
        color_by_dir[active_dirs[3]] = random.choice(used_colors)

    if not _place_walls(grid, marker, active_dirs, distances, color_by_dir):
        return None

    mask_colors = sorted(set(color_by_dir.values()))
    masks = {color: random.choice(MASK_LIBRARY).copy() for color in mask_colors}
    if not _place_templates(grid, marker, mask_colors, masks):
        return None

    if not _independent_scene_check(grid, require_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-stamp puzzle input")

