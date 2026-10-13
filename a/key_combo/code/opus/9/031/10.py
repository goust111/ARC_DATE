from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


AREA_TO_DIR = {
    3: (-1, 0),  # up
    4: (0, 1),   # right
    5: (1, 0),   # down
    6: (0, -1),  # left
}

GLYPH_TEMPLATES = {
    3: [
        np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=int),
    ],
    4: [
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    ],
    6: [
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=int),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, ignore: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignore:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _find_divider(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5)]
    if not cols:
        raise ValueError("no full-height grey divider found")
    return int(cols[0])


def _extract_glyphs(left: np.ndarray) -> list[tuple[int, int]]:
    glyphs: list[tuple[int, int, int]] = []
    for comp in _components_by_color(left, ignore={0, 5, 4}):
        color = int(comp["color"])
        area = int(comp["area"])
        if color not in {1, 2, 3} or area not in AREA_TO_DIR:
            continue
        r0, _c0, _r1, _c1 = comp["bbox"]
        glyphs.append((int(r0), color, area))
    glyphs.sort(key=lambda item: item[0])
    return [(color, area) for _, color, area in glyphs]


def _is_rect_wall(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return expected == set(coords)


def _extract_regions(right: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    regions: dict[int, tuple[int, int, int, int]] = {}
    for comp in _components_by_color(right, ignore={0, 4, 5}):
        color = int(comp["color"])
        if color not in {1, 2, 3}:
            continue
        coords = comp["coords"]
        if not _is_rect_wall(coords):
            continue
        if color in regions:
            raise ValueError("duplicate wall color region")
        regions[color] = tuple(map(int, comp["bbox"]))
    if not regions:
        raise ValueError("no colored rectangular regions found")
    return regions


def _slide_region_once(
    grid: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    *,
    occupancy_block: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    r0, c0, r1, c1 = region_bbox
    dr, dc = direction
    movers = {
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) == 4
    }
    new_pos = set()
    moved = 0
    blocked_wall = 0
    blocked_occ = 0
    for r, c in movers:
        nr, nc = r + dr, c + dc
        if not (r0 + 1 <= nr <= r1 - 1 and c0 + 1 <= nc <= c1 - 1):
            new_pos.add((r, c))
            blocked_wall += 1
            continue
        if occupancy_block and (nr, nc) in movers:
            new_pos.add((r, c))
            blocked_occ += 1
            continue
        new_pos.add((nr, nc))
        moved += 1

    out = grid.copy()
    for r, c in movers:
        out[r, c] = 0
    for r, c in new_pos:
        out[r, c] = 4
    return out, {
        "moved": moved,
        "blocked_wall": blocked_wall,
        "blocked_occ": blocked_occ,
        "count": len(movers),
    }


def _simulate(
    right: np.ndarray,
    regions: dict[int, tuple[int, int, int, int]],
    instructions: list[tuple[int, int]],
    *,
    color_sensitive: bool = True,
    occupancy_block: bool = True,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    state = right.copy()
    stats: list[dict[str, int]] = []
    for color, area in instructions:
        direction = AREA_TO_DIR[area]
        target_colors = [color] if color_sensitive else sorted(regions)
        total = {"moved": 0, "blocked_wall": 0, "blocked_occ": 0, "count": 0}
        for target in target_colors:
            state, cur = _slide_region_once(
                state,
                regions[target],
                direction,
                occupancy_block=occupancy_block,
            )
            for key in total:
                total[key] += cur[key]
        stats.append(total)
    return state, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode left-side glyph instructions and slide yellow pixels inside the matching colored closed regions.

    Input structure:
    - A black background contains a full-height grey divider splitting the grid into a left instruction half and a right region half.
    - On the left, 4-6 separated 3x3 glyph tiles contain single-color 4-connected shapes in blue/red/green; the glyph color identifies a target region and its pixel area is 3-6.
    - On the right, 2-3 colored rectangular wall loops use those same blue/red/green colors, and each enclosed interior contains yellow movable pixels on black.

    Rule:
    - Read the left glyphs from top to bottom.
    - For each glyph, find the right-side enclosed region whose wall color matches the glyph color, decode its area as a direction (3=up, 4=right, 5=down, 6=left), and move every yellow pixel in that region one cell simultaneously in that direction.
    - A yellow pixel stays put if that one-step destination is outside the region interior or occupied by another yellow pixel's current position; after all instructions, output only the right half with walls and final yellow pixels.

    Key insight:
    - The left glyph contributes two independent controls at once: color chooses which region is affected, while glyph area chooses the movement direction, so repeated same-color glyphs accumulate state changes in order.
    """
    g = np.asarray(grid, dtype=int)
    divider = _find_divider(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :].copy()
    glyphs = _extract_glyphs(left)
    regions = _extract_regions(right)
    out, _stats = _simulate(right, regions, glyphs)
    return out


def _random_template(area: int, rng: random.Random) -> np.ndarray:
    tile = GLYPH_TEMPLATES[area][rng.randrange(len(GLYPH_TEMPLATES[area]))].copy()
    for _ in range(rng.randrange(4)):
        tile = np.rot90(tile)
    if rng.random() < 0.5:
        tile = np.fliplr(tile)
    if rng.random() < 0.5:
        tile = np.flipud(tile)
    return tile


def _embed_glyphs(
    grid: np.ndarray,
    instructions: list[tuple[int, int]],
    *,
    left_w: int,
    rng: random.Random,
) -> None:
    needed_h = 4 * len(instructions) - 1
    top_margin = rng.randint(0, grid.shape[0] - needed_h)
    for i, (color, area) in enumerate(instructions):
        r0 = top_margin + 4 * i
        c0 = 1
        tile = _random_template(area, rng)
        grid[r0 : r0 + 3, c0 : c0 + 3][tile.astype(bool)] = color


def _place_regions(
    h: int,
    right_w: int,
    colors: list[int],
    rng: random.Random,
) -> dict[int, tuple[int, int, int, int]]:
    for _ in range(200):
        dims: list[tuple[int, int]] = []
        for _color in colors:
            inner_h = rng.randint(4, 4 if len(colors) == 3 else 6)
            inner_w = rng.randint(4, min(6, right_w - 2))
            dims.append((inner_h + 2, inner_w + 2))
        min_total = sum(oh for oh, _ow in dims) + (len(colors) - 1)
        if min_total > h:
            continue
        extra = h - min_total
        spaces = [0] + [1] * (len(colors) - 1) + [0]
        for _ in range(extra):
            spaces[rng.randrange(len(spaces))] += 1
        y = spaces[0]
        regions: dict[int, tuple[int, int, int, int]] = {}
        ok = True
        for idx, color in enumerate(colors):
            oh, ow = dims[idx]
            max_x = right_w - ow
            if max_x < 0:
                ok = False
                break
            x = rng.randint(0, max_x)
            regions[color] = (y, x, y + oh - 1, x + ow - 1)
            y += oh + spaces[idx + 1]
        if ok:
            return regions
    raise RuntimeError("failed to place regions")


def _safe_cells(region_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region_bbox
    return [
        (r, c)
        for r in range(r0 + 2, r1 - 1)
        for c in range(c0 + 2, c1 - 1)
    ]


def _initial_yellows(
    regions: dict[int, tuple[int, int, int, int]],
    repeat_color: int,
    rng: random.Random,
) -> dict[int, list[tuple[int, int]]]:
    yellow_map: dict[int, list[tuple[int, int]]] = {}
    for color, bbox in regions.items():
        cells = _safe_cells(bbox)
        if color == repeat_color:
            cells_set = set(cells)
            placed = None
            for r, c in cells:
                cand = [(r, c), (r, c + 1), (r + 1, c)]
                if all(pt in cells_set for pt in cand):
                    placed = cand
                    break
            if placed is None:
                raise RuntimeError("repeat region lacks special yellow pattern")
            yellow_map[color] = placed
        else:
            count = rng.randint(2, min(3, len(cells)))
            yellow_map[color] = rng.sample(cells, count)
    return yellow_map


def _render_right(
    h: int,
    right_w: int,
    regions: dict[int, tuple[int, int, int, int]],
    yellow_map: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    right = np.zeros((h, right_w), dtype=int)
    for color, (r0, c0, r1, c1) in regions.items():
        right[r0, c0 : c1 + 1] = color
        right[r1, c0 : c1 + 1] = color
        right[r0 : r1 + 1, c0] = color
        right[r0 : r1 + 1, c1] = color
        for r, c in yellow_map[color]:
            right[r, c] = 4
    return right


def _build_color_sequence(colors: list[int], repeat_color: int, n_instr: int, rng: random.Random) -> list[int]:
    others = [c for c in colors if c != repeat_color]
    seq = [repeat_color]
    if others:
        seq.append(rng.choice(others))
    seq.append(repeat_color)
    remaining = [c for c in others if c not in seq]
    seq.extend(remaining)
    while len(seq) < n_instr:
        seq.append(rng.choice(colors))
    if len(seq) > n_instr:
        seq = seq[:n_instr]
    tail = seq[3:]
    rng.shuffle(tail)
    return seq[:3] + tail


def _direction_candidates(
    state: np.ndarray,
    region_bbox: tuple[int, int, int, int],
) -> list[int]:
    out = []
    for area, direction in AREA_TO_DIR.items():
        _next, stats = _slide_region_once(state, region_bbox, direction)
        if stats["moved"] > 0:
            out.append(area)
    return out


def _construct_instructions(
    right: np.ndarray,
    regions: dict[int, tuple[int, int, int, int]],
    colors: list[int],
    repeat_color: int,
    n_instr: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    color_seq = _build_color_sequence(colors, repeat_color, n_instr, rng)
    state = right.copy()
    instructions: list[tuple[int, int]] = []
    repeat_seen = 0
    used_areas: set[int] = set()
    first_repeat_area = None

    for idx, color in enumerate(color_seq):
        bbox = regions[color]
        candidates = _direction_candidates(state, bbox)
        if not candidates:
            return None

        preferred = candidates[:]
        if color == repeat_color and repeat_seen == 0:
            preferred = []
            for area in candidates:
                _next, stats = _slide_region_once(state, bbox, AREA_TO_DIR[area])
                if stats["blocked_occ"] > 0 and 0 < stats["moved"] < stats["count"]:
                    preferred.append(area)
            if not preferred:
                return None
        elif color == repeat_color and repeat_seen == 1 and first_repeat_area is not None:
            first_dir = AREA_TO_DIR[first_repeat_area]
            preferred = [
                area
                for area in candidates
                if AREA_TO_DIR[area][0] != first_dir[0] or AREA_TO_DIR[area][1] != first_dir[1]
            ]
            # Prefer an orthogonal follow-up over repeating the same direction.
            orth = [
                area
                for area in preferred
                if AREA_TO_DIR[area][0] != 0 and first_dir[0] == 0
                or AREA_TO_DIR[area][1] != 0 and first_dir[1] == 0
            ]
            if orth:
                preferred = orth
            if not preferred:
                preferred = candidates[:]
        else:
            unseen = [area for area in candidates if area not in used_areas]
            if unseen and (idx < 4 or len(used_areas) < 3):
                preferred = unseen

        area = rng.choice(preferred)
        instructions.append((color, area))
        state, _stats = _slide_region_once(state, bbox, AREA_TO_DIR[area])
        used_areas.add(area)
        if color == repeat_color:
            if repeat_seen == 0:
                first_repeat_area = area
            repeat_seen += 1

    if len(used_areas) < 3:
        return None
    if sum(1 for color, _area in instructions if color == repeat_color) < 2:
        return None
    return instructions


def _validate_semantics(
    full_grid: np.ndarray,
    right: np.ndarray,
    regions: dict[int, tuple[int, int, int, int]],
    instructions: list[tuple[int, int]],
) -> bool:
    correct, step_stats = _simulate(right, regions, instructions)
    if any(stats["moved"] == 0 for stats in step_stats):
        return False
    if not any(0 < stats["moved"] < stats["count"] and stats["blocked_occ"] > 0 for stats in step_stats):
        return False

    reversed_out, _ = _simulate(right, regions, list(reversed(instructions)))
    if np.array_equal(correct, reversed_out):
        return False

    uncolored_out, _ = _simulate(right, regions, instructions, color_sensitive=False)
    if np.array_equal(correct, uncolored_out):
        return False

    no_occ_out, _ = _simulate(right, regions, instructions, occupancy_block=False)
    if np.array_equal(correct, no_occ_out):
        return False

    solver_out = main(full_grid)
    if not np.array_equal(solver_out, correct):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(2, 3)
    colors = rng.sample([1, 2, 3], n_regions)
    repeat_color = rng.choice(colors)
    n_instr = rng.randint(max(4, n_regions + 1), 6)

    h = max(20, 4 * n_instr - 1 + rng.randint(0, 3))
    left_w = 5
    right_w = rng.randint(9, 11)
    full_w = left_w + 1 + right_w

    try:
        regions = _place_regions(h, right_w, colors, rng)
        yellow_map = _initial_yellows(regions, repeat_color, rng)
    except RuntimeError:
        return None

    right = _render_right(h, right_w, regions, yellow_map)
    instructions = _construct_instructions(right, regions, colors, repeat_color, n_instr, rng)
    if instructions is None:
        return None

    full = np.zeros((h, full_w), dtype=int)
    full[:, left_w] = 5
    _embed_glyphs(full, instructions, left_w=left_w, rng=rng)
    full[:, left_w + 1 :] = right

    if not _validate_semantics(full, right, regions, instructions):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(800):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
