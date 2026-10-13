from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
WALL = 5
PALETTE_POOL = (1, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
ROOM_MIN = 4
LEAF_TARGET_RANGE = (4, 7)
INNER_H_RANGE = (12, 20)
INNER_W_RANGE = (12, 20)

ISLAND_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_palette(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    def parse_bottom() -> tuple[list[int], np.ndarray] | None:
        row = g[-1]
        colors: list[int] = []
        mask = np.zeros_like(g, dtype=bool)
        c = 0
        while c < w:
            if int(row[c]) == BG:
                c += 1
                continue
            color = int(row[c])
            if color in {BG, RED, WALL}:
                return None
            if c + 1 >= w or int(row[c + 1]) != color:
                return None
            if c + 2 < w and int(row[c + 2]) == color:
                return None
            colors.append(color)
            mask[-1, c : c + 2] = True
            c += 2
        if len(colors) >= 2 and len(colors) == len(set(colors)):
            return colors, mask
        return None

    def parse_right() -> tuple[list[int], np.ndarray] | None:
        col = g[:, -1]
        colors: list[int] = []
        mask = np.zeros_like(g, dtype=bool)
        r = 0
        while r < h:
            if int(col[r]) == BG:
                r += 1
                continue
            color = int(col[r])
            if color in {BG, RED, WALL}:
                return None
            if r + 1 >= h or int(col[r + 1]) != color:
                return None
            if r + 2 < h and int(col[r + 2]) == color:
                return None
            colors.append(color)
            mask[r : r + 2, -1] = True
            r += 2
        if len(colors) >= 2 and len(colors) == len(set(colors)):
            return colors, mask
        return None

    candidates = []
    bottom = parse_bottom()
    if bottom is not None:
        candidates.append(bottom)
    right = parse_right()
    if right is not None:
        candidates.append(right)

    assert candidates, "expected a bottom-row or right-column palette"
    colors, mask = max(candidates, key=lambda item: len(item[0]))
    return colors, mask


def _enclosed_regions(grid: np.ndarray, palette_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    walkable = (grid != WALL) & ~palette_mask
    h, w = grid.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(walkable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _red_island_count(grid: np.ndarray, region: list[tuple[int, int]]) -> int:
    region_mask = np.zeros_like(grid, dtype=bool)
    for r, c in region:
        region_mask[r, c] = True
    return len(_components(region_mask & (grid == RED)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed chamber with the palette color indexed by its red-island count.

    Input structure:
    - The background is black.
    - A connected gray wall structure partitions the interior into several
      mutually separated enclosed regions.
    - Some enclosed regions contain one or more small red 4-connected island
      objects, while others contain none.
    - The bottom edge or right edge contains a palette strip made of distinct
      2x1 color blocks separated by black cells.

    Rule:
    - Read the palette left-to-right on the bottom edge or top-to-bottom on the
      right edge so that palette index 0 corresponds to zero islands, index 1
      to one island, and so on.
    - For every enclosed region bounded by gray walls, count the number of red
      4-connected islands inside it.
    - Recolor only the black background cells of that region to the palette
      color at the matching index.
    - Leave gray walls, red islands, and the palette itself unchanged.

    Key insight:
    - The task is a segmentation-then-counting lookup: first isolate each
      closed room, then count its red island components, then use that count as
      a palette index.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    palette, palette_mask = _parse_palette(g)
    out = g.copy()

    for region in _enclosed_regions(g, palette_mask):
        count = _red_island_count(g, region)
        assert count < len(palette), "palette too short for enclosed region island count"
        fill = int(palette[count])
        for r, c in region:
            if int(g[r, c]) == BG:
                out[r, c] = fill

    return out


def _room_orientations(room: tuple[int, int, int, int]) -> list[str]:
    r0, c0, r1, c1 = room
    opts: list[str] = []
    if r1 - r0 + 1 >= 2 * ROOM_MIN + 1:
        opts.append("h")
    if c1 - c0 + 1 >= 2 * ROOM_MIN + 1:
        opts.append("v")
    return opts


def _split_room(room: tuple[int, int, int, int], orient: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[str, int, int, int]]:
    r0, c0, r1, c1 = room
    if orient == "v":
        split_c = random.randint(c0 + ROOM_MIN, c1 - ROOM_MIN)
        left = (r0, c0, r1, split_c - 1)
        right = (r0, split_c + 1, r1, c1)
        wall = ("v", split_c, r0 - 1, r1 + 1)
        return left, right, wall

    split_r = random.randint(r0 + ROOM_MIN, r1 - ROOM_MIN)
    top = (r0, c0, split_r - 1, c1)
    bottom = (split_r + 1, c0, r1, c1)
    wall = ("h", split_r, c0 - 1, c1 + 1)
    return top, bottom, wall


def _draw_wall(chamber: np.ndarray, wall: tuple[str, int, int, int]) -> None:
    orient, fixed, start, end = wall
    if orient == "v":
        chamber[start : end + 1, fixed] = WALL
    else:
        chamber[fixed, start : end + 1] = WALL


def _make_chamber() -> tuple[np.ndarray, list[list[tuple[int, int]]]] | None:
    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)
    chamber = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    chamber[0, :] = WALL
    chamber[-1, :] = WALL
    chamber[:, 0] = WALL
    chamber[:, -1] = WALL

    rooms = [(1, 1, inner_h, inner_w)]
    walls: list[tuple[str, int, int, int]] = []
    target_leaves = random.randint(*LEAF_TARGET_RANGE)

    while len(rooms) < target_leaves:
        splittable = [(idx, _room_orientations(room)) for idx, room in enumerate(rooms)]
        splittable = [(idx, opts) for idx, opts in splittable if opts]
        if not splittable:
            break
        idx, opts = random.choice(splittable)
        room = rooms.pop(idx)
        r0, c0, r1, c1 = room
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if len(opts) == 2:
            orient = random.choices(["h", "v"], weights=[w, h])[0]
        else:
            orient = opts[0]
        room_a, room_b, wall = _split_room(room, orient)
        _draw_wall(chamber, wall)
        walls.append(wall)
        rooms.extend([room_a, room_b])

    if len(rooms) < 3:
        return None

    max_open = min(2, len(walls))
    open_goal = min(random.choices([0, 1, 2], weights=[1, 2, 1])[0], max_open)
    wall_order = walls[:]
    random.shuffle(wall_order)
    opened = 0
    for orient, fixed, start, end in wall_order:
        if opened >= open_goal:
            break
        interior_len = end - start - 1
        max_gap = min(3, interior_len - 2)
        if max_gap < 1:
            continue
        gap_len = random.randint(1, max_gap)
        gap_start = random.randint(start + 2, end - gap_len - 1)
        if orient == "v":
            chamber[gap_start : gap_start + gap_len, fixed] = BG
        else:
            chamber[fixed, gap_start : gap_start + gap_len] = BG
        opened += 1

    if len(_components(chamber == WALL)) != 1:
        return None

    regions = _components(chamber != WALL)
    if not (3 <= len(regions) <= 6):
        return None

    return chamber, regions


def _placement_block(cells: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set(cells)
    for r, c in cells:
        blocked.update(_neighbors4(r, c))
    return blocked


def _choose_islands(region: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []

    region_set = set(region)
    seen: set[frozenset[tuple[int, int]]] = set()
    candidates: list[frozenset[tuple[int, int]]] = []
    for r, c in region:
        for shape in ISLAND_SHAPES:
            cells = frozenset((r + dr, c + dc) for dr, dc in shape)
            if cells in seen:
                continue
            if all(cell in region_set for cell in cells):
                seen.add(cells)
                candidates.append(cells)

    if len(candidates) < count:
        return None

    for _ in range(8):
        random.shuffle(candidates)
        blocks = [_placement_block(cells) for cells in candidates]

        def dfs(start: int, chosen: list[int], blocked: set[tuple[int, int]]) -> list[int] | None:
            if len(chosen) == count:
                return chosen
            for idx in range(start, len(candidates)):
                if candidates[idx] & blocked:
                    continue
                result = dfs(idx + 1, chosen + [idx], blocked | blocks[idx])
                if result is not None:
                    return result
            return None

        chosen = dfs(0, [], set())
        if chosen is None:
            continue
        out: list[tuple[int, int]] = []
        for idx in chosen:
            out.extend(sorted(candidates[idx]))
        return out

    return None


def _assign_islands(chamber: np.ndarray, regions: list[list[tuple[int, int]]]) -> tuple[np.ndarray, list[int], list[int]] | None:
    region_order = sorted(range(len(regions)), key=lambda idx: len(regions[idx]), reverse=True)
    max_k = min(4, len(regions) - 1)
    ks = list(range(2, max_k + 1))
    random.shuffle(ks)

    for k in ks:
        counts_desc = list(range(k + 1))
        counts_desc.extend(random.randint(0, k) for _ in range(len(regions) - len(counts_desc)))
        counts_desc.sort(reverse=True)

        if any(len(regions[idx]) < max(1, 3 * count - 1) for idx, count in zip(region_order, counts_desc)):
            continue

        count_by_region = [0] * len(regions)
        placements: dict[int, list[tuple[int, int]]] = {}
        ok = True
        for idx, count in zip(region_order, counts_desc):
            cells = _choose_islands(regions[idx], count)
            if cells is None:
                ok = False
                break
            count_by_region[idx] = count
            placements[idx] = cells
        if not ok:
            continue

        chamber_with_islands = chamber.copy()
        for cells in placements.values():
            for r, c in cells:
                chamber_with_islands[r, c] = RED
        return chamber_with_islands, count_by_region, list(range(k + 1))

    return None


def _palette_layout(colors: list[int], orient: str, shape: tuple[int, int]) -> list[tuple[slice, slice, int]] | None:
    h, w = shape
    gaps = [random.randint(1, 2) for _ in range(len(colors) - 1)]
    total = 2 * len(colors) + sum(gaps)
    spans: list[tuple[slice, slice, int]] = []

    if orient == "bottom":
        if total > w:
            return None
        start = random.randint(0, w - total)
        c = start
        for idx, color in enumerate(colors):
            spans.append((slice(h - 1, h), slice(c, c + 2), color))
            c += 2
            if idx + 1 < len(colors):
                c += gaps[idx]
        return spans

    if total > h:
        return None
    start = random.randint(0, h - total)
    r = start
    for idx, color in enumerate(colors):
        spans.append((slice(r, r + 2), slice(w - 1, w), color))
        r += 2
        if idx + 1 < len(colors):
            r += gaps[idx]
    return spans


def _paint_palette(grid: np.ndarray, spans: list[tuple[slice, slice, int]]) -> None:
    for rs, cs, color in spans:
        grid[rs, cs] = color


def _embed_chamber(chamber: np.ndarray, solved_chamber: np.ndarray, palette_colors: list[int], orient: str) -> tuple[np.ndarray, np.ndarray] | None:
    top = random.randint(1, 2)
    left = random.randint(1, 2)
    if orient == "bottom":
        right_margin = random.randint(1, 2)
        h = top + chamber.shape[0] + 2
        w = left + chamber.shape[1] + right_margin
    else:
        bottom_margin = random.randint(1, 2)
        h = top + chamber.shape[0] + bottom_margin
        w = left + chamber.shape[1] + 2

    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    solved = np.zeros((h, w), dtype=int)
    rs = slice(top, top + chamber.shape[0])
    cs = slice(left, left + chamber.shape[1])
    grid[rs, cs] = chamber
    solved[rs, cs] = solved_chamber
    spans = _palette_layout(palette_colors, orient, grid.shape)
    if spans is None:
        return None
    _paint_palette(grid, spans)
    _paint_palette(solved, spans)
    return grid, solved


def _gen_attempt() -> np.ndarray | None:
    chamber_data = _make_chamber()
    if chamber_data is None:
        return None
    chamber, regions = chamber_data

    island_data = _assign_islands(chamber, regions)
    if island_data is None:
        return None
    chamber_with_islands, count_by_region, palette_indices = island_data

    palette_colors = random.sample(PALETTE_POOL, len(palette_indices))
    solved_chamber = chamber_with_islands.copy()
    for region_idx, region in enumerate(regions):
        fill = palette_colors[count_by_region[region_idx]]
        for r, c in region:
            if int(chamber_with_islands[r, c]) == BG:
                solved_chamber[r, c] = fill

    orient = random.choice(["bottom", "right"])
    embedded = _embed_chamber(chamber_with_islands, solved_chamber, palette_colors, orient)
    if embedded is None:
        return None
    inp, expected = embedded

    try:
        palette, palette_mask = _parse_palette(inp)
    except AssertionError:
        return None
    if palette != palette_colors:
        return None
    if len(_components(inp == WALL)) != 1:
        return None
    regions_full = _enclosed_regions(inp, palette_mask)
    if len(regions_full) != len(regions):
        return None
    counts_found = sorted(_red_island_count(inp, region) for region in regions_full)
    if counts_found != sorted(count_by_region):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the new ARC task")
