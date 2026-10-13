import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _legend_components(grid: np.ndarray, bg: int) -> list[dict]:
    mask = (grid != bg) & (grid != GREY)
    comps = []
    for coords in _components(mask):
        rows = {r for r, _ in coords}
        cols = sorted(c for _, c in coords)
        assert len(rows) == 1, "legend entries must be horizontal segments"
        row = next(iter(rows))
        c0, c1 = cols[0], cols[-1]
        assert len(coords) == c1 - c0 + 1, "legend segments must be contiguous"
        color = int(grid[coords[0]])
        comps.append(
            {
                "coords": coords,
                "row": int(row),
                "c0": int(c0),
                "c1": int(c1),
                "length": int(len(coords)),
                "color": color,
            }
        )
    comps.sort(key=lambda item: (item["row"], item["c0"]))
    return comps


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in _legend_components(grid, bg):
        length = int(item["length"])
        assert length not in mapping, "legend lengths must be unique"
        mapping[length] = int(item["color"])
    assert mapping, "missing legend"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey sprite by matching its pixel area to the length of a legend bar.

    Input structure:
    - The grid has black background and several grey 4-connected sprite bodies.
    - The only non-grey foreground pixels form a small top-left legend of separate horizontal color bars.
    - Each legend bar has one distinct color and one distinct pixel length.

    Rule:
    - Read every non-grey legend bar as a mapping: bar length -> bar color.
    - For every grey 4-connected component, count its total number of pixels.
    - Find the legend bar whose length equals that area, and recolor the whole component to that legend color.
    - Keep the legend unchanged and preserve every sprite's position and shape.

    Key insight:
    - The legend's geometry carries the code: the bar length is the lookup key, and each grey sprite's area is the query value.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping = _legend_mapping(g, bg)

    out = g.copy()
    for coords in _components(g == GREY):
        area = len(coords)
        assert area in mapping, "every grey sprite area must be listed in the legend"
        color = int(mapping[area])
        for r, c in coords:
            out[r, c] = color
    return out


def _random_lengths(count: int, max_len: int) -> list[int]:
    lengths = list(range(3, max_len + 1))
    assert len(lengths) >= count
    while True:
        picked = sorted(random.sample(lengths, count))
        if any(length >= 4 for length in picked):
            return picked


def _grow_polyomino(area: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                frontier.add((r + dr, c + dc))
        frontier -= cells
        choices = []
        for cell in frontier:
            score = 0
            for dr, dc in DIRS4:
                if (cell[0] + dr, cell[1] + dc) in cells:
                    score += 1
            choices.extend([cell] * max(1, score))
        cells.add(random.choice(choices))
    coords = list(cells)
    r0, c0, _, _ = _bbox(coords)
    return [(r - r0, c - c0) for r, c in coords]


def _make_shape(area: int, forbidden: set[tuple[tuple[int, int], ...]] | None = None) -> list[tuple[int, int]]:
    forbidden = forbidden or set()
    for _ in range(200):
        coords = _grow_polyomino(area)
        r0, c0, r1, c1 = _bbox(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        norm = _normalize_shape(coords)
        if hh < 2 or ww < 2:
            continue
        if hh > 4 or ww > 4:
            continue
        if len(coords) == hh * ww and min(hh, ww) == 2 and max(hh, ww) > 3:
            continue
        if norm in forbidden:
            continue
        return list(norm)
    raise RuntimeError(f"failed to build shape with area {area}")


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr, dc in DIRS8 + ((0, 0),):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: list[tuple[int, int]],
    color: int,
    legend_bottom: int,
    legend_right: int,
    force_zone: str | None,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    _, _, sh1, sw1 = _bbox(shape)
    sh = sh1 + 1
    sw = sw1 + 1
    for _ in range(250):
        r0 = random.randint(0, h - sh)
        c0 = random.randint(0, w - sw)
        coords = [(r0 + r, c0 + c) for r, c in shape]
        if force_zone == "below" and r0 <= legend_bottom + 1:
            continue
        if force_zone == "right" and c0 <= legend_right + 1:
            continue
        if force_zone is None and not (r0 > legend_bottom + 1 or c0 > legend_right + 1):
            continue
        if any(blocked[r, c] for r, c in coords):
            continue
        for r, c in coords:
            grid[r, c] = int(color)
        _mark_blocked(blocked, coords)
        return coords
    return None


def _sprite_areas(legend_lengths: list[int], sprite_count: int) -> list[int]:
    repeated_pool = [length for length in legend_lengths if length >= 4]
    repeat_len = random.choice(repeated_pool)
    areas = list(legend_lengths)
    while len(areas) < sprite_count:
        areas.append(repeat_len if len(areas) == len(legend_lengths) else random.choice(legend_lengths))
    random.shuffle(areas)
    return areas


def _legend_rows(start: int, count: int) -> list[int]:
    rows = []
    cur = start
    for i in range(count):
        rows.append(cur)
        cur += 1 + random.randint(1, 2 if i < count - 1 else 1)
    return rows


def _build_attempt() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(15, 19)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend_count = random.randint(3, 4)
    legend_c0 = random.randint(0, 1)
    legend_r0 = random.randint(0, 1)
    max_len = min(8, w - legend_c0 - 4)
    if max_len < 5:
        return None
    legend_lengths = _random_lengths(legend_count, max_len)
    legend_colors = random.sample(PALETTE, legend_count)
    rows = _legend_rows(legend_r0, legend_count)
    if rows[-1] >= h - 5:
        return None

    for row, length, color in zip(rows, legend_lengths, legend_colors):
        grid[row, legend_c0 : legend_c0 + length] = int(color)
        coords = [(row, legend_c0 + dc) for dc in range(length)]
        _mark_blocked(blocked, coords)

    legend_bottom = max(rows)
    legend_right = legend_c0 + max(legend_lengths) - 1

    min_sprites = max(4, legend_count + 1)
    sprite_count = random.randint(min_sprites, 6)
    areas = _sprite_areas(legend_lengths, sprite_count)
    shapes: list[list[tuple[int, int]]] = []
    used_shapes_by_area: dict[int, set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    repeated_areas = {area for area in areas if areas.count(area) >= 2 and area >= 4}
    for idx, area in enumerate(areas):
        forbid = set()
        if area in repeated_areas and len(used_shapes_by_area[area]) == 1:
            forbid = set(used_shapes_by_area[area])
        shape = _make_shape(area, forbidden=forbid)
        used_shapes_by_area[area].add(_normalize_shape(shape))
        shapes.append(shape)

    zones = ["below", "right"] + [None] * (sprite_count - 2)
    random.shuffle(zones)
    placed: list[list[tuple[int, int]]] = []
    for shape, area, zone in zip(shapes, areas, zones):
        coords = _place_shape(grid, blocked, shape, GREY, legend_bottom, legend_right, zone)
        if coords is None:
            return None
        placed.append(coords)

    if not _validate_instance(grid):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = _get_bg(grid)
    if bg != BG:
        return False

    legend = _legend_components(grid, bg)
    if not (3 <= len(legend) <= 4):
        return False

    c0s = {item["c0"] for item in legend}
    colors = {item["color"] for item in legend}
    lengths = [item["length"] for item in legend]
    if len(c0s) != 1 or len(colors) != len(legend) or len(set(lengths)) != len(legend):
        return False
    if max(item["row"] for item in legend) >= grid.shape[0] // 2 + 2:
        return False

    grey_comps = _components(grid == GREY)
    if not (4 <= len(grey_comps) <= 6):
        return False

    mapping = {item["length"]: item["color"] for item in legend}
    used = Counter()
    norms_by_area: dict[int, set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    for comp in grey_comps:
        area = len(comp)
        if area not in mapping:
            return False
        used[area] += 1
        norms_by_area[area].add(_normalize_shape(comp))

    if any(used[length] == 0 for length in mapping):
        return False
    if not any(count >= 2 for count in used.values()):
        return False
    if not any(len(norms_by_area[area]) >= 2 for area, count in used.items() if count >= 2):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.any(out == GREY):
        return False
    if not np.array_equal(out[(grid != GREY)], grid[(grid != GREY)]):
        return False

    for comp in grey_comps:
        area = len(comp)
        target = mapping[area]
        if any(int(out[r, c]) != int(target) for r, c in comp):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-legend recoloring task")
