import random
from collections import Counter, deque

import numpy as np


BG = 0
TEMPLATE = 5
WALL = 8
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 9]

MIN_ZONE_H = 7
MIN_ZONE_W = 7
MAX_ATTEMPTS = 400

SWATCH_SHAPES = [
    np.ones((2, 2), dtype=bool),
    np.ones((2, 3), dtype=bool),
    np.ones((3, 2), dtype=bool),
    np.ones((1, 3), dtype=bool),
    np.ones((3, 1), dtype=bool),
]

TEMPLATE_SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    cur = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cur += 1
            labels[r, c] = cur
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cur
                        q.append((nr, nc))
    return labels, cur


def _component_palette_info(
    grid: np.ndarray,
    wall_color: int,
    bg_color: int,
    template_color: int,
) -> tuple[np.ndarray, int, list[tuple[np.ndarray, set[int], bool]]]:
    labels, count = _label_components(grid != wall_color)
    info = []
    for idx in range(1, count + 1):
        mask = labels == idx
        colors = set(map(int, np.unique(grid[mask])))
        palettes = colors - {bg_color, template_color, wall_color}
        has_template = bool(np.any(mask & (grid == template_color)))
        info.append((mask, palettes, has_template))
    return labels, count, info


def _infer_wall_color(grid: np.ndarray, bg_color: int, template_color: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (bg_color, template_color)]
    assert colors, "expected at least one non-background non-template color"

    ranked: list[tuple[int, int, int]] = []
    template_cells = int(np.count_nonzero(grid == template_color))
    for wall_color in colors:
        _labels, _count, info = _component_palette_info(grid, wall_color, bg_color, template_color)
        good = True
        template_covered = 0
        palette_components = 0
        distinct_palette_colors = set()
        for mask, palettes, has_template in info:
            if palettes:
                if len(palettes) != 1 or not has_template:
                    good = False
                    break
                palette_components += 1
                distinct_palette_colors |= palettes
                template_covered += int(np.count_nonzero(mask & (grid == template_color)))
            elif has_template:
                good = False
                break
        if not good or palette_components == 0 or template_covered != template_cells:
            continue
        ranked.append(
            (
                palette_components,
                len(distinct_palette_colors),
                int(np.count_nonzero(grid == wall_color)),
            )
        )

    assert ranked, "failed to infer wall color"
    best_score = max(ranked)
    for wall_color in colors:
        score = None
        _labels, _count, info = _component_palette_info(grid, wall_color, bg_color, template_color)
        good = True
        template_covered = 0
        palette_components = 0
        distinct_palette_colors = set()
        for mask, palettes, has_template in info:
            if palettes:
                if len(palettes) != 1 or not has_template:
                    good = False
                    break
                palette_components += 1
                distinct_palette_colors |= palettes
                template_covered += int(np.count_nonzero(mask & (grid == template_color)))
            elif has_template:
                good = False
                break
        if good and palette_components > 0 and template_covered == template_cells:
            score = (
                palette_components,
                len(distinct_palette_colors),
                int(np.count_nonzero(grid == wall_color)),
            )
        if score == best_score:
            return wall_color
    raise AssertionError("no wall color matched best score")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey template by the palette swatch reachable through its corridor component.

    Input structure:
    - The background is the most frequent color, and one wall color forms thin orthogonal barriers.
    - Grey template shapes and small solid palette swatches occupy the passable cells between those walls.
    - Each wall-blocked 4-connected passable component contains exactly one palette swatch and one or more grey templates.

    Rule:
    - Treat the wall color as blocked and partition every non-wall cell into 4-connected components.
    - In each component, read its unique non-background, non-grey palette color.
    - Recolor every grey template cell in that component to the palette color, erase the palette swatch back to background, and leave the walls unchanged.

    Key insight:
    - The correct color is determined by reachability through the wall-and-corridor topology, not by straight-line proximity to the nearest swatch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg_color = _get_bg(g)
    assert int(np.count_nonzero(g == TEMPLATE)) > 0, "expected grey template cells"
    wall_color = _infer_wall_color(g, bg_color, TEMPLATE)

    _labels, _count, info = _component_palette_info(g, wall_color, bg_color, TEMPLATE)
    out = g.copy()
    seen = 0
    for mask, palettes, has_template in info:
        if not palettes:
            assert not has_template
            continue
        assert len(palettes) == 1 and has_template
        palette = next(iter(palettes))
        out[mask & (g == TEMPLATE)] = palette
        out[mask & (g == palette)] = bg_color
        seen += 1

    assert seen > 0
    return out


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _random_template_shape() -> np.ndarray:
    shape = random.choice(TEMPLATE_SHAPES).copy()
    if random.random() < 0.5:
        shape = np.fliplr(shape)
    shape = np.rot90(shape, random.randrange(4))
    return _trim(shape)


def _opposite_room(room: str | None) -> str | None:
    return {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }.get(room)


def _room_pref_from_edge(divider: dict | None, edge: str, mode: str) -> str | None:
    if divider is None:
        return None
    if divider["orientation"] == "v" and edge in ("west", "east"):
        near = "left" if edge == "west" else "right"
    elif divider["orientation"] == "h" and edge in ("north", "south"):
        near = "top" if edge == "north" else "bottom"
    else:
        return None
    return near if mode == "near" else _opposite_room(near)


def _shape_room(origin: tuple[int, int], shape: np.ndarray, divider: dict | None) -> str | None:
    if divider is None:
        return None
    r, c = origin
    h, w = shape.shape
    if divider["orientation"] == "v":
        return "left" if c + w - 1 < divider["line"] else "right"
    return "top" if r + h - 1 < divider["line"] else "bottom"


def _edge_distance(origin: tuple[int, int], shape: np.ndarray, zone: tuple[int, int, int, int], edge: str) -> float:
    r, c = origin
    h, w = shape.shape
    r0, c0, r1, c1 = zone
    if edge == "west":
        return float(c - c0)
    if edge == "east":
        return float(c1 - (c + w - 1))
    if edge == "north":
        return float(r - r0)
    return float(r1 - (r + h - 1))


def _align_distance(origin: tuple[int, int], shape: np.ndarray, align: tuple[str, float] | None) -> float:
    if align is None:
        return 0.0
    axis, target = align
    r, c = origin
    h, w = shape.shape
    center_r = r + (h - 1) / 2.0
    center_c = c + (w - 1) / 2.0
    return abs((center_r if axis == "row" else center_c) - target)


def _candidate_origins(
    grid: np.ndarray,
    occupied: np.ndarray,
    zone: tuple[int, int, int, int],
    shape: np.ndarray,
    divider: dict | None,
    room_pref: str | None,
) -> list[tuple[int, int]]:
    h, w = shape.shape
    r0, c0, r1, c1 = zone
    out = []
    for r in range(r0, r1 - h + 2):
        for c in range(c0, c1 - w + 2):
            patch = grid[r : r + h, c : c + w]
            if patch.shape != shape.shape or not np.all(patch == BG):
                continue
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(grid.shape[0], r + h + 1)
            cc1 = min(grid.shape[1], c + w + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            if room_pref is not None and _shape_room((r, c), shape, divider) != room_pref:
                continue
            out.append((r, c))
    return out


def _choose_origin(
    candidates: list[tuple[int, int]],
    shape: np.ndarray,
    zone: tuple[int, int, int, int],
    edge_pref: tuple[str, str] | None,
    align: tuple[str, float] | None,
) -> tuple[int, int] | None:
    if not candidates:
        return None
    scored = []
    for cand in candidates:
        parts: list[float] = []
        if edge_pref is not None:
            edge, mode = edge_pref
            dist = _edge_distance(cand, shape, zone, edge)
            parts.append(dist if mode == "near" else -dist)
        if align is not None:
            parts.append(_align_distance(cand, shape, align))
        scored.append((tuple(parts), cand))
    scored.sort(key=lambda item: item[0])
    band = [cand for _score, cand in scored[: min(5, len(scored))]]
    return random.choice(band)


def _paint_shape(grid: np.ndarray, occupied: np.ndarray, shape: np.ndarray, origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    for dr, dc in np.argwhere(shape):
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        grid[rr, cc] = color
        occupied[rr, cc] = True


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    zone_info: dict,
    shape: np.ndarray,
    color: int,
    edge_pref: tuple[str, str] | None = None,
    align: tuple[str, float] | None = None,
    room_pref: str | None = None,
) -> tuple[int, int] | None:
    zone = zone_info["bbox"]
    divider = zone_info["divider"]
    candidates = _candidate_origins(grid, occupied, zone, shape, divider, room_pref)
    if not candidates and room_pref is not None:
        candidates = _candidate_origins(grid, occupied, zone, shape, divider, None)
    origin = _choose_origin(candidates, shape, zone, edge_pref, align)
    if origin is None:
        return None
    _paint_shape(grid, occupied, shape, origin, color)
    return origin


def _partition_zones(grid: np.ndarray, target: int) -> list[tuple[int, int, int, int]] | None:
    zones = [(1, 1, grid.shape[0] - 2, grid.shape[1] - 2)]
    for _ in range(target - 1):
        options = []
        for idx, (r0, c0, r1, c1) in enumerate(zones):
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            feasible = []
            if w >= 2 * MIN_ZONE_W + 1:
                feasible.append("v")
            if h >= 2 * MIN_ZONE_H + 1:
                feasible.append("h")
            if feasible:
                options.append((h * w, idx, feasible))
        if not options:
            return None
        options.sort(reverse=True)
        _area, idx, feasible = random.choice(options[: min(3, len(options))])
        r0, c0, r1, c1 = zones.pop(idx)
        orientation = random.choice(feasible)
        if orientation == "v":
            split = random.randint(c0 + MIN_ZONE_W, c1 - MIN_ZONE_W)
            grid[r0 : r1 + 1, split] = WALL
            zones.extend([(r0, c0, r1, split - 1), (r0, split + 1, r1, c1)])
        else:
            split = random.randint(r0 + MIN_ZONE_H, r1 - MIN_ZONE_H)
            grid[split, c0 : c1 + 1] = WALL
            zones.extend([(r0, c0, split - 1, c1), (split + 1, c0, r1, c1)])
    return zones


def _maybe_add_divider(grid: np.ndarray, zone: tuple[int, int, int, int]) -> dict | None:
    r0, c0, r1, c1 = zone
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    feasible = []
    if w >= 8:
        feasible.append("v")
    if h >= 8:
        feasible.append("h")
    if not feasible or random.random() < 0.2:
        return None
    orientation = random.choice(feasible)
    if orientation == "v":
        line = random.randint(c0 + 3, c1 - 3)
        door = random.randint(r0 + 1, r1 - 1)
        grid[r0 : r1 + 1, line] = WALL
        grid[door, line] = BG
    else:
        line = random.randint(r0 + 3, r1 - 3)
        door = random.randint(c0 + 1, c1 - 1)
        grid[line, c0 : c1 + 1] = WALL
        grid[line, door] = BG
    return {"orientation": orientation, "line": line, "door": door}


def _adjacent_pairs(zones: list[dict]) -> list[tuple[int, int, str, str, tuple[str, float]]]:
    pairs = []
    for i in range(len(zones)):
        r0a, c0a, r1a, c1a = zones[i]["bbox"]
        for j in range(i + 1, len(zones)):
            r0b, c0b, r1b, c1b = zones[j]["bbox"]
            row_overlap0 = max(r0a, r0b)
            row_overlap1 = min(r1a, r1b)
            col_overlap0 = max(c0a, c0b)
            col_overlap1 = min(c1a, c1b)
            if c1a + 2 == c0b and row_overlap0 <= row_overlap1:
                target = (row_overlap0 + row_overlap1) / 2.0
                pairs.append((i, j, "east", "west", ("row", target)))
            if c1b + 2 == c0a and row_overlap0 <= row_overlap1:
                target = (row_overlap0 + row_overlap1) / 2.0
                pairs.append((i, j, "west", "east", ("row", target)))
            if r1a + 2 == r0b and col_overlap0 <= col_overlap1:
                target = (col_overlap0 + col_overlap1) / 2.0
                pairs.append((i, j, "south", "north", ("col", target)))
            if r1b + 2 == r0a and col_overlap0 <= col_overlap1:
                target = (col_overlap0 + col_overlap1) / 2.0
                pairs.append((i, j, "north", "south", ("col", target)))
    return pairs


def _analyze_generated_components(grid: np.ndarray):
    labels, count = _label_components(grid != WALL)
    components = []
    for idx in range(1, count + 1):
        mask = labels == idx
        colors = set(map(int, np.unique(grid[mask])))
        palette_colors = list(colors - {BG, TEMPLATE, WALL})
        grey_mask = mask & (grid == TEMPLATE)
        if not palette_colors and not np.any(grey_mask):
            continue
        assert len(palette_colors) == 1
        palette = palette_colors[0]
        swatch_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask & (grid == palette))]
        grey_labels, grey_count = _label_components(grey_mask)
        grey_parts = []
        for gidx in range(1, grey_count + 1):
            grey_parts.append([tuple(map(int, rc)) for rc in np.argwhere(grey_labels == gidx)])
        components.append({"palette": palette, "swatch": swatch_cells, "grey_parts": grey_parts})
    return components


def _min_pairwise_distance(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            best = min(best, abs(r0 - r1) + abs(c0 - c1))
    return best


def _has_proximity_decoy(grid: np.ndarray) -> bool:
    components = _analyze_generated_components(grid)
    if len(components) < 2:
        return False
    for idx, comp in enumerate(components):
        own_swatch = comp["swatch"]
        for grey_part in comp["grey_parts"]:
            own = _min_pairwise_distance(grey_part, own_swatch)
            foreign = min(
                _min_pairwise_distance(grey_part, other["swatch"])
                for j, other in enumerate(components)
                if j != idx
            )
            if foreign < own:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(19, 27)
    w = random.randint(19, 27)
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    target_zones = random.choice([2, 2, 3])
    zones = _partition_zones(grid, target_zones)
    if zones is None or len(zones) != target_zones:
        return None

    zone_infos = []
    for bbox in zones:
        zone_infos.append({"bbox": bbox, "divider": _maybe_add_divider(grid, bbox)})

    pairs = _adjacent_pairs(zone_infos)
    if not pairs:
        return None
    a, b, side_a, side_b, align = random.choice(pairs)
    if random.random() < 0.5:
        a, b = b, a
        side_a, side_b = side_b, side_a

    colors = random.sample(PALETTE_COLORS, len(zone_infos))
    specs = []
    for idx, zone_info in enumerate(zone_infos):
        specs.append(
            {
                "color": colors[idx],
                "swatch_shape": random.choice(SWATCH_SHAPES),
                "template_total": random.randint(1, 3),
                "swatch_room": None,
            }
        )

    occupied = np.zeros_like(grid, dtype=bool)

    for idx, zone_info in enumerate(zone_infos):
        spec = specs[idx]
        if idx == a:
            edge_pref = (side_a, "far")
            room_pref = _room_pref_from_edge(zone_info["divider"], side_a, "far")
        elif idx == b:
            edge_pref = (side_b, "near")
            room_pref = _room_pref_from_edge(zone_info["divider"], side_b, "near")
        else:
            edge_pref = None if random.random() < 0.4 else (random.choice(["north", "south", "west", "east"]), "near")
            room_pref = None
        origin = _place_shape(
            grid,
            occupied,
            zone_info,
            spec["swatch_shape"],
            spec["color"],
            edge_pref=edge_pref,
            align=align if idx == b else None,
            room_pref=room_pref,
        )
        if origin is None:
            return None
        spec["swatch_room"] = _shape_room(origin, spec["swatch_shape"], zone_info["divider"])

    special_shape = _random_template_shape()
    special_origin = _place_shape(
        grid,
        occupied,
        zone_infos[a],
        special_shape,
        TEMPLATE,
        edge_pref=(side_a, "near"),
        align=align,
        room_pref=_room_pref_from_edge(zone_infos[a]["divider"], side_a, "near"),
    )
    if special_origin is None:
        return None
    specs[a]["template_total"] -= 1

    for idx, zone_info in enumerate(zone_infos):
        spec = specs[idx]
        for t in range(spec["template_total"]):
            placed = False
            for _ in range(20):
                shape = _random_template_shape()
                room_pref = None
                if zone_info["divider"] is not None and random.random() < 0.75:
                    room_pref = _opposite_room(spec["swatch_room"])
                edge_pref = None
                if t == 0 and random.random() < 0.5:
                    edge_pref = (random.choice(["north", "south", "west", "east"]), random.choice(["near", "far"]))
                origin = _place_shape(
                    grid,
                    occupied,
                    zone_info,
                    shape,
                    TEMPLATE,
                    edge_pref=edge_pref,
                    room_pref=room_pref,
                )
                if origin is not None:
                    placed = True
                    break
            if not placed:
                return None

    if _get_bg(grid) != BG:
        return None
    if not _has_proximity_decoy(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 8")
