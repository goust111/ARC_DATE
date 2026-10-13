import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

WALL_COLORS = (BLUE, RED)
TILE_COLORS = (GREEN, YELLOW, ORANGE)
WALL_TO_TILE = {BLUE: 3, RED: 2}

GEN_ATTEMPTS = 600
GEN_GRID_RANGE = (22, 29)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    if connectivity == 8:
        nbrs = _neighbors8
    else:
        def nbrs(r: int, c: int):
            yield r - 1, c
            yield r + 1, c
            yield r, c - 1
            yield r, c + 1

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in nbrs(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % th, c % tw])
    return out


def _majority(values: list[int]) -> int:
    counts = Counter(values)
    ordered = counts.most_common()
    assert ordered, "expected non-empty vote bucket"
    if len(ordered) > 1:
        assert ordered[0][1] > ordered[1][1], "tile vote must be unique"
    return int(ordered[0][0])


def _infer_tile(region: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = region.shape
    assert h % tile_size == 0 and w % tile_size == 0
    tile = np.zeros((tile_size, tile_size), dtype=int)
    for rr in range(tile_size):
        for cc in range(tile_size):
            vals = [
                int(region[r, c])
                for r in range(rr, h, tile_size)
                for c in range(cc, w, tile_size)
                if int(region[r, c]) != GREY
            ]
            tile[rr, cc] = _majority(vals)
    return tile


def _find_regions(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    wall_mask = np.isin(g, WALL_COLORS)
    comps = _components(wall_mask, connectivity=8)
    assert 2 <= len(comps) <= 3, "expected 2..3 wall rectangles"

    region_map = np.full(g.shape, -1, dtype=int)
    wall_info: dict[tuple[int, int], tuple[int, bool, bool]] = {}
    regions: list[dict] = []

    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each wall rectangle must be monochrome"
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "wall rectangle must have an interior"

        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == expected, "wall component must be a 1-cell rectangular frame"

        rid = len(regions)
        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        assert ir0 <= ir1 and ic0 <= ic1
        assert np.all(region_map[ir0 : ir1 + 1, ic0 : ic1 + 1] == -1), "regions must not overlap"
        assert not np.any(wall_mask[ir0 : ir1 + 1, ic0 : ic1 + 1]), "walls must stay on the border"
        region_map[ir0 : ir1 + 1, ic0 : ic1 + 1] = rid

        for c in range(c0, c1 + 1):
            wall_info[(r0, c)] = (color, False, True)
            wall_info[(r1, c)] = (color, False, True)
        for r in range(r0, r1 + 1):
            wall_info[(r, c0)] = (color, True, False)
            wall_info[(r, c1)] = (color, True, False)
        wall_info[(r0, c0)] = (color, True, True)
        wall_info[(r0, c1)] = (color, True, True)
        wall_info[(r1, c0)] = (color, True, True)
        wall_info[(r1, c1)] = (color, True, True)

        regions.append(
            {
                "wall_color": color,
                "bbox": (r0, c0, r1, c1),
                "interior": (ir0, ic0, ir1, ic1),
            }
        )
    return regions, wall_info, region_map


def _find_arrow(grid: np.ndarray, region_map: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    mask = (g == GREY) & (region_map < 0)
    comps = _components(mask, connectivity=8)
    candidates = []
    h, w = g.shape

    for coords in comps:
        if int(coords.shape[0]) != 3:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0, c1 - c0) != (1, 1):
            continue
        if (r0, c0, r1, c1) not in {
            (0, 0, 1, 1),
            (0, w - 2, 1, w - 1),
            (h - 2, 0, h - 1, 1),
            (h - 2, w - 2, h - 1, w - 1),
        }:
            continue
        local = {(int(r - r0), int(c - c0)) for r, c in coords}
        missing = ({(0, 0), (0, 1), (1, 0), (1, 1)} - local).pop()
        dr = 1 if missing[0] == 1 else -1
        dc = 1 if missing[1] == 1 else -1
        candidates.append(((r0 + missing[0], c0 + missing[1]), (dr, dc)))

    assert len(candidates) == 1, "expected one corner arrow sprite"
    return candidates[0]


def _trace_ray(
    grid: np.ndarray,
    regions: list[dict],
    wall_info: dict[tuple[int, int], tuple[int, bool, bool]],
    region_map: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r, c = start
    dr, dc = direction
    last_color = -1
    reached: dict[int, int] = {}
    seen: set[tuple[int, int, int, int, int]] = set()

    for _ in range(h * w * 12):
        state = (r, c, dr, dc, last_color)
        if state in seen:
            break
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        info = wall_info.get((nr, nc))
        if info is not None:
            color, vertical, horizontal = info
            last_color = int(color)
            if vertical and horizontal:
                dr, dc = -dr, -dc
            elif vertical:
                dr = -dr
            elif horizontal:
                dc = -dc
            r, c = nr, nc
            continue

        r, c = nr, nc
        rid = int(region_map[r, c])
        if rid >= 0 and rid not in reached:
            assert last_color == regions[rid]["wall_color"], "entered region must inherit that frame color"
            reached[rid] = last_color

    return reached


def _repair_reached(grid: np.ndarray, regions: list[dict], reached: dict[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rid, wall_color in reached.items():
        tile_size = WALL_TO_TILE[int(wall_color)]
        ir0, ic0, ir1, ic1 = regions[rid]["interior"]
        patch = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        tile = _infer_tile(patch, tile_size)
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = _tile_fill(patch.shape, tile)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair only the periodic regions reached by the corner ray after mirror-like wall bounces.

    Input structure:
    - Black background contains 2..3 disjoint 1-cell-thick rectangular wall frames colored blue or red.
    - Each frame encloses a tiled rectangle using green, yellow, and orange, with several grey noise pixels.
    - A grey 2x2 L-arrow sits in one corner; its missing corner points along the diagonal launch direction.

    Rule:
    - Launch a diagonal ray from the arrow's missing-corner cell and move it through all non-wall cells without painting.
    - Whenever the ray steps onto a wall pixel, reflect it using that wall pixel's orientation and remember that wall color.
    - Each time the ray first enters a framed region, use the last remembered wall color to choose that region's tile family, infer the periodic tile by residue-class majority while ignoring grey votes, and replace the whole region with the clean repetition.
    - Leave every unreached region, every wall, and the arrow sprite unchanged.

    Key insight:
    The ray path decides which noisy regions are trustworthy enough to repair, and the color inherited from the bounce determines which periodic template size applies there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions, wall_info, region_map = _find_regions(g)
    start, direction = _find_arrow(g, region_map)
    reached = _trace_ray(g, regions, wall_info, region_map, start, direction)
    return _repair_reached(g, regions, reached)


def _make_base_tile(tile_size: int) -> np.ndarray | None:
    for _ in range(200):
        color_count = random.choice((2, 3))
        colors = random.sample(TILE_COLORS, color_count)
        tile = np.array(random.choices(colors, k=tile_size * tile_size), dtype=int).reshape(tile_size, tile_size)
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if tile_size == 2:
            if np.array_equal(tile[0], tile[1]):
                continue
            if np.array_equal(tile[:, 0], tile[:, 1]):
                continue
        else:
            rows = {tuple(map(int, row)) for row in tile}
            cols = {tuple(map(int, tile[:, c])) for c in range(tile.shape[1])}
            if len(rows) == 1 or len(cols) == 1:
                continue
        return tile
    return None


def _add_noise(clean: np.ndarray, tile_size: int) -> np.ndarray | None:
    noisy = clean.copy()
    h, w = clean.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    target = random.randint(2, 4)
    chosen: list[tuple[int, int]] = []
    residue_hits = Counter()

    for r, c in cells:
        rr = r % tile_size
        cc = c % tile_size
        residue_total = (h // tile_size) * (w // tile_size)
        if residue_hits[(rr, cc)] + 1 >= residue_total:
            continue
        chosen.append((r, c))
        residue_hits[(rr, cc)] += 1
        if len(chosen) == target:
            break

    if len(chosen) != target:
        return None

    for r, c in chosen:
        noisy[r, c] = GREY
    return noisy


def _build_region_patch(wall_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    tile_size = WALL_TO_TILE[int(wall_color)]
    tile = _make_base_tile(tile_size)
    if tile is None:
        return None

    if tile_size == 2:
        h = tile_size * random.choice((3, 4))
        w = tile_size * random.choice((3, 4))
    else:
        h = tile_size * random.choice((2, 3))
        w = tile_size * random.choice((2, 3))

    clean = _tile_fill((h, w), tile)
    noisy = _add_noise(clean, tile_size)
    if noisy is None:
        return None
    return clean, noisy


def _can_place(bboxes: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int], gap: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in bboxes:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _place_rectangles(
    shape: tuple[int, int], region_specs: list[dict]
) -> list[tuple[int, int, dict]] | None:
    h, w = shape
    placed: list[tuple[int, int, dict]] = []
    bboxes: list[tuple[int, int, int, int]] = []
    order = list(region_specs)
    random.shuffle(order)

    for spec in order:
        patch = spec["noisy"]
        ph, pw = patch.shape
        oh, ow = ph + 2, pw + 2
        positions = [
            (r0, c0)
            for r0 in range(2, h - oh - 1)
            for c0 in range(2, w - ow - 1)
        ]
        random.shuffle(positions)
        chosen = None
        for r0, c0 in positions:
            bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            if _can_place(bboxes, bbox, gap=1):
                chosen = (r0, c0)
                bboxes.append(bbox)
                break
        if chosen is None:
            return None
        placed.append((chosen[0], chosen[1], spec))

    return placed


def _draw_canonical_grid(shape: tuple[int, int], placed: list[tuple[int, int, dict]]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)

    arrow = np.array([[GREY, GREY], [GREY, BG]], dtype=int)
    grid[:2, :2] = arrow

    for r0, c0, spec in placed:
        patch = spec["noisy"]
        wall_color = spec["wall_color"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph + 2, c0 : c0 + pw + 2] = wall_color
        grid[r0 + 1 : r0 + ph + 1, c0 + 1 : c0 + pw + 1] = patch
    return grid


def _all_repair_output(grid: np.ndarray) -> np.ndarray:
    regions, _, _ = _find_regions(grid)
    reached = {rid: region["wall_color"] for rid, region in enumerate(regions)}
    return _repair_reached(grid, regions, reached)


def _transform_variant(grid: np.ndarray) -> np.ndarray:
    variant = random.choice(("id", "lr", "ud", "both"))
    if variant == "lr":
        return np.fliplr(grid)
    if variant == "ud":
        return np.flipud(grid)
    if variant == "both":
        return np.flipud(np.fliplr(grid))
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        regions, wall_info, region_map = _find_regions(grid)
        start, direction = _find_arrow(grid, region_map)
        reached = _trace_ray(grid, regions, wall_info, region_map, start, direction)
        out = main(grid)
        all_fixed = _all_repair_output(grid)
    except Exception:
        return False

    if not (1 <= len(reached) < len(regions)):
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, all_fixed):
        return False

    repaired_colors = {regions[rid]["wall_color"] for rid in reached}
    if not repaired_colors.issubset(set(WALL_COLORS)):
        return False

    for rid, region in enumerate(regions):
        ir0, ic0, ir1, ic1 = region["interior"]
        before = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
        after = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        if rid in reached:
            if np.count_nonzero(after == GREY) != 0:
                return False
            if np.array_equal(before, after):
                return False
        else:
            if not np.array_equal(before, after):
                return False
            if np.count_nonzero(before == GREY) == 0:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    n_regions = random.choice((2, 3, 3))

    wall_colors = [BLUE, RED]
    while len(wall_colors) < n_regions:
        wall_colors.append(random.choice(WALL_COLORS))
    random.shuffle(wall_colors)

    region_specs: list[dict] = []
    for wall_color in wall_colors:
        built = _build_region_patch(wall_color)
        if built is None:
            return None
        clean, noisy = built
        region_specs.append({"wall_color": wall_color, "clean": clean, "noisy": noisy})

    placed = _place_rectangles((h, w), region_specs)
    if placed is None:
        return None

    grid = _draw_canonical_grid((h, w), placed)
    grid = _transform_variant(grid)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bounce-constrained tile-repair input")
