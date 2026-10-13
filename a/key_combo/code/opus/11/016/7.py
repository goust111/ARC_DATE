from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
BASE_LINE_COLORS = (1, 2, 3)  # blue, red, green

GEN_RETRIES = 500
GRID_SIZE_RANGE = (13, 16)
REGION_MIN_AREA = 24
LAYOUT_TYPES = ("cross4", "t3", "triple4")
T_ORIENTATIONS = ("top", "bottom", "left", "right")
TRIPLE_ORIENTATIONS = ("vertical", "horizontal")
COMPONENT_PLACEMENT_RETRIES = 120
POLYOMINO_RETRIES = 80


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _segment_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        c0, c1 = sorted((int(c0), int(c1)))
        return [(int(r0), c) for c in range(c0, c1 + 1)]
    r0, r1 = sorted((int(r0), int(r1)))
    return [(r, int(c0)) for r in range(r0, r1 + 1)]


def _extract_marker_pairs(grid: np.ndarray):
    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, GREY))
    pairs = []
    for color in colors:
        coords = np.argwhere(grid == color)
        if coords.shape != (2, 2):
            raise AssertionError(f"color {color} must appear exactly twice")
        (r0, c0), (r1, c1) = coords.tolist()
        if (r0 == r1) == (c0 == c1):
            raise AssertionError(f"color {color} markers must share exactly one row or one column")
        orient = "h" if r0 == r1 else "v"
        pairs.append((int(color), int(r0), int(c0), int(r1), int(c1), orient))
    return pairs


def _draw_segments(grid: np.ndarray, pairs) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(grid, dtype=int).copy()
    line_mask = np.zeros(out.shape, dtype=bool)

    for color, r0, c0, r1, c1, orient in pairs:
        if orient != "h":
            continue
        for rr, cc in _segment_cells(r0, c0, r1, c1):
            out[rr, cc] = color
            line_mask[rr, cc] = True

    for color, r0, c0, r1, c1, orient in pairs:
        if orient != "v":
            continue
        for rr, cc in _segment_cells(r0, c0, r1, c1):
            out[rr, cc] = color
            line_mask[rr, cc] = True

    return out, line_mask


def _sorted_regions(line_mask: np.ndarray) -> list[np.ndarray]:
    comps = _component_coords(~line_mask)
    comps.sort(key=lambda coords: _bbox(coords)[:2])
    return comps


def _region_boundary_profile(
    line_grid: np.ndarray, line_mask: np.ndarray, region: np.ndarray
) -> tuple[dict[int, int], int, bool]:
    counts: dict[int, int] = {}
    first_touch: dict[int, tuple[int, int]] = {}
    region_set = _coords_set(region)
    h, w = line_grid.shape
    for r, c in region:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in region_set or not line_mask[nr, nc]:
                continue
            color = int(line_grid[nr, nc])
            counts[color] = counts.get(color, 0) + 1
            if color not in first_touch or (nr, nc) < first_touch[color]:
                first_touch[color] = (nr, nc)
    if not counts:
        raise AssertionError("every region must touch at least one line segment")

    winner = max(
        counts,
        key=lambda color: (
            counts[color],
            -first_touch[color][0],
            -first_touch[color][1],
        ),
    )
    top_counts = sorted(counts.values(), reverse=True)
    unique = len(top_counts) == 1 or top_counts[0] > top_counts[1]
    return counts, int(winner), bool(unique)


def _largest_component(comps: list[np.ndarray]) -> np.ndarray:
    return max(comps, key=lambda coords: (len(coords), -_bbox(coords)[0], -_bbox(coords)[1]))


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pairs = _extract_marker_pairs(g)
    drawn, line_mask = _draw_segments(g, pairs)

    out = drawn.copy()
    out[out == GREY] = BG

    for region in _sorted_regions(line_mask):
        grey_mask = np.zeros(g.shape, dtype=bool)
        for r, c in region:
            rr = int(r)
            cc = int(c)
            if int(drawn[rr, cc]) == GREY:
                grey_mask[rr, cc] = True

        grey_comps = _component_coords(grey_mask)
        if not grey_comps:
            continue

        keep = _largest_component(grey_comps)
        _counts, color, _unique = _region_boundary_profile(drawn, line_mask, region)
        for r, c in keep:
            out[int(r), int(c)] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw same-color divider segments, then keep and recolor the largest grey component in each divided region.

    Input structure:
    - A black background contains several separate grey 4-connected components.
    - There are 2-3 same-color marker pairs in blue/red/optional green; each pair shares a row or a column and indicates a horizontal or vertical segment.
    - Grey components lie in the open space that will be partitioned once the marker pairs are connected.

    Rule:
    - Connect each same-color marker pair with its full axis-aligned segment, drawing horizontal segments first and vertical segments second so vertical color wins at crossings.
    - Treat all segment pixels, including the markers, as blocking walls and split the remaining cells into 4-connected regions.
    - Inside each region, find the grey components, erase every non-largest one, and recolor the unique largest grey component with the line color that touches that region boundary most often.

    Key insight:
    - The drawn segments serve two roles at once: they partition the grey objects into comparison groups, and their boundary-contact counts determine the recolor assigned to each surviving component.
    """
    return _transform(grid)


def _blank(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, BG, dtype=int)


def _line_grid_from_pairs(shape: tuple[int, int], pairs) -> tuple[np.ndarray, np.ndarray]:
    return _draw_segments(_blank(shape), pairs)


def _make_cross_layout(h: int, w: int, rng: random.Random):
    for _ in range(200):
        row = rng.randint(4, h - 5)
        col = rng.randint(4, w - 5)
        pairs = [
            (2, row, 0, row, w - 1, "h"),
            (1, 0, col, h - 1, col, "v"),
        ]
        line_grid, line_mask = _line_grid_from_pairs((h, w), pairs)
        regions = _sorted_regions(line_mask)
        if len(regions) != 4:
            continue
        if min(len(region) for region in regions) < REGION_MIN_AREA:
            continue
        return pairs, line_grid, line_mask, regions
    return None


def _make_t_layout(h: int, w: int, rng: random.Random):
    for _ in range(250):
        orient = rng.choice(T_ORIENTATIONS)
        if orient in ("top", "bottom"):
            row = rng.randint(4, h - 5)
            col = rng.randint(4, w - 5)
            pairs = [(2, row, 0, row, w - 1, "h")]
            if orient == "top":
                pairs.append((1, 0, col, row, col, "v"))
            else:
                pairs.append((1, row, col, h - 1, col, "v"))
        else:
            row = rng.randint(4, h - 5)
            col = rng.randint(4, w - 5)
            pairs = [(2, 0, col, h - 1, col, "v")]
            if orient == "left":
                pairs.append((1, row, 0, row, col, "h"))
            else:
                pairs.append((1, row, col, row, w - 1, "h"))

        line_grid, line_mask = _line_grid_from_pairs((h, w), pairs)
        regions = _sorted_regions(line_mask)
        if len(regions) != 3:
            continue
        if min(len(region) for region in regions) < REGION_MIN_AREA:
            continue
        return pairs, line_grid, line_mask, regions
    return None


def _make_triple_layout(h: int, w: int, rng: random.Random):
    for _ in range(250):
        orient = rng.choice(TRIPLE_ORIENTATIONS)
        if orient == "vertical":
            row = rng.randint(4, h - 5)
            col = rng.randint(4, w - 5)
            pairs = [
                (2, row, 0, row, w - 1, "h"),
                (1, 0, col, row - 1, col, "v"),
                (3, row + 1, col, h - 1, col, "v"),
            ]
        else:
            row = rng.randint(4, h - 5)
            col = rng.randint(4, w - 5)
            pairs = [
                (2, 0, col, h - 1, col, "v"),
                (1, row, 0, row, col - 1, "h"),
                (3, row, col + 1, row, w - 1, "h"),
            ]

        line_grid, line_mask = _line_grid_from_pairs((h, w), pairs)
        regions = _sorted_regions(line_mask)
        if len(regions) != 4:
            continue
        if min(len(region) for region in regions) < REGION_MIN_AREA:
            continue
        return pairs, line_grid, line_mask, regions
    return None


def _choose_layout(rng: random.Random):
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    layout_type = rng.choices(LAYOUT_TYPES, weights=(4, 3, 3), k=1)[0]
    maker = {
        "cross4": _make_cross_layout,
        "t3": _make_t_layout,
        "triple4": _make_triple_layout,
    }[layout_type]
    return layout_type, maker(h, w, rng)


def _random_polyomino(size: int, rng: random.Random) -> np.ndarray:
    for _ in range(POLYOMINO_RETRIES):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < size and frontier:
            nxt = rng.choice(tuple(frontier))
            frontier.remove(nxt)
            cells.add(nxt)
            r, c = nxt
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != size:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        min_r = min(rows)
        min_c = min(cols)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        coords = np.asarray(norm, dtype=int)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h == 1 or w == 1:
            continue
        if h * w == size:
            continue
        return coords
    raise RuntimeError("failed to create an irregular polyomino")


def _sample_component_sizes(region_area: int, count: int, rng: random.Random) -> list[int] | None:
    limit_ratio = 0.48 if count == 2 else 0.58
    for _ in range(120):
        sizes = sorted(rng.sample(range(3, 9), count), reverse=True)
        if sum(sizes) <= int(region_area * limit_ratio):
            return sizes
    return None


def _dilated_forbidden(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _place_components_in_region(
    shape: tuple[int, int],
    region: np.ndarray,
    count: int,
    rng: random.Random,
) -> list[tuple[int, np.ndarray]] | None:
    region_set = _coords_set(region)
    region_area = len(region)
    sizes = _sample_component_sizes(region_area, count, rng)
    if sizes is None:
        return None

    r0, c0, r1, c1 = _bbox(region)
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    placed: list[tuple[int, np.ndarray]] = []
    forbidden: set[tuple[int, int]] = set()

    for size in sizes:
        success = False
        for _ in range(COMPONENT_PLACEMENT_RETRIES):
            poly = _random_polyomino(size, rng)
            poly_h = int(poly[:, 0].max()) + 1
            poly_w = int(poly[:, 1].max()) + 1
            if poly_h > region_h or poly_w > region_w:
                continue

            anchors = [
                (ar, ac)
                for ar in range(r0, r1 - poly_h + 2)
                for ac in range(c0, c1 - poly_w + 2)
            ]
            rng.shuffle(anchors)
            for ar, ac in anchors:
                coords = poly + np.array([ar, ac], dtype=int)
                coord_list = [(int(r), int(c)) for r, c in coords]
                if any((rr, cc) not in region_set for rr, cc in coord_list):
                    continue
                if any((rr, cc) in forbidden for rr, cc in coord_list):
                    continue
                placed.append((size, coords.copy()))
                forbidden.update(_dilated_forbidden(coords, shape))
                success = True
                break
            if success:
                break
        if not success:
            return None

    return placed


def _region_component_counts(layout_type: str, regions: list[np.ndarray], rng: random.Random) -> list[int]:
    counts = [2] * len(regions)
    if layout_type == "t3":
        extra = rng.choices([0, 1, 2], weights=(2, 3, 2), k=1)[0]
        area_order = sorted(range(len(regions)), key=lambda idx: len(regions[idx]), reverse=True)
        for idx in area_order[:extra]:
            counts[idx] += 1
    return counts


def _build_expected(
    shape: tuple[int, int],
    line_grid: np.ndarray,
    region_data: list[tuple[np.ndarray, int]],
    placements: list[list[tuple[int, np.ndarray]]],
) -> np.ndarray:
    out = line_grid.copy()
    for (region, color), region_parts in zip(region_data, placements):
        keep_size, keep_coords = max(region_parts, key=lambda item: (item[0], -_bbox(item[1])[0], -_bbox(item[1])[1]))
        _ = keep_size
        for r, c in keep_coords:
            out[int(r), int(c)] = int(color)
    return out


def _render_input(shape: tuple[int, int], pairs, placements) -> np.ndarray:
    inp = _blank(shape)
    for color, r0, c0, r1, c1, _orient in pairs:
        inp[int(r0), int(c0)] = int(color)
        inp[int(r1), int(c1)] = int(color)
    for region_parts in placements:
        for _size, coords in region_parts:
            for r, c in coords:
                inp[int(r), int(c)] = GREY
    return inp


def _crossing_count(pairs) -> int:
    horizontals = [p for p in pairs if p[-1] == "h"]
    verticals = [p for p in pairs if p[-1] == "v"]
    total = 0
    for _hc, hr0, hc0, _hr1, hc1, _ in horizontals:
        row = hr0
        c0 = min(hc0, hc1)
        c1 = max(hc0, hc1)
        for _vc, vr0, vc0, vr1, _vc1, _ in verticals:
            col = vc0
            r0 = min(vr0, vr1)
            r1 = max(vr0, vr1)
            if r0 <= row <= r1 and c0 <= col <= c1:
                total += 1
    return total


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    choice = _choose_layout(rng)
    layout_type, payload = choice
    if payload is None:
        return None

    pairs, line_grid, line_mask, regions = payload
    if _crossing_count(pairs) > 1:
        return None

    region_infos: list[tuple[np.ndarray, int]] = []
    winner_colors = []
    for region in regions:
        counts, winner, unique = _region_boundary_profile(line_grid, line_mask, region)
        if not unique:
            return None
        if len(counts) > 1 and max(counts.values()) == min(counts.values()):
            return None
        region_infos.append((region, winner))
        winner_colors.append(winner)

    if len(set(winner_colors)) < 2:
        return None

    counts = _region_component_counts(layout_type, regions, rng)
    if not (5 <= sum(counts) <= 8):
        return None

    placements: list[list[tuple[int, np.ndarray]]] = []
    for region, count in zip(regions, counts):
        placed = _place_components_in_region(line_grid.shape, region, count, rng)
        if placed is None:
            return None
        placements.append(placed)

    inp = _render_input(line_grid.shape, pairs, placements)
    expected = _build_expected(line_grid.shape, line_grid, region_infos, placements)

    if int((inp == GREY).sum()) < 18:
        return None
    if any((inp == color).sum() != 2 for color in sorted({pair[0] for pair in pairs})):
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(inp, actual):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
