from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
DEFAULT = 4
SOURCE = 7
VOTE_COLORS = (1, 2, 3)

TILE_SIZE = 6
CROSS = 3
TILE_GAP = 1
GRID_TILES = 4
MARGIN_CHOICES = (0, 1)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


BASE_H_TOP = {
    (0, 0): 2,
    (0, 1): 1,
    (0, 2): 1,
    (0, 3): 4,
    (1, 0): 2,
    (1, 3): 1,
    (2, 0): 4,
    (2, 1): 2,
    (2, 2): 3,
    (2, 3): 4,
    (3, 0): 4,
    (3, 1): 4,
    (3, 2): 3,
    (3, 3): 4,
}

BASE_V_LEFT = {
    (0, 0): 4,
    (0, 1): 1,
    (0, 2): 4,
    (0, 3): 4,
    (1, 0): 1,
    (1, 2): 1,
    (1, 3): 2,
    (2, 0): 1,
    (2, 2): 4,
    (2, 3): 2,
    (3, 0): 1,
    (3, 1): 4,
    (3, 2): 3,
    (3, 3): 3,
}

BASE_V_RIGHT = {
    (0, 0): 4,
    (0, 1): 4,
    (0, 2): 2,
    (0, 3): 1,
    (1, 0): 2,
    (1, 1): 3,
    (1, 3): 1,
    (2, 0): 4,
    (2, 1): 3,
    (2, 3): 1,
    (3, 0): 2,
    (3, 1): 4,
    (3, 2): 2,
    (3, 3): 1,
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _tile_patch(color: int) -> np.ndarray:
    patch = np.full((TILE_SIZE, TILE_SIZE), GREY, dtype=int)
    patch[CROSS, :] = BG
    patch[:, CROSS] = BG
    patch[1:3, 1:3] = int(color)
    return patch


def _tile_origin(tile_rc: tuple[int, int], margin: int) -> tuple[int, int]:
    tr, tc = tile_rc
    r0 = int(margin + tr * (TILE_SIZE + TILE_GAP))
    c0 = int(margin + tc * (TILE_SIZE + TILE_GAP))
    return r0, c0


def _grid_shape(margin: int) -> tuple[int, int]:
    side = 2 * int(margin) + GRID_TILES * TILE_SIZE + (GRID_TILES - 1) * TILE_GAP
    return side, side


def _flip_rows_map(color_map: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    return {(GRID_TILES - 1 - r, c): color for (r, c), color in color_map.items()}


def _flip_cols_map(color_map: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    return {(r, GRID_TILES - 1 - c): color for (r, c), color in color_map.items()}


def _flip_rows_tiles(tiles: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((GRID_TILES - 1 - r, c) for r, c in tiles))


def _flip_cols_tiles(tiles: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r, GRID_TILES - 1 - c) for r, c in tiles))


SOURCE_CASES = [
    {
        "sources": ((1, 1), (1, 2)),
        "colors": BASE_H_TOP,
    },
    {
        "sources": _flip_rows_tiles(((1, 1), (1, 2))),
        "colors": _flip_rows_map(BASE_H_TOP),
    },
    {
        "sources": ((1, 1), (2, 1)),
        "colors": BASE_V_LEFT,
    },
    {
        "sources": ((1, 2), (2, 2)),
        "colors": BASE_V_RIGHT,
    },
]


def _render_case(source_tiles: tuple[tuple[int, int], ...], color_map: dict[tuple[int, int], int], margin: int) -> np.ndarray:
    h, w = _grid_shape(margin)
    grid = np.full((h, w), BG, dtype=int)
    for tr in range(GRID_TILES):
        for tc in range(GRID_TILES):
            color = SOURCE if (tr, tc) in source_tiles else int(color_map[(tr, tc)])
            r0, c0 = _tile_origin((tr, tc), margin)
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = _tile_patch(color)
    return grid


def _parse_tiles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mark_colors = sorted(int(c) for c in np.unique(g) if c not in (BG, GREY))
    tiles: list[dict] = []

    for color in mark_colors:
        for coords in components_mask(g == color):
            assert coords.shape[0] == 4, "each tile marker must be a 2x2 block"
            rb0, cb0, rb1, cb1 = bbox_coords(coords)
            assert rb1 == rb0 + 1 and cb1 == cb0 + 1, "marker blocks must be 2x2"
            r0 = rb0 - 1
            c0 = cb0 - 1
            r1 = r0 + TILE_SIZE - 1
            c1 = c0 + TILE_SIZE - 1
            assert in_bounds(g.shape, r0, c0) and in_bounds(g.shape, r1, c1), "tile bbox must stay in-bounds"

            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert patch.shape == (TILE_SIZE, TILE_SIZE)
            assert np.all(patch[1:3, 1:3] == color), "marker position must be consistent"
            assert np.all(patch[CROSS, :] == BG), "channel row must stay black"
            assert np.all(patch[:, CROSS] == BG), "channel col must stay black"

            expected_grey = np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)
            expected_grey[CROSS, :] = False
            expected_grey[:, CROSS] = False
            expected_grey[1:3, 1:3] = False
            assert np.all(patch[expected_grey] == GREY), "non-channel scaffold must stay grey"

            tiles.append(
                {
                    "color": int(color),
                    "mark_bbox": (rb0, cb0, rb1, cb1),
                    "tile_bbox": (r0, c0, r1, c1),
                    "center": (r0 + CROSS, c0 + CROSS),
                    "tile_rc": ((r0 - min(r0, 0)) // (TILE_SIZE + TILE_GAP), (c0 - min(c0, 0)) // (TILE_SIZE + TILE_GAP)),
                }
            )

    assert len(tiles) == GRID_TILES * GRID_TILES, "expected a 4x4 tile lattice"
    bboxes = [tile["tile_bbox"] for tile in tiles]
    assert len({bbox for bbox in bboxes}) == len(bboxes), "tile bboxes must be unique"
    return tiles


def _compute_ray_mask(grid: np.ndarray, tiles: list[dict], *, sources: set[tuple[int, int]] | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    ray = np.zeros(g.shape, dtype=bool)

    for tile in tiles:
        if tile["color"] != SOURCE:
            continue
        if sources is not None and tile["center"] not in sources:
            continue

        cr, cc = tile["center"]
        assert g[cr, cc] == BG, "source tile hub must be black"
        ray[cr, cc] = True
        for dr, dc in DIRS4:
            r = cr + dr
            c = cc + dc
            while in_bounds(g.shape, r, c) and int(g[r, c]) == BG:
                ray[r, c] = True
                r += dr
                c += dc

    return ray


def _channel_regions(grid: np.ndarray, ray_mask: np.ndarray) -> list[np.ndarray]:
    black = (np.asarray(grid, dtype=int) == BG) & (~ray_mask)
    return components_mask(black)


def _region_touches_tile(region: np.ndarray, tile_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = tile_bbox
    for r, c in region:
        if r0 - 1 <= int(r) <= r1 + 1 and c0 - 1 <= int(c) <= c1 + 1:
            return True
    return False


def _region_majorities(tiles: list[dict], regions: list[np.ndarray]) -> list[int]:
    winners: list[int] = []
    for region in regions:
        eligible: list[int] = []
        for tile in tiles:
            color = int(tile["color"])
            if color in (DEFAULT, SOURCE):
                continue
            if _region_touches_tile(region, tile["tile_bbox"]):
                eligible.append(color)

        assert len(eligible) >= 2, "every channel region must see at least two voting tiles"
        counts = Counter(eligible).most_common()
        assert counts, "expected at least one voting color"
        if len(counts) > 1:
            assert counts[0][1] > counts[1][1], "majority color must be unique"
        winners.append(int(counts[0][0]))
    assert len(set(winners)) >= 2, "not all regions should vote the same way"
    return winners


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray, list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    tiles = _parse_tiles(g)
    ray_mask = _compute_ray_mask(g, tiles)
    regions = _channel_regions(g, ray_mask)
    assert len(regions) >= 3, "rays must split the channel network into at least three regions"
    winners = _region_majorities(tiles, regions)

    out = g.copy()
    out[ray_mask] = SOURCE
    for region, color in zip(regions, winners):
        out[region[:, 0], region[:, 1]] = int(color)
    return out, tiles, ray_mask, regions, winners


def main(grid: np.ndarray) -> np.ndarray:
    """Split the black channel network with orange source-tile rays, then flood each cut region with the local majority marker color.

    Input structure:
    - Black is the background channel color.
    - A regular grey tile lattice contains one solid 2x2 marker block per tile.
    - Most markers are yellow defaults; two orange marker tiles are ray sources; the remaining blue/red/green markers are voters.
    - The black cross-shaped passages inside the tiles and the black gaps between tiles form one connected channel network.

    Rule:
    - Recover each tile from its 2x2 marker block and locate the two orange source tiles.
    - From the black channel hub of every orange tile, cast straight rays in the four cardinal directions through black channel cells, stopping at the first grey scaffold or the grid boundary, and repaint those traversed channel cells orange.
    - Treat the orange rays as new walls, split the remaining black channel cells into 4-connected regions, and for each region count the non-yellow non-orange marker colors of tiles whose tile bbox 8-touches that region.
    - Fill every region with its unique majority color while leaving the grey tile scaffolds, all 2x2 tile markers, and the orange ray cells unchanged.

    Key insight:
    - The orange tiles both carve the channel topology and opt out of voting, so the answer comes from first cutting the channel network and only then taking a majority vote among the neighboring non-default marker tiles of each resulting compartment.
    """
    out, _tiles, _ray_mask, _regions, _winners = _solve(np.asarray(grid, dtype=int))
    return out


def _permute_votes(color_map: dict[tuple[int, int], int], perm: dict[int, int]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for tile, color in color_map.items():
        out[tile] = perm.get(int(color), int(color))
    return out


def _source_centers(source_tiles: tuple[tuple[int, int], ...], margin: int) -> set[tuple[int, int]]:
    centers = set()
    for tile_rc in source_tiles:
        r0, c0 = _tile_origin(tile_rc, margin)
        centers.add((r0 + CROSS, c0 + CROSS))
    return centers


def _region_signature(regions: list[np.ndarray]) -> tuple[tuple[int, int, int, int, int], ...]:
    sig = []
    for region in regions:
        r0, c0, r1, c1 = bbox_coords(region)
        sig.append((int(region.shape[0]), r0, c0, r1, c1))
    return tuple(sorted(sig))


def _validate_generated_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    base_regions = components_mask(g == BG)
    assert len(base_regions) == 1, "the raw channel network should start connected"

    out, tiles, ray_mask, regions, winners = _solve(g)
    assert not np.any(out == BG), "all non-ray channels should be flood-filled"
    assert np.any(ray_mask), "expected orange rays"
    assert not np.array_equal(out, g), "the puzzle should change the grid"

    tile_counts = Counter(tile["color"] for tile in tiles if tile["color"] != SOURCE)
    assert tile_counts[DEFAULT] >= max(tile_counts[c] for c in VOTE_COLORS), "yellow should stay the most common non-source marker color"
    assert tile_counts[DEFAULT] > 0
    assert len([tile for tile in tiles if tile["color"] == SOURCE]) == 2
    assert len(set(winners)) >= 2

    full_sig = _region_signature(regions)
    source_centers = _source_centers(tuple(sorted(tile["tile_rc"] for tile in tiles if tile["color"] == SOURCE)), 0)
    if not source_centers:
        source_centers = {tile["center"] for tile in tiles if tile["color"] == SOURCE}

    all_source_centers = [tile["center"] for tile in tiles if tile["color"] == SOURCE]
    for center in all_source_centers:
        reduced_sources = {c for c in all_source_centers if c != center}
        reduced_ray = _compute_ray_mask(g, tiles, sources=reduced_sources)
        reduced_regions = _channel_regions(g, reduced_ray)
        assert _region_signature(reduced_regions) != full_sig, "each source should matter to the split"


def generate_input() -> np.ndarray:
    for _ in range(100):
        case = random.choice(SOURCE_CASES)
        perm_values = random.sample(VOTE_COLORS, k=len(VOTE_COLORS))
        perm = {src: dst for src, dst in zip(VOTE_COLORS, perm_values)}
        color_map = _permute_votes(case["colors"], perm)
        margin = random.choice(MARGIN_CHOICES)
        grid = _render_case(case["sources"], color_map, margin)
        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid ray-split majority-fill input")
