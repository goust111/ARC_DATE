import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste, random_free_location


TILE_SIZE = 5
INNER_MIN = 1
INNER_MAX = TILE_SIZE - 2
ARRAY_GAP = 1
ARRAY_SIDE = 3
ARRAY_SPAN = ARRAY_SIDE * TILE_SIZE + (ARRAY_SIDE - 1) * ARRAY_GAP
SPRITE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE]
MAX_ATTEMPTS = 400

SHAPE_CLASSES = {
    "domino": ((0, 0), (0, 1)),
    "line3": ((0, 0), (0, 1), (0, 2)),
    "el3": ((0, 0), (1, 0), (1, 1)),
    "square4": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "el4": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "tee4": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "zig4": ((0, 1), (0, 2), (1, 0), (1, 1)),
}
ASYMMETRIC_SHAPES = ["el3", "el4", "zig4"]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = np.asarray(list(coords), dtype=int)
    if pts.size == 0:
        return ()
    pts -= pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _bbox_of_coords(coords) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _is_connected(coords) -> bool:
    cells = {tuple(map(int, cell)) for cell in coords}
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _patch_variants(patch: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(patch, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = (candidate.shape, candidate.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(candidate.copy())
    return variants


def _shape_variants(coords) -> list[tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox_of_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = 1
    variants = []
    seen = set()
    for patch in _patch_variants(mask):
        norm = _normalize_coords(np.argwhere(patch != 0))
        if norm in seen:
            continue
        seen.add(norm)
        variants.append(norm)
    return variants


SHAPE_VARIANTS = {name: _shape_variants(coords) for name, coords in SHAPE_CLASSES.items()}


def _coords_variants(coords) -> list[tuple[tuple[int, int], ...]]:
    return _shape_variants(coords)


def _coords_to_patch(match_coords, other_coords, match_color: int, other_color: int) -> np.ndarray:
    union = list(match_coords) + list(other_coords)
    r0, c0, r1, c1 = _bbox_of_coords(union)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in match_coords:
        patch[int(r - r0), int(c - c0)] = int(match_color)
    for r, c in other_coords:
        patch[int(r - r0), int(c - c0)] = int(other_color)
    return patch


def _parse_tiles(grid: np.ndarray) -> list[dict]:
    grey_comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
    )
    tiles = []
    for comp in grey_comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (TILE_SIZE, TILE_SIZE):
            continue
        patch = crop_bbox(grid, (r0, c0, r1, c1))
        cavity = [
            (rr, cc)
            for rr in range(1, TILE_SIZE - 1)
            for cc in range(1, TILE_SIZE - 1)
            if int(patch[rr, cc]) == Color.BLACK
        ]
        cavity_norm = _normalize_coords(cavity)
        tiles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "cavity": tuple(sorted((int(r), int(c)) for r, c in cavity)),
                "cavity_norm": cavity_norm,
                "cavity_min": None if not cavity else tuple(map(int, np.min(np.asarray(cavity), axis=0))),
            }
        )

    assert len(tiles) == 9, "expected a 3x3 grey tile array"
    row_starts = sorted({tile["bbox"][0] for tile in tiles})
    col_starts = sorted({tile["bbox"][1] for tile in tiles})
    assert len(row_starts) == 3 and len(col_starts) == 3
    for tile in tiles:
        tile["row"] = row_starts.index(tile["bbox"][0])
        tile["col"] = col_starts.index(tile["bbox"][1])
        tile["id"] = tile["row"] * ARRAY_SIDE + tile["col"]
    return sorted(tiles, key=lambda tile: tile["id"])


def _parse_sprites(grid: np.ndarray) -> list[dict]:
    sprite_grid = np.array(grid, dtype=int)
    sprite_grid[sprite_grid == Color.GREY] = Color.BLACK
    components = find_components(
        sprite_grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
    )
    sprites = []
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(sprite_grid, (r0, c0, r1, c1))
        colors = sorted(int(color) for color in np.unique(patch) if int(color) != Color.BLACK)
        if len(colors) != 2:
            continue
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "area": int(comp["area"]),
                "colors": colors,
            }
        )
    return sorted(sprites, key=lambda sprite: (-sprite["area"], sprite["bbox"][1], sprite["bbox"][0]))


def _match_candidates(patch: np.ndarray, tiles: list[dict]) -> list[dict]:
    candidates = []
    for variant in _patch_variants(patch):
        colors = sorted(int(color) for color in np.unique(variant) if int(color) != Color.BLACK)
        for color in colors:
            shape = _normalize_coords(np.argwhere(variant == int(color)))
            for tile in tiles:
                if tile["cavity_norm"] and shape == tile["cavity_norm"]:
                    candidates.append(
                        {
                            "tile_id": tile["id"],
                            "patch": variant,
                            "match_color": int(color),
                        }
                    )
    return candidates


def _paint_docked_patch(
    out: np.ndarray,
    tile: dict,
    patch: np.ndarray,
    match_color: int,
    swap_colors: bool,
) -> None:
    colors = sorted(int(color) for color in np.unique(patch) if int(color) != Color.BLACK)
    assert len(colors) == 2
    remap = {colors[0]: colors[0], colors[1]: colors[1]}
    if swap_colors:
        remap = {colors[0]: colors[1], colors[1]: colors[0]}

    match_coords = np.argwhere(patch == int(match_color))
    mr0, mc0 = map(int, match_coords.min(axis=0))
    cavity_r, cavity_c = tile["cavity_min"]
    tr0, tc0, tr1, tc1 = tile["bbox"]
    place_r = int(tr0 + cavity_r - mr0)
    place_c = int(tc0 + cavity_c - mc0)

    for rr, cc in np.argwhere(patch != Color.BLACK):
        gr = int(place_r + rr)
        gc = int(place_c + cc)
        if tr0 <= gr <= tr1 and tc0 <= gc <= tc1:
            out[gr, gc] = remap[int(patch[rr, cc])]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each scattered two-color sprite into the unique cavity-matching tile, then swap colors on odd array coordinates.

    Input structure:
    - The upper part contains a 3x3 array of separated grey tiles on a black background.
    - Some tiles are solid grey; the others contain a connected black cavity inside the tile interior.
    - The lower part contains several disconnected 4-connected sprites, each made of exactly two non-grey colors.

    Rule:
    - For each sprite, search all rotations and horizontal-flip variants and find the unique tile whose cavity shape exactly matches one sprite color-region shape.
    - Dock the whole transformed sprite into that tile by aligning the matching color region to the cavity; clip any non-matching pixels that would leave the tile bbox.
    - If the matched tile is at array coordinate (row, col) with odd row+col parity, swap the sprite's two colors after docking.
    - Erase all original lower-half sprites to black and leave unmatched or cavity-free tiles unchanged.

    Key insight:
    - Shape matching is done on one color region of each sprite, but the final color arrangement is controlled separately by the tile's checkerboard parity in the 3x3 array.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sprite_mask = (g != Color.BLACK) & (g != Color.GREY)
    out[sprite_mask] = Color.BLACK

    tiles = _parse_tiles(g)
    active_tiles = [tile for tile in tiles if tile["cavity_norm"]]
    sprites = _parse_sprites(g)
    used_tiles = set()

    for sprite in sprites:
        remaining = [tile for tile in active_tiles if tile["id"] not in used_tiles]
        candidates = _match_candidates(sprite["patch"], remaining)
        assert len(candidates) == 1, "each sprite must have a unique cavity match"
        chosen = candidates[0]
        tile = next(tile for tile in remaining if tile["id"] == chosen["tile_id"])
        used_tiles.add(tile["id"])
        _paint_docked_patch(
            out,
            tile,
            chosen["patch"],
            chosen["match_color"],
            swap_colors=((tile["row"] + tile["col"]) % 2 == 1),
        )

    return out


def _visible_other_pixels(tile: dict, patch: np.ndarray, match_color: int) -> int:
    match_coords = np.argwhere(patch == int(match_color))
    mr0, mc0 = map(int, match_coords.min(axis=0))
    cavity_r, cavity_c = tile["cavity_min"]
    tr0, tc0, tr1, tc1 = tile["bbox"]
    place_r = int(tr0 + cavity_r - mr0)
    place_c = int(tc0 + cavity_c - mc0)
    count = 0
    for rr, cc in np.argwhere(patch != Color.BLACK):
        gr = int(place_r + rr)
        gc = int(place_c + cc)
        if not (tr0 <= gr <= tr1 and tc0 <= gc <= tc1):
            continue
        if int(patch[rr, cc]) != int(match_color):
            count += 1
    return count


def _build_dock_patch(
    cavity_norm,
    match_color: int,
    other_color: int,
    other_size: int,
    prefer_outward: bool,
) -> np.ndarray | None:
    match_cells = {tuple(map(int, cell)) for cell in cavity_norm}
    match_bbox = _bbox_of_coords(match_cells)
    low, high = -2, 4

    def outside_score(cell: tuple[int, int]) -> int:
        r, c = cell
        score = 0
        if r < match_bbox[0]:
            score += match_bbox[0] - r
        if r > match_bbox[2]:
            score += r - match_bbox[2]
        if c < match_bbox[1]:
            score += match_bbox[1] - c
        if c > match_bbox[3]:
            score += c - match_bbox[3]
        return score

    for _ in range(200):
        other = set()
        first = [
            (nr, nc)
            for r, c in match_cells
            for nr, nc in _neighbors4(r, c)
            if low <= nr <= high and low <= nc <= high and (nr, nc) not in match_cells
        ]
        first = list(dict.fromkeys(first))
        if not first:
            continue
        if prefer_outward:
            first.sort(key=outside_score, reverse=True)
            top = [cell for cell in first if outside_score(cell) == outside_score(first[0])]
            other.add(random.choice(top))
        else:
            other.add(random.choice(first))

        while len(other) < other_size:
            frontier = set()
            for r, c in match_cells | other:
                for nr, nc in _neighbors4(r, c):
                    if low <= nr <= high and low <= nc <= high and (nr, nc) not in match_cells and (nr, nc) not in other:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier = sorted(frontier, key=outside_score, reverse=prefer_outward)
            if prefer_outward:
                best = outside_score(frontier[0])
                picks = [cell for cell in frontier if outside_score(cell) >= max(0, best - 1)]
                other.add(random.choice(picks))
            else:
                other.add(random.choice(frontier))

        if len(other) != other_size:
            continue
        union = match_cells | other
        if not _is_connected(union):
            continue
        r0, c0, r1, c1 = _bbox_of_coords(union)
        if (r1 - r0 + 1) > TILE_SIZE or (c1 - c0 + 1) > TILE_SIZE:
            continue
        return _coords_to_patch(match_cells, other, match_color, other_color)
    return None


def _render_expected(grid: np.ndarray, tiles: list[dict], sprites: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int)
    out[(out != Color.BLACK) & (out != Color.GREY)] = Color.BLACK
    for sprite in sprites:
        tile = next(tile for tile in tiles if tile["id"] == sprite["tile_id"])
        _paint_docked_patch(
            out,
            tile,
            sprite["dock_patch"],
            sprite["match_color"],
            swap_colors=((tile["row"] + tile["col"]) % 2 == 1),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(25, 30)
    width = random.randint(22, 30)
    top = random.randint(0, 1)
    left = random.randint(0, width - ARRAY_SPAN)
    lower_gap = random.randint(1, 2)
    lower_top = top + ARRAY_SPAN + lower_gap
    if height - lower_top < 6:
        return None

    grid = np.zeros((height, width), dtype=int)
    tiles = []
    for row in range(ARRAY_SIDE):
        for col in range(ARRAY_SIDE):
            r0 = top + row * (TILE_SIZE + ARRAY_GAP)
            c0 = left + col * (TILE_SIZE + ARRAY_GAP)
            patch = np.full((TILE_SIZE, TILE_SIZE), Color.GREY, dtype=int)
            paste(grid, patch, r0, c0)
            tiles.append(
                {
                    "bbox": (r0, c0, r0 + TILE_SIZE - 1, c0 + TILE_SIZE - 1),
                    "row": row,
                    "col": col,
                    "id": row * ARRAY_SIDE + col,
                    "cavity_norm": (),
                    "cavity_min": None,
                }
            )

    active_count = len(ASYMMETRIC_SHAPES)
    while True:
        active_ids = sorted(random.sample(range(9), active_count))
        parities = {(tile_id // ARRAY_SIDE + tile_id % ARRAY_SIDE) % 2 for tile_id in active_ids}
        if parities == {0, 1}:
            break

    shape_names = random.sample(ASYMMETRIC_SHAPES, active_count)
    tiles_by_id = {tile["id"]: tile for tile in tiles}
    for tile_id, shape_name in zip(active_ids, shape_names):
        tile = tiles_by_id[tile_id]
        shape = random.choice(SHAPE_VARIANTS[shape_name])
        sr0, sc0, sr1, sc1 = _bbox_of_coords(shape)
        shift_r = INNER_MIN + random.randint(0, (INNER_MAX - INNER_MIN + 1) - (sr1 - sr0 + 1))
        shift_c = INNER_MIN + random.randint(0, (INNER_MAX - INNER_MIN + 1) - (sc1 - sc0 + 1))
        concrete = tuple(sorted((int(r + shift_r), int(c + shift_c)) for r, c in shape))
        tile["cavity_norm"] = _normalize_coords(concrete)
        tile["cavity_min"] = tuple(map(int, np.min(np.asarray(concrete), axis=0)))
        tr0, tc0, tr1, tc1 = tile["bbox"]
        for rr, cc in concrete:
            grid[tr0 + rr, tc0 + cc] = Color.BLACK

    area_hist = {}
    for tile in tiles:
        area = len(tile["cavity_norm"])
        if area:
            area_hist[area] = area_hist.get(area, 0) + 1
    if not any(count >= 2 for count in area_hist.values()):
        return None

    base_pairs = [tuple(random.sample(SPRITE_COLORS, 2)) for _ in range(3)]
    sprite_specs = []
    active_norms = {tiles_by_id[idx]["cavity_norm"] for idx in active_ids}
    for tile_id in active_ids:
        tile = tiles_by_id[tile_id]
        cavity_area = len(tile["cavity_norm"])
        color_a, color_b = random.choice(base_pairs)
        if random.random() < 0.5:
            match_color, other_color = color_a, color_b
        else:
            match_color, other_color = color_b, color_a

        options = [size for size in (2, 3, 4) if cavity_area + size <= 6]
        if cavity_area in options and random.random() < 0.45:
            other_size = cavity_area
        else:
            other_size = random.choice(options)

        dock_patch = None
        source_patch = None
        chosen_match_color = None
        for _ in range(250):
            candidate = _build_dock_patch(
                tile["cavity_norm"],
                match_color,
                other_color,
                other_size,
                prefer_outward=(random.random() < 0.5),
            )
            if candidate is None:
                continue
            other_shape = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(candidate == int(other_color))))
            if any(variant in active_norms for variant in _coords_variants(other_shape)):
                continue
            if _visible_other_pixels(tile, candidate, match_color) == 0:
                continue
            variants = _patch_variants(candidate)
            non_identity = [variant for variant in variants if not np.array_equal(variant, candidate)]
            if non_identity and random.random() < 0.75:
                scrambled = random.choice(non_identity)
            else:
                scrambled = random.choice(variants)

            matches = _match_candidates(scrambled, [tiles_by_id[idx] for idx in active_ids])
            if len(matches) != 1 or matches[0]["tile_id"] != tile_id:
                continue
            dock_patch = candidate
            source_patch = scrambled
            chosen_match_color = match_color
            break
        if dock_patch is None or source_patch is None:
            return None

        sprite_specs.append(
            {
                "tile_id": tile_id,
                "dock_patch": dock_patch,
                "source_patch": source_patch,
                "match_color": chosen_match_color,
            }
        )

    lower = grid[lower_top:, :]
    for spec in random.sample(sprite_specs, len(sprite_specs)):
        try:
            rr, cc = random_free_location(
                lower,
                spec["source_patch"],
                transparent=Color.BLACK,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(lower, spec["source_patch"], rr, cc, transparent=Color.BLACK)
        h, w = spec["source_patch"].shape
        spec["bbox"] = (lower_top + rr, cc, lower_top + rr + h - 1, cc + w - 1)

    parsed_tiles = _parse_tiles(grid)
    parsed_active = [tile for tile in parsed_tiles if tile["cavity_norm"]]
    parsed_sprites = _parse_sprites(grid)
    if len(parsed_active) != len(active_ids) or len(parsed_sprites) != len(sprite_specs):
        return None

    for sprite in parsed_sprites:
        matches = _match_candidates(sprite["patch"], parsed_active)
        if len(matches) != 1:
            return None

    expected = _render_expected(grid, tiles, sprite_specs)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
