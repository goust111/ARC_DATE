from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 500


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def reflect_bbox(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0, width - 1 - c1, r1, width - 1 - c0)


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def dihedral_variants(tile: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    cur = tile
    for _ in range(4):
        variants.append(cur)
        variants.append(np.fliplr(cur))
        cur = np.rot90(cur)
    return variants


def dihedral_unique(tile: np.ndarray) -> bool:
    sigs = {variant.tobytes() for variant in dihedral_variants(tile)}
    return len(sigs) == 8


def split_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_has_fg = np.any(grid != BLACK, axis=1)
    fg_rows = np.flatnonzero(row_has_fg)
    assert fg_rows.size > 0, "expected foreground"

    top_start = int(fg_rows[0])
    sep_row = None
    for r in range(top_start, grid.shape[0]):
        if not row_has_fg[r]:
            sep_row = r
            break
    assert sep_row is not None, "expected a background separator row"

    top_band = grid[top_start:sep_row]
    top_box = bbox(top_band != BLACK)
    assert top_box is not None, "expected a nonempty top pattern"
    pattern = crop_bbox(top_band, top_box)

    lower_rows = np.flatnonzero(row_has_fg[sep_row + 1 :])
    assert lower_rows.size > 0, "expected candidate patches below the separator"
    lower_start = sep_row + 1 + int(lower_rows[0])
    lower_end = sep_row + 1 + int(lower_rows[-1])
    lower_band = grid[lower_start : lower_end + 1]
    lower_box = bbox(lower_band != BLACK)
    assert lower_box is not None, "expected nonempty candidate region"
    candidates_region = crop_bbox(lower_band, lower_box)
    return pattern, candidates_region


def find_holes(pattern: np.ndarray) -> tuple[list[tuple[int, int, int, int]], int]:
    h, w = pattern.shape
    hole_boxes: list[tuple[int, int, int, int]] = []
    sizes: list[int] = []
    for comp in components_mask(pattern == BLACK):
        box = bbox_coords(comp)
        r0, c0, r1, c1 = box
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if side_r != side_c:
            continue
        if comp.shape[0] != side_r * side_c:
            continue
        hole_boxes.append(box)
        sizes.append(side_r)
    assert len(hole_boxes) == 3, "expected exactly three square holes"
    assert len(set(sizes)) == 1, "all holes must share one size"
    return sorted(hole_boxes), int(sizes[0])


def classify_holes(
    holes: list[tuple[int, int, int, int]], width: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    hole_set = set(holes)
    singleton = None
    pair: list[tuple[int, int, int, int]] = []
    for box in holes:
        mirrored = reflect_bbox(box, width)
        if mirrored in hole_set and mirrored != box:
            pair.append(box)
        else:
            singleton = box
    assert singleton is not None, "expected one hole with an intact mirror donor"
    assert len(pair) == 2, "expected one mirrored hole pair"
    pair.sort(key=lambda box: box[1])
    return singleton, pair[0], pair[1]


def candidate_tiles(region: np.ndarray, side: int) -> list[np.ndarray]:
    tiles: list[np.ndarray] = []
    for comp in components_mask(region != BLACK):
        box = bbox_coords(comp)
        tile = crop_bbox(region, box)
        assert tile.shape == (side, side), "candidate patch size must match hole size"
        assert np.all(tile != BLACK), "candidate patches must be fully colored"
        tiles.append(tile)
    assert len(tiles) >= 2, "expected multiple candidate patches"
    return tiles


def fill_box(dst: np.ndarray, box: tuple[int, int, int, int], tile: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    assert tile.shape == (r1 - r0 + 1, c1 - c0 + 1)
    dst[r0 : r1 + 1, c0 : c1 + 1] = tile


def box_boundary_matches(canvas: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    tile = canvas[r0 : r1 + 1, c0 : c1 + 1]
    if not np.array_equal(tile[0, :], canvas[r0 - 1, c0 : c1 + 1]):
        return False
    if not np.array_equal(tile[-1, :], canvas[r1 + 1, c0 : c1 + 1]):
        return False
    if not np.array_equal(tile[:, 0], canvas[r0 : r1 + 1, c0 - 1]):
        return False
    if not np.array_equal(tile[:, -1], canvas[r0 : r1 + 1, c1 + 1]):
        return False
    return True


def _passing_trials(
    pattern: np.ndarray,
    tiles: list[np.ndarray],
    left_box: tuple[int, int, int, int],
    right_box: tuple[int, int, int, int],
) -> list[dict]:
    passed: list[dict] = []
    for tile_idx, tile in enumerate(tiles):
        for rot in range(4):
            oriented = np.rot90(tile, rot)
            for anchor_left in (True, False):
                trial = pattern.copy()
                if anchor_left:
                    left_tile = oriented
                    right_tile = np.fliplr(oriented)
                else:
                    right_tile = oriented
                    left_tile = np.fliplr(oriented)
                fill_box(trial, left_box, left_tile)
                fill_box(trial, right_box, right_tile)
                if np.any(trial == BLACK):
                    continue
                if not box_boundary_matches(trial, left_box):
                    continue
                if not box_boundary_matches(trial, right_box):
                    continue
                if np.array_equal(trial, np.fliplr(trial)):
                    passed.append(
                        {
                            "tile_idx": tile_idx,
                            "rot": rot,
                            "anchor_left": anchor_left,
                            "left_tile": left_tile.copy(),
                            "output": trial,
                        }
                    )
    return passed


def solve_instance(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    pattern, region = split_regions(grid)
    holes, side = find_holes(pattern)
    singleton, left_box, right_box = classify_holes(holes, pattern.shape[1])
    tiles = candidate_tiles(region, side)

    repaired = pattern.copy()
    fill_box(repaired, singleton, np.fliplr(pattern)[singleton[0] : singleton[2] + 1, singleton[1] : singleton[3] + 1])
    assert np.all(repaired[singleton[0] : singleton[2] + 1, singleton[1] : singleton[3] + 1] != BLACK)

    passed = _passing_trials(repaired, tiles, left_box, right_box)
    assert passed, "expected one candidate/orientation to restore symmetry"

    unique_outputs: dict[bytes, np.ndarray] = {}
    for item in passed:
        out = item["output"]
        unique_outputs[out.tobytes() + str(out.shape).encode()] = out
    assert len(unique_outputs) == 1, "expected a unique repaired output"

    info = {
        "pattern": pattern,
        "singleton": singleton,
        "left_box": left_box,
        "right_box": right_box,
        "tiles": tiles,
        "passed": passed,
    }
    return next(iter(unique_outputs.values())).copy(), info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the three black holes in the upper symmetric picture and discard the lower candidate strip.

    Input structure:
    - The upper object is a fully colored rectangular picture that is symmetric under
      left-right reflection and contains exactly three same-size black square holes.
    - One hole has an intact reflected counterpart, while the other two holes are
      themselves mirror positions of one another.
    - Below a black separator band sit several same-size candidate patches, each
      already rotated by some multiple of ninety degrees.

    Rule:
    - Crop the upper picture region and locate its left-right mirror axis from that
      picture, not from the whole canvas.
    - Refill the lone hole by copying its intact reflected counterpart.
    - For the mirrored hole pair, try every candidate patch in every quarter-turn
      orientation, place it in one hole and its horizontal mirror in the other, and
      keep the unique trial whose patch edges fit the surrounding picture and whose
      completed picture restores exact left-right symmetry everywhere.

    Key insight:
    - The three holes split into two roles: only one can be solved by direct mirror
      copying, while the mirrored pair needs the lower candidates, hole-boundary
      fit, and a global symmetry check to identify the one viable patch orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out, _ = solve_instance(g)
    return out


def make_half_pattern(height: int, half_width: int, palette: list[int]) -> np.ndarray:
    left = np.full((height, half_width), random.choice(palette), dtype=int)
    mode = random.randrange(3)
    for _ in range(random.randint(6, 11)):
        color = random.choice(palette)
        r0 = random.randint(0, height - 1)
        r1 = random.randint(r0, height - 1)
        c0 = random.randint(0, half_width - 1)
        c1 = random.randint(c0, half_width - 1)
        left[r0 : r1 + 1, c0 : c1 + 1] = color
    if mode == 0:
        for _ in range(random.randint(2, 4)):
            row = random.randrange(height)
            left[row, :] = random.choice(palette)
    elif mode == 1:
        for _ in range(random.randint(2, 4)):
            col = random.randrange(half_width)
            left[:, col] = random.choice(palette)
    else:
        for _ in range(random.randint(3, 6)):
            r0 = random.randrange(height)
            c0 = random.randrange(half_width)
            bh = random.randint(1, min(3, height - r0))
            bw = random.randint(1, min(3, half_width - c0))
            left[r0 : r0 + bh, c0 : c0 + bw] = random.choice(palette)
    return left


def build_pattern(height: int, width: int, palette: list[int]) -> np.ndarray:
    half_width = width // 2
    left = make_half_pattern(height, half_width, palette)
    if width % 2:
        center = np.array([random.choice(palette) for _ in range(height)], dtype=int)[:, None]
        pattern = np.concatenate([left, center, np.fliplr(left)], axis=1)
    else:
        pattern = np.concatenate([left, np.fliplr(left)], axis=1)
    return pattern.astype(int)


def pair_positions(pattern: np.ndarray, side: int) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    h, w = pattern.shape
    positions: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for r0 in range(1, h - side):
        if r0 + side > h - 1:
            continue
        for c0 in range(1, w - side):
            if c0 + side > w - 1:
                continue
            mirror_c0 = w - side - c0
            if c0 >= mirror_c0:
                continue
            left_box = (r0, c0, r0 + side - 1, c0 + side - 1)
            right_box = (r0, mirror_c0, r0 + side - 1, mirror_c0 + side - 1)
            if boxes_too_close(left_box, right_box, gap=1):
                continue
            patch = crop_bbox(pattern, left_box)
            if np.unique(patch).size < 2:
                continue
            if not box_boundary_matches(pattern, left_box):
                continue
            if not dihedral_unique(patch):
                continue
            positions.append((left_box, right_box))
    random.shuffle(positions)
    return positions


def singleton_positions(
    pattern: np.ndarray,
    side: int,
    left_box: tuple[int, int, int, int],
    right_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = pattern.shape
    blocked = [left_box, right_box]
    options: list[tuple[int, int, int, int]] = []
    for r0 in range(1, h - side):
        if r0 + side > h - 1:
            continue
        for c0 in range(1, w - side):
            if c0 + side > w - 1:
                continue
            box = (r0, c0, r0 + side - 1, c0 + side - 1)
            donor = reflect_bbox(box, w)
            if donor == box:
                continue
            if boxes_overlap(box, donor):
                continue
            if any(boxes_too_close(box, other, gap=1) for other in blocked):
                continue
            options.append(box)
    random.shuffle(options)
    return options


def mutate_patch(tile: np.ndarray, palette: list[int], changes: int) -> np.ndarray:
    out = tile.copy()
    cells = random.sample([(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])], changes)
    for r, c in cells:
        choices = [color for color in palette if color != int(out[r, c])]
        out[r, c] = random.choice(choices)
    return out


def mutate_boundary_patch(tile: np.ndarray, palette: list[int], changes: int) -> np.ndarray:
    out = tile.copy()
    side = tile.shape[0]
    boundary = []
    for r in range(side):
        for c in range(side):
            if r in (0, side - 1) or c in (0, side - 1):
                boundary.append((r, c))
    cells = random.sample(boundary, changes)
    for r, c in cells:
        choices = [color for color in palette if color != int(out[r, c])]
        out[r, c] = random.choice(choices)
    return out


def random_patch(side: int, palette: list[int]) -> np.ndarray:
    tile = np.array([[random.choice(palette) for _ in range(side)] for _ in range(side)], dtype=int)
    return tile


def min_rotation_distance(tile: np.ndarray, target: np.ndarray) -> int:
    return min(int(np.sum(np.rot90(tile, rot) != target)) for rot in range(4))


def min_dihedral_distance(tile: np.ndarray, target: np.ndarray) -> int:
    return min(int(np.sum(variant != target)) for variant in dihedral_variants(tile))


def semantic_checks(grid: np.ndarray) -> bool:
    try:
        output, info = solve_instance(grid)
    except AssertionError:
        return False
    pattern = info["pattern"]
    passed = info["passed"]
    tiles = info["tiles"]
    left_box = info["left_box"]
    correct_left = passed[0]["left_tile"]

    if len(passed) != 1:
        return False
    if not np.array_equal(output, np.fliplr(output)):
        return False
    if np.any(output == BLACK):
        return False
    if output.shape != pattern.shape:
        return False
    near_miss = False
    for idx, tile in enumerate(tiles):
        if idx == passed[0]["tile_idx"]:
            continue
        dist = min_rotation_distance(tile, correct_left)
        if dist in (1, 2):
            near_miss = True
            break
    if not near_miss:
        return False
    lr0, lc0, lr1, lc1 = left_box
    if not np.array_equal(output[lr0 : lr1 + 1, lc0 : lc1 + 1], correct_left):
        return False
    return True


def build_candidate_tiles(correct_patch: np.ndarray, palette: list[int], count: int) -> list[np.ndarray]:
    tiles: list[np.ndarray] = [np.rot90(correct_patch, random.randrange(4))]

    def try_add(source: np.ndarray) -> bool:
        shown = np.rot90(source, random.randrange(4))
        if min_dihedral_distance(shown, correct_patch) == 0:
            return False
        if any(min_dihedral_distance(shown, existing) == 0 for existing in tiles):
            return False
        tiles.append(shown)
        return True

    while len(tiles) < min(count, 3):
        needed_changes = 1 if len(tiles) == 1 else 2
        try_add(mutate_boundary_patch(correct_patch, palette, needed_changes))
    while len(tiles) < count:
        if random.random() < 0.5:
            candidate = mutate_patch(correct_patch, palette, random.choice((1, 2)))
        else:
            candidate = random_patch(correct_patch.shape[0], palette)
        try_add(candidate)
    random.shuffle(tiles)
    return tiles


def assemble_grid(pattern: np.ndarray, tiles: list[np.ndarray]) -> np.ndarray | None:
    side = tiles[0].shape[0]
    pattern_h, pattern_w = pattern.shape
    sep_rows = random.randint(1, 3)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    tile_gaps = [random.randint(1, 3) for _ in range(len(tiles) - 1)]
    tile_span = len(tiles) * side + sum(tile_gaps)
    candidate_left = random.randint(1, 3)
    candidate_width = candidate_left + tile_span + random.randint(1, 3)
    input_w = max(pattern_w + random.randint(2, 10), candidate_width)
    if input_w > 30:
        return None

    pattern_left = random.randint(1, input_w - pattern_w - 1)
    if pattern_left + (pattern_w - 1) / 2 == (input_w - 1) / 2:
        if pattern_left < input_w - pattern_w - 1:
            pattern_left += 1
        elif pattern_left > 1:
            pattern_left -= 1
        else:
            return None

    cand_top = top_margin + pattern_h + sep_rows
    input_h = cand_top + side + bottom_margin
    if input_h > 30:
        return None

    grid = np.zeros((input_h, input_w), dtype=int)
    grid[top_margin : top_margin + pattern_h, pattern_left : pattern_left + pattern_w] = pattern

    c = candidate_left
    for idx, tile in enumerate(tiles):
        grid[cand_top : cand_top + side, c : c + side] = tile
        c += side
        if idx < len(tile_gaps):
            c += tile_gaps[idx]
    return grid


def build_instance() -> np.ndarray | None:
    palette = random.sample(PALETTE, random.randint(4, 6))
    height = random.randint(8, 14)
    width_parity = random.choice((0, 1))
    width = random.randrange(10 + width_parity, 19, 2)
    pattern = build_pattern(height, width, palette)

    side_choices = list(range(2, min(4, height - 2, width // 3) + 1))
    random.shuffle(side_choices)
    for side in side_choices:
        pairs = pair_positions(pattern, side)
        if not pairs:
            continue
        for left_box, right_box in pairs[:20]:
            singles = singleton_positions(pattern, side, left_box, right_box)
            if not singles:
                continue
            singleton = singles[0]
            correct_patch = crop_bbox(pattern, left_box)
            tiles = build_candidate_tiles(correct_patch, palette, random.randint(3, 5))

            holed = pattern.copy()
            fill_box(holed, singleton, np.zeros((side, side), dtype=int))
            fill_box(holed, left_box, np.zeros((side, side), dtype=int))
            fill_box(holed, right_box, np.zeros((side, side), dtype=int))

            grid = assemble_grid(holed, tiles)
            if grid is None:
                continue
            if semantic_checks(grid):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid three-hole symmetry-repair input")
