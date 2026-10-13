from __future__ import annotations

import itertools
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, bbox, crop_bbox, find_components, object_interior, paste
except ImportError:
    toolbox = Path("/home/lixinyu/snap/new/solvers")
    if str(toolbox) not in sys.path:
        sys.path.append(str(toolbox))
    from arc_utils import Color, bbox, crop_bbox, find_components, object_interior, paste


BG = Color.BLACK
GREY = Color.GREY
PIECE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
PAIR_AREA_CHOICES = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
GEN_MAX_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each gray block hole with the unique two-piece colored assembly and erase the used pieces.

    Input structure:
    - The background is black.
    - Several gray connected blocks each contain one connected black hole.
    - Away from the gray blocks lie several solid monochrome color pieces with distinct colors.

    Rule:
    - For every hole, find the unique pair of scattered pieces whose rotated and/or mirrored copies can tile that hole exactly.
    - Paint the hole cells with the colors of the two matching pieces according to the unique tiling.
    - Erase every used piece from its original location, while leaving unused decoys and gray blocks unchanged.

    Key insight:
    - The hole is never a single-piece match; only a unique two-piece geometric assembly explains each hole, and decoys are rejected by exact tiling.
    """
    g = np.asarray(grid, dtype=int)
    holes = _extract_holes(g)
    pieces = _extract_pieces(g)

    candidates = [_hole_candidates(hole, pieces) for hole in holes]
    assignment = _unique_assignment(candidates)

    out = g.copy()
    used_piece_ids: set[int] = set()
    for hole_index, candidate in assignment.items():
        hole = holes[hole_index]
        piece_a, piece_b = candidate["piece_ids"]
        used_piece_ids.update((piece_a, piece_b))
        hole_r0, hole_c0 = hole["origin"]
        color_a = pieces[piece_a]["color"]
        color_b = pieces[piece_b]["color"]
        for r, c in candidate["regions"][0]:
            out[hole_r0 + r, hole_c0 + c] = color_a
        for r, c in candidate["regions"][1]:
            out[hole_r0 + r, hole_c0 + c] = color_b

    for piece_id in used_piece_ids:
        rr, cc = pieces[piece_id]["coords"].T
        out[rr, cc] = BG

    return out


def _extract_holes(grid: np.ndarray) -> list[dict]:
    holes = []
    gray_objects = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )

    for obj in gray_objects:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        sprite = crop_bbox(grid, (r0, c0, r1, c1))
        hole_mask = (sprite == BG) & object_interior(sprite, background=BG)
        hole_components = find_components(
            hole_mask.astype(int),
            bg=0,
            connectivity=4,
            monochromatic=False,
            return_format="dict",
        )
        assert len(hole_components) == 1, "each gray block must contain exactly one connected hole"
        hb = bbox(hole_mask)
        assert hb is not None
        hr0, hc0, hr1, hc1 = hb
        hole_crop = crop_bbox(hole_mask.astype(int), hb).astype(bool)
        hole_shape = _shape_from_mask(hole_crop)
        holes.append(
            {
                "origin": (r0 + hr0, c0 + hc0),
                "shape": hole_shape,
                "area": len(hole_shape),
            }
        )

    holes.sort(key=lambda hole: hole["origin"])
    return holes


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces = []
    components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in components:
        color = int(comp["color"])
        if color in (BG, GREY):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        sprite = crop_bbox(grid, (r0, c0, r1, c1))
        mask = sprite != BG
        pieces.append(
            {
                "color": color,
                "coords": comp["coords"],
                "shape": _shape_from_mask(mask),
                "area": int(comp["area"]),
                "bbox": (r0, c0, r1, c1),
            }
        )

    pieces.sort(key=lambda piece: piece["bbox"])
    return pieces


def _hole_candidates(hole: dict, pieces: list[dict]) -> list[dict]:
    candidates = []
    for i, j in itertools.combinations(range(len(pieces)), 2):
        if pieces[i]["area"] + pieces[j]["area"] != hole["area"]:
            continue
        tilings = _tilings(hole["shape"], pieces[i]["shape"], pieces[j]["shape"])
        for region_a, region_b in tilings:
            candidates.append(
                {
                    "piece_ids": (i, j),
                    "regions": (tuple(sorted(region_a)), tuple(sorted(region_b))),
                }
            )
    assert candidates, "every hole must have at least one two-piece tiling candidate"
    return candidates


def _unique_assignment(candidates_by_hole: list[list[dict]]) -> dict[int, dict]:
    order = sorted(range(len(candidates_by_hole)), key=lambda idx: len(candidates_by_hole[idx]))
    solutions: list[dict[int, dict]] = []

    def dfs(step: int, used: set[int], chosen: dict[int, dict]) -> None:
        if len(solutions) > 1:
            return
        if step == len(order):
            solutions.append(chosen.copy())
            return

        hole_index = order[step]
        for candidate in candidates_by_hole[hole_index]:
            i, j = candidate["piece_ids"]
            if i in used or j in used:
                continue
            chosen[hole_index] = candidate
            used.update((i, j))
            dfs(step + 1, used, chosen)
            used.remove(i)
            used.remove(j)
            del chosen[hole_index]

    dfs(0, set(), {})
    assert len(solutions) == 1, "expected exactly one global piece-to-hole assembly"
    return solutions[0]


def _tilings(hole_shape: tuple[tuple[int, int], ...], shape_a: tuple[tuple[int, int], ...], shape_b: tuple[tuple[int, int], ...]) -> list[tuple[frozenset, frozenset]]:
    hole_cells = frozenset(hole_shape)
    hole_h, hole_w = _shape_dims(hole_shape)
    placements_a = _placements_within(hole_cells, hole_h, hole_w, shape_a)
    placements_b = _placements_within(hole_cells, hole_h, hole_w, shape_b)
    placements_b_lookup = set(placements_b)

    solutions = []
    seen = set()
    for placed_a in placements_a:
        placed_b = frozenset(hole_cells.difference(placed_a))
        if placed_b not in placements_b_lookup:
            continue
        key = (tuple(sorted(placed_a)), tuple(sorted(placed_b)))
        if key in seen:
            continue
        seen.add(key)
        solutions.append((placed_a, placed_b))
    return solutions


def _placements_within(
    hole_cells: frozenset[tuple[int, int]],
    hole_h: int,
    hole_w: int,
    shape: tuple[tuple[int, int], ...],
) -> list[frozenset[tuple[int, int]]]:
    placements = set()
    for variant in _transform_variants(shape):
        var_h, var_w = _shape_dims(variant)
        if var_h > hole_h or var_w > hole_w:
            continue
        for dr in range(hole_h - var_h + 1):
            for dc in range(hole_w - var_w + 1):
                placed = frozenset((r + dr, c + dc) for r, c in variant)
                if placed.issubset(hole_cells):
                    placements.add(placed)
    return sorted(placements, key=lambda cells: tuple(sorted(cells)))


def _normalize_shape(cells) -> tuple[tuple[int, int], ...]:
    points = list(cells)
    assert points
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in points))


def _shape_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _normalize_shape((tuple(map(int, coord)) for coord in coords))


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


@lru_cache(maxsize=None)
def _transform_variants(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    mask = _shape_to_mask(shape)
    variants = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            variants.add(_shape_from_mask(np.rot90(base, k)))
    return tuple(sorted(variants))


def _canonical_free(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_transform_variants(shape))


def _connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


@lru_cache(maxsize=None)
def _shape_bank(max_area: int = 8) -> dict[int, tuple[tuple[tuple[int, int], ...], ...]]:
    by_area: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for area in range(1, max_area):
        grown = set()
        for shape in by_area[area]:
            cell_set = set(shape)
            frontier = set()
            for r, c in cell_set:
                frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
            frontier.difference_update(cell_set)
            for cell in frontier:
                grown.add(_canonical_free(_normalize_shape(cell_set | {cell})))
        by_area[area + 1] = grown
    return {area: tuple(sorted(shapes)) for area, shapes in by_area.items()}


def _random_shape(
    area: int,
    *,
    min_h: int = 1,
    min_w: int = 1,
    max_h: int = 5,
    max_w: int = 5,
    exclude_canonical: set[tuple[tuple[int, int], ...]] | None = None,
) -> tuple[tuple[int, int], ...] | None:
    exclude_canonical = exclude_canonical or set()
    candidates = []
    for shape in _shape_bank(max(area, 8))[area]:
        h, w = _shape_dims(shape)
        if min_h <= h <= max_h and min_w <= w <= max_w and _canonical_free(shape) not in exclude_canonical:
            candidates.append(shape)
    if not candidates:
        return None
    return random.choice(candidates)


def _compose_hole(shape_a: tuple[tuple[int, int], ...], shape_b: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], tuple[frozenset, frozenset]] | None:
    for _ in range(200):
        variant_a = random.choice(_transform_variants(shape_a))
        variant_b = random.choice(_transform_variants(shape_b))
        ah, aw = _shape_dims(variant_a)
        bh, bw = _shape_dims(variant_b)
        canvas_h = random.randint(max(ah, bh), min(6, max(ah, bh) + 2))
        canvas_w = random.randint(max(aw, bw), min(6, max(aw, bw) + 2))

        placements_a = _canvas_placements(variant_a, canvas_h, canvas_w)
        placements_b = _canvas_placements(variant_b, canvas_h, canvas_w)
        random.shuffle(placements_a)
        random.shuffle(placements_b)

        for placed_a in placements_a:
            for placed_b in placements_b:
                if placed_a & placed_b:
                    continue
                union = set(placed_a | placed_b)
                if not _connected(union):
                    continue
                hole_shape = _normalize_shape(union)
                hole_h, hole_w = _shape_dims(hole_shape)
                if hole_h < 2 or hole_w < 2 or hole_h > 5 or hole_w > 5:
                    continue
                tilings = _tilings(hole_shape, shape_a, shape_b)
                if len(tilings) == 1:
                    return hole_shape, tilings[0]
    return None


def _canvas_placements(shape: tuple[tuple[int, int], ...], canvas_h: int, canvas_w: int) -> list[frozenset[tuple[int, int]]]:
    h, w = _shape_dims(shape)
    placements = []
    for dr in range(canvas_h - h + 1):
        for dc in range(canvas_w - w + 1):
            placements.append(frozenset((r + dr, c + dc) for r, c in shape))
    return placements


def _render_shape(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    mask = _shape_to_mask(shape)
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _build_gray_block(hole_shape: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, tuple[int, int]]:
    hole_mask = _shape_to_mask(hole_shape)
    hole_h, hole_w = hole_mask.shape
    top = random.randint(1, 2)
    left = random.randint(1, 2)
    bottom = random.randint(1, 2)
    right = random.randint(1, 2)
    block = np.full((hole_h + top + bottom, hole_w + left + right), GREY, dtype=int)
    window = block[top : top + hole_h, left : left + hole_w]
    window[hole_mask] = BG
    return block, (top, left)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_sprites(sprites: list[np.ndarray], grid_h: int, grid_w: int, gap: int = 1) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sprites)), key=lambda idx: sprites[idx].shape[0] * sprites[idx].shape[1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(sprites)
    boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        h, w = sprites[idx].shape
        if h > grid_h or w > grid_w:
            return None
        placed = False
        for _ in range(300):
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_bbox_overlap(box, other, gap) for other in boxes):
                continue
            placements[idx] = (r0, c0)
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return [tuple(pos) for pos in placements]


def _build_expected_output(inp: np.ndarray, hole_records: list[dict], piece_records: list[dict]) -> np.ndarray:
    out = inp.copy()
    used = set()
    for hole in hole_records:
        piece_a, piece_b = hole["piece_ids"]
        used.update((piece_a, piece_b))
        hole_r0, hole_c0 = hole["origin"]
        color_a = piece_records[piece_a]["color"]
        color_b = piece_records[piece_b]["color"]
        for r, c in hole["regions"][0]:
            out[hole_r0 + r, hole_c0 + c] = color_a
        for r, c in hole["regions"][1]:
            out[hole_r0 + r, hole_c0 + c] = color_b

    for piece_id in used:
        sprite = piece_records[piece_id]["sprite"]
        r0, c0 = piece_records[piece_id]["pos"]
        out[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]][sprite != BG] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    n_holes = random.randint(1, 3)
    extra_decoys = random.randint(0, 1 if n_holes >= 2 else 2)
    total_pieces = 2 * n_holes + 1 + extra_decoys
    if total_pieces > len(PIECE_COLORS):
        return None

    chosen_colors = random.sample(PIECE_COLORS, total_pieces)
    piece_records = []
    hole_records = []
    hole_shapes = []
    piece_index = 0

    for _ in range(n_holes):
        for _retry in range(80):
            area_a, area_b = random.choice(PAIR_AREA_CHOICES)
            shape_a = _random_shape(area_a, max_h=4, max_w=4)
            shape_b = _random_shape(area_b, max_h=4, max_w=4)
            if shape_a is None or shape_b is None:
                continue
            composed = _compose_hole(shape_a, shape_b)
            if composed is None:
                continue
            hole_shape, regions = composed
            block_sprite, hole_offset = _build_gray_block(hole_shape)

            color_a = chosen_colors[piece_index]
            color_b = chosen_colors[piece_index + 1]
            piece_sprite_a = _render_shape(random.choice(_transform_variants(shape_a)), color_a)
            piece_sprite_b = _render_shape(random.choice(_transform_variants(shape_b)), color_b)

            hole_records.append(
                {
                    "block_sprite": block_sprite,
                    "hole_offset": hole_offset,
                    "piece_ids": (piece_index, piece_index + 1),
                    "regions": regions,
                    "hole_shape": hole_shape,
                }
            )
            hole_shapes.append(hole_shape)
            piece_records.append({"color": color_a, "sprite": piece_sprite_a})
            piece_records.append({"color": color_b, "sprite": piece_sprite_b})
            piece_index += 2
            break
        else:
            return None

    target_hole = random.choice(hole_shapes)
    target_area = len(target_hole)
    target_canonical = _canonical_free(target_hole)
    decoy_shape = _random_shape(target_area, max_h=5, max_w=5, exclude_canonical={target_canonical})
    if decoy_shape is None:
        return None
    piece_records.append(
        {
            "color": chosen_colors[piece_index],
            "sprite": _render_shape(random.choice(_transform_variants(decoy_shape)), chosen_colors[piece_index]),
        }
    )
    piece_index += 1

    for _ in range(extra_decoys):
        area = random.choice([2, 3, 4, 5])
        decoy = _random_shape(area, max_h=4, max_w=4)
        if decoy is None:
            return None
        piece_records.append(
            {
                "color": chosen_colors[piece_index],
                "sprite": _render_shape(random.choice(_transform_variants(decoy)), chosen_colors[piece_index]),
            }
        )
        piece_index += 1

    sprites = [hole["block_sprite"] for hole in hole_records] + [piece["sprite"] for piece in piece_records]
    largest_h = max(sprite.shape[0] for sprite in sprites)
    largest_w = max(sprite.shape[1] for sprite in sprites)
    footprint = sum((sprite.shape[0] + 1) * (sprite.shape[1] + 1) for sprite in sprites)

    placements = None
    for _ in range(60):
        base = max(int(np.ceil(np.sqrt(footprint))) + 3, largest_h + 2, largest_w + 2, 14)
        grid_h = random.randint(base, min(30, base + 4))
        grid_w = random.randint(base, min(30, base + 4))
        placements = _place_sprites(sprites, grid_h, grid_w, gap=1)
        if placements is not None:
            break
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for sprite, (r0, c0) in zip(sprites, placements):
        paste(grid, sprite, r0, c0, transparent=BG)

    for hole, (r0, c0) in zip(hole_records, placements[: len(hole_records)]):
        hole["origin"] = (r0 + hole["hole_offset"][0], c0 + hole["hole_offset"][1])
    for piece, (r0, c0) in zip(piece_records, placements[len(hole_records) :]):
        piece["pos"] = (r0, c0)

    expected = _build_expected_output(grid, hole_records, piece_records)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid two-piece hole-assembly input")
