from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 800

Coord = tuple[int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr_raw, sc_raw in np.argwhere(mask):
        sr, sc = int(sr_raw), int(sc_raw)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(coords, dtype=int))
    return out


def _normalize(coords: list[Coord] | tuple[Coord, ...] | np.ndarray) -> tuple[Coord, ...]:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in arr)
    return tuple(rel)


def _shape_size(rel: tuple[Coord, ...]) -> tuple[int, int]:
    return max(r for r, _c in rel) + 1, max(c for _r, c in rel) + 1


def _rel_to_mask(rel: tuple[Coord, ...]) -> np.ndarray:
    h, w = _shape_size(rel)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in rel:
        mask[int(r), int(c)] = True
    return mask


def _mask_to_rel(mask: np.ndarray) -> tuple[Coord, ...]:
    return _normalize(np.argwhere(mask))


def _paint_cells(g: np.ndarray, coords: np.ndarray | list[Coord], color: int) -> None:
    for r, c in coords:
        g[int(r), int(c)] = int(color)


def _cavity_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    cavities: list[dict] = []
    for coords in _components(g == BG):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        cells = {(int(r), int(c)) for r, c in coords}
        ok = True
        touches_wall = False
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if (rr, cc) in cells:
                    continue
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(g[rr, cc]) != WALL:
                    ok = False
                    break
                touches_wall = True
            if not ok:
                break
        if ok and touches_wall:
            b = _bbox(coords)
            rel = _normalize(coords)
            cavities.append(
                {
                    "coords": tuple(sorted(cells)),
                    "set": cells,
                    "bbox": b,
                    "area": int(coords.shape[0]),
                    "rel": rel,
                }
            )
    cavities.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["area"]))
    return cavities


def _sprite_components(g: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    seen_colors: set[int] = set()
    for color in sorted(set(map(int, g.ravel())) - {BG, WALL}):
        comps = _components(g == int(color))
        assert len(comps) == 1, "each sprite color must form exactly one component"
        assert int(color) not in seen_colors, "sprite colors must be unique"
        seen_colors.add(int(color))
        coords = comps[0]
        b = _bbox(coords)
        rel = _normalize(coords)
        sprites.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": b,
                "rel": rel,
                "area": int(coords.shape[0]),
            }
        )
    sprites.sort(key=lambda item: (len(item["rel"]), item["bbox"][0], item["bbox"][1], item["color"]))
    return sprites


def _placement_options(cavities: list[dict], sprites: list[dict]) -> list[list[tuple[int, int, tuple[Coord, ...]]]]:
    all_options: list[list[tuple[int, int, tuple[Coord, ...]]]] = []
    for sprite in sprites:
        rel = sprite["rel"]
        sh, sw = _shape_size(rel)
        options: list[tuple[int, int, tuple[Coord, ...]]] = []
        for cav_i, cavity in enumerate(cavities):
            r0, c0, r1, c1 = cavity["bbox"]
            if sh > r1 - r0 + 1 or sw > c1 - c0 + 1:
                continue
            index = {p: k for k, p in enumerate(cavity["coords"])}
            for ar in range(r0, r1 - sh + 2):
                for ac in range(c0, c1 - sw + 2):
                    shifted = tuple(sorted((ar + r, ac + c) for r, c in rel))
                    if all(p in index for p in shifted):
                        bits = 0
                        for p in shifted:
                            bits |= 1 << index[p]
                        options.append((cav_i, bits, shifted))
        assert options, "every sprite must have at least one translation into a cavity"
        options.sort(key=lambda item: (item[0], item[2]))
        all_options.append(options)
    return all_options


def _find_unique_assembly(cavities: list[dict], sprites: list[dict]) -> list[tuple[int, int, tuple[Coord, ...]]]:
    full_masks = [(1 << int(cavity["area"])) - 1 for cavity in cavities]
    options = _placement_options(cavities, sprites)
    remaining_area_by_sprite = [int(sprite["area"]) for sprite in sprites]
    solutions: list[list[tuple[int, int, tuple[Coord, ...]]]] = []

    def recurse(remaining: tuple[int, ...], covered: tuple[int, ...], chosen: list[tuple[int, int, tuple[Coord, ...]]]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            if all(mask == 0 or mask == full for mask, full in zip(covered, full_masks)):
                solutions.append(list(chosen))
            return

        total_remaining = sum(remaining_area_by_sprite[i] for i in remaining)
        forced_deficit = 0
        for mask, full in zip(covered, full_masks):
            if mask not in (0, full):
                forced_deficit += int(full.bit_count() - mask.bit_count())
        if forced_deficit > total_remaining:
            return

        best_sprite = -1
        best_options: list[tuple[int, int, tuple[Coord, ...]]] | None = None
        for sprite_i in remaining:
            feasible = [
                option
                for option in options[sprite_i]
                if (option[1] & covered[option[0]]) == 0
            ]
            if not feasible:
                return
            if best_options is None or len(feasible) < len(best_options):
                best_sprite = int(sprite_i)
                best_options = feasible

        assert best_options is not None
        next_remaining = tuple(i for i in remaining if i != best_sprite)
        for cav_i, bits, coords in best_options:
            new_covered = list(covered)
            new_covered[cav_i] |= bits
            recurse(next_remaining, tuple(new_covered), chosen + [(best_sprite, cav_i, coords)])

    recurse(tuple(range(len(sprites))), tuple(0 for _ in cavities), [])
    assert solutions, "the sprites must have a valid translation-only cavity assembly"
    assert len(solutions) == 1, "the cavity assembly must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate free colored sprites into gray-wall cavities so each used cavity is exactly filled.

    Input structure:
    - Black background contains one gray terrain wall with several enclosed black cavities.
    - Outside the wall are separated monochrome free sprites, with each sprite color used once.
    - Some cavities may be distractors, including a rotated-but-not-translated shape match.

    Rule:
    - Find all black components fully enclosed by gray wall cells and all non-black, non-gray sprite components.
    - Move every sprite by translation only into a cavity; rotations and reflections are not allowed.
    - A cavity that receives sprites must be covered exactly once on every cell, while unused cavities stay black.
    - Clear the sprites from their original positions, preserve the gray wall, and paint the unique assembly.

    Key insight:
    - The destination is determined by exact complementary shape coverage inside the cavities, not by distance, gravity,
      color order, or one-to-one congruent matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30
    assert np.any(g == WALL), "expected gray wall cells"

    cavities = _cavity_components(g)
    sprites = _sprite_components(g)
    assert cavities, "expected at least one enclosed cavity"
    assert sprites, "expected at least one free sprite"

    assembly = _find_unique_assembly(cavities, sprites)

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for sprite_i, _cav_i, coords in assembly:
        _paint_cells(out, coords, sprites[sprite_i]["color"])
    return out


# ---------------- generator ----------------


MULTI_TEMPLATES = [
    ["AA..", ".AAB", "..BB"],
    ["AAA.", "..AB", ".BBB"],
    ["AA.B", "AABB", "..BB"],
    ["AA..", "ABBC", ".BCC", ".B.."],
    [".AA.", "BAAC", "BBCC", "B..."],
]

SINGLE_TEMPLATES = [
    ["XX.", ".X.", ".XX"],
    ["XXX", "X..", "XX."],
    [".XX", "XX.", ".X."],
    ["X..", "XXX", "..X"],
    ["XX.", ".XX", "..X"],
]


def _connected_rel(rel: tuple[Coord, ...]) -> bool:
    cells = set(rel)
    q: deque[Coord] = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(cells)


def _same_shape(a: tuple[Coord, ...], b: tuple[Coord, ...]) -> bool:
    return tuple(a) == tuple(b)


def _rotations(rel: tuple[Coord, ...]) -> list[tuple[Coord, ...]]:
    mask = _rel_to_mask(rel)
    out: list[tuple[Coord, ...]] = []
    for k in range(1, 4):
        out.append(_mask_to_rel(np.rot90(mask, k)))
    return out


def _crop_label_canvas(labels: np.ndarray) -> np.ndarray:
    coords = np.argwhere(labels != ".")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return labels[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_label_transform(lines: list[str]) -> np.ndarray:
    h = len(lines)
    w = max(len(row) for row in lines)
    labels = np.full((h, w), ".", dtype="<U1")
    for r, row in enumerate(lines):
        for c, ch in enumerate(row):
            labels[r, c] = ch

    labels = np.rot90(labels, random.randrange(4))
    if random.random() < 0.5:
        labels = np.flipud(labels)
    if random.random() < 0.5:
        labels = np.fliplr(labels)
    return _crop_label_canvas(labels)


def _label_parts(labels: np.ndarray) -> tuple[tuple[Coord, ...], list[tuple[str, tuple[Coord, ...]]]]:
    cavity = _mask_to_rel(labels != ".")
    letters: list[str] = []
    for raw in labels.ravel():
        ch = str(raw)
        if ch != "." and ch not in letters:
            letters.append(ch)

    parts: list[tuple[str, tuple[Coord, ...]]] = []
    for ch in letters:
        coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(labels == ch)))
        parts.append((ch, coords))
    return cavity, parts


def _choose_plan() -> tuple[list[tuple[Coord, ...]], list[list[tuple[int, tuple[Coord, ...]]]], list[tuple[Coord, ...]], int]:
    cavities: list[tuple[Coord, ...]] = []
    assignments: list[list[tuple[int, tuple[Coord, ...]]]] = []
    pieces: list[tuple[Coord, ...]] = []

    multi_count = 2 if random.random() < 0.35 else 1
    multi_templates = random.sample(MULTI_TEMPLATES, multi_count)
    for template in multi_templates:
        labels = _random_label_transform(template)
        cavity, parts = _label_parts(labels)
        part_assignments: list[tuple[int, tuple[Coord, ...]]] = []
        for _ch, local_cells in parts:
            rel = _normalize(local_cells)
            assert _connected_rel(rel)
            piece_i = len(pieces)
            pieces.append(rel)
            part_assignments.append((piece_i, local_cells))
        cavities.append(cavity)
        assignments.append(part_assignments)

    single_count = random.randint(1, 2)
    for template in random.sample(SINGLE_TEMPLATES, single_count):
        labels = _random_label_transform(template)
        cavity, parts = _label_parts(labels)
        assert len(parts) == 1
        local_cells = parts[0][1]
        rel = _normalize(local_cells)
        assert _connected_rel(rel)
        assignments.append([(len(pieces), local_cells)])
        pieces.append(rel)
        cavities.append(cavity)

    nonsymmetric = [i for i, piece in enumerate(pieces) if all(not _same_shape(piece, rot) for rot in _rotations(piece))]
    if not nonsymmetric:
        return _choose_plan()
    trap_piece_i = random.choice(nonsymmetric)
    trap_shape = random.choice([rot for rot in _rotations(pieces[trap_piece_i]) if not _same_shape(rot, pieces[trap_piece_i])])
    cavities.append(trap_shape)
    assignments.append([])

    order = list(range(len(cavities)))
    random.shuffle(order)
    old_to_new = {old: new for new, old in enumerate(order)}
    shuffled_cavities = [cavities[i] for i in order]
    shuffled_assignments = [assignments[i] for i in order]
    trap_cavity_i = old_to_new[len(cavities) - 1]
    return shuffled_cavities, shuffled_assignments, pieces, trap_cavity_i


def _dilated(mask: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                if abs(dr) + abs(dc) <= pad:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr, cc] = True
    return out


def _find_random_location(occupied: np.ndarray, rel: tuple[Coord, ...], border: int, padding: int) -> Coord | None:
    h, w = occupied.shape
    sh, sw = _shape_size(rel)
    blocked = _dilated(occupied, padding)
    candidates: list[Coord] = []
    for r0 in range(border, h - sh - border + 1):
        for c0 in range(border, w - sw - border + 1):
            cells = [(r0 + r, c0 + c) for r, c in rel]
            if all(not blocked[r, c] for r, c in cells):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_cavities(
    grid: np.ndarray,
    wall_box: tuple[int, int, int, int],
    cavities: list[tuple[Coord, ...]],
) -> list[tuple[Coord, ...]] | None:
    r0, c0, r1, c1 = wall_box
    wall_h = r1 - r0 + 1
    wall_w = c1 - c0 + 1
    occupied = np.zeros((wall_h, wall_w), dtype=bool)
    placed: list[tuple[Coord, ...]] = []
    order = list(range(len(cavities)))
    order.sort(key=lambda i: len(cavities[i]), reverse=True)
    local_by_index: dict[int, tuple[Coord, ...]] = {}

    for cav_i in order:
        rel = cavities[cav_i]
        loc = _find_random_location(occupied, rel, border=1, padding=2)
        if loc is None:
            return None
        lr, lc = loc
        local = tuple(sorted((lr + r, lc + c) for r, c in rel))
        for r, c in local:
            occupied[r, c] = True
        local_by_index[cav_i] = local

    for cav_i in range(len(cavities)):
        global_cells = tuple(sorted((r0 + r, c0 + c) for r, c in local_by_index[cav_i]))
        for r, c in global_cells:
            grid[r, c] = BG
        placed.append(global_cells)
    return placed


def _place_sprites(
    grid: np.ndarray,
    wall_bottom: int,
    pieces: list[tuple[Coord, ...]],
    colors: list[int],
) -> list[tuple[Coord, ...]] | None:
    h, w = grid.shape
    zone_top = wall_bottom + 2
    if zone_top >= h - 1:
        return None
    zone = np.zeros((h - zone_top - 1, w - 2), dtype=bool)
    placed_global: list[tuple[Coord, ...]] = []

    order = list(range(len(pieces)))
    random.shuffle(order)
    local_by_piece: dict[int, tuple[Coord, ...]] = {}
    for piece_i in order:
        rel = pieces[piece_i]
        loc = _find_random_location(zone, rel, border=0, padding=1)
        if loc is None:
            return None
        lr, lc = loc
        local = tuple(sorted((lr + r, lc + c) for r, c in rel))
        for r, c in local:
            zone[r, c] = True
        local_by_piece[piece_i] = local

    for piece_i in range(len(pieces)):
        color = int(colors[piece_i])
        global_cells = tuple(sorted((zone_top + r, 1 + c) for r, c in local_by_piece[piece_i]))
        for r, c in global_cells:
            grid[r, c] = color
        placed_global.append(global_cells)
    return placed_global


def _render_expected(
    inp: np.ndarray,
    original_cavities: list[tuple[Coord, ...]],
    placed_cavities: list[tuple[Coord, ...]],
    assignments: list[list[tuple[int, tuple[Coord, ...]]]],
    colors: list[int],
    sprite_cells: list[tuple[Coord, ...]],
) -> np.ndarray:
    out = inp.copy()
    for cells in sprite_cells:
        for r, c in cells:
            out[r, c] = BG

    for cav_i, piece_indices in enumerate(assignments):
        if not piece_indices:
            continue
        anchor_r = min(r for r, _c in placed_cavities[cav_i])
        anchor_c = min(c for _r, c in placed_cavities[cav_i])

        cav_local = original_cavities[cav_i]
        for piece_i, target_cells in piece_indices:
            for pr, pc in target_cells:
                out[anchor_r + pr, anchor_c + pc] = int(colors[piece_i])
        filled = {(anchor_r + r, anchor_c + c) for r, c in cav_local}
        expected = set(placed_cavities[cav_i])
        assert filled == expected
    return out


def _has_multicolor_cavity(out: np.ndarray, cavities: list[tuple[Coord, ...]]) -> bool:
    for cells in cavities:
        colors = {int(out[r, c]) for r, c in cells if int(out[r, c]) not in (BG, WALL)}
        if len(colors) >= 2:
            return True
    return False


def _trap_is_rotated_sprite(
    trap_cells: tuple[Coord, ...],
    placed_cavities: list[tuple[Coord, ...]],
    trap_i: int,
    pieces: list[tuple[Coord, ...]],
) -> bool:
    del placed_cavities
    trap_rel = _normalize(trap_cells)
    for piece in pieces:
        if _same_shape(piece, trap_rel):
            continue
        if any(_same_shape(rot, trap_rel) for rot in _rotations(piece)):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    original_cavities, assignments, pieces, trap_i = _choose_plan()
    if len(pieces) > len(PALETTE):
        return None
    if not any(len(x) >= 2 for x in assignments):
        return None

    rows = random.randint(24, 30)
    cols = random.randint(24, 30)
    wall_top = 1
    wall_left = 1
    wall_h = random.randint(12, min(16, rows - 9))
    wall_box = (wall_top, wall_left, wall_top + wall_h - 1, cols - 2)
    grid = np.full((rows, cols), BG, dtype=int)
    r0, c0, r1, c1 = wall_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = WALL

    placed_cavities = _place_cavities(grid, wall_box, original_cavities)
    if placed_cavities is None:
        return None

    colors = random.sample(PALETTE, len(pieces))
    sprite_cells = _place_sprites(grid, r1, pieces, colors)
    if sprite_cells is None:
        return None

    try:
        expected = _render_expected(
            grid,
            original_cavities,
            placed_cavities,
            assignments,
            colors,
            sprite_cells,
        )
    except AssertionError:
        return None

    if not _trap_is_rotated_sprite(placed_cavities[trap_i], placed_cavities, trap_i, pieces):
        return None

    if any(int(expected[r, c]) != BG for r, c in placed_cavities[trap_i]):
        return None
    if not _has_multicolor_cavity(expected, placed_cavities):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/4/002/4 input")
