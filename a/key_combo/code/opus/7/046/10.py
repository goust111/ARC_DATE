import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
CONTAINER_COLOR = 1
SPRITE_COLORS = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 500


def _shape_from_rows(*rows: str) -> frozenset[tuple[int, int]]:
    cells = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                cells.append((r, c))
    return _normalize_shape(cells)


def _normalize_shape(cells) -> frozenset[tuple[int, int]]:
    coords = list(cells)
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _shape_dims(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _mirror_shape(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    _h, w = _shape_dims(shape)
    return _normalize_shape((r, w - 1 - c) for r, c in shape)


def _shape_key(shape: frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(shape))


RAW_SHAPES = (
    _shape_from_rows("#.", "##"),
    _shape_from_rows("##.", ".##"),
    _shape_from_rows("#.", "#.", "##"),
    _shape_from_rows("#..", "###"),
    _shape_from_rows("#..", "#..", "###"),
    _shape_from_rows(".##", "##.", "#.."),
    _shape_from_rows("##.", ".##", "..#"),
    _shape_from_rows("#..", "##.", ".##"),
    _shape_from_rows(".#.", "##.", ".##"),
    _shape_from_rows("##..", ".###"),
    _shape_from_rows("#...", "####"),
    _shape_from_rows("##.", "#..", "###"),
    _shape_from_rows("#..", "###", "..#"),
    _shape_from_rows("#..", "###", ".#."),
    _shape_from_rows("##.", ".#.", ".##"),
)


def _build_shape_bank() -> tuple[frozenset[tuple[int, int]], ...]:
    bank = []
    seen = set()
    for shape in RAW_SHAPES:
        mirror = _mirror_shape(shape)
        if shape == mirror:
            continue
        key = min(_shape_key(shape), _shape_key(mirror))
        if key in seen:
            continue
        seen.add(key)
        bank.append(shape)
    return tuple(bank)


SHAPE_BANK = _build_shape_bank()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    components = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components_mask(grid == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _find_component_holes(grid: np.ndarray, comp: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    local = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    h, w = local.shape
    holes = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if local[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not local[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if touches_edge:
                continue
            coords = np.array([(r0 + rr, c0 + cc) for rr, cc in cells], dtype=int)
            shape = _normalize_shape(cells)
            holes.append(
                {
                    "coords": coords,
                    "cells": frozenset((int(rr), int(cc)) for rr, cc in coords.tolist()),
                    "bbox": (
                        int(coords[:, 0].min()),
                        int(coords[:, 1].min()),
                        int(coords[:, 0].max()),
                        int(coords[:, 1].max()),
                    ),
                    "shape": shape,
                    "area": int(coords.shape[0]),
                    "color": int(bg),
                }
            )
    return holes


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    bg = _get_bg(grid)
    components = _find_monochrome_components(grid, bg)

    container_indices = set()
    holes = []
    for idx, comp in enumerate(components):
        comp_holes = _find_component_holes(grid, comp, bg)
        if comp_holes:
            container_indices.add(idx)
            holes.extend(comp_holes)

    containers = [components[idx] for idx in sorted(container_indices)]
    sprites = []
    for idx, comp in enumerate(components):
        if idx in container_indices:
            continue
        shape = _normalize_shape(map(tuple, comp["coords"].tolist()))
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "shape": shape,
                "mirror_shape": _mirror_shape(shape),
                "area": int(comp["area"]),
            }
        )

    return containers, holes, sprites


def _solve_assignment(sprites: list[dict], holes: list[dict], *, max_solutions: int = 2):
    if len(sprites) != len(holes):
        return []
    if sum(sprite["area"] for sprite in sprites) != sum(hole["area"] for hole in holes):
        return []

    candidates = {}
    for si, sprite in enumerate(sprites):
        options = []
        for hi, hole in enumerate(holes):
            if sprite["shape"] == hole["shape"]:
                options.append((hi, False))
            if sprite["mirror_shape"] == hole["shape"]:
                options.append((hi, True))
        options = sorted(set(options))
        if not options:
            return []
        candidates[si] = options

    solutions = []

    def search(remaining_sprites: set[int], used_holes: set[int], assignment: dict[int, tuple[int, bool]]):
        if len(solutions) >= max_solutions:
            return
        if not remaining_sprites:
            solutions.append(dict(assignment))
            return
        sprite_index = min(
            remaining_sprites,
            key=lambda si: sum(1 for hi, _m in candidates[si] if hi not in used_holes),
        )
        for hole_index, use_mirror in candidates[sprite_index]:
            if hole_index in used_holes:
                continue
            assignment[sprite_index] = (hole_index, use_mirror)
            search(remaining_sprites - {sprite_index}, used_holes | {hole_index}, assignment)
            del assignment[sprite_index]

    search(set(range(len(sprites))), set(), {})
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outside sprite, using horizontal mirroring only when needed, into the unique matching enclosed hole inside the blue containers.

    Input structure:
    - A black background contains one or two large blue monochromatic components.
    - Each blue component may contain one or more enclosed black holes.
    - Separate small monochromatic sprites in red, green, or yellow sit outside the blue objects.

    Rule:
    - Detect the enclosed black hole components inside the holed blue objects.
    - For each outside sprite, consider both its original mask and its left-right mirror inside its tight bbox.
    - Find the unique one-to-one assignment where every sprite exactly matches one hole in one of those two forms.
    - Erase the sprites from their original locations and fill the matched hole cells with the sprite colors.

    Key insight:
    - The task is a joint constraint problem: mirror choice and hole assignment must be solved together because only some sprites need flipping.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    containers, holes, sprites = _extract_scene(g)

    assert bg == BACKGROUND
    assert containers
    assert holes
    assert sprites

    solutions = _solve_assignment(sprites, holes, max_solutions=2)
    assert len(solutions) == 1
    assignment = solutions[0]

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for sprite_index, (hole_index, _use_mirror) in assignment.items():
        color = sprites[sprite_index]["color"]
        coords = holes[hole_index]["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _shape_to_patch(shape: frozenset[tuple[int, int]], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _fits_with_padding(grid: np.ndarray, patch: np.ndarray, top: int, left: int, *, padding: int, border: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < border or left < border or top + h > gh - border or left + w > gw - border:
        return False

    cells = np.argwhere(patch != BACKGROUND)
    for dr, dc in cells:
        rr = top + int(dr)
        cc = left + int(dc)
        if grid[rr, cc] != BACKGROUND:
            return False
        for nr in range(rr - padding, rr + padding + 1):
            for nc in range(cc - padding, cc + padding + 1):
                if 0 <= nr < gh and 0 <= nc < gw and grid[nr, nc] != BACKGROUND:
                    return False
    return True


def _random_place_patch(grid: np.ndarray, patch: np.ndarray, *, padding: int, border: int) -> tuple[int, int] | None:
    gh, gw = grid.shape
    h, w = patch.shape
    candidates = []
    for top in range(border, gh - border - h + 1):
        for left in range(border, gw - border - w + 1):
            if _fits_with_padding(grid, patch, top, left, padding=padding, border=border):
                candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BACKGROUND
    target = grid[top : top + h, left : left + w]
    target[mask] = patch[mask]


def _container_patch_for_holes(hole_shapes: list[frozenset[tuple[int, int]]]):
    total_hole_area = sum(len(shape) for shape in hole_shapes)
    order = sorted(range(len(hole_shapes)), key=lambda idx: len(hole_shapes[idx]), reverse=True)

    for _ in range(200):
        max_h = max(_shape_dims(shape)[0] for shape in hole_shapes)
        max_w = max(_shape_dims(shape)[1] for shape in hole_shapes)
        inner_h = random.randint(max_h + (0 if len(hole_shapes) == 1 else 1), max_h + 3 + len(hole_shapes))
        inner_w = random.randint(max_w + len(hole_shapes), max_w + 4 + 2 * len(hole_shapes))
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        container_area = outer_h * outer_w - total_hole_area
        if not (20 <= container_area <= 52):
            continue

        occupied = set()
        placements = {}
        ok = True
        for idx in order:
            shape = hole_shapes[idx]
            sh, sw = _shape_dims(shape)
            candidates = []
            for top in range(inner_h - sh + 1):
                for left in range(inner_w - sw + 1):
                    conflict = False
                    for r, c in shape:
                        rr = top + r
                        cc = left + c
                        for nr in range(rr - 1, rr + 2):
                            for nc in range(cc - 1, cc + 2):
                                if (nr, nc) in occupied:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                    if not conflict:
                        candidates.append((top, left))
            if not candidates:
                ok = False
                break
            placements[idx] = random.choice(candidates)
            for r, c in shape:
                occupied.add((placements[idx][0] + r, placements[idx][1] + c))
        if not ok:
            continue

        patch = np.full((outer_h, outer_w), CONTAINER_COLOR, dtype=int)
        hole_records = []
        for idx, shape in enumerate(hole_shapes):
            top, left = placements[idx]
            cells = []
            for r, c in shape:
                rr = top + 1 + r
                cc = left + 1 + c
                patch[rr, cc] = BACKGROUND
                cells.append((rr, cc))
            hole_records.append({"shape": shape, "cells": tuple(cells)})
        return patch, hole_records

    return None, None


def _sample_holes() -> tuple[list[frozenset[tuple[int, int]]], list[bool]]:
    n_holes = random.randint(2, 4)
    shapes = random.sample(list(SHAPE_BANK), k=n_holes)
    oriented_shapes = []
    for shape in shapes:
        oriented_shapes.append(shape if random.random() < 0.5 else _mirror_shape(shape))

    needs_flip = [False] * n_holes
    flip_indices = random.sample(range(n_holes), k=random.randint(1, n_holes - 1))
    for idx in flip_indices:
        needs_flip[idx] = True
    return oriented_shapes, needs_flip


def _container_split(n_holes: int) -> list[int]:
    if n_holes == 2:
        return [2] if random.random() < 0.6 else [1, 1]
    if n_holes == 3:
        return random.choice(([3], [2, 1], [1, 2]))
    return random.choice(([4], [2, 2], [3, 1], [1, 3]))


def _choose_sprite_colors(n: int) -> list[int]:
    colors = [random.choice(SPRITE_COLORS) for _ in range(n)]
    if n > 1 and len(set(colors)) == 1:
        colors[random.randrange(n)] = random.choice([c for c in SPRITE_COLORS if c != colors[0]])
    return colors


def _generate_attempt() -> np.ndarray | None:
    hole_shapes, needs_flip = _sample_holes()
    groups = _container_split(len(hole_shapes))

    containers_data = []
    hole_index = 0
    for count in groups:
        chunk = hole_shapes[hole_index : hole_index + count]
        patch, hole_records = _container_patch_for_holes(chunk)
        if patch is None:
            return None
        containers_data.append((patch, hole_records))
        hole_index += count

    max_container_h = max(patch.shape[0] for patch, _holes in containers_data)
    total_container_w = sum(patch.shape[1] for patch, _holes in containers_data)
    min_h = max(15, max_container_h + 6)
    min_w = max(15, total_container_w + 8)
    max_h = min(24, max(min_h, 18))
    max_w = min(28, max(min_w, 20))
    if min_h > max_h or min_w > max_w:
        return None
    grid_h = random.randint(min_h, max_h)
    grid_w = random.randint(min_w, max_w)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    placed_holes = []
    for patch, hole_records in sorted(containers_data, key=lambda item: item[0].shape[1], reverse=True):
        loc = _random_place_patch(grid, patch, padding=2, border=1)
        if loc is None:
            return None
        top, left = loc
        _paste_patch(grid, patch, top, left)
        for record in hole_records:
            placed_holes.append(
                {
                    "shape": record["shape"],
                    "cells": frozenset((top + r, left + c) for r, c in record["cells"]),
                }
            )

    sprite_colors = _choose_sprite_colors(len(hole_shapes))
    for idx, hole_shape in enumerate(hole_shapes):
        sprite_shape = _mirror_shape(hole_shape) if needs_flip[idx] else hole_shape
        sprite_patch = _shape_to_patch(sprite_shape, sprite_colors[idx])
        loc = _random_place_patch(grid, sprite_patch, padding=1, border=0)
        if loc is None:
            return None
        _paste_patch(grid, sprite_patch, loc[0], loc[1])

    _containers, holes, sprites = _extract_scene(grid)
    if len(holes) != len(hole_shapes) or len(sprites) != len(hole_shapes):
        return None

    solutions = _solve_assignment(sprites, holes, max_solutions=3)
    if len(solutions) != 1:
        return None
    solution = solutions[0]
    mirror_count = sum(1 for _si, (_hi, use_mirror) in solution.items() if use_mirror)
    if mirror_count == 0 or mirror_count == len(sprites):
        return None

    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
