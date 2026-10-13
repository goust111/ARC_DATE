from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
SPRITE_COLORS = [2, 3, 4, 6, 7, 8, 9]

GEN_HEIGHT_RANGE = (14, 18)
GEN_WIDTH_RANGE = (14, 18)
GEN_SPRITE_COUNT_RANGE = (3, 5)
GEN_SPRITE_AREA_RANGE = (3, 7)
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _orient(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(cells, dtype=int)
    variants = set()
    for k in range(4):
        pts = arr.copy()
        for _ in range(k):
            pts = np.stack([-pts[:, 1], pts[:, 0]], axis=1)
        for flip in (1, -1):
            cur = pts.copy()
            cur[:, 1] *= flip
            variants.add(_normalize(set(map(tuple, cur))))
    return random.choice(sorted(variants))


def _random_polyomino(area: int) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        oriented = _orient(_normalize(cells))
        rows = [r for r, _ in oriented]
        cols = [c for _, c in oriented]
        if max(rows) <= 3 and max(cols) <= 3:
            return oriented
    raise RuntimeError("failed to generate sprite shape")


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = color
    return patch


def _sprite_components(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BLACK, BLUE, GREY}):
        for comp in _components(grid == color):
            sprites.append(
                {
                    "color": color,
                    "coords": comp,
                    "area": len(comp),
                    "bbox": _bbox(comp),
                }
            )
    return sprites


def _expand_halos(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    grays = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    for r, c in grays:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == BLACK:
                    out[nr, nc] = BLUE
        out[r, c] = GREY
    return out


def _scene_summary(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    sprites = _sprite_components(grid)
    owner = np.full(grid.shape, -1, dtype=int)
    for idx, sprite in enumerate(sprites):
        for r, c in sprite["coords"]:
            owner[r, c] = idx

    expanded = _expand_halos(grid)
    comps = []
    for comp in _components(expanded != BLACK):
        sprite_ids = sorted({int(owner[r, c]) for r, c in comp if owner[r, c] >= 0})
        comps.append({"coords": comp, "sprite_ids": sprite_ids})
    return sprites, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bridge sprites through gray-marker halos, recolor each bridged group to its largest sprite, and erase isolated sprites.

    Input structure:
    - The grid is a black background with 3 to 5 monochrome sprite components in distinct non-black, non-blue, non-gray colors.
    - Gray single-cell markers sit in the empty space between sprites and are not orthogonally adjacent to any sprite pixel.
    - Sprites are separated so they do not directly touch; the hidden interaction comes only from halo overlap after marker expansion.

    Rule:
    - Expand every gray marker into a clipped 3x3 blue halo over black cells only, then keep the marker center gray.
    - Build 4-connected components over all non-black cells after the halos are drawn, ignoring color differences.
    - In each component containing at least two sprites, recolor every sprite in that component to the color of the largest sprite; ties break by the smallest sprite bbox top row, then left column.
    - Erase any sprite that belongs to a component containing only that one sprite, then remove every blue halo pixel and gray marker.

    Key insight:
    - The gray markers define an invisible connectivity graph: only sprites linked by overlapping halo chains survive, and the dominant sprite in each linked group determines the final color.
    """
    g = np.asarray(grid, dtype=int)
    sprites, comps = _scene_summary(g)
    out = np.zeros_like(g)

    for comp in comps:
        sprite_ids = comp["sprite_ids"]
        if len(sprite_ids) < 2:
            continue
        leader = min(
            sprite_ids,
            key=lambda idx: (
                -sprites[idx]["area"],
                sprites[idx]["bbox"][0],
                sprites[idx]["bbox"][1],
            ),
        )
        leader_color = sprites[leader]["color"]
        for sprite_id in sprite_ids:
            for r, c in sprites[sprite_id]["coords"]:
                out[r, c] = leader_color

    return out


def _sprite_distance_ok(
    candidate: list[tuple[int, int]],
    placed: list[list[tuple[int, int]]],
) -> bool:
    for other in placed:
        for r1, c1 in candidate:
            for r2, c2 in other:
                if abs(r1 - r2) + abs(c1 - c2) < 3:
                    return False
    return True


def _place_patch_in_zone(
    grid: np.ndarray,
    patch: np.ndarray,
    zone: tuple[int, int, int, int],
    placed: list[list[tuple[int, int]]],
) -> list[tuple[int, int]] | None:
    h, w = patch.shape
    r0, r1, c0, c1 = zone
    if r1 - r0 < h or c1 - c0 < w:
        return None

    candidates = [
        (rr, cc)
        for rr in range(r0, r1 - h + 1)
        for cc in range(c0, c1 - w + 1)
    ]
    random.shuffle(candidates)
    for rr, cc in candidates:
        coords = [(rr + r, cc + c) for r, c in zip(*np.where(patch != BLACK), strict=False)]
        if not _sprite_distance_ok(coords, placed):
            continue
        for r, c in coords:
            grid[r, c] = int(patch[r - rr, c - cc])
        return coords
    return None


def _bridge_and_isolated_zones(h: int, w: int) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    corner_h = max(4, h // 3)
    corner_w = max(4, w // 3)
    if random.random() < 0.5:
        band_top = max(1, h // 2 - 3)
        band_bot = min(h - 1, h // 2 + 3)
        bridges = [
            (band_top, band_bot, 1, corner_w + 1),
            (band_top, band_bot, w - corner_w - 1, w - 1),
        ]
    else:
        band_left = max(1, w // 2 - 3)
        band_right = min(w - 1, w // 2 + 3)
        bridges = [
            (1, corner_h + 1, band_left, band_right),
            (h - corner_h - 1, h - 1, band_left, band_right),
        ]

    isolated = [
        (1, corner_h + 1, 1, corner_w + 1),
        (1, corner_h + 1, w - corner_w - 1, w - 1),
        (h - corner_h - 1, h - 1, 1, corner_w + 1),
        (h - corner_h - 1, h - 1, w - corner_w - 1, w - 1),
    ]
    return bridges, isolated


def _valid_marker_cell(cell: tuple[int, int], sprite_occ: np.ndarray) -> bool:
    r, c = cell
    if sprite_occ[r, c]:
        return False
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(sprite_occ.shape, nr, nc) and sprite_occ[nr, nc]:
            return False
    return True


def _marker_patch(cell: tuple[int, int], sprite_occ: np.ndarray) -> frozenset[tuple[int, int]]:
    r, c = cell
    cells = {(r, c)}
    h, w = sprite_occ.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not sprite_occ[nr, nc]:
                cells.add((nr, nc))
    return frozenset(cells)


def _touching_sprites(
    patch: frozenset[tuple[int, int]],
    sprite_sets: list[set[tuple[int, int]]],
) -> frozenset[int]:
    touched = set()
    for idx, sprite in enumerate(sprite_sets):
        found = False
        for r, c in patch:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in sprite:
                    touched.add(idx)
                    found = True
                    break
            if found:
                break
    return frozenset(touched)


def _patches_connect(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _find_marker_chain(
    grid: np.ndarray,
    sprite_sets: list[set[tuple[int, int]]],
    first_idx: int,
    second_idx: int,
) -> list[tuple[int, int]] | None:
    sprite_occ = grid != BLACK
    candidates: dict[tuple[int, int], dict] = {}
    allowed_touch = {first_idx, second_idx}

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cell = (r, c)
            if not _valid_marker_cell(cell, sprite_occ):
                continue
            patch = _marker_patch(cell, sprite_occ)
            touch = _touching_sprites(patch, sprite_sets)
            if not touch.issubset(allowed_touch):
                continue
            candidates[cell] = {"patch": patch, "touch": touch}

    starts = [cell for cell, info in candidates.items() if first_idx in info["touch"]]
    ends = {cell for cell, info in candidates.items() if second_idx in info["touch"]}
    if not starts or not ends:
        return None

    q = deque(starts)
    prev = {cell: None for cell in starts}
    while q:
        cell = q.popleft()
        if cell in ends:
            path = []
            cur = cell
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path if len(path) >= 2 else None
        r, c = cell
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt not in candidates or nxt in prev:
                    continue
                if _patches_connect(candidates[cell]["patch"], candidates[nxt]["patch"]):
                    prev[nxt] = cell
                    q.append(nxt)
    return None


def _add_distractor_markers(
    grid: np.ndarray,
    sprite_sets: list[set[tuple[int, int]]],
    reserved: set[tuple[int, int]],
) -> None:
    sprite_occ = grid != BLACK
    reserved_patches = [_marker_patch(cell, sprite_occ) for cell in reserved]
    candidates = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cell = (r, c)
            if cell in reserved or not _valid_marker_cell(cell, sprite_occ):
                continue
            patch = _marker_patch(cell, sprite_occ)
            if _touching_sprites(patch, sprite_sets):
                continue
            if any(_patches_connect(patch, other) for other in reserved_patches):
                continue
            candidates.append((cell, patch))

    random.shuffle(candidates)
    extras = random.randint(0, 2)
    chosen: list[frozenset[tuple[int, int]]] = []
    placed = 0
    for cell, patch in candidates:
        if placed >= extras:
            break
        if any(_patches_connect(patch, other) for other in chosen):
            continue
        grid[cell] = GREY
        chosen.append(patch)
        placed += 1


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    sprites = _sprite_components(grid)
    if not (3 <= len(sprites) <= 5):
        return False
    colors = [sprite["color"] for sprite in sprites]
    if len(colors) != len(set(colors)):
        return False
    if any(not (3 <= sprite["area"] <= 7) for sprite in sprites):
        return False

    sprite_coords = [sprite["coords"] for sprite in sprites]
    for i in range(len(sprite_coords)):
        for j in range(i + 1, len(sprite_coords)):
            if not _sprite_distance_ok(sprite_coords[i], [sprite_coords[j]]):
                return False

    for r, c in zip(*np.where(grid == GREY), strict=False):
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in {BLACK, GREY}:
                return False

    _, comps = _scene_summary(grid)
    sprite_counts = sorted(len(comp["sprite_ids"]) for comp in comps if comp["sprite_ids"])
    if not sprite_counts or sprite_counts[-1] < 2 or sprite_counts[0] != 1:
        return False
    if sum(count >= 2 for count in sprite_counts) != 1:
        return False

    out = main(grid)
    if out.shape != grid.shape or np.any(out == GREY) or np.any(out == BLUE):
        return False
    return int(np.count_nonzero(out)) > 0


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    n_sprites = random.randint(*GEN_SPRITE_COUNT_RANGE)
    areas = [random.randint(*GEN_SPRITE_AREA_RANGE) for _ in range(n_sprites)]
    if random.random() < 0.35:
        tie_area = random.randint(*GEN_SPRITE_AREA_RANGE)
        areas[0] = tie_area
        areas[1] = tie_area

    colors = random.sample(SPRITE_COLORS, n_sprites)
    patches = [_shape_to_patch(_random_polyomino(area), color) for area, color in zip(areas, colors, strict=False)]

    grid = np.zeros((h, w), dtype=int)
    bridges, isolated_zones = _bridge_and_isolated_zones(h, w)
    random.shuffle(isolated_zones)

    placed: list[list[tuple[int, int]]] = []
    sprite_sets: list[set[tuple[int, int]]] = []

    for idx in range(2):
        coords = _place_patch_in_zone(grid, patches[idx], bridges[idx], placed)
        if coords is None:
            return None
        placed.append(coords)
        sprite_sets.append(set(coords))

    for idx in range(2, n_sprites):
        zone = isolated_zones[idx - 2]
        coords = _place_patch_in_zone(grid, patches[idx], zone, placed)
        if coords is None:
            return None
        placed.append(coords)
        sprite_sets.append(set(coords))

    path = _find_marker_chain(grid, sprite_sets, 0, 1)
    if path is None:
        return None
    for cell in path:
        grid[cell] = GREY

    _add_distractor_markers(grid, sprite_sets, set(path))

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
