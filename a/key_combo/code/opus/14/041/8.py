from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
FOREGROUND = (RED, GREEN)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 500
ODD_SIZES = [15, 17, 19]
EVEN_OCCLUDERS = [(2, 2), (2, 3), (3, 2), (1, 4), (4, 1)]
SMALL_AREA_CHOICES = [2, 3, 4, 4]


def _rot_coord(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    return h - 1 - int(r), w - 1 - int(c)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _restore_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    restored = g.copy()
    blue = np.argwhere(g == BLUE)
    for r, c in blue:
        rr, cc = _rot_coord(g.shape, int(r), int(c))
        restored[int(r), int(c)] = int(g[rr, cc])
    return restored


def _erase_large_components(restored: np.ndarray, threshold: int) -> np.ndarray:
    out = restored.copy()
    fg_mask = np.isin(restored, FOREGROUND)
    for comp in _components_from_mask(fg_mask):
        if int(comp.shape[0]) >= int(threshold):
            out[comp[:, 0], comp[:, 1]] = BLACK
    return out


def _blue_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = grid == BLUE
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components_from_mask(mask):
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
        if area != int(comp.shape[0]):
            raise AssertionError("blue occluders must be solid rectangles")
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the pixels hidden by blue rectangles using 180-degree rotation, then erase restored foreground components whose area reaches the blue-area threshold.

    Input structure:
    - The grid has a black background plus a red/green pattern that would be 180-degree rotationally symmetric if not for blue rectangular occluders.
    - Two or three solid blue rectangles cover part of that pattern, and each blue cell's 180-degree counterpart is visible elsewhere.
    - Red and green are foreground colors; their 4-neighbor adjacencies define connected components after restoration.

    Rule:
    - Count all blue pixels to get K, the total occluder area.
    - Replace every blue cell by the color at its 180-degree rotated counterpart in the input grid, thereby restoring the hidden red/green pattern.
    - On the restored grid, find every 4-connected foreground component formed by red and green pixels together.
    - Erase every component with area at least K by turning all of its cells black; keep smaller components unchanged.

    Key insight:
    - The blue rectangles encode two linked facts at once: they mark exactly which cells must be recovered from the half-turn counterpart, and their total area is the size threshold that decides which restored components survive.
    """
    g = np.asarray(grid, dtype=int)
    restored = _restore_grid(g)
    threshold = int(np.count_nonzero(g == BLUE))
    return _erase_large_components(restored, threshold)


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        patch[r, c] = True
    patch = np.rot90(patch, random.randint(0, 3))
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _path_candidates(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    h: int,
    w: int,
    center: tuple[int, int],
) -> list[tuple[int, int]]:
    cr, cc = center
    r, c = path[-1]
    candidates: list[tuple[int, int]] = []
    dirs = DIR4[:]
    random.shuffle(dirs)
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if not (1 <= nr < h - 1 and 1 <= nc < cc):
            continue
        if (nr, nc) in used or (nr, nc) == center:
            continue
        if abs(nr - cr) + abs(nc - cc) == 1 and (nr, nc) != (cr, cc - 1):
            continue
        if any((nr + rr, nc + cc2) in used and (nr + rr, nc + cc2) != path[-1] for rr, cc2 in DIR4):
            continue
        candidates.append((nr, nc))
    return candidates


def _build_main_component(h: int, w: int, total_blue_area: int, bridge_style: str) -> set[tuple[int, int]] | None:
    cr, cc = h // 2, w // 2
    left_len = (total_blue_area - 1) // 2
    if bridge_style == "vertical":
        path = [(cr, cc - 1), (cr - 1, cc - 1)]
    else:
        path = [(cr, cc - 1), (cr, cc - 2)]
    if left_len < len(path):
        return None
    used = set(path)

    def dfs() -> bool:
        if len(path) == left_len:
            return True
        for nxt in _path_candidates(path, used, h, w, (cr, cc)):
            path.append(nxt)
            used.add(nxt)
            if dfs():
                return True
            used.remove(nxt)
            path.pop()
        return False

    if not dfs():
        return None

    main = {(cr, cc), *path}
    for r, c in path:
        main.add(_rot_coord((h, w), r, c))
    return main


def _coords_touch(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r, c) in b:
            return True
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _place_small_pair(
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    area: int,
    quadrant: str,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    patch = _random_polyomino(area)
    pr, pc = np.argwhere(patch).T
    shape = list(zip(pr.tolist(), pc.tolist()))
    ph, pw = patch.shape
    cr, cc = h // 2, w // 2

    if quadrant == "nw":
        row_range = range(1, cr - ph - 1)
        col_range = range(1, cc - pw - 2)
    else:
        row_range = range(1, cr - ph - 1)
        col_range = range(cc + 2, w - pw - 1)

    positions = [(r0, c0) for r0 in row_range for c0 in col_range]
    random.shuffle(positions)
    for r0, c0 in positions:
        seed = {(r0 + dr, c0 + dc) for dr, dc in shape}
        pair = {_rot_coord((h, w), r, c) for r, c in seed}
        all_cells = seed | pair
        if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
            continue
        if len(all_cells) != 2 * len(seed):
            continue
        if _coords_touch(all_cells, occupied):
            continue
        if _coords_touch(seed, pair):
            continue
        return seed, pair
    return None


def _assign_colors(coords: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    ordered = sorted(coords)
    seen: set[tuple[int, int]] = set()
    pair_reps: list[tuple[int, int]] = []
    for cell in ordered:
        if cell in seen:
            continue
        rr, cc = _rot_coord((h, w), cell[0], cell[1])
        seen.add(cell)
        seen.add((rr, cc))
        pair_reps.append(cell)
        color = random.choice(FOREGROUND)
        grid[cell] = color
        grid[rr, cc] = color
    if not np.any(grid == RED):
        r, c = pair_reps[0]
        rr, cc = _rot_coord((h, w), r, c)
        grid[r, c] = grid[rr, cc] = RED
    if not np.any(grid == GREEN):
        r, c = pair_reps[-1]
        rr, cc = _rot_coord((h, w), r, c)
        grid[r, c] = grid[rr, cc] = GREEN
    return grid


def _rect_cells(r0: int, c0: int, rh: int, rw: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}


def _can_add_blue_rect(
    cells: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    rot_cells = {_rot_coord((h, w), r, c) for r, c in cells}
    if cells & rot_cells:
        return False
    if cells & blue_cells or rot_cells & blue_cells:
        return False
    for r, c in cells:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in blue_cells:
                return False
    return True


def _find_rect_positions(
    pattern: np.ndarray,
    blue_cells: set[tuple[int, int]],
    rect_shape: tuple[int, int],
    target_cells: set[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    h, w = pattern.shape
    rh, rw = rect_shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - rh + 1):
        for c0 in range(w - rw + 1):
            cells = _rect_cells(r0, c0, rh, rw)
            if not _can_add_blue_rect(cells, blue_cells, h, w):
                continue
            if not any(pattern[r, c] in FOREGROUND for r, c in cells):
                continue
            if target_cells is not None and not any((r, c) in target_cells for r, c in cells):
                continue
            positions.append((r0, c0))
    return positions


def _semantic_check(inp: np.ndarray, restored: np.ndarray, expected: np.ndarray, main_area: int) -> bool:
    try:
        boxes = _blue_rectangles(inp)
    except AssertionError:
        return False
    if len(boxes) not in (2, 3):
        return False

    k = int(np.count_nonzero(inp == BLUE))
    if main_area != k:
        return False
    if not np.array_equal(restored, _restore_grid(inp)):
        return False
    if not np.array_equal(restored, np.rot90(restored, 2)):
        return False
    if np.any(restored == BLUE):
        return False

    restored_comps = _components_from_mask(np.isin(restored, FOREGROUND))
    restored_areas = sorted(int(comp.shape[0]) for comp in restored_comps)
    if k not in restored_areas:
        return False
    if not any(area < k for area in restored_areas):
        return False

    visible = np.where(inp == BLUE, BLACK, inp)
    visible_areas = sorted(int(comp.shape[0]) for comp in _components_from_mask(np.isin(visible, FOREGROUND)))
    if visible_areas == restored_areas:
        return False

    for r, c in np.argwhere(inp == BLUE):
        rr, cc = _rot_coord(inp.shape, int(r), int(c))
        if inp[rr, cc] == BLUE:
            return False
        if restored[r, c] != inp[rr, cc]:
            return False

    if np.array_equal(expected, restored):
        return False
    if np.array_equal(expected, _erase_large_components(restored, k + 1)):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.choice(ODD_SIZES)
    w = random.choice(ODD_SIZES)
    bridge_style = random.choice(["vertical", "horizontal"])
    rect_shapes = [(3, 1) if bridge_style == "vertical" else (1, 3)]
    rect_count = random.choice([2, 3])
    rect_shapes.extend(random.sample(EVEN_OCCLUDERS, rect_count - 1))
    total_blue_area = sum(rh * rw for rh, rw in rect_shapes)

    main_coords = _build_main_component(h, w, total_blue_area, bridge_style)
    if main_coords is None:
        return None

    occupied = set(main_coords)
    components: list[set[tuple[int, int]]] = [set(main_coords)]
    small_union: set[tuple[int, int]] = set()
    small_pair_count = 1 if random.random() < 0.6 else 2
    quadrants = ["nw", "ne"]
    random.shuffle(quadrants)
    for i in range(small_pair_count):
        placed = _place_small_pair(occupied, h, w, random.choice(SMALL_AREA_CHOICES), quadrants[i % 2])
        if placed is None:
            if i == 0:
                return None
            break
        a, b = placed
        occupied |= a | b
        components.extend([a, b])
        small_union |= a | b

    pattern = _assign_colors(occupied, h, w)

    cr, cc = h // 2, w // 2
    if bridge_style == "vertical":
        first_pos = (cr - 1, cc - 1)
    else:
        first_pos = (cr, cc - 3)
    blue_boxes: list[tuple[int, int, int, int]] = []
    blue_cells: set[tuple[int, int]] = set()

    for idx, (rh, rw) in enumerate(rect_shapes):
        if idx == 0:
            r0, c0 = first_pos
        else:
            target = main_coords
            if idx == len(rect_shapes) - 1 and small_union and random.random() < 0.4:
                target = small_union
            positions = _find_rect_positions(pattern, blue_cells, (rh, rw), target)
            if not positions:
                positions = _find_rect_positions(pattern, blue_cells, (rh, rw), None)
            if not positions:
                return None
            r0, c0 = random.choice(positions)
        cells = _rect_cells(r0, c0, rh, rw)
        if not _can_add_blue_rect(cells, blue_cells, h, w):
            return None
        blue_cells |= cells
        blue_boxes.append((r0, c0, r0 + rh - 1, c0 + rw - 1))

    inp = pattern.copy()
    for r0, c0, r1, c1 in blue_boxes:
        inp[r0 : r1 + 1, c0 : c1 + 1] = BLUE

    restored = pattern
    expected = _erase_large_components(restored, total_blue_area)
    if not _semantic_check(inp, restored, expected, len(main_coords)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
