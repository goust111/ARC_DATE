from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
SEED_COLORS = (BLUE, RED, GREEN)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = ORTHO_DIRS
    else:
        dirs = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _seed_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    seeds: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if int(grid[r, c]) in SEED_COLORS:
                seeds.append((r, c))
    return seeds


def _nonblack_component8_from(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(grid[nr, nc]) == BLACK or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _cavity_records(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    cavities: list[dict[str, object]] = []
    for coords in _components_mask(grid == BLACK, connectivity=4):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        region = {tuple(map(int, rc)) for rc in coords}
        boundary: set[tuple[int, int]] = set()
        valid = True
        for r, c in region:
            for dr, dc in ORTHO_DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in region:
                    continue
                if not _in_bounds(grid.shape, nr, nc):
                    valid = False
                    break
                boundary.add((nr, nc))
                if int(grid[nr, nc]) != GREY:
                    valid = False
                    break
            if not valid:
                break
        if valid and boundary:
            cavities.append({"coords": coords, "region": region, "boundary": boundary})
    return cavities


def _supports_cavity(boundary: set[tuple[int, int]], component: set[tuple[int, int]]) -> bool:
    if boundary & component:
        return True
    for r, c in boundary:
        for dr, dc in ORTHO_DIRS:
            if (r + dr, c + dc) in component:
                return True
    return False


def _fill_stage(grid: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray], list[dict[str, object]], dict[int, set[tuple[int, int]]]]:
    out = np.array(grid, dtype=int, copy=True)
    seeds = _seed_positions(out)
    assert 2 <= len(seeds) <= 3, "expected 2-3 border seeds"

    components_by_color: dict[int, set[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()
    for seed in seeds:
        color = int(out[seed])
        assert color not in components_by_color, "seed colors must be unique"
        comp = _nonblack_component8_from(out, seed)
        assert not (occupied & comp), "seed components must stay disjoint"
        occupied |= comp
        components_by_color[color] = comp

    cavities = _cavity_records(out)
    filled_lists: dict[int, list[tuple[int, int]]] = {color: [] for color in components_by_color}
    for cavity in cavities:
        boundary = cavity["boundary"]
        coords = cavity["coords"]
        supporters = [color for color, comp in components_by_color.items() if _supports_cavity(boundary, comp)]
        assert len(supporters) == 1, "each cavity must belong to exactly one seed component"
        fill_color = supporters[0]
        out[coords[:, 0], coords[:, 1]] = fill_color
        filled_lists[fill_color].extend((int(r), int(c)) for r, c in coords)

    filled_by_color = {
        color: np.array(cells, dtype=int)
        for color, cells in filled_lists.items()
        if cells
    }
    return out, filled_by_color, cavities, components_by_color


def _flip_filled_regions(grid: np.ndarray, filled_by_color: dict[int, np.ndarray]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for color, coords in filled_by_color.items():
        r0, _c0, r1, _c1 = _bbox_coords(coords)
        out[coords[:, 0], coords[:, 1]] = BLACK
        rr = r0 + r1 - coords[:, 0]
        cc = coords[:, 1]
        for nr, nc in zip(rr, cc):
            if int(out[nr, nc]) == BLACK:
                out[nr, nc] = color
    return out


def _mark_diagonals(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    src = np.argwhere(~np.isin(grid, [BLACK, GREY, ORANGE]))
    h, w = grid.shape
    for r, c in src:
        for dr, dc in DIAG_DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BLACK:
                out[nr, nc] = ORANGE
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def main(grid: np.ndarray) -> np.ndarray:
    """Fill grey-enclosed cavities from border seeds, flip each filled color mask vertically in its own bbox, then mark black diagonal neighbors orange.

    Input structure:
    - A black background contains 3-5 enclosed black cavities bounded by grey line drawings.
    - Two or three unique border seed pixels use colors from blue, red, and green.
    - Each seed touches exactly one disconnected non-black drawing component, and each cavity belongs to exactly one such seed component.

    Rule:
    - Find every enclosed black cavity whose 4-neighbor boundary is entirely grey.
    - Assign each cavity to the unique seed whose 8-connected non-black component touches that cavity boundary, and fill the cavity with that seed color.
    - For each fill color independently, vertically reflect only the filled cavity pixels inside that color's tight bbox; grey stays fixed and any reflected pixel landing on grey is dropped.
    - After all flips, for every non-black, non-grey, non-orange pixel, recolor each black diagonal neighbor orange.

    Key insight:
    - The flood-fill step creates the color groups, those groups define the per-color flip bboxes, and the flipped shapes then determine a different diagonal-decoration pattern than the unflipped fill would produce.
    """
    g = np.asarray(grid, dtype=int)
    filled, filled_by_color, _cavities, _components = _fill_stage(g)
    flipped = _flip_filled_regions(filled, filled_by_color)
    return _mark_diagonals(flipped)


ROOM_GAP = 2
GEN_ATTEMPTS = 400


def _room_patch(height: int, width: int) -> np.ndarray | None:
    ih, iw = height - 2, width - 2
    for _ in range(60):
        patch = np.zeros((height, width), dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY

        side = random.choice(("top", "bottom"))
        max_depth = max(1, min(2, ih - 1))
        depth = random.randint(1, max_depth)
        max_span = iw if iw == 1 else iw - 1
        span = random.randint(1, max_span)
        offset = random.randint(0, iw - span)
        if side == "top":
            patch[1 : 1 + depth, 1 + offset : 1 + offset + span] = GREY
        else:
            patch[height - 1 - depth : height - 1, 1 + offset : 1 + offset + span] = GREY

        if ih >= 3 and iw >= 3 and random.random() < 0.5:
            side2 = random.choice(("left", "right"))
            vspan = random.randint(1, ih - 1)
            voff = random.randint(0, ih - vspan)
            if side2 == "left":
                patch[1 + voff : 1 + voff + vspan, 1] = GREY
            else:
                patch[1 + voff : 1 + voff + vspan, width - 2] = GREY

        black_components = _components_mask(patch == BLACK, connectivity=4)
        if len(black_components) != 1:
            continue
        cavity_mask = patch[1:-1, 1:-1] == BLACK
        if int(cavity_mask.sum()) < 3:
            continue
        if np.array_equal(cavity_mask, np.flipud(cavity_mask)):
            continue
        return patch
    return None


def _sample_component_counts(component_count: int) -> list[int]:
    if component_count == 2:
        return list(random.choice(((2, 1), (1, 2), (2, 2))))
    return list(random.choices(((2, 1, 1), (2, 2, 1), (1, 1, 1)), weights=(4, 3, 1), k=1)[0])


def _construct_attempt() -> np.ndarray | None:
    component_count = random.choice((2, 3))
    colors = random.sample(list(SEED_COLORS), component_count)
    cavity_counts = _sample_component_counts(component_count)
    random.shuffle(cavity_counts)

    room_widths = [random.randint(5, 6) for _ in range(component_count)]
    band_widths = [width + 1 for width in room_widths]
    col_gap = random.randint(2, 3)
    total_width = 2 + sum(band_widths) + col_gap * (component_count - 1)
    if total_width > 30:
        return None

    room_heights = [[random.randint(5, 7) for _ in range(count)] for count in cavity_counts]
    stack_heights = [sum(heights) + ROOM_GAP * (len(heights) - 1) for heights in room_heights]
    min_height = max(stack_heights) + 4
    if min_height > 30:
        return None
    total_height = random.randint(min_height, min(30, min_height + 6))

    grid = np.zeros((total_height, total_width), dtype=int)
    band_start = 1
    for idx in range(component_count):
        color = int(colors[idx])
        room_width = room_widths[idx]
        room_c0 = band_start + 1
        corridor_col = band_start
        side = random.choice(("top", "bottom"))
        heights = room_heights[idx]
        stack_height = stack_heights[idx]
        top = random.randint(2, total_height - stack_height - 2)

        room_rows: list[int] = []
        cursor = top
        for h in heights:
            room_rows.append(cursor)
            cursor += h + ROOM_GAP

        for r0, room_height in zip(room_rows, heights):
            patch = _room_patch(room_height, room_width)
            if patch is None:
                return None
            view = grid[r0 : r0 + room_height, room_c0 : room_c0 + room_width]
            if np.any((view != BLACK) & (patch != BLACK)):
                return None
            view[patch != BLACK] = patch[patch != BLACK]

        first_top = room_rows[0]
        last_bottom = room_rows[-1] + heights[-1] - 1
        if side == "top":
            grid[1 : last_bottom + 1, corridor_col] = GREY
            grid[0, corridor_col] = color
        else:
            grid[first_top : total_height - 1, corridor_col] = GREY
            grid[total_height - 1, corridor_col] = color

        band_start += band_widths[idx] + col_gap

    if random.random() < 0.5:
        grid = grid.T.copy()
    if random.random() < 0.5:
        grid = np.flipud(grid).copy()
    if random.random() < 0.5:
        grid = np.fliplr(grid).copy()
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        filled, filled_by_color, cavities, components_by_color = _fill_stage(grid)
    except AssertionError:
        return False

    seeds = _seed_positions(grid)
    if not (2 <= len(seeds) <= 3):
        return False
    if len({int(grid[r, c]) for r, c in seeds}) != len(seeds):
        return False
    if not (3 <= len(cavities) <= 5):
        return False

    supporting_colors: list[int] = []
    for cavity in cavities:
        supporters = [color for color, comp in components_by_color.items() if _supports_cavity(cavity["boundary"], comp)]
        if len(supporters) != 1:
            return False
        supporting_colors.append(supporters[0])
    if len(set(supporting_colors)) < 2:
        return False

    color_boxes: list[tuple[int, int, int, int]] = []
    for color in sorted(filled_by_color):
        coords = filled_by_color[color]
        box = _bbox_coords(coords)
        if any(_boxes_overlap(box, other) for other in color_boxes):
            return False
        color_boxes.append(box)

    flipped = _flip_filled_regions(filled, filled_by_color)
    final = _mark_diagonals(flipped)
    preflip_marked = _mark_diagonals(filled)

    if np.array_equal(filled, grid):
        return False
    if np.array_equal(flipped, filled):
        return False
    if np.array_equal(final, flipped):
        return False
    if np.array_equal(final, preflip_marked):
        return False
    if int(np.count_nonzero(final == ORANGE)) == 0:
        return False

    for color, coords in filled_by_color.items():
        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        comps = _components_mask(mask, connectivity=4)
        if len(comps) > 1:
            break
    else:
        if len(filled_by_color) >= 3:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _construct_attempt()
        if grid is None:
            continue
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate valid input")
