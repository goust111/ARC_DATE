from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

STAMP_ORDER = [RED, ORANGE, YELLOW, GREEN, AZURE, BLUE, MAGENTA]
GEN_RETRIES = 400
MAX_TOTAL_WIDTH = 30
MAX_TOTAL_HEIGHT = 14


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(coords, dtype=int)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def _border_mask(h: int, w: int) -> int:
    mask = 0
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                mask |= 1 << (r * w + c)
    return mask


def _placement_masks(shape: np.ndarray, h: int, w: int) -> list[int]:
    sh, sw = _shape_dims(shape)
    out: list[int] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            mask = 0
            for r, c in shape:
                mask |= 1 << ((r0 + int(r)) * w + (c0 + int(c)))
            out.append(mask)
    return out


def _reconstruct_holes(fragments: list[np.ndarray]) -> np.ndarray:
    shapes = [_normalize(fragment) for fragment in fragments]
    total_area = sum(int(shape.shape[0]) for shape in shapes)
    min_h = max(3, max(_shape_dims(shape)[0] for shape in shapes))
    min_w = max(3, max(_shape_dims(shape)[1] for shape in shapes))

    solutions: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    for h in range(min_h, total_area + 1):
        for w in range(min_w, total_area + 1):
            rect_area = h * w
            hole_count = rect_area - total_area
            if hole_count < 1:
                continue
            if hole_count > (h - 2) * (w - 2):
                continue
            if 2 * h + 2 * w - 4 > total_area:
                continue

            placements = [_placement_masks(shape, h, w) for shape in shapes]
            if any(not masks for masks in placements):
                continue

            order = sorted(range(len(shapes)), key=lambda idx: (-len(shapes[idx]), len(placements[idx])))
            remaining = [0] * (len(order) + 1)
            for i in range(len(order) - 1, -1, -1):
                remaining[i] = remaining[i + 1] + len(shapes[order[i]])

            border = _border_mask(h, w)
            unions: set[int] = set()

            def dfs(pos: int, occupied: int) -> None:
                if len(unions) > 1:
                    return
                missing_border = border & ~occupied
                if missing_border.bit_count() > remaining[pos]:
                    return
                if pos == len(order):
                    if (occupied & border) == border:
                        unions.add(occupied)
                    return
                frag_idx = order[pos]
                for placement in placements[frag_idx]:
                    if occupied & placement:
                        continue
                    dfs(pos + 1, occupied | placement)
                    if len(unions) > 1:
                        return

            dfs(0, 0)
            for occupied in unions:
                key = (h, w, occupied)
                if key not in seen:
                    seen.add(key)
                    solutions.append(key)
                    if len(solutions) > 1:
                        break
            if len(solutions) > 1:
                break
        if len(solutions) > 1:
            break

    if len(solutions) != 1:
        raise ValueError(f"expected unique rectangle reconstruction, got {len(solutions)}")

    h, w, occupied = solutions[0]
    holes = np.ones((h, w), dtype=bool)
    for idx in range(h * w):
        if (occupied >> idx) & 1:
            holes[idx // w, idx % w] = False
    return holes


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if len(cols) != 1:
        raise ValueError("expected exactly one grey separator column")
    return int(cols[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-color fragments into holey rectangles, then stamp the holes onto the right canvas.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    marker_pts = np.argwhere(right == RED)
    if marker_pts.shape[0] != 1:
        raise ValueError("expected exactly one red start marker on the right canvas")
    start_r, start_c = map(int, marker_pts[0])

    out = np.full(right.shape, BG, dtype=int)
    cursor_c = start_c

    colors_present = {int(v) for v in np.unique(left) if int(v) != BG}
    ordered_colors = [color for color in STAMP_ORDER if color in colors_present]

    for color in ordered_colors:
        mask = left == color
        fragments = [_normalize(comp) for comp in _components(mask)]
        holes = _reconstruct_holes(fragments)
        hole_rows, hole_cols = np.where(holes)
        h, w = holes.shape
        if start_r + h > out.shape[0] or cursor_c + w > out.shape[1]:
            raise ValueError("stamp does not fit inside output canvas")
        out[start_r + hole_rows, cursor_c + hole_cols] = color
        cursor_c += w + 1

    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _random_hole_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if len(interior) < 2:
        return None

    max_holes = min(5, len(interior) - 1)
    target_holes = rng.randint(1, max_holes)
    seed_count = rng.randint(1, min(3, target_holes))

    hole_cells = set(rng.sample(interior, seed_count))
    frontier = set()
    for cell in hole_cells:
        for nbr in _neighbors4(*cell):
            if nbr in interior and nbr not in hole_cells:
                frontier.add(nbr)

    while len(hole_cells) < target_holes:
        if frontier and rng.random() < 0.8:
            cell = rng.choice(tuple(frontier))
        else:
            remaining = [cell for cell in interior if cell not in hole_cells]
            if not remaining:
                break
            cell = rng.choice(remaining)
        hole_cells.add(cell)
        frontier.discard(cell)
        for nbr in _neighbors4(*cell):
            if nbr in interior and nbr not in hole_cells:
                frontier.add(nbr)

    hole_mask = np.zeros((h, w), dtype=bool)
    for r, c in hole_cells:
        hole_mask[r, c] = True

    filled = ~hole_mask
    if not _mask_is_connected(filled):
        return None
    return hole_mask


def _partition_connected(mask: np.ndarray, rng: random.Random) -> list[np.ndarray] | None:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    area = len(cells)
    max_parts = min(4, max(2, area // 3))
    if max_parts < 2:
        return None

    for _ in range(80):
        part_count = rng.randint(2, max_parts)
        seeds = rng.sample(cells, part_count)
        parts = [set([seed]) for seed in seeds]
        assigned = set(seeds)
        unassigned = set(cells) - assigned

        while unassigned:
            active: list[int] = []
            frontiers: list[list[tuple[int, int]]] = []
            for idx, part in enumerate(parts):
                frontier = []
                for cell in part:
                    for nbr in _neighbors4(*cell):
                        if nbr in unassigned:
                            frontier.append(nbr)
                frontier = list(dict.fromkeys(frontier))
                frontiers.append(frontier)
                if frontier:
                    active.append(idx)
            if not active:
                break
            idx = rng.choice(active)
            cell = rng.choice(frontiers[idx])
            parts[idx].add(cell)
            assigned.add(cell)
            unassigned.remove(cell)

        if unassigned:
            continue
        if min(len(part) for part in parts) < 2:
            continue
        return [_normalize(np.array(sorted(part), dtype=int)) for part in parts]
    return None


def _shape_spec(rng: random.Random, color: int) -> dict[str, object] | None:
    for _ in range(120):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        holes = _random_hole_mask(rng, h, w)
        if holes is None:
            continue
        fragments = _partition_connected(~holes, rng)
        if fragments is None:
            continue
        try:
            solved = _reconstruct_holes(fragments)
        except ValueError:
            continue
        if solved.shape != holes.shape or not np.array_equal(solved, holes):
            continue
        return {"color": color, "holes": holes, "fragments": fragments}
    return None


def _place_fragments(
    rng: random.Random,
    specs: list[dict[str, object]],
    height: int,
    left_w: int,
) -> np.ndarray | None:
    canvas = np.full((height, left_w), BG, dtype=int)
    occupied = np.zeros((height, left_w), dtype=bool)

    items: list[tuple[int, np.ndarray]] = []
    for spec in specs:
        color = int(spec["color"])
        for fragment in spec["fragments"]:
            items.append((color, np.asarray(fragment, dtype=int)))

    rng.shuffle(items)
    items.sort(key=lambda item: -len(item[1]))

    for color, fragment in items:
        frag_h, frag_w = _shape_dims(fragment)
        candidates: list[tuple[int, int]] = []
        for r0 in range(height - frag_h + 1):
            for c0 in range(left_w - frag_w + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(height, r0 + frag_h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(left_w, c0 + frag_w + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        rows = r0 + fragment[:, 0]
        cols = c0 + fragment[:, 1]
        canvas[rows, cols] = color
        occupied[rows, cols] = True
    return canvas


def _expected_output(specs: list[dict[str, object]], height: int, width: int) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    cursor = 0
    spec_by_color = {int(spec["color"]): spec for spec in specs}
    for color in STAMP_ORDER:
        if color not in spec_by_color:
            continue
        holes = np.asarray(spec_by_color[color]["holes"], dtype=bool)
        rows, cols = np.where(holes)
        out[rows, cursor + cols] = color
        cursor += holes.shape[1] + 1
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(1, 3)
    chosen_colors = sorted(rng.sample(STAMP_ORDER, color_count), key=STAMP_ORDER.index)

    specs: list[dict[str, object]] = []
    for color in chosen_colors:
        spec = _shape_spec(rng, color)
        if spec is None:
            return None
        specs.append(spec)

    canvas_w = sum(int(np.asarray(spec["holes"]).shape[1]) for spec in specs) + len(specs) - 1
    required_h = max(int(np.asarray(spec["holes"]).shape[0]) for spec in specs)
    if canvas_w > 18:
        return None

    height = rng.randint(required_h, min(MAX_TOTAL_HEIGHT, required_h + 4))
    fragment_width_sum = sum(_shape_dims(np.asarray(fragment))[1] for spec in specs for fragment in spec["fragments"])
    left_w_min = max(8, max(_shape_dims(np.asarray(fragment))[1] for spec in specs for fragment in spec["fragments"]))
    left_w_max = min(14, MAX_TOTAL_WIDTH - canvas_w - 1)
    if left_w_min > left_w_max:
        return None

    left_w_choices = list(range(left_w_min, left_w_max + 1))
    rng.shuffle(left_w_choices)
    left_grid = None
    left_w = None
    for candidate_w in left_w_choices:
        if candidate_w < min(14, fragment_width_sum // 2):
            continue
        placed = _place_fragments(rng, specs, height, candidate_w)
        if placed is not None:
            left_grid = placed
            left_w = candidate_w
            break
    if left_grid is None or left_w is None:
        return None

    grid = np.full((height, left_w + 1 + canvas_w), BG, dtype=int)
    grid[:, :left_w] = left_grid
    grid[:, left_w] = GREY
    grid[0, left_w + 1] = RED

    expected = _expected_output(specs, height, canvas_w)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
