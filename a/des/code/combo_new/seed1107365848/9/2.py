from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
GROUP_BUILD_ATTEMPTS = 200
PARTITION_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 60


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    r0, c0, _, _ = _bbox(pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _find_frame_for_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(np.any(grid == color, axis=1))
    cols = np.flatnonzero(np.any(grid == color, axis=0))
    if len(rows) < 3 or len(cols) < 3:
        return None

    best: tuple[int, int, int, int] | None = None
    best_area = -1
    for r0_i, r0 in enumerate(rows[:-2]):
        for r1 in rows[r0_i + 1 :]:
            if r1 - r0 < 2:
                continue
            for c0_i, c0 in enumerate(cols[:-2]):
                for c1 in cols[c0_i + 1 :]:
                    if c1 - c0 < 2:
                        continue
                    if grid[r0, c0] != color or grid[r0, c1] != color or grid[r1, c0] != color or grid[r1, c1] != color:
                        continue
                    if not np.all(grid[r0, c0 : c1 + 1] == color):
                        continue
                    if not np.all(grid[r1, c0 : c1 + 1] == color):
                        continue
                    if not np.all(grid[r0 : r1 + 1, c0] == color):
                        continue
                    if not np.all(grid[r0 : r1 + 1, c1] == color):
                        continue
                    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                    if interior.size == 0 or not np.any(interior != color):
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area > best_area:
                        best_area = area
                        best = (r0, c0, r1, c1)
    return best


def _solve_translation_cover(
    inner_h: int,
    inner_w: int,
    anchors: set[tuple[int, int]],
    pieces: list[tuple[tuple[int, int], ...]],
    limit: int = 1,
) -> tuple[int, list[tuple[tuple[int, int], ...]] | None]:
    target = {(r, c) for r in range(inner_h) for c in range(inner_w)} - anchors
    if sum(len(piece) for piece in pieces) != len(target):
        return 0, None

    placements_by_piece: list[list[tuple[tuple[int, int], ...]]] = []
    cell_to_options: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    for piece_index, piece in enumerate(pieces):
        ph, pw = _shape_dims(piece)
        seen = set()
        options: list[tuple[tuple[int, int], ...]] = []
        for dr in range(inner_h - ph + 1):
            for dc in range(inner_w - pw + 1):
                placed = tuple(sorted((r + dr, c + dc) for r, c in piece))
                if any(cell in anchors for cell in placed):
                    continue
                if any(cell not in target for cell in placed):
                    continue
                if placed in seen:
                    continue
                seen.add(placed)
                options.append(placed)
        if not options:
            return 0, None
        placements_by_piece.append(options)
        for option_index, option in enumerate(options):
            for cell in option:
                cell_to_options[cell].append((piece_index, option_index))

    uncovered = set(target)
    used = [False] * len(pieces)
    assignment: list[tuple[tuple[int, int], ...] | None] = [None] * len(pieces)
    solution_count = 0
    first_solution: list[tuple[tuple[int, int], ...]] | None = None

    def search() -> None:
        nonlocal solution_count, first_solution
        if solution_count >= limit:
            return
        if not uncovered:
            solution_count += 1
            if first_solution is None:
                first_solution = [option for option in assignment if option is not None]
            return

        best_cell: tuple[int, int] | None = None
        best_options: list[tuple[int, int]] | None = None
        for cell in uncovered:
            current: list[tuple[int, int]] = []
            for piece_index, option_index in cell_to_options[cell]:
                if used[piece_index]:
                    continue
                option = placements_by_piece[piece_index][option_index]
                if all(pt in uncovered for pt in option):
                    current.append((piece_index, option_index))
            if not current:
                return
            if best_options is None or len(current) < len(best_options):
                best_cell = cell
                best_options = current
                if len(best_options) == 1:
                    break

        assert best_cell is not None and best_options is not None
        for piece_index, option_index in best_options:
            option = placements_by_piece[piece_index][option_index]
            used[piece_index] = True
            assignment[piece_index] = option
            removed = list(option)
            for cell in removed:
                uncovered.remove(cell)
            search()
            for cell in removed:
                uncovered.add(cell)
            assignment[piece_index] = None
            used[piece_index] = False
            if solution_count >= limit:
                return

    search()
    return solution_count, first_solution


def _solve_one_color(grid: np.ndarray, color: int, frame: tuple[int, int, int, int]) -> tuple[bool, list[list[tuple[int, int]]]]:
    r0, c0, r1, c1 = frame
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if inner_h <= 0 or inner_w <= 0:
        return False, []

    anchors = {
        (r - (r0 + 1), c - (c0 + 1))
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) == color
    }
    if any(int(grid[r, c]) not in (BG, color) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)):
        return False, []

    outside_mask = np.ones(grid.shape, dtype=bool)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    fragment_mask = (grid == color) & outside_mask
    fragments = _components(fragment_mask)
    pieces = [_normalize(fragment) for fragment in fragments]
    count, _ = _solve_translation_cover(inner_h, inner_w, anchors, pieces, limit=1)
    return count == 1, fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each color's scattered fragments into the matching hollow frame so the anchored pieces fill its interior.

    Input structure:
    - The background is black.
    - Several colors each form one hollow rectangular frame with a black interior.
    - Inside every frame there are one or two same-color anchor pixels already placed.
    - Outside the frames, the remaining same-color pixels appear as small 4-connected fragments.

    Rule:
    - For each color, locate its rectangular frame and read the same-color anchor pixels inside it.
    - Collect every outside fragment of that color and translate the fragments, without rotation or reflection,
      into the frame interior so that the fragments plus anchors cover every interior cell exactly once.
    - After reassembly, erase the fragments from their original outside positions.
    - Keep the frame borders unchanged and fill the full frame interior with the frame's color.

    Key insight:
    - Frame color and fragment color are the matching key, while the pre-drawn anchor pixels fix the only valid
      translation pattern for that color's pieces.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        frame = _find_frame_for_color(g, color)
        if frame is not None:
            frames[color] = frame

    assert frames, "expected at least one colored frame"

    for color, frame in frames.items():
        ok, fragments = _solve_one_color(g, color, frame)
        assert ok, f"color {color} fragments do not tile their frame interior by translation"
        for fragment in fragments:
            for r, c in fragment:
                out[r, c] = BG
        r0, c0, r1, c1 = frame
        out[r0 + 1 : r1, c0 + 1 : c1] = color

    return out


def _remaining_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = {start}
        remaining.remove(start)
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _connected_group_candidates(
    rng: random.Random,
    remaining: set[tuple[int, int]],
    start: tuple[int, int],
    size: int,
    tries: int = 20,
) -> list[set[tuple[int, int]]]:
    candidates: list[set[tuple[int, int]]] = []
    seen = set()
    for _ in range(tries):
        group = {start}
        frontier = {cell for cell in _neighbors4(*start) if cell in remaining}
        while len(group) < size and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            group.add(cell)
            for nxt in _neighbors4(*cell):
                if nxt in remaining and nxt not in group:
                    frontier.add(nxt)
        if len(group) != size:
            continue
        key = tuple(sorted(group))
        if key not in seen:
            seen.add(key)
            candidates.append(set(group))
    candidates.sort(key=lambda comp: tuple(sorted(comp)))
    return candidates


def _partition_free_cells(
    rng: random.Random,
    free_cells: set[tuple[int, int]],
    attempts: int = PARTITION_ATTEMPTS,
) -> list[set[tuple[int, int]]] | None:
    def rec(remaining: set[tuple[int, int]]) -> list[set[tuple[int, int]]] | None:
        if not remaining:
            return []
        start = min(remaining)
        size_choices = [size for size in range(2, 6) if size <= len(remaining)]
        rng.shuffle(size_choices)
        for size in size_choices:
            if len(remaining) - size == 1:
                continue
            for group in _connected_group_candidates(rng, remaining, start, size, tries=attempts):
                rest = remaining - group
                if any(len(comp) == 1 for comp in _remaining_components(rest)):
                    continue
                tail = rec(rest)
                if tail is not None:
                    return [group] + tail
        return None

    return rec(set(free_cells))


def _build_group(rng: random.Random) -> dict | None:
    for _ in range(GROUP_BUILD_ATTEMPTS):
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 6)
        area = inner_h * inner_w
        anchor_count = 1 if rng.random() < 0.5 else 2
        if area - anchor_count < 2:
            continue

        all_cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
        anchors = set(rng.sample(all_cells, anchor_count))
        free_cells = set(all_cells) - anchors
        pieces_abs = _partition_free_cells(rng, free_cells)
        if not pieces_abs:
            continue

        shapes = [_normalize(piece) for piece in pieces_abs]
        if len(set(shapes)) != len(shapes):
            continue

        solution_count, _ = _solve_translation_cover(inner_h, inner_w, anchors, shapes, limit=2)
        if solution_count != 1:
            continue

        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "anchors": tuple(sorted(anchors)),
            "shapes": tuple(shapes),
        }
    return None


def _can_place_rect(occ: np.ndarray, top: int, left: int, height: int, width: int, gap: int) -> bool:
    h, w = occ.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + height + gap)
    c1 = min(w, left + width + gap)
    return not np.any(occ[r0:r1, c0:c1])


def _occupy_rect(occ: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occ[top : top + height, left : left + width] = True


def _random_place_rect(
    rng: random.Random,
    occ: np.ndarray,
    height: int,
    width: int,
    gap: int,
) -> tuple[int, int] | None:
    h, w = occ.shape
    positions = [(r, c) for r in range(h - height + 1) for c in range(w - width + 1)]
    rng.shuffle(positions)
    for top, left in positions:
        if _can_place_rect(occ, top, left, height, width, gap):
            _occupy_rect(occ, top, left, height, width)
            return top, left
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _draw_shape(grid: np.ndarray, top: int, left: int, shape: tuple[tuple[int, int], ...], color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _attempt_layout(rng: random.Random, groups: list[dict], colors: list[int]) -> np.ndarray | None:
    frame_boxes = [(group["inner_h"] + 2) * (group["inner_w"] + 2) for group in groups]
    piece_boxes = []
    for group in groups:
        for shape in group["shapes"]:
            ph, pw = _shape_dims(shape)
            piece_boxes.append(ph * pw)
    total_box_area = sum(frame_boxes) + sum(piece_boxes)
    max_box_h = max([group["inner_h"] + 2 for group in groups] + [1])
    max_box_w = max([group["inner_w"] + 2 for group in groups] + [1])
    base = max(max_box_h + 4, max_box_w + 4, int(math.ceil(math.sqrt(total_box_area * 2.4))))
    if base > 30:
        return None
    grid_h = rng.randint(base, min(30, base + 6))
    grid_w = rng.randint(base, min(30, base + 6))

    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    frame_positions: list[tuple[int, int]] = []
    order = list(range(len(groups)))
    rng.shuffle(order)
    for idx in order:
        group = groups[idx]
        fh = group["inner_h"] + 2
        fw = group["inner_w"] + 2
        pos = _random_place_rect(rng, occ, fh, fw, gap=1)
        if pos is None:
            return None
        frame_positions.append((idx, pos[0], pos[1]))

    frame_lookup = {idx: (top, left) for idx, top, left in frame_positions}
    for idx, group in enumerate(groups):
        color = colors[idx]
        top, left = frame_lookup[idx]
        _draw_frame(inp, top, left, group["inner_h"], group["inner_w"], color)
        _draw_frame(expected, top, left, group["inner_h"], group["inner_w"], color)
        expected[top + 1 : top + 1 + group["inner_h"], left + 1 : left + 1 + group["inner_w"]] = color
        for ar, ac in group["anchors"]:
            inp[top + 1 + ar, left + 1 + ac] = color

    piece_jobs = []
    for idx, group in enumerate(groups):
        for shape in group["shapes"]:
            piece_jobs.append((idx, shape))
    rng.shuffle(piece_jobs)
    for idx, shape in piece_jobs:
        color = colors[idx]
        ph, pw = _shape_dims(shape)
        pos = _random_place_rect(rng, occ, ph, pw, gap=1)
        if pos is None:
            return None
        _draw_shape(inp, pos[0], pos[1], shape, color)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_count = 2 if rng.random() < 0.5 else 3
    colors = rng.sample(COLOR_POOL, group_count)
    groups: list[dict] = []
    for _ in range(group_count):
        group = _build_group(rng)
        if group is None:
            return None
        groups.append(group)
    for _ in range(PLACEMENT_ATTEMPTS):
        layout = _attempt_layout(rng, groups, colors)
        if layout is not None:
            return layout
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
