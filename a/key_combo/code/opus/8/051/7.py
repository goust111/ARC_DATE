from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
FENCE = 5
MARKER = 8
PALETTE = (2, 1, 3, 4)

MIN_SIDE = 4
MAX_SIDE = 8
GEN_ATTEMPTS = 500

Box = tuple[int, int, int, int]


def _draw_border(grid: np.ndarray, box: Box, color: int = FENCE) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _border_cells(box: Box) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _interior_cells(box: Box) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _footprint_cells(box: Box) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _center(box: Box) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _render_occluded_frames(shape: tuple[int, int], front_to_back: list[Box]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for box in reversed(front_to_back):
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG
        _draw_border(grid, box, FENCE)
    return grid


def _render_completed_output(
    shape: tuple[int, int], front_to_back: list[Box], marker: tuple[int, int]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for box in front_to_back:
        _draw_border(out, box, FENCE)

    owner = np.full(shape, -1, dtype=int)
    for idx, box in enumerate(front_to_back):
        r0, c0, r1, c1 = box
        view = owner[r0 + 1 : r1, c0 + 1 : c1]
        view[view < 0] = idx

    distances = []
    mr, mc = marker
    for box in front_to_back:
        cr, cc = _center(box)
        distances.append((abs(cr - mr) + abs(cc - mc), cr, cc, box))
    rank_colors = {item[3]: PALETTE[i] for i, item in enumerate(sorted(distances))}

    for idx, box in enumerate(front_to_back):
        out[owner == idx] = rank_colors[box]

    out[marker] = MARKER
    return out


def _marker_and_frame(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one cyan marker"
    marker = tuple(map(int, marker_coords[0]))
    frame_grid = g.copy()
    frame_grid[marker] = BG
    allowed = {BG, FENCE, MARKER}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected colors in input"
    return marker, frame_grid


def _side_values(frame_grid: np.ndarray, box: Box) -> list[np.ndarray]:
    r0, c0, r1, c1 = box
    return [
        frame_grid[r0, c0 : c1 + 1],
        frame_grid[r1, c0 : c1 + 1],
        frame_grid[r0 : r1 + 1, c0],
        frame_grid[r0 : r1 + 1, c1],
    ]


def _visible_side_profile(frame_grid: np.ndarray, box: Box) -> tuple[int, int]:
    values = _side_values(frame_grid, box)
    full_count = sum(bool(np.all(side == FENCE)) for side in values)
    partial_count = sum(bool(np.any(side == FENCE) and np.any(side == BG)) for side in values)
    return full_count, partial_count


def _candidate_boxes(frame_grid: np.ndarray) -> list[dict]:
    h, w = frame_grid.shape
    candidates: list[dict] = []
    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(frame_grid == FENCE)}
    for r0 in range(h):
        for c0 in range(w):
            if frame_grid[r0, c0] != FENCE:
                continue
            for height in range(MIN_SIDE, min(MAX_SIDE, h - r0) + 1):
                r1 = r0 + height - 1
                for width in range(MIN_SIDE, min(MAX_SIDE, w - c0) + 1):
                    c1 = c0 + width - 1
                    if frame_grid[r0, c1] != FENCE or frame_grid[r1, c0] != FENCE or frame_grid[r1, c1] != FENCE:
                        continue
                    box = (r0, c0, r1, c1)
                    values = _side_values(frame_grid, box)
                    if any(not set(map(int, np.unique(side))).issubset({BG, FENCE}) for side in values):
                        continue
                    full_count, partial_count = _visible_side_profile(frame_grid, box)
                    if full_count < 3 or partial_count > 1:
                        continue
                    border = _border_cells(box)
                    candidates.append(
                        {
                            "box": box,
                            "visible": border & grey_cells,
                        }
                    )
    dedup: dict[Box, dict] = {}
    for cand in candidates:
        dedup[cand["box"]] = cand
    return list(dedup.values())


def _find_valid_solutions(frame_grid: np.ndarray) -> list[list[Box]]:
    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(frame_grid == FENCE)}
    candidates = _candidate_boxes(frame_grid)
    solutions: list[list[Box]] = []
    seen: set[tuple[Box, ...]] = set()

    for count in (3, 4):
        if len(candidates) < count:
            continue
        for combo in itertools.combinations(range(len(candidates)), count):
            visible_union: set[tuple[int, int]] = set()
            for idx in combo:
                visible_union |= candidates[idx]["visible"]
            if visible_union != grey_cells:
                continue

            combo_boxes = [candidates[idx]["box"] for idx in combo]
            for perm in itertools.permutations(range(count)):
                front_to_back = [combo_boxes[i] for i in perm]
                key = tuple(front_to_back)
                if key in seen:
                    continue
                seen.add(key)
                if np.array_equal(_render_occluded_frames(frame_grid.shape, front_to_back), frame_grid):
                    solutions.append(front_to_back)
    return solutions


def _unique_solution_outputs(grid: np.ndarray) -> tuple[list[list[Box]], list[np.ndarray], tuple[int, int]]:
    marker, frame_grid = _marker_and_frame(grid)
    solutions = _find_valid_solutions(frame_grid)
    assert solutions, "failed to infer any valid rectangle explanation"
    min_count = min(len(solution) for solution in solutions)
    solutions = [solution for solution in solutions if len(solution) == min_count]

    unique_outputs: list[np.ndarray] = []
    output_keys: set[bytes] = set()
    for solution in solutions:
        out = _render_completed_output(grid.shape, solution, marker)
        key = out.tobytes()
        if key not in output_keys:
            output_keys.add(key)
            unique_outputs.append(out)
    return solutions, unique_outputs, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the mutually occluded hollow rectangles, then rank their interiors by distance to the lone marker and fill them with palette colors.

    Input structure:
    - The grid is black except for 3-4 gray hollow rectangular frames and exactly one cyan marker pixel.
    - All frames use the same gray color, have 1-pixel borders, and partially overlap under a single global depth order.
    - Each frame still exposes at least three complete sides, so its full rectangle can be inferred even if one side is partly hidden.

    Rule:
    - Infer every full rectangle from the visible gray border evidence, using the same row/column coordinate system as the grid.
    - Restore all hidden border pixels to gray, then treat each rectangle's interior as the area strictly inside its completed border.
    - If completed interiors overlap, assign each shared interior pixel to the front-most rectangle from the inferred depth order.
    - Compute each interior center, measure its Manhattan distance to the cyan marker, rank frames by increasing distance with row/column tie-breaks, and fill each owned interior with the palette [red, blue, green, yellow].
    - Keep every gray border pixel and the cyan marker unchanged.

    Key insight:
    - The visible border fragments are not enough by themselves: you must first recover the hidden sides and the occlusion order, because those reconstructed interiors determine both the ownership of overlapping cells and the distance ranking used for recoloring.
    """
    solutions, unique_outputs, _marker = _unique_solution_outputs(np.asarray(grid, dtype=int))
    assert solutions, "expected at least one valid reconstruction"
    assert len(unique_outputs) == 1, "input does not determine a unique output"
    return unique_outputs[0]


def _boxes_overlap(a: Box, b: Box) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _interiors_overlap(a: Box, b: Box) -> bool:
    ar0, ac0, ar1, ac1 = a[0] + 1, a[1] + 1, a[2] - 1, a[3] - 1
    br0, bc0, br1, bc1 = b[0] + 1, b[1] + 1, b[2] - 1, b[3] - 1
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _connected_boxes(boxes: list[Box]) -> bool:
    graph = {i: set() for i in range(len(boxes))}
    for i, box_i in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(box_i, boxes[j]):
                graph[i].add(j)
                graph[j].add(i)
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(boxes)


def _has_black_occlusion_evidence(frame_grid: np.ndarray, front: Box, back: Box) -> bool:
    front_interior = _interior_cells(front)
    for cell in _border_cells(back):
        if cell in front_interior and frame_grid[cell] == BG:
            return True
    return False


def _choose_marker(
    shape: tuple[int, int], boxes: list[Box], centers: list[tuple[int, int]], prefer_tie: bool
) -> tuple[int, int] | None:
    forbidden = set()
    for box in boxes:
        forbidden |= _footprint_cells(box)
    cells = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if (r, c) not in forbidden]
    random.shuffle(cells)
    if not cells:
        return None

    if prefer_tie:
        pairs = list(itertools.combinations(range(len(centers)), 2))
        random.shuffle(pairs)
        for i, j in pairs:
            for cell in cells:
                dists = [abs(cell[0] - cr) + abs(cell[1] - cc) for cr, cc in centers]
                if dists[i] != dists[j]:
                    continue
                if sum(d == dists[i] for d in dists) > 2:
                    continue
                return cell
    return random.choice(cells)


def _side_box(front: Box, side: str, shape: tuple[int, int]) -> Box | None:
    h, w = shape
    fr0, fc0, fr1, fc1 = front
    fh = fr1 - fr0 + 1
    fw = fc1 - fc0 + 1

    for _ in range(80):
        if side in {"top", "bottom"}:
            max_extra = MAX_SIDE - fw
            if max_extra < 1:
                return None
            mode = random.choice(["left", "right"])
            ext = random.randint(1, min(2, max_extra))
            if mode == "left":
                c0 = fc0 - ext
                c1 = fc1
            else:
                c0 = fc0
                c1 = fc1 + ext
            if not (0 <= c0 < c1 < w):
                continue

            if side == "top":
                r0 = random.randint(fr0 + 1, fr1 - 1)
                r1 = fr1 + random.randint(1, 2)
            else:
                r1 = random.randint(fr0 + 1, fr1 - 1)
                r0 = fr0 - random.randint(1, 2)
            if not (0 <= r0 < r1 < h):
                continue
            if not (MIN_SIDE <= r1 - r0 + 1 <= MAX_SIDE):
                continue
            return (r0, c0, r1, c1)

        max_extra = MAX_SIDE - fh
        if max_extra < 1:
            return None
        mode = random.choice(["up", "down"])
        ext = random.randint(1, min(2, max_extra))
        if mode == "up":
            r0 = fr0 - ext
            r1 = fr1
        else:
            r0 = fr0
            r1 = fr1 + ext
        if not (0 <= r0 < r1 < h):
            continue

        if side == "left":
            c0 = random.randint(fc0 + 1, fc1 - 1)
            c1 = fc1 + random.randint(1, 2)
        else:
            c1 = random.randint(fc0 + 1, fc1 - 1)
            c0 = fc0 - random.randint(1, 2)
        if not (0 <= c0 < c1 < w):
            continue
        if not (MIN_SIDE <= c1 - c0 + 1 <= MAX_SIDE):
            continue
        return (r0, c0, r1, c1)
    return None


def _sample_layout() -> tuple[tuple[int, int], list[Box], list[Box], tuple[int, int]] | None:
    for _ in range(120):
        h = random.randint(14, 18)
        w = random.randint(14, 18)
        n_boxes = random.choice([3, 4])

        fh = random.randint(5, 6)
        fw = random.randint(5, 6)
        fr0 = random.randint(3, h - fh - 4)
        fc0 = random.randint(3, w - fw - 4)
        front = (fr0, fc0, fr0 + fh - 1, fc0 + fw - 1)

        side_count = n_boxes - 1
        sides = random.sample(["top", "bottom", "left", "right"], side_count)
        boxes = [front]
        failed = False
        for side in sides:
            placed = None
            for _ in range(50):
                box = _side_box(front, side, (h, w))
                if box is None:
                    continue
                if any(_boxes_overlap(box, other) for other in boxes[1:]):
                    continue
                placed = box
                break
            if placed is None:
                failed = True
                break
            boxes.append(placed)
        if failed:
            continue

        centers = [_center(box) for box in boxes]
        marker = _choose_marker((h, w), boxes, centers, prefer_tie=random.random() < 0.35)
        if marker is None:
            continue

        back = boxes[1:]
        random.shuffle(back)
        front_to_back = [front] + back
        return (h, w), boxes, front_to_back, marker
    return None


def _semantic_check(
    shape: tuple[int, int], boxes: list[Box], front_to_back: list[Box], marker: tuple[int, int]
) -> bool:
    frame_grid = _render_occluded_frames(shape, front_to_back)

    if not _connected_boxes(boxes):
        return False

    for box in boxes:
        full_count, partial_count = _visible_side_profile(frame_grid, box)
        if full_count < 3 or partial_count > 1:
            return False

    for later_idx, back in enumerate(front_to_back[1:], start=1):
        for front in front_to_back[:later_idx]:
            if _interiors_overlap(front, back) and not _has_black_occlusion_evidence(frame_grid, front, back):
                return False

    inp = frame_grid.copy()
    inp[marker] = MARKER

    try:
        solutions, unique_outputs, _ = _unique_solution_outputs(inp)
    except AssertionError:
        return False

    expected = _render_completed_output(shape, front_to_back, marker)
    if len(unique_outputs) != 1 or not np.array_equal(unique_outputs[0], expected):
        return False

    normalized = {tuple(sorted(solution)) for solution in solutions}
    if len(normalized) != 1:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    sampled = _sample_layout()
    if sampled is None:
        return None
    shape, boxes, front_to_back, marker = sampled
    frame_grid = _render_occluded_frames(shape, front_to_back)
    inp = frame_grid.copy()
    inp[marker] = MARKER
    if not _semantic_check(shape, boxes, front_to_back, marker):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-frames input")
