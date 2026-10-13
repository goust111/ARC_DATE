from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
Frame = tuple[int, int, int, int, int]

BG = 0
FRAME_COLORS = (5, 9)
PATH_COLORS = (1, 2, 3, 4, 6, 7)

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
FRAME_COUNT_RANGE = (2, 3)
INTERIOR_H_RANGE = (6, 9)
INTERIOR_W_RANGE = (6, 9)
FRAME_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_PATH_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_bbox(comp: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_from_component(grid: np.ndarray, color: int, comp: list[Coord]) -> Frame | None:
    r0, c0, r1, c1 = _component_bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    border = np.zeros_like(grid, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    comp_mask = np.zeros_like(grid, dtype=bool)
    for r, c in comp:
        comp_mask[r, c] = True
    if len(comp) != 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4:
        return None
    if not np.array_equal(comp_mask, border):
        return None
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return None
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return None
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return None
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return None
    return (color, r0, c0, r1, c1)


def _find_frames(grid: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    for color in FRAME_COLORS:
        for comp in _components(grid == color):
            frame = _frame_from_component(grid, color, comp)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: (item[1], item[2], item[3], item[4], item[0]))
    return frames


def _ordered_path(cells: list[Coord], start: Coord | None = None) -> tuple[list[Coord], list[Coord]]:
    cell_set = set(cells)
    degrees = {
        cell: sum((nbr in cell_set) for nbr in _neighbors4(*cell))
        for cell in cell_set
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must be non-branching"
    if start is None:
        start = endpoints[0]
    prev = None
    cur = start
    path: list[Coord] = []
    while True:
        path.append(cur)
        next_cells = [nbr for nbr in _neighbors4(*cur) if nbr in cell_set and nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]
    assert len(path) == len(cell_set), "path traversal must visit every path cell"
    return path, endpoints


def _choose_start(frame: Frame, endpoints: list[Coord]) -> Coord:
    _, r0, c0, _r1, _c1 = frame
    target = (r0 + 1, c0 + 1)

    def key(cell: Coord) -> tuple[int, int, int]:
        return (abs(cell[0] - target[0]) + abs(cell[1] - target[1]), cell[0], cell[1])

    return min(endpoints, key=key)


def _solve_frame(out: np.ndarray, frame: Frame) -> dict[str, object]:
    color, r0, c0, r1, c1 = frame
    assert color in FRAME_COLORS
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    path_mask = interior != BG
    comps = _components(path_mask)
    assert len(comps) == 1, "each frame interior must contain exactly one path"
    comp = [(r + r0 + 1, c + c0 + 1) for r, c in comps[0]]
    path, endpoints = _ordered_path(comp)
    start = _choose_start(frame, endpoints)
    if path[0] != start:
        path, _ = _ordered_path(comp, start=start)
    colors = [int(out[r, c]) for r, c in path]
    reversed_colors = list(reversed(colors))
    for (r, c), value in zip(path, reversed_colors):
        out[r, c] = value
    fill_color = reversed_colors[0]
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if out[r, c] == BG:
                out[r, c] = fill_color
    return {
        "path": path,
        "endpoints": endpoints,
        "start": start,
        "fill_color": fill_color,
        "original_colors": colors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse each framed interior path and use its far-end color to fill the remaining cavity.

    Input structure:
    - The background is black.
    - The grid contains 2-3 separate one-pixel-thick rectangular frames in grey or maroon.
    - Inside each frame is exactly one multicolor 4-connected one-pixel-wide path, with black empty space around it and at least one black cell separating the path from the frame.

    Rule:
    - Detect every complete rectangular frame and isolate its strict interior.
    - Inside each frame, find the unique non-branching path and its two endpoints.
    - Choose as traversal start the endpoint with smaller Manhattan distance to the interior's top-left cell; break ties by smaller row, then smaller column.
    - Reverse the path's color sequence along the same path cells, then fill every remaining black interior cell with the new start color, which equals the original far endpoint color.
    - Leave frame pixels and exterior background unchanged.

    Key insight:
    - The path reversal is an information-transfer step: it brings the far endpoint's color to the chosen near endpoint, and that transported color becomes the cavity fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _find_frames(g)
    assert 1 <= len(frames), "expected at least one rectangular frame"
    out = g.copy()
    for frame in frames:
        _solve_frame(out, frame)
    return out


def _path_turns(path: list[Coord]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _good_extension(next_cell: Coord, cur: Coord, used: set[Coord], shape: tuple[int, int]) -> bool:
    r, c = next_cell
    h, w = shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if next_cell in used:
        return False
    for nbr in _neighbors4(r, c):
        if nbr in used and nbr != cur:
            return False
    return True


def _build_random_path(height: int, width: int, target_len: int) -> list[Coord] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(MAX_PATH_ATTEMPTS):
        start = random.choice(cells)
        path = [start]
        used = {start}
        last_dir: Coord | None = None
        while len(path) < target_len:
            cur = path[-1]
            candidates: list[Coord] = []
            for nbr in _neighbors4(*cur):
                if _good_extension(nbr, cur, used, (height, width)):
                    candidates.append(nbr)
            if not candidates:
                break
            random.shuffle(candidates)
            if last_dir is not None:
                same_dir = []
                turn_dir = []
                for nbr in candidates:
                    step = (nbr[0] - cur[0], nbr[1] - cur[1])
                    if step == last_dir:
                        same_dir.append(nbr)
                    else:
                        turn_dir.append(nbr)
                if same_dir and turn_dir:
                    candidates = turn_dir if random.random() < 0.55 else same_dir
                elif turn_dir:
                    candidates = turn_dir
                elif same_dir:
                    candidates = same_dir
            nxt = random.choice(candidates)
            last_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            path.append(nxt)
            used.add(nxt)
        if len(path) != target_len:
            continue
        ordered, endpoints = _ordered_path(path)
        if len(ordered) != target_len:
            continue
        if _path_turns(ordered) < 2:
            continue
        return ordered
    return None


def _endpoint_tie(frame_path: list[Coord]) -> bool:
    endpoints = [frame_path[0], frame_path[-1]]
    scores = sorted((r + c, r, c) for r, c in endpoints)
    return scores[0][0] == scores[1][0]


def _sample_color_sequence(length: int) -> list[int]:
    distinct = random.randint(3, 5)
    palette = random.sample(PATH_COLORS, distinct)
    seq = palette[:]
    while len(seq) < length:
        seq.append(random.choice(palette))
    random.shuffle(seq)
    if seq[0] == seq[-1]:
        for i in range(1, len(seq) - 1):
            if seq[i] != seq[0]:
                seq[i], seq[-1] = seq[-1], seq[i]
                break
    if seq == seq[::-1]:
        for i in range(1, len(seq)):
            if seq[i] != seq[0]:
                seq[0], seq[i] = seq[i], seq[0]
                break
    return seq


def _make_frame_patch(interior_h: int, interior_w: int, border_color: int, require_tie: bool | None) -> tuple[np.ndarray, dict[str, object]] | None:
    usable_h = interior_h - 2
    usable_w = interior_w - 2
    interior_area = interior_h * interior_w
    low = int(np.ceil(0.30 * interior_area))
    high = min(int(np.floor(0.60 * interior_area)), usable_h * usable_w)
    if low > high:
        return None

    for _ in range(MAX_PATH_ATTEMPTS):
        target_len = random.randint(low, high)
        path = _build_random_path(usable_h, usable_w, target_len)
        if path is None:
            continue
        if require_tie is not None and _endpoint_tie(path) != require_tie:
            continue
        colors = _sample_color_sequence(len(path))
        if colors[0] == colors[-1]:
            continue
        patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
        patch[0, :] = border_color
        patch[-1, :] = border_color
        patch[:, 0] = border_color
        patch[:, -1] = border_color
        for (ur, uc), value in zip(path, colors):
            patch[ur + 2, uc + 2] = value

        rel_path = [(ur + 1, uc + 1) for ur, uc in path]
        info = {
            "rel_path": rel_path,
            "colors": colors,
            "tie": _endpoint_tie(path),
            "border_color": border_color,
            "interior_shape": (interior_h, interior_w),
        }
        return patch, info
    return None


def _bboxes_separate(boxes: list[tuple[int, int, int, int]], gap: int) -> bool:
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        for rr0, cc0, rr1, cc1 in boxes[i + 1 :]:
            if not (r1 + gap < rr0 or rr1 + gap < r0 or c1 + gap < cc0 or cc1 + gap < c0):
                return False
    return True


def _place_patches(
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
) -> tuple[np.ndarray, list[Frame], list[dict[str, object]]] | None:
    h, w = grid_shape
    grid = np.zeros((h, w), dtype=int)
    frames: list[Frame] = []
    infos: list[dict[str, object]] = []
    boxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(patches)))
    random.shuffle(order)
    placed = [False] * len(patches)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph > h or pw > w:
            return None
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        done = False
        for r0, c0 in positions:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if not _bboxes_separate(boxes + [box], FRAME_GAP):
                continue
            if np.any(grid[r0 : r0 + ph, c0 : c0 + pw] != BG):
                continue
            region = patch != BG
            grid[r0 : r0 + ph, c0 : c0 + pw][region] = patch[region]
            border_color = int(patch[0, 0])
            frames.append((border_color, box[0], box[1], box[2], box[3]))
            boxes.append(box)
            placed[idx] = True
            done = True
            break
        if not done:
            return None

    frame_lookup = {frame[1:]: frame for frame in frames}
    infos_out: list[dict[str, object]] = []
    for frame in frames:
        bbox = frame[1:]
        if bbox not in frame_lookup:
            return None
        infos_out.append({})
    return grid, frames, infos_out


def _construct_input() -> tuple[np.ndarray, dict[str, object]] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    frame_count = random.randint(*FRAME_COUNT_RANGE)

    desired_tie = random.random() < 0.45
    tie_index = random.randrange(frame_count) if desired_tie else None
    patches: list[np.ndarray] = []
    patch_infos: list[dict[str, object]] = []
    far_colors: list[int] = []

    for i in range(frame_count):
        interior_h = random.randint(*INTERIOR_H_RANGE)
        interior_w = random.randint(*INTERIOR_W_RANGE)
        border_color = random.choice(FRAME_COLORS)
        require_tie = True if tie_index == i else None
        built = _make_frame_patch(interior_h, interior_w, border_color, require_tie)
        if built is None:
            return None
        patch, info = built
        patches.append(patch)
        patch_infos.append(info)
        far_colors.append(int(info["colors"][-1]))

    if frame_count >= 2 and len(set(far_colors)) == 1:
        return None

    placed = _place_patches((h, w), patches)
    if placed is None:
        return None
    grid, frames, _ = placed

    frame_meta: list[dict[str, object]] = []
    for frame, info in zip(frames, patch_infos):
        rel_path = info["rel_path"]
        abs_path = [(frame[1] + r, frame[2] + c) for r, c in rel_path]
        frame_meta.append(
            {
                "frame": frame,
                "path": abs_path,
                "colors": list(info["colors"]),
                "tie": bool(info["tie"]),
                "interior_shape": info["interior_shape"],
            }
        )

    meta = {
        "desired_tie": desired_tie,
        "frames": frame_meta,
    }
    return grid, meta


def _wrong_fill_output(grid: np.ndarray, frame: Frame, fill_color: int) -> np.ndarray:
    out = grid.copy()
    solve = _solve_frame(out, frame)
    _ = solve
    _, r0, c0, r1, c1 = frame
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) not in set(solve["path"]):  # type: ignore[index]
                out[r, c] = fill_color
    return out


def _validate_instance(grid: np.ndarray, meta: dict[str, object]) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    frames = _find_frames(grid)
    if len(frames) not in (2, 3):
        return False

    frame_data = meta["frames"]
    if len(frame_data) != len(frames):
        return False

    saw_tie = False
    saw_non_tie = False
    for frame in frames:
        _, r0, c0, r1, c1 = frame
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        path_mask = interior != BG
        comps = _components(path_mask)
        if len(comps) != 1:
            return False
        comp = [(r + r0 + 1, c + c0 + 1) for r, c in comps[0]]
        try:
            path, endpoints = _ordered_path(comp)
        except AssertionError:
            return False
        if len({int(grid[r, c]) for r, c in comp}) < 3 or len({int(grid[r, c]) for r, c in comp}) > 5:
            return False
        if any(r in (r0 + 1, r1 - 1) or c in (c0 + 1, c1 - 1) for r, c in comp):
            return False
        if int(grid[endpoints[0]]) == int(grid[endpoints[1]]):
            return False
        interior_area = (r1 - r0 - 1) * (c1 - c0 - 1)
        ratio = len(comp) / interior_area
        if not (0.30 <= ratio <= 0.60):
            return False
        start = _choose_start(frame, endpoints)
        tie = (
            abs(endpoints[0][0] - (r0 + 1)) + abs(endpoints[0][1] - (c0 + 1))
            == abs(endpoints[1][0] - (r0 + 1)) + abs(endpoints[1][1] - (c0 + 1))
        )
        saw_tie |= tie
        saw_non_tie |= not tie
        if start not in endpoints:
            return False
        colors = [int(grid[r, c]) for r, c in path]
        if path[0] != start:
            path, _ = _ordered_path(comp, start=start)
            colors = [int(grid[r, c]) for r, c in path]
        if colors == colors[::-1]:
            return False
        if colors[0] == colors[-1]:
            return False

    if meta["desired_tie"] and not saw_tie:
        return False
    if not saw_non_tie and len(frames) >= 2:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if np.array_equal(out, grid):
        return False

    for frame in frames:
        _, r0, c0, r1, c1 = frame
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if not np.any(interior == BG):
            return False
        path_mask = interior != BG
        comp = [(r + r0 + 1, c + c0 + 1) for r, c in _components(path_mask)[0]]
        path, endpoints = _ordered_path(comp)
        start = _choose_start(frame, endpoints)
        if path[0] != start:
            path, _ = _ordered_path(comp, start=start)
        colors = [int(grid[r, c]) for r, c in path]
        correct_fill = colors[-1]
        wrong_fill = colors[0]
        frame_out = out[r0 + 1 : r1, c0 + 1 : c1]
        if not np.all(frame_out[interior == BG] == correct_fill):
            return False

        no_reverse = grid.copy()
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if no_reverse[r, c] == BG:
                    no_reverse[r, c] = correct_fill
        if np.array_equal(no_reverse, out):
            return False

        wrong = grid.copy()
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if wrong[r, c] == BG:
                    wrong[r, c] = wrong_fill
        if np.array_equal(wrong, out):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _construct_input()
        if built is None:
            continue
        grid, meta = built
        if _validate_instance(grid, meta):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
