from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
AZURE = 8
PIPE_VALUES = (GREY, ORANGE, AZURE)
MARKER_VALUES = (1, 2, 3, 4, 6, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _right(direction: tuple[int, int]) -> tuple[int, int]:
    return (int(direction[1]), int(-direction[0]))


def _left(direction: tuple[int, int]) -> tuple[int, int]:
    return (int(-direction[1]), int(direction[0]))


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _pipe_mask(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, PIPE_VALUES)


def _pipe_neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nxt = (int(r + dr), int(c + dc))
        if _in_bounds(mask.shape, nxt) and bool(mask[nxt]):
            out.append(nxt)
    return out


def _find_entries(grid: np.ndarray) -> list[dict]:
    mask = _pipe_mask(grid)
    entries: list[dict] = []
    marker_mask = (~np.isin(grid, PIPE_VALUES)) & (grid != BG)
    for r, c in np.argwhere(marker_mask):
        marker = (int(r), int(c))
        nbs = _pipe_neighbors(mask, marker)
        assert len(nbs) == 1, "each entry marker must touch exactly one pipe cell"
        start = nbs[0]
        entries.append(
            {
                "marker": marker,
                "color": int(grid[marker]),
                "start": start,
                "direction": (int(start[0] - marker[0]), int(start[1] - marker[1])),
            }
        )
    assert entries, "expected at least one colored entry marker"
    colors = [entry["color"] for entry in entries]
    assert len(colors) == len(set(colors)), "entry colors must be unique"
    entries.sort(key=lambda entry: (entry["marker"][0], entry["marker"][1]))
    return entries


def _next_step(
    grid: np.ndarray,
    mask: np.ndarray,
    cell: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r, c = cell
    value = int(grid[cell])
    if value == ORANGE:
        next_dir = direction
        nxt = (int(r + next_dir[0]), int(c + next_dir[1]))
        if _in_bounds(grid.shape, nxt) and bool(mask[nxt]):
            return nxt, next_dir
        return None
    if value == AZURE:
        next_dir = _right(direction)
        nxt = (int(r + next_dir[0]), int(c + next_dir[1]))
        if _in_bounds(grid.shape, nxt) and bool(mask[nxt]):
            return nxt, next_dir
        return None

    prev = (int(r - direction[0]), int(c - direction[1]))
    options = [nxt for nxt in _pipe_neighbors(mask, cell) if nxt != prev]
    if len(options) != 1:
        return None
    nxt = options[0]
    next_dir = (int(nxt[0] - r), int(nxt[1] - c))
    return nxt, next_dir


def _trace_path(grid: np.ndarray, entry: dict) -> list[tuple[int, int]]:
    mask = _pipe_mask(grid)
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    cell = entry["start"]
    direction = entry["direction"]
    while True:
        if cell in seen:
            break
        path.append(cell)
        seen.add(cell)
        step = _next_step(grid, mask, cell, direction)
        if step is None:
            break
        cell, direction = step
    return path


def _claimed_pipe_cells(grid: np.ndarray) -> dict[tuple[int, int], int]:
    entries = _find_entries(grid)
    paths = [_trace_path(grid, entry) for entry in entries]

    best_step: dict[tuple[int, int], int] = {}
    best_src: dict[tuple[int, int], int] = {}
    tied: set[tuple[int, int]] = set()

    for src, path in enumerate(paths):
        for step, cell in enumerate(path):
            prev = best_step.get(cell)
            if prev is None or int(step) < int(prev):
                best_step[cell] = int(step)
                best_src[cell] = int(src)
                tied.discard(cell)
            elif int(step) == int(prev) and int(best_src[cell]) != int(src):
                tied.add(cell)

    claims: dict[tuple[int, int], int] = {}
    for src, path in enumerate(paths):
        color = int(entries[src]["color"])
        for cell in path:
            if cell in tied or int(best_src.get(cell, -1)) != int(src):
                break
            claims[cell] = color
    return claims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each pipe route from its entry marker while obeying straight-through and right-turn cross controls.

    Input structure:
    - The background is black.
    - A one-cell-wide orthogonal pipe network is drawn mostly in grey.
    - Colored singleton entry markers sit next to pipe dead ends and determine the fill colors.
    - Some four-way intersections use orange centers for straight-through flow or azure centers for right-turn flow.

    Rule:
    - For each entry marker, start from its unique adjacent pipe cell and move forward through the network.
    - Recolor every visited grey pipe cell, and every visited orange or azure intersection center, to that entry color.
    - Ordinary pipe cells continue to the unique next pipe cell: straight segments stay straight and L-corners turn naturally.
    - At an orange four-way center the flow keeps the same direction; at an azure four-way center it turns 90 degrees to the right.
    - A route stops before entering a pipe cell that an earlier-arriving route already claims, or when it reaches a dead end.
    - Entry marker pixels themselves keep their original colors.

    Key insight:
    - The pipe network is navigated by direction, and the special orange/azure cross centers are local control tiles that resolve otherwise ambiguous four-way intersections.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    for cell, color in _claimed_pipe_cells(g).items():
        out[cell] = int(color)
    return out


def _shift_patch(
    pipe_cells: set[tuple[int, int]],
    markers: dict[tuple[int, int], int],
    specials: dict[tuple[int, int], int],
) -> np.ndarray:
    all_cells = list(pipe_cells) + list(markers)
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)
    height = int(max_r - min_r + 3)
    width = int(max_c - min_c + 3)
    patch = np.full((height, width), BG, dtype=int)
    shift_r = 1 - int(min_r)
    shift_c = 1 - int(min_c)

    for r, c in pipe_cells:
        patch[int(r + shift_r), int(c + shift_c)] = GREY
    for (r, c), value in specials.items():
        patch[int(r + shift_r), int(c + shift_c)] = int(value)
    for (r, c), value in markers.items():
        patch[int(r + shift_r), int(c + shift_c)] = int(value)
    return patch


def _valid_simple_path(path: list[tuple[int, int]]) -> bool:
    if len(path) < 4:
        return False
    cells = set(path)
    if len(cells) != len(path):
        return False
    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        if abs(int(r0 - r1)) + abs(int(c0 - c1)) != 1:
            return False
    for i, cell in enumerate(path):
        degree = sum((int(cell[0] + dr), int(cell[1] + dc)) in cells for dr, dc in DIRS)
        wanted = 1 if i in (0, len(path) - 1) else 2
        if int(degree) != int(wanted):
            return False
    return True


def _build_walk_patch(color: int) -> np.ndarray:
    for _ in range(200):
        target_len = random.randint(4, 10)
        direction = random.choice(DIRS)
        path = [(0, 0)]
        used = {(0, 0)}

        while len(path) < target_len:
            r, c = path[-1]
            options: list[tuple[int, int]] = []
            weights: list[int] = []
            for cand in (direction, _left(direction), _right(direction)):
                nxt = (int(r + cand[0]), int(c + cand[1]))
                if nxt in used:
                    continue
                if max(abs(nxt[0]), abs(nxt[1])) > 5:
                    continue
                options.append(cand)
                weights.append(3 if cand == direction else 2)
            if not options:
                break
            direction = random.choices(options, weights=weights, k=1)[0]
            nxt = (int(r + direction[0]), int(c + direction[1]))
            used.add(nxt)
            path.append(nxt)

        if not _valid_simple_path(path):
            continue
        start_dir = (int(path[1][0] - path[0][0]), int(path[1][1] - path[0][1]))
        marker = (int(path[0][0] - start_dir[0]), int(path[0][1] - start_dir[1]))
        if marker in used:
            continue
        return _shift_patch(set(path), {marker: int(color)}, {})
    raise AssertionError("failed to build a simple pipe patch")


def _build_cross_patch(center_value: int, colors: list[int]) -> np.ndarray:
    move_dir = random.choice(DIRS)
    in_arm = (int(-move_dir[0]), int(-move_dir[1]))
    out_arm = move_dir if center_value == ORANGE else _right(move_dir)
    side_arms = [direction for direction in DIRS if direction not in (in_arm, out_arm)]

    len_in = random.randint(1, 3)
    len_out = random.randint(2, 5)
    len_side_a = random.randint(len_in + 1, len_in + 4)
    len_side_b = random.randint(len_in + 1, len_in + 4)
    lengths = {
        in_arm: len_in,
        out_arm: len_out,
        side_arms[0]: len_side_a,
        side_arms[1]: len_side_b,
    }

    pipe_cells: set[tuple[int, int]] = {(0, 0)}
    for direction, length in lengths.items():
        for step in range(1, int(length) + 1):
            pipe_cells.add((int(direction[0] * step), int(direction[1] * step)))

    markers: dict[tuple[int, int], int] = {}
    for direction, color in zip([in_arm, side_arms[0], side_arms[1]], colors):
        length = int(lengths[direction])
        endpoint = (int(direction[0] * length), int(direction[1] * length))
        marker = (int(endpoint[0] + direction[0]), int(endpoint[1] + direction[1]))
        markers[marker] = int(color)

    return _shift_patch(pipe_cells, markers, {(0, 0): int(center_value)})


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_patches(patches: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: int(patches[idx].shape[0] * patches[idx].shape[1]), reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(150):
            r0 = random.randint(0, int(height - ph))
            c0 = random.randint(0, int(width - pw))
            rect = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
            expanded = (int(rect[0] - 1), int(rect[1] - 1), int(rect[2] + 1), int(rect[3] + 1))
            if any(_rects_overlap(expanded, other) for other in reserved):
                continue
            mask = patch != BG
            if bool(np.any(grid[r0 : r0 + ph, c0 : c0 + pw][mask] != BG)):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
            reserved.append(rect)
            placed = True
            break
        if not placed:
            return None
    return grid


def _validate_scene(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    entries = _find_entries(grid)
    marker_mask = (~np.isin(grid, PIPE_VALUES)) & (grid != BG)
    assert int(marker_mask.sum()) == len(entries)

    mask = _pipe_mask(grid)
    assert bool(mask.any())
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        degree = len(_pipe_neighbors(mask, cell))
        value = int(grid[cell])
        if value in (ORANGE, AZURE):
            assert degree == 4, "special intersections must be 4-way crosses"
        else:
            assert degree in (1, 2), "ordinary pipe cells may only be straight, corner, or dead-end"

    out = main(grid)
    assert bool(np.any(out != grid)), "scene must actually transform"
    assert not bool(np.isin(out, PIPE_VALUES).any()), "all generated pipe cells should get recolored"


def generate_input() -> np.ndarray:
    for _ in range(400):
        cross_count = 2 if random.random() < 0.35 else 1
        if cross_count == 2 and random.random() < 0.5:
            cross_values = [ORANGE, AZURE]
            random.shuffle(cross_values)
        else:
            cross_values = [random.choice((ORANGE, AZURE)) for _ in range(cross_count)]

        simple_count = 0 if cross_count == 2 else random.randint(0, 3)
        total_markers = 3 * cross_count + simple_count
        if total_markers > len(MARKER_VALUES):
            continue

        colors = list(MARKER_VALUES)
        random.shuffle(colors)
        color_idx = 0
        patches: list[np.ndarray] = []

        for center_value in cross_values:
            patches.append(_build_cross_patch(int(center_value), colors[color_idx : color_idx + 3]))
            color_idx += 3
        for _ in range(simple_count):
            patches.append(_build_walk_patch(int(colors[color_idx])))
            color_idx += 1

        max_h = max(int(patch.shape[0]) for patch in patches)
        max_w = max(int(patch.shape[1]) for patch in patches)
        area_hint = sum(int((patch.shape[0] + 2) * (patch.shape[1] + 2)) for patch in patches)
        low = max(12, max_h + 2, max_w + 2, int(np.ceil(np.sqrt(area_hint))))
        high = 30
        if low > high:
            continue
        height = random.randint(int(low), int(high))
        width = random.randint(int(low), int(high))
        grid = _place_patches(patches, height, width)
        if grid is None:
            continue
        try:
            _validate_scene(grid)
        except AssertionError:
            continue
        return grid
    raise AssertionError("failed to generate a valid pipe scene")
