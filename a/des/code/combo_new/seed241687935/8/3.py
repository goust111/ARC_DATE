from __future__ import annotations

import random

import numpy as np


BG = 0
TOWER_COLOR = 7
DIRS_CW = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
MAX_RETRIES = 400

# Up-facing source bodies: center is supplied separately as the signal color.
UP_BODY_MASKS = [
    np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool),
]


def _dir_index(direction: tuple[int, int]) -> int:
    return DIRS_CW.index(direction)


def _rotate_dir(direction: tuple[int, int], steps_cw: int) -> tuple[int, int]:
    return DIRS_CW[(_dir_index(direction) + steps_cw) % 4]


def _left_of(direction: tuple[int, int]) -> tuple[int, int]:
    return _rotate_dir(direction, -1)


def _right_of(direction: tuple[int, int]) -> tuple[int, int]:
    return _rotate_dir(direction, 1)


def _rotate_cw(arr: np.ndarray, steps_cw: int) -> np.ndarray:
    return np.rot90(arr, -steps_cw)


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _body_tail_direction(body_mask: np.ndarray) -> tuple[int, int]:
    assert body_mask.shape == (3, 3)
    scores = [
        int(body_mask[0, :].sum()),
        int(body_mask[:, 2].sum()),
        int(body_mask[2, :].sum()),
        int(body_mask[:, 0].sum()),
    ]
    best = max(scores)
    assert scores.count(best) == 1, "source tail side must be unique"
    return DIRS_CW[scores.index(best)]


def _full_source_mask(body_mask: np.ndarray) -> np.ndarray:
    full = body_mask.copy()
    full[1, 1] = True
    return full


def _source_nose_local(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return 1 + dr, 1 + dc


def _trace_signal(
    shape: tuple[int, int],
    center: tuple[int, int],
    emit_dir: tuple[int, int],
    towers: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int], int]:
    H, W = shape
    dr, dc = emit_dir
    left = _left_of(emit_dir)
    right = _right_of(emit_dir)
    start = (center[0] + 2 * dr, center[1] + 2 * dc)
    signal_cells: list[tuple[int, int]] = []
    r, c = start
    while 0 <= r < H and 0 <= c < W:
        candidates = [
            (-1, (r + left[0], c + left[1])),
            (0, (r, c)),
            (1, (r + right[0], c + right[1])),
        ]
        hits = [(turn, pos) for turn, pos in candidates if pos in towers]
        assert len(hits) <= 1, "tower corridor must be unambiguous"
        if hits:
            turn, tower = hits[0]
            if turn != 0:
                signal_cells.append((r, c))
            return signal_cells, tower, turn
        signal_cells.append((r, c))
        r += dr
        c += dc
    raise AssertionError("every source must see exactly one tower")


def _copy_cells(
    mask: np.ndarray,
    final_dir: tuple[int, int],
    endpoint: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    anchor_r, anchor_c = _source_nose_local(final_dir)
    top = endpoint[0] - anchor_r
    left = endpoint[1] - anchor_c
    coords = [(top + r, left + c) for r, c in np.argwhere(mask)]
    return coords, (top, left)


def _sorted_sources(g: np.ndarray) -> list[dict]:
    source_mask = (g != BG) & (g != TOWER_COLOR)
    comps = _components8(source_mask)
    sources: list[dict] = []
    for coords in comps:
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "sources must occupy 3x3 windows"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        center = (r0 + 1, c0 + 1)
        signal_color = int(g[center])
        colors = {int(v) for v in np.unique(patch) if int(v) != BG}
        assert len(colors) == 2 and signal_color in colors
        body_colors = [color for color in colors if color != signal_color]
        assert len(body_colors) == 1
        body_color = int(body_colors[0])
        assert int((patch == signal_color).sum()) == 1 and patch[1, 1] == signal_color
        body_mask = patch == body_color
        tail = _body_tail_direction(body_mask)
        emit_dir = _rotate_dir(tail, 2)
        full_mask = patch != BG
        nose_r, nose_c = _source_nose_local(emit_dir)
        assert full_mask[nose_r, nose_c], "every source must have a visible nose cell"
        sources.append(
            {
                "center": center,
                "signal_color": signal_color,
                "body_mask": body_mask,
                "full_mask": full_mask,
                "emit_dir": emit_dir,
            }
        )
    sources.sort(key=lambda src: src["center"])
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast each directional signal source through its tower and stamp a rotated monochrome copy at the terminal edge.

    Input structure:
    - The background is black.
    - Several 3x3 multicolor sprites act as signal sources: each has one unique center pixel in a signal color and the remaining occupied pixels use one body color.
    - Isolated orange singletons are signal towers.
    - A source's body is asymmetric: the densest outer side of the 3x3 body is its tail, so the opposite side is the firing direction.

    Rule:
    - For every source, read the firing direction from the body asymmetry and project a straight centerline ray outward from the first cell beyond the sprite, using the source signal color.
    - The first orange tower found on that centerline or one cell to its left/right decides the turn: same cell = straight, left-adjacent = turn left, right-adjacent = turn right. Continue from the tower to the grid edge in orange.
    - At the terminal edge cell, stamp a copy of the source mask recolored entirely to the signal color and rotated so its pointing side matches the final ray direction.

    Key insight:
    - The sprite silhouette encodes the launch direction, the tower's lateral offset encodes the turn, and the terminal copy preserves only shape while replacing the original body color with the signal color.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    towers = {tuple(map(int, cell)) for cell in np.argwhere(g == TOWER_COLOR)}
    assert towers, "expected at least one orange tower"

    sources = _sorted_sources(g)
    assert sources, "expected at least one source sprite"
    out = g.copy()

    for src in sources:
        signal_cells, tower, turn, = _trace_signal(g.shape, src["center"], src["emit_dir"], towers)
        for r, c in signal_cells:
            assert int(g[r, c]) == BG
            out[r, c] = src["signal_color"]

        final_dir = _rotate_dir(src["emit_dir"], turn)
        orange_cells: list[tuple[int, int]] = []
        r, c = tower
        while 0 <= r < H and 0 <= c < W:
            assert int(g[r, c]) in (BG, TOWER_COLOR)
            out[r, c] = TOWER_COLOR
            orange_cells.append((r, c))
            r += final_dir[0]
            c += final_dir[1]
        assert orange_cells, "tower path must reach the boundary"
        endpoint = orange_cells[-1]

        rot_steps = (_dir_index(final_dir) - _dir_index(src["emit_dir"])) % 4
        rotated_mask = _rotate_cw(src["full_mask"], rot_steps)
        copy_cells, _top_left = _copy_cells(rotated_mask, final_dir, endpoint)
        for rr, cc in copy_cells:
            assert 0 <= rr < H and 0 <= cc < W
            out[rr, cc] = src["signal_color"]

    return out


def _source_patch(body_mask: np.ndarray, body_color: int, signal_color: int) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[body_mask] = body_color
    patch[1, 1] = signal_color
    return patch


def _render_spec_cells(
    H: int,
    W: int,
    center: tuple[int, int],
    emit_dir: tuple[int, int],
    turn: int,
    full_mask: np.ndarray,
) -> dict | None:
    dr, dc = emit_dir
    left = _left_of(emit_dir)
    right = _right_of(emit_dir)
    start = (center[0] + 2 * dr, center[1] + 2 * dc)

    straight_limit = 0
    r, c = start
    while 0 <= r < H and 0 <= c < W:
        straight_limit += 1
        r += dr
        c += dc
    min_k = 2 if turn == 0 else 1
    if straight_limit < min_k:
        return None

    k = random.randint(min_k, straight_limit)
    p = (start[0] + (k - 1) * dr, start[1] + (k - 1) * dc)
    if turn == -1:
        tower = (p[0] + left[0], p[1] + left[1])
    elif turn == 1:
        tower = (p[0] + right[0], p[1] + right[1])
    else:
        tower = p
    if not (0 <= tower[0] < H and 0 <= tower[1] < W):
        return None

    signal_cells = []
    for step in range(k):
        cell = (start[0] + step * dr, start[1] + step * dc)
        if turn == 0 and cell == p:
            break
        signal_cells.append(cell)
    if not signal_cells:
        return None

    scan_cells: set[tuple[int, int]] = set()
    for step in range(k):
        cell = (start[0] + step * dr, start[1] + step * dc)
        scan_cells.add(cell)
        scan_cells.add((cell[0] + left[0], cell[1] + left[1]))
        scan_cells.add((cell[0] + right[0], cell[1] + right[1]))

    final_dir = _rotate_dir(emit_dir, turn)
    orange_cells = []
    r, c = tower
    while 0 <= r < H and 0 <= c < W:
        orange_cells.append((r, c))
        r += final_dir[0]
        c += final_dir[1]
    if len(orange_cells) < 3:
        return None
    endpoint = orange_cells[-1]

    rot_steps = (_dir_index(final_dir) - _dir_index(emit_dir)) % 4
    rotated_mask = _rotate_cw(full_mask, rot_steps)
    copy_cells, _top_left = _copy_cells(rotated_mask, final_dir, endpoint)
    if not copy_cells:
        return None
    if any(not (0 <= r < H and 0 <= c < W) for r, c in copy_cells):
        return None

    source_cells = {(center[0] + r - 1, center[1] + c - 1) for r, c in np.argwhere(full_mask)}
    tower_set = {tower}
    signal_set = set(signal_cells)
    orange_set = set(orange_cells)
    copy_set = set(copy_cells)

    if copy_set & (source_cells | tower_set | signal_set):
        return None

    return {
        "center": center,
        "emit_dir": emit_dir,
        "turn": turn,
        "tower": tower,
        "source_cells": source_cells,
        "signal_cells": signal_cells,
        "scan_cells": scan_cells,
        "orange_cells": orange_cells,
        "copy_cells": copy_cells,
        "final_dir": final_dir,
    }


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            if int(patch[r, c]) != BG:
                grid[top + r, left + c] = int(patch[r, c])


def _dilated(cells: set[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 28)
    W = random.randint(18, 28)
    n_sources = random.randint(2, 4)
    if 2 * n_sources > len(PALETTE):
        return None

    colors = random.sample(PALETTE, 2 * n_sources)
    turns = [-1, 0, 1]
    if n_sources >= 3:
        random.shuffle(turns)
        while len(turns) < n_sources:
            turns.append(random.choice([-1, 0, 1]))
    else:
        turns = random.sample([-1, 0, 1], n_sources)

    specs = []
    occupied_output: set[tuple[int, int]] = set()
    blocked_inputs = _dilated(set(), H, W)
    existing_towers: set[tuple[int, int]] = set()
    forbidden_tower_cells: set[tuple[int, int]] = set()

    for i in range(n_sources):
        body_color = colors[2 * i]
        signal_color = colors[2 * i + 1]
        emit_dir = random.choice(DIRS_CW)
        turn = turns[i]
        body_mask = _rotate_cw(random.choice(UP_BODY_MASKS), _dir_index(emit_dir))
        full_mask = _full_source_mask(body_mask)
        patch = _source_patch(body_mask, body_color, signal_color)

        placed = None
        for _ in range(150):
            center = (random.randint(2, H - 3), random.randint(2, W - 3))
            source_cells = {(center[0] + r - 1, center[1] + c - 1) for r, c in np.argwhere(full_mask)}
            if source_cells & blocked_inputs:
                continue

            cells = _render_spec_cells(H, W, center, emit_dir, turn, full_mask)
            if cells is None:
                continue
            if cells["tower"] in forbidden_tower_cells:
                continue
            if existing_towers & cells["scan_cells"]:
                continue

            final_cells = set(cells["source_cells"]) | {cells["tower"]} | set(cells["signal_cells"]) | set(cells["orange_cells"]) | set(cells["copy_cells"])
            if final_cells & occupied_output:
                continue
            if _dilated(cells["source_cells"] | {cells["tower"]}, H, W) & blocked_inputs:
                continue

            placed = {
                "body_color": body_color,
                "signal_color": signal_color,
                "body_mask": body_mask,
                "full_mask": full_mask,
                "patch": patch,
                **cells,
            }
            break

        if placed is None:
            return None

        specs.append(placed)
        final_cells = set(placed["source_cells"]) | {placed["tower"]} | set(placed["signal_cells"]) | set(placed["orange_cells"]) | set(placed["copy_cells"])
        occupied_output |= final_cells
        blocked_inputs |= _dilated(placed["source_cells"] | {placed["tower"]}, H, W)
        existing_towers.add(placed["tower"])
        forbidden_tower_cells |= placed["scan_cells"]

    inp = np.full((H, W), BG, dtype=int)
    expected = np.full((H, W), BG, dtype=int)

    for spec in specs:
        top = spec["center"][0] - 1
        left = spec["center"][1] - 1
        _draw_patch(inp, spec["patch"], top, left)
        _draw_patch(expected, spec["patch"], top, left)
        inp[spec["tower"]] = TOWER_COLOR
        expected[spec["tower"]] = TOWER_COLOR
        for r, c in spec["signal_cells"]:
            expected[r, c] = spec["signal_color"]
        for r, c in spec["orange_cells"]:
            expected[r, c] = TOWER_COLOR
        for r, c in spec["copy_cells"]:
            expected[r, c] = spec["signal_color"]

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-source puzzle")
