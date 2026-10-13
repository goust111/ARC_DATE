from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
START = 3
BLOCKER = 8
SEED_PALETTE = (1, 2, 4, 6, 7, 9)

TURN_MIN = 2
TURN_MAX = 5
MAX_SEG_LEN = 5
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _left_boundary_pixels(h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r in range(h):
        out.add((r, 0))
        out.add((r, w - 1))
    for c in range(w):
        out.add((0, c))
        out.add((h - 1, c))
    return out


def _find_separator_column(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    assert len(cols) == 1, "expected exactly one separator column"
    return int(cols[0])


def _find_start_bar(left: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(left == START)]
    assert len(coords) == 2, "expected a 2-pixel start bar"
    assert abs(coords[0][0] - coords[1][0]) + abs(coords[0][1] - coords[1][1]) == 1

    boundary = _left_boundary_pixels(*left.shape)
    border_pixels = [p for p in coords if p in boundary]
    assert len(border_pixels) == 1, "expected exactly one boundary start pixel"
    border = border_pixels[0]
    inner = coords[1] if coords[0] == border else coords[0]
    return border, inner


def _count_turns(left: np.ndarray) -> int:
    border, inner = _find_start_bar(left)
    dr = int(inner[0] - border[0])
    dc = int(inner[1] - border[1])
    r, c = inner
    turns = 0
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "corridor walk should not cycle"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(left.shape, nr, nc):
            break
        if int(left[nr, nc]) != BLOCKER:
            r, c = nr, nc
            continue

        choices: list[tuple[int, int]] = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = r + tdr, c + tdc
            if _in_bounds(left.shape, tr, tc) and int(left[tr, tc]) != BLOCKER:
                choices.append((tdr, tdc))
        assert len(choices) == 1, "each collision must force a unique turn"
        dr, dc = choices[0]
        turns += 1

    return turns


def _extract_seed(right: np.ndarray) -> np.ndarray:
    coords = np.argwhere(right != BG)
    assert len(coords) > 0, "expected a non-empty seed tile"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r0) == 0 and int(c0) == 0, "seed tile must start at the right area's top-left corner"
    return right[: int(r1) + 1, : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count corridor turns on the left side and use that count as the seed tile's tiling factor.

    Input structure:
    - A vertical grey separator splits the grid into a left corridor panel and a right seed panel.
    - In the left panel, a green 2-pixel start bar touches the panel boundary and teal blocker pixels force a one-cell-wide turning corridor.
    - In the right panel, a small multicolor seed tile sits at the panel's top-left corner on a black background.

    Rule:
    - Find the start bar in the left panel, identify its unique boundary pixel, and walk from the inner pixel in the inward direction.
    - Move straight through non-blocker cells; whenever the next cell is a blocker, turn to the unique available perpendicular direction and count one turn; stop when the next step would leave the left panel.
    - Extract the minimal top-left-anchored seed tile from the right panel and tile it vertically and horizontally by the counted turn total.

    Key insight:
    - The corridor is only a turn-counting device: its topology computes a number that controls an otherwise unrelated geometric tiling transform on the seed tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_column(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0, "separator must leave non-empty panels on both sides"

    turns = _count_turns(left)
    assert turns >= 1
    seed = _extract_seed(right)
    return np.tile(seed, (turns, turns))


def _choose_entry(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    side = random.choices(
        ["top", "bottom", "left", "right"],
        weights=[2, 2, 2, 3],
        k=1,
    )[0]
    if side == "top":
        c = random.randint(1, w - 2)
        return (0, c), (1, c), (1, 0)
    if side == "bottom":
        c = random.randint(1, w - 2)
        return (h - 1, c), (h - 2, c), (-1, 0)
    if side == "left":
        r = random.randint(1, h - 2)
        return (r, 0), (r, 1), (0, 1)
    r = random.randint(1, h - 2)
    return (r, w - 1), (r, w - 2), (0, -1)


def _free_run(
    h: int,
    w: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    path_cells: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], bool]:
    cells: list[tuple[int, int]] = []
    rr, cc = r, c
    while True:
        nr, nc = rr + dr, cc + dc
        if not _in_bounds((h, w), nr, nc):
            return cells, True
        if (nr, nc) in path_cells or (nr, nc) in blockers:
            return cells, False
        cells.append((nr, nc))
        rr, cc = nr, nc


def _search_corridor(
    h: int,
    w: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    remaining_turns: int,
    path_cells: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if remaining_turns == 0:
        final_cells, exits = _free_run(h, w, r, c, dr, dc, path_cells, blockers)
        if len(final_cells) < 1 or not exits:
            return None
        return final_cells, []

    free_cells, _ = _free_run(h, w, r, c, dr, dc, path_cells, blockers)
    max_seg = min(MAX_SEG_LEN, len(free_cells) - 1)
    if max_seg < 1:
        return None

    seg_lengths = list(range(1, max_seg + 1))
    random.shuffle(seg_lengths)
    for seg_len in seg_lengths:
        segment = free_cells[:seg_len]
        end_r, end_c = segment[-1]
        ahead = (end_r + dr, end_c + dc)
        if not _in_bounds((h, w), ahead[0], ahead[1]) or ahead in path_cells or ahead in blockers:
            continue

        path_next = path_cells | set(segment)
        turn_dirs = [(-dc, dr), (dc, -dr)]
        random.shuffle(turn_dirs)
        for ndr, ndc in turn_dirs:
            side = (end_r + ndr, end_c + ndc)
            wrong = (end_r - ndr, end_c - ndc)
            if not _in_bounds((h, w), side[0], side[1]):
                continue
            if side in path_next or side in blockers:
                continue
            if _in_bounds((h, w), wrong[0], wrong[1]) and (wrong in path_next or wrong in blockers):
                continue

            added_blockers = [ahead]
            next_blockers = blockers | {ahead}
            if _in_bounds((h, w), wrong[0], wrong[1]):
                added_blockers.append(wrong)
                next_blockers.add(wrong)

            result = _search_corridor(
                h,
                w,
                end_r,
                end_c,
                ndr,
                ndc,
                remaining_turns - 1,
                path_next,
                next_blockers,
            )
            if result is None:
                continue
            tail_path, tail_blockers = result
            return segment + tail_path, added_blockers + tail_blockers

    return None


def _build_left_panel(h: int, w: int, turns: int) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    if h < 6 or w < 6:
        return None
    border, inner, direction = _choose_entry(h, w)
    path_cells = {border, inner}
    result = _search_corridor(
        h,
        w,
        inner[0],
        inner[1],
        direction[0],
        direction[1],
        turns,
        path_cells,
        set(),
    )
    if result is None:
        return None
    traversed, blockers = result
    all_path = path_cells | set(traversed)

    panel = np.full((h, w), BG, dtype=int)
    panel[border] = START
    panel[inner] = START
    for r, c in blockers:
        panel[r, c] = BLOCKER

    extras = random.randint(0, 3)
    safe = []
    for r in range(h):
        for c in range(w):
            if panel[r, c] != BG or (r, c) in all_path:
                continue
            if any((r + dr, c + dc) in all_path for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                continue
            safe.append((r, c))
    random.shuffle(safe)
    for r, c in safe[:extras]:
        panel[r, c] = BLOCKER

    if _count_turns(panel) != turns:
        return None
    return panel, all_path


def _make_seed(tile_h: int, tile_w: int) -> np.ndarray:
    palette = random.sample(SEED_PALETTE, k=random.randint(2, min(4, len(SEED_PALETTE))))
    weights = [0.25] + [0.75 / len(palette)] * len(palette)
    seed = np.random.choice([BG] + palette, size=(tile_h, tile_w), p=weights).astype(int)

    seed[0, 0] = palette[0]
    seed[tile_h - 1, tile_w - 1] = palette[1]
    if tile_w > 1:
        seed[0, tile_w - 1] = palette[-1]
    if tile_h > 1:
        seed[tile_h - 1, 0] = palette[min(1, len(palette) - 1)]

    if len(set(map(int, seed.flatten())) - {BG}) < 2:
        seed[0, 0] = palette[0]
        seed[tile_h - 1, tile_w - 1] = palette[1]

    if np.count_nonzero(seed == BG) == 0 and random.random() < 0.5:
        rr = random.randint(0, tile_h - 1)
        cc = random.randint(0, tile_w - 1)
        if (rr, cc) not in {(0, 0), (tile_h - 1, tile_w - 1)}:
            seed[rr, cc] = BG
    return seed


def _validate_instance(grid: np.ndarray, turns: int, seed: np.ndarray) -> bool:
    sep = _find_separator_column(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    if _count_turns(left) != turns:
        return False
    extracted = _extract_seed(right)
    if not np.array_equal(extracted, seed):
        return False
    expected = np.tile(seed, (turns, turns))
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        turns = random.randint(TURN_MIN, TURN_MAX)
        tile_h, tile_w = random.choice([(2, 2), (2, 3), (3, 2), (3, 3)])

        right_h_needed = turns * tile_h
        h_min = max(right_h_needed, 2 * turns + 6)
        h_max = min(24, h_min + 6)
        if h_min > h_max:
            continue
        h = random.randint(h_min, h_max)

        extra_right = random.randint(0, 4)
        right_w = turns * tile_w + extra_right
        left_w_min = max(8, turns + 4)
        left_w_max = min(13, 29 - right_w)
        if left_w_min > left_w_max:
            continue
        left_w = random.randint(left_w_min, left_w_max)

        built = _build_left_panel(h, left_w, turns)
        if built is None:
            continue
        left, _path = built

        seed = _make_seed(tile_h, tile_w)
        right = np.full((h, right_w), BG, dtype=int)
        right[:tile_h, :tile_w] = seed

        grid = np.full((h, left_w + 1 + right_w), BG, dtype=int)
        grid[:, :left_w] = left
        grid[:, left_w] = SEPARATOR
        grid[:, left_w + 1 :] = right

        if max(grid.shape) > 30:
            continue
        if max((turns * tile_h, turns * tile_w)) > 30:
            continue
        if _validate_instance(grid, turns, seed):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
