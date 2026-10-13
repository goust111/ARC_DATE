from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PALETTE = (1, 2, 3, 6, 7, 8, 9)
TILE_SIZE = 3
MACRO_SIZE = 3
GRID_SIZE = 11
SEPARATORS = (3, 7)
ALL_CELLS = [(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE)]
ALL_SLOTS = [(r, c) for r in range(MACRO_SIZE) for c in range(MACRO_SIZE)]
MAX_GEN_ATTEMPTS = 400


def _bands(size: int, separators: tuple[int, int]) -> list[tuple[int, int]]:
    starts = [0] + [s + 1 for s in separators]
    ends = list(separators) + [size]
    return list(zip(starts, ends))


def _tile_bounds(slot: tuple[int, int]) -> tuple[slice, slice]:
    br, bc = slot
    r0 = br * (TILE_SIZE + 1)
    c0 = bc * (TILE_SIZE + 1)
    return slice(r0, r0 + TILE_SIZE), slice(c0, c0 + TILE_SIZE)


def _empty_canvas() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BLACK, dtype=int)
    grid[list(SEPARATORS), :] = GREY
    grid[:, list(SEPARATORS)] = GREY
    return grid


def _slot_tile(grid: np.ndarray, slot: tuple[int, int]) -> np.ndarray:
    rs, cs = _tile_bounds(slot)
    return grid[rs, cs]


def _parse(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected an 11x11 canvas"

    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert row_seps == list(SEPARATORS) and col_seps == list(SEPARATORS), "expected fixed grey separators"

    row_bands = _bands(g.shape[0], SEPARATORS)
    col_bands = _bands(g.shape[1], SEPARATORS)
    assert row_bands == col_bands == [(0, 3), (4, 7), (8, 11)]

    tiles: dict[tuple[int, int], dict] = {}
    occupied: list[tuple[int, int]] = []
    for slot in ALL_SLOTS:
        tile = _slot_tile(g, slot).copy()
        assert not np.any(tile == GREY), "grey may appear only on separator lines"
        non_black = np.argwhere(tile != BLACK)
        if non_black.size == 0:
            continue

        marker_coords = np.argwhere(tile == YELLOW)
        assert marker_coords.shape == (1, 2), "every non-empty tile must contain exactly one yellow marker"
        marker = tuple(map(int, marker_coords[0]))

        patch = tile.copy()
        patch[marker] = BLACK
        assert np.any(patch != BLACK), "every non-empty tile must contain colored pattern pixels besides the marker"

        tiles[slot] = {"marker": marker, "patch": patch}
        occupied.append(slot)

    assert occupied, "expected at least one non-empty tile"

    edges = {slot: tuple(map(int, tiles[slot]["marker"])) for slot in occupied}
    for slot, nxt in edges.items():
        assert nxt in tiles, f"pointer from {slot} must target another occupied slot"

    incoming = {slot: 0 for slot in occupied}
    for nxt in edges.values():
        incoming[nxt] += 1

    heads = [slot for slot in occupied if incoming[slot] == 0]
    assert len(heads) == 1, "expected exactly one unpointed chain head"
    head = heads[0]

    tails = [slot for slot in occupied if edges[slot] == slot]
    assert len(tails) == 1, "expected exactly one self-pointing tail"
    tail = tails[0]

    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    cur = head
    while True:
        assert cur not in seen, "pointer chain must not cycle before the tail"
        seen.add(cur)
        order.append(cur)
        nxt = edges[cur]
        if nxt == cur:
            break
        cur = nxt

    assert set(order) == set(occupied), "head traversal must visit every non-empty tile exactly once"
    assert order[-1] == tail, "tail must be the last visited tile"

    composite = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    last_painter = np.full((TILE_SIZE, TILE_SIZE), -1, dtype=int)
    paint_history: dict[tuple[int, int], list[tuple[int, int]]] = {(r, c): [] for r, c in ALL_CELLS}
    paint_colors: dict[tuple[int, int], list[int]] = {(r, c): [] for r, c in ALL_CELLS}
    for idx, slot in enumerate(order):
        patch = tiles[slot]["patch"]
        mask = patch != BLACK
        composite[mask] = patch[mask]
        last_painter[mask] = idx
        for r, c in np.argwhere(mask):
            cell = (int(r), int(c))
            paint_history[cell].append(idx)
            paint_colors[cell].append(int(patch[cell]))

    return {
        "grid": g,
        "tiles": tiles,
        "occupied": occupied,
        "edges": edges,
        "incoming": incoming,
        "head": head,
        "tail": tail,
        "order": order,
        "composite": composite,
        "last_painter": last_painter,
        "paint_history": paint_history,
        "paint_colors": paint_colors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Follow the yellow pointer chain, overlay its tile patterns, and stamp the composite into the tail slot.

    Input structure:
    - An 11x11 canvas is partitioned by full grey separator rows and columns into a 3x3 array of 3x3 tile slots.
    - Some slots are empty black; every non-empty slot contains colored pattern pixels plus exactly one yellow marker pixel.
    - The yellow marker sits on an otherwise empty tile cell, and its local 3x3 coordinate names the next macro slot in a single pointer chain.

    Rule:
    - Find the unique non-empty slot that no pointer targets, then follow pointers until the unique self-pointing tail.
    - Overlay the colored pixels of the visited tiles, ignoring yellow markers; later tiles overwrite earlier colors on overlapping cells.
    - Clear every tile slot to black, preserve the grey separators, and write the composite 3x3 stamp only into the tail slot.

    Key insight:
    - The yellow pixel is not part of the picture: it is a coordinate pointer from tile-local space into macro-slot space, and the chain order determines overlay priority.
    """
    info = _parse(grid)
    out = _empty_canvas()
    rs, cs = _tile_bounds(info["tail"])
    out[rs, cs] = info["composite"]
    return out


def _assign_exclusive_cells(
    yellows: list[tuple[int, int]], overlap_cell: tuple[int, int]
) -> list[tuple[int, int]] | None:
    options = [
        [cell for cell in ALL_CELLS if cell != overlap_cell and cell != yellows[idx]]
        for idx in range(len(yellows))
    ]
    order = sorted(range(len(yellows)), key=lambda idx: len(options[idx]))
    chosen: list[tuple[int, int] | None] = [None] * len(yellows)
    used: set[tuple[int, int]] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        cells = options[idx][:]
        random.shuffle(cells)
        for cell in cells:
            if cell in used:
                continue
            used.add(cell)
            chosen[idx] = cell
            if backtrack(pos + 1):
                return True
            used.remove(cell)
            chosen[idx] = None
        return False

    if not backtrack(0):
        return None
    return [cell for cell in chosen if cell is not None]


def _make_input_from_chain(chain_slots: list[tuple[int, int]]) -> np.ndarray | None:
    length = len(chain_slots)
    yellows = chain_slots[1:] + [chain_slots[-1]]
    shared_candidates = [cell for cell in ALL_CELLS if cell not in set(yellows)]
    if not shared_candidates:
        return None
    overlap_cell = random.choice(shared_candidates)

    exclusive_cells = _assign_exclusive_cells(yellows, overlap_cell)
    if exclusive_cells is None:
        return None

    overlap_colors = random.sample(PALETTE, length)
    exclusive_colors = []
    for overlap_color in overlap_colors:
        choices = [color for color in PALETTE if color != overlap_color]
        exclusive_colors.append(random.choice(choices))

    grid = _empty_canvas()
    for idx, slot in enumerate(chain_slots):
        tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
        tile[overlap_cell] = overlap_colors[idx]
        tile[exclusive_cells[idx]] = exclusive_colors[idx]
        tile[yellows[idx]] = YELLOW
        rs, cs = _tile_bounds(slot)
        grid[rs, cs] = tile
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        info = _parse(grid)
    except AssertionError:
        return False

    order = info["order"]
    if not (3 <= len(order) <= 6):
        return False
    if len(info["occupied"]) >= len(ALL_SLOTS):
        return False

    tail = info["tail"]
    composite = info["composite"]
    patches = [info["tiles"][slot]["patch"] for slot in order]

    if any(np.count_nonzero(patch) < 2 for patch in patches):
        return False

    if any(not np.any(info["last_painter"] == idx) for idx in range(len(order))):
        return False

    overwrite_cells = []
    for cell in ALL_CELLS:
        history = info["paint_history"][cell]
        colors = info["paint_colors"][cell]
        if len(history) >= 2 and len(set(colors)) >= 2:
            overwrite_cells.append(cell)
    if not overwrite_cells:
        return False

    if any(np.array_equal(composite, patch) for patch in patches):
        return False

    out = main(grid)
    expected = _empty_canvas()
    rs, cs = _tile_bounds(tail)
    expected[rs, cs] = composite
    if not np.array_equal(out, expected):
        return False

    # Reject accidental symmetry where reversing the chain would produce the same stamp.
    reverse_stamp = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for patch in reversed(patches):
        mask = patch != BLACK
        reverse_stamp[mask] = patch[mask]
    if np.array_equal(reverse_stamp, composite):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        chain_length = random.randint(3, 6)
        chain_slots = random.sample(ALL_SLOTS, chain_length)
        grid = _make_input_from_chain(chain_slots)
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain overlay input")
