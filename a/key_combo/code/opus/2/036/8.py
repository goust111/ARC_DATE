from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
TEAL = 8
MAROON = 9

GRID_SHAPE = (19, 19)
TILE_SIZE = 5
TILE_STRIDE = 6
TILE_OFFSET = 1
SLOT_VALUES = (-1, 0, 1)
SLOTS = tuple((sr, sc) for sr in SLOT_VALUES for sc in SLOT_VALUES)
INTERIOR_CELLS = tuple((r, c) for r in range(1, 4) for c in range(1, 4))
LEGEND = np.array(
    [
        [BLUE, ORANGE, MAGENTA],
        [YELLOW, GREY, GREY],
        [RED, MAROON, GREEN],
    ],
    dtype=int,
)

PATTERN_BANK: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], ...]] = {
    (-1, -1): (
        ((1, 1), (1, 2)),
        ((1, 1), (2, 1)),
        ((1, 1), (1, 2), (2, 1)),
        ((1, 1), (1, 2), (2, 1), (2, 2)),
    ),
    (-1, 0): (
        ((1, 2), (2, 2)),
        ((1, 1), (1, 2), (1, 3)),
        ((1, 2), (2, 1), (2, 2), (2, 3)),
        ((1, 1), (1, 2), (1, 3), (2, 2)),
    ),
    (-1, 1): (
        ((1, 3), (1, 2)),
        ((1, 3), (2, 3)),
        ((1, 3), (1, 2), (2, 3)),
        ((1, 2), (1, 3), (2, 2), (2, 3)),
    ),
    (0, -1): (
        ((2, 1), (2, 2)),
        ((1, 1), (2, 1), (3, 1)),
        ((1, 2), (2, 1), (2, 2), (3, 2)),
        ((1, 1), (2, 1), (2, 2), (3, 1)),
    ),
    (0, 0): (
        ((1, 2), (2, 2), (3, 2)),
        ((2, 1), (2, 2), (2, 3)),
        ((1, 1), (2, 2), (3, 3)),
        ((1, 3), (2, 2), (3, 1)),
    ),
    (0, 1): (
        ((2, 2), (2, 3)),
        ((1, 3), (2, 3), (3, 3)),
        ((1, 2), (2, 2), (2, 3), (3, 2)),
        ((1, 3), (2, 2), (2, 3), (3, 3)),
    ),
    (1, -1): (
        ((3, 1), (2, 1)),
        ((3, 1), (3, 2)),
        ((3, 1), (2, 1), (3, 2)),
        ((2, 1), (2, 2), (3, 1), (3, 2)),
    ),
    (1, 0): (
        ((2, 2), (3, 2)),
        ((3, 1), (3, 2), (3, 3)),
        ((2, 1), (2, 2), (3, 2), (3, 3)),
        ((2, 2), (3, 1), (3, 2), (3, 3)),
    ),
    (1, 1): (
        ((3, 3), (2, 3)),
        ((3, 3), (3, 2)),
        ((2, 3), (3, 2), (3, 3)),
        ((2, 2), (2, 3), (3, 2), (3, 3)),
    ),
}

GEN_RETRY_LIMIT = 400


def _sign(value: int | float) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _slot_origin(slot: tuple[int, int]) -> tuple[int, int]:
    slot_r, slot_c = slot
    tile_r = slot_r + 1
    tile_c = slot_c + 1
    return TILE_OFFSET + TILE_STRIDE * tile_r, TILE_OFFSET + TILE_STRIDE * tile_c


def _tile_view(grid: np.ndarray, slot: tuple[int, int]) -> np.ndarray:
    r0, c0 = _slot_origin(slot)
    return grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]


def _separator_ok(grid: np.ndarray) -> bool:
    for idx in (0, 6, 12, 18):
        if np.any(grid[idx, :] != BLACK) or np.any(grid[:, idx] != BLACK):
            return False
    return True


def _pattern_slot(coords: Iterable[tuple[int, int]]) -> tuple[int, int]:
    row_sum = 0
    col_sum = 0
    for r, c in coords:
        row_sum += r - 2
        col_sum += c - 2
    return _sign(row_sum), _sign(col_sum)


def _classify_tile(tile: np.ndarray) -> str:
    colors = set(map(int, np.unique(tile)))
    if colors == {BLACK}:
        return "empty"
    if RED in colors:
        assert colors.issubset({RED, TEAL}), "fragment tiles use teal fill plus red markers"
        return "fragment"

    assert colors.issubset({BLACK, TEAL}), "border tiles use black and teal only"
    assert np.all(tile[[0, -1], :] == BLACK)
    assert np.all(tile[:, [0, -1]] == BLACK)
    assert np.any(tile[1:4, 1:4] == TEAL)
    assert np.any(tile[1:4, 1:4] == BLACK)
    return "border"


def _fragment_infos(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    fragments: list[dict] = []
    borders: list[tuple[int, int]] = []
    for slot in SLOTS:
        tile = _tile_view(grid, slot)
        kind = _classify_tile(tile)
        if kind == "fragment":
            coords = [tuple(map(int, rc)) for rc in np.argwhere(tile == RED)]
            assert 2 <= len(coords) <= 4
            fragments.append(
                {
                    "source": slot,
                    "target": _pattern_slot(coords),
                    "coords": tuple(coords),
                }
            )
        elif kind == "border":
            borders.append(slot)
    return fragments, borders


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move red marker fragments into the slot named by their centroid direction, then fill bordered slots with a fixed legend wherever black gaps remain.

    Input structure:
    - The input is a 19x19 black-separated 3x3 array of 5x5 slots, with black separator rows and columns at 0, 6, 12, and 18.
    - Fragment slots contain teal filler plus a small red marker fragment whose centroid direction relative to the tile center identifies one output slot.
    - Border slots contain a one-pixel black frame around a 3x3 interior of teal cells and black missing cells; all other slots are fully black.

    Rule:
    - For each fragment slot, compute the sign of the red fragment centroid offset from the local center (2,2) and map that sign pair to one of the nine global slots.
    - Remove the red pixels from the source slot by turning them black, and stamp the same red pixels into the matched target slot at the same local coordinates.
    - After all fragment moves, fill every remaining black cell inside each border slot's 3x3 interior with the fixed legend 1 7 6 / 4 5 5 / 2 9 3, while preserving existing teal cells and any moved red pixels.

    Key insight:
    - The fragment direction must be resolved before legend filling, because a fragment can land inside a border slot and block some legend positions from being recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE
    assert _separator_ok(g)

    fragments, borders = _fragment_infos(g)
    assert 3 <= len(fragments) <= 4
    assert 1 <= len(borders) <= 2

    out = g.copy()
    for info in fragments:
        src_tile = _tile_view(out, info["source"])
        dst_tile = _tile_view(out, info["target"])
        for r, c in info["coords"]:
            src_tile[r, c] = BLACK
        for r, c in info["coords"]:
            dst_tile[r, c] = RED

    for slot in borders:
        tile = _tile_view(out, slot)
        for rr in range(3):
            for cc in range(3):
                if tile[rr + 1, cc + 1] == BLACK:
                    tile[rr + 1, cc + 1] = int(LEGEND[rr, cc])

    return out


def _check_pattern_bank() -> None:
    for slot, patterns in PATTERN_BANK.items():
        assert slot in SLOTS
        assert patterns
        for coords in patterns:
            assert 2 <= len(coords) <= 4
            assert len(set(coords)) == len(coords)
            assert _pattern_slot(coords) == slot
            for r, c in coords:
                assert 1 <= r <= 3 and 1 <= c <= 3


_check_pattern_bank()


def _fragment_tile(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    tile = np.full((TILE_SIZE, TILE_SIZE), TEAL, dtype=int)
    for r, c in coords:
        tile[r, c] = RED
    return tile


def _border_tile(teal_cells: Iterable[tuple[int, int]]) -> np.ndarray:
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for r, c in teal_cells:
        tile[r, c] = TEAL
    return tile


def _build_input(
    fragments: list[dict],
    border_teal: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    for info in fragments:
        _tile_view(grid, info["source"])[:, :] = _fragment_tile(info["coords"])
    for slot, teal_cells in border_teal.items():
        _tile_view(grid, slot)[:, :] = _border_tile(teal_cells)
    return grid


def _oracle_output(
    inp: np.ndarray,
    fragments: list[dict],
    border_slots: tuple[tuple[int, int], ...],
) -> np.ndarray:
    out = inp.copy()
    for info in fragments:
        src_tile = _tile_view(out, info["source"])
        dst_tile = _tile_view(out, info["target"])
        for r, c in info["coords"]:
            src_tile[r, c] = BLACK
        for r, c in info["coords"]:
            dst_tile[r, c] = RED

    for slot in border_slots:
        tile = _tile_view(out, slot)
        for rr in range(3):
            for cc in range(3):
                if tile[rr + 1, cc + 1] == BLACK:
                    tile[rr + 1, cc + 1] = int(LEGEND[rr, cc])
    return out


def _pick_fragment_pattern(slot: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return random.choice(PATTERN_BANK[slot])


def _sample_border_teal(
    occupied_by_fragment: tuple[tuple[int, int], ...] | None,
) -> tuple[tuple[int, int], ...]:
    if occupied_by_fragment is None:
        available = list(INTERIOR_CELLS)
        teal_count = random.randint(1, 5)
    else:
        occupied = set(occupied_by_fragment)
        available = [cell for cell in INTERIOR_CELLS if cell not in occupied]
        teal_count = random.randint(1, min(4, len(available) - 1))
    return tuple(sorted(random.sample(available, teal_count)))


def _attempt() -> np.ndarray | None:
    n_fragments = random.choice((3, 4))
    n_borders = random.choice((1, 2))

    source_slots = random.sample(list(SLOTS), n_fragments)
    non_source_slots = [slot for slot in SLOTS if slot not in source_slots]
    border_slots = random.sample(non_source_slots, n_borders)
    empty_slots = [slot for slot in non_source_slots if slot not in border_slots]

    targeted_border_count = random.randint(1, min(n_fragments, n_borders))
    targeted_border_slots = random.sample(border_slots, targeted_border_count)
    target_slots = targeted_border_slots + random.sample(empty_slots, n_fragments - targeted_border_count)
    random.shuffle(target_slots)

    fragments: list[dict] = []
    border_targets = {slot: None for slot in border_slots}
    for source, target in zip(source_slots, target_slots):
        coords = _pick_fragment_pattern(target)
        fragments.append({"source": source, "target": target, "coords": coords})
        if target in border_targets:
            border_targets[target] = coords

    border_teal = {
        slot: _sample_border_teal(border_targets[slot]) for slot in border_slots
    }
    inp = _build_input(fragments, border_teal)

    if not _separator_ok(inp):
        return None

    expected = _oracle_output(inp, fragments, tuple(border_slots))
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
