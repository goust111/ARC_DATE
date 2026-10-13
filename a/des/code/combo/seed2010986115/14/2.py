from __future__ import annotations

import random
import numpy as np


_DIRECTIONS = ("up", "right", "down", "left")
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Canonical 5-pixel directional arrows. The middle pixel of the leading 3-cell edge
# is treated as the tip that must align to a border slot.
_T_SHAPES = {
    "up": {
        "cells": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
        "tip": (0, 1),
    },
    "right": {
        "cells": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
        "tip": (1, 2),
    },
    "down": {
        "cells": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
        "tip": (2, 1),
    },
    "left": {
        "cells": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
        "tip": (1, 0),
    },
}

# The prose example describes a 5-pixel cross-like variant. It is only directional
# when exactly one aligned border slot is available, so the solver accepts it too.
_CROSS_SHAPE = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
_CROSS_TIPS = {
    "up": (0, 1),
    "right": (1, 2),
    "down": (2, 1),
    "left": (1, 0),
}

_BACK_STEP = {
    "up": (1, 0),
    "right": (0, -1),
    "down": (-1, 0),
    "left": (0, 1),
}

_GEN_BG = 0
_GEN_PALETTE = tuple(range(1, 10))
_GEN_SIZE_RANGE = (12, 19)
_GEN_ARROW_COUNT_RANGE = (2, 6)
_GEN_RETRIES = 400
_GEN_EDGE_MARGIN = 2
_GEN_PADDING = 1
_GEN_MIN_SHIFT = 2


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _normalize(coords: list[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return frozenset((r - r0, c - c0) for r, c in coords), int(r0), int(c0)


def _slot_color(grid: np.ndarray, bg: int, direction: str, tip_r: int, tip_c: int) -> int | None:
    h, w = grid.shape
    if direction == "up":
        if 0 <= tip_c < w and int(grid[0, tip_c]) != bg:
            return int(grid[0, tip_c])
        return None
    if direction == "down":
        if 0 <= tip_c < w and int(grid[h - 1, tip_c]) != bg:
            return int(grid[h - 1, tip_c])
        return None
    if direction == "left":
        if 0 <= tip_r < h and int(grid[tip_r, 0]) != bg:
            return int(grid[tip_r, 0])
        return None
    if direction == "right":
        if 0 <= tip_r < h and int(grid[tip_r, w - 1]) != bg:
            return int(grid[tip_r, w - 1])
        return None
    raise AssertionError(direction)


def _identify_arrow(
    grid: np.ndarray, bg: int, coords: list[tuple[int, int]]
) -> tuple[str, tuple[int, int], int]:
    assert len(coords) == 5, "each arrow should have exactly 5 pixels"
    norm, r0, c0 = _normalize(coords)
    candidates: list[tuple[str, tuple[int, int], int]] = []

    for direction, spec in _T_SHAPES.items():
        if norm != spec["cells"]:
            continue
        tr, tc = spec["tip"]
        tip = (int(r0 + tr), int(c0 + tc))
        color = _slot_color(grid, bg, direction, tip[0], tip[1])
        assert color is not None, "directional arrow tip must align with a border slot"
        candidates.append((direction, tip, color))

    if norm == _CROSS_SHAPE:
        for direction, (tr, tc) in _CROSS_TIPS.items():
            tip = (int(r0 + tr), int(c0 + tc))
            color = _slot_color(grid, bg, direction, tip[0], tip[1])
            if color is not None:
                candidates.append((direction, tip, color))

    assert len(candidates) == 1, f"expected a unique arrow interpretation, got {candidates}"
    return candidates[0]


def _translated(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each 5-pixel arrow sprite straight to its aligned border slot and recolor it
    with the slot's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)

    interior = g.copy()
    interior[0, :] = bg
    interior[h - 1, :] = bg
    interior[:, 0] = bg
    interior[:, w - 1] = bg
    comps = _components4(interior != bg)
    assert comps, "expected at least one interior arrow"

    out = g.copy()
    for comp in comps:
        for r, c in comp:
            out[r, c] = bg

    for comp in comps:
        direction, tip, slot_color = _identify_arrow(g, bg, comp)
        if direction == "up":
            dr, dc = -tip[0], 0
        elif direction == "down":
            dr, dc = h - 1 - tip[0], 0
        elif direction == "left":
            dr, dc = 0, -tip[1]
        elif direction == "right":
            dr, dc = 0, w - 1 - tip[1]
        else:
            raise AssertionError(direction)

        moved = _translated(comp, dr, dc)
        target_tip = (int(tip[0] + dr), int(tip[1] + dc))
        assert 0 <= target_tip[0] < h and 0 <= target_tip[1] < w

        for r, c in moved:
            assert 0 <= r < h and 0 <= c < w, "moved arrow left the grid"
            current = int(out[r, c])
            if (r, c) == target_tip:
                assert current == slot_color, "tip must land on its slot"
            else:
                assert current == bg, "moved arrows must not overwrite other content"

        for r, c in moved:
            out[r, c] = slot_color

    return out


def _shape_cells(direction: str, tip_r: int, tip_c: int) -> list[tuple[int, int]]:
    spec = _T_SHAPES[direction]
    tr, tc = spec["tip"]
    return [(int(tip_r + r - tr), int(tip_c + c - tc)) for r, c in spec["cells"]]


def _pad_mark(mask: np.ndarray, cells: list[tuple[int, int]], pad: int) -> None:
    h, w = mask.shape
    for r, c in cells:
        r0 = max(0, int(r - pad))
        r1 = min(h, int(r + pad + 1))
        c0 = max(0, int(c - pad))
        c1 = min(w, int(c + pad + 1))
        mask[r0:r1, c0:c1] = True


def _cells_clear(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not bool(mask[r, c]) for r, c in cells)


def _inner_cells(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells)


def _border_side(pos: tuple[int, int], h: int, w: int) -> str:
    r, c = pos
    if r == 0:
        return "up"
    if c == w - 1:
        return "right"
    if r == h - 1:
        return "down"
    if c == 0:
        return "left"
    raise AssertionError(pos)


def _side_positions(direction: str, h: int, w: int, margin: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(0, c) for c in range(margin, w - margin)]
    if direction == "down":
        return [(h - 1, c) for c in range(margin, w - margin)]
    if direction == "left":
        return [(r, 0) for r in range(margin, h - margin)]
    if direction == "right":
        return [(r, w - 1) for r in range(margin, h - margin)]
    raise AssertionError(direction)


def _shift_candidates(cells: list[tuple[int, int]], direction: str, h: int, w: int) -> list[int]:
    dr, dc = _BACK_STEP[direction]
    steps: list[int] = []
    step = _GEN_MIN_SHIFT
    while True:
        shifted = _translated(cells, dr * step, dc * step)
        if not _inner_cells(shifted, h, w):
            break
        steps.append(step)
        step += 1
    return steps


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*_GEN_SIZE_RANGE)
    w = random.randint(*_GEN_SIZE_RANGE)
    max_arrows = min(_GEN_ARROW_COUNT_RANGE[1], max(2, (h + w) // 6))
    arrow_count = random.randint(_GEN_ARROW_COUNT_RANGE[0], max_arrows)

    arrow_color = random.choice(_GEN_PALETTE)
    side_palette = [c for c in _GEN_PALETTE if c != arrow_color]
    random.shuffle(side_palette)
    side_colors = {
        "up": side_palette[0],
        "right": side_palette[1],
        "down": side_palette[2],
        "left": side_palette[3],
    }

    side_k = random.randint(2, min(4, arrow_count))
    active_sides = random.sample(list(_DIRECTIONS), side_k)
    directions = active_sides + [random.choice(active_sides) for _ in range(arrow_count - side_k)]
    random.shuffle(directions)

    out_forbid = np.zeros((h, w), dtype=bool)
    in_forbid = np.zeros((h, w), dtype=bool)
    slot_map: dict[tuple[int, int], int] = {}
    final_arrows: list[tuple[str, tuple[int, int], list[tuple[int, int]]]] = []
    input_arrows: list[list[tuple[int, int]]] = []

    for direction in directions:
        tips = _side_positions(direction, h, w, _GEN_EDGE_MARGIN)
        random.shuffle(tips)
        placed = False

        for tip in tips:
            if tip in slot_map:
                continue
            final_cells = _shape_cells(direction, tip[0], tip[1])
            if not _cells_clear(out_forbid, final_cells):
                continue

            shifts = _shift_candidates(final_cells, direction, h, w)
            random.shuffle(shifts)
            for step in shifts:
                dr, dc = _BACK_STEP[direction]
                input_cells = _translated(final_cells, dr * step, dc * step)
                if not _cells_clear(in_forbid, input_cells):
                    continue

                slot_map[tip] = side_colors[direction]
                final_arrows.append((direction, tip, final_cells))
                input_arrows.append(input_cells)
                _pad_mark(out_forbid, final_cells, _GEN_PADDING)
                _pad_mark(in_forbid, input_cells, _GEN_PADDING)
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    blocked_slots = {
        (r, c)
        for _direction, _tip, cells in final_arrows
        for r, c in cells
        if r == 0 or r == h - 1 or c == 0 or c == w - 1
    }

    for side in _DIRECTIONS:
        existing = [pos for pos in slot_map if _border_side(pos, h, w) == side]
        target_total = len(existing)
        if target_total == 0:
            target_total = 1
        target_total += random.randint(0, 2)

        candidates = [pos for pos in _side_positions(side, h, w, 1) if pos not in blocked_slots and pos not in slot_map]
        random.shuffle(candidates)
        for pos in candidates:
            if len(existing) >= target_total:
                break
            same_side_gap_ok = all(
                abs(pos[0] - other[0]) + abs(pos[1] - other[1]) >= 2
                for other in existing
            )
            if not same_side_gap_ok:
                continue
            slot_map[pos] = side_colors[side]
            existing.append(pos)

        if len(existing) < 1:
            return None

    inp = np.full((h, w), _GEN_BG, dtype=int)
    expected = np.full((h, w), _GEN_BG, dtype=int)

    for pos, color in slot_map.items():
        inp[pos] = int(color)
        expected[pos] = int(color)

    for cells in input_arrows:
        for r, c in cells:
            if inp[r, c] != _GEN_BG:
                return None
            inp[r, c] = int(arrow_color)

    for _direction, tip, cells in final_arrows:
        color = int(slot_map[tip])
        for r, c in cells:
            expected[r, c] = color

    if np.array_equal(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _attempt_generate()
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid arrow-docking instance")
