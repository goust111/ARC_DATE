from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
BASE = 8
COUNTER = 5
MARKER_COLORS = (1, 2, 3, 4)

RECT_SIDE_OPTIONS = (5, 6)
RECT_COUNT_OPTIONS = (3, 4)
GRID_SIDE_RANGE = (14, 18)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _components(g != bg)
    assert 3 <= len(comps) <= 4

    rects: list[dict] = []
    for coords in comps:
        box = _bbox(coords)
        patch = _crop(g, box)
        assert bool(np.all(patch != bg))
        assert bool(np.isin(patch, (BASE, COUNTER, *MARKER_COLORS)).all())
        priority = int(np.count_nonzero(patch[-1] == COUNTER))
        clean = patch.copy()
        clean[clean == COUNTER] = BASE
        rects.append(
            {
                "bbox": box,
                "patch": patch,
                "clean": clean,
                "priority": priority,
                "markers": {
                    (int(r), int(c)): int(clean[r, c])
                    for r, c in np.argwhere(clean != BASE)
                },
            }
        )

    shapes = {rect["clean"].shape for rect in rects}
    assert len(shapes) == 1
    priorities = [rect["priority"] for rect in rects]
    assert len(set(priorities)) == len(priorities)
    assert all(0 <= priority <= 3 for priority in priorities)
    return rects


def _merge_in_order(rects: list[dict], order: list[int] | tuple[int, ...]) -> np.ndarray:
    h, w = rects[0]["clean"].shape
    out = np.full((h, w), BASE, dtype=int)
    for idx in order:
        patch = rects[int(idx)]["clean"]
        marker_mask = patch != BASE
        out[marker_mask] = patch[marker_mask]
    return out


def _semantic_summary(grid: np.ndarray) -> dict:
    rects = _parse_rectangles(grid)
    priority_order = sorted(range(len(rects)), key=lambda idx: rects[idx]["priority"])
    merged = _merge_in_order(rects, priority_order)

    conflicts: list[dict] = []
    winners: list[int] = []
    h, w = merged.shape
    for r in range(h):
        for c in range(w):
            participants = [
                idx
                for idx, rect in enumerate(rects)
                if (r, c) in rect["markers"]
            ]
            if len(participants) < 2:
                continue
            colors = {rects[idx]["markers"][(r, c)] for idx in participants}
            if len(colors) < 2:
                continue
            winner = max(participants, key=lambda idx: rects[idx]["priority"])
            winners.append(int(winner))
            conflicts.append(
                {
                    "pos": (r, c),
                    "participants": tuple(participants),
                    "winner": int(winner),
                }
            )

    visible_layers = set()
    for idx, rect in enumerate(rects):
        for (r, c), color in rect["markers"].items():
            if int(merged[r, c]) == int(color):
                visible_layers.add(int(idx))
                break

    return {
        "rects": rects,
        "priority_order": priority_order,
        "merged": merged,
        "conflicts": conflicts,
        "conflict_winners": winners,
        "visible_layers": visible_layers,
    }


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        summary = _semantic_summary(grid)
    except AssertionError:
        return False

    rects = summary["rects"]
    merged = summary["merged"]
    conflicts = summary["conflicts"]
    conflict_winners = summary["conflict_winners"]

    if len(conflicts) < 2:
        return False
    if len(set(conflict_winners)) < 2:
        return False
    if len(summary["visible_layers"]) != len(rects):
        return False
    if any(np.array_equal(merged, rect["clean"]) for rect in rects):
        return False

    variants = {
        _merge_in_order(rects, order).tobytes()
        for order in itertools.permutations(range(len(rects)))
    }
    if len(variants) < 3:
        return False

    return bool(np.array_equal(main(grid), merged))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge same-sized teal rectangles into one rectangle using gray-count priority to resolve marker conflicts.

    Input structure:
    - A black background contains 3 or 4 separated filled rectangles of the same size.
    - Every rectangle uses teal as its shared base color, contains colored marker pixels in matching local coordinates,
      and may have gray pixels only on its bottom row.
    - The bottom-row gray count differs across rectangles and acts as a hidden priority code.

    Rule:
    - Detect the same-sized non-black rectangles and count gray pixels on each rectangle's bottom row.
    - Treat more gray pixels as higher priority, then replace all gray pixels with the teal base color.
    - Create one output rectangle of the shared size, fill it with teal, and overlay each rectangle's non-teal markers
      from low priority to high priority so later higher-priority markers overwrite earlier conflicting ones.

    Key insight:
    - The rectangles are aligned copies of one local coordinate frame, but the gray counter pixels only encode layer order,
      so counting must happen before overlaying to know which color wins at shared marker positions.
    """
    rects = _parse_rectangles(grid)
    order = sorted(range(len(rects)), key=lambda idx: rects[idx]["priority"])
    return _merge_in_order(rects, order)


def _sample_priorities(count: int) -> list[int]:
    values = list(range(4)) if count == 4 else random.sample(range(4), count)
    random.shuffle(values)
    return [int(v) for v in values]


def _pick_conflict_winners(priorities: list[int]) -> list[int]:
    order = sorted(range(len(priorities)), key=lambda idx: priorities[idx])
    winners = [order[1], order[-1]]
    if len(order) == 4 and random.random() < 0.5:
        middle = random.choice(order[1:-1])
        if middle not in winners:
            winners.append(middle)
    return winners


def _build_assignments(side: int, priorities: list[int]) -> list[dict[tuple[int, int], int]] | None:
    count = len(priorities)
    interior = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
    needed = count + 2
    if len(interior) < needed:
        return None

    random.shuffle(interior)
    assignments = [dict() for _ in range(count)]

    exclusive_positions = interior[:count]
    cursor = count
    for layer, pos in enumerate(exclusive_positions):
        assignments[layer][pos] = random.choice(MARKER_COLORS)

    winners = _pick_conflict_winners(priorities)
    for winner in winners:
        if cursor >= len(interior):
            return None
        pos = interior[cursor]
        cursor += 1
        lower = [idx for idx, value in enumerate(priorities) if value < priorities[winner]]
        if not lower:
            return None
        participant_count = random.randint(2, min(3, len(lower) + 1))
        participants = [winner] + random.sample(lower, participant_count - 1)
        colors = random.sample(MARKER_COLORS, len(participants))
        for layer, color in zip(participants, colors):
            assignments[layer][pos] = int(color)

    extra_cap = min(2, len(interior) - cursor)
    for layer in range(count):
        extra_count = random.randint(0, extra_cap)
        for _ in range(extra_count):
            if cursor >= len(interior):
                break
            pos = interior[cursor]
            cursor += 1
            assignments[layer][pos] = random.choice(MARKER_COLORS)

    return assignments


def _make_patch(side: int, markers: dict[tuple[int, int], int], priority: int) -> np.ndarray:
    patch = np.full((side, side), BASE, dtype=int)
    for (r, c), color in markers.items():
        patch[int(r), int(c)] = int(color)
    if priority:
        cols = random.sample(range(side), priority)
        patch[-1, cols] = COUNTER
    return patch


def _placement_positions(grid_h: int, grid_w: int, side: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(1, grid_h - side)
        for c in range(1, grid_w - side)
    ]


def _can_place(used: list[tuple[int, int]], pos: tuple[int, int], side: int) -> bool:
    r, c = pos
    for rr, cc in used:
        if not (r + side < rr - 1 or rr + side < r - 1 or c + side < cc - 1 or cc + side < c - 1):
            return False
    return True


def _place_patches(patches: list[np.ndarray], grid_h: int, grid_w: int) -> np.ndarray | None:
    side = patches[0].shape[0]
    candidates = _placement_positions(grid_h, grid_w, side)
    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    chosen: list[tuple[int, int]] = []
    for patch in patches:
        pos = next((cell for cell in candidates if _can_place(placed, cell, side)), None)
        if pos is None:
            return None
        chosen.append(pos)
        placed.append(pos)
        candidates = [cell for cell in candidates if cell != pos]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for patch, (r, c) in zip(patches, chosen):
        grid[r : r + side, c : c + side] = patch
    return grid


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(RECT_COUNT_OPTIONS)
    side = random.choice(RECT_SIDE_OPTIONS)
    priorities = _sample_priorities(count)
    assignments = _build_assignments(side, priorities)
    if assignments is None:
        return None

    patches = [
        _make_patch(side, markers, priority)
        for markers, priority in zip(assignments, priorities)
    ]

    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    grid = _place_patches(patches, grid_h, grid_w)
    if grid is None:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
