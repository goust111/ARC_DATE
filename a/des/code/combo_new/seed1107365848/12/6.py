from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

DIRS = (
    (-1, 0, "up"),
    (0, -1, "left"),
    (1, 0, "down"),
    (0, 1, "right"),
)
DIR_PRIORITY = {name: i for i, (_dr, _dc, name) in enumerate(DIRS)}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            rows = pts[:, 0]
            cols = pts[:, 1]
            comps.append(
                {
                    "coords": pts,
                    "touches_edge": touches_edge,
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                }
            )

    return comps


def _room_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []

    for comp in _components(g != GREY):
        if comp["touches_edge"]:
            continue

        r0, c0, r1, c1 = comp["bbox"]
        room_mask = np.zeros(g.shape, dtype=bool)
        room_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

        if not room_mask[r0 : r1 + 1, c0 : c1 + 1].all():
            continue
        if not np.any(g[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
            continue

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": room_mask,
            }
        )

    return rooms


def _visible_candidates(
    grid: np.ndarray,
    room_mask: np.ndarray,
    r: int,
    c: int,
) -> list[tuple[int, int, int, str]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, str]] = []

    for dr, dc, name in DIRS:
        nr = r + dr
        nc = c + dc
        dist = 1
        while 0 <= nr < h and 0 <= nc < w and room_mask[nr, nc]:
            value = int(grid[nr, nc])
            if value != BLACK:
                candidates.append((dist, DIR_PRIORITY[name], value, name))
                break
            nr += dr
            nc += dc
            dist += 1

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed room by assigning every floor cell to the first visible dye source on a cardinal ray.

    Input structure:
    - Black cells are floor, grey cells are one-cell-thick walls, and non-black non-grey cells are singleton dye sources.
    - The walls form one or more enclosed rectangular rooms, separated from the outer black background.
    - Each room contains 2-3 differently colored sources, placed so every floor cell has at least one visible source along its row or column.

    Rule:
    - Process every enclosed room independently while leaving the outer background unchanged.
    - For each black floor cell, look upward, leftward, downward, and rightward until the ray leaves the room; in each direction keep the first source seen before any wall.
    - Choose the visible source with the smallest orthogonal distance. If several are tied, prefer one seen from above, then from the left, then continue with the remaining directions.
    - Recolor the floor cell with that source color. Keep source pixels and grey walls unchanged.

    Key insight:
    - The partition is controlled by cardinal line-of-sight, not by plain Euclidean closeness: a source only matters if it is the first visible colored pixel on the same row or column inside the wall-bounded room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for room in _room_components(g):
        room_mask = room["mask"]
        floor = np.argwhere(room_mask & (g == BLACK))
        for r_raw, c_raw in floor:
            r = int(r_raw)
            c = int(c_raw)
            candidates = _visible_candidates(g, room_mask, r, c)
            if not candidates:
                continue
            _dist, _prio, color, _name = min(candidates, key=lambda item: (item[0], item[1]))
            out[r, c] = color

    return out


ROOM_TYPES = ("h2", "h3", "v2", "v3")
LONG_SIDE_RANGE = (5, 7)
OUTER_MARGIN_RANGE = (1, 3)
GAP_RANGE = (2, 4)
GENERATE_RETRIES = 300


def _make_room(kind: str) -> dict:
    if kind == "h2":
        floor_h = 2
        floor_w = random.randint(*LONG_SIDE_RANGE)
        base = random.randint(1, floor_w - 3)
        sources = [(0, base), (1, base + 1)]
    elif kind == "h3":
        floor_h = 3
        floor_w = random.randint(*LONG_SIDE_RANGE)
        base = random.randint(1, floor_w - 3)
        patterns = (
            [base, base + 1, base + 2],
            [base + 1, base, base + 2],
            [base, base + 2, base + 1],
        )
        cols = random.choice(patterns)
        sources = [(row, col) for row, col in enumerate(cols)]
    elif kind == "v2":
        floor_w = 2
        floor_h = random.randint(*LONG_SIDE_RANGE)
        base = random.randint(1, floor_h - 3)
        sources = [(base, 0), (base + 1, 1)]
    elif kind == "v3":
        floor_w = 3
        floor_h = random.randint(*LONG_SIDE_RANGE)
        base = random.randint(1, floor_h - 3)
        patterns = (
            [base, base + 1, base + 2],
            [base + 1, base, base + 2],
            [base, base + 2, base + 1],
        )
        rows = random.choice(patterns)
        sources = [(row, col) for col, row in enumerate(rows)]
    else:
        raise ValueError(f"unknown room type: {kind}")

    room = np.full((floor_h + 2, floor_w + 2), GREY, dtype=int)
    room[1:-1, 1:-1] = BLACK

    colors = random.sample(SOURCE_COLORS, len(sources))
    for color, (sr, sc) in zip(colors, sources):
        room[1 + sr, 1 + sc] = color

    return {
        "kind": kind,
        "array": room,
        "floor_shape": (floor_h, floor_w),
    }


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _branch_stats(grid: np.ndarray) -> tuple[dict, set[int], set[str]]:
    stats = {
        "no_candidate": 0,
        "single_dir": 0,
        "multi_dir": 0,
        "strict_multi": 0,
        "up_tie": 0,
        "left_tie": 0,
    }
    source_counts: set[int] = set()
    orientations: set[str] = set()

    for room in _room_components(grid):
        r0, c0, r1, c1 = room["bbox"]
        inner_h = int(r1 - r0 + 1)
        inner_w = int(c1 - c0 + 1)
        if inner_h in (2, 3):
            orientations.add("horizontal")
        if inner_w in (2, 3):
            orientations.add("vertical")

        source_count = int(np.sum(room["mask"] & (grid != BLACK)))
        source_counts.add(source_count)

        floor = np.argwhere(room["mask"] & (grid == BLACK))
        for r_raw, c_raw in floor:
            r = int(r_raw)
            c = int(c_raw)
            candidates = _visible_candidates(grid, room["mask"], r, c)
            if not candidates:
                stats["no_candidate"] += 1
                continue

            if len(candidates) == 1:
                stats["single_dir"] += 1
            else:
                stats["multi_dir"] += 1

            best_dist = min(item[0] for item in candidates)
            tied = [item for item in candidates if item[0] == best_dist]
            if len(tied) == 1 and len(candidates) > 1:
                stats["strict_multi"] += 1
            elif len(tied) > 1:
                choice = min(tied, key=lambda item: item[1])
                if choice[3] == "up":
                    stats["up_tie"] += 1
                elif choice[3] == "left":
                    stats["left_tie"] += 1

    return stats, source_counts, orientations


def _valid_generated_grid(grid: np.ndarray) -> bool:
    rooms = _room_components(grid)
    if len(rooms) != 4:
        return False

    stats, source_counts, orientations = _branch_stats(grid)
    if stats["no_candidate"] != 0:
        return False
    if stats["single_dir"] == 0 or stats["multi_dir"] == 0 or stats["strict_multi"] == 0:
        return False
    if stats["up_tie"] == 0 or stats["left_tie"] == 0:
        return False
    if source_counts != {2, 3}:
        return False
    if orientations != {"horizontal", "vertical"}:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out == GREY, grid == GREY):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    rooms = [_make_room(kind) for kind in random.sample(ROOM_TYPES, len(ROOM_TYPES))]
    top_left, top_right, bottom_left, bottom_right = rooms

    col0_w = max(top_left["array"].shape[1], bottom_left["array"].shape[1])
    col1_w = max(top_right["array"].shape[1], bottom_right["array"].shape[1])
    row0_h = max(top_left["array"].shape[0], top_right["array"].shape[0])
    row1_h = max(bottom_left["array"].shape[0], bottom_right["array"].shape[0])

    margin_top = random.randint(*OUTER_MARGIN_RANGE)
    margin_bottom = random.randint(*OUTER_MARGIN_RANGE)
    margin_left = random.randint(*OUTER_MARGIN_RANGE)
    margin_right = random.randint(*OUTER_MARGIN_RANGE)
    gap_h = random.randint(*GAP_RANGE)
    gap_w = random.randint(*GAP_RANGE)

    height = margin_top + row0_h + gap_h + row1_h + margin_bottom
    width = margin_left + col0_w + gap_w + col1_w + margin_right
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    slots = [
        (margin_top, margin_left, row0_h, col0_w, top_left["array"]),
        (margin_top, margin_left + col0_w + gap_w, row0_h, col1_w, top_right["array"]),
        (margin_top + row0_h + gap_h, margin_left, row1_h, col0_w, bottom_left["array"]),
        (
            margin_top + row0_h + gap_h,
            margin_left + col0_w + gap_w,
            row1_h,
            col1_w,
            bottom_right["array"],
        ),
    ]

    for slot_r, slot_c, slot_h, slot_w, room in slots:
        room_h, room_w = room.shape
        offset_r = random.randint(0, slot_h - room_h)
        offset_c = random.randint(0, slot_w - room_w)
        _paste(grid, room, slot_r + offset_r, slot_c + offset_c)

    if _valid_generated_grid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dye-room puzzle")
