from __future__ import annotations

import random

import numpy as np


BG = 0
PATH = 8
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _canonical_dims(height: int, width: int) -> tuple[int, int]:
    return (int(height), int(width)) if height <= width else (int(width), int(height))


def _room_infos(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    rooms: list[dict[str, object]] = []
    for cells in _components(grid == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            continue
        cell_set = set(cells)
        border_colors: set[int] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if (nr, nc) in cell_set:
                    continue
                border_colors.add(int(grid[nr, nc]))
        if border_colors != {PATH}:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if len(cells) != height * width:
            continue
        wall_cells = _room_wall_cells(r0, c0, r1, c1)
        if any(int(grid[r, c]) != PATH for r, c in wall_cells):
            continue
        rooms.append(
            {
                "top": int(r0),
                "left": int(c0),
                "height": int(height),
                "width": int(width),
                "walls": wall_cells,
            }
        )
    rooms.sort(key=lambda room: (int(room["top"]), int(room["left"])))
    return rooms


def _pattern_mapping(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    pattern_mask = (grid != BG) & (grid != PATH)
    mapping: dict[tuple[int, int], int] = {}
    for cells in _components(pattern_mask):
        colors = {int(grid[r, c]) for r, c in cells}
        assert len(colors) == 1, "each external pattern must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(cells)
        key = _canonical_dims(r1 - r0 + 1, c1 - c0 + 1)
        assert key not in mapping, "room-to-pattern matching must be unambiguous"
        mapping[key] = int(color)
    return mapping, pattern_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match rectangular room interiors to external monochrome patterns by bounding-box size,
    then recolor the room walls with the matched pattern color.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _room_infos(g)
    assert rooms, "expected at least one enclosed rectangular room"

    pattern_colors, pattern_mask = _pattern_mapping(g)
    assert pattern_colors, "expected at least one external pattern"

    out = g.copy()
    out[pattern_mask] = BG

    painted: set[tuple[int, int]] = set()
    for room in rooms:
        key = _canonical_dims(int(room["height"]), int(room["width"]))
        assert key in pattern_colors, "every room must have a matching external pattern"
        color = int(pattern_colors[key])
        for r, c in room["walls"]:
            if (r, c) not in painted:
                out[r, c] = color
                painted.add((r, c))
    return out


def _room_wall_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    walls: set[tuple[int, int]] = set()
    for c in range(c0 - 1, c1 + 2):
        walls.add((r0 - 1, c))
        walls.add((r1 + 1, c))
    for r in range(r0, r1 + 1):
        walls.add((r, c0 - 1))
        walls.add((r, c1 + 1))
    return walls


def _draw_network(
    grid: np.ndarray, frame_r0: int, frame_c0: int, room_width: int, heights: list[int]
) -> list[dict[str, object]]:
    frame_height = sum(heights) + len(heights) + 1
    frame_width = room_width + 2
    frame_r1 = frame_r0 + frame_height - 1
    frame_c1 = frame_c0 + frame_width - 1

    grid[frame_r0 : frame_r1 + 1, frame_c0] = PATH
    grid[frame_r0 : frame_r1 + 1, frame_c1] = PATH

    divider = frame_r0
    grid[divider, frame_c0 : frame_c1 + 1] = PATH
    rooms: list[dict[str, object]] = []
    current_r = frame_r0 + 1
    for height in heights:
        r0 = current_r
        r1 = r0 + height - 1
        c0 = frame_c0 + 1
        c1 = c0 + room_width - 1
        rooms.append(
            {
                "top": int(r0),
                "left": int(c0),
                "height": int(height),
                "width": int(room_width),
                "walls": _room_wall_cells(r0, c0, r1, c1),
            }
        )
        divider = r1 + 1
        grid[divider, frame_c0 : frame_c1 + 1] = PATH
        current_r = divider + 1
    return rooms


def _mark_blocked_rect(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(blocked.shape[0], int(r0) + int(h) + pad)
    cc1 = min(blocked.shape[1], int(c0) + int(w) + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _mark_blocked_cells(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    for r, c in cells:
        rr0 = max(0, int(r) - pad)
        cc0 = max(0, int(c) - pad)
        rr1 = min(blocked.shape[0], int(r) + pad + 1)
        cc1 = min(blocked.shape[1], int(c) + pad + 1)
        blocked[rr0:rr1, cc0:cc1] = True


def _random_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        steps = []
        if r != er:
            steps.append((1 if er > r else -1, 0))
        if c != ec:
            steps.append((0, 1 if ec > c else -1))
        dr, dc = random.choice(steps)
        r += dr
        c += dc
        mask[r, c] = True


def _pattern_shape(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    anchors = [
        (0, random.randrange(width)),
        (random.randrange(height), width - 1),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
    ]
    random.shuffle(anchors)
    current = anchors[0]
    mask[current] = True
    for nxt in anchors[1:]:
        _random_path(mask, current, nxt)
        current = nxt

    extra = random.randint(max(1, (height + width) // 2), max(1, height * width // 2))
    for _ in range(extra):
        filled = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        if not filled:
            break
        r, c = random.choice(filled)
        neighbors = []
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if neighbors:
            nr, nc = random.choice(neighbors)
            mask[nr, nc] = True
    return mask


def _rotated_pattern_flags(heights: list[int], room_width: int) -> list[bool]:
    flags = []
    visible = [i for i, height in enumerate(heights) if height != room_width]
    for height in heights:
        flags.append(height != room_width and random.random() < 0.5)
    if visible and not any(flags[i] for i in visible):
        flags[random.choice(visible)] = True
    if all(flags):
        flags[random.randrange(len(flags))] = False
    return flags


def _stub_cells(
    side: str, anchor: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    r, c = anchor
    if side == "top":
        return [(r - k, c) for k in range(1, length + 1)]
    if side == "bottom":
        return [(r + k, c) for k in range(1, length + 1)]
    if side == "left":
        return [(r, c - k) for k in range(1, length + 1)]
    return [(r, c + k) for k in range(1, length + 1)]


def _add_stubs(
    grid: np.ndarray,
    frame_r0: int,
    frame_c0: int,
    frame_height: int,
    frame_width: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> list[tuple[int, int]]:
    frame_r1 = frame_r0 + frame_height - 1
    frame_c1 = frame_c0 + frame_width - 1
    candidates: list[tuple[str, tuple[int, int], int]] = []

    if top_margin > 1:
        for c in range(frame_c0 + 1, frame_c1):
            candidates.append(("top", (frame_r0, c), top_margin - 1))
    if bottom_margin > 1:
        for c in range(frame_c0 + 1, frame_c1):
            candidates.append(("bottom", (frame_r1, c), bottom_margin - 1))
    if left_margin > 1:
        for r in range(frame_r0 + 1, frame_r1):
            candidates.append(("left", (r, frame_c0), left_margin - 1))
    if right_margin > 1:
        for r in range(frame_r0 + 1, frame_r1):
            candidates.append(("right", (r, frame_c1), right_margin - 1))

    if not candidates:
        return []

    random.shuffle(candidates)
    wanted = random.choice([0, 1, 2])
    used_sides: set[str] = set()
    new_cells: list[tuple[int, int]] = []
    for side, anchor, max_length in candidates:
        if len(used_sides) >= wanted:
            break
        if side in used_sides:
            continue
        length = random.randint(1, max_length)
        cells = _stub_cells(side, anchor, length)
        for r, c in cells:
            grid[r, c] = PATH
        used_sides.add(side)
        new_cells.extend(cells)
    return new_cells


def _can_place_box(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    if r0 < 1 or c0 < 1 or r0 + h >= blocked.shape[0] or c0 + w >= blocked.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    return not bool(np.any(blocked[rr0:rr1, cc0:cc1]))


def _expected_output(
    grid: np.ndarray, rooms: list[dict[str, object]], patterns: list[dict[str, object]]
) -> np.ndarray:
    out = grid.copy()
    for pattern in patterns:
        for r, c in pattern["cells"]:
            out[int(r), int(c)] = BG
    painted: set[tuple[int, int]] = set()
    for room in sorted(rooms, key=lambda item: (int(item["top"]), int(item["left"]))):
        color = int(room["match_color"])
        for r, c in room["walls"]:
            if (r, c) not in painted:
                out[r, c] = color
                painted.add((r, c))
    return out


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(2, 4)
    room_width = random.randint(3, 6)
    heights = random.sample(list(range(2, 7)), room_count)
    frame_height = sum(heights) + room_count + 1
    frame_width = room_width + 2
    max_dim = max([room_width] + heights)

    top_margin = random.randint(2, max_dim + 2)
    bottom_margin = random.randint(2, max_dim + 2)
    left_margin = random.randint(2, max_dim + 3)
    right_margin = random.randint(2, max_dim + 3)

    height = frame_height + top_margin + bottom_margin
    width = frame_width + left_margin + right_margin
    grid = np.full((height, width), BG, dtype=int)

    frame_r0 = top_margin
    frame_c0 = left_margin
    rooms = _draw_network(grid, frame_r0, frame_c0, room_width, heights)

    stub_cells = _add_stubs(
        grid,
        frame_r0,
        frame_c0,
        frame_height,
        frame_width,
        top_margin,
        bottom_margin,
        left_margin,
        right_margin,
    )

    blocked = np.zeros((height, width), dtype=bool)
    _mark_blocked_rect(blocked, frame_r0, frame_c0, frame_height, frame_width, pad=1)
    _mark_blocked_cells(blocked, stub_cells, pad=1)

    colors = random.sample(COLOR_POOL, room_count)
    rotated = _rotated_pattern_flags(heights, room_width)
    patterns: list[dict[str, object]] = []

    for idx, room in enumerate(rooms):
        room["match_color"] = int(colors[idx])
        ph = int(room["width"]) if rotated[idx] else int(room["height"])
        pw = int(room["height"]) if rotated[idx] else int(room["width"])
        mask = _pattern_shape(ph, pw)

        placements = []
        for r0 in range(1, height - ph):
            for c0 in range(1, width - pw):
                if _can_place_box(blocked, r0, c0, ph, pw, pad=1):
                    placements.append((r0, c0))
        if not placements:
            return None

        r0, c0 = random.choice(placements)
        cells: list[tuple[int, int]] = []
        for dr, dc in np.argwhere(mask):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            grid[rr, cc] = int(colors[idx])
            cells.append((rr, cc))

        _mark_blocked_rect(blocked, r0, c0, ph, pw, pad=1)
        patterns.append({"color": int(colors[idx]), "cells": cells})

    expected = _expected_output(grid, rooms, patterns)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
