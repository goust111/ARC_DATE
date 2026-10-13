from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SPRITE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()


SPRITES = {
    "up": {
        "cells": frozenset({(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)}),
        "tip": (0, 0),
    },
    "right": {
        "cells": frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)}),
        "tip": (0, 2),
    },
    "down": {
        "cells": frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)}),
        "tip": (2, 1),
    },
    "left": {
        "cells": frozenset({(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}),
        "tip": (1, 0),
    },
}
SHAPE_TO_ORIENTATION = {
    spec["cells"]: (name, spec["tip"]) for name, spec in SPRITES.items()
}
TRACE_DIRS = {
    "up": (-1, 1),
    "down": (1, -1),
    "left": (-1, -1),
    "right": (1, 1),
}

GRID_SIDE_RANGE = (18, 27)
SPRITE_COUNT_RANGE = (4, 7)
COLOR_STRIP_LENGTHS = (3, 4)
MAX_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds((H, W), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            components.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return components


def _extract_color_strip(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row = grid.shape[0] - 1
    mask = (grid[row] != BG) & (grid[row] != SPRITE)
    cols = np.flatnonzero(mask)
    assert 3 <= len(cols) <= 4, "expected a 3-4 pixel color strip on the bottom row"
    assert np.all(np.diff(cols) == 1), "color strip must be contiguous"
    colors = [int(grid[row, c]) for c in cols]
    assert len(set(colors)) == len(colors), "color strip colors must be distinct"
    return colors, [int(c) for c in cols]


def _classify_sprite(component: dict) -> dict:
    coords = component["coords"]
    r0, c0, _r1, _c1 = component["bbox"]
    rel = frozenset((r - r0, c - c0) for r, c in coords)
    assert rel in SHAPE_TO_ORIENTATION, f"unexpected grey sprite shape: {sorted(rel)}"
    orientation, tip_rel = SHAPE_TO_ORIENTATION[rel]
    return {
        "orientation": orientation,
        "tip": (r0 + tip_rel[0], c0 + tip_rel[1]),
        "bbox": component["bbox"],
        "coords": coords,
        "area": component["area"],
    }


def _trace_path(
    occupied: np.ndarray,
    tip: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], str]:
    H, W = occupied.shape
    dr, dc = direction
    r, c = tip
    path: list[tuple[int, int]] = []
    step = 1
    while True:
        nr = r + dr * step
        nc = c + dc * step
        if not (0 <= nr < H and 0 <= nc < W):
            return path, "boundary"
        if int(occupied[nr, nc]) != BG:
            return path, "obstacle"
        path.append((nr, nc))
        step += 1


def _analyze_input(grid: np.ndarray) -> tuple[list[int], list[dict]]:
    colors, _strip_cols = _extract_color_strip(grid)
    components = _find_components_of_color(grid, SPRITE)
    assert 4 <= len(components) <= 6, "expected 4-6 grey sprites"
    sprites = [_classify_sprite(comp) for comp in components]
    for sprite in sprites:
        path, stop_reason = _trace_path(grid, sprite["tip"], TRACE_DIRS[sprite["orientation"]])
        sprite["path"] = path
        sprite["stop_reason"] = stop_reason
    return colors, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """Classify each grey sprite by orientation, then draw a cyclically colored diagonal trace from its tip.

    Input structure
    - A black background contains 4-6 small grey 4-connected sprites, each matching one of four oriented tip-bearing templates.
    - The bottom row also contains one contiguous 3-4 pixel color strip of distinct non-grey colors that defines an ordered cycle.
    - Grey sprites are separated so their tip-driven traces can be read unambiguously.

    Rule
    - Read the color strip from left to right to obtain the cyclic color sequence.
    - Classify each grey sprite by its tip orientation: up -> trace up-right, down -> trace down-left, left -> trace up-left, right -> trace down-right.
    - Starting one step away from the tip, paint the diagonal trace until the next step would leave the grid or hit any non-background cell; step k uses the ((k-1) mod C)-th strip color.
    - Keep the original grey sprites and the bottom color strip unchanged.

    Key insight
    - The grey sprite is not copied directly: its orientation encodes a diagonal motion vector, and the bottom strip independently encodes the repeating colors used along that motion path.
    """
    g = np.asarray(grid, dtype=int)
    colors, sprites = _analyze_input(g)
    out = g.copy()
    for sprite in sorted(sprites, key=lambda item: item["tip"]):
        path, _stop_reason = _trace_path(out, sprite["tip"], TRACE_DIRS[sprite["orientation"]])
        for i, (r, c) in enumerate(path):
            out[r, c] = colors[i % len(colors)]
    return out


def _sprite_cells(top_left: tuple[int, int], orientation: str) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr, dc in SPRITES[orientation]["cells"]]


def _tip_from_top_left(top_left: tuple[int, int], orientation: str) -> tuple[int, int]:
    tr, tc = SPRITES[orientation]["tip"]
    return top_left[0] + tr, top_left[1] + tc


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _placeable(
    occupied_mask: np.ndarray,
    reserved_mask: np.ndarray,
    cells: list[tuple[int, int]],
) -> bool:
    forbidden = _dilate8(occupied_mask | reserved_mask)
    for r, c in cells:
        if forbidden[r, c]:
            return False
    return True


def _candidate_top_lefts(H: int, W: int, orientation: str) -> list[tuple[int, int]]:
    rel_rows = [r for r, _ in SPRITES[orientation]["cells"]]
    rel_cols = [c for _, c in SPRITES[orientation]["cells"]]
    h = max(rel_rows) + 1
    w = max(rel_cols) + 1
    return [(r0, c0) for r0 in range(H - h) for c0 in range(W - w + 1)]


def _targeted_strip_candidates(
    H: int,
    strip_cols: list[int],
    orientation: str,
) -> list[tuple[int, int]]:
    tip_rel_r, tip_rel_c = SPRITES[orientation]["tip"]
    candidates: list[tuple[int, int]] = []
    steps = list(range(3, H - 2))
    RNG.shuffle(steps)
    target_cols = strip_cols[:]
    RNG.shuffle(target_cols)
    for target_c in target_cols:
        for step in steps:
            if orientation == "down":
                tip = (H - 1 - step, target_c + step)
            else:
                tip = (H - 1 - step, target_c - step)
            top_left = (tip[0] - tip_rel_r, tip[1] - tip_rel_c)
            candidates.append(top_left)
    return candidates


def _sample_orientations(count: int) -> list[tuple[str, str | None]]:
    obstacle_orientation = str(RNG.choice(["down", "right"]))
    boundary_orientation = str(RNG.choice(["up", "left"]))
    roles: list[tuple[str, str | None]] = [
        (obstacle_orientation, "obstacle"),
        (boundary_orientation, "boundary"),
    ]

    remaining = count - 2
    if remaining > 0 and count >= 4 and RNG.random() < 0.7:
        forced = [name for name in ("up", "right", "down", "left") if name not in {obstacle_orientation, boundary_orientation}]
        RNG.shuffle(forced)
        take = min(len(forced), remaining)
        roles.extend((name, None) for name in forced[:take])
        remaining -= take

    for _ in range(remaining):
        roles.append((str(RNG.choice(["up", "right", "down", "left"])), None))

    tail = roles[2:]
    RNG.shuffle(tail)
    roles = roles[:2] + tail
    return roles


def _try_place_sprite(
    grid: np.ndarray,
    occupied_mask: np.ndarray,
    reserved_mask: np.ndarray,
    orientation: str,
    role: str | None,
    strip_cols: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], str] | None:
    H, W = grid.shape
    if role == "obstacle":
        candidates = _targeted_strip_candidates(H, strip_cols, orientation)
    else:
        candidates = _candidate_top_lefts(H, W, orientation)
        RNG.shuffle(candidates)

    for top_left in candidates:
        cells = _sprite_cells(top_left, orientation)
        if not all(_in_bounds((H, W), r, c) for r, c in cells):
            continue
        if any(r == H - 1 for r, _ in cells):
            continue
        if not _placeable(occupied_mask, reserved_mask, cells):
            continue

        tip = _tip_from_top_left(top_left, orientation)
        test = grid.copy()
        for r, c in cells:
            test[r, c] = SPRITE
        path, stop_reason = _trace_path(test, tip, TRACE_DIRS[orientation])
        if len(path) < 2:
            continue
        if role is not None and stop_reason != role:
            continue
        if any(reserved_mask[r, c] for r, c in path):
            continue
        return cells, path, stop_reason
    return None


def _render_scene(
    H: int,
    W: int,
    strip_colors: list[int],
    strip_start: int,
    placements: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for offset, color in enumerate(strip_colors):
        grid[H - 1, strip_start + offset] = int(color)
    for cells in placements:
        for r, c in cells:
            grid[r, c] = SPRITE
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        colors, sprites = _analyze_input(grid)
    except Exception:
        return False

    orientations = [sprite["orientation"] for sprite in sprites]
    if len(set(orientations)) < 2:
        return False
    if not any(o in {"up", "left"} for o in orientations):
        return False
    if not any(o in {"down", "right"} for o in orientations):
        return False

    all_trace_cells = [cell for sprite in sprites for cell in sprite["path"]]
    if len(set(all_trace_cells)) != len(all_trace_cells):
        return False
    if not all(len(sprite["path"]) >= 2 for sprite in sprites):
        return False
    if sum(len(sprite["path"]) for sprite in sprites) < 14:
        return False
    if max(len(sprite["path"]) for sprite in sprites) < 4:
        return False
    if {sprite["stop_reason"] for sprite in sprites} != {"boundary", "obstacle"}:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    painted = sum(int(out[r, c] != grid[r, c]) for r in range(grid.shape[0]) for c in range(grid.shape[1]))
    if painted != len(all_trace_cells):
        return False

    if len(colors) not in COLOR_STRIP_LENGTHS:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = int(RNG.integers(*GRID_SIDE_RANGE))
    W = int(RNG.integers(*GRID_SIDE_RANGE))
    strip_len = int(RNG.choice(COLOR_STRIP_LENGTHS))
    strip_start = int(RNG.integers(1, W - strip_len))
    strip_colors = list(map(int, RNG.choice(PALETTE, size=strip_len, replace=False)))

    grid = np.full((H, W), BG, dtype=int)
    for offset, color in enumerate(strip_colors):
        grid[H - 1, strip_start + offset] = color

    sprite_count = int(RNG.integers(*SPRITE_COUNT_RANGE))
    roles = _sample_orientations(sprite_count)

    occupied_mask = grid != BG
    reserved_mask = np.zeros((H, W), dtype=bool)
    placements: list[list[tuple[int, int]]] = []
    stop_reasons: list[str] = []

    for orientation, role in roles:
        placed = _try_place_sprite(grid, occupied_mask, reserved_mask, orientation, role, list(range(strip_start, strip_start + strip_len)))
        if placed is None:
            return None
        cells, path, stop_reason = placed
        placements.append(cells)
        stop_reasons.append(stop_reason)
        for r, c in cells:
            grid[r, c] = SPRITE
            occupied_mask[r, c] = True
        for r, c in path:
            reserved_mask[r, c] = True

    if "obstacle" not in stop_reasons or "boundary" not in stop_reasons:
        return None

    final_grid = _render_scene(H, W, strip_colors, strip_start, placements)
    if not _semantic_checks(final_grid):
        return None
    return final_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
