from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, r0, c0


def _arrow_sprite(direction: str, shaft_len: int) -> np.ndarray:
    shaft_len = int(shaft_len)
    assert shaft_len >= 2
    if direction == "right":
        sprite = np.zeros((3, shaft_len + 2), dtype=bool)
        sprite[1, :] = True
        sprite[:, -2] = True
        return sprite
    if direction == "left":
        sprite = np.zeros((3, shaft_len + 2), dtype=bool)
        sprite[1, :] = True
        sprite[:, 1] = True
        return sprite
    if direction == "down":
        sprite = np.zeros((shaft_len + 2, 3), dtype=bool)
        sprite[:, 1] = True
        sprite[-2, :] = True
        return sprite
    if direction == "up":
        sprite = np.zeros((shaft_len + 2, 3), dtype=bool)
        sprite[:, 1] = True
        sprite[1, :] = True
        return sprite
    raise AssertionError(f"unknown direction: {direction}")


def _detect_arrow(coords: list[tuple[int, int]]) -> tuple[str, np.ndarray, int, int]:
    mask, r0, c0 = _crop_mask(coords)
    h, w = mask.shape

    if h == 3 and w >= 4 and mask[1, :].all():
        if mask[0].sum() == 1 and mask[2].sum() == 1 and np.array_equal(mask[0], mask[2]):
            head_col = int(np.flatnonzero(mask[0])[0])
            if head_col == 1:
                return "left", mask, r0, c0
            if head_col == w - 2:
                return "right", mask, r0, c0

    if w == 3 and h >= 4 and mask[:, 1].all():
        if mask[:, 0].sum() == 1 and mask[:, 2].sum() == 1 and np.array_equal(mask[:, 0], mask[:, 2]):
            head_row = int(np.flatnonzero(mask[:, 0])[0])
            if head_row == 1:
                return "up", mask, r0, c0
            if head_row == h - 2:
                return "down", mask, r0, c0

    raise AssertionError("could not recognize arrow shape")


def _distance_to_object(cell: tuple[int, int], coords: list[tuple[int, int]]) -> int:
    r, c = cell
    return min(abs(r - rr) + abs(c - cc) for rr, cc in coords)


def _fuel_order(fuels: list[tuple[int, int]], coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(fuels, key=lambda p: (_distance_to_object(p, coords), p[0], p[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored arrow by its same-color fuel count, erasing exactly the fuel dots spent on the move.

    Input structure:
    - The background is black, grey cells are wall bars, and every moving object is a single monochrome arrow with a clear up/down/left/right orientation.
    - Each arrow color is unique and that color also appears as isolated single-pixel fuel dots elsewhere on the canvas.
    - Fuel dots are separated from the arrow body and from other objects, and the slide lanes do not interfere with each other.

    Rule:
    - For each non-grey foreground color, treat its largest 4-connected component as the arrow object and its same-color single-pixel components as fuel.
    - Sort that color's fuel dots by increasing Manhattan distance to the starting arrow pixels; one dot is consumed per successful step.
    - Move the arrow along its pointing direction until fuel runs out or the next translated position would leave the canvas or hit a grey wall; erase the consumed fuel dots, keep unused fuel dots and walls unchanged, and place the arrow at its final location.

    Key insight:
    - The visible invariant is conservation: actual displacement equals the number of removed same-color fuel dots, while any blocked extra fuel remains on the board.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    colors = sorted(int(color) for color in np.unique(g) if color not in (BG, WALL))
    arrow_infos = []

    for color in colors:
        comps = _components(g == color)
        assert comps, "expected at least one component for every foreground color"
        arrow_comp = max(comps, key=len)
        fuel_comps = [comp for comp in comps if comp is not arrow_comp]
        assert len(arrow_comp) >= 6, "arrow object must be larger than a fuel dot"
        assert all(len(comp) == 1 for comp in fuel_comps), "fuel dots must be isolated single pixels"

        direction, _mask, _r0, _c0 = _detect_arrow(arrow_comp)
        dr, dc = DIRECTIONS[direction]
        ordered_fuels = _fuel_order([comp[0] for comp in fuel_comps], arrow_comp)

        steps = 0
        while steps < len(ordered_fuels):
            moved = [(r + dr * (steps + 1), c + dc * (steps + 1)) for r, c in arrow_comp]
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in moved):
                break
            blocked = False
            for rr, cc in moved:
                cell = int(g[rr, cc])
                if cell == WALL or cell not in (BG, color):
                    blocked = True
                    break
            if blocked:
                break
            steps += 1

        arrow_infos.append((color, arrow_comp, ordered_fuels, dr, dc, steps))

    for color, arrow_comp, ordered_fuels, _dr, _dc, steps in arrow_infos:
        for r, c in arrow_comp:
            out[r, c] = BG
        for r, c in ordered_fuels[:steps]:
            out[r, c] = BG

    for color, arrow_comp, _ordered_fuels, dr, dc, steps in arrow_infos:
        for r, c in arrow_comp:
            out[r + dr * steps, c + dc * steps] = color

    return out


def _sample_spaced_starts(max_start: int, count: int, min_delta: int) -> list[int] | None:
    starts = list(range(1, max_start + 1))
    for _ in range(100):
        random.shuffle(starts)
        picked: list[int] = []
        for pos in starts:
            if all(abs(pos - prev) >= min_delta for prev in picked):
                picked.append(pos)
                if len(picked) == count:
                    return sorted(picked)
    return None


def _sprite_coords(sprite: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = []
    for r, c in np.argwhere(sprite):
        coords.append((top + int(r), left + int(c)))
    return coords


def _touches_foreground(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[rr, cc]) != BG:
                return True
    return False


def _place_fuels(
    grid: np.ndarray,
    arrow_coords: list[tuple[int, int]],
    color: int,
    count: int,
    *,
    banned_rows: set[int] | None = None,
    banned_cols: set[int] | None = None,
) -> list[tuple[int, int]] | None:
    groups: dict[int, list[tuple[int, int]]] = {}
    h, w = grid.shape
    for r in range(h):
        if banned_rows is not None and r in banned_rows:
            continue
        for c in range(w):
            if banned_cols is not None and c in banned_cols:
                continue
            if int(grid[r, c]) != BG or _touches_foreground(grid, r, c):
                continue
            dist = _distance_to_object((r, c), arrow_coords)
            groups.setdefault(dist, []).append((r, c))

    placed: list[tuple[int, int]] = []
    for dist in sorted(groups):
        cells = groups[dist][:]
        random.shuffle(cells)
        for r, c in cells:
            if int(grid[r, c]) != BG or _touches_foreground(grid, r, c):
                continue
            grid[r, c] = color
            placed.append((r, c))
            break
        if len(placed) == count:
            return placed
    return None


def _sample_modes(count: int) -> list[str]:
    modes = [random.choice(["full", "wall_stop", "border_stop"]) for _ in range(count)]
    if count >= 2:
        if not any(mode == "full" for mode in modes):
            modes[random.randrange(count)] = "full"
        if not any(mode != "full" for mode in modes):
            idx = random.randrange(count)
            other = (idx + 1) % count
            modes[other] = random.choice(["wall_stop", "border_stop"])
    return modes


def _plan_horizontal_arrow(width: int, row_top: int, color: int, mode: str) -> dict | None:
    direction = random.choice(["left", "right"])
    shaft_len = random.randint(2, 5)
    sprite = _arrow_sprite(direction, shaft_len)
    sprite_h, sprite_w = sprite.shape

    if direction == "right":
        if mode == "full":
            max_fuel = min(6, width - sprite_w - 1)
            if max_fuel < 1:
                return None
            fuel_count = random.randint(1, max_fuel)
            left = random.randint(1, width - sprite_w - fuel_count)
            free_steps = width - sprite_w - left
            wall_col = None
        elif mode == "border_stop":
            max_steps = min(4, width - sprite_w - 1)
            if max_steps < 1:
                return None
            free_steps = random.randint(1, max_steps)
            left = width - sprite_w - free_steps
            fuel_count = free_steps + random.randint(1, 3)
            wall_col = None
        else:
            max_steps = min(4, width - sprite_w - 2)
            if max_steps < 1:
                return None
            free_steps = random.randint(1, max_steps)
            max_left = width - sprite_w - free_steps - 1
            if max_left < 1:
                return None
            left = random.randint(1, max_left)
            fuel_count = free_steps + random.randint(1, 3)
            wall_col = left + sprite_w + free_steps
    else:
        if mode == "full":
            max_fuel = min(6, width - sprite_w - 1)
            if max_fuel < 1:
                return None
            fuel_count = random.randint(1, max_fuel)
            min_left = fuel_count
            max_left = width - sprite_w - 1
            if min_left > max_left:
                return None
            left = random.randint(min_left, max_left)
            free_steps = left
            wall_col = None
        elif mode == "border_stop":
            max_steps = min(4, width - sprite_w - 1)
            if max_steps < 1:
                return None
            free_steps = random.randint(1, max_steps)
            left = free_steps
            fuel_count = free_steps + random.randint(1, 3)
            wall_col = None
        else:
            max_steps = min(4, width - sprite_w - 2)
            if max_steps < 1:
                return None
            free_steps = random.randint(1, max_steps)
            min_left = free_steps + 1
            max_left = width - sprite_w - 1
            if min_left > max_left:
                return None
            left = random.randint(min_left, max_left)
            fuel_count = free_steps + random.randint(1, 3)
            wall_col = left - free_steps - 1

    coords = _sprite_coords(sprite, row_top, left)
    wall_cells = [] if wall_col is None else [(row_top + dr, wall_col) for dr in range(sprite_h)]
    return {
        "color": color,
        "direction": direction,
        "coords": coords,
        "fuel_count": fuel_count,
        "actual_steps": min(fuel_count, free_steps),
        "wall_cells": wall_cells,
    }


def _plan_vertical_arrow(height: int, col_left: int, color: int, mode: str) -> dict | None:
    direction = random.choice(["up", "down"])
    shaft_len = random.randint(2, 5)
    sprite = _arrow_sprite(direction, shaft_len)
    sprite_h, sprite_w = sprite.shape

    if direction == "down":
        if mode == "full":
            max_fuel = min(6, height - sprite_h - 1)
            if max_fuel < 1:
                return None
            fuel_count = random.randint(1, max_fuel)
            top = random.randint(1, height - sprite_h - fuel_count)
            free_steps = height - sprite_h - top
            wall_row = None
        elif mode == "border_stop":
            max_steps = min(4, height - sprite_h - 1)
            if max_steps < 1:
                return None
            free_steps = random.randint(1, max_steps)
            top = height - sprite_h - free_steps
            fuel_count = free_steps + random.randint(1, 3)
            wall_row = None
        else:
            max_steps = min(4, height - sprite_h - 2)
            if max_steps < 1:
                return None
            free_steps = random.randint(1, max_steps)
            max_top = height - sprite_h - free_steps - 1
            if max_top < 1:
                return None
            top = random.randint(1, max_top)
            fuel_count = free_steps + random.randint(1, 3)
            wall_row = top + sprite_h + free_steps
    else:
        if mode == "full":
            max_fuel = min(6, height - sprite_h - 1)
            if max_fuel < 1:
                return None
            fuel_count = random.randint(1, max_fuel)
            min_top = fuel_count
            max_top = height - sprite_h - 1
            if min_top > max_top:
                return None
            top = random.randint(min_top, max_top)
            free_steps = top
            wall_row = None
        elif mode == "border_stop":
            max_steps = min(4, height - sprite_h - 1)
            if max_steps < 1:
                return None
            free_steps = random.randint(1, max_steps)
            top = free_steps
            fuel_count = free_steps + random.randint(1, 3)
            wall_row = None
        else:
            max_steps = min(4, height - sprite_h - 2)
            if max_steps < 1:
                return None
            free_steps = random.randint(1, max_steps)
            min_top = free_steps + 1
            max_top = height - sprite_h - 1
            if min_top > max_top:
                return None
            top = random.randint(min_top, max_top)
            fuel_count = free_steps + random.randint(1, 3)
            wall_row = top - free_steps - 1

    coords = _sprite_coords(sprite, top, col_left)
    wall_cells = [] if wall_row is None else [(wall_row, col_left + dc) for dc in range(sprite_w)]
    return {
        "color": color,
        "direction": direction,
        "coords": coords,
        "fuel_count": fuel_count,
        "actual_steps": min(fuel_count, free_steps),
        "wall_cells": wall_cells,
    }


def _build_expected(inp: np.ndarray, infos: list[dict]) -> np.ndarray:
    out = inp.copy()
    for info in infos:
        for r, c in info["coords"]:
            out[r, c] = BG
        for r, c in info["fuel_cells"][: info["actual_steps"]]:
            out[r, c] = BG

    for info in infos:
        dr, dc = DIRECTIONS[info["direction"]]
        for r, c in info["coords"]:
            out[r + dr * info["actual_steps"], c + dc * info["actual_steps"]] = info["color"]
    return out


def _generate_horizontal_attempt() -> np.ndarray | None:
    n_arrows = random.randint(1, 4)
    height = random.randint(max(10, 4 * n_arrows + 4), 30)
    width = random.randint(16, 30)
    starts = _sample_spaced_starts(height - 4, n_arrows, 4)
    if starts is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    infos: list[dict] = []
    for row_top, color, mode in zip(starts, random.sample(PALETTE, n_arrows), _sample_modes(n_arrows)):
        info = _plan_horizontal_arrow(width, row_top, color, mode)
        if info is None:
            return None
        for r, c in info["coords"]:
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = color
        for r, c in info["wall_cells"]:
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = WALL
        infos.append(info)

    banned_rows = {r for row_top in starts for r in range(row_top, row_top + 3)}
    for info in infos:
        fuels = _place_fuels(grid, info["coords"], info["color"], info["fuel_count"], banned_rows=banned_rows)
        if fuels is None:
            return None
        info["fuel_cells"] = fuels

    expected = _build_expected(grid, infos)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _generate_vertical_attempt() -> np.ndarray | None:
    n_arrows = random.randint(1, 4)
    height = random.randint(16, 30)
    width = random.randint(max(10, 4 * n_arrows + 4), 30)
    starts = _sample_spaced_starts(width - 4, n_arrows, 4)
    if starts is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    infos: list[dict] = []
    for col_left, color, mode in zip(starts, random.sample(PALETTE, n_arrows), _sample_modes(n_arrows)):
        info = _plan_vertical_arrow(height, col_left, color, mode)
        if info is None:
            return None
        for r, c in info["coords"]:
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = color
        for r, c in info["wall_cells"]:
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = WALL
        infos.append(info)

    banned_cols = {c for col_left in starts for c in range(col_left, col_left + 3)}
    for info in infos:
        fuels = _place_fuels(grid, info["coords"], info["color"], info["fuel_count"], banned_cols=banned_cols)
        if fuels is None:
            return None
        info["fuel_cells"] = fuels

    expected = _build_expected(grid, infos)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        if random.choice([False, True]):
            grid = _generate_horizontal_attempt()
        else:
            grid = _generate_vertical_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRY_LIMIT} attempts")
