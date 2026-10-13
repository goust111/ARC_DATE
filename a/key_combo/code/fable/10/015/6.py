from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
RED = 2
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # E, S, W, N
SPRITE_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Sprite:
    color: int
    cells: tuple[tuple[int, int], ...]
    origin: tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_sprites(grid: np.ndarray) -> list[Sprite]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    sprites: list[Sprite] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in (BG, RED):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds((h, w), nr, nc) or visited[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            r0 = min(rr for rr, _ in coords)
            c0 = min(cc for _, cc in coords)
            rel = tuple(sorted((rr - r0, cc - c0) for rr, cc in coords))
            sprites.append(Sprite(color=color, cells=rel, origin=(r0, c0)))

    sprites.sort(key=lambda s: (s.origin[0], s.origin[1], s.color, len(s.cells)))
    return sprites


def _abs_cells(sprite: Sprite, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = origin
    return [(r0 + dr, c0 + dc) for dr, dc in sprite.cells]


def _render(
    shape: tuple[int, int],
    sprites: list[Sprite],
    origins: list[tuple[int, int]],
    red_points: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in red_points:
        out[r, c] = RED
    for sprite, origin in zip(sprites, origins):
        for r, c in _abs_cells(sprite, origin):
            out[r, c] = sprite.color
    return out


def _simulate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    sprites = _component_sprites(g)
    assert sprites, "expected at least one movable sprite"
    red_points = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    origins = [sprite.origin for sprite in sprites]
    headings = [0 for _ in sprites]
    terminated = [False for _ in sprites]

    trace = {
        "turns_total": 0,
        "moves_total": 0,
        "stops_total": 0,
        "wait_blocked_total": 0,
        "wait_conflict_total": 0,
        "turns_from": Counter(),
        "turns_per_sprite": [0 for _ in sprites],
        "rounds": 0,
        "terminal_waiters": [],
    }

    max_rounds = max(1, 16 * h * w)
    for _ in range(max_rounds):
        occ = np.full((h, w), -1, dtype=int)
        sprite_cells: list[list[tuple[int, int]]] = []
        for idx, sprite in enumerate(sprites):
            cells = _abs_cells(sprite, origins[idx])
            sprite_cells.append(cells)
            for r, c in cells:
                assert occ[r, c] == -1
                occ[r, c] = idx

        actions: list[str] = []
        targets: list[list[tuple[int, int]] | None] = []
        for idx, sprite in enumerate(sprites):
            if terminated[idx]:
                actions.append("done")
                targets.append(None)
                continue
            dr, dc = DIRECTIONS[headings[idx]]
            candidate = [(r + dr, c + dc) for r, c in sprite_cells[idx]]
            targets.append(candidate)
            if any(not _in_bounds((h, w), r, c) for r, c in candidate):
                actions.append("stop")
                continue
            if any((r, c) in red_points for r, c in candidate):
                actions.append("turn")
                continue
            if any(occ[r, c] not in (-1, idx) for r, c in candidate):
                actions.append("wait_blocked")
                continue
            actions.append("move")

        claims: dict[tuple[int, int], list[int]] = {}
        for idx, action in enumerate(actions):
            if action != "move":
                continue
            assert targets[idx] is not None
            for cell in targets[idx]:
                claims.setdefault(cell, []).append(idx)
        for claimers in claims.values():
            if len(claimers) < 2:
                continue
            for idx in claimers:
                if actions[idx] == "move":
                    actions[idx] = "wait_conflict"

        moved = False
        turned = False
        for idx, action in enumerate(actions):
            if action == "move":
                dr, dc = DIRECTIONS[headings[idx]]
                r0, c0 = origins[idx]
                origins[idx] = (r0 + dr, c0 + dc)
                trace["moves_total"] += 1
                moved = True
            elif action == "turn":
                trace["turns_total"] += 1
                trace["turns_from"][headings[idx]] += 1
                trace["turns_per_sprite"][idx] += 1
                headings[idx] = (headings[idx] + 1) % 4
                turned = True
            elif action == "stop":
                terminated[idx] = True
                trace["stops_total"] += 1
            elif action == "wait_blocked":
                trace["wait_blocked_total"] += 1
            elif action == "wait_conflict":
                trace["wait_conflict_total"] += 1

        trace["rounds"] += 1
        if not moved and not turned:
            trace["terminal_waiters"] = [
                idx for idx, action in enumerate(actions) if action.startswith("wait")
            ]
            break
    else:
        raise AssertionError("simulation did not settle")

    return {
        "grid": _render((h, w), sprites, origins, red_points),
        "trace": trace,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every rigid sprite until red turners rotate its heading clockwise and blockers make it wait.

    Input structure:
    - The grid has a black background, several disconnected monochrome foreground sprites, and some fixed red single pixels.
    - Every non-red foreground component is a rigid sprite that keeps its shape and color while moving.
    - Red pixels never move; the layout is chosen so the process eventually reaches a static terminal state.

    Rule:
    - Run synchronous rounds with every non-terminated sprite starting with heading east.
    - In each round a sprite first checks the one-step translated footprint in its current heading: if any target cell is red, it stays put and rotates 90 degrees clockwise.
    - Otherwise, if the translated footprint would leave the grid, that sprite permanently stops; if the translated footprint is occupied by another sprite, it waits for this round; otherwise it moves one cell.
    - Repeat until a full round has no movement and no turns, then keep the red pixels and return the terminal grid.

    Key insight:
    - Red pixels are relative turners, not absolute south arrows or solid walls, and already-stopped sprites can later become the blockers that freeze another sprite's path.
    """
    return _simulate(np.asarray(grid, dtype=int))["grid"]


def _rect_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(top, top + height) for c in range(left, left + width)]


def _can_place_sprite(
    shape: tuple[int, int],
    sprite_mask: np.ndarray,
    cells: list[tuple[int, int]],
) -> bool:
    h, w = shape
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds((h, w), r, c) or sprite_mask[r, c]:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set:
                continue
            if _in_bounds((h, w), nr, nc) and sprite_mask[nr, nc]:
                return False
    return True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_red(grid: np.ndarray, coord: tuple[int, int]) -> bool:
    r, c = coord
    if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
        return False
    grid[r, c] = RED
    return True


def _build_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(24, 28)
    grid = np.full((h, w), BG, dtype=int)
    sprite_mask = np.zeros((h, w), dtype=bool)
    colors = random.sample(SPRITE_COLORS, 3)

    main_h = random.choice((2, 3))
    w0 = random.choice((2, 3))
    w1 = random.choice((2, 3))

    c0 = random.randint(1, 3)
    r0 = random.randint(1, max(2, h // 4))
    east0 = random.randint(2, 4)
    sprite0 = _rect_cells(r0, c0, main_h, w0)
    if not _can_place_sprite((h, w), sprite_mask, sprite0):
        return None
    _paint_cells(grid, sprite0, colors[0])
    _paint_cells(sprite_mask, sprite0, True)
    red0 = (r0 + random.randrange(main_h), c0 + w0 + east0)
    if not _place_red(grid, red0):
        return None

    south1 = random.randint(2, 5)
    r1 = h - main_h - 1 - south1
    if r1 < 1 or abs(r1 - r0) <= main_h:
        return None
    c1 = random.randint(1, 4)
    blocker_right = c0 + east0 + w0 - 1
    zone2_lo = max(w // 2, blocker_right + 8)
    zone2_hi = w - 9
    if zone2_lo > zone2_hi:
        return None
    zone2_left = random.randint(zone2_lo, zone2_hi)
    x1_min = blocker_right + 4
    x1_max = zone2_left - 5
    if x1_min > x1_max:
        return None
    x1 = random.randint(x1_min, x1_max)
    east1 = x1 - c1
    if east1 < 3:
        return None
    sprite1 = _rect_cells(r1, c1, main_h, w1)
    if not _can_place_sprite((h, w), sprite_mask, sprite1):
        return None
    _paint_cells(grid, sprite1, colors[1])
    _paint_cells(sprite_mask, sprite1, True)
    red1 = (r1 + random.randrange(main_h), x1 + w1)
    red2 = (h - main_h + random.randrange(main_h), x1 + random.randrange(w1))
    if not _place_red(grid, red1) or not _place_red(grid, red2):
        return None

    h2 = random.choice((2, 3))
    w2 = random.choice((2, 3))
    east2 = random.randint(1, 3)
    south2 = random.randint(2, 4)
    west2 = random.randint(2, 4)
    r2_max = min(3, h - h2 - south2 - 2)
    if r2_max < 1:
        return None
    r2 = random.randint(1, r2_max)
    c2_min = zone2_left
    c2_max = w - (w2 + east2 + 2)
    if c2_min > c2_max:
        return None
    c2 = random.randint(c2_min, c2_max)
    north_col = c2 + east2 - west2
    if north_col < zone2_left - 1:
        return None
    sprite2 = _rect_cells(r2, c2, h2, w2)
    if not _can_place_sprite((h, w), sprite_mask, sprite2):
        return None
    _paint_cells(grid, sprite2, colors[2])
    _paint_cells(sprite_mask, sprite2, True)
    red3 = (r2 + random.randrange(h2), c2 + w2 + east2)
    red4 = (r2 + south2 + h2, c2 + east2 + random.randrange(w2))
    red5 = (r2 + south2 + random.randrange(h2), north_col - 1)
    if not (_place_red(grid, red3) and _place_red(grid, red4) and _place_red(grid, red5)):
        return None

    try:
        solved = _simulate(grid)
    except AssertionError:
        return None

    out = solved["grid"]
    trace = solved["trace"]
    if np.array_equal(out, grid):
        return None
    if len(_component_sprites(grid)) != 3:
        return None
    if trace["wait_conflict_total"] != 0:
        return None
    if sorted(trace["turns_per_sprite"]) != [1, 2, 3]:
        return None
    if trace["turns_from"][1] < 1 or trace["turns_from"][2] < 1:
        return None
    if trace["wait_blocked_total"] < 1:
        return None
    if trace["stops_total"] < 2:
        return None
    if len(trace["terminal_waiters"]) != 1:
        return None
    if trace["rounds"] < 10:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
