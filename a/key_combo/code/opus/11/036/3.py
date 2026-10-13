from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (BLUE, GREEN, YELLOW)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(grid: np.ndarray) -> np.ndarray:
    nz = np.argwhere(grid != BLACK)
    if nz.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = _bbox(nz)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _render_cells(cells: list[tuple[int, int, int]]) -> np.ndarray:
    coords = np.array([(r, c) for r, c, _ in cells], dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c, color in cells:
        rr = r - r0
        cc = c - c0
        prev = int(out[rr, cc])
        if prev != BLACK and prev != color:
            raise AssertionError("docked sprites overlap inconsistently")
        out[rr, cc] = color
    return out


def _find_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BLACK)
    assert len(comps) == 2, "expected exactly two sprites"
    sprites: list[dict] = []
    for coords in comps:
        b = _bbox(coords)
        red_coords = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == RED]
        assert len(red_coords) == 1, "expected one red connector per sprite"
        cell_map = {(int(r), int(c)): int(g[r, c]) for r, c in coords}
        sprites.append({"bbox": b, "coords": coords, "red": red_coords[0], "cells": cell_map})
    sprites.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return sprites


def _dock_sprites(grid: np.ndarray) -> np.ndarray:
    base, plug = _find_sprites(grid)
    br, bc = base["red"]
    pr, pc = plug["red"]
    dr = br - 1 - pr
    dc = bc - pc
    moved: list[tuple[int, int, int]] = []
    for (r, c), color in base["cells"].items():
        moved.append((r, c, GREY if color == RED else color))
    for (r, c), color in plug["cells"].items():
        moved.append((r + dr, c + dc, GREY if color == RED else color))
    return _render_cells(moved)


def _enclosed_chambers(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    chambers: list[np.ndarray] = []
    for coords in _components(g == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        chambers.append(coords)
    return chambers


def _vote_color(grid: np.ndarray, chamber: np.ndarray, global_counts: Counter[int]) -> int:
    adjacent: dict[tuple[int, int], int] = {}
    h, w = grid.shape
    for r0, c0 in chamber:
        for r, c in _neighbors4(int(r0), int(c0)):
            if 0 <= r < h and 0 <= c < w:
                color = int(grid[r, c])
                if color not in (BLACK, GREY):
                    adjacent[(r, c)] = color
    assert adjacent, "each chamber must touch at least one marker"

    counts = Counter(adjacent.values())
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    if len(winners) == 1:
        return int(winners[0])

    rarest = min(global_counts[color] for color in winners)
    winners = [color for color in winners if global_counts[color] == rarest]
    assert len(winners) == 1, "tie-break must resolve uniquely"
    return int(winners[0])


def _solve_docked(grid: np.ndarray) -> np.ndarray:
    merged = _dock_sprites(grid)
    out = merged.copy()
    global_counts = Counter(int(color) for color in merged.ravel() if int(color) not in (BLACK, GREY))
    for chamber in _enclosed_chambers(merged):
        fill = _vote_color(merged, chamber, global_counts)
        out[chamber[:, 0], chamber[:, 1]] = fill
    return _tight_crop(out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the two open sprites, then fill the newly enclosed chamber(s) using adjacent marker-color voting.

    Input structure:
    - The grid contains black background and exactly two separated 4-connected sprites made from grey body pixels, one red connector pixel per sprite, and boundary marker pixels in blue/green/yellow.
    - Each sprite is individually open: it has an inward-facing gap near the red connector and contains no enclosed black chamber on its own.

    Rule:
    - Choose the sprite whose bounding-box top-left corner comes first in row-major order as the base.
    - Translate the other sprite so its red connector lands exactly one cell above the base connector, then recolor both red connectors to grey and merge the sprites.
    - In the merged shape, find every 4-connected black region that is fully enclosed by non-black pixels.
    - For each enclosed chamber, count the orthogonally adjacent non-grey marker pixels; fill the chamber with the majority color, breaking ties by choosing the tied color that appears fewer times in the merged figure overall.
    - Return the tight crop around all non-black pixels of the filled merged figure.

    Key insight:
    - The chambers do not exist before docking; they emerge only when the two complementary openings are aligned by the red connectors, so docking must be inferred before chamber filling can be determined.
    """
    return _solve_docked(np.asarray(grid, dtype=int))


def _build_layout() -> dict:
    chamber_count = random.choice((1, 2))
    h = random.randint(7, 9)
    w_min = 8 if chamber_count == 1 else 9
    w = random.randint(w_min, 11)
    split = random.randint(2, h - 2)
    divider = None
    connector_options = [0, w - 1]
    if chamber_count == 2:
        divider = random.randint(3, w - 4)
        connector_options.append(divider)
    connector_col = random.choice(connector_options)
    return {
        "h": h,
        "w": w,
        "split": split,
        "chamber_count": chamber_count,
        "divider": divider,
        "connector_col": connector_col,
    }


def _wall_cells(layout: dict) -> set[tuple[int, int]]:
    h = layout["h"]
    w = layout["w"]
    cells: set[tuple[int, int]] = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(1, h - 1):
        cells.add((r, 0))
        cells.add((r, w - 1))
    if layout["divider"] is not None:
        for r in range(1, h - 1):
            cells.add((r, layout["divider"]))
    return cells


def _chamber_slots(layout: dict, which: str) -> list[tuple[int, int]]:
    h = layout["h"]
    w = layout["w"]
    slots: list[tuple[int, int]]
    if which == "full":
        slots = [
            (0, 2),
            (0, w - 3),
            (h - 1, 2),
            (h - 1, w - 3),
            (2, 0),
            (h - 3, 0),
            (2, w - 1),
            (h - 3, w - 1),
        ]
    elif which == "left":
        d = layout["divider"]
        slots = [
            (0, 1),
            (0, d - 1),
            (h - 1, 1),
            (h - 1, d - 1),
            (2, 0),
            (h - 3, 0),
        ]
    else:
        d = layout["divider"]
        slots = [
            (0, d + 1),
            (0, w - 2),
            (h - 1, d + 1),
            (h - 1, w - 2),
            (2, w - 1),
            (h - 3, w - 1),
        ]
    out: list[tuple[int, int]] = []
    seen = set()
    for slot in slots:
        if slot not in seen:
            out.append(slot)
            seen.add(slot)
    return out


def _safe_bias_slots(layout: dict) -> list[tuple[int, int]]:
    h = layout["h"]
    w = layout["w"]
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _make_plan(kind: str) -> dict:
    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    if kind == "tie":
        winner, loser = colors[:2]
        return {"kind": "tie", "winner": winner, "loser": loser, "counts": {winner: 2, loser: 2}}

    winner = colors[0]
    if random.random() < 0.45:
        loser1, loser2 = colors[1], colors[2]
        counts = {winner: 2, loser1: 1, loser2: 1}
    else:
        loser = colors[1]
        counts = {winner: 2, loser: 1}
    return {"kind": "strict", "winner": winner, "counts": counts}


def _plan_mix(chamber_count: int) -> list[dict]:
    if chamber_count == 1:
        kind = random.choice(("strict", "tie"))
        return [_make_plan(kind)]
    pattern = random.choice(("both_strict", "left_tie", "right_tie"))
    if pattern == "both_strict":
        return [_make_plan("strict"), _make_plan("strict")]
    if pattern == "left_tie":
        return [_make_plan("tie"), _make_plan("strict")]
    return [_make_plan("strict"), _make_plan("tie")]


def _paint_markers(layout: dict) -> np.ndarray | None:
    grid = np.zeros((layout["h"], layout["w"]), dtype=int)
    for r, c in _wall_cells(layout):
        grid[r, c] = GREY

    connector_positions = {
        (layout["split"] - 1, layout["connector_col"]),
        (layout["split"], layout["connector_col"]),
    }
    plans = _plan_mix(layout["chamber_count"])
    chamber_names = ["full"] if layout["chamber_count"] == 1 else ["left", "right"]

    used = set(connector_positions)
    tie_constraints: list[tuple[int, int]] = []
    for which, plan in zip(chamber_names, plans):
        slots = [slot for slot in _chamber_slots(layout, which) if slot not in used]
        need = sum(plan["counts"].values())
        if len(slots) < need:
            return None
        picked = random.sample(slots, need)
        index = 0
        for color, count in plan["counts"].items():
            for _ in range(count):
                r, c = picked[index]
                index += 1
                grid[r, c] = color
                used.add((r, c))
        if plan["kind"] == "tie":
            tie_constraints.append((plan["winner"], plan["loser"]))

    safe_slots = [slot for slot in _safe_bias_slots(layout) if slot not in used]
    for winner, loser in tie_constraints:
        counts = Counter(int(v) for v in grid.ravel() if int(v) in MARKER_COLORS)
        while counts[loser] <= counts[winner]:
            if not safe_slots:
                return None
            r, c = safe_slots.pop(0)
            grid[r, c] = loser
            used.add((r, c))
            counts[loser] += 1

    for r, c in connector_positions:
        grid[r, c] = GREY
    return grid


def _split_sprite_parts(assembled: np.ndarray, layout: dict) -> tuple[np.ndarray, np.ndarray]:
    split = layout["split"]
    conn_col = layout["connector_col"]
    top = np.zeros_like(assembled)
    bottom = np.zeros_like(assembled)
    top[:split, :] = assembled[:split, :]
    bottom[split:, :] = assembled[split:, :]
    top[split - 1, conn_col] = RED
    bottom[split, conn_col] = RED
    return _tight_crop(top), _tight_crop(bottom)


def _place_sprites(top: np.ndarray, bottom: np.ndarray) -> np.ndarray | None:
    hb, wb = bottom.shape
    hp, wp = top.shape
    canvas_h = max(hb, hp) + random.randint(6, 11)
    canvas_w = max(wb, wp) + random.randint(7, 12)
    if canvas_h > 30 or canvas_w > 30:
        return None

    base_r = random.randint(1, 3)
    base_c = random.randint(1, 4)
    if base_r + hb > canvas_h - 1 or base_c + wb > canvas_w - 1:
        return None

    candidates: list[tuple[int, int]] = []
    for r in range(1, canvas_h - hp):
        for c in range(1, canvas_w - wp):
            if (r, c) <= (base_r, base_c):
                continue
            if r + hp < base_r - 1 or base_r + hb < r - 1 or c + wp < base_c - 1 or base_c + wb < c - 1:
                candidates.append((r, c))
    if not candidates:
        return None

    plug_r, plug_c = random.choice(candidates)
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    for r, c in np.argwhere(bottom != BLACK):
        grid[base_r + r, base_c + c] = int(bottom[r, c])
    for r, c in np.argwhere(top != BLACK):
        grid[plug_r + r, plug_c + c] = int(top[r, c])
    return grid


def _sprite_without_red(sprite: np.ndarray) -> np.ndarray:
    out = sprite.copy()
    out[out == RED] = GREY
    return out


def _chamber_marker_colors(grid: np.ndarray, chamber: np.ndarray) -> set[int]:
    h, w = grid.shape
    colors = set()
    for r0, c0 in chamber:
        for r, c in _neighbors4(int(r0), int(c0)):
            if 0 <= r < h and 0 <= c < w:
                color = int(grid[r, c])
                if color not in (BLACK, GREY):
                    colors.add(color)
    return colors


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        sprites = _find_sprites(grid)
        if len(sprites) != 2:
            return False

        for sprite in sprites:
            r0, c0, r1, c1 = sprite["bbox"]
            local = _sprite_without_red(grid[r0 : r1 + 1, c0 : c1 + 1])
            if _enclosed_chambers(_tight_crop(local)):
                return False

        merged = _dock_sprites(grid)
        chambers = _enclosed_chambers(merged)
        if len(chambers) not in (1, 2):
            return False

        global_counts = Counter(int(color) for color in merged.ravel() if int(color) in MARKER_COLORS)
        for chamber in chambers:
            if len(_chamber_marker_colors(merged, chamber)) < 2:
                return False
            _vote_color(merged, chamber, global_counts)

        out = main(grid)
        if out.size == 0 or max(out.shape) > 30:
            return False
        if np.array_equal(out, _tight_crop(merged)):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        layout = _build_layout()
        assembled = _paint_markers(layout)
        if assembled is None:
            continue
        top, bottom = _split_sprite_parts(assembled, layout)
        grid = _place_sprites(top, bottom)
        if grid is None:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
