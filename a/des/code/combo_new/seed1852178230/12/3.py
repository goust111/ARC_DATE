from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 5
MAROON = 9

SYM_LR = "lr"
SYM_UD = "ud"
SYM_ROT = "rot"
SYMMETRY_TYPES = (SYM_LR, SYM_UD, SYM_ROT)

LEGEND_PATTERNS: dict[str, tuple[np.ndarray, ...]] = {
    SYM_LR: (
        np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=bool),
    ),
    SYM_UD: (
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    ),
    SYM_ROT: (
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]], dtype=bool),
    ),
}


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = _neighbors4 if connectivity == 4 else _neighbors8
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
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _occupied_span(mask: np.ndarray) -> tuple[int, int]:
    rows = int(np.count_nonzero(mask.any(axis=1)))
    cols = int(np.count_nonzero(mask.any(axis=0)))
    return rows, cols


def _symmetry_type(mask: np.ndarray) -> str | None:
    lr = bool(np.array_equal(mask, np.fliplr(mask)))
    ud = bool(np.array_equal(mask, np.flipud(mask)))
    rot = bool(np.array_equal(mask, np.rot90(mask, 2)))
    if lr and not ud:
        return SYM_LR
    if ud and not lr:
        return SYM_UD
    if rot and not lr and not ud:
        return SYM_ROT
    return None


def _find_legend_mapping(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    max_r = min(3, h - 8)
    max_c = min(3, w - 3)
    for r0 in range(max_r + 1):
        for c0 in range(max_c + 1):
            mapping: dict[str, int] = {}
            ok = True
            for band in range(3):
                rs = r0 + 3 * band
                color_col = grid[rs : rs + 3, c0]
                non_bg_rows = [i for i, v in enumerate(color_col.tolist()) if int(v) != BG]
                if len(non_bg_rows) != 1:
                    ok = False
                    break
                color = int(color_col[non_bg_rows[0]])
                if color in (BG, WALL, MAROON):
                    ok = False
                    break

                patch = grid[rs : rs + 3, c0 + 1 : c0 + 4]
                if patch.shape != (3, 3):
                    ok = False
                    break
                vals = set(map(int, np.unique(patch)))
                if not vals <= {BG, MAROON} or MAROON not in vals:
                    ok = False
                    break
                typ = _symmetry_type(patch == MAROON)
                if typ is None or typ in mapping:
                    ok = False
                    break
                mapping[typ] = color
            if ok and len(mapping) == 3:
                return mapping
    raise AssertionError("failed to locate the top-left color/symmetry legend")


def _room_border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    border: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        border.append((r0, c))
        border.append((r1, c))
    for r in range(r0 + 1, r1):
        border.append((r, c0))
        border.append((r, c1))
    return border


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for comp in _components(grid == WALL, connectivity=4):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue

        comp_set = set(comp)
        border = _room_border_cells(r0, c0, r1, c1)
        border_set = set(border)
        if not comp_set <= border_set:
            continue
        if len(comp_set) != len(border_set) - 1:
            continue

        missing = [cell for cell in border if cell not in comp_set]
        if len(missing) != 1:
            continue
        door = missing[0]
        if grid[door] != BG:
            continue
        if door[0] in (r0, r1) and door[1] in (c0, c1):
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior == WALL):
            continue
        pattern_mask = interior != BG
        if not bool(pattern_mask.any()):
            continue

        typ = _symmetry_type(pattern_mask)
        if typ is None:
            continue

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "door": door,
                "symmetry": typ,
            }
        )

    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each room's doorway with the legend color matching its floor-pattern symmetry.

    Input structure:
    - The background is black.
    - Several grey rectangular room walls appear as 1-pixel-thick frames with exactly one 1-pixel doorway gap on the border.
    - Each room interior contains a small single-color floor pattern on black background.
    - A top-left legend has three stacked entries: each entry contains one color square and an adjacent 3x3 maroon sample pattern on black.

    Rule:
    - Read the legend by classifying each maroon sample as left-right mirror symmetric, up-down mirror symmetric, or 180-degree-rotation-only symmetric.
    - Detect every grey room, locate its unique doorway gap, and classify the symmetry of the interior floor-pattern mask inside the room.
    - Fill only the doorway pixel with the legend color assigned to that symmetry type.
    - Leave all room walls, floor patterns, and the legend unchanged.

    Key insight:
    - The legend does not map from visible colors inside the rooms; it maps from geometric symmetry classes, so solving requires a legend lookup plus a separate per-room symmetry test.
    """
    g = np.asarray(grid, dtype=int)
    mapping = _find_legend_mapping(g)
    out = g.copy()
    for room in _find_rooms(g):
        dr, dc = room["door"]
        out[dr, dc] = mapping[room["symmetry"]]
    return out


def _random_orbit_mask(h: int, w: int, kind: str, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    p = rng.uniform(0.28, 0.58)

    if kind == SYM_LR:
        for r in range(h):
            for c in range((w + 1) // 2):
                if rng.random() < p:
                    mask[r, c] = True
                    mask[r, w - 1 - c] = True
    elif kind == SYM_UD:
        for r in range((h + 1) // 2):
            for c in range(w):
                if rng.random() < p:
                    mask[r, c] = True
                    mask[h - 1 - r, c] = True
    else:
        for r in range(h):
            for c in range(w):
                rr, cc = h - 1 - r, w - 1 - c
                if (r > rr) or (r == rr and c > cc):
                    continue
                if rng.random() < p:
                    mask[r, c] = True
                    mask[rr, cc] = True
    return mask


def _fallback_pattern(h: int, w: int, kind: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mr = h // 2
    mc = w // 2
    if kind == SYM_LR:
        for c in range(w):
            if abs(c - mc) <= max(1, w // 3):
                mask[0, c] = True
        mask[min(h - 1, 1), mc] = True
        if h > 3:
            mask[2, max(0, mc - 1)] = True
            mask[2, min(w - 1, mc + 1)] = True
    elif kind == SYM_UD:
        for r in range(h):
            if abs(r - mr) <= max(1, h // 3):
                mask[r, 0] = True
        mask[mr, min(w - 1, 1)] = True
        if w > 3:
            mask[max(0, mr - 1), 2] = True
            mask[min(h - 1, mr + 1), 2] = True
    else:
        pts = {(0, 0), (h - 1, w - 1), (min(h - 1, 1), min(w - 1, 1))}
        pts.add((h - 1 - min(h - 1, 1), w - 1 - min(w - 1, 1)))
        if h % 2 == 1 and w % 2 == 1:
            pts.add((mr, mc))
        for r, c in pts:
            mask[r, c] = True
    return mask


def _make_room_pattern(inner_h: int, inner_w: int, kind: str, rng: random.Random) -> np.ndarray:
    area = inner_h * inner_w
    for _ in range(400):
        mask = _random_orbit_mask(inner_h, inner_w, kind, rng)
        filled = int(mask.sum())
        if filled < 3 or filled > area - 2:
            continue
        if _symmetry_type(mask) != kind:
            continue
        rows, cols = _occupied_span(mask)
        if rows < 2 or cols < 2:
            continue
        comps = _components(mask, connectivity=8)
        if len(comps) != 1:
            continue
        return mask

    mask = _fallback_pattern(inner_h, inner_w, kind)
    if _symmetry_type(mask) != kind:
        raise AssertionError("fallback pattern has wrong symmetry")
    return mask


def _draw_legend(grid: np.ndarray, r0: int, c0: int, mapping: dict[str, int], rng: random.Random) -> None:
    for idx, kind in enumerate((SYM_LR, SYM_UD, SYM_ROT)):
        rs = r0 + 3 * idx
        grid[rs + 1, c0] = mapping[kind]
        pattern = rng.choice(LEGEND_PATTERNS[kind])
        grid[rs : rs + 3, c0 + 1 : c0 + 4][pattern] = MAROON


def _door_position(r0: int, c0: int, r1: int, c1: int, side: int, rng: random.Random) -> tuple[int, int]:
    if side == 0:
        return r0, rng.randint(c0 + 1, c1 - 1)
    if side == 1:
        return r1, rng.randint(c0 + 1, c1 - 1)
    if side == 2:
        return rng.randint(r0 + 1, r1 - 1), c0
    return rng.randint(r0 + 1, r1 - 1), c1


def _place_rooms(
    h: int,
    w: int,
    legend_box: tuple[int, int, int, int],
    room_specs: list[dict[str, object]],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    lr0, lc0, lr1, lc1 = legend_box
    occupied[max(0, lr0 - 1) : min(h, lr1 + 2), max(0, lc0 - 1) : min(w, lc1 + 2)] = True

    placed: list[dict[str, object]] = []
    for spec in room_specs:
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        for _ in range(300):
            r0 = rng.randint(1, h - oh - 1)
            c0 = rng.randint(1, w - ow - 1)
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            pad = occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
            if bool(pad.any()):
                continue
            occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            placed.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "inner_h": int(spec["inner_h"]),
                    "inner_w": int(spec["inner_w"]),
                    "symmetry": spec["symmetry"],
                }
            )
            break
        else:
            return None
    return placed


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.choice((3, 4))
    room_types = [SYM_LR, SYM_UD, SYM_ROT] + [rng.choice(SYMMETRY_TYPES) for _ in range(room_count - 3)]
    rng.shuffle(room_types)

    fill_colors = rng.sample([1, 2, 3, 4, 6, 7, 8], 3)
    legend_mapping = {
        SYM_LR: fill_colors[0],
        SYM_UD: fill_colors[1],
        SYM_ROT: fill_colors[2],
    }

    pattern_palette = [c for c in range(1, 10) if c not in {BG, WALL, MAROON}]
    room_specs: list[dict[str, object]] = []
    for kind in room_types:
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 6)
        room_specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "symmetry": kind,
            }
        )

    room_area = sum(int(spec["outer_h"]) * int(spec["outer_w"]) for spec in room_specs)
    min_h = max(15, 10 + max(int(spec["outer_h"]) for spec in room_specs))
    min_w = max(15, 6 + max(int(spec["outer_w"]) for spec in room_specs))
    max_h = min(30, max(min_h, 18 + room_count * 2))
    max_w = min(30, max(min_w, 18 + room_count * 2))

    if room_area > 420 or min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    grid = np.full((h, w), BG, dtype=int)

    legend_r0 = rng.randint(0, 1)
    legend_c0 = rng.randint(0, 1)
    if legend_r0 + 8 >= h or legend_c0 + 3 >= w:
        return None
    _draw_legend(grid, legend_r0, legend_c0, legend_mapping, rng)
    legend_box = (legend_r0, legend_c0, legend_r0 + 8, legend_c0 + 3)

    placed = _place_rooms(h, w, legend_box, room_specs, rng)
    if placed is None:
        return None

    room_colors = rng.sample(pattern_palette, len(placed))
    for idx, room in enumerate(placed):
        r0 = int(room["r0"])
        c0 = int(room["c0"])
        r1 = int(room["r1"])
        c1 = int(room["c1"])
        side = rng.randrange(4)
        door = _door_position(r0, c0, r1, c1, side, rng)

        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[door] = BG

        pattern_mask = _make_room_pattern(int(room["inner_h"]), int(room["inner_w"]), str(room["symmetry"]), rng)
        patch = grid[r0 + 1 : r1, c0 + 1 : c1]
        patch[pattern_mask] = room_colors[idx]

    try:
        out = main(grid)
    except Exception:
        return None

    rooms = _find_rooms(grid)
    if len(rooms) != room_count:
        return None
    if {str(room["symmetry"]) for room in rooms} != set(SYMMETRY_TYPES):
        return None
    if not bool((out != grid).any()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid random room/legend puzzle")
