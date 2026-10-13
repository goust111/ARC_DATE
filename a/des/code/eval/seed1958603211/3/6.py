from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
AZURE = 8
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- shared geometry helpers ---------------


def _add(pos: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return pos[0] + direction[0] * steps, pos[1] + direction[1] * steps


def _dot(a: tuple[int, int], b: tuple[int, int]) -> int:
    return a[0] * b[0] + a[1] * b[1]


def _perp(direction: tuple[int, int]) -> tuple[int, int]:
    return (0, 1) if direction[0] else (1, 0)


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((0, -1), (0, 1)) if direction[0] else ((-1, 0), (1, 0))


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(pos: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = pos
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _expand(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    out.add((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


# --------------- solver ---------------


def _ordered_wings(center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    pr, pc = _perp(direction)
    return [(center[0] - pr, center[1] - pc), (center[0] + pr, center[1] + pc)]


def _arrow_shape(center: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    wings = _ordered_wings(center, direction)
    return {
        center,
        _add(center, direction, 1),
        _add(center, direction, 2),
        wings[0],
        wings[1],
    }


def _parse_arrow(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    coord_set = set(coords)
    if len(coord_set) != 5:
        return None
    for center in coord_set:
        for direction in DIRS:
            expected = _arrow_shape(center, direction)
            if expected != coord_set:
                continue
            core = int(grid[center])
            if int(grid[_add(center, direction, 1)]) != core:
                continue
            if int(grid[_add(center, direction, 2)]) != core:
                continue
            wings = _ordered_wings(center, direction)
            return {
                "color": core,
                "center": center,
                "dir": direction,
                "tip": _add(center, direction, 2),
                "wings": wings,
                "cells": coord_set,
            }
    return None


def _find_arrows(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != AZURE)
    arrows: list[dict] = []
    for comp in _components(mask):
        arrow = _parse_arrow(grid, comp)
        assert arrow is not None, f"expected a 5-cell arrow, got component {comp}"
        arrows.append(arrow)
    assert arrows, "expected at least one arrow pair"
    return arrows


def _mirror_lookup(grid: np.ndarray) -> dict[tuple[int, int], dict]:
    lookup: dict[tuple[int, int], dict] = {}
    for comp in _components(grid == AZURE):
        coords = set(comp)
        rows = sorted({r for r, _c in coords})
        cols = sorted({c for _r, c in coords})
        if len(rows) == 1:
            orientation = "horizontal"
            assert cols == list(range(cols[0], cols[-1] + 1))
        elif len(cols) == 1:
            orientation = "vertical"
            assert rows == list(range(rows[0], rows[-1] + 1))
        else:
            raise AssertionError(f"mirror must be straight: {comp}")
        assert 2 <= len(coords) <= 5
        for cell in coords:
            neighbors = [n for n in _neighbors4(cell) if n in coords]
            lookup[cell] = {"orientation": orientation, "neighbors": neighbors}
    return lookup


def _trace_ray(
    grid: np.ndarray,
    start_tip: tuple[int, int],
    direction: tuple[int, int],
    target_tip: tuple[int, int],
    mirror_lookup: dict[tuple[int, int], dict],
) -> dict | None:
    cur = start_tip
    dr, dc = direction
    path: list[tuple[int, int]] = []
    reflections = 0
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (cur[0], cur[1], dr, dc)
        if state in seen:
            return None
        seen.add(state)
        nxt = (cur[0] + dr, cur[1] + dc)
        if not _in_bounds(grid.shape, nxt):
            return None
        if nxt == target_tip:
            return {"path": path, "reflections": reflections}
        if nxt in mirror_lookup:
            info = mirror_lookup[nxt]
            if info["orientation"] == "horizontal":
                if dr == 0:
                    return None
            else:
                if dc == 0:
                    return None
            if len(info["neighbors"]) != 1:
                return None
            neighbor = info["neighbors"][0]
            dr = nxt[0] - neighbor[0]
            dc = nxt[1] - neighbor[1]
            reflections += 1
            cur = nxt
            continue
        if int(grid[nxt]) != BG:
            return None
        path.append(nxt)
        cur = nxt


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows = _find_arrows(g)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for arrow in arrows:
        by_color[int(arrow["color"])].append(arrow)
    mirror_lookup = _mirror_lookup(g)

    out = g.copy()
    for color, pair in by_color.items():
        assert len(pair) == 2, f"color {color} should have exactly two arrows"
        a, b = pair
        a_trace = _trace_ray(g, a["tip"], a["dir"], b["tip"], mirror_lookup)
        b_trace = _trace_ray(g, b["tip"], b["dir"], a["tip"], mirror_lookup)
        assert (a_trace is None) != (b_trace is None), f"color {color} needs one unique emitter"
        emitter, receiver, trace = (a, b, a_trace) if a_trace is not None else (b, a, b_trace)
        assert trace is not None
        for r, c in trace["path"]:
            out[r, c] = int(color)
        wing_colors = [int(g[pos]) for pos in emitter["wings"]]
        if trace["reflections"] % 2 == 1:
            wing_colors.reverse()
        for pos, wing_color in zip(receiver["wings"], wing_colors):
            out[pos] = int(wing_color)
    return out


# --------------- generator ---------------


SPEC_ATTEMPTS = 200
PLACE_ATTEMPTS = 120
GEN_ATTEMPTS = 400
GRID_SIZE_RANGE = (18, 28)
PAIR_COUNT_RANGE = (1, 3)
SEG_LEN_RANGE = (3, 6)
MIRROR_LEN_RANGE = (2, 5)


def _draw_arrow_cells(
    cell_colors: dict[tuple[int, int], int],
    center: tuple[int, int],
    direction: tuple[int, int],
    core_color: int,
    wing_colors: list[int],
) -> bool:
    shaft = [center, _add(center, direction, 1), _add(center, direction, 2)]
    for pos in shaft:
        if pos in cell_colors:
            return False
        cell_colors[pos] = int(core_color)
    for pos, wing_color in zip(_ordered_wings(center, direction), wing_colors):
        if pos in cell_colors:
            return False
        cell_colors[pos] = int(wing_color)
    return True


def _build_pair_spec(rng: random.Random, core_color: int) -> dict | None:
    wing_palette = [c for c in PAIR_COLORS if c != core_color]
    for _ in range(SPEC_ATTEMPTS):
        reflections = 2 if rng.random() < 0.5 else rng.choice((1, 3))
        directions = [rng.choice(DIRS)]
        for _turn in range(reflections):
            directions.append(rng.choice(_perpendiculars(directions[-1])))
        seg_lengths = [rng.randint(*SEG_LEN_RANGE) for _ in range(reflections + 1)]
        points = [(0, 0)]
        for direction, seg_len in zip(directions, seg_lengths):
            points.append(_add(points[-1], direction, seg_len))

        emitter_tip = points[0]
        receiver_tip = points[-1]
        emitter_dir = directions[0]
        final_dir = directions[-1]
        receiver_dir = (-final_dir[0], -final_dir[1])

        vec_to_receiver = (receiver_tip[0] - emitter_tip[0], receiver_tip[1] - emitter_tip[1])
        if _dot(emitter_dir, vec_to_receiver) <= 0:
            continue
        vec_to_emitter = (-vec_to_receiver[0], -vec_to_receiver[1])
        if _dot(receiver_dir, vec_to_emitter) > 0:
            continue

        emitter_center = _add(emitter_tip, emitter_dir, -2)
        receiver_center = _add(receiver_tip, final_dir, 2)

        emitter_wings = rng.sample(wing_palette, 2)
        projected_wings = emitter_wings if reflections % 2 == 0 else emitter_wings[::-1]
        receiver_wings = rng.sample(wing_palette, 2)
        if receiver_wings == projected_wings:
            continue

        cell_colors: dict[tuple[int, int], int] = {}
        if not _draw_arrow_cells(cell_colors, emitter_center, emitter_dir, core_color, emitter_wings):
            continue
        if not _draw_arrow_cells(cell_colors, receiver_center, receiver_dir, core_color, receiver_wings):
            continue

        for idx in range(reflections):
            mirror_start = points[idx + 1]
            out_dir = directions[idx + 1]
            mirror_len = rng.randint(*MIRROR_LEN_RANGE)
            for step in range(mirror_len):
                pos = _add(mirror_start, out_dir, -step)
                if pos in cell_colors:
                    break
                cell_colors[pos] = AZURE
            else:
                continue
            break
        else:
            expected_path: set[tuple[int, int]] = set()
            ok = True
            for idx, direction in enumerate(directions):
                start = points[idx]
                seg_len = seg_lengths[idx]
                for step in range(1, seg_len + 1):
                    pos = _add(start, direction, step)
                    if idx < reflections and step == seg_len:
                        if cell_colors.get(pos) != AZURE:
                            ok = False
                            break
                        continue
                    if idx == reflections and step == seg_len:
                        if pos != receiver_tip:
                            ok = False
                        continue
                    if pos in cell_colors:
                        ok = False
                        break
                    expected_path.add(pos)
                if not ok:
                    break
            if not ok:
                continue

            all_cells = set(cell_colors) | expected_path | set(points)
            min_r = min(r for r, _c in all_cells) - 1
            max_r = max(r for r, _c in all_cells) + 1
            min_c = min(c for _r, c in all_cells) - 1
            max_c = max(c for _r, c in all_cells) + 1
            shift = (-min_r, -min_c)
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            local = np.zeros((h, w), dtype=int)
            input_cells: list[tuple[int, int, int]] = []
            for (r, c), color in cell_colors.items():
                rr = r + shift[0]
                cc = c + shift[1]
                local[rr, cc] = int(color)
                input_cells.append((rr, cc, int(color)))
            shifted_path = {(r + shift[0], c + shift[1]) for r, c in expected_path}

            try:
                out = main(local)
            except AssertionError:
                continue
            actual_path = {
                (r, c)
                for r in range(h)
                for c in range(w)
                if int(local[r, c]) == BG and int(out[r, c]) != BG
            }
            if actual_path != shifted_path:
                continue
            if not any(int(local[r, c]) != int(out[r, c]) for r, c, _color in input_cells):
                continue
            return {"shape": local.shape, "cells": input_cells, "path": actual_path}
    return None


def _place_spec(
    grid: np.ndarray,
    spec: dict,
    occupied_forbidden: set[tuple[int, int]],
    path_forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    sh, sw = spec["shape"]
    if sh > h or sw > w:
        return None
    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(0, h - sh)
        left = rng.randint(0, w - sw)
        translated_input = {(r + top, c + left) for r, c, _color in spec["cells"]}
        translated_path = {(r + top, c + left) for r, c in spec["path"]}
        if translated_input & occupied_forbidden:
            continue
        if translated_input & path_forbidden:
            continue
        if translated_path & occupied_forbidden:
            continue
        if translated_path & path_forbidden:
            continue
        for r, c, color in spec["cells"]:
            grid[r + top, c + left] = int(color)
        return translated_input, translated_path
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    n_pairs = rng.randint(*PAIR_COUNT_RANGE)
    core_colors = rng.sample(PAIR_COLORS, n_pairs)

    specs: list[dict] = []
    for core_color in core_colors:
        spec = _build_pair_spec(rng, core_color)
        if spec is None:
            return None
        specs.append(spec)

    grid = np.zeros((h, w), dtype=int)
    occupied_forbidden: set[tuple[int, int]] = set()
    path_forbidden: set[tuple[int, int]] = set()
    for spec in sorted(specs, key=lambda x: x["shape"][0] * x["shape"][1], reverse=True):
        placed = _place_spec(grid, spec, occupied_forbidden, path_forbidden, rng)
        if placed is None:
            return None
        input_cells, path_cells = placed
        occupied_forbidden |= _expand(input_cells, grid.shape)
        path_forbidden |= _expand(path_cells, grid.shape)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
