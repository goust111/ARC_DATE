from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PRISM_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)

TAIL_LENGTHS = (1, 2)
GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
ARROW_COUNT_CHOICES = [1, 2, 3]
SPLIT_MODE_WEIGHTS = [2, 5, 2]  # none / some / all
EXTRA_PRISM_COUNT_WEIGHTS = [5, 3, 1]  # 0 / 1 / 2
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (int(a[0] + b[0]), int(a[1] + b[1]))


def _scale(v: tuple[int, int], k: int) -> tuple[int, int]:
    return (int(v[0] * k), int(v[1] * k))


def _rot_cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (int(dc), int(-dr))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], frozenset[tuple[int, int]]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    rel = frozenset((int(r - min_r), int(c - min_c)) for r, c in cells)
    return (int(min_r), int(min_c)), rel


def _arrow_offsets(direction: tuple[int, int], tail_len: int) -> list[tuple[int, int]]:
    if direction == RIGHT:
        cells = [(-1, -2), (0, -2), (1, -2), (0, -1), (0, 0)]
        cells.extend((0, int(-2 - k)) for k in range(1, tail_len + 1))
        return cells
    if direction == LEFT:
        cells = [(-1, 2), (0, 2), (1, 2), (0, 1), (0, 0)]
        cells.extend((0, int(2 + k)) for k in range(1, tail_len + 1))
        return cells
    if direction == UP:
        cells = [(2, -1), (2, 0), (2, 1), (1, 0), (0, 0)]
        cells.extend((int(2 + k), 0) for k in range(1, tail_len + 1))
        return cells
    if direction == DOWN:
        cells = [(-2, -1), (-2, 0), (-2, 1), (-1, 0), (0, 0)]
        cells.extend((int(-2 - k), 0) for k in range(1, tail_len + 1))
        return cells
    raise AssertionError(direction)


def _arrow_cells_from_tip(
    tip: tuple[int, int], direction: tuple[int, int], tail_len: int
) -> set[tuple[int, int]]:
    tr, tc = tip
    return {(int(tr + dr), int(tc + dc)) for dr, dc in _arrow_offsets(direction, tail_len)}


_ARROW_PATTERNS: dict[frozenset[tuple[int, int]], dict[str, object]] = {}
for _dir in DIRS:
    for _tail_len in TAIL_LENGTHS:
        _cells = [(0 + dr, 0 + dc) for dr, dc in _arrow_offsets(_dir, _tail_len)]
        (_min_r, _min_c), _rel = _normalize(_cells)
        _ARROW_PATTERNS[_rel] = {
            "direction": _dir,
            "tail_len": int(_tail_len),
            "tip_rel": (int(-_min_r), int(-_min_c)),
        }


def _match_arrow_component(cells: list[tuple[int, int]]) -> dict[str, object] | None:
    (r0, c0), rel = _normalize(cells)
    pattern = _ARROW_PATTERNS.get(rel)
    if pattern is None:
        return None
    tip_rel = pattern["tip_rel"]
    assert isinstance(tip_rel, tuple)
    tip = (int(r0 + tip_rel[0]), int(c0 + tip_rel[1]))
    return {
        "cells": set(cells),
        "direction": pattern["direction"],
        "tail_len": pattern["tail_len"],
        "tip": tip,
    }


def _find_arrows(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    candidates: list[tuple[int, list[dict[str, object]]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        comps = _components(grid == int(color))
        if not comps:
            continue
        infos: list[dict[str, object]] = []
        ok = True
        for comp in comps:
            info = _match_arrow_component(comp)
            if info is None:
                ok = False
                break
            infos.append(info)
        if ok:
            candidates.append((int(color), infos))

    if len(candidates) == 1:
        color, infos = candidates[0]
    else:
        fallback = [(color, infos) for color, infos in candidates if int(color) == GREY]
        assert len(fallback) == 1, "expected a unique arrow color"
        color, infos = fallback[0]

    infos.sort(key=lambda info: (info["tip"][0], info["tip"][1]))  # type: ignore[index]
    return int(color), infos


def _match_prism_component(cells: list[tuple[int, int]], color: int) -> dict[str, object] | None:
    rows = sorted({int(r) for r, _c in cells})
    cols = sorted({int(c) for _r, c in cells})
    cell_set = set(cells)
    if len(rows) == 1:
        row = rows[0]
        expected = {(int(row), int(c)) for c in range(cols[0], cols[-1] + 1)}
        if cell_set != expected:
            return None
        return {
            "color": int(color),
            "orientation": "h",
            "cells": expected,
        }
    if len(cols) == 1:
        col = cols[0]
        expected = {(int(r), int(col)) for r in range(rows[0], rows[-1] + 1)}
        if cell_set != expected:
            return None
        return {
            "color": int(color),
            "orientation": "v",
            "cells": expected,
        }
    return None


def _find_prisms(grid: np.ndarray, bg: int, arrow_color: int) -> list[dict[str, object]]:
    prisms: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg), int(arrow_color)}):
        for comp in _components(grid == int(color)):
            info = _match_prism_component(comp, int(color))
            assert info is not None, "non-arrow foreground components must be straight prisms"
            prisms.append(info)
    return prisms


def _simulate(
    shape: tuple[int, int],
    arrows: list[dict[str, object]],
    prism_by_cell: dict[tuple[int, int], int],
    out: np.ndarray,
    arrow_color: int,
) -> np.ndarray:
    beams: list[dict[str, object]] = [
        {
            "tip": arrow["tip"],
            "direction": arrow["direction"],
            "color": int(arrow_color),
            "can_split": True,
        }
        for arrow in arrows
    ]
    h, w = shape

    while beams:
        beam = beams.pop()
        r, c = beam["tip"]  # type: ignore[misc]
        dr, dc = beam["direction"]  # type: ignore[misc]
        step = 0
        can_split = bool(beam["can_split"])
        paint_color = int(beam["color"])

        while True:
            r += int(dr)
            c += int(dc)
            step += 1
            if not (0 <= r < h and 0 <= c < w):
                break

            prism_color = prism_by_cell.get((int(r), int(c)))
            if prism_color is not None:
                if can_split:
                    beams.append(
                        {
                            "tip": (int(r), int(c)),
                            "direction": _rot_cw((int(dr), int(dc))),
                            "color": int(prism_color),
                            "can_split": False,
                        }
                    )
                    can_split = False
                continue

            if step % 2 == 1:
                out[int(r), int(c)] = int(paint_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the grey arrow objects, keep the colored prisms, and trace each arrow tip.

    The prompt mixes two incompatible phrases for the split branch. This solver uses the
    self-consistent interpretation implemented by the generator: on the first prism hit,
    the original beam continues straight and a single copy turns 90 degrees clockwise,
    painting a dashed trail in the prism's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    arrow_color, arrows = _find_arrows(g, bg)
    prisms = _find_prisms(g, bg, arrow_color)

    out = g.copy()
    out[out == int(arrow_color)] = int(bg)

    prism_by_cell: dict[tuple[int, int], int] = {}
    for prism in prisms:
        color = int(prism["color"])
        for cell in prism["cells"]:  # type: ignore[assignment]
            prism_by_cell[cell] = int(color)

    return _simulate(g.shape, arrows, prism_by_cell, out, arrow_color)


def _steps_to_edge(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = start
    dr, dc = direction
    steps = 0
    while True:
        r += int(dr)
        c += int(dc)
        if not (0 <= r < h and 0 <= c < w):
            return int(steps)
        steps += 1


def _prism_cells_from_hit(
    hit: tuple[int, int], direction: tuple[int, int], prism_len: int
) -> set[tuple[int, int]]:
    clone_dir = _rot_cw(direction)
    hr, hc = hit
    cells: list[tuple[int, int]] = []
    if direction in (UP, DOWN):
        if clone_dir == LEFT:
            cells = [(int(hr), int(hc + k)) for k in range(prism_len)]
        else:
            assert clone_dir == RIGHT
            cells = [(int(hr), int(hc - k)) for k in range(prism_len)]
    else:
        if clone_dir == DOWN:
            cells = [(int(hr - k), int(hc)) for k in range(prism_len)]
        else:
            assert clone_dir == UP
            cells = [(int(hr + k), int(hc)) for k in range(prism_len)]
    return set(cells)


def _scenario_from_params(
    shape: tuple[int, int],
    tip: tuple[int, int],
    direction: tuple[int, int],
    tail_len: int,
    split_step: int | None,
    prism_len: int | None,
) -> dict[str, object] | None:
    arrow_cells = _arrow_cells_from_tip(tip, direction, tail_len)
    if not all(_in_bounds(shape, r, c) for r, c in arrow_cells):
        return None

    path_len = _steps_to_edge(shape, tip, direction)
    if path_len < 3:
        return None

    prism_cells: set[tuple[int, int]] = set()
    clone_positions: list[tuple[int, int]] = []
    clone_paint: list[tuple[int, int]] = []
    if split_step is not None:
        if prism_len is None:
            return None
        if not (2 <= split_step <= path_len - 1):
            return None
        hit = _add(tip, _scale(direction, split_step))
        prism_cells = _prism_cells_from_hit(hit, direction, prism_len)
        if hit not in prism_cells:
            return None
        if not all(_in_bounds(shape, r, c) for r, c in prism_cells):
            return None
        if prism_cells & arrow_cells:
            return None

        clone_dir = _rot_cw(direction)
        clone_len = _steps_to_edge(shape, hit, clone_dir)
        if clone_len < 3:
            return None
        for step in range(1, clone_len + 1):
            pos = _add(hit, _scale(clone_dir, step))
            clone_positions.append(pos)
            if step % 2 == 1 and pos not in prism_cells:
                clone_paint.append(pos)
        if len(clone_paint) < 2:
            return None

    original_positions: list[tuple[int, int]] = []
    original_paint: list[tuple[int, int]] = []
    for step in range(1, path_len + 1):
        pos = _add(tip, _scale(direction, step))
        original_positions.append(pos)
        if split_step is not None and step == split_step:
            continue
        if step % 2 == 1:
            original_paint.append(pos)

    if len(original_paint) < 2:
        return None

    return {
        "tip": tip,
        "direction": direction,
        "tail_len": int(tail_len),
        "arrow_cells": arrow_cells,
        "prism_cells": prism_cells,
        "original_positions": original_positions,
        "original_paint": original_paint,
        "clone_positions": clone_positions,
        "clone_paint": clone_paint,
        "split": split_step is not None,
    }


def _halo(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _random_tip(shape: tuple[int, int], direction: tuple[int, int], tail_len: int) -> tuple[int, int] | None:
    h, w = shape
    offsets = _arrow_offsets(direction, tail_len)
    min_r = min(dr for dr, _dc in offsets)
    max_r = max(dr for dr, _dc in offsets)
    min_c = min(dc for _dr, dc in offsets)
    max_c = max(dc for _dr, dc in offsets)
    rows = list(range(int(-min_r), int(h - max_r)))
    cols = list(range(int(-min_c), int(w - max_c)))
    if not rows or not cols:
        return None
    return (int(random.choice(rows)), int(random.choice(cols)))


def _pick_split_flags(n_arrows: int) -> list[bool]:
    mode = random.choices(["none", "some", "all"], weights=SPLIT_MODE_WEIGHTS, k=1)[0]
    if n_arrows == 1:
        if mode == "none":
            return [False]
        if mode == "all":
            return [True]
        return [bool(random.getrandbits(1))]
    if mode == "none":
        return [False] * n_arrows
    if mode == "all":
        return [True] * n_arrows
    count = random.randint(1, n_arrows - 1)
    idxs = set(random.sample(range(n_arrows), count))
    return [i in idxs for i in range(n_arrows)]


def _render_candidate(
    shape: tuple[int, int],
    scenarios: list[dict[str, object]],
    prism_infos: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)

    for scenario in scenarios:
        for r, c in scenario["arrow_cells"]:  # type: ignore[assignment]
            inp[int(r), int(c)] = GREY
        for r, c in scenario["original_paint"]:  # type: ignore[assignment]
            out[int(r), int(c)] = GREY

    for prism in prism_infos:
        color = int(prism["color"])
        for r, c in prism["cells"]:  # type: ignore[assignment]
            inp[int(r), int(c)] = int(color)
            out[int(r), int(c)] = int(color)

    for scenario in scenarios:
        color = scenario.get("prism_color")
        if color is None:
            continue
        for r, c in scenario["clone_paint"]:  # type: ignore[assignment]
            out[int(r), int(c)] = int(color)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (int(h), int(w))

    n_arrows = random.choice(ARROW_COUNT_CHOICES)
    directions = random.sample(list(DIRS), n_arrows)
    tail_lengths = [random.choice(TAIL_LENGTHS) for _ in range(n_arrows)]
    split_flags = _pick_split_flags(n_arrows)

    occupied_input: set[tuple[int, int]] = set()
    occupied_halo: set[tuple[int, int]] = set()
    reserved_path: set[tuple[int, int]] = set()
    reserved_paint: set[tuple[int, int]] = set()

    scenarios: list[dict[str, object]] = []
    for direction, tail_len, wants_split in zip(directions, tail_lengths, split_flags, strict=True):
        placed = False
        for _ in range(250):
            tip = _random_tip(shape, direction, tail_len)
            if tip is None:
                break
            path_len = _steps_to_edge(shape, tip, direction)
            if wants_split:
                if path_len < 5:
                    continue
                split_lo = 3 if path_len >= 6 else 2
                split_step = random.randint(split_lo, path_len - 1)
                hit = _add(tip, _scale(direction, split_step))
                clone_dir = _rot_cw(direction)
                clone_len = _steps_to_edge(shape, hit, clone_dir)
                if clone_len < 3:
                    continue
                if direction == DOWN:
                    max_prism_len = w - hit[1]
                elif direction == UP:
                    max_prism_len = hit[1] + 1
                elif direction == RIGHT:
                    max_prism_len = hit[0] + 1
                else:
                    max_prism_len = h - hit[0]
                if max_prism_len < 3:
                    continue
                prism_len = random.randint(3, min(6, max_prism_len))
            else:
                split_step = None
                prism_len = None

            scenario = _scenario_from_params(shape, tip, direction, tail_len, split_step, prism_len)
            if scenario is None:
                continue

            input_cells = set(scenario["arrow_cells"]) | set(scenario["prism_cells"])  # type: ignore[arg-type]
            path_cells = (
                set(scenario["original_positions"])  # type: ignore[arg-type]
                | set(scenario["clone_positions"])  # type: ignore[arg-type]
                | set(scenario["prism_cells"])  # type: ignore[arg-type]
            )
            paint_cells = set(scenario["original_paint"]) | set(scenario["clone_paint"])  # type: ignore[arg-type]

            if input_cells & occupied_halo:
                continue
            if input_cells & reserved_path:
                continue
            if path_cells & occupied_input:
                continue
            if path_cells & reserved_path:
                continue
            if paint_cells & reserved_paint:
                continue

            occupied_input |= input_cells
            occupied_halo |= _halo(shape, input_cells)
            reserved_path |= path_cells
            reserved_paint |= paint_cells
            scenarios.append(scenario)
            placed = True
            break
        if not placed:
            return None

    extra_prisms_needed = random.choices([0, 1, 2], weights=EXTRA_PRISM_COUNT_WEIGHTS, k=1)[0]
    split_count = sum(1 for scenario in scenarios if bool(scenario["split"]))
    total_prisms = split_count + int(extra_prisms_needed)
    if total_prisms > len(PRISM_COLORS):
        return None
    colors = random.sample(PRISM_COLORS, total_prisms)

    prism_infos: list[dict[str, object]] = []
    color_idx = 0
    for scenario in scenarios:
        if not bool(scenario["split"]):
            continue
        prism_color = int(colors[color_idx])
        color_idx += 1
        scenario["prism_color"] = int(prism_color)
        prism_infos.append(
            {
                "color": int(prism_color),
                "cells": set(scenario["prism_cells"]),  # type: ignore[arg-type]
            }
        )

    for _ in range(extra_prisms_needed):
        placed = False
        prism_color = int(colors[color_idx])
        color_idx += 1
        for _ in range(200):
            orient = random.choice(["h", "v"])
            prism_len = random.randint(3, 6)
            if orient == "h":
                if prism_len > w:
                    continue
                r = random.randint(0, h - 1)
                c0 = random.randint(0, w - prism_len)
                cells = {(int(r), int(c)) for c in range(c0, c0 + prism_len)}
            else:
                if prism_len > h:
                    continue
                r0 = random.randint(0, h - prism_len)
                c = random.randint(0, w - 1)
                cells = {(int(r), int(c)) for r in range(r0, r0 + prism_len)}

            if cells & occupied_halo:
                continue
            if cells & reserved_path:
                continue

            occupied_input |= cells
            occupied_halo |= _halo(shape, cells)
            prism_infos.append({"color": int(prism_color), "cells": cells})
            placed = True
            break
        if not placed:
            return None

    inp, expected = _render_candidate(shape, scenarios, prism_infos)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(inp, expected):
        return None
    if not np.any(expected == GREY):
        return None
    used_split_colors = [int(scenario["prism_color"]) for scenario in scenarios if "prism_color" in scenario]
    if used_split_colors and not np.any(np.isin(expected, used_split_colors)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
