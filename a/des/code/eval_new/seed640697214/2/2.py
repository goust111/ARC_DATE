import random
from typing import Callable

import numpy as np


BG = 8
PALETTE = (0, 1, 2, 3, 4, 5, 6, 7, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr:
        return ((0, -1), (0, 1))
    return ((-1, 0), (1, 0))


def _color_components(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
    comps: list[set[tuple[int, int]]] = []
    while cells:
        start = cells.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                nxt = (r + dr, c + dc)
                if nxt in cells:
                    cells.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _parse_arrow_component(comp: set[tuple[int, int]]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for seed in comp:
        sr, sc = seed
        for direction in DIRS:
            dr, dc = direction
            if (sr + dr, sc + dc) not in comp:
                continue
            length = 0
            while (sr + (length + 1) * dr, sc + (length + 1) * dc) in comp:
                length += 1
            if length < 1:
                continue
            stem = {(sr + k * dr, sc + k * dc) for k in range(length + 1)}
            end = (sr + length * dr, sc + length * dc)
            cap_options = {tuple(np.add(end, perp)) for perp in _perpendiculars(direction)}
            extra = comp - stem
            if not extra or not extra <= cap_options:
                continue
            candidates.append(
                {
                    "seed": seed,
                    "direction": direction,
                    "length": length,
                    "cap": tuple(sorted(extra)),
                }
            )
    unique: list[dict[str, object]] = []
    seen = set()
    for item in candidates:
        key = (
            item["seed"],
            item["direction"],
            item["length"],
            item["cap"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    assert len(unique) == 1, f"expected one valid arrow parse, got {len(unique)}"
    return unique[0]


def _band_mask(
    shape: tuple[int, int],
    seed: tuple[int, int],
    direction: tuple[int, int],
    radius: int,
) -> np.ndarray:
    h, w = shape
    sr, sc = seed
    dr, dc = direction
    mask = np.zeros(shape, dtype=bool)
    if dr == 0:
        r0 = max(0, sr - radius)
        r1 = min(h, sr + radius + 1)
        c0, c1 = (sc, w) if dc > 0 else (0, sc + 1)
        mask[r0:r1, c0:c1] = True
    else:
        c0 = max(0, sc - radius)
        c1 = min(w, sc + radius + 1)
        r0, r1 = (sr, h) if dr > 0 else (0, sr + 1)
        mask[r0:r1, c0:c1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each seed color toward the grid edge using the direction and length encoded by its same-color arrow.

    Input structure:
    - The background is the dominant azure field.
    - Every non-background color forms exactly one connected seed-plus-arrow object.
    - Each object has a seed endpoint, a straight stem extending up/down/left/right, and a perpendicular terminal cap at the far end.

    Rule:
    - Read the seed as the endpoint from which the straight stem starts; the stem length N is the number of same-color stem pixels after the seed.
    - Remove the arrow geometry, keep the seed, and paint a strip from the seed to the grid edge in the stem direction with perpendicular half-width N.
    - If strips overlap, the longer stem wins; equal lengths are resolved by the smaller seed row, with seed column used only as a deterministic final tiebreak.

    Key insight:
    - One visual measurement, the stem length, controls both the strip width and the overlap priority, so the arrow encodes two coupled parameters at once.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    arrows: list[dict[str, object]] = []
    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        comps = _color_components(g, color)
        assert len(comps) == 1, "each color should form one connected arrow object"
        parsed = _parse_arrow_component(comps[0])
        parsed["color"] = color
        arrows.append(parsed)
    assert arrows, "expected at least one arrow"

    out = np.full_like(g, bg)
    seed_colors: dict[tuple[int, int], int] = {}
    for arrow in arrows:
        seed = tuple(arrow["seed"])
        seed_colors[seed] = int(arrow["color"])

    ordered = sorted(
        arrows,
        key=lambda item: (
            int(item["length"]),
            -int(item["seed"][0]),
            -int(item["seed"][1]),
        ),
    )
    for arrow in ordered:
        mask = _band_mask(
            g.shape,
            tuple(arrow["seed"]),
            tuple(arrow["direction"]),
            int(arrow["length"]),
        )
        out[mask] = int(arrow["color"])

    for seed, color in seed_colors.items():
        out[seed] = color
    return out


def _component_offsets(
    direction: tuple[int, int],
    length: int,
    cap: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    offsets = [(0, 0)]
    offsets.extend((dr * k, dc * k) for k in range(1, length + 1))
    offsets.append((dr * length + cap[0], dc * length + cap[1]))
    return offsets


def _component_cells(
    seed: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    cap: tuple[int, int],
) -> set[tuple[int, int]]:
    sr, sc = seed
    return {(sr + dr, sc + dc) for dr, dc in _component_offsets(direction, length, cap)}


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                out.add((rr, cc))
    return out


def _max_length(shape: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    return (h - 2) if dr else (w - 2)


def _sample_arrow(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    direction: tuple[int, int] | None = None,
    length: int | None = None,
    seed_ok: Callable[[tuple[int, int]], bool] | None = None,
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(250):
        cur_dir = direction if direction is not None else random.choice(DIRS)
        cur_max = min(5, _max_length(shape, cur_dir))
        if cur_max < 2:
            return None
        cur_len = length if length is not None else random.randint(2, cur_max)
        if cur_len < 2 or cur_len > cur_max:
            continue
        cap = random.choice(_perpendiculars(cur_dir))
        offsets = _component_offsets(cur_dir, cur_len, cap)
        row_offsets = [dr for dr, _ in offsets]
        col_offsets = [dc for _, dc in offsets]
        r_lo = -min(row_offsets)
        r_hi = h - 1 - max(row_offsets)
        c_lo = -min(col_offsets)
        c_hi = w - 1 - max(col_offsets)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        seed = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
        if seed_ok is not None and not seed_ok(seed):
            continue
        cells = _component_cells(seed, cur_dir, cur_len, cap)
        if any(cell in blocked for cell in cells):
            continue
        return {
            "seed": seed,
            "direction": cur_dir,
            "length": cur_len,
            "cap_dir": cap,
            "cells": cells,
        }
    return None


def _band_cells(shape: tuple[int, int], arrow: dict[str, object]) -> set[tuple[int, int]]:
    return {
        tuple(map(int, rc))
        for rc in np.argwhere(
            _band_mask(
                shape,
                tuple(arrow["seed"]),
                tuple(arrow["direction"]),
                int(arrow["length"]),
            )
        )
    }


def _overlap_modes(shape: tuple[int, int], arrows: list[dict[str, object]]) -> dict[str, bool] | None:
    seed_cells = {tuple(arrow["seed"]) for arrow in arrows}
    bands = [_band_cells(shape, arrow) for arrow in arrows]
    has_tie = False
    has_longer = False
    for i in range(len(arrows)):
        for j in range(i + 1, len(arrows)):
            overlap = (bands[i] & bands[j]) - seed_cells
            if not overlap:
                continue
            len_i = int(arrows[i]["length"])
            len_j = int(arrows[j]["length"])
            row_i = int(arrows[i]["seed"][0])
            row_j = int(arrows[j]["seed"][0])
            if len_i == len_j:
                if row_i == row_j:
                    return None
                has_tie = True
            else:
                has_longer = True
    return {"tie": has_tie, "longer": has_longer}


def _render_grid(shape: tuple[int, int], arrows: list[dict[str, object]], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for arrow, color in zip(arrows, colors):
        for r, c in arrow["cells"]:
            grid[r, c] = color
    return grid


def _build_scene(mode: str) -> np.ndarray | None:
    h = random.randint(12, 26)
    w = random.randint(12, 26)
    shape = (h, w)
    direction = random.choice(DIRS)

    max_len = min(5, _max_length(shape, direction))
    if max_len < 2:
        return None

    if mode == "tie":
        pair_len = random.randint(2, max_len)
        pair_lengths = (pair_len, pair_len)
    else:
        if max_len < 3:
            return None
        pair_lengths = tuple(sorted(random.sample(range(2, max_len + 1), 2)))

    blocked: set[tuple[int, int]] = set()
    arrows: list[dict[str, object]] = []

    first = _sample_arrow(shape, blocked, direction=direction, length=pair_lengths[0])
    if first is None:
        return None
    arrows.append(first)
    blocked |= _halo(first["cells"], shape)

    if direction[0] == 0:
        max_delta = pair_lengths[0] + pair_lengths[1]
        second_ok = lambda seed: abs(seed[0] - first["seed"][0]) <= max_delta and (
            mode != "tie" or seed[0] != first["seed"][0]
        )
    else:
        max_delta = pair_lengths[0] + pair_lengths[1]
        second_ok = lambda seed: abs(seed[1] - first["seed"][1]) <= max_delta and (
            mode != "tie" or seed[1] != first["seed"][1]
        )

    second = _sample_arrow(
        shape,
        blocked,
        direction=direction,
        length=pair_lengths[1],
        seed_ok=second_ok,
    )
    if second is None:
        return None
    arrows.append(second)
    blocked |= _halo(second["cells"], shape)

    num_arrows = random.randint(2, 5)
    while len(arrows) < num_arrows:
        extra = _sample_arrow(shape, blocked)
        if extra is None:
            return None
        arrows.append(extra)
        blocked |= _halo(extra["cells"], shape)

    if len({tuple(arrow["direction"]) for arrow in arrows}) == 1 and len(arrows) > 2:
        return None

    colors = random.sample(PALETTE, len(arrows))
    grid = _render_grid(shape, arrows, colors)
    overlap_info = _overlap_modes(shape, arrows)
    if overlap_info is None or not overlap_info[mode]:
        return None

    out = main(grid)
    if np.count_nonzero(out != BG) > int(0.72 * h * w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_scene(random.choice(("tie", "longer")))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-propagation input")
