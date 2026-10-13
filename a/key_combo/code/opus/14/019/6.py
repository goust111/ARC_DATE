from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
NON_RED_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
CORNER_TO_STEP = {
    (0, 0): (-1, -1),
    (0, 1): (-1, 1),
    (1, 0): (1, -1),
    (1, 1): (1, 1),
}
GEN_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ray_positions(
    top_left: tuple[int, int], step: tuple[int, int], h: int, draw_w: int
) -> list[tuple[int, int]]:
    r0, c0 = top_left
    dr, dc = step
    out = [(int(r0), int(c0))]
    rr = r0 + dr
    cc = c0 + dc
    while 0 <= rr and rr + 1 < h and 0 <= cc and cc + 1 < draw_w:
        out.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return out


def _stamp_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)}


def _footprint(top_left: tuple[int, int], step: tuple[int, int], h: int, draw_w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for pos in _ray_positions(top_left, step, h, draw_w):
        cells |= _stamp_cells(pos)
    return cells


def _palette_info(grid: np.ndarray) -> tuple[list[int], list[int]]:
    palette_col = grid[:, -1]
    rows = [int(r) for r in np.flatnonzero(palette_col != BLACK)]
    colors = [int(palette_col[r]) for r in rows]
    assert 3 <= len(colors) <= 4
    assert len(colors) == len(set(colors))
    assert all(color != RED for color in colors)
    return rows, colors


def _extract_seeds(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    draw_w = w - 1
    body = grid[:, :draw_w]
    comps = _components4(body != BLACK)
    seeds: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 == 1 and c1 - c0 == 1
        patch = body[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (2, 2)
        assert np.all(patch != BLACK)
        red_locs = np.argwhere(patch == RED)
        assert red_locs.shape == (1, 2)
        non_red = {int(v) for v in np.unique(patch)} - {RED}
        assert len(non_red) == 1
        color = non_red.pop()
        assert int(np.count_nonzero(patch == color)) == 3
        corner = tuple(map(int, red_locs[0]))
        step = CORNER_TO_STEP[corner]
        positions = _ray_positions((r0, c0), step, h, draw_w)
        assert len(positions) >= 2
        seed_cells = _stamp_cells((r0, c0))
        for pos in positions[1:]:
            for cell in _stamp_cells(pos):
                if grid[cell] != BLACK:
                    assert cell in seed_cells
        seeds.append(
            {
                "top_left": (int(r0), int(c0)),
                "color": int(color),
                "corner": corner,
                "step": step,
                "positions": positions,
                "length": len(positions) - 1,
                "footprint": _footprint((r0, c0), step, h, draw_w),
            }
        )
    assert 3 <= len(seeds) <= 4
    lengths = [seed["length"] for seed in seeds]
    assert len(lengths) == len(set(lengths))
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            assert not (seeds[i]["footprint"] & seeds[j]["footprint"])
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each 2x2 seed diagonally, sort the resulting rays by length, and recolor them by the right-edge palette order.

    Input structure:
    - The grid is black except for 3-4 isolated 2x2 seed blocks and a palette in the rightmost column.
    - Each seed block contains exactly one red corner pixel and three pixels of one non-red color.
    - The red corner position encodes one diagonal direction, and the rightmost column contains one distinct palette pixel per seed.

    Rule:
    - Replace each seed's red corner with the seed's non-red color conceptually, then extend that 2x2 block one row and one column at a time along the encoded diagonal.
    - A ray's length is the number of extra 2x2 stamps that fit before entering the grid boundary or the palette column.
    - Sort the rays by decreasing length, recolor the longest ray with the top palette color, the next ray with the next palette color, and so on.
    - The output keeps the same shape but clears the entire palette column to black.

    Key insight:
    - Final colors come only from the global ranking of ray lengths; the original seed colors merely identify each ray before that ranking is computed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30 and w >= 4

    _palette_rows, palette_colors = _palette_info(g)
    seeds = _extract_seeds(g)
    assert len(seeds) == len(palette_colors)

    out = np.zeros_like(g)
    order = sorted(range(len(seeds)), key=lambda i: seeds[i]["length"], reverse=True)
    for rank, seed_idx in enumerate(order):
        paint = palette_colors[rank]
        for r, c in seeds[seed_idx]["footprint"]:
            out[r, c] = paint
    return out


def _seed_order(seed: dict) -> tuple[int, int]:
    return seed["top_left"]


def _seed_gap_ok(a: dict, b: dict) -> bool:
    ar, ac = a["top_left"]
    br, bc = b["top_left"]
    dr = max(0, max(ar - (br + 1), br - (ar + 1)))
    dc = max(0, max(ac - (bc + 1), bc - (ac + 1)))
    return max(dr, dc) >= 4


def _corner_candidate_positions(
    h: int, draw_w: int, corner: tuple[int, int]
) -> list[dict]:
    candidates: list[dict] = []
    step = CORNER_TO_STEP[corner]
    for r0 in range(1, h - 2):
        for c0 in range(1, draw_w - 2):
            positions = _ray_positions((r0, c0), step, h, draw_w)
            length = len(positions) - 1
            if length < 2:
                continue
            candidates.append(
                {
                    "top_left": (int(r0), int(c0)),
                    "corner": corner,
                    "step": step,
                    "positions": positions,
                    "length": int(length),
                    "footprint": _footprint((r0, c0), step, h, draw_w),
                }
            )
    random.shuffle(candidates)
    return candidates


def _heuristic_orders(seeds: list[dict]) -> list[tuple[int, ...]]:
    idxs = list(range(len(seeds)))
    heuristics = [
        tuple(sorted(idxs, key=lambda i: seeds[i]["top_left"][0])),
        tuple(sorted(idxs, key=lambda i: seeds[i]["top_left"][0], reverse=True)),
        tuple(sorted(idxs, key=lambda i: seeds[i]["top_left"][1])),
        tuple(sorted(idxs, key=lambda i: seeds[i]["top_left"][1], reverse=True)),
        tuple(sorted(idxs, key=lambda i: seeds[i]["top_left"])),
        tuple(sorted(idxs, key=lambda i: seeds[i]["top_left"], reverse=True)),
        tuple(sorted(idxs, key=lambda i: sum(seeds[i]["top_left"]))),
        tuple(sorted(idxs, key=lambda i: sum(seeds[i]["top_left"]), reverse=True)),
        tuple(sorted(idxs, key=lambda i: seeds[i]["corner"])),
    ]
    return heuristics


def _expected_output(h: int, w: int, seeds: list[dict], palette_colors: list[int]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    order = sorted(range(len(seeds)), key=lambda i: seeds[i]["length"], reverse=True)
    for rank, seed_idx in enumerate(order):
        for r, c in seeds[seed_idx]["footprint"]:
            out[r, c] = palette_colors[rank]
    return out


def _build_input(
    h: int, w: int, seeds: list[dict], base_colors: list[int], palette_rows: list[int], palette_colors: list[int]
) -> np.ndarray:
    inp = np.zeros((h, w), dtype=int)
    for seed, color in zip(seeds, base_colors):
        r0, c0 = seed["top_left"]
        inp[r0 : r0 + 2, c0 : c0 + 2] = color
        rr, cc = seed["corner"]
        inp[r0 + rr, c0 + cc] = RED
    for r, color in zip(palette_rows, palette_colors):
        inp[r, w - 1] = color
    return inp


def _choose_seed_layout(h: int, w: int, corners: list[tuple[int, int]]) -> list[dict] | None:
    draw_w = w - 1
    candidate_map = {corner: _corner_candidate_positions(h, draw_w, corner) for corner in corners}
    if any(not candidate_map[corner] for corner in corners):
        return None

    for _ in range(300):
        chosen: list[dict] = []
        used_lengths: set[int] = set()
        used_cells: set[tuple[int, int]] = set()
        for corner in random.sample(corners, len(corners)):
            feasible = []
            for cand in candidate_map[corner]:
                if cand["length"] in used_lengths:
                    continue
                if cand["footprint"] & used_cells:
                    continue
                if not all(_seed_gap_ok(cand, prev) for prev in chosen):
                    continue
                feasible.append(cand)
            if not feasible:
                break
            picked = random.choice(feasible)
            chosen.append(picked)
            used_lengths.add(picked["length"])
            used_cells |= picked["footprint"]
        if len(chosen) != len(corners):
            continue
        if max(seed["length"] for seed in chosen) - min(seed["length"] for seed in chosen) < 2:
            continue
        true_order = tuple(sorted(range(len(chosen)), key=lambda i: chosen[i]["length"], reverse=True))
        if any(order == true_order for order in _heuristic_orders(chosen)):
            continue
        return chosen
    return None


def _validate_generated(inp: np.ndarray, expected: np.ndarray, seeds: list[dict], palette_colors: list[int]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.any(expected[:, -1] != BLACK):
        return False
    parsed = _extract_seeds(inp)
    if len(parsed) != len(seeds):
        return False
    order = tuple(sorted(range(len(parsed)), key=lambda i: parsed[i]["length"], reverse=True))
    if any(heur == order for heur in _heuristic_orders(parsed)):
        return False
    wrong_orders = [heur for heur in _heuristic_orders(parsed) if heur != order]
    if not wrong_orders:
        return False
    for wrong in wrong_orders[:3]:
        wrong_out = np.zeros_like(expected)
        for rank, seed_idx in enumerate(wrong):
            for r, c in parsed[seed_idx]["footprint"]:
                wrong_out[r, c] = palette_colors[rank]
        if np.array_equal(wrong_out, expected):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    n = 4 if random.random() < 0.6 else 3
    corners = random.sample(list(CORNER_TO_STEP), n)
    seeds = _choose_seed_layout(h, w, corners)
    if seeds is None:
        return None

    seed_colors = random.sample(NON_RED_COLORS, n)
    remaining = [color for color in NON_RED_COLORS if color not in seed_colors]
    palette_colors = random.sample(remaining, n)
    palette_rows = sorted(random.sample(range(h), n))
    inp = _build_input(h, w, seeds, seed_colors, palette_rows, palette_colors)
    expected = _expected_output(h, w, seeds, palette_colors)
    if not _validate_generated(inp, expected, seeds, palette_colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid diagonal-ray sorting puzzle")
