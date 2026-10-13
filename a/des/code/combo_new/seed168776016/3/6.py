from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0

GRID_H_RANGE = (17, 30)
GRID_W_RANGE = (17, 30)
LAYER_RANGE = (2, 4)
INNER_SIZE_MAX = 7
MAX_EXTRA_SEEDS = 2
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
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


def _rect_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _shell_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = [(r0, c) for c in range(c0, c1 + 1)]
    if r1 > r0:
        pts.extend((r1, c) for c in range(c0, c1 + 1))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 > c0:
            pts.append((r, c1))
    return pts


def _uniform_shell_color(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> int | None:
    color = None
    for r, c in _shell_coords(r0, c0, r1, c1):
        value = int(grid[r, c])
        if value == BG:
            return None
        if color is None:
            color = value
        elif value != color:
            return None
    return color


def _extract_structure(grid: np.ndarray) -> dict[str, object]:
    h, w = grid.shape
    best = None
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0

    for comp in _components(grid == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        inner_box = _bbox(comp)
        if len(comp) != _rect_area(inner_box):
            continue

        inner_to_outer: list[int] = []
        for depth in range(1, max(h, w)):
            r0 = inner_box[0] - depth
            c0 = inner_box[1] - depth
            r1 = inner_box[2] + depth
            c1 = inner_box[3] + depth
            if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                break
            color = _uniform_shell_color(grid, r0, c0, r1, c1)
            if color is None:
                break
            inner_to_outer.append(color)

        if not (2 <= len(inner_to_outer) <= 4):
            continue
        if len(set(inner_to_outer)) != len(inner_to_outer):
            continue

        k = len(inner_to_outer)
        outer_box = (
            inner_box[0] - k,
            inner_box[1] - k,
            inner_box[2] + k,
            inner_box[3] + k,
        )
        dist2 = (
            ((outer_box[0] + outer_box[2]) / 2.0 - center_r) ** 2
            + ((outer_box[1] + outer_box[3]) / 2.0 - center_c) ** 2
        )
        candidate = {
            "inner_box": inner_box,
            "outer_box": outer_box,
            "outer_to_inner": inner_to_outer[::-1],
            "layers": k,
            "score": (_rect_area(outer_box), k, -dist2),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        raise ValueError("failed to locate the concentric frame structure")
    return best


def _square_cells(
    h: int, w: int, row: int, col: int, layer: int, box: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for rr in range(max(0, row - layer), min(h - 1, row + layer) + 1):
        for cc in range(max(0, col - layer), min(w - 1, col + layer) + 1):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            cells.add((rr, cc))
    return cells


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    info = _extract_structure(g)
    outer_box = info["outer_box"]
    outer_to_inner = list(info["outer_to_inner"])
    layer_by_color = {color: idx + 1 for idx, color in enumerate(outer_to_inner)}

    out = g.copy()
    seeds_by_layer: dict[int, list[tuple[int, int, int]]] = {layer: [] for layer in range(1, len(outer_to_inner) + 1)}
    r0, c0, r1, c1 = outer_box
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            color = int(g[r, c])
            if color in layer_by_color:
                seeds_by_layer[layer_by_color[color]].append((r, c, color))

    for layer in range(1, len(outer_to_inner) + 1):
        for r, c, color in seeds_by_layer[layer]:
            for rr in range(max(0, r - layer), min(h - 1, r + layer) + 1):
                for cc in range(max(0, c - layer), min(w - 1, c + layer) + 1):
                    if r0 <= rr <= r1 and c0 <= cc <= c1:
                        continue
                    out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand exterior seed pixels into layer-sized solid squares while leaving a concentric frame board unchanged.

    Input structure:
    - Background is black.
    - A central object is a stack of 2 to 4 concentric 1-pixel rectangular frames with distinct colors and a black rectangular hole in the middle.
    - Outside that rectangle, single-pixel seeds appear in the black field, and every seed color matches one of the frame layers.

    Rule:
    - Read the frame colors from the outer border inward to assign layer numbers 1..K.
    - A seed with layer number L expands to a solid square of side length 2L+1 centered on that seed.
    - Draw lower layer numbers first and higher layer numbers afterward, clip at the grid edge, and never modify the concentric-board rectangle.

    Key insight:
    - Square size is determined by the seed color's geometric depth in the nested frames, so the visually smaller inner-frame colors produce the largest expansions.
    """
    return _solve(np.asarray(grid, dtype=int))


def _paint_shell(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_structure(
    grid: np.ndarray,
    outer_r0: int,
    outer_c0: int,
    inner_h: int,
    inner_w: int,
    colors: list[int],
) -> tuple[int, int, int, int]:
    k = len(colors)
    outer_h = inner_h + 2 * k
    outer_w = inner_w + 2 * k
    outer_r1 = outer_r0 + outer_h - 1
    outer_c1 = outer_c0 + outer_w - 1
    for depth, color in enumerate(colors):
        _paint_shell(
            grid,
            outer_r0 + depth,
            outer_c0 + depth,
            outer_r1 - depth,
            outer_c1 - depth,
            color,
        )
    return outer_r0, outer_c0, outer_r1, outer_c1


def _footprints_overlap(
    a: set[tuple[int, int]], b: set[tuple[int, int]]
) -> bool:
    return not a.isdisjoint(b)


def _chebyshev_far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1


def _pick_overlap_pair(specs: list[dict[str, object]], rng: random.Random) -> tuple[dict[str, object], dict[str, object]] | None:
    choices = [
        (specs[i], specs[j])
        for i in range(len(specs))
        for j in range(len(specs))
        if i != j and specs[i]["layer"] != specs[j]["layer"]
    ]
    if not choices:
        return None
    return rng.choice(choices)


def _place_seeds(
    h: int,
    w: int,
    box: tuple[int, int, int, int],
    layers: int,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    specs = [{"layer": layer} for layer in range(1, layers + 1)]
    for _ in range(rng.randint(0, MAX_EXTRA_SEEDS)):
        specs.append({"layer": rng.randint(1, layers)})
    rng.shuffle(specs)

    edge_spec = rng.choice(specs) if rng.random() < 0.5 else None
    overlap_pair = _pick_overlap_pair(specs, rng) if len(specs) >= 2 and rng.random() < 0.5 else None

    if overlap_pair is not None:
        anchor, partner = overlap_pair
        pair_ids = {id(anchor), id(partner)}
        order = [anchor, partner] + [spec for spec in specs if id(spec) not in pair_ids]
        partner["overlap_anchor"] = anchor
    else:
        order = list(specs)
        rng.shuffle(order)

    if edge_spec is not None:
        edge_spec["force_edge"] = True

    placed: list[tuple[int, int, int]] = []
    footprints: list[set[tuple[int, int]]] = []
    spec_to_footprint: dict[int, set[tuple[int, int]]] = {}

    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (box[0] <= r <= box[2] and box[1] <= c <= box[3])
    ]

    for spec in order:
        layer = int(spec["layer"])
        rng.shuffle(candidates)
        chosen = None
        chosen_cells = None
        for r, c in candidates:
            if any((r, c) == (pr, pc) for pr, pc, _ in placed):
                continue
            if any(not _chebyshev_far_enough((r, c), (pr, pc)) for pr, pc, _ in placed):
                continue

            cells = _square_cells(h, w, r, c, layer, box)
            if not cells:
                continue

            if spec.get("force_edge") and not (r < layer or r + layer >= h or c < layer or c + layer >= w):
                continue

            anchor = spec.get("overlap_anchor")
            if anchor is not None:
                anchor_cells = spec_to_footprint.get(id(anchor))
                if anchor_cells is None:
                    continue
                if not _footprints_overlap(cells, anchor_cells):
                    continue
                if any(
                    other is not anchor_cells and _footprints_overlap(cells, other)
                    for other in footprints
                ):
                    continue
            else:
                if any(_footprints_overlap(cells, other) for other in footprints):
                    continue

            chosen = (r, c, layer)
            chosen_cells = cells
            break

        if chosen is None or chosen_cells is None:
            raise ValueError("failed to place seeds")

        placed.append(chosen)
        footprints.append(chosen_cells)
        spec_to_footprint[id(spec)] = chosen_cells

    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    layers = rng.randint(*LAYER_RANGE)

    max_inner_h = min(INNER_SIZE_MAX, h - 2 * layers - 8)
    max_inner_w = min(INNER_SIZE_MAX, w - 2 * layers - 8)
    if max_inner_h < 1 or max_inner_w < 1:
        return None

    inner_h = rng.randint(1, max_inner_h)
    inner_w = rng.randint(1, max_inner_w)
    outer_h = inner_h + 2 * layers
    outer_w = inner_w + 2 * layers

    base_r = (h - outer_h) // 2
    base_c = (w - outer_w) // 2
    r_shift = min(2, base_r - 1, h - outer_h - 1 - base_r)
    c_shift = min(2, base_c - 1, w - outer_w - 1 - base_c)
    outer_r0 = base_r + rng.randint(-max(0, r_shift), max(0, r_shift))
    outer_c0 = base_c + rng.randint(-max(0, c_shift), max(0, c_shift))

    grid = np.zeros((h, w), dtype=int)
    colors = rng.sample(list(range(1, 10)), layers)
    box = _paint_structure(grid, outer_r0, outer_c0, inner_h, inner_w, colors)

    try:
        seeds = _place_seeds(h, w, box, layers, rng)
    except ValueError:
        return None

    for r, c, layer in seeds:
        grid[r, c] = colors[layer - 1]

    try:
        parsed = _extract_structure(grid)
    except ValueError:
        return None
    if tuple(parsed["outer_box"]) != box:
        return None

    out = _solve(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
