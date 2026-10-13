from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8], dtype=int)

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400
PLACEMENT_RETRIES = 60
INPUT_PAD = 1
OUTPUT_PAD = 1


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_solid_square(g: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    size_r = r1 - r0 + 1
    size_c = c1 - c0 + 1
    if size_r != size_c or size_r not in (2, 3):
        return False
    if len(coords) != size_r * size_c:
        return False
    return bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color))


def _edge_cells(bbox: tuple[int, int, int, int], move: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    dr, dc = move
    if (dr, dc) == (0, 1):
        return [(r, c0) for r in range(r0, r1 + 1)]
    if (dr, dc) == (0, -1):
        return [(r, c1) for r in range(r0, r1 + 1)]
    if (dr, dc) == (1, 0):
        return [(r0, c) for c in range(c0, c1 + 1)]
    return [(r1, c) for c in range(c0, c1 + 1)]


def _perp(move: tuple[int, int]) -> tuple[int, int]:
    dr, dc = move
    if dr == 0:
        return (1, 0)
    return (0, 1)


def _available_forward(
    bbox: tuple[int, int, int, int], move: tuple[int, int], h: int, w: int
) -> int:
    r0, c0, r1, c1 = bbox
    dr, dc = move
    if (dr, dc) == (0, 1):
        return w - 1 - c1
    if (dr, dc) == (0, -1):
        return c0
    if (dr, dc) == (1, 0):
        return h - 1 - r1
    return r0


def _find_arrow(
    g: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = g.shape
    matches: list[tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]] = []

    for move in DIRS:
        dr, dc = move
        pr, pc = _perp(move)
        for edge_r, edge_c in _edge_cells(bbox, move):
            tip = (edge_r - dr, edge_c - dc)
            back = (tip[0] - dr, tip[1] - dc)
            wing_a = (back[0] + pr, back[1] + pc)
            wing_b = (back[0] - pr, back[1] - pc)
            if not all(_in_bounds(h, w, rr, cc) for rr, cc in (tip, wing_a, wing_b)):
                continue
            if int(g[tip]) != GREY or int(g[wing_a]) != GREY or int(g[wing_b]) != GREY:
                continue

            stripe: list[tuple[int, int]] = []
            cur = back
            while _in_bounds(h, w, cur[0], cur[1]) and int(g[cur]) == MAROON:
                stripe.append(cur)
                cur = (cur[0] - dr, cur[1] - dc)
            if not stripe:
                continue
            matches.append((move, [tip, wing_a, wing_b], stripe))

    if len(matches) != 1:
        return None
    return matches[0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BG)

    for color in sorted(int(v) for v in np.unique(g)):
        if color in (BG, GREY, MAROON):
            continue
        for comp in _components(g == color):
            if not _is_solid_square(g, color, comp):
                for r, c in comp:
                    out[r, c] = color
                continue

            bbox = _bbox(comp)
            marker = _find_arrow(g, bbox)
            if marker is None:
                for r, c in comp:
                    out[r, c] = color
                continue

            move, _arrow_cells, stripe = marker
            dist = len(stripe)
            shift = min(dist, _available_forward(bbox, move, h, w))
            dr, dc = move
            for r, c in comp:
                out[r + shift * dr, c + shift * dc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each solid square away from its adjacent arrow by the maroon stripe length.

    The grey V-shaped arrow points into the block, while the block moves through
    itself and away from the arrow. The maroon stripe behind the arrow encodes the
    push distance. Arrows and stripes disappear after the move, and pushes clip at
    the grid boundary.
    """
    return _solve(grid)


def _anchor_cell(
    r0: int, c0: int, size: int, move: tuple[int, int], anchor_idx: int
) -> tuple[int, int]:
    dr, dc = move
    if (dr, dc) == (0, 1):
        return r0 + anchor_idx, c0
    if (dr, dc) == (0, -1):
        return r0 + anchor_idx, c0 + size - 1
    if (dr, dc) == (1, 0):
        return r0, c0 + anchor_idx
    return r0 + size - 1, c0 + anchor_idx


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - pad), min(h, r + pad + 1)):
            for nc in range(max(0, c - pad), min(w, c + pad + 1)):
                expanded.add((nr, nc))
    return expanded


def _build_spec(
    h: int,
    w: int,
    size: int,
    move: tuple[int, int],
    dist: int,
    r0: int,
    c0: int,
    anchor_idx: int,
    color: int,
) -> dict[str, object] | None:
    dr, dc = move
    block = {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}
    anchor = _anchor_cell(r0, c0, size, move, anchor_idx)
    tip = (anchor[0] - dr, anchor[1] - dc)
    back = (tip[0] - dr, tip[1] - dc)
    pr, pc = _perp(move)
    arrow = {tip, (back[0] + pr, back[1] + pc), (back[0] - pr, back[1] - pc)}
    stripe = {(tip[0] - step * dr, tip[1] - step * dc) for step in range(1, dist + 1)}
    input_cells = block | arrow | stripe

    if not all(_in_bounds(h, w, rr, cc) for rr, cc in input_cells):
        return None
    if len(input_cells) != len(block) + len(arrow) + len(stripe):
        return None

    bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
    shift = min(dist, _available_forward(bbox, move, h, w))
    moved = {(r + shift * dr, c + shift * dc) for r, c in block}
    if not all(_in_bounds(h, w, rr, cc) for rr, cc in moved):
        return None

    return {
        "color": color,
        "size": size,
        "move": move,
        "dist": dist,
        "shift": shift,
        "block": block,
        "arrow": arrow,
        "stripe": stripe,
        "output": moved,
    }


def _paint_scene(h: int, w: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["block"]:
            grid[r, c] = color
        for r, c in spec["arrow"]:
            grid[r, c] = GREY
        for r, c in spec["stripe"]:
            grid[r, c] = MAROON
    return grid


def _expected_output(h: int, w: int, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["output"]:
            out[r, c] = color
    return out


def _target_count(rng: np.random.Generator) -> int:
    return int(rng.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.3, 0.2]))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 19))
    w = int(rng.integers(10, 19))
    target_n = min(_target_count(rng), len(PALETTE))
    colors = rng.permutation(PALETTE)[:target_n].tolist()
    specs: list[dict[str, object]] = []
    input_reserved: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()
    must_have_clip = bool(rng.random() < 0.45)
    has_clip = False

    for obj_idx, color in enumerate(colors):
        placed = False
        force_clip = must_have_clip and not has_clip and obj_idx == len(colors) - 1

        for _ in range(PLACEMENT_RETRIES):
            size = int(rng.choice([2, 3]))
            move = DIRS[int(rng.integers(0, len(DIRS)))]
            dist = int(rng.integers(1, 5))
            want_clip = force_clip or bool(rng.random() < 0.3)
            candidates: list[dict[str, object]] = []

            for r0 in range(h - size + 1):
                for c0 in range(w - size + 1):
                    for anchor_idx in range(size):
                        spec = _build_spec(h, w, size, move, dist, r0, c0, anchor_idx, color)
                        if spec is None:
                            continue
                        clipped = int(spec["shift"]) < dist
                        if want_clip and not clipped:
                            continue
                        if force_clip and not clipped:
                            continue

                        input_cells = _expand_cells(set(spec["block"]) | set(spec["arrow"]) | set(spec["stripe"]), h, w, INPUT_PAD)
                        output_cells = _expand_cells(set(spec["output"]), h, w, OUTPUT_PAD)
                        if input_cells & input_reserved:
                            continue
                        if output_cells & output_reserved:
                            continue
                        candidates.append(spec)

            if not candidates:
                continue

            spec = candidates[int(rng.integers(0, len(candidates)))]
            specs.append(spec)
            input_reserved |= _expand_cells(set(spec["block"]) | set(spec["arrow"]) | set(spec["stripe"]), h, w, INPUT_PAD)
            output_reserved |= _expand_cells(set(spec["output"]), h, w, OUTPUT_PAD)
            has_clip = has_clip or (int(spec["shift"]) < int(spec["dist"]))
            placed = True
            break

        if not placed:
            return None

    if must_have_clip and not has_clip:
        return None

    grid = _paint_scene(h, w, specs)
    expected = _expected_output(h, w, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
