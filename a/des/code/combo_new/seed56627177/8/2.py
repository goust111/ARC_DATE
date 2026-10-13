from __future__ import annotations

from functools import lru_cache

import numpy as np


BG = 0
YELLOW = 4
COLOR_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
MAX_SIDE = 30
MAX_GEN_ATTEMPTS = 400
SCENE_PAD = 2

DIR_VECTORS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_ARROW_SPECS = {
    "right": {
        "shape": ((1, 0), (1, 1), (1, 2), (0, 3), (2, 3)),
        "tail": (1, 0),
        "head": (1, 3),
    },
    "left": {
        "shape": ((0, 0), (1, 1), (1, 2), (1, 3), (2, 0)),
        "tail": (1, 3),
        "head": (1, 0),
    },
    "down": {
        "shape": ((0, 1), (1, 1), (2, 1), (3, 0), (3, 2)),
        "tail": (0, 1),
        "head": (3, 1),
    },
    "up": {
        "shape": ((0, 0), (0, 2), (1, 1), (2, 1), (3, 1)),
        "tail": (3, 1),
        "head": (0, 1),
    },
}


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    norm = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(norm)


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    shifted = coords.copy()
    shifted[:, 0] += int(dr)
    shifted[:, 1] += int(dc)
    return shifted


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _proj_values(coords: np.ndarray, vector: tuple[int, int]) -> np.ndarray:
    vr, vc = vector
    return coords[:, 0] * int(vr) + coords[:, 1] * int(vc)


def _min_manhattan(coords: np.ndarray, point: tuple[int, int]) -> int:
    pr, pc = point
    dists = np.abs(coords[:, 0] - int(pr)) + np.abs(coords[:, 1] - int(pc))
    return int(dists.min())


def _parse_arrow(coords: np.ndarray) -> dict[str, object]:
    norm = _normalize_shape(coords)
    for direction, spec in _ARROW_SPECS.items():
        if norm != spec["shape"]:
            continue
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        tail_off = spec["tail"]
        head_off = spec["head"]
        return {
            "direction": direction,
            "vector": DIR_VECTORS[direction],
            "coords": coords,
            "tail": (r0 + int(tail_off[0]), c0 + int(tail_off[1])),
            "head": (r0 + int(head_off[0]), c0 + int(head_off[1])),
        }
    raise AssertionError("yellow component is not a valid arrow")


def _slide_back(
    fixed: np.ndarray,
    mover: np.ndarray,
    move_vec: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    fixed_cells = _coord_set(fixed)
    cur = mover.copy()
    dr, dc = move_vec
    while True:
        nxt = _translate(cur, dr, dc)
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= h) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= w):
            break
        if any((int(r), int(c)) in fixed_cells for r, c in nxt):
            break
        cur = nxt

    assert any((int(r + dr), int(c + dc)) in fixed_cells for r, c in cur), "moved fragment must end adjacent"
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each same-color fragment pair by reading the yellow arrow between them.

    Input structure:
    - The background is black.
    - Each non-background, non-yellow color appears in exactly two separated 4-connected fragments.
    - A single yellow arrow sits in the blank space between the two fragments of each color.
    - Every arrow is an 8-connected five-pixel motif: a length-3 horizontal or vertical shaft plus a two-pixel head pointing up, down, left, or right.

    Rule:
    - Find every yellow arrow and determine its direction, tail side, and head side.
    - Match each arrow to the unique same-color fragment pair whose fixed fragment is nearer the tail and whose moving fragment is nearer the head.
    - Remove the yellow arrow, then translate the head-side fragment back along the arrow axis until one more step would overlap the tail-side fragment.
    - Leave the tail-side fragment in place and return the full scene on the original grid size.

    Key insight:
    - The arrow does two jobs at once: its orientation gives the assembly axis, and its head-versus-tail asymmetry reveals which fragment is the one that must move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, YELLOW))
    fragments_by_color: dict[int, list[np.ndarray]] = {}
    for color in colors:
        comps = _components(g == color, connectivity=4)
        assert len(comps) == 2, "each non-background color must form exactly two fragments"
        fragments_by_color[color] = comps

    arrow_comps = _components(g == YELLOW, connectivity=8)
    arrows = [_parse_arrow(comp) for comp in arrow_comps]
    arrows.sort(key=lambda a: (int(a["tail"][0]), int(a["tail"][1])))
    assert len(arrows) == len(colors), "each color pair must have exactly one arrow"

    color_list = list(colors)
    pair_costs: list[list[tuple[int, int, int]]] = []
    for arrow in arrows:
        vector = arrow["vector"]
        tail_proj = int(arrow["tail"][0] * vector[0] + arrow["tail"][1] * vector[1])
        head_proj = int(arrow["head"][0] * vector[0] + arrow["head"][1] * vector[1])
        row: list[tuple[int, int, int]] = []
        for color in color_list:
            comps = fragments_by_color[color]
            best: tuple[int, int, int] | None = None
            for fixed_idx, mover_idx in ((0, 1), (1, 0)):
                fixed = comps[fixed_idx]
                mover = comps[mover_idx]
                fixed_max = int(_proj_values(fixed, vector).max())
                mover_min = int(_proj_values(mover, vector).min())
                penalty = 0
                if fixed_max >= tail_proj:
                    penalty += 100
                if mover_min <= head_proj:
                    penalty += 100
                cost = (
                    _min_manhattan(fixed, arrow["tail"])
                    + _min_manhattan(mover, arrow["head"])
                    + penalty
                )
                candidate = (int(cost), int(fixed_idx), int(mover_idx))
                if best is None or candidate < best:
                    best = candidate
            assert best is not None
            row.append(best)
        pair_costs.append(row)

    @lru_cache(maxsize=None)
    def _assign(i: int, used_mask: int) -> int:
        if i == len(arrows):
            return 0
        best = 10**9
        for j in range(len(color_list)):
            if used_mask & (1 << j):
                continue
            score = pair_costs[i][j][0] + _assign(i + 1, used_mask | (1 << j))
            if score < best:
                best = score
        return best

    assigned: list[int] = []
    used = 0
    for i in range(len(arrows)):
        best_j = None
        best_score = None
        for j in range(len(color_list)):
            if used & (1 << j):
                continue
            score = pair_costs[i][j][0] + _assign(i + 1, used | (1 << j))
            if best_score is None or score < best_score:
                best_score = score
                best_j = j
        assert best_j is not None
        assigned.append(best_j)
        used |= 1 << best_j

    out = np.full_like(g, BG)
    final_components: list[tuple[int, np.ndarray]] = []
    for arrow_idx, color_idx in enumerate(assigned):
        arrow = arrows[arrow_idx]
        color = color_list[color_idx]
        fixed_idx, mover_idx = pair_costs[arrow_idx][color_idx][1:]
        fixed = fragments_by_color[color][fixed_idx]
        mover = fragments_by_color[color][mover_idx]
        vr, vc = arrow["vector"]
        moved = _slide_back(fixed, mover, (-int(vr), -int(vc)), (h, w))
        final_components.append((color, fixed))
        final_components.append((color, moved))

    occupied: set[tuple[int, int]] = set()
    for color, coords in final_components:
        for r, c in coords:
            cell = (int(r), int(c))
            assert cell not in occupied, "reassembled fragments must not overlap"
            occupied.add(cell)
            out[cell] = int(color)
    return out


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, 4):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _random_polyomino(rng: np.random.Generator, height: int, width: int, area: int) -> np.ndarray:
    start = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    cells = {start}
    frontier = set(_neighbors(start[0], start[1], 4))
    frontier = {(r, c) for r, c in frontier if 0 <= r < height and 0 <= c < width}
    while len(cells) < area:
        assert frontier, "frontier exhausted while growing polyomino"
        idx = int(rng.integers(0, len(frontier)))
        r, c = list(frontier)[idx]
        frontier.remove((r, c))
        cells.add((r, c))
        for nr, nc in _neighbors(r, c, 4):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    coords = np.asarray(sorted(cells), dtype=int)
    return coords


def _split_shape(
    coords: np.ndarray,
    direction: str,
) -> list[dict[str, object]]:
    cells = _coord_set(coords)
    options: list[dict[str, object]] = []
    if direction in ("left", "right"):
        min_c = int(coords[:, 1].min())
        max_c = int(coords[:, 1].max())
        for cut in range(min_c, max_c):
            left = {cell for cell in cells if cell[1] <= cut}
            right = {cell for cell in cells if cell[1] > cut}
            if min(len(left), len(right)) < 2:
                continue
            if not _is_connected(left) or not _is_connected(right):
                continue
            contacts = [r for r in range(int(coords[:, 0].min()), int(coords[:, 0].max()) + 1) if (r, cut) in left and (r, cut + 1) in right]
            if not contacts:
                continue
            options.append(
                {
                    "cut": cut,
                    "left": np.asarray(sorted(left), dtype=int),
                    "right": np.asarray(sorted(right), dtype=int),
                    "contacts": contacts,
                }
            )
    else:
        min_r = int(coords[:, 0].min())
        max_r = int(coords[:, 0].max())
        for cut in range(min_r, max_r):
            top = {cell for cell in cells if cell[0] <= cut}
            bottom = {cell for cell in cells if cell[0] > cut}
            if min(len(top), len(bottom)) < 2:
                continue
            if not _is_connected(top) or not _is_connected(bottom):
                continue
            contacts = [c for c in range(int(coords[:, 1].min()), int(coords[:, 1].max()) + 1) if (cut, c) in top and (cut + 1, c) in bottom]
            if not contacts:
                continue
            options.append(
                {
                    "cut": cut,
                    "top": np.asarray(sorted(top), dtype=int),
                    "bottom": np.asarray(sorted(bottom), dtype=int),
                    "contacts": contacts,
                }
            )
    return options


def _arrow_cells(direction: str, tail_center: tuple[int, int]) -> np.ndarray:
    tr, tc = tail_center
    rel = np.asarray(_ARROW_SPECS[direction]["shape"], dtype=int)
    tail_off = np.asarray(_ARROW_SPECS[direction]["tail"], dtype=int)
    shift = np.asarray((int(tr), int(tc)), dtype=int) - tail_off
    return rel + shift


def _build_pair_scene(rng: np.random.Generator, color: int, direction: str) -> dict[str, object]:
    for _ in range(200):
        if direction in ("left", "right"):
            height = int(rng.integers(2, 6))
            width = int(rng.integers(4, 7))
        else:
            height = int(rng.integers(4, 7))
            width = int(rng.integers(2, 6))
        max_area = min(height * width, 11)
        area = int(rng.integers(6, max_area + 1))
        whole = _random_polyomino(rng, height, width, area)
        splits = _split_shape(whole, direction)
        if not splits:
            continue

        split = splits[int(rng.integers(0, len(splits)))]
        vr, vc = DIR_VECTORS[direction]
        tail_gap = int(rng.integers(1, 3))
        head_gap = int(rng.integers(1, 4))
        shift = tail_gap + 4 + head_gap

        if direction == "right":
            fixed = split["left"]
            mover = split["right"]
            contact_r = int(split["contacts"][int(rng.integers(0, len(split["contacts"])))])
            fixed_boundary = (contact_r, int(split["cut"]))
        elif direction == "left":
            fixed = split["right"]
            mover = split["left"]
            contact_r = int(split["contacts"][int(rng.integers(0, len(split["contacts"])))])
            fixed_boundary = (contact_r, int(split["cut"]) + 1)
        elif direction == "down":
            fixed = split["top"]
            mover = split["bottom"]
            contact_c = int(split["contacts"][int(rng.integers(0, len(split["contacts"])))])
            fixed_boundary = (int(split["cut"]), contact_c)
        else:
            fixed = split["bottom"]
            mover = split["top"]
            contact_c = int(split["contacts"][int(rng.integers(0, len(split["contacts"])))])
            fixed_boundary = (int(split["cut"]) + 1, contact_c)

        mover_input = _translate(mover, shift * vr, shift * vc)
        arrow_tail = (
            int(fixed_boundary[0] + (tail_gap + 1) * vr),
            int(fixed_boundary[1] + (tail_gap + 1) * vc),
        )
        arrow = _arrow_cells(direction, arrow_tail)

        input_coords = np.vstack([fixed, mover_input, arrow])
        shift_r = -int(input_coords[:, 0].min())
        shift_c = -int(input_coords[:, 1].min())

        fixed_n = _translate(fixed, shift_r, shift_c)
        mover_in_n = _translate(mover_input, shift_r, shift_c)
        mover_out_n = _translate(mover, shift_r, shift_c)
        arrow_n = _translate(arrow, shift_r, shift_c)

        fragment_cells = _coord_set(np.vstack([fixed_n, mover_in_n]))
        if any(max(abs(int(r1 - r2)), abs(int(c1 - c2))) <= 1 for r1, c1 in arrow_n for r2, c2 in fragment_cells):
            continue

        input_cells = [(int(r), int(c), int(color)) for r, c in fixed_n]
        input_cells += [(int(r), int(c), int(color)) for r, c in mover_in_n]
        input_cells += [(int(r), int(c), YELLOW) for r, c in arrow_n]
        output_cells = [(int(r), int(c), int(color)) for r, c in fixed_n]
        output_cells += [(int(r), int(c), int(color)) for r, c in mover_out_n]

        h = max(r for r, _c, _v in input_cells) + 1
        w = max(c for _r, c, _v in input_cells) + 1
        return {
            "input_cells": input_cells,
            "output_cells": output_cells,
            "height": int(h),
            "width": int(w),
        }
    raise RuntimeError("failed to build a valid pair scene")


def _can_place(occupied: np.ndarray, cells: list[tuple[int, int, int]], top: int, left: int) -> bool:
    h, w = occupied.shape
    for r, c, _ in cells:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r0 = max(0, rr - SCENE_PAD)
        r1 = min(h, rr + SCENE_PAD + 1)
        c0 = max(0, cc - SCENE_PAD)
        c1 = min(w, cc + SCENE_PAD + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _paint(canvas: np.ndarray, cells: list[tuple[int, int, int]], top: int, left: int) -> None:
    for r, c, color in cells:
        canvas[int(top + r), int(left + c)] = int(color)


def _component_count_by_color(grid: np.ndarray, color: int) -> int:
    return len(_components(grid == color, connectivity=4))


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = int(rng.integers(2, 5))
    colors = [int(c) for c in rng.choice(COLOR_POOL, size=pair_count, replace=False)]
    directions = list(rng.permutation(np.asarray(("up", "down", "left", "right"), dtype=object))[:pair_count])
    scenes = [_build_pair_scene(rng, color, str(direction)) for color, direction in zip(colors, directions)]
    scenes.sort(key=lambda scene: (-(scene["height"] * scene["width"]), -scene["height"]))

    input_canvas = np.full((MAX_SIDE, MAX_SIDE), BG, dtype=int)
    output_canvas = np.full((MAX_SIDE, MAX_SIDE), BG, dtype=int)
    occupied = np.zeros((MAX_SIDE, MAX_SIDE), dtype=bool)

    for scene in scenes:
        h = int(scene["height"])
        w = int(scene["width"])
        placed = False
        for _ in range(250):
            top = int(rng.integers(0, MAX_SIDE - h + 1))
            left = int(rng.integers(0, MAX_SIDE - w + 1))
            if not _can_place(occupied, scene["input_cells"], top, left):
                continue
            _paint(input_canvas, scene["input_cells"], top, left)
            _paint(output_canvas, scene["output_cells"], top, left)
            for r, c, _color in scene["input_cells"]:
                occupied[top + int(r), left + int(c)] = True
            placed = True
            break
        if not placed:
            return None

    used = np.argwhere(input_canvas != BG)
    if used.size == 0:
        return None
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)

    extra_top = int(rng.integers(0, min(4, int(r0)) + 1))
    extra_left = int(rng.integers(0, min(4, int(c0)) + 1))
    extra_bottom = int(rng.integers(0, min(4, int(MAX_SIDE - 1 - r1)) + 1))
    extra_right = int(rng.integers(0, min(4, int(MAX_SIDE - 1 - c1)) + 1))

    rs = slice(int(r0 - extra_top), int(r1 + extra_bottom + 1))
    cs = slice(int(c0 - extra_left), int(c1 + extra_right + 1))
    inp = input_canvas[rs, cs].copy()
    out = output_canvas[rs, cs].copy()

    if max(inp.shape) > MAX_SIDE or max(out.shape) > MAX_SIDE:
        return None
    if np.array_equal(inp, out):
        return None
    if len(_components(inp == YELLOW, connectivity=8)) != pair_count:
        return None
    for color in colors:
        if _component_count_by_color(inp, color) != 2:
            return None
        if _component_count_by_color(out, color) != 1:
            return None

    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is None:
            continue
        inp, expected = result
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
