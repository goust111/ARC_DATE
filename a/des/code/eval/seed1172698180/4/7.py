from __future__ import annotations

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
DIR_ORDER = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_MAX_ATTEMPTS = 400
PAIR_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120

Cell = tuple[int, int]
Shape = tuple[Cell, ...]


def _normalize(cells: set[Cell] | list[Cell] | tuple[Cell, ...]) -> Shape:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotate(shape: Shape) -> Shape:
    max_c = max(c for _, c in shape)
    return _normalize({(c, max_c - r) for r, c in shape})


def _flip_h(shape: Shape) -> Shape:
    max_c = max(c for _, c in shape)
    return _normalize({(r, max_c - c) for r, c in shape})


def _flip_v(shape: Shape) -> Shape:
    max_r = max(r for r, _ in shape)
    return _normalize({(max_r - r, c) for r, c in shape})


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_cells(shape: Shape, top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _neighbor4(cell: Cell) -> tuple[Cell, Cell, Cell, Cell]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _random_shape(rng: np.random.Generator, area: int) -> Shape:
    for _ in range(80):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        if h * w < area:
            continue
        cells = {(int(rng.integers(0, h)), int(rng.integers(0, w)))}
        while len(cells) < area:
            frontier: list[Cell] = []
            for cell in cells:
                for nxt in _neighbor4(cell):
                    rr, cc = nxt
                    if 0 <= rr < h and 0 <= cc < w and nxt not in cells:
                        frontier.append(nxt)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != area:
            continue
        shape = _normalize(cells)
        for _ in range(int(rng.integers(0, 4))):
            shape = _rotate(shape)
        if rng.random() < 0.5:
            shape = _flip_h(shape)
        if rng.random() < 0.5:
            shape = _flip_v(shape)
        return _normalize(shape)
    raise RuntimeError("failed to sample connected shape")


def _coords_to_set(coords: np.ndarray) -> set[Cell]:
    return {(int(r), int(c)) for r, c in coords}


def _cells_clear(cells: set[Cell], reserved: set[Cell], margin: int = 1) -> bool:
    for r, c in cells:
        for rr, cc in reserved:
            if max(abs(r - rr), abs(c - cc)) <= margin:
                return False
    return True


def _pole_available(cell: Cell, sprite_cells: set[Cell], poles: set[Cell], legend_cells: set[Cell]) -> bool:
    if cell in sprite_cells or cell in poles or cell in legend_cells:
        return False
    for other in poles:
        if max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) <= 1:
            return False
    return True


def _legend_colors_and_cells(g: np.ndarray) -> tuple[list[int], set[Cell]]:
    h, w = g.shape
    if w < 2:
        return [], set()
    colors: list[int] = []
    cells: set[Cell] = set()
    for r in range(h):
        left = int(g[r, w - 2])
        right = int(g[r, w - 1])
        if left != right or left in {BG, YELLOW}:
            break
        colors.append(left)
        cells.add((r, w - 2))
        cells.add((r, w - 1))
    return colors, cells


def _border_poles(g: np.ndarray, legend_cells: set[Cell]) -> list[Cell]:
    h, w = g.shape
    poles: list[Cell] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in legend_cells:
                continue
            if r not in {0, h - 1} and c not in {0, w - 1}:
                continue
            if int(g[r, c]) == YELLOW:
                poles.append((r, c))
    return poles


def _find_components(g: np.ndarray, forbidden: set[Cell]) -> list[dict[str, object]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or (r, c) in forbidden:
                continue
            color = int(g[r, c])
            if color in {BG, YELLOW}:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Cell] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbor4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if (nr, nc) in forbidden:
                            continue
                        if int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": np.array(sorted(coords), dtype=int),
                }
            )
    return comps


def _direction_for_component(coords: np.ndarray, poles: list[Cell], h: int, w: int) -> tuple[str, bool]:
    top_cols = {c for r, c in poles if r == 0}
    bottom_cols = {c for r, c in poles if r == h - 1}
    left_rows = {r for r, c in poles if c == 0}
    right_rows = {r for r, c in poles if c == w - 1}

    row_bounds: dict[int, list[int]] = {}
    col_bounds: dict[int, list[int]] = {}
    for r, c in coords:
        r = int(r)
        c = int(c)
        if r not in row_bounds:
            row_bounds[r] = [c, c]
        else:
            row_bounds[r][0] = min(row_bounds[r][0], c)
            row_bounds[r][1] = max(row_bounds[r][1], c)
        if c not in col_bounds:
            col_bounds[c] = [r, r]
        else:
            col_bounds[c][0] = min(col_bounds[c][0], r)
            col_bounds[c][1] = max(col_bounds[c][1], r)

    best_by_dir: dict[str, int] = {}
    for c, (rmin, rmax) in col_bounds.items():
        if c in top_cols:
            best_by_dir["up"] = min(best_by_dir.get("up", h + w), rmin)
        if c in bottom_cols:
            best_by_dir["down"] = min(best_by_dir.get("down", h + w), (h - 1) - rmax)
    for r, (cmin, cmax) in row_bounds.items():
        if r in left_rows:
            best_by_dir["left"] = min(best_by_dir.get("left", h + w), cmin)
        if r in right_rows:
            best_by_dir["right"] = min(best_by_dir.get("right", h + w), (w - 1) - cmax)

    if not best_by_dir:
        raise AssertionError("each sprite must align with at least one pole")

    best_dist = min(best_by_dir.values())
    best_dirs = [d for d, dist in best_by_dir.items() if dist == best_dist]
    best_dirs.sort(key=lambda d: DIR_ORDER.index(d))
    return best_dirs[0], len(best_dirs) == 1


def _simulate_sprites(sprites: list[dict[str, object]], h: int, w: int) -> dict[str, object]:
    states: list[dict[str, object]] = []
    for sprite in sprites:
        states.append(
            {
                "color": int(sprite["color"]),
                "dir": str(sprite["dir"]),
                "coords": np.array(sprite["coords"], dtype=int),
                "offset": np.array([0, 0], dtype=int),
                "stop_reason": None,
            }
        )

    moving = list(range(len(states)))
    stopped_prev: set[Cell] = set()
    guard = h + w + max(h, w) + 10
    for _ in range(guard):
        if not moving:
            break
        newly_stopped: list[int] = []
        movers: list[int] = []
        for idx in moving:
            state = states[idx]
            dr, dc = DIR_VECS[str(state["dir"])]
            current = np.array(state["coords"], dtype=int) + np.array(state["offset"], dtype=int)
            nxt = current + np.array([dr, dc], dtype=int)
            hit_edge = bool(
                np.any(nxt[:, 0] < 0)
                or np.any(nxt[:, 0] >= h)
                or np.any(nxt[:, 1] < 0)
                or np.any(nxt[:, 1] >= w)
            )
            hit_stopped = any((int(r), int(c)) in stopped_prev for r, c in nxt)
            if hit_edge or hit_stopped:
                state["stop_reason"] = "edge" if hit_edge else "blocked"
                newly_stopped.append(idx)
            else:
                movers.append(idx)

        for idx in movers:
            dr, dc = DIR_VECS[str(states[idx]["dir"])]
            states[idx]["offset"] = np.array(states[idx]["offset"], dtype=int) + np.array([dr, dc], dtype=int)

        moving = [idx for idx in moving if idx not in newly_stopped]
        for idx in newly_stopped:
            current = np.array(states[idx]["coords"], dtype=int) + np.array(states[idx]["offset"], dtype=int)
            stopped_prev.update((int(r), int(c)) for r, c in current)
    else:
        raise RuntimeError("simulation did not terminate")

    if moving:
        raise RuntimeError("simulation ended with moving sprites")

    final_cells: list[set[Cell]] = []
    stop_reasons: list[str] = []
    occupancy: dict[Cell, int] = {}
    for state in states:
        current = np.array(state["coords"], dtype=int) + np.array(state["offset"], dtype=int)
        cells = {(int(r), int(c)) for r, c in current}
        final_cells.append(cells)
        stop_reasons.append(str(state["stop_reason"]))
        for cell in cells:
            occupancy[cell] = occupancy.get(cell, 0) + 1

    return {
        "states": states,
        "final_cells": final_cells,
        "stop_reasons": stop_reasons,
        "has_overlap": any(count > 1 for count in occupancy.values()),
    }


def _render_output(shape: tuple[int, int], poles: list[Cell], priority: list[int], sim: dict[str, object]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    rank = {color: idx for idx, color in enumerate(priority)}
    best_rank: dict[Cell, int] = {}
    states = list(sim["states"])
    final_cells = list(sim["final_cells"])
    for state, cells in zip(states, final_cells):
        color = int(state["color"])
        color_rank = rank.get(color, len(rank))
        for cell in cells:
            prev = best_rank.get(cell)
            if prev is None or color_rank < prev:
                best_rank[cell] = color_rank
                out[cell] = color
    for r, c in poles:
        out[r, c] = YELLOW
    return out


def _scene_from_grid(g: np.ndarray) -> tuple[list[dict[str, object]], list[Cell], list[int]]:
    h, w = g.shape
    priority, legend_cells = _legend_colors_and_cells(g)
    poles = _border_poles(g, legend_cells)
    work = np.array(g, dtype=int, copy=True)
    for r, c in legend_cells:
        work[r, c] = BG
    components = _find_components(work, legend_cells)
    sprites: list[dict[str, object]] = []
    for comp in components:
        coords = np.array(comp["coords"], dtype=int)
        direction, unique = _direction_for_component(coords, poles, h, w)
        if not unique:
            raise AssertionError("pole attraction must be unambiguous")
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "dir": direction,
            }
        )
    return sprites, poles, priority


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each monochrome sprite toward its nearest aligned yellow border pole, then resolve overlaps by the top-right priority legend.

    The legend is a 2xN strip in the upper-right corner, one color per row from highest to lowest priority.
    Poles remain visible in the output and the legend is removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sprites, poles, priority = _scene_from_grid(g)
    sim = _simulate_sprites(sprites, *g.shape)
    return _render_output(g.shape, poles, priority, sim)


def _legend_cells(h: int, w: int, count: int) -> set[Cell]:
    return {(r, w - 2) for r in range(count)} | {(r, w - 1) for r in range(count)}


def _direction_bias_range(axis_len: int, span: int, direction: str) -> tuple[int, int] | None:
    lo = 1
    hi = axis_len - span - 1
    if hi < lo:
        return None
    half = axis_len // 2
    if direction in {"up", "left"}:
        hi = min(hi, max(lo, half - span))
    else:
        lo = max(lo, half)
    if hi < lo:
        return None
    return lo, hi


def _randint_inclusive(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _make_sprite(color: int, direction: str, coords: np.ndarray) -> dict[str, object]:
    return {
        "color": int(color),
        "dir": direction,
        "coords": np.array(coords, dtype=int),
    }


def _valid_vertical_pole_cols(shape: Shape, left: int, w: int, direction: str) -> list[int]:
    cols = sorted({left + c for _, c in shape})
    if direction == "up":
        return [c for c in cols if c < w - 2]
    return cols


def _valid_right_pole_rows(shape: Shape, top: int, legend_h: int) -> list[int]:
    return [top + r for r, _ in shape if top + r >= legend_h]


def _can_use_direction(coords: np.ndarray, direction: str, base_poles: set[Cell], h: int, w: int) -> bool:
    best_dir, unique = _direction_for_component(coords, sorted(base_poles), h, w)
    return unique and best_dir == direction


def _maybe_add_opposite_pole(
    coords: np.ndarray,
    direction: str,
    sprite_cells: set[Cell],
    poles: set[Cell],
    legend_cells: set[Cell],
    legend_h: int,
    h: int,
    w: int,
    rng: np.random.Generator,
) -> set[Cell]:
    if rng.random() >= 0.5:
        return poles

    if direction == "up":
        candidates = sorted({int(c) for _, c in coords})
        rng.shuffle(candidates)
        for col in candidates:
            cell = (h - 1, col)
            trial = set(poles)
            if not _pole_available(cell, sprite_cells, trial, legend_cells):
                continue
            trial.add(cell)
            if _can_use_direction(coords, direction, trial, h, w):
                return trial
    elif direction == "down":
        candidates = sorted({int(c) for _, c in coords if int(c) < w})
        rng.shuffle(candidates)
        for col in candidates:
            cell = (0, col)
            trial = set(poles)
            if not _pole_available(cell, sprite_cells, trial, legend_cells):
                continue
            trial.add(cell)
            if _can_use_direction(coords, direction, trial, h, w):
                return trial
    elif direction == "left":
        candidates = sorted({int(r) for r, _ in coords})
        rng.shuffle(candidates)
        for row in candidates:
            cell = (row, w - 1)
            trial = set(poles)
            if row < legend_h:
                continue
            if not _pole_available(cell, sprite_cells, trial, legend_cells):
                continue
            trial.add(cell)
            if _can_use_direction(coords, direction, trial, h, w):
                return trial
    else:
        candidates = sorted({int(r) for r, _ in coords})
        rng.shuffle(candidates)
        for row in candidates:
            cell = (row, 0)
            trial = set(poles)
            if not _pole_available(cell, sprite_cells, trial, legend_cells):
                continue
            trial.add(cell)
            if _can_use_direction(coords, direction, trial, h, w):
                return trial
    return poles


def _make_random_independent(
    rng: np.random.Generator,
    h: int,
    w: int,
    color: int,
    legend_h: int,
    legend_cells: set[Cell],
    sprite_cells: set[Cell],
    poles: set[Cell],
) -> tuple[dict[str, object], set[Cell]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        direction = str(rng.choice(DIR_ORDER))
        shape = _random_shape(rng, int(rng.integers(3, 7)))
        sh, sw = _shape_dims(shape)
        if direction in {"up", "down"}:
            r_bounds = _direction_bias_range(h, sh, direction)
            c_bounds = (1, w - sw - 1)
        else:
            r_bounds = (1, h - sh - 1)
            c_bounds = _direction_bias_range(w, sw, direction)
        if r_bounds is None or c_bounds is None or r_bounds[1] < r_bounds[0] or c_bounds[1] < c_bounds[0]:
            continue
        top = _randint_inclusive(rng, r_bounds)
        left = _randint_inclusive(rng, c_bounds)
        coords = _shape_cells(shape, top, left)
        cells = _coords_to_set(coords)
        if not _cells_clear(cells, sprite_cells | poles | legend_cells, margin=1):
            continue

        trial_poles = set(poles)
        if direction in {"up", "down"}:
            valid_cols = _valid_vertical_pole_cols(shape, left, w, direction)
            rng.shuffle(valid_cols)
            placed = False
            for col in valid_cols:
                pole = (0, col) if direction == "up" else (h - 1, col)
                if not _pole_available(pole, sprite_cells | cells, trial_poles, legend_cells):
                    continue
                trial_poles.add(pole)
                placed = True
                break
            if not placed:
                continue
        elif direction == "left":
            rows = sorted({top + r for r, _ in shape})
            rng.shuffle(rows)
            placed = False
            for row in rows:
                pole = (row, 0)
                if not _pole_available(pole, sprite_cells | cells, trial_poles, legend_cells):
                    continue
                trial_poles.add(pole)
                placed = True
                break
            if not placed:
                continue
        else:
            rows = _valid_right_pole_rows(shape, top, legend_h)
            rng.shuffle(rows)
            placed = False
            for row in rows:
                pole = (row, w - 1)
                if not _pole_available(pole, sprite_cells | cells, trial_poles, legend_cells):
                    continue
                trial_poles.add(pole)
                placed = True
                break
            if not placed:
                continue

        if not _can_use_direction(coords, direction, trial_poles, h, w):
            continue
        trial_poles = _maybe_add_opposite_pole(
            coords, direction, sprite_cells | cells, trial_poles, legend_cells, legend_h, h, w, rng
        )
        if not _can_use_direction(coords, direction, trial_poles, h, w):
            continue
        return _make_sprite(color, direction, coords), trial_poles
    return None


def _make_block_pair(
    rng: np.random.Generator,
    h: int,
    w: int,
    colors: tuple[int, int],
    legend_h: int,
    legend_cells: set[Cell],
    sprite_cells: set[Cell],
    poles: set[Cell],
) -> tuple[list[dict[str, object]], set[Cell]] | None:
    for _ in range(PAIR_ATTEMPTS):
        direction = str(rng.choice(DIR_ORDER))
        shape = _random_shape(rng, int(rng.integers(3, 7)))
        sh, sw = _shape_dims(shape)
        trial_poles = set(poles)

        if direction == "up":
            left_max = w - sw - 1
            top_a_max = h - 2 * sh - 4
            if left_max < 1 or top_a_max < 1:
                continue
            left = int(rng.integers(1, left_max + 1))
            top_a = _randint_inclusive(rng, (1, top_a_max))
            top_b_min = top_a + sh + 2
            top_b_max = h - sh - 2
            if top_b_min > top_b_max:
                continue
            top_b = _randint_inclusive(rng, (top_b_min, top_b_max))
            coords_a = _shape_cells(shape, top_a, left)
            coords_b = _shape_cells(shape, top_b, left)
            cols = _valid_vertical_pole_cols(shape, left, w, direction)
            rng.shuffle(cols)
            if not cols:
                continue
            pole = (0, cols[0])
        elif direction == "down":
            left_max = w - sw - 1
            top_b_max = h - 2 * sh - 4
            if left_max < 1 or top_b_max < 1:
                continue
            left = int(rng.integers(1, left_max + 1))
            top_b = _randint_inclusive(rng, (1, top_b_max))
            top_a_min = top_b + sh + 2
            top_a_max = h - sh - 2
            if top_a_min > top_a_max:
                continue
            top_a = _randint_inclusive(rng, (top_a_min, top_a_max))
            coords_a = _shape_cells(shape, top_a, left)
            coords_b = _shape_cells(shape, top_b, left)
            cols = _valid_vertical_pole_cols(shape, left, w, direction)
            rng.shuffle(cols)
            if not cols:
                continue
            pole = (h - 1, cols[0])
        elif direction == "left":
            top_max = h - sh - 1
            left_a_max = w - 2 * sw - 4
            if top_max < 1 or left_a_max < 1:
                continue
            top = int(rng.integers(1, top_max + 1))
            left_a = _randint_inclusive(rng, (1, left_a_max))
            left_b_min = left_a + sw + 2
            left_b_max = w - sw - 2
            if left_b_min > left_b_max:
                continue
            left_b = _randint_inclusive(rng, (left_b_min, left_b_max))
            coords_a = _shape_cells(shape, top, left_a)
            coords_b = _shape_cells(shape, top, left_b)
            rows = sorted({top + r for r, _ in shape})
            rng.shuffle(rows)
            pole = (rows[0], 0)
        else:
            top_max = h - sh - 1
            left_b_max = w - 2 * sw - 4
            if top_max < 1 or left_b_max < 1:
                continue
            top = int(rng.integers(1, top_max + 1))
            left_b = _randint_inclusive(rng, (1, left_b_max))
            left_a_min = left_b + sw + 2
            left_a_max = w - sw - 2
            if left_a_min > left_a_max:
                continue
            left_a = _randint_inclusive(rng, (left_a_min, left_a_max))
            coords_a = _shape_cells(shape, top, left_a)
            coords_b = _shape_cells(shape, top, left_b)
            rows = _valid_right_pole_rows(shape, top, legend_h)
            rng.shuffle(rows)
            if not rows:
                continue
            pole = (rows[0], w - 1)

        cells_a = _coords_to_set(coords_a)
        cells_b = _coords_to_set(coords_b)
        if not _cells_clear(cells_a, sprite_cells | poles | legend_cells, margin=1):
            continue
        if not _cells_clear(cells_b, sprite_cells | poles | legend_cells | cells_a, margin=1):
            continue
        if not _pole_available(pole, sprite_cells | cells_a | cells_b, trial_poles, legend_cells):
            continue
        trial_poles.add(pole)
        if not _can_use_direction(coords_a, direction, trial_poles, h, w):
            continue
        if not _can_use_direction(coords_b, direction, trial_poles, h, w):
            continue

        trial_poles = _maybe_add_opposite_pole(
            coords_a, direction, sprite_cells | cells_a | cells_b, trial_poles, legend_cells, legend_h, h, w, rng
        )
        if not _can_use_direction(coords_a, direction, trial_poles, h, w):
            continue
        if not _can_use_direction(coords_b, direction, trial_poles, h, w):
            continue

        sprites = [
            _make_sprite(colors[0], direction, coords_a),
            _make_sprite(colors[1], direction, coords_b),
        ]
        return sprites, trial_poles
    return None


def _make_overlap_pair(
    rng: np.random.Generator,
    h: int,
    w: int,
    colors: tuple[int, int],
    legend_h: int,
    legend_cells: set[Cell],
    sprite_cells: set[Cell],
    poles: set[Cell],
) -> tuple[list[dict[str, object]], set[Cell]] | None:
    for _ in range(PAIR_ATTEMPTS):
        vdir = str(rng.choice(["up", "down"]))
        hdir = str(rng.choice(["left", "right"]))
        vshape = _random_shape(rng, int(rng.integers(3, 7)))
        hshape = _random_shape(rng, int(rng.integers(3, 7)))
        vh, vw = _shape_dims(vshape)
        hh, hw = _shape_dims(hshape)
        vmax_r = vh - 1
        hmax_c = hw - 1
        if vdir == "up":
            vcells = [(r, c) for r, c in vshape if r > 0]
            final_v_r = 0
        else:
            vcells = [(r, c) for r, c in vshape if r < vmax_r]
            final_v_r = h - vh
        if hdir == "left":
            hcells = [(r, c) for r, c in hshape if c > 0]
            final_h_c = 0
        else:
            hcells = [(r, c) for r, c in hshape if c < hmax_c]
            final_h_c = w - hw
        if not vcells or not hcells:
            continue

        av = vcells[int(rng.integers(0, len(vcells)))]
        ah = hcells[int(rng.integers(0, len(hcells)))]
        fixed_top_h = final_v_r + av[0] - ah[0]
        fixed_left_v = final_h_c + ah[1] - av[1]
        if not (1 <= fixed_top_h <= h - hh - 1):
            continue
        if not (1 <= fixed_left_v <= w - vw - 1):
            continue

        if vdir == "up":
            top_v_bounds = (2, h - vh - 1)
        else:
            top_v_bounds = (1, h - vh - 2)
        if hdir == "left":
            left_h_bounds = (2, w - hw - 1)
        else:
            left_h_bounds = (1, w - hw - 2)
        if top_v_bounds[1] < top_v_bounds[0] or left_h_bounds[1] < left_h_bounds[0]:
            continue

        top_v = _randint_inclusive(rng, top_v_bounds)
        left_h = _randint_inclusive(rng, left_h_bounds)
        coords_v = _shape_cells(vshape, top_v, fixed_left_v)
        coords_h = _shape_cells(hshape, fixed_top_h, left_h)
        cells_v = _coords_to_set(coords_v)
        cells_h = _coords_to_set(coords_h)
        if cells_v & cells_h:
            continue
        if not _cells_clear(cells_v, sprite_cells | poles | legend_cells, margin=1):
            continue
        if not _cells_clear(cells_h, sprite_cells | poles | legend_cells | cells_v, margin=1):
            continue

        trial_poles = set(poles)
        valid_v_cols = _valid_vertical_pole_cols(vshape, fixed_left_v, w, vdir)
        valid_h_rows = (
            sorted({fixed_top_h + r for r, _ in hshape})
            if hdir == "left"
            else _valid_right_pole_rows(hshape, fixed_top_h, legend_h)
        )
        rng.shuffle(valid_v_cols)
        rng.shuffle(valid_h_rows)
        if not valid_v_cols or not valid_h_rows:
            continue

        vpole = (0, valid_v_cols[0]) if vdir == "up" else (h - 1, valid_v_cols[0])
        hpole = (valid_h_rows[0], 0) if hdir == "left" else (valid_h_rows[0], w - 1)
        if not _pole_available(vpole, sprite_cells | cells_v | cells_h, trial_poles, legend_cells):
            continue
        trial_poles.add(vpole)
        if not _pole_available(hpole, sprite_cells | cells_v | cells_h, trial_poles, legend_cells):
            continue
        trial_poles.add(hpole)

        if not _can_use_direction(coords_v, vdir, trial_poles, h, w):
            continue
        if not _can_use_direction(coords_h, hdir, trial_poles, h, w):
            continue

        sprites = [
            _make_sprite(colors[0], vdir, coords_v),
            _make_sprite(colors[1], hdir, coords_h),
        ]
        sim = _simulate_sprites(sprites, h, w)
        if not bool(sim["has_overlap"]):
            continue
        if any(reason == "blocked" for reason in sim["stop_reasons"]):
            continue
        return sprites, trial_poles
    return None


def _add_spare_poles(
    rng: np.random.Generator,
    h: int,
    w: int,
    legend_h: int,
    legend_cells: set[Cell],
    sprite_cells: set[Cell],
    poles: set[Cell],
) -> set[Cell]:
    trial = set(poles)
    used_rows = {r for r, _ in sprite_cells}
    used_cols = {c for _, c in sprite_cells}
    target_count = int(rng.integers(0, 3))
    for _ in range(target_count):
        candidates: list[Cell] = []
        for col in range(1, w - 2):
            if col not in used_cols:
                candidates.append((0, col))
        for col in range(1, w - 1):
            if col not in used_cols:
                candidates.append((h - 1, col))
        for row in range(1, h - 1):
            if row not in used_rows:
                candidates.append((row, 0))
        for row in range(max(legend_h, 1), h - 1):
            if row not in used_rows:
                candidates.append((row, w - 1))
        rng.shuffle(candidates)
        placed = False
        for cell in candidates:
            if _pole_available(cell, sprite_cells, trial, legend_cells):
                trial.add(cell)
                placed = True
                break
        if not placed:
            break
    return trial


def _render_input(h: int, w: int, sprites: list[dict[str, object]], poles: set[Cell], priority: list[int]) -> np.ndarray:
    g = np.full((h, w), BG, dtype=int)
    for idx, color in enumerate(priority):
        g[idx, w - 2] = int(color)
        g[idx, w - 1] = int(color)
    for r, c in sorted(poles):
        g[r, c] = YELLOW
    for sprite in sprites:
        coords = np.array(sprite["coords"], dtype=int)
        g[coords[:, 0], coords[:, 1]] = int(sprite["color"])
    return g


def _feature_match(mode: str, sim: dict[str, object]) -> bool:
    has_overlap = bool(sim["has_overlap"])
    has_block = any(reason == "blocked" for reason in sim["stop_reasons"])
    if mode == "plain":
        return not has_overlap and not has_block
    if mode == "overlap":
        return has_overlap and not has_block
    if mode == "block":
        return has_block and not has_overlap
    return has_overlap and has_block


def _build_mode_scene(rng: np.random.Generator, mode: str) -> np.ndarray | None:
    if mode == "both":
        n_sprites = 4
        h = int(rng.integers(15, 20))
        w = int(rng.integers(17, 22))
    elif mode == "plain":
        n_sprites = int(rng.integers(2, 5))
        h = int(rng.integers(12, 18))
        w = int(rng.integers(14, 20))
    else:
        n_sprites = int(rng.integers(3, 5))
        h = int(rng.integers(13, 19))
        w = int(rng.integers(15, 21))

    if w < 8 or h < n_sprites + 4:
        return None

    colors = [int(v) for v in rng.choice(np.array(PALETTE, dtype=int), size=n_sprites, replace=False)]
    priority = [int(v) for v in rng.permutation(colors)]
    legend_h = len(priority)
    legend_cells = _legend_cells(h, w, legend_h)

    sprites: list[dict[str, object]] = []
    poles: set[Cell] = set()
    sprite_cells: set[Cell] = set()
    color_idx = 0

    if mode in {"block", "both"}:
        block = _make_block_pair(
            rng, h, w, (colors[color_idx], colors[color_idx + 1]), legend_h, legend_cells, sprite_cells, poles
        )
        if block is None:
            return None
        new_sprites, poles = block
        sprites.extend(new_sprites)
        for sprite in new_sprites:
            sprite_cells |= _coords_to_set(np.array(sprite["coords"], dtype=int))
        color_idx += 2

    if mode in {"overlap", "both"}:
        overlap = _make_overlap_pair(
            rng, h, w, (colors[color_idx], colors[color_idx + 1]), legend_h, legend_cells, sprite_cells, poles
        )
        if overlap is None:
            return None
        new_sprites, poles = overlap
        sprites.extend(new_sprites)
        for sprite in new_sprites:
            sprite_cells |= _coords_to_set(np.array(sprite["coords"], dtype=int))
        color_idx += 2

    while color_idx < n_sprites:
        built = _make_random_independent(rng, h, w, colors[color_idx], legend_h, legend_cells, sprite_cells, poles)
        if built is None:
            return None
        sprite, poles = built
        sprites.append(sprite)
        sprite_cells |= _coords_to_set(np.array(sprite["coords"], dtype=int))
        color_idx += 1

    poles = _add_spare_poles(rng, h, w, legend_h, legend_cells, sprite_cells, poles)
    poles = set(sorted(poles))

    for sprite in sprites:
        coords = np.array(sprite["coords"], dtype=int)
        best_dir, unique = _direction_for_component(coords, sorted(poles), h, w)
        if not unique or best_dir != str(sprite["dir"]):
            return None

    sim = _simulate_sprites(sprites, h, w)
    if not _feature_match(mode, sim):
        return None

    inp = _render_input(h, w, sprites, poles, priority)
    try:
        out = main(inp)
    except Exception:
        return None
    visible_colors = set(int(v) for v in np.unique(out)) - {BG, YELLOW}
    if len(visible_colors) < 2:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    modes = ("plain", "overlap", "block", "both")
    for _ in range(GEN_MAX_ATTEMPTS):
        mode = str(rng.choice(modes))
        grid = _build_mode_scene(rng, mode)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
