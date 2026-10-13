from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")

# --------------- generator config ---------------

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 34
OBJECT_COUNT_RANGE = (2, 4)
OPEN_LEN_RANGE = (2, 6)
PATTERN_DEPTH_RANGE = (1, 4)
INNER_GAP_RANGE = (1, 2)
SIDE_MARGIN_EXTRA_RANGE = (0, 2)
PLACEMENT_TRIES = 200
GEN_RETRIES = 300


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            q = deque([(r0, c0)])
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, c))
        if h > 1:
            coords.append((h - 1, c))
    for r in range(1, h - 1):
        coords.append((r, 0))
        if w > 1:
            coords.append((r, w - 1))
    return coords


def _opening_side(crop: np.ndarray) -> str | None:
    top = crop[0, :]
    bottom = crop[-1, :]
    left = crop[:, 0]
    right = crop[:, -1]
    side_runs = {
        "top": np.flatnonzero(top == BG),
        "bottom": np.flatnonzero(bottom == BG),
        "left": np.flatnonzero(left == BG),
        "right": np.flatnonzero(right == BG),
    }
    open_sides = [side for side, idxs in side_runs.items() if len(idxs) > 0]
    if len(open_sides) != 1:
        return None
    idxs = side_runs[open_sides[0]]
    if len(idxs) == 0:
        return None
    if len(idxs) > 1 and not np.all(np.diff(idxs) == 1):
        return None
    return open_sides[0]


def _analyze_envelope(crop: np.ndarray) -> dict | None:
    h, w = crop.shape
    if h < 3 or w < 3:
        return None

    side = _opening_side(crop)
    if side is None:
        return None

    boundary_colors = [
        int(crop[r, c])
        for r, c in _perimeter_coords(h, w)
        if int(crop[r, c]) != BG
    ]
    if not boundary_colors:
        return None
    envelope_color, _ = Counter(boundary_colors).most_common(1)[0]
    if any(color != envelope_color for color in boundary_colors):
        return None

    pattern_cells = [
        (int(r), int(c), int(crop[r, c]))
        for r in range(h)
        for c in range(w)
        if int(crop[r, c]) not in (BG, envelope_color)
    ]
    if not pattern_cells:
        return None

    rows = [r for r, _, _ in pattern_cells]
    cols = [c for _, c, _ in pattern_cells]
    if side == "top":
        shift = (-max(rows) - 1, 0)
    elif side == "bottom":
        shift = (h - min(rows), 0)
    elif side == "left":
        shift = (0, -max(cols) - 1)
    else:
        shift = (0, w - min(cols))

    return {
        "side": side,
        "envelope_color": int(envelope_color),
        "pattern_cells": pattern_cells,
        "shift": shift,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open every envelope by sliding its hidden pattern out through the gap.

    Each non-background connected component is an envelope-like rectangle:
    one boundary side has a contiguous background opening, the rest of the
    boundary is a single envelope color, and all other non-envelope pixels
    inside the bbox form the hidden pattern.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for comp in _components4(g != BG):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        crop = g[r0 : r1 + 1, c0 : c1 + 1]

        info = _analyze_envelope(crop)
        if info is None:
            continue

        envelope_color = int(info["envelope_color"])
        shift_r, shift_c = info["shift"]
        pattern_cells = info["pattern_cells"]

        block = out[r0 : r1 + 1, c0 : c1 + 1]
        block[:] = envelope_color
        block[crop == BG] = BG

        for pr, pc, color in pattern_cells:
            nr = r0 + pr + shift_r
            nc = c0 + pc + shift_c
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = int(color)

    return out


# --------------- generator helpers ---------------


def _random_spanning_mask(
    rng: random.Random,
    height: int,
    width: int,
    span_axis: str,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)

    if span_axis == "horizontal":
        row_path = [rng.randrange(height)]
        for _ in range(1, width):
            row_path.append(max(0, min(height - 1, row_path[-1] + rng.choice((-1, 0, 1)))))
        for c in range(width - 1):
            r0 = row_path[c]
            r1 = row_path[c + 1]
            mask[r0, c] = True
            mask[r0, c + 1] = True
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                mask[r, c + 1] = True
    else:
        col_path = [rng.randrange(width)]
        for _ in range(1, height):
            col_path.append(max(0, min(width - 1, col_path[-1] + rng.choice((-1, 0, 1)))))
        for r in range(height - 1):
            c0 = col_path[r]
            c1 = col_path[r + 1]
            mask[r, c0] = True
            mask[r + 1, c0] = True
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                mask[r + 1, c] = True

    total = height * width
    base_area = int(mask.sum())
    min_area = max(base_area, int(np.ceil(0.45 * total)))
    max_area = max(min_area, int(np.ceil(0.9 * total)))
    if total >= 4 and max_area == total and max_area > min_area:
        max_area -= 1
    target_area = rng.randint(min_area, max_area)

    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            break
        nr, nc = rng.choice(frontier)
        mask[nr, nc] = True

    return mask


def _paint_mask(
    rng: random.Random,
    mask: np.ndarray,
    available_colors: list[int],
) -> np.ndarray:
    h, w = mask.shape
    patch = np.zeros((h, w), dtype=int)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert cells

    if len(cells) <= 2:
        color_count = 1
    else:
        color_count = rng.choices([1, 2, 3], weights=[2, 4, 3], k=1)[0]
        color_count = min(color_count, len(cells), len(available_colors))
    colors = rng.sample(available_colors, color_count)

    if color_count == 1:
        for r, c in cells:
            patch[r, c] = colors[0]
        return patch

    mode = rng.choice(("rows", "cols", "checker", "random"))
    if mode == "rows":
        ranks = {row: i for i, row in enumerate(sorted({r for r, _ in cells}))}
        for r, c in cells:
            patch[r, c] = colors[ranks[r] % color_count]
    elif mode == "cols":
        ranks = {col: i for i, col in enumerate(sorted({c for _, c in cells}))}
        for r, c in cells:
            patch[r, c] = colors[ranks[c] % color_count]
    elif mode == "checker":
        for r, c in cells:
            patch[r, c] = colors[(r + c) % color_count]
    else:
        shuffled = cells[:]
        rng.shuffle(shuffled)
        for i, (r, c) in enumerate(shuffled):
            patch[r, c] = colors[i % color_count]

    used = {int(patch[r, c]) for r, c in cells}
    missing = [color for color in colors if color not in used]
    if missing:
        forced = cells[:]
        rng.shuffle(forced)
        for color, (r, c) in zip(missing, forced):
            patch[r, c] = color

    return patch


def _make_envelope_spec(
    rng: random.Random,
    envelope_color: int,
    side: str,
) -> dict:
    open_len = rng.randint(*OPEN_LEN_RANGE)
    depth = rng.randint(*PATTERN_DEPTH_RANGE)
    front_gap = rng.randint(*INNER_GAP_RANGE)
    back_gap = rng.randint(*INNER_GAP_RANGE)
    extra_a = rng.randint(*SIDE_MARGIN_EXTRA_RANGE)
    extra_b = rng.randint(*SIDE_MARGIN_EXTRA_RANGE)

    if side in ("top", "bottom"):
        env_h = 2 + front_gap + depth + back_gap
        env_w = 2 + extra_a + open_len + extra_b
        pattern_r0 = 1 + front_gap
        pattern_c0 = 1 + extra_a
        mask = _random_spanning_mask(rng, depth, open_len, "horizontal")
        opening = (0 if side == "top" else env_h - 1, slice(pattern_c0, pattern_c0 + open_len))
    else:
        env_h = 2 + extra_a + open_len + extra_b
        env_w = 2 + front_gap + depth + back_gap
        pattern_r0 = 1 + extra_a
        pattern_c0 = 1 + front_gap
        mask = _random_spanning_mask(rng, open_len, depth, "vertical")
        opening = (slice(pattern_r0, pattern_r0 + open_len), 0 if side == "left" else env_w - 1)

    patch = np.full((env_h, env_w), int(envelope_color), dtype=int)
    patch[opening] = BG
    pattern = _paint_mask(
        rng,
        mask,
        [color for color in COLORS if color != int(envelope_color)],
    )
    for r, c in np.argwhere(mask):
        patch[pattern_r0 + r, pattern_c0 + c] = int(pattern[r, c])

    info = _analyze_envelope(patch)
    assert info is not None
    pattern_cells = [(r, c, color) for r, c, color in info["pattern_cells"]]
    shift_r, shift_c = info["shift"]
    moved_cells = [(r + shift_r, c + shift_c, color) for r, c, color in pattern_cells]

    rel_core = {(r, c) for r in range(env_h) for c in range(env_w)}
    rel_core.update((r, c) for r, c, _ in moved_cells)

    return {
        "patch": patch,
        "envelope_color": int(envelope_color),
        "pattern_cells": pattern_cells,
        "moved_cells": moved_cells,
        "rel_core": rel_core,
        "row_bounds": (min(r for r, _ in rel_core), max(r for r, _ in rel_core)),
        "col_bounds": (min(c for _, c in rel_core), max(c for _, c in rel_core)),
    }


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((int(nr), int(nc)))
    return out


def _choose_directions(rng: random.Random, count: int) -> list[str]:
    if count <= 4 and rng.random() < 0.75:
        return rng.sample(list(SIDES), count)
    return [rng.choice(SIDES) for _ in range(count)]


def _place_specs(
    rng: random.Random,
    shape: tuple[int, int],
    specs: list[dict],
) -> list[tuple[dict, int, int]] | None:
    h, w = shape
    reserved = np.zeros((h, w), dtype=bool)
    placements: list[tuple[dict, int, int]] = []

    for spec in specs:
        min_r, max_r = spec["row_bounds"]
        min_c, max_c = spec["col_bounds"]
        r_lo = -min_r
        r_hi = h - 1 - max_r
        c_lo = -min_c
        c_hi = w - 1 - max_c
        if r_lo > r_hi or c_lo > c_hi:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            core = {(r0 + r, c0 + c) for r, c in spec["rel_core"]}
            if any(reserved[r, c] for r, c in core):
                continue
            for rr, cc in _halo(core, h, w):
                reserved[rr, cc] = True
            placements.append((spec, int(r0), int(c0)))
            placed = True
            break
        if not placed:
            return None

    return placements


def _render_input(
    shape: tuple[int, int],
    placements: list[tuple[dict, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec, r0, c0 in placements:
        patch = spec["patch"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return grid


def _render_expected(
    grid: np.ndarray,
    placements: list[tuple[dict, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for spec, r0, c0 in placements:
        patch = spec["patch"]
        ph, pw = patch.shape
        block = out[r0 : r0 + ph, c0 : c0 + pw]
        block[:] = int(spec["envelope_color"])
        block[patch == BG] = BG
        for mr, mc, color in spec["moved_cells"]:
            out[r0 + mr, c0 + mc] = int(color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = rng.randint(*OBJECT_COUNT_RANGE)
    min_side = max(GRID_SIDE_MIN, 10 + 4 * obj_count)
    h = rng.randint(min_side, GRID_SIDE_MAX)
    w = rng.randint(min_side, GRID_SIDE_MAX)

    directions = _choose_directions(rng, obj_count)
    envelope_colors = rng.sample(list(COLORS), obj_count)
    specs = [
        _make_envelope_spec(rng, envelope_color, side)
        for envelope_color, side in zip(envelope_colors, directions)
    ]
    specs.sort(key=lambda spec: -len(spec["rel_core"]))

    placements = _place_specs(rng, (h, w), specs)
    if placements is None:
        return None

    grid = _render_input((h, w), placements)
    expected = _render_expected(grid, placements)

    if np.array_equal(grid, expected):
        return None
    if int(np.count_nonzero(grid == BG)) <= int(np.count_nonzero(grid != BG)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
