from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 2
SEED = 5
MOVING_COLORS = (1, 3, 4, 7)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_by_color(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    allowed = set(int(c) for c in colors)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in allowed or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "min_r": int(coords[:, 0].min()),
                    "max_r": int(coords[:, 0].max()),
                    "min_c": int(coords[:, 1].min()),
                    "max_c": int(coords[:, 1].max()),
                }
            )
    return comps


def _drop_coords(coords: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    h, w = occupied.shape
    placed = coords.astype(int).copy()
    while True:
        nxt = placed + np.array([1, 0], dtype=int)
        if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= h) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= w):
            break
        if np.any(occupied[nxt[:, 0], nxt[:, 1]]):
            break
        placed = nxt
    return placed


def _settle_objects(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    moving = _components_by_color(g, MOVING_COLORS)
    for comp in moving:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = BG

    occupied = np.zeros(g.shape, dtype=bool)
    occupied[-1, :] = g[-1, :] == WALL
    settled: list[dict] = []
    order = sorted(
        moving,
        key=lambda comp: (-int(comp["max_r"]), int(comp["min_c"]), int(comp["min_r"]), int(comp["max_c"])),
    )
    for comp in order:
        dropped = _drop_coords(comp["coords"], occupied)
        occupied[dropped[:, 0], dropped[:, 1]] = True
        out[dropped[:, 0], dropped[:, 1]] = int(comp["color"])
        settled.append({"color": int(comp["color"]), "coords": dropped})

    seed_coords = np.argwhere(g == SEED)
    if seed_coords.size:
        out[seed_coords[:, 0], seed_coords[:, 1]] = SEED
    return out, settled


def _seed_stack_profile(settled: list[dict], col: int) -> tuple[int, list[int]]:
    occupied_rows: set[int] = set()
    seq: list[int] = []
    for obj in sorted(settled, key=lambda item: -int(item["coords"][:, 0].max())):
        column_rows = [int(r) for r, c in obj["coords"] if int(c) == int(col)]
        if not column_rows:
            continue
        occupied_rows.update(column_rows)
        seq.append(int(obj["color"]))
    return len(occupied_rows), seq


def main(grid: np.ndarray) -> np.ndarray:
    """Drop the colored objects onto the red floor, then use each seeded column's stack height as the radius of concentric square rings.

    Input structure:
    - The grid has black background, a full red bottom wall, several small monochrome 4-connected objects, and 1-3 grey seed pixels in distinct columns.
    - Movable objects use colors from blue/green/yellow/orange and slide only vertically.
    - Every seed column intersects at least two movable objects once those objects have stacked against the wall.

    Rule:
    - Keep the red wall fixed and erase the movable objects from their original positions.
    - Settle each movable object straight downward, nearest-to-wall first, until the next step would hit the wall or a previously settled object.
    - For each grey seed, count how many rows of settled object cells occupy that seed's column; draw that many Chebyshev square rings around the seed.
    - Ring color d comes from the bottom-to-top color order of objects in that seed column, cycling when the radius exceeds the number of stacked objects; the seed stays grey and settled objects are never overwritten.

    Key insight:
    - The post-gravity stack encodes two ring parameters at once: occupied height gives the number of rings, while bottom-to-top object order gives the repeating ring palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 2
    assert np.all(g[-1, :] == WALL)

    stacked_grid, settled = _settle_objects(g)
    out = stacked_grid.copy()

    protected = np.zeros(g.shape, dtype=bool)
    protected[-1, :] = True
    for obj in settled:
        protected[obj["coords"][:, 0], obj["coords"][:, 1]] = True
    seed_coords = np.argwhere(g == SEED)
    if seed_coords.size:
        protected[seed_coords[:, 0], seed_coords[:, 1]] = True

    h, w = g.shape
    for sr, sc in sorted((int(r), int(c)) for r, c in seed_coords):
        radius, colors = _seed_stack_profile(settled, int(sc))
        if radius <= 0 or not colors:
            continue
        for dist in range(1, int(radius) + 1):
            color = int(colors[(dist - 1) % len(colors)])
            r0 = max(0, int(sr) - dist)
            r1 = min(h - 1, int(sr) + dist)
            c0 = max(0, int(sc) - dist)
            c1 = min(w - 1, int(sc) + dist)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if max(abs(rr - int(sr)), abs(cc - int(sc))) != dist:
                        continue
                    if not protected[rr, cc]:
                        out[rr, cc] = color
    return out


_TEMPLATES = {
    1: [
        {"cells": ((0, 0), (0, 1)), "focus": 0},
        {"cells": ((0, 0), (0, 1)), "focus": 1},
        {"cells": ((0, 0), (0, 1), (0, 2)), "focus": 1},
    ],
    2: [
        {"cells": ((0, 0), (1, 0)), "focus": 0},
        {"cells": ((0, 0), (1, 0), (1, 1)), "focus": 0},
        {"cells": ((0, 1), (1, 0), (1, 1)), "focus": 1},
        {"cells": ((0, 1), (1, 0), (1, 1), (1, 2)), "focus": 1},
    ],
    3: [
        {"cells": ((0, 0), (1, 0), (2, 0)), "focus": 0},
    ],
}

_HEIGHT_COMPOSITIONS = {
    2: {2: ((1, 1),), 3: ((1, 2), (2, 1)), 4: ((2, 2), (1, 3), (3, 1))},
    3: {3: ((1, 1, 1),), 4: ((1, 1, 2), (1, 2, 1), (2, 1, 1))},
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _shift_to_row(coords: np.ndarray, top_row: int) -> np.ndarray:
    return coords + np.array([int(top_row) - int(coords[:, 0].min()), 0], dtype=int)


def _sample_distinct_columns(width: int, count: int) -> list[int] | None:
    min_gap = 5 if count >= 3 else 4
    options = list(range(1, width - 1))
    random.shuffle(options)
    if count >= 1 and random.random() < 0.45:
        edge_choices = [c for c in (1, 2, width - 3, width - 2) if 1 <= c < width - 1]
        random.shuffle(edge_choices)
        ordered = edge_choices + options
    else:
        ordered = options

    cols: list[int] = []
    for col in ordered:
        if all(abs(int(col) - int(prev)) >= min_gap for prev in cols):
            cols.append(int(col))
            if len(cols) == count:
                return sorted(cols)
    return None


def _sample_group_colors(count: int) -> list[int]:
    colors = [random.choice(MOVING_COLORS) for _ in range(count)]
    if count >= 2 and len(set(colors)) == 1:
        idx = random.randrange(count)
        choices = [c for c in MOVING_COLORS if c != colors[idx]]
        colors[idx] = random.choice(choices)
    return colors


def _choose_template(contrib: int, seed_col: int, width: int) -> dict | None:
    choices = []
    for template in _TEMPLATES[int(contrib)]:
        cols = [int(c) - int(template["focus"]) + int(seed_col) for _r, c in template["cells"]]
        if min(cols) < 0 or max(cols) >= width:
            continue
        choices.append(template)
    return random.choice(choices) if choices else None


def _place_seed_group(
    occupied: np.ndarray,
    seed_col: int,
    group_size: int,
    target_height: int,
    group_id: int,
) -> list[dict] | None:
    compositions = _HEIGHT_COMPOSITIONS[int(group_size)][int(target_height)]
    contribs = list(random.choice(compositions))
    colors = _sample_group_colors(group_size)
    group_objects: list[dict] = []

    for idx, contrib in enumerate(contribs):
        template = _choose_template(int(contrib), int(seed_col), occupied.shape[1])
        if template is None:
            return None
        local = np.array(
            [(int(r), int(c) - int(template["focus"]) + int(seed_col)) for r, c in template["cells"]],
            dtype=int,
        )
        placed = _drop_coords(local, occupied)
        occupied[placed[:, 0], placed[:, 1]] = True
        group_objects.append(
            {
                "color": int(colors[idx]),
                "coords": placed,
                "group": int(group_id),
                "seed_col": int(seed_col),
            }
        )

    height, seq = _seed_stack_profile(group_objects, int(seed_col))
    if int(height) != int(target_height) or len(seq) != int(group_size):
        return None
    return group_objects


def _occupied_with_objects(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[-1, :] = True
    for obj in objects:
        mask[obj["coords"][:, 0], obj["coords"][:, 1]] = True
    return mask


def _sample_extra_columns(width: int, blocked_cols: set[int], needed: int) -> list[int] | None:
    candidates = []
    for col in range(1, width - 1):
        if all(abs(int(col) - int(b)) >= 4 for b in blocked_cols):
            candidates.append(int(col))
    random.shuffle(candidates)
    return candidates[:needed] if len(candidates) >= needed else None


def _place_extra_object(occupied: np.ndarray, col: int, group_id: int) -> dict | None:
    contrib = random.choice((1, 1, 2))
    template = _choose_template(int(contrib), int(col), occupied.shape[1])
    if template is None:
        return None
    local = np.array(
        [(int(r), int(c) - int(template["focus"]) + int(col)) for r, c in template["cells"]],
        dtype=int,
    )
    placed = _drop_coords(local, occupied)
    occupied[placed[:, 0], placed[:, 1]] = True
    return {
        "color": int(random.choice(MOVING_COLORS)),
        "coords": placed,
        "group": int(group_id),
        "seed_col": None,
    }


def _scatter_group(final_objects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    scattered: list[dict] = []
    objs = sorted(final_objects, key=lambda obj: int(obj["coords"][:, 0].min()))
    next_top = random.randint(1, 2)
    for obj in objs:
        r0, c0, r1, c1 = _bbox(obj["coords"])
        height = int(r1 - r0 + 1)
        max_top = int(r0) - 1
        if next_top > max_top:
            return None
        top = random.randint(next_top, min(max_top, next_top + 2))
        moved = _shift_to_row(obj["coords"], top)
        if moved[:, 0].max() >= shape[0] - 1:
            return None
        scattered.append({**obj, "coords": moved})
        next_top = int(top) + int(height) + random.randint(1, 2)
    return scattered


def _choose_seed_rows(
    shape: tuple[int, int],
    seed_specs: list[tuple[int, int]],
    object_mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, _w = shape
    placements: list[tuple[int, int]] = []
    taken: set[tuple[int, int]] = set()
    for index, (seed_col, radius) in enumerate(seed_specs):
        style_roll = random.random()
        if style_roll < 0.35:
            preferred = list(range(1, min(h - 1, radius + 2)))
        elif style_roll < 0.70:
            preferred = list(range(max(1, h - radius - 4), h - 1))
        else:
            mid_lo = max(1, h // 3 - 1)
            mid_hi = min(h - 2, (2 * h) // 3 + 1)
            preferred = list(range(mid_lo, mid_hi + 1))
        fallback = [r for r in range(1, h - 1) if r not in preferred]
        rows = preferred + fallback
        if index and random.random() < 0.4:
            rows = rows[::-1]

        chosen = None
        for row in rows:
            if object_mask[int(row), int(seed_col)]:
                continue
            if (int(row), int(seed_col)) in taken:
                continue
            chosen = (int(row), int(seed_col))
            break
        if chosen is None:
            return None
        placements.append(chosen)
        taken.add(chosen)
    return placements


def _render_grid(shape: tuple[int, int], objects: list[dict], seeds: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[-1, :] = WALL
    for obj in objects:
        grid[obj["coords"][:, 0], obj["coords"][:, 1]] = int(obj["color"])
    for r, c in seeds:
        grid[int(r), int(c)] = SEED
    return grid


def _valid_input(grid: np.ndarray, total_objects: int, seed_cols: list[int]) -> bool:
    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False
    if not np.all(grid[-1, :] == WALL):
        return False
    if len(set(seed_cols)) != len(seed_cols):
        return False
    comps = _components_by_color(grid, MOVING_COLORS)
    if len(comps) != int(total_objects):
        return False
    for comp in comps:
        if not (2 <= len(comp["coords"]) <= 4):
            return False
    stacked_grid, settled = _settle_objects(grid)
    if np.any(stacked_grid[-1, :] != WALL):
        return False
    for col in seed_cols:
        height, seq = _seed_stack_profile(settled, int(col))
        if not (2 <= height <= 4):
            return False
        if len(seq) < 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 19)
    width = random.randint(15, 20)
    seed_count = random.choices((1, 2, 3), weights=(3, 5, 2), k=1)[0]
    if seed_count == 3:
        width = max(width, 17)

    seed_cols = _sample_distinct_columns(width, seed_count)
    if seed_cols is None:
        return None

    min_objects = max(4, 2 * seed_count)
    max_objects = min(7, 2 * seed_count + 2)
    total_objects = random.randint(min_objects, max_objects)

    seed_group_sizes = [2] * seed_count
    remaining = total_objects - 2 * seed_count
    growable = list(range(seed_count))
    random.shuffle(growable)
    while remaining > 0 and growable:
        idx = growable.pop()
        seed_group_sizes[idx] += 1
        remaining -= 1
    extra_objects = remaining

    occupied = np.zeros((height, width), dtype=bool)
    occupied[-1, :] = True
    final_objects: list[dict] = []
    blocked_cols: set[int] = set()

    for group_id, (seed_col, group_size) in enumerate(zip(seed_cols, seed_group_sizes)):
        target_heights = [4, 3] if group_size == 3 else [4, 3, 2]
        random.shuffle(target_heights)
        placed_group = None
        for target_height in target_heights:
            trial_occ = occupied.copy()
            trial = _place_seed_group(trial_occ, int(seed_col), int(group_size), int(target_height), group_id)
            if trial is not None:
                occupied = trial_occ
                placed_group = trial
                break
        if placed_group is None:
            return None
        final_objects.extend(placed_group)
        for obj in placed_group:
            _r0, c0, _r1, c1 = _bbox(obj["coords"])
            for col in range(c0 - 1, c1 + 2):
                if 0 <= col < width:
                    blocked_cols.add(int(col))

    if extra_objects > 0:
        extra_cols = _sample_extra_columns(width, blocked_cols, extra_objects)
        if extra_cols is None:
            return None
        for offset, col in enumerate(extra_cols):
            obj = _place_extra_object(occupied, int(col), seed_count + offset)
            if obj is None:
                return None
            final_objects.append(obj)

    groups: dict[int, list[dict]] = {}
    for obj in final_objects:
        groups.setdefault(int(obj["group"]), []).append(obj)

    scattered_objects: list[dict] = []
    for group_objects in groups.values():
        scattered = _scatter_group(group_objects, (height, width))
        if scattered is None:
            return None
        scattered_objects.extend(scattered)

    object_mask = _occupied_with_objects((height, width), scattered_objects)
    seed_specs = []
    for seed_col in seed_cols:
        radius, _seq = _seed_stack_profile(final_objects, int(seed_col))
        seed_specs.append((int(seed_col), int(radius)))
    seeds = _choose_seed_rows((height, width), seed_specs, object_mask)
    if seeds is None:
        return None

    inp = _render_grid((height, width), scattered_objects, seeds)
    if not _valid_input(inp, total_objects, seed_cols):
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(out == SEED) != seed_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 13/029/1")
