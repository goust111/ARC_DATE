from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))

    return components


def _emitter_from_position(height: int, width: int, r: int, c: int, color: int) -> dict:
    borders = int(r == 0) + int(r == height - 1) + int(c == 0) + int(c == width - 1)
    assert borders == 1, "each emitter must lie on exactly one border edge"

    if r == 0:
        axis = "col"
        line = c
    elif r == height - 1:
        axis = "col"
        line = c
    elif c == 0:
        axis = "row"
        line = r
    else:
        axis = "row"
        line = r

    return {"pos": (int(r), int(c)), "axis": axis, "line": int(line), "color": int(color)}


def _winner_for_object(coords: np.ndarray, emitters: list[dict]) -> int:
    rows = {int(r) for r in coords[:, 0]}
    cols = {int(c) for c in coords[:, 1]}
    counts: Counter[int] = Counter()
    best_pos: dict[int, tuple[int, int]] = {}

    for emitter in emitters:
        hits = emitter["line"] in rows if emitter["axis"] == "row" else emitter["line"] in cols
        if not hits:
            continue
        color = int(emitter["color"])
        counts[color] += 1
        pos = emitter["pos"]
        if color not in best_pos or pos < best_pos[color]:
            best_pos[color] = pos

    assert counts, "every object must be covered by at least one emitter ray"
    max_votes = max(counts.values())
    tied = [color for color, votes in counts.items() if votes == max_votes]
    return min(tied, key=lambda color: best_pos[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each interior gray object by the majority color of the invisible edge rays that pass through it.

    Input structure:
    - The most common color is the background, and every non-background border pixel is a one-cell emitter.
    - Each emitter lies on exactly one edge and defines a full-row or full-column ray along that edge's inward normal.
    - Inside the grid there are several small 4-connected gray objects that do not touch the border.

    Rule:
    - Treat every non-background border pixel as an emitter and extend its same-color logical ray across the whole grid.
    - Find each 4-connected interior gray object and count how many emitter rays of each color cross any pixel of that object.
    - Recolor the whole object to the color with the largest ray count; if several colors tie, choose the one whose contributing emitter is earliest in row-major border order.

    Key insight:
    - The rays are never drawn, so the transformation comes from inferring row and column coverage from border emitters and then doing a per-object color vote.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 5 and width >= 5, "grid is too small for border emitters and interior objects"

    bg = _most_common_color(g)
    border_mask = np.zeros_like(g, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    emitters: list[dict] = []
    for r, c in np.argwhere(border_mask & (g != bg)):
        emitters.append(_emitter_from_position(height, width, int(r), int(c), int(g[r, c])))
    assert emitters, "expected at least one border emitter"

    interior_non_bg = (~border_mask) & (g != bg)
    object_colors = {int(v) for v in np.unique(g[interior_non_bg])}
    assert len(object_colors) == 1, "expected exactly one interior object color"
    object_color = next(iter(object_colors))

    objects = _components4(g == object_color)
    assert objects, "expected at least one gray object"

    out = g.copy()
    for coords in objects:
        assert 2 <= len(coords) <= 6, "objects must have 2-6 pixels"
        assert np.all((coords[:, 0] > 0) & (coords[:, 0] < height - 1)), "objects must stay off the border"
        assert np.all((coords[:, 1] > 0) & (coords[:, 1] < width - 1)), "objects must stay off the border"
        winner = _winner_for_object(coords, emitters)
        out[coords[:, 0], coords[:, 1]] = winner

    return out


BG = 8
OBJECT = 5
EMITTER_COLORS = (2, 3, 4)
GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (10, 22)
OBJECT_COUNT_OPTIONS = (2, 3, 4)
GENERATION_RETRIES = 500
OBJECT_RETRIES = 120
EMITTER_ASSIGN_RETRIES = 200
MAX_SHAPE_SPAN = 4


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(area: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(random.choice(tuple(frontier)))

        shape = _normalize_cells(cells)
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        if height <= MAX_SHAPE_SPAN and width <= MAX_SHAPE_SPAN:
            return shape
    return None


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                blocked[rr, cc] = True


def _place_objects(height: int, width: int, n_objects: int) -> list[dict] | None:
    blocked = np.zeros((height, width), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    objects: list[dict] = []

    for _ in range(n_objects):
        placed = False
        for _attempt in range(OBJECT_RETRIES):
            area = random.randint(2, 6)
            shape = _random_shape(area)
            if shape is None:
                continue

            shape_h = max(r for r, _ in shape) + 1
            shape_w = max(c for _, c in shape) + 1
            if shape_h >= height - 1 or shape_w >= width - 1:
                continue

            r0 = random.randint(1, height - shape_h - 1)
            c0 = random.randint(1, width - shape_w - 1)
            cells = [(r0 + r, c0 + c) for r, c in shape]
            rows = {r for r, _ in cells}
            cols = {c for _, c in cells}

            if rows & used_rows or cols & used_cols:
                continue
            if any(blocked[r, c] for r, c in cells):
                continue

            obj = {
                "coords": np.array(cells, dtype=int),
                "rows": sorted(rows),
                "cols": sorted(cols),
            }
            objects.append(obj)
            used_rows.update(rows)
            used_cols.update(cols)
            _mark_blocked(blocked, cells)
            placed = True
            break
        if not placed:
            return None

    return objects


def _object_slots(height: int, width: int, obj: dict) -> list[dict]:
    slots: list[dict] = []
    for row in obj["rows"]:
        slots.append({"pos": (row, 0), "axis": "row", "line": int(row)})
        slots.append({"pos": (row, width - 1), "axis": "row", "line": int(row)})
    for col in obj["cols"]:
        slots.append({"pos": (0, col), "axis": "col", "line": int(col)})
        slots.append({"pos": (height - 1, col), "axis": "col", "line": int(col)})
    return slots


def _assignment_outcome(assignments: list[tuple[dict, int]]) -> tuple[int, bool]:
    counts: Counter[int] = Counter()
    best_pos: dict[int, tuple[int, int]] = {}
    for slot, color in assignments:
        counts[int(color)] += 1
        pos = slot["pos"]
        if color not in best_pos or pos < best_pos[color]:
            best_pos[int(color)] = pos
    max_votes = max(counts.values())
    tied = [color for color, votes in counts.items() if votes == max_votes]
    winner = min(tied, key=lambda color: best_pos[color])
    return int(winner), len(tied) > 1


def _sample_object_emitters(
    height: int,
    width: int,
    obj: dict,
    active_colors: tuple[int, ...],
    want_tie: bool,
) -> tuple[list[tuple[dict, int]], int] | None:
    slots = _object_slots(height, width, obj)
    max_pick = min(len(slots), 6)
    min_pick = 2 if len(active_colors) >= 2 else 1

    for _ in range(EMITTER_ASSIGN_RETRIES):
        count = random.randint(min_pick, max_pick)
        chosen = random.sample(slots, count)

        # Bias tie samples toward using at least two colors.
        if want_tie:
            palette = random.sample(active_colors, k=min(len(active_colors), random.randint(2, len(active_colors))))
        else:
            palette = list(active_colors)

        assignments = [(slot, int(random.choice(palette))) for slot in chosen]
        winner, is_tie = _assignment_outcome(assignments)

        used_colors = {color for _slot, color in assignments}
        if want_tie != is_tie:
            continue
        if want_tie and len(used_colors) < 2:
            continue
        if not want_tie and len(used_colors) == 1 and random.random() < 0.6:
            continue

        return assignments, winner

    return None


def _render_grid(height: int, width: int, objects: list[dict], emitter_specs: list[tuple[dict, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for obj in objects:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = OBJECT
    for slot, color in emitter_specs:
        r, c = slot["pos"]
        grid[r, c] = int(color)
    return grid


def _is_valid_instance(grid: np.ndarray, require_tie: bool) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    input_colors = set(map(int, np.unique(grid)))
    if BG not in input_colors or OBJECT not in input_colors:
        return False

    out_colors = set(map(int, np.unique(out)))
    if OBJECT in out_colors:
        return False

    border_mask = np.zeros_like(grid, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    emitters = [
        _emitter_from_position(grid.shape[0], grid.shape[1], int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere(border_mask & (grid != BG))
    ]
    objects = _components4(grid == OBJECT)
    tie_count = 0
    winners = set()
    for coords in objects:
        rows = {int(r) for r in coords[:, 0]}
        cols = {int(c) for c in coords[:, 1]}
        counts: Counter[int] = Counter()
        best_pos: dict[int, tuple[int, int]] = {}
        for emitter in emitters:
            hits = emitter["line"] in rows if emitter["axis"] == "row" else emitter["line"] in cols
            if not hits:
                continue
            color = int(emitter["color"])
            counts[color] += 1
            if color not in best_pos or emitter["pos"] < best_pos[color]:
                best_pos[color] = emitter["pos"]
        max_votes = max(counts.values())
        tied = [color for color, votes in counts.items() if votes == max_votes]
        tie_count += int(len(tied) > 1)
        winners.add(min(tied, key=lambda color: best_pos[color]))

    if require_tie and tie_count == 0:
        return False
    if len(winners) < 2:
        return False
    return True


def _attempt_generate() -> tuple[np.ndarray, bool] | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    n_objects = random.choice(OBJECT_COUNT_OPTIONS)
    objects = _place_objects(height, width, n_objects)
    if objects is None:
        return None

    active_colors = tuple(random.sample(EMITTER_COLORS, k=random.randint(2, len(EMITTER_COLORS))))
    require_tie = bool(random.getrandbits(1))

    tie_flags = [False] * n_objects
    if require_tie:
        n_ties = random.randint(1, max(1, n_objects - 1))
        for idx in random.sample(range(n_objects), k=n_ties):
            tie_flags[idx] = True
    random.shuffle(tie_flags)

    emitter_specs: list[tuple[dict, int]] = []
    winners: list[int] = []
    for obj, want_tie in zip(objects, tie_flags):
        sampled = _sample_object_emitters(height, width, obj, active_colors, want_tie)
        if sampled is None:
            return None
        obj_emitters, winner = sampled
        emitter_specs.extend(obj_emitters)
        winners.append(winner)

    if len(set(winners)) < 2:
        return None

    grid = _render_grid(height, width, objects, emitter_specs)
    if not _is_valid_instance(grid, require_tie):
        return None
    return grid, require_tie


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _attempt_generate()
        if result is None:
            continue
        grid, _require_tie = result
        return grid
    raise RuntimeError(f"failed to generate input after {GENERATION_RETRIES} attempts")
