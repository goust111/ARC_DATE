from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
WALL = 8
PARTICLE = 6
PATH = 3
COUNT = 7

_RIGHT = (0, 1)
_DOWN = (1, 0)
_DIRS = (_RIGHT, _DOWN)

_GROUP_CHOICES = (3, 4, 5)
_GROUP_WEIGHTS = (4, 4, 2)
_SIZE_RANGE = (3, 6)
_GRID_RANGE = (18, 24)
_GEN_TRIES = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_share_row_or_col(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _merge_fragment_groups(components: list[dict]) -> list[dict]:
    groups = [{"coords": comp["coords"].copy(), "bbox": comp["bbox"]} for comp in components]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if _bboxes_share_row_or_col(groups[i]["bbox"], groups[j]["bbox"]):
                    merged_coords = np.vstack([groups[i]["coords"], groups[j]["coords"]])
                    groups[i] = {"coords": merged_coords, "bbox": _bbox_coords(merged_coords)}
                    del groups[j]
                    changed = True
                    break
    groups.sort(key=lambda group: tuple(group["bbox"]))
    return groups


def _draw_hollow_box(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int = WALL) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _planned_wall_grid(shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for bbox in bboxes:
        _draw_hollow_box(grid, bbox, WALL)
    return grid


def _trace_particle(
    walls: np.ndarray, start: tuple[int, int]
) -> dict[str, object]:
    h, w = walls.shape
    r, c = map(int, start)
    direction = 0
    path = [(r, c)]
    collisions = 0
    first_collision: str | None = None
    stopped_after_turn = False

    for _ in range(h * w * 2):
        dr, dc = _DIRS[direction]
        nr, nc = r + dr, c + dc
        hit_boundary = not (0 <= nr < h and 0 <= nc < w)
        blocked = hit_boundary or bool(walls[nr, nc])
        if not blocked:
            r, c = int(nr), int(nc)
            path.append((r, c))
            continue

        collisions += 1
        if first_collision is None:
            first_collision = "boundary" if hit_boundary else "wall"

        direction = 1 - direction
        dr, dc = _DIRS[direction]
        nr, nc = r + dr, c + dc
        turned_hit_boundary = not (0 <= nr < h and 0 <= nc < w)
        if turned_hit_boundary or bool(walls[nr, nc]):
            stopped_after_turn = True
            break

        r, c = int(nr), int(nc)
        path.append((r, c))
    else:
        raise AssertionError("particle trace exceeded the monotone path bound")

    return {
        "path": path,
        "collisions": int(collisions),
        "first_collision": first_collision,
        "stopped_after_turn": bool(stopped_after_turn),
    }


def _render_from_bboxes(
    shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]], particle_rows: list[int]
) -> np.ndarray:
    out = _planned_wall_grid(shape, bboxes)
    walls = out == WALL
    total_collisions = 0
    for row in particle_rows:
        info = _trace_particle(walls, (int(row), 0))
        total_collisions += int(info["collisions"])
        for r, c in info["path"]:
            out[int(r), int(c)] = PATH
    out[: min(total_collisions, shape[0]), shape[1] - 1] = COUNT
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Merge teal fragments into hollow walls, then trace particles and encode total collisions.

    Input structure:
    - Black background with several disconnected teal fragment components and 2-3 magenta particles.
    - Each teal fragment is a 4-connected 1-3 pixel shard; shards from the same logical wall share row-span or column-span overlap transitively.
    - Every particle starts on the left boundary and all meaningful obstacles come from the reconstructed teal walls.

    Rule:
    - Find teal 4-connected fragments and merge them transitively whenever two fragment bounding boxes overlap in row span or in column span.
    - Replace every merged group by the 1-pixel-thick teal border of its full bounding box, leaving the box interior black.
    - For every magenta particle, move right through empty cells; when the next cell is a teal wall or the grid boundary, count one collision and turn clockwise (right to down, down to right). If the turned next cell is still blocked, stop; otherwise continue. Paint every visited cell green.
    - Count all particle collisions and fill that many cells from the top of the rightmost column with orange, overwriting anything already there.

    Key insight:
    - The fragments do not describe the corridor directly: only after span-based fragment merging and hollow-box reconstruction does the obstacle layout become fixed enough for the clockwise collision trace to determine the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    fragments = _find_color_components(g, WALL)
    groups = _merge_fragment_groups(fragments)
    particle_rows = [int(r) for r in range(h) if int(g[r, 0]) == PARTICLE]

    out = np.full((h, w), BG, dtype=int)
    for group in groups:
        _draw_hollow_box(out, group["bbox"], WALL)

    walls = out == WALL
    total_collisions = 0
    for row in particle_rows:
        info = _trace_particle(walls, (int(row), 0))
        total_collisions += int(info["collisions"])
        for r, c in info["path"]:
            out[int(r), int(c)] = PATH

    out[: min(total_collisions, h), w - 1] = COUNT
    return out


def _random_partition(total: int, slots: int) -> list[int]:
    parts = [0] * int(slots)
    for _ in range(int(total)):
        parts[random.randrange(int(slots))] += 1
    return parts


def _place_monotone(
    lengths: list[int], total: int, lead_min: int, between_min: int, trail_min: int
) -> list[int] | None:
    need = lead_min + trail_min + sum(lengths) + between_min * (len(lengths) - 1)
    if need > total:
        return None
    extra = total - need
    extras = _random_partition(extra, len(lengths) + 1)

    positions: list[int] = []
    cur = lead_min + extras[0]
    for i, length in enumerate(lengths):
        positions.append(int(cur))
        cur += int(length)
        if i < len(lengths) - 1:
            cur += between_min + extras[i + 1]
    return positions


def _sample_bboxes() -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    for _ in range(200):
        n_groups = random.choices(_GROUP_CHOICES, weights=_GROUP_WEIGHTS, k=1)[0]
        heights = [random.randint(*_SIZE_RANGE) for _ in range(n_groups)]
        widths = [random.randint(*_SIZE_RANGE) for _ in range(n_groups)]

        min_h = max(_GRID_RANGE[0], sum(heights) + 2 * (n_groups - 1) + 2)
        min_w = max(_GRID_RANGE[0], sum(widths) + 2 * (n_groups - 1) + 4)
        if min_h > _GRID_RANGE[1] or min_w > _GRID_RANGE[1]:
            continue

        h = random.randint(min_h, _GRID_RANGE[1])
        w = random.randint(min_w, _GRID_RANGE[1])
        row_starts = _place_monotone(heights, h, lead_min=1, between_min=2, trail_min=1)
        col_starts = _place_monotone(widths, w, lead_min=3, between_min=2, trail_min=1)
        if row_starts is None or col_starts is None:
            continue

        bboxes = []
        for r0, c0, hh, ww in zip(row_starts, col_starts, heights, widths):
            bboxes.append((int(r0), int(c0), int(r0 + hh - 1), int(c0 + ww - 1)))

        ok = True
        for a, b in combinations(bboxes, 2):
            if _bboxes_share_row_or_col(a, b):
                ok = False
                break
            if int(b[0]) - int(a[2]) < 3 or int(b[1]) - int(a[3]) < 3:
                ok = False
                break
        if ok:
            return h, w, bboxes
    return None


def _group_fragments(bbox: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, bbox)
    return [
        [(r0, c0)],
        [(r0, c1)],
        [(r1, c0)],
        [(r1, c1)],
    ]


def _particle_rows_for_layout(
    h: int, bboxes: list[tuple[int, int, int, int]]
) -> list[int] | None:
    walls = _planned_wall_grid((h, max(b[3] for b in bboxes) + 2), bboxes)
    top_row = 0
    bottom_row = h - 1

    wall_candidates = sorted(
        {
            int((bbox[0] + bbox[2]) // 2)
            for bbox in bboxes
            if 0 < int((bbox[0] + bbox[2]) // 2) < h - 1
        }
    )
    random.shuffle(wall_candidates)

    top_info = _trace_particle(walls, (top_row, 0))
    bottom_info = _trace_particle(walls, (bottom_row, 0))
    if top_info["first_collision"] != "boundary" or int(top_info["collisions"]) != 2:
        return None
    if bottom_info["first_collision"] != "boundary" or int(bottom_info["collisions"]) != 1:
        return None

    wall_row = None
    for candidate in wall_candidates:
        info = _trace_particle(walls, (candidate, 0))
        if info["first_collision"] == "wall" and int(info["collisions"]) >= 3:
            wall_row = int(candidate)
            break
    if wall_row is None:
        return None

    if random.random() < 0.65:
        return [top_row, wall_row, bottom_row]
    if random.random() < 0.5:
        return [top_row, wall_row]
    return [wall_row, bottom_row]


def _validate_input(
    inp: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    particle_rows: list[int],
) -> bool:
    h, w = inp.shape
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int(np.count_nonzero(inp == PARTICLE)) != len(particle_rows):
        return False
    if any(int(inp[row, 0]) != PARTICLE for row in particle_rows):
        return False
    if np.any((inp == PARTICLE) & (np.arange(w)[None, :] != 0)):
        return False

    fragments = _find_color_components(inp, WALL)
    if not (10 <= len(fragments) <= 20):
        return False
    if any(not (1 <= int(comp["area"]) <= 3) for comp in fragments):
        return False

    merged = _merge_fragment_groups(fragments)
    got = sorted(tuple(group["bbox"]) for group in merged)
    want = sorted(tuple(bbox) for bbox in bboxes)
    if got != want:
        return False

    for a, b in combinations(want, 2):
        if _bboxes_share_row_or_col(a, b):
            return False
        if int(b[0]) - int(a[2]) < 3 or int(b[1]) - int(a[3]) < 3:
            return False

    walls = _planned_wall_grid((h, w), bboxes)
    selected_infos = [_trace_particle(walls, (row, 0)) for row in particle_rows]
    if not any(info["first_collision"] == "wall" for info in selected_infos):
        return False
    if not any(info["first_collision"] == "boundary" for info in selected_infos):
        return False
    if not any(bool(info["stopped_after_turn"]) for info in selected_infos):
        return False

    if 0 in particle_rows:
        info = _trace_particle(walls, (0, 0))
        if info["first_collision"] != "boundary" or int(info["collisions"]) != 2:
            return False
    if (h - 1) in particle_rows:
        info = _trace_particle(walls, (h - 1, 0))
        if info["first_collision"] != "boundary" or int(info["collisions"]) != 1:
            return False

    expected = _render_from_bboxes((h, w), bboxes, particle_rows)
    produced = main(inp)
    return bool(np.array_equal(expected, produced))


def _build_candidate() -> np.ndarray | None:
    layout = _sample_bboxes()
    if layout is None:
        return None
    h, w, bboxes = layout
    particle_rows = _particle_rows_for_layout(h, bboxes)
    if particle_rows is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for bbox in bboxes:
        for fragment in _group_fragments(bbox):
            for r, c in fragment:
                grid[int(r), int(c)] = WALL

    for row in particle_rows:
        if int(grid[row, 0]) != BG:
            return None
        grid[int(row), 0] = PARTICLE

    if not _validate_input(grid, bboxes, particle_rows):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_TRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
