from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

SOURCE_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS4 = (UP, DOWN, LEFT, RIGHT)
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

EDGE_TO_DIR = {"top": DOWN, "bottom": UP, "left": RIGHT, "right": LEFT}
EDGES = tuple(EDGE_TO_DIR)

MIN_GRID = 16
MAX_GRID = 28
HIT_COUNT_CHOICES = (1, 2, 2, 3)
MISS_COUNT_CHOICES = (1, 1, 2)
UNUSED_TARGET_CHOICES = (0, 1, 1, 2)
MIN_SEG_LEN = 2
MAX_SEG_LEN = 7
RECT_SIZE_RANGE = (2, 4)
MAX_GENERATE_ATTEMPTS = 500


def _in_bounds(height: int, width: int, r: int, c: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def _is_boundary(height: int, width: int, r: int, c: int) -> bool:
    return r in (0, height - 1) or c in (0, width - 1)


def _is_corner(height: int, width: int, r: int, c: int) -> bool:
    return r in (0, height - 1) and c in (0, width - 1)


def _is_interior(height: int, width: int, r: int, c: int) -> bool:
    return 0 < r < height - 1 and 0 < c < width - 1


def _neighbors(mask: np.ndarray, r: int, c: int, connectivity: int):
    dirs = DIRS4 if connectivity == 4 else DIRS8
    height, width = mask.shape
    for dr, dc in dirs:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    result: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(mask, rr, cc, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            result.append(cells)
    return result


def _source_direction(height: int, width: int, r: int, c: int) -> tuple[int, int]:
    assert _is_boundary(height, width, r, c), "source must lie on the grid boundary"
    assert not _is_corner(height, width, r, c), "corner sources are ambiguous"
    if r == 0:
        return DOWN
    if r == height - 1:
        return UP
    if c == 0:
        return RIGHT
    return LEFT


def _turn_dir(direction: tuple[int, int], sign: int) -> tuple[int, int]:
    dr, dc = direction
    assert sign in (-1, 1)
    if dr != 0:
        return LEFT if sign < 0 else RIGHT
    return UP if sign < 0 else DOWN


def _mirror_kind(incoming: tuple[int, int], outgoing: tuple[int, int]) -> str:
    dr, dc = incoming
    if outgoing == (dc, dr):
        return "\\"
    if outgoing == (-dc, -dr):
        return "/"
    raise AssertionError("incoming and outgoing directions are not a valid 45-degree reflection pair")


def _reflect(direction: tuple[int, int], kind: str) -> tuple[int, int]:
    dr, dc = direction
    if kind == "\\":
        return dc, dr
    if kind == "/":
        return -dc, -dr
    raise AssertionError(f"unknown mirror kind: {kind}")


def _distance_to_edge(height: int, width: int, pos: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    if dr == 1:
        return height - 1 - r
    if dr == -1:
        return r
    if dc == 1:
        return width - 1 - c
    return c


def _walk(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _mark_cells(mask: np.ndarray, cells: list[tuple[int, int]], pad: int = 0) -> None:
    height, width = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(height - 1, r + pad) + 1):
            for cc in range(max(0, c - pad), min(width - 1, c + pad) + 1):
                mask[rr, cc] = True


def _mirror_map(grid: np.ndarray) -> dict[tuple[int, int], str]:
    g = np.asarray(grid, dtype=int)
    mirrors: dict[tuple[int, int], str] = {}
    for cells in _components(g == GREY, connectivity=8):
        assert 2 <= len(cells) <= 4, "each mirror must be a short diagonal segment of length 2-4"
        coords = sorted(cells)
        diff_keys = {r - c for r, c in coords}
        sum_keys = {r + c for r, c in coords}
        if len(diff_keys) == 1:
            kind = "\\"
            ordered = sorted(coords)
            for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
                assert (r1, c1) == (r0 + 1, c0 + 1), "backslash mirror pixels must be consecutive"
        else:
            assert len(sum_keys) == 1, "grey component is not a straight 45-degree mirror"
            kind = "/"
            ordered = sorted(coords)
            for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
                assert (r1, c1) == (r0 + 1, c0 - 1), "slash mirror pixels must be consecutive"
        for cell in cells:
            mirrors[cell] = kind
    return mirrors


def _target_components(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    targets: list[list[tuple[int, int]]] = []
    cell_to_target: dict[tuple[int, int], int] = {}

    for cells in _components(g == MAROON, connectivity=4):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1), "targets must be solid rectangles"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == MAROON), "target rectangle must be monochrome maroon"
        idx = len(targets)
        targets.append(cells)
        for cell in cells:
            cell_to_target[cell] = idx
    return targets, cell_to_target


def _sources(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    result: list[tuple[int, int, int]] = []
    for r in range(height):
        for c in range(width):
            if not _is_boundary(height, width, r, c):
                continue
            color = int(g[r, c])
            if color in (BLACK, GREY, MAROON):
                continue
            assert not _is_corner(height, width, r, c), "corner sources are not supported"
            result.append((r, c, color))
    return result


def _trace_ray(
    grid: np.ndarray,
    mirrors: dict[tuple[int, int], str],
    target_of: dict[tuple[int, int], int],
    source: tuple[int, int, int],
) -> tuple[int | None, int]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    r, c, _color = source
    dr, dc = _source_direction(height, width, r, c)
    reflections = 0
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        if state in seen:
            return None, reflections
        seen.add(state)

        nr = r + dr
        nc = c + dc
        if not _in_bounds(height, width, nr, nc):
            return None, reflections
        if (nr, nc) in target_of:
            return target_of[(nr, nc)], reflections
        if (nr, nc) in mirrors:
            dr, dc = _reflect((dr, dc), mirrors[(nr, nc)])
            reflections += 1
            r, c = nr, nc
            continue

        r, c = nr, nc
        if _is_boundary(height, width, r, c):
            return None, reflections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each maroon target rectangle with the color of the reflected boundary ray that eventually hits it.

    Input structure:
    - Black background fills the grid.
    - Some boundary cells are single-pixel colored light sources; each source sits on exactly one outer edge.
    - Grey pixels form short 45-degree diagonal segments of length 2-4, which act as mirrors.
    - Maroon pixels form solid rectangular target regions away from the boundary.

    Rule:
    - Each source emits a ray straight inward, perpendicular to its boundary edge.
    - When the ray enters a grey mirror cell, it reflects according to that mirror's slash or backslash orientation and continues.
    - The ray stops when it reaches the grid boundary again or when it first enters a maroon target rectangle.
    - If a source ray stops by entering a maroon target, recolor the entire target rectangle to that source color; the ray path itself stays invisible and grey mirrors remain unchanged.

    Key insight:
    - The task is pure ray tracing on a grid: identify which grey diagonal each ray hits, update the direction after every reflection, and transfer the source color only to the final target component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mirrors = _mirror_map(g)
    targets, target_of = _target_components(g)
    sources = _sources(g)

    out = g.copy()
    for source in sources:
        target_id, _reflections = _trace_ray(g, mirrors, target_of, source)
        if target_id is None:
            continue
        for r, c in targets[target_id]:
            out[r, c] = int(source[2])
    return out


def _cells_clear(
    reserved: np.ndarray,
    cells: list[tuple[int, int]],
    local_used: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if reserved[r, c] or (r, c) in local_used:
            return False
    return True


def _sample_source_cell(
    height: int,
    width: int,
    reserved: np.ndarray,
    edge: str,
) -> tuple[int, int] | None:
    if edge in ("top", "bottom"):
        r = 0 if edge == "top" else height - 1
        cols = [c for c in range(1, width - 1) if not reserved[r, c]]
        if not cols:
            return None
        return r, random.choice(cols)
    c = 0 if edge == "left" else width - 1
    rows = [r for r in range(1, height - 1) if not reserved[r, c]]
    if not rows:
        return None
    return random.choice(rows), c


def _sample_target(
    height: int,
    width: int,
    reserved: np.ndarray,
    local_used: set[tuple[int, int]],
    current: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, object] | None:
    for _ in range(140):
        rect_h = random.randint(*RECT_SIZE_RANGE)
        rect_w = random.randint(*RECT_SIZE_RANGE)
        gap = random.randint(2, 6)
        entry = (current[0] + direction[0] * gap, current[1] + direction[1] * gap)

        if direction in (UP, DOWN):
            off = random.randint(0, rect_w - 1)
            c0 = entry[1] - off
            c1 = c0 + rect_w - 1
            if direction == DOWN:
                r0 = entry[0]
                r1 = entry[0] + rect_h - 1
            else:
                r0 = entry[0] - rect_h + 1
                r1 = entry[0]
        else:
            off = random.randint(0, rect_h - 1)
            r0 = entry[0] - off
            r1 = r0 + rect_h - 1
            if direction == RIGHT:
                c0 = entry[1]
                c1 = entry[1] + rect_w - 1
            else:
                c0 = entry[1] - rect_w + 1
                c1 = entry[1]

        if not (1 <= r0 <= r1 <= height - 2 and 1 <= c0 <= c1 <= width - 2):
            continue

        approach = _walk(current, direction, gap - 1)
        if not _cells_clear(reserved, approach, local_used):
            continue

        target_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if not _cells_clear(reserved, target_cells, local_used):
            continue

        return {"approach": approach, "target": target_cells}
    return None


def _place_mirror_segment(
    height: int,
    width: int,
    turn_cell: tuple[int, int],
    kind: str,
    path_blocked: np.ndarray,
    shape_blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    r, c = turn_cell
    if shape_blocked[r, c]:
        return None
    candidates: list[list[tuple[int, int]]] = []
    for length in range(2, 5):
        for idx in range(length):
            if kind == "\\":
                r0 = r - idx
                c0 = c - idx
                cells = [(r0 + step, c0 + step) for step in range(length)]
            else:
                r0 = r - idx
                c0 = c + idx
                cells = [(r0 + step, c0 - step) for step in range(length)]
            if any(not _is_interior(height, width, rr, cc) for rr, cc in cells):
                continue
            ok = True
            for rr, cc in cells:
                if (rr, cc) == turn_cell:
                    continue
                if path_blocked[rr, cc] or shape_blocked[rr, cc]:
                    ok = False
                    break
            if ok:
                candidates.append(cells)
    if not candidates:
        return None
    return random.choice(candidates)


def _plan_system(
    height: int,
    width: int,
    reserved: np.ndarray,
    turns: int,
    hit_target: bool,
) -> dict[str, object] | None:
    for _ in range(220):
        edge = random.choice(EDGES)
        source = _sample_source_cell(height, width, reserved, edge)
        if source is None:
            continue

        direction = EDGE_TO_DIR[edge]
        current = source
        local_used = {source}
        path_cells = [source]
        turn_specs: list[tuple[tuple[int, int], str]] = []
        ok = True

        for turn_index in range(turns):
            max_len = min(MAX_SEG_LEN, _distance_to_edge(height, width, current, direction) - 1)
            if max_len < MIN_SEG_LEN:
                ok = False
                break

            choices: list[tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], str]] = []
            signs = [-1, 1]
            random.shuffle(signs)
            lengths = list(range(MIN_SEG_LEN, max_len + 1))
            random.shuffle(lengths)

            for length in lengths:
                seg = _walk(current, direction, length)
                if not _cells_clear(reserved, seg, local_used):
                    continue
                turn_cell = seg[-1]
                for sign in signs:
                    next_dir = _turn_dir(direction, sign)
                    remaining_turns = turns - turn_index - 1
                    room = _distance_to_edge(height, width, turn_cell, next_dir)
                    if room < remaining_turns + 2:
                        continue
                    choices.append((seg, turn_cell, next_dir, _mirror_kind(direction, next_dir)))
            if not choices:
                ok = False
                break

            seg, turn_cell, next_dir, kind = random.choice(choices)
            path_cells.extend(seg)
            local_used.update(seg)
            turn_specs.append((turn_cell, kind))
            current = turn_cell
            direction = next_dir

        if not ok:
            continue

        target_cells: list[tuple[int, int]] = []
        if hit_target:
            sampled = _sample_target(height, width, reserved, local_used, current, direction)
            if sampled is None:
                continue
            approach = list(sampled["approach"])
            target_cells = list(sampled["target"])
            path_cells.extend(approach)
            local_used.update(approach)
        else:
            exit_len = _distance_to_edge(height, width, current, direction)
            if exit_len < 2:
                continue
            exit_cells = _walk(current, direction, exit_len)
            exit_cell = exit_cells[-1]
            if _is_corner(height, width, exit_cell[0], exit_cell[1]):
                continue
            if not _cells_clear(reserved, exit_cells, local_used):
                continue
            path_cells.extend(exit_cells)
            local_used.update(exit_cells)

        path_blocked = np.array(reserved, copy=True)
        _mark_cells(path_blocked, path_cells)
        shape_blocked = np.array(reserved, copy=True)
        if target_cells:
            _mark_cells(shape_blocked, target_cells, pad=1)

        mirror_segments: list[list[tuple[int, int]]] = []
        for turn_cell, kind in turn_specs:
            segment = _place_mirror_segment(height, width, turn_cell, kind, path_blocked, shape_blocked)
            if segment is None:
                ok = False
                break
            mirror_segments.append(segment)
            _mark_cells(shape_blocked, segment, pad=1)

        if not ok:
            continue

        return {
            "source": source,
            "path": path_cells,
            "target": target_cells,
            "mirrors": mirror_segments,
            "mirror_kinds": [kind for _turn_cell, kind in turn_specs],
        }
    return None


def _paint_system(grid: np.ndarray, reserved: np.ndarray, spec: dict[str, object], color: int) -> None:
    source = tuple(spec["source"])
    path_cells = list(spec["path"])
    target_cells = list(spec["target"])
    mirrors = list(spec["mirrors"])

    grid[source] = color
    _mark_cells(reserved, [source], pad=1)
    _mark_cells(reserved, path_cells)

    if target_cells:
        for r, c in target_cells:
            grid[r, c] = MAROON
        _mark_cells(reserved, target_cells, pad=1)

    for segment in mirrors:
        for r, c in segment:
            grid[r, c] = GREY
        _mark_cells(reserved, segment, pad=1)


def _place_unused_target(grid: np.ndarray, reserved: np.ndarray) -> bool:
    height, width = grid.shape
    for _ in range(180):
        rect_h = random.randint(*RECT_SIZE_RANGE)
        rect_w = random.randint(*RECT_SIZE_RANGE)
        r0 = random.randint(1, height - rect_h - 2)
        c0 = random.randint(1, width - rect_w - 2)
        cells = [(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)]
        if any(reserved[r, c] for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = MAROON
        _mark_cells(reserved, cells, pad=1)
        return True
    return False


def _place_decoy_mirror(grid: np.ndarray, reserved: np.ndarray, kind: str) -> bool:
    height, width = grid.shape
    candidates: list[list[tuple[int, int]]] = []
    for length in range(2, 5):
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if kind == "\\":
                    cells = [(r + step, c + step) for step in range(length)]
                else:
                    cells = [(r + step, c - step) for step in range(length)]
                if any(not _is_interior(height, width, rr, cc) for rr, cc in cells):
                    continue
                if any(reserved[rr, cc] for rr, cc in cells):
                    continue
                candidates.append(cells)
    if not candidates:
        return False

    cells = random.choice(candidates)
    for r, c in cells:
        grid[r, c] = GREY
    _mark_cells(reserved, cells, pad=1)
    return True


def _scene_analysis(grid: np.ndarray) -> list[dict[str, int | None]]:
    mirrors = _mirror_map(grid)
    _targets, target_of = _target_components(grid)
    result: list[dict[str, int | None]] = []
    for r, c, color in _sources(grid):
        target_id, reflections = _trace_ray(grid, mirrors, target_of, (r, c, color))
        result.append(
            {
                "row": r,
                "col": c,
                "color": color,
                "target": target_id,
                "reflections": reflections,
            }
        )
    return result


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_GRID, MAX_GRID)
    width = random.randint(MIN_GRID, MAX_GRID)
    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    hit_count = random.choice(HIT_COUNT_CHOICES)
    miss_count = random.choice(MISS_COUNT_CHOICES)
    unused_targets = random.choice(UNUSED_TARGET_CHOICES)
    total_sources = hit_count + miss_count
    if total_sources > len(SOURCE_COLORS):
        return None

    colors = random.sample(SOURCE_COLORS, total_sources)
    specs: list[tuple[dict[str, object], int]] = []
    used_mirror_kinds: set[str] = set()

    hit_turns = [random.choice((1, 2, 3))]
    hit_turns.extend(random.choice((0, 1, 2, 3)) for _ in range(hit_count - 1))
    random.shuffle(hit_turns)

    miss_turns = [0]
    miss_turns.extend(random.choice((0, 1, 2)) for _ in range(miss_count - 1))
    random.shuffle(miss_turns)

    color_idx = 0
    for turns in hit_turns:
        spec = _plan_system(height, width, reserved, turns=turns, hit_target=True)
        if spec is None:
            return None
        specs.append((spec, colors[color_idx]))
        used_mirror_kinds.update(spec["mirror_kinds"])
        _paint_system(grid, reserved, spec, colors[color_idx])
        color_idx += 1

    for turns in miss_turns:
        spec = _plan_system(height, width, reserved, turns=turns, hit_target=False)
        if spec is None:
            return None
        specs.append((spec, colors[color_idx]))
        used_mirror_kinds.update(spec["mirror_kinds"])
        _paint_system(grid, reserved, spec, colors[color_idx])
        color_idx += 1

    for _ in range(unused_targets):
        if not _place_unused_target(grid, reserved):
            return None

    for kind in ("\\", "/"):
        if kind not in used_mirror_kinds and not _place_decoy_mirror(grid, reserved, kind):
            return None

    analysis = _scene_analysis(grid)
    if len(analysis) != total_sources:
        return None

    hit_targets = [item["target"] for item in analysis if item["target"] is not None]
    if len(hit_targets) < hit_count or len(hit_targets) == 0:
        return None
    if len(hit_targets) != len(set(hit_targets)):
        return None
    if not any(item["target"] is None for item in analysis):
        return None
    if not any(int(item["reflections"]) >= 1 for item in analysis):
        return None
    if not any(int(item["reflections"]) == 0 for item in analysis):
        return None

    if not _target_components(grid)[0]:
        return None
    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid reflected-ray puzzle after {MAX_GENERATE_ATTEMPTS} attempts")
