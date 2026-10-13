from __future__ import annotations

import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np


try:
    ROOT = Path(__file__).resolve().parents[6]
    ARC_UTILS_DIR = ROOT / "new" / "solvers"
    if str(ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(ARC_UTILS_DIR))
    from arc_utils import Color, crop_bbox, find_components, in_bounds, paste
except Exception:
    class Color:
        BLACK = 0
        RED = 2
        GREEN = 3
        GREY = 5
        TEAL = 8

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, bbox)
        return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        out = np.asarray(dst, dtype=int)
        patch = np.asarray(patch, dtype=int)
        h, w = patch.shape
        if clip:
            rr0 = max(0, int(r0))
            cc0 = max(0, int(c0))
            rr1 = min(out.shape[0], int(r0) + h)
            cc1 = min(out.shape[1], int(c0) + w)
            pr0 = rr0 - int(r0)
            pc0 = cc0 - int(c0)
            pr1 = pr0 + rr1 - rr0
            pc1 = pc0 + cc1 - cc0
            if rr0 >= rr1 or cc0 >= cc1:
                return out
            view = patch[pr0:pr1, pc0:pc1]
            if transparent is None:
                out[rr0:rr1, cc0:cc1] = view
            else:
                mask = view != int(transparent)
                out[rr0:rr1, cc0:cc1][mask] = view[mask]
            return out
        if int(r0) < 0 or int(c0) < 0 or int(r0) + h > out.shape[0] or int(c0) + w > out.shape[1]:
            raise ValueError("paste out of bounds")
        if transparent is None:
            out[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch
        else:
            mask = patch != int(transparent)
            out[int(r0) : int(r0) + h, int(c0) : int(c0) + w][mask] = patch[mask]
        return out

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        return_format: str = "dict",
    ) -> list[dict]:
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
        assert monochromatic and connectivity == 4 and return_format == "dict"
        if colors is None:
            targets = [int(v) for v in np.unique(g) if int(v) != int(bg)]
        else:
            targets = [int(v) for v in colors if int(v) != int(bg)]
        out: list[dict] = []
        for color in targets:
            seen = np.zeros(g.shape, dtype=bool)
            for r_raw, c_raw in np.argwhere((g == color) & (~seen)):
                r = int(r_raw)
                c = int(c_raw)
                q = deque([(r, c)])
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": color,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500
PANEL_RETRIES = 400
PANEL_HEIGHT_RANGE = (5, 7)
PANEL_WIDTH_RANGE = (7, 9)
SCENE_HEIGHT_RANGE = (20, 28)
SCENE_WIDTH_RANGE = (20, 28)
EXTRA_OBSTACLE_TRIES = 60


def _on_edge(r: int, c: int, h: int, w: int) -> bool:
    return int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1


def _is_corner(r: int, c: int, h: int, w: int) -> bool:
    return (int(r) in {0, h - 1}) and (int(c) in {0, w - 1})


def _edge_direction(start: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, start)
    h, w = map(int, shape)
    choices: list[tuple[int, int]] = []
    if r == 0:
        choices.append((1, 0))
    if r == h - 1:
        choices.append((-1, 0))
    if c == 0:
        choices.append((0, 1))
    if c == w - 1:
        choices.append((0, -1))
    assert len(choices) == 1, "start must lie on exactly one panel edge"
    return choices[0]


def _extract_panels(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.TEAL],
        return_format="dict",
    )
    panels: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        panels.append((bbox, crop_bbox(grid, bbox)))
    panels.sort(key=lambda item: item[0])
    return panels


def _trace_panel(panel: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    g = np.asarray(panel, dtype=int)
    h, w = g.shape

    starts = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.GREEN)]
    ends = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.RED)]
    assert len(starts) == 1 and len(ends) == 1, "each panel must contain exactly one start and one end"

    start = starts[0]
    dr, dc = _edge_direction(start, g.shape)
    r, c = start
    turns = 0
    out = g.copy()
    path_cells = [start]
    states = [(int(r), int(c), int(dr), int(dc))]
    seen_states = {(int(r), int(c), int(dr), int(dc))}

    while True:
        nr, nc = int(r + dr), int(c + dc)
        assert in_bounds(g.shape, nr, nc), "path left the panel before reaching the end"
        nxt = int(g[nr, nc])

        if nxt == Color.RED:
            break

        if nxt == Color.TEAL:
            out[nr, nc] = Color.GREEN
            r, c = nr, nc
            state = (int(r), int(c), int(dr), int(dc))
            assert state not in seen_states, "path should not revisit the same state"
            seen_states.add(state)
            states.append(state)
            path_cells.append((int(r), int(c)))
            continue

        assert nxt == Color.GREY, "only grey obstacles may trigger turns"

        options: list[tuple[int, int]] = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = int(r + tdr), int(c + tdc)
            if in_bounds(g.shape, tr, tc) and int(g[tr, tc]) == Color.TEAL:
                options.append((int(tdr), int(tdc)))
        assert len(options) == 1, "each obstacle collision must force a unique perpendicular turn"

        dr, dc = options[0]
        turns += 1
        state = (int(r), int(c), int(dr), int(dc))
        assert state not in seen_states, "turning should not create a cycle"
        seen_states.add(state)
        states.append(state)

    assert turns > 0, "panels must contain at least one turn"
    return out, int(turns), path_cells, states


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the forced obstacle-turn path inside every teal panel and crop the panel with the most turns.

    Input structure:
    - The grid is mostly black, with 3 or 4 separated teal rectangular panels.
    - Inside each panel, teal cells form the corridor background, grey cells are turn-triggering obstacles,
      one green edge cell is the start, and one red edge cell is the end.
    - The teal pixels of each panel remain 4-connected so the panel bbox can be recovered from the teal region.

    Rule:
    - Find each teal panel and crop its bounding box.
    - For every panel, start at the green cell and move inward from its edge. Keep moving straight through teal
      cells; when the next cell is grey, turn to the unique perpendicular teal neighbor; stop when the next cell
      would be the red end cell.
    - Paint the traced corridor cells green, count how many obstacle-triggered turns occurred in that panel, and
      select the panel with the largest turn count.
    - Return only that selected panel crop, including its teal background, grey obstacles, green traced path, and
      red end marker.

    Key insight:
    - The chosen panel is determined by a dynamic path-complexity computation: the path must be fully traced in
      every panel before the maximum-turn crop can be identified.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panels = _extract_panels(g)
    assert 3 <= len(panels) <= 4, "expected 3 or 4 teal panels"

    solved_panels: list[np.ndarray] = []
    turn_counts: list[int] = []
    for _bbox, panel in panels:
        solved, turns, _cells, _states = _trace_panel(panel)
        solved_panels.append(solved)
        turn_counts.append(int(turns))

    best_turns = max(turn_counts)
    assert turn_counts.count(best_turns) == 1, "maximum turn count must be unique"
    return solved_panels[turn_counts.index(best_turns)]


def _random_edge_cell(h: int, w: int) -> tuple[int, int]:
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    if side == "left":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _rotate_dir(dr: int, dc: int, k: int) -> tuple[int, int]:
    for _ in range(int(k) % 4):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _transform_panel(panel: np.ndarray) -> np.ndarray:
    transformed = np.asarray(panel, dtype=int)
    k = random.randint(0, 3)
    transformed = np.rot90(transformed, k)
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _plan_path(
    h: int,
    w: int,
    start: tuple[int, int],
    turn_target: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]] | None:
    start_dir = _edge_direction(start, (h, w))
    start_cell = (int(start[0]), int(start[1]))

    def cell_free(
        cell: tuple[int, int],
        visited: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> bool:
        r, c = cell
        return in_bounds((h, w), r, c) and cell not in visited and cell not in blocked

    def dfs(
        r: int,
        c: int,
        dr: int,
        dc: int,
        turns_done: int,
        obstacle_count: int,
        visited: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]] | None:
        max_len = 0
        while cell_free((r + (max_len + 1) * dr, c + (max_len + 1) * dc), visited, blocked):
            max_len += 1
        lengths = list(range(1, max_len + 1))
        random.shuffle(lengths)

        if turns_done == turn_target:
            for seg_len in lengths:
                extension = [(int(r + step * dr), int(c + step * dc)) for step in range(1, seg_len + 1)]
                end = (int(r + (seg_len + 1) * dr), int(c + (seg_len + 1) * dc))
                if not in_bounds((h, w), end[0], end[1]):
                    continue
                if end in visited or end in blocked or end in extension:
                    continue
                if not _on_edge(end[0], end[1], h, w) or _is_corner(end[0], end[1], h, w):
                    continue
                if end == start_cell:
                    continue
                return visited | set(extension), set(blocked), end

        if turns_done >= turn_target:
            return None

        for seg_len in lengths:
            extension = [(int(r + step * dr), int(c + step * dc)) for step in range(1, seg_len + 1)]
            turn_cell = extension[-1]
            ahead = (int(turn_cell[0] + dr), int(turn_cell[1] + dc))
            if not in_bounds((h, w), ahead[0], ahead[1]):
                continue
            if ahead in visited or ahead in blocked or ahead in extension:
                continue

            next_dirs = [(-dc, dr), (dc, -dr)]
            random.shuffle(next_dirs)
            for ndr, ndc in next_dirs:
                chosen = (int(turn_cell[0] + ndr), int(turn_cell[1] + ndc))
                if not cell_free(chosen, visited | set(extension), blocked):
                    continue

                other = (int(turn_cell[0] - ndr), int(turn_cell[1] - ndc))
                extra_obstacles = {ahead}
                extra_cost = 1

                if in_bounds((h, w), other[0], other[1]):
                    if other in visited or other in extension:
                        continue
                    if other == start_cell:
                        pass
                    elif other in blocked:
                        pass
                    else:
                        extra_obstacles.add(other)
                        extra_cost += 1

                new_obstacle_count = obstacle_count + extra_cost
                min_remaining = turn_target - (turns_done + 1)
                if new_obstacle_count + min_remaining > 6:
                    continue

                result = dfs(
                    turn_cell[0],
                    turn_cell[1],
                    ndr,
                    ndc,
                    turns_done + 1,
                    new_obstacle_count,
                    visited | set(extension),
                    blocked | extra_obstacles,
                )
                if result is not None:
                    return result
        return None

    return dfs(
        start_cell[0],
        start_cell[1],
        start_dir[0],
        start_dir[1],
        0,
        0,
        {start_cell},
        set(),
    )


def _add_extra_obstacles(
    panel: np.ndarray,
    path_cells: set[tuple[int, int]],
    end_cell: tuple[int, int],
    min_total: int = 3,
    max_total: int = 6,
) -> np.ndarray | None:
    out = np.asarray(panel, dtype=int).copy()
    existing = {tuple(map(int, rc)) for rc in np.argwhere(out == Color.GREY)}
    target_total = random.randint(max(min_total, len(existing)), max_total)
    if len(existing) > target_total:
        return None

    forbidden = set(path_cells) | {end_cell} | {tuple(map(int, rc)) for rc in np.argwhere(out == Color.GREEN)}
    candidates = [
        (int(r), int(c))
        for r in range(out.shape[0])
        for c in range(out.shape[1])
        if int(out[r, c]) == Color.TEAL and (int(r), int(c)) not in forbidden and (int(r), int(c)) not in existing
    ]
    random.shuffle(candidates)
    for cell in candidates:
        if len(existing) >= target_total:
            break
        out[cell] = Color.GREY
        existing.add(cell)

    if len(existing) < min_total or len(existing) > max_total:
        return None
    return out


def _make_panel(turn_target: int) -> np.ndarray | None:
    for _ in range(PANEL_RETRIES):
        h = random.randint(max(PANEL_HEIGHT_RANGE[0], turn_target + 3), PANEL_HEIGHT_RANGE[1])
        w = random.randint(max(PANEL_WIDTH_RANGE[0], turn_target + 4), PANEL_WIDTH_RANGE[1])
        start = _random_edge_cell(h, w)
        planned = _plan_path(h, w, start, turn_target)
        if planned is None:
            continue
        path_cells, blockers, end_cell = planned

        panel = np.full((h, w), Color.TEAL, dtype=int)
        sr, sc = start
        er, ec = end_cell
        panel[sr, sc] = Color.GREEN
        panel[er, ec] = Color.RED
        for br, bc in blockers:
            panel[br, bc] = Color.GREY

        panel = _add_extra_obstacles(panel, path_cells, end_cell)
        if panel is None:
            continue

        transformed = _transform_panel(panel)
        try:
            teal_comps = find_components(
                transformed,
                bg=Color.BLACK,
                connectivity=4,
                monochromatic=True,
                colors=[Color.TEAL],
                return_format="dict",
            )
            if len(teal_comps) != 1:
                continue
            if tuple(map(int, teal_comps[0]["bbox"])) != (0, 0, transformed.shape[0] - 1, transformed.shape[1] - 1):
                continue
            solved, turns, path_trace, _states = _trace_panel(transformed)
            if turns != turn_target:
                continue
            if not np.any(solved == Color.GREEN):
                continue
            if len(path_trace) < turn_target + 3:
                continue
        except Exception:
            continue
        return transformed
    return None


def _can_place(grid: np.ndarray, panel: np.ndarray, top: int, left: int) -> bool:
    h, w = panel.shape
    H, W = grid.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(H, top + h + 1)
    c1 = min(W, left + w + 1)
    return np.all(grid[r0:r1, c0:c1] == Color.BLACK)


def _place_panels(panels: list[np.ndarray]) -> np.ndarray | None:
    total_h = sum(panel.shape[0] for panel in panels) + len(panels) + 3
    total_w = sum(panel.shape[1] for panel in panels) + len(panels) + 3
    min_h = max(SCENE_HEIGHT_RANGE[0], max(panel.shape[0] for panel in panels) + 2, min(30, total_h // 2))
    min_w = max(SCENE_WIDTH_RANGE[0], max(panel.shape[1] for panel in panels) + 2, min(30, total_w // 2))
    H = random.randint(min_h, min(30, SCENE_HEIGHT_RANGE[1]))
    W = random.randint(min_w, min(30, SCENE_WIDTH_RANGE[1]))
    grid = np.full((H, W), Color.BLACK, dtype=int)

    for panel in random.sample(panels, len(panels)):
        placed = False
        h, w = panel.shape
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(positions)
        for top, left in positions[:200]:
            if not _can_place(grid, panel, top, left):
                continue
            paste(grid, panel, top, left, transparent=Color.BLACK)
            placed = True
            break
        if not placed:
            return None
    return grid


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        panels = _extract_panels(grid)
        if not (3 <= len(panels) <= 4):
            return False

        turn_counts: list[int] = []
        solved_panels: list[np.ndarray] = []
        for bbox, panel in panels:
            r0, c0, r1, c1 = bbox
            if np.any(np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1] == Color.BLACK):
                return False
            if int(np.count_nonzero(panel == Color.GREEN)) != 1:
                return False
            if int(np.count_nonzero(panel == Color.RED)) != 1:
                return False
            obstacle_count = int(np.count_nonzero(panel == Color.GREY))
            if not (3 <= obstacle_count <= 6):
                return False

            teal_comps = find_components(
                panel,
                bg=Color.BLACK,
                connectivity=4,
                monochromatic=True,
                colors=[Color.TEAL],
                return_format="dict",
            )
            if len(teal_comps) != 1:
                return False
            if tuple(map(int, teal_comps[0]["bbox"])) != (0, 0, panel.shape[0] - 1, panel.shape[1] - 1):
                return False

            solved, turns, _cells, _states = _trace_panel(panel)
            if turns <= 0:
                return False
            turn_counts.append(int(turns))
            solved_panels.append(solved)

        if len(set(turn_counts)) != len(turn_counts):
            return False

        best_idx = max(range(len(turn_counts)), key=lambda idx: turn_counts[idx])
        expected = solved_panels[best_idx]
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if not np.array_equal(actual, solved_panels[best_idx]):
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_panels = random.choice([3, 4])
        turn_choices = random.sample([1, 2, 3, 4], n_panels)

        panels: list[np.ndarray] = []
        ok = True
        for turns in turn_choices:
            panel = _make_panel(turns)
            if panel is None:
                ok = False
                break
            panels.append(panel)
        if not ok:
            continue

        grid = _place_panels(panels)
        if grid is None:
            continue
        if _validate_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid panel-path-complexity input")
