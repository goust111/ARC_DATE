from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
CELL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
POSES = tuple(range(8))
QUARTER_TURNS = {1, 3}
REFLECTIONS = {4, 5, 6, 7}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    winners = vals[counts == counts.max()]
    assert winners.size == 1, "background color must be uniquely most common"
    return int(winners[0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _parse_frame_component(
    grid: np.ndarray, color: int, coords: np.ndarray
) -> tuple[list[int], list[int]] | None:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)

    row_lines = [r for r in range(sub.shape[0]) if bool(sub[r, :].all())]
    col_lines = [c for c in range(sub.shape[1]) if bool(sub[:, c].all())]
    if len(row_lines) < 3 or len(col_lines) < 3:
        return None
    if row_lines[0] != 0 or row_lines[-1] != sub.shape[0] - 1:
        return None
    if col_lines[0] != 0 or col_lines[-1] != sub.shape[1] - 1:
        return None
    if any(a + 1 >= b for a, b in zip(row_lines, row_lines[1:])):
        return None
    if any(a + 1 >= b for a, b in zip(col_lines, col_lines[1:])):
        return None

    expected = np.zeros_like(sub, dtype=bool)
    expected[row_lines, :] = True
    expected[:, col_lines] = True
    if not np.array_equal(sub, expected):
        return None

    rows = [int(r0) + r for r in row_lines]
    cols = [int(c0) + c for c in col_lines]
    return rows, cols


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[list[int], list[int]]]]:
    candidates: list[tuple[int, list[tuple[list[int], list[int]]]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        frames = []
        for coords in _components(grid == color):
            frame = _parse_frame_component(grid, color, coords)
            if frame is not None:
                frames.append(frame)
        if len(frames) < 2:
            continue
        dims = {(len(rows) - 1, len(cols) - 1) for rows, cols in frames}
        if len(dims) == 1:
            frames.sort(key=lambda rc: (rc[0][0], rc[1][0]))
            candidates.append((color, frames))

    assert len(candidates) == 1, "expected exactly one lattice-frame color"
    return candidates[0]


def _pose_coord(pose: int, r: int, c: int, n: int) -> tuple[int, int]:
    r = int(r)
    c = int(c)
    last = int(n) - 1
    if pose == 0:
        return r, c
    if pose == 1:
        return c, last - r
    if pose == 2:
        return last - r, last - c
    if pose == 3:
        return last - c, r
    if pose == 4:
        return last - r, c
    if pose == 5:
        return r, last - c
    if pose == 6:
        return c, r
    if pose == 7:
        return last - c, last - r
    raise ValueError("bad pose")


def _view_from_table(table: np.ndarray, pose: int) -> np.ndarray:
    n = table.shape[0]
    out = np.empty_like(table)
    for r in range(n):
        for c in range(n):
            tr, tc = _pose_coord(pose, r, c, n)
            out[r, c] = table[tr, tc]
    return out


def _slot_observations(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frames: list[tuple[list[int], list[int]]],
) -> list[dict[tuple[int, int], int]]:
    observations: list[dict[tuple[int, int], int]] = []
    slot_rows = len(frames[0][0]) - 1
    slot_cols = len(frames[0][1]) - 1
    for rows, cols in frames:
        obs: dict[tuple[int, int], int] = {}
        for sr in range(slot_rows):
            for sc in range(slot_cols):
                region = grid[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]]
                if np.all(region == int(bg)):
                    continue
                assert not np.any(region == int(frame_color)), "frame color cannot fill a slot"
                colors = np.unique(region)
                assert colors.size == 1 and int(colors[0]) != int(bg), "filled slots must be solid"
                obs[(sr, sc)] = int(colors[0])
        observations.append(obs)
    return observations


def _solutions(
    observations: list[dict[tuple[int, int], int]], n: int, stop_after: int = 2
) -> list[tuple[tuple[int, ...], dict[tuple[int, int], int]]]:
    found: list[tuple[tuple[int, ...], dict[tuple[int, int], int]]] = []
    if not observations:
        return found

    pose_ranges = [[0]] + [list(POSES) for _ in range(len(observations) - 1)]
    for pose_tuple in itertools.product(*pose_ranges):
        table: dict[tuple[int, int], int] = {}
        ok = True
        for frame_index, obs in enumerate(observations):
            pose = int(pose_tuple[frame_index])
            for (sr, sc), color in obs.items():
                key = _pose_coord(pose, sr, sc, n)
                old = table.get(key)
                if old is not None and old != int(color):
                    ok = False
                    break
                table[key] = int(color)
            if not ok:
                break
        if ok and len(table) == n * n:
            found.append((tuple(int(p) for p in pose_tuple), table))
            if len(found) >= stop_after:
                break
    return found


def _unique_solution(
    observations: list[dict[tuple[int, int], int]], n: int
) -> tuple[tuple[int, ...], dict[tuple[int, int], int]] | None:
    sols = _solutions(observations, n, stop_after=2)
    if len(sols) != 1:
        return None
    return sols[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every lattice frame after registering each frame's rotation or reflection pose.

    Input structure:
    - The background is the uniquely most common color.
    - One foreground color forms several separated one-pixel rectangular lattice frames.
    - Every frame has the same square number of logical slots, but its slot heights and widths may differ.
    - A slot is either solidly filled by one non-background, non-frame color, or left as background.

    Rule:
    - Treat the first lattice in reading order as the reference view of the hidden logical color table.
    - For every other lattice, enumerate the eight rotation/reflection poses and keep only globally consistent pose assignments.
    - The valid assignment must uniquely determine every logical table cell from the union of filled slots.
    - Fill every empty slot in every lattice with the table color mapped through that lattice's pose.

    Key insight:
    - Slot colors cannot be copied by absolute row and column: each frame is a different D4 pose of the same logical table, so the poses must be registered before completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, frames = _find_frames(g, bg)
    n_rows = len(frames[0][0]) - 1
    n_cols = len(frames[0][1]) - 1
    assert n_rows == n_cols and n_rows >= 2, "pose registration uses a square logical table"
    for rows, cols in frames:
        assert len(rows) - 1 == n_rows and len(cols) - 1 == n_cols

    observations = _slot_observations(g, bg, frame_color, frames)
    assert all(obs for obs in observations), "each lattice must provide at least one filled slot"
    solved = _unique_solution(observations, n_rows)
    assert solved is not None, "expected one globally consistent pose assignment"
    poses, table = solved

    out = g.copy()
    for frame_index, (rows, cols) in enumerate(frames):
        pose = poses[frame_index]
        for sr in range(n_rows):
            for sc in range(n_cols):
                color = table[_pose_coord(pose, sr, sc, n_rows)]
                out[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]] = int(color)
    return out


# ---------------- generator ----------------


def _line_indices(sizes: tuple[int, ...]) -> list[int]:
    out = [0]
    cur = 0
    for size in sizes:
        cur += int(size) + 1
        out.append(cur)
    return out


def _frame_shape(row_sizes: tuple[int, ...], col_sizes: tuple[int, ...]) -> tuple[int, int]:
    return sum(row_sizes) + len(row_sizes) + 1, sum(col_sizes) + len(col_sizes) + 1


def _asymmetric_table(table: np.ndarray) -> bool:
    return all(not np.array_equal(_view_from_table(table, pose), table) for pose in POSES if pose != 0)


def _make_table(n: int) -> np.ndarray | None:
    for _ in range(200):
        if n == 2:
            vals = random.sample(CELL_COLORS, 4)
        else:
            vals = CELL_COLORS[:] + [random.choice(CELL_COLORS)]
            random.shuffle(vals)
        table = np.array(vals, dtype=int).reshape(n, n)
        if _asymmetric_table(table):
            return table
    return None


def _choose_poses(n_frames: int) -> list[int]:
    base = [0, random.choice(tuple(QUARTER_TURNS)), random.choice(tuple(REFLECTIONS))]
    while len(base) < n_frames:
        base.append(random.randrange(8))
    random.shuffle(base)
    return base[:n_frames]


def _choose_specs(n: int, n_frames: int) -> list[dict]:
    specs: list[dict] = []
    lo, hi = (2, 5) if n == 2 else (1, 4)
    for _ in range(n_frames):
        rows = tuple(random.randint(lo, hi) for _ in range(n))
        cols = tuple(random.randint(lo, hi) for _ in range(n))
        specs.append({"row_sizes": rows, "col_sizes": cols})
    if len({(s["row_sizes"], s["col_sizes"]) for s in specs}) == 1:
        return []
    return specs


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> bool:
    rows, cols = occupied.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(rows, r0 + h + padding)
    cc1 = min(cols, c0 + w + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_records(records: list[dict], rows: int, cols: int) -> list[dict] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    placed: list[dict] = []
    order = sorted(records, key=lambda r: _frame_shape(r["row_sizes"], r["col_sizes"])[0] * _frame_shape(r["row_sizes"], r["col_sizes"])[1], reverse=True)
    for rec in order:
        h, w = _frame_shape(rec["row_sizes"], rec["col_sizes"])
        candidates = [
            (r, c)
            for r in range(1, rows - h)
            for c in range(1, cols - w)
            if _can_place(occupied, r, c, h, w, padding=1)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        new = dict(rec)
        new["r0"] = int(r0)
        new["c0"] = int(c0)
        placed.append(new)
    placed.sort(key=lambda r: (r["r0"], r["c0"]))
    return placed


def _observations_from_visible(
    records: list[dict], table: np.ndarray, visible: list[set[tuple[int, int]]]
) -> list[dict[tuple[int, int], int]]:
    n = table.shape[0]
    observations: list[dict[tuple[int, int], int]] = []
    for rec, keep in zip(records, visible):
        obs: dict[tuple[int, int], int] = {}
        pose = int(rec["pose"])
        for sr, sc in sorted(keep):
            tr, tc = _pose_coord(pose, sr, sc, n)
            obs[(sr, sc)] = int(table[tr, tc])
        observations.append(obs)
    return observations


def _absolute_shortcut_fails(
    observations: list[dict[tuple[int, int], int]],
    visible: list[set[tuple[int, int]]],
    poses: tuple[int, ...],
    table: dict[tuple[int, int], int],
    n: int,
) -> bool:
    by_visual: dict[tuple[int, int], int] = {}
    for obs in observations:
        for cell, color in obs.items():
            if cell in by_visual and by_visual[cell] != int(color):
                return True
            by_visual[cell] = int(color)

    for frame_index, keep in enumerate(visible):
        for sr in range(n):
            for sc in range(n):
                if (sr, sc) in keep:
                    continue
                expected = table[_pose_coord(poses[frame_index], sr, sc, n)]
                copied = by_visual.get((sr, sc))
                if copied is not None and int(copied) != int(expected):
                    return True
    return False


def _semantic_ok(
    records: list[dict], table_array: np.ndarray, visible: list[set[tuple[int, int]]]
) -> tuple[tuple[int, ...], dict[tuple[int, int], int]] | None:
    n = table_array.shape[0]
    total = n * n
    observations = _observations_from_visible(records, table_array, visible)
    solved = _unique_solution(observations, n)
    if solved is None:
        return None
    poses, table = solved

    if not all(0 < total - len(keep) < total for keep in visible):
        return None
    if len(observations[0]) == total:
        return None
    if not any(p in QUARTER_TURNS for p in poses[1:]):
        return None
    if not any(p in REFLECTIONS for p in poses[1:]):
        return None

    moved_clue = False
    for frame_index, obs in enumerate(observations):
        pose = poses[frame_index]
        for sr, sc in obs:
            mapped = _pose_coord(pose, sr, sc, n)
            if mapped != (sr, sc) and table[mapped] != table[(sr, sc)]:
                moved_clue = True
                break
        if moved_clue:
            break
    if not moved_clue:
        return None
    if not _absolute_shortcut_fails(observations, visible, poses, table, n):
        return None
    return solved


def _choose_visible(records: list[dict], table: np.ndarray) -> list[set[tuple[int, int]]] | None:
    n = table.shape[0]
    cells = [(r, c) for r in range(n) for c in range(n)]
    visible = [set(cells) for _ in records]
    min_seen = 2 if n == 2 else 4

    candidates = [(fi, cell) for fi in range(len(records)) for cell in cells]
    random.shuffle(candidates)
    for fi, cell in candidates:
        if len(visible[fi]) <= min_seen or random.random() < 0.18:
            continue
        visible[fi].remove(cell)
        if _unique_solution(_observations_from_visible(records, table, visible), n) is None:
            visible[fi].add(cell)

    for fi in range(len(records)):
        if len(visible[fi]) < n * n:
            continue
        forced = cells[:]
        random.shuffle(forced)
        removed = False
        for cell in forced:
            if len(visible[fi]) <= min_seen:
                break
            visible[fi].remove(cell)
            if _unique_solution(_observations_from_visible(records, table, visible), n) is not None:
                removed = True
                break
            visible[fi].add(cell)
        if not removed:
            return None

    if _semantic_ok(records, table, visible) is None:
        return None
    return visible


def _render_patch(
    table: np.ndarray,
    pose: int,
    row_sizes: tuple[int, ...],
    col_sizes: tuple[int, ...],
    visible: set[tuple[int, int]],
    full: bool,
) -> np.ndarray:
    n = table.shape[0]
    row_lines = _line_indices(row_sizes)
    col_lines = _line_indices(col_sizes)
    patch = np.full((row_lines[-1] + 1, col_lines[-1] + 1), BG, dtype=int)
    patch[row_lines, :] = FRAME
    patch[:, col_lines] = FRAME

    for sr in range(n):
        for sc in range(n):
            if full or (sr, sc) in visible:
                tr, tc = _pose_coord(pose, sr, sc, n)
                patch[row_lines[sr] + 1 : row_lines[sr + 1], col_lines[sc] + 1 : col_lines[sc + 1]] = int(table[tr, tc])
    return patch


def _render_grids(
    rows: int, cols: int, records: list[dict], table: np.ndarray, visible: list[set[tuple[int, int]]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)
    for rec, keep in zip(records, visible):
        args = (table, int(rec["pose"]), rec["row_sizes"], rec["col_sizes"])
        in_patch = _render_patch(*args, visible=keep, full=False)
        out_patch = _render_patch(*args, visible=keep, full=True)
        r0, c0 = int(rec["r0"]), int(rec["c0"])
        h, w = in_patch.shape
        inp[r0 : r0 + h, c0 : c0 + w] = in_patch
        out[r0 : r0 + h, c0 : c0 + w] = out_patch
    return inp, out


def _attempt() -> np.ndarray | None:
    n = 3 if random.random() < 0.62 else 2
    n_frames = random.randint(3, 4) if n == 3 else random.randint(4, 5)
    table = _make_table(n)
    if table is None:
        return None

    specs = _choose_specs(n, n_frames)
    if not specs:
        return None
    poses = _choose_poses(n_frames)
    for spec, pose in zip(specs, poses):
        spec["pose"] = int(pose)

    rows = random.randint(25, 30)
    cols = random.randint(25, 30)
    placed = _place_records(specs, rows, cols)
    if placed is None:
        return None

    visible = _choose_visible(placed, table)
    if visible is None:
        return None

    inp, expected = _render_grids(rows, cols, placed, table, visible)
    if max(inp.shape) > 30:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fable 4/008/6 input")
