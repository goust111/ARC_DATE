import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 18
GRID_MAX = 24
FRAME_SIZE_RANGE = (3, 5)
FRAME_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for row, col in np.argwhere(mask):
        row = int(row)
        col = int(col)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _make_frame(r0: int, c0: int, height: int, width: int) -> dict:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    border: list[tuple[int, int]] = []
    interior: list[tuple[int, int]] = []
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            if row in (r0, r1) or col in (c0, c1):
                border.append((row, col))
            else:
                interior.append((row, col))
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": height,
        "w": width,
        "border": tuple(border),
        "interior": tuple(interior),
    }


def _bar_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == YELLOW)]
    assert len(rows) == 1, "expected exactly one full-width trigger bar"
    return rows[0]


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == GRAY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "frame must enclose a non-empty interior"
        frame = _make_frame(r0, c0, height, width)
        assert set(comp) == set(frame["border"]), "gray component must be a one-pixel hollow rectangle"
        frames.append(frame)
    return frames


def _build_lookup(frames: list[dict]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in frame["border"]:
            lookup[cell] = idx
    return lookup


def _trace_ray(
    out: np.ndarray,
    start: tuple[int, int],
    dc: int,
    frames: list[dict],
    lookup: dict[tuple[int, int], int],
) -> None:
    row, col = start
    while True:
        row -= 1
        col += dc
        if not _in_bounds(out.shape, row, col):
            return

        cell = int(out[row, col])
        if cell == BG:
            out[row, col] = RED
            continue
        if cell == GREEN:
            continue
        if cell == GRAY:
            frame_idx = lookup[(row, col)]
            frame = frames[frame_idx]
            if not frame["opened"]:
                for rr, cc in frame["border"]:
                    out[rr, cc] = BG
                for rr, cc in frame["interior"]:
                    out[rr, cc] = GREEN
                frame["opened"] = True
            out[row, col] = RED
            continue
        if cell in (RED, YELLOW):
            return
        raise AssertionError(f"unexpected color on ray path: {cell}")


def _run_layout(
    grid: np.ndarray,
    bar_row: int,
    red_positions: list[tuple[int, int]],
    frames: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    frame_states = [dict(frame, opened=False) for frame in frames]
    lookup = _build_lookup(frame_states)

    out[out == RED] = BG
    moved_reds: list[tuple[int, int]] = []
    for row, col in red_positions:
        new_row = row + 1
        assert new_row < out.shape[0], "red pixel cannot move outside the grid"
        moved_reds.append((new_row, col))
    for row, col in moved_reds:
        out[row, col] = RED

    triggers = sorted((row, col) for row, col in moved_reds if row == bar_row - 1)
    for start in triggers:
        _trace_ray(out, start, -1, frame_states, lookup)
        _trace_ray(out, start, 1, frame_states, lookup)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move every red pixel down by one, then let the trigger-adjacent ones shoot upward diagonal rays that open blocking hollow frames.

    Input structure:
    - The grid has black background, one full-width yellow trigger bar near the bottom, several gray one-pixel hollow rectangles above it, and 2-3 red pixels above the bar.
    - Some red pixels start exactly two rows above the trigger bar, so after moving down they land directly above the bar and become ray sources.
    - The gray objects are disjoint hollow rectangles whose interiors are black in the input.

    Rule:
    - Erase every red pixel from its original location and redraw it one row lower in the same column.
    - Each moved red pixel that ends directly above the yellow bar emits one ray up-left and one ray up-right.
    - A ray paints black cells red, passes through green cells unchanged, stops on red, yellow, or the grid edge, and when it first hits a gray frame border it opens that whole frame by erasing the gray border to black and filling the enclosed interior with green before continuing through the opened gap.

    Key insight:
    - Lower frames are both targets and blockers: a ray must open the lower hollow frame first, which removes the obstacle and lets the same diagonal continue upward to open higher frames in cascade.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bar = _bar_row(g)
    reds = [tuple(map(int, cell)) for cell in np.argwhere(g == RED)]
    frames = _detect_frames(g)
    return _run_layout(g, bar, reds, frames)


def _ray_step(start: tuple[int, int], dc: int, step: int) -> tuple[int, int]:
    return start[0] - step, start[1] + dc * step


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], dc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    step = 1
    while True:
        row, col = _ray_step(start, dc, step)
        if not _in_bounds(shape, row, col):
            return cells
        cells.append((row, col))
        step += 1


def _ray_cell_set(shape: tuple[int, int], start: tuple[int, int], dc: int) -> set[tuple[int, int]]:
    return set(_ray_cells(shape, start, dc))


def _ray_index(start: tuple[int, int], dc: int, cell: tuple[int, int]) -> int | None:
    dr = start[0] - cell[0]
    dc_delta = cell[1] - start[1]
    if dr < 1:
        return None
    if dc_delta != dc * dr:
        return None
    return dr


def _bbox_touches(a: dict, b: dict, pad: int = 1) -> bool:
    return not (
        a["r1"] + pad < b["r0"]
        or b["r1"] + pad < a["r0"]
        or a["c1"] + pad < b["c0"]
        or b["c1"] + pad < a["c0"]
    )


def _frame_contains_cell(frame: dict, cell: tuple[int, int]) -> bool:
    row, col = cell
    return frame["r0"] <= row <= frame["r1"] and frame["c0"] <= col <= frame["c1"]


def _candidate_attack_frame(
    shape: tuple[int, int],
    start: tuple[int, int],
    dc: int,
    hit_step: int,
    height: int,
    width: int,
    offset: int,
) -> tuple[dict, int] | None:
    hit_row, hit_col = _ray_step(start, dc, hit_step)
    if not _in_bounds(shape, hit_row, hit_col):
        return None

    r0 = hit_row - height + 1
    c0 = hit_col - offset
    if r0 < 0 or c0 < 0 or r0 + height > shape[0] or c0 + width > shape[1]:
        return None

    if dc == 1:
        span = 1 + min(height - 1, width - 1 - offset)
    else:
        span = 1 + min(height - 1, offset)
    if span < 3:
        return None

    frame = _make_frame(r0, c0, height, width)
    return frame, hit_step + span - 1


def _choose_trigger_cols(width: int, triggered_count: int) -> list[int] | None:
    cols = list(range(4, width - 4))
    if triggered_count == 1:
        return [random.choice(cols)]

    pairs = [
        (a, b)
        for a in cols
        for b in cols
        if a < b and b - a >= 5 and (b - a) % 2 == 1
    ]
    if not pairs:
        return None
    return list(random.choice(pairs))


def _build_ray_plan(triggered_count: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], int]:
    ray_keys = [(idx, dc) for idx in range(triggered_count) for dc in (-1, 1)]
    primary = random.choice(ray_keys)
    plan: dict[tuple[int, int], int] = {primary: random.choice([2, 2, 3])}

    others = [key for key in ray_keys if key != primary]
    random.shuffle(others)
    for key in others:
        if sum(plan.values()) >= 4:
            break
        if random.random() < 0.35:
            plan[key] = 1

    for src in range(triggered_count):
        if not any(key[0] == src for key in plan):
            candidates = [key for key in ray_keys if key[0] == src]
            key = random.choice(candidates)
            if key not in plan and sum(plan.values()) < 4:
                plan[key] = 1

    if len(plan) == len(ray_keys):
        removable = [key for key, count in plan.items() if key != primary and count == 1]
        if removable:
            del plan[random.choice(removable)]

    attacked_count = sum(plan.values())
    target_frames = random.randint(max(FRAME_COUNT_RANGE[0], attacked_count + 1), FRAME_COUNT_RANGE[1])
    order = [primary] + [key for key in ray_keys if key in plan and key != primary]
    return order, plan, target_frames


def _place_attack_path(
    shape: tuple[int, int],
    bar_row: int,
    start: tuple[int, int],
    dc: int,
    count: int,
    ray_key: tuple[int, int],
    all_rays: dict[tuple[int, int], set[tuple[int, int]]],
    frames: list[dict],
    blocker_cells: set[tuple[int, int]],
) -> list[dict] | None:
    placed: list[dict] = []
    previous_exit = 0
    for _ in range(count):
        candidates: list[dict] = []
        steps = list(range(previous_exit + 2, start[0] + 1))
        random.shuffle(steps)
        sizes = [
            (height, width)
            for height in range(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1] + 1)
            for width in range(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1] + 1)
        ]
        random.shuffle(sizes)
        for hit_step in steps:
            for height, width in sizes:
                offsets = list(range(0, width - 2)) if dc == 1 else list(range(2, width))
                random.shuffle(offsets)
                for offset in offsets:
                    candidate = _candidate_attack_frame(shape, start, dc, hit_step, height, width, offset)
                    if candidate is None:
                        continue
                    frame, exit_step = candidate
                    if frame["r1"] >= bar_row - 1:
                        continue
                    existing_frames = frames + [item["frame"] for item in placed]
                    if any(_bbox_touches(frame, other) for other in existing_frames):
                        continue
                    if any(
                        any(cell in other_line for cell in frame["border"])
                        for other_key, other_line in all_rays.items()
                        if other_key != ray_key
                    ):
                        continue
                    blocked = False
                    for step in range(previous_exit + 1, hit_step):
                        if _ray_step(start, dc, step) in blocker_cells:
                            blocked = True
                            break
                    if blocked:
                        continue
                    candidates.append(
                        {
                            "frame": frame,
                            "hit_step": hit_step,
                            "exit_step": exit_step,
                        }
                    )
        if not candidates:
            return None
        chosen = random.choice(candidates)
        placed.append(chosen)
        previous_exit = chosen["exit_step"]
    return placed


def _frame_intersects_any_ray(frame: dict, ray_lines: dict[tuple[int, int], set[tuple[int, int]]]) -> bool:
    return any(any(cell in line for cell in frame["border"]) for line in ray_lines.values())


def _pick_nontrigger_reds(
    shape: tuple[int, int],
    bar_row: int,
    count: int,
    frames: list[dict],
    ray_lines: dict[tuple[int, int], set[tuple[int, int]]],
    occupied_reds: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    choices = [
        (old_row, col)
        for old_row in range(1, bar_row - 3)
        for col in range(1, shape[1] - 1)
    ]
    random.shuffle(choices)
    picked: list[tuple[int, int]] = []
    for old_row, col in choices:
        old_cell = (old_row, col)
        moved_cell = (old_row + 1, col)
        if old_cell in occupied_reds or moved_cell in occupied_reds:
            continue
        if any(old_cell in line or moved_cell in line for line in ray_lines.values()):
            continue
        if any(_frame_contains_cell(frame, old_cell) or _frame_contains_cell(frame, moved_cell) for frame in frames):
            continue
        if any(max(abs(old_row - rr), abs(col - cc)) <= 1 for rr, cc in occupied_reds | set(picked)):
            continue
        picked.append(old_cell)
        if len(picked) == count:
            return picked
    return None


def _place_distractors(
    shape: tuple[int, int],
    bar_row: int,
    target_frames: int,
    frames: list[dict],
    ray_lines: dict[tuple[int, int], set[tuple[int, int]]],
    red_positions: set[tuple[int, int]],
) -> bool:
    attempts = 0
    while len(frames) < target_frames and attempts < 600:
        attempts += 1
        if random.random() < 0.65:
            height = random.randint(*FRAME_SIZE_RANGE)
            width_choices = [w for w in range(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1] + 1) if w != height]
            width = random.choice(width_choices)
        else:
            height = random.randint(*FRAME_SIZE_RANGE)
            width = random.randint(*FRAME_SIZE_RANGE)
        r0 = random.randint(0, bar_row - height - 1)
        c0 = random.randint(0, shape[1] - width)
        frame = _make_frame(r0, c0, height, width)
        if any(_bbox_touches(frame, other) for other in frames):
            continue
        if _frame_intersects_any_ray(frame, ray_lines):
            continue
        if any(_frame_contains_cell(frame, cell) for cell in red_positions):
            continue
        frames.append(frame)
    return len(frames) == target_frames


def _render_scene(
    shape: tuple[int, int],
    bar_row: int,
    red_positions: list[tuple[int, int]],
    frames: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[bar_row, :] = YELLOW
    for frame in frames:
        for row, col in frame["border"]:
            grid[row, col] = GRAY
    for row, col in red_positions:
        grid[row, col] = RED
    return grid


def _validate_scene(scene: dict, grid: np.ndarray, expected: np.ndarray) -> None:
    shape = tuple(map(int, grid.shape))
    bar_row = scene["bar_row"]
    red_positions = scene["red_positions"]
    frames = scene["frames"]
    plan = scene["plan"]
    ray_lines = scene["ray_lines"]

    assert FRAME_COUNT_RANGE[0] <= len(frames) <= FRAME_COUNT_RANGE[1]
    assert 2 <= len(red_positions) <= 3
    assert scene["triggered_count"] >= 1
    assert len(red_positions) > scene["triggered_count"]
    assert any(len(entries) >= 2 for entries in plan.values())
    assert len(plan) < len(ray_lines), "need at least one emitted ray with no frame on it"

    planned_frames = {entry["index"] for entries in plan.values() for entry in entries}
    for src in range(scene["triggered_count"]):
        assert any(key[0] == src for key in plan), "every triggering red needs a reachable frame on at least one side"

    for a_idx, a_line in ray_lines.items():
        for b_idx, b_line in ray_lines.items():
            if a_idx >= b_idx:
                continue
            assert not (a_line & b_line), "emitted rays must not intersect and create order-dependent outputs"

    for key, entries in plan.items():
        start = scene["trigger_sources"][key[0]]
        actual = []
        for idx, frame in enumerate(frames):
            steps = [_ray_index(start, key[1], cell) for cell in frame["border"]]
            hits = sorted(step for step in steps if step is not None)
            if hits:
                actual.append((hits[0], idx))
        actual.sort()
        assert [idx for _, idx in actual] == [entry["index"] for entry in entries]
        if len(entries) >= 2:
            for lower, upper in zip(entries, entries[1:]):
                assert lower["exit_step"] < upper["hit_step"], "cascade frames must be vertically ordered on the same ray"

    for idx, frame in enumerate(frames):
        border_vals = [int(expected[row, col]) for row, col in frame["border"]]
        interior_vals = [int(expected[row, col]) for row, col in frame["interior"]]
        if idx in planned_frames:
            assert all(val != GRAY for val in border_vals)
            assert all(val == GREEN for val in interior_vals)
        else:
            assert all(val == GRAY for val in border_vals)
            assert all(val == BG for val in interior_vals)

    moved_reds = {(row + 1, col) for row, col in red_positions}
    for row, col in moved_reds:
        assert expected[row, col] == RED
        assert _in_bounds(shape, row, col)
    assert np.array_equal(main(grid), expected)


def _generate_scene() -> dict | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    shape = (height, width)
    bar_row = height - random.choice([3, 4])
    red_count = random.randint(2, 3)
    triggered_count = 1 if red_count == 2 else random.choice([1, 2])

    trigger_cols = _choose_trigger_cols(width, triggered_count)
    if trigger_cols is None:
        return None
    trigger_sources = [(bar_row - 1, col) for col in trigger_cols]
    trigger_reds = [(bar_row - 2, col) for col in trigger_cols]

    ray_order, ray_counts, target_frames = _build_ray_plan(triggered_count)
    ray_lines = {
        (idx, dc): _ray_cell_set(shape, trigger_sources[idx], dc)
        for idx in range(triggered_count)
        for dc in (-1, 1)
    }

    frames: list[dict] = []
    plan: dict[tuple[int, int], list[dict]] = {}
    blocker_cells = set(trigger_sources)
    for key in ray_order:
        start = trigger_sources[key[0]]
        placed = _place_attack_path(
            shape,
            bar_row,
            start,
            key[1],
            ray_counts[key],
            key,
            ray_lines,
            frames,
            blocker_cells,
        )
        if placed is None:
            return None
        plan[key] = []
        for item in placed:
            frame = item["frame"]
            frame_index = len(frames)
            frames.append(frame)
            blocker_cells.update(frame["border"])
            plan[key].append(
                {
                    "index": frame_index,
                    "hit_step": item["hit_step"],
                    "exit_step": item["exit_step"],
                }
            )

    nontrigger_count = red_count - triggered_count
    nontrigger_reds = _pick_nontrigger_reds(
        shape,
        bar_row,
        nontrigger_count,
        frames,
        ray_lines,
        set(trigger_reds) | set(trigger_sources),
    )
    if nontrigger_reds is None:
        return None

    red_positions = trigger_reds + nontrigger_reds
    if not _place_distractors(shape, bar_row, target_frames, frames, ray_lines, set(red_positions)):
        return None

    scene = {
        "shape": shape,
        "bar_row": bar_row,
        "red_positions": red_positions,
        "trigger_sources": trigger_sources,
        "triggered_count": triggered_count,
        "frames": frames,
        "plan": plan,
        "ray_lines": ray_lines,
    }
    return scene


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _generate_scene()
        if scene is None:
            continue
        grid = _render_scene(scene["shape"], scene["bar_row"], scene["red_positions"], scene["frames"])
        expected = _run_layout(grid, scene["bar_row"], scene["red_positions"], scene["frames"])
        try:
            _validate_scene(scene, grid, expected)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid cascading frame scene")
