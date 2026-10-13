from __future__ import annotations

import numpy as np


BG = 0
MIRROR = 8
TARGET = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
CARDINALS = (UP, DOWN, LEFT, RIGHT)
PERPENDICULAR = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}

BACKSLASH = "\\"
SLASH = "/"
MIRROR_EXTENSIONS = {
    BACKSLASH: ((-1, -1), (1, 1)),
    SLASH: ((-1, 1), (1, -1)),
}
REFLECTIONS = {
    BACKSLASH: {
        UP: LEFT,
        DOWN: RIGHT,
        LEFT: UP,
        RIGHT: DOWN,
    },
    SLASH: {
        UP: RIGHT,
        DOWN: LEFT,
        LEFT: DOWN,
        RIGHT: UP,
    },
}

SIDE_TO_DIR = {
    "top": DOWN,
    "bottom": UP,
    "left": RIGHT,
    "right": LEFT,
}

GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 28
MIN_EDGE_MARGIN = 2
EDGE_GAP = 4
MIN_SEG_LEN = 2


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return int(r) in (0, h - 1) or int(c) in (0, w - 1)


def _neighbors8(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = int(r + dr)
            cc = int(c + dc)
            if _in_bounds(shape, rr, cc):
                out.append((rr, cc))
    return out


def _reserve_visible_neighborhood(
    forbidden_visible: set[tuple[int, int]],
    cell: tuple[int, int],
    shape: tuple[int, int],
) -> None:
    forbidden_visible.add(cell)
    forbidden_visible.update(_neighbors8(cell, shape))


def _source_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    on_top = int(r) == 0
    on_bottom = int(r) == h - 1
    on_left = int(c) == 0
    on_right = int(c) == w - 1
    assert on_top + on_bottom + on_left + on_right == 1, "sources must occupy a non-corner edge cell"
    if on_top:
        return DOWN
    if on_bottom:
        return UP
    if on_left:
        return RIGHT
    return LEFT


def _mirror_kind_for_turn(
    incoming: tuple[int, int],
    outgoing: tuple[int, int],
) -> str:
    if (incoming, outgoing) in {
        (UP, LEFT),
        (LEFT, UP),
        (DOWN, RIGHT),
        (RIGHT, DOWN),
    }:
        return BACKSLASH
    assert (incoming, outgoing) in {
        (UP, RIGHT),
        (RIGHT, UP),
        (DOWN, LEFT),
        (LEFT, DOWN),
    }
    return SLASH


def _build_mirror_kinds(grid: np.ndarray) -> dict[tuple[int, int], str]:
    h, w = map(int, grid.shape)
    mirrors = set(map(tuple, np.argwhere(grid == MIRROR)))
    kinds: dict[tuple[int, int], str] = {}
    for r, c in mirrors:
        has_backslash = (int(r - 1), int(c - 1)) in mirrors or (int(r + 1), int(c + 1)) in mirrors
        has_slash = (int(r - 1), int(c + 1)) in mirrors or (int(r + 1), int(c - 1)) in mirrors
        assert 0 <= r < h and 0 <= c < w
        assert has_backslash != has_slash, "each azure mirror pixel must belong to exactly one diagonal segment"
        kinds[(int(r), int(c))] = BACKSLASH if has_backslash else SLASH
    return kinds


def _trace_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    mirror_kinds: dict[tuple[int, int], str],
) -> tuple[int, int] | None:
    h, w = map(int, grid.shape)
    r, c = map(int, start)
    dr, dc = map(int, direction)
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        value = int(grid[r, c])
        if value == MIRROR:
            dr, dc = REFLECTIONS[mirror_kinds[(r, c)]][(dr, dc)]
            continue
        if value == TARGET:
            return (r, c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect edge-emitted rays on azure diagonal mirrors and color the grey targets they reach.

    Input structure:
    - The background is black.
    - Azure pixels form one-cell-thick 45-degree diagonal mirror segments of length at least 2.
    - Non-azure, non-grey singleton source pixels sit on the grid edge and point inward perpendicular to that edge.
    - Grey singleton target pixels are scattered in the interior.

    Rule:
    - Each edge source emits one straight ray into the grid.
    - When a ray enters an azure mirror cell, it reflects by 90 degrees according to the mirror's diagonal orientation.
    - The ray keeps traveling until it hits a grey target cell or leaves the grid at an edge.
    - Any hit grey target is recolored to the source's color, while mirrors, sources, and untouched targets stay unchanged.

    Key insight:
    - The azure diagonal segments are reflectors rather than blockers, so the task is to simulate axis-aligned ray paths with discrete 90-degree turns at mirror cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    mirror_kinds = _build_mirror_kinds(g)

    h, w = map(int, g.shape)
    sources: list[tuple[int, int, int, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    edge_cells = (
        [(0, c) for c in range(w)]
        + [(h - 1, c) for c in range(w)]
        + [(r, 0) for r in range(1, h - 1)]
        + [(r, w - 1) for r in range(1, h - 1)]
    )
    for r, c in edge_cells:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        value = int(g[r, c])
        if value in {bg, MIRROR, TARGET}:
            continue
        sources.append((r, c, value, _source_direction(g.shape, r, c)))

    out = g.copy()
    hits: dict[tuple[int, int], int] = {}
    for r, c, color, direction in sorted(sources):
        hit = _trace_ray(g, (r, c), direction, mirror_kinds)
        if hit is None:
            continue
        if hit in hits:
            assert hits[hit] == color, "a target cannot be assigned conflicting source colors"
        hits[hit] = color
    for (r, c), color in hits.items():
        out[r, c] = color
    return out


def _pick_spaced_positions(
    rng: np.random.Generator,
    limit: int,
    count: int,
    margin: int,
    min_gap: int,
) -> list[int] | None:
    candidates = list(range(margin, limit - margin))
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(100):
        rng.shuffle(candidates)
        chosen: list[int] = []
        for value in candidates:
            if all(abs(int(value) - int(existing)) >= min_gap for existing in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _choose_sources(
    rng: np.random.Generator,
    shape: tuple[int, int],
) -> list[dict[str, object]] | None:
    h, w = map(int, shape)
    source_count = 4 + int(rng.random() < 0.5)
    sides = ["top", "bottom", "left", "right"]
    if source_count == 5:
        sides.append(str(rng.choice(["top", "bottom", "left", "right"])))

    counts = {side: sides.count(side) for side in ("top", "bottom", "left", "right")}
    top_cols = _pick_spaced_positions(rng, w, counts["top"], MIN_EDGE_MARGIN, EDGE_GAP)
    bottom_cols = _pick_spaced_positions(rng, w, counts["bottom"], MIN_EDGE_MARGIN, EDGE_GAP)
    left_rows = _pick_spaced_positions(rng, h, counts["left"], MIN_EDGE_MARGIN, EDGE_GAP)
    right_rows = _pick_spaced_positions(rng, h, counts["right"], MIN_EDGE_MARGIN, EDGE_GAP)
    if any(positions is None for positions in (top_cols, bottom_cols, left_rows, right_rows)):
        return None

    colors = list(rng.permutation(SOURCE_COLORS)[:source_count])
    records: list[dict[str, object]] = []
    for side, positions in (
        ("top", top_cols),
        ("bottom", bottom_cols),
        ("left", left_rows),
        ("right", right_rows),
    ):
        assert positions is not None
        for pos in positions:
            if side == "top":
                cell = (0, int(pos))
            elif side == "bottom":
                cell = (h - 1, int(pos))
            elif side == "left":
                cell = (int(pos), 0)
            else:
                cell = (int(pos), w - 1)
            records.append(
                {
                    "side": side,
                    "cell": cell,
                    "dir": SIDE_TO_DIR[side],
                    "color": int(colors[len(records)]),
                }
            )
    rng.shuffle(records)
    return records


def _touches_mirror_neighbor(
    cell: tuple[int, int],
    mirror_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    return any(neighbor in mirror_cells for neighbor in _neighbors8(cell, shape))


def _can_place_mirror_segment(
    shape: tuple[int, int],
    a: tuple[int, int],
    b: tuple[int, int],
    blocked: set[tuple[int, int]],
    forbidden_visible: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
) -> bool:
    if not _in_bounds(shape, *a) or not _in_bounds(shape, *b):
        return False
    if _is_edge(shape, *a) or _is_edge(shape, *b):
        return False
    if a in blocked or b in blocked:
        return False
    if a in forbidden_visible or b in forbidden_visible:
        return False
    if max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) != 1:
        return False
    for cell in (a, b):
        for neighbor in _neighbors8(cell, shape):
            if neighbor in mirror_cells and neighbor not in {a, b}:
                return False
    return True


def _can_place_target(
    shape: tuple[int, int],
    cell: tuple[int, int],
    blocked: set[tuple[int, int]],
    forbidden_visible: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
) -> bool:
    if not _in_bounds(shape, *cell) or _is_edge(shape, *cell):
        return False
    if cell in blocked or cell in forbidden_visible:
        return False
    return not _touches_mirror_neighbor(cell, mirror_cells, shape)


def _plan_path(
    rng: np.random.Generator,
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    success: bool,
    turns: int,
    blocked: set[tuple[int, int]],
    forbidden_visible: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
) -> dict[str, object] | None:
    def dfs(
        curr: tuple[int, int],
        curr_dir: tuple[int, int],
        turns_left: int,
        blocked_now: set[tuple[int, int]],
        mirror_now: set[tuple[int, int]],
        path_cells_now: set[tuple[int, int]],
        segments_now: list[tuple[tuple[int, int], tuple[int, int], str]],
    ) -> dict[str, object] | None:
        line: list[tuple[int, int]] = []
        r, c = map(int, curr)
        dr, dc = map(int, curr_dir)
        while True:
            r += dr
            c += dc
            if not _in_bounds(shape, r, c):
                break
            if (r, c) in blocked_now:
                break
            line.append((r, c))

        if turns_left == 0:
            if success:
                candidate_indices = [
                    idx
                    for idx, cell in enumerate(line)
                    if idx + 1 >= MIN_SEG_LEN and _can_place_target(shape, cell, blocked_now, forbidden_visible, mirror_now)
                ]
                if not candidate_indices:
                    return None
                rng.shuffle(candidate_indices)
                idx = int(candidate_indices[0])
                return {
                    "path_cells": path_cells_now | set(line[: idx + 1]),
                    "mirror_segments": segments_now,
                    "target": line[idx],
                }

            if len(line) >= MIN_SEG_LEN and line and _is_edge(shape, *line[-1]):
                return {
                    "path_cells": path_cells_now | set(line),
                    "mirror_segments": segments_now,
                    "target": None,
                }
            return None

        candidate_indices = [
            idx
            for idx, cell in enumerate(line)
            if idx + 1 >= MIN_SEG_LEN and not _is_edge(shape, *cell) and cell not in forbidden_visible
        ]
        rng.shuffle(candidate_indices)
        for idx in candidate_indices:
            turn = line[idx]
            segment_cells = set(line[: idx + 1])
            new_dirs = list(PERPENDICULAR[curr_dir])
            rng.shuffle(new_dirs)
            for new_dir in new_dirs:
                kind = _mirror_kind_for_turn(curr_dir, new_dir)
                ext_options = list(MIRROR_EXTENSIONS[kind])
                rng.shuffle(ext_options)
                for ext_dr, ext_dc in ext_options:
                    extension = (int(turn[0] + ext_dr), int(turn[1] + ext_dc))
                    if not _can_place_mirror_segment(
                        shape,
                        turn,
                        extension,
                        blocked_now,
                        forbidden_visible,
                        mirror_now,
                    ):
                        continue
                    result = dfs(
                        turn,
                        new_dir,
                        turns_left - 1,
                        blocked_now | segment_cells | {extension},
                        mirror_now | {turn, extension},
                        path_cells_now | segment_cells,
                        segments_now + [(turn, extension, kind)],
                    )
                    if result is not None:
                        return result
        return None

    return dfs(start, direction, turns, blocked, mirror_cells, set(), [])


def _place_extra_target(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    forbidden_visible: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = map(int, shape)
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _can_place_target(shape, (r, c), blocked, forbidden_visible, mirror_cells)
    ]
    if not candidates:
        return None
    return tuple(candidates[int(rng.integers(len(candidates)))])


def _place_distractor_mirror(
    rng: np.random.Generator,
    shape: tuple[int, int],
    kind: str,
    blocked: set[tuple[int, int]],
    forbidden_visible: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, shape)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            a = (r, c)
            for dr, dc in MIRROR_EXTENSIONS[kind]:
                b = (int(r + dr), int(c + dc))
                if _can_place_mirror_segment(shape, a, b, blocked, forbidden_visible, mirror_cells):
                    candidates.append((a, b))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _paint_segments(grid: np.ndarray, segments: list[tuple[tuple[int, int], tuple[int, int], str]]) -> None:
    for a, b, _kind in segments:
        grid[a] = MIRROR
        grid[b] = MIRROR


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    shape = (h, w)

    sources = _choose_sources(rng, shape)
    if sources is None:
        return None

    blocked: set[tuple[int, int]] = set()
    forbidden_visible: set[tuple[int, int]] = set()
    mirror_cells: set[tuple[int, int]] = set()

    for record in sources:
        cell = tuple(record["cell"])
        blocked.add(cell)
        _reserve_visible_neighborhood(forbidden_visible, cell, shape)

    source_count = len(sources)
    success_count = 2 if source_count == 4 else 3
    success_indices = set(map(int, rng.choice(source_count, size=success_count, replace=False)))
    turns_by_index = []
    for idx in range(source_count):
        if idx in success_indices:
            turns = int(rng.choice([0, 1, 2], p=[0.2, 0.5, 0.3]))
        else:
            turns = int(rng.choice([0, 1, 2], p=[0.35, 0.45, 0.2]))
        turns_by_index.append(turns)
    if all(turns_by_index[idx] == 0 for idx in success_indices):
        chosen = int(rng.choice(sorted(success_indices)))
        turns_by_index[chosen] = int(rng.choice([1, 2]))

    for idx, record in enumerate(sources):
        record["success"] = idx in success_indices
        record["turns"] = int(turns_by_index[idx])

    build_order = sorted(
        sources,
        key=lambda rec: (
            -int(bool(rec["success"])),
            -int(rec["turns"]),
            int(rec["cell"][0]),
            int(rec["cell"][1]),
        ),
    )

    planned_records: list[dict[str, object]] = []
    used_targets: set[tuple[int, int]] = set()
    all_segments: list[tuple[tuple[int, int], tuple[int, int], str]] = []

    for record in build_order:
        path = _plan_path(
            rng,
            shape,
            tuple(record["cell"]),
            tuple(record["dir"]),
            bool(record["success"]),
            int(record["turns"]),
            blocked,
            forbidden_visible,
            mirror_cells,
        )
        if path is None:
            return None
        target = path["target"]
        if target is not None:
            target = tuple(target)
            if target in used_targets:
                return None
            used_targets.add(target)
            blocked.add(target)
            _reserve_visible_neighborhood(forbidden_visible, target, shape)
        segments = list(path["mirror_segments"])
        all_segments.extend(segments)
        for a, b, _kind in segments:
            mirror_cells.add(a)
            mirror_cells.add(b)
            blocked.add(a)
            blocked.add(b)
        blocked.update(set(path["path_cells"]))
        planned_records.append(
            {
                "cell": tuple(record["cell"]),
                "color": int(record["color"]),
                "target": target,
                "segments": segments,
            }
        )

    if not any(record["target"] is None for record in planned_records):
        return None
    if not any(record["target"] is not None and record["segments"] for record in planned_records):
        return None

    extra_targets: list[tuple[int, int]] = []
    for _ in range(int(rng.integers(1, 4))):
        target = _place_extra_target(rng, shape, blocked, forbidden_visible, mirror_cells)
        if target is None:
            break
        extra_targets.append(target)
        blocked.add(target)
        _reserve_visible_neighborhood(forbidden_visible, target, shape)
    if not extra_targets:
        return None

    present_kinds = {kind for _a, _b, kind in all_segments}
    needed_kinds = [kind for kind in (BACKSLASH, SLASH) if kind not in present_kinds]
    if rng.random() < 0.4:
        needed_kinds.append(str(rng.choice([BACKSLASH, SLASH])))

    distractor_segments: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for kind in needed_kinds:
        placed = _place_distractor_mirror(rng, shape, kind, blocked, forbidden_visible, mirror_cells)
        if placed is None:
            return None
        a, b = placed
        distractor_segments.append((a, b, kind))
        mirror_cells.add(a)
        mirror_cells.add(b)
        blocked.add(a)
        blocked.add(b)

    grid = np.full(shape, BG, dtype=int)
    for record in sources:
        grid[tuple(record["cell"])] = int(record["color"])
    for target in extra_targets:
        grid[target] = TARGET
    for record in planned_records:
        if record["target"] is not None:
            grid[tuple(record["target"])] = TARGET
        _paint_segments(grid, list(record["segments"]))
    _paint_segments(grid, distractor_segments)

    expected = grid.copy()
    for record in planned_records:
        if record["target"] is not None:
            expected[tuple(record["target"])] = int(record["color"])
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray ARC input")
