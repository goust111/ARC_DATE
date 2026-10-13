import heapq
import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8

INNER_COLORS = (BLUE, RED, YELLOW, ORANGE)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
HOLE_LOCALS = (
    (0, 1),
    (0, 2),
    (1, 0),
    (2, 0),
    (1, 3),
    (2, 3),
    (3, 1),
    (3, 2),
)

FRAME_SIZE = 4
FRAME_PAD = 2
MIN_GRID = 18
MAX_GRID = 28
GEN_ATTEMPTS = 400


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _turn_left(d: int) -> int:
    return (d - 1) % 4


def _turn_right(d: int) -> int:
    return (d + 1) % 4


def _outward_dir(hole_local: tuple[int, int]) -> int:
    rr, cc = hole_local
    if rr == 0:
        return 0
    if cc == 3:
        return 1
    if rr == 3:
        return 2
    return 3


def _build_frame(top: int, left: int, color: int, hole_local: tuple[int, int]) -> dict:
    wall = set()
    hole = None
    bbox = {(top + rr, left + cc) for rr in range(FRAME_SIZE) for cc in range(FRAME_SIZE)}
    for rr in range(FRAME_SIZE):
        for cc in range(FRAME_SIZE):
            if rr in (0, FRAME_SIZE - 1) or cc in (0, FRAME_SIZE - 1):
                if (rr, cc) == hole_local:
                    hole = (top + rr, left + cc)
                else:
                    wall.add((top + rr, left + cc))
    assert hole is not None
    return {
        "top": top,
        "left": left,
        "inner_color": int(color),
        "hole": hole,
        "dir": _outward_dir(hole_local),
        "wall": frozenset(wall),
        "bbox": frozenset(bbox),
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames = []
    for top in range(h - FRAME_SIZE + 1):
        for left in range(w - FRAME_SIZE + 1):
            patch = g[top : top + FRAME_SIZE, left : left + FRAME_SIZE]
            inner = patch[1:3, 1:3]
            inner_color = int(inner[0, 0])
            if inner_color in (BLACK, GREEN) or not np.all(inner == inner_color):
                continue

            hole_local = None
            wall = set()
            ok = True
            for rr in range(FRAME_SIZE):
                for cc in range(FRAME_SIZE):
                    if rr not in (0, FRAME_SIZE - 1) and cc not in (0, FRAME_SIZE - 1):
                        continue
                    cell = int(patch[rr, cc])
                    is_corner = rr in (0, FRAME_SIZE - 1) and cc in (0, FRAME_SIZE - 1)
                    if is_corner:
                        if cell != GREEN:
                            ok = False
                            break
                        wall.add((top + rr, left + cc))
                        continue
                    if cell == GREEN:
                        wall.add((top + rr, left + cc))
                    elif cell == BLACK and hole_local is None and (rr, cc) in HOLE_LOCALS:
                        hole_local = (rr, cc)
                    else:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or hole_local is None or len(wall) != 11:
                continue
            frames.append(_build_frame(top, left, inner_color, hole_local))

    frames.sort(key=lambda frame: (frame["top"], frame["left"]))
    occupied = set()
    unique = []
    for frame in frames:
        if occupied & frame["bbox"]:
            continue
        occupied |= set(frame["bbox"])
        unique.append(frame)
    return unique


def _trace_beams(grid: np.ndarray, frames: list[dict], strict: bool = False) -> tuple[list[list[tuple[int, int, int]]], dict[int, int], dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_owner = {}
    bbox_owner = {}
    for idx, frame in enumerate(frames):
        for cell in frame["wall"]:
            wall_owner[cell] = idx
        for cell in frame["bbox"]:
            bbox_owner[cell] = idx

    traced = []
    incoming = {}
    painted = set()
    turn_counts = {"left": 0, "right": 0}

    for idx, frame in enumerate(frames):
        color = int(frame["inner_color"])
        r, c = frame["hole"]
        d = int(frame["dir"])
        states = []
        seen_states = set()
        while True:
            state = (r, c, d)
            if state in seen_states:
                if strict:
                    raise AssertionError("beam loop detected")
                break
            seen_states.add(state)
            if strict and (r, c) in painted:
                raise AssertionError("beam paths must be disjoint")
            painted.add((r, c))
            states.append(state)

            dr, dc = DIRS[d]
            nr, nc = r + dr, c + dc
            if not _inside(h, w, nr, nc):
                break

            owner = wall_owner.get((nr, nc))
            if owner is not None:
                if owner != idx:
                    if strict and owner in incoming:
                        raise AssertionError("a frame should not receive two beams")
                    incoming[owner] = color
                break

            if strict:
                if bbox_owner.get((nr, nc)) is not None:
                    raise AssertionError("beam entered a frame body")
                cell = int(g[nr, nc])
                if cell not in (BLACK, AZURE, MAGENTA):
                    raise AssertionError("beam must traverse black cells or turners")

            r, c = nr, nc
            cell = int(g[r, c])
            if cell == AZURE:
                d = _turn_left(d)
                turn_counts["left"] += 1
            elif cell == MAGENTA:
                d = _turn_right(d)
                turn_counts["right"] += 1
        traced.append(states)

    return traced, incoming, turn_counts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    out = g.copy()

    paths, incoming, _turns = _trace_beams(g, frames, strict=False)
    for idx, states in enumerate(paths):
        color = int(frames[idx]["inner_color"])
        for r, c, _d in states:
            out[r, c] = color

    for target_idx, color in incoming.items():
        top = int(frames[target_idx]["top"])
        left = int(frames[target_idx]["left"])
        out[top + 1 : top + 3, left + 1 : left + 3] = int(color)

    return out


def _boxes_too_close(a: dict, b: dict, pad: int) -> bool:
    return not (
        a["top"] + FRAME_SIZE - 1 + pad < b["top"]
        or b["top"] + FRAME_SIZE - 1 + pad < a["top"]
        or a["left"] + FRAME_SIZE - 1 + pad < b["left"]
        or b["left"] + FRAME_SIZE - 1 + pad < a["left"]
    )


def _choose_frames(rng: np.random.Generator, h: int, w: int, count: int) -> list[dict] | None:
    colors = [int(c) for c in rng.choice(INNER_COLORS, size=count, replace=False)]
    frames = []
    for color in colors:
        placed = None
        for _ in range(200):
            top = int(rng.integers(1, h - FRAME_SIZE))
            left = int(rng.integers(1, w - FRAME_SIZE))
            hole_local = HOLE_LOCALS[int(rng.integers(len(HOLE_LOCALS)))]
            frame = _build_frame(top, left, color, hole_local)
            if all(not _boxes_too_close(frame, other, FRAME_PAD) for other in frames):
                placed = frame
                break
        if placed is None:
            return None
        frames.append(placed)
    frames.sort(key=lambda frame: (frame["top"], frame["left"]))
    return frames


def _assign_goals(rng: np.random.Generator, frames: list[dict]) -> list[dict] | None:
    count = len(frames)
    while True:
        hit_flags = [bool(x) for x in rng.integers(0, 2, size=count)]
        if any(hit_flags) and not all(hit_flags):
            break

    hit_sources = [idx for idx, flag in enumerate(hit_flags) if flag]
    choices = [
        perm
        for perm in itertools.permutations(range(count), len(hit_sources))
        if all(target != source for source, target in zip(hit_sources, perm))
    ]
    if not choices:
        return None
    targets = choices[int(rng.integers(len(choices)))]
    target_map = dict(zip(hit_sources, targets))

    goals = []
    for idx, frame in enumerate(frames):
        start_dir = int(frame["dir"])
        if hit_flags[idx]:
            goals.append({"mode": "hit", "target": int(target_map[idx])})
            continue
        exit_dirs = [
            _turn_left(start_dir),
            _turn_right(start_dir),
            start_dir,
            (start_dir + 2) % 4,
        ]
        weights = np.array([3, 3, 2, 2], dtype=float)
        exit_dir = int(rng.choice(exit_dirs, p=weights / weights.sum()))
        goals.append({"mode": "exit", "dir": exit_dir})
    return goals


def _is_terminal(state: tuple[int, int, int], goal: dict, wall_owner: dict[tuple[int, int], int], h: int, w: int) -> bool:
    r, c, d = state
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    if goal["mode"] == "exit":
        return d == goal["dir"] and not _inside(h, w, nr, nc)
    owner = wall_owner.get((nr, nc))
    return owner == goal["target"]


def _search_path(
    rng: np.random.Generator,
    h: int,
    w: int,
    frames: list[dict],
    occupied: set[tuple[int, int]],
    source_idx: int,
    goal: dict,
    max_turns: int,
) -> list[tuple[int, int, int]] | None:
    wall_owner = {}
    blocked = set()
    for idx, frame in enumerate(frames):
        for cell in frame["wall"]:
            wall_owner[cell] = idx
        blocked.update(frame["bbox"])

    start_frame = frames[source_idx]
    start = (int(start_frame["hole"][0]), int(start_frame["hole"][1]), int(start_frame["dir"]))
    blocked.discard((start[0], start[1]))

    heap = [(0, 0, start)]
    best = {start: (0, 0)}
    prev = {}

    while heap:
        turns, length, state = heapq.heappop(heap)
        if best.get(state) != (turns, length):
            continue
        if _is_terminal(state, goal, wall_owner, h, w):
            path = []
            cur = state
            while True:
                path.append(cur)
                if cur == start:
                    break
                cur = prev[cur]
            path.reverse()
            return path

        r, c, d = state
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if not _inside(h, w, nr, nc):
            continue
        if (nr, nc) in blocked or (nr, nc) in occupied:
            continue

        turn_options = [d, _turn_left(d), _turn_right(d)]
        if rng.random() < 0.5:
            turn_options[1], turn_options[2] = turn_options[2], turn_options[1]

        for nd in turn_options:
            next_turns = turns + (nd != d)
            if next_turns > max_turns:
                continue
            next_state = (nr, nc, nd)
            next_cost = (next_turns, length + 1)
            if next_cost < best.get(next_state, (10**9, 10**9)):
                best[next_state] = next_cost
                prev[next_state] = state
                heapq.heappush(heap, (next_turns, length + 1, next_state))

    return None


def _route_beams(
    rng: np.random.Generator, h: int, w: int, frames: list[dict], goals: list[dict]
) -> dict[int, list[tuple[int, int, int]]] | None:
    order = sorted(range(len(frames)), key=lambda idx: 0 if goals[idx]["mode"] == "hit" else 1)
    occupied = set()
    paths = {}
    for idx in order:
        path = None
        for max_turns in (2, 3, 4):
            path = _search_path(rng, h, w, frames, occupied, idx, goals[idx], max_turns=max_turns)
            if path is not None:
                break
        if path is None:
            return None
        occupied.update((r, c) for r, c, _d in path)
        paths[idx] = path
    return paths


def _render_grid(h: int, w: int, frames: list[dict], paths: dict[int, list[tuple[int, int, int]]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for frame in frames:
        top = int(frame["top"])
        left = int(frame["left"])
        grid[top + 1 : top + 3, left + 1 : left + 3] = int(frame["inner_color"])
        for r, c in frame["wall"]:
            grid[r, c] = GREEN

    for path in paths.values():
        for prev_state, cur_state in zip(path, path[1:]):
            prev_dir = prev_state[2]
            cur_dir = cur_state[2]
            diff = (cur_dir - prev_dir) % 4
            if diff == 3:
                grid[cur_state[0], cur_state[1]] = AZURE
            elif diff == 1:
                grid[cur_state[0], cur_state[1]] = MAGENTA
    return grid


def _valid_generated_grid(grid: np.ndarray, expected_count: int) -> bool:
    frames = _find_frames(grid)
    if len(frames) != expected_count:
        return False
    try:
        _trace_beams(grid, frames, strict=True)
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        frame_count = int(rng.integers(2, 4))
        h = int(rng.integers(MIN_GRID + 2 * (frame_count - 2), MAX_GRID + 1))
        w = int(rng.integers(MIN_GRID + 2 * (frame_count - 2), MAX_GRID + 1))

        frames = _choose_frames(rng, h, w, frame_count)
        if frames is None:
            continue

        goals = _assign_goals(rng, frames)
        if goals is None:
            continue

        paths = _route_beams(rng, h, w, frames, goals)
        if paths is None:
            continue

        turn_total = sum(
            1
            for path in paths.values()
            for prev_state, cur_state in zip(path, path[1:])
            if prev_state[2] != cur_state[2]
        )
        if turn_total == 0:
            continue

        grid = _render_grid(h, w, frames, paths)
        if not _valid_generated_grid(grid, frame_count):
            continue
        return grid

    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
