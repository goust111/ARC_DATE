from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_TO_DIR = {"top": 2, "right": 3, "bottom": 0, "left": 1}
TURN_OPTIONS = [0, 1, 2, 3]
TURN_WEIGHTS = [1, 3, 3, 2]
MAX_GENERATE_ATTEMPTS = 400
MAX_BEAM_ATTEMPTS = 80
MIN_SEG_LEN = 2


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_edge(h: int, w: int, r: int, c: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _is_corner(h: int, w: int, r: int, c: int) -> bool:
    return (r in (0, h - 1)) and (c in (0, w - 1))


def _left_dir(d: int) -> int:
    return (d - 1) % 4


def _right_dir(d: int) -> int:
    return (d + 1) % 4


def _direction_from_edge(h: int, w: int, r: int, c: int) -> int:
    if r == 0 and 0 < c < w - 1:
        return SIDE_TO_DIR["top"]
    if r == h - 1 and 0 < c < w - 1:
        return SIDE_TO_DIR["bottom"]
    if c == 0 and 0 < r < h - 1:
        return SIDE_TO_DIR["left"]
    if c == w - 1 and 0 < r < h - 1:
        return SIDE_TO_DIR["right"]
    raise AssertionError("emitter must lie on exactly one non-corner edge cell")


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("waypoints must be axis-aligned")


def _path_cells(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for i in range(len(points) - 1):
        cells.update(_segment_cells(points[i], points[i + 1]))
    return cells


def _reflect(direction: int, mirror: str) -> int:
    if mirror == "\\":
        return {0: 3, 1: 2, 2: 1, 3: 0}[direction]
    if mirror == "/":
        return {0: 1, 1: 0, 2: 3, 3: 2}[direction]
    raise AssertionError("unknown mirror orientation")


def _mirror_from_turn(in_dir: int, out_dir: int) -> str:
    for mirror in ("\\", "/"):
        if _reflect(in_dir, mirror) == out_dir:
            return mirror
    raise AssertionError("turn is not a legal mirror reflection")


def _mirror_conditions(grid: np.ndarray, r: int, c: int) -> tuple[bool, bool]:
    h, w = grid.shape

    def is_black_or_outside(rr: int, cc: int) -> bool:
        return not _in_bounds(h, w, rr, cc) or int(grid[rr, cc]) == BLACK

    backslash = is_black_or_outside(r - 1, c - 1) and is_black_or_outside(r + 1, c + 1)
    slash = is_black_or_outside(r - 1, c + 1) and is_black_or_outside(r + 1, c - 1)
    return backslash, slash


def _infer_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirrors: dict[tuple[int, int], str] = {}
    for r, c in np.argwhere(grid == ORANGE):
        rr = int(r)
        cc = int(c)
        backslash, slash = _mirror_conditions(grid, rr, cc)
        assert backslash ^ slash, "each orange pixel must encode exactly one mirror orientation"
        mirrors[(rr, cc)] = "\\" if backslash else "/"
    return mirrors


def _collect_roles(grid: np.ndarray) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]], dict[tuple[int, int], str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mirrors = _infer_mirrors(g)
    emitters: dict[int, tuple[int, int]] = {}
    targets: dict[int, tuple[int, int]] = {}

    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val in (BLACK, ORANGE):
                continue
            if _is_edge(h, w, r, c):
                assert not _is_corner(h, w, r, c), "emitters may not occupy corners"
                assert val not in emitters, "each color must have exactly one emitter"
                emitters[val] = (r, c)
            else:
                assert val not in targets, "each color must have exactly one target"
                targets[val] = (r, c)

    assert emitters, "expected at least one emitter"
    assert set(emitters) == set(targets), "emitter colors must match target colors"
    return emitters, targets, mirrors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw every beam path from an edge emitter to its same-color target, reflecting on orange mirrors.

    Input structure:
    - Background is black.
    - Each foreground color other than orange appears exactly twice: one edge emitter and one interior target.
    - Orange pixels are mirrors whose orientation is inferred from diagonal blackness.

    Rule:
    - Each emitter fires inward orthogonally from the edge it touches.
    - Traversed black cells are painted with the emitter color.
    - Hitting a mirror rotates the direction by 90 degrees; the mirror stays orange.
    - Tracing stops exactly on the same-color target, which keeps its color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    emitters, targets, mirrors = _collect_roles(g)
    out = g.copy()

    for color in sorted(emitters):
        r, c = emitters[color]
        tr, tc = targets[color]
        direction = _direction_from_edge(h, w, r, c)
        seen = {(r, c, direction)}

        while True:
            dr, dc = DIRS[direction]
            nr = r + dr
            nc = c + dc
            assert _in_bounds(h, w, nr, nc), "beam must stay inside the grid until its target"

            if (nr, nc) == (tr, tc):
                assert int(g[nr, nc]) == int(color), "beam must terminate on its same-color target"
                break

            if (nr, nc) in mirrors:
                direction = _reflect(direction, mirrors[(nr, nc)])
                r, c = nr, nc
            else:
                assert int(g[nr, nc]) == BLACK, "beams may pass only through black cells before the target"
                out[nr, nc] = int(color)
                r, c = nr, nc

            state = (r, c, direction)
            assert state not in seen, "beam paths must not cycle"
            seen.add(state)

    return out


def _shuffle(seq):
    seq = list(seq)
    random.shuffle(seq)
    return seq


def _feature_clear(
    point: tuple[int, int],
    h: int,
    w: int,
    feature_cells: set[tuple[int, int]],
) -> bool:
    r, c = point
    if not _in_bounds(h, w, r, c):
        return False
    if point in feature_cells:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in feature_cells:
                return False
    return True


def _edge_positions(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    raise AssertionError("unknown side")


def _reachable_steps(
    h: int,
    w: int,
    start: tuple[int, int],
    direction: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], list[tuple[int, int]]]]:
    dr, dc = DIRS[direction]
    r, c = start
    path: list[tuple[int, int]] = []
    steps = []
    step = 0
    while True:
        step += 1
        r += dr
        c += dc
        if not _in_bounds(h, w, r, c) or (r, c) in blocked:
            break
        path.append((r, c))
        steps.append((step, (r, c), path.copy()))
    return steps


def _search_beam_path(
    h: int,
    w: int,
    start: tuple[int, int],
    direction: int,
    turns_left: int,
    used_path: set[tuple[int, int]],
    used_features: set[tuple[int, int]],
    local_features: list[tuple[int, int]],
    local_used: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    blocked = used_path | local_used
    steps = _reachable_steps(h, w, start, direction, blocked)
    if turns_left == 0:
        final_choices = []
        for step, end, seg in steps:
            if step < MIN_SEG_LEN:
                continue
            if _is_edge(h, w, end[0], end[1]):
                continue
            if not _feature_clear(end, h, w, used_features | set(local_features)):
                continue
            final_choices.append((end, seg))
        random.shuffle(final_choices)
        if not final_choices:
            return None
        end, _seg = final_choices[0]
        return [end]

    turn_choices = []
    for step, turn_cell, seg in steps:
        if step < MIN_SEG_LEN:
            continue
        if _is_edge(h, w, turn_cell[0], turn_cell[1]):
            continue
        if not _feature_clear(turn_cell, h, w, used_features | set(local_features)):
            continue
        turn_choices.append((turn_cell, seg))
    random.shuffle(turn_choices)

    for turn_cell, seg in turn_choices:
        next_local_features = local_features + [turn_cell]
        next_local_used = set(local_used)
        next_local_used.update(seg)
        next_dirs = _shuffle([_left_dir(direction), _right_dir(direction)])
        for next_direction in next_dirs:
            tail = _search_beam_path(
                h=h,
                w=w,
                start=turn_cell,
                direction=next_direction,
                turns_left=turns_left - 1,
                used_path=used_path,
                used_features=used_features,
                local_features=next_local_features,
                local_used=next_local_used,
            )
            if tail is not None:
                return [turn_cell] + tail
    return None


def _build_beam_path(
    h: int,
    w: int,
    side: str,
    turns: int,
    used_path: set[tuple[int, int]],
    used_features: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    emitters = _shuffle(_edge_positions(side, h, w))
    for emitter in emitters:
        if emitter in used_path:
            continue
        if not _feature_clear(emitter, h, w, used_features):
            continue
        direction = SIDE_TO_DIR[side]
        tail = _search_beam_path(
            h=h,
            w=w,
            start=emitter,
            direction=direction,
            turns_left=turns,
            used_path=used_path,
            used_features=used_features,
            local_features=[emitter],
            local_used={emitter},
        )
        if tail is not None:
            return [emitter] + tail
    return None


def _turn_orientations(points: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    for i in range(1, len(points) - 1):
        prev_cell = points[i - 1]
        cur_cell = points[i]
        next_cell = points[i + 1]
        in_seg = _segment_cells(prev_cell, cur_cell)
        out_seg = _segment_cells(cur_cell, next_cell)
        in_dir = None
        out_dir = None
        if len(in_seg) >= 2:
            a, b = in_seg[-2], in_seg[-1]
            in_dir = DIRS.index((b[0] - a[0], b[1] - a[1]))
        if len(out_seg) >= 2:
            a, b = out_seg[0], out_seg[1]
            out_dir = DIRS.index((b[0] - a[0], b[1] - a[1]))
        assert in_dir is not None and out_dir is not None
        orientations[cur_cell] = _mirror_from_turn(in_dir, out_dir)
    return orientations


def _cue_candidates(r: int, c: int, mirror: str) -> list[tuple[int, int]]:
    if mirror == "\\":
        return [(r - 1, c + 1), (r + 1, c - 1)]
    return [(r - 1, c - 1), (r + 1, c + 1)]


def _build_input(
    h: int,
    w: int,
    beams: list[dict[str, object]],
    cues: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for beam in beams:
        color = int(beam["color"])
        points = beam["points"]
        grid[points[0]] = color
        grid[points[-1]] = color
        for cell in points[1:-1]:
            grid[cell] = ORANGE
    for cell in cues:
        grid[cell] = ORANGE
    return grid


def _paint_expected(inp: np.ndarray, beams: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for beam in beams:
        color = int(beam["color"])
        points = beam["points"]
        for i in range(len(points) - 1):
            seg = _segment_cells(points[i], points[i + 1])
            for cell in seg[1:-1]:
                out[cell] = color
    return out


def _assign_cues(
    h: int,
    w: int,
    beams: list[dict[str, object]],
    used_path: set[tuple[int, int]],
    used_features: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    actual_mirrors: list[tuple[tuple[int, int], str]] = []
    for beam in beams:
        points = beam["points"]
        orientations = beam["orientations"]
        for cell in points[1:-1]:
            actual_mirrors.append((cell, orientations[cell]))

    candidate_lists = []
    for cell, mirror in actual_mirrors:
        r, c = cell
        choices = []
        for cue in _cue_candidates(r, c, mirror):
            if not _in_bounds(h, w, cue[0], cue[1]):
                continue
            if cue in used_path or cue in used_features:
                continue
            choices.append(cue)
        if not choices:
            return None
        candidate_lists.append((cell, mirror, _shuffle(choices)))

    def backtrack(idx: int, chosen: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
        if idx == len(candidate_lists):
            inp = _build_input(h, w, beams, chosen)
            try:
                mirrors = _infer_mirrors(inp)
            except AssertionError:
                return None
            for cell, expected in actual_mirrors:
                if mirrors.get(cell) != expected:
                    return None
            return chosen

        cell, _mirror, choices = candidate_lists[idx]
        for cue in choices:
            if cue in chosen:
                continue
            trial = set(chosen)
            trial.add(cue)
            result = backtrack(idx + 1, trial)
            if result is not None:
                return result
        return None

    return backtrack(0, set())


def _max_pairs(h: int, w: int) -> int:
    area = h * w
    if area >= 450:
        return 5
    if area >= 320:
        return 4
    return 3


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(15, 25)
    w = random.randint(15, 25)
    num_pairs = random.randint(2, _max_pairs(h, w))
    colors = random.sample(PALETTE, num_pairs)

    used_path: set[tuple[int, int]] = set()
    used_features: set[tuple[int, int]] = set()
    beams: list[dict[str, object]] = []

    for color in colors:
        placed = False
        for _ in range(MAX_BEAM_ATTEMPTS):
            side = random.choice(list(SIDE_TO_DIR))
            turns = random.choices(TURN_OPTIONS, weights=TURN_WEIGHTS, k=1)[0]
            path = _build_beam_path(
                h=h,
                w=w,
                side=side,
                turns=turns,
                used_path=used_path,
                used_features=used_features,
            )
            if path is None:
                continue
            beam_cells = _path_cells(path)
            orientations = _turn_orientations(path)
            beams.append({"color": color, "points": path, "orientations": orientations})
            used_path.update(beam_cells)
            used_features.update(path)
            placed = True
            break
        if not placed:
            return None

    total_turns = sum(len(beam["points"]) - 2 for beam in beams)
    if total_turns == 0:
        return None

    cues = _assign_cues(h, w, beams, used_path, used_features)
    if cues is None:
        return None

    inp = _build_input(h, w, beams, cues)
    out = _paint_expected(inp, beams)

    try:
        solved = main(inp)
        emitters, targets, mirrors = _collect_roles(inp)
    except AssertionError:
        return None

    if len(emitters) != num_pairs or len(targets) != num_pairs:
        return None
    if len(mirrors) < total_turns:
        return None
    if not np.array_equal(solved, out):
        return None

    slash_count = sum(1 for beam in beams for m in beam["orientations"].values() if m == "/")
    backslash_count = total_turns - slash_count
    if total_turns >= 2 and slash_count == 0:
        return None
    if total_turns >= 2 and backslash_count == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    """
    Sample a valid laser-and-mirror instance.

    The generator first creates non-overlapping orthogonal beam polylines, then
    erases the beam trails back to black and leaves only emitters, targets, and
    orange mirrors. Extra orange cue mirrors are added only where needed to make
    each used mirror's orientation uniquely inferable from diagonal neighbors.
    """
    for _ in range(MAX_GENERATE_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid ARC input")
