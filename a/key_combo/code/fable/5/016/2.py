from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
GRAY = 5
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _component(mask: np.ndarray, seed: tuple[int, int]) -> set[tuple[int, int]]:
    """Return one 4-connected component, using the shared ARC (row, col) convention."""
    h, w = mask.shape
    todo = deque([seed])
    seen = {seed}
    while todo:
        r, c = todo.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                todo.append((rr, cc))
    return seen


def _contour_slots(core: tuple[int, int, int, int]) -> list[dict]:
    """Clockwise unit edges of the recovered rectangle, starting at its top-left vertex."""
    r0, c0, r1, c1 = core
    slots: list[dict] = []

    # Each non-corner slot records its supporting core cell and the one-cell
    # outward location.  Corner edges are retained in the contour indexing but
    # deliberately cannot carry a feature.
    for c in range(c0, c1 + 1):
        slots.append({"base": (r0, c), "out": (r0 - 1, c), "usable": c0 < c < c1, "side": 0})
    for r in range(r0, r1 + 1):
        slots.append({"base": (r, c1), "out": (r, c1 + 1), "usable": r0 < r < r1, "side": 1})
    for c in range(c1, c0 - 1, -1):
        slots.append({"base": (r1, c), "out": (r1 + 1, c), "usable": c0 < c < c1, "side": 2})
    for r in range(r1, r0 - 1, -1):
        slots.append({"base": (r, c0), "out": (r, c0 - 1), "usable": r0 < r < r1, "side": 3})
    return slots


def _candidate_core(mask: np.ndarray, core: tuple[int, int, int, int]) -> tuple[bool, list[tuple[int, str]]]:
    """Check whether mask is exactly a rectangle plus isolated 1-pixel tabs/notches."""
    h, w = mask.shape
    r0, c0, r1, c1 = core
    if not (0 <= r0 < r1 < h and 0 <= c0 < c1 < w):
        return False, []
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False, []
    if not all(mask[r, c] for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
        return False, []
    if not bool(mask[r0 + 1 : r1, c0 + 1 : c1].all()):
        return False, []

    slots = _contour_slots(core)
    usable = {i: slot for i, slot in enumerate(slots) if slot["usable"]}
    base_to_i = {slot["base"]: i for i, slot in usable.items()}
    out_to_i = {slot["out"]: i for i, slot in usable.items()}

    features: dict[int, str] = {}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if mask[r, c]:
                continue
            i = base_to_i.get((r, c))
            if i is None or i in features:
                return False, []
            features[i] = "L"  # missing boundary cell = left turn

    for r, c in map(tuple, np.argwhere(mask)):
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        i = out_to_i.get((r, c))
        if i is None or i in features:
            return False, []
        features[i] = "R"  # outward boundary cell = right turn

    ordered = sorted(features.items())
    if len(ordered) < 2:
        return False, []
    # Adjacent defects merge visually and no longer form unambiguous single-pixel features.
    period = len(slots)
    positions = [p for p, _ in ordered]
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    gaps.append(period - positions[-1] + positions[0])
    if min(gaps) < 2:
        return False, []
    return True, ordered


def _recover_tape(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[tuple[int, str]]]:
    """Recover the sole colored instruction object and its unique clean rectangle."""
    other_colors = set(map(int, np.unique(grid))) - {BG, BLUE, GRAY}
    assert len(other_colors) == 1, "the instruction band must have one unambiguous color"
    color = other_colors.pop()
    mask = grid == color
    pts = np.argwhere(mask)
    assert pts.size, "instruction band is missing"
    seed = tuple(map(int, pts[0]))
    assert len(_component(mask, seed)) == int(mask.sum()), "instruction band must be 4-connected"

    min_r, min_c = map(int, pts.min(axis=0))
    max_r, max_c = map(int, pts.max(axis=0))
    row_starts = sorted({min_r, min_r + 1})
    row_ends = sorted({max_r, max_r - 1})
    col_starts = sorted({min_c, min_c + 1})
    col_ends = sorted({max_c, max_c - 1})

    valid: list[tuple[tuple[int, int, int, int], list[tuple[int, str]]]] = []
    for r0 in row_starts:
        for r1 in row_ends:
            for c0 in col_starts:
                for c1 in col_ends:
                    core = (r0, c0, r1, c1)
                    ok, features = _candidate_core(mask, core)
                    if ok:
                        valid.append((core, features))
    assert len(valid) == 1, "tabs and notches must determine one unique rectangular body"
    core, features = valid[0]
    return int(color), core, features


def _decode_lengths(features: list[tuple[int, str]]) -> tuple[list[int], list[str]]:
    """Convert contour event positions to actual clockwise contour-walk distances."""
    positions = [p for p, _ in features]
    turns = [kind for _, kind in features]
    assert positions[0] > 0, "the first feature may not coincide with the traversal start"
    lengths = [positions[0]]
    # A one-cell protrusion or indentation replaces one straight contour edge by
    # three edges.  Therefore the already encountered feature contributes two
    # real boundary steps before the following feature is reached.
    lengths.extend(positions[i] - positions[i - 1] + 2 for i in range(1, len(positions)))
    assert all(length > 0 for length in lengths)
    return lengths, turns


def _initial_marker(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    gray = np.argwhere(grid == GRAY)
    assert gray.shape[0] == 1, "there must be exactly one gray start marker"
    start = tuple(map(int, gray[0]))
    adjacent_blue: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        r, c = start[0] + dr, start[1] + dc
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == BLUE:
            adjacent_blue.append((r, c))
    assert len(adjacent_blue) == 1, "one adjacent blue pixel must uniquely indicate initial direction"
    b = adjacent_blue[0]
    return start, (b[0] - start[0], b[1] - start[1])


def _execute(grid: np.ndarray, lengths: list[int], turns: list[str],
             start: tuple[int, int], direction: tuple[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r, c = start
    dr, dc = direction
    for length, turn in zip(lengths, turns):
        for _ in range(length):
            r, c = r + dr, c + dc
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "decoded polyline leaves the grid"
            out[r, c] = BLUE
        if turn == "R":
            dr, dc = dc, -dr
        else:
            dr, dc = -dc, dr
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode boundary tabs and notches as turns and draw their spacing program as a blue polyline.

    Input structure:
    - Black background contains one thick monochrome rectilinear block whose rectangular boundary has isolated one-pixel tabs and notches.
    - A unique gray start pixel lies elsewhere; its sole orthogonally adjacent blue pixel indicates the initial travel direction.

    Rule:
    - Recover the block's solid rectangular body and traverse its pixel outline clockwise from the top-left outline vertex.
    - A tab is a right turn and a notch is a left turn.  The actual outline-step distance from the start to the first feature, then between consecutive features, is the corresponding segment length; each prior one-pixel detour contributes its two extra outline steps.
    - Starting at gray, move and paint blue for each decoded length, then perform that feature's turn.  Preserve every input pixel and stop when the features are exhausted.

    Key insight:
    - The colored block is an instruction tape rather than the object to transform: defect type encodes turning while clockwise contour spacing encodes distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _color, _core, features = _recover_tape(g)
    lengths, turns = _decode_lengths(features)
    start, direction = _initial_marker(g)
    return _execute(g, lengths, turns, start, direction)


# ------------------------------- generator -------------------------------


def _program_path(lengths: list[int], turns: list[str], initial: int) -> list[tuple[int, int]] | None:
    """Simulate in local coordinates and reject crossings or retraced pixels."""
    r = c = 0
    dr, dc = DIRS4[initial]
    path = [(r, c)]
    occupied = {(r, c)}
    for length, turn in zip(lengths, turns):
        for _ in range(length):
            r, c = r + dr, c + dc
            if (r, c) in occupied:
                return None
            occupied.add((r, c))
            path.append((r, c))
        if turn == "R":
            dr, dc = dc, -dr
        else:
            dr, dc = -dc, dr
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    if max(rs) - min(rs) + 1 > 16 or max(cs) - min(cs) + 1 > 16:
        return None
    return path


def _paint_feature(patch: np.ndarray, slots: list[dict], p: int, kind: str, color: int) -> None:
    br, bc = slots[p]["base"]
    er, ec = slots[p]["out"]
    patch[br, bc] = color if kind == "R" else BG
    patch[er, ec] = color if kind == "R" else BG
    if kind == "L":
        patch[br, bc] = BG


def _make_tape() -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, str]], int] | None:
    body_h = random.randint(8, 12)
    body_w = random.randint(8, 12)
    color = random.choice([2, 3, 4, 6, 7, 8, 9])
    patch = np.zeros((body_h + 2, body_w + 2), dtype=int)
    core = (1, 1, body_h, body_w)
    patch[1 : body_h + 1, 1 : body_w + 1] = color
    slots = _contour_slots(core)
    corners = {0, body_w - 1, body_w, body_w + body_h - 1,
               2 * body_w + body_h - 1, 2 * body_w + body_h,
               len(slots) - 1}

    count = random.randint(4, 6)
    for _ in range(120):
        first = random.randint(3, 6)
        positions = [first]
        for _j in range(1, count):
            positions.append(positions[-1] + random.randint(3, 5))
        if positions[-1] >= len(slots) - 2 or any(p in corners or not slots[p]["usable"] for p in positions):
            continue

        kinds = []
        for p in positions:
            # A top-side protrusion would become the topmost start cell, so top
            # features are notches; tabs remain equally available on other sides.
            kinds.append("L" if slots[p]["side"] == 0 else random.choice(["L", "R"]))
        if "R" not in kinds[:-1] or "L" not in kinds[:-1]:
            continue
        if len(set(positions[i] - positions[i - 1] for i in range(1, count))) < 2:
            continue
        features = list(zip(positions, kinds))
        for p, kind in features:
            _paint_feature(patch, slots, p, kind, color)
        ok, recovered = _candidate_core(patch == color, core)
        if ok and recovered == features:
            return patch, core, features, color
        patch.fill(BG)
        patch[1 : body_h + 1, 1 : body_w + 1] = color
    return None


def _compose(tape: np.ndarray, path: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    pr0, pc0 = min(rs), min(cs)
    path_h, path_w = max(rs) - pr0 + 1, max(cs) - pc0 + 1
    th, tw = tape.shape

    layouts = []
    if tw + path_w + 5 <= 30 and max(th, path_h) + 2 <= 30:
        layouts.append("horizontal")
    if th + path_h + 5 <= 30 and max(tw, path_w) + 2 <= 30:
        layouts.append("vertical")
    if not layouts:
        return None
    layout = random.choice(layouts)
    reverse = bool(random.getrandbits(1))

    if layout == "horizontal":
        gap = random.randint(3, min(5, 28 - tw - path_w))
        local_h = max(th, path_h)
        local_w = tw + gap + path_w
        a_r = random.randint(0, local_h - th)
        b_r = random.randint(0, local_h - path_h)
        if not reverse:
            tape_pos, path_pos = (a_r, 0), (b_r, tw + gap)
        else:
            path_pos, tape_pos = (b_r, 0), (a_r, path_w + gap)
    else:
        gap = random.randint(3, min(5, 28 - th - path_h))
        local_h = th + gap + path_h
        local_w = max(tw, path_w)
        a_c = random.randint(0, local_w - tw)
        b_c = random.randint(0, local_w - path_w)
        if not reverse:
            tape_pos, path_pos = (0, a_c), (th + gap, b_c)
        else:
            path_pos, tape_pos = (0, b_c), (path_h + gap, a_c)

    h = random.randint(local_h + 2, min(30, local_h + 7))
    w = random.randint(local_w + 2, min(30, local_w + 7))
    off_r = random.randint(1, h - local_h - 1)
    off_c = random.randint(1, w - local_w - 1)
    tape_pos = (tape_pos[0] + off_r, tape_pos[1] + off_c)
    path_pos = (path_pos[0] + off_r, path_pos[1] + off_c)

    grid = np.zeros((h, w), dtype=int)
    tr, tc = tape_pos
    grid[tr : tr + th, tc : tc + tw] = tape

    normalized = [(r - pr0 + path_pos[0], c - pc0 + path_pos[1]) for r, c in path]
    start = normalized[0]
    first = normalized[1]
    grid[start] = GRAY
    grid[first] = BLUE
    return grid, tape_pos, start


def _mutate_feature(grid: np.ndarray, core: tuple[int, int, int, int], p: int,
                    old_kind: str, new_p: int, new_kind: str, color: int) -> np.ndarray:
    out = grid.copy()
    slots = _contour_slots(core)
    old_base, old_extra = slots[p]["base"], slots[p]["out"]
    out[old_base] = color
    out[old_extra] = BG
    base, extra = slots[new_p]["base"], slots[new_p]["out"]
    out[base] = color if new_kind == "R" else BG
    out[extra] = color if new_kind == "R" else BG
    return out


def _semantic_mutation_checks(grid: np.ndarray, output: np.ndarray, core: tuple[int, int, int, int],
                              features: list[tuple[int, str]], color: int) -> bool:
    """Reject samples where turn type, spacing, or initial direction is observationally dispensable."""
    # Turn-type counterexample: flip a nonterminal feature at the same contour position.
    p, kind = features[1]
    flipped = _mutate_feature(grid, core, p, kind, p, "L" if kind == "R" else "R", color)
    try:
        if np.array_equal(main(flipped), output):
            return False
    except (AssertionError, IndexError):
        return False

    # Spacing counterexample: shift one nonterminal feature by one clean contour edge.
    occupied = {q for q, _ in features}
    slots = _contour_slots(core)
    moved_ok = False
    for delta in (-1, 1):
        q = p + delta
        if q in occupied or not (0 <= q < len(slots)) or not slots[q]["usable"]:
            continue
        if kind == "R" and slots[q]["side"] == 0:
            continue
        moved = _mutate_feature(grid, core, p, kind, q, kind, color)
        try:
            if not np.array_equal(main(moved), output):
                moved_ok = True
                break
        except (AssertionError, IndexError):
            continue
    if not moved_ok:
        return False

    # Initial-direction counterexample: rotate the adjacent blue direction if the
    # alternate full program stays inside the same grid.
    start, direction = _initial_marker(grid)
    for alt in ((direction[1], -direction[0]), (-direction[1], direction[0]),
                (-direction[0], -direction[1])):
        nr, nc = start[0] + alt[0], start[1] + alt[1]
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or grid[nr, nc] != BG:
            continue
        changed = grid.copy()
        changed[start[0] + direction[0], start[1] + direction[1]] = BG
        changed[nr, nc] = BLUE
        try:
            altered_out = main(changed)
        except (AssertionError, IndexError):
            continue
        if not np.array_equal(altered_out, output):
            return True
    return False


def _generate_attempt() -> np.ndarray | None:
    tape_data = _make_tape()
    if tape_data is None:
        return None
    tape, local_core, features, color = tape_data
    lengths, turns = _decode_lengths(features)
    initial = random.randrange(4)
    path = _program_path(lengths, turns, initial)
    if path is None:
        return None
    composed = _compose(tape, path)
    if composed is None:
        return None
    grid, tape_pos, _start = composed
    global_core = tuple((local_core[i] + tape_pos[i % 2]) for i in range(4))

    try:
        decoded_color, decoded_core, decoded_features = _recover_tape(grid)
        decoded_lengths, decoded_turns = _decode_lengths(decoded_features)
        start, decoded_direction = _initial_marker(grid)
        output = main(grid)
    except (AssertionError, IndexError):
        return None
    if decoded_color != color or decoded_core != global_core:
        return None
    expected_features = [(p, kind) for p, kind in features]
    if decoded_features != expected_features or decoded_lengths != lengths or decoded_turns != turns:
        return None
    if decoded_direction != DIRS4[initial]:
        return None
    expected = _execute(grid, lengths, turns, start, DIRS4[initial])
    if not np.array_equal(output, expected) or np.array_equal(output, grid):
        return None
    if len(set(lengths)) < 2 or not {"L", "R"}.issubset(set(turns[:-1])):
        return None
    if not _semantic_mutation_checks(grid, output, global_core, features, color):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instruction band and a separated drawing area."""
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a non-degenerate boundary-instruction sample")
