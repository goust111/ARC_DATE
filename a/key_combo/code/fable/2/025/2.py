from __future__ import annotations

import numpy as np


PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
EMITTER_COUNT_MIN = 2
EMITTER_COUNT_MAX = 4
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 22
MAX_GENERATE_ATTEMPTS = 300
ARC_COLORS = np.arange(1, 10, dtype=int)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_value(orient: str, const: int, point: tuple[int, int]) -> int:
    r, c = point
    if orient == "\\":
        return int(r - c - const)
    return int(r + c - const)


def _mirror_col(orient: str, const: int, row: int) -> int:
    if orient == "\\":
        return int(row - const)
    return int(const - row)


def _reflect_point(orient: str, const: int, point: tuple[int, int]) -> tuple[int, int]:
    r, c = point
    if orient == "\\":
        return int(c + const), int(r - const)
    return int(const - c), int(const - r)


def _vertical_step(orient: str, horizontal_step: int) -> int:
    if orient == "\\":
        return 1 if horizontal_step > 0 else -1
    return -1 if horizontal_step > 0 else 1


def _ray_segments(
    shape: tuple[int, int],
    orient: str,
    const: int,
    source: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]], str]:
    rows, cols = map(int, shape)
    r, c = map(int, source)
    mirror_c = _mirror_col(orient, const, r)
    assert 0 <= mirror_c < cols
    assert c != mirror_c

    hstep = 1 if mirror_c > c else -1
    side = "left" if c < mirror_c else "right"
    horiz = [(r, cc) for cc in range(c + hstep, mirror_c, hstep)]

    vstep = _vertical_step(orient, hstep)
    vert: list[tuple[int, int]] = []
    rr = r + vstep
    while 0 <= rr < rows:
        vert.append((rr, mirror_c))
        rr += vstep

    return horiz, (r, mirror_c), vert, side


def _infer_pair(
    p: tuple[int, int], q: tuple[int, int]
) -> tuple[str, int]:
    (r1, c1), (r2, c2) = p, q
    if r1 + c1 == r2 + c2:
        const = r1 - c2
        return "\\", int(const)
    if r1 - c1 == r2 - c2:
        const = r1 + c2
        return "/", int(const)
    raise AssertionError("pair does not define a 45-degree mirror")


def _infer_mirror(
    grid: np.ndarray, bg: int
) -> tuple[str, int, dict[int, tuple[tuple[int, int], tuple[int, int]]], dict[int, tuple[int, int]]]:
    coords_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(grid != bg):
        coords_by_color.setdefault(int(grid[r, c]), []).append((int(r), int(c)))

    pair_points: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    single_points: dict[int, tuple[int, int]] = {}
    candidates: set[tuple[str, int]] = set()

    for color, pts in coords_by_color.items():
        pts = sorted(pts)
        if len(pts) == 1:
            single_points[color] = pts[0]
        elif len(pts) == 2:
            p, q = pts
            orient, const = _infer_pair(p, q)
            assert _reflect_point(orient, const, p) == q
            assert _reflect_point(orient, const, q) == p
            assert _line_value(orient, const, p) != 0
            pair_points[color] = (p, q)
            candidates.add((orient, const))
        else:
            raise AssertionError("every foreground color must appear once or twice")

    assert len(pair_points) >= 2
    assert len(single_points) >= 1
    assert len(candidates) == 1
    orient, const = next(iter(candidates))
    return orient, int(const), pair_points, single_points


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    orient, const, _pairs, emitters = _infer_mirror(g, bg)

    out = g.copy()
    for color, source in sorted(emitters.items(), key=lambda item: item[1]):
        horiz, mirror_cell, vert, _side = _ray_segments(g.shape, orient, const, source)
        mr, mc = mirror_cell
        assert int(g[mr, mc]) == bg

        for r, c in horiz:
            assert int(g[r, c]) == bg
            assert int(out[r, c]) in (bg, color)
            out[r, c] = int(color)
        for r, c in vert:
            assert int(g[r, c]) == bg
            assert int(out[r, c]) in (bg, color)
            out[r, c] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer an invisible 45-degree mirror from reflected color pairs, then bend each singleton's horizontal ray into a vertical one.

    Input structure:
    - A black background contains several colored singleton pixels plus several colors that appear exactly twice.
    - Every repeated color forms one mirrored pair across the same unseen 45-degree diagonal line, and there are at least two such pairs.
    - Every singleton is an emitter on a row that meets that diagonal, and the cells from the emitter to the mirror are background.

    Rule:
    - Separate repeated-color pairs from singleton emitters by counting occurrences of each foreground color.
    - Reconstruct the unique 45-degree mirror line shared by all pairs; the line itself always stays black and never becomes visible.
    - For each singleton, extend its own color horizontally toward the mirror without recoloring the mirror cell, then continue from that reflection point vertically in the direction determined by whether the mirror is `/` or `\\`.

    Key insight:
    - The mirror is encoded only indirectly: repeated colors are the reflected evidence, while singleton colors are ray sources whose black kink at the mirror cell reveals where the invisible diagonal must be.
    """
    return _expected_output(grid)


def _expand4(
    cells: set[tuple[int, int]], shape: tuple[int, int]
) -> set[tuple[int, int]]:
    rows, cols = map(int, shape)
    out = set(cells)
    for r, c in list(cells):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                out.add((rr, cc))
    return out


def _visible_emitter_candidates(
    shape: tuple[int, int],
    orient: str,
    const: int,
    blocked: set[tuple[int, int]],
) -> list[dict]:
    rows, cols = map(int, shape)
    candidates: list[dict] = []
    for r in range(rows):
        mirror_c = _mirror_col(orient, const, r)
        if not (0 <= mirror_c < cols):
            continue
        for c in range(cols):
            source = (r, c)
            if c == mirror_c or source in blocked:
                continue
            horiz, mirror_cell, vert, side = _ray_segments(shape, orient, const, source)
            if len(horiz) < 1 or len(vert) < 2:
                continue
            path = set(horiz + vert)
            if mirror_cell in blocked or path & blocked:
                continue
            candidates.append(
                {
                    "src": source,
                    "side": side,
                    "row": r,
                    "path": path,
                }
            )
    return candidates


def _shuffled(items: list, rng: np.random.Generator) -> list:
    if not items:
        return []
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def _greedy_emitters(
    candidates: list[dict],
    rng: np.random.Generator,
    want_both_sides: bool,
) -> list[dict]:
    chosen: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    used_rows: set[int] = set()

    if want_both_sides:
        for side in _shuffled(["left", "right"], rng):
            placed = False
            for cand in _shuffled([c for c in candidates if c["side"] == side], rng):
                if cand["row"] in used_rows or cand["src"] in blocked or cand["path"] & blocked:
                    continue
                chosen.append(cand)
                blocked.add(cand["src"])
                blocked |= cand["path"]
                used_rows.add(int(cand["row"]))
                placed = True
                break
            if not placed:
                return []

    for cand in _shuffled(candidates, rng):
        if len(chosen) >= EMITTER_COUNT_MAX:
            break
        if cand["row"] in used_rows or cand["src"] in blocked or cand["path"] & blocked:
            continue
        chosen.append(cand)
        blocked.add(cand["src"])
        blocked |= cand["path"]
        used_rows.add(int(cand["row"]))

    return chosen


def _choose_emitters(
    shape: tuple[int, int],
    orient: str,
    const: int,
    rng: np.random.Generator,
) -> list[dict] | None:
    candidates = _visible_emitter_candidates(shape, orient, const, blocked=set())
    if len(candidates) < EMITTER_COUNT_MIN:
        return None

    left = [cand for cand in candidates if cand["side"] == "left"]
    right = [cand for cand in candidates if cand["side"] == "right"]
    best: list[dict] = []

    for _ in range(60):
        want_both = bool(left and right and rng.random() < 0.75)
        trial = _greedy_emitters(candidates, rng, want_both_sides=want_both)
        if len(trial) > len(best):
            best = trial
        if len(best) >= EMITTER_COUNT_MAX:
            break

    if len(best) < EMITTER_COUNT_MIN:
        return None

    target = int(rng.integers(EMITTER_COUNT_MIN, min(EMITTER_COUNT_MAX, len(best)) + 1))
    best = _shuffled(best, rng)
    if target >= 2:
        left_best = [cand for cand in best if cand["side"] == "left"]
        right_best = [cand for cand in best if cand["side"] == "right"]
        if left_best and right_best and rng.random() < 0.75:
            subset = [left_best[0], right_best[0]]
            used = {left_best[0]["src"], right_best[0]["src"]}
            for cand in best:
                if len(subset) >= target:
                    break
                if cand["src"] not in used:
                    subset.append(cand)
                    used.add(cand["src"])
            return subset

    return best[:target]


def _pair_candidates(
    shape: tuple[int, int],
    orient: str,
    const: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rows, cols = map(int, shape)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for r in range(rows):
        for c in range(cols):
            point = (r, c)
            if point in blocked or _line_value(orient, const, point) == 0:
                continue
            if abs(_line_value(orient, const, point)) < 2:
                continue

            mate = _reflect_point(orient, const, point)
            rr, cc = mate
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            if mate in blocked or mate == point:
                continue
            if abs(r - rr) + abs(c - cc) < 4:
                continue

            pair = tuple(sorted((point, mate)))
            if pair in seen:
                continue
            seen.add(pair)
            candidates.append(pair)

    return candidates


def _greedy_pairs(
    candidates: list[tuple[tuple[int, int], tuple[int, int]]],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
    current_blocked = set(blocked)

    for a, b in _shuffled(candidates, rng):
        if a in current_blocked or b in current_blocked:
            continue
        chosen.append((a, b))
        current_blocked |= _expand4({a, b}, shape)
        if len(chosen) >= PAIR_COUNT_MAX:
            break

    return chosen


def _choose_pairs(
    shape: tuple[int, int],
    orient: str,
    const: int,
    emitters: list[dict],
    rng: np.random.Generator,
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    reserved = {cand["src"] for cand in emitters}
    for cand in emitters:
        reserved |= cand["path"]
    base_blocked = _expand4(reserved, shape)

    candidates = _pair_candidates(shape, orient, const, base_blocked)
    if len(candidates) < PAIR_COUNT_MIN:
        return None

    best: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for _ in range(60):
        trial = _greedy_pairs(candidates, base_blocked, shape, rng)
        if len(trial) > len(best):
            best = trial
        if len(best) >= PAIR_COUNT_MAX:
            break

    if len(best) < PAIR_COUNT_MIN:
        return None

    target = int(rng.integers(PAIR_COUNT_MIN, min(PAIR_COUNT_MAX, len(best)) + 1))
    return _shuffled(best, rng)[:target]


def _line_constants(shape: tuple[int, int], orient: str) -> list[int]:
    rows, cols = map(int, shape)
    if orient == "\\":
        return list(range(-(cols - 1), rows))
    return list(range(rows + cols - 1))


def _render_input(
    shape: tuple[int, int],
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    emitters: list[dict],
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    colors = rng.choice(ARC_COLORS, size=len(pairs) + len(emitters), replace=False)

    for color, (a, b) in zip(colors[: len(pairs)], pairs):
        grid[a] = int(color)
        grid[b] = int(color)
    for color, cand in zip(colors[len(pairs) :], emitters):
        grid[cand["src"]] = int(color)

    return grid


def _validate_generated_instance(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        orient, const, pairs, emitters = _infer_mirror(grid, bg)
        assert len(pairs) >= PAIR_COUNT_MIN
        assert len(emitters) >= EMITTER_COUNT_MIN

        painted: set[tuple[int, int]] = set()
        for color, (p, q) in pairs.items():
            assert _reflect_point(orient, const, p) == q
            assert _reflect_point(orient, const, q) == p
            assert color != bg

        for color, source in emitters.items():
            horiz, mirror_cell, vert, _side = _ray_segments(grid.shape, orient, const, source)
            mr, mc = mirror_cell
            assert int(grid[mr, mc]) == bg
            for cell in horiz + vert:
                assert cell not in painted
                assert int(grid[cell]) == bg
                painted.add(cell)
            assert horiz
            assert len(vert) >= 2
            assert color != bg

        expected = _expected_output(grid)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        return True
    except AssertionError:
        return False


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    cols = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    shape = (rows, cols)
    orient = "\\" if rng.random() < 0.5 else "/"

    for const in _shuffled(_line_constants(shape, orient), rng):
        emitters = _choose_emitters(shape, orient, const, rng)
        if emitters is None:
            continue
        pairs = _choose_pairs(shape, orient, const, emitters, rng)
        if pairs is None:
            continue
        grid = _render_input(shape, pairs, emitters, rng)
        if _validate_generated_instance(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-diagonal-mirror input")
