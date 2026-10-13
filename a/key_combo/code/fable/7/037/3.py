from __future__ import annotations

import itertools
import random

import numpy as np


GRAY = 5
PALETTE = tuple(range(10))
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (row, column) coordinate arrays."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result: list[np.ndarray] = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        cells = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if 0 <= q[0] < h and 0 <= q[1] < w and q in unseen:
                    unseen.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox, following arc_utils' (r0, c0, r1, c1) convention."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _periodic_patterns(shape: tuple[int, int], a: int, b: int):
    """Yield the supported two-color stripe/checker grounds, without duplicates."""
    h, w = shape
    rr, cc = np.indices(shape)
    seen: set[bytes] = set()

    def emit(parity: np.ndarray):
        pattern = np.where(parity == 0, a, b).astype(int)
        key = pattern.tobytes()
        if key not in seen:
            seen.add(key)
            return pattern
        return None

    for band in (1, 2, 3):
        for phase in range(2 * band):
            for parity in (((rr + phase) // band) % 2, ((cc + phase) // band) % 2):
                pattern = emit(parity)
                if pattern is not None:
                    yield pattern

    for rb in (1, 2, 3):
        for cb in (1, 2, 3):
            for rp in range(2 * rb):
                for cp in range(2 * cb):
                    parity = (((rr + rp) // rb) + ((cc + cp) // cb)) % 2
                    pattern = emit(parity)
                    if pattern is not None:
                        yield pattern


def _infer_ground(g: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    colors = [int(x) for x in np.unique(g) if int(x) != GRAY]
    assert len(colors) >= 3, "expected two ground colors and at least one slider color"
    best_score = -1
    best: dict[bytes, tuple[np.ndarray, tuple[int, int]]] = {}

    for a, b in itertools.combinations(colors, 2):
        observed = (g == a) | (g == b)
        score = int(observed.sum())
        if not np.any(g == a) or not np.any(g == b):
            continue
        for pattern in _periodic_patterns(g.shape, a, b):
            if np.all(g[observed] == pattern[observed]):
                if score > best_score:
                    best_score = score
                    best.clear()
                if score == best_score:
                    best[pattern.tobytes()] = (pattern, (a, b))

    assert len(best) == 1, "visible cells must determine one periodic ground"
    return next(iter(best.values()))


def _trail_length(g: np.ndarray, box: tuple[int, int, int, int], direction: tuple[int, int]) -> int:
    """Length of consecutive full-width gray cross-sections outside one bbox side."""
    h, w = g.shape
    r0, c0, r1, c1 = box
    dr, dc = direction
    length = 0
    while True:
        k = length + 1
        if dr:
            r = r0 - k if dr < 0 else r1 + k
            if not (0 <= r < h) or not np.all(g[r, c0 : c1 + 1] == GRAY):
                break
        else:
            c = c0 - k if dc < 0 else c1 + k
            if not (0 <= c < w) or not np.all(g[r0 : r1 + 1, c] == GRAY):
                break
        length += 1
    return length


def _slider_records(g: np.ndarray, ground_colors: tuple[int, int]):
    slider_colors = set(map(int, np.unique(g))) - set(ground_colors) - {GRAY}
    records = []
    for color in slider_colors:
        for coords in _components(g == color):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "sliders must be solid rectangles"
            trails = [(d, _trail_length(g, box, d)) for d in DIRS]
            trails = [(d, n) for d, n in trails if n > 0]
            assert len(trails) == 1, "each slider must have one uniquely aligned gray trail"
            records.append((color, coords, box, trails[0][0], trails[0][1]))
    assert records, "expected at least one slider"
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """Move every rectangular slider back along its aligned gray trail and restore all erased ground cells.

    Input structure:
    A two-color periodic striped or checker ground is partly covered by solid colored rectangular sliders and gray
    cells.  Each slider has exactly one four-neighboring gray band whose aligned cross-section equals the slider's
    full width; other detached or incorrectly sized gray marks are noise.

    Rule:
    Infer the unique periodic ground from its visible cells.  For every slider, measure its uniquely full-width
    adjacent gray band and translate the slider toward that band by the band's length.  Apply all moves
    simultaneously, restoring the old slider cells, every trail cell, and every gray noise cell from the pattern.

    Key insight:
    Full cross-section alignment identifies the causal trail and hence both displacement direction and distance,
    while the ground's spatial period—not gray's shape or color identity—determines every erased cell's restoration.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    ground, ground_colors = _infer_ground(g)
    records = _slider_records(g, ground_colors)

    out = ground.copy()
    claimed = np.zeros(g.shape, dtype=bool)
    all_slider = np.zeros(g.shape, dtype=bool)
    for _color, coords, _box, _direction, _length in records:
        all_slider[coords[:, 0], coords[:, 1]] = True

    h, w = g.shape
    for color, coords, _box, (dr, dc), length in records:
        moved = coords + np.array((dr * length, dc * length), dtype=int)
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < h) & (0 <= moved[:, 1]) & (moved[:, 1] < w))
        destination_other_slider = all_slider[moved[:, 0], moved[:, 1]] & (g[moved[:, 0], moved[:, 1]] != color)
        assert not np.any(destination_other_slider), "a slider destination may not cross another slider"
        assert not np.any(claimed[moved[:, 0], moved[:, 1]]), "restored sliders may not overlap"
        claimed[moved[:, 0], moved[:, 1]] = True
        out[moved[:, 0], moved[:, 1]] = color
    return out


def _ground(shape: tuple[int, int], colors: tuple[int, int], kind: str, x: int, y: int) -> np.ndarray:
    rr, cc = np.indices(shape)
    if kind == "horizontal":
        parity = ((rr + y) // x) % 2
    elif kind == "vertical":
        parity = ((cc + y) // x) % 2
    else:
        parity = ((rr + y) // x + (cc + (y * 2 + 1)) // max(1, 4 - x)) % 2
    return np.where(parity == 0, colors[0], colors[1]).astype(int)


def _rect_cells(r0: int, c0: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + height) for c in range(c0, c0 + width)}


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    return {
        (r + dr, c + dc)
        for r, c in cells
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if 0 <= r + dr < h and 0 <= c + dc < w
    }


def _place_slider(shape, reserved, height, width, length, direction):
    h, w = shape
    dr, dc = direction
    candidates = [(r, c) for r in range(2, h - height - 1) for c in range(2, w - width - 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        current = _rect_cells(r0, c0, height, width)
        destination = {(r + dr * length, c + dc * length) for r, c in current}
        if not all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in destination):
            continue
        if dr:
            rows = range(r0 - length, r0) if dr < 0 else range(r0 + height, r0 + height + length)
            trail = {(r, c) for r in rows for c in range(c0, c0 + width)}
        else:
            cols = range(c0 - length, c0) if dc < 0 else range(c0 + width, c0 + width + length)
            trail = {(r, c) for r in range(r0, r0 + height) for c in cols}
        footprint = current | trail | destination
        if not footprint & reserved and not _expanded(footprint, shape) & reserved:
            return current, trail, destination, _expanded(footprint, shape)
    return None


def _attempt_generate() -> np.ndarray | None:
    shape = (random.randint(15, 23), random.randint(15, 25))
    available = [c for c in PALETTE if c != GRAY]
    ground_colors = tuple(random.sample(available, 2))
    object_colors = random.sample([c for c in available if c not in ground_colors], random.choice((2, 3)))
    kind = random.choice(("horizontal", "vertical", "checker"))
    band = random.randint(1, 3)
    base = _ground(shape, ground_colors, kind, band, random.randrange(6))
    inp = base.copy()
    expected = base.copy()
    reserved: set[tuple[int, int]] = set()
    records = []

    axes = [random.choice(((-1, 0), (1, 0))), random.choice(((0, -1), (0, 1)))]
    axes += [random.choice(DIRS) for _ in range(len(object_colors) - 2)]
    random.shuffle(axes)
    for color, direction in zip(object_colors, axes):
        rh, rw = random.randint(2, 3), random.randint(2, 4)
        length = random.randint(max(rh, rw) + 1, max(rh, rw) + 4)
        placed = _place_slider(shape, reserved, rh, rw, length, direction)
        if placed is None:
            return None
        current, trail, destination, guarded = placed
        reserved |= guarded
        for r, c in current:
            inp[r, c] = color
        for r, c in trail:
            inp[r, c] = GRAY
        for r, c in destination:
            expected[r, c] = color
        records.append((color, current, trail, destination, direction))

    # An attached but too-narrow gray mark is a deliberate false trail.
    color, current, true_trail, _destination, true_dir = random.choice(records)
    rows = [r for r, _ in current]
    cols = [c for _, c in current]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    false_dirs = [d for d in DIRS if d != true_dir and d != (-true_dir[0], -true_dir[1])]
    fr, fc = random.choice(false_dirs)
    if fr:
        noise = (r0 - 1 if fr < 0 else r1 + 1, random.randint(c0, c1))
    else:
        noise = (random.randint(r0, r1), c0 - 1 if fc < 0 else c1 + 1)
    if noise in true_trail or int(inp[noise]) not in ground_colors:
        return None
    inp[noise] = GRAY

    # Detached gray flecks force the global erase/inpaint rule.
    free = []
    slider_union = set().union(*(rec[1] for rec in records))
    gray_now = set(map(tuple, np.argwhere(inp == GRAY)))
    for r in range(1, shape[0] - 1):
        for c in range(1, shape[1] - 1):
            if (r, c) in reserved or (r, c) == noise:
                continue
            if all((r + dr, c + dc) not in slider_union | gray_now for dr, dc in DIRS):
                free.append((r, c))
    if len(free) < 3:
        return None
    detached = random.sample(free, random.randint(2, min(5, len(free))))
    for r, c in detached:
        inp[r, c] = GRAY

    # Semantic checks use generation metadata, not merely solver closure.
    erased = (inp == GRAY) | np.isin(inp, object_colors)
    if set(map(int, base[erased])) != set(ground_colors):
        return None
    try:
        inferred, inferred_colors = _infer_ground(inp)
        found = _slider_records(inp, inferred_colors)
        solved = main(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(inferred, base) or not np.array_equal(solved, expected):
        return None
    if len(found) != len(records) or np.array_equal(inp, expected):
        return None
    if not ({d for *_rest, d in records} & {(-1, 0), (1, 0)} and {d for *_rest, d in records} & {(0, -1), (0, 1)}):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique, non-degenerate trail restoration puzzle")
