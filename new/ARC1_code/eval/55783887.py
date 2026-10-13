import itertools
from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _diagonal_segment(a: tuple[int, int], b: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac
    assert dr != 0 and abs(dr) == abs(dc)
    sr, sc = _sign(dr), _sign(dc)
    return [(ar + i * sr, ac + i * sc) for i in range(abs(dr) + 1)], (sr, sc)


def _full_diagonal(
    shape: tuple[int, int], anchor: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = anchor
    while in_bounds(shape, r - dr, c - dc):
        r -= dr
        c -= dc

    coords = []
    while in_bounds(shape, r, c):
        coords.append((r, c))
        r += dr
        c += dc
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the selected diagonal marker segment, then let any blockers on it emit perpendicular diagonals.

    Input structure
    A mostly background grid with sparse foreground marker pixels. Among same-color
    marker pairs, exactly one diagonal pair is the primary segment: it is the pair
    whose segment contains the most differently colored foreground markers.

    Rule
    Paint the primary diagonal segment in its endpoint color, preserving any
    differently colored markers already on that segment. Each preserved marker on
    the segment paints the full perpendicular diagonal across the grid in its own
    color. Other foreground markers stay unchanged.

    Key insight
    The differently colored markers on the primary diagonal are blockers for the
    primary line and also control pixels for perpendicular line emission.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    locs_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(g != bg):
        locs_by_color[int(g[r, c])].append((int(r), int(c)))

    candidates = []
    for color, locs in locs_by_color.items():
        for a, b in itertools.combinations(locs, 2):
            if abs(b[0] - a[0]) != abs(b[1] - a[1]) or a[0] == b[0]:
                continue
            segment, step = _diagonal_segment(a, b)
            blockers = [p for p in segment if int(g[p]) not in (bg, color)]
            candidates.append((len(blockers), color, segment, step, blockers))

    assert candidates
    max_blockers = max(count for count, _color, _segment, _step, _blockers in candidates)
    best = [candidate for candidate in candidates if candidate[0] == max_blockers]
    assert len(best) == 1

    _count, line_color, segment, step, blockers = best[0]
    out = g.copy()

    for p in segment:
        if int(g[p]) in (bg, line_color):
            out[p] = line_color

    perpendicular = (step[0], -step[1])
    for marker in blockers:
        marker_color = int(g[marker])
        for p in _full_diagonal(g.shape, marker, perpendicular):
            original = int(g[p])
            assert original in (bg, marker_color)
            out[p] = marker_color

    return out

# --------------- generator (o5x) ---------------

# Fixed ARC colors for this task.
LINE_COLOR, BLOCKER_COLOR = 1, 6
BACKGROUND_COLORS = (0, 2, 3, 4, 5, 7, 8, 9)

# Canvas and primary diagonal bounds.
MIN_ROWS, MAX_ROWS = 8, 19
MIN_COLS, MAX_COLS = 9, 16
BORDER_MARGIN = 1
MIN_SEGMENT_LENGTH, MAX_SEGMENT_LENGTH = 6, 10
DIAGONAL_DCS = (-1, 1)

# Sparse marker counts and retry budget.
MAX_BLOCKERS, MAX_DISTRACTORS = 2, 5
GEN_RETRY_LIMIT = 300


def _gen_candidates(grid, bg):
    locs_by_color = defaultdict(list)
    for r, c in np.argwhere(grid != bg):
        locs_by_color[int(grid[r, c])].append((int(r), int(c)))

    candidates = []
    for color, locs in locs_by_color.items():
        for a, b in itertools.combinations(locs, 2):
            if a[0] == b[0] or abs(a[0] - b[0]) != abs(a[1] - b[1]):
                continue
            segment, _step = _diagonal_segment(a, b)
            blockers = [p for p in segment if int(grid[p]) not in (bg, color)]
            candidates.append((len(blockers), color, frozenset(segment)))
    return candidates


def _gen_primary_is_unique(grid, bg, line_col, primary_segment, primary_blocker_count):
    candidates = _gen_candidates(grid, bg)
    if not candidates:
        return False

    primary_key = (line_col, frozenset(primary_segment))
    max_blockers = max(count for count, _color, _segment in candidates)
    if max_blockers != primary_blocker_count:
        return False

    winners = [(color, segment) for count, color, segment in candidates if count == max_blockers]
    return len(winners) == 1 and winners[0] == primary_key


def _sample_primary_segment():
    length = int(np.random.randint(MIN_SEGMENT_LENGTH, MAX_SEGMENT_LENGTH + 1))
    h = int(np.random.randint(max(MIN_ROWS, length + 2 * BORDER_MARGIN), MAX_ROWS + 1))
    w = int(np.random.randint(max(MIN_COLS, length + 2 * BORDER_MARGIN), MAX_COLS + 1))
    dc = int(np.random.choice(DIAGONAL_DCS))

    r0 = int(np.random.randint(BORDER_MARGIN, h - length))
    if dc == 1:
        c0 = int(np.random.randint(BORDER_MARGIN, w - length))
    else:
        c0 = int(np.random.randint(length, w - BORDER_MARGIN))

    return (h, w), dc, [(r0 + i, c0 + i * dc) for i in range(length)]


def _sample_blockers(segment):
    interior = np.arange(1, len(segment) - 1)
    np.random.shuffle(interior)
    count = int(np.random.randint(MAX_BLOCKERS + 1))
    indices = set(int(i) for i in interior[:count])
    return count, [segment[i] for i in sorted(indices)]


def _build_output(shape, bg, segment, blockers, dc):
    """Render the target grid and return its forbidden distractor cells."""
    out = np.full(shape, bg, dtype=int)
    for p in segment:
        out[p] = LINE_COLOR
    for p in blockers:
        out[p] = BLOCKER_COLOR

    segment_set = set(segment)
    forbidden = set(segment)
    perpendicular = (1, -dc)
    for marker in blockers:
        for p in _full_diagonal(out.shape, marker, perpendicular):
            if p != marker and p in segment_set:
                return None, None
            out[p] = BLOCKER_COLOR
            forbidden.add(p)
    return out, forbidden


def _valid_distractor_options(
    inp, out, bg, primary_segment, primary_blocker_count, forbidden, required_color
):
    options = []
    h, w = inp.shape
    colors = (LINE_COLOR, BLOCKER_COLOR)
    for r in range(h):
        for c in range(w):
            p = (r, c)
            if p in forbidden or int(inp[p]) != bg or int(out[p]) != bg:
                continue
            for color in colors:
                if required_color is not None and color != required_color:
                    continue
                trial = inp.copy()
                trial[p] = color
                if _gen_primary_is_unique(
                    trial, bg, LINE_COLOR, primary_segment, primary_blocker_count
                ):
                    options.append((p, color))
    return options


def _gen_add_distractors(
    inp,
    out,
    bg,
    primary_segment,
    primary_blocker_count,
    forbidden,
    count,
    force_first_blocker,
):
    for i in range(count):
        required_color = BLOCKER_COLOR if force_first_blocker and i == 0 else None
        options = _valid_distractor_options(
            inp,
            out,
            bg,
            primary_segment,
            primary_blocker_count,
            forbidden,
            required_color,
        )
        if not options:
            return False

        p, color = options[int(np.random.randint(len(options)))]
        inp[p] = color
        out[p] = color
        forbidden.add(p)

    return True


def _gen_attempt():
    bg = int(np.random.choice(BACKGROUND_COLORS))
    shape, dc, segment = _sample_primary_segment()
    blocker_count, blockers = _sample_blockers(segment)

    out, forbidden = _build_output(shape, bg, segment, blockers, dc)
    if out is None:
        return None

    inp = np.full(shape, bg, dtype=int)
    inp[segment[0]] = LINE_COLOR
    inp[segment[-1]] = LINE_COLOR
    for p in blockers:
        inp[p] = BLOCKER_COLOR
    if not _gen_primary_is_unique(inp, bg, LINE_COLOR, segment, blocker_count):
        return None

    distractor_count = int(np.random.randint(MAX_DISTRACTORS + 1))
    force_first_blocker = blocker_count == 0
    if force_first_blocker and distractor_count == 0:
        distractor_count = 1

    if not _gen_add_distractors(
        inp,
        out,
        bg,
        segment,
        blocker_count,
        forbidden,
        distractor_count,
        force_first_blocker,
    ):
        return None

    if not _gen_primary_is_unique(inp, bg, LINE_COLOR, segment, blocker_count):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRY_LIMIT} attempts")
