from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, colors_present, find_components, get_bg, paste


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _plus_centers(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 1, "expected one foreground plus color"

    components = find_components(g, bg=bg, colors=[colors[0]], connectivity=4)
    assert len(components) >= 2, "expected multiple plus objects"

    centers: list[tuple[int, int]] = []
    for component in components:
        assert component["area"] == 5, "each object must be a five-pixel plus"
        box = component["bbox"]
        assert bbox_h(box) == 3 and bbox_w(box) == 3, "plus bbox must be 3x3"

        r0, c0, _r1, _c1 = box
        center = (int(r0) + 1, int(c0) + 1)
        cr, cc = center
        expected = {
            center,
            (cr - 1, cc),
            (cr + 1, cc),
            (cr, cc - 1),
            (cr, cc + 1),
        }
        assert set(map(tuple, component["coords"].tolist())) == expected, "object must have plus shape"
        centers.append(center)

    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red diagonal connectors between green plus objects whose centers lie on the same clear 45-degree diagonal.

    Input structure
    - The canvas has a single background color and multiple same-colored 5-pixel plus objects.
    - Each plus is a 4-connected object with a 3x3 bbox and pixels at the center plus the four cardinal neighbors.
    - Some plus centers share an unobstructed diagonal line of background cells.

    Rule
    - Keep every input pixel unchanged.
    - For each pair of plus centers with equal row and column separation, scan the straight diagonal segment between them.
    - If every interior cell on that diagonal segment is background, recolor those interior cells red.

    Key insight
    - The added pixels are not extensions of plus arms; they are the empty center-to-center diagonal gaps between aligned plus objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    centers = _plus_centers(g, bg)

    out = g.copy()
    connector_count = 0
    for i, (r1, c1) in enumerate(centers):
        for r2, c2 in centers[i + 1 :]:
            dr = int(r2 - r1)
            dc = int(c2 - c1)
            if abs(dr) != abs(dc):
                continue

            sr = _sign(dr)
            sc = _sign(dc)
            gap = [(r1 + k * sr, c1 + k * sc) for k in range(1, abs(dr))]
            if all(int(g[r, c]) == bg for r, c in gap):
                for r, c in gap:
                    out[r, c] = Color.RED
                connector_count += 1

    assert connector_count > 0, "expected at least one clear diagonal connector"
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GEN_H_RANGE = (12, 18)
GEN_W_RANGE = (10, 17)
GEN_CENTER_MARGIN = 1
GEN_BACKGROUND_COLOR, GEN_PLUS_COLOR, GEN_CONNECTOR_COLOR = Color.BLACK, Color.GREEN, Color.RED

# Counts and topology.
GEN_PLUS_COUNT_RANGE = (3, 5)
GEN_PAIR_COUNT_RANGE = (1, 3)
GEN_HUB_PROB_FOR_FOUR_PLUSES = 0.5

# Diagonal connector geometry and placement retries.
GEN_PAIR_DISTANCE_RANGE = (3, 7)
GEN_GAP_LEN_RANGE = (2, 5)
GEN_PAIR_ATTEMPTS = 160
GEN_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

# Object separation and outer retry budget.
GEN_CENTER_MIN_CHEBYSHEV = 3
GEN_PLUS_TOUCH_RADIUS = 1
GEN_GENERATE_ATTEMPTS = 300

GEN_PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
GEN_PLUS_SPRITE = np.array(
    [[0, GEN_PLUS_COLOR, 0], [GEN_PLUS_COLOR, GEN_PLUS_COLOR, GEN_PLUS_COLOR], [0, GEN_PLUS_COLOR, 0]],
    dtype=int,
)


def _gen_plus_cells(center):
    r, c = center
    return tuple((r + dr, c + dc) for dr, dc in GEN_PLUS_OFFSETS)


def _gen_all_plus_cells(centers):
    cells = set()
    for center in centers:
        cells.update(_gen_plus_cells(center))
    return cells


def _gen_center_ok(center, h, w):
    r, c = center
    return GEN_CENTER_MARGIN <= r < h - GEN_CENTER_MARGIN and GEN_CENTER_MARGIN <= c < w - GEN_CENTER_MARGIN


def _gen_centers_far(centers):
    for i, (r1, c1) in enumerate(centers):
        for r2, c2 in centers[i + 1 :]:
            if max(abs(r2 - r1), abs(c2 - c1)) < GEN_CENTER_MIN_CHEBYSHEV:
                return False
    return True


def _gen_pluses_separate(centers):
    groups = [set(_gen_plus_cells(center)) for center in centers]
    for i, first in enumerate(groups):
        for second in groups[i + 1 :]:
            for r1, c1 in first:
                for r2, c2 in second:
                    if max(abs(r2 - r1), abs(c2 - c1)) <= GEN_PLUS_TOUCH_RADIUS:
                        return False
    return True


def _gen_gap_cells(a, b):
    r1, c1 = a
    r2, c2 = b
    dr, dc = r2 - r1, c2 - c1
    if abs(dr) != abs(dc):
        return ()
    sr = (dr > 0) - (dr < 0)
    sc = (dc > 0) - (dc < 0)
    return tuple((r1 + k * sr, c1 + k * sc) for k in range(1, abs(dr)))


def _gen_edge(a, b):
    return tuple(sorted((a, b)))


def _gen_no_extra_diagonal(new_centers, old_centers, allowed_base=None):
    for r1, c1 in new_centers:
        for old in old_centers:
            if allowed_base is not None and old == allowed_base:
                continue
            r2, c2 = old
            if abs(r2 - r1) == abs(c2 - c1):
                return False
    return True


def _gen_pair_ok(h, w, a, b, centers, red_cells):
    if not all(_gen_center_ok(center, h, w) for center in centers):
        return False
    if not _gen_centers_far(centers) or not _gen_pluses_separate(centers):
        return False
    gap = _gen_gap_cells(a, b)
    if not GEN_GAP_LEN_RANGE[0] <= len(gap) <= GEN_GAP_LEN_RANGE[1]:
        return False
    plus_cells = _gen_all_plus_cells(centers)
    return not any(cell in plus_cells or cell in red_cells for cell in gap)


def _gen_make_pair(h, w, centers, red_cells, base=None):
    for _ in range(GEN_PAIR_ATTEMPTS):
        dist = int(np.random.randint(*GEN_PAIR_DISTANCE_RANGE))
        sr, sc = GEN_DIRECTIONS[int(np.random.randint(0, len(GEN_DIRECTIONS)))]
        if base is None:
            r_low = max(GEN_CENTER_MARGIN, GEN_CENTER_MARGIN - sr * dist)
            r_high = min(h - GEN_CENTER_MARGIN - 1, h - GEN_CENTER_MARGIN - 1 - sr * dist)
            c_low = max(GEN_CENTER_MARGIN, GEN_CENTER_MARGIN - sc * dist)
            c_high = min(w - GEN_CENTER_MARGIN - 1, w - GEN_CENTER_MARGIN - 1 - sc * dist)
            if r_low > r_high or c_low > c_high:
                continue
            a = (
                int(np.random.randint(r_low, r_high + 1)),
                int(np.random.randint(c_low, c_high + 1)),
            )
            b = (a[0] + sr * dist, a[1] + sc * dist)
            new_centers = [a, b]
        else:
            a = base
            b = (a[0] + sr * dist, a[1] + sc * dist)
            if not _gen_center_ok(b, h, w):
                continue
            new_centers = [b]

        if any(center in centers for center in new_centers):
            continue
        if not _gen_no_extra_diagonal(new_centers, centers, allowed_base=base):
            continue
        trial_centers = centers + new_centers
        if _gen_pair_ok(h, w, a, b, trial_centers, red_cells):
            return a, b
    return None


def _gen_add_distractor(h, w, centers, red_cells):
    candidates = [(r, c) for r in range(GEN_CENTER_MARGIN, h - GEN_CENTER_MARGIN) for c in range(GEN_CENTER_MARGIN, w - GEN_CENTER_MARGIN)]
    order = np.random.permutation(len(candidates))
    for idx in order:
        center = candidates[int(idx)]
        if not _gen_no_extra_diagonal([center], centers):
            continue
        trial_centers = centers + [center]
        if not _gen_centers_far(trial_centers) or not _gen_pluses_separate(trial_centers):
            continue
        if any(cell in red_cells for cell in _gen_plus_cells(center)):
            continue
        return center
    return None


def _gen_clear_edges(centers):
    plus_cells = _gen_all_plus_cells(centers)
    edges = set()
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            gap = _gen_gap_cells(a, b)
            if gap and all(cell not in plus_cells for cell in gap):
                edges.add(_gen_edge(a, b))
    return edges


def _gen_build_output(h, w, centers, pairs):
    out = np.full((h, w), GEN_BACKGROUND_COLOR, dtype=int)
    for r, c in centers:
        paste(out, GEN_PLUS_SPRITE, r - GEN_CENTER_MARGIN, c - GEN_CENTER_MARGIN, transparent=GEN_BACKGROUND_COLOR)
    for a, b in pairs:
        for r, c in _gen_gap_cells(a, b):
            out[r, c] = GEN_CONNECTOR_COLOR
    return out


def _gen_sample_plan():
    h = int(np.random.randint(*GEN_H_RANGE))
    w = int(np.random.randint(*GEN_W_RANGE))
    num_pluses = int(np.random.randint(*GEN_PLUS_COUNT_RANGE))
    num_pairs = int(np.random.randint(*GEN_PAIR_COUNT_RANGE))
    if num_pairs == 1:
        topology = "single"
    elif num_pluses == 3:
        topology = "hub"
    else:
        topology = "hub" if np.random.random() < GEN_HUB_PROB_FOR_FOUR_PLUSES else "independent"
    return h, w, num_pluses, topology


def _gen_record_pair(pair, centers, pairs, red_cells, new_centers):
    pairs.append(pair)
    centers.extend(new_centers)
    red_cells.update(_gen_gap_cells(*pair))


def _gen_place_connected_pairs(h, w, topology):
    centers = []
    pairs = []
    red_cells = set()

    first = _gen_make_pair(h, w, centers, red_cells)
    if first is None:
        return None
    _gen_record_pair(first, centers, pairs, red_cells, first)

    if topology == "hub":
        hub = pairs[0][int(np.random.randint(0, 2))]
        second = _gen_make_pair(h, w, centers, red_cells, base=hub)
        if second is None:
            return None
        _gen_record_pair(second, centers, pairs, red_cells, [second[1]])
    elif topology == "independent":
        second = _gen_make_pair(h, w, centers, red_cells)
        if second is None:
            return None
        _gen_record_pair(second, centers, pairs, red_cells, second)

    return centers, pairs, red_cells


def _gen_fill_distractors(h, w, num_pluses, centers, red_cells):
    while len(centers) < num_pluses:
        center = _gen_add_distractor(h, w, centers, red_cells)
        if center is None:
            return False
        centers.append(center)
    return True


def _gen_attempt() -> np.ndarray | None:
    h, w, num_pluses, topology = _gen_sample_plan()
    layout = _gen_place_connected_pairs(h, w, topology)
    if layout is None:
        return None
    centers, pairs, red_cells = layout
    if not _gen_fill_distractors(h, w, num_pluses, centers, red_cells):
        return None
    expected_edges = {_gen_edge(a, b) for a, b in pairs}
    if len(centers) != num_pluses or _gen_clear_edges(centers) != expected_edges:
        return None
    out = _gen_build_output(h, w, centers, pairs)
    inp = out.copy()
    inp[inp == GEN_CONNECTOR_COLOR] = GEN_BACKGROUND_COLOR
    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
