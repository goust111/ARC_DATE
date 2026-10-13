import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

LEFT_SMALL = frozenset({(0, 1), (1, 0), (1, 1)})
RIGHT_SMALL = frozenset({(0, 0), (1, 0), (1, 1)})
LEFT_LARGE = frozenset({(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)})
RIGHT_LARGE = frozenset({(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)})

TEMPLATES = {
    ("L", "small"): LEFT_SMALL,
    ("R", "small"): RIGHT_SMALL,
    ("L", "large"): LEFT_LARGE,
    ("R", "large"): RIGHT_LARGE,
}
SHAPE_TO_ORIENTATION = {
    LEFT_SMALL: "L",
    RIGHT_SMALL: "R",
    LEFT_LARGE: "L",
    RIGHT_LARGE: "R",
}

EDGE_TO_DIRECTION = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

GEN_RETRY_LIMIT = 500
GRID_ROWS = (12, 16)
GRID_COLS = (12, 16)
OBJECT_COUNTS = (6, 9)
_RNG = np.random.default_rng()


def _normalize(coords):
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _bbox(coords):
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_objects(grid):
    mask = np.asarray(grid) == RED
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    objects = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            shape = _normalize(coords)
            assert shape in SHAPE_TO_ORIENTATION, f"unexpected sprite shape: {sorted(shape)}"
            objects.append(
                {
                    "coords": tuple(sorted(coords)),
                    "orientation": SHAPE_TO_ORIENTATION[shape],
                    "bbox": _bbox(coords),
                }
            )

    assert objects, "expected at least one red sprite"
    return objects


def _marker_direction(grid):
    g = np.asarray(grid)
    h, w = g.shape
    markers = np.argwhere(g == GREEN)
    assert markers.shape == (1, 2), "expected exactly one green edge marker"
    r, c = map(int, markers[0])
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert sum([on_top, on_bottom, on_left, on_right]) == 1, "marker must lie on exactly one edge"
    assert not (on_top or on_bottom) or 0 < c < w - 1, "marker must not be a corner"
    assert not (on_left or on_right) or 0 < r < h - 1, "marker must not be a corner"
    if on_left:
        return "left"
    if on_right:
        return "right"
    if on_top:
        return "up"
    return "down"


def _sort_key(obj, direction):
    r0, c0, r1, c1 = obj["bbox"]
    if direction == "left":
        return (c0, r0, c1, r1)
    if direction == "right":
        return (-c1, r0, -c0, r1)
    if direction == "up":
        return (r0, c0, r1, c1)
    return (-r1, c0, -r0, c1)


def _compress_objects(shape, objects, direction):
    h, w = map(int, shape)
    dr, dc = EDGE_TO_DIRECTION[direction]
    settled = set()
    compressed = []

    for obj in sorted(objects, key=lambda item: _sort_key(item, direction)):
        coords = [tuple(map(int, cell)) for cell in obj["coords"]]
        while True:
            moved = [(r + dr, c + dc) for r, c in coords]
            if any(not (0 <= r < h and 0 <= c < w) for r, c in moved):
                break
            if any((r, c) in settled for r, c in moved):
                break
            coords = moved

        settled.update(coords)
        compressed.append(
            {
                "coords": tuple(sorted(coords)),
                "orientation": obj["orientation"],
            }
        )

    return compressed


def _line_counts(shape, objects, direction):
    compressed = _compress_objects(shape, objects, direction)
    horizontal = direction in {"left", "right"}
    line_sets = []
    for obj in compressed:
        if horizontal:
            lines = {c for _r, c in obj["coords"]}
        else:
            lines = {r for r, _c in obj["coords"]}
        line_sets.append((obj["orientation"], lines))

    all_lines = sorted({line for _orientation, lines in line_sets for line in lines})
    assert all_lines, "compressed scene must occupy at least one row or column"
    start = all_lines[0]
    stop = all_lines[-1]
    span = list(range(start, stop + 1))
    left_counts = [0] * len(span)
    right_counts = [0] * len(span)

    for orientation, lines in line_sets:
        for line in lines:
            idx = line - start
            if orientation == "L":
                left_counts[idx] += 1
            else:
                right_counts[idx] += 1

    return span, left_counts, right_counts


def _render_panel(shape, objects, direction):
    _span, left_counts, right_counts = _line_counts(shape, objects, direction)
    panel_h = max(max(left_counts), max(right_counts))
    panel_w = 2 * len(left_counts)
    out = np.zeros((panel_h, panel_w), dtype=int)

    for i, (left_count, right_count) in enumerate(zip(left_counts, right_counts)):
        if left_count:
            out[:left_count, 2 * i] = BLUE
        if right_count:
            out[:right_count, 2 * i + 1] = YELLOW

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress the red sprites toward the green-marked edge and tally shared compressed lines by orientation.

    Input structure:
    - A black grid contains several disconnected red sprites, each matching one of two left/right-facing template families.
    - Exactly one green marker sits on a non-corner border cell and chooses the compression edge.
    - Every sprite is treated as a rigid body; background is otherwise empty.

    Rule:
    - Read the green marker: left/right markers compress horizontally, top/bottom markers compress vertically.
    - Move the red sprites one rigid step at a time toward that edge, processing sprites nearest to the target edge first and stopping each sprite at the wall or just before a settled sprite.
    - After compression, count how many left-facing and right-facing sprites touch each occupied compressed column (for horizontal compression) or row (for vertical compression), counting a sprite once per line it spans.
    - Render a black output panel made of adjacent 2-column micro-panels, one per compressed line from top-to-bottom or left-to-right order: the left slot column is blue for the left-facing count, the right slot column is yellow for the right-facing count, and each column is filled from the top downward.

    Key insight:
    - The green edge marker does not merely pick an axis label; it determines the rigid compaction order, which changes which sprites end up sharing the same compressed rows or columns before the orientation counts are tallied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREEN}
    direction = _marker_direction(g)
    objects = _find_objects(g)
    return _render_panel(g.shape, objects, direction)


def _randint(lo, hi):
    return int(_RNG.integers(int(lo), int(hi)))


def _choice(seq):
    return seq[_randint(0, len(seq))]


def _shuffled(seq):
    idx = _RNG.permutation(len(seq))
    return [seq[int(i)] for i in idx]


def _template_cells(orientation, size_key):
    return tuple(sorted(TEMPLATES[(orientation, size_key)]))


def _placement_candidates(h, w, rel_cells):
    rows = [r for r, _c in rel_cells]
    cols = [c for _r, c in rel_cells]
    ph = max(rows) + 1
    pw = max(cols) + 1
    r_lo = 1
    c_lo = 1
    r_hi = h - ph
    c_hi = w - pw
    if r_hi < r_lo or c_hi < c_lo:
        return []
    candidates = [(r0, c0) for r0 in range(r_lo, r_hi + 1) for c0 in range(c_lo, c_hi + 1)]
    return _shuffled(candidates)


def _can_place(occupied, abs_cells):
    h, w = occupied.shape
    for r, c in abs_cells:
        if occupied[r, c]:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _line_profile_ok(left_counts, right_counts):
    if max(max(left_counts), max(right_counts)) < 2:
        return False
    if sum(left_counts) == 0 or sum(right_counts) == 0:
        return False
    nonzero_pairs = [(l, r) for l, r in zip(left_counts, right_counts) if l or r]
    if len(set(nonzero_pairs)) < 2:
        return False
    if not any(l > 0 and r > 0 for l, r in nonzero_pairs):
        return False
    if not any((l > 0) != (r > 0) for l, r in nonzero_pairs):
        return False
    return True


def _panel_signature(panel):
    return (tuple(panel.shape), tuple(map(int, panel.flat)))


def _edge_candidates(grid, edge):
    h, w = grid.shape
    if edge == "left":
        return [(r, 0) for r in range(1, h - 1)]
    if edge == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    if edge == "up":
        return [(0, c) for c in range(1, w - 1)]
    return [(h - 1, c) for c in range(1, w - 1)]


def _marker_ok(occupied, cell, edge):
    r, c = cell
    h, w = occupied.shape
    if occupied[r, c]:
        return False
    checks = {(r, c)}
    if edge == "left":
        checks.add((r, 1))
    elif edge == "right":
        checks.add((r, w - 2))
    elif edge == "up":
        checks.add((1, c))
    else:
        checks.add((h - 2, c))
    for rr, cc in checks:
        if occupied[rr, cc]:
            return False
    return True


def _build_grid(h, w, objects, marker):
    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        for r, c in obj["coords"]:
            grid[r, c] = RED
    mr, mc = marker
    grid[mr, mc] = GREEN
    return grid


def _sample_specs(object_count):
    left_count = _randint(2, object_count - 1)
    while object_count - left_count < 2:
        left_count = _randint(2, object_count - 1)
    orientations = ["L"] * left_count + ["R"] * (object_count - left_count)
    orientations = _shuffled(orientations)

    size_keys = ["small", "large"]
    specs = []
    for i, orientation in enumerate(orientations):
        if i == 0:
            size_key = "small"
        elif i == 1:
            size_key = "large"
        else:
            size_key = _choice(size_keys)
        specs.append((orientation, size_key))
    return _shuffled(specs)


def _place_objects(h, w, specs):
    occupied = np.zeros((h, w), dtype=bool)
    objects = []

    order = sorted(specs, key=lambda item: (-len(TEMPLATES[item]), item[0], item[1]))
    for orientation, size_key in order:
        rel_cells = _template_cells(orientation, size_key)
        placed = False
        for r0, c0 in _placement_candidates(h, w, rel_cells):
            abs_cells = tuple(sorted((r0 + r, c0 + c) for r, c in rel_cells))
            if not _can_place(occupied, abs_cells):
                continue
            for r, c in abs_cells:
                occupied[r, c] = True
            objects.append(
                {
                    "coords": abs_cells,
                    "orientation": orientation,
                    "bbox": _bbox(abs_cells),
                }
            )
            placed = True
            break
        if not placed:
            return None

    return objects


def _gen_attempt():
    h = _randint(*GRID_ROWS)
    w = _randint(*GRID_COLS)
    direction = _choice(["left", "right", "up", "down"])
    object_count = _randint(*OBJECT_COUNTS)
    specs = _sample_specs(object_count)
    objects = _place_objects(h, w, specs)
    if objects is None:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    for obj in objects:
        for r, c in obj["coords"]:
            occupied[r, c] = True

    marker_options = [cell for cell in _edge_candidates(occupied, direction) if _marker_ok(occupied, cell, direction)]
    if not marker_options:
        return None
    marker = _choice(marker_options)

    span, left_counts, right_counts = _line_counts((h, w), objects, direction)
    if len(span) < 3 or not _line_profile_ok(left_counts, right_counts):
        return None

    chosen_panel = _render_panel((h, w), objects, direction)
    if max(chosen_panel.shape) > 30:
        return None

    signatures = set()
    for alt_direction in ("left", "right", "up", "down"):
        alt_panel = _render_panel((h, w), objects, alt_direction)
        signatures.add(_panel_signature(alt_panel))
    if len(signatures) < 2:
        return None

    orthogonal = {"left": ("up", "down"), "right": ("up", "down"), "up": ("left", "right"), "down": ("left", "right")}
    chosen_sig = _panel_signature(chosen_panel)
    if all(_panel_signature(_render_panel((h, w), objects, alt)) == chosen_sig for alt in orthogonal[direction]):
        return None

    grid = _build_grid(h, w, objects, marker)
    assert np.array_equal(main(grid), chosen_panel)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 5")
