import random
from itertools import combinations

import numpy as np


BLACK = 0
_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_ORTHOGONAL = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}
_PAIR_TO_POINTER = {
    frozenset(("up", "left")): ((0, 1), (1, 0), (1, 1)),
    frozenset(("up", "right")): ((0, 0), (1, 0), (1, 1)),
    frozenset(("down", "left")): ((0, 0), (0, 1), (1, 1)),
    frozenset(("down", "right")): ((0, 0), (0, 1), (1, 0)),
}
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_ATTEMPTS = 300


def _normalize_shape(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _rotate_shape(coords):
    h, _ = _shape_dims(coords)
    return _normalize_shape([(c, h - 1 - r) for r, c in coords])


def _mirror_shape(coords):
    _, w = _shape_dims(coords)
    return _normalize_shape([(r, w - 1 - c) for r, c in coords])


def _build_pattern_library():
    bases = [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ]

    variants = set()
    for base in bases:
        shape = _normalize_shape(base)
        for _ in range(4):
            variants.add(shape)
            variants.add(_mirror_shape(shape))
            shape = _rotate_shape(shape)

    return sorted(variants, key=lambda shape: (len(shape), _shape_dims(shape), shape))


_PATTERN_LIBRARY = _build_pattern_library()


def _neighbor(cell, direction):
    dr, dc = _DIRS[direction]
    return cell[0] + dr, cell[1] + dc


def _pointer_candidate_dirs(coords):
    if len(coords) != 3:
        return ()

    dirs = set()
    pts = [tuple(map(int, coord)) for coord in coords]
    for i, j in combinations(range(3), 2):
        a = pts[i]
        b = pts[j]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            continue

        k = 3 - i - j
        third = pts[k]
        if a[0] == b[0]:
            for elbow in (a, b):
                if third[1] == elbow[1] and abs(third[0] - elbow[0]) == 1:
                    dirs.add("up" if third[0] < elbow[0] else "down")
        elif a[1] == b[1]:
            for elbow in (a, b):
                if third[0] == elbow[0] and abs(third[1] - elbow[1]) == 1:
                    dirs.add("left" if third[1] < elbow[1] else "right")

    return tuple(sorted(dirs))


def _find_partition_layout(grid):
    h, w = grid.shape
    candidates = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(row_lines) != len(col_lines) or len(row_lines) not in (4, 5):
            continue
        if row_lines[0] != 0 or row_lines[-1] != h - 1 or col_lines[0] != 0 or col_lines[-1] != w - 1:
            continue

        row_steps = np.diff(row_lines)
        col_steps = np.diff(col_lines)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue

        n_parts = len(row_lines) - 1
        cell_h = int(row_steps[0]) - 1
        cell_w = int(col_steps[0]) - 1
        if n_parts not in (3, 4):
            continue
        if not (5 <= cell_h <= 7 and 5 <= cell_w <= 7):
            continue
        candidates.append((color, row_lines, col_lines, cell_h, cell_w))

    assert candidates, "expected a uniform grey partition grid"
    return candidates[0]


def _slice_for_cell(row_lines, col_lines, r, c):
    return (
        slice(row_lines[r] + 1, row_lines[r + 1]),
        slice(col_lines[c] + 1, col_lines[c + 1]),
    )


def _analyze_cells(grid, grey, row_lines, col_lines):
    n_rows = len(row_lines) - 1
    n_cols = len(col_lines) - 1
    info = {}
    for rr in range(n_rows):
        for cc in range(n_cols):
            rs, cs = _slice_for_cell(row_lines, col_lines, rr, cc)
            patch = grid[rs, cs]
            coords = [tuple(map(int, p)) for p in np.argwhere((patch != BLACK) & (patch != grey))]
            if not coords:
                info[(rr, cc)] = {"kind": "empty", "coords": (), "candidate_dirs": (), "color": BLACK}
                continue

            colors = {int(patch[r, c]) for r, c in coords}
            assert len(colors) == 1, "expected monochrome objects inside each partition"
            candidate_dirs = _pointer_candidate_dirs(coords)
            kind = "ambiguous" if candidate_dirs else "pattern"
            info[(rr, cc)] = {
                "kind": kind,
                "coords": tuple(sorted(coords)),
                "candidate_dirs": candidate_dirs,
                "color": next(iter(colors)),
            }
    return info


def _resolve_roles(info, n_rows, n_cols):
    ambiguous = [cell for cell, cell_info in sorted(info.items()) if cell_info["kind"] == "ambiguous"]
    definite_patterns = {cell for cell, cell_info in info.items() if cell_info["kind"] == "pattern"}

    solutions = []
    best_pointer_count = -1
    for mask in range(1 << len(ambiguous)):
        pattern_cells = set(definite_patterns)
        pointer_cells = set()
        for idx, cell in enumerate(ambiguous):
            if mask & (1 << idx):
                pointer_cells.add(cell)
            else:
                pattern_cells.add(cell)

        pointer_dirs = {}
        ok = True
        for cell in pointer_cells:
            valid_dirs = []
            for direction in info[cell]["candidate_dirs"]:
                nbr = _neighbor(cell, direction)
                if 0 <= nbr[0] < n_rows and 0 <= nbr[1] < n_cols and nbr in pattern_cells:
                    valid_dirs.append(direction)
            if len(valid_dirs) != 1:
                ok = False
                break
            pointer_dirs[cell] = valid_dirs[0]

        if ok:
            roles = {}
            for cell, cell_info in info.items():
                if cell_info["kind"] == "empty":
                    roles[cell] = "empty"
                elif cell in pointer_cells:
                    roles[cell] = "pointer"
                else:
                    roles[cell] = "pattern"
            pointer_count = len(pointer_cells)
            if pointer_count > best_pointer_count:
                best_pointer_count = pointer_count
                solutions = [(roles, pointer_dirs)]
            elif pointer_count == best_pointer_count:
                solutions.append((roles, pointer_dirs))

    assert solutions, "failed to classify partitions into patterns and pointers"
    return solutions[0], len(solutions) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each pointed-at neighboring pattern into the pointer partition and recolor it with the pointer color.

    Input structure:
    - Black background is split by full grey grid lines into a uniform 3x3 or 4x4 array of equal rectangular partitions.
    - Some partitions contain a small monochrome pattern of 2-6 non-black, non-grey pixels.
    - Other partitions contain a monochrome 3-pixel right-angle pointer whose L-shape encodes one of two possible directions until the neighboring pattern partition disambiguates it.
    - At least one partition is empty, and grey divider lines stay intact.

    Rule:
    - Detect the grey partition lines and inspect each partition independently.
    - Treat non-empty non-pointer shapes as source patterns; for each 3-pixel right-angle pointer, choose the unique candidate direction whose adjacent partition contains a pattern.
    - Remove the pointer, then copy the pointed partition's pattern coordinates into the pointer partition while recoloring the copied pixels to the pointer's color.
    - Leave original pattern partitions, empty partitions, and grey grid lines unchanged.

    Key insight:
    - The pointer shape contributes direction while its color contributes the repaint color, so solving requires separating where to copy from and how to recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey, row_lines, col_lines, _, _ = _find_partition_layout(g)
    info = _analyze_cells(g, grey, row_lines, col_lines)
    (roles, pointer_dirs), _ = _resolve_roles(info, len(row_lines) - 1, len(col_lines) - 1)

    out = g.copy()
    for cell, role in roles.items():
        if role != "pointer":
            continue

        rs, cs = _slice_for_cell(row_lines, col_lines, cell[0], cell[1])
        out[rs, cs] = BLACK

        source = _neighbor(cell, pointer_dirs[cell])
        pattern_coords = info[source]["coords"]
        pointer_color = info[cell]["color"]
        for r, c in pattern_coords:
            out[rs.start + r, cs.start + c] = pointer_color

    return out


def _choose_cell_geometry():
    n_parts = random.choice((3, 4))
    max_dim = 7 if n_parts == 3 else 6
    cell_h = random.randint(5, max_dim)
    cell_w = random.randint(5, max_dim)
    return n_parts, cell_h, cell_w


def _collect_pointer_options(n_parts, pattern_cells):
    pattern_cells = set(pattern_cells)
    options = {}
    all_cells = [(r, c) for r in range(n_parts) for c in range(n_parts)]
    for cell in all_cells:
        if cell in pattern_cells:
            continue
        cell_options = []
        for direction in _DIRS:
            src = _neighbor(cell, direction)
            if src not in pattern_cells:
                continue
            for decoy in _ORTHOGONAL[direction]:
                side = _neighbor(cell, decoy)
                if side not in pattern_cells:
                    cell_options.append((direction, decoy))
        if cell_options:
            random.shuffle(cell_options)
            options[cell] = cell_options
    return options


def _sample_layout(n_parts):
    all_cells = [(r, c) for r in range(n_parts) for c in range(n_parts)]
    max_patterns = min(6, len(_PATTERN_LIBRARY), len(_COLOR_POOL), len(all_cells) - 2)
    if max_patterns < 2:
        return None

    pattern_count = random.randint(2, max_patterns)
    pattern_cells = set(random.sample(all_cells, pattern_count))
    pointer_options = _collect_pointer_options(n_parts, pattern_cells)
    max_pointers = min(5, len(pointer_options), len(all_cells) - pattern_count - 1)
    if max_pointers < 1:
        return None

    pointer_count = random.randint(1, max_pointers)
    pointer_cells = random.sample(list(pointer_options), pointer_count)
    pointer_specs = {cell: random.choice(pointer_options[cell]) for cell in pointer_cells}
    return pattern_cells, pointer_specs


def _place_pattern(shape, cell_h, cell_w):
    shape_h, shape_w = _shape_dims(shape)
    assert shape_h <= cell_h - 2 and shape_w <= cell_w - 2
    top = random.randint(1, cell_h - shape_h - 1)
    left = random.randint(1, cell_w - shape_w - 1)
    return tuple(sorted((top + r, left + c) for r, c in shape))


def _sample_patterns(pattern_cells, cell_h, cell_w):
    shapes = random.sample(_PATTERN_LIBRARY, len(pattern_cells))
    colors = random.sample(_COLOR_POOL, len(pattern_cells))
    pattern_data = {}
    for cell, shape, color in zip(sorted(pattern_cells), shapes, colors):
        pattern_data[cell] = {
            "coords": _place_pattern(shape, cell_h, cell_w),
            "color": color,
        }
    return pattern_data


def _pointer_offsets(direction, decoy):
    return _PAIR_TO_POINTER[frozenset((direction, decoy))]


def _place_pointer(direction, decoy, cell_h, cell_w):
    offsets = _pointer_offsets(direction, decoy)
    top = random.randint(1, cell_h - 3)
    left = random.randint(1, cell_w - 3)
    return tuple(sorted((top + r, left + c) for r, c in offsets))


def _render_grid(n_parts, cell_h, cell_w, pattern_data, pointer_data, grey=5):
    h = n_parts * cell_h + (n_parts + 1)
    w = n_parts * cell_w + (n_parts + 1)
    grid = np.full((h, w), BLACK, dtype=int)

    row_lines = [i * (cell_h + 1) for i in range(n_parts + 1)]
    col_lines = [i * (cell_w + 1) for i in range(n_parts + 1)]
    grid[row_lines, :] = grey
    grid[:, col_lines] = grey

    for cell, data in pattern_data.items():
        rs, cs = _slice_for_cell(row_lines, col_lines, cell[0], cell[1])
        for r, c in data["coords"]:
            grid[rs.start + r, cs.start + c] = data["color"]

    for cell, data in pointer_data.items():
        rs, cs = _slice_for_cell(row_lines, col_lines, cell[0], cell[1])
        for r, c in data["coords"]:
            grid[rs.start + r, cs.start + c] = data["color"]

    return grid


def _expected_output(grid, pattern_data, pointer_data):
    out = grid.copy()
    grey, row_lines, col_lines, _, _ = _find_partition_layout(out)
    for cell, data in pointer_data.items():
        rs, cs = _slice_for_cell(row_lines, col_lines, cell[0], cell[1])
        out[rs, cs] = BLACK
        source_coords = pattern_data[data["source"]]["coords"]
        for r, c in source_coords:
            out[rs.start + r, cs.start + c] = data["color"]
    out[row_lines, :] = grey
    out[:, col_lines] = grey
    return out


def _gen_attempt():
    n_parts, cell_h, cell_w = _choose_cell_geometry()
    sampled = _sample_layout(n_parts)
    if sampled is None:
        return None
    pattern_cells, pointer_specs = sampled
    pattern_data = _sample_patterns(pattern_cells, cell_h, cell_w)

    pointer_data = {}
    for cell, (direction, decoy) in pointer_specs.items():
        source = _neighbor(cell, direction)
        source_color = pattern_data[source]["color"]
        color_choices = [color for color in _COLOR_POOL if color != source_color]
        pointer_data[cell] = {
            "coords": _place_pointer(direction, decoy, cell_h, cell_w),
            "color": random.choice(color_choices),
            "direction": direction,
            "source": source,
        }

    grid = _render_grid(n_parts, cell_h, cell_w, pattern_data, pointer_data)
    solved = main(grid)
    expected = _expected_output(grid, pattern_data, pointer_data)
    if not np.array_equal(solved, expected):
        return None

    grey, row_lines, col_lines, _, _ = _find_partition_layout(grid)
    info = _analyze_cells(grid, grey, row_lines, col_lines)
    _, unique = _resolve_roles(info, n_parts, n_parts)
    if not unique:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_ATTEMPTS} attempts")
