from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
MAROON = 9
MARKER_COLORS = (BLUE, RED, YELLOW)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _grey_graph(grid: np.ndarray) -> tuple[set[Coord], dict[Coord, list[Coord]], dict[Coord, int]]:
    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    neighbors: dict[Coord, list[Coord]] = {}
    degrees: dict[Coord, int] = {}
    for r, c in grey_cells:
        nbrs = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in grey_cells:
                nbrs.append((nr, nc))
        neighbors[(r, c)] = nbrs
        degrees[(r, c)] = len(nbrs)
    return grey_cells, neighbors, degrees


def _dead_end_markers(grid: np.ndarray, neighbors: dict[Coord, list[Coord]], degrees: dict[Coord, int]) -> dict[Coord, int]:
    markers: dict[Coord, int] = {}
    marker_cells: list[Coord] = []
    for cell, degree in degrees.items():
        if degree != 1:
            continue
        (nr, nc) = neighbors[cell][0]
        dr, dc = cell[0] - nr, cell[1] - nc
        mr, mc = cell[0] + dr, cell[1] + dc
        if not _in_bounds(grid.shape, mr, mc):
            continue
        value = int(grid[mr, mc])
        if value not in {BLACK, GREY}:
            markers[cell] = value
            marker_cells.append((mr, mc))

    colored_cells = {
        tuple(map(int, rc))
        for rc in np.argwhere((grid != BLACK) & (grid != GREY))
    }
    assert len(set(marker_cells)) == len(marker_cells), "each marker must belong to one dead end"
    assert colored_cells == set(marker_cells), "all colored cells must be dead-end markers"
    return markers


def _trace_edges(neighbors: dict[Coord, list[Coord]], degrees: dict[Coord, int]) -> list[list[Coord]]:
    nodes = {cell for cell, degree in degrees.items() if degree != 2}
    visited: set[tuple[Coord, Coord]] = set()
    edges: list[list[Coord]] = []

    for node in sorted(nodes):
        for nbr in neighbors[node]:
            link = tuple(sorted((node, nbr)))
            if link in visited:
                continue
            visited.add(link)
            path = [node, nbr]
            prev, cur = node, nbr
            while cur not in nodes:
                cur_neighbors = neighbors[cur]
                assert len(cur_neighbors) == 2, "non-node pipe cells must have degree 2"
                nxt = cur_neighbors[0] if cur_neighbors[1] == prev else cur_neighbors[1]
                link = tuple(sorted((cur, nxt)))
                visited.add(link)
                path.append(nxt)
                prev, cur = cur, nxt
            edges.append(path)

    return edges


def _prefer_first_endpoint(a: Coord, b: Coord) -> bool:
    return a < b


def _add_uniform_color(path: list[Coord], color: int, flows: dict[Coord, set[int]]) -> None:
    for cell in path:
        flows[cell].add(color)


def _add_split_colors(path: list[Coord], first_color: int, second_color: int, prefer_first: bool,
                      flows: dict[Coord, set[int]]) -> None:
    n = len(path)
    if prefer_first:
        first_limit = (n - 1) // 2
        for i, cell in enumerate(path):
            flows[cell].add(first_color if i <= first_limit else second_color)
    else:
        first_limit = n // 2
        for i, cell in enumerate(path):
            flows[cell].add(first_color if i < first_limit else second_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each grey pipe run from its marked dead-end entrances and paint multi-color junctions maroon.

    Input structure
    - The background is black.
    - Grey pixels form one-cell-wide orthogonal pipe networks whose dead ends have degree 1 and whose junctions have degree 3 or 4.
    - Some dead ends have a single colored marker exactly one cell farther outward along the same direction; other dead ends end against black.

    Rule
    - Treat every maximal non-branching grey run between dead ends or junctions as one pipe segment.
    - A segment with one marked dead end is fully filled by that marker color; a segment with two marked dead ends is filled from both sides and the single middle cell, when present, belongs to the upper end or, if tied horizontally, the left end.
    - Segment colors accumulate on shared junction pixels: one arriving color keeps that color, while two or more different arriving colors turn the junction maroon.
    - Segments with no marked entrance remain grey, and the colored marker pixels themselves stay unchanged.

    Key insight
    - The task separates into two layers: first infer color flow independently along each non-branching pipe run from dead-end markers, then merge the colors only at graph nodes where several runs share the same grey junction pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, GREY, MAROON, *MARKER_COLORS}, "unexpected color in grid"

    grey_cells, neighbors, degrees = _grey_graph(g)
    assert grey_cells, "expected at least one pipe pixel"
    assert all(degree <= 4 for degree in degrees.values()), "pipe pixels must be orthogonal"

    marker_by_end = _dead_end_markers(g, neighbors, degrees)
    edges = _trace_edges(neighbors, degrees)
    flows = {cell: set() for cell in grey_cells}

    for path in edges:
        first, last = path[0], path[-1]
        first_color = marker_by_end.get(first)
        last_color = marker_by_end.get(last)
        if first_color is None and last_color is None:
            continue
        if first_color is None:
            _add_uniform_color(path, int(last_color), flows)
            continue
        if last_color is None or first_color == last_color:
            _add_uniform_color(path, int(first_color), flows)
            continue

        _add_split_colors(
            path,
            int(first_color),
            int(last_color),
            _prefer_first_endpoint(first, last),
            flows,
        )

    out = g.copy()
    for cell, colors_here in flows.items():
        if not colors_here:
            continue
        out[cell] = next(iter(colors_here)) if len(colors_here) == 1 else MAROON

    return out


def _transform(coord: Coord, rotate_quarter_turns: int, flip_horizontal: bool) -> Coord:
    r, c = coord
    if flip_horizontal:
        c = -c
    for _ in range(rotate_quarter_turns % 4):
        r, c = c, -r
    return r, c


def _render_patch(pipe_path: list[Coord], markers: dict[Coord, int]) -> np.ndarray:
    pipe_set = set(pipe_path)
    assert pipe_set
    assert not (pipe_set & markers.keys())
    coords = list(pipe_set) + list(markers)
    min_r = min(r for r, _c in coords) - 1
    max_r = max(r for r, _c in coords) + 1
    min_c = min(c for _r, c in coords) - 1
    max_c = max(c for _r, c in coords) + 1

    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BLACK, dtype=int)
    for r, c in pipe_set:
        patch[r - min_r, c - min_c] = GREY
    for (r, c), color in markers.items():
        patch[r - min_r, c - min_c] = color
    return patch


def _same_or_other(color: int) -> int:
    if random.choice([False, True]):
        return color
    return random.choice([c for c in MARKER_COLORS if c != color])


def _simple_path_patch(mode: str) -> tuple[np.ndarray, set[str]]:
    shape = random.choice(["straight", "elbow", "zigzag"])
    if shape == "straight":
        length = random.randint(3, 8)
        pipe = [(0, c) for c in range(length)]
        slots = {"start": (0, -1), "end": (0, length)}
    elif shape == "elbow":
        down = random.randint(2, 5)
        right = random.randint(2, 5)
        pipe = [(r, 0) for r in range(down)] + [(down - 1, c) for c in range(1, right)]
        slots = {"start": (-1, 0), "end": (down - 1, right)}
    else:
        first = random.randint(2, 4)
        drop = random.randint(2, 4)
        second = random.randint(2, 4)
        pipe = (
            [(0, c) for c in range(first)]
            + [(r, first - 1) for r in range(1, drop)]
            + [(drop - 1, first - 1 + c) for c in range(1, second)]
        )
        slots = {"start": (0, -1), "end": (drop - 1, first - 1 + second)}

    rot = random.randrange(4)
    flip = random.choice([False, True])
    pipe = [_transform(cell, rot, flip) for cell in pipe]
    transformed_slots = {name: _transform(cell, rot, flip) for name, cell in slots.items()}

    markers: dict[Coord, int] = {}
    tags: set[str] = set()
    if mode == "none":
        tags.add("unreached")
    elif mode == "single":
        tags.add("single_path")
        end_name = random.choice(["start", "end"])
        markers[transformed_slots[end_name]] = random.choice(MARKER_COLORS)
    else:
        assert mode == "double"
        tags.add("double_path")
        first_color = random.choice(MARKER_COLORS)
        markers[transformed_slots["start"]] = first_color
        markers[transformed_slots["end"]] = _same_or_other(first_color)

    return _render_patch(pipe, markers), tags


def _cross_patch(mixed: bool) -> tuple[np.ndarray, set[str]]:
    left = random.randint(1, 4)
    right = random.randint(1, 4)
    up = random.randint(1, 4)
    down = random.randint(1, 4)
    pipe = (
        [(0, c) for c in range(-left, right + 1)]
        + [(r, 0) for r in range(-up, down + 1) if r != 0]
    )
    slots = {
        "left": (0, -left - 1),
        "right": (0, right + 1),
        "top": (-up - 1, 0),
        "bottom": (down + 1, 0),
    }
    markers: dict[Coord, int] = {}
    tags = {"intersection_mixed" if mixed else "intersection_single"}

    if mixed:
        h_mode = random.choice(["left", "right", "both"])
        v_mode = random.choice(["top", "bottom", "both"])
        h_color = random.choice(MARKER_COLORS)
        v_color = random.choice([c for c in MARKER_COLORS if c != h_color])
        if h_mode in {"left", "both"}:
            markers[slots["left"]] = h_color
        if h_mode in {"right", "both"}:
            markers[slots["right"]] = h_color
        if v_mode in {"top", "both"}:
            markers[slots["top"]] = v_color
        if v_mode in {"bottom", "both"}:
            markers[slots["bottom"]] = v_color
    else:
        shared_color = random.choice(MARKER_COLORS)
        while True:
            h_mode = random.choice(["none", "left", "right", "both"])
            v_mode = random.choice(["none", "top", "bottom", "both"])
            if h_mode != "none" or v_mode != "none":
                break
        if h_mode in {"left", "both"}:
            markers[slots["left"]] = shared_color
        if h_mode in {"right", "both"}:
            markers[slots["right"]] = shared_color
        if v_mode in {"top", "both"}:
            markers[slots["top"]] = shared_color
        if v_mode in {"bottom", "both"}:
            markers[slots["bottom"]] = shared_color

    return _render_patch(pipe, markers), tags


def _tee_patch(mixed: bool) -> tuple[np.ndarray, set[str]]:
    left = random.randint(1, 4)
    right = random.randint(1, 4)
    stem = random.randint(1, 4)
    pipe = (
        [(0, c) for c in range(-left, right + 1)]
        + [(r, 0) for r in range(1, stem + 1)]
    )
    slots = {
        "left": (0, -left - 1),
        "right": (0, right + 1),
        "stem": (stem + 1, 0),
    }

    markers: dict[Coord, int] = {}
    tags = {"intersection_mixed" if mixed else "intersection_single"}
    arms = ["left", "right", "stem"]
    if mixed:
        marked_arms = random.sample(arms, random.randint(2, 3))
        first_color = random.choice(MARKER_COLORS)
        second_color = random.choice([c for c in MARKER_COLORS if c != first_color])
        color_by_arm = {
            marked_arms[0]: first_color,
            marked_arms[1]: second_color,
        }
        if len(marked_arms) == 3:
            color_by_arm[marked_arms[2]] = random.choice((first_color, second_color))
        for arm, color in color_by_arm.items():
            markers[slots[arm]] = color
    else:
        shared_color = random.choice(MARKER_COLORS)
        marked_arms = random.sample(arms, random.randint(1, 3))
        for arm in marked_arms:
            markers[slots[arm]] = shared_color

    rot = random.randrange(4)
    flip = random.choice([False, True])
    pipe = [_transform(cell, rot, flip) for cell in pipe]
    markers = {
        _transform(cell, rot, flip): color
        for cell, color in markers.items()
    }
    return _render_patch(pipe, markers), tags


def _random_patch() -> tuple[np.ndarray, set[str]]:
    choice = random.choice(
        ["single", "double", "none", "cross_single", "cross_mixed", "tee_single", "tee_mixed"]
    )
    if choice == "single":
        return _simple_path_patch("single")
    if choice == "double":
        return _simple_path_patch("double")
    if choice == "none":
        return _simple_path_patch("none")
    if choice == "cross_single":
        return _cross_patch(False)
    if choice == "cross_mixed":
        return _cross_patch(True)
    if choice == "tee_single":
        return _tee_patch(False)
    return _tee_patch(True)


def _can_place(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    ph, pw = patch.shape
    if top + ph > canvas.shape[0] or left + pw > canvas.shape[1]:
        return False

    for pr, pc in np.argwhere(patch != BLACK):
        gr = top + int(pr)
        gc = left + int(pc)
        if canvas[gr, gc] != BLACK:
            return False
        for dr, dc in DIRS:
            nr, nc = gr + dr, gc + dc
            if _in_bounds(canvas.shape, nr, nc) and canvas[nr, nc] != BLACK:
                return False
    return True


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    view = canvas[top : top + ph, left : left + pw]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _build_scene() -> tuple[np.ndarray, set[str]]:
    patches: list[np.ndarray] = []
    features: set[str] = set()

    patch, tags = _simple_path_patch(random.choice(["single", "double"]))
    patches.append(patch)
    features |= tags

    if random.choice([False, True]) and len(patches) < 4:
        patch, tags = _simple_path_patch("none")
        patches.append(patch)
        features |= tags
    if random.choice([False, True]) and len(patches) < 4:
        patch, tags = _simple_path_patch("double")
        patches.append(patch)
        features |= tags
    if random.choice([False, True]) and len(patches) < 4:
        patch, tags = random.choice([_cross_patch, _tee_patch])(random.choice([False, True]))
        patches.append(patch)
        features |= tags

    target_count = random.randint(len(patches), 4)
    while len(patches) < target_count:
        patch, tags = _random_patch()
        patches.append(patch)
        features |= tags

    min_side = min(30, max(16, 10 + 3 * len(patches)))
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    canvas = np.full((height, width), BLACK, dtype=int)

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(120):
            top = random.randint(0, height - ph)
            left = random.randint(0, width - pw)
            if not _can_place(canvas, patch, top, left):
                continue
            _paste_patch(canvas, patch, top, left)
            placed = True
            break
        if not placed:
            raise ValueError("failed to place patch")

    return canvas, features


def _valid_generated_scene(grid: np.ndarray, features: set[str]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if "unreached" in features and not np.any((grid == GREY) & (out == GREY)):
        return False
    if "intersection_mixed" in features and not np.any(out == MAROON):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            grid, features = _build_scene()
        except ValueError:
            continue
        if _valid_generated_scene(grid, features):
            return grid
    raise RuntimeError("failed to generate a valid pipe-network input")
