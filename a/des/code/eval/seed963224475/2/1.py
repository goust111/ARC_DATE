import random
from collections import Counter, defaultdict

import numpy as np


DIR_TO_VEC = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
VEC_TO_DIR = {value: key for key, value in DIR_TO_VEC.items()}


GEN_BG = 0
GEN_GREY = 5
GEN_MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_GRID_MIN = 15
GEN_GRID_MAX = 23
GEN_PAIR_COUNT = (1, 3)
GEN_ANCHOR_GAP = (3, 7)
GEN_PAD = 1
GEN_MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))


def find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(coords)
    return components


def rotate_vector(dr: int, dc: int, k: int) -> tuple[int, int]:
    dr, dc = int(dr), int(dc)
    for _ in range(k % 4):
        dr, dc = -dc, dr
    return int(dr), int(dc)


def rotate_direction(direction: str, k: int) -> str:
    return VEC_TO_DIR[rotate_vector(*DIR_TO_VEC[direction], k)]


def rotation_steps(src: str, dst: str) -> int:
    for k in range(4):
        if rotate_direction(src, k) == dst:
            return k
    raise AssertionError("unreachable")


def rotate_rel_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    rotated = []
    for r, c, color in cells:
        nr, nc = rotate_vector(int(r), int(c), k)
        rotated.append((int(nr), int(nc), int(color)))
    return rotated


def translate_cells(cells: list[tuple[int, int, int]], origin: tuple[int, int]) -> list[tuple[int, int, int]]:
    r0, c0 = map(int, origin)
    return [(r0 + int(r), c0 + int(c), int(color)) for r, c, color in cells]


def sort_anchor_pair(anchors: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert len(anchors) == 2
    a0, a1 = anchors
    if a0[0] == a1[0]:
        return sorted(anchors, key=lambda rc: (rc[1], rc[0]))
    assert a0[1] == a1[1]
    return sorted(anchors, key=lambda rc: (rc[0], rc[1]))


def desired_directions_for_pair(anchors: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    ordered = sort_anchor_pair(list(anchors))
    a0, a1 = ordered
    if a0[0] == a1[0]:
        return {a0: "right", a1: "left"}
    return {a0: "down", a1: "up"}


def analyze_fragment(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    g = np.asarray(grid, dtype=int)
    counts = Counter(int(g[r, c]) for r, c in coords)
    assert len(counts) == 2, "each fragment must contain grey plus one marker color"

    marker_colors = [color for color, count in counts.items() if count == 1]
    assert len(marker_colors) == 1, "fragment must contain exactly one marker pixel"
    marker_color = int(marker_colors[0])
    marker_positions = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == marker_color]
    assert len(marker_positions) == 1
    marker = marker_positions[0]

    coord_set = set((int(r), int(c)) for r, c in coords)
    exposed = []
    for name, (dr, dc) in DIR_TO_VEC.items():
        if (marker[0] + int(dr), marker[1] + int(dc)) not in coord_set:
            exposed.append(name)
    assert len(exposed) == 1, "marker must expose exactly one interface direction"

    rel_cells = []
    for r, c in coords:
        rel_cells.append((int(r - marker[0]), int(c - marker[1]), int(g[r, c])))

    return {
        "marker_color": marker_color,
        "marker_pos": marker,
        "interface": exposed[0],
        "rel_cells": rel_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each marked grey fragment so its marker-side interface faces the matching anchor,
    then translate the marker onto the nearer same-color anchor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    anchors_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    fragments_by_color: dict[int, list[dict]] = defaultdict(list)

    for coords in find_components(g, bg):
        if len(coords) == 1:
            r, c = coords[0]
            anchors_by_color[int(g[r, c])].append((int(r), int(c)))
        else:
            fragment = analyze_fragment(g, coords)
            fragments_by_color[int(fragment["marker_color"])].append(fragment)

    assert anchors_by_color
    assert set(anchors_by_color) == set(fragments_by_color)

    out = np.full((h, w), bg, dtype=int)
    for color in fragments_by_color:
        anchors = sort_anchor_pair(anchors_by_color[color])
        fragments = fragments_by_color[color]
        assert len(anchors) == 2
        assert len(fragments) == 2

        desired = desired_directions_for_pair(anchors)
        assigned: dict[int, dict] = {}
        for fragment in fragments:
            mr, mc = fragment["marker_pos"]
            distances = [abs(mr - ar) + abs(mc - ac) for ar, ac in anchors]
            assert distances[0] != distances[1], "each fragment must have a unique nearer anchor"
            idx = 0 if distances[0] < distances[1] else 1
            assert idx not in assigned, "both fragments cannot map to the same anchor"
            assigned[idx] = fragment
        assert len(assigned) == 2

        for idx, anchor in enumerate(anchors):
            fragment = assigned[idx]
            k = rotation_steps(fragment["interface"], desired[anchor])
            placed = translate_cells(rotate_rel_cells(fragment["rel_cells"], k), anchor)
            for r, c, cell_color in placed:
                assert 0 <= r < h and 0 <= c < w
                assert int(out[r, c]) == bg, "placed fragments must not overlap"
                out[r, c] = int(cell_color)

    return out


def reserve_cells(mask: np.ndarray, cells: list[tuple[int, int, int]], pad: int) -> None:
    h, w = mask.shape
    for r, c, _color in cells:
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        mask[r0:r1, c0:c1] = True


def cells_are_clear(mask: np.ndarray, cells: list[tuple[int, int, int]]) -> bool:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    for r, c, _color in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (int(r), int(c)) in seen:
            return False
        seen.add((int(r), int(c)))
        if bool(mask[int(r), int(c)]):
            return False
    return True


def paint_cells(grid: np.ndarray, cells: list[tuple[int, int, int]], bg: int) -> None:
    for r, c, color in cells:
        current = int(grid[r, c])
        assert current in (bg, int(color))
        grid[r, c] = int(color)


def grow_fragment(color: int) -> list[tuple[int, int, int]]:
    cells = {
        (0, 0): int(color),
        (0, -1): GEN_GREY,
        (0, 1): GEN_GREY,
        (1, 0): GEN_GREY,
    }
    target_area = random.randint(4, 6)
    while len(cells) < target_area:
        frontier = set()
        for r, c in list(cells):
            for nr, nc in neighbors4(int(r), int(c)):
                if (nr, nc) in cells:
                    continue
                if nr < 0 or nr > 3 or abs(nc) > 3:
                    continue
                frontier.add((int(nr), int(nc)))
        choice = random.choice(tuple(frontier))
        cells[choice] = GEN_GREY
    return [(int(r), int(c), int(cell_color)) for (r, c), cell_color in sorted(cells.items())]


def pair_abs_cells(
    base_a: list[tuple[int, int, int]],
    base_b: list[tuple[int, int, int]],
    orientation: str,
    anchors: list[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    if orientation == "h":
        desired_a, desired_b = "right", "left"
    else:
        desired_a, desired_b = "down", "up"
    rel_a = rotate_rel_cells(base_a, rotation_steps("up", desired_a))
    rel_b = rotate_rel_cells(base_b, rotation_steps("up", desired_b))
    return translate_cells(rel_a, anchors[0]), translate_cells(rel_b, anchors[1])


def try_place_output_pair(
    h: int,
    w: int,
    reserved: np.ndarray,
    color: int,
    orientation: str,
) -> dict | None:
    base_a = grow_fragment(color)
    base_b = grow_fragment(color)

    max_gap = min(GEN_ANCHOR_GAP[1], (w - 1) if orientation == "h" else (h - 1))
    if max_gap < GEN_ANCHOR_GAP[0]:
        return None

    for _ in range(250):
        gap = random.randint(GEN_ANCHOR_GAP[0], max_gap)
        if orientation == "h":
            row = random.randrange(h)
            c0 = random.randrange(w - gap)
            anchors = [(int(row), int(c0)), (int(row), int(c0 + gap))]
        else:
            col = random.randrange(w)
            r0 = random.randrange(h - gap)
            anchors = [(int(r0), int(col)), (int(r0 + gap), int(col))]

        cells_a, cells_b = pair_abs_cells(base_a, base_b, orientation, anchors)
        union = cells_a + cells_b
        if not cells_are_clear(reserved, union):
            continue

        reserve_cells(reserved, union, GEN_PAD)
        return {
            "color": int(color),
            "orientation": orientation,
            "anchors": anchors,
            "fragments": [
                {"base_cells": base_a, "anchor_idx": 0},
                {"base_cells": base_b, "anchor_idx": 1},
            ],
            "output_cells": [cells_a, cells_b],
        }
    return None


def nearer_assigned_anchor(
    marker: tuple[int, int],
    anchors: list[tuple[int, int]],
    anchor_idx: int,
) -> bool:
    target = anchors[anchor_idx]
    other = anchors[1 - anchor_idx]
    d_target = abs(marker[0] - target[0]) + abs(marker[1] - target[1])
    d_other = abs(marker[0] - other[0]) + abs(marker[1] - other[1])
    return d_target < d_other


def try_place_input_fragment(
    grid: np.ndarray,
    reserved: np.ndarray,
    pair_info: dict,
    fragment_info: dict,
) -> bool:
    h, w = grid.shape
    k = random.randrange(4)
    rel_cells = rotate_rel_cells(fragment_info["base_cells"], k)
    candidates = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidates)

    for marker in candidates:
        if not nearer_assigned_anchor(marker, pair_info["anchors"], fragment_info["anchor_idx"]):
            continue
        abs_cells = translate_cells(rel_cells, marker)
        if not cells_are_clear(reserved, abs_cells):
            continue
        paint_cells(grid, abs_cells, GEN_BG)
        reserve_cells(reserved, abs_cells, GEN_PAD)
        return True
    return False


def build_output_grid(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.full(shape, GEN_BG, dtype=int)
    for pair in pairs:
        for cells in pair["output_cells"]:
            paint_cells(out, cells, GEN_BG)
    return out


def build_input_grid(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray | None:
    inp = np.full(shape, GEN_BG, dtype=int)
    reserved = np.zeros(shape, dtype=bool)

    for pair in pairs:
        for anchor in pair["anchors"]:
            r, c = anchor
            if bool(reserved[r, c]):
                return None
            inp[r, c] = pair["color"]
            reserve_cells(reserved, [(r, c, pair["color"])], GEN_PAD)

    fragment_jobs = [(pair, fragment) for pair in pairs for fragment in pair["fragments"]]
    random.shuffle(fragment_jobs)
    for pair, fragment in fragment_jobs:
        if not try_place_input_fragment(inp, reserved, pair, fragment):
            return None
    return inp


def gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    pair_count = random.randint(*GEN_PAIR_COUNT)
    colors = random.sample(GEN_MARKER_COLORS, pair_count)

    reserved = np.zeros((h, w), dtype=bool)
    pairs = []
    for color in colors:
        orientation = random.choice(("h", "v"))
        pair = try_place_output_pair(h, w, reserved, color, orientation)
        if pair is None:
            return None
        pairs.append(pair)

    out = build_output_grid((h, w), pairs)
    inp = build_input_grid((h, w), pairs)
    if inp is None:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
