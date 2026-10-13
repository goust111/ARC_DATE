import numpy as np


BG = 8
MARKER = 9
PALETTE = (1, 2, 3, 4, 6)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 300
GROUP_ATTEMPTS = 120
SATELLITE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 160
GRID_MIN = 14
GRID_MAX = 30
GROUP_PAD = 2
GAP_CHOICES = (2, 3, 4, 5, 6)


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def in_bounds(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def adjacency_pairs(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in a:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in b:
                pairs.add(((r, c), (nr, nc)))
    return pairs


def touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    return bool(a & b) or bool(adjacency_pairs(a, b))


def count_adjacent(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((nr, nc) in cells for nr, nc in neighbors4(r, c))


def find_components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                comp.add((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def marker_info(grid: np.ndarray, cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    hits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in cells:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == MARKER:
                hits.append(((r, c), (dr, dc)))
    assert len(hits) == 1, "each fragment must have exactly one leader pixel"
    return hits[0]


def steps_until_leader_touches(
    cells: set[tuple[int, int]],
    leader: tuple[int, int],
    direction: tuple[int, int],
    targets: set[tuple[int, int]],
    foreign: set[tuple[int, int]],
    shape: tuple[int, int],
) -> int:
    dr, dc = direction
    limit = shape[0] + shape[1] + 2
    for steps in range(limit):
        shifted = translate(cells, steps * dr, steps * dc)
        if not in_bounds(shifted, shape):
            break
        if shifted & targets:
            continue
        if shifted & foreign:
            continue
        lr = leader[0] + steps * dr
        lc = leader[1] + steps * dc
        if any((lr + ar, lc + ac) in targets for ar, ac in DIRS4):
            return steps
    raise AssertionError("fragment never reaches another same-color block")


def merge_blocks(blocks: list[dict]) -> list[dict]:
    out: list[dict] = []
    used = [False] * len(blocks)
    for i in range(len(blocks)):
        if used[i]:
            continue
        used[i] = True
        group = [i]
        stack = [i]
        while stack:
            cur = stack.pop()
            for j in range(len(blocks)):
                if used[j]:
                    continue
                if touching(blocks[cur]["coords"], blocks[j]["coords"]):
                    used[j] = True
                    group.append(j)
                    stack.append(j)
        if len(group) == 1:
            out.append(blocks[i])
            continue
        merged: set[tuple[int, int]] = set()
        for j in group:
            merged |= blocks[j]["coords"]
        out.append(
            {
                "coords": merged,
                "area": len(merged),
                "leader": None,
                "direction": None,
            }
        )
    return out


def solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = BG if np.any(g == BG) else get_bg(g)
    out = g.copy()
    out[out == MARKER] = bg

    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, MARKER)]
    foreign_by_color: dict[int, set[tuple[int, int]]] = {}
    for color in colors:
        foreign: set[tuple[int, int]] = set()
        for r, c in zip(*np.where((g != bg) & (g != MARKER) & (g != color))):
            foreign.add((int(r), int(c)))
        foreign_by_color[color] = foreign

    for color in colors:
        comps = find_components_of_color(g, color)
        assert len(comps) >= 2, "each active color must start as multiple fragments"
        blocks: list[dict] = []
        for comp in comps:
            leader, direction = marker_info(g, comp)
            blocks.append(
                {
                    "coords": set(comp),
                    "area": len(comp),
                    "leader": leader,
                    "direction": direction,
                }
            )

        while len(blocks) > 1:
            areas = [int(block["area"]) for block in blocks]
            largest = max(areas)
            assert areas.count(largest) == 1, "largest block must be unique"
            largest_idx = areas.index(largest)
            largest_coords = blocks[largest_idx]["coords"]

            choice = None
            for i, block in enumerate(blocks):
                if i == largest_idx:
                    continue
                assert block["leader"] is not None and block["direction"] is not None
                other_coords: set[tuple[int, int]] = set()
                for j, other in enumerate(blocks):
                    if j != i:
                        other_coords |= other["coords"]
                order_dist = steps_until_leader_touches(
                    block["coords"],
                    block["leader"],
                    block["direction"],
                    largest_coords,
                    foreign_by_color[color],
                    (h, w),
                )
                stop_dist = steps_until_leader_touches(
                    block["coords"],
                    block["leader"],
                    block["direction"],
                    other_coords,
                    foreign_by_color[color],
                    (h, w),
                )
                r0, c0, _, _ = bbox(block["coords"])
                key = (order_dist, stop_dist, -r0, -c0)
                if choice is None or key > choice[0]:
                    choice = (key, i, stop_dist)

            assert choice is not None
            _, move_idx, move_steps = choice
            dr, dc = blocks[move_idx]["direction"]
            blocks[move_idx] = {
                "coords": translate(blocks[move_idx]["coords"], move_steps * dr, move_steps * dc),
                "area": blocks[move_idx]["area"],
                "leader": (blocks[move_idx]["leader"][0] + move_steps * dr, blocks[move_idx]["leader"][1] + move_steps * dc),
                "direction": blocks[move_idx]["direction"],
            }
            blocks = merge_blocks(blocks)

        final_cells = blocks[0]["coords"]
        for r, c in zip(*np.where(g == color)):
            out[int(r), int(c)] = bg
        for r, c in final_cells:
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather each color's scattered fragments by sliding them along their maroon-indicated contact side until they fuse into one shape.

    Input structure:
    - The background is azure.
    - Several monochrome 4-connected fragments of the same color are scattered apart; active colors are chosen from red, green, blue, yellow, and magenta.
    - Each fragment has exactly one maroon singleton marker orthogonally adjacent to one of its pixels, which identifies the fragment's leader pixel and direction.

    Rule:
    - For every fragment, the maroon marker's position relative to the leader pixel gives the translation direction.
    - For each color, treat the unique largest fragment as the anchor block, then repeatedly move the farthest remaining same-color fragment along its own direction until its leader pixel first becomes orthogonally adjacent to another same-color block.
    - After all same-color fragments have fused into one connected figure, remove every maroon marker and keep the assembled colored shapes.

    Key insight:
    - The maroon pixels are not part of the shapes: they encode a hidden leader pixel and a move direction, and the fragments must be merged color-by-color around the largest block.
    """
    return solve_grid(np.asarray(grid, dtype=int))


def rotate90(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return normalize({(-c, r) for r, c in cells})


def reflect(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return normalize({(r, -c) for r, c in cells})


def random_polyomino(area: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)
    cells = normalize(cells)
    for _ in range(int(rng.integers(0, 4))):
        cells = rotate90(cells)
    if bool(rng.integers(0, 2)):
        cells = reflect(cells)
    return normalize(cells)


def shuffled(items: list, rng: np.random.Generator) -> list:
    if len(items) <= 1:
        return list(items)
    order = list(rng.permutation(len(items)))
    return [items[i] for i in order]


def marker_candidates(cells: set[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    out: list[tuple[int, int]] = []
    for cell in cells:
        marker = (cell[0] + dr, cell[1] + dc)
        if marker in cells:
            continue
        if count_adjacent(marker, cells) == 1:
            out.append(cell)
    return out


def base_targets(base: set[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [cell for cell in base if (cell[0] - dr, cell[1] - dc) not in base]


def build_group(color: int, rng: np.random.Generator) -> dict | None:
    for _ in range(GROUP_ATTEMPTS):
        sat_count = int(rng.integers(1, 4))
        dir_indices = list(rng.choice(len(DIRS4), size=sat_count, replace=False))
        directions = [DIRS4[i] for i in dir_indices]

        base_area = int(rng.integers(5, 9))
        base = random_polyomino(base_area, rng)
        if len(base) != base_area:
            continue

        satellites = []
        final_union = set(base)
        ok = True
        for direction in directions:
            placed = False
            targets = shuffled(base_targets(base, direction), rng)
            if not targets:
                ok = False
                break
            for _sat in range(SATELLITE_ATTEMPTS):
                sat_area = int(rng.integers(2, base_area))
                sat = random_polyomino(sat_area, rng)
                if len(sat) >= len(base):
                    continue
                leads = shuffled(marker_candidates(sat, direction), rng)
                if not leads:
                    continue
                for leader in leads:
                    for target in targets:
                        dr = target[0] - direction[0] - leader[0]
                        dc = target[1] - direction[1] - leader[1]
                        placed_sat = translate(sat, dr, dc)
                        if placed_sat & final_union:
                            continue
                        pairs = adjacency_pairs(placed_sat, base)
                        if pairs != {((leader[0] + dr, leader[1] + dc), target)}:
                            continue
                        if any(touching(placed_sat, item["final"]) for item in satellites):
                            continue
                        satellites.append(
                            {
                                "direction": direction,
                                "final": placed_sat,
                                "leader_final": (leader[0] + dr, leader[1] + dc),
                            }
                        )
                        final_union |= placed_sat
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        gaps = list(rng.choice(GAP_CHOICES, size=sat_count, replace=False))
        input_fragments = [set(base)]
        output_fragments = [set(base)]
        markers: list[tuple[int, int]] = []
        input_cells = set(base)

        for sat, gap in zip(satellites, gaps):
            dr, dc = sat["direction"]
            shifted = translate(sat["final"], -gap * dr, -gap * dc)
            leader_in = (sat["leader_final"][0] - gap * dr, sat["leader_final"][1] - gap * dc)
            marker = (leader_in[0] + dr, leader_in[1] + dc)
            if touching(shifted, input_cells):
                ok = False
                break
            if marker in shifted or marker in input_cells or marker in markers:
                ok = False
                break
            temp_cells = input_cells | shifted
            if count_adjacent(marker, temp_cells) != 1:
                ok = False
                break
            input_fragments.append(shifted)
            output_fragments.append(set(sat["final"]))
            input_cells |= shifted
            markers.append(marker)
        if not ok:
            continue

        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for i, a in enumerate(markers) for b in markers[i + 1 :]):
            continue

        preferred_dirs = [d for d in DIRS4 if d not in directions] + [d for d in DIRS4 if d in directions]
        base_marker = None
        for direction in preferred_dirs:
            for leader in shuffled(list(base), rng):
                marker = (leader[0] + direction[0], leader[1] + direction[1])
                if marker in input_cells or marker in markers:
                    continue
                if count_adjacent(marker, input_cells) == 1:
                    base_marker = marker
                    break
            if base_marker is not None:
                break
        if base_marker is None:
            continue
        markers.append(base_marker)

        all_input = set(input_cells) | set(markers)
        min_r, min_c, max_r, max_c = bbox(all_input)
        norm_input = {(r - min_r, c - min_c) for r, c in input_cells}
        norm_output = {(r - min_r, c - min_c) for frag in output_fragments for r, c in frag}
        norm_markers = {(r - min_r, c - min_c) for r, c in markers}
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        inp = np.full((height, width), BG, dtype=int)
        out = np.full((height, width), BG, dtype=int)
        for r, c in norm_input:
            inp[r, c] = color
        for r, c in norm_markers:
            inp[r, c] = MARKER
        for r, c in norm_output:
            out[r, c] = color

        if len(find_components_of_color(inp, color)) != sat_count + 1:
            continue
        marker_comps = find_components_of_color(inp, MARKER)
        if len(marker_comps) != sat_count + 1 or any(len(comp) != 1 for comp in marker_comps):
            continue
        for comp in find_components_of_color(inp, color):
            try:
                marker_info(inp, comp)
            except AssertionError:
                ok = False
                break
        if not ok:
            continue

        return {
            "color": color,
            "input_color_cells": norm_input,
            "output_color_cells": norm_output,
            "markers": norm_markers,
            "shape": (height, width),
            "fragment_count": sat_count + 1,
        }
    return None


def place_groups(groups: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = max(group["shape"][0] for group in groups)
    max_w = max(group["shape"][1] for group in groups)
    min_h = max(GRID_MIN, max_h + 2)
    min_w = max(GRID_MIN, max_w + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = int(rng.integers(min_h, GRID_MAX + 1))
    w = int(rng.integers(min_w, GRID_MAX + 1))
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    ordered = sorted(groups, key=lambda g: g["shape"][0] * g["shape"][1], reverse=True)
    placed_rects: list[tuple[int, int, int, int]] = []
    placements: list[tuple[dict, int, int]] = []

    for group in ordered:
        gh, gw = group["shape"]
        placed = False
        if gh > h or gw > w:
            return None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(0, h - gh + 1))
            c0 = int(rng.integers(0, w - gw + 1))
            rect = (r0, c0, r0 + gh - 1, c0 + gw - 1)
            clash = False
            for rr0, cc0, rr1, cc1 in placed_rects:
                if not (
                    rect[2] + GROUP_PAD < rr0
                    or rr1 + GROUP_PAD < rect[0]
                    or rect[3] + GROUP_PAD < cc0
                    or cc1 + GROUP_PAD < rect[1]
                ):
                    clash = True
                    break
            if clash:
                continue
            placed = True
            placed_rects.append(rect)
            placements.append((group, r0, c0))
            break
        if not placed:
            return None

    for group, r0, c0 in placements:
        color = group["color"]
        for r, c in group["input_color_cells"]:
            inp[r0 + r, c0 + c] = color
        for r, c in group["markers"]:
            inp[r0 + r, c0 + c] = MARKER
        for r, c in group["output_color_cells"]:
            out[r0 + r, c0 + c] = color

    return inp, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    group_count = int(rng.integers(1, 4))
    colors = shuffled(list(rng.choice(PALETTE, size=group_count, replace=False)), rng)
    groups = []
    for color in colors:
        group = build_group(int(color), rng)
        if group is None:
            return None
        groups.append(group)
    placed = place_groups(groups, rng)
    if placed is None:
        return None
    inp, expected = placed
    produced = solve_grid(inp)
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
