from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                    "touches_edge": touches_edge(coords, grid.shape),
                }
            )
    return components


def coords_to_set(coords: np.ndarray) -> frozenset[Coord]:
    return frozenset((int(r), int(c)) for r, c in coords)


def translate_set(cells: frozenset[Coord], dr: int, dc: int) -> frozenset[Coord]:
    return frozenset((r + dr, c + dc) for r, c in cells)


def solid_completion_offsets(
    grid: np.ndarray,
    source: frozenset[Coord],
    anchor: frozenset[Coord],
    bg: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    offsets: list[tuple[int, int]] = []

    for dr in range(1 - h, h):
        for dc in range(1 - w, w):
            moved = translate_set(source, dr, dc)
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in moved):
                continue
            if moved & anchor:
                continue
            if any(int(grid[r, c]) != int(bg) for r, c in moved):
                continue
            union = moved | anchor
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            rect_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
            if len(union) == rect_area:
                offsets.append((dr, dc))

    return offsets


def find_active_translation(grid: np.ndarray) -> tuple[list[dict], int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg)
    by_color: dict[int, list[int]] = defaultdict(list)
    for idx, comp in enumerate(components):
        by_color[int(comp["color"])].append(idx)

    candidates: list[tuple[int, int, int, int]] = []
    for color, idxs in by_color.items():
        if len(idxs) != 2:
            continue
        edge_idxs = [idx for idx in idxs if components[idx]["touches_edge"]]
        anchor_idxs = [idx for idx in idxs if not components[idx]["touches_edge"]]
        if len(edge_idxs) != 1 or len(anchor_idxs) != 1:
            continue

        source_idx = edge_idxs[0]
        anchor_idx = anchor_idxs[0]
        source = coords_to_set(components[source_idx]["coords"])
        anchor = coords_to_set(components[anchor_idx]["coords"])
        offsets = solid_completion_offsets(g, source, anchor, bg)
        if len(offsets) == 1:
            dr, dc = offsets[0]
            candidates.append((source_idx, anchor_idx, dr, dc))

    assert len(candidates) == 1, "expected one unique split pair with one unique completion vector"
    source_idx, anchor_idx, dr, dc = candidates[0]
    return components, source_idx, anchor_idx, dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the edge fragment to complete its same-colored anchor rectangle, then apply that vector to every other object.

    Input structure:
    - The background is the dominant color.
    - Exactly one foreground color is split into two 4-connected fragments: one touches a grid edge, and one anchor fragment stays away from all edges.
    - Translating the edge fragment by one unique vector makes its union with the anchor a filled axis-aligned rectangle.
    - Every other foreground color forms exactly one complete 4-connected object.

    Rule:
    - Find the unique translation vector that moves the edge fragment onto background cells so it complements the anchor into a solid rectangle.
    - Clear the original cells of the moved fragment and place it at that rectangle-completing location.
    - Apply the same translation vector to every other complete foreground object as well, clearing their original cells first.
    - Keep the anchor fixed; all translated objects remain in bounds and do not overlap.

    Key insight:
    - The split pair is not the whole task: it encodes a single displacement vector, and that locally recovered vector must be reused as the global motion rule for the rest of the scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    components, source_idx, anchor_idx, dr, dc = find_active_translation(g)

    out = g.copy()
    for idx, comp in enumerate(components):
        if idx == anchor_idx:
            continue
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for idx, comp in enumerate(components):
        if idx == anchor_idx:
            continue
        color = int(comp["color"])
        shifted = comp["coords"] + np.array([dr, dc], dtype=int)
        assert bool(
            np.all(
                (0 <= shifted[:, 0])
                & (shifted[:, 0] < g.shape[0])
                & (0 <= shifted[:, 1])
                & (shifted[:, 1] < g.shape[1])
            )
        )
        target = out[shifted[:, 0], shifted[:, 1]]
        assert bool(np.all(target == int(bg))), "translated objects must land on empty background"
        out[shifted[:, 0], shifted[:, 1]] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
RECT_H_RANGE = (3, 6)
RECT_W_RANGE = (4, 8)
EXTRA_OBJECT_COUNT_RANGE = (1, 3)
EXTRA_AREA_RANGE = (4, 8)
MAX_ATTEMPTS = 500


def shift_coords(cells: set[Coord] | frozenset[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in cells}


def bbox_of_set(cells: set[Coord] | frozenset[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_coords(cells: set[Coord]) -> tuple[set[Coord], int, int]:
    r0, c0, r1, c1 = bbox_of_set(cells)
    return {(r - r0, c - c0) for r, c in cells}, r1 - r0 + 1, c1 - c0 + 1


def padded(cells: set[Coord], shape: tuple[int, int], radius: int = 1) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def make_split_patch(rect_h: int, rect_w: int, axis: str) -> tuple[set[Coord], set[Coord]]:
    if axis == "vertical":
        while True:
            anchor_widths = [random.randint(1, rect_w - 1) for _ in range(rect_h)]
            if len(set(anchor_widths)) > 1:
                break
        source = {
            (r, c)
            for r, width in enumerate(anchor_widths)
            for c in range(rect_w - width)
        }
    else:
        while True:
            anchor_heights = [random.randint(1, rect_h - 1) for _ in range(rect_w)]
            if len(set(anchor_heights)) > 1:
                break
        source = {
            (r, c)
            for c, height in enumerate(anchor_heights)
            for r in range(rect_h - height)
        }
    full = {(r, c) for r in range(rect_h) for c in range(rect_w)}
    anchor = full - source
    return source, anchor


def flip_local(cells: set[Coord], rect_h: int, rect_w: int, flip_r: bool, flip_c: bool) -> set[Coord]:
    out = set()
    for r, c in cells:
        rr = rect_h - 1 - r if flip_r else r
        cc = rect_w - 1 - c if flip_c else c
        out.add((rr, cc))
    return out


def random_polyomino() -> tuple[set[Coord], int, int]:
    for _ in range(100):
        area = random.randint(*EXTRA_AREA_RANGE)
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if any((cell[0] + dr, cell[1] + dc) in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                cells.add(cell)
                for nr, nc in neighbors4(*cell):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
        norm, h, w = normalize_coords(cells)
        if h >= 2 and w >= 2 and len(norm) < h * w:
            return norm, h, w
    raise RuntimeError("failed to sample polyomino")


def choose_source_input_top(
    grid_shape: tuple[int, int],
    source_h: int,
    source_w: int,
    final_bbox_top: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = grid_shape
    side = random.choice(("left", "right", "top", "bottom"))
    fr, fc = final_bbox_top

    if side in ("left", "right"):
        c0 = 0 if side == "left" else w - source_w
        row_choices = list(range(0, h - source_h + 1))
        if 0 <= fr <= h - source_h and random.random() < 0.35:
            row_choices = [fr]
        r0 = random.choice(row_choices)
    else:
        r0 = 0 if side == "top" else h - source_h
        col_choices = list(range(0, w - source_w + 1))
        if 0 <= fc <= w - source_w and random.random() < 0.35:
            col_choices = [fc]
        c0 = random.choice(col_choices)

    if (r0, c0) == final_bbox_top:
        return None
    return int(r0), int(c0)


def place_extra_objects(
    grid_shape: tuple[int, int],
    vector: tuple[int, int],
    occupied_in: set[Coord],
    blocked_in: set[Coord],
    occupied_out: set[Coord],
    blocked_out: set[Coord],
    colors: list[int],
) -> tuple[list[tuple[int, set[Coord], set[Coord]]], set[Coord], set[Coord], set[Coord], set[Coord]] | None:
    h, w = grid_shape
    dr, dc = vector
    placed: list[tuple[int, set[Coord], set[Coord]]] = []

    for color in colors:
        shape, sh, sw = random_polyomino()
        variants = []
        for flip_r in (False, True):
            for flip_c in (False, True):
                var = {(sh - 1 - r if flip_r else r, sw - 1 - c if flip_c else c) for r, c in shape}
                norm, vh, vw = normalize_coords(var)
                variants.append((norm, vh, vw))
        random.shuffle(variants)

        found = False
        for variant, vh, vw in variants:
            positions = [(r0, c0) for r0 in range(h - vh + 1) for c0 in range(w - vw + 1)]
            random.shuffle(positions)
            for out_r0, out_c0 in positions:
                out_cells = {(out_r0 + r, out_c0 + c) for r, c in variant}
                in_cells = {(r - dr, c - dc) for r, c in out_cells}
                if any(r < 0 or r >= h or c < 0 or c >= w for r, c in in_cells):
                    continue
                if out_cells & blocked_out or in_cells & blocked_in:
                    continue
                occupied_out |= out_cells
                occupied_in |= in_cells
                blocked_out |= padded(out_cells, grid_shape)
                blocked_in |= padded(in_cells, grid_shape)
                placed.append((color, in_cells, out_cells))
                found = True
                break
            if found:
                break
        if not found:
            return None

    return placed, occupied_in, blocked_in, occupied_out, blocked_out


def render_grid(shape: tuple[int, int], bg: int, objects: list[tuple[int, set[Coord]]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, cells in objects:
        for r, c in cells:
            grid[r, c] = int(color)
    return grid


def rectangle_only_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components, source_idx, anchor_idx, dr, dc = find_active_translation(g)
    out = g.copy()
    source = components[source_idx]
    out[source["coords"][:, 0], source["coords"][:, 1]] = int(bg)
    shifted = source["coords"] + np.array([dr, dc], dtype=int)
    out[shifted[:, 0], shifted[:, 1]] = int(source["color"])
    return out


def generator_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)
    if rect_h >= h - 1 or rect_w >= w - 1:
        return None

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    split_color = random.choice(palette)
    palette.remove(split_color)

    axis = random.choice(("vertical", "horizontal"))
    source_local, anchor_local = make_split_patch(rect_h, rect_w, axis)
    flip_r = random.choice((False, True))
    flip_c = random.choice((False, True))
    source_local = flip_local(source_local, rect_h, rect_w, flip_r, flip_c)
    anchor_local = flip_local(anchor_local, rect_h, rect_w, flip_r, flip_c)
    if source_local & anchor_local:
        return None

    rect_r0 = random.randint(1, h - rect_h - 1)
    rect_c0 = random.randint(1, w - rect_w - 1)
    source_final = {(rect_r0 + r, rect_c0 + c) for r, c in source_local}
    anchor_final = {(rect_r0 + r, rect_c0 + c) for r, c in anchor_local}

    source_norm, source_h, source_w = normalize_coords(source_local)
    sr0, sc0, _, _ = bbox_of_set(source_local)
    final_bbox_top = (rect_r0 + sr0, rect_c0 + sc0)
    source_input_top = choose_source_input_top((h, w), source_h, source_w, final_bbox_top)
    if source_input_top is None:
        return None
    source_input = {(source_input_top[0] + r, source_input_top[1] + c) for r, c in source_norm}
    if source_input & anchor_final:
        return None
    if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in source_input):
        return None
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in anchor_final):
        return None

    vector = (final_bbox_top[0] - source_input_top[0], final_bbox_top[1] - source_input_top[1])
    if vector == (0, 0):
        return None

    rect_full = {(rect_r0 + r, rect_c0 + c) for r in range(rect_h) for c in range(rect_w)}
    occupied_out = set(rect_full)
    blocked_out = padded(rect_full, (h, w))
    occupied_in = set(anchor_final) | set(source_input)
    blocked_in = padded(anchor_final, (h, w)) | padded(source_input, (h, w))

    extra_count = random.randint(*EXTRA_OBJECT_COUNT_RANGE)
    extra_colors = random.sample(palette, extra_count)
    placed = place_extra_objects(
        (h, w),
        vector,
        occupied_in,
        blocked_in,
        occupied_out,
        blocked_out,
        extra_colors,
    )
    if placed is None:
        return None
    extra_objects, _, _, _, _ = placed

    inp_objects = [(split_color, anchor_final), (split_color, source_input)] + [
        (color, in_cells) for color, in_cells, _ in extra_objects
    ]
    out_objects = [(split_color, rect_full)] + [
        (color, out_cells) for color, _, out_cells in extra_objects
    ]
    inp = render_grid((h, w), bg, inp_objects)
    out = render_grid((h, w), bg, out_objects)

    if get_bg(inp) != bg or int(np.count_nonzero(inp == bg)) <= max(int(np.count_nonzero(inp == split_color)), *(int(np.count_nonzero(inp == c)) for c in extra_colors), 0):
        return None

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, out):
        return None

    shortcut = rectangle_only_output(inp)
    if np.array_equal(shortcut, out):
        return None

    components = find_components(inp, bg)
    by_color: dict[int, int] = defaultdict(int)
    edge_flags: dict[int, list[bool]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])] += 1
        edge_flags[int(comp["color"])].append(bool(comp["touches_edge"]))
    if by_color[split_color] != 2:
        return None
    if sorted(edge_flags[split_color]) != [False, True]:
        return None
    if any(by_color[color] != 1 for color in extra_colors):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = generator_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
