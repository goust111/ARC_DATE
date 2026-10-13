import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3)
GRID_LATTICE_SIZE = 4
MAX_GENERATION_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return (r0, c0), mask


def _stamp(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _extract_sprite_info(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    comps = _components(grid == GREY)
    assert 7 <= len(comps) <= 9, "expected 7-9 occupied grey sprites"
    template = None
    sprites = []
    for coords in comps:
        top_left, mask = _mask_from_coords(coords)
        area = int(mask.sum())
        assert 4 <= area <= 6, "sprite area must be 4-6"
        if template is None:
            template = mask
        else:
            assert mask.shape == template.shape and np.array_equal(mask, template), "all sprites must be identical"
        sprites.append({"coords": coords, "top_left": top_left, "mask": mask})
    assert template is not None
    return template, sprites


def _touching_marker_color(grid: np.ndarray, coords: np.ndarray, bg: int) -> int | None:
    h, w = grid.shape
    ordered = sorted((int(r), int(c)) for r, c in coords)
    for r, c in ordered:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                value = int(grid[nr, nc])
                if value not in (bg, GREY):
                    return value
    return None


def _slot_cells(top: int, left: int, mask: np.ndarray) -> np.ndarray:
    rel = np.argwhere(mask)
    out = rel.copy()
    out[:, 0] += int(top)
    out[:, 1] += int(left)
    return out


def _slot_color(grid: np.ndarray, top: int, left: int, mask: np.ndarray) -> int:
    coords = _slot_cells(top, left, mask)
    values = {int(grid[r, c]) for r, c in coords}
    assert len(values) == 1
    return values.pop()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer marker colors into adjacent grey sprites, erase the markers, then fill the missing lattice slots using row-first color propagation.

    Input structure:
    - A black background contains 7-9 identical grey 4-connected sprites arranged on a hidden 4x4 row-column lattice.
    - Three single-pixel markers of distinct non-grey colors sit 4-neighbor-adjacent to three occupied sprites.
    - The three marked sprites lie on different lattice rows; unoccupied lattice slots are pure background.

    Rule:
    - For each grey sprite, if a non-background non-grey marker pixel touches it by 4-neighborhood, recolor the whole sprite with that marker color.
    - Erase every marker pixel back to background.
    - Read the common sprite mask plus all occupied sprite top-left positions to recover the lattice row starts and column starts.
    - For every empty lattice slot, stamp a copy of the sprite mask: use the color of an already-transferred sprite in the same row if one exists, otherwise use the transferred color from the same column if one exists, otherwise keep the new sprite grey.
    - Existing unmarked grey sprites stay grey.

    Key insight:
    - The markers only inject color into three existing sprites; the real completion rule is that empty lattice cells inherit color from the active row first and from the active column only as a fallback.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == BG, "expected black background"

    template, sprites = _extract_sprite_info(g)
    rows = sorted({int(info["top_left"][0]) for info in sprites})
    cols = sorted({int(info["top_left"][1]) for info in sprites})
    assert len(rows) == GRID_LATTICE_SIZE and len(cols) == GRID_LATTICE_SIZE, "expected a 4x4 lattice"

    occupied = {(int(info["top_left"][0]), int(info["top_left"][1])): info for info in sprites}
    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}

    out = g.copy()
    marker_mask = (g != bg) & (g != GREY)
    out[marker_mask] = bg

    for info in sprites:
        top, left = map(int, info["top_left"])
        color = _touching_marker_color(g, info["coords"], bg)
        if color is None:
            continue
        _stamp(out, top, left, template, color)
        row_colors[top] = int(color)
        col_colors[left] = int(color)

    for top in rows:
        for left in cols:
            if (top, left) in occupied:
                continue
            color = row_colors.get(top)
            if color is None:
                color = col_colors.get(left, GREY)
            _stamp(out, top, left, template, color)

    return out


def _random_sprite_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        canvas_h = rng.randint(3, 4)
        canvas_w = rng.randint(3, 4)
        target_area = rng.randint(4, 6)
        start = (rng.randrange(canvas_h), rng.randrange(canvas_w))
        cells = {start}
        frontier = {start}
        while len(cells) < target_area and frontier:
            base = rng.choice(tuple(frontier))
            candidates = []
            for nr, nc in neighbors4(*base):
                if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                    candidates.append((nr, nc))
            if candidates:
                nxt = rng.choice(candidates)
                cells.add(nxt)
                frontier.add(nxt)
            else:
                frontier.remove(base)
        if len(cells) != target_area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        _, mask = _mask_from_coords(coords)
        h, w = mask.shape
        area = int(mask.sum())
        if area < 4 or area > 6:
            continue
        if h < 2 or w < 2:
            continue
        if area == h * w:
            continue
        return mask
    raise RuntimeError("failed to sample a valid irregular sprite mask")


def _sample_axis(rng: random.Random, span: int) -> tuple[list[int], int]:
    margin_start = rng.randint(1, 3)
    starts = [margin_start]
    for _ in range(GRID_LATTICE_SIZE - 1):
        gap = rng.randint(span + 2, 6)
        starts.append(starts[-1] + gap)
    margin_end = rng.randint(1, 3)
    total = starts[-1] + span + margin_end
    assert total <= 30
    return starts, total


def _build_occupancy(rng: random.Random) -> tuple[set[tuple[int, int]], dict]:
    row_ids = list(range(GRID_LATTICE_SIZE))
    col_ids = list(range(GRID_LATTICE_SIZE))
    rng.shuffle(row_ids)
    rng.shuffle(col_ids)

    marked_rows = row_ids[:3]
    unmarked_row = row_ids[3]
    marked_cols = col_ids[:3]
    unmarked_col = col_ids[3]

    marked_slots = [(marked_rows[i], marked_cols[i]) for i in range(3)]
    forced_empty = {
        (marked_rows[0], marked_cols[1]),
        (marked_rows[1], unmarked_col),
        (unmarked_row, marked_cols[2]),
        (unmarked_row, unmarked_col),
    }

    occupied = set(marked_slots)
    occupied.add((unmarked_row, marked_cols[0]))
    occupied.add((marked_rows[2], unmarked_col))

    all_slots = {(r, c) for r in range(GRID_LATTICE_SIZE) for c in range(GRID_LATTICE_SIZE)}
    candidates = sorted(all_slots - occupied - forced_empty)
    target_count = rng.randint(7, 9)
    while len(occupied) < target_count and candidates:
        slot = rng.choice(candidates)
        candidates.remove(slot)
        occupied.add(slot)

    info = {
        "marked_rows": marked_rows,
        "unmarked_row": unmarked_row,
        "marked_cols": marked_cols,
        "unmarked_col": unmarked_col,
        "marked_slots": marked_slots,
        "forced_empty": forced_empty,
        "conflict_slot": (marked_rows[0], marked_cols[1]),
        "row_only_slot": (marked_rows[1], unmarked_col),
        "column_only_slot": (unmarked_row, marked_cols[2]),
        "grey_slot": (unmarked_row, unmarked_col),
        "support_row_slot": (unmarked_row, marked_cols[0]),
        "support_col_slot": (marked_rows[2], unmarked_col),
    }
    return occupied, info


def _place_all_sprites(grid: np.ndarray, occupancy: set[tuple[int, int]], row_starts: list[int], col_starts: list[int], mask: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    slot_cells: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for rr, cc in occupancy:
        top = row_starts[rr]
        left = col_starts[cc]
        _stamp(grid, top, left, mask, GREY)
        slot_cells[(rr, cc)] = {tuple(map(int, cell)) for cell in _slot_cells(top, left, mask)}
    return slot_cells


def _marker_candidates(
    sprite_cells: set[tuple[int, int]],
    all_sprite_cells: set[tuple[int, int]],
    used_markers: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    choices = []
    for r, c in sorted(sprite_cells):
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in all_sprite_cells or (nr, nc) in used_markers:
                continue
            touching = 0
            for ar, ac in neighbors4(nr, nc):
                if (ar, ac) in sprite_cells:
                    touching += 1
                elif (ar, ac) in all_sprite_cells:
                    touching = -99
                    break
            if touching >= 1:
                choices.append((nr, nc))
    return sorted(set(choices))


def _build_expected_output(
    shape: tuple[int, int],
    mask: np.ndarray,
    row_starts: list[int],
    col_starts: list[int],
    occupancy: set[tuple[int, int]],
    marked_slots: list[tuple[int, int]],
    color_by_marked_slot: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    row_colors = {row_starts[r]: int(color_by_marked_slot[(r, c)]) for r, c in marked_slots}
    col_colors = {col_starts[c]: int(color_by_marked_slot[(r, c)]) for r, c in marked_slots}

    for r, c in occupancy:
        color = color_by_marked_slot.get((r, c), GREY)
        _stamp(out, row_starts[r], col_starts[c], mask, color)

    for r in range(GRID_LATTICE_SIZE):
        for c in range(GRID_LATTICE_SIZE):
            if (r, c) in occupancy:
                continue
            color = row_colors.get(row_starts[r])
            if color is None:
                color = col_colors.get(col_starts[c], GREY)
            _stamp(out, row_starts[r], col_starts[c], mask, color)

    return out


def _validate_example(
    inp: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    row_starts: list[int],
    col_starts: list[int],
    occupancy: set[tuple[int, int]],
    info: dict,
    color_by_marked_slot: dict[tuple[int, int], int],
) -> bool:
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    template, sprites = _extract_sprite_info(inp)
    if not np.array_equal(template, mask):
        return False
    rows = sorted({int(s["top_left"][0]) for s in sprites})
    cols = sorted({int(s["top_left"][1]) for s in sprites})
    if rows != row_starts or cols != col_starts:
        return False

    marker_cells = list(zip(*np.where((inp != BG) & (inp != GREY))))
    if len(marker_cells) != 3:
        return False
    marker_values = {int(inp[r, c]) for r, c in marker_cells}
    if marker_values != set(MARKER_COLORS):
        return False

    comp_map = {}
    for idx, sprite in enumerate(sprites):
        for r, c in sprite["coords"]:
            comp_map[(int(r), int(c))] = idx

    touched_rows = set()
    touched_cols = set()
    for r, c in marker_cells:
        touched = set()
        for nr, nc in neighbors4(int(r), int(c)):
            if (nr, nc) in comp_map:
                touched.add(comp_map[(nr, nc)])
        if len(touched) != 1:
            return False
        sprite = sprites[next(iter(touched))]
        touched_rows.add(int(sprite["top_left"][0]))
        touched_cols.add(int(sprite["top_left"][1]))
    if len(touched_rows) != 3 or len(touched_cols) != 3:
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    conflict_row, conflict_col = info["conflict_slot"]
    row_only_row, row_only_col = info["row_only_slot"]
    column_row, column_col = info["column_only_slot"]
    grey_row, grey_col = info["grey_slot"]

    conflict_color = color_by_marked_slot[(conflict_row, info["marked_cols"][0])]
    conflict_column_color = color_by_marked_slot[(info["marked_rows"][1], conflict_col)]
    if conflict_color == conflict_column_color:
        return False

    if _slot_color(actual, row_starts[conflict_row], col_starts[conflict_col], mask) != conflict_color:
        return False
    if _slot_color(actual, row_starts[row_only_row], col_starts[row_only_col], mask) != color_by_marked_slot[(row_only_row, info["marked_cols"][1])]:
        return False
    if _slot_color(actual, row_starts[column_row], col_starts[column_col], mask) != color_by_marked_slot[(info["marked_rows"][2], column_col)]:
        return False
    if _slot_color(actual, row_starts[grey_row], col_starts[grey_col], mask) != GREY:
        return False

    if _slot_color(actual, row_starts[info["support_row_slot"][0]], col_starts[info["support_row_slot"][1]], mask) != GREY:
        return False
    if _slot_color(actual, row_starts[info["support_col_slot"][0]], col_starts[info["support_col_slot"][1]], mask) != GREY:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mask = _random_sprite_mask(rng)
    row_starts, grid_h = _sample_axis(rng, mask.shape[0])
    col_starts, grid_w = _sample_axis(rng, mask.shape[1])

    occupancy, info = _build_occupancy(rng)
    color_order = list(MARKER_COLORS)
    rng.shuffle(color_order)
    color_by_marked_slot = {slot: color_order[i] for i, slot in enumerate(info["marked_slots"])}

    inp = np.zeros((grid_h, grid_w), dtype=int)
    slot_cells = _place_all_sprites(inp, occupancy, row_starts, col_starts, mask)
    all_sprite_cells = set().union(*slot_cells.values())
    used_markers: set[tuple[int, int]] = set()

    for slot in info["marked_slots"]:
        candidates = _marker_candidates(slot_cells[slot], all_sprite_cells, used_markers, inp.shape)
        if not candidates:
            return None
        pos = rng.choice(candidates)
        inp[pos] = color_by_marked_slot[slot]
        used_markers.add(pos)

    expected = _build_expected_output(inp.shape, mask, row_starts, col_starts, occupancy, info["marked_slots"], color_by_marked_slot)
    if not _validate_example(inp, expected, mask, row_starts, col_starts, occupancy, info, color_by_marked_slot):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
