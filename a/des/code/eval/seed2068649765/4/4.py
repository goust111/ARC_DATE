import random

import numpy as np


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]] | None:
    if len(separators) != 2:
        return None
    bounds = [-1, *sorted(map(int, separators)), int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(3)]
    sizes = [end - start for start, end in intervals]
    if min(sizes) <= 0 or len(set(sizes)) != 1:
        return None
    return intervals


def _find_partition(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    row_colors: dict[int, list[int]] = {}
    col_colors: dict[int, list[int]] = {}

    for r in range(h):
        if np.all(g[r, :] == g[r, 0]):
            row_colors.setdefault(int(g[r, 0]), []).append(int(r))
    for c in range(w):
        if np.all(g[:, c] == g[0, c]):
            col_colors.setdefault(int(g[0, c]), []).append(int(c))

    for color in sorted(set(row_colors) & set(col_colors)):
        row_intervals = _regular_intervals(row_colors[color], h)
        col_intervals = _regular_intervals(col_colors[color], w)
        if row_intervals is not None and col_intervals is not None:
            return int(color), row_intervals, col_intervals

    raise AssertionError("failed to detect the 3x3 grey partition")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    (r0, c0), (r1, c1) = pts.min(axis=0), pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _unique_rotation(ref: np.ndarray, other: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(np.rot90(ref, k), other)]
    assert len(matches) == 1, "stamp rotations must be uniquely identifiable"
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the empty subregion of a 3x3 grey-separated board with the correctly rotated stamp.

    The board is split into nine equal cells by single-pixel separator lines. Eight cells
    contain the same colored stamp in different rotations; one cell is blank. Rotations follow
    the additive row/column pattern:
        0 1 2
        1 2 3
        2 3 0
    where each step means another 90-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_color, row_intervals, col_intervals = _find_partition(g)
    payload = g[g != line_color]
    values, counts = np.unique(payload, return_counts=True)
    bg = int(values[np.argmax(counts)])

    blocks: dict[tuple[int, int], np.ndarray] = {}
    empty_pos = None
    nonempty_positions: list[tuple[int, int]] = []
    bboxes: list[tuple[int, int, int, int]] = []

    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            block = g[r0:r1, c0:c1]
            blocks[(br, bc)] = block
            if np.all(block == bg):
                assert empty_pos is None
                empty_pos = (br, bc)
                continue
            nonempty_positions.append((br, bc))
            bboxes.append(_bbox(block != bg))

    assert empty_pos is not None and len(nonempty_positions) == 8
    bbox = bboxes[0]
    assert all(candidate == bbox for candidate in bboxes)
    r0b, c0b, r1b, c1b = bbox
    assert r1b - r0b == c1b - c0b

    ref_pos = min(nonempty_positions)
    ref_crop = blocks[ref_pos][r0b : r1b + 1, c0b : c1b + 1]
    ref_delta = ref_pos[0] + ref_pos[1]

    for pos in nonempty_positions:
        crop = blocks[pos][r0b : r1b + 1, c0b : c1b + 1]
        actual = _unique_rotation(ref_crop, crop)
        expected = (pos[0] + pos[1] - ref_delta) % 4
        assert actual == expected

    missing_turns = (empty_pos[0] + empty_pos[1] - ref_delta) % 4
    missing_crop = np.rot90(ref_crop, missing_turns)

    out = g.copy()
    mr0, mr1 = row_intervals[empty_pos[0]]
    mc0, mc1 = col_intervals[empty_pos[1]]
    out[mr0:mr1, mc0:mc1] = bg
    out[mr0 + r0b : mr0 + r1b + 1, mc0 + c0b : mc0 + c1b + 1] = missing_crop
    return out


# --------------- generator ---------------

BG_COLOR = 0
LINE_COLOR = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

CELL_SIZE_RANGE = (6, 11)
STAMP_SIZE_MIN = 3
STAMP_COLORS_RANGE = (2, 4)
STAMP_FILL_RANGE = (0.35, 0.72)

MAX_MASK_ATTEMPTS = 100
MAX_STAMP_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, side: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < side and 0 <= nc < side:
            yield nr, nc


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(r, c)]
    axes = [0, 1]
    random.shuffle(axes)
    for axis in axes:
        if axis == 0:
            step = 1 if tr >= r else -1
            while r != tr:
                r += step
                cells.append((r, c))
        else:
            step = 1 if tc >= c else -1
            while c != tc:
                c += step
                cells.append((r, c))
    return cells


def _make_mask(side: int) -> np.ndarray | None:
    area = side * side
    min_fill = max(5, int(np.ceil(area * STAMP_FILL_RANGE[0])))
    max_fill = min(area - 1, int(np.floor(area * STAMP_FILL_RANGE[1])))
    if min_fill > max_fill:
        return None

    for _ in range(MAX_MASK_ATTEMPTS):
        mask = np.zeros((side, side), dtype=bool)
        center = (random.randrange(side), random.randrange(side))
        anchors = [
            (0, random.randrange(side)),
            (side - 1, random.randrange(side)),
            (random.randrange(side), 0),
            (random.randrange(side), side - 1),
        ]
        for anchor in anchors + [center]:
            for r, c in _random_path(anchor, center):
                mask[r, c] = True

        target = random.randint(min_fill, max_fill)
        frontier = {
            nb
            for r, c in zip(*np.where(mask), strict=False)
            for nb in _neighbors4(int(r), int(c), side)
            if not mask[nb]
        }

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for nb in _neighbors4(cell[0], cell[1], side):
                if not mask[nb]:
                    frontier.add(nb)

        if int(mask.sum()) != target:
            continue
        if not (
            mask[0, :].any()
            and mask[-1, :].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        ):
            continue
        if len({mask.tobytes(), np.rot90(mask, 1).tobytes(), np.rot90(mask, 2).tobytes(), np.rot90(mask, 3).tobytes()}) != 4:
            continue
        return mask

    return None


def _assign_colors(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    side = mask.shape[0]
    stamp = np.full((side, side), BG_COLOR, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    seeds = random.sample(coords, len(colors))

    owner = -np.ones((side, side), dtype=int)
    frontiers = [set() for _ in colors]
    for idx, (r, c) in enumerate(seeds):
        owner[r, c] = idx
        frontiers[idx].add((r, c))

    remaining = int(mask.sum()) - len(colors)
    while remaining > 0:
        choices = [idx for idx, frontier in enumerate(frontiers) if frontier]
        idx = random.choice(choices)
        r, c = random.choice(tuple(frontiers[idx]))
        frontiers[idx].remove((r, c))
        for nr, nc in _neighbors4(r, c, side):
            if mask[nr, nc] and owner[nr, nc] == -1:
                owner[nr, nc] = idx
                remaining -= 1
                frontiers[idx].add((nr, nc))

    for idx, color in enumerate(colors):
        stamp[owner == idx] = int(color)
    return stamp


def _make_stamp(side: int) -> np.ndarray | None:
    for _ in range(MAX_STAMP_ATTEMPTS):
        mask = _make_mask(side)
        if mask is None:
            continue

        color_count = random.randint(STAMP_COLORS_RANGE[0], min(STAMP_COLORS_RANGE[1], int(mask.sum())))
        colors = random.sample(STAMP_COLORS, color_count)
        stamp = _assign_colors(mask, colors)

        rotations = [np.rot90(stamp, k) for k in range(4)]
        if len({rot.tobytes() for rot in rotations}) != 4:
            continue
        return stamp

    return None


def _build_full_grid(cell_side: int, stamp: np.ndarray, top: int, left: int) -> np.ndarray:
    size = 3 * cell_side + 2
    full = np.full((size, size), BG_COLOR, dtype=int)
    full[cell_side, :] = LINE_COLOR
    full[2 * cell_side + 1, :] = LINE_COLOR
    full[:, cell_side] = LINE_COLOR
    full[:, 2 * cell_side + 1] = LINE_COLOR

    stamp_side = stamp.shape[0]
    for br in range(3):
        for bc in range(3):
            turns = (br + bc) % 4
            patch = np.rot90(stamp, turns)
            r0 = br * (cell_side + 1) + top
            c0 = bc * (cell_side + 1) + left
            full[r0 : r0 + stamp_side, c0 : c0 + stamp_side] = patch

    return full


def _gen_attempt() -> np.ndarray | None:
    cell_side = random.randint(*CELL_SIZE_RANGE)
    max_stamp_side = cell_side - 1
    if max_stamp_side < STAMP_SIZE_MIN:
        return None
    stamp_side = random.randint(STAMP_SIZE_MIN, max_stamp_side)
    stamp = _make_stamp(stamp_side)
    if stamp is None:
        return None

    top = random.randint(0, cell_side - stamp_side)
    left = random.randint(0, cell_side - stamp_side)
    full = _build_full_grid(cell_side, stamp, top, left)

    missing_index = random.randrange(9)
    missing_pos = divmod(missing_index, 3)
    inp = full.copy()
    r0 = missing_pos[0] * (cell_side + 1)
    c0 = missing_pos[1] * (cell_side + 1)
    inp[r0 : r0 + cell_side, c0 : c0 + cell_side] = BG_COLOR

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(out, full):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotating-stamp puzzle")
