from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
GREY = 5
FRAME_COLORS = (1, 3, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
PAIR_COUNT_OPTIONS = (1, 2, 3, 4)
INTERIOR_RANGES = {
    1: ((3, 8), (3, 8)),
    2: ((3, 6), (3, 7)),
    3: ((3, 4), (3, 6)),
    4: ((2, 3), (3, 5)),
}
SHAPE_MODES = ("blob", "rect", "rays")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _frame_record(
    grid: np.ndarray, color: int, coords: list[tuple[int, int]]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        sub[r - r0, c - c0] = True

    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(sub, expected):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    yellow = np.argwhere(interior == YELLOW)
    assert yellow.shape[0] == 1, "each frame interior must contain exactly one yellow anchor"
    grey = np.argwhere(interior == GREY)
    assert grey.shape[0] >= 1, "each frame interior must contain a grey shape"
    allowed = {BG, GREY, YELLOW}
    extra = sorted(int(v) for v in np.unique(interior) if int(v) not in allowed)
    assert not extra, "frame interiors may only contain background, grey, and yellow"

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "anchor": (int(yellow[0, 0]), int(yellow[0, 1])),
        "grey": [(int(r), int(c)) for r, c in grey],
    }


def _extract_pairs(grid: np.ndarray) -> list[tuple[tuple[int, int], list[dict[str, object]]]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, GREY, YELLOW})
    frames: list[dict[str, object]] = []

    for color in colors:
        for comp in _components(grid == int(color)):
            frame = _frame_record(grid, int(color), comp)
            assert frame is not None, "every non-grey/yellow foreground component must be a frame"
            frames.append(frame)

    assert frames and len(frames) % 2 == 0
    sizes = {
        (
            int(frame["bbox"][2]) - int(frame["bbox"][0]) + 1,
            int(frame["bbox"][3]) - int(frame["bbox"][1]) + 1,
        )
        for frame in frames
    }
    assert len(sizes) == 1, "all frames must be the same size"

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)

    pairs: list[tuple[tuple[int, int], list[dict[str, object]]]] = []
    for items in by_color.values():
        assert len(items) == 2, "frame colors must define unique pairs"
        key = (
            min(int(item["bbox"][0]) for item in items),
            min(int(item["bbox"][1]) for item in items),
        )
        pairs.append((key, items))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _compose_pair(pair: list[dict[str, object]]) -> np.ndarray:
    offsets: set[tuple[int, int]] = set()
    for frame in pair:
        ar, ac = frame["anchor"]
        for gr, gc in frame["grey"]:
            offsets.add((int(gr) - int(ar), int(gc) - int(ac)))

    rows = [0] + [dr for dr, _ in offsets]
    cols = [0] + [dc for _, dc in offsets]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for dr, dc in offsets:
        out[dr - min_r, dc - min_c] = GREY
    out[-min_r, -min_c] = RED
    return out


def _stack_patterns(patterns: list[np.ndarray]) -> np.ndarray:
    assert patterns
    height = sum(int(p.shape[0]) for p in patterns) + 2 * (len(patterns) - 1)
    width = max(int(p.shape[1]) for p in patterns)
    out = np.full((height, width), BG, dtype=int)
    r0 = 0
    for pattern in patterns:
        ph, pw = pattern.shape
        out[r0 : r0 + ph, :pw] = pattern
        r0 += ph + 2
    return out


def _compose_from_interiors(interiors: list[np.ndarray]) -> np.ndarray:
    pair: list[dict[str, object]] = []
    for interior in interiors:
        yellow = np.argwhere(interior == YELLOW)
        grey = np.argwhere(interior == GREY)
        assert yellow.shape[0] == 1 and grey.shape[0] >= 1
        pair.append(
            {
                "anchor": (int(yellow[0, 0]), int(yellow[0, 1])),
                "grey": [(int(r), int(c)) for r, c in grey],
            }
        )
    return _compose_pair(pair)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each same-frame-color pair by its yellow anchor and merge the two interior motifs into stacked composites.

    Input structure:
    - The background is black.
    - The grid contains an even number of equal-sized one-pixel-thick rectangular frames.
    - Each frame encloses background, one grey shape, and exactly one yellow anchor pixel.
    - The two frames in a pair have the same border color.

    Rule:
    - For each same-color frame pair, discard the borders and extract the interior motif formed by the grey pixels and the yellow anchor.
    - Translate the two motifs so that their yellow anchors coincide.
    - Draw the union of the two grey shapes in the output, and recolor the shared anchor position red.
    - Stack the composite results from topmost pair to bottommost pair, separating consecutive composites by two black rows.

    Key insight:
    - The yellow pixel is an alignment reference rather than part of the grey figure, and border color is the only cue that tells which two boxes must be merged.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _extract_pairs(g)
    patterns = [_compose_pair(pair) for _, pair in pairs]
    return _stack_patterns(patterns)


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(int(cut - prev - 1))
        prev = cut
    return values


def _layout_positions(
    n_frames: int, outer_h: int, outer_w: int
) -> tuple[int, int, list[tuple[int, int]]] | None:
    layouts: list[tuple[int, int, int, int]] = []
    for slot_rows in range(1, n_frames + 1):
        slot_cols = (n_frames + slot_rows - 1) // slot_rows
        min_rows = slot_rows * outer_h + (slot_rows + 1)
        min_cols = slot_cols * outer_w + (slot_cols + 1)
        if min_rows <= 30 and min_cols <= 30:
            layouts.append((slot_rows, slot_cols, min_rows, min_cols))
    if not layouts:
        return None

    slot_rows, slot_cols, min_rows, min_cols = random.choice(layouts)
    rows = min_rows + random.randint(0, 30 - min_rows)
    cols = min_cols + random.randint(0, 30 - min_cols)

    row_gaps = [1 + v for v in _random_partition(rows - min_rows, slot_rows + 1)]
    col_gaps = [1 + v for v in _random_partition(cols - min_cols, slot_cols + 1)]

    row_starts: list[int] = []
    cursor = row_gaps[0]
    for idx in range(slot_rows):
        row_starts.append(int(cursor))
        cursor += outer_h + row_gaps[idx + 1]

    col_starts: list[int] = []
    cursor = col_gaps[0]
    for idx in range(slot_cols):
        col_starts.append(int(cursor))
        cursor += outer_w + col_gaps[idx + 1]

    cells = [(rr, cc) for rr in range(slot_rows) for cc in range(slot_cols)]
    chosen = random.sample(cells, n_frames)
    positions = [(row_starts[rr], col_starts[cc]) for rr, cc in chosen]
    random.shuffle(positions)
    return int(rows), int(cols), positions


def _random_anchor(h: int, w: int) -> tuple[int, int]:
    return random.randrange(h), random.randrange(w)


def _blob_shape(h: int, w: int, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    choices = [(r, c) for r in range(h) for c in range(w) if (r, c) != anchor]
    seed = random.choice(choices)
    target = random.randint(1, min(len(choices), max(2, (h * w) // 2)))
    cells = {seed}
    frontier = [cell for cell in _neighbors4(*seed) if 0 <= cell[0] < h and 0 <= cell[1] < w and cell != anchor]

    while frontier and len(cells) < target:
        idx = random.randrange(len(frontier))
        cell = frontier.pop(idx)
        if cell in cells or cell == anchor:
            continue
        cells.add(cell)
        for nxt in _neighbors4(*cell):
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt != anchor and nxt not in cells:
                frontier.append(nxt)
    return cells


def _rect_shape(h: int, w: int, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    rect_h = random.randint(1, h)
    rect_w = random.randint(1, w)
    r0 = random.randint(0, h - rect_h)
    c0 = random.randint(0, w - rect_w)
    cells = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}
    if len(cells) >= 3 and random.random() < 0.6:
        corners = [(r0, c0), (r0, c0 + rect_w - 1), (r0 + rect_h - 1, c0), (r0 + rect_h - 1, c0 + rect_w - 1)]
        cell = random.choice(corners)
        if cell in cells and cell != anchor:
            cells.remove(cell)
    cells.discard(anchor)
    return cells


def _rays_shape(h: int, w: int, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    choices = [(r, c) for r in range(h) for c in range(w) if (r, c) != anchor]
    center = random.choice(choices)
    cells = {center}
    dirs = random.sample(list(_neighbors4(0, 0)), random.randint(2, 4))
    for dr, dc in dirs:
        max_len = 0
        rr, cc = center
        while True:
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            max_len += 1
        if max_len == 0:
            continue
        length = random.randint(1, max_len)
        rr, cc = center
        for _ in range(length):
            rr += dr
            cc += dc
            if (rr, cc) != anchor:
                cells.add((rr, cc))
    cells.discard(anchor)
    return cells


def _shape_cells(h: int, w: int, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    for _ in range(12):
        mode = random.choice(SHAPE_MODES)
        if mode == "blob":
            cells = _blob_shape(h, w, anchor)
        elif mode == "rect":
            cells = _rect_shape(h, w, anchor)
        else:
            cells = _rays_shape(h, w, anchor)
        if cells:
            return cells
    choices = [(r, c) for r in range(h) for c in range(w) if (r, c) != anchor]
    return {random.choice(choices)}


def _random_interior(h: int, w: int) -> np.ndarray:
    interior = np.full((h, w), BG, dtype=int)
    anchor = _random_anchor(h, w)
    for r, c in _shape_cells(h, w, anchor):
        interior[r, c] = GREY
    interior[anchor] = YELLOW
    return interior


def _frame_patch(border_color: int, interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    patch[1:-1, 1:-1] = interior
    return patch


def _sample_dimensions(pair_count: int) -> tuple[int, int]:
    row_range, col_range = INTERIOR_RANGES[pair_count]
    return random.randint(*row_range), random.randint(*col_range)


def _expected_output(
    pair_data: list[dict[str, object]], placements: list[tuple[int, int]]
) -> np.ndarray:
    keyed: list[tuple[tuple[int, int], np.ndarray]] = []
    for pair, (r_a, c_a), (r_b, c_b) in zip(pair_data, placements[::2], placements[1::2]):
        key = (min(int(r_a), int(r_b)), min(int(c_a), int(c_b)))
        pattern = _compose_from_interiors(pair["interiors"])
        keyed.append((key, pattern))
    keyed.sort(key=lambda item: item[0])
    return _stack_patterns([pattern for _, pattern in keyed])


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    inner_h, inner_w = _sample_dimensions(pair_count)
    max_out_h = pair_count * (2 * inner_h - 1) + 2 * (pair_count - 1)
    max_out_w = 2 * inner_w - 1
    if max_out_h > 30 or max_out_w > 30:
        return None

    pair_colors = random.sample(FRAME_COLORS, pair_count)
    pair_data: list[dict[str, object]] = []
    patches: list[np.ndarray] = []
    for border_color in pair_colors:
        interiors = [_random_interior(inner_h, inner_w), _random_interior(inner_h, inner_w)]
        pair_data.append({"color": int(border_color), "interiors": interiors})
        for interior in interiors:
            patches.append(_frame_patch(int(border_color), interior))

    layout = _layout_positions(len(patches), inner_h + 2, inner_w + 2)
    if layout is None:
        return None
    rows, cols, placements = layout

    grid = np.full((rows, cols), BG, dtype=int)
    for patch, (r0, c0) in zip(patches, placements):
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch

    expected = _expected_output(pair_data, placements)
    if max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed1583749884/13/4")
