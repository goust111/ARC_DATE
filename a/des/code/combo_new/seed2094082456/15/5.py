from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_LIBRARY = [
    (),
    ("L",),
    ("R",),
    ("L", "L"),
    ("L", "R"),
    ("R", "L"),
    ("R", "R"),
    ("L", "L", "R"),
    ("L", "R", "L"),
    ("L", "R", "R"),
    ("R", "L", "L"),
    ("R", "L", "R"),
    ("R", "R", "L"),
]
PATTERN_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA]

DICT_ENTRY_H = 4
DICT_SYMBOL_W = 6
DICT_GAP_W = 1
DICT_PATTERN_W = 4
DICT_W = DICT_SYMBOL_W + DICT_GAP_W + DICT_PATTERN_W

GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 120
PLACE_ATTEMPTS = 200
PATTERN_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _turn_dir(direction: tuple[int, int], turn: str) -> tuple[int, int]:
    idx = CARDINALS.index(direction)
    if turn == "L":
        return CARDINALS[(idx - 1) % 4]
    assert turn == "R"
    return CARDINALS[(idx + 1) % 4]


def _sequence_from_symbol(mask: np.ndarray) -> tuple[str, ...]:
    shape = _tight_crop(mask.astype(bool))
    row_counts = shape.sum(axis=1)
    baseline_r = int(np.argmax(row_counts))
    baseline_cols = np.flatnonzero(shape[baseline_r])
    assert baseline_cols.size >= 2, "symbol baseline must have length >= 2"
    c0 = int(baseline_cols[0])
    c1 = int(baseline_cols[-1])
    assert baseline_cols.tolist() == list(range(c0, c1 + 1)), "symbol baseline must be contiguous"
    baseline_len = c1 - c0 + 1
    turn_count = baseline_len - 2
    turns: list[str] = []
    for idx in range(turn_count):
        col = c0 + 2 + idx
        up = baseline_r > 0 and bool(shape[baseline_r - 1, col])
        down = baseline_r + 1 < shape.shape[0] and bool(shape[baseline_r + 1, col])
        assert up ^ down, "each encoded turn needs exactly one stub"
        turns.append("L" if up else "R")
    expected_area = baseline_len + turn_count
    assert int(shape.sum()) == expected_area, "unexpected extra pixels in symbol glyph"
    return tuple(turns)


def _ordered_path(coords: list[tuple[int, int]], anchor_box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    cells = set(coords)
    neighbors = {cell: [nb for nb in _neighbors4(*cell) if nb in cells] for cell in cells}
    endpoints = [cell for cell, nbs in neighbors.items() if len(nbs) == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    for cell, nbs in neighbors.items():
        assert len(nbs) in (1, 2), "path must not branch"

    ar0, ac0, ar1, ac1 = anchor_box

    def dist_to_anchor(cell: tuple[int, int]) -> int:
        r, c = cell
        dr = 0 if ar0 <= r <= ar1 else min(abs(r - ar0), abs(r - ar1))
        dc = 0 if ac0 <= c <= ac1 else min(abs(c - ac0), abs(c - ac1))
        return dr + dc

    start = max(endpoints, key=dist_to_anchor)
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nb for nb in neighbors[cur] if nb != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must be unique"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    return ordered


def _turn_sequence_from_path(coords: list[tuple[int, int]], anchor_box: tuple[int, int, int, int]) -> tuple[str, ...]:
    ordered = _ordered_path(coords, anchor_box)
    dirs: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(ordered, ordered[1:], strict=False):
        dirs.append((r1 - r0, c1 - c0))
    assert dirs, "path must contain at least one step"

    segment_dirs = [dirs[0]]
    for d in dirs[1:]:
        if d != segment_dirs[-1]:
            segment_dirs.append(d)

    turns: list[str] = []
    for prev, cur in zip(segment_dirs, segment_dirs[1:], strict=False):
        cross = prev[0] * cur[1] - prev[1] * cur[0]
        assert cross in (-1, 1), "path turn must be a 90-degree turn"
        turns.append("L" if cross > 0 else "R")
    return tuple(turns)


def _dictionary_entries(grid: np.ndarray) -> dict[tuple[str, ...], np.ndarray]:
    h, w = grid.shape
    grey_cells = np.argwhere(grid == GREY)
    assert grey_cells.size > 0, "dictionary separators not found"
    dict_c0 = int(grey_cells[:, 1].min())
    dict_c1 = int(grey_cells[:, 1].max())

    dict_mask = (grid != BLACK) & (grid != AZURE) & (grid != ORANGE)
    content_rows = [r for r in range(h) if np.any(dict_mask[r, dict_c0 : dict_c1 + 1])]
    assert content_rows, "dictionary content not found"
    top = min(content_rows)
    bottom = max(content_rows)

    sep_rows = [
        r
        for r in range(top, bottom + 1)
        if np.all(grid[r, dict_c0 : dict_c1 + 1] == GREY)
    ]
    bounds = [top - 1, *sep_rows, bottom + 1]

    mapping: dict[tuple[str, ...], np.ndarray] = {}
    for a, b in zip(bounds, bounds[1:], strict=False):
        if b - a <= 1:
            continue
        sub = grid[a + 1 : b, dict_c0 : dict_c1 + 1]
        maroon_mask = sub == MAROON
        pattern_mask = ~np.isin(sub, [BLACK, GREY, MAROON])
        assert maroon_mask.any() and pattern_mask.any(), "each dictionary row needs a symbol and a pattern"

        seq = _sequence_from_symbol(maroon_mask)
        pr0, pc0, pr1, pc1 = _bbox(list(map(tuple, np.argwhere(pattern_mask))))
        pattern = sub[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        assert seq not in mapping, "dictionary symbols must be unique"
        mapping[seq] = pattern

    assert 2 <= len(mapping) <= 4, "dictionary must contain 2 to 4 entries"
    return mapping


def _orange_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = []
    for comp in _components_of_color(grid, ORANGE):
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "orange marker must be 2x2"
        assert len(comp) == 4, "orange marker must be solid"
        boxes.append((r0, c0, r1, c1))
    assert boxes, "at least one orange marker is required"
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each azure path's turn sequence to a dictionary symbol and stamp the linked pattern at the orange endpoint marker.

    Input structure:
    - The background is black.
    - A dictionary panel sits on the left, split by grey horizontal separator rows into 2 to 4 entries.
    - In each entry, a maroon turn glyph on the left encodes a left/right turn sequence, and a colored 3x3 or 4x4 pattern on the right is the value for that glyph.
    - On the right side, one or more azure 4-connected paths end next to distinct 2x2 orange marker blocks.

    Rule:
    - Read every dictionary entry by decoding its maroon glyph into an ordered sequence of left and right turns.
    - For each azure path, start from the endpoint farther from its orange marker, walk along the path, and record the left/right turn sequence created by its 90-degree direction changes.
    - Look up the matching dictionary entry and copy that entry's colored pattern onto the orange marker's top-left anchor, replacing the orange block while leaving the dictionary and the azure paths unchanged.

    Key insight:
    - The maroon symbols are not miniature copies of the full paths; they are canonical encodings of the abstract turn sequence, so the task is solved by comparing turning behavior rather than raw geometry.
    """
    g = np.asarray(grid, dtype=int)
    mapping = _dictionary_entries(g)
    orange_boxes = _orange_boxes(g)
    paths = _components_of_color(g, AZURE)
    assert paths, "at least one azure path is required"

    out = g.copy()
    used_boxes: set[int] = set()
    for comp in paths:
        endpoints = [cell for cell in comp if sum(nb in set(comp) for nb in _neighbors4(*cell)) == 1]
        assert len(endpoints) == 2

        def box_distance(box: tuple[int, int, int, int]) -> int:
            return min(
                abs(er - r) + abs(ec - c)
                for er, ec in endpoints
                for r in range(box[0], box[2] + 1)
                for c in range(box[1], box[3] + 1)
            )

        box_index = min(range(len(orange_boxes)), key=lambda idx: box_distance(orange_boxes[idx]))
        assert box_index not in used_boxes, "each orange marker must belong to exactly one path"
        box = orange_boxes[box_index]
        assert box_distance(box) == 1, "path endpoint must touch its orange marker"
        used_boxes.add(box_index)

        turns = _turn_sequence_from_path(comp, box)
        assert turns in mapping, "path turn sequence missing from dictionary"
        pattern = mapping[turns]
        r0, c0, _r1, _c1 = box
        ph, pw = pattern.shape
        assert r0 + ph <= out.shape[0] and c0 + pw <= out.shape[1], "pattern placement out of bounds"
        out[r0 : r0 + ph, c0 : c0 + pw] = pattern

    return out


def _sequence_key(arr: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return arr.shape[0], arr.shape[1], tuple(int(x) for x in arr.ravel())


def _random_pattern() -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        size = random.choice([3, 4])
        colors = random.sample(PATTERN_COLORS, random.randint(2, min(4, len(PATTERN_COLORS))))
        kind = random.choice(["checker", "vstripe", "hstripe", "diagonal", "frame", "quadrants"])
        pat = np.zeros((size, size), dtype=int)
        if kind == "checker":
            for r in range(size):
                for c in range(size):
                    pat[r, c] = colors[(r + c) % 2]
        elif kind == "vstripe":
            for c in range(size):
                pat[:, c] = colors[c % len(colors)]
        elif kind == "hstripe":
            for r in range(size):
                pat[r, :] = colors[r % len(colors)]
        elif kind == "diagonal":
            pat[:, :] = colors[0]
            for i in range(size):
                pat[i, i] = colors[1 % len(colors)]
            if len(colors) >= 3:
                for i in range(size):
                    pat[i, size - 1 - i] = colors[2]
        elif kind == "frame":
            pat[:, :] = colors[1 % len(colors)]
            pat[0, :] = colors[0]
            pat[-1, :] = colors[0]
            pat[:, 0] = colors[0]
            pat[:, -1] = colors[0]
            if len(colors) >= 3:
                pat[size // 2, size // 2] = colors[2]
        else:
            rmid = size // 2
            cmid = size // 2
            for r in range(size):
                for c in range(size):
                    idx = 0
                    if r >= rmid:
                        idx += 1
                    if c >= cmid:
                        idx += 2
                    pat[r, c] = colors[idx % len(colors)]
        if len(set(map(int, pat.ravel()))) >= 2:
            return pat
    raise RuntimeError("failed to generate a pattern")


def _draw_symbol(entry: np.ndarray, turns: tuple[str, ...]) -> None:
    base_r = 1
    base_c = 1
    base_len = len(turns) + 2
    entry[base_r, base_c : base_c + base_len] = MAROON
    for idx, turn in enumerate(turns):
        col = base_c + 2 + idx
        entry[base_r - 1 if turn == "L" else base_r + 1, col] = MAROON


def _build_dictionary(entries: list[tuple[tuple[str, ...], np.ndarray]], top: int) -> np.ndarray:
    height = len(entries) * DICT_ENTRY_H + (len(entries) - 1)
    canvas = np.full((top + height, DICT_W), BLACK, dtype=int)
    for idx, (turns, pattern) in enumerate(entries):
        r0 = top + idx * (DICT_ENTRY_H + 1)
        block = canvas[r0 : r0 + DICT_ENTRY_H, :]
        _draw_symbol(block, turns)
        ph, pw = pattern.shape
        pr = random.randint(0, DICT_ENTRY_H - ph)
        pc = DICT_SYMBOL_W + DICT_GAP_W + random.randint(0, DICT_PATTERN_W - pw)
        block[pr : pr + ph, pc : pc + pw] = pattern
        if idx + 1 < len(entries):
            canvas[r0 + DICT_ENTRY_H, :] = GREY
    return canvas


def _last_direction_for_sequence(turns: tuple[str, ...], start: tuple[int, int]) -> tuple[int, int]:
    direction = start
    for turn in turns:
        direction = _turn_dir(direction, turn)
    return direction


def _start_directions_for_last(turns: tuple[str, ...], last_dir: tuple[int, int]) -> list[tuple[int, int]]:
    return [d for d in CARDINALS if _last_direction_for_sequence(turns, d) == last_dir]


def _build_path_cells(
    turns: tuple[str, ...],
    start_dir: tuple[int, int],
    segment_lengths: list[int],
) -> list[tuple[int, int]] | None:
    cells = [(0, 0)]
    seen = {(0, 0)}
    cur_r, cur_c = 0, 0
    direction = start_dir
    for seg_idx, seg_len in enumerate(segment_lengths):
        dr, dc = direction
        for _ in range(seg_len):
            cur_r += dr
            cur_c += dc
            if (cur_r, cur_c) in seen:
                return None
            cells.append((cur_r, cur_c))
            seen.add((cur_r, cur_c))
        if seg_idx < len(turns):
            direction = _turn_dir(direction, turns[seg_idx])
    return cells


def _adjacent_to_block(cells: list[tuple[int, int]], block: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        if any(abs(r - br) + abs(c - bc) == 1 for br, bc in block):
            out.add((r, c))
    return out


def _make_path_object(turns: tuple[str, ...], pattern: np.ndarray) -> dict | None:
    ph, pw = pattern.shape
    for _ in range(PATH_ATTEMPTS):
        last_dir = random.choice([(0, 1), (1, 0)])
        start_options = _start_directions_for_last(turns, last_dir)
        start_dir = random.choice(start_options)
        segment_lengths = [random.randint(2, 5) for _ in range(len(turns) + 1)]
        if not turns:
            segment_lengths = [random.randint(4, 8)]
        path = _build_path_cells(turns, start_dir, segment_lengths)
        if path is None:
            continue

        end_r, end_c = path[-1]
        if last_dir == (0, 1):
            orange_tl = (end_r, end_c + 1)
        else:
            orange_tl = (end_r + 1, end_c)

        orange = {(orange_tl[0] + dr, orange_tl[1] + dc) for dr in range(2) for dc in range(2)}
        reserve = {(orange_tl[0] + dr, orange_tl[1] + dc) for dr in range(ph) for dc in range(pw)}
        if reserve & set(path):
            continue
        if _adjacent_to_block(path, orange) != {path[-1]}:
            continue

        footprint = set(path) | reserve
        r0, c0, r1, c1 = _bbox(footprint)
        shift_r = -r0
        shift_c = -c0

        path_shifted = [(r + shift_r, c + shift_c) for r, c in path]
        orange_tl_shifted = (orange_tl[0] + shift_r, orange_tl[1] + shift_c)
        shape = (r1 - r0 + 1, c1 - c0 + 1)
        return {
            "turns": turns,
            "pattern": pattern,
            "path": path_shifted,
            "orange_tl": orange_tl_shifted,
            "shape": shape,
        }
    return None


def _place_objects(
    objects: list[dict],
    height: int,
    width: int,
    min_col: int,
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int]] = []
    for obj in objects:
        oh, ow = obj["shape"]
        if oh + 2 > height or ow + 2 > width or min_col + ow + 1 > width:
            return None
        placed = None
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(1, height - oh - 1)
            c0 = random.randint(min_col, width - ow - 1)
            if occupied[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1].any():
                continue
            occupied[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1] = True
            placed = (r0, c0)
            break
        if placed is None:
            return None
        placements.append(placed)
    return placements


def _render_scene(
    dictionary: np.ndarray,
    placements: list[tuple[int, int]],
    objects: list[dict],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BLACK, dtype=int)
    out = np.full((height, width), BLACK, dtype=int)

    dh, dw = dictionary.shape
    inp[:dh, :dw] = dictionary
    out[:dh, :dw] = dictionary

    for (r0, c0), obj in zip(placements, objects, strict=False):
        for pr, pc in obj["path"]:
            inp[r0 + pr, c0 + pc] = AZURE
            out[r0 + pr, c0 + pc] = AZURE

        or0, oc0 = obj["orange_tl"]
        inp[r0 + or0 : r0 + or0 + 2, c0 + oc0 : c0 + oc0 + 2] = ORANGE
        pattern = obj["pattern"]
        ph, pw = pattern.shape
        out[r0 + or0 : r0 + or0 + ph, c0 + oc0 : c0 + oc0 + pw] = pattern

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    entry_count = random.randint(2, 4)
    used_sequences: set[tuple[str, ...]] = set()
    entries: list[tuple[tuple[str, ...], np.ndarray]] = []
    used_patterns: set[tuple[int, int, tuple[int, ...]]] = set()

    while len(entries) < entry_count:
        want_len = random.randint(0, 3)
        options = [seq for seq in TURN_LIBRARY if len(seq) == want_len and seq not in used_sequences]
        if not options:
            options = [seq for seq in TURN_LIBRARY if seq not in used_sequences]
        seq = random.choice(options)
        pattern = _random_pattern()
        key = _sequence_key(pattern)
        if key in used_patterns:
            continue
        used_sequences.add(seq)
        used_patterns.add(key)
        entries.append((seq, pattern))

    dict_top = random.randint(0, 3)
    dictionary = _build_dictionary(entries, dict_top)
    dh = dictionary.shape[0]

    path_count = random.randint(1, 3)
    entry_turns = [seq for seq, _pattern in entries]
    chosen_turns = random.sample(entry_turns, k=min(path_count, len(entry_turns)))
    while len(chosen_turns) < path_count:
        chosen_turns.append(random.choice(entry_turns))
    random.shuffle(chosen_turns)

    mapping = {seq: pattern for seq, pattern in entries}
    objects = []
    for seq in chosen_turns:
        obj = _make_path_object(seq, mapping[seq])
        if obj is None:
            return None
        objects.append(obj)

    max_obj_h = max(obj["shape"][0] for obj in objects)
    max_obj_w = max(obj["shape"][1] for obj in objects)
    min_col = DICT_W + random.randint(2, 4)
    min_h = max(dh + random.randint(0, 2), max_obj_h + 2, 12)
    min_w = max(min_col + max_obj_w + 1, 20)
    if min_h > 30 or min_w > 30:
        return None
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)

    placements = _place_objects(objects, height, width, min_col)
    if placements is None:
        return None

    inp, expected = _render_scene(dictionary, placements, objects, height, width)
    actual = main(inp)
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the turn-dictionary puzzle")
