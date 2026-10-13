import random

import numpy as np


PATTERN_SIDE = 9
COARSE_SIDE = 3
BLOCK_SIDE = 3
SLOT_GAP = 1
GEN_ATTEMPTS = 300
PALETTE = tuple(range(10))
MARKER_COUNT_CHOICES = (1, 2, 3, 4, 5)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(cells, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coarse_code(patch: np.ndarray, marker: int) -> np.ndarray:
    code = np.zeros((COARSE_SIDE, COARSE_SIDE), dtype=bool)
    for br in range(COARSE_SIDE):
        for bc in range(COARSE_SIDE):
            block = patch[
                br * BLOCK_SIDE : (br + 1) * BLOCK_SIDE,
                bc * BLOCK_SIDE : (bc + 1) * BLOCK_SIDE,
            ]
            code[br, bc] = bool(np.any(block == marker))
    return code


def _code_sprite(code: np.ndarray, fill: int, marker: int) -> np.ndarray:
    return np.where(code, marker, fill).astype(int)


def _expand_self_similar(code: np.ndarray, fill: int, marker: int) -> np.ndarray:
    micro = _code_sprite(code, fill, marker)
    patch = np.full((PATTERN_SIDE, PATTERN_SIDE), fill, dtype=int)
    for br in range(COARSE_SIDE):
        for bc in range(COARSE_SIDE):
            r0 = br * BLOCK_SIDE
            c0 = bc * BLOCK_SIDE
            if code[br, bc]:
                patch[r0 : r0 + BLOCK_SIDE, c0 : c0 + BLOCK_SIDE] = micro
            else:
                patch[r0 : r0 + BLOCK_SIDE, c0 : c0 + BLOCK_SIDE] = fill
    return patch


def _self_similar_signature(
    patch: np.ndarray,
) -> tuple[np.ndarray, int, int] | None:
    patch = np.asarray(patch, dtype=int)
    if patch.shape != (PATTERN_SIDE, PATTERN_SIDE):
        return None

    colors, counts = np.unique(patch, return_counts=True)
    if len(colors) != 2:
        return None

    fill = int(colors[np.argmax(counts)])
    marker = int(colors[np.argmin(counts)])
    if int(np.max(counts)) <= int(np.min(counts)):
        return None

    code = _coarse_code(patch, marker)
    micro = _code_sprite(code, fill, marker)
    for br in range(COARSE_SIDE):
        for bc in range(COARSE_SIDE):
            block = patch[
                br * BLOCK_SIDE : (br + 1) * BLOCK_SIDE,
                bc * BLOCK_SIDE : (bc + 1) * BLOCK_SIDE,
            ]
            if code[br, bc]:
                if not np.array_equal(block, micro):
                    return None
            elif not np.all(block == fill):
                return None
    return code, fill, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique 9x9 bicolor pattern whose marked blocks recursively encode its own 3x3 coarse code.

    Input structure:
    - A dominant background surrounds several separated 9x9 two-color patterns.
    - Each pattern is fully filled by one dominant fill color plus a sparser marker color.
    - Every pattern can be partitioned into a 3x3 array of 3x3 coarse blocks.

    Rule:
    - For each 9x9 pattern, compress it to a 3x3 coarse code by marking every coarse block that contains any marker pixel.
    - A pattern is valid only when every marked coarse block contains the full 3x3 coarse code again, while every unmarked coarse block is pure fill.
    - Exactly one pattern is valid; output that pattern's 3x3 coarse code using its own fill and marker colors.

    Key insight:
    - There is no external key: the correct object's coarse code is the same template repeated inside each of its marked blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    matches: list[tuple[np.ndarray, int, int]] = []
    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape != (PATTERN_SIDE, PATTERN_SIDE):
            continue
        if np.any(patch == bg):
            continue
        sig = _self_similar_signature(patch)
        if sig is not None:
            matches.append(sig)

    assert len(matches) == 1, "expected exactly one self-similar pattern"
    code, fill, marker = matches[0]
    return _code_sprite(code, fill, marker)


def _random_code() -> np.ndarray:
    count = random.choice(MARKER_COUNT_CHOICES)
    code = np.zeros((COARSE_SIDE, COARSE_SIDE), dtype=bool)
    for idx in random.sample(range(COARSE_SIDE * COARSE_SIDE), count):
        code[idx // COARSE_SIDE, idx % COARSE_SIDE] = True
    return code


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _make_decoy(fill: int, marker: int) -> np.ndarray | None:
    for _ in range(80):
        code = _random_code()
        base = _expand_self_similar(code, fill, marker)
        marked_blocks = [tuple(map(int, rc)) for rc in np.argwhere(code)]
        fill_blocks = [tuple(map(int, rc)) for rc in np.argwhere(~code)]

        modes = []
        if marked_blocks:
            modes.append("wrong_marked_block")
        if fill_blocks:
            modes.append("marker_in_fill_block")
        if marked_blocks and int(code.sum()) > 1:
            modes.append("missing_marker")
        random.shuffle(modes)

        for mode in modes:
            patch = base.copy()
            if mode == "wrong_marked_block":
                br, bc = random.choice(marked_blocks)
                alt = _random_code()
                while np.array_equal(alt, code):
                    alt = _random_code()
                patch[
                    br * BLOCK_SIDE : (br + 1) * BLOCK_SIDE,
                    bc * BLOCK_SIDE : (bc + 1) * BLOCK_SIDE,
                ] = _code_sprite(alt, fill, marker)
            elif mode == "marker_in_fill_block":
                br, bc = random.choice(fill_blocks)
                block = np.full((BLOCK_SIDE, BLOCK_SIDE), fill, dtype=int)
                marker_cells = random.sample(range(BLOCK_SIDE * BLOCK_SIDE), random.randint(1, 3))
                for idx in marker_cells:
                    block[idx // BLOCK_SIDE, idx % BLOCK_SIDE] = marker
                patch[
                    br * BLOCK_SIDE : (br + 1) * BLOCK_SIDE,
                    bc * BLOCK_SIDE : (bc + 1) * BLOCK_SIDE,
                ] = block
            else:
                br, bc = random.choice(marked_blocks)
                block = patch[
                    br * BLOCK_SIDE : (br + 1) * BLOCK_SIDE,
                    bc * BLOCK_SIDE : (bc + 1) * BLOCK_SIDE,
                ].copy()
                marker_cells = [tuple(map(int, rc)) for rc in np.argwhere(block == marker)]
                if len(marker_cells) <= 1:
                    continue
                rr, cc = random.choice(marker_cells)
                block[rr, cc] = fill
                patch[
                    br * BLOCK_SIDE : (br + 1) * BLOCK_SIDE,
                    bc * BLOCK_SIDE : (bc + 1) * BLOCK_SIDE,
                ] = block

            if _self_similar_signature(patch) is None:
                return patch
    return None


def _choose_layout(
    pattern_count: int,
) -> tuple[int, int, list[int], list[int], list[tuple[int, int]]] | None:
    feasible_slots = [
        (rows, cols)
        for rows in range(1, 4)
        for cols in range(1, 4)
        if rows * cols >= pattern_count
    ]
    if not feasible_slots:
        return None

    slot_rows, slot_cols = random.choice(feasible_slots)
    base_h = slot_rows * PATTERN_SIDE + (slot_rows - 1) * SLOT_GAP
    base_w = slot_cols * PATTERN_SIDE + (slot_cols - 1) * SLOT_GAP
    max_extra_h = 30 - base_h
    max_extra_w = 30 - base_w
    if max_extra_h < 0 or max_extra_w < 0:
        return None

    for _ in range(60):
        extra_h = random.randint(0, max_extra_h)
        extra_w = random.randint(0, max_extra_w)
        height = base_h + extra_h
        width = base_w + extra_w
        if height * width - pattern_count * (PATTERN_SIDE * PATTERN_SIDE) <= 80:
            continue
        row_slack = _random_composition(extra_h, slot_rows + 1)
        col_slack = _random_composition(extra_w, slot_cols + 1)
        row_starts = []
        pos = row_slack[0]
        for ridx in range(slot_rows):
            row_starts.append(pos)
            pos += PATTERN_SIDE
            if ridx < slot_rows - 1:
                pos += SLOT_GAP + row_slack[ridx + 1]
        col_starts = []
        pos = col_slack[0]
        for cidx in range(slot_cols):
            col_starts.append(pos)
            pos += PATTERN_SIDE
            if cidx < slot_cols - 1:
                pos += SLOT_GAP + col_slack[cidx + 1]

        slots = [(r, c) for r in range(slot_rows) for c in range(slot_cols)]
        chosen_slots = random.sample(slots, pattern_count)
        return height, width, row_starts, col_starts, chosen_slots

    return None


def _render_patterns(
    height: int,
    width: int,
    bg: int,
    row_starts: list[int],
    col_starts: list[int],
    slots: list[tuple[int, int]],
    patches: list[np.ndarray],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for patch, (slot_r, slot_c) in zip(patches, slots):
        r0 = row_starts[slot_r]
        c0 = col_starts[slot_c]
        grid[r0 : r0 + PATTERN_SIDE, c0 : c0 + PATTERN_SIDE] = patch
    return grid


def _gen_attempt() -> np.ndarray | None:
    pattern_count = random.randint(2, 4)
    layout = _choose_layout(pattern_count)
    if layout is None:
        return None
    height, width, row_starts, col_starts, slots = layout

    bg = random.choice(PALETTE)
    colors = [color for color in PALETTE if color != bg]
    random.shuffle(colors)
    if len(colors) < 2 * pattern_count:
        return None

    pairs = [(colors[2 * i], colors[2 * i + 1]) for i in range(pattern_count)]
    target_index = random.randrange(pattern_count)
    target_code = _random_code()
    target_fill, target_marker = pairs[target_index]

    patches: list[np.ndarray] = []
    for idx, (fill, marker) in enumerate(pairs):
        if idx == target_index:
            patches.append(_expand_self_similar(target_code, fill, marker))
        else:
            patch = _make_decoy(fill, marker)
            if patch is None:
                return None
            patches.append(patch)

    random.shuffle(slots)
    grid = _render_patterns(height, width, bg, row_starts, col_starts, slots, patches)
    expected = _code_sprite(target_code, target_fill, target_marker)

    if _get_bg(grid) != bg:
        return None
    if len(_components(grid != bg)) != pattern_count:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
