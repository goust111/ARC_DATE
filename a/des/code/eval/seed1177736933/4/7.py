from __future__ import annotations

import numpy as np


GREY = 5
MAROON = 9
PALETTE = tuple(range(10))
NOTE_COUNT_RANGE = (3, 6)
SEGMENT_LEN_OPTIONS = (4, 5, 6)
STAFF_SPACING_OPTIONS = (2, 3, 4)
TOP_MARGIN_OPTIONS = (1, 2, 3, 4)
LINE_START_OPTIONS = (1, 2)
LINE_RIGHT_PAD_OPTIONS = (1, 2)
PATTERN_GAP_OPTIONS = (2, 3, 4)
PATTERN_BOTTOM_PAD_OPTIONS = (1, 2, 3)
MAX_GRID_H = 30
MAX_GRID_W = 38
MAX_STAGE_W = 16
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != bg and bool(np.all(grid[:, c] == grid[0, c]))
    ]
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    sep_col = int(candidates[0])
    return sep_col, int(grid[0, sep_col])


def _rotate_mask(mask: np.ndarray, segment: int) -> np.ndarray:
    if segment == 0:
        return np.rot90(mask, 1)
    if segment == 1:
        return mask.copy()
    if segment == 2:
        return np.rot90(mask, -1)
    raise AssertionError(f"invalid segment index: {segment}")


def _note_patch(master_mask: np.ndarray, segment: int, scale: int, color: int, bg: int) -> np.ndarray:
    rotated = _rotate_mask(master_mask, segment)
    scaled = np.repeat(np.repeat(rotated, scale, axis=0), scale, axis=1)
    patch = np.full(scaled.shape, bg, dtype=int)
    patch[scaled] = int(color)
    return patch


def _pack_geometry(shapes: list[tuple[int, int]], stage_w: int) -> tuple[list[tuple[int, int]], int, int]:
    placements: list[tuple[int, int]] = []
    x = 0
    y = 0
    row_h = 0
    row_count = 0
    for h, w in shapes:
        assert w <= stage_w
        if x > 0 and x + w > stage_w:
            y += row_h + 1
            x = 0
            row_h = 0
        if row_h == 0:
            row_count += 1
        placements.append((y, x))
        x += w + 1
        row_h = max(row_h, h)
    used_h = y + row_h
    return placements, used_h, row_count


def _render_stage(patches: list[np.ndarray], stage_h: int, stage_w: int, bg: int) -> np.ndarray:
    stage = np.full((stage_h, stage_w), bg, dtype=int)
    shapes = [tuple(map(int, patch.shape)) for patch in patches]
    placements, used_h, _row_count = _pack_geometry(shapes, stage_w)
    assert used_h <= stage_h, "stage is not tall enough for packed output"
    for patch, (r0, c0) in zip(patches, placements):
        h, w = patch.shape
        stage[r0 : r0 + h, c0 : c0 + w] = patch
    return stage


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    sep_col, staff_color = _find_separator(g, bg)
    assert 0 < sep_col < g.shape[1] - 1

    left = g[:, :sep_col]
    stage = g[:, sep_col + 1 :]
    assert stage.size > 0
    assert bool(np.all(stage == bg)), "stage area must start blank"

    staff_rows = [r for r in range(left.shape[0]) if bool(np.any(left[r] == staff_color))]
    assert len(staff_rows) == 3, "expected three staff rows"
    spacing = int(staff_rows[1] - staff_rows[0])
    assert spacing > 0 and staff_rows[2] - staff_rows[1] == spacing, "staff rows must be equally spaced"

    spans: list[tuple[int, int]] = []
    for r in staff_rows:
        cols = np.flatnonzero(left[r] != bg)
        assert cols.size > 0
        c0 = int(cols[0])
        c1 = int(cols[-1])
        assert c1 - c0 + 1 == int(cols.size), "each staff row must form one contiguous segment"
        spans.append((c0, c1))
    assert spans[0] == spans[1] == spans[2], "staff rows must share a common span"
    line_start, line_end = spans[0]
    line_len = int(line_end - line_start + 1)
    assert line_len % 3 == 0
    segment_len = line_len // 3

    notes: list[dict[str, int]] = []
    for line_idx, r in enumerate(staff_rows):
        for c in range(line_start, line_end + 1):
            color = int(left[r, c])
            if color in {bg, staff_color}:
                continue
            segment = int((c - line_start) // segment_len)
            assert 0 <= segment < 3
            notes.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "color": color,
                    "scale": int(line_idx + 1),
                    "segment": segment,
                }
            )
    assert notes, "expected at least one note"
    note_colors = [note["color"] for note in notes]
    assert len(set(note_colors)) == len(note_colors), "note colors must be unique"

    below_mask = np.zeros_like(left, dtype=bool)
    below_mask[staff_rows[-1] + 1 :, :] = True
    master_coords = np.argwhere(below_mask & (left != bg) & (left != staff_color))
    assert master_coords.size > 0, "expected a master pattern below the staff"
    master_colors = {int(left[r, c]) for r, c in master_coords}
    assert len(master_colors) == 1, "master pattern must have one color"
    master_color = int(next(iter(master_colors)))

    pr0, pc0 = map(int, master_coords.min(axis=0))
    pr1, pc1 = map(int, master_coords.max(axis=0))
    master_patch = left[pr0 : pr1 + 1, pc0 : pc1 + 1]
    h, w = master_patch.shape
    assert h == w and h in {2, 3}, "master pattern bbox must be 2x2 or 3x3"
    assert set(map(int, np.unique(master_patch))) <= {bg, master_color}
    master_mask = master_patch == master_color

    return {
        "bg": int(bg),
        "sep_col": int(sep_col),
        "staff_color": int(staff_color),
        "staff_rows": [int(r) for r in staff_rows],
        "line_start": int(line_start),
        "segment_len": int(segment_len),
        "master_mask": np.asarray(master_mask, dtype=bool),
        "notes": notes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read note color, staff line, and third-segment position to recolor, scale,
    and rotate the master pattern, then pack the results into the blank stage.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_instance(g)

    bg = int(parsed["bg"])
    sep_col = int(parsed["sep_col"])
    master_mask = np.asarray(parsed["master_mask"], dtype=bool)
    stage_w = int(g.shape[1] - sep_col - 1)
    stage_h = int(g.shape[0])

    ordered_notes = sorted(parsed["notes"], key=lambda note: (int(note["col"]), int(note["row"])))
    patches = [
        _note_patch(
            master_mask=master_mask,
            segment=int(note["segment"]),
            scale=int(note["scale"]),
            color=int(note["color"]),
            bg=bg,
        )
        for note in ordered_notes
    ]
    stage = _render_stage(patches, stage_h=stage_h, stage_w=stage_w, bg=bg)

    out = g.copy()
    out[:, sep_col + 1 :] = stage
    return out


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRS4:
            rr = r + dr
            cc = c + dc
            nxt = (rr, cc)
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _is_asymmetric(mask: np.ndarray) -> bool:
    for k in range(1, 4):
        if np.array_equal(mask, np.rot90(mask, k)):
            return False
    for k in range(4):
        if np.array_equal(mask, np.fliplr(np.rot90(mask, k))):
            return False
        if np.array_equal(mask, np.flipud(np.rot90(mask, k))):
            return False
    return True


def _random_master_mask() -> np.ndarray:
    for _ in range(200):
        size = int(_RNG.choice((2, 3)))
        area = int(_RNG.integers(size + 1, size * size))
        flat = np.zeros(size * size, dtype=bool)
        chosen = _RNG.choice(size * size, size=area, replace=False)
        flat[np.asarray(chosen, dtype=int)] = True
        mask = flat.reshape(size, size)
        if bool(mask.all()) or not bool(mask.any()):
            continue
        if not bool(np.all(mask.any(axis=0))) or not bool(np.all(mask.any(axis=1))):
            continue
        if not _is_connected(mask):
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric master pattern")


def _inner_segment_columns(line_start: int, segment_len: int, segment: int) -> list[int]:
    seg_start = int(line_start + segment * segment_len)
    return list(range(seg_start + 1, seg_start + segment_len - 1))


def _build_note_specs(
    staff_rows: list[int],
    line_start: int,
    segment_len: int,
    bg: int,
) -> list[dict[str, int]]:
    total_notes = int(_RNG.integers(NOTE_COUNT_RANGE[0], NOTE_COUNT_RANGE[1]))
    perm = list(map(int, _RNG.permutation(3)))

    slots: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for line_idx, segment in enumerate(perm):
        row = int(staff_rows[line_idx])
        col_choices = _inner_segment_columns(line_start, segment_len, segment)
        col = int(_RNG.choice(col_choices))
        slot = (row, col)
        used.add(slot)
        slots.append(slot)

    remaining: list[tuple[int, int]] = []
    for row in staff_rows:
        for segment in range(3):
            for col in _inner_segment_columns(line_start, segment_len, segment):
                slot = (int(row), int(col))
                if slot not in used:
                    remaining.append(slot)
    _RNG.shuffle(remaining)
    extras_needed = int(total_notes - len(slots))
    slots.extend(remaining[:extras_needed])

    note_palette = [color for color in PALETTE if color not in {bg, GREY, MAROON}]
    colors = _RNG.choice(note_palette, size=len(slots), replace=False)
    row_to_scale = {int(row): idx + 1 for idx, row in enumerate(staff_rows)}

    notes: list[dict[str, int]] = []
    for (row, col), color in zip(slots, colors):
        segment = int((col - line_start) // segment_len)
        notes.append(
            {
                "row": int(row),
                "col": int(col),
                "color": int(color),
                "scale": int(row_to_scale[int(row)]),
                "segment": segment,
            }
        )
    return notes


def _render_input(
    height: int,
    sep_col: int,
    bg: int,
    staff_rows: list[int],
    line_start: int,
    segment_len: int,
    notes: list[dict[str, int]],
    master_mask: np.ndarray,
    master_top: int,
    master_left: int,
) -> np.ndarray:
    stage_w = MAX_GRID_W - sep_col - 1
    width = sep_col + 1 + stage_w
    grid = np.full((height, width), bg, dtype=int)
    grid[:, sep_col] = GREY

    line_len = int(segment_len * 3)
    for row in staff_rows:
        grid[row, line_start : line_start + line_len] = GREY

    for note in notes:
        grid[int(note["row"]), int(note["col"])] = int(note["color"])

    mh, mw = master_mask.shape
    patch = grid[master_top : master_top + mh, master_left : master_left + mw]
    patch[master_mask] = MAROON
    return grid


def _render_expected(
    grid: np.ndarray,
    master_mask: np.ndarray,
    notes: list[dict[str, int]],
    sep_col: int,
    bg: int,
    stage_w: int,
) -> np.ndarray:
    ordered = sorted(notes, key=lambda note: (int(note["col"]), int(note["row"])))
    patches = [
        _note_patch(master_mask, int(note["segment"]), int(note["scale"]), int(note["color"]), bg)
        for note in ordered
    ]
    stage = _render_stage(patches, stage_h=int(grid.shape[0]), stage_w=stage_w, bg=bg)
    out = grid.copy()
    out[:, sep_col + 1 : sep_col + 1 + stage_w] = stage
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = int(_RNG.choice([color for color in PALETTE if color not in {GREY, MAROON}]))
    segment_len = int(_RNG.choice(SEGMENT_LEN_OPTIONS))
    line_start = int(_RNG.choice(LINE_START_OPTIONS))
    line_right_pad = int(_RNG.choice(LINE_RIGHT_PAD_OPTIONS))
    line_len = int(segment_len * 3)
    sep_col = int(line_start + line_len + line_right_pad)

    if sep_col + 2 >= MAX_GRID_W:
        return None

    staff_top = int(_RNG.choice(TOP_MARGIN_OPTIONS))
    staff_spacing = int(_RNG.choice(STAFF_SPACING_OPTIONS))
    staff_rows = [staff_top + i * staff_spacing for i in range(3)]

    master_mask = _random_master_mask()
    master_gap = int(_RNG.choice(PATTERN_GAP_OPTIONS))
    master_top = int(staff_rows[-1] + master_gap)
    master_left_max = sep_col - master_mask.shape[1] - 1
    if master_left_max < 1:
        return None
    master_left = int(_RNG.integers(1, master_left_max + 1))

    bottom_pad = int(_RNG.choice(PATTERN_BOTTOM_PAD_OPTIONS))
    left_required_h = int(master_top + master_mask.shape[0] + bottom_pad)
    if left_required_h > MAX_GRID_H:
        return None

    notes = _build_note_specs(staff_rows, line_start, segment_len, bg)
    ordered = sorted(notes, key=lambda note: (int(note["col"]), int(note["row"])))
    patches = [
        _note_patch(master_mask, int(note["segment"]), int(note["scale"]), int(note["color"]), bg)
        for note in ordered
    ]

    max_patch_w = max(int(patch.shape[1]) for patch in patches)
    nowrap_w = sum(int(patch.shape[1]) for patch in patches) + len(patches) - 1
    max_stage_w = min(MAX_STAGE_W, MAX_GRID_W - sep_col - 1, nowrap_w)
    if max_patch_w > max_stage_w:
        return None

    wrap_candidates: list[tuple[int, int]] = []
    nowrap_candidates: list[tuple[int, int]] = []
    all_candidates: list[tuple[int, int]] = []
    for stage_w in range(max_patch_w, max_stage_w + 1):
        _placements, used_h, row_count = _pack_geometry([tuple(map(int, p.shape)) for p in patches], stage_w)
        if used_h > MAX_GRID_H:
            continue
        candidate = (int(stage_w), int(used_h))
        all_candidates.append(candidate)
        if row_count == 1:
            nowrap_candidates.append(candidate)
        else:
            wrap_candidates.append(candidate)
    if not all_candidates:
        return None

    if wrap_candidates and nowrap_candidates:
        candidate_pool = wrap_candidates if bool(_RNG.integers(0, 2)) else nowrap_candidates
    else:
        candidate_pool = all_candidates
    stage_w, used_h = candidate_pool[int(_RNG.integers(len(candidate_pool)))]

    min_height = max(left_required_h, used_h)
    if min_height > MAX_GRID_H:
        return None
    extra_bottom = int(_RNG.integers(0, min(4, MAX_GRID_H - min_height + 1)))
    height = int(min_height + extra_bottom)

    grid = _render_input(
        height=height,
        sep_col=sep_col,
        bg=bg,
        staff_rows=staff_rows,
        line_start=line_start,
        segment_len=segment_len,
        notes=notes,
        master_mask=master_mask,
        master_top=master_top,
        master_left=master_left,
    )
    grid = grid[:, : sep_col + 1 + stage_w]
    expected = _render_expected(grid, master_mask, notes, sep_col, bg, stage_w)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
