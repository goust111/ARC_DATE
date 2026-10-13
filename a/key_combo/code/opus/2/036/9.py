from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg, in_bounds


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BG = Color.BLACK
FG = Color.BLUE
RAY = Color.RED

GRID_ROWS_RANGE = (13, 19)
GRID_COLS_RANGE = (13, 19)
BODY_SIDE_RANGE = (6, 9)
BODY_MARGIN = 2
TAB_COUNT_RANGE = (2, 4)
NOTCH_COUNT_RANGE = (2, 4)
GEN_RETRY_LIMIT = 300


def _runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(line):
        if bool(value) and start is None:
            start = i
        if start is not None and (not bool(value) or i == len(line) - 1):
            end = i - 1 if not bool(value) else i
            runs.append((start, end))
            start = None
    return runs


def _body_thickness(mask: np.ndarray) -> int:
    lengths: list[int] = []
    for r in range(mask.shape[0]):
        lengths.extend(end - start + 1 for start, end in _runs(mask[r]) if end - start + 1 >= 2)
    for c in range(mask.shape[1]):
        lengths.extend(end - start + 1 for start, end in _runs(mask[:, c]) if end - start + 1 >= 2)
    assert lengths, "foreground object must contain nontrivial row/column runs"
    counts = Counter(lengths)
    max_count = max(counts.values())
    return min(length for length, count in counts.items() if count == max_count)


def _fg_neighbors4(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr, dc in DIRS4
        if in_bounds(mask.shape, r + dr, c + dc) and bool(mask[r + dr, c + dc])
    ]


def _n8(mask: np.ndarray, r: int, c: int) -> int:
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if in_bounds(mask.shape, r + dr, c + dc) and bool(mask[r + dr, c + dc]):
                count += 1
    return count


def _ray_len(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    length = 0
    rr, cc = r, c
    while in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
        length += 1
        rr += dr
        cc += dc
    return length


def _opposite_edge(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int, int]:
    if dr:
        rows = np.flatnonzero(mask[:, c])
        assert rows.size, "tab column must contain foreground"
        return int(rows.max() if dr > 0 else rows.min()), c
    cols = np.flatnonzero(mask[r])
    assert cols.size, "tab row must contain foreground"
    return r, int(cols.max() if dc > 0 else cols.min())


def _outside_opposite_edge(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int, int] | None:
    if dr:
        rows = np.flatnonzero(mask[:, c])
        assert rows.size, "notch column must contain foreground"
        rr = int(rows.max() + 1 if dr > 0 else rows.min() - 1)
        return (rr, c) if in_bounds(mask.shape, rr, c) else None
    cols = np.flatnonzero(mask[r])
    assert cols.size, "notch row must contain foreground"
    cc = int(cols.max() + 1 if dc > 0 else cols.min() - 1)
    return (r, cc) if in_bounds(mask.shape, r, cc) else None


def _tab_sources(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    thickness = _body_thickness(mask)
    long_ray_threshold = 2 * thickness - 1
    tabs: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]):
                continue

            neighbors = _fg_neighbors4(mask, r, c)
            if len(neighbors) != 1:
                continue

            dr, dc = neighbors[0]
            length = _ray_len(mask, r, c, dr, dc)
            has_side_support = _n8(mask, r, c) >= 3
            crosses_long_body = length > long_ray_threshold
            if length >= 3 and (has_side_support or crosses_long_body):
                target = _opposite_edge(mask, r, c, dr, dc)
                if target != (r, c):
                    tabs.append(((r, c), (dr, dc), target))

    return tabs


def _notch_sources(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    tab_cells = {cell for cell, _direction, _target in _tab_sources(mask)}
    notches: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if bool(mask[r, c]):
                continue

            fg_dirs: list[tuple[int, int]] = []
            bg_dirs: list[tuple[int, int]] = []
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
                    fg_dirs.append((dr, dc))
                else:
                    bg_dirs.append((dr, dc))

            if len(fg_dirs) != 3 or len(bg_dirs) != 1:
                continue

            outside_dr, outside_dc = bg_dirs[0]
            dr, dc = -outside_dr, -outside_dc
            side_a = (r - dc, c + dr)
            side_b = (r + dc, c - dr)
            if side_a in tab_cells or side_b in tab_cells:
                continue

            target = _outside_opposite_edge(mask, r, c, dr, dc)
            if target is not None and not bool(mask[target]):
                notches.append(((r, c), (dr, dc), target))

    return notches


def _paint_rays(base: np.ndarray, ray_sources: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    out = base.copy()
    for (r, c), (dr, dc) in ray_sources:
        rr, cc = r + dr, c + dc
        while in_bounds(out.shape, rr, cc):
            out[rr, cc] = RAY
            rr += dr
            cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete opposite tabs/notches and emit red outward rays only from the newly added tabs.

    Input structure
    - The grid has a black background and one blue 4-connected thick rectilinear foreground shape.
    - The blue boundary contains one-pixel tabs and one-pixel notches.
    - A tab is a blue boundary pixel with one 4-neighbor, and a notch is a black boundary gap with three blue 4-neighbors.

    Rule
    - For every tab, project across its full row or column span and remove the opposite edge pixel to create the matching notch.
    - For every notch, project across the same span and add one blue pixel just outside the opposite edge to create the matching tab, ignoring notch-like gaps immediately beside a tab source.
    - Keep the completed blue shape, then start a red ray one cell beyond each newly added tab and extend it outward in the tab's protrusion direction until the canvas boundary.

    Key insight
    - The ray sources are not all visible tabs: only the tabs created during the symmetry completion step emit rays, so the solver must first track which blue pixels were newly added.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert bg == BG, "expected black background"
    assert fg_colors == [FG], "expected one blue foreground shape"

    mask = g == FG
    out = g.copy()

    for _cell, _direction, target in _tab_sources(mask):
        out[target] = BG

    new_tabs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for _cell, direction, target in _notch_sources(mask):
        out[target] = FG
        new_tabs.append((target, direction))

    return _paint_rays(out, new_tabs)


def _slot_from_row(r: int, c0: int, c1: int) -> dict[str, object]:
    return {
        "axis": "row",
        "neg_boundary": (r, c0),
        "neg_out": (r, c0 - 1),
        "neg_dir": (0, -1),
        "pos_boundary": (r, c1),
        "pos_out": (r, c1 + 1),
        "pos_dir": (0, 1),
    }


def _slot_from_col(c: int, r0: int, r1: int) -> dict[str, object]:
    return {
        "axis": "col",
        "neg_boundary": (r0, c),
        "neg_out": (r0 - 1, c),
        "neg_dir": (-1, 0),
        "pos_boundary": (r1, c),
        "pos_out": (r1 + 1, c),
        "pos_dir": (1, 0),
    }


def _center_biased_start(length: int, span: int) -> int:
    lo = BODY_MARGIN
    hi = length - span - BODY_MARGIN
    positions = list(range(lo, hi + 1))
    center = (length - span) // 2
    weights = [max(1, 6 - abs(pos - center)) for pos in positions]
    return random.choices(positions, weights=weights, k=1)[0]


def _gen_clean_mask() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    body_h = random.randint(BODY_SIDE_RANGE[0], min(BODY_SIDE_RANGE[1], rows - 2 * BODY_MARGIN))
    body_w = random.randint(BODY_SIDE_RANGE[0], min(BODY_SIDE_RANGE[1], cols - 2 * BODY_MARGIN))
    r0 = _center_biased_start(rows, body_h)
    c0 = _center_biased_start(cols, body_w)
    r1 = r0 + body_h - 1
    c1 = c0 + body_w - 1

    mask = np.zeros((rows, cols), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask, (r0, c0, r1, c1)


def _choose_slot_mix(row_slots: list[dict[str, object]], col_slots: list[dict[str, object]], total: int) -> list[dict[str, object]]:
    if total >= 2:
        row_low = max(1, total - len(col_slots))
        row_high = min(len(row_slots), total - 1)
        if row_low <= row_high:
            row_count = random.randint(row_low, row_high)
            col_count = total - row_count
            chosen = random.sample(row_slots, row_count) + random.sample(col_slots, col_count)
            random.shuffle(chosen)
            return chosen
    all_slots = row_slots + col_slots
    return random.sample(all_slots, total)


def _slot_roles(
    slot: dict[str, object], source_side: str
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    if source_side == "neg":
        return (
            slot["neg_boundary"],
            slot["neg_out"],
            slot["pos_boundary"],
            slot["pos_out"],
            slot["pos_dir"],
        )
    return (
        slot["pos_boundary"],
        slot["pos_out"],
        slot["neg_boundary"],
        slot["neg_out"],
        slot["neg_dir"],
    )


def _render_masks(
    clean: np.ndarray, chosen_specs: list[tuple[dict[str, object], str, str]]
) -> tuple[np.ndarray, np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    out_mask = clean.copy()
    ray_sources: list[tuple[tuple[int, int], tuple[int, int]]] = []
    expected_tab_cells: set[tuple[int, int]] = set()
    expected_notch_cells: set[tuple[int, int]] = set()

    for slot, source_type, source_side in chosen_specs:
        src_boundary, src_tab, opposite_boundary, opposite_tab, ray_dir = _slot_roles(slot, source_side)
        if source_type == "tab":
            out_mask[src_tab] = True
            out_mask[opposite_boundary] = False
            expected_tab_cells.add(src_tab)
        else:
            out_mask[src_boundary] = False
            out_mask[opposite_tab] = True
            expected_notch_cells.add(src_boundary)
            ray_sources.append((opposite_tab, ray_dir))

    inp_mask = out_mask.copy()
    for slot, source_type, source_side in chosen_specs:
        _src_boundary, _src_tab, opposite_boundary, opposite_tab, _ray_dir = _slot_roles(slot, source_side)
        if source_type == "tab":
            inp_mask[opposite_boundary] = True
        else:
            inp_mask[opposite_tab] = False

    return inp_mask, out_mask, ray_sources, expected_tab_cells, expected_notch_cells


def _mask_to_grid(mask: np.ndarray) -> np.ndarray:
    grid = np.full(mask.shape, BG, dtype=int)
    grid[mask] = FG
    return grid


def _gen_expected_output(out_mask: np.ndarray, ray_sources: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    return _paint_rays(_mask_to_grid(out_mask), ray_sources)


def _gen_attempt() -> np.ndarray | None:
    clean, (r0, c0, r1, c1) = _gen_clean_mask()
    row_slots = [_slot_from_row(r, c0, c1) for r in range(r0 + 1, r1)]
    col_slots = [_slot_from_col(c, r0, r1) for c in range(c0 + 1, c1)]

    tab_count = random.randint(*TAB_COUNT_RANGE)
    notch_count = random.randint(*NOTCH_COUNT_RANGE)
    total = tab_count + notch_count
    if total > len(row_slots) + len(col_slots):
        return None

    chosen_slots = _choose_slot_mix(row_slots, col_slots, total)
    types = ["tab"] * tab_count + ["notch"] * notch_count
    random.shuffle(types)
    chosen_specs = [
        (slot, source_type, random.choice(("neg", "pos")))
        for slot, source_type in zip(chosen_slots, types)
    ]

    inp_mask, out_mask, ray_sources, expected_tab_cells, expected_notch_cells = _render_masks(clean, chosen_specs)
    if len(ray_sources) < 2:
        return None

    inp = _mask_to_grid(inp_mask)
    expected = _gen_expected_output(out_mask, ray_sources)
    if not np.any(expected == RAY):
        return None

    found_tabs = {cell for cell, _direction, _target in _tab_sources(inp_mask)}
    found_notches = {cell for cell, _direction, _target in _notch_sources(inp_mask)}
    found_new_tabs = {(target, direction) for _cell, direction, target in _notch_sources(inp_mask)}
    expected_new_tabs = set(ray_sources)

    if found_tabs != expected_tab_cells:
        return None
    if found_notches != expected_notch_cells:
        return None
    if found_new_tabs != expected_new_tabs:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
