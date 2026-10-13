from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FG = 1
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty foreground"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_fg_neighbors4(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            out.append((dr, dc))
    return out


def _tab_sources(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    tabs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]):
                continue
            neighbors = _count_fg_neighbors4(mask, r, c)
            if len(neighbors) == 1:
                tabs.append(((r, c), neighbors[0]))
    return tabs


def _core_bbox(mask: np.ndarray, tabs: list[tuple[tuple[int, int], tuple[int, int]]]) -> tuple[int, int, int, int]:
    core = mask.copy()
    for (r, c), _dir_in in tabs:
        core[r, c] = False
    return _bbox(core)


def _opposite_notch_target(
    core_bbox: tuple[int, int, int, int], r: int, c: int, dr_in: int, dc_in: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = core_bbox
    if dr_in == 1:
        return r1, c
    if dr_in == -1:
        return r0, c
    if dc_in == 1:
        return r, c1
    assert dc_in == -1
    return r, c0


def _opposite_tab_target(
    core_bbox: tuple[int, int, int, int], r: int, c: int, outside_dr: int, outside_dc: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = core_bbox
    if outside_dr == -1:
        return r1 + 1, c
    if outside_dr == 1:
        return r0 - 1, c
    if outside_dc == -1:
        return r, c1 + 1
    assert outside_dc == 1
    return r, c0 - 1


def _notch_sources(
    mask: np.ndarray,
    tab_cells: set[tuple[int, int]],
    core_bbox: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = core_bbox
    notches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if bool(mask[r, c]):
                continue
            fg_dirs = []
            bg_dirs = []
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
                    fg_dirs.append((dr, dc))
                else:
                    bg_dirs.append((dr, dc))
            if len(fg_dirs) != 3 or len(bg_dirs) != 1:
                continue
            outside_dr, outside_dc = bg_dirs[0]
            dr_in, dc_in = -outside_dr, -outside_dc
            side_a = (r - dc_in, c + dr_in)
            side_b = (r + dc_in, c - dr_in)
            if side_a in tab_cells or side_b in tab_cells:
                continue
            target = _opposite_tab_target(core_bbox, r, c, outside_dr, outside_dc)
            if _in_bounds(mask.shape, *target):
                notches.append(((r, c), (outside_dr, outside_dc)))
    return notches


def _complete_shape(
    mask: np.ndarray,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]], tuple[int, int, int, int]]:
    tabs = _tab_sources(mask)
    tab_cells = {cell for cell, _dir_in in tabs}
    core_bbox = _core_bbox(mask, tabs)
    notches = _notch_sources(mask, tab_cells, core_bbox)

    completed = mask.copy()
    added_tabs: set[tuple[int, int]] = set()
    removed_pixels: set[tuple[int, int]] = set()

    for (r, c), (dr_in, dc_in) in tabs:
        rr, cc = _opposite_notch_target(core_bbox, r, c, dr_in, dc_in)
        if completed[rr, cc]:
            completed[rr, cc] = False
            removed_pixels.add((rr, cc))

    for (r, c), (outside_dr, outside_dc) in notches:
        rr, cc = _opposite_tab_target(core_bbox, r, c, outside_dr, outside_dc)
        if _in_bounds(mask.shape, rr, cc) and not completed[rr, cc]:
            completed[rr, cc] = True
            added_tabs.add((rr, cc))

    return completed, added_tabs, removed_pixels, core_bbox


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not bool(mask[r, c]) and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(mask[r, c]) and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and not bool(mask[nr, nc]) and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _outline_sets(
    completed_mask: np.ndarray, added_tabs: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    exterior = _exterior_background(completed_mask)
    green: set[tuple[int, int]] = set()
    yellow: set[tuple[int, int]] = set()

    for r in range(completed_mask.shape[0]):
        for c in range(completed_mask.shape[1]):
            if completed_mask[r, c] or not exterior[r, c]:
                continue
            touches_fg = any(
                _in_bounds(completed_mask.shape, nr, nc) and bool(completed_mask[nr, nc])
                for nr, nc in _neighbors8(r, c)
            )
            if not touches_fg:
                continue
            touches_new_tab = any((nr, nc) in added_tabs for nr, nc in _neighbors8(r, c))
            if touches_new_tab:
                yellow.add((r, c))
            else:
                green.add((r, c))

    return green, yellow


def _count_components(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    n = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            n += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return n


def _oracle_from_input(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    completed, added_tabs, removed_pixels, core_bbox = _complete_shape(mask)
    green, yellow = _outline_sets(completed, added_tabs)
    out = np.full(mask.shape, BG, dtype=int)
    out[completed] = FG
    for r, c in green:
        out[r, c] = GREEN
    for r, c in yellow:
        out[r, c] = YELLOW

    meta = {
        "completed_mask": completed,
        "added_tabs": added_tabs,
        "removed_pixels": removed_pixels,
        "green": green,
        "yellow": yellow,
        "core_bbox": core_bbox,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the rectangle's missing opposite tabs/notches, then color the exterior outline of the completed shape with green and yellow.

    Input structure:
    - The grid has black background and one blue thick rectangular foreground object.
    - The rectangle boundary may contain one-pixel tabs (blue protrusions) and one-pixel notches (black dents).
    - Green and yellow do not appear in the input.

    Rule:
    - First perform tab-notch symmetric completion across the rectangle body: every tab creates an opposite notch on the same row or column, and every notch creates an opposite tab just outside the opposite side.
    - Ignore notch candidates that are only the local side-gap of a tab.
    - Then flood-fill exterior background on the completed shape and color every exterior background cell that 8-neighbors the completed foreground as outline.
    - Outline cells touching a newly added tab become yellow; all other exterior outline cells become green. Foreground stays blue.

    Key insight:
    - The final outline must be traced on the post-completion shape, and the yellow subset identifies boundary created specifically by newly added symmetric tabs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    fg_colors = _colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected one foreground color"
    assert GREEN not in fg_colors and YELLOW not in fg_colors, "green/yellow are output-only"
    fg = fg_colors[0]
    assert int(bg) == BG and int(fg) == FG, "task uses black background and blue foreground"

    mask = g == fg
    assert _count_components(mask) == 1, "expected one connected foreground object"

    out, _meta = _oracle_from_input(mask)
    return out


GEN_ATTEMPTS = 400
GEN_GRID_ROWS = (14, 18)
GEN_GRID_COLS = (14, 18)
GEN_CORE_H = (6, 9)
GEN_CORE_W = (7, 10)
GEN_MARGIN = 2
GEN_TOTAL_FEATURES = (2, 4)


def _feature_source_cell(
    core_bbox: tuple[int, int, int, int], kind: str, side: str, offset: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = core_bbox
    if side == "top":
        return (r0 - 1, c0 + offset) if kind == "tab" else (r0, c0 + offset)
    if side == "bottom":
        return (r1 + 1, c0 + offset) if kind == "tab" else (r1, c0 + offset)
    if side == "left":
        return (r0 + offset, c0 - 1) if kind == "tab" else (r0 + offset, c0)
    assert side == "right"
    return (r0 + offset, c1 + 1) if kind == "tab" else (r0 + offset, c1)


def _feature_target_cell(
    core_bbox: tuple[int, int, int, int], kind: str, side: str, offset: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = core_bbox
    if kind == "tab":
        if side == "top":
            return r1, c0 + offset
        if side == "bottom":
            return r0, c0 + offset
        if side == "left":
            return r0 + offset, c1
        return r0 + offset, c0
    if side == "top":
        return r1 + 1, c0 + offset
    if side == "bottom":
        return r0 - 1, c0 + offset
    if side == "left":
        return r0 + offset, c1 + 1
    return r0 + offset, c0 - 1


def _sample_feature_plan(
    core_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[dict] | None:
    r0, c0, r1, c1 = core_bbox
    body_h = r1 - r0 + 1
    body_w = c1 - c0 + 1
    row_offsets = list(range(1, body_h - 1))
    col_offsets = list(range(1, body_w - 1))
    if not row_offsets or not col_offsets:
        return None

    total = random.randint(*GEN_TOTAL_FEATURES)
    notch_count = random.randint(1, min(2, total))
    tab_count = total - notch_count
    if tab_count > 2:
        tab_count = 2
        notch_count = total - tab_count
    if notch_count == 0:
        notch_count = 1
        tab_count = total - 1

    kinds = ["notch"] * notch_count + ["tab"] * tab_count
    random.shuffle(kinds)

    available_rows = row_offsets[:]
    available_cols = col_offsets[:]
    used_source: set[tuple[int, int]] = set()
    used_target: set[tuple[int, int]] = set()
    features: list[dict] = []

    for kind in kinds:
        choices = []
        if available_rows:
            choices.append("row")
        if available_cols:
            choices.append("col")
        if not choices:
            return None
        random.shuffle(choices)

        placed = False
        for axis in choices:
            offsets = available_rows[:] if axis == "row" else available_cols[:]
            random.shuffle(offsets)
            sides = ["left", "right"] if axis == "row" else ["top", "bottom"]
            for offset in offsets:
                random.shuffle(sides)
                for side in sides:
                    src = _feature_source_cell(core_bbox, kind, side, offset)
                    tgt = _feature_target_cell(core_bbox, kind, side, offset)
                    if not _in_bounds(shape, *src) or not _in_bounds(shape, *tgt):
                        continue
                    if src in used_source or src in used_target or tgt in used_source or tgt in used_target:
                        continue
                    features.append(
                        {
                            "kind": kind,
                            "side": side,
                            "offset": offset,
                            "source": src,
                            "target": tgt,
                        }
                    )
                    used_source.add(src)
                    used_target.add(tgt)
                    if axis == "row":
                        available_rows.remove(offset)
                    else:
                        available_cols.remove(offset)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    return features


def _build_input_mask(shape: tuple[int, int], core_bbox: tuple[int, int, int, int], features: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = core_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for feature in features:
        sr, sc = feature["source"]
        if feature["kind"] == "tab":
            mask[sr, sc] = True
        else:
            mask[sr, sc] = False
    return mask


def _outline_without_completion(mask: np.ndarray) -> np.ndarray:
    exterior = _exterior_background(mask)
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = FG
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] or not exterior[r, c]:
                continue
            if any(_in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) for nr, nc in _neighbors8(r, c)):
                out[r, c] = GREEN
    return out


def _validate_example(mask: np.ndarray, features: list[dict]) -> bool:
    if _count_components(mask) != 1:
        return False

    expected, meta = _oracle_from_input(mask)
    if not np.array_equal(main(mask.astype(int)), expected):
        return False

    expected_added = {tuple(feature["target"]) for feature in features if feature["kind"] == "notch"}
    expected_removed = {tuple(feature["target"]) for feature in features if feature["kind"] == "tab"}
    if meta["added_tabs"] != expected_added:
        return False
    if meta["removed_pixels"] != expected_removed:
        return False

    if len(meta["added_tabs"]) == 0:
        return False
    if len(meta["yellow"]) == 0:
        return False
    if len(meta["green"]) == 0:
        return False

    naive_outline = _outline_without_completion(mask)
    if np.array_equal(expected, naive_outline):
        return False

    all_green = expected.copy()
    all_green[all_green == YELLOW] = GREEN
    if np.array_equal(expected, all_green):
        return False

    completed = meta["completed_mask"]
    exterior = _exterior_background(completed)
    for r, c in meta["yellow"]:
        if not exterior[r, c]:
            return False
        if not any((nr, nc) in meta["added_tabs"] for nr, nc in _neighbors8(r, c)):
            return False

    for feature in features:
        sr, sc = feature["source"]
        if feature["kind"] == "tab":
            if not bool(mask[sr, sc]):
                return False
            if len(_count_fg_neighbors4(mask, sr, sc)) != 1:
                return False
        else:
            if bool(mask[sr, sc]):
                return False
            if len(_count_fg_neighbors4(mask, sr, sc)) != 3:
                return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_ROWS)
    w = random.randint(*GEN_GRID_COLS)
    core_h = random.randint(*GEN_CORE_H)
    core_w = random.randint(*GEN_CORE_W)
    if core_h + 2 * GEN_MARGIN > h or core_w + 2 * GEN_MARGIN > w:
        return None

    r0 = random.randint(GEN_MARGIN, h - core_h - GEN_MARGIN)
    c0 = random.randint(GEN_MARGIN, w - core_w - GEN_MARGIN)
    core_bbox = (r0, c0, r0 + core_h - 1, c0 + core_w - 1)

    features = _sample_feature_plan(core_bbox, (h, w))
    if not features:
        return None

    mask = _build_input_mask((h, w), core_bbox, features)
    if not _validate_example(mask, features):
        return None
    return mask.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
