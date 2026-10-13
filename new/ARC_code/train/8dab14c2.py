from collections import Counter

import numpy as np

from arc_utils import colors_present, get_bg, in_bounds
import random


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(line):
        if bool(v) and start is None:
            start = i
        if start is not None and (not bool(v) or i == len(line) - 1):
            end = i - 1 if not bool(v) else i
            runs.append((start, end))
            start = None
    return runs


def _body_thickness(mask: np.ndarray) -> int:
    lengths: list[int] = []
    for r in range(mask.shape[0]):
        lengths.extend(e - s + 1 for s, e in _runs(mask[r]) if e - s + 1 >= 2)
    for c in range(mask.shape[1]):
        lengths.extend(e - s + 1 for s, e in _runs(mask[:, c]) if e - s + 1 >= 2)

    assert lengths, "foreground object must contain runs wider than one pixel"
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


def _outside_opposite_edge(
    mask: np.ndarray, r: int, c: int, dr: int, dc: int
) -> tuple[int, int] | None:
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
                assert target != (r, c), "tab must project to the opposite side"
                tabs.append(((r, c), (dr, dc), target))

    return tabs


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each boundary tab or notch by adding the complementary feature on the opposite side of the foreground object.

    Input structure
    - The grid has exactly two colors: a dominant background and one foreground object.
    - The foreground object is a thick rectilinear shape with one-pixel tabs and notches on its boundary.

    Rule
    - A foreground tab is a one-neighbor boundary pixel; project through its full row or column span and remove the opposite edge pixel to make a matching notch.
    - A background notch is a one-pixel cut with three foreground 4-neighbors; project across the full row or column span and add a tab just outside the opposite edge.
    - Notches immediately beside a tab source are not independent sources, because that gap belongs to the tab's local boundary shape.

    Key insight
    - The changed pixels are not color-specific noise; they are paired jigsaw-like boundary features on opposite sides of the same thick row or column cross-section.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g)
    assert len(colors) == 2, "task expects exactly one foreground color and one background"
    fg_colors = [color for color in colors if color != bg]
    assert len(fg_colors) == 1, "task expects a single foreground color"
    fg = fg_colors[0]

    mask = g == fg
    out = g.copy()

    tabs = _tab_sources(mask)
    tab_cells = {cell for cell, _direction, _target in tabs}
    changes = 0

    for _cell, _direction, target in tabs:
        if out[target] != bg:
            out[target] = bg
            changes += 1

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
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
                if out[target] != fg:
                    out[target] = fg
                    changes += 1

    assert changes > 0, "at least one complementary boundary feature must be produced"
    return out

# --------------- generator (att2) ---------------

_GEN_BG = 8
_GEN_FG = 1
_GEN_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Canvas and placement bounds.
_GEN_ROWS_RANGE = (13, 16)
_GEN_COLS_RANGE = (15, 17)
_GEN_MARGIN = 1

# Clean L-shape arm bounds.
_GEN_H_THICK_RANGE = (3, 5)
_GEN_V_WIDTH_RANGE = (3, 6)
_GEN_V_EXTRA_RANGE = (4, 10)
_GEN_H_WIDTH_RANGE = (11, 14)
_GEN_MAX_BODY_HEIGHT = 14
_GEN_MIN_MODE_GAP = 2
_GEN_MISSING_CORNERS = ("TL", "TR", "BL", "BR")

# Boundary source feature counts.
_GEN_PAIR_COUNT_RANGE = (1, 3)
_GEN_MAX_TABS = 4

# Retry limits.
_GEN_MAX_ATTEMPTS = 300


def _gen_in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _gen_has_tangent_support(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> bool:
    if dr:
        sides = ((r, c - 1), (r, c + 1))
    else:
        sides = ((r - 1, c), (r + 1, c))
    return all(_gen_in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]) for rr, cc in sides)


def _gen_line_edges(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if dr:
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0:
            return None
        near_r = int(rows.min() if dr > 0 else rows.max())
        far_r = int(rows.max() if dr > 0 else rows.min())
        return (near_r, c), (far_r, c)

    cols = np.flatnonzero(mask[r])
    if cols.size == 0:
        return None
    near_c = int(cols.min() if dc > 0 else cols.max())
    far_c = int(cols.max() if dc > 0 else cols.min())
    return (r, near_c), (r, far_c)


def _gen_feature_slots(mask: np.ndarray) -> list[dict]:
    slots: list[dict] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]):
                continue
            for dr, dc in _GEN_DIRS4:
                outside_r, outside_c = r - dr, c - dc
                if not _gen_in_bounds(mask.shape, outside_r, outside_c):
                    continue
                if bool(mask[outside_r, outside_c]):
                    continue

                edges = _gen_line_edges(mask, r, c, dr, dc)
                if edges is None or edges[0] != (r, c):
                    continue
                near, far = edges
                far_out = (far[0] + dr, far[1] + dc)
                if not _gen_in_bounds(mask.shape, far_out[0], far_out[1]):
                    continue
                if bool(mask[far_out]):
                    continue
                near_margin = (outside_r - dr, outside_c - dc)
                far_margin = (far_out[0] + dr, far_out[1] + dc)
                if not _gen_in_bounds(mask.shape, near_margin[0], near_margin[1]):
                    continue
                if not _gen_in_bounds(mask.shape, far_margin[0], far_margin[1]):
                    continue

                inner_r, inner_c = r + dr, c + dc
                if not _gen_in_bounds(mask.shape, inner_r, inner_c) or not bool(mask[inner_r, inner_c]):
                    continue
                if not _gen_has_tangent_support(mask, near[0], near[1], dr, dc):
                    continue
                if not _gen_has_tangent_support(mask, far[0], far[1], dr, dc):
                    continue

                slots.append(
                    {
                        "near": near,
                        "near_out": (outside_r, outside_c),
                        "far": far,
                        "far_out": far_out,
                        "dir": (dr, dc),
                    }
                )
    return slots


def _gen_protected_cells(slot: dict) -> set[tuple[int, int]]:
    return {slot["near"], slot["near_out"], slot["far"], slot["far_out"]}


def _gen_l_configs() -> list[tuple[int, int, int, int]]:
    """Return clean L dimensions whose solver thickness mode is unambiguous."""
    configs: list[tuple[int, int, int, int]] = []
    for h_thick in range(_GEN_H_THICK_RANGE[0], _GEN_H_THICK_RANGE[1] + 1):
        for v_width in range(_GEN_V_WIDTH_RANGE[0], _GEN_V_WIDTH_RANGE[1] + 1):
            for v_extra in range(_GEN_V_EXTRA_RANGE[0], _GEN_V_EXTRA_RANGE[1] + 1):
                height = h_thick + v_extra
                if height > _GEN_MAX_BODY_HEIGHT:
                    continue
                for h_width in range(_GEN_H_WIDTH_RANGE[0], _GEN_H_WIDTH_RANGE[1] + 1):
                    run_counts: dict[int, int] = {}
                    for length, count in (
                        (h_width, h_thick),
                        (v_width, v_extra),
                        (height, v_width),
                        (h_thick, h_width - v_width),
                    ):
                        run_counts[length] = run_counts.get(length, 0) + count
                    max_count = max(run_counts.values())
                    mode = min(length for length, count in run_counts.items() if count == max_count)
                    counts = sorted(run_counts.values(), reverse=True)
                    if mode == min(h_thick, v_width) and counts[0] >= counts[1] + _GEN_MIN_MODE_GAP:
                        configs.append((h_thick, v_width, v_extra, h_width))
    return configs


def _gen_clean_l() -> tuple[np.ndarray, int, int, int, int, str] | None:
    h_thick, v_width, v_extra, h_width = random.choice(_gen_l_configs())

    height = h_thick + v_extra
    rows = random.randint(max(_GEN_ROWS_RANGE[0], height + 2 * _GEN_MARGIN), _GEN_ROWS_RANGE[1])
    cols = random.randint(max(_GEN_COLS_RANGE[0], h_width + 2 * _GEN_MARGIN), _GEN_COLS_RANGE[1])
    r0 = random.randint(_GEN_MARGIN, rows - height - _GEN_MARGIN)
    c0 = random.randint(_GEN_MARGIN, cols - h_width - _GEN_MARGIN)

    missing_corner = random.choice(_GEN_MISSING_CORNERS)
    h_on_top = missing_corner[0] == "B"
    v_on_left = missing_corner[1] == "R"

    mask = np.zeros((rows, cols), dtype=bool)
    h_rows = slice(r0, r0 + h_thick) if h_on_top else slice(r0 + v_extra, r0 + height)
    v_cols = slice(c0, c0 + v_width) if v_on_left else slice(c0 + h_width - v_width, c0 + h_width)
    mask[h_rows, c0 : c0 + h_width] = True
    mask[r0 : r0 + height, v_cols] = True
    return mask, h_thick, v_width, v_extra, h_width, missing_corner


def _gen_notch_sources(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]]:
    tab_cells = {cell for cell, _direction, _target in _tab_sources(mask)}
    notches: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if bool(mask[r, c]):
                continue

            fg_dirs: list[tuple[int, int]] = []
            bg_dirs: list[tuple[int, int]] = []
            for dr, dc in _GEN_DIRS4:
                rr, cc = r + dr, c + dc
                if _gen_in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
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

            if dr:
                rows = np.flatnonzero(mask[:, c])
                target = None if rows.size == 0 else (int(rows.max() + 1 if dr > 0 else rows.min() - 1), c)
            else:
                cols = np.flatnonzero(mask[r])
                target = None if cols.size == 0 else (r, int(cols.max() + 1 if dc > 0 else cols.min() - 1))
            if target is not None and not _gen_in_bounds(mask.shape, target[0], target[1]):
                target = None
            notches.append(((r, c), (dr, dc), target))
    return notches


def _gen_apply_source_features(clean: np.ndarray, chosen: list[tuple[dict, str]]) -> np.ndarray:
    mask = clean.copy()
    for slot, source_type in chosen:
        if source_type == "tab":
            mask[slot["near_out"]] = True
        else:
            mask[slot["near"]] = False
    return mask


def _gen_source_sets_match(mask: np.ndarray, chosen: list[tuple[dict, str]]) -> bool:
    expected_tab_cells = {slot["near_out"] for slot, source_type in chosen if source_type == "tab"}
    found_tab_cells = {cell for cell, _direction, _target in _tab_sources(mask)}
    if found_tab_cells != expected_tab_cells:
        return False

    expected_notch_cells = {slot["near"] for slot, source_type in chosen if source_type == "notch"}
    found_notch_cells = {
        cell
        for cell, _direction, target in _gen_notch_sources(mask)
        if target is not None and not bool(mask[target])
    }
    return found_notch_cells == expected_notch_cells


def _gen_source_types() -> list[str]:
    source_count = 2 * random.randint(_GEN_PAIR_COUNT_RANGE[0], _GEN_PAIR_COUNT_RANGE[1])
    min_tabs = max(0, source_count - _GEN_MAX_TABS)
    max_tabs = min(_GEN_MAX_TABS, source_count)
    tab_count = random.randint(min_tabs, max_tabs)
    source_types = ["tab"] * tab_count + ["notch"] * (source_count - tab_count)
    random.shuffle(source_types)
    return source_types


def _gen_choose_features(
    clean: np.ndarray,
    slots: list[dict],
    source_types: list[str],
    idx: int = 0,
    chosen: list[tuple[dict, str]] | None = None,
    blocked: set[tuple[int, int]] | None = None,
) -> list[tuple[dict, str]] | None:
    if chosen is None:
        chosen = []
    if blocked is None:
        blocked = set()
    if idx == len(source_types):
        return chosen

    candidates = slots[:]
    random.shuffle(candidates)
    source_type = source_types[idx]
    for slot in candidates:
        protected = _gen_protected_cells(slot)
        if protected & blocked:
            continue
        trial = chosen + [(slot, source_type)]
        source_mask = _gen_apply_source_features(clean, trial)
        if not _gen_source_sets_match(source_mask, trial):
            continue
        result = _gen_choose_features(clean, slots, source_types, idx + 1, trial, blocked | protected)
        if result is not None:
            return result
    return None


def _gen_render_pair(clean: np.ndarray, chosen: list[tuple[dict, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Return input and expected-output masks for the selected source features."""
    out_mask = clean.copy()
    for slot, source_type in chosen:
        if source_type == "tab":
            out_mask[slot["near_out"]] = True
            out_mask[slot["far"]] = False
        else:
            out_mask[slot["near"]] = False
            out_mask[slot["far_out"]] = True

    inp_mask = out_mask.copy()
    for slot, source_type in chosen:
        if source_type == "tab":
            inp_mask[slot["far"]] = True
        else:
            inp_mask[slot["far_out"]] = False
    return inp_mask, out_mask


def _gen_to_grid(mask: np.ndarray) -> np.ndarray:
    grid = np.full(mask.shape, _GEN_BG, dtype=int)
    grid[mask] = _GEN_FG
    return grid


def _gen_attempt() -> np.ndarray | None:
    clean_result = _gen_clean_l()
    if clean_result is None:
        return None
    clean, h_thick, v_width, _v_extra, _h_width, _missing_corner = clean_result

    slots = _gen_feature_slots(clean)
    random.shuffle(slots)
    chosen = _gen_choose_features(clean, slots, _gen_source_types())
    if chosen is None:
        return None

    inp_mask, out_mask = _gen_render_pair(clean, chosen)

    intended_thickness = min(h_thick, v_width)
    if _body_thickness(inp_mask) != intended_thickness:
        return None

    if not _gen_source_sets_match(inp_mask, chosen):
        return None

    inp = _gen_to_grid(inp_mask)
    expected = _gen_to_grid(out_mask)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 8dab14c2 input")
