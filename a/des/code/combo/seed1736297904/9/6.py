import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)

LINE_GAP_TOLERANCE = 3
GEN_RETRIES = 400
REGION_LAYOUTS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
REGION_SIZE_RANGE = (4, 7)
CHAIN_PROB = 0.35


def _colored(value: int) -> bool:
    return int(value) not in (BLACK, GREY, MAROON)


def _detect_boundary_lines(g: np.ndarray, axis: int) -> list[int]:
    maroon = (g == MAROON)
    counts = maroon.sum(axis=1 - axis)
    threshold = int(counts.max()) - LINE_GAP_TOLERANCE
    lines = [int(i) for i, cnt in enumerate(counts) if int(cnt) >= threshold]
    assert lines, "expected at least one partition line"
    assert lines[0] == 0 and lines[-1] == g.shape[axis] - 1, "outer border must be maroon"
    assert all(b - a >= 2 for a, b in zip(lines, lines[1:], strict=False)), "regions must have non-empty interiors"
    return lines


def _region_boxes(row_lines: list[int], col_lines: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for ri in range(len(row_lines) - 1):
        for cj in range(len(col_lines) - 1):
            r0 = int(row_lines[ri] + 1)
            r1 = int(row_lines[ri + 1] - 1)
            c0 = int(col_lines[cj] + 1)
            c1 = int(col_lines[cj + 1] - 1)
            assert r0 <= r1 and c0 <= c1
            boxes[(ri, cj)] = (r0, r1, c0, c1)
    return boxes


def _find_regions(g: np.ndarray) -> tuple[list[int], list[int], dict[tuple[int, int], dict]]:
    row_lines = _detect_boundary_lines(g, axis=0)
    col_lines = _detect_boundary_lines(g, axis=1)
    boxes = _region_boxes(row_lines, col_lines)

    regions: dict[tuple[int, int], dict] = {}
    for key, (r0, r1, c0, c1) in boxes.items():
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        marker_coords = np.argwhere((sub != BLACK) & (sub != GREY) & (sub != MAROON))
        assert marker_coords.shape[0] == 1, "each region must contain exactly one colored marker"
        mr = int(marker_coords[0, 0] + r0)
        mc = int(marker_coords[0, 1] + c0)
        regions[key] = {
            "box": (r0, r1, c0, c1),
            "marker": (mr, mc),
            "color": int(g[mr, mc]),
        }
    return row_lines, col_lines, regions


def _local_cross_fill(out: np.ndarray, regions: dict[tuple[int, int], dict]) -> None:
    for region in regions.values():
        r0, r1, c0, c1 = region["box"]
        mr, mc = region["marker"]
        color = region["color"]

        row = out[mr, c0 : c1 + 1]
        row[row == GREY] = color
        col = out[r0 : r1 + 1, mc]
        col[col == GREY] = color


def _extract_gaps(
    g: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
) -> list[dict]:
    gaps: list[dict] = []

    for cj in range(1, len(col_lines) - 1):
        c = int(col_lines[cj])
        for ri in range(len(row_lines) - 1):
            r_start = int(row_lines[ri] + 1)
            r_end = int(row_lines[ri + 1] - 1)
            for r in range(r_start, r_end + 1):
                if int(g[r, c]) == MAROON:
                    continue
                gaps.append({"orient": "v", "row": r, "col": c, "ri": ri, "cj": cj - 1})

    for ri in range(1, len(row_lines) - 1):
        r = int(row_lines[ri])
        for cj in range(len(col_lines) - 1):
            c_start = int(col_lines[cj] + 1)
            c_end = int(col_lines[cj + 1] - 1)
            for c in range(c_start, c_end + 1):
                if int(g[r, c]) == MAROON:
                    continue
                gaps.append({"orient": "h", "row": r, "col": c, "ri": ri - 1, "cj": cj})

    gaps.sort(key=lambda gap: (int(gap["row"]), int(gap["col"]), int(gap["orient"] == "v")))
    return gaps


def _spill_once(out: np.ndarray, regions: dict[tuple[int, int], dict], gap: dict) -> bool:
    changed = False
    orient = gap["orient"]
    r = int(gap["row"])
    c = int(gap["col"])
    ri = int(gap["ri"])
    cj = int(gap["cj"])

    if orient == "v":
        left = regions[(ri, cj)]["box"]
        right = regions[(ri, cj + 1)]["box"]

        left_color = int(out[r, c - 1])
        if _colored(left_color):
            _, _, rc0, rc1 = right
            row = out[r, rc0 : rc1 + 1]
            mask = row == GREY
            if np.any(mask):
                row[mask] = left_color
                changed = True

        right_color = int(out[r, c + 1])
        if _colored(right_color):
            _, _, lc0, lc1 = left
            row = out[r, lc0 : lc1 + 1]
            mask = row == GREY
            if np.any(mask):
                row[mask] = right_color
                changed = True
    else:
        top = regions[(ri, cj)]["box"]
        bottom = regions[(ri + 1, cj)]["box"]

        top_color = int(out[r - 1, c])
        if _colored(top_color):
            br0, br1, _, _ = bottom
            col = out[br0 : br1 + 1, c]
            mask = col == GREY
            if np.any(mask):
                col[mask] = top_color
                changed = True

        bottom_color = int(out[r + 1, c])
        if _colored(bottom_color):
            tr0, tr1, _, _ = top
            col = out[tr0 : tr1 + 1, c]
            mask = col == GREY
            if np.any(mask):
                col[mask] = bottom_color
                changed = True

    return changed


def _propagate_spills(out: np.ndarray, regions: dict[tuple[int, int], dict], gaps: list[dict]) -> None:
    changed = True
    while changed:
        changed = False
        for gap in gaps:
            if _spill_once(out, regions, gap):
                changed = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each maroon-bounded rectangular region, the unique colored marker recolors same-row and
    same-column gray cells. One-cell gaps on the partition lines then let already-colored pixels
    spill into adjacent regions, recoloring gray cells on the matching row or column while never
    overwriting existing non-gray colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_lines, col_lines, regions = _find_regions(g)
    gaps = _extract_gaps(g, row_lines, col_lines)

    out = g.copy()
    _local_cross_fill(out, regions)
    _propagate_spills(out, regions, gaps)
    return out


def _build_lines(sizes: list[int]) -> list[int]:
    lines = [0]
    pos = 0
    for size in sizes:
        pos += int(size) + 1
        lines.append(pos)
    return lines


def _sample_sizes(count: int) -> list[int]:
    return [random.randint(*REGION_SIZE_RANGE) for _ in range(count)]


def _choose_layout() -> tuple[list[int], list[int]]:
    for _ in range(GEN_RETRIES):
        nr, nc = random.choice(REGION_LAYOUTS)
        row_sizes = _sample_sizes(nr)
        col_sizes = _sample_sizes(nc)
        h = sum(row_sizes) + nr + 1
        w = sum(col_sizes) + nc + 1
        if 12 <= h <= 26 and 12 <= w <= 26:
            return row_sizes, col_sizes
    raise RuntimeError("failed to choose a valid region layout")


def _initial_region_data(
    row_lines: list[int],
    col_lines: list[int],
) -> dict[tuple[int, int], dict]:
    boxes = _region_boxes(row_lines, col_lines)
    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    assert len(boxes) <= len(colors)

    regions: dict[tuple[int, int], dict] = {}
    for key, (r0, r1, c0, c1) in boxes.items():
        mr = random.randint(r0, r1)
        mc = random.randint(c0, c1)
        regions[key] = {
            "box": (r0, r1, c0, c1),
            "marker": (mr, mc),
            "color": colors[len(regions)],
            "gray": set(),
        }

    for region in regions.values():
        r0, r1, c0, c1 = region["box"]
        mr, mc = region["marker"]

        row_cells = [(mr, c) for c in range(c0, c1 + 1) if c != mc]
        col_cells = [(r, mc) for r in range(r0, r1 + 1) if r != mr]
        other_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if r != mr and c != mc]

        row_count = random.randint(1, min(3, len(row_cells)))
        col_count = random.randint(1, min(3, len(col_cells)))
        other_count = random.randint(1, min(4, len(other_cells)))

        region["gray"].update(random.sample(row_cells, row_count))
        region["gray"].update(random.sample(col_cells, col_count))
        region["gray"].update(random.sample(other_cells, other_count))

    return regions


def _add_gray(region: dict, cell: tuple[int, int]) -> None:
    if cell != region["marker"]:
        region["gray"].add((int(cell[0]), int(cell[1])))


def _remove_gray(region: dict, cell: tuple[int, int]) -> None:
    region["gray"].discard((int(cell[0]), int(cell[1])))


def _vertical_adjacencies(row_lines: list[int], col_lines: list[int]) -> list[dict]:
    adjs = []
    for ri in range(len(row_lines) - 1):
        for cj in range(len(col_lines) - 2):
            adjs.append({"orient": "v", "ri": ri, "cj": cj, "col": int(col_lines[cj + 1])})
    return adjs


def _horizontal_adjacencies(row_lines: list[int], col_lines: list[int]) -> list[dict]:
    adjs = []
    for ri in range(len(row_lines) - 2):
        for cj in range(len(col_lines) - 1):
            adjs.append({"orient": "h", "ri": ri, "cj": cj, "row": int(row_lines[ri + 1])})
    return adjs


def _target_row_candidates(region: dict, row: int, banned_cols: set[int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = region["box"]
    assert r0 <= row <= r1
    candidates = []
    for c in range(c0, c1 + 1):
        if c in banned_cols:
            continue
        if (row, c) == region["marker"]:
            continue
        candidates.append((row, c))
    return candidates


def _target_col_candidates(region: dict, col: int, banned_rows: set[int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = region["box"]
    assert c0 <= col <= c1
    candidates = []
    for r in range(r0, r1 + 1):
        if r in banned_rows:
            continue
        if (r, col) == region["marker"]:
            continue
        candidates.append((r, col))
    return candidates


def _setup_vertical_active(
    regions: dict[tuple[int, int], dict],
    ri: int,
    cj: int,
    source_is_left: bool,
    gaps: list[dict],
) -> bool:
    left = regions[(ri, cj)]
    right = regions[(ri, cj + 1)]
    boundary = int(left["box"][3] + 1)
    source = left if source_is_left else right
    target = right if source_is_left else left
    row = int(source["marker"][0])

    if row == int(target["marker"][0]):
        return False

    source_adj = (row, boundary - 1) if source_is_left else (row, boundary + 1)
    target_adj = (row, boundary + 1) if source_is_left else (row, boundary - 1)

    gaps.append({"orient": "v", "row": row, "col": boundary, "ri": ri, "cj": cj})
    _add_gray(source, source_adj)
    _remove_gray(target, target_adj)

    banned_cols = {target_adj[1], int(target["marker"][1])}
    spill_targets = _target_row_candidates(target, row, banned_cols)
    if not spill_targets:
        return False
    count = random.randint(1, min(3, len(spill_targets)))
    for cell in random.sample(spill_targets, count):
        _add_gray(target, cell)

    priority_col = int(target["marker"][1])
    if priority_col != target_adj[1] and random.random() < 0.45:
        _add_gray(target, (row, priority_col))
    return True


def _setup_horizontal_active(
    regions: dict[tuple[int, int], dict],
    ri: int,
    cj: int,
    source_is_top: bool,
    gaps: list[dict],
) -> bool:
    top = regions[(ri, cj)]
    bottom = regions[(ri + 1, cj)]
    boundary = int(top["box"][1] + 1)
    source = top if source_is_top else bottom
    target = bottom if source_is_top else top
    col = int(source["marker"][1])

    if col == int(target["marker"][1]):
        return False

    source_adj = (boundary - 1, col) if source_is_top else (boundary + 1, col)
    target_adj = (boundary + 1, col) if source_is_top else (boundary - 1, col)

    gaps.append({"orient": "h", "row": boundary, "col": col, "ri": ri, "cj": cj})
    _add_gray(source, source_adj)
    _remove_gray(target, target_adj)

    banned_rows = {target_adj[0], int(target["marker"][0])}
    spill_targets = _target_col_candidates(target, col, banned_rows)
    if not spill_targets:
        return False
    count = random.randint(1, min(3, len(spill_targets)))
    for cell in random.sample(spill_targets, count):
        _add_gray(target, cell)

    priority_row = int(target["marker"][0])
    if priority_row != target_adj[0] and random.random() < 0.45:
        _add_gray(target, (priority_row, col))
    return True


def _setup_vertical_inactive(regions: dict[tuple[int, int], dict], ri: int, cj: int, gaps: list[dict]) -> bool:
    left = regions[(ri, cj)]
    right = regions[(ri, cj + 1)]
    boundary = int(left["box"][3] + 1)
    r0 = int(left["box"][0])
    r1 = int(left["box"][1])
    rows = [r for r in range(r0, r1 + 1) if r not in {left["marker"][0], right["marker"][0]}]
    if not rows:
        return False
    row = random.choice(rows)
    gaps.append({"orient": "v", "row": row, "col": boundary, "ri": ri, "cj": cj})
    _remove_gray(left, (row, boundary - 1))
    _remove_gray(right, (row, boundary + 1))
    return True


def _setup_horizontal_inactive(regions: dict[tuple[int, int], dict], ri: int, cj: int, gaps: list[dict]) -> bool:
    top = regions[(ri, cj)]
    bottom = regions[(ri + 1, cj)]
    boundary = int(top["box"][1] + 1)
    c0 = int(top["box"][2])
    c1 = int(top["box"][3])
    cols = [c for c in range(c0, c1 + 1) if c not in {top["marker"][1], bottom["marker"][1]}]
    if not cols:
        return False
    col = random.choice(cols)
    gaps.append({"orient": "h", "row": boundary, "col": col, "ri": ri, "cj": cj})
    _remove_gray(top, (boundary - 1, col))
    _remove_gray(bottom, (boundary + 1, col))
    return True


def _try_add_chain(
    row_lines: list[int],
    col_lines: list[int],
    regions: dict[tuple[int, int], dict],
    used: set[tuple[str, int, int]],
    gaps: list[dict],
) -> bool:
    candidates = []
    for ri in range(len(row_lines) - 1):
        for cj in range(len(col_lines) - 3):
            a = regions[(ri, cj)]
            b = regions[(ri, cj + 1)]
            c = regions[(ri, cj + 2)]
            row = int(a["marker"][0])
            if row != int(b["marker"][0]) and row != int(c["marker"][0]):
                key1 = ("v", ri, cj)
                key2 = ("v", ri, cj + 1)
                if key1 not in used and key2 not in used:
                    candidates.append(("row", ri, cj))

    for ri in range(len(row_lines) - 3):
        for cj in range(len(col_lines) - 1):
            a = regions[(ri, cj)]
            b = regions[(ri + 1, cj)]
            c = regions[(ri + 2, cj)]
            col = int(a["marker"][1])
            if col != int(b["marker"][1]) and col != int(c["marker"][1]):
                key1 = ("h", ri, cj)
                key2 = ("h", ri + 1, cj)
                if key1 not in used and key2 not in used:
                    candidates.append(("col", ri, cj))

    if not candidates:
        return False

    kind, ri, cj = random.choice(candidates)
    if kind == "row":
        a = regions[(ri, cj)]
        b = regions[(ri, cj + 1)]
        c = regions[(ri, cj + 2)]
        row = int(a["marker"][0])
        boundary1 = int(a["box"][3] + 1)
        boundary2 = int(b["box"][3] + 1)

        gaps.append({"orient": "v", "row": row, "col": boundary1, "ri": ri, "cj": cj})
        gaps.append({"orient": "v", "row": row, "col": boundary2, "ri": ri, "cj": cj + 1})
        used.add(("v", ri, cj))
        used.add(("v", ri, cj + 1))

        _add_gray(a, (row, boundary1 - 1))
        _remove_gray(b, (row, boundary1 + 1))

        _add_gray(b, (row, boundary2 - 1))
        _remove_gray(c, (row, boundary2 + 1))

        mid_targets = _target_row_candidates(b, row, {boundary1 + 1})
        mid_targets = [cell for cell in mid_targets if cell[1] != boundary2 - 1 and cell[1] != b["marker"][1]]
        if mid_targets:
            _add_gray(b, random.choice(mid_targets))

        final_targets = _target_row_candidates(c, row, {boundary2 + 1, c["marker"][1]})
        if not final_targets:
            return False
        for cell in random.sample(final_targets, random.randint(1, min(3, len(final_targets)))):
            _add_gray(c, cell)
        return True

    a = regions[(ri, cj)]
    b = regions[(ri + 1, cj)]
    c = regions[(ri + 2, cj)]
    col = int(a["marker"][1])
    boundary1 = int(a["box"][1] + 1)
    boundary2 = int(b["box"][1] + 1)

    gaps.append({"orient": "h", "row": boundary1, "col": col, "ri": ri, "cj": cj})
    gaps.append({"orient": "h", "row": boundary2, "col": col, "ri": ri + 1, "cj": cj})
    used.add(("h", ri, cj))
    used.add(("h", ri + 1, cj))

    _add_gray(a, (boundary1 - 1, col))
    _remove_gray(b, (boundary1 + 1, col))

    _add_gray(b, (boundary2 - 1, col))
    _remove_gray(c, (boundary2 + 1, col))

    mid_targets = _target_col_candidates(b, col, {boundary1 + 1})
    mid_targets = [cell for cell in mid_targets if cell[0] != boundary2 - 1 and cell[0] != b["marker"][0]]
    if mid_targets:
        _add_gray(b, random.choice(mid_targets))

    final_targets = _target_col_candidates(c, col, {boundary2 + 1, c["marker"][0]})
    if not final_targets:
        return False
    for cell in random.sample(final_targets, random.randint(1, min(3, len(final_targets)))):
        _add_gray(c, cell)
    return True


def _render_input(row_lines: list[int], col_lines: list[int], regions: dict[tuple[int, int], dict], gaps: list[dict]) -> np.ndarray:
    h = int(row_lines[-1] + 1)
    w = int(col_lines[-1] + 1)
    g = np.full((h, w), BLACK, dtype=int)
    g[row_lines, :] = MAROON
    g[:, col_lines] = MAROON

    for gap in gaps:
        g[int(gap["row"]), int(gap["col"])] = BLACK

    for region in regions.values():
        mr, mc = region["marker"]
        g[mr, mc] = region["color"]
        for r, c in region["gray"]:
            assert g[r, c] == BLACK
            g[r, c] = GREY
    return g


def _oracle_from_metadata(
    inp: np.ndarray,
    regions: dict[tuple[int, int], dict],
    gaps: list[dict],
) -> np.ndarray:
    parsed = {
        key: {"box": region["box"], "marker": region["marker"], "color": region["color"]}
        for key, region in regions.items()
    }
    out = inp.copy()
    _local_cross_fill(out, parsed)
    _propagate_spills(out, parsed, sorted(gaps, key=lambda gap: (int(gap["row"]), int(gap["col"]), int(gap["orient"] == "v"))))
    return out


def _local_only(inp: np.ndarray, regions: dict[tuple[int, int], dict]) -> np.ndarray:
    parsed = {
        key: {"box": region["box"], "marker": region["marker"], "color": region["color"]}
        for key, region in regions.items()
    }
    out = inp.copy()
    _local_cross_fill(out, parsed)
    return out


def _add_more_gaps(
    row_lines: list[int],
    col_lines: list[int],
    regions: dict[tuple[int, int], dict],
    used: set[tuple[str, int, int]],
    gaps: list[dict],
) -> tuple[int, int]:
    active_added = 0
    inactive_added = 0

    active_choices = []
    for adj in _vertical_adjacencies(row_lines, col_lines):
        key = ("v", int(adj["ri"]), int(adj["cj"]))
        if key in used:
            continue
        active_choices.append(("v", int(adj["ri"]), int(adj["cj"]), True))
        active_choices.append(("v", int(adj["ri"]), int(adj["cj"]), False))
    for adj in _horizontal_adjacencies(row_lines, col_lines):
        key = ("h", int(adj["ri"]), int(adj["cj"]))
        if key in used:
            continue
        active_choices.append(("h", int(adj["ri"]), int(adj["cj"]), True))
        active_choices.append(("h", int(adj["ri"]), int(adj["cj"]), False))
    random.shuffle(active_choices)

    wanted_active = 1 + int(random.random() < 0.45)
    for orient, ri, cj, direction in active_choices:
        if active_added >= wanted_active:
            break
        ok = (
            _setup_vertical_active(regions, ri, cj, direction, gaps)
            if orient == "v"
            else _setup_horizontal_active(regions, ri, cj, direction, gaps)
        )
        if ok:
            used.add((orient, ri, cj))
            active_added += 1

    inactive_choices = []
    for adj in _vertical_adjacencies(row_lines, col_lines):
        key = ("v", int(adj["ri"]), int(adj["cj"]))
        if key not in used:
            inactive_choices.append(("v", int(adj["ri"]), int(adj["cj"])))
    for adj in _horizontal_adjacencies(row_lines, col_lines):
        key = ("h", int(adj["ri"]), int(adj["cj"]))
        if key not in used:
            inactive_choices.append(("h", int(adj["ri"]), int(adj["cj"])))
    random.shuffle(inactive_choices)

    wanted_inactive = random.randint(0, min(2, len(inactive_choices)))
    for orient, ri, cj in inactive_choices:
        if inactive_added >= wanted_inactive:
            break
        ok = (
            _setup_vertical_inactive(regions, ri, cj, gaps)
            if orient == "v"
            else _setup_horizontal_inactive(regions, ri, cj, gaps)
        )
        if ok:
            used.add((orient, ri, cj))
            inactive_added += 1

    return active_added, inactive_added


def _generate_candidate() -> np.ndarray | None:
    row_sizes, col_sizes = _choose_layout()
    row_lines = _build_lines(row_sizes)
    col_lines = _build_lines(col_sizes)
    regions = _initial_region_data(row_lines, col_lines)

    gaps: list[dict] = []
    used: set[tuple[str, int, int]] = set()
    chain_added = False
    if random.random() < CHAIN_PROB:
        chain_added = _try_add_chain(row_lines, col_lines, regions, used, gaps)

    active_added, inactive_added = _add_more_gaps(row_lines, col_lines, regions, used, gaps)
    if not chain_added and active_added == 0:
        return None

    inp = _render_input(row_lines, col_lines, regions, gaps)
    expected = _oracle_from_metadata(inp, regions, gaps)
    local = _local_only(inp, regions)

    if not np.any((local == GREY) & (_colored_array(expected))):
        return None
    if not np.any(expected == GREY):
        return None
    if len(gaps) == 0:
        return None
    if inactive_added == 0 and random.random() < 0.35:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def _colored_array(g: np.ndarray) -> np.ndarray:
    return (g != BLACK) & (g != GREY) & (g != MAROON)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
