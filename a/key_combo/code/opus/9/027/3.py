from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
ORANGE = 7
FOREGROUND = {BLUE, GREEN, ORANGE}
CORNERS = ("tl", "tr", "bl", "br")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    fg = grid != BLACK
    seen = np.zeros_like(fg, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _corner_from_center(center_r: float, center_c: float, h: int, w: int) -> str:
    top = center_r < h / 2.0
    left = center_c < w / 2.0
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _extract_corner_data(grid: np.ndarray) -> dict[str, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components(g)
    assert len(comps) == 4, "expected four corner components"

    out: dict[str, dict] = {}
    h, w = g.shape
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        corner = _corner_from_center(center_r, center_c, h, w)
        assert corner not in out, "components must occupy distinct quadrants"

        patch = _crop_bbox(g, (r0, c0, r1, c1))
        data = {
            "coords": coords,
            "bbox": (r0, c0, r1, c1),
            "patch": patch,
            "height": r1 - r0 + 1,
            "width": c1 - c0 + 1,
            "orange_local": [(rr - r0, cc - c0) for rr, cc in coords if int(g[rr, cc]) == ORANGE],
        }
        if corner == "tl":
            data["top_offset"] = r0
            data["left_offset"] = c0
        elif corner == "tr":
            data["top_offset"] = r0
            data["right_offset"] = w - 1 - c1
        elif corner == "bl":
            data["bottom_offset"] = h - 1 - r1
            data["left_offset"] = c0
        else:
            data["bottom_offset"] = h - 1 - r1
            data["right_offset"] = w - 1 - c1
        out[corner] = data

    assert set(out) == set(CORNERS)
    return out


def _placement_for_corner(info: dict[str, dict], corner: str, max_h: int, max_w: int) -> tuple[int, int]:
    item = info[corner]
    ph = int(item["height"])
    pw = int(item["width"])
    if corner == "tl":
        return int(item["top_offset"]), int(item["left_offset"])
    if corner == "tr":
        return int(item["top_offset"]), 2 * max_w + 1 - int(item["right_offset"]) - pw
    if corner == "bl":
        return 2 * max_h + 1 - int(item["bottom_offset"]) - ph, int(item["left_offset"])
    return 2 * max_h + 1 - int(item["bottom_offset"]) - ph, 2 * max_w + 1 - int(item["right_offset"]) - pw


def _paste_nonblack(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _connect_orange(out: np.ndarray) -> None:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in np.argwhere(out == ORANGE):
        rows[int(r)].append(int(c))
        cols[int(c)].append(int(r))

    for r, cs in rows.items():
        cs = sorted(cs)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                c0, c1 = cs[i], cs[j]
                for c in range(c0 + 1, c1):
                    if out[r, c] == BLACK:
                        out[r, c] = ORANGE
    for c, rs in cols.items():
        rs = sorted(rs)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                r0, r1 = rs[i], rs[j]
                for r in range(r0 + 1, r1):
                    if out[r, c] == BLACK:
                        out[r, c] = ORANGE


def _contract_without_lines(grid: np.ndarray) -> tuple[np.ndarray, dict[str, dict], int, int]:
    info = _extract_corner_data(grid)
    max_h = max(int(item["height"]) for item in info.values())
    max_w = max(int(item["width"]) for item in info.values())
    out = np.zeros((2 * max_h + 1, 2 * max_w + 1), dtype=int)
    for corner in CORNERS:
        r0, c0 = _placement_for_corner(info, corner, max_h, max_w)
        _paste_nonblack(out, info[corner]["patch"], r0, c0)
    return out, info, max_h, max_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink the four corner patterns into a tight cross-separated canvas and connect aligned orange markers there.

    Input structure:
    - A black background contains exactly one 4-connected multicolor pattern in each corner quadrant.
    - Each corner pattern uses blue and green cells, optionally with orange marker cells embedded in the same component.
    - The middle of the input is empty, so the four corner components are isolated from one another.

    Rule:
    - Compute the minimal bounding box of each corner component and take the maximum component height and width.
    - Create a black output canvas of shape `(2*max_h+1, 2*max_w+1)` whose middle row and middle column remain black separators.
    - Copy each corner component's bbox crop into the matching output quadrant while preserving its offset from the same outer edges of the corner region.
    - After all four crops are placed, inspect every pair of orange cells in the shrunken canvas; whenever a pair shares a row or a column, fill the black cells strictly between them with orange, leaving existing non-black cells unchanged.

    Key insight:
    - The task is solved in the contracted corner-coordinate system: first normalize the four corner bboxes onto one minimal canvas, then draw row/column connections only between orange markers in that reduced layout.
    """
    contracted, _info, _max_h, _max_w = _contract_without_lines(np.asarray(grid, dtype=int))
    out = contracted.copy()
    _connect_orange(out)
    return out


def _coord_ranges(spec: dict, corner: str) -> tuple[list[int], list[int]]:
    start_r = int(spec["placements"][corner][0])
    start_c = int(spec["placements"][corner][1])
    ph = int(spec["corner_specs"][corner]["height"])
    pw = int(spec["corner_specs"][corner]["width"])
    return list(range(start_r, start_r + ph)), list(range(start_c, start_c + pw))


def _rand_from_intersection(a: list[int], b: list[int]) -> int | None:
    vals = sorted(set(a) & set(b))
    if not vals:
        return None
    return random.choice(vals)


def _sample_marker_plan(spec: dict, mode: str) -> dict[str, list[tuple[int, int]]] | None:
    ranges = {corner: _coord_ranges(spec, corner) for corner in CORNERS}
    planned_abs = {corner: [] for corner in CORNERS}

    def choose_col(corner: str, banned: set[int]) -> int | None:
        cols = [c for c in ranges[corner][1] if c not in banned]
        return random.choice(cols) if cols else None

    def choose_row(corner: str, banned: set[int]) -> int | None:
        rows = [r for r in ranges[corner][0] if r not in banned]
        return random.choice(rows) if rows else None

    if mode == "row_row":
        top_r = _rand_from_intersection(ranges["tl"][0], ranges["tr"][0])
        bot_r = _rand_from_intersection(ranges["bl"][0], ranges["br"][0])
        if top_r is None or bot_r is None:
            return None
        tl_c = choose_col("tl", set())
        tr_c = choose_col("tr", {tl_c} if tl_c is not None else set())
        bl_c = choose_col("bl", {tl_c, tr_c} - {None})
        br_c = choose_col("br", {tl_c, tr_c, bl_c} - {None})
        if None in (tl_c, tr_c, bl_c, br_c):
            return None
        planned_abs["tl"] = [(top_r, tl_c)]
        planned_abs["tr"] = [(top_r, tr_c)]
        planned_abs["bl"] = [(bot_r, bl_c)]
        planned_abs["br"] = [(bot_r, br_c)]
    elif mode == "col_col":
        left_c = _rand_from_intersection(ranges["tl"][1], ranges["bl"][1])
        right_c = _rand_from_intersection(ranges["tr"][1], ranges["br"][1])
        if left_c is None or right_c is None:
            return None
        tl_r = choose_row("tl", set())
        bl_r = choose_row("bl", {tl_r} if tl_r is not None else set())
        tr_r = choose_row("tr", {tl_r, bl_r} - {None})
        br_r = choose_row("br", {tl_r, bl_r, tr_r} - {None})
        if None in (tl_r, bl_r, tr_r, br_r):
            return None
        planned_abs["tl"] = [(tl_r, left_c)]
        planned_abs["bl"] = [(bl_r, left_c)]
        planned_abs["tr"] = [(tr_r, right_c)]
        planned_abs["br"] = [(br_r, right_c)]
    elif mode == "l_top":
        top_r = _rand_from_intersection(ranges["tl"][0], ranges["tr"][0])
        left_c = _rand_from_intersection(ranges["tl"][1], ranges["bl"][1])
        right_c = _rand_from_intersection(ranges["tr"][1], ranges["br"][1])
        if None in (top_r, left_c, right_c):
            return None
        bl_r_candidates = [r for r in ranges["bl"][0] if r != top_r]
        br_r_candidates = [r for r in ranges["br"][0] if r != top_r]
        if not bl_r_candidates or not br_r_candidates:
            return None
        bl_r = random.choice(bl_r_candidates)
        br_r_choices = [r for r in br_r_candidates if r != bl_r]
        if not br_r_choices:
            return None
        br_r = random.choice(br_r_choices)
        planned_abs["tl"] = [(top_r, left_c)]
        planned_abs["tr"] = [(top_r, right_c)]
        planned_abs["bl"] = [(bl_r, left_c)]
        planned_abs["br"] = [(br_r, right_c)]
    elif mode == "l_bottom":
        bot_r = _rand_from_intersection(ranges["bl"][0], ranges["br"][0])
        left_c = _rand_from_intersection(ranges["tl"][1], ranges["bl"][1])
        right_c = _rand_from_intersection(ranges["tr"][1], ranges["br"][1])
        if None in (bot_r, left_c, right_c):
            return None
        tl_r = random.choice(ranges["tl"][0])
        tr_choices = [r for r in ranges["tr"][0] if r != tl_r]
        if not tr_choices:
            return None
        tr_r = random.choice(tr_choices)
        planned_abs["tl"] = [(tl_r, left_c)]
        planned_abs["tr"] = [(tr_r, right_c)]
        planned_abs["bl"] = [(bot_r, left_c)]
        planned_abs["br"] = [(bot_r, right_c)]
    elif mode == "mixed_tr":
        top_r = _rand_from_intersection(ranges["tl"][0], ranges["tr"][0])
        right_c = _rand_from_intersection(ranges["tr"][1], ranges["br"][1])
        if None in (top_r, right_c):
            return None
        tl_cols = [c for c in ranges["tl"][1] if c != right_c]
        br_rows = [r for r in ranges["br"][0] if r != top_r]
        if not tl_cols or not br_rows:
            return None
        tl_c = random.choice(tl_cols)
        br_r = random.choice(br_rows)
        planned_abs["tl"] = [(top_r, tl_c)]
        planned_abs["tr"] = [(top_r, right_c)]
        planned_abs["br"] = [(br_r, right_c)]
    elif mode == "mixed_bl":
        bot_r = _rand_from_intersection(ranges["bl"][0], ranges["br"][0])
        left_c = _rand_from_intersection(ranges["tl"][1], ranges["bl"][1])
        if None in (bot_r, left_c):
            return None
        br_cols = [c for c in ranges["br"][1] if c != left_c]
        tl_rows = [r for r in ranges["tl"][0] if r != bot_r]
        if not br_cols or not tl_rows:
            return None
        tl_r = random.choice(tl_rows)
        br_c = random.choice(br_cols)
        planned_abs["tl"] = [(tl_r, left_c)]
        planned_abs["bl"] = [(bot_r, left_c)]
        planned_abs["br"] = [(bot_r, br_c)]
    else:
        raise ValueError(f"unknown marker-plan mode: {mode}")

    planned_local: dict[str, list[tuple[int, int]]] = {}
    for corner in CORNERS:
        start_r, start_c = spec["placements"][corner]
        local = []
        for r, c in planned_abs[corner]:
            local.append((r - start_r, c - start_c))
        planned_local[corner] = local
    return planned_local


def _make_connected_shape(height: int, width: int, required: list[tuple[int, int]]) -> set[tuple[int, int]] | None:
    req = list(dict.fromkeys(required))
    for _ in range(120):
        cells = set(req)
        if not cells:
            cells.add((random.randrange(height), random.randrange(width)))

        top_candidates = [(0, c) for c in range(width)]
        bot_candidates = [(height - 1, c) for c in range(width)]
        left_candidates = [(r, 0) for r in range(height)]
        right_candidates = [(r, width - 1) for r in range(height)]
        for edge_cells in (top_candidates, bot_candidates, left_candidates, right_candidates):
            if not any(cell in cells for cell in edge_cells):
                cells.add(random.choice(edge_cells))

        pending = list(cells)
        root = pending[0]
        connected = {root}

        def add_path(target: tuple[int, int]) -> None:
            rr, cc = random.choice(tuple(connected))
            tr, tc = target
            if random.random() < 0.5:
                while cc != tc:
                    cc += 1 if tc > cc else -1
                    connected.add((rr, cc))
                while rr != tr:
                    rr += 1 if tr > rr else -1
                    connected.add((rr, cc))
            else:
                while rr != tr:
                    rr += 1 if tr > rr else -1
                    connected.add((rr, cc))
                while cc != tc:
                    cc += 1 if tc > cc else -1
                    connected.add((rr, cc))

        for cell in pending[1:]:
            add_path(cell)

        if len(connected) > 8:
            continue
        while len(connected) < 4:
            frontier = [
                nbr
                for r, c in connected
                for nbr in _neighbors4(r, c)
                if 0 <= nbr[0] < height and 0 <= nbr[1] < width and nbr not in connected
            ]
            if not frontier:
                break
            connected.add(random.choice(frontier))
        if len(connected) < 4:
            continue
        target_area = random.randint(len(connected), 8)
        while len(connected) < target_area:
            frontier = [
                nbr
                for r, c in connected
                for nbr in _neighbors4(r, c)
                if 0 <= nbr[0] < height and 0 <= nbr[1] < width and nbr not in connected
            ]
            if not frontier:
                break
            connected.add(random.choice(frontier))

        rows = [r for r, _ in connected]
        cols = [c for _, c in connected]
        if min(rows) != 0 or max(rows) != height - 1 or min(cols) != 0 or max(cols) != width - 1:
            continue
        if not set(req).issubset(connected):
            continue
        if len(connected) == height * width and height * width > 1:
            continue
        if 4 <= len(connected) <= 8:
            return connected
    return None


def _color_patch(height: int, width: int, cells: set[tuple[int, int]], markers: list[tuple[int, int]]) -> np.ndarray | None:
    patch = np.zeros((height, width), dtype=int)
    marker_set = set(markers)
    base_cells = sorted(cells - marker_set)
    if len(base_cells) < 2:
        return None
    random.shuffle(base_cells)
    colors = {base_cells[0]: BLUE, base_cells[1]: GREEN}
    for cell in base_cells[2:]:
        colors[cell] = random.choice((BLUE, GREEN))
    for (r, c), color in colors.items():
        patch[r, c] = color
    for r, c in marker_set:
        patch[r, c] = ORANGE
    return patch


def _count_alignment_pairs(grid: np.ndarray) -> int:
    oranges = [tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)]
    count = 0
    for i, (r1, c1) in enumerate(oranges):
        for r2, c2 in oranges[i + 1 :]:
            if r1 == r2 or c1 == c2:
                count += 1
    return count


def _try_add_extra_marker(spec: dict, patches: dict[str, np.ndarray], markers: dict[str, list[tuple[int, int]]]) -> None:
    existing_abs = []
    for corner, local_cells in markers.items():
        start_r, start_c = spec["placements"][corner]
        for r, c in local_cells:
            existing_abs.append((start_r + r, start_c + c))

    row_counts = defaultdict(int)
    col_counts = defaultdict(int)
    for r, c in existing_abs:
        row_counts[r] += 1
        col_counts[c] += 1

    candidate_corners = [corner for corner in CORNERS if len(markers[corner]) < 2]
    random.shuffle(candidate_corners)
    for corner in candidate_corners:
        patch = patches[corner]
        start_r, start_c = spec["placements"][corner]
        local_existing = set(markers[corner])
        candidates = []
        for r, c in np.argwhere(patch != BLACK):
            cell = (int(r), int(c))
            if cell in local_existing:
                continue
            if any(r == rr or c == cc for rr, cc in local_existing):
                continue
            abs_r, abs_c = start_r + int(r), start_c + int(c)
            if row_counts[abs_r] or col_counts[abs_c]:
                continue
            candidates.append(cell)
        if candidates:
            markers[corner].append(random.choice(candidates))
            return


def _build_input_from_spec(spec: dict) -> np.ndarray:
    grid = np.zeros((spec["input_h"], spec["input_w"]), dtype=int)
    for corner in CORNERS:
        patch = spec["patches"][corner]
        start_r, start_c = spec["input_starts"][corner]
        h, w = patch.shape
        grid[start_r : start_r + h, start_c : start_c + w][patch != BLACK] = patch[patch != BLACK]
    return grid


def _build_oracle_from_spec(spec: dict) -> np.ndarray:
    out = np.zeros((2 * spec["max_h"] + 1, 2 * spec["max_w"] + 1), dtype=int)
    for corner in CORNERS:
        patch = spec["patches"][corner]
        start_r, start_c = spec["placements"][corner]
        out[start_r : start_r + patch.shape[0], start_c : start_c + patch.shape[1]][patch != BLACK] = patch[patch != BLACK]
    expected = out.copy()
    _connect_orange(expected)
    return expected


def _make_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    max_h = random.randint(3, 5)
    max_w = random.randint(3, 5)
    min_input_h = max(20, 2 * max_h + 10)
    min_input_w = max(20, 2 * max_w + 10)
    input_h = random.randint(min_input_h, 24)
    input_w = random.randint(min_input_w, 24)

    corner_specs = {}
    for corner in CORNERS:
        ph = random.randint(2, max_h)
        pw = random.randint(2, max_w)
        if corner in ("tl", "tr"):
            row_offset = random.randint(0, max_h - ph)
            corner_specs.setdefault(corner, {})["top_offset"] = row_offset
        else:
            bottom_offset = random.randint(0, max_h - ph)
            corner_specs.setdefault(corner, {})["bottom_offset"] = bottom_offset
        if corner in ("tl", "bl"):
            col_offset = random.randint(0, max_w - pw)
            corner_specs.setdefault(corner, {})["left_offset"] = col_offset
        else:
            right_offset = random.randint(0, max_w - pw)
            corner_specs.setdefault(corner, {})["right_offset"] = right_offset
        corner_specs[corner]["height"] = ph
        corner_specs[corner]["width"] = pw

    placements = {
        "tl": (corner_specs["tl"]["top_offset"], corner_specs["tl"]["left_offset"]),
        "tr": (corner_specs["tr"]["top_offset"], 2 * max_w + 1 - corner_specs["tr"]["right_offset"] - corner_specs["tr"]["width"]),
        "bl": (2 * max_h + 1 - corner_specs["bl"]["bottom_offset"] - corner_specs["bl"]["height"], corner_specs["bl"]["left_offset"]),
        "br": (2 * max_h + 1 - corner_specs["br"]["bottom_offset"] - corner_specs["br"]["height"], 2 * max_w + 1 - corner_specs["br"]["right_offset"] - corner_specs["br"]["width"]),
    }
    input_starts = {
        "tl": (corner_specs["tl"]["top_offset"], corner_specs["tl"]["left_offset"]),
        "tr": (corner_specs["tr"]["top_offset"], input_w - corner_specs["tr"]["right_offset"] - corner_specs["tr"]["width"]),
        "bl": (input_h - corner_specs["bl"]["bottom_offset"] - corner_specs["bl"]["height"], corner_specs["bl"]["left_offset"]),
        "br": (input_h - corner_specs["br"]["bottom_offset"] - corner_specs["br"]["height"], input_w - corner_specs["br"]["right_offset"] - corner_specs["br"]["width"]),
    }

    spec = {
        "max_h": max_h,
        "max_w": max_w,
        "input_h": input_h,
        "input_w": input_w,
        "corner_specs": corner_specs,
        "placements": placements,
        "input_starts": input_starts,
    }

    mode = random.choice(("row_row", "col_col", "l_top", "l_bottom", "mixed_tr", "mixed_bl"))
    markers = _sample_marker_plan(spec, mode)
    if markers is None:
        return None

    patches: dict[str, np.ndarray] = {}
    for corner in CORNERS:
        ph = corner_specs[corner]["height"]
        pw = corner_specs[corner]["width"]
        shape = _make_connected_shape(ph, pw, markers[corner])
        if shape is None:
            return None
        patch = _color_patch(ph, pw, shape, markers[corner])
        if patch is None:
            return None
        patches[corner] = patch

    if random.random() < 0.45:
        _try_add_extra_marker(spec, patches, markers)
        for corner in CORNERS:
            shape_cells = {tuple(map(int, rc)) for rc in np.argwhere(patches[corner] != BLACK)}
            recolored = _color_patch(patches[corner].shape[0], patches[corner].shape[1], shape_cells, markers[corner])
            if recolored is None:
                return None
            patches[corner] = recolored

    spec["patches"] = patches
    inp = _build_input_from_spec(spec)
    expected = _build_oracle_from_spec(spec)
    if _validate_instance(inp, expected, spec):
        return inp, expected
    return None


def _validate_instance(inp: np.ndarray, expected: np.ndarray, spec: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if inp.shape[0] < 20 or inp.shape[1] < 20:
        return False
    if not np.all(inp[5:-5, 5:-5] == BLACK):
        return False

    comps = _components(inp)
    if len(comps) != 4:
        return False

    quadrant_markers: dict[str, list[tuple[int, int]]] = {}
    for coords in comps:
        colors = {int(inp[r, c]) for r, c in coords}
        if not colors <= FOREGROUND:
            return False
        if BLUE not in colors or GREEN not in colors:
            return False
        if not (4 <= len(coords) <= 8):
            return False
        r0, c0, r1, c1 = _bbox(coords)
        corner = _corner_from_center((r0 + r1) / 2.0, (c0 + c1) / 2.0, inp.shape[0], inp.shape[1])
        if corner in quadrant_markers:
            return False
        oranges = [(r, c) for r, c in coords if int(inp[r, c]) == ORANGE]
        if len(oranges) > 2:
            return False
        if len(oranges) == 2:
            (r1m, c1m), (r2m, c2m) = oranges
            if r1m == r2m or c1m == c2m:
                return False
        quadrant_markers[corner] = oranges

    if set(quadrant_markers) != set(CORNERS):
        return False

    contracted, _info, max_h, max_w = _contract_without_lines(inp)
    if (max_h, max_w) != (spec["max_h"], spec["max_w"]):
        return False
    if contracted.shape != (2 * max_h + 1, 2 * max_w + 1):
        return False

    orange_positions = [tuple(map(int, rc)) for rc in np.argwhere(contracted == ORANGE)]
    pair_count = _count_alignment_pairs(contracted)
    if pair_count not in (2, 3):
        return False

    if not np.array_equal(main(inp), expected):
        return False

    rows = defaultdict(int)
    cols = defaultdict(int)
    for r, c in orange_positions:
        rows[r] += 1
        cols[c] += 1
    if not (any(v >= 2 for v in rows.values()) or any(v >= 2 for v in cols.values())):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _make_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid corner-contraction connector puzzle")
