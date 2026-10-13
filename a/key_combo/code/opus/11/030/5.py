import math

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
LINE_COLORS = (1, 2, 3, 6, 7, 8, 9)
STEP_VALUES = (2, 3, 4)
ROW_MAJOR_DIRECTIONS = ((0, 1), (1, -1), (1, 0), (1, 1))
PATTERN_TEMPLATES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
)
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _coord_in_box(coord: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = coord
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _apply_transform(coords: np.ndarray, rotations: int, flip: bool) -> np.ndarray:
    pts = coords.copy()
    for _ in range(rotations % 4):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    if flip:
        pts[:, 1] *= -1
    pts -= pts.min(axis=0)
    return pts


def _coords_to_mask(coords: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    sr, sc = shift
    for r, c in coords:
        mask[int(r) + sr, int(c) + sc] = True
    return mask


def _sample_patterns(rng: np.random.Generator) -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()
    template_order = rng.permutation(len(PATTERN_TEMPLATES))
    for idx in template_order:
        base = np.array(PATTERN_TEMPLATES[int(idx)], dtype=int)
        for _ in range(16):
            pts = _apply_transform(base, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
            h = int(pts[:, 0].max()) + 1
            w = int(pts[:, 1].max()) + 1
            sr = int(rng.integers(0, 4 - h))
            sc = int(rng.integers(0, 4 - w))
            mask = _coords_to_mask(pts, (sr, sc))
            key = mask.tobytes()
            if key not in seen:
                seen.add(key)
                patterns.append(mask)
                break
        if len(patterns) == 4:
            break
    if len(patterns) != 4:
        raise RuntimeError("failed to sample four distinct patterns")
    return patterns


def _paint_library(
    grid: np.ndarray,
    patterns: list[np.ndarray],
    orientation: str,
    start: int,
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    if orientation == "bottom":
        r0 = h - 3
        c0 = start
        for i, mask in enumerate(patterns):
            cc = c0 + 4 * i
            patch = np.where(mask, YELLOW, BLACK)
            grid[r0 : r0 + 3, cc : cc + 3] = patch
            if i < 3:
                grid[r0 : r0 + 3, cc + 3] = GREY
        return (r0, c0, r0 + 2, c0 + 14)

    c0 = w - 3
    r0 = start
    for i, mask in enumerate(patterns):
        rr = r0 + 4 * i
        patch = np.where(mask, YELLOW, BLACK)
        grid[rr : rr + 3, c0 : c0 + 3] = patch
        if i < 3:
            grid[rr + 3, c0 : c0 + 3] = GREY
    return (r0, c0, r0 + 14, c0 + 2)


def _extract_library(grid: np.ndarray) -> tuple[str, list[np.ndarray], tuple[int, int, int, int]]:
    h, w = grid.shape

    bottom_grey_cols = [c for c in range(w) if np.all(grid[h - 3 : h, c] == GREY)]
    right_grey_rows = [r for r in range(h) if np.all(grid[r, w - 3 : w] == GREY)]

    is_bottom = len(bottom_grey_cols) == 3 and np.all(np.diff(bottom_grey_cols) == 4)
    is_right = len(right_grey_rows) == 3 and np.all(np.diff(right_grey_rows) == 4)
    assert is_bottom ^ is_right

    patterns: list[np.ndarray] = []
    if is_bottom:
        c0 = bottom_grey_cols[0] - 3
        assert 0 <= c0 and c0 + 14 < w
        r0 = h - 3
        for i in range(4):
            cc = c0 + 4 * i
            block = grid[r0 : r0 + 3, cc : cc + 3]
            assert block.shape == (3, 3)
            assert np.all(np.isin(block, (BLACK, YELLOW)))
            patterns.append(block == YELLOW)
            if i < 3:
                assert np.all(grid[r0 : r0 + 3, cc + 3] == GREY)
        return "bottom", patterns, (r0, c0, r0 + 2, c0 + 14)

    r0 = right_grey_rows[0] - 3
    assert 0 <= r0 and r0 + 14 < h
    c0 = w - 3
    for i in range(4):
        rr = r0 + 4 * i
        block = grid[rr : rr + 3, c0 : c0 + 3]
        assert block.shape == (3, 3)
        assert np.all(np.isin(block, (BLACK, YELLOW)))
        patterns.append(block == YELLOW)
        if i < 3:
            assert np.all(grid[rr + 3, c0 : c0 + 3] == GREY)
    return "right", patterns, (r0, c0, r0 + 14, c0 + 2)


def _find_lines(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BLACK, YELLOW, GREY)]
    assert 2 <= len(colors) <= 3
    lines = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (3, 2)
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
        d1 = coords[1] - coords[0]
        d2 = coords[2] - coords[1]
        assert np.array_equal(d1, d2)
        dr, dc = map(int, d1)
        assert dr != 0 or dc != 0
        step = math.gcd(abs(dr), abs(dc))
        assert step in STEP_VALUES
        center = tuple(map(int, coords[2] + d1))
        box = (center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1)
        lines.append(
            {
                "color": color,
                "coords": coords,
                "delta": (dr, dc),
                "step": step,
                "center": center,
                "box": box,
            }
        )
    return sorted(lines, key=lambda item: item["color"])


def _overlay_stamp(out: np.ndarray, pattern: np.ndarray, center: tuple[int, int]) -> None:
    r0 = center[0] - 1
    c0 = center[1] - 1
    for rr in range(3):
        for cc in range(3):
            if not pattern[rr, cc]:
                continue
            r = r0 + rr
            c = c0 + cc
            if out[r, c] == BLACK:
                out[r, c] = YELLOW


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    patterns_orientation, patterns, library_box = _extract_library(g)
    line_infos = _find_lines(g)

    out = g.copy()
    collision_count = 0
    additions = 0
    for info in line_infos:
        pattern = patterns[info["step"] - 1]
        r0 = info["center"][0] - 1
        c0 = info["center"][1] - 1
        assert 0 <= r0 and 0 <= c0 and r0 + 2 < g.shape[0] and c0 + 2 < g.shape[1]
        patch = g[r0 : r0 + 3, c0 : c0 + 3]
        collision_count += int(np.count_nonzero(pattern & (patch != BLACK)))
        before = int(np.count_nonzero(out == YELLOW))
        _overlay_stamp(out, pattern, info["center"])
        after = int(np.count_nonzero(out == YELLOW))
        additions += after - before

    meta = {
        "orientation": patterns_orientation,
        "patterns": patterns,
        "library_box": library_box,
        "lines": line_infos,
        "collision_count": collision_count,
        "additions": additions,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect arithmetic marker lines, use each line's step as a pattern index, and stamp that pattern one step beyond the last marker.

    Input structure:
    - The grid has a black background with 2 or 3 arithmetic lines, each made of exactly 3 same-color markers.
    - Marker groups use distinct non-yellow, non-grey colors; the 3 markers of a group are equally spaced and collinear.
    - A library of four 3x3 yellow patterns sits either on the bottom edge or on the right edge, with grey separators between entries.

    Rule:
    - Read the library in order: left-to-right for a bottom library, top-to-bottom for a right-side library.
    - For each marker color, sort its 3 cells in row-major order, take the repeated delta between consecutive markers, and let k be that delta's lattice step length.
    - Select library pattern k, then place its 3x3 frame with center one more equal step beyond the third marker; copy only its yellow cells and only onto black cells.
    - Keep the original markers and the pattern library unchanged.

    Key insight:
    - The line geometry encodes two independent clues: the equal spacing chooses which library entry to use, and the extrapolated line direction chooses where that entry must be stamped.
    """
    out, _ = _solve(grid)
    return out


def _candidate_target_cells(pattern: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int]]:
    cells = []
    r0 = center[0] - 1
    c0 = center[1] - 1
    for rr in range(3):
        for cc in range(3):
            if pattern[rr, cc]:
                cells.append((r0 + rr, c0 + cc))
    return cells


def _play_limits(shape: tuple[int, int], orientation: str) -> tuple[int, int]:
    h, w = shape
    max_r = h - 5 if orientation == "bottom" else h - 2
    max_c = w - 5 if orientation == "right" else w - 2
    return max_r, max_c


def _place_line(
    grid: np.ndarray,
    step: int,
    color: int,
    patch_boxes: list[tuple[int, int, int, int]],
    orientation: str,
    rng: np.random.Generator,
    required_marker: tuple[tuple[int, int], int] | None = None,
) -> dict | None:
    h, w = grid.shape
    max_r, max_c = _play_limits(grid.shape, orientation)
    directions = list(ROW_MAJOR_DIRECTIONS)
    rng.shuffle(directions)

    for dr_unit, dc_unit in directions:
        dr = dr_unit * step
        dc = dc_unit * step

        if required_marker is None:
            centers = []
            for _ in range(70):
                center = (int(rng.integers(1, max_r + 1)), int(rng.integers(1, max_c + 1)))
                centers.append(center)
        else:
            target, idx = required_marker
            multiplier = 3 - idx
            centers = [(target[0] + multiplier * dr, target[1] + multiplier * dc)]

        for center in centers:
            r, c = center
            box = (r - 1, c - 1, r + 1, c + 1)
            if not (0 <= box[0] and 0 <= box[1] and box[2] < h and box[3] < w):
                continue
            if box[2] > max_r or box[3] > max_c:
                continue
            if any(_boxes_overlap(box, prev) for prev in patch_boxes):
                continue

            markers = [(r - 3 * dr, c - 3 * dc), (r - 2 * dr, c - 2 * dc), (r - dr, c - dc)]
            if not all(_in_bounds(grid.shape, mr, mc) for mr, mc in markers):
                continue
            if not all(mr <= max_r and mc <= max_c for mr, mc in markers):
                continue

            collision_hits = 0
            valid = True
            for mi, marker in enumerate(markers):
                if grid[marker] != BLACK:
                    valid = False
                    break
                in_existing_patch = any(_coord_in_box(marker, prev) for prev in patch_boxes)
                if required_marker is None:
                    if in_existing_patch:
                        valid = False
                        break
                else:
                    if mi == required_marker[1]:
                        if marker != required_marker[0] or not in_existing_patch:
                            valid = False
                            break
                        collision_hits += 1
                    elif in_existing_patch:
                        valid = False
                        break

            if not valid:
                continue
            if required_marker is not None and collision_hits != 1:
                continue

            for mr, mc in markers:
                grid[mr, mc] = color
            return {
                "color": color,
                "step": step,
                "delta": (dr, dc),
                "center": center,
                "box": box,
                "markers": markers,
            }
    return None


def _validate_generated_case(grid: np.ndarray, expect_collision: bool | None = None) -> dict:
    out, meta = _solve(grid)
    assert not np.array_equal(grid, out)
    assert meta["orientation"] in ("bottom", "right")
    assert 2 <= len(meta["lines"]) <= 3
    used_steps = [info["step"] for info in meta["lines"]]
    assert len(set(used_steps)) == len(used_steps)
    assert all(step in STEP_VALUES for step in used_steps)
    assert len(meta["patterns"]) == 4
    pattern_keys = {pat.tobytes() for pat in meta["patterns"]}
    assert len(pattern_keys) == 4
    for line in meta["lines"]:
        assert not _boxes_overlap(line["box"], meta["library_box"])
        assert 0 <= line["box"][0] and 0 <= line["box"][1]
        assert line["box"][2] < grid.shape[0] and line["box"][3] < grid.shape[1]

    for i, a in enumerate(meta["lines"]):
        for b in meta["lines"][i + 1 :]:
            assert not _boxes_overlap(a["box"], b["box"])

    assert meta["additions"] > 0

    for line in meta["lines"]:
        pattern = meta["patterns"][line["step"] - 1]
        wrong_options = [meta["patterns"][j] for j in range(4) if (j + 1) != line["step"]]
        box = line["box"]
        visible = grid[box[0] : box[2] + 1, box[1] : box[3] + 1] == BLACK
        assert np.any(pattern & visible)
        assert any(np.any((pattern != wrong) & visible) for wrong in wrong_options)

        dr, dc = line["delta"]
        first = tuple(map(int, line["coords"][0]))
        reverse_center = (first[0] - dr, first[1] - dc)
        same_pattern = pattern
        if 1 <= reverse_center[0] < grid.shape[0] - 1 and 1 <= reverse_center[1] < grid.shape[1] - 1:
            reverse_cells = {
                (reverse_center[0] - 1 + rr, reverse_center[1] - 1 + cc)
                for rr in range(3)
                for cc in range(3)
                if same_pattern[rr, cc]
            }
            forward_cells = set(_candidate_target_cells(pattern, line["center"]))
            assert reverse_cells != forward_cells

    if expect_collision is True:
        assert meta["collision_count"] > 0
    if expect_collision is False:
        assert meta["collision_count"] == 0

    meta["output"] = out
    return meta


def _generate_attempt(rng: np.random.Generator) -> tuple[np.ndarray, bool] | None:
    orientation = "bottom" if bool(rng.integers(0, 2)) else "right"
    h = int(rng.integers(18, 25))
    w = int(rng.integers(18, 25))
    if orientation == "bottom" and w < 16:
        return None
    if orientation == "right" and h < 16:
        return None

    grid = np.zeros((h, w), dtype=int)
    patterns = _sample_patterns(rng)
    if orientation == "bottom":
        start = int(rng.integers(0, w - 14))
    else:
        start = int(rng.integers(0, h - 14))
    _paint_library(grid, patterns, orientation, start)

    n_lines = 2 if bool(rng.integers(0, 2)) else 3
    steps = list(rng.choice(np.array(STEP_VALUES, dtype=int), size=n_lines, replace=False))
    rng.shuffle(steps)
    colors = list(rng.choice(np.array(LINE_COLORS, dtype=int), size=n_lines, replace=False))

    force_collision = n_lines >= 2 and rng.random() < 0.45
    line_records: list[dict] = []
    patch_boxes: list[tuple[int, int, int, int]] = []

    first = _place_line(grid, int(steps[0]), int(colors[0]), patch_boxes, orientation, rng)
    if first is None:
        return None
    line_records.append(first)
    patch_boxes.append(first["box"])

    second_index = 1
    if force_collision:
        target_cells = _candidate_target_cells(patterns[first["step"] - 1], first["center"])
        rng.shuffle(target_cells)
        placed = None
        for target in target_cells:
            marker_idx_order = list(range(3))
            rng.shuffle(marker_idx_order)
            for marker_idx in marker_idx_order:
                placed = _place_line(
                    grid,
                    int(steps[second_index]),
                    int(colors[second_index]),
                    patch_boxes,
                    orientation,
                    rng,
                    required_marker=(target, marker_idx),
                )
                if placed is not None:
                    break
            if placed is not None:
                break
        if placed is None:
            return None
        line_records.append(placed)
        patch_boxes.append(placed["box"])
        second_index = 2

    while second_index < n_lines:
        placed = _place_line(grid, int(steps[second_index]), int(colors[second_index]), patch_boxes, orientation, rng)
        if placed is None:
            return None
        line_records.append(placed)
        patch_boxes.append(placed["box"])
        second_index += 1

    for record in line_records:
        record["pattern"] = patterns[record["step"] - 1]

    try:
        meta = _validate_generated_case(grid, expect_collision=True if force_collision else None)
    except AssertionError:
        return None

    if force_collision and meta["collision_count"] == 0:
        return None
    return grid, force_collision


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        result = _generate_attempt(rng)
        if result is None:
            continue
        grid, _ = result
        return grid
    raise RuntimeError("failed to generate a valid arithmetic-stamp input")
