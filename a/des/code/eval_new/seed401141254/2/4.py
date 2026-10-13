from __future__ import annotations

import numpy as np


BG = 0
CORNER = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
RNG = np.random.default_rng()

RIGHT = (0, 1)
LEFT = (0, -1)
DOWN = (1, 0)
UP = (-1, 0)
BASE_DIRECTIONS = [RIGHT, DOWN, LEFT, DOWN, LEFT, UP]
MAX_GENERATION_ATTEMPTS = 400


def _axis(vec: tuple[int, int]) -> str:
    return "H" if vec[0] == 0 else "V"


def _rotate_cw(vec: tuple[int, int]) -> tuple[int, int]:
    r, c = vec
    return (c, -r)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _label_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _extract_segment(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    length = len(coords)
    assert 3 <= length <= 7

    if len(rows) == 1:
        row = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1))
        ordered = [(row, c) for c in cols]
        orient = "H"
    else:
        assert len(cols) == 1
        col = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1))
        ordered = [(r, col) for r in rows]
        orient = "V"

    first = ordered[0]
    second = ordered[-1]
    first_color = int(grid[first])
    second_color = int(grid[second])
    assert first_color != second_color

    body_cells = ordered[1:-1]
    body_colors = {int(grid[cell]) for cell in body_cells}
    assert len(body_colors) == 1
    body_color = next(iter(body_colors))

    return {
        "coords": ordered,
        "bbox": (ordered[0][0], ordered[0][1]),
        "orient": orient,
        "length": length,
        "first_color": first_color,
        "second_color": second_color,
        "body_color": int(body_color),
    }


def _segment_step(seg: dict, entry_color: int, prev_axis: str | None) -> tuple[int, int]:
    base = RIGHT if seg["orient"] == "H" else DOWN
    step = base if int(seg["first_color"]) == int(entry_color) else (-base[0], -base[1])
    if prev_axis is not None and _axis(step) == prev_axis:
        step = _rotate_cw(step)
    return (int(step[0]), int(step[1]))


def _trace_cycle(segments: list[dict]) -> tuple[list[tuple[list[tuple[int, int]], int]], set[tuple[int, int]]]:
    by_color: dict[int, list[int]] = {}
    for idx, seg in enumerate(segments):
        for color in (int(seg["first_color"]), int(seg["second_color"])):
            by_color.setdefault(color, []).append(idx)

    for ids in by_color.values():
        assert len(ids) == 2
        assert ids[0] != ids[1]

    start_id = min(range(len(segments)), key=lambda i: tuple(map(int, segments[i]["bbox"])))
    start_seg = segments[start_id]
    start_entry = int(start_seg["first_color"])

    used: list[int] = []
    placements: list[tuple[list[tuple[int, int]], int]] = []
    corners: list[tuple[int, int]] = []
    current_id = int(start_id)
    entry_color = int(start_entry)
    entry_pos = (0, 0)
    prev_axis: str | None = None

    for step_idx in range(len(segments)):
        assert current_id not in used
        seg = segments[current_id]
        step = _segment_step(seg, entry_color, prev_axis)
        cells = [
            (int(entry_pos[0] + step[0] * k), int(entry_pos[1] + step[1] * k))
            for k in range(int(seg["length"]))
        ]
        placements.append((cells, int(seg["body_color"])))
        corners.append((int(entry_pos[0]), int(entry_pos[1])))
        used.append(int(current_id))

        exit_pos = cells[-1]
        exit_color = int(seg["second_color"] if entry_color == int(seg["first_color"]) else seg["first_color"])
        prev_axis = _axis(step)
        if step_idx == len(segments) - 1:
            assert exit_color == start_entry
            assert exit_pos == (0, 0)
            break

        a, b = by_color[exit_color]
        current_id = int(b if a == current_id else a)
        entry_color = int(exit_color)
        entry_pos = (int(exit_pos[0]), int(exit_pos[1]))

    assert len(used) == len(segments)
    return placements, set(corners)


def _render_output(shape: tuple[int, int], placements: list[tuple[list[tuple[int, int]], int]], corners: set[tuple[int, int]]) -> np.ndarray:
    all_cells = [cell for cells, _color in placements for cell in cells]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    shifted_placements = []
    for cells, color in placements:
        shifted = [(int(r - min_r + 1), int(c - min_c + 1)) for r, c in cells]
        shifted_placements.append((shifted, int(color)))
    shifted_corners = {(int(r - min_r + 1), int(c - min_c + 1)) for r, c in corners}

    out = np.full(shape, BG, dtype=int)
    for cells, color in shifted_placements:
        for r, c in cells:
            assert 0 <= r < shape[0] and 0 <= c < shape[1]
            out[r, c] = int(color)
    for r, c in shifted_corners:
        out[r, c] = CORNER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered endpoint-coded line fragments into one normalized orthogonal loop and mark each junction maroon.

    Input structure:
    - Black background with several isolated 4-connected straight segments.
    - Every segment is horizontal or vertical, length 3-7, with a uniform body color.
    - The two endpoint cells of each segment have distinct interface colors, and each interface color appears on exactly two segment endpoints overall.

    Rule:
    - Detect every straight fragment and read its two endpoint colors in left-to-right or top-to-bottom order.
    - Starting from the top-leftmost fragment, keep that fragment fixed and follow the unique endpoint-color pairings to attach the remaining fragments one by one.
    - When the next fragment already has the opposite orientation, translate it so the matched endpoints overlap; when it has the same orientation, rotate that fragment clockwise by 90 degrees about the matched endpoint before attaching it.
    - Draw the reconstructed loop on a clean black canvas of the same size, shifting the whole loop to the upper-left, and recolor every shared junction cell to maroon.

    Key insight:
    - Endpoint colors are connector labels: equal endpoint colors identify which fragments meet, while the fragment's own left/right or top/bottom ordering determines how a translated or rotated piece extends away from each junction.
    """
    g = np.asarray(grid, dtype=int)
    comps = _label_components(g != BG)
    segments = [_extract_segment(g, comp) for comp in comps]
    assert len(segments) >= 4
    placements, corners = _trace_cycle(segments)
    return _render_output(g.shape, placements, corners)


def _choices_for_segment(
    desired_dir: tuple[int, int], prev_axis: str, rotate_branch: bool
) -> list[tuple[str, bool]]:
    choices: list[tuple[str, bool]] = []
    for orient in ("H", "V"):
        base = RIGHT if orient == "H" else DOWN
        for first_is_entry in (True, False):
            local = base if first_is_entry else (-base[0], -base[1])
            rotates = _axis(local) == prev_axis
            final = _rotate_cw(local) if rotates else local
            if rotates == rotate_branch and final == desired_dir:
                choices.append((orient, bool(first_is_entry)))
    return choices


def _sample_lengths() -> list[int]:
    while True:
        a = int(RNG.integers(4, 7))
        c = int(RNG.integers(2, a - 1))
        b = int(RNG.integers(2, 5))
        d = int(RNG.integers(2, 5))
        if b + d > 6:
            continue
        lengths = [a + 1, b + 1, c + 1, d + 1, a - c + 1, b + d + 1]
        if all(3 <= x <= 7 for x in lengths):
            return lengths


def _sample_branch_flags(count: int) -> list[bool]:
    while True:
        flags = [bool(x) for x in RNG.integers(0, 2, size=count)]
        if any(flags) and not all(flags):
            return flags


def _sample_colors(n_segments: int) -> tuple[list[int], list[int]]:
    endpoint_colors = list(map(int, RNG.choice(PALETTE, size=n_segments, replace=False)))
    body_colors: list[int] = []
    for idx in range(n_segments):
        banned = {endpoint_colors[idx], endpoint_colors[(idx + 1) % n_segments]}
        if body_colors:
            banned.add(body_colors[-1])
        if idx == n_segments - 1 and body_colors:
            banned.add(body_colors[0])
        choices = [c for c in PALETTE if c not in banned]
        if not choices:
            choices = [c for c in PALETTE if c not in {endpoint_colors[idx], endpoint_colors[(idx + 1) % n_segments]}]
        body_colors.append(int(RNG.choice(choices)))
    return endpoint_colors, body_colors


def _build_segment_specs() -> tuple[list[dict], list[tuple[int, int]], list[int], list[int]]:
    n_segments = len(BASE_DIRECTIONS)
    lengths = _sample_lengths()
    rotate_flags = _sample_branch_flags(n_segments - 1)
    endpoint_colors, body_colors = _sample_colors(n_segments)

    specs: list[dict] = []
    for idx, desired_dir in enumerate(BASE_DIRECTIONS):
        entry_color = int(endpoint_colors[idx])
        exit_color = int(endpoint_colors[(idx + 1) % n_segments])
        body_color = int(body_colors[idx])
        length = int(lengths[idx])

        if idx == 0:
            orient = "H"
            first_color, second_color = entry_color, exit_color
        else:
            prev_axis = _axis(BASE_DIRECTIONS[idx - 1])
            choices = _choices_for_segment(desired_dir, prev_axis, rotate_flags[idx - 1])
            orient, first_is_entry = choices[int(RNG.integers(0, len(choices)))]
            if first_is_entry:
                first_color, second_color = entry_color, exit_color
            else:
                first_color, second_color = exit_color, entry_color

        specs.append(
            {
                "orient": orient,
                "length": length,
                "first_color": int(first_color),
                "second_color": int(second_color),
                "body_color": int(body_color),
            }
        )

    return specs, BASE_DIRECTIONS, lengths, body_colors


def _expected_output(shape: tuple[int, int], directions: list[tuple[int, int]], lengths: list[int], body_colors: list[int]) -> np.ndarray:
    placements: list[tuple[list[tuple[int, int]], int]] = []
    corners: set[tuple[int, int]] = set()
    pos = (0, 0)
    for direction, length, body_color in zip(directions, lengths, body_colors):
        corners.add((int(pos[0]), int(pos[1])))
        cells = [(int(pos[0] + direction[0] * k), int(pos[1] + direction[1] * k)) for k in range(int(length))]
        placements.append((cells, int(body_color)))
        pos = cells[-1]
    assert pos == (0, 0)
    return _render_output(shape, placements, corners)


def _segment_patch(spec: dict) -> np.ndarray:
    if spec["orient"] == "H":
        patch = np.full((1, int(spec["length"])), int(spec["body_color"]), dtype=int)
        patch[0, 0] = int(spec["first_color"])
        patch[0, -1] = int(spec["second_color"])
    else:
        patch = np.full((int(spec["length"]), 1), int(spec["body_color"]), dtype=int)
        patch[0, 0] = int(spec["first_color"])
        patch[-1, 0] = int(spec["second_color"])
    return patch


def _can_place(forbidden: np.ndarray, patch_shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = patch_shape
    if r0 < 0 or c0 < 0 or r0 + h > forbidden.shape[0] or c0 + w > forbidden.shape[1]:
        return False
    return not bool(forbidden[r0 : r0 + h, c0 : c0 + w].any())


def _mark_forbidden(forbidden: np.ndarray, patch_shape: tuple[int, int], r0: int, c0: int) -> None:
    h, w = patch_shape
    rr0 = max(0, int(r0 - 1))
    cc0 = max(0, int(c0 - 1))
    rr1 = min(forbidden.shape[0], int(r0 + h + 1))
    cc1 = min(forbidden.shape[1], int(c0 + w + 1))
    forbidden[rr0:rr1, cc0:cc1] = True


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.full(shape, BG, dtype=int)
    forbidden = np.zeros(shape, dtype=bool)
    positions: list[tuple[int, int]] = []

    start_patch = _segment_patch(specs[0])
    start_rows = range(1, min(4, h - start_patch.shape[0]) + 1)
    start_cols = range(1, min(4, w - start_patch.shape[1]) + 1)
    start_choices = [(r, c) for r in start_rows for c in start_cols if _can_place(forbidden, start_patch.shape, r, c)]
    if not start_choices:
        return None
    start_r, start_c = start_choices[int(RNG.integers(0, len(start_choices)))]
    grid[start_r : start_r + start_patch.shape[0], start_c : start_c + start_patch.shape[1]] = start_patch
    _mark_forbidden(forbidden, start_patch.shape, start_r, start_c)
    positions.append((int(start_r), int(start_c)))

    min_other_row = int(start_r + 4)
    for spec in specs[1:]:
        patch = _segment_patch(spec)
        placed = False
        candidate_rows = np.arange(max(1, min_other_row), max(2, h - patch.shape[0] + 1))
        if len(candidate_rows) == 0:
            return None
        for _ in range(120):
            r0 = int(candidate_rows[int(RNG.integers(0, len(candidate_rows)))])
            c0 = int(RNG.integers(0, w - patch.shape[1] + 1))
            if not _can_place(forbidden, patch.shape, r0, c0):
                continue
            grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
            _mark_forbidden(forbidden, patch.shape, r0, c0)
            positions.append((int(r0), int(c0)))
            placed = True
            break
        if not placed:
            return None

    if min(positions) != positions[0]:
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (int(RNG.integers(20, 29)), int(RNG.integers(20, 29)))
    specs, directions, lengths, body_colors = _build_segment_specs()
    grid = _render_input(specs, shape)
    if grid is None:
        return None
    expected = _expected_output(shape, directions, lengths, body_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid scattered-path instance")
