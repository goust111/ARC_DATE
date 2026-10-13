from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 5
LEFT_FRAME = 8
RIGHT_FRAME = 2
OCCLUDER = 6
SQUARE = 5
INNER = 3
PALETTE = (1, 3, 4, 7)
FRAME_COLORS = (LEFT_FRAME, RIGHT_FRAME)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _row_overlap(r0_a: int, r0_b: int) -> bool:
    return _ranges_overlap(r0_a, r0_a + SQUARE - 1, r0_b, r0_b + SQUARE - 1)


def _rect_gap(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + SQUARE - 1, ac0 + SQUARE - 1
    br1, bc1 = br0 + SQUARE - 1, bc0 + SQUARE - 1
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return int(row_gap), int(col_gap)


def _rects_too_close(a: tuple[int, int], b: tuple[int, int], min_gap: int = 2) -> bool:
    row_gap, col_gap = _rect_gap(a, b)
    return row_gap < min_gap and col_gap < min_gap


def _components_of_color(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_frame_patch(patch: np.ndarray, frame: int) -> bool:
    if patch.shape != (SQUARE, SQUARE):
        return False
    border = np.concatenate((patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]))
    return bool(np.all(border == int(frame)))


def _find_squares(grid: np.ndarray) -> list[dict]:
    squares: list[dict] = []
    for frame in FRAME_COLORS:
        for coords in _components_of_color(grid == int(frame)):
            if int(coords.shape[0]) != 16:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != (SQUARE, SQUARE):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            if not _is_frame_patch(patch, frame):
                continue
            inner = patch[1:-1, 1:-1]
            if int(np.count_nonzero(inner == OCCLUDER)) != 1:
                continue
            squares.append(
                {
                    "frame": int(frame),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )
    squares.sort(key=lambda item: (int(item["r0"]), int(item["c0"]), int(item["frame"])))
    return squares


def _restore_patch(patch: np.ndarray) -> np.ndarray:
    out = np.array(patch, dtype=int, copy=True)
    frame = int(out[0, 0])
    inner = out[1:-1, 1:-1].copy()
    occ = np.argwhere(inner == OCCLUDER)
    assert occ.shape == (1, 2), "expected exactly one occluded interior pixel"
    rr, cc = map(int, occ[0])
    if frame == LEFT_FRAME:
        mirror_r, mirror_c = rr, INNER - 1 - cc
    else:
        mirror_r, mirror_c = INNER - 1 - rr, cc
    assert (mirror_r, mirror_c) != (rr, cc), "occluder cannot lie on the active symmetry axis"
    value = int(inner[mirror_r, mirror_c])
    assert value != OCCLUDER, "mirror source must be visible"
    inner[rr, cc] = value
    out[1:-1, 1:-1] = inner
    return out


def _place_side(out: np.ndarray, squares: list[dict], *, frame: int) -> None:
    h, w = out.shape
    if frame == LEFT_FRAME:
        ordered = sorted(squares, key=lambda item: int(item["c0"]))
        edge_col = 0
        inner_col = SQUARE
    else:
        ordered = sorted(squares, key=lambda item: -int(item["c0"]))
        edge_col = int(w - SQUARE)
        inner_col = int(w - 2 * SQUARE)

    edge_rows: list[tuple[int, int]] = []
    for square in ordered:
        r0 = int(square["r0"])
        r1 = r0 + SQUARE - 1
        has_overlap = any(_ranges_overlap(r0, r1, er0, er1) for er0, er1 in edge_rows)
        dest_c = inner_col if has_overlap else edge_col
        out[r0 : r0 + SQUARE, dest_c : dest_c + SQUARE] = _restore_patch(square["patch"])
        if dest_c == edge_col:
            edge_rows.append((r0, r1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered squares to the side implied by frame color, then recover each hidden interior pixel from the symmetry tied to that movement.

    Input structure:
    - The grid has a grey background and 3 to 5 separate 5x5 framed squares.
    - Each square has a 1-cell frame in teal or red and a 3x3 multicolor interior with exactly one magenta occluder.
    - Teal-framed squares contain a vertically symmetric interior pattern; red-framed squares contain a horizontally symmetric interior pattern.

    Rule:
    - Move every teal-framed square to the left side and every red-framed square to the right side, keeping rows fixed.
    - On each side, process squares nearest to that side first; if a later square overlaps the row-span of an already placed edge square of the same frame color, place it one square-width inward instead.
    - Clear the original locations and replace each magenta occluder by copying the color from its mirror partner across the symmetry axis associated with that moved square.

    Key insight:
    - The border color determines movement direction, and that destination side in turn selects which single-axis mirror should be used to reconstruct the hidden interior cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    out = np.full_like(g, bg)

    squares = _find_squares(g)
    assert squares, "expected at least one framed square"

    _place_side(out, [sq for sq in squares if int(sq["frame"]) == LEFT_FRAME], frame=LEFT_FRAME)
    _place_side(out, [sq for sq in squares if int(sq["frame"]) == RIGHT_FRAME], frame=RIGHT_FRAME)
    return out


def _is_vertical_symmetric(inner: np.ndarray) -> bool:
    return bool(np.array_equal(inner, np.fliplr(inner)))


def _is_horizontal_symmetric(inner: np.ndarray) -> bool:
    return bool(np.array_equal(inner, np.flipud(inner)))


def _gen_vertical_inner() -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(200):
        palette = random.sample(PALETTE, k=random.choice((2, 3)))
        top_side = random.choice(palette)
        bottom_side = random.choice([color for color in palette if color != top_side])
        inner = np.array(
            [
                [top_side, random.choice(palette), top_side],
                [random.choice(palette), random.choice(palette), random.choice(palette)],
                [bottom_side, random.choice(palette), bottom_side],
            ],
            dtype=int,
        )
        used = set(map(int, inner.flat))
        if not (2 <= len(used) <= 3):
            continue
        if not _is_vertical_symmetric(inner):
            continue
        if _is_horizontal_symmetric(inner):
            continue
        occ = random.choice(((0, 0), (0, 2), (2, 0), (2, 2)))
        rr, cc = occ
        if int(inner[rr, INNER - 1 - cc]) == int(inner[INNER - 1 - rr, cc]):
            continue
        return inner, occ
    raise RuntimeError("failed to build a vertical-only interior")


def _gen_horizontal_inner() -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(200):
        palette = random.sample(PALETTE, k=random.choice((2, 3)))
        left_side = random.choice(palette)
        right_side = random.choice([color for color in palette if color != left_side])
        inner = np.array(
            [
                [left_side, random.choice(palette), right_side],
                [random.choice(palette), random.choice(palette), random.choice(palette)],
                [left_side, random.choice(palette), right_side],
            ],
            dtype=int,
        )
        used = set(map(int, inner.flat))
        if not (2 <= len(used) <= 3):
            continue
        if not _is_horizontal_symmetric(inner):
            continue
        if _is_vertical_symmetric(inner):
            continue
        occ = random.choice(((0, 0), (0, 2), (2, 0), (2, 2)))
        rr, cc = occ
        if int(inner[INNER - 1 - rr, cc]) == int(inner[rr, INNER - 1 - cc]):
            continue
        return inner, occ
    raise RuntimeError("failed to build a horizontal-only interior")


def _make_patch(frame: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if frame == LEFT_FRAME:
        inner, occ = _gen_vertical_inner()
    else:
        inner, occ = _gen_horizontal_inner()
    patch = np.full((SQUARE, SQUARE), int(frame), dtype=int)
    occluded = inner.copy()
    occluded[occ] = OCCLUDER
    patch[1:-1, 1:-1] = occluded
    return patch, inner, occ


def _sample_edge_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(0, height - SQUARE + 1))
    random.shuffle(candidates)
    rows: list[int] = []
    for row in candidates:
        if all(not _row_overlap(row, other) for other in rows):
            rows.append(int(row))
            if len(rows) == count:
                return rows
    return None


def _sample_side_rows(height: int, count: int) -> tuple[list[int], list[int]] | None:
    if count == 1:
        rows = _sample_edge_rows(height, 1)
        if rows is None:
            return None
        return rows, []

    edge_count = count - 1
    edge_rows = _sample_edge_rows(height, edge_count)
    if edge_rows is None:
        return None

    inward_rows: list[int] = []
    for _ in range(200):
        base = random.choice(edge_rows)
        delta_choices = [delta for delta in (-4, -3, -2, -1, 1, 2, 3, 4)]
        random.shuffle(delta_choices)
        picked = None
        for delta in delta_choices:
            row = int(base + delta)
            if not (0 <= row <= height - SQUARE):
                continue
            if not any(_row_overlap(row, edge) for edge in edge_rows):
                continue
            picked = row
            break
        if picked is not None:
            inward_rows.append(picked)
            return edge_rows, inward_rows
    return None


def _sample_processing_order(entries: list[dict], ascending: bool) -> list[dict]:
    edges = [entry for entry in entries if bool(entry["is_edge"])]
    inwards = [entry for entry in entries if not bool(entry["is_edge"])]
    random.shuffle(edges)
    order = edges[:]
    random.shuffle(inwards)
    for inward in inwards:
        deps = [
            idx
            for idx, edge in enumerate(order)
            if bool(edge["is_edge"]) and _row_overlap(int(inward["row"]), int(edge["row"]))
        ]
        latest_dep = max(deps) if deps else -1
        insert_at = random.randint(latest_dep + 1, len(order))
        order.insert(insert_at, inward)
    return order if ascending else order


def _assign_input_positions(
    width: int,
    entries: list[dict],
    *,
    ascending: bool,
) -> list[dict] | None:
    all_cols = list(range(0, width - SQUARE + 1))
    needed = len(entries)
    if len(all_cols) < needed:
        return None

    for _ in range(300):
        ordered = _sample_processing_order(entries, ascending=ascending)
        cols = sorted(random.sample(all_cols, needed))
        if not ascending:
            cols = cols[::-1]

        placed: list[dict] = []
        ok = True
        for idx, entry in enumerate(ordered):
            input_col = int(cols[idx])
            if input_col == int(entry["output_col"]):
                ok = False
                break
            placed.append({**entry, "input_col": input_col})
        if not ok:
            continue

        rects = [(int(item["row"]), int(item["input_col"])) for item in placed]
        if any(_rects_too_close(rects[i], rects[j]) for i in range(len(rects)) for j in range(i + 1, len(rects))):
            continue
        return placed
    return None


def _expected_side_columns(frame: int, width: int) -> tuple[int, int]:
    if frame == LEFT_FRAME:
        return 0, SQUARE
    return width - SQUARE, width - 2 * SQUARE


def _build_expected(height: int, width: int, metas: list[dict]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for meta in metas:
        patch = np.full((SQUARE, SQUARE), int(meta["frame"]), dtype=int)
        patch[1:-1, 1:-1] = np.array(meta["full_inner"], dtype=int)
        row = int(meta["row"])
        col = int(meta["output_col"])
        out[row : row + SQUARE, col : col + SQUARE] = patch
    return out


def _validate_case(inp: np.ndarray, expected: np.ndarray, metas: list[dict]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if inp.shape != expected.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not (3 <= len(metas) <= 5):
        return False

    frame_set = {int(meta["frame"]) for meta in metas}
    if frame_set != {LEFT_FRAME, RIGHT_FRAME}:
        return False

    input_rects = [(int(meta["row"]), int(meta["input_col"])) for meta in metas]
    for i in range(len(input_rects)):
        for j in range(i + 1, len(input_rects)):
            if _rects_too_close(input_rects[i], input_rects[j]):
                return False

    has_inward = False
    for meta in metas:
        frame = int(meta["frame"])
        patch = np.array(meta["patch"], dtype=int)
        full_inner = np.array(meta["full_inner"], dtype=int)
        occ_r, occ_c = map(int, meta["occ"])
        if int(np.count_nonzero(patch[1:-1, 1:-1] == OCCLUDER)) != 1:
            return False
        if frame == LEFT_FRAME:
            if not _is_vertical_symmetric(full_inner) or _is_horizontal_symmetric(full_inner):
                return False
            recovered = int(full_inner[occ_r, occ_c])
            wrong = int(full_inner[INNER - 1 - occ_r, occ_c])
            if recovered == wrong:
                return False
        else:
            if not _is_horizontal_symmetric(full_inner) or _is_vertical_symmetric(full_inner):
                return False
            recovered = int(full_inner[occ_r, occ_c])
            wrong = int(full_inner[occ_r, INNER - 1 - occ_c])
            if recovered == wrong:
                return False

        edge_col, inner_col = _expected_side_columns(frame, inp.shape[1])
        if int(meta["output_col"]) == inner_col:
            has_inward = True
        elif int(meta["output_col"]) != edge_col:
            return False

    if not has_inward:
        return False

    detected = _find_squares(inp)
    if len(detected) != len(metas):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 22)
    width = random.randint(18, 22)
    if width < 10:
        return None

    count_options = [((2, 1), 1), ((1, 2), 1), ((2, 2), 3), ((3, 2), 2), ((2, 3), 2)]
    weighted = [counts for counts, weight in count_options for _ in range(weight)]
    left_count, right_count = random.choice(weighted)

    left_rows = _sample_side_rows(height, left_count)
    right_rows = _sample_side_rows(height, right_count)
    if left_rows is None or right_rows is None:
        return None

    metas_by_side: list[dict] = []
    for frame, side_rows in ((LEFT_FRAME, left_rows), (RIGHT_FRAME, right_rows)):
        edge_rows, inward_rows = side_rows
        edge_col, inward_col = _expected_side_columns(frame, width)
        side_entries: list[dict] = []
        for row in edge_rows:
            patch, full_inner, occ = _make_patch(frame)
            side_entries.append(
                {
                    "frame": frame,
                    "row": int(row),
                    "output_col": int(edge_col),
                    "is_edge": True,
                    "patch": patch,
                    "full_inner": full_inner,
                    "occ": occ,
                }
            )
        for row in inward_rows:
            patch, full_inner, occ = _make_patch(frame)
            side_entries.append(
                {
                    "frame": frame,
                    "row": int(row),
                    "output_col": int(inward_col),
                    "is_edge": False,
                    "patch": patch,
                    "full_inner": full_inner,
                    "occ": occ,
                }
            )
        assigned = _assign_input_positions(width, side_entries, ascending=(frame == LEFT_FRAME))
        if assigned is None:
            return None
        metas_by_side.extend(assigned)

    input_rects = [(int(meta["row"]), int(meta["input_col"])) for meta in metas_by_side]
    for i in range(len(input_rects)):
        for j in range(i + 1, len(input_rects)):
            if _rects_too_close(input_rects[i], input_rects[j]):
                return None

    inp = np.full((height, width), BG, dtype=int)
    for meta in metas_by_side:
        row = int(meta["row"])
        col = int(meta["input_col"])
        inp[row : row + SQUARE, col : col + SQUARE] = np.array(meta["patch"], dtype=int)

    expected = _build_expected(height, width, metas_by_side)
    return inp if _validate_case(inp, expected, metas_by_side) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
