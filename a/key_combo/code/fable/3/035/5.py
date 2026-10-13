from __future__ import annotations

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
MAX_REPAIR_GAP = 3


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(
    grid: np.ndarray,
    row_start: int,
    row_stop: int,
    *,
    connectivity: int,
) -> list[dict]:
    height, width = map(int, grid.shape)
    visited = np.zeros_like(grid, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    components: list[dict] = []
    for r0 in range(int(row_start), int(row_stop)):
        for c0 in range(width):
            color = int(grid[r0, c0])
            if color == BLACK or visited[r0, c0]:
                continue
            stack = [(r0, c0)]
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in steps:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if not (row_start <= rr < row_stop):
                        continue
                    if visited[rr, cc] or int(grid[rr, cc]) != color:
                        continue
                    visited[rr, cc] = True
                    stack.append((rr, cc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return components


def _bool_components(mask: np.ndarray, *, connectivity: int) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    visited = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    components: list[list[tuple[int, int]]] = []
    for r0 in range(height):
        for c0 in range(width):
            if visited[r0, c0] or not bool(mask[r0, c0]):
                continue
            visited[r0, c0] = True
            stack = [(r0, c0)]
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in steps:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if visited[rr, cc] or not bool(mask[rr, cc]):
                        continue
                    visited[rr, cc] = True
                    stack.append((rr, cc))
            components.append(coords)
    return components


def _orth_count(mask: np.ndarray, r: int, c: int) -> int:
    height, width = map(int, mask.shape)
    count = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]):
            count += 1
    return count


def _has_diagonal(mask: np.ndarray, r: int, c: int) -> bool:
    height, width = map(int, mask.shape)
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]):
            return True
    return False


def _split_bands(grid: np.ndarray) -> int:
    occupied_rows = np.where(np.any(grid != BLACK, axis=1))[0]
    assert occupied_rows.size >= 2, "expected both floating objects and a structure layer"
    gaps = np.diff(occupied_rows)
    split_idx = int(np.argmax(gaps))
    assert int(gaps[split_idx]) >= 2, "expected a clear empty band between objects and structure"
    return int(occupied_rows[split_idx])


def _repair_structure(grid: np.ndarray, structure_row_start: int) -> np.ndarray:
    repaired = grid.copy()
    repaired[:structure_row_start, :] = BLACK
    structure = repaired.copy()

    for color in sorted(set(map(int, np.unique(structure))) - {BLACK}):
        mask = structure == color
        noise = np.zeros_like(mask, dtype=bool)
        for r in range(structure_row_start, structure.shape[0]):
            for c in range(structure.shape[1]):
                if not bool(mask[r, c]):
                    continue
                if _orth_count(mask, r, c) == 0 and _has_diagonal(mask, r, c):
                    noise[r, c] = True

        repaired[noise] = BLACK
        color_mask = repaired == color

        max_gap = MAX_REPAIR_GAP
        for component in _bool_components(noise, connectivity=8):
            r0, c0, r1, c1 = _bbox(component)
            max_gap = max(max_gap, r1 - r0 + 1, c1 - c0 + 1)

        for r in range(structure_row_start, repaired.shape[0]):
            cols = np.where(color_mask[r])[0]
            if cols.size < 2:
                continue
            segments: list[tuple[int, int]] = []
            start = int(cols[0])
            prev = int(cols[0])
            for c in cols[1:]:
                c = int(c)
                if c == prev + 1:
                    prev = c
                else:
                    segments.append((start, prev))
                    start = prev = c
            segments.append((start, prev))

            for (_a0, a1), (b0, _b1) in zip(segments, segments[1:]):
                gap0 = a1 + 1
                gap1 = b0 - 1
                gap_len = gap1 - gap0 + 1
                if gap_len <= 0 or gap_len > max_gap:
                    continue
                if not np.all(repaired[r, gap0 : gap1 + 1] == BLACK):
                    continue
                rr0 = max(structure_row_start, r - 2)
                rr1 = min(repaired.shape[0] - 1, r + 2)
                cc0 = max(0, gap0 - 1)
                cc1 = min(repaired.shape[1] - 1, gap1 + 1)
                if bool(noise[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                    repaired[r, gap0 : gap1 + 1] = color
                    color_mask[r, gap0 : gap1 + 1] = True

    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise, repair the broken horizontal structure, and drop colored markers onto it.

    Input structure:
    - Black is the background.
    - Several monochrome floating objects sit in the upper band of the grid.
    - A lower structure band contains horizontal line segments, plus same-colored diagonal-only noise chains.

    Rule:
    - Split the scene into the upper object band and the lower structure band using the empty gap between them.
    - Inside the structure band, delete every pixel whose same-colored neighbors are diagonal-only, then fill short horizontal gaps bracketed by same-colored line pieces when the missing run is explained by nearby deleted noise.
    - For each upper object, take the center column of its bounding box and place one marker in the cell directly above the first repaired structure pixel below it; the marker uses that supporting line's color. If no repaired structure exists in that column, add nothing.

    Key insight:
    - The landing surface is the repaired structure, not the raw input: diagonal hints are non-solid noise, and some valid landing cells exist only after the hidden horizontal line has been reconstructed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert 1 <= height <= 30 and 1 <= width <= 30, "grid must fit ARC size limits"

    split_row = _split_bands(g)
    structure_row_start = split_row + 1
    repaired = _repair_structure(g, structure_row_start)

    objects = _components_by_color(g, 0, structure_row_start, connectivity=4)
    assert objects, "expected floating objects above the structure"

    out = g.copy()
    out[structure_row_start:, :] = repaired[structure_row_start:, :]

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert (c1 - c0 + 1) % 2 == 1, "expected floating objects with a unique center column"
        center_c = (c0 + c1) // 2
        support_row = None
        for r in range(r1 + 1, height):
            if int(repaired[r, center_c]) != BLACK:
                support_row = r
                break
        if support_row is None:
            continue
        marker_row = support_row - 1
        assert marker_row > r1, "marker must land below its source object"
        assert int(repaired[marker_row, center_c]) == BLACK, "marker cell must be empty above the repaired structure"
        out[marker_row, center_c] = int(repaired[support_row, center_c])

    return out


def _pick_spaced_values(
    rng: np.random.Generator,
    values: list[int],
    count: int,
    min_gap: int,
) -> list[int]:
    order = values[:]
    rng.shuffle(order)
    picked: list[int] = []
    for value in order:
        if all(abs(value - prev) >= min_gap for prev in picked):
            picked.append(int(value))
            if len(picked) == count:
                return sorted(picked)
    return []


def _choose_line_rows(
    rng: np.random.Generator,
    height: int,
    upper_bottom: int,
) -> list[int] | None:
    count = int(rng.integers(2, 4))
    start = upper_bottom + int(rng.integers(7, 9))
    rows = [start]
    for _ in range(count - 1):
        rows.append(rows[-1] + int(rng.integers(3, 5)))
    if rows[-1] >= height - 2:
        return None
    return rows


def _build_clean_structure(
    rng: np.random.Generator,
    height: int,
    width: int,
    line_rows: list[int],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((height, width), dtype=int)
    colors = list(PALETTE)
    rng.shuffle(colors)
    if len(line_rows) > len(colors):
        return None

    segments: list[dict] = []
    for row, color in zip(line_rows, colors):
        max_len = min(13, width - 4)
        min_len = 7
        if max_len < min_len:
            return None
        length = int(rng.integers(min_len, max_len + 1))
        c0 = int(rng.integers(2, width - length - 1))
        c1 = c0 + length - 1
        grid[row, c0 : c1 + 1] = int(color)
        segments.append({"row": int(row), "c0": c0, "c1": c1, "color": int(color)})
    return grid, segments


def _gap_chain_candidates(
    height: int,
    width: int,
    row: int,
    gap0: int,
    gap1: int,
    chain_len: int,
) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    for vdir in (-1, 1):
        left_chain = [(row + vdir * step, gap0 + step - 1) for step in range(1, chain_len + 1)]
        right_chain = [(row + vdir * step, gap1 - step + 1) for step in range(1, chain_len + 1)]
        for chain in (left_chain, right_chain):
            if all(0 <= r < height and 0 <= c < width for r, c in chain):
                candidates.append(chain)
    return candidates


def _validate_noise_chain(
    inp: np.ndarray,
    chain: list[tuple[int, int]],
    color: int,
) -> bool:
    chain_set = set(chain)
    height, width = map(int, inp.shape)
    for r, c in chain:
        if int(inp[r, c]) != BLACK:
            return False
        has_diag = False
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and (
                (rr, cc) in chain_set or int(inp[rr, cc]) == color
            ):
                has_diag = True
                break
        if not has_diag:
            return False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and (
                (rr, cc) in chain_set or int(inp[rr, cc]) == color
            ):
                return False
    return True


def _corrupt_structure(
    rng: np.random.Generator,
    clean_structure: np.ndarray,
    segments: list[dict],
) -> tuple[np.ndarray, list[dict]] | None:
    inp = clean_structure.copy()
    top_row = min(seg["row"] for seg in segments)
    top_segments = [seg for seg in segments if seg["row"] == top_row]
    assert len(top_segments) == 1
    top_segment = top_segments[0]

    breakable = [seg for seg in segments if seg["c1"] - seg["c0"] + 1 >= 7]
    if not breakable:
        return None

    planned = [top_segment]
    extra = [seg for seg in breakable if seg is not top_segment]
    rng.shuffle(extra)
    planned.extend(extra[: int(rng.integers(0, min(2, len(extra)) + 1))])

    breaks: list[dict] = []
    for seg in planned:
        seg_len = seg["c1"] - seg["c0"] + 1
        gap_len = int(rng.integers(1, min(3, seg_len - 4) + 1))
        gap0 = int(rng.integers(seg["c0"] + 2, seg["c1"] - gap_len))
        gap1 = gap0 + gap_len - 1
        inp[seg["row"], gap0 : gap1 + 1] = BLACK

        chain_len = 2
        chains = _gap_chain_candidates(inp.shape[0], inp.shape[1], seg["row"], gap0, gap1, chain_len)
        rng.shuffle(chains)
        chosen = None
        for chain in chains:
            if _validate_noise_chain(inp, chain, int(seg["color"])):
                chosen = chain
                break
        if chosen is None:
            return None
        for r, c in chosen:
            inp[r, c] = int(seg["color"])
        breaks.append(
            {
                "row": int(seg["row"]),
                "gap0": gap0,
                "gap1": gap1,
                "color": int(seg["color"]),
            }
        )
    return inp, breaks


def _top_support_rows(structure: np.ndarray) -> dict[int, int]:
    support: dict[int, int] = {}
    for c in range(structure.shape[1]):
        rows = np.where(structure[:, c] != BLACK)[0]
        if rows.size:
            support[int(c)] = int(rows[0])
    return support


def _choose_object_columns(
    rng: np.random.Generator,
    width: int,
    clean_structure: np.ndarray,
    breaks: list[dict],
) -> list[int] | None:
    top_support = _top_support_rows(clean_structure)
    top_row = min(top_support.values())
    interior = [c for c in range(1, width - 1)]
    gap_cols = sorted(
        {
            c
            for info in breaks
            if info["row"] == top_row
            for c in range(info["gap0"], info["gap1"] + 1)
        }
    )
    if not gap_cols:
        return None
    lower_only = [c for c, r in top_support.items() if r > top_row and 1 <= c <= width - 2]
    solid_support = [c for c, r in top_support.items() if r == top_row and c not in gap_cols and 1 <= c <= width - 2]
    no_support = [c for c in interior if c not in top_support]

    columns: list[int] = []

    def pick_from(candidates: list[int]) -> int | None:
        options = [c for c in candidates if all(abs(c - prev) >= 4 for prev in columns)]
        if not options:
            return None
        return int(options[int(rng.integers(len(options)))])

    gap_col = pick_from(gap_cols)
    if gap_col is None:
        return None
    columns.append(gap_col)

    second_pool = lower_only if lower_only and bool(rng.random() < 0.7) else solid_support
    second_col = pick_from(second_pool)
    if second_col is None and solid_support:
        second_col = pick_from(solid_support)
    if second_col is not None:
        columns.append(second_col)

    if no_support and bool(rng.random() < 0.8):
        no_support_col = pick_from(no_support)
        if no_support_col is not None:
            columns.append(no_support_col)

    mixed_pool = sorted(set(lower_only + solid_support + no_support) - set(columns))
    if mixed_pool and bool(rng.random() < 0.4):
        extra_col = pick_from(mixed_pool)
        if extra_col is not None:
            columns.append(extra_col)

    if len(columns) < 2:
        fallback = pick_from(sorted(set(solid_support + lower_only) - set(columns)))
        if fallback is not None:
            columns.append(fallback)
    if len(columns) < 2:
        return None
    return columns


def _place_objects(
    rng: np.random.Generator,
    grid: np.ndarray,
    columns: list[int],
    upper_bottom: int,
    forbidden_colors: set[int],
) -> list[dict] | None:
    available = [color for color in PALETTE if color not in forbidden_colors]
    if len(available) < len(columns):
        available = list(PALETTE)
    rng.shuffle(available)
    objects: list[dict] = []
    used_boxes: list[tuple[int, int, int, int]] = []

    for idx, center_c in enumerate(columns):
        color = int(available[idx])
        height = int(rng.integers(2, 4))
        width = 3
        c0 = center_c - 1
        c1 = center_c + 1
        if c0 < 0 or c1 >= grid.shape[1]:
            return None
        placed = False
        row_choices = list(range(1, upper_bottom - height + 2))
        rng.shuffle(row_choices)
        for r0 in row_choices:
            r1 = r0 + height - 1
            box = (r0, c0, r1, c1)
            if any(not (box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1]) for other in used_boxes):
                continue
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            objects.append({"bbox": box, "center_c": int(center_c)})
            used_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return objects


def _expected_output(
    objects_only_grid: np.ndarray,
    clean_structure: np.ndarray,
    objects: list[dict],
) -> np.ndarray:
    out = objects_only_grid.copy()
    out[clean_structure != BLACK] = clean_structure[clean_structure != BLACK]
    support = _top_support_rows(clean_structure)
    for obj in objects:
        r0, _c0, r1, _c1 = obj["bbox"]
        center_c = int(obj["center_c"])
        support_row = support.get(center_c)
        if support_row is None:
            continue
        marker_row = support_row - 1
        assert marker_row > r1
        out[marker_row, center_c] = int(clean_structure[support_row, center_c])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(20, 29))
    width = int(rng.integers(18, 29))
    upper_bottom = int(rng.integers(3, 7))

    line_rows = _choose_line_rows(rng, height, upper_bottom)
    if line_rows is None:
        return None

    built = _build_clean_structure(rng, height, width, line_rows)
    if built is None:
        return None
    clean_structure, segments = built

    corrupted = _corrupt_structure(rng, clean_structure, segments)
    if corrupted is None:
        return None
    structure_input, breaks = corrupted

    object_columns = _choose_object_columns(rng, width, clean_structure, breaks)
    if object_columns is None:
        return None

    inp = structure_input.copy()
    objects = _place_objects(
        rng,
        inp,
        object_columns,
        upper_bottom,
        {int(seg["color"]) for seg in segments},
    )
    if objects is None:
        return None

    objects_only = inp.copy()
    objects_only[structure_input != BLACK] = BLACK
    expected = _expected_output(objects_only, clean_structure, objects)

    occupied_rows = np.where(np.any(inp != BLACK, axis=1))[0]
    if occupied_rows.size < 2:
        return None
    if int(np.max(np.diff(occupied_rows))) < 4:
        return None
    try:
        split = _split_bands(inp)
    except AssertionError:
        return None
    if split >= min(line_rows):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 035/5")
