import numpy as np


BACKGROUND = 0
SEPARATOR = 5
OUTLINE_COLORS = (1, 2, 3, 4, 6)
REFERENCE_COLORS = (7, 8, 9)
GEN_RETRIES = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEPARATOR)]
    assert len(rows) == 1, "expected exactly one full grey separator row"
    return int(rows[0])


def _find_upper_outlines(grid: np.ndarray, separator_row: int) -> list[dict]:
    upper = np.asarray(grid[:separator_row], dtype=int)
    H, W = upper.shape
    visited = np.zeros((H, W), dtype=bool)
    outlines: list[dict] = []

    for r in range(H):
        for c in range(W):
            color = int(upper[r, c])
            if color in (BACKGROUND, SEPARATOR) or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if visited[nr, nc] or int(upper[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            rows = sorted({int(rr) for rr in arr[:, 0]})
            row_pairs: dict[int, tuple[int, int]] = {}
            for rr in rows:
                cols = sorted(int(x) for x in arr[arr[:, 0] == rr, 1])
                assert len(cols) == 2 and cols[0] < cols[1], "each outline row must have exactly two boundary pixels"
                c0, c1 = cols
                span = upper[rr, c0 : c1 + 1]
                assert np.all((span == BACKGROUND) | (span == color)), "outline row span must contain only background and its own color"
                row_pairs[rr] = (c0, c1)

            outlines.append({"color": color, "coords": arr, "rows": rows, "row_pairs": row_pairs})

    assert outlines, "expected at least one outline above the separator"
    return outlines


def _find_reference_pixels(grid: np.ndarray, separator_row: int) -> dict[int, tuple[int, int]]:
    refs = {}
    cols = set()
    for r in range(separator_row + 1, grid.shape[0]):
        non_bg = np.flatnonzero((grid[r] != BACKGROUND) & (grid[r] != SEPARATOR))
        if non_bg.size == 0:
            continue
        assert non_bg.size == 1, "each reference row must contain exactly one reference pixel"
        c = int(non_bg[0])
        cols.add(c)
        refs[r] = (c, int(grid[r, c]))

    assert len(cols) <= 1, "reference pixels must lie on a single column"
    return refs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each upper outline with the mirrored reference color when a lower reference pixel aligns across the separator, otherwise erase it.

    Input structure:
    - The grid has black background and one full grey horizontal separator row.
    - Above the separator are several monochromatic 8-connected outlines; each occupied row of an outline contains exactly two boundary pixels.
    - Below the separator are isolated reference pixels placed on a single column, with at most one reference row mirroring any outline.

    Rule:
    - Find the grey separator row S.
    - For each upper outline, inspect every occupied row r and look for a lower reference pixel on row 2S-r.
    - If exactly one aligned reference exists for that outline, recolor every horizontal span between its left and right boundary pixels with the reference color.
    - If no aligned reference exists for that outline, erase the outline to black; leave the separator, reference pixels, and all other background cells unchanged.

    Key insight:
    - The outline geometry determines where filling happens, but the fill color is injected from the opposite side by row-wise mirror alignment across the separator.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_row = _find_separator_row(g)
    outlines = _find_upper_outlines(g, separator_row)
    refs = _find_reference_pixels(g, separator_row)

    out = g.copy()
    for outline in outlines:
        aligned = {
            refs[mirror_row][1]
            for r in outline["rows"]
            if (mirror_row := 2 * separator_row - int(r)) in refs
        }
        assert len(aligned) <= 1, "each outline must align with at most one reference color"

        if aligned:
            fill_color = int(next(iter(aligned)))
            for r in outline["rows"]:
                c0, c1 = outline["row_pairs"][int(r)]
                out[int(r), c0 : c1 + 1] = fill_color
        else:
            for r, c in outline["coords"]:
                out[int(r), int(c)] = BACKGROUND

    return out


def _sample_profiles(rng: np.random.Generator, heights: list[int]) -> tuple[list[list[tuple[int, int]]], list[int]] | None:
    profiles: list[list[tuple[int, int]]] = []
    widths: list[int] = []
    for height in heights:
        rel = [(0, 1)]
        for _ in range(height - 1):
            left, right = rel[-1]
            width = right - left + 1
            options = [(left, right)]
            if width < 4:
                options.append((left - 1, right))
                options.append((left, right + 1))
                options.append((left - 1, right + 1))
            elif width == 4:
                options.append((left - 1, right))
                options.append((left, right + 1))
            rel.append(options[int(rng.integers(len(options)))])

        if height > 2 and len(set(rel)) == 1:
            return None

        if bool(rng.integers(2)):
            rel = rel[::-1]
        min_left = min(left for left, _ in rel)
        normalized = [(left - min_left, right - min_left) for left, right in rel]
        width = max(right for _, right in normalized) + 1
        if width > 4:
            return None
        profiles.append(normalized)
        widths.append(width)
    return profiles, widths


def _distribute_slack(rng: np.random.Generator, slots: int, total: int) -> list[int]:
    values = [0] * slots
    for _ in range(total):
        values[int(rng.integers(slots))] += 1
    return values


def _build_outline_meta(
    row_starts: list[int],
    heights: list[int],
    col_starts: list[int],
    profiles: list[list[tuple[int, int]]],
    colors: list[int],
) -> list[dict]:
    outlines = []
    for r0, height, c0, profile, color in zip(row_starts, heights, col_starts, profiles, colors, strict=True):
        row_pairs = {}
        coords = []
        rows = []
        for dr in range(height):
            rr = int(r0 + dr)
            left, right = profile[dr]
            c_left = int(c0 + left)
            c_right = int(c0 + right)
            row_pairs[rr] = (c_left, c_right)
            rows.append(rr)
            coords.append((rr, c_left))
            coords.append((rr, c_right))
        outlines.append(
            {
                "color": int(color),
                "rows": rows,
                "row_pairs": row_pairs,
                "coords": coords,
            }
        )
    return outlines


def _render_grid(H: int, W: int, separator_row: int, outlines: list[dict], ref_col: int, refs: dict[int, int]) -> np.ndarray:
    grid = np.full((H, W), BACKGROUND, dtype=int)
    grid[separator_row, :] = SEPARATOR
    for outline in outlines:
        for r, c in outline["coords"]:
            grid[r, c] = outline["color"]
    for r, color in refs.items():
        grid[int(r), int(ref_col)] = int(color)
    return grid


def _validate_instance(
    grid: np.ndarray,
    separator_row: int,
    outlines: list[dict],
    ref_col: int,
    ref_rows: dict[int, int],
    aligned_map: dict[int, int],
) -> bool:
    out = main(grid)

    if not np.all(out[separator_row] == SEPARATOR):
        return False
    for r, color in ref_rows.items():
        if int(out[int(r), int(ref_col)]) != int(color):
            return False
    lower_mask = np.zeros_like(grid, dtype=bool)
    for r in ref_rows:
        lower_mask[int(r), int(ref_col)] = True
    untouched_lower = (np.arange(grid.shape[0])[:, None] > separator_row) & (~lower_mask)
    if not np.array_equal(out[untouched_lower], grid[untouched_lower]):
        return False

    saw_filled = False
    saw_erased = False
    for idx, outline in enumerate(outlines):
        fill_color = aligned_map.get(idx)
        for r in outline["rows"]:
            c0, c1 = outline["row_pairs"][r]
            span = out[r, c0 : c1 + 1]
            if fill_color is None:
                if not np.all(span == BACKGROUND):
                    return False
                saw_erased = True
            else:
                if not np.all(span == fill_color):
                    return False
                saw_filled = True

    if not (saw_filled and saw_erased):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_RETRIES):
        count = int(rng.integers(3, 6))

        heights = [2] * count
        extra_height = int(rng.integers(0, min(5, 13 - 2 * count) + 1))
        for _ in range(extra_height):
            choices = [i for i, h in enumerate(heights) if h < 4]
            if not choices:
                break
            heights[int(rng.choice(choices))] += 1

        min_upper = sum(heights)
        upper_height = int(rng.integers(max(8, min_upper), min(13, min_upper + 3) + 1))
        vertical_slack = upper_height - min_upper
        gap_slots = _distribute_slack(rng, count + 1, vertical_slack)
        row_starts = []
        cursor = gap_slots[0]
        for idx, height in enumerate(heights):
            row_starts.append(int(cursor))
            cursor += height
            if idx < count - 1:
                cursor += gap_slots[idx + 1]
        separator_row = upper_height

        H_min = 2 * separator_row + 1
        H_max = min(30, H_min + 6)
        if H_min > H_max:
            continue
        H = int(rng.integers(H_min, H_max + 1))

        sampled = _sample_profiles(rng, heights)
        if sampled is None:
            continue
        profiles, widths = sampled

        min_width = sum(widths) + (count - 1) + 2
        if min_width > 30:
            continue
        W = int(rng.integers(max(18, min_width), min(30, min_width + 8) + 1))
        horizontal_slack = W - min_width
        h_slots = _distribute_slack(rng, count + 1, horizontal_slack)
        col_starts = []
        cursor = 1 + h_slots[0]
        for idx, width in enumerate(widths):
            col_starts.append(int(cursor))
            cursor += width
            if idx < count - 1:
                cursor += 1 + h_slots[idx + 1]

        colors = list(map(int, rng.choice(np.array(OUTLINE_COLORS), size=count, replace=False)))
        outlines = _build_outline_meta(row_starts, heights, col_starts, profiles, colors)

        aligned_count = int(rng.integers(1, min(3, count - 1) + 1))
        aligned_indices = list(map(int, rng.choice(np.arange(count), size=aligned_count, replace=False)))
        ref_colors = list(map(int, rng.choice(np.array(REFERENCE_COLORS), size=aligned_count, replace=False)))
        ref_col = int(rng.integers(1, W - 1))

        ref_rows: dict[int, int] = {}
        aligned_map: dict[int, int] = {}
        ok = True
        for idx, ref_color in zip(aligned_indices, ref_colors, strict=True):
            chosen_row = int(rng.choice(outlines[idx]["rows"]))
            ref_row = 2 * separator_row - chosen_row
            if ref_row <= separator_row or ref_row >= H or ref_row in ref_rows:
                ok = False
                break
            ref_rows[ref_row] = int(ref_color)
            aligned_map[idx] = int(ref_color)
        if not ok:
            continue

        grid = _render_grid(H, W, separator_row, outlines, ref_col, ref_rows)

        if not _validate_instance(grid, separator_row, outlines, ref_col, ref_rows, aligned_map):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 2")
