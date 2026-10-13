import itertools

import numpy as np


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(set(map(int, grid.ravel())))


def _repair_central_axis(
    crop: np.ndarray, pair: tuple[int, int], corruption: int, orientation: str
) -> np.ndarray | None:
    h, w = crop.shape
    out = crop.copy()

    if orientation == "vertical":
        for r in range(h):
            for c in range(w):
                cc = w - 1 - c
                a = int(crop[r, c])
                b = int(crop[r, cc])
                if c == cc:
                    if a == corruption:
                        return None
                    continue
                if a == corruption and b == corruption:
                    return None
                if a != corruption and b != corruption and a != b:
                    return None
                if a == corruption:
                    if b not in pair:
                        return None
                    out[r, c] = b
                elif b == corruption:
                    if a not in pair:
                        return None
                    out[r, cc] = a
    else:
        for r in range(h):
            rr = h - 1 - r
            for c in range(w):
                a = int(crop[r, c])
                b = int(crop[rr, c])
                if r == rr:
                    if a == corruption:
                        return None
                    continue
                if a == corruption and b == corruption:
                    return None
                if a != corruption and b != corruption and a != b:
                    return None
                if a == corruption:
                    if b not in pair:
                        return None
                    out[r, c] = b
                elif b == corruption:
                    if a not in pair:
                        return None
                    out[rr, c] = a

    if np.any(out == corruption):
        return None
    if set(map(int, out.ravel())) != set(pair):
        return None
    return out


def _find_candidate(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []

    for pair in itertools.combinations(_colors_present(g), 2):
        mask = np.isin(g, pair)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        if not rows.any() or not cols.any():
            continue
        if int(rows.sum()) == h and int(cols.sum()) == w:
            continue

        crop = g[np.ix_(rows, cols)]
        crop_colors = set(map(int, crop.ravel()))
        extras = sorted(crop_colors - set(pair))
        if len(extras) != 1:
            continue
        corruption = extras[0]
        counts = {color: int(np.sum(crop == color)) for color in crop_colors}
        if not (counts[corruption] < counts[pair[0]] and counts[corruption] < counts[pair[1]]):
            continue

        repaired = []
        for orientation in ("vertical", "horizontal"):
            fixed = _repair_central_axis(crop, pair, corruption, orientation)
            if fixed is not None:
                repaired.append((orientation, fixed))
        if len(repaired) != 1:
            continue

        candidates.append(
            {
                "pair": tuple(map(int, pair)),
                "corruption": int(corruption),
                "rows": rows,
                "cols": cols,
                "orientation": repaired[0][0],
                "output": repaired[0][1],
            }
        )

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the hidden two-color crop and repair its damaged cells by compact-frame symmetry.

    Input structure
    - The grid contains many colors, but exactly one hidden two-color image is spread across a sparse set of rows and columns.
    - A few cells of that image are overwritten by one conspicuous corruption color, while the remaining colors are only outer filler or separators.

    Rule
    - Find the unique two-color pair whose touched rows and touched columns induce a subgrid containing exactly those two colors plus one rarer corruption color.
    - Keep only those rows and columns, then use the compacted subgrid's mirror symmetry along its central vertical or horizontal axis to recover every corruption cell from its reflected partner.
    - Return the repaired compacted two-color subgrid.

    Key insight
    - The symmetry is not usable in the original coordinates because inactive separator rows and columns destroy equal mirror spacing; only after row/column compaction does the bilateral symmetry become exact.
    """
    return _find_candidate(np.asarray(grid, dtype=int))["output"].copy()


GEN_RETRIES = 400
SIZE_RANGE = range(5, 10)
INPUT_MAX_SIDE = 30


def _build_symmetric_output(
    height: int, width: int, orientation: str, color_a: int, color_b: int
) -> np.ndarray | None:
    if orientation == "vertical":
        source = np.full((height, (width + 1) // 2), color_a, dtype=int)
    else:
        source = np.full(((height + 1) // 2, width), color_a, dtype=int)

    for _ in range(np.random.randint(3, 7)):
        sh, sw = source.shape
        kind = int(np.random.randint(3))
        if kind == 0:
            rh = int(np.random.randint(1, min(4, sh) + 1))
            rw = int(np.random.randint(1, min(4, sw) + 1))
            r0 = int(np.random.randint(0, sh - rh + 1))
            c0 = int(np.random.randint(0, sw - rw + 1))
            source[r0 : r0 + rh, c0 : c0 + rw] = color_b
        elif kind == 1:
            r = int(np.random.randint(sh))
            c0 = int(np.random.randint(sw))
            c1 = int(np.random.randint(c0 + 1, sw + 1))
            source[r, c0:c1] = color_b
        else:
            c = int(np.random.randint(sw))
            r0 = int(np.random.randint(sh))
            r1 = int(np.random.randint(r0 + 1, sh + 1))
            source[r0:r1, c] = color_b

    out = np.full((height, width), color_a, dtype=int)
    if orientation == "vertical":
        left_w = (width + 1) // 2
        out[:, :left_w] = source
        if width % 2:
            out[:, left_w:] = source[:, : left_w - 1][:, ::-1]
        else:
            out[:, left_w:] = source[:, ::-1]
    else:
        top_h = (height + 1) // 2
        out[:top_h, :] = source
        if height % 2:
            out[top_h:, :] = source[: top_h - 1, :][::-1, :]
        else:
            out[top_h:, :] = source[::-1, :]

    count_b = int(np.sum(out == color_b))
    count_a = int(out.size - count_b)
    if min(count_a, count_b) < 6:
        return None
    ratio = count_b / float(out.size)
    if not (0.18 <= ratio <= 0.62):
        return None

    other = "horizontal" if orientation == "vertical" else "vertical"
    if _repair_central_axis(out, (color_a, color_b), -1, other) is not None:
        return None
    return out


def _axis_orbits(shape: tuple[int, int], orientation: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    seen = set()
    orbits = []
    for r in range(h):
        for c in range(w):
            if orientation == "vertical":
                mirror = (r, w - 1 - c)
            else:
                mirror = (h - 1 - r, c)
            if mirror == (r, c):
                continue
            key = tuple(sorted(((r, c), mirror)))
            if key in seen:
                continue
            seen.add(key)
            orbits.append((key[0], key[1]))
    return orbits


def _pick_corruptions(
    full: np.ndarray, orientation: str, color_a: int, color_b: int, corruption: int
) -> np.ndarray | None:
    out = full.copy()
    orbits = _axis_orbits(full.shape, orientation)
    if not orbits:
        return None

    np.random.shuffle(orbits)
    target_count = int(np.random.randint(1, min(5, len(orbits) + 1)))
    chosen = []
    used_colors = set()

    for first, second in orbits:
        if len(chosen) >= target_count:
            break
        options = [first, second]
        np.random.shuffle(options)
        pick = options[0]
        if len(chosen) < 2 and len(used_colors) < 2:
            options.sort(key=lambda rc: int(full[rc]) in used_colors)
            pick = options[0]
        chosen.append(pick)
        used_colors.add(int(full[pick]))

    for r, c in chosen:
        out[r, c] = corruption

    pair_mask = np.isin(out, (color_a, color_b))
    if not np.all(np.any(pair_mask, axis=1)):
        return None
    if not np.all(np.any(pair_mask, axis=0)):
        return None
    if np.sum(out == corruption) >= min(np.sum(out == color_a), np.sum(out == color_b)):
        return None
    return out


def _sample_positions(count: int, require_non_palindrome: bool) -> tuple[int, list[int]] | None:
    for _ in range(120):
        gaps = np.random.randint(1, 4, size=max(0, count - 1))
        pad0 = int(np.random.randint(1, 5))
        pad1 = int(np.random.randint(1, 5))
        total = pad0 + count + int(gaps.sum()) + pad1
        if total > INPUT_MAX_SIDE:
            continue
        signature = (pad0, *map(int, gaps), pad1)
        if require_non_palindrome and signature == signature[::-1]:
            continue
        positions = []
        cur = pad0
        for i in range(count):
            positions.append(int(cur))
            cur += 1
            if i < count - 1:
                cur += int(gaps[i])
        return total, positions
    return None

def _raw_frame_has_direct_symmetry(
    grid: np.ndarray, rows: list[int], cols: list[int], pair: tuple[int, int], corruption: int
) -> bool:
    frame = grid[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    for orientation in ("vertical", "horizontal"):
        if _repair_central_axis(frame, pair, corruption, orientation) is not None:
            return True
    return False


def _attempt_generate() -> np.ndarray | None:
    colors = list(np.random.permutation(np.arange(1, 10))[:5])
    color_a, color_b, corruption, filler_a, filler_b = map(int, colors)
    orientation = "vertical" if np.random.randint(2) == 0 else "horizontal"

    for _ in range(120):
        height = int(np.random.choice(list(SIZE_RANGE)))
        width = int(np.random.choice(list(SIZE_RANGE)))
        full = _build_symmetric_output(height, width, orientation, color_a, color_b)
        if full is None:
            continue

        damaged = _pick_corruptions(full, orientation, color_a, color_b, corruption)
        if damaged is None:
            continue

        row_spec = _sample_positions(height, orientation == "horizontal")
        col_spec = _sample_positions(width, orientation == "vertical")
        if row_spec is None or col_spec is None:
            continue
        grid_h, active_rows = row_spec
        grid_w, active_cols = col_spec
        if grid_h > INPUT_MAX_SIDE or grid_w > INPUT_MAX_SIDE:
            continue

        grid = np.full((grid_h, grid_w), filler_a, dtype=int)
        active_row_mask = np.zeros(grid_h, dtype=bool)
        active_col_mask = np.zeros(grid_w, dtype=bool)
        active_row_mask[active_rows] = True
        active_col_mask[active_cols] = True
        for r in range(grid_h):
            for c in range(grid_w):
                if not active_row_mask[r] or not active_col_mask[c]:
                    if (not active_row_mask[r] and not active_col_mask[c]) or (r + c) % 2:
                        grid[r, c] = filler_b
        for rr, r in enumerate(active_rows):
            for cc, c in enumerate(active_cols):
                grid[r, c] = int(damaged[rr, cc])

        try:
            candidate = _find_candidate(grid)
        except AssertionError:
            continue
        if candidate["pair"] != tuple(sorted((color_a, color_b))):
            continue
        if candidate["corruption"] != corruption:
            continue
        if not np.array_equal(candidate["output"], full):
            continue
        if _raw_frame_has_direct_symmetry(grid, active_rows, active_cols, candidate["pair"], corruption):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compressed-symmetry puzzle")
