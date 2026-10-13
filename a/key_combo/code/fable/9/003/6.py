from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5

PATTERN_COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _mode_color(grid: np.ndarray, exclude: set[int] | None = None) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = sorted(
        (
            (int(count), int(value))
            for value, count in zip(values, counts, strict=False)
            if exclude is None or int(value) not in exclude
        ),
        reverse=True,
    )
    assert order
    return order[0][1]


def _is_asymmetric(pattern: np.ndarray) -> bool:
    return not any(
        np.array_equal(pattern, candidate)
        for candidate in (
            np.fliplr(pattern),
            np.flipud(pattern),
            np.rot90(pattern, 2),
        )
    )


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) == GREY and bool(np.all(grid[r] == GREY))
    ]
    assert len(rows) == 1, "expected one grey separator row"
    return int(rows[0])


def _extract_pattern_and_canvas(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep = _find_separator_row(grid)
    assert 0 < sep < grid.shape[0] - 1
    top = np.asarray(grid[:sep], dtype=int)
    canvas = np.asarray(grid[sep + 1 :], dtype=int)
    bg = _mode_color(np.concatenate([top.ravel(), canvas.ravel()]), exclude={GREY})
    coords = np.argwhere(top != bg)
    assert coords.size, "legend must contain one full stamp"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    pattern = top[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert pattern.shape[0] == pattern.shape[1] and pattern.shape[0] % 2 == 1
    assert np.all(pattern != bg), "solver assumes a dense stamp pattern"
    return pattern, canvas


def _enumerate_candidates(pattern: np.ndarray, canvas: np.ndarray) -> list[dict]:
    h, w = canvas.shape
    k = pattern.shape[0]
    rad = k // 2
    candidates: list[dict] = []
    for cr in range(-rad, h + rad):
        for cc in range(-rad, w + rad):
            rows: list[int] = []
            cols: list[int] = []
            colors: list[int] = []
            for pr in range(k):
                rr = cr + pr - rad
                if rr < 0 or rr >= h:
                    continue
                for pc in range(k):
                    cc2 = cc + pc - rad
                    if 0 <= cc2 < w:
                        rows.append(int(rr))
                        cols.append(int(cc2))
                        colors.append(int(pattern[pr, pc]))
            if not rows:
                continue
            candidates.append(
                {
                    "center": (int(cr), int(cc)),
                    "rows": np.array(rows, dtype=np.int16),
                    "cols": np.array(cols, dtype=np.int16),
                    "colors": np.array(colors, dtype=np.int8),
                }
            )
    return candidates


def _solve_centers(pattern: np.ndarray, canvas: np.ndarray) -> list[tuple[int, int]]:
    canvas = np.asarray(canvas, dtype=int)
    h, w = canvas.shape
    bg = _mode_color(canvas)
    candidates = _enumerate_candidates(pattern, canvas)
    cover_map: list[list[list[int]]] = [[[] for _ in range(w)] for _ in range(h)]
    for idx, cand in enumerate(candidates):
        for rr, cc, color in zip(cand["rows"], cand["cols"], cand["colors"], strict=False):
            if int(canvas[int(rr), int(cc)]) == int(color):
                cover_map[int(rr)][int(cc)].append(idx)

    memo: dict[bytes, list[int] | None] = {}
    fg_mask = canvas != bg

    def compatible(idx: int, known_mask: np.ndarray) -> bool:
        cand = candidates[idx]
        rr = cand["rows"]
        cc = cand["cols"]
        active = known_mask[rr, cc]
        if not bool(active.any()):
            return False
        return not bool((canvas[rr[active], cc[active]] != cand["colors"][active]).any())

    def dfs(known_mask: np.ndarray) -> list[int] | None:
        key = known_mask.tobytes()
        if key in memo:
            return memo[key]

        remaining = known_mask & fg_mask
        if not bool(remaining.any()):
            memo[key] = []
            return []

        best_options: list[int] | None = None
        best_cell: tuple[int, int] | None = None
        for r_raw, c_raw in np.argwhere(remaining):
            r = int(r_raw)
            c = int(c_raw)
            options = [idx for idx in cover_map[r][c] if compatible(idx, known_mask)]
            if not options:
                memo[key] = None
                return None
            if best_options is None or len(options) < len(best_options):
                best_options = options
                best_cell = (r, c)
                if len(options) == 1:
                    break

        assert best_options is not None and best_cell is not None
        best_options.sort(
            key=lambda idx: (
                -int((known_mask[candidates[idx]["rows"], candidates[idx]["cols"]]).sum()),
                abs(candidates[idx]["center"][0] - best_cell[0])
                + abs(candidates[idx]["center"][1] - best_cell[1]),
                candidates[idx]["center"],
            )
        )

        for idx in best_options:
            cand = candidates[idx]
            next_mask = known_mask.copy()
            next_mask[cand["rows"], cand["cols"]] = False
            suffix = dfs(next_mask)
            if suffix is not None:
                result = [idx, *suffix]
                memo[key] = result
                return result

        memo[key] = None
        return None

    start_mask = np.ones(canvas.shape, dtype=bool)
    solution = dfs(start_mask)
    assert solution is not None, "failed to explain all visible stamp residues"
    return [candidates[idx]["center"] for idx in solution]


def _anchors_to_output(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BLACK, dtype=int)
    for r, c in centers:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover every hidden stamp placement from layered residues and mark only the anchors.

    Input structure:
    - The top band shows one complete odd-sized, multi-color, asymmetric stamp pattern.
    - A full-width grey separator row sits directly below the legend.
    - Under the separator is a canvas where the same dense stamp was applied several times;
      later stamps cover earlier ones opaquely, and clipped edge stamps are allowed.

    Rule:
    - Infer every stamp center whose visible residue is consistent with the legend pattern at
      the corresponding offset, even if parts of that stamp were covered later or clipped by
      the canvas boundary.
    - Output only the canvas area, not the legend or separator.
    - Start from a black canvas of the same size and place one red pixel at every recovered
      stamp center that lies inside the canvas.

    Key insight:
    - The visible fragments are explained by repeatedly identifying which full clipped stamp
      could be the current top layer; removing that layer turns its whole footprint into
      unknown space and exposes the need for lower stamps underneath.
    """
    g = np.asarray(grid, dtype=int)
    pattern, canvas = _extract_pattern_and_canvas(g)
    centers = _solve_centers(pattern, canvas)
    return _anchors_to_output(canvas.shape, centers)


def _make_pattern() -> np.ndarray:
    while True:
        size = random.choice((3, 5))
        palette_size = random.randint(3, min(len(PATTERN_COLOR_POOL), size + 2))
        palette = random.sample(list(PATTERN_COLOR_POOL), palette_size)
        pattern = np.array(
            [[random.choice(palette) for _ in range(size)] for _ in range(size)],
            dtype=int,
        )
        corners = {
            int(pattern[0, 0]),
            int(pattern[0, -1]),
            int(pattern[-1, 0]),
            int(pattern[-1, -1]),
        }
        if len(set(map(int, np.unique(pattern)))) < 3:
            continue
        if len(corners) < 3:
            continue
        if not _is_asymmetric(pattern):
            continue
        return pattern


def _render_canvas(
    pattern: np.ndarray, shape: tuple[int, int], centers: list[tuple[int, int]]
) -> np.ndarray:
    h, w = shape
    k = pattern.shape[0]
    rad = k // 2
    canvas = np.full((h, w), BLACK, dtype=int)
    for cr, cc in centers:
        for pr in range(k):
            rr = cr + pr - rad
            if rr < 0 or rr >= h:
                continue
            for pc in range(k):
                cc2 = cc + pc - rad
                if 0 <= cc2 < w:
                    canvas[rr, cc2] = int(pattern[pr, pc])
    return canvas


def _footprint_cells(
    shape: tuple[int, int], center: tuple[int, int], size: int
) -> list[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    rad = size // 2
    cells = []
    for pr in range(size):
        rr = cr + pr - rad
        if rr < 0 or rr >= h:
            continue
        for pc in range(size):
            cc2 = cc + pc - rad
            if 0 <= cc2 < w:
                cells.append((int(rr), int(cc2)))
    return cells


def _analyze_visibility(
    shape: tuple[int, int], centers: list[tuple[int, int]], size: int
) -> list[dict]:
    covered: set[tuple[int, int]] = set()
    info = [None] * len(centers)
    for idx in range(len(centers) - 1, -1, -1):
        cells = _footprint_cells(shape, centers[idx], size)
        visible = [cell for cell in cells if cell not in covered]
        info[idx] = {
            "area": len(cells),
            "visible": len(visible),
            "clipped": len(cells) < size * size,
        }
        covered.update(cells)
    return info


def _overlap_count(shape: tuple[int, int], centers: list[tuple[int, int]], size: int) -> int:
    footprints = [set(_footprint_cells(shape, center, size)) for center in centers]
    overlaps = 0
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            if footprints[i] & footprints[j]:
                overlaps += 1
    return overlaps


def _choose_canvas_shape(size: int) -> tuple[int, int]:
    if size == 3:
        return random.randint(8, 14), random.randint(8, 14)
    return random.randint(9, 13), random.randint(9, 13)


def _random_center(h: int, w: int, rad: int, clipped: bool | None = None) -> tuple[int, int]:
    if clipped is None:
        return random.randint(-rad, h - 1 + rad), random.randint(-rad, w - 1 + rad)
    if clipped:
        side = random.choice(("top", "bottom", "left", "right"))
        if side == "top":
            return random.randint(-rad, -1), random.randint(-rad, w - 1 + rad)
        if side == "bottom":
            return random.randint(h, h - 1 + rad), random.randint(-rad, w - 1 + rad)
        if side == "left":
            return random.randint(-rad, h - 1 + rad), random.randint(-rad, -1)
        return random.randint(-rad, h - 1 + rad), random.randint(w, w - 1 + rad)
    return random.randint(0, h - 1), random.randint(0, w - 1)


def _sample_centers(shape: tuple[int, int], size: int) -> list[tuple[int, int]] | None:
    h, w = shape
    rad = size // 2
    n_stamps = random.randint(2, 4)
    want_clipped = bool(random.getrandbits(1))
    centers: list[tuple[int, int]] = []

    first = _random_center(h, w, rad, clipped=want_clipped if random.random() < 0.5 else None)
    centers.append(first)
    made_overlap = False

    for i in range(1, n_stamps):
        for _ in range(80):
            base = random.choice(centers)
            if not made_overlap or random.random() < 0.7:
                dr = random.randint(-(size - 1), size - 1)
                dc = random.randint(-(size - 1), size - 1)
                if dr == 0 and dc == 0:
                    continue
                cand = (base[0] + dr, base[1] + dc)
            else:
                need_clip = want_clipped and not any(
                    not (0 <= r < h and 0 <= c < w) for r, c in centers
                )
                cand = _random_center(h, w, rad, clipped=need_clip if random.random() < 0.8 else None)
            if not _footprint_cells(shape, cand, size):
                continue
            centers.append(cand)
            made_overlap = _overlap_count(shape, centers, size) > 0
            break
        else:
            return None

    if _overlap_count(shape, centers, size) == 0:
        return None
    if want_clipped and not any(not (0 <= r < h and 0 <= c < w) for r, c in centers):
        return None
    return centers


def _make_input_grid(pattern: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    ph, pw = pattern.shape
    h, w = canvas.shape
    grid = np.full((ph + 1 + h, w), BLACK, dtype=int)
    c0 = random.randint(0, w - pw)
    grid[:ph, c0 : c0 + pw] = pattern
    grid[ph, :] = GREY
    grid[ph + 1 :, :] = canvas
    return grid


def _true_solution_matches(
    pattern: np.ndarray, canvas: np.ndarray, centers: list[tuple[int, int]]
) -> bool:
    try:
        solved = sorted(_solve_centers(pattern, canvas))
    except AssertionError:
        return False
    return solved == sorted(centers)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pattern = _make_pattern()
        shape = _choose_canvas_shape(pattern.shape[0])
        centers = _sample_centers(shape, pattern.shape[0])
        if centers is None:
            continue

        canvas = _render_canvas(pattern, shape, centers)
        visible_info = _analyze_visibility(shape, centers, pattern.shape[0])
        fill_ratio = float(np.mean(canvas != BLACK))

        if any(item["visible"] < 2 for item in visible_info):
            continue
        if not any(item["visible"] < item["area"] for item in visible_info):
            continue
        if not (0.12 <= fill_ratio <= 0.6):
            continue
        if not _true_solution_matches(pattern, canvas, centers):
            continue

        grid = _make_input_grid(pattern, canvas)
        if not np.array_equal(main(grid), _anchors_to_output(shape, centers)):
            continue
        return grid

    raise RuntimeError("failed to generate a valid residue-to-anchor ARC input")
