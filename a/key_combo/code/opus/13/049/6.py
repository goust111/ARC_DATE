import numpy as np


BLUE = 1
BLACK = 0
NON_SPECIAL_COLORS = tuple(range(2, 10))
PERIOD_CANDIDATES = ((2, 2), (2, 3))
GEN_MAX_ATTEMPTS = 400


def _mode_color(values: np.ndarray, *, require_unique: bool) -> tuple[int, int]:
    counts = np.bincount(values.astype(int), minlength=10)
    best = int(counts.max())
    assert best > 0, "each residue class must have at least one observed color"
    winners = np.flatnonzero(counts == best)
    if require_unique:
        assert len(winners) == 1, "period color must be uniquely determined"
    return int(winners[0]), best


def _infer_tile_and_support(
    grid: np.ndarray, period: tuple[int, int], *, require_unique: bool
) -> tuple[np.ndarray, int] | None:
    pr, pc = map(int, period)
    h, w = grid.shape
    if pr > h or pc > w:
        return None

    tile = np.zeros((pr, pc), dtype=int)
    support = 0
    for rr in range(pr):
        for cc in range(pc):
            residue = grid[rr:h:pr, cc:w:pc]
            residue = residue[residue != BLACK]
            if residue.size == 0:
                return None
            color, count = _mode_color(residue, require_unique=require_unique)
            tile[rr, cc] = color
            support += count
    return tile, int(support)


def _best_period(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray, int]:
    scored: list[tuple[int, int, int, np.ndarray]] = []
    for pr, pc in PERIOD_CANDIDATES:
        inferred = _infer_tile_and_support(grid, (pr, pc), require_unique=False)
        if inferred is None:
            continue
        tile, support = inferred
        scored.append((support, pr * pc, pr, tile))

    assert scored, "no valid period candidate found"
    scored.sort(key=lambda item: (-item[0], item[1]))
    support, _area, pr, _tile = scored[0]
    chosen = (pr, scored[0][3].shape[1])
    unique_inferred = _infer_tile_and_support(grid, chosen, require_unique=True)
    assert unique_inferred is not None
    tile, support = unique_inferred
    return chosen, tile, int(support)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal hidden deviation pixels by inferring the carrier's smallest repeating tile.

    Input structure:
    - The whole grid is filled by a near-periodic color carrier with no empty border.
    - Most non-black pixels follow a 2x2 or 2x3 repeating tile whose colors never include blue.
    - A few black pixels are missing carrier samples, and a few non-black pixels are deliberate off-period deviations.

    Rule:
    - Infer whether the carrier period is 2x2 or 2x3 by majority vote over non-black pixels in each residue class.
    - For each residue class, the most frequent non-black color is the expected carrier color.
    - Fill black holes with that expected color, keep matching pixels unchanged, and recolor every non-black mismatch to blue.

    Key insight:
    - The black cells are missing data to repair, but the non-black mismatches are the real signal: once the periodic carrier is reconstructed, those deviations become the hidden blue pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    (pr, pc), tile, _support = _best_period(g)
    h, w = g.shape
    out = g.copy()
    for r in range(h):
        for c in range(w):
            expected = int(tile[r % pr, c % pc])
            if out[r, c] == BLACK:
                out[r, c] = expected
            elif out[r, c] != expected:
                out[r, c] = BLUE
    return out


def _has_smaller_period(tile: np.ndarray) -> bool:
    th, tw = tile.shape
    for pr in range(1, th + 1):
        for pc in range(1, tw + 1):
            if pr == th and pc == tw:
                continue
            if pr * pc >= th * tw:
                continue
            ok = True
            for r in range(th):
                for c in range(tw):
                    if tile[r, c] != tile[r % pr, c % pc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _sample_tile(rng: np.random.Generator, period: tuple[int, int]) -> np.ndarray:
    pr, pc = period
    for _ in range(200):
        n_colors = int(rng.integers(2, 4))
        palette = rng.choice(NON_SPECIAL_COLORS, size=n_colors, replace=False)
        tile = rng.choice(palette, size=(pr, pc), replace=True).astype(int)
        if len(set(map(int, tile.flat))) < 2:
            continue
        if not set(map(int, palette)).issubset(set(map(int, tile.flat))):
            continue
        if _has_smaller_period(tile):
            continue
        return tile
    raise RuntimeError("failed to sample a minimal tile")


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pr, pc = tile.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % pr, c % pc])
    return out


def _neighbors4(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    h, w = shape
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _self_avoiding_path(
    rng: np.random.Generator, target_len: int, box_h: int, box_w: int
) -> set[tuple[int, int]] | None:
    for _ in range(120):
        start = (int(rng.integers(0, box_h)), int(rng.integers(0, box_w)))
        cells = [start]
        used = {start}
        prev_dir = None
        while len(cells) < target_len:
            r, c = cells[-1]
            choices = []
            weights = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < box_h and 0 <= nc < box_w):
                    continue
                if (nr, nc) in used:
                    continue
                choices.append((nr, nc))
                weights.append(3.0 if prev_dir == (dr, dc) else 1.0)
            if not choices:
                break
            weights_arr = np.asarray(weights, dtype=float)
            weights_arr /= weights_arr.sum()
            idx = int(rng.choice(len(choices), p=weights_arr))
            nr, nc = choices[idx]
            prev_dir = (nr - r, nc - c)
            cells.append((nr, nc))
            used.add((nr, nc))
        if len(used) == target_len:
            return used
    return None


def _rect_frame_cells(box_h: int, box_w: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(box_h):
        for c in range(box_w):
            if r in (0, box_h - 1) or c in (0, box_w - 1):
                cells.add((r, c))
    return cells


def _sample_pattern(rng: np.random.Generator, shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    area = h * w
    min_cells = max(8, int(np.ceil(area * 0.09)))
    max_cells = max(min_cells, int(np.floor(area * 0.15)))
    target = int(rng.integers(min_cells, max_cells + 1))

    for _ in range(160):
        if rng.random() < 0.4:
            frame_options = []
            for box_h in range(3, 7):
                for box_w in range(3, 7):
                    if box_h > h or box_w > w:
                        continue
                    cells = _rect_frame_cells(box_h, box_w)
                    if len(cells) == target:
                        frame_options.append((box_h, box_w, cells))
            if frame_options:
                box_h, box_w, local = frame_options[int(rng.integers(len(frame_options)))]
                r0 = int(rng.integers(0, h - box_h + 1))
                c0 = int(rng.integers(0, w - box_w + 1))
                return {(r0 + r, c0 + c) for r, c in local}

        path_boxes = [
            (box_h, box_w)
            for box_h in range(4, min(8, h) + 1)
            for box_w in range(4, min(8, w) + 1)
            if box_h * box_w >= target
        ]
        if not path_boxes:
            continue
        box_h, box_w = path_boxes[int(rng.integers(len(path_boxes)))]
        local = _self_avoiding_path(rng, target, box_h, box_w)
        if local is None:
            continue
        r0 = int(rng.integers(0, h - box_h + 1))
        c0 = int(rng.integers(0, w - box_w + 1))
        return {(r0 + r, c0 + c) for r, c in local}
    return None


def _is_connected(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        cell = stack.pop()
        for nbr in _neighbors4(cell, shape):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return seen == cells


def _sample_holes(
    rng: np.random.Generator, shape: tuple[int, int], pattern: set[tuple[int, int]]
) -> set[tuple[int, int]] | None:
    h, w = shape
    forbidden = set(pattern)
    for cell in pattern:
        forbidden.update(_neighbors4(cell, shape))

    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    if len(candidates) < 3:
        return None

    min_holes = max(3, int(np.ceil(h * w * 0.035)))
    max_holes = min(int(np.floor(h * w * 0.06)), len(candidates))
    if max_holes < min_holes:
        return None
    count = int(rng.integers(min_holes, max_holes + 1))
    chosen = rng.choice(len(candidates), size=count, replace=False)
    return {candidates[int(idx)] for idx in np.atleast_1d(chosen)}


def _paint_deviations(
    rng: np.random.Generator,
    clean: np.ndarray,
    pattern: set[tuple[int, int]],
    tile_palette: set[int],
) -> tuple[np.ndarray, set[int]]:
    inp = clean.copy()
    replacement_colors: list[int] = []
    for r, c in sorted(pattern):
        expected = int(clean[r, c])
        if rng.random() < 0.65 and len(tile_palette - {expected}) > 0:
            choices = sorted(tile_palette - {expected})
        else:
            choices = [color for color in NON_SPECIAL_COLORS if color != expected]
        new_color = int(choices[int(rng.integers(len(choices)))])
        inp[r, c] = new_color
        replacement_colors.append(new_color)
    return inp, set(replacement_colors)


def _oracle_output(
    clean: np.ndarray, pattern: set[tuple[int, int]], holes: set[tuple[int, int]]
) -> np.ndarray:
    out = clean.copy()
    for r, c in pattern:
        out[r, c] = BLUE
    for r, c in holes:
        out[r, c] = int(clean[r, c])
    return out


def _support_by_period(grid: np.ndarray, period: tuple[int, int]) -> tuple[np.ndarray, int]:
    inferred = _infer_tile_and_support(grid, period, require_unique=True)
    assert inferred is not None
    return inferred


def _validate_instance(
    inp: np.ndarray,
    clean: np.ndarray,
    expected: np.ndarray,
    period: tuple[int, int],
    pattern: set[tuple[int, int]],
    holes: set[tuple[int, int]],
    replacement_colors: set[int],
) -> bool:
    h, w = inp.shape
    if max(h, w) > 30:
        return False
    if not pattern or not holes:
        return False
    if not _is_connected(pattern, inp.shape):
        return False
    if any(inp[r, c] != BLACK for r, c in holes):
        return False
    if any(expected[r, c] != BLUE for r, c in pattern):
        return False
    if any(inp[r, c] == BLUE for r, c in pattern):
        return False
    for hole in holes:
        for nbr in _neighbors4(hole, inp.shape):
            if nbr in pattern:
                return False

    pr, pc = period
    tile = clean[:pr, :pc].copy()
    true_tile, true_support = _support_by_period(inp, period)
    if not np.array_equal(true_tile, tile):
        return False

    known_count = int(np.sum(inp != BLACK))
    if true_support != known_count - len(pattern):
        return False

    scored = []
    for candidate in PERIOD_CANDIDATES:
        inferred = _infer_tile_and_support(inp, candidate, require_unique=False)
        if inferred is None:
            return False
        cand_tile, cand_support = inferred
        scored.append((candidate, cand_tile, cand_support))

    scored.sort(key=lambda item: (-item[2], item[0][0] * item[0][1]))
    if scored[0][0] != period:
        return False
    if len(scored) > 1 and scored[0][2] - scored[1][2] < max(4, len(pattern) // 3):
        return False

    pattern_residues = {(r % pr, c % pc) for r, c in pattern}
    if len(pattern_residues) < min(3, pr * pc):
        return False
    if len(replacement_colors) < 2:
        return False

    repaired = main(inp)
    if not np.array_equal(repaired, expected):
        return False
    return True


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 17))
    w = int(rng.integers(12, 18))
    period = PERIOD_CANDIDATES[int(rng.integers(len(PERIOD_CANDIDATES)))]

    tile = _sample_tile(rng, period)
    clean = _repeat_tile(tile, (h, w))
    pattern = _sample_pattern(rng, (h, w))
    if pattern is None:
        return None
    holes = _sample_holes(rng, (h, w), pattern)
    if holes is None:
        return None

    inp, replacement_colors = _paint_deviations(rng, clean, pattern, set(map(int, tile.flat)))
    for r, c in holes:
        inp[r, c] = BLACK
    expected = _oracle_output(clean, pattern, holes)

    if not _validate_instance(inp, clean, expected, period, pattern, holes, replacement_colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-deviation puzzle")
