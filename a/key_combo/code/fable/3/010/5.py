from __future__ import annotations

import random

import numpy as np


MISSING = 0
COLORS = list(range(1, 10))
Coord = tuple[int, int]


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return int(r0), int(r1), int(c0), int(c1)


def _canon_vec(vec: Coord) -> Coord:
    dr, dc = vec
    if dr < 0 or (dr == 0 and dc < 0):
        return -dr, -dc
    return dr, dc


def _best_translation(grid: np.ndarray, missing: int = MISSING) -> tuple[Coord, int]:
    h, w = grid.shape
    best_support = 0
    best: list[Coord] = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = _overlap_slices(grid.shape, dr, dc)
            if r0 >= r1 or c0 >= c1:
                continue
            a = grid[r0:r1, c0:c1]
            b = grid[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            visible = (a != missing) & (b != missing)
            support = int(np.count_nonzero(visible))
            if support == 0:
                continue
            if not bool(np.all(a[visible] == b[visible])):
                continue
            if support > best_support:
                best_support = support
                best = [(dr, dc)]
            elif support == best_support:
                best.append((dr, dc))

    assert best_support > 0, "expected a visible self-translation"
    canonical = {_canon_vec(vec) for vec in best}
    assert len(canonical) == 1, "strongest self-translation must be unique up to direction"
    return next(iter(canonical)), int(best_support)


def _fill_by_translation(partial: np.ndarray) -> np.ndarray:
    if not bool(np.any(partial == MISSING)):
        return partial.copy()

    h, w = partial.shape
    vec, _support = _best_translation(partial, MISSING)
    shifts = [vec, (-vec[0], -vec[1])]
    parent = list(range(h * w))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for dr, dc in shifts:
        r0, r1, c0, c1 = _overlap_slices(partial.shape, dr, dc)
        for r in range(r0, r1):
            for c in range(c0, c1):
                union(r * w + c, (r + dr) * w + c + dc)

    colors_by_root: dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(partial[r, c])
            if color == MISSING:
                continue
            colors_by_root.setdefault(find(r * w + c), set()).add(color)

    assert all(len(colors) == 1 for colors in colors_by_root.values()), "translation classes must be monochrome"

    out = partial.copy()
    for r, c in np.argwhere(partial == MISSING):
        root = find(int(r) * w + int(c))
        colors = colors_by_root.get(root, set())
        assert len(colors) == 1, "every missing translation class needs one visible color"
        out[int(r), int(c)] = next(iter(colors))
    return out


def _infer_bijection(first: np.ndarray, second: np.ndarray) -> tuple[dict[int, int], dict[int, int], int]:
    assert first.shape == second.shape
    forward: dict[int, int] = {}
    inverse: dict[int, int] = {}
    support = 0
    for r in range(first.shape[0]):
        for c in range(first.shape[1]):
            a = int(first[r, c])
            b = int(second[r, c])
            if a == MISSING or b == MISSING:
                continue
            support += 1
            if a in forward:
                assert forward[a] == b, "copy-axis color map has a conflict"
            else:
                forward[a] = b
            if b in inverse:
                assert inverse[b] == a, "copy-axis inverse color map has a conflict"
            else:
                inverse[b] = a

    first_colors = set(map(int, first[first != MISSING]))
    second_colors = set(map(int, second[second != MISSING]))
    assert len(first_colors) >= 3, "the first half must contain a real color system"
    assert set(forward) == first_colors, "every first-half color must be mapped by visible pairs"
    assert set(inverse) == second_colors, "every second-half color must be mapped by visible pairs"
    assert len(forward) == len(inverse) == len(first_colors) == len(second_colors)
    assert support >= 2 * len(forward), "the color bijection needs redundant evidence"
    return forward, inverse, int(support)


def _borrow_from_second(first: np.ndarray, second: np.ndarray, inverse: dict[int, int]) -> np.ndarray:
    out = first.copy()
    borrow = (out == MISSING) & (second != MISSING)
    for color in np.unique(second[borrow]):
        mapped = inverse.get(int(color))
        assert mapped is not None, "borrowed color must be in the recovered inverse map"
        out[borrow & (second == int(color))] = int(mapped)
    return out


def _solve_candidate(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, dict[int, int], int]:
    forward, inverse, support = _infer_bijection(first, second)
    partial = _borrow_from_second(first, second, inverse)
    out = _fill_by_translation(partial)
    assert not bool(np.any(out == MISSING)), "all missing cells must be recoverable"
    return out, forward, support


def _candidate_outputs(grid: np.ndarray) -> list[tuple[str, np.ndarray, dict[int, int], int]]:
    h, w = grid.shape
    candidates: list[tuple[str, np.ndarray, dict[int, int], int]] = []
    if w % 2 == 0:
        first = grid[:, : w // 2]
        second = grid[:, w // 2 :]
        try:
            out, cmap, support = _solve_candidate(first, second)
            candidates.append(("horizontal", out, cmap, support))
        except AssertionError:
            pass
    if h % 2 == 0:
        first = grid[: h // 2, :]
        second = grid[h // 2 :, :]
        try:
            out, cmap, support = _solve_candidate(first, second)
            candidates.append(("vertical", out, cmap, support))
        except AssertionError:
            pass
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover one repaired tile from two adjacent color-bijective damaged copies.

    Input structure:
    - The grid consists of two same-size adjacent copies of one rectangular tile,
      either left/right or top/bottom.
    - The second copy uses a uniform one-to-one replacement of the first copy's
      non-black colors.
    - Black cells are missing pixels in either copy; at a few tile positions both
      copies are missing.
    - The underlying tile has a smaller internal repeating translation.

    Rule:
    - Choose the split axis whose aligned visible cell pairs give one consistent
      complete color bijection; the other possible split has mapping conflicts.
    - Output the first copy's color system as a single tile.
    - Keep visible first-copy pixels, fill first-copy black pixels from the second
      copy through the inverse color map, and fill positions black in both copies
      from the strongest no-conflict self-translation of the partially restored tile.

    Key insight:
    - The two halves look unrelated until their colors are decoded as a bijection;
      only after that cross-color repair do the remaining holes expose the tile's
      internal period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    candidates = _candidate_outputs(g)
    assert len(candidates) == 1, "exactly one copy axis must be valid"
    return candidates[0][1].astype(int, copy=True)


# --------------- generator ---------------

_GEN_RETRIES = 800
_MASK_RETRIES = 200


def _apply_map(tile: np.ndarray, cmap: dict[int, int]) -> np.ndarray:
    out = np.empty_like(tile)
    for color, mapped in cmap.items():
        out[tile == int(color)] = int(mapped)
    return out


def _tile_with_period(
    h: int,
    w: int,
    palette: list[int],
    period_axis: str,
) -> tuple[np.ndarray, Coord] | None:
    for _ in range(80):
        if period_axis == "cols":
            choices = [p for p in range(2, min(5, w)) if w >= 2 * p + 1]
            if not choices:
                return None
            period = random.choice(choices)
            base = np.array(random.choices(palette, k=h * period), dtype=int).reshape(h, period)
            for color in palette:
                rr = random.randrange(h)
                cc = random.randrange(period)
                base[rr, cc] = int(color)
            tile = base[:, np.arange(w) % period]
            vec = (0, period)
        else:
            choices = [p for p in range(2, min(5, h)) if h >= 2 * p + 1]
            if not choices:
                return None
            period = random.choice(choices)
            base = np.array(random.choices(palette, k=period * w), dtype=int).reshape(period, w)
            for color in palette:
                rr = random.randrange(period)
                cc = random.randrange(w)
                base[rr, cc] = int(color)
            tile = base[np.arange(h) % period, :]
            vec = (period, 0)

        if set(map(int, np.unique(tile))) != set(map(int, palette)):
            continue
        try:
            best, support = _best_translation(tile, MISSING)
        except AssertionError:
            continue
        if best == _canon_vec(vec) and support >= (h * w) // 3:
            return tile.astype(int), _canon_vec(vec)
    return None


def _choose_geometry(copy_axis: str, period_axis: str) -> tuple[int, int] | None:
    heights = [h for h in range(6, 15) if h % 2 == 0]
    widths = [w for w in range(6, 15) if w % 2 == 0]
    if copy_axis == "horizontal":
        h_choices = heights
        w_choices = list(range(6, 15))
    else:
        h_choices = list(range(6, 15))
        w_choices = widths

    legal: list[tuple[int, int]] = []
    for h in h_choices:
        for w in w_choices:
            if copy_axis == "horizontal" and 2 * w > 30:
                continue
            if copy_axis == "vertical" and 2 * h > 30:
                continue
            if period_axis == "cols" and not any(w >= 2 * p + 1 for p in range(2, min(5, w))):
                continue
            if period_axis == "rows" and not any(h >= 2 * p + 1 for p in range(2, min(5, h))):
                continue
            legal.append((h, w))
    return random.choice(legal) if legal else None


def _translation_class(coord: Coord, shape: tuple[int, int], vec: Coord) -> set[Coord]:
    h, w = shape
    dr, dc = vec
    out: set[Coord] = set()
    for k in range(-max(h, w), max(h, w) + 1):
        r = coord[0] + k * dr
        c = coord[1] + k * dc
        if 0 <= r < h and 0 <= c < w:
            out.add((r, c))
    return out


def _random_masks(tile: np.ndarray, vec: Coord) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = tile.shape
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    area = h * w
    common_count = random.randint(1, max(1, min(4, area // 18)))
    first_only_count = random.randint(2, max(3, min(8, area // 8)))
    second_only_count = random.randint(2, max(3, min(8, area // 8)))

    for _ in range(_MASK_RETRIES):
        shuffled = all_cells[:]
        random.shuffle(shuffled)
        common = set(shuffled[:common_count])
        idx = common_count
        first_only = set(shuffled[idx : idx + first_only_count])
        idx += first_only_count
        second_only = set(shuffled[idx : idx + second_only_count])

        mask1 = np.zeros(tile.shape, dtype=bool)
        mask2 = np.zeros(tile.shape, dtype=bool)
        for r, c in common | first_only:
            mask1[r, c] = True
        for r, c in common | second_only:
            mask2[r, c] = True

        # Mapping evidence: every color must have at least one aligned visible pair.
        ok = True
        for color in np.unique(tile):
            visible_pair = (tile == int(color)) & (~mask1) & (~mask2)
            if int(np.count_nonzero(visible_pair)) < 2:
                ok = False
                break
        if not ok:
            continue

        # Fallback evidence: no common hole may be the only member of its class.
        for cell in common:
            cls = _translation_class(cell, tile.shape, vec)
            if cls <= common:
                ok = False
                break
        if not ok:
            continue
        return mask1, mask2
    return None


def _split_halves(inp: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    if axis == "horizontal":
        return inp[:, : inp.shape[1] // 2], inp[:, inp.shape[1] // 2 :]
    return inp[: inp.shape[0] // 2, :], inp[inp.shape[0] // 2 :, :]


def _build_input(
    tile: np.ndarray,
    recolored: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    axis: str,
) -> np.ndarray:
    first = tile.copy()
    second = recolored.copy()
    first[mask1] = MISSING
    second[mask2] = MISSING
    if axis == "horizontal":
        return np.concatenate([first, second], axis=1).astype(int)
    return np.concatenate([first, second], axis=0).astype(int)


def _semantic_checks(
    inp: np.ndarray,
    tile: np.ndarray,
    cmap: dict[int, int],
    axis: str,
    mask1: np.ndarray,
    mask2: np.ndarray,
    vec: Coord,
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or max(tile.shape) > 30:
        return False
    first, second = _split_halves(inp, axis)

    try:
        inferred, inverse, _support = _infer_bijection(first, second)
    except AssertionError:
        return False
    if inferred != cmap:
        return False

    first_only = mask1 & ~mask2
    second_only = mask2 & ~mask1
    common = mask1 & mask2
    if int(np.count_nonzero(first_only)) < 2:
        return False
    if int(np.count_nonzero(second_only)) < 2:
        return False
    if int(np.count_nonzero(common)) < 1:
        return False
    if len(set(map(int, tile[first_only]))) < 2:
        return False

    partial = _borrow_from_second(first, second, inverse)
    if int(np.count_nonzero(partial == MISSING)) != int(np.count_nonzero(common)):
        return False
    try:
        best, _support = _best_translation(partial, MISSING)
        repaired = _fill_by_translation(partial)
    except AssertionError:
        return False
    if best != _canon_vec(vec):
        return False
    if not np.array_equal(repaired, tile):
        return False

    candidates = _candidate_outputs(inp)
    if len(candidates) != 1 or candidates[0][0] != axis:
        return False
    if not np.array_equal(candidates[0][1], tile):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    copy_axis = random.choice(["horizontal", "vertical"])
    period_axis = random.choice(["cols", "rows"])
    geom = _choose_geometry(copy_axis, period_axis)
    if geom is None:
        return None
    h, w = geom

    color_count = random.choice([3, 4])
    picked = random.sample(COLORS, 2 * color_count)
    first_palette = picked[:color_count]
    second_palette = picked[color_count:]
    random.shuffle(second_palette)
    cmap = {int(a): int(b) for a, b in zip(first_palette, second_palette)}

    made = _tile_with_period(h, w, first_palette, period_axis)
    if made is None:
        return None
    tile, vec = made
    recolored = _apply_map(tile, cmap)

    masks = _random_masks(tile, vec)
    if masks is None:
        return None
    mask1, mask2 = masks
    inp = _build_input(tile, recolored, mask1, mask2, copy_axis)

    if not _semantic_checks(inp, tile, cmap, copy_axis, mask1, mask2, vec):
        return None
    try:
        out = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(out, tile):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
