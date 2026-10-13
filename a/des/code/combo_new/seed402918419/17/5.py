from __future__ import annotations

import random

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_GEN_MAX_ATTEMPTS = 400
_GEN_INNER_SIZE_RANGE = (11, 25)
_GEN_WINDOW_MIN_SPAN = 3
_GEN_PATCH_MAX_SPAN = 5
_GEN_PALETTE = tuple(range(10))


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _segment_bounds(line: np.ndarray, ruler_color: int) -> tuple[int, int]:
    indices = np.flatnonzero(np.asarray(line, dtype=int) != int(ruler_color))
    assert indices.size > 0
    start = int(indices[0])
    end = int(indices[-1])
    assert np.all(np.asarray(line[start : end + 1], dtype=int) != int(ruler_color))
    return start, end


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            for rr, cc in queue:
                comp.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter interior objects using the row and column ranges encoded by edge rulers.

    Input structure:
    - The grid has a dominant background color in the interior plus a top ruler row and a left ruler column.
    - The ruler cells are mostly one border color, with one contiguous contrasting segment on each ruler.
    - The top-left corner belongs to the ruler border color.
    - The interior region contains several non-background 4-connected colored objects on the background.

    Rule:
    - Remove the top ruler row and the left ruler column; the output keeps only the interior area.
    - Read the contrasting segment on the left ruler as an inclusive row interval in interior coordinates.
    - Read the contrasting segment on the top ruler as an inclusive column interval in interior coordinates.
    - For each non-background 4-connected interior object, keep it only if every one of its pixels lies inside the rectangle defined by those row and column intervals; otherwise erase the whole object to background.

    Key insight: The edge markers do not crop the canvas; they define a rectangular keep-window that filters whole objects by full containment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    inner = g[1:, 1:].copy()
    bg = _most_common_color(inner)
    ruler_color = int(g[0, 0])
    r1, r2 = _segment_bounds(g[1:, 0], ruler_color)
    c1, c2 = _segment_bounds(g[0, 1:], ruler_color)

    out = inner.copy()
    for comp in _components(inner != int(bg)):
        if all(r1 <= r <= r2 and c1 <= c <= c2 for r, c in comp):
            continue
        for r, c in comp:
            out[r, c] = int(bg)
    return out


def _random_connected_patch(rng: random.Random, min_cells: int, max_cells: int) -> np.ndarray:
    for _ in range(80):
        ph = rng.randint(1, _GEN_PATCH_MAX_SPAN)
        pw = rng.randint(1, _GEN_PATCH_MAX_SPAN)
        area = ph * pw
        if area < min_cells:
            continue
        target = rng.randint(min_cells, min(max_cells, area))
        seed = (rng.randrange(ph), rng.randrange(pw))
        cells = {seed}
        frontier: set[tuple[int, int]] = set()
        while len(cells) < target:
            sr, sc = rng.choice(tuple(cells))
            options = []
            for dr, dc in _NEIGHBORS4:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cells:
                    options.append((nr, nc))
            frontier.update(options)
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
            frontier -= cells
        if len(cells) != target:
            continue
        mask = np.zeros((ph, pw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    raise RuntimeError("failed to sample connected patch")


def _make_patch(category: str, rng: random.Random) -> np.ndarray:
    min_cells = 2 if category == "cross" else 1
    max_cells = 7 if category == "cross" else 6
    mode = rng.choice(["poly", "rect", "line"])
    if mode == "rect":
        for _ in range(40):
            ph = rng.randint(1, 4 if category == "cross" else 3)
            pw = rng.randint(1, 4 if category == "cross" else 3)
            if ph * pw >= min_cells:
                return np.ones((ph, pw), dtype=bool)
    if mode == "line":
        length = rng.randint(min_cells, _GEN_PATCH_MAX_SPAN)
        if rng.random() < 0.5:
            return np.ones((1, length), dtype=bool)
        return np.ones((length, 1), dtype=bool)
    return _random_connected_patch(rng, min_cells, max_cells)


def _origin_candidates(
    inner_h: int,
    inner_w: int,
    patch_h: int,
    patch_w: int,
    window: tuple[int, int, int, int],
    category: str,
) -> list[tuple[int, int]]:
    r1, r2, c1, c2 = window
    out: list[tuple[int, int]] = []
    for top in range(inner_h - patch_h + 1):
        bottom = top + patch_h - 1
        for left in range(inner_w - patch_w + 1):
            right = left + patch_w - 1
            fully_inside = top >= r1 and bottom <= r2 and left >= c1 and right <= c2
            disjoint = bottom < r1 or top > r2 or right < c1 or left > c2
            if category == "keep" and fully_inside:
                out.append((top, left))
            elif category == "outside" and disjoint:
                out.append((top, left))
            elif category == "cross" and not fully_inside and not disjoint:
                out.append((top, left))
    return out


def _placed_cells(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    for rr, cc in np.argwhere(patch):
        cells.append((int(top + rr), int(left + cc)))
    return cells


def _fits_reserved(cells: list[tuple[int, int]], reserved: np.ndarray) -> bool:
    return all(not bool(reserved[r, c]) for r, c in cells)


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for dr, dc in _NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                reserved[nr, nc] = True
        reserved[r, c] = True


def _inside_window(cells: list[tuple[int, int]], window: tuple[int, int, int, int]) -> list[bool]:
    r1, r2, c1, c2 = window
    return [r1 <= r <= r2 and c1 <= c <= c2 for r, c in cells]


def _paint_object(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    bg: int,
    rng: random.Random,
) -> None:
    palette = [color for color in _GEN_PALETTE if color != int(bg)]
    color_count = rng.randint(1, min(3, len(cells), len(palette)))
    colors = rng.sample(palette, color_count)
    shuffled = cells[:]
    rng.shuffle(shuffled)
    for idx, (r, c) in enumerate(shuffled):
        if idx < color_count:
            grid[r, c] = int(colors[idx])
        else:
            grid[r, c] = int(rng.choice(colors))


def _sample_window(inner_h: int, inner_w: int, rng: random.Random) -> tuple[int, int, int, int]:
    window_h = rng.randint(_GEN_WINDOW_MIN_SPAN, inner_h - 5)
    window_w = rng.randint(_GEN_WINDOW_MIN_SPAN, inner_w - 5)
    top = rng.randint(1, inner_h - window_h - 1)
    left = rng.randint(1, inner_w - window_w - 1)
    return top, top + window_h - 1, left, left + window_w - 1


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    inner_h = rng.randint(*_GEN_INNER_SIZE_RANGE)
    inner_w = rng.randint(*_GEN_INNER_SIZE_RANGE)
    bg, ruler_color, mark_color = rng.sample(_GEN_PALETTE, 3)
    window = _sample_window(inner_h, inner_w, rng)

    inner = np.full((inner_h, inner_w), int(bg), dtype=int)
    expected = inner.copy()
    reserved = np.zeros((inner_h, inner_w), dtype=bool)

    categories = ["cross"] * rng.randint(1, 2)
    categories += ["keep"] * rng.randint(1, 3)
    categories += ["outside"] * rng.randint(1, 2)
    categories.sort(key={"cross": 0, "keep": 1, "outside": 2}.get)

    kept = 0
    removed = 0
    crossed = 0
    for category in categories:
        placed = False
        for _ in range(120):
            patch = _make_patch(category, rng)
            origins = _origin_candidates(inner_h, inner_w, patch.shape[0], patch.shape[1], window, category)
            if not origins:
                continue
            rng.shuffle(origins)
            for top, left in origins:
                cells = _placed_cells(patch, top, left)
                if not _fits_reserved(cells, reserved):
                    continue
                mask = _inside_window(cells, window)
                if category == "keep" and not all(mask):
                    continue
                if category == "outside" and any(mask):
                    continue
                if category == "cross" and (all(mask) or not any(mask)):
                    continue
                _paint_object(inner, cells, int(bg), rng)
                if category == "keep":
                    rows = [r for r, _ in cells]
                    cols = [c for _, c in cells]
                    expected[rows, cols] = inner[rows, cols]
                    kept += 1
                else:
                    removed += 1
                    if category == "cross":
                        crossed += 1
                _mark_reserved(reserved, cells)
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError(f"failed to place {category} object")

    assert kept >= 1 and removed >= 1 and crossed >= 1

    top, bottom, left, right = window
    full = np.full((inner_h + 1, inner_w + 1), int(bg), dtype=int)
    full[0, :] = int(ruler_color)
    full[:, 0] = int(ruler_color)
    full[top + 1 : bottom + 2, 0] = int(mark_color)
    full[0, left + 1 : right + 2] = int(mark_color)
    full[1:, 1:] = inner
    return full, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_MAX_ATTEMPTS):
        try:
            inp, expected = _build_case(rng)
        except RuntimeError:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate valid ruler-window task input")
