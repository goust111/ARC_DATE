from __future__ import annotations

from itertools import combinations
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_RETRIES = 400


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((int(start), int(idx)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(mask))))
    return runs


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in arr.ravel())


def _candidate_offsets(existing: np.ndarray, active_mask: np.ndarray) -> list[tuple[int, int]]:
    n_rows, n_cols = existing.shape
    mask_h, mask_w = active_mask.shape
    minority_cells = [tuple(map(int, rc)) for rc in np.argwhere(existing)]
    candidates: list[tuple[int, int]] = []

    for top in range(n_rows - mask_h + 1):
        for left in range(n_cols - mask_w + 1):
            ok = True
            for r, c in minority_cells:
                rr = r - top
                cc = c - left
                if not (0 <= rr < mask_h and 0 <= cc < mask_w and active_mask[rr, cc]):
                    ok = False
                    break
            if ok:
                candidates.append((int(top), int(left)))
    return candidates


def _analyze_with_separator(g: np.ndarray, separator: int) -> dict | None:
    row_sep = np.all(g == separator, axis=1)
    col_sep = np.all(g == separator, axis=0)
    row_runs = _runs(~row_sep)
    col_runs = _runs(~col_sep)

    if len(row_runs) < 2 or len(col_runs) < 2 or len(row_runs) != len(col_runs):
        return None
    if len({r1 - r0 for r0, r1 in row_runs}) != 1:
        return None
    if len({c1 - c0 for c0, c1 in col_runs}) != 1:
        return None

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    counts: dict[tuple[int, ...], int] = {}
    cell_keys: list[list[tuple[int, ...]]] = []
    for r0, r1 in row_runs:
        row_keys: list[tuple[int, ...]] = []
        for c0, c1 in col_runs:
            patch = g[r0:r1, c0:c1].copy()
            key = _pattern_key(patch)
            patterns.setdefault(key, patch)
            counts[key] = counts.get(key, 0) + 1
            row_keys.append(key)
        cell_keys.append(row_keys)

    if len(counts) != 2:
        return None

    ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    minority_key, minority_count = ordered[0]
    majority_key, majority_count = ordered[-1]
    if minority_count <= 0 or majority_count <= minority_count:
        return None

    majority = patterns[majority_key]
    minority = patterns[minority_key]
    active_mask = minority != majority
    if not active_mask.any() or active_mask.all():
        return None

    existing = np.zeros((len(row_runs), len(col_runs)), dtype=bool)
    for r in range(len(row_runs)):
        for c in range(len(col_runs)):
            existing[r, c] = cell_keys[r][c] == minority_key

    placements = _candidate_offsets(existing, active_mask)
    if len(placements) != 1:
        return None

    return {
        "separator": int(separator),
        "row_runs": row_runs,
        "col_runs": col_runs,
        "majority": majority,
        "minority": minority,
        "active_mask": active_mask,
        "top": int(placements[0][0]),
        "left": int(placements[0][1]),
        "score": int(row_sep.sum() + col_sep.sum()),
    }


def _infer_model(g: np.ndarray) -> dict:
    models = []
    for color in np.unique(g):
        model = _analyze_with_separator(g, int(color))
        if model is not None:
            models.append(model)
    assert models, "failed to infer the separated cell grid"
    models.sort(
        key=lambda m: (
            m["score"],
            len(m["row_runs"]) * len(m["col_runs"]),
            m["active_mask"].sum(),
        ),
        reverse=True,
    )
    return models[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a sparse minority cell pattern fractally across the cell grid.

    The minority cell differs from the majority cell only at the blueprint
    pixels; those differing pixels act as the active/inactive code when scaled
    up from pixel level to cell level.
    """
    g = np.asarray(grid, dtype=int)
    model = _infer_model(g)

    out = g.copy()
    row_runs = model["row_runs"]
    col_runs = model["col_runs"]
    active_mask = model["active_mask"]
    majority = model["majority"]
    minority = model["minority"]
    top = model["top"]
    left = model["left"]

    for r in range(active_mask.shape[0]):
        for c in range(active_mask.shape[1]):
            patch = minority if active_mask[r, c] else majority
            rr0, rr1 = row_runs[top + r]
            cc0, cc1 = col_runs[left + c]
            out[rr0:rr1, cc0:cc1] = patch

    return out


def _cell_spans(count: int, inner: int, bordered: bool) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []

    if bordered:
        separators.append(0)
        for idx in range(count):
            start = 1 + idx * (inner + 1)
            spans.append((int(start), int(start + inner)))
            separators.append(int(start + inner))
        total = separators[-1] + 1
    else:
        for idx in range(count):
            start = idx * (inner + 1)
            spans.append((int(start), int(start + inner)))
            if idx < count - 1:
                separators.append(int(start + inner))
        total = spans[-1][1]

    return spans, separators, int(total)


def _sample_majority_pattern(inner: int, bg: int, fg: int) -> np.ndarray:
    target = random.randint(inner, min(inner * inner - 2, inner * 2))
    cells: set[tuple[int, int]] = set()

    for r in range(inner):
        cells.add((r, random.randrange(inner)))
    for c in range(inner):
        cells.add((random.randrange(inner), c))
    while len(cells) < target:
        cells.add((random.randrange(inner), random.randrange(inner)))

    arr = np.full((inner, inner), bg, dtype=int)
    for r, c in cells:
        arr[r, c] = fg
    return arr


def _grow_connected_mask(inner: int, count: int) -> np.ndarray:
    seed = (random.randrange(inner), random.randrange(inner))
    cells = {seed}
    while len(cells) < count:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < inner and 0 <= cc < inner and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))

    mask = np.zeros((inner, inner), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_active_mask(inner: int) -> np.ndarray:
    area = inner * inner
    active_count = random.randint(max(2, inner - 1), min(area - 2, inner + 2))

    for _ in range(120):
        if random.random() < 0.5:
            mask = _grow_connected_mask(inner, active_count)
        else:
            mask = np.zeros((inner, inner), dtype=bool)
            cells = random.sample([(r, c) for r in range(inner) for c in range(inner)], active_count)
            for r, c in cells:
                mask[r, c] = True
        rows = {int(r) for r, _c in np.argwhere(mask)}
        cols = {int(c) for _r, c in np.argwhere(mask)}
        if len(rows) >= 2 and len(cols) >= 2:
            return mask

    raise RuntimeError("failed to sample a valid active mask")


def _build_grid(
    cell_count: int,
    inner: int,
    bordered: bool,
    bg: int,
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    minority_cells: set[tuple[int, int]],
) -> np.ndarray:
    row_spans, row_seps, height = _cell_spans(cell_count, inner, bordered)
    col_spans, col_seps, width = _cell_spans(cell_count, inner, bordered)
    grid = np.full((height, width), bg, dtype=int)

    for r in row_seps:
        grid[r, :] = separator
    for c in col_seps:
        grid[:, c] = separator

    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            patch = minority if (rr, cc) in minority_cells else majority
            grid[r0:r1, c0:c1] = patch

    return grid


def _offset_candidates(cell_count: int, active_mask: np.ndarray, given_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    existing = np.zeros((cell_count, cell_count), dtype=bool)
    for r, c in given_cells:
        existing[r, c] = True
    return _candidate_offsets(existing, active_mask)


def _choose_given_cells(
    cell_count: int,
    active_mask: np.ndarray,
    top: int,
    left: int,
) -> set[tuple[int, int]] | None:
    active_local = [tuple(map(int, rc)) for rc in np.argwhere(active_mask)]
    sizes = list(range(1, len(active_local)))
    random.shuffle(sizes)

    for size in sizes:
        combos = list(combinations(active_local, size))
        random.shuffle(combos)
        for combo in combos:
            given_cells = {(top + r, left + c) for r, c in combo}
            if _offset_candidates(cell_count, active_mask, given_cells) == [(top, left)]:
                return given_cells
    return None


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return all(int(counts[bg]) > int(counts[color]) for color in range(10) if color != bg)


def _gen_attempt() -> np.ndarray | None:
    inner = random.randint(3, 5)
    cell_count = inner + random.randint(1, 3)
    bordered = random.random() < 0.5

    bg, separator, fg, accent = random.sample(range(10), 4)
    majority = _sample_majority_pattern(inner, bg, fg)
    active_mask = _sample_active_mask(inner)
    minority = majority.copy()
    minority[active_mask] = accent

    top = random.randint(0, cell_count - inner)
    left = random.randint(0, cell_count - inner)
    active_cells = {
        (top + int(r), left + int(c))
        for r, c in np.argwhere(active_mask)
    }
    given_cells = _choose_given_cells(cell_count, active_mask, top, left)
    if given_cells is None:
        return None

    if len(given_cells) >= len(active_cells):
        return None

    inp = _build_grid(cell_count, inner, bordered, bg, separator, majority, minority, given_cells)
    out = _build_grid(cell_count, inner, bordered, bg, separator, majority, minority, active_cells)

    if not _bg_is_dominant(inp, bg):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(solved, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fractal-cell input")
