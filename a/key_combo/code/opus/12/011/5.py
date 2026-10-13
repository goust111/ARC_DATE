import random
from collections import deque

import numpy as np


GREY = 5
BLACK = 0
TEAL = 8
GRID_SHAPE = (10, 10)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
GEN_ATTEMPTS = 1000
_GENERATED_BLACK_COUNTS: set[int] = set()


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIRS8:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _is_closed_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    grey = crop == GREY
    h, w = grey.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if grey[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grey[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grey[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any(grey & ~outside))


def _maximal_diagonal_runs(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    runs = []
    for dr, dc in ((1, 1), (1, -1)):
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue

                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((int(rr), int(cc)))
                    rr += dr
                    cc += dc

                if len(cells) >= 2:
                    diag_id = r - c if dc == 1 else r + c
                    runs.append(
                        {
                            "orientation": (dr, dc),
                            "cells": cells,
                            "diag_id": int(diag_id),
                            "length": len(cells),
                        }
                    )
    return runs


def _choose_longest_run(runs: list[dict]) -> dict:
    assert runs, "expected at least one diagonal run"
    max_len = max(run["length"] for run in runs)
    candidates = [run for run in runs if run["length"] == max_len]

    groups = []
    for orientation in {(run["orientation"]) for run in candidates}:
        oriented = sorted(
            [run for run in candidates if run["orientation"] == orientation],
            key=lambda run: run["diag_id"],
        )
        group = [oriented[0]]
        for run in oriented[1:]:
            if run["diag_id"] == group[-1]["diag_id"] + 1:
                group.append(run)
            else:
                groups.append(group)
                group = [run]
        groups.append(group)

    assert len(groups) == 1, "max-length runs must form a single candidate band"
    group = groups[0]
    assert len(group) % 2 == 1, "parallel candidate band must have odd width"
    return group[len(group) // 2]


def _analyze_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for coords in _components8(grid == BLACK):
        mask = _component_mask(grid.shape, coords)
        runs = _maximal_diagonal_runs(mask)
        comps.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(len(coords)),
                "closed": _is_closed_component(grid, coords),
                "runs": runs,
                "max_run": max((run["length"] for run in runs), default=0),
            }
        )
    return comps


def _selected_run(grid: np.ndarray, *, open_only: bool) -> dict:
    comps = _analyze_components(grid)
    runs = []
    for comp in comps:
        if open_only and comp["closed"]:
            continue
        runs.extend(comp["runs"])
    return _choose_longest_run(runs)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the winning open-component diagonal stroke after filtering out closed black loops.

    Input structure:
    - A 10x10 grid uses grey as background and black for 8-connected foreground components.
    - There are several separated black components; some enclose grey pixels inside their own bounding boxes and some do not.
    - Candidate straight strokes appear as maximal continuous black segments along the two diagonal directions.

    Rule:
    - Split the black pixels into 8-connected components.
    - For each component, crop to its bounding box and classify it as closed if some grey cell in that box is not 4-connected to the box border through grey cells.
    - Discard every closed component, then collect all maximal black diagonal segments from the remaining open components.
    - Select the longest such segment; if the longest candidates form one odd-width band of adjacent parallel diagonals, choose the middle segment and recolor it teal.

    Key insight:
    - The longest visible diagonal is not always correct globally: a longer stroke can live inside a topologically closed component and must be ignored before comparing diagonal runs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE
    assert set(np.unique(g).tolist()) <= {BLACK, GREY}

    chosen = _selected_run(g, open_only=True)
    out = g.copy()
    for r, c in chosen["cells"]:
        out[r, c] = TEAL
    return out


def _patch(*rows: str) -> np.ndarray:
    arr = np.full((len(rows), len(rows[0])), GREY, dtype=int)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "1":
                arr[r, c] = BLACK
    return arr


CLOSED_TEMPLATES = [
    _patch(
        "11111",
        "11001",
        "10101",
        "10011",
        "11111",
    ),
]

OPEN_WINNER_LINE = [
    _patch(
        "1100",
        "0110",
        "0011",
        "0011",
    ),
    _patch(
        "1100",
        "1110",
        "0111",
        "0011",
    ),
]

OPEN_WINNER_BAND = [
    _patch(
        "111000",
        "011100",
        "001110",
        "000111",
    ),
]

OPEN_EXTRA = [
    _patch(
        "11",
        "11",
    ),
    _patch(
        "110",
        "010",
        "011",
    ),
    _patch(
        "1110",
        "0100",
        "0110",
    ),
    _patch(
        "1110",
        "0110",
        "0010",
    ),
    _patch(
        "111",
        "111",
        "110",
    ),
    _patch(
        "11100",
        "01110",
        "00111",
    ),
    _patch(
        "0100",
        "1110",
        "0111",
        "0010",
    ),
    _patch(
        "1110",
        "0010",
        "0011",
        "0011",
    ),
]


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _chebyshev_block(mask: np.ndarray, coords: np.ndarray) -> None:
    h, w = mask.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    indexed = sorted(
        enumerate(patches),
        key=lambda item: int(np.count_nonzero(item[1] == BLACK)),
        reverse=True,
    )

    def rec(i: int, grid: np.ndarray, blocked: np.ndarray) -> np.ndarray | None:
        if i == len(indexed):
            return grid

        _, patch = indexed[i]
        coords = np.argwhere(patch == BLACK)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        anchors = [(r0, c0) for r0 in range(11 - h) for c0 in range(11 - w)]
        random.shuffle(anchors)

        for r0, c0 in anchors:
            shifted = coords + np.array([r0, c0], dtype=int)
            if blocked[shifted[:, 0], shifted[:, 1]].any():
                continue
            next_grid = grid.copy()
            next_grid[shifted[:, 0], shifted[:, 1]] = BLACK
            next_blocked = blocked.copy()
            _chebyshev_block(next_blocked, shifted)
            out = rec(i + 1, next_grid, next_blocked)
            if out is not None:
                return out
        return None

    start_grid = np.full(GRID_SHAPE, GREY, dtype=int)
    start_blocked = np.zeros(GRID_SHAPE, dtype=bool)
    return rec(0, start_grid, start_blocked)


def _sample_extra_patch(max_run: int) -> np.ndarray:
    for _ in range(50):
        patch = _transform_patch(random.choice(OPEN_EXTRA))
        comp = _analyze_components(patch)[0]
        if not comp["closed"] and 2 <= comp["max_run"] <= max_run:
            return patch
    raise RuntimeError("failed to sample extra open patch")


def _validate_grid(grid: np.ndarray) -> bool:
    comps = _analyze_components(grid)
    if not 3 <= len(comps) <= 6:
        return False
    if any(not (4 <= comp["area"] <= 25) for comp in comps):
        return False

    closed = [comp for comp in comps if comp["closed"]]
    open_comps = [comp for comp in comps if not comp["closed"]]
    if len(closed) < 1 or len(open_comps) < 2:
        return False

    if max(comp["max_run"] for comp in closed) <= max(comp["max_run"] for comp in open_comps):
        return False

    try:
        chosen_open = _selected_run(grid, open_only=True)
        chosen_all = _selected_run(grid, open_only=False)
    except AssertionError:
        return False

    if set(chosen_open["cells"]) == set(chosen_all["cells"]):
        return False

    out = main(grid)
    if int(np.count_nonzero(out == TEAL)) != chosen_open["length"]:
        return False

    recolored = set(map(tuple, np.argwhere(out == TEAL)))
    if recolored != set(chosen_open["cells"]):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n_total = random.choices([3, 4, 5, 6], weights=[5, 5, 3, 2], k=1)[0]

    closed_patch = _transform_patch(random.choice(CLOSED_TEMPLATES))

    band_prob = 0.7 if n_total == 3 else 0.35
    winner_pool = OPEN_WINNER_BAND if random.random() < band_prob else OPEN_WINNER_LINE
    winner_patch = _transform_patch(random.choice(winner_pool))

    winner_meta = _analyze_components(winner_patch)[0]
    if winner_meta["closed"] or winner_meta["max_run"] != 4:
        return None

    patches = [closed_patch, winner_patch]
    for _ in range(n_total - 2):
        patches.append(_sample_extra_patch(max_run=3))

    grid = _place_patches(patches)
    if grid is None:
        return None
    return grid if _validate_grid(grid) else None


def generate_input() -> np.ndarray:
    global _GENERATED_BLACK_COUNTS
    fallback = None
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is None:
            continue
        if len(_GENERATED_BLACK_COUNTS) >= 20:
            return grid
        if fallback is None:
            fallback = grid
        black_count = int(np.count_nonzero(grid == BLACK))
        if black_count not in _GENERATED_BLACK_COUNTS:
            _GENERATED_BLACK_COUNTS.add(black_count)
            return grid
    if fallback is not None:
        return fallback
    raise RuntimeError("generate_input: failed to build a valid puzzle")
