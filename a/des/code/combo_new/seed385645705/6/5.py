from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CHAIN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_SIZE_RANGE = (18, 26)
CHAIN_COUNT_RANGE = (2, 4)
CHAIN_LENGTH_RANGE = (4, 8)
CHAIN_MARGIN = 2
CHAIN_CLEARANCE = 3
MAX_SHIFT = 3
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_bbox(
    bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _gear_touches_chain(
    gear_cells: set[tuple[int, int]], chain_cells: list[tuple[int, int]]
) -> bool:
    chain_set = set(chain_cells)
    for r, c in gear_cells:
        for nr, nc in _neighbors4(r, c, (10**9, 10**9)):
            if (nr, nc) in chain_set:
                return True
    return False


def _detect_gears(grid: np.ndarray) -> list[dict[str, object]]:
    gears: list[dict[str, object]] = []
    for comp in _component_cells(grid == GREY):
        assert len(comp) == 4, "grey components must be 2x2 gears"
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0, c1 - c0) == (1, 1), "gear bbox must be 2x2"
        cells = {(r, c) for r, c in comp}
        assert cells == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "gear must fill its 2x2 bbox"
        gears.append({"cells": cells, "bbox": (r0, c0, r1, c1)})
    return gears


def _detect_chains(grid: np.ndarray) -> list[dict[str, object]]:
    mask = (grid != BG) & (grid != GREY)
    comps = _component_cells(mask)
    assert comps, "expected at least one colored chain"

    chains: list[dict[str, object]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        area = len(comp)
        assert 4 <= area <= 8, "chain length out of range"
        if r0 == r1:
            cols = sorted(c for _, c in comp)
            assert cols == list(range(c0, c1 + 1)), "horizontal chain must be contiguous"
            cells = [(r0, c) for c in cols]
            orientation = "h"
        else:
            assert c0 == c1, "chain must be horizontal or vertical"
            rows = sorted(r for r, _ in comp)
            assert rows == list(range(r0, r1 + 1)), "vertical chain must be contiguous"
            cells = [(r, c0) for r in rows]
            orientation = "v"
        chains.append(
            {
                "cells": cells,
                "bbox": (r0, c0, r1, c1),
                "length": area,
                "orientation": orientation,
            }
        )
    return chains


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each colored line-chain by the number of orthogonally adjacent grey gears.

    Input structure:
    - The background is black.
    - Foreground chains are straight contiguous line segments of non-grey colors, each either horizontal or vertical.
    - Gears are grey 2x2 filled blocks that do not overlap chains, though they may sit orthogonally adjacent to them.

    Rule:
    - Identify every gear and every colored chain.
    - For each chain, count how many distinct grey 2x2 gears have at least one pixel 4-neighbor-adjacent to that chain.
    - Cyclically shift the chain's color sequence by that count: horizontal chains shift right, vertical chains shift down.
    - Keep the gears and the background unchanged.

    Key insight:
    - The grey squares are not part of the chains themselves; they are side markers whose adjacency count determines how far each conveyor-like chain rotates its colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    gears = _detect_gears(g)
    chains = _detect_chains(g)

    out = g.copy()
    for chain in chains:
        cells = chain["cells"]
        shift = sum(1 for gear in gears if _gear_touches_chain(gear["cells"], cells)) % int(chain["length"])
        colors = [int(g[r, c]) for r, c in cells]
        if shift:
            colors = colors[-shift:] + colors[:-shift]
        for (r, c), color in zip(cells, colors):
            out[r, c] = color

    return out


def _make_chain_spec(
    shape: tuple[int, int], orientation: str, length: int
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        if orientation == "h":
            r = random.randint(CHAIN_MARGIN, h - 1 - CHAIN_MARGIN)
            c0 = random.randint(CHAIN_MARGIN, w - length - CHAIN_MARGIN)
            cells = [(r, c0 + i) for i in range(length)]
            bbox = (r, c0, r, c0 + length - 1)
        else:
            r0 = random.randint(CHAIN_MARGIN, h - length - CHAIN_MARGIN)
            c = random.randint(CHAIN_MARGIN, w - 1 - CHAIN_MARGIN)
            cells = [(r0 + i, c) for i in range(length)]
            bbox = (r0, c, r0 + length - 1, c)

        yield {
            "orientation": orientation,
            "length": length,
            "cells": cells,
            "bbox": bbox,
        }


def _place_chains(shape: tuple[int, int], count: int) -> list[dict[str, object]] | None:
    orientations = ["h", "v"] + [random.choice(["h", "v"]) for _ in range(count - 2)]
    random.shuffle(orientations)

    chains: list[dict[str, object]] = []
    expanded_boxes: list[tuple[int, int, int, int]] = []

    for orientation in orientations:
        length = random.randint(*CHAIN_LENGTH_RANGE)
        placed = None
        for candidate in _make_chain_spec(shape, orientation, length):
            bbox = _expanded_bbox(candidate["bbox"], CHAIN_CLEARANCE, shape)
            if any(_boxes_overlap(bbox, other) for other in expanded_boxes):
                continue
            placed = candidate
            expanded_boxes.append(bbox)
            chains.append(candidate)
            break
        if placed is None:
            return None
    return chains


def _touches_any_other_chain(
    gear_cells: set[tuple[int, int]], chain_index: int, chains: list[dict[str, object]]
) -> bool:
    for other_idx, other in enumerate(chains):
        if other_idx == chain_index:
            continue
        if _gear_touches_chain(gear_cells, other["cells"]):
            return True
    return False


def _candidate_gears(chain_index: int, chains: list[dict[str, object]], shape: tuple[int, int]):
    h, w = shape
    chain = chains[chain_index]
    r0, c0, r1, c1 = chain["bbox"]
    all_chain_cells = {cell for item in chains for cell in item["cells"]}
    candidates: list[dict[str, object]] = []

    row_start = max(0, r0 - 2)
    row_stop = min(h - 2, r1 + 1)
    col_start = max(0, c0 - 2)
    col_stop = min(w - 2, c1 + 1)

    for gr in range(row_start, row_stop + 1):
        for gc in range(col_start, col_stop + 1):
            cells = {(gr, gc), (gr + 1, gc), (gr, gc + 1), (gr + 1, gc + 1)}
            if cells & all_chain_cells:
                continue
            if not _gear_touches_chain(cells, chain["cells"]):
                continue
            if _touches_any_other_chain(cells, chain_index, chains):
                continue
            candidates.append({"top_left": (gr, gc), "cells": cells})

    unique: dict[tuple[int, int], dict[str, object]] = {}
    for cand in candidates:
        unique[cand["top_left"]] = cand
    return list(unique.values())


def _gear_conflict(a: dict[str, object], b: dict[str, object]) -> bool:
    for r1, c1 in a["cells"]:
        for r2, c2 in b["cells"]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _pick_non_touching_subset(
    candidates: list[dict[str, object]], count: int
) -> list[dict[str, object]] | None:
    if count == 0:
        return []

    order = candidates[:]
    random.shuffle(order)
    chosen: list[dict[str, object]] = []

    def dfs(start: int) -> bool:
        need = count - len(chosen)
        if need == 0:
            return True
        if len(order) - start < need:
            return False
        for idx in range(start, len(order)):
            cand = order[idx]
            if any(_gear_conflict(cand, prev) for prev in chosen):
                continue
            chosen.append(cand)
            if dfs(idx + 1):
                return True
            chosen.pop()
        return False

    if dfs(0):
        return chosen[:]
    return None


def _selection_options(candidates: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    options: dict[int, list[dict[str, object]]] = {0: []}
    for count in range(1, min(MAX_SHIFT, len(candidates)) + 1):
        subset = _pick_non_touching_subset(candidates, count)
        if subset is not None:
            options[count] = subset
    return options


def _choose_targets(options_per_chain: list[dict[int, list[dict[str, object]]]]) -> list[int]:
    targets = [random.choice(sorted(options)) for options in options_per_chain]

    positive_idxs = [i for i, options in enumerate(options_per_chain) if any(k > 0 for k in options)]
    if not any(target > 0 for target in targets):
        idx = random.choice(positive_idxs)
        targets[idx] = random.choice([k for k in options_per_chain[idx] if k > 0])

    if len(targets) >= 2 and random.random() < 0.7:
        zeroable = [i for i, options in enumerate(options_per_chain) if 0 in options]
        if zeroable and not any(target == 0 for target in targets):
            idx = random.choice([i for i in zeroable if targets[i] > 0] or zeroable)
            targets[idx] = 0

    multi_shift = [i for i, options in enumerate(options_per_chain) if any(k >= 2 for k in options)]
    if multi_shift and random.random() < 0.55:
        idx = random.choice(multi_shift)
        targets[idx] = random.choice([k for k in options_per_chain[idx] if k >= 2])

    return targets


def _paint_grid(
    shape: tuple[int, int], chains: list[dict[str, object]], gear_groups: list[list[dict[str, object]]]
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)

    for chain in chains:
        colors = random.sample(CHAIN_COLORS, int(chain["length"]))
        chain["colors"] = colors
        for (r, c), color in zip(chain["cells"], colors):
            grid[r, c] = color

    for group in gear_groups:
        for gear in group:
            for r, c in gear["cells"]:
                grid[r, c] = GREY

    return grid


def _expected_output(
    grid: np.ndarray, chains: list[dict[str, object]], targets: list[int]
) -> np.ndarray:
    out = grid.copy()
    for chain, shift in zip(chains, targets):
        colors = list(chain["colors"])
        shift %= int(chain["length"])
        if shift:
            colors = colors[-shift:] + colors[:-shift]
        for (r, c), color in zip(chain["cells"], colors):
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIZE_RANGE),
        random.randint(*GRID_SIZE_RANGE),
    )
    chain_count = random.randint(*CHAIN_COUNT_RANGE)

    chains = _place_chains(shape, chain_count)
    if chains is None:
        return None

    options_per_chain: list[dict[int, list[dict[str, object]]]] = []
    for idx in range(chain_count):
        options = _selection_options(_candidate_gears(idx, chains, shape))
        if max(options) == 0:
            return None
        options_per_chain.append(options)

    targets = _choose_targets(options_per_chain)
    if not any(target > 0 for target in targets):
        return None

    gear_groups = [options_per_chain[i][targets[i]] for i in range(chain_count)]
    grid = _paint_grid(shape, chains, gear_groups)
    expected = _expected_output(grid, chains, targets)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
