from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
TEAL = 8

ACTIVE_COLORS = {GREEN, TEAL}
OBSTACLE_COLOR = RED

GEN_RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_monochrome_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_solid_square(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return h == w and int(component["area"]) == h * w


def _infer_block_layout(grid: np.ndarray) -> tuple[int, int, int, dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    anchor_comps = _find_monochrome_components(g, {GREEN, OBSTACLE_COLOR})
    if not anchor_comps:
        anchor_comps = _find_monochrome_components(g, ACTIVE_COLORS | {OBSTACLE_COLOR})
    assert anchor_comps, "expected at least one green or red square block"
    assert all(_is_solid_square(comp) for comp in anchor_comps), "green and red anchors must be solid squares"

    sizes = sorted({int(comp["bbox"][2] - comp["bbox"][0] + 1) for comp in anchor_comps})
    assert len(sizes) == 1, "all blocks must share one side length"
    size = sizes[0]

    row_offsets = {int(comp["bbox"][0]) % size for comp in anchor_comps}
    col_offsets = {int(comp["bbox"][1]) % size for comp in anchor_comps}
    assert len(row_offsets) == 1 and len(col_offsets) == 1, "blocks must align to one hidden block lattice"
    row_offset = next(iter(row_offsets))
    col_offset = next(iter(col_offsets))

    allowed = ACTIVE_COLORS | {OBSTACLE_COLOR}
    block_colors: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(np.isin(g, list(allowed))):
        rr = int(r)
        cc = int(c)
        assert rr >= row_offset and cc >= col_offset, "occupied pixels must lie on or below the lattice origin"
        br = (rr - row_offset) // size
        bc = (cc - col_offset) // size
        key = (int(br), int(bc))
        color = int(g[rr, cc])
        prev = block_colors.setdefault(key, color)
        assert prev == color, "each block cell must be monochromatic"

    assert block_colors, "expected at least one occupied block cell"
    h, w = g.shape
    for (br, bc), color in block_colors.items():
        r0 = row_offset + int(br) * size
        c0 = col_offset + int(bc) * size
        assert r0 + size <= h and c0 + size <= w, "input blocks must be fully visible"
        block = g[r0 : r0 + size, c0 : c0 + size]
        assert np.all(block == color), "every occupied lattice cell must be one full solid square block"

    return size, row_offset, col_offset, block_colors


def _coarse_state_from_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    size, row_offset, col_offset, block_colors = _infer_block_layout(g)
    max_br = max(br for br, _ in block_colors)
    max_bc = max(bc for _, bc in block_colors)
    active = np.zeros((max_br + 1, max_bc + 1), dtype=bool)
    red = np.zeros((max_br + 1, max_bc + 1), dtype=bool)
    for (br, bc), color in block_colors.items():
        if color in ACTIVE_COLORS:
            active[br, bc] = True
        else:
            assert color == OBSTACLE_COLOR
            red[br, bc] = True
    assert not np.any(active & red)
    assert np.any(active), "expected at least one active green block"
    return active, red, size, row_offset, col_offset


def _simulate_chain(active_seed: np.ndarray, red: np.ndarray) -> dict:
    active = np.asarray(active_seed, dtype=bool).copy()
    red = np.asarray(red, dtype=bool)
    rows, cols = active.shape
    steps: list[set[tuple[int, int]]] = []
    blocked_windows = 0
    partial_windows = 0
    trigger_windows = 0

    while True:
        new_cells: set[tuple[int, int]] = set()
        for r in range(rows - 1):
            for c in range(cols - 1):
                nw = bool(active[r, c])
                ne = bool(active[r, c + 1])
                sw = bool(active[r + 1, c])
                se = bool(active[r + 1, c + 1])

                targets: tuple[tuple[int, int], tuple[int, int]] | None = None
                if nw and se:
                    trigger_windows += 1
                    targets = ((r, c + 1), (r + 1, c))
                elif ne and sw:
                    trigger_windows += 1
                    targets = ((r, c), (r + 1, c + 1))
                else:
                    continue

                blocked_here = 0
                added_here = 0
                for rr, cc in targets:
                    if active[rr, cc]:
                        continue
                    if red[rr, cc]:
                        blocked_here += 1
                        continue
                    new_cells.add((rr, cc))
                    added_here += 1
                if blocked_here:
                    blocked_windows += 1
                if blocked_here and added_here:
                    partial_windows += 1

        if not new_cells:
            break
        for rr, cc in new_cells:
            active[rr, cc] = True
        steps.append(new_cells)

    return {
        "active": active,
        "steps": steps,
        "trigger_windows": int(trigger_windows),
        "blocked_windows": int(blocked_windows),
        "partial_windows": int(partial_windows),
    }


def _paint_block(out: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    h, w = out.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r0) + int(size))
    cc1 = min(w, int(c0) + int(size))
    if rr0 < rr1 and cc0 < cc1:
        out[rr0:rr1, cc0:cc1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Repeatedly fill the other diagonal of every active 2x2 block window with teal until no new square blocks can appear.

    Input structure:
    - The grid has a black background; every non-background object is a solid square block of one shared side length aligned to one hidden block lattice.
    - Green blocks are the initial active blocks, and red blocks are fixed obstacles on the same block lattice.
    - Blocks may be separated by background gaps, and only the block-grid coordinates matter for all matching.

    Rule:
    - Preserve all existing green and red blocks.
    - In block-grid coordinates, whenever one diagonal of a 2x2 window contains two active blocks (green or already-created teal), fill the other diagonal's empty positions with teal.
    - If one target position is occupied by a red block, skip that position but still fill the other target if it is empty; repeat this monotone growth until a fixpoint, clipping any painted block to the image boundary.

    Key insight:
    - This is a block-grid closure process rather than a one-pass scan: each newly created teal block becomes active immediately for later rounds, while red blocks never participate and can prune whole downstream branches.
    """
    g = np.asarray(grid, dtype=int)
    active_seed, red, size, row_offset, col_offset = _coarse_state_from_grid(g)
    result = _simulate_chain(active_seed, red)
    final_active = result["active"]

    out = g.copy()
    new_teal = final_active & ~active_seed
    for br, bc in np.argwhere(new_teal):
        _paint_block(out, row_offset + int(br) * size, col_offset + int(bc) * size, size, TEAL)
    return out


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_seed_clusters(rows: int, cols: int, want_multistep: bool) -> np.ndarray | None:
    active = np.zeros((rows, cols), dtype=bool)
    placed_boxes: list[tuple[int, int, int, int]] = []

    cluster_count = random.randint(1, 2)
    max_size = min(rows, cols, 5)
    if max_size < 2:
        return None

    size_choices = [2, 3, 4, 5]
    size_choices = [k for k in size_choices if k <= max_size]
    if not size_choices:
        return None

    chosen_sizes: list[int] = []
    if want_multistep:
        multi_sizes = [k for k in size_choices if k >= 3]
        if not multi_sizes:
            return None
        chosen_sizes.append(random.choice(multi_sizes))
        while len(chosen_sizes) < cluster_count:
            chosen_sizes.append(random.choice(size_choices))
    else:
        for _ in range(cluster_count):
            chosen_sizes.append(random.choice(size_choices))

    for idx, size in enumerate(chosen_sizes):
        orientations = [0, 1]
        random.shuffle(orientations)
        positions = [
            (r0, c0, orientation)
            for orientation in orientations
            for r0 in range(rows - size + 1)
            for c0 in range(cols - size + 1)
            if all(_bbox_separated((r0, c0, r0 + size - 1, c0 + size - 1), box, 1) for box in placed_boxes)
        ]
        random.shuffle(positions)
        if not positions:
            if idx == 0:
                return None
            break
        r0, c0, orientation = positions[0]
        placed_boxes.append((r0, c0, r0 + size - 1, c0 + size - 1))
        for i in range(size):
            rr = r0 + i
            cc = c0 + i if orientation == 0 else c0 + size - 1 - i
            active[rr, cc] = True

    return active if np.any(active) else None


def _candidate_red_cells(active: np.ndarray) -> list[tuple[int, int]]:
    full = _simulate_chain(active, np.zeros_like(active, dtype=bool))["active"]
    candidates = [(int(r), int(c)) for r, c in np.argwhere(full & ~active)]
    random.shuffle(candidates)
    return candidates


def _add_obstacles(active: np.ndarray, want_obstacles: bool) -> np.ndarray:
    red = np.zeros_like(active, dtype=bool)
    if not want_obstacles:
        return red

    candidates = _candidate_red_cells(active)
    if not candidates:
        return red

    target_count = random.randint(1, min(4, len(candidates)))
    for r, c in candidates:
        if int(red.sum()) >= target_count:
            break
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in np.argwhere(red)):
            continue
        red[r, c] = True
    return red


def _semantic_report(active: np.ndarray, red: np.ndarray) -> dict:
    no_red = _simulate_chain(active, np.zeros_like(red, dtype=bool))
    with_red = _simulate_chain(active, red)
    final_active = with_red["active"]
    new_active = final_active & ~active
    pruned_nonred = no_red["active"] & ~final_active & ~red
    return {
        "no_red": no_red,
        "with_red": with_red,
        "new_count": int(new_active.sum()),
        "step_count": len(with_red["steps"]),
        "pruned_nonred_count": int(pruned_nonred.sum()),
    }


def _render_input(
    rows: int,
    cols: int,
    size: int,
    row_offset: int,
    col_offset: int,
    tail_rows: int,
    tail_cols: int,
    active: np.ndarray,
    red: np.ndarray,
) -> np.ndarray:
    height = row_offset + rows * size + tail_rows
    width = col_offset + cols * size + tail_cols
    grid = np.full((height, width), BLACK, dtype=int)
    for br, bc in np.argwhere(active):
        _paint_block(grid, row_offset + int(br) * size, col_offset + int(bc) * size, size, GREEN)
    for br, bc in np.argwhere(red):
        _paint_block(grid, row_offset + int(br) * size, col_offset + int(bc) * size, size, RED)
    return grid


def _expected_output(
    inp: np.ndarray,
    size: int,
    row_offset: int,
    col_offset: int,
    active: np.ndarray,
    red: np.ndarray,
) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    final_active = _simulate_chain(active, red)["active"]
    for br, bc in np.argwhere(final_active & ~active):
        _paint_block(out, row_offset + int(br) * size, col_offset + int(bc) * size, size, TEAL)
    return out


def _valid_case(
    inp: np.ndarray,
    active: np.ndarray,
    red: np.ndarray,
    want_multistep: bool,
    want_obstacles: bool,
    want_prune: bool,
) -> bool:
    report = _semantic_report(active, red)
    if report["new_count"] < 2:
        return False
    if report["step_count"] < (2 if want_multistep else 1):
        return False
    if report["with_red"]["trigger_windows"] == 0:
        return False
    if want_obstacles:
        if int(red.sum()) == 0:
            return False
        if report["with_red"]["blocked_windows"] == 0:
            return False
        if report["with_red"]["partial_windows"] == 0:
            return False
    if want_prune and report["pruned_nonred_count"] == 0:
        return False

    size, row_offset, col_offset, _ = _infer_block_layout(inp)
    expected = _expected_output(inp, size, row_offset, col_offset, active, red)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if Counter(actual.ravel().tolist())[TEAL] == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(1, 4)
    row_offset = random.randint(0, size - 1)
    col_offset = random.randint(0, size - 1)
    tail_rows = random.randint(0, size - 1)
    tail_cols = random.randint(0, size - 1)

    max_rows = (30 - row_offset - tail_rows) // size
    max_cols = (30 - col_offset - tail_cols) // size
    if max_rows < 5 or max_cols < 5:
        return None

    rows = random.randint(5, min(8, max_rows))
    cols = random.randint(5, min(8, max_cols))
    want_multistep = random.random() < 0.65
    want_obstacles = random.random() < 0.7
    want_prune = want_obstacles and random.random() < 0.65

    active = _place_seed_clusters(rows, cols, want_multistep)
    if active is None:
        return None
    red = _add_obstacles(active, want_obstacles)
    inp = _render_input(rows, cols, size, row_offset, col_offset, tail_rows, tail_cols, active, red)
    if not _valid_case(inp, active, red, want_multistep, want_obstacles, want_prune):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-chain input")
