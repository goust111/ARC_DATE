from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR_VECS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

ARROW_MASKS = {
    "U": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "D": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "L": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "R": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}

TIP_OFFSETS = {
    "U": (0, 1),
    "D": (2, 1),
    "L": (1, 0),
    "R": (1, 2),
}

NEXT_DIR_OPTIONS = {
    "U": ("L", "U", "R"),
    "D": ("L", "D", "R"),
    "L": ("U", "L", "D"),
    "R": ("U", "R", "D"),
}

# --------------- solver ---------------


def _background_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    winners = vals[counts == counts.max()]
    assert winners.size == 1, "background color must be uniquely most common"
    return int(winners[0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _box_cells(top_left: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r0, c0 = top_left
    return frozenset((r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3))


def _make_arrow(top_left: tuple[int, int], direction: str, color: int) -> dict[str, object]:
    r0, c0 = top_left
    tip_dr, tip_dc = TIP_OFFSETS[direction]
    shape_cells = frozenset((r0 + dr, c0 + dc) for dr, dc in ARROW_MASKS[direction])
    return {
        "top_left": top_left,
        "direction": direction,
        "color": int(color),
        "box_cells": _box_cells(top_left),
        "shape_cells": shape_cells,
        "tip": (r0 + tip_dr, c0 + tip_dc),
    }


def _detect_arrows(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid must be at least 3x3"

    arrows: list[dict[str, object]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            window = g[r0 : r0 + 3, c0 : c0 + 3]
            colors = np.unique(window)
            non_bg = [int(v) for v in colors.tolist() if int(v) != bg]
            if len(non_bg) != 1:
                continue
            color = non_bg[0]
            rel_cells = frozenset(
                (int(r), int(c))
                for r, c in np.argwhere(window == color).tolist()
            )
            matches = [d for d, mask in ARROW_MASKS.items() if rel_cells == mask]
            if len(matches) != 1:
                continue
            arrows.append(_make_arrow((r0, c0), matches[0], color))

    assert arrows, "expected at least one 3x3 arrow"

    non_bg_cells = frozenset(
        (int(r), int(c)) for r, c in np.argwhere(g != bg).tolist()
    )
    covered = frozenset().union(*(arrow["shape_cells"] for arrow in arrows))
    assert covered == non_bg_cells, "all non-background pixels must belong to arrows"

    total_shape_area = sum(len(arrow["shape_cells"]) for arrow in arrows)
    assert total_shape_area == len(covered), "arrow shapes must not overlap"

    total_box_area = sum(len(arrow["box_cells"]) for arrow in arrows)
    merged_boxes = frozenset().union(*(arrow["box_cells"] for arrow in arrows))
    assert total_box_area == len(merged_boxes), "arrow 3x3 footprints must be disjoint"

    return arrows


def _trace_rays(
    arrows: list[dict[str, object]], shape: tuple[int, int]
) -> list[dict[str, object]]:
    area_to_arrow: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["box_cells"]:
            area_to_arrow[cell] = idx

    traces: list[dict[str, object]] = []
    for idx, arrow in enumerate(arrows):
        dr, dc = DIR_VECS[arrow["direction"]]
        r, c = arrow["tip"]
        r += dr
        c += dc
        cells: list[tuple[int, int]] = []
        hit = None
        while _in_bounds(shape, r, c):
            owner = area_to_arrow.get((r, c))
            if owner is not None:
                hit = int(owner)
                assert hit != idx, "a ray must not re-enter its own 3x3 footprint"
                break
            cells.append((r, c))
            r += dr
            c += dc
        traces.append({"cells": tuple(cells), "hit": hit})
    return traces


def _activated_output(
    g: np.ndarray, arrows: list[dict[str, object]], traces: list[dict[str, object]], bg: int
) -> np.ndarray:
    indegree = [0] * len(arrows)
    for trace in traces:
        if trace["hit"] is not None:
            indegree[int(trace["hit"])] += 1

    assert max(indegree) <= 1, "each arrow may be activated by at most one upstream ray"
    roots = [idx for idx, deg in enumerate(indegree) if deg == 0]
    assert roots, "expected at least one source arrow"

    activated: dict[int, int] = {}
    queue: deque[int] = deque()
    for idx in roots:
        activated[idx] = int(arrows[idx]["color"])
        queue.append(idx)

    painted: dict[tuple[int, int], int] = {}
    while queue:
        idx = queue.popleft()
        color = activated[idx]
        trace = traces[idx]
        for cell in trace["cells"]:
            assert int(g[cell]) == bg, "rays may only pass through background cells"
            prev = painted.get(cell)
            assert prev is None or prev == color, "different chains cannot paint the same cell"
            painted[cell] = color

        hit = trace["hit"]
        if hit is None:
            continue
        if hit not in activated:
            activated[hit] = color
            queue.append(hit)
        else:
            assert activated[hit] == color, "an activated arrow must not receive a new color"

    assert len(activated) == len(arrows), "arrow graph must be acyclic and source-reachable"

    out = g.copy()
    for cell, color in painted.items():
        out[cell] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays through chains of 3x3 arrows, propagating the source color downstream.

    Input structure:
    - The background is a single dominant color.
    - Every foreground object is a monochrome 3x3 arrow in one of four cardinal directions.
    - Arrow footprints do not overlap.

    Rule:
    - A ray entering an arrow's footprint activates that arrow if it has no upstream activator yet.
    - The activated arrow emits from its own tip, keeps the inherited upstream color, and continues the cascade.
    - Rays color only background cells; arrow pixels keep their original colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    arrows = _detect_arrows(g, bg)
    traces = _trace_rays(arrows, g.shape)
    return _activated_output(g, arrows, traces, bg)


# --------------- generator ---------------


GEN_BG = 0
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
TOTAL_ARROWS_RANGE = (2, 8)
MAX_CHAINS = 4

MIN_LINK_PAINT = 1
MIN_EDGE_PAINT = 2
GEN_ATTEMPTS = 400
CHAIN_ATTEMPTS = 120


def _all_placements(h: int, w: int) -> list[dict[str, object]]:
    placements = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            for direction in ("U", "D", "L", "R"):
                placements.append(_make_arrow((r0, c0), direction, 1))
    return placements


def _box_clear(
    arrow: dict[str, object],
    occupied_boxes: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
) -> bool:
    cells = set(arrow["box_cells"])
    return not (cells & occupied_boxes) and not (cells & occupied_paths)


def _entry_top_lefts(
    entry: tuple[int, int], direction: str, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = entry
    h, w = shape
    out = []
    if direction == "R":
        for r0 in (r - 2, r - 1, r):
            if 0 <= r0 <= h - 3 and 0 <= c <= w - 3:
                out.append((r0, c))
    elif direction == "L":
        for r0 in (r - 2, r - 1, r):
            c0 = c - 2
            if 0 <= r0 <= h - 3 and 0 <= c0 <= w - 3:
                out.append((r0, c0))
    elif direction == "D":
        for c0 in (c - 2, c - 1, c):
            if 0 <= r <= h - 3 and 0 <= c0 <= w - 3:
                out.append((r, c0))
    else:
        for c0 in (c - 2, c - 1, c):
            r0 = r - 2
            if 0 <= r0 <= h - 3 and 0 <= c0 <= w - 3:
                out.append((r0, c0))
    return out


def _path_to_boundary(
    arrow: dict[str, object],
    shape: tuple[int, int],
    occupied_boxes: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    dr, dc = DIR_VECS[arrow["direction"]]
    r, c = arrow["tip"]
    r += dr
    c += dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        if (r, c) in occupied_boxes or (r, c) in occupied_paths:
            return None
        cells.append((r, c))
        r += dr
        c += dc
    return tuple(cells) if len(cells) >= MIN_EDGE_PAINT else None


def _next_candidates(
    current: dict[str, object],
    shape: tuple[int, int],
    occupied_boxes: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
) -> list[tuple[dict[str, object], tuple[tuple[int, int], ...]]]:
    dr, dc = DIR_VECS[current["direction"]]
    r, c = current["tip"]
    r += dr
    c += dc
    prefix: list[tuple[int, int]] = []
    candidates: list[tuple[dict[str, object], tuple[tuple[int, int], ...]]] = []

    while _in_bounds(shape, r, c):
        if (r, c) in occupied_boxes or (r, c) in occupied_paths:
            break

        if len(prefix) >= MIN_LINK_PAINT:
            for top_left in _entry_top_lefts((r, c), current["direction"], shape):
                for next_dir in NEXT_DIR_OPTIONS[current["direction"]]:
                    arrow = _make_arrow(top_left, next_dir, 1)
                    if not _box_clear(arrow, occupied_boxes, occupied_paths):
                        continue
                    if set(arrow["box_cells"]) & set(prefix):
                        continue
                    candidates.append((arrow, tuple(prefix)))

        prefix.append((r, c))
        r += dr
        c += dc

    random.shuffle(candidates)
    return candidates


def _extend_chain(
    arrows: list[dict[str, object]],
    segments: list[tuple[tuple[int, int], ...]],
    target_len: int,
    shape: tuple[int, int],
    occupied_boxes: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
) -> tuple[list[dict[str, object]], list[tuple[tuple[int, int], ...]]] | None:
    current = arrows[-1]
    if len(arrows) == target_len:
        tail = _path_to_boundary(current, shape, occupied_boxes, occupied_paths)
        if tail is None:
            return None
        return arrows, segments + [tail]

    for nxt, segment in _next_candidates(current, shape, occupied_boxes, occupied_paths):
        new_boxes = occupied_boxes | set(nxt["box_cells"])
        new_paths = occupied_paths | set(segment)
        result = _extend_chain(
            arrows + [nxt],
            segments + [segment],
            target_len,
            shape,
            new_boxes,
            new_paths,
        )
        if result is not None:
            return result
    return None


def _build_chain(
    chain_len: int,
    shape: tuple[int, int],
    occupied_boxes: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
) -> tuple[list[dict[str, object]], list[tuple[tuple[int, int], ...]]] | None:
    h, w = shape
    roots = _all_placements(h, w)
    random.shuffle(roots)
    for root in roots[:CHAIN_ATTEMPTS]:
        if not _box_clear(root, occupied_boxes, occupied_paths):
            continue
        result = _extend_chain(
            [root],
            [],
            chain_len,
            shape,
            occupied_boxes | set(root["box_cells"]),
            occupied_paths,
        )
        if result is not None:
            return result
    return None


def _sample_chain_lengths(total_arrows: int) -> list[int]:
    max_chains = min(MAX_CHAINS, total_arrows - 1)
    chain_count = random.randint(1, max_chains)
    lengths = [1] * chain_count
    lengths[random.randrange(chain_count)] += 1
    remaining = total_arrows - sum(lengths)
    while remaining > 0:
        lengths[random.randrange(chain_count)] += 1
        remaining -= 1
    random.shuffle(lengths)
    return lengths


def _render_grid(shape: tuple[int, int], chains: list[list[dict[str, object]]]) -> np.ndarray:
    grid = np.full(shape, GEN_BG, dtype=int)
    for chain in chains:
        root_color = random.choice(GEN_COLORS)
        for idx, arrow in enumerate(chain):
            color = root_color if idx == 0 else random.choice(GEN_COLORS)
            for cell in arrow["shape_cells"]:
                grid[cell] = color
    return grid


def _chain_lengths_from_graph(traces: list[dict[str, object]]) -> list[int]:
    indegree = [0] * len(traces)
    for trace in traces:
        if trace["hit"] is not None:
            indegree[int(trace["hit"])] += 1
    roots = [idx for idx, deg in enumerate(indegree) if deg == 0]
    lengths = []
    for root in roots:
        length = 0
        cur = root
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            length += 1
            cur = traces[cur]["hit"]
        lengths.append(length)
    return lengths


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if int(np.argmax(np.bincount(grid.ravel(), minlength=10))) != GEN_BG:
        return False
    try:
        bg = _background_color(grid)
        arrows = _detect_arrows(grid, bg)
        traces = _trace_rays(arrows, grid.shape)
        out = _activated_output(grid, arrows, traces, bg)
    except Exception:
        return False

    if bg != GEN_BG or len(arrows) < 2:
        return False
    if np.array_equal(out, grid):
        return False
    lengths = _chain_lengths_from_graph(traces)
    return bool(lengths and max(lengths) >= 2)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    total_arrows = random.randint(*TOTAL_ARROWS_RANGE)
    chain_lengths = _sample_chain_lengths(total_arrows)

    occupied_boxes: set[tuple[int, int]] = set()
    occupied_paths: set[tuple[int, int]] = set()
    chains: list[list[dict[str, object]]] = []

    for chain_len in chain_lengths:
        built = _build_chain(chain_len, (h, w), occupied_boxes, occupied_paths)
        if built is None:
            return None
        arrows, segments = built
        chains.append(arrows)
        for arrow in arrows:
            occupied_boxes |= set(arrow["box_cells"])
        for segment in segments:
            occupied_paths |= set(segment)

    grid = _render_grid((h, w), chains)
    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-cascade input")
