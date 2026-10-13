from __future__ import annotations

import heapq
import random

import numpy as np


DIR_TO_VEC = {
    "R": (0, 1),
    "L": (0, -1),
    "U": (-1, 0),
    "D": (1, 0),
}
OPPOSITE = {"R": "L", "L": "R", "U": "D", "D": "U"}

PATTERN_TO_DIR = {
    frozenset({(0, 0), (1, 0), (2, 0), (1, 1)}): ("R", (1, 1)),
    frozenset({(1, 0), (0, 1), (1, 1), (2, 1)}): ("L", (1, 0)),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}): ("D", (1, 1)),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}): ("U", (0, 1)),
}

TIP_SHAPES = {
    "R": ((0, 0), (-1, -1), (0, -1), (1, -1)),
    "L": ((0, 0), (-1, 1), (0, 1), (1, 1)),
    "D": ((0, 0), (-1, -1), (-1, 0), (-1, 1)),
    "U": ((0, 0), (1, -1), (1, 0), (1, 1)),
}

DIR_ORDER = ("R", "L", "U", "D")
COLOR_POOL = tuple(range(1, 10))

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
PALETTE_SIZE_RANGE = (3, 6)
CHAIN_GAP_RANGE = (4, 8)
CHAIN_RETRIES = 160
ISOLATED_RETRIES = 180
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _arrow_cells_from_tip(tip: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    tr, tc = tip
    return [(tr + dr, tc + dc) for dr, dc in TIP_SHAPES[direction]]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _make_arrow(tip: tuple[int, int], direction: str, color: int) -> dict:
    cells = _arrow_cells_from_tip(tip, direction)
    bbox = _bbox(cells)
    return {
        "tip": (int(tip[0]), int(tip[1])),
        "dir": direction,
        "color": int(color),
        "cells": tuple((int(r), int(c)) for r, c in cells),
        "bbox": tuple(int(v) for v in bbox),
        "sort_key": (int(bbox[0]), int(bbox[1]), int(tip[0]), int(tip[1])),
    }


def _sort_and_index_arrows(arrows: list[dict]) -> list[dict]:
    ordered = sorted(arrows, key=lambda arrow: arrow["sort_key"])
    out = []
    for idx, arrow in enumerate(ordered):
        item = dict(arrow)
        item["id"] = idx
        out.append(item)
    return out


def _in_bounds(shape: tuple[int, int], cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _build_owner_map(arrows: list[dict]) -> dict[tuple[int, int], int]:
    owner: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            assert cell not in owner, "arrows must not overlap"
            owner[cell] = idx
    return owner


def _trace_index(arrows: list[dict], idx: int, shape: tuple[int, int]) -> tuple[list[tuple[int, int]], int | None, int | None]:
    owner = _build_owner_map(arrows)
    arrow = arrows[idx]
    dr, dc = DIR_TO_VEC[arrow["dir"]]
    r, c = arrow["tip"]
    path: list[tuple[int, int]] = []
    step = 0

    while True:
        step += 1
        r += dr
        c += dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            return path, None, None
        hit = owner.get((r, c))
        if hit is not None and hit != idx:
            return path, int(hit), step
        path.append((r, c))


def _collect_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _parse_arrows(g: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _collect_components(g):
        coords = comp["coords"]
        assert len(coords) == 4, "each arrow must be a monochromatic 4-pixel component"
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        rel = frozenset((r - r0, c - c0) for r, c in coords)
        assert rel in PATTERN_TO_DIR, "every component must be one of the four arrow glyphs"
        direction, tip_offset = PATTERN_TO_DIR[rel]
        tip = (r0 + tip_offset[0], c0 + tip_offset[1])
        arrow = {
            "tip": (int(tip[0]), int(tip[1])),
            "dir": direction,
            "color": int(comp["color"]),
            "cells": tuple(sorted((int(r), int(c)) for r, c in coords)),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "sort_key": (int(r0), int(c0), int(tip[0]), int(tip[1])),
        }
        arrows.append(arrow)

    assert arrows, "expected at least one arrow"
    return _sort_and_index_arrows(arrows)


def _geometry(arrows: list[dict], shape: tuple[int, int]) -> tuple[list[dict], bool]:
    raw_traces: list[tuple[list[tuple[int, int]], int | None, int | None]] = []
    overlap = False
    path_sets: list[set[tuple[int, int]]] = []
    for idx in range(len(arrows)):
        path, hit_target, hit_steps = _trace_index(arrows, idx, shape)
        raw_traces.append((path, hit_target, hit_steps))
        path_set = set(path)
        for prev in path_sets:
            if path_set & prev:
                overlap = True
        path_sets.append(path_set)

    traces = []
    for idx, (path, hit_target, hit_steps) in enumerate(raw_traces):
        traces.append(
            {
                "arrow_id": idx,
                "path": tuple(path),
                "path_set": path_sets[idx],
                "hit_target": hit_target,
                "hit_steps": hit_steps,
            }
        )
    return traces, overlap


def _simulate(arrows: list[dict], shape: tuple[int, int], base_grid: np.ndarray | None = None) -> dict:
    traces, overlap = _geometry(arrows, shape)
    out = None if base_grid is None else np.array(base_grid, dtype=int, copy=True)

    heap: list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int], int, int, int]] = []
    activated: set[int] = set()
    incoming_color: dict[int, int] = {}
    activation_order: list[tuple[int, int, int, int]] = []
    initial_hit_count = 0
    boundary_count = 0
    ambiguous = False

    for arrow in arrows:
        trace = traces[arrow["id"]]
        if out is not None:
            for r, c in trace["path"]:
                out[r, c] = arrow["color"]
        if trace["hit_target"] is None:
            boundary_count += 1
            continue
        initial_hit_count += 1
        target_id = int(trace["hit_target"])
        hit_steps = int(trace["hit_steps"])
        heapq.heappush(
            heap,
            (
                hit_steps,
                arrows[target_id]["sort_key"],
                arrow["sort_key"],
                arrow["id"],
                target_id,
                arrow["color"],
            ),
        )

    while heap:
        time, _target_key, _source_key, source_id, target_id, color = heapq.heappop(heap)
        same_time_hits = [(source_id, color)]
        while heap and heap[0][0] == time and heap[0][4] == target_id:
            _, _, _, other_source_id, _, other_color = heapq.heappop(heap)
            same_time_hits.append((other_source_id, other_color))

        if target_id in activated:
            continue

        if len({hit_color for _src, hit_color in same_time_hits}) > 1:
            ambiguous = True

        activated.add(target_id)
        incoming_color[target_id] = int(color)
        activation_order.append((int(time), int(target_id), int(color), int(source_id)))

        trace = traces[target_id]
        if out is not None:
            for r, c in trace["path"]:
                out[r, c] = color

        if trace["hit_target"] is None:
            continue
        next_target = int(trace["hit_target"])
        next_time = int(time) + int(trace["hit_steps"])
        heapq.heappush(
            heap,
            (
                next_time,
                arrows[next_target]["sort_key"],
                arrows[target_id]["sort_key"],
                target_id,
                next_target,
                int(color),
            ),
        )

    return {
        "traces": traces,
        "output": out,
        "path_overlap": overlap,
        "ambiguous_same_target": ambiguous,
        "initial_hit_count": initial_hit_count,
        "boundary_count": boundary_count,
        "activated": activated,
        "incoming_color": incoming_color,
        "activation_order": activation_order,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw all arrow rays, then propagate activations in arrival order so later
    infected rays overwrite earlier same-path rays from the activated arrows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(g >= 0), "colors must be non-negative integers"

    arrows = _parse_arrows(g)
    info = _simulate(arrows, g.shape, base_grid=g)
    return info["output"]


def _arrow_halo(shape: tuple[int, int], arrow: dict) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in arrow["cells"]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    halo.add((nr, nc))
    return halo


def _can_place_arrow(shape: tuple[int, int], arrow: dict, blocked: set[tuple[int, int]]) -> bool:
    return _in_bounds(shape, arrow["cells"]) and all(cell not in blocked for cell in arrow["cells"])


def _random_color(palette: list[int], avoid: int | None = None) -> int:
    choices = [color for color in palette if color != avoid]
    if choices and random.random() < 0.8:
        return int(random.choice(choices))
    return int(random.choice(palette))


def _random_arrow(shape: tuple[int, int], palette: list[int]) -> dict:
    h, w = shape
    while True:
        direction = random.choice(DIR_ORDER)
        tip = (random.randrange(h), random.randrange(w))
        arrow = _make_arrow(tip, direction, random.choice(palette))
        if _in_bounds(shape, arrow["cells"]):
            return arrow


def _render_input(shape: tuple[int, int], arrows: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = arrow["color"]
    return grid


def _layout_metrics(shape: tuple[int, int], arrows: list[dict]) -> dict:
    ordered = _sort_and_index_arrows(arrows)
    grid = _render_input(shape, ordered)
    parsed = _parse_arrows(grid)
    assert len(parsed) == len(ordered), "generator produced an unparsable layout"
    info = _simulate(parsed, shape, base_grid=grid)
    info["grid"] = grid
    info["arrows"] = parsed
    return info


def _find_arrow_index(arrows: list[dict], tip: tuple[int, int]) -> int:
    for idx, arrow in enumerate(arrows):
        if arrow["tip"] == tip:
            return idx
    raise AssertionError("missing arrow tip")


def _try_place_chain(
    shape: tuple[int, int],
    existing: list[dict],
    blocked: set[tuple[int, int]],
    length: int,
    palette: list[int],
    require_turn: bool,
) -> tuple[list[dict], set[tuple[int, int]]] | None:
    for _ in range(CHAIN_RETRIES):
        arrows = list(existing)
        local_blocked = set(blocked)
        chain: list[dict] = []

        start = _random_arrow(shape, palette)
        if not _can_place_arrow(shape, start, local_blocked):
            continue
        arrows.append(start)
        chain.append(start)
        local_blocked |= _arrow_halo(shape, start)
        turned = False

        ok = True
        for i in range(1, length):
            prev = chain[-1]
            dr, dc = DIR_TO_VEC[prev["dir"]]
            candidate_dirs = [direction for direction in DIR_ORDER if direction != OPPOSITE[prev["dir"]]]
            if require_turn and not turned and i == length - 1:
                candidate_dirs = [direction for direction in candidate_dirs if direction != prev["dir"]] or candidate_dirs
            random.shuffle(candidate_dirs)

            placed = None
            for _ in range(40):
                gap = random.randint(*CHAIN_GAP_RANGE)
                new_tip = (prev["tip"][0] + dr * gap, prev["tip"][1] + dc * gap)
                if not (0 <= new_tip[0] < shape[0] and 0 <= new_tip[1] < shape[1]):
                    continue
                dirs = list(candidate_dirs)
                random.shuffle(dirs)
                for direction in dirs:
                    color = _random_color(palette, avoid=prev["color"])
                    arrow = _make_arrow(new_tip, direction, color)
                    if not _can_place_arrow(shape, arrow, local_blocked):
                        continue
                    temp = list(arrows) + [arrow]
                    prev_idx = len(arrows) - 1
                    new_idx = len(arrows)
                    path, hit_target, _hit_steps = _trace_index(temp, prev_idx, shape)
                    if hit_target != new_idx or not path:
                        continue
                    if i == length - 1:
                        last_path, last_hit_target, _ = _trace_index(temp, new_idx, shape)
                        if last_hit_target is not None or not last_path:
                            continue
                    placed = arrow
                    break
                if placed is not None:
                    break

            if placed is None:
                ok = False
                break

            turned = turned or placed["dir"] != prev["dir"]
            arrows.append(placed)
            chain.append(placed)
            local_blocked |= _arrow_halo(shape, placed)

        if ok and (not require_turn or turned):
            return chain, local_blocked
    return None


def _try_add_isolated(
    shape: tuple[int, int],
    arrows: list[dict],
    blocked: set[tuple[int, int]],
    palette: list[int],
) -> tuple[dict, set[tuple[int, int]]] | None:
    for _ in range(ISOLATED_RETRIES):
        arrow = _random_arrow(shape, palette)
        if not _can_place_arrow(shape, arrow, blocked):
            continue
        temp = list(arrows) + [arrow]
        metrics = _layout_metrics(shape, temp)
        if metrics["path_overlap"] or metrics["ambiguous_same_target"]:
            continue

        ordered = metrics["arrows"]
        new_idx = _find_arrow_index(ordered, arrow["tip"])
        new_trace = metrics["traces"][new_idx]
        if new_trace["hit_target"] is not None or not new_trace["path"]:
            continue
        if any(trace["hit_target"] == new_idx for idx, trace in enumerate(metrics["traces"]) if idx != new_idx):
            continue
        return arrow, blocked | _arrow_halo(shape, arrow)
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        shape = (
            random.randint(*GRID_H_RANGE),
            random.randint(*GRID_W_RANGE),
        )
        palette = random.sample(list(COLOR_POOL), random.randint(*PALETTE_SIZE_RANGE))
        want_long_chain = random.random() < 0.55
        chain_len = random.randint(3, 5) if want_long_chain else random.randint(2, 3)
        require_turn = chain_len >= 2 and random.random() < 0.7

        arrows: list[dict] = []
        blocked: set[tuple[int, int]] = set()

        chain_result = _try_place_chain(shape, arrows, blocked, chain_len, palette, require_turn)
        if chain_result is None:
            continue
        chain, blocked = chain_result
        arrows.extend(chain)

        isolated_target = random.randint(1, 3)
        ok = True
        for _ in range(isolated_target):
            isolated = _try_add_isolated(shape, arrows, blocked, palette)
            if isolated is None:
                ok = False
                break
            arrow, blocked = isolated
            arrows.append(arrow)
        if not ok:
            continue

        metrics = _layout_metrics(shape, arrows)
        if metrics["path_overlap"] or metrics["ambiguous_same_target"]:
            continue
        if metrics["initial_hit_count"] < 1 or metrics["boundary_count"] < 1:
            continue
        if len(metrics["activated"]) < 1:
            continue
        if want_long_chain and len(metrics["activated"]) < 2:
            continue
        if len({arrow["dir"] for arrow in metrics["arrows"]}) < 2:
            continue
        if not any(
            metrics["incoming_color"][idx] != metrics["arrows"][idx]["color"]
            for idx in metrics["incoming_color"]
        ):
            continue
        if np.count_nonzero(metrics["output"] != metrics["grid"]) < 6:
            continue
        return metrics["grid"]

    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
