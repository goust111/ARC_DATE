from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 7
MISSING_TO_PATH_COLOR = {
    (0, 0): 3,  # top-left -> green
    (0, 1): 8,  # top-right -> teal
    (1, 0): 4,  # bottom-left -> yellow
    (1, 1): 6,  # bottom-right -> magenta
}
ENDPOINT_COLOR_CHOICES = (1, 2, 5, 9)
ORIENTATIONS = tuple(MISSING_TO_PATH_COLOR)
GEN_MAX_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _marker_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    components: list[np.ndarray] = []
    for r, c in np.argwhere(grid == MARKER_COLOR):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        seen.add(cell)
        queue = deque([cell])
        coords: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            coords.append((int(rr), int(cc)))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == MARKER_COLOR and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(np.array(coords, dtype=int))
    return components


def _parse_markers(grid: np.ndarray) -> list[dict[str, object]]:
    markers: list[dict[str, object]] = []
    for coords in _marker_components(grid):
        assert coords.shape[0] == 3, "each marker must be an L-triomino"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "marker must occupy a 2x2 box"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert int(np.sum(patch == MARKER_COLOR)) == 3, "marker patch must contain three orange cells"
        missing = np.argwhere(patch != MARKER_COLOR)
        assert missing.shape == (1, 2), "marker must have exactly one missing corner"
        mr, mc = map(int, missing[0])
        assert int(patch[mr, mc]) == BG_COLOR, "marker hole must be background"
        markers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "missing": (mr, mc),
                "path_color": int(MISSING_TO_PATH_COLOR[(mr, mc)]),
            }
        )
    return markers


def _group_endpoints(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    endpoints: dict[int, list[tuple[int, int]]] = defaultdict(list)
    mask = (grid != BG_COLOR) & (grid != MARKER_COLOR)
    for r, c in np.argwhere(mask):
        endpoints[int(grid[r, c])].append((int(r), int(c)))
    return endpoints


def _path_middle(start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    tr, tc = target
    assert sr != tr and sc != tc, "endpoint pair must require both legs of the L-path"
    cells: list[tuple[int, int]] = []
    step_c = 1 if tc > sc else -1
    for c in range(sc + step_c, tc + step_c, step_c):
        cells.append((int(sr), int(c)))
    step_r = 1 if tr > sr else -1
    for r in range(sr + step_r, tr + step_r, step_r):
        cells.append((int(r), int(tc)))
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        if cell in seen or cell == start or cell == target:
            continue
        seen.add(cell)
        out.append(cell)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair with an L-path whose color is encoded by the start marker.

    Input structure:
    - The grid has black background, 2-3 same-colored endpoint pairs, and orange 2x2 L-triomino markers.
    - Each pair consists of two single-pixel endpoints that are not in the same row or column.
    - Exactly one endpoint in each pair has a marker directly above it, with the endpoint below one of the marker's two columns.

    Rule:
    - Pair endpoints by their shared color, and treat the endpoint with the marker above it as the start.
    - Decode the marker's missing corner into the path color: top-left -> 3, top-right -> 8, bottom-left -> 4, bottom-right -> 6.
    - Erase all orange marker pixels to black, then draw a horizontal segment from the start to the target column and a vertical segment from that corner to the target row.
    - Keep both endpoint pixels in their original colors; only the intermediate path pixels use the decoded path color.

    Key insight:
    - The same L marker has two jobs at once: it tells which endpoint is the start, and its orientation separately encodes the color of the new path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers = _parse_markers(g)
    endpoints_by_color = _group_endpoints(g)
    out = g.copy()
    out[out == MARKER_COLOR] = BG_COLOR

    used_markers: set[int] = set()
    for color, coords in endpoints_by_color.items():
        assert len(coords) == 2, f"endpoint color {color} must appear exactly twice"
        start_info: list[tuple[tuple[int, int], int]] = []
        for coord in coords:
            r, c = coord
            matches = []
            for idx, marker in enumerate(markers):
                mr0, mc0, mr1, mc1 = marker["bbox"]
                if int(mr1) + 1 == int(r) and int(mc0) <= int(c) <= int(mc1):
                    matches.append(idx)
            assert len(matches) <= 1, "an endpoint cannot belong to multiple markers"
            if matches:
                start_info.append((coord, int(matches[0])))

        assert len(start_info) == 1, f"color {color} must have exactly one marked start endpoint"
        start, marker_idx = start_info[0]
        assert marker_idx not in used_markers, "each marker must be assigned once"
        used_markers.add(marker_idx)
        target = coords[0] if coords[1] == start else coords[1]
        path_color = int(markers[marker_idx]["path_color"])
        for r, c in _path_middle(start, target):
            out[r, c] = path_color

    assert len(used_markers) == len(markers), "all markers must map to some endpoint pair"
    return out


def _build_marker_cells(start: tuple[int, int], anchor_side: int, missing: tuple[int, int]) -> set[tuple[int, int]]:
    sr, sc = start
    mc0 = int(sc - anchor_side)
    cells = set()
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                cells.add((int(sr - 2 + dr), int(mc0 + dc)))
    return cells


def _marker_support_cells(start: tuple[int, int], anchor_side: int) -> set[tuple[int, int]]:
    sr, sc = start
    mc0 = int(sc - anchor_side)
    return {(int(sr), int(mc0)), (int(sr), int(mc0 + 1))}


def _marker_bbox_cells(start: tuple[int, int], anchor_side: int) -> set[tuple[int, int]]:
    sr, sc = start
    mc0 = int(sc - anchor_side)
    return {(int(sr - 2 + dr), int(mc0 + dc)) for dr in range(2) for dc in range(2)}


def _sample_signed_pair(limit: int, min_start: int, sign: int) -> tuple[int, int]:
    if sign > 0:
        start = random.randint(min_start, limit - 3)
        target = random.randint(start + 2, limit - 1)
    else:
        target = random.randint(0, limit - 3)
        start = random.randint(max(min_start, target + 2), limit - 1)
    return int(start), int(target)


def _expected_from_pairs(shape: tuple[int, int], pairs: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    inp = np.full((h, w), BG_COLOR, dtype=int)
    out = np.full((h, w), BG_COLOR, dtype=int)
    for pair in pairs:
        start = pair["start"]
        target = pair["target"]
        endpoint_color = int(pair["endpoint_color"])
        path_color = int(MISSING_TO_PATH_COLOR[pair["missing"]])
        inp[start] = endpoint_color
        inp[target] = endpoint_color
        out[start] = endpoint_color
        out[target] = endpoint_color
        for r, c in pair["marker_cells"]:
            inp[int(r), int(c)] = MARKER_COLOR
        for r, c in _path_middle(start, target):
            out[int(r), int(c)] = path_color
    return inp, out


def _markers_are_separate(existing: set[tuple[int, int]], new_cells: set[tuple[int, int]]) -> bool:
    for r, c in new_cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (int(r + dr), int(c + dc)) in existing:
                return False
    return True


def _pick_pair(
    h: int,
    w: int,
    endpoint_color: int,
    missing: tuple[int, int],
    row_sign: int,
    col_sign: int,
    occupied_input: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
    endpoint_cells: set[tuple[int, int]],
    blocked_endpoint_cells: set[tuple[int, int]],
    marker_cells_existing: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(600):
        sr, tr = _sample_signed_pair(h, 2, row_sign)
        sc, tc = _sample_signed_pair(w, 0, col_sign)
        start = (int(sr), int(sc))
        target = (int(tr), int(tc))
        anchor_side = random.randint(0, 1)
        mc0 = int(sc - anchor_side)
        if not (0 <= mc0 and mc0 + 1 < w):
            continue

        marker_cells = _build_marker_cells(start, anchor_side, missing)
        marker_bbox = _marker_bbox_cells(start, anchor_side)
        support_cells = _marker_support_cells(start, anchor_side)
        endpoints = {start, target}
        current_input = marker_cells | endpoints
        path_cells = set(_path_middle(start, target))

        if endpoints & blocked_endpoint_cells:
            continue
        if any(cell in endpoint_cells and cell != start for cell in support_cells):
            continue
        if current_input & occupied_input:
            continue
        if marker_bbox & occupied_input:
            continue
        if current_input & reserved_output:
            continue
        if marker_bbox & reserved_output:
            continue
        if path_cells & occupied_input:
            continue
        if path_cells & reserved_output:
            continue
        if not _markers_are_separate(marker_cells_existing, marker_cells):
            continue

        return {
            "start": start,
            "target": target,
            "endpoint_color": int(endpoint_color),
            "missing": missing,
            "anchor_side": int(anchor_side),
            "marker_cells": marker_cells,
            "marker_bbox": marker_bbox,
            "support_cells": support_cells,
            "path_cells": path_cells,
        }
    return None


def _generation_semantics_ok(inp: np.ndarray, expected: np.ndarray, pairs: list[dict[str, object]]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    all_paths = [set(pair["path_cells"]) for pair in pairs]
    if len(set().union(*all_paths)) != sum(len(path) for path in all_paths):
        return False

    endpoints = []
    for pair in pairs:
        endpoints.extend([pair["start"], pair["target"]])
        sr, sc = pair["start"]
        tr, tc = pair["target"]
        if abs(sr - tr) < 2 or abs(sc - tc) < 2:
            return False
        if not any(cell[0] == sr and cell[1] != tc for cell in pair["path_cells"]):
            return False
        if not any(cell[1] == tc and cell[0] != sr for cell in pair["path_cells"]):
            return False

    endpoint_set = set(endpoints)
    for path in all_paths:
        if path & endpoint_set:
            return False

    for marker in _parse_markers(inp):
        mr0, mc0, mr1, mc1 = marker["bbox"]
        candidates = []
        for r, c in endpoint_set:
            if int(r) == int(mr1) + 1 and int(mc0) <= int(c) <= int(mc1):
                candidates.append((r, c))
        if len(candidates) != 1:
            return False

    return True


def _gen_attempt(call_id: int) -> np.ndarray | None:
    n_pairs = 3 if call_id % 2 else 2
    min_side = 14 if n_pairs == 3 else 12
    h = random.randint(min_side, 22)
    w = random.randint(min_side, 22)

    endpoint_colors = random.sample(ENDPOINT_COLOR_CHOICES, k=n_pairs)
    lead_orientation = ORIENTATIONS[call_id % len(ORIENTATIONS)]
    other_orientations = random.sample([ori for ori in ORIENTATIONS if ori != lead_orientation], k=n_pairs - 1)
    orientations = [lead_orientation, *other_orientations]

    quadrant_cycle = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    occupied_input: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()
    endpoint_cells: set[tuple[int, int]] = set()
    blocked_endpoint_cells: set[tuple[int, int]] = set()
    marker_cells_existing: set[tuple[int, int]] = set()
    pairs: list[dict[str, object]] = []

    for idx in range(n_pairs):
        if idx == 0:
            row_sign, col_sign = quadrant_cycle[call_id % len(quadrant_cycle)]
        else:
            row_sign, col_sign = random.choice(quadrant_cycle)
        pair = _pick_pair(
            h=h,
            w=w,
            endpoint_color=endpoint_colors[idx],
            missing=orientations[idx],
            row_sign=row_sign,
            col_sign=col_sign,
            occupied_input=occupied_input,
            reserved_output=reserved_output,
            endpoint_cells=endpoint_cells,
            blocked_endpoint_cells=blocked_endpoint_cells,
            marker_cells_existing=marker_cells_existing,
        )
        if pair is None:
            return None

        pairs.append(pair)
        endpoints = {pair["start"], pair["target"]}
        endpoint_cells |= endpoints
        occupied_input |= endpoints | set(pair["marker_bbox"])
        reserved_output |= endpoints | set(pair["path_cells"])
        blocked_endpoint_cells |= set(pair["support_cells"])
        marker_cells_existing |= set(pair["marker_cells"])

    inp, expected = _expected_from_pairs((h, w), pairs)
    if not _generation_semantics_ok(inp, expected, pairs):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_id = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(call_id)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
