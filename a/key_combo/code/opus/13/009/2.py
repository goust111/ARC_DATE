from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    dirs = DIR4 if connectivity == 4 else DIR8
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def dims_of_norm(norm: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in norm) + 1, max(c for _, c in norm) + 1


def mask_from_norm(norm: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = dims_of_norm(norm)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def color_components(grid: np.ndarray, colors: set[int]) -> list[np.ndarray]:
    return components_mask(np.isin(grid, list(colors)), connectivity=4)


def extract_cavities(grid: np.ndarray) -> list[dict]:
    blue_coords = np.argwhere(grid == BLUE)
    assert blue_coords.size > 0, "expected one blue structure"
    r0, c0, r1, c1 = bbox_coords(blue_coords)
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(box[0, :] == BLUE), "cavities must sit under a blue roof"
    cavities = []
    for part in components_mask(box == BLACK, connectivity=4):
        if np.any(part[:, 0] == box.shape[0] - 1):
            abs_coords = part + np.array([r0, c0], dtype=int)
            cavities.append(
                {
                    "coords": abs_coords,
                    "bbox": bbox_coords(abs_coords),
                    "norm": normalize_cells([tuple(map(int, rc)) for rc in part]),
                }
            )
    cavities.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return cavities


def fold_shape(
    total_len: int,
    fixed_len: int,
    rod_orient: str,
    endpoint_sign: int,
    fold_sign: int,
) -> tuple[tuple[int, int], ...]:
    tail_len = total_len - fixed_len
    cells = {(0, 0)}
    if rod_orient == "h":
        for step in range(1, fixed_len):
            cells.add((0, endpoint_sign * step))
        for step in range(1, tail_len + 1):
            cells.add((fold_sign * step, 0))
    else:
        for step in range(1, fixed_len):
            cells.add((endpoint_sign * step, 0))
        for step in range(1, tail_len + 1):
            cells.add((0, fold_sign * step))
    return normalize_cells(cells)


def parse_rod(grid: np.ndarray, coords: np.ndarray) -> dict:
    cells = [tuple(map(int, rc)) for rc in coords]
    colors = {(r, c): int(grid[r, c]) for r, c in cells}
    endpoints = [(r, c) for r, c in cells if colors[(r, c)] == GREEN]
    hinges = [(r, c) for r, c in cells if colors[(r, c)] == RED]
    gray = [(r, c) for r, c in cells if colors[(r, c)] == GREY]

    assert len(endpoints) == 1 and len(hinges) == 1, "rod needs one endpoint and one hinge"
    endpoint = endpoints[0]
    hinge = hinges[0]

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    assert len(rows) == 1 or len(cols) == 1, "rod must be straight"

    if len(rows) == 1:
        orient = "h"
        ordered = sorted(cells, key=lambda p: p[1])
        values = [p[1] for p in ordered]
        assert values == list(range(values[0], values[-1] + 1)), "rod must be contiguous"
        endpoint_index = ordered.index(endpoint)
        hinge_index = ordered.index(hinge)
        assert endpoint_index in (0, len(ordered) - 1), "endpoint must sit at an end"
        assert 0 < hinge_index < len(ordered) - 1, "hinge must be interior"
        fixed_len = abs(endpoint_index - hinge_index) + 1
        endpoint_sign = -1 if endpoint[1] < hinge[1] else 1
    else:
        orient = "v"
        ordered = sorted(cells, key=lambda p: p[0])
        values = [p[0] for p in ordered]
        assert values == list(range(values[0], values[-1] + 1)), "rod must be contiguous"
        endpoint_index = ordered.index(endpoint)
        hinge_index = ordered.index(hinge)
        assert endpoint_index in (0, len(ordered) - 1), "endpoint must sit at an end"
        assert 0 < hinge_index < len(ordered) - 1, "hinge must be interior"
        fixed_len = abs(endpoint_index - hinge_index) + 1
        endpoint_sign = -1 if endpoint[0] < hinge[0] else 1

    total_len = len(ordered)
    tail_len = total_len - fixed_len
    assert fixed_len >= 3 and tail_len >= 1, "rod geometry must support folding"

    candidates = {
        fold_shape(total_len, fixed_len, orient, endpoint_sign, fold_sign)
        for fold_sign in (-1, 1)
    }
    return {
        "coords": coords,
        "gray_coords": gray,
        "bbox": bbox_coords(coords),
        "candidates": candidates,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rod when one 90-degree fold makes it exactly match a lower cavity in the blue structure.

    Input structure:
    - A black background contains one lower blue 4-connected structure whose bounding box encloses several black cavities touching its bottom edge.
    - Above it sit several straight gray rods, each carrying one green endpoint marker at a rod end and one red hinge marker in the interior.
    - Every rod is a 1-pixel-wide horizontal or vertical line component formed by gray cells plus those two markers.

    Rule:
    - Treat the segment from the green endpoint through the red hinge as fixed, and the gray tail on the far side of the hinge as the moving segment.
    - Fold that tail 90 degrees around the hinge in both possible perpendicular directions, producing two candidate shapes for the whole rod.
    - If either folded shape is a translation-exact match for an unused cavity and therefore does not overlap blue, recolor only that rod's original gray cells to red; otherwise leave the rod unchanged.

    Key insight:
    - The cavity geometry determines the correct fold direction, so solving requires trying both hinge folds and matching the resulting L-shape against the bottom cavities.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    cavities = extract_cavities(g)
    cavity_norms = [item["norm"] for item in cavities]
    used: set[int] = set()

    rods = [parse_rod(g, part) for part in color_components(g, {GREY, RED, GREEN})]
    rods.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

    for rod in rods:
        chosen = None
        for idx, norm in enumerate(cavity_norms):
            if idx in used:
                continue
            if norm in rod["candidates"]:
                chosen = idx
                break
        if chosen is None:
            continue
        used.add(chosen)
        for r, c in rod["gray_coords"]:
            out[r, c] = RED
    return out


def _all_match_options() -> list[dict]:
    options = []
    for total_len in range(4, 7):
        for fixed_len in range(3, total_len):
            for rod_orient in ("h", "v"):
                for endpoint_sign in (-1, 1):
                    a = fold_shape(total_len, fixed_len, rod_orient, endpoint_sign, -1)
                    b = fold_shape(total_len, fixed_len, rod_orient, endpoint_sign, 1)
                    options.append(
                        {
                            "total_len": total_len,
                            "fixed_len": fixed_len,
                            "rod_orient": rod_orient,
                            "endpoint_sign": endpoint_sign,
                            "match_shape": a,
                            "alt_shape": b,
                        }
                    )
                    options.append(
                        {
                            "total_len": total_len,
                            "fixed_len": fixed_len,
                            "rod_orient": rod_orient,
                            "endpoint_sign": endpoint_sign,
                            "match_shape": b,
                            "alt_shape": a,
                        }
                    )
    random.shuffle(options)
    return options


MATCH_OPTIONS = _all_match_options()


def _decoy_shapes(max_height: int) -> list[tuple[tuple[int, int], ...]]:
    shapes = set()
    for length in range(3, 7):
        shapes.add(tuple((0, c) for c in range(length)))
    for length in range(3, max_height + 1):
        shapes.add(tuple((r, 0) for r in range(length)))
    for anchor in (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ):
        shapes.add(tuple(sorted(anchor)))
    return [shape for shape in shapes if dims_of_norm(shape)[0] <= max_height]


def _sample_cavity_plan() -> tuple[int, list[dict], list[tuple[tuple[int, int], ...]]] | None:
    cavity_count = random.randint(2, 4)
    matched_count = random.randint(1, min(2, cavity_count - 1))
    box_h = random.randint(3, 4)
    max_cavity_h = box_h - 1

    valid_options = [
        opt
        for opt in MATCH_OPTIONS
        if dims_of_norm(opt["match_shape"])[0] <= max_cavity_h
    ]
    if not valid_options:
        return None

    matched: list[dict] = []
    blocked_shapes: set[tuple[tuple[int, int], ...]] = set()
    for opt in valid_options:
        if len(matched) == matched_count:
            break
        shape = opt["match_shape"]
        alt = opt["alt_shape"]
        if shape in blocked_shapes or alt in blocked_shapes:
            continue
        matched.append(opt)
        blocked_shapes.add(shape)
        blocked_shapes.add(alt)
    if len(matched) != matched_count:
        return None

    decoys = [
        shape
        for shape in _decoy_shapes(max_cavity_h)
        if shape not in blocked_shapes
    ]
    if len(decoys) < cavity_count - matched_count:
        return None
    random.shuffle(decoys)
    chosen_decoys = decoys[: cavity_count - matched_count]
    return box_h, matched, chosen_decoys


def _render_structure(
    box_h: int,
    cavity_shapes: list[tuple[tuple[int, int], ...]],
) -> tuple[np.ndarray, list[dict]]:
    order = list(cavity_shapes)
    random.shuffle(order)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gaps = [random.randint(1, 2) for _ in range(max(0, len(order) - 1))]

    width = left_margin + right_margin + sum(dims_of_norm(shape)[1] for shape in order) + sum(gaps)
    patch = np.full((box_h, width), BLUE, dtype=int)

    cavity_infos = []
    col = left_margin
    for idx, shape in enumerate(order):
        h, w = dims_of_norm(shape)
        row = box_h - h
        for rr, cc in shape:
            patch[row + rr, col + cc] = BLACK
        cavity_infos.append(
            {
                "norm": shape,
                "local_cells": {(row + rr, col + cc) for rr, cc in shape},
            }
        )
        col += w
        if idx < len(gaps):
            col += gaps[idx]

    blue_parts = components_mask(patch == BLUE, connectivity=4)
    assert len(blue_parts) == 1, "structure must stay connected"
    return patch, cavity_infos


def _rod_input_cells(spec: dict, start_r: int, start_c: int) -> dict:
    total_len = spec["total_len"]
    fixed_len = spec["fixed_len"]
    endpoint_sign = spec["endpoint_sign"]
    rod_orient = spec["rod_orient"]
    tail_len = total_len - fixed_len
    if rod_orient == "h":
        cells = [(start_r, start_c + offset) for offset in range(total_len)]
        if endpoint_sign == -1:
            endpoint = cells[0]
            hinge = cells[fixed_len - 1]
        else:
            endpoint = cells[-1]
            hinge = cells[tail_len]
    else:
        cells = [(start_r + offset, start_c) for offset in range(total_len)]
        if endpoint_sign == -1:
            endpoint = cells[0]
            hinge = cells[fixed_len - 1]
        else:
            endpoint = cells[-1]
            hinge = cells[tail_len]
    return {
        "cells": cells,
        "endpoint": endpoint,
        "hinge": hinge,
        "gray": [cell for cell in cells if cell not in (endpoint, hinge)],
    }


def _expanded(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in cells:
        for dr, dc in DIR8 + [(0, 0)]:
            blocked.add((r + dr, c + dc))
    return blocked


def _place_rods(grid: np.ndarray, specs: list[dict], top_limit: int) -> bool:
    blocked = _expanded({tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)})
    h, w = grid.shape
    placed: list[dict] = []

    for spec in specs:
        positions = []
        if spec["rod_orient"] == "h":
            for r in range(0, top_limit):
                for c in range(0, w - spec["total_len"] + 1):
                    rec = _rod_input_cells(spec, r, c)
                    if any(cell in blocked for cell in rec["cells"]):
                        continue
                    positions.append(rec)
        else:
            for r in range(0, top_limit - spec["total_len"] + 1):
                for c in range(0, w):
                    rec = _rod_input_cells(spec, r, c)
                    if any(cell in blocked for cell in rec["cells"]):
                        continue
                    positions.append(rec)
        if not positions:
            return False
        rec = random.choice(positions)
        placed.append(rec)
        blocked |= _expanded(rec["cells"])

    for rec in placed:
        for r, c in rec["gray"]:
            grid[r, c] = GREY
        er, ec = rec["endpoint"]
        hr, hc = rec["hinge"]
        grid[er, ec] = GREEN
        grid[hr, hc] = RED
    return True


def _sample_unmatched_spec() -> dict:
    total_len = random.randint(7, 8)
    fixed_len = random.randint(3, total_len - 1)
    return {
        "total_len": total_len,
        "fixed_len": fixed_len,
        "rod_orient": random.choice(["h", "v"]),
        "endpoint_sign": random.choice([-1, 1]),
    }


def _attempt_generate() -> np.ndarray | None:
    plan = _sample_cavity_plan()
    if plan is None:
        return None
    box_h, matched_options, decoy_shapes = plan
    cavity_shapes = [opt["match_shape"] for opt in matched_options] + decoy_shapes
    structure_patch, _cavity_infos = _render_structure(box_h, cavity_shapes)

    min_width = structure_patch.shape[1]
    grid_w = random.randint(max(12, min_width), min(30, min_width + 6))
    rod_count = random.randint(max(3, len(matched_options) + 1), 5)
    grid_h = random.randint(max(15, box_h + 10), 24)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    struct_c0 = random.randint(0, grid_w - structure_patch.shape[1])
    bottom_margin = random.randint(1, 3)
    struct_r0 = grid_h - bottom_margin - box_h
    if struct_r0 < 8:
        return None
    grid[struct_r0 : struct_r0 + box_h, struct_c0 : struct_c0 + structure_patch.shape[1]] = structure_patch

    specs = [
        {
            "total_len": opt["total_len"],
            "fixed_len": opt["fixed_len"],
            "rod_orient": opt["rod_orient"],
            "endpoint_sign": opt["endpoint_sign"],
        }
        for opt in matched_options
    ]
    while len(specs) < rod_count:
        specs.append(_sample_unmatched_spec())
    random.shuffle(specs)

    if not _place_rods(grid, specs, struct_r0 - 1):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    rod_parts = [parse_rod(grid, part) for part in color_components(grid, {GREY, RED, GREEN})]
    matched = 0
    for rod in rod_parts:
        if any(out[r, c] == RED and grid[r, c] == GREY for r, c in rod["gray_coords"]):
            matched += 1
    if matched < 1 or matched >= len(rod_parts):
        return None

    cavities = extract_cavities(grid)
    cavity_norms = [cavity["norm"] for cavity in cavities]
    if len(cavity_norms) != len(set(cavity_norms)):
        return None
    if len(cavity_norms) != len(cavity_shapes):
        return None

    for cavity in cavities:
        r0, c0, r1, c1 = cavity["bbox"]
        if r0 == struct_r0 or c0 == struct_c0 or c1 == struct_c0 + structure_patch.shape[1] - 1:
            return None

    if not np.any(grid == GREY):
        return None
    if not np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fold-to-cavity ARC input")
