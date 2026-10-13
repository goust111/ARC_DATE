from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
ANCHOR_COUNT_OPTIONS = (3, 4)
RAY_COUNT_POOL = (1, 2, 3, 4, 5, 6)
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (16, 22)
MAX_GENERATION_ATTEMPTS = 800

_SEEN_INPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _anchor_cells(anchor: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _find_anchor(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    anchors: list[tuple[int, int, int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            if not bool(mask[r : r + 3, c : c + 3].all()):
                continue
            halo_r0 = max(0, r - 1)
            halo_c0 = max(0, c - 1)
            halo_r1 = min(h, r + 4)
            halo_c1 = min(w, c + 4)
            halo = mask[halo_r0:halo_r1, halo_c0:halo_c1].copy()
            halo[r - halo_r0 : r - halo_r0 + 3, c - halo_c0 : c - halo_c0 + 3] = False
            assert not bool(halo.any()), "anchor must be a clean 3x3 block"
            anchors.append((r, c, r + 2, c + 2))
    assert len(anchors) == 1, "expected exactly one 3x3 anchor per color"
    return anchors[0]


def _ray_terminal(
    coord: tuple[int, int],
    anchor: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = coord
    r0, c0, r1, c1 = anchor
    dr = 1 if r < r0 else -1 if r > r1 else 0
    dc = 1 if c < c0 else -1 if c > c1 else 0
    assert (dr, dc) != (0, 0), "ray pixel cannot lie inside the anchor"

    cr, cc = r, c
    for _ in range(max(shape) * 2):
        nr, nc = cr + dr, cc + dc
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            return (cr, cc), (dr, dc)
        assert _in_bounds(shape, nr, nc), "ray pixel must align to its anchor on row/col/diag"
        cr, cc = nr, nc
    raise AssertionError("ray pixel does not reach its anchor")


def _ray_lines(anchor: tuple[int, int, int, int], shape: tuple[int, int]) -> list[dict[str, object]]:
    r0, c0, r1, c1 = anchor
    raw_lines: list[tuple[str, tuple[int, int], tuple[int, int], str]] = []
    for r in range(r0, r1 + 1):
        raw_lines.append((f"L{r}", (r, c0 - 1), (0, -1), "horizontal"))
        raw_lines.append((f"R{r}", (r, c1 + 1), (0, 1), "horizontal"))
    for c in range(c0, c1 + 1):
        raw_lines.append((f"U{c}", (r0 - 1, c), (-1, 0), "vertical"))
        raw_lines.append((f"D{c}", (r1 + 1, c), (1, 0), "vertical"))
    raw_lines.extend(
        [
            ("UL", (r0 - 1, c0 - 1), (-1, -1), "diagonal"),
            ("UR", (r0 - 1, c1 + 1), (-1, 1), "diagonal"),
            ("DL", (r1 + 1, c0 - 1), (1, -1), "diagonal"),
            ("DR", (r1 + 1, c1 + 1), (1, 1), "diagonal"),
        ]
    )

    lines: list[dict[str, object]] = []
    for line_id, terminal, step, orientation in raw_lines:
        dr, dc = step
        cells: list[tuple[int, int]] = []
        rr, cc = terminal[0] + dr, terminal[1] + dc
        while _in_bounds(shape, rr, cc):
            cells.append((rr, cc))
            rr += dr
            cc += dc
        if cells:
            lines.append(
                {
                    "id": line_id,
                    "terminal": terminal,
                    "step": step,
                    "orientation": orientation,
                    "cells": cells,
                }
            )
    return lines


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a[0] - gap, a[1] - gap, a[2] + gap, a[3] + gap
    br0, bc0, br1, bc1 = b[0] - gap, b[1] - gap, b[2] + gap, b[3] + gap
    rows_overlap = not (ar1 < br0 or br1 < ar0)
    cols_overlap = not (ac1 < bc0 or bc1 < ac0)
    return rows_overlap and cols_overlap


def _specs_are_geometrically_valid(shape: tuple[int, int], specs: list[dict[str, object]]) -> bool:
    if len({int(spec["color"]) for spec in specs}) != len(specs):
        return False

    anchors = [tuple(spec["anchor"]) for spec in specs]
    for i, anchor_i in enumerate(anchors):
        for anchor_j in anchors[i + 1 :]:
            if _boxes_too_close(anchor_i, anchor_j):
                return False

    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        for cell in _anchor_cells(tuple(spec["anchor"])):
            if cell in occupied:
                return False
            occupied.add(cell)

    for spec in specs:
        if len(spec["rays"]) != int(spec["count"]):
            return False
        for ray in spec["rays"]:
            cell = tuple(ray["cell"])
            if cell in occupied:
                return False
            occupied.add(cell)

    for spec in specs:
        anchor = tuple(spec["anchor"])
        for ray in spec["rays"]:
            cell = tuple(ray["cell"])
            for nr, nc in _neighbors8(*cell):
                if _in_bounds(shape, nr, nc) and (nr, nc) in occupied:
                    return False
            try:
                _ray_terminal(cell, anchor, shape)
            except AssertionError:
                return False

    return True


def _can_place_isolated(cell: tuple[int, int], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c) or cell in occupied:
        return False
    for nr, nc in _neighbors8(r, c):
        if _in_bounds(shape, nr, nc) and (nr, nc) in occupied:
            return False
    return True


def _render_output(specs: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(
        specs,
        key=lambda spec: (
            int(spec["count"]),
            int(spec["center_r"]),
            int(spec["center_c"]),
        ),
    )
    height = max(int(spec["count"]) for spec in ordered)
    out = np.full((height, len(ordered)), BG, dtype=int)
    for col, spec in enumerate(ordered):
        count = int(spec["count"])
        out[height - count :, col] = int(spec["color"])
    return out


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.tobytes()


def _sample_anchor_layout(height: int, width: int, n_anchors: int) -> list[tuple[int, int, int, int]] | None:
    candidates = [(r, c) for r in range(2, height - 4) for c in range(2, width - 4)]
    for _ in range(200):
        random.shuffle(candidates)
        anchors: list[tuple[int, int, int, int]] = []
        for r0, c0 in candidates:
            anchor = (r0, c0, r0 + 2, c0 + 2)
            if all(not _boxes_too_close(anchor, other) for other in anchors):
                anchors.append(anchor)
                if len(anchors) == n_anchors:
                    return anchors
    return None


def _order(indices: list[int], key) -> tuple[int, ...]:
    return tuple(sorted(indices, key=key))


def _anti_shortcut_ok(specs: list[dict[str, object]]) -> bool:
    indices = list(range(len(specs)))
    count_order = _order(indices, lambda i: (int(specs[i]["count"]),))
    row_order = _order(indices, lambda i: (int(specs[i]["center_r"]), int(specs[i]["center_c"])))
    col_order = _order(indices, lambda i: (int(specs[i]["center_c"]), int(specs[i]["center_r"])))
    color_order = _order(indices, lambda i: (int(specs[i]["color"]),))
    banned = {
        row_order,
        row_order[::-1],
        col_order,
        col_order[::-1],
        color_order,
        color_order[::-1],
    }
    return count_order not in banned


def _pick_rays_for_anchor(
    anchor: tuple[int, int, int, int],
    count: int,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[dict[str, object]] | None:
    lines = _ray_lines(anchor, shape)
    if not lines:
        return None

    for _ in range(250):
        trial_occupied = set(occupied)
        selected: list[dict[str, object]] = []
        used_lines: set[str] = set()

        for step_index in range(count):
            candidates: list[dict[str, object]] = []
            for line in lines:
                line_id = str(line["id"])
                cells = list(line["cells"])
                if step_index < min(2, count - 1):
                    random.shuffle(cells)
                for cell in cells:
                    if _can_place_isolated(cell, trial_occupied, shape):
                        candidates.append(
                            {
                                "cell": cell,
                                "line_id": line_id,
                                "orientation": str(line["orientation"]),
                            }
                        )
            if not candidates:
                break

            if step_index < min(2, count - 1):
                fresh = [cand for cand in candidates if cand["line_id"] not in used_lines]
                if fresh:
                    candidates = fresh

            preferred = [cand for cand in candidates if cand["orientation"] == "diagonal"]
            if preferred and random.random() < 0.35:
                candidates = preferred

            choice = random.choice(candidates)
            cell = tuple(choice["cell"])
            selected.append(choice)
            used_lines.add(str(choice["line_id"]))
            trial_occupied.add(cell)
        else:
            if count >= 4 and len(used_lines) < 2:
                continue
            occupied.clear()
            occupied.update(trial_occupied)
            return selected

    return None


def _build_input_from_specs(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for cell in _anchor_cells(tuple(spec["anchor"])):
            grid[cell] = color
        for ray in spec["rays"]:
            grid[tuple(ray["cell"])] = color
    return grid


def _generation_constraints_ok(specs: list[dict[str, object]]) -> bool:
    counts = [int(spec["count"]) for spec in specs]
    if len(set(counts)) != len(counts):
        return False
    if max(counts) < 4 or min(counts) > 2:
        return False

    orientations = [str(ray["orientation"]) for spec in specs for ray in spec["rays"]]
    if "diagonal" not in orientations:
        return False
    if not any(ori in {"horizontal", "vertical"} for ori in orientations):
        return False

    return _anti_shortcut_ok(specs)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    n_anchors = random.choice(ANCHOR_COUNT_OPTIONS)
    anchors = _sample_anchor_layout(height, width, n_anchors)
    if anchors is None:
        return None

    colors = random.sample(COLOR_POOL, n_anchors)
    counts = random.sample(RAY_COUNT_POOL, n_anchors)
    random.shuffle(colors)
    random.shuffle(counts)

    specs: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    for anchor, color, count in zip(anchors, colors, counts):
        r0, c0, _r1, _c1 = anchor
        spec = {
            "anchor": anchor,
            "color": int(color),
            "count": int(count),
            "center_r": r0 + 1,
            "center_c": c0 + 1,
            "rays": [],
        }
        specs.append(spec)
        occupied.update(_anchor_cells(anchor))

    if not _anti_shortcut_ok(specs):
        return None

    for spec in specs:
        rays = _pick_rays_for_anchor(tuple(spec["anchor"]), int(spec["count"]), (height, width), occupied)
        if rays is None:
            return None
        spec["rays"] = rays

    if not _generation_constraints_ok(specs):
        return None
    if not _specs_are_geometrically_valid((height, width), specs):
        return None

    grid = _build_input_from_specs((height, width), specs)
    out = _render_output(specs)

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    input_sig = _signature(grid)
    output_sig = _signature(out)
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None

    _SEEN_INPUT_SIGNATURES.add(input_sig)
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each anchor block's ray pixels and rewrite the colors as vertical bars sorted by ray count.

    Input structure:
    - A mostly black background contains 3-4 distinct non-background colors.
    - For each color, exactly one clean 3x3 solid block is the anchor.
    - Any remaining pixel of that color is a single isolated ray pixel lying on a row, column, or 45-degree diagonal that points back to the anchor.

    Rule:
    - For each non-background color, identify its unique 3x3 anchor block and count that color's non-anchor ray pixels.
    - Sort the colors by ascending ray count; if counts tie, break ties by the anchor center row.
    - Output a grid whose width is the number of anchors and whose height is the maximum ray count; each sorted color fills one bottom-aligned vertical bar of exactly its count.

    Key insight:
    - The large input layout matters only to recover one number per anchor color, and that hidden count becomes the height and left-to-right order of a completely new bar-chart output.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert 3 <= len(colors) <= 4, "expected 3-4 anchor colors"

    specs: list[dict[str, int]] = []
    for color in colors:
        mask = g == color
        anchor = _find_anchor(mask)
        r0, c0, r1, c1 = anchor
        ray_count = 0
        for raw_r, raw_c in np.argwhere(mask):
            r, c = int(raw_r), int(raw_c)
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            same_color_neighbors = 0
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == color:
                    same_color_neighbors += 1
            assert same_color_neighbors == 0, "ray pixels must be isolated singletons"
            _ray_terminal((r, c), anchor, g.shape)
            ray_count += 1
        assert ray_count > 0, "each anchor must own at least one ray pixel"
        specs.append(
            {
                "color": color,
                "count": ray_count,
                "center_r": r0 + 1,
                "center_c": c0 + 1,
            }
        )

    ordered = sorted(specs, key=lambda spec: (spec["count"], spec["center_r"], spec["center_c"]))
    height = max(spec["count"] for spec in ordered)
    out = np.full((height, len(ordered)), bg, dtype=int)
    for col, spec in enumerate(ordered):
        out[height - spec["count"] :, col] = spec["color"]
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
