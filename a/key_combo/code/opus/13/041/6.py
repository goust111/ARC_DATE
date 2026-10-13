from __future__ import annotations

import itertools
import random

import numpy as np


_DIRS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]

_BG = 0
_ANCHOR_PALETTE = (1, 2, 5, 9)
_RAY_PALETTE = (3, 4, 6, 7, 8)
_GEN_RETRIES = 400
_SAFE_DIRS_BY_TAG = {
    "TL": [(-1, 0), (0, -1), (-1, -1)],
    "TR": [(-1, 0), (0, 1), (-1, 1)],
    "BL": [(1, 0), (0, -1), (1, -1)],
    "BR": [(1, 0), (0, 1), (1, 1)],
}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_anchors(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    anchors: list[dict[str, object]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            block = grid[r : r + 3, c : c + 3]
            color = int(block[0, 0])
            if color == bg or not np.all(block == color):
                continue
            cells = {(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3)}
            if any(cells & anchor["cells"] for anchor in anchors):
                continue
            anchors.append(
                {
                    "color": color,
                    "r0": r,
                    "c0": c,
                    "center": (r + 1, c + 1),
                    "cells": cells,
                }
            )
    return anchors


def _ray_cell(anchor: dict[str, object], direction: tuple[int, int], step: int) -> tuple[int, int]:
    cr, cc = anchor["center"]
    dr, dc = direction
    return int(cr + (step + 1) * dr), int(cc + (step + 1) * dc)


def _in_bounds(shape: tuple[int, int], rc: tuple[int, int]) -> bool:
    r, c = rc
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _scan_path(
    anchor: dict[str, object],
    direction: tuple[int, int],
    shape: tuple[int, int],
    other_anchor_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    step = 1
    while True:
        rc = _ray_cell(anchor, direction, step)
        if not _in_bounds(shape, rc) or rc in other_anchor_cells:
            break
        path.append(rc)
        step += 1
    return path


def _infer_period(observations: list[tuple[int, int]]) -> tuple[int, list[int]]:
    assert observations
    steps = [step for step, _color in observations]
    max_step = max(steps)
    for period in range(1, max_step + 1):
        phase_to_color: dict[int, int] = {}
        ok = True
        for step, color in observations:
            phase = (step - 1) % period
            if phase in phase_to_color and phase_to_color[phase] != color:
                ok = False
                break
            phase_to_color[phase] = color
        if ok and len(phase_to_color) == period:
            return period, [phase_to_color[i] for i in range(period)]
    raise AssertionError("ray period is not uniquely recoverable")


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild each sparse periodic ray and pack the full color sequence against its anchor block.

    Input structure:
    - A mostly black grid contains exactly two solid 3x3 monochrome anchor blocks in different colors.
    - From each anchor, 2-4 rays extend in distinct 8-neighbor directions using colors different from the anchor colors.
    - Along each ray, some positions of an underlying period-2 or period-3 color sequence are erased to background, so visible samples can start after a gap and can have internal holes.

    Rule:
    - Detect the 3x3 anchors and, for every anchor-direction line, collect all visible non-background pixels together with their step distance from the anchor.
    - For each line with at least two visible pixels, infer the shortest periodic color order consistent with the observed step/color pairs.
    - Remove every non-anchor foreground pixel, then repaint the complete sequence from step 1 through the farthest observed step contiguously from the anchor outward along the same direction.

    Key insight:
    - The sparse ray pixels are incomplete samples of a short periodic sequence, so the task is to recover missing phases first and only then collapse the full reconstructed prefix back against the anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    anchors = _find_anchors(g, bg)
    assert len(anchors) == 2

    anchor_cells = set().union(*(anchor["cells"] for anchor in anchors))
    out = np.full_like(g, bg)
    for anchor in anchors:
        for r, c in anchor["cells"]:
            out[r, c] = int(anchor["color"])

    for idx, anchor in enumerate(anchors):
        other_cells = set().union(*(anchors[j]["cells"] for j in range(len(anchors)) if j != idx))
        for direction in _DIRS:
            path = _scan_path(anchor, direction, g.shape, other_cells)
            observations = [
                (step, int(g[r, c]))
                for step, (r, c) in enumerate(path, start=1)
                if int(g[r, c]) != bg
            ]
            if len(observations) < 2:
                continue
            period, colors = _infer_period(observations)
            farthest = max(step for step, _color in observations)
            for step in range(1, farthest + 1):
                r, c = path[step - 1]
                out[r, c] = colors[(step - 1) % period]

    return out


def _sample_anchor_positions(height: int, width: int) -> list[tuple[str, tuple[int, int]]]:
    layouts = [
        ("TL", "BR"),
        ("TR", "BL"),
    ]
    a_tag, b_tag = random.choice(layouts)
    top_rows = (8, 8)
    bottom_rows = (height - 11, height - 11)
    left_cols = (8, 8)
    right_cols = (width - 11, width - 11)

    def pick(tag: str) -> tuple[int, int]:
        if tag == "TL":
            return (
                random.randint(*top_rows),
                random.randint(*left_cols),
            )
        if tag == "TR":
            return (
                random.randint(*top_rows),
                random.randint(*right_cols),
            )
        if tag == "BL":
            return (
                random.randint(*bottom_rows),
                random.randint(*left_cols),
            )
        return (
            random.randint(*bottom_rows),
            random.randint(*right_cols),
        )

    first = pick(a_tag)
    second = pick(b_tag)
    return [(a_tag, first), (b_tag, second)]


def _candidate_rays(
    anchor: dict[str, object],
    shape: tuple[int, int],
    other_anchor_cells: set[tuple[int, int]],
) -> list[dict[str, object]]:
    rays: list[dict[str, object]] = []
    for direction in _SAFE_DIRS_BY_TAG[anchor["tag"]]:
        path = _scan_path(anchor, direction, shape, other_anchor_cells)
        if len(path) < 8:
            continue
        rays.append({"direction": direction, "path": path, "path_set": set(path)})
    return rays


def _pick_ray_set(
    anchors: list[dict[str, object]],
    shape: tuple[int, int],
) -> list[tuple[dict[str, object], list[dict[str, object]]]]:
    bundles: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    all_cells = [anchor["cells"] for anchor in anchors]
    all_scan_sets: list[list[set[tuple[int, int]]]] = []
    for idx, anchor in enumerate(anchors):
        other_anchor_cells = set().union(*(all_cells[j] for j in range(len(anchors)) if j != idx))
        candidates = _candidate_rays(anchor, shape, other_anchor_cells)
        scan_sets = [set(_scan_path(anchor, direction, shape, other_anchor_cells)) for direction in _DIRS]
        if len(candidates) < 2:
            raise ValueError("not enough ray directions")
        bundles.append((anchor, candidates))
        all_scan_sets.append(scan_sets)

    safe_bundles: list[tuple[dict[str, object], list[dict[str, object]]]] = []
    for idx, (anchor, candidates) in enumerate(bundles):
        other_scan_union = set().union(*all_scan_sets[1 - idx])
        safe = [cand for cand in candidates if not (cand["path_set"] & other_scan_union)]
        if len(safe) < 2:
            raise ValueError("rays intrude on foreign scan lines")
        safe_bundles.append((anchor, safe))

    subset_options: list[list[tuple[dict[str, object], ...]]] = []
    for _anchor, candidates in safe_bundles:
        options: list[tuple[dict[str, object], ...]] = []
        for size in range(2, min(4, len(candidates)) + 1):
            options.extend(itertools.combinations(candidates, size))
        random.shuffle(options)
        subset_options.append(options)

    valid_pairs: list[tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]] = []
    for left in subset_options[0]:
        left_union = set().union(*(ray["path_set"] for ray in left))
        for right in subset_options[1]:
            right_union = set().union(*(ray["path_set"] for ray in right))
            if left_union & right_union:
                continue
            valid_pairs.append((left, right))
    if not valid_pairs:
        raise ValueError("ray lines overlap")

    left, right = random.choice(valid_pairs)
    return [
        (safe_bundles[0][0], [dict(ray) for ray in left]),
        (safe_bundles[1][0], [dict(ray) for ray in right]),
    ]


def _all_missing_choices(length: int, period: int, missing_count: int) -> list[tuple[int, ...]]:
    valid: list[tuple[int, ...]] = []
    for miss in itertools.combinations(range(1, length + 1), missing_count):
        visible = [step for step in range(1, length + 1) if step not in miss]
        if len(visible) < 2 * period:
            continue
        if {((step - 1) % period) for step in visible} != set(range(period)):
            continue
        valid.append(miss)
    return valid


def _build_ray(
    path: list[tuple[int, int]],
    forced_period: int | None = None,
    missing_mode: str = "random",
) -> dict[str, object]:
    period = forced_period if forced_period is not None else random.choice((2, 3))
    missing_count = random.choice((1, 2))
    min_length = 2 * period + missing_count
    max_length = min(len(path), 9)
    if max_length < min_length:
        raise ValueError("ray too short for visible cycles")
    length = random.randint(min_length, max_length)
    colors = random.sample(_RAY_PALETTE, period)
    missing_choices = _all_missing_choices(length, period, missing_count)
    if missing_mode == "leading":
        missing_choices = [miss for miss in missing_choices if 1 in miss]
    elif missing_mode == "internal":
        missing_choices = [
            miss for miss in missing_choices if any(1 < step < length for step in miss)
        ]
    if not missing_choices:
        raise ValueError("no non-degenerate missing set")
    missing = set(random.choice(missing_choices))
    return {
        "period": period,
        "length": length,
        "colors": colors,
        "missing": missing,
    }


def _render_case(
    shape: tuple[int, int],
    anchors: list[dict[str, object]],
    ray_choices: list[tuple[dict[str, object], list[dict[str, object]]]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    inp = np.full(shape, _BG, dtype=int)
    out = np.full(shape, _BG, dtype=int)
    specs: list[dict[str, object]] = []

    for anchor in anchors:
        r0 = int(anchor["r0"])
        c0 = int(anchor["c0"])
        color = int(anchor["color"])
        inp[r0 : r0 + 3, c0 : c0 + 3] = color
        out[r0 : r0 + 3, c0 : c0 + 3] = color

    all_rays = [ray for _anchor, rays in ray_choices for ray in rays]
    random.shuffle(all_rays)
    forced_modes = [("leading", 2), ("internal", 3)]

    for ray, (missing_mode, forced_period) in zip(all_rays, forced_modes):
        ray["spec"] = _build_ray(ray["path"], forced_period=forced_period, missing_mode=missing_mode)

    for ray in all_rays[len(forced_modes) :]:
        ray["spec"] = _build_ray(ray["path"])

    for _anchor, rays in ray_choices:
        random.shuffle(rays)
        for ray in rays:
            spec = ray["spec"]
            specs.append(spec)
            for step in range(1, int(spec["length"]) + 1):
                r, c = ray["path"][step - 1]
                color = int(spec["colors"][(step - 1) % int(spec["period"])])
                out[r, c] = color
                if step not in spec["missing"]:
                    inp[r, c] = color
    return inp, out, specs


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    ray_choices: list[tuple[dict[str, object], list[dict[str, object]]]],
) -> bool:
    if np.array_equal(inp, out):
        return False
    if not np.array_equal(main(inp), out):
        return False

    anchors = _find_anchors(inp, _BG)
    if len(anchors) != 2:
        return False

    saw_period2 = False
    saw_period3 = False
    saw_internal_gap = False
    saw_leading_gap = False

    for _anchor, rays in ray_choices:
        if not (2 <= len(rays) <= 4):
            return False
        for ray in rays:
            spec = ray["spec"]
            observations = []
            for step in range(1, int(spec["length"]) + 1):
                if step in spec["missing"]:
                    continue
                color = int(spec["colors"][(step - 1) % int(spec["period"])])
                observations.append((step, color))
            period, colors = _infer_period(observations)
            if period != int(spec["period"]):
                return False
            if colors != list(spec["colors"]):
                return False
            if len(observations) < 2 * period:
                return False
            saw_period2 |= period == 2
            saw_period3 |= period == 3
            saw_leading_gap |= 1 in spec["missing"]
            saw_internal_gap |= any(1 < step < int(spec["length"]) for step in spec["missing"])

    return saw_period2 and saw_period3 and saw_leading_gap and saw_internal_gap


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(22, 24)
    width = random.randint(22, 24)
    anchors_pos = _sample_anchor_positions(height, width)
    anchor_colors = random.sample(_ANCHOR_PALETTE, 2)
    anchors: list[dict[str, object]] = []
    for (tag, (r0, c0)), color in zip(anchors_pos, anchor_colors):
        anchors.append(
            {
                "tag": tag,
                "color": color,
                "r0": r0,
                "c0": c0,
                "center": (r0 + 1, c0 + 1),
                "cells": {(rr, cc) for rr in range(r0, r0 + 3) for cc in range(c0, c0 + 3)},
            }
        )

    ray_choices = _pick_ray_set(anchors, (height, width))
    inp, out, _specs = _render_case((height, width), anchors, ray_choices)
    if not _validate_case(inp, out, ray_choices):
        raise ValueError("semantic validation failed")
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        try:
            inp, _out = _generate_case()
            return inp
        except ValueError:
            continue
    raise RuntimeError("failed to generate a valid periodic-anchor ARC input")
