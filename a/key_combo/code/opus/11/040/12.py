from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
LINE_COLORS = (BLUE, RED, GREEN)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ROTATION_BY_FACING = {
    (-1, 0): 0,   # up
    (0, 1): 3,    # right
    (1, 0): 2,    # down
    (0, -1): 1,   # left
}
GEN_MAX_ATTEMPTS = 400
GEN_HEIGHT_RANGE = (13, 18)
GEN_WIDTH_RANGE = (13, 18)
GEN_SEGMENT_COUNT_RANGE = (2, 4)
GEN_SEGMENT_LENGTH_RANGE = (4, 6)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _centered_patch(mask: np.ndarray, center: tuple[int, int], color: int) -> set[tuple[int, int]]:
    h, w = mask.shape
    cr, cc = h // 2, w // 2
    base_r, base_c = center
    pts = set()
    for rr in range(h):
        for cc2 in range(w):
            if mask[rr, cc2]:
                pts.add((base_r + rr - cr, base_c + cc2 - cc))
    return pts


def _mask_is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask.astype(bool))) == 1


def _build_template_library() -> list[np.ndarray]:
    lib: list[np.ndarray] = []
    for bits in range(1 << 8):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        idx = 0
        for r in range(3):
            for c in range(3):
                if (r, c) == (1, 1):
                    continue
                if (bits >> idx) & 1:
                    mask[r, c] = True
                idx += 1
        if not _mask_is_connected(mask):
            continue
        if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        rotations = [np.rot90(mask, k) for k in range(4)]
        if len({rot.tobytes() for rot in rotations}) != 4:
            continue
        bottom_rows = {tuple(rot[2].astype(int).tolist()) for rot in rotations}
        if len(bottom_rows) < 2:
            continue
        lib.append(mask.copy())
    assert lib
    return lib


TEMPLATE_LIBRARY = _build_template_library()


def _extract_template(grid: np.ndarray, baseline_r: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    yellow_mask = grid == YELLOW
    comps = _components(yellow_mask)
    assert len(comps) == 1, "expected exactly one template component"
    coords = comps[0]
    assert all(r > baseline_r for r, _ in coords), "template must be below the baseline"
    r0, c0, r1, c1 = _bbox(coords)
    patch = yellow_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[0] % 2 == 1 and patch.shape[1] % 2 == 1
    assert patch[patch.shape[0] // 2, patch.shape[1] // 2]
    return patch, coords


def _baseline_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey baseline"
    return rows[0]


def _line_components(grid: np.ndarray, baseline_r: int) -> list[dict]:
    comps: list[dict] = []
    for color in LINE_COLORS:
        mask = np.zeros_like(grid, dtype=bool)
        mask[:baseline_r, :] = grid[:baseline_r, :] == color
        for coords in _components(mask):
            pts = set(coords)
            endpoints: list[tuple[int, int]] = []
            for r, c in coords:
                nbrs = [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pts]
                degree = len(nbrs)
                assert degree in (1, 2), "segments must be simple one-pixel-wide chains"
                if degree == 1:
                    endpoints.append((r, c))
            assert 1 <= len(endpoints) <= 2
            comps.append({"color": color, "cells": pts, "endpoints": endpoints})
    assert 2 <= len(comps) <= 4
    return comps


def _stamp(
    out: np.ndarray,
    patch: np.ndarray,
    center: tuple[int, int],
    color: int,
) -> set[tuple[int, int]]:
    painted: set[tuple[int, int]] = set()
    h, w = patch.shape
    cr, cc = h // 2, w // 2
    base_r, base_c = center
    for rr in range(h):
        for cc2 in range(w):
            if not patch[rr, cc2]:
                continue
            tr = base_r + rr - cr
            tc = base_c + cc2 - cc
            if out[tr, tc] in (BLACK, GREY):
                out[tr, tc] = color
                painted.add((tr, tc))
    return painted


def _transform(grid: np.ndarray, *, rotate_by_facing: bool = True) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    baseline_r = _baseline_row(g)
    template_mask, template_cells = _extract_template(g, baseline_r)
    lines = _line_components(g, baseline_r)

    out = g.copy()
    baseline_hits_by_color: Counter[int] = Counter()
    baseline_adjacent_endpoint_counts: Counter[int] = Counter()
    directions: list[tuple[int, int]] = []
    endpoints_info: list[dict] = []

    for comp in lines:
        color = int(comp["color"])
        cells = comp["cells"]
        for r, c in comp["endpoints"]:
            nbrs = [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in cells]
            assert len(nbrs) == 1
            nr, nc = nbrs[0]
            facing = (r - nr, c - nc)
            directions.append(facing)
            if r == baseline_r - 1:
                baseline_adjacent_endpoint_counts[color] += 1
            k = ROTATION_BY_FACING[facing] if rotate_by_facing else 0
            rotated = np.rot90(template_mask, k)
            painted = _stamp(out, rotated, (r, c), color)
            hits = sum(1 for rr, _ in painted if rr == baseline_r)
            if hits:
                baseline_hits_by_color[color] += hits
            endpoints_info.append(
                {"center": (r, c), "color": color, "facing": facing, "baseline_hits": hits}
            )

    baseline_before_fill = out[baseline_r].copy()
    counts = Counter(int(v) for v in baseline_before_fill if v not in (BLACK, GREY))
    assert len(counts) >= 1
    winner_color, winner_count = counts.most_common(1)[0]
    assert sum(1 for count in counts.values() if count == winner_count) == 1, "baseline majority must be unique"
    out[baseline_r, :] = winner_color
    for r, c in template_cells:
        out[r, c] = BLACK

    info = {
        "baseline_r": baseline_r,
        "baseline_before_fill": baseline_before_fill,
        "baseline_counts": dict(counts),
        "baseline_hits_by_color": dict(baseline_hits_by_color),
        "baseline_adjacent_endpoint_counts": dict(baseline_adjacent_endpoint_counts),
        "winner_color": int(winner_color),
        "directions": directions,
        "endpoints": endpoints_info,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate a template by each segment endpoint's facing, stamp it onto the endpoints, then recolor the whole baseline row by the stamped majority color.

    Input structure:
    - A black background contains one full-width grey baseline row in the lower part of the grid.
    - Above the baseline are 2-4 isolated one-pixel-wide horizontal or vertical colored line segments.
    - A single yellow asymmetric template sits below the baseline in the lower-left area; its odd-sized bbox has a clear center.

    Rule:
    - Find every open endpoint of every colored segment and infer its facing from the segment interior toward the outside.
    - Rotate the yellow template so its default upward orientation matches that facing, then stamp the rotated template centered on the endpoint using the segment's color.
    - Stamping is transparent onto black cells and may overwrite the grey baseline, but it does not replace pre-existing colored line or stamp pixels.
    - Count the non-grey stamped pixels on the baseline row, replace the entire baseline with the unique majority color, and erase the original yellow template.

    Key insight:
    - The endpoint facing controls the template rotation, which changes where colored stamp pixels land on the baseline and therefore determines the baseline's final majority color.
    """
    out, _info = _transform(grid, rotate_by_facing=True)
    return out


def _segment_cells(spec: dict) -> list[tuple[int, int]]:
    if spec["kind"] == "h":
        return [(spec["r"], c) for c in range(spec["c0"], spec["c0"] + spec["length"])]
    return [(r, spec["c"]) for r in range(spec["r0"], spec["r0"] + spec["length"])]


def _segment_endpoints(spec: dict) -> list[tuple[int, int]]:
    cells = _segment_cells(spec)
    return [cells[0], cells[-1]]


def _boxes_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 2


def _choose_template() -> np.ndarray:
    return random.choice(TEMPLATE_LIBRARY).copy()


def _render_input(
    height: int,
    width: int,
    baseline_r: int,
    template_mask: np.ndarray,
    template_c0: int,
    segments: list[dict],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[baseline_r, :] = GREY
    r0 = baseline_r + 1
    for rr in range(template_mask.shape[0]):
        for cc in range(template_mask.shape[1]):
            if template_mask[rr, cc]:
                grid[r0 + rr, template_c0 + cc] = YELLOW
    for seg in segments:
        for r, c in _segment_cells(seg):
            grid[r, c] = seg["color"]
    return grid


def _segment_spec(
    pattern: str,
    height: int,
    width: int,
    baseline_r: int,
) -> dict | None:
    length = random.randint(*GEN_SEGMENT_LENGTH_RANGE)
    if pattern == "h_base":
        max_c0 = width - length - 1
        if max_c0 < 1:
            return None
        return {
            "kind": "h",
            "r": baseline_r - 1,
            "c0": random.randint(1, max_c0),
            "length": length,
        }
    if pattern == "v_down":
        top = baseline_r - length
        if top < 1:
            return None
        return {
            "kind": "v",
            "r0": top,
            "c": random.randint(1, width - 2),
            "length": length,
        }
    if pattern == "h_upper":
        max_r = baseline_r - 3
        max_c0 = width - length - 1
        if max_r < 1 or max_c0 < 1:
            return None
        return {
            "kind": "h",
            "r": random.randint(1, max_r),
            "c0": random.randint(1, max_c0),
            "length": length,
        }
    max_r0 = baseline_r - length - 1
    if max_r0 < 1:
        return None
    return {
        "kind": "v",
        "r0": random.randint(1, max_r0),
        "c": random.randint(1, width - 2),
        "length": length,
    }


def _candidate_ok(
    seg: dict,
    segments: list[dict],
    template_cells: set[tuple[int, int]],
) -> bool:
    new_cells = set(_segment_cells(seg))
    new_endpoints = _segment_endpoints(seg)

    if new_cells & template_cells:
        return False

    for r, c in new_cells:
        if r <= 0 or c <= 0:
            return False
        if (r, c) in template_cells:
            return False

    for old in segments:
        old_cells = set(_segment_cells(old))
        if new_cells & old_cells:
            return False
        old_endpoints = _segment_endpoints(old)
        for ep in new_endpoints:
            for oep in old_endpoints:
                if _boxes_overlap(ep, oep):
                    return False
        for ep in new_endpoints:
            for cell in old_cells:
                if abs(ep[0] - cell[0]) <= 1 and abs(ep[1] - cell[1]) <= 1:
                    return False
        for oep in old_endpoints:
            for cell in new_cells:
                if abs(oep[0] - cell[0]) <= 1 and abs(oep[1] - cell[1]) <= 1:
                    return False
    return True


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        out, info = _transform(grid, rotate_by_facing=True)
        alt, _ = _transform(grid, rotate_by_facing=False)
    except Exception:
        return False

    counts = info["baseline_counts"]
    if len(counts) < 2:
        return False
    max_count = max(counts.values())
    if sum(1 for v in counts.values() if v == max_count) != 1:
        return False
    if len(set(info["directions"])) < 2:
        return False
    if np.array_equal(out, alt):
        return False
    if not np.any(info["baseline_before_fill"] == GREY):
        return False

    endpoint_majority = info["baseline_adjacent_endpoint_counts"]
    hit_counts = info["baseline_hits_by_color"]
    if endpoint_majority and all(
        hit_counts.get(color, 0) == endpoint_majority.get(color, 0)
        for color in set(endpoint_majority) | set(hit_counts)
    ):
        return False

    if np.array_equal(out, grid):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    baseline_r = height - 4
    template_mask = _choose_template()
    template_c0 = random.randint(0, min(2, width - template_mask.shape[1]))
    template_cells = {
        (baseline_r + 1 + r, template_c0 + c)
        for r in range(template_mask.shape[0])
        for c in range(template_mask.shape[1])
        if template_mask[r, c]
    }

    segment_count = random.randint(*GEN_SEGMENT_COUNT_RANGE)
    base_patterns = ["h_base", "v_down"]
    extra_patterns = ["h_base", "v_down", "h_upper", "v_upper"]
    random.shuffle(base_patterns)
    pattern_plan = base_patterns[: min(2, segment_count)]
    while len(pattern_plan) < segment_count:
        pattern_plan.append(random.choice(extra_patterns))
    random.shuffle(pattern_plan)

    segments: list[dict] = []
    for pattern in pattern_plan:
        placed = False
        for _ in range(80):
            seg = _segment_spec(pattern, height, width, baseline_r)
            if seg is None:
                continue
            if _candidate_ok(seg, segments, template_cells):
                segments.append(seg)
                placed = True
                break
        if not placed:
            return None

    base_indices = [
        idx
        for idx, seg in enumerate(segments)
        if (seg["kind"] == "h" and seg["r"] == baseline_r - 1)
        or (seg["kind"] == "v" and seg["r0"] + seg["length"] - 1 == baseline_r - 1)
    ]
    if len(base_indices) < 2:
        return None

    chosen_base_colors = random.sample(LINE_COLORS, 2)
    segments[base_indices[0]]["color"] = chosen_base_colors[0]
    segments[base_indices[1]]["color"] = chosen_base_colors[1]
    for idx in base_indices[2:]:
        segments[idx]["color"] = random.choice(LINE_COLORS)
    for idx, seg in enumerate(segments):
        if idx not in base_indices:
            seg["color"] = random.choice(LINE_COLORS)

    base_colors = {segments[idx]["color"] for idx in base_indices}
    if len(base_colors) < 2:
        return None

    grid = _render_input(height, width, baseline_r, template_mask, template_c0, segments)
    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 12")
