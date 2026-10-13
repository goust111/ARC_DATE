from __future__ import annotations

import random

import numpy as np


BG = 0
# ARC standard visible color order:
# blue, red, green, yellow, grey, magenta, orange, azure, maroon.
COLOR_ORDER = (1, 2, 3, 4, 5, 6, 7, 8, 9)
COLOR_RANK = {color: rank for rank, color in enumerate(COLOR_ORDER)}

GRID_MIN = 18
GRID_MAX = 30
GROUP_COUNT_OPTIONS = (1, 2, 3)
DEPTH_OPTIONS = (2, 3, 4)
CORE_SIZE_OPTIONS = (2, 3)
GAP_OPTIONS = (1, 2)
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
GROUP_BUILD_RETRIES = 80
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400


def _components_by_color(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    h, w = _bbox_size(bbox)
    return h * w


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _is_frame(comp: dict) -> bool:
    bbox = comp["bbox"]
    h, w = _bbox_size(bbox)
    if h < 3 or w < 3:
        return False
    expected: set[tuple[int, int]] = set()
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return comp["coord_set"] == expected


def _is_solid_rectangle(comp: dict) -> bool:
    return comp["area"] == _bbox_area(comp["bbox"])


def _prefer_earlier(color_a: int, color_b: int) -> int:
    if COLOR_RANK[color_a] <= COLOR_RANK[color_b]:
        return color_a
    return color_b


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested hollow rectangular containers and propagate the core color outward.

    For each group:
    - find the solid core and all containing frame layers,
    - walk from the innermost frame to the outermost frame,
    - after each layer, keep the earlier color in the fixed palette order,
    - fill the blank ring between the current layer and the next inner object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    comps = _components_by_color(g)
    frames = [comp for comp in comps if _is_frame(comp)]
    cores = [comp for comp in comps if _is_solid_rectangle(comp) and not _is_frame(comp)]

    for core in sorted(cores, key=lambda comp: _bbox_area(comp["bbox"])):
        containing = [frame for frame in frames if _strictly_contains(frame["bbox"], core["bbox"])]
        if not containing:
            continue
        containing.sort(key=lambda frame: _bbox_area(frame["bbox"]))

        carry = int(core["color"])
        inner_bbox = core["bbox"]
        for frame in containing:
            carry = _prefer_earlier(int(frame["color"]), carry)
            r0, c0, r1, c1 = frame["bbox"]
            ir0, ic0, ir1, ic1 = inner_bbox

            fill_mask = np.zeros_like(out, dtype=bool)
            fill_mask[r0 + 1 : r1, c0 + 1 : c1] = True
            fill_mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
            out[fill_mask & (out == BG)] = carry

            inner_bbox = frame["bbox"]

    return out


def _build_group_patch() -> tuple[np.ndarray, list[str]]:
    for _ in range(GROUP_BUILD_RETRIES):
        depth = random.choice(DEPTH_OPTIONS)
        core_size = random.choice(CORE_SIZE_OPTIONS)
        core_color = random.choice(COLOR_ORDER)
        core_bbox = (0, 0, core_size - 1, core_size - 1)

        frames: list[dict] = []
        carry = core_color
        inner_bbox = core_bbox
        outcomes: list[str] = []

        for _layer in range(depth):
            top_gap = random.choice(GAP_OPTIONS)
            bottom_gap = random.choice(GAP_OPTIONS)
            left_gap = random.choice(GAP_OPTIONS)
            right_gap = random.choice(GAP_OPTIONS)

            r0, c0, r1, c1 = inner_bbox
            frame_bbox = (
                r0 - top_gap - 1,
                c0 - left_gap - 1,
                r1 + bottom_gap + 1,
                c1 + right_gap + 1,
            )

            frame_choices = [color for color in COLOR_ORDER if color != carry]
            frame_color = random.choice(frame_choices)
            outcomes.append("frame" if COLOR_RANK[frame_color] < COLOR_RANK[carry] else "carry")

            frames.append({"bbox": frame_bbox, "color": frame_color})
            carry = _prefer_earlier(frame_color, carry)
            inner_bbox = frame_bbox

        min_r = min([core_bbox[0]] + [frame["bbox"][0] for frame in frames])
        min_c = min([core_bbox[1]] + [frame["bbox"][1] for frame in frames])
        shifted_core = _shift_bbox(core_bbox, -min_r, -min_c)
        shifted_frames = [
            {"bbox": _shift_bbox(frame["bbox"], -min_r, -min_c), "color": int(frame["color"])}
            for frame in frames
        ]
        max_r = max([shifted_core[2]] + [frame["bbox"][2] for frame in shifted_frames])
        max_c = max([shifted_core[3]] + [frame["bbox"][3] for frame in shifted_frames])

        patch = np.zeros((max_r + 1, max_c + 1), dtype=int)
        for frame in reversed(shifted_frames):
            _draw_frame(patch, frame["bbox"], int(frame["color"]))
        cr0, cc0, cr1, cc1 = shifted_core
        patch[cr0 : cr1 + 1, cc0 : cc1 + 1] = core_color

        if np.any(main(patch) != patch):
            return patch, outcomes
    raise RuntimeError("failed to build a valid group patch")


def _rect_fits(
    top: int,
    left: int,
    height: int,
    width: int,
    placed: list[tuple[int, int, int, int]],
) -> bool:
    for prev_top, prev_left, prev_h, prev_w in placed:
        if top + height + PLACEMENT_GAP <= prev_top:
            continue
        if prev_top + prev_h + PLACEMENT_GAP <= top:
            continue
        if left + width + PLACEMENT_GAP <= prev_left:
            continue
        if prev_left + prev_w + PLACEMENT_GAP <= left:
            continue
        return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _generate_attempt() -> np.ndarray | None:
    group_count = random.choice(GROUP_COUNT_OPTIONS)
    groups = [_build_group_patch() for _ in range(group_count)]

    outcomes = [outcome for _patch, group_outcomes in groups for outcome in group_outcomes]
    if len(outcomes) >= 2 and not ("frame" in outcomes and "carry" in outcomes):
        return None

    patches = [patch for patch, _group_outcomes in groups]
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    sum_h = sum(int(patch.shape[0]) for patch in patches)
    sum_w = sum(int(patch.shape[1]) for patch in patches)

    h_lo = max(GRID_MIN, max_h + 2 * PLACEMENT_MARGIN, (sum_h + 1) // 2 + 2)
    w_lo = max(GRID_MIN, max_w + 2 * PLACEMENT_MARGIN, (sum_w + 1) // 2 + 2)
    if h_lo > GRID_MAX or w_lo > GRID_MAX:
        return None

    height = random.randint(h_lo, GRID_MAX)
    width = random.randint(w_lo, GRID_MAX)
    grid = np.zeros((height, width), dtype=int)

    placed: list[tuple[int, int, int, int]] = []
    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ph, pw = map(int, patch.shape)
        row_max = height - ph - PLACEMENT_MARGIN
        col_max = width - pw - PLACEMENT_MARGIN
        if row_max < PLACEMENT_MARGIN or col_max < PLACEMENT_MARGIN:
            return None

        success = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(PLACEMENT_MARGIN, row_max)
            left = random.randint(PLACEMENT_MARGIN, col_max)
            if not _rect_fits(top, left, ph, pw, placed):
                continue
            _paste_patch(grid, patch, top, left)
            placed.append((top, left, ph, pw))
            success = True
            break
        if not success:
            return None

    non_bg_colors = {int(value) for value in np.unique(grid) if int(value) != BG}
    if len(non_bg_colors) < 3:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
