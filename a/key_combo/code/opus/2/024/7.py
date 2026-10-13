from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, bbox_coords, neighbors4
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def neighbors4(r: int, c: int):
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1


BG = Color.BLACK
NOISE = Color.BLUE
MARK = Color.RED
FILL = Color.GREEN

MAX_TRIES = 300
GRID_MIN = 20
GRID_MAX = 26
CORRIDOR_COUNT_CHOICES = (2, 3)
CORRIDOR_THICKNESS_CHOICES = (3, 4)
MARKED_THICKNESS = 4
MIN_CORE_SPAN = 8
DISTRACTOR_COUNT_CHOICES = (1,)
CLUTTER_COUNT_RANGE = (4, 8)


def _rect_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _expand(box: tuple[int, int, int, int], delta: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return int(r0 - delta), int(c0 - delta), int(r1 + delta), int(c1 + delta)


def _segments(line: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(line.astype(bool).tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            out.append((start, idx - 1))
            start = None
    if start is not None:
        out.append((start, len(line) - 1))
    return out


def _local_safe(grid: np.ndarray, r: int, c: int) -> bool:
    r0 = max(0, int(r) - 1)
    c0 = max(0, int(c) - 1)
    r1 = min(grid.shape[0] - 1, int(r) + 1)
    c1 = min(grid.shape[1] - 1, int(c) + 1)
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all((block == BG) | (block == MARK)))


def _candidate_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe = np.zeros_like(grid, dtype=bool)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) in (BG, MARK):
                safe[r, c] = _local_safe(grid, r, c)
    candidate_black = (grid == BG) & safe
    support = candidate_black | ((grid == MARK) & safe)
    return candidate_black, support


def _scan_core_rectangles(support: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = support.shape
    rects: set[tuple[int, int, int, int]] = set()

    for band_h in (1, 2):
        if band_h > h:
            continue
        for r0 in range(h - band_h + 1):
            band = np.all(support[r0 : r0 + band_h, :], axis=0)
            for c0, c1 in _segments(band):
                if r0 > 0 and bool(np.all(support[r0 - 1, c0 : c1 + 1])):
                    continue
                if r0 + band_h < h and bool(np.all(support[r0 + band_h, c0 : c1 + 1])):
                    continue
                rects.add((r0, c0, r0 + band_h - 1, c1))

    for band_w in (1, 2):
        if band_w > w:
            continue
        for c0 in range(w - band_w + 1):
            band = np.all(support[:, c0 : c0 + band_w], axis=1)
            for r0, r1 in _segments(band):
                if c0 > 0 and bool(np.all(support[r0 : r1 + 1, c0 - 1])):
                    continue
                if c0 + band_w < w and bool(np.all(support[r0 : r1 + 1, c0 + band_w])):
                    continue
                rects.add((r0, c0, r1, c0 + band_w - 1))

    return sorted(rects, key=lambda box: (-_rect_area(box), box))


def _orientation_and_span(box: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> tuple[str, int] | None:
    r0, c0, r1, c1 = box
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    grid_h, grid_w = map(int, grid_shape)
    if h <= 2 and w > h:
        return "H", grid_w
    if w <= 2 and h > w:
        return "V", grid_h
    return None


def _has_blue_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = grid.shape
    if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
        return False
    top = grid[r0 - 1, c0 - 1 : c1 + 2]
    bottom = grid[r1 + 1, c0 - 1 : c1 + 2]
    left = grid[r0 - 1 : r1 + 2, c0 - 1]
    right = grid[r0 - 1 : r1 + 2, c1 + 1]
    return bool(
        np.all(top == NOISE)
        and np.all(bottom == NOISE)
        and np.all(left == NOISE)
        and np.all(right == NOISE)
    )


def _connected_component_area(mask: np.ndarray, start: tuple[int, int]) -> int:
    if not mask[start]:
        return 0
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen)


def _detect_corridors(grid: np.ndarray) -> list[dict]:
    grid = np.asarray(grid, dtype=int)
    _, support = _candidate_masks(grid)
    core_rects = _scan_core_rectangles(support)
    accepted_core = np.zeros_like(grid, dtype=bool)
    seen_actual: set[tuple[int, int, int, int]] = set()
    corridors: list[dict] = []

    for core in core_rects:
        orient_info = _orientation_and_span(core, grid.shape)
        if orient_info is None:
            continue
        orientation, span_dim = orient_info
        r0, c0, r1, c1 = core
        core_long = max(r1 - r0 + 1, c1 - c0 + 1)
        if core_long < max(MIN_CORE_SPAN, math.ceil(span_dim / 2)):
            continue

        actual = _expand(core, 1)
        ar0, ac0, ar1, ac1 = actual
        if ar0 < 0 or ac0 < 0 or ar1 >= grid.shape[0] or ac1 >= grid.shape[1]:
            continue
        patch = grid[ar0 : ar1 + 1, ac0 : ac1 + 1]
        if not bool(np.all((patch == BG) | (patch == MARK))):
            continue
        if not _has_blue_frame(grid, actual):
            continue

        core_mask = np.zeros_like(grid, dtype=bool)
        core_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        new_mask = core_mask & ~accepted_core
        if not new_mask.any():
            continue
        seed = tuple(map(int, np.argwhere(new_mask)[0]))
        new_area = _connected_component_area(new_mask, seed)
        if new_area < math.ceil(span_dim / 3):
            continue
        if actual in seen_actual:
            continue

        marker_coords = np.argwhere(grid[ar0 : ar1 + 1, ac0 : ac1 + 1] == MARK)
        marker_bbox = None
        if marker_coords.size:
            marker_coords = marker_coords.copy()
            marker_coords[:, 0] += ar0
            marker_coords[:, 1] += ac0
            marker_bbox = bbox_coords(marker_coords)

        corridors.append(
            {
                "bbox": actual,
                "core_bbox": core,
                "orientation": orientation,
                "marker_bbox": marker_bbox,
            }
        )
        accepted_core |= core_mask
        seen_actual.add(actual)

    return sorted(corridors, key=lambda item: (item["bbox"][0], item["bbox"][1], item["orientation"]))


def _strict_interior(box: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    return int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)


def _render_from_corridors(grid: np.ndarray, corridors: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for corridor in corridors:
        r0, c0, r1, c1 = corridor["bbox"]
        protected = _strict_interior(corridor["marker_bbox"])
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(out[r, c]) != BG:
                    continue
                if protected is not None:
                    pr0, pc0, pr1, pc1 = protected
                    if pr0 <= r <= pr1 and pc0 <= c <= pc1:
                        continue
                out[r, c] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect long black corridors inside blue clutter and fill them green while preserving marker-defined windows.

    Input structure:
    - The background is black, with blue foreground clutter forming irregular blocking walls.
    - Hidden inside the black negative space are 2-3 long thin axis-aligned corridor rectangles.
    - Some corridors contain red marker pixels whose bbox defines a protected inner window; green is absent.

    Rule:
    - Mark black corridor-core pixels whose cropped 3x3 neighborhood contains only black or red cells.
    - Recover maximal thin core rectangles from those candidate pixels, keep the long ones, and expand each core by one cell of black/red margin to obtain the full corridor.
    - For each accepted corridor, paint every black cell green except the strict interior of the red markers' bbox; keep red markers, blue clutter, and all non-corridor black cells unchanged.

    Key insight:
    - The corridor is recognized as negative space by a local blue-free test, but the final fill is clipped by a second structure: the rectangle implied by the markers inside that same corridor.
    """
    corridors = _detect_corridors(np.asarray(grid, dtype=int))
    return _render_from_corridors(np.asarray(grid, dtype=int), corridors)


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    outer = _expand(box, 1)
    or0, oc0, or1, oc1 = outer
    grid[or0, oc0 : oc1 + 1] = NOISE
    grid[or1, oc0 : oc1 + 1] = NOISE
    grid[or0 : or1 + 1, oc0] = NOISE
    grid[or0 : or1 + 1, oc1] = NOISE


def _reserve_ok(mask: np.ndarray, box: tuple[int, int, int, int], padding: int = 0) -> bool:
    r0, c0, r1, c1 = _expand(box, padding)
    if r0 < 0 or c0 < 0 or r1 >= mask.shape[0] or c1 >= mask.shape[1]:
        return False
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_reserved(mask: np.ndarray, box: tuple[int, int, int, int], padding: int = 0) -> None:
    r0, c0, r1, c1 = _expand(box, padding)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _rand_length(span_dim: int, distractor: bool) -> int:
    if distractor:
        hi = min(max(7, math.ceil(span_dim / 2) + 1), 9)
        return random.randint(6, hi)
    lo = max(MIN_CORE_SPAN + 2, math.ceil(span_dim / 2) + 2)
    hi = min(span_dim - 8, lo + 4)
    if lo > hi:
        return lo
    return random.randint(lo, hi)


def _place_corridor(
    grid: np.ndarray,
    reserved: np.ndarray,
    orientation: str,
    *,
    marked: bool,
    distractor: bool,
) -> dict | None:
    h, w = grid.shape
    span_dim = w if orientation == "H" else h
    thickness = MARKED_THICKNESS if marked else random.choice(CORRIDOR_THICKNESS_CHOICES)
    length = _rand_length(span_dim, distractor=distractor)

    if orientation == "H":
        actual_h, actual_w = thickness, length
        r_choices = range(2, h - actual_h - 1)
        c_choices = range(2, w - actual_w - 1)
    else:
        actual_h, actual_w = length, thickness
        r_choices = range(2, h - actual_h - 1)
        c_choices = range(2, w - actual_w - 1)

    if not r_choices or not c_choices:
        return None

    for _ in range(120):
        r0 = random.choice(list(r_choices))
        c0 = random.choice(list(c_choices))
        box = (r0, c0, r0 + actual_h - 1, c0 + actual_w - 1)
        outer = _expand(box, 1)
        if not _reserve_ok(reserved, outer, padding=0):
            continue

        _paint_frame(grid, box)
        _mark_reserved(reserved, outer, padding=0)
        spec = {"bbox": box, "orientation": orientation, "marker_bbox": None, "distractor": distractor}

        if marked:
            if orientation == "H":
                mr0 = r0 + 1
                mr1 = r0 + 3
                mw = random.randint(4, actual_w - 1)
                mc0 = random.randint(c0 + 1, c0 + actual_w - mw)
                mc1 = mc0 + mw - 1
            else:
                mc0 = c0 + 1
                mc1 = c0 + 3
                mh = random.randint(4, actual_h - 1)
                mr0 = random.randint(r0 + 1, r0 + actual_h - mh)
                mr1 = mr0 + mh - 1

            corners = [(mr0, mc0), (mr0, mc1), (mr1, mc0), (mr1, mc1)]
            if random.random() < 0.5:
                corners = [corners[0], corners[3]]
            for rr, cc in corners:
                grid[rr, cc] = MARK
            spec["marker_bbox"] = (mr0, mc0, mr1, mc1)

        return spec

    return None


def _add_frame_protrusions(grid: np.ndarray, reserved: np.ndarray, box: tuple[int, int, int, int], orientation: str) -> None:
    r0, c0, r1, c1 = box
    candidates: list[tuple[int, int, int, int]] = []
    if orientation == "H":
        for col in range(c0 + 1, c1, 2):
            length = random.randint(1, 3)
            candidates.append((r0 - 3, col, r0 - 1, col))
            candidates.append((r1 + 1, col, r1 + length, col))
    else:
        for row in range(r0 + 1, r1, 2):
            length = random.randint(1, 3)
            candidates.append((row, c0 - 3, row, c0 - 1))
            candidates.append((row, c1 + 1, row, c1 + length))

    random.shuffle(candidates)
    for extra in candidates[: random.randint(1, min(3, len(candidates)))]:
        er0, ec0, er1, ec1 = extra
        er0 = max(0, er0)
        ec0 = max(0, ec0)
        er1 = min(grid.shape[0] - 1, er1)
        ec1 = min(grid.shape[1] - 1, ec1)
        if er0 > er1 or ec0 > ec1:
            continue
        if not _reserve_ok(reserved, (er0, ec0, er1, ec1), padding=0):
            continue
        grid[er0 : er1 + 1, ec0 : ec1 + 1] = NOISE
        _mark_reserved(reserved, (er0, ec0, er1, ec1), padding=0)


def _add_clutter(grid: np.ndarray, reserved: np.ndarray) -> None:
    for _ in range(random.randint(*CLUTTER_COUNT_RANGE)):
        hh = random.randint(1, 3)
        ww = random.randint(1, 4)
        r0 = random.randint(0, grid.shape[0] - hh)
        c0 = random.randint(0, grid.shape[1] - ww)
        box = (r0, c0, r0 + hh - 1, c0 + ww - 1)
        if not _reserve_ok(reserved, box, padding=0):
            continue
        grid[r0 : r0 + hh, c0 : c0 + ww] = NOISE
        _mark_reserved(reserved, box, padding=0)
        if hh > 1 and ww > 1 and random.random() < 0.5:
            grid[r0 + random.randrange(hh), c0 + random.randrange(ww)] = BG


def _expected_from_specs(grid: np.ndarray, corridors: list[dict]) -> np.ndarray:
    active = [
        {
            "bbox": spec["bbox"],
            "orientation": spec["orientation"],
            "marker_bbox": spec["marker_bbox"],
        }
        for spec in corridors
        if not spec["distractor"]
    ]
    return _render_from_corridors(grid, active)


def _validate_instance(grid: np.ndarray, specs: list[dict]) -> bool:
    real_specs = [spec for spec in specs if not spec["distractor"]]
    marked_specs = [spec for spec in real_specs if spec["marker_bbox"] is not None]
    unmarked_specs = [spec for spec in real_specs if spec["marker_bbox"] is None]
    if not marked_specs or not unmarked_specs:
        return False
    if int(np.max(grid.shape)) > 30 or int(np.min(grid.shape)) < 1:
        return False
    if np.any(grid == FILL):
        return False

    detected = _detect_corridors(grid)
    detected_boxes = sorted(c["bbox"] for c in detected)
    spec_boxes = sorted(spec["bbox"] for spec in real_specs)
    if detected_boxes != spec_boxes:
        return False

    for corridor, spec in zip(sorted(detected, key=lambda x: x["bbox"]), sorted(real_specs, key=lambda x: x["bbox"])):
        if corridor["marker_bbox"] != spec["marker_bbox"]:
            return False
        if spec["marker_bbox"] is not None:
            interior = _strict_interior(spec["marker_bbox"])
            if interior is None:
                return False
            pr0, pc0, pr1, pc1 = interior
            if not bool(np.all(grid[pr0 : pr1 + 1, pc0 : pc1 + 1] == BG)):
                return False

    expected = _expected_from_specs(grid, specs)
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    corridor_count = random.choice(CORRIDOR_COUNT_CHOICES if min(h, w) >= 24 else (2,))
    if corridor_count == 2:
        orientations = ["H", "V"]
    else:
        orientations = ["H", "V", random.choice(["H", "V"])]
    random.shuffle(orientations)

    marked_count = random.randint(1, corridor_count - 1)
    marked_flags = [True] * marked_count + [False] * (corridor_count - marked_count)
    random.shuffle(marked_flags)

    specs: list[dict] = []
    for orientation, marked in zip(orientations, marked_flags):
        spec = _place_corridor(grid, reserved, orientation, marked=marked, distractor=False)
        if spec is None:
            return None
        specs.append(spec)

    distractor_orients = [random.choice(["H", "V"]) for _ in range(random.choice(DISTRACTOR_COUNT_CHOICES))]
    for orientation in distractor_orients:
        spec = _place_corridor(grid, reserved, orientation, marked=False, distractor=True)
        if spec is None:
            return None
        specs.append(spec)

    _add_clutter(grid, reserved)

    if not _validate_instance(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-window instance")
