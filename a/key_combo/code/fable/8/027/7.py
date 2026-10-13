from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

MAX_GEN_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _bbox_h(bbox: BBox) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: BBox) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _rect_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(bbox: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    return (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(h - 1, bbox[2] + pad),
        min(w - 1, bbox[3] + pad),
    )


def _strictly_inside(inner: BBox, outer: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _ring_patch(height: int, width: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _is_hollow_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    height = _bbox_h(component["bbox"])
    width = _bbox_w(component["bbox"])
    if height < 3 or width < 3:
        return False
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    mask = np.zeros((height, width), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    expected = np.ones((height, width), dtype=bool)
    expected[1:-1, 1:-1] = False
    return bool(np.array_equal(mask, expected))


def _grow_polyomino(area: int) -> set[Coord]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    return cells


def _normalize_cells(cells: set[Coord]) -> list[Coord]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _sprite_patch(area: int, color: int, bg: int) -> np.ndarray:
    if area == 1:
        return np.array([[color]], dtype=int)
    while True:
        cells = _normalize_cells(_grow_polyomino(area))
        height = max(r for r, _ in cells) + 1
        width = max(c for _, c in cells) + 1
        if height <= 4 and width <= 4:
            patch = np.full((height, width), bg, dtype=int)
            for r, c in cells:
                patch[r, c] = color
            return patch


def _partition_area(total: int, require_multi: bool) -> list[int]:
    if not require_multi:
        return [1] * total

    choices: list[list[int]] = []
    max_parts = min(3, total)
    for n_parts in range(1, max_parts + 1):
        parts = [1] * n_parts
        remain = total - n_parts
        while remain > 0:
            i = random.randrange(n_parts)
            parts[i] += 1
            remain -= 1
        if any(part > 1 for part in parts):
            choices.append(parts.copy())
    parts = random.choice(choices)
    random.shuffle(parts)
    return parts


def _dot_columns(interior_width: int, claim: int) -> list[int]:
    candidates = list(range(2, interior_width, 2))
    assert claim <= len(candidates)
    cols = sorted(random.sample(candidates, claim))
    return cols


def _choose_frame_sizes(frame_count: int) -> list[tuple[int, int]]:
    while True:
        sizes = []
        for _ in range(frame_count):
            height = random.choice([5, 6, 7, 8])
            width = random.choice([7, 9, 11])
            sizes.append((height, width))
        rows = max(h for h, _ in sizes) + 2
        cols = sum(w for _, w in sizes) + 2 * (frame_count - 1) + 2
        if rows <= 18 and cols <= 28:
            return sizes


def _choose_canvas(frame_count: int, sizes: list[tuple[int, int]]) -> tuple[int, int]:
    min_h = max(h for h, _ in sizes) + 4
    min_w = sum(w for _, w in sizes) + 2 * (frame_count - 1) + 4
    height = random.randint(min_h, min(30, max(min_h, 18 + frame_count)))
    width = random.randint(min_w, min(30, max(min_w, 20 + 2 * frame_count)))
    return height, width


def _place_frames(grid: np.ndarray, colors: list[int], sizes: list[tuple[int, int]], bg: int) -> list[dict] | None:
    height, width = grid.shape
    total_width = sum(w for _, w in sizes) + 2 * (len(sizes) - 1)
    slack = width - total_width - 2
    if slack < 0:
        return None
    base_left = 1 + random.randint(0, slack)

    frame_specs: list[dict] = []
    cursor = base_left
    for color, (fh, fw) in zip(colors, sizes):
        max_top = height - fh - 1
        if max_top < 1:
            return None
        top = random.randint(1, max_top)
        left = cursor
        cursor += fw + 2
        bbox = (top, left, top + fh - 1, left + fw - 1)
        frame_patch = _ring_patch(fh, fw, color, bg)
        paste(grid, frame_patch, top, left, transparent=bg)
        frame_coords = {
            (top, c) for c in range(left, left + fw)
        } | {
            (top + fh - 1, c) for c in range(left, left + fw)
        } | {
            (r, left) for r in range(top, top + fh)
        } | {
            (r, left + fw - 1) for r in range(top, top + fh)
        }
        frame_specs.append(
            {
                "color": color,
                "bbox": bbox,
                "frame_coords": frame_coords,
                "forbidden": _expand_bbox(bbox, 1, grid.shape),
            }
        )
    return frame_specs


def _place_patch_random(grid: np.ndarray, patch: np.ndarray, bg: int, forbidden: list[BBox]) -> tuple[int, int] | None:
    height, width = grid.shape
    ph, pw = patch.shape
    mask = patch != bg
    candidates: list[tuple[int, int]] = []
    for r0 in range(height - ph + 1):
        for c0 in range(width - pw + 1):
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_rect_overlap(bbox, bad) for bad in forbidden):
                continue
            view = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any((view != bg) & mask):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _expected_output(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        if spec["truth"]:
            for r, c in spec["pattern_coords"]:
                out[r, c] = bg
        else:
            for r, c in spec["frame_coords"] | spec["dot_coords"]:
                out[r, c] = bg
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Verify each claimed count inside a hollow frame against the total same-color sprite area, then keep the truthful side and erase the false side.

    Input structure:
    - The most common color is the background.
    - Several different colors each form one hollow rectangular frame, with same-color dot claims placed on that frame's horizontal midline interior.
    - Outside the frames, each frame color also appears in one or more disconnected monochrome sprites.

    Rule:
    - For each frame color C, count the same-color dots strictly inside that frame as the claimed total.
    - Sum the pixel area of all same-color sprites outside the frame's bbox to get the true total for color C.
    - If claim and true total match, keep the frame and its dots and erase the same-color outside sprites; otherwise erase the frame and dots and keep the outside sprites.

    Key insight:
    - The inside dots are not the output pattern; they are a claim about total same-color area elsewhere, and truth swaps which side survives.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = int(get_bg(grid))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    by_color: dict[int, list[dict]] = {}
    for component in components:
        by_color.setdefault(int(component["color"]), []).append(component)

    out = grid.copy()
    for color, objs in by_color.items():
        frames = [obj for obj in objs if _is_hollow_frame(obj)]
        assert len(frames) == 1, f"expected one hollow frame for color {color}"
        frame = frames[0]
        inner = [obj for obj in objs if obj is not frame and _strictly_inside(obj["bbox"], frame["bbox"])]
        outer = [obj for obj in objs if obj is not frame and not _strictly_inside(obj["bbox"], frame["bbox"])]

        claim = sum(int(obj["area"]) for obj in inner)
        actual = sum(int(obj["area"]) for obj in outer)
        assert claim > 0 and actual > 0

        if claim == actual:
            for obj in outer:
                coords = obj["coords"]
                out[coords[:, 0], coords[:, 1]] = bg
        else:
            for obj in [frame, *inner]:
                coords = obj["coords"]
                out[coords[:, 0], coords[:, 1]] = bg
    return out


def _generate_case(return_spec: bool = False):
    for _ in range(MAX_GEN_ATTEMPTS):
        frame_count = random.choices([2, 3], weights=[2, 5])[0]
        bg = random.choice(PALETTE)
        colors = random.sample([c for c in PALETTE if c != bg], frame_count)
        sizes = _choose_frame_sizes(frame_count)
        height, width = _choose_canvas(frame_count, sizes)
        grid = np.full((height, width), bg, dtype=int)

        specs = _place_frames(grid, colors, sizes, bg)
        if specs is None:
            continue

        truths = [random.choice([True, False]) for _ in range(frame_count)]
        modes = [random.choice(["single", "multi"]) for _ in range(frame_count)]
        trap_idx, single_idx = random.sample(range(frame_count), 2)
        modes[trap_idx] = "multi"
        modes[single_idx] = "single"
        truths[single_idx] = not truths[trap_idx]

        all_forbidden = [spec["forbidden"] for spec in specs]
        valid = True
        saw_single = False
        saw_multi = False
        count_trap = False
        for idx, (spec, truth, mode) in enumerate(zip(specs, truths, modes)):
            bbox = spec["bbox"]
            interior_width = _bbox_w(bbox) - 2
            capacity = len(range(2, interior_width, 2))
            max_area = min(5, capacity)
            min_area = 2 if mode == "multi" else 1
            if min_area > max_area:
                valid = False
                break
            actual = random.randint(min_area, max_area)
            if mode == "single":
                parts = [1] * actual
                saw_single = True
            else:
                parts = _partition_area(actual, require_multi=True)
                saw_multi = True

            object_count = len(parts)
            if idx == trap_idx and truth:
                claim = actual
                count_trap = count_trap or (claim != object_count)
            elif idx == trap_idx and not truth:
                claim = object_count
                if claim == actual:
                    valid = False
                    break
                count_trap = True
            elif truth:
                claim = actual
            else:
                claim_choices = [n for n in range(1, capacity + 1) if n != actual]
                if not claim_choices:
                    valid = False
                    break
                claim = random.choice(claim_choices)

            dot_row = bbox[0] + _bbox_h(bbox) // 2
            dot_cols = _dot_columns(interior_width, claim)
            dot_coords = {(dot_row, bbox[1] + col) for col in dot_cols}
            for r, c in dot_coords:
                grid[r, c] = spec["color"]

            pattern_coords: set[Coord] = set()
            for part in parts:
                patch = _sprite_patch(part, spec["color"], bg)
                loc = _place_patch_random(grid, patch, bg, all_forbidden)
                if loc is None:
                    valid = False
                    break
                paste(grid, patch, loc[0], loc[1], transparent=bg)
                for rr, cc in np.argwhere(patch == spec["color"]):
                    pattern_coords.add((loc[0] + int(rr), loc[1] + int(cc)))
            if not valid:
                break

            spec["truth"] = truth
            spec["claim"] = claim
            spec["actual"] = actual
            spec["dot_coords"] = dot_coords
            spec["pattern_coords"] = pattern_coords
            spec["pattern_parts"] = parts
            spec["object_count"] = object_count

        if not valid or not (saw_single and saw_multi and count_trap):
            continue

        expected = _expected_output(grid, specs, bg)
        got = main(grid)
        if not np.array_equal(got, expected):
            continue

        parsed = _parse_case(grid)
        if parsed is None:
            continue
        if sorted((p["truth"], p["claim"], p["actual"]) for p in parsed) != sorted(
            (s["truth"], s["claim"], s["actual"]) for s in specs
        ):
            continue

        if return_spec:
            return grid, expected, specs
        return grid
    raise RuntimeError("failed to generate a valid ARC input")


def _parse_case(grid: np.ndarray) -> list[dict] | None:
    bg = int(get_bg(grid))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    by_color: dict[int, list[dict]] = {}
    for component in components:
        by_color.setdefault(int(component["color"]), []).append(component)

    parsed: list[dict] = []
    for color, objs in by_color.items():
        frames = [obj for obj in objs if _is_hollow_frame(obj)]
        if len(frames) != 1:
            return None
        frame = frames[0]
        inner = [obj for obj in objs if obj is not frame and _strictly_inside(obj["bbox"], frame["bbox"])]
        outer = [obj for obj in objs if obj is not frame and not _strictly_inside(obj["bbox"], frame["bbox"])]
        if not inner or not outer:
            return None
        claim = sum(int(obj["area"]) for obj in inner)
        actual = sum(int(obj["area"]) for obj in outer)
        outer_areas = sorted(int(obj["area"]) for obj in outer)
        parsed.append(
            {
                "color": color,
                "claim": claim,
                "actual": actual,
                "truth": claim == actual,
                "outer_areas": outer_areas,
            }
        )
    has_single_only = any(all(area == 1 for area in item["outer_areas"]) for item in parsed)
    has_multi_area = any(any(area > 1 for area in item["outer_areas"]) for item in parsed)
    has_true = any(item["truth"] for item in parsed)
    has_false = any(not item["truth"] for item in parsed)
    has_count_trap = any((item["claim"] == item["actual"]) != (item["claim"] == len(item["outer_areas"])) for item in parsed)
    if not (has_single_only and has_multi_area and has_true and has_false and has_count_trap):
        return None
    return parsed


def generate_input() -> np.ndarray:
    return _generate_case(return_spec=False)
