from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_task_013_4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BG = 0
MARKER = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 26
PLACEMENT_GAP = 2
GEN_RETRIES = 400


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    return int(bbox_h(box) * bbox_w(box))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _point_in_open_bbox(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = map(int, point)
    r0, c0, r1, c1 = map(int, box)
    return r0 < r < r1 and c0 < c < c1


def _is_rectangle_frame(coords: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    h = int(bbox_h(box))
    w = int(bbox_w(box))
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return False

    r0, c0, _, _ = map(int, box)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, 0].all() or not mask[:, -1].all():
        return False
    if h > 2 and w > 2 and mask[1:-1, 1:-1].any():
        return False
    return True


def _extract_frames(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    frames = []
    for comp in components:
        color = int(comp["color"])
        if color == MARKER:
            continue
        if _is_rectangle_frame(comp["coords"], comp["bbox"]):
            frames.append(
                {
                    "color": color,
                    "coords": comp["coords"],
                    "bbox": tuple(map(int, comp["bbox"])),
                    "area": int(comp["area"]),
                }
            )
    assert frames, "expected at least one rectangular frame"
    return frames


def _frame_parents(frames: list[dict]) -> list[int | None]:
    parents: list[int | None] = [None] * len(frames)
    for i, frame in enumerate(frames):
        candidates = [
            j
            for j, other in enumerate(frames)
            if i != j and _strictly_contains(other["bbox"], frame["bbox"])
        ]
        if candidates:
            parents[i] = min(candidates, key=lambda j: _bbox_area(frames[j]["bbox"]))
    return parents


def _root_index(index: int, parents: list[int | None]) -> int:
    cur = int(index)
    while parents[cur] is not None:
        cur = int(parents[cur])
    return cur


def _solve_groups(grid: np.ndarray) -> list[dict]:
    frames = _extract_frames(grid)
    parents = _frame_parents(frames)
    roots = [i for i, parent in enumerate(parents) if parent is None]
    assert roots, "expected at least one top-level frame"

    marker_coords = np.argwhere(grid == MARKER)
    assert marker_coords.size > 0, "expected grey marker pixels"

    counts = {root: 0 for root in roots}
    for coord in marker_coords:
        owners = [root for root in roots if _point_in_open_bbox(tuple(coord), frames[root]["bbox"])]
        assert len(owners) == 1, "each marker must belong to exactly one top-level frame"
        counts[owners[0]] += 1

    groups = []
    for root in roots:
        count = int(counts[root])
        assert count > 0, "every top-level frame must own at least one marker"
        groups.append(
            {
                "color": int(frames[root]["color"]),
                "count": count,
                "bbox": tuple(map(int, frames[root]["bbox"])),
            }
        )
    groups.sort(key=lambda group: (group["count"], group["bbox"][0], group["bbox"][1]))
    return groups


def _groups_to_output(groups: list[dict]) -> np.ndarray:
    width = max(int(group["count"]) for group in groups)
    out = np.zeros((len(groups), width), dtype=int)
    for row, group in enumerate(groups):
        out[row, : int(group["count"])] = int(group["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular frames to their outermost color, then summarize marker counts by outer frame.

    Input structure:
    - Black background surrounds 2 to 3 separated monochrome rectangular frames.
    - A top-level frame may contain one strictly nested inner frame of a different color.
    - Grey marker pixels lie inside frame interiors and never touch any frame border.

    Rule:
    - Detect every one-pixel-thick rectangular frame and its strict containment relation.
    - For each top-level frame, treat every nested frame inside it as inheriting the top-level frame color.
    - Count all grey markers inside each top-level frame's bbox interior, including markers inside nested descendants.
    - Sort the top-level groups by marker count ascending, breaking ties by the top-level bbox upper-left position.
    - Output one row per group, filled with that top-level color repeated count times and padded with black to the maximum count.

    Key insight:
    - The inner-frame color is not a separate final group: inheritance collapses every nested subtree into its outermost frame before counting and sorting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    groups = _solve_groups(g)
    return _groups_to_output(groups)


def _draw_frame(patch: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    patch[r0, c0 : c1 + 1] = int(color)
    patch[r1, c0 : c1 + 1] = int(color)
    patch[r0 : r1 + 1, c0] = int(color)
    patch[r0 : r1 + 1, c1] = int(color)


def _interior_cells(
    box: tuple[int, int, int, int],
    *,
    exclude_box: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if exclude_box is not None:
                er0, ec0, er1, ec1 = map(int, exclude_box)
                if er0 <= r <= er1 and ec0 <= c <= ec1:
                    continue
            cells.append((r, c))
    return cells


def _sample_scattered_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if count > len(cells):
        raise ValueError("not enough candidate cells")

    for _ in range(80):
        shuffled = list(cells)
        random.shuffle(shuffled)
        picked: list[tuple[int, int]] = []
        for cell in shuffled:
            if all(abs(cell[0] - rr) + abs(cell[1] - cc) != 1 for rr, cc in picked):
                picked.append(cell)
                if len(picked) == count:
                    return picked
    return random.sample(cells, count)


def _make_group_patch(outer_color: int, inner_color: int | None, nested: bool) -> dict:
    if nested:
        outer_h = random.randint(9, 11)
        outer_w = random.randint(9, 11)
        inner_h = random.randint(5, min(6, outer_h - 4))
        inner_w = random.randint(5, min(6, outer_w - 4))
        inner_r = random.randint(2, outer_h - inner_h - 2)
        inner_c = random.randint(2, outer_w - inner_w - 2)
        outer_count = random.randint(1, 5)
        inner_count = random.randint(1, 5)
    else:
        outer_h = random.randint(6, 9)
        outer_w = random.randint(6, 9)
        inner_h = inner_w = inner_r = inner_c = 0
        outer_count = random.randint(1, 5)
        inner_count = 0

    patch = np.zeros((outer_h, outer_w), dtype=int)
    outer_box = (0, 0, outer_h - 1, outer_w - 1)
    _draw_frame(patch, outer_box, int(outer_color))

    inner_box = None
    if nested:
        inner_box = (inner_r, inner_c, inner_r + inner_h - 1, inner_c + inner_w - 1)
        assert inner_color is not None
        _draw_frame(patch, inner_box, int(inner_color))

    outer_cells = _interior_cells(outer_box, exclude_box=inner_box)
    outer_markers = _sample_scattered_cells(outer_cells, outer_count)
    for r, c in outer_markers:
        patch[r, c] = MARKER

    inner_markers: list[tuple[int, int]] = []
    if inner_box is not None:
        inner_cells = _interior_cells(inner_box)
        inner_markers = _sample_scattered_cells(inner_cells, inner_count)
        for r, c in inner_markers:
            patch[r, c] = MARKER

    return {
        "patch": patch,
        "outer_color": int(outer_color),
        "inner_color": None if inner_color is None else int(inner_color),
        "nested": bool(nested),
        "outer_count": int(outer_count),
        "inner_count": int(inner_count),
        "total": int(outer_count + inner_count),
    }


def _rect_conflicts(
    top: int,
    left: int,
    h: int,
    w: int,
    placed: list[tuple[int, int, int, int]],
    gap: int,
) -> bool:
    r0, c0, r1, c1 = int(top), int(left), int(top + h - 1), int(left + w - 1)
    for pr0, pc0, pr1, pc1 in placed:
        if not (
            r1 + gap < pr0
            or pr1 + gap < r0
            or c1 + gap < pc0
            or pc1 + gap < c0
        ):
            return True
    return False


def _place_group_patches(specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    max_h = max(int(spec["patch"].shape[0]) for spec in specs)
    max_w = max(int(spec["patch"].shape[1]) for spec in specs)

    for _ in range(120):
        height = random.randint(max(GRID_SIDE_MIN, max_h + 2), GRID_SIDE_MAX)
        width = random.randint(max(GRID_SIDE_MIN, max_w + 2), GRID_SIDE_MAX)
        grid = np.zeros((height, width), dtype=int)

        order = list(range(len(specs)))
        random.shuffle(order)
        boxes_by_index: dict[int, tuple[int, int, int, int]] = {}
        placed_boxes: list[tuple[int, int, int, int]] = []

        ok = True
        for idx in order:
            patch = specs[idx]["patch"]
            h, w = map(int, patch.shape)
            candidates = [
                (r, c)
                for r in range(1, height - h)
                for c in range(1, width - w)
                if not _rect_conflicts(r, c, h, w, placed_boxes, PLACEMENT_GAP)
            ]
            if not candidates:
                ok = False
                break
            top, left = random.choice(candidates)
            box = (top, left, top + h - 1, left + w - 1)
            boxes_by_index[idx] = box
            placed_boxes.append(box)
            view = grid[top : top + h, left : left + w]
            mask = patch != BG
            view[mask] = patch[mask]

        if not ok:
            continue

        placed_specs = []
        for idx, spec in enumerate(specs):
            placed = dict(spec)
            placed["bbox"] = boxes_by_index[idx]
            placed_specs.append(placed)
        return grid, placed_specs
    return None


def _expected_output_from_specs(specs: list[dict]) -> np.ndarray:
    groups = [
        {
            "color": int(spec["outer_color"]),
            "count": int(spec["total"]),
            "bbox": tuple(map(int, spec["bbox"])),
        }
        for spec in specs
    ]
    groups.sort(key=lambda group: (group["count"], group["bbox"][0], group["bbox"][1]))
    return _groups_to_output(groups)


def _wrong_deepest_frame_output(grid: np.ndarray) -> np.ndarray:
    frames = _extract_frames(grid)
    marker_coords = np.argwhere(grid == MARKER)
    counts = [0] * len(frames)

    for coord in marker_coords:
        containing = [i for i, frame in enumerate(frames) if _point_in_open_bbox(tuple(coord), frame["bbox"])]
        assert containing, "each marker must be inside at least one frame"
        owner = min(containing, key=lambda i: _bbox_area(frames[i]["bbox"]))
        counts[owner] += 1

    groups = []
    for frame, count in zip(frames, counts):
        if count > 0:
            groups.append(
                {
                    "color": int(frame["color"]),
                    "count": int(count),
                    "bbox": tuple(map(int, frame["bbox"])),
                }
            )
    groups.sort(key=lambda group: (group["count"], group["bbox"][0], group["bbox"][1]))
    return _groups_to_output(groups)


def _validate_generated_case(grid: np.ndarray, specs: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if sum(int(spec["nested"]) for spec in specs) < 1:
        return False
    if sum(not bool(spec["nested"]) for spec in specs) < 1:
        return False

    totals = [int(spec["total"]) for spec in specs]
    if len(set(totals)) != len(totals):
        return False

    expected = _expected_output_from_specs(specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    groups = _solve_groups(grid)
    if [group["count"] for group in groups] != sorted(totals):
        return False

    wrong = _wrong_deepest_frame_output(grid)
    if np.array_equal(actual, wrong):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.choice([2, 3])
    if n_groups == 2:
        nested_flags = [True, False]
    else:
        nested_flags = [True, False, random.choice([False, True])]
    random.shuffle(nested_flags)

    n_nested = sum(nested_flags)
    colors = random.sample(FRAME_COLORS, n_groups + n_nested)
    outer_colors = colors[:n_groups]
    inner_colors = colors[n_groups:]
    inner_iter = iter(inner_colors)

    specs = []
    used_totals: set[int] = set()
    for outer_color, nested in zip(outer_colors, nested_flags):
        inner_color = int(next(inner_iter)) if nested else None
        for _ in range(60):
            spec = _make_group_patch(
                outer_color=int(outer_color),
                inner_color=inner_color,
                nested=bool(nested),
            )
            if int(spec["total"]) not in used_totals:
                specs.append(spec)
                used_totals.add(int(spec["total"]))
                break
        else:
            return None

    placed = _place_group_patches(specs)
    if placed is None:
        return None
    grid, placed_specs = placed
    if not _validate_generated_case(grid, placed_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame counting input")


def _manual_tie_case() -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((15, 19), dtype=int)
    specs = []
    for box, color, markers in [
        ((1, 1, 5, 5), 1, [(2, 2), (4, 4)]),
        ((1, 9, 5, 13), 2, [(2, 10), (4, 12)]),
        ((8, 4, 12, 8), 3, [(9, 5), (11, 7)]),
    ]:
        _draw_frame(grid, box, color)
        for r, c in markers:
            grid[r, c] = MARKER
        specs.append({"outer_color": color, "total": 2, "bbox": box})
    return grid, _expected_output_from_specs(specs)
