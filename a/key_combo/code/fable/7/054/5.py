from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
GREY = Color.GREY
BASELINE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_RETRIES = 400


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


ROOT_SHAPES = [
    _mask(["1111"]),
    _mask(["11111"]),
    _mask(["1111", "1111"]),
    _mask(["111", "111"]),
]

OTHER_SHAPES = [
    _mask(["11"]),
    _mask(["111"]),
    _mask(["1", "1"]),
    _mask(["1", "1", "1"]),
    _mask(["11", "11"]),
    _mask(["10", "11"]),
    _mask(["01", "11"]),
    _mask(["111", "010"]),
    _mask(["110", "011"]),
]


def _shape_info(mask: np.ndarray) -> dict:
    coords = np.argwhere(mask)
    low_r = int(coords[:, 0].max())
    low_left_c = int(coords[coords[:, 0] == low_r, 1].min())
    return {
        "mask": mask,
        "coords": coords.astype(int),
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "low_left_rel_c": low_left_c,
    }


ROOT_INFOS = [_shape_info(mask) for mask in ROOT_SHAPES]
OTHER_INFOS = [_shape_info(mask) for mask in OTHER_SHAPES]


def _segment_spans(baseline: np.ndarray) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    start = 0
    for c in range(1, int(baseline.shape[0]) + 1):
        if c == baseline.shape[0] or int(baseline[c]) != int(baseline[start]):
            spans.append((start, c - 1, int(baseline[start])))
            start = c
    return spans


def _lowest_left_col(coords: np.ndarray) -> int:
    low_r = int(coords[:, 0].max())
    return int(coords[coords[:, 0] == low_r, 1].min())


def _analyze_virtual_fall(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 4 <= h <= 30 and 4 <= w <= 30

    bg = int(get_bg(g))
    baseline = g[-1].copy()
    assert np.all(baseline != bg), "bottom row must be fully colored"
    assert GREY not in set(map(int, baseline.tolist())), "grey may only appear in floating objects"

    upper = g[:-1]
    upper_fg = sorted(set(map(int, upper.flatten())) - {bg})
    assert upper_fg == [GREY], "rows above the baseline must contain only grey objects on the background"

    objects = find_components(
        g,
        bg=bg,
        colors=[GREY],
        connectivity=4,
        monochromatic=True,
    )
    assert objects, "expected at least one floating grey object"

    ordered = sorted(
        objects,
        key=lambda obj: (-int(obj["bbox"][2]), int(obj["bbox"][1]), int(obj["bbox"][0])),
    )

    settled_owner = np.full((h - 1, w), -1, dtype=int)
    analyzed_objects: list[dict] = []

    for idx, obj in enumerate(ordered):
        coords = np.asarray(obj["coords"], dtype=int)
        drop = 0
        while True:
            trial = coords + np.array([drop + 1, 0], dtype=int)
            if int(trial[:, 0].max()) > h - 2:
                break
            if np.any(settled_owner[trial[:, 0], trial[:, 1]] != -1):
                break
            drop += 1

        final_coords = coords + np.array([drop, 0], dtype=int)
        below_r = final_coords[:, 0] + 1
        below_c = final_coords[:, 1]
        direct_base_color = int(baseline[_lowest_left_col(coords)])

        supported_by_floor = bool(np.any(below_r == h - 1))
        support_ids = sorted(
            {
                int(owner)
                for owner in settled_owner[below_r[below_r < h - 1], below_c[below_r < h - 1]].tolist()
                if int(owner) >= 0
            }
        )

        if supported_by_floor:
            assert not support_ids, "an object may not be supported by both the baseline and another object"
            support = None
            color = direct_base_color
        else:
            assert len(support_ids) == 1, "support must be uniquely determined"
            support = support_ids[0]
            color = int(analyzed_objects[support]["color"])

        assert np.all(settled_owner[final_coords[:, 0], final_coords[:, 1]] == -1), "settled objects may not overlap"
        settled_owner[final_coords[:, 0], final_coords[:, 1]] = idx

        analyzed_objects.append(
            {
                "coords": coords,
                "bbox": tuple(map(int, obj["bbox"])),
                "final_coords": final_coords,
                "drop": int(drop),
                "support": support,
                "color": int(color),
                "direct_base_color": int(direct_base_color),
            }
        )

    return {
        "bg": bg,
        "baseline": baseline,
        "objects": analyzed_objects,
    }


def _paint_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for obj in analysis["objects"]:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every grey object in place using the baseline color reached by its invisible support chain.

    Input structure:
    - The bottom row is the only colored region and is split into contiguous color segments with different adjacent colors.
    - Above it, the background is uniform except for several separated grey 4-connected floating objects.

    Rule:
    - Process the grey objects from lower initial position to higher initial position and imagine each one falling straight down rigidly.
    - Each object stops when one of its pixels would land just above the baseline, or when it would land on an already-settled object.
    - An object that lands on the baseline takes the color directly below its lowest-leftmost pixel; an object that lands on another object inherits that support object's color.
    - In the output, nothing moves: keep every object in its original location and only replace grey with the inferred color, leaving the baseline unchanged.

    Key insight:
    - The visible recoloring is determined by an unseen gravity simulation, so stacked objects must inherit the color of the lowest baseline-touching object in their support chain rather than simply reading the segment beneath themselves.
    """
    return _paint_output(grid, _analyze_virtual_fall(grid))


def _random_composition(total: int, parts: int, min_each: int, rng: np.random.Generator) -> list[int]:
    widths = [min_each] * parts
    extra = total - parts * min_each
    for _ in range(extra):
        widths[int(rng.integers(0, parts))] += 1
    rng.shuffle(widths)
    return widths


def _build_baseline(width: int, rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    max_parts = min(5, width // 3)
    n_parts = int(rng.integers(3, max_parts + 1))
    widths = _random_composition(width, n_parts, 3, rng)
    colors = rng.choice(BASELINE_COLORS, size=n_parts, replace=False).astype(int).tolist()
    baseline = np.empty(width, dtype=int)
    spans = []
    c0 = 0
    for seg_w, color in zip(widths, colors, strict=True):
        baseline[c0 : c0 + seg_w] = int(color)
        spans.append((c0, c0 + seg_w - 1, int(color)))
        c0 += seg_w
    return baseline, spans


def _interval_conflicts(c0: int, c1: int, protected: list[tuple[int, int]], gap: int = 0) -> bool:
    for p0, p1 in protected:
        if not (c1 + gap < p0 or c0 - gap > p1):
            return True
    return False


def _can_place(mask: np.ndarray, top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + mh + 1)
    c1 = min(w, left + mw + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _stamp(mask: np.ndarray, top: int, left: int, grid: np.ndarray, occupied: np.ndarray) -> None:
    rr, cc = np.nonzero(mask)
    grid[top + rr, left + cc] = GREY
    occupied[top + rr, left + cc] = True


def _pick_chain_layout(
    width: int,
    spans: list[tuple[int, int, int]],
    rng: np.random.Generator,
) -> tuple[dict, dict, int, int] | None:
    boundary_choices = [i for i in range(len(spans) - 1) if spans[i][1] - spans[i][0] + 1 >= 2 and spans[i + 1][1] - spans[i + 1][0] + 1 >= 2]
    if not boundary_choices:
        return None

    rng.shuffle(boundary_choices)
    for boundary_idx in boundary_choices:
        left_s, left_e, _left_color = spans[boundary_idx]
        right_s, right_e, _right_color = spans[boundary_idx + 1]
        boundary = left_e

        root_order = np.arange(len(ROOT_INFOS))
        rng.shuffle(root_order)
        for root_idx in root_order:
            root = ROOT_INFOS[int(root_idx)]
            root_starts = [
                c0
                for c0 in range(max(0, boundary - 2), min(boundary + 1, width - root["w"] + 1))
                if c0 <= boundary < c0 + root["w"] - 1 and left_s <= c0 <= left_e
            ]
            if not root_starts:
                continue
            rng.shuffle(root_starts)
            for root_left in root_starts:
                child_options: list[tuple[dict, int]] = []
                child_order = np.arange(len(OTHER_INFOS))
                rng.shuffle(child_order)
                for child_idx in child_order:
                    child = OTHER_INFOS[int(child_idx)]
                    if child["w"] > root["w"] - 1:
                        continue
                    start_min = root_left
                    start_max = root_left + root["w"] - child["w"]
                    for child_left in range(start_min, start_max + 1):
                        low_left_abs = child_left + child["low_left_rel_c"]
                        if right_s <= low_left_abs <= right_e:
                            child_options.append((child, child_left))
                if child_options:
                    child, child_left = child_options[int(rng.integers(0, len(child_options)))]
                    return root, child, int(root_left), int(child_left)
    return None


def _place_chain(
    height: int,
    width: int,
    root: dict,
    child: dict,
    root_left: int,
    child_left: int,
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    root_top_min = max(2, height - root["h"] - 8)
    root_top_max = height - root["h"] - 2
    if root_top_min > root_top_max:
        return None

    root_tops = list(range(root_top_min, root_top_max + 1))
    rng.shuffle(root_tops)
    for root_top in root_tops:
        child_top_max = root_top - child["h"] - 2
        if child_top_max < 1:
            continue
        child_tops = list(range(1, child_top_max + 1))
        rng.shuffle(child_tops)
        if not _can_place(root["mask"], root_top, root_left, occupied):
            continue
        for child_top in child_tops:
            if not _can_place(child["mask"], child_top, child_left, occupied):
                continue
            _stamp(root["mask"], root_top, root_left, grid, occupied)
            _stamp(child["mask"], child_top, child_left, grid, occupied)
            return (int(root_top), int(root_left)), (int(child_top), int(child_left))
    return None


def _place_extras(
    grid: np.ndarray,
    occupied: np.ndarray,
    protected_cols: list[tuple[int, int]],
    rng: np.random.Generator,
) -> int:
    h, w = grid.shape
    count = int(rng.integers(1, 4))
    placed = 0
    shape_order = rng.permutation(len(OTHER_INFOS)).tolist()

    for shape_idx in shape_order:
        if placed >= count:
            break
        shape = OTHER_INFOS[int(shape_idx)]
        candidates = [
            (top, left)
            for top in range(1, h - shape["h"] + 1)
            for left in range(0, w - shape["w"] + 1)
            if top + shape["h"] - 1 <= h - 2
            if not _interval_conflicts(left, left + shape["w"] - 1, protected_cols, gap=1)
            if _can_place(shape["mask"], top, left, occupied)
        ]
        if not candidates:
            continue
        top, left = candidates[int(rng.integers(0, len(candidates)))]
        _stamp(shape["mask"], int(top), int(left), grid, occupied)
        protected_cols.append((int(left), int(left + shape["w"] - 1)))
        placed += 1

    return placed


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_virtual_fall(grid)
    except AssertionError:
        return False

    baseline = analysis["baseline"]
    spans = _segment_spans(baseline)
    objects = analysis["objects"]
    inherited = [obj for obj in objects if obj["support"] is not None]
    roots = [obj for obj in objects if obj["support"] is None]
    colors_used = {int(obj["color"]) for obj in objects}

    if len(spans) < 3:
        return False
    if len(objects) < 3 or not inherited or not roots:
        return False
    if len(colors_used) < 2:
        return False
    if not any(int(obj["color"]) != int(obj["direct_base_color"]) for obj in inherited):
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(13, 23))
    width = int(rng.integers(15, 29))
    grid = np.full((height, width), BG, dtype=int)
    baseline, spans = _build_baseline(width, rng)
    grid[-1, :] = baseline

    layout = _pick_chain_layout(width, spans, rng)
    if layout is None:
        return None
    root, child, root_left, child_left = layout

    occupied = np.zeros((height - 1, width), dtype=bool)
    if _place_chain(height - 1, width, root, child, root_left, child_left, rng, grid[:-1], occupied) is None:
        return None

    protected_cols = [(int(root_left), int(root_left + root["w"] - 1))]
    if _place_extras(grid[:-1], occupied, protected_cols, rng) == 0:
        return None

    if not _semantic_ok(grid):
        return None

    expected = _paint_output(grid, _analyze_virtual_fall(grid))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _attempt_generate(np.random.default_rng())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_RETRIES} attempts")
