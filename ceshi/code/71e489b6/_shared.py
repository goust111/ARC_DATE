from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask  # noqa: E402


VARIANT_TITLES = {
    "D1/0": "十字形标记替代方形光晕",
    "D1/1": "可变半径光晕",
    "D1/2": "填充缺陷而非环绕",
    "D1/4": "方向性射线标记",
    "D2/0": "仅检测单像素缺陷",
    "D2/2": "异常检测：蓝色区域中的非蓝色像素",
    "D3/0": "毛刺保留并标记",
    "D3/1": "毛刺扩展而非吸收",
    "D4/1": "基于距离的检测",
    "D5/0": "按缺陷类型分色",
    "D5/1": "光晕颜色由缺陷大小编码",
    "D7/0": "光晕仅限蓝色区域内",
    "D7/1": "光晕避让其他缺陷",
    "D7/2": "光晕仅覆写蓝色像素",
    "D9/0": "黑色区域中的灰色噪点",
    "D9/1": "多种颜色的干扰毛刺",
    "D10/0": "多色前景区域",
    "D11/0": "迭代检测与标记",
    "D12/0": "缺陷像素也涂为光晕色",
    "D12/1": "缺陷像素变为第三种标记色",
    "D12/2": "光晕中心留空但保留原色",
    "D13/0": "8连通分量检测",
}

REGION_LAYOUTS = [
    [(2, 2, 8, 8)],
    [(2, 2, 8, 8), (2, 10, 8, 16)],
    [(2, 2, 8, 8), (2, 10, 8, 16), (10, 4, 16, 13)],
]

DEFAULT_HALO = 7
SPUR_MARK = 6
REGION_HALO_BY_COLOR = {1: 7, 3: 6, 2: 4}
REGION_HALO_BY_INDEX = [7, 6, 4]


def _variant_doc(variant: str) -> str:
    title = VARIANT_TITLES[variant]
    return (
        f"Variant solver for 71e489b6 ({variant}).\n\n"
        "Input structure:\n"
        "- One or more large foreground rectangles contain small defects or boundary intrusions.\n"
        "- Background may also contain removable spur/noise pixels depending on the variant.\n"
        "- The chosen variant changes exactly one part of the defect-marking pipeline.\n\n"
        "Rule:\n"
        f"- Detect foreground regions, clean or reinterpret the background noise, then apply `{title}`\n"
        "  to decide which defect pixels are targeted and how they are marked.\n\n"
        "Key insight: The puzzle stays in-place, but the neighborhood, color, or sequencing rule for\n"
        "marking defects changes in a visibly structured way."
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _region_mask(region: dict) -> np.ndarray:
    r0, c0, r1, c1 = region["bbox"]
    mask = np.zeros(region["grid_shape"], dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _in_bbox(region: dict, r: int, c: int) -> bool:
    r0, c0, r1, c1 = region["bbox"]
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_distance(region: dict, r: int, c: int) -> int:
    r0, c0, r1, c1 = region["bbox"]
    return min(r - r0, r1 - r, c - c0, c1 - c)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _foreground_colors(variant: str) -> set[int]:
    return {1, 2, 3} if variant == "D10/0" else {1}


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    comps = []
    for comp in components_mask(mask, connectivity=connectivity):
        comps.append([(int(r), int(c)) for r, c in comp])
    return comps


def _detect_regions(grid: np.ndarray, variant: str) -> list[dict]:
    regions: list[dict] = []
    min_area = 16
    for color in sorted(_foreground_colors(variant)):
        comps = _components_from_mask(grid == color, connectivity=4)
        for comp in comps:
            if len(comp) < min_area:
                continue
            arr = np.array(comp, dtype=int)
            regions.append(
                {
                    "color": color,
                    "bbox": _bbox(arr),
                    "cells": set(comp),
                    "grid_shape": grid.shape,
                }
            )
    regions.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    for idx, region in enumerate(regions):
        region["index"] = idx
    return regions


def _outside_region_noise_comp(region_masks: list[np.ndarray], comp: list[tuple[int, int]]) -> bool:
    for r, c in comp:
        for mask in region_masks:
            if mask[r, c]:
                return False
    return True


def _preprocess(grid: np.ndarray, variant: str, regions: list[dict]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    work = np.array(grid, dtype=int)
    spur_marks: list[tuple[int, int]] = []
    fg_colors = _foreground_colors(variant)
    region_masks = [_region_mask(region) for region in regions]

    # Foreground-colored spurs in background.
    for color in sorted(fg_colors):
        comps = _components_from_mask(work == color, connectivity=4)
        for comp in comps:
            if len(comp) >= 16 or not _outside_region_noise_comp(region_masks, comp):
                continue
            if variant == "D3/0":
                spur_marks.extend(comp)
            elif variant == "D3/1":
                for r, c in comp:
                    for rr, cc in _neighbors4(r, c):
                        if 0 <= rr < work.shape[0] and 0 <= cc < work.shape[1] and work[rr, cc] == 0:
                            work[rr, cc] = color
            else:
                for r, c in comp:
                    work[r, c] = 0

    # Non-foreground noise in the background.
    removable_noise = set()
    if variant == "D9/0":
        removable_noise = {5}
    elif variant == "D9/1":
        removable_noise = {2, 3, 4, 5, 6, 7, 8, 9}

    for color in sorted(removable_noise):
        comps = _components_from_mask(work == color, connectivity=4)
        for comp in comps:
            if _outside_region_noise_comp(region_masks, comp):
                for r, c in comp:
                    work[r, c] = 0

    return work, spur_marks


def _collect_targets_for_region(work: np.ndarray, region: dict, variant: str) -> list[dict]:
    r0, c0, r1, c1 = region["bbox"]
    color = region["color"]
    defect_mask = np.zeros_like(work, dtype=bool)
    defect_mask[r0 : r1 + 1, c0 : c1 + 1] = work[r0 : r1 + 1, c0 : c1 + 1] != color

    connectivity = 8 if variant == "D13/0" else 4
    comps = _components_from_mask(defect_mask, connectivity=connectivity)
    targets: list[dict] = []

    if variant == "D11/0":
        peel = defect_mask.copy()
        while True:
            round_pts = []
            for comp in _components_from_mask(peel, connectivity=4):
                comp_set = set(comp)
                for r, c in comp:
                    deg = sum((rr, cc) in comp_set for rr, cc in _neighbors4(r, c))
                    if deg <= 1:
                        round_pts.append((r, c))
            if not round_pts:
                break
            for r, c in round_pts:
                targets.append({"pixels": {(r, c)}, "size": 1, "kind": "iter", "region": region})
                peel[r, c] = False
        return targets

    if variant == "D4/1":
        for comp in comps:
            pts = {(r, c) for r, c in comp if _bbox_distance(region, r, c) >= 2}
            if pts:
                targets.append({"pixels": pts, "size": len(comp), "kind": "internal", "region": region})
        return targets

    if variant == "D2/0":
        for comp in comps:
            if len(comp) == 1:
                pt = comp[0]
                targets.append({"pixels": {pt}, "size": 1, "kind": "internal", "region": region})
        return targets

    if variant == "D2/2":
        for comp in comps:
            targets.append({"pixels": set(comp), "size": len(comp), "kind": "internal", "region": region})
        return targets

    if variant == "D13/0":
        for comp in comps:
            if len(comp) <= 3:
                targets.append({"pixels": set(comp), "size": len(comp), "kind": "internal", "region": region})
        return targets

    leaf_pts = set()
    for comp in comps:
        touch_edge = any(_bbox_distance(region, r, c) == 0 for r, c in comp)
        if touch_edge:
            leaf_pts.update(comp)
        elif len(comp) <= 3:
            targets.append({"pixels": set(comp), "size": len(comp), "kind": "internal", "region": region})

    if leaf_pts:
        targets.append({"pixels": leaf_pts, "size": 1, "kind": "leaf", "region": region})
    return targets


def _collect_targets(work: np.ndarray, regions: list[dict], variant: str) -> list[dict]:
    targets: list[dict] = []
    for region in regions:
        targets.extend(_collect_targets_for_region(work, region, variant))
    return targets


def _halo_color(variant: str, item: dict) -> int:
    region = item["region"]
    if variant == "D5/0":
        return 7 if item["kind"] == "internal" else 6
    if variant == "D5/1":
        return {1: 7, 2: 4, 3: 6}.get(min(item["size"], 3), 7)
    if variant == "D10/0":
        return REGION_HALO_BY_COLOR[region["color"]]
    if variant == "D5/2":
        return REGION_HALO_BY_INDEX[region["index"] % len(REGION_HALO_BY_INDEX)]
    return DEFAULT_HALO


def _center_color(variant: str, item: dict, halo_color: int) -> int:
    if variant == "D12/0" or variant == "D1/2":
        return halo_color
    if variant == "D12/1":
        return 2
    if variant == "D12/2":
        return item["region"]["color"]
    return 0


def _footprint(item: dict, variant: str, work: np.ndarray) -> set[tuple[int, int]]:
    region = item["region"]
    out: set[tuple[int, int]] = set()
    pixels = item["pixels"]

    if variant == "D1/2":
        return set(pixels)

    if variant == "D1/4":
        for r, c in pixels:
            out.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                while _in_bbox(region, rr, cc) and work[rr, cc] == region["color"]:
                    out.add((rr, cc))
                    rr += dr
                    cc += dc
        return out

    if variant == "D1/0":
        offsets = [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    elif variant == "D1/1":
        radius = max(1, min(3, item["size"]))
        offsets = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)]
    else:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

    for r, c in pixels:
        for dr, dc in offsets:
            out.add((r + dr, c + dc))
    return out


def _allowed_paint(variant: str, region: dict, work: np.ndarray, r: int, c: int) -> bool:
    if not (0 <= r < work.shape[0] and 0 <= c < work.shape[1]):
        return False
    if variant == "D7/0":
        return _in_bbox(region, r, c)
    if variant == "D7/2":
        return _in_bbox(region, r, c) and work[r, c] == region["color"]
    return True


def _paint_output(grid: np.ndarray, work: np.ndarray, targets: list[dict], spur_marks: list[tuple[int, int]], variant: str) -> np.ndarray:
    out = np.array(work, dtype=int)
    coverage: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    centers: list[tuple[set[tuple[int, int]], int]] = []

    for item in targets:
        halo_color = _halo_color(variant, item)
        center_color = _center_color(variant, item, halo_color)
        centers.append((set(item["pixels"]), center_color))
        for r, c in _footprint(item, variant, work):
            if _allowed_paint(variant, item["region"], work, r, c):
                coverage[(r, c)].append(halo_color)

    for (r, c), colors in coverage.items():
        if variant == "D7/1" and len(colors) > 1:
            continue
        out[r, c] = colors[-1]

    for pixels, center_color in centers:
        for r, c in pixels:
            out[r, c] = center_color

    if variant == "D3/0":
        for r, c in spur_marks:
            out[r, c] = SPUR_MARK

    return out


def solve_variant(variant: str, grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    regions = _detect_regions(g, variant)
    work, spur_marks = _preprocess(g, variant, regions)
    targets = _collect_targets(work, regions, variant)
    return _paint_output(g, work, targets, spur_marks, variant)


def _fill_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _put_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _shifted(layout: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    dr = random.choice([0, 0, 1])
    dc = random.choice([0, 0, 1])
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in layout]


def _make_regions(variant: str) -> list[dict]:
    if variant in {"D5/2", "D10/0"}:
        layout = REGION_LAYOUTS[2]
    elif variant in {"D5/0", "D9/0", "D9/1"}:
        layout = REGION_LAYOUTS[1]
    else:
        layout = random.choice(REGION_LAYOUTS[:2])
    layout = _shifted(layout)
    colors = [1] * len(layout)
    if variant == "D10/0":
        colors = [1, 3, 2][: len(layout)]
    return [{"bbox": bbox, "color": color} for bbox, color in zip(layout, colors)]


def _region_points(region: dict) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = region["bbox"]
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    return {
        "center": (rm, cm),
        "near": (r0 + 1, c0 + 1),
        "far": (r0 + 3, c0 + 3),
        "edge_top": (r0, cm),
        "edge_left": (rm, c0),
        "diag1": (r0 + 1, c0 + 1),
        "diag2": (r0 + 2, c0 + 2),
        "diag3": (r0 + 3, c0 + 3),
        "diag4": (r0 + 4, c0 + 4),
        "tl": (r0 + 1, c0 + 1),
        "tr": (r0 + 1, c1 - 2),
        "bl": (r1 - 2, c0 + 2),
    }


def _default_defects(grid: np.ndarray, regions: list[dict]) -> None:
    for idx, region in enumerate(regions):
        p = _region_points(region)
        if idx == 0:
            _put_cells(grid, [p["center"]], 0)
            _put_cells(grid, [(p["center"][0], p["center"][1] + 2), (p["center"][0], p["center"][1] + 3)], 0)
            _put_cells(grid, [p["edge_top"]], 0)
        else:
            _put_cells(grid, [p["center"]], 0)


def _background_noise_cells() -> list[tuple[int, int]]:
    return [(0, 0), (0, 17), (17, 0), (17, 17), (9, 0), (0, 9)]


def _build_input(variant: str) -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    regions = _make_regions(variant)
    for region in regions:
        _fill_rect(grid, region["bbox"], region["color"])

    if variant == "D2/0":
        p = _region_points(regions[0])
        _put_cells(grid, [p["center"]], 0)
        _put_cells(grid, [(p["center"][0], p["center"][1] + 2), (p["center"][0], p["center"][1] + 3)], 0)
    elif variant == "D2/2":
        p = _region_points(regions[0])
        _put_cells(grid, [p["center"]], 2)
        _put_cells(grid, [(p["center"][0] + 1, p["center"][1] + 2)], 5)
        if len(regions) > 1:
            q = _region_points(regions[1])
            _put_cells(grid, [q["center"]], 6)
    elif variant == "D4/1":
        p = _region_points(regions[0])
        _put_cells(grid, [p["center"]], 0)
        _put_cells(grid, [p["near"]], 0)
    elif variant == "D5/0":
        p = _region_points(regions[0])
        q = _region_points(regions[1])
        _put_cells(grid, [p["center"]], 0)
        _put_cells(grid, [p["edge_top"]], 0)
        _put_cells(grid, [q["center"]], 0)
    elif variant == "D5/1":
        p = _region_points(regions[0])
        _put_cells(grid, [p["tl"]], 0)
        _put_cells(grid, [p["tr"], (p["tr"][0] + 1, p["tr"][1])], 0)
        _put_cells(grid, [p["bl"], (p["bl"][0], p["bl"][1] + 1), (p["bl"][0] - 1, p["bl"][1] + 1)], 0)
    elif variant == "D7/1":
        p = _region_points(regions[0])
        _put_cells(grid, [(p["center"][0], p["center"][1] - 1), (p["center"][0], p["center"][1] + 1)], 0)
    elif variant == "D7/0":
        p = _region_points(regions[0])
        _put_cells(grid, [p["edge_top"]], 0)
    elif variant == "D7/2":
        p = _region_points(regions[0])
        _put_cells(grid, [p["center"], (p["center"][0], p["center"][1] + 1)], 0)
    elif variant == "D10/0":
        for region in regions:
            _put_cells(grid, [_region_points(region)["center"]], 0)
    elif variant == "D11/0":
        p = _region_points(regions[0])
        line = [(p["center"][0], p["center"][1] - 1), p["center"], (p["center"][0], p["center"][1] + 1)]
        _put_cells(grid, line, 0)
        _put_cells(grid, [p["edge_top"]], 0)
    elif variant == "D13/0":
        p = _region_points(regions[0])
        diag = [p["diag1"], p["diag2"], p["diag3"], p["diag4"]]
        _put_cells(grid, diag, 0)
        _put_cells(grid, [p["center"]], 0)
    elif variant in {"D1/1", "D1/4"}:
        p = _region_points(regions[0])
        if variant == "D1/1":
            _put_cells(grid, [p["tl"]], 0)
            _put_cells(grid, [p["tr"], (p["tr"][0] + 1, p["tr"][1])], 0)
            _put_cells(grid, [p["bl"], (p["bl"][0], p["bl"][1] + 1), (p["bl"][0] - 1, p["bl"][1] + 1)], 0)
        else:
            _put_cells(grid, [p["center"]], 0)
            _put_cells(grid, [(p["center"][0], p["center"][1] + 2), (p["center"][0], p["center"][1] + 3)], 0)
            _put_cells(grid, [(p["center"][0] + 2, p["center"][1] - 1), (p["center"][0] + 2, p["center"][1]), (p["center"][0] + 3, p["center"][1])], 0)
    else:
        _default_defects(grid, regions)

    if variant == "D3/0":
        _put_cells(grid, [(0, 3), (16, 17)], 1)
    elif variant == "D3/1":
        _put_cells(grid, [(0, 3), (15, 17)], 1)
    elif variant == "D9/0":
        _put_cells(grid, [(0, 0), (17, 17), (9, 0)], 5)
    elif variant == "D9/1":
        noise = [(0, 0, 2), (17, 17, 3), (9, 0, 4), (0, 9, 6)]
        for r, c, col in noise:
            grid[r, c] = col
    else:
        for r, c in _background_noise_cells()[:2]:
            if variant not in {"D2/2", "D10/0"}:
                grid[r, c] = 1

    if variant == "D2/2":
        # no foreground-colored spurs; keep anomaly rule focused.
        pass
    elif variant == "D10/0":
        # use one spur in the color of the first region.
        grid[0, 0] = 1

    return grid


def generate_variant(variant: str) -> np.ndarray:
    for _ in range(50):
        inp = _build_input(variant)
        try:
            solve_variant(variant, inp)
            return inp
        except Exception:
            continue
    raise RuntimeError(f"failed to generate valid input for {variant}")


def make_variant(variant: str):
    def main(grid: np.ndarray) -> np.ndarray:
        return solve_variant(variant, np.array(grid, dtype=int))

    def generate_input() -> np.ndarray:
        return generate_variant(variant)

    main.__doc__ = _variant_doc(variant)
    return main, generate_input
