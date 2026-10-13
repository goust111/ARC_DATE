from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if UTILS_DIR.exists() and str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
OUTER = 9
GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (18, 24)
TARGET_SIDE_RANGE = (5, 8)
MIN_RAY_OUTSIDE = 2

BASE_INTERIOR = np.array(
    [
        [1, 4, 4, 3],
        [3, 1, 1, 4],
        [3, 1, 1, 4],
        [4, 3, 3, 1],
    ],
    dtype=int,
)
SIDES = ("top", "bottom", "left", "right")


def _line_runs(pattern: np.ndarray, axis: int) -> list[tuple[int, int, tuple[int, ...]]]:
    lines = pattern if int(axis) == 0 else pattern.T
    runs: list[tuple[int, int, tuple[int, ...]]] = []
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or not np.array_equal(lines[i], lines[start]):
            runs.append((start, i, tuple(map(int, lines[start]))))
            start = i
    return runs


def _stretch_run_index(runs: list[tuple[int, int, tuple[int, ...]]]) -> int:
    color_counts = [len(set(line)) for _start, _end, line in runs]
    max_count = max(color_counts)
    candidates = [i for i, count in enumerate(color_counts) if count == max_count]
    if len(candidates) == 1:
        return candidates[0]

    lengths = [end - start for start, end, _line in runs]
    max_length = max(lengths[i] for i in candidates)
    candidates = [i for i in candidates if lengths[i] == max_length]
    assert len(candidates) == 1, "expected one stretchable run"
    return candidates[0]


def _axis_indices(pattern: np.ndarray, axis: int, target_length: int) -> list[int]:
    runs = _line_runs(pattern, axis)
    stretch_i = _stretch_run_index(runs)
    assert 0 < stretch_i < len(runs) - 1, "stretchable run must be interior"
    fixed_length = sum(end - start for i, (start, end, _line) in enumerate(runs) if i != stretch_i)
    stretch_length = int(target_length) - fixed_length
    assert stretch_length > 0, "target is too small for the fixed bands"

    indices: list[int] = []
    for i, (start, end, _line) in enumerate(runs):
        if i == stretch_i:
            indices.extend([start] * stretch_length)
        else:
            indices.extend(range(start, end))
    assert len(indices) == int(target_length)
    return indices


def _stretch_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row_indices = _axis_indices(pattern, 0, shape[0])
    col_indices = _axis_indices(pattern, 1, shape[1])
    return pattern[np.ix_(row_indices, col_indices)]


def _is_exemplar_patch(patch: np.ndarray, bg: int) -> bool:
    h, w = patch.shape
    if h < 4 or w < 4 or np.any(patch == int(bg)):
        return False
    outer = int(patch[0, 0])
    if not all(
        int(x) == outer for x in (patch[0, 0], patch[0, -1], patch[-1, 0], patch[-1, -1])
    ):
        return False
    if not np.all(patch[0, :] == outer):
        return False
    if not np.all(patch[-1, :] == outer):
        return False
    if not np.all(patch[:, 0] == outer):
        return False
    if not np.all(patch[:, -1] == outer):
        return False
    inner = patch[1:-1, 1:-1]
    return len(np.unique(inner)) >= 3 and outer not in set(map(int, inner.ravel()))


def _target_gap(patch: np.ndarray, bg: int) -> tuple[tuple[int, int], int] | None:
    h, w = patch.shape
    if h < 5 or w < 5:
        return None
    if np.any(patch[1:-1, 1:-1] == int(bg)):
        return None

    holes: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(patch[r, c]) != int(bg):
                continue
            if 0 < r < h - 1 and 0 < c < w - 1:
                return None
            if r in (0, h - 1) and c in (0, w - 1):
                return None
            holes.append((int(r), int(c)))

    if len(holes) != 1:
        return None

    non_bg = [int(v) for v in patch.ravel() if int(v) != int(bg)]
    if len(non_bg) != h * w - 1:
        return None
    colors = set(non_bg)
    if len(colors) != 1:
        return None
    return holes[0], next(iter(colors))


def _gap_side(local_gap: tuple[int, int], h: int, w: int) -> tuple[str, tuple[int, int]]:
    r, c = map(int, local_gap)
    if r == 0:
        return "top", (-1, 0)
    if r == h - 1:
        return "bottom", (1, 0)
    if c == 0:
        return "left", (0, -1)
    assert c == w - 1
    return "right", (0, 1)


def _sample_gap_color(stretched: np.ndarray, local_gap: tuple[int, int], outer: int) -> int:
    gh, gw = stretched.shape
    r, c = map(int, local_gap)
    if r == 0:
        color = int(stretched[0, c - 1])
    elif r == gh + 1:
        color = int(stretched[-1, c - 1])
    elif c == 0:
        color = int(stretched[r - 1, 0])
    else:
        color = int(stretched[r - 1, -1])
    assert color != int(outer)
    return color


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) >= 3, "expected one exemplar and multiple targets"

    regions = []
    for comp in components:
        bbox = tuple(map(int, comp["bbox"]))
        regions.append({"bbox": bbox, "patch": crop_bbox(g, bbox)})

    exemplar_regions = [region for region in regions if _is_exemplar_patch(region["patch"], bg)]
    assert len(exemplar_regions) == 1, "expected exactly one exemplar"
    exemplar_region = exemplar_regions[0]
    exemplar = exemplar_region["patch"]
    outer = int(exemplar[0, 0])
    inner = exemplar[1:-1, 1:-1]

    targets = []
    for region in regions:
        if region["bbox"] == exemplar_region["bbox"]:
            continue
        gap_info = _target_gap(region["patch"], bg)
        assert gap_info is not None, "every non-exemplar region must be a gapped target rectangle"
        local_gap, target_color = gap_info
        assert int(target_color) == outer, "targets must use the exemplar border color"
        targets.append(
            {
                "bbox": region["bbox"],
                "patch": region["patch"],
                "local_gap": tuple(map(int, local_gap)),
            }
        )

    assert targets, "expected at least one target rectangle"
    targets.sort(key=lambda item: item["bbox"])
    return {
        "bg": bg,
        "outer": outer,
        "exemplar_bbox": exemplar_region["bbox"],
        "exemplar": exemplar,
        "inner": inner,
        "targets": targets,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped solid rectangle with a stretched copy of the exemplar interior, then fire a color ray from the gap.

    Input structure:
    - The background is black and separates one multicolor exemplar rectangle from several solid outer-color target rectangles.
    - The exemplar has a maroon border and a blue/yellow/green interior whose identical row and column runs contain one uniquely stretchable interior band per axis.
    - Each target is a solid maroon rectangle with exactly one missing non-corner border pixel.

    Rule:
    - Identify the exemplar and remove its maroon border to obtain the interior template.
    - Run-length encode identical rows and columns of that interior; in each axis, keep all non-stretch runs fixed and resize the unique interior run to match the target interior size.
    - Paste the stretched template into each target interior, keep the remaining maroon border unchanged, sample the gap color from the adjacent stretched boundary cell on the same side, then color the gap and every background cell straight outward from it with that sampled color.

    Key insight:
    - The ray color is not stored directly on the target rectangle; it is revealed only after elastic template stretching aligns the gap with a specific boundary position of the resized exemplar interior.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    out = g.copy()
    h_grid, w_grid = g.shape

    for target in scene["targets"]:
        r0, c0, r1, c1 = map(int, target["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        stretched = _stretch_pattern(scene["inner"], (h - 2, w - 2))
        paste(out, stretched, r0 + 1, c0 + 1)

        local_gap = target["local_gap"]
        side, (dr, dc) = _gap_side(local_gap, h, w)
        del side
        color = _sample_gap_color(stretched, local_gap, scene["outer"])

        r, c = r0 + local_gap[0], c0 + local_gap[1]
        while 0 <= r < h_grid and 0 <= c < w_grid:
            assert int(g[r, c]) == scene["bg"], "ray path must be empty in the input"
            out[r, c] = color
            r += dr
            c += dc

    return out


def _transform_inner() -> np.ndarray:
    inner = BASE_INTERIOR.copy()
    if random.random() < 0.5:
        inner = np.flipud(inner)
    if random.random() < 0.5:
        inner = np.fliplr(inner)
    if random.random() < 0.5:
        inner = inner.T

    perm_src = [1, 3, 4]
    perm_dst = perm_src[:]
    random.shuffle(perm_dst)
    mapped = inner.copy()
    for src, dst in zip(perm_src, perm_dst):
        mapped[inner == int(src)] = int(dst)
    return mapped


def _make_exemplar(inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    exemplar = np.full((h + 2, w + 2), OUTER, dtype=int)
    exemplar[1:-1, 1:-1] = inner
    return exemplar


def _mark_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    h, w = mask.shape
    rr0 = max(0, r0 - int(pad))
    cc0 = max(0, c0 - int(pad))
    rr1 = min(h - 1, r1 + int(pad))
    cc1 = min(w - 1, c1 + int(pad))
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = map(int, start)
    dr, dc = map(int, direction)
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _place_target_spec(
    shape: tuple[int, int],
    place_block: np.ndarray,
    occupied_bbox: np.ndarray,
    ray_block: np.ndarray,
    h: int,
    w: int,
    local_gap: tuple[int, int],
) -> dict | None:
    grid_h, grid_w = shape
    side, direction = _gap_side(local_gap, h, w)

    r_min, r_max = 1, grid_h - h - 1
    c_min, c_max = 1, grid_w - w - 1
    if side == "top":
        r_min = MIN_RAY_OUTSIDE
    elif side == "bottom":
        r_max = grid_h - h - MIN_RAY_OUTSIDE
    elif side == "left":
        c_min = MIN_RAY_OUTSIDE
    else:
        c_max = grid_w - w - MIN_RAY_OUTSIDE

    if r_min > r_max or c_min > c_max:
        return None

    candidates: list[dict] = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            if np.any(place_block[r0 : r0 + h, c0 : c0 + w]):
                continue
            if np.any(ray_block[r0 : r0 + h, c0 : c0 + w]):
                continue

            gap_abs = (r0 + local_gap[0], c0 + local_gap[1])
            ray = _ray_cells(shape, gap_abs, direction)
            outside = ray[1:]
            if len(outside) < MIN_RAY_OUTSIDE:
                continue
            if any(bool(occupied_bbox[rr, cc]) or bool(ray_block[rr, cc]) for rr, cc in outside):
                continue

            candidates.append(
                {
                    "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                    "local_gap": tuple(map(int, local_gap)),
                    "side": side,
                    "outside_ray": outside,
                }
            )

    if not candidates:
        return None
    return random.choice(candidates)


def _random_gap(side: str, h: int, w: int) -> tuple[int, int]:
    if side in ("top", "bottom"):
        cols = list(dict.fromkeys([1, w // 2, w - 2, random.randint(1, w - 2)]))
        c = random.choice(cols)
        return (0, c) if side == "top" else (h - 1, c)

    rows = list(dict.fromkeys([1, h // 2, h - 2, random.randint(1, h - 2)]))
    r = random.choice(rows)
    return (r, 0) if side == "left" else (r, w - 1)


def _pair_specs(side: str) -> list[tuple[int, int, tuple[int, int]]]:
    if side in ("top", "bottom"):
        w_small = random.choice([5, 6])
        w_large = random.randint(w_small + 2, TARGET_SIDE_RANGE[1])
        h1 = random.randint(*TARGET_SIDE_RANGE)
        h2 = random.randint(*TARGET_SIDE_RANGE)
        offset = w_small - 2
        gap1 = (0, offset) if side == "top" else (h1 - 1, offset)
        gap2 = (0, offset) if side == "top" else (h2 - 1, offset)
        return [(h1, w_small, gap1), (h2, w_large, gap2)]

    h_small = random.choice([5, 6])
    h_large = random.randint(h_small + 2, TARGET_SIDE_RANGE[1])
    w1 = random.randint(*TARGET_SIDE_RANGE)
    w2 = random.randint(*TARGET_SIDE_RANGE)
    offset = h_small - 2
    gap1 = (offset, 0) if side == "left" else (offset, w1 - 1)
    gap2 = (offset, 0) if side == "left" else (offset, w2 - 1)
    return [(h_small, w1, gap1), (h_large, w2, gap2)]


def _render_expected(grid: np.ndarray, inner: np.ndarray, targets: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for target in targets:
        r0, c0, r1, c1 = map(int, target["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        stretched = _stretch_pattern(inner, (h - 2, w - 2))
        paste(out, stretched, r0 + 1, c0 + 1)
        color = _sample_gap_color(stretched, target["local_gap"], OUTER)
        r, c = r0 + target["local_gap"][0], c0 + target["local_gap"][1]
        dr, dc = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}[target["side"]]
        while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = color
            r += dr
            c += dc
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray, inner: np.ndarray, targets: list[dict]) -> bool:
    try:
        scene = _parse_scene(inp)
    except Exception:
        return False
    if len(scene["targets"]) != len(targets):
        return False

    colors = set()
    for target in targets:
        r0, c0, r1, c1 = map(int, target["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        stretched = _stretch_pattern(inner, (h - 2, w - 2))
        if not np.array_equal(out[r0 + 1 : r1, c0 + 1 : c1], stretched):
            return False

        gap_r = r0 + target["local_gap"][0]
        gap_c = c0 + target["local_gap"][1]
        color = _sample_gap_color(stretched, target["local_gap"], OUTER)
        colors.add(color)

        border = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        border[1:-1, 1:-1] = OUTER
        border[target["local_gap"]] = OUTER
        if not np.all(border == OUTER):
            return False

        for rr, cc in [(gap_r, gap_c), *target["outside_ray"]]:
            if int(out[rr, cc]) != color:
                return False

    if len(colors) < 2:
        return False

    same_side_pairs = 0
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            if targets[i]["side"] == targets[j]["side"] and targets[i]["ray_color"] != targets[j]["ray_color"]:
                same_side_pairs += 1
    return same_side_pairs >= 1


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    inner = _transform_inner()
    exemplar = _make_exemplar(inner)
    ex_r = random.randint(1, 2)
    ex_c = random.randint(1, 2)

    place_block = np.zeros((grid_h, grid_w), dtype=bool)
    occupied_bbox = np.zeros((grid_h, grid_w), dtype=bool)
    ray_block = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_bbox(place_block, (ex_r, ex_c, ex_r + exemplar.shape[0] - 1, ex_c + exemplar.shape[1] - 1), pad=1)
    _mark_bbox(occupied_bbox, (ex_r, ex_c, ex_r + exemplar.shape[0] - 1, ex_c + exemplar.shape[1] - 1))

    target_count = 2 if random.random() < 0.45 else 3
    pair_side = random.choice(SIDES)
    planned = _pair_specs(pair_side)
    if target_count == 3:
        extra_side_choices = [side for side in SIDES if side != pair_side]
        extra_side = random.choice(extra_side_choices)
        extra_h = random.randint(*TARGET_SIDE_RANGE)
        extra_w = random.randint(*TARGET_SIDE_RANGE)
        planned.append((extra_h, extra_w, _random_gap(extra_side, extra_h, extra_w)))

    planned.sort(key=lambda spec: spec[0] * spec[1], reverse=True)
    targets: list[dict] = []
    for h, w, local_gap in planned:
        placed = _place_target_spec((grid_h, grid_w), place_block, occupied_bbox, ray_block, h, w, local_gap)
        if placed is None:
            return None
        stretched = _stretch_pattern(inner, (h - 2, w - 2))
        color = _sample_gap_color(stretched, local_gap, OUTER)
        placed["ray_color"] = color
        targets.append(placed)
        _mark_bbox(place_block, placed["bbox"], pad=1)
        _mark_bbox(occupied_bbox, placed["bbox"])
        for rr, cc in placed["outside_ray"]:
            ray_block[rr, cc] = True

    if len({target["ray_color"] for target in targets}) < 2:
        return None

    pair_targets = [target for target in targets if target["side"] == pair_side]
    if len(pair_targets) < 2:
        return None
    if len({target["ray_color"] for target in pair_targets}) < 2:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    paste(grid, exemplar, ex_r, ex_c)
    for target in targets:
        r0, c0, r1, c1 = map(int, target["bbox"])
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), OUTER, dtype=int)
        patch[target["local_gap"]] = BG
        paste(grid, patch, r0, c0)

    expected = _render_expected(grid, inner, targets)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid, expected, inner, targets):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
