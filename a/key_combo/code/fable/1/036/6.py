from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
import random

import numpy as np


def _load_arc_utils():
    path = "/home/lixinyu/snap/new/solvers/arc_utils.py"
    try:
        spec = spec_from_file_location("_arc_utils_shared", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


class _FallbackColor:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


Color = _ARC_UTILS.Color if _ARC_UTILS is not None else _FallbackColor


def get_bg(grid: np.ndarray) -> int:
    if _ARC_UTILS is not None:
        return int(_ARC_UTILS.get_bg(grid))
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    if _ARC_UTILS is not None:
        return bool(_ARC_UTILS.in_bounds(shape, r, c))
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    if int(connectivity) == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
    return_format: str = "dict",
) -> list[dict]:
    if _ARC_UTILS is not None:
        return _ARC_UTILS.find_components(
            grid,
            bg=bg,
            connectivity=connectivity,
            monochromatic=monochromatic,
            return_format=return_format,
        )
    if not monochromatic or return_format != "dict":
        raise ValueError("fallback find_components only supports monochromatic dict components")
    out: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        for coords in _components_mask(grid == color, connectivity=connectivity):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return out


@dataclass(frozen=True)
class Comb:
    color: int
    side: str
    center: tuple[int, int]
    direction: tuple[int, int]
    teeth_count: int
    body_cells: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class CombSpec:
    side: str
    color: int
    teeth_count: int
    pad_left: int
    pad_right: int
    center_axis: int


_SIDE_DIR = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

_PAD_CHOICES = [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)]
_COMB_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
_GEN_CALL_COUNT = 0


def _edge_len(shape: tuple[int, int], side: str) -> int:
    h, w = map(int, shape)
    return w if side in {"top", "bottom"} else h


def _depth_len(shape: tuple[int, int], side: str) -> int:
    h, w = map(int, shape)
    return h if side in {"top", "bottom"} else w


def _spine_len(spec: CombSpec) -> int:
    return 2 * int(spec.teeth_count) - 1 + int(spec.pad_left) + int(spec.pad_right)


def _axis_center_bounds(shape: tuple[int, int], spec: CombSpec, *, corner_gap: int = 1) -> tuple[int, int]:
    edge = _edge_len(shape, spec.side)
    half = _spine_len(spec) // 2
    lo = half + int(corner_gap)
    hi = edge - 1 - half - int(corner_gap)
    return int(lo), int(hi)


def _axis_progression_positions(length: int, side: str, step: int) -> list[int]:
    vals = []
    if side in {"top", "left"}:
        cur = int(step)
        while cur < length - 1:
            vals.append(cur)
            cur += int(step)
    else:
        cur = length - 1 - int(step)
        while cur > 0:
            vals.append(cur)
            cur -= int(step)
    return vals


def _build_spec(shape: tuple[int, int], spec: CombSpec) -> dict:
    h, w = map(int, shape)
    length = _spine_len(spec)
    half = length // 2
    start = int(spec.center_axis) - half
    tooth_offsets = [int(spec.pad_left) + 2 * i for i in range(int(spec.teeth_count))]

    if spec.side == "top":
        center = (0, int(spec.center_axis))
        spine = {(0, start + j) for j in range(length)}
        teeth = {(1, start + off) for off in tooth_offsets}
    elif spec.side == "bottom":
        center = (h - 1, int(spec.center_axis))
        spine = {(h - 1, start + j) for j in range(length)}
        teeth = {(h - 2, start + off) for off in tooth_offsets}
    elif spec.side == "left":
        center = (int(spec.center_axis), 0)
        spine = {(start + j, 0) for j in range(length)}
        teeth = {(start + off, 1) for off in tooth_offsets}
    else:
        center = (int(spec.center_axis), w - 1)
        spine = {(start + j, w - 1) for j in range(length)}
        teeth = {(start + off, w - 2) for off in tooth_offsets}

    body = spine | teeth
    assert all(in_bounds(shape, r, c) for r, c in body)
    return {
        "color": int(spec.color),
        "side": spec.side,
        "teeth_count": int(spec.teeth_count),
        "center": center,
        "direction": _SIDE_DIR[spec.side],
        "spine": spine,
        "teeth": teeth,
        "body": body,
    }


def _edge_counts(coords: np.ndarray, shape: tuple[int, int]) -> dict[str, int]:
    h, w = map(int, shape)
    return {
        "top": int(np.sum(coords[:, 0] == 0)),
        "bottom": int(np.sum(coords[:, 0] == h - 1)),
        "left": int(np.sum(coords[:, 1] == 0)),
        "right": int(np.sum(coords[:, 1] == w - 1)),
    }


def _contiguous(vals: list[int]) -> bool:
    if not vals:
        return False
    return vals == list(range(vals[0], vals[-1] + 1))


def _extract_comb(comp: dict, shape: tuple[int, int]) -> Comb:
    coords = np.asarray(comp["coords"], dtype=int)
    counts = _edge_counts(coords, shape)
    side, side_count = max(counts.items(), key=lambda kv: kv[1])
    assert side_count >= 3, "a comb spine must occupy at least three edge cells"
    assert list(counts.values()).count(side_count) == 1, "comb must touch one dominant edge"

    h, w = map(int, shape)
    if side == "top":
        spine = coords[coords[:, 0] == 0]
        teeth = coords[coords[:, 0] != 0]
        assert teeth.size > 0 and np.all(teeth[:, 0] == 1)
        axis_vals = sorted(map(int, spine[:, 1]))
        tooth_vals = sorted(map(int, teeth[:, 1]))
        center = (0, axis_vals[len(axis_vals) // 2])
        direction = (1, 0)
    elif side == "bottom":
        spine = coords[coords[:, 0] == h - 1]
        teeth = coords[coords[:, 0] != h - 1]
        assert teeth.size > 0 and np.all(teeth[:, 0] == h - 2)
        axis_vals = sorted(map(int, spine[:, 1]))
        tooth_vals = sorted(map(int, teeth[:, 1]))
        center = (h - 1, axis_vals[len(axis_vals) // 2])
        direction = (-1, 0)
    elif side == "left":
        spine = coords[coords[:, 1] == 0]
        teeth = coords[coords[:, 1] != 0]
        assert teeth.size > 0 and np.all(teeth[:, 1] == 1)
        axis_vals = sorted(map(int, spine[:, 0]))
        tooth_vals = sorted(map(int, teeth[:, 0]))
        center = (axis_vals[len(axis_vals) // 2], 0)
        direction = (0, 1)
    else:
        spine = coords[coords[:, 1] == w - 1]
        teeth = coords[coords[:, 1] != w - 1]
        assert teeth.size > 0 and np.all(teeth[:, 1] == w - 2)
        axis_vals = sorted(map(int, spine[:, 0]))
        tooth_vals = sorted(map(int, teeth[:, 0]))
        center = (axis_vals[len(axis_vals) // 2], w - 1)
        direction = (0, -1)

    assert _contiguous(axis_vals), "spine must be a contiguous edge strip"
    assert len(axis_vals) % 2 == 1, "spine length must be odd so the center cell is unique"
    assert len(set(tooth_vals)) == len(tooth_vals), "each tooth is a singleton"
    assert set(tooth_vals).issubset(set(axis_vals)), "every tooth must attach to the spine"
    if len(tooth_vals) >= 2:
        diffs = np.diff(tooth_vals)
        assert int(np.min(diffs)) > 0
        assert len(set(map(int, diffs.tolist()))) == 1, "tooth positions must be equally spaced"

    return Comb(
        color=int(comp["color"]),
        side=side,
        center=center,
        direction=direction,
        teeth_count=len(tooth_vals),
        body_cells=frozenset(map(tuple, coords.tolist())),
    )


def _ray_cells(center: tuple[int, int], direction: tuple[int, int], step: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, center)
    dr, dc = map(int, direction)
    out = []
    mul = 1
    while True:
        r = r0 + mul * int(step) * dr
        c = c0 + mul * int(step) * dc
        if not in_bounds(shape, r, c):
            break
        out.append((int(r), int(c)))
        mul += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a periodic inward ray from each edge comb, using the tooth count as the step size.

    Input structure:
    - The background is black.
    - Several monochrome combs sit on the grid edges, each made of a contiguous
      edge-aligned spine plus equally spaced one-cell teeth pointing inward.
    - Different combs have different colors and tooth counts, and may appear on any edge.

    Rule:
    - For each monochrome comb, count its teeth to obtain k and take the center cell of its spine.
    - Starting from that spine center, move inward along the tooth direction and place dots
      at distances k, 2k, 3k, ... until leaving the grid.
    - Keep every comb body unchanged; if dots from different combs land on the same empty cell,
      recolor that cell pink.

    Key insight:
    - The teeth encode both the ray direction and the numeric period, so the visible output
      is the set of sampled beat positions rather than a fully drawn line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == Color.BLACK, "background must be black"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one comb"
    combs = [_extract_comb(comp, g.shape) for comp in comps]

    hits: dict[tuple[int, int], set[int]] = defaultdict(set)
    for comb in combs:
        for cell in _ray_cells(comb.center, comb.direction, comb.teeth_count, g.shape):
            hits[cell].add(comb.color)

    out = g.copy()
    for (r, c), colors in hits.items():
        if int(g[r, c]) != bg:
            continue
        out[r, c] = Color.PINK if len(colors) >= 2 else next(iter(colors))
    return out


def _random_spec(
    shape: tuple[int, int],
    side: str,
    color: int,
    teeth_count: int,
    *,
    fixed_center: int | None = None,
) -> CombSpec | None:
    pad_choices = list(_PAD_CHOICES)
    random.shuffle(pad_choices)
    for pad_left, pad_right in pad_choices:
        spec = CombSpec(side, color, teeth_count, pad_left, pad_right, -1)
        lo, hi = _axis_center_bounds(shape, spec)
        if lo > hi:
            continue
        if fixed_center is None:
            center = random.randint(lo, hi)
        else:
            if not (lo <= int(fixed_center) <= hi):
                continue
            center = int(fixed_center)
        return CombSpec(side, color, teeth_count, pad_left, pad_right, center)
    return None


def _render_specs(shape: tuple[int, int], specs: list[CombSpec]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        built = _build_spec(shape, spec)
        for r, c in built["body"]:
            assert grid[r, c] == 0
            grid[r, c] = int(spec.color)
    return grid


def _analyze_specs(shape: tuple[int, int], specs: list[CombSpec]) -> dict:
    built = [_build_spec(shape, spec) for spec in specs]
    body_cells = set()
    for item in built:
        if body_cells & item["body"]:
            raise AssertionError("comb bodies overlap")
        body_cells |= item["body"]

    hits_by_cell: dict[tuple[int, int], set[int]] = defaultdict(set)
    for item in built:
        for cell in _ray_cells(item["center"], item["direction"], item["teeth_count"], shape):
            hits_by_cell[cell].add(item["color"])

    ray_body_overlap = {cell for cell in hits_by_cell if cell in body_cells}
    collision_cells = {cell for cell, colors in hits_by_cell.items() if len(colors) >= 2}
    return {
        "built": built,
        "body_cells": body_cells,
        "hits_by_cell": hits_by_cell,
        "collision_cells": collision_cells,
        "ray_body_overlap": ray_body_overlap,
    }


def _expected_output_from_specs(shape: tuple[int, int], specs: list[CombSpec]) -> np.ndarray:
    grid = _render_specs(shape, specs)
    out = grid.copy()
    analysis = _analyze_specs(shape, specs)
    for (r, c), colors in analysis["hits_by_cell"].items():
        if grid[r, c] != Color.BLACK:
            continue
        out[r, c] = Color.PINK if len(colors) >= 2 else next(iter(colors))
    return out


def _choose_shape() -> tuple[int, int]:
    return random.randint(17, 29), random.randint(17, 29)


def _choose_side_set(require_collision: bool | None) -> list[str]:
    sides = ["top", "bottom", "left", "right"]
    while True:
        count = random.randint(2, 4)
        chosen = random.sample(sides, count)
        if require_collision is True:
            if any(s in chosen for s in ("top", "bottom")) and any(s in chosen for s in ("left", "right")):
                return chosen
        else:
            return chosen


def _choose_teeth_counts(count: int) -> list[int]:
    vals = [2, 3, 4, 5, 6]
    return random.sample(vals, count)


def _collision_pair_specs(shape: tuple[int, int], sides: list[str], colors: list[int], ks: list[int]) -> list[CombSpec] | None:
    side_to_idx = {side: i for i, side in enumerate(sides)}
    vertical_sides = [side for side in sides if side in {"top", "bottom"}]
    horizontal_sides = [side for side in sides if side in {"left", "right"}]
    v_side = random.choice(vertical_sides)
    h_side = random.choice(horizontal_sides)
    v_idx = side_to_idx[v_side]
    h_idx = side_to_idx[h_side]

    for _ in range(40):
        v_stub = _random_spec(shape, v_side, colors[v_idx], ks[v_idx])
        h_stub = _random_spec(shape, h_side, colors[h_idx], ks[h_idx])
        if v_stub is None or h_stub is None:
            continue

        row_candidates = [
            pos
            for pos in _axis_progression_positions(shape[0], v_side, ks[v_idx])
            if _axis_center_bounds(shape, h_stub)[0] <= pos <= _axis_center_bounds(shape, h_stub)[1]
        ]
        col_candidates = [
            pos
            for pos in _axis_progression_positions(shape[1], h_side, ks[h_idx])
            if _axis_center_bounds(shape, v_stub)[0] <= pos <= _axis_center_bounds(shape, v_stub)[1]
        ]
        if not row_candidates or not col_candidates:
            continue

        row_center = random.choice(row_candidates)
        col_center = random.choice(col_candidates)
        v_spec = _random_spec(shape, v_side, colors[v_idx], ks[v_idx], fixed_center=col_center)
        h_spec = _random_spec(shape, h_side, colors[h_idx], ks[h_idx], fixed_center=row_center)
        if v_spec is None or h_spec is None:
            continue

        specs: list[CombSpec] = []
        for side, color, k in zip(sides, colors, ks):
            if side == v_side:
                specs.append(v_spec)
            elif side == h_side:
                specs.append(h_spec)
            else:
                spec = _random_spec(shape, side, color, k)
                if spec is None:
                    return None
                specs.append(spec)
        return specs
    return None


def _random_specs(shape: tuple[int, int], sides: list[str], colors: list[int], ks: list[int]) -> list[CombSpec] | None:
    specs = []
    for side, color, k in zip(sides, colors, ks):
        spec = _random_spec(shape, side, color, k)
        if spec is None:
            return None
        specs.append(spec)
    return specs


def _valid_case(shape: tuple[int, int], specs: list[CombSpec], require_collision: bool | None) -> bool:
    if len({spec.side for spec in specs}) != len(specs):
        return False

    analysis = _analyze_specs(shape, specs)
    if analysis["ray_body_overlap"]:
        return False

    collision_count = len(analysis["collision_cells"])
    if require_collision is True and collision_count != 1:
        return False
    if require_collision is False and collision_count != 0:
        return False

    grid = _render_specs(shape, specs)
    if np.sum(grid == Color.BLACK) <= grid.size // 2:
        return False

    expected = _expected_output_from_specs(shape, specs)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(require_collision: bool | None) -> np.ndarray | None:
    shape = _choose_shape()
    sides = _choose_side_set(require_collision)
    colors = random.sample(_COMB_COLORS, len(sides))
    ks = _choose_teeth_counts(len(sides))

    if require_collision is True:
        specs = _collision_pair_specs(shape, sides, colors, ks)
    else:
        specs = _random_specs(shape, sides, colors, ks)
    if specs is None:
        return None

    if not _valid_case(shape, specs, require_collision):
        return None
    return _render_specs(shape, specs)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    mode = _GEN_CALL_COUNT % 3
    if mode == 1:
        require_collision = True
    elif mode == 2:
        require_collision = False
    else:
        require_collision = random.random() < 0.55

    for _ in range(400):
        grid = _gen_attempt(require_collision)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid comb-ray puzzle")
