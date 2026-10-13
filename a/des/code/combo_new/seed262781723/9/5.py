from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 300
GAP_CHOICES = (2, 4, 6)


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            rel = tuple(sorted((int(rr - r0), int(cc - c0)) for rr, cc in coords))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "shape_key": rel,
                }
            )
    return comps


def _pair_info(a: dict, b: dict, grid: np.ndarray) -> dict:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])

    if ar0 == br0 and ar1 == br1:
        left, right = (a, b) if ac0 < bc0 else (b, a)
        lr0, lc0, lr1, lc1 = map(int, left["bbox"])
        rr0, rc0, rr1, rc1 = map(int, right["bbox"])
        gap = int(rc0 - lc1 - 1)
        if gap > 0 and np.all(grid[lr0 : lr1 + 1, lc1 + 1 : rc0] == BG):
            return {"axis": "h", "first": left, "second": right, "gap": gap}

    if ac0 == bc0 and ac1 == bc1:
        top, bottom = (a, b) if ar0 < br0 else (b, a)
        tr0, tc0, tr1, tc1 = map(int, top["bbox"])
        br0, bc0, br1, bc1 = map(int, bottom["bbox"])
        gap = int(br0 - tr1 - 1)
        if gap > 0 and np.all(grid[tr1 + 1 : br0, tc0 : tc1 + 1] == BG):
            return {"axis": "v", "first": top, "second": bottom, "gap": gap}

    raise AssertionError("paired objects must share a straight black corridor")


def _fill_pair_channel(out: np.ndarray, info: dict) -> None:
    first = info["first"]
    second = info["second"]
    color_a = int(first["color"])
    color_b = int(second["color"])

    if info["axis"] == "h":
        r0, _c0, r1, c1 = map(int, first["bbox"])
        _r0b, c0b, _r1b, _c1b = map(int, second["bbox"])
        for offset, c in enumerate(range(c1 + 1, c0b)):
            out[r0 : r1 + 1, c] = color_a if offset % 2 == 0 else color_b
        return

    r0, c0, r1, _c1 = map(int, first["bbox"])
    r0b, _c0b, _r1b, c1b = map(int, second["bbox"])
    for offset, r in enumerate(range(r1 + 1, r0b)):
        out[r, c0 : c1b + 1] = color_a if offset % 2 == 0 else color_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the straight gap between each translated shape pair with alternating color stripes.

    Input structure:
    - The background is black.
    - Several 4-connected monochrome objects appear in pairs.
    - Within each pair, the two objects have exactly the same shape under translation only, but different colors.
    - Each pair is aligned either left-right or top-bottom with one unobstructed black corridor between them.

    Rule:
    - Find objects and group them by translation-invariant shape.
    - Each shape group contains exactly two objects; determine whether they are connected by a horizontal or vertical corridor.
    - Fill the black corridor rectangle between the pair with 1-pixel alternating stripes perpendicular to the corridor direction.
    - The stripe touching one object starts with that object's color, and the original objects stay unchanged.

    Key insight:
    - Colors and positions alone are insufficient; pairing must be recovered from exact shape congruence before the corridor can be striped.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in _components(g):
        groups[comp["shape_key"]].append(comp)

    assert groups, "expected at least one foreground pair"
    for pair in groups.values():
        assert len(pair) == 2, "each shape must appear exactly twice"
        assert int(pair[0]["color"]) != int(pair[1]["color"]), "paired objects must have different colors"
        info = _pair_info(pair[0], pair[1], g)
        _fill_pair_channel(out, info)

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_shape(existing: set[tuple[tuple[int, int], ...]]) -> tuple[tuple[int, int], ...] | None:
    for _ in range(200):
        if RNG.random() < 0.2:
            if RNG.random() < 0.5:
                h = 1
                w = int(RNG.integers(2, 6))
            else:
                h = int(RNG.integers(2, 6))
                w = 1
        else:
            h = int(RNG.integers(2, 6))
            w = int(RNG.integers(2, 6))

        max_area = min(h * w - 1 if h * w > 3 else h * w, 8)
        if max_area < 2:
            continue
        area = int(RNG.integers(2, max_area + 1))

        start = (int(RNG.integers(h)), int(RNG.integers(w)))
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(RNG.integers(len(frontier)))]
            cells.add(pick)

        if len(cells) != area:
            continue

        key = _normalize_shape(cells)
        shape_h = 1 + max(r for r, _ in key)
        shape_w = 1 + max(c for _, c in key)
        if len(key) == shape_h * shape_w and max(shape_h, shape_w) > 1 and RNG.random() < 0.8:
            continue
        if key in existing:
            continue
        return key
    return None


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _mark_used(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    used[rr0:rr1, cc0:cc1] = True


def _place_pair(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    shape = spec["shape"]
    shape_h = int(spec["shape_h"])
    shape_w = int(spec["shape_w"])
    gap = int(spec["gap"])
    color_a = int(spec["color_a"])
    color_b = int(spec["color_b"])

    if spec["axis"] == "h":
        left_r = top
        left_c = left
        right_r = top
        right_c = left + shape_w + gap
        for dr, dc in shape:
            grid[left_r + dr, left_c + dc] = color_a
            grid[right_r + dr, right_c + dc] = color_b
        return

    top_r = top
    top_c = left
    bottom_r = top + shape_h + gap
    bottom_c = left
    for dr, dc in shape:
        grid[top_r + dr, top_c + dc] = color_a
        grid[bottom_r + dr, bottom_c + dc] = color_b


def _build_specs(pair_count: int) -> list[dict] | None:
    specs: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    colors = RNG.choice(COLOR_POOL, size=2 * pair_count, replace=False).tolist()

    for idx in range(pair_count):
        shape = _random_shape(used_shapes)
        if shape is None:
            return None
        used_shapes.add(shape)

        shape_h = 1 + max(r for r, _ in shape)
        shape_w = 1 + max(c for _, c in shape)
        axis = "h" if RNG.random() < 0.5 else "v"
        gap = int(RNG.choice(GAP_CHOICES))

        if axis == "h":
            block_h = shape_h
            block_w = 2 * shape_w + gap
        else:
            block_h = 2 * shape_h + gap
            block_w = shape_w

        color_a, color_b = colors[2 * idx : 2 * idx + 2]
        if RNG.random() < 0.5:
            color_a, color_b = color_b, color_a

        specs.append(
            {
                "shape": shape,
                "shape_h": shape_h,
                "shape_w": shape_w,
                "axis": axis,
                "gap": gap,
                "block_h": block_h,
                "block_w": block_w,
                "color_a": color_a,
                "color_b": color_b,
            }
        )
    return specs


def _sample_canvas(specs: list[dict]) -> tuple[int, int]:
    max_h = max(int(spec["block_h"]) for spec in specs)
    max_w = max(int(spec["block_w"]) for spec in specs)
    footprint = sum((int(spec["block_h"]) + 2) * (int(spec["block_w"]) + 2) for spec in specs)
    side_hint = int(np.ceil(np.sqrt(footprint * 1.4)))

    min_h = max(10, max_h + 2, side_hint - 2)
    min_w = max(10, max_w + 2, side_hint - 2)
    min_h = min(min_h, 30)
    min_w = min(min_w, 30)

    h = int(RNG.integers(min_h, 31))
    w = int(RNG.integers(min_w, 31))
    return h, w


def _layout_grid(specs: list[dict]) -> np.ndarray | None:
    for _ in range(80):
        h, w = _sample_canvas(specs)
        grid = np.zeros((h, w), dtype=int)
        used = np.zeros((h, w), dtype=bool)
        order = RNG.permutation(len(specs))
        placements: dict[int, tuple[int, int]] = {}
        ok = True

        for idx in order:
            spec = specs[int(idx)]
            block_h = int(spec["block_h"])
            block_w = int(spec["block_w"])
            if h < block_h + 2 or w < block_w + 2:
                ok = False
                break

            candidates: list[tuple[int, int]] = []
            for _ in range(200):
                r0 = int(RNG.integers(1, h - block_h))
                c0 = int(RNG.integers(1, w - block_w))
                if _can_place(used, r0, c0, block_h, block_w, pad=1):
                    candidates.append((r0, c0))
                    if len(candidates) >= 12:
                        break
            if not candidates:
                ok = False
                break

            r0, c0 = candidates[int(RNG.integers(len(candidates)))]
            placements[int(idx)] = (r0, c0)
            _mark_used(used, r0, c0, block_h, block_w, pad=1)

        if not ok:
            continue

        for idx, spec in enumerate(specs):
            top, left = placements[idx]
            _place_pair(grid, top, left, spec)
        return grid
    return None


def _is_valid_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    comps = _components(grid)
    if len(comps) < 2 or len(comps) % 2 != 0:
        return False

    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in comps:
        groups[comp["shape_key"]].append(comp)

    if 2 * len(groups) != len(comps):
        return False

    for pair in groups.values():
        if len(pair) != 2:
            return False
        if int(pair[0]["color"]) == int(pair[1]["color"]):
            return False
        try:
            info = _pair_info(pair[0], pair[1], grid)
        except AssertionError:
            return False
        if int(info["gap"]) % 2 != 0:
            return False

    try:
        out = main(grid)
    except Exception:
        return False

    return out.shape == grid.shape and bool(np.any(out != grid))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair_count = int(RNG.integers(1, 4))
        specs = _build_specs(pair_count)
        if specs is None:
            continue
        grid = _layout_grid(specs)
        if grid is None:
            continue
        if _is_valid_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
