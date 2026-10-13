from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
INK_COLORS = [1, 2, 3, 4, 6, 7, 8]

GEN_ATTEMPTS = 400
MIN_PAIRS = 2
MAX_PAIRS = 4
PLACEMENT_GAP = 2
LAYOUTS = ("tb", "bt", "lr", "rl")


def _neighbors(r: int, c: int, connectivity: int) -> Iterable[tuple[int, int]]:
    if connectivity == 4:
        dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        dirs = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    for dr, dc in dirs:
        yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(pts)
    return parts


def _normalize(points: Iterable[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in points]
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    min_r, min_c = min(rs), min(cs)
    max_r, max_c = max(rs), max(cs)
    norm = tuple(sorted((r - min_r, c - min_c) for r, c in pts))
    return norm, (max_r - min_r + 1, max_c - min_c + 1)


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = [(int(r), int(c)) for r, c in points]
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _is_canvas_component(grid: np.ndarray, pts: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(pts)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    for c in range(c0, c1 + 1):
        if grid[r0, c] != MAROON or grid[r1, c] != MAROON:
            return False
    for r in range(r0, r1 + 1):
        if grid[r, c0] != MAROON or grid[r, c1] != MAROON:
            return False
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == MAROON):
        return False
    return len(pts) == 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find stamp-pad pairs, route them through matching canvas markers, and stamp
    the grey shape into each maroon frame using the corresponding ink color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()

    stamp_parts = _components(g == GREY, connectivity=8)
    stamps: list[dict[str, object]] = []
    for pts in stamp_parts:
        shape, shape_size = _normalize(pts)
        shape_set = set(shape)
        assert (0, 0) in shape_set
        stamps.append({"pts": pts, "shape": shape, "size": shape_size})
    assert stamps

    pad_by_color: dict[int, list[tuple[int, int]]] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GREY, MAROON):
            continue
        comps = _components(g == color, connectivity=8)
        pad_candidates = []
        for pts in comps:
            if len(pts) != 4:
                continue
            r0, c0, r1, c1 = _bbox(pts)
            if (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
                pad_candidates.append(pts)
        assert len(pad_candidates) == 1
        pad_by_color[color] = pad_candidates[0]

    shape_by_color: dict[int, tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = {}
    for color, pad_pts in pad_by_color.items():
        matches = []
        pad_set = {(int(r), int(c)) for r, c in pad_pts}
        for stamp in stamps:
            stamp_pts = stamp["pts"]
            adjacent = False
            for sr, sc in stamp_pts:
                for pr, pc in pad_set:
                    if max(abs(sr - pr), abs(sc - pc)) == 1:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent:
                matches.append(stamp)
        assert len(matches) == 1
        matched = matches[0]
        shape_by_color[color] = (matched["shape"], matched["size"])  # type: ignore[index]

    canvas_parts = _components(g == MAROON, connectivity=4)
    canvases = [pts for pts in canvas_parts if _is_canvas_component(g, pts)]
    assert len(canvases) == len(shape_by_color)

    for pts in canvases:
        r0, c0, r1, c1 = _bbox(pts)
        marker_r, marker_c = r0 + 1, c0 + 1
        ink_color = int(g[marker_r, marker_c])
        assert ink_color in shape_by_color
        shape, (shape_h, shape_w) = shape_by_color[ink_color]
        assert marker_r + shape_h - 1 <= r1 - 1
        assert marker_c + shape_w - 1 <= c1 - 1
        for dr, dc in shape:
            out[marker_r + dr, marker_c + dc] = ink_color
    return out


def _random_shape(
    rng: random.Random, used: set[tuple[tuple[int, int], ...]]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]] | None:
    for _ in range(120):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        min_cells = max(3, h + w - 2)
        max_cells = min(6, h * w - 1)
        if max_cells < min_cells:
            continue
        target = rng.randint(min_cells, max_cells)
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0)}
        frontier = {(r, c) for r, c in frontier if r < h and c < w}
        while len(cells) < target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            cr, cc = cell
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != target:
            continue
        shape, shape_size = _normalize(cells)
        if shape_size != (h, w):
            continue
        if (0, 0) not in shape:
            continue
        if shape in used:
            continue
        used.add(shape)
        return shape, shape_size
    return None


def _pair_object(
    shape: tuple[tuple[int, int], ...], color: int, rng: random.Random
) -> dict[str, object] | None:
    _, (shape_h, shape_w) = _normalize(shape)
    shape_set = set(shape)
    candidates: list[tuple[int, int]] = []
    for pr in range(-2, shape_h + 1):
        for pc in range(-2, shape_w + 1):
            pad_cells = {(pr + dr, pc + dc) for dr in range(2) for dc in range(2)}
            if pad_cells & shape_set:
                continue
            if 0 <= pr and pr + 1 < shape_h and 0 <= pc and pc + 1 < shape_w:
                continue
            adjacent = False
            for sr, sc in shape_set:
                for rr, cc in pad_cells:
                    if max(abs(sr - rr), abs(sc - cc)) == 1:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent:
                candidates.append((pr, pc))
    if not candidates:
        return None
    pr, pc = rng.choice(candidates)
    pad_cells = {(pr + dr, pc + dc) for dr in range(2) for dc in range(2)}
    all_cells = list(shape_set | pad_cells)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    shape_cells = tuple(sorted((r - min_r, c - min_c) for r, c in shape_set))
    pad_rel = tuple(sorted((r - min_r, c - min_c) for r, c in pad_cells))
    _, (obj_h, obj_w) = _normalize(all_cells)
    return {
        "shape": shape_cells,
        "pad": pad_rel,
        "height": obj_h,
        "width": obj_w,
        "color": color,
    }


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = blocked.shape
    r0 = max(0, top - PLACEMENT_GAP)
    c0 = max(0, left - PLACEMENT_GAP)
    r1 = min(h, top + height + PLACEMENT_GAP)
    c1 = min(w, left + width + PLACEMENT_GAP)
    blocked[r0:r1, c0:c1] = True


def _zone_positions(
    zone: tuple[int, int, int, int], height: int, width: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = zone
    if r1 - r0 < height or c1 - c0 < width:
        return []
    return [
        (r, c)
        for r in range(r0, r1 - height + 1)
        for c in range(c0, c1 - width + 1)
    ]


def _place_rect(
    blocked: np.ndarray,
    zone: tuple[int, int, int, int],
    height: int,
    width: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    positions = _zone_positions(zone, height, width)
    rng.shuffle(positions)
    for top, left in positions:
        if not blocked[top : top + height, left : left + width].any():
            _mark_blocked(blocked, top, left, height, width)
            return top, left
    return None


def _draw_pair(grid: np.ndarray, obj: dict[str, object], top: int, left: int) -> None:
    color = int(obj["color"])
    for dr, dc in obj["shape"]:  # type: ignore[index]
        grid[top + dr, left + dc] = GREY
    for dr, dc in obj["pad"]:  # type: ignore[index]
        grid[top + dr, left + dc] = color


def _draw_canvas(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top, left : left + width] = MAROON
    grid[top + height - 1, left : left + width] = MAROON
    grid[top : top + height, left] = MAROON
    grid[top : top + height, left + width - 1] = MAROON
    grid[top + 1, left + 1] = color


def _render_expected(
    grid: np.ndarray,
    canvas_specs: list[dict[str, object]],
    shape_by_color: dict[int, tuple[tuple[int, int], ...]],
) -> np.ndarray:
    out = grid.copy()
    for canvas in canvas_specs:
        color = int(canvas["color"])
        top = int(canvas["top"])
        left = int(canvas["left"])
        for dr, dc in shape_by_color[color]:
            out[top + 1 + dr, left + 1 + dc] = color
    return out


def _zones_for_layout(
    layout: str, h: int, w: int, rng: random.Random
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if layout in ("tb", "bt"):
        split = rng.randint(h // 2 - 2, h // 2 + 2)
        top_zone = (1, 1, split - 1, w - 1)
        bottom_zone = (split + 1, 1, h - 1, w - 1)
        if layout == "tb":
            return top_zone, bottom_zone
        return bottom_zone, top_zone
    split = rng.randint(w // 2 - 2, w // 2 + 2)
    left_zone = (1, 1, h - 1, split - 1)
    right_zone = (1, split + 1, h - 1, w - 1)
    if layout == "lr":
        return left_zone, right_zone
    return right_zone, left_zone


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.randint(MIN_PAIRS, MAX_PAIRS)
    layout = rng.choice(LAYOUTS)
    base_size = 20 + 3 * n_pairs
    h = rng.randint(base_size, base_size + 4)
    w = rng.randint(base_size, base_size + 6)

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    colors = rng.sample(INK_COLORS, n_pairs)
    pair_specs: list[dict[str, object]] = []
    shape_by_color: dict[int, tuple[tuple[int, int], ...]] = {}
    canvas_shapes: list[tuple[int, int, int, int]] = []

    for color in colors:
        made = _random_shape(rng, used_shapes)
        if made is None:
            return None
        shape, (shape_h, shape_w) = made
        pair_obj = _pair_object(shape, color, rng)
        if pair_obj is None:
            return None
        pair_specs.append(pair_obj)
        shape_by_color[color] = shape
        extra_h = rng.randint(0, 3)
        extra_w = rng.randint(0, 3)
        canvas_shapes.append((shape_h + extra_h + 2, shape_w + extra_w + 2, shape_h, shape_w))

    pair_specs.sort(key=lambda obj: int(obj["height"]) * int(obj["width"]), reverse=True)
    canvas_order = list(range(n_pairs))
    canvas_order.sort(key=lambda i: canvas_shapes[i][0] * canvas_shapes[i][1], reverse=True)

    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    pair_zone, canvas_zone = _zones_for_layout(layout, h, w, rng)

    for spec in pair_specs:
        pos = _place_rect(
            blocked,
            pair_zone,
            int(spec["height"]),
            int(spec["width"]),
            rng,
        )
        if pos is None:
            return None
        top, left = pos
        spec["top"] = top
        spec["left"] = left
        _draw_pair(grid, spec, top, left)

    canvas_specs: list[dict[str, object]] = []
    for idx in canvas_order:
        color = colors[idx]
        height, width, shape_h, shape_w = canvas_shapes[idx]
        pos = _place_rect(blocked, canvas_zone, height, width, rng)
        if pos is None:
            return None
        top, left = pos
        _draw_canvas(grid, top, left, height, width, color)
        canvas_specs.append(
            {
                "color": color,
                "top": top,
                "left": left,
                "height": height,
                "width": width,
                "shape_h": shape_h,
                "shape_w": shape_w,
            }
        )

    expected = _render_expected(grid, canvas_specs, shape_by_color)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
