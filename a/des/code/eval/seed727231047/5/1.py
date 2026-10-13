from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity in (4, 8)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def monochrome_components(grid: np.ndarray, active_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not bool(active_mask[r, c]) or bool(seen[r, c]):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(active_mask[nr, nc]) and not bool(seen[nr, nc]):
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append({"color": color, "coords": np.array(coords, dtype=int)})
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def is_rect_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not np.all(local[0, :]) or not np.all(local[-1, :]):
        return False
    if not np.all(local[:, 0]) or not np.all(local[:, -1]):
        return False
    if h > 2 and w > 2 and bool(local[1:-1, 1:-1].any()):
        return False
    return True


def find_palette(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, list[int]]] = []
    for row in (0, h - 1):
        nz = np.flatnonzero(grid[row] != bg)
        if nz.size == 0:
            continue
        c0 = int(nz[0])
        c1 = int(nz[-1])
        if c1 - c0 + 1 != int(nz.size):
            continue
        colors = [int(v) for v in grid[row, c0 : c1 + 1]]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((row, c0, c1, colors))

    assert len(candidates) == 1, "expected exactly one edge palette strip"
    row, c0, c1, colors = candidates[0]
    mask = np.zeros((h, w), dtype=bool)
    mask[row, c0 : c1 + 1] = True
    return mask, colors


def depth_by_color_from_rings(rings: list[dict]) -> dict[int, int]:
    bboxes_by_color: dict[int, set[tuple[int, int, int, int]]] = defaultdict(set)
    for ring in rings:
        bboxes_by_color[int(ring["color"])].add(tuple(map(int, ring["bbox"])))

    depth_by_color: dict[int, int] = {}
    for color, bbox_set in bboxes_by_color.items():
        depths: set[int] = set()
        for bbox in bbox_set:
            parent = (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
            if parent in bbox_set:
                continue
            depth = 1
            cur = bbox
            while True:
                child = (cur[0] + 2, cur[1] + 2, cur[2] - 2, cur[3] - 2)
                if child in bbox_set:
                    depth += 1
                    cur = child
                else:
                    break
            depths.add(depth)
        assert depths, "each container color needs at least one outer ring"
        assert len(depths) == 1, "one border color cannot encode multiple depths"
        depth_by_color[color] = int(next(iter(depths)))
    return depth_by_color


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    palette_mask, palette_colors = find_palette(g, bg)
    active = (g != bg) & (~palette_mask)
    components = monochrome_components(g, active)

    rings: list[dict] = []
    markers: list[dict] = []
    for comp in components:
        coords = comp["coords"]
        if int(coords.shape[0]) == 1:
            markers.append(comp)
            continue
        assert is_rect_ring(coords), "non-palette foreground must be a ring or singleton marker"
        rings.append({"color": int(comp["color"]), "bbox": bbox_coords(coords)})

    depth_by_color = depth_by_color_from_rings(rings)
    assert depth_by_color, "expected at least one container"
    assert max(depth_by_color.values()) == len(palette_colors), "palette length must equal max depth"

    out = g.copy()
    for marker in markers:
        color = int(marker["color"])
        if color not in depth_by_color:
            continue
        depth = int(depth_by_color[color])
        r, c = map(int, marker["coords"][0])
        out[r, c] = int(palette_colors[depth - 1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map each container's nesting depth to the same-position palette color,
    then repaint all isolated same-color markers with that mapped color.
    """
    return _solve(grid)


def draw_rect_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def bbox_free(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    h, w = blocked.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def feasible_positions(blocked: np.ndarray, box_h: int, box_w: int, pad: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    out: list[tuple[int, int]] = []
    for r0 in range(1, h - box_h - 1):
        r1 = r0 + box_h - 1
        for c0 in range(1, w - box_w - 1):
            c1 = c0 + box_w - 1
            if bbox_free(blocked, r0, c0, r1, c1, pad=pad):
                out.append((r0, c0))
    return out


def paint_container(grid: np.ndarray, r0: int, c0: int, depth: int, center_h: int, center_w: int, color: int) -> tuple[int, int, int, int]:
    outer_h = center_h + 4 * depth - 2
    outer_w = center_w + 4 * depth - 2
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    for layer in range(depth):
        inset = 2 * layer
        draw_rect_ring(grid, r0 + inset, c0 + inset, r1 - inset, c1 - inset, color)
    return r0, c0, r1, c1


GEN_RETRIES = 300


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    max_depth = rng.randint(1, 4)
    bg = rng.randrange(10)
    palette_side = rng.choice(["top", "bottom"])

    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    border_colors = colors[:max_depth]
    palette_colors = colors[max_depth : 2 * max_depth]

    specs = []
    for depth, border_color in zip(range(1, max_depth + 1), border_colors):
        center_h = rng.randint(1, 3)
        center_w = rng.randint(1, 3)
        outer_h = center_h + 4 * depth - 2
        outer_w = center_w + 4 * depth - 2
        specs.append(
            {
                "depth": depth,
                "color": border_color,
                "center_h": center_h,
                "center_w": center_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )

    base_h = 16 + 4 * max_depth
    base_w = 18 + 4 * max_depth
    h = rng.randint(base_h, min(34, base_h + 2))
    w = rng.randint(base_w, min(36, base_w + 2))
    if h < 7 or w < max_depth + 2:
        return None

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    palette_row = 0 if palette_side == "top" else h - 1
    buffer_row = 1 if palette_side == "top" else h - 2
    blocked[palette_row, :] = True
    blocked[buffer_row, :] = True
    start_c = rng.randint(0, w - max_depth)
    for i, color in enumerate(palette_colors):
        grid[palette_row, start_c + i] = color

    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        if h - spec["outer_h"] - 2 < 1 or w - spec["outer_w"] - 2 < 1:
            return None

        positions = feasible_positions(blocked, int(spec["outer_h"]), int(spec["outer_w"]), pad=1)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        bbox = paint_container(
            grid,
            r0,
            c0,
            int(spec["depth"]),
            int(spec["center_h"]),
            int(spec["center_w"]),
            int(spec["color"]),
        )
        mark_blocked(blocked, *bbox, pad=1)
        placed.append({"color": int(spec["color"]), "depth": int(spec["depth"]), "bbox": bbox})

    rng.shuffle(placed)
    for container in placed:
        marker_count = rng.randint(1, 3)
        for _ in range(marker_count):
            ok = False
            for _ in range(250):
                r = rng.randint(1, h - 2)
                c = rng.randint(1, w - 2)
                if not bbox_free(blocked, r, c, r, c, pad=1):
                    continue
                grid[r, c] = int(container["color"])
                mark_blocked(blocked, r, c, r, c, pad=1)
                ok = True
                break
            if not ok:
                return None

    try:
        out = _solve(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
