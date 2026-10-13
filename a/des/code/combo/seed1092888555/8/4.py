import math
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            touches = {
                "top": int(r0 == 0),
                "bottom": int(r1 == h - 1),
                "left": int(c0 == 0),
                "right": int(c1 == w - 1),
            }
            assert sum(touches.values()) == 1
            side = next(name for name, v in touches.items() if v)
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "area": int(len(coords)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "side": side,
                }
            )
    return comps


def _packed_block(area: int, thickness: int, color: int, vertical: bool) -> np.ndarray:
    assert area > 0 and thickness > 0
    long_dim = int(math.ceil(area / thickness))
    if vertical:
        block = np.full((long_dim, thickness), -1, dtype=int)
    else:
        block = np.full((thickness, long_dim), -1, dtype=int)
    flat = block.reshape(-1)
    flat[:area] = color
    return block


def _build_group_strip(group: list[dict], side: str) -> np.ndarray:
    if not group:
        return np.zeros((0, 0), dtype=int)

    if side in ("top", "bottom"):
        group = sorted(group, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
        parts = []
        thickness = 0
        total_w = 0
        for comp in group:
            r0, c0, r1, c1 = comp["bbox"]
            depth = r1 - r0 + 1
            part = _packed_block(comp["area"], depth, comp["color"], vertical=False)
            parts.append(part)
            thickness = max(thickness, part.shape[0])
            total_w += part.shape[1]
        strip = np.full((thickness, total_w), -1, dtype=int)
        cursor = 0
        for part in parts:
            ph, pw = part.shape
            if side == "top":
                strip[:ph, cursor : cursor + pw] = part
            else:
                strip[thickness - ph :, cursor : cursor + pw] = part
            cursor += pw
        return strip

    group = sorted(group, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    parts = []
    thickness = 0
    total_h = 0
    for comp in group:
        r0, c0, r1, c1 = comp["bbox"]
        depth = c1 - c0 + 1
        part = _packed_block(comp["area"], depth, comp["color"], vertical=True)
        parts.append(part)
        thickness = max(thickness, part.shape[1])
        total_h += part.shape[0]
    strip = np.full((total_h, thickness), -1, dtype=int)
    cursor = 0
    for part in parts:
        ph, pw = part.shape
        if side == "left":
            strip[cursor : cursor + ph, :pw] = part
        else:
            strip[cursor : cursor + ph, thickness - pw :] = part
        cursor += ph
    return strip


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components

    groups = {side: [] for side in ("top", "bottom", "left", "right")}
    for comp in components:
        groups[comp["side"]].append(comp)

    top = _build_group_strip(groups["top"], "top")
    bottom = _build_group_strip(groups["bottom"], "bottom")
    left = _build_group_strip(groups["left"], "left")
    right = _build_group_strip(groups["right"], "right")

    top_h, top_w = top.shape
    bottom_h, bottom_w = bottom.shape
    left_h, left_w = left.shape
    right_h, right_w = right.shape

    inner_h = max(left_h, right_h, 1)
    inner_w = max(top_w, bottom_w, 1)
    out_h = top_h + inner_h + bottom_h
    out_w = left_w + inner_w + right_w
    out = np.full((out_h, out_w), bg, dtype=int)

    if top_h:
        out[:top_h, left_w : left_w + top_w][top != -1] = top[top != -1]
    if bottom_h:
        region = out[out_h - bottom_h :, left_w : left_w + bottom_w]
        region[bottom != -1] = bottom[bottom != -1]
    if left_w:
        region = out[top_h : top_h + left_h, :left_w]
        region[left != -1] = left[left != -1]
    if right_w:
        region = out[top_h : top_h + right_h, out_w - right_w :]
        region[right != -1] = right[right != -1]
    return out


_BG = 5
_RETRY = 400


def _random_color(rng: np.random.Generator, forbidden: set[int]) -> int:
    palette = [c for c in range(10) if c not in forbidden]
    return int(rng.choice(palette))


def _edge_shape(rng: np.random.Generator, depth: int, span: int, side: str) -> np.ndarray:
    mask = np.zeros((depth, span), dtype=bool)
    if rng.random() < 0.4:
        mask[:, :] = True
        return mask

    if side in ("top", "bottom"):
        seeds = [(0, int(rng.integers(span)))]
    else:
        seeds = [(int(rng.integers(depth)), 0)]
    target = int(rng.integers(max(depth + span - 1, 3), depth * span + 1))

    for r, c in seeds:
        mask[r, c] = True
    frontier = list(seeds)
    while frontier and int(mask.sum()) < target:
        idx = int(rng.integers(len(frontier)))
        r, c = frontier[idx]
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < depth and 0 <= nc < span and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.pop(idx)
            continue
        nr, nc = nbrs[int(rng.integers(len(nbrs)))]
        mask[nr, nc] = True
        frontier.append((nr, nc))

    if side == "top":
        mask[0, :] |= np.any(mask, axis=0)
    elif side == "bottom":
        mask[-1, :] |= np.any(mask, axis=0)
    elif side == "left":
        mask[:, 0] |= np.any(mask, axis=1)
    else:
        mask[:, -1] |= np.any(mask, axis=1)
    return mask


def _place_top_or_bottom(
    grid: np.ndarray,
    rng: np.random.Generator,
    side: str,
    count: int,
    color_pool: list[int],
) -> None:
    h, w = grid.shape
    starts = []
    cursor = 1
    max_depth = max(2, h // 4)
    for i in range(count):
        remaining = count - i
        max_span = w - 2 - cursor - 2 * (remaining - 1)
        if max_span < 2:
            raise ValueError("no horizontal room")
        span = int(rng.integers(2, min(6, max_span) + 1))
        starts.append((cursor, span))
        cursor += span + int(rng.integers(1, 3))
    for c0, span in starts:
        depth = int(rng.integers(1, max_depth + 1))
        color = color_pool.pop()
        mask = _edge_shape(rng, depth, span, side)
        rows = range(depth) if side == "top" else range(h - depth, h)
        rr0 = 0 if side == "top" else h - depth
        for rr in range(depth):
            for cc in range(span):
                if mask[rr, cc]:
                    grid[rr0 + rr, c0 + cc] = color


def _place_left_or_right(
    grid: np.ndarray,
    rng: np.random.Generator,
    side: str,
    count: int,
    color_pool: list[int],
) -> None:
    h, w = grid.shape
    starts = []
    cursor = 1
    max_depth = max(2, w // 4)
    for i in range(count):
        remaining = count - i
        max_span = h - 2 - cursor - 2 * (remaining - 1)
        if max_span < 2:
            raise ValueError("no vertical room")
        span = int(rng.integers(2, min(6, max_span) + 1))
        starts.append((cursor, span))
        cursor += span + int(rng.integers(1, 3))
    for r0, span in starts:
        depth = int(rng.integers(1, max_depth + 1))
        color = color_pool.pop()
        shape_side = "left" if side == "left" else "right"
        mask = _edge_shape(rng, span, depth, shape_side)
        cc0 = 0 if side == "left" else w - depth
        for rr in range(span):
            for cc in range(depth):
                if mask[rr, cc]:
                    grid[r0 + rr, cc0 + cc] = color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_RETRY):
        h = int(rng.integers(10, 19))
        w = int(rng.integers(10, 19))
        grid = np.full((h, w), _BG, dtype=int)

        side_counts = {}
        active_sides = [side for side in ("top", "bottom", "left", "right") if rng.random() < 0.75]
        if len(active_sides) < 2:
            active_sides = list(rng.choice(["top", "bottom", "left", "right"], size=2, replace=False))
        for side in ("top", "bottom", "left", "right"):
            side_counts[side] = int(rng.integers(1, 4)) if side in active_sides else 0

        total = sum(side_counts.values())
        if total == 0 or total > 8:
            continue
        colors = list(rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9], size=total, replace=False))
        rng.shuffle(colors)

        try:
            if side_counts["top"]:
                _place_top_or_bottom(grid, rng, "top", side_counts["top"], colors)
            if side_counts["bottom"]:
                _place_top_or_bottom(grid, rng, "bottom", side_counts["bottom"], colors)
            if side_counts["left"]:
                _place_left_or_right(grid, rng, "left", side_counts["left"], colors)
            if side_counts["right"]:
                _place_left_or_right(grid, rng, "right", side_counts["right"], colors)
        except (ValueError, IndexError):
            continue

        try:
            comps = _find_components(grid, _BG)
        except AssertionError:
            continue
        if len(comps) != total:
            continue
        by_side = Counter(comp["side"] for comp in comps)
        if any(by_side[side] != side_counts[side] for side in side_counts):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
