from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(pts, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _outside_background(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == bg and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == bg and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    black = np.argwhere(grid == 0)
    if len(black) == 0:
        raise ValueError("legend border/dots must use black")
    r0, c0, r1, c1 = _bbox(black)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if sub.shape[0] < 3 or sub.shape[1] < 5:
        raise ValueError("legend frame is too small")
    if not (
        np.all(sub[0, :] == 0)
        and np.all(sub[-1, :] == 0)
        and np.all(sub[:, 0] == 0)
        and np.all(sub[:, -1] == 0)
    ):
        raise ValueError("black pixels must bound a rectangular legend frame")
    h, w = grid.shape
    in_corner = (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )
    if not in_corner:
        raise ValueError("legend must sit in a grid corner")
    return r0, c0, r1, c1


def _legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _legend_bbox(grid)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mapping: dict[int, int] = {}
    for row in inner:
        colored_cols = np.flatnonzero((row != bg) & (row != 0))
        if len(colored_cols) == 0:
            continue
        if len(colored_cols) != 1:
            raise ValueError("each legend row must contain exactly one color swatch")
        dot_count = int(np.sum(row == 0))
        if dot_count <= 0:
            raise ValueError("legend rows with a swatch must contain at least one dot")
        mapping[dot_count] = int(row[int(colored_cols[0])])
    if not mapping:
        raise ValueError("legend mapping is empty")
    return mapping, (r0, c0, r1, c1)


def _object_color(grid: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = legend_bbox
    mask = np.ones(grid.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    values = grid[mask]
    colors = [int(v) for v in values if int(v) not in (bg, 0)]
    if not colors:
        raise ValueError("expected at least one object color outside the legend")
    return Counter(colors).most_common(1)[0][0]


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    legend, legend_bbox = _legend_mapping(g, bg)
    obj_color = _object_color(g, bg, legend_bbox)
    outside_bg = _outside_background(g, bg)

    obj_mask = g == obj_color
    lr0, lc0, lr1, lc1 = legend_bbox
    obj_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = False

    out = g.copy()
    for comp in _components(obj_mask):
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True
        hole_count = _count_holes(local)
        if hole_count <= 0 or hole_count not in legend:
            continue
        ring_color = int(legend[hole_count])
        for r, c in comp:
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    if g[nr, nc] == bg and outside_bg[nr, nc]:
                        out[nr, nc] = ring_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each object outward by one colored ring chosen from a legend indexed by hole count.

    Input structure:
    - The background is the dominant single color.
    - One corner contains a small legend boxed by a black rectangular frame; each legend row has a 1-cell color swatch and black dots to its right.
    - Outside the legend are several 4-connected objects, all drawn in one common object color; each object may contain zero or more enclosed background holes.

    Rule:
    - Count the number of enclosed background holes inside each object.
    - Use the legend row whose black-dot count matches that hole count, and read that row's swatch color.
    - For objects with a positive matched hole count, color every exterior background cell that is 4-neighbor-adjacent to the object with the legend color.
    - Leave the original object pixels unchanged, and leave zero-hole objects unchanged.

    Key insight:
    The object attribute that matters is topology, not size or shape: hole count selects a legend color, then that color is applied only to the object's outside 4-neighbor contour.
    """
    return _transform(np.asarray(grid, dtype=int))


def _reserve(forbidden: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(forbidden.shape[0], r0 + h + pad)
    cc1 = min(forbidden.shape[1], c0 + w + pad)
    forbidden[rr0:rr1, cc0:cc1] = True


def _can_place(forbidden: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return not bool(np.any(forbidden[r0 : r0 + h, c0 : c0 + w]))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _hole_slots(height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, height - 1, 2) for c in range(1, width - 1, 2)]


def _make_object_mask(hole_count: int, rng: random.Random) -> np.ndarray:
    if hole_count == 0:
        heights = list(range(3, 7))
        widths = list(range(3, 7))
    else:
        heights = list(range(5, 9))
        widths = list(range(5, 10))
    dims = [(h, w) for h in heights for w in widths if len(_hole_slots(h, w)) >= hole_count]
    if not dims:
        raise RuntimeError("no valid object dimensions for requested hole count")
    rect_h, rect_w = rng.choice(dims)
    mask = np.zeros((rect_h + 2, rect_w + 2), dtype=bool)
    mask[1 : rect_h + 1, 1 : rect_w + 1] = True

    if hole_count:
        slots = _hole_slots(rect_h, rect_w)
        chosen = rng.sample(slots, hole_count)
        for r, c in chosen:
            mask[r + 1, c + 1] = False

    n_tabs = rng.randint(1, 4)
    for _ in range(n_tabs):
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            length = rng.randint(1, min(3, rect_w))
            start = rng.randint(1, rect_w - length + 1)
            row = 0 if side == "top" else rect_h + 1
            mask[row, start : start + length] = True
        else:
            length = rng.randint(1, min(3, rect_h))
            start = rng.randint(1, rect_h - length + 1)
            col = 0 if side == "left" else rect_w + 1
            mask[start : start + length, col] = True

    cropped = _crop_mask(mask)
    if _count_holes(cropped) != hole_count:
        raise RuntimeError("object generation changed the requested hole count")
    return cropped


def _build_legend_patch(colors: list[int], bg: int) -> np.ndarray:
    rows = len(colors)
    inner_w = 2 * rows + 1
    patch = np.full((rows + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = 0
    patch[-1, :] = 0
    patch[:, 0] = 0
    patch[:, -1] = 0
    for idx, color in enumerate(colors, start=1):
        patch[idx, 1] = int(color)
        for dot in range(idx):
            patch[idx, 3 + 2 * dot] = 0
    return patch


def _place_patch_in_corner(grid: np.ndarray, patch: np.ndarray, corner: str) -> tuple[int, int]:
    h, w = grid.shape
    ph, pw = patch.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return r0, c0


def _sample_hole_counts(n_rows: int, n_objects: int, rng: random.Random) -> list[int]:
    positives = list(range(1, n_rows + 1))
    rng.shuffle(positives)
    n_positive = rng.randint(1, min(len(positives), n_objects - 1))
    counts = [0] + positives[:n_positive]
    pool = [0] + positives
    while len(counts) < n_objects:
        counts.append(rng.choice(pool))
    rng.shuffle(counts)
    return counts


def _place_objects(
    grid: np.ndarray,
    object_masks: list[np.ndarray],
    object_color: int,
    legend_bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> bool:
    forbidden = np.zeros(grid.shape, dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    _reserve(forbidden, lr0, lc0, lr1 - lr0 + 1, lc1 - lc0 + 1, pad=2)

    order = sorted(range(len(object_masks)), key=lambda idx: int(object_masks[idx].sum()), reverse=True)
    for idx in order:
        mask = object_masks[idx]
        mh, mw = mask.shape
        if grid.shape[0] - mh - 1 <= 1 or grid.shape[1] - mw - 1 <= 1:
            return False
        placed = False
        for _ in range(250):
            r0 = rng.randint(1, grid.shape[0] - mh - 1)
            c0 = rng.randint(1, grid.shape[1] - mw - 1)
            if not _can_place(forbidden, r0, c0, mh, mw):
                continue
            grid[r0 : r0 + mh, c0 : c0 + mw][mask] = object_color
            _reserve(forbidden, r0, c0, mh, mw, pad=2)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(2, 4)
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    bg = all_colors.pop()
    object_color = all_colors.pop()
    legend_colors = [all_colors.pop() for _ in range(n_rows)]
    legend_patch = _build_legend_patch(legend_colors, bg)

    object_count = rng.randint(2, min(5, n_rows + 2))
    hole_counts = _sample_hole_counts(n_rows, object_count, rng)
    object_masks = [_make_object_mask(h, rng) for h in hole_counts]

    total_area = sum((mask.shape[0] + 4) * (mask.shape[1] + 4) for mask in object_masks)
    side_hint = int(np.ceil(np.sqrt(total_area + legend_patch.size))) + 2
    min_h = max(legend_patch.shape[0] + 4, max(mask.shape[0] for mask in object_masks) + 4, side_hint)
    min_w = max(legend_patch.shape[1] + 4, max(mask.shape[1] for mask in object_masks) + 4, side_hint)
    if min_h > 30 or min_w > 30:
        return None

    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)
    grid = np.full((height, width), bg, dtype=int)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    legend_r0, legend_c0 = _place_patch_in_corner(grid, legend_patch, corner)
    legend_bbox = (
        legend_r0,
        legend_c0,
        legend_r0 + legend_patch.shape[0] - 1,
        legend_c0 + legend_patch.shape[1] - 1,
    )

    if not _place_objects(grid, object_masks, object_color, legend_bbox, rng):
        return None

    if _dominant_color(grid) != bg:
        return None

    mapping, parsed_bbox = _legend_mapping(grid, bg)
    expected_mapping = {idx + 1: color for idx, color in enumerate(legend_colors)}
    if mapping != expected_mapping or parsed_bbox != legend_bbox:
        return None

    parsed_object_color = _object_color(grid, bg, legend_bbox)
    if parsed_object_color != object_color:
        return None

    comps = _components((grid == object_color))
    if len(comps) != len(object_masks):
        return None

    component_holes = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True
        component_holes.append(_count_holes(local))
    if sorted(component_holes) != sorted(hole_counts):
        return None
    if 0 not in component_holes or not any(h > 0 for h in component_holes):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
