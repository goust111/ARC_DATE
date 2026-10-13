from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 500
TEMPLATE_GAP = 2
FENCE_GAP = 1
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (24, 32)
FENCE_COUNT_RANGE = (2, 4)
FENCE_H_RANGE = (5, 10)
FENCE_W_RANGE = (5, 10)


_BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),          # plus
    ((-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)),        # T
    ((-1, 0), (0, 0), (1, 0), (1, 1)),                   # L
    ((-1, 0), (0, 0), (0, 1), (0, 2)),                   # gamma
    ((-1, 0), (0, 0), (0, 1), (1, 1)),                   # zig
    ((-1, 0), (0, -1), (0, 0), (1, 0), (1, 1)),          # fork
)


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, "expected a unique dominant color"
    return int(winners[0])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_fence(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if len(coords) != expected:
        return False
    cells = {tuple(map(int, xy)) for xy in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in cells):
                return False
    return True


def _parse_template(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    values = [int(grid[r, c]) for r, c in coords]
    counts = Counter(values)
    assert len(counts) == 2, "template must use body color plus one marker color"
    singletons = [color for color, count in counts.items() if count == 1]
    assert len(singletons) == 1, "template marker color must appear exactly once"
    center_color = int(singletons[0])
    body_color = int(next(color for color in counts if color != center_color))
    center_cells = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == center_color]
    assert len(center_cells) == 1
    cr, cc = center_cells[0]
    offsets = [(int(r - cr), int(c - cc)) for r, c in coords]
    assert len(offsets) >= 3, "template footprint should be visually meaningful"
    return {
        "center_color": center_color,
        "body_color": body_color,
        "offsets": offsets,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fence seed color to the template carrying the same marker color,
    then stamp that template inside the fence with the seed as the anchor.
    Any part that would cross the fence border is clipped away.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)

    fences: list[dict[str, object]] = []
    fence_mask = np.zeros_like(g, dtype=bool)
    for color in sorted(set(map(int, g.reshape(-1))) - {bg}):
        for coords in _components(g == color):
            if _is_fence(coords):
                fences.append(
                    {
                        "bbox": _bbox(coords),
                        "color": int(color),
                    }
                )
                fence_mask[coords[:, 0], coords[:, 1]] = True

    assert fences, "expected at least one fence"

    seeds: list[dict[str, int]] = []
    seed_mask = np.zeros_like(g, dtype=bool)
    for fence in fences:
        r0, c0, r1, c1 = map(int, fence["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere(interior != bg)
        assert len(seed_coords) == 1, "every fence must contain exactly one seed"
        sr = int(seed_coords[0, 0] + r0 + 1)
        sc = int(seed_coords[0, 1] + c0 + 1)
        seeds.append({"row": sr, "col": sc, "color": int(g[sr, sc])})
        seed_mask[sr, sc] = True

    template_mask = (g != bg) & ~fence_mask & ~seed_mask
    templates = [_parse_template(g, coords) for coords in _components(template_mask)]
    assert len(templates) == len(fences), "template count must match fence count"

    template_by_seed: dict[int, dict[str, object]] = {}
    for template in templates:
        center_color = int(template["center_color"])
        assert center_color not in template_by_seed, "template seed colors must be unique"
        template_by_seed[center_color] = template

    seed_colors = {int(seed["color"]) for seed in seeds}
    assert seed_colors == set(template_by_seed), "seed colors must match template marker colors"

    out = g.copy()
    for fence, seed in zip(fences, seeds):
        r0, c0, r1, c1 = map(int, fence["bbox"])
        sr = int(seed["row"])
        sc = int(seed["col"])
        template = template_by_seed[int(seed["color"])]
        body_color = int(template["body_color"])
        for dr, dc in template["offsets"]:
            rr = sr + int(dr)
            cc = sc + int(dc)
            if r0 < rr < r1 and c0 < cc < c1:
                out[rr, cc] = body_color
    return out


def _transform_offsets(
    offsets: tuple[tuple[int, int], ...],
    rotate_k: int,
    mirror: bool,
) -> tuple[tuple[int, int], ...]:
    transformed = []
    for r, c in offsets:
        rr, cc = int(r), int(c)
        if mirror:
            cc = -cc
        for _ in range(rotate_k % 4):
            rr, cc = cc, -rr
        transformed.append((rr, cc))
    return tuple(sorted(set(transformed)))


def _shape_bounds(offsets: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    return min(rows), min(cols), max(rows), max(cols)


def _render_template(
    offsets: tuple[tuple[int, int], ...],
    body_color: int,
    marker_color: int,
) -> np.ndarray:
    r0, c0, r1, c1 = _shape_bounds(offsets)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for dr, dc in offsets:
        patch[dr - r0, dc - c0] = int(body_color)
    patch[-r0, -c0] = int(marker_color)
    return patch


def _template_total_width(patches: list[np.ndarray]) -> int:
    return sum(int(p.shape[1]) for p in patches) + TEMPLATE_GAP * (len(patches) - 1)


def _paint_template_band(
    grid: np.ndarray,
    patches: list[np.ndarray],
    top_side: bool,
    left_side: bool,
) -> tuple[int, int]:
    band_h = max(int(p.shape[0]) for p in patches)
    total_w = _template_total_width(patches)
    edge_margin = random.randint(1, 2)

    if top_side:
        band_r = edge_margin
    else:
        band_r = grid.shape[0] - band_h - edge_margin

    if left_side:
        cursor = edge_margin
    else:
        cursor = grid.shape[1] - total_w - edge_margin

    for patch in patches:
        dr = random.randint(0, band_h - int(patch.shape[0]))
        r0 = band_r + dr
        c0 = cursor
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = np.where(patch != BG, patch, grid[r0 : r0 + h, c0 : c0 + w])
        cursor += w + TEMPLATE_GAP

    return band_r, band_r + band_h - 1


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_fences(
    grid_h: int,
    grid_w: int,
    top_rows: int,
    bottom_rows: int,
    dims: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    row_min = top_rows
    row_max = grid_h - bottom_rows - 1
    for h, w in sorted(dims, key=lambda item: item[0] * item[1], reverse=True):
        if row_max - row_min + 1 < h:
            return None
        placed = None
        for _ in range(80):
            r0 = random.randint(row_min, row_max - h + 1)
            c0 = random.randint(1, grid_w - w - 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_intersects(box, other, FENCE_GAP) for other in boxes):
                continue
            placed = box
            break
        if placed is None:
            return None
        boxes.append(placed)
    return boxes


def _visible_count(
    offsets: tuple[tuple[int, int], ...],
    seed_r: int,
    seed_c: int,
    bbox: tuple[int, int, int, int],
) -> int:
    r0, c0, r1, c1 = bbox
    count = 0
    for dr, dc in offsets:
        rr = seed_r + dr
        cc = seed_c + dc
        if r0 < rr < r1 and c0 < cc < c1:
            count += 1
    return count


def _seed_positions(
    offsets: tuple[tuple[int, int], ...],
    bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    safe: list[tuple[int, int]] = []
    clipped: list[tuple[int, int]] = []
    useful: list[tuple[int, int]] = []
    for sr in range(r0 + 1, r1):
        for sc in range(c0 + 1, c1):
            visible = _visible_count(offsets, sr, sc, bbox)
            if visible >= 2:
                useful.append((sr, sc))
            if visible == len(offsets):
                safe.append((sr, sc))
            elif 2 <= visible < len(offsets):
                clipped.append((sr, sc))
    return safe, clipped, useful


def _build_attempt() -> np.ndarray | None:
    n_fences = random.randint(*FENCE_COUNT_RANGE)
    needed_colors = 1 + 2 * n_fences
    palette = random.sample(COLORS, needed_colors)
    fence_color = int(palette[0])
    seed_colors = [int(x) for x in palette[1 : 1 + n_fences]]
    body_colors = [int(x) for x in palette[1 + n_fences :]]

    specs = []
    patches = []
    dims = []
    safe_choices = []

    for seed_color, body_color in zip(seed_colors, body_colors):
        base = random.choice(_BASE_SHAPES)
        offsets = _transform_offsets(base, random.randint(0, 3), random.choice([False, True]))
        patch = _render_template(offsets, body_color=body_color, marker_color=seed_color)
        patches.append(patch)

        min_r, min_c, max_r, max_c = _shape_bounds(offsets)
        span_h = max_r - min_r + 1
        span_w = max_c - min_c + 1
        want_safe = random.choice([False, True])
        h_lo = max(FENCE_H_RANGE[0], span_h + 2) if want_safe else FENCE_H_RANGE[0]
        w_lo = max(FENCE_W_RANGE[0], span_w + 2) if want_safe else FENCE_W_RANGE[0]
        if h_lo > FENCE_H_RANGE[1] or w_lo > FENCE_W_RANGE[1]:
            return None
        dims.append((random.randint(h_lo, FENCE_H_RANGE[1]), random.randint(w_lo, FENCE_W_RANGE[1])))
        safe_choices.append(want_safe)
        specs.append(
            {
                "seed_color": seed_color,
                "body_color": body_color,
                "offsets": offsets,
            }
        )

    total_template_w = _template_total_width(patches)
    max_template_h = max(int(p.shape[0]) for p in patches)
    grid_h = random.randint(max(GRID_H_RANGE[0], max_template_h + 12), GRID_H_RANGE[1])
    grid_w = random.randint(max(GRID_W_RANGE[0], total_template_w + 4), GRID_W_RANGE[1])
    grid = np.zeros((grid_h, grid_w), dtype=int)

    top_side = random.choice([False, True])
    left_side = random.choice([False, True])
    band_r0, band_r1 = _paint_template_band(grid, patches, top_side=top_side, left_side=left_side)

    if top_side:
        top_rows = band_r1 + 2
        bottom_rows = 1
    else:
        top_rows = 1
        bottom_rows = grid_h - band_r0 + 1

    boxes = _place_fences(grid_h, grid_w, top_rows=top_rows, bottom_rows=bottom_rows, dims=dims)
    if boxes is None or len(boxes) != n_fences:
        return None

    box_by_dim = {dim: [] for dim in dims}
    for box, dim in zip(boxes, sorted(dims, key=lambda item: item[0] * item[1], reverse=True)):
        box_by_dim[dim].append(box)

    ordered_boxes: list[tuple[int, int, int, int]] = []
    for dim in dims:
        ordered_boxes.append(box_by_dim[dim].pop())

    for spec, bbox, want_safe in zip(specs, ordered_boxes, safe_choices):
        r0, c0, r1, c1 = bbox
        grid[r0, c0 : c1 + 1] = fence_color
        grid[r1, c0 : c1 + 1] = fence_color
        grid[r0 : r1 + 1, c0] = fence_color
        grid[r0 : r1 + 1, c1] = fence_color

        safe, clipped, useful = _seed_positions(spec["offsets"], bbox)
        if not useful:
            return None
        if want_safe and safe:
            seed_r, seed_c = random.choice(safe)
        elif (not want_safe) and clipped:
            seed_r, seed_c = random.choice(clipped)
        elif clipped:
            seed_r, seed_c = random.choice(clipped)
        elif safe:
            seed_r, seed_c = random.choice(safe)
        else:
            seed_r, seed_c = random.choice(useful)
        grid[seed_r, seed_c] = int(spec["seed_color"])

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if _dominant_color(grid) != BG:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
