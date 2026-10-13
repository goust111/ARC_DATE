from __future__ import annotations

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _boundary_bg(g: np.ndarray) -> int:
    boundary = np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))
    colors, counts = np.unique(boundary, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = np.min(arr, axis=0)
            r1, c1 = np.max(arr, axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _is_rect_ring(comp: dict) -> bool:
    mask = _component_mask(comp)
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _is_solid_block(comp: dict) -> bool:
    return bool(np.all(_component_mask(comp)))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _ring_icon(color: int, bg: int) -> np.ndarray:
    icon = np.full((5, 5), bg, dtype=int)
    icon[0, :] = color
    icon[-1, :] = color
    icon[:, 0] = color
    icon[:, -1] = color
    return icon


def _core_icon(color: int) -> np.ndarray:
    return np.full((3, 3), color, dtype=int)


def _render_output(bg: int, core_color: int, ring_colors: list[int]) -> np.ndarray:
    segments = [_core_icon(core_color)] + [_ring_icon(color, bg) for color in ring_colors]
    out_h = 5
    out_w = sum(seg.shape[1] for seg in segments) + len(segments) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c = 0
    for seg in segments:
        h, w = seg.shape
        top = (out_h - h) // 2
        out[top : top + h, c : c + w] = seg
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Linearize nested rectangular rings from the core outward as normalized doll icons."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _boundary_bg(g)
    comps = _components(g, bg)
    assert comps, "expected foreground objects"

    rings = [comp for comp in comps if _is_rect_ring(comp)]
    others = [comp for comp in comps if not _is_rect_ring(comp)]
    assert rings, "expected at least one rectangular ring"
    assert len(others) == 1, "expected one core marker"

    core = others[0]
    assert _is_solid_block(core), "core marker must be a solid block"
    core_bbox = core["bbox"]
    for ring in rings:
        assert _strictly_contains(ring["bbox"], core_bbox), "every ring must contain the core"

    ranked: list[tuple[int, int, dict]] = []
    for ring in rings:
        contain_count = sum(
            _strictly_contains(other["bbox"], ring["bbox"])
            for other in rings
            if other is not ring
        )
        area = (ring["bbox"][2] - ring["bbox"][0] + 1) * (ring["bbox"][3] - ring["bbox"][1] + 1)
        ranked.append((int(contain_count), int(area), ring))

    counts = sorted(count for count, _area, _ring in ranked)
    assert counts == list(range(len(rings))), "rings must form one strict nesting chain"
    ranked.sort(key=lambda item: (-item[0], item[1]))

    ring_colors = [int(ring["color"]) for _count, _area, ring in ranked]
    return _render_output(bg, int(core["color"]), ring_colors)


GEN_BG = 0
GEN_COLORS = tuple(range(1, 10))
GEN_RING_COUNT_MIN = 2
GEN_RING_COUNT_MAX = 4
GEN_CORE_SIZE_MIN = 1
GEN_CORE_SIZE_MAX = 3
GEN_GAP_MIN = 1
GEN_GAP_MAX = 3
GEN_MARGIN_MIN = 1
GEN_MARGIN_MAX = 4
GEN_GRID_MAX = 30
GEN_MAX_ATTEMPTS = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _expand_bbox(
    inner: tuple[int, int, int, int],
    top_gap: int,
    bottom_gap: int,
    left_gap: int,
    right_gap: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = inner
    return (
        int(r0 - top_gap - 1),
        int(c0 - left_gap - 1),
        int(r1 + bottom_gap + 1),
        int(c1 + right_gap + 1),
    )


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc))


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_solid(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    ring_count = _randint(rng, GEN_RING_COUNT_MIN, GEN_RING_COUNT_MAX)
    core_size = _randint(rng, GEN_CORE_SIZE_MIN, GEN_CORE_SIZE_MAX)
    colors = [int(c) for c in rng.choice(GEN_COLORS, size=ring_count + 1, replace=False)]
    core_color = colors[0]
    ring_colors = colors[1:]

    core_bbox = (0, 0, core_size - 1, core_size - 1)
    inner = core_bbox
    ring_bboxes = []
    gap_hi = GEN_GAP_MAX if ring_count <= 3 else GEN_GAP_MAX - 1
    for _ in range(ring_count):
        top_gap = _randint(rng, GEN_GAP_MIN, gap_hi)
        bottom_gap = _randint(rng, GEN_GAP_MIN, gap_hi)
        left_gap = _randint(rng, GEN_GAP_MIN, gap_hi)
        right_gap = _randint(rng, GEN_GAP_MIN, gap_hi)
        inner = _expand_bbox(inner, top_gap, bottom_gap, left_gap, right_gap)
        ring_bboxes.append(inner)

    outer_bbox = ring_bboxes[-1]
    outer_h = outer_bbox[2] - outer_bbox[0] + 1
    outer_w = outer_bbox[3] - outer_bbox[1] + 1
    if outer_h > GEN_GRID_MAX - 2 or outer_w > GEN_GRID_MAX - 2:
        return None

    top_margin = _randint(rng, GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    bottom_margin = _randint(rng, GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    left_margin = _randint(rng, GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    right_margin = _randint(rng, GEN_MARGIN_MIN, GEN_MARGIN_MAX)

    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin
    if grid_h > GEN_GRID_MAX or grid_w > GEN_GRID_MAX:
        return None

    shift_r = top_margin - outer_bbox[0]
    shift_c = left_margin - outer_bbox[1]
    grid = np.full((grid_h, grid_w), GEN_BG, dtype=int)

    for bbox, color in zip(ring_bboxes, ring_colors):
        _paint_ring(grid, _shift_bbox(bbox, shift_r, shift_c), color)
    _paint_solid(grid, _shift_bbox(core_bbox, shift_r, shift_c), core_color)

    if np.count_nonzero(grid == GEN_BG) <= grid.size // 2:
        return None

    expected = _render_output(GEN_BG, core_color, ring_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
