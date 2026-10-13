from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GAP_DIRS = ("top", "bottom", "left", "right")
COLOR_IDS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            pts: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r0 for r0, _ in pts]
            cols = [c0 for _, c0 in pts]
            components.append(
                {
                    "color": color,
                    "pts": pts,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _ring_from_component(grid: np.ndarray, bg: int, comp: dict[str, object]) -> dict[str, object] | None:
    color = int(comp["color"])
    pts = list(comp["pts"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    if np.any(mask[1:-1, 1:-1]):
        return None

    boundary = np.zeros((height, width), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    if int(mask.sum()) != int(boundary.sum()) - 1:
        return None

    missing = np.argwhere(boundary & (~mask))
    if missing.shape != (1, 2):
        return None
    mr, mc = map(int, missing[0])
    if (mr in (0, height - 1)) and (mc in (0, width - 1)):
        return None

    gap_r = r0 + mr
    gap_c = c0 + mc
    if int(grid[gap_r, gap_c]) != bg:
        return None

    if mr == 0:
        gap_dir = "top"
        gap_index = gap_c
    elif mr == height - 1:
        gap_dir = "bottom"
        gap_index = gap_c
    elif mc == 0:
        gap_dir = "left"
        gap_index = gap_r
    else:
        gap_dir = "right"
        gap_index = gap_r

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "gap_dir": gap_dir,
        "gap_index": int(gap_index),
    }


def _aligned_outer_mark(inner_ring: dict[str, object], outer_ring: dict[str, object]) -> tuple[int, int]:
    ir0, ic0, ir1, ic1 = map(int, inner_ring["bbox"])
    or0, oc0, or1, oc1 = map(int, outer_ring["bbox"])
    gap_dir = str(inner_ring["gap_dir"])
    gap_index = int(inner_ring["gap_index"])

    if gap_dir == "top":
        assert oc0 <= gap_index <= oc1
        return or0, gap_index
    if gap_dir == "bottom":
        assert oc0 <= gap_index <= oc1
        return or1, gap_index
    if gap_dir == "left":
        assert or0 <= gap_index <= or1
        return gap_index, oc0
    assert gap_dir == "right"
    assert or0 <= gap_index <= or1
    return gap_index, oc1


def _paint_gap_side(out: np.ndarray, ring: dict[str, object], fill_color: int) -> None:
    r0, c0, r1, c1 = map(int, ring["bbox"])
    gap_dir = str(ring["gap_dir"])
    if gap_dir == "top":
        out[r0, c0 : c1 + 1] = fill_color
    elif gap_dir == "bottom":
        out[r1, c0 : c1 + 1] = fill_color
    elif gap_dir == "left":
        out[r0 : r1 + 1, c0] = fill_color
    else:
        out[r0 : r1 + 1, c1] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each nested ring structure, propagate the innermost core color outward conceptually,
    then recolor the outermost ring's gap side with that core color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _same_color_components(g, bg)

    rings: list[dict[str, object]] = []
    cores: list[tuple[int, int, int]] = []
    for comp in components:
        ring = _ring_from_component(g, bg, comp)
        if ring is not None:
            ring["id"] = len(rings)
            rings.append(ring)
            continue

        pts = list(comp["pts"])
        assert len(pts) == 1, "unexpected non-ring foreground object"
        r, c = pts[0]
        cores.append((int(r), int(c), int(comp["color"])))

    assert rings, "expected at least one rectangular ring"
    assert cores, "expected at least one core pixel"

    out = g.copy()
    used_ring_ids: set[int] = set()

    for core_r, core_c, core_color in cores:
        chain = [
            ring
            for ring in rings
            if int(ring["bbox"][0]) < core_r < int(ring["bbox"][2]) and int(ring["bbox"][1]) < core_c < int(ring["bbox"][3])
        ]
        assert 2 <= len(chain) <= 4, "each core must be inside one 2-to-4-ring chain"
        chain.sort(key=lambda ring: (int(ring["bbox"][2]) - int(ring["bbox"][0]), int(ring["bbox"][3]) - int(ring["bbox"][1])))

        inner = chain[0]
        ir0, ic0, ir1, ic1 = map(int, inner["bbox"])
        assert ir0 < core_r < ir1 and ic0 < core_c < ic1

        for inner_ring, outer_ring in zip(chain, chain[1:], strict=False):
            in_r0, in_c0, in_r1, in_c1 = map(int, inner_ring["bbox"])
            out_r0, out_c0, out_r1, out_c1 = map(int, outer_ring["bbox"])
            assert out_r0 == in_r0 - 2 and out_c0 == in_c0 - 2
            assert out_r1 == in_r1 + 2 and out_c1 == in_c1 + 2

            mark_r, mark_c = _aligned_outer_mark(inner_ring, outer_ring)
            assert (
                mark_r == out_r0 or mark_r == out_r1 or mark_c == out_c0 or mark_c == out_c1
            ), "propagated mark must land on the adjacent outer ring side"

        for ring in chain:
            ring_id = int(ring["id"])
            assert ring_id not in used_ring_ids, "ring chains must be disjoint"
            used_ring_ids.add(ring_id)

        _paint_gap_side(out, chain[-1], core_color)

    assert len(used_ring_ids) == len(rings), "every ring must belong to exactly one core-centered chain"
    return out


def _draw_ring(patch: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap_dir: str, gap_index: int, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    patch[r0, c0 : c1 + 1] = color
    patch[r1, c0 : c1 + 1] = color
    patch[r0 : r1 + 1, c0] = color
    patch[r0 : r1 + 1, c1] = color

    if gap_dir == "top":
        patch[r0, gap_index] = bg
    elif gap_dir == "bottom":
        patch[r1, gap_index] = bg
    elif gap_dir == "left":
        patch[gap_index, c0] = bg
    else:
        patch[gap_index, c1] = bg


def _random_gap(bbox: tuple[int, int, int, int]) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox
    gap_dir = random.choice(GAP_DIRS)
    if gap_dir in ("top", "bottom"):
        return gap_dir, random.randint(c0 + 1, c1 - 1)
    return gap_dir, random.randint(r0 + 1, r1 - 1)


def _build_group(bg: int, core_color: int, ring_count: int, inner_h: int, inner_w: int) -> np.ndarray:
    assert 2 <= ring_count <= 4
    assert inner_h >= 1 and inner_w >= 1

    inner_bbox_h = inner_h + 2
    inner_bbox_w = inner_w + 2
    outer_h = inner_bbox_h + 4 * (ring_count - 1)
    outer_w = inner_bbox_w + 4 * (ring_count - 1)
    patch = np.full((outer_h, outer_w), bg, dtype=int)

    ring_colors = random.sample([c for c in COLOR_IDS if c not in (bg, core_color)], ring_count)
    inner_bbox: tuple[int, int, int, int] | None = None

    for depth in range(ring_count):
        inset = 2 * (ring_count - 1 - depth)
        bbox = (inset, inset, outer_h - 1 - inset, outer_w - 1 - inset)
        gap_dir, gap_index = _random_gap(bbox)
        _draw_ring(patch, bbox, ring_colors[depth], gap_dir, gap_index, bg)
        if depth == 0:
            inner_bbox = bbox

    assert inner_bbox is not None
    r0, c0, r1, c1 = inner_bbox
    core_r = random.randint(r0 + 1, r1 - 1)
    core_c = random.randint(c0 + 1, c1 - 1)
    patch[core_r, core_c] = core_color
    return patch


def _placement_candidates(occupied: np.ndarray, patch_shape: tuple[int, int], border: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch_shape
    positions: list[tuple[int, int]] = []
    for r0 in range(border, h - ph - border + 1):
        for c0 in range(border, w - pw - border + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if not np.any(occupied[rr0:rr1, cc0:cc1]):
                positions.append((r0, c0))
    return positions


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    sub = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != bg
    sub[mask] = patch[mask]


def _sample_group_count() -> int:
    return random.choice((1, 2, 3))


def _group_shape_params(group_count: int) -> tuple[int, int, int]:
    if group_count == 1:
        ring_count = random.randint(2, 4)
        inner_h = random.randint(1, 5)
        inner_w = random.randint(1, 5)
    elif group_count == 2:
        ring_count = random.randint(2, 4)
        inner_h = random.randint(1, 4)
        inner_w = random.randint(1, 4)
    else:
        ring_count = random.randint(2, 3)
        inner_h = random.randint(1, 3)
        inner_w = random.randint(1, 3)
    return ring_count, inner_h, inner_w


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    bg = random.choice(COLOR_IDS)
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    group_count = _sample_group_count()
    core_colors = random.sample([c for c in COLOR_IDS if c != bg], group_count)
    patches: list[np.ndarray] = []
    for core_color in core_colors:
        ring_count, inner_h, inner_w = _group_shape_params(group_count)
        patches.append(_build_group(bg, core_color, ring_count, inner_h, inner_w))

    patches.sort(key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True)
    for patch in patches:
        candidates = _placement_candidates(occupied, patch.shape, border=1)
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _paste_non_bg(grid, patch, top, left, bg)
        rr0 = max(0, top - 1)
        cc0 = max(0, left - 1)
        rr1 = min(height, top + patch.shape[0] + 1)
        cc1 = min(width, left + patch.shape[1] + 1)
        occupied[rr0:rr1, cc0:cc1] = True

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
