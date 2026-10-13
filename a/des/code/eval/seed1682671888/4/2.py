import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(1, 10))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = blocked.copy()
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _rect_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_ring(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(comp["coords"]) != 2 * (h + w) - 4:
        return False
    mask = _rect_mask(comp["coords"], comp["bbox"])
    want = np.zeros_like(mask)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return np.array_equal(mask, want)


def _is_blob(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 1 or w < 1:
        return False
    if len(comp["coords"]) != h * w:
        return False
    return np.all(_rect_mask(comp["coords"], comp["bbox"]))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _corner_candidate(grid: np.ndarray, corner: str) -> tuple[list[int], np.ndarray] | None:
    h, w = grid.shape
    if h < 5 or w < 3:
        return None
    top = corner in {"tl", "tr"}
    left = corner in {"tl", "bl"}
    c0 = 0 if left else w - 2
    guard_c = 2 if left else w - 3
    if c0 < 0 or c0 + 1 >= w:
        return None

    colors: list[int] = []
    starts: list[int] = []
    if top:
        row_iter = list(range(0, h - 1, 3))
    else:
        row_iter = list(range(h - 2, -1, -3))

    for r0 in row_iter:
        block = grid[r0 : r0 + 2, c0 : c0 + 2]
        if block.shape != (2, 2):
            break
        color = int(block[0, 0])
        if color == BG or not np.all(block == color):
            break
        sep_r = r0 + 2 if top else r0 - 1
        if 0 <= sep_r < h and np.any(grid[sep_r, c0 : c0 + 2] != BG):
            return None
        colors.append(color)
        starts.append(r0)

    if len(colors) < 2:
        return None

    if not top:
        colors = colors[::-1]
        starts = starts[::-1]

    span_r0 = starts[0]
    span_r1 = starts[-1] + 1
    if 0 <= guard_c < w and np.any(grid[span_r0 : span_r1 + 1, guard_c] != BG):
        return None

    mask = np.zeros_like(grid, dtype=bool)
    for r0 in starts:
        mask[r0 : r0 + 2, c0 : c0 + 2] = True
    return colors, mask


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    candidates = []
    for corner in ("tl", "tr", "bl", "br"):
        cand = _corner_candidate(grid, corner)
        if cand is not None:
            candidates.append(cand)
    assert len(candidates) == 1, "expected a unique corner legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular rings by containment depth using a corner legend.

    The input contains one vertical corner legend made of 2x2 color swatches.
    From top to bottom, swatch k defines the output color for depth k.
    Rings are 1-pixel thick rectangular borders. Optional filled blobs sit inside
    innermost rings and have depth one greater than that ring. The legend itself
    is removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_colors, legend_mask = _find_legend(g)
    comps = _components(g, legend_mask)
    assert comps, "no objects found outside the legend"

    rings = [comp for comp in comps if _is_ring(comp)]
    blobs = [comp for comp in comps if _is_blob(comp)]
    assert len(rings) + len(blobs) == len(comps), "unexpected non-rectangular object"
    assert rings, "at least one ring is required"

    out = g.copy()
    out[legend_mask] = BG

    for ring in rings:
        depth = 1 + sum(_strictly_contains(other["bbox"], ring["bbox"]) for other in rings)
        assert depth <= len(legend_colors), "ring depth exceeds legend"
        out[tuple(ring["coords"].T)] = legend_colors[depth - 1]

    for blob in blobs:
        containing = [ring for ring in rings if _strictly_contains(ring["bbox"], blob["bbox"])]
        assert containing, "blob must be inside at least one ring"
        depth = len(containing) + 1
        assert depth <= len(legend_colors), "blob depth exceeds legend"
        out[tuple(blob["coords"].T)] = legend_colors[depth - 1]

    return out


def _legend_rows(height: int, depth_count: int, top: bool) -> list[int]:
    total_h = 3 * depth_count - 1
    start = 0 if top else height - total_h
    return [start + 3 * k for k in range(depth_count)]


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _reserve(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _draw_cluster(
    grid: np.ndarray,
    top_left: tuple[int, int],
    ring_depth: int,
    outer_h: int,
    outer_w: int,
    object_colors: list[int],
) -> None:
    r0, c0 = top_left
    for level in range(ring_depth):
        inset = 2 * level
        bbox = (r0 + inset, c0 + inset, r0 + outer_h - 1 - inset, c0 + outer_w - 1 - inset)
        _paint_ring(grid, bbox, object_colors[level])


def _expected_from_spec(
    shape: tuple[int, int],
    legend_mask: np.ndarray,
    ring_specs: list[dict],
    legend_colors: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[legend_mask] = BG
    for spec in ring_specs:
        r0, c0, outer_h, outer_w = spec["r0"], spec["c0"], spec["outer_h"], spec["outer_w"]
        ring_depth = spec["ring_depth"]
        for level in range(ring_depth):
            inset = 2 * level
            bbox = (r0 + inset, c0 + inset, r0 + outer_h - 1 - inset, c0 + outer_w - 1 - inset)
            _paint_ring(out, bbox, legend_colors[level])
        if spec["blob"] is not None:
            br0, bc0, size = spec["blob"]
            out[br0 : br0 + size, bc0 : bc0 + size] = legend_colors[ring_depth]
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()
    legend_len = int(rng.integers(2, 6))
    max_clusters = 3 if legend_len <= 4 else 2
    n_clusters = int(rng.integers(1, max_clusters + 1))

    min_side = max(18, 3 * legend_len + 3, 16 + 2 * (n_clusters - 1))
    h = int(rng.integers(min_side, 31))
    w = int(rng.integers(min_side, 31))
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    corner = str(rng.choice(["tl", "tr", "bl", "br"]))
    top = corner in {"tl", "tr"}
    left = corner in {"tl", "bl"}
    legend_colors = list(map(int, rng.choice(PALETTE, size=legend_len, replace=False)))
    legend_cols = (0, 1) if left else (w - 2, w - 1)
    legend_rows = _legend_rows(h, legend_len, top)
    legend_mask = np.zeros((h, w), dtype=bool)
    for row, color in zip(legend_rows, legend_colors):
        grid[row : row + 2, legend_cols[0] : legend_cols[1] + 1] = color
        legend_mask[row : row + 2, legend_cols[0] : legend_cols[1] + 1] = True
    legend_top = legend_rows[0]
    legend_bottom = legend_rows[-1] + 1
    guard_c = 2 if left else w - 3
    if 0 <= guard_c < w:
        occupied[legend_top : legend_bottom + 1, guard_c] = True
    _reserve(occupied, (legend_top, legend_cols[0], legend_bottom, legend_cols[1]), pad=1)

    specs: list[dict] = []
    special = int(rng.integers(0, n_clusters))
    for idx in range(n_clusters):
        if idx == special:
            has_blob = bool(rng.integers(0, 2))
            ring_depth = legend_len - int(has_blob)
            if ring_depth < 1:
                ring_depth = 1
                has_blob = False
        else:
            has_blob = bool(rng.integers(0, 2))
            ring_depth = int(rng.integers(1, legend_len - int(has_blob) + 1))

        inner_min = 5 if has_blob else 3
        inner_extra = 4 if n_clusters == 1 else 3 if n_clusters == 2 else 2
        inner_h = inner_min + int(rng.integers(0, inner_extra + 1))
        inner_w = inner_min + int(rng.integers(0, inner_extra + 1))
        outer_h = inner_h + 4 * (ring_depth - 1)
        outer_w = inner_w + 4 * (ring_depth - 1)
        specs.append(
            {
                "ring_depth": ring_depth,
                "has_blob": has_blob,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )

    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)

    ring_specs: list[dict] = []
    for spec in specs:
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        if outer_h + 2 > h or outer_w + 2 > w:
            return None
        placed = False
        for _ in range(80):
            r0 = int(rng.integers(1, h - outer_h))
            c0 = int(rng.integers(1, w - outer_w))
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _free(occupied, bbox, pad=1):
                continue
            placed = True
            _reserve(occupied, bbox, pad=1)
            ring_specs.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "ring_depth": spec["ring_depth"],
                    "blob": None,
                }
            )
            break
        if not placed:
            return None

    legend_color_set = set(legend_colors)
    object_colors = [c for c in PALETTE if c not in legend_color_set]
    if not object_colors:
        object_colors = list(PALETTE)

    for spec in ring_specs:
        colors = list(map(int, rng.choice(object_colors, size=spec["ring_depth"], replace=True)))
        spec["ring_colors"] = colors
        _draw_cluster(
            grid,
            (spec["r0"], spec["c0"]),
            spec["ring_depth"],
            spec["outer_h"],
            spec["outer_w"],
            colors,
        )

    for source, target in zip(specs, ring_specs):
        if not source["has_blob"]:
            continue
        r0, c0 = target["r0"], target["c0"]
        ring_depth = target["ring_depth"]
        outer_h = target["outer_h"]
        outer_w = target["outer_w"]
        inner_r0 = r0 + 2 * (ring_depth - 1)
        inner_c0 = c0 + 2 * (ring_depth - 1)
        inner_r1 = r0 + outer_h - 1 - 2 * (ring_depth - 1)
        inner_c1 = c0 + outer_w - 1 - 2 * (ring_depth - 1)
        center_h = inner_r1 - inner_r0 - 3
        center_w = inner_c1 - inner_c0 - 3
        if center_h < 1 or center_w < 1:
            return None
        blob_size = int(rng.integers(1, min(3, center_h, center_w) + 1))
        blob_r0 = inner_r0 + 2 + (center_h - blob_size) // 2
        blob_c0 = inner_c0 + 2 + (center_w - blob_size) // 2
        blob_choices = [c for c in object_colors if c not in target["ring_colors"]]
        if not blob_choices:
            blob_choices = object_colors
        blob_color = int(rng.choice(blob_choices))
        grid[blob_r0 : blob_r0 + blob_size, blob_c0 : blob_c0 + blob_size] = blob_color
        target["blob"] = (blob_r0, blob_c0, blob_size)

    expected = _expected_from_spec(grid.shape, legend_mask, ring_specs, legend_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid nested-ring input")
