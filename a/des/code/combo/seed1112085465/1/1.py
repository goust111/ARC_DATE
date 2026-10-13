from __future__ import annotations

from collections import Counter, deque

import numpy as np


RNG = np.random.default_rng()

BACKGROUND = 0
FRAME_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
EDGE_OPTIONS = ("top", "bottom", "left", "right")
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        queue = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    base = _trim_mask(mask)
    seen = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(base, k)
        for cand in (rot, np.fliplr(rot)):
            trimmed = _trim_mask(cand)
            key = (trimmed.shape, trimmed.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(trimmed.copy())
    return out


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((cand.shape, cand.astype(np.uint8).tobytes()) for cand in _dihedral_masks(mask))


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _non_bg_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for coords in _connected_components(grid != bg):
        bbox = _bbox_from_coords(coords)
        mask = _mask_from_coords(coords, bbox)
        values = sorted({int(grid[r, c]) for r, c in coords})
        r0, c0, r1, c1 = bbox
        comps.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "colors": values,
                "area": int(len(coords)),
                "touches_edge": _touches_edge(coords, grid.shape),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
            }
        )
    return comps


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    comps = _non_bg_components(g, bg)
    assert comps, "expected non-background objects"

    frame = max(comps, key=lambda comp: (comp["area"], comp["height"] * comp["width"]))
    fr0, fc0, fr1, fc1 = frame["bbox"]
    frame_crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    frame_color = int(g[frame["coords"][0, 0], frame["coords"][0, 1]])

    hole_infos = []
    for coords in _connected_components(frame_crop == bg):
        if _touches_edge(coords, frame_crop.shape):
            continue
        bbox = _bbox_from_coords(coords)
        mask = _mask_from_coords(coords, bbox)
        hole_infos.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "key": _shape_key(mask),
            }
        )
    hole_infos.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert hole_infos, "frame must contain at least one enclosed hole"

    other_comps = [comp for comp in comps if comp is not frame]
    palette_candidates = [
        comp
        for comp in other_comps
        if comp["touches_edge"]
        and (comp["height"] == 1 or comp["width"] == 1)
        and comp["area"] == max(comp["height"], comp["width"])
    ]
    assert palette_candidates, "expected one palette strip on the border"

    exact_len = [comp for comp in palette_candidates if comp["area"] == len(hole_infos)]
    palette = exact_len[0] if exact_len else max(palette_candidates, key=lambda comp: comp["area"])
    pr0, pc0, pr1, pc1 = palette["bbox"]
    if palette["height"] == 1:
        palette_colors = [int(g[pr0, c]) for c in range(pc0, pc1 + 1)]
    else:
        palette_colors = [int(g[r, pc0]) for r in range(pr0, pr1 + 1)]
    assert len(palette_colors) == len(hole_infos), "palette length must match hole count"

    sprite_comps = [comp for comp in other_comps if comp is not palette]
    sprite_keys = Counter(_shape_key(comp["mask"]) for comp in sprite_comps)
    hole_keys = Counter(info["key"] for info in hole_infos)
    assert sprite_keys == hole_keys, "outside sprites must match hole shapes"

    out = np.full(frame_crop.shape, frame_color, dtype=int)
    for color, hole in zip(palette_colors, hole_infos):
        rows = hole["coords"][:, 0]
        cols = hole["coords"][:, 1]
        out[rows, cols] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(200):
        h = int(RNG.integers(2, 5))
        w = int(RNG.integers(2, 5))
        max_area = min(int(h * w - 1), 6)
        if max_area < 3:
            continue
        area = int(RNG.integers(3, max_area + 1))
        start = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = list(frontier)[int(RNG.integers(0, len(frontier)))]
            cells.add(choice)
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if mask.all():
            continue
        return mask
    return None


def _sample_unique_shapes(count: int) -> list[np.ndarray] | None:
    for _ in range(200):
        shapes: list[np.ndarray] = []
        seen = set()
        while len(shapes) < count:
            mask = _random_connected_mask()
            if mask is None:
                break
            key = _shape_key(mask)
            if key in seen:
                continue
            seen.add(key)
            shapes.append(mask)
        if len(shapes) != count:
            continue
        if any(len(_dihedral_masks(mask)) > 1 for mask in shapes):
            return shapes
    return None


def _place_holes_in_frame(shapes: list[np.ndarray]) -> tuple[np.ndarray, list[dict]] | None:
    approx_area = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in shapes)
    for _ in range(80):
        inner_h = int(RNG.integers(8, 15))
        inner_w = int(RNG.integers(9, 17))
        if inner_h * inner_w < approx_area:
            continue
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        holes = np.zeros((frame_h, frame_w), dtype=bool)
        blocked = np.zeros_like(holes)
        placed: list[dict] = []
        ok = True
        for idx in RNG.permutation(len(shapes)):
            mask = shapes[int(idx)]
            h, w = mask.shape
            coords = np.argwhere(mask)
            found = None
            for _ in range(250):
                r0 = int(RNG.integers(1, frame_h - h))
                c0 = int(RNG.integers(1, frame_w - w))
                if np.any(blocked[r0 : r0 + h, c0 : c0 + w] & mask):
                    continue
                found = (r0, c0)
                break
            if found is None:
                ok = False
                break
            r0, c0 = found
            sub = holes[r0 : r0 + h, c0 : c0 + w]
            sub[mask] = True
            for rr, cc in coords:
                rr = int(rr)
                cc = int(cc)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr = r0 + rr + dr
                        nc = c0 + cc + dc
                        if 0 <= nr < frame_h and 0 <= nc < frame_w:
                            blocked[nr, nc] = True
            placed.append({"mask": mask, "r0": r0, "c0": c0})
        if not ok:
            continue
        return holes, placed
    return None


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = blocked.shape
    for rr, cc in np.argwhere(mask):
        rr = int(rr)
        cc = int(cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r0 + rr + dr
                nc = c0 + cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_mask(blocked: np.ndarray, mask: np.ndarray, allow_edge: bool = False) -> tuple[int, int] | None:
    h, w = mask.shape
    grid_h, grid_w = blocked.shape
    row_lo = 0 if allow_edge else 1
    col_lo = 0 if allow_edge else 1
    row_hi = grid_h - h if allow_edge else grid_h - h - 1
    col_hi = grid_w - w if allow_edge else grid_w - w - 1
    if row_hi < row_lo or col_hi < col_lo:
        return None
    for _ in range(300):
        r0 = int(RNG.integers(row_lo, row_hi + 1))
        c0 = int(RNG.integers(col_lo, col_hi + 1))
        if np.any(blocked[r0 : r0 + h, c0 : c0 + w] & mask):
            continue
        return r0, c0
    return None


def _random_variant(shape: np.ndarray, force_change: bool) -> np.ndarray:
    variants = _dihedral_masks(shape)
    if force_change:
        changed = [cand for cand in variants if not np.array_equal(cand, shape)]
        if changed:
            return changed[int(RNG.integers(0, len(changed)))]
    return variants[int(RNG.integers(0, len(variants)))]


def _gen_attempt() -> np.ndarray | None:
    hole_count = int(RNG.integers(2, 5))
    shapes = _sample_unique_shapes(hole_count)
    if shapes is None:
        return None

    placed = _place_holes_in_frame(shapes)
    if placed is None:
        return None
    hole_mask, hole_infos = placed
    frame_h, frame_w = hole_mask.shape

    hole_infos.sort(key=lambda item: (item["r0"], item["c0"]))
    hole_areas = [int(item["mask"].sum()) for item in hole_infos]
    frame_pixels = int(frame_h * frame_w - sum(hole_areas))

    edge = EDGE_OPTIONS[int(RNG.integers(0, len(EDGE_OPTIONS)))]
    color_order = list(RNG.permutation(COLOR_POOL))
    sprite_colors = [int(color_order[i]) for i in range(hole_count)]
    palette_colors = [int(color_order[hole_count + i]) for i in range(hole_count)]

    asymmetric = [i for i, shape in enumerate(shapes) if any(not np.array_equal(v, shape) for v in _dihedral_masks(shape))]
    force_idx = int(asymmetric[int(RNG.integers(0, len(asymmetric)))]) if asymmetric else -1
    sprite_specs = []
    for i, shape in enumerate(shapes):
        sprite_specs.append(
            {
                "key": _shape_key(shape),
                "mask": _random_variant(shape, force_change=(i == force_idx)),
                "color": sprite_colors[i],
            }
        )
    sprite_specs = [sprite_specs[i] for i in RNG.permutation(len(sprite_specs))]

    total_object_pixels = frame_pixels + sum(int(spec["mask"].sum()) for spec in sprite_specs) + hole_count
    total_needed = total_object_pixels + frame_pixels + 30
    canvas_h = max(frame_h + int(RNG.integers(8, 14)), hole_count + 4)
    canvas_w = max(frame_w + int(RNG.integers(10, 16)), hole_count + 4)
    while canvas_h * canvas_w < total_needed:
        if canvas_h <= canvas_w:
            canvas_h += 1
        else:
            canvas_w += 1

    grid = np.full((canvas_h, canvas_w), BACKGROUND, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    if edge in ("top", "bottom"):
        if canvas_w < hole_count + 2:
            return None
        row = 0 if edge == "top" else canvas_h - 1
        col = int(RNG.integers(1, canvas_w - hole_count))
        palette_mask = np.ones((1, hole_count), dtype=bool)
        _mark_blocked(blocked, palette_mask, row, col)
        for i, color in enumerate(palette_colors):
            grid[row, col + i] = color
    else:
        if canvas_h < hole_count + 2:
            return None
        row = int(RNG.integers(1, canvas_h - hole_count))
        col = 0 if edge == "left" else canvas_w - 1
        palette_mask = np.ones((hole_count, 1), dtype=bool)
        _mark_blocked(blocked, palette_mask, row, col)
        for i, color in enumerate(palette_colors):
            grid[row + i, col] = color

    frame_pos = _place_mask(blocked, np.ones((frame_h, frame_w), dtype=bool), allow_edge=False)
    if frame_pos is None:
        return None
    frame_r0, frame_c0 = frame_pos
    _mark_blocked(blocked, np.ones((frame_h, frame_w), dtype=bool), frame_r0, frame_c0)
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = FRAME_COLOR
    for item in hole_infos:
        mask = item["mask"]
        r0 = frame_r0 + item["r0"]
        c0 = frame_c0 + item["c0"]
        sub = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        sub[mask] = BACKGROUND

    for spec in sprite_specs:
        mask = spec["mask"]
        pos = _place_mask(blocked, mask, allow_edge=False)
        if pos is None:
            return None
        r0, c0 = pos
        _mark_blocked(blocked, mask, r0, c0)
        sub = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        sub[mask] = int(spec["color"])

    bg_count = int(np.sum(grid == BACKGROUND))
    if bg_count <= frame_pixels:
        return None

    expected = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
    for idx, item in enumerate(hole_infos):
        mask = item["mask"]
        sub = expected[item["r0"] : item["r0"] + mask.shape[0], item["c0"] : item["c0"] + mask.shape[1]]
        sub[mask] = palette_colors[idx]

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
