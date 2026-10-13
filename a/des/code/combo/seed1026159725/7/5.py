import numpy as np


_RNG = np.random.default_rng()
_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, pts[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _mask_coords(mask: np.ndarray):
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _has_horizontal_axis(coords) -> bool:
    pts = set(coords)
    r0, _, r1, _ = _bbox_from_coords(coords)
    total = r0 + r1
    return all((int(total - r), int(c)) in pts for r, c in pts)


def _has_vertical_axis(coords) -> bool:
    pts = set(coords)
    _, c0, _, c1 = _bbox_from_coords(coords)
    total = c0 + c1
    return all((int(r), int(total - c)) in pts for r, c in pts)


def _has_diagonal_axis(coords) -> bool:
    pts = set(coords)
    diffs = [r - c for r, c in coords]
    total = min(diffs) + max(diffs)
    if total % 2 != 0:
        return False
    axis_offset = total // 2
    return all((int(c + axis_offset), int(r - axis_offset)) in pts for r, c in pts)


def _detect_axis(coords):
    axes = []
    if _has_horizontal_axis(coords):
        axes.append("horizontal")
    if _has_vertical_axis(coords):
        axes.append("vertical")
    if _has_diagonal_axis(coords):
        axes.append("diagonal")
    assert len(axes) == 1, f"expected one symmetry axis, got {axes}"
    return axes[0]


def _axis_span(coords, axis: str) -> int:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if axis == "horizontal":
        return int(c1 - c0 + 1)
    if axis == "vertical":
        return int(r1 - r0 + 1)
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    assert side_h == side_w, "diagonal-symmetric objects should have a square bbox"
    return side_h


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each foreground object along its unique reflection-symmetry axis.

    Horizontal-axis objects move right by their width.
    Vertical-axis objects move down by their height.
    Diagonal-axis objects move down-right by their square-bbox side length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = np.full((h, w), bg, dtype=int)

    for comp in _find_components(g, bg):
        coords = comp["coords"]
        axis = _detect_axis(coords)
        dist = _axis_span(coords, axis)
        if axis == "horizontal":
            dr, dc = 0, dist
        elif axis == "vertical":
            dr, dc = dist, 0
        else:
            dr, dc = dist, dist
        for r, c in coords:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(comp["color"])
    return out


def _mask_axis(mask: np.ndarray) -> str | None:
    if not _is_connected(mask):
        return None
    try:
        return _detect_axis(_mask_coords(mask))
    except AssertionError:
        return None


def _adjacent_to_mask(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    return any(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))


def _gen_horizontal_mask() -> np.ndarray:
    for _ in range(200):
        height = int(_RNG.choice([3, 5, 7]))
        min_width = max(5, height + 1)
        width = int(_RNG.integers(min_width, min_width + 4))
        center = height // 2
        mask = np.zeros((height, width), dtype=bool)
        mask[center, :] = True

        profile = np.zeros(width, dtype=int)
        arm_count = int(_RNG.integers(2, min(6, width + 1)))
        cols = _RNG.choice(width, size=arm_count, replace=False)
        for c in cols:
            arm = int(_RNG.integers(1, center + 1))
            profile[c] = arm
            for d in range(1, arm + 1):
                mask[center - d, c] = True
                mask[center + d, c] = True

        if np.array_equal(profile, profile[::-1]):
            continue
        if _mask_axis(mask) == "horizontal":
            return mask
    raise RuntimeError("failed to generate horizontal object")


def _gen_vertical_mask() -> np.ndarray:
    for _ in range(200):
        width = int(_RNG.choice([3, 5, 7]))
        min_height = max(5, width + 1)
        height = int(_RNG.integers(min_height, min_height + 4))
        center = width // 2
        mask = np.zeros((height, width), dtype=bool)
        mask[:, center] = True

        profile = np.zeros(height, dtype=int)
        arm_count = int(_RNG.integers(2, min(6, height + 1)))
        rows = _RNG.choice(height, size=arm_count, replace=False)
        for r in rows:
            arm = int(_RNG.integers(1, center + 1))
            profile[r] = arm
            for d in range(1, arm + 1):
                mask[r, center - d] = True
                mask[r, center + d] = True

        if np.array_equal(profile, profile[::-1]):
            continue
        if _mask_axis(mask) == "vertical":
            return mask
    raise RuntimeError("failed to generate vertical object")


def _gen_diagonal_mask() -> np.ndarray:
    for _ in range(200):
        side = int(_RNG.integers(4, 8))
        mask = np.zeros((side, side), dtype=bool)
        for i in range(side):
            mask[i, i] = True
        for i in range(side - 1):
            mask[i, i + 1] = True
            mask[i + 1, i] = True

        candidates = [(r, c) for r in range(side) for c in range(r + 2, side)]
        _RNG.shuffle(candidates)
        extra_target = int(_RNG.integers(0, side - 1))
        added = 0
        for r, c in candidates:
            if added >= extra_target:
                break
            if _adjacent_to_mask(mask, r, c) and _adjacent_to_mask(mask, c, r):
                mask[r, c] = True
                mask[c, r] = True
                added += 1

        if _mask_axis(mask) == "diagonal":
            return mask
    raise RuntimeError("failed to generate diagonal object")


def _axis_delta(axis: str, dist: int) -> tuple[int, int]:
    if axis == "horizontal":
        return 0, dist
    if axis == "vertical":
        return dist, 0
    return dist, dist


def _fits_with_clearance(occupied: np.ndarray, coords, clearance: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        r0 = max(0, r - clearance)
        r1 = min(h, r + clearance + 1)
        c0 = max(0, c - clearance)
        c1 = min(w, c + clearance + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _candidate_anchors(obj, grid_h: int, grid_w: int, want_clip: bool):
    mask = obj["mask"]
    coords = obj["coords"]
    local_h, local_w = mask.shape
    dist = int(obj["dist"])
    axis = obj["axis"]
    dr, dc = _axis_delta(axis, dist)
    anchors = []
    for r0 in range(grid_h - local_h + 1):
        for c0 in range(grid_w - local_w + 1):
            src = [(int(r0 + r), int(c0 + c)) for r, c in coords]
            dst = []
            for r, c in coords:
                rr = int(r0 + r + dr)
                cc = int(c0 + c + dc)
                if 0 <= rr < grid_h and 0 <= cc < grid_w:
                    dst.append((rr, cc))
            if not dst:
                continue
            clipped = len(dst) < len(coords)
            if clipped != want_clip:
                continue
            anchors.append((r0, c0, src, dst))
    return anchors


def _choose_axes(n_objects: int):
    global _GEN_CALL_COUNT
    extra_axis = ["horizontal", "vertical", "diagonal"][_GEN_CALL_COUNT % 3]
    _GEN_CALL_COUNT += 1
    axes = ["horizontal", "vertical", "diagonal"]
    if n_objects == 4:
        axes.append(extra_axis)
    _RNG.shuffle(axes)
    return axes


def _gen_object(axis: str) -> dict:
    if axis == "horizontal":
        mask = _gen_horizontal_mask()
    elif axis == "vertical":
        mask = _gen_vertical_mask()
    else:
        mask = _gen_diagonal_mask()
    coords = _mask_coords(mask)
    return {
        "axis": axis,
        "mask": mask,
        "coords": coords,
        "dist": _axis_span(coords, axis),
        "area": len(coords),
    }


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(_RNG.integers(16, 25))
    grid_w = int(_RNG.integers(16, 25))
    bg = int(_RNG.integers(0, 10))
    n_objects = 3 if float(_RNG.random()) < 0.6 else 4
    axes = _choose_axes(n_objects)
    colors = list(map(int, _RNG.choice([c for c in range(10) if c != bg], size=n_objects, replace=False)))
    objects = [_gen_object(axis) for axis in axes]

    clip_candidates = [i for i, obj in enumerate(objects) if _candidate_anchors(obj, grid_h, grid_w, True)]
    clip_index = None
    if clip_candidates and float(_RNG.random()) < 0.45:
        clip_index = int(_RNG.choice(clip_candidates))

    order = sorted(range(n_objects), key=lambda i: (-objects[i]["area"], i))
    src_occ = np.zeros((grid_h, grid_w), dtype=bool)
    dst_occ = np.zeros((grid_h, grid_w), dtype=bool)
    placements = [None] * n_objects

    for idx in order:
        obj = objects[idx]
        anchors = _candidate_anchors(obj, grid_h, grid_w, want_clip=(idx == clip_index))
        if not anchors:
            return None
        _RNG.shuffle(anchors)
        placed = False
        for r0, c0, src, dst in anchors:
            if not _fits_with_clearance(src_occ, src, clearance=1):
                continue
            if not _fits_with_clearance(dst_occ, dst, clearance=1):
                continue
            placements[idx] = (r0, c0, src, dst)
            for r, c in src:
                src_occ[r, c] = True
            for r, c in dst:
                dst_occ[r, c] = True
            placed = True
            break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for idx, color in enumerate(colors):
        _, _, src, _ = placements[idx]
        for r, c in src:
            grid[r, c] = int(color)

    if _get_bg(grid) != bg:
        return None
    comps = _find_components(grid, bg)
    if len(comps) != n_objects:
        return None
    if {int(comp["color"]) for comp in comps} != set(colors):
        return None
    try:
        detected_axes = {int(comp["color"]): _detect_axis(comp["coords"]) for comp in comps}
    except AssertionError:
        return None
    for idx, color in enumerate(colors):
        if detected_axes[int(color)] != axes[idx]:
            return None

    out = main(grid)
    if np.count_nonzero(out != bg) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
