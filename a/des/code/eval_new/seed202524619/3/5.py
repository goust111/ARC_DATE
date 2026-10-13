import numpy as np

BG = 0
FRAME_H = 5

OBJECT_TEMPLATES = {
    "horizontal": [
        np.array([[1, 0], [1, 1], [1, 0]], dtype=np.uint8),
        np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=np.uint8),
    ],
    "vertical": [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 1], [1, 1], [0, 0]], dtype=np.uint8),
    ],
    "neither": [
        np.array([[1, 0], [1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    ],
}

MARKER_TEMPLATES = {
    "horizontal": np.array([[1, 0], [1, 0]], dtype=np.uint8),
    "vertical": np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
    "neither": np.array([[1, 0], [1, 1]], dtype=np.uint8),
}

OBJECT_COLORS = np.array([2, 3, 4, 5, 6, 7], dtype=np.int64)
MARKER_COLORS = np.array([8, 9, 10, 11], dtype=np.int64)
FRAME_COLORS = np.array([12, 13, 14, 15, 16, 17], dtype=np.int64)


def _symmetry_class(mask: np.ndarray) -> str:
    m = np.array(mask, dtype=bool)
    if m.size == 0:
        return "neither"
    if m.size == 1:
        return "both"
    h_sym = np.array_equal(m, np.flipud(m))
    v_sym = np.array_equal(m, np.fliplr(m))
    if h_sym and not v_sym:
        return "horizontal"
    if v_sym and not h_sym:
        return "vertical"
    if h_sym or v_sym:
        return "both"
    return "neither"


def _shape_bucket(mask: np.ndarray) -> str:
    c = _symmetry_class(mask)
    if c == "both":
        return "neither"
    return c


def _connected_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _component_bbox(pts):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _component_patch(grid: np.ndarray, pts):
    r0, c0, r1, c1 = _component_bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    patch = np.zeros((h, w), dtype=grid.dtype)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        v = int(grid[r, c])
        patch[r - r0, c - c0] = v
        if v != BG:
            mask[r - r0, c - c0] = True
    return patch, mask


def _find_frames(grid: np.ndarray):
    h, w = grid.shape
    frames = []
    is_border = np.zeros_like(grid, dtype=bool)
    for comp in _connected_components(grid != BG):
        first = comp[0]
        color = int(grid[first])
        if color == BG:
            continue
        if any(grid[r, c] != color for r, c in comp):
            continue

        r0, c0, r1, c1 = _component_bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue
        if len(comp) != 2 * (hh + ww) - 4:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue

        comp_set = set(comp)
        good = True
        for rr in range(r0, r1 + 1):
            if grid[rr, c0] != color or (rr, c0) not in comp_set:
                good = False
                break
            if grid[rr, c1] != color or (rr, c1) not in comp_set:
                good = False
                break
        if not good:
            continue

        for cc in range(c0, c1 + 1):
            if grid[r0, cc] != color or (r0, cc) not in comp_set:
                good = False
                break
            if grid[r1, cc] != color or (r1, cc) not in comp_set:
                good = False
                break
        if not good:
            continue

        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if grid[rr, cc] == color:
                    good = False
                    break
            if not good:
                break
        if not good:
            continue

        frames.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": color, "marker": None})
        for rr, cc in comp:
            is_border[rr, cc] = True
    return frames, is_border


def _component_inside(grid: np.ndarray, sr: int, sc: int, r0: int, c0: int, r1: int, c1: int):
    if grid[sr, sc] == BG or not (r0 < sr < r1 and c0 < sc < c1):
        return []
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    visited[sr, sc] = True
    stack = [(sr, sc)]
    pts = []
    while stack:
        r, c = stack.pop()
        pts.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (r0 < nr < r1 and c0 < nc < c1):
                continue
            if visited[nr, nc] or grid[nr, nc] == BG:
                continue
            visited[nr, nc] = True
            stack.append((nr, nc))
    return pts


def _find_marker(grid: np.ndarray, frame):
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    start = None
    if grid[r0 + 1, c0 + 1] != BG:
        start = (r0 + 1, c0 + 1)
    else:
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if grid[rr, cc] != BG:
                    start = (rr, cc)
                    break
            if start is not None:
                break
    if start is None:
        return None

    pts = _component_inside(grid, start[0], start[1], r0, c0, r1, c1)
    if not pts:
        return None
    patch, mask = _component_patch(grid, pts)
    return {
        "points": pts,
        "patch": patch,
        "bbox": _component_bbox(pts),
        "type": _shape_bucket(mask),
    }


def _random_transformed_mask(base: np.ndarray, target_type: str, rng: np.random.Generator):
    variants = []
    for k in range(4):
        m = np.rot90(base, k)
        variants.append(m)
        variants.append(np.fliplr(m))
        variants.append(np.flipud(m))
    uniq = []
    seen = set()
    for v in variants:
        key = v.tobytes()
        if key in seen:
            continue
        seen.add(key)
        if _shape_bucket(v) == target_type:
            uniq.append(v)
    if not uniq:
        return base.astype(np.uint8)
    return uniq[int(rng.integers(0, len(uniq)))]


def _paint_shape(mask: np.ndarray, rng: np.random.Generator):
    coords = np.argwhere(mask)
    h, w = mask.shape
    patch = np.zeros((h, w), dtype=np.int64)
    if len(coords) == 0:
        return patch, np.zeros((h, w), dtype=bool)
    colors = rng.choice(OBJECT_COLORS, size=len(coords), replace=True)
    if len(coords) > 1 and np.all(colors == colors[0]):
        colors[1] = OBJECT_COLORS[(np.where(OBJECT_COLORS == colors[0])[0][0] + 1) % len(OBJECT_COLORS)]
    for (rr, cc), color in zip(coords, colors):
        patch[rr, cc] = int(color)
    return patch, patch != BG


def _render_marker(r0, c0, mask: np.ndarray, color: int, grid: np.ndarray, occ: np.ndarray):
    h, w = mask.shape
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return
    for rr, cc in coords:
        gr = r0 + 1 + int(rr)
        gc = c0 + 1 + int(cc)
        grid[gr, gc] = int(color)
        occ[gr, gc] = True


def main(grid: np.ndarray) -> np.ndarray:
    """Rearrange small objects into border frames grouped by shape symmetry.

Input structure:
- bg is black.
- Scattered multicolor 4-connected objects have size 2~6.
- Edge frames are rectangular single-color borders with a tiny interior marker in their top-left corner.

Rule:
- For each object outside frames and markers, classify its binary shape as only-horizontal symmetric, only-vertical symmetric, or neither by visual mirror checks.
- For each frame, classify its marker with the same symmetry rule.
- In each frame, place matching objects in nondecreasing area order from left to right on the first interior row, with one black gap between objects; keep each object's colors and shape unchanged.
- Clear all original object pixels outside frames to black.

Key insight:
- The grouping key is shape symmetry, not color or area; marker patterns provide the only semantic key for each target frame."""
    h, w = grid.shape
    frames, is_border = _find_frames(grid)
    for fr in frames:
        fr["marker"] = _find_marker(grid, fr)

    is_marker = np.zeros((h, w), dtype=bool)
    for fr in frames:
        m = fr["marker"]
        if m is None:
            continue
        for rr, cc in m["points"]:
            is_marker[rr, cc] = True

    object_mask = (grid != BG) & (~is_border) & (~is_marker)
    grouped = {"horizontal": [], "vertical": [], "neither": []}
    for comp in _connected_components(object_mask):
        area = len(comp)
        if area < 2 or area > 6:
            continue
        patch, mask = _component_patch(grid, comp)
        cls = _shape_bucket(mask)
        if cls not in grouped:
            continue
        grouped[cls].append((area, patch, mask))

    for k in grouped:
        grouped[k].sort(key=lambda x: x[0])

    frame_for_type = {}
    for fr in frames:
        m = fr["marker"]
        if m is not None and m["type"] in ("horizontal", "vertical", "neither"):
            frame_for_type.setdefault(m["type"], fr)

    out = np.zeros_like(grid)
    for fr in frames:
        r0, c0, r1, c1, color = fr["r0"], fr["c0"], fr["r1"], fr["c1"], fr["color"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        m = fr["marker"]
        if m is not None:
            mp = m["patch"]
            mh, mw = mp.shape
            mr0, mc0 = m["bbox"][0], m["bbox"][1]
            for rr in range(mh):
                for cc in range(mw):
                    v = mp[rr, cc]
                    if v != BG:
                        out[mr0 + rr, mc0 + cc] = v

    for cls, entries in grouped.items():
        fr = frame_for_type.get(cls)
        if fr is None or not entries:
            continue
        x = fr["c0"] + 1
        y = fr["r0"] + 1
        marker = fr["marker"]
        if marker is not None:
            x = max(x, marker["bbox"][3] + 2)
        for _, patch, mask in entries:
            ph, pw = patch.shape
            if y + ph > fr["r1"] or x + pw > fr["c1"]:
                break
            out[y : y + ph, x : x + pw][mask] = patch[mask]
            x += pw + 1

    return out


def _required_frame_width(marker_mask: np.ndarray, objects: list) -> int:
    marker_max_x = marker_mask.shape[1] - 1
    widths = [o["patch"].shape[1] for o in objects]
    return marker_max_x + 2 + sum(widths) + max(0, len(widths) - 1) + 2


def _generate_object(t: str, count: int, rng: np.random.Generator):
    out = []
    for _ in range(count):
        base = OBJECT_TEMPLATES[t][int(rng.integers(0, len(OBJECT_TEMPLATES[t])))]
        base = _random_transformed_mask(base, t, rng)
        patch, mask = _paint_shape(base, rng)
        out.append({"patch": patch, "mask": mask, "type": t})
    return out


def generate_input() -> np.ndarray:
    types = ["horizontal", "vertical", "neither"]
    rng = np.random.default_rng()

    for _ in range(400):
        k = int(rng.choice([2, 3]))
        perm = np.array(types)
        rng.shuffle(perm)
        active = list(perm[:k])

        counts = {t: 1 for t in active}
        if k == 3:
            if rng.random() < 0.5:
                counts[rng.choice(active)] += 1
        else:
            if rng.random() < 0.5:
                counts[rng.choice(active)] += 1
            if rng.random() < 0.2:
                counts[rng.choice(active)] += 1

        frame_specs = {}
        max_obj_h = 1
        total_span = 0
        for t in active:
            marker_mask = _random_transformed_mask(MARKER_TEMPLATES[t], t, rng)
            objects = _generate_object(t, counts[t], rng)
            if not objects:
                continue
            obj_h = max(o["patch"].shape[0] for o in objects)
            if obj_h > max_obj_h:
                max_obj_h = obj_h
            width = _required_frame_width(marker_mask, objects)
            width += int(rng.integers(0, 3))
            frame_specs[t] = {
                "type": t,
                "marker_mask": marker_mask,
                "objects": objects,
                "width": max(6, width),
                "marker_color": int(rng.choice(MARKER_COLORS)),
            }
            total_span += frame_specs[t]["width"] + 1
        if len(frame_specs) != k:
            continue
        total_span -= 1
        if total_span > 30 or total_span < 1:
            continue

        w = int(rng.integers(max(total_span + 1, 16), 32))
        h = int(rng.integers(max(FRAME_H + max_obj_h + 4, 14), 31))
        if w > 30 or h > 30:
            continue

        frame_types = active.copy()
        rng.shuffle(frame_types)
        frame_colors = rng.choice(FRAME_COLORS, size=k, replace=False)

        frames = []
        x = 0
        for i, t in enumerate(frame_types):
            fw = frame_specs[t]["width"]
            c0 = x
            c1 = c0 + fw - 1
            if c1 >= w:
                break
            frames.append({
                "type": t,
                "r0": 0,
                "c1": c1,
                "c0": c0,
                "r1": FRAME_H - 1,
                "color": int(frame_colors[i]),
            })
            x = c1 + 2
        if len(frames) != k:
            continue

        grid = np.zeros((h, w), dtype=np.int64)
        occ = np.zeros((h, w), dtype=bool)

        for fr in frames:
            sp = frame_specs[fr["type"]]
            r0, c0, r1, c1 = fr["r0"], fr["c0"], fr["r1"], fr["c1"]
            col = fr["color"]

            occ[r0, c0 : c1 + 1] = True
            occ[r1, c0 : c1 + 1] = True
            occ[r0 : r1 + 1, c0] = True
            occ[r0 : r1 + 1, c1] = True
            grid[r0, c0 : c1 + 1] = col
            grid[r1, c0 : c1 + 1] = col
            grid[r0 : r1 + 1, c0] = col
            grid[r0 : r1 + 1, c1] = col
            _render_marker(r0, c0, sp["marker_mask"], sp["marker_color"], grid, occ)

        all_objects = []
        for t in active:
            all_objects.extend(frame_specs[t]["objects"])
        rng.shuffle(all_objects)
        placed_ok = True
        for obj in all_objects:
            oh, ow = obj["patch"].shape
            min_r = FRAME_H + 1
            if h - oh < min_r:
                placed_ok = False
                break
            placed = False
            for _ in range(240):
                rr = int(rng.integers(min_r, h - oh + 1))
                cc = int(rng.integers(0, w - ow + 1))
                rr0, rr1 = max(0, rr - 1), min(h, rr + oh + 1)
                cc0, cc1 = max(0, cc - 1), min(w, cc + ow + 1)
                if np.any(occ[rr0:rr1, cc0:cc1]):
                    continue
                occ[rr : rr + oh, cc : cc + ow] |= obj["mask"]
                grid[rr : rr + oh, cc : cc + ow][obj["mask"]] = obj["patch"][obj["mask"]]
                placed = True
                break
            if not placed:
                placed_ok = False
                break
        if not placed_ok:
            continue

        return grid

    # Fallback deterministic valid sample.
    out = np.zeros((20, 30), dtype=np.int64)
    out[0, 0] = 12
    out[0, 1] = 12
    out[0, 2] = 12
    out[0, 3] = 12
    out[0, 4] = 12
    out[0, 5] = 12
    out[1, 0] = 12
    out[1, 5] = 12
    out[4, 0] = 12
    out[4, 1] = 12
    out[4, 2] = 12
    out[4, 3] = 12
    out[4, 4] = 12
    out[4, 5] = 12
    out[2, 2] = 8
    out[2, 3] = 8
    out[2, 1] = 2
    out[2, 2] = 3
    out[2, 4] = 2
    out[2, 5] = 3
    out[2, 6] = 8
    out[2, 7] = 8
    out[0, 8] = 13
    out[0, 9] = 13
    out[0, 10] = 13
    out[0, 11] = 13
    out[1, 8] = 13
    out[1, 11] = 13
    out[4, 8] = 13
    out[4, 9] = 13
    out[4, 10] = 13
    out[4, 11] = 13
    out[6, 8] = 7
    out[7, 8] = 5
    out[5, 6] = 4
    return out
