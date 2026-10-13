import random

import numpy as np


BG = 0
GREEN = 3
FRAME_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MOVE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERP_SIDES = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("top", "bottom"),
    "right": ("top", "bottom"),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _frame_border_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c0 + w):
        cells.add((r0, c))
        cells.add((r0 + h - 1, c))
    for r in range(r0 + 1, r0 + h - 1):
        cells.add((r, c0))
        cells.add((r, c0 + w - 1))
    return cells


def _missing_cells(r0: int, c0: int, h: int, w: int, side: str, gap_start: int, gap_len: int) -> set[tuple[int, int]]:
    if side == "top":
        return {(r0, c0 + gap_start + k) for k in range(gap_len)}
    if side == "bottom":
        return {(r0 + h - 1, c0 + gap_start + k) for k in range(gap_len)}
    if side == "left":
        return {(r0 + gap_start + k, c0) for k in range(gap_len)}
    return {(r0 + gap_start + k, c0 + w - 1) for k in range(gap_len)}


def _make_spec(
    color: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    side: str,
    gap_start: int,
    gap_len: int,
    dist: int,
) -> dict:
    missing = _missing_cells(r0, c0, h, w, side, gap_start, gap_len)
    border = _frame_border_cells(r0, c0, h, w)
    frame = border - missing
    orow, ocol = OUTWARD[side]
    fragment = {(r + orow * dist, c + ocol * dist) for r, c in missing}
    drow, dcol = MOVE[side]
    travel: set[tuple[int, int]] = set()
    for step in range(dist):
        for r, c in fragment:
            travel.add((r + drow * step, c + dcol * step))
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
        "side": side,
        "gap_start": int(gap_start),
        "gap_len": int(gap_len),
        "dist": int(dist),
        "frame": frame,
        "missing": missing,
        "fragment": fragment,
        "travel": travel,
    }


def _cells_in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _frame_from_component(coords: list[tuple[int, int]], color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = border & ~mask
    if not missing.any():
        return None
    if missing[0, 0] or missing[0, -1] or missing[-1, 0] or missing[-1, -1]:
        return None

    side_indices = {
        "top": np.where(missing[0])[0].tolist(),
        "bottom": np.where(missing[-1])[0].tolist(),
        "left": np.where(missing[:, 0])[0].tolist(),
        "right": np.where(missing[:, -1])[0].tolist(),
    }
    nonempty = [side for side, idxs in side_indices.items() if idxs]
    if len(nonempty) != 1:
        return None
    side = nonempty[0]
    idxs = side_indices[side]
    if idxs != list(range(idxs[0], idxs[-1] + 1)):
        return None
    if side in ("top", "bottom"):
        if idxs[0] <= 0 or idxs[-1] >= w - 1:
            return None
    else:
        if idxs[0] <= 0 or idxs[-1] >= h - 1:
            return None
    if int(missing.sum()) != len(idxs):
        return None

    miss_abs = {(r0 + r, c0 + c) for r in range(h) for c in range(w) if missing[r, c]}
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "pixels": set(coords),
        "missing": miss_abs,
    }


def _find_frames(g: np.ndarray, bg: int) -> tuple[list[dict], dict[int, list[list[tuple[int, int]]]]]:
    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _components(g == color)
        comps_by_color[color] = comps
        candidates = [_frame_from_component(comp, color) for comp in comps]
        candidates = [cand for cand in candidates if cand is not None]
        if candidates:
            assert len(candidates) == 1, f"color {color} must contain exactly one frame"
            frames.append(candidates[0])
    assert frames, "expected at least one gapped frame"
    return frames, comps_by_color


def _fragment_on_side(fragment_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int], side: str) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    rr0, rc0, rr1, rc1 = fragment_bbox
    if side == "top":
        return rr1 < fr0
    if side == "bottom":
        return rr0 > fr1
    if side == "left":
        return rc1 < fc0
    return rc0 > fc1


def _match_fragment(frame: dict, comps: list[list[tuple[int, int]]]) -> dict:
    want_shape = _normalize(frame["missing"])
    want_set = set(frame["missing"])
    side = frame["side"]
    move_r, move_c = MOVE[side]
    miss_r0, miss_c0, _, _ = _bbox(frame["missing"])
    candidates = []
    for comp in comps:
        comp_set = set(comp)
        if comp_set == frame["pixels"]:
            continue
        if _normalize(comp) != want_shape:
            continue
        if not _fragment_on_side(_bbox(comp), frame["bbox"], side):
            continue
        comp_r0, comp_c0, _, _ = _bbox(comp)
        shift_r = miss_r0 - comp_r0
        shift_c = miss_c0 - comp_c0
        shifted = {(r + shift_r, c + shift_c) for r, c in comp}
        if shifted != want_set:
            continue
        if move_r != 0:
            if shift_c != 0 or shift_r == 0 or (1 if shift_r > 0 else -1) != move_r:
                continue
            steps = abs(shift_r)
        else:
            if shift_r != 0 or shift_c == 0 or (1 if shift_c > 0 else -1) != move_c:
                continue
            steps = abs(shift_c)
        candidates.append({"pixels": comp_set, "steps": int(steps)})
    assert len(candidates) == 1, f"frame color {frame['color']} must have exactly one matching fragment"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each frame's matching outside fragment inward through its gap, completing the frame and marking frame collisions green.

    Input structure:
    - The background is black.
    - Several differently colored hollow rectangular frames are present; each frame has exactly one contiguous gap on one side of its 1-pixel border.
    - Outside each frame, on the same side as its gap, there is a separate same-color fragment whose shape exactly matches the missing border cells. Other differently colored distractor fragments may also appear.

    Rule:
    - Detect every gapped rectangular frame and determine which side contains its missing border segment.
    - For each frame, find the separate same-color fragment whose shape is a translated copy of the missing segment, then translate that fragment straight inward until it lands exactly on the gap.
    - Clear the fragment's original location to black and fill the frame gap with the frame color.
    - If any intermediate translated fragment position overlaps boundary pixels of another frame, recolor those overlapped boundary pixels green.

    Key insight:
    - The missing side tells both where the matching fragment must be and which cardinal direction it moves; the fragment is identified by same-color shape matching rather than by absolute position.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    frames, comps_by_color = _find_frames(g, bg)
    all_frame_pixels = set().union(*(frame["pixels"] for frame in frames))

    out = g.copy()
    collisions: set[tuple[int, int]] = set()
    for frame in frames:
        fragment = _match_fragment(frame, comps_by_color[frame["color"]])
        for r, c in fragment["pixels"]:
            out[r, c] = bg

        move_r, move_c = MOVE[frame["side"]]
        other_frames = all_frame_pixels - frame["pixels"]
        for step in range(1, fragment["steps"]):
            for r, c in fragment["pixels"]:
                cell = (r + move_r * step, c + move_c * step)
                if cell in other_frames:
                    collisions.add(cell)

        for r, c in frame["missing"]:
            out[r, c] = frame["color"]

    for r, c in collisions:
        out[r, c] = GREEN
    return out


def _transform_coord(r: int, c: int, h: int, w: int, source_side: str) -> tuple[int, int]:
    if source_side == "top":
        return r, c
    if source_side == "bottom":
        return h - 1 - r, c
    if source_side == "left":
        return w - 1 - c, r
    return c, h - 1 - r


def _transform_bundle(bundle: dict, source_side: str) -> dict:
    h, w = bundle["shape"]
    if source_side in ("top", "bottom"):
        new_h, new_w = h, w
    else:
        new_h, new_w = w, h
    occupied = {
        _transform_coord(r, c, h, w, source_side): color for (r, c), color in bundle["occupied"].items()
    }
    path = {_transform_coord(r, c, h, w, source_side) for r, c in bundle["path"]}
    return {"shape": (new_h, new_w), "occupied": occupied, "path": path}


def _canonical_top_collision_pair(color_a: int, color_b: int) -> dict | None:
    for _ in range(200):
        gap_len = random.randint(1, 3)
        h_b = random.randint(5, 8)
        w_b = random.randint(max(5, gap_len + 4), 9)
        h_a = random.randint(5, 8)
        w_a = random.randint(max(5, gap_len + 4), 9)
        gap_a = random.randint(1, w_a - gap_len - 1)
        side_b = random.choice(("left", "right"))
        gap_b_len = random.randint(1, min(3, h_b - 2))
        gap_b = random.randint(1, h_b - gap_b_len - 1)
        dist_b = random.randint(2, 4)
        gap_between = random.randint(2, 4)
        extra_above = random.randint(2, 4)

        x = random.randint(1, w_b - gap_len - 1)
        r_b, c_b = 0, 0
        r_a = h_b + gap_between
        c_a = x - gap_a
        dist_a = h_b + gap_between + extra_above

        spec_a = _make_spec(color_a, r_a, c_a, h_a, w_a, "top", gap_a, gap_len, dist_a)
        spec_b = _make_spec(color_b, r_b, c_b, h_b, w_b, side_b, gap_b, gap_b_len, dist_b)

        cells = (
            spec_a["frame"] | spec_a["fragment"] | spec_a["travel"] |
            spec_b["frame"] | spec_b["fragment"] | spec_b["travel"]
        )
        min_r, min_c, max_r, max_c = _bbox(cells)
        shift_r = -min_r
        shift_c = -min_c

        def shift_cells(src: set[tuple[int, int]]) -> set[tuple[int, int]]:
            return {(r + shift_r, c + shift_c) for r, c in src}

        spec_a = {
            **spec_a,
            "frame": shift_cells(spec_a["frame"]),
            "fragment": shift_cells(spec_a["fragment"]),
            "travel": shift_cells(spec_a["travel"]),
        }
        spec_b = {
            **spec_b,
            "frame": shift_cells(spec_b["frame"]),
            "fragment": shift_cells(spec_b["fragment"]),
            "travel": shift_cells(spec_b["travel"]),
        }

        occ_a = spec_a["frame"] | spec_a["fragment"]
        occ_b = spec_b["frame"] | spec_b["fragment"]
        if occ_a & occ_b:
            continue
        if spec_a["travel"] & spec_b["fragment"]:
            continue
        if spec_b["travel"] & occ_a:
            continue
        collision = spec_a["travel"] & spec_b["frame"]
        if not collision:
            continue

        occupied = {cell: color_a for cell in occ_a}
        occupied.update({cell: color_b for cell in occ_b})
        shape = (max_r - min_r + 1, max_c - min_c + 1)
        return {
            "shape": shape,
            "occupied": occupied,
            "path": spec_a["travel"] | spec_b["travel"],
        }
    return None


def _collision_pair_bundle(color_a: int, color_b: int, source_side: str) -> dict | None:
    base = _canonical_top_collision_pair(color_a, color_b)
    if base is None:
        return None
    return _transform_bundle(base, source_side)


def _reserve(mask: np.ndarray, cells: set[tuple[int, int]] | dict[tuple[int, int], int], pad: int = 1) -> None:
    h, w = mask.shape
    pts = cells.keys() if isinstance(cells, dict) else cells
    for r, c in pts:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                mask[rr, cc] = True


def _any_reserved(mask: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    return any(mask[r, c] for r, c in cells)


def _draw_spec(grid: np.ndarray, spec: dict) -> None:
    for r, c in spec["frame"]:
        grid[r, c] = spec["color"]
    for r, c in spec["fragment"]:
        grid[r, c] = spec["color"]


def _independent_spec(h: int, w: int, color: int, reserved: np.ndarray) -> dict | None:
    for _ in range(300):
        side = random.choice(SIDES)
        fh = random.randint(5, 9)
        fw = random.randint(5, 9)
        max_gap = min(3, (fw - 2) if side in ("top", "bottom") else (fh - 2))
        gap_len = random.randint(1, max_gap)
        dist = random.randint(2, 5)
        if side in ("top", "bottom"):
            gap_start = random.randint(1, fw - gap_len - 1)
            r_lo = dist if side == "top" else 0
            r_hi = h - fh if side == "top" else h - fh - dist
            c_lo, c_hi = 0, w - fw
        else:
            gap_start = random.randint(1, fh - gap_len - 1)
            r_lo, r_hi = 0, h - fh
            c_lo = dist if side == "left" else 0
            c_hi = w - fw if side == "left" else w - fw - dist
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        spec = _make_spec(color, r0, c0, fh, fw, side, gap_start, gap_len, dist)
        occ = spec["frame"] | spec["fragment"]
        if not _cells_in_bounds(occ | spec["travel"], h, w):
            continue
        if _any_reserved(reserved, occ | spec["travel"]):
            continue
        return spec
    return None


def _random_distractor_shape() -> set[tuple[int, int]]:
    shapes = [
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0)},
        {(0, 0), (0, 1), (1, 0)},
        {(0, 0), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (1, 1)},
        {(0, 0), (0, 1), (1, 0), (1, 1)},
    ]
    return set(random.choice(shapes))


def _place_distractor(grid: np.ndarray, color: int, reserved: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(100):
        shape = _random_distractor_shape()
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if sh > h or sw > w:
            return False
        r0 = random.randint(0, h - sh)
        c0 = random.randint(0, w - sw)
        cells = {(r0 + r, c0 + c) for r, c in shape}
        if _any_reserved(reserved, cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        _reserve(reserved, cells, pad=1)
        return True
    return False


def _generate_candidate() -> tuple[np.ndarray, bool]:
    frame_count = random.randint(2, 4)
    include_collision = random.random() < 0.5
    frame_colors = random.sample(FRAME_COLORS, frame_count)
    distractor_colors = [c for c in FRAME_COLORS if c not in frame_colors]

    if include_collision:
        pair = _collision_pair_bundle(frame_colors[0], frame_colors[1], random.choice(SIDES))
        assert pair is not None
        min_h = max(22, pair["shape"][0] + 2)
        min_w = max(22, pair["shape"][1] + 2)
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
    else:
        pair = None
        h = random.randint(22, 30)
        w = random.randint(22, 30)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    if pair is not None:
        ph, pw = pair["shape"]
        off_r = random.randint(0, h - ph)
        off_c = random.randint(0, w - pw)
        shifted_occ = {(r + off_r, c + off_c): color for (r, c), color in pair["occupied"].items()}
        shifted_path = {(r + off_r, c + off_c) for r, c in pair["path"]}
        for (r, c), color in shifted_occ.items():
            grid[r, c] = color
        _reserve(reserved, shifted_occ, pad=1)
        _reserve(reserved, shifted_path, pad=1)
        used = 2
    else:
        used = 0

    for color in frame_colors[used:]:
        spec = _independent_spec(h, w, color, reserved)
        assert spec is not None
        _draw_spec(grid, spec)
        _reserve(reserved, spec["frame"] | spec["fragment"], pad=1)
        _reserve(reserved, spec["travel"], pad=1)

    distractor_count = random.randint(0, min(3, len(distractor_colors)))
    for color in random.sample(distractor_colors, distractor_count):
        if not _place_distractor(grid, color, reserved):
            break

    return grid, include_collision


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, include_collision = _generate_candidate()
        out = main(grid)
        if include_collision and not np.any(out == GREEN):
            continue
        if not include_collision and np.any(out == GREEN):
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
