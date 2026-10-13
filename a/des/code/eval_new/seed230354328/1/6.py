import random
from collections import Counter

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_RETRIES = 300
SHAPE_RETRIES = 120
PACK_RETRIES = 120
PACK_CANVAS = 17


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _corner_coord(corner: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if corner == 0:
        return 0, 0
    if corner == 1:
        return 0, w - 1
    if corner == 2:
        return h - 1, w - 1
    if corner == 3:
        return h - 1, 0
    raise ValueError(f"invalid corner index: {corner}")


def _ring_corner_coord(corner: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if corner == 0:
        return 0, 0
    if corner == 1:
        return 0, w + 1
    if corner == 2:
        return h + 1, w + 1
    if corner == 3:
        return h + 1, 0
    raise ValueError(f"invalid corner index: {corner}")


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    return len(_components(mask, connectivity=4)) == 1


def _bbox_of_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _points_to_mask(points: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox_of_points(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return (r0, c0, r1, c1), mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _outline_from_shape(shape: np.ndarray) -> np.ndarray:
    padded = np.pad(shape, 1, constant_values=False)
    return _dilate8(padded) & (~padded)


def _missing_corner(mask: np.ndarray) -> int:
    missing = []
    for corner in range(4):
        rr, cc = _corner_coord(corner, mask.shape)
        if not mask[rr, cc]:
            missing.append(corner)
    assert len(missing) == 1, "mask must miss exactly one bbox corner"
    return int(missing[0])


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(color)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best = None
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        pts = np.argwhere(grid == int(color))
        if len(pts) == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        border = np.zeros_like(grid, dtype=bool)
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        if np.array_equal(grid == int(color), border):
            area = int(border.sum())
            cand = (area, int(color), (int(r0), int(c0), int(r1), int(c1)))
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None, "expected a unique rectangular frame"
    return best[1], best[2]


def _find_regions(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int, bg: int
) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame_bbox
    ir0, ic0 = fr0 + 1, fc0 + 1
    inner = grid[ir0:fr1, ic0:fc1]
    regions = []
    for pts in _components(inner == int(bg), connectivity=4):
        if any(r == 0 or c == 0 or r == inner.shape[0] - 1 or c == inner.shape[1] - 1 for r, c in pts):
            continue
        abs_pts = [(ir0 + r, ic0 + c) for r, c in pts]
        color_counts: Counter[int] = Counter()
        pt_set = set(abs_pts)
        for r, c in abs_pts:
            for dr, dc in DIRS8:
                nr = r + dr
                nc = c + dc
                if (nr, nc) in pt_set or not (fr0 <= nr <= fr1 and fc0 <= nc <= fc1):
                    continue
                v = int(grid[nr, nc])
                if v != int(bg) and v != int(frame_color):
                    color_counts[v] += 1
        assert color_counts, "each enclosed region must touch a colored outline"
        color = int(color_counts.most_common(1)[0][0])
        (r0, c0, r1, c1), shape = _points_to_mask(abs_pts)
        patch = grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
        missing = []
        for corner in range(4):
            rr, cc = _ring_corner_coord(corner, shape.shape)
            if int(patch[rr, cc]) != color:
                missing.append(corner)
        assert len(missing) == 1, "outline must expose exactly one missing ring corner"
        target = shape.copy()
        cr, cc = _corner_coord(int(missing[0]), target.shape)
        target[cr, cc] = False
        regions.append(
            {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": shape,
                "corner": int(missing[0]),
                "target": target,
            }
        )
    regions.sort(key=lambda d: int(d["color"]))
    assert regions, "template must contain at least one enclosed region"
    assert len({int(r["color"]) for r in regions}) == len(regions), "region colors must be unique"
    return regions


def _find_fragments(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int
) -> dict[int, dict]:
    fr0, fc0, fr1, fc1 = frame_bbox
    outside = np.ones_like(grid, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    pieces = {}
    for pts in _components(outside & (grid != int(bg)), connectivity=4):
        colors = {int(grid[r, c]) for r, c in pts}
        assert len(colors) == 1, "each fragment must be monochrome"
        color = colors.pop()
        bbox, mask = _points_to_mask(pts)
        _missing_corner(mask)
        pieces[int(color)] = {"bbox": bbox, "mask": mask, "points": pts}
    assert pieces, "expected scattered fragments outside the frame"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered monochrome fragments into the matching outlined template regions inside the central frame.

    Input structure:
    - The background is black, and a large grey rectangular frame sits near the middle of the grid.
    - Inside that frame, each target region is a black connected shape enclosed by a one-pixel colored outline.
      Each outline has exactly one missing corner pixel in its outer ring.
    - Outside the frame, there is one solid monochrome fragment per outline color. Each fragment equals the
      corresponding region shape except that one bbox-corner pixel is missing, and the fragment may be rotated.

    Rule:
    - Find the grey rectangular frame, then locate the enclosed black regions inside it.
    - Match each outside fragment to the inside region with the same color outline.
    - Use the outline's missing ring corner and the fragment's missing bbox corner to determine the needed
      rotation, then place the rotated fragment into that region's bbox.
    - Erase the fragment from its original location and keep the frame plus outlines unchanged.

    Key insight:
    - Color identifies which fragment belongs to which region, but the missing corner in the outline and the
      missing corner in the fragment are the orientation markers that fix the fragment's rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    regions = _find_regions(g, frame_bbox, frame_color, bg)
    fragments = _find_fragments(g, frame_bbox, bg)

    out = g.copy()
    for frag in fragments.values():
        for r, c in frag["points"]:
            out[r, c] = int(bg)

    for region in regions:
        color = int(region["color"])
        assert color in fragments, "every outline color must have a matching fragment"
        frag = fragments[color]
        chosen = None
        for rot in range(4):
            rotated = np.rot90(frag["mask"], rot)
            if rotated.shape != region["target"].shape:
                continue
            if _missing_corner(rotated) != int(region["corner"]):
                continue
            if np.array_equal(rotated, region["target"]):
                chosen = rotated
                break
        assert chosen is not None, "a matching rotation must exist for every fragment"
        r0, c0, _r1, _c1 = region["bbox"]
        _paint_mask(out, chosen, int(r0), int(c0), color)
    return out


def _make_shape(rng: random.Random, max_dim: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = rng.randint(4, max_dim)
        w = rng.randint(4, max_dim)
        shape = np.ones((h, w), dtype=bool)
        notch_count = rng.randint(1, 3)
        for _ in range(notch_count):
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                max_depth = min(2, h - 2)
                max_span = w - 2
                if max_depth < 1 or max_span < 1:
                    continue
                depth = rng.randint(1, max_depth)
                span = rng.randint(1, max_span)
                start = rng.randint(1, w - span - 1)
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                shape[rows, start : start + span] = False
            else:
                max_depth = min(2, w - 2)
                max_span = h - 2
                if max_depth < 1 or max_span < 1:
                    continue
                depth = rng.randint(1, max_depth)
                span = rng.randint(1, max_span)
                start = rng.randint(1, h - span - 1)
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                shape[start : start + span, cols] = False

        if np.all(shape):
            continue
        if not all(shape[_corner_coord(corner, shape.shape)] for corner in range(4)):
            continue
        if int(shape.sum()) < 8 or not _is_connected(shape):
            continue

        valid = True
        for corner in range(4):
            trial = shape.copy()
            rr, cc = _corner_coord(corner, shape.shape)
            trial[rr, cc] = False
            if not _is_connected(trial):
                valid = False
                break
            if not trial[0].any() or not trial[-1].any() or not trial[:, 0].any() or not trial[:, -1].any():
                valid = False
                break
        if valid:
            return shape
    raise RuntimeError("failed to generate a valid region shape")


def _pack_regions(rng: random.Random, specs: list[dict]) -> tuple[tuple[int, int], dict[int, tuple[int, int]]]:
    order = list(range(len(specs)))
    order.sort(key=lambda idx: -int((specs[idx]["shape"].shape[0] + 2) * (specs[idx]["shape"].shape[1] + 2)))
    for _ in range(PACK_RETRIES):
        occupied = np.zeros((PACK_CANVAS, PACK_CANVAS), dtype=bool)
        placed: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            h, w = specs[idx]["shape"].shape
            rh, rw = h + 2, w + 2
            candidates = [(r, c) for r in range(PACK_CANVAS - rh + 1) for c in range(PACK_CANVAS - rw + 1)]
            rng.shuffle(candidates)
            found = False
            for r0, c0 in candidates:
                rr0 = max(0, r0 - 1)
                rr1 = min(PACK_CANVAS, r0 + rh + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(PACK_CANVAS, c0 + rw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                occupied[rr0:rr1, cc0:cc1] = True
                placed[idx] = (r0, c0)
                found = True
                break
            if not found:
                ok = False
                break
        if not ok:
            continue

        min_r = min(placed[idx][0] for idx in placed)
        min_c = min(placed[idx][1] for idx in placed)
        max_r = max(placed[idx][0] + specs[idx]["shape"].shape[0] + 1 for idx in placed)
        max_c = max(placed[idx][1] + specs[idx]["shape"].shape[1] + 1 for idx in placed)
        free_top = min_r
        free_left = min_c
        free_bottom = PACK_CANVAS - 1 - max_r
        free_right = PACK_CANVAS - 1 - max_c
        extra_top = rng.randint(0, min(1, free_top))
        extra_left = rng.randint(0, min(1, free_left))
        extra_bottom = rng.randint(0, min(1, free_bottom))
        extra_right = rng.randint(0, min(1, free_right))
        inner_r0 = min_r - extra_top
        inner_c0 = min_c - extra_left
        inner_r1 = max_r + extra_bottom
        inner_c1 = max_c + extra_right
        inner_h = inner_r1 - inner_r0 + 1
        inner_w = inner_c1 - inner_c0 + 1
        if inner_h > 17 or inner_w > 17:
            continue

        positions = {}
        for idx, (r0, c0) in placed.items():
            positions[idx] = (r0 - inner_r0 + 1, c0 - inner_c0 + 1)
        return (int(inner_h), int(inner_w)), positions
    raise RuntimeError("failed to pack template regions")


def _draw_region(
    grid: np.ndarray,
    top: int,
    left: int,
    shape: np.ndarray,
    color: int,
    corner: int,
) -> None:
    outline = _outline_from_shape(shape)
    rr, cc = _ring_corner_coord(corner, shape.shape)
    outline = outline.copy()
    outline[rr, cc] = False
    _paint_mask(grid, outline, top - 1, left - 1, color)


def _place_fragments(
    rng: random.Random,
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    piece_specs: list[dict],
) -> list[dict] | None:
    h, w = grid.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    occupied = grid != int(BG)
    placements: list[dict] = []

    ordered = list(range(len(piece_specs)))
    ordered.sort(key=lambda idx: -(piece_specs[idx]["mask"].shape[0] * piece_specs[idx]["mask"].shape[1]))
    for idx in ordered:
        mask = piece_specs[idx]["mask"]
        ph, pw = mask.shape
        bands = []
        if fr0 - ph >= 1:
            bands.append(("top", 0, fr0 - ph - 1, 0, w - pw))
        if h - ph >= fr1 + 2:
            bands.append(("bottom", fr1 + 2, h - ph, 0, w - pw))
        if fc0 - pw >= 1:
            bands.append(("left", 0, h - ph, 0, fc0 - pw - 1))
        if w - pw >= fc1 + 2:
            bands.append(("right", 0, h - ph, fc1 + 2, w - pw))
        rng.shuffle(bands)
        placed = None
        for _name, rmin, rmax, cmin, cmax in bands:
            if rmin > rmax or cmin > cmax:
                continue
            candidates = [(r, c) for r in range(rmin, rmax + 1) for c in range(cmin, cmax + 1)]
            rng.shuffle(candidates)
            for top, left in candidates:
                rr0 = max(0, top - 1)
                rr1 = min(h, top + ph + 1)
                cc0 = max(0, left - 1)
                cc1 = min(w, left + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                occupied[rr0:rr1, cc0:cc1] = True
                _paint_mask(grid, mask, top, left, int(piece_specs[idx]["color"]))
                pts = [(top + r, left + c) for r, c in np.argwhere(mask)]
                placed = {"top": int(top), "left": int(left), "points": pts}
                placements.append(placed)
                break
            if placed is not None:
                break
        if placed is None:
            return None
    return placements


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    num_regions = rng.choice([2, 3, 4])
    max_dim = 6 if num_regions <= 3 else 5
    colors = rng.sample(PALETTE, num_regions)

    specs = []
    for color in colors:
        shape = _make_shape(rng, max_dim)
        corner = rng.randrange(4)
        target = shape.copy()
        rr, cc = _corner_coord(corner, target.shape)
        target[rr, cc] = False
        spin = rng.randrange(4)
        fragment = np.rot90(target, spin).copy()
        specs.append(
            {
                "color": int(color),
                "shape": shape,
                "corner": int(corner),
                "target": target,
                "mask": fragment,
            }
        )

    try:
        (inner_h, inner_w), positions = _pack_regions(rng, specs)
    except RuntimeError:
        return None

    max_piece_h = max(spec["mask"].shape[0] for spec in specs)
    max_piece_w = max(spec["mask"].shape[1] for spec in specs)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    min_h = frame_h + max_piece_h + 3
    min_w = frame_w + max_piece_w + 3
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    fr0 = rng.randint(max_piece_h + 1, grid_h - frame_h - 2)
    fc0 = rng.randint(max_piece_w + 1, grid_w - frame_w - 2)
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid = np.full((grid_h, grid_w), int(BG), dtype=int)
    grid[fr0, fc0 : fc1 + 1] = int(FRAME_COLOR)
    grid[fr1, fc0 : fc1 + 1] = int(FRAME_COLOR)
    grid[fr0 : fr1 + 1, fc0] = int(FRAME_COLOR)
    grid[fr0 : fr1 + 1, fc1] = int(FRAME_COLOR)

    region_infos = []
    for idx, spec in enumerate(specs):
        local_r, local_c = positions[idx]
        top = fr0 + 1 + local_r
        left = fc0 + 1 + local_c
        _draw_region(grid, top, left, spec["shape"], int(spec["color"]), int(spec["corner"]))
        region_infos.append(
            {
                "color": int(spec["color"]),
                "bbox": (int(top), int(left), int(top + spec["shape"].shape[0] - 1), int(left + spec["shape"].shape[1] - 1)),
                "target": spec["target"],
            }
        )

    placements = _place_fragments(rng, grid, (fr0, fc0, fr1, fc1), specs)
    if placements is None:
        return None

    expected = grid.copy()
    for placed in placements:
        for r, c in placed["points"]:
            expected[r, c] = int(BG)
    for info in region_infos:
        r0, c0, _r1, _c1 = info["bbox"]
        _paint_mask(expected, info["target"], int(r0), int(c0), int(info["color"]))

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = _gen_attempt(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle input")
