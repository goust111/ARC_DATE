import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


SHAPE_LIBRARY = [
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return mask[:0, :0].copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_and_crop(mask: np.ndarray, k: int) -> np.ndarray:
    return _crop_mask(np.rot90(mask, k))


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    keys = set()
    for k in range(4):
        rot = _rotate_and_crop(mask, k)
        keys.add(tuple(tuple(int(x) for x in row) for row in rot))
    return keys


for _shape in SHAPE_LIBRARY:
    assert len(_rotation_keys(_shape)) == 4


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    want = set()
    for c in range(c0, c1 + 1):
        want.add((r0, c))
        want.add((r1, c))
    for r in range(r0, r1 + 1):
        want.add((r, c0))
        want.add((r, c1))
    return set(coords) == want


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for coords in _components(g == color):
            if not _is_rect_frame(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "coords": coords,
                }
            )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered monochrome fragments back into frame holes by color and rotation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    frame_mask = np.zeros((h, w), dtype=bool)
    interior_mask = np.zeros((h, w), dtype=bool)
    for frame in frames:
        for r, c in frame["coords"]:
            frame_mask[r, c] = True
        r0, c0, r1, c1 = frame["bbox"]
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    holes: dict[int, dict] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside = g[r0 + 1 : r1, c0 + 1 : c1] == BG
        for local_coords in _components(inside):
            abs_coords = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in local_coords]
            adj_colors = set()
            for rr, cc in abs_coords:
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if frame_mask[nr, nc]:
                        continue
                    val = int(g[nr, nc])
                    if val != BG:
                        adj_colors.add(val)
            assert len(adj_colors) == 1, "each hole must have a unique owner color"
            color = next(iter(adj_colors))
            assert color not in holes, "owner colors must be globally unique"
            hr0, hc0, hr1, hc1 = _bbox(abs_coords)
            hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
            for rr, cc in abs_coords:
                hole_mask[rr - hr0, cc - hc0] = True
            holes[color] = {
                "bbox": (hr0, hc0, hr1, hc1),
                "mask": hole_mask,
                "coords": abs_coords,
            }

    fragments: dict[int, dict] = {}
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for coords in _components(g == color):
            if all(frame_mask[r, c] for r, c in coords):
                continue
            if all(interior_mask[r, c] for r, c in coords):
                continue
            assert not any(frame_mask[r, c] or interior_mask[r, c] for r, c in coords), (
                "fragments must lie completely outside the frames"
            )
            assert color not in fragments, "each fragment color must be unique"
            fr0, fc0, fr1, fc1 = _bbox(coords)
            frag_mask = np.zeros((fr1 - fr0 + 1, fc1 - fc0 + 1), dtype=bool)
            for rr, cc in coords:
                frag_mask[rr - fr0, cc - fc0] = True
            fragments[color] = {
                "bbox": (fr0, fc0, fr1, fc1),
                "mask": frag_mask,
                "coords": coords,
            }

    out = g.copy()
    for color, hole in holes.items():
        assert color in fragments, "missing fragment for hole color"
        fragment = fragments[color]
        matches = []
        for k in range(4):
            if np.array_equal(_rotate_and_crop(fragment["mask"], k), hole["mask"]):
                matches.append(k)
        assert matches, "fragment must match its hole after rotation"
        for rr, cc in fragment["coords"]:
            out[rr, cc] = BG
        hr0, hc0, hr1, hc1 = hole["bbox"]
        out[hr0 : hr1 + 1, hc0 : hc1 + 1][hole["mask"]] = color
    return out


def _make_frame(owner_colors: list[int], filler_color: int, rng: random.Random) -> dict | None:
    k = len(owner_colors)
    hole_specs = []
    for color in owner_colors:
        base = rng.choice(SHAPE_LIBRARY)
        hole_rot = rng.randrange(4)
        frag_rot = rng.randrange(4)
        hole_mask = _rotate_and_crop(base, hole_rot)
        frag_mask = _rotate_and_crop(base, frag_rot)
        hole_specs.append({"color": color, "hole_mask": hole_mask, "frag_mask": frag_mask})

    orientation = rng.choice(["vertical", "horizontal"])
    if k == 1:
        regions = [{"kind": "owner", "spec": hole_specs[0]}, {"kind": "filler", "color": filler_color}]
        rng.shuffle(regions)
    else:
        regions = [{"kind": "owner", "spec": spec} for spec in hole_specs]
        rng.shuffle(regions)

    if orientation == "vertical":
        min_h = max(max(spec["hole_mask"].shape[0] + 2, 5) for spec in hole_specs)
        ih = min_h + rng.randint(0, 3)
        widths = []
        for region in regions:
            if region["kind"] == "owner":
                widths.append(region["spec"]["hole_mask"].shape[1] + 2 + rng.randint(0, 2))
            else:
                widths.append(rng.randint(2, 4))
        iw = sum(widths)
        if ih > 10 or iw > 12:
            return None
        interior = np.zeros((ih, iw), dtype=int)
        segs = []
        cur = 0
        for width, region in zip(widths, regions):
            interior[:, cur : cur + width] = region["spec"]["color"] if region["kind"] == "owner" else region["color"]
            segs.append((0, cur, ih, cur + width, region))
            cur += width
    else:
        min_w = max(max(spec["hole_mask"].shape[1] + 2, 5) for spec in hole_specs)
        iw = min_w + rng.randint(0, 3)
        heights = []
        for region in regions:
            if region["kind"] == "owner":
                heights.append(region["spec"]["hole_mask"].shape[0] + 2 + rng.randint(0, 2))
            else:
                heights.append(rng.randint(2, 4))
        ih = sum(heights)
        if ih > 12 or iw > 10:
            return None
        interior = np.zeros((ih, iw), dtype=int)
        segs = []
        cur = 0
        for height, region in zip(heights, regions):
            interior[cur : cur + height, :] = region["spec"]["color"] if region["kind"] == "owner" else region["color"]
            segs.append((cur, 0, cur + height, iw, region))
            cur += height

    local_holes = []
    for sr0, sc0, sr1, sc1, region in segs:
        if region["kind"] != "owner":
            continue
        hole_mask = region["spec"]["hole_mask"]
        hh, hw = hole_mask.shape
        top_lo = sr0 + 1
        top_hi = sr1 - hh - 1
        left_lo = sc0 + 1
        left_hi = sc1 - hw - 1
        if top_lo > top_hi or left_lo > left_hi:
            return None
        top = rng.randint(top_lo, top_hi)
        left = rng.randint(left_lo, left_hi)
        interior[top : top + hh, left : left + hw][hole_mask] = BG
        coords = [(top + rr, left + cc) for rr in range(hh) for cc in range(hw) if hole_mask[rr, cc]]
        local_holes.append(
            {
                "color": region["spec"]["color"],
                "coords": coords,
                "frag_mask": region["spec"]["frag_mask"],
            }
        )

    fh, fw = ih + 2, iw + 2
    frame = np.full((fh, fw), BG, dtype=int)
    frame[0, :] = FRAME
    frame[-1, :] = FRAME
    frame[:, 0] = FRAME
    frame[:, -1] = FRAME
    frame[1:-1, 1:-1] = interior
    return {"array": frame, "holes": local_holes}


def _rect_is_free(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = blocked.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not blocked[r0:r1, c0:c1].any()


def _reserve_rect(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _paint(canvas: np.ndarray, arr: np.ndarray, top: int, left: int) -> None:
    h, w = arr.shape
    mask = arr != BG
    canvas[top : top + h, left : left + w][mask] = arr[mask]


def _sample_layout(rng: random.Random) -> list[int]:
    return rng.choice([[1], [2], [1, 1], [1, 2], [2, 1], [2, 2]])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    holes_per_frame = _sample_layout(rng)
    total_holes = sum(holes_per_frame)
    owner_colors = rng.sample(COLOR_POOL, total_holes)
    filler_pool = [c for c in COLOR_POOL if c not in owner_colors]
    filler_needed = sum(1 for x in holes_per_frame if x == 1)
    filler_colors = rng.sample(filler_pool, filler_needed)

    color_idx = 0
    filler_idx = 0
    hole_records = []
    fragment_records = []

    for hole_count in holes_per_frame:
        owners = owner_colors[color_idx : color_idx + hole_count]
        color_idx += hole_count
        filler = filler_colors[filler_idx] if hole_count == 1 else filler_pool[0]
        if hole_count == 1:
            filler_idx += 1
        frame_spec = _make_frame(owners, filler, rng)
        if frame_spec is None:
            return None
        arr = frame_spec["array"]
        fh, fw = arr.shape
        placed = False
        for _ in range(200):
            top = rng.randint(1, h - fh - 1)
            left = rng.randint(1, w - fw - 1)
            if not _rect_is_free(blocked, top, left, fh, fw, pad=1):
                continue
            _paint(grid, arr, top, left)
            _paint(expected, arr, top, left)
            _reserve_rect(blocked, top, left, fh, fw, pad=1)
            for hole in frame_spec["holes"]:
                abs_coords = [(top + 1 + rr, left + 1 + cc) for rr, cc in hole["coords"]]
                hole_records.append(
                    {
                        "color": hole["color"],
                        "coords": abs_coords,
                        "frag_mask": hole["frag_mask"],
                    }
                )
                for rr, cc in abs_coords:
                    expected[rr, cc] = hole["color"]
            placed = True
            break
        if not placed:
            return None

    rng.shuffle(hole_records)
    for hole in hole_records:
        mask = hole["frag_mask"]
        mh, mw = mask.shape
        arr = np.zeros((mh, mw), dtype=int)
        arr[mask] = hole["color"]
        placed = False
        for _ in range(200):
            top = rng.randint(0, h - mh)
            left = rng.randint(0, w - mw)
            if not _rect_is_free(blocked, top, left, mh, mw, pad=1):
                continue
            _paint(grid, arr, top, left)
            _reserve_rect(blocked, top, left, mh, mw, pad=1)
            coords = [(top + rr, left + cc) for rr in range(mh) for cc in range(mw) if mask[rr, cc]]
            fragment_records.append({"color": hole["color"], "coords": coords})
            placed = True
            break
        if not placed:
            return None

    for fragment in fragment_records:
        for rr, cc in fragment["coords"]:
            expected[rr, cc] = BG

    if len(hole_records) != len(fragment_records):
        return None
    if not np.any(grid == FRAME):
        return None
    if sum(len(_rotation_keys(record["frag_mask"])) == 4 for record in hole_records) != len(hole_records):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
