import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
LIBRARY_BAND = 6
GEN_ATTEMPTS = 400
PATTERN_PLACEMENT_TRIES = 120
FRAME_PLACEMENT_TRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = mask.shape
    rows = (np.arange(target_h) * src_h) // target_h
    cols = (np.arange(target_w) * src_w) // target_w
    return mask[rows[:, None], cols[None, :]]


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _components(grid == GREY):
        box = _frame_bbox(comp)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        pts = np.argwhere((inner != BG) & (inner != GREY))
        assert len(pts) == 1, "each frame must contain exactly one colored key pixel"
        dr, dc = map(int, pts[0])
        color = int(inner[dr, dc])
        frames.append(
            {
                "bbox": box,
                "color": color,
                "inner_h": int(r1 - r0 - 1),
                "inner_w": int(c1 - c0 - 1),
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    assert frames, "expected at least one grey frame"
    return frames


def _component_inside_frame(comp: list[tuple[int, int]], frame: dict) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    return all(r0 < r < r1 and c0 < c < c1 for r, c in comp)


def _find_library_patterns(grid: np.ndarray, frames: list[dict]) -> tuple[dict[int, np.ndarray], list[tuple[int, int]]]:
    patterns: dict[int, np.ndarray] = {}
    external_cells: list[tuple[int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, GREY}):
        for comp in _components(grid == color):
            if any(_component_inside_frame(comp, frame) for frame in frames):
                continue
            assert color not in patterns, "each color must have exactly one external pattern"
            mask, _ = _coords_to_mask(comp)
            patterns[color] = mask
            external_cells.extend(comp)
    for frame in frames:
        assert frame["color"] in patterns, "every frame color must have a matching library pattern"
    return patterns, external_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each frame's interior color as a key into the edge template library.

    The matched single-color template is cropped to its own bounding box,
    resized with nearest-neighbor sampling to the frame's interior size,
    written into the frame interior, and then the source template is erased.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    patterns, external_cells = _find_library_patterns(g, frames)

    for r, c in external_cells:
        out[r, c] = BG

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner.fill(BG)
        scaled = _resize_mask(patterns[frame["color"]], frame["inner_h"], frame["inner_w"])
        inner[scaled] = frame["color"]

    return out


def _random_polyomino(rng: random.Random, h: int, w: int) -> np.ndarray:
    area_min = max(2, (h * w + 2) // 3)
    area_max = max(area_min, h * w - 1)
    target = rng.randint(area_min, area_max)
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = set()
    for dr, dc in DIR4:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while len(cells) < target and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        cr, cc = cell
        for dr, dc in DIR4:
            nr = cr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _normalize_mask(mask)


def _l_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    thick = 1 if min(h, w) <= 3 else rng.randint(1, min(2, min(h, w) - 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :thick] = True
    mask[h - thick :, :] = True
    return mask


def _plus_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    if h >= 5:
        mask[h // 2 - 1 : h // 2 + 2, w // 2] = True
    if w >= 5:
        mask[h // 2, w // 2 - 1 : w // 2 + 2] = True
    return mask


def _ring_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if h > 2 and w > 2:
        mask[1:-1, 1:-1] = False
    return mask


def _stair_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r = 0
    c = 0
    mask[r, c] = True
    while r < h - 1 or c < w - 1:
        if r == h - 1:
            c += 1
        elif c == w - 1:
            r += 1
        elif rng.random() < 0.5:
            r += 1
        else:
            c += 1
        mask[r, c] = True
        if r + 1 < h and rng.random() < 0.25:
            mask[r + 1, c] = True
        if c + 1 < w and rng.random() < 0.25:
            mask[r, c + 1] = True
    return _normalize_mask(mask)


def _random_pattern_mask(rng: random.Random) -> np.ndarray:
    for _ in range(40):
        kind = rng.choice(["poly", "poly", "l", "plus", "ring", "stair"])
        if kind == "poly":
            h = rng.randint(2, 5)
            w = rng.randint(2, 5)
            mask = _random_polyomino(rng, h, w)
        elif kind == "l":
            h = rng.randint(2, 5)
            w = rng.randint(2, 5)
            mask = _l_mask(rng, h, w)
        elif kind == "plus":
            h = rng.choice([3, 5])
            w = rng.choice([3, 5])
            mask = _plus_mask(h, w)
        elif kind == "ring":
            h = rng.randint(3, 5)
            w = rng.randint(3, 5)
            mask = _ring_mask(h, w)
        else:
            h = rng.randint(2, 5)
            w = rng.randint(2, 5)
            mask = _stair_mask(rng, h, w)

        k = rng.randrange(4)
        mask = np.rot90(mask, k)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = _normalize_mask(mask)
        if mask.sum() < 2:
            continue
        if mask.all():
            continue
        if len(_components(mask)) != 1:
            continue
        if max(mask.shape) > LIBRARY_BAND:
            continue
        return mask
    raise RuntimeError("failed to build a connected pattern mask")


def _rect_is_free(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _mark_rect(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[r0 + r, c0 + c] = color
                coords.append((r0 + r, c0 + c))
    return coords


def _paint_frame(grid: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, color: int, rng: random.Random) -> dict:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    dot_r = rng.randint(r0 + 1, r1 - 1)
    dot_c = rng.randint(c0 + 1, c1 - 1)
    grid[dot_r, dot_c] = color
    return {"bbox": (r0, c0, r1, c1), "color": color}


def _expected_output(inp: np.ndarray, library: list[dict], frames: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in library:
        for r, c in item["cells"]:
            out[r, c] = BG
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask = _resize_mask(frame["pattern"], r1 - r0 - 1, c1 - c0 - 1)
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner.fill(BG)
        inner[mask] = frame["color"]
    return out


def _choose_scales(rng: random.Random, count: int, masks: list[np.ndarray]) -> list[int]:
    scales = []
    for mask in masks:
        if count >= 4:
            options = [1, 2]
        elif count == 3:
            options = [1, 2, 3]
        else:
            options = [1, 2, 3, 4]
        if max(mask.shape) >= 5:
            options = [s for s in options if s <= 2] or [1]
        scales.append(rng.choice(options))
    return scales


def _grid_candidates(frame_sizes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    max_h = max(h for h, _ in frame_sizes)
    max_w = max(w for _, w in frame_sizes)
    need_area = sum((h + 1) * (w + 1) for h, w in frame_sizes)
    choices = []
    for h in range(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1):
        for w in range(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1):
            center_h = h - 2 * LIBRARY_BAND
            center_w = w - 2 * LIBRARY_BAND
            if center_h < max_h + 1 or center_w < max_w + 1:
                continue
            if center_h * center_w < need_area:
                continue
            choices.append((h, w))
    return choices


def _place_library(grid: np.ndarray, occ: np.ndarray, items: list[dict], rng: random.Random) -> bool:
    h, w = grid.shape
    order = sorted(range(len(items)), key=lambda idx: int(items[idx]["mask"].sum()), reverse=True)
    for idx in order:
        item = items[idx]
        mask = item["mask"]
        ph, pw = mask.shape
        side_order = ["top", "bottom", "left", "right"]
        rng.shuffle(side_order)
        candidates: list[tuple[int, int]] = []
        for side in side_order:
            if side == "top":
                for c0 in range(0, w - pw + 1):
                    if _rect_is_free(occ, 0, c0, ph, pw):
                        candidates.append((0, c0))
            elif side == "bottom":
                r0 = h - ph
                for c0 in range(0, w - pw + 1):
                    if _rect_is_free(occ, r0, c0, ph, pw):
                        candidates.append((r0, c0))
            elif side == "left":
                for r0 in range(0, h - ph + 1):
                    if _rect_is_free(occ, r0, 0, ph, pw):
                        candidates.append((r0, 0))
            else:
                c0 = w - pw
                for r0 in range(0, h - ph + 1):
                    if _rect_is_free(occ, r0, c0, ph, pw):
                        candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        item["cells"] = _paint_mask(grid, mask, r0, c0, item["color"])
        _mark_rect(occ, r0, c0, ph, pw)
    return True


def _place_frames(grid: np.ndarray, occ: np.ndarray, frames: list[dict], rng: random.Random) -> bool:
    h, w = grid.shape
    order = sorted(range(len(frames)), key=lambda idx: frames[idx]["outer_h"] * frames[idx]["outer_w"], reverse=True)
    for idx in order:
        frame = frames[idx]
        fh = frame["outer_h"]
        fw = frame["outer_w"]
        placements = []
        for _ in range(FRAME_PLACEMENT_TRIES):
            r0 = rng.randint(LIBRARY_BAND, h - LIBRARY_BAND - fh)
            c0 = rng.randint(LIBRARY_BAND, w - LIBRARY_BAND - fw)
            if _rect_is_free(occ, r0, c0, fh, fw):
                placements.append((r0, c0))
                if len(placements) >= 20:
                    break
        if not placements:
            return False
        r0, c0 = rng.choice(placements)
        painted = _paint_frame(grid, r0, c0, fh, fw, frame["color"], rng)
        frame["bbox"] = painted["bbox"]
        _mark_rect(occ, r0, c0, fh, fw)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 5)
    colors = rng.sample(COLOR_POOL, count)
    library = [{"color": color, "mask": _random_pattern_mask(rng)} for color in colors]
    scales = _choose_scales(rng, count, [item["mask"] for item in library])

    frame_specs = []
    for item, scale in zip(library, scales):
        inner_h = int(item["mask"].shape[0] * scale)
        inner_w = int(item["mask"].shape[1] * scale)
        frame_specs.append(
            {
                "color": item["color"],
                "pattern": item["mask"],
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )

    candidates = _grid_candidates([(frame["outer_h"], frame["outer_w"]) for frame in frame_specs])
    if not candidates:
        return None
    grid_h, grid_w = rng.choice(candidates)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    if not _place_library(inp, occ, library, rng):
        return None
    if not _place_frames(inp, occ, frame_specs, rng):
        return None

    out = _expected_output(inp, library, frame_specs)
    if np.array_equal(inp, out):
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
