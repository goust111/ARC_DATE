import random
from collections import Counter

import numpy as np


BG = 0
FRAME_DIRECTIONS = ("top", "bottom", "left", "right")
COLOR_POOL = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
_RNG = random.Random()


def _components4(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_shape(height: int, width: int, gap: str):
    coords = set()
    for c in range(width):
        coords.add((0, c))
        coords.add((height - 1, c))
    for r in range(1, height - 1):
        coords.add((r, 0))
        coords.add((r, width - 1))
    if gap == "top":
        for c in range(1, width - 1):
            coords.discard((0, c))
    elif gap == "bottom":
        for c in range(1, width - 1):
            coords.discard((height - 1, c))
    elif gap == "left":
        for r in range(1, height - 1):
            coords.discard((r, 0))
    elif gap == "right":
        for r in range(1, height - 1):
            coords.discard((r, width - 1))
    else:
        raise ValueError(gap)
    return coords


def _parse_frame_component(points):
    r0, c0, r1, c1 = _bbox(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    rel = {(r - r0, c - c0) for r, c in points}
    for gap in FRAME_DIRECTIONS:
        if rel == _frame_shape(height, width, gap):
            return {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "height": height,
                "width": width,
            }
    return None


def _find_active_frame(grid: np.ndarray):
    frames = []
    singletons = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for comp in _components4(grid == color):
            if len(comp) == 1:
                singletons.append((color, comp[0]))
            parsed = _parse_frame_component(comp)
            if parsed is not None:
                parsed["color"] = color
                frames.append(parsed)

    selector_matches = []
    for color, pt in singletons:
        matches = [frame for frame in frames if frame["color"] == color]
        if len(matches) != 1:
            continue
        frame = matches[0]
        r, c = pt
        r0, c0, r1, c1 = frame["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        selector_matches.append(frame)

    assert len(selector_matches) == 1, "expected exactly one selector-matched frame"
    return selector_matches[0]


def _extract_sprite_patch(grid: np.ndarray, frame):
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1:r1, c0 + 1:c1]
    mask = interior != BG
    assert mask.any(), "active frame must contain a sprite"
    rows, cols = np.where(mask)
    sr0, sc0 = int(rows.min()), int(cols.min())
    sr1, sc1 = int(rows.max()), int(cols.max())
    patch = interior[sr0:sr1 + 1, sc0:sc1 + 1].copy()
    return patch, r0 + 1 + sr0, c0 + 1 + sc0


def _paste_transparent(grid: np.ndarray, patch: np.ndarray, top: int, left: int):
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= grid.shape[0] and left + w <= grid.shape[1]
    region = grid[top:top + h, left:left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the frame keyed by the matching selector pixel and project its interior sprite through the frame gap.

    Input structure:
    - The background is black.
    - Several non-overlapping colored hollow rectangular frames appear, each missing exactly one full inner side segment.
    - Every frame contains a non-black sprite whose colors differ from that frame border color.
    - A lone selector pixel has the same color as exactly one frame border.

    Rule:
    - Find the unique frame whose border color matches the selector pixel.
    - Detect which side of that frame has the continuous opening: top, bottom, left, or right.
    - Copy the enclosed sprite unchanged and translate it straight out through that opening so the copied sprite sits just outside the frame on the gap side.
    - Keep the original frames, sprites, and all inactive structures unchanged.

    Key insight:
    - The selector color identifies the active frame, while the missing border side supplies the projection direction; both cues are required.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frame = _find_active_frame(g)
    patch, sprite_top, sprite_left = _extract_sprite_patch(g, frame)
    r0, c0, r1, c1 = frame["bbox"]
    ph, pw = patch.shape

    if frame["gap"] == "top":
        dest_r, dest_c = r0 - ph, sprite_left
    elif frame["gap"] == "bottom":
        dest_r, dest_c = r1 + 1, sprite_left
    elif frame["gap"] == "left":
        dest_r, dest_c = sprite_top, c0 - pw
    elif frame["gap"] == "right":
        dest_r, dest_c = sprite_top, c1 + 1
    else:
        raise ValueError(frame["gap"])

    _paste_transparent(out, patch, dest_r, dest_c)
    return out


def _random_connected_mask(height: int, width: int, rng: random.Random):
    total = height * width
    target = 1 if total == 1 else rng.randint(2, total)
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    while len(cells) < target:
        expandable = []
        for r, c in cells:
            opts = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    opts.append((nr, nc))
            if opts:
                expandable.append(opts)
        if not expandable:
            break
        cells.add(rng.choice(rng.choice(expandable)))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sprite(max_h: int, max_w: int, palette, rng: random.Random):
    base_h = rng.randint(1, max_h)
    base_w = rng.randint(1, max_w)
    if max_h * max_w > 1 and base_h == 1 and base_w == 1:
        if max_h > 1:
            base_h = rng.randint(1, max_h)
        if base_w > 1:
            base_w = rng.randint(1, max_w)

    mask = _random_connected_mask(base_h, base_w, rng)
    rows, cols = np.where(mask)
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())
    mask = mask[r0:r1 + 1, c0:c1 + 1]

    colors = rng.sample(list(palette), rng.randint(1, min(3, len(palette))))
    patch = np.full(mask.shape, BG, dtype=int)
    occupied = list(zip(*np.where(mask)))
    rng.shuffle(occupied)
    for idx, (r, c) in enumerate(occupied):
        if len(colors) == 1:
            patch[r, c] = colors[0]
        else:
            patch[r, c] = colors[idx % len(colors)] if rng.random() < 0.7 else rng.choice(colors)
    return patch


def _projection_bbox(spec, top: int, left: int):
    sprite = spec["sprite"]
    ph, pw = sprite.shape
    sr0 = top + 1 + spec["sprite_top"]
    sc0 = left + 1 + spec["sprite_left"]
    sr1 = sr0 + ph - 1
    sc1 = sc0 + pw - 1
    r1 = top + spec["fh"] - 1
    c1 = left + spec["fw"] - 1

    if spec["gap"] == "top":
        return top - ph, sc0, top - 1, sc1
    if spec["gap"] == "bottom":
        return r1 + 1, sc0, r1 + ph, sc1
    if spec["gap"] == "left":
        return sr0, left - pw, sr1, left - 1
    if spec["gap"] == "right":
        return sr0, c1 + 1, sr1, c1 + pw
    raise ValueError(spec["gap"])


def _regions_for(spec, top: int, left: int):
    regions = [(top, left, top + spec["fh"] - 1, left + spec["fw"] - 1)]
    if spec["active"]:
        regions.append(_projection_bbox(spec, top, left))
    return regions


def _rects_clear(blocked: np.ndarray, rects):
    h, w = blocked.shape
    for r0, c0, r1, c1 in rects:
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            return False
        pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
        pr1, pc1 = min(h - 1, r1 + 1), min(w - 1, c1 + 1)
        if blocked[pr0:pr1 + 1, pc0:pc1 + 1].any():
            return False
    return True


def _mark_rects(blocked: np.ndarray, rects):
    h, w = blocked.shape
    for r0, c0, r1, c1 in rects:
        pr0, pc0 = max(0, r0 - 1), max(0, c0 - 1)
        pr1, pc1 = min(h - 1, r1 + 1), min(w - 1, c1 + 1)
        blocked[pr0:pr1 + 1, pc0:pc1 + 1] = True


def _draw_frame(grid: np.ndarray, spec, top: int, left: int):
    color = spec["color"]
    for rr, cc in _frame_shape(spec["fh"], spec["fw"], spec["gap"]):
        grid[top + rr, left + cc] = color
    _paste_transparent(
        grid,
        spec["sprite"],
        top + 1 + spec["sprite_top"],
        left + 1 + spec["sprite_left"],
    )


def _gen_attempt(rng: random.Random):
    frame_count = rng.randint(2, 4)
    min_side = {2: 18, 3: 21, 4: 24}[frame_count]
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)
    interior_max = {2: 7, 3: 6, 4: 5}[frame_count]

    frame_colors = rng.sample(list(COLOR_POOL), frame_count)
    sprite_palette = [c for c in COLOR_POOL if c not in frame_colors]
    active_index = rng.randrange(frame_count)
    active_gap = FRAME_DIRECTIONS[rng.randrange(4)]

    specs = []
    for idx in range(frame_count):
        ih = rng.randint(3, interior_max)
        iw = rng.randint(3, interior_max)
        sprite = _make_sprite(ih, iw, sprite_palette, rng)
        ph, pw = sprite.shape
        sprite_top = rng.randint(0, ih - ph)
        sprite_left = rng.randint(0, iw - pw)
        specs.append(
            {
                "idx": idx,
                "active": idx == active_index,
                "color": frame_colors[idx],
                "gap": active_gap if idx == active_index else rng.choice(FRAME_DIRECTIONS),
                "ih": ih,
                "iw": iw,
                "fh": ih + 2,
                "fw": iw + 2,
                "sprite": sprite,
                "sprite_top": sprite_top,
                "sprite_left": sprite_left,
            }
        )

    active_spec = specs[active_index]
    distractors = [spec for spec in specs if not spec["active"]]
    distractors.sort(key=lambda spec: spec["fh"] * spec["fw"], reverse=True)
    placement_order = [active_spec] + distractors

    blocked = np.zeros((height, width), dtype=bool)
    placements = {}
    for spec in placement_order:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(0, height - spec["fh"])
            left = rng.randint(0, width - spec["fw"])
            rects = _regions_for(spec, top, left)
            if not _rects_clear(blocked, rects):
                continue
            placements[spec["idx"]] = (top, left)
            _mark_rects(blocked, rects)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        _draw_frame(grid, spec, *placements[spec["idx"]])

    selector_candidates = np.argwhere((grid == BG) & (~blocked))
    if selector_candidates.size == 0:
        return None
    sr, sc = selector_candidates[rng.randrange(len(selector_candidates))]
    grid[int(sr), int(sc)] = active_spec["color"]

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    diff = out != grid
    if not diff.any():
        return None
    if not np.all(grid[diff] == BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
