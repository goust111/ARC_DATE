from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS = ("U", "R", "D", "L")
STEP = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
ROT_K = {"U": 0, "R": 3, "D": 2, "L": 1}  # np.rot90 k-values (counter-clockwise)
ARROW_NEIGHBORS = {
    "U": {(-1, 0), (0, -1), (0, 1)},
    "R": {(0, 1), (-1, 0), (1, 0)},
    "D": {(1, 0), (0, -1), (0, 1)},
    "L": {(0, -1), (-1, 0), (1, 0)},
}
PATTERN_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))
SIDES = ("L", "R", "U", "D")
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": _bbox(arr),
                }
            )
    return components


def _arrow_from_component(comp: dict) -> dict | None:
    if int(comp["area"]) != 4:
        return None
    cells = {tuple(map(int, rc)) for rc in comp["coords"]}
    degrees = {}
    for cell in cells:
        r, c = cell
        deg = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in cells:
                deg += 1
        degrees[cell] = deg
    centers = [cell for cell, deg in degrees.items() if deg == 3]
    if len(centers) != 1:
        return None
    center = centers[0]
    cr, cc = center
    nbrs = {(r - cr, c - cc) for r, c in cells if (r, c) != center}
    for direction, offsets in ARROW_NEIGHBORS.items():
        if nbrs == offsets:
            return {"color": int(comp["color"]), "center": center, "dir": direction}
    return None


def _crop_component(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    color = int(comp["color"])
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = color
    return patch


def _rotate_patch(patch: np.ndarray, bg: int, direction: str) -> np.ndarray:
    rotated = np.rot90(patch, ROT_K[direction])
    return np.array(rotated, dtype=int, copy=True)


def _stamp_preserving(out: np.ndarray, patch: np.ndarray, center: tuple[int, int], bg: int) -> None:
    pr, pc = patch.shape
    cr, cc = center
    r0 = cr - pr // 2
    c0 = cc - pc // 2
    for rr, cc2 in np.argwhere(patch != bg):
        tr = r0 + int(rr)
        tc = c0 + int(cc2)
        assert 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]
        if int(out[tr, tc]) == bg:
            out[tr, tc] = int(patch[int(rr), int(cc2)])


def _next_arrow(current: tuple[int, int], direction: str, arrows: list[dict]) -> dict | None:
    cr, cc = current
    best = None
    best_dist = None
    for arrow in arrows:
        ar, ac = arrow["center"]
        if (ar, ac) == current:
            continue
        if direction == "U" and ac == cc and ar < cr:
            dist = cr - ar
        elif direction == "D" and ac == cc and ar > cr:
            dist = ar - cr
        elif direction == "L" and ar == cr and ac < cc:
            dist = cc - ac
        elif direction == "R" and ar == cr and ac > cc:
            dist = ac - cc
        else:
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = arrow
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow colored signpost arrows to relay a source pattern, rotating it at each visited arrow.

    Input structure:
    - The dominant color is the background.
    - One monochrome source pattern sits near one side of the grid.
    - Several uniquely colored arrows appear as 4-pixel T-shapes: one center pixel with three orthogonal neighbors; the lone branch direction is the arrow direction.
    - One single-pixel marker matches the color of the first arrow in the relay chain.

    Rule:
    - Identify every T-shaped arrow, decoding its center and its pointing direction from the missing arm of the T.
    - Find the marker's matching arrow color to choose the first visited arrow.
    - Carry the source pattern through the arrow chain: at each visited arrow, rotate the carried pattern by the amount encoded by that arrow (up=0, right=90 clockwise, down=180, left=270), then stamp that rotated pattern centered on the arrow.
    - From the current arrow center, ray-cast in the arrow's pointing direction to the nearest next arrow center on the same row or column; stop when no further arrow is found.
    - Preserve the original source pattern, marker, and all arrow pixels while adding the rotated pattern copies onto background cells.

    Key insight:
    - The T-shape geometry simultaneously tells you which objects are arrows, where each arrow is centered, and which direction the relay ray must follow; the marker color only selects the chain's starting arrow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _find_components(g, bg)
    arrows_by_color = {}
    singletons = []
    sources = []
    for comp in components:
        arrow = _arrow_from_component(comp)
        if arrow is not None:
            color = int(arrow["color"])
            assert color not in arrows_by_color, "each arrow color must be unique"
            arrows_by_color[color] = arrow
        elif int(comp["area"]) == 1:
            singletons.append(comp)
        else:
            sources.append(comp)

    assert arrows_by_color, "expected at least one arrow"
    marker_candidates = [comp for comp in singletons if int(comp["color"]) in arrows_by_color]
    assert len(marker_candidates) == 1, "expected exactly one start marker"
    assert len(sources) == 1, "expected exactly one non-arrow source pattern"

    source = sources[0]
    state = _crop_component(g, source, bg)
    start_color = int(marker_candidates[0]["color"])
    current = arrows_by_color[start_color]
    all_arrows = list(arrows_by_color.values())

    out = g.copy()
    visited = set()
    while current is not None:
        center = tuple(current["center"])
        assert center not in visited, "arrow chain must not cycle"
        visited.add(center)
        state = _rotate_patch(state, bg, current["dir"])
        _stamp_preserving(out, state, center, bg)
        current = _next_arrow(center, current["dir"], all_arrows)

    return out


def _grow_mask(height: int, width: int, target: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (height // 2, width // 2)
    mask[start] = True
    while int(mask.sum()) < target:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _is_arrow_mask(mask: np.ndarray) -> bool:
    if int(mask.sum()) != 4:
        return False
    coords = np.argwhere(mask)
    comp = {"coords": coords, "area": 4, "color": 1, "bbox": _bbox(coords)}
    return _arrow_from_component(comp) is not None


def _make_source_mask() -> np.ndarray | None:
    for _ in range(200):
        height, width = random.choice(PATTERN_SHAPES)
        target = random.randint(3, min(height * width - 1, 8))
        mask = _grow_mask(height, width, target)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if mask.shape[0] % 2 == 0 or mask.shape[1] % 2 == 0:
            continue
        if int(mask.sum()) < 3 or _is_arrow_mask(mask):
            continue
        sigs = {_mask_signature(np.rot90(mask, k)) for k in range(4)}
        if len(sigs) != 4:
            continue
        return mask
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _center_box(center: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    r, c = center
    return r - radius, c - radius, r + radius, c + radius


def _arrow_cells(center: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    cr, cc = center
    cells = {(cr, cc)}
    for dr, dc in ARROW_NEIGHBORS[direction]:
        cells.add((cr + dr, cc + dc))
    return cells


def _place_source(height: int, width: int, patch_h: int, patch_w: int, side: str) -> tuple[int, int, int, int] | None:
    if side == "L":
        c0 = random.randint(1, 2)
        if c0 + patch_w >= width - 2:
            return None
        r0 = random.randint(1, height - patch_h - 2)
    elif side == "R":
        c0 = width - patch_w - random.randint(2, 3)
        if c0 <= 1:
            return None
        r0 = random.randint(1, height - patch_h - 2)
    elif side == "U":
        r0 = random.randint(1, 2)
        if r0 + patch_h >= height - 2:
            return None
        c0 = random.randint(1, width - patch_w - 2)
    else:
        r0 = height - patch_h - random.randint(2, 3)
        if r0 <= 1:
            return None
        c0 = random.randint(1, width - patch_w - 2)
    return r0, c0, r0 + patch_h - 1, c0 + patch_w - 1


def _marker_pos(source_box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = source_box
    if side == "L":
        return (r0 + r1) // 2, c1 + 1
    if side == "R":
        return (r0 + r1) // 2, c0 - 1
    if side == "U":
        return r1 + 1, (c0 + c1) // 2
    return r0 - 1, (c0 + c1) // 2


def _side_first_centers(
    height: int,
    width: int,
    side: str,
    source_box: tuple[int, int, int, int],
    radius: int,
    reserved: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = source_box
    src_cr = (r0 + r1) // 2
    src_cc = (c0 + c1) // 2
    candidates = []
    row_band = range(max(radius, src_cr - 3), min(height - radius, src_cr + 4))
    col_band = range(max(radius, src_cc - 3), min(width - radius, src_cc + 4))

    if side == "L":
        rows = list(row_band) or list(range(radius, height - radius))
        cols = range(max(radius, c1 + 4), width - radius)
        pairs = [(r, c) for r in rows for c in cols]
    elif side == "R":
        rows = list(row_band) or list(range(radius, height - radius))
        cols = range(radius, min(width - radius, c0 - 3))
        pairs = [(r, c) for r in rows for c in cols]
    elif side == "U":
        cols = list(col_band) or list(range(radius, width - radius))
        rows = range(max(radius, r1 + 4), height - radius)
        pairs = [(r, c) for r in rows for c in cols]
    else:
        cols = list(col_band) or list(range(radius, width - radius))
        rows = range(radius, min(height - radius, r0 - 3))
        pairs = [(r, c) for r in rows for c in cols]

    for center in pairs:
        box = _center_box(center, radius)
        if any(_boxes_overlap(box, other) for other in reserved):
            continue
        candidates.append(center)
    random.shuffle(candidates)
    return candidates


def _blocked_by_existing(current: tuple[int, int], direction: str, centers: list[tuple[int, int]]) -> bool:
    cr, cc = current
    for r, c in centers[:-1]:
        if direction == "U" and c == cc and r < cr:
            return True
        if direction == "D" and c == cc and r > cr:
            return True
        if direction == "L" and r == cr and c < cc:
            return True
        if direction == "R" and r == cr and c > cc:
            return True
    return False


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    step = 1 if b[0] > a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _next_candidates(
    current: tuple[int, int],
    direction: str,
    centers: list[tuple[int, int]],
    reserved: list[tuple[int, int, int, int]],
    radius: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    if _blocked_by_existing(current, direction, centers):
        return []

    used_rows = {r for r, _ in centers}
    used_cols = {c for _, c in centers}
    cr, cc = current
    min_sep = 2 * radius + 3
    candidates = []

    if direction == "U":
        positions = [(r, cc) for r in range(cr - min_sep, radius - 1, -1)]
    elif direction == "D":
        positions = [(r, cc) for r in range(cr + min_sep, height - radius)]
    elif direction == "L":
        positions = [(cr, c) for c in range(cc - min_sep, radius - 1, -1)]
    else:
        positions = [(cr, c) for c in range(cc + min_sep, width - radius)]

    for cand in positions:
        r, c = cand
        if cand in centers:
            continue
        if direction in ("U", "D") and r in used_rows:
            continue
        if direction in ("L", "R") and c in used_cols:
            continue
        box = _center_box(cand, radius)
        if any(_boxes_overlap(box, other) for other in reserved):
            continue
        candidates.append(cand)
    random.shuffle(candidates)
    return candidates


def _clear_end_dirs(center: tuple[int, int], centers: list[tuple[int, int]]) -> list[str]:
    clear = []
    cr, cc = center
    for direction in DIRS:
        blocked = False
        for r, c in centers:
            if (r, c) == center:
                continue
            if direction == "U" and c == cc and r < cr:
                blocked = True
            if direction == "D" and c == cc and r > cr:
                blocked = True
            if direction == "L" and r == cr and c < cc:
                blocked = True
            if direction == "R" and r == cr and c > cc:
                blocked = True
        if not blocked:
            clear.append(direction)
    random.shuffle(clear)
    return clear


def _build_chain(
    height: int,
    width: int,
    side: str,
    source_box: tuple[int, int, int, int],
    radius: int,
) -> tuple[list[tuple[int, int]], list[str], list[tuple[int, int, int, int]]] | None:
    base_reserved = [
        (
            max(0, source_box[0] - 1),
            max(0, source_box[1] - 1),
            min(height - 1, source_box[2] + 1),
            min(width - 1, source_box[3] + 1),
        )
    ]
    first_centers = _side_first_centers(height, width, side, source_box, radius, base_reserved)
    for first in first_centers[:80]:
        centers = [first]
        reserved = base_reserved + [_center_box(first, radius)]
        dirs = []
        length = random.randint(2, 4)
        ok = True
        for _ in range(length - 1):
            options = []
            for direction in random.sample(DIRS, k=len(DIRS)):
                cands = _next_candidates(centers[-1], direction, centers, reserved, radius, height, width)
                if cands:
                    options.append((direction, cands))
            if not options:
                ok = False
                break
            direction, cands = random.choice(options)
            nxt = random.choice(cands[: min(len(cands), 10)])
            dirs.append(direction)
            centers.append(nxt)
            reserved.append(_center_box(nxt, radius))
        if not ok:
            continue
        end_dirs = _clear_end_dirs(centers[-1], centers)
        if not end_dirs:
            continue
        dirs.append(random.choice(end_dirs))
        if all(direction == "U" for direction in dirs):
            continue
        return centers, dirs, reserved
    return None


def _place_distractors(
    count: int,
    height: int,
    width: int,
    radius: int,
    centers: list[tuple[int, int]],
    reserved: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int]], list[str]] | None:
    distractor_centers = []
    distractor_dirs = []
    used_rows = {r for r, _ in centers}
    used_cols = {c for _, c in centers}
    local_reserved = list(reserved)

    for _ in range(count):
        candidates = []
        for r in range(radius, height - radius):
            if r in used_rows:
                continue
            for c in range(radius, width - radius):
                if c in used_cols:
                    continue
                box = _center_box((r, c), radius)
                if any(_boxes_overlap(box, other) for other in local_reserved):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        center = random.choice(candidates)
        distractor_centers.append(center)
        distractor_dirs.append(random.choice(DIRS))
        used_rows.add(center[0])
        used_cols.add(center[1])
        local_reserved.append(_center_box(center, radius))
    return distractor_centers, distractor_dirs


def _paint_source(grid: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, _, _ = box
    for r, c in np.argwhere(mask):
        grid[r0 + int(r), c0 + int(c)] = color


def _paint_arrow(grid: np.ndarray, center: tuple[int, int], direction: str, color: int) -> None:
    for r, c in _arrow_cells(center, direction):
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    bg = random.randint(0, 9)

    source_mask = _make_source_mask()
    if source_mask is None:
        return None
    patch_h, patch_w = map(int, source_mask.shape)
    radius = max(patch_h // 2, patch_w // 2, 1)

    side = random.choice(SIDES)
    source_box = _place_source(height, width, patch_h, patch_w, side)
    if source_box is None:
        return None

    chain = _build_chain(height, width, side, source_box, radius)
    if chain is None:
        return None
    chain_centers, chain_dirs, reserved = chain

    distractor_count = random.randint(0, 2)
    distractors = _place_distractors(distractor_count, height, width, radius, chain_centers, reserved)
    if distractors is None:
        return None
    distractor_centers, distractor_dirs = distractors

    needed_colors = 1 + len(chain_centers) + len(distractor_centers)
    palette = [color for color in range(10) if color != bg]
    if needed_colors > len(palette):
        return None
    colors = random.sample(palette, needed_colors)
    source_color = colors[0]
    arrow_colors = colors[1:]

    grid = np.full((height, width), bg, dtype=int)
    _paint_source(grid, source_mask, source_box, source_color)

    marker = _marker_pos(source_box, side)
    if int(grid[marker]) != bg:
        return None
    grid[marker] = arrow_colors[0]

    for center, direction, color in zip(chain_centers, chain_dirs, arrow_colors[: len(chain_centers)]):
        cells = _arrow_cells(center, direction)
        if any(not (0 <= r < height and 0 <= c < width) or int(grid[r, c]) != bg for r, c in cells):
            return None
        _paint_arrow(grid, center, direction, color)

    offset = len(chain_centers)
    for center, direction, color in zip(distractor_centers, distractor_dirs, arrow_colors[offset:]):
        cells = _arrow_cells(center, direction)
        if any(not (0 <= r < height and 0 <= c < width) or int(grid[r, c]) != bg for r, c in cells):
            return None
        _paint_arrow(grid, center, direction, color)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
