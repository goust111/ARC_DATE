from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 8
FRAME = 5
SEED = 2
TEMPLATE_COLORS = (1, 3, 4, 6, 7, 9)
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_patch(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == FRAME):
            continue
        if np.any(patch[1:-1, 1:-1] == FRAME):
            continue
        if int(coords.shape[0]) != int(border.sum()):
            continue
        if not np.any(patch[1:-1, 1:-1] == SEED):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one hollow frame"
    return candidates[0]


def _template_components(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    mask = (grid != int(bg)) & (grid != FRAME) & (grid != SEED)
    comps = _components(mask)
    info: list[dict[str, object]] = []
    cell_to_id: dict[tuple[int, int], int] = {}
    for idx, coords in enumerate(comps):
        b = _bbox(coords)
        patch = _crop_patch(grid, b)
        rec = {"id": idx, "coords": coords, "bbox": b, "patch": patch}
        info.append(rec)
        for r, c in coords:
            cell_to_id[(int(r), int(c))] = idx
    return info, cell_to_id


def _seed_hits(
    grid: np.ndarray,
    seed: tuple[int, int],
    cell_to_id: dict[tuple[int, int], int],
    bg: int,
) -> set[int]:
    h, w = grid.shape
    hits: set[int] = set()
    for dr, dc in DIRS.values():
        r = int(seed[0] + dr)
        c = int(seed[1] + dc)
        while 0 <= r < h and 0 <= c < w:
            value = int(grid[r, c])
            if value not in (int(bg), FRAME, SEED):
                comp_id = cell_to_id.get((r, c))
                if comp_id is not None:
                    hits.add(int(comp_id))
                break
            r += dr
            c += dc
    return hits


def _select_template(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame = _find_frame(g)
    r0, c0, r1, c1 = frame
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = interior == SEED
    seeds = np.argwhere(seed_mask)
    assert 1 <= int(seeds.shape[0]) <= 3, "expected 1-3 red seeds"

    templates, cell_to_id = _template_components(g, bg)
    assert 2 <= len(templates) <= 3, "expected 2-3 outer templates"

    hit_sets: list[set[int]] = []
    for sr, sc in seeds:
        hits = _seed_hits(g, (int(r0 + 1 + sr), int(c0 + 1 + sc)), cell_to_id, bg)
        assert hits, "each seed must ray-hit at least one template"
        hit_sets.append(hits)

    candidates = set.intersection(*hit_sets) if hit_sets else set()
    assert candidates, "seed rays must agree on a template"

    matches: list[tuple[np.ndarray, dict[str, object]]] = []
    ih, iw = interior.shape
    for comp_id in sorted(candidates):
        template = templates[comp_id]
        patch = np.array(template["patch"], dtype=int)
        th, tw = patch.shape
        if ih % th != 0 or iw % tw != 0:
            continue
        sr = ih // th
        sc = iw // tw
        scaled = np.kron(patch, np.ones((sr, sc), dtype=int))
        if np.all(scaled[seed_mask] != int(bg)):
            matches.append((scaled, template))

    assert len(matches) == 1, "rays and scaled seed coverage must select one template"
    return frame, interior, matches[0][0], matches[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer which outer template the red seeds point to, then scale it to complete the frame interior.

    Input structure:
    - The background is teal, with one grey one-cell-thick hollow rectangular frame.
    - The frame interior contains 1-3 red seed pixels on otherwise empty cells.
    - Outside the frame, 2-3 disconnected multicolor templates sit on different cardinal sides.

    Rule:
    - Detect the hollow grey frame and its interior size.
    - From every red seed, cast orthogonal rays and collect the first non-background, non-grey template hit.
    - The unique template hit by all seeds is the source template; scale its bbox to the frame interior size.
    - Replace the frame interior with that scaled template and erase every outer template and seed pixel.

    Key insight:
    - The red seeds are not partial fill pixels directly; they are line-of-sight selectors that identify which outside template must be magnified into the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame, _interior, scaled, _template = _select_template(g)
    r0, c0, r1, c1 = frame

    out = np.full(g.shape, int(bg), dtype=int)
    out[r0 : r1 + 1, c0] = FRAME
    out[r0 : r1 + 1, c1] = FRAME
    out[r0, c0 : c1 + 1] = FRAME
    out[r1, c0 : c1 + 1] = FRAME
    out[r0 + 1 : r1, c0 + 1 : c1] = scaled
    return out


def _random_mask(h: int, w: int) -> np.ndarray | None:
    area_min = max(5, (h * w + 1) // 2)
    area_max = h * w - 1
    for _ in range(120):
        target_area = random.randint(area_min, area_max)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if np.all(mask) or int(mask.sum()) < area_min:
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    return None


def _colorize_mask(mask: np.ndarray, colors: tuple[int, ...]) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h, w), BG, dtype=int)
    fg = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    cells = fg[:]
    random.shuffle(cells)
    buckets = [set() for _ in colors]
    for idx, cell in enumerate(cells[: len(colors)]):
        buckets[idx].add(cell)
    for cell in cells[len(colors) :]:
        buckets[random.randrange(len(colors))].add(cell)
    for color, bucket in zip(colors, buckets):
        for r, c in bucket:
            patch[r, c] = int(color)
    return patch


def _random_template_patch(shape: tuple[int, int], used_masks: set[bytes]) -> np.ndarray | None:
    h, w = shape
    for _ in range(120):
        mask = _random_mask(h, w)
        if mask is None:
            continue
        sig = mask.tobytes()
        if sig in used_masks:
            continue
        used_masks.add(sig)
        n_colors = random.randint(2, 3)
        colors = tuple(random.sample(TEMPLATE_COLORS, n_colors))
        return _colorize_mask(mask, colors)
    return None


def _direction_position(
    frame_bbox: tuple[int, int, int, int],
    patch_shape: tuple[int, int],
    direction: str,
    gap: int,
    phase: int | None = None,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    ph, pw = patch_shape
    if direction == "N":
        top = r0 - gap - ph
        left = c0 + 1 + (0 if phase is None else phase)
        return top, left
    if direction == "S":
        top = r1 + gap + 1
        left = c0 + 1 + (0 if phase is None else phase)
        return top, left
    if direction == "W":
        top = r0 + 1 + (0 if phase is None else phase)
        left = c0 - gap - pw
        return top, left
    top = r0 + 1 + (0 if phase is None else phase)
    left = c1 + gap + 1
    return top, left


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _fit_rects(
    h: int,
    w: int,
    frame_bbox: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    direction: str,
    patch: np.ndarray,
    gap: int,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    r0, c0, r1, c1 = frame_bbox
    if direction in ("N", "S"):
        top = r0 - gap - ph if direction == "N" else r1 + gap + 1
        lo = max(1, c0 - pw + 2)
        hi = min(w - pw - 1, c1 - 1)
        candidates = [(top, left) for left in range(lo, hi + 1)]
    else:
        left = c0 - gap - pw if direction == "W" else c1 + gap + 1
        lo = max(1, r0 - ph + 2)
        hi = min(h - ph - 1, r1 - 1)
        candidates = [(top, left) for top in range(lo, hi + 1)]
    random.shuffle(candidates)
    for top, left in candidates:
        rect = (top, left, top + ph - 1, left + pw - 1)
        if rect[0] < 0 or rect[1] < 0 or rect[2] >= h or rect[3] >= w:
            continue
        if any(_rect_overlap(rect, other) for other in placed):
            continue
        return top, left
    return None


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    area = grid[top : top + ph, left : left + pw]
    mask = patch != BG
    area[mask] = patch[mask]


def _build_expected(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    scaled_target: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    out[r0 : r1 + 1, c0] = FRAME
    out[r0 : r1 + 1, c1] = FRAME
    out[r0, c0 : c1 + 1] = FRAME
    out[r1, c0 : c1 + 1] = FRAME
    out[r0 + 1 : r1, c0 + 1 : c1] = scaled_target
    return out


def _make_attempt() -> np.ndarray | None:
    n_templates = random.randint(2, 3)
    target_direction = random.choice(tuple(DIRS))
    other_directions = random.sample([d for d in DIRS if d != target_direction], n_templates - 1)
    directions = [target_direction, *other_directions]
    random.shuffle(directions)

    target_shape = random.choice(((3, 3), (3, 4), (4, 3), (4, 4)))
    valid_scales: list[tuple[int, int]] = []
    for sr in (1, 2):
        for sc in (1, 2):
            ih = target_shape[0] * sr
            iw = target_shape[1] * sc
            if 4 <= ih <= 8 and 4 <= iw <= 8 and (sr, sc) != (1, 1):
                valid_scales.append((sr, sc))
    scale_r, scale_c = random.choice(valid_scales)
    interior_h = target_shape[0] * scale_r
    interior_w = target_shape[1] * scale_c
    outer_h = interior_h + 2
    outer_w = interior_w + 2

    used_masks: set[bytes] = set()
    target_patch = _random_template_patch(target_shape, used_masks)
    if target_patch is None:
        return None

    patches: dict[str, np.ndarray] = {target_direction: target_patch}
    for direction in other_directions:
        patch = _random_template_patch(random.choice(((3, 3), (3, 4), (4, 3), (4, 4))), used_masks)
        if patch is None:
            return None
        patches[direction] = patch

    gaps = {direction: random.randint(1, 3) for direction in directions}
    north_need = patches["N"].shape[0] + gaps["N"] if "N" in patches else 0
    south_need = patches["S"].shape[0] + gaps["S"] if "S" in patches else 0
    west_need = patches["W"].shape[1] + gaps["W"] if "W" in patches else 0
    east_need = patches["E"].shape[1] + gaps["E"] if "E" in patches else 0

    min_h = north_need + outer_h + south_need + 1
    min_w = west_need + outer_w + east_need + 1
    h_lo = max(18, min_h)
    h_hi = min(26, max(h_lo, min_h + 4))
    w_lo = max(18, min_w)
    w_hi = min(26, max(w_lo, min_w + 4))
    if h_lo > h_hi or w_lo > w_hi:
        return None
    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)
    if h > 30 or w > 30:
        return None

    top_lo = north_need
    top_hi = h - outer_h - south_need
    left_lo = west_need
    left_hi = w - outer_w - east_need
    if top_lo > top_hi or left_lo > left_hi:
        return None
    fr = random.randint(top_lo, top_hi)
    fc = random.randint(left_lo, left_hi)
    frame_bbox = (fr, fc, fr + outer_h - 1, fc + outer_w - 1)

    scaled_target = np.kron(target_patch, np.ones((scale_r, scale_c), dtype=int))
    target_mask = scaled_target != BG
    if target_direction in ("N", "S"):
        phase = random.randrange(scale_c)
        allowed = np.argwhere(target_mask & ((np.arange(interior_w)[None, :] % scale_c) == phase))
    else:
        phase = random.randrange(scale_r)
        allowed = np.argwhere(target_mask & ((np.arange(interior_h)[:, None] % scale_r) == phase))
    if allowed.size == 0:
        return None

    target_top, target_left = _direction_position(
        frame_bbox, target_patch.shape, target_direction, gaps[target_direction], phase
    )
    target_rect = (
        target_top,
        target_left,
        target_top + target_patch.shape[0] - 1,
        target_left + target_patch.shape[1] - 1,
    )
    if target_rect[0] < 0 or target_rect[1] < 0 or target_rect[2] >= h or target_rect[3] >= w:
        return None

    placed = [frame_bbox, target_rect]
    positions = {target_direction: (target_top, target_left)}
    for direction in other_directions:
        pos = _fit_rects(h, w, frame_bbox, placed, direction, patches[direction], gaps[direction])
        if pos is None:
            return None
        top, left = pos
        rect = (top, left, top + patches[direction].shape[0] - 1, left + patches[direction].shape[1] - 1)
        placed.append(rect)
        positions[direction] = pos

    input_grid = np.full((h, w), BG, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    input_grid[r0 : r1 + 1, c0] = FRAME
    input_grid[r0 : r1 + 1, c1] = FRAME
    input_grid[r0, c0 : c1 + 1] = FRAME
    input_grid[r1, c0 : c1 + 1] = FRAME

    for direction, patch in patches.items():
        top, left = positions[direction]
        _paste(input_grid, patch, top, left)

    seed_count = random.randint(1, min(3, int(allowed.shape[0])))
    chosen = random.sample(range(int(allowed.shape[0])), seed_count)
    for idx in chosen:
        sr, sc = map(int, allowed[idx])
        input_grid[r0 + 1 + sr, c0 + 1 + sc] = SEED

    expected = _build_expected((h, w), frame_bbox, scaled_target)
    try:
        actual = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
