import random

import numpy as np


BG = 0
RED = 2
GREY = 5
ENVELOPE_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
STAMP_RETRIES = 200
LAYOUT_RETRIES = 80


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), np.asarray(mask, dtype=bool).tobytes()


def _all_rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _embed(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    canvas[:h, :w] = mask
    return canvas


def _find_rect_frames(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    frames = []
    for color in map(int, np.unique(grid)):
        if color in (BG, RED):
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            local = np.zeros((h, w), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            frame = np.zeros((h, w), dtype=bool)
            frame[0, :] = True
            frame[-1, :] = True
            frame[:, 0] = True
            frame[:, -1] = True
            if np.array_equal(local, frame):
                frames.append((color, (r0, c0, r1, c1)))
    return frames


def _extract_stamp(grid: np.ndarray, frames: list[tuple[int, tuple[int, int, int, int]]]):
    candidates = []
    for color, bbox in frames:
        if color != GREY:
            continue
        r0, c0, r1, c1 = bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        red = inner == RED
        if red.any():
            candidates.append((bbox, _crop_mask(red)))
    assert len(candidates) == 1
    return candidates[0]


def _matching_rotations(clue: np.ndarray, rotations: list[np.ndarray]) -> list[int]:
    matches = []
    ih, iw = clue.shape
    for idx, rot in enumerate(rotations):
        rh, rw = rot.shape
        if rh > ih or rw > iw:
            continue
        allowed = np.zeros((ih, iw), dtype=bool)
        allowed[:rh, :rw] = rot
        if not np.any(clue & ~allowed):
            matches.append(idx)
    return matches


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _find_rect_frames(g)
    template_bbox, stamp = _extract_stamp(g, frames)
    rotations = _all_rotations(stamp)

    out = g.copy()
    for color, bbox in frames:
        if color == GREY and bbox == template_bbox:
            continue
        r0, c0, r1, c1 = bbox
        clue = g[r0 + 1 : r1, c0 + 1 : c1] == RED
        assert clue.any()
        matches = _matching_rotations(clue, rotations)
        assert len(matches) == 1
        full = rotations[matches[0]]
        rh, rw = full.shape
        patch = out[r0 + 1 : r0 + 1 + rh, c0 + 1 : c0 + 1 + rw]
        patch[full] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the stamp template, infer each envelope's rotation from its partial imprint,
    then complete the rotated stamp inside every envelope.
    """
    return _solve(grid)


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return
    canvas[top + pts[:, 0], left + pts[:, 1]] = color


def _random_stamp() -> np.ndarray:
    for _ in range(STAMP_RETRIES):
        base_h = random.randint(3, 5)
        base_w = random.randint(3, 5)
        area_min = max(4, (base_h * base_w + 2) // 3)
        area_max = min(base_h * base_w - 1, max(area_min, int(round(base_h * base_w * 0.72))))
        if area_min > area_max:
            continue
        target = random.randint(area_min, area_max)
        start = (random.randrange(base_h), random.randrange(base_w))
        cells = {start}
        frontier = set()

        def add_frontier(rc: tuple[int, int]) -> None:
            r, c = rc
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < base_h and 0 <= nc < base_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(start)
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            add_frontier(cell)
        if len(cells) != target:
            continue

        mask = np.zeros((base_h, base_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if not (3 <= mask.shape[0] <= 5 and 3 <= mask.shape[1] <= 5):
            continue
        if len({_mask_key(rot) for rot in _all_rotations(mask)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate asymmetric stamp")


def _make_template_patch(stamp: np.ndarray) -> np.ndarray:
    sh, sw = stamp.shape
    pad_top = random.randint(0, 1)
    pad_bottom = random.randint(0, 1)
    pad_left = random.randint(0, 1)
    pad_right = random.randint(0, 1)
    inner_h = sh + pad_top + pad_bottom
    inner_w = sw + pad_left + pad_right
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    _paint_mask(patch, stamp, 1 + pad_top, 1 + pad_left, RED)
    return patch


def _choose_envelope_size(selected: np.ndarray, stamp: np.ndarray) -> tuple[int, int]:
    sh, sw = stamp.shape
    rh, rw = selected.shape
    if random.random() < 0.5:
        side_h = max(sh, sw)
        side_w = max(sh, sw)
        return side_h + random.randint(0, 2), side_w + random.randint(0, 2)
    return rh + random.randint(0, 2), rw + random.randint(0, 2)


def _choose_clue(
    selected: np.ndarray, rotations: list[np.ndarray], interior_shape: tuple[int, int]
) -> np.ndarray | None:
    selected_canvas = _embed(selected, interior_shape)
    selected_cells = [tuple(pt) for pt in np.argwhere(selected_canvas)]
    mandatory = []
    seen = set()

    ih, iw = interior_shape
    for rot in rotations:
        rh, rw = rot.shape
        if rh > ih or rw > iw:
            continue
        alt_canvas = _embed(rot, interior_shape)
        if np.array_equal(alt_canvas, selected_canvas):
            continue
        diff = np.argwhere(selected_canvas & ~alt_canvas)
        if len(diff) == 0:
            return None
        cell = tuple(diff[random.randrange(len(diff))])
        if cell not in seen:
            seen.add(cell)
            mandatory.append(cell)

    if not mandatory:
        return None

    min_cells = max((len(selected_cells) + 2) // 3, len(mandatory))
    max_cells = len(selected_cells) - 1
    if max_cells < min_cells:
        return None
    target = random.randint(min_cells, max_cells)
    extras = [cell for cell in selected_cells if cell not in seen]
    random.shuffle(extras)
    chosen = mandatory + extras[: target - len(mandatory)]

    clue = np.zeros(interior_shape, dtype=bool)
    for r, c in chosen:
        clue[r, c] = True
    matches = _matching_rotations(clue, rotations)
    if len(matches) != 1:
        return None
    return clue


def _make_envelope_patches(color: int, clue: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ih, iw = clue.shape
    inp = np.full((ih + 2, iw + 2), BG, dtype=int)
    out = np.full((ih + 2, iw + 2), BG, dtype=int)
    inp[0, :] = color
    inp[-1, :] = color
    inp[:, 0] = color
    inp[:, -1] = color
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color
    _paint_mask(inp, clue, 1, 1, RED)
    _paint_mask(out, full, 1, 1, RED)
    return inp, out


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    rows, cols = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rows, top + h + pad)
    c1 = min(cols, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _occupy(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    rows, cols = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rows, top + h + pad)
    c1 = min(cols, left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_patches(shapes: list[tuple[int, int]]):
    total_area = sum(h * w for h, w in shapes)
    max_h = max(h for h, _ in shapes)
    max_w = max(w for _, w in shapes)

    for _ in range(LAYOUT_RETRIES):
        density = random.uniform(0.35, 0.55)
        base = int(np.ceil(np.sqrt(total_area / density)))
        rows = max(15, max_h + 4, base + random.randint(-3, 4))
        cols = max(15, max_w + 4, base + random.randint(-3, 4))
        rows = min(30, rows)
        cols = min(30, cols)
        if rows * cols < total_area or rows <= max_h + 2 or cols <= max_w + 2:
            continue

        occupied = np.zeros((rows, cols), dtype=bool)
        placements = [None] * len(shapes)

        th, tw = shapes[0]
        top_max = min(3, rows - th - 1)
        left_max = min(3, cols - tw - 1)
        if top_max < 1 or left_max < 1:
            continue
        ttop = random.randint(1, top_max)
        tleft = random.randint(1, left_max)
        placements[0] = (ttop, tleft)
        _occupy(occupied, ttop, tleft, th, tw)

        order = sorted(
            range(1, len(shapes)),
            key=lambda idx: shapes[idx][0] * shapes[idx][1],
            reverse=True,
        )
        ok = True
        for idx in order:
            h, w = shapes[idx]
            candidates = [
                (r, c)
                for r in range(1, rows - h)
                for c in range(1, cols - w)
                if _can_place(occupied, r, c, h, w)
            ]
            if not candidates:
                ok = False
                break
            pos = random.choice(candidates)
            placements[idx] = pos
            _occupy(occupied, pos[0], pos[1], h, w)
        if ok:
            return rows, cols, placements
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    stamp = _random_stamp()
    rotations = _all_rotations(stamp)
    template_patch = _make_template_patch(stamp)

    env_count = random.randint(2, 5)
    colors = random.sample(ENVELOPE_COLORS, env_count)
    env_inputs = []
    env_outputs = []
    for color in colors:
        rot_idx = random.randrange(4)
        full = rotations[rot_idx]
        interior_shape = _choose_envelope_size(full, stamp)
        clue = _choose_clue(full, rotations, interior_shape)
        if clue is None:
            return None
        env_inp, env_out = _make_envelope_patches(color, clue, full)
        env_inputs.append(env_inp)
        env_outputs.append(env_out)

    shapes = [tuple(template_patch.shape)] + [tuple(p.shape) for p in env_inputs]
    placed = _place_patches(shapes)
    if placed is None:
        return None
    rows, cols, positions = placed

    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)

    ttop, tleft = positions[0]
    th, tw = template_patch.shape
    inp[ttop : ttop + th, tleft : tleft + tw] = template_patch
    out[ttop : ttop + th, tleft : tleft + tw] = template_patch

    for idx, (env_inp, env_out) in enumerate(zip(env_inputs, env_outputs), start=1):
        top, left = positions[idx]
        h, w = env_inp.shape
        inp[top : top + h, left : left + w] = env_inp
        out[top : top + h, left : left + w] = env_out

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
