import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
STAMP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_canvas_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "expected a grey canvas border"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "canvas border must enclose a non-empty interior"

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert np.all(sub[border] == FRAME), "canvas perimeter must be entirely grey"
    assert not np.any(sub[~border] == FRAME), "grey pixels may only appear on the canvas perimeter"

    exact = np.zeros_like(grid, dtype=bool)
    exact[r0 : r1 + 1, c0 : c1 + 1] = border
    assert np.array_equal(exact, grid == FRAME), "expected a single rectangular canvas border"
    return r0, c0, r1, c1


def _read_sequence(grid: np.ndarray, canvas_bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, _r1, c1 = canvas_bbox
    assert r0 > 0, "marker row must sit above the canvas border"
    seq = [int(grid[r0 - 1, c]) for c in range(c0, c1 + 1) if int(grid[r0 - 1, c]) != BG]
    assert seq, "expected at least one color marker above the canvas"
    return seq


def _extract_stamps(grid: np.ndarray, canvas_bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, _r1, c1 = canvas_bbox
    work = np.array(grid, copy=True)
    work[work == FRAME] = BG
    work[r0 - 1, c0 : c1 + 1] = BG

    stamps: dict[int, np.ndarray] = {}
    for coords in _components(work != BG):
        values = {int(v) for v in work[coords[:, 0], coords[:, 1]]}
        assert len(values) == 1, "each stamp must be monochromatic"
        color = values.pop()
        br0, bc0, br1, bc1 = _bbox(coords)
        mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
        mask[coords[:, 0] - br0, coords[:, 1] - bc0] = True
        assert color not in stamps, "stamp colors must be unique"
        stamps[color] = mask

    assert stamps, "expected at least one stamp in the library"
    return stamps


def _simulate_sequence(
    stamp_masks: dict[int, np.ndarray],
    sequence: list[int],
    inner_h: int,
    inner_w: int,
    *,
    paint: bool,
) -> dict | None:
    canvas = np.full((inner_h, inner_w), BG, dtype=int) if paint else None
    clip = False
    top = 0
    left = 0

    for color in sequence:
        assert color in stamp_masks, "every marker color must map to a stamp"
        coords = np.argwhere(stamp_masks[color])
        drawn: list[tuple[int, int]] = []
        for sr, sc in coords:
            rr = top + int(sr)
            cc = left + int(sc)
            if 0 <= rr < inner_h and 0 <= cc < inner_w:
                drawn.append((rr, cc))
                if paint:
                    canvas[rr, cc] = color
        if not drawn:
            return None
        if len(drawn) != int(coords.shape[0]):
            clip = True
        anchor_r, anchor_c = max(drawn)
        top = anchor_r + 1
        left = anchor_c + 1

    return {"canvas": canvas, "clip": clip}


def _required_inner_size(stamp_masks: dict[int, np.ndarray], sequence: list[int]) -> tuple[int, int]:
    max_r = -1
    max_c = -1
    top = 0
    left = 0

    for color in sequence:
        coords = np.argwhere(stamp_masks[color])
        placed = [(top + int(sr), left + int(sc)) for sr, sc in coords]
        anchor_r, anchor_c = max(placed)
        max_r = max(max_r, max(r for r, _ in placed))
        max_c = max(max_c, max(c for _, c in placed))
        top = anchor_r + 1
        left = anchor_c + 1

    return max_r + 1, max_c + 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    canvas_bbox = _find_canvas_bbox(g)
    sequence = _read_sequence(g, canvas_bbox)
    stamps = _extract_stamps(g, canvas_bbox)
    assert set(sequence).issubset(stamps), "marker colors must all exist in the stamp library"

    r0, c0, r1, c1 = canvas_bbox
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    sim = _simulate_sequence(stamps, sequence, inner.shape[0], inner.shape[1], paint=True)
    assert sim is not None, "generated inputs always keep every stamped step at least partially visible"

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = sim["canvas"]
    return out


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        steps = []
        if r != tr:
            steps.append((r + (1 if tr > r else -1), c))
        if c != tc:
            steps.append((r, c + (1 if tc > c else -1)))
        r, c = rng.choice(steps)
        mask[r, c] = True


def _make_path_stamp(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    points = [
        (0, rng.randrange(w)),
        (h - 1, rng.randrange(w)),
        (rng.randrange(h), 0),
        (rng.randrange(h), w - 1),
    ]
    rng.shuffle(points)
    current = points[0]
    mask[current] = True
    for nxt in points[1:]:
        _draw_path(mask, current, nxt, rng)
        current = nxt

    min_target = max(int(mask.sum()), max(5, (h * w) // 2))
    max_target = max(min_target, h * w - 1)
    target = rng.randint(min_target, max_target)
    while int(mask.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = rng.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _make_stamp_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    if h >= 4 and w >= 4 and rng.random() < 0.25:
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
        return mask
    return _make_path_stamp(h, w, rng)


def _arrange_library(
    stamp_masks: dict[int, np.ndarray], rng: random.Random
) -> tuple[int, int, dict[int, tuple[int, int]]] | None:
    items = sorted(stamp_masks.items(), key=lambda kv: kv[1].shape[0] * kv[1].shape[1], reverse=True)
    for _ in range(120):
        patch_h = rng.randint(11, 18)
        patch_w = rng.randint(10, 16)
        blocked = np.zeros((patch_h, patch_w), dtype=bool)
        placements: dict[int, tuple[int, int]] = {}
        ok = True
        for color, mask in items:
            h, w = mask.shape
            choices = []
            for r in range(patch_h - h + 1):
                for c in range(patch_w - w + 1):
                    if not blocked[r : r + h, c : c + w].any():
                        choices.append((r, c))
            if not choices:
                ok = False
                break
            r, c = rng.choice(choices)
            placements[color] = (r, c)
            blocked[max(0, r - 1) : min(patch_h, r + h + 1), max(0, c - 1) : min(patch_w, c + w + 1)] = True
        if not ok:
            continue

        min_r = min(r for r, _ in placements.values())
        min_c = min(c for _, c in placements.values())
        max_r = max(placements[color][0] + stamp_masks[color].shape[0] - 1 for color in placements)
        max_c = max(placements[color][1] + stamp_masks[color].shape[1] - 1 for color in placements)
        cropped = {color: (r - min_r, c - min_c) for color, (r, c) in placements.items()}
        return max_r - min_r + 1, max_c - min_c + 1, cropped
    return None


def _choose_sequence(colors: list[int], rng: random.Random) -> list[int]:
    length = rng.randint(2, 6)
    if rng.random() < 0.5:
        sequence = [rng.choice(colors) for _ in range(length)]
        if len(colors) > 1 and len(set(sequence)) == length and length >= 3:
            sequence[rng.randrange(length)] = sequence[rng.randrange(length)]
        return sequence

    pool = colors[:]
    rng.shuffle(pool)
    sequence: list[int] = []
    while len(sequence) < length:
        block = colors[:]
        rng.shuffle(block)
        sequence.extend(block)
    return sequence[:length]


def _choose_canvas_size(
    stamp_masks: dict[int, np.ndarray],
    sequence: list[int],
    force_clip: bool,
    rng: random.Random,
) -> tuple[int, int] | None:
    req_h, req_w = _required_inner_size(stamp_masks, sequence)
    first_h, first_w = stamp_masks[sequence[0]].shape

    if not force_clip:
        inner_h = max(7, req_h + rng.randint(0, 3))
        inner_w = max(8, req_w + rng.randint(0, 3))
        if inner_h > 22 or inner_w > 22:
            return None
        sim = _simulate_sequence(stamp_masks, sequence, inner_h, inner_w, paint=False)
        if sim is None or sim["clip"]:
            return None
        return inner_h, inner_w

    min_h = max(6, first_h)
    min_w = max(6, first_w)
    max_h = min(22, max(min_h, req_h))
    max_w = min(22, max(min_w, req_w))
    for _ in range(160):
        inner_h = rng.randint(min_h, max_h)
        inner_w = rng.randint(min_w, max_w)
        if inner_h >= req_h and inner_w >= req_w:
            continue
        sim = _simulate_sequence(stamp_masks, sequence, inner_h, inner_w, paint=False)
        if sim is not None and sim["clip"]:
            return inner_h, inner_w
    return None


def _render_library(
    grid: np.ndarray,
    stamp_masks: dict[int, np.ndarray],
    placements: dict[int, tuple[int, int]],
    top: int,
    left: int,
) -> None:
    for color, mask in stamp_masks.items():
        pr, pc = placements[color]
        cells = np.argwhere(mask)
        for dr, dc in cells:
            grid[top + pr + int(dr), left + pc + int(dc)] = color


def _expected_output_from_parts(
    grid: np.ndarray,
    canvas_bbox: tuple[int, int, int, int],
    stamp_masks: dict[int, np.ndarray],
    sequence: list[int],
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = canvas_bbox
    sim = _simulate_sequence(stamp_masks, sequence, r1 - r0 - 1, c1 - c0 - 1, paint=True)
    assert sim is not None
    out[r0 + 1 : r1, c0 + 1 : c1] = sim["canvas"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stamp_count = rng.randint(2, 4)
    colors = rng.sample(STAMP_COLORS, stamp_count)
    stamp_masks = {color: _make_stamp_mask(rng) for color in colors}

    arrangement = _arrange_library(stamp_masks, rng)
    if arrangement is None:
        return None
    lib_h, lib_w, placements = arrangement

    sequence = _choose_sequence(colors, rng)
    force_clip = rng.random() < 0.5
    inner_size = _choose_canvas_size(stamp_masks, sequence, force_clip, rng)
    if inner_size is None:
        return None
    inner_h, inner_w = inner_size

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if len(sequence) > outer_w:
        return None

    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    library_top = rng.randint(1, 3)
    canvas_top = rng.randint(1, 3)
    gap = rng.randint(2, 5)

    canvas_left = left_margin + lib_w + gap
    grid_h = max(library_top + lib_h, canvas_top + outer_h) + bottom_margin
    grid_w = canvas_left + outer_w + right_margin
    grid = np.zeros((grid_h, grid_w), dtype=int)

    _render_library(grid, stamp_masks, placements, library_top, left_margin)

    r0 = canvas_top
    c0 = canvas_left
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    marker_start = c0 + rng.randint(0, outer_w - len(sequence))
    for idx, color in enumerate(sequence):
        grid[r0 - 1, marker_start + idx] = color

    expected = _expected_output_from_parts(grid, (r0, c0, r1, c1), stamp_masks, sequence)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    sim = _simulate_sequence(stamp_masks, sequence, inner_h, inner_w, paint=False)
    if sim is None or sim["clip"] != force_clip:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
