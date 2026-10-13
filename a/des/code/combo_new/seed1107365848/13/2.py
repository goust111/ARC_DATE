from __future__ import annotations

import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
QUADRANTS = ("tl", "tr", "br", "bl")
ROTATIONS_CCW = {"tl": 0, "tr": 3, "br": 2, "bl": 1}
GEN_RETRIES = 300
COLOR_POOL = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _quadrant_boxes(frame: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame
    top, left = r0 + 1, c0 + 1
    bottom, right = r1 - 1, c1 - 1
    inner_h = bottom - top + 1
    inner_w = right - left + 1
    half_h = inner_h // 2
    half_w = inner_w // 2
    return {
        "tl": (top, left, top + half_h - 1, left + half_w - 1),
        "tr": (top, left + half_w, top + half_h - 1, right),
        "br": (top + half_h, left + half_w, bottom, right),
        "bl": (top + half_h, left, bottom, left + half_w - 1),
    }


def _place_in_corner(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    quad: str,
    box: tuple[int, int, int, int],
) -> None:
    qr0, qc0, qr1, qc1 = box
    sh, sw = mask.shape
    qh = qr1 - qr0 + 1
    qw = qc1 - qc0 + 1
    assert sh <= qh and sw <= qw, "stamp must fit inside its destination quadrant"

    if quad == "tl":
        sr, sc = qr0, qc0
    elif quad == "tr":
        sr, sc = qr0, qc1 - sw + 1
    elif quad == "br":
        sr, sc = qr1 - sh + 1, qc1 - sw + 1
    else:
        sr, sc = qr1 - sh + 1, qc0

    view = out[sr : sr + sh, sc : sc + sw]
    view[mask] = int(color)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, dict[str, int]]:
    best = None
    colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    for color in colors:
        pts = np.argwhere(grid == int(color))
        if pts.shape[0] < 16:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 6 or w < 6:
            continue
        if int(pts.shape[0]) != 2 * (h + w) - 4:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(box[0, :] == color):
            continue
        if not np.all(box[-1, :] == color):
            continue
        if not np.all(box[:, 0] == color):
            continue
        if not np.all(box[:, -1] == color):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = inner.shape
        if inner_h < 2 or inner_w < 2 or inner_h % 2 or inner_w % 2:
            continue

        marker_pos = {
            "tl": (r0 + 1, c0 + 1),
            "tr": (r0 + 1, c1 - 1),
            "br": (r1 - 1, c1 - 1),
            "bl": (r1 - 1, c0 + 1),
        }
        marker_colors = {quad: int(grid[r, c]) for quad, (r, c) in marker_pos.items()}
        vals = list(marker_colors.values())
        if any(v == int(bg) or v == int(color) for v in vals):
            continue
        if len(set(vals)) != 4:
            continue

        clean_inner = inner.copy()
        clean_inner[0, 0] = bg
        clean_inner[0, -1] = bg
        clean_inner[-1, 0] = bg
        clean_inner[-1, -1] = bg
        if not np.all(clean_inner == bg):
            continue

        area = h * w
        cand = ((r0, c0, r1, c1), int(color), marker_colors)
        if best is None or area > best[0]:
            best = (area, cand)

    assert best is not None, "expected a unique rectangular frame with four inner-corner markers"
    return best[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each exterior monochrome stamp into the frame quadrant selected by the matching corner marker.

    Input structure:
    - The grid has a dominant background color, one large monochrome rectangular frame, and a black/empty interior.
    - A single colored marker sits at each of the four inner frame corners, with four distinct colors.
    - Outside the frame, each marker color appears as exactly one small monochrome 4-connected stamp.

    Rule:
    - Match every exterior stamp to the inner-corner marker of the same color.
    - Split the frame interior into four equal quadrants.
    - Rotate the matched stamp by a quadrant-specific amount: top-left stays unchanged, top-right rotates 90 degrees clockwise, bottom-right rotates 180 degrees, and bottom-left rotates 270 degrees clockwise.
    - Place the rotated stamp flush to that quadrant's corner and erase the original exterior stamps and corner markers.

    Key insight:
    - Marker color selects which stamp belongs to each quadrant, while the quadrant position itself determines the required rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    (r0, c0, r1, c1), frame_color, marker_colors = _find_frame(g, bg)
    boxes = _quadrant_boxes((r0, c0, r1, c1))

    out = np.full_like(g, int(bg))
    out[r0 : r1 + 1, c0] = int(frame_color)
    out[r0 : r1 + 1, c1] = int(frame_color)
    out[r0, c0 : c1 + 1] = int(frame_color)
    out[r1, c0 : c1 + 1] = int(frame_color)

    for quad in QUADRANTS:
        color = int(marker_colors[quad])
        mask = g == int(color)
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
        comps = _components(mask)
        assert comps, f"missing exterior stamp for color {color}"
        comp = max(comps, key=len)
        stamp = np.zeros_like(mask, dtype=bool)
        for rr, cc in comp:
            stamp[rr, cc] = True
        stamp = _crop_mask(stamp)
        rotated = np.rot90(stamp, ROTATIONS_CCW[quad])
        _place_in_corner(out, rotated, color, quad, boxes[quad])

    return out


def _random_connected_mask(max_side: int, rng: random.Random, forbidden_turn: int) -> np.ndarray:
    for _ in range(200):
        raw_h = rng.randint(1, max_side)
        raw_w = rng.randint(1, max_side)
        total = raw_h * raw_w
        min_cells = 2 if total > 1 else 1
        target = rng.randint(min_cells, total)
        start = (rng.randrange(raw_h), rng.randrange(raw_w))
        cells = {start}
        frontier = set()
        while len(cells) < target:
            frontier.clear()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        mask = np.zeros((raw_h, raw_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if int(mask.sum()) < 2:
            continue
        if forbidden_turn and np.array_equal(mask, np.rot90(mask, forbidden_turn)):
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate stamp mask")


def _grow_blocked(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                out[nr, nc] = True
    return out


def _find_slot(
    blocked: np.ndarray,
    stamp: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    sh, sw = stamp.shape
    candidates = [
        (r, c)
        for r in range(h - sh + 1)
        for c in range(w - sw + 1)
        if not blocked[r : r + sh, c : c + sw].any()
    ]
    if not candidates:
        return None
    rng.shuffle(candidates)
    return candidates[0]


def _build_expected(
    shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    frame_color: int,
    stamps: dict[str, tuple[int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r0, c0, r1, c1 = frame
    out[r0 : r1 + 1, c0] = int(frame_color)
    out[r0 : r1 + 1, c1] = int(frame_color)
    out[r0, c0 : c1 + 1] = int(frame_color)
    out[r1, c0 : c1 + 1] = int(frame_color)
    boxes = _quadrant_boxes(frame)
    for quad in QUADRANTS:
        color, stamp = stamps[quad]
        rotated = np.rot90(stamp, ROTATIONS_CCW[quad])
        _place_in_corner(out, rotated, int(color), quad, boxes[quad])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0
    q = rng.randint(3, 6)
    frame_h = 2 * q + 2
    frame_w = 2 * q + 2
    min_margin = q + 2
    h = rng.randint(frame_h + 2 * min_margin, 30)
    w = rng.randint(frame_w + 2 * min_margin, 30)
    r0 = rng.randint(min_margin, h - frame_h - min_margin)
    c0 = rng.randint(min_margin, w - frame_w - min_margin)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    palette = list(COLOR_POOL)
    rng.shuffle(palette)
    frame_color = int(palette[0])
    marker_colors = {
        quad: int(color)
        for quad, color in zip(QUADRANTS, palette[1:5])
    }

    grid = np.full((h, w), int(bg), dtype=int)
    grid[r0 : r1 + 1, c0] = int(frame_color)
    grid[r0 : r1 + 1, c1] = int(frame_color)
    grid[r0, c0 : c1 + 1] = int(frame_color)
    grid[r1, c0 : c1 + 1] = int(frame_color)

    marker_pos = {
        "tl": (r0 + 1, c0 + 1),
        "tr": (r0 + 1, c1 - 1),
        "br": (r1 - 1, c1 - 1),
        "bl": (r1 - 1, c0 + 1),
    }
    for quad, (rr, cc) in marker_pos.items():
        grid[rr, cc] = int(marker_colors[quad])

    stamps: dict[str, tuple[int, np.ndarray]] = {}
    seen_masks: set[bytes] = set()
    for quad in QUADRANTS:
        base = _random_connected_mask(q, rng, ROTATIONS_CCW[quad])
        key = base.astype(np.uint8).tobytes() + str(base.shape).encode()
        if key in seen_masks:
            return None
        seen_masks.add(key)
        input_mask = np.rot90(base, rng.randrange(4))
        stamps[quad] = (int(marker_colors[quad]), input_mask)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    for quad in rng.sample(list(QUADRANTS), len(QUADRANTS)):
        color, stamp = stamps[quad]
        slot = _find_slot(_grow_blocked(blocked), stamp, rng)
        if slot is None:
            return None
        sr, sc = slot
        sh, sw = stamp.shape
        view = grid[sr : sr + sh, sc : sc + sw]
        view[stamp] = int(color)
        stamp_box = np.zeros_like(grid, dtype=bool)
        stamp_box[sr : sr + sh, sc : sc + sw] = True
        blocked |= stamp_box

    expected = _build_expected(grid.shape, (r0, c0, r1, c1), frame_color, stamps, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retry limit")
