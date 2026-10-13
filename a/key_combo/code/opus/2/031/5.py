import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

GLYPH_PATTERNS = {
    "A": np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    "B": np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    "C": np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    "D": np.array(
        [
            [1, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    ),
}
GLYPH_TO_COLOR = {"A": RED, "B": TEAL, "C": YELLOW, "D": GREEN}
GLYPH_KEY_TO_COLOR = {
    tuple(mask.astype(int).ravel()): GLYPH_TO_COLOR[name] for name, mask in GLYPH_PATTERNS.items()
}
ANCHOR_COLORS = (RED, TEAL, YELLOW, GREEN)

GEN_RETRIES = 400
GEN_MIN_H = 13
GEN_MAX_H = 20
GEN_MAX_W = 26

_GEN_EMITTED = 0
_GEN_PENDING_GLYPHS: list[str] = []


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(
    grid: np.ndarray,
    *,
    bg: int = BLACK,
    colors: list[int] | tuple[int, ...] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    allowed = set(map(int, np.unique(g) if colors is None else colors))
    allowed.discard(int(bg))
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color not in allowed:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if not in_bounds(g.shape, nr, nc) or seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )
    return out


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(mask.astype(int).ravel())


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns) % 4)


def _column_groups(cols: np.ndarray) -> list[tuple[int, int]]:
    if cols.size == 0:
        return []
    groups = []
    start = prev = int(cols[0])
    for value in map(int, cols[1:]):
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _decode_glyphs(grid: np.ndarray) -> list[dict]:
    grey_cols = np.where(np.any(grid == GREY, axis=0))[0]
    groups = _column_groups(grey_cols)
    glyphs = []
    for c0, c1 in groups:
        rows = np.where(np.any(grid[:, c0 : c1 + 1] == GREY, axis=1))[0]
        assert c1 - c0 + 1 == 4
        assert rows[-1] - rows[0] + 1 == 4
        glyph = grid[rows[0] : rows[-1] + 1, c0 : c1 + 1] == GREY
        color = GLYPH_KEY_TO_COLOR[_mask_key(glyph)]
        glyphs.append({"color": color, "bbox": (int(rows[0]), c0, int(rows[-1]), c1)})
    assert 2 <= len(glyphs) <= 4
    return glyphs


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    comps = find_components(grid, bg=BLACK, colors=[ORANGE], connectivity=4)
    assert len(comps) == 1
    bbox = comps[0]["bbox"]
    return crop_bbox(grid == ORANGE, bbox)


def _find_anchors(grid: np.ndarray) -> list[tuple[int, int, int]]:
    comps = find_components(grid, bg=BLACK, colors=ANCHOR_COLORS, connectivity=4)
    anchors = []
    for comp in comps:
        assert comp["area"] == 1
        r, c = map(int, comp["coords"][0])
        anchors.append((int(comp["color"]), r, c))
    assert 2 <= len(anchors) <= 4
    return anchors


def _stamp_cells(mask: np.ndarray, anchor_r: int, anchor_c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    top = int(anchor_r) - h // 2
    left = int(anchor_c) - w // 2
    return [(top + r, left + c) for r, c in np.argwhere(mask)]


def _stamp_on_background(out: np.ndarray, mask: np.ndarray, anchor_r: int, anchor_c: int, color: int) -> None:
    for r, c in _stamp_cells(mask, anchor_r, anchor_c):
        if in_bounds(out.shape, r, c) and int(out[r, c]) == BLACK:
            out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the top glyph sequence, use the decoded color to choose an anchor, and stamp the rotated sprite at that anchor.

    Input structure:
    - The background is black.
    - The upper strip contains 2-3 separated 4x4 grey glyph masks plus one orange connected sprite template on the right.
    - The lower area contains isolated single-pixel anchors whose colors come from {red, green, yellow, teal}.

    Rule:
    - Read the grey glyphs from left to right and decode each fixed 4x4 mask into one anchor color.
    - For each lower anchor, find the glyph with the same decoded color; if that glyph is at sequence index k, rotate the orange sprite clockwise by k quarter-turns.
    - Center the rotated sprite on the anchor and paint only onto black cells, using the anchor's own color; keep the top glyphs, the orange template, and the original anchors unchanged.

    Key insight:
    - A glyph carries two independent signals at once: its shape determines which anchor it controls, while its left-to-right position determines how much the sprite must rotate before stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    glyphs = _decode_glyphs(g)
    color_to_index: dict[int, int] = {}
    for idx, glyph in enumerate(glyphs):
        color = int(glyph["color"])
        assert color not in color_to_index
        color_to_index[color] = idx

    sprite = _extract_sprite(g)
    anchors = _find_anchors(g)
    jobs = []
    for color, r, c in anchors:
        assert color in color_to_index
        jobs.append((color_to_index[color], color, r, c))
    jobs.sort()

    out = g.copy()
    for turns, color, r, c in jobs:
        _stamp_on_background(out, _rotate_cw(sprite, turns), r, c, color)
    return out


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q = deque([tuple(map(int, coords[0]))])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not in_bounds(mask.shape, nr, nc) or not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_sprite_library() -> list[np.ndarray]:
    sprites = []
    cells = [(r, c) for r in range(3) for c in range(3)]
    for size in range(4, 7):
        for combo in itertools.combinations(cells, size):
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if not mask[1, 1]:
                continue
            if not mask[0].any() or not mask[2].any() or not mask[:, 0].any() or not mask[:, 2].any():
                continue
            if not _connected4(mask):
                continue
            if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
                continue
            key = min(_mask_key(np.rot90(mask, k)) for k in range(4))
            sprites.append((key, mask))
    dedup = {}
    for key, mask in sprites:
        dedup.setdefault(key, mask)
    return list(dedup.values())


SPRITE_LIBRARY = _build_sprite_library()


def _sample_glyph_names(n_glyphs: int) -> list[str]:
    global _GEN_PENDING_GLYPHS
    if not _GEN_PENDING_GLYPHS:
        _GEN_PENDING_GLYPHS = list(GLYPH_PATTERNS)
        random.shuffle(_GEN_PENDING_GLYPHS)
    forced = []
    while _GEN_PENDING_GLYPHS and len(forced) < n_glyphs:
        forced.append(_GEN_PENDING_GLYPHS.pop())
    remaining = [name for name in GLYPH_PATTERNS if name not in forced]
    extras = random.sample(remaining, n_glyphs - len(forced))
    chosen = forced + extras
    random.shuffle(chosen)
    return chosen


def _pick_anchor_positions(
    grid: np.ndarray,
    sprite: np.ndarray,
    jobs: list[tuple[int, int]],
) -> dict[int, tuple[int, int]] | None:
    h, w = grid.shape
    occupied = grid != BLACK
    anchors: dict[int, tuple[int, int]] = {}
    anchor_points: list[tuple[int, int]] = []
    job_order = jobs[:]
    random.shuffle(job_order)

    def search(i: int) -> bool:
        if i == len(job_order):
            return True
        turns, color = job_order[i]
        mask = _rotate_cw(sprite, turns)
        candidates = [(r, c) for r in range(8, h - 1) for c in range(1, w - 1)]
        random.shuffle(candidates)
        for r, c in candidates:
            if any(abs(r - ar) + abs(c - ac) <= 1 for ar, ac in anchor_points):
                continue
            cells = _stamp_cells(mask, r, c)
            if any((not in_bounds(grid.shape, rr, cc)) or occupied[rr, cc] for rr, cc in cells):
                continue
            anchors[color] = (r, c)
            anchor_points.append((r, c))
            for rr, cc in cells:
                occupied[rr, cc] = True
            if search(i + 1):
                return True
            for rr, cc in cells:
                occupied[rr, cc] = False
            anchor_points.pop()
            anchors.pop(color)
        return False

    return anchors if search(0) else None


def _gen_attempt(forced_n_glyphs: int | None) -> np.ndarray | None:
    n_glyphs = forced_n_glyphs if forced_n_glyphs is not None else (3 if random.random() < 0.65 else 2)
    top_row = random.randint(0, 1)
    sprite_row = random.randint(0, 2)
    left_margin = random.randint(0, 2)
    gap_to_sprite = random.randint(2, 4)
    right_margin = random.randint(0, 2)

    glyph_names = _sample_glyph_names(n_glyphs)
    sprite = random.choice(SPRITE_LIBRARY)

    glyph_cols = []
    c = left_margin
    for i in range(n_glyphs):
        glyph_cols.append(c)
        c += 4
        if i != n_glyphs - 1:
            c += 1
    sprite_col = c + gap_to_sprite
    min_w = sprite_col + 3 + right_margin
    if min_w > GEN_MAX_W:
        return None
    w = random.randint(max(15, min_w), GEN_MAX_W)
    h = random.randint(max(GEN_MIN_H, 11), GEN_MAX_H)

    grid = np.zeros((h, w), dtype=int)
    for name, col in zip(glyph_names, glyph_cols):
        mask = GLYPH_PATTERNS[name]
        grid[top_row : top_row + 4, col : col + 4][mask] = GREY
    grid[sprite_row : sprite_row + 3, sprite_col : sprite_col + 3][sprite] = ORANGE

    jobs = [(idx, GLYPH_TO_COLOR[name]) for idx, name in enumerate(glyph_names)]
    anchors = _pick_anchor_positions(grid, sprite, jobs)
    if anchors is None:
        return None
    for color, (r, c) in anchors.items():
        grid[r, c] = color

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out[grid != BLACK], grid[grid != BLACK]):
        return None
    if int(np.count_nonzero(out)) <= int(np.count_nonzero(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_EMITTED
    if _GEN_EMITTED == 0:
        forced_n = 3
    elif _GEN_EMITTED == 1:
        forced_n = 2
    else:
        forced_n = None

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(forced_n)
        if grid is not None:
            _GEN_EMITTED += 1
            return grid
    raise RuntimeError("failed to generate a valid input")
