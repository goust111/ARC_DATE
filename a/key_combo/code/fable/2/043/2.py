from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PAIR_MIN_COMPONENT = 4
PAIR_MIN_COLORS = 2
GEN_RETRIES = 500
POSITION_SAMPLE_LIMIT = 24


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _largest_component_size(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q: deque[tuple[int, int]] = deque([(rr, cc)])
        size = 0
        while q:
            cr, cc = q.popleft()
            size += 1
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        best = max(best, size)
    return int(best)


def _split_frames(grid: np.ndarray) -> tuple[str, int, list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    def parse_row_separators() -> tuple[str, int, list[np.ndarray]] | None:
        rows = [r for r in range(h) if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != bg]
        if not rows:
            return None
        colors = {int(g[r, 0]) for r in rows}
        if len(colors) != 1:
            return None
        cuts = [-1] + rows + [h]
        sizes = [cuts[i + 1] - cuts[i] - 1 for i in range(len(cuts) - 1)]
        if min(sizes) <= 0 or len(set(sizes)) != 1:
            return None
        frames = [g[cuts[i] + 1 : cuts[i + 1], :].copy() for i in range(len(cuts) - 1)]
        return "vertical", int(next(iter(colors))), frames

    def parse_col_separators() -> tuple[str, int, list[np.ndarray]] | None:
        cols = [c for c in range(w) if bool(np.all(g[:, c] == g[0, c])) and int(g[0, c]) != bg]
        if not cols:
            return None
        colors = {int(g[0, c]) for c in cols}
        if len(colors) != 1:
            return None
        cuts = [-1] + cols + [w]
        sizes = [cuts[i + 1] - cuts[i] - 1 for i in range(len(cuts) - 1)]
        if min(sizes) <= 0 or len(set(sizes)) != 1:
            return None
        frames = [g[:, cuts[i] + 1 : cuts[i + 1]].copy() for i in range(len(cuts) - 1)]
        return "horizontal", int(next(iter(colors))), frames

    options = [opt for opt in (parse_row_separators(), parse_col_separators()) if opt is not None]
    assert len(options) == 1, "expected exactly one separator orientation"
    layout, sep_color, frames = options[0]
    assert len(frames) >= 2, "expected at least two frames"
    return layout, sep_color, frames, bg


def _find_wall(frames: list[np.ndarray], bg: int, sep_color: int) -> tuple[int, np.ndarray]:
    colors = sorted({int(v) for frame in frames for v in np.unique(frame)} - {int(bg), int(sep_color)})
    candidates: list[tuple[int, np.ndarray]] = []
    for color in colors:
        masks = [frame == color for frame in frames]
        if any(not bool(mask.any()) for mask in masks):
            continue
        if any(not np.array_equal(mask, masks[0]) for mask in masks[1:]):
            continue
        box = _bbox(masks[0])
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if int(masks[0].sum()) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        candidates.append((int(color), masks[0]))
    assert len(candidates) == 1, "expected one static rectangular wall"
    return candidates[0]


def _pair_candidates(frame_a: np.ndarray, frame_b: np.ndarray, bg: int) -> list[tuple[int, int]]:
    a = np.asarray(frame_a, dtype=int)
    b = np.asarray(frame_b, dtype=int)
    h, w = a.shape
    a_cells = [(int(r), int(c), int(a[r, c])) for r, c in np.argwhere(a != bg)]
    b_mask = b != bg
    candidates: list[tuple[int, int]] = []

    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 or dc == 0:
                continue
            overlap = np.zeros((h, w), dtype=bool)
            colors: set[int] = set()
            conflict = False
            for r, c, color in a_cells:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and b_mask[rr, cc]:
                    if int(b[rr, cc]) != color:
                        conflict = True
                        break
                    overlap[rr, cc] = True
                    colors.add(color)
            if conflict or len(colors) < PAIR_MIN_COLORS:
                continue
            if _largest_component_size(overlap) >= PAIR_MIN_COMPONENT:
                candidates.append((int(dr), int(dc)))

    return sorted(set(candidates))


def _reconstruct_union(
    frames: list[np.ndarray],
    shift: tuple[int, int],
    bg: int,
    times: list[int] | None = None,
) -> np.ndarray:
    dr, dc = shift
    if times is None:
        times = list(range(len(frames)))
    pixels: dict[tuple[int, int], int] = {}
    for frame, t in zip(frames, times, strict=True):
        for r, c in np.argwhere(frame != bg):
            rr = int(r) - int(t) * int(dr)
            cc = int(c) - int(t) * int(dc)
            color = int(frame[int(r), int(c)])
            if (rr, cc) in pixels:
                assert pixels[(rr, cc)] == color, "overlapping aligned pixels must agree"
            pixels[(rr, cc)] = color

    assert pixels, "reconstruction cannot be empty"
    rows = [r for r, _c in pixels]
    cols = [c for _r, c in pixels]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for (r, c), color in pixels.items():
        out[r - r0, c - c0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct one moving multicolor sprite by globally aligning several occluded frames.

    Input structure:
    - Full-row or full-column separator lines split the grid into equal frames whose spatial order is the time order.
    - Each frame contains the same solid rectangular wall at the same coordinates on a black background.
    - Besides the wall, each frame shows only the currently visible part of one multicolor sprite.

    Rule:
    - Remove the static wall and test every translation between each neighboring frame pair.
    - Keep the translations whose overlapping visible pixels agree exactly on a substantial repeated patch, then intersect those candidate sets across the whole sequence.
    - The unique common translation is the constant frame-to-frame motion; shift every frame back into one reference system, union the visible sprite pixels, and crop to the minimal bounding box.

    Key insight:
    - Any one adjacent pair is locally ambiguous because the sprite texture repeats, but only one shift remains consistent across all frames at once.
    """
    layout, sep_color, frames, bg = _split_frames(np.asarray(grid, dtype=int))
    _ = layout  # layout is only needed to parse the frames.
    wall_color, _wall_mask = _find_wall(frames, bg, sep_color)
    sprite_frames = [np.where(frame == wall_color, bg, frame).astype(int) for frame in frames]

    pair_candidates = [_pair_candidates(sprite_frames[i], sprite_frames[i + 1], bg) for i in range(len(frames) - 1)]
    assert all(len(cands) >= 2 for cands in pair_candidates), "each adjacent pair should be locally ambiguous"

    common = set(pair_candidates[0])
    for cands in pair_candidates[1:]:
        common &= set(cands)
    assert len(common) == 1, "expected exactly one globally consistent shift"
    shift = next(iter(common))

    return _reconstruct_union(sprite_frames, shift, bg)


def _make_sprite(h: int, w: int, shift: tuple[int, int], palette: list[int]) -> np.ndarray:
    dr, dc = shift
    period = random.choice((2, 3))
    colors = random.sample(palette, period + 4)
    texture = colors[:period]
    motif_a, motif_b = colors[period : period + 2]
    accent_a, accent_b = colors[period + 2 : period + 4]

    sprite = np.zeros((h, w), dtype=int)
    same_diag = dr * dc > 0
    for r in range(h):
        for c in range(w):
            key = (r - c) if same_diag else (r + c)
            sprite[r, c] = texture[key % period]

    motif = np.array([[motif_a, motif_b], [motif_b, motif_a]], dtype=int)
    anchors = [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]
    if h >= 5 and w >= 5:
        anchors.append((h // 2 - 1, w // 2 - 1))

    anchor_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, first in enumerate(anchors):
        for second in anchors[i + 1 :]:
            if abs(first[0] - second[0]) >= 2 or abs(first[1] - second[1]) >= 2:
                anchor_pairs.append((first, second))
    first_anchor, second_anchor = random.choice(anchor_pairs)

    reserved = np.zeros((h, w), dtype=bool)
    for r0, c0 in (first_anchor, second_anchor):
        sprite[r0 : r0 + 2, c0 : c0 + 2] = motif
        reserved[r0 : r0 + 2, c0 : c0 + 2] = True

    accent_pool = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (0, w // 2),
        (h - 1, w // 2),
        (h // 2, 0),
        (h // 2, w - 1),
        (h // 2, w // 2),
    ]
    accent_cells = [(r, c) for r, c in accent_pool if not reserved[r, c]]
    if len(accent_cells) >= 2:
        chosen = random.sample(accent_cells, 2)
        sprite[chosen[0]] = accent_a
        sprite[chosen[1]] = accent_b
    elif accent_cells:
        sprite[accent_cells[0]] = accent_a

    return sprite


def _render_frame(
    sprite: np.ndarray,
    frame_shape: tuple[int, int],
    wall_rect: tuple[int, int, int, int],
    wall_color: int,
    top_left: tuple[int, int],
) -> np.ndarray:
    h, w = frame_shape
    out = np.full((h, w), BG, dtype=int)
    sr, sc = top_left
    sh, sw = sprite.shape
    out[sr : sr + sh, sc : sc + sw] = sprite
    r0, c0, r1, c1 = wall_rect
    out[r0 : r1 + 1, c0 : c1 + 1] = int(wall_color)
    return out


def _join_frames(frames: list[np.ndarray], layout: str, sep_color: int) -> np.ndarray:
    if layout == "horizontal":
        sep = np.full((frames[0].shape[0], 1), int(sep_color), dtype=int)
        pieces: list[np.ndarray] = [frames[0]]
        for frame in frames[1:]:
            pieces.extend([sep, frame])
        return np.hstack(pieces)

    sep = np.full((1, frames[0].shape[1]), int(sep_color), dtype=int)
    pieces = [frames[0]]
    for frame in frames[1:]:
        pieces.extend([sep, frame])
    return np.vstack(pieces)


def _scene_is_valid(
    sprite: np.ndarray,
    frames: list[np.ndarray],
    shift: tuple[int, int],
    wall_color: int,
    sep_color: int,
    layout: str,
) -> bool:
    sprite_frames = [np.where(frame == wall_color, BG, frame).astype(int) for frame in frames]
    full_area = int(np.count_nonzero(sprite))
    if any(int(np.count_nonzero(frame)) == 0 or int(np.count_nonzero(frame)) == full_area for frame in sprite_frames):
        return False

    pair_candidates = [_pair_candidates(sprite_frames[i], sprite_frames[i + 1], BG) for i in range(len(sprite_frames) - 1)]
    if any(len(cands) < 2 for cands in pair_candidates):
        return False
    if any(shift not in cands for cands in pair_candidates):
        return False

    common = set(pair_candidates[0])
    for cands in pair_candidates[1:]:
        common &= set(cands)
    if common != {shift}:
        return False

    if not np.array_equal(_reconstruct_union(sprite_frames, shift, BG), sprite):
        return False

    direct_shift = (shift[0] * (len(sprite_frames) - 1), shift[1] * (len(sprite_frames) - 1))
    direct_candidates = _pair_candidates(sprite_frames[0], sprite_frames[-1], BG)
    if direct_shift not in direct_candidates or len(direct_candidates) < 2:
        return False

    grid = _join_frames(frames, layout, sep_color)
    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, sprite)


def _generate_attempt() -> np.ndarray | None:
    layout = random.choice(("horizontal", "vertical"))
    step = random.choice((1, 1, 2))
    shift = (random.choice((-step, step)), random.choice((-step, step)))

    if layout == "horizontal":
        frame_h = random.randint(8, 12)
        frame_w = 9
        max_sprite_h = min(7, frame_h - 2 * step)
        max_sprite_w = min(6, frame_w - 2 * step)
    else:
        frame_h = 9
        frame_w = random.randint(8, 12)
        max_sprite_h = min(6, frame_h - 2 * step)
        max_sprite_w = min(7, frame_w - 2 * step)

    if max_sprite_h < 4 or max_sprite_w < 4:
        return None

    sprite_h = random.randint(4, max_sprite_h)
    sprite_w = random.randint(4, max_sprite_w)
    sep_color, wall_color = random.sample(range(1, 10), 2)
    sprite_palette = [color for color in range(1, 10) if color not in {sep_color, wall_color}]
    sprite = _make_sprite(sprite_h, sprite_w, shift, sprite_palette)

    wall_h = random.randint(2, min(4, frame_h - 2))
    wall_w = random.randint(2, min(4, frame_w - 2))
    wall_r0 = random.randint(1, frame_h - wall_h - 1)
    wall_c0 = random.randint(1, frame_w - wall_w - 1)
    wall_rect = (wall_r0, wall_c0, wall_r0 + wall_h - 1, wall_c0 + wall_w - 1)

    positions: list[tuple[int, int]] = []
    dr, dc = shift
    for r0 in range(frame_h - sprite_h + 1):
        for c0 in range(frame_w - sprite_w + 1):
            top_lefts = [(r0 + t * dr, c0 + t * dc) for t in range(3)]
            if any(not (0 <= rr <= frame_h - sprite_h and 0 <= cc <= frame_w - sprite_w) for rr, cc in top_lefts):
                continue
            frames = [
                _render_frame(sprite, (frame_h, frame_w), wall_rect, wall_color, pos)
                for pos in top_lefts
            ]
            positions.append((r0, c0))

    random.shuffle(positions)
    for r0, c0 in positions[:POSITION_SAMPLE_LIMIT]:
        top_lefts = [(r0 + t * dr, c0 + t * dc) for t in range(3)]
        frames = [
            _render_frame(sprite, (frame_h, frame_w), wall_rect, wall_color, pos)
            for pos in top_lefts
        ]
        if _scene_is_valid(sprite, frames, shift, wall_color, sep_color, layout):
            return _join_frames(frames, layout, sep_color)

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid multi-frame motion-reconstruction example")
