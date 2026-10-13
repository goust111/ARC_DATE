from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6)
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    cells = {tuple(map(int, p)) for p in coords}
    if len(cells) != 2 * (h + w) - 4:
        return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    return True


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if x not in (BG, GREY)):
        for coords in _components(grid == color):
            if _is_hollow_rectangle(coords):
                frames.append({"color": color, "bbox": _bbox(coords), "coords": coords})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _rotate_sprite_patch(patch: np.ndarray, turns: int) -> np.ndarray:
    turns = int(turns) % 4
    if turns == 0:
        return patch.copy()
    return np.rot90(patch, -turns).copy()


def _repaint_rotated_sprite(
    out: np.ndarray,
    sprite_coords: np.ndarray,
    color: int,
    turns: int,
    frame_bbox: tuple[int, int, int, int],
) -> None:
    sr0, sc0, sr1, sc1 = _bbox(sprite_coords)
    patch = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=int)
    for r, c in sprite_coords:
        patch[int(r) - sr0, int(c) - sc0] = int(color)
    rotated = _rotate_sprite_patch(patch, turns)

    for r, c in sprite_coords:
        out[int(r), int(c)] = BG

    center_r = (sr0 + sr1) / 2.0
    center_c = (sc0 + sc1) / 2.0
    rh, rw = rotated.shape
    new_r0 = int(round(center_r - (rh - 1) / 2.0))
    new_c0 = int(round(center_c - (rw - 1) / 2.0))

    fr0, fc0, fr1, fc1 = frame_bbox
    for rr in range(rh):
        for cc in range(rw):
            if rotated[rr, cc] != color:
                continue
            gr = new_r0 + rr
            gc = new_c0 + cc
            if fr0 < gr < fr1 and fc0 < gc < fc1:
                out[gr, gc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count gray markers inside each frame and rotate the enclosed sprite by that many quarter-turns.

    Input structure:
    - A black background contains several differently colored hollow rectangular frames.
    - Each frame encloses one disconnected same-color sprite and zero to three gray marker pixels.
    - Gray markers lie strictly inside the frame interior and are separate from the sprite.

    Rule:
    - For each frame, count the gray markers strictly inside its bounding box interior.
    - Rotate the enclosed sprite clockwise by N * 90 degrees around its tight-bbox center, clipping any rotated cells that land on the frame border or outside the frame interior.
    - Erase all gray markers to black, while leaving the frames and the rest of the background unchanged.

    Key insight:
    - The gray pixels are per-frame rotation parameters: containment decides which markers control which sprite, and the markers disappear after their count has been used.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    frames = _find_frames(g)
    assert frames, "expected at least one hollow frame"

    out = g.copy()
    out[out == GREY] = BG

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        turns = int(np.count_nonzero(interior == GREY))

        sprite_mask = interior == color
        comps = _components(sprite_mask)
        assert len(comps) == 1, "each frame must contain exactly one same-color sprite"
        sprite_coords = comps[0] + np.array([r0 + 1, c0 + 1], dtype=int)
        _repaint_rotated_sprite(out, sprite_coords, color, turns, (r0, c0, r1, c1))

    return out


_SPRITE_SHAPES = (
    np.array(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _has_ninety_symmetry(mask: np.ndarray) -> bool:
    tight = _tight_mask(mask)
    return any(np.array_equal(tight, np.rot90(tight, k)) for k in (1, 3))


for _shape in _SPRITE_SHAPES:
    assert np.count_nonzero(_shape) in range(3, 7)
    assert not _has_ninety_symmetry(_shape.astype(bool))


def _random_sprite() -> np.ndarray:
    shape = random.choice(_SPRITE_SHAPES)
    turns = random.randint(0, 3)
    return _rotate_sprite_patch(shape, turns)


def _reserve_ok(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + h + 1)
    cc1 = min(reserved.shape[1], c0 + w + 1)
    return not reserved[rr0:rr1, cc0:cc1].any()


def _frame_positions(height: int, width: int, frame_sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    reserved = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * len(frame_sizes)
    order = sorted(range(len(frame_sizes)), key=lambda i: frame_sizes[i][0] * frame_sizes[i][1], reverse=True)
    for idx in order:
        fh, fw = frame_sizes[idx]
        candidates = [
            (r, c)
            for r in range(height - fh + 1)
            for c in range(width - fw + 1)
            if _reserve_ok(reserved, r, c, fh, fw)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        reserved[r0 : r0 + fh, c0 : c0 + fw] = True
    return placements


def _place_sprite(frame: np.ndarray, color: int, sprite: np.ndarray) -> np.ndarray | None:
    fh, fw = frame.shape
    sh, sw = sprite.shape
    interior_h = fh - 2
    interior_w = fw - 2
    if sh > interior_h or sw > interior_w:
        return None

    center_r = (interior_h - sh) / 2.0
    center_c = (interior_w - sw) / 2.0
    candidates = [
        (ir, ic)
        for ir in range(interior_h - sh + 1)
        for ic in range(interior_w - sw + 1)
    ]
    candidates.sort(key=lambda pos: (abs(pos[0] - center_r) + abs(pos[1] - center_c), random.random()))
    ir, ic = candidates[0]
    patch = np.where(sprite, color, BG)
    frame[1 + ir : 1 + ir + sh, 1 + ic : 1 + ic + sw][sprite.astype(bool)] = int(color)
    return np.array([(1 + ir + r, 1 + ic + c) for r, c in np.argwhere(sprite)], dtype=int)


def _marker_candidates(frame: np.ndarray, sprite_coords: np.ndarray) -> list[tuple[int, int]]:
    fh, fw = frame.shape
    blocked = np.zeros_like(frame, dtype=bool)
    for r, c in sprite_coords:
        for rr in range(max(1, int(r) - 1), min(fh - 1, int(r) + 2)):
            for cc in range(max(1, int(c) - 1), min(fw - 1, int(c) + 2)):
                blocked[rr, cc] = True
    choices: list[tuple[int, int]] = []
    for r in range(1, fh - 1):
        for c in range(1, fw - 1):
            if frame[r, c] == BG and not blocked[r, c]:
                choices.append((r, c))
    return choices


def _draw_frame_patch(color: int, frame_h: int, frame_w: int, marker_count: int) -> np.ndarray | None:
    frame = np.zeros((frame_h, frame_w), dtype=int)
    frame[0, :] = color
    frame[-1, :] = color
    frame[:, 0] = color
    frame[:, -1] = color

    sprite = _random_sprite()
    sprite_coords = _place_sprite(frame, color, sprite)
    if sprite_coords is None:
        return None

    marker_cells = _marker_candidates(frame, sprite_coords)
    if len(marker_cells) < marker_count:
        return None
    random.shuffle(marker_cells)
    for r, c in marker_cells[:marker_count]:
        frame[r, c] = GREY
    return frame


def _sample_counts(n_frames: int) -> list[int]:
    base = [0, 1, 2, 3]
    random.shuffle(base)
    counts = base[: min(n_frames, 4)]
    while len(counts) < n_frames:
        counts.append(random.randint(0, 3))
    random.shuffle(counts)
    return counts


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(3, 5)
    min_side = 24 if n_frames == 3 else 26
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    colors = random.sample(FRAME_COLORS, n_frames)
    counts = _sample_counts(n_frames)
    frame_sizes = [(random.randint(7, 10), random.randint(7, 10)) for _ in range(n_frames)]
    positions = _frame_positions(height, width, frame_sizes)
    if positions is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    changed_turns = 0
    for color, marker_count, (fh, fw), (r0, c0) in zip(colors, counts, frame_sizes, positions):
        patch = _draw_frame_patch(color, fh, fw, marker_count)
        if patch is None:
            return None
        grid[r0 : r0 + fh, c0 : c0 + fw] = patch
        if marker_count % 4 != 0:
            changed_turns += 1

    if changed_turns == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
