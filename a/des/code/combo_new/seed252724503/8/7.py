from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")

GRID_MIN = 16
GRID_MAX = 30
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
FRAME_MIN = 5
FRAME_MAX = 8
GAP_MIN = 1
GAP_MAX = 3
OBJECT_MAX_SPAN = 5
GEN_RETRIES = 500
LOCAL_RETRIES = 200


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
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
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_hollow_frame(
    grid: np.ndarray,
    coords: np.ndarray,
    color: int,
    bg: int,
) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.full((h, w), bg, dtype=int)
    expected[0, :] = color
    expected[-1, :] = color
    expected[:, 0] = color
    expected[:, -1] = color
    return len(coords) == 2 * h + 2 * w - 4 and np.array_equal(patch, expected)


def _project_object(
    out: np.ndarray,
    object_coords: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    or0, oc0, or1, oc1 = _bbox(object_coords)

    if or1 < fr0:
        assert fc0 + 1 <= oc0 and oc1 <= fc1 - 1
        out[fr0 + 1, np.unique(object_coords[:, 1])] = color
        return
    if or0 > fr1:
        assert fc0 + 1 <= oc0 and oc1 <= fc1 - 1
        out[fr1 - 1, np.unique(object_coords[:, 1])] = color
        return
    if oc1 < fc0:
        assert fr0 + 1 <= or0 and or1 <= fr1 - 1
        out[np.unique(object_coords[:, 0]), fc0 + 1] = color
        return
    if oc0 > fc1:
        assert fr0 + 1 <= or0 and or1 <= fr1 - 1
        out[np.unique(object_coords[:, 0]), fc1 - 1] = color
        return

    raise AssertionError("object must lie wholly on one side of its frame")


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each external object into its same-colored frame as a one-dimensional projection.

    Input structure:
    - The background is black.
    - Each non-background color appears in exactly one pair: a hollow rectangular
      frame and a separate 4-connected object of the same color.
    - Every object lies wholly above, below, left, or right of its matching frame
      and does not touch it.

    Rule:
    - Pair each object with the frame of the same color.
    - Determine which side of the frame the object occupies.
    - If the object is above or below the frame, project it by column into a
      single row placed just inside the corresponding top or bottom inner wall.
    - If the object is left or right of the frame, project it by row into a
      single column placed just inside the corresponding left or right inner wall.
    - Erase the original external object pixels and keep the frame unchanged.

    Key insight:
    - The transformation depends on three linked cues at once: same-color pairing,
      relative side-of-frame, and projection along the axis pointing toward the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        assert len(comps) == 2, "each color must form exactly one frame-object pair"

        frame = None
        obj = None
        for comp in comps:
            if _is_hollow_frame(g, comp, color, bg):
                assert frame is None, "each color must have exactly one hollow frame"
                frame = comp
            else:
                assert obj is None, "each color must have exactly one non-frame object"
                obj = comp

        assert frame is not None and obj is not None
        out[obj[:, 0], obj[:, 1]] = bg
        _project_object(out, obj, _bbox(frame), color)

    return out


def _draw_frame(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[r0, c0 : c0 + w] = color
    canvas[r0 + h - 1, c0 : c0 + w] = color
    canvas[r0 : r0 + h, c0] = color
    canvas[r0 : r0 + h, c0 + w - 1] = color


def _paste_non_bg(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(mask & (target != BG))
    target[mask] = patch[mask]


def _random_blob(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(LOCAL_RETRIES):
        seed_h = int(rng.integers(2, max_h + 1))
        seed_w = int(rng.integers(2, max_w + 1))
        area = seed_h * seed_w
        if area <= 4:
            continue

        min_cells = max(3, (area + 1) // 2)
        target_cells = int(rng.integers(min_cells, area))

        mask = np.zeros((seed_h, seed_w), dtype=bool)
        start = (int(rng.integers(seed_h)), int(rng.integers(seed_w)))
        mask[start] = True
        frontier = set(_neighbors4(start[0], start[1], seed_h, seed_w))

        while int(mask.sum()) < target_cells and frontier:
            frontier_list = tuple(frontier)
            cell = frontier_list[int(rng.integers(len(frontier_list)))]
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            for nbr in _neighbors4(cell[0], cell[1], seed_h, seed_w):
                if not mask[nbr]:
                    frontier.add(nbr)

        if int(mask.sum()) != target_cells:
            continue

        mask = _crop_mask(mask)
        h, w = mask.shape
        if h < 2 or w < 2 or mask.all():
            continue
        if h >= 3 and w >= 3:
            frame = np.zeros((h, w), dtype=bool)
            frame[0, :] = True
            frame[-1, :] = True
            frame[:, 0] = True
            frame[:, -1] = True
            if np.array_equal(mask, frame):
                continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        return mask
    return None


def _render_pair(
    rng: np.random.Generator,
    color: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    frame_h = int(rng.integers(FRAME_MIN, FRAME_MAX + 1))
    frame_w = int(rng.integers(FRAME_MIN, FRAME_MAX + 1))
    gap = int(rng.integers(GAP_MIN, GAP_MAX + 1))

    if direction in ("up", "down"):
        obj_mask = _random_blob(rng, OBJECT_MAX_SPAN, frame_w - 2)
        if obj_mask is None:
            return None
        obj_h, obj_w = obj_mask.shape
        offset = int(rng.integers(0, frame_w - 2 - obj_w + 1))
        patch_h = obj_h + gap + frame_h
        patch_w = frame_w
        frame_r0 = obj_h + gap if direction == "up" else 0
        frame_c0 = 0
        obj_r0 = 0 if direction == "up" else frame_h + gap
        obj_c0 = 1 + offset
        target_r = frame_r0 + 1 if direction == "up" else frame_r0 + frame_h - 2
        target_c = None
    else:
        obj_mask = _random_blob(rng, frame_h - 2, OBJECT_MAX_SPAN)
        if obj_mask is None:
            return None
        obj_h, obj_w = obj_mask.shape
        offset = int(rng.integers(0, frame_h - 2 - obj_h + 1))
        patch_h = frame_h
        patch_w = obj_w + gap + frame_w
        frame_r0 = 0
        frame_c0 = obj_w + gap if direction == "left" else 0
        obj_r0 = 1 + offset
        obj_c0 = 0 if direction == "left" else frame_w + gap
        target_r = None
        target_c = frame_c0 + 1 if direction == "left" else frame_c0 + frame_w - 2

    inp = np.full((patch_h, patch_w), BG, dtype=int)
    _draw_frame(inp, frame_r0, frame_c0, frame_h, frame_w, color)

    obj_coords = np.argwhere(obj_mask)
    inp[obj_r0 + obj_coords[:, 0], obj_c0 + obj_coords[:, 1]] = color

    out = inp.copy()
    out[obj_r0 + obj_coords[:, 0], obj_c0 + obj_coords[:, 1]] = BG
    if target_r is not None:
        cols = obj_c0 + np.flatnonzero(obj_mask.any(axis=0))
        out[target_r, cols] = color
    else:
        rows = obj_r0 + np.flatnonzero(obj_mask.any(axis=1))
        out[rows, target_c] = color

    return inp, out


def _canvas_choices(patches: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[int, int]]:
    min_h = max(patch.shape[0] + 2 for patch, _ in patches)
    min_w = max(patch.shape[1] + 2 for patch, _ in patches)
    reserve_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch, _ in patches)

    choices = []
    for h in range(max(GRID_MIN, min_h), GRID_MAX + 1):
        for w in range(max(GRID_MIN, min_w), GRID_MAX + 1):
            if h * w >= reserve_area:
                choices.append((h, w))
    return choices


def _place_patches(
    rng: np.random.Generator,
    shape: tuple[int, int],
    patches: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    reserve = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1],
        reverse=True,
    )
    for idx in order:
        patch = patches[idx][0]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                if not reserve[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        choice = candidates[int(rng.integers(len(candidates)))]
        positions[idx] = choice
        r0, c0 = choice
        reserve[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for idx, (inp_patch, out_patch) in enumerate(patches):
        r0, c0 = positions[idx]
        assert r0 is not None and c0 is not None
        _paste_non_bg(inp, inp_patch, r0, c0)
        _paste_non_bg(out, out_patch, r0, c0)
    return inp, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX + 1))
    colors = [int(v) for v in rng.choice(PALETTE, size=pair_count, replace=False)]
    directions = [str(v) for v in rng.choice(np.array(DIRECTIONS), size=pair_count, replace=False)]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color, direction in zip(colors, directions):
        pair = _render_pair(rng, color, direction)
        if pair is None:
            return None
        patches.append(pair)

    canvas_options = _canvas_choices(patches)
    if not canvas_options:
        return None

    order = rng.permutation(len(canvas_options))
    for idx in order[: min(len(canvas_options), 40)]:
        placed = _place_patches(rng, canvas_options[int(idx)], patches)
        if placed is None:
            continue
        inp, out = placed
        if np.array_equal(main(inp), out):
            return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
