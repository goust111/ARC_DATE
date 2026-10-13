from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_CORNER_ROLES = {
    ((0, 0), (0, 1), (1, 0)): "TL",
    ((0, 0), (0, 1), (1, 1)): "TR",
    ((0, 0), (1, 0), (1, 1)): "BL",
    ((0, 1), (1, 0), (1, 1)): "BR",
}

_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NON_BG_COLORS = tuple(range(1, 10))
_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _monochrome_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
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
                coords.append((int(rr), int(cc)))
                for dr, dc in _DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_anchor(role: str, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if role == "TL":
        return int(r0), int(c0)
    if role == "TR":
        return int(r0), int(c1)
    if role == "BL":
        return int(r1), int(c0)
    return int(r1), int(c1)


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        corners: dict[str, tuple[int, int]] = {}
        for coords in _monochrome_components(grid, color):
            if int(coords.shape[0]) != 3:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if (r1 - r0, c1 - c0) != (1, 1):
                continue
            rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
            role = _CORNER_ROLES.get(rel)
            if role is None:
                continue
            anchor = _corner_anchor(role, (r0, c0, r1, c1))
            if role in corners:
                raise AssertionError("duplicate corner role for one frame color")
            corners[role] = anchor
        if set(corners) != {"TL", "TR", "BL", "BR"}:
            continue
        top, left = corners["TL"]
        top2, right = corners["TR"]
        bottom, left2 = corners["BL"]
        bottom2, right2 = corners["BR"]
        assert top == top2 and left == left2 and bottom == bottom2 and right == right2
        assert bottom - top >= 3 and right - left >= 3
        inner = grid[top + 1 : bottom, left + 1 : right].copy()
        frames.append(
            {
                "color": int(color),
                "outer_bbox": (int(top), int(left), int(bottom), int(right)),
                "inner": inner,
                "inner_shape": (int(inner.shape[0]), int(inner.shape[1])),
            }
        )
    assert frames
    return frames


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(patch, np.ones((int(factor), int(factor)), dtype=patch.dtype))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the corner-marked frames, pick the only small frame whose interior scales uniformly to the large empty frame, and output the enlarged interior pattern.

    Input structure:
    - A black background contains several rectangular frames indicated only by four same-color 2x2 L-corner triominoes.
    - One frame is a larger empty canvas, while every other smaller frame contains a multicolor interior pattern that may include background cells.
    - Frame edges are absent; only the four corners of each frame are drawn.

    Rule:
    - Reconstruct each frame from its four same-color corner markers and read its interior rectangle.
    - Identify the unique non-empty small frame whose interior height and width both scale to the large empty frame by the same integer factor.
    - Output the large frame's interior size, filled with the selected small interior pattern enlarged by that factor with per-pixel replication.

    Key insight:
    - The corners define frame geometry, and the correct source pattern is selected purely by matching the two interior dimensions with one shared integer scale factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frames = _find_frames(g, bg)

    empty_frames = [frame for frame in frames if np.all(frame["inner"] == bg)]
    assert len(empty_frames) == 1
    canvas = empty_frames[0]
    target_h, target_w = canvas["inner_shape"]

    candidates: list[tuple[dict, int]] = []
    for frame in frames:
        if frame is canvas:
            continue
        src_h, src_w = frame["inner_shape"]
        if src_h == 0 or src_w == 0:
            continue
        if target_h % src_h != 0 or target_w % src_w != 0:
            continue
        scale_h = target_h // src_h
        scale_w = target_w // src_w
        if scale_h == scale_w and scale_h >= 2:
            candidates.append((frame, int(scale_h)))

    assert len(candidates) == 1
    source, factor = candidates[0]
    out = _scale_patch(source["inner"], factor)
    assert out.shape == (target_h, target_w)
    return out


def _paint_frame(interior: np.ndarray, marker_color: int) -> np.ndarray:
    ih, iw = interior.shape
    patch = np.zeros((ih + 2, iw + 2), dtype=int)
    patch[0, 0] = patch[0, 1] = patch[1, 0] = int(marker_color)
    patch[0, -2] = patch[0, -1] = patch[1, -1] = int(marker_color)
    patch[-2, 0] = patch[-1, 0] = patch[-1, 1] = int(marker_color)
    patch[-2, -1] = patch[-1, -2] = patch[-1, -1] = int(marker_color)
    patch[1:-1, 1:-1] = interior
    return patch


def _grow_blob(free: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    starts = np.argwhere(free)
    if starts.size == 0:
        return None
    sr, sc = random.choice(starts).tolist()
    cells = [(int(sr), int(sc))]
    used = {(int(sr), int(sc))}
    free[int(sr), int(sc)] = False
    while len(cells) < int(size):
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in _DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and free[nr, nc] and (nr, nc) not in used:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        cells.append((int(nr), int(nc)))
        used.add((int(nr), int(nc)))
        free[int(nr), int(nc)] = False
    return cells


def _has_l_triominio_component(patch: np.ndarray) -> bool:
    for color in sorted(set(map(int, patch.ravel())) - {0}):
        for coords in _monochrome_components(patch, color):
            if int(coords.shape[0]) != 3:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if (r1 - r0, c1 - c0) != (1, 1):
                continue
            rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
            if rel in _CORNER_ROLES:
                return True
    return False


def _make_pattern(height: int, width: int, palette: list[int]) -> np.ndarray | None:
    area = int(height * width)
    max_blobs = min(3, len(palette), area)
    for _ in range(80):
        patch = np.zeros((height, width), dtype=int)
        free = np.ones((height, width), dtype=bool)
        if max_blobs < 2:
            return None
        blob_count = random.randint(2, max_blobs)
        colors = random.sample(palette, blob_count)
        fill_low = max(2, blob_count, area // 4)
        fill_high = max(fill_low, (2 * area) // 3)
        if area >= 4:
            fill_high = min(fill_high, area - 1)
        else:
            fill_high = min(fill_high, area)
        if fill_low > fill_high:
            continue
        total_fill = random.randint(fill_low, fill_high)
        remaining = total_fill
        sizes: list[int] = []
        for i in range(blob_count):
            left = blob_count - i - 1
            min_here = 1
            max_here = remaining - left
            if i == blob_count - 1:
                size = remaining
            else:
                size = random.randint(min_here, max_here)
            sizes.append(size)
            remaining -= size
        random.shuffle(sizes)
        ok = True
        for color, size in zip(colors, sizes):
            cells = _grow_blob(free, size)
            if cells is None:
                ok = False
                break
            for r, c in cells:
                patch[r, c] = int(color)
        if not ok:
            continue
        non_bg = sorted(set(map(int, patch.ravel())) - {0})
        if len(non_bg) < 2:
            continue
        if area >= 4 and not np.any(patch == 0):
            continue
        if _has_l_triominio_component(patch):
            continue
        if np.array_equal(patch, np.flipud(patch)) and np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    return None


def _choose_target_dims() -> tuple[int, int, int, int, int]:
    for _ in range(200):
        scale = random.randint(2, 4)
        src_h = random.randint(3, 6)
        src_w = random.randint(3, 6)
        target_h = src_h * scale
        target_w = src_w * scale
        if target_h >= 6 and target_w >= 6 and target_h + 2 <= 22 and target_w + 2 <= 22:
            return src_h, src_w, target_h, target_w, scale
    raise RuntimeError("failed to choose target dimensions")


def _make_distractor_dims(target_h: int, target_w: int, used: set[tuple[int, int]]) -> tuple[int, int] | None:
    choices: list[tuple[int, int]] = []
    target_area = int(target_h * target_w)
    for h in range(3, min(7, target_h - 1) + 1):
        for w in range(3, min(7, target_w - 1) + 1):
            if (h, w) in used:
                continue
            if h * w >= target_area:
                continue
            if target_h % h == 0 and target_w % w == 0 and target_h // h == target_w // w:
                continue
            choices.append((h, w))
    return random.choice(choices) if choices else None


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = canvas[top : top + h, left : left + w]
    mask = patch != 0
    region[mask] = patch[mask]


def _place_frames(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    for _ in range(120):
        canvas_h = random.randint(max(max_h + 4, 16), 30)
        canvas_w = random.randint(max(max_w + 4, 16), 30)
        canvas = np.zeros((canvas_h, canvas_w), dtype=int)
        occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
        ok = True
        for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            ph, pw = patch.shape
            spots: list[tuple[int, int]] = []
            for _ in range(140):
                top = random.randint(0, canvas_h - ph)
                left = random.randint(0, canvas_w - pw)
                if _can_place(occupied, top, left, ph, pw, pad=1):
                    spots.append((top, left))
            if not spots:
                ok = False
                break
            top, left = random.choice(spots)
            occupied[top : top + ph, left : left + pw] = True
            _paste_patch(canvas, patch, top, left)
        if ok:
            return canvas
    return None


def _gen_attempt() -> np.ndarray | None:
    src_h, src_w, target_h, target_w, scale = _choose_target_dims()

    frame_count = random.randint(3, 5)
    frame_colors = random.sample(_NON_BG_COLORS, frame_count)
    pattern_palette = [color for color in _NON_BG_COLORS if color not in frame_colors]
    if len(pattern_palette) < 2:
        return None

    chosen_pattern = _make_pattern(src_h, src_w, pattern_palette)
    if chosen_pattern is None:
        return None
    expected = _scale_patch(chosen_pattern, scale)

    patches = [_paint_frame(np.zeros((target_h, target_w), dtype=int), frame_colors[0])]
    patches.append(_paint_frame(chosen_pattern, frame_colors[1]))

    used_dims = {(src_h, src_w)}
    for idx in range(2, frame_count):
        dims = _make_distractor_dims(target_h, target_w, used_dims)
        if dims is None:
            return None
        used_dims.add(dims)
        distractor = _make_pattern(dims[0], dims[1], pattern_palette)
        if distractor is None or np.all(distractor == 0):
            return None
        patches.append(_paint_frame(distractor, frame_colors[idx]))

    canvas = _place_frames(patches)
    if canvas is None:
        return None
    if max(canvas.shape) > 30:
        return None
    actual = main(canvas)
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 9")
