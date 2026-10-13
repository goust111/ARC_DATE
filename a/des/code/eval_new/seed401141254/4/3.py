from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SUBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_SIDES = ("top", "bottom", "left", "right")
ALL_DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 300


ARROW_PATCHES = {
    "up": np.array(
        [
            [GREY, GREY, GREY],
            [BG, GREY, BG],
            [BG, GREY, BG],
            [BG, GREY, BG],
        ],
        dtype=int,
    ),
    "down": np.array(
        [
            [BG, GREY, BG],
            [BG, GREY, BG],
            [BG, GREY, BG],
            [GREY, GREY, GREY],
        ],
        dtype=int,
    ),
    "left": np.array(
        [
            [GREY, BG, BG, BG],
            [GREY, GREY, GREY, GREY],
            [GREY, BG, BG, BG],
        ],
        dtype=int,
    ),
    "right": np.array(
        [
            [BG, BG, BG, GREY],
            [GREY, GREY, GREY, GREY],
            [BG, BG, BG, GREY],
        ],
        dtype=int,
    ),
}

ARROW_SIGNATURES = {
    direction: frozenset(map(tuple, np.argwhere(patch == GREY)))
    for direction, patch in ARROW_PATCHES.items()
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    assert len(coords) > 0, "expected some foreground pixels"
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _detect_arrow_direction(coords: np.ndarray) -> str:
    r0, c0, _r1, _c1 = _bbox(coords)
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    matches = [direction for direction, signature in ARROW_SIGNATURES.items() if norm == signature]
    assert len(matches) == 1, "grey component must be one of the four arrow templates"
    return matches[0]


def _shrunken_patches(subject_patch: np.ndarray, bg: int, repeat_count: int) -> list[np.ndarray]:
    h, w = subject_patch.shape
    patches: list[np.ndarray] = []
    for layer in range(repeat_count):
        if 2 * layer >= h or 2 * layer >= w:
            break
        patch = subject_patch[layer : h - layer, layer : w - layer]
        if patch.size == 0 or not np.any(patch != bg):
            break
        patches.append(np.array(patch, copy=True))
    assert patches, "at least one propagated copy must survive"
    return patches


def _stack_patches(patches: list[np.ndarray], direction: str, bg: int) -> np.ndarray:
    if direction in ("left", "right"):
        out_h = max(int(p.shape[0]) for p in patches)
        out_w = sum(int(p.shape[1]) for p in patches) + len(patches) - 1
        out = np.full((out_h, out_w), bg, dtype=int)
        if direction == "right":
            cursor = 0
            for patch in patches:
                ph, pw = patch.shape
                r = (out_h - int(ph)) // 2
                out[r : r + ph, cursor : cursor + pw] = patch
                cursor += int(pw) + 1
        else:
            cursor = out_w
            for patch in patches:
                ph, pw = patch.shape
                cursor -= int(pw)
                r = (out_h - int(ph)) // 2
                out[r : r + ph, cursor : cursor + pw] = patch
                cursor -= 1
        return _crop_non_bg(out, bg)

    out_h = sum(int(p.shape[0]) for p in patches) + len(patches) - 1
    out_w = max(int(p.shape[1]) for p in patches)
    out = np.full((out_h, out_w), bg, dtype=int)
    if direction == "down":
        cursor = 0
        for patch in patches:
            ph, pw = patch.shape
            c = (out_w - int(pw)) // 2
            out[cursor : cursor + ph, c : c + pw] = patch
            cursor += int(ph) + 1
    else:
        cursor = out_h
        for patch in patches:
            ph, pw = patch.shape
            cursor -= int(ph)
            c = (out_w - int(pw)) // 2
            out[cursor : cursor + ph, c : c + pw] = patch
            cursor -= 1
    return _crop_non_bg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the seed pattern along the grey arrow direction while cropping one outer layer from each later copy.

    Input structure:
    - The background is black. One connected multicolor subject occupies a compact 4x4 to 6x6 style bbox.
    - A single grey 6-pixel arrow component sits outside the subject on one side; its template points up, down, left, or right.
    - A separate maroon straight bar of length 1 to 4 sits outside the subject on another side and encodes the repeat count.

    Rule:
    - Detect the arrow's pointing direction and count the number of maroon pixels.
    - Crop the subject to its tight bounding box, then produce repeated copies along the arrow direction.
    - Copy 1 uses the full subject crop; copy 2 removes the outermost layer from all four bbox sides; each later copy removes one more layer.
    - Stop when a crop would have zero extent or no foreground, place one black cell of spacing between consecutive copies, and center smaller copies on the perpendicular axis.

    Key insight:
    - The grey arrow gives only the propagation axis, the maroon bar gives only the repeat count, and the apparent perspective comes from repeatedly peeling one bbox layer off the original subject before each next copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())

    arrow_parts = _components(g == GREY)
    bar_parts = _components(g == MAROON)
    assert len(arrow_parts) == 1, "expected one grey arrow component"
    assert len(bar_parts) == 1, "expected one maroon count bar"

    direction = _detect_arrow_direction(arrow_parts[0])
    repeat_count = int(len(bar_parts[0]))
    assert 1 <= repeat_count <= 4

    subject_mask = (g != bg) & (g != GREY) & (g != MAROON)
    subject_parts = _components(subject_mask)
    assert len(subject_parts) == 1, "expected exactly one non-marker subject component"

    r0, c0, r1, c1 = _bbox(subject_parts[0])
    subject_patch = np.array(g[r0 : r1 + 1, c0 : c1 + 1], copy=True)
    patches = _shrunken_patches(subject_patch, bg, repeat_count)
    out = _stack_patches(patches, direction, bg)

    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    view[patch != BG] = patch[patch != BG]


def _make_subject_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    variant = random.choice(("solid", "one_corner", "two_corners", "edge_nibble", "corner_step"))

    if variant == "one_corner":
        r, c = random.choice(((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
        mask[r, c] = False
    elif variant == "two_corners":
        for r, c in random.choice(
            (((0, 0), (h - 1, w - 1)), ((0, w - 1), (h - 1, 0)), ((0, 0), (0, w - 1)), ((h - 1, 0), (h - 1, w - 1)))
        ):
            mask[r, c] = False
    elif variant == "edge_nibble":
        if random.random() < 0.5:
            r = random.choice((0, h - 1))
            start = random.randint(1, max(1, w - 2))
            end = min(w - 1, start + random.randint(0, 1))
            mask[r, start : end + 1] = False
        else:
            c = random.choice((0, w - 1))
            start = random.randint(1, max(1, h - 2))
            end = min(h - 1, start + random.randint(0, 1))
            mask[start : end + 1, c] = False
    elif variant == "corner_step":
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            mask[0, 0] = False
            if w >= 5:
                mask[0, 1] = False
        elif corner == "tr":
            mask[0, -1] = False
            if w >= 5:
                mask[0, -2] = False
        elif corner == "bl":
            mask[-1, 0] = False
            if w >= 5:
                mask[-1, 1] = False
        else:
            mask[-1, -1] = False
            if w >= 5:
                mask[-1, -2] = False

    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        mask[:] = True
    if len(_components(mask)) != 1:
        mask[:] = True
    return mask


def _pattern_index(mode: str, r: int, c: int, h: int, w: int) -> int:
    if mode == "solid":
        return 0
    if mode == "rows":
        return r
    if mode == "cols":
        return c
    if mode == "checker":
        return r + c
    if mode == "diag":
        return r + 2 * c
    if mode == "rings":
        return min(r, c, h - 1 - r, w - 1 - c)
    return (r >= h // 2) + 2 * (c >= w // 2)


def _make_subject_patch() -> np.ndarray:
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    mask = _make_subject_mask(h, w)
    color_count = random.randint(1, 3)
    colors = random.sample(list(SUBJECT_COLORS), color_count)
    mode = random.choice(("solid", "rows", "cols", "checker", "diag", "rings", "quadrants"))

    patch = np.full((h, w), BG, dtype=int)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            idx = _pattern_index(mode, r, c, h, w) % color_count
            patch[r, c] = int(colors[idx])
    return patch


def _line_patch(length: int, side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.full((1, length), MAROON, dtype=int)
    return np.full((length, 1), MAROON, dtype=int)


def _side_position(container_h: int, container_w: int, patch_h: int, patch_w: int, side: str) -> tuple[int, int]:
    if side == "top":
        left = random.randint(0, container_w - patch_w)
        return -patch_h - 1, left
    if side == "bottom":
        left = random.randint(0, container_w - patch_w)
        return container_h + 1, left
    if side == "left":
        top = random.randint(0, container_h - patch_h)
        return top, -patch_w - 1
    top = random.randint(0, container_h - patch_h)
    return top, container_w + 1


def _compose_input(subject: np.ndarray, direction: str, arrow_side: str, bar_side: str, repeat_count: int) -> np.ndarray:
    arrow = ARROW_PATCHES[direction]
    bar = _line_patch(repeat_count, bar_side)
    h, w = subject.shape
    ah, aw = arrow.shape
    bh, bw = bar.shape

    ar, ac = _side_position(h, w, ah, aw, arrow_side)
    br, bc = _side_position(h, w, bh, bw, bar_side)

    min_r = min(0, ar, br)
    min_c = min(0, ac, bc)
    max_r = max(h - 1, ar + ah - 1, br + bh - 1)
    max_c = max(w - 1, ac + aw - 1, bc + bw - 1)
    scene_h = max_r - min_r + 1
    scene_w = max_c - min_c + 1

    extra_h = random.randint(0, min(12, 30 - scene_h))
    extra_w = random.randint(0, min(12, 30 - scene_w))
    top_pad = random.randint(0, extra_h)
    left_pad = random.randint(0, extra_w)

    grid = np.full((scene_h + extra_h, scene_w + extra_w), BG, dtype=int)
    shift_r = top_pad - min_r
    shift_c = left_pad - min_c
    _paint_patch(grid, subject, shift_r, shift_c)
    _paint_patch(grid, arrow, shift_r + ar, shift_c + ac)
    _paint_patch(grid, bar, shift_r + br, shift_c + bc)
    return grid


def _gen_attempt() -> np.ndarray | None:
    subject = _make_subject_patch()
    direction = random.choice(ALL_DIRECTIONS)
    arrow_side = random.choice(ALL_SIDES)
    bar_side = random.choice([side for side in ALL_SIDES if side != arrow_side])
    repeat_count = random.randint(1, 4)

    grid = _compose_input(subject, direction, arrow_side, bar_side, repeat_count)

    try:
        out = main(grid)
    except Exception:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(_components((grid != BG) & (grid != GREY) & (grid != MAROON))) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
