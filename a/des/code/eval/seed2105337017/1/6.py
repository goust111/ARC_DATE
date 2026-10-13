from __future__ import annotations

import random

import numpy as np


BG = 0
FENCE = 3
TAG_COLORS = tuple(color for color in range(10) if color not in (BG, FENCE))
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))

    return comps


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 3, "frame must enclose at least a 1x1 interior"

    expected = 2 * (h + w) - 4
    assert len(coords) == expected, "green component must be a one-pixel rectangular frame"
    for r0c, c0c in coords:
        r = int(r0c)
        c = int(c0c)
        assert r in (r0, r1) or c in (c0, c1), "green component must lie on its bbox perimeter"

    return r0, c0, r1, c1


def _tag_side(frame: tuple[int, int, int, int], r: int, c: int) -> str | None:
    r0, c0, r1, c1 = frame
    if r == r0 - 1 and c0 <= c <= c1:
        return "top"
    if r == r1 + 1 and c0 <= c <= c1:
        return "bottom"
    if c == c0 - 1 and r0 <= r <= r1:
        return "left"
    if c == c1 + 1 and r0 <= r <= r1:
        return "right"
    return None


def _balanced_sizes(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _partition_lines(axis_len: int, n_lines: int) -> list[int]:
    assert n_lines >= 1
    usable = axis_len - n_lines
    sizes = _balanced_sizes(usable, n_lines + 1)
    offsets: list[int] = []
    acc = 0
    for i in range(n_lines):
        acc += sizes[i]
        offsets.append(acc + i)
    return offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find green rectangular pastures and nearby same-color tag pixels, then draw
    equally spaced colored fence partitions inside each pasture.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = [_frame_bbox(comp) for comp in _components(g == FENCE)]
    assert frames, "expected at least one green frame"

    tags_by_frame: list[list[tuple[int, int, int, str]]] = [[] for _ in frames]
    for r0, c0 in np.argwhere((g != BG) & (g != FENCE)):
        r = int(r0)
        c = int(c0)
        color = int(g[r, c])
        matches: list[tuple[int, str]] = []
        for idx, frame in enumerate(frames):
            side = _tag_side(frame, r, c)
            if side is not None:
                matches.append((idx, side))
        assert len(matches) == 1, "each colored tag must belong to exactly one pasture"
        idx, side = matches[0]
        tags_by_frame[idx].append((r, c, color, side))

    out = g.copy()
    for frame, tags in zip(frames, tags_by_frame, strict=False):
        assert tags, "each pasture must have at least one tag"
        r0, c0, r1, c1 = frame
        colors = {color for _r, _c, color, _side in tags}
        assert len(colors) == 1, "all tags of one pasture must share a color"
        tag_color = colors.pop()
        n_tags = len(tags)
        assert 1 <= n_tags <= 4, "tag count must be between 1 and 4"

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.size > 0 and np.all(inner == BG), "pasture interior must start empty"

        side_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        for _r, _c, _color, side in tags:
            side_counts[side] += 1

        horizontal_weight = side_counts["top"] + side_counts["bottom"]
        vertical_weight = side_counts["left"] + side_counts["right"]
        draw_horizontal = horizontal_weight >= vertical_weight

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        axis_len = inner_h if draw_horizontal else inner_w
        assert axis_len >= 2 * n_tags + 1, "interior too small for the requested number of partitions"
        line_offsets = _partition_lines(axis_len, n_tags)

        if draw_horizontal:
            for offset in line_offsets:
                out[r0 + 1 + offset, c0 + 1 : c1] = tag_color
        else:
            for offset in line_offsets:
                out[r0 + 1 : r1, c0 + 1 + offset] = tag_color

    return out


def _split_total(total: int) -> tuple[int, int]:
    first = random.randint(0, total)
    return first, total - first


def _sample_mode(n_tags: int) -> str:
    modes = ["horizontal", "vertical"]
    if n_tags % 2 == 0:
        modes.append("tie")
    return random.choice(modes)


def _sample_side_counts(n_tags: int, mode: str) -> dict[str, int]:
    if mode == "horizontal":
        tb_total = random.choice([k for k in range(n_tags + 1) if k > n_tags - k])
        lr_total = n_tags - tb_total
    elif mode == "vertical":
        lr_total = random.choice([k for k in range(n_tags + 1) if k > n_tags - k])
        tb_total = n_tags - lr_total
    else:
        assert n_tags % 2 == 0
        tb_total = n_tags // 2
        lr_total = n_tags // 2

    top, bottom = _split_total(tb_total)
    left, right = _split_total(lr_total)
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _rand_range(lo: int, hi: int) -> int:
    assert lo <= hi
    return random.randint(lo, hi)


def _build_pasture_patch(color: int, n_tags: int, mode: str) -> np.ndarray:
    primary_min = 2 * n_tags + 1
    if mode == "vertical":
        inner_h = _rand_range(3, 8)
        inner_w = _rand_range(primary_min, 10)
    else:
        inner_h = _rand_range(primary_min, 10)
        inner_w = _rand_range(3, 8)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    patch = np.full((outer_h + 2, outer_w + 2), BG, dtype=int)

    patch[1, 1 : outer_w + 1] = FENCE
    patch[outer_h, 1 : outer_w + 1] = FENCE
    patch[1 : outer_h + 1, 1] = FENCE
    patch[1 : outer_h + 1, outer_w] = FENCE

    counts = _sample_side_counts(n_tags, mode)
    if counts["top"]:
        cols = random.sample(list(range(1, outer_w + 1)), counts["top"])
        patch[0, cols] = color
    if counts["bottom"]:
        cols = random.sample(list(range(1, outer_w + 1)), counts["bottom"])
        patch[outer_h + 1, cols] = color
    if counts["left"]:
        rows = random.sample(list(range(1, outer_h + 1)), counts["left"])
        patch[rows, 0] = color
    if counts["right"]:
        rows = random.sample(list(range(1, outer_h + 1)), counts["right"])
        patch[rows, outer_w + 1] = color

    return patch


def _reserve_slice(limit: int, start: int, length: int) -> slice:
    return slice(max(0, start - 1), min(limit, start + length + 1))


def _can_place(reserved: np.ndarray, r: int, c: int, patch: np.ndarray) -> bool:
    h, w = reserved.shape
    ph, pw = patch.shape
    if r + ph > h or c + pw > w:
        return False
    return not reserved[_reserve_slice(h, r, ph), _reserve_slice(w, c, pw)].any()


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, r: int, c: int) -> None:
    ph, pw = patch.shape
    sub = grid[r : r + ph, c : c + pw]
    mask = patch != BG
    sub[mask] = patch[mask]
    reserved[_reserve_slice(grid.shape[0], r, ph), _reserve_slice(grid.shape[1], c, pw)] = True


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    reserve_area = sum(int((p.shape[0] + 2) * (p.shape[1] + 2)) for p in patches)
    side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(reserve_area))) + 3)
    h_lo = max(18, side - 2)
    w_lo = max(18, side - 2)
    h_hi = min(36, side + 4)
    w_hi = min(36, side + 6)
    if h_lo > h_hi or w_lo > w_hi:
        return None

    grid = np.full((_rand_range(h_lo, h_hi), _rand_range(w_lo, w_hi)), BG, dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)

    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        positions = [(r, c) for r in range(grid.shape[0] - ph + 1) for c in range(grid.shape[1] - pw + 1)]
        random.shuffle(positions)
        placed = False
        for r, c in positions[:PLACEMENT_ATTEMPTS]:
            if _can_place(reserved, r, c, patch):
                _place_patch(grid, reserved, patch, r, c)
                placed = True
                break
        if not placed:
            for r, c in positions[PLACEMENT_ATTEMPTS:]:
                if _can_place(reserved, r, c, patch):
                    _place_patch(grid, reserved, patch, r, c)
                    placed = True
                    break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_pastures = random.randint(1, 4)
    colors = random.sample(TAG_COLORS, n_pastures)
    patches: list[np.ndarray] = []
    for color in colors:
        n_tags = random.randint(1, 4)
        mode = _sample_mode(n_tags)
        patches.append(_build_pasture_patch(color, n_tags, mode))

    grid = _pack_patches(patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pasture-tag puzzle")
