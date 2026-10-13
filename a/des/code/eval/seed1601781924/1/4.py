from __future__ import annotations

import random

import numpy as np


GAP_TO_TURNS = {
    "top": 0,
    "right": 1,
    "bottom": 2,
    "left": 3,
}


BG_COLOR = 0
SEED_COLOR_CANDIDATES = (6, 7, 8, 9)
FRAME_COLOR_CANDIDATES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
FRAME_COUNT_RANGE = (2, 5)
INNER_FRAME_SIDE_RANGE = (6, 10)
FRAME_GAP_CHOICES = (2, 3)
FRAME_INSET_CHOICES = (2, 3)
SEED_SIDE_CHOICES = ("top", "right", "bottom", "left")
SEED_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))
BASE_MARGIN_RANGE = (1, 4)
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                }
            )
    return comps


def _frame_gap_side(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(mask & ~border):
        return None

    sides = {
        "top": np.flatnonzero(~mask[0, :]),
        "right": np.flatnonzero(~mask[:, -1]),
        "bottom": np.flatnonzero(~mask[-1, :]),
        "left": np.flatnonzero(~mask[:, 0]),
    }

    gap_side: str | None = None
    gap_len = 0
    for side, missing in sides.items():
        if missing.size == 0:
            continue
        if gap_side is not None:
            return None
        if int(missing.size) not in FRAME_GAP_CHOICES:
            return None
        if int(missing[0]) == 0 or int(missing[-1]) == (mask.shape[1] - 1 if side in ("top", "bottom") else mask.shape[0] - 1):
            return None
        if not np.all(np.diff(missing) == 1):
            return None
        gap_side = side
        gap_len = int(missing.size)

    if gap_side is None:
        return None
    if int(mask.sum()) != int(border.sum()) - gap_len:
        return None
    return gap_side


def _boxes_strictly_nested(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _same_center(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] + a[2] == b[0] + b[2] and a[1] + a[3] == b[1] + b[3]


def _side_adjacent(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    vertical = (ar1 + 1 == br0 or br1 + 1 == ar0) and not (ac1 < bc0 or bc1 < ac0)
    horizontal = (ac1 + 1 == bc0 or bc1 + 1 == ac0) and not (ar1 < br0 or br1 < ar0)
    return vertical or horizontal


def _rotate_cw(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, -int(turns) % 4)


def _stamp_non_bg(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    region = out[top : top + h, left : left + w]
    mask = patch != int(bg)
    region[mask] = patch[mask]


def _center_position(
    bbox: tuple[int, int, int, int],
    patch_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = patch_shape
    assert ph <= inner_h and pw <= inner_w
    top = r0 + 1 + (inner_h - ph) // 2
    left = c0 + 1 + (inner_w - pw) // 2
    return top, left


def _extract_scene(g: np.ndarray) -> tuple[int, list[dict], dict]:
    bg = _get_bg(g)
    comps = _components(g, bg)
    assert comps

    frames: list[dict] = []
    others: list[dict] = []
    for comp in comps:
        mask = comp["patch"] != bg
        gap_side = _frame_gap_side(mask)
        if gap_side is None:
            others.append(comp)
        else:
            frame = dict(comp)
            frame["gap_side"] = gap_side
            frames.append(frame)

    assert frames, "expected at least one frame"
    frames.sort(
        key=lambda comp: (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1),
        reverse=True,
    )
    for outer, inner in zip(frames, frames[1:]):
        assert _boxes_strictly_nested(outer["bbox"], inner["bbox"])
        assert _same_center(outer["bbox"], inner["bbox"])
        assert inner["bbox"][0] - outer["bbox"][0] >= 2
        assert inner["bbox"][1] - outer["bbox"][1] >= 2

    assert len(others) == 1, "expected exactly one external seed sprite"
    seed = others[0]
    assert _side_adjacent(seed["bbox"], frames[0]["bbox"]), "seed must sit just outside the outer frame"
    return bg, frames, seed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the gap side on each nested rectangular frame, accumulate clockwise
    rotations from the outside in, and stamp the rotated seed pattern at the
    center of the innermost frame interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, frames, seed = _extract_scene(g)
    turns = sum(GAP_TO_TURNS[frame["gap_side"]] for frame in frames) % 4
    rotated = _rotate_cw(seed["patch"], turns)

    out = g.copy()
    top, left = _center_position(frames[-1]["bbox"], rotated.shape)
    _stamp_non_bg(out, rotated, top, left, bg)
    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    cells = random.randint(max(4, (h * w) // 2), h * w - 1)
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = {start}

    while int(mask.sum()) < cells:
        base = random.choice(tuple(frontier))
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = base[0] + dr
            nc = base[1] + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if nbrs:
            choice = random.choice(nbrs)
            mask[choice] = True
            frontier.add(choice)
        else:
            frontier.remove(base)
            if not frontier:
                frontier = {tuple(cell) for cell in np.argwhere(mask)}
    return mask


def _seed_patch(bg: int, color: int) -> np.ndarray | None:
    h, w = random.choice(SEED_SHAPES)
    for _ in range(80):
        mask = _random_connected_mask(h, w)
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        patch = np.full((h, w), bg, dtype=int)
        patch[mask] = color
        if _frame_gap_side(mask) is not None:
            continue
        if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            continue
        return patch
    return None


def _gap_segment(length: int) -> tuple[int, int]:
    gap_len = random.choice([g for g in FRAME_GAP_CHOICES if g <= length - 2])
    start = random.randint(1, length - gap_len - 1)
    return start, gap_len


def _frame_bboxes(
    outer_top: int,
    outer_left: int,
    outer_h: int,
    outer_w: int,
    insets: list[int],
) -> list[tuple[int, int, int, int]]:
    bboxes = []
    offset = 0
    for idx in range(len(insets) + 1):
        r0 = outer_top + offset
        c0 = outer_left + offset
        r1 = outer_top + outer_h - 1 - offset
        c1 = outer_left + outer_w - 1 - offset
        bboxes.append((r0, c0, r1, c1))
        if idx < len(insets):
            offset += insets[idx]
    return bboxes


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    gap_side: str,
    gap_start: int,
    gap_len: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if gap_side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_len] = bg
    elif gap_side == "right":
        grid[r0 + gap_start : r0 + gap_start + gap_len, c1] = bg
    elif gap_side == "bottom":
        grid[r1, c0 + gap_start : c0 + gap_start + gap_len] = bg
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_len, c0] = bg


def _place_seed(
    patch: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = outer_bbox
    ph, pw = patch.shape
    if side == "top":
        return r0 - ph, random.randint(c0, c1 - pw + 1)
    if side == "bottom":
        return r1 + 1, random.randint(c0, c1 - pw + 1)
    if side == "left":
        return random.randint(r0, r1 - ph + 1), c0 - pw
    return random.randint(r0, r1 - ph + 1), c1 + 1


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[bg]) > int(max(counts[color] for color in range(len(counts)) if color != bg))


def _expected_output(
    grid: np.ndarray,
    inner_bbox: tuple[int, int, int, int],
    seed_patch: np.ndarray,
    gap_sides: list[str],
    bg: int,
) -> np.ndarray:
    turns = sum(GAP_TO_TURNS[side] for side in gap_sides) % 4
    rotated = _rotate_cw(seed_patch, turns)
    out = grid.copy()
    top, left = _center_position(inner_bbox, rotated.shape)
    _stamp_non_bg(out, rotated, top, left, bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = BG_COLOR
    seed_color = random.choice(SEED_COLOR_CANDIDATES)
    frame_colors = random.sample([c for c in FRAME_COLOR_CANDIDATES if c != seed_color], random.randint(*FRAME_COUNT_RANGE))
    frame_count = len(frame_colors)

    seed_patch = _seed_patch(bg, seed_color)
    if seed_patch is None:
        return None
    seed_h, seed_w = seed_patch.shape

    min_inner = max(seed_h, seed_w) + 2
    inner_h = random.randint(min_inner, INNER_FRAME_SIDE_RANGE[1])
    inner_w = random.randint(min_inner, INNER_FRAME_SIDE_RANGE[1])
    if inner_h < INNER_FRAME_SIDE_RANGE[0] or inner_w < INNER_FRAME_SIDE_RANGE[0]:
        return None

    insets = [random.choice(FRAME_INSET_CHOICES) for _ in range(frame_count - 1)]
    outer_h = inner_h + 2 * sum(insets)
    outer_w = inner_w + 2 * sum(insets)

    seed_side = random.choice(SEED_SIDE_CHOICES)
    top_extra = seed_h if seed_side == "top" else 0
    bottom_extra = seed_h if seed_side == "bottom" else 0
    left_extra = seed_w if seed_side == "left" else 0
    right_extra = seed_w if seed_side == "right" else 0

    top_margin = random.randint(*BASE_MARGIN_RANGE) + top_extra
    bottom_margin = random.randint(*BASE_MARGIN_RANGE) + bottom_extra
    left_margin = random.randint(*BASE_MARGIN_RANGE) + left_extra
    right_margin = random.randint(*BASE_MARGIN_RANGE) + right_extra

    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin
    if grid_h > MAX_GRID_SIDE or grid_w > MAX_GRID_SIDE:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    bboxes = _frame_bboxes(top_margin, left_margin, outer_h, outer_w, insets)
    gap_sides: list[str] = []

    for color, bbox in zip(frame_colors, bboxes):
        r0, c0, r1, c1 = bbox
        side = random.choice(SEED_SIDE_CHOICES)
        length = c1 - c0 + 1 if side in ("top", "bottom") else r1 - r0 + 1
        gap_start, gap_len = _gap_segment(length)
        _draw_frame(grid, bbox, color, side, gap_start, gap_len, bg)
        gap_sides.append(side)

    seed_r0, seed_c0 = _place_seed(seed_patch, bboxes[0], seed_side)
    if seed_r0 < 0 or seed_c0 < 0 or seed_r0 + seed_h > grid_h or seed_c0 + seed_w > grid_w:
        return None
    _stamp_non_bg(grid, seed_patch, seed_r0, seed_c0, bg)

    if not _background_is_dominant(grid, bg):
        return None

    expected = _expected_output(grid, bboxes[-1], seed_patch, gap_sides, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-frame rotation input")
