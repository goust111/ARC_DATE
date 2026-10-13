import random
from collections import deque

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


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
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_frame_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = mask[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if np.array_equal(sub, expected):
            frames.append((r0, c0, r1, c1))
    return frames


def _find_frame_color_and_boxes(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        frames = _rect_frame_components(grid == color)
        if frames:
            total_bbox_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in frames)
            candidates.append((len(frames), total_bbox_area, color, frames))
    assert candidates, "expected at least one frame color"
    candidates.sort(reverse=True)
    count, _, color, frames = candidates[0]
    assert count >= 2, "expected multiple frame objects"
    return int(color), frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each empty rectangular frame using the legend template whose interior size matches.

    Input structure:
    - Background is a uniform color.
    - One corner contains a legend made of framed examples; each example frame has a colored interior pattern.
    - Elsewhere there are empty frames of the same border color. Their interiors are background-only.

    Rule:
    - Detect all one-pixel-thick rectangular frames of the common frame color.
    - Use every non-empty frame as a template keyed by its interior shape.
    - For each empty frame, copy the matching template interior into it.
    - Keep the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    _, frames = _find_frame_color_and_boxes(g, bg)

    templates: dict[tuple[int, int], np.ndarray] = {}
    empty_frames: list[tuple[int, int, int, int]] = []

    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        key = tuple(map(int, interior.shape))
        if np.any(interior != bg):
            assert key not in templates, f"duplicate template size {key}"
            templates[key] = interior.copy()
        else:
            empty_frames.append((r0, c0, r1, c1))

    assert templates, "expected at least one legend frame"
    assert empty_frames, "expected at least one empty frame"

    out = g.copy()
    for r0, c0, r1, c1 in empty_frames:
        key = (r1 - r0 - 1, c1 - c0 - 1)
        assert key in templates, f"missing template for size {key}"
        out[r0 + 1 : r1, c0 + 1 : c1] = templates[key]
    return out


# --------------- generator ---------------

BG = 7
FRAME = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 8, 9)
GRID_MIN = 20
GRID_MAX = 36
INNER_DIM_RANGE = (1, 5)
TEMPLATE_COUNT_RANGE = (2, 4)
EXTRA_TARGET_RANGE = (0, 3)
LEGEND_GAP = 1
TARGET_PAD = 1
MAX_GENERATION_ATTEMPTS = 300


def _size_pool() -> list[tuple[int, int]]:
    lo, hi = INNER_DIM_RANGE
    return [
        (h, w)
        for h in range(lo, hi + 1)
        for w in range(lo, hi + 1)
        if h * w <= 16 and h + w <= 8
    ]


def _pattern_colors(k: int) -> list[int]:
    colors = list(PATTERN_COLORS)
    random.shuffle(colors)
    return colors[:k]


def _make_pattern(h: int, w: int) -> np.ndarray:
    colors = _pattern_colors(random.randint(2, min(4, len(PATTERN_COLORS))))
    style = random.randrange(6)
    out = np.full((h, w), BG, dtype=int)

    if style == 0:
        for r in range(h):
            out[r, :] = colors[r % len(colors)]
        if h > 1 and w > 1:
            out[::2, ::2] = BG
    elif style == 1:
        for c in range(w):
            out[:, c] = colors[c % len(colors)]
        if h > 1 and w > 1:
            out[1::2, 1::2] = BG
    elif style == 2:
        for r in range(h):
            for c in range(w):
                if (r + c) % 2 == 0:
                    out[r, c] = colors[(r + c) % len(colors)]
    elif style == 3:
        base = colors[0]
        out[:, :] = base
        for r in range(h):
            for c in range(w):
                if r == c or r + c == w - 1:
                    out[r, c] = colors[(r + c) % len(colors)]
        if h > 2 and w > 2:
            out[1:-1, 1:-1] = np.where(
                ((np.indices((h - 2, w - 2)).sum(axis=0) + 1) % 3) == 0,
                BG,
                out[1:-1, 1:-1],
            )
    elif style == 4:
        rmid = h // 2
        cmid = w // 2
        out[: rmid + 1, : cmid + 1] = colors[0]
        out[: rmid + 1, cmid:] = colors[1 % len(colors)]
        out[rmid:, : cmid + 1] = colors[2 % len(colors)]
        out[rmid:, cmid:] = colors[3 % len(colors)]
        if h > 1 and w > 1:
            out[::2, 1::2] = BG
    else:
        density = random.uniform(0.45, 0.75)
        for r in range(h):
            for c in range(w):
                if random.random() < density:
                    out[r, c] = colors[(2 * r + c) % len(colors)]

    if not np.any(out != BG):
        out[h // 2, w // 2] = colors[0]
    return out


def _frame_patch(pattern: np.ndarray) -> np.ndarray:
    h, w = pattern.shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    patch[1:-1, 1:-1] = pattern
    return patch


def _legend_positions(
    grid_h: int,
    grid_w: int,
    patch_shapes: list[tuple[int, int]],
    orientation: str,
    corner: str,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    positions: list[tuple[int, int]] = []

    if orientation == "h":
        total_w = sum(w for _, w in patch_shapes) + LEGEND_GAP * (len(patch_shapes) - 1)
        max_h = max(h for h, _ in patch_shapes)
        if total_w > grid_w or max_h > grid_h:
            return None
        if corner in ("tl", "bl"):
            c = 0
            for ph, pw in patch_shapes:
                r = 0 if corner == "tl" else grid_h - ph
                positions.append((r, c))
                c += pw + LEGEND_GAP
        else:
            c = grid_w
            for ph, pw in patch_shapes:
                c -= pw
                r = 0 if corner == "tr" else grid_h - ph
                positions.append((r, c))
                c -= LEGEND_GAP
    else:
        total_h = sum(h for h, _ in patch_shapes) + LEGEND_GAP * (len(patch_shapes) - 1)
        max_w = max(w for _, w in patch_shapes)
        if total_h > grid_h or max_w > grid_w:
            return None
        if corner in ("tl", "tr"):
            r = 0
            for ph, pw in patch_shapes:
                c = 0 if corner == "tl" else grid_w - pw
                positions.append((r, c))
                r += ph + LEGEND_GAP
        else:
            r = grid_h
            for ph, pw in patch_shapes:
                r -= ph
                c = 0 if corner == "bl" else grid_w - pw
                positions.append((r, c))
                r -= LEGEND_GAP

    r0 = min(r for r, _ in positions)
    c0 = min(c for _, c in positions)
    r1 = max(r + ph - 1 for (r, c), (ph, pw) in zip(positions, patch_shapes))
    c1 = max(c + pw - 1 for (r, c), (ph, pw) in zip(positions, patch_shapes))
    return positions, (r0, c0, r1, c1)


def _reserve(mask: np.ndarray, box: tuple[int, int, int, int], pad: int) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc1 = min(w, c1 + pad + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _find_target_spot(mask: np.ndarray, box_h: int, box_w: int) -> tuple[int, int] | None:
    h, w = mask.shape
    candidates = []
    for r in range(1, h - box_h):
        for c in range(1, w - box_w):
            if np.any(mask[r - TARGET_PAD : r + box_h + TARGET_PAD, c - TARGET_PAD : c + box_w + TARGET_PAD]):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _draw_frame(grid: np.ndarray, r: int, c: int, inner_h: int, inner_w: int) -> None:
    box_h = inner_h + 2
    box_w = inner_w + 2
    grid[r : r + box_h, c : c + box_w] = BG
    grid[r, c : c + box_w] = FRAME
    grid[r + box_h - 1, c : c + box_w] = FRAME
    grid[r : r + box_h, c] = FRAME
    grid[r : r + box_h, c + box_w - 1] = FRAME


def _gen_attempt() -> np.ndarray | None:
    size_pool = _size_pool()
    n_templates = random.randint(*TEMPLATE_COUNT_RANGE)
    inner_sizes = random.sample(size_pool, n_templates)
    inner_sizes.sort(key=lambda hw: (hw[0] * hw[1], hw[0] + hw[1], hw))

    patterns = {size: _make_pattern(*size) for size in inner_sizes}
    legend_patches = [_frame_patch(patterns[size]) for size in inner_sizes]
    patch_shapes = [patch.shape for patch in legend_patches]

    target_sizes = list(inner_sizes)
    for _ in range(random.randint(*EXTRA_TARGET_RANGE)):
        target_sizes.append(random.choice(inner_sizes))
    random.shuffle(target_sizes)

    max_target_h = max(h + 2 for h, _ in target_sizes)
    max_target_w = max(w + 2 for _, w in target_sizes)
    approx_area = sum((h + 4) * (w + 4) for h, w in target_sizes)

    orientation = random.choice(("h", "v"))
    corner = random.choice(("tl", "tr", "bl", "br"))

    legend_h = max(ph for ph, _ in patch_shapes) if orientation == "h" else sum(
        ph for ph, _ in patch_shapes
    ) + LEGEND_GAP * (len(patch_shapes) - 1)
    legend_w = sum(pw for _, pw in patch_shapes) + LEGEND_GAP * (len(patch_shapes) - 1) if orientation == "h" else max(
        pw for _, pw in patch_shapes
    )
    side_hint = int(np.ceil(np.sqrt(approx_area))) + 4

    min_h = max(GRID_MIN, legend_h + 2, max_target_h + 3, side_hint)
    min_w = max(GRID_MIN, legend_w + 2, max_target_w + 3, side_hint)
    if orientation == "h":
        min_h = max(min_h, legend_h + max_target_h + 3)
    else:
        min_w = max(min_w, legend_w + max_target_w + 3)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = random.randint(min_h, GRID_MAX)
    grid_w = random.randint(min_w, GRID_MAX)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    legend_layout = _legend_positions(grid_h, grid_w, patch_shapes, orientation, corner)
    if legend_layout is None:
        return None
    positions, legend_box = legend_layout

    for patch, (r, c) in zip(legend_patches, positions):
        ph, pw = patch.shape
        grid[r : r + ph, c : c + pw] = patch

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _reserve(occupied, legend_box, pad=1)

    for inner_h, inner_w in sorted(target_sizes, key=lambda hw: hw[0] * hw[1], reverse=True):
        box_h = inner_h + 2
        box_w = inner_w + 2
        spot = _find_target_spot(occupied, box_h, box_w)
        if spot is None:
            return None
        r, c = spot
        _draw_frame(grid, r, c, inner_h, inner_w)
        _reserve(occupied, (r, c, r + box_h - 1, c + box_w - 1), pad=1)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
