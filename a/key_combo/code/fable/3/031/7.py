from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _touches_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _find_template(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], np.ndarray, int]:
    candidates = []
    for color in sorted({int(v) for v in grid.flat if int(v) != bg}):
        for coords in _components(grid == color):
            bbox = _bbox(coords)
            if not _touches_corner(bbox, grid.shape):
                continue
            r0, c0, r1, c1 = bbox
            outer_h = r1 - r0 + 1
            outer_w = c1 - c0 + 1
            if outer_h != outer_w or outer_h < 4:
                continue

            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((outer_h, outer_w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(block[border] == color):
                continue
            if np.any(block[~border] == color):
                continue

            inner = block[1:-1, 1:-1]
            inner_colors = {int(v) for v in inner.flat}
            fg_colors = sorted(inner_colors - {bg})
            if len(fg_colors) != 1:
                continue
            fg = int(fg_colors[0])
            if not np.any(inner == fg) or not np.any(inner == bg):
                continue
            candidates.append((int(color), bbox, inner == fg, fg))

    assert len(candidates) == 1, "expected exactly one framed template"
    return candidates[0]


def _find_marker_squares(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int, frame_color: int
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    markers: list[tuple[int, int, int, int]] = []
    colors = sorted({int(v) for v in grid[outside].flat if int(v) not in {bg, frame_color}})
    for color in colors:
        for coords in _components((grid == color) & outside):
            box = _bbox(coords)
            rr0, cc0, rr1, cc1 = box
            side_r = rr1 - rr0 + 1
            side_c = cc1 - cc0 + 1
            if side_r != side_c:
                continue
            if len(coords) != side_r * side_c:
                continue
            markers.append((rr0, cc0, side_r, color))

    markers.sort()
    assert markers, "expected at least one marker square"
    return markers


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((factor, factor), dtype=np.uint8)).astype(bool)


def _paint_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = dst.shape
    mh, mw = mask.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r0) + mh)
    cc1 = min(w, int(c0) + mw)
    if rr0 >= rr1 or cc0 >= cc1:
        return

    mr0 = rr0 - int(r0)
    mc0 = cc0 - int(c0)
    mr1 = mr0 + (rr1 - rr0)
    mc1 = mc0 + (cc1 - cc0)
    submask = mask[mr0:mr1, mc0:mc1]
    view = dst[rr0:rr1, cc0:cc1]
    view[submask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a corner template onto solid marker squares, scaling by square size and inverting by marker color.

    Input structure:
    - A grey border in one corner encloses an m x m template made of one foreground color F and black background.
    - The main area contains several solid k x k marker squares.
    - Marker squares use either the template color F or one other color G.

    Rule:
    - Keep the framed template unchanged.
    - Replace each marker square by an m-scaled stamp whose top-left corner matches the square's top-left corner.
    - A marker of color F paints the template's F cells in color F; a marker of color G paints the template's black cells in color G. Out-of-bounds parts are clipped.

    Key insight:
    - The marker side length and marker color are two independent control channels: size sets the scale, and color selects positive versus negative polarity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    frame_color, frame_bbox, template_fg, fg_color = _find_template(g, bg)
    markers = _find_marker_squares(g, frame_bbox, bg, frame_color)

    out = g.copy()
    for r0, c0, side, _color in markers:
        out[r0 : r0 + side, c0 : c0 + side] = bg

    for r0, c0, side, color in markers:
        stamp_mask = template_fg if color == fg_color else ~template_fg
        _paint_mask(out, _scale_mask(stamp_mask, side), r0, c0, fg_color if color == fg_color else color)

    return out


def _corner_bbox(corner: str, height: int, width: int, outer: int) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, outer - 1, outer - 1
    if corner == "tr":
        return 0, width - outer, outer - 1, width - 1
    if corner == "bl":
        return height - outer, 0, height - 1, outer - 1
    return height - outer, width - outer, height - 1, width - 1


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sample_template_mask(size: int) -> np.ndarray:
    while True:
        density = random.uniform(0.3, 0.7)
        mask = np.random.random((size, size)) < density
        fg_count = int(mask.sum())
        if not (2 <= fg_count <= size * size - 2):
            continue
        fg_rows = int(np.unique(np.argwhere(mask)[:, 0]).size)
        fg_cols = int(np.unique(np.argwhere(mask)[:, 1]).size)
        bg_rows = int(np.unique(np.argwhere(~mask)[:, 0]).size)
        bg_cols = int(np.unique(np.argwhere(~mask)[:, 1]).size)
        if min(fg_rows, fg_cols, bg_rows, bg_cols) < 2:
            continue
        return mask


def _visible_paint_count(mask: np.ndarray, r0: int, c0: int, height: int, width: int) -> int:
    mh, mw = mask.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(height, r0 + mh)
    cc1 = min(width, c0 + mw)
    if rr0 >= rr1 or cc0 >= cc1:
        return 0
    return int(mask[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0].sum())


def _try_place_markers(
    height: int,
    width: int,
    frame_bbox: tuple[int, int, int, int],
    template_fg: np.ndarray,
    fg_color: int,
    neg_color: int,
) -> list[tuple[int, int, int, int]] | None:
    n_markers = random.randint(2, 4)

    while True:
        scales = random.choices([1, 2, 3, 4], weights=[2, 4, 3, 1], k=n_markers)
        if len(set(scales)) >= 2:
            break

    marker_colors = [fg_color, neg_color]
    for _ in range(n_markers - 2):
        marker_colors.append(random.choice([fg_color, neg_color]))
    random.shuffle(marker_colors)

    clip_index = random.randrange(n_markers) if random.random() < 0.4 else None
    if clip_index is not None and scales[clip_index] == 1:
        scales[clip_index] = random.choice([2, 3, 4])

    specs = []
    for idx, (scale, color) in enumerate(zip(scales, marker_colors)):
        stamp_mask = _scale_mask(template_fg if color == fg_color else ~template_fg, scale)
        specs.append(
            {
                "scale": int(scale),
                "color": int(color),
                "mask": stamp_mask,
                "needs_clip": idx == clip_index,
            }
        )
    specs.sort(key=lambda item: (item["needs_clip"], item["mask"].shape[0]), reverse=True)

    placed_boxes: list[tuple[int, int, int, int]] = []
    markers: list[tuple[int, int, int, int]] = []
    for spec in specs:
        scale = int(spec["scale"])
        color = int(spec["color"])
        stamp_mask = spec["mask"]
        stamp_size = int(stamp_mask.shape[0])
        total_painted = int(stamp_mask.sum())
        needs_clip = bool(spec["needs_clip"])

        candidates = []
        for r0 in range(0, height - scale + 1):
            for c0 in range(0, width - scale + 1):
                clipped = r0 + stamp_size > height or c0 + stamp_size > width
                if clipped != needs_clip:
                    continue

                vis_box = (r0, c0, min(height - 1, r0 + stamp_size - 1), min(width - 1, c0 + stamp_size - 1))
                if _boxes_too_close(vis_box, frame_bbox, gap=1):
                    continue
                if any(_boxes_too_close(vis_box, other, gap=1) for other in placed_boxes):
                    continue

                visible_painted = _visible_paint_count(stamp_mask, r0, c0, height, width)
                if visible_painted < min(total_painted, scale * scale):
                    continue
                if needs_clip and visible_painted == total_painted:
                    continue

                candidates.append((r0, c0, vis_box))

        if not candidates:
            return None

        r0, c0, vis_box = random.choice(candidates)
        placed_boxes.append(vis_box)
        markers.append((r0, c0, scale, color))

    return sorted(markers)


def _gen_attempt() -> np.ndarray | None:
    fg_color, neg_color = random.sample(PALETTE, 2)
    template_size = random.choice([3, 4])
    template_fg = _sample_template_mask(template_size)

    max_scale = 4
    min_dim = max(20, template_size * max_scale + 5)
    height = random.randint(min_dim, 30)
    width = random.randint(min_dim, 30)

    corner = random.choice(["tl", "tr", "bl", "br"])
    frame_bbox = _corner_bbox(corner, height, width, template_size + 2)
    markers = _try_place_markers(height, width, frame_bbox, template_fg, fg_color, neg_color)
    if markers is None:
        return None

    grid = np.full((height, width), BG, dtype=int)

    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME

    inner = np.where(template_fg, fg_color, BG)
    grid[fr0 + 1 : fr1, fc0 + 1 : fc1] = inner

    for r0, c0, scale, color in markers:
        grid[r0 : r0 + scale, c0 : c0 + scale] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    main_colors = {color for _, _, _, color in markers}
    if main_colors != {fg_color, neg_color}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
