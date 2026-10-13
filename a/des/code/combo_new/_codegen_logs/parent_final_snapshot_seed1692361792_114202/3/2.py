from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    mask = np.zeros((height, width), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return np.array_equal(mask, _frame_border_mask(height, width))


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered same-color fragment set into the matching hollow frame while preserving its relative layout.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping one-pixel-thick hollow rectangular frames, each in a unique non-background color, with empty black interiors.
    - Outside the frames, there are one or more small 4-connected monochrome fragment objects; every fragment color matches exactly one frame color.

    Rule:
    - Remove all fragment pixels from their original outside locations.
    - For each frame color, collect every fragment pixel of that color and compute the minimal bounding box covering the whole set.
    - Translate that full same-color layout unchanged so the bounding-box top-left aligns with the top-left cell of the matching frame's interior, then redraw it there in the same color.
    - Keep every frame border unchanged and leave all other cells black.

    Key insight:
    - Color identifies which fragments belong to which frame, while the fragments' internal spacing must be preserved by translating their combined bounding box as one rigid layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    out = np.full_like(g, BG)

    for color in colors:
        comps = _components(g == color)
        frame_comps = [comp for comp in comps if _is_rect_frame(comp)]
        assert len(frame_comps) == 1, "expected exactly one frame per color"
        frame = frame_comps[0]
        frag_comps = [comp for comp in comps if not _is_rect_frame(comp)]

        out[frame[:, 0], frame[:, 1]] = color
        if not frag_comps:
            continue

        frag = np.concatenate(frag_comps, axis=0)
        sr0, sc0, sr1, sc1 = _bbox(frag)
        fr0, fc0, fr1, fc1 = _bbox(frame)
        ir0, ic0 = fr0 + 1, fc0 + 1
        ir1, ic1 = fr1 - 1, fc1 - 1

        translated = frag - np.array([sr0, sc0], dtype=int) + np.array([ir0, ic0], dtype=int)
        assert translated[:, 0].min() >= ir0 and translated[:, 0].max() <= ir1
        assert translated[:, 1].min() >= ic0 and translated[:, 1].max() <= ic1
        out[translated[:, 0], translated[:, 1]] = color

    return out


def _random_connected_patch() -> np.ndarray:
    height = random.randint(1, 3)
    width = random.randint(1, 3)
    target = random.randint(1, min(4, height * width))

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    coords = np.asarray(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    patch[local[:, 0], local[:, 1]] = True
    return patch


def _can_place_rect(occ: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occ.shape
    if r0 < 0 or c0 < 0 or r0 + height > h or c0 + width > w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_rect(occ: np.ndarray, height: int, width: int, pad: int = 1) -> tuple[int, int] | None:
    positions: list[tuple[int, int]] = []
    for r0 in range(occ.shape[0] - height + 1):
        for c0 in range(occ.shape[1] - width + 1):
            if _can_place_rect(occ, r0, c0, height, width, pad=pad):
                positions.append((r0, c0))
    if not positions:
        return None
    r0, c0 = random.choice(positions)
    occ[r0 : r0 + height, c0 : c0 + width] = True
    return r0, c0


def _build_fragment_layout(max_height: int, max_width: int) -> np.ndarray | None:
    for _ in range(120):
        canvas_h = random.randint(1, max_height)
        canvas_w = random.randint(1, max_width)
        canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
        comp_count = random.choice((1, 2, 3))

        ok = True
        for _ in range(comp_count):
            patch = _random_connected_patch()
            ph, pw = patch.shape
            if ph > canvas_h or pw > canvas_w:
                ok = False
                break
            positions = []
            for r0 in range(canvas_h - ph + 1):
                for c0 in range(canvas_w - pw + 1):
                    rr0 = max(0, r0 - 1)
                    cc0 = max(0, c0 - 1)
                    rr1 = min(canvas_h, r0 + ph + 1)
                    cc1 = min(canvas_w, c0 + pw + 1)
                    if canvas[rr0:rr1, cc0:cc1].any():
                        continue
                    positions.append((r0, c0))
            if not positions:
                ok = False
                break
            r0, c0 = random.choice(positions)
            canvas[r0 : r0 + ph, c0 : c0 + pw] |= patch

        if not ok or not canvas.any():
            continue

        comps = _components(canvas)
        if len(comps) != comp_count:
            continue

        coords = np.argwhere(canvas)
        r0, c0, r1, c1 = _bbox(coords)
        cropped = canvas[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] <= max_height and cropped.shape[1] <= max_width:
            return cropped
    return None


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(patch)
    grid[top + rows, left + cols] = color


def _scene_attempt() -> np.ndarray | None:
    color_count = random.choice((2, 3, 4))
    colors = random.sample(PALETTE, color_count)

    min_side = 18 + 2 * max(0, color_count - 2)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    if random.random() < 0.5:
        width = min(30, max(min_side, height + random.randint(-4, 4)))

    specs = []
    for color in colors:
        interior_h = random.randint(3, 8)
        interior_w = random.randint(3, 8)
        layout = _build_fragment_layout(interior_h, interior_w)
        if layout is None:
            return None
        specs.append(
            {
                "color": color,
                "interior_h": interior_h,
                "interior_w": interior_w,
                "frame_h": interior_h + 2,
                "frame_w": interior_w + 2,
                "layout": layout,
                "source_h": int(layout.shape[0]),
                "source_w": int(layout.shape[1]),
            }
        )

    occ = np.zeros((height, width), dtype=bool)
    objects = []
    for idx, spec in enumerate(specs):
        objects.append(("frame", idx, spec["frame_h"], spec["frame_w"]))
        objects.append(("source", idx, spec["source_h"], spec["source_w"]))
    objects.sort(key=lambda item: item[2] * item[3], reverse=True)

    placements: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, obj_h, obj_w in objects:
        pos = _place_rect(occ, obj_h, obj_w, pad=1)
        if pos is None:
            return None
        placements[(kind, idx)] = pos

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    for idx, spec in enumerate(specs):
        color = int(spec["color"])
        fr0, fc0 = placements[("frame", idx)]
        sr0, sc0 = placements[("source", idx)]
        frame_bbox = (fr0, fc0, fr0 + spec["frame_h"] - 1, fc0 + spec["frame_w"] - 1)

        _draw_frame(inp, frame_bbox, color)
        _draw_frame(out, frame_bbox, color)
        _paint_patch(inp, spec["layout"], sr0, sc0, color)
        _paint_patch(out, spec["layout"], fr0 + 1, fc0 + 1, color)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
