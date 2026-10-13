from __future__ import annotations

import random
from collections import deque

import numpy as np


SIDES = ("top", "bottom", "left", "right")
BG = 0
MAX_GEN_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _fence_info_from_component(coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros_like(local, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(local & ~border):
        return None

    missing = np.argwhere(border & ~local)
    if missing.shape[0] != 1:
        return None

    gr, gc = map(int, missing[0])
    side: str | None = None
    if gr == 0 and 0 < gc < w - 1:
        side = "top"
    elif gr == h - 1 and 0 < gc < w - 1:
        side = "bottom"
    elif gc == 0 and 0 < gr < h - 1:
        side = "left"
    elif gc == w - 1 and 0 < gr < h - 1:
        side = "right"
    if side is None:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap": (r0 + gr, c0 + gc),
    }


def _mirror_shape(shape: np.ndarray, side: str) -> np.ndarray:
    if side in ("left", "right"):
        return np.fliplr(shape)
    return np.flipud(shape)


def _projection_top_left(frame_bbox: tuple[int, int, int, int], gap: tuple[int, int], shape: np.ndarray, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    gap_r, gap_c = gap
    ph, pw = shape.shape
    if side == "right":
        return gap_r - ph // 2, c1 + 1
    if side == "left":
        return gap_r - ph // 2, c0 - pw
    if side == "top":
        return r0 - ph, gap_c - pw // 2
    return r1 + 1, gap_c - pw // 2


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]

    fences: list[dict[str, object]] = []
    for color in colors:
        for coords in _components_of_color(g, color):
            info = _fence_info_from_component(coords)
            if info is None:
                continue
            r0, c0, r1, c1 = info["bbox"]
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = [int(v) for v in np.unique(interior) if int(v) not in (bg, color)]
            if len(inner_colors) != 1:
                continue
            inner_color = inner_colors[0]
            shape_mask = interior == inner_color
            if not np.any(shape_mask):
                continue
            fences.append(
                {
                    "fence_color": color,
                    "frame_bbox": info["bbox"],
                    "side": info["side"],
                    "gap": info["gap"],
                    "shape": _crop_mask(shape_mask),
                }
            )

    assert fences, "expected at least one gapped rectangular fence"

    out = g.copy()
    h, w = out.shape
    for fence in sorted(fences, key=lambda item: tuple(item["frame_bbox"])):
        frame_bbox = fence["frame_bbox"]
        side = str(fence["side"])
        gap = tuple(fence["gap"])
        fence_color = int(fence["fence_color"])
        mirrored = _mirror_shape(np.asarray(fence["shape"], dtype=bool), side)
        top, left = _projection_top_left(frame_bbox, gap, mirrored, side)
        ph, pw = mirrored.shape
        assert 0 <= top and 0 <= left and top + ph <= h and left + pw <= w
        out[top : top + ph, left : left + pw][mirrored] = fence_color
    return out


def _random_connected_shape(side: str) -> np.ndarray:
    needs_lr = side in ("left", "right")
    for _ in range(200):
        canvas_h = random.choice((3, 5))
        canvas_w = random.choice((3, 5))
        target = random.randint(3, min(8, canvas_h * canvas_w - 1))
        seed = (random.randrange(canvas_h), random.randrange(canvas_w))
        cells = {seed}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.shape[0] % 2 == 0 or mask.shape[1] % 2 == 0:
            continue
        if needs_lr and np.array_equal(mask, np.fliplr(mask)):
            continue
        if not needs_lr and np.array_equal(mask, np.flipud(mask)):
            continue
        if mask.shape[1 if needs_lr else 0] == 1:
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric connected shape")


def _make_object_spec(side: str) -> dict[str, object]:
    shape = _random_connected_shape(side)
    ph, pw = shape.shape
    frame_h = random.randint(ph + 4, ph + 8)
    frame_w = random.randint(pw + 4, pw + 8)

    if side in ("left", "right"):
        gap_offset = random.randint(1, frame_h - 2)
    else:
        gap_offset = random.randint(1, frame_w - 2)

    shape_r = random.randint(2, frame_h - ph - 2)
    shape_c = random.randint(2, frame_w - pw - 2)

    if side == "right":
        proj_r, proj_c = gap_offset - ph // 2, frame_w
    elif side == "left":
        proj_r, proj_c = gap_offset - ph // 2, -pw
    elif side == "top":
        proj_r, proj_c = -ph, gap_offset - pw // 2
    else:
        proj_r, proj_c = frame_h, gap_offset - pw // 2

    min_r = min(0, proj_r)
    max_r = max(frame_h - 1, proj_r + ph - 1)
    min_c = min(0, proj_c)
    max_c = max(frame_w - 1, proj_c + pw - 1)

    return {
        "side": side,
        "shape": shape,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "gap_offset": gap_offset,
        "shape_r": shape_r,
        "shape_c": shape_c,
        "out_bbox_rel": (min_r, min_c, max_r, max_c),
    }


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _place_specs(rows: int, cols: int, specs: list[dict[str, object]]) -> list[tuple[dict[str, object], int, int]] | None:
    reserved: list[tuple[int, int, int, int]] = []
    placements: list[tuple[dict[str, object], int, int]] = []
    ordered = sorted(
        specs,
        key=lambda spec: (
            (int(spec["out_bbox_rel"][2]) - int(spec["out_bbox_rel"][0]) + 1)
            * (int(spec["out_bbox_rel"][3]) - int(spec["out_bbox_rel"][1]) + 1)
        ),
        reverse=True,
    )

    for spec in ordered:
        min_r, min_c, max_r, max_c = spec["out_bbox_rel"]
        r_lo = 1 - int(min_r)
        r_hi = rows - 2 - int(max_r)
        c_lo = 1 - int(min_c)
        c_hi = cols - 2 - int(max_c)
        if r_lo > r_hi or c_lo > c_hi:
            return None

        placed = False
        for _ in range(250):
            fr0 = random.randint(r_lo, r_hi)
            fc0 = random.randint(c_lo, c_hi)
            out_box = (
                fr0 + int(min_r),
                fc0 + int(min_c),
                fr0 + int(max_r),
                fc0 + int(max_c),
            )
            padded = _expand_bbox(out_box, 1)
            if any(_intersects(padded, other) for other in reserved):
                continue
            reserved.append(padded)
            placements.append((spec, fr0, fc0))
            placed = True
            break
        if not placed:
            return None
    return placements


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int, side: str, gap_offset: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    if side == "top":
        grid[top, left + gap_offset] = BG
    elif side == "bottom":
        grid[bottom, left + gap_offset] = BG
    elif side == "left":
        grid[top + gap_offset, left] = BG
    else:
        grid[top + gap_offset, right] = BG


def _render_input(rows: int, cols: int, placements: list[tuple[dict[str, object], int, int]], fence_color: int, inner_color: int) -> np.ndarray:
    grid = np.full((rows, cols), BG, dtype=int)
    for spec, fr0, fc0 in placements:
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        _draw_frame(
            grid,
            fr0,
            fc0,
            frame_h,
            frame_w,
            fence_color,
            str(spec["side"]),
            int(spec["gap_offset"]),
        )
        shape = np.asarray(spec["shape"], dtype=bool)
        sr = fr0 + int(spec["shape_r"])
        sc = fc0 + int(spec["shape_c"])
        grid[sr : sr + shape.shape[0], sc : sc + shape.shape[1]][shape] = inner_color
    return grid


def _grid_shape_for_count(count: int) -> tuple[int, int]:
    if count == 1:
        return random.randint(16, 22), random.randint(16, 22)
    if count == 2:
        return random.randint(20, 26), random.randint(20, 26)
    return random.randint(24, 30), random.randint(24, 30)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(1, 3)
    sides = random.sample(SIDES, count)
    specs = [_make_object_spec(side) for side in sides]
    rows, cols = _grid_shape_for_count(count)

    placements = _place_specs(rows, cols, specs)
    if placements is None:
        return None

    palette = [color for color in range(1, 10)]
    fence_color, inner_color = random.sample(palette, 2)
    grid = _render_input(rows, cols, placements, fence_color, inner_color)

    if _dominant_color(grid) != BG:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
