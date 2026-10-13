from __future__ import annotations

import random
from collections import Counter

import numpy as np


FRAME_COLOR = 5
BRANCHES = ("tb", "lr", "rot")
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _tight_patch_from_coords(
    coords: list[tuple[int, int]], color: int, bg: int
) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch, patch != bg


def _colored_components(
    grid: np.ndarray, ignored: set[int]
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignored:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _transform(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tb":
        return np.flipud(arr)
    if kind == "lr":
        return np.fliplr(arr)
    assert kind == "rot"
    return np.rot90(arr, 2)


def _classify_symmetry(mask: np.ndarray) -> str:
    tb = bool(np.array_equal(mask, np.flipud(mask)))
    lr = bool(np.array_equal(mask, np.fliplr(mask)))
    rot = bool(np.array_equal(mask, np.rot90(mask, 2)))
    if tb and not lr and not rot:
        return "tb"
    if lr and not tb and not rot:
        return "lr"
    if rot and not tb and not lr:
        return "rot"
    raise AssertionError("marker symmetry must indicate exactly one transform")


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, mask: np.ndarray | None = None) -> None:
    h, w = patch.shape
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    rr = slice(r0, r0 + h)
    cc = slice(c0, c0 + w)
    region = dst[rr, cc]
    region[mask] = patch[mask]


def _detect_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == FRAME_COLOR)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    assert int(r1) - int(r0) >= 2 and int(c1) - int(c0) >= 2
    return int(r0), int(c0), int(r1), int(c1)


def _frame_dots(
    grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]
) -> dict[int, tuple[str, int]]:
    r0, c0, r1, c1 = frame_bbox
    dots: dict[int, tuple[str, int]] = {}
    for c in range(c0 + 1, c1):
        top = int(grid[r0, c])
        if top not in {bg, FRAME_COLOR}:
            dots[top] = ("top", c - c0 - 1)
        bottom = int(grid[r1, c])
        if bottom not in {bg, FRAME_COLOR}:
            dots[bottom] = ("bottom", c - c0 - 1)
    for r in range(r0 + 1, r1):
        left = int(grid[r, c0])
        if left not in {bg, FRAME_COLOR}:
            dots[left] = ("left", r - r0 - 1)
        right = int(grid[r, c1])
        if right not in {bg, FRAME_COLOR}:
            dots[right] = ("right", r - r0 - 1)
    return dots


def _place_from_dot(
    side: str, anchor: int, patch_shape: tuple[int, int], interior_shape: tuple[int, int]
) -> tuple[int, int]:
    ih, iw = interior_shape
    ph, pw = patch_shape
    if side == "top":
        return 0, anchor
    if side == "bottom":
        return ih - ph, anchor
    if side == "left":
        return anchor, 0
    assert side == "right"
    return anchor, iw - pw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the symmetry of the marker inside the frame as a transform instruction.

    The marker is a single monochrome sprite inside a grey frame. Its shape is either
    top-bottom symmetric, left-right symmetric, or only 180-degree symmetric. That
    selects one transform shared by every outside source sprite. Each source color has
    a matching 1-pixel dot on the frame edge; the dot tells which side to touch and
    where the transformed sprite starts along that side. The output is only the frame
    interior, without the marker or the frame itself.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_bbox = _detect_frame_bbox(g)
    fr0, fc0, fr1, fc1 = frame_bbox
    dots = _frame_dots(g, bg, frame_bbox)

    comps = _colored_components(g, {bg, FRAME_COLOR})
    marker_mask: np.ndarray | None = None
    sources: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        inside = all(fr0 < r < fr1 and fc0 < c < fc1 for r, c in coords)
        on_border = all(
            ((r in (fr0, fr1) and fc0 <= c <= fc1) or (c in (fc0, fc1) and fr0 <= r <= fr1))
            for r, c in coords
        )
        if inside:
            patch, mask = _tight_patch_from_coords(coords, color, bg)
            if marker_mask is not None:
                raise AssertionError("expected a single marker inside the frame")
            marker_mask = mask
        elif on_border:
            continue
        else:
            patch, mask = _tight_patch_from_coords(coords, color, bg)
            sources[color] = (patch, mask)

    assert marker_mask is not None
    assert set(dots) == set(sources)
    kind = _classify_symmetry(marker_mask)

    interior = np.full((fr1 - fr0 - 1, fc1 - fc0 - 1), bg, dtype=int)
    for color, (side, anchor) in dots.items():
        patch, mask = sources[color]
        t_patch = _transform(patch, kind)
        t_mask = _transform(mask, kind)
        pr, pc = _place_from_dot(side, anchor, t_patch.shape, interior.shape)
        _paste(interior, t_patch, pr, pc, t_mask)
    return interior


def _grow_connected_mask(
    h: int, w: int, target_area: int, forced_cells: list[tuple[int, int]] | None = None
) -> np.ndarray | None:
    forced = set(forced_cells or [])
    if not forced:
        forced.add((random.randrange(h), random.randrange(w)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in forced:
        if not (0 <= r < h and 0 <= c < w):
            return None
        mask[r, c] = True
    frontier = set()
    for r, c in forced:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    while int(mask.sum()) < target_area:
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        mask[nr, nc] = True
        for xr, xc in _neighbors4(nr, nc):
            if 0 <= xr < h and 0 <= xc < w and not mask[xr, xc]:
                frontier.add((xr, xc))
    return _crop_mask(mask)


def _random_source_mask(kind: str, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        area = random.randint(3, max(3, min(h * w - 1, h * w - max(1, (h * w) // 3))))
        mask = _grow_connected_mask(h, w, area)
        if mask is None or not _is_connected(mask):
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if np.array_equal(mask, _transform(mask, kind)):
            continue
        return mask
    return None


def _random_tb_marker() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(3, 5)
        w = random.randint(3, 6)
        top_h = (h + 1) // 2
        min_area = max(2, w // 2)
        max_area = max(min_area, top_h * w - 1)
        area = random.randint(min_area, max_area)
        seed = _grow_connected_mask(top_h, w, area, forced_cells=[(top_h - 1, random.randrange(w))])
        if seed is None:
            continue
        if h % 2 == 0:
            full = np.vstack([seed, np.flipud(seed)])
        else:
            full = np.vstack([seed, np.flipud(seed[:-1])])
        if not _is_connected(full):
            continue
        if np.array_equal(full, np.fliplr(full)):
            continue
        return full
    return None


def _random_lr_marker() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(3, 6)
        w = random.randint(3, 5)
        left_w = (w + 1) // 2
        min_area = max(2, h // 2)
        max_area = max(min_area, h * left_w - 1)
        area = random.randint(min_area, max_area)
        seed = _grow_connected_mask(h, left_w, area, forced_cells=[(random.randrange(h), left_w - 1)])
        if seed is None:
            continue
        if w % 2 == 0:
            full = np.hstack([seed, np.fliplr(seed)])
        else:
            full = np.hstack([seed, np.fliplr(seed[:, :-1])])
        if not _is_connected(full):
            continue
        if np.array_equal(full, np.flipud(full)):
            continue
        return full
    return None


def _random_rot_marker() -> np.ndarray | None:
    for _ in range(160):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        center_r = (h - 1) / 2.0
        center_c = (w - 1) / 2.0
        starters = []
        for r in range(h):
            for c in range(w):
                if abs(r - center_r) + abs(c - center_c) <= 1.5:
                    starters.append((r, c))
        area = random.randint(2, max(2, min(h * w - 1, h * w // 2 + 1)))
        base = _grow_connected_mask(h, w, area, forced_cells=[random.choice(starters)])
        if base is None:
            continue
        full = _crop_mask(base | np.rot90(base, 2))
        if not _is_connected(full):
            continue
        if np.array_equal(full, np.flipud(full)) or np.array_equal(full, np.fliplr(full)):
            continue
        return full
    return None


def _random_marker_mask(kind: str) -> np.ndarray | None:
    if kind == "tb":
        return _random_tb_marker()
    if kind == "lr":
        return _random_lr_marker()
    return _random_rot_marker()


def _reserve_rect(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 0) -> bool:
    hh, ww = used.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    if used[rr0:rr1, cc0:cc1].any():
        return False
    used[r0 : r0 + h, c0 : c0 + w] = True
    return True


def _sample_side_position(
    side: str, ph: int, pw: int, interior_shape: tuple[int, int], used: np.ndarray
) -> tuple[int, int] | None:
    ih, iw = interior_shape
    if ph > ih or pw > iw:
        return None
    for _ in range(80):
        if side == "top":
            pr = 0
            pc = random.randint(0, iw - pw)
        elif side == "bottom":
            pr = ih - ph
            pc = random.randint(0, iw - pw)
        elif side == "left":
            pr = random.randint(0, ih - ph)
            pc = 0
        else:
            pr = random.randint(0, ih - ph)
            pc = iw - pw
        probe = used.copy()
        if _reserve_rect(probe, pr, pc, ph, pw, pad=1):
            return pr, pc
    return None


def _sample_marker_position(
    marker_mask: np.ndarray, interior_shape: tuple[int, int], used: np.ndarray
) -> tuple[int, int] | None:
    ih, iw = interior_shape
    mh, mw = marker_mask.shape
    if mh >= ih - 1 or mw >= iw - 1:
        return None
    for _ in range(120):
        pr = random.randint(1, ih - mh - 1)
        pc = random.randint(1, iw - mw - 1)
        probe = used.copy()
        if _reserve_rect(probe, pr, pc, mh, mw, pad=1):
            return pr, pc
    return None


def _place_outside_frame(
    grid: np.ndarray,
    patch: np.ndarray,
    blocked: np.ndarray,
    bg: int,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(300):
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        probe = blocked.copy()
        if not _reserve_rect(probe, r0, c0, ph, pw, pad=1):
            continue
        blocked[:] = probe
        _paste(grid, patch, r0, c0, patch != bg)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    kind = random.choice(BRANCHES)
    interior_h = random.randint(10, 15)
    interior_w = random.randint(10, 15)
    source_count = random.randint(2, 4)
    sides = random.sample(SIDES, source_count)

    palette = [c for c in range(10) if c != FRAME_COLOR]
    bg = random.choice(palette)
    usable = [c for c in palette if c != bg]
    random.shuffle(usable)
    source_colors = usable[:source_count]
    marker_color = usable[source_count]

    output = np.full((interior_h, interior_w), bg, dtype=int)
    used = np.zeros((interior_h, interior_w), dtype=bool)
    placements: list[dict[str, object]] = []

    for side, color in zip(sides, source_colors):
        placed = False
        for _ in range(120):
            mask = _random_source_mask(kind, min(5, interior_h), min(5, interior_w))
            if mask is None:
                continue
            ph, pw = mask.shape
            pos = _sample_side_position(side, ph, pw, output.shape, used)
            if pos is None:
                continue
            pr, pc = pos
            used[pr : pr + ph, pc : pc + pw] = True
            patch = np.full((ph, pw), bg, dtype=int)
            patch[mask] = color
            _paste(output, patch, pr, pc, mask)
            anchor = pc if side in {"top", "bottom"} else pr
            placements.append(
                {
                    "side": side,
                    "anchor": anchor,
                    "color": color,
                    "output_patch": patch,
                    "output_mask": mask,
                }
            )
            placed = True
            break
        if not placed:
            return None

    marker_mask = _random_marker_mask(kind)
    if marker_mask is None:
        return None
    marker_pos = _sample_marker_position(marker_mask, output.shape, used)
    if marker_pos is None:
        return None

    extra_h = random.randint(8, 12)
    extra_w = random.randint(8, 12)
    top_margin = random.randint(3, extra_h - 3)
    left_margin = random.randint(3, extra_w - 3)
    bottom_margin = extra_h - top_margin
    right_margin = extra_w - left_margin

    frame_h = interior_h + 2
    frame_w = interior_w + 2
    grid_h = frame_h + top_margin + bottom_margin
    grid_w = frame_w + left_margin + right_margin
    fr0 = top_margin
    fc0 = left_margin
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR

    mr, mc = marker_pos
    marker_patch = np.full(marker_mask.shape, bg, dtype=int)
    marker_patch[marker_mask] = marker_color
    _paste(grid, marker_patch, fr0 + 1 + mr, fc0 + 1 + mc, marker_mask)

    blocked = np.zeros_like(grid, dtype=bool)
    rr0 = max(0, fr0 - 1)
    cc0 = max(0, fc0 - 1)
    rr1 = min(grid_h, fr1 + 2)
    cc1 = min(grid_w, fc1 + 2)
    blocked[rr0:rr1, cc0:cc1] = True

    for info in placements:
        side = str(info["side"])
        anchor = int(info["anchor"])
        color = int(info["color"])
        if side == "top":
            grid[fr0, fc0 + 1 + anchor] = color
        elif side == "bottom":
            grid[fr1, fc0 + 1 + anchor] = color
        elif side == "left":
            grid[fr0 + 1 + anchor, fc0] = color
        else:
            grid[fr0 + 1 + anchor, fc1] = color

        src_patch = _transform(np.asarray(info["output_patch"]), kind)
        if not _place_outside_frame(grid, src_patch, blocked, bg):
            return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
