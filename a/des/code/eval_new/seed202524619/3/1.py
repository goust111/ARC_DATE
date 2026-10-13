from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 7
SIDES = ("up", "right", "down", "left")
ROT_BY_SIDE = {"up": 0, "right": 3, "down": 2, "left": 1}
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox_of_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or visited[sr, sc]:
                continue
            queue = deque([(int(sr), int(sc))])
            visited[sr, sc] = True
            cells: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            comps.append(np.array(cells, dtype=int))

    return comps


def _is_rect_frame(comp: np.ndarray) -> tuple[int, int, int, int] | None:
    if comp.size == 0:
        return None

    r0, c0, r1, c1 = _bbox_of_coords(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    if int(comp.shape[0]) != 2 * (h + w) - 4:
        return None

    comp_set = {tuple(map(int, p)) for p in comp}
    for c in range(c0, c1 + 1):
        if (r0, c) not in comp_set or (r1, c) not in comp_set:
            return None
    for r in range(r0 + 1, r1):
        if (r, c0) not in comp_set or (r, c1) not in comp_set:
            return None
    return r0, c0, r1, c1


def _marker_side(fr0: int, fc0: int, fr1: int, fc1: int, mr: int, mc: int) -> str | None:
    inner_top = fr0 + 1
    inner_bottom = fr1 - 1
    inner_left = fc0 + 1
    inner_right = fc1 - 1

    if mr == inner_top and inner_left < mc < inner_right:
        return "up"
    if mr == inner_bottom and inner_left < mc < inner_right:
        return "down"
    if mc == inner_left and inner_top < mr < inner_bottom:
        return "left"
    if mc == inner_right and inner_top < mr < inner_bottom:
        return "right"
    return None


def _mask_from_component(comp: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_coords(comp)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp:
        out[int(r) - r0, int(c) - c0] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior with the matching outside stamp, rotated by marker-side direction.

    Input structure:
    - bg = 0.
    - One non-bg frame-border color draws several closed 1-cell rectangular frames; interiors are black.
    - Each frame interior contains exactly one single marker pixel: its color is unique per frame.
    - Each marker touches exactly one interior wall (not a corner) and indicates rotation direction.
    - Outside frames, each marker color appears as a separate monochrome stamp component.

    Rule:
    - Find the frame-border color and all rectangular frame components.
    - For each frame, pick its interior marker color and matching external stamp component.
    - Rotate the stamp by marker position: top→0°, right→clockwise 90°, bottom→180°, left→counter-clockwise 90°.
    - Clear all non-bg non-frame pixels and paste each rotated stamp centered in the frame interior using that color.

    Key insight:
    - A marker has two meanings at once: the color selects which stamp to use, while its wall-side selects orientation.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return np.asarray(g, dtype=int).copy()
    bg = int(BG)

    out = g.copy()
    out[(out != bg) & (out != FRAME_COLOR)] = bg

    # detect frame color and frame components
    frame_candidates: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in [int(c) for c in np.unique(g) if int(c) != bg]:
        rects: list[tuple[int, int, int, int]] = []
        for comp in _connected_components(g == color):
            frame_box = _is_rect_frame(comp)
            if frame_box is not None:
                rects.append(frame_box)
        if rects:
            frame_candidates[color] = rects
    if not frame_candidates:
        return out

    frame_color = max(frame_candidates, key=lambda c: len(frame_candidates[c]))
    frame_rects = frame_candidates[frame_color]

    color_to_comps: dict[int, list[np.ndarray]] = {}
    for color in [int(c) for c in np.unique(g) if int(c) not in (bg, frame_color)]:
        comps = _connected_components(g == color)
        comps = [comp for comp in comps if len(comp) > 1]
        if comps:
            color_to_comps[color] = comps

    for fr0, fc0, fr1, fc1 in frame_rects:
        ih = fr1 - fr0 - 1
        iw = fc1 - fc0 - 1
        if ih < 3 or iw < 3:
            continue

        markers: list[tuple[int, int, int]] = []
        for r in range(fr0 + 1, fr1):
            for c in range(fc0 + 1, fc1):
                v = int(g[r, c])
                if v != bg and v != frame_color:
                    markers.append((r, c, v))
        if len(markers) != 1:
            continue
        mr, mc, mcol = markers[0]

        side = _marker_side(fr0, fc0, fr1, fc1, mr, mc)
        if side is None:
            continue

        stamp_comp: np.ndarray | None = None
        for comp in color_to_comps.get(mcol, []):
            if _component_overlaps_rect(comp, fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1):
                continue
            if any(int(rr) == mr and int(cc) == mc for rr, cc in comp):
                continue
            if stamp_comp is None or len(comp) > len(stamp_comp):
                stamp_comp = comp
        if stamp_comp is None:
            continue

        stamp = np.rot90(_mask_from_component(stamp_comp), ROT_BY_SIDE[side])
        sh, sw = map(int, stamp.shape)
        if sh > ih or sw > iw:
            continue

        top = fr0 + 1 + (ih - sh) // 2
        left = fc0 + 1 + (iw - sw) // 2
        msr = (mr - (fr0 + 1)) - (ih - sh) // 2
        msc = (mc - (fc0 + 1)) - (iw - sw) // 2
        if not (0 <= msr < sh and 0 <= msc < sw):
            continue
        if not bool(stamp[int(msr), int(msc)]):
            continue

        for sr, sc in np.argwhere(stamp):
            out[int(top + sr), int(left + sc)] = mcol

    return out


def _component_to_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        out[r - r0, c - c0] = True
    return out


def _build_stamp_with_full_rows(h: int, w: int) -> np.ndarray:
    for _ in range(120):
        cells: set[tuple[int, int]] = set()
        c = random.randrange(w)
        for r in range(h):
            if r == 0:
                cells.add((0, c))
                continue

            next_c = random.randrange(w)
            if next_c > c:
                for cc in range(c, next_c + 1):
                    cells.add((r, cc))
            else:
                for cc in range(next_c, c + 1):
                    cells.add((r, cc))
            c = next_c
            cells.add((r, c))

        base = len(cells)
        target = random.randint(base, h * w)
        while len(cells) < target:
            r0, c0 = random.choice(tuple(cells))
            dr, dc = random.choice(DIRS4)
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))

        out = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            out[rr, cc] = True
        if out.sum() >= 2:
            return out
    return np.eye(h, w, dtype=bool)


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        r = np.rot90(mask, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def _is_asymmetric(mask: np.ndarray) -> bool:
    keys = {(m.shape, m.tobytes()) for m in _all_orientations(mask)}
    return len(keys) >= 7


def _component_overlaps_rect(comp: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rows = comp[:, 0]
    cols = comp[:, 1]
    return bool(np.any((rows >= r0) & (rows <= r1) & (cols >= c0) & (cols <= c1)))


def _external_stamp_from_final(side: str, final_mask: np.ndarray) -> np.ndarray:
    if side == "up":
        return final_mask.copy()
    if side == "right":
        return np.rot90(final_mask, 1)
    if side == "down":
        return np.rot90(final_mask, 2)
    return np.rot90(final_mask, 3)


def _build_final_stamp(ih: int, iw: int, side: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(180):
        fh = random.randint(3, ih)
        fw = random.randint(3, iw)
        stamp = _build_stamp_with_full_rows(fh, fw)
        if not _is_asymmetric(stamp):
            continue

        if side in ("up", "down"):
            mr = 0 if side == "up" else fh - 1
            marker_col_min = 1
            marker_col_max = fw - 2
            if marker_col_min > marker_col_max:
                continue
            mc = random.randint(marker_col_min, marker_col_max)
        else:
            mc = 0 if side == "left" else fw - 1
            marker_row_min = 1
            marker_row_max = fh - 2
            if marker_row_min > marker_row_max:
                continue
            mr = random.randint(marker_row_min, marker_row_max)

        stamp[mr, mc] = True
        return stamp, (mr, mc)
    return None


def _can_place_frame(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    H, W = used.shape
    top = max(0, r0 - 1)
    left = max(0, c0 - 1)
    bottom = min(H, r1 + 2)
    right = min(W, c1 + 2)
    if used[top:bottom, left:right].any():
        return False

    return True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _place_frame(used: np.ndarray, grid: np.ndarray, h: int, w: int) -> tuple[int, int, int, int] | None:
    H, W = grid.shape
    if h >= H or w >= W:
        return None

    for _ in range(120):
        r0 = random.randint(0, H - h)
        c0 = random.randint(0, W - w)
        r1 = r0 + h - 1
        c1 = c0 + w - 1

        if not _can_place_frame(used, r0, c0, r1, c1):
            continue

        _draw_frame(grid, r0, c0, r1, c1)
        used[r0 : r1 + 1, c0 : c1 + 1] = True
        return r0, c0, r1, c1

    return None


def _can_place(free: np.ndarray, cells: np.ndarray, tr: int, tc: int) -> bool:
    for r, c in cells:
        if not free[int(tr + r), int(tc + c)]:
            return False
    return True


def _paint_mask(grid: np.ndarray, free: np.ndarray, color: int, tr: int, tc: int, cells: np.ndarray) -> None:
    for r, c in cells:
        rr = int(tr + r)
        cc = int(tc + c)
        grid[rr, cc] = int(color)
        free[rr, cc] = False


def _build_frame_info(fr0: int, fc0: int, fr1: int, fc1: int, side: str, color: int) -> dict[str, object] | None:
    ih = fr1 - fr0 - 1
    iw = fc1 - fc0 - 1
    assert ih >= 4 and iw >= 4

    if side in ("up", "down"):
        stamp_h = ih
        stamp_w = random.randint(3, iw)
        for _ in range(40):
            stamp = _build_stamp_with_full_rows(stamp_h, stamp_w)
            if not _is_asymmetric(stamp):
                continue
            start_row = (ih - stamp_h) // 2
            start_col = (iw - stamp.shape[1]) // 2
            marker_row = 0 if side == "up" else ih - 1
            marker_cols = [
                c for c in range(start_col, start_col + stamp.shape[1]) if 1 <= c <= iw - 2
            ]
            if not marker_cols:
                continue
            marker_col = random.choice(marker_cols)
            marker_r = fr0 + 1 + marker_row
            marker_c = fc0 + 1 + marker_col
            return {"stamp": stamp, "marker": (marker_r, marker_c), "color": color}
        return None

    stamp_w = iw
    stamp_h = random.randint(3, ih)
    for _ in range(40):
        stamp = np.rot90(_build_stamp_with_full_rows(stamp_w, stamp_h), 1)
        if not _is_asymmetric(stamp):
            continue
        start_row = (ih - stamp.shape[0]) // 2
        start_col = (iw - stamp.shape[1]) // 2
        marker_col = 0 if side == "left" else iw - 1
        marker_rows = [
            r for r in range(start_row, start_row + stamp.shape[0]) if 1 <= r <= ih - 2
        ]
        if not marker_rows:
            continue
        marker_row = random.choice(marker_rows)
        marker_r = fr0 + 1 + marker_row
        marker_c = fc0 + 1 + marker_col
        return {"stamp": stamp, "marker": (marker_r, marker_c), "color": color}

    return None


def generate_input() -> np.ndarray:
    """
    Generate valid random inputs: random number of frames, random marker orientations, and random asymmetric stamps.
    """
    for _ in range(1500):
        H = random.randint(18, 30)
        W = random.randint(18, 30)
        n_frames = random.randint(2, 4)

        color_pool = [c for c in range(1, 10) if c != BG and c != FRAME_COLOR]
        if len(color_pool) < n_frames:
            continue
        marker_colors = random.sample(color_pool, n_frames)

        grid = np.full((H, W), BG, dtype=int)
        used = np.zeros((H, W), dtype=bool)
        frame_infos: list[dict[str, object]] = []
        ok = True

        for i in range(n_frames):
            ih = random.randint(4, 8)
            iw = random.randint(4, 8)
            side = random.choice(SIDES)
            rect = _place_frame(used, grid, ih + 2, iw + 2)
            if rect is None:
                ok = False
                break

            fr0, fc0, fr1, fc1 = rect
            build = _build_final_stamp(ih, iw, side)
            if build is None:
                ok = False
                break

            final_stamp, marker_offset = build
            marker_r = fr0 + 1 + marker_offset[0]
            marker_c = fc0 + 1 + marker_offset[1]
            if grid[marker_r, marker_c] != BG:
                ok = False
                break

            grid[marker_r, marker_c] = marker_colors[i]
            frame_infos.append(
                {
                    "side": side,
                    "color": marker_colors[i],
                    "marker": (marker_r, marker_c),
                    "stamp": _external_stamp_from_final(side, final_stamp),
                    "rect": (fr0, fc0, fr1, fc1),
                }
            )

        if not ok or len(frame_infos) != n_frames:
            continue

        free = ~used
        for info in frame_infos:
            color = int(info["color"])
            stamp = info["stamp"]
            sh, sw = map(int, stamp.shape)
            cells = np.argwhere(stamp)
            placed = False
            for _ in range(220):
                tr = random.randint(0, H - sh)
                tc = random.randint(0, W - sw)
                if not _can_place(free, cells, tr, tc):
                    continue
                _paint_mask(grid, free, color, tr, tc, cells)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        # quick sanity: generated sample must follow the stated transform rules
        out = main(grid)
        if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape == grid.shape and max(out.shape) <= 30:
            return grid

    raise RuntimeError("failed to generate valid input")
