import math
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GREY = 5
GEN_RETRIES = 400
OBJECT_RETRIES = 120
PLACE_RETRIES = 240


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _trim(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1].copy()


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _rotate_flip(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _valid_object(mask: np.ndarray, hole_count: int) -> bool:
    if not np.any(mask):
        return False
    mask = _trim(mask)
    comps = _components(mask)
    if len(comps) != 1:
        return False
    if _hole_count(mask) != hole_count:
        return False
    h, w = mask.shape
    if h < 4 or w < 4 or h > 12 or w > 12:
        return False
    if int(mask.sum()) < 10:
        return False
    return True


def _carve_notch(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    side = random.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        if w < 4:
            return out
        depth = random.randint(1, max(1, min(2, h // 3)))
        span = random.randint(2, max(2, w - 2))
        lo = 1 if span < w else 0
        hi = w - span - 1 if span < w else 0
        start = random.randint(lo, hi) if hi >= lo else 0
        if side == "top":
            out[:depth, start : start + span] = False
        else:
            out[h - depth :, start : start + span] = False
    else:
        if h < 4:
            return out
        depth = random.randint(1, max(1, min(2, w // 3)))
        span = random.randint(2, max(2, h - 2))
        lo = 1 if span < h else 0
        hi = h - span - 1 if span < h else 0
        start = random.randint(lo, hi) if hi >= lo else 0
        if side == "left":
            out[start : start + span, :depth] = False
        else:
            out[start : start + span, w - depth :] = False
    return out


def _corner_cut(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    cut_h = random.randint(1, min(2, h - 1))
    cut_w = random.randint(1, min(2, w - 1))
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        out[:cut_h, :cut_w] = False
    elif corner == "tr":
        out[:cut_h, w - cut_w :] = False
    elif corner == "bl":
        out[h - cut_h :, :cut_w] = False
    else:
        out[h - cut_h :, w - cut_w :] = False
    return out


def _add_arm(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    side = random.choice(["top", "bottom", "left", "right"])
    arm_len = random.randint(1, 2)
    if side in ("top", "bottom"):
        span = random.randint(2, max(2, w))
        start = random.randint(0, max(0, w - span))
        out = np.zeros((h + arm_len, w), dtype=bool)
        if side == "top":
            out[arm_len:, :] = mask
            out[:arm_len, start : start + span] = True
        else:
            out[:h, :] = mask
            out[h:, start : start + span] = True
    else:
        span = random.randint(2, max(2, h))
        start = random.randint(0, max(0, h - span))
        out = np.zeros((h, w + arm_len), dtype=bool)
        if side == "left":
            out[:, arm_len:] = mask
            out[start : start + span, :arm_len] = True
        else:
            out[:, :w] = mask
            out[start : start + span, w:] = True
    return out


def _base_mask(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        h = random.randint(5, 9)
        w = random.randint(6, 10)
        return np.ones((h, w), dtype=bool)

    if hole_count >= 3:
        row_count = 2
    else:
        row_count = random.randint(1, min(2, hole_count))
    col_count = math.ceil(hole_count / row_count)
    row_sizes = [random.randint(1, 2) for _ in range(row_count)]
    col_sizes = [random.randint(1, 2) for _ in range(col_count)]
    top = random.randint(2, 3)
    bottom = random.randint(2, 3)
    left = random.randint(2, 3)
    right = random.randint(2, 3)
    row_gap = random.randint(2, 3)
    col_gap = random.randint(2, 3)

    h = top + sum(row_sizes) + row_gap * (row_count - 1) + bottom
    w = left + sum(col_sizes) + col_gap * (col_count - 1) + right
    mask = np.ones((h, w), dtype=bool)

    slots: list[tuple[int, int, int, int]] = []
    rr = top
    for hs in row_sizes:
        cc = left
        for ws in col_sizes:
            slots.append((rr, cc, hs, ws))
            cc += ws + col_gap
        rr += hs + row_gap

    for r0, c0, hh, ww in random.sample(slots, hole_count):
        mask[r0 : r0 + hh, c0 : c0 + ww] = False
    return mask


def _make_object(hole_count: int) -> np.ndarray:
    for _ in range(OBJECT_RETRIES):
        mask = _base_mask(hole_count)
        for _ in range(random.randint(1, 2)):
            op = random.choice(["notch", "corner", "arm"])
            if op == "notch":
                mask = _carve_notch(mask)
            elif op == "corner":
                mask = _corner_cut(mask)
            else:
                mask = _add_arm(mask)
        mask = _rotate_flip(mask)
        mask = _trim(mask)
        if _valid_object(mask, hole_count):
            return mask
    raise RuntimeError(f"failed to build object with {hole_count} holes")


def _mark_forbidden(forbidden: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = forbidden.shape
    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        gr = int(top + r)
        gc = int(left + c)
        for dr, dc in DIR8:
            nr = int(gr + dr)
            nc = int(gc + dc)
            if 0 <= nr < h and 0 <= nc < w:
                forbidden[nr, nc] = True
        forbidden[gr, gc] = True


def _place_objects(
    grid: np.ndarray,
    play_width: int,
    objects: list[tuple[int, np.ndarray]],
) -> bool:
    h, w = grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[0, :] = True
    forbidden[h - 1, :] = True
    forbidden[:, 0] = True
    forbidden[:, play_width:] = True

    for _hole_count_value, mask in sorted(objects, key=lambda item: -int(item[1].sum())):
        mh, mw = mask.shape
        max_r = h - mh - 1
        max_c = play_width - mw - 1
        if max_r < 1 or max_c < 1:
            return False

        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(1, max_r)
            left = random.randint(1, max_c)
            region = forbidden[top : top + mh, left : left + mw]
            if np.any(region & mask):
                continue
            window = grid[top : top + mh, left : left + mw]
            window[mask] = GREY
            _mark_forbidden(forbidden, mask, top, left)
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    non_bg = g[g != bg]
    assert non_bg.size > 0
    vals, cnts = np.unique(non_bg, return_counts=True)
    grey = int(vals[int(np.argmax(cnts))])

    legend_mask = (g != bg) & (g != grey)
    legend_pts = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(legend_mask)]
    assert legend_pts
    legend_pts.sort(key=lambda item: (item[0], item[1]))
    legend_cols = {c for _, c, _ in legend_pts}
    assert len(legend_cols) == 1
    mapping = {idx + 1: color for idx, (_, _c, color) in enumerate(legend_pts)}

    out = np.full(g.shape, bg, dtype=int)
    for pts in _components(g == grey):
        r0, c0, r1, c1 = _bbox(pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[int(r - r0), int(c - c0)] = True
        hole_count = _hole_count(mask)
        if hole_count == 0 or hole_count not in mapping:
            continue
        boundary = _boundary_pixels(mask)
        window = out[r0 : r1 + 1, c0 : c1 + 1]
        window[boundary] = int(mapping[hole_count])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the right-side vertical legend to recolor grey objects by hole count,
    then keep only their background-adjacent boundary pixels.
    """
    return _expected_output(grid)


def _build_case() -> np.ndarray:
    legend_len = random.randint(2, 4)
    hole_counts = list(range(legend_len + 1))
    extra_count = random.randint(0, 1)
    for _ in range(extra_count):
        hole_counts.append(random.randint(0, legend_len))
    random.shuffle(hole_counts)

    objects = [(hc, _make_object(hc)) for hc in hole_counts]
    max_h = max(mask.shape[0] for _, mask in objects)
    max_w = max(mask.shape[1] for _, mask in objects)
    obj_area = sum(int(mask.sum()) for _, mask in objects)

    legend_gap = random.randint(1, 3)

    side_hint = int(math.ceil(math.sqrt(obj_area)))
    min_h = max(16, max_h + 4, legend_len + 3, side_hint + 4)
    min_w = max(20, max_w + legend_gap + 4, side_hint + 8)
    if min_h > 30 or min_w > 30:
        raise RuntimeError("canvas too small for sampled objects")
    grid_h = random.randint(min_h, min(30, min_h + 5))
    grid_w = random.randint(min_w, min(30, min_w + 6))
    if grid_h * grid_w <= obj_area * 2 + legend_len:
        raise RuntimeError("canvas too small")

    bg_choices = [c for c in range(10) if c != GREY]
    bg = random.choice(bg_choices)
    legend_colors = random.sample([c for c in bg_choices if c != bg], legend_len)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    play_width = grid_w - legend_gap - 1
    if not _place_objects(grid, play_width, objects):
        raise RuntimeError("placement failed")

    legend_start = random.randint(1, grid_h - legend_len - 1)
    for idx, color in enumerate(legend_colors):
        grid[legend_start + idx, grid_w - 1] = int(color)

    grey_count = int(np.sum(grid == GREY))
    if grey_count == 0:
        raise RuntimeError("missing grey objects")
    if int(np.sum(grid == bg)) <= grey_count:
        raise RuntimeError("background not dominant")

    non_bg = grid[grid != bg]
    vals, cnts = np.unique(non_bg, return_counts=True)
    if int(vals[int(np.argmax(cnts))]) != GREY:
        raise RuntimeError("grey must be the dominant non-background color")

    grey_comps = _components(grid == GREY)
    if len(grey_comps) != len(objects):
        raise RuntimeError("grey objects merged")

    observed = []
    for pts in grey_comps:
        r0, c0, r1, c1 = _bbox(pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[int(r - r0), int(c - c0)] = True
        observed.append(_hole_count(mask))
    if sorted(observed) != sorted(hole_counts):
        raise RuntimeError("hole counts changed during placement")
    if 0 not in observed or max(observed) < 1:
        raise RuntimeError("missing required branch coverage")

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _build_case()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
