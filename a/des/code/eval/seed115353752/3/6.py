import random
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLOR = 5
STAMP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
INNER_H_RANGE = (9, 17)
INNER_W_RANGE = (9, 17)
TRACE_COUNT_RANGE = (2, 5)
SCENE_MARGIN_RANGE = (1, 5)
OUTSIDE_GAP_RANGE = (2, 5)
SIDE_SLACK_RANGE = (0, 7)
PLACE_GAP = 1
MAX_GEN_RETRIES = 400
PARTIAL_RETRIES = 200
STAMP_RETRIES = 200


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out, (r0, c0, r1, c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k).copy() for k in range(4)]


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {BG}):
        for comp in _components(grid == color):
            if comp.shape[0] < 8:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            sub = np.zeros((h, w), dtype=bool)
            sub[comp[:, 0] - r0, comp[:, 1] - c0] = True
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.array_equal(sub, border):
                candidates.append((int(color), (r0, c0, r1, c1)))
    assert len(candidates) == 1, "expected one rectangular frame"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete partial stamp traces inside a framed canvas by matching the stamp's rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_color, (r0, c0, r1, c1) = _find_frame(g)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]

    colors = sorted(set(map(int, g.ravel())) - {BG, frame_color})
    assert len(colors) == 1, "expected one stamp color"
    stamp_color = int(colors[0])

    stamp_components = _components(g == stamp_color)
    assert stamp_components, "missing stamp-colored pixels"
    full_stamp_coords = max(stamp_components, key=lambda comp: int(comp.shape[0]))
    stamp_mask, _ = _coords_to_mask(full_stamp_coords)
    rotations = _rotations(stamp_mask)

    observed = inner == stamp_color
    trace_components = _components(observed)
    assert trace_components, "canvas must contain at least one partial trace"

    out = np.zeros_like(inner)
    occupied = np.zeros_like(inner, dtype=bool)

    for comp in trace_components:
        _, (cr0, cc0, cr1, cc1) = _coords_to_mask(comp)
        candidates: list[tuple[int, int, int]] = []
        for rot_idx, full_mask in enumerate(rotations):
            fh, fw = map(int, full_mask.shape)
            row_lo = max(0, cr1 - fh + 1)
            row_hi = min(cr0, inner.shape[0] - fh)
            col_lo = max(0, cc1 - fw + 1)
            col_hi = min(cc0, inner.shape[1] - fw)
            if row_lo > row_hi or col_lo > col_hi:
                continue
            for top in range(row_lo, row_hi + 1):
                for left in range(col_lo, col_hi + 1):
                    local = np.zeros((fh, fw), dtype=bool)
                    ok = True
                    for rr, cc in comp:
                        lr = int(rr - top)
                        lc = int(cc - left)
                        if lr < 0 or lr >= fh or lc < 0 or lc >= fw or not full_mask[lr, lc]:
                            ok = False
                            break
                        local[lr, lc] = True
                    if not ok:
                        continue
                    window = observed[top : top + fh, left : left + fw]
                    if not np.array_equal(window, local):
                        continue
                    if int(local.sum()) * 2 < int(full_mask.sum()):
                        continue
                    candidates.append((rot_idx, top, left))
        assert len(candidates) == 1, "each trace must match exactly one rotated stamp"
        rot_idx, top, left = candidates[0]
        full_mask = rotations[rot_idx]
        fh, fw = full_mask.shape
        overlap = occupied[top : top + fh, left : left + fw] & full_mask
        assert not np.any(overlap), "completed stamps must not overlap"
        out[top : top + fh, left : left + fw][full_mask] = stamp_color
        occupied[top : top + fh, left : left + fw] |= full_mask

    assert np.all(out[observed] == stamp_color)
    return out


def _generate_stamp() -> np.ndarray | None:
    for _ in range(STAMP_RETRIES):
        raw_h = random.randint(3, 5)
        raw_w = random.randint(3, 5)
        max_area = min(raw_h * raw_w - 1, 9)
        min_area = max(6, raw_h + raw_w - 2)
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        mask = np.zeros((raw_h, raw_w), dtype=bool)
        seed = (random.randrange(raw_h), random.randrange(raw_w))
        mask[seed] = True
        cells = {seed}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in list(cells):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < raw_h and 0 <= nc < raw_w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True
            cells.add((nr, nc))
        if int(mask.sum()) != target_area:
            continue
        mask = _tight_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < 6:
            continue
        if not _is_connected(mask):
            continue
        if len({_mask_key(rot) for rot in _rotations(mask)}) != 4:
            continue
        return mask
    return None


def _orientation_matches(partial: np.ndarray, rotations: list[np.ndarray]) -> list[int]:
    matches: list[int] = []
    for idx, rot in enumerate(rotations):
        if rot.shape != partial.shape:
            continue
        if np.all(rot[partial]):
            matches.append(idx)
    return matches


def _make_partial(full_mask: np.ndarray, rotations: list[np.ndarray]) -> np.ndarray | None:
    full_area = int(full_mask.sum())
    min_keep = (full_area + 1) // 2
    coords = [tuple(map(int, cell)) for cell in np.argwhere(full_mask)]
    for _ in range(PARTIAL_RETRIES):
        keep = full_mask.copy()
        target_keep = random.randint(min_keep, full_area - 1)
        order = coords[:]
        random.shuffle(order)
        for r, c in order:
            if int(keep.sum()) <= target_keep:
                break
            keep[r, c] = False
            if not keep.any(axis=1).all() or not keep.any(axis=0).all() or not _is_connected(keep):
                keep[r, c] = True
        if int(keep.sum()) == full_area or int(keep.sum()) < min_keep:
            continue
        if not keep.any(axis=1).all() or not keep.any(axis=0).all():
            continue
        if not _is_connected(keep):
            continue
        if len(_orientation_matches(keep, rotations)) != 1:
            continue
        return keep
    return None


def _place_rectangles(inner_h: int, inner_w: int, shapes: list[np.ndarray]) -> list[tuple[int, int]] | None:
    occupancy = np.zeros((inner_h, inner_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    order = list(range(len(shapes)))
    order.sort(key=lambda idx: shapes[idx].shape[0] * shapes[idx].shape[1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}
    for idx in order:
        h, w = map(int, shapes[idx].shape)
        options: list[tuple[int, int]] = []
        for top in range(inner_h - h + 1):
            for left in range(inner_w - w + 1):
                rr0 = max(0, top - PLACE_GAP)
                rr1 = min(inner_h, top + h + PLACE_GAP)
                cc0 = max(0, left - PLACE_GAP)
                cc1 = min(inner_w, left + w + PLACE_GAP)
                if occupancy[rr0:rr1, cc0:cc1].any():
                    continue
                options.append((top, left))
        if not options:
            return None
        placed[idx] = random.choice(options)
        top, left = placed[idx]
        occupancy[top : top + h, left : left + w] = True
    for idx in range(len(shapes)):
        placements.append(placed[idx])
    return placements


def _scene_layout(
    inner_h: int,
    inner_w: int,
    stamp_h: int,
    stamp_w: int,
) -> tuple[int, int, int, int, int, int] | None:
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    side = random.choice(["top", "bottom", "left", "right"])
    gap = random.randint(*OUTSIDE_GAP_RANGE)
    top_margin = random.randint(*SCENE_MARGIN_RANGE)
    bottom_margin = random.randint(*SCENE_MARGIN_RANGE)
    left_margin = random.randint(*SCENE_MARGIN_RANGE)
    right_margin = random.randint(*SCENE_MARGIN_RANGE)
    slack = random.randint(*SIDE_SLACK_RANGE)

    if side in ("top", "bottom"):
        span_w = max(frame_w, stamp_w) + slack
        total_h = top_margin + bottom_margin + frame_h + stamp_h + gap
        total_w = left_margin + right_margin + span_w
        if not (GRID_H_RANGE[0] <= total_h < GRID_H_RANGE[1] and GRID_W_RANGE[0] <= total_w < GRID_W_RANGE[1]):
            return None
        frame_c = left_margin + random.randint(0, span_w - frame_w)
        stamp_c = left_margin + random.randint(0, span_w - stamp_w)
        if side == "top":
            stamp_r = top_margin
            frame_r = top_margin + stamp_h + gap
        else:
            frame_r = top_margin
            stamp_r = top_margin + frame_h + gap
    else:
        span_h = max(frame_h, stamp_h) + slack
        total_h = top_margin + bottom_margin + span_h
        total_w = left_margin + right_margin + frame_w + stamp_w + gap
        if not (GRID_H_RANGE[0] <= total_h < GRID_H_RANGE[1] and GRID_W_RANGE[0] <= total_w < GRID_W_RANGE[1]):
            return None
        frame_r = top_margin + random.randint(0, span_h - frame_h)
        stamp_r = top_margin + random.randint(0, span_h - stamp_h)
        if side == "left":
            stamp_c = left_margin
            frame_c = left_margin + stamp_w + gap
        else:
            frame_c = left_margin
            stamp_c = left_margin + frame_w + gap
    return total_h, total_w, frame_r, frame_c, stamp_r, stamp_c


def _gen_attempt() -> np.ndarray | None:
    stamp_mask = _generate_stamp()
    if stamp_mask is None:
        return None
    rotations = _rotations(stamp_mask)
    stamp_color = random.choice(STAMP_COLORS)

    trace_count = random.randint(*TRACE_COUNT_RANGE)
    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)
    chosen_rotations = [random.randrange(4) for _ in range(trace_count)]
    full_masks = [rotations[idx] for idx in chosen_rotations]
    placements = _place_rectangles(inner_h, inner_w, full_masks)
    if placements is None:
        return None

    output_inner = np.zeros((inner_h, inner_w), dtype=int)
    partial_inner = np.zeros((inner_h, inner_w), dtype=int)

    for full_mask, (top, left) in zip(full_masks, placements):
        partial = _make_partial(full_mask, rotations)
        if partial is None:
            return None
        h, w = full_mask.shape
        output_inner[top : top + h, left : left + w][full_mask] = stamp_color
        partial_inner[top : top + h, left : left + w][partial] = stamp_color

    layout = _scene_layout(inner_h, inner_w, stamp_mask.shape[0], stamp_mask.shape[1])
    if layout is None:
        return None
    total_h, total_w, frame_r, frame_c, stamp_r, stamp_c = layout

    grid = np.zeros((total_h, total_w), dtype=int)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid[frame_r, frame_c : frame_c + frame_w] = FRAME_COLOR
    grid[frame_r + frame_h - 1, frame_c : frame_c + frame_w] = FRAME_COLOR
    grid[frame_r : frame_r + frame_h, frame_c] = FRAME_COLOR
    grid[frame_r : frame_r + frame_h, frame_c + frame_w - 1] = FRAME_COLOR
    grid[frame_r + 1 : frame_r + frame_h - 1, frame_c + 1 : frame_c + frame_w - 1] = partial_inner
    grid[stamp_r : stamp_r + stamp_mask.shape[0], stamp_c : stamp_c + stamp_mask.shape[1]][stamp_mask] = stamp_color

    if np.any(grid == FRAME_COLOR) and np.any(grid == stamp_color):
        try:
            actual = main(grid)
        except AssertionError:
            return None
        if np.array_equal(actual, output_inner):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
