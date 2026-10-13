import random
from collections import Counter, deque

import numpy as np


SIDE_CYCLE = ["top", "left", "bottom", "right"]
TOP_DENSE_BASES = [
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _rotate_side(side: str, k: int) -> str:
    idx = SIDE_CYCLE.index(side)
    return SIDE_CYCLE[(idx + k) % 4]


def _rotation_k(from_side: str, to_side: str) -> int:
    return (SIDE_CYCLE.index(to_side) - SIDE_CYCLE.index(from_side)) % 4


def _dominant_side(mask: np.ndarray) -> str:
    row_counts = mask.sum(axis=1).astype(int)
    col_counts = mask.sum(axis=0).astype(int)
    max_count = int(max(int(row_counts.max()), int(col_counts.max())))
    candidates = []
    if int(row_counts[0]) == max_count:
        candidates.append("top")
    if int(row_counts[-1]) == max_count:
        candidates.append("bottom")
    if int(col_counts[0]) == max_count:
        candidates.append("left")
    if int(col_counts[-1]) == max_count:
        candidates.append("right")
    if len(candidates) == 1:
        return candidates[0]

    boundary_scores = {
        "top": int(row_counts[0]),
        "bottom": int(row_counts[-1]),
        "left": int(col_counts[0]),
        "right": int(col_counts[-1]),
    }
    return max(["top", "bottom", "left", "right"], key=lambda side: (boundary_scores[side], -["top", "bottom", "left", "right"].index(side)))


def _map_marker_index(idx: int, span: int, obj_span: int) -> int:
    slots = span - obj_span + 1
    assert slots >= 1
    if slots == 1 or span == 1:
        return 0
    return int(round(idx * (slots - 1) / (span - 1)))


def _frame_border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _find_frame(grid: np.ndarray, bg: int):
    candidates = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        border = _frame_border_cells(r0, c0, r1, c1)
        if any(int(grid[r, c]) == bg for r, c in border):
            continue
        if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
            continue
        color_counts = Counter(int(grid[r, c]) for r, c in border)
        frame_color, frame_count = color_counts.most_common(1)[0]
        markers = [(r, c) for r, c in border if int(grid[r, c]) != frame_color]
        side_counts = Counter()
        corner_hit = False
        for r, c in markers:
            if r == r0 and c0 < c < c1:
                side_counts["top"] += 1
            elif r == r1 and c0 < c < c1:
                side_counts["bottom"] += 1
            elif c == c0 and r0 < r < r1:
                side_counts["left"] += 1
            elif c == c1 and r0 < r < r1:
                side_counts["right"] += 1
            else:
                corner_hit = True
                break
        if corner_hit or any(v > 1 for v in side_counts.values()):
            continue
        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "frame_color": frame_color,
                "markers": markers,
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                "frame_count": frame_count,
            }
        )
    assert candidates, "frame not found"
    best = max(candidates, key=lambda item: (item["area"], item["frame_count"]))
    return best


def _extract_stamps(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = frame_bbox
    mask = grid != bg
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    stamps = {}
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = next(iter(colors))
        comp_mask = np.zeros(grid.shape, dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True
        cropped = _crop_mask(comp_mask)
        stamps[color] = cropped
    return stamps


def _marker_info(grid: np.ndarray, frame: dict) -> dict[str, tuple[int, int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    markers = {}
    for r, c in frame["markers"]:
        color = int(grid[r, c])
        if r == r0:
            markers["top"] = (r, c, color)
        elif r == r1:
            markers["bottom"] = (r, c, color)
        elif c == c0:
            markers["left"] = (r, c, color)
        elif c == c1:
            markers["right"] = (r, c, color)
    return markers


def _placement_for_marker(
    frame_bbox: tuple[int, int, int, int],
    side: str,
    marker_r: int,
    marker_c: int,
    obj_h: int,
    obj_w: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if side == "top":
        idx = marker_c - inner_c0
        return inner_r0, inner_c0 + _map_marker_index(idx, inner_w, obj_w)
    if side == "bottom":
        idx = marker_c - inner_c0
        return r1 - obj_h, inner_c0 + _map_marker_index(idx, inner_w, obj_w)
    if side == "left":
        idx = marker_r - inner_r0
        return inner_r0 + _map_marker_index(idx, inner_h, obj_h), inner_c0
    idx = marker_r - inner_r0
    return inner_r0 + _map_marker_index(idx, inner_h, obj_h), c1 - obj_w


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frame = _find_frame(g, bg)
    stamps = _extract_stamps(g, bg, frame["bbox"])
    markers = _marker_info(g, frame)

    out = np.full_like(g, bg)
    r0, c0, r1, c1 = frame["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]

    for side, (mr, mc, color) in markers.items():
        assert color in stamps, f"missing stamp for marker color {color}"
        stamp = stamps[color]
        base_side = _dominant_side(stamp)
        rot = np.rot90(stamp, _rotation_k(base_side, side))
        pr, pc = _placement_for_marker(frame["bbox"], side, mr, mc, rot.shape[0], rot.shape[1])
        for rr, cc in np.argwhere(rot):
            out[pr + int(rr), pc + int(cc)] = color

    return out


def _random_top_dense_stamp(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        if rng.random() < 0.45:
            mask = TOP_DENSE_BASES[rng.randrange(len(TOP_DENSE_BASES))].copy()
        else:
            h = rng.randint(2, 4)
            w = rng.randint(2, 4)
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            row_counts = [w] + [0] * (h - 1)
            col_counts = [1] * w
            target_area = rng.randint(max(3, w), min(8, h * w))
            while int(mask.sum()) < target_area:
                candidates = []
                for r in range(1, h):
                    for c in range(w):
                        if mask[r, c]:
                            continue
                        if row_counts[r] >= w - 1 or col_counts[c] >= w - 1:
                            continue
                        if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                            candidates.append((r, c))
                if not candidates:
                    break
                rr, cc = candidates[rng.randrange(len(candidates))]
                mask[rr, cc] = True
                row_counts[rr] += 1
                col_counts[cc] += 1
            mask = _crop_mask(mask)
        if 3 <= int(mask.sum()) <= 8 and _dominant_side(mask) == "top":
            if not np.all(mask):
                return mask
    raise RuntimeError("failed to build stamp")


def _random_stamp_with_side(rng: random.Random, side: str) -> np.ndarray:
    return np.rot90(_random_top_dense_stamp(rng), _rotation_k("top", side))


def _cells_after_place(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + int(r), left + int(c)) for r, c in np.argwhere(mask)}


def _expanded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _marker_cell(frame_bbox: tuple[int, int, int, int], side: str, idx: int) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return r0, c0 + 1 + idx
    if side == "bottom":
        return r1, c0 + 1 + idx
    if side == "left":
        return r0 + 1 + idx, c0
    return r0 + 1 + idx, c1


def _place_inside_canvas(
    rng: random.Random,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    stamp_rot: np.ndarray,
    occupied: set[tuple[int, int]],
    padded: set[tuple[int, int]],
) -> tuple[int, int, int, set[tuple[int, int]]] | None:
    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    obj_h, obj_w = stamp_rot.shape
    span = inner_w if side in ("top", "bottom") else inner_h
    indices = list(range(span))
    rng.shuffle(indices)
    for idx in indices:
        mr, mc = _marker_cell(frame_bbox, side, idx)
        pr, pc = _placement_for_marker(frame_bbox, side, mr, mc, obj_h, obj_w)
        cells = _cells_after_place(stamp_rot, pr, pc)
        if any(cell in padded for cell in cells):
            continue
        return idx, pr, pc, cells
    return None


def _place_outside(
    rng: random.Random,
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    stamps: list[tuple[int, np.ndarray]],
    bg: int,
):
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True
    for color, mask in stamps:
        mh, mw = mask.shape
        candidates = []
        for top in range(0, h - mh + 1):
            for left in range(0, w - mw + 1):
                ok = True
                for rr, cc in np.argwhere(mask):
                    if blocked[top + int(rr), left + int(cc)]:
                        ok = False
                        break
                if ok:
                    candidates.append((top, left))
        if not candidates:
            return False
        top, left = candidates[rng.randrange(len(candidates))]
        for rr, cc in np.argwhere(mask):
            grid[top + int(rr), left + int(cc)] = color
        cells = _cells_after_place(mask, top, left)
        for rr, cc in _expanded(cells, h, w):
            blocked[rr, cc] = True
    return True


def _make_case(rng: random.Random):
    num_markers = rng.randint(2, 4)
    sides = rng.sample(["top", "bottom", "left", "right"], num_markers)

    interior_h = rng.randint(8, 12)
    interior_w = rng.randint(8, 12)
    outer_h = interior_h + 2
    outer_w = interior_w + 2

    top_margin = rng.randint(3, 7)
    bottom_margin = rng.randint(3, 7)
    left_margin = rng.randint(3, 7)
    right_margin = rng.randint(3, 7)
    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    frame_bbox = (r0, c0, r1, c1)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    frame_color = rng.choice(palette)
    palette.remove(frame_color)
    marker_colors = rng.sample(palette, num_markers)

    original_stamps: dict[int, np.ndarray] = {}
    final_stamps: dict[str, tuple[int, np.ndarray, int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    padded: set[tuple[int, int]] = set()

    for side, color in zip(sides, marker_colors):
        base_side = rng.choice(["top", "bottom", "left", "right"])
        original = _random_stamp_with_side(rng, base_side)
        rotated = np.rot90(original, _rotation_k(_dominant_side(original), side))
        if rotated.shape[0] > interior_h or rotated.shape[1] > interior_w:
            return None
        placed = _place_inside_canvas(rng, frame_bbox, side, rotated, occupied, padded)
        if placed is None:
            return None
        idx, pr, pc, cells = placed
        original_stamps[color] = original
        final_stamps[side] = (color, rotated, pr, pc)
        occupied |= cells
        padded |= _expanded(cells, grid_h, grid_w)
        final_stamps[side] = (color, rotated, pr, pc, idx)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = np.full((grid_h, grid_w), bg, dtype=int)

    for r, c in _frame_border_cells(r0, c0, r1, c1):
        inp[r, c] = frame_color
        out[r, c] = frame_color

    for side, item in final_stamps.items():
        color, rotated, pr, pc, idx = item
        mr, mc = _marker_cell(frame_bbox, side, idx)
        inp[mr, mc] = color
        out[mr, mc] = color
        for rr, cc in np.argwhere(rotated):
            out[pr + int(rr), pc + int(cc)] = color

    outside_stamps = sorted(original_stamps.items(), key=lambda item: item[0])
    if not _place_outside(rng, inp, frame_bbox, outside_stamps, bg):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        built = _make_case(rng)
        if built is None:
            continue
        inp, out = built
        got = main(inp)
        if np.array_equal(got, out):
            return inp
    raise RuntimeError("generate_input: failed after retries")
