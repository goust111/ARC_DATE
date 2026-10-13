from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique dominant color"
    return int(winners[0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or visited[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in dirs:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == int(bg) or visited[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            visited[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if visited[rr, cc] or int(g[rr, cc]) != color:
                        continue
                    visited[rr, cc] = True
                    stack.append((int(rr), int(cc)))
            arr = np.array(coords, dtype=int)
            bbox = _bbox_from_coords(arr)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "area": int(len(arr)),
                }
            )
    return comps


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _extract_notches(frame_crop: np.ndarray, bg: int) -> list[dict]:
    h, w = frame_crop.shape
    notch_mask = frame_crop == int(bg)
    out: list[dict] = []
    for coords in _components(notch_mask, connectivity=4):
        if not any(
            r == 0 or c == 0 or r == h - 1 or c == w - 1
            for r, c in coords
        ):
            continue
        bbox = _bbox_from_coords(coords)
        out.append(
            {
                "bbox": bbox,
                "mask": _mask_from_coords(coords, bbox),
            }
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the cropped frame by translating each outside piece into the notch with the same size and jagged edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _unique_mode(g)

    comps = _color_components(g, bg)
    assert comps, "expected a non-background frame"
    largest = max(comp["area"] for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == int(largest)]
    assert len(frame_candidates) == 1, "frame must be the unique largest monochrome component"
    frame = frame_candidates[0]

    fr0, fc0, fr1, fc1 = frame["bbox"]
    frame_crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    notches = _extract_notches(frame_crop, bg)
    assert notches, "expected border notches inside the frame crop"

    pieces = [comp for comp in comps if comp is not frame]
    assert len(pieces) == len(notches), "every notch must have one outside piece"

    piece_colors: dict[tuple[tuple[int, int], bytes], int] = {}
    for piece in pieces:
        key = _shape_key(_mask_from_coords(piece["coords"], piece["bbox"]))
        assert key not in piece_colors, "piece shapes must be unique"
        piece_colors[key] = int(piece["color"])

    out = frame_crop.copy()
    for notch in notches:
        nr0, nc0, nr1, nc1 = notch["bbox"]
        key = _shape_key(notch["mask"])
        assert key in piece_colors, "missing matching piece for a notch"
        patch = out[nr0 : nr1 + 1, nc0 : nc1 + 1]
        patch[notch["mask"]] = int(piece_colors[key])
    return out


# --------------- generator ---------------

GRID_H_RANGE = (24, 31)
GRID_W_RANGE = (24, 31)
FRAME_H_RANGE = (11, 18)
FRAME_W_RANGE = (11, 18)
NOTCH_COUNT_RANGE = (3, 6)
DEPTH_RANGE = (2, 4)
SPAN_RANGE = (3, 7)
GEN_ATTEMPTS = 400
NOTCH_ATTEMPTS = 120
PIECE_ATTEMPTS = 200


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dr, dc in DIR8:
        src_r0 = max(0, -dr)
        src_r1 = min(h, h - dr)
        src_c0 = max(0, -dc)
        src_c1 = min(w, w - dc)
        dst_r0 = max(0, dr)
        dst_r1 = min(h, h + dr)
        dst_c0 = max(0, dc)
        dst_c1 = min(w, w + dc)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _sample_profile(depth: int, length: int) -> tuple[int, ...]:
    assert depth >= 2 and length >= 3
    for _ in range(80):
        vals = [random.randint(1, depth)]
        for _j in range(length - 1):
            prev = vals[-1]
            lo = max(1, prev - 1)
            hi = min(depth, prev + 1)
            vals.append(random.randint(lo, hi))
        if max(vals) < depth:
            vals[random.randrange(length)] = depth
        if len(set(vals)) == 1:
            idx = random.randrange(length)
            vals[idx] = depth - 1 if vals[idx] == depth else depth
        if max(vals) == depth and len(set(vals)) >= 2:
            return tuple(int(v) for v in vals)
    return tuple(depth if i % 2 == 0 else depth - 1 for i in range(length))


def _build_notch_mask(side: str, box_h: int, box_w: int, profile: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros((box_h, box_w), dtype=bool)
    if side == "top":
        assert len(profile) == box_w
        for c, d in enumerate(profile):
            mask[:d, c] = True
    elif side == "bottom":
        assert len(profile) == box_w
        for c, d in enumerate(profile):
            mask[box_h - d :, c] = True
    elif side == "left":
        assert len(profile) == box_h
        for r, d in enumerate(profile):
            mask[r, :d] = True
    else:
        assert side == "right"
        assert len(profile) == box_h
        for r, d in enumerate(profile):
            mask[r, box_w - d :] = True
    return mask


def _feasible_shapes(frame_h: int, frame_w: int) -> list[tuple[int, int]]:
    shapes = set()
    for depth in range(DEPTH_RANGE[0], DEPTH_RANGE[1] + 1):
        for span in range(SPAN_RANGE[0], SPAN_RANGE[1] + 1):
            if span <= frame_w - 2 and depth < frame_h:
                shapes.add((depth, span))
            if span <= frame_h - 2 and depth < frame_w:
                shapes.add((span, depth))
    return sorted(shapes)


def _compatible_sides(shape: tuple[int, int], frame_h: int, frame_w: int) -> list[str]:
    box_h, box_w = shape
    sides: list[str] = []
    if DEPTH_RANGE[0] <= box_h <= DEPTH_RANGE[1] and SPAN_RANGE[0] <= box_w <= SPAN_RANGE[1]:
        if box_w <= frame_w - 2 and box_h < frame_h:
            sides.extend(["top", "bottom"])
    if SPAN_RANGE[0] <= box_h <= SPAN_RANGE[1] and DEPTH_RANGE[0] <= box_w <= DEPTH_RANGE[1]:
        if box_h <= frame_h - 2 and box_w < frame_w:
            sides.extend(["left", "right"])
    return sides


def _sample_notch(shape: tuple[int, int], frame_h: int, frame_w: int, occupied: np.ndarray, used_keys: set) -> dict | None:
    box_h, box_w = shape
    sides = _compatible_sides(shape, frame_h, frame_w)
    if not sides:
        return None
    blocked = _dilate8(occupied)
    for _ in range(NOTCH_ATTEMPTS):
        side = random.choice(sides)
        if side in ("top", "bottom"):
            profile = _sample_profile(box_h, box_w)
        else:
            profile = _sample_profile(box_w, box_h)
        local_mask = _build_notch_mask(side, box_h, box_w, profile)
        key = _shape_key(local_mask)
        if key in used_keys:
            continue
        if side == "top":
            rr = 0
            cc = random.randint(1, frame_w - box_w - 1)
        elif side == "bottom":
            rr = frame_h - box_h
            cc = random.randint(1, frame_w - box_w - 1)
        elif side == "left":
            rr = random.randint(1, frame_h - box_h - 1)
            cc = 0
        else:
            rr = random.randint(1, frame_h - box_h - 1)
            cc = frame_w - box_w

        full_mask = np.zeros((frame_h, frame_w), dtype=bool)
        full_mask[rr : rr + box_h, cc : cc + box_w] = local_mask
        if bool((full_mask & blocked).any()):
            continue
        return {
            "bbox": (int(rr), int(cc), int(rr + box_h - 1), int(cc + box_w - 1)),
            "mask": local_mask,
            "full_mask": full_mask,
        }
    return None


def _generate_notches(frame_h: int, frame_w: int, ambiguous: bool) -> list[dict] | None:
    shapes = _feasible_shapes(frame_h, frame_w)
    if not shapes:
        return None

    min_count = NOTCH_COUNT_RANGE[0] + (1 if ambiguous else 0)
    max_count = NOTCH_COUNT_RANGE[1]
    if min_count > max_count:
        return None

    for _ in range(160):
        count = random.randint(min_count, max_count)
        available = list(shapes)
        if ambiguous:
            rep_shape = random.choice(available)
            singles = [shape for shape in available if shape != rep_shape]
            if len(singles) < count - 2:
                continue
            plan = [rep_shape, rep_shape] + random.sample(singles, count - 2)
        else:
            if len(available) < count:
                continue
            plan = random.sample(available, count)
        random.shuffle(plan)

        occupied = np.zeros((frame_h, frame_w), dtype=bool)
        used_keys: set[tuple[tuple[int, int], bytes]] = set()
        notches: list[dict] = []
        ok = True
        for shape in plan:
            notch = _sample_notch(shape, frame_h, frame_w, occupied, used_keys)
            if notch is None:
                ok = False
                break
            occupied |= notch["full_mask"]
            used_keys.add(_shape_key(notch["mask"]))
            notches.append(notch)
        if not ok:
            continue

        shape_counts = Counter(tuple(map(int, notch["mask"].shape)) for notch in notches)
        if ambiguous and max(shape_counts.values()) < 2:
            continue
        if not ambiguous and max(shape_counts.values()) != 1:
            continue

        frame_patch = np.ones((frame_h, frame_w), dtype=int)
        frame_patch[occupied] = 0
        if len(_components(frame_patch.astype(bool), connectivity=4)) != 1:
            continue
        if len(_extract_notches(frame_patch, bg=0)) != len(notches):
            continue
        return notches
    return None


def _paint(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(bg)
    sub[mask] = patch[mask]


def _rect_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_pieces(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    piece_patches: list[np.ndarray],
    bg: int,
) -> bool:
    h, w = grid.shape
    placed_rects = [frame_bbox]
    order = sorted(range(len(piece_patches)), key=lambda i: piece_patches[i].shape[0] * piece_patches[i].shape[1], reverse=True)
    for idx in order:
        patch = piece_patches[idx]
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(PIECE_ATTEMPTS):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            rect = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
            if any(_rect_too_close(rect, other, gap=1) for other in placed_rects):
                continue
            _paint(grid, patch, r0, c0, bg)
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return False
    return True


def _sample_frame_origin(grid_h: int, grid_w: int, frame_h: int, frame_w: int) -> tuple[int, int]:
    base_r = (grid_h - frame_h) // 2
    base_c = (grid_w - frame_w) // 2
    r_min = 2
    c_min = 2
    r_max = grid_h - frame_h - 2
    c_max = grid_w - frame_w - 2
    jitter_r = random.randint(-1, 1)
    jitter_c = random.randint(-1, 1)
    r0 = min(max(base_r + jitter_r, r_min), r_max)
    c0 = min(max(base_c + jitter_c, c_min), c_max)
    return int(r0), int(c0)


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    frame_h = random.randint(FRAME_H_RANGE[0], min(FRAME_H_RANGE[1], grid_h - 4))
    frame_w = random.randint(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], grid_w - 4))
    ambiguous = bool(random.randrange(2))

    notches = _generate_notches(frame_h, frame_w, ambiguous)
    if notches is None:
        return None

    piece_count = len(notches)
    if piece_count + 2 > 10:
        return None
    colors = list(range(10))
    bg = random.choice(colors)
    remaining = [c for c in colors if c != bg]
    frame_color = random.choice(remaining)
    remaining = [c for c in remaining if c != frame_color]
    piece_colors = random.sample(remaining, piece_count)

    frame_patch = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    expected = frame_patch.copy()
    piece_patches: list[np.ndarray] = []
    for notch, color in zip(notches, piece_colors):
        frame_patch[notch["full_mask"]] = int(bg)
        nr0, nc0, nr1, nc1 = notch["bbox"]
        expected[nr0 : nr1 + 1, nc0 : nc1 + 1][notch["mask"]] = int(color)
        patch = np.full(notch["mask"].shape, int(bg), dtype=int)
        patch[notch["mask"]] = int(color)
        piece_patches.append(patch)

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    fr0, fc0 = _sample_frame_origin(grid_h, grid_w, frame_h, frame_w)
    _paint(grid, frame_patch, fr0, fc0, bg)
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)
    if not _place_pieces(grid, frame_bbox, piece_patches, bg):
        return None

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[int(bg)] != max(counts.values()) or list(counts.values()).count(counts[int(bg)]) != 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
