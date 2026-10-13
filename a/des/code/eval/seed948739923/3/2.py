from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BORDER_COLORS = (1, 2, 3, 4, 5)
SIDE_NAMES = ("top", "right", "bottom", "left")
SIDE_INDEX = {name: i for i, name in enumerate(SIDE_NAMES)}

GEN_BG_POOL = (0, 6, 7, 8, 9)
GEN_MAX_ATTEMPTS = 400
GEN_SIZE_CHOICES = (3, 4, 5, 6)  # inner square sizes
GEN_COUNT_CHOICES = (1, 2, 3)


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _perimeter_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords += [(r, w - 1) for r in range(1, h)]
    coords += [(h - 1, c) for c in range(w - 2, -1, -1)]
    coords += [(r, 0) for r in range(h - 2, 0, -1)]
    return coords


def _side_of_point(r: int, c: int, h: int, w: int) -> str | None:
    if r == 0 and 0 < c < w - 1:
        return "top"
    if c == w - 1 and 0 < r < h - 1:
        return "right"
    if r == h - 1 and 0 < c < w - 1:
        return "bottom"
    if c == 0 and 0 < r < h - 1:
        return "left"
    return None


def _extract_coded_rectangles(grid: np.ndarray) -> list[dict]:
    mask = np.isin(grid, BORDER_COLORS)
    rects: list[dict] = []
    for pts in _components(mask):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue

        submask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            submask[r - r0, c - c0] = True

        perim = _perimeter_coords(h, w)
        if len(pts) != len(perim):
            continue
        if any(not submask[r, c] for r, c in perim):
            continue
        if submask[1:-1, 1:-1].any():
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_vals = [int(sub[r, c]) for r, c in perim]
        counts = Counter(border_vals)
        if set(counts) != set(BORDER_COLORS):
            continue
        count_values = list(counts.values())
        if len(set(count_values)) != len(count_values):
            continue

        light_color = min(counts, key=counts.get)
        light_pts = [(r, c) for r, c in perim if int(sub[r, c]) == light_color]
        if len(light_pts) != 1:
            continue
        up_side = _side_of_point(light_pts[0][0], light_pts[0][1], h, w)
        if up_side is None:
            continue

        inner = sub[1:-1, 1:-1].copy()
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": tuple(map(int, inner.shape)),
                "inner": inner,
                "up_side": up_side,
                "is_hole": bool(np.all(inner == inner[0, 0])),
            }
        )
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble edge-scattered fragments into coded missing rectangles.

    Input structure:
    - bg is the most common color.
    - Several one-pixel-thick multi-color rectangular borders exist.
    - Borders inside the large central picture surround uniform occluder fills (holes).
    - Borders near the outer edges surround non-uniform fragment contents.

    Rule:
    - Pair holes and fragments by inner size.
    - Decode each border's orientation from the side containing its rarest border color.
    - Rotate the fragment interior by the difference between fragment and hole orientations.
    - Fill the hole interior, and erase the original fragment position to bg.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    rects = _extract_coded_rectangles(g)
    holes_by_shape: dict[tuple[int, int], list[dict]] = defaultdict(list)
    frags_by_shape: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rect in rects:
        if rect["is_hole"]:
            holes_by_shape[rect["shape"]].append(rect)
        else:
            frags_by_shape[rect["shape"]].append(rect)

    for shape, holes in holes_by_shape.items():
        frags = frags_by_shape.get(shape, [])
        assert len(holes) == len(frags), "hole/fragment count mismatch for a size class"
        holes = sorted(holes, key=lambda x: x["bbox"])
        frags = sorted(frags, key=lambda x: x["bbox"])
        for hole, frag in zip(holes, frags):
            rot_k = (SIDE_INDEX[frag["up_side"]] - SIDE_INDEX[hole["up_side"]]) % 4
            restored = np.rot90(frag["inner"], rot_k)
            hr0, hc0, hr1, hc1 = hole["bbox"]
            fr0, fc0, fr1, fc1 = frag["bbox"]
            out[hr0 + 1 : hr1, hc0 + 1 : hc1] = restored
            out[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg

    return out


def _pattern_patch(h: int, w: int, colors: list[int], rng: random.Random) -> np.ndarray:
    mode = rng.randrange(4)
    a, b, c = rng.sample((1, 2, 3, 4, 5), 3)
    patch = np.zeros((h, w), dtype=int)
    for r in range(h):
        for col in range(w):
            if mode == 0:
                idx = (a * r + b * col + c * ((r + col) % 2)) % len(colors)
            elif mode == 1:
                idx = ((r // 2) + a * col + (r * col) % 3 + c) % len(colors)
            elif mode == 2:
                idx = ((a * (r % 3)) + (b * (col % 4)) + c * ((r * col) % 2)) % len(colors)
            else:
                idx = (a * r + b * col + c * ((r + 2 * col) % 3)) % len(colors)
            patch[r, col] = colors[int(idx)]
    return patch


def _nontrivial_piece(piece: np.ndarray) -> bool:
    if len(np.unique(piece)) < 2:
        return False
    return all(not np.array_equal(piece, np.rot90(piece, k)) for k in (1, 2, 3))


def _make_coded_patch(inner: np.ndarray, up_side: str, rng: random.Random) -> np.ndarray:
    s = int(inner.shape[0])
    outer = s + 2
    patch = np.zeros((outer, outer), dtype=int)
    patch[1:-1, 1:-1] = inner

    perim = _perimeter_coords(outer, outer)
    start_choices = [i for i, (r, c) in enumerate(perim) if _side_of_point(r, c, outer, outer) == up_side]
    start_idx = rng.choice(start_choices)
    run_lengths = [1, 2, 3, 4, len(perim) - 10]
    idx = start_idx
    for color, run_len in zip(BORDER_COLORS, run_lengths):
        for _ in range(run_len):
            r, c = perim[idx % len(perim)]
            patch[r, c] = color
            idx += 1
    return patch


def _fits_with_padding(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))]) == int(bg)


def _place_holes(
    obj: np.ndarray,
    sizes: list[int],
    occ: np.ndarray,
    occluder: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | None:
    obj_h, obj_w = obj.shape
    inp_obj = obj.copy()
    holes: list[dict] = []
    for size in sizes:
        outer = size + 2
        candidates: list[tuple[int, int, np.ndarray]] = []
        for r0 in range(1, obj_h - outer):
            for c0 in range(1, obj_w - outer):
                if not _fits_with_padding(occ, r0, c0, outer, outer):
                    continue
                piece = obj[r0 + 1 : r0 + 1 + size, c0 + 1 : c0 + 1 + size]
                if not _nontrivial_piece(piece):
                    continue
                candidates.append((r0, c0, piece.copy()))
        if not candidates:
            return None
        r0, c0, piece = rng.choice(candidates)
        hole_side = rng.choice(SIDE_NAMES)
        frame = _make_coded_patch(np.full((size, size), occluder, dtype=int), hole_side, rng)
        inp_obj[r0 : r0 + outer, c0 : c0 + outer] = frame
        occ[r0 : r0 + outer, c0 : c0 + outer] = True
        holes.append(
            {
                "size": size,
                "hole_side": hole_side,
                "piece": piece,
                "frame": frame.copy(),
                "bbox": (r0, c0, r0 + outer - 1, c0 + outer - 1),
            }
        )
    return inp_obj, holes


def _edge_candidates(H: int, W: int, side: str, size: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c0) for c0 in range(W - size + 1)]
    if side == "bottom":
        return [(H - size, c0) for c0 in range(W - size + 1)]
    if side == "left":
        return [(r0, 0) for r0 in range(H - size + 1)]
    return [(r0, W - size) for r0 in range(H - size + 1)]


def _place_fragments(
    grid: np.ndarray,
    occ: np.ndarray,
    holes: list[dict],
    rng: random.Random,
) -> bool:
    H, W = grid.shape
    for hole in holes:
        size = int(hole["size"])
        outer = size + 2
        piece_side = rng.choice(SIDE_NAMES)
        rot_k = (SIDE_INDEX[hole["hole_side"]] - SIDE_INDEX[piece_side]) % 4
        fragment_inner = np.rot90(hole["piece"], rot_k)
        frag_patch = _make_coded_patch(fragment_inner, piece_side, rng)

        side_order = list(SIDE_NAMES)
        rng.shuffle(side_order)
        placed = False
        for side in side_order:
            candidates = _edge_candidates(H, W, side, outer)
            rng.shuffle(candidates)
            for r0, c0 in candidates:
                if not _fits_with_padding(occ, r0, c0, outer, outer):
                    continue
                grid[r0 : r0 + outer, c0 : c0 + outer] = frag_patch
                occ[r0 : r0 + outer, c0 : c0 + outer] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    H = rng.randint(20, 29)
    W = rng.randint(20, 29)
    bg = rng.choice(GEN_BG_POOL)

    remaining = [c for c in GEN_BG_POOL if c != bg]
    rng.shuffle(remaining)
    occluder = remaining[0]
    object_colors = remaining[1:]
    if len(object_colors) < 3:
        return None

    count = rng.choice(GEN_COUNT_CHOICES)
    sizes = sorted(rng.sample(GEN_SIZE_CHOICES, count))
    max_outer = max(sizes) + 2

    min_obj = max(10, 2 * max_outer + 1)
    if H - 4 < min_obj or W - 4 < min_obj:
        return None

    obj_h_hi = min(H - 4, 18)
    obj_w_hi = min(W - 4, 18)
    if min_obj > obj_h_hi or min_obj > obj_w_hi:
        return None
    obj_h = rng.randint(min_obj, obj_h_hi)
    obj_w = rng.randint(min_obj, obj_w_hi)
    obj_r0 = rng.randint(2, H - obj_h - 2)
    obj_c0 = rng.randint(2, W - obj_w - 2)

    obj = _pattern_patch(obj_h, obj_w, object_colors[:3], rng)
    hole_occ = np.zeros((obj_h, obj_w), dtype=bool)
    hole_result = _place_holes(obj, sizes, hole_occ, occluder, rng)
    if hole_result is None:
        return None
    inp_obj, holes = hole_result

    expected = np.full((H, W), bg, dtype=int)
    expected[obj_r0 : obj_r0 + obj_h, obj_c0 : obj_c0 + obj_w] = obj
    input_grid = expected.copy()
    input_grid[obj_r0 : obj_r0 + obj_h, obj_c0 : obj_c0 + obj_w] = inp_obj

    for hole in holes:
        local_r0, local_c0, local_r1, local_c1 = hole["bbox"]
        size = int(hole["size"])
        expected[obj_r0 + local_r0 : obj_r0 + local_r1 + 1, obj_c0 + local_c0 : obj_c0 + local_c1 + 1] = hole["frame"]
        expected[
            obj_r0 + local_r0 + 1 : obj_r0 + local_r0 + 1 + size,
            obj_c0 + local_c0 + 1 : obj_c0 + local_c0 + 1 + size,
        ] = hole["piece"]

    occ = np.zeros((H, W), dtype=bool)
    occ[input_grid != bg] = True
    if not _place_fragments(input_grid, occ, holes, rng):
        return None

    if not _bg_is_dominant(input_grid, bg):
        return None

    try:
        solved = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return input_grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate valid input")
