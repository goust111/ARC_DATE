from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 120
LAYOUT_ATTEMPTS = 80
CANVAS_ATTEMPTS = 60


def _boundary_majority_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    top = g[0, :]
    bottom = g[-1, :]
    left = g[:, 0]
    right = g[:, -1]
    vals = list(map(int, np.concatenate([top, bottom, left, right])))
    return int(Counter(vals).most_common(1)[0][0])


def _binary_components(mask: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    h, w = map(int, m.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not m[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox(arr)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for k in range(4):
        rot = np.rot90(np.asarray(mask, dtype=bool), k)
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(variants)


def _is_connected(mask: np.ndarray) -> bool:
    m = np.asarray(mask, dtype=bool)
    return int(m.sum()) > 0 and len(_binary_components(m)) == 1


def _has_internal_hole(mask: np.ndarray) -> bool:
    m = np.asarray(mask, dtype=bool)
    h, w = map(int, m.shape)
    empty = ~np.pad(m, 1, constant_values=False)
    seen = np.zeros_like(empty, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if not m[r - 1, c - 1] and not seen[r, c]:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each black chamber inside the largest grey template with the color of the
    uniquely shape-matching outside piece, allowing only rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _boundary_majority_color(g)

    comps = _monochrome_components(g, bg)
    assert comps, "input must contain a non-background template"
    frame = max(comps, key=lambda comp: int(comp["area"]))
    frame_color = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])

    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    hole_mask = crop == int(bg)
    holes = []
    for coords in _binary_components(hole_mask):
        mask, (hr0, hc0, _hr1, _hc1) = _coords_to_mask(coords)
        holes.append({"mask": mask, "r0": int(hr0), "c0": int(hc0)})
    assert holes, "template must contain at least one chamber"

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    pieces_by_key: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in _monochrome_components(outside, bg):
        mask, _ = _coords_to_mask(comp["coords"])
        key = _canonical_rotation_key(mask)
        if key not in pieces_by_key:
            pieces_by_key[key] = int(comp["color"])

    out = crop.copy()
    for hole in holes:
        key = _canonical_rotation_key(hole["mask"])
        assert key in pieces_by_key, "every chamber must have a matching piece"
        color = int(pieces_by_key[key])
        h, w = map(int, hole["mask"].shape)
        region = out[hole["r0"] : hole["r0"] + h, hole["c0"] : hole["c0"] + w]
        region[hole["mask"]] = color

    # The template crop should only contain the frame color and filled chamber colors.
    out[crop != frame_color] = out[crop != frame_color]
    return out


def _random_cavity_shape() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        target_area = random.randint(5, 10)
        dims = [
            (h, w)
            for h in range(3, 6)
            for w in range(3, 7)
            if target_area + 2 <= h * w <= target_area + 6
        ]
        if not dims:
            continue
        h, w = random.choice(dims)
        mask = np.ones((h, w), dtype=bool)
        remove_steps = int(h * w - target_area)
        touched_sides: set[str] = set()
        ok = True
        for _step in range(remove_steps):
            options = []
            for r in range(h):
                for c in range(w):
                    if not mask[r, c]:
                        continue
                    if r not in (0, h - 1) and c not in (0, w - 1):
                        continue
                    cand = mask.copy()
                    cand[r, c] = False
                    if not cand.any(axis=0).all() or not cand.any(axis=1).all():
                        continue
                    if not cand[0].any() or not cand[-1].any():
                        continue
                    if not cand[:, 0].any() or not cand[:, -1].any():
                        continue
                    if not _is_connected(cand) or _has_internal_hole(cand):
                        continue
                    sides = set()
                    if r == 0:
                        sides.add("top")
                    if r == h - 1:
                        sides.add("bottom")
                    if c == 0:
                        sides.add("left")
                    if c == w - 1:
                        sides.add("right")
                    options.append((r, c, sides))
            if not options:
                ok = False
                break
            r, c, sides = random.choice(options)
            mask[r, c] = False
            touched_sides |= sides

        row_sums = tuple(map(int, mask.sum(axis=1)))
        col_sums = tuple(map(int, mask.sum(axis=0)))
        if (
            ok
            and len(touched_sides) >= 2
            and int(mask.sum()) == target_area
            and not _has_internal_hole(mask)
            and _is_connected(mask)
            and (len(set(row_sums)) > 1 or len(set(col_sums)) > 1)
        ):
            return mask
    return None


def _place_shapes_interior(
    shapes: list[np.ndarray], interior_h: int, interior_w: int
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((interior_h, interior_w), dtype=bool)
    order = sorted(
        range(len(shapes)),
        key=lambda idx: int(shapes[idx].shape[0] * shapes[idx].shape[1]),
        reverse=True,
    )
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    for idx in order:
        mask = shapes[idx]
        h, w = map(int, mask.shape)
        candidates = [(r, c) for r in range(interior_h - h + 1) for c in range(interior_w - w + 1)]
        random.shuffle(candidates)
        placed = None
        for r, c in candidates:
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(interior_h, r + h + 1)
            c1 = min(interior_w, c + w + 1)
            if blocked[r0:r1, c0:c1].any():
                continue
            blocked[r : r + h, c : c + w] = True
            placed = (int(r), int(c))
            break
        if placed is None:
            return None
        positions[idx] = placed
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _build_frame_layout(shapes: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    total_area = sum(int(mask.sum()) for mask in shapes)
    max_h = max(int(mask.shape[0]) for mask in shapes)
    max_w = max(int(mask.shape[1]) for mask in shapes)
    need = Counter(_canonical_rotation_key(mask) for mask in shapes)

    for _ in range(LAYOUT_ATTEMPTS):
        interior_h = random.randint(max(max_h + 1, 7), 13)
        interior_w = random.randint(max(max_w + 1, 8), 15)
        density = total_area / float(interior_h * interior_w)
        if not (0.35 <= density <= 0.62):
            continue
        positions = _place_shapes_interior(shapes, interior_h, interior_w)
        if positions is None:
            continue

        frame = np.full((interior_h + 2, interior_w + 2), FRAME_COLOR, dtype=int)
        for mask, (r, c) in zip(shapes, positions):
            h, w = map(int, mask.shape)
            region = frame[1 + r : 1 + r + h, 1 + c : 1 + c + w]
            region[mask] = BG_COLOR

        if len(_binary_components(frame == FRAME_COLOR)) != 1:
            continue

        holes = _binary_components(frame == BG_COLOR)
        if len(holes) != len(shapes):
            continue

        got = Counter()
        for coords in holes:
            mask, _ = _coords_to_mask(coords)
            got[_canonical_rotation_key(mask)] += 1
        if got != need:
            continue

        return frame, positions
    return None


def _place_on_canvas(frame: np.ndarray, pieces: list[np.ndarray]) -> np.ndarray | None:
    frame_h, frame_w = map(int, frame.shape)
    min_h = max(18, frame_h + 6)
    min_w = max(18, frame_w + 6)
    if min_h > 30 or min_w > 30:
        return None

    order = sorted(
        range(len(pieces)),
        key=lambda idx: int(pieces[idx].shape[0] * pieces[idx].shape[1]),
        reverse=True,
    )

    for _ in range(CANVAS_ATTEMPTS):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        grid = np.full((h, w), BG_COLOR, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        fr = (h - frame_h) // 2 + random.randint(-2, 2)
        fc = (w - frame_w) // 2 + random.randint(-2, 2)
        fr = min(max(1, fr), h - frame_h - 1)
        fc = min(max(1, fc), w - frame_w - 1)

        grid[fr : fr + frame_h, fc : fc + frame_w] = frame
        blocked[fr : fr + frame_h, fc : fc + frame_w] = True

        ok = True
        for idx in order:
            patch = pieces[idx]
            ph, pw = map(int, patch.shape)
            candidates = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
            random.shuffle(candidates)
            placed = None
            for r, c in candidates:
                r0 = max(0, r - 1)
                c0 = max(0, c - 1)
                r1 = min(h, r + ph + 1)
                c1 = min(w, c + pw + 1)
                if blocked[r0:r1, c0:c1].any():
                    continue
                placed = (int(r), int(c))
                blocked[r : r + ph, c : c + pw] = True
                break
            if placed is None:
                ok = False
                break
            r, c = placed
            region = grid[r : r + ph, c : c + pw]
            region[patch != BG_COLOR] = patch[patch != BG_COLOR]

        if ok:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(3, 5)
    shapes: list[np.ndarray] = []
    seen = set()
    for _ in range(200):
        if len(shapes) == room_count:
            break
        shape = _random_cavity_shape()
        if shape is None:
            continue
        key = _canonical_rotation_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
    if len(shapes) != room_count:
        return None

    area_counts = Counter(int(mask.sum()) for mask in shapes)
    box_counts = Counter(tuple(map(int, mask.shape)) for mask in shapes)
    if max(area_counts.values(), default=0) < 2 and max(box_counts.values(), default=0) < 2:
        return None

    layout = _build_frame_layout(shapes)
    if layout is None:
        return None
    frame, positions = layout

    colors = random.sample(PIECE_COLORS, len(shapes))
    rotations = [random.randrange(4) for _ in shapes]
    if all(rot == 0 for rot in rotations):
        rotations[random.randrange(len(rotations))] = random.choice([1, 2, 3])

    expected = frame.copy()
    pieces: list[np.ndarray] = []
    for mask, (r, c), color, rot in zip(shapes, positions, colors, rotations):
        h, w = map(int, mask.shape)
        region = expected[1 + r : 1 + r + h, 1 + c : 1 + c + w]
        region[mask] = int(color)

        rot_mask = np.rot90(mask, rot)
        patch = np.full(rot_mask.shape, BG_COLOR, dtype=int)
        patch[rot_mask] = int(color)
        pieces.append(patch)

    grid = _place_on_canvas(frame, pieces)
    if grid is None:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a black background with one central grey multi-chamber template and
    several rotated monochrome pieces scattered outside it.
    """
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
