from __future__ import annotations

from collections import defaultdict

import numpy as np


SIDE_NAMES = ("top", "right", "bottom", "left")
ROT_CCW_BY_SIDE = {"top": 0, "right": 3, "bottom": 2, "left": 1}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _mask_and_bbox(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _set_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nb in _neighbors4(r, c):
            if nb in coords and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(coords)


def _perimeter_set(h: int, w: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(w):
        pts.add((0, c))
        pts.add((h - 1, c))
    for r in range(h):
        pts.add((r, 0))
        pts.add((r, w - 1))
    return pts


def _dominant_side(missing: set[tuple[int, int]], h: int, w: int) -> str | None:
    counts = [
        sum(r == 0 and 0 < c < w - 1 for r, c in missing),
        sum(c == w - 1 and 0 < r < h - 1 for r, c in missing),
        sum(r == h - 1 and 0 < c < w - 1 for r, c in missing),
        sum(c == 0 and 0 < r < h - 1 for r, c in missing),
    ]
    best = max(counts)
    if best <= 0 or counts.count(best) != 1:
        return None
    active = [idx for idx, value in enumerate(counts) if value > 0]
    if len(active) > 2:
        return None
    if len(active) == 2:
        a, b = active
        if abs(a - b) not in (1, 3):
            return None
    return SIDE_NAMES[int(np.argmax(counts))]


def _detect_walls(grid: np.ndarray, bg: int) -> list[dict]:
    walls: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            if len(coords) < 6:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue

            rel = {(r - r0, c - c0) for r, c in coords}
            if any(rr not in (0, h - 1) and cc not in (0, w - 1) for rr, cc in rel):
                continue

            perimeter = _perimeter_set(h, w)
            missing = perimeter - rel
            if not missing or not _set_connected(missing):
                continue

            side = _dominant_side(missing, h, w)
            if side is None:
                continue

            inside = []
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    value = int(grid[rr, cc])
                    if value != bg:
                        inside.append((rr, cc, value))
            if len(inside) != 1 or inside[0][2] == color:
                continue

            gap_mask, gap_bbox_rel = _mask_and_bbox([(rr, cc) for rr, cc in missing])
            gr0, gc0, gr1, gc1 = gap_bbox_rel
            walls.append(
                {
                    "wall_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "marker_pos": (int(inside[0][0]), int(inside[0][1])),
                    "piece_color": int(inside[0][2]),
                    "gap_bbox": (r0 + gr0, c0 + gc0, r0 + gr1, c0 + gc1),
                    "gap_mask": gap_mask,
                }
            )
    return walls


def _component_records_by_color(grid: np.ndarray, bg: int) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            mask, bbox = _mask_and_bbox(coords)
            out[color].append(
                {
                    "coords": coords,
                    "coord_set": set(coords),
                    "mask": mask,
                    "bbox": bbox,
                    "area": len(coords),
                }
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored fragment with the wall whose inner marker has the same color,
    rotate the fragment according to the missing side, and move it into the gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    walls = _detect_walls(g, bg)
    if not walls:
        return g.copy()

    comps_by_color = _component_records_by_color(g, bg)
    out = g.copy()

    for wall in walls:
        piece_color = int(wall["piece_color"])
        marker_pos = tuple(wall["marker_pos"])
        candidates = []
        for comp in comps_by_color.get(piece_color, []):
            if marker_pos in comp["coord_set"]:
                continue
            candidates.append(comp)
        assert len(candidates) == 1, "each marker color must identify exactly one external fragment"
        piece = candidates[0]

        rot_k = ROT_CCW_BY_SIDE[str(wall["side"])]
        rotated = np.rot90(piece["mask"], rot_k)
        gap_mask = wall["gap_mask"]
        assert rotated.shape == gap_mask.shape and np.array_equal(rotated, gap_mask)

        for r, c in piece["coords"]:
            out[r, c] = bg
        mr, mc = marker_pos
        out[mr, mc] = bg

        gr0, gc0, _gr1, _gc1 = wall["gap_bbox"]
        target = out[gr0 : gr0 + rotated.shape[0], gc0 : gc0 + rotated.shape[1]]
        target[rotated] = piece_color

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _sample_canonical_gap(rng: np.random.Generator) -> np.ndarray:
    base_h = int(rng.integers(5, 9))
    base_w = int(rng.integers(6, 11))
    full = np.zeros((base_h, base_w), dtype=bool)

    hand = str(rng.choice(["left", "right"]))
    min_top = max(3, (base_w + 1) // 2)
    top_len = int(rng.integers(min_top, base_w))
    max_leg = min(base_h - 2, top_len - 2)
    leg_len = int(rng.integers(1, max_leg + 1))

    if hand == "left":
        full[0, :top_len] = True
        full[1 : leg_len + 1, 0] = True
    else:
        full[0, base_w - top_len :] = True
        full[1 : leg_len + 1, -1] = True

    return full


def _placeable(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], pad: int) -> bool:
    r0, c0, r1, c1 = box
    for er0, ec0, er1, ec1 in boxes:
        if not (r1 + pad < er0 or er1 + pad < r0 or c1 + pad < ec0 or ec1 + pad < c0):
            return False
    return True


def _random_bbox_position(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    obj_h: int,
    obj_w: int,
    boxes: list[tuple[int, int, int, int]],
    pad: int,
    margin: int,
) -> tuple[int, int] | None:
    if obj_h + 2 * margin > grid_h or obj_w + 2 * margin > grid_w:
        return None
    max_r = grid_h - margin - obj_h
    max_c = grid_w - margin - obj_w
    for _ in range(120):
        r0 = int(rng.integers(margin, max_r + 1))
        c0 = int(rng.integers(margin, max_c + 1))
        box = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
        if _placeable(box, boxes, pad):
            return r0, c0
    return None


def _paste_color(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    sub[mask] = patch[mask]


def _build_wall_patches(
    rng: np.random.Generator, side: str, wall_color: int, piece_color: int, bg: int
) -> dict:
    canonical_full = _sample_canonical_gap(rng)
    piece_mask = _crop_mask(canonical_full)
    gap_full = np.rot90(canonical_full, ROT_CCW_BY_SIDE[side])

    wall_h, wall_w = gap_full.shape
    wall = np.full((wall_h, wall_w), bg, dtype=int)
    wall[0, :] = wall_color
    wall[-1, :] = wall_color
    wall[:, 0] = wall_color
    wall[:, -1] = wall_color
    wall[gap_full] = bg

    marker_r = int(rng.integers(1, wall_h - 1))
    marker_c = int(rng.integers(1, wall_w - 1))

    inp_wall = wall.copy()
    inp_wall[marker_r, marker_c] = piece_color

    out_wall = wall.copy()
    out_wall[gap_full] = piece_color

    piece = np.full(piece_mask.shape, bg, dtype=int)
    piece[piece_mask] = piece_color

    return {
        "input_wall": inp_wall,
        "output_wall": out_wall,
        "piece": piece,
    }


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_walls = int(rng.integers(2, 5))
    min_side = 15 + 3 * n_walls
    grid_h = int(rng.integers(min_side, min_side + 8))
    grid_w = int(rng.integers(min_side, min_side + 8))

    palette = [int(x) for x in rng.choice(np.arange(10), size=n_walls + 2, replace=False)]
    bg = palette[0]
    wall_color = palette[1]
    piece_colors = palette[2:]

    side_order = list(SIDE_NAMES)
    rng.shuffle(side_order)
    sides = side_order[:n_walls]

    objects = [
        _build_wall_patches(rng, side, wall_color, piece_color, bg)
        for side, piece_color in zip(sides, piece_colors)
    ]
    objects.sort(
        key=lambda obj: max(
            obj["input_wall"].shape[0] * obj["input_wall"].shape[1],
            obj["piece"].shape[0] * obj["piece"].shape[1],
        ),
        reverse=True,
    )

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = np.full((grid_h, grid_w), bg, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    for obj in objects:
        wh, ww = obj["input_wall"].shape
        pos = _random_bbox_position(rng, grid_h, grid_w, wh, ww, boxes, pad=2, margin=1)
        if pos is None:
            return None
        wr, wc = pos
        _paste_color(inp, obj["input_wall"], wr, wc, bg)
        _paste_color(out, obj["output_wall"], wr, wc, bg)
        boxes.append((wr, wc, wr + wh - 1, wc + ww - 1))

    for obj in objects:
        ph, pw = obj["piece"].shape
        pos = _random_bbox_position(rng, grid_h, grid_w, ph, pw, boxes, pad=1, margin=1)
        if pos is None:
            return None
        pr, pc = pos
        _paste_color(inp, obj["piece"], pr, pc, bg)
        boxes.append((pr, pc, pr + ph - 1, pc + pw - 1))

    bg_count = int(np.count_nonzero(inp == bg))
    if bg_count <= inp.size // 2:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
