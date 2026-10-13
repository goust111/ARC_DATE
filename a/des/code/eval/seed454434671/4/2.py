from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 300
PLACE_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        rs = coords[:, 0]
        cs = coords[:, 1]
        return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[int(r) - r0, int(c) - c0] = True
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _unique_orientations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    base = np.asarray(mask, dtype=bool)
    for flipped in (False, True):
        cur = np.fliplr(base) if flipped else base
        for _ in range(4):
            key = (cur.tobytes(), tuple(map(int, cur.shape)))
            if key not in seen:
                seen.add(key)
                out.append(cur.copy())
            cur = np.rot90(cur)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    return min((m.tobytes(), tuple(map(int, m.shape))) for m in _unique_orientations(mask))


def _largest_component(mask: np.ndarray) -> list[tuple[int, int]]:
    comps = _components(mask)
    assert comps
    comps.sort(key=lambda comp: (-len(comp), _bbox(comp)))
    return comps[0]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = map(int, mask.shape)
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _extract_template(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    template = _largest_component(grid == GREY)
    r0, c0, r1, c1 = _bbox(template)
    return (int(r0), int(c0), int(r1), int(c1)), np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _extract_holes(crop: np.ndarray) -> list[dict]:
    bg_mask = crop == BG
    holes: list[dict] = []
    for comp in _components(bg_mask):
        if any(
            r == 0 or c == 0 or r == crop.shape[0] - 1 or c == crop.shape[1] - 1
            for r, c in comp
        ):
            continue
        mask = _mask_from_coords(comp)
        holes.append(
            {
                "coords": [(int(r), int(c)) for r, c in comp],
                "mask": mask,
                "key": _canonical_key(mask),
                "area": int(mask.sum()),
            }
        )
    assert holes
    return holes


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != GREY)
    pieces: list[dict] = []
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = next(iter(colors))
        piece_mask = _mask_from_coords(comp)
        pieces.append(
            {
                "coords": [(int(r), int(c)) for r, c in comp],
                "mask": piece_mask,
                "key": _canonical_key(piece_mask),
                "area": int(piece_mask.sum()),
                "color": int(color),
            }
        )
    assert pieces
    return pieces


def _match_by_shape(holes: list[dict], pieces: list[dict]) -> list[dict]:
    hole_by_key: dict[tuple[bytes, tuple[int, int]], list[dict]] = {}
    piece_by_key: dict[tuple[bytes, tuple[int, int]], list[dict]] = {}
    for hole in holes:
        hole_by_key.setdefault(hole["key"], []).append(hole)
    for piece in pieces:
        piece_by_key.setdefault(piece["key"], []).append(piece)
    assert set(hole_by_key) == set(piece_by_key)

    matched: list[dict] = []
    for key in sorted(hole_by_key, key=lambda item: (item[1], item[0])):
        hs = hole_by_key[key]
        ps = piece_by_key[key]
        assert len(hs) == len(ps) == 1, "shape matching must be unique"
        hole = hs[0]
        piece = ps[0]
        matched.append(
            {
                "hole_coords": hole["coords"],
                "area": int(hole["area"]),
                "orig_color": int(piece["color"]),
            }
        )
    return matched


def _adjacency_from_crop(crop: np.ndarray, placed_id: np.ndarray, count: int) -> list[set[int]]:
    h, w = map(int, crop.shape)
    adj = [set() for _ in range(count)]
    for r in range(h):
        for c in range(w):
            if int(crop[r, c]) != GREY:
                continue
            if 0 < c < w - 1:
                a = int(placed_id[r, c - 1])
                b = int(placed_id[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    adj[a].add(b)
                    adj[b].add(a)
            if 0 < r < h - 1:
                a = int(placed_id[r - 1, c])
                b = int(placed_id[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    adj[a].add(b)
                    adj[b].add(a)
    return adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered pieces to template holes up to rotation/flip, place them back,
    then recolor smaller pieces to a larger adjacent neighbor across a one-cell grey wall.
    """
    g = np.asarray(grid, dtype=int)
    _template_bbox, crop = _extract_template(g)
    holes = _extract_holes(crop)
    pieces = _extract_pieces(g)
    assert len(holes) == len(pieces)

    matched = _match_by_shape(holes, pieces)
    placed_id = np.full(crop.shape, -1, dtype=int)
    out = crop.copy()
    areas: list[int] = []
    orig_colors: list[int] = []
    for idx, item in enumerate(matched):
        areas.append(int(item["area"]))
        orig_colors.append(int(item["orig_color"]))
        for r, c in item["hole_coords"]:
            placed_id[r, c] = int(idx)

    adj = _adjacency_from_crop(crop, placed_id, len(matched))
    final_colors = orig_colors[:]
    for i in range(len(matched)):
        larger = [j for j in adj[i] if areas[j] > areas[i] and orig_colors[j] != orig_colors[i]]
        if not larger:
            continue
        max_area = max(areas[j] for j in larger)
        best = sorted(j for j in larger if areas[j] == max_area)
        chosen_colors = {orig_colors[j] for j in best}
        if len(chosen_colors) == 1:
            final_colors[i] = int(next(iter(chosen_colors)))

    for idx, item in enumerate(matched):
        for r, c in item["hole_coords"]:
            out[r, c] = int(final_colors[idx])
    return out


def _random_connected_shape(area: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, min(4, area))
        w = random.randint(2, min(4, area))
        if area > h * w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            pick = random.choice(tuple(frontier))
            mask[pick] = True
        if int(mask.sum()) != area:
            continue
        trimmed = _trim_mask(mask)
        th, tw = map(int, trimmed.shape)
        if min(th, tw) < 2 or max(th, tw) > 4:
            continue
        if len(_components(trimmed)) != 1:
            continue
        return trimmed
    return None


def _area_sequence(n: int) -> tuple[list[int], str]:
    mode = random.choice(["strict", "equal_pair"])
    if mode == "strict":
        vals = sorted(random.sample(range(4, 9), n))
        return vals, mode

    dup_at = random.randrange(n - 1)
    base = sorted(random.sample(range(4, 9), n - 1))
    vals: list[int] = []
    for i, v in enumerate(base):
        vals.append(int(v))
        if i == dup_at:
            vals.append(int(v))
    return vals[:n], mode


def _make_shapes(areas: list[int]) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    used: set[tuple[bytes, tuple[int, int]]] = set()
    for area in areas:
        shape = None
        for _ in range(SHAPE_ATTEMPTS):
            cand = _random_connected_shape(int(area))
            if cand is None:
                continue
            key = _canonical_key(cand)
            if key in used:
                continue
            shape = cand
            used.add(key)
            break
        if shape is None:
            return None
        shapes.append(shape)
    return shapes


def _boundary_rows(mask: np.ndarray, side: str) -> list[int]:
    if side == "left":
        return [int(r) for r in range(mask.shape[0]) if bool(mask[r, 0])]
    if side == "right":
        return [int(r) for r in range(mask.shape[0]) if bool(mask[r, -1])]
    raise AssertionError("invalid side")


def _boundary_cols(mask: np.ndarray, side: str) -> list[int]:
    if side == "top":
        return [int(c) for c in range(mask.shape[1]) if bool(mask[0, c])]
    if side == "bottom":
        return [int(c) for c in range(mask.shape[1]) if bool(mask[-1, c])]
    raise AssertionError("invalid side")


def _build_template(shapes: list[np.ndarray], mode: str) -> tuple[np.ndarray, list[dict]] | None:
    n = len(shapes)
    orientation = random.choice(["horizontal", "vertical"])
    gaps = [random.choice([1, 2]) for _ in range(n - 1)]
    if mode == "strict" and all(gap != 1 for gap in gaps):
        gaps[random.randrange(n - 1)] = 1
    if mode == "equal_pair":
        equal_edges = [i for i in range(n - 1) if int(shapes[i].sum()) == int(shapes[i + 1].sum())]
        if not equal_edges:
            return None
        gaps[random.choice(equal_edges)] = 1

    holes: list[dict] = []
    if orientation == "horizontal":
        c_positions = [1]
        for i in range(1, n):
            c_positions.append(c_positions[-1] + shapes[i - 1].shape[1] + gaps[i - 1])
        r_positions = [0]
        for i in range(1, n):
            if gaps[i - 1] == 1:
                left_rows = _boundary_rows(shapes[i - 1], "right")
                right_rows = _boundary_rows(shapes[i], "left")
                rr0 = random.choice(left_rows)
                rr1 = random.choice(right_rows)
                r_positions.append(r_positions[-1] + rr0 - rr1)
            else:
                r_positions.append(r_positions[-1] + random.randint(-2, 2))
        min_r = min(r_positions)
        r_positions = [r - min_r + 1 for r in r_positions]
        height = max(r + shape.shape[0] for r, shape in zip(r_positions, shapes, strict=False)) + 1
        width = c_positions[-1] + shapes[-1].shape[1] + 1
        if height > 13 or width > 13:
            return None
        crop = np.full((height, width), GREY, dtype=int)
        for i, shape in enumerate(shapes):
            holes.append({"mask": shape, "r0": int(r_positions[i]), "c0": int(c_positions[i])})
    else:
        r_positions = [1]
        for i in range(1, n):
            r_positions.append(r_positions[-1] + shapes[i - 1].shape[0] + gaps[i - 1])
        c_positions = [0]
        for i in range(1, n):
            if gaps[i - 1] == 1:
                top_cols = _boundary_cols(shapes[i - 1], "bottom")
                bottom_cols = _boundary_cols(shapes[i], "top")
                cc0 = random.choice(top_cols)
                cc1 = random.choice(bottom_cols)
                c_positions.append(c_positions[-1] + cc0 - cc1)
            else:
                c_positions.append(c_positions[-1] + random.randint(-2, 2))
        min_c = min(c_positions)
        c_positions = [c - min_c + 1 for c in c_positions]
        height = r_positions[-1] + shapes[-1].shape[0] + 1
        width = max(c + shape.shape[1] for c, shape in zip(c_positions, shapes, strict=False)) + 1
        if height > 13 or width > 13:
            return None
        crop = np.full((height, width), GREY, dtype=int)
        for i, shape in enumerate(shapes):
            holes.append({"mask": shape, "r0": int(r_positions[i]), "c0": int(c_positions[i])})

    for hole in holes:
        _paint_mask(crop, hole["mask"], hole["r0"], hole["c0"], BG)

    expected_edges: set[tuple[int, int]] = set()
    for i, gap in enumerate(gaps):
        if gap == 1:
            expected_edges.add((i, i + 1))

    hole_grid = np.full(crop.shape, -1, dtype=int)
    for idx, hole in enumerate(holes):
        mask = hole["mask"]
        r0 = int(hole["r0"])
        c0 = int(hole["c0"])
        for rr, cc in np.argwhere(mask):
            hole_grid[r0 + int(rr), c0 + int(cc)] = int(idx)

    actual_edges: set[tuple[int, int]] = set()
    adj = _adjacency_from_crop(crop, hole_grid, len(holes))
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            actual_edges.add((min(i, j), max(i, j)))
    if actual_edges != expected_edges:
        return None

    extracted = _extract_holes(crop)
    if len(extracted) != len(holes):
        return None
    return crop, holes


def _dilated(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _scatter_pieces(
    grid: np.ndarray,
    zone_c0: int,
    shapes: list[np.ndarray],
    colors: list[int],
) -> bool:
    h, w = map(int, grid.shape)
    occupied = grid != BG
    transformed_any = False

    order = list(range(len(shapes)))
    random.shuffle(order)
    order.sort(key=lambda i: int(shapes[i].sum()), reverse=True)

    chosen_masks: dict[int, np.ndarray] = {}
    for i, shape in enumerate(shapes):
        orientations = _unique_orientations(shape)
        mask = random.choice(orientations).copy()
        if not np.array_equal(mask, shape):
            transformed_any = True
        chosen_masks[i] = mask
    if not transformed_any:
        candidates = [i for i, shape in enumerate(shapes) if len(_unique_orientations(shape)) > 1]
        if candidates:
            i = random.choice(candidates)
            alts = [m for m in _unique_orientations(shapes[i]) if not np.array_equal(m, shapes[i])]
            if alts:
                chosen_masks[i] = random.choice(alts).copy()

    for idx in order:
        mask = chosen_masks[idx]
        ph, pw = map(int, mask.shape)
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(zone_c0, w - pw)
            if c0 < zone_c0:
                continue
            test = np.zeros_like(grid, dtype=bool)
            test[r0 : r0 + ph, c0 : c0 + pw][mask] = True
            if bool((_dilated(occupied) & test).any()):
                continue
            _paint_mask(grid, mask, r0, c0, colors[idx])
            occupied |= test
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 4)
    areas, mode = _area_sequence(n)
    if random.random() < 0.5:
        areas = list(reversed(areas))

    shapes = _make_shapes(areas)
    if shapes is None:
        return None
    template = _build_template(shapes, mode)
    if template is None:
        return None
    crop, holes = template

    colors = random.sample(list(PIECE_COLORS), n)

    expected = crop.copy()
    hole_info: list[dict] = []
    for hole, color in zip(holes, colors, strict=False):
        area = int(hole["mask"].sum())
        hole_info.append(
            {
                "coords": [
                    (int(hole["r0"]) + int(r), int(hole["c0"]) + int(c))
                    for r, c in np.argwhere(hole["mask"])
                ],
                "area": area,
                "orig_color": int(color),
            }
        )

    hole_grid = np.full(crop.shape, -1, dtype=int)
    for idx, item in enumerate(hole_info):
        for r, c in item["coords"]:
            hole_grid[r, c] = int(idx)
    adj = _adjacency_from_crop(crop, hole_grid, n)
    final_colors = [int(item["orig_color"]) for item in hole_info]
    for i in range(n):
        larger = [j for j in adj[i] if hole_info[j]["area"] > hole_info[i]["area"]]
        if len(larger) > 1:
            return None
        if larger:
            final_colors[i] = int(hole_info[larger[0]]["orig_color"])
    for idx, item in enumerate(hole_info):
        for r, c in item["coords"]:
            expected[r, c] = int(final_colors[idx])

    th, tw = map(int, crop.shape)
    max_piece_w = max(int(shape.shape[1]) for shape in shapes)
    left_margin = random.randint(1, 2)
    gap_between = random.randint(3, 4)
    right_zone_w = random.randint(max(8, max_piece_w + 4), 12)
    grid_h = random.randint(max(th + 2, 15), 22)
    grid_w = left_margin + tw + gap_between + right_zone_w + 1
    if grid_w > 30:
        return None
    template_r0 = random.randint(1, grid_h - th - 1)
    template_c0 = left_margin
    if template_c0 + tw - 1 >= grid_w // 2:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[template_r0 : template_r0 + th, template_c0 : template_c0 + tw] = crop
    zone_c0 = max(template_c0 + tw + gap_between, grid_w // 2)
    if zone_c0 >= grid_w - 1:
        return None
    if not _scatter_pieces(grid, zone_c0, shapes, colors):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
