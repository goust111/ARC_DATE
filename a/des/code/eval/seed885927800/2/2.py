from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
FILL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 30)
TEMPLATE_COUNT_CHOICES = (1, 2, 3)
MARGIN_RANGE = (1, 3)
EDGE_MARGIN = 1
OBJECT_PAD = 1
GEN_ATTEMPTS = 400


RAW_SHAPES = [
    [[1, 0], [1, 0], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 0], [1, 1], [0, 1]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1], [1, 1], [1, 0]],
    [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
    [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
    [[0, 1, 1], [1, 1, 0], [1, 0, 0]],
    [[1, 1, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 1, 1, 0], [0, 0, 1, 1]],
    [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


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
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_points(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for k in range(4):
        cur = np.rot90(mask, k)
        keys.append((cur.shape, cur.tobytes()))
    return min(keys)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        cur = np.rot90(mask, k)
        key = (cur.shape, cur.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(cur)
    return out


def _is_rectangular_border_component(
    g: np.ndarray, points: list[tuple[int, int]], color: int
) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    perimeter = 2 * (h + w) - 4
    if len(points) != perimeter:
        return False
    point_set = set(points)
    for r, c in points:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border and (rr, cc) not in point_set:
                return False
            if on_border and int(g[rr, cc]) != int(color):
                return False
    return True


def _find_templates(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    templates: list[dict] = []
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    for color in colors:
        for comp in _components(g == int(color)):
            if not _is_rectangular_border_component(g, comp, int(color)):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            holes = []
            for hole in _components(inner == BG):
                if any(
                    rr == 0
                    or cc == 0
                    or rr == inner.shape[0] - 1
                    or cc == inner.shape[1] - 1
                    for rr, cc in hole
                ):
                    continue
                holes.append(hole)
            if len(holes) != 1:
                continue
            hole = holes[0]
            hole_abs = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in hole]
            templates.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "hole_points": hole_abs,
                    "key": _rotation_key(_mask_from_points(hole_abs)),
                }
            )
    templates.sort(key=lambda item: item["bbox"])
    return templates


def _find_pieces(g: np.ndarray, templates: list[dict]) -> list[dict]:
    h, w = g.shape
    blocked = np.zeros((h, w), dtype=bool)
    for tpl in templates:
        r0, c0, r1, c1 = tpl["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    pieces: list[dict] = []
    for comp in _components((g != BG) & (~blocked)):
        vals = {int(g[r, c]) for r, c in comp}
        if len(vals) != 1:
            continue
        pieces.append(
            {
                "color": int(next(iter(vals))),
                "points": comp,
                "key": _rotation_key(_mask_from_points(comp)),
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    templates = _find_templates(g)
    pieces = _find_pieces(g, templates)

    templates_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    pieces_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)

    for tpl in templates:
        templates_by_key[tpl["key"]].append(tpl)
    for piece in pieces:
        pieces_by_key[piece["key"]].append(piece)

    for key, group in templates_by_key.items():
        matched = pieces_by_key.get(key, [])
        if len(matched) != len(group):
            raise AssertionError("template-piece shape counts must match")
        for piece in matched:
            for r, c in piece["points"]:
                out[r, c] = BG
        for tpl in group:
            for r, c in tpl["hole_points"]:
                out[r, c] = tpl["color"]

    return out


def _build_shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen = set()
    for raw in RAW_SHAPES:
        mask = np.array(raw, dtype=bool)
        key = _rotation_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    return shapes


SHAPE_LIBRARY = _build_shape_library()


def _sample_template_specs(rng: random.Random):
    count = rng.choice(TEMPLATE_COUNT_CHOICES)
    if 2 * count > len(COLOR_POOL):
        return None
    chosen_shapes = rng.sample(SHAPE_LIBRARY, count)
    chosen_colors = rng.sample(COLOR_POOL, 2 * count)
    border_colors = chosen_colors[:count]
    piece_colors = chosen_colors[count:]

    specs = []
    for idx in range(count):
        hole_mask = rng.choice(_unique_rotations(chosen_shapes[idx]))
        piece_mask = rng.choice(_unique_rotations(chosen_shapes[idx]))
        top = rng.randint(*MARGIN_RANGE)
        bottom = rng.randint(*MARGIN_RANGE)
        left = rng.randint(*MARGIN_RANGE)
        right = rng.randint(*MARGIN_RANGE)
        outer_h = hole_mask.shape[0] + top + bottom + 2
        outer_w = hole_mask.shape[1] + left + right + 2
        hole_row = 1 + top
        hole_col = 1 + left
        specs.append(
            {
                "hole_mask": hole_mask,
                "piece_mask": piece_mask,
                "border_color": int(border_colors[idx]),
                "piece_color": int(piece_colors[idx]),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
                "hole_row": int(hole_row),
                "hole_col": int(hole_col),
            }
        )
    return specs


def _find_positions(
    occupied: np.ndarray, obj_h: int, obj_w: int, edge_margin: int, pad: int
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(edge_margin, h - obj_h - edge_margin + 1):
        for c0 in range(edge_margin, w - obj_w - edge_margin + 1):
            rr0 = max(0, r0 - pad)
            rr1 = min(h, r0 + obj_h + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c0 + obj_w + pad)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _occupy(occupied: np.ndarray, r0: int, c0: int, obj_h: int, obj_w: int, pad: int) -> None:
    h, w = occupied.shape
    occupied[max(0, r0 - pad) : min(h, r0 + obj_h + pad), max(0, c0 - pad) : min(w, c0 + obj_w + pad)] = True


def _render_template_patch(spec: dict) -> np.ndarray:
    patch = np.full((spec["outer_h"], spec["outer_w"]), FILL, dtype=int)
    patch[0, :] = spec["border_color"]
    patch[-1, :] = spec["border_color"]
    patch[:, 0] = spec["border_color"]
    patch[:, -1] = spec["border_color"]
    hr0 = spec["hole_row"]
    hc0 = spec["hole_col"]
    hmask = spec["hole_mask"]
    patch[hr0 : hr0 + hmask.shape[0], hc0 : hc0 + hmask.shape[1]][hmask] = BG
    return patch


def _render_piece_patch(spec: dict) -> np.ndarray:
    patch = np.zeros(spec["piece_mask"].shape, dtype=int)
    patch[spec["piece_mask"]] = spec["piece_color"]
    return patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    specs = _sample_template_specs(rng)
    if specs is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    out = grid.copy()
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in specs:
        patch = _render_template_patch(spec)
        positions = _find_positions(
            occupied, patch.shape[0], patch.shape[1], EDGE_MARGIN, OBJECT_PAD
        )
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch

        hr0 = r0 + spec["hole_row"]
        hc0 = c0 + spec["hole_col"]
        hmask = spec["hole_mask"]
        out[hr0 : hr0 + hmask.shape[0], hc0 : hc0 + hmask.shape[1]][hmask] = spec["border_color"]
        _occupy(occupied, r0, c0, patch.shape[0], patch.shape[1], OBJECT_PAD)

    for spec in specs:
        patch = _render_piece_patch(spec)
        positions = _find_positions(
            occupied, patch.shape[0], patch.shape[1], EDGE_MARGIN, OBJECT_PAD
        )
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][spec["piece_mask"]] = spec["piece_color"]
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][spec["piece_mask"]] = BG
        _occupy(occupied, r0, c0, patch.shape[0], patch.shape[1], OBJECT_PAD)

    try:
        if not np.array_equal(main(grid), out):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
