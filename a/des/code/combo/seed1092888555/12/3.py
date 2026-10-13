from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
GEN_RETRY_LIMIT = 400
SHAPE_LIBRARY = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _norm_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _shape_key(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], bytes]:
    mask = _norm_mask(coords)
    return tuple(map(int, mask.shape)), mask.tobytes()


def _largest_multicolor_component(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = _components(grid != BG)
    assert comps, "expected foreground"
    colored = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        colored.append((len(comp), len(colors), comp))
    colored.sort(key=lambda item: (item[0], item[1]))
    main = colored[-1][2]
    assert len({int(grid[r, c]) for r, c in main}) >= 2, "main pattern must be multicolor"
    return main


def _holes_for_main(grid: np.ndarray, main_coords: list[tuple[int, int]]) -> list[dict]:
    h, w = grid.shape
    main_mask = np.zeros((h, w), dtype=bool)
    for r, c in main_coords:
        main_mask[r, c] = True

    holes = []
    for comp in _components(grid == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        boundary_colors: list[int] = []
        ok = True
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                if grid[nr, nc] != BG:
                    if not main_mask[nr, nc]:
                        ok = False
                        break
                    boundary_colors.append(int(grid[nr, nc]))
            if not ok:
                break
        if not ok or not boundary_colors:
            continue
        color = Counter(boundary_colors).most_common(1)[0][0]
        holes.append({"coords": comp, "shape_key": _shape_key(comp), "color": color})
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed background hole in the large multicolor pattern using the
    matching scattered patch shape, recolored to the hole's surrounding color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = g.copy()
    main_comp = _largest_multicolor_component(g)
    holes = _holes_for_main(g, main_comp)

    all_comps = _components(g != BG)
    main_set = set(main_comp)
    patches = []
    for comp in all_comps:
        if set(comp) == main_set:
            continue
        patches.append(comp)

    patch_by_shape = {}
    for comp in patches:
        key = _shape_key(comp)
        assert key not in patch_by_shape, "patch shapes must be unique"
        patch_by_shape[key] = comp

    for hole in holes:
        patch = patch_by_shape[hole["shape_key"]]
        for r, c in patch:
            out[r, c] = BG
        for r, c in hole["coords"]:
            out[r, c] = int(hole["color"])

    return out


def _rand_partition(total: int, parts: int, min_size: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total - min_size * parts + parts), parts - 1))
    vals = []
    prev = 0
    for cut in cuts + [total - min_size * parts + parts]:
        vals.append(cut - prev)
        prev = cut
    return [v + min_size - 1 for v in vals]


def _paint_stripes(patch: np.ndarray, colors: list[int], vertical: bool) -> None:
    h, w = patch.shape
    spans = _rand_partition(w if vertical else h, len(colors), 3)
    pos = 0
    for span, color in zip(spans, colors):
        if vertical:
            patch[:, pos : pos + span] = color
        else:
            patch[pos : pos + span, :] = color
        pos += span


def _paint_blocks(patch: np.ndarray, colors: list[int]) -> None:
    h, w = patch.shape
    split_r = random.randint(3, h - 3)
    split_c = random.randint(3, w - 3)
    block_colors = random.sample(colors, 4) if len(colors) >= 4 else [colors[0], colors[1], colors[0], colors[1]]
    patch[:split_r, :split_c] = block_colors[0]
    patch[:split_r, split_c:] = block_colors[1]
    patch[split_r:, :split_c] = block_colors[2]
    patch[split_r:, split_c:] = block_colors[3]


def _make_main_patch(h: int, w: int) -> np.ndarray:
    colors = random.sample(range(1, 10), random.randint(2, 4))
    patch = np.zeros((h, w), dtype=int)
    mode = random.choice(["v", "h", "blocks"])
    if mode == "v":
        _paint_stripes(patch, colors, vertical=True)
    elif mode == "h":
        _paint_stripes(patch, colors, vertical=False)
    else:
        if h < 6 or w < 6:
            _paint_stripes(patch, colors, vertical=random.choice([True, False]))
        else:
            _paint_blocks(patch, colors)
    return patch


def _region_candidates(main_patch: np.ndarray, shape: np.ndarray) -> list[tuple[int, int, int]]:
    ph, pw = shape.shape
    h, w = main_patch.shape
    spots = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            region = main_patch[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1]
            color = int(main_patch[r0, c0])
            if np.all(region == color) and np.all(main_patch[r0 : r0 + ph, c0 : c0 + pw][shape] == color):
                spots.append((r0, c0, color))
    random.shuffle(spots)
    return spots


def _can_place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    ph, pw = mask.shape
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    rs = max(0, r0 - pad)
    re = min(h, r0 + ph + pad)
    cs = max(0, c0 - pad)
    ce = min(w, c0 + pw + pad)
    return not np.any(grid[rs:re, cs:ce] != BG)


def _place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    ph, pw = mask.shape
    view = grid[r0 : r0 + ph, c0 : c0 + pw]
    view[mask] = color


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    H = random.randint(18, 28)
    W = random.randint(18, 28)
    grid = np.zeros((H, W), dtype=int)

    main_h = random.randint(10, min(16, H - 5))
    main_w = random.randint(10, min(16, W - 5))
    main_r = random.randint(1, H - main_h - 1)
    main_c = random.randint(1, W - main_w - 1)
    main_patch = _make_main_patch(main_h, main_w)

    hole_count = random.randint(1, min(4, len(SHAPE_LIBRARY)))
    shapes = random.sample(SHAPE_LIBRARY, hole_count)
    placed_holes: list[tuple[np.ndarray, int, int, int]] = []
    used_shapes = set()
    work_patch = main_patch.copy()

    for shape in shapes:
        key = (tuple(map(int, shape.shape)), shape.tobytes())
        if key in used_shapes:
            continue
        spots = _region_candidates(work_patch, shape)
        chosen = None
        for r0, c0, color in spots:
            rr = slice(r0 - 1, r0 + shape.shape[0] + 1)
            cc = slice(c0 - 1, c0 + shape.shape[1] + 1)
            local = np.zeros_like(work_patch, dtype=bool)
            local[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]][shape] = True
            if not any(np.any(local[rr, cc] & prev_local[rr, cc]) for prev_local in []):
                chosen = (r0, c0, color)
                break
        if chosen is None:
            continue
        r0, c0, color = chosen
        work_patch[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]][shape] = BG
        placed_holes.append((shape.copy(), r0, c0, color))
        used_shapes.add(key)

    assert placed_holes, "need at least one hole"

    grid[main_r : main_r + main_h, main_c : main_c + main_w] = work_patch
    expected = grid.copy()
    for shape, r0, c0, color in placed_holes:
        expected[main_r + r0 : main_r + r0 + shape.shape[0], main_c + c0 : main_c + c0 + shape.shape[1]][shape] = color

    patch_colors = random.sample(range(1, 10), len(placed_holes))
    for (shape, _r0, _c0, _color), patch_color in zip(placed_holes, patch_colors):
        free_positions = [
            (r, c)
            for r in range(H - shape.shape[0] + 1)
            for c in range(W - shape.shape[1] + 1)
            if _can_place_mask(grid, shape, r, c, pad=1)
        ]
        assert free_positions, "patch must fit"
        pr, pc = random.choice(free_positions)
        _place_mask(grid, shape, pr, pc, patch_color)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            inp, expected = _build_case()
            out = main(inp)
            if np.array_equal(out, expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
