import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bool(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("cannot crop an empty mask")
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "coords": tuple(coords),
                }
            )
    return comps


def _placement_options(limit_a: int, size_b: int) -> list[tuple[int, int]]:
    opts = [(0, offset) for offset in range(limit_a + 1)]
    opts.extend((offset, 0) for offset in range(1, size_b + 1))
    return opts


def _find_pair_layout(mask_a: np.ndarray, mask_b: np.ndarray):
    h1, w1 = map(int, mask_a.shape)
    h2, w2 = map(int, mask_b.shape)
    row_opts = _placement_options(h1, h2)
    col_opts = _placement_options(w1, w2)
    best = None

    for r1, r2 in row_opts:
        for c1, c2 in col_opts:
            h = max(r1 + h1, r2 + h2)
            w = max(c1 + w1, c2 + w2)
            arr_a = np.zeros((h, w), dtype=bool)
            arr_b = np.zeros((h, w), dtype=bool)
            arr_a[r1 : r1 + h1, c1 : c1 + w1] |= mask_a
            arr_b[r2 : r2 + h2, c2 : c2 + w2] |= mask_b
            if np.any(arr_a & arr_b):
                continue
            if not np.all(arr_a | arr_b):
                continue
            cand = (h * w, h, w, r1, c1, r2, c2)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None
    _, h, w, r1, c1, r2, c2 = best
    return {"shape": (h, w), "a_offset": (r1, c1), "b_offset": (r2, c2)}


def _render_pair(comp_a: dict, comp_b: dict, layout: dict) -> np.ndarray:
    h, w = layout["shape"]
    out = np.full((h, w), BG, dtype=int)
    r1, c1 = layout["a_offset"]
    r2, c2 = layout["b_offset"]
    mask_a = comp_a["mask"]
    mask_b = comp_b["mask"]
    out[r1 : r1 + mask_a.shape[0], c1 : c1 + mask_a.shape[1]][mask_a] = int(comp_a["color"])
    out[r2 : r2 + mask_b.shape[0], c2 : c2 + mask_b.shape[1]][mask_b] = int(comp_b["color"])
    return out


def _matching_signature(pair_rect: np.ndarray) -> tuple:
    flat = tuple(int(x) for x in pair_rect.flatten())
    return pair_rect.shape[0], pair_rect.shape[1], flat


def _search_matching(comps: list[dict], edges: dict[tuple[int, int], dict]):
    remaining = tuple(range(len(comps)))

    def dfs(todo: tuple[int, ...]):
        if not todo:
            return []
        first = todo[0]
        rest = set(todo[1:])
        for other in sorted(rest):
            key = (first, other)
            if key not in edges:
                continue
            next_todo = tuple(idx for idx in todo[1:] if idx != other)
            tail = dfs(next_todo)
            if tail is not None:
                return [(first, other)] + tail
        return None

    return dfs(remaining)


def _solve_components(comps: list[dict]) -> list[np.ndarray]:
    if len(comps) % 2 != 0:
        raise AssertionError("expected an even number of fragments")

    edges: dict[tuple[int, int], dict] = {}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            layout = _find_pair_layout(comps[i]["mask"], comps[j]["mask"])
            if layout is not None:
                edges[(i, j)] = layout

    matching = _search_matching(comps, edges)
    if matching is None:
        raise AssertionError("could not pair every fragment")

    rects: list[np.ndarray] = []
    for i, j in matching:
        rects.append(_render_pair(comps[i], comps[j], edges[(i, j)]))

    rects.sort(key=lambda arr: (arr.shape[0] * arr.shape[1],) + _matching_signature(arr))
    return rects


def _assemble_output(rects: list[np.ndarray]) -> np.ndarray:
    height = max(int(arr.shape[0]) for arr in rects)
    width = sum(int(arr.shape[1]) for arr in rects) + max(0, len(rects) - 1)
    out = np.full((height, width), BG, dtype=int)
    col = 0
    for idx, arr in enumerate(rects):
        h, w = arr.shape
        out[:h, col : col + w] = arr
        col += w
        if idx + 1 < len(rects):
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments into full rectangles, then sort those rectangles by area.
    """
    comps = _find_components(np.asarray(grid, dtype=int))
    assert comps, "expected at least one fragment"
    rects = _solve_components(comps)
    return _assemble_output(rects)


def _make_vertical_pair(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    cuts = [rng.randint(1, w - 1) for _ in range(h)]
    if rng.random() < 0.6 and w >= 4:
        # Encourage non-rectangular cuts in most samples.
        while len(set(cuts)) == 1:
            cuts = [rng.randint(1, w - 1) for _ in range(h)]
    mask_a = np.zeros((h, w), dtype=bool)
    for r, cut in enumerate(cuts):
        mask_a[r, :cut] = True
    return mask_a, ~mask_a


def _make_horizontal_pair(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    cuts = [rng.randint(1, h - 1) for _ in range(w)]
    if rng.random() < 0.6 and h >= 4:
        while len(set(cuts)) == 1:
            cuts = [rng.randint(1, h - 1) for _ in range(w)]
    mask_a = np.zeros((h, w), dtype=bool)
    for c, cut in enumerate(cuts):
        mask_a[:cut, c] = True
    return mask_a, ~mask_a


def _make_pair_masks(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        return _make_vertical_pair(rng, h, w)
    return _make_horizontal_pair(rng, h, w)


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    cells = np.argwhere(mask)
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _place_fragment(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    frag_mask: np.ndarray,
    color: int,
    rng: random.Random,
) -> bool:
    gh, gw = grid.shape
    fh, fw = frag_mask.shape
    for _ in range(PLACE_ATTEMPTS):
        r0 = rng.randint(0, gh - fh)
        c0 = rng.randint(0, gw - fw)
        coords = np.argwhere(frag_mask)
        target_rows = coords[:, 0] + r0
        target_cols = coords[:, 1] + c0
        if np.any(blocked[target_rows, target_cols]):
            continue
        grid[target_rows, target_cols] = color
        occupied[target_rows, target_cols] = True
        blocked[:] = _dilate(occupied)
        return True
    return False


def _expected_from_pairs(pairs: list[dict]) -> np.ndarray:
    rects = [np.where(item["mask_a"], item["color_a"], item["color_b"]).astype(int) for item in pairs]
    rects.sort(key=lambda arr: (arr.shape[0] * arr.shape[1],) + _matching_signature(arr))
    return _assemble_output(rects)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_pairs = rng.randint(2, 4)
    dims_pool = [(h, w) for h in range(3, 7) for w in range(4, 8)]
    rng.shuffle(dims_pool)

    chosen: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for h, w in dims_pool:
        area = h * w
        if area in used_areas:
            continue
        chosen.append((h, w))
        used_areas.add(area)
        if len(chosen) == num_pairs:
            break
    if len(chosen) != num_pairs:
        return None

    colors = rng.sample(PALETTE, 2 * num_pairs)
    pairs: list[dict] = []
    frags: list[tuple[np.ndarray, int]] = []
    for idx, (h, w) in enumerate(chosen):
        mask_a, mask_b = _make_pair_masks(rng, h, w)
        crop_a, _ = _crop_bool(mask_a)
        crop_b, _ = _crop_bool(mask_b)
        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        pairs.append(
            {
                "mask_a": mask_a,
                "mask_b": mask_b,
                "color_a": color_a,
                "color_b": color_b,
            }
        )
        frags.append((crop_a, color_a))
        frags.append((crop_b, color_b))

    max_frag_h = max(mask.shape[0] for mask, _ in frags)
    total_frag_w = sum(mask.shape[1] for mask, _ in frags)
    grid_h = max(16, max_frag_h + rng.randint(8, 14))
    grid_w = max(20, total_frag_w + rng.randint(8, 16))
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    blocked = np.zeros_like(grid, dtype=bool)
    frag_order = list(range(len(frags)))
    rng.shuffle(frag_order)

    for idx in frag_order:
        frag_mask, color = frags[idx]
        if not _place_fragment(grid, occupied, blocked, frag_mask, color, rng):
            return None

    expected = _expected_from_pairs(pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
