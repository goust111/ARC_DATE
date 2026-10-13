import random
import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
MIN_GAP = 2

GRID_H_RANGE = (24, 31)
GRID_W_RANGE = (24, 31)
OBJECT_COUNT_CHOICES = (2, 3)
LAYER_CHOICES = (1, 2, 3, 4)
PAIR_COUNT_CHOICES = (3, 4)

PATTERN_MASKS = {
    3: [
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    ],
    4: [
        np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool),
        np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=bool),
        np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=bool),
    ],
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_values(block: np.ndarray) -> np.ndarray:
    h, w = block.shape
    parts = [block[0, :], block[-1, :]]
    if h > 2:
        parts.append(block[1:-1, 0])
        parts.append(block[1:-1, -1])
    return np.concatenate(parts)


def _iterate_mapping(block: np.ndarray, mapping: dict[int, int], steps: int) -> np.ndarray:
    out = np.array(block, dtype=int, copy=True)
    for _ in range(steps):
        nxt = out.copy()
        for src, dst in mapping.items():
            nxt[out == src] = dst
        out = nxt
    return out


def _read_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, _w = grid.shape
    row_data: list[tuple[int, int, int, int]] = []
    for r in range(h):
        cols = np.flatnonzero(grid[r] != BG)
        if len(cols) == 2 and cols[1] == cols[0] + 1:
            c0 = int(cols[0])
            row_data.append((r, c0, int(grid[r, c0]), int(grid[r, c0 + 1])))

    assert row_data, "expected a left-bottom legend"
    start_col = min(c0 for _, c0, _, _ in row_data)
    by_row = {r: (src, dst) for r, c0, src, dst in row_data if c0 == start_col}
    assert by_row, "legend rows must share a start column"

    row = max(by_row)
    legend_rows: list[tuple[int, int, int]] = []
    while row in by_row:
        src, dst = by_row[row]
        legend_rows.append((row, src, dst))
        row -= 2

    assert len(legend_rows) >= 2, "legend should have multiple mapping rows"
    mapping: dict[int, int] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r, src, dst in legend_rows:
        legend_mask[r, start_col : start_col + 2] = True
        mapping[src] = dst
    assert len(mapping) == len(legend_rows), "legend source colors must be unique"
    return mapping, legend_mask


def _parse_object(crop: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    top = left = 0
    bottom = crop.shape[0] - 1
    right = crop.shape[1] - 1
    layer_colors: list[int] = []

    while bottom - top + 1 > 4:
        view = crop[top : bottom + 1, left : right + 1]
        border = _border_values(view)
        color = int(border[0])
        assert color != BG and np.all(border == color), "every peeled border must be a uniform ring"
        assert color not in layer_colors, "ring colors must be distinct within each object"
        layer_colors.append(color)
        top += 1
        left += 1
        bottom -= 1
        right -= 1

    center = crop[top : bottom + 1, left : right + 1]
    assert center.shape[0] == center.shape[1] and center.shape[0] in (3, 4), "center must be 3x3 or 4x4"
    assert 1 <= len(layer_colors) <= 4, "expected 1-4 rings"
    assert np.any(center != BG), "center pattern must be non-empty"
    return len(layer_colors), (top, left, bottom, right)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    mapping, legend_mask = _read_legend(g)
    object_mask = (g != BG) & (~legend_mask)
    comps = _components(object_mask)
    assert comps, "expected at least one nested-frame object"

    out = g.copy()
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        steps, (cr0, cc0, cr1, cc1) = _parse_object(crop)
        center = crop[cr0 : cr1 + 1, cc0 : cc1 + 1]
        out[r0 + cr0 : r0 + cr1 + 1, c0 + cc0 : c0 + cc1 + 1] = _iterate_mapping(center, mapping, steps)
    return out


def _sample_mapping(rng: random.Random) -> tuple[str, list[tuple[int, int]], dict[int, int], list[int]]:
    style = rng.choice(("cycle", "chain"))
    pair_count = rng.choice(PAIR_COUNT_CHOICES)

    if style == "cycle":
        colors = rng.sample(list(COLORS), pair_count)
        pairs = [(colors[i], colors[(i + 1) % pair_count]) for i in range(pair_count)]
    else:
        colors = rng.sample(list(COLORS), pair_count + 1)
        pairs = [(colors[i], colors[i + 1]) for i in range(pair_count)]

    mapping = {src: dst for src, dst in pairs}
    stable_colors = [color for color in COLORS if color not in mapping]
    assert stable_colors, "generator needs at least one unmapped color"
    return style, pairs, mapping, stable_colors


def _sample_sizes(rng: random.Random, count: int) -> list[int]:
    if count == 2 and rng.random() < 0.6:
        sizes = [3, 4]
        rng.shuffle(sizes)
        return sizes
    return [rng.choice((3, 4)) for _ in range(count)]


def _sample_layers(rng: random.Random, count: int, style: str, cycle_len: int) -> list[int]:
    allowed = list(LAYER_CHOICES)
    if style == "cycle":
        allowed = [n for n in allowed if n != cycle_len]
    layers = [rng.choice(allowed) for _ in range(count)]
    if len(set(layers)) == 1:
        alternatives = [n for n in allowed if n != layers[0]]
        if alternatives:
            layers[-1] = rng.choice(alternatives)
    return layers


def _paint_pattern(
    rng: random.Random,
    mask: np.ndarray,
    mapping: dict[int, int],
    stable_colors: list[int],
    steps: int,
    require_stable: bool,
) -> np.ndarray | None:
    src_colors = list(mapping)
    cells = [tuple(x) for x in np.argwhere(mask)]
    max_colors = min(4, len(cells))

    for _ in range(80):
        chosen = [rng.choice(src_colors)]

        if stable_colors and (require_stable or rng.random() < 0.55):
            chosen.append(rng.choice(stable_colors))
        else:
            other_src = [c for c in src_colors if c != chosen[0]]
            if other_src:
                chosen.append(rng.choice(other_src))
            elif stable_colors:
                chosen.append(rng.choice(stable_colors))

        palette = [c for c in src_colors + stable_colors if c not in chosen]
        while palette and len(chosen) < max_colors and rng.random() < 0.35:
            pick = rng.choice(palette)
            chosen.append(pick)
            palette.remove(pick)

        if len(set(chosen)) < 2:
            continue

        assigned = chosen[:] + [rng.choice(chosen) for _ in range(len(cells) - len(chosen))]
        rng.shuffle(assigned)

        block = np.zeros(mask.shape, dtype=int)
        for (r, c), color in zip(cells, assigned):
            block[r, c] = color

        if require_stable and not any(color in stable_colors for color in assigned):
            continue
        if np.array_equal(_iterate_mapping(block, mapping, steps), block):
            continue
        return block
    return None


def _build_object(
    rng: random.Random,
    mapping: dict[int, int],
    stable_colors: list[int],
    layers: int,
    size: int,
    require_stable: bool,
) -> dict[str, object] | None:
    mask = rng.choice(PATTERN_MASKS[size]).copy()
    pattern = _paint_pattern(rng, mask, mapping, stable_colors, layers, require_stable)
    if pattern is None:
        return None

    box = size + 2 * layers
    crop = np.zeros((box, box), dtype=int)
    ring_colors = rng.sample(list(COLORS), layers)
    for depth, color in enumerate(ring_colors):
        lo = depth
        hi = box - 1 - depth
        crop[lo, lo : hi + 1] = color
        crop[hi, lo : hi + 1] = color
        crop[lo : hi + 1, lo] = color
        crop[lo : hi + 1, hi] = color

    crop[layers : layers + size, layers : layers + size] = pattern
    return {
        "crop": crop,
        "layers": layers,
        "size": size,
    }


def _reserve(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - MIN_GAP)
    cc0 = max(0, c0 - MIN_GAP)
    rr1 = min(occ.shape[0], r0 + h + MIN_GAP)
    cc1 = min(occ.shape[1], c0 + w + MIN_GAP)
    occ[rr0:rr1, cc0:cc1] = True


def _place_patch(rng: random.Random, grid: np.ndarray, occ: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = grid.shape
    if h > H or w > W:
        return None

    r_lo = 1 if H - h >= 2 else 0
    c_lo = 1 if W - w >= 2 else 0
    r_hi = H - h - 1 if H - h >= 2 else H - h
    c_hi = W - w - 1 if W - w >= 2 else W - w
    if r_hi < r_lo or c_hi < c_lo:
        return None

    for _ in range(300):
        r0 = rng.randint(r_lo, r_hi)
        c0 = rng.randint(c_lo, c_hi)
        if occ[r0 : r0 + h, c0 : c0 + w].any():
            continue
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        _reserve(occ, r0, c0, h, w)
        return r0, c0
    return None


def _expected_output(grid: np.ndarray, placements: list[tuple[int, int, dict[str, object]]], mapping: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for r0, c0, obj in placements:
        layers = int(obj["layers"])
        size = int(obj["size"])
        center = out[r0 + layers : r0 + layers + size, c0 + layers : c0 + layers + size]
        out[r0 + layers : r0 + layers + size, c0 + layers : c0 + layers + size] = _iterate_mapping(
            center, mapping, layers
        )
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    style, pairs, mapping, stable_colors = _sample_mapping(rng)
    cycle_len = len(pairs)
    obj_count = rng.choice(OBJECT_COUNT_CHOICES)
    sizes = _sample_sizes(rng, obj_count)
    layers = _sample_layers(rng, obj_count, style, cycle_len)

    objects: list[dict[str, object]] = []
    stable_used = False
    for idx, (layer_count, size) in enumerate(zip(layers, sizes)):
        require_stable = bool(stable_colors) and (idx == obj_count - 1 and not stable_used)
        obj = _build_object(rng, mapping, stable_colors, layer_count, size, require_stable)
        if obj is None:
            return None
        crop = obj["crop"]
        center = crop[layer_count : layer_count + size, layer_count : layer_count + size]
        if np.any(np.isin(center, stable_colors)):
            stable_used = True
        objects.append(obj)

    legend_pairs = pairs[:]
    rng.shuffle(legend_pairs)
    legend_height = 2 * len(legend_pairs) - 1
    max_obj = max(int(obj["crop"].shape[0]) for obj in objects)
    total_area = sum(int(obj["crop"].shape[0]) * int(obj["crop"].shape[1]) for obj in objects)

    min_h = max(GRID_H_RANGE[0], legend_height + 3, max_obj + 2)
    min_w = max(GRID_W_RANGE[0], max_obj + 4)
    candidates = [
        (h, w)
        for h in range(min_h, GRID_H_RANGE[1])
        for w in range(min_w, GRID_W_RANGE[1])
        if h * w >= total_area + 220
    ]
    if not candidates:
        return None
    H, W = rng.choice(candidates)

    grid = np.zeros((H, W), dtype=int)
    occ = np.zeros((H, W), dtype=bool)

    legend_top = H - legend_height
    legend_rows = list(range(legend_top, H, 2))
    for r, (src, dst) in zip(legend_rows, legend_pairs):
        grid[r, 0] = src
        grid[r, 1] = dst
    occ[max(0, legend_top - MIN_GAP) : H, 0 : min(W, 2 + MIN_GAP)] = True

    placements: list[tuple[int, int, dict[str, object]]] = []
    for obj in sorted(objects, key=lambda item: int(item["crop"].size), reverse=True):
        pos = _place_patch(rng, grid, occ, np.asarray(obj["crop"], dtype=int))
        if pos is None:
            return None
        placements.append((pos[0], pos[1], obj))

    expected = _expected_output(grid, placements, mapping)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame mapping task")
