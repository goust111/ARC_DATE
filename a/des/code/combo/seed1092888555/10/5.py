import numpy as np


_RNG = np.random.default_rng()
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    cur = mask
    for _ in range(4):
        out.append(_trim_mask(cur))
        out.append(_trim_mask(np.fliplr(cur)))
        cur = np.rot90(cur)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    options = []
    for variant in _mask_variants(mask):
        arr = np.ascontiguousarray(variant.astype(np.uint8))
        options.append((arr.shape, arr.tobytes()))
    return min(options)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in coords]
            cs = [p[1] for p in coords]
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                    "colors": {int(grid[rr, cc]) for rr, cc in coords},
                    "sample_color": color,
                }
            )
    return comps


def _corner_rank(comp: dict, h: int, w: int) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    corners = [
        r0 + c0,
        r0 + (w - 1 - c1),
        (h - 1 - r1) + c0,
        (h - 1 - r1) + (w - 1 - c1),
    ]
    return int(min(corners))


def _extract_component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    comps = _find_components(g, bg)
    assert len(comps) == 5

    central = max(comps, key=lambda comp: (comp["area"], (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1)))
    icons = [comp for comp in comps if comp is not central]
    assert len(icons) == 4

    ch0, cw0, ch1, cw1 = central["bbox"]
    ph = ch1 - ch0 + 1
    pw = cw1 - cw0 + 1
    assert ph % 2 == 0 and pw % 2 == 0
    hh = ph // 2
    hw = pw // 2

    patch = g[ch0 : ch1 + 1, cw0 : cw1 + 1]
    quad_slices = {
        "ul": (slice(0, hh), slice(0, hw)),
        "ur": (slice(0, hh), slice(hw, pw)),
        "ll": (slice(hh, ph), slice(0, hw)),
        "lr": (slice(hh, ph), slice(hw, pw)),
    }
    quad_masks = {}
    quad_keys = {}
    for name, (rs, cs) in quad_slices.items():
        mask = patch[rs, cs] != bg
        assert mask.any()
        key = _canonical_key(mask)
        assert key not in quad_keys.values()
        quad_masks[name] = mask
        quad_keys[name] = key

    out = np.full_like(g, bg)
    matched = set()
    for comp in sorted(icons, key=lambda comp: _corner_rank(comp, *g.shape)):
        assert len(comp["colors"]) == 1
        color = next(iter(comp["colors"]))
        key = _canonical_key(_extract_component_mask(comp))
        names = [name for name, qkey in quad_keys.items() if qkey == key]
        assert len(names) == 1
        name = names[0]
        assert name not in matched
        matched.add(name)
        rs, cs = quad_slices[name]
        sub = out[ch0 : ch1 + 1, cw0 : cw1 + 1]
        sub_view = sub[rs, cs]
        sub_view[quad_masks[name]] = color
    assert len(matched) == 4
    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(pts[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(size: int, required: set[tuple[int, int]]) -> np.ndarray:
    required = set(required)
    target = int(_RNG.integers(size + 1, size * size))
    mask = np.zeros((size, size), dtype=bool)
    order = list(required)
    _RNG.shuffle(order)
    first = order[0]
    mask[first] = True
    frontier = {first}
    pending = set(required) - {first}

    def neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = cell
        out = []
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                out.append((nr, nc))
        return out

    while pending:
        options = []
        for cell in pending:
            if any(mask[nr, nc] for nr, nc in neighbors(cell)):
                options.append(cell)
        if options:
            cell = options[int(_RNG.integers(len(options)))]
        else:
            grown = []
            for r in range(size):
                for c in range(size):
                    if not mask[r, c] and any(mask[nr, nc] for nr, nc in neighbors((r, c))):
                        grown.append((r, c))
            cell = grown[int(_RNG.integers(len(grown)))]
        mask[cell] = True
        frontier.add(cell)
        pending.discard(cell)

    while int(mask.sum()) < target:
        options = []
        for r in range(size):
            for c in range(size):
                if not mask[r, c] and any(mask[nr, nc] for nr, nc in neighbors((r, c))):
                    options.append((r, c))
        if not options:
            break
        cell = options[int(_RNG.integers(len(options)))]
        mask[cell] = True

    assert _connected(mask)
    return mask


def _generate_quadrant_masks(size: int) -> list[np.ndarray] | None:
    bridge_rows = {
        "top": int(_RNG.integers(size)),
        "bottom": int(_RNG.integers(size)),
    }
    bridge_cols = {
        "left": int(_RNG.integers(size)),
        "right": int(_RNG.integers(size)),
    }
    required_by_quad = {
        "ul": {(bridge_rows["top"], size - 1), (size - 1, bridge_cols["left"])},
        "ur": {(bridge_rows["top"], 0), (size - 1, bridge_cols["right"])},
        "ll": {(0, bridge_cols["left"]), (bridge_rows["bottom"], size - 1)},
        "lr": {(0, bridge_cols["right"]), (bridge_rows["bottom"], 0)},
    }

    masks = []
    keys = set()
    for name in ["ul", "ur", "ll", "lr"]:
        for _ in range(80):
            mask = _random_connected_mask(size, required_by_quad[name])
            key = _canonical_key(mask)
            if key not in keys:
                keys.add(key)
                masks.append(mask)
                break
        else:
            return None
    return masks


def _paste_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _orient_mask(mask: np.ndarray, code: int) -> np.ndarray:
    out = mask
    if code >= 4:
        out = np.fliplr(out)
        code -= 4
    return np.rot90(out, code)


def _gen_attempt() -> np.ndarray | None:
    q = int(_RNG.integers(3, 6))
    center_h = center_w = 2 * q
    grid_h = int(_RNG.integers(max(2 * q + 6, 14), max(2 * q + 12, 20)))
    grid_w = int(_RNG.integers(max(2 * q + 6, 14), max(2 * q + 12, 20)))
    if grid_h < center_h + q + 2 or grid_w < center_w + q + 2:
        return None

    masks = _generate_quadrant_masks(q)
    if masks is None:
        return None

    bg = 0
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    top_min = q + 1
    left_min = q + 1
    top_max = grid_h - q - center_h - 1
    left_max = grid_w - q - center_w - 1
    if top_max < top_min or left_max < left_min:
        return None
    top = int(_RNG.integers(top_min, top_max + 1))
    left = int(_RNG.integers(left_min, left_max + 1))

    used_colors = list(_RNG.choice(_PALETTE, size=8, replace=False))
    center_colors = used_colors[:4]
    icon_colors = used_colors[4:]

    quadrants = {
        "ul": masks[0],
        "ur": masks[1],
        "ll": masks[2],
        "lr": masks[3],
    }
    positions = {
        "ul": (top, left),
        "ur": (top, left + q),
        "ll": (top + q, left),
        "lr": (top + q, left + q),
    }
    for idx, name in enumerate(["ul", "ur", "ll", "lr"]):
        _paste_mask(grid, positions[name][0], positions[name][1], quadrants[name], center_colors[idx])

    corner_boxes = {
        "ul": (int(_RNG.integers(0, 2)), int(_RNG.integers(0, 2))),
        "ur": (int(_RNG.integers(0, 2)), grid_w - q - int(_RNG.integers(0, 2))),
        "ll": (grid_h - q - int(_RNG.integers(0, 2)), int(_RNG.integers(0, 2))),
        "lr": (grid_h - q - int(_RNG.integers(0, 2)), grid_w - q - int(_RNG.integers(0, 2))),
    }
    for idx, name in enumerate(["ul", "ur", "ll", "lr"]):
        oriented = _orient_mask(quadrants[name], int(_RNG.integers(8)))
        _paste_mask(grid, corner_boxes[name][0], corner_boxes[name][1], oriented, icon_colors[idx])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
