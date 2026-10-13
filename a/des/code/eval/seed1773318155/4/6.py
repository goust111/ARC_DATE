import random

import numpy as np


DIRS = ("up", "right", "down", "left")
STAMP_INNER_SHAPES = ((3, 3), (3, 1), (1, 3))
PADDING_CHOICES = (0, 2, 4)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300
PATTERN_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _ring_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for rr in range(r0, r1 + 1):
        cells.add((int(rr), int(c0)))
        cells.add((int(rr), int(c1)))
    for cc in range(c0, c1 + 1):
        cells.add((int(r0), int(cc)))
        cells.add((int(r1), int(cc)))
    return cells


def _halo_is_bg(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    for rr in range(max(0, r0 - 1), min(h, r1 + 2)):
        for cc in range(max(0, c0 - 1), min(w, c1 + 2)):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            if int(grid[rr, cc]) != int(bg):
                return False
    return True


def _gap_direction(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    mr, mc = missing
    if (c1 - c0 + 1) % 2 == 1:
        mid_c = c0 + (c1 - c0) // 2
        if mr == r0 and mc == mid_c:
            return "up"
        if mr == r1 and mc == mid_c:
            return "down"
    if (r1 - r0 + 1) % 2 == 1:
        mid_r = r0 + (r1 - r0) // 2
        if mr == mid_r and mc == c0:
            return "left"
        if mr == mid_r and mc == c1:
            return "right"
    return None


def _rotate_pattern(pattern: np.ndarray, direction: str) -> np.ndarray:
    k = {"up": 0, "right": -1, "down": 2, "left": 1}[direction]
    return np.rot90(pattern, k)


def _classify_components(grid: np.ndarray, bg: int) -> tuple[dict[int, dict], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    stamps: dict[int, dict] = {}
    canvases: dict[int, dict] = {}

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _color_components(g, color):
            bbox = comp["bbox"]
            r0, c0, r1, c1 = bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            if not _halo_is_bg(g, bg, bbox):
                continue
            ring = _ring_cells(r0, c0, r1, c1)
            coords = comp["coord_set"]
            inner = g[r0 + 1 : r1, c0 + 1 : c1]

            if coords == ring and 3 <= h <= 5 and 3 <= w <= 5 and inner.size and np.any(inner != bg):
                assert color not in stamps, "stamp border colors must be unique"
                stamps[color] = {
                    "color": color,
                    "bbox": bbox,
                    "pattern": inner.copy(),
                }
                continue

            missing = list(ring - coords)
            if len(missing) == 1 and coords == ring - {missing[0]} and inner.size and np.all(inner == bg):
                direction = _gap_direction(missing[0], bbox)
                if direction is None:
                    continue
                assert color not in canvases, "canvas border colors must be unique"
                canvases[color] = {
                    "color": color,
                    "bbox": bbox,
                    "direction": direction,
                    "inner_shape": inner.shape,
                }

    assert stamps and canvases, "expected at least one stamp and one canvas"
    assert set(stamps) == set(canvases), "stamp and canvas colors must match one-to-one"

    covered = np.zeros_like(g, dtype=bool)
    for stamp in stamps.values():
        r0, c0, r1, c1 = stamp["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
    for canvas in canvases.values():
        r0, c0, r1, c1 = canvas["bbox"]
        ring = _ring_cells(r0, c0, r1, c1)
        for rr, cc in ring:
            covered[rr, cc] = True
    assert np.all(covered[g != bg]), "all foreground must belong to recognized objects"
    return stamps, canvases


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each canvas to the stamp with the same border color, rotate the
    stamp's interior pattern according to the canvas gap direction, center it
    inside the canvas, and erase all stamps.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    stamps, canvases = _classify_components(g, bg)

    out = g.copy()
    for stamp in stamps.values():
        r0, c0, r1, c1 = stamp["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    for color, canvas in canvases.items():
        pattern = _rotate_pattern(stamps[color]["pattern"], canvas["direction"])
        inner_h, inner_w = map(int, canvas["inner_shape"])
        pat_h, pat_w = map(int, pattern.shape)
        assert pat_h <= inner_h and pat_w <= inner_w, "rotated pattern must fit in its canvas"
        r0, c0, _r1, _c1 = canvas["bbox"]
        top = r0 + 1 + (inner_h - pat_h) // 2
        left = c0 + 1 + (inner_w - pat_w) // 2
        out[top : top + pat_h, left : left + pat_w] = pattern

    return out


def _rotation_signatures(pattern: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    sigs = set()
    for k in range(4):
        rot = np.rot90(pattern, k)
        sigs.add((tuple(map(int, rot.shape)), rot.tobytes()))
    return sigs


def _random_pattern(rng: random.Random, shape: tuple[int, int], palette: list[int]) -> np.ndarray | None:
    h, w = shape
    for _ in range(PATTERN_ATTEMPTS):
        palette_size = rng.randint(2, min(4, len(palette)))
        colors = rng.sample(palette, palette_size)
        arr = np.array([[rng.choice(colors) for _ in range(w)] for _ in range(h)], dtype=int)
        if len({int(v) for v in arr.flat}) < 2:
            continue
        if len(_rotation_signatures(arr)) != 4:
            continue
        return arr
    return None


def _make_stamp(border_color: int, pattern: np.ndarray) -> np.ndarray:
    ph, pw = map(int, pattern.shape)
    out = np.full((ph + 2, pw + 2), int(border_color), dtype=int)
    out[1:-1, 1:-1] = pattern
    return out


def _make_canvas(border_color: int, bg: int, direction: str, pattern_shape: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, tuple[int, int]]:
    ph, pw = map(int, pattern_shape)
    for _ in range(50):
        extra_h = rng.choice(PADDING_CHOICES)
        extra_w = rng.choice(PADDING_CHOICES)
        if extra_h == 0 and extra_w == 0:
            continue
        inner_h = ph + extra_h
        inner_w = pw + extra_w
        out = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
        out[0, :] = border_color
        out[-1, :] = border_color
        out[:, 0] = border_color
        out[:, -1] = border_color
        mid_r = out.shape[0] // 2
        mid_c = out.shape[1] // 2
        if direction == "up":
            out[0, mid_c] = bg
        elif direction == "down":
            out[-1, mid_c] = bg
        elif direction == "left":
            out[mid_r, 0] = bg
        else:
            out[mid_r, -1] = bg
        return out, (inner_h, inner_w)
    return None, None


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    if r < 1 or c < 1 or r + h >= hh or c + w >= ww:
        return False
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(hh, r + h + pad)
    cc1 = min(ww, c + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r: int, c: int, bg: int) -> None:
    h, w = patch.shape
    window = grid[r : r + h, c : c + w]
    mask = patch != bg
    assert not np.any(window[mask] != bg)
    window[mask] = patch[mask]
    occupied[r : r + h, c : c + w] = True


def _render_expected(grid: np.ndarray, bg: int, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        sr, sc = pair["stamp_pos"]
        sh, sw = pair["stamp"].shape
        out[sr : sr + sh, sc : sc + sw] = int(bg)

    for pair in pairs:
        cr, cc = pair["canvas_pos"]
        inner_h, inner_w = pair["canvas_inner_shape"]
        rotated = _rotate_pattern(pair["pattern"], pair["direction"])
        rh, rw = rotated.shape
        top = cr + 1 + (inner_h - rh) // 2
        left = cc + 1 + (inner_w - rw) // 2
        out[top : top + rh, left : left + rw] = rotated

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = rng.randrange(10)
    n_pairs = rng.randint(1, 4)
    border_colors = rng.sample([c for c in range(10) if c != bg], n_pairs)
    interior_palette = [c for c in range(10) if c not in {bg, *border_colors}]
    rng.shuffle(interior_palette)
    assert len(interior_palette) >= 2

    directions = list(DIRS)
    rng.shuffle(directions)
    directions = directions[:n_pairs]

    pairs = []
    objects = []
    total_bbox_area = 0
    max_h = 0
    max_w = 0

    for idx, border_color in enumerate(border_colors):
        base_shape = rng.choice(STAMP_INNER_SHAPES)
        pattern = _random_pattern(rng, base_shape, interior_palette)
        if pattern is None:
            return None
        stamp = _make_stamp(border_color, pattern)
        direction = directions[idx]
        rotated = _rotate_pattern(pattern, direction)
        canvas, inner_shape = _make_canvas(border_color, bg, direction, rotated.shape, rng)
        if canvas is None:
            return None

        pair = {
            "color": int(border_color),
            "direction": direction,
            "pattern": pattern,
            "stamp": stamp,
            "canvas": canvas,
            "canvas_inner_shape": inner_shape,
            "stamp_pos": None,
            "canvas_pos": None,
        }
        pairs.append(pair)
        objects.append({"kind": "stamp", "pair": pair, "patch": stamp})
        objects.append({"kind": "canvas", "pair": pair, "patch": canvas})
        total_bbox_area += int(stamp.shape[0] * stamp.shape[1] + canvas.shape[0] * canvas.shape[1])
        max_h = max(max_h, int(stamp.shape[0]), int(canvas.shape[0]))
        max_w = max(max_w, int(stamp.shape[1]), int(canvas.shape[1]))

    base_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_bbox_area * 4.0))))
    H = rng.randint(base_side, base_side + 8)
    W = rng.randint(base_side, base_side + 8)

    grid = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    placement_order = objects[:]
    rng.shuffle(placement_order)

    for obj in placement_order:
        patch = obj["patch"]
        h, w = map(int, patch.shape)
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r = rng.randint(1, H - h - 1)
            c = rng.randint(1, W - w - 1)
            if not _can_place(occupied, r, c, h, w):
                continue
            _place_patch(grid, occupied, patch, r, c, bg)
            obj["pair"][f"{obj['kind']}_pos"] = (int(r), int(c))
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != bg:
        return None

    expected = _render_expected(grid, bg, pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
