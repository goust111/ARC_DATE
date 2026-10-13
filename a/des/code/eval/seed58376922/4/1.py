import random
from collections import Counter

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
MARKER_COLOR = 1
FRAME_COLOR_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 300
GRID_H_RANGE = (26, 34)
GRID_W_RANGE = (26, 34)


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_points(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _marker_side_from_pos(r: int, c: int, h: int, w: int) -> str:
    if r == 0 and 0 < c < w - 1:
        return "top"
    if r == h - 1 and 0 < c < w - 1:
        return "bottom"
    if c == 0 and 0 < r < h - 1:
        return "left"
    if c == w - 1 and 0 < r < h - 1:
        return "right"
    raise AssertionError("marker must lie on a non-corner edge cell")


def _marker_side(template: np.ndarray, marker_color: int) -> str:
    pos = np.argwhere(template == marker_color)
    assert len(pos) == 1, "template must contain exactly one marker"
    r, c = map(int, pos[0])
    h, w = template.shape
    return _marker_side_from_pos(r, c, h, w)


def _size_key(shape: tuple[int, int]) -> tuple[int, int]:
    a, b = map(int, shape)
    return (a, b) if a <= b else (b, a)


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    comps = _components(grid != bg)
    best = None
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        local_pts = [(r - r0, c - c0) for r, c in pts]
        mask = _mask_from_points(sub.shape, local_pts)
        colors = set(map(int, sub[mask]))
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        if sub.shape[0] < 5 or sub.shape[1] < 5:
            continue
        if not (
            np.all(sub[0, :] == color)
            and np.all(sub[-1, :] == color)
            and np.all(sub[:, 0] == color)
            and np.all(sub[:, -1] == color)
        ):
            continue
        if not np.any(sub[1:-1, 1:-1] == bg):
            continue
        score = ((r1 - r0 + 1) * (c1 - c0 + 1), len(pts))
        if best is None or score > best[0]:
            full_mask = np.zeros(grid.shape, dtype=bool)
            for r, c in pts:
                full_mask[r, c] = True
            best = (
                score,
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": int(color),
                    "mask": full_mask,
                },
            )
    assert best is not None, "failed to locate frame component"
    return best[1]


def _extract_partitions(grid: np.ndarray, frame: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = frame["bbox"]
    frame_color = frame["color"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    parts = []
    for pts in _components(inner == bg):
        rr0, cc0, rr1, cc1 = _bbox(pts)
        local_mask = _mask_from_points((rr1 - rr0 + 1, cc1 - cc0 + 1), [(r - rr0, c - cc0) for r, c in pts])
        assert np.all(local_mask), "partitions must be rectangular"
        pr0, pc0 = r0 + 1 + rr0, c0 + 1 + cc0
        pr1, pc1 = r0 + 1 + rr1, c0 + 1 + cc1
        ph, pw = pr1 - pr0 + 1, pc1 - pc0 + 1

        touched = []
        if pr0 == r0 + 1:
            touched.append(("top", pw))
        if pr1 == r1 - 1:
            touched.append(("bottom", pw))
        if pc0 == c0 + 1:
            touched.append(("left", ph))
        if pc1 == c1 - 1:
            touched.append(("right", ph))
        assert touched, "every partition must touch the frame boundary"
        best_len = max(length for _, length in touched)
        best_sides = [side for side, length in touched if length == best_len]
        assert len(best_sides) == 1, "partition outward side must be unique"

        parts.append(
            {
                "bbox": (pr0, pc0, pr1, pc1),
                "shape": (ph, pw),
                "open_side": best_sides[0],
                "size_key": _size_key((ph, pw)),
            }
        )

    parts.sort(key=lambda item: item["bbox"])
    keys = [item["size_key"] for item in parts]
    assert len(set(keys)) == len(keys), "partition sizes must be unique up to rotation"
    return parts


def _extract_templates(grid: np.ndarray, frame_mask: np.ndarray, bg: int) -> list[dict]:
    templates = []
    for pts in _components((grid != bg) & (~frame_mask)):
        r0, c0, r1, c1 = _bbox(pts)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        local_pts = [(r - r0, c - c0) for r, c in pts]
        mask = _mask_from_points(sub.shape, local_pts)
        assert np.all(mask), "template must fill its bounding box"

        counts = Counter(map(int, sub.reshape(-1)))
        marker = None
        for color, count in counts.items():
            if count != 1:
                continue
            pos = np.argwhere(sub == color)
            rr, cc = map(int, pos[0])
            try:
                side = _marker_side_from_pos(rr, cc, sub.shape[0], sub.shape[1])
            except AssertionError:
                continue
            assert marker is None, "template must have exactly one edge marker"
            marker = (int(color), side)

        assert marker is not None, "failed to find template marker"
        marker_color, marked_side = marker
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "sub": sub.copy(),
                "shape": tuple(map(int, sub.shape)),
                "size_key": _size_key(sub.shape),
                "marker_color": marker_color,
                "marker_side": marked_side,
            }
        )

    templates.sort(key=lambda item: item["bbox"])
    keys = [item["size_key"] for item in templates]
    assert len(set(keys)) == len(keys), "template sizes must be unique up to rotation"
    return templates


def _orient_template(template: np.ndarray, marker_color: int, target_shape: tuple[int, int], target_side: str) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(template, k)
        if rotated.shape != target_shape:
            continue
        if _marker_side(rotated, marker_color) == target_side:
            return rotated
    raise AssertionError("no matching rotation found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame partition to the outside template with the same size,
    rotate so the template's marked edge faces the partition's outward side,
    and paste the rotated template into the partition without the marker pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frame = _find_frame(g, bg)
    partitions = _extract_partitions(g, frame, bg)
    templates = _extract_templates(g, frame["mask"], bg)

    by_size = {item["size_key"]: item for item in templates}
    assert len(partitions) == len(templates), "expected one template per partition"

    out = g.copy()
    for part in partitions:
        tmpl = by_size.get(part["size_key"])
        assert tmpl is not None, "missing template for partition"
        rotated = _orient_template(tmpl["sub"], tmpl["marker_color"], part["shape"], part["open_side"])
        pr0, pc0, pr1, pc1 = part["bbox"]
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = rotated
        region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        region[rotated == tmpl["marker_color"]] = bg
        out[pr0 : pr1 + 1, pc0 : pc1 + 1] = region
    return out


def _valid_partition_shapes(shapes: list[tuple[int, int]]) -> bool:
    keys = []
    for h, w in shapes:
        if min(h, w) < 2:
            return False
        if h == w:
            return False
        if h * w < 6:
            return False
        if max(h, w) < 3:
            return False
        key = _size_key((h, w))
        if key in keys:
            return False
        keys.append(key)
    return True


def _layout_cross(rng: random.Random) -> np.ndarray | None:
    for _ in range(100):
        top_h = rng.randint(2, 5)
        bottom_h = rng.randint(2, 5)
        left_w = rng.randint(2, 5)
        right_w = rng.randint(2, 5)
        shapes = [(top_h, left_w), (top_h, right_w), (bottom_h, left_w), (bottom_h, right_w)]
        if not _valid_partition_shapes(shapes):
            continue
        wall = np.zeros((top_h + 1 + bottom_h, left_w + 1 + right_w), dtype=bool)
        wall[top_h, :] = True
        wall[:, left_w] = True
        return np.rot90(wall, rng.randrange(4))
    return None


def _layout_t(rng: random.Random) -> np.ndarray | None:
    for _ in range(100):
        top_h = rng.randint(2, 5)
        bottom_h = rng.randint(2, 5)
        left_w = rng.randint(2, 5)
        right_w = rng.randint(2, 5)
        total_h = top_h + 1 + bottom_h
        shapes = [(top_h, left_w), (bottom_h, left_w), (total_h, right_w)]
        if total_h <= right_w or not _valid_partition_shapes(shapes):
            continue
        wall = np.zeros((total_h, left_w + 1 + right_w), dtype=bool)
        wall[:, left_w] = True
        wall[top_h, : left_w + 1] = True
        return np.rot90(wall, rng.randrange(4))
    return None


def _layout_comb(rng: random.Random) -> np.ndarray | None:
    for _ in range(100):
        top_h = rng.randint(2, 4)
        mid_h = rng.randint(2, 4)
        bot_h = rng.randint(2, 4)
        left_w = rng.randint(2, 5)
        right_w = rng.randint(2, 5)
        total_h = top_h + 1 + mid_h + 1 + bot_h
        shapes = [(top_h, left_w), (mid_h, left_w), (bot_h, left_w), (total_h, right_w)]
        if total_h <= right_w or not _valid_partition_shapes(shapes):
            continue
        wall = np.zeros((total_h, left_w + 1 + right_w), dtype=bool)
        wall[:, left_w] = True
        wall[top_h, : left_w + 1] = True
        wall[top_h + 1 + mid_h, : left_w + 1] = True
        return np.rot90(wall, rng.randrange(4))
    return None


def _build_frame_patch(wall_mask: np.ndarray, frame_color: int) -> np.ndarray:
    ih, iw = wall_mask.shape
    patch = np.full((ih + 2, iw + 2), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    inner = patch[1:-1, 1:-1]
    inner[wall_mask] = frame_color
    return patch


def _choose_marker_position(shape: tuple[int, int], side: str, rng: random.Random) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return 0, rng.randrange(1, w - 1)
    if side == "bottom":
        return h - 1, rng.randrange(1, w - 1)
    if side == "left":
        return rng.randrange(1, h - 1), 0
    if side == "right":
        return rng.randrange(1, h - 1), w - 1
    raise AssertionError("bad side")


def _make_body_pattern(shape: tuple[int, int], colors: list[int], rng: random.Random) -> np.ndarray:
    h, w = shape
    assert len(colors) >= 2
    a, b = rng.sample(colors, 2)
    phase = rng.randrange(2)
    arr = np.zeros((h, w), dtype=int)
    for r in range(h):
        for col in range(w):
            arr[r, col] = a if (r + col + phase) % 2 == 0 else b
    return arr


def _make_template(shape: tuple[int, int], frame_color: int, rng: random.Random) -> np.ndarray:
    h, w = shape
    body_pool = [c for c in FRAME_COLOR_CHOICES if c != frame_color and c != MARKER_COLOR]
    for _ in range(100):
        valid_sides = []
        if w >= 3:
            valid_sides.extend(["top", "bottom"])
        if h >= 3:
            valid_sides.extend(["left", "right"])
        side = rng.choice(valid_sides)
        body = _make_body_pattern(shape, body_pool, rng)
        mr, mc = _choose_marker_position(shape, side, rng)
        body[mr, mc] = MARKER_COLOR
        counts = Counter(map(int, body.reshape(-1)))
        if counts[MARKER_COLOR] != 1:
            continue
        if any(count == 1 for color, count in counts.items() if color != MARKER_COLOR):
            continue
        return body
    raise RuntimeError("failed to build template")


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _place_templates(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], templates: list[np.ndarray], rng: random.Random) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    blocked[max(0, fr0 - 1) : min(h, fr1 + 2), max(0, fc0 - 1) : min(w, fc1 + 2)] = True
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    order = list(range(len(templates)))
    rng.shuffle(order)
    for idx in order:
        patch = templates[idx]
        ph, pw = patch.shape
        spots = []
        for r in range(1, h - ph):
            for c in range(1, w - pw):
                if blocked[r : r + ph, c : c + pw].any():
                    continue
                spots.append((r, c))
        if not spots:
            return False
        top, left = rng.choice(spots)
        _place_patch(grid, patch, top, left)
        blocked[max(0, top - 1) : min(h, top + ph + 1), max(0, left - 1) : min(w, left + pw + 1)] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout_builder = rng.choice([_layout_cross, _layout_t, _layout_comb])
    wall_mask = layout_builder(rng)
    if wall_mask is None:
        return None

    frame_color = rng.choice(FRAME_COLOR_CHOICES)
    frame_patch = _build_frame_patch(wall_mask, frame_color)
    fh, fw = frame_patch.shape

    min_h = max(GRID_H_RANGE[0], fh + 6)
    min_w = max(GRID_W_RANGE[0], fw + 6)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = rng.randint(min_h, GRID_H_RANGE[1])
    grid_w = rng.randint(min_w, GRID_W_RANGE[1])
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frame_top = rng.randint(2, grid_h - fh - 2)
    frame_left = rng.randint(2, grid_w - fw - 2)
    _place_patch(grid, frame_patch, frame_top, frame_left)

    frame = _find_frame(grid, BG)
    partitions = _extract_partitions(grid, frame, BG)

    templates = []
    for part in partitions:
        ph, pw = part["shape"]
        displayed_shape = (ph, pw) if rng.random() < 0.5 else (pw, ph)
        template = _make_template(displayed_shape, frame_color, rng)
        templates.append(template)

    if not _place_templates(grid, frame["bbox"], templates, rng):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == MARKER_COLOR) and not np.any(grid == MARKER_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
