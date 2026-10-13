import random
from collections import Counter, deque

import numpy as np


RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_NAMES = ("TL", "TR", "BL", "BR")
PATTERN_KINDS = ("checker_a", "checker_b", "stripe_v", "stripe_h")
GEN_ATTEMPTS = 600


def _bg_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.flat)).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _nearest_corner(anchor: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    ar, ac = anchor
    rel_r = ar - r0
    rel_c = ac - c0
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    distances = [
        rel_r + rel_c,
        rel_r + (w - 1 - rel_c),
        (h - 1 - rel_r) + rel_c,
        (h - 1 - rel_r) + (w - 1 - rel_c),
    ]
    best = min(distances)
    for name, dist in zip(CORNER_NAMES, distances):
        if dist == best:
            return name
    raise AssertionError("unreachable")


def _pattern_indices(
    r: int,
    c: int,
    bbox: tuple[int, int, int, int],
    corner: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if corner in ("TL", "TR"):
        pr = (r - r0) % 3
    else:
        pr = (2 - ((r1 - r) % 3)) % 3
    if corner in ("TL", "BL"):
        pc = (c - c0) % 3
    else:
        pc = (2 - ((c1 - c) % 3)) % 3
    return pr, pc


def _shape_objects(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _components(grid != bg)
    objects = []
    for pts in comps:
        colors = {int(grid[r, c]) for r, c in pts}
        red_pts = [(r, c) for r, c in pts if int(grid[r, c]) == RED]
        if len(red_pts) != 1:
            continue
        body_colors = colors - {RED}
        assert len(body_colors) == 1, "shape component must use exactly one non-red color"
        color = next(iter(body_colors))
        objects.append(
            {
                "color": int(color),
                "anchor": red_pts[0],
                "coords": pts,
                "bbox": _bbox(pts),
            }
        )
    assert objects, "expected at least one anchored shape"
    return objects


def _template_for_color(
    grid: np.ndarray,
    bg: int,
    shape_mask: np.ndarray,
    color: int,
) -> np.ndarray:
    pts = np.argwhere((grid == color) & (~shape_mask))
    assert pts.size > 0, f"missing template for color {color}"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "template bbox must be 3x3"
    template = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    allowed = {int(bg), int(color)}
    assert set(map(int, template.flat)) <= allowed, "template must be bi-color"
    local_pts = {(int(r - r0), int(c - c0)) for r, c in pts}
    expected_pts = {
        (rr, cc)
        for rr in range(3)
        for cc in range(3)
        if int(template[rr, cc]) == int(color)
    }
    assert local_pts == expected_pts, "template pixels must occupy the whole detected 3x3 box"
    return template


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)

    shapes = _shape_objects(g, bg)
    colors = {shape["color"] for shape in shapes}
    other_colors = {int(v) for v in np.unique(g) if int(v) not in {bg, RED}}
    assert colors == other_colors, "every non-bg non-red color must belong to one shape"

    shape_mask = np.zeros_like(g, dtype=bool)
    for shape in shapes:
        for r, c in shape["coords"]:
            shape_mask[r, c] = True

    templates = {color: _template_for_color(g, bg, shape_mask, color) for color in colors}

    out = np.full_like(g, bg)
    for shape in shapes:
        color = int(shape["color"])
        template = templates[color]
        corner = _nearest_corner(shape["anchor"], shape["bbox"])
        for r, c in shape["coords"]:
            pr, pc = _pattern_indices(r, c, shape["bbox"], corner)
            out[r, c] = color if int(template[pr, pc]) == color else bg
    return out


def _make_pattern(bg: int, color: int, kind: str) -> np.ndarray:
    arr = np.full((3, 3), bg, dtype=int)
    if kind == "checker_a":
        for r in range(3):
            for c in range(3):
                if (r + c) % 2 == 0:
                    arr[r, c] = color
    elif kind == "checker_b":
        for r in range(3):
            for c in range(3):
                if (r + c) % 2 == 1:
                    arr[r, c] = color
    elif kind == "stripe_v":
        arr[:, ::2] = color
    elif kind == "stripe_h":
        arr[::2, :] = color
    else:
        raise ValueError(kind)
    return arr


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen[start] = True
    total = 0
    while q:
        r, c = q.popleft()
        total += 1
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return total == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any(bg & (~seen)))


def _eligible_anchors(mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = mask.shape
    result = {name: [] for name in CORNER_NAMES}
    bbox = (0, 0, h - 1, w - 1)
    for r, c in map(tuple, np.argwhere(mask)):
        result[_nearest_corner((int(r), int(c)), bbox)].append((int(r), int(c)))
    return result


def _generate_shape_mask(rng: random.Random, corner: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(200):
        bh = rng.randint(4, 6)
        bw = rng.randint(4, 6)
        if bh * bw < 10:
            continue
        min_area = max(8, int(np.ceil(0.6 * bh * bw)))
        max_area = min(20, bh * bw - 1)
        if min_area > max_area:
            continue
        target_area = rng.randint(min_area, max_area)
        mask = np.ones((bh, bw), dtype=bool)
        protected = {
            "TL": (0, 0),
            "TR": (0, bw - 1),
            "BL": (bh - 1, 0),
            "BR": (bh - 1, bw - 1),
        }[corner]
        while int(mask.sum()) > target_area:
            candidates = [tuple(map(int, p)) for p in np.argwhere(mask) if tuple(map(int, p)) != protected]
            rng.shuffle(candidates)
            removed = False
            for r, c in candidates:
                trial = mask.copy()
                trial[r, c] = False
                if not _touches_all_sides(trial):
                    continue
                if not _mask_connected(trial):
                    continue
                if _has_hole(trial):
                    continue
                mask = trial
                removed = True
                break
            if not removed:
                break
        if int(mask.sum()) != target_area:
            continue
        if not np.any(~mask):
            continue
        anchors = _eligible_anchors(mask)[corner]
        if not anchors:
            continue
        non_corner = [pt for pt in anchors if pt != protected]
        if non_corner and rng.random() < 0.7:
            anchor = rng.choice(non_corner)
        else:
            anchor = rng.choice(anchors)
        return mask, anchor
    return None


def _can_place_box(
    occupied: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    pad: int = 1,
) -> bool:
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _reserve_box(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occupied[top : top + h, left : left + w] = True


def _shape_output_quality(
    output: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
) -> bool:
    values = [int(output[r, c]) for r, c in coords]
    color_count = sum(v == color for v in values)
    return 0 < color_count < len(values)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg_choices = [c for c in range(10) if c != RED]
    bg = rng.choice(bg_choices)
    palette = [c for c in range(10) if c not in {bg, RED}]
    rng.shuffle(palette)

    num_pairs = rng.randint(2, 4)
    colors = palette[:num_pairs]
    H = rng.randint(18, 25)
    W = rng.randint(18, 25)
    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    specs = []
    shape_corners = list(CORNER_NAMES)
    rng.shuffle(shape_corners)
    for idx, color in enumerate(colors):
        corner = shape_corners[idx % 4]
        if idx >= 4:
            corner = rng.choice(CORNER_NAMES)
        shape = _generate_shape_mask(rng, corner)
        if shape is None:
            return None
        mask, anchor = shape
        pattern_kind = rng.choice(PATTERN_KINDS)
        specs.append(
            {
                "color": color,
                "corner": corner,
                "mask": mask,
                "anchor": anchor,
                "pattern": _make_pattern(bg, color, pattern_kind),
                "shape_pos": None,
                "template_pos": None,
            }
        )

    rng.shuffle(specs)
    for spec in specs:
        mask = spec["mask"]
        bh, bw = mask.shape
        placed = False
        for _ in range(250):
            top = rng.randint(0, H - bh)
            left = rng.randint(0, W - bw)
            if not _can_place_box(occupied, top, left, bh, bw, pad=1):
                continue
            _reserve_box(occupied, top, left, bh, bw)
            spec["shape_pos"] = (top, left)
            for r in range(bh):
                for c in range(bw):
                    if mask[r, c]:
                        grid[top + r, left + c] = spec["color"]
            ar, ac = spec["anchor"]
            grid[top + ar, left + ac] = RED
            placed = True
            break
        if not placed:
            return None

    rng.shuffle(specs)
    for spec in specs:
        placed = False
        for _ in range(250):
            top = rng.randint(0, H - 3)
            left = rng.randint(0, W - 3)
            if not _can_place_box(occupied, top, left, 3, 3, pad=1):
                continue
            _reserve_box(occupied, top, left, 3, 3)
            spec["template_pos"] = (top, left)
            patch = spec["pattern"]
            for r in range(3):
                for c in range(3):
                    grid[top + r, left + c] = int(patch[r, c])
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.any(out == RED):
        return None

    if _bg_color(grid) != bg:
        return None

    for spec in specs:
        top, left = spec["shape_pos"]
        coords = [
            (top + r, left + c)
            for r in range(spec["mask"].shape[0])
            for c in range(spec["mask"].shape[1])
            if spec["mask"][r, c]
        ]
        if not _shape_output_quality(out, coords, spec["color"]):
            return None
        tt, tl = spec["template_pos"]
        if np.any(out[tt : tt + 3, tl : tl + 3] != bg):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
