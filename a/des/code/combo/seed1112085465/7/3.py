import numpy as np


BG = 0
OPS = ("identity", "hflip", "vflip", "rot180")
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
RNG = np.random.default_rng()


def _bg_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(flat, minlength=max(10, int(flat.max()) + 1))
    return int(np.argmax(counts))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _apply_op(pattern: np.ndarray, op: str) -> np.ndarray:
    if op == "identity":
        return pattern.copy()
    if op == "hflip":
        return np.fliplr(pattern)
    if op == "vflip":
        return np.flipud(pattern)
    if op == "rot180":
        return np.rot90(pattern, 2)
    raise ValueError(f"unknown op {op}")


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, str], tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int, list[int], tuple[int, int, int, int]]] = []
    for pts in _components_mask(grid != bg):
        bbox = _bbox(pts)
        r0, c0, r1, c1 = bbox
        hh, ww = _bbox_hw(bbox)
        if ww != 1 or hh not in (3, 4) or len(pts) != hh:
            continue
        colors = [int(grid[r, c0]) for r in range(r0, r1 + 1)]
        if len(set(colors)) != hh:
            continue
        corner_dist = min(
            r0 + c0,
            r0 + (w - 1 - c1),
            (h - 1 - r1) + c0,
            (h - 1 - r1) + (w - 1 - c1),
        )
        candidates.append((corner_dist, r0, c0, hh, colors, bbox))
    assert candidates, "legend not found"
    _, _, _, hh, colors, bbox = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    mapping = {int(colors[i]): OPS[i] for i in range(hh)}
    return mapping, bbox


def _is_ring_component(grid: np.ndarray, points: list[tuple[int, int]]) -> bool:
    bbox = _bbox(points)
    r0, c0, r1, c1 = bbox
    hh, ww = _bbox_hw(bbox)
    if hh < 7 or ww < 7:
        return False
    if len(points) != 2 * (hh + ww) - 4:
        return False
    color = int(grid[points[0][0], points[0][1]])
    pset = set(points)
    for r, c in points:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
        if int(grid[r, c]) != color:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pset or (r1, c) not in pset:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pset or (r, c1) not in pset:
            return False
    return True


def _is_full_3x3(points: list[tuple[int, int]]) -> bool:
    bbox = _bbox(points)
    return _bbox_hw(bbox) == (3, 3) and len(points) == 9


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the frame-color transformation stack to the enclosed 3x3 pattern.

    Input structure:
    - Background is black.
    - A colored 3x3 pattern sits inside one or more concentric 1-cell-thick frames.
    - A vertical corner legend maps frame colors to identity / horizontal flip /
      vertical flip / 180-degree rotation.

    Rule:
    - Read the corner legend top-to-bottom to build the color->operation mapping.
    - Process the nested frames from inner to outer, applying each mapped
      transformation to the current 3x3 pattern.
    - Return only the final 3x3 pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mapping, legend_bbox = _find_legend(g, bg)

    frame_records: list[dict[str, object]] = []
    pattern_records: list[dict[str, object]] = []
    for pts in _components_mask(g != bg):
        bbox = _bbox(pts)
        if bbox == legend_bbox:
            continue
        center = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        color = int(g[pts[0][0], pts[0][1]])
        if color in mapping and _is_ring_component(g, pts):
            frame_records.append({"bbox": bbox, "center": center, "color": color})
        elif _is_full_3x3(pts):
            pattern_records.append({"bbox": bbox, "center": center})

    assert frame_records, "no frames found"
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for frame in frame_records:
        groups.setdefault(frame["center"], []).append(frame)

    h, w = g.shape
    doubled_grid_center = (h - 1, w - 1)
    center, frames = max(
        groups.items(),
        key=lambda item: (
            len(item[1]),
            -(abs(item[0][0] - doubled_grid_center[0]) + abs(item[0][1] - doubled_grid_center[1])),
        ),
    )
    frames = sorted(
        frames,
        key=lambda frame: _bbox_hw(frame["bbox"])[0] * _bbox_hw(frame["bbox"])[1],  # type: ignore[index]
    )

    pattern_bbox = None
    for record in pattern_records:
        if record["center"] == center:
            pattern_bbox = record["bbox"]  # type: ignore[assignment]
            break
    if pattern_bbox is None:
        inner = frames[0]["bbox"]  # type: ignore[index]
        r0, c0, r1, c1 = inner
        pattern = g[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1].copy()
    else:
        r0, c0, r1, c1 = pattern_bbox
        pattern = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    assert pattern.shape == (3, 3), "pattern must be 3x3"

    out = pattern.copy()
    for frame in frames:
        out = _apply_op(out, mapping[int(frame["color"])])  # type: ignore[index]
    return out


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _random_pattern(colors: list[int]) -> np.ndarray:
    for _ in range(200):
        pattern = RNG.choice(colors, size=(3, 3), replace=True).astype(int)
        if len(set(map(int, pattern.ravel()))) < 3:
            continue
        if any(np.array_equal(pattern, _apply_op(pattern, op)) for op in OPS[1:]):
            continue
        return pattern
    raise RuntimeError("failed to sample an asymmetric 3x3 pattern")


def _place_legend(grid: np.ndarray, colors: list[int], corner: str) -> None:
    h, w = grid.shape
    legend_h = len(colors)
    if corner == "tl":
        r0, c = 0, 0
    elif corner == "tr":
        r0, c = 0, w - 1
    elif corner == "bl":
        r0, c = h - legend_h, 0
    elif corner == "br":
        r0, c = h - legend_h, w - 1
    else:
        raise ValueError(f"unknown corner {corner}")
    for i, color in enumerate(colors):
        grid[r0 + i, c] = color


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    legend_len = int(RNG.choice([3, 4]))
    legend_colors = list(map(int, RNG.choice(COLOR_POOL, size=legend_len, replace=False)))
    mapping = {legend_colors[i]: OPS[i] for i in range(legend_len)}

    n_layers = int(RNG.integers(1, legend_len + 1))
    frame_colors = list(map(int, RNG.choice(legend_colors, size=n_layers, replace=False)))

    pattern_colors = [c for c in COLOR_POOL if c not in legend_colors]
    pattern = _random_pattern(pattern_colors)

    expected = pattern.copy()
    for color in frame_colors:
        expected = _apply_op(expected, mapping[color])

    outer_size = 4 * n_layers + 3
    h = int(RNG.integers(outer_size + 4, outer_size + 13))
    w = int(RNG.integers(outer_size + 4, outer_size + 13))
    grid = np.full((h, w), BG, dtype=int)

    top = (h - outer_size) // 2
    left = (w - outer_size) // 2
    pattern_top = top + 2 * n_layers
    pattern_left = left + 2 * n_layers
    grid[pattern_top : pattern_top + 3, pattern_left : pattern_left + 3] = pattern

    for inner_idx, color in enumerate(frame_colors):
        size = 4 * inner_idx + 7
        inset = 2 * (n_layers - 1 - inner_idx)
        bbox = (top + inset, left + inset, top + inset + size - 1, left + inset + size - 1)
        _paint_ring(grid, bbox, color)

    corner = str(RNG.choice(["tl", "tr", "bl", "br"]))
    _place_legend(grid, legend_colors, corner)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    try:
        grid, expected = _build_example()
    except RuntimeError:
        return None
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
