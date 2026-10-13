import numpy as np
from collections import Counter, deque


SEPARATOR_COLOR = 5
SIDE_NAMES = ("top", "right", "bottom", "left")
ALL_COLORS = tuple(range(10))
ICON_PATTERNS = (
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
)


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        column = grid[:, c]
        if int(column[0]) != bg and np.all(column == column[0]):
            candidates.append(c)
    assert candidates, "expected a full-height separator column"
    mid = w // 2
    return min(candidates, key=lambda c: (abs(c - mid), c))


def _dashed_pattern(length: int) -> np.ndarray:
    pat = np.zeros(length, dtype=bool)
    pat[::2] = True
    pat[-1] = True
    return pat


def _classify_side(mask: np.ndarray) -> str | None:
    if np.all(mask):
        return "solid"
    if np.array_equal(mask, _dashed_pattern(len(mask))):
        return "dashed"
    return None


def _parse_sample_window(sub: np.ndarray, bg: int):
    if sub.shape[0] != sub.shape[1]:
        return None
    size = sub.shape[0]
    if size < 5 or size % 2 == 0:
        return None

    mid = size // 2
    label = int(sub[mid, mid])
    if label == bg:
        return None
    if int(np.count_nonzero(sub == label)) != 1:
        return None

    perimeter = np.zeros_like(sub, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    inner = (~perimeter).copy()
    inner[mid, mid] = False
    if np.any(sub[inner] != bg):
        return None

    border_colors = {int(v) for v in sub[perimeter].ravel() if int(v) != bg}
    if len(border_colors) != 1:
        return None
    border_color = next(iter(border_colors))
    if border_color == label:
        return None

    corners = (sub[0, 0], sub[0, -1], sub[-1, 0], sub[-1, -1])
    if any(int(v) != border_color for v in corners):
        return None

    side_masks = {
        "top": sub[0, :] == border_color,
        "right": sub[:, -1] == border_color,
        "bottom": sub[-1, :] == border_color,
        "left": sub[:, 0] == border_color,
    }
    side_types = {}
    for side, mask in side_masks.items():
        side_kind = _classify_side(mask)
        if side_kind is None:
            return None
        side_types[side] = side_kind

    return {
        "label_color": label,
        "border_color": border_color,
        "sample_size": size,
        "sides": side_types,
    }


def _boxes_overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _extract_styles(left: np.ndarray, bg: int) -> dict[int, dict]:
    h, w = left.shape
    found: dict[int, dict] = {}
    taken_boxes = []
    max_size = min(h, w)
    for size in range(max_size if max_size % 2 == 1 else max_size - 1, 4, -2):
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                parsed = _parse_sample_window(left[r0 : r0 + size, c0 : c0 + size], bg)
                if parsed is None:
                    continue
                box = (r0, c0, r0 + size - 1, c0 + size - 1)
                if any(_boxes_overlap(box, other) for other in taken_boxes):
                    continue
                label = int(parsed["label_color"])
                assert label not in found, "label colors must uniquely identify styles"
                parsed["bbox"] = box
                found[label] = parsed
                taken_boxes.append(box)
    assert found, "expected at least one style sample"
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _extract_containers(right: np.ndarray, bg: int, styles: dict[int, dict]) -> list[dict]:
    containers = []
    for comp in _components(right != bg):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        sub = right[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg):
            continue
        vals, counts = np.unique(sub, return_counts=True)
        order = np.argsort(-counts)
        colors = [int(vals[i]) for i in order]
        fill_color = colors[0]
        other_colors = [color for color in colors[1:] if color != fill_color]
        if len(other_colors) != 1:
            continue
        label_color = other_colors[0]
        if label_color not in styles:
            continue
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        containers.append(
            {
                "row": int(r0),
                "col": int(c0),
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
                "fill_color": int(fill_color),
                "label_color": int(label_color),
            }
        )
    assert containers, "expected at least one container"
    return containers


def _edge_indices(length: int, side_kind: str) -> np.ndarray:
    if side_kind == "solid":
        return np.arange(length)
    return np.flatnonzero(_dashed_pattern(length))


def _draw_styled_rect(out: np.ndarray, top: int, left: int, h: int, w: int, fill_color: int, style: dict) -> None:
    out[top : top + h, left : left + w] = fill_color
    border_color = int(style["border_color"])

    top_idx = _edge_indices(w, style["sides"]["top"])
    bottom_idx = _edge_indices(w, style["sides"]["bottom"])
    left_idx = _edge_indices(h, style["sides"]["left"])
    right_idx = _edge_indices(h, style["sides"]["right"])

    out[top, left + top_idx] = border_color
    out[top + h - 1, left + bottom_idx] = border_color
    out[top + left_idx, left] = border_color
    out[top + right_idx, left + w - 1] = border_color


def _render_nested(containers: list[dict], styles: dict[int, dict]) -> np.ndarray:
    ordered = sorted(
        containers,
        key=lambda item: (-item["area"], -item["h"], -item["w"], item["row"], item["col"]),
    )
    outer_h = int(ordered[0]["h"])
    outer_w = int(ordered[0]["w"])
    out = np.full((outer_h, outer_w), int(ordered[0]["fill_color"]), dtype=int)

    prev_top = 0
    prev_left = 0
    prev_h = outer_h
    prev_w = outer_w
    for idx, container in enumerate(ordered):
        h = int(container["h"])
        w = int(container["w"])
        if idx == 0:
            top = 0
            left = 0
        else:
            assert h <= prev_h and w <= prev_w, "nested containers must fit inside the previous one"
            top = prev_top + (prev_h - h) // 2
            left = prev_left + (prev_w - w) // 2
        _draw_styled_rect(
            out,
            top,
            left,
            h,
            w,
            int(container["fill_color"]),
            styles[int(container["label_color"])],
        )
        prev_top, prev_left, prev_h, prev_w = top, left, h, w
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep = _find_separator_col(g, bg)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    styles = _extract_styles(left, bg)
    containers = _extract_containers(right, bg, styles)
    return _render_nested(containers, styles)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each right-side container to the left-side border sample by label color,
    then nest the containers by descending area using the matched border styles.
    """
    return _solve(grid)


def _sample_odd(rng: np.random.Generator, low: int, high: int) -> int:
    choices = [v for v in range(low, high + 1) if v % 2 == 1]
    return int(rng.choice(choices))


def _make_sample(style: dict, bg: int) -> np.ndarray:
    size = int(style["sample_size"])
    sample = np.full((size, size), bg, dtype=int)
    _draw_styled_rect(sample, 0, 0, size, size, bg, style)
    mid = size // 2
    sample[mid, mid] = int(style["label_color"])
    return sample


def _place_icon(container: np.ndarray, label_color: int, pattern: np.ndarray, rng: np.random.Generator) -> None:
    h, w = container.shape
    ph, pw = pattern.shape
    base_r = (h - ph) // 2
    base_c = (w - pw) // 2

    min_dr = max(1 - base_r, -1)
    max_dr = min((h - 1) - (base_r + ph), 1)
    min_dc = max(1 - base_c, -1)
    max_dc = min((w - 1) - (base_c + pw), 1)

    dr_choices = list(range(min_dr, max_dr + 1))
    dc_choices = list(range(min_dc, max_dc + 1))
    dr = int(rng.choice(dr_choices))
    dc = int(rng.choice(dc_choices))

    r0 = base_r + dr
    c0 = base_c + dc
    mask = pattern.astype(bool)
    container[r0 : r0 + ph, c0 : c0 + pw][mask] = label_color


def _make_container(h: int, w: int, fill_color: int, label_color: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.full((h, w), fill_color, dtype=int)
    pattern = ICON_PATTERNS[int(rng.integers(len(ICON_PATTERNS)))]
    _place_icon(arr, label_color, pattern, rng)
    return arr


def _container_dims(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    dims = []
    min_outer = 5 + 2 * (count - 1)
    h = _sample_odd(rng, min_outer, 15)
    w = _sample_odd(rng, min_outer, 15)
    dims.append((h, w))
    for idx in range(1, count):
        remain = count - idx
        prev_h, prev_w = dims[-1]
        min_h = 5 + 2 * (remain - 1)
        min_w = 5 + 2 * (remain - 1)
        h_options = [prev_h - delta for delta in (2, 4) if prev_h - delta >= min_h]
        w_options = [prev_w - delta for delta in (2, 4) if prev_w - delta >= min_w]
        if not h_options or not w_options:
            raise ValueError("unable to sample nested dimensions")
        dims.append((int(rng.choice(h_options)), int(rng.choice(w_options))))
    return dims


def _place_rectangles(
    rects: list[np.ndarray],
    height: int,
    width: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    occupied = np.zeros((height, width), dtype=bool)
    placements = [None] * len(rects)
    order = sorted(range(len(rects)), key=lambda i: rects[i].shape[0] * rects[i].shape[1], reverse=True)
    for idx in order:
        rect = rects[idx]
        h, w = rect.shape
        placed = False
        for _ in range(300):
            r0 = int(rng.integers(1, height - h))
            c0 = int(rng.integers(1, width - w))
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            raise ValueError("failed to place container")
    return placements


def _choose_styles(rng: np.random.Generator, num_styles: int, bg: int) -> dict[int, dict]:
    palette = [c for c in ALL_COLORS if c not in {bg, SEPARATOR_COLOR}]
    rng.shuffle(palette)
    label_colors = palette[:num_styles]
    border_colors = palette[num_styles : 2 * num_styles]

    pattern_pool = []
    for bits in range(16):
        pattern_pool.append(tuple("dashed" if (bits >> i) & 1 else "solid" for i in range(4)))
    chosen = rng.choice(len(pattern_pool), size=num_styles, replace=False)

    styles = {}
    all_edges = []
    for i in range(num_styles):
        sides = {SIDE_NAMES[j]: pattern_pool[int(chosen[i])][j] for j in range(4)}
        all_edges.extend(sides.values())
        label = int(label_colors[i])
        styles[label] = {
            "label_color": label,
            "border_color": int(border_colors[i]),
            "sample_size": int(rng.choice([5, 7])),
            "sides": sides,
        }

    if all(edge == "solid" for edge in all_edges) or all(edge == "dashed" for edge in all_edges):
        raise ValueError("styles must cover both solid and dashed edges")
    return styles


def _gen_attempt(rng: np.random.Generator):
    bg = int(rng.choice([c for c in ALL_COLORS if c != SEPARATOR_COLOR]))
    num_styles = int(rng.integers(2, 5))
    num_containers = int(rng.integers(2, min(5, num_styles + 1)))
    repeat_labels = num_containers >= 3 and bool(rng.integers(2))
    distinct_used = num_containers - 1 if repeat_labels else num_containers
    if distinct_used > num_styles:
        raise ValueError("not enough styles for chosen container count")

    styles = _choose_styles(rng, num_styles, bg)
    style_labels = list(styles.keys())

    used_labels = list(rng.choice(style_labels, size=distinct_used, replace=False))
    assigned_labels = used_labels.copy()
    while len(assigned_labels) < num_containers:
        assigned_labels.append(int(rng.choice(used_labels)))
    rng.shuffle(assigned_labels)

    dims = _container_dims(rng, num_containers)
    fill_palette = [c for c in ALL_COLORS if c not in {bg, SEPARATOR_COLOR}]
    fills = []
    for idx, label in enumerate(assigned_labels):
        banned = {bg, SEPARATOR_COLOR, int(label), int(styles[int(label)]["border_color"])}
        candidates = [c for c in fill_palette if c not in banned]
        if idx > 0:
            nonrepeat = [c for c in candidates if c != fills[-1]]
            if nonrepeat:
                candidates = nonrepeat
        fills.append(int(rng.choice(candidates)))

    container_specs = []
    for idx, ((h, w), label, fill) in enumerate(zip(dims, assigned_labels, fills)):
        container_specs.append(
            {
                "row": idx,
                "col": 0,
                "h": int(h),
                "w": int(w),
                "area": int(h * w),
                "fill_color": int(fill),
                "label_color": int(label),
            }
        )

    sample_sizes = [int(styles[label]["sample_size"]) for label in styles]
    gap_between = [int(rng.integers(1, 3)) for _ in range(max(0, num_styles - 1))]
    sample_block_h = sum(sample_sizes) + sum(gap_between)
    outer_h = max(h for h, _ in dims)
    outer_w = max(w for _, w in dims)
    grid_h = max(sample_block_h + 2, outer_h + 10, 20 + 2 * num_containers) + int(rng.integers(0, 4))
    left_w = max(sample_sizes) + int(rng.integers(4, 7))
    right_w = max(outer_w + 8, 18 + 3 * num_containers) + int(rng.integers(0, 5))

    left = np.full((grid_h, left_w), bg, dtype=int)
    sample_order = style_labels.copy()
    rng.shuffle(sample_order)
    total_sample_h = sum(int(styles[label]["sample_size"]) for label in sample_order) + sum(gap_between)
    top_pad = int(rng.integers(1, grid_h - total_sample_h))
    cursor = top_pad
    for idx, label in enumerate(sample_order):
        sample = _make_sample(styles[int(label)], bg)
        size = sample.shape[0]
        c0 = int(rng.integers(1, left_w - size))
        left[cursor : cursor + size, c0 : c0 + size] = sample
        cursor += size
        if idx < len(gap_between):
            cursor += gap_between[idx]

    rects = [_make_container(spec["h"], spec["w"], spec["fill_color"], spec["label_color"], rng) for spec in container_specs]
    placements = _place_rectangles(rects, grid_h, right_w, rng)
    right = np.full((grid_h, right_w), bg, dtype=int)
    placed_specs = []
    for spec, rect, (r0, c0) in zip(container_specs, rects, placements):
        h, w = rect.shape
        right[r0 : r0 + h, c0 : c0 + w] = rect
        placed = dict(spec)
        placed["row"] = int(r0)
        placed["col"] = int(c0)
        placed_specs.append(placed)

    grid = np.full((grid_h, left_w + 1 + right_w), bg, dtype=int)
    grid[:, :left_w] = left
    grid[:, left_w] = SEPARATOR_COLOR
    grid[:, left_w + 1 :] = right

    expected = _render_nested(placed_specs, styles)

    sep = _find_separator_col(grid, bg)
    if sep != left_w:
        raise ValueError("separator detection mismatch")
    parsed_styles = _extract_styles(grid[:, :sep], bg)
    if set(parsed_styles.keys()) != set(styles.keys()):
        raise ValueError("style parsing mismatch")
    parsed_containers = _extract_containers(grid[:, sep + 1 :], bg, parsed_styles)
    if len(parsed_containers) != num_containers:
        raise ValueError("container parsing mismatch")
    if not np.array_equal(_solve(grid), expected):
        raise ValueError("solver/output mismatch")

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        try:
            return _gen_attempt(rng)
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("generate_input failed after 300 attempts")
