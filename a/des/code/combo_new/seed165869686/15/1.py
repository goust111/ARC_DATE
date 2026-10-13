import random

import numpy as np


BACKGROUND = 0
REGION_FRAME = 5
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
PATTERN_KINDS = ("checker", "hstripe", "vstripe", "cross", "diag", "dots")
GEN_RETRIES = 300
STRIP_GAP = 2
BLOCK_GAP_CHOICES = (1, 2)
REGION_LAYOUTS = {
    2: ((2,), (1, 1)),
    3: ((2, 1), (1, 2)),
    4: ((2, 2),),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

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
            parts.append(np.array(pts, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_edge(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _is_rect_frame(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not (
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    ):
        return False

    inner = sub[1:-1, 1:-1]
    return inner.size > 0 and np.any(inner != color)


def _find_frame_groups(grid: np.ndarray, bg: int) -> dict[int, list[tuple[int, int, int, int]]]:
    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        boxes = []
        for comp in _components(grid == color):
            box = _bbox(comp)
            if _is_rect_frame(grid, color, box):
                boxes.append(box)
        if boxes:
            groups[color] = boxes
    return groups


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ph, pw = pattern.shape
    h, w = shape
    rows = np.arange(h) % ph
    cols = np.arange(w) % pw
    return pattern[rows[:, None], cols[None, :]].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray enclosed region by tiling the interior pattern from the same-colored edge template.

    Input structure:
    - The background is black.
    - Several gray rectangular frames enclose empty regions; each enclosed region contains exactly one colored marker pixel.
    - Along one outer edge there are one or more colored rectangular template blocks whose border color matches one marker color, and whose interior contains a black-and-color pattern.

    Rule:
    - Detect each gray closed rectangle and read the unique marker color inside it.
    - Find the edge template whose border has the same color, and extract that template's interior pattern.
    - Replace the whole interior of the gray rectangle, including the marker, with a top-left aligned periodic tiling of the extracted pattern.
    - Keep the gray region borders and the template blocks unchanged.

    Key insight:
    - The interior marker is only an index key; it points to a same-colored template on the boundary, and that template's interior is the pattern to repeat inside the matching gray region.
    """
    g = np.asarray(grid, dtype=int)
    bg = BACKGROUND

    frame_groups = _find_frame_groups(g, bg)
    assert frame_groups, "expected at least one rectangular frame"

    gray = max(
        frame_groups,
        key=lambda color: (
            sum(not _touches_edge(box, g.shape) for box in frame_groups[color]),
            len(frame_groups[color]),
        ),
    )
    region_boxes = [box for box in frame_groups[gray] if not _touches_edge(box, g.shape)]
    assert region_boxes, "expected at least one interior gray frame"

    templates: dict[int, np.ndarray] = {}
    for color, boxes in frame_groups.items():
        if color == gray:
            continue
        edge_boxes = [box for box in boxes if _touches_edge(box, g.shape)]
        if not edge_boxes:
            continue
        box = max(edge_boxes, key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))
        r0, c0, r1, c1 = box
        templates[color] = g[r0 + 1 : r1, c0 + 1 : c1].copy()

    assert templates, "expected at least one edge template"

    out = g.copy()
    for r0, c0, r1, c1 in region_boxes:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_colors = [int(v) for v in np.unique(inner) if int(v) not in {bg, gray} and int(v) in templates]
        assert len(marker_colors) == 1, "each region must contain exactly one template-linked marker color"
        pattern = templates[marker_colors[0]]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(pattern, inner.shape)

    return out


def _pattern_mask(kind: str, h: int, w: int) -> np.ndarray:
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]

    if kind == "checker":
        return (rr + cc) % 2 == 0
    if kind == "hstripe":
        return rr % 2 == 0
    if kind == "vstripe":
        return cc % 2 == 0
    if kind == "cross":
        return (rr == h // 2) | (cc == w // 2)
    if kind == "diag":
        return (rr - cc) % 3 == 0
    if kind == "dots":
        return (rr % 2 == 0) & (cc % 2 == 0)
    raise ValueError(f"unknown pattern kind: {kind}")


def _make_pattern(rng: random.Random, color: int, seen: set[tuple[bytes, tuple[int, int]]]) -> np.ndarray:
    for _ in range(40):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        kind = rng.choice(PATTERN_KINDS)
        mask = _pattern_mask(kind, h, w)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.25:
            mask = ~mask
        if not mask.any() or mask.all():
            continue
        sig = (mask.tobytes(), mask.shape)
        if sig in seen:
            continue
        seen.add(sig)
        return np.where(mask, color, BACKGROUND).astype(int)

    raise RuntimeError("failed to create a non-degenerate pattern")


def _choose_layout(n: int, rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.choice(REGION_LAYOUTS[n]))


def _pack_regions(specs: list[dict], rng: random.Random):
    items = specs[:]
    rng.shuffle(items)
    layout = _choose_layout(len(items), rng)

    rows = []
    idx = 0
    for count in layout:
        row_items = items[idx : idx + count]
        idx += count
        gaps = [rng.choice(BLOCK_GAP_CHOICES) for _ in range(max(0, count - 1))]
        row_w = sum(item["outer_w"] for item in row_items) + sum(gaps)
        row_h = max(item["outer_h"] for item in row_items)
        rows.append((row_items, gaps, row_w, row_h))

    v_gaps = [rng.choice(BLOCK_GAP_CHOICES) for _ in range(max(0, len(rows) - 1))]
    total_w = max(row_w for _, _, row_w, _ in rows)
    total_h = sum(row_h for _, _, _, row_h in rows) + sum(v_gaps)

    placed = []
    top = 0
    for row_index, (row_items, gaps, row_w, row_h) in enumerate(rows):
        left = rng.randint(0, total_w - row_w)
        cursor = left
        for item_index, item in enumerate(row_items):
            jitter = rng.randint(0, row_h - item["outer_h"])
            r0 = top + jitter
            c0 = cursor
            placed.append((item, (r0, c0, r0 + item["outer_h"] - 1, c0 + item["outer_w"] - 1)))
            cursor += item["outer_w"]
            if item_index < len(gaps):
                cursor += gaps[item_index]
        top += row_h
        if row_index < len(v_gaps):
            top += v_gaps[row_index]

    return placed, total_h, total_w


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int):
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_input(specs: list[dict], placements: list[tuple[dict, tuple[int, int, int, int]]], orient: str, rng: random.Random):
    if orient == "right":
        strip_w = max(spec["template_w"] for spec in specs)
        template_gaps = [rng.choice(BLOCK_GAP_CHOICES) for _ in range(max(0, len(specs) - 1))]
        template_total_h = sum(spec["template_h"] for spec in specs) + sum(template_gaps)
        pack_h = max(box[2] for _, box in placements) + 1
        pack_w = max(box[3] for _, box in placements) + 1

        min_h = max(pack_h + 2, template_total_h)
        min_w = pack_w + 1 + STRIP_GAP + strip_w
        if min_h > 30 or min_w > 30:
            return None

        h = rng.randint(min_h, 30)
        w = rng.randint(min_w, 30)
        main_w = w - STRIP_GAP - strip_w
        region_r_off = rng.randint(1, h - pack_h - 1)
        region_c_off = rng.randint(1, main_w - pack_w)
        template_top = rng.randint(0, h - template_total_h)

        grid = np.full((h, w), BACKGROUND, dtype=int)
        for spec, (r0, c0, r1, c1) in placements:
            box = (r0 + region_r_off, c0 + region_c_off, r1 + region_r_off, c1 + region_c_off)
            spec["region_bbox"] = box
            _draw_frame(grid, box, REGION_FRAME)
            ir0, ic0, ir1, ic1 = box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1
            mr = rng.randint(ir0, ir1)
            mc = rng.randint(ic0, ic1)
            grid[mr, mc] = spec["color"]

        top = template_top
        template_specs = specs[:]
        rng.shuffle(template_specs)
        for index, spec in enumerate(template_specs):
            bh, bw = spec["template_h"], spec["template_w"]
            r0 = top
            c0 = w - bw
            box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            spec["template_bbox"] = box
            _draw_frame(grid, box, spec["color"])
            grid[r0 + 1 : r0 + bh - 1, c0 + 1 : c0 + bw - 1] = spec["pattern"]
            top += bh
            if index < len(template_gaps):
                top += template_gaps[index]

        return grid

    strip_h = max(spec["template_h"] for spec in specs)
    template_gaps = [rng.choice(BLOCK_GAP_CHOICES) for _ in range(max(0, len(specs) - 1))]
    template_total_w = sum(spec["template_w"] for spec in specs) + sum(template_gaps)
    pack_h = max(box[2] for _, box in placements) + 1
    pack_w = max(box[3] for _, box in placements) + 1

    min_h = pack_h + 1 + STRIP_GAP + strip_h
    min_w = max(pack_w + 2, template_total_w)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    main_h = h - STRIP_GAP - strip_h
    region_r_off = rng.randint(1, main_h - pack_h)
    region_c_off = rng.randint(1, w - pack_w - 1)
    template_left = rng.randint(0, w - template_total_w)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for spec, (r0, c0, r1, c1) in placements:
        box = (r0 + region_r_off, c0 + region_c_off, r1 + region_r_off, c1 + region_c_off)
        spec["region_bbox"] = box
        _draw_frame(grid, box, REGION_FRAME)
        ir0, ic0, ir1, ic1 = box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1
        mr = rng.randint(ir0, ir1)
        mc = rng.randint(ic0, ic1)
        grid[mr, mc] = spec["color"]

    left = template_left
    template_specs = specs[:]
    rng.shuffle(template_specs)
    for index, spec in enumerate(template_specs):
        bh, bw = spec["template_h"], spec["template_w"]
        r0 = h - bh
        c0 = left
        box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        spec["template_bbox"] = box
        _draw_frame(grid, box, spec["color"])
        grid[r0 + 1 : r0 + bh - 1, c0 + 1 : c0 + bw - 1] = spec["pattern"]
        left += bw
        if index < len(template_gaps):
            left += template_gaps[index]

    return grid


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["region_bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(spec["pattern"], (spec["inner_h"], spec["inner_w"]))
    return out


def _gen_attempt(rng: random.Random):
    count = rng.randint(2, 4)
    colors = rng.sample(TEMPLATE_COLORS, count)
    seen_patterns: set[tuple[bytes, tuple[int, int]]] = set()
    specs = []

    for color in colors:
        pattern = _make_pattern(rng, color, seen_patterns)
        ph, pw = pattern.shape
        inner_h = min(7, ph + rng.randint(1, 3))
        inner_w = min(7, pw + rng.randint(1, 3))
        specs.append(
            {
                "color": color,
                "pattern": pattern,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "template_h": ph + 2,
                "template_w": pw + 2,
            }
        )

    placements, _, _ = _pack_regions(specs, rng)
    orient = rng.choice(("right", "bottom"))
    grid = _render_input(specs, placements, orient, rng)
    if grid is None:
        return None

    expected = _render_expected(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
