import random
import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
AZURE = 8


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, ...], ...]]:
    return mask.shape, tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _maximal_step2_segments(indices: list[int]) -> list[tuple[int, int]]:
    if len(indices) < 2:
        return []
    segments: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    length = 1
    for value in indices[1:]:
        if value - prev == 2:
            prev = value
            length += 1
            continue
        if length >= 2:
            segments.append((start, prev))
        start = value
        prev = value
        length = 1
    if length >= 2:
        segments.append((start, prev))
    return segments


def _is_dashed_rectangle(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r1 <= r0 or c1 <= c0:
        return False
    if (r1 - r0) % 2 != 0 or (c1 - c0) % 2 != 0:
        return False
    for c in range(c0, c1 + 1):
        expected = (c - c0) % 2 == 0
        if bool(grid[r0, c] == AZURE) != expected:
            return False
        if bool(grid[r1, c] == AZURE) != expected:
            return False
    for r in range(r0, r1 + 1):
        expected = (r - r0) % 2 == 0
        if bool(grid[r, c0] == AZURE) != expected:
            return False
        if bool(grid[r, c1] == AZURE) != expected:
            return False
    if r1 - r0 > 1 and c1 - c0 > 1 and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == AZURE):
        return False
    return True


def _extract_templates(grid: np.ndarray) -> list[np.ndarray]:
    greens = sorted((tuple(map(int, p)) for p in np.argwhere(grid == GREEN)), key=lambda rc: rc[1])
    assert len(greens) == 4

    red_components = []
    for pts in _components(grid == RED):
        r0, c0, r1, c1 = _bbox(pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True
        red_components.append((r0, c0, r1, c1, mask))

    templates: list[np.ndarray] = []
    used: set[int] = set()
    for gr, gc in greens:
        matches = [
            idx
            for idx, (_, c0, r1, _, _) in enumerate(red_components)
            if idx not in used and c0 == gc and r1 + 1 == gr
        ]
        assert len(matches) == 1
        used.add(matches[0])
        templates.append(red_components[matches[0]][4].copy())
    return templates


def _detect_dashed_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    azure = grid == AZURE

    horizontal: dict[tuple[int, int], list[int]] = {}
    for r in range(h):
        cols = [int(c) for c in np.flatnonzero(azure[r])]
        for c0, c1 in _maximal_step2_segments(cols):
            horizontal.setdefault((c0, c1), []).append(r)

    vertical: set[tuple[int, int, int]] = set()
    for c in range(w):
        rows = [int(r) for r in np.flatnonzero(azure[:, c])]
        for r0, r1 in _maximal_step2_segments(rows):
            vertical.add((c, r0, r1))

    rectangles = set()
    for (c0, c1), rows in horizontal.items():
        rows = sorted(rows)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                r0 = rows[i]
                r1 = rows[j]
                if (c0, r0, r1) not in vertical or (c1, r0, r1) not in vertical:
                    continue
                if _is_dashed_rectangle(grid, r0, c0, r1, c1):
                    rectangles.add((r0, c0, r1, c1))
    return sorted(rectangles)


def _stamp_centered(
    out: np.ndarray,
    template: np.ndarray,
    inner_r0: int,
    inner_c0: int,
    inner_r1: int,
    inner_c1: int,
) -> None:
    inner_h = inner_r1 - inner_r0 + 1
    inner_w = inner_c1 - inner_c0 + 1
    th, tw = template.shape
    assert th <= inner_h and tw <= inner_w
    top = inner_r0 + (inner_h - th) // 2
    left = inner_c0 + (inner_w - tw) // 2
    for dr, dc in np.argwhere(template):
        out[top + int(dr), left + int(dc)] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect azure dashed rectangles, count yellow markers inside each one,
    then stamp the corresponding bottom template at the region center.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    templates = _extract_templates(g)
    rectangles = _detect_dashed_rectangles(g)

    for r0, c0, r1, c1 in rectangles:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(inner == YELLOW))
        assert 1 <= count <= 4
        out[r0 + 1 : r1, c0 + 1 : c1][out[r0 + 1 : r1, c0 + 1 : c1] == YELLOW] = BLACK
        _stamp_centered(out, templates[count - 1], r0 + 1, c0 + 1, r1 - 1, c1 - 1)

    return out


def _shape(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    _shape(("#",)),
    _shape(("##",)),
    _shape(("###",)),
    _shape(("#.", "##")),
    _shape((".#", "##")),
    _shape(("##", "##")),
    _shape(("###", ".#.")),
    _shape(("#.", "#.", "##")),
    _shape((".#", ".#", "##")),
    _shape(("##.", ".##")),
    _shape((".##", "##.")),
    _shape((".#.", "###", ".#.")),
]


def _random_template(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    transformed = mask.copy()
    if rng.random() < 0.5:
        transformed = np.fliplr(transformed)
    transformed = np.rot90(transformed, rng.randrange(4))
    return _crop_mask(transformed)


def _choose_templates(rng: random.Random) -> list[np.ndarray]:
    for _ in range(200):
        templates: list[np.ndarray] = []
        seen = set()
        for idx in rng.sample(range(len(SHAPE_LIBRARY)), 4):
            template = _random_template(SHAPE_LIBRARY[idx], rng)
            sig = _shape_signature(template)
            if sig in seen:
                break
            seen.add(sig)
            templates.append(template)
        if len(templates) == 4:
            return templates
    raise RuntimeError("failed to choose distinct templates")


def _odd_at_least(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _draw_dashed_rectangle(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    for c in range(c0, c1 + 1, 2):
        grid[r0, c] = AZURE
        grid[r1, c] = AZURE
    for r in range(r0, r1 + 1, 2):
        grid[r, c0] = AZURE
        grid[r, c1] = AZURE


def _choose_counts(rng: random.Random, n_rectangles: int) -> list[int]:
    if n_rectangles == 4 or rng.random() < 0.5:
        counts = rng.sample([1, 2, 3, 4], n_rectangles)
    else:
        counts = [rng.randint(1, 4) for _ in range(n_rectangles)]
        if n_rectangles > 1 and len(set(counts)) == 1:
            counts[-1] = 1 + counts[-1] % 4
    rng.shuffle(counts)
    return counts


def _expected_output_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    templates = meta["templates"]
    for rect in meta["rectangles"]:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1][out[r0 + 1 : r1, c0 + 1 : c1] == YELLOW] = BLACK
        _stamp_centered(out, templates[rect["count"] - 1], r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    templates = _choose_templates(rng)
    widths = [int(t.shape[1]) for t in templates]
    heights = [int(t.shape[0]) for t in templates]
    gaps = [rng.randint(2, 4) for _ in range(3)]

    n_rectangles = rng.choice([2, 3, 4, 4])
    counts = _choose_counts(rng, n_rectangles)

    rect_specs = []
    for count in counts:
        th, tw = templates[count - 1].shape
        inner_h = _odd_at_least(max(3, int(th))) + 2 * rng.randint(0, 2)
        inner_w = _odd_at_least(max(3, int(tw))) + 2 * rng.randint(0, 3)
        rect_specs.append(
            {
                "count": count,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "bbox_h": inner_h + 2,
                "bbox_w": inner_w + 2,
            }
        )

    max_template_h = max(heights)
    rect_zone_h = max(max(spec["bbox_h"] for spec in rect_specs) + 1, 8 + 4 * n_rectangles) + rng.randint(0, 4)
    template_span = sum(widths) + sum(gaps)
    rect_span = max(spec["bbox_w"] for spec in rect_specs)
    width = max(template_span + 2, rect_span + 4, 12 + 5 * n_rectangles) + rng.randint(0, 6)
    height = rect_zone_h + max_template_h + 2

    grid = np.full((height, width), BLACK, dtype=int)

    total_template_width = sum(widths) + sum(gaps)
    start_col = rng.randint(1, width - total_template_width - 1)
    label_row = height - 1
    template_bottom = label_row - 1

    template_cols: list[int] = []
    cursor = start_col
    for idx, template in enumerate(templates):
        template_cols.append(cursor)
        th, tw = template.shape
        top = template_bottom - th + 1
        grid[top : top + th, cursor : cursor + tw][template] = RED
        grid[label_row, cursor] = GREEN
        cursor += tw
        if idx < 3:
            cursor += gaps[idx]

    occupied = np.zeros((rect_zone_h, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * n_rectangles
    order = sorted(
        range(n_rectangles),
        key=lambda idx: rect_specs[idx]["bbox_h"] * rect_specs[idx]["bbox_w"],
        reverse=True,
    )
    for idx in order:
        spec = rect_specs[idx]
        bh = spec["bbox_h"]
        bw = spec["bbox_w"]
        placed = False
        for _ in range(250):
            r0 = rng.randint(0, rect_zone_h - bh)
            c0 = rng.randint(0, width - bw)
            er0 = max(0, r0 - 1)
            ec0 = max(0, c0 - 1)
            er1 = min(rect_zone_h, r0 + bh + 1)
            ec1 = min(width, c0 + bw + 1)
            if occupied[er0:er1, ec0:ec1].any():
                continue
            occupied[er0:er1, ec0:ec1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    rectangles_meta = []
    for spec, placement in zip(rect_specs, placements):
        assert placement is not None
        r0, c0 = placement
        r1 = r0 + spec["bbox_h"] - 1
        c1 = c0 + spec["bbox_w"] - 1
        _draw_dashed_rectangle(grid, r0, c0, r1, c1)

        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        rng.shuffle(cells)
        for r, c in cells[: spec["count"]]:
            grid[r, c] = YELLOW

        rectangles_meta.append({"bbox": (r0, c0, r1, c1), "count": spec["count"]})

    meta = {"templates": templates, "rectangles": rectangles_meta, "template_cols": template_cols}
    return grid, meta


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        result = _generate_case(rng)
        if result is None:
            continue
        grid, meta = result
        expected = _expected_output_from_meta(grid, meta)
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
