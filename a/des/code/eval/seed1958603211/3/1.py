import math
import random

import numpy as np


BG = 0
COLORS = list(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _outer_path(frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    path: list[tuple[int, int]] = []
    for c in range(c0 - 1, c1 + 2):
        path.append((r0 - 1, c))
    for r in range(r0, r1 + 2):
        path.append((r, c1 + 1))
    for c in range(c1, c0 - 2, -1):
        path.append((r1 + 1, c))
    for r in range(r1, r0 - 1, -1):
        path.append((r, c0 - 1))
    return path


def _find_template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for comp in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(comp)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        touches_corner = (r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)
        if not touches_corner:
            continue
        if len(comp) != area:
            continue
        if r1 == r0 and c1 == c0:
            continue
        candidates.append((area, r1 - r0 + 1, (r0, c0, r1, c1)))
    assert candidates, "template block not found"
    candidates.sort()
    return candidates[-1][2]


def _is_frame_component(
    grid: np.ndarray, color: int, comp: list[tuple[int, int]]
) -> tuple[bool, tuple[int, int, int, int] | None]:
    r0, c0, r1, c1 = _bbox(comp)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False, None
    if len(comp) != 2 * (height + width) - 4:
        return False, None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return False, None
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False, None
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False, None
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False, None
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False, None
    return True, (r0, c0, r1, c1)


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _max_nested_depth(
    frame_bbox: tuple[int, int, int, int], object_bboxes: list[tuple[int, int, int, int]]
) -> int:
    fr0, fc0, fr1, fc1 = frame_bbox
    inside = [
        bbox
        for bbox in object_bboxes
        if fr0 < bbox[0] and fc0 < bbox[1] and bbox[2] < fr1 and bbox[3] < fc1
    ]
    if not inside:
        return 0

    memo: dict[int, int] = {}

    def depth_at(i: int) -> int:
        if i in memo:
            return memo[i]
        best = 1
        for j, child in enumerate(inside):
            if i != j and _strictly_contains(inside[i], child):
                best = max(best, 1 + depth_at(j))
        memo[i] = best
        return best

    return max(depth_at(i) for i in range(len(inside)))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template_bbox = _find_template_bbox(g)
    tr0, tc0, tr1, tc1 = template_bbox
    templates = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    template_mask = np.zeros(g.shape, dtype=bool)
    template_mask[tr0 : tr1 + 1, tc0 : tc1 + 1] = True

    frames: list[tuple[int, int, int, int]] = []
    inner_bboxes: list[tuple[int, int, int, int]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        mask = (g == color) & (~template_mask)
        for comp in _components(mask):
            is_frame, bbox = _is_frame_component(g, color, comp)
            if is_frame:
                assert bbox is not None
                frames.append(bbox)
            else:
                inner_bboxes.append(_bbox(comp))

    out = g.copy()
    for frame_bbox in frames:
        depth = _max_nested_depth(frame_bbox, inner_bboxes)
        assert depth < templates.shape[0]
        pattern = templates[depth]
        for idx, (r, c) in enumerate(_outer_path(frame_bbox)):
            if not template_mask[r, c]:
                out[r, c] = int(pattern[idx % pattern.size])
    return out


TEMPLATE_ROWS = 4
FRAME_COUNT_RANGE = (2, 4)
GRID_SIDE_RANGE = (26, 36)
FRAME_PAD = 2
TEMPLATE_PAD = 1
TOP_LEVEL_MARGIN = 1
INNER_LEVEL_MARGIN = 2
MAX_DEPTH = TEMPLATE_ROWS - 1
GEN_ATTEMPTS = 400


def _min_box_size(levels: int) -> int:
    return 2 + 4 * max(0, levels - 1)


def _sample_shrink(
    container: tuple[int, int, int, int], min_margin: int, min_h: int, min_w: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = container
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height - min_h >= 2 * min_margin
    assert width - min_w >= 2 * min_margin

    top = random.randint(min_margin, height - min_h - min_margin)
    bottom = random.randint(min_margin, height - min_h - top)
    left = random.randint(min_margin, width - min_w - min_margin)
    right = random.randint(min_margin, width - min_w - left)
    return (r0 + top, c0 + left, r1 - bottom, c1 - right)


def _sample_frame_spec(depth: int) -> dict:
    inner_min = 3 if depth == 0 else _min_box_size(depth) + 2 * TOP_LEVEL_MARGIN
    inner_h = random.randint(inner_min, inner_min + 4)
    inner_w = random.randint(inner_min, inner_min + 6)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    frame_color = random.choice(COLORS)
    chain_colors = random.sample([c for c in COLORS if c != frame_color], depth)

    rects: list[dict] = []
    container = (1, 1, frame_h - 2, frame_w - 2)
    for level in range(depth):
        remaining = depth - level
        min_size = _min_box_size(remaining)
        min_margin = TOP_LEVEL_MARGIN if level == 0 else INNER_LEVEL_MARGIN
        bbox = _sample_shrink(container, min_margin, min_size, min_size)
        rects.append({"bbox": bbox, "color": chain_colors[level]})
        container = bbox

    return {
        "depth": depth,
        "height": frame_h,
        "width": frame_w,
        "frame_color": frame_color,
        "rects": rects,
    }


def _frame_perimeter(spec: dict) -> int:
    return 2 * (int(spec["height"]) + int(spec["width"])) + 4


def _divisors(n: int) -> list[int]:
    return [d for d in range(2, min(8, n) + 1) if n % d == 0]


def _sample_templates(length: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    while len(rows) < TEMPLATE_ROWS:
        row = np.array([random.choice(COLORS) for _ in range(length)], dtype=int)
        if len(set(int(v) for v in row)) < 2:
            row[random.randrange(length)] = random.choice([c for c in COLORS if c != int(row[0])])
        if any(np.array_equal(row, prev) for prev in rows):
            continue
        rows.append(row)
    return np.vstack(rows)


def _template_position(corner: str, grid_h: int, grid_w: int, width: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, grid_w - width
    if corner == "bl":
        return grid_h - TEMPLATE_ROWS, 0
    if corner == "br":
        return grid_h - TEMPLATE_ROWS, grid_w - width
    raise ValueError(corner)


def _reserve(rect: tuple[int, int, int, int], pad: int, occ: np.ndarray) -> bool:
    h, w = occ.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    if np.any(occ[rr0 : rr1 + 1, cc0 : cc1 + 1]):
        return False
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _place_frames(specs: list[dict], grid_h: int, grid_w: int, template_rect: tuple[int, int, int, int]) -> bool:
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    if not _reserve(template_rect, TEMPLATE_PAD, occ):
        return False

    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        placed = False
        for _ in range(250):
            max_r = grid_h - int(spec["height"]) - FRAME_PAD - 1
            max_c = grid_w - int(spec["width"]) - FRAME_PAD - 1
            if max_r < FRAME_PAD or max_c < FRAME_PAD:
                return False
            r0 = random.randint(FRAME_PAD, max_r)
            c0 = random.randint(FRAME_PAD, max_c)
            rect = (r0, c0, r0 + int(spec["height"]) - 1, c0 + int(spec["width"]) - 1)
            if _reserve(rect, FRAME_PAD, occ):
                spec["bbox"] = rect
                placed = True
                break
        if not placed:
            return False
    return True


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_input(
    grid_h: int, grid_w: int, templates: np.ndarray, corner: str, specs: list[dict]
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    tr, tc = _template_position(corner, grid_h, grid_w, templates.shape[1])
    grid[tr : tr + TEMPLATE_ROWS, tc : tc + templates.shape[1]] = templates

    for spec in specs:
        bbox = spec["bbox"]
        _draw_frame(grid, bbox, int(spec["frame_color"]))
        base_r, base_c = bbox[0], bbox[1]
        for rect in spec["rects"]:
            rr0, cc0, rr1, cc1 = rect["bbox"]
            grid[
                base_r + rr0 : base_r + rr1 + 1,
                base_c + cc0 : base_c + cc1 + 1,
            ] = int(rect["color"])
    return grid


def _render_expected(inp: np.ndarray, templates: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        pattern = templates[int(spec["depth"])]
        for idx, (r, c) in enumerate(_outer_path(spec["bbox"])):
            out[r, c] = int(pattern[idx % pattern.size])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    depths = [random.randint(0, MAX_DEPTH) for _ in range(n_frames)]
    specs = [_sample_frame_spec(depth) for depth in depths]

    common = 0
    for spec in specs:
        perimeter = _frame_perimeter(spec)
        common = perimeter if common == 0 else math.gcd(common, perimeter)
    lengths = _divisors(common)
    template_len = random.choice(lengths if lengths else [2])
    templates = _sample_templates(template_len)

    total_heights = sum(int(spec["height"]) for spec in specs)
    total_widths = sum(int(spec["width"]) for spec in specs)
    min_side = max(
        GRID_SIDE_RANGE[0],
        max(int(spec["height"]) for spec in specs) + 2 * FRAME_PAD + 2,
        max(int(spec["width"]) for spec in specs) + 2 * FRAME_PAD + 2,
        total_heights // 2 + 6,
        total_widths // 2 + 6,
        TEMPLATE_ROWS + 4,
        template_len + 4,
    )
    if min_side > GRID_SIDE_RANGE[1]:
        return None

    grid_h = random.randint(min_side, GRID_SIDE_RANGE[1])
    grid_w = random.randint(min_side, GRID_SIDE_RANGE[1])
    corner = random.choice(["tl", "tr", "bl", "br"])
    tr, tc = _template_position(corner, grid_h, grid_w, template_len)
    template_rect = (tr, tc, tr + TEMPLATE_ROWS - 1, tc + template_len - 1)

    if not _place_frames(specs, grid_h, grid_w, template_rect):
        return None

    inp = _render_input(grid_h, grid_w, templates, corner, specs)
    expected = _render_expected(inp, templates, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
