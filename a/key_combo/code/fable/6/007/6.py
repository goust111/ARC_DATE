import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 1
BODY_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
SYMBOL_STEP = 4


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _find_mono_components(
    grid: np.ndarray,
    bg: int,
    *,
    allowed_mask: np.ndarray | None = None,
    connectivity: int = 4,
) -> list[dict]:
    comps = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        mask = grid == color
        if allowed_mask is not None:
            mask &= allowed_mask
        for coords in _components(mask, connectivity=connectivity):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if comp["area"] != 2 * (h + w) - 4:
        return False
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return set(comp["coords"]) == expected


def _is_hollow_symbol(comp: dict, grid: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if comp["area"] != 8 or (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.full((3, 3), color, dtype=int)
    expected[1, 1] = int(bg)
    return np.array_equal(patch, expected)


def _extract_training_pair(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    comps = _find_mono_components(grid, bg, connectivity=4)
    frames = [comp for comp in comps if _is_frame_component(comp)]
    assert len(frames) == 2, "expected exactly two framed training panels"
    frames.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    boxes = []
    inners = []
    for comp in frames:
        r0, c0, r1, c1 = comp["bbox"]
        boxes.append((r0, c0, r1, c1))
        inners.append(grid[r0 + 1 : r1, c0 + 1 : c1].copy())
    return inners[0], inners[1], boxes


def _compress_body(grid: np.ndarray, panel_boxes: list[tuple[int, int, int, int]], bg: int) -> np.ndarray:
    allowed = np.ones(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in panel_boxes:
        allowed[r0 : r1 + 1, c0 : c1 + 1] = False

    comps = _find_mono_components(grid, bg, allowed_mask=allowed, connectivity=4)
    assert comps, "expected hollow symbols outside the training panels"

    symbols = []
    for comp in comps:
        assert _is_hollow_symbol(comp, grid, bg), "non-panel foreground must be hollow 3x3 symbols"
        r0, c0, _r1, _c1 = comp["bbox"]
        symbols.append((r0, c0, int(comp["color"])))

    rows = sorted({r for r, _c, _color in symbols})
    cols = sorted({c for _r, c, _color in symbols})
    assert rows and cols
    if len(rows) > 1:
        assert len({b - a for a, b in zip(rows, rows[1:])}) == 1
    if len(cols) > 1:
        assert len({b - a for a, b in zip(cols, cols[1:])}) == 1

    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(cols)}
    coarse = np.full((len(rows), len(cols)), int(bg), dtype=int)
    for r, c, color in symbols:
        rr = row_index[r]
        cc = col_index[c]
        assert coarse[rr, cc] == bg
        coarse[rr, cc] = int(color)

    assert np.all(np.any(coarse != bg, axis=1)), "every coarse row must be represented"
    assert np.all(np.any(coarse != bg, axis=0)), "every coarse column must be represented"
    return coarse


GEOM_TRANSFORMS = {
    "flip_h": lambda x: np.fliplr(x),
    "flip_v": lambda x: np.flipud(x),
    "rot90": lambda x: np.rot90(x, 1),
    "rot180": lambda x: np.rot90(x, 2),
    "rot270": lambda x: np.rot90(x, 3),
    "transpose": lambda x: x.T,
}


def _infer_recolor(src: np.ndarray, dst: np.ndarray, bg: int) -> dict[int, int] | None:
    if src.shape != dst.shape:
        return None
    if not np.array_equal(src == bg, dst == bg):
        return None
    src_colors = sorted(int(v) for v in np.unique(src) if int(v) != int(bg))
    if len(src_colors) < 1:
        return None
    mapping: dict[int, int] = {}
    for color in src_colors:
        mapped = sorted(int(v) for v in np.unique(dst[src == color]))
        if len(mapped) != 1 or mapped[0] == bg:
            return None
        mapping[int(color)] = int(mapped[0])
    if len(set(mapping.values())) != len(mapping):
        return None
    if all(int(k) == int(v) for k, v in mapping.items()):
        return None
    return mapping


def _transform_signature(spec: tuple[str, object]) -> tuple:
    kind, payload = spec
    if kind == "geom":
        return ("geom", str(payload))
    assert kind == "recolor"
    return ("recolor", tuple(sorted((int(k), int(v)) for k, v in payload.items())))


def _candidate_specs(src: np.ndarray, dst: np.ndarray, bg: int) -> list[tuple[str, object]]:
    matches = []
    for name, fn in GEOM_TRANSFORMS.items():
        if np.array_equal(np.asarray(fn(src), dtype=int), dst):
            matches.append(("geom", name))
    mapping = _infer_recolor(src, dst, bg)
    if mapping is not None:
        matches.append(("recolor", mapping))
    return matches


def _infer_transform(src: np.ndarray, dst: np.ndarray, bg: int) -> tuple[str, object]:
    matches = _candidate_specs(src, dst, bg)
    assert len(matches) == 1, "training pair must determine a unique transform"
    return matches[0]


def _apply_transform(spec: tuple[str, object], grid: np.ndarray) -> np.ndarray:
    kind, payload = spec
    if kind == "geom":
        return np.asarray(GEOM_TRANSFORMS[str(payload)](grid), dtype=int).copy()
    assert kind == "recolor"
    mapping = payload
    out = np.array(grid, dtype=int, copy=True)
    src = np.asarray(grid, dtype=int)
    for src_color, dst_color in mapping.items():
        out[src == int(src_color)] = int(dst_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the large hollow-symbol scene into a coarse grid, then imitate the corner training pair's transformation on it.

    Input structure:
    - The main scene is a large picture made of monochrome hollow 3x3 symbols placed on a regular coarse lattice, with some coarse cells empty.
    - One corner contains two small framed panels placed side by side; the left panel is a source pattern and the right panel is its transformed result.
    - The panel interiors are unrelated in size to the coarse grid recovered from the large symbol picture.

    Rule:
    - Ignore the frames and compress each hollow 3x3 symbol in the large scene into one coarse-grid pixel of the same color, leaving empty coarse cells as background.
    - Compare the two training panels and infer the unique demonstrated transformation, which may be geometric or a direct color relabeling.
    - Apply that inferred transformation to the compressed coarse grid and output only the transformed coarse grid.

    Key insight:
    - The puzzle uses two linked representation levels: first read the big picture as symbolic coarse pixels, then solve the analogy shown by the framed training pair and transfer that rule to the compressed picture.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    src, dst, panel_boxes = _extract_training_pair(g, bg)
    coarse = _compress_body(g, panel_boxes, bg)
    transform = _infer_transform(src, dst, bg)
    return _apply_transform(transform, coarse)


def _random_pattern(shape: tuple[int, int], colors: list[int], min_non_bg: int) -> np.ndarray:
    h, w = shape
    total = h * w
    count = random.randint(min_non_bg, total)
    cells = random.sample([(r, c) for r in range(h) for c in range(w)], count)
    arr = np.full((h, w), BG, dtype=int)
    used = colors[:]
    random.shuffle(used)
    for (r, c), color in zip(cells[: len(used)], used):
        arr[r, c] = int(color)
    for r, c in cells[len(used) :]:
        arr[r, c] = int(random.choice(colors))
    return arr


def _random_derangement(colors: list[int]) -> dict[int, int]:
    target = colors[:]
    while True:
        random.shuffle(target)
        if all(a != b for a, b in zip(colors, target)):
            return {int(a): int(b) for a, b in zip(colors, target)}


def _make_training_pair(transform_name: str, body_palette: list[int]) -> tuple[np.ndarray, np.ndarray, tuple[str, object]]:
    for _ in range(400):
        if transform_name in {"rot90", "rot270", "transpose"}:
            h = random.randint(3, 5)
            w = random.randint(3, 5)
            if h == w:
                continue
        else:
            h = random.randint(3, 5)
            w = random.randint(3, 5)

        if transform_name == "recolor":
            n_colors = random.randint(2, min(4, len(body_palette)))
            colors = random.sample(body_palette, n_colors)
            src = _random_pattern((h, w), colors, min_non_bg=max(n_colors + 1, 4))
            mapping = _random_derangement(colors)
            dst = _apply_transform(("recolor", mapping), src)
            spec = ("recolor", mapping)
        else:
            n_colors = random.randint(2, 4)
            colors = random.sample(BODY_COLORS, n_colors)
            src = _random_pattern((h, w), colors, min_non_bg=max(n_colors + 1, 4))
            spec = ("geom", transform_name)
            dst = _apply_transform(spec, src)

        if np.array_equal(src, dst):
            continue
        cands = _candidate_specs(src, dst, BG)
        if len(cands) != 1:
            continue
        if _transform_signature(cands[0]) != _transform_signature(spec):
            continue
        return src, dst, spec
    raise RuntimeError("failed to generate an unambiguous training pair")


def _covered_mask(h: int, w: int, count: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(h):
        cells.add((r, random.randrange(w)))
    for c in range(w):
        if not any(cc == c for _rr, cc in cells):
            cells.add((random.randrange(h), c))
    remaining = [(r, c) for r in range(h) for c in range(w) if (r, c) not in cells]
    random.shuffle(remaining)
    while len(cells) < count and remaining:
        cells.add(remaining.pop())
    return cells


def _make_body_pattern(spec: tuple[str, object]) -> np.ndarray:
    for _ in range(400):
        h = random.randint(4, 5)
        w = random.randint(4, 6)
        min_count = max(h, w) + 1
        max_count = h * w - 2
        occupied_count = random.randint(min_count, max_count)
        cells = list(_covered_mask(h, w, occupied_count))
        if len(cells) != occupied_count:
            continue

        coarse = np.full((h, w), BG, dtype=int)
        if spec[0] == "recolor":
            moved_colors = [c for c, d in spec[1].items() if int(c) != int(d)]
            required = random.sample(moved_colors, random.randint(2, len(moved_colors)))
            palette = sorted(set(required + random.sample(BODY_COLORS, random.randint(0, 2))))
        else:
            palette = random.sample(BODY_COLORS, random.randint(2, 5))
            required = []

        random.shuffle(cells)
        idx = 0
        for color in required:
            coarse[cells[idx]] = int(color)
            idx += 1
        for r, c in cells[idx:]:
            coarse[r, c] = int(random.choice(palette))

        out = _apply_transform(spec, coarse)
        if np.array_equal(out, coarse):
            continue
        cands = _candidate_specs(coarse, out, BG)
        if len(cands) != 1:
            continue
        if _transform_signature(cands[0]) != _transform_signature(spec):
            continue
        return coarse
    raise RuntimeError("failed to generate a non-degenerate coarse body")


def _render_body(coarse: np.ndarray) -> np.ndarray:
    h, w = coarse.shape
    out = np.full((h * SYMBOL_STEP - 1, w * SYMBOL_STEP - 1), BG, dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(coarse[r, c])
            if color == BG:
                continue
            rr = r * SYMBOL_STEP
            cc = c * SYMBOL_STEP
            out[rr : rr + 3, cc : cc + 3] = color
            out[rr + 1, cc + 1] = BG
    return out


def _render_panel(inner: np.ndarray) -> np.ndarray:
    h, w = inner.shape
    out = np.full((h + 2, w + 2), BG, dtype=int)
    out[0, :] = FRAME_COLOR
    out[-1, :] = FRAME_COLOR
    out[:, 0] = FRAME_COLOR
    out[:, -1] = FRAME_COLOR
    out[1:-1, 1:-1] = inner
    return out


def _compose_input(coarse: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    body = _render_body(coarse)
    left = _render_panel(src)
    right = _render_panel(dst)

    panel_h = max(left.shape[0], right.shape[0])
    panel_w = left.shape[1] + 1 + right.shape[1]
    min_h = panel_h + 1 + body.shape[0]
    min_w = max(panel_w, body.shape[1])
    assert min_h <= 30 and min_w <= 30

    H = min_h + random.randint(0, 30 - min_h)
    W = min_w + random.randint(0, 30 - min_w)
    grid = np.full((H, W), BG, dtype=int)

    corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    panel_left = 0 if "left" in corner else W - panel_w
    left_col = panel_left
    right_col = panel_left + left.shape[1] + 1

    if "top" in corner:
        left_row = 0
        right_row = 0
        body_row = random.randint(panel_h + 1, H - body.shape[0])
    else:
        left_row = H - left.shape[0]
        right_row = H - right.shape[0]
        body_row = random.randint(0, H - panel_h - 1 - body.shape[0])

    body_col = random.randint(0, W - body.shape[1])

    grid[left_row : left_row + left.shape[0], left_col : left_col + left.shape[1]] = left
    grid[right_row : right_row + right.shape[0], right_col : right_col + right.shape[1]] = right
    grid[body_row : body_row + body.shape[0], body_col : body_col + body.shape[1]] = body
    return grid


def _validate_case(grid: np.ndarray, coarse: np.ndarray, src: np.ndarray, dst: np.ndarray, spec: tuple[str, object]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        recovered_src, recovered_dst, boxes = _extract_training_pair(grid, BG)
        recovered_coarse = _compress_body(grid, boxes, BG)
        inferred = _infer_transform(recovered_src, recovered_dst, BG)
        expected = _apply_transform(spec, coarse)
    except AssertionError:
        return False

    if not np.array_equal(recovered_src, src):
        return False
    if not np.array_equal(recovered_dst, dst):
        return False
    if not np.array_equal(recovered_coarse, coarse):
        return False
    if _transform_signature(inferred) != _transform_signature(spec):
        return False
    if np.array_equal(expected, coarse):
        return False
    if len(_candidate_specs(coarse, expected, BG)) != 1:
        return False
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def generate_input() -> np.ndarray:
    for _ in range(600):
        transform_name = random.choice(list(GEOM_TRANSFORMS) + ["recolor"])
        body_palette = random.sample(BODY_COLORS, random.randint(3, 5))
        src, dst, spec = _make_training_pair(transform_name, body_palette)
        coarse = _make_body_pattern(spec)
        grid = _compose_input(coarse, src, dst)
        if _validate_case(grid, coarse, src, dst, spec):
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
