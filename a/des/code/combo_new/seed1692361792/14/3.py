import numpy as np


BLACK = 0
GREEN = 3
PALETTE = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _move_delta(direction: str, dist: int) -> tuple[int, int]:
    if direction == "up":
        return -dist, 0
    if direction == "down":
        return dist, 0
    if direction == "left":
        return 0, -dist
    if direction == "right":
        return 0, dist
    raise ValueError(f"bad direction: {direction}")


def _decode_arrow(cells: list[tuple[int, int]]) -> str:
    if len(cells) != 4:
        raise AssertionError("arrow must contain exactly 4 cells")
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if h == 2 and w == 3:
        rows = {}
        for r, c in cells:
            rows.setdefault(r, []).append(c)
        counts = sorted(len(cols) for cols in rows.values())
        if counts != [1, 3]:
            raise AssertionError("bad horizontal arrow")
        line_row = next(r for r, cols in rows.items() if len(cols) == 3)
        extra_row = next(r for r, cols in rows.items() if len(cols) == 1)
        line_cols = sorted(rows[line_row])
        extra_col = rows[extra_row][0]
        if line_cols != list(range(line_cols[0], line_cols[0] + 3)):
            raise AssertionError("arrow line must be contiguous")
        if extra_col not in (line_cols[0], line_cols[-1]):
            raise AssertionError("arrow bend must touch an endpoint")
        return "up" if extra_row < line_row else "down"

    if h == 3 and w == 2:
        cols = {}
        for r, c in cells:
            cols.setdefault(c, []).append(r)
        counts = sorted(len(rows) for rows in cols.values())
        if counts != [1, 3]:
            raise AssertionError("bad vertical arrow")
        line_col = next(c for c, rows in cols.items() if len(rows) == 3)
        extra_col = next(c for c, rows in cols.items() if len(rows) == 1)
        line_rows = sorted(cols[line_col])
        extra_row = cols[extra_col][0]
        if line_rows != list(range(line_rows[0], line_rows[0] + 3)):
            raise AssertionError("arrow line must be contiguous")
        if extra_row not in (line_rows[0], line_rows[-1]):
            raise AssertionError("arrow bend must touch an endpoint")
        return "left" if extra_col < line_col else "right"

    raise AssertionError("unrecognized arrow shape")


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    rects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GREEN)):
        for cells in _components(g == color):
            r0, c0, r1, c1 = _bbox(cells)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if len(cells) != hh * ww:
                raise AssertionError("foreground objects must be solid rectangles")
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch == color):
                raise AssertionError("rectangle patch must be monochrome")
            rects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "h": hh,
                    "w": ww,
                    "area": hh * ww,
                }
            )

    rects.sort(key=lambda rect: (rect["bbox"][0], rect["bbox"][1], rect["color"]))
    rect_map = np.full((h, w), -1, dtype=int)
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        rect_map[r0 : r1 + 1, c0 : c1 + 1] = idx

    arrow_dirs: dict[int, str] = {}
    for cells in _components(g == GREEN):
        direction = _decode_arrow(cells)
        hits = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and rect_map[nr, nc] >= 0:
                    hits.add(int(rect_map[nr, nc]))
        if len(hits) != 1:
            raise AssertionError("each arrow must touch exactly one rectangle")
        idx = hits.pop()
        if idx in arrow_dirs:
            raise AssertionError("each rectangle must have exactly one arrow")
        arrow_dirs[idx] = direction

    if len(arrow_dirs) != len(rects):
        raise AssertionError("every rectangle must have one arrow")

    objects: list[dict] = []
    for idx, rect in enumerate(rects):
        dist = min(rect["h"], rect["w"])
        dy, dx = _move_delta(arrow_dirs[idx], dist)
        r0, c0, r1, c1 = rect["bbox"]
        objects.append(
            {
                "color": rect["color"],
                "area": rect["area"],
                "direction": arrow_dirs[idx],
                "source_bbox": rect["bbox"],
                "target_bbox": (r0 + dy, c0 + dx, r1 + dy, c1 + dx),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each solid colored rectangle in its arrow-indicated direction, with larger moved rectangles covering smaller overlaps.

    Input structure:
    - The background is black.
    - Several non-green colors form solid axis-aligned rectangles of size 2x2 to 5x5.
    - Each rectangle has one adjacent green L-shaped arrow made from a 3-cell line plus one bent endpoint cell.

    Rule:
    - Decode each green L shape: horizontal line plus a cell above means up, horizontal plus below means down, vertical plus left means left, and vertical plus right means right.
    - Pair every arrow with the unique rectangle it touches, then move that rectangle by its own short side length in the decoded direction.
    - Remove all source rectangles and all arrows, and draw only the moved rectangles on a black canvas of the same size.
    - At every output pixel, only the rectangle with strictly largest area remains visible; rectangles fully hidden by larger ones disappear completely.

    Key insight:
    The green marker contributes only geometry: its bend gives the direction, the rectangle's short side gives the distance, and area decides visibility after motion.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)
    owner_area = np.full(g.shape, -1, dtype=int)

    for obj in _extract_objects(g):
        r0, c0, r1, c1 = obj["target_bbox"]
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(g.shape[0] - 1, r1)
        cc1 = min(g.shape[1] - 1, c1)
        if rr0 > rr1 or cc0 > cc1:
            continue
        for r in range(rr0, rr1 + 1):
            for c in range(cc0, cc1 + 1):
                if obj["area"] > owner_area[r, c]:
                    owner_area[r, c] = obj["area"]
                    out[r, c] = obj["color"]

    return out


def _arrow_variants(
    source_r: int,
    source_c: int,
    hh: int,
    ww: int,
    direction: str,
    grid_h: int,
    grid_w: int,
) -> list[tuple[tuple[int, int], ...]]:
    r0, c0 = source_r, source_c
    r1, c1 = source_r + hh - 1, source_c + ww - 1
    variants: set[tuple[tuple[int, int], ...]] = set()

    if direction in ("up", "down"):
        line_r = r0 - 1 if direction == "up" else r1 + 1
        extra_r = line_r - 1 if direction == "up" else line_r + 1
        if not (0 <= line_r < grid_h and 0 <= extra_r < grid_h):
            return []
        for base_c in range(c0 - 2, c1 + 1):
            cols = [base_c, base_c + 1, base_c + 2]
            if cols[0] < 0 or cols[-1] >= grid_w:
                continue
            overlap = sum(c0 <= col <= c1 for col in cols)
            if overlap < 2:
                continue
            for extra_c in (cols[0], cols[-1]):
                cells = tuple(sorted([(line_r, col) for col in cols] + [(extra_r, extra_c)]))
                variants.add(cells)
        return sorted(variants)

    line_c = c0 - 1 if direction == "left" else c1 + 1
    extra_c = line_c - 1 if direction == "left" else line_c + 1
    if not (0 <= line_c < grid_w and 0 <= extra_c < grid_w):
        return []
    for base_r in range(r0 - 2, r1 + 1):
        rows = [base_r, base_r + 1, base_r + 2]
        if rows[0] < 0 or rows[-1] >= grid_h:
            continue
        overlap = sum(r0 <= row <= r1 for row in rows)
        if overlap < 2:
            continue
        for extra_r in (rows[0], rows[-1]):
            cells = tuple(sorted([(row, line_c) for row in rows] + [(extra_r, extra_c)]))
            variants.add(cells)
    return sorted(variants)


def _target_to_source(target_r: int, target_c: int, hh: int, ww: int, direction: str) -> tuple[int, int]:
    dist = min(hh, ww)
    dy, dx = _move_delta(direction, dist)
    return target_r - dy, target_c - dx


def _bbox_relation(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> str:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ir0 = max(ar0, br0)
    ic0 = max(ac0, bc0)
    ir1 = min(ar1, br1)
    ic1 = min(ac1, bc1)
    if ir0 > ir1 or ic0 > ic1:
        return "none"

    inter = (ir1 - ir0 + 1) * (ic1 - ic0 + 1)
    area_a = (ar1 - ar0 + 1) * (ac1 - ac0 + 1)
    area_b = (br1 - br0 + 1) * (bc1 - bc0 + 1)
    if inter == area_a == area_b:
        return "equal"
    if inter == area_a:
        return "a_in_b"
    if inter == area_b:
        return "b_in_a"
    return "partial"


def _rect_cells(r0: int, c0: int, hh: int, ww: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)}


def _mark_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                blocked[nr, nc] = True


def _try_place_from_target(
    desc: dict,
    target_r: int,
    target_c: int,
    grid_h: int,
    grid_w: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    hh = desc["h"]
    ww = desc["w"]
    source_r, source_c = _target_to_source(target_r, target_c, hh, ww, desc["direction"])
    if source_r < 0 or source_c < 0 or source_r + hh > grid_h or source_c + ww > grid_w:
        return None

    rect_cells = _rect_cells(source_r, source_c, hh, ww)
    if any(blocked[r, c] for r, c in rect_cells):
        return None

    arrow_options = []
    for arrow_cells_tuple in _arrow_variants(source_r, source_c, hh, ww, desc["direction"], grid_h, grid_w):
        arrow_cells = set(arrow_cells_tuple)
        if rect_cells & arrow_cells:
            continue
        if any(blocked[r, c] for r, c in arrow_cells):
            continue
        arrow_options.append(arrow_cells)
    if not arrow_options:
        return None

    arrow_cells = arrow_options[int(rng.integers(0, len(arrow_options)))]
    target_bbox = (target_r, target_c, target_r + hh - 1, target_c + ww - 1)
    return {
        "color": desc["color"],
        "h": hh,
        "w": ww,
        "area": hh * ww,
        "direction": desc["direction"],
        "source_bbox": (source_r, source_c, source_r + hh - 1, source_c + ww - 1),
        "target_bbox": target_bbox,
        "rect_cells": rect_cells,
        "arrow_cells": arrow_cells,
    }


def _random_desc(color: int, rng: np.random.Generator) -> dict:
    hh = int(rng.integers(2, 6))
    ww = int(rng.integers(2, 6))
    direction = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
    return {"color": int(color), "h": hh, "w": ww, "direction": direction}


def _random_full_desc_pair(colors: np.ndarray, rng: np.random.Generator) -> list[dict]:
    large_h = int(rng.integers(3, 6))
    large_w = int(rng.integers(3, 6))
    while True:
        small_h = int(rng.integers(2, min(large_h, 5) + 1))
        small_w = int(rng.integers(2, min(large_w, 5) + 1))
        if small_h <= large_h and small_w <= large_w and small_h * small_w < large_h * large_w:
            break
    return [
        {"color": int(colors[0]), "h": large_h, "w": large_w, "direction": DIRECTIONS[int(rng.integers(0, 4))]},
        {"color": int(colors[1]), "h": small_h, "w": small_w, "direction": DIRECTIONS[int(rng.integers(0, 4))]},
    ]


def _place_random_nonoverlap(
    desc: dict,
    grid_h: int,
    grid_w: int,
    blocked: np.ndarray,
    target_bboxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> dict | None:
    for _ in range(250):
        target_r = int(rng.integers(0, grid_h - desc["h"] + 1))
        target_c = int(rng.integers(0, grid_w - desc["w"] + 1))
        target_bbox = (target_r, target_c, target_r + desc["h"] - 1, target_c + desc["w"] - 1)
        if any(_bbox_relation(target_bbox, other) != "none" for other in target_bboxes):
            continue
        placed = _try_place_from_target(desc, target_r, target_c, grid_h, grid_w, blocked, rng)
        if placed is not None:
            return placed
    return None


def _place_partial_overlap(
    desc: dict,
    anchor_bbox: tuple[int, int, int, int],
    grid_h: int,
    grid_w: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    ar0, ac0, ar1, ac1 = anchor_bbox
    row_lo = max(0, ar0 - desc["h"] + 1)
    row_hi = min(grid_h - desc["h"], ar1)
    col_lo = max(0, ac0 - desc["w"] + 1)
    col_hi = min(grid_w - desc["w"], ac1)
    if row_lo > row_hi or col_lo > col_hi:
        return None
    for _ in range(400):
        target_r = int(rng.integers(row_lo, row_hi + 1))
        target_c = int(rng.integers(col_lo, col_hi + 1))
        target_bbox = (target_r, target_c, target_r + desc["h"] - 1, target_c + desc["w"] - 1)
        if _bbox_relation(target_bbox, anchor_bbox) != "partial":
            continue
        placed = _try_place_from_target(desc, target_r, target_c, grid_h, grid_w, blocked, rng)
        if placed is not None:
            return placed
    return None


def _place_full_overlap(
    small_desc: dict,
    large_bbox: tuple[int, int, int, int],
    grid_h: int,
    grid_w: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    lr0, lc0, lr1, lc1 = large_bbox
    row_lo = lr0
    row_hi = lr1 - small_desc["h"] + 1
    col_lo = lc0
    col_hi = lc1 - small_desc["w"] + 1
    if row_lo > row_hi or col_lo > col_hi:
        return None
    for _ in range(300):
        target_r = int(rng.integers(row_lo, row_hi + 1))
        target_c = int(rng.integers(col_lo, col_hi + 1))
        placed = _try_place_from_target(small_desc, target_r, target_c, grid_h, grid_w, blocked, rng)
        if placed is None:
            continue
        if _bbox_relation(placed["target_bbox"], large_bbox) != "a_in_b":
            continue
        return placed
    return None


def _collect_relations(target_bboxes: list[tuple[int, int, int, int]]) -> list[str]:
    rels = []
    for i in range(len(target_bboxes)):
        for j in range(i + 1, len(target_bboxes)):
            rels.append(_bbox_relation(target_bboxes[i], target_bboxes[j]))
    return rels


def _build_grid(grid_h: int, grid_w: int, placed: list[dict]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for obj in placed:
        for r, c in obj["rect_cells"]:
            grid[r, c] = obj["color"]
        for r, c in obj["arrow_cells"]:
            grid[r, c] = GREEN
    return grid


def _validate_generated(grid: np.ndarray, style: str, expected_count: int) -> bool:
    try:
        objs = _extract_objects(grid)
        out = main(grid)
    except AssertionError:
        return False

    if len(objs) != expected_count:
        return False
    if grid.shape != out.shape:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    rels = _collect_relations([obj["target_bbox"] for obj in objs])
    if any(rel == "equal" for rel in rels):
        return False

    if style == "none":
        if any(rel != "none" for rel in rels):
            return False
    elif style == "partial":
        if rels.count("partial") != 1:
            return False
        if any(rel not in ("none", "partial") for rel in rels):
            return False
    else:
        full_count = sum(rel in ("a_in_b", "b_in_a") for rel in rels)
        if full_count != 1:
            return False
        if any(rel not in ("none", "a_in_b", "b_in_a") for rel in rels):
            return False
        input_colors = {int(v) for v in np.unique(grid)} - {BLACK, GREEN}
        output_colors = {int(v) for v in np.unique(out)} - {BLACK}
        if len(output_colors) != len(input_colors) - 1:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(_RNG.integers(18, 29))
    grid_w = int(_RNG.integers(18, 29))
    style = ("none", "partial", "full")[int(_RNG.integers(0, 3))]
    count = int(_RNG.integers(2, 5 if style == "none" else 4))
    colors = _RNG.choice(PALETTE, size=count, replace=False)

    if style == "full":
        descs = _random_full_desc_pair(colors[:2], _RNG)
        while len(descs) < count:
            descs.append(_random_desc(int(colors[len(descs)]), _RNG))
    else:
        descs = [_random_desc(int(color), _RNG) for color in colors]
        if style == "partial":
            while descs[0]["h"] * descs[0]["w"] == descs[1]["h"] * descs[1]["w"]:
                descs[1] = _random_desc(int(colors[1]), _RNG)

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict] = []
    target_bboxes: list[tuple[int, int, int, int]] = []

    first = _place_random_nonoverlap(descs[0], grid_h, grid_w, blocked, target_bboxes, _RNG)
    if first is None:
        return None
    placed.append(first)
    target_bboxes.append(first["target_bbox"])
    _mark_blocked(blocked, first["rect_cells"] | first["arrow_cells"])

    if style == "none":
        second = _place_random_nonoverlap(descs[1], grid_h, grid_w, blocked, target_bboxes, _RNG)
    elif style == "partial":
        second = _place_partial_overlap(descs[1], first["target_bbox"], grid_h, grid_w, blocked, _RNG)
    else:
        second = _place_full_overlap(descs[1], first["target_bbox"], grid_h, grid_w, blocked, _RNG)
    if second is None:
        return None

    placed.append(second)
    target_bboxes.append(second["target_bbox"])
    _mark_blocked(blocked, second["rect_cells"] | second["arrow_cells"])

    for desc in descs[2:]:
        extra = _place_random_nonoverlap(desc, grid_h, grid_w, blocked, target_bboxes, _RNG)
        if extra is None:
            return None
        placed.append(extra)
        target_bboxes.append(extra["target_bbox"])
        _mark_blocked(blocked, extra["rect_cells"] | extra["arrow_cells"])

    grid = _build_grid(grid_h, grid_w, placed)
    if not _validate_generated(grid, style, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
