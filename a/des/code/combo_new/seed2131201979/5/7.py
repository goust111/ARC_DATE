import numpy as np
import random


BG = 0
ARC_COLORS = tuple(range(1, 10))
PALETTE_LEN = 5
MAX_GRID = 30
GENERATION_ATTEMPTS = 300


def _run_colors(row: np.ndarray) -> list[int]:
    colors: list[int] = []
    last = None
    for value in row:
        color = int(value)
        if color != last:
            colors.append(color)
            last = color
    return colors


def _box_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _one_gap_neighbor(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    row_overlap = max(ar0, br0) <= min(ar1, br1)
    col_overlap = max(ac0, bc0) <= min(ac1, bc1)

    if row_overlap and (bc0 - ac1 == 2 or ac0 - bc1 == 2):
        return True
    if col_overlap and (br0 - ar1 == 2 or ar0 - br1 == 2):
        return True
    return False


def _find_band_row(grid: np.ndarray) -> tuple[int, list[int]]:
    candidates: list[tuple[int, list[int]]] = []
    for r in (0, grid.shape[0] - 1):
        runs = _run_colors(grid[r])
        if len(runs) >= PALETTE_LEN and all(color != BG for color in runs):
            candidates.append((int(r), runs))
    assert len(candidates) == 1, "expected exactly one palette band on the top or bottom edge"
    return candidates[0]


def _detect_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    band_r, _ = _find_band_row(grid)
    body = np.array(grid, copy=True)
    body[band_r, :] = BG

    records: list[dict] = []
    for color in sorted({int(v) for v in body.ravel()} - {BG}):
        coords = np.argwhere(body == color)
        assert coords.size > 0, "every object color must still appear outside the palette row"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        mask = body[r0 : r1 + 1, c0 : c1 + 1] == color

        # Every object still exposes the full border of its rectangle, even when
        # smaller inner rectangles overwrite part of its interior.
        assert mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()
        records.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "area": _box_area((r0, c0, r1, c1)),
            }
        )

    for record in records:
        r0, c0, r1, c1 = record["bbox"]
        sub = body[r0 : r1 + 1, c0 : c1 + 1]
        holes = np.argwhere((sub != BG) & (sub != record["color"]))
        for dr, dc in holes:
            rr = r0 + int(dr)
            cc = c0 + int(dc)
            assert any(
                other is not record
                and _strict_contains(record["bbox"], other["bbox"])
                and other["bbox"][0] <= rr <= other["bbox"][2]
                and other["bbox"][1] <= cc <= other["bbox"][3]
                for other in records
            ), "non-owner pixels inside a rectangle must belong to strictly contained rectangles"

    return band_r, records


def _has_containment(records: list[dict]) -> bool:
    for i, record in enumerate(records):
        for j, other in enumerate(records):
            if i != j and _strict_contains(record["bbox"], other["bbox"]):
                return True
    return False


def _neighbor_sets(records: list[dict]) -> list[set[int]]:
    boxes = [record["bbox"] for record in records]
    neighbors = [set() for _ in records]
    for i, box_i in enumerate(boxes):
        for j, box_j in enumerate(boxes):
            if i == j:
                continue
            if _one_gap_neighbor(box_i, box_j):
                neighbors[i].add(j)
            elif _strict_contains(box_j, box_i):
                neighbors[i].add(j)
    return neighbors


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each rectangle from an edge palette whose reading direction is chosen by whether any rectangle contains another.

    Input structure:
    - The background is black.
    - One full top or bottom row is a palette band made of at least five contiguous color runs.
    - Away from that band are uniquely colored axis-aligned rectangles. Some smaller rectangles may sit strictly inside larger differently colored rectangles, so the outer rectangles can appear as rectangular rings after the inner ones overwrite part of their interiors.

    Rule:
    - Recover every rectangle by its bounding box outside the palette row.
    - If any rectangle strictly contains another rectangle, read the palette band left-to-right; otherwise read it right-to-left.
    - For each rectangle, count neighbors: every other rectangle separated from it by exactly one background row or column and overlapping along the perpendicular axis counts once; additionally, an inner rectangle also counts each rectangle that strictly contains it.
    - Repaint each rectangle with the (N+1)-th distinct palette color in the chosen reading order, while leaving the palette band unchanged. Paint larger rectangles before smaller contained ones so nested rectangles stay visible.

    Key insight:
    - Two different spatial relations matter at once: containment decides which end of the palette is index 0, while one-cell edge gaps decide the lookup index itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= MAX_GRID and 1 <= g.shape[1] <= MAX_GRID

    band_r, records = _detect_objects(g)
    assert records, "expected at least one rectangle object"

    palette = _run_colors(g[band_r])
    assert len(palette) >= PALETTE_LEN, "palette band must expose at least five distinct colors"
    if not _has_containment(records):
        palette = list(reversed(palette))

    neighbors = _neighbor_sets(records)
    out = np.array(g, copy=True)
    for idx, record in sorted(enumerate(records), key=lambda item: (-item[1]["area"], item[0])):
        count = len(neighbors[idx])
        assert count < len(palette), "neighbor count must fit inside the palette band"
        _paint_rect(out, record["bbox"], palette[count])

    out[band_r, :] = g[band_r, :]
    return out


# --------------- generator ---------------

MIN_BODY_HEIGHT = 5
MIN_GRID_SIZE = 12
PADDING_LIMIT = 8


def _positive_composition(total: int, parts: int) -> list[int]:
    assert total >= parts
    extras = np.random.multinomial(total - parts, [1.0 / parts] * parts)
    return [1 + int(v) for v in extras]


def _shift_objects(objects: list[dict], dr: int, dc: int) -> list[dict]:
    shifted: list[dict] = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        shifted.append(
            {
                "bbox": (r0 + dr, c0 + dc, r1 + dr, c1 + dc),
                "layer": int(obj["layer"]),
            }
        )
    return shifted


def _normalize_objects(objects: list[dict]) -> list[dict]:
    min_r = min(obj["bbox"][0] for obj in objects)
    min_c = min(obj["bbox"][1] for obj in objects)
    return _shift_objects(objects, -min_r, -min_c)


def _objects_span(objects: list[dict]) -> tuple[int, int]:
    max_r = max(obj["bbox"][2] for obj in objects)
    max_c = max(obj["bbox"][3] for obj in objects)
    return max_r + 1, max_c + 1


def _validate_geometry(objects: list[dict], expect_containment: bool) -> bool:
    boxes = [obj["bbox"] for obj in objects]
    containments = 0
    for i, box_i in enumerate(boxes):
        for j, box_j in enumerate(boxes):
            if i >= j:
                continue
            if _strict_contains(box_i, box_j) or _strict_contains(box_j, box_i):
                containments += 1
                continue
            if _boxes_overlap(box_i, box_j):
                return False

    if expect_containment != (containments > 0):
        return False

    records = [{"bbox": obj["bbox"]} for obj in objects]
    if max(len(nbs) for nbs in _neighbor_sets(records)) >= PALETTE_LEN:
        return False

    body_h, body_w = _objects_span(objects)
    return body_h <= MAX_GRID - 3 and body_w <= MAX_GRID


def _make_horizontal_chain() -> list[dict]:
    n = random.randint(3, 4)
    heights = [random.randint(2, 4) for _ in range(n)]
    widths = [random.randint(2, 4) for _ in range(n)]
    gap_patterns = {
        3: ((2, 2), (2, 4), (4, 2), (2, 5), (5, 2)),
        4: ((2, 2, 2), (2, 4, 2), (4, 2, 2), (2, 2, 4), (2, 5, 2), (5, 2, 4)),
    }
    gaps = random.choice(gap_patterns[n])

    objects: list[dict] = []
    c = 0
    for i, (h, w) in enumerate(zip(heights, widths)):
        objects.append({"bbox": (0, c, h - 1, c + w - 1), "layer": 0})
        if i + 1 < n:
            c += w + gaps[i]
    return objects


def _make_vertical_chain() -> list[dict]:
    n = random.randint(3, 4)
    heights = [random.randint(2, 4) for _ in range(n)]
    widths = [random.randint(2, 4) for _ in range(n)]
    gap_patterns = {
        3: ((2, 2), (2, 4), (4, 2), (2, 5), (5, 2)),
        4: ((2, 2, 2), (2, 4, 2), (4, 2, 2), (2, 2, 4), (2, 5, 2), (5, 2, 4)),
    }
    gaps = random.choice(gap_patterns[n])

    objects: list[dict] = []
    r = 0
    for i, (h, w) in enumerate(zip(heights, widths)):
        objects.append({"bbox": (r, 0, r + h - 1, w - 1), "layer": 0})
        if i + 1 < n:
            r += h + gaps[i]
    return objects


def _make_star() -> list[dict]:
    center_h = random.randint(2, 4)
    center_w = random.randint(2, 4)
    center = {"bbox": (0, 0, center_h - 1, center_w - 1), "layer": 0}

    leaf_count = random.choice((2, 3))
    sides = random.sample(("left", "right", "top", "bottom"), leaf_count)
    objects = [center]
    for side in sides:
        if side in ("left", "right"):
            h = random.randint(2, center_h)
            w = random.randint(2, 4)
            r0 = random.randint(0, center_h - h)
            if side == "left":
                c1 = -2
                c0 = c1 - w + 1
            else:
                c0 = center_w + 1
                c1 = c0 + w - 1
            objects.append({"bbox": (r0, c0, r0 + h - 1, c1), "layer": 0})
        else:
            h = random.randint(2, 4)
            w = random.randint(2, center_w)
            c0 = random.randint(0, center_w - w)
            if side == "top":
                r1 = -2
                r0 = r1 - h + 1
            else:
                r0 = center_h + 1
                r1 = r0 + h - 1
            objects.append({"bbox": (r0, c0, r1, c0 + w - 1), "layer": 0})
    return _normalize_objects(objects)


def _random_inner_bbox(outer_h: int, outer_w: int) -> tuple[int, int, int, int]:
    inner_h = random.randint(2, outer_h - 2)
    inner_w = random.randint(2, outer_w - 2)
    top = random.randint(1, outer_h - inner_h - 1)
    left = random.randint(1, outer_w - inner_w - 1)
    return top, left, top + inner_h - 1, left + inner_w - 1


def _corner_child_bbox(outer_h: int, outer_w: int, corner: str) -> tuple[int, int, int, int]:
    child_h = random.randint(2, min(3, outer_h - 2))
    child_w = random.randint(2, min(3, outer_w - 2))
    if corner == "tl":
        r0 = random.randint(1, 2)
        c0 = random.randint(1, 2)
    elif corner == "tr":
        r0 = random.randint(1, 2)
        c0 = outer_w - child_w - random.randint(1, 2)
    elif corner == "bl":
        r0 = outer_h - child_h - random.randint(1, 2)
        c0 = random.randint(1, 2)
    else:
        assert corner == "br"
        r0 = outer_h - child_h - random.randint(1, 2)
        c0 = outer_w - child_w - random.randint(1, 2)
    return r0, c0, r0 + child_h - 1, c0 + child_w - 1


def _external_bbox(side: str, outer_h: int, outer_w: int) -> tuple[int, int, int, int]:
    if side in ("left", "right"):
        h = random.randint(2, max(2, outer_h - 1))
        w = random.randint(2, 4)
        r0 = random.randint(0, outer_h - h)
        if side == "left":
            c1 = -2
            c0 = c1 - w + 1
        else:
            c0 = outer_w + 1
            c1 = c0 + w - 1
        return r0, c0, r0 + h - 1, c1

    h = random.randint(2, 4)
    w = random.randint(2, max(2, outer_w - 1))
    c0 = random.randint(0, outer_w - w)
    if side == "top":
        r1 = -2
        r0 = r1 - h + 1
    else:
        assert side == "bottom"
        r0 = outer_h + 1
        r1 = r0 + h - 1
    return r0, c0, r1, c0 + w - 1


def _make_containment_scene() -> list[dict]:
    outer_h = random.randint(7, 10)
    outer_w = random.randint(7, 10)
    objects: list[dict] = [{"bbox": (0, 0, outer_h - 1, outer_w - 1), "layer": 0}]

    child_count = random.choice((1, 2))
    if child_count == 1:
        objects.append({"bbox": _random_inner_bbox(outer_h, outer_w), "layer": 1})
        side_count = random.choice((1, 2))
        sides = random.sample(("left", "right", "top", "bottom"), side_count)
    else:
        corners = random.choice((("tl", "br"), ("tr", "bl")))
        objects.extend({"bbox": _corner_child_bbox(outer_h, outer_w, corner), "layer": 1} for corner in corners)
        sides = [random.choice(("left", "right", "top", "bottom"))]

    for side in sides:
        objects.append({"bbox": _external_bbox(side, outer_h, outer_w), "layer": 0})

    return _normalize_objects(objects)


def _make_band_row(width: int, colors: list[int]) -> np.ndarray:
    run_widths = _positive_composition(width, len(colors))
    row = np.empty(width, dtype=int)
    c = 0
    for color, run_w in zip(colors, run_widths):
        row[c : c + run_w] = int(color)
        c += run_w
    return row


def _build_scene() -> tuple[list[dict], bool] | None:
    has_containment = random.random() < 0.5
    if has_containment:
        objects = _make_containment_scene()
    else:
        objects = random.choice((_make_horizontal_chain, _make_vertical_chain, _make_star))()

    if not _validate_geometry(objects, has_containment):
        return None
    return objects, has_containment


def _paint_input(
    objects: list[dict],
    band_top: bool,
    band_colors: list[int],
    object_colors: list[int],
) -> tuple[np.ndarray, list[dict], int]:
    body_h, body_w = _objects_span(objects)
    min_h = max(MIN_GRID_SIZE, body_h + 3)
    min_w = max(MIN_GRID_SIZE, body_w)
    if min_h > MAX_GRID or min_w > MAX_GRID:
        raise ValueError("scene does not fit inside ARC bounds")

    H = min_h + random.randint(0, min(PADDING_LIMIT, MAX_GRID - min_h))
    W = min_w + random.randint(0, min(PADDING_LIMIT, MAX_GRID - min_w))

    vertical_budget = H - 3 - body_h
    horizontal_budget = W - body_w
    if band_top:
        base_r = 2 + random.randint(0, vertical_budget)
        band_r = 0
    else:
        base_r = random.randint(0, vertical_budget)
        band_r = H - 1
    base_c = random.randint(0, horizontal_budget)

    placed = _shift_objects(objects, base_r, base_c)
    grid = np.full((H, W), BG, dtype=int)
    grid[band_r, :] = _make_band_row(W, band_colors)

    for obj, color in zip(placed, object_colors):
        obj["color"] = int(color)
    for obj in sorted(placed, key=lambda item: (item["layer"], -_box_area(item["bbox"]))):
        _paint_rect(grid, obj["bbox"], obj["color"])

    return grid, placed, band_r


def _expected_output(inp: np.ndarray, objects: list[dict], band_r: int, has_containment: bool) -> np.ndarray:
    out = np.array(inp, copy=True)
    palette = _run_colors(inp[band_r])
    if not has_containment:
        palette = list(reversed(palette))

    neighbors = _neighbor_sets(objects)
    for idx, obj in sorted(enumerate(objects), key=lambda item: (-_box_area(item[1]["bbox"]), item[0])):
        _paint_rect(out, obj["bbox"], palette[len(neighbors[idx])])
    out[band_r, :] = inp[band_r, :]
    return out


def _gen_attempt() -> np.ndarray | None:
    scene = _build_scene()
    if scene is None:
        return None
    objects, has_containment = scene

    total_needed = PALETTE_LEN + len(objects)
    if total_needed > len(ARC_COLORS):
        return None

    colors = random.sample(ARC_COLORS, total_needed)
    band_colors = colors[:PALETTE_LEN]
    object_colors = colors[PALETTE_LEN:]
    band_top = random.random() < 0.5

    try:
        inp, placed, band_r = _paint_input(objects, band_top, band_colors, object_colors)
    except ValueError:
        return None

    expected = _expected_output(inp, placed, band_r, has_containment)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
