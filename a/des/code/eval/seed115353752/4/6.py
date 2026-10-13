import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GEN_ATTEMPTS = 400
ANCHOR_TRIES = 80
PLACE_TRIES = 80

INNER_H_RANGE = (8, 15)
INNER_W_RANGE = (8, 15)
TOP_MARGIN_RANGE = (5, 10)
BOTTOM_MARGIN_RANGE = (3, 7)
SIDE_MARGIN_RANGE = (4, 9)
NUM_SOURCES_RANGE = (2, 6)
SOURCE_H_RANGE = (2, 6)
SOURCE_W_RANGE = (2, 6)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _frame_border_coords(frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == FRAME)
    assert grey.size > 0, "no grey frame pixels found"
    r0, c0, r1, c1 = _bbox_from_coords(grey)
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must be at least 5x5"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0 and np.all(interior == BG), "frame interior must be black in the input"

    border_values = [int(grid[r, c]) for r, c in _frame_border_coords((r0, c0, r1, c1))]
    assert all(v != BG for v in border_values), "frame border cannot contain black gaps"
    assert all(v == FRAME or v in COLOR_POOL for v in border_values), "unexpected border color"
    return r0, c0, r1, c1


def _anchor_side(anchor: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    r, c = anchor
    r0, c0, r1, c1 = frame_bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    assert c == c1, "anchor must sit on the frame border"
    return "right"


def _extract_anchors(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict[int, tuple[int, int]]:
    anchors: dict[int, tuple[int, int]] = {}
    for r, c in _frame_border_coords(frame_bbox):
        color = int(grid[r, c])
        if color == FRAME:
            continue
        assert color != BG, "anchor cannot be black"
        assert color not in anchors, "anchor colors must be unique"
        anchors[color] = (r, c)
    assert anchors, "at least one anchor is required"
    return anchors


def _aligned_edge_index(mask: np.ndarray, side: str) -> int:
    if side == "top":
        idx = np.flatnonzero(mask[0])
    elif side == "bottom":
        idx = np.flatnonzero(mask[-1])
    elif side == "left":
        idx = np.flatnonzero(mask[:, 0])
    else:
        idx = np.flatnonzero(mask[:, -1])
    assert len(idx) == 1, "the aligned source edge must have exactly one active pixel"
    return int(idx[0])


def _extract_source(
    grid: np.ndarray,
    color: int,
    anchor: tuple[int, int],
) -> dict[str, object]:
    mask = grid == color
    mask[anchor] = False
    comps = _components(mask)
    assert len(comps) == 1, f"expected one source component for color {color}"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "color": int(color),
        "anchor": anchor,
        "bbox": (r0, c0, r1, c1),
        "mask": sub,
        "coords": coords,
    }


def _project_cells(
    mask: np.ndarray,
    side: str,
    anchor: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    h, w = mask.shape
    k = _aligned_edge_index(mask, side)
    cells: list[tuple[int, int]] = []

    for dr, dc in np.argwhere(mask):
        if side == "top":
            rr = r0 + 1 + int(dr)
            cc = anchor[1] + int(dc) - k
        elif side == "bottom":
            rr = r1 - h + int(dr)
            cc = anchor[1] + int(dc) - k
        elif side == "left":
            rr = anchor[0] + int(dr) - k
            cc = c0 + 1 + int(dc)
        else:
            rr = anchor[0] + int(dr) - k
            cc = c1 - w + int(dc)

        if r0 < rr < r1 and c0 < cc < c1:
            cells.append((int(rr), int(cc)))
    return cells


def _build_output(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    sources: list[dict[str, object]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = frame_bbox

    for source in sources:
        coords = source["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    out[r0 + 1 : r1, c0 + 1 : c1] = BG

    parity = np.zeros_like(out, dtype=np.int16)
    last = np.full_like(out, BG)
    order = sorted(sources, key=lambda spec: (spec["anchor"][0], spec["anchor"][1]))
    for source in order:
        side = _anchor_side(source["anchor"], frame_bbox)
        for rr, cc in _project_cells(source["mask"], side, source["anchor"], frame_bbox):
            parity[rr, cc] += 1
            last[rr, cc] = int(source["color"])

    interior = (slice(r0 + 1, r1), slice(c0 + 1, c1))
    keep = (parity[interior] % 2) == 1
    inner = out[interior]
    inner[keep] = last[interior][keep]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each outside monochrome source through the same-color border anchor into the frame.

    Sources are treated as binary templates. The anchor side picks the inward projection direction,
    and the unique occupied cell on the corresponding source edge is aligned with the anchor. Overlaps
    use parity: even coverage cancels to black, odd coverage keeps the last projected color.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox = _find_frame_bbox(g)
    anchors = _extract_anchors(g, frame_bbox)
    sources = []
    for color, anchor in anchors.items():
        source = _extract_source(g, color, anchor)
        _aligned_edge_index(source["mask"], _anchor_side(anchor, frame_bbox))
        sources.append(source)
    return _build_output(g, frame_bbox, sources)


def _on_forbidden_edge(cell: tuple[int, int], side: str, h: int, w: int, entry: tuple[int, int]) -> bool:
    r, c = cell
    if cell == entry:
        return False
    if side == "top":
        return r == 0
    if side == "bottom":
        return r == h - 1
    if side == "left":
        return c == 0
    return c == w - 1


def _step_inward(cell: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = cell
    if side == "top":
        return r + 1, c
    if side == "bottom":
        return r - 1, c
    if side == "left":
        return r, c + 1
    return r, c - 1


def _connect_path(
    start: tuple[int, int],
    target: tuple[int, int],
    side: str,
    h: int,
    w: int,
    entry: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    r, c = start
    tr, tc = target

    if (r, c) != (tr, tc) and _on_forbidden_edge((r, c), side, h, w, entry):
        r, c = _step_inward((r, c), side)
        path.append((r, c))

    while r != tr:
        r += 1 if tr > r else -1
        if _on_forbidden_edge((r, c), side, h, w, entry):
            c += 1 if tc >= c else -1
        path.append((r, c))
    while c != tc:
        c += 1 if tc > c else -1
        path.append((r, c))
    return path


def _random_source_mask(rng: np.random.Generator, h: int, w: int, side: str) -> np.ndarray:
    for _ in range(80):
        if side == "top":
            entry = (0, int(rng.integers(w)))
            left_target = (int(rng.integers(1, h)), 0)
            right_target = (int(rng.integers(1, h)), w - 1)
            bottom_target = (h - 1, int(rng.integers(w)))
            targets = [entry, left_target, right_target, bottom_target]
        elif side == "bottom":
            entry = (h - 1, int(rng.integers(w)))
            left_target = (int(rng.integers(0, h - 1)), 0)
            right_target = (int(rng.integers(0, h - 1)), w - 1)
            top_target = (0, int(rng.integers(w)))
            targets = [entry, left_target, right_target, top_target]
        elif side == "left":
            entry = (int(rng.integers(h)), 0)
            top_target = (0, int(rng.integers(1, w)))
            bottom_target = (h - 1, int(rng.integers(1, w)))
            right_target = (int(rng.integers(h)), w - 1)
            targets = [entry, top_target, bottom_target, right_target]
        else:
            entry = (int(rng.integers(h)), w - 1)
            top_target = (0, int(rng.integers(0, w - 1)))
            bottom_target = (h - 1, int(rng.integers(0, w - 1)))
            left_target = (int(rng.integers(h)), 0)
            targets = [entry, top_target, bottom_target, left_target]

        cells = {entry}
        failed = False
        for target in targets[1:]:
            start = list(cells)[int(rng.integers(len(cells)))]
            for cell in _connect_path(start, target, side, h, w, entry):
                if _on_forbidden_edge(cell, side, h, w, entry):
                    failed = True
                    break
                cells.add(cell)
            if failed:
                break
            cells.add(target)
        if failed:
            continue

        target_area = int(rng.integers(len(cells), h * w + 1))
        target_area = max(target_area, max(4, (h * w + 1) // 2))
        frontier: set[tuple[int, int]] = set()

        def refresh_frontier() -> None:
            frontier.clear()
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        if not _on_forbidden_edge((nr, nc), side, h, w, entry):
                            frontier.add((nr, nc))

        refresh_frontier()
        while len(cells) < target_area and frontier:
            choice = list(frontier)[int(rng.integers(len(frontier)))]
            cells.add(choice)
            refresh_frontier()

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if len(_components(mask)) != 1:
            continue
        try:
            _aligned_edge_index(mask, side)
        except AssertionError:
            continue
        if not (np.any(mask[0]) and np.any(mask[-1]) and np.any(mask[:, 0]) and np.any(mask[:, -1])):
            continue
        return mask

    raise RuntimeError("failed to build a connected source mask")


def _expanded_free(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0] - 1, r1 + 1)
    cc1 = min(occupied.shape[1] - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_occupied(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_source_bbox(
    rng: np.random.Generator,
    side: str,
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> tuple[int, int, int, int] | None:
    h, w = grid_shape
    fr0, fc0, fr1, fc1 = frame_bbox
    for _ in range(PLACE_TRIES):
        sh = int(rng.integers(SOURCE_H_RANGE[0], SOURCE_H_RANGE[1]))
        sw = int(rng.integers(SOURCE_W_RANGE[0], SOURCE_W_RANGE[1]))

        if side == "top":
            if fr0 - sh - 2 < 0:
                continue
            sr0 = int(rng.integers(0, fr0 - sh - 1))
            sc0 = int(rng.integers(0, w - sw + 1))
        elif side == "bottom":
            if fr1 + sh + 2 >= h:
                continue
            sr0 = int(rng.integers(fr1 + 2, h - sh + 1))
            sc0 = int(rng.integers(0, w - sw + 1))
        elif side == "left":
            if fc0 - sw - 2 < 0:
                continue
            sr0 = int(rng.integers(0, h - sh + 1))
            sc0 = int(rng.integers(0, fc0 - sw - 1))
        else:
            if fc1 + sw + 2 >= w:
                continue
            sr0 = int(rng.integers(0, h - sh + 1))
            sc0 = int(rng.integers(fc1 + 2, w - sw + 1))

        bbox = (sr0, sc0, sr0 + sh - 1, sc0 + sw - 1)
        if _expanded_free(occupied, bbox):
            return bbox
    return None


def _projection_bounds(
    mask: np.ndarray,
    side: str,
    anchor: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    pts = _project_cells(mask, side, anchor, frame_bbox)
    assert pts, "every source must project at least one cell into the frame"
    arr = np.asarray(pts, dtype=int)
    return _bbox_from_coords(arr)


def _spec_output(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], specs: list[dict[str, object]]) -> np.ndarray:
    sources = []
    for spec in specs:
        bbox = spec["bbox"]
        r0, c0, _, _ = bbox
        coords = np.argwhere(spec["mask"])
        abs_coords = coords + np.asarray([r0, c0], dtype=int)
        sources.append(
            {
                "color": spec["color"],
                "anchor": spec["anchor"],
                "bbox": bbox,
                "mask": spec["mask"],
                "coords": abs_coords,
            }
        )
    return _build_output(grid, frame_bbox, sources)


def _candidate_anchor_sets(
    rng: np.random.Generator,
    frame_bbox: tuple[int, int, int, int],
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    fr0, fc0, fr1, fc1 = frame_bbox
    used_top: set[int] = set()
    used_bottom: set[int] = set()
    used_left: set[int] = set()
    used_right: set[int] = set()

    anchored: list[dict[str, object]] = []
    for spec in specs:
        side = spec["side"]
        mask = spec["mask"]
        if side == "top":
            k = _aligned_edge_index(mask, side)
            candidates = [c for c in range(fc0 + 1, fc1) if c not in used_top]
            edge_used = used_top
            anchor = (fr0, int(candidates[int(rng.integers(len(candidates)))]) if candidates else -1)
        elif side == "bottom":
            k = _aligned_edge_index(mask, side)
            candidates = [c for c in range(fc0 + 1, fc1) if c not in used_bottom]
            edge_used = used_bottom
            anchor = (fr1, int(candidates[int(rng.integers(len(candidates)))]) if candidates else -1)
        elif side == "left":
            k = _aligned_edge_index(mask, side)
            candidates = [r for r in range(fr0 + 1, fr1) if r not in used_left]
            edge_used = used_left
            anchor = (int(candidates[int(rng.integers(len(candidates)))]) if candidates else -1, fc0)
        else:
            k = _aligned_edge_index(mask, side)
            candidates = [r for r in range(fr0 + 1, fr1) if r not in used_right]
            edge_used = used_right
            anchor = (int(candidates[int(rng.integers(len(candidates)))]) if candidates else -1, fc1)

        if not candidates:
            return None
        edge_used.add(anchor[1] if side in ("top", "bottom") else anchor[0])
        anchored.append({**spec, "anchor": anchor, "edge_index": k})
    return anchored


def _coverage_stats(
    frame_bbox: tuple[int, int, int, int],
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, int, int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    counts = np.zeros((fr1 - fr0 - 1, fc1 - fc0 - 1), dtype=int)
    clipped_sources = 0
    projected_cells = 0

    for spec in specs:
        side = spec["side"]
        cells = _project_cells(spec["mask"], side, spec["anchor"], frame_bbox)
        total = int(np.count_nonzero(spec["mask"]))
        if len(cells) < total:
            clipped_sources += 1
        if not cells:
            return counts, clipped_sources, 0, 0
        projected_cells += len(cells)
        for rr, cc in cells:
            counts[rr - fr0 - 1, cc - fc0 - 1] += 1

    return counts, clipped_sources, projected_cells, int(counts.max()) if counts.size else 0


def _paint_input(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    specs: list[dict[str, object]],
) -> np.ndarray:
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    for c in range(fc0, fc1 + 1):
        grid[fr0, c] = FRAME
        grid[fr1, c] = FRAME
    for r in range(fr0 + 1, fr1):
        grid[r, fc0] = FRAME
        grid[r, fc1] = FRAME

    for spec in specs:
        ar, ac = spec["anchor"]
        grid[ar, ac] = int(spec["color"])
        r0, c0, _, _ = spec["bbox"]
        coords = np.argwhere(spec["mask"])
        for dr, dc in coords:
            grid[r0 + dr, c0 + dc] = int(spec["color"])
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    inner_h = int(rng.integers(INNER_H_RANGE[0], INNER_H_RANGE[1]))
    inner_w = int(rng.integers(INNER_W_RANGE[0], INNER_W_RANGE[1]))
    top_margin = int(rng.integers(TOP_MARGIN_RANGE[0], TOP_MARGIN_RANGE[1]))
    bottom_margin = int(rng.integers(BOTTOM_MARGIN_RANGE[0], BOTTOM_MARGIN_RANGE[1]))
    left_margin = int(rng.integers(SIDE_MARGIN_RANGE[0], SIDE_MARGIN_RANGE[1]))
    right_margin = int(rng.integers(SIDE_MARGIN_RANGE[0], SIDE_MARGIN_RANGE[1]))

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid_h = top_margin + frame_h + bottom_margin
    grid_w = left_margin + frame_w + right_margin
    frame_bbox = (top_margin, left_margin, top_margin + frame_h - 1, left_margin + frame_w - 1)

    num_sources = int(rng.integers(NUM_SOURCES_RANGE[0], NUM_SOURCES_RANGE[1]))
    colors = list(rng.choice(COLOR_POOL, size=num_sources, replace=False))
    base_sides = list(rng.permutation(SIDES))
    sides = base_sides[: min(4, num_sources)]
    while len(sides) < num_sources:
        sides.append(str(SIDES[int(rng.integers(len(SIDES)))]))
    rng.shuffle(sides)

    want_overlap = bool(rng.integers(0, 2))
    want_clip = bool(rng.integers(0, 2))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_occupied(occupied, frame_bbox)
    specs: list[dict[str, object]] = []
    for color, side in zip(colors, sides):
        bbox = _place_source_bbox(rng, side, (grid_h, grid_w), frame_bbox, occupied)
        if bbox is None:
            return None
        _mark_occupied(occupied, bbox)
        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        mask = _random_source_mask(rng, h, w, side)
        specs.append({"color": int(color), "side": side, "bbox": bbox, "mask": mask})

    fr0, fc0, fr1, fc1 = frame_bbox
    min_expected_cells = max(6, num_sources * 3)

    for _ in range(ANCHOR_TRIES):
        anchored = _candidate_anchor_sets(rng, frame_bbox, specs)
        if anchored is None:
            return None

        counts, clipped_sources, projected_cells, max_cover = _coverage_stats(frame_bbox, anchored)
        if projected_cells < min_expected_cells:
            continue
        if max_cover > 2:
            continue

        overlap_cells = int(np.count_nonzero(counts == 2))
        if want_overlap and overlap_cells == 0:
            continue
        if not want_overlap and overlap_cells != 0:
            continue
        if want_clip and clipped_sources == 0:
            continue
        if not want_clip and clipped_sources != 0:
            continue

        output_mask = counts % 2 == 1
        if int(np.count_nonzero(output_mask)) < max(4, projected_cells // 3):
            continue

        grid = _paint_input((grid_h, grid_w), frame_bbox, anchored)
        expected = _spec_output(grid, frame_bbox, anchored)
        solved = main(grid)
        if not np.array_equal(expected, solved):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
