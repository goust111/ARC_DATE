from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

MARKER_SIZE = 2
FRAME_SIZES = ((4, 4), (4, 5), (5, 4), (5, 5))

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (13, 18)
PAIR_COUNT_CHOICES = (2, 3)
FRAME_COUNT_CHOICES = (2, 3)
PAIR_GAP_MIN = 3
PAIR_GAP_MAX = 8
PAIR_ROW_SEPARATION = 4
MAX_GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ar0 -= int(pad)
    ac0 -= int(pad)
    ar1 += int(pad)
    ac1 += int(pad)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _cell_in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    assert c == c1
    return "right"


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    side = _gap_side(gap, bbox)
    if side == "top":
        return -1, 0
    if side == "bottom":
        return 1, 0
    if side == "left":
        return 0, -1
    return 0, 1


def _ray_path(
    gap: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = gap
    out: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        out.append((int(r), int(c)))
        r += int(dr)
        c += int(dc)
    return out


def _detect_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    comps = _components_of_color(g, BLUE)
    frames: list[dict[str, object]] = []
    for comp in comps:
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4
        assert all(r in (r0, r1) or c in (c0, c1) for r, c in comp)
        missing: list[tuple[int, int]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border:
                    if int(g[r, c]) != BLUE:
                        missing.append((int(r), int(c)))
                else:
                    assert int(g[r, c]) == BG
        assert len(missing) in (0, 1)
        if missing:
            gap = missing[0]
            assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        expected_area = 2 * h + 2 * w - 4 - len(missing)
        assert len(comp) == expected_area
        frames.append({"bbox": bbox, "gap": missing[0] if missing else None})
    frames.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    assert 2 <= len(frames) <= 3
    return frames


def _marker_pairs(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    comps = _components_of_color(g, RED)
    blocks: list[tuple[int, int, int, int]] = []
    for comp in comps:
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        assert len(comp) == 4
        assert r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == RED)
        blocks.append(bbox)
    blocks.sort()
    assert 4 <= len(blocks) <= 6 and len(blocks) % 2 == 0

    by_rows: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for bbox in blocks:
        by_rows.setdefault((bbox[0], bbox[2]), []).append(bbox)

    pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for row_span in sorted(by_rows):
        row_blocks = sorted(by_rows[row_span], key=lambda bbox: bbox[1])
        assert len(row_blocks) == 2
        left, right = row_blocks
        assert left[0] == right[0] and left[2] == right[2]
        assert left[3] + 1 <= right[1] - 1
        pairs.append((left, right))
    assert 2 <= len(pairs) <= 3
    return pairs


def _extract_regions(grid: np.ndarray, pairs) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    regions: list[np.ndarray] = []
    for left, right in pairs:
        stripe = g[left[0] : left[2] + 1, left[3] + 1 : right[1]].copy()
        assert stripe.shape[0] == 2 and stripe.shape[1] > 0
        regions.append(stripe)
    return regions


def _stack_regions(regions: list[np.ndarray]) -> np.ndarray:
    assert regions
    max_w = max(int(region.shape[1]) for region in regions)
    total_h = sum(int(region.shape[0]) for region in regions)
    out = np.full((total_h, max_w), BG, dtype=int)
    cursor = 0
    for region in regions:
        h, w = map(int, region.shape)
        out[cursor : cursor + h, :w] = region
        cursor += h
    return out


def _apply_frame_effects(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _detect_frames(g)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        out[r0 + 1 : r1, c0 + 1 : c1] = TEAL

    h, w = g.shape
    for frame in frames:
        gap = frame["gap"]
        if gap is None:
            continue
        bbox = frame["bbox"]  # type: ignore[misc]
        dr, dc = _gap_direction(gap, bbox)  # type: ignore[arg-type]
        r, c = gap
        while 0 <= r < h and 0 <= c < w and int(out[r, c]) in (BG, GREY):
            out[r, c] = TEAL
            r += int(dr)
            c += int(dc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill blue frames, emit outward teal rays from one-cell gaps, then extract and stack the row-marked strips those rays pass through.

    Input structure:
    - The grid has a black background, 2-3 blue hollow axis-aligned rectangular frames, and 4-6 red 2x2 marker blocks.
    - Each red pair shares the same two rows and defines a 2-row extraction strip strictly between the left and right marker blocks.
    - Some frame borders have one non-corner gap; gray noise may appear on the background, including inside extraction strips.

    Rule:
    - For every blue frame, fill its interior with teal; if it has a one-cell gap, paint a teal ray from that gap straight outward, perpendicular to the side, through black or gray cells until a non-black, non-gray cell blocks it or the ray leaves the grid.
    - Identify the red 2x2 blocks, pair the two blocks that share each two-row span, and crop the rectangle strictly between each pair after the teal fills and rays have already been drawn.
    - Sort the cropped strips from top to bottom and stack them vertically; left-align narrower strips and pad their right side with black.

    Key insight:
    - The final payload is not present in the raw marker strips: teal pixels must first be created by frame gaps and only then extracted, so the ray-frame analysis and the marker-strip stacking are both essential.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    decorated = _apply_frame_effects(g)
    pairs = _marker_pairs(g)
    return _stack_regions(_extract_regions(decorated, pairs))


def _render_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int] | None,
) -> None:
    r0, c0, r1, c1 = bbox
    for c in range(c0, c1 + 1):
        if gap != (r0, c):
            grid[r0, c] = BLUE
        if gap != (r1, c):
            grid[r1, c] = BLUE
    for r in range(r0 + 1, r1):
        if gap != (r, c0):
            grid[r, c0] = BLUE
        if gap != (r, c1):
            grid[r, c1] = BLUE


def _sample_rows(rng: np.random.Generator, height: int, num_pairs: int) -> list[int] | None:
    choices = np.arange(2, height - 2)
    if len(choices) < num_pairs:
        return None
    for _ in range(300):
        rows = sorted(int(x) for x in rng.choice(choices, size=num_pairs, replace=False))
        if any(rows[i + 1] - rows[i] < PAIR_ROW_SEPARATION for i in range(num_pairs - 1)):
            continue
        return rows
    return None


def _pair_static_specs(
    rng: np.random.Generator,
    height: int,
    width: int,
    num_pairs: int,
) -> list[dict[str, object]] | None:
    rows = _sample_rows(rng, height, num_pairs)
    if rows is None:
        return None

    max_gap = min(PAIR_GAP_MAX, width - 4)
    gap_options = list(range(PAIR_GAP_MIN, max_gap + 1))
    if len(gap_options) < num_pairs:
        return None
    widths = list(map(int, rng.choice(gap_options, size=num_pairs, replace=False)))
    rng.shuffle(widths)

    specs: list[dict[str, object]] = []
    for row, gap_width in zip(rows, widths):
        left_max = width - gap_width - 4
        if left_max < 0:
            return None
        left = int(rng.integers(0, left_max + 1))
        right = int(left + MARKER_SIZE + gap_width)
        specs.append(
            {
                "row": int(row),
                "left": int(left),
                "width": int(gap_width),
                "left_bbox": (int(row), int(left), int(row + 1), int(left + 1)),
                "right_bbox": (int(row), int(right), int(row + 1), int(right + 1)),
                "extract_bbox": (int(row), int(left + MARKER_SIZE), int(row + 1), int(right - 1)),
            }
        )
    return specs


def _gap_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _choose_target_frame(
    rng: np.random.Generator,
    shape: tuple[int, int],
    pair_specs: list[dict[str, object]],
) -> dict[str, object] | None:
    height, width = shape
    marker_bboxes = [p["left_bbox"] for p in pair_specs] + [p["right_bbox"] for p in pair_specs]
    extract_bboxes = [p["extract_bbox"] for p in pair_specs]
    options: list[dict[str, object]] = []

    for idx, pair in enumerate(pair_specs):
        row = int(pair["row"])
        _ext_r0, ext_c0, _ext_r1, ext_c1 = pair["extract_bbox"]  # type: ignore[misc]
        for fh, fw in FRAME_SIZES:
            for clearance in (1, 2):
                r1 = row - clearance
                r0 = r1 - fh + 1
                if r0 >= 0:
                    for c0 in range(0, width - fw + 1):
                        bbox = (int(r0), int(c0), int(r1), int(c0 + fw - 1))
                        if any(_bbox_intersects(bbox, other, pad=0) for other in marker_bboxes + extract_bboxes):
                            continue
                        for gap in _gap_cells(bbox):
                            if _gap_side(gap, bbox) != "bottom":
                                continue
                            if not (ext_c0 <= gap[1] <= ext_c1):
                                continue
                            path = _ray_path(gap, (1, 0), (height, width))
                            impacted = [
                                j
                                for j, extract_bbox in enumerate(extract_bboxes)
                                if any(_cell_in_bbox(cell, extract_bbox) for cell in path)
                            ]
                            if impacted != [idx]:
                                continue
                            if any(
                                any(_cell_in_bbox(cell, marker_bbox) for marker_bbox in marker_bboxes)
                                for cell in path[1:]
                            ):
                                continue
                            options.append(
                                {
                                    "bbox": bbox,
                                    "gap": gap,
                                    "path": path,
                                    "pair_index": int(idx),
                                    "score": int(len(path)),
                                }
                            )

                r0 = row + MARKER_SIZE + clearance - 1
                r1 = r0 + fh - 1
                if r1 < height:
                    for c0 in range(0, width - fw + 1):
                        bbox = (int(r0), int(c0), int(r1), int(c0 + fw - 1))
                        if any(_bbox_intersects(bbox, other, pad=0) for other in marker_bboxes + extract_bboxes):
                            continue
                        for gap in _gap_cells(bbox):
                            if _gap_side(gap, bbox) != "top":
                                continue
                            if not (ext_c0 <= gap[1] <= ext_c1):
                                continue
                            path = _ray_path(gap, (-1, 0), (height, width))
                            impacted = [
                                j
                                for j, extract_bbox in enumerate(extract_bboxes)
                                if any(_cell_in_bbox(cell, extract_bbox) for cell in path)
                            ]
                            if impacted != [idx]:
                                continue
                            if any(
                                any(_cell_in_bbox(cell, marker_bbox) for marker_bbox in marker_bboxes)
                                for cell in path[1:]
                            ):
                                continue
                            options.append(
                                {
                                    "bbox": bbox,
                                    "gap": gap,
                                    "path": path,
                                    "pair_index": int(idx),
                                    "score": int(len(path)),
                                }
                            )

    if not options:
        return None
    best = min(int(option["score"]) for option in options)
    shortlist = [option for option in options if int(option["score"]) <= best + 2]
    return shortlist[int(rng.integers(len(shortlist)))]


def _sample_extra_frame(
    rng: np.random.Generator,
    shape: tuple[int, int],
    marker_bboxes: list[tuple[int, int, int, int]],
    extract_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
    frame_mask: np.ndarray,
    ray_mask: np.ndarray,
    want_gap: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int] | None, list[tuple[int, int]]] | None:
    height, width = shape
    forbidden_path = frame_mask | ray_mask
    for bbox in frame_bboxes:
        r0, c0, r1, c1 = bbox
        forbidden_path[r0 : r1 + 1, c0 : c1 + 1] = True
    for bbox in marker_bboxes + extract_bboxes:
        r0, c0, r1, c1 = bbox
        forbidden_path[r0 : r1 + 1, c0 : c1 + 1] = True

    plain: list[tuple[tuple[int, int, int, int], tuple[int, int] | None, list[tuple[int, int]]]] = []
    gap_cands: list[tuple[tuple[int, int, int, int], tuple[int, int] | None, list[tuple[int, int]]]] = []

    for fh, fw in FRAME_SIZES:
        for r0 in range(0, height - fh + 1):
            r1 = r0 + fh - 1
            for c0 in range(0, width - fw + 1):
                c1 = c0 + fw - 1
                bbox = (int(r0), int(c0), int(r1), int(c1))
                if any(_bbox_intersects(bbox, other, pad=0) for other in marker_bboxes + extract_bboxes):
                    continue
                if any(_bbox_intersects(bbox, other, pad=1) for other in frame_bboxes):
                    continue
                if bool(ray_mask[r0 : r1 + 1, c0 : c1 + 1].any()):
                    continue
                plain.append((bbox, None, []))
                for gap in _gap_cells(bbox):
                    dr, dc = _gap_direction(gap, bbox)
                    path = _ray_path(gap, (dr, dc), (height, width))
                    if any(forbidden_path[r, c] for r, c in path[1:]):
                        continue
                    gap_cands.append((bbox, gap, path))

    if want_gap and gap_cands:
        side_cands = [cand for cand in gap_cands if _gap_side(cand[1], cand[0]) in ("left", "right")]
        if side_cands and float(rng.random()) < 0.7:
            gap_cands = side_cands
        return gap_cands[int(rng.integers(len(gap_cands)))]
    if plain:
        return plain[int(rng.integers(len(plain)))]
    if gap_cands:
        return gap_cands[int(rng.integers(len(gap_cands)))]
    return None


def _add_noise(
    rng: np.random.Generator,
    grid: np.ndarray,
    pair_specs: list[dict[str, object]],
    target_path: list[tuple[int, int]],
    frame_mask: np.ndarray,
) -> bool:
    path_set = set(target_path)
    target_pair = None
    for pair in pair_specs:
        extract_bbox = pair["extract_bbox"]  # type: ignore[misc]
        touched = [cell for cell in target_path if _cell_in_bbox(cell, extract_bbox)]
        if touched:
            target_pair = pair
            break
    if target_pair is None:
        return False

    target_extract = target_pair["extract_bbox"]  # type: ignore[misc]
    target_path_cells = [cell for cell in target_path if _cell_in_bbox(cell, target_extract)]
    if not target_path_cells:
        return False

    on_path = target_path_cells[int(rng.integers(len(target_path_cells)))]
    if frame_mask[on_path]:
        return False
    grid[on_path] = GREY

    off_path = [
        (r, c)
        for r in range(target_extract[0], target_extract[2] + 1)
        for c in range(target_extract[1], target_extract[3] + 1)
        if (r, c) not in path_set and not frame_mask[r, c] and int(grid[r, c]) == BG
    ]
    if not off_path:
        return False
    grid[off_path[int(rng.integers(len(off_path)))]] = GREY

    for pair in pair_specs:
        extract_bbox = pair["extract_bbox"]  # type: ignore[misc]
        cells = [
            (r, c)
            for r in range(extract_bbox[0], extract_bbox[2] + 1)
            for c in range(extract_bbox[1], extract_bbox[3] + 1)
            if not frame_mask[r, c] and int(grid[r, c]) == BG and (r, c) not in path_set
        ]
        if pair is target_pair:
            if cells and float(rng.random()) < 0.5:
                grid[cells[int(rng.integers(len(cells)))]] = GREY
            continue
        n = int(rng.integers(0, min(2, len(cells)) + 1))
        if n:
            picks = rng.choice(len(cells), size=n, replace=False)
            for idx in np.atleast_1d(picks):
                grid[cells[int(idx)]] = GREY

    outside = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if not frame_mask[r, c] and int(grid[r, c]) == BG and (r, c) not in path_set
    ]
    extra = min(len(outside), int(rng.integers(0, 4)))
    if extra:
        picks = rng.choice(len(outside), size=extra, replace=False)
        for idx in np.atleast_1d(picks):
            grid[outside[int(idx)]] = GREY
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        frames = _detect_frames(g)
        pairs = _marker_pairs(g)
        if not any(frame["gap"] is None for frame in frames):
            return False
        if not any(frame["gap"] is not None for frame in frames):
            return False
        decorated = _apply_frame_effects(g)
        raw_regions = _extract_regions(g, pairs)
        dec_regions = _extract_regions(decorated, pairs)
        widths = [int(region.shape[1]) for region in dec_regions]
        if len(widths) > 1 and len(set(widths)) < 2:
            return False
        changed = [not np.array_equal(raw, dec) for raw, dec in zip(raw_regions, dec_regions)]
        if sum(changed) != 1:
            return False
        raw_stack = _stack_regions(raw_regions)
        dec_stack = _stack_regions(dec_regions)
        if np.array_equal(raw_stack, dec_stack):
            return False
        if not np.any((raw_stack == GREY) & (dec_stack == TEAL)):
            return False
        if not np.any((raw_stack == GREY) & (dec_stack == GREY)):
            return False
        if not np.any(dec_stack == TEAL):
            return False
        if not np.array_equal(main(g), dec_stack):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    width = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    num_pairs = int(rng.choice(PAIR_COUNT_CHOICES))
    frame_count = int(rng.choice(FRAME_COUNT_CHOICES))

    pair_specs = _pair_static_specs(rng, height, width, num_pairs)
    if pair_specs is None:
        return None

    target = _choose_target_frame(rng, (height, width), pair_specs)
    if target is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    marker_bboxes = [p["left_bbox"] for p in pair_specs] + [p["right_bbox"] for p in pair_specs]
    extract_bboxes = [p["extract_bbox"] for p in pair_specs]

    for pair in pair_specs:
        row = int(pair["row"])
        left = int(pair["left"])
        gap_width = int(pair["width"])
        right = left + MARKER_SIZE + gap_width
        grid[row : row + MARKER_SIZE, left : left + MARKER_SIZE] = RED
        grid[row : row + MARKER_SIZE, right : right + MARKER_SIZE] = RED

    frame_bboxes = [target["bbox"]]  # type: ignore[list-item]
    frame_specs = [target]
    frame_mask = np.zeros((height, width), dtype=bool)
    r0, c0, r1, c1 = target["bbox"]  # type: ignore[misc]
    frame_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    ray_mask = np.zeros((height, width), dtype=bool)
    for r, c in target["path"]:  # type: ignore[index]
        ray_mask[r, c] = True

    for extra_idx in range(frame_count - 1):
        want_gap = extra_idx == 0 and float(rng.random()) < 0.65
        if extra_idx > 0:
            want_gap = float(rng.random()) < 0.35
        extra = _sample_extra_frame(
            rng,
            (height, width),
            marker_bboxes,  # type: ignore[arg-type]
            extract_bboxes,  # type: ignore[arg-type]
            frame_bboxes,  # type: ignore[arg-type]
            frame_mask.copy(),
            ray_mask.copy(),
            want_gap,
        )
        if extra is None:
            return None
        bbox, gap, path = extra
        frame_bboxes.append(bbox)
        frame_specs.append({"bbox": bbox, "gap": gap, "path": path})
        er0, ec0, er1, ec1 = bbox
        frame_mask[er0 : er1 + 1, ec0 : ec1 + 1] = True
        for r, c in path:
            ray_mask[r, c] = True

    for spec in frame_specs:
        _render_frame(grid, spec["bbox"], spec["gap"])  # type: ignore[arg-type]

    if not _add_noise(rng, grid, pair_specs, target["path"], frame_mask):  # type: ignore[arg-type]
        return None

    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
