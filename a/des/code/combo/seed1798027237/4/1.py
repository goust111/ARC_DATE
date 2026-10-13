from __future__ import annotations

import numpy as np


SIDES = ("top", "bottom", "left", "right")
REQUIRED_NOTCH = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(
    grid: np.ndarray, excluded: set[int]
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _contiguous_run(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0
    start = int(indices[0])
    end = int(indices[-1])
    assert np.array_equal(indices, np.arange(start, end + 1))
    return start, end


def _edge_gap(
    line: np.ndarray, frame_color: int, offset: int
) -> tuple[int, int]:
    missing = np.where(line != int(frame_color))[0]
    start, end = _contiguous_run(missing)
    length = end - start + 1
    assert length in (2, 3)
    return int(offset + start), int(offset + end)


def _closest_side(
    bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]
) -> str:
    r0, c0, r1, c1 = bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    dists = {
        "top": int(r0 - fr0),
        "bottom": int(fr1 - r1),
        "left": int(c0 - fc0),
        "right": int(fc1 - c1),
    }
    ordered = sorted(dists.items(), key=lambda item: (item[1], item[0]))
    assert ordered[0][1] < ordered[1][1]
    return ordered[0][0]


def _analyze_stamp(
    grid: np.ndarray, comp: dict[str, object], bg: int
) -> dict[str, object]:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    filled = patch != int(bg)
    missing = ~filled
    total_missing = int(missing.sum())
    assert total_missing in (2, 3)

    candidates: list[tuple[str, int, int]] = []
    edge_masks = {
        "top": np.pad(
            np.ones((1, patch.shape[1]), dtype=bool),
            ((0, patch.shape[0] - 1), (0, 0)),
            constant_values=False,
        ),
        "bottom": np.pad(
            np.ones((1, patch.shape[1]), dtype=bool),
            ((patch.shape[0] - 1, 0), (0, 0)),
            constant_values=False,
        ),
        "left": np.pad(
            np.ones((patch.shape[0], 1), dtype=bool),
            ((0, 0), (0, patch.shape[1] - 1)),
            constant_values=False,
        ),
        "right": np.pad(
            np.ones((patch.shape[0], 1), dtype=bool),
            ((0, 0), (patch.shape[1] - 1, 0)),
            constant_values=False,
        ),
    }

    for side, edge_mask in edge_masks.items():
        if not np.array_equal(missing, missing & edge_mask):
            continue
        if side in ("top", "bottom"):
            idx = np.where(missing[0] if side == "top" else missing[-1])[0]
        else:
            idx = np.where(missing[:, 0] if side == "left" else missing[:, -1])[0]
        if idx.size != total_missing:
            continue
        start, end = _contiguous_run(idx)
        candidates.append((side, int(start), int(end - start + 1)))

    assert len(candidates) == 1
    notch_side, notch_start, notch_len = candidates[0]
    return {
        "color": int(comp["color"]),  # type: ignore[index]
        "coords": list(comp["coords"]),  # type: ignore[index]
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
        "notch_side": notch_side,
        "notch_start": int(notch_start),
        "notch_len": int(notch_len),
        "height": int(patch.shape[0]),
        "width": int(patch.shape[1]),
    }


def _paste_non_bg(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= out.shape[0] and c0 + w <= out.shape[1]
    mask = patch != int(bg)
    view = out[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _moved_stamp_position(
    side: str,
    stamp: dict[str, object],
    frame_bbox: tuple[int, int, int, int],
    gaps: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    gap_start, _gap_end = gaps[side]
    notch_start = int(stamp["notch_start"])
    h = int(stamp["height"])
    w = int(stamp["width"])

    if side == "top":
        return int(fr0 - h + 1), int(gap_start - notch_start)
    if side == "bottom":
        return int(fr1), int(gap_start - notch_start)
    if side == "left":
        return int(gap_start - notch_start), int(fc0 - w + 1)
    return int(gap_start - notch_start), int(fc1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame-side gap to a corner stamp via the largest nearest interior block color.

    The grey frame color is inferred as the most frequent non-background color.
    Each corner stamp is removed from its original location, translated so its notch aligns
    with the matching frame gap, and the frame border is then redrawn solid grey.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    non_bg = [int(v) for v in np.unique(g) if int(v) != int(bg)]
    assert non_bg
    counts = {color: int(np.sum(g == color)) for color in non_bg}
    frame_color = max(non_bg, key=lambda color: counts[color])

    frame_bbox = _bbox_from_mask(g == int(frame_color))
    fr0, fc0, fr1, fc1 = frame_bbox
    gaps = {
        "top": _edge_gap(g[fr0, fc0 : fc1 + 1], frame_color, fc0),
        "bottom": _edge_gap(g[fr1, fc0 : fc1 + 1], frame_color, fc0),
        "left": _edge_gap(g[fr0 : fr1 + 1, fc0], frame_color, fr0),
        "right": _edge_gap(g[fr0 : fr1 + 1, fc1], frame_color, fr0),
    }

    comps = _find_components(g, excluded={int(bg), int(frame_color)})
    inside: list[dict[str, object]] = []
    outside: list[dict[str, object]] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
        if fr0 < r0 and r1 < fr1 and fc0 < c0 and c1 < fc1:
            inside.append(comp)
        else:
            outside.append(comp)

    best_by_side: dict[str, dict[str, object]] = {}
    for comp in inside:
        side = _closest_side(comp["bbox"], frame_bbox)  # type: ignore[arg-type]
        current = best_by_side.get(side)
        if current is None or int(comp["area"]) > int(current["area"]):
            best_by_side[side] = comp
    assert all(side in best_by_side for side in SIDES)

    stamps_by_color: dict[int, dict[str, object]] = {}
    for comp in outside:
        stamp = _analyze_stamp(g, comp, bg)
        stamps_by_color[int(stamp["color"])] = stamp

    out = g.copy()
    for stamp in stamps_by_color.values():
        for r, c in stamp["coords"]:  # type: ignore[index]
            out[int(r), int(c)] = int(bg)

    for side in SIDES:
        color = int(best_by_side[side]["color"])
        stamp = stamps_by_color[color]
        assert str(stamp["notch_side"]) == REQUIRED_NOTCH[side]
        r0, c0 = _moved_stamp_position(side, stamp, frame_bbox, gaps)
        _paste_non_bg(out, stamp["patch"], r0, c0, bg)  # type: ignore[arg-type]

    out[fr0, fc0 : fc1 + 1] = int(frame_color)
    out[fr1, fc0 : fc1 + 1] = int(frame_color)
    out[fr0 : fr1 + 1, fc0] = int(frame_color)
    out[fr0 : fr1 + 1, fc1] = int(frame_color)
    return out


def _random_polyomino(
    rng: np.random.Generator, area: int
) -> np.ndarray:
    dims = [(h, w) for h in range(1, 5) for w in range(1, 5) if h * w >= area]
    for _ in range(60):
        h, w = dims[int(rng.integers(0, len(dims)))]
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != area:
            continue
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        patch = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=int)
        for r, c in cells:
            patch[r - min(rows), c - min(cols)] = 1
        return patch
    raise RuntimeError("failed to sample block shape")


def _mark_blocked(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_block(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color: int,
    patch_mask: np.ndarray,
    desired_side: str,
) -> dict[str, object] | None:
    fr0, fc0, fr1, fc1 = frame_bbox
    h, w = patch_mask.shape
    row_min = fr0 + 1
    row_max = fr1 - h
    col_min = fc0 + 1
    col_max = fc1 - w
    if row_min > row_max or col_min > col_max:
        return None

    for _ in range(200):
        r0 = int(rng.integers(row_min, row_max + 1))
        c0 = int(rng.integers(col_min, col_max + 1))
        cells = np.argwhere(patch_mask > 0)
        coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in cells]
        if any(blocked[rr, cc] for rr, cc in coords):
            continue
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        dists = {
            "top": int(bbox[0] - fr0),
            "bottom": int(fr1 - bbox[2]),
            "left": int(bbox[1] - fc0),
            "right": int(fc1 - bbox[3]),
        }
        ordered = sorted(dists.items(), key=lambda item: (item[1], item[0]))
        if ordered[0][1] == ordered[1][1] or ordered[0][0] != desired_side:
            continue
        for rr, cc in coords:
            grid[rr, cc] = int(color)
        _mark_blocked(blocked, bbox)
        return {
            "color": int(color),
            "area": int(len(coords)),
            "bbox": bbox,
            "coords": coords,
            "side": desired_side,
        }
    return None


def _make_stamp(
    size: int, notch_side: str, notch_start: int, notch_len: int, color: int, bg: int
) -> np.ndarray:
    patch = np.full((size, size), int(color), dtype=int)
    if notch_side == "top":
        patch[0, notch_start : notch_start + notch_len] = int(bg)
    elif notch_side == "bottom":
        patch[-1, notch_start : notch_start + notch_len] = int(bg)
    elif notch_side == "left":
        patch[notch_start : notch_start + notch_len, 0] = int(bg)
    else:
        patch[notch_start : notch_start + notch_len, -1] = int(bg)
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _draw_broken_frame(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    frame_color: int,
    gaps: dict[str, tuple[int, int]],
) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = int(frame_color)
    grid[fr1, fc0 : fc1 + 1] = int(frame_color)
    grid[fr0 : fr1 + 1, fc0] = int(frame_color)
    grid[fr0 : fr1 + 1, fc1] = int(frame_color)

    top0, top1 = gaps["top"]
    bot0, bot1 = gaps["bottom"]
    left0, left1 = gaps["left"]
    right0, right1 = gaps["right"]
    grid[fr0, top0 : top1 + 1] = 0
    grid[fr1, bot0 : bot1 + 1] = 0
    grid[left0 : left1 + 1, fc0] = 0
    grid[right0 : right1 + 1, fc1] = 0


def _compose_expected(
    inp: np.ndarray,
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    corner_stamps: list[dict[str, object]],
    side_stamps: dict[str, dict[str, object]],
    gaps: dict[str, tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for stamp in corner_stamps:
        patch = stamp["patch"]  # type: ignore[index]
        r0 = int(stamp["r0"])
        c0 = int(stamp["c0"])
        h, w = patch.shape
        out[r0 : r0 + h, c0 : c0 + w][patch != int(bg)] = int(bg)

    for side in SIDES:
        stamp = side_stamps[side]
        r0, c0 = _moved_stamp_position(side, stamp, frame_bbox, gaps)
        _paste_non_bg(out, stamp["patch"], r0, c0, bg)  # type: ignore[arg-type]

    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0, fc0 : fc1 + 1] = int(frame_color)
    out[fr1, fc0 : fc1 + 1] = int(frame_color)
    out[fr0 : fr1 + 1, fc0] = int(frame_color)
    out[fr0 : fr1 + 1, fc1] = int(frame_color)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = 0
    frame_color = 5

    stamp_sizes = {side: int(rng.integers(3, 5)) for side in SIDES}
    margin_top = int(rng.integers(max(stamp_sizes["left"], stamp_sizes["top"]), 8))
    margin_bottom = int(rng.integers(max(stamp_sizes["left"], stamp_sizes["bottom"]), 8))
    margin_left = int(rng.integers(max(stamp_sizes["top"], stamp_sizes["left"]), 8))
    margin_right = int(rng.integers(max(stamp_sizes["top"], stamp_sizes["right"]), 8))

    frame_h = int(rng.integers(11, 17))
    frame_w = int(rng.integers(11, 17))
    height = margin_top + frame_h + margin_bottom
    width = margin_left + frame_w + margin_right

    frame_bbox = (
        int(margin_top),
        int(margin_left),
        int(margin_top + frame_h - 1),
        int(margin_left + frame_w - 1),
    )
    fr0, fc0, fr1, fc1 = frame_bbox

    palette = [int(x) for x in rng.permutation([1, 2, 3, 4, 6, 7, 8, 9])]
    primary_colors = {side: palette[i] for i, side in enumerate(SIDES)}
    extra_colors = palette[4:]

    primary_areas = {side: int(rng.integers(4, 6)) for side in SIDES}
    n_extra = int(rng.integers(1, len(extra_colors) + 1))
    extra_specs = []
    for i in range(n_extra):
        side = SIDES[int(rng.integers(0, len(SIDES)))]
        max_area = primary_areas[side] - 1
        area = 2 if max_area <= 2 else int(rng.integers(2, max_area + 1))
        extra_specs.append((extra_colors[i], side, area))

    grid = np.zeros((height, width), dtype=int)

    gap_lens = {
        "top": int(rng.integers(2, 4)),
        "bottom": int(rng.integers(2, 4)),
        "left": int(rng.integers(2, 4)),
        "right": int(rng.integers(2, 4)),
    }
    gaps = {
        "top": (
            int(fc0 + rng.integers(1, frame_w - gap_lens["top"])),
            0,
        ),
        "bottom": (
            int(fc0 + rng.integers(1, frame_w - gap_lens["bottom"])),
            0,
        ),
        "left": (
            int(fr0 + rng.integers(1, frame_h - gap_lens["left"])),
            0,
        ),
        "right": (
            int(fr0 + rng.integers(1, frame_h - gap_lens["right"])),
            0,
        ),
    }
    gaps["top"] = (gaps["top"][0], gaps["top"][0] + gap_lens["top"] - 1)
    gaps["bottom"] = (gaps["bottom"][0], gaps["bottom"][0] + gap_lens["bottom"] - 1)
    gaps["left"] = (gaps["left"][0], gaps["left"][0] + gap_lens["left"] - 1)
    gaps["right"] = (gaps["right"][0], gaps["right"][0] + gap_lens["right"] - 1)

    side_stamps: dict[str, dict[str, object]] = {}
    moved_boxes: list[tuple[int, int, int, int]] = []
    for side in SIDES:
        size = stamp_sizes[side]
        notch_len = gap_lens[side]
        notch_start = int(rng.integers(0, size - notch_len + 1))
        patch = _make_stamp(
            size=size,
            notch_side=REQUIRED_NOTCH[side],
            notch_start=notch_start,
            notch_len=notch_len,
            color=primary_colors[side],
            bg=bg,
        )
        stamp = _analyze_stamp(
            patch,
            {"bbox": (0, 0, size - 1, size - 1), "color": primary_colors[side], "coords": []},
            bg,
        )
        stamp["patch"] = patch
        stamp["height"] = size
        stamp["width"] = size
        r0, c0 = _moved_stamp_position(side, stamp, frame_bbox, gaps)
        box = (r0, c0, r0 + size - 1, c0 + size - 1)
        if not (0 <= box[0] and 0 <= box[1] and box[2] < height and box[3] < width):
            return None
        if any(_boxes_overlap(box, other) for other in moved_boxes):
            return None
        moved_boxes.append(box)
        side_stamps[side] = stamp

    _draw_broken_frame(grid, frame_bbox, frame_color, gaps)

    blocked = np.zeros_like(grid, dtype=bool)
    primary_blocks: list[dict[str, object]] = []
    for side in SIDES:
        patch_mask = _random_polyomino(rng, primary_areas[side])
        block = _place_block(
            rng, grid, blocked, frame_bbox, primary_colors[side], patch_mask, side
        )
        if block is None:
            return None
        primary_blocks.append(block)

    for color, side, area in extra_specs:
        patch_mask = _random_polyomino(rng, area)
        block = _place_block(rng, grid, blocked, frame_bbox, color, patch_mask, side)
        if block is None:
            return None

    corner_regions = {
        "tl": (0, fr0 - 1, 0, fc0 - 1),
        "tr": (0, fr0 - 1, fc1 + 1, width - 1),
        "bl": (fr1 + 1, height - 1, 0, fc0 - 1),
        "br": (fr1 + 1, height - 1, fc1 + 1, width - 1),
    }
    corners = list(corner_regions.keys())
    rng.shuffle(corners)
    corner_stamps: list[dict[str, object]] = []
    for side, corner_name in zip(SIDES, corners):
        stamp = side_stamps[side]
        patch = stamp["patch"]  # type: ignore[index]
        size = int(stamp["height"])
        rr0, rr1, cc0, cc1 = corner_regions[corner_name]
        if rr1 - rr0 + 1 < size or cc1 - cc0 + 1 < size:
            return None
        r0 = int(rng.integers(rr0, rr1 - size + 2))
        c0 = int(rng.integers(cc0, cc1 - size + 2))
        _paste_non_bg(grid, patch, r0, c0, bg)
        corner_stamps.append({"patch": patch, "r0": r0, "c0": c0})

    expected = _compose_expected(
        grid, bg, frame_color, frame_bbox, corner_stamps, side_stamps, gaps
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
