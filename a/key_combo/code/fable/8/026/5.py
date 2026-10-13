from __future__ import annotations

from collections import Counter

import numpy as np


FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | tuple[int, ...] | None = None,
    min_area: int = 1,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)
    allowed = None if colors is None else {int(c) for c in colors}
    if int(connectivity) != 4:
        raise ValueError("This puzzle only needs 4-connectivity.")

    seen = np.zeros(g.shape, dtype=bool)
    out: list[dict] = []
    h, w = g.shape

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            color0 = int(g[r0, c0])
            if color0 == bg:
                continue
            if allowed is not None and color0 not in allowed and monochromatic:
                continue
            if allowed is not None and not monochromatic and int(g[r0, c0]) not in allowed:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in FOUR_NEIGHBORS:
                    rr = int(r) + dr
                    cc = int(c) + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc]:
                        continue
                    value = int(g[rr, cc])
                    if value == bg:
                        continue
                    if allowed is not None and value not in allowed:
                        continue
                    if monochromatic and value != color0:
                        continue
                    seen[rr, cc] = True
                    stack.append((rr, cc))

            if len(coords) < int(min_area):
                continue
            coords_arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color0 if monochromatic else None,
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(len(coords)),
                }
            )
    return out


def _sorted_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _mod_delta(src: tuple[int, int], dst: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    return (int(dst[0] - src[0]) % h, int(dst[1] - src[1]) % w)


def _unique_modular_shift(
    src_markers: tuple[tuple[int, int], tuple[int, int]],
    dst_markers: tuple[tuple[int, int], tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for perm in ((0, 1), (1, 0)):
        d0 = _mod_delta(src_markers[0], dst_markers[perm[0]], shape)
        d1 = _mod_delta(src_markers[1], dst_markers[perm[1]], shape)
        if d0 == d1:
            candidates.append((d0, perm))
    assert len(candidates) == 1, "marker pairs must define exactly one modular shift"
    return candidates[0]


def _has_plain_alignment(
    src_markers: tuple[tuple[int, int], tuple[int, int]],
    dst_markers: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    for perm in ((0, 1), (1, 0)):
        dr0 = int(dst_markers[perm[0]][0] - src_markers[0][0])
        dc0 = int(dst_markers[perm[0]][1] - src_markers[0][1])
        dr1 = int(dst_markers[perm[1]][0] - src_markers[1][0])
        dc1 = int(dst_markers[perm[1]][1] - src_markers[1][1])
        if (dr0, dc0) == (dr1, dc1):
            return True
    return False


def _would_wrap(coord: tuple[int, int], delta: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = map(int, coord)
    dr, dc = map(int, delta)
    h, w = map(int, shape)
    return r + dr >= h or c + dc >= w


def _wrap_signature(
    markers: tuple[tuple[int, int], tuple[int, int]],
    delta: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[bool, bool]:
    h, w = map(int, shape)
    dr, dc = map(int, delta)
    row_wraps = [int(r) + dr >= h for r, _c in markers]
    col_wraps = [int(c) + dc >= w for _r, c in markers]
    return row_wraps[0] != row_wraps[1], col_wraps[0] != col_wraps[1]


def _render_modular_translation(
    src_region: np.ndarray,
    src_fill: int,
    delta: tuple[int, int],
    out_fill: int,
) -> np.ndarray:
    h, w = src_region.shape
    dr, dc = map(int, delta)
    out = np.full((h, w), int(out_fill), dtype=int)
    for r, c in np.argwhere(src_region != int(src_fill)):
        rr = (int(r) + dr) % h
        cc = (int(c) + dc) % w
        out[rr, cc] = int(src_region[r, c])
    return out


def _render_plain_translation(
    src_region: np.ndarray,
    src_fill: int,
    shift: tuple[int, int],
    out_shape: tuple[int, int],
    out_fill: int,
) -> np.ndarray:
    out = np.full(out_shape, int(out_fill), dtype=int)
    dr, dc = map(int, shift)
    h, w = out_shape
    for r, c in np.argwhere(src_region != int(src_fill)):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = int(src_region[r, c])
    return out


def _extract_task(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    windows = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(windows) == 2, "expected exactly two windows"

    items = []
    for comp in windows:
        region = crop_bbox(g, comp["bbox"])
        assert int(comp["area"]) == int(region.size), "windows must be solid rectangles"
        fill = int(Counter(map(int, region.ravel())).most_common(1)[0][0])
        items.append({"bbox": comp["bbox"], "region": region, "fill": fill})

    assert items[0]["region"].shape == items[1]["region"].shape, "windows must be equal-sized"
    shape = tuple(map(int, items[0]["region"].shape))

    shared = set(map(int, np.unique(items[0]["region"]))) & set(map(int, np.unique(items[1]["region"])))
    marker_candidates = [
        color
        for color in sorted(shared)
        if color not in {items[0]["fill"], items[1]["fill"]}
        and np.count_nonzero(items[0]["region"] == color) == 2
        and np.count_nonzero(items[1]["region"] == color) == 2
    ]
    assert len(marker_candidates) == 1, "expected one shared marker color"
    marker = int(marker_candidates[0])

    parsed = []
    for item in items:
        region = item["region"]
        fill = int(item["fill"])
        markers = _sorted_coords(np.argwhere(region == marker))
        assert len(markers) == 2
        extra = int(np.count_nonzero((region != fill) & (region != marker)))
        parsed.append(
            {
                "bbox": item["bbox"],
                "region": region,
                "fill": fill,
                "markers": markers,
                "extra": extra,
            }
        )

    sources = [item for item in parsed if item["extra"] > 0]
    destinations = [item for item in parsed if item["extra"] == 0 and np.count_nonzero(item["region"] != item["fill"]) == 2]
    assert len(sources) == 1 and len(destinations) == 1, "expected one sprite window and one empty window"

    src = sources[0]
    dst = destinations[0]
    delta, perm = _unique_modular_shift(src["markers"], dst["markers"], shape)

    return {
        "bg": bg,
        "shape": shape,
        "marker": marker,
        "src_region": src["region"],
        "src_fill": int(src["fill"]),
        "src_markers": src["markers"],
        "dst_fill": int(dst["fill"]),
        "dst_markers": dst["markers"],
        "delta": delta,
        "perm": perm,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Align the marked sprite window to the marked empty window by the unique toroidal shift.

    Input structure:
    - The dominant color is an outer background containing exactly two solid rectangular windows.
    - The windows have the same size but different fill colors; one contains a multicolor sprite and
      exactly two shared-color marker pixels, while the other contains only two marker pixels.
    - The marker pairs disagree under ordinary translation but admit one unique translation modulo the
      window height and width.

    Rule:
    - Detect the source sprite window, the empty destination window, and the shared marker color.
    - Find the unique shift on the window torus that maps both source markers onto the destination
      markers simultaneously.
    - Output a grid with the destination window size and destination fill color, then move every
      non-fill pixel from the source window by that shift, wrapping rows and columns modulo the
      output shape.

    Key insight:
    - The contradictory marker offsets are the clue that the window is not planar: both marker
      correspondences become consistent only after interpreting the window as a torus.
    """
    task = _extract_task(grid)
    return _render_modular_translation(
        task["src_region"],
        task["src_fill"],
        task["delta"],
        task["dst_fill"],
    )


def _random_path(
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    r, c = map(int, start)
    tr, tc = map(int, end)
    steps: list[tuple[int, int]] = []
    dr = 1 if tr > r else -1
    dc = 1 if tc > c else -1
    steps.extend([(dr, 0)] * abs(tr - r))
    steps.extend([(0, dc)] * abs(tc - c))
    if steps:
        order = rng.permutation(len(steps))
        steps = [steps[int(i)] for i in order]
    path = [(r, c)]
    for step_r, step_c in steps:
        r += int(step_r)
        c += int(step_c)
        path.append((int(r), int(c)))
    return path


def _mask_frontier(mask: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    frontier: set[tuple[int, int]] = set()
    for r, c in mask:
        for dr, dc in FOUR_NEIGHBORS:
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in mask:
                frontier.add((rr, cc))
    return sorted(frontier)


def _build_sprite_mask(
    shape: tuple[int, int],
    markers: tuple[tuple[int, int], tuple[int, int]],
    delta: tuple[int, int],
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    h, w = map(int, shape)
    path = _random_path(markers[0], markers[1], rng)
    mask = set(path)
    if len(mask) > h * w - 4:
        return None

    target_lo = max(len(mask) + 2, 6)
    target_hi = min(h * w - 4, max(target_lo, int(round(0.45 * h * w))))
    if target_lo > target_hi:
        return None
    target = int(rng.integers(target_lo, target_hi + 1))

    while len(mask) < target:
        frontier = _mask_frontier(mask, shape)
        if not frontier:
            return None
        need_wrapped_nonmarker = not any(
            coord not in markers and _would_wrap(coord, delta, shape) for coord in mask
        )
        wrapped_frontier = [coord for coord in frontier if _would_wrap(coord, delta, shape)]
        pool = wrapped_frontier if need_wrapped_nonmarker and wrapped_frontier else frontier
        mask.add(pool[int(rng.integers(len(pool)))])

    if not any(coord not in markers and _would_wrap(coord, delta, shape) for coord in mask):
        frontier = _mask_frontier(mask, shape)
        wrapped_frontier = [coord for coord in frontier if _would_wrap(coord, delta, shape)]
        if not wrapped_frontier or len(mask) >= h * w - 4:
            return None
        mask.add(wrapped_frontier[int(rng.integers(len(wrapped_frontier)))])

    return mask


def _paint_sprite_region(
    shape: tuple[int, int],
    mask: set[tuple[int, int]],
    markers: tuple[tuple[int, int], tuple[int, int]],
    src_fill: int,
    marker: int,
    sprite_palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = map(int, shape)
    region = np.full((h, w), int(src_fill), dtype=int)
    nonmarkers = [coord for coord in sorted(mask) if coord not in set(markers)]
    palette_size = min(len(sprite_palette), 2 + int(len(nonmarkers) >= 6))
    colors = [int(c) for c in rng.choice(sprite_palette, size=palette_size, replace=False)]
    mode = str(rng.choice(["checker", "rows", "cols", "bands", "noise"]))

    for idx, (r, c) in enumerate(nonmarkers):
        if mode == "checker":
            color = colors[(int(r) + int(c)) % len(colors)]
        elif mode == "rows":
            color = colors[int(r) % len(colors)]
        elif mode == "cols":
            color = colors[int(c) % len(colors)]
        elif mode == "bands":
            color = colors[(2 * int(r) + int(c)) % len(colors)]
        else:
            color = colors[int(rng.integers(len(colors)))]
        region[int(r), int(c)] = int(color)

    used = set(int(region[r, c]) for r, c in nonmarkers)
    missing = [color for color in colors if color not in used]
    for color, coord in zip(missing, nonmarkers):
        region[coord] = int(color)

    for r, c in markers:
        region[int(r), int(c)] = int(marker)

    return region


def _sample_marker_configuration(
    shape: tuple[int, int],
    wrap_mode: str,
    target_perm: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]], tuple[int, int]] | None:
    h, w = map(int, shape)

    for _ in range(800):
        if wrap_mode == "row":
            dr = int(rng.integers(1, h))
            dc = 0 if rng.random() < 0.7 else int(rng.integers(0, w))
        elif wrap_mode == "col":
            dr = 0 if rng.random() < 0.7 else int(rng.integers(0, h))
            dc = int(rng.integers(1, w))
        else:
            dr = int(rng.integers(1, h))
            dc = int(rng.integers(1, w))

        if (dr % h, dc % w) == (0, 0):
            continue

        picks = rng.choice(h * w, size=2, replace=False)
        src = _sorted_coords([(int(k // w), int(k % w)) for k in picks])
        if abs(src[0][0] - src[1][0]) + abs(src[0][1] - src[1][1]) < 2:
            continue

        dst_unsorted = [((r + dr) % h, (c + dc) % w) for r, c in src]
        dst = _sorted_coords(dst_unsorted)

        try:
            delta, perm = _unique_modular_shift(src, dst, shape)
        except AssertionError:
            continue

        if perm != target_perm:
            continue
        if delta != (dr % h, dc % w):
            continue
        if _has_plain_alignment(src, dst):
            continue

        row_mixed, col_mixed = _wrap_signature(src, delta, shape)
        if wrap_mode == "row" and not (row_mixed and not col_mixed):
            continue
        if wrap_mode == "col" and not (col_mixed and not row_mixed):
            continue
        if wrap_mode == "both" and not (row_mixed and col_mixed):
            continue

        return src, dst, delta

    return None


def _sample_window_positions(
    grid_shape: tuple[int, int],
    window_shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    gh, gw = map(int, grid_shape)
    h, w = map(int, window_shape)
    if gh < h + 4 or gw < w + 4:
        return None

    for _ in range(400):
        top1 = int(rng.integers(1, gh - h))
        left1 = int(rng.integers(1, gw - w))
        top2 = int(rng.integers(1, gh - h))
        left2 = int(rng.integers(1, gw - w))

        rows_separate = top1 + h + 1 < top2 or top2 + h + 1 < top1
        cols_separate = left1 + w + 1 < left2 or left2 + w + 1 < left1
        if rows_separate or cols_separate:
            return (top1, left1), (top2, left2)
    return None


def _validate_generated_case(grid: np.ndarray) -> bool:
    task = _extract_task(grid)
    src_region = task["src_region"]
    src_fill = int(task["src_fill"])
    dst_fill = int(task["dst_fill"])
    delta = task["delta"]
    shape = task["shape"]

    assert not _has_plain_alignment(task["src_markers"], task["dst_markers"])
    assert any(
        coord not in task["src_markers"] and _would_wrap(coord, delta, shape)
        for coord in _sorted_coords(np.argwhere(src_region != src_fill))
    )

    out = main(grid)
    oracle = _render_modular_translation(src_region, src_fill, delta, dst_fill)
    assert np.array_equal(out, oracle)

    plain_renders = []
    for src_marker in task["src_markers"]:
        for dst_marker in task["dst_markers"]:
            shift = (
                int(dst_marker[0] - src_marker[0]),
                int(dst_marker[1] - src_marker[1]),
            )
            plain_renders.append(_render_plain_translation(src_region, src_fill, shift, shape, dst_fill))
    assert all(not np.array_equal(out, candidate) for candidate in plain_renders)
    return True


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    wrap_mode = str(rng.choice(["row", "col", "both"]))
    target_perm = (0, 1) if rng.random() < 0.5 else (1, 0)

    h = int(rng.integers(5, 10))
    w = int(rng.integers(5, 10))
    shape = (h, w)
    marker_case = _sample_marker_configuration(shape, wrap_mode, target_perm, rng)
    if marker_case is None:
        return None
    src_markers, dst_markers, delta = marker_case

    palette = list(map(int, rng.permutation(np.arange(10))))
    bg, src_fill, dst_fill, marker = palette[:4]
    sprite_palette = palette[4:8]

    mask = _build_sprite_mask(shape, src_markers, delta, rng)
    if mask is None or len(mask) - 2 < 4:
        return None

    src_region = _paint_sprite_region(shape, mask, src_markers, src_fill, marker, sprite_palette, rng)
    dst_region = np.full(shape, int(dst_fill), dtype=int)
    for r, c in dst_markers:
        dst_region[int(r), int(c)] = int(marker)

    gh = int(rng.integers(max(16, h + 6), 31))
    gw = int(rng.integers(max(16, w + 6), 31))
    positions = _sample_window_positions((gh, gw), shape, rng)
    if positions is None:
        return None
    src_pos, dst_pos = positions if rng.random() < 0.5 else positions[::-1]

    grid = np.full((gh, gw), int(bg), dtype=int)
    sr, sc = src_pos
    dr, dc = dst_pos
    grid[sr : sr + h, sc : sc + w] = src_region
    grid[dr : dr + h, dc : dc + w] = dst_region

    try:
        _validate_generated_case(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid modular-wrap marker-alignment task")
