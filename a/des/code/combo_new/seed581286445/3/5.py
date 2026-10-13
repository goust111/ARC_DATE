import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PAIR_COLORS = (1, 2, 3, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
PATCH_BUILD_ATTEMPTS = 120

RNG = np.random.default_rng()


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = set() if exclude is None else set(exclude)
    return [int(color) for color in np.unique(grid) if int(color) not in exclude]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each smaller same-colored rectangle across its yellow mirror marker onto the larger rectangle.

    Input structure:
    - Black background.
    - Each non-background, non-yellow object color forms exactly two solid filled rectangles with unequal areas.
    - The two rectangles of a color are arranged either left-right with overlapping row span or top-bottom with overlapping column span.
    - A single yellow pixel lies in the empty gap between the two rectangles of each color and marks the mirror axis.

    Rule:
    - For every rectangle color, locate its two solid rectangles and identify the yellow marker sitting between them.
    - Use the marker column as a vertical mirror axis for left-right pairs, or the marker row as a horizontal mirror axis for top-bottom pairs.
    - Erase the smaller rectangle and erase the yellow marker.
    - Mirror every pixel of the smaller rectangle across that axis: cells landing on the larger rectangle's original area become grey, while cells landing on background keep the pair's original color.

    Key insight:
    - The yellow pixel is a control marker for the reflection axis, and only the smaller rectangle moves; the larger one stays in place except where overlap is recolored grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    unused_markers = {tuple(map(int, pos)) for pos in np.argwhere(g == YELLOW)}
    pairs: list[dict[str, object]] = []

    for color in _colors_present(g, exclude={BG, YELLOW}):
        comps = _components(g == color)
        assert len(comps) == 2, f"expected exactly two rectangles for color {color}"
        rects = []
        for coords in comps:
            assert _is_solid_rectangle(coords), f"color {color} must form solid rectangles"
            r0, c0, r1, c1 = _bbox(coords)
            rects.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )

        rects.sort(key=lambda item: int(item["area"]))
        small = rects[0]
        big = rects[1]
        assert int(small["area"]) < int(big["area"])

        sr0, sc0, sr1, sc1 = small["bbox"]  # type: ignore[index]
        br0, bc0, br1, bc1 = big["bbox"]  # type: ignore[index]
        row_overlap = _overlap_len(sr0, sr1, br0, br1)
        col_overlap = _overlap_len(sc0, sc1, bc0, bc1)
        cols_disjoint = sc1 < bc0 or bc1 < sc0
        rows_disjoint = sr1 < br0 or br1 < sr0

        horizontal = row_overlap > 0 and cols_disjoint
        vertical = col_overlap > 0 and rows_disjoint
        assert horizontal != vertical, f"ambiguous arrangement for color {color}"

        if horizontal:
            left, right = sorted(rects, key=lambda item: int(item["bbox"][1]))  # type: ignore[index]
            lr0, lc0, lr1, lc1 = left["bbox"]  # type: ignore[index]
            rr0, rc0, rr1, rc1 = right["bbox"]  # type: ignore[index]
            row0 = max(lr0, rr0)
            row1 = min(lr1, rr1)
            candidates = [
                marker
                for marker in unused_markers
                if row0 <= marker[0] <= row1 and lc1 < marker[1] < rc0
            ]
            assert len(candidates) == 1, f"expected one yellow axis marker for color {color}"
            marker = candidates[0]
            axis = marker[1]
        else:
            top, bottom = sorted(rects, key=lambda item: int(item["bbox"][0]))  # type: ignore[index]
            tr0, tc0, tr1, tc1 = top["bbox"]  # type: ignore[index]
            br0, bc0, br1, bc1 = bottom["bbox"]  # type: ignore[index]
            col0 = max(tc0, bc0)
            col1 = min(tc1, bc1)
            candidates = [
                marker
                for marker in unused_markers
                if col0 <= marker[1] <= col1 and tr1 < marker[0] < br0
            ]
            assert len(candidates) == 1, f"expected one yellow axis marker for color {color}"
            marker = candidates[0]
            axis = marker[0]

        unused_markers.remove(marker)
        pairs.append(
            {
                "color": color,
                "small_coords": list(small["coords"]),
                "big_coords": set(big["coords"]),
                "marker": marker,
                "axis": axis,
                "horizontal": horizontal,
            }
        )

    assert not unused_markers, "every yellow marker must belong to a rectangle pair"

    for pair in pairs:
        color = int(pair["color"])
        small_coords = pair["small_coords"]  # type: ignore[assignment]
        big_coords = pair["big_coords"]  # type: ignore[assignment]
        marker = pair["marker"]  # type: ignore[assignment]
        axis = int(pair["axis"])
        horizontal = bool(pair["horizontal"])

        out[marker] = BG
        for r, c in small_coords:
            out[r, c] = BG

        for r, c in small_coords:
            rr, cc = (r, 2 * axis - c) if horizontal else (2 * axis - r, c)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            if (rr, cc) in big_coords:
                out[rr, cc] = GREY
            else:
                assert g[rr, cc] == BG
                out[rr, cc] = color

    return out


def _build_horizontal_patch(rng: np.random.Generator, color: int, mode: str) -> np.ndarray:
    for _ in range(PATCH_BUILD_ATTEMPTS):
        small_h = int(rng.integers(2, 5))
        small_w = int(rng.integers(2, 6))
        gap = int(rng.integers(1, 3))

        if mode == "full":
            big_h = small_h + int(rng.integers(0, 3))
            big_w = small_w + int(rng.integers(0, 4))
            if big_h == small_h and big_w == small_w:
                if rng.random() < 0.5:
                    big_h += 1
                else:
                    big_w += 1
            spill = 0
        else:
            big_h = small_h + int(rng.integers(1, 3))
            spill = int(rng.integers(1, small_w))
            big_w = int(rng.integers(max(1, small_w - spill), small_w + 3))
            while big_h * big_w <= small_h * small_w and big_w < small_w + 3:
                big_w += 1
            while big_h * big_w <= small_h * small_w and big_h < small_h + 3:
                big_h += 1
            if big_h * big_w <= small_h * small_w:
                continue

        if big_h > 7 or big_w > 8:
            continue

        top_margin = 1
        left_margin = 1
        big_r0 = top_margin
        small_r0 = top_margin + (big_h - small_h) // 2
        small_c0 = left_margin
        small_c1 = small_c0 + small_w - 1
        axis_c = small_c1 + gap
        mirrored_c0 = axis_c + gap

        if mode == "full":
            shift_left_max = min(gap - 1, max(0, big_w - small_w))
            shift_left = int(rng.integers(0, shift_left_max + 1))
            big_c0 = mirrored_c0 - shift_left
        else:
            big_c0 = mirrored_c0 + spill

        big_c1 = big_c0 + big_w - 1
        patch_h = max(big_r0 + big_h, small_r0 + small_h) + 1
        patch_w = big_c1 + 2
        if patch_h > 12 or patch_w > 18:
            continue

        patch = np.full((patch_h, patch_w), BG, dtype=int)
        patch[small_r0 : small_r0 + small_h, small_c0 : small_c0 + small_w] = color
        patch[big_r0 : big_r0 + big_h, big_c0 : big_c0 + big_w] = color

        marker_r = int(rng.integers(small_r0, small_r0 + small_h))
        patch[marker_r, axis_c] = YELLOW
        return patch

    raise RuntimeError("failed to build pair patch")


def _build_pair_patch(rng: np.random.Generator, color: int, orientation: str, mode: str) -> np.ndarray:
    patch = _build_horizontal_patch(rng, color, mode)
    if orientation == "horizontal":
        if rng.random() < 0.5:
            patch = np.fliplr(patch).copy()
    else:
        patch = patch.T.copy()
        if rng.random() < 0.5:
            patch = np.flipud(patch).copy()

    pad_top = int(rng.integers(0, 3))
    pad_bottom = int(rng.integers(0, 3))
    pad_left = int(rng.integers(0, 3))
    pad_right = int(rng.integers(0, 3))
    return np.pad(
        patch,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=BG,
    )


def _partition_counts(n: int, rng: np.random.Generator) -> list[int]:
    if n == 2:
        return [2] if rng.random() < 0.5 else [1, 1]
    options = ([3], [1, 2], [2, 1], [1, 1, 1])
    return list(options[int(rng.integers(0, len(options)))])


def _pack_patches(rng: np.random.Generator, patches: list[np.ndarray]) -> np.ndarray | None:
    order = list(patches)
    rng.shuffle(order)
    row_counts = _partition_counts(len(order), rng)
    rows: list[list[np.ndarray]] = []
    cursor = 0
    for count in row_counts:
        rows.append(order[cursor : cursor + count])
        cursor += count

    row_gaps: list[list[int]] = []
    row_widths: list[int] = []
    row_heights: list[int] = []
    for row in rows:
        gaps = [int(rng.integers(1, 5)) for _ in range(max(0, len(row) - 1))]
        row_gaps.append(gaps)
        row_heights.append(max(int(patch.shape[0]) for patch in row))
        row_widths.append(sum(int(patch.shape[1]) for patch in row) + sum(gaps))

    vertical_gaps = [int(rng.integers(1, 5)) for _ in range(max(0, len(rows) - 1))]
    margin_top = int(rng.integers(0, 4))
    margin_bottom = int(rng.integers(0, 4))
    margin_left = int(rng.integers(0, 4))
    margin_right = int(rng.integers(0, 4))

    min_h = margin_top + margin_bottom + sum(row_heights) + sum(vertical_gaps)
    min_w = margin_left + margin_right + max(row_widths)
    if min_h > 30 or min_w > 30:
        return None

    h = min_h + int(rng.integers(0, 31 - min_h))
    w = min_w + int(rng.integers(0, 31 - min_w))
    grid = np.full((h, w), BG, dtype=int)

    row_start_r = margin_top
    for row_index, (row, gaps, row_h, row_w) in enumerate(zip(rows, row_gaps, row_heights, row_widths, strict=False)):
        row_start_c = margin_left + int(rng.integers(0, w - margin_left - margin_right - row_w + 1))
        cursor_c = row_start_c
        for patch_index, patch in enumerate(row):
            ph, pw = patch.shape
            offset_r = int(rng.integers(0, row_h - ph + 1))
            mask = patch != BG
            target = grid[row_start_r + offset_r : row_start_r + offset_r + ph, cursor_c : cursor_c + pw]
            if np.any(target[mask] != BG):
                return None
            target[mask] = patch[mask]
            grid[row_start_r + offset_r : row_start_r + offset_r + ph, cursor_c : cursor_c + pw] = target
            if patch_index < len(gaps):
                cursor_c += pw + gaps[patch_index]
        row_start_r += row_h
        if row_index < len(vertical_gaps):
            row_start_r += vertical_gaps[row_index]

    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.choice((2, 3)))
    colors = list(RNG.choice(PAIR_COLORS, size=pair_count, replace=False))

    orientations = ["horizontal", "vertical"]
    while len(orientations) < pair_count:
        orientations.append("horizontal" if RNG.random() < 0.5 else "vertical")
    RNG.shuffle(orientations)

    modes = ["full", "partial"]
    while len(modes) < pair_count:
        modes.append("full" if RNG.random() < 0.5 else "partial")
    RNG.shuffle(modes)

    patches: list[np.ndarray] = []
    for color, orientation, mode in zip(colors, orientations, modes, strict=False):
        patch = _build_pair_patch(RNG, int(color), orientation, mode)
        patches.append(patch)

    grid = _pack_patches(RNG, patches)
    if grid is None:
        return None

    if max(grid.shape) > 30:
        return None

    yellow_count = int(np.count_nonzero(grid == YELLOW))
    if yellow_count != pair_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(out == GREY):
        return None
    if not np.any((grid == BG) & (out != BG) & (out != GREY)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
