import numpy as np


AZURE = 8
MAROON = 9
FRAGMENT_COLORS = [1, 2, 3, 4, 5, 6, 7]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray, exclude: int | None = None) -> int:
    values = grid.ravel()
    if exclude is not None:
        values = values[values != exclude]
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIRS8:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, c0


def _bottom_center_col(patch: np.ndarray, bg: int) -> int:
    bottom_cols = np.flatnonzero(patch[-1] != bg)
    return int((int(bottom_cols[0]) + int(bottom_cols[-1])) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 4
    h, w = g.shape

    base_color = int(g[-1, 0])
    assert np.all(g[-1] == base_color)
    anchor_cols = [int(c) for c in np.flatnonzero(g[-2] == base_color)]
    assert anchor_cols

    bg = _most_common_color(g, exclude=base_color)
    components = []
    for color in sorted(set(map(int, g.ravel())) - {bg, base_color}):
        for comp in _find_components(g == color):
            patch, left_col = _crop_component(g, comp, bg)
            components.append(
                {
                    "area": len(comp),
                    "left": left_col,
                    "patch": patch,
                }
            )

    assert len(components) == len(anchor_cols)
    components.sort(key=lambda item: (-item["area"], item["left"]))
    anchor_cols.sort()

    out = np.full_like(g, bg)
    out[-1, :] = base_color
    out[-2, anchor_cols] = base_color

    for anchor_c, item in zip(anchor_cols, components):
        patch = np.fliplr(item["patch"])
        ph, pw = patch.shape
        center = _bottom_center_col(patch, bg)
        dst_r0 = h - ph - 2
        dst_c0 = anchor_c - center
        assert 0 <= dst_r0 and 0 <= dst_c0 and dst_c0 + pw <= w
        for rr in range(ph):
            for cc in range(pw):
                color = int(patch[rr, cc])
                if color == bg:
                    continue
                tr = dst_r0 + rr
                tc = dst_c0 + cc
                assert out[tr, tc] == bg
                out[tr, tc] = color

    return out


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _flip_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    width = max(c for _, c in coords) + 1
    return sorted((r, width - 1 - c) for r, c in coords)


def _make_patch(coords: list[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    height = max(r for r, _ in coords) + 1
    width = max(c for _, c in coords) + 1
    patch = np.full((height, width), bg, dtype=int)
    for r, c in coords:
        patch[r, c] = color
    return patch


def _random_shape(area: int, require_asymmetric: bool) -> list[tuple[int, int]] | None:
    for _ in range(200):
        coords = {(0, 0)}
        while len(coords) < area:
            border = []
            for r, c in coords:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) not in coords and -4 <= nr <= 4 and -4 <= nc <= 4:
                        border.append((nr, nc))
            if not border:
                break
            nr, nc = border[int(RNG.integers(len(border)))]
            coords.add((nr, nc))
        if len(coords) != area:
            continue
        norm = _normalize(coords)
        height = max(r for r, _ in norm) + 1
        width = max(c for _, c in norm) + 1
        if height > 4 or width > 5:
            continue
        bottom_cols = sorted(c for r, c in norm if r == height - 1)
        if not bottom_cols:
            continue
        if bottom_cols[-1] - bottom_cols[0] + 1 != len(bottom_cols):
            continue
        if len(bottom_cols) % 2 == 0:
            continue
        if area == height * width:
            continue
        if require_asymmetric and norm == _flip_coords(norm):
            continue
        return norm
    return None


def _sample_areas(count: int) -> list[int] | None:
    values = [3, 4, 5, 6]
    if count > len(values):
        return None
    if RNG.random() < 0.5:
        return list(RNG.choice(values, size=count, replace=False))
    dup_count = int(RNG.integers(2, min(3, count) + 1))
    dup_area = int(RNG.choice(values))
    remaining_pool = [value for value in values if value != dup_area]
    if count - dup_count > len(remaining_pool):
        return None
    remainder = list(RNG.choice(remaining_pool, size=count - dup_count, replace=False))
    areas = [dup_area] * dup_count + remainder
    RNG.shuffle(areas)
    return areas


def _min_output_width(widths: list[int]) -> int:
    return 2 + sum(widths) + max(0, len(widths) - 1)


def _spaced_starts(total_width: int, widths: list[int]) -> list[int] | None:
    if not widths:
        return []
    min_used = _min_output_width(widths)
    if total_width < min_used:
        return None
    extra = total_width - min_used
    slots = len(widths) + 1
    add = [0] * slots
    for _ in range(extra):
        add[int(RNG.integers(slots))] += 1
    starts = []
    cur = 1 + add[0]
    for idx, width in enumerate(widths):
        starts.append(cur)
        cur += width
        if idx < len(widths) - 1:
            cur += 1 + add[idx + 1]
    return starts


def _try_place_inputs(
    height: int,
    width: int,
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    top_limit = height - 6
    placed: list[tuple[int, int]] = []
    rects: list[tuple[int, int, int, int]] = []
    for patch in patches:
        ph, pw = patch.shape
        if top_limit - ph < 0 or width - pw < 0:
            return None
        for _ in range(200):
            r0 = int(RNG.integers(0, top_limit - ph + 1))
            c0 = int(RNG.integers(0, width - pw + 1))
            r1 = r0 + ph - 1
            c1 = c0 + pw - 1
            ok = True
            for er0, ec0, er1, ec1 in rects:
                if not (r1 < er0 - 1 or er1 < r0 - 1 or c1 < ec0 - 1 or ec1 < c0 - 1):
                    ok = False
                    break
            if ok:
                placed.append((r0, c0))
                rects.append((r0, c0, r1, c1))
                break
        else:
            return None
    return placed


def _build_grid(
    height: int,
    width: int,
    patches: list[np.ndarray],
    placements: list[tuple[int, int]],
    anchors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), AZURE, dtype=int)
    grid[-1, :] = MAROON
    grid[-2, anchors] = MAROON
    for patch, (r0, c0) in zip(patches, placements):
        ph, pw = patch.shape
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        assert np.all(region == AZURE)
        mask = patch != AZURE
        region[mask] = patch[mask]
    return grid


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.integers(2, 5))
    areas = _sample_areas(count)
    if areas is None:
        return None
    colors = list(RNG.choice(FRAGMENT_COLORS, size=count, replace=False))

    asym_index = int(RNG.integers(count))
    patches = []
    visibly_flippable = 0
    for idx, (area, color) in enumerate(zip(areas, colors)):
        require_asymmetric = idx == asym_index
        shape = _random_shape(int(area), require_asymmetric=require_asymmetric)
        if shape is None:
            return None
        patch = _make_patch(shape, int(color), AZURE)
        if not np.array_equal(patch, np.fliplr(patch)):
            visibly_flippable += 1
        patches.append(patch)
    if visibly_flippable == 0:
        return None

    max_patch_h = max(patch.shape[0] for patch in patches)
    min_width = _min_output_width([patch.shape[1] for patch in patches])
    if min_width > 22:
        return None
    width_low = max(12, min_width)
    width_high = min(23, min_width + 6)
    if width_low >= width_high:
        width_high = width_low + 1
    width = int(RNG.integers(width_low, width_high))

    height_low = max(10, max_patch_h + 7)
    height_high = min(18, height_low + 5)
    if height_low >= height_high:
        height_high = height_low + 1
    height = int(RNG.integers(height_low, height_high))

    placements = _try_place_inputs(height, width, patches)
    if placements is None:
        return None

    objects = []
    for patch, (r0, c0) in zip(patches, placements):
        objects.append(
            {
                "patch": patch,
                "area": int(np.count_nonzero(patch != AZURE)),
                "left": c0,
            }
        )
    tied_lefts = {}
    for obj in objects:
        tied_lefts.setdefault(obj["area"], []).append(obj["left"])
    if any(len(cols) != len(set(cols)) for cols in tied_lefts.values() if len(cols) > 1):
        return None

    ordered = sorted(objects, key=lambda item: (-item["area"], item["left"]))
    widths = [item["patch"].shape[1] for item in ordered]
    starts = _spaced_starts(width, widths)
    if starts is None:
        return None
    anchors = []
    for start, item in zip(starts, ordered):
        flipped = np.fliplr(item["patch"])
        center = _bottom_center_col(flipped, AZURE)
        anchor = start + center
        if not (0 <= anchor < width):
            return None
        anchors.append(anchor)
    if len(set(anchors)) != count:
        return None

    grid = _build_grid(height, width, patches, placements, anchors)
    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.all(out[-1, :] == MAROON):
        return None
    if not np.array_equal(np.flatnonzero(out[-2] == MAROON), np.array(anchors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
