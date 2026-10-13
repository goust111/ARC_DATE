from __future__ import annotations

from itertools import combinations
import numpy as np


BG = 0
GRAY = 5
ROTATION_COLOR_TO_K = {
    2: 0,  # red
    3: 1,  # green
    1: 2,  # blue
    4: 3,  # yellow
}
ROTATION_K_TO_COLOR = {v: k for k, v in ROTATION_COLOR_TO_K.items()}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    return _bbox_from_coords(pts)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _find_material_frame(top: np.ndarray) -> tuple[int, int, int, int]:
    gray_comps = _components(top == GRAY)
    candidates: list[tuple[int, int, int, int]] = []
    for comp in gray_comps:
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        sub = top[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(sub == GRAY, border):
            candidates.append((r0, c0, r1, c1))
    assert candidates, "gray material frame not found"
    return min(candidates, key=lambda b: (b[1], -(b[2] - b[0] + 1) * (b[3] - b[1] + 1)))


def _parse_palette(grid: np.ndarray, bg: int) -> dict[int, int]:
    row = grid[-1]
    cols = np.where(row != bg)[0]
    assert cols.size >= 2 and cols.size % 2 == 0, "palette must occupy an even-length run"
    start = int(cols[0])
    end = int(cols[-1])
    assert end - start + 1 == cols.size, "palette must be contiguous"
    run = row[start : end + 1]
    mapping: dict[int, int] = {}
    for i in range(0, len(run), 2):
        mold_color = int(run[i])
        rot_color = int(run[i + 1])
        assert rot_color in ROTATION_COLOR_TO_K, "unknown rotation color"
        mapping[mold_color] = int(ROTATION_COLOR_TO_K[rot_color])
    assert mapping, "palette mapping must be non-empty"
    return mapping


def _find_unique_source_patch(material: np.ndarray, hole_mask: np.ndarray, bg: int) -> np.ndarray:
    h, w = hole_mask.shape
    matches: list[np.ndarray] = []
    for r in range(material.shape[0] - h + 1):
        for c in range(material.shape[1] - w + 1):
            window = material[r : r + h, c : c + w]
            if np.array_equal(window != bg, hole_mask):
                matches.append(window.copy())
    assert len(matches) == 1, "hole shape must match exactly one source patch"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each mold hole to an unrotated source patch, then rotate the patch according
    to the bottom palette before filling the hole. Return only the filled mold region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 6

    bg = _most_common_color(g)
    rot_by_color = _parse_palette(g, bg)
    top = g[:-1].copy()

    fr0, fc0, fr1, fc1 = _find_material_frame(top)
    material = top[fr0 + 1 : fr1, fc0 + 1 : fc1]

    work = top.copy()
    work[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg

    molds: list[dict] = []
    for mold_color, rot_k in rot_by_color.items():
        comps = _components(work == int(mold_color))
        assert len(comps) == 1, "each mold color must define one connected mold"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        sub = work[r0 : r1 + 1, c0 : c1 + 1]
        hole_area = sub == bg
        assert hole_area.any(), "mold must contain a hole"
        hole_comps = _components(hole_area)
        assert len(hole_comps) == 1, "hole must be a single component"
        hole_coords = hole_comps[0]
        hr0, hc0, hr1, hc1 = _bbox_from_coords(hole_coords)
        hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        hole_mask[hole_coords[:, 0] - hr0, hole_coords[:, 1] - hc0] = True
        assert bool(np.all(sub[~hole_area] == int(mold_color))), "mold body must be monochrome"
        source_patch = _find_unique_source_patch(material, hole_mask, bg)
        rotated = np.rot90(source_patch, int(rot_k))
        assert np.array_equal(rotated != bg, hole_mask), "rotated patch must fit its hole"
        filled = top[r0 : r1 + 1, c0 : c1 + 1].copy()
        target = filled[hr0 : hr1 + 1, hc0 : hc1 + 1]
        target[hole_mask] = rotated[hole_mask]
        top[r0 : r1 + 1, c0 : c1 + 1] = filled
        molds.append({"bbox": (r0, c0, r1, c1)})

    r0 = min(int(m["bbox"][0]) for m in molds)
    c0 = min(int(m["bbox"][1]) for m in molds)
    r1 = max(int(m["bbox"][2]) for m in molds)
    c1 = max(int(m["bbox"][3]) for m in molds)
    return top[r0 : r1 + 1, c0 : c1 + 1]


def _orbit(cell: tuple[int, int], size: int) -> tuple[tuple[int, int], ...]:
    r, c = cell
    pts = []
    for _ in range(4):
        pts.append((int(r), int(c)))
        r, c = c, size - 1 - r
    return tuple(sorted(set(pts)))


def _orbit_touches(mask: np.ndarray, orb: tuple[tuple[int, int], ...]) -> bool:
    h, w = mask.shape
    for r, c in orb:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _hole_compatible(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for comp in _components(~mask):
        if not any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp):
            return False
    return True


def _build_mask_library(size: int) -> list[np.ndarray]:
    c = size // 2
    base = np.zeros((size, size), dtype=bool)
    base[c, :] = True
    base[:, c] = True

    orbits: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(size):
        for cc in range(size):
            if base[r, cc]:
                continue
            orb = _orbit((r, cc), size)
            if orb not in seen:
                seen.add(orb)
                orbits.append(orb)

    found: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    max_extra = min(len(orbits), 4 if size >= 7 else 3)
    for extra_count in range(max_extra + 1):
        for subset in combinations(orbits, extra_count):
            mask = base.copy()
            for orb in subset:
                for r, c2 in orb:
                    mask[r, c2] = True
            if mask.all():
                continue
            if not _is_connected(mask):
                continue
            if not _hole_compatible(mask):
                continue
            key = _shape_key(mask)
            found[key] = mask

    return sorted(found.values(), key=lambda m: (int(m.sum()), _shape_key(m)[1]))


MASK_LIBRARY = {
    size: _build_mask_library(size) for size in (3, 5, 7)
}


def _sample_symmetric_mask(
    rng: np.random.Generator, used: set[tuple[tuple[int, int], bytes]], size: int
) -> np.ndarray | None:
    choices = [mask for mask in MASK_LIBRARY[size] if _shape_key(mask) not in used]
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))].copy()


def _sample_colored_patch(
    rng: np.random.Generator, mask: np.ndarray, required_rot_k: int, forbidden_colors: set[int]
) -> np.ndarray | None:
    palette = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in forbidden_colors]
    cells = np.argwhere(mask)
    for _ in range(300):
        n_colors = int(rng.integers(2, 5))
        if n_colors > len(palette):
            n_colors = len(palette)
        colors = rng.choice(palette, size=n_colors, replace=False)
        patch = np.zeros(mask.shape, dtype=int)
        picks = rng.integers(0, n_colors, size=len(cells))
        for idx, (r, c) in enumerate(cells):
            patch[int(r), int(c)] = int(colors[int(picks[idx])])
        if len(set(map(int, patch[mask].tolist()))) < 2:
            continue
        if required_rot_k != 0 and np.array_equal(patch, np.rot90(patch, int(required_rot_k))):
            continue
        return patch
    return None


def _place_rectangles(
    rng: np.random.Generator,
    sizes: list[tuple[int, int]],
    gap: int,
    extra_span: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    sum_h = sum(h for h, _ in sizes)
    sum_w = sum(w for _, w in sizes)
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    min_h = max(max_h, sum_h // 2) + gap * (len(sizes) - 1)
    min_w = max(max_w, sum_w // 2) + gap * (len(sizes) - 1)
    max_h_lim = min_h + extra_span
    max_w_lim = min_w + extra_span

    for _ in range(200):
        canvas_h = int(rng.integers(min_h, max_h_lim + 1))
        canvas_w = int(rng.integers(min_w, max_w_lim + 1))
        occ = np.zeros((canvas_h, canvas_w), dtype=bool)
        pos: list[tuple[int, int] | None] = [None] * len(sizes)
        order = list(rng.permutation(len(sizes)))
        ok = True
        for idx in order:
            h, w = sizes[idx]
            candidates: list[tuple[int, int]] = []
            for r in range(canvas_h - h + 1):
                for c in range(canvas_w - w + 1):
                    rr0 = max(0, r - gap)
                    cc0 = max(0, c - gap)
                    rr1 = min(canvas_h, r + h + gap)
                    cc1 = min(canvas_w, c + w + gap)
                    if not occ[rr0:rr1, cc0:cc1].any():
                        candidates.append((int(r), int(c)))
            if not candidates:
                ok = False
                break
            choice = candidates[int(rng.integers(len(candidates)))]
            pos[idx] = choice
            r, c = choice
            occ[r : r + h, c : c + w] = True
        if ok:
            return (canvas_h, canvas_w), [(int(r), int(c)) for r, c in pos if r is not None]
    return None


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox(grid != bg)
    assert box is not None
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _build_material(
    rng: np.random.Generator, patches: list[np.ndarray]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    sizes = [tuple(map(int, p.shape)) for p in patches]
    placed = _place_rectangles(rng, sizes, gap=1, extra_span=8)
    assert placed is not None
    (h, w), pos = placed
    material = np.full((h, w), BG, dtype=int)
    for patch, (r, c) in zip(patches, pos, strict=False):
        ph, pw = patch.shape
        material[r : r + ph, c : c + pw][patch != BG] = patch[patch != BG]
    material = _crop_non_bg(material, BG)

    pad_t = int(rng.integers(1, 3))
    pad_b = int(rng.integers(1, 3))
    pad_l = int(rng.integers(1, 3))
    pad_r = int(rng.integers(1, 3))
    framed = np.full((material.shape[0] + pad_t + pad_b, material.shape[1] + pad_l + pad_r), BG, dtype=int)
    framed[pad_t : pad_t + material.shape[0], pad_l : pad_l + material.shape[1]] = material
    return framed, pos


def _build_mold_patch(
    rng: np.random.Generator, hole_mask: np.ndarray, mold_color: int
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = hole_mask.shape
    pad_t = int(rng.integers(1, 3))
    pad_b = int(rng.integers(1, 3))
    pad_l = int(rng.integers(1, 3))
    pad_r = int(rng.integers(1, 3))
    out = np.full((h + pad_t + pad_b, w + pad_l + pad_r), int(mold_color), dtype=int)
    out[pad_t : pad_t + h, pad_l : pad_l + w][hole_mask] = BG
    return out, (pad_t, pad_l)


def _count_shape_matches(material: np.ndarray, mask: np.ndarray) -> int:
    count = 0
    h, w = mask.shape
    for r in range(material.shape[0] - h + 1):
        for c in range(material.shape[1] - w + 1):
            if np.array_equal(material[r : r + h, c : c + w] != BG, mask):
                count += 1
    return count


def _generate_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_molds = int(rng.integers(2, 5))
    mold_colors = list(map(int, rng.choice([6, 7, 8, 9], size=n_molds, replace=False)))
    rot_ks = list(map(int, rng.choice([0, 1, 2, 3], size=n_molds, replace=False)))
    valid_sizes = [size for size, masks in MASK_LIBRARY.items() if len(masks) >= n_molds]
    shape_size = int(rng.choice(valid_sizes))

    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    hole_masks: list[np.ndarray] = []
    for _ in range(n_molds):
        mask = _sample_symmetric_mask(rng, used_shapes, shape_size)
        if mask is None:
            return None
        used_shapes.add(_shape_key(mask))
        hole_masks.append(mask)

    source_patches: list[np.ndarray] = []
    for hole_mask, rot_k in zip(hole_masks, rot_ks, strict=False):
        patch = _sample_colored_patch(rng, hole_mask, rot_k, set(mold_colors))
        if patch is None:
            return None
        source_patches.append(patch)

    material, _ = _build_material(rng, source_patches)
    for hole_mask in hole_masks:
        if _count_shape_matches(material, hole_mask) != 1:
            return None

    mold_parts: list[np.ndarray] = []
    hole_offsets: list[tuple[int, int]] = []
    mold_sizes: list[tuple[int, int]] = []
    for hole_mask, mold_color in zip(hole_masks, mold_colors, strict=False):
        part, offset = _build_mold_patch(rng, hole_mask, mold_color)
        mold_parts.append(part)
        hole_offsets.append(offset)
        mold_sizes.append(tuple(map(int, part.shape)))

    placed = _place_rectangles(rng, mold_sizes, gap=2, extra_span=10)
    if placed is None:
        return None
    (out_h, out_w), mold_pos = placed

    mold_canvas = np.full((out_h, out_w), BG, dtype=int)
    expected = np.full((out_h, out_w), BG, dtype=int)
    expected[:, :] = mold_canvas
    expected.fill(BG)
    for mold_part, patch, rot_k, (mr, mc), (hr, hc), hole_mask in zip(
        mold_parts, source_patches, rot_ks, mold_pos, hole_offsets, hole_masks, strict=False
    ):
        mh, mw = mold_part.shape
        mold_canvas[mr : mr + mh, mc : mc + mw] = mold_part
        expected[mr : mr + mh, mc : mc + mw] = mold_part
        rotated = np.rot90(patch, int(rot_k))
        assert np.array_equal(rotated != BG, hole_mask)
        target = expected[mr + hr : mr + hr + hole_mask.shape[0], mc + hc : mc + hc + hole_mask.shape[1]]
        target[hole_mask] = rotated[hole_mask]

    mold_canvas = _crop_non_bg(mold_canvas, BG)
    expected = _crop_non_bg(expected, BG)

    frame = np.full((material.shape[0] + 2, material.shape[1] + 2), BG, dtype=int)
    frame[0, :] = GRAY
    frame[-1, :] = GRAY
    frame[:, 0] = GRAY
    frame[:, -1] = GRAY
    frame[1:-1, 1:-1] = material

    top_gap = int(rng.integers(1, 4))
    left_margin = int(rng.integers(0, 3))
    right_margin = int(rng.integers(0, 3))
    between_gap = int(rng.integers(2, 5))
    src_top = int(rng.integers(0, 3))
    mold_top = int(rng.integers(0, 3))
    top_h = max(src_top + frame.shape[0], mold_top + mold_canvas.shape[0]) + top_gap
    full_w = left_margin + frame.shape[1] + between_gap + mold_canvas.shape[1] + right_margin

    top = np.full((top_h, full_w), BG, dtype=int)
    src_left = left_margin
    mold_left = left_margin + frame.shape[1] + between_gap
    top[src_top : src_top + frame.shape[0], src_left : src_left + frame.shape[1]] = frame
    top[mold_top : mold_top + mold_canvas.shape[0], mold_left : mold_left + mold_canvas.shape[1]] = mold_canvas

    palette_pairs = list(zip(mold_colors, rot_ks, strict=False))
    order = list(rng.permutation(len(palette_pairs)))
    ordered_pairs = [palette_pairs[i] for i in order]
    palette_len = 2 * len(ordered_pairs)
    palette_start = int(rng.integers(0, full_w - palette_len + 1))
    palette_row = np.full((full_w,), BG, dtype=int)
    for idx, (mold_color, rot_k) in enumerate(ordered_pairs):
        palette_row[palette_start + 2 * idx] = int(mold_color)
        palette_row[palette_start + 2 * idx + 1] = int(ROTATION_K_TO_COLOR[int(rot_k)])

    inp = np.vstack([top, palette_row[np.newaxis, :]])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        built = _generate_example(rng)
        if built is not None:
            inp, _ = built
            return inp
    raise RuntimeError("failed to generate valid puzzle 7 input")
