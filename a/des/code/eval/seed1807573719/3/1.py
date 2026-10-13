from __future__ import annotations

import random

import numpy as np


_SIDE_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_SIDES = tuple(_SIDE_TO_VEC)
_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
_MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    tight = _tight_mask(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(tight, k)
        for variant in (rot, np.fliplr(rot)):
            clipped = _tight_mask(variant)
            key = (tuple(map(int, clipped.shape)), clipped.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(clipped)
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(
        (tuple(map(int, variant.shape)), variant.tobytes()) for variant in _dihedral_variants(mask)
    )


def _frame_info(grid: np.ndarray, frame_color: int, coords: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    rel = coords - np.array([[r0, c0]], dtype=int)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    comp_mask = np.zeros((h, w), dtype=bool)
    comp_mask[rel[:, 0], rel[:, 1]] = True
    if np.any(comp_mask & ~border_mask):
        return None
    if int(border_mask.sum()) != int(comp_mask.sum()) + 1:
        return None

    missing = np.argwhere(border_mask & ~comp_mask)
    if missing.shape[0] != 1:
        return None
    mr, mc = map(int, missing[0])
    side_names = []
    if mr == 0:
        side_names.append("up")
    if mr == h - 1:
        side_names.append("down")
    if mc == 0:
        side_names.append("left")
    if mc == w - 1:
        side_names.append("right")
    if len(side_names) != 1:
        return None

    gap = (int(r0 + mr), int(c0 + mc))
    if int(grid[gap]) != int(bg):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    island_mask = (interior != int(bg)) & (interior != int(frame_color))
    if not island_mask.any():
        return None
    if (
        bool(island_mask[0, :].any())
        or bool(island_mask[-1, :].any())
        or bool(island_mask[:, 0].any())
        or bool(island_mask[:, -1].any())
    ):
        return None
    if len(_components(island_mask)) != 1:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": gap,
        "side": side_names[0],
        "dir": _SIDE_TO_VEC[side_names[0]],
    }


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidate_color = None
    candidate_infos: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == int(bg):
            continue
        infos = []
        for comp in _components(grid == color):
            info = _frame_info(grid, color, comp, bg)
            if info is not None:
                infos.append(info)
        if infos:
            assert candidate_color is None, "expected a unique frame color"
            candidate_color = int(color)
            candidate_infos = infos
    assert candidate_color is not None and candidate_infos, "expected at least one frame"
    candidate_infos.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return int(candidate_color), candidate_infos


def _extract_dictionary(grid: np.ndarray, bg: int, frame_color: int) -> dict[tuple[tuple[int, int], bytes], int]:
    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in _components(grid != int(bg)):
        colors = sorted({int(grid[r, c]) for r, c in comp})
        if len(colors) != 2 or int(frame_color) in colors:
            continue
        by_color = {color: comp[(grid[comp[:, 0], comp[:, 1]] == color)] for color in colors}

        block_color = None
        shape_color = None
        for color, cells in by_color.items():
            if cells.shape[0] != 2:
                continue
            r0, c0, r1, c1 = _bbox(cells)
            if (r1 - r0 + 1, c1 - c0 + 1) not in {(1, 2), (2, 1)}:
                continue
            other = next(other_color for other_color in colors if other_color != color)
            if by_color[other].shape[0] < 3:
                continue
            block_color = int(color)
            shape_color = int(other)
            break

        if block_color is None or shape_color is None:
            continue

        shape_cells = by_color[shape_color]
        sr0, sc0, sr1, sc1 = _bbox(shape_cells)
        shape_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        shape_mask[shape_cells[:, 0] - sr0, shape_cells[:, 1] - sc0] = True
        key = _canonical_shape_key(shape_mask)
        assert key not in mapping, "dictionary shapes must be unique up to rotation/flip"
        mapping[key] = int(block_color)

    assert mapping, "expected at least one dictionary entry"
    return mapping


def _island_key(grid: np.ndarray, frame: dict, bg: int, frame_color: int) -> tuple[tuple[int, int], bytes]:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    island_mask = (interior != int(bg)) & (interior != int(frame_color))
    comps = _components(island_mask)
    assert len(comps) == 1, "expected one island per frame"
    coords = comps[0]
    ir0, ic0, ir1, ic1 = _bbox(coords)
    tight = np.zeros((ir1 - ir0 + 1, ic1 - ic0 + 1), dtype=bool)
    tight[coords[:, 0] - ir0, coords[:, 1] - ic0] = True
    return _canonical_shape_key(tight)


def _paint_ray(
    source: np.ndarray,
    out: np.ndarray,
    bg: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> None:
    h, w = source.shape
    r, c = map(int, start)
    dr, dc = map(int, direction)
    while 0 <= r < h and 0 <= c < w and int(source[r, c]) == int(bg):
        out[r, c] = int(color)
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed island to the dictionary shape, then fire a color ray outward through the frame gap.

    The dictionary is read structurally: each entry is a two-color connected component where one color is a
    2-cell domino block and the other color is the sample shape. Each grey frame is recognized by its geometry,
    not by a fixed location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    frame_color, frames = _find_frames(g, bg)
    dictionary = _extract_dictionary(g, bg, frame_color)

    out = g.copy()
    for frame in frames:
        key = _island_key(g, frame, bg, frame_color)
        assert key in dictionary, "every island shape must appear in the dictionary"
        _paint_ray(g, out, bg, frame["gap"], frame["dir"], dictionary[key])
    return out


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        _shape(["10", "11"]),
        _shape(["111", "010"]),
        _shape(["110", "011"]),
        _shape(["11", "11", "10"]),
        _shape(["101", "111"]),
        _shape(["100", "110", "011"]),
        _shape(["010", "111", "010"]),
        _shape(["10", "11", "10", "10"]),
    ]
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for mask in raw:
        key = _canonical_shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        out.append(mask)
    return out


_SHAPE_LIBRARY = _build_shape_library()


def _random_variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_dihedral_variants(mask)).copy()


def _dictionary_patch(shape_mask: np.ndarray, bg: int, shape_color: int, output_color: int) -> np.ndarray:
    side = random.choice(_SIDES)
    h, w = map(int, shape_mask.shape)
    if side == "left":
        edge_rows = [r for r in range(h) if shape_mask[r, 0]]
        row = random.choice(edge_rows)
        patch = np.full((h, w + 2), int(bg), dtype=int)
        patch[:, 2:] = np.where(shape_mask, shape_color, bg)
        patch[row, 0:2] = output_color
    elif side == "right":
        edge_rows = [r for r in range(h) if shape_mask[r, w - 1]]
        row = random.choice(edge_rows)
        patch = np.full((h, w + 2), int(bg), dtype=int)
        patch[:, :w] = np.where(shape_mask, shape_color, bg)
        patch[row, w : w + 2] = output_color
    elif side == "up":
        edge_cols = [c for c in range(w) if shape_mask[0, c]]
        col = random.choice(edge_cols)
        patch = np.full((h + 2, w), int(bg), dtype=int)
        patch[2:, :] = np.where(shape_mask, shape_color, bg)
        patch[0:2, col] = output_color
    else:
        edge_cols = [c for c in range(w) if shape_mask[h - 1, c]]
        col = random.choice(edge_cols)
        patch = np.full((h + 2, w), int(bg), dtype=int)
        patch[:h, :] = np.where(shape_mask, shape_color, bg)
        patch[h : h + 2, col] = output_color
    return patch


def _frame_patch(
    shape_mask: np.ndarray,
    bg: int,
    frame_color: int,
    shape_color: int,
    forced_side: str | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], str]:
    sh, sw = map(int, shape_mask.shape)
    outer_h = sh + random.randint(4, 6)
    outer_w = sw + random.randint(4, 6)
    inner_h = outer_h - 2
    inner_w = outer_w - 2
    top = random.randint(1, inner_h - sh - 1)
    left = random.randint(1, inner_w - sw - 1)

    patch = np.full((outer_h, outer_w), int(bg), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    window = patch[1 + top : 1 + top + sh, 1 + left : 1 + left + sw]
    window[shape_mask] = shape_color

    side = forced_side or random.choice(_SIDES)
    if side == "up":
        gap = (0, random.randint(1, outer_w - 2))
    elif side == "down":
        gap = (outer_h - 1, random.randint(1, outer_w - 2))
    elif side == "left":
        gap = (random.randint(1, outer_h - 2), 0)
    else:
        gap = (random.randint(1, outer_h - 2), outer_w - 1)
    patch[gap] = int(bg)
    return patch, gap, _SIDE_TO_VEC[side], side


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(bg)
    window[mask] = patch[mask]


def _rect_is_clear(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h_all, w_all = mask.shape
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(h_all, int(r0) + h + pad)
    cc1 = min(w_all, int(c0) + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _mark_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    h_all, w_all = mask.shape
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(h_all, int(r0) + h + pad)
    cc1 = min(w_all, int(c0) + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _trace_ray(occupied: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    h, w = occupied.shape
    r, c = map(int, start)
    dr, dc = map(int, direction)
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if occupied[r, c]:
            return cells, (int(r), int(c))
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells, None


def _dictionary_positions(
    patches: list[np.ndarray], corner: str, height: int, width: int
) -> list[tuple[np.ndarray, int, int]] | None:
    gap = random.randint(1, 2)
    total_h = sum(int(p.shape[0]) for p in patches) + gap * (len(patches) - 1)
    max_w = max(int(p.shape[1]) for p in patches)
    inset_r = random.randint(1, 2)
    inset_c = random.randint(1, 2)

    if total_h + inset_r > height - 2 or max_w + inset_c > width - 4:
        return None

    if corner.startswith("top"):
        top = inset_r
    else:
        top = height - inset_r - total_h
    if corner.endswith("left"):
        anchor_c = inset_c
        align = lambda pw: anchor_c
    else:
        anchor_c = width - inset_c - max_w
        align = lambda pw: anchor_c + max_w - pw

    placed = []
    row = top
    for patch in patches:
        ph, pw = map(int, patch.shape)
        col = int(align(pw))
        placed.append((patch, int(row), col))
        row += ph + gap
    return placed


def _candidate_frame_positions(
    patch: np.ndarray,
    gap_rel: tuple[int, int],
    direction: tuple[int, int],
    grid: np.ndarray,
    occupied: np.ndarray,
    object_forbidden: np.ndarray,
    ray_reserved: np.ndarray,
    require_hit_mask: np.ndarray | None,
    bg: int,
    min_ray_len: int,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    h_all, w_all = grid.shape
    ph, pw = map(int, patch.shape)
    out: list[tuple[int, int, list[tuple[int, int]]]] = []
    for r0 in range(h_all - ph + 1):
        for c0 in range(w_all - pw + 1):
            if not _rect_is_clear(object_forbidden, r0, c0, ph, pw, pad=1):
                continue
            if not _rect_is_clear(ray_reserved, r0, c0, ph, pw, pad=0):
                continue
            abs_gap = (int(r0 + gap_rel[0]), int(c0 + gap_rel[1]))
            ray_cells, hit = _trace_ray(occupied, abs_gap, direction)
            if len(ray_cells) < min_ray_len:
                continue
            if any(ray_reserved[r, c] for r, c in ray_cells):
                continue
            if require_hit_mask is None:
                if hit is not None:
                    continue
            else:
                if hit is None or not bool(require_hit_mask[hit]):
                    continue
            if any(int(grid[r, c]) != int(bg) for r, c in ray_cells):
                continue
            out.append((int(r0), int(c0), ray_cells))
    random.shuffle(out)
    return out


def _render_expected(
    inp: np.ndarray,
    bg: int,
    rays: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for ray in rays:
        _paint_ray(inp, out, bg, ray["gap"], ray["dir"], ray["color"])
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(24, 30)
    width = random.randint(24, 30)

    colors = list(range(10))
    random.shuffle(colors)
    bg, frame_color, shape_color = map(int, colors[:3])
    n_items = random.randint(2, 4)
    output_colors = list(map(int, colors[3 : 3 + n_items]))
    corner = random.choice(_CORNERS)
    want_blocked = random.random() < 0.5

    shapes = random.sample(_SHAPE_LIBRARY, n_items)
    random.shuffle(output_colors)
    dictionary_shapes = [_random_variant(shape) for shape in shapes]
    dictionary_patches = [
        _dictionary_patch(shape_mask, bg, shape_color, out_color)
        for shape_mask, out_color in zip(dictionary_shapes, output_colors, strict=False)
    ]
    dictionary_layout = _dictionary_positions(dictionary_patches, corner, height, width)
    if dictionary_layout is None:
        raise RuntimeError("dictionary did not fit")

    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    object_forbidden = np.zeros((height, width), dtype=bool)
    ray_reserved = np.zeros((height, width), dtype=bool)
    dictionary_mask = np.zeros((height, width), dtype=bool)

    for patch, r0, c0 in dictionary_layout:
        ph, pw = map(int, patch.shape)
        _paint_patch(grid, patch, r0, c0, bg)
        non_bg = patch != int(bg)
        occupied[r0 : r0 + ph, c0 : c0 + pw][non_bg] = True
        dictionary_mask[r0 : r0 + ph, c0 : c0 + pw][non_bg] = True
        _mark_rect(object_forbidden, r0, c0, ph, pw, pad=1)

    pairs = list(zip(shapes, output_colors, strict=False))
    random.shuffle(pairs)
    rays: list[dict] = []

    if want_blocked:
        forced_side = "left" if corner.endswith("left") else "right"
        shape_mask, out_color = pairs.pop()
        placed = False
        for _ in range(40):
            island = _random_variant(shape_mask)
            patch, gap_rel, direction, _side = _frame_patch(
                island, bg, frame_color, shape_color, forced_side=forced_side
            )
            candidates = _candidate_frame_positions(
                patch,
                gap_rel,
                direction,
                grid,
                occupied,
                object_forbidden,
                ray_reserved,
                require_hit_mask=dictionary_mask,
                bg=bg,
                min_ray_len=2,
            )
            if not candidates:
                continue
            r0, c0, ray_cells = random.choice(candidates)
            ph, pw = map(int, patch.shape)
            _paint_patch(grid, patch, r0, c0, bg)
            non_bg = patch != int(bg)
            occupied[r0 : r0 + ph, c0 : c0 + pw][non_bg] = True
            _mark_rect(object_forbidden, r0, c0, ph, pw, pad=1)
            for rr, cc in ray_cells:
                ray_reserved[rr, cc] = True
            rays.append(
                {
                    "gap": (int(r0 + gap_rel[0]), int(c0 + gap_rel[1])),
                    "dir": direction,
                    "color": int(out_color),
                }
            )
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place blocked frame")

    for shape_mask, out_color in pairs:
        placed = False
        for _ in range(60):
            island = _random_variant(shape_mask)
            patch, gap_rel, direction, _side = _frame_patch(island, bg, frame_color, shape_color)
            candidates = _candidate_frame_positions(
                patch,
                gap_rel,
                direction,
                grid,
                occupied,
                object_forbidden,
                ray_reserved,
                require_hit_mask=None,
                bg=bg,
                min_ray_len=2,
            )
            if not candidates:
                continue
            r0, c0, ray_cells = random.choice(candidates)
            ph, pw = map(int, patch.shape)
            _paint_patch(grid, patch, r0, c0, bg)
            non_bg = patch != int(bg)
            occupied[r0 : r0 + ph, c0 : c0 + pw][non_bg] = True
            _mark_rect(object_forbidden, r0, c0, ph, pw, pad=1)
            for rr, cc in ray_cells:
                ray_reserved[rr, cc] = True
            rays.append(
                {
                    "gap": (int(r0 + gap_rel[0]), int(c0 + gap_rel[1])),
                    "dir": direction,
                    "color": int(out_color),
                }
            )
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place clear frame")

    expected = _render_expected(grid, bg, rays)
    if _most_common_color(grid) != int(bg):
        raise RuntimeError("background must stay dominant")
    if not np.array_equal(main(grid), expected):
        raise RuntimeError("oracle mismatch")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        try:
            inp, _out = _generate_case()
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid sample")
