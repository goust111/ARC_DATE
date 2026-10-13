import random

import numpy as np


_GEN_RETRIES = 300
_SIDES = ("top", "bottom", "left", "right")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mask_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "coords": np.array(cells, dtype=int)})
    return comps


def _bbox(coords: np.ndarray):
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _shape_key_from_coords(coords: np.ndarray):
    norm = coords - coords.min(axis=0)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[norm[:, 0], norm[:, 1]] = 1
    return h, w, mask.tobytes()


def _shape_key_from_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    return _shape_key_from_coords(coords)


def _has_hole(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    bg = ~mask
    for comp in _mask_components(bg):
        if (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == mask.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == mask.shape[1] - 1)
        ):
            continue
        return True
    return False


def _extract_frame(grid: np.ndarray, bg: int, comp: dict):
    coords = comp["coords"]
    color = int(comp["color"])
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((patch != bg) & (patch != color)):
        return None

    comp_mask = np.zeros_like(patch, dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    comp_mask[local[:, 0], local[:, 1]] = True
    if not np.array_equal(comp_mask, patch == color):
        return None
    if not comp_mask[0, :].all() or not comp_mask[-1, :].all():
        return None
    if not comp_mask[:, 0].all() or not comp_mask[:, -1].all():
        return None

    interior = patch[1:-1, 1:-1]
    if interior.size == 0:
        return None
    hole_mask = interior == bg
    if not hole_mask.any():
        return None
    hole_components = _mask_components(hole_mask)
    if len(hole_components) != 1:
        return None
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "hole_mask": hole_mask,
        "hole_coords": hole_components[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame's negative-space hole with the same-shaped palette piece color, then erase the palette.

    Input structure:
    - The background is the most common color.
    - Several monochrome rectangular frame objects appear in the interior; each has a full one-cell border and extra same-colored fill inside, leaving exactly one connected background-shaped hole.
    - A palette sits along one grid side as separated monochrome 4-connected colored pieces, each piece using a different color and shape.

    Rule:
    - Detect every frame and extract the connected background hole shape inside its border.
    - Detect the non-frame palette pieces and match them to holes by exact shape equality up to translation only; rotation and reflection do not count.
    - Recolor each frame hole with the matched palette piece color while leaving the frame pixels unchanged.
    - Clear all palette pieces back to the background color.

    Key insight:
    - The palette gives positive shapes, while each frame encodes its target as negative space; solving requires matching those normalized silhouettes rather than colors or positions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _color_components(g, bg)

    frames = []
    frame_color_set = set()
    for comp in comps:
        frame = _extract_frame(g, bg, comp)
        if frame is not None:
            frames.append(frame)
            frame_color_set.add(int(frame["color"]))

    palette_pieces = []
    for comp in comps:
        if int(comp["color"]) in frame_color_set:
            maybe_frame = _extract_frame(g, bg, comp)
            if maybe_frame is not None:
                continue
        palette_pieces.append(comp)

    shape_to_piece = {}
    for piece in palette_pieces:
        key = _shape_key_from_coords(piece["coords"])
        if key in shape_to_piece:
            raise AssertionError("palette shapes must be unique")
        shape_to_piece[key] = piece

    out = g.copy()
    for piece in palette_pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg

    for frame in frames:
        key = _shape_key_from_coords(frame["hole_coords"])
        if key not in shape_to_piece:
            raise AssertionError("every frame hole must have a matching palette piece")
        fill_color = int(shape_to_piece[key]["color"])
        r0, c0, _r1, _c1 = frame["bbox"]
        hole = frame["hole_coords"]
        out[r0 + 1 + hole[:, 0], c0 + 1 + hole[:, 1]] = fill_color

    return out


def _random_shape(rng: random.Random):
    for _ in range(200):
        h = rng.randint(1, 4)
        w = rng.randint(1, 5)
        target = rng.randint(1, min(h * w, 7))
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        hh = int(coords[:, 0].max()) + 1
        ww = int(coords[:, 1].max()) + 1
        mask = np.zeros((hh, ww), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        if _has_hole(mask):
            continue
        return mask
    raise RuntimeError("failed to create shape")


def _make_frame_patch(shape: np.ndarray, frame_color: int, bg: int, rng: random.Random):
    sh, sw = shape.shape
    extra_h = rng.randint(0, 2)
    extra_w = rng.randint(0, 2)
    if bool(shape.all()) and extra_h == 0 and extra_w == 0:
        if rng.random() < 0.5:
            extra_h = 1
        else:
            extra_w = 1
    ih = sh + extra_h
    iw = sw + extra_w
    top = rng.randint(0, ih - sh)
    left = rng.randint(0, iw - sw)
    hole_mask = np.zeros((ih, iw), dtype=bool)
    hole_mask[top : top + sh, left : left + sw] = shape
    if hole_mask.all():
        if ih < iw:
            hole_mask[0, 0] = False
        else:
            hole_mask[-1, -1] = False
    patch = np.full((ih + 2, iw + 2), frame_color, dtype=int)
    patch[1:-1, 1:-1][hole_mask] = bg
    return patch, hole_mask


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    if r < 0 or c < 0 or r + h > hh or c + w > ww:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(hh, r + h + pad)
    c1 = min(ww, c + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r: int, c: int):
    h, w = patch.shape
    grid[r : r + h, c : c + w] = patch
    occupied[r : r + h, c : c + w] = True


def _place_palette(grid: np.ndarray, occupied: np.ndarray, pieces: list, side: str, rng: random.Random):
    h, w = grid.shape
    order = pieces[:]
    rng.shuffle(order)

    if side in ("top", "bottom"):
        total_w = sum(piece["patch"].shape[1] for piece in order)
        total_w += sum(rng.randint(1, 3) for _ in range(len(order) - 1))
        if total_w > w:
            return None
        band_h = max(piece["patch"].shape[0] for piece in order)
        start_c = rng.randint(0, w - total_w)
        c = start_c
        for idx, piece in enumerate(order):
            patch = piece["patch"]
            ph, pw = patch.shape
            r = 0 if side == "top" else h - ph
            if not _can_place(occupied, r, c, ph, pw, pad=1):
                return None
            _place_patch(grid, occupied, patch, r, c)
            piece["placed_bbox"] = (r, c, r + ph - 1, c + pw - 1)
            if idx + 1 < len(order):
                c += pw + rng.randint(1, 3)
        return band_h

    total_h = sum(piece["patch"].shape[0] for piece in order)
    total_h += sum(rng.randint(1, 3) for _ in range(len(order) - 1))
    if total_h > h:
        return None
    band_w = max(piece["patch"].shape[1] for piece in order)
    start_r = rng.randint(0, h - total_h)
    r = start_r
    for idx, piece in enumerate(order):
        patch = piece["patch"]
        ph, pw = patch.shape
        c = 0 if side == "left" else w - pw
        if not _can_place(occupied, r, c, ph, pw, pad=1):
            return None
        _place_patch(grid, occupied, patch, r, c)
        piece["placed_bbox"] = (r, c, r + ph - 1, c + pw - 1)
        if idx + 1 < len(order):
            r += ph + rng.randint(1, 3)
    return band_w


def _expected_output(grid: np.ndarray, frames: list, pieces: list, bg: int):
    out = grid.copy()
    shape_to_color = {}
    for piece in pieces:
        key = _shape_key_from_mask(piece["shape"])
        shape_to_color[key] = int(piece["color"])
        r0, c0, r1, c1 = piece["placed_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][piece["patch"] != bg] = bg

    for frame in frames:
        key = _shape_key_from_mask(frame["shape"])
        fill_color = shape_to_color[key]
        r0, c0, _r1, _c1 = frame["placed_bbox"]
        hole = frame["hole_mask"]
        out[r0 + 1 : r0 + 1 + hole.shape[0], c0 + 1 : c0 + 1 + hole.shape[1]][hole] = fill_color
    return out


def _gen_attempt(rng: random.Random):
    n_frames = rng.randint(2, 4)
    side = rng.choice(_SIDES)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    frame_color = rng.choice(palette)
    palette.remove(frame_color)
    piece_colors = rng.sample(palette, n_frames)

    shapes = []
    seen_shapes = set()
    while len(shapes) < n_frames:
        shape = _random_shape(rng)
        key = _shape_key_from_mask(shape)
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        shapes.append(shape)

    frames = []
    pieces = []
    for shape, color in zip(shapes, piece_colors):
        frame_patch, hole_mask = _make_frame_patch(shape, frame_color, bg, rng)
        frames.append(
            {
                "shape": shape,
                "patch": frame_patch,
                "hole_mask": hole_mask,
                "color": frame_color,
            }
        )
        piece_patch = np.full(shape.shape, bg, dtype=int)
        piece_patch[shape] = color
        pieces.append({"shape": shape, "patch": piece_patch, "color": color})

    frame_max_h = max(frame["patch"].shape[0] for frame in frames)
    frame_max_w = max(frame["patch"].shape[1] for frame in frames)
    piece_max_h = max(piece["patch"].shape[0] for piece in pieces)
    piece_max_w = max(piece["patch"].shape[1] for piece in pieces)
    piece_span_h = sum(piece["patch"].shape[0] for piece in pieces) + 3 * (len(pieces) - 1)
    piece_span_w = sum(piece["patch"].shape[1] for piece in pieces) + 3 * (len(pieces) - 1)

    if side in ("top", "bottom"):
        min_h = max(14, frame_max_h + piece_max_h + 4)
        min_w = max(14, piece_span_w + 1, frame_max_w + 6)
    else:
        min_h = max(14, piece_span_h + 1, frame_max_h + 6)
        min_w = max(14, frame_max_w + piece_max_w + 4)

    max_h = min(30, min_h + 8)
    max_w = min(30, min_w + 8)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    band = _place_palette(grid, occupied, pieces, side, rng)
    if band is None:
        return None

    if side == "top":
        row_lo, row_hi = band + 1, h
        col_lo, col_hi = 0, w
    elif side == "bottom":
        row_lo, row_hi = 0, h - band - 1
        col_lo, col_hi = 0, w
    elif side == "left":
        row_lo, row_hi = 0, h
        col_lo, col_hi = band + 1, w
    else:
        row_lo, row_hi = 0, h
        col_lo, col_hi = 0, w - band - 1

    if row_hi - row_lo < frame_max_h or col_hi - col_lo < frame_max_w:
        return None

    frame_order = frames[:]
    rng.shuffle(frame_order)
    for frame in frame_order:
        patch = frame["patch"]
        ph, pw = patch.shape
        placed = False
        for _ in range(200):
            r = rng.randint(row_lo, row_hi - ph)
            c = rng.randint(col_lo, col_hi - pw)
            if not _can_place(occupied, r, c, ph, pw, pad=1):
                continue
            _place_patch(grid, occupied, patch, r, c)
            frame["placed_bbox"] = (r, c, r + ph - 1, c + pw - 1)
            placed = True
            break
        if not placed:
            return None

    if int((grid == bg).sum()) <= max(int((grid == frame_color).sum()), *(int((grid == color).sum()) for color in piece_colors)):
        return None

    expected = _expected_output(grid, frames, pieces, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
