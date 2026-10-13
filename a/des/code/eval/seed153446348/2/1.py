import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
FRAME_COUNT_CHOICES = (1, 2, 3, 4)
THICKNESS_CHOICES = (2, 3)
FRAME_PAD = 2
PIECE_PAD = 1
PLACE_RETRIES = 200
GEN_RETRIES = 300

SIDES = ("top", "bottom", "left", "right")


def _mask(rows):
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    _mask(["11"]),
    _mask(["111"]),
    _mask(["10", "11"]),
    _mask(["1111"]),
    _mask(["11", "11"]),
    _mask(["111", "010"]),
    _mask(["100", "111"]),
    _mask(["001", "111"]),
    _mask(["011", "110"]),
    _mask(["110", "011"]),
]


def _bg_color(grid):
    return Counter(int(x) for x in np.ravel(grid)).most_common(1)[0][0]


def _trim(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _rotations(mask):
    out = []
    seen = set()
    for k in range(4):
        rot = _trim(np.rot90(mask, k))
        key = (rot.shape, rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rot)
    return out


def _canonical_key(mask):
    return min((rot.shape[0], rot.shape[1], rot.tobytes()) for rot in _rotations(mask))


def _bbox(coords):
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _coords_to_mask(coords):
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _border_mask(height, width, thickness):
    rr, cc = np.indices((height, width))
    edge_dist = np.minimum.reduce([rr, cc, height - 1 - rr, width - 1 - cc])
    return edge_dist < thickness


def _frame_descriptor(grid, color, coords, bg):
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 6 or width < 6:
        return None

    sub = np.zeros((height, width), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True

    thickness = 0
    max_t = min(height, width) // 2
    for cand in range(1, max_t + 1):
        if height <= 2 * cand or width <= 2 * cand:
            break
        if (
            sub[:cand, :cand].all()
            and sub[:cand, width - cand :].all()
            and sub[height - cand :, :cand].all()
            and sub[height - cand :, width - cand :].all()
        ):
            thickness = cand
    if thickness < 2:
        return None

    border = _border_mask(height, width, thickness)
    if np.any(sub & ~border):
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(patch[~border] != bg):
        return None

    missing = border & ~sub
    if np.any(patch[missing] != bg):
        return None

    missing_count = int(missing.sum())
    if missing_count < 2 or missing_count > 4:
        return None

    missing_components = _components(missing)
    if len(missing_components) != 1:
        return None

    side_names = set()
    for rr, cc in missing_components[0]:
        if rr < thickness:
            side_names.add("top")
        if rr >= height - thickness:
            side_names.add("bottom")
        if cc < thickness:
            side_names.add("left")
        if cc >= width - thickness:
            side_names.add("right")
    if len(side_names) != 1:
        return None

    if int(sub.sum()) != int(border.sum() - missing_count):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "thickness": int(thickness),
        "border": border,
        "missing_shape": _trim(missing),
    }


def _find_frames(grid, bg):
    frame_cells = np.zeros(grid.shape, dtype=bool)
    frames = []
    for color in (int(c) for c in np.unique(grid) if int(c) != bg):
        for coords in _components(grid == color):
            desc = _frame_descriptor(grid, color, coords, bg)
            if desc is None:
                continue
            frames.append(desc)
            frame_cells[coords[:, 0], coords[:, 1]] = True
    frames.sort(key=lambda item: item["bbox"])
    return frames, frame_cells


def _find_pieces(grid, bg, frame_cells):
    piece_mask = (grid != bg) & ~frame_cells
    pieces = []
    for color in (int(c) for c in np.unique(grid[piece_mask])):
        for coords in _components(piece_mask & (grid == color)):
            size = int(coords.shape[0])
            if size < 2 or size > 4:
                raise AssertionError("all non-frame foreground components must be 2-4 cell pieces")
            mask = _coords_to_mask(coords)
            pieces.append(
                {
                    "color": int(color),
                    "mask": mask,
                    "key": _canonical_key(mask),
                }
            )
    pieces.sort(key=lambda item: (item["color"], item["mask"].shape))
    return pieces, piece_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's missing border shape to a rotated scattered piece.

    The matched piece color fills the missing cells and recolors the whole frame
    border, while the frame interior stays background. The scattered source
    pieces are removed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    frames, frame_cells = _find_frames(g, bg)
    pieces, piece_cells = _find_pieces(g, bg, frame_cells)

    assert frames, "expected at least one frame"
    assert len(frames) == len(pieces), "frame and piece counts must match"

    piece_by_key = {}
    for piece in pieces:
        assert piece["key"] not in piece_by_key, "piece shapes must be unique up to rotation"
        piece_by_key[piece["key"]] = piece

    out = g.copy()
    out[piece_cells] = bg

    for frame in frames:
        key = _canonical_key(frame["missing_shape"])
        assert key in piece_by_key, "every frame notch must match one piece"
        fill_color = piece_by_key[key]["color"]
        r0, c0, r1, c1 = frame["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[frame["border"]] = fill_color

    return out


def _valid_shape_for_side(base_shape, side, thickness):
    valid = []
    for rot in _rotations(base_shape):
        height, width = rot.shape
        if side in ("top", "bottom") and height <= thickness:
            valid.append(rot)
        if side in ("left", "right") and width <= thickness:
            valid.append(rot)
    return valid


def _make_frame_spec(color, base_shape):
    for _ in range(40):
        side = random.choice(SIDES)
        thickness = random.choice(THICKNESS_CHOICES)
        valid = _valid_shape_for_side(base_shape, side, thickness)
        if not valid:
            continue
        notch_shape = random.choice(valid)
        notch_h, notch_w = notch_shape.shape

        if side in ("top", "bottom"):
            inner_h_min = 4
            inner_w_min = notch_w + 2
        else:
            inner_h_min = notch_h + 2
            inner_w_min = 4

        inner_h = random.randint(inner_h_min, inner_h_min + 4)
        inner_w = random.randint(inner_w_min, inner_w_min + 5)
        height = inner_h + 2 * thickness
        width = inner_w + 2 * thickness

        if side == "top":
            notch_r = 0
            notch_c = random.randint(thickness + 1, width - thickness - notch_w - 1)
        elif side == "bottom":
            notch_r = height - notch_h
            notch_c = random.randint(thickness + 1, width - thickness - notch_w - 1)
        elif side == "left":
            notch_r = random.randint(thickness + 1, height - thickness - notch_h - 1)
            notch_c = 0
        else:
            notch_r = random.randint(thickness + 1, height - thickness - notch_h - 1)
            notch_c = width - notch_w

        border = _border_mask(height, width, thickness)
        cutout = np.zeros((height, width), dtype=bool)
        cutout[notch_r : notch_r + notch_h, notch_c : notch_c + notch_w] = notch_shape

        if not np.all(cutout <= border):
            continue

        return {
            "color": int(color),
            "shape": notch_shape,
            "height": int(height),
            "width": int(width),
            "thickness": int(thickness),
            "border": border,
            "cutout": cutout,
        }
    return None


def _can_place_bbox(occupied, top, left, height, width, pad):
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_bbox(occupied, top, left, height, width):
    occupied[top : top + height, left : left + width] = True


def _place_frames(specs, grid_h, grid_w):
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(0, grid_h - spec["height"])
            left = random.randint(0, grid_w - spec["width"])
            if not _can_place_bbox(occupied, top, left, spec["height"], spec["width"], FRAME_PAD):
                continue
            spec["top"] = int(top)
            spec["left"] = int(left)
            _mark_bbox(occupied, top, left, spec["height"], spec["width"])
            placed = True
            break
        if not placed:
            return False
    return True


def _render_output(specs, grid_h, grid_w):
    out = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        sub = out[top : top + spec["height"], left : left + spec["width"]]
        sub[spec["border"]] = spec["color"]
    return out


def _render_input_frames(specs, grid_h, grid_w):
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        sub = grid[top : top + spec["height"], left : left + spec["width"]]
        sub[spec["border"]] = GREY
        sub[spec["cutout"]] = BG
    return grid


def _place_pieces(grid, specs):
    occupied = np.zeros(grid.shape, dtype=bool)
    for spec in specs:
        top = spec["top"]
        left = spec["left"]
        _mark_bbox(occupied, top, left, spec["height"], spec["width"])
    for spec in specs:
        piece_mask = random.choice(_rotations(spec["shape"]))
        height, width = piece_mask.shape
        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(0, grid.shape[0] - height)
            left = random.randint(0, grid.shape[1] - width)
            if not _can_place_bbox(occupied, top, left, height, width, PIECE_PAD):
                continue
            sub = grid[top : top + height, left : left + width]
            if np.any(sub[piece_mask] != BG):
                continue
            sub[piece_mask] = spec["color"]
            _mark_bbox(occupied, top, left, height, width)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt():
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    colors = random.sample(PIECE_COLORS, frame_count)
    bases = random.sample(SHAPE_LIBRARY, frame_count)

    specs = []
    for color, base_shape in zip(colors, bases):
        spec = _make_frame_spec(color, base_shape)
        if spec is None:
            return None
        specs.append(spec)

    total_area = sum(spec["height"] * spec["width"] for spec in specs)
    min_h = max(max(spec["height"] for spec in specs) + 2, GRID_H_RANGE[0], int((total_area * 2) ** 0.5))
    min_w = max(max(spec["width"] for spec in specs) + 2, GRID_W_RANGE[0], int((total_area * 2) ** 0.5))
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_w, GRID_W_RANGE[1])
    if not _place_frames(specs, grid_h, grid_w):
        return None

    expected = _render_output(specs, grid_h, grid_w)
    inp = _render_input_frames(specs, grid_h, grid_w)
    if not _place_pieces(inp, specs):
        return None

    if _bg_color(inp) != BG:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input")
