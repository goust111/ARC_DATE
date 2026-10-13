import random
from collections import Counter, deque

import numpy as np


BG = 0
MAROON = 9
SIDES = ("top", "right", "bottom", "left")
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def label_components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    else:
        steps = DIRS4

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def marker_side(mask: np.ndarray, marker_rc: tuple[int, int]) -> str | None:
    pts = np.argwhere(mask)
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)
    r, c = marker_rc
    sides = []
    if r == int(rmin):
        sides.append("top")
    if r == int(rmax):
        sides.append("bottom")
    if c == int(cmin):
        sides.append("left")
    if c == int(cmax):
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


def side_center(shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return 0, w // 2
    if side == "bottom":
        return h - 1, w // 2
    if side == "left":
        return h // 2, 0
    if side == "right":
        return h // 2, w - 1
    raise ValueError(side)


def detect_frame(patch: np.ndarray, color: int, bg: int) -> str | None:
    h, w = patch.shape
    if h < 5 or w < 5:
        return None
    if h % 2 == 0 or w % 2 == 0:
        return None
    if set(map(int, np.unique(patch))) - {int(bg), int(color)}:
        return None
    if not np.all(patch[1:-1, 1:-1] == int(bg)):
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    misses = np.argwhere(border & (patch == int(bg)))
    if misses.shape[0] != 1:
        return None
    mr, mc = map(int, misses[0])

    if (mr, mc) == (0, w // 2):
        side = "top"
    elif (mr, mc) == (h - 1, w // 2):
        side = "bottom"
    elif (mr, mc) == (h // 2, 0):
        side = "left"
    elif (mr, mc) == (h // 2, w - 1):
        side = "right"
    else:
        return None

    filled = patch.copy()
    filled[mr, mc] = int(color)
    if not np.all(filled[border] == int(color)):
        return None
    return side


def detect_frames(grid: np.ndarray, bg: int) -> dict[int, dict]:
    frames: dict[int, dict] = {}
    colors = sorted(set(map(int, grid.flatten())) - {int(bg), int(MAROON)})
    for color in colors:
        for coords in label_components(grid == int(color), connectivity=4):
            bbox = bbox_coords(coords)
            patch = crop_bbox(grid, bbox)
            gap_side = detect_frame(patch, int(color), int(bg))
            if gap_side is None:
                continue
            if color in frames:
                raise AssertionError("each frame color must be unique")
            frames[color] = {
                "color": int(color),
                "bbox": bbox,
                "gap_side": gap_side,
                "inner_size": (int(patch.shape[0] - 2), int(patch.shape[1] - 2)),
            }
    return frames


def detect_pieces(grid: np.ndarray, bg: int) -> dict[int, dict]:
    pieces: dict[int, dict] = {}
    for coords in label_components(grid != int(bg), connectivity=4):
        bbox = bbox_coords(coords)
        patch = np.full((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), int(bg), dtype=int)
        for r, c in coords:
            patch[int(r - bbox[0]), int(c - bbox[1])] = int(grid[int(r), int(c)])

        colors = set(map(int, patch[patch != int(bg)].flatten()))
        if MAROON not in colors:
            continue
        if int((patch == int(MAROON)).sum()) != 1:
            continue
        other = colors - {int(MAROON)}
        if len(other) != 1:
            continue

        piece_color = int(next(iter(other)))
        marker_rc = tuple(map(int, np.argwhere(patch == int(MAROON))[0]))
        side = marker_side(patch != int(bg), marker_rc)
        if side is None:
            raise AssertionError("marker must lie on exactly one piece side")

        if piece_color in pieces:
            raise AssertionError("each piece color must be unique")
        pieces[piece_color] = {
            "color": piece_color,
            "bbox": bbox,
            "patch": patch,
            "marker_side": side,
            "coords": coords,
        }
    return pieces


def orient_piece_to_side(piece_patch: np.ndarray, target_side: str, bg: int) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(piece_patch, k)
        markers = np.argwhere(rotated == int(MAROON))
        if markers.shape[0] != 1:
            continue
        side = marker_side(rotated != int(bg), tuple(map(int, markers[0])))
        if side == target_side:
            return rotated
    raise AssertionError("piece must have a rotation that matches the frame gap")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = detect_frames(g, bg)
    pieces = detect_pieces(g, bg)
    assert frames, "expected at least one frame"
    assert set(frames) == set(pieces), "frame-piece color pairing must be one-to-one"

    out = g.copy()
    for piece in pieces.values():
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)

    for color, frame in frames.items():
        piece = pieces[int(color)]
        rotated = orient_piece_to_side(piece["patch"], frame["gap_side"], bg)
        inner_h, inner_w = map(int, frame["inner_size"])
        if rotated.shape != (inner_h, inner_w):
            raise AssertionError("rotated piece must match the frame cavity size")
        if int((rotated != int(bg)).sum()) != int(inner_h * inner_w):
            raise AssertionError("piece must fully fill the cavity")
        r0, c0, _r1, _c1 = frame["bbox"]
        out[int(r0 + 1) : int(r0 + 1 + inner_h), int(c0 + 1) : int(c0 + 1 + inner_w)] = rotated

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored marked piece into the matching hollow frame after the marker-guided rotation.

    Input structure:
    - bg = black. The grid contains several colored rectangular frames whose borders are one pixel thick and whose interiors are empty.
    - Every frame color is unique, and each frame border has exactly one one-pixel gap centered on one side.
    - A scattered piece of the same color exists for each frame; each piece contains exactly one maroon marker pixel on one side edge.

    Rule:
    - Pair every piece with the frame of the same color.
    - Read the frame gap side and rotate the piece by 0/90/180/270 degrees until the maroon marker lies on the same side.
    - Translate the rotated piece into the frame interior so it fills the cavity completely, keeping the maroon marker.
    - Leave the frames fixed and clear the original piece locations back to background.

    Key insight:
    Color identifies which piece belongs to which frame, but the hidden instruction for the needed rotation is the direction match between the frame gap and the piece's maroon entry marker.
    """
    return _solve(grid)


COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
INNER_SIZE_CHOICES = [3, 5, 7]
N_OBJECTS_CHOICES = [2, 3, 4]
GEN_ATTEMPTS = 400


def make_frame_patch(color: int, inner_size: int, gap_side: str) -> np.ndarray:
    outer = inner_size + 2
    patch = np.full((outer, outer), int(BG), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    gr, gc = side_center((outer, outer), gap_side)
    patch[int(gr), int(gc)] = int(BG)
    return patch


def make_piece_patch(color: int, size: int, marker_side: str) -> np.ndarray:
    patch = np.full((size, size), int(color), dtype=int)
    mr, mc = side_center((size, size), marker_side)
    patch[int(mr), int(mc)] = int(MAROON)
    return patch


def can_place(occ: np.ndarray, top: int, left: int, shape: tuple[int, int], pad: int = 1) -> bool:
    h, w = map(int, shape)
    H, W = occ.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def reserve(occ: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> None:
    h, w = map(int, shape)
    occ[top : top + h, left : left + w] = True


def paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def build_expected(specs: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    out = np.full(grid_shape, int(BG), dtype=int)
    for spec in specs:
        fr, fc = spec["frame_pos"]
        frame_patch = spec["frame_patch"]
        paste(out, frame_patch, int(fr), int(fc))
        rotated = orient_piece_to_side(spec["piece_patch"], spec["gap_side"], BG)
        inner = int(spec["inner_size"])
        paste(out, rotated, int(fr + 1), int(fc + 1))
        assert rotated.shape == (inner, inner)
    return out


def gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice(N_OBJECTS_CHOICES)
    colors = rng.sample(COLOR_POOL, n_objects)
    gap_sides = rng.sample(list(SIDES), n_objects)

    specs = []
    item_shapes = []
    for color, gap_side in zip(colors, gap_sides):
        inner_size = rng.choice(INNER_SIZE_CHOICES)
        marker_side = rng.choice(SIDES)
        frame_patch = make_frame_patch(int(color), int(inner_size), gap_side)
        piece_patch = make_piece_patch(int(color), int(inner_size), marker_side)
        specs.append(
            {
                "color": int(color),
                "inner_size": int(inner_size),
                "gap_side": gap_side,
                "frame_patch": frame_patch,
                "piece_patch": piece_patch,
            }
        )
        item_shapes.append(frame_patch.shape)
        item_shapes.append(piece_patch.shape)

    total_area = sum(int(h * w) for h, w in item_shapes)
    max_dim = max(max(int(h), int(w)) for h, w in item_shapes) + 2
    min_side = max(15, max_dim, int(np.ceil(np.sqrt(total_area * 1.9))))
    if min_side > 30:
        return None

    H = rng.randint(min_side, 30)
    W = rng.randint(min_side, 30)
    occ = np.zeros((H, W), dtype=bool)
    inp = np.full((H, W), int(BG), dtype=int)

    for spec in sorted(specs, key=lambda s: -int(s["frame_patch"].shape[0])):
        frame_patch = spec["frame_patch"]
        fh, fw = frame_patch.shape
        placed = False
        for _ in range(200):
            if H - fh - 1 < 1 or W - fw - 1 < 1:
                return None
            fr = rng.randint(1, H - fh - 1)
            fc = rng.randint(1, W - fw - 1)
            if not can_place(occ, fr, fc, frame_patch.shape, pad=1):
                continue
            reserve(occ, fr, fc, frame_patch.shape)
            paste(inp, frame_patch, fr, fc)
            spec["frame_pos"] = (int(fr), int(fc))
            placed = True
            break
        if not placed:
            return None

        piece_patch = spec["piece_patch"]
        ph, pw = piece_patch.shape
        placed = False
        for _ in range(200):
            if H - ph - 1 < 1 or W - pw - 1 < 1:
                return None
            pr = rng.randint(1, H - ph - 1)
            pc = rng.randint(1, W - pw - 1)
            if not can_place(occ, pr, pc, piece_patch.shape, pad=1):
                continue
            reserve(occ, pr, pc, piece_patch.shape)
            paste(inp, piece_patch, pr, pc)
            spec["piece_pos"] = (int(pr), int(pc))
            placed = True
            break
        if not placed:
            return None

    expected = build_expected(specs, inp.shape)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid puzzle")
