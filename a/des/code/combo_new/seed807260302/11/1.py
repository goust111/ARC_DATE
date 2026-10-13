from collections import deque

import numpy as np


BG = 0
MARKER = 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _is_complete_rect_frame(pts: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    pset = set(pts)
    border_count = 2 * height + 2 * width - 4
    if len(pset) != border_count:
        return False

    for c in range(c0, c1 + 1):
        if (r0, c) not in pset or (r1, c) not in pset:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pset or (r, c1) not in pset:
            return False
    return True


def _marker_side(
    marker: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    r, c = marker
    r0, c0, r1, c1 = bbox
    sides: list[str] = []
    if r == r0 + 1:
        sides.append("top")
    if r == r1 - 1:
        sides.append("bottom")
    if c == c0 + 1:
        sides.append("left")
    if c == c1 - 1:
        sides.append("right")
    if len(sides) != 1:
        return None

    side = sides[0]
    if side == "top":
        return (1, 0)
    if side == "bottom":
        return (-1, 0)
    if side == "left":
        return (0, 1)
    return (0, -1)


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, MARKER):
            continue
        for pts in _components(grid == color):
            bbox = _bbox(pts)
            if not _is_complete_rect_frame(pts, bbox):
                continue
            r0, c0, r1, c1 = bbox
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            marker_locs = np.argwhere(interior == MARKER)
            if marker_locs.shape[0] != 1:
                continue

            mr = int(marker_locs[0, 0] + r0 + 1)
            mc = int(marker_locs[0, 1] + c0 + 1)
            direction = _marker_side((mr, mc), bbox)
            if direction is None:
                continue

            obj_pts: list[tuple[int, int, int]] = []
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    if (rr, cc) == (mr, mc):
                        continue
                    v = int(grid[rr, cc])
                    if v != BG and v != color:
                        obj_pts.append((rr, cc, v))
            if not obj_pts:
                continue

            frames.append(
                {
                    "bbox": bbox,
                    "color": color,
                    "marker": (mr, mc),
                    "direction": direction,
                    "objects": obj_pts,
                }
            )
    frames.sort(key=lambda f: f["bbox"])  # type: ignore[index]
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the contents of each closed fence out through the side opposite its red marker.

    Input structure:
    - The background is black, with one or more complete one-pixel rectangular frames in a non-background color.
    - Each frame interior contains at least one small foreground object and exactly one red pixel.
    - The red pixel touches exactly one inner wall of its frame; frames do not overlap and there is enough outside space.

    Rule:
    - For each frame, the red marker's inner-wall side gives the blocked side, so the objects move in the opposite cardinal direction.
    - Move every non-marker foreground pixel inside that frame as one rigid group until the group's leading bbox edge is just outside the frame's outer wall.
    - Clear the frame interior, including the red marker, while keeping the frame border unchanged.

    Key insight: The red pixel is only a direction marker; it is not part of the moving object group.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    moves: list[tuple[list[tuple[int, int, int]], int, int]] = []

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        dr, dc = frame["direction"]  # type: ignore[misc]
        objects = frame["objects"]  # type: ignore[assignment]

        rows = [p[0] for p in objects]
        cols = [p[1] for p in objects]
        or0, oc0, or1, oc1 = min(rows), min(cols), max(rows), max(cols)

        shift_r = 0
        shift_c = 0
        if dr == 1:
            shift_r = r1 + 1 - or0
        elif dr == -1:
            shift_r = r0 - 1 - or1
        elif dc == 1:
            shift_c = c1 + 1 - oc0
        elif dc == -1:
            shift_c = c0 - 1 - oc1

        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        moves.append((objects, shift_r, shift_c))

    h, w = out.shape
    for objects, shift_r, shift_c in moves:
        for r, c, color in objects:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(color)

    return out


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
)


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    cells = [tuple(p) for p in _SHAPES[int(rng.integers(0, len(_SHAPES)))]]
    if bool(rng.integers(0, 2)):
        max_c = max(c for _, c in cells)
        cells = [(r, max_c - c) for r, c in cells]
    if bool(rng.integers(0, 2)):
        max_r = max(r for r, _ in cells)
        cells = [(max_r - r, c) for r, c in cells]
    if bool(rng.integers(0, 2)):
        cells = [(c, r) for r, c in cells]
    return _normalize(cells)


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _marker_for_direction(
    rng: np.random.Generator, frame_h: int, frame_w: int, direction: tuple[int, int]
) -> tuple[int, int]:
    dr, dc = direction
    if dr == -1:
        return frame_h - 2, int(rng.integers(2, frame_w - 2))
    if dr == 1:
        return 1, int(rng.integers(2, frame_w - 2))
    if dc == -1:
        return int(rng.integers(2, frame_h - 2)), frame_w - 2
    return int(rng.integers(2, frame_h - 2)), 1


def _make_objects(
    rng: np.random.Generator,
    frame_h: int,
    frame_w: int,
    marker: tuple[int, int],
    color_pool: tuple[int, ...],
) -> list[tuple[list[tuple[int, int]], int]]:
    object_cells: set[tuple[int, int]] = set()
    objects: list[tuple[list[tuple[int, int]], int]] = []
    wanted = int(rng.integers(1, 4))

    for _ in range(wanted):
        placed = False
        for _try in range(80):
            shape = _random_shape(rng)
            sh, sw = _shape_dims(shape)
            if sh > frame_h - 2 or sw > frame_w - 2:
                continue
            top = int(rng.integers(1, frame_h - sh))
            left = int(rng.integers(1, frame_w - sw))
            cells = [(top + r, left + c) for r, c in shape]
            if marker in cells:
                continue
            if any(cell in object_cells for cell in cells):
                continue

            too_close = False
            for r, c in cells:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if (rr, cc) in object_cells:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    break
            if too_close:
                continue

            color = int(color_pool[int(rng.integers(0, len(color_pool)))])
            object_cells.update(cells)
            objects.append((cells, color))
            placed = True
            break
        if not placed and not objects:
            for rr in range(1, frame_h - 1):
                for cc in range(1, frame_w - 1):
                    if (rr, cc) != marker:
                        color = int(color_pool[int(rng.integers(0, len(color_pool)))])
                        objects.append(([(rr, cc)], color))
                        return objects
    return objects


def _build_spec(
    rng: np.random.Generator,
    frame_color: int,
    direction: tuple[int, int],
    color_pool: tuple[int, ...],
) -> dict[str, object] | None:
    frame_h = int(rng.integers(6, 10))
    frame_w = int(rng.integers(6, 10))
    marker = _marker_for_direction(rng, frame_h, frame_w, direction)
    objects = _make_objects(rng, frame_h, frame_w, marker, color_pool)
    if not objects:
        return None

    obj_pixels: list[tuple[int, int, int]] = []
    for cells, color in objects:
        for r, c in cells:
            obj_pixels.append((r, c, color))

    obj_rows = [r for r, _, _ in obj_pixels]
    obj_cols = [c for _, c, _ in obj_pixels]
    or0, oc0, or1, oc1 = min(obj_rows), min(obj_cols), max(obj_rows), max(obj_cols)
    dr, dc = direction
    shift_r = 0
    shift_c = 0
    if dr == 1:
        shift_r = frame_h - or0
    elif dr == -1:
        shift_r = -1 - or1
    elif dc == 1:
        shift_c = frame_w - oc0
    elif dc == -1:
        shift_c = -1 - oc1

    target_pixels = [(r + shift_r, c + shift_c) for r, c, _ in obj_pixels]
    frame_area = [(r, c) for r in range(frame_h) for c in range(frame_w)]
    all_reserved = frame_area + target_pixels
    min_r = min(r for r, _ in all_reserved)
    min_c = min(c for _, c in all_reserved)
    max_r = max(r for r, _ in all_reserved)
    max_c = max(c for _, c in all_reserved)
    off_r = -min_r
    off_c = -min_c

    input_pixels: list[tuple[int, int, int]] = []
    for c in range(frame_w):
        input_pixels.append((off_r, off_c + c, frame_color))
        input_pixels.append((off_r + frame_h - 1, off_c + c, frame_color))
    for r in range(1, frame_h - 1):
        input_pixels.append((off_r + r, off_c, frame_color))
        input_pixels.append((off_r + r, off_c + frame_w - 1, frame_color))
    input_pixels.append((off_r + marker[0], off_c + marker[1], MARKER))
    for r, c, color in obj_pixels:
        input_pixels.append((off_r + r, off_c + c, color))

    return {
        "height": max_r - min_r + 1,
        "width": max_c - min_c + 1,
        "input_pixels": input_pixels,
    }


def _place_spec(
    grid: np.ndarray, reserved: np.ndarray, spec: dict[str, object], rng: np.random.Generator
) -> bool:
    h, w = grid.shape
    ph = int(spec["height"])
    pw = int(spec["width"])
    if ph > h or pw > w:
        return False

    for _ in range(150):
        top = int(rng.integers(0, h - ph + 1))
        left = int(rng.integers(0, w - pw + 1))
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + ph + 1)
        c1 = min(w, left + pw + 1)
        if np.any(reserved[r0:r1, c0:c1]):
            continue

        for rr, cc, color in spec["input_pixels"]:  # type: ignore[union-attr]
            grid[top + int(rr), left + int(cc)] = int(color)
        reserved[top : top + ph, left : left + pw] = True
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    frame_choices = tuple(c for c in range(1, 10) if c != MARKER)
    frame_color = int(frame_choices[int(rng.integers(0, len(frame_choices)))])
    object_colors = tuple(c for c in range(1, 10) if c not in (MARKER, frame_color))

    frame_count = int(rng.integers(1, 4))
    dir_indices = rng.permutation(len(_DIRECTIONS))[:frame_count]
    directions = [_DIRECTIONS[int(i)] for i in dir_indices]

    for direction in directions:
        spec = _build_spec(rng, frame_color, direction, object_colors)
        if spec is None or not _place_spec(grid, reserved, spec, rng):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-marker input")
