import random
from collections import Counter

import numpy as np


BLACK = 0
YELLOW = 4
MAROON = 9
PALETTE = (1, 2, 3, 5, 6, 7, 8)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_SHAPES = {
    "right": {(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)},
    "left": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)},
    "up": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "down": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
}
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _stamp(out: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = out.shape
    ph, pw = patch.shape
    for rr in range(ph):
        for cc in range(pw):
            val = int(patch[rr, cc])
            if val == BLACK:
                continue
            tr = r0 + rr
            tc = c0 + cc
            if 0 <= tr < h and 0 <= tc < w:
                out[tr, tc] = val


def _parse_arrow(patch: np.ndarray) -> str:
    yellow = np.argwhere(patch == YELLOW)
    if yellow.shape[0] != 5:
        raise ValueError("object must contain exactly one 5-pixel arrow")
    r0, c0 = yellow.min(axis=0)
    norm = {(int(r - r0), int(c - c0)) for r, c in yellow}
    for direction, shape in ARROW_SHAPES.items():
        if norm == shape:
            return direction
    raise ValueError("unrecognized arrow shape")


def _object_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    vals = [int(grid[r, c]) for r, c in coords]
    body_counts = Counter(v for v in vals if v not in (BLACK, YELLOW, MAROON))
    if not body_counts:
        raise ValueError("missing body color")
    body_color = int(body_counts.most_common(1)[0][0])
    direction = _parse_arrow(patch)
    step = int(sum(v == MAROON for v in vals))
    if step <= 0:
        raise ValueError("missing maroon marker")
    move_span = patch.shape[1] if direction in ("left", "right") else patch.shape[0]
    return {
        "r0": int(r0),
        "c0": int(c0),
        "patch": patch,
        "mask": patch != BLACK,
        "body_color": body_color,
        "direction": direction,
        "step": step,
        "period": int(move_span + step),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate each stamped object along its internal arrow direction until a
    same-colored blocker line would be covered.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    h, w = g.shape

    comps = _components4(g != BLACK)
    objects: list[dict] = []
    blocker_pixels: dict[int, set[tuple[int, int]]] = {}
    static_components: list[np.ndarray] = []

    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        if YELLOW in colors:
            objects.append(_object_info(g, coords))
        else:
            static_components.append(coords)
            by_color = Counter(int(g[r, c]) for r, c in coords)
            if len(by_color) == 1:
                color = int(next(iter(by_color)))
                blocker_pixels.setdefault(color, set()).update(
                    (int(r), int(c)) for r, c in coords
                )

    objects.sort(key=lambda item: (item["r0"], item["c0"]))

    out = np.full_like(g, BLACK)
    for coords in static_components:
        for r, c in coords:
            out[int(r), int(c)] = int(g[int(r), int(c)])

    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        patch = obj["patch"]
        mask = obj["mask"]
        dr, dc = DIRS[obj["direction"]]
        period = int(obj["period"])
        same_color_blockers = blocker_pixels.get(int(obj["body_color"]), set())

        _stamp(out, patch, r0, c0)

        k = 1
        while True:
            nr = int(r0 + dr * period * k)
            nc = int(c0 + dc * period * k)
            blocked = False
            for rr, cc in np.argwhere(mask):
                tr = int(nr + rr)
                tc = int(nc + cc)
                if not (0 <= tr < h and 0 <= tc < w):
                    blocked = True
                    break
                if (tr, tc) in same_color_blockers:
                    blocked = True
                    break
            if blocked:
                break
            _stamp(out, patch, nr, nc)
            k += 1

    return out


def _build_patch(
    rng: np.random.Generator, body_color: int, direction: str, step: int
) -> np.ndarray | None:
    if direction in ("left", "right"):
        height = max(3, step) + int(rng.integers(1, 4))
        width = int(rng.integers(5, 9))
    else:
        height = int(rng.integers(5, 9))
        width = max(3, step) + int(rng.integers(1, 4))

    patch = np.full((height, width), body_color, dtype=int)
    arrow_top = int(rng.integers(0, height - 2))
    arrow_left = int(rng.integers(0, width - 2))
    arrow_cells = {(arrow_top + r, arrow_left + c) for r, c in ARROW_SHAPES[direction]}

    if direction in ("left", "right"):
        marker_cols = [c for c in range(width) if c < arrow_left or c > arrow_left + 2]
        if not marker_cols:
            return None
        marker_col = int(rng.choice(marker_cols))
        marker_top = int(rng.integers(0, height - step + 1))
        marker_cells = {(marker_top + i, marker_col) for i in range(step)}
    else:
        marker_rows = [r for r in range(height) if r < arrow_top or r > arrow_top + 2]
        if not marker_rows:
            return None
        marker_row = int(rng.choice(marker_rows))
        marker_left = int(rng.integers(0, width - step + 1))
        marker_cells = {(marker_row, marker_left + i) for i in range(step)}

    if arrow_cells & marker_cells:
        return None

    for r, c in arrow_cells:
        patch[r, c] = YELLOW
    for r, c in marker_cells:
        patch[r, c] = MAROON
    return patch


def _expand_rect(
    h: int, w: int, r0: int, c0: int, r1: int, c1: int, pad: int = 1
) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _can_use_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_spec(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    body_color: int,
    direction: str,
    copies: int,
) -> bool:
    h, w = grid.shape
    for _ in range(120):
        step = int(rng.integers(1, 5))
        patch = _build_patch(rng, body_color, direction, step)
        if patch is None:
            continue
        ph, pw = patch.shape
        period = (pw if direction in ("left", "right") else ph) + step
        ghost_index = copies + 1

        if direction == "right":
            max_c0 = w - pw - ghost_index * period
            if max_c0 < 0 or h - ph < 0:
                continue
            c0 = int(rng.integers(0, max_c0 + 1))
            r0 = int(rng.integers(0, h - ph + 1))
        elif direction == "left":
            min_c0 = ghost_index * period
            max_c0 = w - pw
            if min_c0 > max_c0 or h - ph < 0:
                continue
            c0 = int(rng.integers(min_c0, max_c0 + 1))
            r0 = int(rng.integers(0, h - ph + 1))
        elif direction == "down":
            max_r0 = h - ph - ghost_index * period
            if max_r0 < 0 or w - pw < 0:
                continue
            r0 = int(rng.integers(0, max_r0 + 1))
            c0 = int(rng.integers(0, w - pw + 1))
        else:
            min_r0 = ghost_index * period
            max_r0 = h - ph
            if min_r0 > max_r0 or w - pw < 0:
                continue
            r0 = int(rng.integers(min_r0, max_r0 + 1))
            c0 = int(rng.integers(0, w - pw + 1))

        rects = []
        ok = True
        for k in range(ghost_index + 1):
            rr = r0 + DIRS[direction][0] * period * k
            cc = c0 + DIRS[direction][1] * period * k
            br = _expand_rect(h, w, rr, cc, rr + ph - 1, cc + pw - 1, pad=1)
            if not _can_use_rect(reserved, *br):
                ok = False
                break
            rects.append(br)
        if not ok:
            continue

        block_r0 = r0 + DIRS[direction][0] * period * ghost_index
        block_c0 = c0 + DIRS[direction][1] * period * ghost_index
        if direction in ("left", "right"):
            line_row = int(rng.integers(block_r0, block_r0 + ph))
            line_len = int(rng.integers(2, pw + 1))
            line_c0 = int(rng.integers(block_c0, block_c0 + pw - line_len + 1))
            blocker = [(line_row, line_c0 + i) for i in range(line_len)]
        else:
            line_col = int(rng.integers(block_c0, block_c0 + pw))
            line_len = int(rng.integers(2, ph + 1))
            line_r0 = int(rng.integers(block_r0, block_r0 + ph - line_len + 1))
            blocker = [(line_r0 + i, line_col) for i in range(line_len)]

        for rr, cc in blocker:
            er0, ec0, er1, ec1 = _expand_rect(h, w, rr, cc, rr, cc, pad=1)
            if not _can_use_rect(reserved, er0, ec0, er1, ec1):
                ok = False
                break
        if not ok:
            continue

        _stamp(grid, patch, r0, c0)
        for rr, cc in blocker:
            grid[rr, cc] = body_color
        for br in rects:
            _mark_rect(reserved, *br)
        for rr, cc in blocker:
            er0, ec0, er1, ec1 = _expand_rect(h, w, rr, cc, rr, cc, pad=1)
            _mark_rect(reserved, er0, ec0, er1, ec1)
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    object_count = int(rng.integers(2, 5))
    if object_count > len(PALETTE):
        return None
    colors = list(rng.choice(PALETTE, size=object_count, replace=False))
    directions = list(rng.permutation(["up", "down", "left", "right"]))[:object_count]
    copies_list = [int(rng.integers(0, 4)) for _ in range(object_count)]
    if all(c == 0 for c in copies_list):
        copies_list[int(rng.integers(0, object_count))] = int(rng.integers(1, 4))

    specs = list(zip(colors, directions, copies_list))
    random.shuffle(specs)

    for color, direction, copies in specs:
        if not _place_spec(rng, grid, reserved, int(color), direction, int(copies)):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
