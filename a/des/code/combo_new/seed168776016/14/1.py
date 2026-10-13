import random
from collections import deque

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRECTIONS = ["up", "down", "left", "right"]
GEN_ATTEMPTS = 500

SHAPES_BY_AREA = {
    2: [
        ((0, 0), (0, 1)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ],
    5: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _transform_shape(
    shape: tuple[tuple[int, int], ...], turns: int, flip: bool
) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    for _ in range(turns % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    return _normalize_shape(tuple(pts))


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = {
        _transform_shape(shape, turns, flip)
        for turns in range(4)
        for flip in (False, True)
    }
    return sorted(variants)


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(list(shape))
    return r1 - r0 + 1, c1 - c0 + 1


def _frame_info(
    grid: np.ndarray, color: int, coords: list[tuple[int, int]], bg: int
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    coord_set = set(coords)
    boundary = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if not coord_set.issubset(boundary):
        return None

    missing = boundary - coord_set
    if len(missing) != 1:
        return None
    gap = next(iter(missing))
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if len(coord_set) != len(boundary) - 1:
        return None

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != bg:
                return None

    if gap[0] == r0:
        direction = "up"
    elif gap[0] == r1:
        direction = "down"
    elif gap[1] == c0:
        direction = "left"
    else:
        direction = "right"

    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
    }


def _component_on_side(
    coords: list[tuple[int, int]], frame_bbox: tuple[int, int, int, int], direction: str
) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    fr0, fc0, fr1, fc1 = frame_bbox
    if direction == "up":
        return r1 < fr0
    if direction == "down":
        return r0 > fr1
    if direction == "left":
        return c1 < fc0
    return c0 > fc1


def _scene_pairs(grid: np.ndarray, bg: int) -> list[dict]:
    pairs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components_mask(grid == color)
        frames: list[dict] = []
        others: list[list[tuple[int, int]]] = []
        for comp in comps:
            frame = _frame_info(grid, color, comp, bg)
            if frame is None:
                others.append(comp)
            else:
                frames.append(frame)
        if not frames:
            continue
        if len(frames) != 1:
            raise ValueError(f"expected one frame for color {color}, got {len(frames)}")
        frame = frames[0]
        fragments = [comp for comp in others if _component_on_side(comp, frame["bbox"], frame["direction"])]
        if len(fragments) != 1:
            raise ValueError(f"expected one fragment for color {color}, got {len(fragments)}")
        pairs.append(
            {
                "color": color,
                "frame": frame,
                "fragment": {
                    "coords": fragments[0],
                    "bbox": _bbox(fragments[0]),
                },
            }
        )
    return pairs


def _translated_fragment(
    coords: list[tuple[int, int]], frame: dict
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    fr0, fc0, fr1, fc1 = _bbox(coords)
    direction = frame["direction"]

    if direction == "up":
        dr, dc = (r0 + 1) - fr0, 0
    elif direction == "down":
        dr, dc = (r1 - 1) - fr1, 0
    elif direction == "left":
        dr, dc = 0, (c0 + 1) - fc0
    else:
        dr, dc = 0, (c1 - 1) - fc1

    moved = [(r + dr, c + dc) for r, c in coords]
    for r, c in moved:
        if not (r0 < r < r1 and c0 < c < c1):
            raise ValueError("translated fragment must end strictly inside the frame")
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each same-color outside fragment through its frame gap and place it just inside the hollow rectangle.

    Input structure:
    - The background is black.
    - Several differently colored hollow rectangular frames appear, each drawn with a 1-pixel border and one 1-pixel gap on exactly one side.
    - Each frame interior is empty background.
    - A single small connected fragment of the same color sits outside the frame on the side pointed to by the gap.

    Rule:
    - For each colored frame, find the missing border pixel to determine the gap side.
    - Find the same-colored outside fragment on that side of the frame.
    - Translate the fragment straight inward, opposite the gap's outward direction, until its tight bounding box is fully inside the frame interior.
    - Erase the fragment from its old location while leaving the frame border and the gap itself unchanged.

    Key insight:
    - The missing border pixel is an entrance marker: it identifies which external same-color fragment gets absorbed and the only direction in which it should slide.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    pairs = _scene_pairs(g, bg)
    for pair in pairs:
        for r, c in pair["fragment"]["coords"]:
            out[r, c] = bg
    for pair in pairs:
        moved = _translated_fragment(pair["fragment"]["coords"], pair["frame"])
        for r, c in moved:
            out[r, c] = pair["color"]

    return out


def _random_shape(
    rng: random.Random, inner_h: int, inner_w: int
) -> tuple[tuple[int, int], ...]:
    area = rng.randint(2, 5)
    variants: list[tuple[tuple[int, int], ...]] = []
    for base in SHAPES_BY_AREA[area]:
        for variant in _shape_variants(base):
            h, w = _shape_size(variant)
            if h <= inner_h and w <= inner_w:
                variants.append(variant)
    if not variants:
        raise ValueError("no fragment variant fits the frame interior")
    return rng.choice(variants)


def _build_object_patch(
    rng: random.Random, color: int, direction: str
) -> np.ndarray:
    inner_h = rng.randint(3, 5)
    inner_w = rng.randint(3, 5)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    shape = _random_shape(rng, inner_h, inner_w)
    shape_h, shape_w = _shape_size(shape)
    sep = rng.randint(2, 4)

    if direction in ("left", "right"):
        side_space = shape_w + sep - 1
        patch_h = frame_h
        patch_w = frame_w + side_space
        frame_r0 = 0
        frame_c0 = side_space if direction == "left" else 0
        row_offset = rng.randint(0, inner_h - shape_h)
        frag_r0 = frame_r0 + 1 + row_offset
        if direction == "left":
            frag_c0 = frame_c0 - sep - shape_w + 1
            gap_c = frame_c0
        else:
            frag_c0 = frame_c0 + frame_w - 1 + sep
            gap_c = frame_c0 + frame_w - 1
        gap_r = frame_r0 + 1 + row_offset + rng.choice(sorted({r for r, _ in shape}))
    else:
        side_space = shape_h + sep - 1
        patch_h = frame_h + side_space
        patch_w = frame_w
        frame_r0 = side_space if direction == "up" else 0
        frame_c0 = 0
        col_offset = rng.randint(0, inner_w - shape_w)
        frag_c0 = frame_c0 + 1 + col_offset
        if direction == "up":
            frag_r0 = frame_r0 - sep - shape_h + 1
            gap_r = frame_r0
        else:
            frag_r0 = frame_r0 + frame_h - 1 + sep
            gap_r = frame_r0 + frame_h - 1
        gap_c = frame_c0 + 1 + col_offset + rng.choice(sorted({c for _, c in shape}))

    patch = np.full((patch_h, patch_w), BG, dtype=int)
    patch[frame_r0, frame_c0:frame_c0 + frame_w] = color
    patch[frame_r0 + frame_h - 1, frame_c0:frame_c0 + frame_w] = color
    patch[frame_r0:frame_r0 + frame_h, frame_c0] = color
    patch[frame_r0:frame_r0 + frame_h, frame_c0 + frame_w - 1] = color
    patch[gap_r, gap_c] = BG

    for sr, sc in shape:
        patch[frag_r0 + sr, frag_c0 + sc] = color

    return patch


def _layout_options(n_objects: int) -> list[tuple[int, int]]:
    if n_objects == 2:
        return [(1, 2), (2, 1)]
    if n_objects == 3:
        return [(2, 2), (1, 3), (3, 1)]
    return [(2, 2)]


def _distribute_extra(rng: random.Random, total: int, buckets: int) -> list[int]:
    dist = [0] * buckets
    for _ in range(total):
        dist[rng.randrange(buckets)] += 1
    return dist


def _assemble_grid(rng: random.Random, patches: list[np.ndarray]) -> np.ndarray | None:
    n_objects = len(patches)
    options = _layout_options(n_objects)
    rng.shuffle(options)

    for rows, cols in options:
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        order = list(patches)
        rng.shuffle(order)
        chosen_slots = slots[:]
        rng.shuffle(chosen_slots)
        assignment = {slot: patch for slot, patch in zip(chosen_slots, order)}

        row_heights = [0] * rows
        col_widths = [0] * cols
        for (r, c), patch in assignment.items():
            ph, pw = patch.shape
            row_heights[r] = max(row_heights[r], ph)
            col_widths[c] = max(col_widths[c], pw)
        if any(v == 0 for v in row_heights) or any(v == 0 for v in col_widths):
            continue

        min_h = 2 + sum(row_heights) + (rows - 1)
        min_w = 2 + sum(col_widths) + (cols - 1)
        if min_h > 30 or min_w > 30:
            continue

        extra_h = rng.randint(0, 30 - min_h)
        extra_w = rng.randint(0, 30 - min_w)
        row_gaps = [1] * max(0, rows - 1)
        col_gaps = [1] * max(0, cols - 1)
        h_dist = _distribute_extra(rng, extra_h, 2 + rows + len(row_gaps))
        w_dist = _distribute_extra(rng, extra_w, 2 + cols + len(col_gaps))

        top = 1 + h_dist[0]
        bottom = 1 + h_dist[1]
        for i in range(rows):
            row_heights[i] += h_dist[2 + i]
        for i in range(len(row_gaps)):
            row_gaps[i] += h_dist[2 + rows + i]

        left = 1 + w_dist[0]
        right = 1 + w_dist[1]
        for i in range(cols):
            col_widths[i] += w_dist[2 + i]
        for i in range(len(col_gaps)):
            col_gaps[i] += w_dist[2 + cols + i]

        h = top + bottom + sum(row_heights) + sum(row_gaps)
        w = left + right + sum(col_widths) + sum(col_gaps)
        if h > 30 or w > 30:
            continue

        row_starts: list[int] = []
        cur = top
        for i in range(rows):
            row_starts.append(cur)
            cur += row_heights[i]
            if i < len(row_gaps):
                cur += row_gaps[i]

        col_starts: list[int] = []
        cur = left
        for i in range(cols):
            col_starts.append(cur)
            cur += col_widths[i]
            if i < len(col_gaps):
                cur += col_gaps[i]

        grid = np.full((h, w), BG, dtype=int)
        for (slot_r, slot_c), patch in assignment.items():
            ph, pw = patch.shape
            max_r_off = row_heights[slot_r] - ph
            max_c_off = col_widths[slot_c] - pw
            rr = row_starts[slot_r] + rng.randint(0, max_r_off)
            cc = col_starts[slot_c] + rng.randint(0, max_c_off)
            grid[rr:rr + ph, cc:cc + pw] = patch

        return grid

    return None


def _validate_generated(grid: np.ndarray, expected_objects: int) -> bool:
    try:
        bg = _get_bg(grid)
        pairs = _scene_pairs(grid, bg)
        if len(pairs) != expected_objects:
            return False
        out = main(grid)
        for pair in pairs:
            color = pair["color"]
            gap_r, gap_c = pair["frame"]["gap"]
            if int(out[gap_r, gap_c]) != bg:
                return False
            for r, c in pair["fragment"]["coords"]:
                if int(out[r, c]) != bg:
                    return False
            moved = _translated_fragment(pair["fragment"]["coords"], pair["frame"])
            for r, c in moved:
                if int(out[r, c]) != color:
                    return False
    except Exception:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(2, 4)
    colors = rng.sample(COLORS, n_objects)
    directions = rng.sample(DIRECTIONS, n_objects)
    patches = [_build_object_patch(rng, color, direction) for color, direction in zip(colors, directions)]
    grid = _assemble_grid(rng, patches)
    if grid is None:
        return None
    if not _validate_generated(grid, n_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid scene")
