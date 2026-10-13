
from collections import defaultdict
import numpy as np

BG = 0
GRAY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "right", "bottom", "left")
ROTATIONS = {"top": 0, "right": 1, "bottom": 2, "left": 3}


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    rows, cols = mask.shape
    seen = np.zeros((rows, cols), dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in DIRS4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _normalize(cells):
    cells = list(cells)
    if not cells:
        return tuple()
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_size(cells):
    if not cells:
        return 0, 0
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _rotate_cells(cells, turns):
    cells = _normalize(cells)
    for _ in range(turns % 4):
        h, _ = _shape_size(cells)
        cells = _normalize((c, h - 1 - r) for r, c in cells)
    return cells


def _dominant_visible_color(values):
    counts = {}
    for value in np.asarray(values).ravel():
        value = int(value)
        if value in (BG, GRAY):
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _side_colors(crop):
    h, w = crop.shape
    left_slice = crop[1:-1, 0] if h > 2 else crop[:, 0]
    right_slice = crop[1:-1, w - 1] if h > 2 else crop[:, w - 1]
    return {
        "top": _dominant_visible_color(crop[0, :]),
        "right": _dominant_visible_color(right_slice),
        "bottom": _dominant_visible_color(crop[h - 1, :]),
        "left": _dominant_visible_color(left_slice),
    }


def _hole_side(hole, crop):
    h, w = crop.shape
    contacts = {side: 0 for side in SIDES}
    for r, c in hole:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or crop[nr, nc] == BG:
                continue
            if nr == 0:
                contacts["top"] += 1
            elif nr == h - 1:
                contacts["bottom"] += 1
            elif nc == 0:
                contacts["left"] += 1
            elif nc == w - 1:
                contacts["right"] += 1
    side = max(contacts, key=contacts.get)
    if contacts[side]:
        return side
    return min(
        {
            "top": min(r for r, _ in hole),
            "right": w - 1 - max(c for _, c in hole),
            "bottom": h - 1 - max(r for r, _ in hole),
            "left": min(c for _, c in hole),
        },
        key=lambda name: {
            "top": min(r for r, _ in hole),
            "right": w - 1 - max(c for _, c in hole),
            "bottom": h - 1 - max(r for r, _ in hole),
            "left": min(c for _, c in hole),
        }[name],
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Fill rotated side-matched holes in a framed gray template and crop it.

    Input structure:
    The grid has black background, one large gray rectangular template, and four
    differently colored one-pixel border sides. Black connected holes sit inside
    the gray template, each 4-adjacent to exactly one colored side. Outside the
    template are loose solid-color fragments whose colors match the border sides.
    Rule:
    Locate the gray template component and crop to its bounding rectangle. For
    each black hole in that crop, read the adjacent side color, find a loose
    fragment of that color, and compare the fragment after the side-coded
    rotation: top is 0 degrees, right is 90 degrees clockwise, bottom is 180
    degrees, and left is 270 degrees clockwise. Fill the hole with gray and
    discard everything outside the cropped template.
    Key insight:
    The fragment color selects the target side, while the side position encodes
    the rotation needed before the piece fits the hole.
    """
    grid = np.asarray(grid)
    comps = _components(grid != BG)
    if not comps:
        return grid.copy()

    gray_candidates = []
    for idx, comp in enumerate(comps):
        if any(grid[r, c] == GRAY for r, c in comp):
            gray_candidates.append((idx, comp))
    if gray_candidates:
        template_idx, template = max(gray_candidates, key=lambda item: len(item[1]))
    else:
        template_idx, template = max(enumerate(comps), key=lambda item: len(item[1]))

    min_r = min(r for r, _ in template)
    max_r = max(r for r, _ in template)
    min_c = min(c for _, c in template)
    max_c = max(c for _, c in template)
    crop = grid[min_r:max_r + 1, min_c:max_c + 1].copy()

    colors_by_side = _side_colors(crop)
    fragments_by_color = defaultdict(list)
    for idx, comp in enumerate(comps):
        if idx == template_idx:
            continue
        values = [int(grid[r, c]) for r, c in comp if grid[r, c] != BG]
        if not values:
            continue
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        color = max(counts, key=counts.get)
        fragments_by_color[color].append(_normalize(comp))

    used_fragments = defaultdict(set)
    for hole in _components(crop == BG):
        side = _hole_side(hole, crop)
        color = colors_by_side.get(side)
        hole_shape = _normalize(hole)
        for idx, frag_shape in enumerate(fragments_by_color.get(color, [])):
            if idx in used_fragments[color]:
                continue
            if _rotate_cells(frag_shape, ROTATIONS[side]) == hole_shape:
                used_fragments[color].add(idx)
                break
        for r, c in hole:
            crop[r, c] = GRAY
    return crop


def _random_polyomino(rng):
    for _ in range(200):
        target = int(rng.integers(3, 7))
        cells = {(0, 0)}
        while len(cells) < target:
            base = list(cells)[int(rng.integers(0, len(cells)))]
            dr, dc = DIRS4[int(rng.integers(0, 4))]
            cells.add((base[0] + dr, base[1] + dc))
        shape = _normalize(cells)
        h, w = _shape_size(shape)
        if h <= 4 and w <= 4:
            return shape
    return ((0, 0), (0, 1), (1, 0))


def _near_existing(cells, blocked, gap=1):
    if not blocked:
        return False
    for r, c in cells:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                if (r + dr, c + dc) in blocked:
                    return True
    return False


def _candidate_hole_origin(rng, side, shape, template_h, template_w):
    h, w = _shape_size(shape)
    if side == "top":
        max_c = template_w - w - 2
        if max_c < 2:
            return None
        return 1, int(rng.integers(2, max_c + 1))
    if side == "bottom":
        max_c = template_w - w - 2
        if max_c < 2:
            return None
        return template_h - h - 1, int(rng.integers(2, max_c + 1))
    if side == "left":
        max_r = template_h - h - 2
        if max_r < 2:
            return None
        return int(rng.integers(2, max_r + 1)), 1
    max_r = template_h - h - 2
    if max_r < 2:
        return None
    return int(rng.integers(2, max_r + 1)), template_w - w - 1


def _build_case(rng):
    rows = int(rng.integers(24, 31))
    cols = int(rng.integers(24, 31))
    template_h = int(rng.integers(12, min(18, rows - 6) + 1))
    template_w = int(rng.integers(12, min(18, cols - 6) + 1))
    top = int(rng.integers(1, rows - template_h))
    left = int(rng.integers(1, cols - template_w))

    palette = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
    picked = [int(x) for x in rng.choice(palette, size=4, replace=False)]
    side_color = dict(zip(SIDES, picked))

    template = np.full((template_h, template_w), GRAY, dtype=int)
    template[1:-1, 0] = side_color["left"]
    template[1:-1, template_w - 1] = side_color["right"]
    template[0, :] = side_color["top"]
    template[template_h - 1, :] = side_color["bottom"]

    holes = []
    occupied_holes = set()
    side_order = list(SIDES)
    rng.shuffle(side_order)
    for side in side_order:
        placed = False
        for _ in range(300):
            piece_shape = _random_polyomino(rng)
            hole_shape = _rotate_cells(piece_shape, ROTATIONS[side])
            origin = _candidate_hole_origin(rng, side, hole_shape, template_h, template_w)
            if origin is None:
                continue
            r0, c0 = origin
            cells = {(r0 + r, c0 + c) for r, c in hole_shape}
            if any(r <= 0 or c <= 0 or r >= template_h - 1 or c >= template_w - 1 for r, c in cells):
                continue
            if _near_existing(cells, occupied_holes, 1):
                continue
            occupied_holes.update(cells)
            for r, c in cells:
                template[r, c] = BG
            holes.append({"side": side, "piece": piece_shape})
            placed = True
            break
        if not placed:
            raise RuntimeError("could not place all holes")

    grid = np.zeros((rows, cols), dtype=int)
    grid[top:top + template_h, left:left + template_w] = template

    reserved = set()
    for r in range(max(0, top - 1), min(rows, top + template_h + 1)):
        for c in range(max(0, left - 1), min(cols, left + template_w + 1)):
            reserved.add((r, c))

    fragments = holes[:]
    rng.shuffle(fragments)
    occupied_fragments = set()
    for item in fragments:
        shape = item["piece"]
        h, w = _shape_size(shape)
        placed = False
        for _ in range(1000):
            r0 = int(rng.integers(0, rows - h + 1))
            c0 = int(rng.integers(0, cols - w + 1))
            cells = {(r0 + r, c0 + c) for r, c in shape}
            if cells & reserved:
                continue
            if _near_existing(cells, occupied_fragments, 1):
                continue
            if any(grid[r, c] != BG for r, c in cells):
                continue
            color = side_color[item["side"]]
            for r, c in cells:
                grid[r, c] = color
            occupied_fragments.update(cells)
            placed = True
            break
        if not placed:
            raise RuntimeError("could not place fragments")
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(200):
        try:
            return _build_case(rng)
        except RuntimeError:
            pass
    raise RuntimeError("failed to generate a valid ARC-style input")
