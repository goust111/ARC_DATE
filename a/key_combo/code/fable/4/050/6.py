import random

import numpy as np
from scipy.ndimage import label


BG = 0
FRAME = 5
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_MAX_ATTEMPTS = 400
GEN_GRID_RANGE = (18, 30)
GEN_MASTER_HW = (3, 6)
GEN_MASTER_AREA = (5, 11)
GEN_TRUE_COUNT = (1, 3)
GEN_FAKE_COUNT = (1, 3)
GEN_BOX_MARGIN = (0, 2)
GEN_PADDING = 1
GEN_COMPONENT_ATTEMPTS = 120
GEN_PARTIAL_ATTEMPTS = 200


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def component_coords(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    lab, n = label(mask.astype(bool), structure=struct)
    return [np.argwhere(lab == k) for k in range(1, int(n) + 1)]


def is_connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    return int(mask.sum()) > 0 and len(component_coords(mask, connectivity=connectivity)) == 1


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    return crop_bbox(mask.astype(bool), box).astype(bool)


def rigid_orientations(mask: np.ndarray) -> list[np.ndarray]:
    mask = normalize_mask(mask)
    out: list[np.ndarray] = []
    seen = set()
    for flipped in (mask, np.fliplr(mask)):
        for k in range(4):
            orient = normalize_mask(np.rot90(flipped, k))
            key = (orient.shape, orient.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(orient)
    return out


def border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def inside_box(box, r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def find_frame_bbox(grid: np.ndarray):
    grey = grid == FRAME
    candidates = []
    for coords in component_coords(grey, connectivity=4):
        comp = np.zeros_like(grey, dtype=bool)
        comp[coords[:, 0], coords[:, 1]] = True
        box = bbox(comp)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        ring = border_mask(r1 - r0 + 1, c1 - c0 + 1)
        if not np.array_equal(crop_bbox(comp, box), ring):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not np.any((inner != BG) & (inner != FRAME)):
            continue
        candidates.append(box)
    assert len(candidates) == 1, "expected exactly one grey hollow frame"
    return candidates[0]


def extract_master_mask(grid: np.ndarray, frame_box) -> np.ndarray:
    r0, c0, r1, c1 = frame_box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != BG) & (inner != FRAME)
    box = bbox(mask)
    assert box is not None, "expected a master sprite inside the frame"
    return crop_bbox(mask.astype(bool), box).astype(bool)


def outside_components(grid: np.ndarray, frame_box):
    bg = get_bg(grid)
    work = grid.copy()
    r0, c0, r1, c1 = frame_box
    work[r0 : r1 + 1, c0 : c1 + 1] = bg
    comps = []
    for coords in component_coords(work != bg, connectivity=8):
        colors = {int(work[r, c]) for r, c in coords}
        assert len(colors) == 1, "each outside fragment must be monochromatic"
        comps.append((int(next(iter(colors))), coords))
    return comps


def local_completion_keys(partial_mask: np.ndarray, orientations: list[np.ndarray]) -> set[tuple[tuple[int, int], ...]]:
    partial = np.argwhere(partial_mask)
    if partial.size == 0:
        return set()
    partial_set = {tuple(map(int, xy)) for xy in partial}
    keys: set[tuple[tuple[int, int], ...]] = set()
    for orient in orientations:
        full = np.argwhere(orient)
        for pr, pc in partial_set:
            for fr, fc in full:
                dr, dc = int(pr - fr), int(pc - fc)
                placed = tuple(sorted((int(r + dr), int(c + dc)) for r, c in full))
                if partial_set.issubset(set(placed)):
                    keys.add(placed)
    return keys


def completion_candidates(
    coords: np.ndarray,
    orientations: list[np.ndarray],
    grid_shape: tuple[int, int],
    grid: np.ndarray,
    frame_box,
) -> list[np.ndarray]:
    H, W = grid_shape
    partial_set = {tuple(map(int, xy)) for xy in coords}
    out: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for orient in orientations:
        full = np.argwhere(orient)
        for pr, pc in partial_set:
            for fr, fc in full:
                dr, dc = int(pr - fr), int(pc - fc)
                placed = full + np.array([dr, dc], dtype=int)
                if placed[:, 0].min() < 0 or placed[:, 1].min() < 0:
                    continue
                if placed[:, 0].max() >= H or placed[:, 1].max() >= W:
                    continue
                placed_set = {tuple(map(int, xy)) for xy in placed}
                if not partial_set.issubset(placed_set):
                    continue
                blocked = False
                for r, c in placed_set:
                    if inside_box(frame_box, r, c):
                        blocked = True
                        break
                    if (r, c) not in partial_set and int(grid[r, c]) != BG:
                        blocked = True
                        break
                if blocked:
                    continue
                key = tuple(sorted(placed_set))
                out[key] = placed.copy()
    return list(out.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each outside fragment into the boxed master shape when it uniquely embeds, otherwise erase it.

    Input structure:
    - The background is black, with one grey hollow rectangle enclosing a complete monochrome master sprite.
    - Outside the grey box sit several monochromatic fragments on the background.
    - Each outside fragment is either a partial rigid copy of the master or a non-embeddable distractor.

    Rule:
    - Extract the master sprite inside the grey frame and enumerate its eight rigid orientations.
    - For each outside fragment, find all in-bounds placements of those orientations that cover every visible fragment pixel.
    - If exactly one completion exists, paint the whole completed master shape in that fragment's own color; otherwise erase the fragment to background.

    Key insight:
    - Completion and deletion are the two outcomes of the same unique-embedding test against the boxed master shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == BG, "expected black background"

    frame_box = find_frame_bbox(g)
    master = extract_master_mask(g, frame_box)
    orientations = rigid_orientations(master)
    assert orientations, "expected a non-empty master shape"

    out = g.copy()
    for color, coords in outside_components(g, frame_box):
        matches = completion_candidates(coords, orientations, g.shape, g, frame_box)
        if len(matches) == 1:
            for r, c in matches[0]:
                out[int(r), int(c)] = int(color)
        else:
            for r, c in coords:
                out[int(r), int(c)] = bg
    return out


def gen_connected_shape(h: int, w: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = normalize_mask(mask)
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return None
    if not is_connected(mask, connectivity=8):
        return None
    return mask


def gen_master_shape() -> np.ndarray | None:
    for _ in range(GEN_COMPONENT_ATTEMPTS):
        h = random.randint(*GEN_MASTER_HW)
        w = random.randint(*GEN_MASTER_HW)
        max_area = min(GEN_MASTER_AREA[1], h * w - 1)
        min_area = min(GEN_MASTER_AREA[0], max_area)
        if min_area < 5:
            continue
        mask = gen_connected_shape(h, w, random.randint(min_area, max_area))
        if mask is None:
            continue
        if len(rigid_orientations(mask)) != 8:
            continue
        return mask
    return None


def mask_to_patch(mask: np.ndarray, color: int, *, bg: int = BG) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def gen_partial_mask(full_mask: np.ndarray, orientations: list[np.ndarray]) -> np.ndarray | None:
    full_coords = np.argwhere(full_mask)
    area = len(full_coords)
    keep_min = max(3, area // 3)
    if keep_min >= area:
        keep_min = area - 1
    if keep_min < 2:
        return None
    for _ in range(GEN_PARTIAL_ATTEMPTS):
        keep_n = random.randint(keep_min, area - 1)
        idxs = random.sample(range(area), keep_n)
        part = np.zeros_like(full_mask, dtype=bool)
        chosen = full_coords[idxs]
        part[chosen[:, 0], chosen[:, 1]] = True
        if not is_connected(part, connectivity=8):
            continue
        if len(local_completion_keys(part, orientations)) != 1:
            continue
        return part
    return None


def gen_fake_mask(orientations: list[np.ndarray]) -> np.ndarray | None:
    for _ in range(GEN_COMPONENT_ATTEMPTS):
        h = random.randint(2, 6)
        w = random.randint(2, 6)
        max_area = min(7, h * w)
        mask = gen_connected_shape(h, w, random.randint(2, max_area))
        if mask is None:
            continue
        if local_completion_keys(mask, orientations):
            continue
        return mask
    return None


def can_place(occupied: np.ndarray, reserve_mask: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    H, W = occupied.shape
    coords = np.argwhere(reserve_mask)
    for dr, dc in coords:
        r = int(r0 + dr)
        c = int(c0 + dc)
        if not (0 <= r < H and 0 <= c < W):
            return False
        r1 = max(0, r - padding)
        r2 = min(H, r + padding + 1)
        c1 = max(0, c - padding)
        c2 = min(W, c + padding + 1)
        if np.any(occupied[r1:r2, c1:c2]):
            return False
    return True


def mark_occupied(occupied: np.ndarray, reserve_mask: np.ndarray, r0: int, c0: int) -> None:
    coords = np.argwhere(reserve_mask)
    occupied[r0 + coords[:, 0], c0 + coords[:, 1]] = True


def random_position(occupied: np.ndarray, reserve_mask: np.ndarray, padding: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    h, w = reserve_mask.shape
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if can_place(occupied, reserve_mask, r0, c0, padding):
            return int(r0), int(c0)
    return None


def build_frame_patch(master_mask: np.ndarray, master_color: int) -> tuple[np.ndarray, np.ndarray]:
    top_gap = random.randint(*GEN_BOX_MARGIN)
    bottom_gap = random.randint(*GEN_BOX_MARGIN)
    left_gap = random.randint(*GEN_BOX_MARGIN)
    right_gap = random.randint(*GEN_BOX_MARGIN)
    inner_h = master_mask.shape[0] + top_gap + bottom_gap
    inner_w = master_mask.shape[1] + left_gap + right_gap
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    colored_master = mask_to_patch(master_mask, master_color)
    paste(patch, colored_master, 1 + top_gap, 1 + left_gap, transparent=BG)
    reserve = np.ones_like(patch, dtype=bool)
    return patch, reserve


def gen_attempt() -> np.ndarray | None:
    master = gen_master_shape()
    if master is None:
        return None
    orientations = rigid_orientations(master)

    n_true = random.randint(*GEN_TRUE_COUNT)
    n_fake = random.randint(*GEN_FAKE_COUNT)
    total = n_true + n_fake
    master_color = random.choice(PALETTE)
    usable_colors = [c for c in PALETTE if c != master_color]
    if total > len(usable_colors):
        return None
    colors = random.sample(usable_colors, total)

    frame_patch, frame_reserve = build_frame_patch(master, master_color)

    objects = []
    for idx in range(n_true):
        orient = random.choice(orientations)
        partial = gen_partial_mask(orient, orientations)
        if partial is None:
            return None
        color = colors[idx]
        objects.append(
            {
                "kind": "true",
                "input_patch": mask_to_patch(partial, color),
                "reserve_mask": orient,
                "output_patch": mask_to_patch(orient, color),
            }
        )
    for idx in range(n_true, total):
        fake = gen_fake_mask(orientations)
        if fake is None:
            return None
        color = colors[idx]
        objects.append(
            {
                "kind": "fake",
                "input_patch": mask_to_patch(fake, color),
                "reserve_mask": fake,
                "output_patch": None,
            }
        )

    H = random.randint(*GEN_GRID_RANGE)
    W = random.randint(*GEN_GRID_RANGE)
    if frame_patch.shape[0] > H or frame_patch.shape[1] > W:
        return None

    occupied = np.zeros((H, W), dtype=bool)
    inp = np.full((H, W), BG, dtype=int)
    expected = np.full((H, W), BG, dtype=int)

    frame_pos = random_position(occupied, frame_reserve, padding=GEN_PADDING)
    if frame_pos is None:
        return None
    fr, fc = frame_pos
    paste(inp, frame_patch, fr, fc, transparent=BG)
    paste(expected, frame_patch, fr, fc, transparent=BG)
    mark_occupied(occupied, frame_reserve, fr, fc)

    random.shuffle(objects)
    objects.sort(key=lambda obj: int(obj["reserve_mask"].sum()), reverse=True)
    for obj in objects:
        pos = random_position(occupied, obj["reserve_mask"], padding=GEN_PADDING)
        if pos is None:
            return None
        r0, c0 = pos
        paste(inp, obj["input_patch"], r0, c0, transparent=BG)
        mark_occupied(occupied, obj["reserve_mask"], r0, c0)
        if obj["kind"] == "true":
            paste(expected, obj["output_patch"], r0, c0, transparent=BG)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
