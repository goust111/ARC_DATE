import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _mask(rows):
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


BASE_KEYS = [
    _mask(["1"]),
    _mask(["11"]),
    _mask(["111"]),
    _mask(["10", "11"]),
    _mask(["11", "11"]),
    _mask(["111", "010"]),
    _mask(["110", "011"]),
    _mask(["010", "111", "010"]),
    _mask(["110", "111"]),
]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_local_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _shape_signature(mask: np.ndarray) -> tuple:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((rot.shape, tuple(int(x) for x in rot.ravel())))
    return min(variants)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in np.argwhere(mask):
        out[r : r + 3, c : c + 3] = True
    return out


def _make_lock(key: np.ndarray) -> np.ndarray:
    dilated = _dilate8(key)
    core = dilated[1 : 1 + key.shape[0], 1 : 1 + key.shape[1]]
    core[key] = False
    return dilated


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        col = grid[:, c]
        if col[0] == BG:
            continue
        if np.all(col == col[0]) and np.any(grid[:, :c] != BG) and np.any(grid[:, c + 1 :] != BG):
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, "expected a unique full-height separator column"
    return candidates[0]


def _extract_single_hole(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    holes = (~mask) & (~exterior)
    comps = _components(holes)
    assert len(comps) == 1, "each object must contain exactly one enclosed hole"
    hole_mask, _ = _coords_to_local_mask(comps[0])
    return hole_mask


def _match_rotation(key: np.ndarray, hole: np.ndarray) -> int | None:
    for k in range(4):
        if np.array_equal(np.rot90(key, k), hole):
            return k
    return None


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    assert 0 <= top and 0 <= left and top + h <= grid.shape[0] and left + w <= grid.shape[1]
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each main-region object by the lock paired with its internal hole shape.

    The template library sits to the left of a full-height grey separator. For each
    library color there are exactly two monochrome components: a key on the left and
    a lock on the right. A main-region object's single enclosed black hole is matched
    against the keys up to rotation; the same rotation is then applied to the paired
    lock before drawing it in the object's color with bbox centers aligned.
    """
    g = np.asarray(grid, dtype=int)
    sep_col, sep_color = _find_separator(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert sep_color != BG

    templates = []
    for color in sorted(int(x) for x in np.unique(left) if x != BG):
        comp_masks = []
        for coords in _components(left == color):
            local_mask, bbox = _coords_to_local_mask(coords)
            comp_masks.append((bbox, local_mask))
        assert len(comp_masks) == 2, "each template color must produce one key and one lock"
        comp_masks.sort(key=lambda item: (item[0][1], item[0][0]))
        templates.append({"key": comp_masks[0][1], "lock": comp_masks[1][1]})

    assert templates, "template library must be non-empty"

    replacements = []
    clear_cells = []
    for color in sorted(int(x) for x in np.unique(right) if x != BG):
        for coords in _components(right == color):
            obj_mask, (r0, c0, r1, c1) = _coords_to_local_mask(coords)
            hole = _extract_single_hole(obj_mask)
            match = None
            for tpl in templates:
                rot = _match_rotation(tpl["key"], hole)
                if rot is None:
                    continue
                assert match is None, "hole shape should match exactly one template"
                match = np.rot90(tpl["lock"], rot)
            assert match is not None, "every hole must match a template key"
            top_num = int(r0 + r1 - (match.shape[0] - 1))
            left_num = int(c0 + c1 - (match.shape[1] - 1))
            assert top_num % 2 == 0 and left_num % 2 == 0, "center alignment must land on integer cells"
            replacements.append((top_num // 2, sep_col + 1 + left_num // 2, match, color))
            clear_cells.append((coords[:, 0], coords[:, 1] + sep_col + 1))

    out = g.copy()
    for rows, cols in clear_cells:
        out[rows, cols] = BG
    for top, left_col, mask, color in replacements:
        _draw_mask(out, mask, top, left_col, color)
    return out


def _carve_notch(mask: np.ndarray, side: str, limit: int, rng: random.Random) -> None:
    if limit <= 0:
        return
    h, w = mask.shape
    if side in ("top", "bottom"):
        if w < 3:
            return
        depth = rng.randint(1, limit)
        start = rng.randint(1, w - 2)
        end = rng.randint(start + 1, w - 1)
        if side == "top":
            mask[:depth, start:end] = False
        else:
            mask[h - depth :, start:end] = False
    else:
        if h < 3:
            return
        depth = rng.randint(1, limit)
        start = rng.randint(1, h - 2)
        end = rng.randint(start + 1, h - 1)
        if side == "left":
            mask[start:end, :depth] = False
        else:
            mask[start:end, w - depth :] = False


def _make_object_mask(hole: np.ndarray, rng: random.Random) -> np.ndarray:
    hh, ww = hole.shape
    for _ in range(80):
        pad_r = rng.randint(2, 3)
        pad_c = rng.randint(2, 3)
        mask = np.ones((hh + 2 * pad_r, ww + 2 * pad_c), dtype=bool)
        center = mask[pad_r : pad_r + hh, pad_c : pad_c + ww]
        center[hole] = False
        for side in ("top", "bottom", "left", "right"):
            if rng.random() < 0.75:
                _carve_notch(mask, side, (pad_r - 1) if side in ("top", "bottom") else (pad_c - 1), rng)
        if rng.random() < 0.35:
            side = rng.choice(("top", "bottom", "left", "right"))
            _carve_notch(mask, side, (pad_r - 1) if side in ("top", "bottom") else (pad_c - 1), rng)
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(_components(mask)) != 1:
            continue
        try:
            recovered = _extract_single_hole(mask)
        except AssertionError:
            continue
        if np.array_equal(recovered, hole):
            return mask
    raise RuntimeError("failed to build a valid object shell")


def _sample_templates(rng: random.Random, count: int) -> list[dict]:
    shapes = BASE_KEYS[:]
    rng.shuffle(shapes)
    chosen = []
    seen = set()
    for shape in shapes:
        sig = _shape_signature(shape)
        if sig in seen:
            continue
        seen.add(sig)
        chosen.append(shape)
        if len(chosen) == count:
            break
    specs = []
    colors = COLOR_POOL[:]
    rng.shuffle(colors)
    for i, shape in enumerate(chosen):
        key = np.rot90(shape, rng.randrange(4))
        specs.append({"color": colors[i], "key": key, "lock": _make_lock(key)})
    return specs


def _slot_layout(count: int, rng: random.Random) -> tuple[int, int]:
    if count == 1:
        return 1, 1
    if count == 2:
        return (1, 2) if rng.random() < 0.5 else (2, 1)
    return 2, 2


def _generate_attempt(rng: random.Random) -> np.ndarray:
    template_count = rng.randint(2, 4)
    object_count = rng.randint(1, 3)
    templates = _sample_templates(rng, template_count)

    all_colors = COLOR_POOL[:]
    rng.shuffle(all_colors)
    template_colors = {tpl["color"] for tpl in templates}
    object_colors = [c for c in all_colors if c not in template_colors][:object_count]
    if len(object_colors) < object_count:
        return None

    objects = []
    for color in object_colors:
        tpl = rng.choice(templates)
        rot = rng.randrange(4)
        hole = np.rot90(tpl["key"], rot)
        lock = np.rot90(tpl["lock"], rot)
        shell = _make_object_mask(hole, rng)
        objects.append({"color": color, "mask": shell, "lock": lock})

    max_key_w = max(tpl["key"].shape[1] for tpl in templates)
    max_lock_w = max(tpl["lock"].shape[1] for tpl in templates)
    pair_heights = [max(tpl["key"].shape[0], tpl["lock"].shape[0]) for tpl in templates]
    library_height = 1 + sum(pair_heights) + (template_count - 1) + 1
    library_width = 1 + max_key_w + 1 + max_lock_w + 1

    max_obj_h = max(obj["mask"].shape[0] for obj in objects)
    max_obj_w = max(obj["mask"].shape[1] for obj in objects)
    rows, cols = _slot_layout(object_count, rng)
    slot_h = max_obj_h + 2
    slot_w = max_obj_w + 2
    main_height = rows * slot_h + (rows - 1) + 2
    main_width = cols * slot_w + (cols - 1) + 2

    height = max(library_height, main_height)
    width = library_width + 1 + main_width
    grid = np.zeros((height, width), dtype=int)
    grid[:, library_width] = GREY

    key_col = 1
    lock_col = 1 + max_key_w + 1
    row = 1
    for tpl, pair_h in zip(templates, pair_heights):
        key_top = row + (pair_h - tpl["key"].shape[0]) // 2
        lock_top = row + (pair_h - tpl["lock"].shape[0]) // 2
        _draw_mask(grid, tpl["key"], key_top, key_col, tpl["color"])
        _draw_mask(grid, tpl["lock"], lock_top, lock_col, tpl["color"])
        row += pair_h + 1

    main_top = rng.randint(0, height - main_height)
    main_left = library_width + 1
    slot_cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(slot_cells)

    for obj, (slot_r, slot_c) in zip(objects, slot_cells):
        obj_h, obj_w = obj["mask"].shape
        base_r = main_top + 1 + slot_r * (slot_h + 1)
        base_c = main_left + 1 + slot_c * (slot_w + 1)
        top = base_r + rng.randint(0, slot_h - obj_h)
        left = base_c + rng.randint(0, slot_w - obj_w)
        _draw_mask(grid, obj["mask"], top, left, obj["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
