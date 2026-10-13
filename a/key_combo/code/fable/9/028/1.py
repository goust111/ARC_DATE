import random

import numpy as np


BLACK = 0
COLOR_IDS = tuple(range(1, 10))
MAX_GENERATE_ATTEMPTS = 500
GRID_H_RANGE = (8, 14)
PANEL_W_RANGE = (8, 13)
PAIR_RECT_SIDES = (1, 2, 3, 4)
HOLE_RECT_SIDES = (1, 2, 3)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        if 2 * c + 1 != w:
            continue
        if not np.all(grid[:, c] == grid[0, c]):
            continue
        color = int(grid[0, c])
        if int(np.count_nonzero(grid == color)) != h:
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one separator column"
    return candidates[0]


def _rot_coord(h: int, w: int, r: int, c: int) -> tuple[int, int]:
    return h - 1 - r, w - 1 - c


def _mirror_rect(panel_w: int, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, rh, rw = rect
    return r0, panel_w - c0 - rw, rh, rw


def _rot_rect(panel_h: int, panel_w: int, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, rh, rw = rect
    return panel_h - r0 - rh, panel_w - c0 - rw, rh, rw


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, rh, rw = rect
    mask[r0 : r0 + rh, c0 : c0 + rw] = True
    return mask


def _rect_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1, ac1 = ar0 + ah - 1, ac0 + aw - 1
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _can_place(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, rh, rw = rect
    r1 = r0 + rh - 1
    c1 = c0 + rw - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not np.any(mask[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _paint_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, rh, rw = rect
    mask[r0 : r0 + rh, c0 : c0 + rw] = True


def _make_base_panel(h: int, w: int, palette: list[int]) -> np.ndarray | None:
    bg = random.choice(palette)
    accents = [c for c in palette if c != bg]
    if len(accents) < 3:
        return None

    panel = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    pair_count = random.randint(3, min(5, len(accents) + 1))

    for i in range(pair_count):
        color = accents[i % len(accents)]
        placed = False
        for _ in range(120):
            rh = random.choice(PAIR_RECT_SIDES)
            rw = random.choice(PAIR_RECT_SIDES)
            if rh > h or rw > w:
                continue
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            rect = (r0, c0, rh, rw)
            mate = _rot_rect(h, w, rect)
            if rect == mate:
                continue
            if _rect_intersect(rect, mate):
                continue
            if not _can_place(occupied, rect, pad=1):
                continue
            if not _can_place(occupied, mate, pad=1):
                continue
            _paint_rect(occupied, rect)
            _paint_rect(occupied, mate)
            rr0, cc0, _, _ = rect
            mr0, mc0, _, _ = mate
            panel[rr0 : rr0 + rh, cc0 : cc0 + rw] = color
            panel[mr0 : mr0 + rh, mc0 : mc0 + rw] = color
            placed = True
            break
        if not placed:
            return None

    if np.array_equal(panel, panel[:, ::-1]):
        return None
    if len(np.unique(panel)) < 4:
        return None
    return panel


def _random_rect(h: int, w: int, sides: tuple[int, ...]) -> tuple[int, int, int, int]:
    rh = random.choice([s for s in sides if s <= h])
    rw = random.choice([s for s in sides if s <= w])
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    return r0, c0, rh, rw


def _sample_type_a_rect(h: int, w: int, left_holes: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(200):
        rect = _random_rect(h, w, HOLE_RECT_SIDES)
        mate = _rot_rect(h, w, rect)
        if rect == mate:
            continue
        if _rect_intersect(rect, mate):
            continue
        if not _can_place(left_holes, rect, pad=1):
            continue
        if not _can_place(left_holes, mate, pad=1):
            continue
        temp = np.zeros_like(left_holes)
        _paint_rect(temp, rect)
        if not _can_place(temp, mate, pad=1):
            continue
        return rect, mate
    return None


def _sample_type_b_rect(
    h: int,
    w: int,
    left_holes: np.ndarray,
    reserved_right_visible: np.ndarray,
) -> tuple[int, int, int, int] | None:
    for _ in range(300):
        rect = _random_rect(h, w, HOLE_RECT_SIDES)
        rot = _rot_rect(h, w, rect)
        mirror = _mirror_rect(w, rect)
        if rect == rot or _rect_intersect(rect, rot):
            continue
        if not _can_place(left_holes, rect, pad=1):
            continue
        if np.any(_rect_mask((h, w), rot) & left_holes):
            continue
        if np.any(_rect_mask((h, w), mirror) & reserved_right_visible):
            continue
        return rect
    return None


def _sample_extra_right_rect(
    h: int,
    w: int,
    right_holes: np.ndarray,
    forbidden: np.ndarray,
) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        rect = _random_rect(h, w, HOLE_RECT_SIDES)
        if not _can_place(right_holes, rect, pad=1):
            continue
        if np.any(_rect_mask((h, w), rect) & forbidden):
            continue
        return rect
    return None


def _rect_components_are_rectangles(mask: np.ndarray) -> bool:
    comps = _components(mask)
    if not comps:
        return False
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
    return True


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    left_base: np.ndarray,
    right_base: np.ndarray,
    left_holes: np.ndarray,
    right_holes: np.ndarray,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    sep = _find_separator(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    h, w = left.shape
    if not np.array_equal(expected, left_base):
        return False
    if not np.array_equal(right_base, left_base[:, ::-1]):
        return False
    if not np.array_equal(left_base, np.rot90(left_base, 2)):
        return False
    if np.array_equal(left_base, left_base[:, ::-1]):
        return False
    if not _rect_components_are_rectangles(left_holes):
        return False
    if not _rect_components_are_rectangles(right_holes):
        return False
    if len(_components(left_holes)) < 3 or len(_components(right_holes)) < 2:
        return False

    needs_mirror = False
    needs_rotation = False
    for r, c in np.argwhere(left_holes):
        rr, rc = _rot_coord(h, w, r, c)
        mc = w - 1 - c
        right_missing = bool(right_holes[r, mc])
        rot_missing = bool(left_holes[rr, rc])
        if right_missing and rot_missing:
            return False
        if right_missing:
            needs_rotation = True
            if left_base[rr, rc] != left_base[r, c]:
                return False
        else:
            if right_base[r, mc] != left_base[r, c]:
                return False
            if rot_missing:
                needs_mirror = True
            elif left_base[rr, rc] != left_base[r, c]:
                return False
    if not (needs_mirror and needs_rotation):
        return False
    if np.any(left[left_holes] != BLACK) or np.any(right[right_holes] != BLACK):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the left panel by first borrowing colors from the mirrored right panel, then falling back to the left panel's hidden 180-degree symmetry.

    Input structure:
    - One solid separator column splits the grid into equal-width left and right panels.
    - The right panel is the left panel mirrored left-to-right, and the underlying left-panel pattern is 180-degree rotationally symmetric but not itself left-right symmetric.
    - Black rectangular patches mark missing cells on both panels.

    Rule:
    - Ignore the separator and repair only the left panel.
    - For each black cell in the left panel, first copy the color from the right panel cell at the mirrored column in the same row.
    - If that right-panel mirror cell is also black, instead copy the color from the left panel cell at the 180-degree rotated position.

    Key insight:
    - The two panels provide a first symmetry source, but some mirrored source cells are also damaged, so the hidden 180-degree self-symmetry of the left panel is required as a second repair source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "panels must have equal size"

    h, w = left.shape
    out = left.copy()
    missing = left == BLACK
    assert np.any(missing), "expected black missing cells in the left panel"

    for r, c in np.argwhere(missing):
        rr, rc = _rot_coord(h, w, int(r), int(c))
        mc = w - 1 - int(c)
        mirror_color = int(right[int(r), mc])
        rotate_color = int(left[rr, rc])
        if mirror_color != BLACK:
            if rotate_color != BLACK:
                assert mirror_color == rotate_color, "the two symmetry sources must agree"
            out[int(r), int(c)] = mirror_color
        else:
            assert rotate_color != BLACK, "a second-stage 180-degree source is required here"
            out[int(r), int(c)] = rotate_color

    assert not np.any(out == BLACK), "repair must fill every left-panel hole"
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*PANEL_W_RANGE)
    sep_color = random.choice(COLOR_IDS)
    palette_size = random.randint(5, 7)
    palette = random.sample([c for c in COLOR_IDS if c != sep_color], palette_size)
    left_base = _make_base_panel(h, w, palette)
    if left_base is None:
        return None
    right_base = left_base[:, ::-1].copy()

    left_holes = np.zeros((h, w), dtype=bool)
    right_holes = np.zeros((h, w), dtype=bool)

    type_a = _sample_type_a_rect(h, w, left_holes)
    if type_a is None:
        return None
    rect_a, rect_a_rot = type_a
    _paint_rect(left_holes, rect_a)
    _paint_rect(left_holes, rect_a_rot)

    reserved_right_visible = np.zeros((h, w), dtype=bool)
    _paint_rect(reserved_right_visible, _mirror_rect(w, rect_a))
    _paint_rect(reserved_right_visible, _mirror_rect(w, rect_a_rot))

    rect_b = _sample_type_b_rect(h, w, left_holes, reserved_right_visible)
    if rect_b is None:
        return None
    _paint_rect(left_holes, rect_b)
    rect_b_mirror = _mirror_rect(w, rect_b)
    if not _can_place(right_holes, rect_b_mirror, pad=1):
        return None
    _paint_rect(right_holes, rect_b_mirror)

    forbidden_extra_right = reserved_right_visible | _rect_mask((h, w), rect_b_mirror)
    rect_c = _sample_extra_right_rect(h, w, right_holes, forbidden_extra_right)
    if rect_c is None:
        return None
    _paint_rect(right_holes, rect_c)

    grid = np.empty((h, 2 * w + 1), dtype=int)
    left = left_base.copy()
    right = right_base.copy()
    left[left_holes] = BLACK
    right[right_holes] = BLACK
    grid[:, :w] = left
    grid[:, w] = sep_color
    grid[:, w + 1 :] = right

    if _validate_instance(grid, left_base, left_base, right_base, left_holes, right_holes):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
