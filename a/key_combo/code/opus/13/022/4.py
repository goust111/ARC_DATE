import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
FG = 5
GRID_MIN = 8
GRID_MAX = 15
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_fg_neighbors(mask: np.ndarray, r: int, c: int) -> int:
    total = 0
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            total += 1
    return total


def _component_count(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0
    seen: set[tuple[int, int]] = set()
    parts = 0
    for r, c in map(tuple, coords):
        if (r, c) in seen:
            continue
        parts += 1
        stack = [(r, c)]
        seen.add((r, c))
        while stack:
            rr, cc = stack.pop()
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return parts


def _has_tab(mask: np.ndarray) -> bool:
    for r, c in map(tuple, np.argwhere(mask)):
        if _count_fg_neighbors(mask, int(r), int(c)) == 1:
            return True
    return False


def _find_notches(mask: np.ndarray) -> list[tuple[int, int]]:
    box = _bbox(mask)
    assert box is not None, "foreground object required"
    r0, c0, r1, c1 = box
    out: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if bool(mask[r, c]):
                continue
            if _count_fg_neighbors(mask, r, c) != 3:
                continue
            if r in (r0, r1) or c in (c0, c1):
                out.append((r, c))
    return out


def _notch_side(mask: np.ndarray) -> str:
    box = _bbox(mask)
    assert box is not None, "foreground object required"
    r0, c0, r1, c1 = box
    notches = _find_notches(mask)
    assert notches, "expected at least one boundary notch"

    candidates = []
    if all(c == c0 for _, c in notches):
        candidates.append("left")
    if all(c == c1 for _, c in notches):
        candidates.append("right")
    if all(r == r0 for r, _ in notches):
        candidates.append("up")
    if all(r == r1 for r, _ in notches):
        candidates.append("down")
    assert len(candidates) == 1, "all notches must lie on one bbox side"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the side of the boundary notch and concatenate the whole grid with its matching mirror image.

    Input structure:
    - The grid contains a black background and one thick rectilinear foreground object.
    - The foreground object has one or two one-cell notches on its boundary, and all notches lie on the same side of the object's bounding box.

    Rule:
    - Find every notch cell: a background cell with exactly three foreground 4-neighbors.
    - Interpret the common notch side as a reflection instruction for the entire grid.
    - For right/left notches, horizontally mirror the full grid and concatenate left-right; for down/up notches, vertically mirror the full grid and concatenate top-bottom, with the mirrored half placed on the notch-indicated side.

    Key insight:
    - The notch is not repaired locally; its side encodes which global reflection axis to use when duplicating the whole scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    colors = sorted(set(map(int, g.flatten())))
    assert BG in colors and len(colors) == 2, "expected black background plus one foreground color"
    fg = next(color for color in colors if color != BG)
    mask = g == fg
    assert _component_count(mask) == 1, "expected a single connected foreground object"

    side = _notch_side(mask)
    if side == "right":
        return np.hstack((g, np.fliplr(g)))
    if side == "left":
        return np.hstack((np.fliplr(g), g))
    if side == "down":
        return np.vstack((g, np.flipud(g)))
    return np.vstack((np.flipud(g), g))


def _make_rectangle(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _make_l_cut(h: int, w: int) -> np.ndarray | None:
    if h < 6 or w < 6:
        return None
    mask = np.ones((h, w), dtype=bool)
    cut_h = random.randint(2, h - 3)
    cut_w = random.randint(2, w - 3)
    corner = random.choice(["TL", "TR", "BL", "BR"])
    if corner == "TL":
        mask[:cut_h, :cut_w] = False
    elif corner == "TR":
        mask[:cut_h, w - cut_w :] = False
    elif corner == "BL":
        mask[h - cut_h :, :cut_w] = False
    else:
        mask[h - cut_h :, w - cut_w :] = False
    return mask


def _make_t_shape(h: int, w: int) -> np.ndarray | None:
    if h < 7 or w < 7:
        return None
    mask = np.zeros((h, w), dtype=bool)
    orient = random.choice(["up", "down", "left", "right"])
    if orient in {"up", "down"}:
        bar_h = random.randint(3, min(5, h - 2))
        stem_w = random.randint(3, min(5, w - 2))
        stem_c0 = random.randint(1, w - stem_w - 1)
        mask[:, stem_c0 : stem_c0 + stem_w] = True
        if orient == "up":
            mask[:bar_h, :] = True
        else:
            mask[h - bar_h :, :] = True
    else:
        bar_w = random.randint(3, min(5, w - 2))
        stem_h = random.randint(3, min(5, h - 2))
        stem_r0 = random.randint(1, h - stem_h - 1)
        mask[stem_r0 : stem_r0 + stem_h, :] = True
        if orient == "left":
            mask[:, :bar_w] = True
        else:
            mask[:, w - bar_w :] = True
    return mask


def _make_plus(h: int, w: int) -> np.ndarray | None:
    if h < 7 or w < 7:
        return None
    mask = np.zeros((h, w), dtype=bool)
    bar_h = random.randint(3, min(5, h - 2))
    bar_w = random.randint(3, min(5, w - 2))
    r0 = random.randint(1, h - bar_h - 1)
    c0 = random.randint(1, w - bar_w - 1)
    mask[r0 : r0 + bar_h, :] = True
    mask[:, c0 : c0 + bar_w] = True
    return mask


def _build_local_shape(h: int, w: int) -> np.ndarray | None:
    kind = random.choice(["rect", "lcut", "t", "plus"])
    if kind == "rect":
        return _make_rectangle(h, w)
    if kind == "lcut":
        return _make_l_cut(h, w)
    if kind == "t":
        return _make_t_shape(h, w)
    return _make_plus(h, w)


def _candidate_notch_cells(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    candidates: list[tuple[int, int]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        r = int(r)
        c = int(c)
        if side == "left" and c != c0:
            continue
        if side == "right" and c != c1:
            continue
        if side == "up" and r != r0:
            continue
        if side == "down" and r != r1:
            continue
        if (side in {"left", "right"} and r in (r0, r1)) or (side in {"up", "down"} and c in (c0, c1)):
            continue
        if _count_fg_neighbors(mask, r, c) == 3:
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _has_valid_structure(mask: np.ndarray, side: str, notch_count: int) -> bool:
    if _component_count(mask) != 1:
        return False
    if _has_tab(mask):
        return False
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
        return False
    notches = _find_notches(mask)
    if len(notches) != notch_count:
        return False
    try:
        found_side = _notch_side(mask)
    except AssertionError:
        return False
    if found_side != side:
        return False
    return True


def _is_valid_input(mask: np.ndarray, side: str, notch_count: int, shape: tuple[int, int]) -> bool:
    if min(shape) < 1 or max(shape) > 30:
        return False
    if not (0.40 <= mask.mean() <= 0.60):
        return False
    return _has_valid_structure(mask, side, notch_count)


def _carve_notches(mask: np.ndarray, side: str, notch_count: int) -> np.ndarray | None:
    candidates = _candidate_notch_cells(mask, side)
    if len(candidates) < notch_count:
        return None

    if notch_count == 1:
        choices = [[cell] for cell in candidates]
    else:
        choices = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1:
                    continue
                choices.append([a, b])
        random.shuffle(choices)

    for cells in choices:
        carved = mask.copy()
        for r, c in cells:
            carved[r, c] = False
        if _has_valid_structure(carved, side, notch_count):
            return carved
    return None


def _embed_mask(local_mask: np.ndarray, H: int, W: int) -> np.ndarray | None:
    h, w = local_mask.shape
    if h > H - 2 or w > W - 2:
        return None
    top = random.randint(1, H - h - 1)
    left = random.randint(1, W - w - 1)
    grid_mask = np.zeros((H, W), dtype=bool)
    grid_mask[top : top + h, left : left + w] = local_mask
    return grid_mask


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(["left", "right", "up", "down"])
    notch_count = random.choice([1, 2])

    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    if side in {"left", "right"}:
        W = min(W, 15)
    else:
        H = min(H, 15)

    local_h = random.randint(max(5, H - 6), H - 2)
    local_w = random.randint(max(5, W - 6), W - 2)
    local_mask = _build_local_shape(local_h, local_w)
    if local_mask is None:
        return None
    if _component_count(local_mask) != 1 or _has_tab(local_mask):
        return None

    carved_local = _carve_notches(local_mask, side, notch_count)
    if carved_local is None:
        return None

    mask = _embed_mask(carved_local, H, W)
    if mask is None:
        return None
    if not _is_valid_input(mask, side, notch_count, (H, W)):
        return None

    grid = np.full((H, W), BG, dtype=int)
    grid[mask] = FG

    if side in {"left", "right"} and np.array_equal(grid, np.fliplr(grid)):
        return None
    if side in {"up", "down"} and np.array_equal(grid, np.flipud(grid)):
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notch-reflection ARC input")
