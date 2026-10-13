from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
PINK = 6
ORANGE = 7
TEAL = 8

LR_SIDES = ("left", "right")
TB_SIDES = ("top", "bottom")
ALL_SIDES = LR_SIDES + TB_SIDES

BASE_TEMPLATES = (
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
)

GEN_GRID_MIN = 16
GEN_GRID_MAX = 22
GEN_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1)) if connectivity == 4 else (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def _tab_side(mask: np.ndarray) -> str:
    counts = {
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, -1].sum()),
        "top": int(mask[0, :].sum()),
        "bottom": int(mask[-1, :].sum()),
    }
    candidates = [side for side, count in counts.items() if count == 1]
    assert len(candidates) == 1, f"expected exactly one tab side, got {counts}"
    return candidates[0]


def _mirror_target(mask: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> tuple[np.ndarray, int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = mask.shape
    if side in LR_SIDES:
        reflected = np.fliplr(mask)
        copy_color = BLUE
        path_color = ORANGE
        top = r0
        left = c1 + 2 if side == "left" else c0 - w - 1
    else:
        reflected = np.flipud(mask)
        copy_color = TEAL
        path_color = PINK
        top = r1 + 2 if side == "top" else r0 - h - 1
        left = c0
    return reflected, top, left, copy_color, path_color


def _nearest_pair(a_coords: np.ndarray, b_coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    best_key: tuple[int, int, int, int, int] | None = None
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    for ar, ac in a_coords:
        p = (int(ar), int(ac))
        for br, bc in b_coords:
            q = (int(br), int(bc))
            key = (max(abs(p[0] - q[0]), abs(p[1] - q[1])), p[0], p[1], q[0], q[1])
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (p, q)
    assert best_pair is not None
    return best_pair


def _chebyshev_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = p
    qr, qc = q
    path = [(r, c)]
    while (r, c) != (qr, qc):
        r += _sign(qr - r)
        c += _sign(qc - c)
        path.append((r, c))
    return path


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords.astype(int)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _interior_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(mask, dtype=bool)
    queue = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    while queue:
        rr, cc = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return bool((bg & ~seen).any())


def _is_ring_mask(mask: np.ndarray) -> bool:
    if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    if len(_components_mask(mask, connectivity=4)) != 1:
        return False
    if not _interior_hole(mask):
        return False
    h, w = mask.shape
    degrees = []
    for r, c in np.argwhere(mask):
        deg = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                deg += 1
        degrees.append(deg)
    return min(degrees) >= 2 and max(degrees) <= 3


def _random_base_mask() -> np.ndarray:
    for _ in range(100):
        mask = random.choice(BASE_TEMPLATES).copy()
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = mask.T
        if random.random() < 0.35:
            axis = random.choice((0, 1))
            idx = random.randrange(mask.shape[axis])
            mask = np.insert(mask, idx, mask[idx, :] if axis == 0 else mask[:, idx], axis=axis)
        if _is_ring_mask(mask) and mask.shape[0] <= 5 and mask.shape[1] <= 5:
            return mask.astype(bool)
    raise RuntimeError("failed to build a valid ring mask")


def _build_tabbed_patch(base: np.ndarray, side: str) -> np.ndarray:
    h, w = base.shape
    if side in LR_SIDES:
        patch = np.zeros((h, w + 1), dtype=bool)
        edge_rows = [r for r in range(h) if base[r, 0 if side == "left" else w - 1]]
        tab_r = random.choice(edge_rows)
        if side == "left":
            patch[:, 1:] = base
            patch[tab_r, 0] = True
        else:
            patch[:, :w] = base
            patch[tab_r, w] = True
    else:
        patch = np.zeros((h + 1, w), dtype=bool)
        edge_cols = [c for c in range(w) if base[0 if side == "top" else h - 1, c]]
        tab_c = random.choice(edge_cols)
        if side == "top":
            patch[1:, :] = base
            patch[0, tab_c] = True
        else:
            patch[:h, :] = base
            patch[h, tab_c] = True
    assert _tab_side(patch) == side
    return patch


def _pair_layout(patch: np.ndarray, side: str, grid_shape: tuple[int, int]) -> dict | None:
    ph, pw = patch.shape
    gh, gw = grid_shape
    if side in LR_SIDES:
        union_h, union_w = ph, 2 * pw + 1
    else:
        union_h, union_w = 2 * ph + 1, pw
    if union_h > gh or union_w > gw:
        return None

    union_top = random.randint(0, gh - union_h)
    union_left = random.randint(0, gw - union_w)
    if side == "left":
        orig_top, orig_left = union_top, union_left
    elif side == "right":
        orig_top, orig_left = union_top, union_left + pw + 1
    elif side == "top":
        orig_top, orig_left = union_top, union_left
    else:
        orig_top, orig_left = union_top + ph + 1, union_left

    orig_coords = _coords_from_mask(patch, orig_top, orig_left)
    bbox = (orig_top, orig_left, orig_top + ph - 1, orig_left + pw - 1)
    reflected, copy_top, copy_left, copy_color, path_color = _mirror_target(patch, bbox, side)
    copy_coords = _coords_from_mask(reflected, copy_top, copy_left)
    if copy_coords.size == 0:
        return None
    if copy_coords[:, 0].min() < 0 or copy_coords[:, 1].min() < 0:
        return None
    if copy_coords[:, 0].max() >= gh or copy_coords[:, 1].max() >= gw:
        return None

    original_set = {tuple(map(int, rc)) for rc in orig_coords}
    copy_set = {tuple(map(int, rc)) for rc in copy_coords}
    if original_set & copy_set:
        return None

    p, q = _nearest_pair(orig_coords, copy_coords)
    path = _chebyshev_path(p, q)
    inner = path[1:-1]
    if not inner:
        return None
    if any(cell in original_set or cell in copy_set for cell in inner):
        return None

    all_coords = np.array(list(original_set | copy_set | set(inner)), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(all_coords)
    return {
        "side": side,
        "patch": patch,
        "orig_coords": orig_coords,
        "copy_coords": copy_coords,
        "copy_color": copy_color,
        "path_color": path_color,
        "path_inner": inner,
        "union_bbox": (r0, c0, r1, c1),
        "nearest_pair": (p, q),
    }


def _bboxes_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0


def _build_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    components = _components_mask(grid == GREEN, connectivity=4)
    assert len(components) == 2, f"expected 2 green components, got {len(components)}"
    components.sort(key=lambda coords: _bbox_coords(coords))

    for coords in components:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN
        side = _tab_side(patch)
        reflected, top, left, copy_color, path_color = _mirror_target(patch, bbox, side)
        h, w = reflected.shape
        assert 0 <= top and 0 <= left and top + h <= out.shape[0] and left + w <= out.shape[1], "copy must stay in bounds"

        copy_coords = _coords_from_mask(reflected, top, left)
        for rr, cc in copy_coords:
            assert int(out[rr, cc]) == BLACK, "copy must land on background"
            out[rr, cc] = copy_color

        p, q = _nearest_pair(coords, copy_coords)
        path = _chebyshev_path(p, q)
        for rr, cc in path[1:-1]:
            assert int(out[rr, cc]) == BLACK, "path must cross only background"
            out[rr, cc] = path_color
    return out


def _instance_signature(grid: np.ndarray) -> list[dict]:
    components = _components_mask(grid == GREEN, connectivity=4)
    comps = []
    for coords in components:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN
        side = _tab_side(patch)
        comps.append(
            {
                "bbox": bbox,
                "side": side,
                "shape": patch.shape,
                "tab_count": int(patch.sum()),
                "side_counts": (
                    int(patch[:, 0].sum()),
                    int(patch[:, -1].sum()),
                    int(patch[0, :].sum()),
                    int(patch[-1, :].sum()),
                ),
            }
        )
    comps.sort(key=lambda item: item["bbox"])
    return comps


def _validate_instance(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape == out.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert set(map(int, np.unique(inp))) <= {BLACK, GREEN}
    assert set(map(int, np.unique(out))) <= {BLACK, GREEN, BLUE, TEAL, ORANGE, PINK}
    assert np.array_equal(out, _build_output(inp))

    signature = _instance_signature(inp)
    assert len(signature) == 2
    side_set = {item["side"] in LR_SIDES for item in signature}
    assert side_set == {False, True}, "need one horizontal-mirror case and one vertical-mirror case"
    assert any(item["shape"][0] == 5 or item["shape"][1] == 5 for item in signature), "cover the larger bbox branch"

    delta = out.copy()
    delta[inp == GREEN] = BLACK
    assert np.count_nonzero(delta == BLUE) > 0
    assert np.count_nonzero(delta == TEAL) > 0
    assert np.count_nonzero(delta == ORANGE) > 0
    assert np.count_nonzero(delta == PINK) > 0

    out_components = _components_mask((out == BLUE) | (out == TEAL) | (out == GREEN), connectivity=4)
    assert len(out_components) >= 4, "copies must remain visually separate"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each green tabbed outline to the side opposite its tab, then connect the original and copy by a shortest king-path.

    Input structure:
    - A black background contains exactly two 4-connected green hollow outline objects.
    - Each object has exactly one one-pixel tab protruding from one side of its tight bbox.
    - One tab is horizontal (left or right) and the other is vertical (top or bottom).

    Rule:
    - Detect the tab side by the unique bbox edge containing exactly one green pixel.
    - Left/right tabs create a left-right mirrored blue copy across the opposite bbox side; top/bottom tabs create an up-down mirrored teal copy across the opposite bbox side.
    - For each original-copy pair, choose the Chebyshev-nearest endpoint pair, breaking ties by row then column, and color every intermediate cell of a deterministic shortest king-path between them.
    - Orange paths belong to the left/right mirror case, pink paths belong to the top/bottom mirror case, and all original green pixels stay unchanged.

    Key insight:
    - The single tab controls three coupled outputs at once: mirror direction, reflected copy color, and connection color; the path only exists because the mirror creates its endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"
    assert set(map(int, np.unique(g))) <= {BLACK, GREEN}, "expected only black background and green outlines"
    return _build_output(g)


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    width = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    shapes = (height, width)

    pair_specs: list[dict] = []
    sides = [random.choice(LR_SIDES), random.choice(TB_SIDES)]
    random.shuffle(sides)

    for side in sides:
        placed = False
        for _ in range(120):
            patch = _build_tabbed_patch(_random_base_mask(), side)
            pair = _pair_layout(patch, side, shapes)
            if pair is None:
                continue
            if any(not _bboxes_far_apart(pair["union_bbox"], other["union_bbox"], padding=1) for other in pair_specs):
                continue
            pair_specs.append(pair)
            placed = True
            break
        if not placed:
            return None

    inp = np.zeros(shapes, dtype=int)
    for pair in pair_specs:
        for r, c in pair["orig_coords"]:
            if inp[r, c] != BLACK:
                return None
            inp[r, c] = GREEN

    out = main(inp)
    try:
        _validate_instance(inp, out)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
