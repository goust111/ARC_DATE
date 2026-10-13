import random
from collections import Counter, deque

import numpy as np


BG = 0
DOOR = 5
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ["left", "right", "top", "bottom"]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_MAX_ATTEMPTS = 400
GEN_COUNT_RANGE = (1, 4)
GEN_EDGE_MARGIN = 1
GEN_GAP = 1
GEN_DIMENSIONS = {
    1: ((20, 26), (20, 26)),
    2: ((26, 34), (26, 34)),
    3: ((32, 40), (32, 40)),
    4: ((38, 46), (38, 46)),
}


def _get_bg(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]]
    ).astype(int)
    return int(Counter(border.tolist()).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out, (r0, c0, r1, c1)


def _box_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1


def _rotations(arr: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(arr, k) for k in range(4)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored key so it matches the same-colored lock cavity inside a grey panel.

    Input structure:
    - Background is black.
    - Grey rectangles are door panels.
    - Each panel contains one colored outline surrounding a black cavity.
    - Outside the panels, one solid sprite of the same color appears as the key.

    Rule:
    - Pair each key with the same-colored lock outline.
    - Extract the enclosed black cavity inside that panel.
    - Rotate the key by 0/90/180/270 degrees until its solid mask exactly matches the cavity.
    - Fill the cavity with the rotated key and erase the original key location.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    non_bg_colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    assert non_bg_colors, "expected non-background objects"

    panel_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in non_bg_colors:
        for comp in _components_mask(g == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.concatenate(
                [patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]]
            )
            has_inner_detail = bool(((patch != int(bg)) & (patch != int(color))).any())
            if bool((border == int(color)).all()) and has_inner_detail:
                panel_candidates.append((int(color), (r0, c0, r1, c1)))

    door_colors = sorted({color for color, _box in panel_candidates})
    assert len(door_colors) == 1, "expected exactly one panel color"
    door_color = int(door_colors[0])
    panel_boxes = [box for color, box in panel_candidates if int(color) == int(door_color)]
    assert panel_boxes, "expected at least one door panel"

    move_colors = [c for c in non_bg_colors if int(c) != int(door_color)]
    for color in move_colors:
        comps = _components_mask(g == int(color), connectivity=8)
        assert len(comps) == 2, "each key color must appear as one outline and one key"

        key_coords = None
        panel_box = None
        for comp in comps:
            comp_box = _bbox(comp)
            owner = next((box for box in panel_boxes if _box_inside(comp_box, box)), None)
            if owner is None:
                assert key_coords is None, "expected a single key component outside the panels"
                key_coords = comp
            else:
                assert panel_box is None, "expected a single outline component inside a panel"
                panel_box = owner

        assert key_coords is not None and panel_box is not None, "failed to classify key vs lock"

        kr0, kc0, kr1, kc1 = _bbox(key_coords)
        key_patch = g[kr0 : kr1 + 1, kc0 : kc1 + 1]
        key_mask = key_patch == int(color)

        pr0, pc0, pr1, pc1 = panel_box
        panel = g[pr0 : pr1 + 1, pc0 : pc1 + 1]

        holes = []
        for comp in _components_mask(panel == int(bg)):
            if any(
                r == 0 or c == 0 or r == panel.shape[0] - 1 or c == panel.shape[1] - 1
                for r, c in comp
            ):
                continue
            hole_mask, hole_box = _tight_mask(comp)
            holes.append((hole_mask, hole_box))

        assert holes, "each panel must contain an enclosed cavity"

        matches = []
        for rot_k, rot_patch in enumerate(_rotations(key_patch)):
            rot_mask = rot_patch == int(color)
            if int(rot_mask.sum()) != int(key_mask.sum()):
                continue
            for hole_mask, (hr0, hc0, hr1, hc1) in holes:
                if rot_mask.shape == hole_mask.shape and np.array_equal(rot_mask, hole_mask):
                    matches.append((rot_k, rot_patch, hole_mask, (hr0, hc0, hr1, hc1)))

        assert len(matches) == 1, "expected exactly one rotation/cavity match"
        _rot_k, rot_patch, hole_mask, (hr0, hc0, hr1, hc1) = matches[0]

        out[key_coords[:, 0], key_coords[:, 1]] = int(bg)
        target = out[pr0 + hr0 : pr0 + hr1 + 1, pc0 + hc0 : pc0 + hc1 + 1]
        target[hole_mask] = rot_patch[rot_patch == int(color)]

    return out


def _make_profile(length: int, max_depth: int) -> list[int]:
    inner = list(range(1, length - 1))
    active_count = random.randint(2, len(inner) - 1)
    active = set(random.sample(inner, active_count))
    depths = []
    for idx in range(length):
        if idx in active:
            depths.append(random.randint(1, max_depth))
        else:
            depths.append(0)
    if len({d for d in depths if d > 0}) == 1:
        pos = random.choice(list(active))
        depths[pos] = max_depth if depths[pos] == 1 else depths[pos] - 1
        if depths[pos] == 0:
            depths[pos] = max_depth
    return depths


def _make_key_mask(side: str) -> np.ndarray:
    if side in {"left", "right"}:
        body_h = random.randint(5, 7)
        body_w = random.randint(4, 6)
        profile = _make_profile(body_h, random.randint(1, 2))
        max_depth = max(profile)
        mask = np.zeros((body_h, body_w + max_depth), dtype=bool)
        if side == "left":
            mask[:, max_depth : max_depth + body_w] = True
            for r, depth in enumerate(profile):
                if depth:
                    mask[r, max_depth - depth : max_depth] = True
        else:
            mask[:, :body_w] = True
            for r, depth in enumerate(profile):
                if depth:
                    mask[r, body_w : body_w + depth] = True
    else:
        body_h = random.randint(4, 6)
        body_w = random.randint(5, 7)
        profile = _make_profile(body_w, random.randint(1, 2))
        max_depth = max(profile)
        mask = np.zeros((body_h + max_depth, body_w), dtype=bool)
        if side == "top":
            mask[max_depth : max_depth + body_h, :] = True
            for c, depth in enumerate(profile):
                if depth:
                    mask[max_depth - depth : max_depth, c] = True
        else:
            mask[:body_h, :] = True
            for c, depth in enumerate(profile):
                if depth:
                    mask[body_h : body_h + depth, c] = True
    assert bool(mask.any())
    return mask


def _render_key(color: int, mask: np.ndarray) -> np.ndarray:
    arr = np.full(mask.shape, BG, dtype=int)
    arr[mask] = int(color)
    return arr


def _render_lock(color: int, hole_mask: np.ndarray) -> np.ndarray:
    h, w = hole_mask.shape
    hole = np.zeros((h + 2, w + 2), dtype=bool)
    hole[1 : h + 1, 1 : w + 1] = hole_mask
    outline = np.zeros_like(hole, dtype=bool)
    for r in range(hole.shape[0]):
        for c in range(hole.shape[1]):
            if not bool(hole[r, c]):
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < hole.shape[0] and 0 <= nc < hole.shape[1] and not bool(hole[nr, nc]):
                    outline[nr, nc] = True
    art = np.full(hole.shape, DOOR, dtype=int)
    art[outline] = int(color)
    art[hole] = BG
    return art


def _make_panel(color: int, hole_mask: np.ndarray) -> np.ndarray:
    lock = _render_lock(color, hole_mask)
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    panel = np.full((lock.shape[0] + top + bottom, lock.shape[1] + left + right), DOOR, dtype=int)
    panel[top : top + lock.shape[0], left : left + lock.shape[1]] = lock
    return panel


def _boxes_clear(
    box: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
    gap: int,
) -> bool:
    r0, c0, r1, c1 = box
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != int(BG)
    sub[mask] = patch[mask]


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    gap: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    gh, gw = grid.shape
    rows = list(range(GEN_EDGE_MARGIN, gh - h - GEN_EDGE_MARGIN + 1))
    cols = list(range(GEN_EDGE_MARGIN, gw - w - GEN_EDGE_MARGIN + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    for top in rows:
        for left in cols:
            box = (top, left, top + h - 1, left + w - 1)
            if _boxes_clear(box, boxes, gap):
                _paste_patch(grid, patch, top, left)
                boxes.append(box)
                return top, left
    return None


def _gen_object_specs(count: int) -> list[dict]:
    colors = random.sample(KEY_COLORS, count)
    specs = []
    for color in colors:
        side = random.choice(SIDES)
        hole_mask = _make_key_mask(side)
        key = _render_key(color, hole_mask)
        rot_k = random.randrange(4)
        panel = _make_panel(color, hole_mask)
        specs.append(
            {
                "color": int(color),
                "panel": panel,
                "key": np.rot90(key, rot_k),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*GEN_COUNT_RANGE)
    (h_lo, h_hi), (w_lo, w_hi) = GEN_DIMENSIONS[count]
    grid = np.full((random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)), BG, dtype=int)
    specs = _gen_object_specs(count)
    boxes: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda s: s["panel"].shape[0] * s["panel"].shape[1], reverse=True):
        if _place_patch(grid, spec["panel"], boxes, GEN_GAP) is None:
            return None

    key_specs = specs[:]
    random.shuffle(key_specs)
    for spec in key_specs:
        if _place_patch(grid, spec["key"], boxes, GEN_GAP) is None:
            return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock puzzle input")
