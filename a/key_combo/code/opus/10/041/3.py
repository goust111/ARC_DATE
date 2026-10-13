import numpy as np


BG = 0
TEMPLATE_COLOR = 1
CONTROL_COLOR = 2
PARTIAL_COLOR = 3
TEMPLATE_SIZE = 5
GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return bool(coords.size) and len(_component_coords(mask)) == 1


def _rotate_clockwise(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(mask.astype(bool), -int(quarter_turns))


def _extract_blue_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    blue_components = _component_coords(grid == TEMPLATE_COLOR)
    assert blue_components, "expected a blue template"
    h, w = grid.shape
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    best = None
    for coords in blue_components:
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 != TEMPLATE_SIZE or c1 - c0 + 1 != TEMPLATE_SIZE:
            continue
        patch = _crop_mask(grid == TEMPLATE_COLOR, bbox)
        score = abs((r0 + r1) / 2.0 - center_r) + abs((c0 + c1) / 2.0 - center_c)
        area = int(patch.sum())
        item = (score, -area, patch, bbox)
        if best is None or item[:2] < best[:2]:
            best = item
    assert best is not None, "could not locate the central 5x5 template"
    template = np.asarray(best[2], dtype=bool)
    rotations = [_rotate_clockwise(template, k) for k in range(4)]
    assert len({rot.tobytes() for rot in rotations}) == 4, "template must be rotation-distinct"
    return template, best[3]


def _extract_red_components(grid: np.ndarray) -> list[np.ndarray]:
    return _component_coords(grid == CONTROL_COLOR)


def _extract_partial_components(grid: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    out = []
    for coords in _component_coords(grid == PARTIAL_COLOR):
        bbox = _bbox_from_coords(coords)
        out.append((coords, bbox))
    return out


def _match_control_cluster(
    bbox: tuple[int, int, int, int], red_components: list[np.ndarray]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    matches = []
    for coords in red_components:
        rows = coords[:, 0]
        cols = coords[:, 1]
        above = np.all(rows == r0 - 1) and np.all((c0 <= cols) & (cols <= c1))
        left = np.all(cols == c0 - 1) and np.all((r0 <= rows) & (rows <= r1))
        if above or left:
            matches.append(coords)
    assert len(matches) == 1, "each partial must have exactly one adjacent red cluster"
    return matches[0]


def _core_mask() -> np.ndarray:
    core = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
    core[2, :] = True
    core[:, 2] = True
    return core


def _build_expected_from_scene(
    grid: np.ndarray,
    template: np.ndarray,
    partials: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    red_components: list[np.ndarray],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for _coords, bbox in partials:
        r0, c0, r1, c1 = bbox
        control = _match_control_cluster(bbox, red_components)
        k = int(control.shape[0]) % 4
        rotated = _rotate_clockwise(template, k)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        observed = patch == PARTIAL_COLOR
        assert observed.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE)
        assert np.all(observed <= rotated), "partial must be a subset of the chosen rotation"
        patch[rotated & (~observed)] = PARTIAL_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each green partial copy with the correctly rotated blue template chosen by a red pixel count.

    Input structure:
    - Background is black.
    - A single complete blue 5x5 template sits near the grid center and has four distinct quarter-turn rotations.
    - Elsewhere there are several green partial 5x5 copies plus a nearby red control cluster placed immediately above or immediately left of each partial.

    Rule:
    - Read the central blue 5x5 template.
    - For each green partial copy, find its unique adjacent red cluster and count its pixels.
    - Rotate the blue template clockwise by that many quarter-turns modulo 4; a 4-pixel cluster therefore means a full turn and selects the unrotated template.
    - Align the rotated template to the green partial's 5x5 bounding box and fill every missing template pixel with green, while leaving the blue template and red clusters unchanged.

    Key insight:
    - The green partials are intentionally ambiguous on their own, so the red pixel count is the only signal that selects which rotated template should be used for completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    template, _template_bbox = _extract_blue_template(g)
    red_components = _extract_red_components(g)
    partials = _extract_partial_components(g)
    assert partials, "expected green partial copies"
    return _build_expected_from_scene(g, template, partials, red_components)


def _template_rotations(template: np.ndarray) -> list[np.ndarray]:
    return [_rotate_clockwise(template, k) for k in range(4)]


def _random_template() -> np.ndarray:
    core = _core_mask()
    candidates = [
        (0, 1), (0, 3),
        (1, 0), (1, 1), (1, 3), (1, 4),
        (3, 0), (3, 1), (3, 3), (3, 4),
        (4, 1), (4, 3),
    ]
    for _ in range(300):
        extras = _RNG.choice(len(candidates), size=3, replace=False)
        mask = core.copy()
        for idx in extras:
            r, c = candidates[int(idx)]
            mask[r, c] = True
        if not _is_connected(mask):
            continue
        rotations = _template_rotations(mask)
        if len({rot.tobytes() for rot in rotations}) != 4:
            continue
        return mask
    raise RuntimeError("failed to build a rotation-distinct 5x5 template")


def _control_cells(
    top: int, left: int, side: str, size: int, offset: int
) -> list[tuple[int, int]]:
    if side == "above":
        return [(top - 1, left + offset + dc) for dc in range(size)]
    return [(top + offset + dr, left - 1) for dr in range(size)]


def _can_place(
    blocked: np.ndarray,
    top: int,
    left: int,
    controls: list[tuple[int, int]],
) -> bool:
    h, w = blocked.shape
    if not (0 <= top <= h - TEMPLATE_SIZE and 0 <= left <= w - TEMPLATE_SIZE):
        return False
    cells = [(r, c) for r in range(top, top + TEMPLATE_SIZE) for c in range(left, left + TEMPLATE_SIZE)]
    cells.extend(controls)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if min(rs) < 0 or min(cs) < 0 or max(rs) >= h or max(cs) >= w:
        return False
    r0 = max(0, min(rs) - 1)
    c0 = max(0, min(cs) - 1)
    r1 = min(h - 1, max(rs) + 1)
    c1 = min(w - 1, max(cs) + 1)
    return not blocked[r0 : r1 + 1, c0 : c1 + 1].any()


def _reserve(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = max(0, min(rs) - 1)
    c0 = max(0, min(cs) - 1)
    r1 = min(h - 1, max(rs) + 1)
    c1 = min(w - 1, max(cs) + 1)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _manual_output(
    grid: np.ndarray,
    template: np.ndarray,
    placements: list[dict],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    rotations = _template_rotations(template)
    for item in placements:
        top = item["top"]
        left = item["left"]
        patch = out[top : top + TEMPLATE_SIZE, left : left + TEMPLATE_SIZE]
        patch[rotations[item["k"]] & (patch == BG)] = PARTIAL_COLOR
    return out


def _validate_generated_scene(
    grid: np.ndarray,
    template: np.ndarray,
    placements: list[dict],
) -> None:
    assert max(grid.shape) <= 30
    blue = (grid == TEMPLATE_COLOR)
    assert len(_component_coords(blue)) == 1
    template_bbox = _bbox_from_coords(np.argwhere(blue))
    assert template_bbox[2] - template_bbox[0] + 1 == TEMPLATE_SIZE
    assert template_bbox[3] - template_bbox[1] + 1 == TEMPLATE_SIZE

    partials = _extract_partial_components(grid)
    reds = _extract_red_components(grid)
    assert len(partials) == len(placements)
    assert len(reds) == len(placements)

    seen_rotations = set()
    rotations = _template_rotations(template)
    for partial_coords, bbox in partials:
        r0, c0, r1, c1 = bbox
        assert (r1 - r0 + 1, c1 - c0 + 1) == (TEMPLATE_SIZE, TEMPLATE_SIZE)
        observed = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
        observed[partial_coords[:, 0] - r0, partial_coords[:, 1] - c0] = True
        assert np.array_equal(observed, _core_mask())
        control = _match_control_cluster(bbox, reds)
        k = int(control.shape[0]) % 4
        seen_rotations.add(k)
        assert np.all(observed <= rotations[k])
        compatible = [j for j in range(4) if np.all(observed <= rotations[j])]
        assert len(compatible) >= 2, "partial must stay ambiguous without the control count"
        completions = {rotations[j].tobytes() for j in compatible}
        assert len(completions) >= 2, "control must change the completion"

    assert len(seen_rotations) >= 3
    expected = _manual_output(grid, template, placements)
    actual = main(grid)
    assert np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(18, 25))
    w = int(_RNG.integers(18, 25))
    grid = np.zeros((h, w), dtype=int)

    template = _random_template()
    top_t = h // 2 - 2
    left_t = w // 2 - 2
    grid[top_t : top_t + TEMPLATE_SIZE, left_t : left_t + TEMPLATE_SIZE][template] = TEMPLATE_COLOR

    blocked = np.zeros((h, w), dtype=bool)
    _reserve(
        blocked,
        [(r, c) for r in range(top_t, top_t + TEMPLATE_SIZE) for c in range(left_t, left_t + TEMPLATE_SIZE)],
    )

    count = 4 if _RNG.random() < 0.7 else 3
    ks = list(_RNG.permutation(4)[:count])
    sides = ["above", "left"]
    required_sides = ["above", "left"]
    while len(required_sides) < count:
        required_sides.append(sides[int(_RNG.integers(0, 2))])
    _RNG.shuffle(required_sides)

    placements: list[dict] = []
    core = _core_mask()
    for k, side in zip(ks, required_sides):
        size = 4 if k == 0 else k
        placed = False
        for _ in range(300):
            top = int(_RNG.integers(0, h - TEMPLATE_SIZE + 1))
            left = int(_RNG.integers(0, w - TEMPLATE_SIZE + 1))
            if side == "above" and top == 0:
                continue
            if side == "left" and left == 0:
                continue
            offset = int(_RNG.integers(0, TEMPLATE_SIZE - size + 1))
            controls = _control_cells(top, left, side, size, offset)
            if not _can_place(blocked, top, left, controls):
                continue
            patch_cells = [(top + r, left + c) for r, c in np.argwhere(core)]
            for r, c in patch_cells:
                grid[r, c] = PARTIAL_COLOR
            for r, c in controls:
                grid[r, c] = CONTROL_COLOR
            reserve_cells = [(r, c) for r in range(top, top + TEMPLATE_SIZE) for c in range(left, left + TEMPLATE_SIZE)]
            reserve_cells.extend(controls)
            _reserve(blocked, reserve_cells)
            placements.append({"top": top, "left": left, "k": k, "side": side, "size": size})
            placed = True
            break
        if not placed:
            return None

    if len(placements) != count:
        return None

    try:
        _validate_generated_scene(grid, template, placements)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
