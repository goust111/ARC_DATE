import numpy as np


BG = 0
CROSS = 5
MARKER = 6
PANEL_ORDER = ("BL", "TR", "TL", "BR")
QUADRANTS = ("TL", "TR", "BL", "BR")
SIDE_TO_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
GEN_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each cross-separated panel by its marker and overlay the four results into one canvas.

    Input structure:
    - A black background is split by one full grey row and one full grey column into four equal panels.
    - Each panel is square, contains one connected non-black shape in its own color, and has one magenta marker near the midpoint of one edge.
    - The grey cross is only a separator; the four panels use local coordinates relative to their own square.

    Rule:
    - Remove the grey cross and inspect each panel's magenta marker.
    - Top, right, bottom, and left edge markers encode 0, 90 clockwise, 180, and 270 clockwise rotation respectively.
    - Erase the marker, rotate that panel's colored content in-place, then transparently overlay the rotated panels in fixed order: bottom-left, top-right, top-left, bottom-right.
    - Black is transparent; later non-black pixels overwrite earlier ones.

    Key insight:
    - The marker does not indicate where to move a panel; it changes the panel's internal orientation before the fixed quadrant-based layer stack is composited.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    cross_r, cross_c = _find_cross(g)
    panels = _split_panels(g, cross_r, cross_c)
    panel_shapes = {panel.shape for panel in panels.values()}
    assert len(panel_shapes) == 1
    panel_h, panel_w = next(iter(panel_shapes))
    assert panel_h == panel_w, "panel rotation is only well-defined for square panels"

    panel_colors = []
    rotated = {}
    for name, panel in panels.items():
        assert CROSS not in np.unique(panel), f"{name} panel cannot contain separator color"
        marker_coords = np.argwhere(panel == MARKER)
        assert len(marker_coords) == 1, f"{name} panel must contain exactly one marker"
        fg_colors = [int(v) for v in np.unique(panel) if int(v) not in (BG, MARKER)]
        assert len(fg_colors) == 1, f"{name} panel must contain exactly one foreground color"
        panel_colors.append(fg_colors[0])

        side = _decode_marker_side(tuple(map(int, marker_coords[0])), panel.shape)
        cleaned = panel.copy()
        cleaned[cleaned == MARKER] = BG
        rotated[name] = np.rot90(cleaned, k=SIDE_TO_K[side])

    assert len(set(panel_colors)) == 4, "the four panels should use distinct foreground colors"
    return _compose({name: rotated[name] for name in QUADRANTS}, panel_h, panel_w, PANEL_ORDER)


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == CROSS)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == CROSS)]
    assert len(rows) == 1 and len(cols) == 1, "expected exactly one full grey cross"
    cross_r, cross_c = int(rows[0]), int(cols[0])
    assert grid.shape[0] == 2 * cross_r + 1
    assert grid.shape[1] == 2 * cross_c + 1
    return cross_r, cross_c


def _split_panels(grid: np.ndarray, cross_r: int, cross_c: int) -> dict[str, np.ndarray]:
    return {
        "TL": grid[:cross_r, :cross_c],
        "TR": grid[:cross_r, cross_c + 1 :],
        "BL": grid[cross_r + 1 :, :cross_c],
        "BR": grid[cross_r + 1 :, cross_c + 1 :],
    }


def _decode_marker_side(coord: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = coord
    h, w = shape
    assert h == w
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    distances = {
        "top": r,
        "right": w - 1 - c,
        "bottom": h - 1 - r,
        "left": c,
    }
    min_dist = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == min_dist]
    assert len(sides) == 1, "marker must be closest to exactly one edge"
    side = sides[0]
    if side in ("top", "bottom"):
        assert abs(c - center_c) <= 1.5
    else:
        assert abs(r - center_r) <= 1.5
    assert min_dist <= 1
    return side


def _compose(panels: dict[str, np.ndarray], panel_h: int, panel_w: int, order: tuple[str, ...]) -> np.ndarray:
    out = np.full((panel_h, panel_w), BG, dtype=int)
    for name in order:
        panel = panels[name]
        mask = panel != BG
        out[mask] = panel[mask]
    return out


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return ()
    coords = coords - coords.min(axis=0)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return tuple((int(coords[i, 0]), int(coords[i, 1])) for i in order)


def _rotation_variant_count(mask: np.ndarray) -> int:
    return len({_mask_signature(np.rot90(mask, k=k)) for k in range(4)})


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, mask.shape):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _special_points(n: int) -> dict[str, tuple[int, int]]:
    pivot_r = n // 2
    pivot_c = n // 2
    return {
        "pivot": (pivot_r, pivot_c),
        "tltr": (pivot_r - 1, pivot_c),
        "bltr": (pivot_r + 1, pivot_c),
        "u_tl": (pivot_r - 2, pivot_c - 1),
        "u_tr": (pivot_r - 2, pivot_c + 1),
        "u_bl": (pivot_r + 2, pivot_c - 1),
        "u_br": (pivot_r + 1, pivot_c + 2),
    }


def _panel_specs(n: int):
    pts = _special_points(n)
    specs = {
        "TL": {
            "must": {pts["pivot"], pts["tltr"], pts["u_tl"]},
            "forbid": {pts["bltr"], pts["u_tr"], pts["u_bl"], pts["u_br"]},
        },
        "TR": {
            "must": {pts["pivot"], pts["tltr"], pts["bltr"], pts["u_tr"]},
            "forbid": {pts["u_tl"], pts["u_bl"], pts["u_br"]},
        },
        "BL": {
            "must": {pts["pivot"], pts["bltr"], pts["u_bl"]},
            "forbid": {pts["tltr"], pts["u_tl"], pts["u_tr"], pts["u_br"]},
        },
        "BR": {
            "must": {pts["pivot"], pts["u_br"]},
            "forbid": {pts["tltr"], pts["bltr"], pts["u_tl"], pts["u_tr"], pts["u_bl"]},
        },
    }
    return specs


def _random_shortest_path(start: tuple[int, int], goal: tuple[int, int], allowed: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = allowed.shape
    queue = [start]
    parents = {start: None}
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == goal:
            break
        neigh = list(_neighbors4(cur[0], cur[1], (h, w)))
        np.random.shuffle(neigh)
        for nxt in neigh:
            if not allowed[nxt] or nxt in parents:
                continue
            parents[nxt] = cur
            queue.append(nxt)
    if goal not in parents:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _frontier(cells: set[tuple[int, int]], allowed: np.ndarray) -> list[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c, allowed.shape):
            if allowed[nr, nc] and (nr, nc) not in cells:
                out.add((nr, nc))
    return list(out)


def _build_rotated_mask(name: str, n: int, size: int) -> np.ndarray | None:
    specs = _panel_specs(n)[name]
    allowed = np.ones((n, n), dtype=bool)
    for r, c in specs["forbid"]:
        allowed[r, c] = False

    must = sorted(specs["must"])
    pivot = _special_points(n)["pivot"]
    cells = {pivot}
    for target in must:
        path = _random_shortest_path(pivot, target, allowed)
        if path is None:
            return None
        cells.update(path)
    if len(cells) > size:
        return None

    for _ in range(40):
        if len(cells) >= size:
            break
        fringe = _frontier(cells, allowed)
        if not fringe:
            return None
        weights = np.array(
            [1.0 / (1.0 + abs(r - n / 2.0) + abs(c - n / 2.0)) for r, c in fringe],
            dtype=float,
        )
        weights /= weights.sum()
        pick = fringe[int(np.random.choice(len(fringe), p=weights))]
        cells.add(pick)
    if len(cells) != size:
        return None

    mask = np.zeros((n, n), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if not _is_connected(mask):
        return None
    if _rotation_variant_count(mask) != 4:
        return None
    return mask


def _marker_candidates(n: int, side: str) -> list[tuple[int, int]]:
    center_band = sorted(
        {
            v
            for v in (n // 2 - 1, (n - 1) // 2, n // 2, n // 2 + 1)
            if 0 <= v < n
        }
    )
    if side == "top":
        return [(r, c) for r in (0, 1) for c in center_band]
    if side == "bottom":
        return [(r, c) for r in (n - 1, n - 2) for c in center_band]
    if side == "left":
        return [(r, c) for c in (0, 1) for r in center_band]
    return [(r, c) for c in (n - 1, n - 2) for r in center_band]


def _pick_marker(mask: np.ndarray, side: str) -> tuple[int, int] | None:
    candidates = _marker_candidates(mask.shape[0], side)
    np.random.shuffle(candidates)
    for coord in candidates:
        if not mask[coord]:
            return coord
    return None


def _render_panel(mask: np.ndarray, color: int, marker: tuple[int, int]) -> np.ndarray:
    panel = np.full(mask.shape, BG, dtype=int)
    panel[mask] = color
    panel[marker] = MARKER
    return panel


def _assemble_input(panels: dict[str, np.ndarray], n: int) -> np.ndarray:
    grid = np.full((2 * n + 1, 2 * n + 1), BG, dtype=int)
    grid[n, :] = CROSS
    grid[:, n] = CROSS
    grid[:n, :n] = panels["TL"]
    grid[:n, n + 1 :] = panels["TR"]
    grid[n + 1 :, :n] = panels["BL"]
    grid[n + 1 :, n + 1 :] = panels["BR"]
    return grid


def _coverage(rot_masks: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack([rot_masks[name] for name in QUADRANTS], axis=0)
    return stacked.sum(axis=0)


def _colorize_masks(rot_masks: dict[str, np.ndarray], colors: dict[str, int]) -> dict[str, np.ndarray]:
    return {
        name: np.where(rot_masks[name], colors[name], BG).astype(int)
        for name in QUADRANTS
    }


def _validate_design(
    inp: np.ndarray,
    rot_masks: dict[str, np.ndarray],
    input_masks: dict[str, np.ndarray],
    colors: dict[str, int],
    sides: dict[str, str],
) -> bool:
    actual_panels = _colorize_masks(rot_masks, colors)
    expected = _compose(actual_panels, inp.shape[0] // 2, inp.shape[1] // 2, PANEL_ORDER)
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    if len(set(sides.values())) != 4:
        return False
    if sum(side != "top" for side in sides.values()) < 2:
        return False

    overlap = _coverage(rot_masks)
    if int((overlap >= 2).sum()) < 3:
        return False

    actual_colors = set(int(v) for v in np.unique(expected) if v != BG)
    if actual_colors != set(colors.values()):
        return False

    alt1 = _compose(actual_panels, expected.shape[0], expected.shape[1], ("BL", "TL", "TR", "BR"))
    alt2 = _compose(actual_panels, expected.shape[0], expected.shape[1], ("TR", "BL", "TL", "BR"))
    if np.array_equal(expected, alt1) or np.array_equal(expected, alt2):
        return False

    unrotated_panels = _colorize_masks(input_masks, colors)
    no_rotation = _compose(unrotated_panels, expected.shape[0], expected.shape[1], PANEL_ORDER)
    if np.array_equal(expected, no_rotation):
        return False

    for name, side in sides.items():
        if side == "top":
            continue
        mixed = dict(actual_panels)
        mixed[name] = unrotated_panels[name]
        if np.array_equal(expected, _compose(mixed, expected.shape[0], expected.shape[1], PANEL_ORDER)):
            return False

    decoded = {}
    n = inp.shape[0] // 2
    panels = _split_panels(inp, n, n)
    for name, panel in panels.items():
        marker = tuple(map(int, np.argwhere(panel == MARKER)[0]))
        decoded[name] = _decode_marker_side(marker, panel.shape)
    return decoded == sides


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(7, 11))
    sizes = {name: int(np.random.randint(4, 9)) for name in QUADRANTS}
    rot_masks = {}
    for name in QUADRANTS:
        mask = _build_rotated_mask(name, n, sizes[name])
        if mask is None:
            return None
        rot_masks[name] = mask

    side_perm = list(np.random.permutation(["top", "right", "bottom", "left"]))
    sides = {name: side_perm[i] for i, name in enumerate(QUADRANTS)}
    colors_perm = list(np.random.permutation([1, 2, 3, 4]))
    colors = {name: int(colors_perm[i]) for i, name in enumerate(QUADRANTS)}

    input_masks = {}
    panels = {}
    for name in QUADRANTS:
        k = SIDE_TO_K[sides[name]]
        original_mask = np.rot90(rot_masks[name], k=-k)
        marker = _pick_marker(original_mask, sides[name])
        if marker is None:
            return None
        input_masks[name] = original_mask
        panels[name] = _render_panel(original_mask, colors[name], marker)

    inp = _assemble_input(panels, n)
    if not _validate_design(inp, rot_masks, input_masks, colors, sides):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-rotation overlay input")
