import numpy as np


BG = 0
KEY_COLOR = 5
CONTAINER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")

MIN_OBJECTS = 2
MAX_OBJECTS = 4
GRID_MIN = 20
GRID_MAX = 30
RECT_MIN = 6
RECT_MAX = 10
SHAPE_MIN_AREA = 4
SHAPE_MAX_AREA = 8
PLACEMENT_RETRIES = 300
GEN_RETRIES = 400

RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_signature(mask: np.ndarray) -> tuple:
    variants = []
    for rot in range(4):
        turned = _trim_mask(np.rot90(mask, rot))
        variants.append((turned.shape, tuple(turned.astype(int).ravel().tolist())))
    return min(variants)


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for rot in range(4):
        turned = _trim_mask(np.rot90(mask, rot))
        if not any(v.shape == turned.shape and np.array_equal(v, turned) for v in variants):
            variants.append(turned)
    return variants


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _has_hole(mask: np.ndarray) -> bool:
    bg_mask = ~mask
    if not bg_mask.any():
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack = []

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        rr, cc = stack.pop()
        for nr, nc in _neighbors4(rr, cc, h, w):
            if bg_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any(bg_mask & ~seen))


def _build_container_patch(
    color: int,
    rect_h: int,
    rect_w: int,
    side: str,
    notch_mask: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((rect_h, rect_w), color, dtype=int)
    sh, sw = notch_mask.shape
    missing = np.zeros((rect_h, rect_w), dtype=bool)

    if side == "top":
        r0, c0 = 0, offset
    elif side == "bottom":
        r0, c0 = rect_h - sh, offset
    elif side == "left":
        r0, c0 = offset, 0
    elif side == "right":
        r0, c0 = offset, rect_w - sw
    else:
        raise ValueError(f"unknown side: {side}")

    missing[r0 : r0 + sh, c0 : c0 + sw] = notch_mask
    patch[missing] = BG
    return patch, missing


def _extract_container(grid: np.ndarray, coords: np.ndarray, color: int) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((patch == color) | (patch == BG))

    filled = patch == color
    missing = ~filled
    assert missing.any()
    assert _count_components(missing) == 1

    touches = [
        bool(missing[0, :].any()),
        bool(missing[:, -1].any()),
        bool(missing[-1, :].any()),
        bool(missing[:, 0].any()),
    ]
    assert sum(touches) == 1

    notch = _trim_mask(missing)
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "missing": missing,
        "notch": notch,
    }


def _extract_key(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "mask": mask,
    }


def _collect_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    keys = [_extract_key(coords) for coords in _find_color_components(grid, KEY_COLOR)]

    containers = []
    for color in CONTAINER_COLORS:
        for coords in _find_color_components(grid, color):
            containers.append(_extract_container(grid, coords, color))

    assert keys
    assert containers
    assert len(keys) == len(containers)
    return keys, containers


def _unique_assignment(keys: list[dict], containers: list[dict]) -> dict[int, tuple[int, int]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for ki, key in enumerate(keys):
        options = []
        for ci, container in enumerate(containers):
            for rot in range(4):
                turned = _trim_mask(np.rot90(key["mask"], rot))
                if turned.shape == container["notch"].shape and np.array_equal(turned, container["notch"]):
                    options.append((ci, rot))
                    break
        assert options, "every key must match at least one lockhole"
        candidates[ki] = options

    order = sorted(range(len(keys)), key=lambda idx: len(candidates[idx]))
    used = set()
    chosen: dict[int, tuple[int, int]] = {}
    solutions: list[dict[int, tuple[int, int]]] = []

    def backtrack(pos: int):
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(chosen.copy())
            return
        ki = order[pos]
        for ci, rot in candidates[ki]:
            if ci in used:
                continue
            used.add(ci)
            chosen[ki] = (ci, rot)
            backtrack(pos + 1)
            used.remove(ci)
            del chosen[ki]

    backtrack(0)
    assert len(solutions) == 1, "shape matching must yield a unique pairing"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each grey key into its uniquely matching lockhole, recolor it to the container, and mark the key's original bbox corner.

    Input structure:
    - The background is black.
    - Several large monochrome container components each occupy the bbox of a filled rectangle except for one connected black notch carved inward from exactly one side.
    - Several smaller grey connected components lie on the background outside the containers.
    - Each grey component's shape matches exactly one container notch after a rotation by 0, 90, 180, or 270 degrees.

    Rule:
    - Treat every grey component as a key and every large colored almost-rectangle as a container with one missing lockhole.
    - Match each key to the unique container whose missing-notch mask equals one rotation of the key's binary shape.
    - Remove the key from its original position, fill the matched notch with the container color, and then place a single pixel of that container color at the top-left corner of the key's original bounding box.

    Key insight:
    - The only reliable correspondence is geometric: the missing black region inside each container is the lockhole, and the grey object that matches it up to rotation identifies both where to insert and which color to leave behind as the original-position marker.
    """
    g = np.asarray(grid, dtype=int)
    keys, containers = _collect_objects(g)
    assignment = _unique_assignment(keys, containers)

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = BG

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[container["missing"]] = container["color"]

    for ki, (ci, _rot) in assignment.items():
        r0, c0, _r1, _c1 = keys[ki]["bbox"]
        out[r0, c0] = containers[ci]["color"]

    return out


def _random_connected_shape(used_signatures: set[tuple]) -> tuple[np.ndarray, tuple] | None:
    for _ in range(200):
        h = int(RNG.integers(2, 5))
        w = int(RNG.integers(2, 5))
        max_area = min(SHAPE_MAX_AREA, h * w - 1)
        min_area = max(SHAPE_MIN_AREA, h + w - 2)
        if min_area > max_area:
            continue
        target = int(RNG.integers(min_area, max_area + 1))

        start = (int(RNG.integers(h)), int(RNG.integers(w)))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(RNG.integers(len(frontier)))])
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if min(mask.shape) < 2:
            continue
        if mask.all():
            continue
        if _has_hole(mask):
            continue

        variants = _rotation_variants(mask)
        if len(variants) != 4:
            continue

        sig = _mask_signature(mask)
        if sig in used_signatures:
            continue
        return mask, sig
    return None


def _can_reserve(used: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = used.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not bool(used[r0:r1, c0:c1].any())


def _reserve(used: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = used.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    used[r0:r1, c0:c1] = True


def _pick_rectangle_dims(shape: np.ndarray, side: str) -> tuple[int, int, int]:
    sh, sw = shape.shape
    for _ in range(100):
        if side in ("top", "bottom"):
            min_h = max(RECT_MIN, sh + 1)
            min_w = max(RECT_MIN, sw + 2)
            rect_h = int(RNG.integers(min_h, RECT_MAX + 1))
            rect_w = int(RNG.integers(min_w, RECT_MAX + 1))
            if rect_w - sw < 2:
                continue
            offset = int(RNG.integers(1, rect_w - sw))
            return rect_h, rect_w, offset
        min_h = max(RECT_MIN, sh + 2)
        min_w = max(RECT_MIN, sw + 1)
        rect_h = int(RNG.integers(min_h, RECT_MAX + 1))
        rect_w = int(RNG.integers(min_w, RECT_MAX + 1))
        if rect_h - sh < 2:
            continue
        offset = int(RNG.integers(1, rect_h - sh))
        return rect_h, rect_w, offset
    raise RuntimeError("failed to size container rectangle")


def _expected_output(layout: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in layout:
        tr, tc = spec["container_top"], spec["container_left"]
        patch = np.full((spec["rect_h"], spec["rect_w"]), spec["color"], dtype=int)
        out[tr : tr + spec["rect_h"], tc : tc + spec["rect_w"]] = patch

    for spec in layout:
        kr, kc = spec["key_top"], spec["key_left"]
        out[kr, kc] = spec["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = int(RNG.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    min_side = 18 + 2 * (n_objects - 2)
    h = int(RNG.integers(min_side, GRID_MAX + 1))
    w = int(RNG.integers(min_side, GRID_MAX + 1))

    layout = []
    used_signatures: set[tuple] = set()
    colors = list(RNG.choice(CONTAINER_COLORS, size=n_objects, replace=False))

    for idx in range(n_objects):
        result = _random_connected_shape(used_signatures)
        if result is None:
            return None
        shape, sig = result
        used_signatures.add(sig)
        side = SIDES[int(RNG.integers(4))]
        rect_h, rect_w, offset = _pick_rectangle_dims(shape, side)
        key_rot = int(RNG.integers(4))
        key_patch = np.rot90(shape.astype(int) * KEY_COLOR, key_rot)
        container_patch, missing = _build_container_patch(colors[idx], rect_h, rect_w, side, shape, offset)
        layout.append(
            {
                "color": int(colors[idx]),
                "shape": shape,
                "side": side,
                "rect_h": rect_h,
                "rect_w": rect_w,
                "container_patch": container_patch,
                "missing": missing,
                "key_patch": key_patch,
                "key_rot": key_rot,
            }
        )

    used = np.zeros((h, w), dtype=bool)
    grid = np.full((h, w), BG, dtype=int)

    for spec in sorted(layout, key=lambda item: item["rect_h"] * item["rect_w"], reverse=True):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            max_top = h - spec["rect_h"] - 1
            max_left = w - spec["rect_w"] - 1
            if max_top < 1 or max_left < 1:
                return None
            top = int(RNG.integers(1, max_top + 1))
            left = int(RNG.integers(1, max_left + 1))
            if not _can_reserve(used, top, left, spec["rect_h"], spec["rect_w"], pad=1):
                continue
            grid[top : top + spec["rect_h"], left : left + spec["rect_w"]] = spec["container_patch"]
            _reserve(used, top, left, spec["rect_h"], spec["rect_w"], pad=1)
            spec["container_top"] = top
            spec["container_left"] = left
            placed = True
            break
        if not placed:
            return None

    for spec in layout:
        kh, kw = spec["key_patch"].shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            max_top = h - kh - 1
            max_left = w - kw - 1
            if max_top < 1 or max_left < 1:
                return None
            top = int(RNG.integers(1, max_top + 1))
            left = int(RNG.integers(1, max_left + 1))
            if not _can_reserve(used, top, left, kh, kw, pad=1):
                continue
            key_mask = spec["key_patch"] == KEY_COLOR
            patch = grid[top : top + kh, left : left + kw]
            if np.any(patch[key_mask] != BG):
                continue
            patch[key_mask] = KEY_COLOR
            _reserve(used, top, left, kh, kw, pad=1)
            spec["key_top"] = top
            spec["key_left"] = left
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(layout, grid.shape)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock container sample")
