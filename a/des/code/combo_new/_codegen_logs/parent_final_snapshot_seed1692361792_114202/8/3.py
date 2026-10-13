import numpy as np
from collections import deque


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _flood_from_edges(blocked: np.ndarray) -> np.ndarray:
    height, width = blocked.shape
    seen = np.zeros_like(blocked, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(height):
        for c in (0, width - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, height, width):
            if not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _find_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fences: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for coords in _components(g == int(color)):
            comp = np.zeros_like(g, dtype=bool)
            comp[coords[:, 0], coords[:, 1]] = True
            outside = _flood_from_edges(comp)
            interior = (~outside) & (~comp)
            clue_colors = sorted(
                int(c)
                for c in np.unique(g[interior])
                if int(c) not in (BG, int(color))
            )
            if not clue_colors:
                continue
            fences.append(
                {
                    "color": int(color),
                    "bbox": _bbox(coords),
                    "mask": comp,
                    "current": comp.copy(),
                    "interior": interior,
                    "outside_black": outside & (g == BG),
                    "layers": len(clue_colors),
                }
            )
    fences.sort(key=lambda item: (item["bbox"], item["color"]))
    return fences


def _collect_growth_candidates(
    current: np.ndarray, outside_black: np.ndarray, occupied: np.ndarray
) -> set[tuple[int, int]]:
    height, width = current.shape
    candidates: set[tuple[int, int]] = set()
    for r, c in np.argwhere(current):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c, height, width):
            if outside_black[nr, nc] and not occupied[nr, nc]:
                candidates.add((nr, nc))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each closed fence's trapped clue colors, then grow the same-colored outer decoration by that many layers.

    Input structure:
    - The background is black.
    - One or more monochrome 1-pixel closed fences appear as separate 4-connected loops.
    - Each fence surrounds an enclosed region containing small clue pixels in 1 to 3 distinct non-black, non-fence colors.
    - Different fences are spatially separated, though their outward growth may compete for nearby black cells.

    Rule:
    - Detect every monochrome component whose removal leaves an enclosed interior region.
    - For each fence, count the number N of distinct clue colors inside that enclosed region.
    - Clear all interior clue pixels to black, then grow the fence outward by N synchronous black-only layers.
    - When different fences would overlap or touch during the same outward growth, reject those conflicting frontier cells so at least one black cell remains between final colored regions.

    Key insight:
    - The interior clue colors are not preserved; they encode how thick the fence's exterior decoration must become.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    fences = _find_fences(g)
    if not fences:
        return out

    for fence in fences:
        interior = fence["interior"]
        clue_mask = interior & (g != BG) & (g != fence["color"])
        out[clue_mask] = BG

    occupied = np.zeros_like(g, dtype=bool)
    for fence in fences:
        occupied |= fence["current"]

    max_layers = max(int(fence["layers"]) for fence in fences)
    height, width = g.shape

    for layer in range(1, max_layers + 1):
        proposal_map: dict[tuple[int, int], list[int]] = {}
        proposals_by_fence: list[set[tuple[int, int]]] = [set() for _ in fences]

        for idx, fence in enumerate(fences):
            if layer > int(fence["layers"]):
                continue
            candidates = _collect_growth_candidates(
                fence["current"], fence["outside_black"], occupied
            )
            proposals_by_fence[idx] = candidates
            for cell in candidates:
                proposal_map.setdefault(cell, []).append(idx)

        accepted: list[set[tuple[int, int]]] = [set() for _ in fences]
        for cell, owners in proposal_map.items():
            if len(owners) != 1:
                continue
            idx = int(owners[0])
            r, c = cell
            blocked = False
            for nr, nc in _neighbors4(r, c, height, width):
                for other_idx, other in enumerate(fences):
                    if other_idx != idx and other["current"][nr, nc]:
                        blocked = True
                        break
                if blocked:
                    break
                if (nr, nc) in proposal_map:
                    if any(other_idx != idx for other_idx in proposal_map[(nr, nc)]):
                        blocked = True
                        break
            if not blocked:
                accepted[idx].add(cell)

        for idx, cells in enumerate(accepted):
            if not cells:
                continue
            color = int(fences[idx]["color"])
            for r, c in cells:
                fences[idx]["current"][r, c] = True
                occupied[r, c] = True
                out[r, c] = color

    return out


def _draw_line(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    r = r0
    c = c0
    while True:
        mask[r, c] = True
        if (r, c) == (r1, c1):
            return
        r += dr
        c += dc


def _rectangle_loop(rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(5, 10))
    width = int(rng.integers(5, 10))
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _bumped_loop(rng: np.random.Generator) -> np.ndarray:
    side = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    if side in {"top", "bottom"}:
        base_h = int(rng.integers(6, 10))
        base_w = int(rng.integers(6, 10))
        depth = int(rng.integers(2, 4))
        bump_w = int(rng.integers(2, base_w - 2))
        start = int(rng.integers(1, base_w - bump_w))
        patch_h = base_h + depth
        patch_w = base_w
        top = depth if side == "top" else 0
        bottom = top + base_h - 1
        left = 0
        right = base_w - 1
        mask = np.zeros((patch_h, patch_w), dtype=bool)
        s = left + start
        e = s + bump_w
        if side == "top":
            _draw_line(mask, bottom, left, top, left)
            _draw_line(mask, top, left, top, s)
            _draw_line(mask, top, s, 0, s)
            _draw_line(mask, 0, s, 0, e)
            _draw_line(mask, 0, e, top, e)
            _draw_line(mask, top, e, top, right)
            _draw_line(mask, top, right, bottom, right)
            _draw_line(mask, bottom, right, bottom, left)
        else:
            _draw_line(mask, top, left, bottom, left)
            _draw_line(mask, bottom, left, bottom, s)
            _draw_line(mask, bottom, s, patch_h - 1, s)
            _draw_line(mask, patch_h - 1, s, patch_h - 1, e)
            _draw_line(mask, patch_h - 1, e, bottom, e)
            _draw_line(mask, bottom, e, bottom, right)
            _draw_line(mask, bottom, right, top, right)
            _draw_line(mask, top, right, top, left)
        return mask

    base_h = int(rng.integers(6, 10))
    base_w = int(rng.integers(6, 10))
    depth = int(rng.integers(2, 4))
    bump_h = int(rng.integers(2, base_h - 2))
    start = int(rng.integers(1, base_h - bump_h))
    patch_h = base_h
    patch_w = base_w + depth
    top = 0
    bottom = base_h - 1
    left = depth if side == "left" else 0
    right = left + base_w - 1
    mask = np.zeros((patch_h, patch_w), dtype=bool)
    s = top + start
    e = s + bump_h
    if side == "left":
        _draw_line(mask, top, right, top, left)
        _draw_line(mask, top, left, s, left)
        _draw_line(mask, s, left, s, 0)
        _draw_line(mask, s, 0, e, 0)
        _draw_line(mask, e, 0, e, left)
        _draw_line(mask, e, left, bottom, left)
        _draw_line(mask, bottom, left, bottom, right)
        _draw_line(mask, bottom, right, top, right)
    else:
        _draw_line(mask, top, left, top, right)
        _draw_line(mask, top, right, s, right)
        _draw_line(mask, s, right, s, patch_w - 1)
        _draw_line(mask, s, patch_w - 1, e, patch_w - 1)
        _draw_line(mask, e, patch_w - 1, e, right)
        _draw_line(mask, e, right, bottom, right)
        _draw_line(mask, bottom, right, bottom, left)
        _draw_line(mask, bottom, left, top, left)
    return mask


def _loop_interior(loop_mask: np.ndarray) -> np.ndarray:
    return (~_flood_from_edges(loop_mask)) & (~loop_mask)


def _choose_clue_colors(
    rng: np.random.Generator, fence_color: int, count: int
) -> list[int]:
    choices = [color for color in PALETTE if int(color) != int(fence_color)]
    picked = rng.choice(np.array(choices, dtype=int), size=count, replace=False)
    return [int(color) for color in picked.tolist()]


def _place_clues(
    rng: np.random.Generator,
    patch: np.ndarray,
    interior: np.ndarray,
    fence_color: int,
    layer_count: int,
) -> bool:
    clue_colors = _choose_clue_colors(rng, fence_color, layer_count)
    blocked = np.zeros_like(interior, dtype=bool)
    height, width = patch.shape

    for color in clue_colors:
        anchors = [
            (int(r), int(c))
            for r, c in np.argwhere(interior & (~blocked))
        ]
        if not anchors:
            anchors = [(int(r), int(c)) for r, c in np.argwhere(interior & (patch == BG))]
        if not anchors:
            return False
        ar, ac = anchors[int(rng.integers(len(anchors)))]
        cluster = [(ar, ac)]
        if rng.random() < 0.4:
            neighbors = [
                (nr, nc)
                for nr, nc in _neighbors4(ar, ac, height, width)
                if interior[nr, nc] and patch[nr, nc] == BG and not blocked[nr, nc]
            ]
            if neighbors:
                cluster.append(neighbors[int(rng.integers(len(neighbors)))])
        for rr, cc in cluster:
            patch[rr, cc] = int(color)
        for rr, cc in cluster:
            blocked[rr, cc] = True
            for nr, nc in _neighbors4(rr, cc, height, width):
                blocked[nr, nc] = True
    return True


def _make_object(
    rng: np.random.Generator, fence_color: int, force_rectangle: bool = False
) -> dict | None:
    layer_count = int(rng.integers(1, 4))
    if force_rectangle or rng.random() < 0.5:
        loop = _rectangle_loop(rng)
    else:
        loop = _bumped_loop(rng)
    interior = _loop_interior(loop)
    if int(interior.sum()) < layer_count:
        return None
    patch = np.zeros(loop.shape, dtype=int)
    patch[loop] = int(fence_color)
    if not _place_clues(rng, patch, interior, int(fence_color), layer_count):
        return None
    return {
        "patch": patch,
        "height": int(patch.shape[0]),
        "width": int(patch.shape[1]),
        "layers": layer_count,
    }


def _boxes_too_close(
    box_a: tuple[int, int, int, int],
    layers_a: int,
    box_b: tuple[int, int, int, int],
    layers_b: int,
) -> bool:
    pad = int(layers_a + layers_b + 2)
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _try_place_spaced(
    rng: np.random.Generator,
    height: int,
    width: int,
    obj: dict,
    placed: list[dict],
) -> tuple[int, int] | None:
    if obj["height"] > height or obj["width"] > width:
        return None
    for _ in range(120):
        r0 = int(rng.integers(0, height - obj["height"] + 1))
        c0 = int(rng.integers(0, width - obj["width"] + 1))
        box = (r0, c0, r0 + obj["height"] - 1, c0 + obj["width"] - 1)
        if any(
            _boxes_too_close(box, obj["layers"], item["box"], item["layers"])
            for item in placed
        ):
            continue
        return r0, c0
    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _validate_generated(grid: np.ndarray, expected_fences: int) -> bool:
    fences = _find_fences(grid)
    if len(fences) != int(expected_fences):
        return False
    if any(int(fence["layers"]) not in (1, 2, 3) for fence in fences):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return bool(out.shape == grid.shape and max(grid.shape) <= 30 and np.any(out != grid))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_ATTEMPTS):
        obj_count = int(rng.integers(1, 4))
        close_mode = bool(obj_count >= 2 and rng.random() < 0.5)
        fence_colors = rng.choice(np.array(PALETTE, dtype=int), size=obj_count, replace=False)

        objects: list[dict] = []
        ok = True
        for idx in range(obj_count):
            obj = _make_object(rng, int(fence_colors[idx]), force_rectangle=close_mode and idx < 2)
            if obj is None:
                ok = False
                break
            objects.append(obj)
        if not ok:
            continue

        grid_h = int(rng.integers(18, 31))
        grid_w = int(rng.integers(18, 31))
        grid = np.zeros((grid_h, grid_w), dtype=int)
        placed: list[dict] = []

        if close_mode:
            first = objects[0]
            second = objects[1]
            gap_max = min(5, first["layers"] + second["layers"])
            gap = int(rng.integers(2, gap_max + 1))
            horizontal = bool(rng.random() < 0.5)

            if horizontal:
                total_h = max(first["height"], second["height"])
                total_w = first["width"] + second["width"] + gap
                if total_h > grid_h or total_w > grid_w:
                    continue
                top = int(rng.integers(0, grid_h - total_h + 1))
                left = int(rng.integers(0, grid_w - total_w + 1))
                pos1 = (top, left)
                pos2 = (top, left + first["width"] + gap)
            else:
                total_h = first["height"] + second["height"] + gap
                total_w = max(first["width"], second["width"])
                if total_h > grid_h or total_w > grid_w:
                    continue
                top = int(rng.integers(0, grid_h - total_h + 1))
                left = int(rng.integers(0, grid_w - total_w + 1))
                pos1 = (top, left)
                pos2 = (top + first["height"] + gap, left)

            for obj, (r0, c0) in ((first, pos1), (second, pos2)):
                _paste_patch(grid, obj["patch"], r0, c0)
                placed.append(
                    {
                        "box": (r0, c0, r0 + obj["height"] - 1, c0 + obj["width"] - 1),
                        "layers": obj["layers"],
                    }
                )

            remaining = sorted(
                objects[2:], key=lambda item: (item["height"] * item["width"]), reverse=True
            )
        else:
            remaining = sorted(
                objects, key=lambda item: (item["height"] * item["width"]), reverse=True
            )

        failed = False
        for obj in remaining:
            pos = _try_place_spaced(rng, grid_h, grid_w, obj, placed)
            if pos is None:
                failed = True
                break
            r0, c0 = pos
            _paste_patch(grid, obj["patch"], r0, c0)
            placed.append(
                {
                    "box": (r0, c0, r0 + obj["height"] - 1, c0 + obj["width"] - 1),
                    "layers": obj["layers"],
                }
            )
        if failed:
            continue

        if _validate_generated(grid, obj_count):
            return grid

    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
