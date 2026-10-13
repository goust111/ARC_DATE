import random

import numpy as np


BG = 0
GEN_RETRIES = 300
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 200
CANVAS_RETRIES = 200
COLOR_POOL = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(np.asarray(coords, dtype=int))

    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    trimmed = _trim_mask(mask)
    return tuple(map(int, trimmed.shape)), trimmed.astype(np.uint8).tobytes()


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _rotation_match(piece_mask: np.ndarray, hole_mask: np.ndarray) -> bool:
    for k in range(4):
        if np.array_equal(_trim_mask(np.rot90(piece_mask, k)), hole_mask):
            return True
    return False


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel = coords - np.array([r0, c0])
            mask[rel[:, 0], rel[:, 1]] = True
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(len(coords)),
                }
            )
    return comps


def _hole_regions(crop: np.ndarray, bg: int) -> list[dict]:
    holes: list[dict] = []
    h, w = map(int, crop.shape)
    for coords in _components(crop == bg):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([r0, c0])
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "area": int(len(coords)),
            }
        )
    return holes


def _match_pieces_to_holes(holes: list[dict], pieces: list[dict]) -> list[int]:
    assert len(holes) == len(pieces), "piece count must match hole count"
    candidates: list[list[int]] = []
    for piece in pieces:
        matched_holes = []
        for hi, hole in enumerate(holes):
            if int(piece["area"]) != int(hole["area"]):
                continue
            if _rotation_match(piece["mask"], hole["mask"]):
                matched_holes.append(int(hi))
        assert matched_holes, "every piece must match at least one hole"
        candidates.append(matched_holes)

    order = sorted(range(len(pieces)), key=lambda i: (len(candidates[i]), -int(pieces[i]["area"])))
    assignment = [-1] * len(pieces)
    used: set[int] = set()

    def search(idx: int) -> bool:
        if idx == len(order):
            return True
        pi = order[idx]
        for hi in candidates[pi]:
            if hi in used:
                continue
            used.add(hi)
            assignment[pi] = hi
            if search(idx + 1):
                return True
            used.remove(hi)
            assignment[pi] = -1
        return False

    assert search(0), "pieces must admit a one-to-one hole assignment"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Refill a hollow left-side template by rotating the scattered right-side pieces into matching holes.

    Input structure:
    - The background is black.
    - The largest foreground object is a single-color filled rectangle on the left with several disconnected black holes inside it.
    - The remaining foreground objects are smaller monochrome fragments scattered to the right; each fragment's shape equals one hole up to 0/90/180/270-degree rotation.

    Rule:
    - Find the large left template and crop to its bounding box.
    - Extract every enclosed black hole inside that crop as a 4-connected shape.
    - Extract every scattered fragment, compare its shape against all hole shapes under the four rotations, and pair fragments to holes by exact shape equality.
    - Fill each hole with the matched fragment's own color and output only the tight crop of the template region.

    Key insight:
    Colors do not identify destinations; the solution comes from matching each fragment's silhouette to a hole after trying the four right-angle rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    comps = _foreground_components(g, bg)
    assert comps, "expected at least one foreground component"
    template = max(comps, key=lambda comp: int(comp["area"]))
    tr0, tc0, tr1, tc1 = map(int, template["bbox"])
    crop = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()

    holes = _hole_regions(crop, bg)
    pieces = [comp for comp in comps if comp is not template]
    assert holes, "template must contain at least one hole"
    assert pieces, "expected scattered pieces outside the template"

    assignment = _match_pieces_to_holes(holes, pieces)
    out = crop.copy()
    for piece, hole_index in zip(pieces, assignment):
        coords = holes[int(hole_index)]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(piece["color"])

    return out


def _random_shape() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(6, h * w - 1)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if min(mask.shape) < 2:
            continue
        if mask.all():
            continue
        if len({_mask_key(np.rot90(mask, k)) for k in range(4)}) < 4:
            continue
        return mask

    raise RuntimeError("failed to generate an asymmetric connected shape")


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = map(int, blocked.shape)
    for r, c in coords:
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _generate_holes() -> list[dict]:
    count = random.randint(2, 4)
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(SHAPE_RETRIES * 2):
        if len(shapes) == count:
            break
        mask = _random_shape()
        key = _canonical_rotation_key(mask)
        if key in seen:
            continue
        shapes.append(mask)
        seen.add(key)
    if len(shapes) != count:
        raise RuntimeError("failed to generate enough distinct hole shapes")

    ordered = sorted(shapes, key=lambda mask: (-int(mask.sum()), int(mask.shape[0]), int(mask.shape[1])))
    for _ in range(PLACEMENT_RETRIES):
        outer_h = random.randint(max(8, max(mask.shape[0] for mask in ordered) + 4), 16)
        outer_w = random.randint(max(8, max(mask.shape[1] for mask in ordered) + 4), 16)
        blocked = np.zeros((outer_h, outer_w), dtype=bool)
        holes: list[dict] = []

        for mask in ordered:
            mh, mw = map(int, mask.shape)
            mask_coords = np.argwhere(mask)
            candidates = []
            for r0 in range(1, outer_h - mh):
                for c0 in range(1, outer_w - mw):
                    coords = mask_coords + np.array([r0, c0])
                    if np.any(blocked[coords[:, 0], coords[:, 1]]):
                        continue
                    candidates.append((r0, c0, coords))
            if not candidates:
                holes = []
                break

            r0, c0, coords = random.choice(candidates)
            holes.append(
                {
                    "mask": mask.copy(),
                    "coords": coords.copy(),
                    "bbox": (int(r0), int(c0), int(r0 + mh - 1), int(c0 + mw - 1)),
                    "area": int(mask.sum()),
                }
            )
            _mark_blocked(blocked, coords)

        if holes:
            return holes

    raise RuntimeError("failed to place holes inside the template")


def _template_grids(holes: list[dict], template_color: int, fill_colors: list[int] | None) -> tuple[np.ndarray, np.ndarray]:
    r1 = max(int(hole["bbox"][2]) for hole in holes)
    c1 = max(int(hole["bbox"][3]) for hole in holes)
    h = int(r1 + 2)
    w = int(c1 + 2)
    template = np.full((h, w), int(template_color), dtype=int)
    filled = template.copy()

    for index, hole in enumerate(holes):
        coords = hole["coords"]
        template[coords[:, 0], coords[:, 1]] = BG
        if fill_colors is None:
            filled[coords[:, 0], coords[:, 1]] = BG
        else:
            filled[coords[:, 0], coords[:, 1]] = int(fill_colors[index])

    return template, filled


def _scatter_scene(template: np.ndarray, pieces: list[dict]) -> np.ndarray | None:
    th, tw = map(int, template.shape)
    max_piece_w = max(int(piece["patch"].shape[1]) for piece in pieces)
    template_area = int(np.count_nonzero(template != BG))
    piece_area = sum(int(piece["area"]) for piece in pieces)
    min_grid_area = 2 * (template_area + piece_area) + 40

    for _ in range(CANVAS_RETRIES):
        h_low = max(th + 2, 15)
        h_high = min(28, max(h_low, th + 10))
        H = random.randint(h_low, h_high)

        gap = random.randint(2, 4)
        w_low = max(tw + gap + max_piece_w + 4, 20, (min_grid_area + H - 1) // H)
        if w_low > 30:
            continue
        W = random.randint(w_low, 30)

        template_r = random.randint(1, H - th - 1)
        template_c = random.randint(1, min(3, W - tw - 1))
        right_start = template_c + tw + gap
        if right_start + max_piece_w >= W - 1:
            continue

        grid = np.full((H, W), BG, dtype=int)
        grid[template_r : template_r + th, template_c : template_c + tw] = template
        blocked = np.zeros((H, W), dtype=bool)
        template_coords = np.argwhere(template != BG) + np.array([template_r, template_c])
        _mark_blocked(blocked, template_coords)

        failed = False
        for piece in pieces:
            patch = piece["patch"]
            mask = piece["mask"]
            ph, pw = map(int, patch.shape)
            piece_coords = np.argwhere(mask)
            candidates = []
            for r0 in range(1, H - ph):
                for c0 in range(right_start, W - pw):
                    coords = piece_coords + np.array([r0, c0])
                    if np.any(blocked[coords[:, 0], coords[:, 1]]):
                        continue
                    candidates.append((r0, c0, coords))
            if not candidates:
                failed = True
                break

            r0, c0, coords = random.choice(candidates)
            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            sub[mask] = patch[mask]
            _mark_blocked(blocked, coords)

        if failed:
            continue

        if _get_bg(grid) != BG:
            continue
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    holes = _generate_holes()
    random.shuffle(holes)

    available = list(COLOR_POOL)
    template_color = random.choice(available)
    available.remove(template_color)
    random.shuffle(available)
    fill_colors = available[: len(holes)]

    template, expected = _template_grids(holes, template_color, fill_colors)

    pieces = []
    for hole, color in zip(holes, fill_colors):
        rotation = random.randrange(4)
        rotated_mask = np.rot90(hole["mask"], rotation)
        patch = np.full(rotated_mask.shape, BG, dtype=int)
        patch[rotated_mask] = int(color)
        pieces.append(
            {
                "mask": rotated_mask,
                "patch": patch,
                "color": int(color),
                "area": int(rotated_mask.sum()),
            }
        )

    random.shuffle(pieces)
    grid = _scatter_scene(template, pieces)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
