import numpy as np


BG = 0
AXIS_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS8 = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)
)
RNG = np.random.default_rng()
GEN_RETRY_LIMIT = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    packed = np.ascontiguousarray(mask.astype(np.uint8))
    return tuple(map(int, mask.shape)), packed.tobytes()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _extract_objects(grid: np.ndarray, bg: int, axis_color: int = AXIS_COLOR) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or color == axis_color or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bbox = _bbox_coords(arr)
            mask = _coords_to_mask(arr, bbox)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "mask": mask,
                    "self_symmetric": bool(np.array_equal(mask, np.fliplr(mask))),
                }
            )
    return objects


def _find_axis_column(grid: np.ndarray) -> int:
    grey_cells = np.argwhere(grid == AXIS_COLOR)
    assert grey_cells.shape[0] == 1, "expected exactly one grey axis marker"
    r, c = map(int, grey_cells[0])
    assert r == 0, "axis marker must sit on the top edge"
    return c


def _object_side(obj: dict, axis_col: int) -> str:
    _r0, c0, _r1, c1 = obj["bbox"]
    if c1 < axis_col:
        return "L"
    if c0 > axis_col:
        return "R"
    return "X"


def _objects_match(left: dict, right: dict, axis_col: int) -> bool:
    if _object_side(left, axis_col) != "L" or _object_side(right, axis_col) != "R":
        return False
    if left["color"] != right["color"]:
        return False
    lr0, _lc0, lr1, _lc1 = left["bbox"]
    rr0, _rc0, rr1, _rc1 = right["bbox"]
    if (lr0, lr1) != (rr0, rr1):
        return False
    if left["self_symmetric"] or right["self_symmetric"]:
        return False
    return bool(np.array_equal(np.fliplr(left["mask"]), right["mask"]))


def _find_unique_pairs(objects: list[dict], axis_col: int) -> list[tuple[dict, dict]]:
    left_ids = [i for i, obj in enumerate(objects) if _object_side(obj, axis_col) == "L"]
    right_ids = [i for i, obj in enumerate(objects) if _object_side(obj, axis_col) == "R"]
    left_edges = {i: [] for i in left_ids}
    right_edges = {j: [] for j in right_ids}

    for i in left_ids:
        for j in right_ids:
            if _objects_match(objects[i], objects[j], axis_col):
                left_edges[i].append(j)
                right_edges[j].append(i)

    pairs: list[tuple[dict, dict]] = []
    for i in left_ids:
        if len(left_edges[i]) != 1:
            continue
        j = left_edges[i][0]
        if len(right_edges[j]) == 1:
            pairs.append((objects[i], objects[j]))
    pairs.sort(key=lambda pair: (pair[0]["bbox"][0], pair[0]["bbox"][1]))
    return pairs


def _paint_mask(
    out: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int, bg: int
) -> None:
    h, w = mask.shape
    view = out[r0 : r0 + h, c0 : c0 + w]
    assert view.shape == mask.shape, "paint destination out of bounds"
    assert not np.any(view[mask] != int(bg)), "pair destinations must not overlap"
    view[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather each chiral left-right mirror pair against the marked grey axis and erase every non-pairing object.

    Input structure:
    - A dominant background contains several monochrome 8-connected objects.
    - Exactly one grey pixel sits on the top edge and marks a vertical axis column.
    - Every non-grey object lies fully on the left or right side of that axis column.

    Rule:
    - Two objects form a valid pair iff they share a color, share the same row interval, are on opposite sides of the axis, and one bbox-normalized mask is the horizontal flip of the other.
    - Reject any object whose mask is itself left-right symmetric, even if a flipped copy exists.
    - Keep only uniquely matchable valid pairs, sliding each left member horizontally until its right edge touches the column just left of the axis and each right member until its left edge touches the column just right of the axis; keep rows unchanged.
    - Erase every unpaired object and preserve the single grey marker.

    Key insight:
    - The hidden criterion is chirality: identical copies and self-symmetric shapes are noise, while only non-symmetric mirror mates should converge toward the grey axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    axis_col = _find_axis_column(g)
    assert np.all(g[1:, axis_col] == bg), "axis column must be empty below the grey marker"

    objects = _extract_objects(g, bg)
    assert all(_object_side(obj, axis_col) in {"L", "R"} for obj in objects), "objects must not cross the axis"
    pairs = _find_unique_pairs(objects, axis_col)

    out = np.full_like(g, bg)
    out[0, axis_col] = AXIS_COLOR
    for left, right in pairs:
        r0 = int(left["bbox"][0])
        left_w = int(left["mask"].shape[1])
        right_w = int(right["mask"].shape[1])
        _paint_mask(out, left["mask"], left["color"], r0, axis_col - left_w, bg)
        _paint_mask(out, right["mask"], right["color"], r0, axis_col + 1, bg)
        assert axis_col - left_w >= 0 and axis_col + 1 + right_w <= g.shape[1]
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
    mask[start] = True
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[int(RNG.integers(0, len(frontier)))]
        mask[nr, nc] = True
        cells.add((nr, nc))
    return _crop_mask(mask)


def _random_chiral_mask() -> np.ndarray:
    for _ in range(200):
        h = int(RNG.integers(2, 6))
        w = int(RNG.integers(2, 6))
        max_area = h * w - 1
        if max_area < 3:
            continue
        area = int(RNG.integers(3, max_area + 1))
        mask = _grow_connected_mask(h, w, area)
        if mask.size == 0 or mask.shape[1] < 2 or int(mask.sum()) < 3:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample a chiral mask")


def _random_symmetric_mask() -> np.ndarray:
    for _ in range(200):
        h = int(RNG.integers(2, 6))
        w = int(RNG.integers(2, 6))
        half_w = (w + 1) // 2
        max_area = h * half_w
        area = int(RNG.integers(max(2, half_w), max_area + 1))
        half = _grow_connected_mask(h, half_w, area)
        if half.size == 0 or not np.any(half[:, -1]):
            continue
        hh, hw = half.shape
        full_w = hw * 2 if RNG.random() < 0.5 else hw * 2 - 1
        if full_w < 2:
            continue
        full = np.zeros((hh, full_w), dtype=bool)
        full[:, :hw] = half
        if full_w % 2 == 0:
            full[:, hw:] = np.fliplr(half)
        else:
            full[:, hw:] = np.fliplr(half[:, :-1])
        full = _crop_mask(full)
        if int(full.sum()) < 3:
            continue
        if not np.array_equal(full, np.fliplr(full)):
            continue
        if len(_component_coords(full)) != 1:
            continue
        return full
    raise RuntimeError("failed to sample a symmetric mask")


def _choose_color() -> int:
    return int(PALETTE[int(RNG.integers(0, len(PALETTE)))])


def _make_group(kind: str) -> dict:
    color = _choose_color()
    if kind == "pair":
        base = _random_chiral_mask()
        left_mask, right_mask = base, np.fliplr(base)
        if RNG.random() < 0.5:
            left_mask, right_mask = right_mask, left_mask
        return {"kind": kind, "color": color, "left": left_mask, "right": right_mask}
    if kind == "fake":
        base = _random_chiral_mask()
        if RNG.random() < 0.5:
            base = np.fliplr(base)
        return {"kind": kind, "color": color, "left": base, "right": base.copy()}
    if kind == "symmetric":
        base = _random_symmetric_mask()
        return {"kind": kind, "color": color, "left": base, "right": base.copy()}
    if kind == "single":
        base = _random_chiral_mask() if RNG.random() < 0.7 else _random_symmetric_mask()
        side = "left" if RNG.random() < 0.5 else "right"
        return {"kind": kind, "color": color, "left": base if side == "left" else None, "right": base if side == "right" else None}
    raise ValueError(f"unknown group kind: {kind}")


def _group_height(group: dict) -> int:
    heights = []
    for key in ("left", "right"):
        mask = group[key]
        if mask is not None:
            heights.append(int(mask.shape[0]))
    return max(heights)


def _row_pad_mask(mask: np.ndarray, height: int) -> np.ndarray:
    if mask.shape[0] == height:
        return mask.copy()
    top = int(RNG.integers(0, height - mask.shape[0] + 1))
    out = np.zeros((height, mask.shape[1]), dtype=bool)
    out[top : top + mask.shape[0], :] = mask
    return out


def _place_group(
    grid: np.ndarray,
    occupancy: np.ndarray,
    axis_col: int,
    row0: int,
    group: dict,
) -> tuple[list[tuple[int, int, np.ndarray, int]], int] | None:
    height = _group_height(group)
    placements: list[tuple[int, int, np.ndarray, int]] = []
    for side in ("left", "right"):
        mask = group[side]
        if mask is None:
            continue
        padded = _row_pad_mask(mask, height)
        width = int(padded.shape[1])
        if side == "left":
            max_c0 = axis_col - width - 2
            if max_c0 < 0:
                return None
            c0 = int(RNG.integers(0, max_c0 + 1))
        else:
            min_c0 = axis_col + 2
            max_c0 = grid.shape[1] - width
            if min_c0 > max_c0:
                return None
            c0 = int(RNG.integers(min_c0, max_c0 + 1))
        if row0 + height > grid.shape[0]:
            return None
        view = occupancy[row0 : row0 + height, c0 : c0 + width]
        if np.any(view & padded):
            return None
        placements.append((row0, c0, padded, int(group["color"])))
        view[padded] = True
    return placements, height


def _build_input() -> tuple[np.ndarray, int] | None:
    h = int(RNG.integers(18, 31))
    w = int(RNG.integers(18, 31))
    axis_col = int(RNG.integers(6, w - 6))
    grid = np.full((h, w), BG, dtype=int)
    grid[0, axis_col] = AXIS_COLOR
    occupancy = np.zeros((h, w), dtype=bool)
    occupancy[0, axis_col] = True

    group_kinds = (
        ["pair"] * int(RNG.integers(2, 5))
        + ["fake"] * int(RNG.integers(1, 3))
        + ["symmetric"] * int(RNG.integers(1, 3))
        + ["single"] * int(RNG.integers(0, 3))
    )
    RNG.shuffle(group_kinds)
    groups = [_make_group(kind) for kind in group_kinds]

    row = 2
    placed_true_pairs = 0
    for group in groups:
        height = _group_height(group)
        if row + height > h - 1:
            return None
        placed = _place_group(grid, occupancy, axis_col, row, group)
        if placed is None:
            return None
        placements, _ = placed
        for r0, c0, mask, color in placements:
            patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
            patch[mask] = color
        if group["kind"] == "pair":
            placed_true_pairs += 1
        row += height + int(RNG.integers(2, 4))

    if np.count_nonzero(grid == BG) <= (grid.size // 2):
        return None

    return grid, placed_true_pairs


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _build_input()
        if built is None:
            continue
        grid, true_pair_count = built
        axis_col = _find_axis_column(grid)
        assert axis_col >= 0
        objects = _extract_objects(grid, BG)
        pairs = _find_unique_pairs(objects, axis_col)
        if len(pairs) != true_pair_count:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid chirality-axis sample")
