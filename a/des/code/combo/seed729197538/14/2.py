from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]

GEN_RETRIES = 300
CONTAINER_RETRIES = 160
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 500


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best: tuple[int, int, bytes] | None = None
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _looks_like_container(crop: np.ndarray) -> bool:
    h, w = crop.shape
    if h < 9 or w < 9:
        return False
    if not np.all(crop[-1, :] == GREY):
        return False
    if not np.all(crop[:, 0] == GREY):
        return False
    if not np.all(crop[:, -1] == GREY):
        return False
    marker_cols = np.where((crop[0, :] != GREY) & (crop[0, :] != BLACK))[0]
    if marker_cols.size not in (1, 2):
        return False
    if marker_cols.size == 2 and int(marker_cols[1]) != int(marker_cols[0]) + 1:
        return False
    marker_colors = set(int(x) for x in crop[0, marker_cols])
    if len(marker_colors) != 1:
        return False
    top_ok = np.all((crop[0, :] == GREY) | (crop[0, :] == next(iter(marker_colors))))
    return bool(top_ok and np.any(crop == BLACK))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scattered piece to a same-shaped black hole inside a grey container,
    allowing quarter turns, then fill that hole with the container's top-band color.
    The original piece locations are cleared to black.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    components = sorted(_components_from_mask(g != BLACK), key=_bbox)
    hole_targets: dict[tuple[int, int, bytes], dict[str, object]] = {}
    pieces: list[dict[str, object]] = []

    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(int(g[r, c]) for r, c in coords)

        if GREY in colors:
            assert _looks_like_container(crop), "grey component must be a container"
            marker_color = int(next(iter(set(int(x) for x in crop[0, :] if x not in (BLACK, GREY)))))
            black_components = _components_from_mask(crop == BLACK)
            assert len(black_components) >= 2, "container must have one cavity and at least one hole"
            black_components.sort(key=lambda comp: (-int(comp.shape[0]), _bbox(comp)))
            cavity = max(black_components, key=lambda comp: int(comp.shape[0]))
            cavity_area = int(cavity.shape[0])
            for hole in black_components:
                if hole is cavity:
                    continue
                assert int(hole.shape[0]) < cavity_area, "largest black region must be the interior cavity"
                mask = _tight_mask(hole)
                key = _shape_key(mask)
                assert key not in hole_targets, "hole shapes must be unique up to rotation"
                hole_targets[key] = {
                    "coords": hole + np.array([r0, c0]),
                    "color": marker_color,
                }
        else:
            assert len(colors) == 1, "each scattered fragment must be monochrome"
            color = next(iter(colors))
            assert color not in (BLACK, GREY), "fragment colors must be non-black and non-grey"
            mask = _tight_mask(coords)
            pieces.append({"coords": coords, "key": _shape_key(mask)})

    assert hole_targets, "expected at least one container hole"
    assert pieces, "expected at least one scattered piece"
    assert len(hole_targets) == len(pieces), "piece count must equal hole count"

    seen_piece_keys: set[tuple[int, int, bytes]] = set()
    for piece in pieces:
        key = piece["key"]
        assert key not in seen_piece_keys, "piece shapes must be unique up to rotation"
        seen_piece_keys.add(key)
        assert key in hole_targets, "every piece must match a container hole"

        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK
        hole_info = hole_targets.pop(key)
        hole_coords = hole_info["coords"]
        out[hole_coords[:, 0], hole_coords[:, 1]] = int(hole_info["color"])

    assert not hole_targets, "every hole must receive a piece"
    return out


def _random_connected_shape(max_h: int, max_w: int, area_min: int, area_max: int) -> np.ndarray | None:
    if max_h <= 0 or max_w <= 0:
        return None
    for _ in range(SHAPE_RETRIES):
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        upper = min(area_max, h * w)
        if upper < area_min:
            continue
        target = random.randint(area_min, upper)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask = _tight_mask(coords)
        if 1 <= mask.shape[0] <= max_h and 1 <= mask.shape[1] <= max_w:
            return mask
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _candidate_positions(side: str, container_shape: tuple[int, int], mask_shape: tuple[int, int], band: int) -> list[tuple[int, int]]:
    H, W = container_shape
    h, w = mask_shape
    positions: list[tuple[int, int]] = []
    if side == "top":
        row_values = range(1, 1 + band - h + 1)
        col_values = range(2, W - 2 - w + 1)
    elif side == "bottom":
        row_values = range(H - 1 - band, H - 1 - band + band - h + 1)
        col_values = range(2, W - 2 - w + 1)
    elif side == "left":
        row_values = range(2, H - 2 - h + 1)
        col_values = range(1, 1 + band - w + 1)
    else:
        row_values = range(2, H - 2 - h + 1)
        col_values = range(W - 1 - band, W - 1 - band + band - w + 1)
    for r in row_values:
        for c in col_values:
            positions.append((int(r), int(c)))
    random.shuffle(positions)
    return positions


def _build_container(marker_color: int, hole_count: int, used_keys: set[tuple[int, int, bytes]]) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    for _ in range(CONTAINER_RETRIES):
        thickness = random.choice([4, 5])
        min_side = 2 * thickness + 3
        H = random.randint(min_side, min_side + 4)
        W = random.randint(min_side, min_side + 5)
        band = thickness - 2
        cavity_h = H - 2 * thickness
        cavity_w = W - 2 * thickness
        cavity_area = cavity_h * cavity_w
        hole_area_max = min(6, cavity_area - 1)
        if hole_area_max < 2:
            continue

        patch = np.full((H, W), GREY, dtype=int)
        patch[thickness : H - thickness, thickness : W - thickness] = BLACK
        hole_records: list[dict[str, object]] = []
        local_keys: set[tuple[int, int, bytes]] = set()
        used_bboxes: list[tuple[int, int, int, int]] = []

        ok = True
        for side in random.sample(SIDES, hole_count):
            placed = False
            for _ in range(SHAPE_RETRIES):
                if side in ("top", "bottom"):
                    shape = _random_connected_shape(band, min(5, W - 4), 2, hole_area_max)
                else:
                    shape = _random_connected_shape(min(5, H - 4), band, 2, hole_area_max)
                if shape is None:
                    continue
                key = _shape_key(shape)
                if key in used_keys or key in local_keys:
                    continue
                for top, left in _candidate_positions(side, patch.shape, shape.shape, band):
                    bbox = (top, left, top + shape.shape[0] - 1, left + shape.shape[1] - 1)
                    if any(_boxes_overlap(bbox, other, pad=1) for other in used_bboxes):
                        continue
                    local_coords = np.argwhere(shape) + np.array([top, left])
                    patch[local_coords[:, 0], local_coords[:, 1]] = BLACK
                    hole_records.append(
                        {
                            "coords": local_coords,
                            "mask": shape.copy(),
                            "key": key,
                            "color": marker_color,
                        }
                    )
                    local_keys.add(key)
                    used_bboxes.append(bbox)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        marker_len = random.choice([1, 2])
        marker_start = random.randint(1, W - marker_len - 2)
        patch[0, marker_start : marker_start + marker_len] = marker_color
        used_keys.update(local_keys)
        return patch, hole_records
    return None


def _choose_canvas(dims: Iterable[tuple[int, int]]) -> tuple[int, int]:
    dims = list(dims)
    padded_area = sum((h + 2) * (w + 2) for h, w in dims)
    max_h = max(h for h, _ in dims) + 2
    max_w = max(w for _, w in dims) + 2
    side_base = int(np.ceil(np.sqrt(padded_area))) + 1
    H_min = max(24, max_h, side_base)
    W_min = max(24, max_w, side_base)
    return random.randint(H_min, H_min + 6), random.randint(W_min, W_min + 6)


def _reserve_box(mask: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(mask.shape[0], top + h + 1)
    c1 = min(mask.shape[1], left + w + 1)
    mask[r0:r1, c0:c1] = True


def _find_free_box(reserved: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = reserved.shape
    if h > H - 2 or w > W - 2:
        return None
    row_lo = 1
    row_hi = H - h - 1
    col_lo = 1
    col_hi = W - w - 1
    if row_hi < row_lo or col_hi < col_lo:
        return None
    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(row_lo, row_hi)
        left = random.randint(col_lo, col_hi)
        r0 = top - 1
        c0 = left - 1
        r1 = top + h + 1
        c1 = left + w + 1
        if not reserved[r0:r1, c0:c1].any():
            return top, left
    return None


def _gen_attempt() -> np.ndarray | None:
    container_count = random.choice([2, 3])
    max_holes = min(5, 2 * container_count)
    total_holes = random.randint(container_count, max_holes)
    holes_per_container = [1] * container_count
    for _ in range(total_holes - container_count):
        choices = [idx for idx, count in enumerate(holes_per_container) if count < 2]
        holes_per_container[random.choice(choices)] += 1

    random.shuffle(PALETTE)
    marker_colors = list(PALETTE[:container_count])
    piece_colors = list(PALETTE[container_count : container_count + total_holes])

    used_keys: set[tuple[int, int, bytes]] = set()
    containers: list[dict[str, object]] = []
    piece_templates: list[dict[str, object]] = []
    piece_color_idx = 0

    for marker_color, hole_count in zip(marker_colors, holes_per_container):
        built = _build_container(marker_color, hole_count, used_keys)
        if built is None:
            return None
        patch, holes = built
        container_entry = {"patch": patch, "holes": holes, "marker_color": marker_color}
        containers.append(container_entry)
        for hole in holes:
            piece_mask = np.rot90(hole["mask"], random.randrange(4))
            piece_color = piece_colors[piece_color_idx]
            piece_color_idx += 1
            piece_templates.append(
                {
                    "mask": piece_mask,
                    "color": piece_color,
                    "key": hole["key"],
                }
            )

    dims = [tuple(container["patch"].shape) for container in containers] + [tuple(piece["mask"].shape) for piece in piece_templates]
    H, W = _choose_canvas(dims)
    grid = np.zeros((H, W), dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    hole_targets: dict[tuple[int, int, bytes], dict[str, object]] = {}
    placed_containers = sorted(containers, key=lambda item: int(np.prod(item["patch"].shape)), reverse=True)
    for container in placed_containers:
        patch = container["patch"]
        h, w = patch.shape
        pos = _find_free_box(reserved, h, w)
        if pos is None:
            return None
        top, left = pos
        grid[top : top + h, left : left + w] = patch
        _reserve_box(reserved, top, left, h, w)
        for hole in container["holes"]:
            hole_targets[hole["key"]] = {
                "coords": hole["coords"] + np.array([top, left]),
                "color": container["marker_color"],
            }

    piece_records: list[dict[str, object]] = []
    placed_pieces = sorted(piece_templates, key=lambda item: int(np.prod(item["mask"].shape)), reverse=True)
    for piece in placed_pieces:
        mask = piece["mask"]
        h, w = mask.shape
        pos = _find_free_box(reserved, h, w)
        if pos is None:
            return None
        top, left = pos
        grid[top : top + h, left : left + w][mask] = int(piece["color"])
        _reserve_box(reserved, top, left, h, w)
        piece_records.append({"coords": np.argwhere(mask) + np.array([top, left]), "key": piece["key"]})

    expected = grid.copy()
    for piece in piece_records:
        coords = piece["coords"]
        expected[coords[:, 0], coords[:, 1]] = BLACK
        hole = hole_targets[piece["key"]]
        hole_coords = hole["coords"]
        expected[hole_coords[:, 0], hole_coords[:, 1]] = int(hole["color"])

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
