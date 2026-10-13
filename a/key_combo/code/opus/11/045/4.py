from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
OBJ = 5
MARKER_COLORS = (1, 2, 3)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR_NAMES = ("up", "right", "down", "left")
MIN_RAY_LEN = 3
MAX_GEN_ATTEMPTS = 400
MIN_NONBG_CHEB = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _rotate_cw(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, -int(k))


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    variants = []
    for k in range(4):
        rotated = _rotate_cw(mask, k)
        variants.append((tuple(rotated.shape), tuple(int(v) for v in rotated.astype(np.uint8).ravel())))
    return min(variants)


def _rotation_steps(from_mask: np.ndarray, to_mask: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(_rotate_cw(from_mask, k), to_mask)]
    assert len(matches) == 1, "shape orientation must be uniquely determined"
    return int(matches[0])


def _rotate_dir(dir_idx: int, k: int) -> int:
    return (int(dir_idx) + int(k)) % 4


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _front_anchor(bbox: tuple[int, int, int, int], dir_idx: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if dir_idx == 0:
        return r0 - 1, mid_c
    if dir_idx == 1:
        return mid_r, c1 + 1
    if dir_idx == 2:
        return r1 + 1, mid_c
    return mid_r, c0 - 1


def _ray_cells(shape: tuple[int, int], bbox: tuple[int, int, int, int], dir_idx: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    r, c = _front_anchor(bbox, dir_idx)
    cells = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _parse_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _components_of_color(grid, OBJ):
        bbox = _bbox(coords)
        mask = _mask_from_coords(coords)
        markers = []
        for dir_idx in range(4):
            ar, ac = _front_anchor(bbox, dir_idx)
            if _in_bounds(grid.shape, ar, ac) and int(grid[ar, ac]) in MARKER_COLORS:
                markers.append((dir_idx, (ar, ac), int(grid[ar, ac])))
        assert len(markers) <= 1, "each object can have at most one front marker"
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "signature": _shape_signature(mask),
                "marker": markers[0] if markers else None,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group grey placeholder shapes by rotation-normalized form, then project colored rays using each instance's rotation relative to its marked exemplar.

    Input structure:
    - Black is the background.
    - Grey pixels form 6-9 separated 4-connected placeholder objects.
    - Objects fall into 2-3 shape groups under translation normalization and 90-degree rotation.
    - Each group has exactly one marked exemplar: one red, blue, or green pixel sits just outside the midpoint of one bbox side.

    Rule:
    - Partition grey objects into groups by normalized mask, treating 90-degree rotations as the same shape.
    - In each group, the marked exemplar provides both the group's ray color and its canonical front direction.
    - For every unmarked object, compute the unique clockwise rotation that maps the exemplar mask to that object's mask.
    - Rotate the exemplar's front direction by that amount, then draw a one-pixel-wide ray from the background cell immediately beyond that object's front-side bbox midpoint until a non-black cell or the grid boundary is reached.
    - Keep all original grey objects and colored marker pixels unchanged.

    Key insight:
    The ray direction is not absolute: it is inherited from the marked exemplar and must be rotated by the instance's relative orientation inside its shape group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _parse_objects(g)
    assert 6 <= len(objects) <= 9, "expected 6-9 grey objects"

    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = defaultdict(list)
    for obj in objects:
        groups[obj["signature"]].append(obj)
    assert 2 <= len(groups) <= 3, "expected 2-3 shape groups"

    out = g.copy()
    for members in groups.values():
        marked = [obj for obj in members if obj["marker"] is not None]
        assert len(marked) == 1, "each shape group must have exactly one marked exemplar"
        exemplar = marked[0]
        exemplar_front, _marker_pos, ray_color = exemplar["marker"]
        for obj in members:
            if obj is exemplar:
                continue
            rotation = _rotation_steps(exemplar["mask"], obj["mask"])
            front_dir = _rotate_dir(exemplar_front, rotation)
            for r, c in _ray_cells(g.shape, obj["bbox"], front_dir):
                if int(g[r, c]) != BG:
                    break
                assert int(out[r, c]) == BG, "generated rays must not overlap"
                out[r, c] = int(ray_color)
    return out


def _mask_from_strings(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


TEMPLATE_MASKS = [
    _mask_from_strings(["110", "111", "010"]),
    _mask_from_strings(["010", "110", "011", "010", "010"]),
    _mask_from_strings(["00100", "11110", "00011"]),
]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _validate_template_catalog() -> None:
    sigs = set()
    for mask in TEMPLATE_MASKS:
        h, w = mask.shape
        area = int(mask.sum())
        assert h % 2 == 1 and w % 2 == 1
        assert 4 <= area <= 7
        assert _is_connected(mask)
        assert not np.array_equal(mask, np.rot90(mask, 2))
        assert not np.array_equal(mask, np.fliplr(mask))
        assert not np.array_equal(mask, np.flipud(mask))
        sig = _shape_signature(mask)
        assert sig not in sigs, "template groups must be distinct up to rotation"
        sigs.add(sig)


_validate_template_catalog()


def _group_size_options(group_count: int) -> list[tuple[int, ...]]:
    if group_count == 2:
        return [(2, 4), (3, 3), (3, 4), (4, 4)]
    return [(2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 3, 3), (2, 3, 4), (3, 3, 3)]


def _build_instances(rng: random.Random) -> list[dict]:
    group_count = rng.choice((2, 3))
    group_sizes = list(rng.choice(_group_size_options(group_count)))
    rng.shuffle(group_sizes)
    template_ids = rng.sample(range(len(TEMPLATE_MASKS)), k=group_count)
    colors = list(rng.sample(list(MARKER_COLORS), k=group_count))
    rng.shuffle(colors)

    instances = []
    for group_index, (template_id, group_size, color) in enumerate(zip(template_ids, group_sizes, colors)):
        chosen_orientations = rng.sample(range(4), k=group_size)
        canonical_orientation = rng.choice(chosen_orientations)
        canonical_front = rng.randrange(4)
        for orientation in chosen_orientations:
            mask = _rotate_cw(TEMPLATE_MASKS[template_id], orientation)
            rel_rotation = (orientation - canonical_orientation) % 4
            front_dir = _rotate_dir(canonical_front, rel_rotation)
            instances.append(
                {
                    "group_index": group_index,
                    "template_id": template_id,
                    "color": color,
                    "canonical": orientation == canonical_orientation,
                    "orientation": orientation,
                    "canonical_orientation": canonical_orientation,
                    "canonical_front": canonical_front,
                    "front_dir": front_dir,
                    "rel_rotation": rel_rotation,
                    "mask": mask,
                }
            )
    return instances


def _candidate_origins(
    grid_shape: tuple[int, int],
    mask_shape: tuple[int, int],
    dir_idx: int,
    canonical: bool,
    rng: random.Random,
) -> list[tuple[int, int]]:
    H, W = grid_shape
    h, w = mask_shape
    r_lo, r_hi = 0, H - h
    c_lo, c_hi = 0, W - w
    need = 1 if canonical else MIN_RAY_LEN
    if dir_idx == 0:
        r_lo = need
    elif dir_idx == 1:
        c_hi = W - w - need
    elif dir_idx == 2:
        r_hi = H - h - need
    else:
        c_lo = need
    if r_lo > r_hi or c_lo > c_hi:
        return []
    origins = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    rng.shuffle(origins)
    return origins


def _too_close(cells: list[tuple[int, int]], occupied_nonbg: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr, cc in occupied_nonbg:
            if max(abs(r - rr), abs(c - cc)) < MIN_NONBG_CHEB:
                return True
    return False


def _place_instances(instances: list[dict], grid_shape: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    grid = np.zeros(grid_shape, dtype=int)
    reserved = np.zeros(grid_shape, dtype=bool)
    occupied_nonbg: list[tuple[int, int]] = []

    ordered = sorted(
        instances,
        key=lambda inst: (
            inst["canonical"],
            -(inst["mask"].shape[0] * inst["mask"].shape[1]),
        ),
    )
    for inst in ordered:
        mask = inst["mask"]
        h, w = mask.shape
        placed = False
        for r0, c0 in _candidate_origins(grid_shape, (h, w), int(inst["front_dir"]), bool(inst["canonical"]), rng):
            coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            path = _ray_cells(grid_shape, bbox, int(inst["front_dir"]))
            if not path:
                continue
            if not inst["canonical"] and len(path) < MIN_RAY_LEN:
                continue
            object_cells = [tuple(int(v) for v in cell) for cell in coords]
            anchor = path[0]
            input_cells = list(object_cells)
            if inst["canonical"]:
                input_cells.append(anchor)
            if reserved[coords[:, 0], coords[:, 1]].any():
                continue
            if any(reserved[r, c] for r, c in path):
                continue
            if inst["canonical"] and int(grid[anchor]) != BG:
                continue
            if _too_close(input_cells, occupied_nonbg):
                continue

            grid[coords[:, 0], coords[:, 1]] = OBJ
            reserved[coords[:, 0], coords[:, 1]] = True
            for pr, pc in path:
                reserved[pr, pc] = True
            occupied_nonbg.extend(object_cells)
            if inst["canonical"]:
                grid[anchor] = int(inst["color"])
                occupied_nonbg.append(anchor)

            inst["bbox"] = bbox
            inst["coords"] = coords
            inst["path"] = path
            inst["anchor"] = anchor
            placed = True
            break
        if not placed:
            return None
    return grid


def _expected_from_instances(grid: np.ndarray, instances: list[dict]) -> np.ndarray:
    out = grid.copy()
    for inst in instances:
        if inst["canonical"]:
            continue
        for r, c in inst["path"]:
            if int(grid[r, c]) != BG:
                break
            out[r, c] = int(inst["color"])
    return out


def _wrong_without_rotation(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = defaultdict(list)
    for obj in objects:
        groups[obj["signature"]].append(obj)
    for members in groups.values():
        exemplar = [obj for obj in members if obj["marker"] is not None][0]
        exemplar_front, _marker_pos, ray_color = exemplar["marker"]
        for obj in members:
            if obj is exemplar:
                continue
            for r, c in _ray_cells(grid.shape, obj["bbox"], exemplar_front):
                if int(grid[r, c]) != BG:
                    break
                if int(out[r, c]) != BG:
                    break
                out[r, c] = int(ray_color)
    return out


def _validate_generated(grid: np.ndarray, instances: list[dict]) -> bool:
    if grid is None:
        return False
    objects = _parse_objects(grid)
    if not (6 <= len(objects) <= 9):
        return False
    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = defaultdict(list)
    for obj in objects:
        groups[obj["signature"]].append(obj)
    if not (2 <= len(groups) <= 3):
        return False
    if any(len([obj for obj in members if obj["marker"] is not None]) != 1 for members in groups.values()):
        return False
    if len({len(members) for members in groups.values()}) == 1 and len(groups) == 1:
        return False
    if len({tuple(obj["bbox"]) for obj in objects}) != len(objects):
        return False

    ray_directions = []
    has_nonzero_rotation_per_group = defaultdict(bool)
    for inst in instances:
        if not inst["canonical"]:
            ray_directions.append(int(inst["front_dir"]))
            has_nonzero_rotation_per_group[int(inst["group_index"])] |= int(inst["rel_rotation"]) != 0
            if len(inst["path"]) < MIN_RAY_LEN:
                return False
    if len(set(ray_directions)) < 2:
        return False
    if not all(has_nonzero_rotation_per_group.values()):
        return False

    expected = _expected_from_instances(grid, instances)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(expected, grid):
        return False
    if np.array_equal(_wrong_without_rotation(grid, objects), expected):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(20, 27)
    W = rng.randint(20, 27)
    instances = _build_instances(rng)
    grid = _place_instances(instances, (H, W), rng)
    if not _validate_generated(grid, instances):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
