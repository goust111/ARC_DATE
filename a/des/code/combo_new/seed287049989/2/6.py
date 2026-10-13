from __future__ import annotations

import math
import random

import numpy as np


BG = 0
ORANGE = 7
BORDER_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
DIRECTIONS = ("up", "right", "down", "left")
ROTATIONS = {"up": 0, "right": 3, "down": 2, "left": 1}

PAIR_COUNT_RANGE = (1, 3)
SHAPE_SIDE_RANGE = (2, 4)
CONTAINER_EXTRA_RANGE = (0, 2)
GRID_SIDE_RANGE = (16, 30)
PLACEMENT_MARGIN = 1
GENERATION_RETRIES = 400
_GENERATE_COUNTER = 0


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(counts.argmax())])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _container_mask(h: int, w: int, side: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if side == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif side == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif side == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    return mask


def _classify_component(
    grid: np.ndarray, color: int, coords: np.ndarray, bg: int
) -> dict[str, object] | None:
    mask, bbox = _mask_from_coords(coords)
    h, w = mask.shape
    if h < 3 or w < 3:
        return None

    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None

    for side in DIRECTIONS:
        if np.array_equal(mask, _container_mask(h, w, side)) and np.all(inner == bg):
            return {"kind": "container", "color": int(color), "bbox": bbox, "side": side}

    if np.array_equal(mask, _frame_mask(h, w)):
        if not np.any(inner == ORANGE):
            return None
        inner_colors = set(map(int, np.unique(inner)))
        if inner_colors <= {int(bg), ORANGE}:
            pattern = np.where(inner == ORANGE, ORANGE, bg)
            return {
                "kind": "stamp",
                "color": int(color),
                "bbox": bbox,
                "pattern": _crop_to_mask(pattern, inner == ORANGE),
            }
    return None


def _extract_scene(
    grid: np.ndarray,
) -> tuple[int, dict[int, dict[str, object]], dict[int, dict[str, object]]]:
    bg = _most_common_color(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) not in {int(bg), ORANGE}]
    containers: dict[int, dict[str, object]] = {}
    stamps: dict[int, dict[str, object]] = {}

    for color in colors:
        for coords in _components(grid == color):
            obj = _classify_component(grid, color, coords, bg)
            assert obj is not None, "unexpected non-background component"
            if obj["kind"] == "container":
                assert color not in containers, "each color must have one container"
                containers[color] = obj
            else:
                assert color not in stamps, "each color must have one stamp"
                stamps[color] = obj

    assert containers and containers.keys() == stamps.keys(), "color pairing must be unambiguous"
    return int(bg), containers, stamps


def main(grid: np.ndarray) -> np.ndarray:
    """Match each colored stamp to its open-sided container, rotate the stamp, and print it into the container.

    Input structure:
    - The background is black.
    - Several colored container borders appear as axis-aligned three-sided rectangular frames with one open side and empty interiors.
    - Separate colored stamp borders appear as closed one-pixel rectangular frames; each stamp interior contains only black and an orange motif, and each stamp border color matches one container border color.

    Rule:
    - Pair every stamp with the unique container of the same border color.
    - Read the container opening direction and rotate the stamp motif so its original top faces that opening: up = 0 degrees, right = 90 degrees clockwise, down = 180 degrees, left = 90 degrees counterclockwise.
    - Remove each original stamp completely, keep the container border unchanged, and place the rotated orange motif centered inside the matching container interior.

    Key insight:
    - Border color gives the correspondence, while the missing side of the container gives the rotation instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, containers, stamps = _extract_scene(g)
    out = g.copy()

    for stamp in stamps.values():
        r0, c0, r1, c1 = stamp["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for color, container in containers.items():
        stamp = stamps[color]
        pattern = np.asarray(stamp["pattern"], dtype=int)
        rotated = np.rot90(pattern, int(ROTATIONS[str(container["side"])]))

        r0, c0, r1, c1 = container["bbox"]
        inner_h = int(r1 - r0 - 1)
        inner_w = int(c1 - c0 - 1)
        rh, rw = rotated.shape
        assert rh <= inner_h and rw <= inner_w, "rotated motif must fit inside its container"

        top = int(r0 + 1 + (inner_h - rh) // 2)
        left = int(c0 + 1 + (inner_w - rw) // 2)
        window = out[top : top + rh, left : left + rw]
        window[rotated == ORANGE] = ORANGE

    return out


def _array_key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.astype(np.uint8).tobytes()


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    for _ in range(40):
        start = (random.randrange(h), random.randrange(w))
        chosen = {start}
        frontier = {start}
        while len(chosen) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.discard(cell)
            neighbors = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                    neighbors.append((nr, nc))
            random.shuffle(neighbors)
            for nb in neighbors:
                if len(chosen) >= area:
                    break
                chosen.add(nb)
                frontier.add(nb)
        if len(chosen) == area:
            mask = np.zeros((h, w), dtype=bool)
            for r, c in chosen:
                mask[r, c] = True
            return mask
    return None


def _sample_pattern() -> np.ndarray:
    min_side, max_side = SHAPE_SIDE_RANGE
    for _ in range(200):
        h = random.randint(min_side, max_side)
        w = random.randint(min_side, max_side)
        max_area = h * w - 1
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue
        mask = _crop_to_mask(mask.astype(int), mask).astype(bool)
        if min(mask.shape) < 2:
            continue
        if len({_array_key(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        return np.where(mask, ORANGE, BG)
    raise RuntimeError("failed to sample a non-degenerate stamp pattern")


def _build_container_patch(color: int, side: str, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[_container_mask(inner_h + 2, inner_w + 2, side)] = color
    return patch


def _build_stamp_patch(color: int, pattern: np.ndarray) -> np.ndarray:
    ph, pw = pattern.shape
    patch = np.full((ph + 2, pw + 2), BG, dtype=int)
    patch[_frame_mask(ph + 2, pw + 2)] = color
    patch[1 : ph + 1, 1 : pw + 1] = pattern
    return patch


def _sample_directions(count: int) -> list[str]:
    global _GENERATE_COUNTER
    forced = DIRECTIONS[_GENERATE_COUNTER % len(DIRECTIONS)]
    _GENERATE_COUNTER += 1
    directions = [forced]
    while len(directions) < count:
        directions.append(random.choice(DIRECTIONS))
    random.shuffle(directions)
    return directions


def _candidate_positions(
    occupied: np.ndarray, patch_shape: tuple[int, int], margin: int
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch_shape
    positions: list[tuple[int, int]] = []
    for r in range(margin, h - ph - margin + 1):
        for c in range(margin, w - pw - margin + 1):
            r0 = r - margin
            c0 = c - margin
            r1 = r + ph + margin
            c1 = c + pw + margin
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _place_objects(
    patches: list[np.ndarray], grid_shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    occupied = np.zeros(grid_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(
        range(len(patches)),
        key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]),
        reverse=True,
    )
    for idx in order:
        options = _candidate_positions(occupied, patches[idx].shape, PLACEMENT_MARGIN)
        if not options:
            return None
        top, left = random.choice(options)
        placements[idx] = (top, left)
        ph, pw = patches[idx].shape
        occupied[
            top - PLACEMENT_MARGIN : top + ph + PLACEMENT_MARGIN,
            left - PLACEMENT_MARGIN : left + pw + PLACEMENT_MARGIN,
        ] = True
    return [pos for pos in placements if pos is not None]


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    canvas[top : top + h, left : left + w] = patch


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(BORDER_COLORS, pair_count)
    directions = _sample_directions(pair_count)

    objects: list[dict[str, object]] = []
    patches: list[np.ndarray] = []
    for pair_id, (color, side) in enumerate(zip(colors, directions, strict=False)):
        pattern = _sample_pattern()
        rotated = np.rot90(pattern, int(ROTATIONS[side]))
        extra_h = random.randint(*CONTAINER_EXTRA_RANGE)
        extra_w = random.randint(*CONTAINER_EXTRA_RANGE)
        inner_h = int(rotated.shape[0] + extra_h)
        inner_w = int(rotated.shape[1] + extra_w)

        container_patch = _build_container_patch(int(color), side, inner_h, inner_w)
        stamp_patch = _build_stamp_patch(int(color), pattern)

        objects.append(
            {
                "kind": "container",
                "pair_id": pair_id,
                "patch": container_patch,
                "rotated": rotated,
            }
        )
        patches.append(container_patch)
        objects.append({"kind": "stamp", "pair_id": pair_id, "patch": stamp_patch})
        patches.append(stamp_patch)

    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    min_side = max(
        GRID_SIDE_RANGE[0],
        max_h + 2 * PLACEMENT_MARGIN,
        max_w + 2 * PLACEMENT_MARGIN,
        int(math.ceil(math.sqrt(total_area * 1.8))),
    )
    if min_side > GRID_SIDE_RANGE[1]:
        return None

    for _ in range(80):
        rows = random.randint(min_side, GRID_SIDE_RANGE[1])
        cols = random.randint(min_side, GRID_SIDE_RANGE[1])
        placements = _place_objects(patches, (rows, cols))
        if placements is None:
            continue

        inp = np.full((rows, cols), BG, dtype=int)
        out = np.full((rows, cols), BG, dtype=int)
        for obj, (top, left) in zip(objects, placements, strict=False):
            patch = np.asarray(obj["patch"], dtype=int)
            _paste(inp, patch, int(top), int(left))
            if obj["kind"] != "container":
                continue

            _paste(out, patch, int(top), int(left))
            rotated = np.asarray(obj["rotated"], dtype=int)
            inner_h = int(patch.shape[0] - 2)
            inner_w = int(patch.shape[1] - 2)
            rh, rw = rotated.shape
            inner_top = int(top + 1 + (inner_h - rh) // 2)
            inner_left = int(left + 1 + (inner_w - rw) // 2)
            out_slice = out[inner_top : inner_top + rh, inner_left : inner_left + rw]
            out_slice[rotated == ORANGE] = ORANGE

        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _make_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
