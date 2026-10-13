import random
from typing import Optional

import numpy as np


BG = 0
YELLOW = 4
COLOR_POOL = [1, 2, 3, 5, 6, 7, 8, 9]
SIDE_ORDER = ["up", "left", "down", "right"]
GEN_RETRIES = 300
PLACE_RETRIES = 250
SHAPE_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _detect_envelope(mask: np.ndarray) -> Optional[str]:
    h, w = mask.shape
    if h < 3 or w < 3:
        return None
    if mask[1:-1, 1:-1].any():
        return None

    boundary = np.zeros_like(mask, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    if np.any(mask & ~boundary):
        return None

    missing = np.argwhere(boundary & ~mask)
    if len(missing) != 2:
        return None

    (r0, c0), (r1, c1) = missing
    if abs(int(r0) - int(r1)) + abs(int(c0) - int(c1)) != 1:
        return None

    if r0 == r1 == 0:
        return "up"
    if r0 == r1 == h - 1:
        return "down"
    if c0 == c1 == 0:
        return "left"
    if c0 == c1 == w - 1:
        return "right"
    return None


def _marker_candidates(mask: np.ndarray) -> dict[str, list[int]]:
    return {
        "up": [int(c) for c in np.where(~mask[0])[0]],
        "down": [int(c) for c in np.where(~mask[-1])[0]],
        "left": [int(r) for r in np.where(~mask[:, 0])[0]],
        "right": [int(r) for r in np.where(~mask[:, -1])[0]],
    }


def _find_marker_for_stamp(
    bbox: tuple[int, int, int, int], mask: np.ndarray, yellow_positions: set[tuple[int, int]]
) -> Optional[tuple[str, tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    allowed = _marker_candidates(mask)
    found: list[tuple[str, tuple[int, int]]] = []

    for c in allowed["up"]:
        pos = (r0 - 1, c0 + c)
        if pos in yellow_positions:
            found.append(("up", pos))
    for c in allowed["down"]:
        pos = (r1 + 1, c0 + c)
        if pos in yellow_positions:
            found.append(("down", pos))
    for r in allowed["left"]:
        pos = (r0 + r, c0 - 1)
        if pos in yellow_positions:
            found.append(("left", pos))
    for r in allowed["right"]:
        pos = (r0 + r, c1 + 1)
        if pos in yellow_positions:
            found.append(("right", pos))

    if len(found) != 1:
        return None
    return found[0]


def _rotation_steps(marker_dir: str, target_dir: str) -> int:
    return (SIDE_ORDER.index(target_dir) - SIDE_ORDER.index(marker_dir)) % 4


def _analyze_objects(grid: np.ndarray):
    yellow_positions = {tuple(map(int, pos)) for pos in np.argwhere(grid == YELLOW)}
    envelopes: dict[int, dict] = {}
    stamps: dict[int, dict] = {}

    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, YELLOW)):
        for coords in _find_color_components(grid, color):
            bbox = _bbox_from_coords(coords)
            mask = _mask_from_coords(coords, bbox)
            gap_dir = _detect_envelope(mask)
            if gap_dir is not None:
                envelopes[color] = {"color": color, "bbox": bbox, "gap_dir": gap_dir}
                continue
            marker = _find_marker_for_stamp(bbox, mask, yellow_positions)
            if marker is None:
                continue
            marker_dir, marker_pos = marker
            stamps[color] = {
                "color": color,
                "bbox": bbox,
                "coords": coords,
                "mask": mask,
                "marker_dir": marker_dir,
                "marker_pos": marker_pos,
            }
    return envelopes, stamps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == YELLOW] = BG

    envelopes, stamps = _analyze_objects(g)
    for stamp in stamps.values():
        out[stamp["coords"][:, 0], stamp["coords"][:, 1]] = BG

    for color in sorted(set(envelopes) & set(stamps)):
        env = envelopes[color]
        stamp = stamps[color]
        rotated = np.rot90(stamp["mask"].astype(int), _rotation_steps(stamp["marker_dir"], env["gap_dir"])).astype(bool)
        er0, ec0, er1, ec1 = env["bbox"]
        inner_h = er1 - er0 - 1
        inner_w = ec1 - ec0 - 1
        sh, sw = rotated.shape
        top = er0 + 1 + (inner_h - sh) // 2
        left = ec0 + 1 + (inner_w - sw) // 2
        rr, cc = np.nonzero(rotated)
        out[top + rr, left + cc] = color

    return out


def _grow_connected_mask(rng: random.Random, height: int, width: int, area: int) -> np.ndarray:
    cells = {(rng.randrange(height), rng.randrange(width))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _tight_crop(mask)


def _random_stamp_shape(rng: random.Random, marker_dir: str) -> Optional[np.ndarray]:
    for _ in range(SHAPE_RETRIES):
        height = rng.randint(3, 4)
        width = rng.randint(3, 4)
        area = rng.randint(4, min(height * width - 1, 7))
        mask = _grow_connected_mask(rng, height, width, area)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if _detect_envelope(mask) is not None:
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if not _marker_candidates(mask)[marker_dir]:
            continue
        return mask
    return None


def _make_envelope_patch(
    rng: random.Random, color: int, inner_h: int, inner_w: int, gap_dir: str
) -> tuple[np.ndarray, int]:
    patch = np.full((inner_h + 2, inner_w + 2), color, dtype=int)
    patch[1:-1, 1:-1] = BG
    if gap_dir in ("up", "down"):
        start = rng.randrange(1, patch.shape[1] - 2)
        row = 0 if gap_dir == "up" else patch.shape[0] - 1
        patch[row, start : start + 2] = BG
        return patch, start
    start = rng.randrange(1, patch.shape[0] - 2)
    col = 0 if gap_dir == "left" else patch.shape[1] - 1
    patch[start : start + 2, col] = BG
    return patch, start


def _make_stamp_patch(
    rng: random.Random, shape_mask: np.ndarray, color: int, marker_dir: str
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = shape_mask.shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    patch[1 : h + 1, 1 : w + 1][shape_mask] = color
    choices = _marker_candidates(shape_mask)[marker_dir]
    pick = rng.choice(choices)

    if marker_dir == "up":
        marker_pos = (0, pick + 1)
    elif marker_dir == "down":
        marker_pos = (h + 1, pick + 1)
    elif marker_dir == "left":
        marker_pos = (pick + 1, 0)
    else:
        marker_pos = (pick + 1, w + 1)
    patch[marker_pos] = YELLOW
    return patch, marker_pos


def _place_patch(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, rng: random.Random, pad: int = 1
) -> Optional[tuple[int, int]]:
    h, w = patch.shape
    gh, gw = grid.shape
    if h > gh or w > gw:
        return None

    for _ in range(PLACE_RETRIES):
        top = rng.randrange(0, gh - h + 1)
        left = rng.randrange(0, gw - w + 1)
        r0 = max(0, top - pad)
        c0 = max(0, left - pad)
        r1 = min(gh, top + h + pad)
        c1 = min(gw, left + w + pad)
        if reserved[r0:r1, c0:c1].any():
            continue
        sub = grid[top : top + h, left : left + w]
        mask = patch != BG
        sub[mask] = patch[mask]
        reserved[top : top + h, left : left + w] = True
        return top, left
    return None


def _build_expected_output(grid: np.ndarray, pair_data: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pair_data:
        coords = pair["stamp_coords"]
        out[coords[:, 0], coords[:, 1]] = BG
        mr, mc = pair["marker_pos"]
        out[mr, mc] = BG

    for pair in pair_data:
        rotated = np.rot90(
            pair["shape_mask"].astype(int),
            _rotation_steps(pair["marker_dir"], pair["gap_dir"]),
        ).astype(bool)
        er0, ec0, er1, ec1 = pair["envelope_bbox"]
        inner_h = er1 - er0 - 1
        inner_w = ec1 - ec0 - 1
        sh, sw = rotated.shape
        top = er0 + 1 + (inner_h - sh) // 2
        left = ec0 + 1 + (inner_w - sw) // 2
        rr, cc = np.nonzero(rotated)
        out[top + rr, left + cc] = pair["color"]
    return out


def _generate_attempt(rng: random.Random) -> Optional[np.ndarray]:
    pair_count = rng.randint(1, 3)
    colors = rng.sample(COLOR_POOL, pair_count)
    pair_specs = []
    total_reserved_area = 0
    max_dim = 0

    for color in colors:
        marker_dir = rng.choice(SIDE_ORDER)
        gap_dir = rng.choice(SIDE_ORDER)
        shape_mask = _random_stamp_shape(rng, marker_dir)
        if shape_mask is None:
            return None

        max_side = max(shape_mask.shape)
        inner_h = rng.randint(max_side + 1, max_side + 2)
        inner_w = rng.randint(max_side + 1, max_side + 2)
        envelope_patch, _gap_start = _make_envelope_patch(rng, color, inner_h, inner_w, gap_dir)
        stamp_patch, marker_local = _make_stamp_patch(rng, shape_mask, color, marker_dir)

        pair_specs.append(
            {
                "color": color,
                "gap_dir": gap_dir,
                "marker_dir": marker_dir,
                "shape_mask": shape_mask,
                "envelope_patch": envelope_patch,
                "stamp_patch": stamp_patch,
                "marker_local": marker_local,
            }
        )
        total_reserved_area += int(np.prod(envelope_patch.shape) + np.prod(stamp_patch.shape))
        max_dim = max(max_dim, envelope_patch.shape[0], envelope_patch.shape[1], stamp_patch.shape[0], stamp_patch.shape[1])

    min_side = max(18, max_dim + 2, int(np.ceil(np.sqrt(total_reserved_area * 2.4))))
    max_side = min(30, min_side + 6)
    if min_side > max_side:
        return None

    height = rng.randint(min_side, max_side)
    width = rng.randint(min_side, max_side)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    pair_data = []

    objects = []
    for idx, spec in enumerate(pair_specs):
        env_area = int(np.prod(spec["envelope_patch"].shape))
        stamp_area = int(np.prod(spec["stamp_patch"].shape))
        objects.append((env_area, idx, "envelope"))
        objects.append((stamp_area, idx, "stamp"))
    objects.sort(reverse=True)

    placed: dict[tuple[int, str], tuple[int, int]] = {}
    for _area, idx, kind in objects:
        patch = pair_specs[idx]["envelope_patch"] if kind == "envelope" else pair_specs[idx]["stamp_patch"]
        pos = _place_patch(grid, reserved, patch, rng, pad=1)
        if pos is None:
            return None
        placed[(idx, kind)] = pos

    for idx, spec in enumerate(pair_specs):
        env_top, env_left = placed[(idx, "envelope")]
        stamp_top, stamp_left = placed[(idx, "stamp")]
        shape_mask = spec["shape_mask"]
        rr, cc = np.nonzero(shape_mask)
        stamp_coords = np.stack((stamp_top + 1 + rr, stamp_left + 1 + cc), axis=1).astype(int)
        marker_r, marker_c = spec["marker_local"]
        pair_data.append(
            {
                "color": spec["color"],
                "gap_dir": spec["gap_dir"],
                "marker_dir": spec["marker_dir"],
                "shape_mask": shape_mask,
                "envelope_bbox": (
                    env_top,
                    env_left,
                    env_top + spec["envelope_patch"].shape[0] - 1,
                    env_left + spec["envelope_patch"].shape[1] - 1,
                ),
                "stamp_coords": stamp_coords,
                "marker_pos": (stamp_top + marker_r, stamp_left + marker_c),
            }
        )

    expected = _build_expected_output(grid, pair_data)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
