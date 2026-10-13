from __future__ import annotations

import numpy as np


BG = 0
MARKER = 8
SIDES = ("up", "down", "left", "right")
RING_PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
_RNG = np.random.default_rng()

GEN_ATTEMPTS = 400
RING_ATTEMPTS = 120
MIN_RING_SIZE = 6
GRID_MIN_BY_COUNT = {1: 18, 2: 22, 3: 26}
GRID_MAX = 30
MAX_RING_SIZE_BY_COUNT = {1: 12, 2: 9, 3: 7}


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rectangular_ring(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = sub.shape
    if height < 3 or width < 3:
        return None
    border = _border_mask(height, width)
    if not np.array_equal(sub == int(color), border):
        return None
    return r0, c0, r1, c1


def _nearest_side(inner_r: int, inner_c: int, inner_h: int, inner_w: int) -> str | None:
    dists = {
        "up": int(inner_r),
        "down": int(inner_h - 1 - inner_r),
        "left": int(inner_c),
        "right": int(inner_w - 1 - inner_c),
    }
    best = min(dists.values())
    matches = [side for side, dist in dists.items() if dist == best]
    if len(matches) != 1:
        return None
    return matches[0]


def _copy_box(r0: int, c0: int, height: int, width: int, side: str) -> tuple[int, int, int, int]:
    copy_h = height // 2
    copy_w = width // 2
    if side == "up":
        return r0 - copy_h, c0 + (width - copy_w) // 2, copy_h, copy_w
    if side == "down":
        return r0 + height, c0 + (width - copy_w) // 2, copy_h, copy_w
    if side == "left":
        return r0 + (height - copy_h) // 2, c0 - copy_w, copy_h, copy_w
    assert side == "right"
    return r0 + (height - copy_h) // 2, c0 + width, copy_h, copy_w


def _clip_box(
    box: tuple[int, int, int, int], grid_h: int, grid_w: int
) -> tuple[int, int, int, int] | None:
    r0, c0, height, width = box
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(int(grid_h), int(r0 + height))
    cc1 = min(int(grid_w), int(c0 + width))
    if rr0 >= rr1 or cc0 >= cc1:
        return None
    return rr0, cc0, rr1, cc1


def _clip_status(r0: int, c0: int, height: int, width: int, side: str, grid_h: int, grid_w: int) -> str:
    box = _copy_box(r0, c0, height, width, side)
    clipped = _clip_box(box, grid_h, grid_w)
    if clipped is None:
        return "full"
    br0, bc0, bh, bw = box
    original = (int(br0), int(bc0), int(br0 + bh), int(bc0 + bw))
    if clipped == original:
        return "none"
    return "partial"


def _footprint_bbox(
    r0: int, c0: int, height: int, width: int, sides: list[str], grid_h: int, grid_w: int
) -> tuple[int, int, int, int]:
    boxes = [(int(r0), int(c0), int(r0 + height), int(c0 + width))]
    for side in sides:
        clipped = _clip_box(_copy_box(r0, c0, height, width, side), grid_h, grid_w)
        if clipped is not None:
            boxes.append(clipped)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    return (
        a[2] + margin <= b[0]
        or b[2] + margin <= a[0]
        or a[3] + margin <= b[1]
        or b[3] + margin <= a[1]
    )


def _draw_ring(grid: np.ndarray, color: int, r0: int, c0: int, height: int, width: int) -> None:
    h, w = grid.shape
    r1 = int(r0 + height - 1)
    c1 = int(c0 + width - 1)
    for c in range(int(c0), int(c0 + width)):
        if 0 <= r0 < h and 0 <= c < w:
            grid[int(r0), int(c)] = int(color)
        if 0 <= r1 < h and 0 <= c < w:
            grid[int(r1), int(c)] = int(color)
    for r in range(int(r0 + 1), int(r0 + height - 1)):
        if 0 <= r < h and 0 <= c0 < w:
            grid[int(r), int(c0)] = int(color)
        if 0 <= r < h and 0 <= c1 < w:
            grid[int(r), int(c1)] = int(color)


def _marker_candidates(height: int, width: int) -> dict[str, list[tuple[int, int]]]:
    inner_h = int(height - 2)
    inner_w = int(width - 2)
    out = {side: [] for side in SIDES}
    for inner_r in range(inner_h):
        for inner_c in range(inner_w):
            side = _nearest_side(inner_r, inner_c, inner_h, inner_w)
            if side is None:
                continue
            out[side].append((int(inner_r + 1), int(inner_c + 1)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate each hollow rectangle outside itself according to the side-biased marker pixels inside it.

    Input structure:
    - The background is black.
    - Several separated one-pixel-thick rectangular rings appear, each in a distinct non-marker color.
    - Each ring contains 1-4 pixels of a fixed marker color inside its hollow interior.

    Rule:
    - For every ring, examine each marker pixel and assign it to the unique nearest interior edge: top, bottom, left, or right.
    - Erase all marker pixels from the ring interiors.
    - For each assigned direction, place a same-colored half-size rectangular ring outside that side, touching the original outer wall and centered on that side.
    - The copy height and width are floor(original_height/2) and floor(original_width/2); any part outside the grid is clipped away.

    Key insight:
    - The interior markers do not encode absolute destinations; their bias toward one interior edge chooses which exterior side receives a reduced copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, MARKER)]

    for color in colors:
        bbox = _is_rectangular_ring(g, color)
        assert bbox is not None, "foreground objects must be rectangular rings"
        r0, c0, r1, c1 = bbox
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        copy_h = height // 2
        copy_w = width // 2
        assert copy_h >= 3 and copy_w >= 3, "scaled copies must remain rings"

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere(inner == MARKER)
        assert markers.size > 0, "each ring must have at least one marker"

        directions: list[str] = []
        for inner_r, inner_c in markers:
            side = _nearest_side(int(inner_r), int(inner_c), inner.shape[0], inner.shape[1])
            assert side is not None, "marker must have a unique nearest side"
            directions.append(side)

        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_out[inner == MARKER] = BG
        for side in directions:
            rr0, cc0, hh, ww = _copy_box(r0, c0, height, width, side)
            _draw_ring(out, color, rr0, cc0, hh, ww)

    return out


def _sample_ring(
    grid_h: int,
    grid_w: int,
    reserved: list[tuple[int, int, int, int]],
    total_rings: int,
    force_clip: bool,
) -> dict | None:
    max_size = int(MAX_RING_SIZE_BY_COUNT[total_rings])
    for _ in range(RING_ATTEMPTS):
        height = int(_RNG.integers(MIN_RING_SIZE, max_size + 1))
        width = int(_RNG.integers(MIN_RING_SIZE, max_size + 1))
        if height // 2 < 3 or width // 2 < 3:
            continue

        marker_count = int(_RNG.integers(1, 5))
        dir_order = list(SIDES)
        _RNG.shuffle(dir_order)
        directions = dir_order[:marker_count]

        candidates = _marker_candidates(height, width)
        if any(not candidates[side] for side in directions):
            continue

        clip_side = None
        if force_clip:
            clip_side = directions[int(_RNG.integers(0, len(directions)))]

        positions: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for r0 in range(grid_h - height + 1):
            for c0 in range(grid_w - width + 1):
                okay = True
                for side in directions:
                    status = _clip_status(r0, c0, height, width, side, grid_h, grid_w)
                    if force_clip and side == clip_side:
                        if status != "partial":
                            okay = False
                            break
                    elif status != "none":
                        okay = False
                        break
                if not okay:
                    continue

                footprint = _footprint_bbox(r0, c0, height, width, directions, grid_h, grid_w)
                if any(not _boxes_separated(footprint, other, margin=1) for other in reserved):
                    continue
                positions.append((int(r0), int(c0), footprint))

        if not positions:
            continue

        r0, c0, footprint = positions[int(_RNG.integers(0, len(positions)))]
        markers: list[tuple[int, int]] = []
        used = set()
        for side in directions:
            options = [cell for cell in candidates[side] if cell not in used]
            if not options:
                markers = []
                break
            cell = options[int(_RNG.integers(0, len(options)))]
            used.add(cell)
            markers.append(cell)
        if not markers:
            continue

        return {
            "r0": int(r0),
            "c0": int(c0),
            "height": int(height),
            "width": int(width),
            "directions": list(directions),
            "markers": list(markers),
            "footprint": footprint,
            "forced_clip_side": clip_side,
        }

    return None


def _make_candidate() -> np.ndarray | None:
    ring_count = int(_RNG.integers(1, 4))
    min_size = int(GRID_MIN_BY_COUNT[ring_count])
    grid_h = int(_RNG.integers(min_size, GRID_MAX + 1))
    grid_w = int(_RNG.integers(min_size, GRID_MAX + 1))
    want_clip = bool(_RNG.integers(0, 2))

    if ring_count > len(RING_PALETTE):
        return None
    colors = list(_RNG.choice(np.array(RING_PALETTE, dtype=int), size=ring_count, replace=False))

    reserved: list[tuple[int, int, int, int]] = []
    rings: list[dict] = []
    for idx in range(ring_count):
        force_clip = want_clip and idx == 0
        ring = _sample_ring(grid_h, grid_w, reserved, ring_count, force_clip)
        if ring is None:
            return None
        ring["color"] = int(colors[idx])
        reserved.append(ring["footprint"])
        rings.append(ring)

    if want_clip:
        clipped = False
        for ring in rings:
            side = ring["forced_clip_side"]
            if side is None:
                continue
            status = _clip_status(
                ring["r0"], ring["c0"], ring["height"], ring["width"], side, grid_h, grid_w
            )
            clipped = clipped or status == "partial"
        if not clipped:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for ring in rings:
        _draw_ring(grid, ring["color"], ring["r0"], ring["c0"], ring["height"], ring["width"])
        for inner_r, inner_c in ring["markers"]:
            grid[int(ring["r0"] + inner_r), int(ring["c0"] + inner_c)] = MARKER

    # Validate the constructed puzzle instance against the solver assumptions.
    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
