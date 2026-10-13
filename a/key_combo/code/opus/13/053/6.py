from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

STRIP_HEIGHT = 3
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPENINGS = ("up", "right", "down", "left")
ROTATION_K = {"up": 0, "right": 3, "down": 2, "left": 1}
MIN_CUP_GAP = 3
GENERATION_RETRIES = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _opening_from_local_wall(local_wall: np.ndarray) -> str:
    h, w = local_wall.shape
    counts = {
        "up": int(np.count_nonzero(local_wall[0, :])),
        "down": int(np.count_nonzero(local_wall[-1, :])),
        "left": int(np.count_nonzero(local_wall[:, 0])),
        "right": int(np.count_nonzero(local_wall[:, -1])),
    }
    lengths = {"up": w, "down": w, "left": h, "right": h}
    opening = min(counts, key=counts.get)
    assert counts[opening] == 2
    assert sum(count == 2 for count in counts.values()) == 1
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}[opening]
    assert counts[opposite] == lengths[opposite]
    if opening in {"up", "down"}:
        assert counts["left"] == h and counts["right"] == h
        assert (h, w) == (4, 5)
    else:
        assert counts["up"] == w and counts["down"] == w
        assert (h, w) == (5, 4)
    assert not local_wall[1:-1, 1:-1].any()
    return opening


def _interior_bbox_from_wall_bbox(
    wall_bbox: tuple[int, int, int, int], opening: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = wall_bbox
    if opening == "up":
        return r0, c0 + 1, r1 - 1, c1 - 1
    if opening == "down":
        return r0 + 1, c0 + 1, r1, c1 - 1
    if opening == "left":
        return r0 + 1, c0, r1 - 1, c1 - 1
    return r0 + 1, c0 + 1, r1 - 1, c1


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    strip = grid[:STRIP_HEIGHT, :]
    comps = _component_coords(strip == BLUE)
    assert len(comps) == 1
    coords = comps[0]
    assert 3 <= len(coords) <= 5
    sprite = _crop_to_bbox(strip == BLUE)
    sprite = np.where(sprite, BLUE, BLACK)
    assert sprite.shape[0] <= 3 and sprite.shape[1] <= 3
    return sprite


def _detect_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for coords in _component_coords(grid == RED):
        wall_bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = wall_bbox
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        shifted = coords - np.array([r0, c0])
        local[shifted[:, 0], shifted[:, 1]] = True
        opening = _opening_from_local_wall(local)
        interior_bbox = _interior_bbox_from_wall_bbox(wall_bbox, opening)
        ir0, ic0, ir1, ic1 = interior_bbox
        assert (ir1 - ir0 + 1, ic1 - ic0 + 1) == (3, 3)
        assert np.all(grid[ir0 : ir1 + 1, ic0 : ic1 + 1] == BLACK)
        cups.append(
            {
                "wall_bbox": wall_bbox,
                "opening": opening,
                "interior_bbox": interior_bbox,
            }
        )
    cups.sort(key=lambda cup: (cup["wall_bbox"][0], cup["wall_bbox"][1]))
    return cups


def _rotate_sprite(sprite: np.ndarray, opening: str) -> np.ndarray:
    return np.rot90(sprite, ROTATION_K[opening]).copy()


def _stamp_sprite(
    out: np.ndarray, sprite: np.ndarray, interior_bbox: tuple[int, int, int, int]
) -> None:
    ir0, ic0, ir1, ic1 = interior_bbox
    assert (ir1 - ir0 + 1, ic1 - ic0 + 1) == (3, 3)
    h, w = sprite.shape
    assert h <= 3 and w <= 3
    target = out[ir0 : ir0 + h, ic0 : ic0 + w]
    target[sprite == BLUE] = BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the demonstrated blue sprite according to each red cup's opening direction, then stamp it into that cup interior.

    Input structure:
    - The top 3 rows form a grey demonstration strip containing one blue 4-connected sprite on grey background.
    - Below the strip is a black main region containing 2-4 separated red cup-shaped wall objects with one open side each.
    - Every cup has a one-pixel-thick wall and a 3x3 black interior region whose exposed side indicates the cup opening.

    Rule:
    - Crop the blue sprite from the demonstration strip as the base orientation, interpreted as the sprite for an upward-opening cup.
    - Detect each red cup and determine whether its opening faces up, right, down, or left.
    - Rotate the base sprite by 0, 90 clockwise, 180, or 270 clockwise respectively.
    - Stamp the rotated sprite into the cup's 3x3 interior, aligned to the interior's top-left corner, changing only those cells where the rotated sprite is blue.

    Key insight:
    - The cup geometry does not just mark a destination: its opening direction encodes the rotation angle that must be applied before the sprite is copied.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sprite = _extract_sprite(g)
    cups = _detect_cups(g)
    assert 2 <= len(cups) <= 4
    for cup in cups:
        _stamp_sprite(out, _rotate_sprite(sprite, cup["opening"]), cup["interior_bbox"])
    return out


def _sprite_candidates() -> list[np.ndarray]:
    candidates: dict[tuple[bytes, tuple[int, int]], np.ndarray] = {}
    for bits in range(1 << 9):
        if bits.bit_count() < 3 or bits.bit_count() > 5:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        if len(_component_coords(mask)) != 1:
            continue
        crop = _crop_to_bbox(mask)
        variants: list[np.ndarray] = []
        embedded: list[np.ndarray] = []
        ok = True
        for k in range(4):
            rot = _crop_to_bbox(np.rot90(crop, k))
            if any(np.array_equal(rot, old) for old in variants):
                ok = False
                break
            variants.append(rot)
            panel = np.zeros((3, 3), dtype=int)
            h, w = rot.shape
            panel[:h, :w] = rot.astype(int)
            embedded.append(panel)
        if not ok:
            continue
        if len({arr.tobytes() for arr in embedded}) != 4:
            continue
        sprite = np.where(crop, BLUE, BLACK)
        key = (sprite.tobytes(), sprite.shape)
        candidates[key] = sprite
    return list(candidates.values())


SPRITE_LIBRARY = _sprite_candidates()


def _cup_patch(opening: str) -> np.ndarray:
    if opening == "up":
        patch = np.zeros((4, 5), dtype=int)
        patch[:, 0] = RED
        patch[:, -1] = RED
        patch[-1, :] = RED
        return patch
    if opening == "down":
        patch = np.zeros((4, 5), dtype=int)
        patch[:, 0] = RED
        patch[:, -1] = RED
        patch[0, :] = RED
        return patch
    if opening == "left":
        patch = np.zeros((5, 4), dtype=int)
        patch[0, :] = RED
        patch[-1, :] = RED
        patch[:, -1] = RED
        return patch
    patch = np.zeros((5, 4), dtype=int)
    patch[0, :] = RED
    patch[-1, :] = RED
    patch[:, 0] = RED
    return patch


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (
        a[2] + MIN_CUP_GAP < b[0]
        or b[2] + MIN_CUP_GAP < a[0]
        or a[3] + MIN_CUP_GAP < b[1]
        or b[3] + MIN_CUP_GAP < a[1]
    )


def _choose_directions(rng: np.random.Generator, count: int) -> list[str]:
    vertical = list(rng.choice(["up", "down"], size=1, replace=False))
    horizontal = list(rng.choice(["left", "right"], size=1, replace=False))
    remaining = [d for d in OPENINGS if d not in vertical + horizontal]
    extra = []
    if count > 2:
        extra = list(rng.choice(remaining, size=count - 2, replace=False))
    directions = vertical + horizontal + extra
    rng.shuffle(directions)
    return directions


def _place_cups(
    rng: np.random.Generator, shape: tuple[int, int], directions: list[str]
) -> list[dict] | None:
    h, w = shape
    for _ in range(120):
        cups: list[dict] = []
        order = directions.copy()
        rng.shuffle(order)
        success = True
        for opening in order:
            patch = _cup_patch(opening)
            ph, pw = patch.shape
            positions = []
            for r0 in range(STRIP_HEIGHT, h - ph + 1):
                for c0 in range(0, w - pw + 1):
                    bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                    if all(_bbox_gap_ok(bbox, cup["wall_bbox"]) for cup in cups):
                        positions.append((r0, c0))
            if not positions:
                success = False
                break
            r0, c0 = positions[int(rng.integers(len(positions)))]
            wall_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            interior_bbox = _interior_bbox_from_wall_bbox(wall_bbox, opening)
            cups.append(
                {
                    "opening": opening,
                    "patch": patch,
                    "wall_bbox": wall_bbox,
                    "interior_bbox": interior_bbox,
                    "top_left": (r0, c0),
                }
            )
        if success:
            cups.sort(key=lambda cup: (cup["wall_bbox"][0], cup["wall_bbox"][1]))
            return cups
    return None


def _render_expected(
    grid: np.ndarray, sprite: np.ndarray, cups: list[dict]
) -> np.ndarray:
    out = grid.copy()
    for cup in cups:
        _stamp_sprite(out, _rotate_sprite(sprite, cup["opening"]), cup["interior_bbox"])
    return out


def _validate_sample(
    grid: np.ndarray, sprite: np.ndarray, cups: list[dict], expected: np.ndarray
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(cups) < 2 or len(cups) > 4:
        return False
    if not np.all(np.isin(grid[:STRIP_HEIGHT, :], [GREY, BLUE])):
        return False
    if not np.all(np.isin(grid[STRIP_HEIGHT:, :], [BLACK, RED])):
        return False
    extracted = _extract_sprite(grid)
    if not np.array_equal(extracted, sprite):
        return False
    for opening in OPENINGS:
        rot = _rotate_sprite(sprite, opening)
        if rot.shape[0] > 3 or rot.shape[1] > 3:
            return False
    embedded = []
    for opening in OPENINGS:
        panel = np.zeros((3, 3), dtype=int)
        rot = _rotate_sprite(sprite, opening)
        h, w = rot.shape
        panel[:h, :w] = rot
        embedded.append(panel)
    if len({arr.tobytes() for arr in embedded}) != 4:
        return False
    detected = _detect_cups(grid)
    if [(c["wall_bbox"], c["opening"]) for c in detected] != [
        (c["wall_bbox"], c["opening"]) for c in cups
    ]:
        return False
    orientations = {cup["opening"] for cup in cups}
    if not (orientations & {"up", "down"}) or not (orientations & {"left", "right"}):
        return False
    for i in range(len(cups)):
        for j in range(i + 1, len(cups)):
            if not _bbox_gap_ok(cups[i]["wall_bbox"], cups[j]["wall_bbox"]):
                return False
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(actual[:STRIP_HEIGHT, :], grid[:STRIP_HEIGHT, :]):
        return False
    if not np.array_equal(actual[grid == RED], grid[grid == RED]):
        return False
    lower_blue = np.argwhere(actual[STRIP_HEIGHT:, :] == BLUE)
    expected_blue: set[tuple[int, int]] = set()
    for cup in cups:
        rot = _rotate_sprite(sprite, cup["opening"])
        ir0, ic0, _ir1, _ic1 = cup["interior_bbox"]
        for rr, cc in np.argwhere(rot == BLUE):
            expected_blue.add((ir0 + int(rr), ic0 + int(cc)))
    actual_lower_blue = {
        (int(r) + STRIP_HEIGHT, int(c))
        for r, c in np.argwhere(actual[STRIP_HEIGHT:, :] == BLUE)
    }
    if actual_lower_blue != expected_blue:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sprite = SPRITE_LIBRARY[int(rng.integers(len(SPRITE_LIBRARY)))]
    cup_count = int(rng.choice([2, 3, 4], p=[0.3, 0.4, 0.3]))
    directions = _choose_directions(rng, cup_count)

    min_side = {2: 12, 3: 14, 4: 16}[cup_count]
    height = int(rng.integers(min_side, min_side + 5))
    width = int(rng.integers(min_side, min_side + 5))
    cups = _place_cups(rng, (height, width), directions)
    if cups is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:STRIP_HEIGHT, :] = GREY

    sh, sw = sprite.shape
    sprite_r = int(rng.integers(0, STRIP_HEIGHT - sh + 1))
    sprite_c = int(rng.integers(0, width - sw + 1))
    sprite_mask = sprite == BLUE
    grid[sprite_r : sprite_r + sh, sprite_c : sprite_c + sw][sprite_mask] = BLUE

    for cup in cups:
        r0, c0 = cup["top_left"]
        patch = cup["patch"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw][patch == RED] = RED

    expected = _render_expected(grid, sprite, cups)
    if not _validate_sample(grid, sprite, cups, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-rotation ARC input")
