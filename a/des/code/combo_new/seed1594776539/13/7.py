import random

import numpy as np


BG = 0
MAROON = 9
STAMP_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
STAMP_SIZE_CHOICES = [(3, 5), (5, 3), (5, 5)]
CANVAS_INNER_CHOICES = [5, 7]
QUADRANT_TO_ROTATION = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 1): 2,
    (1, 0): 3,
}
ROTATION_TO_QUADRANT = {value: key for key, value in QUADRANT_TO_ROTATION.items()}
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _rotate_mask(mask: np.ndarray, steps_cw: int) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), -int(steps_cw))


def _connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _has_all_quadrants(mask: np.ndarray) -> bool:
    h, w = mask.shape
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    seen = [False, False, False, False]
    for r, c in np.argwhere(mask):
        if r == center_r or c == center_c:
            continue
        q = QUADRANT_TO_ROTATION[(0 if r < center_r else 1, 0 if c < center_c else 1)]
        seen[q] = True
    return all(seen)


def _rotations_unique(mask: np.ndarray) -> bool:
    variants = set()
    for k in range(4):
        rotated = _rotate_mask(mask, k)
        variants.add((rotated.shape, rotated.tobytes()))
    return len(variants) == 4


def _stamp_structure_ok(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    return area >= 5 and area < mask.size and _connected(mask) and _touches_all_sides(mask) and _has_all_quadrants(mask)


def _stamp_ok(mask: np.ndarray) -> bool:
    return _stamp_structure_ok(mask) and _rotations_unique(mask)


def _canonical_rotation_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for k in range(4):
        rotated = _rotate_mask(mask, k)
        variants.append((rotated.shape, rotated.tobytes()))
    return min(variants)


def _make_stamp_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h, w = STAMP_SIZE_CHOICES[rng.randrange(len(STAMP_SIZE_CHOICES))]
        mask = np.ones((h, w), dtype=bool)
        target_area = rng.randint(max(5, (h * w) // 2), h * w - 1)
        for _pass in range(8):
            cells = [(r, c) for r in range(h) for c in range(w)]
            rng.shuffle(cells)
            changed = False
            for r, c in cells:
                if int(mask.sum()) <= target_area:
                    break
                trial = mask.copy()
                trial[r, c] = False
                if _stamp_structure_ok(trial):
                    mask = trial
                    changed = True
            if int(mask.sum()) <= target_area or not changed:
                break
        if _stamp_ok(mask):
            return mask
    raise RuntimeError("failed to create a valid stamp mask")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _canvas_candidates(rotated_mask: np.ndarray, inner_h: int, inner_w: int, rotation: int) -> tuple[int, int, list[tuple[int, int]]]:
    sh, sw = rotated_mask.shape
    top = (inner_h - sh) // 2
    left = (inner_w - sw) // 2
    center_r = (inner_h - 1) / 2.0
    center_c = (inner_w - 1) / 2.0
    want = ROTATION_TO_QUADRANT[int(rotation)]
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(rotated_mask):
        ir = top + int(r)
        ic = left + int(c)
        if ir == center_r or ic == center_c:
            continue
        quadrant = (0 if ir < center_r else 1, 0 if ic < center_c else 1)
        if quadrant == want:
            candidates.append((ir, ic))
    return top, left, candidates


def _extract_canvases(grid: np.ndarray) -> list[dict]:
    canvases = []
    for coords in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(patch[0, :] == MAROON)
            and np.all(patch[-1, :] == MAROON)
            and np.all(patch[:, 0] == MAROON)
            and np.all(patch[:, -1] == MAROON)
        ):
            continue
        inner = patch[1:-1, 1:-1]
        if inner.size == 0 or np.any(inner == MAROON):
            continue
        anchors = np.argwhere((inner != BG) & (inner != MAROON))
        if anchors.shape[0] != 1:
            continue
        ar, ac = map(int, anchors[0])
        if ar * 2 == inner.shape[0] - 1 or ac * 2 == inner.shape[1] - 1:
            raise AssertionError("anchor lies on the canvas center line")
        quadrant = (
            0 if ar < (inner.shape[0] - 1) / 2.0 else 1,
            0 if ac < (inner.shape[1] - 1) / 2.0 else 1,
        )
        canvases.append(
            {
                "outer_bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "anchor": (r0 + 1 + ar, c0 + 1 + ac),
                "anchor_color": int(inner[ar, ac]),
                "rotation": QUADRANT_TO_ROTATION[quadrant],
            }
        )
    assert canvases, "expected at least one maroon canvas"
    return canvases


def _extract_stamps(grid: np.ndarray, canvases: list[dict]) -> dict[int, np.ndarray]:
    blocked = np.zeros(grid.shape, dtype=bool)
    for canvas in canvases:
        r0, c0, r1, c1 = canvas["outer_bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    stamp_mask = (grid != BG) & (grid != MAROON) & (~blocked)
    stamps: dict[int, np.ndarray] = {}
    for coords in _components(stamp_mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "stamp components must be monochrome"
        color = colors.pop()
        assert color not in stamps, "stamp colors must be unique"
        stamps[color] = _mask_from_coords(coords)
    assert stamps, "expected at least one stamp"
    return stamps


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    canvases = _extract_canvases(g)
    stamps = _extract_stamps(g, canvases)

    for canvas in canvases:
        color = int(canvas["anchor_color"])
        assert color in stamps, "every anchor color must match a stamp"
        mask = _rotate_mask(stamps[color], int(canvas["rotation"]))
        ir0, ic0, ir1, ic1 = canvas["inner_bbox"]
        inner_h = ir1 - ir0 + 1
        inner_w = ic1 - ic0 + 1
        sh, sw = mask.shape
        assert sh <= inner_h and sw <= inner_w, "stamp must fit the canvas interior"
        top = ir0 + (inner_h - sh) // 2
        left = ic0 + (inner_w - sw) // 2
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
        _paint_mask(out, mask, top, left, color)

    return out


def _build_scene(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    stamp_count = rng.randint(2, 3)
    canvas_count = 4
    colors = rng.sample(STAMP_COLOR_POOL, stamp_count)

    stamp_masks: dict[int, np.ndarray] = {}
    used_signatures = set()
    for color in colors:
        for _ in range(100):
            mask = _make_stamp_mask(rng)
            signature = _canonical_rotation_signature(mask)
            if signature not in used_signatures:
                used_signatures.add(signature)
                stamp_masks[color] = mask
                break
        else:
            raise RuntimeError("failed to create distinct stamp masks")

    stamp_patches = []
    for color in colors:
        patch = np.zeros(stamp_masks[color].shape, dtype=int)
        patch[stamp_masks[color]] = int(color)
        stamp_patches.append({"kind": "stamp", "color": color, "patch": patch})

    canvas_colors = list(colors)
    while len(canvas_colors) < canvas_count:
        canvas_colors.append(colors[rng.randrange(stamp_count)])
    rng.shuffle(canvas_colors)

    rotations = list(range(4))
    rng.shuffle(rotations)
    while len(rotations) < canvas_count:
        rotations.append(rng.randrange(4))

    canvas_objects = []
    for idx in range(canvas_count):
        color = canvas_colors[idx]
        rotation = rotations[idx]
        inner_h = CANVAS_INNER_CHOICES[rng.randrange(len(CANVAS_INNER_CHOICES))]
        inner_w = CANVAS_INNER_CHOICES[rng.randrange(len(CANVAS_INNER_CHOICES))]
        rotated = _rotate_mask(stamp_masks[color], rotation)
        top, left, candidates = _canvas_candidates(rotated, inner_h, inner_w, rotation)
        assert candidates, "each selected rotation must have a valid anchor location"
        anchor_r, anchor_c = candidates[rng.randrange(len(candidates))]

        patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
        patch[0, :] = MAROON
        patch[-1, :] = MAROON
        patch[:, 0] = MAROON
        patch[:, -1] = MAROON
        patch[1 + anchor_r, 1 + anchor_c] = int(color)

        expected_inner = np.zeros((inner_h, inner_w), dtype=int)
        _paint_mask(expected_inner, rotated, top, left, color)
        canvas_objects.append(
            {
                "kind": "canvas",
                "color": color,
                "rotation": rotation,
                "patch": patch,
                "expected_inner": expected_inner,
            }
        )

    objects = stamp_patches + canvas_objects
    total_area = sum(int(obj["patch"].shape[0] * obj["patch"].shape[1]) for obj in objects)
    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    base_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 1.7))))

    positions = None
    grid_shape = None
    order = sorted(range(len(objects)), key=lambda i: int(objects[i]["patch"].size), reverse=True)
    for _ in range(120):
        h = rng.randint(base_side, 30)
        w = rng.randint(base_side, 30)
        if h < max_h + 2 or w < max_w + 2:
            continue
        occupied = np.zeros((h, w), dtype=bool)
        positions = [None] * len(objects)
        ok = True
        for obj_idx in order:
            patch = objects[obj_idx]["patch"]
            ph, pw = patch.shape
            candidates = []
            for top in range(1, h - ph):
                for left in range(1, w - pw):
                    r0 = top - 1
                    c0 = left - 1
                    r1 = top + ph + 1
                    c1 = left + pw + 1
                    if not occupied[r0:r1, c0:c1].any():
                        candidates.append((top, left))
            if not candidates:
                ok = False
                break
            top, left = candidates[rng.randrange(len(candidates))]
            positions[obj_idx] = (top, left)
            occupied[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True
        if ok:
            grid_shape = (h, w)
            break
    if positions is None or grid_shape is None:
        raise RuntimeError("failed to place all scene objects")

    inp = np.zeros(grid_shape, dtype=int)
    expected = np.zeros(grid_shape, dtype=int)
    for obj, pos in zip(objects, positions):
        top, left = pos
        patch = obj["patch"]
        ph, pw = patch.shape
        view = inp[top : top + ph, left : left + pw]
        nz = patch != BG
        view[nz] = patch[nz]
        expected[top : top + ph, left : left + pw][nz] = patch[nz]

        if obj["kind"] == "canvas":
            expected_inner = obj["expected_inner"]
            eh, ew = expected_inner.shape
            expected[top + 1 : top + 1 + eh, left + 1 : left + 1 + ew] = expected_inner

    return inp, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the color-matched template into each maroon canvas using the anchor quadrant as a rotation code.

    Input structure:
    - The background is black.
    - Several compact monochrome stamp templates sit outside the canvases; each stamp uses a distinct non-maroon color.
    - Several larger hollow rectangles with maroon borders act as canvases, and each canvas interior contains exactly one colored anchor pixel on black.
    - The anchor color matches one stamp color, and the anchor position lies in one of the four interior quadrants of its canvas.

    Rule:
    - Find every maroon canvas and read its unique interior anchor.
    - Match the anchor color to the stamp of the same color.
    - Decode the anchor quadrant as a clockwise rotation: top-left = 0 degrees, top-right = 90 degrees, bottom-right = 180 degrees, bottom-left = 270 degrees.
    - Rotate the matched stamp accordingly, center it inside the canvas interior, and paint it there; the anchor is overwritten while the original stamps and maroon borders remain unchanged.

    Key insight:
    - The anchor carries two independent signals at once: its color chooses the stamp template, while its quadrant chooses the stamp orientation.
    """
    return _solve(np.asarray(grid, dtype=int))


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected = _build_scene(rng)
        except RuntimeError:
            continue
        if np.array_equal(_solve(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
