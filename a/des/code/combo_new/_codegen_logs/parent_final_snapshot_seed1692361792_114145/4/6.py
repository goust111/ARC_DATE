from __future__ import annotations

from collections import Counter
import numpy as np


BG_COLOR = 0
FRAME_COLOR = 9
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_SIDE_RANGE = (22, 31)
PAIR_COUNT_RANGE = (2, 5)
INTERIOR_SIZE_CHOICES = (5, 7)
PLACEMENT_GAP = 1
PLACEMENT_TRIES = 250
MAX_GENERATION_TRIES = 400

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

ROTATION_K = {
    "tl": 0,
    "tr": 3,
    "br": 2,
    "bl": 1,
}

RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_connected(mask: np.ndarray, connectivity: int) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    seen = {tuple(map(int, pts[0]))}
    stack = [tuple(map(int, pts[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
                continue
            if (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _components_of_color(grid: np.ndarray, color: int, connectivity: int) -> list[dict]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    h, w = grid.shape
    comps: list[dict] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        coords = [(r, c)]
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            for dr, dc in neighbors:
                rr, cc2 = cr + dr, cc + dc
                if not (0 <= rr < h and 0 <= cc2 < w and mask[rr, cc2]):
                    continue
                if seen[rr, cc2]:
                    continue
                seen[rr, cc2] = True
                stack.append((rr, cc2))
                coords.append((rr, cc2))

        arr = np.array(coords, dtype=int)
        r0 = int(arr[:, 0].min())
        c0 = int(arr[:, 1].min())
        r1 = int(arr[:, 0].max())
        c1 = int(arr[:, 1].max())
        patch_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch_mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
        comps.append(
            {
                "color": int(color),
                "coords": arr,
                "bbox": (r0, c0, r1, c1),
                "mask": patch_mask,
                "area": int(arr.shape[0]),
            }
        )
    return comps


def _classify_frame(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 5 or w < 5:
        return None
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    if np.any(mask & ~boundary):
        return None
    missing = np.argwhere(boundary & ~mask)
    if len(missing) != 1:
        return None
    mr, mc = map(int, missing[0])
    if int(mask.sum()) != int(boundary.sum() - 1):
        return None
    if (mr, mc) == (0, 0):
        return "tl"
    if (mr, mc) == (0, w - 1):
        return "tr"
    if (mr, mc) == (h - 1, w - 1):
        return "br"
    if (mr, mc) == (h - 1, 0):
        return "bl"
    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for comp in _components_of_color(grid, color, connectivity=4):
            missing_corner = _classify_frame(comp["mask"])
            if missing_corner is None:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            center_r = (r0 + r1) // 2
            center_c = (c0 + c1) // 2
            marker_color = int(grid[center_r, center_c])
            if marker_color in (int(bg), color):
                continue
            frames.append(
                {
                    "bbox": comp["bbox"],
                    "color": color,
                    "missing_corner": missing_corner,
                    "marker_color": marker_color,
                }
            )
    assert frames, "no canvas frames detected"
    return frames


def _find_source_stamps(grid: np.ndarray, colors: set[int]) -> dict[int, np.ndarray]:
    stamps: dict[int, np.ndarray] = {}
    for color in colors:
        comps = sorted(
            _components_of_color(grid, int(color), connectivity=8),
            key=lambda comp: int(comp["area"]),
            reverse=True,
        )
        source = next((comp for comp in comps if int(comp["area"]) > 1), None)
        assert source is not None, f"missing source stamp for color {color}"
        stamps[int(color)] = source["mask"]
    return stamps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored source stamp to its marked frame and print the stamp there with the corner-indicated rotation.

    Input structure:
    - The background is a dominant black field.
    - Several small monochrome source stamps are scattered outside the frames; each is an 8-connected asymmetric shape.
    - Several hollow maroon rectangular frames are also present; each frame has exactly one missing corner pixel and one colored marker pixel at the center of its black interior.

    Rule:
    - Pair every frame with the source stamp of the same color as its center marker pixel.
    - Read the missing frame corner as a rotation code: top-left -> 0 degrees, top-right -> 90 degrees clockwise, bottom-right -> 180 degrees, bottom-left -> 90 degrees counterclockwise.
    - Rotate the matched stamp accordingly, align the stamp's bounding-box center to the frame interior center, and paint it inside the frame.
    - Keep the original source stamps and the frame borders unchanged; only add the rotated stamp inside each frame.

    Key insight:
    - Color performs the object-to-frame matching, while the single missing frame corner encodes which of the four rotations must be applied before centering the stamp inside its canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    marker_colors = {int(frame["marker_color"]) for frame in frames}
    stamps = _find_source_stamps(g, marker_colors)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        stamp = np.rot90(stamps[int(frame["marker_color"])], ROTATION_K[frame["missing_corner"]])
        sh, sw = stamp.shape
        top = center_r - sh // 2
        left = center_c - sw // 2
        assert r0 + 1 <= top and c0 + 1 <= left
        assert top + sh - 1 <= r1 - 1 and left + sw - 1 <= c1 - 1
        patch = out[top : top + sh, left : left + sw]
        patch[stamp] = int(frame["marker_color"])
    return out


def _build_stamp_templates() -> list[np.ndarray]:
    templates: list[np.ndarray] = []
    for bits in range(1 << 9):
        area = int(bits.bit_count())
        if area < 3 or area > 6:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if (bits >> idx) & 1:
                mask[idx // 3, idx % 3] = True
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not _is_connected(mask, connectivity=8):
            continue
        rotations = {np.rot90(mask, k).tobytes() for k in range(4)}
        if len(rotations) != 4:
            continue
        templates.append(mask)
    templates.sort(key=lambda mask: (int(mask.sum()), mask.tobytes()))
    return templates


STAMP_TEMPLATES = _build_stamp_templates()
STAMP_TEMPLATES_CENTER_ON = [mask for mask in STAMP_TEMPLATES if bool(mask[1, 1])]
STAMP_TEMPLATES_CENTER_OFF = [mask for mask in STAMP_TEMPLATES if not bool(mask[1, 1])]


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_far = ar1 + gap < br0 or br1 + gap < ar0
    col_far = ac1 + gap < bc0 or bc1 + gap < ac0
    return not (row_far or col_far)


def _find_free_box(
    h: int,
    w: int,
    box_h: int,
    box_w: int,
    occupied: list[tuple[int, int, int, int]],
    gap: int,
) -> tuple[int, int, int, int] | None:
    for _ in range(PLACEMENT_TRIES):
        r0 = int(RNG.integers(0, h - box_h + 1))
        c0 = int(RNG.integers(0, w - box_w + 1))
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if any(_boxes_too_close(box, other, gap) for other in occupied):
            continue
        return box
    return None


def _balanced_corners(n: int) -> list[str]:
    corners = ["tl", "tr", "br", "bl"]
    out: list[str] = []
    while len(out) < n:
        batch = corners.copy()
        RNG.shuffle(batch)
        out.extend(batch)
    return out[:n]


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], missing_corner: str, marker_color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    if missing_corner == "tl":
        grid[r0, c0] = BG_COLOR
    elif missing_corner == "tr":
        grid[r0, c1] = BG_COLOR
    elif missing_corner == "br":
        grid[r1, c1] = BG_COLOR
    else:
        grid[r1, c0] = BG_COLOR
    grid[(r0 + r1) // 2, (c0 + c1) // 2] = int(marker_color)


def _draw_stamp(grid: np.ndarray, top: int, left: int, color: int, mask: np.ndarray) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = int(color)


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        stamp = np.rot90(spec["stamp_mask"], ROTATION_K[spec["missing_corner"]])
        r0, c0, r1, c1 = spec["frame_box"]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        top = center_r - stamp.shape[0] // 2
        left = center_c - stamp.shape[1] // 2
        _draw_stamp(out, top, left, int(spec["color"]), stamp)
    return out


def _choose_stamp_masks(n_pairs: int) -> list[np.ndarray]:
    assert STAMP_TEMPLATES_CENTER_ON and STAMP_TEMPLATES_CENTER_OFF
    used: set[bytes] = set()
    chosen: list[np.ndarray] = []

    pools = [STAMP_TEMPLATES_CENTER_ON, STAMP_TEMPLATES_CENTER_OFF]
    RNG.shuffle(pools)
    for pool in pools:
        mask = pool[int(RNG.integers(len(pool)))]
        chosen.append(mask)
        used.add(mask.tobytes())

    while len(chosen) < n_pairs:
        mask = STAMP_TEMPLATES[int(RNG.integers(len(STAMP_TEMPLATES)))]
        key = mask.tobytes()
        if key in used and len(used) < len(STAMP_TEMPLATES):
            continue
        chosen.append(mask)
        used.add(key)
    RNG.shuffle(chosen)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(*GRID_SIDE_RANGE))
    w = int(RNG.integers(*GRID_SIDE_RANGE))
    n_pairs = int(RNG.integers(*PAIR_COUNT_RANGE))
    if n_pairs > len(SOURCE_COLORS):
        return None

    colors = [int(x) for x in RNG.choice(SOURCE_COLORS, size=n_pairs, replace=False)]
    corners = _balanced_corners(n_pairs)
    stamps = _choose_stamp_masks(n_pairs)

    specs = []
    for idx in range(n_pairs):
        inner_h = int(RNG.choice(INTERIOR_SIZE_CHOICES))
        inner_w = int(RNG.choice(INTERIOR_SIZE_CHOICES))
        specs.append(
            {
                "color": colors[idx],
                "missing_corner": corners[idx],
                "stamp_mask": stamps[idx],
                "frame_h": inner_h + 2,
                "frame_w": inner_w + 2,
            }
        )

    occupied: list[tuple[int, int, int, int]] = []
    for spec in sorted(specs, key=lambda item: item["frame_h"] * item["frame_w"], reverse=True):
        frame_box = _find_free_box(h, w, spec["frame_h"], spec["frame_w"], occupied, PLACEMENT_GAP)
        if frame_box is None:
            return None
        spec["frame_box"] = frame_box
        occupied.append(frame_box)

    order = np.arange(n_pairs)
    RNG.shuffle(order)
    for idx in order:
        spec = specs[int(idx)]
        sh, sw = spec["stamp_mask"].shape
        source_box = _find_free_box(h, w, sh, sw, occupied, PLACEMENT_GAP)
        if source_box is None:
            return None
        spec["source_box"] = source_box
        occupied.append(source_box)

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        _draw_frame(grid, spec["frame_box"], spec["missing_corner"], spec["color"])
    for spec in specs:
        sr0, sc0, _sr1, _sc1 = spec["source_box"]
        _draw_stamp(grid, sr0, sc0, spec["color"], spec["stamp_mask"])

    expected = _render_expected(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC stamp-frame rotation task")
