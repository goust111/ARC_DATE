import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = (BLUE, RED, GREEN)
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def _box_height(box: tuple[int, int, int, int]) -> int:
    return box[2] - box[0] + 1


def _box_width(box: tuple[int, int, int, int]) -> int:
    return box[3] - box[1] + 1


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _in_box(rc: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = rc
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _template_info(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    yellow = np.argwhere(grid == YELLOW)
    if yellow.shape[0] != 3:
        raise AssertionError("expected exactly three yellow template pixels")
    box = bbox_coords(yellow)
    if _box_height(box) != 2 or _box_width(box) != 2:
        raise AssertionError("template bbox must be 2x2")
    template = crop_bbox(grid, box)
    if np.count_nonzero(template == YELLOW) != 3 or np.count_nonzero(template == BLACK) != 1:
        raise AssertionError("template must be a 3-of-4 yellow 2x2 pattern")
    return template, box


def _marker_groups(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    groups = []
    present = sorted(int(v) for v in np.unique(grid) if int(v) in MARKER_COLORS)
    for color in present:
        coords = np.argwhere(grid == color)
        if coords.shape[0] != 3:
            raise AssertionError(f"color {color} must appear exactly three times")
        groups.append((color, coords))
    if not (2 <= len(groups) <= 3):
        raise AssertionError("expected 2 or 3 marker-color groups")
    return groups


def _trim_candidates(box: tuple[int, int, int, int]) -> list[tuple[str, tuple[int, int, int, int]]]:
    r0, c0, r1, c1 = box
    out = []
    if r0 < r1:
        out.append(("top", (r0 + 1, c0, r1, c1)))
        out.append(("bottom", (r0, c0, r1 - 1, c1)))
    if c0 < c1:
        out.append(("left", (r0, c0 + 1, r1, c1)))
        out.append(("right", (r0, c0, r1, c1 - 1)))
    return out


def _valid_locators(
    grid: np.ndarray, color: int, coords: np.ndarray
) -> list[tuple[str, tuple[int, int, int, int]]]:
    box = bbox_coords(coords)
    valid = []
    for side, trimmed in _trim_candidates(box):
        sub = crop_bbox(grid, trimmed)
        vals = {int(v) for v in np.unique(sub)}
        if vals <= {GREY, color} and int(np.count_nonzero(sub == color)) == 1:
            valid.append((side, trimmed))
    return valid


def _scale_from_rect(
    rect: tuple[int, int, int, int], template: np.ndarray
) -> int:
    rect_h = _box_height(rect)
    rect_w = _box_width(rect)
    frame_h, frame_w = map(int, template.shape)
    frame_area = frame_h * frame_w
    rect_area = rect_h * rect_w
    if rect_area % frame_area != 0:
        raise AssertionError("target rectangle area must be a whole number of template frames")
    ratio = rect_area // frame_area
    k = math.isqrt(ratio)
    if k * k != ratio:
        raise AssertionError("target rectangle area must imply an integer scale")
    if rect_h != frame_h * k or rect_w != frame_w * k:
        raise AssertionError("target rectangle dimensions must match scaled template frame")
    return k


def _solve_details(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    template, template_box = _template_info(g)
    groups = _marker_groups(g)

    located = []
    for color, coords in groups:
        valid = _valid_locators(g, color, coords)
        if len(valid) != 1:
            raise AssertionError(f"marker group {color} must have exactly one valid trim")
        side, rect = valid[0]
        k = _scale_from_rect(rect, template)
        located.append(
            {
                "color": color,
                "coords": coords,
                "side": side,
                "rect": rect,
                "k": k,
            }
        )

    out = g.copy()
    for item in located:
        for r, c in item["coords"]:
            out[int(r), int(c)] = BLACK

    for item in sorted(located, key=lambda rec: rec["color"]):
        r0, c0, r1, c1 = item["rect"]
        k = item["k"]
        scaled = scale_sprite(template, k)
        recolored = scaled.copy()
        recolored[recolored == YELLOW] = item["color"]

        window = out[r0 : r1 + 1, c0 : c1 + 1]
        window[:] = GREY
        mask = recolored != BLACK
        window[mask] = recolored[mask]

    return out, template, template_box, located


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use three same-color markers to locate grey sub-rectangles, then stamp a scaled copy of a 2x2 template into each one.

    Input structure:
    - A black background contains one large solid grey rectangle and one 2x2 yellow template with exactly one black corner.
    - Two or three marker colors from blue/red/green each appear exactly three times.
    - For each color, the three marker pixels define a bbox whose correct one-side trim exposes a single grey target rectangle inside the grey mask.

    Rule:
    - Group the markers by color and compute each color group's axis-aligned bbox.
    - Try trimming away the top, bottom, left, and right border of that bbox; keep the unique trimmed rectangle whose contents are only grey plus that marker color and that contains exactly one marker pixel.
    - Erase all marker pixels, treat the 2x2 template frame area as the scale reference, and convert each target rectangle area into an integer scale factor.
    - Scale the template by that factor, recolor yellow to the marker color, and stamp it at the target rectangle's top-left while black template cells leave the grey background visible.

    Key insight:
    - The markers do not directly fill the rectangle; they first reveal which side of the bbox must be trimmed, and the resulting rectangle's geometry then determines the scale of the stamped template.
    """
    out, _template, _template_box, _located = _solve_details(grid)
    return out


def _build_target(
    mask_box: tuple[int, int, int, int], side: str, start: int, size: int, survivor_end: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[tuple[int, int]]]:
    mr0, mc0, mr1, mc1 = mask_box
    if side == "top":
        rect = (mr0, start, mr0 + size - 1, start + size - 1)
        outer = (mr0 - 1, start, mr0 + size - 1, start + size - 1)
        survivor = (mr0 + size - 1, start if survivor_end == 0 else start + size - 1)
        markers = [(mr0 - 1, start), (mr0 - 1, start + size - 1), survivor]
    elif side == "bottom":
        rect = (mr1 - size + 1, start, mr1, start + size - 1)
        outer = (mr1 - size + 1, start, mr1 + 1, start + size - 1)
        survivor = (mr1 - size + 1, start if survivor_end == 0 else start + size - 1)
        markers = [(mr1 + 1, start), (mr1 + 1, start + size - 1), survivor]
    elif side == "left":
        rect = (start, mc0, start + size - 1, mc0 + size - 1)
        outer = (start, mc0 - 1, start + size - 1, mc0 + size - 1)
        survivor = (start if survivor_end == 0 else start + size - 1, mc0 + size - 1)
        markers = [(start, mc0 - 1), (start + size - 1, mc0 - 1), survivor]
    elif side == "right":
        rect = (start, mc1 - size + 1, start + size - 1, mc1)
        outer = (start, mc1 - size + 1, start + size - 1, mc1 + 1)
        survivor = (start if survivor_end == 0 else start + size - 1, mc1 - size + 1)
        markers = [(start, mc1 + 1), (start + size - 1, mc1 + 1), survivor]
    else:
        raise ValueError(f"unknown side: {side}")
    return rect, outer, markers


def _place_template(
    grid: np.ndarray, mask_box: tuple[int, int, int, int], template: np.ndarray
) -> tuple[int, int, int, int]:
    mr0, mc0, _mr1, _mc1 = mask_box
    max_r = max(0, mr0 - 3)
    max_c = max(0, mc0 - 3)
    r0 = random.randint(0, min(2, max_r))
    c0 = random.randint(0, min(2, max_c))
    grid[r0 : r0 + 2, c0 : c0 + 2] = template
    return r0, c0, r0 + 1, c0 + 1


def _side_choices(n_groups: int) -> list[str]:
    if n_groups == 2 and random.random() < 0.65:
        return [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    return random.sample(SIDES, n_groups)


def _expected_block(template: np.ndarray, k: int, color: int) -> np.ndarray:
    scaled = scale_sprite(template, k)
    out = np.full(scaled.shape, GREY, dtype=int)
    out[scaled == YELLOW] = color
    return out


def _validate_generated(
    grid: np.ndarray,
    template_box: tuple[int, int, int, int],
    specs: list[dict],
) -> None:
    out = main(grid)
    template = crop_bbox(grid, template_box)
    if not np.array_equal(crop_bbox(out, template_box), template):
        raise AssertionError("template must remain unchanged in the output")

    for spec in specs:
        color = spec["color"]
        valid = _valid_locators(grid, color, np.argwhere(grid == color))
        if len(valid) != 1 or valid[0][1] != spec["rect"]:
            raise AssertionError("generated marker group is ambiguous")
        actual = crop_bbox(out, spec["rect"])
        expected = _expected_block(template, spec["k"], color)
        if not np.array_equal(actual, expected):
            raise AssertionError("stamped block does not match the intended scaled template")
        for marker in spec["markers"]:
            if not _in_box(marker, spec["rect"]) and int(out[marker]) != BLACK:
                raise AssertionError("markers outside target rectangles must be erased to black")

    rects = [spec["rect"] for spec in specs]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _boxes_overlap(rects[i], rects[j], pad=0):
                raise AssertionError("target rectangles must not overlap")


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 26)
    w = random.randint(20, 26)
    n_groups = random.choice((2, 3))

    min_mask = 14 if n_groups == 3 else 12
    mask_h = random.randint(min_mask, min(16, h - 6))
    mask_w = random.randint(min_mask, min(16, w - 6))

    base_r = (h - mask_h) // 2
    base_c = (w - mask_w) // 2
    r0_lo = 4
    c0_lo = 4
    r0_hi = h - mask_h - 3
    c0_hi = w - mask_w - 3
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    mask_r0 = min(max(base_r + random.randint(-1, 1), r0_lo), r0_hi)
    mask_c0 = min(max(base_c + random.randint(-1, 1), c0_lo), c0_hi)
    mask_box = (mask_r0, mask_c0, mask_r0 + mask_h - 1, mask_c0 + mask_w - 1)

    grid = np.full((h, w), BLACK, dtype=int)
    mr0, mc0, mr1, mc1 = mask_box
    grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = GREY

    template = np.full((2, 2), YELLOW, dtype=int)
    hole = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
    template[hole] = BLACK
    template_box = _place_template(grid, mask_box, template)

    colors = random.sample(MARKER_COLORS, n_groups)
    sides = _side_choices(n_groups)
    raw_specs = []
    for color, side in zip(colors, sides, strict=True):
        raw_specs.append(
            {
                "color": color,
                "side": side,
                "k": random.choice((2, 3)),
                "survivor_end": random.randint(0, 1),
            }
        )
    raw_specs.sort(key=lambda spec: (-spec["k"], spec["color"]))

    placed: list[dict] = []
    occupied: list[tuple[int, int, int, int]] = []

    for spec in raw_specs:
        size = 2 * spec["k"]
        if spec["side"] in ("top", "bottom"):
            starts = list(range(mc0, mc1 - size + 2))
        else:
            starts = list(range(mr0, mr1 - size + 2))
        random.shuffle(starts)

        chosen = None
        for start in starts:
            rect, outer, markers = _build_target(
                mask_box, spec["side"], start, size, spec["survivor_end"]
            )
            if any(_boxes_overlap(outer, taken, pad=1) for taken in occupied):
                continue
            chosen = {
                "color": spec["color"],
                "side": spec["side"],
                "k": spec["k"],
                "rect": rect,
                "outer": outer,
                "markers": markers,
            }
            break

        if chosen is None:
            return None

        placed.append(chosen)
        occupied.append(chosen["outer"])

    for spec in placed:
        for r, c in spec["markers"]:
            grid[r, c] = spec["color"]

    try:
        _validate_generated(grid, template_box, placed)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
