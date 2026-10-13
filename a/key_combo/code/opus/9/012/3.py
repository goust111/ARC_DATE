from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    for parent in here.parents:
        utils_dir = parent / "new" / "solvers"
        if (utils_dir / "arc_utils.py").exists():
            utils_dir_str = str(utils_dir)
            if utils_dir_str not in sys.path:
                sys.path.append(utils_dir_str)
            from arc_utils import Color, bbox, paste

            return Color, bbox, paste
    raise ImportError("could not locate new/solvers/arc_utils.py")


Color, bbox, paste = _load_arc_utils()

SIDE_ORDER = ("top", "bottom", "left", "right")

GLYPH_BY_DISTANCE = {
    1: np.full((3, 3), Color.GREY, dtype=int),
    2: np.array(
        [
            [Color.GREY, Color.GREY, Color.GREY],
            [Color.GREY, Color.BLACK, Color.GREY],
            [Color.GREY, Color.GREY, Color.GREY],
        ],
        dtype=int,
    ),
    3: np.array(
        [
            [Color.GREY, Color.GREY, Color.GREY],
            [Color.BLACK, Color.BLACK, Color.BLACK],
            [Color.GREY, Color.GREY, Color.GREY],
        ],
        dtype=int,
    ),
    4: np.array(
        [
            [Color.BLACK, Color.GREY, Color.BLACK],
            [Color.GREY, Color.GREY, Color.GREY],
            [Color.BLACK, Color.GREY, Color.BLACK],
        ],
        dtype=int,
    ),
}
DISTANCE_BY_MASK = {
    tuple((patch == Color.BLACK).astype(int).ravel()): distance
    for distance, patch in GLYPH_BY_DISTANCE.items()
}

GENERATE_RETRIES = 400


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    red_mask = grid == Color.RED
    yellow_mask = grid == Color.YELLOW

    rect_bbox = bbox(red_mask)
    assert rect_bbox is not None, "expected one red frame"
    r0, c0, r1, c1 = rect_bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 5 and width >= 5, "rectangle interior must be at least 3x3"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(height, width)
    interior = ~border

    assert np.all(patch[border] == Color.RED), "frame border must be red"
    assert np.all(patch[interior] == Color.YELLOW), "frame interior must be yellow"
    assert int(np.count_nonzero(red_mask)) == int(border.sum()), "unexpected extra red pixels"
    assert int(np.count_nonzero(yellow_mask)) == int(interior.sum()), "unexpected extra yellow pixels"
    return rect_bbox


def _decode_glyph(patch: np.ndarray) -> int | None:
    if patch.shape != (3, 3):
        return None
    colors = set(map(int, np.unique(patch)))
    if not colors <= {Color.BLACK, Color.GREY}:
        return None
    key = tuple((patch == Color.BLACK).astype(int).ravel())
    return DISTANCE_BY_MASK.get(key)


def _find_glyphs(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    glyphs = []
    reserved = np.zeros_like(grid, dtype=bool)

    for r0 in range(height - 2):
        for c0 in range(width - 2):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            distance = _decode_glyph(patch)
            if distance is None:
                continue

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + 4)
            cc1 = min(width, c0 + 4)
            halo = (grid[rr0:rr1, cc0:cc1] == Color.GREY).copy()
            halo[r0 - rr0 : r0 - rr0 + 3, c0 - cc0 : c0 - cc0 + 3] = False
            if np.any(halo):
                continue

            box_mask = np.zeros_like(grid, dtype=bool)
            box_mask[r0 : r0 + 3, c0 : c0 + 3] = True
            assert not np.any(reserved & box_mask), "glyph boxes must not overlap"
            reserved |= box_mask
            glyphs.append(
                {
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "center": (r0 + 1, c0 + 1),
                    "distance": int(distance),
                    "grey_count": int(np.count_nonzero(patch == Color.GREY)),
                }
            )

    assert 2 <= len(glyphs) <= 4, "expected 2 to 4 glyph blocks"
    total_grey = sum(glyph["grey_count"] for glyph in glyphs)
    assert total_grey == int(np.count_nonzero(grid == Color.GREY)), "all grey cells must belong to glyph blocks"
    return glyphs


def _glyph_side(center: tuple[int, int], rect_bbox: tuple[int, int, int, int]) -> str:
    r, c = center
    r0, c0, r1, c1 = rect_bbox
    sides = []
    if r < r0:
        sides.append("top")
    if r > r1:
        sides.append("bottom")
    if c < c0:
        sides.append("left")
    if c > c1:
        sides.append("right")
    assert len(sides) == 1, "glyph must lie on exactly one side of the rectangle"
    return sides[0]


def _paint_rectangle(shape: tuple[int, int], rect_bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    r0, c0, r1, c1 = rect_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = Color.YELLOW
    out[r0, c0 : c1 + 1] = Color.RED
    out[r1, c0 : c1 + 1] = Color.RED
    out[r0 : r1 + 1, c0] = Color.RED
    out[r0 : r1 + 1, c1] = Color.RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the grey 3x3 glyphs around a red-yellow frame and stretch the indicated sides outward.

    Input structure:
    - The grid has a black background, one red rectangular border with a yellow interior, and 2 to 4 grey/black 3x3 glyph blocks outside that rectangle.
    - Each glyph block is exactly 3x3 and uses grey as its carrier color, with black pixels forming one of four predefined masks.
    - At most one glyph appears on each of the top, bottom, left, and right sides of the rectangle.

    Rule:
    - Find the unique red border / yellow interior rectangle and record its four sides.
    - Decode each 3x3 glyph mask as a stretch distance: all grey -> 1, center black -> 2, middle row black -> 3, four black corners -> 4.
    - Use the glyph's position relative to the rectangle to decide which side it controls, then move only that side outward by the decoded distance.
    - Erase the glyphs and redraw the stretched rectangle with a red border, yellow fill, and black background elsewhere.

    Key insight:
    - Each glyph carries two independent pieces of information at once: its internal black mask gives the numeric stretch amount, while its location around the rectangle chooses the side to move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rect_bbox = _find_rectangle(g)
    glyphs = _find_glyphs(g)

    side_to_distance: dict[str, int] = {}
    for glyph in glyphs:
        side = _glyph_side(glyph["center"], rect_bbox)
        assert side not in side_to_distance, "each side may have at most one glyph"
        side_to_distance[side] = glyph["distance"]

    r0, c0, r1, c1 = rect_bbox
    new_bbox = (
        r0 - side_to_distance.get("top", 0),
        c0 - side_to_distance.get("left", 0),
        r1 + side_to_distance.get("bottom", 0),
        c1 + side_to_distance.get("right", 0),
    )

    nr0, nc0, nr1, nc1 = new_bbox
    assert 0 <= nr0 <= nr1 < g.shape[0]
    assert 0 <= nc0 <= nc1 < g.shape[1]
    return _paint_rectangle(g.shape, new_bbox)


def _sample_side_distances(active_sides: list[str]) -> dict[str, int]:
    count = len(active_sides)
    if count == 4 and random.random() < 0.7:
        values = random.sample([1, 2, 3, 4], k=4)
    else:
        while True:
            values = [random.randint(1, 4) for _ in active_sides]
            if len(set(values)) >= 2 and any(value >= 3 for value in values):
                break
    if not any(value >= 3 for value in values):
        values[random.randrange(count)] = random.choice([3, 4])
    if len(set(values)) < 2:
        idx = random.randrange(count)
        values[idx] = 3 if values[idx] != 3 else 4
    return dict(zip(active_sides, values))


def _glyph_needs(active_sides: list[str], distances: dict[str, int], gaps: dict[str, int]) -> dict[str, int]:
    needs = {side: 0 for side in SIDE_ORDER}
    for side in active_sides:
        needs[side] = max(distances[side], gaps[side] + 3)
    return needs


def _axis_mid(lo: int, hi: int) -> int:
    return (lo + hi) // 2


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _glyph_top_left(
    side: str,
    rect_bbox: tuple[int, int, int, int],
    gap: int,
    offset: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect_bbox
    if side == "top":
        center_r = r0 - gap - 2
        center_c = _clamp(_axis_mid(c0, c1) + offset, c0 + 1, c1 - 1)
    elif side == "bottom":
        center_r = r1 + gap + 2
        center_c = _clamp(_axis_mid(c0, c1) + offset, c0 + 1, c1 - 1)
    elif side == "left":
        center_r = _clamp(_axis_mid(r0, r1) + offset, r0 + 1, r1 - 1)
        center_c = c0 - gap - 2
    elif side == "right":
        center_r = _clamp(_axis_mid(r0, r1) + offset, r0 + 1, r1 - 1)
        center_c = c1 + gap + 2
    else:
        raise ValueError(f"unknown side {side}")
    return center_r - 1, center_c - 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expected_output_bbox(
    rect_bbox: tuple[int, int, int, int],
    distances: dict[str, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect_bbox
    return (
        r0 - distances.get("top", 0),
        c0 - distances.get("left", 0),
        r1 + distances.get("bottom", 0),
        c1 + distances.get("right", 0),
    )


def _build_case() -> tuple[np.ndarray, dict]:
    active_count = random.choice([2, 3, 4])
    active_sides = random.sample(list(SIDE_ORDER), k=active_count)
    distances = _sample_side_distances(active_sides)
    gaps = {side: random.randint(1, 2) for side in active_sides}
    offsets = {side: random.choice([-1, 0, 1]) for side in active_sides}

    rect_h = random.randint(5, 10)
    rect_w = random.randint(5, 10)
    needs = _glyph_needs(active_sides, distances, gaps)

    spare_h = random.randint(1, 6)
    spare_w = random.randint(1, 6)
    height = needs["top"] + rect_h + needs["bottom"] + spare_h
    width = needs["left"] + rect_w + needs["right"] + spare_w
    assert height <= 30 and width <= 30

    r0 = needs["top"] + random.randint(0, spare_h)
    c0 = needs["left"] + random.randint(0, spare_w)
    rect_bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

    grid = _paint_rectangle((height, width), rect_bbox)

    glyph_boxes = []
    glyph_specs = {}
    for side in active_sides:
        gr0, gc0 = _glyph_top_left(side, rect_bbox, gaps[side], offsets[side])
        glyph_box = (gr0, gc0, gr0 + 2, gc0 + 2)
        assert not _boxes_overlap(glyph_box, rect_bbox), "glyph must stay outside the original rectangle"
        for other_box in glyph_boxes:
            assert not _boxes_overlap(glyph_box, other_box), "glyph boxes must not overlap"
        glyph_boxes.append(glyph_box)
        glyph_specs[side] = {
            "box": glyph_box,
            "distance": distances[side],
            "gap": gaps[side],
            "offset": offsets[side],
        }
        paste(grid, GLYPH_BY_DISTANCE[distances[side]], gr0, gc0, transparent=Color.BLACK)

    return grid, {
        "rect_bbox": rect_bbox,
        "active_sides": set(active_sides),
        "distances": distances,
        "gaps": gaps,
        "offsets": offsets,
        "glyph_specs": glyph_specs,
        "output_bbox": _expected_output_bbox(rect_bbox, distances),
    }


def _validate_case(grid: np.ndarray, meta: dict) -> None:
    rect_bbox = _find_rectangle(grid)
    assert rect_bbox == meta["rect_bbox"]

    glyphs = _find_glyphs(grid)
    seen_sides = {}
    for glyph in glyphs:
        side = _glyph_side(glyph["center"], rect_bbox)
        seen_sides[side] = glyph["distance"]
    assert set(seen_sides) == meta["active_sides"]
    assert seen_sides == meta["distances"]

    assert len(set(meta["distances"].values())) >= 2, "must not collapse to one common stretch value"
    assert any(value >= 3 for value in meta["distances"].values()), "at least one side must require real decoding"

    out = main(grid)
    expected = _paint_rectangle(grid.shape, meta["output_bbox"])
    assert np.array_equal(out, expected)

    original = meta["rect_bbox"]
    final = meta["output_bbox"]
    moved = {
        "top": final[0] != original[0],
        "left": final[1] != original[1],
        "bottom": final[2] != original[2],
        "right": final[3] != original[3],
    }
    assert {side for side, flag in moved.items() if flag} == meta["active_sides"]


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid, meta = _build_case()
        try:
            _validate_case(grid, meta)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid glyph-stretch ARC sample")
