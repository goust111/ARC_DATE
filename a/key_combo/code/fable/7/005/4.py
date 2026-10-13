from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg, is_contiguous


BG = 0
MAX_SIDE = 30
RETRY_LIMIT = 600
MOVES = ("right", "left", "down", "up")
SHAPE_DIMS = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
PALETTE = tuple(range(1, 10))


def _touches_all_bbox_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _enumerate_shapes() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for h, w in SHAPE_DIMS:
        cells = [(r, c) for r in range(h) for c in range(w)]
        for area in range(4, min(6, h * w) + 1):
            for combo in itertools.combinations(cells, area):
                mask = np.zeros((h, w), dtype=int)
                for r, c in combo:
                    mask[r, c] = 1
                if not is_contiguous(mask, background=0, connectivity=8):
                    continue
                if not _touches_all_bbox_edges(mask):
                    continue
                if h == 2 and w == 2 and int(mask.sum()) == 4:
                    continue
                if np.array_equal(mask, np.fliplr(mask)):
                    continue
                if np.array_equal(mask, np.flipud(mask)):
                    continue
                coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))
                if coords in seen:
                    continue
                seen.add(coords)
                shapes.append(mask)
    return shapes


SHAPES = _enumerate_shapes()


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return tuple(map(int, bbox_coords(np.asarray(coords, dtype=int))))


def _anchor_coords(box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.array([(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)], dtype=int)


def _shift(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    out[:, 0] += int(dr)
    out[:, 1] += int(dc)
    return out


def _coords_key(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.asarray(coords, dtype=int)}


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _is_solid_2x2(comp: dict, grid: np.ndarray) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        int(comp["area"]) == 4
        and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
        and bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])).all())
    )


def _alignment_for_anchor(
    sprite_coords: np.ndarray, anchor_box: tuple[int, int, int, int]
) -> tuple[str, int, int] | None:
    sr0, sc0, sr1, sc1 = _bbox(sprite_coords)
    ar0, ac0, ar1, ac1 = map(int, anchor_box)

    candidates: list[tuple[str, int, int]] = []
    if sc1 < ac0 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append(("right", 0, ac0 - sc1 - 1))
    if sc0 > ac1 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append(("left", 0, ac1 + 1 - sc0))
    if sr1 < ar0 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append(("down", ar0 - sr1 - 1, 0))
    if sr0 > ar1 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append(("up", ar1 + 1 - sr0, 0))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _slide_and_mirror(
    sprite_coords: np.ndarray,
    anchor_box: tuple[int, int, int, int],
    move: str,
    dr: int,
    dc: int,
) -> np.ndarray:
    moved = _shift(sprite_coords, dr, dc)
    ar0, ac0, ar1, ac1 = map(int, anchor_box)
    mirrored = moved.copy()
    if move == "right":
        mirrored[:, 1] = 2 * ac0 - 1 - moved[:, 1]
    elif move == "left":
        mirrored[:, 1] = 2 * ac1 + 1 - moved[:, 1]
    elif move == "down":
        mirrored[:, 0] = 2 * ar0 - 1 - moved[:, 0]
    else:
        mirrored[:, 0] = 2 * ar1 + 1 - moved[:, 0]
    return mirrored


def _trace_relay(
    sprite_coords: np.ndarray, anchors: list[dict]
) -> list[dict]:
    current = np.asarray(sprite_coords, dtype=int)
    remaining = list(range(len(anchors)))
    trace: list[dict] = []
    for _ in range(len(anchors)):
        candidates = []
        for idx in remaining:
            align = _alignment_for_anchor(current, anchors[idx]["bbox"])
            if align is not None:
                candidates.append((idx, align))
        assert len(candidates) == 1, "each relay state must align with exactly one unvisited anchor"
        idx, (move, dr, dc) = candidates[0]
        product = _slide_and_mirror(current, anchors[idx]["bbox"], move, dr, dc)
        trace.append(
            {
                "anchor": anchors[idx],
                "move": move,
                "dr": int(dr),
                "dc": int(dc),
                "product": product,
            }
        )
        remaining.remove(idx)
        current = product
    return trace


def _paint_coords(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    for r, c in np.asarray(coords, dtype=int):
        rr = int(r)
        cc = int(c)
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
        assert int(out[rr, cc]) == BG
        out[rr, cc] = int(color)


def _parse_grid(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)

    sprite: dict | None = None
    anchors: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        box = tuple(map(int, comp["bbox"]))
        if _is_solid_2x2(comp, g):
            anchors.append({"color": color, "coords": coords, "bbox": box})
        else:
            assert sprite is None, "expected exactly one non-anchor sprite"
            sprite = {"color": color, "coords": coords, "bbox": box}

    assert sprite is not None, "missing initial sprite"
    assert len(anchors) >= 2, "relay needs at least two anchors"
    colors = [int(sprite["color"])] + [int(anchor["color"]) for anchor in anchors]
    assert len(colors) == len(set(colors)), "sprite and anchors must use distinct colors"
    anchors.sort(key=lambda item: (item["bbox"], item["color"]))
    return bg, sprite, anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the sprite through the uniquely aligned 2x2 anchors and keep only each mirrored relay product.

    Input structure:
    - A black background contains one 8-connected sprite that is not a solid 2x2 block.
    - Several anchors are present as solid 2x2 squares, each in a distinct color.
    - The initial sprite aligns by row-band or column-band with exactly one anchor, and each new mirrored product aligns with exactly one unvisited anchor.

    Rule:
    - Starting from the initial sprite, repeatedly find the unique unvisited anchor that is cardinally aligned with the current sprite.
    - Slide the current sprite straight toward that anchor until it touches the anchor's near side, then mirror it across that side and recolor the mirrored copy with the anchor color.
    - Use the mirrored copy as the next current sprite, visit every anchor once, and erase the original sprite, all moved intermediates, and all anchors.

    Key insight:
    - The anchors do not come with a global order; each mirrored product becomes the next sprite, so the relay path emerges one station at a time from the current alignment state.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, sprite, anchors = _parse_grid(g)
    trace = _trace_relay(sprite["coords"], anchors)

    out = np.full_like(g, bg)
    for step in trace:
        _paint_coords(out, step["product"], int(step["anchor"]["color"]))
    return out


def _sample_move_sequence(n_anchors: int) -> list[str]:
    while True:
        moves = [random.choice(MOVES)]
        for _ in range(1, n_anchors):
            prev = moves[-1]
            if random.random() < 0.7:
                if prev in ("left", "right"):
                    choices = ["up", "down"]
                else:
                    choices = ["left", "right"]
            else:
                choices = list(MOVES)
            moves.append(random.choice(choices))
        if n_anchors < 3:
            return moves
        axes = ["h" if move in ("left", "right") else "v" for move in moves]
        if len(set(axes)) > 1:
            return moves


def _choose_anchor_box(
    current_coords: np.ndarray, move: str, gap: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(current_coords)
    gap = int(gap)
    if move == "right":
        ar = random.randint(r0 - 1, r1)
        ac = c1 + gap + 1
    elif move == "left":
        ar = random.randint(r0 - 1, r1)
        ac = c0 - gap - 2
    elif move == "down":
        ar = r1 + gap + 1
        ac = random.randint(c0 - 1, c1)
    else:
        ar = r0 - gap - 2
        ac = random.randint(c0 - 1, c1)
    return int(ar), int(ac), int(ar + 1), int(ac + 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _coords_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(_coords_key(a) & _coords_key(b))


def _build_expected_output(shape: tuple[int, int], products: list[np.ndarray], colors: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for coords, color in zip(products, colors):
        _paint_coords(out, coords, int(color))
    return out


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    initial_coords: np.ndarray,
    anchor_specs: list[dict],
    move_sequence: list[str],
) -> bool:
    try:
        bg, sprite, anchors = _parse_grid(inp)
        if bg != BG:
            return False
        if _coords_key(sprite["coords"]) != _coords_key(initial_coords):
            return False
        trace = _trace_relay(sprite["coords"], anchors)
    except Exception:
        return False

    traced_boxes = [tuple(step["anchor"]["bbox"]) for step in trace]
    expected_boxes = [tuple(spec["bbox"]) for spec in anchor_specs]
    traced_moves = [str(step["move"]) for step in trace]
    if traced_boxes != expected_boxes or traced_moves != move_sequence:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    non_bg = int(np.count_nonzero(expected))
    if non_bg <= len(anchor_specs):
        return False
    return True


def _generate_case(return_meta: bool = False):
    n_anchors = random.choices([2, 3, 4], weights=[2, 5, 3], k=1)[0]
    sprite_mask = random.choice(SHAPES)
    initial_coords = np.argwhere(sprite_mask > 0).astype(int)
    move_sequence = _sample_move_sequence(n_anchors)

    current = initial_coords.copy()
    anchor_specs: list[dict] = []
    products: list[np.ndarray] = []
    initial_box = _bbox(initial_coords)

    for step_index, move in enumerate(move_sequence):
        placed = False
        for _ in range(80):
            anchor_box = _choose_anchor_box(current, move, gap=random.randint(1, 5))
            if _boxes_overlap(anchor_box, initial_box):
                continue
            if any(_boxes_overlap(anchor_box, spec["bbox"]) for spec in anchor_specs):
                continue

            align = _alignment_for_anchor(current, anchor_box)
            if align is None:
                continue
            product = _slide_and_mirror(current, anchor_box, *align)
            if any(_coords_overlap(product, earlier) for earlier in products):
                continue

            anchor_specs.append({"bbox": anchor_box, "step": step_index})
            products.append(product)
            current = product
            placed = True
            break
        if not placed:
            return None

    color_pool = random.sample(PALETTE, n_anchors + 1)
    sprite_color = int(color_pool[0])
    anchor_colors = [int(color) for color in color_pool[1:]]
    for spec, color in zip(anchor_specs, anchor_colors):
        spec["color"] = color

    all_parts = [initial_coords] + [_anchor_coords(spec["bbox"]) for spec in anchor_specs] + products
    union = np.vstack(all_parts)
    r0, c0, r1, c1 = _bbox(union)
    inner_h = r1 - r0 + 1
    inner_w = c1 - c0 + 1
    if inner_h > MAX_SIDE or inner_w > MAX_SIDE:
        return None

    extra_h = MAX_SIDE - inner_h
    extra_w = MAX_SIDE - inner_w
    top = random.randint(0, min(4, extra_h))
    bottom = random.randint(0, min(4, extra_h - top))
    left = random.randint(0, min(4, extra_w))
    right = random.randint(0, min(4, extra_w - left))
    height = inner_h + top + bottom
    width = inner_w + left + right

    dr = top - r0
    dc = left - c0
    shifted_initial = _shift(initial_coords, dr, dc)
    shifted_products = [_shift(coords, dr, dc) for coords in products]
    shifted_specs = []
    for spec in anchor_specs:
        ar0, ac0, ar1, ac1 = spec["bbox"]
        shifted_specs.append(
            {
                "bbox": (ar0 + dr, ac0 + dc, ar1 + dr, ac1 + dc),
                "color": int(spec["color"]),
                "step": int(spec["step"]),
            }
        )

    inp = np.full((height, width), BG, dtype=int)
    for r, c in shifted_initial:
        inp[int(r), int(c)] = sprite_color
    for spec in shifted_specs:
        ar0, ac0, ar1, ac1 = spec["bbox"]
        inp[ar0 : ar1 + 1, ac0 : ac1 + 1] = int(spec["color"])

    expected = _build_expected_output((height, width), shifted_products, [spec["color"] for spec in shifted_specs])
    if not _validate_case(inp, expected, shifted_initial, shifted_specs, move_sequence):
        return None

    if return_meta:
        return {
            "input": inp,
            "output": expected,
            "initial": shifted_initial,
            "anchors": shifted_specs,
            "moves": move_sequence,
            "products": shifted_products,
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        case = _generate_case(return_meta=False)
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid mirror-relay ARC input")
