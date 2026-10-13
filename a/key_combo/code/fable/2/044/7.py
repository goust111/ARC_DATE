from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


ALL_COLORS = tuple(range(10))
INPUT_SIDE_CHOICES = (3, 5, 7)
GRID_LIMIT = 30
OBJECT_COUNT_CHOICES = (4, 5)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
RELAY_PROB = 0.6
SECONDARY_TARGET_PROB = 0.7


def _box_for(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    half = int(side) // 2
    r, c = center
    return int(r - half), int(c - half), int(r + half), int(c + half)


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _extract_squares(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1), "foreground objects must have square bboxes"
        assert side % 2 == 1, "square side lengths must be odd"
        assert int(comp["area"]) == side * side, "squares must fill their bboxes"

        patch = crop_bbox(g, comp["bbox"])
        colors = sorted(set(map(int, patch.flatten())) - {bg})
        assert 1 <= len(colors) <= 2, "each square must be monochrome or center-marked"

        center = side // 2
        center_color = int(patch[center, center])
        if len(colors) == 1:
            body_color = int(colors[0])
            marker_color = None
            assert np.all(patch == body_color), "pure square must be solid"
        else:
            assert side >= 3, "marked squares must be at least 3x3"
            ring_mask = np.ones((side, side), dtype=bool)
            ring_mask[center, center] = False
            ring_colors = sorted(set(map(int, patch[ring_mask].flatten())))
            assert len(ring_colors) == 1, "all non-center cells must share one body color"
            body_color = int(ring_colors[0])
            marker_color = center_color
            assert marker_color != body_color, "center marker must differ from body color"
            assert np.all(patch[ring_mask] == body_color), "only the center may be recolored"

        squares.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": (r0 + center, c0 + center),
                "side": side,
                "body_color": body_color,
                "marker_color": marker_color,
            }
        )

    assert squares, "expected at least one square"
    body_colors = [square["body_color"] for square in squares]
    assert len(body_colors) == len(set(body_colors)), "square body colors must be unique"
    by_color = {square["body_color"]: square for square in squares}

    for square in squares:
        marker_color = square["marker_color"]
        if marker_color is None:
            continue
        assert marker_color in by_color, "marker color must point to an existing body color"
        assert marker_color != square["body_color"], "marker must point to another square"

    return g, bg, squares


def _settle_squares(squares: list[dict]) -> list[dict]:
    incoming = Counter(
        int(square["marker_color"])
        for square in squares
        if square["marker_color"] is not None
    )

    settled: list[dict] = []
    for square in squares:
        outflow = 1 if square["marker_color"] is not None else 0
        inflow = int(incoming.get(int(square["body_color"]), 0))
        new_side = int(square["side"]) + 2 * (inflow - outflow)
        assert new_side >= 1 and new_side % 2 == 1, "net settlement must keep a positive odd side"
        settled.append(
            {
                **square,
                "incoming": inflow,
                "outgoing": outflow,
                "new_side": new_side,
            }
        )
    return settled


def _render_squares(
    shape: tuple[int, int],
    bg: int,
    squares: list[dict],
    *,
    side_key: str,
    with_markers: bool,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    painted = np.zeros(shape, dtype=bool)
    H, W = shape

    for square in squares:
        side = int(square[side_key])
        color = int(square["body_color"])
        r0, c0, r1, c1 = _box_for(square["center"], side)
        assert 0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W, "square must fit inside the canvas"
        region = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        assert not painted[region].any(), "redrawn squares must stay separated"

        patch = np.full((side, side), color, dtype=int)
        if with_markers and square["marker_color"] is not None:
            patch[side // 2, side // 2] = int(square["marker_color"])
        paste(out, patch, r0, c0)
        painted[region] = True

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle all ring transfers across the color-pointer square network and redraw each square at its net side length.

    Input structure:
    - The grid has one dominant background color.
    - Each foreground object is a separated solid odd-sided square with a unique body color.
    - Some squares have their center cell recolored to another square's body color; the rest stay pure.

    Rule:
    - Treat each recolored center as a pointer from that square to the square whose body color matches the marker.
    - A pointing square loses one ring of thickness 1, so its side length decreases by 2.
    - Every time a square is pointed to, it gains one ring, so its side length increases by 2 per incoming pointer.
    - Clear the canvas and redraw every square at the same center with its original body color and the net-settled side length; do not redraw the markers.

    Key insight:
    - The center color is a directed pointer, not decoration: each square's final size depends on the whole pointer graph through incoming minus outgoing ring transfers.
    """
    g, bg, squares = _extract_squares(grid)
    settled = _settle_squares(squares)
    return _render_squares(g.shape, bg, settled, side_key="new_side", with_markers=False)


def _sample_graph(n: int, rng: random.Random) -> list[int | None]:
    order = list(range(n))
    rng.shuffle(order)

    hub = order[0]
    sender_a = order[1]
    sender_b = order[2]
    extras = order[3:]

    targets: list[int | None] = [None] * n
    targets[sender_a] = hub
    targets[sender_b] = hub

    if extras and rng.random() < RELAY_PROB:
        relay_target = extras[0]
        targets[hub] = relay_target
        if len(extras) > 1 and rng.random() < SECONDARY_TARGET_PROB:
            targets[extras[1]] = relay_target
    else:
        if len(extras) > 1 and rng.random() < SECONDARY_TARGET_PROB:
            targets[extras[1]] = extras[0]

    return targets


def _pick_input_side(delta: int, rng: random.Random) -> int:
    choices = [side for side in INPUT_SIDE_CHOICES if 1 <= side + 2 * delta <= 9]
    assert choices, "generator only supports net changes that keep output sides in {1,3,5,7,9}"

    weights = []
    for side in choices:
        out_side = side + 2 * delta
        weight = 1
        if out_side in {1, 9}:
            weight += 2
        if side in {3, 7}:
            weight += 1
        weights.append(weight)
    return int(rng.choices(choices, weights=weights, k=1)[0])


def _axis_positions(count: int, max_foot: int, pitch: int, total: int, rng: random.Random) -> list[int]:
    margin = max_foot // 2
    min_total = max_foot + (count - 1) * pitch
    slack = int(total - min_total)
    assert slack >= 0

    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[rng.randrange(count + 1)] += 1

    positions = [margin + extras[0]]
    for idx in range(1, count):
        positions.append(positions[-1] + pitch + extras[idx])

    assert positions[0] >= margin
    assert positions[-1] <= total - 1 - margin - extras[-1]
    return positions


def _sample_layout(n: int, max_foot: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int]]] | None:
    pitch = max_foot + PLACEMENT_GAP
    layouts = []
    for rows in (2, 3):
        cols = (n + rows - 1) // rows
        if rows * cols < n:
            continue
        min_h = max_foot + (rows - 1) * pitch
        min_w = max_foot + (cols - 1) * pitch
        if min_h <= GRID_LIMIT and min_w <= GRID_LIMIT:
            layouts.append((rows, cols, min_h, min_w))
    if not layouts:
        return None

    rows, cols, min_h, min_w = rng.choice(layouts)
    H = rng.randint(min_h, GRID_LIMIT)
    W = rng.randint(min_w, GRID_LIMIT)
    row_positions = _axis_positions(rows, max_foot, pitch, H, rng)
    col_positions = _axis_positions(cols, max_foot, pitch, W, rng)
    slots = [(r, c) for r in row_positions for c in col_positions]
    rng.shuffle(slots)
    centers = slots[:n]
    return H, W, centers


def _build_specs(rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    n = int(rng.choice(OBJECT_COUNT_CHOICES))
    bg = int(rng.choice(ALL_COLORS))
    body_colors = rng.sample([color for color in ALL_COLORS if color != bg], n)
    targets = _sample_graph(n, rng)
    incoming = Counter(target for target in targets if target is not None)

    specs: list[dict] = []
    output_sides = []
    for idx, body_color in enumerate(body_colors):
        outflow = 1 if targets[idx] is not None else 0
        inflow = int(incoming.get(idx, 0))
        delta = inflow - outflow
        input_side = _pick_input_side(delta, rng)
        output_side = input_side + 2 * delta
        specs.append(
            {
                "index": idx,
                "body_color": int(body_color),
                "marker_color": None if targets[idx] is None else int(body_colors[int(targets[idx])]),
                "input_side": int(input_side),
                "output_side": int(output_side),
                "incoming": inflow,
                "outgoing": outflow,
            }
        )
        output_sides.append(int(output_side))

    if len({spec["input_side"] for spec in specs}) == 1 and len(set(output_sides)) == 1:
        return None

    max_foot = max(max(spec["input_side"], spec["output_side"]) for spec in specs)
    layout = _sample_layout(n, max_foot, rng)
    if layout is None:
        return None
    H, W, centers = layout

    for spec, center in zip(specs, centers):
        spec["center"] = center

    for side_key in ("input_side", "output_side"):
        boxes = [_box_for(spec["center"], int(spec[side_key])) for spec in specs]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not _boxes_separated(boxes[i], boxes[j], PLACEMENT_GAP):
                    return None

    inp_specs = [
        {
            "center": spec["center"],
            "body_color": spec["body_color"],
            "marker_color": spec["marker_color"],
            "side": spec["input_side"],
        }
        for spec in specs
    ]
    out_specs = [
        {
            "center": spec["center"],
            "body_color": spec["body_color"],
            "marker_color": None,
            "side": spec["output_side"],
        }
        for spec in specs
    ]

    inp = _render_squares((H, W), bg, inp_specs, side_key="side", with_markers=True)
    out = _render_squares((H, W), bg, out_specs, side_key="side", with_markers=False)
    return inp, out, specs


def _validate_generated_sample(inp: np.ndarray, out: np.ndarray, specs: list[dict]) -> bool:
    try:
        parsed_inp, bg, squares = _extract_squares(inp)
        assert parsed_inp.shape == out.shape
        assert np.array_equal(main(inp), out)

        by_color = {int(square["body_color"]): square for square in squares}
        assert set(by_color) == {int(spec["body_color"]) for spec in specs}

        for spec in specs:
            parsed = by_color[int(spec["body_color"])]
            assert parsed["center"] == spec["center"]
            assert int(parsed["side"]) == int(spec["input_side"])
            assert parsed["marker_color"] == spec["marker_color"]

        incoming = Counter(
            int(spec["marker_color"])
            for spec in specs
            if spec["marker_color"] is not None
        )
        assert incoming and max(incoming.values()) >= 2
        local_only = {
            int(spec["body_color"]): int(spec["input_side"]) - 2 if spec["marker_color"] is not None else int(spec["input_side"])
            for spec in specs
        }
        assert any(int(spec["output_side"]) != local_only[int(spec["body_color"])] for spec in specs)

        out_squares = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
        assert len(out_squares) == len(specs)
        for comp in out_squares:
            color = int(comp["color"])
            spec = next(item for item in specs if int(item["body_color"]) == color)
            r0, c0, r1, c1 = map(int, comp["bbox"])
            side = int(r1 - r0 + 1)
            assert side == int(spec["output_side"])
            assert int(comp["area"]) == side * side

        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_specs(rng)
        if built is None:
            continue
        inp, out, specs = built
        if _validate_generated_sample(inp, out, specs):
            return inp
    raise RuntimeError("failed to generate a valid ring-transfer square input")
