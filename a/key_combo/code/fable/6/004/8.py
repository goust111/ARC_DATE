from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


MAX_ATTEMPTS = 400
FRAME_COUNT_CHOICES = (2, 3, 4)
FRAME_HEIGHT_CHOICES = (5, 7)
FRAME_WIDTH_MAX = 13
GRID_MIN = 22
GRID_MAX = 29
FRAME_COLORS = tuple(range(10))
SOLID_SHAPES = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (2, 5), (5, 2), (3, 4), (4, 3))
HOLLOW_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))
OVERLAP_AREA_PAIR = ((2, 4), (3, 3))


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    h, w = _bbox_hw(bbox)
    return h * w


def _frame_patch(color: int, h: int, w: int, bg: int) -> np.ndarray:
    patch = np.full((h, w), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _solid_patch(color: int, h: int, w: int) -> np.ndarray:
    return np.full((h, w), int(color), dtype=int)


def _hollow_patch(color: int, h: int, w: int, bg: int) -> np.ndarray:
    return _frame_patch(color, h, w, bg)


def _is_filled_rect(sub: np.ndarray, color: int) -> bool:
    return bool(np.all(sub == int(color)))


def _is_rect_ring(sub: np.ndarray, color: int, bg: int) -> bool:
    h, w = sub.shape
    if h < 3 or w < 3:
        return False
    if not np.all(sub[0, :] == int(color)):
        return False
    if not np.all(sub[-1, :] == int(color)):
        return False
    if not np.all(sub[:, 0] == int(color)):
        return False
    if not np.all(sub[:, -1] == int(color)):
        return False
    return bool(np.all(sub[1:-1, 1:-1] == int(bg)))


def _component_kind(grid: np.ndarray, comp: dict, bg: int) -> str:
    color = int(comp["color"])
    sub = crop_bbox(grid, comp["bbox"])
    if _is_filled_rect(sub, color):
        return "solid"
    if _is_rect_ring(sub, color, bg):
        return "ring"
    raise AssertionError(f"unrecognized component shape for color {color}")


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    by_color: dict[int, dict[str, list[dict]]] = {}
    for comp in comps:
        color = int(comp["color"])
        kind = _component_kind(g, comp, bg)
        entry = by_color.setdefault(color, {"solid": [], "ring": []})
        entry[kind].append(comp)

    frames: dict[int, dict] = {}
    pattern_components: list[dict] = []
    counts: dict[int, dict[str, int]] = {}
    for color, entry in by_color.items():
        rings = sorted(entry["ring"], key=lambda comp: _bbox_area(comp["bbox"]), reverse=True)
        solids = list(entry["solid"])
        assert rings, f"color {color} must have a frame ring"
        frame = rings[0]
        frame_h, frame_w = _bbox_hw(frame["bbox"])
        assert frame_h >= 5 and frame_w >= 5, "frame must be visibly larger than sprites"
        for ring in rings[1:]:
            ring_h, ring_w = _bbox_hw(ring["bbox"])
            assert frame_h > ring_h or frame_w > ring_w, "frame ring must be uniquely largest for its color"
        frames[color] = frame
        hollows = rings[1:]
        pattern_components.extend(solids)
        pattern_components.extend(hollows)
        counts[color] = {"solid": len(solids), "hollow": len(hollows)}
        assert counts[color]["solid"] > counts[color]["hollow"], "net count must be positive"

    assert len(frames) >= 2, "expected multiple colored frames"
    return {"bg": bg, "frames": frames, "counts": counts, "patterns": pattern_components}


def _draw_dots_in_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, n: int, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    row = (r0 + r1) // 2
    inner_w = c1 - c0 - 1
    assert n <= (inner_w + 1) // 2, "frame interior too narrow for required dots"
    for i in range(n):
        cc = c1 - 1 - 2 * i
        assert out[row, cc] == int(bg), "dot cells must start empty"
        out[row, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase same-colored solid and hollow sprites, then write each frame's signed count as interior dots.

    Input structure:
    - The dominant color is the background.
    - Several different colors each have one large hollow rectangular frame.
    - Outside the frames, the same colors appear as small sprites: filled rectangles count as solid sprites and small rectangular rings count as hollow sprites.

    Rule:
    - For each frame color C, count n = (# of C-colored solid sprites) - (# of C-colored hollow sprites).
    - Remove every small sprite from the grid while leaving all large frames in place.
    - Inside each frame, on the frame's center row, draw n dots of color C from right to left with one background cell between consecutive dots.

    Key insight:
    - Color only decides which sprites belong to which frame; the positive/negative sign is encoded by solidity versus hollowness of the same-colored sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    analysis = _analyze_grid(g)
    bg = int(analysis["bg"])
    out = g.copy()

    for comp in analysis["patterns"]:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = bg

    for color, frame in analysis["frames"].items():
        counts = analysis["counts"][color]
        net = counts["solid"] - counts["hollow"]
        _draw_dots_in_frame(out, frame["bbox"], int(color), int(net), bg)

    return out


def _sample_frame_specs() -> list[dict]:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    max_net = 4 if frame_count <= 3 else 3
    nets = [random.randint(1, max_net) for _ in range(frame_count)]
    if len(set(nets)) == 1:
        idx = random.randrange(frame_count)
        nets[idx] = 1 if nets[idx] != 1 else min(max_net, 2)

    hollow_enabled = set(random.sample(range(frame_count), random.randint(1, frame_count - 1)))
    specs = []
    for idx in range(frame_count):
        hollow = random.randint(1, 2) if idx in hollow_enabled else 0
        solid = nets[idx] + hollow
        outer_h = random.choice(FRAME_HEIGHT_CHOICES)
        min_w = max(5, 2 * nets[idx] + 1)
        outer_w = random.randint(min_w, FRAME_WIDTH_MAX)
        specs.append(
            {
                "net": nets[idx],
                "solid_count": solid,
                "hollow_count": hollow,
                "frame_h": outer_h,
                "frame_w": outer_w,
            }
        )
    return specs


def _build_pattern_plan(frame_specs: list[dict]) -> None:
    hollow_indices = [i for i, spec in enumerate(frame_specs) if spec["hollow_count"] > 0]
    assert hollow_indices
    solid_indices = [i for i, spec in enumerate(frame_specs) if spec["solid_count"] > 0]

    for spec in frame_specs:
        spec["solid_shapes"] = [random.choice(SOLID_SHAPES) for _ in range(spec["solid_count"])]
        spec["hollow_shapes"] = [random.choice(HOLLOW_SHAPES) for _ in range(spec["hollow_count"])]

    pair_solid_idx = random.choice(solid_indices)
    pair_hollow_idx = random.choice(hollow_indices)
    frame_specs[pair_solid_idx]["solid_shapes"][0] = OVERLAP_AREA_PAIR[0]
    frame_specs[pair_hollow_idx]["hollow_shapes"][0] = OVERLAP_AREA_PAIR[1]


def _sample_canvas(frame_count: int) -> tuple[int, int]:
    low = GRID_MIN + (2 if frame_count == 4 else 0)
    return random.randint(low, GRID_MAX), random.randint(low, GRID_MAX)


def _place_patch(
    canvas: np.ndarray,
    shadow: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    border: int,
    padding: int,
) -> tuple[int, int]:
    rect = np.ones(patch.shape, dtype=int)
    r0, c0 = random_free_location(
        shadow,
        rect,
        transparent=0,
        border=border,
        padding=padding,
        padding_connectivity=8,
    )
    paste(shadow, rect, r0, c0)
    paste(canvas, patch, r0, c0, transparent=bg)
    return int(r0), int(c0)


def _planned_output(inp: np.ndarray, placements: dict[int, dict], bg: int) -> np.ndarray:
    out = np.array(inp, copy=True)
    for info in placements.values():
        for coords in info["pattern_coords"]:
            for r, c in coords:
                out[int(r), int(c)] = int(bg)
    for color, info in placements.items():
        _draw_dots_in_frame(out, info["frame_bbox"], int(color), int(info["net"]), int(bg))
    return out


def _semantic_checks(inp: np.ndarray, placements: dict[int, dict], expected: np.ndarray, bg: int) -> bool:
    analysis = _analyze_grid(inp)
    if int(analysis["bg"]) != int(bg):
        return False

    if set(analysis["frames"]) != set(placements):
        return False

    if not any(info["hollow_count"] > 0 for info in placements.values()):
        return False
    if not any(info["hollow_count"] == 0 for info in placements.values()):
        return False
    if not any(info["hollow_count"] > 0 and info["solid_count"] > info["hollow_count"] for info in placements.values()):
        return False
    if len({info["net"] for info in placements.values()}) < 2:
        return False

    for color, info in placements.items():
        parsed_frame_bbox = tuple(map(int, analysis["frames"][color]["bbox"]))
        if parsed_frame_bbox != tuple(info["frame_bbox"]):
            return False
        parsed_counts = analysis["counts"][color]
        if parsed_counts["solid"] != info["solid_count"] or parsed_counts["hollow"] != info["hollow_count"]:
            return False

    if not np.array_equal(main(inp), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    frame_specs = _sample_frame_specs()
    _build_pattern_plan(frame_specs)

    frame_count = len(frame_specs)
    H, W = _sample_canvas(frame_count)
    palette = random.sample(FRAME_COLORS, frame_count + 1)
    bg = palette[0]
    colors = palette[1:]

    canvas = np.full((H, W), int(bg), dtype=int)
    shadow = np.zeros((H, W), dtype=int)
    placements: dict[int, dict] = {}

    try:
        for color, spec in sorted(zip(colors, frame_specs), key=lambda item: item[1]["frame_h"] * item[1]["frame_w"], reverse=True):
            frame = _frame_patch(color, spec["frame_h"], spec["frame_w"], bg)
            r0, c0 = _place_patch(canvas, shadow, frame, bg=bg, border=1, padding=2)
            frame_bbox = (r0, c0, r0 + spec["frame_h"] - 1, c0 + spec["frame_w"] - 1)
            placements[int(color)] = {
                "frame_bbox": frame_bbox,
                "net": spec["net"],
                "solid_count": spec["solid_count"],
                "hollow_count": spec["hollow_count"],
                "pattern_coords": [],
            }

        pattern_items = []
        for color, spec in zip(colors, frame_specs):
            for h, w in spec["solid_shapes"]:
                pattern_items.append((h * w, color, "solid", (h, w)))
            for h, w in spec["hollow_shapes"]:
                pattern_items.append((h * w, color, "hollow", (h, w)))
        pattern_items.sort(reverse=True)

        for _area, color, kind, shape in pattern_items:
            if kind == "solid":
                patch = _solid_patch(color, shape[0], shape[1])
            else:
                patch = _hollow_patch(color, shape[0], shape[1], bg)
            r0, c0 = _place_patch(canvas, shadow, patch, bg=bg, border=1, padding=1)
            coords = [(r0 + r, c0 + c) for r, c in np.argwhere(patch == int(color))]
            placements[int(color)]["pattern_coords"].append(coords)
    except ValueError:
        return None

    if max(canvas.shape) > 30:
        return None
    if np.count_nonzero(canvas != int(bg)) >= canvas.size // 2:
        return None

    expected = _planned_output(canvas, placements, bg)
    if not _semantic_checks(canvas, placements, expected, bg):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
