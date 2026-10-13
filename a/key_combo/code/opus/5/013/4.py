from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_task_4", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox_coords = _ARC.bbox_coords
components_mask = _ARC.components_mask
get_bg = _ARC.get_bg


BG = Color.BLACK
CENTER = Color.YELLOW
LAYER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.PINK)
GEN_ATTEMPTS = 400


def _structure_info(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], dict[int, int]]:
    yellow = np.argwhere(grid == int(CENTER))
    assert len(yellow) == 1, "expected a unique yellow center pixel"
    center = (int(yellow[0, 0]), int(yellow[0, 1]))

    structure_coords = None
    for comp in components_mask(grid != int(bg), connectivity=4):
        if bool(np.any((comp[:, 0] == center[0]) & (comp[:, 1] == center[1]))):
            structure_coords = comp
            break
    assert structure_coords is not None, "yellow center must belong to the nested structure"

    structure_mask = np.zeros_like(grid, dtype=bool)
    structure_mask[structure_coords[:, 0], structure_coords[:, 1]] = True

    r0, c0, r1, c1 = bbox_coords(structure_coords)
    side = int(r1 - r0 + 1)
    assert side == int(c1 - c0 + 1) and side % 2 == 1, "structure bbox must be an odd square"
    assert bool(structure_mask[r0 : r1 + 1, c0 : c1 + 1].all()), "nested structure must fill its square bbox"

    cr, cc = (int((r0 + r1) // 2), int((c0 + c1) // 2))
    assert (cr, cc) == center, "yellow pixel must sit at the square center"

    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    rr, cc2 = np.indices(crop.shape)
    rr = np.abs(rr - side // 2)
    cc2 = np.abs(cc2 - side // 2)
    dist = np.maximum(rr, cc2)

    shell_colors: list[int] = []
    for d in range(1, side // 2 + 1):
        vals = np.unique(crop[dist == int(d)])
        assert len(vals) == 1 and int(vals[0]) != int(bg), "each shell radius must have one non-bg color"
        shell_colors.append(int(vals[0]))

    color_to_thickness: dict[int, int] = {}
    run_color = None
    run_len = 0
    for col in shell_colors + [-1]:
        if col == run_color:
            run_len += 1
            continue
        if run_color is not None:
            assert run_color not in color_to_thickness, "each layer color must be unique"
            color_to_thickness[int(run_color)] = int(run_len)
        run_color = None if col == -1 else int(col)
        run_len = 1

    assert 2 <= len(color_to_thickness) <= 4, "expected 2-4 colored layers"
    assert set(color_to_thickness).issubset(set(LAYER_COLORS)), "unexpected layer color"
    return structure_mask, (int(r0), int(c0), int(r1), int(c1)), center, color_to_thickness


def _marker_specs(
    grid: np.ndarray,
    bg: int,
    structure_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    color_to_thickness: dict[int, int],
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    marker_mask = (grid != int(bg)) & ~structure_mask
    specs: list[dict] = []
    for comp in components_mask(marker_mask, connectivity=4):
        assert len(comp) == 1, "each outside marker must be a single pixel"
        r, c = map(int, comp[0])
        color = int(grid[r, c])
        assert color in color_to_thickness, "marker color must match a frame layer color"

        if r0 <= r <= r1 and c < c0:
            dr, dc = 0, 1
        elif r0 <= r <= r1 and c > c1:
            dr, dc = 0, -1
        elif c0 <= c <= c1 and r < r0:
            dr, dc = 1, 0
        elif c0 <= c <= c1 and r > r1:
            dr, dc = -1, 0
        else:
            raise AssertionError("marker must lie on a row or column extension of the square")

        specs.append(
            {
                "r": int(r),
                "c": int(c),
                "dr": int(dr),
                "dc": int(dc),
                "color": int(color),
                "steps": int(color_to_thickness[color]),
            }
        )

    assert 4 <= len(specs) <= 8, "expected 4-8 markers"
    return specs


def _slide_markers(grid: np.ndarray, bg: int, specs: list[dict], *, reverse: bool = False) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for spec in specs:
        out[int(spec["r"]), int(spec["c"])] = int(bg)

    order = sorted(specs, key=lambda s: (int(s["r"]), int(s["c"])), reverse=bool(reverse))
    h, w = out.shape
    for spec in order:
        r, c = int(spec["r"]), int(spec["c"])
        dr, dc = int(spec["dr"]), int(spec["dc"])
        for _ in range(int(spec["steps"])):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(out[nr, nc]) != int(bg):
                break
            r, c = nr, nc
        out[r, c] = int(spec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each outside marker toward the concentric square by the thickness of its same-colored layer.

    Input structure:
    - The background is black, and one connected central object is a filled concentric-square structure with a yellow 1x1 center.
    - Around that center, 2-4 square layers use distinct colors, and each layer occupies a thickness of 1-3 shell radii.
    - Outside the square, 4-8 single-pixel markers reuse those layer colors and sit on row or column extensions of the square.

    Rule:
    - Identify the concentric square component containing the yellow center and read each colored layer's thickness.
    - For every outside marker, look up the thickness of the frame layer with the same color.
    - Move the marker horizontally or vertically toward the square by that many steps, stopping early if the next cell is non-background; keep the square unchanged and clear the marker's original cell.

    Key insight:
    - The nested square is a color-keyed numeric legend: each layer's visible thickness becomes the slide distance for markers of that same color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    structure_mask, bbox, _center, color_to_thickness = _structure_info(g, bg)
    specs = _marker_specs(g, bg, structure_mask, bbox, color_to_thickness)
    return _slide_markers(g, bg, specs)


def _sample_layers() -> list[tuple[int, int]]:
    for _ in range(200):
        n_layers = random.randint(2, 4)
        thicknesses = [random.randint(1, 3) for _ in range(n_layers)]
        if sum(thicknesses) > 9:
            continue
        if len(set(thicknesses)) < 2:
            continue
        if max(thicknesses) < 2:
            continue
        colors = random.sample(list(LAYER_COLORS), n_layers)
        return [(int(color), int(thickness)) for color, thickness in zip(colors, thicknesses)]
    raise RuntimeError("failed to sample non-degenerate layer thicknesses")


def _draw_structure(shape: tuple[int, int], layers: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int], dict[int, int]]:
    h, w = map(int, shape)
    total_radius = sum(thickness for _color, thickness in layers)
    side = 1 + 2 * total_radius
    grid = np.zeros((h, w), dtype=int)
    cr, cc = h // 2, w // 2
    r0, c0 = cr - total_radius, cc - total_radius
    r1, c1 = cr + total_radius, cc + total_radius

    rr, cc2 = np.indices(grid.shape)
    dist = np.maximum(np.abs(rr - cr), np.abs(cc2 - cc))
    prev = 0
    color_to_thickness: dict[int, int] = {}
    for color, thickness in layers:
        mask = (dist > int(prev)) & (dist <= int(prev + thickness))
        grid[mask] = int(color)
        color_to_thickness[int(color)] = int(thickness)
        prev += int(thickness)
    grid[cr, cc] = int(CENTER)

    assert side == int(r1 - r0 + 1) == int(c1 - c0 + 1)
    return grid, (int(r0), int(c0), int(r1), int(c1)), color_to_thickness


def _marker_entries(color_to_thickness: dict[int, int]) -> list[dict]:
    colors = list(color_to_thickness)
    total = random.randint(max(4, len(colors)), min(8, 2 * len(colors)))
    dup_colors = random.sample(colors, total - len(colors))
    entries = [{"color": int(color), "thickness": int(color_to_thickness[color])} for color in colors]
    entries.extend({"color": int(color), "thickness": int(color_to_thickness[color])} for color in dup_colors)
    random.shuffle(entries)

    collision_ix = [i for i, entry in enumerate(entries) if int(entry["thickness"]) >= 2]
    assert collision_ix, "at least one marker must support a collision-stop branch"
    forced_collision = random.choice(collision_ix)
    forced_free = random.choice([i for i in range(len(entries)) if i != forced_collision])

    for i, entry in enumerate(entries):
        if i == forced_collision:
            entry["mode"] = "collision"
            entry["orientation"] = "horizontal"
        elif i == forced_free:
            entry["mode"] = "free"
            entry["orientation"] = "vertical"
        else:
            entry["mode"] = random.choice(["collision", "exact", "free"])
            entry["orientation"] = random.choice(["horizontal", "vertical"])
    return entries


def _gap_choices(mode: str, thickness: int, max_gap: int) -> list[int]:
    if mode == "collision":
        return list(range(1, min(int(thickness) - 1, int(max_gap)) + 1))
    if mode == "exact":
        return [int(thickness)] if int(thickness) <= int(max_gap) else []
    if mode == "free":
        return list(range(int(thickness) + 1, int(max_gap) + 1))
    raise ValueError(f"unknown marker mode: {mode}")


def _coord_for_side(side: str, line: int, gap: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "L":
        return int(line), int(c0 - gap - 1)
    if side == "R":
        return int(line), int(c1 + gap + 1)
    if side == "T":
        return int(r0 - gap - 1), int(line)
    if side == "B":
        return int(r1 + gap + 1), int(line)
    raise ValueError(f"unknown side: {side}")


def _placement_candidates(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    entry: dict,
    used_lines: dict[str, set[int]],
    placed: list[dict],
) -> list[dict]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    line_pool = {
        "L": list(range(r0, r1 + 1)),
        "R": list(range(r0, r1 + 1)),
        "T": list(range(c0, c1 + 1)),
        "B": list(range(c0, c1 + 1)),
    }
    max_gap = {
        "L": c0 - 1,
        "R": w - c1 - 2,
        "T": r0 - 1,
        "B": h - r1 - 2,
    }
    side_groups = [["L", "R"], ["T", "B"]] if entry["orientation"] == "horizontal" else [["T", "B"], ["L", "R"]]

    existing = {(int(m["r"]), int(m["c"])) for m in placed}
    candidates: list[dict] = []
    for group in side_groups:
        sides = group[:]
        random.shuffle(sides)
        for side in sides:
            gaps = _gap_choices(str(entry["mode"]), int(entry["thickness"]), int(max_gap[side]))
            if not gaps:
                continue
            random.shuffle(gaps)
            lines = [line for line in line_pool[side] if int(line) not in used_lines[side]]
            random.shuffle(lines)
            for line in lines:
                for gap in gaps:
                    r, c = _coord_for_side(side, int(line), int(gap), bbox)
                    if not (0 <= r < h and 0 <= c < w):
                        continue
                    if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in existing):
                        continue
                    candidates.append(
                        {
                            "r": int(r),
                            "c": int(c),
                            "side": side,
                            "line": int(line),
                            "gap": int(gap),
                            "color": int(entry["color"]),
                            "thickness": int(entry["thickness"]),
                        }
                    )
        if candidates:
            break
    return candidates


def _place_markers(shape: tuple[int, int], bbox: tuple[int, int, int, int], entries: list[dict]) -> list[dict] | None:
    used_lines = {"L": set(), "R": set(), "T": set(), "B": set()}
    placed: list[dict] = []

    order = list(range(len(entries)))
    order.sort(key=lambda i: (entries[i]["mode"] != "collision", -int(entries[i]["thickness"])))
    for idx in order:
        entry = entries[idx]
        candidates = _placement_candidates(shape, bbox, entry, used_lines, placed)
        if not candidates:
            return None
        choice = random.choice(candidates)
        used_lines[str(choice["side"])].add(int(choice["line"]))
        placed.append(choice)
    return placed


def _spec_from_marker(marker: dict, bbox: tuple[int, int, int, int]) -> dict:
    r0, c0, r1, c1 = bbox
    r, c = int(marker["r"]), int(marker["c"])
    if marker["side"] == "L":
        dr, dc = 0, 1
    elif marker["side"] == "R":
        dr, dc = 0, -1
    elif marker["side"] == "T":
        dr, dc = 1, 0
    else:
        dr, dc = -1, 0

    assert (
        (marker["side"] in {"L", "R"} and r0 <= r <= r1)
        or (marker["side"] in {"T", "B"} and c0 <= c <= c1)
    )
    return {
        "r": int(r),
        "c": int(c),
        "dr": int(dr),
        "dc": int(dc),
        "color": int(marker["color"]),
        "steps": int(marker["thickness"]),
    }


def _expected_output(inp: np.ndarray, bbox: tuple[int, int, int, int], markers: list[dict]) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    for marker in markers:
        out[int(marker["r"]), int(marker["c"])] = int(BG)
    for marker in markers:
        spec = _spec_from_marker(marker, bbox)
        move = min(int(marker["thickness"]), int(marker["gap"]))
        rr = int(spec["r"] + spec["dr"] * move)
        cc = int(spec["c"] + spec["dc"] * move)
        out[rr, cc] = int(marker["color"])
    return out


def _validate_case(inp: np.ndarray, expected: np.ndarray, bbox: tuple[int, int, int, int], color_to_thickness: dict[int, int], markers: list[dict]) -> bool:
    try:
        structure_mask, parsed_bbox, _center, parsed_map = _structure_info(inp, BG)
        if tuple(parsed_bbox) != tuple(bbox):
            return False
        if parsed_map != color_to_thickness:
            return False

        specs = _marker_specs(inp, BG, structure_mask, bbox, parsed_map)
        produced = _slide_markers(inp, BG, specs)
        produced_rev = _slide_markers(inp, BG, specs, reverse=True)
        if not np.array_equal(produced, expected):
            return False
        if not np.array_equal(produced_rev, expected):
            return False
        if not np.array_equal(main(inp), expected):
            return False
    except Exception:
        return False

    moves = [min(int(marker["thickness"]), int(marker["gap"])) for marker in markers]
    final_gaps = [int(marker["gap"] - move) for marker, move in zip(markers, moves)]
    if len(set(moves)) < 2:
        return False
    if not any(int(marker["gap"]) < int(marker["thickness"]) for marker in markers):
        return False
    if not any(int(marker["gap"]) > int(marker["thickness"]) for marker in markers):
        return False
    if 0 not in final_gaps or not any(gap > 0 for gap in final_gaps):
        return False
    if {"horizontal", "vertical"} - {("horizontal" if m["side"] in {"L", "R"} else "vertical") for m in markers}:
        return False
    return not np.array_equal(inp, expected)


def _gen_attempt() -> np.ndarray | None:
    layers = _sample_layers()
    total_radius = sum(thickness for _color, thickness in layers)
    side = 1 + 2 * total_radius
    h = random.randint(max(17, side + 10), min(29, side + 12))
    w = random.randint(max(17, side + 10), min(29, side + 12))

    grid, bbox, color_to_thickness = _draw_structure((h, w), layers)
    entries = _marker_entries(color_to_thickness)
    markers = _place_markers((h, w), bbox, entries)
    if markers is None:
        return None

    inp = grid.copy()
    for marker in markers:
        inp[int(marker["r"]), int(marker["c"])] = int(marker["color"])

    expected = _expected_output(inp, bbox, markers)
    if not _validate_case(inp, expected, bbox, color_to_thickness, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-thickness slide puzzle")
