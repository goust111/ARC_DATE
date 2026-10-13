from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BLOCK_SIZES = [3, 4]
GRID_SIZE_RANGE = (22, 29)
GROUP_COUNT_RANGE = (2, 4)
RETRY_LIMIT = 400

SPRITE_LIBRARY = [
    frozenset({(0, 0), (1, 0), (2, 0)}),
    frozenset({(0, 0), (0, 1), (0, 2)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (1, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    frozenset({(0, 1), (1, 1), (2, 1), (2, 0)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1), (2, 1)}),
    frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    frozenset({(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)}),
    frozenset({(0, 2), (1, 2), (2, 0), (2, 1), (2, 2), (1, 1)}),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stepper = _neighbors4 if connectivity == 4 else _neighbors8
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in stepper(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _try_parse_motif(grid: np.ndarray, bg: int, color: int, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h % 2 != 0 or h < 4:
        return None
    size = h // 2
    if len(coords) != 2 * size * size:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not set(map(int, np.unique(sub))).issubset({bg, color}):
        return None
    filled: list[tuple[int, int]] = []
    for br in range(2):
        for bc in range(2):
            block = sub[br * size : (br + 1) * size, bc * size : (bc + 1) * size]
            if np.all(block == color):
                filled.append((br, bc))
            elif not np.all(block == bg):
                return None
    filled_set = set(filled)
    if filled_set not in ({(0, 0), (1, 1)}, {(0, 1), (1, 0)}):
        return None
    missing_origins = []
    for br in range(2):
        for bc in range(2):
            if (br, bc) not in filled_set:
                missing_origins.append((r0 + br * size, c0 + bc * size))
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "size": size,
        "filled": tuple(sorted(filled_set)),
        "missing_origins": tuple(missing_origins),
        "coords": tuple(sorted(coords)),
        "coord_set": set(coords),
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]

    motifs: dict[int, dict] = {}
    for color in colors:
        motif_candidates = []
        for comp in _components(g == color, connectivity=8):
            parsed = _try_parse_motif(g, bg, color, comp)
            if parsed is not None:
                motif_candidates.append(parsed)
        if motif_candidates:
            assert len(motif_candidates) == 1, f"ambiguous motifs for color {color}"
            motifs[color] = motif_candidates[0]
    motif_colors = set(motifs)
    assert 2 <= len(motif_colors) <= 4

    legend_map: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    for comp in _components(g != bg, connectivity=4):
        if len(comp) != 2:
            continue
        pair_colors = {int(g[r, c]) for r, c in comp}
        if len(pair_colors) != 2:
            continue
        a, b = sorted(pair_colors)
        if (a in motif_colors) == (b in motif_colors):
            continue
        src = a if a in motif_colors else b
        tgt = b if src == a else a
        assert src not in legend_map, f"duplicate legend mapping for color {src}"
        legend_map[src] = tgt
        legend_cells.update(comp)
    assert set(legend_map) == motif_colors

    sprites: dict[int, dict] = {}
    for color, motif in motifs.items():
        sprite_candidates = []
        for comp in _components(g == color, connectivity=4):
            area = len(comp)
            if not 3 <= area <= 6:
                continue
            if any(cell in motif["coord_set"] for cell in comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 > motif["size"] or c1 - c0 + 1 > motif["size"]:
                continue
            sprite_candidates.append(comp)
        assert len(sprite_candidates) == 1, f"expected one sprite for color {color}"
        comp = sprite_candidates[0]
        r0, c0, r1, c1 = _bbox(comp)
        sprites[color] = {
            "color": color,
            "origin": (r0, c0),
            "height": r1 - r0 + 1,
            "width": c1 - c0 + 1,
            "shape": _normalize(comp),
            "area": len(comp),
        }

    return {
        "bg": bg,
        "motifs": motifs,
        "legend_map": legend_map,
        "legend_cells": tuple(sorted(legend_cells)),
        "sprites": sprites,
    }


def _render_from_scene(grid: np.ndarray, scene: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(scene["bg"])
    out = np.full_like(g, bg)
    for cell in scene["legend_cells"]:
        out[cell] = g[cell]
    for color, motif in scene["motifs"].items():
        for r, c in motif["coords"]:
            out[r, c] = int(color)
        sprite = scene["sprites"][color]
        tgt = int(scene["legend_map"][color])
        for base_r, base_c in motif["missing_origins"]:
            for dr, dc in sprite["shape"]:
                rr = base_r + dr
                cc = base_c + dc
                if rr >= base_r + motif["size"] or cc >= base_c + motif["size"]:
                    continue
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = tgt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp same-color sprites into each diagonal vacancy and recolor them via the corner legend.

    Input structure:
    - The background is black.
    - Each source color appears in one diagonal 2x2 square-block motif, one detached monochrome sprite, and one two-pixel legend pair.
    - Every motif is a square footprint split into four equal sub-blocks, with exactly two same-color filled blocks on one diagonal and the opposite diagonal left empty.

    Rule:
    - For each motif, identify the two empty sub-block positions complementary to the filled diagonal.
    - Find the detached sprite that uses the same color as the motif and stamp that sprite, aligned to its own top-left bbox corner, into both empty sub-blocks.
    - Recolor the stamped sprite pixels to the target color given by the mixed-color two-pixel legend entry for that source color.
    - Keep the original filled motif blocks and the legend, and erase the original detached sprites.

    Key insight:
    - The vacancies determine where to stamp, the motif color determines which sprite shape to copy, and the legend determines the final color, so the transformation depends on all three cues.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _render_from_scene(np.asarray(grid, dtype=int), scene)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _pick_shapes(block_sizes: list[int]) -> list[tuple[tuple[int, int], ...]] | None:
    order = list(range(len(SPRITE_LIBRARY)))
    random.shuffle(order)
    chosen: list[tuple[tuple[int, int], ...]] = []
    for size in block_sizes:
        match = None
        for idx in order:
            shape = tuple(sorted(SPRITE_LIBRARY[idx]))
            if shape in chosen:
                continue
            h, w = _shape_dims(shape)
            if h <= size and w <= size:
                match = shape
                break
        if match is None:
            return None
        chosen.append(match)
    return chosen


def _reserve_bbox(
    reserved: set[tuple[int, int]],
    h: int,
    w: int,
    r0: int,
    c0: int,
    box_h: int,
    box_w: int,
    pad: int = 1,
) -> None:
    for rr in range(max(0, r0 - pad), min(h, r0 + box_h + pad)):
        for cc in range(max(0, c0 - pad), min(w, c0 + box_w + pad)):
            reserved.add((rr, cc))


def _bbox_free(
    reserved: set[tuple[int, int]],
    h: int,
    w: int,
    r0: int,
    c0: int,
    box_h: int,
    box_w: int,
    pad: int = 1,
) -> bool:
    if r0 < 0 or c0 < 0 or r0 + box_h > h or c0 + box_w > w:
        return False
    for rr in range(max(0, r0 - pad), min(h, r0 + box_h + pad)):
        for cc in range(max(0, c0 - pad), min(w, c0 + box_w + pad)):
            if (rr, cc) in reserved:
                return False
    return True


def _paint_shape(grid: np.ndarray, origin: tuple[int, int], shape: tuple[tuple[int, int], ...], color: int) -> None:
    r0, c0 = origin
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = int(color)


def _paint_motif(grid: np.ndarray, group: dict) -> None:
    size = int(group["size"])
    r0, c0 = group["bbox_origin"]
    for br, bc in group["filled"]:
        grid[r0 + br * size : r0 + (br + 1) * size, c0 + bc * size : c0 + (bc + 1) * size] = int(group["color"])


def _render_case_input(case: dict) -> np.ndarray:
    h, w = case["shape"]
    grid = np.full((h, w), case["bg"], dtype=int)
    for entry in case["legend"]:
        for r, c, color in entry["cells"]:
            grid[r, c] = int(color)
    for group in case["groups"]:
        _paint_motif(grid, group)
        _paint_shape(grid, group["sprite_origin"], group["sprite"], group["color"])
    return grid


def _render_case_output(
    case: dict,
    *,
    color_mode: str = "target",
    shape_override: dict[int, tuple[tuple[int, int], ...]] | None = None,
    use_missing_origins: bool = True,
) -> np.ndarray:
    h, w = case["shape"]
    grid = np.full((h, w), case["bg"], dtype=int)
    for entry in case["legend"]:
        for r, c, color in entry["cells"]:
            grid[r, c] = int(color)
    for group in case["groups"]:
        _paint_motif(grid, group)
        shape = group["sprite"] if shape_override is None else shape_override.get(group["color"], group["sprite"])
        stamp_color = group["target"] if color_mode == "target" else group["color"]
        origins = group["missing_origins"] if use_missing_origins else group["filled_origins"]
        for origin in origins:
            base_r, base_c = origin
            for dr, dc in shape:
                rr = base_r + dr
                cc = base_c + dc
                if rr >= base_r + group["size"] or cc >= base_c + group["size"]:
                    continue
                if 0 <= rr < h and 0 <= cc < w:
                    grid[rr, cc] = int(stamp_color)
    return grid


def _build_legend(h: int, w: int, groups: list[dict]) -> list[dict] | None:
    corner = random.choice(["tl", "tr", "bl", "br"])
    entries = []
    entry_specs = []
    for group in groups:
        orient = random.choice(["h", "v"])
        order = random.choice(["st", "ts"])
        entry_h = 1 if orient == "h" else 2
        entry_w = 2 if orient == "h" else 1
        entry_specs.append((group, orient, order, entry_h, entry_w))
    total_h = sum(spec[3] for spec in entry_specs) + (len(entry_specs) - 1)
    if total_h > h:
        return None
    row = 0 if corner in {"tl", "tr"} else h - total_h
    for group, orient, order, entry_h, entry_w in entry_specs:
        col = 0 if corner in {"tl", "bl"} else w - entry_w
        if orient == "h":
            pos = [(row, col), (row, col + 1)]
        else:
            pos = [(row, col), (row + 1, col)]
        colors = [group["color"], group["target"]] if order == "st" else [group["target"], group["color"]]
        entries.append({"cells": [(pos[0][0], pos[0][1], colors[0]), (pos[1][0], pos[1][1], colors[1])]})
        row += entry_h + 1
    return entries


def _make_group_payload(
    color: int,
    target: int,
    size: int,
    filled: tuple[tuple[int, int], tuple[int, int]],
    bbox_origin: tuple[int, int],
    sprite: tuple[tuple[int, int], ...],
    sprite_origin: tuple[int, int],
) -> dict:
    r0, c0 = bbox_origin
    missing = []
    filled_origins = []
    for br in range(2):
        for bc in range(2):
            origin = (r0 + br * size, c0 + bc * size)
            if (br, bc) in filled:
                filled_origins.append(origin)
            else:
                missing.append(origin)
    return {
        "color": int(color),
        "target": int(target),
        "size": int(size),
        "filled": tuple(sorted(filled)),
        "bbox_origin": (r0, c0),
        "missing_origins": tuple(missing),
        "filled_origins": tuple(filled_origins),
        "sprite": tuple(sorted(sprite)),
        "sprite_origin": tuple(sprite_origin),
    }


def _generate_case() -> dict | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    group_count = random.randint(*GROUP_COUNT_RANGE)
    colors = random.sample(PALETTE, 2 * group_count)
    source_colors = colors[:group_count]
    target_colors = colors[group_count:]
    block_sizes = [random.choice(BLOCK_SIZES) for _ in range(group_count)]
    sprites = _pick_shapes(block_sizes)
    if sprites is None:
        return None

    diagonals = [random.choice([((0, 0), (1, 1)), ((0, 1), (1, 0))]) for _ in range(group_count)]
    if group_count > 1 and len({tuple(d) for d in diagonals}) == 1:
        idx = random.randrange(group_count)
        diagonals[idx] = ((0, 1), (1, 0)) if diagonals[idx] == ((0, 0), (1, 1)) else ((0, 0), (1, 1))

    proto_groups = []
    for idx in range(group_count):
        proto_groups.append(
            {
                "color": source_colors[idx],
                "target": target_colors[idx],
                "size": block_sizes[idx],
                "filled": diagonals[idx],
                "sprite": sprites[idx],
            }
        )

    legend = _build_legend(h, w, proto_groups)
    if legend is None:
        return None

    reserved: set[tuple[int, int]] = set()
    for entry in legend:
        rows = [r for r, _, _ in entry["cells"]]
        cols = [c for _, c, _ in entry["cells"]]
        r0, c0 = min(rows), min(cols)
        box_h = max(rows) - r0 + 1
        box_w = max(cols) - c0 + 1
        _reserve_bbox(reserved, h, w, r0, c0, box_h, box_w)

    groups: list[dict] = []
    order = list(range(group_count))
    order.sort(key=lambda i: proto_groups[i]["size"], reverse=True)
    for idx in order:
        proto = proto_groups[idx]
        size = int(proto["size"])
        box_h = 2 * size
        box_w = 2 * size
        options = [(r, c) for r in range(h - box_h + 1) for c in range(w - box_w + 1)]
        random.shuffle(options)
        placed = None
        for r0, c0 in options:
            if _bbox_free(reserved, h, w, r0, c0, box_h, box_w):
                placed = (r0, c0)
                break
        if placed is None:
            return None
        _reserve_bbox(reserved, h, w, placed[0], placed[1], box_h, box_w)
        proto["bbox_origin"] = placed
        groups.append(proto)

    groups.sort(key=lambda g: g["color"])
    for group in groups:
        sprite_h, sprite_w = _shape_dims(group["sprite"])
        options = [(r, c) for r in range(h - sprite_h + 1) for c in range(w - sprite_w + 1)]
        random.shuffle(options)
        placed = None
        for r0, c0 in options:
            if _bbox_free(reserved, h, w, r0, c0, sprite_h, sprite_w):
                placed = (r0, c0)
                break
        if placed is None:
            return None
        _reserve_bbox(reserved, h, w, placed[0], placed[1], sprite_h, sprite_w)
        group["sprite_origin"] = placed

    payload_groups = [
        _make_group_payload(
            group["color"],
            group["target"],
            group["size"],
            group["filled"],
            group["bbox_origin"],
            group["sprite"],
            group["sprite_origin"],
        )
        for group in groups
    ]
    return {"shape": (h, w), "bg": 0, "groups": payload_groups, "legend": legend}


def _validate_case(case: dict, inp: np.ndarray, expected: np.ndarray) -> bool:
    groups = case["groups"]
    if len({group["color"] for group in groups}) != len(groups):
        return False
    if len({group["target"] for group in groups}) != len(groups):
        return False
    if {group["color"] for group in groups} & {group["target"] for group in groups}:
        return False
    if len({group["sprite"] for group in groups}) != len(groups):
        return False
    if len(groups) > 1 and len({group["filled"] for group in groups}) < 2:
        return False
    if not any(group["color"] != group["target"] for group in groups):
        return False

    scene = _parse_scene(inp)
    if set(scene["motifs"]) != {group["color"] for group in groups}:
        return False
    for group in groups:
        color = group["color"]
        motif = scene["motifs"][color]
        sprite = scene["sprites"][color]
        if motif["size"] != group["size"]:
            return False
        if tuple(motif["filled"]) != tuple(group["filled"]):
            return False
        if tuple(motif["missing_origins"]) != tuple(group["missing_origins"]):
            return False
        if scene["legend_map"][color] != group["target"]:
            return False
        if sprite["shape"] != group["sprite"]:
            return False

    if not np.array_equal(_render_from_scene(inp, scene), expected):
        return False

    no_recolor = _render_case_output(case, color_mode="source")
    if np.array_equal(no_recolor, expected):
        return False
    wrong_slots = _render_case_output(case, use_missing_origins=False)
    if np.array_equal(wrong_slots, expected):
        return False
    first_shape = groups[0]["sprite"]
    if any(group["sprite"] != first_shape for group in groups[1:]):
        overrides = {group["color"]: first_shape for group in groups}
        wrong_shape = _render_case_output(case, shape_override=overrides)
        if np.array_equal(wrong_shape, expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    case = _generate_case()
    if case is None:
        return None
    inp = _render_case_input(case)
    expected = _render_case_output(case)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _validate_case(case, inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-sprite-legend input")
