from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 7, 9)

GRID_MIN = 24
GRID_MAX = 28
EDGE_BAND = 7
BAR_MIN_GAP = 3
MAX_GEN_ATTEMPTS = 600
SHAPE_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


BBox = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components_mask(grid == int(color))


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _all_rotations_distinct(mask: np.ndarray) -> bool:
    sigs = {_mask_signature(np.rot90(mask, k)) for k in range(4)}
    return len(sigs) == 4


def _boxes_touch(a: BBox, b: BBox, pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _dominant_color(line: np.ndarray) -> int:
    vals, counts = np.unique(line, return_counts=True)
    idx = int(np.argmax(counts))
    assert vals[idx] != BG, "bar lines must be non-background"
    assert int(np.sum(counts == counts[idx])) == 1, "bar line needs a unique dominant color"
    assert int(counts[idx]) > line.size // 2, "dominant bar color must occupy most of its line"
    return int(vals[idx])


def _unique_topological_order(nodes: list[int], edges: set[tuple[int, int]]) -> list[int]:
    adj = {node: set() for node in nodes}
    indeg = {node: 0 for node in nodes}
    for low, high in edges:
        assert low in adj and high in adj and low != high
        if high not in adj[low]:
            adj[low].add(high)
            indeg[high] += 1

    remaining = set(nodes)
    order: list[int] = []
    while remaining:
        zeros = [node for node in nodes if node in remaining and indeg[node] == 0]
        assert zeros, "crossing constraints contain a cycle"
        assert len(zeros) == 1, "crossing constraints must determine one total order"
        node = zeros[0]
        remaining.remove(node)
        order.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
    return order


def _decode_scene(grid: np.ndarray) -> tuple[list[int], list[int], dict[int, int], dict[int, int], list[int]]:
    h, w = grid.shape
    bar_rows = [r for r in range(h) if np.all(grid[r, :] != BG)]
    bar_cols = [c for c in range(w) if np.all(grid[:, c] != BG)]
    assert bar_rows and bar_cols, "expected both horizontal and vertical full-span bars"
    assert 3 <= len(bar_rows) + len(bar_cols) <= 5, "expected 3 to 5 bars"

    row_color = {r: _dominant_color(grid[r, :]) for r in bar_rows}
    col_color = {c: _dominant_color(grid[:, c]) for c in bar_cols}
    all_colors = list(row_color.values()) + list(col_color.values())
    assert len(set(all_colors)) == len(all_colors), "bar colors must be unique"

    edges: set[tuple[int, int]] = set()
    for r, h_color in row_color.items():
        for c, v_color in col_color.items():
            visible = int(grid[r, c])
            assert visible in (h_color, v_color), "a crossing must show one of its two bars"
            if visible == h_color:
                edges.add((v_color, h_color))
            else:
                edges.add((h_color, v_color))

    order = _unique_topological_order(all_colors, edges)
    return bar_rows, bar_cols, row_color, col_color, order


def _extract_sprites(grid: np.ndarray, bar_rows: list[int], bar_cols: list[int], colors: list[int]) -> dict[int, dict]:
    stripped = grid.copy()
    stripped[bar_rows, :] = BG
    stripped[:, bar_cols] = BG

    remaining_colors = sorted(set(map(int, stripped.ravel())) - {BG})
    assert remaining_colors == sorted(colors), "every bar color must have one matching sprite"

    sprites: dict[int, dict] = {}
    for color in colors:
        comps = _color_components(stripped, color)
        assert len(comps) == 1, f"expected exactly one sprite for color {color}"
        coords = comps[0]
        box = _bbox(stripped == color)
        assert box is not None
        mask = _crop_bbox(stripped == color, box)
        area = int(mask.sum())
        assert 3 <= area <= 7, "sprite area must stay within the puzzle bounds"
        assert max(mask.shape) <= 4, "sprite bbox must stay within the puzzle bounds"
        sprites[color] = {"coords": coords, "bbox": box, "mask": mask}
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the full-span bars after using their crossing visibility order to rotate same-color corner sprites.

    Input structure:
    - Black background with 3 to 5 unique-colored full-span bars, some horizontal and some vertical.
    - A horizontal bar occupies one whole row and a vertical bar occupies one whole column; crossings show the upper bar's color.
    - Each bar color also appears once as a small 4-connected monochrome sprite away from the bars.

    Rule:
    - Detect the full non-background rows and columns, read each line's dominant color, and use every row/column crossing to infer which bar is above which.
    - Reconstruct the unique bottom-to-top layer order from those crossing constraints.
    - Remove all bars, then for each remaining same-color sprite rotate its bbox clockwise by (layer index mod 4) quarter-turns and place it back at the original bbox top-left.

    Key insight:
    - The bars are only instruction carriers: visible crossing colors encode one unique layer order, and that layer position is the rotation amount for the matching sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bar_rows, bar_cols, _row_color, _col_color, order = _decode_scene(g)
    sprites = _extract_sprites(g, bar_rows, bar_cols, order)

    out = np.zeros_like(g)
    for layer, color in enumerate(order):
        box = sprites[color]["bbox"]
        mask = sprites[color]["mask"]
        rotated = np.rot90(mask, -(layer % 4))
        patch = np.zeros(rotated.shape, dtype=int)
        patch[rotated] = int(color)
        _paste(out, patch, int(box[0]), int(box[1]), transparent=BG)
    return out


def _choose_spaced_positions(lo: int, hi: int, count: int, gap: int) -> list[int]:
    choices = [
        combo
        for combo in itertools.combinations(range(lo, hi + 1), count)
        if all(b - a >= gap for a, b in zip(combo, combo[1:]))
    ]
    assert choices, "failed to find spaced bar positions"
    picked = list(random.choice(choices))
    random.shuffle(picked)
    return picked


def _orientation_pattern(nh: int, nv: int) -> list[str]:
    assert abs(nh - nv) <= 1 and nh >= 1 and nv >= 1
    if nh == nv:
        start = random.choice(("H", "V"))
    else:
        start = "H" if nh > nv else "V"

    pattern: list[str] = []
    remaining = {"H": nh, "V": nv}
    cur = start
    for _ in range(nh + nv):
        assert remaining[cur] > 0
        pattern.append(cur)
        remaining[cur] -= 1
        cur = "V" if cur == "H" else "H"
    assert remaining == {"H": 0, "V": 0}
    return pattern


def _random_sprite_mask(max_box: int, *, require_non_square: bool) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, max_box)
        w = random.randint(2, max_box)
        if require_non_square and h == w:
            continue
        max_area = min(7, h * w - 1)
        if max_area < 3:
            continue
        target = random.randint(3, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        box = _bbox(mask)
        if box is None:
            continue
        mask = _crop_bbox(mask, box)
        if require_non_square and mask.shape[0] == mask.shape[1]:
            continue
        if min(mask.shape) < 2 or max(mask.shape) > max_box:
            continue
        if int(mask.sum()) < 3 or int(mask.sum()) > 7 or np.all(mask):
            continue
        if len(_components_mask(mask)) != 1:
            continue
        if not _all_rotations_distinct(mask):
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate sprite mask")


def _zone_box(zone: int, h: int, w: int) -> BBox:
    if zone == 0:
        return 0, 0, EDGE_BAND - 1, EDGE_BAND - 1
    if zone == 1:
        return 0, w - EDGE_BAND, EDGE_BAND - 1, w - 1
    if zone == 2:
        return h - EDGE_BAND, 0, h - 1, EDGE_BAND - 1
    if zone == 3:
        return h - EDGE_BAND, w - EDGE_BAND, h - 1, w - 1
    raise ValueError("invalid zone")


def _place_sprites(
    grid_h: int,
    grid_w: int,
    colors: list[int],
    layer_of: dict[int, int],
) -> dict[int, dict] | None:
    max_box = 3 if len(colors) == 5 else 4
    odd_colors = [color for color in colors if layer_of[color] % 4 in (1, 3)]
    non_square_color = random.choice(odd_colors)

    masks: dict[int, np.ndarray] = {}
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for color in random.sample(colors, len(colors)):
        require_non_square = color == non_square_color
        for _ in range(SHAPE_ATTEMPTS):
            mask = _random_sprite_mask(max_box, require_non_square=require_non_square)
            sig = _mask_signature(mask)
            if sig not in used_shapes:
                used_shapes.add(sig)
                masks[color] = mask
                break
        else:
            return None

    zone_multiset = [0, 1, 2, 3]
    if len(colors) == 5:
        zone_multiset.append(random.choice(zone_multiset))
    random.shuffle(zone_multiset)

    placements = []
    for color, zone in zip(colors, zone_multiset):
        mask = masks[color]
        rot = np.rot90(mask, -(layer_of[color] % 4))
        placements.append(
            {
                "color": color,
                "zone": zone,
                "mask": mask,
                "rot": rot,
                "size_key": max(mask.shape[0] * mask.shape[1], rot.shape[0] * rot.shape[1]),
            }
        )

    zone_counts = Counter(item["zone"] for item in placements)
    placements.sort(key=lambda item: (zone_counts[item["zone"]], item["size_key"]), reverse=True)

    input_boxes: list[BBox] = []
    output_boxes: list[BBox] = []
    placed: dict[int, dict] = {}

    for item in placements:
        color = int(item["color"])
        mask = item["mask"]
        rot = item["rot"]
        zone = int(item["zone"])
        zh0, zw0, zh1, zw1 = _zone_box(zone, grid_h, grid_w)
        max_h = max(mask.shape[0], rot.shape[0])
        max_w = max(mask.shape[1], rot.shape[1])
        rows = list(range(zh0, zh1 - max_h + 2))
        cols = list(range(zw0, zw1 - max_w + 2))
        if not rows or not cols:
            return None
        anchors = [(r0, c0) for r0 in rows for c0 in cols]
        random.shuffle(anchors)

        found = False
        for r0, c0 in anchors:
            in_box = (r0, c0, r0 + mask.shape[0] - 1, c0 + mask.shape[1] - 1)
            out_box = (r0, c0, r0 + rot.shape[0] - 1, c0 + rot.shape[1] - 1)
            if any(_boxes_touch(in_box, prev, pad=1) for prev in input_boxes):
                continue
            if any(_boxes_touch(out_box, prev, pad=1) for prev in output_boxes):
                continue
            input_boxes.append(in_box)
            output_boxes.append(out_box)
            placed[color] = {"bbox": in_box, "mask": mask, "rot": rot, "zone": zone}
            found = True
            break
        if not found:
            return None

    if not any(
        layer_of[color] % 4 in (1, 3) and placed[color]["mask"].shape != placed[color]["rot"].shape
        for color in colors
    ):
        return None
    return placed


def _render_instance(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    h = int(spec["shape"][0])
    w = int(spec["shape"][1])
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    for color in spec["order"]:
        if spec["orientation"][color] == "H":
            inp[int(spec["row_of"][color]), :] = int(color)
        else:
            inp[:, int(spec["col_of"][color])] = int(color)

    for color in spec["colors"]:
        info = spec["sprites"][color]
        r0, c0, _r1, _c1 = info["bbox"]
        in_patch = np.zeros(info["mask"].shape, dtype=int)
        in_patch[info["mask"]] = int(color)
        _paste(inp, in_patch, int(r0), int(c0), transparent=BG)

        out_patch = np.zeros(info["rot"].shape, dtype=int)
        out_patch[info["rot"]] = int(color)
        _paste(out, out_patch, int(r0), int(c0), transparent=BG)
    return inp, out


def _make_spec() -> dict | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)

    nh, nv = random.choice(((2, 2), (2, 3), (3, 2)))
    order_orients = _orientation_pattern(nh, nv)
    colors = random.sample(list(PALETTE), nh + nv)

    h_colors = random.sample(colors, nh)
    v_colors = [color for color in colors if color not in h_colors]
    random.shuffle(h_colors)
    random.shuffle(v_colors)

    order: list[int] = []
    orientation: dict[int, str] = {}
    h_idx = 0
    v_idx = 0
    for orient in order_orients:
        if orient == "H":
            color = int(h_colors[h_idx])
            h_idx += 1
        else:
            color = int(v_colors[v_idx])
            v_idx += 1
        order.append(color)
        orientation[color] = orient

    layer_of = {color: idx for idx, color in enumerate(order)}

    row_positions = _choose_spaced_positions(EDGE_BAND, h - EDGE_BAND - 1, nh, BAR_MIN_GAP)
    col_positions = _choose_spaced_positions(EDGE_BAND, w - EDGE_BAND - 1, nv, BAR_MIN_GAP)
    row_of = {color: row for color, row in zip(random.sample(h_colors, len(h_colors)), row_positions)}
    col_of = {color: col for color, col in zip(random.sample(v_colors, len(v_colors)), col_positions)}

    sprites = _place_sprites(h, w, colors, layer_of)
    if sprites is None:
        return None

    return {
        "shape": (h, w),
        "colors": list(colors),
        "order": order,
        "orientation": orientation,
        "row_of": row_of,
        "col_of": col_of,
        "layer_of": layer_of,
        "sprites": sprites,
    }


def _validate_instance(inp: np.ndarray, expected: np.ndarray, spec: dict) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30

    bar_rows, bar_cols, row_color, col_color, decoded_order = _decode_scene(inp)
    assert decoded_order == spec["order"], "decoded layer order mismatch"
    assert set(bar_rows) == set(spec["row_of"].values())
    assert set(bar_cols) == set(spec["col_of"].values())
    assert row_color == {row: color for color, row in spec["row_of"].items()}
    assert col_color == {col: color for color, col in spec["col_of"].items()}

    sprites = _extract_sprites(inp, bar_rows, bar_cols, decoded_order)
    assert set(sprites) == set(spec["colors"])

    erase_only = np.zeros_like(inp)
    for color in spec["colors"]:
        box = spec["sprites"][color]["bbox"]
        r0, c0, _r1, _c1 = box
        mask = spec["sprites"][color]["mask"]
        rot = spec["sprites"][color]["rot"]
        decoded_mask = sprites[color]["mask"]
        assert np.array_equal(decoded_mask, mask)
        assert np.array_equal(np.rot90(decoded_mask, -(spec["layer_of"][color] % 4)), rot)
        if spec["layer_of"][color] % 4 != 0:
            assert not np.array_equal(decoded_mask, rot), "non-zero rotation must visibly matter"

        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(color)
        _paste(erase_only, patch, int(r0), int(c0), transparent=BG)

    assert not np.array_equal(erase_only, expected), "rotation step must matter overall"
    assert any(
        spec["layer_of"][color] % 4 in (1, 3) and spec["sprites"][color]["mask"].shape != spec["sprites"][color]["rot"].shape
        for color in spec["colors"]
    ), "need at least one visible non-square odd rotation"
    assert not any(np.all(expected[r, :] != BG) for r in range(expected.shape[0]))
    assert not any(np.all(expected[:, c] != BG) for c in range(expected.shape[1]))
    assert np.array_equal(main(inp), expected), "solver output mismatch"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        spec = _make_spec()
        if spec is None:
            continue
        inp, expected = _render_instance(spec)
        try:
            _validate_instance(inp, expected, spec)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid layered-bar sprite-rotation instance")
