from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_RETRIES = 500
_GEN_COUNTER = 0


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            bbox = _bbox_coords(coords)
            r0, c0, _r1, _c1 = bbox
            mask = np.zeros((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": {(int(rr), int(cc)) for rr, cc in cells},
                    "bbox": bbox,
                    "mask": mask,
                    "area": int(len(cells)),
                }
            )
    return comps


def _reflect_coords(coords: np.ndarray, orient: str, axis2: int) -> set[tuple[int, int]]:
    if orient == "v":
        return {(int(r), int(axis2 - c)) for r, c in coords}
    return {(int(axis2 - r), int(c)) for r, c in coords}


def _side_of_bbox(bbox: tuple[int, int, int, int], orient: str, axis2: int) -> int:
    r0, c0, r1, c1 = bbox
    if orient == "v":
        if 2 * c1 < axis2:
            return -1
        if 2 * c0 > axis2:
            return 1
        return 0
    if 2 * r1 < axis2:
        return -1
    if 2 * r0 > axis2:
        return 1
    return 0


def _candidate_axis_from_pair(a: dict, b: dict, orient: str) -> int | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    if orient == "v":
        if ar0 != br0 or ar1 != br1:
            return None
        axis2_a = ac0 + bc1
        axis2_b = ac1 + bc0
    else:
        if ac0 != bc0 or ac1 != bc1:
            return None
        axis2_a = ar0 + br1
        axis2_b = ar1 + br0
    if axis2_a != axis2_b:
        return None
    axis2 = int(axis2_a)
    if _reflect_coords(a["coords"], orient, axis2) != b["coord_set"]:
        return None
    return axis2


def _cycle_mappings(colors: list[int]) -> list[dict[int, int]]:
    assert len(colors) == 3
    a, b, c = list(colors)
    return [{a: b, b: c, c: a}, {a: c, c: b, b: a}]


def _matching_options(edge_map: dict[tuple[int, int], tuple[int, int, int, int]], n_comps: int) -> list[tuple[list[tuple[int, int, int, int]], int]]:
    results: list[tuple[list[tuple[int, int, int, int]], int]] = []

    def rec(remaining: tuple[int, ...], orphan: int | None, chosen: list[tuple[int, int, int, int]]) -> None:
        if not remaining:
            if orphan is not None:
                results.append((chosen.copy(), orphan))
            return

        first = remaining[0]
        rest = remaining[1:]

        if orphan is None:
            rec(rest, first, chosen)

        for idx, second in enumerate(rest):
            key = (first, second) if first < second else (second, first)
            edge = edge_map.get(key)
            if edge is None:
                continue
            nxt = rest[:idx] + rest[idx + 1 :]
            chosen.append(edge)
            rec(nxt, orphan, chosen)
            chosen.pop()

    rec(tuple(range(n_comps)), None, [])
    return results


def _enumerate_solutions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _find_components(g, bg=BG)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]

    assert len(colors) == 3, "expected exactly three foreground colors"
    assert len(comps) >= 5 and len(comps) % 2 == 1, "expected pairs plus one orphan"

    candidates: set[tuple[str, int]] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            for orient in ("v", "h"):
                axis2 = _candidate_axis_from_pair(comps[i], comps[j], orient)
                if axis2 is not None:
                    candidates.add((orient, axis2))

    solutions: dict[tuple, dict] = {}
    for orient, axis2 in sorted(candidates):
        edge_map: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for i in range(len(comps)):
            side_i = _side_of_bbox(comps[i]["bbox"], orient, axis2)
            if side_i == 0:
                continue
            for j in range(i + 1, len(comps)):
                side_j = _side_of_bbox(comps[j]["bbox"], orient, axis2)
                if side_i * side_j != -1:
                    continue
                left_idx, right_idx = (i, j) if side_i < 0 else (j, i)
                if _reflect_coords(comps[left_idx]["coords"], orient, axis2) != comps[right_idx]["coord_set"]:
                    continue
                edge_map[(i, j)] = (
                    int(left_idx),
                    int(right_idx),
                    int(comps[left_idx]["color"]),
                    int(comps[right_idx]["color"]),
                )

        if len(edge_map) < (len(comps) - 1) // 2:
            continue

        for pairs, orphan_idx in _matching_options(edge_map, len(comps)):
            if len(pairs) < 2:
                continue
            candidate_cycles = [
                mapping
                for mapping in _cycle_mappings(colors)
                if all(mapping[src_color] == dst_color for _li, _ri, src_color, dst_color in pairs)
            ]
            if len(candidate_cycles) != 1:
                continue

            mapping = candidate_cycles[0]
            inverse = {dst: src for src, dst in mapping.items()}
            orphan = comps[orphan_idx]
            orphan_side = _side_of_bbox(orphan["bbox"], orient, axis2)
            if orphan_side == 0:
                continue

            reflected = _reflect_coords(orphan["coords"], orient, axis2)
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in reflected):
                continue
            if any(int(g[rr, cc]) != BG for rr, cc in reflected):
                continue

            orphan_color = int(orphan["color"])
            partner_color = mapping[orphan_color] if orphan_side < 0 else inverse[orphan_color]
            out = g.copy()
            for rr, cc in reflected:
                out[rr, cc] = partner_color

            map_key = tuple(mapping[color] for color in sorted(mapping))
            pair_key = tuple(sorted((li, ri, src, dst) for li, ri, src, dst in pairs))
            key = (orient, axis2, orphan_idx, map_key, pair_key)
            solutions[key] = {
                "orient": orient,
                "axis2": int(axis2),
                "pairs": pairs,
                "mapping": dict(mapping),
                "inverse": inverse,
                "orphan_idx": int(orphan_idx),
                "orphan_side": int(orphan_side),
                "orphan_color": orphan_color,
                "partner_color": int(partner_color),
                "out": out,
            }

    return list(solutions.values())


def _solve_unique(grid: np.ndarray) -> np.ndarray:
    solutions = _enumerate_solutions(grid)
    assert len(solutions) == 1, "expected exactly one valid axis/pairing/color-cycle explanation"
    return solutions[0]["out"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden mirror axis and one-way three-color cycle, then draw the missing mirrored partner of the lone orphan sprite.

    Input structure:
    - The background is black, and every foreground object is a monochromatic 4-connected sprite.
    - All but one sprite can be grouped into mirrored pairs across one common unseen vertical or horizontal axis.
    - In every true pair, the left/top sprite color is transformed into the right/bottom sprite color by one fixed three-color cycle.

    Rule:
    - Recover the unique common mirror axis from the paired sprite shapes.
    - Recover the unique non-involutive three-color mapping that fits every left/up to right/down pair.
    - Find the single unpaired orphan, reflect its shape across the same axis, and paint the missing partner with the forward color if the orphan is on the left/top side or with the inverse color if it is on the right/bottom side.
    - Leave all existing pixels unchanged.

    Key insight:
    - Neither the axis nor the color rule is drawn explicitly; only the joint evidence from multiple mirrored pairs reveals both the axis location and the direction of the three-color cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve_unique(g)


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                cells.append((r, c))
    return cells


def _random_connected_mask(h: int, w: int, cells: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < cells:
        mask[random.choice(_frontier(mask))] = True
    return _crop_bbox(mask)


def _shape_key(mask: np.ndarray) -> tuple:
    variants = [
        mask,
        mask[:, ::-1],
        mask[::-1, :],
        mask[::-1, ::-1],
    ]
    return (mask.shape, min(v.tobytes() for v in variants))


def _asymmetric_enough(mask: np.ndarray, orient: str) -> bool:
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    if int(mask.sum()) == mask.size:
        return False
    if orient == "v" and np.array_equal(mask, mask[:, ::-1]):
        return False
    if orient == "h" and np.array_equal(mask, mask[::-1, :]):
        return False
    if np.array_equal(mask, mask[:, ::-1]) or np.array_equal(mask, mask[::-1, :]):
        return False
    return True


def _random_shape(orient: str, used_keys: set[tuple]) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        max_cells = min(h * w - 1, 8)
        if max_cells < 4:
            continue
        mask = _random_connected_mask(h, w, random.randint(4, max_cells))
        if not _asymmetric_enough(mask, orient):
            continue
        key = _shape_key(mask)
        if key in used_keys:
            continue
        used_keys.add(key)
        return mask
    return None


def _pack_starts(total_len: int, sizes: list[int], margin: int = 1, gap: int = 1) -> list[int] | None:
    need = 2 * margin + sum(sizes) + gap * (len(sizes) - 1)
    if need > total_len:
        return None
    extra = total_len - need
    shares = np.random.multinomial(extra, [1.0 / (len(sizes) + 1)] * (len(sizes) + 1))
    starts: list[int] = []
    pos = margin + int(shares[0])
    for i, size in enumerate(sizes):
        starts.append(int(pos))
        pos += int(size)
        if i + 1 < len(sizes):
            pos += gap + int(shares[i + 1])
    return starts


def _valid_left_starts(width: int, sprite_w: int, axis2: int) -> list[int]:
    valid: list[int] = []
    for c0 in range(width - sprite_w + 1):
        c1 = c0 + sprite_w - 1
        if 2 * c1 >= axis2:
            continue
        mc0 = axis2 - c1
        mc1 = axis2 - c0
        if 0 <= mc0 <= mc1 < width:
            valid.append(int(c0))
    return valid


def _valid_top_starts(height: int, sprite_h: int, axis2: int) -> list[int]:
    valid: list[int] = []
    for r0 in range(height - sprite_h + 1):
        r1 = r0 + sprite_h - 1
        if 2 * r1 >= axis2:
            continue
        mr0 = axis2 - r1
        mr1 = axis2 - r0
        if 0 <= mr0 <= mr1 < height:
            valid.append(int(r0))
    return valid


def _candidate_axes(length: int, spans: list[int], want_parity: int) -> list[int]:
    axes: list[int] = []
    for axis2 in range(1, 2 * length - 2):
        if axis2 % 2 != want_parity:
            continue
        pos = axis2 / 2.0
        if not (0.28 * (length - 1) <= pos <= 0.72 * (length - 1)):
            continue
        fits = True
        for span in spans:
            if not _valid_left_starts(length, span, axis2):
                fits = False
                break
        if fits:
            axes.append(int(axis2))
    return axes


def _candidate_axes_horizontal(length: int, spans: list[int], want_parity: int) -> list[int]:
    axes: list[int] = []
    for axis2 in range(1, 2 * length - 2):
        if axis2 % 2 != want_parity:
            continue
        pos = axis2 / 2.0
        if not (0.28 * (length - 1) <= pos <= 0.72 * (length - 1)):
            continue
        fits = True
        for span in spans:
            if not _valid_top_starts(length, span, axis2):
                fits = False
                break
        if fits:
            axes.append(int(axis2))
    return axes


def _render_pairs(
    orient: str,
    axis2: int,
    shape_masks: list[np.ndarray],
    source_colors: list[int],
    mapping: dict[int, int],
    grid_shape: tuple[int, int],
    anchors: list[tuple[int, int]],
) -> tuple[np.ndarray, list[dict]]:
    h, w = grid_shape
    out = np.zeros((h, w), dtype=int)
    pairs: list[dict] = []

    for mask, src_color, (r0, c0) in zip(shape_masks, source_colors, anchors):
        coords = np.argwhere(mask)
        left = {(int(r0 + rr), int(c0 + cc)) for rr, cc in coords}
        right = _reflect_coords(np.array(sorted(left), dtype=int), orient, axis2)
        right_color = int(mapping[src_color])

        for rr, cc in left:
            out[rr, cc] = int(src_color)
        for rr, cc in right:
            out[rr, cc] = right_color

        pairs.append(
            {
                "shape": mask.copy(),
                "left_coords": left,
                "right_coords": right,
                "left_color": int(src_color),
                "right_color": right_color,
            }
        )
    return out, pairs


def _preferences() -> dict[str, int | str]:
    global _GEN_COUNTER
    idx = _GEN_COUNTER
    _GEN_COUNTER += 1
    return {
        "orient": "v" if idx % 2 == 0 else "h",
        "orphan_side": -1 if (idx // 2) % 2 == 0 else 1,
        "axis_parity": idx % 2,
        "pair_count": 4 if idx % 3 == 0 else 3,
    }


def _build_scene(prefer: dict[str, int | str]) -> tuple[np.ndarray, np.ndarray, dict] | None:
    orient = str(prefer["orient"])
    orphan_side = int(prefer["orphan_side"])
    axis_parity = int(prefer["axis_parity"])
    pair_count = int(prefer["pair_count"])

    cycle_colors = random.sample(list(range(1, 10)), 3)
    random.shuffle(cycle_colors)
    mapping = {
        int(cycle_colors[0]): int(cycle_colors[1]),
        int(cycle_colors[1]): int(cycle_colors[2]),
        int(cycle_colors[2]): int(cycle_colors[0]),
    }

    used_keys: set[tuple] = set()
    shape_masks: list[np.ndarray] = []
    for _ in range(pair_count):
        shape = _random_shape(orient, used_keys)
        if shape is None:
            return None
        shape_masks.append(shape)

    if pair_count == 3:
        source_colors = cycle_colors.copy()
    else:
        source_colors = cycle_colors.copy() + [random.choice(cycle_colors)]
        random.shuffle(source_colors)

    if orient == "v":
        heights = [mask.shape[0] for mask in shape_masks]
        widths = [mask.shape[1] for mask in shape_masks]
        min_h = 2 + sum(heights) + (pair_count - 1)
        if min_h > 29:
            return None
        grid_h = random.randint(min_h, min(29, min_h + 7))
        row_starts = _pack_starts(grid_h, heights)
        if row_starts is None:
            return None
        candidate_widths = list(range(max(12, 2 * max(widths) + 4), 30))
        random.shuffle(candidate_widths)
        axes: list[tuple[int, int]] = []
        for grid_w in candidate_widths:
            for axis2 in _candidate_axes(grid_w, widths, axis_parity):
                axes.append((grid_w, axis2))
        if not axes:
            return None
        grid_w, axis2 = random.choice(axes)
        anchors: list[tuple[int, int]] = []
        for r0, mask in zip(row_starts, shape_masks):
            c0 = random.choice(_valid_left_starts(grid_w, mask.shape[1], axis2))
            anchors.append((int(r0), int(c0)))
        full, pairs = _render_pairs(orient, axis2, shape_masks, source_colors, mapping, (grid_h, grid_w), anchors)
    else:
        heights = [mask.shape[0] for mask in shape_masks]
        widths = [mask.shape[1] for mask in shape_masks]
        min_w = 2 + sum(widths) + (pair_count - 1)
        if min_w > 29:
            return None
        grid_w = random.randint(min_w, min(29, min_w + 7))
        col_starts = _pack_starts(grid_w, widths)
        if col_starts is None:
            return None
        candidate_heights = list(range(max(12, 2 * max(heights) + 4), 30))
        random.shuffle(candidate_heights)
        axes: list[tuple[int, int]] = []
        for grid_h in candidate_heights:
            for axis2 in _candidate_axes_horizontal(grid_h, heights, axis_parity):
                axes.append((grid_h, axis2))
        if not axes:
            return None
        grid_h, axis2 = random.choice(axes)
        anchors = []
        for c0, mask in zip(col_starts, shape_masks):
            r0 = random.choice(_valid_top_starts(grid_h, mask.shape[0], axis2))
            anchors.append((int(r0), int(c0)))
        full, pairs = _render_pairs(orient, axis2, shape_masks, source_colors, mapping, (grid_h, grid_w), anchors)

    orphan_pair = random.randrange(pair_count)
    inp = full.copy()
    removed_coords = pairs[orphan_pair]["right_coords"] if orphan_side < 0 else pairs[orphan_pair]["left_coords"]
    for rr, cc in removed_coords:
        inp[rr, cc] = BG

    observed_color_pairs = [
        (int(pair["left_color"]), int(pair["right_color"]))
        for idx, pair in enumerate(pairs)
        if idx != orphan_pair
    ]
    orphan_color = int(pairs[orphan_pair]["left_color"] if orphan_side < 0 else pairs[orphan_pair]["right_color"])
    inverse = {dst: src for src, dst in mapping.items()}
    expected_partner = mapping[orphan_color] if orphan_side < 0 else inverse[orphan_color]

    meta = {
        "orient": orient,
        "axis2": int(axis2),
        "mapping": dict(mapping),
        "inverse": inverse,
        "orphan_side": int(orphan_side),
        "orphan_color": int(orphan_color),
        "expected_partner": int(expected_partner),
        "observed_color_pairs": observed_color_pairs,
        "pair_count": int(pair_count),
        "shape_masks": [mask.copy() for mask in shape_masks],
        "cycle_colors": cycle_colors[:],
    }
    return inp, full, meta


def _validate_instance(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if inp.shape != expected.shape:
        return False
    if len([color for color in np.unique(inp) if int(color) != BG]) != 3:
        return False
    if meta["pair_count"] < 3:
        return False
    if any(not _asymmetric_enough(mask, meta["orient"]) for mask in meta["shape_masks"]):
        return False
    if len({src for src, _dst in meta["observed_color_pairs"]}) < 2:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    solutions = _enumerate_solutions(inp)
    if len(solutions) != 1:
        return False
    sol = solutions[0]
    if sol["orient"] != meta["orient"] or int(sol["axis2"]) != int(meta["axis2"]):
        return False
    if sol["mapping"] != meta["mapping"]:
        return False
    if int(sol["orphan_side"]) != int(meta["orphan_side"]):
        return False
    if int(sol["orphan_color"]) != int(meta["orphan_color"]):
        return False
    if int(sol["partner_color"]) != int(meta["expected_partner"]):
        return False
    if not np.array_equal(sol["out"], expected):
        return False

    alt_mappings = [mapping for mapping in _cycle_mappings(meta["cycle_colors"]) if mapping != meta["mapping"]]
    if len(alt_mappings) != 1:
        return False
    alt_mapping = alt_mappings[0]
    if all(alt_mapping[src] == dst for src, dst in meta["observed_color_pairs"]):
        return False

    wrong_partner = meta["inverse"][meta["orphan_color"]] if meta["orphan_side"] < 0 else meta["mapping"][meta["orphan_color"]]
    if int(wrong_partner) == int(meta["expected_partner"]):
        return False
    return True


def generate_input() -> np.ndarray:
    prefer = _preferences()
    for _ in range(GEN_RETRIES):
        scene = _build_scene(prefer)
        if scene is None:
            continue
        inp, expected, meta = scene
        if _validate_instance(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid hidden-axis color-cycle input")
