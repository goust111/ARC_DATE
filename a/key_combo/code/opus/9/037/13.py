from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
MAX_GEN_ATTEMPTS = 600
SPRITE_ATTEMPTS = 300
CASE_ATTEMPTS = 160


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_of_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def crop_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def nonzero_cells(patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int, int]]:
    cells = []
    for rr, cc in np.argwhere(patch != BG):
        cells.append((r0 + int(rr), c0 + int(cc), int(patch[rr, cc])))
    return cells


def coord_set(cells: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(r, c) for r, c, _ in cells}


def margin8(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(coords)
    h, w = shape
    for r, c in list(coords):
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                out.add((rr, cc))
    return out


def paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = dst.shape
    for rr, cc in np.argwhere(patch != BG):
        r = r0 + int(rr)
        c = c0 + int(cc)
        if 0 <= r < h and 0 <= c < w:
            dst[r, c] = int(patch[rr, cc])


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def classify_anchor(r: int, c: int, h: int, w: int) -> tuple[str, str, int, int]:
    if r == 0 and c not in (0, w - 1):
        return "top", "h", 0, c
    if c == w - 1:
        return "right", "v", 1, r
    if r == h - 1 and c not in (0, w - 1):
        return "bottom", "h", 2, c
    return "left", "v", 3, r


def detect_anchors(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    anchors = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color == BG:
                continue
            if all((not in_bounds(grid.shape, rr, cc)) or int(grid[rr, cc]) == BG for rr, cc in neighbors4(r, c)):
                edge, axis, group, idx = classify_anchor(r, c, h, w)
                anchors.append(
                    {
                        "pos": (r, c),
                        "color": color,
                        "edge": edge,
                        "axis": axis,
                        "order_key": (group, idx),
                    }
                )
    anchors.sort(key=lambda a: a["order_key"])
    return anchors


def extract_sprite(grid: np.ndarray, comp: list[tuple[int, int]], anchor_by_color: dict[int, dict]) -> dict:
    colors = Counter(int(grid[r, c]) for r, c in comp)
    marker_colors = [color for color, count in colors.items() if count == 1 and color in anchor_by_color]
    if len(marker_colors) != 1:
        raise ValueError("component does not have a unique marker color")
    marker_color = marker_colors[0]
    marker_pos = next((r, c) for r, c in comp if int(grid[r, c]) == marker_color)
    r0, c0, r1, c1 = bbox_of_coords(comp)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in comp:
        patch[r - r0, c - c0] = int(grid[r, c])
    return {
        "anchor": anchor_by_color[marker_color],
        "marker_color": marker_color,
        "marker_abs": marker_pos,
        "marker_rel": (marker_pos[0] - r0, marker_pos[1] - c0),
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "coords": comp,
    }


def transform_sprite(sprite: dict) -> dict:
    anchor = sprite["anchor"]
    patch = sprite["patch"]
    h, w = patch.shape
    r0, c0, _r1, _c1 = sprite["bbox"]
    mr, mc = sprite["marker_rel"]
    if anchor["axis"] == "h":
        delta = anchor["pos"][1] - sprite["marker_abs"][1]
        settled_r, settled_c = r0, c0 + delta
        sign = 1 if delta > 0 else -1
        copy_patch = np.fliplr(patch)
        copy_r, copy_c = settled_r, settled_c - sign * w
    else:
        delta = anchor["pos"][0] - sprite["marker_abs"][0]
        settled_r, settled_c = r0 + delta, c0
        sign = 1 if delta > 0 else -1
        copy_patch = np.flipud(patch)
        copy_r, copy_c = settled_r - sign * h, settled_c
    return {
        **sprite,
        "settled_tl": (settled_r, settled_c),
        "copy_patch": copy_patch,
        "copy_tl": (copy_r, copy_c),
    }


def render_records(shape: tuple[int, int], anchors: list[dict], records: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rec in sorted(records, key=lambda x: x["anchor"]["order_key"]):
        paste_patch(out, rec["patch"], rec["settled_tl"][0], rec["settled_tl"][1])
        paste_patch(out, rec["copy_patch"], rec["copy_tl"][0], rec["copy_tl"][1])
    for anchor in anchors:
        r, c = anchor["pos"]
        out[r, c] = int(anchor["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marker-tagged interior sprite to its matching border anchor, then append a mirrored copy on the side opposite the slide.

    Input structure:
    - Black (0) background fills the grid.
    - The interior contains 2-3 four-connected multicolor sprites, each using one repeated body color plus one marker color that appears exactly once.
    - The border contains isolated non-black anchor pixels whose colors match the sprites' marker colors.

    Rule:
    - Detect isolated border anchors and pair each interior sprite with the anchor sharing its unique marker color.
    - Top/bottom anchors drive horizontal sliding so the sprite marker column matches the anchor column; left/right anchors, including corners, drive vertical sliding so the marker row matches the anchor row.
    - Clear the original sprite positions, place each slid sprite at its aligned location, flip that aligned sprite along the slide axis, and concatenate the flipped copy flush against the side opposite the slide direction.
    - Apply sprites in anchor order `top -> right -> bottom -> left`, with ties broken by border coordinate, and keep the anchors unchanged in the output.

    Key insight:
    - The anchor does more than choose a destination: its edge decides the slide axis, while the actual slide direction decides which side receives the mirrored copy.
    """
    g = np.asarray(grid, dtype=int)
    anchors = detect_anchors(g)
    anchor_by_color = {anchor["color"]: anchor for anchor in anchors}
    if len(anchor_by_color) != len(anchors):
        raise ValueError("anchor colors must be unique")

    work = g.copy()
    for anchor in anchors:
        r, c = anchor["pos"]
        work[r, c] = BG

    comps = connected_components(work != BG)
    sprites = [extract_sprite(g, comp, anchor_by_color) for comp in comps]

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = BG

    records = [transform_sprite(sprite) for sprite in sprites]
    for rec in sorted(records, key=lambda x: x["anchor"]["order_key"]):
        paste_patch(out, rec["patch"], rec["settled_tl"][0], rec["settled_tl"][1])
        paste_patch(out, rec["copy_patch"], rec["copy_tl"][0], rec["copy_tl"][1])

    for anchor in anchors:
        r, c = anchor["pos"]
        out[r, c] = int(anchor["color"])
    return out


def random_connected_shape(min_area: int = 6, max_area: int = 10) -> set[tuple[int, int]]:
    area = random.randint(min_area, max_area)
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for rr, cc in neighbors4(r, c):
                if (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))
    r0, c0, r1, c1 = bbox_of_coords(list(cells))
    return {(r - r0, c - c0) for r, c in cells}


def make_sprite_patch(
    body_color: int,
    marker_color: int,
    axis: str,
    marker_row_mode: str | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(SPRITE_ATTEMPTS):
        cells = random_connected_shape()
        r0, c0, r1, c1 = bbox_of_coords(list(cells))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h not in (3, 4) or w not in (4, 5):
            continue
        perimeter = []
        for r, c in cells:
            if any((rr, cc) not in cells for rr, cc in neighbors4(r, c)):
                perimeter.append((r, c))
        if not perimeter:
            continue
        candidates = perimeter
        if marker_row_mode is not None:
            target_row = 0 if marker_row_mode == "top" else h - 1
            preferred = [(r, c) for r, c in perimeter if r == target_row]
            if preferred:
                candidates = preferred
        mr, mc = random.choice(candidates)
        patch = np.zeros((h, w), dtype=int)
        for r, c in cells:
            patch[r, c] = body_color
        patch[mr, mc] = marker_color
        patch = crop_patch(patch)
        if patch.shape[0] not in (3, 4) or patch.shape[1] not in (4, 5):
            continue
        marker_locs = np.argwhere(patch == marker_color)
        if len(marker_locs) != 1:
            continue
        if axis == "h" and np.array_equal(patch, np.fliplr(patch)):
            continue
        if axis == "v" and np.array_equal(patch, np.flipud(patch)):
            continue
        marker_rel = tuple(int(x) for x in marker_locs[0])
        return patch, marker_rel
    raise RuntimeError("failed to make sprite patch")


def anchor_neighbor_block(anchor: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = {anchor}
    for rr, cc in neighbors4(anchor[0], anchor[1]):
        if in_bounds(shape, rr, cc):
            out.add((rr, cc))
    return out


def valid_patch_tl(shape: tuple[int, int], patch: np.ndarray, tl: tuple[int, int]) -> bool:
    h, w = patch.shape
    r0, c0 = tl
    return 1 <= r0 and 1 <= c0 and r0 + h - 1 <= shape[0] - 2 and c0 + w - 1 <= shape[1] - 2


def valid_output_tl(shape: tuple[int, int], patch: np.ndarray, tl: tuple[int, int]) -> bool:
    h, w = patch.shape
    r0, c0 = tl
    return 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]


def place_record(
    shape: tuple[int, int],
    edge: str,
    body_color: int,
    marker_color: int,
    sign: int,
    force_corner: bool,
    existing_anchors: set[tuple[int, int]],
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
) -> dict | None:
    axis = "h" if edge in ("top", "bottom") else "v"
    use_corner = axis == "v" and force_corner
    corner_mode = None
    if use_corner:
        corner_mode = "top" if sign < 0 else "bottom"

    for _ in range(SPRITE_ATTEMPTS):
        patch, marker_rel = make_sprite_patch(body_color, marker_color, axis, corner_mode)
        patch_h, patch_w = patch.shape
        if corner_mode == "top" and marker_rel[0] != 0:
            continue
        if corner_mode == "bottom" and marker_rel[0] != patch_h - 1:
            continue

        flipped = np.fliplr(patch) if axis == "h" else np.flipud(patch)
        magnitudes = list(range(3, 6))
        random.shuffle(magnitudes)

        if axis == "h":
            anchor_r = 0 if edge == "top" else shape[0] - 1
            anchor_positions = [(anchor_r, c) for c in range(1, shape[1] - 1)]
            settled_rows = list(range(1, shape[0] - patch_h))
            random.shuffle(anchor_positions)
            random.shuffle(settled_rows)
            for mag in magnitudes:
                for anchor in anchor_positions:
                    if anchor in existing_anchors:
                        continue
                    if any(nb in existing_anchors for nb in neighbors4(*anchor)):
                        continue
                    settled_c = anchor[1] - marker_rel[1]
                    for settled_r in settled_rows:
                        settled_tl = (settled_r, settled_c)
                        copy_tl = (settled_r, settled_c - sign * patch_w)
                        input_tl = (settled_r, settled_c - sign * mag)
                        if not valid_output_tl(shape, patch, settled_tl):
                            continue
                        if not valid_output_tl(shape, flipped, copy_tl):
                            continue
                        if not valid_patch_tl(shape, patch, input_tl):
                            continue
                        input_coords = coord_set(nonzero_cells(patch, input_tl[0], input_tl[1]))
                        output_coords = coord_set(nonzero_cells(patch, settled_tl[0], settled_tl[1]))
                        output_coords |= coord_set(nonzero_cells(flipped, copy_tl[0], copy_tl[1]))
                        anchor_block = anchor_neighbor_block(anchor, shape)
                        if input_coords & input_forbidden:
                            continue
                        if output_coords & output_forbidden:
                            continue
                        if input_coords & anchor_block:
                            continue
                        if output_coords & anchor_block:
                            continue
                        return {
                            "edge": edge,
                            "axis": axis,
                            "anchor": {
                                "pos": anchor,
                                "color": marker_color,
                                "edge": edge,
                                "axis": axis,
                                "order_key": classify_anchor(anchor[0], anchor[1], shape[0], shape[1])[2:],
                            },
                            "patch": patch,
                            "copy_patch": flipped,
                            "marker_rel": marker_rel,
                            "settled_tl": settled_tl,
                            "copy_tl": copy_tl,
                            "input_tl": input_tl,
                            "sign": sign,
                            "mag": mag,
                        }
        else:
            anchor_c = 0 if edge == "left" else shape[1] - 1
            if corner_mode == "top":
                anchor_rows = [0]
            elif corner_mode == "bottom":
                anchor_rows = [shape[0] - 1]
            else:
                anchor_rows = list(range(1, shape[0] - 1))
            settled_cols = list(range(1, shape[1] - patch_w))
            random.shuffle(anchor_rows)
            random.shuffle(settled_cols)
            for mag in magnitudes:
                for anchor_r in anchor_rows:
                    anchor = (anchor_r, anchor_c)
                    if anchor in existing_anchors:
                        continue
                    if any(nb in existing_anchors for nb in neighbors4(*anchor)):
                        continue
                    settled_r = anchor_r - marker_rel[0]
                    for settled_c in settled_cols:
                        settled_tl = (settled_r, settled_c)
                        copy_tl = (settled_r - sign * patch_h, settled_c)
                        input_tl = (settled_r - sign * mag, settled_c)
                        if not valid_output_tl(shape, patch, settled_tl):
                            continue
                        if not valid_output_tl(shape, flipped, copy_tl):
                            continue
                        if not valid_patch_tl(shape, patch, input_tl):
                            continue
                        input_coords = coord_set(nonzero_cells(patch, input_tl[0], input_tl[1]))
                        output_coords = coord_set(nonzero_cells(patch, settled_tl[0], settled_tl[1]))
                        output_coords |= coord_set(nonzero_cells(flipped, copy_tl[0], copy_tl[1]))
                        anchor_block = anchor_neighbor_block(anchor, shape)
                        if input_coords & input_forbidden:
                            continue
                        if output_coords & output_forbidden:
                            continue
                        if input_coords & anchor_block:
                            continue
                        if output_coords & anchor_block:
                            continue
                        return {
                            "edge": edge,
                            "axis": axis,
                            "anchor": {
                                "pos": anchor,
                                "color": marker_color,
                                "edge": edge,
                                "axis": axis,
                                "order_key": classify_anchor(anchor[0], anchor[1], shape[0], shape[1])[2:],
                            },
                            "patch": patch,
                            "copy_patch": flipped,
                            "marker_rel": marker_rel,
                            "settled_tl": settled_tl,
                            "copy_tl": copy_tl,
                            "input_tl": input_tl,
                            "sign": sign,
                            "mag": mag,
                        }
    return None


def build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    n = random.choice([2, 3])
    edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    if n == 3:
        edges.append(random.choice(["top", "bottom", "left", "right"]))
    random.shuffle(edges)

    palette = random.sample(list(range(1, 10)), 2 * n)
    marker_colors = palette[:n]
    body_colors = palette[n:]
    signs = [random.choice([-1, 1]), random.choice([-1, 1])]
    if n == 2 and signs[0] == signs[1]:
        signs[1] *= -1
    if n == 3:
        signs = [1, -1, random.choice([-1, 1])]
        random.shuffle(signs)

    shape = (h, w)
    records = []
    existing_anchors: set[tuple[int, int]] = set()
    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()
    vertical_indices = [i for i, edge in enumerate(edges) if edge in ("left", "right")]
    corner_index = None
    if vertical_indices and random.random() < 0.6:
        corner_index = random.choice(vertical_indices)

    for idx, (edge, marker_color, body_color, sign) in enumerate(zip(edges, marker_colors, body_colors, signs)):
        rec = place_record(
            shape,
            edge,
            body_color,
            marker_color,
            sign,
            idx == corner_index,
            existing_anchors,
            input_forbidden,
            output_forbidden,
        )
        if rec is None:
            return None
        records.append(rec)
        anchor = rec["anchor"]["pos"]
        existing_anchors.add(anchor)
        input_forbidden |= margin8(coord_set(nonzero_cells(rec["patch"], rec["input_tl"][0], rec["input_tl"][1])), shape)
        group_coords = coord_set(nonzero_cells(rec["patch"], rec["settled_tl"][0], rec["settled_tl"][1]))
        group_coords |= coord_set(nonzero_cells(rec["copy_patch"], rec["copy_tl"][0], rec["copy_tl"][1]))
        output_forbidden |= margin8(group_coords, shape)
        input_forbidden |= anchor_neighbor_block(anchor, shape)
        output_forbidden |= anchor_neighbor_block(anchor, shape)

    anchors = [rec["anchor"] for rec in records]
    inp = np.zeros(shape, dtype=int)
    for rec in records:
        paste_patch(inp, rec["patch"], rec["input_tl"][0], rec["input_tl"][1])
    for anchor in anchors:
        r, c = anchor["pos"]
        inp[r, c] = int(anchor["color"])

    expected = render_records(shape, anchors, records)
    if not semantic_checks(inp, expected, records):
        return None
    return inp, expected


def semantic_checks(inp: np.ndarray, expected: np.ndarray, records: list[dict]) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    anchors = detect_anchors(inp)
    if len(anchors) != len(records):
        return False
    anchor_colors = {a["color"] for a in anchors}
    if len(anchor_colors) != len(records):
        return False

    work = inp.copy()
    for anchor in anchors:
        work[anchor["pos"]] = BG
    comps = connected_components(work != BG)
    if len(comps) != len(records):
        return False

    axes = {rec["axis"] for rec in records}
    if len(records) >= 2 and axes != {"h", "v"}:
        return False
    if len({rec["sign"] for rec in records}) < 2:
        return False

    for rec in records:
        patch = rec["patch"]
        flipped = rec["copy_patch"]
        if rec["axis"] == "h":
            if np.array_equal(patch, np.fliplr(patch)):
                return False
            alt_copy_tl = (rec["settled_tl"][0], rec["settled_tl"][1] + rec["sign"] * patch.shape[1])
        else:
            if np.array_equal(patch, np.flipud(patch)):
                return False
            alt_copy_tl = (rec["settled_tl"][0] + rec["sign"] * patch.shape[0], rec["settled_tl"][1])
        alt_records = []
        for other in records:
            if other is rec:
                alt_records.append({**other, "copy_tl": alt_copy_tl})
            else:
                alt_records.append(other)
        if np.array_equal(render_records(inp.shape, [r["anchor"] for r in records], alt_records), expected):
            return False
        alt_records = []
        for other in records:
            if other is rec:
                alt_records.append({**other, "copy_patch": patch})
            else:
                alt_records.append(other)
        if np.array_equal(render_records(inp.shape, [r["anchor"] for r in records], alt_records), expected):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        case = build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
