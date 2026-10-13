import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_component_coords(mask)) == 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each main object by the force-dot count-difference vector around its row and column bands.

    Input structure:
    - The grid has a dominant background, several single-color multi-cell main objects,
      and many single-pixel force dots of one shared color.
    - Each main object has its own row band and column band from its bbox, and every
      force dot lies in exactly one object's row band or column band while staying
      outside that object's bbox.

    Rule:
    - Identify the force-dot color as the unique non-background color whose connected
      components are all single pixels.
    - For each main object with bbox `(r0,c0,r1,c1)`, count `L` force dots in rows
      `r0..r1` and columns `< c0`, `R` in rows `r0..r1` and columns `> c1`,
      `U` in columns `c0..c1` and rows `< r0`, and `D` in columns `c0..c1`
      and rows `> r1`.
    - Remove all force dots, then translate that whole object by `(D-U, R-L)`.

    Key insight:
    - The dots are not destinations; the signed differences between opposite-side
      counts independently encode vertical and horizontal displacement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(colors) >= 2, "expected force dots plus at least one main object"

    force_candidates = []
    for color in colors:
        comps = _component_coords(g == color)
        if comps and all(int(comp.shape[0]) == 1 for comp in comps):
            force_candidates.append(color)
    assert len(force_candidates) == 1, "expected a unique force-dot color"
    force_color = int(force_candidates[0])

    force_coords = np.argwhere(g == force_color)
    object_colors = [color for color in colors if color != force_color]
    assert object_colors, "expected at least one main object"

    objects = []
    for color in object_colors:
        mask = g == color
        coords = np.argwhere(mask)
        assert coords.shape[0] >= 2, "main objects must be multi-cell"
        assert _is_connected(mask), "each main object color must form one 4-connected object"
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox(coords),
            }
        )

    for fr, fc in force_coords:
        matches = 0
        for obj in objects:
            r0, c0, r1, c1 = obj["bbox"]
            in_row_band = r0 <= int(fr) <= r1 and (int(fc) < c0 or int(fc) > c1)
            in_col_band = c0 <= int(fc) <= c1 and (int(fr) < r0 or int(fr) > r1)
            if in_row_band or in_col_band:
                matches += 1
        assert matches == 1, "each force dot must align with exactly one object"

    out = np.full_like(g, bg)
    occupied = np.zeros(g.shape, dtype=bool)
    h, w = g.shape

    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        r0, c0, r1, c1 = obj["bbox"]

        left = int(np.sum((r0 <= force_coords[:, 0]) & (force_coords[:, 0] <= r1) & (force_coords[:, 1] < c0)))
        right = int(np.sum((r0 <= force_coords[:, 0]) & (force_coords[:, 0] <= r1) & (force_coords[:, 1] > c1)))
        up = int(np.sum((c0 <= force_coords[:, 1]) & (force_coords[:, 1] <= c1) & (force_coords[:, 0] < r0)))
        down = int(np.sum((c0 <= force_coords[:, 1]) & (force_coords[:, 1] <= c1) & (force_coords[:, 0] > r1)))

        dr = down - up
        dc = right - left
        shifted = coords + np.array([dr, dc], dtype=int)
        assert np.all(shifted[:, 0] >= 0) and np.all(shifted[:, 0] < h)
        assert np.all(shifted[:, 1] >= 0) and np.all(shifted[:, 1] < w)
        assert not np.any(occupied[shifted[:, 0], shifted[:, 1]]), "translated objects must not overlap"

        occupied[shifted[:, 0], shifted[:, 1]] = True
        out[shifted[:, 0], shifted[:, 1]] = int(obj["color"])

    return out


# --------------- generator ---------------

MAX_DIM = 30
OUTER_MARGIN = 2
ROW_GAP_RANGE = (3, 4)
COL_GAP_RANGE = (3, 4)
OBJ_COUNT_OPTIONS = (2, 3)
OBJ_SIZE_RANGE = (2, 3)
SIDE_MARGIN_RANGE = (1, 2)
MAX_SHIFT = 2
MAX_SIDE_COUNT = 3
MAX_ATTEMPTS = 600


def _removable_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    safe: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        test = mask.copy()
        test[r, c] = False
        if not test[0].any() or not test[-1].any() or not test[:, 0].any() or not test[:, -1].any():
            continue
        if _is_connected(test):
            safe.append((int(r), int(c)))
    return safe


def _gen_sprite(height: int, width: int, color: int, bg: int) -> np.ndarray:
    sprite = np.full((int(height), int(width)), int(color), dtype=int)
    max_remove = max(0, min(2, height * width - 3))
    for _ in range(random.randint(0, max_remove)):
        safe = _removable_cells(sprite == color)
        if not safe:
            break
        rr, cc = random.choice(safe)
        sprite[rr, cc] = int(bg)
    return sprite


def _axis_pairs(neg_cap: int, pos_cap: int) -> list[tuple[int, int]]:
    pairs = []
    for neg in range(min(int(neg_cap), MAX_SIDE_COUNT) + 1):
        for pos in range(min(int(pos_cap), MAX_SIDE_COUNT) + 1):
            if abs(pos - neg) <= MAX_SHIFT:
                pairs.append((neg, pos))
    return pairs


def _weighted_choice(options: list[tuple[int, int]], *, prefer_nonzero: bool) -> tuple[int, int]:
    weights = []
    for neg, pos in options:
        delta = pos - neg
        weight = 1.0
        if delta != 0:
            weight *= 3.0 if prefer_nonzero else 1.2
        else:
            weight *= 1.2 if prefer_nonzero else 2.6
        if neg > 0 and pos > 0:
            weight *= 2.5
        elif neg == 0 and pos == 0:
            weight *= 0.5
        if abs(delta) == MAX_SHIFT:
            weight *= 1.2
        weights.append(weight)
    return random.choices(options, weights=weights, k=1)[0]


def _build_layout(count: int):
    dims = []
    for _ in range(count):
        obj_h = random.randint(*OBJ_SIZE_RANGE)
        obj_w = random.randint(*OBJ_SIZE_RANGE)
        top_pad = random.randint(*SIDE_MARGIN_RANGE)
        bottom_pad = random.randint(*SIDE_MARGIN_RANGE)
        left_pad = random.randint(*SIDE_MARGIN_RANGE)
        right_pad = random.randint(*SIDE_MARGIN_RANGE)
        dims.append(
            {
                "obj_h": obj_h,
                "obj_w": obj_w,
                "top_pad": top_pad,
                "bottom_pad": bottom_pad,
                "left_pad": left_pad,
                "right_pad": right_pad,
                "zone_h": top_pad + obj_h + bottom_pad,
                "zone_w": left_pad + obj_w + right_pad,
            }
        )

    row_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(count - 1)]
    col_gaps = [random.randint(*COL_GAP_RANGE) for _ in range(count - 1)]
    height = OUTER_MARGIN * 2 + sum(d["zone_h"] for d in dims) + sum(row_gaps)
    width = OUTER_MARGIN * 2 + sum(d["zone_w"] for d in dims) + sum(col_gaps)
    if height > MAX_DIM or width > MAX_DIM:
        return None

    row_starts = []
    cur = OUTER_MARGIN
    for i, d in enumerate(dims):
        row_starts.append(cur)
        cur += d["zone_h"]
        if i < count - 1:
            cur += row_gaps[i]

    col_slots = []
    cur = OUTER_MARGIN
    for i, d in enumerate(dims):
        col_slots.append(cur)
        cur += d["zone_w"]
        if i < count - 1:
            cur += col_gaps[i]

    return dims, row_starts, col_slots, (height, width)


def _sample_force_counts(meta: dict, prefer_diagonal: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    h = meta["obj_h"]
    w = meta["obj_w"]
    left_cap = (h * meta["left_pad"] + 1) // 2
    right_cap = (h * meta["right_pad"] + 1) // 2
    up_cap = (w * meta["top_pad"] + 1) // 2
    down_cap = (w * meta["bottom_pad"] + 1) // 2

    h_pairs = _axis_pairs(left_cap, right_cap)
    v_pairs = _axis_pairs(up_cap, down_cap)
    assert h_pairs and v_pairs

    horiz = _weighted_choice(h_pairs, prefer_nonzero=True)
    vert = _weighted_choice(v_pairs, prefer_nonzero=prefer_diagonal)
    if horiz == (0, 0) and vert == (0, 0):
        nonzero_h = [pair for pair in h_pairs if pair[0] != pair[1]]
        nonzero_v = [pair for pair in v_pairs if pair[0] != pair[1]]
        if nonzero_h:
            horiz = random.choice(nonzero_h)
        elif nonzero_v:
            vert = random.choice(nonzero_v)
    return horiz, vert


def _sample_non_touching(cells: list[tuple[int, int]], k: int) -> list[tuple[int, int]] | None:
    need = int(k)
    if need == 0:
        return []
    cells = cells[:]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        r, c = cell
        if any(abs(r - rr) + abs(c - cc) == 1 or (r == rr and c == cc) for rr, cc in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == need:
            return chosen
    return None


def _place_force_dots(inp: np.ndarray, force_color: int, obj: dict, counts: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    zr0, zc0, zr1, zc1 = obj["zone"]

    left_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(zc0, c0)]
    right_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c1 + 1, zc1 + 1)]
    up_cells = [(r, c) for r in range(zr0, r0) for c in range(c0, c1 + 1)]
    down_cells = [(r, c) for r in range(r1 + 1, zr1 + 1) for c in range(c0, c1 + 1)]

    chosen = []
    for cells, amount in (
        (left_cells, counts["left"]),
        (right_cells, counts["right"]),
        (up_cells, counts["up"]),
        (down_cells, counts["down"]),
    ):
        picked = _sample_non_touching(cells, amount)
        if picked is None:
            return False
        chosen.extend(picked)
    for r, c in chosen:
        inp[r, c] = int(force_color)
    return True


def _expected_output(inp: np.ndarray, objects: list[dict], bg: int) -> np.ndarray:
    out = np.full_like(inp, bg)
    for obj in objects:
        dr = obj["counts"]["down"] - obj["counts"]["up"]
        dc = obj["counts"]["right"] - obj["counts"]["left"]
        shifted = obj["coords"] + np.array([dr, dc], dtype=int)
        out[shifted[:, 0], shifted[:, 1]] = int(obj["color"])
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray, objects: list[dict], force_color: int, bg: int) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    vectors = []
    rich_axes = 0
    balanced_axes = 0
    any_horizontal = False
    any_vertical = False
    for obj in objects:
        counts = obj["counts"]
        dr = counts["down"] - counts["up"]
        dc = counts["right"] - counts["left"]
        vectors.append((dr, dc))
        any_horizontal |= dc != 0
        any_vertical |= dr != 0
        for a, b in ((counts["left"], counts["right"]), (counts["up"], counts["down"])):
            if a > 0 and b > 0 and a != b:
                rich_axes += 1
            if a > 0 and a == b:
                balanced_axes += 1

    if not any_horizontal or not any_vertical:
        return False
    if rich_axes == 0 or balanced_axes == 0:
        return False
    if len(set(vectors)) == 1:
        return False
    if int(np.sum(inp == force_color)) < len(objects) * 2:
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(OBJ_COUNT_OPTIONS)
    layout = _build_layout(count)
    if layout is None:
        return None
    dims, row_starts, col_starts, (height, width) = layout

    palette = list(range(10))
    bg = random.choice(palette)
    colors = [color for color in palette if color != bg]
    chosen = random.sample(colors, count + 1)
    force_color = chosen[0]
    object_colors = chosen[1:]

    inp = np.full((height, width), bg, dtype=int)
    objects = []
    sprites_seen = set()

    for idx in range(count):
        meta = dims[idx].copy()
        zone_r0 = row_starts[idx]
        zone_c0 = col_starts[idx]
        zone_r1 = zone_r0 + meta["zone_h"] - 1
        zone_c1 = zone_c0 + meta["zone_w"] - 1

        obj_r0 = zone_r0 + meta["top_pad"]
        obj_c0 = zone_c0 + meta["left_pad"]
        sprite = _gen_sprite(meta["obj_h"], meta["obj_w"], object_colors[idx], bg)
        sprite_sig = sprite.tobytes()
        if sprite_sig in sprites_seen:
            return None
        sprites_seen.add(sprite_sig)
        _paste(inp, sprite, obj_r0, obj_c0, transparent=bg)

        coords = np.argwhere(inp == object_colors[idx])
        coords = coords[(coords[:, 0] >= obj_r0) & (coords[:, 0] < obj_r0 + meta["obj_h"]) & (coords[:, 1] >= obj_c0) & (coords[:, 1] < obj_c0 + meta["obj_w"])]
        bbox = _bbox(coords)
        meta["bbox"] = bbox

        horiz, vert = _sample_force_counts(meta, prefer_diagonal=(idx % 2 == 0))
        counts = {
            "left": int(horiz[0]),
            "right": int(horiz[1]),
            "up": int(vert[0]),
            "down": int(vert[1]),
        }

        objects.append(
            {
                "color": int(object_colors[idx]),
                "coords": coords,
                "bbox": bbox,
                "zone": (zone_r0, zone_c0, zone_r1, zone_c1),
                "counts": counts,
            }
        )

    for obj in objects:
        if not _place_force_dots(inp, force_color, obj, obj["counts"]):
            return None

    out = _expected_output(inp, objects, bg)
    if not _semantic_checks(inp, out, objects, force_color, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/10/048/4")
