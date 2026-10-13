import random

import numpy as np


BG_COLOR = 0
BAR_COLOR = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
SPRITE_SIDE_OPTIONS = (3, 5, 7)
BAR_COUNT_RANGE = (4, 6)
BAR_LENGTHS = (1, 2, 3, 4)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("paste out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return np.count_nonzero(mask) > 0 and len(_components4(mask)) == 1


def _rotate_for_bar(patch: np.ndarray, orientation: str, length: int) -> np.ndarray:
    steps = int(length) - 1
    if orientation == "h":
        return np.rot90(patch, -steps)
    if orientation == "v":
        return np.rot90(patch, steps)
    raise ValueError(f"bad orientation: {orientation}")


def _bar_specs(patch: np.ndarray) -> list[dict]:
    count = random.randint(*BAR_COUNT_RANGE)
    lengths = list(BAR_LENGTHS) + [random.choice(BAR_LENGTHS) for _ in range(count - len(BAR_LENGTHS))]
    random.shuffle(lengths)

    orientations = [random.choice(("h", "v")) for _ in range(count)]
    if len(set(orientations)) == 1:
        flip = random.randrange(count)
        orientations[flip] = "v" if orientations[flip] == "h" else "h"
    nontrivial = [i for i, length in enumerate(lengths) if length > 1]
    if nontrivial and len({orientations[i] for i in nontrivial}) == 1:
        flip = random.choice(nontrivial)
        orientations[flip] = "v" if orientations[flip] == "h" else "h"

    specs = []
    for orientation, length in zip(orientations, lengths):
        rotated = _rotate_for_bar(patch, orientation, length)
        rh, rw = rotated.shape
        row_min = -(rh // 2)
        row_max = max(rh // 2, length - 1 if orientation == "v" else rh // 2)
        col_min = -(rw // 2)
        col_max = max(rw // 2, length - 1 if orientation == "h" else rw // 2)

        zone_min_r = row_min - 1
        zone_max_r = row_max + 1
        zone_min_c = col_min - 1
        zone_max_c = col_max + 1
        anchor_r = -zone_min_r
        anchor_c = -zone_min_c

        specs.append(
            {
                "kind": "bar",
                "orientation": orientation,
                "length": int(length),
                "rotated": rotated,
                "zone_h": int(zone_max_r - zone_min_r + 1),
                "zone_w": int(zone_max_c - zone_min_c + 1),
                "anchor_off": (int(anchor_r), int(anchor_c)),
                "patch_off": (int(anchor_r - rh // 2), int(anchor_c - rw // 2)),
            }
        )
    return specs


def _sample_sprite_mask(h: int, w: int) -> np.ndarray | None:
    area_min = max(5, (h * w + 2) // 3)
    area_max = h * w - 1
    center = (h // 2, w // 2)
    for _ in range(300):
        density = random.uniform(0.38, 0.7)
        mask = np.random.random((h, w)) < density
        mask[center] = True
        mask[0, random.randrange(w)] = True
        mask[h - 1, random.randrange(w)] = True
        mask[random.randrange(h), 0] = True
        mask[random.randrange(h), w - 1] = True
        area = int(np.count_nonzero(mask))
        if not (area_min <= area <= area_max):
            continue
        if not _is_connected(mask):
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    return None


def _sample_sprite_patch() -> np.ndarray:
    for _ in range(200):
        h = random.choice(SPRITE_SIDE_OPTIONS)
        w = random.choice(SPRITE_SIDE_OPTIONS)
        mask = _sample_sprite_mask(h, w)
        if mask is None:
            continue
        color = random.choice(SPRITE_COLORS)
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = color
        return patch
    raise RuntimeError("failed to sample sprite patch")


def _layout_candidates(count: int) -> list[tuple[int, int]]:
    candidates = []
    for rows in (2, 3):
        for cols in (2, 3):
            if rows * cols >= count:
                candidates.append((rows, cols))
    return candidates


def _choose_slots(rows: int, cols: int, count: int) -> list[tuple[int, int]] | None:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(80):
        chosen = random.sample(slots, count)
        if len({r for r, _ in chosen}) == rows and len({c for _, c in chosen}) == cols:
            return chosen
    return None


def _assign_layout(objects: list[dict]) -> tuple[int, int] | None:
    count = len(objects)
    layouts = _layout_candidates(count)
    random.shuffle(layouts)
    for rows, cols in layouts:
        slots = _choose_slots(rows, cols, count)
        if slots is None:
            continue

        order = list(range(count))
        random.shuffle(order)
        placements = [(objects[idx], slot) for idx, slot in zip(order, slots)]

        row_heights = [0] * rows
        col_widths = [0] * cols
        for obj, (r, c) in placements:
            row_heights[r] = max(row_heights[r], int(obj["zone_h"]))
            col_widths[c] = max(col_widths[c], int(obj["zone_w"]))

        row_slack = [random.randint(0, 1) for _ in range(rows)]
        col_slack = [random.randint(0, 1) for _ in range(cols)]
        height = sum(row_heights) + sum(row_slack)
        width = sum(col_widths) + sum(col_slack)
        if height > 30 or width > 30:
            continue

        row_starts = []
        cursor = 0
        for row_h, slack in zip(row_heights, row_slack):
            row_starts.append(cursor)
            cursor += row_h + slack

        col_starts = []
        cursor = 0
        for col_w, slack in zip(col_widths, col_slack):
            col_starts.append(cursor)
            cursor += col_w + slack

        for obj, (r, c) in placements:
            cell_h = row_heights[r] + row_slack[r]
            cell_w = col_widths[c] + col_slack[c]
            spare_r = cell_h - int(obj["zone_h"])
            spare_c = cell_w - int(obj["zone_w"])
            top = row_starts[r] + random.randint(0, spare_r)
            left = col_starts[c] + random.randint(0, spare_c)
            obj["zone_top"] = int(top)
            obj["zone_left"] = int(left)
        return height, width
    return None


def _build_expected(inp: np.ndarray, original: dict, bars: list[dict]) -> np.ndarray:
    out = inp.copy()
    out[out == BAR_COLOR] = BG_COLOR
    _paste(out, original["patch"], original["zone_top"] + 1, original["zone_left"] + 1, transparent=BG_COLOR)
    for bar in bars:
        pr = int(bar["zone_top"] + bar["patch_off"][0])
        pc = int(bar["zone_left"] + bar["patch_off"][1])
        _paste(out, bar["rotated"], pr, pc, transparent=BG_COLOR)
    return out


def _gen_attempt() -> np.ndarray | None:
    patch = _sample_sprite_patch()
    bars = _bar_specs(patch)
    original = {
        "kind": "sprite",
        "patch": patch,
        "zone_h": int(patch.shape[0] + 2),
        "zone_w": int(patch.shape[1] + 2),
    }
    objects = [original] + bars

    shape = _assign_layout(objects)
    if shape is None:
        return None
    height, width = shape

    inp = np.zeros((height, width), dtype=int)
    _paste(inp, patch, original["zone_top"] + 1, original["zone_left"] + 1, transparent=BG_COLOR)

    for bar in bars:
        ar = int(bar["zone_top"] + bar["anchor_off"][0])
        ac = int(bar["zone_left"] + bar["anchor_off"][1])
        if bar["orientation"] == "h":
            for delta in range(int(bar["length"])):
                inp[ar, ac + delta] = BAR_COLOR
        else:
            for delta in range(int(bar["length"])):
                inp[ar + delta, ac] = BAR_COLOR

    expected = _build_expected(inp, original, bars)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the unique colored sprite and stamp it onto each gray bar according to that bar's length and orientation.

    Input structure:
    - The background is black.
    - One non-gray color forms a single odd-by-odd sprite bbox, possibly with black holes inside it.
    - The other foreground objects are separated gray straight bars of length 1 to 4, each horizontal or vertical.

    Rule:
    - Keep the original sprite unchanged.
    - For a gray bar of length k, rotate the sprite bbox by (k-1)*90 degrees: clockwise for horizontal bars and counterclockwise for vertical bars.
    - Stamp the rotated bbox transparently so its bbox center lands on the bar's uppermost-leftmost endpoint, then erase the whole gray bar.

    Key insight:
    - The gray bars are rotation dials: length selects the quarter-turn count, and orientation selects whether the turn is clockwise or counterclockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprite_colors = sorted(set(map(int, np.unique(g))) - {BG_COLOR, BAR_COLOR})
    assert len(sprite_colors) == 1
    sprite_color = int(sprite_colors[0])

    sprite_mask = g == sprite_color
    sprite_bbox = _bbox(sprite_mask)
    assert sprite_bbox is not None
    r0, c0, r1, c1 = sprite_bbox
    assert (r1 - r0 + 1) % 2 == 1 and (c1 - c0 + 1) % 2 == 1

    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert set(map(int, np.unique(patch))) <= {BG_COLOR, sprite_color}

    out = g.copy()
    out[out == BAR_COLOR] = BG_COLOR

    bars = []
    for comp in _components4(g == BAR_COLOR):
        coords = np.array(comp, dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        length = int(len(comp))
        if int(rows.min()) == int(rows.max()):
            assert int(cols.max()) - int(cols.min()) + 1 == length
            orientation = "h"
        elif int(cols.min()) == int(cols.max()):
            assert int(rows.max()) - int(rows.min()) + 1 == length
            orientation = "v"
        else:
            raise AssertionError("gray component must be a straight bar")
        anchor = (int(rows.min()), int(cols.min()))
        bars.append((anchor, orientation, length))

    bars.sort(key=lambda item: item[0])
    for (anchor_r, anchor_c), orientation, length in bars:
        rotated = _rotate_for_bar(patch, orientation, length)
        pr = int(anchor_r - rotated.shape[0] // 2)
        pc = int(anchor_c - rotated.shape[1] // 2)
        _paste(out, rotated, pr, pc, transparent=BG_COLOR)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
