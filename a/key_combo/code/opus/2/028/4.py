import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

REGION_COLORS = (BLUE, GREEN, YELLOW, TEAL)
GEN_RETRIES = 400
COUNT_OPTIONS = {
    "first_more": ((2, 1), (3, 1), (3, 2), (4, 1)),
    "second_more": ((1, 2), (1, 3), (2, 3), (1, 4)),
    "tie": ((1, 1), (2, 2)),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
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
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_input_regions(grid: np.ndarray) -> list[dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    regions: list[dict[str, Any]] = []
    for coords in _components(g != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == BLACK):
            raise AssertionError("every foreground component must be a filled rectangle")
        colors = set(map(int, np.unique(patch)))
        base_colors = sorted(colors - {BLACK, RED, GREY})
        if len(base_colors) != 1:
            raise AssertionError("each region must have exactly one base color")
        markers = np.argwhere(patch == GREY)
        if len(markers) > 1:
            raise AssertionError("input regions may contain at most one grey marker")
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch.copy(),
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "base_color": int(base_colors[0]),
                "marker": None if len(markers) == 0 else (int(markers[0, 0]), int(markers[0, 1])),
                "reds": [(int(r), int(c)) for r, c in np.argwhere(patch == RED)],
            }
        )
    return regions


def _marked_stats(region: dict[str, Any]) -> dict[str, Any]:
    marker = region["marker"]
    if marker is None:
        raise AssertionError("expected a marked region")
    h = int(region["height"])
    w = int(region["width"])
    if h == w:
        raise AssertionError("regions must be non-square")
    mr, mc = marker
    reds = region["reds"]
    if w > h:
        first_count = sum(c < mc for _, c in reds)
        second_count = sum(c > mc for _, c in reds)
        if any(c == mc for _, c in reds):
            raise AssertionError("red dots cannot lie on the vertical divider")
        return {
            "orientation": "wide",
            "split_index": int(mc),
            "first_count": int(first_count),
            "second_count": int(second_count),
            "first_span": int(mc),
            "second_span": int(w - mc - 1),
        }
    first_count = sum(r < mr for r, _ in reds)
    second_count = sum(r > mr for r, _ in reds)
    if any(r == mr for r, _ in reds):
        raise AssertionError("red dots cannot lie on the horizontal divider")
    return {
        "orientation": "tall",
        "split_index": int(mr),
        "first_count": int(first_count),
        "second_count": int(second_count),
        "first_span": int(mr),
        "second_span": int(h - mr - 1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each marked rectangle along its short-axis divider and recolor the two sides by red-dot counts.

    Input structure:
    - The grid contains several non-overlapping non-square rectangular regions on a black background.
    - Each region is otherwise monochrome in blue, green, yellow, or teal.
    - Some regions contain exactly one grey marker and several red dots; unmarked regions contain no extra colors.

    Rule:
    - For every marked region, use the region's aspect ratio to choose the divider direction.
    - Wide regions get a full-height grey vertical line through the marker column; tall regions get a full-width grey horizontal line through the marker row.
    - Count red dots on the two sides of that divider, recolor the side with more dots orange and the side with fewer dots pink, erase the red dots into their side color, and keep ties in the original region color; leave unmarked regions unchanged.

    Key insight:
    - The marker does not decide the divider direction by itself: the rectangle's own wide-vs-tall shape fixes the cut, which changes the counted subregions and therefore the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    out = g.copy()
    for region in _extract_input_regions(g):
        marker = region["marker"]
        if marker is None:
            continue

        stats = _marked_stats(region)
        base = int(region["base_color"])
        h = int(region["height"])
        w = int(region["width"])
        first_count = int(stats["first_count"])
        second_count = int(stats["second_count"])
        if first_count == second_count:
            first_color = second_color = base
        elif first_count > second_count:
            first_color, second_color = ORANGE, PINK
        else:
            first_color, second_color = PINK, ORANGE

        patch = np.empty((h, w), dtype=int)
        if stats["orientation"] == "wide":
            split = int(stats["split_index"])
            patch[:, :split] = first_color
            patch[:, split + 1 :] = second_color
            patch[:, split] = GREY
        else:
            split = int(stats["split_index"])
            patch[:split, :] = first_color
            patch[split + 1 :, :] = second_color
            patch[split, :] = GREY

        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _valid_dims(slot_h: int, slot_w: int, orientation: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if orientation == "wide":
        for h in range(4, min(6, slot_h) + 1):
            for w in range(max(6, h + 1), slot_w + 1):
                pairs.append((h, w))
    else:
        for h in range(6, slot_h + 1):
            for w in range(4, min(6, slot_w) + 1):
                if h > w:
                    pairs.append((h, w))
    return pairs


def _pick_marker_index(length: int, relation: str, prefer_smaller_more: bool | None) -> int:
    choices: list[int] = []
    for split in range(1, length - 1):
        first_span = split
        second_span = length - split - 1
        if relation == "tie":
            if first_span != second_span:
                choices.append(split)
            continue
        more_first = relation == "first_more"
        more_span = first_span if more_first else second_span
        less_span = second_span if more_first else first_span
        if prefer_smaller_more is True and more_span >= less_span:
            continue
        if prefer_smaller_more is False and more_span <= less_span:
            continue
        choices.append(split)
    if not choices:
        raise AssertionError("no valid marker position for requested relation")
    return int(random.choice(choices))


def _sample_side_cells(
    h: int, w: int, orientation: str, split: int, first_count: int, second_count: int
) -> list[tuple[int, int]]:
    first_cells: list[tuple[int, int]] = []
    second_cells: list[tuple[int, int]] = []
    if orientation == "wide":
        for r in range(h):
            for c in range(w):
                if c < split:
                    first_cells.append((r, c))
                elif c > split:
                    second_cells.append((r, c))
    else:
        for r in range(h):
            for c in range(w):
                if r < split:
                    first_cells.append((r, c))
                elif r > split:
                    second_cells.append((r, c))
    if len(first_cells) < first_count or len(second_cells) < second_count:
        raise AssertionError("not enough room for requested red dots")
    return random.sample(first_cells, first_count) + random.sample(second_cells, second_count)


def _render_marked_region(
    h: int,
    w: int,
    base_color: int,
    orientation: str,
    relation: str,
    prefer_smaller_more: bool | None,
) -> tuple[np.ndarray, np.ndarray]:
    input_patch = np.full((h, w), base_color, dtype=int)
    first_count, second_count = random.choice(COUNT_OPTIONS[relation])
    if orientation == "wide":
        split = _pick_marker_index(w, relation, prefer_smaller_more)
        marker = (random.randint(1, h - 2), split)
    else:
        split = _pick_marker_index(h, relation, prefer_smaller_more)
        marker = (split, random.randint(1, w - 2))
    reds = _sample_side_cells(h, w, orientation, split, first_count, second_count)
    for r, c in reds:
        input_patch[r, c] = RED
    input_patch[marker] = GREY

    expected_patch = np.full((h, w), base_color, dtype=int)
    if first_count == second_count:
        first_color = second_color = base_color
    elif first_count > second_count:
        first_color, second_color = ORANGE, PINK
    else:
        first_color, second_color = PINK, ORANGE

    if orientation == "wide":
        expected_patch[:, :split] = first_color
        expected_patch[:, split + 1 :] = second_color
        expected_patch[:, split] = GREY
    else:
        expected_patch[:split, :] = first_color
        expected_patch[split + 1 :, :] = second_color
        expected_patch[split, :] = GREY
    return input_patch, expected_patch


def _slot_layout(region_count: int) -> tuple[int, int]:
    if region_count == 5:
        return random.choice(((2, 3), (3, 2)))
    return random.choice(((2, 3), (3, 2)))


def _slot_size(count: int, axis: str) -> int:
    if axis == "row":
        return random.randint(7, 10) if count == 2 else random.randint(7, 8)
    return random.randint(7, 10) if count == 2 else random.randint(7, 8)


def _anti_shortcut_output(grid: np.ndarray, mode: str) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for region in _extract_input_regions(grid):
        marker = region["marker"]
        if marker is None:
            continue
        stats = _marked_stats(region)
        base = int(region["base_color"])
        h = int(region["height"])
        w = int(region["width"])

        if mode == "always_first":
            first_color = ORANGE
            second_color = PINK
        else:
            first_span = int(stats["first_span"])
            second_span = int(stats["second_span"])
            if first_span == second_span:
                first_color = second_color = base
            elif first_span > second_span:
                first_color, second_color = ORANGE, PINK
            else:
                first_color, second_color = PINK, ORANGE

        if stats["first_count"] == stats["second_count"]:
            first_color = second_color = base

        patch = np.empty((h, w), dtype=int)
        if stats["orientation"] == "wide":
            split = int(stats["split_index"])
            patch[:, :split] = first_color
            patch[:, split + 1 :] = second_color
            patch[:, split] = GREY
        else:
            split = int(stats["split_index"])
            patch[:split, :] = first_color
            patch[split + 1 :, :] = second_color
            patch[split, :] = GREY

        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        regions = _extract_input_regions(inp)
    except AssertionError:
        return False
    if len(regions) not in {5, 6}:
        return False

    marked = [region for region in regions if region["marker"] is not None]
    unmarked = [region for region in regions if region["marker"] is None]
    if len(unmarked) < 1 or len(marked) < 4:
        return False

    orientations = {"wide": 0, "tall": 0}
    relations: set[str] = set()
    smaller_more = False
    larger_more = False
    tie_with_unequal_sizes = False
    changed = 0

    for region in marked:
        stats = _marked_stats(region)
        relation = "tie"
        if stats["first_count"] > stats["second_count"]:
            relation = "first_more"
        elif stats["first_count"] < stats["second_count"]:
            relation = "second_more"
        relations.add(relation)
        orientations[stats["orientation"]] += 1

        total_red = int(stats["first_count"] + stats["second_count"])
        if total_red < 2 or total_red > 5:
            return False
        if min(stats["first_count"], stats["second_count"]) < 1:
            return False

        first_span = int(stats["first_span"])
        second_span = int(stats["second_span"])
        if relation == "tie":
            if first_span != second_span:
                tie_with_unequal_sizes = True
        else:
            more_span = first_span if relation == "first_more" else second_span
            less_span = second_span if relation == "first_more" else first_span
            if more_span < less_span:
                smaller_more = True
            if more_span > less_span:
                larger_more = True

        r0, c0, r1, c1 = region["bbox"]
        before = inp[r0 : r1 + 1, c0 : c1 + 1]
        after = expected[r0 : r1 + 1, c0 : c1 + 1]
        changed += int(np.sum(before != after))

    if orientations["wide"] < 2 or orientations["tall"] < 2:
        return False
    if not {"first_more", "second_more"} <= relations:
        return False
    if len(regions) == 6 and "tie" not in relations:
        return False
    if len(regions) == 6 and not tie_with_unequal_sizes:
        return False
    if not (smaller_more and larger_more):
        return False
    if changed < 20:
        return False
    if np.array_equal(expected, _anti_shortcut_output(inp, "always_first")):
        return False
    if np.array_equal(expected, _anti_shortcut_output(inp, "larger_side")):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    region_count = random.choices((5, 6), weights=(2, 3), k=1)[0]
    slot_rows, slot_cols = _slot_layout(region_count)
    slot_h = _slot_size(slot_rows, "row")
    slot_w = _slot_size(slot_cols, "col")
    grid_h = slot_rows * slot_h + slot_rows + 1
    grid_w = slot_cols * slot_w + slot_cols + 1

    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    active_slots = random.sample(slots, region_count)
    roles: list[tuple[str, str | None] | None] = [
        ("wide", "first_more"),
        ("wide", "second_more"),
        ("tall", "first_more"),
        ("tall", "second_more"),
        None,
    ]
    if region_count == 6:
        roles.append((random.choice(("wide", "tall")), "tie"))
    random.shuffle(roles)

    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    expected = np.full((grid_h, grid_w), BLACK, dtype=int)

    for (sr, sc), role in zip(active_slots, roles):
        slot_r0 = 1 + sr * (slot_h + 1)
        slot_c0 = 1 + sc * (slot_w + 1)

        if role is None:
            orientation = random.choice(("wide", "tall"))
            relation = None
        else:
            orientation, relation = role

        dims = _valid_dims(slot_h, slot_w, orientation)
        if not dims:
            return None
        rect_h, rect_w = random.choice(dims)
        top = slot_r0 + random.randint(0, slot_h - rect_h)
        left = slot_c0 + random.randint(0, slot_w - rect_w)
        color = random.choice(REGION_COLORS)

        if relation is None:
            patch_in = patch_out = np.full((rect_h, rect_w), color, dtype=int)
        else:
            prefer_smaller_more: bool | None
            if relation == "tie":
                prefer_smaller_more = None
            else:
                prefer_smaller_more = bool(random.getrandbits(1))
            try:
                patch_in, patch_out = _render_marked_region(
                    rect_h, rect_w, color, orientation, relation, prefer_smaller_more
                )
            except AssertionError:
                return None

        inp[top : top + rect_h, left : left + rect_w] = patch_in
        expected[top : top + rect_h, left : left + rect_w] = patch_out

    rot_k = random.randint(0, 3)
    if rot_k:
        inp = np.rot90(inp, k=rot_k).copy()
        expected = np.rot90(expected, k=rot_k).copy()

    add_h = random.randint(0, min(2, 30 - inp.shape[0]))
    add_w = random.randint(0, min(2, 30 - inp.shape[1]))
    if add_h or add_w:
        padded_in = np.full((inp.shape[0] + add_h, inp.shape[1] + add_w), BLACK, dtype=int)
        padded_out = np.full_like(padded_in, BLACK)
        off_r = random.randint(0, add_h)
        off_c = random.randint(0, add_w)
        padded_in[off_r : off_r + inp.shape[0], off_c : off_c + inp.shape[1]] = inp
        padded_out[off_r : off_r + expected.shape[0], off_c : off_c + expected.shape[1]] = expected
        inp, expected = padded_in, padded_out

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
