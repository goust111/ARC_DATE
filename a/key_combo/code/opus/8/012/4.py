import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

OBJECT_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
BRACKET_WIDTH = 4
GEN_RETRY_LIMIT = 500


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    """Return 4-connected components for one color as (N,2) coord arrays."""
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    mm = np.asarray(mask, dtype=np.uint8)
    return mm.shape, tuple(map(int, mm.ravel()))


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for base in (np.asarray(mask, dtype=bool), np.fliplr(np.asarray(mask, dtype=bool))):
        for k in range(4):
            var = np.rot90(base, k)
            key = _mask_key(var)
            if key not in seen:
                seen.add(key)
                variants.append(np.ascontiguousarray(var, dtype=bool))
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_mask_key(var) for var in _d4_variants(mask))


def _is_connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_l_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return False
    cells = set(map(tuple, coords))
    expected = {(r, c1) for r in range(r0, r1 + 1)}
    expected |= {(r1, c) for c in range(c0, c1 + 1)}
    return cells == expected


def _is_left_open_bracket(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    cells = set(map(tuple, coords))
    expected = {(r0, c) for c in range(c0, c1 + 1)}
    expected |= {(r1, c) for c in range(c0, c1 + 1)}
    expected |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells == expected


def _parse_frame_and_template(grid: np.ndarray) -> tuple[dict, np.ndarray]:
    grey = [coords for coords in _components_of_color(grid, GREY) if _is_l_frame(coords)]
    assert len(grey) == 1, "expected exactly one grey L-frame"
    frame_coords = grey[0]
    fr0, fc0, fr1, fc1 = _bbox_coords(frame_coords)
    candidates = []
    for coords in _components_of_color(grid, RED):
        r0, c0, r1, c1 = _bbox_coords(coords)
        if fr0 <= r0 and r1 < fr1 and fc0 <= c0 and c1 < fc1:
            candidates.append(coords)
    assert len(candidates) == 1, "expected exactly one red template inside the frame"
    template = _crop_mask(candidates[0])
    assert 3 <= int(template.sum()) <= 6, "template area must be 3..6"
    return {"coords": frame_coords, "bbox": (fr0, fc0, fr1, fc1)}, template


def _parse_brackets(grid: np.ndarray) -> list[dict]:
    brackets = []
    for coords in _components_of_color(grid, PINK):
        if _is_left_open_bracket(coords):
            r0, c0, r1, c1 = _bbox_coords(coords)
            brackets.append({"coords": coords, "bbox": (r0, c0, r1, c1)})
    brackets.sort(key=lambda item: item["bbox"])
    assert 2 <= len(brackets) <= 3, "expected 2 or 3 bracket channels"
    for i in range(len(brackets) - 1):
        assert brackets[i + 1]["bbox"][0] - brackets[i]["bbox"][2] >= 3, "brackets must be vertically separated"
    return brackets


def _split_body_and_rod(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    max_c = local.shape[1] - 1
    candidates = []
    for row in range(local.shape[0]):
        if not local[row, max_c]:
            continue
        cols = np.flatnonzero(local[row])
        for start in cols:
            rod_len = max_c - start + 1
            if not 1 <= rod_len <= 4:
                continue
            rod = np.zeros_like(local, dtype=bool)
            rod[row, start : max_c + 1] = True
            if np.any(local[:, start:] & ~rod[:, start:]):
                continue
            if not local[row, start - 1]:
                continue
            vertical_ok = True
            for cc in range(start, max_c + 1):
                if row > 0 and local[row - 1, cc]:
                    vertical_ok = False
                if row + 1 < local.shape[0] and local[row + 1, cc]:
                    vertical_ok = False
            if not vertical_ok:
                continue
            body = local & ~rod
            area = int(body.sum())
            if not 3 <= area <= 6:
                continue
            if not _is_connected_mask(body):
                continue
            body_coords_local = np.argwhere(body)
            candidates.append(
                {
                    "body_coords": body_coords_local + np.array([[r0, c0]]),
                    "rod_coords": np.argwhere(rod) + np.array([[r0, c0]]),
                    "body_mask": body,
                    "rod_len": rod_len,
                    "attach_row": r0 + row,
                    "body_bbox": _bbox_coords(body_coords_local + np.array([[r0, c0]])),
                }
            )
    assert candidates, "each middle component must admit a body-plus-rod split"
    max_len = max(candidate["rod_len"] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate["rod_len"] == max_len]
    assert len(best) == 1, "each middle component must have a unique longest rightward rod"
    body_mask = _crop_mask(best[0]["body_coords"])
    best[0]["body_mask"] = body_mask
    best[0]["shape_key"] = _canonical_shape_key(body_mask)
    return best[0]


def _parse_movers(grid: np.ndarray) -> list[dict]:
    movers = []
    for color in OBJECT_COLORS:
        comps = _components_of_color(grid, color)
        assert len(comps) <= 1, "each object color must appear in at most one component"
        for coords in comps:
            info = _split_body_and_rod(coords)
            info["color"] = color
            info["component_coords"] = coords
            movers.append(info)
    assert 3 <= len(movers) <= 4, "expected 3 or 4 movable objects"
    return movers


def _channel_for_row(brackets: list[dict], row: int) -> dict:
    hits = []
    for bracket in brackets:
        r0, _c0, r1, _c1 = bracket["bbox"]
        if r0 < row < r1:
            hits.append(bracket)
    assert len(hits) == 1, "each rod extension row must target exactly one bracket"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match template-shaped bodies, slide the matches into right-side bracket channels, and erase the nonmatches.

    Input structure:
    - Black background contains one grey L-frame in the upper-left with a red 4-connected template inside it.
    - The middle region contains 3 or 4 monochrome 4-connected bodies, each followed immediately by a same-color horizontal rod to its right.
    - The right side contains 2 or 3 magenta left-open bracket channels with disjoint vertical spans.

    Rule:
    - Extract the red template mask from inside the grey frame and compare each body mask against all 8 D4 transforms of the template.
    - For a nonmatching body, erase both the body and its rod to black.
    - For a matching body, erase its source component, recolor the body grey, and shift it horizontally by rod_length + 1 so its right edge lands on the targeted bracket's inner left column.
    - The target bracket is the unique bracket whose interior row span contains the rod's extension row; the frame, template, and brackets stay unchanged.

    Key insight:
    - The grey frame only defines the template shape; the rod is ignored for matching but is essential for deciding how far the surviving body slides into its bracket lane.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _frame, template = _parse_frame_and_template(g)
    template_keys = {_mask_key(var) for var in _d4_variants(template)}
    brackets = _parse_brackets(g)
    movers = _parse_movers(g)

    out = g.copy()
    for mover in movers:
        comp = mover["component_coords"]
        out[comp[:, 0], comp[:, 1]] = BG

    match_count = 0
    used_rows = set()
    for mover in movers:
        is_match = _mask_key(mover["body_mask"]) in template_keys
        if not is_match:
            continue
        match_count += 1
        bracket = _channel_for_row(brackets, mover["attach_row"])
        _br0, bc0, _br1, _bc1 = bracket["bbox"]
        _r0, _c0, _r1, body_c1 = mover["body_bbox"]
        shift = mover["rod_len"] + 1
        assert body_c1 + shift == bc0, "matching body must land against its bracket's left interior edge"
        dest = mover["body_coords"].copy()
        dest[:, 1] += shift
        assert not any((int(r), int(c)) in used_rows for r, c in dest), "matches must not overlap each other"
        assert np.all(out[dest[:, 0], dest[:, 1]] == BG), "moved body must land on background"
        used_rows |= set(map(tuple, dest))
        out[dest[:, 0], dest[:, 1]] = GREY

    assert 1 <= match_count <= 2, "expected 1 or 2 template matches"
    return out


_BASE_SHAPES = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
]

_SHAPE_CLASSES: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
_SHAPES_BY_AREA: dict[int, list[np.ndarray]] = defaultdict(list)
for _mask in _BASE_SHAPES:
    _key = _canonical_shape_key(_mask)
    if _key not in _SHAPE_CLASSES:
        _SHAPE_CLASSES[_key] = _mask
for _mask in _SHAPE_CLASSES.values():
    _SHAPES_BY_AREA[int(_mask.sum())].append(_mask)


def _rand_variant(mask: np.ndarray, require_different_key: tuple[tuple[int, int], tuple[int, ...]] | None = None) -> np.ndarray:
    variants = _d4_variants(mask)
    if require_different_key is not None:
        choices = [var for var in variants if _mask_key(var) != require_different_key]
        if choices:
            variants = choices
    return np.asarray(random.choice(variants), dtype=bool)


def _attach_rows(mask: np.ndarray) -> list[int]:
    right = np.max(np.argwhere(mask)[:, 1])
    return [r for r in range(mask.shape[0]) if mask[r, right]]


def _draw_l_frame(grid: np.ndarray, top: int, left: int, h: int, w: int) -> tuple[int, int, int, int]:
    grid[top : top + h, left + w - 1] = GREY
    grid[top + h - 1, left : left + w] = GREY
    return top, left, top + h - 1, left + w - 1


def _draw_bracket(grid: np.ndarray, top: int, left: int, h: int) -> tuple[int, int, int, int]:
    right = left + BRACKET_WIDTH - 1
    grid[top, left : right + 1] = PINK
    grid[top + h - 1, left : right + 1] = PINK
    grid[top : top + h, right] = PINK
    return top, left, top + h - 1, right


def _place_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _neighborhood(cells: list[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _object_cells(top: int, left: int, body: np.ndarray, rod_len: int, attach_row: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    body_cells = [(top + r, left + c) for r, c in np.argwhere(body)]
    rod_start = left + body.shape[1]
    rod_cells = [(top + attach_row, rod_start + i) for i in range(rod_len)]
    return body_cells, rod_cells


def _matchable_nonmatch_choices(template_mask: np.ndarray) -> list[np.ndarray]:
    template_area = int(template_mask.sum())
    template_key = _canonical_shape_key(template_mask)
    same_area = [mask for mask in _SHAPES_BY_AREA[template_area] if _canonical_shape_key(mask) != template_key]
    others = [mask for mask in _SHAPE_CLASSES.values() if _canonical_shape_key(mask) != template_key]
    return same_area or others


def _build_spec() -> dict | None:
    height = random.randint(20, 28)
    width = random.randint(22, 28)
    grid = np.zeros((height, width), dtype=int)

    frame_h = random.randint(5, 7)
    frame_w = random.randint(5, 7)
    frame_top = 1
    frame_left = 1
    frame_bbox = _draw_l_frame(grid, frame_top, frame_left, frame_h, frame_w)

    template_base = random.choice([mask for mask in _SHAPE_CLASSES.values() if len(_d4_variants(mask)) > 1])
    template_mask = _rand_variant(template_base)
    th, tw = template_mask.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    if th > frame_h - 1 or tw > frame_w - 1:
        return None
    ttop = random.randint(fr0, fr1 - th)
    tleft = random.randint(fc0, fc1 - tw)
    template_cells = [(ttop + r, tleft + c) for r, c in np.argwhere(template_mask)]
    _place_cells(grid, template_cells, RED)

    bracket_count = random.choice((2, 3))
    bracket_heights = [random.randint(5, 7) for _ in range(bracket_count)]
    min_total = sum(bracket_heights) + 2 * (bracket_count - 1)
    if min_total + 2 > height:
        return None
    slack = height - 2 - min_total
    gap_extras = [0] * (bracket_count - 1)
    top_extra = random.randint(0, slack)
    slack -= top_extra
    bottom_extra = random.randint(0, slack)
    slack -= bottom_extra
    for i in range(len(gap_extras)):
        extra = random.randint(0, slack)
        gap_extras[i] = extra
        slack -= extra
    random.shuffle(gap_extras)
    bracket_left = width - BRACKET_WIDTH
    brackets = []
    cursor = 1 + top_extra
    for i, bh in enumerate(bracket_heights):
        brackets.append({"bbox": _draw_bracket(grid, cursor, bracket_left, bh)})
        cursor += bh
        if i + 1 < bracket_count:
            cursor += 2 + gap_extras[i]

    object_count = random.choice((3, 4))
    match_count = random.randint(1, min(2, bracket_count, object_count - 1))
    colors = random.sample(list(OBJECT_COLORS), object_count)
    matched_brackets = random.sample(range(bracket_count), match_count)
    template_key = _canonical_shape_key(template_mask)
    nonmatch_pool = _matchable_nonmatch_choices(template_mask)

    plans = []
    for i in range(match_count):
        require_diff = _mask_key(template_mask) if i == 0 else None
        plans.append(
            {
                "color": colors[i],
                "base": template_base,
                "force_not_template_orientation": require_diff,
                "bracket_idx": matched_brackets[i],
                "is_match": True,
            }
        )

    remaining = object_count - match_count
    special_nonmatch = random.choice(nonmatch_pool)
    plans.append(
        {
            "color": colors[match_count],
            "base": special_nonmatch,
            "force_not_template_orientation": None,
            "bracket_idx": random.randrange(bracket_count),
            "is_match": False,
        }
    )
    for i in range(1, remaining):
        plans.append(
            {
                "color": colors[match_count + i],
                "base": random.choice(nonmatch_pool),
                "force_not_template_orientation": None,
                "bracket_idx": random.randrange(bracket_count),
                "is_match": False,
            }
        )

    occupied = set(map(tuple, np.argwhere(grid != BG)))
    object_padding = set()
    placed = []
    plans.sort(key=lambda item: (not item["is_match"], -int(item["base"].sum())))
    for plan in plans:
        placed_here = False
        for _ in range(80):
            body = _rand_variant(plan["base"], require_different_key=plan["force_not_template_orientation"])
            attach_options = _attach_rows(body)
            if not attach_options:
                continue
            attach_row = random.choice(attach_options)
            rod_len = random.randint(1, 4)
            bracket_idx = plan["bracket_idx"]
            br0, bc0, br1, _bc1 = brackets[bracket_idx]["bbox"]
            body_h, body_w = body.shape
            top_choices = list(range(br0 + 1, br1 - body_h + 1))
            if not top_choices:
                continue
            random.shuffle(top_choices)
            body_right = bc0 - rod_len - 1
            left = body_right - body_w + 1
            if left <= fc1 + 2:
                continue
            for top in top_choices:
                body_cells, rod_cells = _object_cells(top, left, body, rod_len, attach_row)
                all_cells = body_cells + rod_cells
                if any((r, c) in occupied for r, c in all_cells):
                    continue
                if any((r, c) in object_padding for r, c in all_cells):
                    continue
                if any(not (0 <= r < height and 0 <= c < width) for r, c in all_cells):
                    continue
                _place_cells(grid, all_cells, plan["color"])
                occupied.update(all_cells)
                object_padding |= _neighborhood(all_cells, height, width, pad=1)
                placed.append(
                    {
                        "color": plan["color"],
                        "body": body,
                        "attach_row": attach_row,
                        "rod_len": rod_len,
                        "top": top,
                        "left": left,
                        "bracket_idx": bracket_idx,
                        "is_match": plan["is_match"],
                        "shape_key": _canonical_shape_key(body),
                        "template_key": template_key,
                        "body_cells": body_cells,
                        "rod_cells": rod_cells,
                    }
                )
                placed_here = True
                break
            if placed_here:
                break
        if not placed_here:
            return None

    if sum(int(spec["is_match"]) for spec in placed) != match_count:
        return None
    if len({spec["shape_key"] for spec in placed if spec["is_match"]}) != 1:
        return None
    if any(spec["shape_key"] == template_key for spec in placed if not spec["is_match"]):
        return None
    if not any(int(spec["body"].sum()) == int(template_mask.sum()) for spec in placed if not spec["is_match"]):
        return None
    if match_count > 1 and len({spec["bracket_idx"] for spec in placed if spec["is_match"]}) != match_count:
        return None

    return {
        "grid": grid,
        "template_mask": template_mask,
        "template_key": template_key,
        "frame_bbox": frame_bbox,
        "brackets": brackets,
        "objects": placed,
    }


def _expected_output_from_spec(spec: dict) -> np.ndarray:
    out = spec["grid"].copy()
    for obj in spec["objects"]:
        for r, c in obj["body_cells"] + obj["rod_cells"]:
            out[r, c] = BG
    for obj in spec["objects"]:
        if not obj["is_match"]:
            continue
        shifted = [(r, c + obj["rod_len"] + 1) for r, c in obj["body_cells"]]
        for r, c in shifted:
            out[r, c] = GREY
    return out


def _semantic_check(spec: dict) -> bool:
    grid = spec["grid"]
    parsed_frame, parsed_template = _parse_frame_and_template(grid)
    if _canonical_shape_key(parsed_template) != spec["template_key"]:
        return False
    _ = parsed_frame
    brackets = _parse_brackets(grid)
    movers = _parse_movers(grid)
    if len(brackets) != len(spec["brackets"]) or len(movers) != len(spec["objects"]):
        return False
    template_keys = {_mask_key(var) for var in _d4_variants(parsed_template)}
    matchers = [m for m in movers if _mask_key(m["body_mask"]) in template_keys]
    nonmatchers = [m for m in movers if _mask_key(m["body_mask"]) not in template_keys]
    if not (1 <= len(matchers) <= 2):
        return False
    if len(nonmatchers) == 0:
        return False
    if not any(int(m["body_mask"].sum()) == int(parsed_template.sum()) for m in nonmatchers):
        return False
    for mover in matchers:
        bracket = _channel_for_row(brackets, mover["attach_row"])
        if mover["body_bbox"][3] + mover["rod_len"] + 1 != bracket["bbox"][1]:
            return False
    return np.array_equal(main(grid), _expected_output_from_spec(spec))


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        spec = _build_spec()
        if spec is None:
            continue
        if _semantic_check(spec):
            return spec["grid"]
    raise RuntimeError("failed to generate a valid puzzle input")
