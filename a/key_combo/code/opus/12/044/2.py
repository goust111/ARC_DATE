from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 5
MARKER_COLORS = (6, 9)
BODY_COLORS = (1, 2, 3, 4, 7)
SIDES = ("T", "B", "L", "R")
OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}
SIDE_DIR = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}
TYPE_HOLLOW = "hollow"
TYPE_SOLID = "solid"
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = src.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = src
        return
    mask = src != int(transparent)
    view[mask] = src[mask]


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def marker_side(r: int, c: int, h: int, w: int) -> str:
    sides = []
    if r == 0:
        sides.append("T")
    if r == h - 1:
        sides.append("B")
    if c == 0:
        sides.append("L")
    if c == w - 1:
        sides.append("R")
    assert len(sides) == 1, "marker must lie on exactly one border side"
    return sides[0]


def side_coord(side: str, offset: int) -> tuple[int, int]:
    if side == "T":
        return 0, offset
    if side == "B":
        return 3, offset
    if side == "L":
        return offset, 0
    return offset, 3


def classify_patch(patch: np.ndarray, bg: int) -> str:
    return TYPE_HOLLOW if bool((patch[1:3, 1:3] == int(bg)).all()) else TYPE_SOLID


def analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects: list[dict] = []
    for comp in find_components(g, bg):
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 == 4 and c1 - c0 + 1 == 4, "expected only 4x4 objects"
        patch = crop_bbox(g, bbox)
        marker_locs = np.argwhere(np.isin(patch, MARKER_COLORS))
        assert marker_locs.shape == (1, 2), "each object must have exactly one connector"
        mr, mc = map(int, marker_locs[0])
        objects.append(
            {
                "bbox": bbox,
                "patch": patch,
                "kind": classify_patch(patch, bg),
                "top": int(r0),
                "left": int(c0),
                "marker_color": int(patch[mr, mc]),
                "marker_pos": (mr, mc),
                "marker_side": marker_side(mr, mc, 4, 4),
            }
        )
    assert 4 <= len(objects) <= 6
    return objects


def assemble_pair(a: dict, b: dict, bg: int) -> np.ndarray:
    assert a["kind"] == b["kind"]
    assert a["marker_color"] == b["marker_color"]
    sa = a["marker_side"]
    sb = b["marker_side"]
    assert OPPOSITE[sa] == sb, "paired connectors must face each other"

    mr_a, mc_a = a["marker_pos"]
    mr_b, mc_b = b["marker_pos"]
    dr, dc = SIDE_DIR[sa]
    base_a = (0, 0)
    base_b = (mr_a + dr - mr_b, mc_a + dc - mc_b)

    min_r = min(base_a[0], base_b[0])
    min_c = min(base_a[1], base_b[1])
    max_r = max(base_a[0] + 3, base_b[0] + 3)
    max_c = max(base_a[1] + 3, base_b[1] + 3)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)

    ra, ca = base_a[0] - min_r, base_a[1] - min_c
    rb, cb = base_b[0] - min_r, base_b[1] - min_c
    paste(out, a["patch"], ra, ca, transparent=bg)
    overlap = (out[rb : rb + 4, cb : cb + 4] != int(bg)) & (b["patch"] != int(bg))
    assert not bool(overlap.any()), "docked objects must not overlap"
    paste(out, b["patch"], rb, cb, transparent=bg)
    return out


def build_items(objects: list[dict], kind: str, bg: int) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        if obj["kind"] == kind:
            groups[obj["marker_color"]].append(obj)

    items: list[dict] = []
    for color, members in groups.items():
        assert len(members) in (1, 2), f"ambiguous pairing for {kind} connector color {color}"
        members = sorted(members, key=lambda o: (o["top"], o["left"]))
        if len(members) == 2:
            patch = assemble_pair(members[0], members[1], bg)
            items.append(
                {
                    "patch": patch,
                    "top": min(m["top"] for m in members),
                    "left": min(m["left"] for m in members),
                }
            )
        else:
            items.append(
                {
                    "patch": members[0]["patch"],
                    "top": members[0]["top"],
                    "left": members[0]["left"],
                }
            )

    items.sort(key=lambda item: (item["top"], item["left"]))
    return items


def stack_columns(hollow_items: list[dict], solid_items: list[dict], bg: int) -> np.ndarray:
    left_w = max((item["patch"].shape[1] for item in hollow_items), default=0)
    right_w = max((item["patch"].shape[1] for item in solid_items), default=0)
    left_h = sum(int(item["patch"].shape[0]) for item in hollow_items)
    right_h = sum(int(item["patch"].shape[0]) for item in solid_items)
    out = np.full((max(left_h, right_h), left_w + right_w), int(bg), dtype=int)

    r = 0
    for item in hollow_items:
        patch = item["patch"]
        paste(out, patch, r, 0, transparent=bg)
        r += int(patch.shape[0])

    r = 0
    for item in solid_items:
        patch = item["patch"]
        paste(out, patch, r, left_w, transparent=bg)
        r += int(patch.shape[0])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-type connector-marked 4x4 squares, then stack hollow results left and solid results right.

    Input structure:
    - The grid has grey background cells and 4-6 scattered 4x4 monochrome square objects.
    - Each object is either a hollow frame with a grey 2x2 interior or a solid 4x4 block.
    - Exactly one non-corner border pixel of each object is replaced by a connector marker in color 6 or 9.

    Rule:
    - Classify every 4x4 object by its 2x2 interior: all-grey interior means hollow, otherwise solid.
    - Within each class separately, pair the two objects whose connector markers have the same color; same-colored markers across classes do not pair.
    - Dock each paired duo edge-to-edge according to the exposed connector sides so the two marker pixels become 4-neighbors, while leaving unpaired objects as single 4x4 items.
    - Stack all hollow-class composites or singletons from top to bottom in the left column and all solid-class ones in the right column, ordered by each item's topmost input row; pad the remaining output cells with grey.

    Key insight:
    - Connector color alone is insufficient: the pairing scope is first restricted by hollow-vs-solid classification, so cross-class same-color markers are deliberate decoys.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = analyze_objects(g)
    hollow_items = build_items(objects, TYPE_HOLLOW, bg)
    solid_items = build_items(objects, TYPE_SOLID, bg)
    return stack_columns(hollow_items, solid_items, bg)


def make_object(kind: str, body_color: int, marker_color: int, side: str, offset: int) -> np.ndarray:
    patch = np.full((4, 4), int(body_color), dtype=int)
    if kind == TYPE_HOLLOW:
        patch[1:3, 1:3] = BG
    r, c = side_coord(side, offset)
    patch[r, c] = int(marker_color)
    return patch


def plan_to_expected(objects: list[dict]) -> np.ndarray:
    hollow_items = build_items(objects, TYPE_HOLLOW, BG)
    solid_items = build_items(objects, TYPE_SOLID, BG)
    return stack_columns(hollow_items, solid_items, BG)


def choose_sides_for_pair() -> tuple[str, str]:
    axis = random.choice(("vertical", "horizontal"))
    if axis == "vertical":
        return random.choice((("T", "B"), ("B", "T")))
    return random.choice((("L", "R"), ("R", "L")))


def object_plan(kind: str, marker_color: int, body_color: int, side: str) -> dict:
    offset = random.choice((1, 2))
    patch = make_object(kind, body_color, marker_color, side, offset)
    mr, mc = map(int, np.argwhere(np.isin(patch, MARKER_COLORS))[0])
    return {
        "kind": kind,
        "marker_color": int(marker_color),
        "marker_side": side,
        "marker_pos": (mr, mc),
        "patch": patch,
        "top": None,
        "left": None,
        "bbox": None,
    }


def assign_body_colors(count: int) -> list[int]:
    colors = list(BODY_COLORS)
    random.shuffle(colors)
    while len(colors) < count:
        colors.append(random.choice(BODY_COLORS))
    return colors[:count]


def build_generation_plan() -> list[dict]:
    scenario = random.choices(
        ("one_pair", "two_pairs_one_single", "two_pairs_two_singles"),
        weights=(3, 4, 3),
        k=1,
    )[0]
    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    pair_a, pair_b = colors

    spec: list[tuple[str, int]] = []
    if scenario == "one_pair":
        pair_kind = random.choice((TYPE_HOLLOW, TYPE_SOLID))
        other_kind = TYPE_SOLID if pair_kind == TYPE_HOLLOW else TYPE_HOLLOW
        spec.extend([(pair_kind, pair_a), (pair_kind, pair_a)])
        spec.append((other_kind, pair_a))
        spec.append((random.choice((TYPE_HOLLOW, TYPE_SOLID)), pair_b))
    elif scenario == "two_pairs_one_single":
        spec.extend(
            [
                (TYPE_HOLLOW, pair_a),
                (TYPE_HOLLOW, pair_a),
                (TYPE_SOLID, pair_b),
                (TYPE_SOLID, pair_b),
                random.choice(((TYPE_SOLID, pair_a), (TYPE_HOLLOW, pair_b))),
            ]
        )
    else:
        spec.extend(
            [
                (TYPE_HOLLOW, pair_a),
                (TYPE_HOLLOW, pair_a),
                (TYPE_SOLID, pair_b),
                (TYPE_SOLID, pair_b),
                (TYPE_SOLID, pair_a),
                (TYPE_HOLLOW, pair_b),
            ]
        )

    hollow_n = sum(kind == TYPE_HOLLOW for kind, _ in spec)
    solid_n = sum(kind == TYPE_SOLID for kind, _ in spec)
    assert hollow_n >= 1 and solid_n >= 1

    pair_sides: dict[tuple[str, int], tuple[str, str]] = {}
    for kind, color in ((TYPE_HOLLOW, pair_a), (TYPE_SOLID, pair_b), (TYPE_SOLID, pair_a), (TYPE_HOLLOW, pair_b)):
        count = sum(1 for skind, scolor in spec if skind == kind and scolor == color)
        if count == 2:
            pair_sides[(kind, color)] = choose_sides_for_pair()

    body_colors = assign_body_colors(len(spec))
    plans: list[dict] = []
    pair_usage: dict[tuple[str, int], int] = defaultdict(int)
    for idx, ((kind, marker_color), body_color) in enumerate(zip(spec, body_colors)):
        key = (kind, marker_color)
        if key in pair_sides:
            side = pair_sides[key][pair_usage[key]]
            pair_usage[key] += 1
        else:
            side = random.choice(SIDES)
        plan = object_plan(kind, marker_color, body_color, side)
        plan["id"] = idx
        plans.append(plan)

    same_type_pairs = 0
    for kind in (TYPE_HOLLOW, TYPE_SOLID):
        for color in MARKER_COLORS:
            count = sum(1 for obj in plans if obj["kind"] == kind and obj["marker_color"] == color)
            assert count in (0, 1, 2)
            if count == 2:
                same_type_pairs += 1
    assert same_type_pairs >= 1
    assert any(
        sum(1 for obj in plans if obj["kind"] == obj0["kind"] and obj["marker_color"] == obj0["marker_color"]) == 1
        for obj0 in plans
    )
    assert any(
        sum(1 for obj in plans if obj["kind"] != obj0["kind"] and obj["marker_color"] == obj0["marker_color"]) >= 1
        for obj0 in plans
    )
    return plans


def place_objects(plans: list[dict]) -> np.ndarray | None:
    hollow_n = sum(obj["kind"] == TYPE_HOLLOW for obj in plans)
    solid_n = sum(obj["kind"] == TYPE_SOLID for obj in plans)
    max_per_kind = max(hollow_n, solid_n)

    h = random.randint(max(19, 7 * max_per_kind - 1), 28)
    w = random.randint(19, 28)
    row_offset = random.randint(0, 2)
    col_offset = random.randint(0, 2)
    row_slots = list(range(row_offset, h - 3, 6))
    col_slots = list(range(col_offset, w - 3, 6))
    if len(row_slots) < max_per_kind or len(col_slots) < 2:
        return None

    hollow_rows = random.sample(row_slots, hollow_n)
    solid_rows = random.sample(row_slots, solid_n)
    random.shuffle(hollow_rows)
    random.shuffle(solid_rows)

    kind_rows = {
        TYPE_HOLLOW: hollow_rows,
        TYPE_SOLID: solid_rows,
    }
    kind_index = defaultdict(int)
    used_slots: set[tuple[int, int]] = set()

    for obj in sorted(plans, key=lambda item: (item["kind"], item["id"])):
        row = kind_rows[obj["kind"]][kind_index[obj["kind"]]]
        kind_index[obj["kind"]] += 1
        candidates = [(row, col) for col in random.sample(col_slots, len(col_slots)) if (row, col) not in used_slots]
        if not candidates:
            return None
        r0, c0 = candidates[0]
        used_slots.add((r0, c0))
        obj["top"] = int(r0)
        obj["left"] = int(c0)
        obj["bbox"] = (int(r0), int(c0), int(r0 + 3), int(c0 + 3))

    grid = np.full((h, w), BG, dtype=int)
    for obj in plans:
        paste(grid, obj["patch"], int(obj["top"]), int(obj["left"]), transparent=BG)
    return grid


def validate_input_grid(grid: np.ndarray, plans: list[dict], expected: np.ndarray) -> bool:
    if grid is None or expected is None:
        return False
    if grid.ndim != 2 or expected.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    objects = analyze_objects(grid)
    if len(objects) != len(plans):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    actual_counts = Counter((obj["kind"], obj["marker_color"]) for obj in objects)
    plan_counts = Counter((obj["kind"], obj["marker_color"]) for obj in plans)
    if actual_counts != plan_counts:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        plans = build_generation_plan()
        grid = place_objects(plans)
        if grid is None:
            continue
        expected = plan_to_expected(plans)
        if validate_input_grid(grid, plans, expected):
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
