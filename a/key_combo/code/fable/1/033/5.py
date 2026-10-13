import random
from collections import defaultdict

import numpy as np


BLACK = 0
NON_BG_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
KINDS = ("tl", "tr", "bl", "br")

# Missing-corner kind -> the visible L pixels around the hidden anchor.
L_PIXEL_OFFSETS = {
    "tl": ((0, 1), (1, 0), (1, 1)),
    "tr": ((0, -1), (1, -1), (1, 0)),
    "bl": ((-1, 0), (-1, 1), (0, 1)),
    "br": ((-1, -1), (-1, 0), (0, -1)),
}

L_BOX_OFFSETS = {
    "tl": (0, 0, 1, 1),
    "tr": (0, -1, 1, 0),
    "bl": (-1, 0, 0, 1),
    "br": (-1, -1, 0, 0),
}

OPEN_DIRS = {
    "tl": {"up", "left"},
    "tr": {"up", "right"},
    "bl": {"down", "left"},
    "br": {"down", "right"},
}

GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid, dtype=int).ravel()).argmax())


def colors_present(grid: np.ndarray, exclude=None) -> list[int]:
    blocked = set() if exclude is None else {int(v) for v in exclude}
    return [int(v) for v in np.unique(grid) if int(v) not in blocked]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _anchor_pixels(anchor: tuple[int, int], kind: str) -> list[tuple[int, int]]:
    r, c = anchor
    return [(r + dr, c + dc) for dr, dc in L_PIXEL_OFFSETS[kind]]


def _anchor_bbox(anchor: tuple[int, int], kind: str) -> tuple[int, int, int, int]:
    r, c = anchor
    dr0, dc0, dr1, dc1 = L_BOX_OFFSETS[kind]
    return r + dr0, c + dc0, r + dr1, c + dc1


def _extract_anchors(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    anchors: list[dict] = []
    for color in colors_present(g, exclude={bg}):
        for coords in components_mask(g == color):
            assert coords.shape[0] == 3, "each foreground object must contain exactly three pixels"
            box = bbox_coords(coords)
            r0, c0, r1, c1 = box
            assert r1 - r0 == 1 and c1 - c0 == 1, "each object must fit in a 2x2 box"
            present = {(int(r), int(c)) for r, c in coords}
            box_cells = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
            missing = box_cells - present
            assert len(missing) == 1, "each 2x2 box must miss exactly one corner"
            anchor = next(iter(missing))
            assert g[anchor] == bg, "the missing corner must stay background"
            anchors.append(
                {
                    "color": int(color),
                    "anchor": (int(anchor[0]), int(anchor[1])),
                    "coords": coords,
                    "bbox": box,
                }
            )
    assert anchors, "expected at least one L object"
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored missing L-corners whenever their hidden anchors are adjacent in the row/column visibility graph.

    Input structure:
    - The grid has a dominant background color and several small monochrome foreground objects.
    - Every foreground object is a 3-pixel L shape inside a 2x2 bounding box, so one corner of that box is missing.
    - Those missing corners are invisible anchors; some anchors share a row or column with background gaps between them.

    Rule:
    - For each L object, recover the anchor as the single missing cell in its 2x2 bounding box.
    - In every row and every column, sort all anchors together and only keep adjacent anchor pairs, so any intermediate anchor blocks longer visibility.
    - When an adjacent pair belongs to the same L color, paint the background cells strictly between the two anchors with that color; keep the anchor cells themselves as background.

    Key insight:
    - The endpoints are never drawn directly: the useful geometry comes from the absent L-corner cells, and differently colored anchors still matter because they block farther same-axis matches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    anchors = _extract_anchors(g)

    rows: dict[int, list[dict]] = defaultdict(list)
    cols: dict[int, list[dict]] = defaultdict(list)
    for item in anchors:
        r, c = item["anchor"]
        rows[r].append(item)
        cols[c].append(item)

    planned: dict[tuple[int, int], int] = {}

    for r, group in rows.items():
        ordered = sorted(group, key=lambda item: item["anchor"][1])
        for left, right in zip(ordered, ordered[1:]):
            if left["color"] != right["color"]:
                continue
            c0 = left["anchor"][1]
            c1 = right["anchor"][1]
            for c in range(c0 + 1, c1):
                if g[r, c] != bg:
                    continue
                prev = planned.get((r, c))
                if prev is not None:
                    assert prev == left["color"], "conflicting connector colors"
                planned[(r, c)] = left["color"]

    for c, group in cols.items():
        ordered = sorted(group, key=lambda item: item["anchor"][0])
        for top, bottom in zip(ordered, ordered[1:]):
            if top["color"] != bottom["color"]:
                continue
            r0 = top["anchor"][0]
            r1 = bottom["anchor"][0]
            for r in range(r0 + 1, r1):
                if g[r, c] != bg:
                    continue
                prev = planned.get((r, c))
                if prev is not None:
                    assert prev == top["color"], "conflicting connector colors"
                planned[(r, c)] = top["color"]

    out = g.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _positive_colors(length: int) -> list[int]:
    a, b, c = random.sample(NON_BG_COLORS, 3)
    options = {
        2: ([a, a],),
        3: ([a, a, b], [b, a, a], [a, a, a]),
        4: ([a, a, b, c], [b, a, a, c], [a, a, b, b], [a, a, a, b]),
    }
    return list(random.choice(options[length]))


def _blocker_colors(length: int) -> list[int]:
    a, b, c = random.sample(NON_BG_COLORS, 3)
    options = {
        3: ([a, b, a], [a, b, c]),
        4: ([a, b, a, c], [a, b, c, a]),
    }
    return list(random.choice(options[length]))


def _make_group(axis: str, role: str) -> dict:
    if role == "positive":
        length = random.choice((2, 3, 4))
        colors = _positive_colors(length)
    elif role == "blocker":
        length = random.choice((3, 4))
        colors = _blocker_colors(length)
    else:
        raise ValueError(f"unknown role: {role}")
    return {"axis": axis, "role": role, "colors": colors, "length": len(colors)}


def _sample_spaced(pool, count: int, min_gap: int, attempts: int = 200) -> list[int] | None:
    values = sorted(pool)
    if len(values) < count:
        return None
    if count == 1:
        return [random.choice(values)]
    for _ in range(attempts):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _sample_run(pool, count: int, min_step: int = 4, max_step: int = 8, attempts: int = 200) -> list[int] | None:
    values = sorted(pool)
    if len(values) < count:
        return None
    if count == 1:
        return [random.choice(values)]
    value_set = set(values)
    for _ in range(attempts):
        start = random.choice(values)
        picks = [start]
        cur = start
        ok = True
        for _ in range(count - 1):
            candidates = [
                nxt for nxt in values
                if nxt not in picks and min_step <= nxt - cur <= max_step
            ]
            if not candidates:
                ok = False
                break
            cur = random.choice(candidates)
            picks.append(cur)
        if ok:
            return sorted(picks)
    for _ in range(attempts):
        picks = sorted(random.sample(values, count))
        if all(min_step <= b - a <= max_step for a, b in zip(picks, picks[1:])):
            return picks
        if all(p in value_set for p in picks):
            continue
    return None


def _candidate_kinds(record: dict) -> list[str]:
    if record["axis"] == "row":
        if record["index"] == 0:
            preferred_dir = "right"
        elif record["index"] == record["count"] - 1:
            preferred_dir = "left"
        else:
            preferred_dir = random.choice(("left", "right"))
    elif record["axis"] == "col":
        if record["index"] == 0:
            preferred_dir = "down"
        elif record["index"] == record["count"] - 1:
            preferred_dir = "up"
        else:
            preferred_dir = random.choice(("up", "down"))
    else:
        preferred_dir = random.choice(("up", "down", "left", "right"))

    primary = [kind for kind in KINDS if preferred_dir in OPEN_DIRS[kind]]
    secondary = [kind for kind in KINDS if kind not in primary]
    random.shuffle(primary)
    random.shuffle(secondary)
    return primary + secondary


def _valid_object(
    anchor: tuple[int, int],
    kind: str,
    shape: tuple[int, int],
    occupied_pixels: set[tuple[int, int]],
    occupied_boxes: set[tuple[int, int]],
) -> bool:
    h, w = shape
    r0, c0, r1, c1 = _anchor_bbox(anchor, kind)
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        return False

    box_cells = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
    if box_cells & occupied_boxes:
        return False

    for r, c in _anchor_pixels(anchor, kind):
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied_pixels:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied_pixels:
                    return False
    return True


def _place_records(shape: tuple[int, int], records: list[dict]) -> np.ndarray | None:
    occupied_pixels: set[tuple[int, int]] = set()
    occupied_boxes: set[tuple[int, int]] = set()
    chosen: list[str] = []

    ordered = records[:]
    random.shuffle(ordered)
    for record in ordered:
        anchor = record["anchor"]
        placed = False
        for kind in _candidate_kinds(record):
            if not _valid_object(anchor, kind, shape, occupied_pixels, occupied_boxes):
                continue
            chosen.append(kind)
            for cell in _anchor_pixels(anchor, kind):
                occupied_pixels.add(cell)
            r0, c0, r1, c1 = _anchor_bbox(anchor, kind)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    occupied_boxes.add((rr, cc))
            record["kind"] = kind
            placed = True
            break
        if not placed:
            return None

    grid = np.full(shape, BLACK, dtype=int)
    for record in records:
        for r, c in _anchor_pixels(record["anchor"], record["kind"]):
            grid[r, c] = record["color"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    row_specs = [_make_group("row", "positive")]
    col_specs = [_make_group("col", "positive")]
    if random.choice((True, False)):
        row_specs.append(_make_group("row", "blocker"))
    else:
        col_specs.append(_make_group("col", "blocker"))

    top_rows = list(range(1, max(2, h // 2 - 1)))
    bottom_rows = list(range(min(h - 2, h // 2 + 2), h - 1))
    if len(top_rows) < len(row_specs) or len(bottom_rows) < sum(spec["length"] for spec in col_specs):
        return None

    row_group_rows = _sample_spaced(top_rows, len(row_specs), min_gap=3)
    col_group_cols = _sample_spaced(range(1, w - 1), len(col_specs), min_gap=3)
    if row_group_rows is None or col_group_cols is None:
        return None

    used_cols = set(col_group_cols)
    used_rows = set(row_group_rows)
    row_records: list[dict] = []
    for spec, row in zip(row_specs, row_group_rows):
        available_cols = [c for c in range(1, w - 1) if c not in used_cols]
        cols = _sample_run(available_cols, spec["length"])
        if cols is None:
            return None
        used_cols.update(cols)
        for index, (col, color) in enumerate(zip(cols, spec["colors"])):
            row_records.append(
                {
                    "anchor": (row, col),
                    "color": int(color),
                    "axis": "row",
                    "index": index,
                    "count": len(cols),
                }
            )

    col_records: list[dict] = []
    for spec, col in zip(col_specs, col_group_cols):
        available_rows = [r for r in bottom_rows if r not in used_rows]
        rows = _sample_run(available_rows, spec["length"])
        if rows is None:
            return None
        used_rows.update(rows)
        for index, (row, color) in enumerate(zip(rows, spec["colors"])):
            col_records.append(
                {
                    "anchor": (row, col),
                    "color": int(color),
                    "axis": "col",
                    "index": index,
                    "count": len(rows),
                }
            )

    records = row_records + col_records

    if random.random() < 0.55:
        spare_rows = [r for r in range(1, h - 1) if r not in used_rows]
        spare_cols = [c for c in range(1, w - 1) if c not in used_cols]
        if spare_rows and spare_cols:
            iso_anchor = (random.choice(spare_rows), random.choice(spare_cols))
            records.append(
                {
                    "anchor": iso_anchor,
                    "color": int(random.choice(NON_BG_COLORS)),
                    "axis": "solo",
                    "index": 0,
                    "count": 1,
                }
            )

    grid = _place_records((h, w), records)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_extract_anchors(grid)) != len(records):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner-anchor puzzle input")
