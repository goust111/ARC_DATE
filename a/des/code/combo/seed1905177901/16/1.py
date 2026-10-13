from __future__ import annotations

import numpy as np


CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 8
GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (20, 31)
ENVELOPE_COUNT_RANGE = (2, 5)
OUTER_SIZE_RANGE = (5, 10)
MAX_FILL_BANDS = 6


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in CARDINALS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _rectangle_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0 + 1, r1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _detect_envelope(comp: dict, grid: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return None

    boundary = _rectangle_boundary(r0, c0, r1, c1)
    coords = set((int(r), int(c)) for r, c in comp["coords"])
    if not coords.issubset(boundary):
        return None

    missing = sorted(boundary - coords)
    if len(missing) != 2 or int(comp["area"]) != len(boundary) - 2:
        return None

    side = None
    gap_span: tuple[int, int] | None = None
    if missing[0][0] == r0 and missing[1][0] == r0:
        cols = sorted([missing[0][1], missing[1][1]])
        if cols[1] == cols[0] + 1:
            side = "top"
            gap_span = (cols[0], cols[1])
    elif missing[0][0] == r1 and missing[1][0] == r1:
        cols = sorted([missing[0][1], missing[1][1]])
        if cols[1] == cols[0] + 1:
            side = "bottom"
            gap_span = (cols[0], cols[1])
    elif missing[0][1] == c0 and missing[1][1] == c0:
        rows = sorted([missing[0][0], missing[1][0]])
        if rows[1] == rows[0] + 1:
            side = "left"
            gap_span = (rows[0], rows[1])
    elif missing[0][1] == c1 and missing[1][1] == c1:
        rows = sorted([missing[0][0], missing[1][0]])
        if rows[1] == rows[0] + 1:
            side = "right"
            gap_span = (rows[0], rows[1])

    if side is None or gap_span is None:
        return None

    return {
        "color": int(comp["color"]),
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_span": gap_span,
    }


def _letter_matches(envelope: dict, pos: tuple[int, int], color: int) -> bool:
    if int(color) != int(envelope["color"]):
        return False

    r, c = pos
    r0, c0, r1, c1 = envelope["bbox"]
    a, b = envelope["gap_span"]
    side = envelope["side"]
    if side == "top":
        return r < r0 and c in (a, b)
    if side == "bottom":
        return r > r1 and c in (a, b)
    if side == "left":
        return c < c0 and r in (a, b)
    return c > c1 and r in (a, b)


def _fill_envelope(out: np.ndarray, envelope: dict, count: int) -> None:
    r0, c0, r1, c1 = envelope["bbox"]
    side = envelope["side"]
    color = int(envelope["color"])
    if side in ("top", "bottom"):
        count = min(count, r1 - r0 - 1)
        if count <= 0:
            return
        if side == "top":
            out[r1 - count : r1, c0 + 1 : c1] = color
        else:
            out[r0 + 1 : r0 + 1 + count, c0 + 1 : c1] = color
    else:
        count = min(count, c1 - c0 - 1)
        if count <= 0:
            return
        if side == "left":
            out[r0 + 1 : r1, c1 - count : c1] = color
        else:
            out[r0 + 1 : r1, c0 + 1 : c0 + 1 + count] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each gapped rectangular "envelope" with same-color single-pixel letters,
    collect the letters that lie on the gap's outward extension, and fill the
    envelope interior from the opposite side with one band per collected letter.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _components(g, bg)

    envelopes: list[dict] = []
    letters: list[tuple[int, int, int]] = []
    for comp in comps:
        envelope = _detect_envelope(comp, g, bg)
        if envelope is not None:
            envelopes.append(envelope)
        elif int(comp["area"]) == 1:
            r, c = comp["coords"][0]
            letters.append((int(r), int(c), int(comp["color"])))

    envelopes.sort(key=lambda env: (env["bbox"][0], env["bbox"][1], env["color"]))
    out = g.copy()
    consumed: set[tuple[int, int]] = set()

    for envelope in envelopes:
        hits: list[tuple[int, int]] = []
        for r, c, color in letters:
            if (r, c) in consumed:
                continue
            if _letter_matches(envelope, (r, c), color):
                hits.append((r, c))
        for r, c in hits:
            consumed.add((r, c))
            out[r, c] = bg
        _fill_envelope(out, envelope, len(hits))

    return out


def _sample_directions(rng: np.random.Generator, n: int) -> list[str]:
    base = ["top", "bottom", "left", "right"]
    rng.shuffle(base)
    if n <= 4:
        return base[:n]
    extra = rng.choice(base, size=n - 4, replace=True).tolist()
    dirs = base + [str(x) for x in extra]
    rng.shuffle(dirs)
    return dirs


def _frame_patch(height: int, width: int, color: int, bg: int, side: str, gap_start: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, gap_start : gap_start + 2] = bg
    elif side == "bottom":
        patch[-1, gap_start : gap_start + 2] = bg
    elif side == "left":
        patch[gap_start : gap_start + 2, 0] = bg
    else:
        patch[gap_start : gap_start + 2, -1] = bg
    return patch


def _gap_cells(record: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = record["bbox"]
    gap_start = int(record["gap_start"])
    side = record["side"]
    if side == "top":
        return (r0, c0 + gap_start), (r0, c0 + gap_start + 1)
    if side == "bottom":
        return (r1, c0 + gap_start), (r1, c0 + gap_start + 1)
    if side == "left":
        return (r0 + gap_start, c0), (r0 + gap_start + 1, c0)
    return (r0 + gap_start, c1), (r0 + gap_start + 1, c1)


def _corridor_cells(record: dict, height: int, width: int) -> set[tuple[int, int]]:
    (a0, b0), (a1, b1) = _gap_cells(record)
    cells: set[tuple[int, int]] = set()
    if record["side"] == "top":
        for r in range(0, a0):
            cells.add((r, b0))
            cells.add((r, b1))
    elif record["side"] == "bottom":
        for r in range(a0 + 1, height):
            cells.add((r, b0))
            cells.add((r, b1))
    elif record["side"] == "left":
        for c in range(0, b0):
            cells.add((a0, c))
            cells.add((a1, c))
    else:
        for c in range(b0 + 1, width):
            cells.add((a0, c))
            cells.add((a1, c))
    return cells


def _reserved_cells(record: dict, height: int, width: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = record["bbox"]
    cells = set()
    for r in range(max(0, r0 - 1), min(height, r1 + 2)):
        for c in range(max(0, c0 - 1), min(width, c1 + 2)):
            cells.add((r, c))
    cells |= _corridor_cells(record, height, width)
    return cells


def _singleton_clear(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG_COLOR:
        return False
    h, w = grid.shape
    for dr, dc in CARDINALS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG_COLOR:
            return False
    return True


def _ray_slots(record: dict, phase: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = record["bbox"]
    (g0r, g0c), (g1r, g1c) = _gap_cells(record)
    slots: list[tuple[int, int]] = []
    if record["side"] == "top":
        cols = (g0c, g1c)
        for step in range(1, r0 + 1):
            slots.append((r0 - step, cols[(step + phase) % 2]))
    elif record["side"] == "bottom":
        cols = (g0c, g1c)
        h = record["grid_shape"][0]
        for step in range(1, h - r1):
            slots.append((r1 + step, cols[(step + phase) % 2]))
    elif record["side"] == "left":
        rows = (g0r, g1r)
        for step in range(1, c0 + 1):
            slots.append((rows[(step + phase) % 2], c0 - step))
    else:
        rows = (g0r, g1r)
        w = record["grid_shape"][1]
        for step in range(1, w - c1):
            slots.append((rows[(step + phase) % 2], c1 + step))
    return slots


def _place_envelopes(rng: np.random.Generator, height: int, width: int, colors: list[int], directions: list[str]) -> list[dict] | None:
    specs: list[dict] = []
    for color, side in zip(colors, directions, strict=False):
        outer_h = int(rng.integers(OUTER_SIZE_RANGE[0], OUTER_SIZE_RANGE[1] + 1))
        outer_w = int(rng.integers(OUTER_SIZE_RANGE[0], OUTER_SIZE_RANGE[1] + 1))
        gap_limit = outer_w if side in ("top", "bottom") else outer_h
        gap_start = int(rng.integers(1, gap_limit - 2))
        specs.append(
            {
                "color": int(color),
                "side": side,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "gap_start": gap_start,
            }
        )

    specs.sort(key=lambda rec: rec["outer_h"] * rec["outer_w"], reverse=True)
    used: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for spec in specs:
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        side = spec["side"]

        r_min = 1 if side == "top" else 0
        r_max = height - outer_h - (1 if side == "bottom" else 0)
        c_min = 1 if side == "left" else 0
        c_max = width - outer_w - (1 if side == "right" else 0)
        if r_min > r_max or c_min > c_max:
            return None

        success = False
        for _ in range(250):
            r0 = int(rng.integers(r_min, r_max + 1))
            c0 = int(rng.integers(c_min, c_max + 1))
            record = {
                **spec,
                "bbox": (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1),
                "grid_shape": (height, width),
            }
            reserve = _reserved_cells(record, height, width)
            if reserve & used:
                continue
            used |= reserve
            placed.append(record)
            success = True
            break
        if not success:
            return None

    return placed


def _choose_counts(rng: np.random.Generator, records: list[dict]) -> list[int]:
    counts: list[int] = []
    positive_slots: list[int] = []
    for idx, record in enumerate(records):
        r0, c0, r1, c1 = record["bbox"]
        side = record["side"]
        capacity = (r1 - r0 - 1) if side in ("top", "bottom") else (c1 - c0 - 1)
        if side == "top":
            clearance = r0
        elif side == "bottom":
            clearance = record["grid_shape"][0] - 1 - r1
        elif side == "left":
            clearance = c0
        else:
            clearance = record["grid_shape"][1] - 1 - c1
        max_letters = min(capacity, clearance, MAX_FILL_BANDS)
        if max_letters > 0:
            positive_slots.append(idx)
        counts.append(int(rng.integers(0, max_letters + 1)))

    if positive_slots and sum(counts) == 0:
        chosen = int(rng.choice(positive_slots))
        r0, c0, r1, c1 = records[chosen]["bbox"]
        side = records[chosen]["side"]
        capacity = (r1 - r0 - 1) if side in ("top", "bottom") else (c1 - c0 - 1)
        if side == "top":
            clearance = r0
        elif side == "bottom":
            clearance = records[chosen]["grid_shape"][0] - 1 - r1
        elif side == "left":
            clearance = c0
        else:
            clearance = records[chosen]["grid_shape"][1] - 1 - c1
        max_letters = min(capacity, clearance, MAX_FILL_BANDS)
        counts[chosen] = int(rng.integers(1, max_letters + 1))
    return counts


def _free_singleton_positions(grid: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            if _singleton_clear(grid, r, c):
                cells.append((r, c))
    return cells


def _expected_output(inp: np.ndarray, records: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for record in records:
        for r, c in record["letters"]:
            out[r, c] = bg
        env = {
            "color": record["color"],
            "bbox": record["bbox"],
            "side": record["side"],
            "gap_span": record["gap_span"],
        }
        _fill_envelope(out, env, len(record["letters"]))
    return out


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    width = int(rng.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    count = int(rng.integers(ENVELOPE_COUNT_RANGE[0], ENVELOPE_COUNT_RANGE[1] + 1))

    palette = [c for c in range(10) if c != BG_COLOR]
    rng.shuffle(palette)
    colors = palette[:count]
    directions = _sample_directions(rng, count)
    records = _place_envelopes(rng, height, width, colors, directions)
    if records is None:
        return None

    inp = np.full((height, width), BG_COLOR, dtype=int)
    corridor_union: set[tuple[int, int]] = set()
    bbox_union: set[tuple[int, int]] = set()
    for record in records:
        r0, c0, r1, c1 = record["bbox"]
        patch = _frame_patch(
            record["outer_h"],
            record["outer_w"],
            record["color"],
            BG_COLOR,
            record["side"],
            record["gap_start"],
        )
        inp[r0 : r1 + 1, c0 : c1 + 1] = patch
        corridor_union |= _corridor_cells(record, height, width)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                bbox_union.add((r, c))

        gap_a, gap_b = _gap_cells(record)
        if record["side"] in ("top", "bottom"):
            record["gap_span"] = tuple(sorted([gap_a[1], gap_b[1]]))
        else:
            record["gap_span"] = tuple(sorted([gap_a[0], gap_b[0]]))

    fill_counts = _choose_counts(rng, records)
    for record, fill_count in zip(records, fill_counts, strict=False):
        slots = _ray_slots(record, int(rng.integers(0, 2)))
        if fill_count > len(slots):
            return None
        picks: list[tuple[int, int]] = []
        if fill_count:
            idxs = rng.choice(len(slots), size=fill_count, replace=False)
            picks = [slots[int(i)] for i in idxs.tolist()]
        record["letters"] = sorted(picks)
        for r, c in record["letters"]:
            if not _singleton_clear(inp, r, c):
                return None
            inp[r, c] = record["color"]

    # Same-color distractor letters that should remain.
    forbid = corridor_union | bbox_union
    for record in records:
        distractor_count = int(rng.integers(0, 3))
        record["distractors"] = []
        for _ in range(distractor_count):
            candidates = _free_singleton_positions(inp, forbid)
            if not candidates:
                break
            choice = candidates[int(rng.integers(0, len(candidates)))]
            inp[choice] = record["color"]
            record["distractors"].append(choice)

    expected = _expected_output(inp, records, BG_COLOR)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
