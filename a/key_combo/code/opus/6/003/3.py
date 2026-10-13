import itertools
from collections import deque

import numpy as np


BG = 0
TOP_COLOR = 1
BOTTOM_COLOR = 2
SEP_COLOR = 3
RECT_COLOR = 5

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _edge_of_index(idx: int, h: int, w: int) -> str:
    top_len = w
    right_len = h - 1
    bottom_len = w - 1
    if idx < top_len:
        return "top"
    idx -= top_len
    if idx < right_len:
        return "right"
    idx -= right_len
    if idx < bottom_len:
        return "bottom"
    return "left"


def _runs_from_missing(missing: list[bool]) -> list[list[int]]:
    n = len(missing)
    runs: list[list[int]] = []
    for i in range(n):
        if not missing[i] or missing[(i - 1) % n]:
            continue
        run = [i]
        j = (i + 1) % n
        while j != i and missing[j]:
            run.append(j)
            j = (j + 1) % n
        runs.append(run)
    return runs


def _detect_defect_starts(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, ...]:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert set(np.unique(patch)).issubset({BG, RECT_COLOR})
    if patch.shape[0] > 2 and patch.shape[1] > 2:
        assert bool(np.all(patch[1:-1, 1:-1] == RECT_COLOR))

    coords = _perimeter_coords(bbox)
    missing = [int(grid[r, c]) == BG for r, c in coords]
    runs = _runs_from_missing(missing)
    assert 1 <= len(runs) <= 2
    starts: list[int] = []
    for run in runs:
        assert len(run) == 2
        starts.append(int(run[0]))

    local = np.full((r1 - r0 + 1, c1 - c0 + 1), RECT_COLOR, dtype=int)
    for start in starts:
        for k in (0, 1):
            rr, cc = coords[(start + k) % len(coords)]
            local[rr - r0, cc - c0] = BG
    comps = _components_mask(local == RECT_COLOR)
    assert len(comps) == 1
    assert _bbox_coords(comps[0]) == (0, 0, local.shape[0] - 1, local.shape[1] - 1)
    return tuple(sorted(starts))


def _paint_defective_rect(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    defect_starts: tuple[int, ...],
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    perimeter = _perimeter_coords(bbox)
    n = len(perimeter)
    for start in defect_starts:
        for k in (0, 1):
            rr, cc = perimeter[(int(start) + k) % n]
            grid[rr, cc] = BG


def _paint_output_rect(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    defect_starts: tuple[int, ...],
    panel: str,
) -> None:
    delta = 2 if panel == "top" else -2
    shifted = tuple(sorted(((start + delta) % (2 * (bbox[2] - bbox[0] + 1) + 2 * (bbox[3] - bbox[1] + 1) - 4)) for start in defect_starts))
    _paint_defective_rect(out, bbox, shifted, RECT_COLOR)


def _mirror_bbox(bbox: tuple[int, int, int, int], sep_row: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return 2 * sep_row - r1, c0, 2 * sep_row - r0, c1


def _corridor_fill(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    sep_row: int,
) -> None:
    r0, c0, r1, c1 = bbox
    mr0, _, mr1, _ = _mirror_bbox(bbox, sep_row)
    rr0 = min(r0, mr0)
    rr1 = max(r1, mr1)
    mask = out[rr0 : rr1 + 1, c0 : c1 + 1] == BG
    out[rr0 : rr1 + 1, c0 : c1 + 1][mask] = SEP_COLOR


def _build_output_from_records(
    inp: np.ndarray,
    records: list[dict],
    *,
    reverse_direction: bool = False,
    mirror_repaired: bool = True,
    keep_original_defects: bool = False,
) -> np.ndarray:
    out = inp.copy()
    sep_row = inp.shape[0] // 2
    for rec in records:
        bbox = rec["bbox"]
        panel = rec["panel"]
        starts = rec["defect_starts"]
        pcoords = _perimeter_coords(bbox)
        n = len(pcoords)
        delta = 2 if panel == "top" else -2
        if reverse_direction:
            delta = -delta
        shifted = starts if keep_original_defects else tuple(sorted(((start + delta) % n) for start in starts))
        _paint_defective_rect(out, bbox, shifted, RECT_COLOR)

        mirror_bbox = _mirror_bbox(bbox, sep_row)
        mirror_color = BOTTOM_COLOR if panel == "top" else TOP_COLOR
        if mirror_repaired:
            mr0, mc0, mr1, mc1 = mirror_bbox
            out[mr0 : mr1 + 1, mc0 : mc1 + 1] = mirror_color
        else:
            _paint_defective_rect(out, mirror_bbox, shifted, mirror_color)

        _corridor_fill(out, bbox, sep_row)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each rectangle's two-pixel edge defects along its perimeter, then mirror a repaired copy across the separator.

    Input structure:
    - A black background contains one full-width green separator row, with a full-width blue row immediately above it and a full-width red row immediately below it.
    - Strictly above the blue band and strictly below the red band sit 1-2 gray near-solid rectangles per panel.
    - Each rectangle is solid except for 1-2 defects: every defect is exactly two consecutive background pixels on the rectangle perimeter.

    Rule:
    - Recover each rectangle's full bounding box and read its perimeter clockwise from the top-left corner.
    - In the upper panel move every defect two perimeter positions forward; in the lower panel move every defect two positions backward.
    - Redraw the original rectangle with the shifted defects, mirror it across the green separator, repaint the mirrored copy with the opposite panel color, and repair all mirrored defects so the copy is a full rectangle.
    - Inside the shared column span between an original rectangle and its mirror, recolor only background pixels to green; keep the separator row and the blue/red panel bands unchanged.

    Key insight:
    - The missing bars must be tracked on the rectangle's circular perimeter rather than on a fixed edge, because a two-step shift can wrap a defect around a corner before the repaired mirror is drawn on the opposite panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep_rows = [r for r in range(h) if bool(np.all(g[r] == SEP_COLOR))]
    assert len(sep_rows) == 1
    sep_row = sep_rows[0]
    assert 0 < sep_row < h - 1
    assert bool(np.all(g[sep_row - 1] == TOP_COLOR))
    assert bool(np.all(g[sep_row + 1] == BOTTOM_COLOR))

    comps = _components_mask(g == RECT_COLOR)
    assert 2 <= len(comps) <= 4

    out = g.copy()
    for coords in comps:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        panel = "top" if r1 < sep_row - 1 else "bottom"
        assert panel == "top" or r0 > sep_row + 1

        starts = _detect_defect_starts(g, bbox)
        delta = 2 if panel == "top" else -2
        shifted = tuple(sorted(((start + delta) % len(_perimeter_coords(bbox))) for start in starts))

        _paint_defective_rect(out, bbox, shifted, RECT_COLOR)
        mr0, mc0, mr1, mc1 = _mirror_bbox(bbox, sep_row)
        out[mr0 : mr1 + 1, mc0 : mc1 + 1] = BOTTOM_COLOR if panel == "top" else TOP_COLOR
        _corridor_fill(out, bbox, sep_row)

    return out


def _composition(total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    cuts = sorted(int(x) for x in RNG.integers(0, total + parts - 1, size=parts - 1))
    vals: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    return vals


def _pack_intervals(length: int, sizes: list[int], *, margin: int, min_gap: int) -> list[tuple[int, int]] | None:
    need = 2 * margin + sum(sizes) + min_gap * (len(sizes) - 1)
    if need > length:
        return None
    extra = length - need
    gaps = _composition(extra, len(sizes) + 1)
    pos = margin + gaps[0]
    out: list[tuple[int, int]] = []
    for i, size in enumerate(sizes):
        out.append((pos, pos + int(size) - 1))
        pos += int(size)
        if i + 1 < len(sizes):
            pos += min_gap + gaps[i + 1]
    return out


def _defect_cells(starts: tuple[int, ...], perimeter_len: int) -> set[int]:
    cells: set[int] = set()
    for start in starts:
        cells.add(int(start) % perimeter_len)
        cells.add((int(start) + 1) % perimeter_len)
    return cells


def _crosses_corner(start: int, h: int, w: int) -> bool:
    p = 2 * h + 2 * w - 4
    a = int(start) % p
    b = (a + 1) % p
    return _edge_of_index(a, h, w) != _edge_of_index(b, h, w)


def _valid_defects(
    h: int,
    w: int,
    starts: tuple[int, ...],
    delta: int,
    *,
    require_corner_shift: bool,
) -> bool:
    p = 2 * h + 2 * w - 4
    old_cells = _defect_cells(starts, p)
    if len(old_cells) != 2 * len(starts):
        return False

    shifted = tuple(sorted(((start + delta) % p) for start in starts))
    new_cells = _defect_cells(shifted, p)
    if len(new_cells) != 2 * len(starts):
        return False
    if old_cells & new_cells:
        return False

    if require_corner_shift and not any(_crosses_corner(start, h, w) for start in shifted):
        return False

    patch = np.full((h, w), RECT_COLOR, dtype=int)
    bbox = (0, 0, h - 1, w - 1)
    _paint_defective_rect(patch, bbox, starts, RECT_COLOR)
    comps = _components_mask(patch == RECT_COLOR)
    if len(comps) != 1:
        return False
    if _bbox_coords(comps[0]) != bbox:
        return False
    return True


def _sample_defects(
    h: int,
    w: int,
    delta: int,
    *,
    require_corner_shift: bool,
    force_two: bool,
) -> tuple[int, ...] | None:
    p = 2 * h + 2 * w - 4
    counts = [2] if force_two else [1, 2] if RNG.random() < 0.55 else [1]
    candidates = list(range(p))
    valid: list[tuple[int, ...]] = []
    for count in counts:
        for combo in itertools.combinations(candidates, count):
            if _valid_defects(h, w, combo, delta, require_corner_shift=require_corner_shift):
                valid.append(tuple(int(x) for x in combo))
        if valid:
            break
    if not valid:
        return None
    return valid[int(RNG.integers(len(valid)))]


def _choose_rows(
    panel: str,
    rect_h: int,
    sep_row: int,
    grid_h: int,
) -> tuple[int, int]:
    if panel == "top":
        r0 = int(RNG.integers(0, sep_row - 1 - rect_h + 1))
    else:
        start = sep_row + 2
        r0 = int(RNG.integers(start, grid_h - rect_h + 1))
    return r0, r0 + rect_h - 1


def _render_input(grid_h: int, grid_w: int, sep_row: int, records: list[dict]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[sep_row, :] = SEP_COLOR
    grid[sep_row - 1, :] = TOP_COLOR
    grid[sep_row + 1, :] = BOTTOM_COLOR
    for rec in records:
        _paint_defective_rect(grid, rec["bbox"], rec["defect_starts"], RECT_COLOR)
    return grid


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, records: list[dict]) -> bool:
    if np.array_equal(inp, expected):
        return False
    if np.array_equal(expected, _build_output_from_records(inp, records, keep_original_defects=True)):
        return False
    if np.array_equal(expected, _build_output_from_records(inp, records, reverse_direction=True)):
        return False
    if np.array_equal(expected, _build_output_from_records(inp, records, mirror_repaired=False)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(RNG.choice(np.arange(19, 30, 2)))
    grid_w = int(RNG.integers(19, 30))
    sep_row = grid_h // 2

    top_count = int(RNG.integers(1, 3))
    bottom_count = int(RNG.integers(1, 3))
    total = top_count + bottom_count

    rect_ws = [int(RNG.integers(3, 6)) for _ in range(total)]
    intervals = _pack_intervals(grid_w, rect_ws, margin=1, min_gap=2)
    if intervals is None:
        return None

    object_specs = []
    for idx in range(total):
        object_specs.append(
            {
                "width": rect_ws[idx],
                "height": int(RNG.integers(4, 7)),
                "interval": intervals[idx],
            }
        )
    RNG.shuffle(object_specs)

    special_index = int(RNG.integers(total))
    special_mode = ("corner", "two", "both")[int(RNG.integers(3))]

    records: list[dict] = []
    for idx, spec in enumerate(object_specs[:top_count]):
        h = spec["height"]
        w = spec["width"]
        delta = 2
        starts = _sample_defects(
            h,
            w,
            delta,
            require_corner_shift=(idx == special_index and special_mode in {"corner", "both"}),
            force_two=(idx == special_index and special_mode in {"two", "both"}),
        )
        if starts is None:
            return None
        c0, c1 = spec["interval"]
        r0, r1 = _choose_rows("top", h, sep_row, grid_h)
        records.append({"panel": "top", "bbox": (r0, c0, r1, c1), "defect_starts": starts})

    for local_idx, spec in enumerate(object_specs[top_count:]):
        idx = top_count + local_idx
        h = spec["height"]
        w = spec["width"]
        delta = -2
        starts = _sample_defects(
            h,
            w,
            delta,
            require_corner_shift=(idx == special_index and special_mode in {"corner", "both"}),
            force_two=(idx == special_index and special_mode in {"two", "both"}),
        )
        if starts is None:
            return None
        c0, c1 = spec["interval"]
        r0, r1 = _choose_rows("bottom", h, sep_row, grid_h)
        records.append({"panel": "bottom", "bbox": (r0, c0, r1, c1), "defect_starts": starts})

    inp = _render_input(grid_h, grid_w, sep_row, records)
    expected = _build_output_from_records(inp, records)
    if not _semantic_checks(inp, expected, records):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
