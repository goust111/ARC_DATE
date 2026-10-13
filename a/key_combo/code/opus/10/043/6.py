from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3)
MARKER = 6
ANOMALY_SIZE = 3
MASK_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4))
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _orbit_positions(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    sr = h - 1 - int(r)
    sc = w - 1 - int(c)
    return sorted({(int(r), int(c)), (int(r), sc), (sr, int(c)), (sr, sc)})


def _find_marker_rect(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    marker_mask = grid == MARKER
    box = _bbox(marker_mask)
    assert box is not None, "expected a marker rectangle"
    r0, c0, r1, c1 = box
    patch = marker_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(patch)), "marker must be a solid rectangle"
    assert int(marker_mask.sum()) == int(patch.size), "marker color must appear only in the rectangle"
    return box, marker_mask


def _visible_orbit_values(
    grid: np.ndarray, marker_mask: np.ndarray, r: int, c: int
) -> list[tuple[tuple[int, int], int]]:
    vals: list[tuple[tuple[int, int], int]] = []
    for rr, cc in _orbit_positions(r, c, grid.shape):
        if not marker_mask[rr, cc]:
            vals.append(((rr, cc), int(grid[rr, cc])))
    return vals


def _suspicious_mask(grid: np.ndarray, marker_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(marker_mask, dtype=bool)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if marker_mask[r, c]:
                continue
            visible = _visible_orbit_values(grid, marker_mask, r, c)
            if len({value for _pos, value in visible}) > 1:
                out[r, c] = True
    return out


def _majority_visible_value(grid: np.ndarray, marker_mask: np.ndarray, r: int, c: int) -> int:
    visible = _visible_orbit_values(grid, marker_mask, r, c)
    counts = Counter(value for _pos, value in visible)
    value, freq = counts.most_common(1)[0]
    assert freq * 2 > len(visible), "anomaly correction must have a strict visible majority"
    return int(value)


def _correct_anomaly(grid: np.ndarray, marker_mask: np.ndarray) -> np.ndarray:
    suspicious = _suspicious_mask(grid, marker_mask)
    comps = _components_mask(suspicious)
    assert comps, "expected an anomalous visible region"

    out = grid.copy()
    updates: dict[tuple[int, int], int] = {}
    for comp in comps:
        for r, c in comp:
            updates[(int(r), int(c))] = _majority_visible_value(grid, marker_mask, int(r), int(c))
    for (r, c), value in updates.items():
        out[r, c] = int(value)
    return out


def _fill_marker(grid: np.ndarray, marker_mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for r, c in np.argwhere(marker_mask):
        candidates: list[int] = []
        h, w = out.shape
        sr = h - 1 - int(r)
        sc = w - 1 - int(c)
        for rr, cc in ((int(r), sc), (sr, int(c)), (sr, sc)):
            if (rr, cc) == (int(r), int(c)):
                continue
            if not marker_mask[rr, cc]:
                candidates.append(int(out[rr, cc]))
        assert candidates, "every marker cell must have a visible symmetric counterpart"
        assert len(set(candidates)) == 1, "marker recovery requires consistent corrected mirrors"
        out[int(r), int(c)] = candidates[0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a doubly symmetric image by fixing one corrupted patch and then recovering a masked rectangle.

    Input structure:
    - A black background contains a red/blue/green pattern that should be symmetric across both the horizontal and vertical center axes.
    - One solid magenta rectangle hides part of that pattern.
    - One visible 3x3 patch has been altered to a non-symmetric red/blue/green pattern.

    Rule:
    - Find the solid magenta rectangle and treat it as unknown content.
    - For each visible cell, compare it with the other visible members of its horizontal/vertical symmetry orbit; any orbit with disagreement belongs to the anomalous patch family.
    - Correct every suspicious visible cell to the strict majority color among the currently visible members of its orbit.
    - After the visible anomaly is corrected, replace each magenta cell with the unique value seen at its horizontal, vertical, or 180-degree symmetric counterpart.

    Key insight:
    - The damaged 3x3 patch is the only place where a symmetry orbit disagrees, so majority voting over the visible orbit repairs it; only then do the masked cells have consistent mirror evidence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _box, marker_mask = _find_marker_rect(g)
    corrected = _correct_anomaly(g, marker_mask)
    return _fill_marker(corrected, marker_mask)


def _paint_rect(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    ph = random.randint(1, min(3, h))
    pw = random.randint(1, min(3, w))
    r0 = random.randint(0, h - ph)
    c0 = random.randint(0, w - pw)
    canvas[r0 : r0 + ph, c0 : c0 + pw] = int(color)


def _paint_line(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    if random.random() < 0.5:
        pw = random.randint(2, min(4, w))
        r0 = random.randint(0, h - 1)
        c0 = random.randint(0, w - pw)
        canvas[r0, c0 : c0 + pw] = int(color)
    else:
        ph = random.randint(2, min(4, h))
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - 1)
        canvas[r0 : r0 + ph, c0] = int(color)


def _paint_l(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    ph = random.randint(2, min(3, h))
    pw = random.randint(2, min(3, w))
    r0 = random.randint(0, h - ph)
    c0 = random.randint(0, w - pw)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if "t" in corner:
        canvas[r0, c0 : c0 + pw] = int(color)
    else:
        canvas[r0 + ph - 1, c0 : c0 + pw] = int(color)
    if "l" in corner:
        canvas[r0 : r0 + ph, c0] = int(color)
    else:
        canvas[r0 : r0 + ph, c0 + pw - 1] = int(color)


def _paint_plus(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    r = random.randint(0, h - 1)
    c = random.randint(0, w - 1)
    canvas[r, c] = int(color)
    if r > 0:
        canvas[r - 1, c] = int(color)
    if r + 1 < h:
        canvas[r + 1, c] = int(color)
    if c > 0:
        canvas[r, c - 1] = int(color)
    if c + 1 < w:
        canvas[r, c + 1] = int(color)


def _build_clean_grid() -> np.ndarray | None:
    h = random.randrange(13, 20, 2)
    w = random.randrange(13, 20, 2)
    domain = np.zeros((h // 2 + 1, w // 2 + 1), dtype=int)

    painters = (_paint_rect, _paint_line, _paint_l, _paint_plus)
    for color in PALETTE:
        for _ in range(random.randint(2, 4)):
            random.choice(painters)(domain, color)
    for _ in range(random.randint(4, 7)):
        random.choice(painters)(domain, random.choice(PALETTE))

    rows = np.minimum(np.arange(h), np.arange(h)[::-1])
    cols = np.minimum(np.arange(w), np.arange(w)[::-1])
    full = domain[rows[:, None], cols[None, :]].astype(int)

    colors = {int(v) for v in np.unique(full) if int(v) != BG}
    nonzero = int(np.count_nonzero(full))
    if len(colors) < 2 or nonzero < 20 or nonzero > (h * w * 3) // 5:
        return None
    return full


def _bbox_slice(box: tuple[int, int, int, int]) -> tuple[slice, slice]:
    r0, c0, r1, c1 = map(int, box)
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _mirror_bboxes(
    box: tuple[int, int, int, int], shape: tuple[int, int]
) -> dict[str, tuple[int, int, int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, box)
    return {
        "ul": (r0, c0, r1, c1),
        "ur": (r0, w - 1 - c1, r1, w - 1 - c0),
        "bl": (h - 1 - r1, c0, h - 1 - r0, c1),
        "br": (h - 1 - r1, w - 1 - c1, h - 1 - r0, w - 1 - c0),
    }


def _contains_box(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_distinct_values(grid: np.ndarray, box: tuple[int, int, int, int]) -> int:
    rs, cs = _bbox_slice(box)
    return len(set(map(int, grid[rs, cs].flat)))


def _make_corrupted_patch(correct: np.ndarray) -> np.ndarray:
    bad = np.empty_like(correct)
    for r in range(correct.shape[0]):
        for c in range(correct.shape[1]):
            choices = [color for color in PALETTE if color != int(correct[r, c])]
            if not choices:
                choices = list(PALETTE)
            bad[r, c] = random.choice(choices)
    if len(set(map(int, bad.flat))) == 1:
        rr = random.randrange(bad.shape[0])
        cc = random.randrange(bad.shape[1])
        bad[rr, cc] = random.choice([color for color in PALETTE if color != int(bad[rr, cc])])
    assert not np.array_equal(bad, correct)
    return bad


def _marker_has_axis_overlap(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    center_r = shape[0] // 2
    center_c = shape[1] // 2
    r0, c0, r1, c1 = box
    return (r0 <= center_r <= r1) or (c0 <= center_c <= c1)


def _marker_crosses_both_axes(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    center_r = shape[0] // 2
    center_c = shape[1] // 2
    r0, c0, r1, c1 = box
    return (r0 <= center_r <= r1) and (c0 <= center_c <= c1)


def _visible_mirror_values_for_marker(
    grid: np.ndarray, marker_mask: np.ndarray, r: int, c: int
) -> list[int]:
    h, w = grid.shape
    sr = h - 1 - int(r)
    sc = w - 1 - int(c)
    vals: list[int] = []
    for rr, cc in ((int(r), sc), (sr, int(c)), (sr, sc)):
        if (rr, cc) == (int(r), int(c)):
            continue
        if not marker_mask[rr, cc]:
            vals.append(int(grid[rr, cc]))
    return vals


def _choose_anomaly_and_marker(
    clean: np.ndarray,
) -> tuple[dict[str, tuple[int, int, int, int]], str, tuple[int, int, int, int], str] | None:
    h, w = clean.shape
    center_r = h // 2
    center_c = w // 2

    family_candidates: list[dict[str, tuple[int, int, int, int]]] = []
    for r0 in range(0, center_r - ANOMALY_SIZE + 1):
        for c0 in range(0, center_c - ANOMALY_SIZE + 1):
            box = (r0, c0, r0 + ANOMALY_SIZE - 1, c0 + ANOMALY_SIZE - 1)
            family = _mirror_bboxes(box, clean.shape)
            if len({family[key] for key in family}) != 4:
                continue
            patch = clean[r0 : r0 + ANOMALY_SIZE, c0 : c0 + ANOMALY_SIZE]
            if int(np.count_nonzero(patch)) < 3:
                continue
            if len(set(map(int, patch.flat))) < 2:
                continue
            family_candidates.append(family)
    random.shuffle(family_candidates)

    desired_branch = random.random() < 0.5
    for family in family_candidates:
        keys = list(family)
        random.shuffle(keys)
        for corrupted_key in keys:
            other_keys = [key for key in family if key != corrupted_key]
            random.shuffle(other_keys)
            marker_options: list[tuple[tuple[int, int, int, int], str]] = []
            fallback_options: list[tuple[tuple[int, int, int, int], str]] = []
            for marker_key in other_keys:
                protected = [family[key] for key in family if key != marker_key]
                for mh, mw in MASK_SIZES:
                    for r0 in range(h - mh + 1):
                        for c0 in range(w - mw + 1):
                            box = (r0, c0, r0 + mh - 1, c0 + mw - 1)
                            if _marker_crosses_both_axes(box, clean.shape):
                                continue
                            if not _contains_box(box, family[marker_key]):
                                continue
                            if any(_boxes_intersect(box, other) for other in protected):
                                continue
                            if len(set(map(int, clean[_bbox_slice(box)].flat))) < 2:
                                continue
                            if int(np.count_nonzero(clean[_bbox_slice(box)])) < 5:
                                continue
                            pair = (box, marker_key)
                            if _marker_has_axis_overlap(box, clean.shape):
                                marker_options.append(pair)
                            else:
                                fallback_options.append(pair)
            if desired_branch and marker_options:
                return family, corrupted_key, *random.choice(marker_options)
            combined = marker_options + fallback_options
            if combined:
                return family, corrupted_key, *random.choice(combined)
    return None


def _semantic_ok(
    inp: np.ndarray,
    clean: np.ndarray,
    family: dict[str, tuple[int, int, int, int]],
    corrupted_key: str,
    marker_box: tuple[int, int, int, int],
    marker_key: str,
) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, clean):
        return False

    found_box, marker_mask = _find_marker_rect(inp)
    if found_box != marker_box:
        return False

    suspicious = _suspicious_mask(inp, marker_mask)
    comps = _components_mask(suspicious)
    if len(comps) < 2:
        return False

    corrupted_box = family[corrupted_key]
    rs, cs = _bbox_slice(corrupted_box)
    if not bool(np.all(suspicious[rs, cs])):
        return False

    ambiguous_cells = 0
    marker_rs, marker_cs = _bbox_slice(marker_box)
    for r in range(marker_box[0], marker_box[2] + 1):
        for c in range(marker_box[1], marker_box[3] + 1):
            vals = _visible_mirror_values_for_marker(inp, marker_mask, r, c)
            if vals and len(set(vals)) > 1:
                ambiguous_cells += 1
    if ambiguous_cells == 0:
        return False

    corrected_visible = _correct_anomaly(inp, marker_mask)
    overlap_box = family[marker_key]
    overlap_rs, overlap_cs = _bbox_slice(overlap_box)
    if not bool(np.all(marker_mask[overlap_rs, overlap_cs])):
        return False
    for r in range(marker_box[0], marker_box[2] + 1):
        for c in range(marker_box[1], marker_box[3] + 1):
            vals = _visible_mirror_values_for_marker(corrected_visible, marker_mask, r, c)
            if not vals or len(set(vals)) != 1:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean_grid()
    if clean is None:
        return None

    choice = _choose_anomaly_and_marker(clean)
    if choice is None:
        return None
    family, corrupted_key, marker_box, marker_key = choice

    inp = clean.copy()
    corrupted_box = family[corrupted_key]
    rs, cs = _bbox_slice(corrupted_box)
    inp[rs, cs] = _make_corrupted_patch(clean[rs, cs])

    marker_rs, marker_cs = _bbox_slice(marker_box)
    inp[marker_rs, marker_cs] = MARKER

    if not _semantic_ok(inp, clean, family, corrupted_key, marker_box, marker_key):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
