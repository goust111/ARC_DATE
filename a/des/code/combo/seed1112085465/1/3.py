import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches_edge = True
                    q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return int(holes)


def _control_runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    c = 0
    while c < row.shape[0]:
        color = int(row[c])
        d = c + 1
        while d < row.shape[0] and int(row[d]) == color:
            d += 1
        if color != 0:
            runs.append((int(c), int(d), int(color)))
        c = d
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each top-row block width as a vertical translation amount for the
    same-color object below it, and use hole-count parity to choose direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    body = g[1:, :]
    out = np.zeros_like(body)

    runs = _control_runs(g[0, :])
    assert runs

    for c0, c1, color in runs:
        seg_mask = body[:, c0:c1] == int(color)
        comps = _components(seg_mask)
        assert len(comps) == 1, "each control block must match one same-color object"

        comp = comps[0]
        holes = _hole_count(_coords_to_mask(comp))
        shift = int(c1 - c0)
        dr = shift if holes % 2 == 0 else -shift

        for r, c in comp:
            nr = int(r + dr)
            nc = int(c0 + c)
            if 0 <= nr < out.shape[0]:
                out[nr, nc] = int(color)

    return out


def _pick_hole_cells(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(interior)
    chosen: list[tuple[int, int]] = []
    for r, c in interior:
        if all(abs(r - rr) + abs(c - cc) != 1 for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == count:
                return chosen
    return None


def _make_object_mask(height: int, width: int, holes: int) -> np.ndarray | None:
    hole_cells = _pick_hole_cells(int(height), int(width), int(holes))
    if hole_cells is None:
        return None

    mask = np.ones((height, width), dtype=bool)
    for r, c in hole_cells:
        mask[r, c] = False

    border = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if r in (0, height - 1) or c in (0, width - 1)
    ]
    random.shuffle(border)
    carve_left = random.randint(0, min(3, len(border)))
    for r, c in border:
        if carve_left == 0:
            break
        if not mask[r, c]:
            continue
        candidate = mask.copy()
        candidate[r, c] = False
        if candidate.any() and len(_components(candidate)) == 1 and _hole_count(candidate) == holes:
            mask = candidate
            carve_left -= 1

    if len(_components(mask)) != 1 or _hole_count(mask) != holes:
        return None
    return mask


def _sample_segment_widths(count: int) -> list[int] | None:
    for _ in range(100):
        widths = [random.randint(1, 5) for _ in range(count)]
        if 9 <= sum(widths) <= 20 and any(w >= 3 for w in widths):
            return widths
    return None


def _choose_odd_flags(widths: list[int]) -> list[bool] | None:
    odd_capable = [i for i, w in enumerate(widths) if w >= 3]
    if not odd_capable:
        return None

    odd_flags = [False] * len(widths)
    for idx in odd_capable:
        odd_flags[idx] = random.random() < 0.5

    if not any(odd_flags[idx] for idx in odd_capable):
        odd_flags[random.choice(odd_capable)] = True
    if not any(not flag for flag in odd_flags):
        odd_flags[random.choice(odd_capable)] = False
    return odd_flags


def _sample_row(body_h: int, obj_h: int, shift: int, odd: bool) -> int | None:
    if odd:
        full = (max(1, shift), body_h - obj_h)
        clipped = (max(1, shift - obj_h + 1), min(shift - 1, body_h - obj_h))
    else:
        full = (1, body_h - obj_h - shift)
        clipped = (
            max(1, body_h - obj_h - shift + 1),
            min(body_h - obj_h, body_h - shift - 1),
        )

    full_ok = full[0] <= full[1]
    clipped_ok = clipped[0] <= clipped[1]
    if not full_ok and not clipped_ok:
        return None

    use_clipped = clipped_ok and random.random() < 0.25
    lo, hi = clipped if use_clipped else full
    if lo > hi:
        lo, hi = clipped
    if lo > hi:
        return None
    return int(random.randint(lo, hi))


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(3, 5)
    widths = _sample_segment_widths(n_segments)
    if widths is None:
        return None

    odd_flags = _choose_odd_flags(widths)
    if odd_flags is None or not any(odd_flags) or not any(not flag for flag in odd_flags):
        return None

    colors = random.sample(list(range(1, 10)), n_segments)
    specs: list[dict[str, object]] = []

    for seg_w, odd in zip(widths, odd_flags):
        if odd:
            obj_w = random.randint(3, seg_w)
            obj_h = random.randint(3, 8)
            holes = 1
        else:
            obj_w = random.randint(1, seg_w)
            obj_h = random.randint(3 if obj_w >= 3 else 2, 8)
            holes = 0
            can_make_two = obj_w >= 3 and obj_h >= 3 and _pick_hole_cells(obj_h, obj_w, 2) is not None
            if can_make_two and random.random() < 0.5:
                holes = 2

        mask = _make_object_mask(obj_h, obj_w, holes)
        if mask is None:
            return None
        specs.append(
            {
                "segment_width": int(seg_w),
                "odd": bool(odd),
                "holes": int(holes),
                "mask": mask,
            }
        )

    min_body_h = max(int(spec["mask"].shape[0]) + int(spec["segment_width"]) + 1 for spec in specs)
    body_h = random.randint(min_body_h, min_body_h + 5)
    total_w = int(sum(widths))
    grid = np.zeros((body_h + 1, total_w), dtype=int)

    cursor = 0
    for color, seg_w, spec in zip(colors, widths, specs):
        c0 = int(cursor)
        c1 = int(cursor + seg_w)
        grid[0, c0:c1] = int(color)

        mask = spec["mask"]
        obj_h, obj_w = map(int, mask.shape)
        col_offset = random.randint(0, seg_w - obj_w)
        row = _sample_row(body_h, obj_h, seg_w, bool(spec["odd"]))
        if row is None:
            return None

        for rr in range(obj_h):
            for cc in range(obj_w):
                if mask[rr, cc]:
                    grid[1 + row + rr, c0 + col_offset + cc] = int(color)

        cursor = c1

    if len(_control_runs(grid[0])) != n_segments:
        return None

    for c0, c1, color in _control_runs(grid[0]):
        comps = _components(grid[1:, c0:c1] == color)
        if len(comps) != 1:
            return None

    out = main(grid)
    if out.shape != (body_h, total_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
