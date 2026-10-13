import numpy as np


_BG = 0
_BLUE = 1
_GRAY = 5

_generate_calls = 0


def _pattern_sym(sym):
    # returns a connected 3-8 pixel pattern with chosen symmetry
    p = np.zeros((5, 5), dtype=int)
    if sym == "V":
        coords = [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)]
    else:
        coords = [(1, 1), (3, 1), (2, 2), (1, 3), (3, 3)]
    for y, x in coords:
        p[y, x] = 1
    p[2, 2] = 1
    return p


def _anomaly_from_base(rng, base):
    p = base.copy()
    ones = list(zip(*np.where(base == 1)))
    move_count = 1 + rng.randint(0, 2)
    for _ in range(move_count):
        from_idx = rng.choice(len(ones))
        y, x = ones[from_idx]
        # move by one cell inside 5x5
        moves = [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
        moves = [(yy, xx) for yy, xx in moves if 0 <= yy < 5 and 0 <= xx < 5]
        ty, tx = moves[rng.randint(0, len(moves))]
        if p[ty, tx] == 0:
            p[ty, tx] = 1
            p[y, x] = 0
            ones = list(zip(*np.where(p == 1)))
    return p


def _infer_axis(pattern):
    h, w = pattern.shape
    pts = set(zip(*np.where(pattern == _BLUE)))
    vert = 0
    hori = 0
    for y, x in pts:
        if (y, w - 1 - x) in pts:
            vert += 1
        if (h - 1 - y, x) in pts:
            hori += 1
    if vert == len(pts) and hori == len(pts):
        return "V", w // 2
    if vert == len(pts):
        return "V", w // 2
    return "H", h // 2


def _apply_axis_completion(motif, axis):
    h, w = motif.shape
    atype, axis_pos = axis
    ones = set(zip(*np.where(motif == _BLUE)))
    completed = set(ones)
    for y, x in ones:
        if atype == "V":
            mx, my = w - 1 - x, y
        else:
            mx, my = x, h - 1 - y
        if (my, mx) not in completed:
            completed.add((my, mx))
    # erase extras
    out = np.zeros_like(motif)
    for y, x in completed:
        if motif[y, x] == _BLUE:
            out[y, x] = _BLUE
        else:
            # mirrored from existing pixel
            out[y, x] = _BLUE
    # remove pixels not backed by original/mirror relation
    for y, x in set(zip(*np.where(motif == _BLUE))):
        if out[y, x] != _BLUE:
            out[y, x] = _BLUE
    return out


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(10000 + _generate_calls)
    _generate_calls += 1

    unit = 5
    R = 3
    C = 3
    h = R * unit + (R - 1)
    w = C * unit + (C - 1)
    grid = np.zeros((h, w), dtype=int)

    # draw separators
    for r in range(R - 1):
        grid[(r + 1) * unit + r, :] = _GRAY
    for c in range(C - 1):
        grid[:, (c + 1) * unit + c] = _GRAY

    for r in range(R):
        base_sym = rng.choice(["V", "H"])
        base = _pattern_sym(base_sym)
        anomaly_col = rng.randint(0, C)
        for c in range(C):
            top = r * (unit + 1)
            left = c * (unit + 1)
            patt = base
            if c == anomaly_col:
                patt = _anomaly_from_base(rng, base)
                patt = patt
            patt = patt * _BLUE
            for y in range(unit):
                for x in range(unit):
                    grid[top + y, left + x] = patt[y, x]

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """1) In each pattern-row, detect the single abnormal cell block among units.
2) Use the majority unit axis (vertical/horizontal symmetry) as repair axis.
3) Extract all abnormal blocks into one-column grid preserving original row order.
4) Apply symmetry completion using borrowed axis and output repaired column."""

    g = np.array(grid)
    unit = 5
    rows = (g.shape[0] + 1) // (unit + 1)

    out_h = 2 * unit
    out = np.zeros((out_h, unit), dtype=int)
    if g.shape[0] >= 2 * unit and g.shape[1] >= unit:
        out = np.zeros((rows * unit + (rows - 1), unit), dtype=int)
        # fill row separators in output
        for r in range(rows - 1):
            out[(r + 1) * unit + r, :] = _GRAY

    row_units = []
    for r in range(rows):
        pats = []
        base_counts = {}
        for c in range(3):
            top = r * (unit + 1)
            left = c * (unit + 1)
            patt = g[top : top + unit, left : left + unit].copy()
            pat_key = tuple(patt.flatten())
            base_counts[pat_key] = base_counts.get(pat_key, 0) + 1
            pats.append(patt)
        # abnormal is unique count 1
        keys = list(base_counts.keys())
        if not keys:
            continue
        if max(base_counts.values()) == 1:
            # all unique: fallback use first as anomaly
            anomaly_idx = 0
        else:
            anomaly_key = min(base_counts.items(), key=lambda kv: kv[1])[0]
            anomaly_idx = [i for i, p in enumerate(pats) if tuple(p.flatten()) == anomaly_key][0]
        majority_key = max(base_counts.items(), key=lambda kv: kv[1])[0]
        majority = np.array(majority_key, dtype=int).reshape(unit, unit)

        axis = _infer_axis(majority)
        repaired = np.zeros((unit, unit), dtype=int)
        an = pats[anomaly_idx]
        an_points = set(zip(*np.where(an == _BLUE)))
        for y, x in an_points:
            repaired[y, x] = _BLUE
            if axis[0] == "V":
                mx, my = unit - 1 - x, y
            else:
                mx, my = x, unit - 1 - y
            if 0 <= my < unit and 0 <= mx < unit:
                repaired[my, mx] = _BLUE

        # clear extras
        for y, x in zip(*np.where(repaired == _BLUE)):
            if majority[y, x] == 0:
                repaired[y, x] = _BLUE

        row_units.append(repaired)

    # compose single-column output
    out = np.zeros((rows * unit + (rows - 1), unit), dtype=int)
    for r, patt in enumerate(row_units):
        top = r * (unit + 1)
        out[top : top + unit, :] = patt
        if r + 1 < rows:
            out[top + unit, :] = 0
    return out


if __name__ == "__main__":
    print(main(generate_input()).shape)
