import numpy as np


BLACK = 0
GREY = 5


def _norm(cells):
    cells = list(cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rotate(shape, turns):
    out = tuple(shape)
    for _ in range(turns % 4):
        out = _norm((c, -r) for r, c in out)
    return out


def _rotations(shape):
    return [_rotate(shape, k) for k in range(4)]


def _canonical(shape):
    return min(_rotations(shape))


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate colored upper keys into matching lower lock cavities.

    Input structure:
    The black grid is divided by one complete grey horizontal line. Above it are
    differently colored connected asymmetric key shapes. Below it are grey lock
    blocks whose enclosed black cavities have the shape of rotated keys.
    Rule:
    Extract each upper key as a shape and color, find the unique lower enclosed
    black cavity matching one of its four rotations, fill that cavity with the
    key color, and remove the original upper keys back to black. Grey separator
    and lock pixels are preserved.
    Key insight:
    Matching is geometric rather than positional or color based, and each key
    must be tested under all four rotations to find its lock.
    """
    grid = np.asarray(grid)
    out = grid.copy()

    full_grey_rows = np.where(np.all(grid == GREY, axis=1))[0]
    if len(full_grey_rows) == 0:
        return out
    sep = int(full_grey_rows[0])

    upper = grid[:sep]
    key_mask = (upper != BLACK) & (upper != GREY)
    keys = []
    for comp in _components(key_mask):
        color = int(upper[comp[0]])
        shape = _norm(comp)
        keys.append({"color": color, "shape": shape})

    lower = grid[sep + 1 :]
    cavity_mask = lower == BLACK
    cavities = []
    for comp in _components(cavity_mask):
        touches_edge = any(
            r == 0 or c == 0 or r == lower.shape[0] - 1 or c == lower.shape[1] - 1
            for r, c in comp
        )
        if touches_edge:
            continue
        cavities.append(comp)

    used_keys = set()
    for comp in cavities:
        cav_shape = _norm(comp)
        matches = []
        for i, key in enumerate(keys):
            if i in used_keys:
                continue
            for turns in range(4):
                if _rotate(key["shape"], turns) == cav_shape:
                    matches.append(i)
                    break
        if len(matches) == 1:
            key_i = matches[0]
            color = keys[key_i]["color"]
            used_keys.add(key_i)
            for r, c in comp:
                out[sep + 1 + r, c] = color

    out[:sep][(out[:sep] != BLACK) & (out[:sep] != GREY)] = BLACK
    return out


def _random_shape(rng):
    for _ in range(1000):
        target = int(rng.integers(4, 8))
        cells = {(0, 0)}
        while len(cells) < target:
            base = list(cells)[int(rng.integers(len(cells)))]
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            trial = set(cells)
            trial.add((base[0] + dr, base[1] + dc))
            shape = _norm(trial)
            h, w = _dims(shape)
            if h <= 5 and w <= 5:
                cells = set(shape)
        shape = _norm(cells)
        if len(set(_rotations(shape))) == 4:
            return shape
    return ((0, 0), (1, 0), (1, 1), (1, 2), (2, 0))


def _can_place(occupied, r, c, h, w, pad):
    rows, cols = occupied.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(rows, r + h + pad)
    c1 = min(cols, c + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark(occupied, r, c, h, w):
    occupied[r : r + h, c : c + w] = True


def _place_rect(rng, occupied, h, w, pad=1, keep_edge=False):
    rows, cols = occupied.shape
    min_r = 1 if keep_edge else 0
    min_c = 1 if keep_edge else 0
    max_r = rows - h - (1 if keep_edge else 0)
    max_c = cols - w - (1 if keep_edge else 0)
    if max_r < min_r or max_c < min_c:
        return None
    for _ in range(250):
        r = int(rng.integers(min_r, max_r + 1))
        c = int(rng.integers(min_c, max_c + 1))
        if _can_place(occupied, r, c, h, w, pad):
            _mark(occupied, r, c, h, w)
            return r, c
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]

    for _ in range(300):
        count = int(rng.integers(2, 5))
        rng.shuffle(colors)
        keys = []
        seen = set()
        for i in range(count):
            for _ in range(200):
                shape = _rotate(_random_shape(rng), int(rng.integers(4)))
                canon = _canonical(shape)
                if canon not in seen:
                    seen.add(canon)
                    keys.append({"shape": shape, "color": colors[i]})
                    break
        if len(keys) != count:
            continue

        cavities = [_rotate(k["shape"], int(rng.integers(4))) for k in keys]
        key_sizes = [_dims(k["shape"]) for k in keys]
        lock_sizes = [(h + 2, w + 2) for h, w in (_dims(s) for s in cavities)]

        min_upper = max(h for h, _ in key_sizes) + 2
        min_lower = max(h for h, _ in lock_sizes) + 2
        min_width = max(max(w for _, w in key_sizes), max(w for _, w in lock_sizes)) + 2

        height_low = max(15, min_upper + min_lower + 1)
        if height_low > 30:
            continue
        height = int(rng.integers(height_low, 31))
        sep_min = min_upper
        sep_max = height - min_lower - 1
        if sep_max < sep_min:
            continue
        sep = int(rng.integers(sep_min, sep_max + 1))
        lower_h = height - sep - 1

        width_low = max(14, min_width)
        if width_low > 30:
            continue
        width = int(rng.integers(width_low, 31))

        grid = np.zeros((height, width), dtype=int)
        grid[sep, :] = GREY

        upper_occ = np.zeros((sep, width), dtype=bool)
        lower_occ = np.zeros((lower_h, width), dtype=bool)

        order = list(range(count))
        rng.shuffle(order)
        ok = True
        key_positions = [None] * count
        for idx in order:
            h, w = key_sizes[idx]
            pos = _place_rect(rng, upper_occ, h, w, pad=1, keep_edge=True)
            if pos is None:
                ok = False
                break
            key_positions[idx] = pos
        if not ok:
            continue

        lock_positions = [None] * count
        rng.shuffle(order)
        for idx in order:
            h, w = lock_sizes[idx]
            pos = _place_rect(rng, lower_occ, h, w, pad=1, keep_edge=True)
            if pos is None:
                ok = False
                break
            lock_positions[idx] = pos
        if not ok:
            continue

        for idx, key in enumerate(keys):
            r0, c0 = key_positions[idx]
            for r, c in key["shape"]:
                grid[r0 + r, c0 + c] = key["color"]

        for idx, cavity in enumerate(cavities):
            lr, lc = lock_positions[idx]
            bh, bw = lock_sizes[idx]
            ar = sep + 1 + lr
            grid[ar : ar + bh, lc : lc + bw] = GREY
            for r, c in cavity:
                grid[ar + 1 + r, lc + 1 + c] = BLACK

        return grid

    raise RuntimeError("failed to generate a valid input")
