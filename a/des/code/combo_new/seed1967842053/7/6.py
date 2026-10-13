import random
from collections import deque

import numpy as np


def _components(grid, bg):
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(cells):
    if not cells:
        return frozenset()
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return frozenset((r - mr, c - mc) for r, c in cells)


def _rotate(cells, k):
    pts = list(cells)
    for _ in range(k % 4):
        pts = [(c, -r) for r, c in pts]
        mr = min(r for r, _ in pts)
        mc = min(c for _, c in pts)
        pts = [(r - mr, c - mc) for r, c in pts]
    return frozenset(pts)


def _outside_dir(direction):
    if direction == "top":
        return (-1, 0)
    if direction == "bottom":
        return (1, 0)
    if direction == "left":
        return (0, -1)
    return (0, 1)


def _find_notches(grid, bg, color, cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    cellset = set(cells)
    full = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    holes = full - cellset
    if not holes:
        return []
    h, w = grid.shape
    seen = set()
    groups = []
    for p in holes:
        if p in seen:
            continue
        q = deque([p])
        seen.add(p)
        group = []
        while q:
            cr, cc = q.popleft()
            group.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                npnt = (cr + dr, cc + dc)
                if npnt in holes and npnt not in seen:
                    seen.add(npnt)
                    q.append(npnt)
        groups.append(group)

    found = []
    for group in groups:
        touches = []
        for gr, gc in group:
            if gr == r0:
                touches.append("top")
            if gr == r1:
                touches.append("bottom")
            if gc == c0:
                touches.append("left")
            if gc == c1:
                touches.append("right")
        sides = sorted(set(touches))
        if len(sides) != 1:
            continue
        side = sides[0]
        dr, dc = _outside_dir(side)
        opens = []
        for gr, gc in group:
            br, bc = gr + dr, gc + dc
            if not (r0 <= br <= r1 and c0 <= bc <= c1) or (br, bc) not in cellset:
                opens.append((gr, gc))
        if opens:
            found.append((side, frozenset(group)))
    return found


def _leaf_blocks(cells):
    s = set(cells)
    leaves = []
    for r, c in s:
        n = sum((r + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if n <= 1:
            leaves.append((r, c))
    blocks = set()
    for p in leaves:
        blocks.add(frozenset([p]))
    for r, c in cells:
        if (r, c + 1) in s:
            n1 = sum((r + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            n2 = sum((r + dr, c + 1 + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if n1 <= 2 and n2 <= 2:
                blocks.add(frozenset([(r, c), (r, c + 1)]))
        if (r + 1, c) in s:
            n1 = sum((r + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            n2 = sum((r + 1 + dr, c + dc) in s for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if n1 <= 2 and n2 <= 2:
                blocks.add(frozenset([(r, c), (r + 1, c)]))
    return sorted(blocks, key=lambda b: (len(b), sorted(b)))


def _split_key(cells):
    s = set(cells)
    best = None
    for handle in _leaf_blocks(cells):
        teeth = s - set(handle)
        if not teeth:
            continue
        start = next(iter(teeth))
        q = deque([start])
        seen = {start}
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if p in teeth and p not in seen:
                    seen.add(p)
                    q.append(p)
        if seen != teeth:
            continue
        contacts = []
        for hr, hc in handle:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (hr + dr, hc + dc)
                if p in teeth:
                    contacts.append(((hr, hc), p, (hr - p[0], hc - p[1])))
        if not contacts:
            continue
        score = (len(handle), -len(contacts))
        if best is None or score > best[0]:
            best = (score, frozenset(handle), frozenset(teeth), contacts[0][2])
    if best is None:
        return frozenset(), frozenset(cells), (0, 1)
    return best[1], best[2], best[3]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each colored key into the matching lock notch and erase its old copy.

    Input structure:
    A background grid contains several large solid rectangles, each with one
    background-colored notch cut into one edge. Separate small connected objects
    of the same colors are keys: a 1-cell or 2-cell handle attached to an
    irregular toothed part.

    Rule:
    Pair each key with the rectangle of the same color. Try the four rotations
    of the toothed part and choose the one whose normalized shape exactly equals
    the rectangle notch. Fill the notch with that color, erase the original key,
    and redraw only the handle outside the notch along the opening direction,
    preserving the handle length.

    Key insight:
    Color identifies the lock/key pair, but the decisive evidence is the
    background notch shape: the key teeth must be mentally rotated until their
    complement exactly restores the colored rectangle.
    """
    arr = np.array(grid, copy=True)
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    comps = _components(arr, bg)

    by_color = {}
    for color, cells in comps:
        by_color.setdefault(color, []).append(cells)

    for color, groups in by_color.items():
        rects = []
        keys = []
        for cells in groups:
            r0, c0, r1, c1 = _bbox(cells)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            holes = area - len(cells)
            if len(cells) >= 12 and holes > 0:
                notches = _find_notches(arr, bg, color, cells)
                if notches:
                    rects.append((cells, notches[0]))
                else:
                    keys.append(cells)
            else:
                keys.append(cells)

        unused_keys = list(keys)
        for rect_cells, (side, notch) in rects:
            target = _norm(notch)
            chosen = None
            for ki, key_cells in enumerate(unused_keys):
                handle, teeth, handle_vec = _split_key(key_cells)
                for k in range(4):
                    if _rotate(_norm(teeth), k) == target:
                        chosen = (ki, key_cells, handle, teeth, handle_vec, k)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                continue

            ki, key_cells, handle, teeth, handle_vec, rot_k = chosen
            unused_keys.pop(ki)
            for r, c in key_cells:
                arr[r, c] = bg
            for r, c in notch:
                arr[r, c] = color

            dr, dc = _outside_dir(side)
            mouth = sorted(
                (p for p in notch if (p[0] + dr, p[1] + dc) not in notch),
                key=lambda p: (p[1], p[0]) if dr else (p[0], p[1]),
            )
            if not mouth:
                continue
            anchor = mouth[len(mouth) // 2]
            handle_len = len(handle) if handle else 1
            for i in range(1, handle_len + 1):
                rr, cc = anchor[0] + dr * i, anchor[1] + dc * i
                if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]:
                    arr[rr, cc] = color
    return arr


def _connected(cells):
    if not cells:
        return False
    s = set(cells)
    q = deque([next(iter(s))])
    seen = {next(iter(s))}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (r + dr, c + dc)
            if p in s and p not in seen:
                seen.add(p)
                q.append(p)
    return seen == s


def _make_notch(depth, width, side):
    while True:
        if side in ("top", "bottom"):
            cells = {(0, c) for c in range(width)}
            for c in range(width):
                d = random.randint(1, depth)
                for r in range(d):
                    cells.add((r, c))
            for _ in range(random.randint(0, width)):
                cells.add((random.randint(0, depth - 1), random.randint(0, width - 1)))
        else:
            cells = {(r, 0) for r in range(width)}
            for r in range(width):
                d = random.randint(1, depth)
                for c in range(d):
                    cells.add((r, c))
            for _ in range(random.randint(0, width)):
                cells.add((random.randint(0, width - 1), random.randint(0, depth - 1)))
        if _connected(cells) and len(cells) >= 3:
            return frozenset(cells)


def _place_notch(rect_r, rect_c, rh, rw, side, local):
    if side == "top":
        start = random.randint(1, rw - (max(c for _, c in local) + 1) - 1)
        return {(rect_r + r, rect_c + start + c) for r, c in local}
    if side == "bottom":
        maxr = max(r for r, _ in local)
        start = random.randint(1, rw - (max(c for _, c in local) + 1) - 1)
        return {(rect_r + rh - 1 - maxr + r, rect_c + start + c) for r, c in local}
    if side == "left":
        start = random.randint(1, rh - (max(r for r, _ in local) + 1) - 1)
        return {(rect_r + start + r, rect_c + c) for r, c in local}
    maxc = max(c for _, c in local)
    start = random.randint(1, rh - (max(r for r, _ in local) + 1) - 1)
    return {(rect_r + start + r, rect_c + rw - 1 - maxc + c) for r, c in local}


def _touches_existing(grid, cells, bg, gap=1):
    h, w = grid.shape
    for r, c in cells:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != bg:
                    return True
    return False


def _rotated_to_abs(shape, top, left, k):
    pts = _rotate(shape, k)
    return {(top + r, left + c) for r, c in pts}


def _candidate_key(teeth_shape, color, grid, bg):
    h, w = grid.shape
    for _ in range(300):
        k = random.randrange(4)
        teeth = _rotate(teeth_shape, k)
        maxr = max(r for r, _ in teeth)
        maxc = max(c for _, c in teeth)
        top = random.randint(1, h - maxr - 3)
        left = random.randint(1, w - maxc - 3)
        abs_teeth = {(top + r, left + c) for r, c in teeth}
        boundary = []
        for r, c in abs_teeth:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if p not in abs_teeth and 0 <= p[0] < h and 0 <= p[1] < w:
                    boundary.append((p, (dr, dc)))
        random.shuffle(boundary)
        for base, vec in boundary:
            handle_len = random.choice([1, 2])
            handle = {base}
            if handle_len == 2:
                handle.add((base[0] + vec[0], base[1] + vec[1]))
            all_cells = abs_teeth | handle
            if all(0 <= r < h and 0 <= c < w for r, c in all_cells) and not _touches_existing(grid, all_cells, bg, 1):
                return abs_teeth, handle
    return None


def _try_generate():
    bg = 0
    h = random.randint(19, 30)
    w = random.randint(19, 30)
    grid = np.full((h, w), bg, dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
    sides = ["top", "right", "bottom", "left"]
    random.shuffle(sides)

    placed = 0
    for i, color in enumerate(colors):
        side = sides[i % 4]
        for _ in range(120):
            depth = random.choice([2, 3])
            width_n = random.choice([2, 3, 4])
            rh = random.randint(max(5, width_n + 3), 8)
            rw = random.randint(max(5, width_n + 3), 9)
            if side in ("top", "bottom"):
                rw = max(rw, width_n + 3)
                local = _make_notch(depth, width_n, side)
            else:
                rh = max(rh, width_n + 3)
                local = _make_notch(depth, width_n, side)
            rr = random.randint(2, h - rh - 3)
            cc = random.randint(2, w - rw - 3)
            rect = {(r, c) for r in range(rr, rr + rh) for c in range(cc, cc + rw)}
            notch = _place_notch(rr, cc, rh, rw, side, local)
            colored = rect - notch
            if _touches_existing(grid, colored, bg, 1):
                continue
            teeth_shape = _norm(notch)
            key = _candidate_key(teeth_shape, color, grid, bg)
            if key is None:
                continue
            for r, c in colored:
                grid[r, c] = color
            for group in key:
                for r, c in group:
                    grid[r, c] = color
            placed += 1
            break
    if placed >= 2:
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _try_generate()
        if grid is not None:
            return grid
    return _fallback()


def _fallback():
    grid = np.zeros((24, 24), dtype=int)
    specs = [
        (2, 2, 6, 7, "top", {(0, 0), (0, 1), (1, 1), (0, 2), (1, 2)}, 1, 15, 2),
        (13, 12, 7, 7, "right", {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)}, 2, 3, 5),
    ]
    for rr, cc, rh, rw, side, local, color, kr, kc in specs:
        rect = {(r, c) for r in range(rr, rr + rh) for c in range(cc, cc + rw)}
        notch = _place_notch(rr, cc, rh, rw, side, local)
        for r, c in rect - notch:
            grid[r, c] = color
        shape = _norm(notch)
        teeth = _rotated_to_abs(shape, kr, kc, 1)
        for r, c in teeth:
            grid[r, c] = color
        ar, ac = min(teeth)
        grid[ar - 1, ac] = color
    return grid
