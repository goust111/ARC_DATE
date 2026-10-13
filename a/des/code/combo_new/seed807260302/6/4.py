from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    neigh = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _center_side_direction(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> int | None:
    r, c = marker
    r0, c0, r1, c1 = bbox
    matches: list[int] = []
    if r == r0 and 2 * c == c0 + c1:
        matches.append(UP)
    if c == c1 and 2 * r == r0 + r1:
        matches.append(RIGHT)
    if r == r1 and 2 * c == c0 + c1:
        matches.append(DOWN)
    if c == c0 and 2 * r == r0 + r1:
        matches.append(LEFT)
    return matches[0] if len(matches) == 1 else None


def _extreme_marker_direction(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> int | None:
    r, c = marker
    r0, c0, r1, c1 = bbox
    matches: list[int] = []
    if r == r0:
        matches.append(UP)
    if c == c1:
        matches.append(RIGHT)
    if r == r1:
        matches.append(DOWN)
    if c == c0:
        matches.append(LEFT)
    return matches[0] if len(matches) == 1 else None


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _find_envelopes(grid: np.ndarray) -> list[dict]:
    envelopes: list[dict] = []
    for comp in _components(grid == GREY, connectivity=4):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = _border_cells(bbox)
        if any(int(grid[r, c]) not in (GREY, MAROON) for r, c in border):
            continue
        markers = [(r, c) for r, c in border if int(grid[r, c]) == MAROON]
        if len(markers) != 1:
            continue
        if not bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK)):
            continue

        direction = _center_side_direction(markers[0], bbox)
        if direction is None:
            continue
        envelopes.append(
            {
                "bbox": bbox,
                "marker": markers[0],
                "direction": direction,
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
            }
        )
    return envelopes


def _find_stamps(grid: np.ndarray, frame_markers: set[tuple[int, int]]) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY)
    for r, c in frame_markers:
        mask[r, c] = False

    stamps: list[dict] = []
    for comp in _components(mask, connectivity=8):
        maroons = [(r, c) for r, c in comp if int(grid[r, c]) == MAROON]
        colored = [(r, c) for r, c in comp if int(grid[r, c]) not in (BLACK, GREY, MAROON)]
        if len(maroons) != 1 or not colored:
            continue

        bbox = _bbox(comp)
        direction = _extreme_marker_direction(maroons[0], bbox)
        if direction is None:
            continue

        r0, c0, r1, c1 = bbox
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = int(grid[r, c])
        stamps.append(
            {
                "bbox": bbox,
                "marker": maroons[0],
                "direction": direction,
                "patch": patch,
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
            }
        )
    return stamps


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, max(ar0, br0) - min(ar1, br1) - 1)
    col_gap = max(0, max(ac0, bc0) - min(ac1, bc1) - 1)
    return row_gap + col_gap


def _pair_stamps_to_envelopes(stamps: list[dict], envelopes: list[dict]) -> list[tuple[dict, dict]]:
    assert len(stamps) == len(envelopes) and stamps, "expected one stamp for each envelope"
    candidates: list[tuple[int, float, int, int]] = []
    for si, stamp in enumerate(stamps):
        sr, sc = stamp["center"]
        for ei, env in enumerate(envelopes):
            er, ec = env["center"]
            dist = _bbox_distance(stamp["bbox"], env["bbox"])
            candidates.append((dist, abs(sr - er) + abs(sc - ec), si, ei))

    used_stamps: set[int] = set()
    used_envs: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    for _dist, _cent_dist, si, ei in sorted(candidates):
        if si in used_stamps or ei in used_envs:
            continue
        used_stamps.add(si)
        used_envs.add(ei)
        pairs.append((stamps[si], envelopes[ei]))
        if len(pairs) == len(stamps):
            break

    assert len(pairs) == len(stamps), "failed to pair all stamps"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each nearby stamp to the matching envelope direction and print it inside the envelope.

    Input structure:
    - The background is black.
    - Several grey hollow rectangular envelope frames have black interiors and one maroon pixel at the center of one frame side.
    - Near each envelope is one small non-grey, non-black stamp component containing a single maroon orientation pixel.
    - Stamps share the same asymmetric shape up to rotation, while their visible colors may differ.

    Rule:
    - Pair each stamp with the nearest envelope by spatial distance.
    - The envelope direction is the side containing its centered maroon marker; the stamp direction is the extreme side of its maroon marker within the stamp bounding box.
    - Rotate the whole stamp by 0, 90, 180, or 270 degrees so the two directions match.
    - Center the rotated stamp footprint in the envelope interior and draw only its colored pixels, omitting the stamp's maroon marker.
    - Preserve the original stamps, envelope frames, and envelope maroon markers.

    Key insight:
    - The task combines nearest-neighbor pairing, directional marker reading, and a rotation-normalized stamp copy into each frame's empty interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    envelopes = _find_envelopes(g)
    assert envelopes, "expected at least one envelope"
    frame_markers = {env["marker"] for env in envelopes}
    stamps = _find_stamps(g, frame_markers)
    pairs = _pair_stamps_to_envelopes(stamps, envelopes)

    out = g.copy()
    for stamp, env in pairs:
        k = (int(stamp["direction"]) - int(env["direction"])) % 4
        rotated = np.rot90(stamp["patch"], k)
        mask = (rotated != BLACK) & (rotated != GREY) & (rotated != MAROON)

        r0, c0, r1, c1 = env["bbox"]
        inner_h, inner_w = r1 - r0 - 1, c1 - c0 - 1
        ph, pw = rotated.shape
        assert ph <= inner_h and pw <= inner_w, "stamp must fit inside paired envelope"
        top = r0 + 1 + (inner_h - ph) // 2
        left = c0 + 1 + (inner_w - pw) // 2

        for rr, cc in np.argwhere(mask):
            tr, tc = top + int(rr), left + int(cc)
            assert out[tr, tc] == BLACK, "envelope interior must be blank before stamping"
            out[tr, tc] = int(rotated[rr, cc])

    return out


# --------------- generator ---------------

COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
STAMP_SPECS = (
    ((5, 4), (0, 2), ((1, 1), (1, 2), (2, 0), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2))),
    ((5, 5), (0, 2), ((1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1))),
    ((4, 5), (0, 2), ((1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (3, 2))),
)
GEN_RETRIES = 400


def _crop_nonzero(arr: np.ndarray) -> np.ndarray:
    pts = np.argwhere(arr != BLACK)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _make_stamp(spec, primary: int, secondary: int | None) -> np.ndarray:
    shape, marker, colored = spec
    arr = np.zeros(shape, dtype=int)
    arr[marker] = MAROON
    for i, (r, c) in enumerate(colored):
        arr[r, c] = secondary if secondary is not None and i % 3 == 1 else primary
    return _crop_nonzero(arr)


def _rotate_stamp_to_direction(arr: np.ndarray, direction: int) -> np.ndarray:
    return _crop_nonzero(np.rot90(arr, (-direction) % 4))


def _draw_frame(grid: np.ndarray, r: int, c: int, h: int, w: int, direction: int) -> tuple[int, int, int, int]:
    grid[r, c : c + w] = GREY
    grid[r + h - 1, c : c + w] = GREY
    grid[r : r + h, c] = GREY
    grid[r : r + h, c + w - 1] = GREY

    if direction == UP:
        mr, mc = r, c + w // 2
    elif direction == RIGHT:
        mr, mc = r + h // 2, c + w - 1
    elif direction == DOWN:
        mr, mc = r + h - 1, c + w // 2
    else:
        mr, mc = r + h // 2, c
    grid[mr, mc] = MAROON
    return r, c, r + h - 1, c + w - 1


def _draw_stamp(grid: np.ndarray, stamp: np.ndarray, r: int, c: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(stamp != BLACK)
    for rr, cc in pts:
        tr, tc = r + int(rr), c + int(cc)
        if int(grid[tr, tc]) != BLACK:
            raise ValueError("stamp placement overlap")
        grid[tr, tc] = int(stamp[rr, cc])
    return r, c, r + stamp.shape[0] - 1, c + stamp.shape[1] - 1


def _slot_bounds(h: int, w: int) -> list[tuple[int, int, int, int]]:
    return [
        (0, 0, h // 2 - 1, w // 2 - 1),
        (0, w // 2, h // 2 - 1, w - 1),
        (h // 2, 0, h - 1, w // 2 - 1),
        (h // 2, w // 2, h - 1, w - 1),
    ]


def _place_pair(grid: np.ndarray, slot: tuple[int, int, int, int], stamp: np.ndarray, env_dir: int) -> bool:
    sr0, sc0, sr1, sc1 = slot
    slot_h, slot_w = sr1 - sr0 + 1, sc1 - sc0 + 1
    sh, sw = stamp.shape

    for _ in range(80):
        side = random.randrange(4)
        gap = random.choice((1, 2))
        frame_sizes = []
        for fh in (7, 9):
            for fw in (7, 9):
                if side in (UP, DOWN):
                    ph = sh + gap + fh
                    pw = max(sw, fw) + random.randrange(0, 3)
                else:
                    ph = max(sh, fh) + random.randrange(0, 3)
                    pw = sw + gap + fw
                if ph <= slot_h and pw <= slot_w:
                    frame_sizes.append((fh, fw, ph, pw))
        if not frame_sizes:
            continue

        fh, fw, ph, pw = random.choice(frame_sizes)
        pr = random.randint(sr0, sr1 - ph + 1)
        pc = random.randint(sc0, sc1 - pw + 1)

        if side == UP:
            tr = pr
            fr = pr + sh + gap
            tc = pc + random.randint(0, pw - sw)
            fc = pc + random.randint(0, pw - fw)
        elif side == DOWN:
            fr = pr
            tr = pr + fh + gap
            tc = pc + random.randint(0, pw - sw)
            fc = pc + random.randint(0, pw - fw)
        elif side == LEFT:
            tc = pc
            fc = pc + sw + gap
            tr = pr + random.randint(0, ph - sh)
            fr = pr + random.randint(0, ph - fh)
        else:
            fc = pc
            tc = pc + fw + gap
            tr = pr + random.randint(0, ph - sh)
            fr = pr + random.randint(0, ph - fh)

        try:
            _draw_frame(grid, fr, fc, fh, fw, env_dir)
            _draw_stamp(grid, stamp, tr, tc)
        except ValueError:
            return False
        return True
    return False


def _pairing_is_clear(grid: np.ndarray) -> bool:
    envs = _find_envelopes(grid)
    stamps = _find_stamps(grid, {env["marker"] for env in envs})
    if len(envs) < 2 or len(envs) != len(stamps):
        return False

    nearest_envs: list[int] = []
    for stamp in stamps:
        dists = sorted((_bbox_distance(stamp["bbox"], env["bbox"]), ei) for ei, env in enumerate(envs))
        if len(dists) > 1 and dists[0][0] >= dists[1][0]:
            return False
        nearest_envs.append(dists[0][1])
    if len(set(nearest_envs)) != len(nearest_envs):
        return False

    nearest_stamps: list[int] = []
    for env in envs:
        dists = sorted((_bbox_distance(stamp["bbox"], env["bbox"]), si) for si, stamp in enumerate(stamps))
        if len(dists) > 1 and dists[0][0] >= dists[1][0]:
            return False
        nearest_stamps.append(dists[0][1])
    return len(set(nearest_stamps)) == len(nearest_stamps)


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(2, 4)
    h = random.randint(26, 30)
    w = random.randint(26, 30)
    grid = np.zeros((h, w), dtype=int)

    spec = random.choice(STAMP_SPECS)
    slots = random.sample(_slot_bounds(h, w), n_pairs)
    primary_colors = random.sample(COLOR_POOL, n_pairs)

    for slot, primary in zip(slots, primary_colors):
        secondary = None
        if random.random() < 0.45:
            secondary = random.choice([c for c in COLOR_POOL if c != primary])
        stamp_dir = random.randrange(4)
        env_dir = random.randrange(4)
        base_stamp = _make_stamp(spec, primary, secondary)
        stamp = _rotate_stamp_to_direction(base_stamp, stamp_dir)
        if not _place_pair(grid, slot, stamp, env_dir):
            return None

    if not _pairing_is_clear(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30 or bool(np.array_equal(out, grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stamp-envelope input")
