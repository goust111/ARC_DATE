import random
from itertools import product

import numpy as np


BLACK = 0
GREY = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

PAIR_COUNT_RANGE = (1, 3)
UNMATCHED_COUNT_RANGE = (0, 2)
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
PLACEMENT_TRIES = 400
GEN_RETRIES = 300
BOX_PAD = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BLACK) -> None:
    h, w = patch.shape
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    sub[mask] = patch[mask]


def _canonical_sig(sig: tuple[int, ...]) -> tuple[int, ...]:
    rev = tuple(reversed(sig))
    return min(tuple(sig), rev)


_SIGNATURES: dict[int, list[tuple[int, ...]]] = {}
for _h in range(3, 6):
    opts: list[tuple[int, ...]] = []
    for bits in product((0, 1), repeat=_h):
        ones = int(sum(bits))
        if 1 <= ones <= 3 and ones < _h:
            opts.append(tuple(int(v) for v in bits))
    _SIGNATURES[_h] = opts


def _extract_lock(g: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    if not (3 <= h <= 5 and 5 <= w <= 7):
        return None
    if np.any((patch != BLACK) & (patch != GREY)):
        return None

    left_sig = tuple(int(v == BLACK) for v in patch[:, 0].tolist())
    right_sig = tuple(int(v == BLACK) for v in patch[:, -1].tolist())
    side = None
    sig = None
    if 1 <= sum(left_sig) <= 3 and sum(right_sig) == 0:
        side = "left"
        sig = left_sig
    if 1 <= sum(right_sig) <= 3 and sum(left_sig) == 0:
        if side is not None:
            return None
        side = "right"
        sig = right_sig
    if side is None or sig is None:
        return None
    return {"bbox": (r0, c0, r1, c1), "side": side, "sig": sig}


def _extract_key(g: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    colors = sorted({int(g[r, c]) for r, c in coords})
    if len(colors) != 1:
        return None
    color = int(colors[0])
    if color in (BLACK, GREY):
        return None

    h, w = patch.shape
    if not (3 <= h <= 5 and w == 2):
        return None
    if np.any((patch != BLACK) & (patch != color)):
        return None

    col0 = tuple(int(v == color) for v in patch[:, 0].tolist())
    col1 = tuple(int(v == color) for v in patch[:, 1].tolist())
    if all(col0) and 1 <= sum(col1) < h:
        return {
            "bbox": (r0, c0, r1, c1),
            "coords": coords,
            "patch": patch.copy(),
            "color": color,
            "side": "left",
            "sig": col1,
        }
    if all(col1) and 1 <= sum(col0) < h:
        return {
            "bbox": (r0, c0, r1, c1),
            "coords": coords,
            "patch": patch.copy(),
            "color": color,
            "side": "right",
            "sig": col0,
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    lock_comps = _components(g == GREY)
    locks = [_extract_lock(g, comp) for comp in lock_comps]
    locks = [lock for lock in locks if lock is not None]
    assert locks, "expected at least one lock"

    key_mask = (g != BLACK) & (g != GREY)
    key_comps = _components(key_mask)
    keys = [_extract_key(g, comp) for comp in key_comps]
    keys = [key for key in keys if key is not None]

    used_keys: set[int] = set()
    for lock in locks:
        matches: list[tuple[int, bool]] = []
        for idx, key in enumerate(keys):
            if idx in used_keys or key["side"] != lock["side"]:
                continue
            if len(key["sig"]) != len(lock["sig"]):
                continue
            if tuple(key["sig"]) == tuple(lock["sig"]):
                matches.append((idx, False))
            elif tuple(reversed(key["sig"])) == tuple(lock["sig"]):
                matches.append((idx, True))
        assert len(matches) == 1, "each lock should match exactly one key"

        key_idx, need_flip = matches[0]
        key = keys[key_idx]
        used_keys.add(int(key_idx))

        for r, c in key["coords"]:
            out[r, c] = BLACK

        r0, c0, r1, c1 = lock["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[sub == BLACK] = int(key["color"])

        moved = np.flipud(key["patch"]).copy() if need_flip else key["patch"].copy()
        if lock["side"] == "left":
            _paste(out, moved, int(r0), int(c0 - 1), transparent=BLACK)
        else:
            _paste(out, moved, int(r0), int(c1), transparent=BLACK)

    return out


def _make_key_patch(h: int, side: str, sig: tuple[int, ...], color: int) -> np.ndarray:
    patch = np.full((h, 2), BLACK, dtype=int)
    if side == "left":
        patch[:, 0] = int(color)
        for r, bit in enumerate(sig):
            if bit:
                patch[r, 1] = int(color)
    else:
        patch[:, 1] = int(color)
        for r, bit in enumerate(sig):
            if bit:
                patch[r, 0] = int(color)
    return patch


def _make_cavities(h: int, w: int, side: str, sig: tuple[int, ...]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if side == "left" and c == 1 and sig[r]:
                continue
            if side == "right" and c == w - 2 and sig[r]:
                continue
            candidates.append((r, c))
    count = random.randint(1, min(3, len(candidates)))
    return random.sample(candidates, count)


def _make_lock_patch(h: int, w: int, side: str, sig: tuple[int, ...], cavities: list[tuple[int, int]]) -> np.ndarray:
    patch = np.full((h, w), GREY, dtype=int)
    col = 0 if side == "left" else w - 1
    for r, bit in enumerate(sig):
        if bit:
            patch[r, col] = BLACK
    for r, c in cavities:
        patch[r, c] = BLACK
    return patch


def _make_output_patch(lock_patch: np.ndarray, key_patch: np.ndarray, side: str, color: int) -> np.ndarray:
    h, w = lock_patch.shape
    patch = np.full((h, w + 1), BLACK, dtype=int)
    if side == "left":
        patch[:, 1:] = lock_patch
        lock_sub = patch[:, 1:]
        lock_sub[lock_sub == BLACK] = int(color)
        _paste(patch, key_patch, 0, 0)
    else:
        patch[:, :w] = lock_patch
        lock_sub = patch[:, :w]
        lock_sub[lock_sub == BLACK] = int(color)
        _paste(patch, key_patch, 0, w - 1)
    return patch


def _can_place_box(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = BOX_PAD) -> bool:
    H, W = occupied.shape
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(H, r + h + pad)
    c1 = min(W, c + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _reserve_box(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    occupied[r : r + h, c : c + w] = True


def _pick_signature(h: int, forbidden: set[tuple[int, ...]], require_asym: bool) -> tuple[int, ...] | None:
    options = [
        sig
        for sig in _SIGNATURES[h]
        if _canonical_sig(sig) not in forbidden and (not require_asym or sig != tuple(reversed(sig)))
    ]
    if not options:
        return None
    return random.choice(options)


def _build_pair_specs(n_pairs: int, colors: list[int]) -> tuple[list[dict], dict[tuple[str, int], set[tuple[int, ...]]]] | None:
    pairs: list[dict] = []
    used: dict[tuple[str, int], set[tuple[int, ...]]] = {}

    for idx in range(n_pairs):
        made = False
        for _ in range(80):
            h = random.randint(3, 5)
            w = random.randint(5, 7)
            side = random.choice(("left", "right"))
            need_flip = bool(random.getrandbits(1))
            used_here = used.setdefault((side, h), set())
            sig = _pick_signature(h, used_here, require_asym=need_flip)
            if sig is None:
                continue
            used_here.add(_canonical_sig(sig))
            color = int(colors[idx])
            cavities = _make_cavities(h, w, side, sig)
            lock_patch = _make_lock_patch(h, w, side, sig, cavities)
            output_key_patch = _make_key_patch(h, side, sig, color)
            input_key_patch = np.flipud(output_key_patch).copy() if need_flip else output_key_patch.copy()
            output_patch = _make_output_patch(lock_patch, output_key_patch, side, color)
            pairs.append(
                {
                    "side": side,
                    "lock_patch": lock_patch,
                    "input_key_patch": input_key_patch,
                    "output_patch": output_patch,
                }
            )
            made = True
            break
        if not made:
            return None

    return pairs, used


def _build_unmatched_specs(
    n_unmatched: int,
    colors: list[int],
    offset: int,
    used: dict[tuple[str, int], set[tuple[int, ...]]],
) -> list[dict] | None:
    specs: list[dict] = []
    for idx in range(n_unmatched):
        color = int(colors[offset + idx])
        if random.random() < 0.35:
            h = random.randint(3, 5)
            specs.append({"patch": np.full((h, 1), color, dtype=int)})
            continue

        made = False
        for _ in range(80):
            h = random.randint(3, 5)
            side = random.choice(("left", "right"))
            forbidden = used.get((side, h), set())
            sig = _pick_signature(h, forbidden, require_asym=False)
            if sig is None:
                continue
            specs.append({"patch": _make_key_patch(h, side, sig, color)})
            made = True
            break
        if not made:
            return None
    return specs


def _place_items(items: list[tuple[str, int, int, int]], H: int, W: int) -> dict[tuple[str, int], tuple[int, int]] | None:
    occupied = np.zeros((H, W), dtype=bool)
    placements: dict[tuple[str, int], tuple[int, int]] = {}

    random.shuffle(items)
    items.sort(key=lambda item: item[2] * item[3], reverse=True)
    for kind, idx, h, w in items:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r = random.randint(0, H - h)
            c = random.randint(0, W - w)
            if not _can_place_box(occupied, r, c, h, w):
                continue
            _reserve_box(occupied, r, c, h, w)
            placements[(kind, idx)] = (r, c)
            placed = True
            break
        if not placed:
            return None
    return placements


def _attempt_generate() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    n_unmatched = random.randint(*UNMATCHED_COUNT_RANGE)
    colors = random.sample(KEY_COLORS, n_pairs + n_unmatched)

    pair_info = _build_pair_specs(n_pairs, colors)
    if pair_info is None:
        return None
    pairs, used = pair_info

    unmatched = _build_unmatched_specs(n_unmatched, colors, n_pairs, used)
    if unmatched is None:
        return None

    H = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    W = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)

    output_items = []
    for idx, pair in enumerate(pairs):
        h, w = pair["output_patch"].shape
        output_items.append(("pair", idx, h, w))
    for idx, spec in enumerate(unmatched):
        h, w = spec["patch"].shape
        output_items.append(("unmatched", idx, h, w))

    output_places = _place_items(output_items, H, W)
    if output_places is None:
        return None

    expected = np.full((H, W), BLACK, dtype=int)
    input_grid = np.full((H, W), BLACK, dtype=int)
    input_occupied = np.zeros((H, W), dtype=bool)

    for idx, pair in enumerate(pairs):
        out_r, out_c = output_places[("pair", idx)]
        _paste(expected, pair["output_patch"], out_r, out_c)

        lock_h, lock_w = pair["lock_patch"].shape
        lock_c = out_c + 1 if pair["side"] == "left" else out_c
        input_grid[out_r : out_r + lock_h, lock_c : lock_c + lock_w] = pair["lock_patch"]
        _reserve_box(input_occupied, out_r, lock_c, lock_h, lock_w)

    for idx, spec in enumerate(unmatched):
        r, c = output_places[("unmatched", idx)]
        patch = spec["patch"]
        h, w = patch.shape
        _paste(expected, patch, r, c)
        _paste(input_grid, patch, r, c)
        _reserve_box(input_occupied, r, c, h, w)

    key_items = []
    for idx, pair in enumerate(pairs):
        h, w = pair["input_key_patch"].shape
        key_items.append(("key", idx, h, w))

    random.shuffle(key_items)
    key_items.sort(key=lambda item: item[2] * item[3], reverse=True)
    for _, idx, h, w in key_items:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r = random.randint(0, H - h)
            c = random.randint(0, W - w)
            if not _can_place_box(input_occupied, r, c, h, w):
                continue
            _paste(input_grid, pairs[idx]["input_key_patch"], r, c)
            _reserve_box(input_occupied, r, c, h, w)
            placed = True
            break
        if not placed:
            return None

    try:
        actual = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
