import numpy as np
from collections import Counter


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _cc4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mode_variation(values: tuple[int, ...]) -> int:
    counts = Counter(values)
    return len(values) - max(counts.values())


def _piece_side_and_profile(mask: np.ndarray) -> tuple[str, tuple[int, ...]]:
    h, w = mask.shape
    leftmost = tuple(int(np.flatnonzero(mask[r])[0]) for r in range(h))
    rightmost = tuple(int(np.flatnonzero(mask[r])[-1]) for r in range(h))
    topmost = tuple(int(np.flatnonzero(mask[:, c])[0]) for c in range(w))
    bottommost = tuple(int(np.flatnonzero(mask[:, c])[-1]) for c in range(w))

    candidates: list[tuple[int, int, str, tuple[int, ...]]] = []

    def add(side: str, arr: tuple[int, ...]) -> None:
        span = max(arr) - min(arr)
        score = _mode_variation(arr)
        norm = tuple(int(v - min(arr)) for v in arr)
        candidates.append((0 if span == 1 and score > 0 else 1, -score, side, norm))

    add("left", leftmost)
    add("right", tuple(int(max(rightmost) - v) for v in rightmost))
    add("top", topmost)
    add("bottom", tuple(int(max(bottommost) - v) for v in bottommost))

    candidates.sort()
    _, _, side, profile = candidates[0]
    return side, profile


def _extract_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    pieces: list[dict] = []
    for coords in _cc4(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        mask = np.zeros((h, w), dtype=bool)
        colors = set()
        for r, c in coords:
            mask[r - r0, c - c0] = True
            colors.add(int(grid[r, c]))
        color = int(next(iter(colors)))
        side, profile = _piece_side_and_profile(mask)
        pieces.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": mask,
                "color": color,
                "side": side,
                "profile": profile,
                "h": h,
                "w": w,
            }
        )
    return pieces


def _stamp(out: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    sub = out[r0 : r0 + h, c0 : c0 + w]
    sub[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary jagged fragments and slide each pair together along the shared normal.

    Input structure:
    - The background is black.
    - The grid contains an even number of monochrome 4-connected fragments, each in a unique color.
    - Every fragment is roughly rectangular: one side carries a 1-pixel jagged tab/notch pattern, while the partner fragment has the complementary pattern on the opposite facing side.
    - Matching pairs already face each other across empty black rows or columns, so only straight translation is needed.

    Rule:
    - For each fragment, identify which of its four sides is the jagged side by finding the only boundary profile that varies by a 1-pixel tab/notch pattern.
    - Match a `right` jagged side with a `left` jagged side, or a `bottom` jagged side with a `top` jagged side, when their 0/1 depth profiles are exact complements.
    - Translate each matched pair toward each other along that side's outward normal until the complementary teeth interlock tightly; clear the original locations.
    - Keep each fragment's original color in the final joined pair.

    Key insight:
    - Colors do not determine the pairing; the decisive signal is the complementary jagged edge profile and the axis along which the two fragments face each other.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _bg_color(g)
    pieces = _extract_pieces(g, bg)
    out = np.full_like(g, bg)

    used: set[int] = set()
    for i, piece in enumerate(pieces):
        if i in used or piece["side"] not in {"right", "bottom"}:
            continue

        profile = piece["profile"]
        if any(v not in (0, 1) for v in profile):
            continue
        partner_side = "left" if piece["side"] == "right" else "top"
        target_profile = tuple(1 - v for v in profile)

        candidates: list[int] = []
        for j, other in enumerate(pieces):
            if j == i or j in used or other["side"] != partner_side or other["profile"] != target_profile:
                continue
            if piece["side"] == "right":
                if piece["h"] != other["h"]:
                    continue
                if other["bbox"][1] <= piece["bbox"][1]:
                    continue
            else:
                if piece["w"] != other["w"]:
                    continue
                if other["bbox"][0] <= piece["bbox"][0]:
                    continue
            candidates.append(j)

        if not candidates:
            continue
        if piece["side"] == "right":
            j = min(candidates, key=lambda idx: (abs(pieces[idx]["bbox"][0] - piece["bbox"][0]), pieces[idx]["bbox"][1]))
            other = pieces[j]
            r0, c0, _, _ = piece["bbox"]
            or0, oc0, _, _ = other["bbox"]
            total = oc0 - c0 - (piece["w"] - 1)
            move_a = max(0, total // 2)
            move_b = max(0, total - move_a)
            _stamp(out, piece["shape"], piece["color"], r0, c0 + move_a)
            _stamp(out, other["shape"], other["color"], or0, oc0 - move_b)
        else:
            j = min(candidates, key=lambda idx: (abs(pieces[idx]["bbox"][1] - piece["bbox"][1]), pieces[idx]["bbox"][0]))
            other = pieces[j]
            r0, c0, _, _ = piece["bbox"]
            or0, oc0, _, _ = other["bbox"]
            total = or0 - r0 - (piece["h"] - 1)
            move_a = max(0, total // 2)
            move_b = max(0, total - move_a)
            _stamp(out, piece["shape"], piece["color"], r0 + move_a, c0)
            _stamp(out, other["shape"], other["color"], or0 - move_b, oc0)
        used.add(i)
        used.add(j)

    return out


def _rand_subset(rng: np.random.Generator, values: list[int]) -> tuple[int, ...]:
    count = int(rng.integers(2, len(values) + 1))
    pick = rng.choice(values, size=count, replace=False)
    return tuple(sorted(int(v) for v in pick))


def _make_horizontal_pair(rng: np.random.Generator, seen: set[tuple[str, tuple[int, ...]]]) -> dict | None:
    for _ in range(100):
        h = int(rng.integers(6, 9))
        left_base = int(rng.integers(3, 7))
        right_w = int(rng.integers(3, 7))
        rows = list(range(2, h - 2))
        if len(rows) < 2:
            continue
        tabs = _rand_subset(rng, rows)
        signature = ("h", tuple(1 if r in tabs else 0 for r in range(h)))
        if signature in seen:
            continue
        seen.add(signature)

        left = np.zeros((h, left_base + 1), dtype=bool)
        left[:, :left_base] = True
        left[list(tabs), left_base] = True

        right = np.ones((h, right_w), dtype=bool)
        right[list(tabs), 0] = False

        gap = int(rng.choice(np.array([1, 3, 5], dtype=int)))
        shift = (gap + 1) // 2
        return {
            "orientation": "h",
            "a_shape": left,
            "b_shape": right,
            "zone_h": h,
            "zone_w": left_base + right_w + 2 * shift,
            "shift": shift,
            "base": left_base,
        }
    return None


def _make_vertical_pair(rng: np.random.Generator, seen: set[tuple[str, tuple[int, ...]]]) -> dict | None:
    for _ in range(100):
        w = int(rng.integers(6, 9))
        top_base = int(rng.integers(3, 7))
        bottom_h = int(rng.integers(3, 7))
        cols = list(range(2, w - 2))
        if len(cols) < 2:
            continue
        tabs = _rand_subset(rng, cols)
        signature = ("v", tuple(1 if c in tabs else 0 for c in range(w)))
        if signature in seen:
            continue
        seen.add(signature)

        top = np.zeros((top_base + 1, w), dtype=bool)
        top[:top_base, :] = True
        top[top_base, list(tabs)] = True

        bottom = np.ones((bottom_h, w), dtype=bool)
        bottom[0, list(tabs)] = False

        gap = int(rng.choice(np.array([1, 3, 5], dtype=int)))
        shift = (gap + 1) // 2
        return {
            "orientation": "v",
            "a_shape": top,
            "b_shape": bottom,
            "zone_h": top_base + bottom_h + 2 * shift,
            "zone_w": w,
            "shift": shift,
            "base": top_base,
        }
    return None


def _fits(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h0, w0 = occupied.shape
    r1, c1 = r0 + h, c0 + w
    if r1 > h0 or c1 > w0:
        return False
    rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
    rr1, cc1 = min(h0, r1 + 1), min(w0, c1 + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_zones(rng: np.random.Generator, zones: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(zones)
    order = sorted(range(len(zones)), key=lambda idx: zones[idx][0] * zones[idx][1], reverse=True)
    for idx in order:
        zh, zw = zones[idx]
        choices = [(r, c) for r in range(h - zh + 1) for c in range(w - zw + 1) if _fits(occupied, r, c, zh, zw)]
        if not choices:
            return None
        pick = choices[int(rng.integers(0, len(choices)))]
        r0, c0 = pick
        occupied[r0 : r0 + zh, c0 : c0 + zw] = True
        placements[idx] = pick
    return [p for p in placements if p is not None]


def _build_pair_spec(rng: np.random.Generator, seen: set[tuple[str, tuple[int, ...]]]) -> dict | None:
    if bool(rng.integers(0, 2)):
        pair = _make_horizontal_pair(rng, seen)
        if pair is not None:
            return pair
        return _make_vertical_pair(rng, seen)
    pair = _make_vertical_pair(rng, seen)
    if pair is not None:
        return pair
    return _make_horizontal_pair(rng, seen)


def _render_from_specs(specs: list[dict], placements: list[tuple[int, int]], colors: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)

    color_idx = 0
    for spec, (zr, zc) in zip(specs, placements):
        color_a = int(colors[color_idx])
        color_b = int(colors[color_idx + 1])
        color_idx += 2

        a_shape = spec["a_shape"]
        b_shape = spec["b_shape"]
        shift = spec["shift"]
        base = spec["base"]

        if spec["orientation"] == "h":
            _stamp(inp, a_shape, color_a, zr, zc)
            _stamp(inp, b_shape, color_b, zr, zc + base + 2 * shift)
            _stamp(out, a_shape, color_a, zr, zc + shift)
            _stamp(out, b_shape, color_b, zr, zc + shift + base)
        else:
            _stamp(inp, a_shape, color_a, zr, zc)
            _stamp(inp, b_shape, color_b, zr + base + 2 * shift, zc)
            _stamp(out, a_shape, color_a, zr + shift, zc)
            _stamp(out, b_shape, color_b, zr + shift + base, zc)
    return inp, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_pairs = int(rng.integers(1, 4))
    seen: set[tuple[str, tuple[int, ...]]] = set()
    specs: list[dict] = []
    for _ in range(num_pairs):
        spec = _build_pair_spec(rng, seen)
        if spec is None:
            return None
        specs.append(spec)

    zones = [(spec["zone_h"], spec["zone_w"]) for spec in specs]
    min_h = max(max(zh for zh, _ in zones), 16)
    min_w = max(max(zw for _, zw in zones), 16)
    total_area = sum(zh * zw for zh, zw in zones)

    for _ in range(80):
        h_low = max(min_h, min(30, int(np.ceil(np.sqrt(total_area))) + 4))
        w_low = max(min_w, min(30, int(np.ceil(np.sqrt(total_area))) + 4))
        if h_low > 30 or w_low > 30:
            return None
        h = int(rng.integers(h_low, 31))
        w = int(rng.integers(w_low, 31))
        placements = _place_zones(rng, zones, h, w)
        if placements is None:
            continue
        colors = rng.permutation(PALETTE)[: 2 * num_pairs]
        inp, expected = _render_from_specs(specs, placements, colors, (h, w))
        if len(_cc4(inp != BG)) != 2 * num_pairs:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
