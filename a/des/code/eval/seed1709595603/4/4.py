import numpy as np


BG = 0
RNG = np.random.default_rng()

# Generator configuration.
MAX_GENERATION_ATTEMPTS = 400
PAIR_COUNT_CHOICES = (2, 3)
PAIR_TYPES = ("lr", "tb")
LENGTH_VALUES = (4, 5, 6, 7, 8)
SIDE_SIZE_VALUES = (2, 3, 4)
CANVAS_MIN = 20
CANVAS_MAX = 30
ZONE_GAP = 1
PLACEMENT_GAP_VALUES = (1, 2, 3)
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for pr, pc in coords:
                mask[pr - r0, pc - c0] = True
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return components


def _lines_contiguous(mask: np.ndarray, axis: int) -> bool:
    if axis == 0:
        count = mask.shape[0]
        for idx in range(count):
            xs = np.flatnonzero(mask[idx, :])
            if xs.size == 0 or int(xs[-1] - xs[0] + 1) != int(xs.size):
                return False
        return True

    count = mask.shape[1]
    for idx in range(count):
        ys = np.flatnonzero(mask[:, idx])
        if ys.size == 0 or int(ys[-1] - ys[0] + 1) != int(ys.size):
            return False
    return True


def _is_alternating(bits: list[int]) -> bool:
    return len(bits) >= 2 and all(bits[i] != bits[i - 1] for i in range(1, len(bits)))


def _analyze_fragment(mask: np.ndarray) -> dict:
    h, w = mask.shape
    row_contig = _lines_contiguous(mask, axis=0)
    col_contig = _lines_contiguous(mask, axis=1)

    if row_contig == col_contig:
        raise AssertionError("expected exactly one contiguous slicing direction")

    if row_contig:
        starts: list[int] = []
        ends: list[int] = []
        for r in range(h):
            xs = np.flatnonzero(mask[r, :])
            starts.append(int(xs[0]))
            ends.append(int(xs[-1]))

        if len(set(starts)) == 1 and max(ends) - min(ends) == 1:
            limit = max(ends)
            side = "right"
            profile = [int(v == limit) for v in ends]
        elif len(set(ends)) == 1 and max(starts) - min(starts) == 1:
            limit = min(starts)
            side = "left"
            profile = [int(v == limit) for v in starts]
        else:
            raise AssertionError("invalid vertical jagged fragment")

        if not _is_alternating(profile):
            raise AssertionError("jagged profile must alternate")

        return {
            "side": side,
            "profile": tuple(profile),
            "length": h,
            "shape": (h, w),
        }

    starts = []
    ends = []
    for c in range(w):
        ys = np.flatnonzero(mask[:, c])
        starts.append(int(ys[0]))
        ends.append(int(ys[-1]))

    if len(set(starts)) == 1 and max(ends) - min(ends) == 1:
        limit = max(ends)
        side = "bottom"
        profile = [int(v == limit) for v in ends]
    elif len(set(ends)) == 1 and max(starts) - min(starts) == 1:
        limit = min(starts)
        side = "top"
        profile = [int(v == limit) for v in starts]
    else:
        raise AssertionError("invalid horizontal jagged fragment")

    if not _is_alternating(profile):
        raise AssertionError("jagged profile must alternate")

    return {
        "side": side,
        "profile": tuple(profile),
        "length": w,
        "shape": (h, w),
    }


def _opposite_side(side: str) -> str:
    return {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }[side]


def _profiles_complement(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return len(a) == len(b) and all(x + y == 1 for x, y in zip(a, b))


def _match_candidates(fragments: list[dict], idx: int, used: set[int]) -> list[int]:
    frag = fragments[idx]
    matches: list[int] = []
    for j, other in enumerate(fragments):
        if j == idx or j in used:
            continue
        if other["length"] != frag["length"]:
            continue
        if other["side"] != _opposite_side(frag["side"]):
            continue
        if _profiles_complement(frag["profile"], other["profile"]):
            matches.append(j)
    return matches


def _pair_fragments(fragments: list[dict]) -> list[tuple[int, int]]:
    def search(used: set[int]) -> list[tuple[int, int]] | None:
        if len(used) == len(fragments):
            return []

        remaining = [i for i in range(len(fragments)) if i not in used]
        pivot = min(remaining, key=lambda i: len(_match_candidates(fragments, i, used)))
        candidates = _match_candidates(fragments, pivot, used)
        if not candidates:
            return None

        for j in candidates:
            used2 = set(used)
            used2.add(pivot)
            used2.add(j)
            rest = search(used2)
            if rest is not None:
                return [(pivot, j)] + rest
        return None

    result = search(set())
    if result is None:
        raise AssertionError("could not find a perfect complementary pairing")
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary jagged monochrome fragments, merge each pair into a rectangle,
    and repaint the rectangle with alternating stripes from the pair's two colors.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    fragments = []
    for comp in _find_components(g):
        analysis = _analyze_fragment(comp["mask"])
        r0, c0, r1, c1 = comp["bbox"]
        fragments.append(
            {
                "bbox": comp["bbox"],
                "color": int(comp["color"]),
                "side": analysis["side"],
                "profile": analysis["profile"],
                "length": int(analysis["length"]),
                "shape": analysis["shape"],
                "top_left": (int(r0), int(c0)),
            }
        )

    if not fragments or len(fragments) % 2 != 0:
        raise AssertionError("expected an even non-zero number of fragments")

    pairs = _pair_fragments(fragments)
    out = np.zeros_like(g)

    for i, j in pairs:
        a = fragments[i]
        b = fragments[j]
        anchor_r, anchor_c = min(a["top_left"], b["top_left"])

        if {a["side"], b["side"]} == {"left", "right"}:
            left_piece = a if a["side"] == "right" else b
            right_piece = b if left_piece is a else a
            out_h = int(left_piece["shape"][0])
            out_w = int(left_piece["shape"][1] + right_piece["shape"][1] - 1)
            for r in range(out_h):
                color = left_piece["color"] if r % 2 == 0 else right_piece["color"]
                out[anchor_r + r, anchor_c : anchor_c + out_w] = color
            continue

        if {a["side"], b["side"]} != {"top", "bottom"}:
            raise AssertionError("pair must be left/right or top/bottom")

        top_piece = a if a["side"] == "bottom" else b
        bottom_piece = b if top_piece is a else a
        out_h = int(top_piece["shape"][0] + bottom_piece["shape"][0] - 1)
        out_w = int(top_piece["shape"][1])
        for c in range(out_w):
            color = top_piece["color"] if c % 2 == 0 else bottom_piece["color"]
            out[anchor_r : anchor_r + out_h, anchor_c + c] = color

    return out


def _make_lr_masks(length: int, left_w: int, right_w: int, start_bit: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((length, left_w), dtype=bool)
    right = np.zeros((length, right_w), dtype=bool)
    for r in range(length):
        bit = (start_bit + r) % 2
        left[r, : left_w - 1 + bit] = True
        right[r, 1 if bit else 0 :] = True
    return left, right


def _make_tb_masks(length: int, top_h: int, bottom_h: int, start_bit: int) -> tuple[np.ndarray, np.ndarray]:
    top = np.zeros((top_h, length), dtype=bool)
    bottom = np.zeros((bottom_h, length), dtype=bool)
    for c in range(length):
        bit = (start_bit + c) % 2
        top[: top_h - 1 + bit, c] = True
        bottom[1 if bit else 0 :, c] = True
    return top, bottom


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    region = grid[top : top + h, left : left + w]
    if np.any(region[mask] != BG):
        raise AssertionError("mask placement overlaps an existing fragment")
    region[mask] = color


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _build_expected(shape: tuple[int, int], pair_specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in pair_specs:
        r0, c0 = spec["anchor"]
        out_h, out_w = spec["out_shape"]
        a_color, b_color = spec["stripe_colors"]
        if spec["pair_type"] == "lr":
            for r in range(out_h):
                out[r0 + r, c0 : c0 + out_w] = a_color if r % 2 == 0 else b_color
        else:
            for c in range(out_w):
                out[r0 : r0 + out_h, c0 + c] = a_color if c % 2 == 0 else b_color
    return out


def _sample_pair_specs(pair_count: int) -> list[dict] | None:
    colors = list(RNG.choice(COLOR_POOL, size=2 * pair_count, replace=False))
    used_lengths: set[tuple[str, int]] = set()
    pair_specs: list[dict] = []

    for pair_idx in range(pair_count):
        pair_type = str(RNG.choice(PAIR_TYPES))
        available = [v for v in LENGTH_VALUES if (pair_type, v) not in used_lengths]
        if not available:
            return None
        length = int(RNG.choice(available))
        used_lengths.add((pair_type, length))
        start_bit = int(RNG.integers(0, 2))

        color_a = int(colors[2 * pair_idx])
        color_b = int(colors[2 * pair_idx + 1])
        anchor_piece = int(RNG.integers(0, 2))
        placement_mode = str(RNG.choice(("right", "below")))
        gap = int(RNG.choice(PLACEMENT_GAP_VALUES))

        if pair_type == "lr":
            left_w = int(RNG.choice(SIDE_SIZE_VALUES))
            right_w = int(RNG.choice(SIDE_SIZE_VALUES))
            left_mask, right_mask = _make_lr_masks(length, left_w, right_w, start_bit)
            pieces = [
                {"mask": left_mask, "color": color_a, "role": "left"},
                {"mask": right_mask, "color": color_b, "role": "right"},
            ]
            out_shape = (length, left_w + right_w - 1)
            stripe_colors = (color_a, color_b)
        else:
            top_h = int(RNG.choice(SIDE_SIZE_VALUES))
            bottom_h = int(RNG.choice(SIDE_SIZE_VALUES))
            top_mask, bottom_mask = _make_tb_masks(length, top_h, bottom_h, start_bit)
            pieces = [
                {"mask": top_mask, "color": color_a, "role": "top"},
                {"mask": bottom_mask, "color": color_b, "role": "bottom"},
            ]
            out_shape = (top_h + bottom_h - 1, length)
            stripe_colors = (color_a, color_b)

        first = pieces[anchor_piece]
        second = pieces[1 - anchor_piece]
        first_h, first_w = first["mask"].shape
        second_h, second_w = second["mask"].shape
        out_h, out_w = out_shape

        if placement_mode == "right":
            second_r = int(RNG.integers(0, max(1, out_h - second_h + 1)))
            second_c = out_w + gap
        else:
            second_r = out_h + gap
            second_c = int(RNG.integers(0, max(1, out_w - second_w + 1)))

        zone_h = max(out_h, first_h, second_r + second_h)
        zone_w = max(out_w, first_w, second_c + second_w)

        pair_specs.append(
            {
                "pair_type": pair_type,
                "pieces": pieces,
                "anchor_piece": anchor_piece,
                "second_offset": (second_r, second_c),
                "zone_shape": (zone_h, zone_w),
                "out_shape": out_shape,
                "stripe_colors": stripe_colors,
            }
        )

    return pair_specs


def _place_zones(pair_specs: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_zone_h = max(spec["zone_shape"][0] for spec in pair_specs)
    max_zone_w = max(spec["zone_shape"][1] for spec in pair_specs)
    min_h = max(CANVAS_MIN, max_zone_h + 2)
    min_w = max(CANVAS_MIN, max_zone_w + 2)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None

    for _ in range(60):
        h = int(RNG.integers(min_h, CANVAS_MAX + 1))
        w = int(RNG.integers(min_w, CANVAS_MAX + 1))
        placements: list[tuple[int, int] | None] = [None] * len(pair_specs)
        rects: list[tuple[int, int, int, int]] = []
        order = sorted(range(len(pair_specs)), key=lambda i: pair_specs[i]["zone_shape"][0] * pair_specs[i]["zone_shape"][1], reverse=True)

        ok = True
        for idx in order:
            zh, zw = pair_specs[idx]["zone_shape"]
            placed = False
            for _ in range(200):
                r0 = int(RNG.integers(0, h - zh + 1))
                c0 = int(RNG.integers(0, w - zw + 1))
                rect = (r0, c0, r0 + zh - 1, c0 + zw - 1)
                if any(_rects_overlap(rect, prev, gap=ZONE_GAP) for prev in rects):
                    continue
                placements[idx] = (r0, c0)
                rects.append(rect)
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return h, w, [p for p in placements if p is not None]

    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.choice(PAIR_COUNT_CHOICES))
    pair_specs = _sample_pair_specs(pair_count)
    if pair_specs is None:
        return None

    placed = _place_zones(pair_specs)
    if placed is None:
        return None

    h, w, zone_origins = placed
    grid = np.zeros((h, w), dtype=int)

    for spec, zone_origin in zip(pair_specs, zone_origins):
        zr, zc = zone_origin
        first = spec["pieces"][spec["anchor_piece"]]
        second = spec["pieces"][1 - spec["anchor_piece"]]
        second_r, second_c = spec["second_offset"]

        _paint_mask(grid, first["mask"], zr, zc, first["color"])
        _paint_mask(grid, second["mask"], zr + second_r, zc + second_c, second["color"])
        spec["anchor"] = (zr, zc)

    if len(_find_components(grid)) != 2 * pair_count:
        return None

    expected = _build_expected(grid.shape, pair_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jagged-fragment puzzle input")
